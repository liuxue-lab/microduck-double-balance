"""Read-only graph inspection plus an isolated MLP-only dynamic INT8 candidate."""
import argparse
from collections import Counter
import copy
import json
from pathlib import Path
import traceback

from deploy_contract import OnnxPolicy, comparison, require, sha
from loop_contract import frozen


def lower_dense(model, onnx, np):
    result = copy.deepcopy(model)
    constants = {x.name: x for x in result.graph.initializer}
    occupied = {n for node in result.graph.node for n in (*node.input, *node.output)} | set(constants)
    nodes, converted = [], []
    def fresh(name):
        require(name not in occupied, 'Candidate intermediate name collision')
        occupied.add(name)
        return name
    for index, node in enumerate(result.graph.node):
        if node.op_type != 'Gemm':
            nodes.append(node)
            continue
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        require(attrs.get('alpha', 1.) == attrs.get('beta', 1.) == 1.
                and attrs.get('transA', 0) == 0 and attrs.get('transB', 0) in (0, 1)
                and len(node.input) == 3 and node.input[1] in constants and node.input[2] in constants,
                'Unsupported Gemm; candidate preparation stops without changing original model')
        weight = onnx.numpy_helper.to_array(constants[node.input[1]])
        bias = onnx.numpy_helper.to_array(constants[node.input[2]])
        require(weight.ndim == 2 and weight.dtype == np.float32 and bias.ndim == 1,
                'Expected original FP32 linear layer')
        if attrs.get('transB', 0):
            weight = weight.T
        require(bias.size == weight.shape[1], 'Linear bias dimension differs')
        prefix = 'md12_candidate_%d' % index
        weight_name, intermediate = fresh(prefix + '_weight'), fresh(prefix + '_matmul')
        result.graph.initializer.append(onnx.numpy_helper.from_array(np.ascontiguousarray(weight), weight_name))
        nodes.append(onnx.helper.make_node('MatMul', [node.input[0], weight_name], [intermediate], name=prefix + '_MatMul'))
        nodes.append(onnx.helper.make_node('Add', [intermediate, node.input[2]], list(node.output), name=prefix + '_Add'))
        converted.append(node.name)
    del result.graph.node[:]
    result.graph.node.extend(nodes)
    # Remove only unused initializers from this derived candidate.
    used = {n for node in result.graph.node for n in node.input}
    kept = [x for x in result.graph.initializer if x.name in used]
    del result.graph.initializer[:]
    result.graph.initializer.extend(kept)
    return result, converted


def inspect_dense(model, onnx, np):
    constants = {x.name: x for x in model.graph.initializer}
    layers = []
    for node in model.graph.node:
        if node.op_type not in ('MatMul', 'Gemm') or len(node.input) < 2 or node.input[1] not in constants:
            continue
        weight = onnx.numpy_helper.to_array(constants[node.input[1]])
        if weight.ndim != 2:
            continue
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        axis = 1 if node.op_type == 'Gemm' and attrs.get('transB', 0) else 0
        matrix = np.moveaxis(weight, axis, -1)
        eligible = matrix.shape[-1] % 4 == 0 and bool((np.count_nonzero(matrix.reshape(-1, 4), axis=1) <= 2).all())
        layers.append(dict(node=node.name, shape=list(weight.shape), weights=int(weight.size),
                           exact_zero_fraction=float(np.count_nonzero(weight == 0) / weight.size),
                           reduction_axis=axis, existing_2_of_4_sparsity=eligible))
    return layers


def prepare(folder, export):
    folder, export = Path(folder), Path(export)
    folder.mkdir(exist_ok=True)
    frozen(export)
    original_sha = sha(export / 'final6h_005304.onnx')
    result = dict(status='QUANTIZATION_CANDIDATE_UNAVAILABLE', original_onnx_sha256=original_sha,
        pruning_applied=False, fine_tuning=False, added_PPO_updates=0, deployment_candidate_activated=False,
        methods_not_applied={'pruning': 'No silent weight deletion or fine-tuning; sparse-kernel benefit requires separate validation',
                             'TensorRT_FP16': 'No provider/graph compatibility or speed evidence for this small recurrent policy yet'})
    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
        from onnxruntime.quantization import quantize_dynamic, QuantType
        model = onnx.load(export / 'final6h_005304.onnx', load_external_data=False)
        require(not any(x.data_location == onnx.TensorProto.EXTERNAL for x in model.graph.initializer),
                'Expected self-contained accepted ONNX')
        result.update(versions=dict(onnx=onnx.__version__, ort=ort.__version__, numpy=np.__version__),
                      original_ops=dict(Counter(n.op_type for n in model.graph.node)),
                      dense_weight_sparsity=inspect_dense(model, onnx, np))
        original_lstm = [n.SerializeToString() for n in model.graph.node if n.op_type == 'LSTM']
        require(original_lstm, 'Do not quantize an unrecognized recurrent graph')
        lowered, converted = lower_dense(model, onnx, np)
        onnx.checker.check_model(lowered)
        lowered_path = folder / 'dense-lowered-fp32.onnx'
        require(not lowered_path.exists(), 'Candidate file already exists')
        onnx.save(lowered, lowered_path)
        reference = OnnxPolicy(lowered_path)
        golden = json.loads((export / 'golden.json').read_bytes())
        error = 0.
        for case in golden['cases']:
            actions = reference.step(case['obs'], reset=case['reset'])
            error = max(error, comparison(actions, case['actions']),
                        comparison(reference.h.reshape(-1).tolist(), case['h']),
                        comparison(reference.c.reshape(-1).tolist(), case['c']))
        require(len(golden['cases']) == 128, 'Golden fixture differs')
        candidate = folder / 'mlp-dynamic-int8.onnx'
        require(not candidate.exists(), 'Candidate file already exists')
        constants = {x.name for x in lowered.graph.initializer}
        producers = {output: n for n in lowered.graph.node for output in n.output}
        def ancestors(value, seen=None):
            seen = set() if seen is None else seen
            if value in seen or value not in producers:
                return seen
            seen.add(value)
            for source in producers[value].input:
                ancestors(source, seen)
            return seen
        action_path = ancestors('actions')
        state_path = ancestors('h_out') | ancestors('c_out')
        recurrent_outputs = {v for n in lowered.graph.node if n.op_type == 'LSTM' for v in n.output}
        targets = [n for n in lowered.graph.node if n.op_type == 'MatMul' and n.input[1] in constants]
        require(all(any(o in action_path for o in n.output) and not any(o in state_path for o in n.output)
                    and bool(ancestors(n.input[0]) & recurrent_outputs) for n in targets),
                'Quantization is restricted to the action head downstream of LSTM')
        names = [n.name for n in targets]
        require(names and all(names), 'No named constant-weight MLP MatMul nodes available')
        quantize_dynamic(str(lowered_path), str(candidate), op_types_to_quantize=['MatMul'],
                         nodes_to_quantize=names, weight_type=QuantType.QInt8, per_channel=True,
                         extra_options={'MatMulConstBOnly': True})
        quantized = onnx.load(candidate, load_external_data=False)
        onnx.checker.check_model(quantized)
        require([n.SerializeToString() for n in quantized.graph.node if n.op_type == 'LSTM'] == original_lstm,
                'LSTM node must remain unchanged')
        qconst = {x.name: x for x in quantized.graph.initializer}
        original_const = {x.name: x for x in model.graph.initializer}
        for node in model.graph.node:
            if node.op_type == 'LSTM':
                for name in node.input:
                    if name in original_const:
                        require(name in qconst, 'LSTM initializer missing')
                        before = onnx.numpy_helper.to_array(original_const[name])
                        after = onnx.numpy_helper.to_array(qconst[name])
                        require(before.dtype == after.dtype and before.shape == after.shape
                                and before.tobytes() == after.tobytes(), 'LSTM weight values changed')
                        # ORT may merely add data_location=DEFAULT. Restore the
                        # exact original TensorProto after proving bit equality.
                        qconst[name].CopyFrom(original_const[name])
                        require(qconst[name].SerializeToString() == original_const[name].SerializeToString(),
                                'LSTM initializer representation differs')
        onnx.checker.check_model(quantized)
        onnx.save(quantized, candidate)
        require(any(n.op_type == 'MatMulInteger' for n in quantized.graph.node), 'No integer matrix kernels were produced')
        result.update(status='QUANTIZATION_CANDIDATE_PREPARED', path=str(candidate), candidate_sha256=sha(candidate),
            candidate_bytes=candidate.stat().st_size, original_bytes=(export / 'final6h_005304.onnx').stat().st_size,
            lowered_gemm_nodes=converted, lowered_fp32_golden_max_error=error,
            quantized_nodes=names, quantized_ops=dict(Counter(n.op_type for n in quantized.graph.node)),
            recurrent_weights_unchanged=True, action_head_only=True,
            candidate_requires_Orin_speed_and_error_measurement=True,
            candidate_requires_new_closed_loop_validation_before_deployment=True)
    except Exception as exc:
        result.update(error=repr(exc), traceback=traceback.format_exc()[-8000:])
    finally:
        require(sha(export / 'final6h_005304.onnx') == original_sha, 'Original model changed')
        frozen(export)
        with (folder / 'candidate-report.json').open('x') as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
    print('CANDIDATE_STATUS=' + result['status'], flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder', required=True)
    parser.add_argument('--export', required=True)
    args = parser.parse_args()
    prepare(args.folder, args.export)
