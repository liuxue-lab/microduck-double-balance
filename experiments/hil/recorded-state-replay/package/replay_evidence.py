"""Validate every exported tick and state before making replay frames."""
import csv
import math
import zipfile
from replay_common import need


def timing_rows(path, expected):
    with open(path, newline='') as stream:
        rows = list(csv.DictReader(stream))
    need(len(rows) == 3000, 'Timing must contain all 3000 formal cycles')
    parsed = []
    epoch = int(rows[0]['release_ns'])
    for index, row in enumerate(rows, 1):
        integer_keys = ('step', 'release_ns', 'started_ns', 'deadline_ns', 'finished_ns')
        item = {key: int(row[key]) for key in integer_keys}
        need(item['step'] == index and item['release_ns'] == epoch+(index-1)*20_000_000 and
             item['deadline_ns'] == item['release_ns']+20_000_000 and
             item['release_ns'] <= item['started_ns'] <= item['finished_ns'],
             'Timing release, deadline or order changed')
        for key, value in row.items():
            if (key.endswith('_ms') or key == 'simulation_time_s') and value not in ('', None):
                number = float(value)
                need(math.isfinite(number) and number >= 0, 'Nonfinite/negative timing: '+key)
                item[key] = number
        actual = (item['finished_ns']-item['release_ns'])/1e6
        need(abs(actual-item['release_to_finish_ms']) < 1e-9 and
             abs(item['simulation_time_s']-index*.02) < 1e-10 and
             abs(item['deadline_overrun_ms']-max(0., actual-20.)) < 1e-9,
             'CSV computed timing differs')
        parsed.append(item)
    durations = [row['release_to_finish_ms'] for row in parsed]
    ordered = sorted(durations)
    statistics = dict(count=len(rows), mean=sum(durations)/len(rows),
                      p95=ordered[math.ceil(.95*len(rows))-1],
                      p99=ordered[math.ceil(.99*len(rows))-1], max=ordered[-1],
                      over_20ms=sum(value > 20. for value in durations))
    wanted = expected['all_steps']['release_to_finish_ms']
    need(all(math.isclose(float(value), float(wanted[key]), rel_tol=1e-10, abs_tol=1e-9)
             for key, value in statistics.items()) and
         statistics['over_20ms'] == expected['deadline_misses'] == 0,
         'Timing CSV no longer matches accepted report')
    return parsed, statistics


def load_states(path, model):
    import numpy as np
    with zipfile.ZipFile(path) as archive:
        need(sum(item.file_size for item in archive.infolist()) <= 32*1024**2,
             'Exported state archive exceeds expected memory size')
    with np.load(path, allow_pickle=False) as stored:
        need(set(stored.files) == {'qpos', 'qvel', 'ctrl', 'control_step', 'simulation_time_s'},
             'Unexpected state schema')
        result = {name: np.array(stored[name], copy=True) for name in stored.files}
    for name, width in (('qpos', model.nq), ('qvel', model.nv), ('ctrl', model.nu)):
        value = result[name]
        need(value.shape == (3001, width) and value.dtype == np.float64 and np.isfinite(value).all(),
             'State dimensions, type or finiteness differ: '+name)
    need(result['control_step'].dtype == np.int64 and
         np.array_equal(result['control_step'], np.arange(3001)) and
         result['simulation_time_s'].dtype == np.float64 and
         np.array_equal(result['simulation_time_s'], np.arange(3001)*.02),
         'State time or step sequence differs')
    for value in result.values():
        value.setflags(write=False)
    return result


def frame_steps(states):
    """50 fps: use every formal sample 1..3000, with initial state as a separate PNG."""
    return range(1, len(states['control_step']))


def restore_visual_state(mujoco, model, data, states, index):
    import numpy as np
    for name in ('qpos', 'qvel', 'ctrl'):
        np.copyto(getattr(data, name), states[name][index])
    data.time = float(states['simulation_time_s'][index])
    # Position-only calculations for rendering; no dynamics, policy or integration.
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    mujoco.mj_camlight(model, data)
    need(all(np.array_equal(getattr(data, name), states[name][index])
             for name in ('qpos', 'qvel', 'ctrl')), 'Visual refresh altered a recorded state')


def scene_ids(model):
    def unique(items, title):
        need(len(items) == 1, 'Expected one '+title+'; found '+str(items))
        return items[0]
    trunk = unique([i for i in range(model.nbody)
                    if model.body(i).name.split('/')[-1] == 'trunk_base'], 'trunk_base')
    head = unique([i for i in range(model.nsite)
                   if model.site(i).name.split('/')[-1] == 'stage11_head_material_frame'], 'head reference site')
    def free_body(prefix):
        return unique([i for i in range(1, model.nbody)
                       if (model.body(i).name == prefix or model.body(i).name.startswith(prefix+'/'))
                       and model.body_jntnum[i] and model.jnt_type[model.body_jntadr[i]] == 0], prefix+' free body')
    return dict(trunk=trunk, head_site=head, lower=free_body('ball'), upper=free_body('top_ball'))
