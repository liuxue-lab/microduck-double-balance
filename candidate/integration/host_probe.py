"""One aggregate CPU integration probe. Reuses v7 config acceptance; no rerun."""
from pathlib import Path
import argparse
import io
import json
import os
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from integration.durable import require, write_json
from stage12.contracts import REPO, EXECUTION_BUDGET


def install_boundary(out):
    def hook(event, args):
        if event in ("socket.connect", "socket.connect_ex", "socket.getaddrinfo", "subprocess.Popen", "os.system"):
            raise RuntimeError("CPU integration probe forbids network/process: " + event)
        if event == "open":
            path, mode, flags = args
            writing = isinstance(mode, str) and any(c in mode for c in "wax+")
            writing = writing or isinstance(flags, int) and bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND))
            if writing and not isinstance(path, int):
                from stage12.git_import import is_null_device
                require(is_null_device(path) or Path(path).resolve().is_relative_to(out.resolve()),
                        "Write outside new integration run refused: " + str(path))
    sys.addaudithook(hook)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    out = args.output
    require(out.is_dir(), "Launcher output missing")
    install_boundary(out)
    from stage12.host_sources import verify
    from integration.planning import proposal
    report = dict(status="STARTED", budget=EXECUTION_BUDGET, training_ready=False,
                  stage12_readiness_v7="REUSED_PASS_NO_RERUN", cases=[], failures=[], guard_attempts=[])
    before = None
    try:
        before = verify(ROOT, out, "before")
        prior = json.loads((ROOT / "prior-readiness-v7/configuration-probe.json").read_bytes())
        require(prior["status"] == "CPU_CONFIGURATION_AND_ADAPTER_PASS", "v7 acceptance missing")
        report["source_reuse"] = before
        write_json(out / "measurement-budget-proposal.json", proposal())
        # Run independent pure contracts as a group, collecting all failures.
        from validation.integration.test_contracts import run_suite
        buf = io.StringIO()
        result = run_suite(buf)
        with (out / "integration-contracts.log").open("x") as f:
            f.write(buf.getvalue())
        report["pure_contracts"] = dict(tests=result.testsRun, errors=len(result.errors),
                                       failures=len(result.failures), passed=result.wasSuccessful())
        if not result.wasSuccessful():
            report["failures"].append("pure_contracts")
        # Only Torch and TensorDict are needed. Avoid re-importing mjlab/GLFW/
        # Matplotlib/GitPython just to retest already-completed v7 configuration.
        try:
            print("Stage12IntegrationPhase=real-CPU-tensor-transition-contracts", flush=True)
            import torch
            require(not torch.cuda.is_initialized(), "Unexpected CUDA initialization")
            def deny(name):
                def call(*a, **k):
                    report["guard_attempts"].append(name)
                    raise RuntimeError("ZERO_BUDGET_BLOCKED: " + name)
                return call
            for owner, name in ((torch, "load"), (torch, "save"), (torch.jit, "load"), (torch.cuda, "_lazy_init")):
                setattr(owner, name, deny(name))
            from tensordict import TensorDict
            from validation.integration.cpu_cases import all_cases
            report["cases"] = all_cases(torch, TensorDict)
            report["tensor_backend"] = "actual installed Torch/TensorDict CPU; fake environments and fake algorithm"
            report["failures"] += [r["name"] for r in report["cases"] if not r["passed"]]
            require(not torch.cuda.is_initialized() and not report["guard_attempts"], "Forbidden operation attempted")
        except Exception as exc:
            report["failures"].append(dict(phase="CPU tensor contracts", error=repr(exc), traceback=traceback.format_exc()))
        # Independent packaging continues even if a contract failed; no rerun
        # is needed to collect the portable reviewed assets and all diagnostics.
        try:
            print("Stage12IntegrationPhase=source-and-asset-bundle", flush=True)
            from integration.source_bundle import prepare
            report["source_bundle"] = prepare(ROOT, out, REPO)
        except Exception as exc:
            report["failures"].append(dict(phase="source bundle", error=repr(exc), traceback=traceback.format_exc()))
    except BaseException as exc:
        report["failures"].append(dict(phase="identity or setup", error=repr(exc), traceback=traceback.format_exc()))
    finally:
        if before is not None:
            try:
                require(verify(ROOT, out, "after") == before, "Source identities changed")
                report["preservation"] = "PASS"
            except BaseException as exc:
                report["failures"].append(dict(phase="preservation", error=repr(exc)))
        report["status"] = "HOST_INTEGRATION_CPU_PASS_PHYSICAL_PENDING" if not report["failures"] else "HOST_INTEGRATION_BLOCKED"
        write_json(out / "host-report.json", report)
    print("Stage12HostIntegration=" + report["status"], flush=True)
    return int(bool(report["failures"]))


if __name__ == "__main__":
    raise SystemExit(main())
