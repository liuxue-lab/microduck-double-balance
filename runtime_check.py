"""One cloud runtime verification, zero environment construction/integration/update."""
from pathlib import Path
import argparse
import importlib.metadata as meta
import os
import sys

ROOT=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path[:0]=[str(ROOT),str(ROOT/'candidate')]
from common import write, require


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args()
    require(sys.version_info[:3]==(3,12,14),'Python must remain 3.12.14')
    from integration.child_bootstrap import owner_bootstrap
    output=a.out/'zero-physics-owner'
    with owner_bootstrap(ROOT/'candidate',output) as report:
        sys.path[:0]=[str(ROOT/'candidate/baseline/g11'),str(a.repo/'src')]
        from worker_g11 import verify_sources
        before=verify_sources(a.repo)
        from integration.manager_contracts import preflight
        import torch
        require(torch.cuda.is_available() and torch.cuda.device_count()==1,'One CUDA GPU required')
        require('5090' in torch.cuda.get_device_name(0),'Expected RTX5090')
        require(torch.__version__.split('+')[0]=='2.9.1','Torch version differs')
        from formal_task import make_cfg
        cfg=make_cfg()
        require(cfg.auto_reset is False,'Manual terminal bridge required')
        require(before==verify_sources(a.repo),'Installed source changed')
        report.update(source_pins=before,python=sys.version,torch=torch.__version__,torch_cuda=torch.version.cuda,
                      device=torch.cuda.get_device_name(0),native_integrations=0,native_forward=0,
                      environment_constructions=0,PPO=0,cloud_training=0,torch_load=0)
        # Use the already allowed bootstrap directory for this final receipt.
        write(output.with_name(output.name+'-bootstrap')/'runtime-verified.json',report)
    return 0


if __name__=='__main__':
    raise SystemExit(main())
