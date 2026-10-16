"""Independent readback of every new accepted macro after owners have closed."""
from pathlib import Path
import sys
from integration.durable import require, write_json, file_digest


def verify(out):
    root=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(root.parent),str(root/'baseline/g11/accepted')]
    import numpy as np
    from ground_scalar_contract import assess
    from batch_position_candidate import assess_cohort
    from recovery.audit import macro_review
    cases=[]
    for i in (0,1):
        for k in range(30):
            rel='two_process_short_pilot/owner-%d/macro-%02d'%(i,k)
            checked=macro_review(Path(out)/rel,np,assess,assess_cohort)
            require(checked['native_acceptance'] and checked['passed'] and
                    len(checked['accepted_replays'])==1,'Independent macro readback rejected')
            cases.append(dict(macro=rel,result=checked))
    result=dict(status='ALL_NEW_MACROS_RECOMPUTED_EXACT',completed=60,expected=60,
                cases=cases,numpy=np.__version__,new_native_operations=0,PPO=0,cloud_training=0,
                assessor_sha256=file_digest(root/'baseline/g11/accepted/ground_scalar_contract.py'))
    write_json(Path(out)/'native-independent-review.json',result)
    return dict(status=result['status'],completed=60,sha256=file_digest(Path(out)/'native-independent-review.json'))
