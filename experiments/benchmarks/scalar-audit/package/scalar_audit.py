"""Keep original Audit bytecode; specialize its two scalar helper lookups.

Only this Audit instance receives cloned functions with private globals. No
module, class, policy, threshold, check ordering, or counter is modified.
The first 40 before/after calls also run an independent original CPU Audit on
the same already-read host data and compare its complete state bit for bit.
"""
import copy
import math
from pathlib import Path
import struct
from types import FunctionType, MethodType

import model_limits as original
from deploy_contract import require, sha
from phase_cache import Bindings

SOURCE_SHA='d318cdb3b14b078797e69dc89da15f4106f79779fbb764eb59d137708cfdd275'
METHODS=('observe_joints','before_integration','after_integration')
CHECK_CALLS=40
COEFFICIENT=32*(2.0**-23)


def tolerance(*values):
    # Same finite-value filtering, starting scale, absolute values, and final
    # multiplication as the original max/generator expression.
    scale=1.;seen=False
    for x in values:
        if math.isfinite(x):
            seen=True
            value=abs(x)
            if value>scale:scale=value
    # Original max(1., *[]) raises TypeError for no finite operands. Keep this
    # cold-path failure instead of silently accepting an invalid tolerance.
    if not seen:return original.tolerance(*values)
    return COEFFICIENT*scale


def excursion(value,lower,upper):
    # Preserve max(0., lower-value, value-upper) ordering, ties, NaN
    # comparisons and positive-zero result. None means an absent bound.
    low=0. if lower is None else lower-value
    high=0. if upper is None else value-upper
    result=low if low>0. else 0.
    return high if high>result else result


def exact(left,right):
    if type(left) is not type(right):return False
    if isinstance(left,float):return struct.pack('!d',left)==struct.pack('!d',right)
    if isinstance(left,dict):
        return list(left)==list(right) and all(exact(left[k],right[k]) for k in left)
    if isinstance(left,(list,tuple)):
        return len(left)==len(right) and all(exact(a,b) for a,b in zip(left,right))
    require(left is None or type(left) in (bool,int,str),'Unexpected Audit state type')
    return left==right


def state(audit):
    return {k:v for k,v in vars(audit).items() if k not in METHODS}


class ScalarAudit(Bindings):
    def __init__(self,audit):
        super().__init__();self.audit=audit;self.closed=False
        self.calls={k:0 for k in METHODS[1:]};self.checked={k:0 for k in METHODS[1:]}
        self.failure=None
        path=Path(original.__file__)
        require(sha(path)==SOURCE_SHA,'Unreviewed original Audit source')
        require(type(audit) is original.Audit,'Expected original Audit instance')
        require(original.FLOAT32_EPS==2.0**-23 and original.PHYSICS_DT==.002,
                'Original audit constants differ')
        self.source=dict(path=str(path),sha256=SOURCE_SHA)
        for name in METHODS:
            require(name not in vars(audit) and
                getattr(audit,name).__func__ is getattr(original.Audit,name),
                'Audit method already replaced: '+name)
        self.shadow=copy.deepcopy(audit)
        private=dict(vars(original),tolerance=tolerance,excursion=excursion)
        try:
            for name in METHODS:
                source=getattr(original.Audit,name)
                function=FunctionType(source.__code__,private,name,source.__defaults__,source.__closure__)
                function.__kwdefaults__=source.__kwdefaults__
                self.set(audit,name,MethodType(function,audit))
            for name in METHODS[1:]:
                method=getattr(audit,name)
                def checked(*args,_name=name,_method=method,**kwargs):
                    return self.run(_name,_method,args,kwargs)
                self.set(audit,name,checked)
        except BaseException:self.close();raise

    def run(self,name,method,args,kwargs):
        self.calls[name]+=1
        if self.checked[name]>=CHECK_CALLS:return method(*args,**kwargs)
        result=None;failure=None
        try:result=method(*args,**kwargs)
        except (Exception,original.LimitStop) as exc:failure=exc
        reference=None;reference_failure=None
        try:reference=getattr(self.shadow,name)(*args,**kwargs)
        except (Exception,original.LimitStop) as exc:reference_failure=exc
        same_error=(type(failure) is type(reference_failure)
            and str(failure)==str(reference_failure)
            and exact(getattr(failure,'event',None),getattr(reference_failure,'event',None)))
        if not (same_error and exact(result,reference) and exact(state(self.audit),state(self.shadow))):
            self.failure=dict(method=name,call=self.calls[name],reason='same host input or complete Audit state differs')
            raise RuntimeError('Scalar Audit shadow comparison failed: '+name)
        self.checked[name]+=1
        if failure is not None:raise failure
        return result

    def close(self):
        super().close();self.closed=True

    def stats(self):
        return dict(source=self.source,calls=dict(self.calls),same_input_checked_calls=dict(self.checked),
            last_comparison_failure=self.failure,closed=self.closed,bindings_restored=not self.saved,
            original_Audit_method_bytecode_retained=True,private_helper_globals=True,
            original_thresholds_and_event_order=True,comparison='complete original Audit state; exact float64 bit patterns',
            extra_physics_steps=0,extra_GPU_reads=0,physics_hz=500,no_check_decimation=True)
