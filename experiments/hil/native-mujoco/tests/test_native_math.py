import math
import random
import struct
import unittest
from support import old
import native_model_limits as native
import model_limits as original

class NativeMath(unittest.TestCase):
    def test_scalar_helpers_match_original_bit_patterns_and_exceptions(self):
        rng=random.Random(42)
        values=[0.,-0.,float('nan'),float('inf'),-float('inf'),1.,-1.,2.**-1022,1e300]
        values += [rng.uniform(-100.,100.) for _ in range(2000)]
        def outcome(fn,args):
            try:return ('value',struct.pack('!d',fn(*args)))
            except Exception as exc:return (type(exc).__name__,str(exc))
        for i,x in enumerate(values):
            cases=[(x,),(x,values[(i+1)%len(values)]),(x,values[(i+7)%len(values)],-x)]
            for args in cases:self.assertEqual(outcome(native.tolerance,args),outcome(original.tolerance,args))
            for bounds in ((None,None),(-1.,1.),(None,1.),(-1.,None)):
                self.assertEqual(outcome(native.excursion,(x,*bounds)),outcome(original.excursion,(x,*bounds)))

    def test_compiled_quaternions_match_original_for_rotations_and_ties(self):
        import torch
        from mjlab.utils.lab_api.math import matrix_from_quat,quat_from_matrix
        from static_math import quat_from_matrix as fast
        generator=torch.Generator().manual_seed(81)
        q=torch.randn(4096,4,generator=generator)
        q=q/torch.linalg.vector_norm(q,dim=-1,keepdim=True)
        q=torch.cat([q,torch.eye(4)],dim=0)
        matrices=matrix_from_quat(q)
        with torch.inference_mode():
            expected=quat_from_matrix(matrices)
            actual=fast(matrices)
        self.assertTrue(torch.equal(expected.contiguous().view(torch.int32),actual.contiguous().view(torch.int32)))
