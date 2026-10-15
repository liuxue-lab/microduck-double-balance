# Copyright 2025 The Newton Developers
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified for this bounded diagnostic: per-row cost output and serial scalar reductions.

"""Diagnostic fixed-order reductions; equations from pinned MuJoCo Warp solver.

Constraint cost equations derive from The Newton Developers' Apache-2.0 solver.
Only scalar accumulation order is an intervention; no force/Jacobian rewrite.
Scope: dense nv=32, pyramidal cone, no flex. Not a production solver replacement.
"""
import warp as wp
from mujoco_warp._src.math import safe_div
wp.set_module_options({'enable_backward':False})

@wp.kernel
def cost_rows(ne:wp.array(dtype=wp.int32),nf:wp.array(dtype=wp.int32),nefc:wp.array(dtype=wp.int32),
              D:wp.array2d(dtype=wp.float32),loss:wp.array2d(dtype=wp.float32),
              Jaref:wp.array2d(dtype=wp.float32),done:wp.array(dtype=wp.bool),rows:wp.array2d(dtype=wp.float32)):
    w,i=wp.tid()
    if i>=nefc[w] or done[w]:
        return
    value=wp.float32(0.0)
    d=D[w,i]
    j=Jaref[w,i]
    if i<ne[w]:
        value=0.5*d*j*j
    elif i<ne[w]+nf[w]:
        f=loss[w,i]
        rf=safe_div(f,d)
        if j<=-rf:
            value=-f*(0.5*rf+j)
        elif j>=rf:
            value=-f*(0.5*rf-j)
        else:
            value=0.5*d*j*j
    elif j<0.0:
        value=0.5*d*j*j
    rows[w,i]=value

@wp.kernel
def reduce_cost(nefc:wp.array(dtype=wp.int32),done:wp.array(dtype=wp.bool),
                rows:wp.array2d(dtype=wp.float32),gauss:wp.array(dtype=wp.float32),
                original:wp.array(dtype=wp.float32),reference:wp.array(dtype=wp.float32)):
    w=wp.tid()
    if done[w]:
        reference[w]=original[w]
        return
    total=wp.float32(0.0)
    for i in range(nefc[w]):
        total=total+rows[w,i]
    reference[w]=total+gauss[w]

@wp.kernel
def norm32(values:wp.array2d(dtype=wp.float32),out:wp.array(dtype=wp.float32)):
    w=wp.tid()
    total=wp.float32(0.0)
    for i in range(32):
        value=values[w,i]
        total=total+value*value
    out[w]=total
