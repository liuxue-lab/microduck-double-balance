"""Canonicalize the native changed-row list before its dense Hessian consumer.

Only the active integer list is permuted; equations, constraints and tolerances
are unchanged. Single world, at most 1024 unique row IDs, one device thread.
"""
import warp as wp
wp.set_module_options({'enable_backward':False})

@wp.kernel
def canonical_changes(count:wp.array(dtype=wp.int32),ids:wp.array2d(dtype=wp.int32)):
    w=wp.tid()
    for i in range(1,count[w]):
        value=ids[w,i]
        j=i-1
        while j>=0:
            if ids[w,j]<=value:
                break
            ids[w,j+1]=ids[w,j]
            j=j-1
        ids[w,j+1]=value
