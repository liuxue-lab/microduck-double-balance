"""Original certified vertex algorithm; only target-pair pose argument types become float64.
No geometric threshold, loop, arithmetic expression, or contact semantics changes.
"""
import warp as wp
wp.set_module_options({"enable_backward":False,"fast_math":False,"fuse_fp":False})

@wp.func
def relative_vertex(v:wp.vec3, r:wp.mat33d, p:wp.vec3d, s:wp.vec3d):
    result=wp.vec3d()
    for k in range(3):
        result[k]=(wp.float64(r[k,0])*wp.float64(v[0])+wp.float64(r[k,1])*wp.float64(v[1])+wp.float64(r[k,2])*wp.float64(v[2]))+(wp.float64(p[k])-wp.float64(s[k]))
    return result

@wp.kernel
def certified_geometry(vertices:wp.array[wp.vec3],positions:wp.array2d[wp.vec3d],rotations:wp.array2d[wp.mat33d],
                       vertex_start:int,vertex_count:int,sphere_id:int,mesh_id:int,radius:float,
                       result:wp.array2d[wp.float64]):
    w=wp.tid()
    r=rotations[w,mesh_id];p=positions[w,mesh_id];s=positions[w,sphere_id]
    best=wp.float64(1.0e300);best_id=int(-1);v_best=wp.vec3d()
    for j in range(vertex_count):
        v=relative_vertex(vertices[vertex_start+j],r,p,s)
        q=wp.dot(v,v)
        if q<best:
            best=q;best_id=j;v_best=v
    min_dot=wp.float64(1.0e300)
    for j in range(vertex_count):
        v=relative_vertex(vertices[vertex_start+j],r,p,s)
        min_dot=wp.min(min_dot,wp.dot(v-v_best,v_best))
    length=wp.sqrt(best)
    result[w,0]=wp.float64(best_id);result[w,1]=min_dot;result[w,2]=length
    result[w,3]=length-wp.float64(radius)
    valid=int(0)
    if best_id>=0 and min_dot>=wp.float64(0.0) and length>wp.float64(0.0):
        valid=1
        normal=v_best/length
        for k in range(3):
            result[w,4+k]=normal[k]
            result[w,7+k]=wp.float64(s[k])+(normal[k]*wp.float64(radius)+v_best[k])*wp.float64(0.5)
    result[w,10]=wp.float64(valid)

