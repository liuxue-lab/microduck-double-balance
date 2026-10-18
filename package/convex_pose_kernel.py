"""Full certified convex hull, with the byte-equivalent v2 vertex fast path.

No separation epsilon, CCD tolerance change, smoothing, or original GJK fallback.
All faces/edges are searched if the exact v2 vertex predicate does not hold.
"""
import warp as wp
from precise_pose_kernel import relative_vertex

wp.set_module_options({'enable_backward':False,'fast_math':False,'fuse_fp':False})


@wp.kernel
def convex_geometry(vertices:wp.array[wp.vec3],facets:wp.array[wp.vec3i],
                    positions:wp.array2d[wp.vec3d],rotations:wp.array2d[wp.mat33d],
                    vertex_start:int,vertex_count:int,facet_count:int,radius:float,
                    result:wp.array2d[wp.float64]):
    w=wp.tid();r=rotations[w,1];p=positions[w,1];s=positions[w,0]
    best=wp.float64(1.0e300);best_id=int(-1);v_best=wp.vec3d()
    for j in range(vertex_count):
        v=relative_vertex(vertices[vertex_start+j],r,p,s);squared=wp.dot(v,v)
        if squared<best:best=squared;best_id=j;v_best=v
    min_dot=wp.float64(1.0e300)
    for j in range(vertex_count):
        v=relative_vertex(vertices[vertex_start+j],r,p,s)
        min_dot=wp.min(min_dot,wp.dot(v-v_best,v_best))
    length=wp.sqrt(best)
    valid=int(0);best_face=int(-1);feature=int(0);inside=int(0)
    bary=wp.vec2d();sign=wp.float64(1.0)
    if not (best_id>=0 and min_dot>=wp.float64(0.0) and length>wp.float64(0.0)):
        best=wp.float64(1.0e300);inside=int(1);nondegenerate=int(1)
        for j in range(facet_count):
            ids=facets[j]
            a=relative_vertex(vertices[vertex_start+ids[0]],r,p,s)
            b=relative_vertex(vertices[vertex_start+ids[1]],r,p,s)
            c=relative_vertex(vertices[vertex_start+ids[2]],r,p,s)
            ab=b-a;ac=c-a;n=wp.cross(ab,ac);den=wp.dot(n,n)
            if wp.dot(n,-a)>wp.float64(0.0):inside=0
            if den<=wp.float64(0.0):nondegenerate=0
            else:
                u=wp.dot(wp.cross(-a,ac),n)/den
                beta=wp.dot(wp.cross(ab,-a),n)/den
                if u>=wp.float64(0.0) and beta>=wp.float64(0.0) and u+beta<=wp.float64(1.0):
                    q=a+u*ab+beta*ac;d2=wp.dot(q,q)
                    if d2<best:best=d2;v_best=q;best_face=j;feature=2;bary=wp.vec2d(u,beta)
            for edge in range(3):
                start=a;end=b
                if edge==1:start=b;end=c
                elif edge==2:start=c;end=a
                delta=end-start;d2=wp.dot(delta,delta)
                if d2<=wp.float64(0.0):nondegenerate=0
                else:
                    t=wp.clamp(-wp.dot(start,delta)/d2,wp.float64(0.0),wp.float64(1.0))
                    q=start+t*delta;distance=wp.dot(q,q)
                    if distance<best:
                        best=distance;v_best=q;best_face=j;feature=1
                        if t==wp.float64(0.0) or t==wp.float64(1.0):feature=0
                        bary=wp.vec2d(t,wp.float64(0.0))
                        if edge==1:bary=wp.vec2d(wp.float64(1.0)-t,t)
                        elif edge==2:bary=wp.vec2d(wp.float64(0.0),wp.float64(1.0)-t)
        best_id=-1;length=wp.sqrt(best)
        min_dot=wp.float64(1.0e300)
        for j in range(vertex_count):
            v=relative_vertex(vertices[vertex_start+j],r,p,s)
            min_dot=wp.min(min_dot,wp.dot(v-v_best,v_best))
        if inside==1:sign=wp.float64(-1.0)
        if nondegenerate==1 and best_face>=0 and length>wp.float64(0.0):valid=1
    else:valid=1
    result[w,0]=wp.float64(best_id);result[w,1]=min_dot;result[w,2]=length
    result[w,3]=sign*length-wp.float64(radius)
    if valid==1:
        normal=sign*v_best/length
        for k in range(3):
            result[w,4+k]=normal[k]
            result[w,7+k]=wp.float64(s[k])+(normal[k]*wp.float64(radius)+v_best[k])*wp.float64(.5)
    result[w,10]=wp.float64(valid);result[w,11]=wp.float64(best_face)
    result[w,12]=wp.float64(feature);result[w,13]=wp.float64(inside)
    result[w,14]=bary[0];result[w,15]=bary[1]
