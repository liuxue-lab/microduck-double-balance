"""Device-only word permutation primitive; no physics, allocations or host reads."""
import warp as wp

@wp.kernel
def rank_rows(count:wp.array(dtype=wp.int32),world:wp.array(dtype=wp.uint32),
              geom:wp.array(dtype=wp.uint32),collision:wp.array(dtype=wp.uint32),
              order:wp.array(dtype=wp.int32),status:wp.array(dtype=wp.int32)):
    n=count[0]
    status[0]=0
    for i in range(256):
        order[i]=i
    if n<0 or n>256:
        status[0]=1
        return
    for i in range(n):
        w=wp.int32(world[i]);g0=wp.int32(geom[2*i]);g1=wp.int32(geom[2*i+1]);c=wp.int32(collision[i])
        if w<0 or w>0 or g0<0 or g0>=85 or g1<0 or g1>=85 or g0==g1:
            status[0]=status[0] | 16
        rank=int(0)
        for j in range(n):
            x=wp.int32(world[j]);a=wp.int32(geom[2*j]);b=wp.int32(geom[2*j+1]);d=wp.int32(collision[j])
            if j!=i and x==w and a==g0 and b==g1 and d==c:
                status[0]=status[0] | 2
            if x<w or (x==w and a<g0) or (x==w and a==g0 and b<g1) or (x==w and a==g0 and b==g1 and d<c):
                rank=rank+1
        order[rank]=i

@wp.kernel
def validate_words(src:wp.array(dtype=wp.uint32),count:wp.array(dtype=wp.int32),
                   status:wp.array(dtype=wp.int32),row_words:int,kind:int):
    n=count[0]
    if n<0 or n>256:
        return
    for i in range(n*row_words):
        v=src[i]
        if kind==1 and (v & wp.uint32(2139095040))==wp.uint32(2139095040):
            status[0]=status[0] | 4
        if kind==2 and v!=wp.uint32(4294967295):
            status[0]=status[0] | 8
        if kind==3 and v!=wp.uint32(1) and v!=wp.uint32(3) and v!=wp.uint32(4) and v!=wp.uint32(6):
            status[0]=status[0] | 32
        if kind==4 and v!=wp.uint32(1) and v!=wp.uint32(2) and v!=wp.uint32(3):
            status[0]=status[0] | 64

@wp.kernel
def gather_words(src:wp.array(dtype=wp.uint32),dst:wp.array(dtype=wp.uint32),
                 count:wp.array(dtype=wp.int32),order:wp.array(dtype=wp.int32),
                 status:wp.array(dtype=wp.int32),row_words:int):
    i=wp.tid();row=i//row_words;source=row
    if status[0]==0 and row<count[0]:
        source=order[row]
    dst[i]=src[source*row_words+i%row_words]
