"""Initialize only active rigid Contact metadata to the native reset value (zero)."""
import warp as wp
wp.set_module_options({'enable_backward':False})

@wp.kernel
def rigid_metadata(nacon:wp.array(dtype=wp.int32),flex:wp.array(dtype=wp.vec2i),vert:wp.array(dtype=wp.vec2i)):
    i=wp.tid()
    if i<nacon[0]:
        flex[i]=wp.vec2i(0,0)
        vert[i]=wp.vec2i(0,0)
