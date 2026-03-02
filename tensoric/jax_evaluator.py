import jax
import jax.numpy as jnp
import numpy as np
from functools import partial
from vectorized import VectorizedIC
from staticic import APP, LAM

# For memory safety on CPU compiling JIT, we constrain MAX_NODES
# A TPU with 16GB HBM could easily handle 33M.
# Python 256MB Array mapped to 67M elements natively
JAX_MAX_NODES = 67108864

def scan_jax_core(state, _):
    heap, interactions = state
    
    # --- PURE JAX REDEX FINDING ---
    tags = (heap >> 26) & 0x1F
    vals = heap & 0x03FFFFFF
    
    is_app = (tags == APP)
    
    fun_locs = vals
    fun_terms = jnp.take(heap, fun_locs)
    fun_tags = (fun_terms >> 26) & 0x1F
    
    # We only trigger when a function is explicitly a LAM
    app_lam_mask = is_app & (fun_tags == LAM)
    
    l_nodes = jnp.arange(JAX_MAX_NODES)
    r_nodes = fun_locs
    
    args = jnp.take(heap, l_nodes + 1)
    bods = jnp.take(heap, r_nodes + 0)
    
    args_sub = args | jnp.uint32(0x80000000)
    
    # Apply substitutions over the entire graph simultaneously
    # jnp.where only overrides the memory blocks where an active APP-LAM happens
    heap = heap.at[l_nodes].set(jnp.where(app_lam_mask, bods, heap[l_nodes]))
    heap = heap.at[r_nodes].set(jnp.where(app_lam_mask, args_sub, heap[r_nodes]))
    
    inters = jnp.sum(app_lam_mask, dtype=jnp.uint32)
    return (heap, interactions + inters), inters
    
compilation_count = 0

@partial(jax.jit, static_argnums=(1,))
def compiled_scan(state, steps):
    global compilation_count
    compilation_count += 1
    print(f"\n[JAX JIT TRACING] Compiling scan for steps={steps} (Compile Count: {compilation_count})\n")
    if compilation_count > 1:
        print("ERROR: JAX Recompilation Detected!")
    return jax.lax.scan(scan_jax_core, state, None, length=steps)

class JaxIC(VectorizedIC):
    def __init__(self, size=JAX_MAX_NODES):
        super().__init__(size)
        self.jax_heap = None
        
    def step_jax(self):
        # Override regular step to just do 1 scan step
        return self.run_scan(steps=1)
        
    def run_scan(self, steps=100, gc=False, root_term=None):
        if gc and root_term is not None:
            root_term = self.compact(root_term)
            self.jax_heap = None
            
        if self.jax_heap is None:
            self.jax_heap = jnp.array(self.heap)
            
        state = (self.jax_heap, jnp.uint32(0))
        final_state, step_interactions = compiled_scan(state, steps)
        self.jax_heap, scanned_inters = final_state
        
        self.heap = np.array(self.jax_heap)
        
        # In actual TPU inference, `scanned_inters` allows us to early exit.
        # But JAX scan computes full lengths always.
        self.interactions += int(scanned_inters)
        
        has_interactions = int(scanned_inters) > 0
        
        # If interactions == 0, it means it halted structurally or had unsupported redexes
        if gc and root_term is not None:
            return has_interactions, root_term
            
        return has_interactions
