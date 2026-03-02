import jax
import jax.numpy as jnp
from tensoric.research.delta_ic import REP, new_rep_node, new_node, APP, LAM
from tensoric.research.delta_vectorized import find_delta_redexes

def test_delta_compilation():
    """
    Tests if the new Delta-Nets memory bank and masks compile under JAX.
    """
    # Create a small heap (100 slots)
    heap = np.zeros(100, dtype=np.int32)
    heap_pos = 0
    
    # 1. Create a Replicator (Fat Node)
    r1_ptr, heap_pos = new_rep_node(heap, heap_pos, p1=20, p2=30, level=1, delta_l=0, delta_r=0)
    
    # 2. Create an opposing Replicator
    r2_ptr, heap_pos = new_rep_node(heap, heap_pos, p1=40, p2=50, level=1, delta_l=0, delta_r=0)
    
    # 3. Link them to form a redex (R1 points to R2, R2 points to R1)
    heap[r1_ptr] = r2_ptr
    heap[r2_ptr] = r1_ptr
    
    print(f"Heap Status after allocations: {heap_pos} slots used.")
    print("Testing JAX mask compilation...")
    
    j_heap = jnp.array(heap)
    # Just run the find_delta_redexes mask to ensure JAX can trace it
    mask = find_delta_redexes(j_heap, jnp.array(0))
    
    print("Compilation successful! Mask generated.")
    
if __name__ == "__main__":
    import numpy as np
    test_delta_compilation()
