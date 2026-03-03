import jax
import jax.numpy as jnp
from staticic import APP, LAM, ERA, NUM, SUC, SWI, VAR, TMP, SP0, SP7, DX0, DX7, DY0, DY7

# Mock JAX_MAX_NODES for fast trace verification
JAX_MAX_NODES = 100

def get_ports(heap, node_idx):
    """Returns the two children (port 1, port 2) of a given node in the graph"""
    term = heap[node_idx]
    tag = (term >> 26) & 0x1F
    val = term & 0x03FFFFFF
    
    # Binary nodes use val (port 1) and val + 1 (port 2) dynamically
    # ERA uses none. NUM uses none.
    
    # Binary constructs: APP, LAM, SWI, TMP, SPx, DXx, DYx
    is_binary = (tag == APP) | (tag == LAM) | (tag == SWI) | (tag == TMP) | ((tag >= SP0) & (tag <= DY7))
    
    p1 = jnp.where(is_binary, val, 0)
    p2 = jnp.where(is_binary, val + 1, 0)
    
    # SUC nodes only have 1 child (port 1 = val)
    p1 = jnp.where((tag == SUC), val, p1)
    
    return p1, p2

def jax_mark_sweep(heap, root_loc):
    """
    Computes a boolean `alive_mask` over the entire heap using parallel BFS/DFS.
    Since JAX doesn't allow dynamic queues, we iterate a fix-point algorithm:
    Start with `alive = zeros`. `alive[root] = 1`.
    Loop:
      new_alive = alive
      for every node in `new_alive`, mask its children as 1 too.
      if `new_alive == alive`, halt (graph is fully explored).
    """
    
    def cond_fun(state):
        _, alive, prev_alive = state
        return jnp.any(alive != prev_alive)
        
    def body_fun(state):
        h, alive, _ = state
        
        # We need to find all children of Currently Alive nodes
        # To do this parallel, we evaluate get_ports over EVERY node, but only keep results for `alive`
        all_nodes = jnp.arange(JAX_MAX_NODES)
        p1, p2 = get_ports(h, all_nodes)
        
        # Keep only ports belonging to alive nodes
        p1_alive = jnp.where(alive, p1, 0)
        p2_alive = jnp.where(alive, p2, 0)
        
        # Scatter updates into the new alive mask
        next_alive = alive
        
        # We need a 1 at index p1_alive and p2_alive
        next_alive = next_alive.at[p1_alive].set(1)
        next_alive = next_alive.at[p2_alive].set(1)
        
        # Address 0 is typically NULL/Empty, but if root was 0, keep it.
        # Actually our allocator never uses 0 so let's enforce index > 0
        next_alive = next_alive.at[0].set(jnp.where(root_loc == 0, 1, 0))

        return (h, next_alive, alive)
        
    init_alive = jnp.zeros(JAX_MAX_NODES, dtype=jnp.uint8)
    init_alive = init_alive.at[root_loc].set(1)
    
    # Force first pass
    dummy_prev = jnp.zeros(JAX_MAX_NODES, dtype=jnp.uint8)
    
    final_heap, final_alive, _ = jax.lax.while_loop(cond_fun, body_fun, (heap, init_alive, dummy_prev))
    return final_alive

def jax_compact(heap, root_loc):
    """
    Uses the alive mask to perform a parallel Prefix Sum (CumSum) compaction!
    """
    alive_mask = jax_mark_sweep(heap, root_loc)
    
    # Prefix Sum translates the boolean mask [0, 1, 0, 1] into new IDs [0, 1, 1, 2]
    new_ids = jnp.cumsum(alive_mask)
    
    # The new length of our heap is the maximum of the cumsum
    num_alive = new_ids[-1]
    
    # 1. We must scatter the ALIVE elements into their new locations!
    # original_indices = [0, 1, 2, 3, 4]
    # alive_mask =       [1, 0, 1, 1, 0]
    # new_ids =          [1, 1, 2, 3, 3]  (Since 0 is null, we might want IDs to start at 1.)
    
    # But wait! Node Pointers inside the terms must ALSO be updated to point to `new_ids[val]`!
    tags = (heap >> 26) & 0x1F
    vals = heap & 0x03FFFFFF
    subs = heap & jnp.uint32(0x80000000)
    labels = (heap >> 26) & 0x3FFFFFF # Everything except the sub bit. But the label is strictly overlapping tags for simplicity here...
    
    # Update all values mapped
    updated_vals = jnp.take(new_ids, vals)
    
    # Recombine term
    updated_terms = (subs) | (heap & 0x7C000000) | updated_vals # keep tag and label exactly, just swap bottom 26 bytes!
    
    # Let's allocate the new zeroed out heap
    compacted_heap = jnp.zeros_like(heap)
    
    # Scatter active nodes
    all_indices = jnp.arange(JAX_MAX_NODES)
    
    # We only scatter where alive_mask == 1
    # Note: `new_ids` is 1-indexed (e.g., first alive element gets 1). 
    # If we want 0-indexed memory bounds, we do `new_ids - 1`
    target_indices = jnp.where(alive_mask, new_ids - 1, 0)
    
    compacted_heap = compacted_heap.at[target_indices].set(jnp.where(alive_mask, updated_terms, 0))
    
    return compacted_heap, new_ids[root_loc] - 1

if __name__ == "__main__":
    # Test it
    h = jnp.zeros(JAX_MAX_NODES, dtype=jnp.uint32)
    # Root
    h = h.at[1].set((APP << 26) | 2) # Node 1 contains APP, pointing to 2
    h = h.at[2].set((LAM << 26) | 3) # Node 2 contains LAM, pointing to 3
    h = h.at[3].set((NUM << 26) | 42) # Node 3 contains NUM
    h = h.at[4].set((NUM << 26) | 99) # DEAD NODE! Node 4 should vanish
    
    print("Initial Heap[1:5]:")
    print(h[1:5])
    
    c_heap, new_root = jax_compact(h, 1)
    print("\nCompacted Heap[0:4]:")
    print(c_heap[0:4])
    print("New Root:", new_root)
