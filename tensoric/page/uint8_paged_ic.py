import jax
import jax.numpy as jnp
import numpy as np

# Node Tags
ERA = 0
LAM = 1
APP = 2
SUP = 3
BRG = 4  # Bridge / Substitution

# Heap Shape: (NUM_PAGES, NODES_PER_PAGE, 4 bytes)
# Node Structure: [TAG, PORT1, PORT2, META]
# A BRG Node Structure: [BRG, TARGET_PAGE_LOW, TARGET_PAGE_HIGH, TARGET_IDX]

@jax.jit
def resolve_pointers(heap, ptr_pages, ptr_idxs):
    """
    Given arrays of pointers (page, idx), traces through BRG nodes 
    to find the concrete target. This uses vectorized jnp.where.
    """
    # Read the tags at the target locations
    target_tags = heap[ptr_pages, ptr_idxs, 0]
    
    is_bridge = (target_tags == BRG)
    
    # Read the BRG payload (interpreting as bytes)
    brg_page_low = heap[ptr_pages, ptr_idxs, 1].astype(jnp.uint16)
    brg_page_high = heap[ptr_pages, ptr_idxs, 2].astype(jnp.uint16)
    brg_idx = heap[ptr_pages, ptr_idxs, 3]
    
    brg_target_page = (brg_page_high << 8) | brg_page_low
    
    # Multiplex the resolved pointers based on the `is_bridge` boolean mask
    resolved_pages = jnp.where(is_bridge, brg_target_page, ptr_pages)
    resolved_idxs = jnp.where(is_bridge, brg_idx, ptr_idxs)
    
    return resolved_pages, resolved_idxs

@jax.jit
def step_uint8_paged(heap):
    """
    Performs one step of parallel APP-LAM reduction on the 3D uint8 paged heap.
    """
    NUM_PAGES, NODES_PER_PAGE, _ = heap.shape
    
    # 1. Identify all APP nodes globally
    tags = heap[:, :, 0]
    is_app = (tags == APP)
    
    # 2. Extract function pointers (PORT 1) for all APPs.
    # We assume pointers are initially local (same page) unless resolved through a BRG.
    app_pages = jnp.broadcast_to(jnp.arange(NUM_PAGES)[:, None], (NUM_PAGES, NODES_PER_PAGE))
    app_fun_idxs = heap[:, :, 1]
    
    # 3. Resolve possible bridges across the entire array concurrently!
    # This evaluates whether the function pointer crosses a page.
    fun_pages, fun_idxs = resolve_pointers(heap, app_pages, app_fun_idxs)
    
    # 4. Check if the resolved true function node is a LAM
    fun_tags = heap[fun_pages, fun_idxs, 0]
    app_lam_mask = is_app & (fun_tags == LAM)
    
    # --- Execute Parallel Overwrites ---
    # The LAM's argument needs to point to the APP's argument.
    # The APP's location needs to forward to the LAM's body.
    
    # Linearize the 2D grid for safe JAX Scatter Updates
    app_arg_idxs = heap[:, :, 2]
    lam_body_idxs = heap[fun_pages, fun_idxs, 1]
    
    flat_app_indices = jnp.arange(NUM_PAGES * NODES_PER_PAGE)
    flat_lam_indices = fun_pages.flatten() * NODES_PER_PAGE + fun_idxs.flatten()
    mask_flat = app_lam_mask.flatten()
    
    flat_heap = heap.reshape(-1, 4)
    
    # 1. Update LAM nodes
    # LAM morphs into BRG -> pointing to APP's ARGUMENT
    new_lam_node_tag = jnp.full_like(flat_lam_indices, BRG, dtype=jnp.uint8)
    new_lam_node_pg_low = (app_pages.flatten() & 0xFF).astype(jnp.uint8)
    new_lam_node_pg_high = ((app_pages.flatten() >> 8) & 0xFF).astype(jnp.uint8)
    new_lam_node_idx = app_arg_idxs.flatten().astype(jnp.uint8)
    
    new_lam_nodes = jnp.stack([
        new_lam_node_tag, new_lam_node_pg_low, new_lam_node_pg_high, new_lam_node_idx
    ], axis=-1)
    
    safe_lam_nodes = jnp.where(mask_flat[:, None], new_lam_nodes, flat_heap[flat_lam_indices])
    flat_heap = flat_heap.at[flat_lam_indices].set(safe_lam_nodes)
    
    # 2. Update APP nodes
    # APP morphs into BRG -> pointing to LAM's BODY
    new_app_node_tag = jnp.full_like(flat_app_indices, BRG, dtype=jnp.uint8)
    new_app_node_pg_low = (fun_pages.flatten() & 0xFF).astype(jnp.uint8)
    new_app_node_pg_high = ((fun_pages.flatten() >> 8) & 0xFF).astype(jnp.uint8)
    new_app_node_idx = lam_body_idxs.flatten().astype(jnp.uint8)
    
    new_app_nodes = jnp.stack([
        new_app_node_tag, new_app_node_pg_low, new_app_node_pg_high, new_app_node_idx
    ], axis=-1)
    
    safe_app_nodes = jnp.where(mask_flat[:, None], new_app_nodes, flat_heap[flat_app_indices])
    flat_heap = flat_heap.at[flat_app_indices].set(safe_app_nodes)
    
    # Recover metrics
    interactions = jnp.sum(app_lam_mask)
    
    return flat_heap.reshape(NUM_PAGES, NODES_PER_PAGE, 4), interactions

if __name__ == "__main__":
    # Test setting up a Long-Distance APP-LAM
    test_heap = np.zeros((2, 64, 4), dtype=np.uint8)
    
    # Page 0, Node 5: APP. Port 1 points to a local BRG node at index 6
    # Port 2 (Arg) points to local node 10
    test_heap[0, 5] = [APP, 6, 10, 0]
    
    # Page 0, Node 6: BRG. It forwards to Page 1, Node 20
    # Target Page = 1 (Low=1, High=0)
    test_heap[0, 6] = [BRG, 1, 0, 20] 
    
    # Page 1, Node 20: LAM. Body points to local node 21
    test_heap[1, 20] = [LAM, 21, 0, 0]
    
    print("--- Pre-Evaluation State ---")
    print("Page 0 Node 5 (APP):  ", test_heap[0, 5])
    print("Page 0 Node 6 (BRG):  ", test_heap[0, 6], " -> Forwards to Page 1, Node 20")
    print("Page 1 Node 20 (LAM): ", test_heap[1, 20])
    
    j_heap = jnp.array(test_heap)
    new_heap, inters = step_uint8_paged(j_heap)
    
    print(f"\nCompleted {inters} Cross-Page interactions natively in early JAX.")
    
    print("\n--- Post-Evaluation State ---")
    print("Page 0 Node 5 (Was APP, now Cross-Page BRG to Lam Body [Page 1, Node 21]):")
    print(new_heap[0, 5])
    print("Page 1 Node 20 (Was LAM, now Cross-Page BRG to App Arg [Page 0, Node 10]):")
    print(new_heap[1, 20])
