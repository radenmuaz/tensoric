import jax
import jax.numpy as jnp
import numpy as np

# A minimal pure continuous tensor evaluator prototype
# based strictly on Adds and Muls for analog hardware mapping

JAX_MAX_NODES = 1024
NUM_TAGS = 8  # Simplified tags: VAR=0, LAM=1, APP=2, ... for prototyping

VAR = 0
LAM = 1
APP = 2
ERA = 3
NUM = 4
SUC = 5

class ContinuousIC:
    def __init__(self, size=JAX_MAX_NODES):
        self.size = size
        # We represent the heap as two dense matrices:
        # 1. Tags: [N_nodes, NUM_TAGS]  (One-hot continuous probabilities)
        # 2. Pointers: [N_nodes, N_nodes] (One-hot continuous routing matrix)
        self.tags = np.zeros((size, NUM_TAGS), dtype=np.float32)
        self.pointers = np.zeros((size, size), dtype=np.float32)
        
        # Initialize an empty node 0 (Null)
        self.tags[0, VAR] = 1.0  # Just a filler
        self.pointers[0, 0] = 1.0
        
        self.heap_pos = 1

    def alloc(self):
        ptr = self.heap_pos
        self.heap_pos += 1
        return ptr

    def set_node(self, loc, tag, target_ptr):
        self.tags[loc] = 0.0
        self.tags[loc, tag] = 1.0
        self.pointers[loc] = 0.0
        self.pointers[loc, target_ptr] = 1.0

    def make_lam(self, bod_ptr):
        loc = self.alloc()
        self.set_node(loc, LAM, bod_ptr)
        return loc

    def make_app(self, fun_ptr, arg_ptr):
        loc = self.alloc()
        # For simplicity, we assume nodes are sequentially packed:
        # App node explicitly points to the function. Arg node immediately follows.
        self.set_node(loc, APP, fun_ptr)
        
        arg_loc = self.alloc()
        # In actual IC structure, an app node usually takes 2 slots
        # For this dense matrix model, we just make the next node point to the argument.
        self.set_node(arg_loc, VAR, arg_ptr)
        return loc
        
    def make_var(self, ptr):
        loc = self.alloc()
        self.set_node(loc, VAR, ptr)
        return loc

    def get_matrices(self):
        return jnp.array(self.tags), jnp.array(self.pointers)


@jax.jit
def continuous_eval_step(tags, pointers):
    """
    Executes a single step of the APP-LAM interaction over the entire memory continuously.
    Returns the updated (tags, pointers)
    """
    N = tags.shape[0]
    
    # 1. Identify which nodes are currently APPs
    # app_mask shape: (N,)
    app_mask = tags[:, APP]
    
    # 2. Extract function targets by dereferencing pointers
    # If pointers[i, j] = 1.0, and tags[j] has LAM, then this is an active redex.
    # To get the tags of the target nodes, we matrix multiply:
    # target_tags = pointers.dot(tags)
    # Shape logic: pointers (N, N) @ tags (N, NUM_TAGS) -> (N, NUM_TAGS)
    target_tags = jnp.dot(pointers, tags)
    
    # 3. Check if target is a LAM
    target_is_lam = target_tags[:, LAM]
    
    # 4. Continuous formulation of "app_lam_mask = is_app & (fun_tags == LAM)"
    # match is the fuzzy continuous boolean logic (Mul) -> Shape: (N,)
    match = app_mask * target_is_lam
    
    # 5. Shift Matrix to get Arguments (+1 from App slot)
    # Shift matrix S: multiplies by shifting values down by 1
    # Note: jnp.roll can be used, but to show pure matrix multiply:
    i = jnp.arange(N)
    S = jnp.zeros((N, N)).at[i, (i + 1) % N].set(1.0)
    
    # 6. Dereference the pointers to find substitution targets and bodies
    # Arg node pointers are at the next location (pointers @ S)
    arg_node_pointers = jnp.dot(S.T, pointers)
    
    # To find the bodies, we need the pointers of the target LAM node.
    # FunNodesPointers = pointers @ pointers (where it points to)
    lam_body_pointers = jnp.dot(pointers, pointers)
    
    # 7. Apply updates with arithmetic mixing
    # H_new_l (updating the APP node to point to the body)
    # H_new_r (updating the LAM node to point to the argument)

    # For APP node: update its pointer to point to the lam_body_pointers
    # pointers_new = match * lam_body_pointers + (1 - match) * pointers
    updated_app_pointers = match[:, None] * lam_body_pointers + (1.0 - match[:, None]) * pointers
    
    # We also need to update the LAM nodes to point to the Arg nodes.
    # However, 'match' is aligned to the APP notes. We must route it forward to the LAM node locations.
    routed_match = jnp.dot(pointers.T, match) # Project match strength to the LAM node
    
    # The argument to substitute is arg_node_pointers
    # We must route arg_node_pointers forward to the LAM node exactly
    routed_arg_pointers = jnp.dot(pointers.T, arg_node_pointers) # Very fuzzy continuous routing
    
    # Update LAM node pointers
    updated_pointers = routed_match[:, None] * routed_arg_pointers + (1.0 - routed_match[:, None]) * updated_app_pointers
    
    # Update Tags for purity
    # The APP node becomes a VAR pointing to the LAM's body
    # The LAM node becomes a VAR pointing to the Argument
    
    # We want to change the APP node's tag to VAR where match == 1
    # tags shape: (N, NUM_TAGS)
    var_tag = jnp.zeros(NUM_TAGS).at[VAR].set(1.0)
    
    # Broadcast match for tags
    match_tags = match[:, None]
    updated_tags = match_tags * var_tag + (1.0 - match_tags) * tags
    
    # We also do the same for the LAM node.
    routed_match_tags = routed_match[:, None]
    final_tags = routed_match_tags * var_tag + (1.0 - routed_match_tags) * updated_tags
           
    return final_tags, updated_pointers

def test_evaluator():
    ic = ContinuousIC(size=8)
    
    # Construct an Identity Function Lam(x -> x)
    # Bod is VAR pointing to param.
    id_param = ic.alloc() # We allocate a slot for the parameter substitution
    id_bod = ic.make_var(id_param)
    id_lam = ic.make_lam(id_bod)
    
    # Construct an Argument Num(42)
    # For now we just use VAR pointing to an external constant slot
    const_slot = ic.alloc()
    ic.set_node(const_slot, NUM, const_slot)
    
    app_node = ic.make_app(id_lam, const_slot)
    
    t, p = ic.get_matrices()
    
    np.set_printoptions(precision=1, suppress=True)
    print("Tags Legend: 0=VAR, 1=LAM, 2=APP, 3=ERA, 4=NUM, 5=SUC")
    print("\nInitial Pointers:")
    print(np.round(p, 1))
    print("\nInitial Tags:")
    print(np.argmax(t, axis=1))
    
    for step in range(3):
        t, p = continuous_eval_step(t, p)
        print(f"\nStep {step+1} Pointers:")
        print(np.round(p, 1))
        print(f"Step {step+1} Tags:")
        print(np.argmax(t, axis=1))

if __name__ == "__main__":
    test_evaluator()
