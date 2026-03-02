import numpy as np

# Core Tags (Reusing existing 0-3 for compatibility, adding REP)
ERA = 0  # Eraser
LAM = 1  # Lambda Abstraction
APP = 2  # Application
SUP = 3  # Superposition / Fan
DUP = 4  # Duplication (Legacy)
VAR = 5  # Variable
OP2 = 6  # Binary Op
NUM = 7  # Number

# Delta-Nets Additions
REP = 8  # Replicator (Fat Pointer)

# Port indexing
P1 = 0
P2 = 1

def new_node(heap, heap_pos, tag, p1, p2):
    heap[heap_pos] = tag
    heap[heap_pos + 1] = p1
    heap[heap_pos + 2] = p2
    return heap_pos, heap_pos + 4

def new_rep_node(heap, heap_pos, p1, p2, level, delta_l, delta_r):
    """
    Allocates a Replicator node.
    It acts as a Fat Pointer: the principal node occupies 4 slots, 
    but its P1/P2 are standard. Its Metadata (level, deltas) is allocated 
    as a contiguous block at the end of the heap.
    """
    # 1. Allocate Metadata Block (Linked List Chunk)
    # [Level, Delta_L, Delta_R, Next_Chunk_Ptr (0 if none)]
    meta_ptr = heap_pos
    heap[meta_ptr] = level
    heap[meta_ptr + 1] = delta_l
    heap[meta_ptr + 2] = delta_r
    heap[meta_ptr + 3] = 0 # No spillover yet
    heap_pos += 4
    
    # 2. Allocate the Replicator Node itself
    # P1/P2 are the aux ports. We need a way to store the meta_ptr.
    # We will hack the 'tag' to include the meta_ptr using bitwise ops or just store it in P1
    # For now, let's keep it simple: A REP node needs 4 slots. 
    # [TAG=REP, P1, P2, META_PTR]
    rep_ptr = heap_pos
    heap[rep_ptr] = REP
    heap[rep_ptr + 1] = p1
    heap[rep_ptr + 2] = p2
    heap[rep_ptr + 3] = meta_ptr
    heap_pos += 4
    
    return rep_ptr, heap_pos

def get_rep_meta(heap, rep_ptr):
    meta_ptr = heap[rep_ptr + 3]
    level = heap[meta_ptr]
    delta_l = heap[meta_ptr + 1]
    delta_r = heap[meta_ptr + 2]
    return level, delta_l, delta_r

def set_rep_meta(heap, rep_ptr, level, delta_l, delta_r):
    meta_ptr = heap[rep_ptr + 3]
    heap[meta_ptr] = level
    heap[meta_ptr + 1] = delta_l
    heap[meta_ptr + 2] = delta_r

def interact_rep_rep(heap, heap_pos, r1_ptr, r2_ptr):
    """
    Implements Replicator-Replicator rule from delta-nets.
    """
    l1, dl1, dr1 = get_rep_meta(heap, r1_ptr)
    l2, dl2, dr2 = get_rep_meta(heap, r2_ptr)
    
    # Annihilation
    if l1 == l2:
        # P1 <-> P1, P2 <-> P2
        heap[heap[r1_ptr + 1]] = heap[r2_ptr + 1]
        heap[heap[r2_ptr + 1]] = heap[r1_ptr + 1]
        
        heap[heap[r1_ptr + 2]] = heap[r2_ptr + 2]
        heap[heap[r2_ptr + 2]] = heap[r1_ptr + 2]
        return heap_pos
        
    # Commutation (Mismatched levels)
    # The lower level Replicator duplicates the higher level one.
    if l1 < l2:
        lower_ptr, lower_l, lower_dl, lower_dr = r1_ptr, l1, dl1, dr1
        higher_ptr, higher_l, higher_dl, higher_dr = r2_ptr, l2, dl2, dr2
    else:
        lower_ptr, lower_l, lower_dl, lower_dr = r2_ptr, l2, dl2, dr2
        higher_ptr, higher_l, higher_dl, higher_dr = r1_ptr, l1, dl1, dr1
        
    # Explosive Commutation (4 new REP nodes)
    # The higher replicator is duplicated into two new ones
    h_left_ptr, heap_pos = new_rep_node(
        heap, heap_pos, p1=0, p2=0, 
        level=higher_l + lower_dl, 
        delta_l=higher_dl, delta_r=higher_dr
    )
    h_right_ptr, heap_pos = new_rep_node(
        heap, heap_pos, p1=0, p2=0, 
        level=higher_l + lower_dr, 
        delta_l=higher_dl, delta_r=higher_dr
    )
    
    # The lower replicator is duplicated into two new ones
    l_left_ptr, heap_pos = new_rep_node(
        heap, heap_pos, p1=0, p2=0, 
        level=lower_l, 
        delta_l=lower_dl, delta_r=lower_dr
    )
    l_right_ptr, heap_pos = new_rep_node(
        heap, heap_pos, p1=0, p2=0, 
        level=lower_l, 
        delta_l=lower_dl, delta_r=lower_dr
    )
    
    # Wire them up according to Delta-Nets Rule
    # High's P1 ports go to Low Left
    heap[h_left_ptr + 1] = l_left_ptr + 1
    heap[l_left_ptr + 1] = h_left_ptr + 1
    
    heap[h_right_ptr + 1] = l_left_ptr + 2
    heap[l_left_ptr + 2] = h_right_ptr + 1
    
    # High's P2 ports go to Low Right
    heap[h_left_ptr + 2] = l_right_ptr + 1
    heap[l_right_ptr + 1] = h_left_ptr + 2
    
    heap[h_right_ptr + 2] = l_right_ptr + 2
    heap[l_right_ptr + 2] = h_right_ptr + 2
    
    # Connect to the original exterior graph
    heap[h_left_ptr] = heap[higher_ptr + 1]
    heap[heap[higher_ptr + 1]] = h_left_ptr
    
    heap[h_right_ptr] = heap[higher_ptr + 2]
    heap[heap[higher_ptr + 2]] = h_right_ptr
    
    heap[l_left_ptr] = heap[lower_ptr + 1]
    heap[heap[lower_ptr + 1]] = l_left_ptr
    
    heap[l_right_ptr] = heap[lower_ptr + 2]
    heap[heap[lower_ptr + 2]] = l_right_ptr
    
    return heap_pos

def interact_rep_fan(heap, heap_pos, rep_ptr, fan_ptr):
    """
    Implements Replicator-Fan rule from delta-nets.
    """
    l, dl, dr = get_rep_meta(heap, rep_ptr)
    
    # The Fan is duplicated, the Replicator is duplicated out the auxiliary ports
    fan_left_ptr, heap_pos = new_node(heap, heap_pos, SUP, 0, 0)
    fan_right_ptr, heap_pos = new_node(heap, heap_pos, SUP, 0, 0)
    
    rep_left_ptr, heap_pos = new_rep_node(
        heap, heap_pos, p1=0, p2=0, level=l, delta_l=dl, delta_r=dr
    )
    rep_right_ptr, heap_pos = new_rep_node(
        heap, heap_pos, p1=0, p2=0, level=l, delta_l=dl, delta_r=dr
    )
    
    # Wire interior Commutation square
    heap[fan_left_ptr + 1] = rep_left_ptr + 1
    heap[rep_left_ptr + 1] = fan_left_ptr + 1
    
    heap[fan_right_ptr + 1] = rep_left_ptr + 2
    heap[rep_left_ptr + 2] = fan_right_ptr + 1
    
    heap[fan_left_ptr + 2] = rep_right_ptr + 1
    heap[rep_right_ptr + 1] = fan_left_ptr + 2
    
    heap[fan_right_ptr + 2] = rep_right_ptr + 2
    heap[rep_right_ptr + 2] = fan_right_ptr + 2
    
    # Wire to original graph
    heap[fan_left_ptr] = heap[rep_ptr + 1]
    heap[heap[rep_ptr + 1]] = fan_left_ptr
    
    heap[fan_right_ptr] = heap[rep_ptr + 2]
    heap[heap[rep_ptr + 2]] = fan_right_ptr
    
    heap[rep_left_ptr] = heap[fan_ptr + 1]
    heap[heap[fan_ptr + 1]] = rep_left_ptr
    
    heap[rep_right_ptr] = heap[fan_ptr + 2]
    heap[heap[fan_ptr + 2]] = rep_right_ptr
    
    return heap_pos
