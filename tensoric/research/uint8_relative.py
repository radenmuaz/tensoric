import sys
import os
import numpy as np
import importlib.util
from typing import Tuple

# Add the parent directory to sys.path so we can import tensoric modules
tensoric_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(tensoric_dir)

from staticic import StaticIC

# Dynamically load the local parser.py to avoid collision with standard library 'parser'
spec = importlib.util.spec_from_file_location("local_parser", os.path.join(tensoric_dir, "parser.py"))
local_parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(local_parser)

# --- 16-bit Relative Node Format Constants ---
# High 8 bits:
#   Bit 15: is_sub flag
#   Bits 10-14: TAG (0-31)
#   Bits 8-9: Unused/Reserved
# Low 8 bits:
#   Bits 0-7: Int8 Signed Offset (-128 to 127)

TERM_SUB_MASK_16 = 0x8000
TERM_TAG_MASK_16 = 0x7C00
TERM_VAL_MASK_16 = 0x00FF # We only care about the bottom 8 bits for the offset

# Standard Tags (Imported from staticic, but redefining explicitly for clarity)
VAR = 0x00
LAM = 0x01
APP = 0x02
ERA = 0x03
NUM = 0x04
SUC = 0x05
SWI = 0x06
TMP = 0x07

SP0 = 0x08
DX0 = 0x10
DY0 = 0x18

def get_tag_32(term: np.uint32) -> int:
    return (term & 0x7C000000) >> 26

def get_val_32(term: np.uint32) -> int:
    return term & 0x03FFFFFF

def is_sub_32(term: np.uint32) -> bool:
    return (term & 0x80000000) != 0

def pack_16(is_sub: bool, tag: int, offset: int) -> np.uint16:
    """Packs Tag, Sub flag, and an 8-bit signed offset into a uint16"""
    term = 0
    if is_sub:
        term |= TERM_SUB_MASK_16
    term |= (tag << 10)
    
    # Pack signed 8-bit integer into bottom 8 bits
    # We must properly map negatives to two's complement 8-bit
    if offset < 0:
        offset = 256 + offset
    term |= (offset & 0xFF)
    return np.uint16(term)

def get_tag_16(term: np.uint16) -> int:
    return (term & TERM_TAG_MASK_16) >> 10

def is_sub_16(term: np.uint16) -> bool:
    return (term & TERM_SUB_MASK_16) != 0

def get_offset_16(term: np.uint16) -> int:
    """Extracts the 8-bit signed offset"""
    val = int(term & 0xFF)
    if val > 127:
        return val - 256
    return val

def resolve_ptr(current_idx: int, term: np.uint16) -> int:
    """Resolves a 16-bit relative pointer back to an absolute index"""
    offset = get_offset_16(term)
    return int(current_idx) + offset

# --- Compiler Bridge ---

def compile_to_relative(static_ic: StaticIC, max_nodes=10000) -> Tuple[np.ndarray, int]:
    """
    Takes a 32-bit absolute StaticIC heap and translates it to a 16-bit relative heap.
    Returns: (uint16_heap, new_root_loc)
    """
    abs_heap = static_ic.heap
    rel_heap = np.zeros(max_nodes, dtype=np.uint16)
    
    for i in range(static_ic.heap_pos):
        term = abs_heap[i]
        tag = get_tag_32(term)
        is_sub = is_sub_32(term)
        abs_ptr = get_val_32(term)
        
        # Calculate Relative Offset (Cast to Python int first to avoid uint32 underflow)
        offset = int(abs_ptr) - int(i)
        
        if offset < -128 or offset > 127:
            if tag == NUM or tag == ERA:
                # NUM and ERA don't actually use the pointer payload in V1, safely ignore
                offset = 0
            else:
                raise ValueError(f"FATAL: Distance between Index {i} and Target {abs_ptr} is {offset}! "
                                 f"Exceeds 8-bit signed offset limits (-128 to 127). "
                                 f"Please implement JMP nodes or VAR extensions.")
                
        rel_heap[i] = pack_16(is_sub, tag, offset)
        
    return rel_heap, static_ic.heap_pos

class Uint8RelativeIC:
    """
    An execution engine that runs entirely on 16-bit Nodes with 8-bit Signed Relative Pointers.
    """
    def __init__(self, heap16: np.ndarray, heap_pos: int):
        self.heap = heap16.copy()
        self.heap_pos = heap_pos
        self.interactions = 0
        
        # V1: We'll use absolute allocation for new nodes to keep GC simple, 
        # but all pointers stored IN the nodes will be strictly 8-bit relative.
        
    def alloc(self, n: int) -> int:
        ptr = self.heap_pos
        self.heap_pos += n
        return ptr
        
    def get_tag(self, index: int) -> int:
        return get_tag_16(self.heap[index])
        
    def is_sub(self, index: int) -> bool:
        return is_sub_16(self.heap[index])
        
    def read_ptr(self, current_idx: int) -> int:
        """Reads the relative offset from current_idx and returns the absolute target address"""
        return resolve_ptr(current_idx, self.heap[current_idx])
        
    def write_val(self, current_idx: int, is_sub: bool, tag: int, target_abs_idx: int):
        """Calculates the 8-bit offset and writes the 16-bit node."""
        offset = target_abs_idx - current_idx
        if offset < -128 or offset > 127:
            if tag != NUM and tag != ERA:
                raise ValueError(f"Out of Bounds Relative Jump at {current_idx} targeting {target_abs_idx} (Offset: {offset})")
            offset = 0
        self.heap[current_idx] = pack_16(is_sub, tag, offset)

    # We port a few essential interaction rules carefully checking relative bindings
    
    def ic_app_lam(self, app_idx: int, lam_idx: int) -> int:
        self.interactions += 1
        
        # In a 32-bit APP-LAM, we did:
        # arg = heap[app_loc + 1]
        # bod = heap[lam_loc + 0]
        # heap[lam_loc] = make_sub(arg)
        # return bod
        
        # Let's read the argument (app_idx + 1)
        # The argument node AT app_idx + 1 points to some target.
        # We need to know where it points.
        arg_target = self.read_ptr(app_idx + 1)
        arg_tag = self.get_tag(app_idx + 1)
        
        # Let's read the body
        bod_target = self.read_ptr(lam_idx + 0)
        bod_tag = self.get_tag(lam_idx + 0)
        bod_is_sub = self.is_sub(lam_idx + 0)
        
        # We now overwrite lam_idx to point to arg_target, marked as sub
        self.write_val(lam_idx, True, arg_tag, arg_target)
        
        # We need to return the body definition so the evaluator can continue.
        # To strictly replicate the API, we need a way to pass the body back as a transient node
        # We'll just return the target absolute address and let the evaluator trace it.
        # However StaticIC returns a packed term. We'll return the absolute Loc.
        
        return bod_target

    def whnf_scan(self, start_loc: int) -> int:
        # A vastly simplified linear structural scanner to prove the relative logic works 
        # on small networks before mapping the full stack machine.
        pass

def run_test(filepath: str):
    print(f"--- Testing {os.path.basename(filepath)} ---")
    
    # 1. Compile to 32-bit Absolute using normal TensorIC compiler
    ic32 = StaticIC(size=500000)
    root32_loc = local_parser.parse_file(ic32, filepath)
    
    print(f"Original 32-bit Interactions: (Running WHNF...)")
    # We copy the engine to run without corrupting our source heap
    import copy
    runner32 = copy.deepcopy(ic32)
    norm32_term = runner32.ic_normal(root32_loc)
    print(f"  > Completed in {runner32.interactions} interactions.")
    print(f"  > Normal Root Tag: {runner32.get_tag(norm32_term)}")
    
    # 2. Compile to 16-bit Relative
    try:
        heap16, heap_pos = compile_to_relative(ic32)
        print(f"Succesfully compiled to 16-bit relative pointers! (Size: {heap_pos} nodes)")
        # Proof of conversion
        for i in range(1, 5):
            if i < heap_pos:
                term32 = ic32.heap[i]
                term16 = heap16[i]
                abs_target = get_val_32(term32)
                offset16 = get_offset_16(term16)
                resolved = resolve_ptr(i, term16)
                print(f"  Node {i}: 32-bit Target={abs_target} | 16-bit Offset={offset16} -> Target={resolved}")
                assert abs_target == resolved or get_tag_32(term32) in [NUM, ERA], "Pointer corruption!"
    except Exception as e:
        print(f"Failed to compile to relative! {e}")

if __name__ == "__main__":
    # We are in tensoric/research, so examples is at ../../examples
    examples_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "examples"))
    # test_0.ic and test_1.ic are massive recursive P19/Church numeral tests
    # that explode to >500,000 nodes without GC. We test our 8-bit pointer
    # resolution on bounded structural tests first.
    skip_files = ["test_0.ic", "test_1.ic"]
    
    for f in sorted(os.listdir(examples_dir)):
        if f.endswith(".ic") and f not in skip_files:
            run_test(os.path.join(examples_dir, f))
            print()
