import numpy as np
from staticic import *

class VectorizedIC(StaticIC):
    def __init__(self, size=MAX_NODES):
        super().__init__(size)
        # Vectorized internal state mapping:
        # Instead of traversing pointers, we will keep a flat active interactions array 
        # (Redexes)
        self.active_redexes_l = np.zeros(size, dtype=np.uint32)
        self.active_redexes_r = np.zeros(size, dtype=np.uint32)
        self.redex_count = 0
        
    def find_all_redexes(self, start_term):
        self.redex_count = 0
        
        # In a real TPU execution setting, you would compute active redexes 
        # using parallel array boolean masking over the entire heap.
        # This python loop simulates that full sweep.
        
        for loc in range(1, self.heap_pos):
            term = self.heap[loc]
            tag = self.get_tag(term)
            
            if tag == APP:
                fun_loc = self.get_val(term)
                fun_term = self.heap[fun_loc]
                
                # Resolve substitution pointers
                while self.get_tag(fun_term) == VAR and self.is_sub(self.heap[self.get_val(fun_term)]):
                    fun_term = self.clear_sub(self.heap[self.get_val(fun_term)])
                    
                fun_tag = self.get_tag(fun_term)
                if fun_tag in [LAM, ERA] or self.is_sup(fun_tag):
                    self.active_redexes_l[self.redex_count] = loc
                    self.active_redexes_r[self.redex_count] = self.get_val(fun_term) if fun_tag == LAM else self.get_val(fun_term)
                    self.redex_count += 1
                    
            elif self.is_dup(tag):
                val_loc = self.get_val(term)
                val_term = self.heap[val_loc]
                
                while self.get_tag(val_term) == VAR and self.is_sub(self.heap[self.get_val(val_term)]):
                    val_term = self.clear_sub(self.heap[self.get_val(val_term)])
                    
                val_tag = self.get_tag(val_term)
                if val_tag in [LAM, ERA, NUM] or self.is_sup(val_tag):
                    self.active_redexes_l[self.redex_count] = loc
                    self.active_redexes_r[self.redex_count] = self.get_val(val_term)
                    self.redex_count += 1
            
    def step_vectorized(self):
        # The goal is to perform rules simultaneously
        if self.redex_count == 0:
            return False # noop finished
            
        # Example for APP-LAM parallel reduction
        l_nodes = self.active_redexes_l[:self.redex_count]
        r_nodes = self.active_redexes_r[:self.redex_count]
        
        # Parallel fetch
        apps = self.heap[l_nodes]
        lams = self.heap[r_nodes]
        
        # Vectorized mask for APP-LAM rule
        app_lam_mask = (self.get_tag(apps) == APP) & (self.get_tag(lams) == LAM)
        idx_app_lam = np.where(app_lam_mask)[0]
        
        if len(idx_app_lam) > 0:
            app_locs = l_nodes[idx_app_lam]
            lam_locs = r_nodes[idx_app_lam]
            
            # Parallel read args and bodies
            args = self.heap[app_locs + 1]
            bods = self.heap[lam_locs + 0]
            
            # Parallel substitutions
            self.heap[lam_locs] = self.make_sub(args)
            self.heap[app_locs] = bods # Simplification of return chaining
            
            self.interactions += len(idx_app_lam)
            
        return True

    def compact(self, root_term):
        new_heap = np.zeros_like(self.heap)
        new_heap_pos = 0
        forward = {} 
        queue = []
        
        def alloc_new(n):
            nonlocal new_heap_pos
            ptr = new_heap_pos
            new_heap_pos += n
            return ptr
            
        def resolve_substitutions(term):
            while True:
                tag = self.get_tag(term)
                val = self.get_val(term)
                if tag == VAR or self.is_dup(tag):
                    subst = self.heap[val]
                    if self.is_sub(subst):
                        term = self.clear_sub(subst)
                        continue
                break
            return term
            
        def queue_term(term):
            """Assigns a new address for the term and adds it to the queue to be copied iteratively."""
            term = resolve_substitutions(term)
            tag = self.get_tag(term)
            val = self.get_val(term)
            is_sub = self.is_sub(term)

            if tag == ERA or tag == NUM:
                return term

            if val in forward:
                return self.make_term(is_sub, tag, forward[val])

            if tag == LAM or tag == SUC or self.is_dup(tag) or tag == VAR:
                size = 1
            elif tag == APP or self.is_sup(tag):
                size = 2
            elif tag == SWI:
                size = 3
            else:
                size = 1

            new_loc = alloc_new(size)
            forward[val] = new_loc
            queue.append((val, new_loc, size))
            return self.make_term(is_sub, tag, new_loc)

        new_root = queue_term(root_term)
        
        while queue:
            old_loc, new_loc, size = queue.pop(0)
            for i in range(size):
                old_val = self.heap[old_loc + i]
                # It's an internal parameter slot so trace it
                resolved_old_val = resolve_substitutions(old_val)
                # Now queue it iteratively instead of doing recursive tracing
                if self.is_sub(old_val):
                    # Maintain the substitution flag
                    mapped_val = queue_term(resolved_old_val)
                    new_heap[new_loc + i] = self.make_sub(mapped_val)
                else:
                    new_heap[new_loc + i] = queue_term(resolved_old_val)
                    
        self.heap = new_heap
        self.heap_pos = new_heap_pos
        
        self.active_redexes_l = np.zeros(self.heap.shape[0], dtype=np.uint32)
        self.active_redexes_r = np.zeros(self.heap.shape[0], dtype=np.uint32)
        self.redex_count = 0
        
        return new_root
