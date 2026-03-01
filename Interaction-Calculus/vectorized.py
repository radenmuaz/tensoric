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
