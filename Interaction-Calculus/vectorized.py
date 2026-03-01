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
        # We simulate the WHNF traversal finding all parallel active pairs (redexes).
        # In a true NPU it would evaluate the graph edges continuously, 
        # but here we do a sweep.
        
        visited = set()
        stack = [start_term]
        self.redex_count = 0
        
        # Simple tree/graph search for interactions
        # This is strictly a bridge to find interactions to simulate NPU parallel execution
        # An actual GPU/NPU representation implies tracking edges explicitly.
        
        def check_interaction(loc):
            if loc in visited: return
            visited.add(loc)
            
            term = self.heap[loc]
            tag = self.get_tag(term)
            
            if tag == APP:
                fun = self.heap[loc]
                arg = self.heap[loc+1]
                fun_tag = self.get_tag(fun)
                if fun_tag in [LAM, ERA] or self.is_sup(fun_tag):
                    self.active_redexes_l[self.redex_count] = loc      # APP loc
                    self.active_redexes_r[self.redex_count] = self.get_val(fun) # interacting fn
                    self.redex_count += 1
                else:
                    check_interaction(self.get_val(fun))
                    check_interaction(self.get_val(arg))
                    
            elif self.is_dup(tag):
                val = self.heap[loc]
                val_tag = self.get_tag(val)
                if val_tag in [LAM, ERA, NUM] or self.is_sup(val_tag):
                    self.active_redexes_l[self.redex_count] = loc      # DUP loc
                    self.active_redexes_r[self.redex_count] = self.get_val(val)
                    self.redex_count += 1
                else:
                    check_interaction(self.get_val(val))
                    
            # ... and so on for SUC and SWI
            # This is complex to extract from the WHNF tree implicitly.
            
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
