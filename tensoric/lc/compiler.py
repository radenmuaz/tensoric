from .ast import LCNode, LCVar, LCLam, LCApp, LCNum, LCSuc, LCSwi
from ..base.ic_ast import ICNode, ICVar, ICLam, ICApp, ICDup, ICRep, ICEra, ICNum, ICSuc, ICSwi

class LCCompiler:
    def __init__(self):
        self.dup_counter = 0

    def get_fresh_name(self, base):
        self.dup_counter += 1
        return f"{base}_{self.dup_counter}"

    def analyze_usage(self, node, env):
        if isinstance(node, LCVar):
            if node.name in env:
                return [node.name]
            return []
        elif isinstance(node, LCLam):
            inner_env = env.copy()
            inner_env.discard(node.param)
            return self.analyze_usage(node.body, inner_env)
        elif isinstance(node, LCApp):
            return self.analyze_usage(node.fun, env) + self.analyze_usage(node.arg, env)
        elif isinstance(node, LCSuc):
            return self.analyze_usage(node.val, env)
        elif isinstance(node, LCSwi):
            return (self.analyze_usage(node.cond, env) + 
                    self.analyze_usage(node.z_branch, env) + 
                    self.analyze_usage(node.s_branch, env))
        return []

    def rewrite_var(self, node, target, fresh_vars):
        if isinstance(node, LCVar):
            if node.name == target:
                if not fresh_vars:
                    return node
                return LCVar(fresh_vars.pop(0))
            return node
        elif isinstance(node, LCLam):
            if node.param == target:
                return node
            return LCLam(node.param, self.rewrite_var(node.body, target, fresh_vars))
        elif isinstance(node, LCApp):
            return LCApp(self.rewrite_var(node.fun, target, fresh_vars),
                         self.rewrite_var(node.arg, target, fresh_vars))
        elif isinstance(node, LCSuc):
            return LCSuc(self.rewrite_var(node.val, target, fresh_vars))
        elif isinstance(node, LCSwi):
            return LCSwi(self.rewrite_var(node.cond, target, fresh_vars),
                         self.rewrite_var(node.z_branch, target, fresh_vars),
                         self.rewrite_var(node.s_branch, target, fresh_vars))
        return node

    def make_dup(self, v0, v1, val, body, scope_level):
        """Standard IC Duplicator: uses labels."""
        lab = self.dup_counter % 8
        self.dup_counter += 1
        return ICDup(v0, v1, val, body, lab)

    def compile(self, node, scope_level=0):
        if isinstance(node, LCVar):
            return ICVar(node.name)
        elif isinstance(node, LCNum):
            return ICNum(node.val)
        elif isinstance(node, LCSuc):
            return ICSuc(self.compile(node.val, scope_level))
        elif isinstance(node, LCSwi):
            return ICSwi(self.compile(node.cond, scope_level), 
                         self.compile(node.z_branch, scope_level), 
                         self.compile(node.s_branch, scope_level))
        elif isinstance(node, LCApp):
            return ICApp(self.compile(node.fun, scope_level), self.compile(node.arg, scope_level))
        elif isinstance(node, LCLam):
            param = node.param
            body = node.body
            
            usages = self.analyze_usage(body, {param})
            count = usages.count(param)
            
            if count == 0:
                return ICLam(param, self.compile(body, scope_level + 1))
            elif count == 1:
                return ICLam(param, self.compile(body, scope_level + 1))
            else:
                fresh_vars = [self.get_fresh_name(param) for _ in range(count)]
                rewritten_body = self.rewrite_var(body, param, fresh_vars.copy())
                
                final_body = self.compile(rewritten_body, scope_level + 1)
                
                current_var = param
                dups = []
                for i in range(count - 1):
                    v0 = fresh_vars[i]
                    if i == count - 2:
                        v1 = fresh_vars[i+1]
                    else:
                        v1 = self.get_fresh_name(f"{param}_rem")
                    
                    dups.append((v0, v1, current_var))
                    current_var = v1
                
                for v0, v1, val in reversed(dups):
                    final_body = self.make_dup(v0, v1, ICVar(val), final_body, scope_level)
                
                return ICLam(param, final_body)
        return None

class DeltaLCCompiler(LCCompiler):
    def make_dup(self, v0, v1, val, body, scope_level):
        """Delta-Nets Replicator: uses scope levels."""
        return ICRep(v0, v1, val, body, level=scope_level, delta_l=0, delta_r=0)
