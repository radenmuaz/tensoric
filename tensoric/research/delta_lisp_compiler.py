from tensoric.lisp_parser import LispSymbol, LispNum, LispList, parse_lisp

class ICNode:
    pass

class ICVar(ICNode):
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return f"{self.name}"

class ICLam(ICNode):
    def __init__(self, param, body):
        self.param = param
        self.body = body
    def __repr__(self):
        return f"λ{self.param}.({self.body})"

class ICApp(ICNode):
    def __init__(self, fun, arg):
        self.fun = fun
        self.arg = arg
    def __repr__(self):
        return f"({self.fun} {self.arg})"

class ICRep(ICNode):
    """
    Delta-Nets Replicator Node.
    Differs from ICDup by storing level and deltas instead of just a label.
    """
    def __init__(self, var0, var1, val, body, level=0, delta_l=0, delta_r=0):
        self.var0 = var0
        self.var1 = var1
        self.val = val
        self.body = body
        self.level = level
        self.delta_l = delta_l
        self.delta_r = delta_r
    def __repr__(self):
        return f"REP[L{self.level} dL{self.delta_l} dR{self.delta_r}]{{{self.var0},{self.var1}}} = {self.val};\n{self.body}"

class ICEra(ICNode):
    def __repr__(self):
        return "*"

class ICNum(ICNode):
    def __init__(self, val):
        self.val = val
    def __repr__(self):
        return str(self.val)

class ICSuc(ICNode):
    def __init__(self, val):
        self.val = val
    def __repr__(self):
        return f"+{self.val}"

class DeltaCompiler:
    def __init__(self):
        self.dup_counter = 0

    def get_fresh_name(self, base):
        self.dup_counter += 1
        return f"{base}_{self.dup_counter}"

    def analyze_usage(self, ast, env):
        # Returns (AST subset using env, list of free variables used)
        if isinstance(ast, LispNum):
            return [], ast
            
        elif isinstance(ast, LispSymbol):
            if ast.name in env:
                return [ast.name], ast
            return [], ast # Global or primitive
            
        elif isinstance(ast, LispList):
            elems = ast.elements
            if not elems:
                return [], ast
            
            head = elems[0]
            if isinstance(head, LispSymbol) and head.name == 'lambda':
                params_list = elems[1].elements
                params = [p.name for p in params_list]
                body = elems[2]
                
                # Exclude params from inner environment search
                inner_env = env.copy()
                for p in params: 
                    inner_env.discard(p)
                    
                used, _ = self.analyze_usage(body, inner_env)
                return used, ast
            else:
                total_used = []
                for e in elems:
                    used, _ = self.analyze_usage(e, env)
                    total_used.extend(used)
                return total_used, ast
        return [], ast

    def compile(self, ast, env=None, scope_level=0):
        if env is None:
            env = set()

        if isinstance(ast, LispNum):
            return ICNum(ast.value)
            
        elif isinstance(ast, LispSymbol):
            if ast.name == "fst":
                return ICLam("p", ICApp(ICVar("p"), ICLam("x", ICLam("y", ICVar("x")))))
            elif ast.name == "snd":
                return ICLam("p", ICApp(ICVar("p"), ICLam("x", ICLam("y", ICVar("y")))))
            elif ast.name == "cons":
                return ICLam("a", ICLam("b", ICLam("f", ICApp(ICApp(ICVar("f"), ICVar("a")), ICVar("b")))))
            elif ast.name == "suc":
                return ICVar("suc")
                
            return ICVar(ast.name)
            
        elif isinstance(ast, LispList):
            elems = ast.elements
            
            # Application
            head = elems[0]
            if isinstance(head, LispSymbol) and head.name == 'lambda':
                params_list = elems[1].elements
                params = [p.name for p in params_list]
                body = elems[2]
                return self.compile_lambda(params, body, env, scope_level)
                
            elif isinstance(head, LispSymbol) and head.name == 'suc':
                return ICSuc(self.compile(elems[1], env, scope_level))
                
            else:
                # Function Application
                comp_head = self.compile(head, env, scope_level)
                for arg in elems[1:]:
                    comp_arg = self.compile(arg, env, scope_level)
                    comp_head = ICApp(comp_head, comp_arg)
                    
                return comp_head

    def compile_lambda(self, params, body, env, scope_level):
        if not params:
            return self.compile(body, env, scope_level)
            
        param = params[0]
        rest_params = params[1:]
        
        inner_env = env.copy()
        inner_env.add(param)
        
        usages, _ = self.analyze_usage(body, set([param]))
        count = usages.count(param)
        
        inner_body_compiled = self.compile_lambda(rest_params, body, inner_env, scope_level + 1)
        
        if count == 0:
            return ICLam(param, inner_body_compiled) 
            
        elif count == 1:
            return ICLam(param, inner_body_compiled)
            
        else:
            # Duplicate N times using Delta-Nets REPLICATORS!
            fresh_vars = [self.get_fresh_name(param) for _ in range(count)]
            rewritten_body_ast = self.rewrite_var(body, param, fresh_vars.copy())
            
            final_body = self.compile_lambda(rest_params, rewritten_body_ast, inner_env, scope_level + 1)
            
            current_var = param
            dups = []
            for i in range(count - 1):
                v0 = fresh_vars[i]
                if i == count - 2:
                    v1 = fresh_vars[i+1]
                else:
                    v1 = self.get_fresh_name(f"{param}_rem")
                    
                # In Delta-Nets, duplicators are encoded with Levels. We can use the lexical scope 
                # depth as the base level, and delta=0 for standard linear branching!
                level = scope_level
                dups.append((v0, v1, current_var, level))
                self.dup_counter += 1
                current_var = v1
                
            for v0, v1, val, level in reversed(dups):
                # Using ICRep instead of ICDup!
                final_body = ICRep(v0, v1, ICVar(val), final_body, level, 0, 0)
                
            return ICLam(param, final_body)

    def rewrite_var(self, ast, target, fresh_vars):
        if isinstance(ast, LispNum):
            return ast
        elif isinstance(ast, LispSymbol):
            if ast.name == target:
                if not fresh_vars:
                    return ast
                return LispSymbol(fresh_vars.pop(0))
            return ast
        elif isinstance(ast, LispList):
            elems = ast.elements
            if not elems:
                return ast
            head = elems[0]
            if isinstance(head, LispSymbol) and head.name == 'lambda':
                params = [p.name for p in elems[1].elements]
                if target in params:
                    return ast
                    
            new_elems = []
            for e in elems:
                new_elems.append(self.rewrite_var(e, target, fresh_vars))
            return LispList(new_elems)

if __name__ == "__main__":
    c = DeltaCompiler()
    # test duplicate lambda
    ast = parse_lisp("(lambda (x) (x x))")[0]
    out = c.compile(ast)
    print("Standard Combinator (lambda (x) (x x)):")
    print(out)
    
    print("\nTriple duplicate (lambda (f x) (f x x x)):")
    ast2 = parse_lisp("(lambda (f x) (f x x x))")[0]
    out2 = c.compile(ast2)
    print(out2)
