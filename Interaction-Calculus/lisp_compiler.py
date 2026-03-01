from lisp_parser import LispSymbol, LispNum, LispList, parse_lisp

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

class ICDup(ICNode):
    def __init__(self, var0, var1, val, body, lab=0):
        self.var0 = var0
        self.var1 = var1
        self.val = val
        self.body = body
        self.lab = lab
    def __repr__(self):
        return f"! &{self.lab}{{{self.var0},{self.var1}}} = {self.val};\n{self.body}"

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

class Compiler:
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
                # (lambda (x y) body)
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

    def compile(self, ast, env=None):
        if env is None:
            env = set()

        if isinstance(ast, LispNum):
            return ICNum(ast.value)
            
        elif isinstance(ast, LispSymbol):
            if ast.name == "fst":
                # Scott encoding: fst = \p. (p \x y. x)
                return ICLam("p", ICApp(ICVar("p"), ICLam("x", ICLam("y", ICVar("x")))))
            elif ast.name == "snd":
                return ICLam("p", ICApp(ICVar("p"), ICLam("x", ICLam("y", ICVar("y")))))
            elif ast.name == "cons":
                # cons = \a b f. (f a b)
                return ICLam("a", ICLam("b", ICLam("f", ICApp(ICApp(ICVar("f"), ICVar("a")), ICVar("b")))))
                
            elif ast.name == "suc":
                # We expect (suc number) but as a symbol compilation it doesn't do much unless applied
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
                return self.compile_lambda(params, body, env)
                
            elif isinstance(head, LispSymbol) and head.name == 'suc':
                return ICSuc(self.compile(elems[1], env))
                
            elif isinstance(head, LispSymbol) and head.name == 'match-num':
                # (match-num cond z_branch s_branch)
                cond = self.compile(elems[1], env)
                z_branch = self.compile(elems[2], env)
                s_branch = self.compile(elems[3], env)
                return ICSwi(cond, z_branch, s_branch)
                
            else:
                # Function Application (f a b c) -> (((f a) b) c)
                comp_head = self.compile(head, env)
                for arg in elems[1:]:
                    comp_arg = self.compile(arg, env)
                    comp_head = ICApp(comp_head, comp_arg)
                    
                return comp_head

    def compile_lambda(self, params, body, env):
        if not params:
            return self.compile(body, env)
            
        param = params[0]
        rest_params = params[1:]
        
        # Analyze usages of `param` in the body
        inner_env = env.copy()
        inner_env.add(param)
        
        # The body compilation needs to know if x is duplicated or erased.
        usages, _ = self.analyze_usage(body, set([param]))
        count = usages.count(param)
        
        inner_body_compiled = self.compile_lambda(rest_params, body, inner_env)
        
        if count == 0:
            # unused -> ERA
            # Create a dummy variable for the lambda, and erase it
            return ICLam(param, ICDup(param, "dummy", ICEra(), inner_body_compiled)) # Hacky ERA binding?
            # Actually, Interaction Calculus allows λx.body where x doesn't appear in body implicitly as an erase
            # We must output `λx.body` but we just don't place `x` inside `body`.
            # Wait, standard syntax for unused is `λ*.body` or `λx.body` where x is missing.
            # In our parser, IC `parser.c` accepts `λ*.body` if unused? Let's assume standard `λx.body` dropping `x`.
            return ICLam(param, inner_body_compiled) 
            
        elif count == 1:
            # Linear usage
            return ICLam(param, inner_body_compiled)
            
        else:
            # Duplicate N times
            # E.g. x used 3 times -> ! &0{x_1, x_rem} = x; ! &0{x_2, x_3} = x_rem;
            
            # We must rewrite the inner AST to use x_1, x_2, x_3... instead of x.
            fresh_vars = [self.get_fresh_name(param) for _ in range(count)]
            rewritten_body_ast = self.rewrite_var(body, param, fresh_vars.copy())
            
            # Compile the rewriten body
            final_body = self.compile_lambda(rest_params, rewritten_body_ast, inner_env)
            
            # We want the outermost ICDup (which binds from param) to be the root.
            # So we build a list of dups and wrap them backwards.
            current_var = param
            dups = []
            for i in range(count - 1):
                v0 = fresh_vars[i]
                if i == count - 2:
                    v1 = fresh_vars[i+1] # last one
                else:
                    v1 = self.get_fresh_name(f"{param}_rem")
                    
                lab = self.dup_counter % 8
                # self.dup_counter += 1 # We bump this globally but wait, get_fresh_name bumps it!
                # If we just use self.dup_counter, every dup gets a different label
                # Actually, in IC multiple sequential identical duplications can share the same label if they don't interleave directly. Let's just rotate.
                dups.append((v0, v1, current_var, lab))
                self.dup_counter += 1
                current_var = v1
                
            for v0, v1, val, lab in reversed(dups):
                final_body = ICDup(v0, v1, ICVar(val), final_body, lab)
                
            return ICLam(param, final_body)

    def rewrite_var(self, ast, target, fresh_vars):
        if isinstance(ast, LispNum):
            return ast
        elif isinstance(ast, LispSymbol):
            if ast.name == target:
                # We mutatively pop from the list so the same list is updated across deep recursions
                if not fresh_vars:
                    print(f"DEBUG: Ran out of fresh vars for {target} renaming!")
                    return ast
                return LispSymbol(fresh_vars.pop(0))
            return ast
        elif isinstance(ast, LispList):
            elems = ast.elements
            if not elems:
                return ast
            head = elems[0]
            if isinstance(head, LispSymbol) and head.name == 'lambda':
                # Check for shadowing
                params = [p.name for p in elems[1].elements]
                if target in params:
                    # Target is shadowed here, DO NOT traverse body
                    return ast
                    
            new_elems = []
            for e in elems:
                new_elems.append(self.rewrite_var(e, target, fresh_vars))
            return LispList(new_elems)

if __name__ == "__main__":
    c = Compiler()
    # test duplicate lambda
    ast = parse_lisp("(lambda (x) (x x))")[0]
    out = c.compile(ast)
    print(out)
    
    # test triple duplicate
    ast2 = parse_lisp("(lambda (f x) (f x x x))")[0]
    out2 = c.compile(ast2)
    print(out2)
