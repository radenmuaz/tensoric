from .lisp_parser import LispSymbol, LispNum, LispList
from ..lc.ast import LCVar, LCLam, LCApp, LCNum, LCSuc, LCSwi

class Compiler:
    def __init__(self):
        pass

    def compile(self, ast):
        if isinstance(ast, LispNum):
            return LCNum(ast.value)
            
        elif isinstance(ast, LispSymbol):
            # Desugaring primitives into pure LC
            if ast.name == "fst":
                # fst = \p. (p \x y. x)
                return LCLam("p", LCApp(LCVar("p"), LCLam("x", LCLam("y", LCVar("x")))))
            elif ast.name == "snd":
                return LCLam("p", LCApp(LCVar("p"), LCLam("x", LCLam("y", LCVar("y")))))
            elif ast.name == "cons":
                # cons = \a b f. (f a b)
                return LCLam("a", LCLam("b", LCLam("f", LCApp(LCApp(LCVar("f"), LCVar("a")), LCVar("b")))))
            elif ast.name == "suc":
                return LCVar("suc") # Placeholder to be caught by Application
            return LCVar(ast.name)
            
        elif isinstance(ast, LispList):
            elems = ast.elements
            if not elems:
                return None
                
            head = elems[0]
            
            # Lambda: (lambda (x y) body)
            if isinstance(head, LispSymbol) and head.name == 'lambda':
                params_list = elems[1].elements
                params = [p.name for p in params_list]
                body = elems[2]
                return self.compile_lambda(params, body)
                
            # Successor: (suc n)
            elif isinstance(head, LispSymbol) and head.name == 'suc':
                return LCSuc(self.compile(elems[1]))
                
            # Match-Num: (match-num cond z_branch s_branch)
            elif isinstance(head, LispSymbol) and head.name == 'match-num':
                cond = self.compile(elems[1])
                z_branch = self.compile(elems[2])
                s_branch = self.compile(elems[3])
                return LCSwi(cond, z_branch, s_branch)
                
            else:
                # Function Application: (f a b c) -> (((f a) b) c)
                comp_head = self.compile(head)
                for arg in elems[1:]:
                    comp_arg = self.compile(arg)
                    comp_head = LCApp(comp_head, comp_arg)
                return comp_head
        return None

    def compile_lambda(self, params, body):
        if not params:
            return self.compile(body)
            
        param = params[0]
        rest_params = params[1:]
        
        # We just generate the LCLam here. 
        # The LC -> IC compiler will handle the complex DUP/ERA logic.
        return LCLam(param, self.compile_lambda(rest_params, body))
