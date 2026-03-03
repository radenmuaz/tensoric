from ..lisp.lisp_parser import parse_lisp
from ..lisp.lisp_compiler import Compiler as LispCompiler
from ..lc.compiler import DeltaLCCompiler
from ..base.ic_ast import ICVar, ICLam, ICApp, ICRep, ICEra, ICNum, ICSuc

# This file now acts as a specialty bridge for Delta-Nets research.
# It uses the unified Lisp->LC frontend and the DeltaLCCompiler backend.

class DeltaCompiler:
    def __init__(self):
        self.lisp_compiler = LispCompiler()
        self.lc_compiler = DeltaLCCompiler()

    def compile(self, ast):
        # Step 1: Lisp AST -> LC AST (Desugaring)
        lc_ast = self.lisp_compiler.compile(ast)
        
        # Step 2: LC AST -> Delta-IC AST (Duplicator Level Calculation)
        ic_ast = self.lc_compiler.compile(lc_ast)
        
        return ic_ast

def ic_to_string(node):
    if isinstance(node, ICVar):
        return node.name
    elif isinstance(node, ICLam):
        return f"λ{node.param}.({ic_to_string(node.body)})"
    elif isinstance(node, ICApp):
        return f"({ic_to_string(node.fun)} {ic_to_string(node.arg)})"
    elif isinstance(node, ICRep):
        return f"REP[L{node.level} dL{node.delta_l} dR{node.delta_r}]{{{node.var0},{node.var1}}} = {ic_to_string(node.val)};\n{ic_to_string(node.body)}"
    elif isinstance(node, ICEra):
        return "*"
    elif isinstance(node, ICNum):
        return str(node.val)
    elif isinstance(node, ICSuc):
        return f"+{ic_to_string(node.val)}"
    return str(node)

if __name__ == "__main__":
    c = DeltaCompiler()
    # test duplicate lambda
    src = "(lambda (x) (x x))"
    ast = parse_lisp(src)[0]
    out = c.compile(ast)
    print("Standard Combinator (lambda (x) (x x)) [Delta-Nets Edition]:")
    print(ic_to_string(out))
    
    print("\nTriple duplicate (lambda (f x) (f x x x)) [Delta-Nets Edition]:")
    src2 = "(lambda (f x) (f x x x))"
    ast2 = parse_lisp(src2)[0]
    out2 = c.compile(ast2)
    print(ic_to_string(out2))
