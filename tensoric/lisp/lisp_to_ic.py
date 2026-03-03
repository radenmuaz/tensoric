from .lisp_parser import parse_lisp
from .lisp_compiler import Compiler as LispCompiler
from ..lc.compiler import LCCompiler
from ..base.ic_ast import ICVar, ICLam, ICApp, ICDup, ICRep, ICEra, ICNum, ICSuc, ICSwi

def ic_to_string(node):
    if isinstance(node, ICVar):
        return node.name
    elif isinstance(node, ICLam):
        return f"λ{node.param}.{ic_to_string(node.body)}"
    elif isinstance(node, ICApp):
        return f"({ic_to_string(node.fun)} {ic_to_string(node.arg)})"
    elif isinstance(node, ICDup):
        return f"!&{node.lab}{{{node.var0},{node.var1}}} = {ic_to_string(node.val)};\n{ic_to_string(node.body)}"
    elif isinstance(node, ICRep):
        return f"REP[L{node.level} dL{node.delta_l} dR{node.delta_r}]{{{node.var0},{node.var1}}} = {ic_to_string(node.val)};\n{ic_to_string(node.body)}"
    elif isinstance(node, ICEra):
        return "*"
    elif isinstance(node, ICNum):
        return str(node.val)
    elif isinstance(node, ICSuc):
        return f"+{ic_to_string(node.val)}"
    elif isinstance(node, ICSwi):
        return f"?{ic_to_string(node.cond)}{{0:{ic_to_string(node.z_branch)};+:{ic_to_string(node.s_branch)};}}"
    
    return str(node)

def compile_lisp_to_ic(lisp_source):
    lisp_parser = parse_lisp
    lisp_compiler = LispCompiler()
    lc_compiler = LCCompiler()
    
    asts = lisp_parser(lisp_source)
    if not asts:
        return ""
        
    compiled_asts = []
    for ast in asts:
        # Step 1: Lisp -> LC
        lc_ast = lisp_compiler.compile(ast)
        # Step 2: LC -> IC
        ic_ast = lc_compiler.compile(lc_ast)
        compiled_asts.append(ic_to_string(ic_ast))
        
    return "\n".join(compiled_asts)

if __name__ == "__main__":
    src = "(lambda (x) (x x))"
    print("--- Lisp ---")
    print(src)
    print("\n--- Compile to IC (via LC IR) ---")
    print(compile_lisp_to_ic(src))
