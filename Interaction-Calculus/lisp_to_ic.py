import sys
from lisp_parser import parse_lisp
from lisp_compiler import Compiler, ICNode, ICVar, ICLam, ICApp, ICDup, ICEra, ICNum, ICSuc

# We need to map our IC AST classes onto the SWI logic
class ICSwi(ICNode):
    def __init__(self, cond, z_branch, s_branch):
        self.cond = cond
        self.z_branch = z_branch
        self.s_branch = s_branch
    def __repr__(self):
        return f"?{self.cond}{{0:{self.z_branch};+:{self.s_branch};}}"

def ic_to_string(node):
    if isinstance(node, ICVar):
        return node.name
    elif isinstance(node, ICLam):
        return f"λ{node.param}.{ic_to_string(node.body)}"
    elif isinstance(node, ICApp):
        return f"({ic_to_string(node.fun)} {ic_to_string(node.arg)})"
    elif isinstance(node, ICDup):
        return f"!&{node.lab}{{{node.var0},{node.var1}}} = {ic_to_string(node.val)};\n{ic_to_string(node.body)}"
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
    compiler = Compiler()
    # Actually SWI class needs to be imported or patched into scope
    # So we patch it in
    import lisp_compiler
    lisp_compiler.ICSwi = ICSwi
    
    asts = parse_lisp(lisp_source)
    if not asts:
        return ""
        
    compiled_asts = []
    for ast in asts:
        c_ast = compiler.compile(ast)
        compiled_asts.append(ic_to_string(c_ast))
        
    return "\n".join(compiled_asts)

if __name__ == "__main__":
    src = "(lambda (x) (if x (lambda (y) x) 0))"
    print("--- Lisp ---")
    print(src)
    print("\n--- Compile to IC ---")
    print(compile_lisp_to_ic(src))
