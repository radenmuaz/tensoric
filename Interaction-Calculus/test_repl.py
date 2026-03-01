from lisp_compiler import Compiler, LispList, LispSymbol
from lisp_parser import parse_lisp
import lisp_compiler
from lisp_to_ic import ic_to_string, ICSwi

lisp_compiler.ICSwi = ICSwi

c = Compiler()
lisp = "(lambda (mul m n) (match-num m 0 (lambda (pred) (add n (mul pred n)))))"
ast = parse_lisp(lisp)[0]
print(ic_to_string(c.compile(ast, set())))
