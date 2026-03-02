from tensoric.lisp_compiler import Compiler, analyze_usage, LispList, LispSymbol
from tensoric.lisp_parser import parse_lisp
from tensoric.lisp_to_ic import ic_to_string

c = Compiler()
# Define a function containing a free variable `false`
func = parse_lisp("(lambda (a b) (a b false))")[0]

# Wrap the usage
code = "((lambda (false) (and true false)) (lambda (t f) f))"
root = parse_lisp(code)[0]

# Compose them manually like REPL does
composed = LispList([
    LispList([LispSymbol("lambda"), LispList([LispSymbol("and")]), root]),
    func
])

print(ic_to_string(c.compile(composed, set())))
