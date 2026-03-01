import sys
import re
from lisp_to_ic import compile_lisp_to_ic
from lisp_parser import parse_lisp, LispSymbol, LispList
from lisp_compiler import Compiler
from parser import parse_string
from jax_evaluator import JaxIC
from show import print_term

def decode_ic(ic_str):
    """
    Decodes the raw IC term back into Lisp values.
    Supports numbers (e.g. `42`), Scott-encoded lists, and booleans.
    It's rudimentary for the MVP.
    """
    ic_str = ic_str.strip()
    
    # 1. Direct number check
    if ic_str.isdigit():
        return int(ic_str)
        
    # 2. Number + Successor checks e.g. `+(+(0))` -> technically IC parses SUC as `+0`
    # Let's see if it's a SUC chain natively:
    if ic_str.startswith("+"):
        # The IC stringifier outputs `+x`, we can just count `+` signs if it ends in a number
        # Actually our show.py outputs `+` and recursively calls, so `++0`
        count = 0
        while ic_str.startswith("+"):
            count += 1
            ic_str = ic_str[1:]
        if ic_str.isdigit():
            return int(ic_str) + count
            
    # For Lists, Scott encoding is `\p. (p a b)` -> `λp.((p a) b)` in our stringifier
    # We can use simple regex to decode pair tuples into python lists conceptually
    pair_match = re.match(r"^λ([a-zA-Z0-9_]+)\.\(\(\1\s+(.*)\)\s+(.*)\)$", ic_str)
    if pair_match:
        fst_raw = pair_match.group(2)
        snd_raw = pair_match.group(3)
        return f"({decode_ic(fst_raw)} . {decode_ic(snd_raw)})"
        
    # Booleans (Church) 
    # true  = \t.\f.t -> λa.λb.a
    # false = \t.\f.f -> λa.λb.b
    if re.match(r"^λ([a-zA-Z0-9_]+)\.λ([a-zA-Z0-9_]+)\.\1$", ic_str):
        return "#t"
    if re.match(r"^λ([a-zA-Z0-9_]+)\.λ([a-zA-Z0-9_]+)\.\2$", ic_str):
        return "#f"
        
    # Otherwise return raw IC AST string
    return f"<IC-closure: {ic_str}>"

def run_repl():
    print("Welcome to Lisp-on-TPU (Interaction Calculus Backend)")
    print("Type 'exit' or Ctrl+C to quit.")
    
    env_defs = {
        "true": parse_lisp("(lambda (t f) t)")[0],
        "false": parse_lisp("(lambda (t f) f)")[0],
        "not": parse_lisp("(lambda (b) (b false true))")[0],
        "and": parse_lisp("(lambda (a b) (a b false))")[0],
        "or": parse_lisp("(lambda (a b) (a true b))")[0],
        "cons": parse_lisp("(lambda (x y f) (f x y))")[0],
        "fst": parse_lisp("(lambda (p) (p (lambda (x y) x)))")[0],
        "snd": parse_lisp("(lambda (p) (p (lambda (x y) y)))")[0],
        # We leave + as a native ICSwi operation if handled in compiler
    }
    
    while True:
        try:
            line = input("\nLisp> ")
            if not line.strip():
                continue
            if line.strip() == "exit":
                break
                
            # Quick hack to extract basic def statements from root node level
            # In a real Lisp parser, defs define globally persistent environments
            asts = parse_lisp(line)
            if not asts: continue
            
            root = asts[0]
            if root.elements and isinstance(root.elements[0], LispSymbol) and root.elements[0].name == 'def':
                # (def name body) -> store it 
                name = root.elements[1].name
                body = root.elements[2]
                env_defs[name] = body
                print(f"Defined {name}")
                continue
            
            # If it's not a def, we compile it inline using a 'let' bound environment of all current defs.
            # Convert dict into explicit substitutions: (\name1.\name2... body def2 def1)
            import copy
            
            composed_ast = root
            for name, body_ast in reversed(env_defs.items()):
                # lambda wrap
                composed_ast = LispList([
                    LispList([LispSymbol("lambda"), LispList([LispSymbol(name)]), composed_ast]),
                    copy.deepcopy(body_ast)
                ])
                
            # Compile into IC graph script
            compiler = Compiler()
            import lisp_compiler
            from lisp_to_ic import ICSwi, ic_to_string
            lisp_compiler.ICSwi = ICSwi
            c_ast = compiler.compile(composed_ast)
            ic_source = ic_to_string(c_ast)
            
            ic = JaxIC()
            term = parse_string(ic, ic_source)
            
            term = ic.ic_normal(term)
                
            raw_out = print_term(ic, term)
            # print("Raw IC Output:", raw_out)
            
            lisp_out = decode_ic(raw_out)
            print("=>", lisp_out)
            print(f"[{ic.interactions} graph interactions evaluated]")
            
        except EOFError:
            break
        except Exception as e:
            print(f"Error: {e}")

def run_file(filename):
    with open(filename, 'r') as f:
        # We split simple lisp expressions by assuming they are top-level parens.
        # But a simple hack for our parser MVP is just parsing the whole file and compiling it
        content = f.read()
    
    content = re.sub(r";.*", "", content)
    asts = parse_lisp(content)
    env_defs = {
        "true": parse_lisp("(lambda (t f) t)")[0],
        "false": parse_lisp("(lambda (t f) f)")[0],
        "not": parse_lisp("(lambda (b) (b false true))")[0],
        "and": parse_lisp("(lambda (a b) (a b false))")[0],
        "or": parse_lisp("(lambda (a b) (a true b))")[0],
        "cons": parse_lisp("(lambda (x y f) (f x y))")[0],
        "fst": parse_lisp("(lambda (p) (p (lambda (x y) x)))")[0],
        "snd": parse_lisp("(lambda (p) (p (lambda (x y) y)))")[0],
        "if": parse_lisp("(lambda (c t f) (c t f))")[0],
        "Z": parse_lisp("(lambda (f) ((lambda (x) (f (lambda (v) ((x x) v)))) (lambda (x) (f (lambda (v) ((x x) v))))))")[0],
    }

    last_out = ""
    for root in asts:
        if isinstance(root, LispList) and root.elements and isinstance(root.elements[0], LispSymbol) and root.elements[0].name == 'def':
            name = root.elements[1].name
            body = root.elements[2]
            env_defs[name] = body
            continue
            
        import copy
        composed_ast = root
        for name, body_ast in reversed(env_defs.items()):
            composed_ast = LispList([
                LispList([LispSymbol("lambda"), LispList([LispSymbol(name)]), composed_ast]),
                copy.deepcopy(body_ast)
            ])
            
        compiler = Compiler()
        import lisp_compiler
        from lisp_to_ic import ICSwi, ic_to_string
        lisp_compiler.ICSwi = ICSwi
        c_ast = compiler.compile(composed_ast)
        ic_source = ic_to_string(c_ast)
        
        ic = JaxIC()
        term = parse_string(ic, ic_source)
        term = ic.ic_normal(term)
        raw_out = print_term(ic, term)
        last_out = decode_ic(raw_out)
        print("=>", last_out)
        print(f"[{ic.interactions} graph interactions evaluated]")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        run_file(sys.argv[1])
    else:
        run_repl()
