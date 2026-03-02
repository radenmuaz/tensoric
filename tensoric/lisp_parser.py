import re

class LispAST:
    pass

class LispSymbol(LispAST):
    def __init__(self, name):
        self.name = name
    def __repr__(self):
        return f"Sym({self.name})"

class LispNum(LispAST):
    def __init__(self, value):
        self.value = value
    def __repr__(self):
        return f"Num({self.value})"

class LispList(LispAST):
    def __init__(self, elements):
        self.elements = elements
    def __repr__(self):
        return f"List({self.elements})"

def tokenize(input_string):
    regex = r"[\s,]*(~@|[\[\]{}()'`~^@]|(?:\\.|[^ \s\[\]{}()'\"`~^@])+)"
    tokens = []
    for match in re.finditer(regex, input_string):
        token = match.group(1)
        if token != "" and not token.isspace():
            tokens.append(token)
    return tokens

def parse_tokens(tokens):
    if not tokens:
        raise SyntaxError("Unexpected EOF")
    token = tokens.pop(0)
    
    if token == '(':
        sublist = []
        while tokens and tokens[0] != ')':
            sublist.append(parse_tokens(tokens))
        if not tokens:
            raise SyntaxError("Unexpected EOF, expected ')'")
        tokens.pop(0) # remove ')'
        return LispList(sublist)
    elif token == ')':
        raise SyntaxError("Unexpected ')'")
    else:
        # Check if number
        try:
            return LispNum(int(token))
        except ValueError:
            return LispSymbol(token)

def parse_lisp(input_string):
    tokens = tokenize(input_string)
    asts = []
    while tokens:
        asts.append(parse_tokens(tokens))
    return asts

if __name__ == "__main__":
    test = "(def add (lambda (x y) (+ x y)))"
    print(parse_lisp(test))
