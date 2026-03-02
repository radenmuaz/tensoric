import sys
from staticic import *

MAX_NAME_LEN = 64
NONE = 0xFFFFFFFF

class Binder:
    def __init__(self, name):
        self.name = name
        self.var = NONE
        self.loc = NONE

class Parser:
    def __init__(self, ic, input_str):
        self.ic = ic
        self.input = input_str
        self.pos = 0
        self.line = 1
        self.col = 1
        
        self.global_vars = []
        self.lexical_vars = []

    def next_char(self):
        if self.pos >= len(self.input):
            return '\0'
        c = self.input[self.pos]
        self.pos += 1
        if c == '\n':
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        return c

    def peek_char(self):
        if self.pos >= len(self.input):
            return '\0'
        return self.input[self.pos]

    def peek_is(self, c):
        return self.peek_char() == c

    def skip(self):
        while True:
            c = self.peek_char()
            if c.isspace():
                self.next_char()
            elif c == '/' and self.pos + 1 < len(self.input) and self.input[self.pos+1] == '/':
                self.next_char()
                self.next_char()
                while self.peek_char() != '\0' and self.peek_char() != '\n':
                    self.next_char()
                if self.peek_char() == '\n':
                    self.next_char()
            else:
                break

    def consume(self, token):
        self.skip()
        if self.input[self.pos:self.pos+len(token)] == token:
            for _ in range(len(token)):
                self.next_char()
            return True
        return False

    def expect(self, token, context):
        if not self.consume(token):
            self.parse_error(f"Expected '{token}' {context}")

    def parse_error(self, message):
        print(f"Parse error at line {self.line}, column {self.col}: {message}", file=sys.stderr)
        print(f"Input:", file=sys.stderr)
        lines = self.input.split('\n')
        err_line = lines[self.line-1] if self.line-1 < len(lines) else ""
        print(err_line, file=sys.stderr)
        print(" " * (self.col-1) + "^", file=sys.stderr)
        sys.exit(1)

    def starts_with_dollar(self, name):
        return name and name[0] == '$'

    def find_or_add_global_var(self, name):
        for i, b in enumerate(self.global_vars):
            if b.name == name:
                return i
        idx = len(self.global_vars)
        self.global_vars.append(Binder(name))
        return idx
        
    def push_lexical_binder(self, name, term):
        b = Binder(name)
        b.var = term
        self.lexical_vars.append(b)

    def pop_lexical_binder(self):
        if self.lexical_vars:
            self.lexical_vars.pop()

    def find_lexical_binder(self, name):
        for b in reversed(self.lexical_vars):
            if b.name == name:
                return b
        return None

    def store_term(self, loc, tag, value):
        self.ic.heap[loc] = self.ic.make_term(False, tag, value)

    def parse_name(self):
        name = ""
        c = self.peek_char()
        if not (c.isalpha() or c in '_$'):
            self.parse_error(f"Expected name starting with letter, _, or $, got {repr(c)}")
        while self.peek_char().isalnum() or self.peek_char() in '_$':
            name += self.next_char()
        return name

    def parse_uint(self):
        val = 0
        has_digit = False
        while self.peek_char().isdigit():
            val = val * 10 + int(self.next_char())
            has_digit = True
        if not has_digit:
            self.parse_error("Expected digit")
        return val
        
    def parse_term_var(self, loc):
        name = self.parse_name()
        if self.starts_with_dollar(name):
            idx = self.find_or_add_global_var(name)
            if self.global_vars[idx].var == NONE:
                self.global_vars[idx].loc = loc
            else:
                self.ic.heap[loc] = self.global_vars[idx].var
        else:
            binder = self.find_lexical_binder(name)
            if binder is None:
                self.parse_error(f"Undefined lexical variable: {name}")
            
            if binder.loc == NONE:
                self.ic.heap[loc] = binder.var
                binder.loc = loc
            else:
                dup_loc = self.ic.alloc(1)
                self.ic.heap[dup_loc] = self.ic.heap[binder.loc]
                dp0 = self.ic.ic_make_co0(0, dup_loc)
                dp1 = self.ic.ic_make_co1(0, dup_loc)
                self.ic.heap[binder.loc] = dp0
                self.ic.heap[loc] = dp1
                binder.loc = loc

    def parse_term_lam(self, loc):
        c = self.peek_char()
        if c == "λ" or c == "\u03BB":
            self.next_char()
        else:
            self.parse_error("Expected 'λ' for lambda")
        name = self.parse_name()
        self.expect(".", "after name in lambda")
        
        lam_node = self.ic.alloc(1)
        var_term = self.ic.make_term(False, VAR, lam_node)
        
        if self.starts_with_dollar(name):
            idx = self.find_or_add_global_var(name)
            if self.global_vars[idx].var != NONE:
                self.parse_error(f"Duplicate global variable binder: {name}")
            self.global_vars[idx].var = var_term
        else:
            self.push_lexical_binder(name, var_term)
            
        self.parse_term(lam_node)
        
        if not self.starts_with_dollar(name):
            self.pop_lexical_binder()
            
        self.store_term(loc, LAM, lam_node)

    def parse_term_app(self, loc):
        self.expect("(", "for application")
        self.parse_term(loc)
        self.skip()
        while self.peek_char() != ')':
            app_node = self.ic.alloc(2)
            self.move_term(loc, app_node + 0)
            self.parse_term(app_node + 1)
            self.store_term(loc, APP, app_node)
            self.skip()
        self.expect(")", "after terms in application")

    def parse_term_sup(self, loc):
        self.expect("&", "for superposition")
        label = self.parse_uint() & 7
        self.expect("{", "after label in superposition")
        sup_node = self.ic.alloc(2)
        self.parse_term(sup_node + 0)
        self.expect(",", "between terms in superposition")
        self.parse_term(sup_node + 1)
        self.expect("}", "after terms in superposition")
        self.ic.heap[loc] = self.ic.ic_make_sup(label, sup_node)

    def parse_term_dup(self, loc):
        self.expect("!&", "for duplication")
        label = self.parse_uint() & 7
        self.expect("{", "after label in duplication")
        x0 = self.parse_name()
        self.expect(",", "between names in duplication")
        x1 = self.parse_name()
        self.expect("}", "after names in duplication")
        self.expect("=", "after names in duplication")
        
        dup_node = self.ic.alloc(1)
        self.parse_term(dup_node)
        self.expect(";", "after value in duplication")
        
        co0_term = self.ic.ic_make_co0(label, dup_node)
        co1_term = self.ic.ic_make_co1(label, dup_node)
        
        if self.starts_with_dollar(x0):
            idx = self.find_or_add_global_var(x0)
            if self.global_vars[idx].var != NONE:
                self.parse_error(f"Duplicate global binder: {x0}")
            self.global_vars[idx].var = co0_term
        else:
            self.push_lexical_binder(x0, co0_term)
            
        if self.starts_with_dollar(x1):
            idx = self.find_or_add_global_var(x1)
            if self.global_vars[idx].var != NONE:
                self.parse_error(f"Duplicate global binder: {x1}")
            self.global_vars[idx].var = co1_term
        else:
            self.push_lexical_binder(x1, co1_term)
            
        self.parse_term(loc)
        
        if not self.starts_with_dollar(x1):
            self.pop_lexical_binder()
        if not self.starts_with_dollar(x0):
            self.pop_lexical_binder()

    def parse_term_era(self, loc):
        self.expect("*", "for erasure")
        self.store_term(loc, ERA, 0)

    def parse_term_num(self, loc):
        val = self.parse_uint()
        self.store_term(loc, NUM, val)

    def parse_term_suc(self, loc):
        self.expect("+", "for successor")
        suc_node = self.ic.alloc(1)
        self.parse_term(suc_node)
        self.store_term(loc, SUC, suc_node)

    def parse_term_swi(self, loc):
        self.expect("?", "for switch")
        swi_node = self.ic.alloc(3)
        self.parse_term(swi_node)
        self.expect("{", "after condition in switch")
        self.expect("0", "for zero case")
        self.expect(":", "after '0'")
        self.parse_term(swi_node + 1)
        self.expect(";", "after zero case")
        self.expect("+", "for successor case")
        self.expect(":", "after '+'")
        self.parse_term(swi_node + 2)
        self.expect(";", "after successor case")
        self.expect("}", "to close switch")
        self.store_term(loc, SWI, swi_node)

    def parse_term_let(self, loc):
        self.expect("!", "for let expression")
        name = self.parse_name()
        self.expect("=", "after name in let")
        app_node = self.ic.alloc(2)
        lam_node = self.ic.alloc(1)
        
        self.parse_term(app_node + 1)
        self.expect(";", "after value in let")
        
        var_term = self.ic.make_term(False, VAR, lam_node)
        if self.starts_with_dollar(name):
            idx = self.find_or_add_global_var(name)
            if self.global_vars[idx].var != NONE:
                self.parse_error(f"Duplicate global binder: {name}")
            self.global_vars[idx].var = var_term
        else:
            self.push_lexical_binder(name, var_term)
            
        self.parse_term(lam_node)
        
        if not self.starts_with_dollar(name):
            self.pop_lexical_binder()
            
        self.store_term(app_node + 0, LAM, lam_node)
        self.store_term(loc, APP, app_node)

    def move_term(self, from_loc, to_loc):
        for b in self.global_vars:
            if b.loc == from_loc:
                b.loc = to_loc
        for b in self.lexical_vars:
            if b.loc == from_loc:
                b.loc = to_loc
        self.ic.heap[to_loc] = self.ic.heap[from_loc]

    def parse_term(self, loc):
        self.skip()
        if self.pos >= len(self.input):
            self.parse_error("Unexpected end of input")
            
        c = self.peek_char()
        if c == 'λ' or c == '\u03BB':
            self.parse_term_lam(loc)
        elif c.isalpha() or c in '_$':
            self.parse_term_var(loc)
        elif c.isdigit():
            self.parse_term_num(loc)
        elif c == '!':
            self.next_char()
            next_c = self.peek_char()
            self.pos -= 1
            if next_c == '&':
                self.parse_term_dup(loc)
            elif next_c.isalpha() or next_c in '_$':
                self.parse_term_let(loc)
            else:
                self.parse_error("Expected '&' or name after '!'")
        elif c == '&':
            self.parse_term_sup(loc)
        elif c == 'λ' or c == '\u03BB':
            self.parse_term_lam(loc)
        elif c == '(':
            self.parse_term_app(loc)
        elif c == '*':
            self.parse_term_era(loc)
        elif c == '+':
            self.parse_term_suc(loc)
        elif c == '?':
            self.parse_term_swi(loc)
        else:
            self.parse_error(f"Unexpected character: {repr(c)}")

    def parse_term_alloc(self):
        loc = self.ic.alloc(1)
        self.parse_term(loc)
        return loc

    def resolve_global_vars(self):
        for b in self.global_vars:
            if b.var == NONE:
                self.parse_error(f"Undefined global variable: {b.name}")
            if b.loc != NONE:
                self.ic.heap[b.loc] = b.var

def parse_string(ic, input_str):
    p = Parser(ic, input_str)
    p.skip()
    term_loc = p.parse_term_alloc()
    p.resolve_global_vars()
    return p.ic.heap[term_loc]

def parse_file(ic, filename):
    with open(filename, 'r') as f:
        return parse_string(ic, f.read())
