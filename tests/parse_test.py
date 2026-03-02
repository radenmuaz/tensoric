import sys
import time
from tensoric.parser import parse_file
from tensoric.staticic import StaticIC
from tensoric.show import print_term
ic = StaticIC()
start = time.time()
term = parse_file(ic, sys.argv[1])
term = ic.ic_normal(term)
end = time.time()
print(print_term(ic, term))
print(f"WORK: {ic.interactions} interactions")
print(f"TIME: {end-start:.7f} seconds")
print(f"SIZE: {ic.heap_pos} nodes")
