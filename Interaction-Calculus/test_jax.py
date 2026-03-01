import sys
import time
from parser import parse_file
from jax_evaluator import JaxIC
from show import print_term

ic = JaxIC()
term = parse_file(ic, "examples/test_4.ic")

print("Initial Term:")
print(print_term(ic, term))

start = time.time()
while True:
    prev_interactions = ic.interactions
    # Scan 10 steps at a time directly on the GPU
    if not ic.run_scan(steps=10):
        break
    if ic.interactions == prev_interactions:
        print("Graph contains unsupported Vectorized Redexes. Halting.")
        break
end = time.time()

print("Final Term:")
print(print_term(ic, term))
print(f"WORK: {ic.interactions} interactions")
print(f"TIME: {end-start:.7f} seconds")
