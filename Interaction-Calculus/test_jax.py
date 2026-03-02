import sys
import time
from parser import parse_file
from jax_evaluator import JaxIC
from show import print_term

ic = JaxIC()
file_path = sys.argv[1] if len(sys.argv) > 1 else "examples/test_4.ic"
term = parse_file(ic, file_path)

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
