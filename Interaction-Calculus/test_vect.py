from parser import parse_file
from vectorized import VectorizedIC
from show import print_term
ic = VectorizedIC()
term = parse_file(ic, "examples/test_0.ic")
ic.find_all_redexes(term)
ic.step_vectorized()
print(f"Redexes simulated: {ic.redex_count}")
