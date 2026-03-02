import sys
import time
from parser import parse_file
from jax_evaluator import JaxIC
from show import print_term

def main():
    ic = JaxIC()
    file_path = sys.argv[1] if len(sys.argv) > 1 else "examples/test_4.ic"
    term = parse_file(ic, file_path)

    print("Initial Term:")
    print(print_term(ic, term))

    steps = 10
    for arg in sys.argv:
        if arg.startswith("--steps="):
            steps = int(arg.split("=")[1])

    print(f"Warming up JIT compiler for {steps} steps...")
    import jax, jax.numpy as jnp
    from jax_evaluator import compiled_scan, JAX_MAX_NODES
    import time
    start_jit = time.time()
    dummy_state = (jnp.zeros(10, dtype=jnp.uint32), jnp.uint32(0)) # JAX shape polymorphism caches on static shape. Wait, JAX_MAX_NODES is static. 
    # We MUST use the exact shape JAX_MAX_NODES otherwise it compiles twice!
    dummy_state = (jnp.zeros(JAX_MAX_NODES, dtype=jnp.uint32), jnp.uint32(0))
    warmup, _ = compiled_scan(dummy_state, steps)
    warmup[0].block_until_ready()
    jit_time = time.time() - start_jit
    print(f"JIT Compilation Time: {jit_time:.5f}s")

    start = time.time()
    while True:
        prev_interactions = ic.interactions
        # Scan N steps at a time directly on the GPU
        _t0 = time.time()
        has_inters = ic.run_scan(steps=steps)
        _t1 = time.time() - _t0
        print(f"  [Scan {steps} steps execution: {_t1:.4f}s]")
        
        if not has_inters:
            break
        if ic.interactions == prev_interactions:
            print("Graph contains unsupported Vectorized Redexes. Halting.")
            break
    end = time.time()

    print("Final Term:")
    print(print_term(ic, term))
    print(f"WORK: {ic.interactions} interactions")
    print(f"TIME: {end-start:.7f} seconds")

if __name__ == "__main__":
    main()
