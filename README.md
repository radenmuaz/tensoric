# tensoric
hash Interaction-Calculus e9f0185

## Interaction Calculus: NumPy & JAX Vectorization Port 

This project is an experimental port of the original [Interaction Calculus](https://github.com/Interaction-Calculus/Interaction-Calculus) C-based evaluator. The primary goal is to prove that its asynchronous, pointer-jumping graph operations can be flattened into static-shape tensor arrays. This unlocks compiling Turing-complete programs explicitly natively onto XLA/TPU hardware.

### Key Milestones
Instead of resolving graph interaction edges through runtime dynamic heap allocations or linked lists, we proved the model works effectively as a strictly static array processor sequence.

1. **`StaticIC`**: The evaluator (`staticic.py`) successfully processes 100% of the C reference examples returning perfectly aligned graph structural sizes and output normalization strings.
   - Pointers replaced by `uint32` indices referencing a fixed-bound 1D `ndarray` heap.
   - Operations strictly constrained to `get_val()` and `get_tag()`.

2. **Parallel Redex Dispatch** (`VectorizedIC`): Replaced sequential recursion finding algorithms with strict $O(N)$ linear scans filtering over active edge boundaries (`self.active_redexes`) which readies parallel multi-node replacements per loop.

3. **JAX Batched Execution** (`JaxIC`): Wrote `jax_evaluator.py` taking advantage of purely functional substitutions via `heap.at[locs].set(values)`.
   - Replaced dynamic list lengths with a `PADDED_SIZE` buffer and an accompanying static boolean `valid_mask`, allowing the compilation engine (`@jax.jit`) to generate strict static shapes.
   - Employs `jax.lax.scan` to dynamically cycle 100s of graph rewrites completely on-device.
   - Successfully validated natively running isolated loops on CPU XLA targets.

### File Structure
- `Interaction-Calculus/staticic.py`: Emulates the reference engine `tag/val` architecture using fixed NumPy pools.
- `Interaction-Calculus/jax_evaluator.py`: Implements JIT-compiled tensor substitutions for executing evaluations without Python orchestration overheads.
- `Interaction-Calculus/vectorized.py`: Extracts unrolled memory substitution mechanisms across arrays matching standard NPU operations (`jnp.where`).
- `Interaction-Calculus/parser.py`: Python syntax parser reading standard `.ic` text files.
- `Interaction-Calculus/show.py`: Graph Stringifier implementing deterministic Variable namespacing mimicking the C reference compiler output exactly.
- `Interaction-Calculus/test_jax.py`: Run this file to exercise the batched tensor engine explicitly simulating the TPU evaluation states perfectly over test files.
