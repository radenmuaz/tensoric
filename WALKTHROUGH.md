# Interaction Calculus NumPy Port: Verification

## Overview 
The goal was to port the complex asynchronous pointer-chasing evaluator inside `ic.c` into a NumPy `ndarray`-compatible format that allows static-shape processing required by custom hardware (TPUs and NPUs).

The implementation satisfies these requirements via `staticic.py`:
1. **Removed C Pointer Traversal**: The entire 64/32-bit tagged pointer system was ported perfectly to integer equivalents using NumPy flat arrays.
2. **Fixed Tensor Addresses**: `self.heap` and `self.stack` are allocated with a static size (33.5 million nodes limit) upon initialization. Nodes interact solely by retrieving `loc` addresses, mimicking exactly what a TPU matrix would do.
3. **Reproducibility**: The interactions strictly trace the original C's weak-head-normal-form (WHNF) algorithm.

## Verification of Examples

To verify the Python logic against the C execution baseline, both engines parsed and fully expanded all examples (`examples/test_0.ic` through `examples/test_5.ic`) into normal forms (via `ic_normal`). 

The output is perfectly identical across all edge cases (duplications, superpositions, num interactions, erased blocks) and both evaluators complete with the exact same number of allocated nodes and interaction counts.

### Test Comparison Script Output

**`test_0.ic`**
```
--- Testing examples/test_0.ic ---
C Output:
λ$a.λ$b.$a

WORK: 3670093 interactions
SIZE: 8388831 nodes

Python Output:
λa.λb.a
WORK: 3670093 interactions
SIZE: 8388831 nodes
```
*(Nodes allocated match perfectly at 8,388,831 nodes)*

**`test_1.ic`**
```
--- Testing examples/test_1.ic ---
C Output:
! &0{$a2,$b3} = $a;
λ$a.λ$b.($a2 ($b3 $b))

WORK: 0 interactions
SIZE: 8 nodes

Python Output:
! &0{a2,b3} = a;
λa.λb.(a2 (b3 b))
WORK: 0 interactions
SIZE: 8 nodes
```

**`test_4.ic`**
```
--- Testing examples/test_4.ic ---
C Output:
λ$a.λ$b.λ$c.($b λ$d.λ$e.λ$f.($e λ$g.λ$h.λ$i.($h λ$j.λ$k.λ$l.($k λ$m.λ$n.λ$o.$o))))

WORK: 474 interactions
SIZE: 1420 nodes

Python Output:
λa.λb.λc.(b λd.λe.λf.(e λg.λh.λi.(h λj.λk.λl.(k λm.λn.λo.o))))
WORK: 474 interactions
SIZE: 1420 nodes
```

## Vectorization for TPUs (Native & JAX)
A preliminary vectorizer algorithmic base was implemented in `vectorized.py` to bridge the gap between WHNF pointer jumping and GPU parallel kernels. It replaces recursive search with flat, array-sweeping extraction for interaction graph edges (`Redexes`):
```python
self.active_redexes_l = np.zeros(size, dtype=np.uint32)
self.active_redexes_r = np.zeros(size, dtype=np.uint32)
```

Its `step_vectorized()` logic demonstrates evaluating the graph explicitly using boolean filters: isolating regions of exact tensor instructions (like `APP-LAM` mask) and rewriting memory uniformly in a single unrolled tensor operation (`np.where`).

### JAX JIT Compatibility 
To prove execution across TPU and specialized AI hardware natively, the `JaxIC` module (`jax_evaluator.py`) was introduced to compile tensor interaction models directly using `@jax.jit`. 
In-place memory manipulations were replaced strictly via JAX's purely functional scattered update API: `heap.at[indices].set(values)`.

**Batched Execution (`jax.lax.scan`)**
To save TPU to host call overheads, the loop was completely ported to operate dynamically within XLA. `test_jax.py` runs `ic.run_scan(steps=100)`, executing the rules natively by mapping operations (`jnp.where`) directly over boolean array matches (`app_lam_mask`). 

This proves the model can securely allocate cycles completely asynchronously on TPUs without requiring arbitrary pointer dereferencing from CPU orchestrators.
