# TensorIC: Interaction Calculus on Accelerators

[![JAX](https://img.shields.io/badge/JAX-Powered-blue)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**TensorIC** is an experimental port of the [Interaction Calculus](https://github.com/Interaction-Calculus/Interaction-Calculus) graph rewrite system to a vectorized, static-array formulation using **[JAX](https://github.com/google/jax)**.

The goal of this project is to explore whether Turing-complete symbolic evaluation—which traditionally relies on dynamic heap allocation, pointers, and asynchronous recursion—can be flattened into fixed-shape, pre-allocated tensor operations suitable for modern hardware accelerators (GPUs & TPUs).

## � Concept

Executing symbolic code on accelerators typically bottlenecks at CPU-device communication (e.g., Python `while` loops coordinating GPU steps). TensorIC explores an alternative: **Parallel Structural Representation**.

- **JIT Compilation:** By representing the interaction rules purely functionally (`heap.at[locs].set(values)`) with static arrays, JAX's `jax.lax.scan` primitive can compile the evaluation sweep. The accelerator executes hundreds of iterations autonomously without dropping back to the Python host.
- **Embedded Evaluation:** This architecture aims to allow Turing-complete logic (like Reinforcement Learning environments or symbolic searches) to run entirely on the accelerator, co-located with neural network weights.
- **Lisp Frontend:** TensorIC includes a compiler that lowers untyped Lisp into affine/linear IC structure matrices, automating the insertion of duplication (`DUP`) and erasure (`ERA`) nodes.

## 📚 Documentation

The design process, mathematical constraints, and potential use cases are documented here:

- [**Architecture & Big-O Analytics** (`BIG_O.md`)](BIG_O.md): Details the translation of dynamic graph memory management into static array operations, including benchmarks of tracing compilation overhead.
- [**Applications & Future Vision** (`APPLICATIONS.md`)](APPLICATIONS.md): Explores the theoretical use cases for zero-roundtrip RL environments and parallel symbolic execution.
- [**Implementation Specifications**](.gemini/antigravity/brain/b126d6eb-cef6-4a0b-8239-6b3685eee277/implementation_plan.md) *(Internal)*: Details on the Vectorized evaluator translation map.

## 📦 Installation & Setup

### Prerequisites
Requires Python 3.9+ and JAX.
```bash
python3 -m venv venv
source venv/bin/activate
# Install CPU version of JAX (for GPU/TPU see: https://github.com/google/jax#installation)
pip install jax jaxlib numpy
```

### Reproducing Benchmarks
The test suite validates graph normalization against the original C implementation and measures JIT compilation timings across varied scan sequence lengths (`--steps`). 

```bash
python3 Interaction-Calculus/run_tests.py
```
*Tests output to `test_logs_steps_XXX.txt` to verify whether XLA recompiles the structure.*

### Lisp REPL
Evaluate a functional Lisp program compiled down and evaluated on the JAX backend:
```bash
python3 Interaction-Calculus/repl.py examples/math.lisp --steps=200
```

## 📜 Supported Lisp Features
Because the Interaction Calculus enforces Affine logic (variables interact exactly once), writing algorithms requires strict linearity formatting. The custom `lisp_compiler.py` automates this.

**Supported Constructs:**
- **First-Class Closures**: Standard `(lambda (x y) ... )` mapping to IC `LAM` and `APP` nodes.
- **Native Numerals & Branching**: 32-bit unsigned math integers (`NUM`), natively supporting Zero-checks and Sucessor operations (`SWI`, `SUC`) for logical branching.
- **Automatic Linearity**: Variables used $N > 1$ times dynamically spawn duplication networks (`DUP`). Unused variables evaluate to erasure bounds (`ERA`).
- **Examples**: Basic Boolean algebra (`examples/bool.lisp`) and recursive Math closures (`examples/math.lisp`).

## ⚠️ Current Limitations

This is a research prototype. There are significant engineering hurdles before this achieves parity with compiled CPU engines (like Rust/C):

1. **Pre-allocated Memory Limits:** In pure JAX, tensor shapes cannot grow dynamically. The `JAX_MAX_NODES` array must be pre-allocated to the maximum plausible graph dimension memory bound. This heavily inflates memory footprint per evaluation context.
2. **Dense Masking vs Sparse Graphs:** The vectorized evaluator checks the *entire array* iteratively (`jnp.where`) looking for active combinations (`APP` + `LAM`). This means $O(N)$ scanning over the heap even if there are sparse localized active redexes.
3. **Garbage Collection Bottleneck:** Unreferenced IC nodes accumulate rapidly. The prototype pure JAX garbage collector (`jax_gc_research.py`) requires a deep fixed-point array prefix-sum iteration which halts the TPU computation pipeline for expensive sweeping logic.

## 🤝 Contributing

We are exploring the intersection of **Programming Language Theory (PLT)** and **Tensor Compaction**. 

Areas for exploration:
- Improving the TPU trace profiles to extract better FLOP ratios for the linear sparse interaction scanning logic natively.
- Refining structural garbage collection heuristics to avoid whole-array traversals.
- Developing differentiable extensions to the IC substitution rules recursively natively.
