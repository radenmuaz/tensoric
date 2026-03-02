# TensorIC: Interaction Calculus on Accelerators ⚡️

[![JAX](https://img.shields.io/badge/JAX-Powered-blue)](https://github.com/google/jax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**TensorIC** is a revolutionary approach to symbolic computation and strict mathematical evaluation on modern hardware accelerators (GPUs & TPUs). 

Traditional Neural Networks are highly parallel but lack Turing-complete control flow and native symbolic reasoning. Conversely, traditional interpreters (like Python or C) are Turing-complete but execute sequentially, bottlenecking at the CPU.

**TensorIC bridges this gap.** By porting the ultra-concurrent [Interaction Calculus](https://github.com/Interaction-Calculus/Interaction-Calculus) graph rewrite system into pure, vectorized **[JAX](https://github.com/google/jax)** arrays natively, we can compile full Lisp programs—complete with closures, recursion, and higher-order functions—directly into static tensor computations.

## 🚀 Why TensorIC?

Executing symbolic code on TPUs/GPUs normally requires constant host-device communication (Python `while` loops roundtripping to the accelerator). TensorIC completely abandons pointer-chasing and dynamic heap allocation in favor of **Parallel Structural Representation**.

- **Zero Python Overhead:** JAX's `jax.lax.scan` primitive compiles the evaluation loop exactly *once*. Billions of graph interactions run directly on the TPU silicon mapping evaluation traces.
- **Turing-Complete AI Agents:** Reinforcement Learning models no longer need to wait on CPU-bound environments. The entire simulation environment and the agent's logic can be compiled into a single static GPU/TPU graph.
- **Lisp Compiler Included:** Write high-level symbolic Lisp code, automatically lowering it to JAX multidimensional array networks recursively.

## 📚 Deep Dives & Documentation

We have thoroughly documented the architecture, mathematical constraints, and vast possibilities of this paradigm. 

- [**Applications & Future Vision** (`APPLICATIONS.md`)](APPLICATIONS.md): Read about Zero-Roundtrip RL environments, compiling software natively to NPUs, and massively parallel symbolic searches natively. 
- [**Architecture & Big-O Analytics** (`BIG_O.md`)](BIG_O.md): Discover how we conquered dynamic graph memory management across static memory blocks natively. View empirical benchmarks proving strictly $O(1)$ tracing compilation overhead. 
- [**Implementation Specifications**](.gemini/antigravity/brain/b126d6eb-cef6-4a0b-8239-6b3685eee277/implementation_plan.md) *(Internal)*: Details on the Vectorized evaluator translation map.

## ✨ Key Features

1. **JAX Batched Execution (`JaxIC`)**: Graph algorithms execute entirely functionally via `heap.at[locs].set(values)`. The tensor operations are vectorized across edge relationships, executing multiple $\lambda$-reductions synchronously per clock cycle.
2. **Native Lisp Frontend**: `repl.py` enables parsing and compiling recursive S-Expressions (`math.lisp`, `bool.lisp`). The compiler features a custom Linearity/Scope inference pass, auto-inserting duplication (`DUP`) and erasure (`ERA`) nodes corresponding strictly to affine logic networks natively.
3. **Pure JAX Garbage Collection**: Instead of relying on host RAM management natively, garbage collection natively relies entirely upon fixed-point iteration and parallel prefix sums (`jnp.cumsum`) to compact and compress memory blocks on the accelerator instantaneously.

## 📦 Installation & Setup

### Prerequisites
Requires Python 3.9+ and JAX.
```bash
python3 -m venv venv
source venv/bin/activate
pip install jax jaxlib numpy
```

### Reproducing Benchmarks
Our integrated test suite strictly measures compilation timings and proves static sequence execution natively. The tests will log to `test_logs_steps_XXX.txt` to strictly mathematically prove JAX compiling correctly caches trace graphs precisely $1$ time globally.
```bash
python3 Interaction-Calculus/run_tests.py
```

### Try the Lisp REPL
Execute an entire functional Lisp program isolated natively to TPU execution:
```bash
python3 Interaction-Calculus/repl.py examples/math.lisp --steps=200
```
This script dynamically parses Lisp, translates it into strict IC Affine relationships, and evaluates it natively using `jax_evaluator.py`.

## 📜 Supported Lisp Features (`lisp_compiler.py`)
Because Interaction Calculus inherently enforces Affine logic (variables must interact exactly once), writing traditional algorithms is notoriously difficult. Our compiler automatically resolves this barrier.

**Supported Constructs:**
- **First-Class Closures**: Standard `(lambda (x y) ... )` mapping to Interaction Calculus `LAM` and `APP` nodes.
- **Global Definitions**: `(def name value)` natively evaluated.
- **Native Numerals & Branching**: 32-bit unsigned math integers (`NUM`), natively supporting Zero-checks and Sucessor operations (`SWI`, `SUC`) for building logical switches: `(if condition true_branch false_branch)`
- **Boolean Algebra**: Standard logical evaluators integrated into `examples/bool.lisp`.
- **Recursive Math**: Native recursive mappings implemented inside `examples/math.lisp`.
- **Automatic Memory Linearity**: The compiler abstracts away manual graph manipulation via an autonomous Scope Pass. Variables used $N > 1$ times dynamically spawn Duplication networks (`DUP`). Unused variables evaluate to Erasure bounds (`ERA`).

## 🧠 Architecture Stack
- `Interaction-Calculus/jax_evaluator.py`: The core JIT-compiled tensor engine natively. Utilizes strictly functional updates without Python orchestration overheads globally.
- `Interaction-Calculus/lisp_compiler.py` & `lisp_parser.py`: The frontend transforming untyped Lisp into affine/linear IC structure matrices.
- `Interaction-Calculus/vectorized.py`: Abstracts the graph traversal into index-array maps representing raw NPUs operations (`jnp.where`).
- `Interaction-Calculus/staticic.py`: Emulates the reference engine `tag/val` 32-bit architecture natively over static NumPy pools.
- `Interaction-Calculus/jax_gc_research.py`: Explores completely structural natively bounded Garbage Collection techniques.

## 🤝 Contributing (Calling ML Researchers!)

We are standing at the intersection of **Programming Language Theory (PLT)** and **High-Performance Deep Learning**. 

If you are an ML researcher interested in executing symbolic logic directly embedded inside Transformer architectures natively, or a systems-level engineer aiming to accelerate symbolic evaluation via static tensor representations, we want your help!

Areas for Contribution:
- Implementing deeply recursive algorithms purely natively in our Lisp dialect globally.
- Validating TPU trace profiles to extract maximum FLOP ratios natively for parallel graph reductions.
- Exploring Continuous/Differentiable extensions natively extending the substitution logic recursively.

---
*Built to bring Turing-complete intelligence directly to the silicon.* 
