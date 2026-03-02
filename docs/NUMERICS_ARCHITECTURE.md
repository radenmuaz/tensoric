# Practical Number and Tensor Representation in TensorIC

The goal is to enable TensorIC to feasibly run numerical simulations (PDEs, rigid body dynamics, etc.) with good precision (f32/f64) and efficient storage, while preserving its JAX-accelerated parallel interaction reduction architecture.

## Background constraints
In `tensoric/staticic.py`, nodes are packed into 32-bit integers:
- 6 bits for the `TAG` (e.g., `LAM`, `APP`, `NUM`).
- 26 bits for the `PAYLOAD` (pointer, label, or 32-bit unsigned offset logic).
Because only 26 bits are available, we cannot store a full IEEE 754 32-bit floating-point number directly inside a standard IC node payload without losing precision (a 26-bit float has severely truncated mantissa/exponent, making it impractical for scientific compute).

## Proposed Architecture: The "Out-of-Band" Value Registers

To make numerical simulation practical, TensorIC must treat Interaction Calculus as the **Control & Coordination** layer, while treating heavy numerical data as **Dataflow Registers** passed around via referential pointers.

### 1. Scalar Floating Point (`FLT`)
We introduce a new node tag: `FLT` (Float).
Instead of storing the float in the 26-bit payload, the payload acts as a 26-bit **pointer (index)** into a parallel auxiliary memory buffer: `float32_heap` (which is a standard `jnp.ndarray` of dtype `float32`).

- **Storage Structure**: 
  - `ic_heap` (uint32): `[..., (FLT << 26) | i, ...]`
  - `float32_heap` (float32): `[..., 3.14159, ...]`
- **Interactions**:
  - We introduce operator nodes like `ADD_F`, `MUL_F`.
  - When `ADD_F` interacts with two `FLT(i)` and `FLT(j)` nodes, the evaluator allocates a new index `k`, sets `float32_heap[k] = float32_heap[i] + float32_heap[j]`, and produces `FLT(k)`.
- **Garbage Collection**: Because `FLT` nodes can be copied via `DUP` networks, floats must be garbage-collected or reference-counted synchronously with the main IC heap. The `jax_gc_research.py` logic easily extends to mark-and-sweep the auxiliary `float32_heap`.

### 2. Dense Tensors (`TSR`)
For PDEs and Rigid Body simulations, doing scalar IC arithmetic is too slow. Spatial grids (like velocity fields in PDEs) must be processed natively.
We introduce the `TSR` (Tensor) tag.

- **Storage Structure**:
  - The payload of `TSR` is an index into a `tensor_registry` (a pool of pre-allocated fixed-size `jnp.ndarray` buffers, or a unified multi-dimensional buffer if sizes are uniform, e.g., grid size $N \times N$).
- **Interactions**:
  - `TSR_CONV2D`, `TSR_MATMUL`, `TSR_ADD` act as primitive operation nodes in IC.
  - When `TSR_ADD` interacts with `TSR(a)` and `TSR(b)`, it dispatches a bulk `jax.numpy.add(tensor_registry[a], tensor_registry[b])` and returns `TSR(c)`.
  - **Why this works seamlessly in JAX**: Since the IC engine uses `jax.lax.scan`, bulk operations over `tensor_registry` can be easily compiled by XLA in the same pipeline as the IC graph reductions.

## Feasibility & Practicality

1. **Precision**: We gain native `f32` (or `f64`) hardware precision. No truncation.
2. **Performance**: Heavy mathematical looping (like PDE finite-difference stencils) is offloaded to pure JAX tensor ops. IC simply wires the inputs to the outputs (e.g., piping the output of timestep $T$ into timestep $T+1$).
3. **Storage**: Storing large ND arrays out-of-band prevents the IC heap from exploding in size, retaining cache locality for the structural IC graph while keeping dense data contiguous for the accelerator's matrix math units.

## Tradeoff Analysis: Out-of-Band vs. Pure IC Numbers

When considering a pure Interaction Calculus implementation natively ported to custom hardware (e.g., ASICs or FPGAs) without an external floating-point ALU, we face a distinct set of constraints compared to GPU/TPU accelerators. Here are the tradeoffs:

### 1. Out-of-Band (Separate Lanes) Architecture
**Pros:**
*   **Accelerator Synergy:** GPUs and TPUs are purpose-built for massive, contiguous floating-point array operations. Separating the IC control graph from the dense numeric data allows JAX to compile the numerics into highly optimized tensor operations.
*   **Memory Efficiency (Graph Size):** The IC graph remains compact (32-bit nodes). We don't bloat every structural node (LAM, APP, SUP) with 64-bit payloads just to accommodate 32-bit floats elsewhere in the graph.
*   **Immediate Practicality:** We can simulate PDE or Rigid body dynamics today using JAX's `jax.lax.scan` at reasonable speeds.
*   **No Precision Loss:** We get full hardware IEEE 754 precision.

**Cons:**
*   **Requires Hybrid Ecosystem:** It relies on an external tensor engine (JAX) and floating-point ALUs. It is not "pure" IC.
*   **Synchronization Overhead:** The IC garbage collector must carefully synchronize and manage reference counts for the external float arrays when `DUP` nodes clone float references.

### 2. Pure IC Numerical Implementation (Custom Hardware Target)
In a pure IC implementation, numbers would be represented either natively within 64-bit nodes or entirely via structural encodings (like Scott-encoded Church numerals/floats or explicit bit-level representations).

**Pros:**
*   **Hardware Unification:** If you build a custom IC ASIC, you do not need a separate Floating Point Unit (FPU) or a separate memory lane for data. *Everything* is graph rewrite logic. The architecture is elegantly unified.
*   **Granular Parallelism:** Arithmetic operations (like multiplying two numbers represented structurally) become graph reduction steps that can be evaluated in parallel across the IC fabric, perfectly aligned with the asynchronous nature of Interaction Calculus.
*   **Elegant Semantics:** DUPlicating a number is just sending a DUP node through its structural representation; garbage collection is also handled natively via ERA nodes without touching external reference counters.

**Cons:**
*   **Catastrophic Memory Explosion:** Representing a 32-bit float purely via IC nodes (e.g., representing the mantissa, exponent, and sign as trees of nodes) would require dozens or hundreds of IC nodes per *single float*. A 1-million node graph might only be able to store a few thousand numbers.
*   **Simulation Speed (Currently):** Without an FPU, performing heavy PDE operations (millions of multiplications per timestep) via graph rewrites would be exponentially slower on current von Neumann architectures (CPUs/GPUs).
*   **Custom ASIC Requirement:** To beat the Out-of-Band approach, you would absolutely need custom hardware designed specifically for ultra-fast, massive-scale IC node substitutions. Software IC engines cannot compete with hardware ALUs for dense math.

### Conclusion for TensorIC
If the goal of `tensoric` is to simulate PDE mathematically using *current* JAX/GPU hardware, the **Out-of-Band (OOB)** approach is the only feasible path. 
However, if `tensoric` is a stepping stone to designing a **Custom IC Hardware Architecture**, exploring pure structurally-encoded numbers or expanding the core node to 64-bits (Embedding) is highly theoretically valuable, as it avoids complex dual-memory-lane synchronization on the ASIC.
