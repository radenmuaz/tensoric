# FPGA Prototype: Interaction Calculus Engine Blueprint

To physically prove the viability of a pure Interaction Calculus (IC) hardware architecture, the first logical step is prototyping on an FPGA (Field-Programmable Gate Array) using Verilog or SystemVerilog. 

This document outlines the initial parameters, limits, and milestones for a "V1" FPGA prototype of TensorIC.

---

## 1. Initial Parameters & Node Specifications

To fit within the memory (BRAM) and logic (LUT) constraints of an affordable development FPGA (like an Artix-7 or Cyclone V), the "V1" engine must be strictly minimalist.

### The 16-bit Node Architecture
We optimize for a **16-bit word size**. This perfectly balances dense packing in Block RAM (BRAM) while providing enough pointer space for a small, provable graph.

*   **Total Node Width:** 16 bits
*   **Tag Space (3 bits):** Allows for 8 primitive node types.
    *   `000` = `ERA` (Eraser)
    *   `001` = `LAM` (Lambda / Constructor)
    *   `010` = `APP` (Application / Destructor)
    *   `011` = `DUP` (Duplicator)
    *   `100` = `SUP` (Superposition / DUP counterpart)
    *   `101` = `NUM` (Tiny Integer or Bit)
    *   `110` = `OP` (Boolean/Math Operator)
    *   `111` = `VAR` (Wire / Indirection)
*   **Payload / Pointer Space (13 bits):**
    *   A 13-bit unsigned integer allows the FPGA to address exactly **8,192 nodes** in hardware memory ($2^{13}$).
    *   If a node needs to store a literal value (when tagged `NUM`), it can store a 13-bit integer (0 to 8,191).
    *   *Note on Labels:* Standard IC `DUP`/`SUP` nodes require labels to prevent incorrect annihilations. In a 16-bit limitation, the pointer payload might need to dynamically split (e.g., 3 bits for label, 10 bits for pointer = 1,024 nodes max). For V1, we may stick to 1,024 nodes to properly support labels.

---

## 2. Memory and Execution Fabric Constraints

How does the FPGA store the graph and execute the substitution rules?

### A. The Memory Fabric: Dual-Port BRAM
The entire IC graph lives in the FPGA's on-chip Block RAM (BRAM). 
*   **Structure:** An array of 1,024 words, where each word is a 32-bit tuple (Main Port Pointer [16 bits] + Aux Port Pointer [16 bits]).
*   **Why Dual-Port?** Dual-port BRAM allows the hardware to read/write two memory addresses per clock cycle simultaneously. This is the absolute minimum requirement to process an interaction (which inherently involves modifying at least two connected nodes).

### B. Execution Fabric: Array-Based / Systolic Evaluator (Mirroring TensorIC)
Instead of a standard CPU-like Finite State Machine (FSM) that sequentially steps through memory, the FPGA should physically mirror TensorIC's vectorized `jax.lax.scan` architecture.

This requires a **1D Spatial Array Processor (Systolic Array)** concept:
*   **The Processor Elements (PEs):** The BRAM array of 1,024 nodes is physically divided into smaller chunks, or each index conceptually acts as a register wired to a massively parallel combinational logic block.
*   **The Vectorized Sweep:** Instead of sequentially scanning with one ALU, the FPGA instantiates `N` lightweight logic gates (Comparators) that simultaneously check every single index in the array for an active redex (e.g., checking if Node at Index `i` is an `APP` and its `main_port` points to a `LAM`).
*   **The `jnp.where` in Hardware:** In TensorIC, `jnp.where(condition, true_vals, original_vals)` evaluates globally. On the FPGA, you use **Masked Parallel Write-Backs**. During a single clock cycle, all indices that match an interaction rule raise a "Write Enable" flag. On the next clock cycle, all triggering interactions are simultaneously written back to their respective BRAM locations.
*   **Routing (The Pointer Redirect):** A critical challenge in this parallel array is pointer redirection (e.g., wiring an output port to a grandparent node during a beta-reduction). The FPGA must use a **non-blocking Crossbar or a specialized routing permutation network** to allow multiple indices to swap pointer payloads in parallel without colliding memory bus read/writes.

---

## 3. The Garbage Collection Architecture (Free-List vs Vectorized Compaction)

A pure hardware engine cannot leak memory, otherwise the 1,024 node space will fill instantly. We have two options for the FPGA, mimicking different architectural goals.

### Option A: The Hardware Free-List (Instant latency, $O(1)$)
This is the simplest hardware solution. 
*   **The Hardware FIFO:** The FPGA maintains a circular buffer (FIFO) containing the 13-bit addresses of all "empty" slots.
*   **Allocation (1 clock cycle):** When an `APP` node duplicates and spawns new nodes, the FSM pops an address off the FIFO.
*   **Deallocation (1 clock cycle):** When an `ERA` node annihilates a data node, its address is instantly pushed back onto the FIFO.
*   **Verdict:** Extremely fast, zero overhead. However, it leads to memory fragmentation.

### Option B: Vectorized JAX GC Port (Parallel Prefix-Sum)
Can the experimental `jax_gc_research.py` algorithm be ported to the Systolic FPGA Array? **Yes.**

The JAX prototype uses two phases: `jax_mark_sweep` and `jax_compact` (Prefix-Sum). This can be mapped to hardware logic:
1.  **Parallel Mark-and-Sweep (Breadth-First Raycasting):** 
    *   In the JAX prototype, a boolean `alive_mask` is updated iteratively. 
    *   In hardware, every PE (Index) wires its pointers to a massive routing grid. Starting from the Root Node, an electrical "Alive Signal" propagates through the graph's connections. 
    *   If the graph depth is 10, it takes exactly 10 clock cycles for the signal to reach every active node. All nodes that hold a `1` on their "Alive" pin keep their data.
2.  **Hardware Prefix-Sum (Blelloch Scan) for Compaction:**
    *   In JAX, `jnp.cumsum(alive_mask)` generates the compacted target indices.
    *   In FPGA hardware, a **Blelloch Parallel Tree Scanner** can compute the Prefix Sum of the 1,024-bit `alive_mask` in exactly $\log_2(1024) = 10$ clock cycles!
    *   On the 11th clock cycle, every alive node simultaneously moves its data to its new compacted `prefix_sum` index, completely defragmenting the BRAM.
*   **Verdict:** Porting the JAX GC to hardware is entirely feasible and solves fragmentation. The tradeoff is resource utilization: building a 1,024-wide Parallel Prefix-Sum tree consumes significant LUTs on the FPGA compared to a simple FIFO queue.

---

## 4. Hardware Resource Estimation (LUT & BRAM Targets)

When targeting a mid-range development board (e.g., Xilinx Artix-7 100T or AMD Zynq-7020), we must consider the physical constraints of Logic Cells (LUTs) and Block RAM (BRAM). 

Here is an aggressive but realistic estimate for a **1,024-node Array-Based** engine:

1.  **Memory (BRAM)**
    *   1,024 nodes $\times$ 32 bits = 32,768 bits (32 Kbit). 
    *   A single Artix-7 36Kb Block RAM tile can hold the entire graph. BRAM utilization is negligible ($< 1\%$).
2.  **Execution Fabric (Comparators)**
    *   To check interactions natively across all 1,024 indices in parallel, each index needs a basic comparator (checking Tag matches). 
    *   *Estimate:* ~10-15 LUTs per index.
    *   *Total Array Logic:* $\sim 15,000$ LUTs.
3.  **Routing/Crossbar Network (The Bottleneck)**
    *   A full $1024 \times 1024$ non-blocking Crossbar is mathematically impossible on mid-range FPGAs (requires $\sim 1,000,000+$ LUTs). 
    *   *Solution:* We must use a **Banyan Network** or **Benes Permutation Network**. For $N=1024$, this requires $N \log_2 N$ switching stages. 
    *   *Estimate:* $\sim 20,000$ LUTs depending on multiplexer pipelining.
4.  **Hardware Garbage Collection (Vectorized Option B)**
    *   A 1,024-wide Blelloch Parallel Prefix-Sum tree requires exactly 1,023 Adders.
    *   Because the sum only reaches a maximum value of 1,024 (a 10-bit integer), we only need 10-bit half-adders.
    *   *Estimate:* 1,023 Adders $\times$ 10 LUTs/Adder = $\sim 10,000$ LUTs.
5.  **Total System Footprint**
    *   $\sim 45,000 - 55,000$ LUTs.
    *   This fits comfortably on an Artix-7 100T (which has 63,400 LUTs). This proves the Array-Based model is dense but achievable.

---

## 5. Initial Demos and Benchmarks (Array-Based Verification)

What programs do we actually compile to the FPGA to prove the parallel architecture functions correctly? Because we are using an Array-Based execution fabric, our demos should explicitly test masked parallel substitutions.

### Demo 1: The Vectorized Broadcast (Testing the Crossbar)
*   **Program:** `DUP(FLT_ARRAY_ROOT)`
*   **Goal:** A massive array of numbers is duplicated at once. This tests the absolute bandwidth capacity of the Benes Permutation Routing network. If the FPGA duplicates 100 nodes in 3 clock cycles without pointer collision, the parallel execution fabric is verified.

### Demo 2: The Parallel Boolean Reducer
*   **Program:** A wide, flat tree of Boolean `AND` and `OR` logic gates (e.g., resolving a 16-bit Parity check).
*   **Goal:** In a sequential CPU engine, reducing 16 boolean gates takes 16 loops. In our FPGA array architecture, all non-dependent gates should be evaluated by the comparators in the exact same clock cycle. We verify that the graph depth collapses logarithmically ($O(\log N)$).

### Demo 3: The JAX GC Compaction Test
*   **Program:** A graph with 500 active nodes and 500 randomly scattered ERA (dead) nodes.
*   **Goal:** Trigger the GC Phase. We connect a Logic Analyzer (ILA) to the FPGA to physically trace the 10-cycle Blelloch Prefix-Sum tree and verify that on the 11th clock cycle, all 500 active nodes perfectly compact into indices `0` to `499` in BRAM.

---

## Next Steps for Development

1.  **Write the Verilog FSM:** Begin drafting the state machine for the 8 primitive interaction rules.
2.  **Simulation:** Use Verilator or ModelSim to simulate the Verilog logic against known software IC traces before flashing physical hardware.
3.  **The Compiler Bridge:** Write a small Python script to take `tensoric`'s normal heap output and format it into a `.hex` file that initializes the FPGA's BRAM upon boot.
