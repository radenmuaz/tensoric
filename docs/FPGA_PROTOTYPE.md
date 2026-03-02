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

## 3. The Garbage Collection FSM

A pure hardware engine cannot leak memory, otherwise the 1,024 node space will fill instantly.

*   **The Hardware Free-List:** Instead of a complex software garbage collector, the FPGA maintains a hardware FIFO queue (a circular buffer) containing the 13-bit addresses of all "empty" slots.
*   **Allocation (1 clock cycle):** When an `APP` node duplicates and spawns new nodes, the FSM pops an address off the free-list.
*   **Deallocation (1 clock cycle):** When an `ERA` node annihilates a data node, its 13-bit address is instantly pushed back onto the free-list FIFO.

---

## 4. Initial Demos and Benchmarks (Proof of Concept)

What programs do we actually compile to the FPGA to prove it works?

### Demo 1: The Boolean Identity (Testing Routing)
*   **Program:** `NOT(NOT(True))`
*   **Nodes Used:** `LAM`, `APP`. 
*   **Goal:** Prove the basic annihilation rules work and that the final output correctly routes to the "Output Pins" of the FPGA (lighting an LED to represent `True`).

### Demo 2: The Exponential DUP (Testing Memory & GC)
*   **Program:** `DUP(DUP(...(NUM)))`
*   **Nodes Used:** `DUP`, `SUP`, `NUM`.
*   **Goal:** This causes a rapid "explosion" of the graph. It proves that the hardware Free-List can allocate memory fast enough without corrupting pointers, and that cross-label duplication functions natively on silicon.

### Demo 3: The Unary Church Adder (Testing $O(1)$ Math)
*   **Program:** Represent 2 and 3 as Church numerals and apply the IC `Add` rule.
*   **Goal:** Because this relies entirely on structure (`LAM` and `APP`), if this evaluates correctly to the Church numeral 5, it proves the FPGA can successfully compute arbitrary turing-complete functions purely via topology rewrites, without utilizing the FPGA's built-in DSP/DSP48 math slices.

---

## Next Steps for Development

1.  **Write the Verilog FSM:** Begin drafting the state machine for the 8 primitive interaction rules.
2.  **Simulation:** Use Verilator or ModelSim to simulate the Verilog logic against known software IC traces before flashing physical hardware.
3.  **The Compiler Bridge:** Write a small Python script to take `tensoric`'s normal heap output and format it into a `.hex` file that initializes the FPGA's BRAM upon boot.
