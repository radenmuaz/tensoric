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

### B. Execution Fabric: The Active Mesh (Parallel vs Sequential)
There are two ways to build the execution logic:

1.  **The Sequential Scanner (V1 Recommended):**
    *   A centralized finite state machine (FSM) linearly scans the BRAM address space.
    *   When it finds an active pair (a redex, e.g., `LAM` pointing to `APP`), it pauses the scan, fetches the surrounding nodes, applies the rewrite logic in 1-3 clock cycles via a hardware ALU, updates the BRAM, and resumes scanning.
    *   *Pros:* Fits on the smallest, cheapest FPGAs. Easy to debug in Verilog.
    *   *Cons:* Does not demonstrate the true parallel speedup of IC.
2.  **The Parallel Pipelined Mesh (V2 Goal):**
    *   The BRAM is divided into chunks (e.g., 8 independent memory banks of 128 nodes).
    *   8 hardware Evaluator units run in parallel, constantly looking for interactions in their local banks and applying rewrites simultaneously.
    *   A crossbar switch handles routing when an interaction crosses banks.

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
