# Interaction Net Reduction in TensorIC: A Step-by-Step Guide

TensorIC evaluates purely functional Lisp programs by translating them into **Interaction Calculus**, a graph-rewrite system. What makes TensorIC unique is how it performs these graph rewrites: instead of using pointer-chasing and dynamic heap allocation as typically done in CPU-based interpreters (like Rust or C), TensorIC flattens the graph into a **static, 1D array** and performs reductions in parallel using vectorized masks (JAX).

This tutorial explains the memory layout and provides a step-by-step example of how a parallel reduction works.

---

## 1. Memory Layout: The Flat Graph Array

The graph is stored in a large one-dimensional array (`self.heap`, pre-allocated to maximum capacity). Each element in this array is a **32-bit integer** representing a specific term (node port):

- **Bit 31 (1 bit):** Substitution Flag (`TERM_SUB_MASK`) - Marks nodes that are currently forwarding to another memory location (pointer indirection).
- **Bits 26-30 (5 bits):** Tag (`TERM_TAG_MASK`) - The node type (e.g., Application `APP`, Lambda `LAM`, Eraser `ERA`, Duplicator `DX/DY`, Number `NUM`).
- **Bits 0-25 (26 bits):** Value (`TERM_VAL_MASK`) - Either an immediate value (for `NUM`) or a pointer (index) to another location in the array.

### Nodes Allocate Adjacent Indices
Nodes that take up more than one slot allocate adjacent cells in the array. 
For example, an Application node (`APP`) allocated at index `loc` occupies two slots:
- `heap[loc]`: The function it's applying to.
- `heap[loc + 1]`: The argument.

A Lambda node (`LAM`) allocated at `loc` occupies one slot:
- `heap[loc]`: The body of the lambda.

---

## 2. The Vectorized Reduction Loop

The evaluation progresses in massive, simultaneous sweeps over the `heap`:

1. **Sweeping/Masking:** TensorIC checks the array using a parallel mask. It looks for active pairs of interacting nodes (called **Redexes**). The most common redex is an Application node pointing to a Lambda node (`APP - LAM`).
2. **Scatter/Gather:** Using `jnp.where` (or `np.where` in the python prototype), it extracts the indices of all left-side nodes (APPs) and right-side nodes (LAMs) that form a valid redex.
3. **Parallel Rewrite:** TensorIC applies the rewriting rules simultaneously across all found indices, swapping out arrays of pointers in a single operation.

---

## 3. Step-by-Step Example: APP-LAM Reduction

Let's look at the classic function application reduction: feeding an argument into an identity lambda function.

Imagine an array state where an `APP` node at index `10` is pointing to a `LAM` node at index `20`. The `APP` node has an argument waiting at `11`, pointing to some number node at index `30`.

### Initial State

| Index | Binary/Hex Tag | Human Readable | Meaning |
|-------|----------------|----------------|---------|
| `10`  | `APP` | `APP 20` | Function pointer (Points to LAM at `20`) |
| `11`  | `VAR` | `VAR 30` | Argument (Points to variable/number at `30`) |
| ... | | | |
| `20`  | `LAM` | `LAM 40` | Body pointer (Points to body logic at `40`) |
| ... | | | |
| `30`  | `NUM` | `NUM 5` | The number 5 |
| `40`  | `VAR` | `VAR 20` | Lambda's internal bound variable (pointing back) |

### Step 1: Masking (Finding the Redex)
TensorIC scans the array.
- It finds an `APP` at index `10`. What's the value of `heap[10]`? It's `20`.
- It checks `heap[20]`. The tag is `LAM`.
- **Match Found!** This is an `APP-LAM` interaction.
- `idx_app_lam` arrays record: `app_locs = [10]`, `lam_locs = [20]`.

### Step 2: Parallel Fetching Context
For all matched pairs, TensorIC fetches the argument and the lambda body simultaneously:
- `args = heap[app_locs + 1]` -> `heap[11]` -> `VAR 30`
- `bods = heap[lam_locs + 0]` -> `heap[20]` -> `LAM 40` (the actual body value pointed by LAM is `40`)

### Step 3: Rewriting (Parallel Substitution)
The Interaction Calculus rule for `APP-LAM` dictates two pointer updates:
1. The **Argument** substitutes the `LAM`'s bound variable. We do this by turning the `LAM` node into a Substitution pointer to the argument.
2. The **Application** node itself is replaced by the `LAM`'s evaluated body.

TensorIC performs these overwrites in parallel using scatter updates:
```python
self.heap[lam_locs] = self.make_sub(args)
self.heap[app_locs] = bods
```

### Resulting State

| Index | Binary/Hex Tag | Human Readable | Meaning |
|-------|----------------|----------------|---------|
| `10`  | (`VAR`) | `VAR 40` | **[MODIFIED]** APP becomes a direct link to the LAM's body (`40`) |
| `11`  | `VAR` | `VAR 30` | Left unchanged (orphaned, will be garbage collected) |
| ... | | | |
| `20`  | (`SUB`) | `[SUB] 30` | **[MODIFIED]** LAM is overwritten as a Substitution flag pointing to the argument `30`. Anything that originally pointed to the LAM's bound variable will now resolve through this `SUB` flag directly to `30`. |
| ... | | | |
| `30`  | `NUM` | `NUM 5` | The number 5 |
| `40`  | `VAR` | `VAR 20` | When evaluated later, it reads `20`, sees the `SUB` flag, and forwards automatically to `30`. |

## Summary
By using substitution flags (`TERM_SUB_MASK`), TensorIC avoids having to recursively walk down the tree to replace bound variables. Instead, it just overwrites the Lambda's root node to become a "forwarding address" (the substitution node) to the new argument. In the next evaluation sweeps, any node reading from that address automatically traverses the forward to find the true value. 

Because all of these `app_locs` and `lam_locs` are numpy arrays (or JAX arrays), **millions of disjoint APP-LAM reductions can happen simultaneously in a single TPU/GPU tensor operation!**
