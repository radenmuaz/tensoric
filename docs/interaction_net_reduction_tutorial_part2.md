# Interaction Net Reduction Tutorial: Part 2
## Understanding `test_1.ic` and `test_3.ic`

If you look into `examples/test_1.ic` and `examples/test_3.ic`, you might wonder: what do these programs do, and how do they reduce step-by-step when evaluated?

This document explores exactly what happens when we run them through the TensorIC JAX Evaluator backend.

### What are these files?
Both files represent functions mapping to mathematical **Church Numerals** in lambda calculus:
- **`test_1.ic`** represents the number **2**: `λf. λx. !&0{f0,f1}=f; (f0 (f1 x))`
- **`test_3.ic`** represents the number **3**: `λf.λx.(f (f (f x)))`

Church numerals encode numbers strictly as functions that apply a given function $f$ exactly $N$ times to an argument $x$. 

### Running The Programs
We can execute these files directly using the JAX evaluator. Let's see what happens:

```bash
# Run test 1
python3 tests/test_jax.py examples/test_1.ic --steps=10

# Output Context:
Initial Term:
! &0{a2,b3} = a;
λa.λb.(a2 (b3 b))
...
Final Term:
! &0{a2,b3} = a;
λa.λb.(a2 (b3 b))
WORK: 0 interactions
```

```bash
# Run test 3
python3 tests/test_jax.py examples/test_3.ic --steps=10

# Output Context:
Initial Term:
! &0{a2,b3} = a;
! &0{a4,b5} = b3;
λa.λb.(a2 (a4 (b5 b)))
...
Final Term:
! &0{a2,b3} = a;
! &0{a4,b5} = b3;
λa.λb.(a2 (a4 (b5 b)))
WORK: 0 interactions
```

### Why are there 0 Reduction Steps?
Notice that both programs finish with **0 interactions** (0 work steps). Why didn't they reduce?

In Interaction Calculus, a "reduction step" occurs only when two adjacent active ports face each other (an **Active Pair** or **Redex**). The most common redex is an Application node facing a Lambda (`APP-LAM`). 

Because both `test_1.ic` and `test_3.ic` are merely **unapplied function definitions**, there are no application nodes actively applying arguments to them. They are structurally completely resolved and exist in **Normal Form** immediately upon parsing. Thus, the JAX vectorized evaluation sweeps the array, finds `0 redexes`, and halts execution without mutating the graph structure.

### The True "Reduction": Automatic Linearity Parsing
While evaluating the graphs costs 0 interactions, there is something very important happening under the hood during the **parsing step**, specifically for `test_3.ic`.

Interaction Calculus enforces **Affine Logic / Strict Linearity**: every variable must be used *exactly once*. 
- In `test_1.ic`, the author explicitly wrote the duplication node: `!&0{f0,f1}=f;` to split the variable `f` into two linear variables `f0` and `f1`. Both are used once in `(f0 (f1 x))`.
- In `test_3.ic`, the author purposely broke linearity by typing `λf.λx.(f (f (f x)))`. The variable `f` is used 3 times without any explicit duplicators!

When `tensoric/parser.py` reads `test_3.ic`, it automatically detects this affine violation. It dynamically "reduces" the syntax tree and inserts a cascade of duplication networks:
1. It splits `f` into `a2` and `b3` 
2. It takes `b3` and splits it again into `a4` and `b5`!

The parsed output immediately transforms into the strictly linear IC format:
`! &0{a2,b3} = a; ! &0{a4,b5} = b3; λa.λb.(a2 (a4 (b5 b)))`

### Summary
- Both programs are Church Numerals.
- They evaluate to themselves in **0 steps** because they are in Normal Form (no applications to reduce).
- `test_3.ic` serves as an integration test to prove that the `.ic` parser successfully detects non-linear variable applications and automatically inserts valid `dup` cascades (`!&`) before handing the array over to JAX.
