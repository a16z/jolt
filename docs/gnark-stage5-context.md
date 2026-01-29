# Stage 5: Register Val Evaluation, RAM Hamming Booleanity, RAM RA Reduction, Lookups Read-RAF

## Overview

Stage 5 contains 4 batched sumchecks:

| Sumcheck | Component | What it verifies |
|----------|-----------|------------------|
| **Register val evaluation** | Registers (Twist) | `f(k,j) = init + Σ Inc · wa` |
| **RAM Hamming booleanity** | RAM (Twist) | RAM addresses are one-hot or zero |
| **RAM ra reduction** | RAM (Twist) | Reduces address polynomial claims |
| **Lookups read-raf** | Instructions (Shout) | Read-after-final for lookup tables |

From the Jolt verifier theory (see `docs/04_Jolt_Verifier_Theory.md`):
- Stage 5 receives claims from Stage 4's output
- Stage 5's output feeds Stage 6's inputs
- COMMITTED claims start appearing (Ra matrices, Inc polynomials)

## RESOLVED: Stage 5 Working (2026-01-29)

Stage 5 passed after fixing a stack overflow in the transpiler.

### Results
```
Assertions: 13 (up from 10 in Stage 4)
Constraints: 1,531,516 (up from 1,048,560)
New constraints: +482,956
Proof size: 164 bytes
Prove time: 3.78s
Verify time: 1.59ms
```

### Bug Fixed: Iterative AST Traversal for Code Generation

**Problem**: The transpiler caused a stack overflow during "Generating Gnark Circuit" step.

**Root Cause**: The `generate_expr` function in `codegen.rs` recursively traversed the AST. With Stage 5's deeper AST (more sumchecks, larger polynomials), the call stack exceeded system limits even with 128MB stack size.

**Fix**: Converted `generate_expr` to an iterative implementation using explicit stacks:

1. **Phase 1 - Post-order traversal**: Build a list of all nodes to visit in post-order (children before parents) using an iterative depth-first traversal
2. **Phase 2 - Generate expressions**: Process nodes in post-order, storing results in `self.generated` HashMap. When processing a node, its children are already available as cached expressions.

### Files Modified

1. **`gnark-transpiler/src/codegen.rs:193-347`**
   - Changed: Replaced recursive `generate_expr` with iterative two-phase implementation
   - Added: `edge_to_gnark_iterative` helper that looks up pre-generated expressions
   - Why: Prevent stack overflow on deep ASTs
   - Before:
     ```rust
     pub fn generate_expr(&mut self, node_id: usize) -> String {
         // Recursive call via edge_to_gnark -> generate_expr
         let l = self.edge_to_gnark(left);
         let r = self.edge_to_gnark(right);
         ...
     }
     ```
   - After:
     ```rust
     pub fn generate_expr(&mut self, root_node_id: usize) -> String {
         // Phase 1: Build post-order traversal
         let mut post_order: Vec<usize> = Vec::new();
         let mut stack: Vec<(usize, bool)> = vec![(root_node_id, false)];
         while let Some((node_id, children_processed)) = stack.pop() {
             // ... iterative traversal
         }

         // Phase 2: Generate expressions in post-order
         for node_id in post_order {
             // Children already in self.generated
             let l = self.edge_to_gnark_iterative(left);
             // ...
         }
     }
     ```

2. **`gnark-transpiler/src/codegen.rs:120-159`** (previous session)
   - Changed: Converted `count_refs` to iterative (similar pattern)
   - Why: First attempt to fix stack overflow, but the actual overflow was in `generate_expr`

## Stage Position in Verification DAG

```
Stage 1 (Spartan outer)
    │ produces Az(r), Bz(r), Cz(r)
    ▼
Stage 2 (Spartan product, RAM raf, Output check, etc.)
    │ produces virtual claims
    ▼
Stage 3 (Spartan shift, Instruction input, Register claim reduction)
    │ produces claims on f(k,j), Inc(j)
    ▼
Stage 4 (Register r/w, RAM val evaluation, RAM val final)
    │ produces claims on ra(r,j), Inc(j)
    ▼
Stage 5 (Register val eval, RAM Hamming, RAM ra reduction, Lookups read-raf)  ← DONE
    │ produces claims on committed polynomials
    ▼
Stage 6 (Bytecode read-raf, RAM ra virtual, Inc reduction, etc.)
    │
    ▼
... continues to Stage 8 (Dory PCS opening)
```

## Technical Details

### Register Val Evaluation

Proves the formula for virtualized register state:
```
f(r_k, r_j) - init(r_k) = Σ_j Inc(j) · wa_rd(r_k, j)
```

### RAM Hamming Booleanity

Verifies RAM addresses encode valid one-hot or zero patterns:
```
hw² - hw = 0  (Hamming weight is 0 or 1)
```

### RAM RA Reduction

Reduces virtual `ra(k, j)` to committed chunks via tensor decomposition:
```
ra(r_k, r_j) = Σ_j eq(r_j, j) · Π_{i=1}^{d} ra_i(r_{k_i}, j)
```

### Lookups Read-RAF

Read-after-final for instruction lookup tables - verifies the lookup table reads are consistent.

## Key Files

### Verifier Implementation
- `jolt-core/src/zkvm/registers/val_evaluation.rs` - Register val evaluation
- `jolt-core/src/zkvm/ram/hamming_booleanity.rs` - RAM Hamming booleanity
- `jolt-core/src/zkvm/ram/ra_reduction.rs` - RAM ra reduction
- `jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs` - Lookups read-raf
- `jolt-core/src/zkvm/transpilable_verifier.rs:392-425` - Stage 5 verification

### Transpilation
- `gnark-transpiler/src/codegen.rs` - Code generation (fixed stack overflow)
- `gnark-transpiler/src/bin/transpile_stages.rs` - Transpilation entry

## Commands

```bash
# Full pipeline with Stage 5
cd /Users/home/dev/parti/cryptography/zkVMs/WonderJolt/jolt && \
cargo run -p fibonacci --release --features transcript-poseidon -- --save 10 && \
cargo run -p gnark-transpiler --bin transpile_stages && \
cd gnark-transpiler/go && go test -v -run TestStages16CircuitProveVerify
```
