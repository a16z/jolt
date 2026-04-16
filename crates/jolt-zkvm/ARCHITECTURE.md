# Architecture — ML Compiler Philosophy

## One-sentence summary

The compiler lowers protocol knowledge into a flat sequence of primitive ops;
the runtime executes them mechanically without interpreting any protocol data.

## The Analogy

Like an ML compiler (XLA, TVM): a high-level graph (protocol description) is
lowered through optimization passes into a sequence of primitive operations
(matmul, scatter, reduce) that a backend executes without knowing what model
they represent.

```
Protocol (jolt-core)  →  Compiler (jolt-compiler)  →  Module (ops + data)
                                                            │
                                                     Runtime (jolt-zkvm)
                                                            │
                                                     Backend (jolt-cpu)
```

## The Test

**For every Op handler in the runtime, ask:**
1. Could this handler execute correctly if I renamed all the fields?
2. Does this handler branch on the meaning of any data, or just pass it through?
3. Is this handler ≤ 30 LOC?

If any answer is "no", the compiler hasn't lowered far enough.

## What "lowered far enough" means

### Standard sumcheck path (DONE — this is the model)

```
Op::SumcheckRound { kernel, round, bind_challenge }
```
Handler: 15 LOC. Calls `backend.reduce(compiled_kernel, inputs, challenges)`.
The runtime doesn't know what the formula computes. The backend doesn't know
either — it just evaluates the compiled arithmetic.

### Address-decomposition path (CURRENT VIOLATION)

```
Op::InstanceScatter { kernel, phase }     // 220 LOC handler — interprets rules
Op::InstanceDecompReduce { ... }          // 114 LOC — interprets prefix/suffix data
Op::InstanceBindBuffers { ... }           // 37 LOC — iterates protocol buffer IDs
Op::InstanceMaterialize { ... }           // 73 LOC — interprets config fields
Op::UpdateInstanceCheckpoints { ... }     // 20 LOC — interprets checkpoint rules
```
Plus `checkpoint_eval.rs` (1059 LOC): a mini-interpreter for prefix/suffix rules.

### Target: decomp path lowered to primitives

The compiler should emit enough primitive ops that the runtime never interprets
any rules, suffix ops, prefix formulas, or combine matrices. Each handler
should be mechanical dispatch to backend methods or simple buffer operations.

The exact primitive ops are a design decision — the right set will emerge from
the work. But the invariant is clear: **no handler interprets protocol data**.

## Invariants

1. Runtime never imports from jolt-core or jolt-instructions
2. Runtime has zero match arms on protocol-specific enum variants
3. Every handler ≤ 30 LOC
4. `checkpoint_eval.rs` does not exist
5. `LookupTraceData` does not exist in RuntimeState
6. `BytecodeData` does not exist in RuntimeState
7. The only InputBinding variants are generic mathematical operations
8. Transcript bytes are identical before and after (test: transcript_divergence)
