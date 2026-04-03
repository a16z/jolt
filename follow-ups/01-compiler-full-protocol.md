# Compiler Full Protocol Coverage

## What

Make the Protocol IR + compiler passes expressive enough to generate the complete Jolt PIOP (all 7 stages), replacing the hand-coded `jolt_core_module.rs` example with a Protocol definition + `compile()` call.

## Why

The 2000-line hand-built Module is the reference specification for the Jolt protocol. The compiler currently handles simple protocols (single sumcheck, small compositions) but not the full Jolt structure. Until the compiler produces the same Module, we're maintaining two parallel representations.

## Scope

The Protocol IR already supports sumcheck vertices, polynomial declarations, evaluations, challenges, and dimension parameters. What's missing is coverage of the specific patterns used in stages 1–7:

**Stage 1 — Outer Spartan:** Uniskip (domain iteration with extended evaluation) + remaining sumcheck. The IR handles this; the compiler's emit pass needs to produce the correct `Iteration::Domain` and `Iteration::Standard` ops with matching binding orders.

**Stage 2 — Product + RamRW + InstructionClaimReduction + RafEval + OutputCheck:** Multiple sumcheck instances fused into one stage. Claim formulas with `eq_eval` products, point overrides, and output checks referencing prior stage evaluations.

**Stages 3–7:** Shift, RegistersRW, BytecodeRaf, Booleanity, Hamming, increment reductions, advice cycles. Each introduces specific composition patterns (hamming booleanity, RAF virtual sumcheck, etc.) that the IR needs to express.

The key work is:
1. Express each stage's sumcheck compositions in the Protocol IR
2. Verify the compiler's staging, batching, and challenge allocation match the hand-built Module
3. Verify the emitted verifier schedule (claim formulas, output checks) matches
4. Delete `jolt_core_module.rs` once equivalence is confirmed

## How It Fits

The Protocol IR lives in `jolt-compiler/src/ir/`. The compiler passes (validate → analyze → stage → solve → emit) are in `jolt-compiler/src/compiler/`. The hand-built reference is `jolt-compiler/examples/jolt_core_module.rs`.

The equivalence test infrastructure (`jolt-equivalence`) already compares jolt-core output against jolt-zkvm output stage-by-stage. This same approach can compare compiler-generated Modules against the hand-built reference.

## Dependencies

None — this is the root of the critical path.

## Unblocks

Everything downstream: R1CS from Module, BlindFold, wrapper, gnark codegen, composition.
