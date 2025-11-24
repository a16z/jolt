# Scope of Work: Groth16 EVM Verifier for Jolt
---

## 1. The Problem

Jolt currently provides a native verifier that operates in ~1.5B RISC-V cycles (using a typical program with $\log_2 N = 16$, i.e. 65,536 cycles). This cannot be directly verified on-chain due to:

- **Gas costs**: Direct on-chain verification would require expensive field operations and cryptographic primitives not available as EVM precompiles (particularly $\mathbb{G}_T$ exponentiations)
- **Proof size**: Native Jolt proofs are ~71KB (measured: 72,529 bytes for Fibonacci with max_trace_length=65536), consuming significant calldata (~1.16M gas for calldata alone, assuming all non-zero bytes)

[PR #975](https://github.com/a16z/jolt/pull/975) introduces hint mechanisms that reduce verification to ~600M cycles, moving closer to the constraint budget needed for Groth16 conversion.

## 2. The Goal

Convert Jolt's verifier into a Groth16 circuit:

- **On-chain verification**: ~280k gas (Groth16 standard on BN254)
- **Compact proofs**: ~192-260 bytes (vs 71KB native)
- **Automatic sync**: Pipeline adapts to Jolt changes without manual rewrite

The conversion should be accomplished by creating a mechanism that upgrades efficiently the Groth16 verifier as the Rust version updates. In other words, the conversion pipeline must adapt to Jolt changes without full manual rewrite.

## 3. Deliverables

### Phase 1: Research & Validation (Current)

- Cycle count analysis document
- Industry approaches (SP1, Risc0) and tradeoffs
- Viable extraction techniques (runtime introspection vs static parsing)

### Phase 2: Proof of Concept (If Approved)

Validate automatic transpilation feasibility for Jolt verifier components.

**Scope:** Stage 1 verification (Spartan outer sumcheck only). This is a self-contained component that excludes expensive Dory PCS operations.

- Select extraction technique (runtime introspection vs static parsing)
- Extract Stage 1 verification logic from `jolt-core/src/zkvm/dag/`
- Generate corresponding Gnark/Go circuit and measure constraint count
- Validate correctness via differential testing (compare outputs with Rust verifier)
- Assess maintainability: how easy to regenerate when Jolt changes?

### Phase 3: Full Implementation (If PoC Successful)

Production-ready Groth16 wrapper for Jolt verifier.

- Complete extraction pipeline for all verifier components
- Generate full Groth16 circuit and measure total constraints
- Implement proof generation pipeline (Jolt proof → Groth16 proof)
- Deploy Solidity verifier contract
- Validate end-to-end: guest program → Jolt proof → Groth16 proof → on-chain verification

## 4. Additional Context

The Jolt team's ongoing optimization work is making Groth16 conversion tractable. [PR #975](https://github.com/a16z/jolt/pull/975) reduces verification from ~1.5B to ~600M cycles by replacing expensive $\mathbb{G}_T$ exponentiations with hint-based proofs. The plan is to extend this approach to other operations. Potential lattice PCS integration will further reduce verifier complexity and potentially eliminate pairing-based bottlenecks.

We assume these optimizations will bring constraint counts into the feasible range. Our work focuses on the conversion pipeline, not on further verifier optimization.

## 5. Existing Approaches

Both major zkVMs use Gnark for Groth16 circuit compilation:

**Risc0** ([stark2snark](https://github.com/risc0/risc0/tree/main/groth16_proof)):

- Hand-written Circom circuits that verify STARK proofs
- Uses Gnark via `circom-compat` bridge for proving
- Pipeline: STARK proof → JSON → Circom circuit → Gnark proving → on-chain proof
- Works because STARK verification protocol is stable

**SP1** ([verifier](https://github.com/succinctlabs/sp1/tree/dev/crates/verifier)):

- Hand-written Gnark circuit that verifies their recursion protocol
- Uses `gnark-ffi` (Rust → Go FFI) for circuit compilation and proving
- Works because recursion protocol is stable

### **Why Gnark:**

- Faster circuit compilation than Arkworks
- Optimized gadgets for field arithmetic, hashing, elliptic curves
- General-purpose language makes Rust → Go transpilation feasible (vs Circom's limited DSL)

Risc0 and SP1 have stable protocols, but Jolt is in active development. PR #975's hint mechanism and upcoming lattice PCS demonstrate the protocol is still undergoing optimization. Manual rewrite would require updating the Gnark circuit with every protocol change.

## 6. Approach Options

### Option A: Automatic Transpilation

Extract verifier logic from Jolt's Rust code and generate Gnark circuits programmatically. Precedent: [PR #1060's `zklean-extractor`](https://github.com/a16z/jolt/pull/1060) targets Lean4 for formal verification; we'd target Gnark for Groth16 proving instead.

**Pipeline:**

1. Extract verifier logic as intermediate representation (IR)
2. Generate Gnark circuits from IR
3. Regenerate circuits when Jolt updates

**Two extraction techniques:**

1. **Runtime Introspection (zkLean's approach):**
    - Run Jolt code with instrumented field types that capture operations as AST during evaluation
    - Example: `instruction.combine_lookups()` called with `ZkLeanReprField` builds AST instead of computing values
    - Translate AST to Gnark
2. **Static Parsing:**
    - Parse Rust verifier source code
    - Build intermediate representation (IR) from AST
    - Translate IR to Gnark syntax
    - Challenge: Would require resolving Rust's type system (generics, trait bounds, indirect operations)

---

### Option B: Manual Rewrite (SP1/Risc0 Pattern)

Hand-write Gnark circuit that verifies Jolt proofs. Update circuit manually when Jolt changes.

**Implementation:**

1. Write Gnark circuit implementing Jolt verifier logic in Go
2. Create Rust → Gnark FFI (similar to SP1's `gnark-ffi`)
3. Proof pipeline: Serialize Jolt proofs → Gnark circuit → Groth16 proof

---

### Comparison

| Criteria | Option A: Automatic Transpilation | Option B: Manual Rewrite |
| --- | --- | --- |
| Maintenance | Automatic sync with Jolt updates | Manual updates required for each protocol change |
| Development effort | Higher upfront (transpiler tooling) | Lower upfront (direct implementation) |
| Code drift risk | None (generated from source) | High (manual tracking needed) |
| Circuit optimization | Generated code, potentially suboptimal | Hand-tuned, full control over performance |
| Production readiness | Novel approach, needs validation | Proven pattern (SP1, Risc0) |
| Tooling complexity | IR design, extraction pipeline | Standard FFI, straightforward integration |

Given the current evolving nature of the components of Jolt involved in this project, we asses that the code drift risk is significant and inclines our preference for Option A, as reflected in the deliverables section.

---

## 7. Open Questions

1. **Constraint feasibility**: With PR #975 and future optimizations, what's the realistic R1CS constraint count?
2. **Extraction technique preference**: Any intuition on runtime introspection vs static parsing?
    - Runtime introspection: Simpler, proven by zkLean, requires running verifier
    - Static parsing: More complex (Rust type system challenges), captures full structure