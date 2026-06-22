# Spec: Akita Fused Increment PIOP

| Field       | Value              |
|-------------|--------------------|
| Author(s)   | @quangvdao         |
| Created     | 2026-06-22         |
| Status      | proposed           |
| PR          |                    |

## Summary

The current modular lattice path already has a fused increment design, but it is the earlier `IncByte(j)` plus `IncSign` signed-magnitude design from `05-onehot-increments.md`.
That design is sound, but it is not the optimal Akita PIOP: it fixes the chunk width to bytes in names and layout, keeps the sign as a separate signed-magnitude view, and places fused-increment translation in the current Stage 6 lattice extension.

This spec supersedes the modular `05` increment surface.
The target lattice/Akita PIOP uses one fused signed increment value `Inc`, commits an unsigned-offset decomposition `UnsignedInc = Inc + 2^64`, opens lower `UnsignedIncChunk(j)` one-hot chunks plus one size-`T` `UnsignedIncMsb`, and inserts a lattice-only increment virtualization stage between Stage 5 and Stage 6.

The curve/Dory PIOP remains unchanged.
This spec is written for the modular architecture:

```text
jolt-claims   owns protocol IDs, formulas, packed-family names, and validity requirements.
jolt-verifier owns proof shape, stage orchestration, verifier checks, and tamper tests.
jolt-akita    owns packed physical layout, view encoding, and Akita statement construction.
jolt-openings owns the generic batch-opening abstraction and should not need a new trait.
```

The `jolt-core` prover integration is a separate line of work.
When that prover line is updated, it should produce proofs matching the modular verifier semantics specified here.

## Intent

### Goal

Refactor the modular lattice PIOP from the implemented byte/sign fused increment surface to the unsigned-offset fused increment surface, while preserving the existing curve/Dory protocol and proof shape.

The protocol-level names are:

- `Inc(t)`: the signed fused increment at cycle `t`.
- `UnsignedInc(t) = Inc(t) + 2^64`: the nonnegative 65-bit value committed through the Akita one-hot path.
- `UnsignedIncChunk(j)`: the `j`-th lower chunk of `UnsignedInc`, where each chunk is a one-hot polynomial over `K = 2^log_k_chunk` values and the cycle domain.
- `UnsignedIncMsb`: the top bit of `UnsignedInc`, represented as a size-`T` boolean polynomial, not as a size-`K * T` one-hot polynomial.

Use `UnsignedIncChunk(j)` rather than `UnsignedIncByte(j)` in code and prose.
The chunk width is feature-configured by `log_k_chunk`, and is 4 bits for small traces and 8 bits for large traces.
`Byte` would be inaccurate for the 4-bit path.

### Invariants

- Curve/Dory keeps the current PIOP: dense committed `RamInc` and `RdInc`, the current `IncClaimReduction`, current stage numbering, current Stage 8 dense increment openings, and current ZK behavior.
- Lattice/Akita is the only path that changes.
- The modular verifier's runtime `JoltProtocolConfig` remains the path selector.
  No new PCS trait or new opening abstraction is required.
- In lattice mode, `RamInc` and `RdInc` remain logical Jolt claims, but they are not physical Akita `W_pack` families and are not opened as dense Akita witness polynomials.
- The previous modular `IncByte(j)` and `IncSign` packed families are replaced for base increments by `UnsignedIncChunk(j)` and `UnsignedIncMsb`.
- In lattice mode, `Inc(t)` is defined by:

```text
Inc(t) = RamInc(t) if Store(t) = 1
Inc(t) = RdInc(t)  if Store(t) = 0
```

- Equivalently, the protocol proves:

```text
RamInc(t) = Store(t) * Inc(t)
RdInc(t)  = (1 - Store(t)) * Inc(t)
```

- `Store(t)` is `CircuitFlags::Store`, the bytecode-backed RAM-write flag defined in `crates/jolt-riscv/src/flags.rs`.
- A load has `Store(t) = 0`; its RAM increment is zero and its register increment, if any, flows through `Inc(t)`.
- A store has `Store(t) = 1`; its register increment is zero and its RAM increment flows through `Inc(t)`.
- A zero-delta store is allowed: `Store(t) = 1` does not imply `Inc(t) != 0`.
- A non-store cycle with no register write is allowed: `Store(t) = 0` and `Inc(t) = 0`.
- `UnsignedInc(t)` is always `Inc(t) + 2^64`.
- `UnsignedInc(t)` is reconstructed as:

```text
UnsignedInc(t)
  = lower_chunks_value(t) + 2^64 * UnsignedIncMsb(t)
```

- `UnsignedIncMsb` is boolean over the cycle domain.
- `UnsignedIncMsb` must have exactly `T` coefficients and must not be packed as a one-hot size-`K * T` polynomial.
- Each `UnsignedIncChunk(j)` is one-hot over the address chunk domain for every cycle.
- The lower chunks and `UnsignedIncMsb` together encode the same `UnsignedInc` value used by the fused increment virtualization relation.
- Every lattice prover claim formula and verifier claim formula must use the same opening points and batching coefficients.
- If Akita ZK support is added later, the BlindFold constraints for the new Akita stage must match the non-ZK claim formulas exactly.
  That work is out of scope for this spec because lattice/Akita mode is transparent-only in the modular verifier.

New direct-evaluation invariants are useful follow-up work but are not required for the first modular verifier implementation.
Good candidates are:

- `akita_fused_inc_trace_equivalence`: for generated trace cycles, check that the fused `Inc`, `UnsignedInc`, `UnsignedIncChunk`, and `UnsignedIncMsb` witnesses reconstruct the old `RamInc` and `RdInc` witnesses.
- `akita_fused_inc_stage_claims`: compare a direct evaluator for the fused increment virtualization relation against the prover and verifier parameter formulas.

### Security Argument

This is a relative soundness argument for the Akita PIOP change, not a standalone proof of the full Jolt zkVM.
It assumes the existing Jolt PIOP is sound for the Dory dense-increment relation, the sumcheck protocol is sound under Fiat-Shamir, bytecode read-RAF correctly binds bytecode-derived virtual claims to the committed bytecode table, the one-hot booleanity and hamming-weight reductions are sound, and the PCS is binding for all committed polynomial openings.

The target statement is:

```text
If a lattice/Akita proof verifies, then every accepted opening claim about the old
dense increment witnesses is consistent with one signed Inc polynomial and
one bytecode-backed Store selector, and every accepted opening claim about
the committed unsigned increment chunks reconstructs UnsignedInc = Inc + 2^64.
```

The argument proceeds in four steps.

First, `IncVirtualization` preserves the old RAM and register increment semantics at all verifier-challenged points.
Its input claim is the same batched scalar that the current modular lattice path obtains from the old logical `RamInc` and `RdInc` claims.
The new relation expands those dense claims as:

```text
RamInc(t) = Store(t) * Inc(t)
RdInc(t)  = (1 - Store(t)) * Inc(t)
```

By sumcheck soundness, a prover that changes this batched claim without changing the corresponding multilinear relation can succeed only with negligible probability.
At the final challenge `r_cycle_inc`, the verifier records `Inc(r_cycle_inc)` and `Store(r_cycle_inc)`, and the expected output claim is the multilinear evaluation of the same relation at that point.
Thus the old increment claims consumed by stages 2 through 5 are reduced to claims about `Inc` and `Store`.

Second, bytecode read-RAF binds the selector to the instruction table.
`Store(r_cycle_inc)` is not trusted as an unconstrained witness value.
Stage 6 adds a bytecode Val claim for `CircuitFlags::Store` at `r_cycle_inc`, and bytecode read-RAF proves that this claim is the value in the committed bytecode row selected by the execution trace.
Under the existing bytecode read-RAF soundness assumption, the prover cannot choose `Store` independently of the bytecode flag table.
Therefore the selector used in `IncVirtualization` is the same store flag enforced by the R1CS and bytecode constraints.

Third, `UnsignedIncClaimReduction` preserves the offset relation between signed and unsigned increment views.
Its input claim is:

```text
UnsignedInc(r_cycle_inc) = Inc(r_cycle_inc) + 2^64
```

and its sumcheck relation is:

```text
sum_t eq(r_cycle_inc, t) * UnsignedInc(t).
```

By sumcheck soundness, the accepted output claim `UnsignedInc(r_cycle_6)` is consistent with the same committed unsigned-increment witness polynomial used by the later chunk checks.
This is the bridge from the signed virtual relation to the unsigned committed representation.

Fourth, Stage 6 and Stage 7 bind the unsigned representation to the committed chunks.
Stage 6 booleanity constrains each lower `UnsignedIncChunk(j)` to be boolean over the address-plus-cycle domain, and Stage 7 hamming checks constrain each lower chunk to be one-hot over the address chunk domain at `r_cycle_6`.
Stage 6b constrains `UnsignedIncMsb` to be boolean over the cycle domain.
Stage 7 then proves:

```text
UnsignedInc(r_cycle_6) - 2^64 * UnsignedIncMsb(r_cycle_6)
  = sum_j weight_j * sum_a identity(a) * UnsignedIncChunk(j)(a, r_cycle_6).
```

Because the lower chunks are one-hot and boolean, the right-hand side is exactly the lower base-`K` value encoded by the chunk witnesses.
Because `UnsignedIncMsb` is boolean and opened as a size-`T` polynomial, the left-hand side is exactly the lower 64-bit portion of the accepted unsigned increment claim.
The Stage 8 PCS opening proof then binds these values to the committed Akita witness polynomials.

Together, these reductions show that replacing the implemented modular byte/sign fused increment surface with one fused `UnsignedInc` decomposition does not give the prover a new degree of freedom, except with the same negligible soundness error as the underlying sumchecks, Fiat-Shamir challenges, bytecode read-RAF, and PCS openings.

The remaining proof gaps are outside this spec:

- There is no fresh formal proof here of the entire Jolt PIOP or zkVM soundness theorem.
- This spec assumes the existing RAM and register read-write checks correctly produce the old dense `RamInc` and `RdInc` claims consumed by `IncVirtualization`.
- This spec assumes bytecode read-RAF already soundly binds bytecode Val claims to the final bytecode table.
- This spec assumes the existing batched-sumcheck suffix convention for shorter instances is sound and implemented symmetrically by prover and verifier.
- This spec does not prove Akita PCS binding from first principles.
- This spec does not cover Akita ZK mode because lattice/Akita mode is transparent-only in the modular verifier.

### Non-Goals

- Do not change the curve/Dory PIOP.
- Do not change Dory proof semantics or Dory ZK behavior.
- Do not change Dory Stage 6, Stage 7, Stage 8, or BlindFold behavior.
- Do not keep a compatibility path for the old modular `IncByte(j)` plus `IncSign` base increment surface.
  The lattice path should fully cut over to `UnsignedIncChunk(j)` plus `UnsignedIncMsb`.
- Do not implement the `jolt-core` Akita prover path in this spec.
  The verifier/protocol surface should be correct first and serve as the oracle for prover work.
- Do not change RISC-V instruction semantics, `CircuitFlags`, RAM access semantics, or register write semantics.
- Do not require Akita ZK support in this work.

## Evaluation

### Acceptance Criteria

Checked items have landed in the modular verifier branch.
Unchecked items are still required before the full verifier/protocol side of this spec is complete.

- [x] Curve/Dory final opening order still includes dense logical `RdInc` and `RamInc`, and curve/Dory proof verification does not know about `UnsignedIncChunk` or `UnsignedIncMsb`.
- [x] Lattice `jolt-claims` packed-family IDs include `UnsignedIncChunk { index }` and `UnsignedIncMsb` for base increments.
- [x] Lattice base-increment formulas no longer use `IncByte { index }`, `IncSign`, `FUSED_INCREMENT_BYTE_LIMBS`, signed-magnitude decode, canonical-zero, inactive-zero, or inactive-source-link relations.
- [x] `FieldRdIncByte` and `FieldRdIncSign` remain separate field-inline auxiliary families and are not confused with base Jolt increments.
- [x] Lattice Stage 8 does not request a dense Akita opening for logical `RamInc` or logical `RdInc`.
- [x] Stage 5 remains responsible for instruction read-RAF and register val evaluation.
- [x] In lattice mode, `RamRaClaimReduction` moves from Stage 5 into the new increment virtualization stage.
- [x] The new modular increment stage proves `IncVirtualization` and outputs claims for `Inc(r_cycle_inc)` and `Store(r_cycle_inc)`.
- [x] Stage 6 bytecode read-RAF validates `Store(r_cycle_inc)` against the committed bytecode row's `CircuitFlags::Store` value before later stages rely on it.
- [x] Lattice Stage 6 no longer carries dense `IncClaimReduction` output claims for logical `RamInc`/`RdInc`; dense increment reduction remains curve/Dory-only.
- [x] Stage 6b reduces `Inc(r_cycle_inc) + 2^64` to `UnsignedInc(r_cycle_6)`.
- [x] Stage 6a/6b booleanity includes lower `UnsignedIncChunk(j)` and size-`T` `UnsignedIncMsb`.
- [x] Stage 7 reconstructs `UnsignedInc(r_cycle_6)` from lower `UnsignedIncChunk(j)` openings and the size-`T` `UnsignedIncMsb(r_cycle_6)` opening.
- [x] Stage 8 opens each lower `UnsignedIncChunk(j)` at its Stage 7 address-plus-cycle point.
- [x] Stage 8 opens `UnsignedIncMsb` at the Stage 6b cycle point with `num_vars = log_T`; it is not represented as an address-plus-cycle one-hot family.
- [x] Existing curve/Dory standard and curve/Dory ZK `muldiv` tests keep passing.
- [ ] Modular lattice verifier/tamper tests reject missing or tampered increment virtualization, store binding, unsigned reduction, chunk reconstruction, and MSB opening claims.
- [ ] When a `jolt-core` Akita prover exists, its e2e proofs must satisfy this modular verifier surface; that e2e is not required for the verifier-only slice.

### Testing Strategy

Run the modular protocol/verifier checks first:

```bash
cargo fmt -q
cargo nextest run -p jolt-claims --cargo-quiet
cargo nextest run -p jolt-akita --cargo-quiet
cargo nextest run -p jolt-openings --cargo-quiet
cargo nextest run -p jolt-verifier --cargo-quiet --features akita,core-fixtures
```

Run curve/Dory correctness and ZK regression tests:

```bash
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk
```

When the prover integration line exists, add its Akita prover e2e command here.
That command is intentionally not specified by this verifier/protocol spec.

Run strict linting for the supported feature combinations:

```bash
cargo clippy --all --features host -q --all-targets -- -D warnings
cargo clippy --all --features host,zk -q --all-targets -- -D warnings
cargo fmt -q
```

Add targeted unit tests for:

- `IncVirtualization` input/output claim formulas in `jolt-claims`,
- `UnsignedIncClaimReduction` input/output claim formulas in `jolt-claims`,
- packed-family layout/digest changes in `jolt-akita`,
- Stage 8 partitioning so precommitted bytecode source facts remain separate openings,
- signed `Inc` derivation helper behavior for future prover fixtures,
- `UnsignedInc = Inc + 2^64`,
- lower chunk extraction for `log_k_chunk = 4` and `log_k_chunk = 8`,
- `UnsignedIncMsb` extraction as a single cycle-domain bit,
- reconstruction of `UnsignedInc` from chunks plus `UnsignedIncMsb`,
- zero-delta stores,
- loads with register increments and zero RAM increments,
- non-store cycles with zero register increments.

### Performance

The expected large-trace Akita packed-polynomial count changes from the implemented byte/sign fused increment layout to the unsigned-offset chunk layout.
For `log_k_chunk = 8`, the preferred packed layout orders fixed-width families first:

```text
16 InstructionRa
 8 UnsignedIncChunk
at most 4 RamRa
at most 3 BytecodeRa
 1 shared slot for UnsignedIncMsb plus smaller advice/bytecode material
```

The target is to fit the dominant Akita packed main commitment into 32 size-`256 * T` lanes for large traces.
`UnsignedIncMsb` must not consume a full size-`256 * T` lane by itself.
The RAM and bytecode counts above are practical upper estimates for the current presets, not protocol constants.
If a future parameter choice exceeds those estimates, the Akita layout may add padding lanes.

The new intermediate stage adds proof bytes and transcript work.
That cost is acceptable if it removes the duplicate increment one-hot family and reduces Stage 5 memory pressure by moving `RamRaClaimReduction`.

Useful follow-up evaluation objectives:

- an Akita main commitment lane-count objective,
- modular verifier proof-size accounting for the new increment virtualization proof,
- an Akita prover-time benchmark for `muldiv_e2e_akita` once the prover integration exists,
- an Akita peak-memory objective around stages 5 through 7 once the prover integration exists.

## Design

### Architecture

The curve/Dory path remains structurally unchanged:

```text
curve/Dory
  Stage 5: InstructionReadRaf + RamRaClaimReduction + RegistersValEvaluation
  Stage 6b: existing IncClaimReduction over dense RamInc/RdInc
  Stage 7: existing HammingWeightClaimReduction
  Stage 8: opens dense RamInc/RdInc from IncClaimReduction
```

The modular lattice/Akita path cuts over from the implemented byte/sign fused increment surface to the unsigned-offset surface:

```text
lattice/Akita
  Stage 5:   InstructionReadRaf + RegistersValEvaluation
  Stage 5i:  RamRaClaimReduction + IncVirtualization
  Stage 6a:  BytecodeReadRafAddressPhase + BooleanityAddressPhase
             including lower UnsignedIncChunk
  Stage 6b:  BytecodeReadRaf cycle phase validates Store(r_cycle_inc)
             + Booleanity cycle phase including lower UnsignedIncChunk
             + UnsignedIncMsb booleanity
             + UnsignedIncClaimReduction
             + existing non-increment reductions
  Stage 7:   HammingWeightClaimReduction for non-increment RA families
             + lower UnsignedIncChunk address reduction and value link
  Stage 8:   opens lower UnsignedIncChunk at Stage 7 points
             + opens UnsignedIncMsb at Stage 6b cycle point
```

In prose, call the inserted step the increment virtualization stage.
Use `Stage 5i` only as a compact stage label and avoid `5.5` in code identifiers.

In the modular proof model, add this as optional lattice-only proof and claim data, following the existing optional lattice proof-field pattern.
Curve/Dory configs require the field to be absent or `None`; lattice configs require it to be present.

### Existing Code Anchors

The current modular implementation is concentrated in:

- `specs/akita/05-onehot-increments.md`, which defines the byte/sign fused increment design superseded by this spec.
- `crates/jolt-claims/src/protocols/jolt/formulas/lattice.rs`, which currently names `IncByte { index }`, `IncSign`, `FUSED_INCREMENT_BYTE_LIMBS`, signed-magnitude decode, canonical-zero validity, fused translation, source-link, inactive-zero, and inactive-source-link formulas.
- `crates/jolt-claims/src/protocols/jolt/ids.rs`, which currently names the fused-increment relation IDs and challenge IDs consumed by Stage 6.
- `crates/jolt-verifier/src/config.rs`, where lattice mode requires `IncrementCommitmentMode::FusedOneHot`.
  The enum may keep that name, but its lattice meaning changes to the unsigned-offset PIOP.
- `crates/jolt-verifier/src/proof.rs`, where `JoltStageProofs` has no increment virtualization proof and `ClearProofClaims` has no increment virtualization claim payload.
- `crates/jolt-verifier/src/stages/stage6/inputs.rs` and `inputs_b.rs`, where current fused-increment output claims live under Stage 6.
- `crates/jolt-verifier/src/stages/stage6/verify.rs`, where the current fused translation, source-link, inactive-zero, and inactive-source-link checks are wired.
- `crates/jolt-verifier/src/stages/stage7/**`, where lower increment chunk hamming/value reconstruction must be added.
- `crates/jolt-verifier/src/stages/stage8/verify.rs`, where `akita_fused_increment_entries` currently opens magnitude/sign and bytecode source-link material.
- `crates/jolt-akita/src/layout.rs` and `crates/jolt-akita/src/views.rs`, where packed physical family IDs and view encodings currently include `IncByte` and `IncSign`.
- `crates/jolt-openings/src/packed_linear.rs` and `crates/jolt-openings/src/schemes.rs`, which should remain reusable without adding a new batching trait.

### Logical And Packed Surface

In lattice mode, replace the base increment packed families:

```text
IncByte(j)
IncSign
```

with:

```text
UnsignedIncChunk(j)
UnsignedIncMsb
```

This is not a change to the base logical `JoltCommittedPolynomial::RamInc` and `JoltCommittedPolynomial::RdInc` names.
Those names remain logical claims used by curve/Dory and by the lattice virtualization input.
The change is that lattice/Akita no longer resolves those logical increment claims through the old signed-magnitude `IncByte`/`IncSign` packed views.

For each cycle:

```text
store = CircuitFlags::Store(cycle)
ram_inc = post_ram - pre_ram if RAMAccess::Write else 0
rd_inc = post_rd - pre_rd if rd_write exists else 0
Inc = store * ram_inc + (1 - store) * rd_inc
UnsignedInc = Inc + 2^64
UnsignedIncChunk(j) = j-th lower chunk of UnsignedInc
UnsignedIncMsb = bit 64 of UnsignedInc
```

The future prover implementation should derive `store` from the final proof-facing instruction row, not from ad hoc tracer-side RAM access shape.
The witness helper may use trace RAM/register values for `ram_inc` and `rd_inc`, but the verifier binds the selector to bytecode through Stage 6 bytecode read-RAF.

In `jolt-akita`, `UnsignedIncMsb` must be represented as a trace-domain bit family with `2 * T` cells, not as a `K * T` one-hot chunk family.

### Increment Virtualization Stage

Add a lattice-only stage between current Stage 5 and Stage 6a.
Call the relation `IncVirtualization`.

This stage is a batched sumcheck with `max_rounds = log_T`.

The stage contains:

- `RamRaClaimReduction`, moved from Stage 5 for Akita only.
- `IncVirtualization`, a degree-3 cycle-only relation.

The fused increment input claim is:

```text
v1 + gamma * v2 + gamma^2 * w1 + gamma^3 * w2
```

where:

```text
v1 = RamInc(r_cycle_stage2)
v2 = RamInc(r_cycle_stage4)
w1 = RdInc(s_cycle_stage4)
w2 = RdInc(s_cycle_stage5)
```

The sumcheck relation is:

```text
sum_t Inc(t) * (
    (eq(r_cycle_stage2, t) + gamma * eq(r_cycle_stage4, t)) * Store(t)
  + gamma^2 * (eq(s_cycle_stage4, t) + gamma * eq(s_cycle_stage5, t)) * (1 - Store(t))
)
```

At the final cycle point `r_cycle_inc`, this stage caches:

```text
Inc(r_cycle_inc)
Store(r_cycle_inc)
```

The output claim for `IncVirtualization` is computed from the cached `Inc(r_cycle_inc)` and `Store(r_cycle_inc)` claims:

```text
Inc(r_cycle_inc) * (
    (eq(r_cycle_stage2, r_cycle_inc) + gamma * eq(r_cycle_stage4, r_cycle_inc)) * Store(r_cycle_inc)
  + gamma^2 * (eq(s_cycle_stage4, r_cycle_inc) + gamma * eq(s_cycle_stage5, r_cycle_inc)) * (1 - Store(r_cycle_inc))
)
```

### Stage 6 Changes For Akita

Stage 6 must validate the `Store(r_cycle_inc)` claim produced by the increment virtualization stage and reduce the signed increment claim into an unsigned increment claim at the Stage 6b cycle point.

In lattice mode only:

- Extend bytecode read-RAF parameter generation to include one additional staged bytecode claim for `CircuitFlags::Store`.
- The additional bytecode Val polynomial is:

```text
Val_inc(k) = CircuitFlags::Store(k)
```

- Its opening point is the increment virtualization stage cycle point `r_cycle_inc`.
- Its claimed value is `Store(r_cycle_inc)`.
- The Stage 6 bytecode read-RAF input claim includes this new staged claim with a fresh batching coefficient.
- The prover computes this Val column from the final bytecode row's circuit flags.
- The verifier computes the same claim formula from the opening accumulator and the same transcript-derived batching coefficient.

Stage 6a and Stage 6b booleanity include the lower `UnsignedIncChunk(j)` polynomials.
They use the same split address/cycle booleanity structure as the current one-hot RA families:

```text
0 = sum_{a,t} eq(r_addr_bool, a) * eq(r_cycle_6, t)
      * sum_j gamma_j * (UnsignedIncChunk(j)(a,t)^2 - UnsignedIncChunk(j)(a,t))
```

Stage 6b also includes `UnsignedIncMsb` booleanity over the cycle domain:

```text
0 = sum_t eq(r_cycle_6, t)
      * (UnsignedIncMsb(t)^2 - UnsignedIncMsb(t))
```

This can be implemented as a separate Stage 6b sumcheck instance or folded into the Booleanity cycle implementation.
The implementation should prefer the shape that reuses generated `UnsignedInc` witness data and avoids a second pass over the trace solely for `UnsignedIncMsb`.

Stage 6b replaces the current lattice use of `IncClaimReduction` plus byte/sign fused translation with `UnsignedIncClaimReduction`.
Its input claim is:

```text
UnsignedInc(r_cycle_inc) = Inc(r_cycle_inc) + 2^64
```

Its relation is:

```text
sum_t eq(r_cycle_inc, t) * UnsignedInc(t)
```

where:

```text
UnsignedInc(t) = Inc(t) + 2^64
```

The relation is cycle-only and active on the Stage 6b cycle rounds.
At the final Stage 6b cycle point `r_cycle_6`, it caches:

```text
UnsignedInc(r_cycle_6)
UnsignedIncMsb(r_cycle_6)
UnsignedIncChunk(j)(r_addr_bool, r_cycle_6) for each lower chunk j
```

Stage 6 still handles the existing bytecode read-RAF, RAM hamming booleanity, RAM RA virtualization, instruction RA virtualization, advice reductions, bytecode reductions, and program-image reductions.
The only Stage 6 changes are the added `Store(r_cycle_inc)` bytecode claim, lower `UnsignedIncChunk(j)` booleanity, `UnsignedIncMsb` booleanity, and `UnsignedIncClaimReduction`.

### Stage 7 Changes For Akita

Stage 7 uses the Stage 6b cycle point `r_cycle_6` for lower increment chunks.
Non-increment RA families also keep the current Stage 6 cycle point.

For non-increment RA families, Stage 7 remains the existing hamming-weight and address-reduction relation.

For lower `UnsignedIncChunk(j)`, Stage 7 proves both one-hot hamming and value reconstruction at `r_cycle_6`.
The hamming claim for every lower chunk is public value `1`:

```text
sum_a UnsignedIncChunk(j)(a, r_cycle_6) = 1
```

The lower-value claim is:

```text
lower_value_claim
  = UnsignedInc(r_cycle_6) - 2^64 * UnsignedIncMsb(r_cycle_6)
```

Stage 7 proves:

```text
lower_value_claim
  = sum_j weight_j * sum_a identity(a) * UnsignedIncChunk(j)(a, r_cycle_6)
```

where `weight_j` is the positional base-`K` weight for chunk `j`.
Equivalently, for:

```text
G_j(a) = sum_t eq(r_cycle_6, t) * UnsignedIncChunk(j)(a,t)
```

Stage 7 batches the increment chunk terms:

```text
sum_j alpha_j * sum_a G_j(a)
+ sum_j beta_j * sum_a eq(r_addr_bool, a) * G_j(a)
+ delta * sum_j weight_j * sum_a identity(a) * G_j(a)
```

with input claim:

```text
sum_j alpha_j
+ sum_j beta_j * UnsignedIncChunk(j)(r_addr_bool, r_cycle_6)
+ delta * lower_value_claim
```

Stage 7 reduces the lower chunk claims to openings:

```text
UnsignedIncChunk(j)(r_addr_stage7_inc, r_cycle_6)
```

`UnsignedIncMsb` does not participate in Stage 7 address reduction.
It is already a size-`T` polynomial and is opened directly at `r_cycle_6` in Stage 8.

### Stage 8 Changes For Akita

In lattice mode, Stage 8 opens:

- every non-increment committed polynomial required by the existing protocol,
- every lower `UnsignedIncChunk(j)` at its Stage 7 address-plus-cycle point,
- `UnsignedIncMsb` at the Stage 6b cycle point with `num_vars = log_T`.

In lattice mode, Stage 8 must not open:

- dense `RamInc`,
- dense `RdInc`,
- `RdIncRa(j)`,
- `RdIncMsb`,
- `RamIncRa(j)`,
- `RamIncMsb`.

In curve/Dory mode, Stage 8 remains unchanged and still opens dense `RamInc` and dense `RdInc` from the existing increment claim reduction.

### Proof Serialization

Use the modular verifier's unified proof model.
Add optional lattice-only fields rather than cfg-gating a separate core proof type:

```rust
pub stage5_increment_sumcheck_proof: Option<SumcheckProof<F, VC::Output>>,
pub stage5_increment: Option<Stage5IncrementClaims<F>>,
```

The exact field placement can be:

- `JoltStageProofs`: optional `stage5_increment_sumcheck_proof`, matching the existing optional lattice proof-field pattern.
- `ClearProofClaims`: optional `stage5_increment` claim payload.
- `stages/stage5_increment`: new verifier module for typed inputs, outputs, and verification.

Curve/Dory configs require both optional fields to be absent or `None`.
Lattice configs require both optional fields to be present.
This preserves curve/Dory behavior while keeping one modular proof type.

In lattice mode, verifier orchestration is:

```text
stage1
stage2
stage3
stage4
stage5
stage5_increment
stage6a
stage6b
stage7
stage8
```

In curve/Dory mode, verifier orchestration remains:

```text
stage1
stage2
stage3
stage4
stage5
stage6a
stage6b
stage7
stage8
```

The future prover must mirror the same selected order when it emits modular proofs.

### Alternatives Considered

The first alternative was to push the virtualization directly into Stage 5 by substituting `RdInc = (1 - Store) * Inc` inside register val evaluation and adding separate virtualization sumchecks for earlier `RamInc` and `RdInc` claims.
That avoids a new stage, but it makes Stage 5 higher degree, increases surgery in the current highest-memory stage, and makes the bytecode flag dependency harder to isolate.

The second alternative was to keep the current modular byte/sign fused increment decomposition.
That is sound, but it bakes byte-sized chunks into protocol names and layout, requires signed-magnitude canonical-zero machinery, and does not match the intended unsigned-offset Akita PIOP.

The chosen design adds a lattice-only intermediate stage.
It is easier to specify and review, isolates the bytecode-backed `Store` dependency, and cuts over the Akita commitment layout to one fused increment decomposition.

## Documentation

No Jolt book update is required for the first implementation because this is an internal lattice/Akita PIOP change and Akita is not the default SDK path.

Update or add developer-facing documentation under `specs/` if later work exposes lattice/Akita as a supported public proving mode.

## Execution

Implement in this order:

1. `jolt-claims`: add target IDs and formulas.
   Add `UnsignedIncChunk { index }`, `UnsignedIncMsb`, `IncVirtualization`, and `UnsignedIncClaimReduction`.
   Verify with formula tests for challenge binding, input/output openings, and deterministic IDs.

2. `jolt-claims`: retire old base-increment byte/sign formulas.
   Remove or stop using `IncByte`, `IncSign`, signed-magnitude decode, canonical-zero, inactive-zero, inactive-source-link, and current fused source-link formulas for base increments.
   Keep `FieldRdIncByte` and `FieldRdIncSign` for field-inline.
   Verify with compile failures removed and lattice formula tests updated.

3. `jolt-akita`: update packed layout and view encoding.
   Replace base increment physical families with lower `UnsignedIncChunk(j)` and a trace-domain `UnsignedIncMsb`.
   Ensure `UnsignedIncMsb` contributes `2 * T` cells, not `K * T`.
   Verify with layout digest, rank/unrank, lane-count, and `log_k_chunk = 4`/`8` tests.

4. `jolt-verifier`: add the increment virtualization stage surface.
   Add optional `stage5_increment_sumcheck_proof`, optional `Stage5IncrementClaims`, and a `stages/stage5_increment` verifier module.
   Lattice config requires these fields; curve config rejects them.
   Verify with proof-shape/config tests.

5. `jolt-verifier`: move lattice `RamRaClaimReduction` consumption.
   Curve Stage 5 remains unchanged.
   Lattice Stage 5 no longer supplies `RamRaClaimReduction` directly to Stage 6; Stage 5 increment does.
   Verify with stage dependency tests covering both PCS families.

6. `jolt-verifier`: implement `IncVirtualization`.
   The verifier computes the same batched input claim from old logical `RamInc`/`RdInc` claims, verifies the new sumcheck, and records `Inc(r_cycle_inc)` and `Store(r_cycle_inc)`.
   Verify with direct formula tests and tamper tests for `Inc` and `Store`.

7. `jolt-verifier`: update Stage 6.
   Add the `Store(r_cycle_inc)` bytecode read-RAF claim, add lower chunk booleanity, add size-`T` `UnsignedIncMsb` booleanity, and replace current lattice increment reduction with `UnsignedIncClaimReduction`.
   Verify with tamper tests for missing store binding, bad store value, bad chunk booleanity, bad MSB booleanity, and bad unsigned offset.

8. `jolt-verifier`: update Stage 7.
   Lower chunks use `r_cycle_6`; `UnsignedIncMsb` is excluded from address reduction.
   Stage 7 proves both lower-chunk one-hot hamming and lower-value reconstruction.
   Verify with reconstruction and hamming tamper tests.

9. `jolt-verifier`: update Stage 8 Akita opening collection.
   Remove old magnitude/sign fused entries.
   Open lower `UnsignedIncChunk(j)` through `W_pack` and `UnsignedIncMsb` as a size-`T` packed statement.
   Keep precommitted bytecode/trusted-advice openings partitioned by original commitments.
   Verify with Stage 8 manifest tests and precommitted-opening rejection tests.

10. `jolt-openings`: keep the packed batching abstraction unchanged unless Stage 8 exposes a concrete bug.
    If a bug appears, fix it in the generic packed-linear path and verify with Dory blanket batching plus Akita packed-linear tests.

11. Run the modular verifier and curve/Dory regression commands from the testing section.
    Only after those pass should the separate prover line wire `jolt-core` or another prover to emit this proof shape.

## References

- `specs/akita/05-onehot-increments.md`: implemented byte/sign fused increment design superseded by this spec.
- `specs/akita/00-roadmap.md`: modular Akita crate ownership and precommitted-opening policy.
- `crates/jolt-claims/src/protocols/jolt/formulas/lattice.rs`: current lattice packed-family IDs, views, validity requirements, and fused-increment formulas.
- `crates/jolt-claims/src/protocols/jolt/ids.rs`: current Jolt relation IDs, opening IDs, and challenge IDs.
- `crates/jolt-claims/src/protocols/jolt/formulas/committed_openings.rs`: curve/Dory final logical opening order.
- `crates/jolt-verifier/src/config.rs`: runtime protocol config and lattice mode validation.
- `crates/jolt-verifier/src/proof.rs`: modular proof model and optional lattice proof fields.
- `crates/jolt-verifier/src/stages/stage5/**`: current Stage 5 verifier inputs and outputs.
- `crates/jolt-verifier/src/stages/stage6/**`: current Stage 6 verifier inputs, split address/cycle verification, and fused-increment checks.
- `crates/jolt-verifier/src/stages/stage7/**`: current hamming/address-reduction verifier surface.
- `crates/jolt-verifier/src/stages/stage8/verify.rs`: current Akita final-opening partitioning and packed/precommitted opening collection.
- `crates/jolt-akita/src/layout.rs`: packed physical family layout, digest, and cell budget tests.
- `crates/jolt-akita/src/views.rs`: packed physical view encoding.
- `crates/jolt-openings/src/packed_linear.rs`: generic packed-linear batching path.
- `crates/jolt-riscv/src/flags.rs:24-34`: `CircuitFlags::Store`.
