# Spec: Consolidate the jolt-field Trait Surface

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @Acentelles                    |
| Created     | 2026-07-21                     |
| Status      | proposed                       |
| PR          | feat/solinas-field-stack (branch, pre-PR) |

## Summary

The Solinas field stack imported by `feat/solinas-field-stack` brings `crates/jolt-field` to 46 public traits across 82 files, with 15 one-trait-per-file micro-files at the crate root (several under 10 lines). For comparison, arkworks' `ff` + `serialize` crates expose 25 public traits while covering serialization, extension towers up to degree 12, FFT domains, and hash-to-field. A usage audit of this workspace and of akita (the stack's main consumer, pinned to this branch) shows that a large fraction of the surface has no generic consumer anywhere: both signed-accumulator trait families, `ExtensionCoeff`, `ScaleI32`, `BalancedDigitLookup`, `MontgomeryConstants`, and the entire `fft.rs` module (1,100 lines). Three parallel deferred-reduction accumulator designs coexist. This spec consolidates the crate to roughly 22 traits and 4 root modules, adopts the explicit per-type implementation style of `arkworks/bn254.rs`, and switches wire serialization for the Solinas types to serde + bincode. The goal is to minimize the audit surface without losing the layering that lets non-BN254 fields and rings implement only what they have.

This spec supersedes parts of [`unify-field-hierarchy.md`](./unify-field-hierarchy.md) (PR #1484). That spec's core insight is preserved: a slim algebraic ladder (`AdditiveGroup` to `RingCore` to `FieldCore`) with capabilities that are not algebraic descendants of the field marker. What is superseded is the granularity: one trait per capability method proved too fine once the extension-field, packed, and unreduced layers landed on top of it.

## Intent

### Goal

Reduce `crates/jolt-field` from 46 public traits to at most 22 by deleting dead traits and merging single-purpose capability traits into cohesive ones, while keeping every generic algorithm in this workspace and in akita expressible.

Key structural decisions:

- The algebraic ladder stays: `AdditiveGroup` to `RingCore` to `FieldCore`, with `Field` as the Jolt compatibility umbrella.
- One canonical-representation trait (`CanonicalRepr`) replaces the seven byte/introspection/challenge traits.
- One `Accumulator` trait replaces the six accumulator traits that have consumers; the two families with zero consumers are deleted.
- The extension-field surface is `ExtField<F>` (absorbing `LiftBase`, `MulBase`, `FrobeniusExtField`), `MulBaseUnreduced<F>`, `FpExt2Config<F>`, and a single merged `ExtMulBackend`.
- Wire serialization of Solinas types is serde + bincode, following the existing serde implementations on `arkworks/bn254.rs`; the hand-rolled akita-style byte encoding is not adopted. Canonical bytes survive only as the Fiat-Shamir transcript surface.
- Concrete types follow the `arkworks/bn254.rs` pattern: everything a type implements is visible in that type's file, either as an explicit `impl` block or as a one-line invocation of a shared macro. The three `native_algebra.rs` side-files are eliminated.

### Invariants

This is a refactor of trait boundaries, not of arithmetic. The `jolt-eval` invariants relevant to field arithmetic (`SplitEqBindLowHigh`, `SplitEqBindHighLow`, `FieldMulScalar`, `Soundness`) must continue to pass. No new `jolt-eval` invariants are required.

1. **BN254 behavior is byte-identical.** Same Fiat-Shamir stream, same proof bytes for the `muldiv` e2e test in standard mode, same verifier outcomes. `Fr`'s existing serde and byte encodings are untouched.
2. **Every operation currently callable through a jolt-field trait remains available** under the merged trait, same semantics, same `#[inline]` discipline on both trait method and impl (carried over from #1484, invariant 7).
3. **The slim hierarchy remains implementable by a non-BN254 field with no arkworks dependency.** The `mersenne61_compat` test (in `jolt-sumcheck`) keeps compiling and passing against `jolt-field --no-default-features`, with its bound list updated to the merged traits.
4. **Rings that are not fields keep a home.** `RingCore` (no inversion) and `AdditiveGroup` (implemented by wide accumulator types that have no multiplication) survive as distinct layers; nothing forces a cyclotomic ring or an accumulator to claim field capabilities.
5. **Transcript bytes never go through bincode.** Fiat-Shamir absorption and challenge derivation use the explicit canonical little-endian encoding on `CanonicalRepr`; bincode is only the proof/wire format.
6. **No codegen regressions.** All dispatch remains static; merged traits change name resolution, not monomorphization.


### Non-Goals

1. **Akita compatibility in lockstep.** Akita pins a git rev of this fork and will adapt its trait bounds when it next bumps the pin. No companion PR, no intermediate compatibility shims.
2. Changing BN254 proof wire format or any transcript byte stream.
3. Unifying jolt-prover-legacy's own `JoltField` deferred-reduction machinery (`mul_to_product_accum`, `Folded256ProductAccum`) with this crate's. Same idea, third copy, separate migration.
4. Macro-generating the packed SIMD intrinsic bodies. NEON and AVX arithmetic are genuinely different; only their selection boilerplate is consolidated.
5. Removing the temporary `akita` bootstrap feature and `src/akita.rs`; that happens in the final migration PR as already planned.
6. Renaming `Field`; it stays the umbrella name (per the #1484 naming decision).

## Evaluation

### Acceptance Criteria

- [ ] `grep -rc '^pub trait' crates/jolt-field/src` totals at most 22.
- [ ] The crate root has at most 4 trait-defining modules (`algebra.rs`, `canonical.rs`, `accumulator.rs`, `field.rs`) plus the feature-gated backend modules; the 15 micro-files are gone.
- [ ] Zero references remain to: `SignedScalarAccumulator`, `WithSmallScalarAccumulator`, `SignedProductAccumulator`, `WithSignedProductAccumulator`, `ExtensionCoeff`, `BalancedDigitLookup` (trait), `ScaleI32`, `PackedValue`, `LiftBase`, `MulBase`, `FrobeniusExtField`, `FpExt4MulBackend`, `FpExt8MulBackend`, `AdditiveAccumulator`, `Invertible`, `RandomSampling`, `MulPow2`, `MulPrimitiveInt`, `CanonicalBytes`, `ReducingBytes`, `FixedByteSize`, `FixedBytes`, `CanonicalU64`, `CanonicalBitLength`, `TranscriptChallenge`, `SmoothFftField`.
- [ ] `fft.rs` is removed from `jolt-field` (it has zero consumers in this workspace).
- [ ] The three `native_algebra.rs` files are deleted; one shared macro provides the supertrait glue, invoked from each concrete type's own file.
- [ ] `Fp32`, `Fp64`, `Fp128`, `FpExt2`, `FpExt4`, `FpExt8` implement `serde::Serialize`/`Deserialize` with canonical encoding; a bincode round-trip test covers each.
- [ ] The degree-4 extension mul/square schedule exists in exactly one place (shared between scalar and packed backends, as degree-8 already is).
- [ ] `muldiv` e2e passes in both modes; standard-mode proof bytes match main (size and content).
- [ ] Serialized size tests: each Solinas field element bincode-encodes to exactly `NUM_BYTES` bytes; a `Vec` of $n$ elements encodes to $n \cdot \texttt{NUM\_BYTES}$ plus a single length prefix.
- [ ] `mersenne61_compat` passes with updated bounds and still no arkworks dependency.

### Testing Strategy

- Full `cargo nextest run --cargo-quiet`; `muldiv` e2e in `--features host` and `--features host,zk`.
- `cargo clippy --all --features host -q --all-targets -- -D warnings` and again with `host,zk`.
- `jolt-field` standalone under feature combos: `bn254`, `solinas`, `bn254,solinas,parallel`, `--no-default-features`.
- The `solinas_field_arith` fuzz target and both criterion benches must still build.
- Existing in-crate unit tests (prime, ext, packed, unreduced) are updated for renamed bounds but not weakened; deleted traits take their dead tests with them.
- New: bincode round-trip tests for all Solinas types; a compile test that the merged `CanonicalRepr` default challenge derivation matches the previous `TranscriptChallenge` behavior on `Fr` (identical bytes in, identical element out).

### Performance

No regression expected: every merge is a compile-time rename and static dispatch is unchanged. Verify, do not assume:

- Run `solinas_field_arith` and `field_arith` criterion benches before and after; results within noise.
- Proof size: standard-mode `muldiv` proof bytes are compared against main (identical, so identical size); Solinas-side serialized sizes are asserted by the per-element `NUM_BYTES` tests in Acceptance Criteria.
- The existing `jolt-eval` objectives for prover time must not move; no new objectives needed.
- All `#[inline]`/`#[inline(always)]` annotations are preserved through the moves.

## Design

### Architecture

Trait disposition, all 46 accounted for:

**Deleted, no consumer in this workspace or akita (11):**

| Trait | Notes |
|---|---|
| `SignedScalarAccumulator`, `WithSmallScalarAccumulator` | plus `NaiveSignedScalarAccumulator`, `FrSmallScalarAccumulator`, `arkworks/small_scalar_accumulator.rs` |
| `SignedProductAccumulator`, `WithSignedProductAccumulator` | plus `NaiveSignedProductAccumulator`, `FrSignedProductAccumulator`, `arkworks/signed_product_accumulator.rs` |
| `ExtensionCoeff` | single blanket impl; inline the bound at its one use in `fp_ext2` |
| `BalancedDigitLookup` | becomes a free function `balanced_digit_lut<F>(log_basis)` |
| `MontgomeryConstants` | one impl (`Fr`), zero consumers; OPEN: confirm no out-of-tree GPU/Metal consumer before deleting, else move private into `arkworks/` |
| `SmoothFftField` | leaves the crate together with `fft.rs` |
| `ScaleI32`, `PackedValue`, `AdditiveAccumulator`, absorbed below | listed here for completeness of the count |

**Merged (22 traits fold into 7 survivors):**

| Survivor | Absorbs | Rationale |
|---|---|---|
| `FieldCore` | `Invertible`, `RandomSampling` | every `FieldCore` type implements both; rings stay unaffected at `RingCore` |
| `FromPrimitiveInt` (gains `RingCore` supertrait) | `MulPow2`, `MulPrimitiveInt` | the absorbed traits are pure default-method helpers over exactly this bound |
| `CanonicalRepr` (new, one file) | `CanonicalBytes`, `ReducingBytes`, `FixedByteSize`, `FixedBytes<N>`, `CanonicalU64`, `CanonicalBitLength`, `TranscriptChallenge` | one trait: `NUM_BYTES`, `to_bytes_le`, `from_le_bytes_mod_order`, `to_canonical_u64_checked`, `num_bits`, `from_challenge_bytes` (defaulted to reducing decode) |
| `Accumulator` | `AdditiveAccumulator` + `RingAccumulator` | the two are only ever implemented and consumed together (`WideAccumulator`, `NaiveAccumulator<R>`) |
| `ExtField<F>` | `LiftBase<F>`, `MulBase<F>`, `FrobeniusExtField<F>` | identical implementor sets (blanket `F` + `FpExt2/4/8`); Frobenius requires a pseudo-Mersenne base, which all current bases are |
| `ExtMulBackend` | `FpExt4MulBackend` + `FpExt8MulBackend` | same three implementors, same role (per-width fused schedules) |
| `ReduceTo<F>` | `ScaleI32` | same three wide-limb implementors |
| `PackedField` | `PackedValue` | identical 13 implementors; nothing bounds on `PackedValue` alone |

**Kept as-is (14):** `AdditiveGroup`, `RingCore`, `Field` (umbrella), `WithAccumulator` (the additive-layer accumulator association that lets rings use `NaiveAccumulator` without field capabilities, per #1484), `OptimizedMul` (consumed by jolt-prover-legacy; candidate to relocate there in a later sweep), `CanonicalField`, `HalvingField` (implemented by extension fields, so it cannot fold into `CanonicalField`), `PseudoMersenneField`, `MulBaseUnreduced`, `FpExt2Config`, `HasUnreducedOps`, `HasOptimizedFold`, `HasWide`, `HasPacking`.

Final count: 7 survivors + 14 kept = at most 22 public traits (21 if `MontgomeryConstants`' deletion is confirmed and `OptimizedMul` relocates).

**Serialization split.** Two distinct concerns, two mechanisms:

- Proof/wire format: serde + bincode. Solinas types serialize their canonical form (never an internal representation), exactly as `arkworks/bn254.rs` already does for `Fr`. Extension fields serialize as arrays of base-field elements. Nothing replicates akita-serialization's `FpExt2Config`-bound custom encode/decode.
- Fiat-Shamir: `CanonicalRepr`'s explicit little-endian canonical encoding. A transcript needs a specified encoding, not whatever an encoder version emits, so this deliberately does not route through bincode.

**File layout.** Crate root: `algebra.rs` (`AdditiveGroup`, `RingCore`, `FieldCore`, `FromPrimitiveInt`, `OptimizedMul`), `canonical.rs` (`CanonicalRepr`), `accumulator.rs` (`Accumulator`, `WithAccumulator`, `NaiveAccumulator`), `field.rs` (`Field` umbrella). Solinas traits (`CanonicalField`, `HalvingField`, `PseudoMersenneField`) move into `prime/mod.rs`. Each concrete type's file shows its full trait surface bn254.rs-style; shared macros are limited to:

1. one `impl_native_algebra!` macro (replacing the three `native_algebra.rs` files and the 230 hand-written lines in `ext/native_algebra.rs`), invoked inside each type's file;
2. one operator-matrix macro in the style of `bn254.rs`'s `delegate_binop!` and legacy's `impl_field_ops_inline!`, deduplicating the hand-written `Add`/`Sub`/`Mul`/`Neg`/`*Assign`/by-reference blocks in `fp32.rs`, `fp64.rs`, `fp128/`, and the ext types (reduction bodies stay hand-written per type; `fp64`'s `C_SHIFT` specialization is math, not boilerplate);
3. the existing capability macro (`native_capability.rs` contents), with invocations moved into the per-type files.

The three near-identical cfg-cascade blocks selecting `Fp{32,64,128}Packing` in `packed/mod.rs:324-415` collapse into one macro. Packed intrinsic bodies are untouched.

### Alternatives Considered

1. **Keep the fine-grained hierarchy of #1484.** Rejected: at 46 traits the decomposition costs more audit surface than it buys in bound minimalism, and several capabilities it kept orthogonal (`TranscriptChallenge`, `FixedBytes`) have exactly the six field types as implementors, making the orthogonality theoretical. Where #1484's rationale is still live (rings without inversion, additive-only accumulator types, no-arkworks implementability), the layer survives.
2. **One monolithic `JoltField` like jolt-prover-legacy.** Rejected: akita's prover is written against the intermediate layers (`CanonicalField` ~578 bound sites, `ExtField` ~358, `HasWide` ~118), and cyclotomic rings need `RingCore` without field claims. A monolith re-creates the problem #1484 fixed.
3. **Macro-generate whole field types (plonky3-style).** Rejected in favor of the bn254.rs pattern: explicit impls are the audit-friendly choice; macros are confined to operator matrices and supertrait glue where the expansion is mechanical and identical across types.
4. **Merging `TranscriptChallenge` into `CanonicalRepr`** was debated against #1484's warning that binary fields and Plonky-style challengers decode challenges differently. Merged anyway with a defaulted method: if a challenge type that is not a canonical field ever appears, re-split the trait with that first consumer. Recorded here so the re-split is a known escape hatch, not a regression.

## Documentation

Internal refactor of a crate the book does not document in detail. After landing, run `/update-docs` against the base commit to catch any book references to renamed traits; expected changes are nil or a few identifier updates in the architecture chapter.

## Execution

Five phases, each independently green (clippy both modes, nextest, `muldiv` both modes):

1. **Deletions.** Signed-accumulator families, `ExtensionCoeff`, `BalancedDigitLookup` to free fn, `fft.rs` + `SmoothFftField` out, `MontgomeryConstants` (after the GPU check). Drop the corresponding supertraits from `Field`.
2. **Accumulators.** Merge `AdditiveAccumulator` + `RingAccumulator` into `Accumulator`; keep `WithAccumulator` as the association point; implementors unchanged.
3. **Root merges + serde.** `FieldCore` and `FromPrimitiveInt` absorptions; `CanonicalRepr`; serde impls + bincode round-trip tests for all Solinas types; root file consolidation to 4 modules; update `mersenne61_compat` bounds.
4. **Ext cluster.** `ExtField` absorbs lift/mul_base/frobenius; `ExtMulBackend`; dedupe the degree-4 schedule shared with `packed/`.
5. **Layout + macros.** `impl_native_algebra!`, the operator-matrix macro, per-type invocation placement, packed cfg-cascade collapse. Benchmark before/after this phase specifically.

Roughly half the touched traits predate this branch on main (the #1484 layer), so the diff extends beyond the branch's own delta; this is intentional and is what the consolidation is for.

Open question blocking only its own line item: does `MontgomeryConstants` have an out-of-tree GPU/Metal consumer? (Its doc comment says it exists for GPU backends.)

## References

- [`unify-field-hierarchy.md`](./unify-field-hierarchy.md), PR #1484: predecessor spec; this spec supersedes its granularity while preserving its layering and invariants 1, 2, 3, 7.
- `crates/jolt-field/src/arkworks/bn254.rs`: the implementation pattern to follow.
- arkworks `ff` + `serialize` (25 public traits): the external calibration point for trait-count parity.
- akita (`LayerZero-Labs/akita`, formerly hachi): consumer whose generic bound sites define which traits are load-bearing; adapts on its next pin bump.
- `scripts/check-shared-field-identity.sh`: unchanged by this spec.
