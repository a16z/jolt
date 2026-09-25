# Spec: Field-Inline Portability — Packed Commitments and Base-Field Instantiation

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Claude |
| Created | 2026-08-19 |
| Status | implemented |
| PR | #1808 |

## Purpose

Field-inline runs on the modular stack with Dory over BN254 Fr and Akita over
fp128. Both commitment modes commit the full `FieldRdInc` polynomial and open
the reduced claim produced by stage 6b. This spec describes the Akita dense
commitment and the field-specific tracer encoding.

## Scope

```text
in scope:
  packed (Akita) treatment of the field-inline committed surface
  instantiating field-inline over fp128
out of scope:
  extension-field sumcheck soundness (base fields ONLY: field registers hold
    elements of the sumcheck field F itself; any base/extension split is a
    separate spec)
  any change to the Twist identities or the virtual/committed split
  zk over the packed axis (akita x zk stays mutually exclusive)
```

## Shared invariants

- Native-field invariant: q = modulus(F). Field-inline accelerates the field
  used by the proof. Field multiplication remains one guarded constraint row
  plus one product lane, without quotient witnesses.
- The Twist memory-checking identities (`crates/jolt-claims/src/twist/`) are
  representation-agnostic and shared by both commitment modes.
- Field-register RA/WA/Val remain virtual and bytecode-anchored. Field-inline
  adds no packed one-hot commitments.
- `FieldRegistersIncClaimReduction` reduces the read/write and value-evaluation
  claims to one opening of `FieldRdInc`, with the same relation in both modes.

## Akita commitment to FieldRdInc

`FieldRdInc` is the extension's only committed polynomial. Each entry is a
field-register delta, `post - pre` in fp128, and can occupy the full field.
The Akita commitment uses the full-width `Dense` configuration, whose
internal digit decomposition supports arbitrary fp128 values. Jolt supplies
one field value per cycle; it does not split increments into external limb
columns or carry separate limb evaluations in the proof.

Only the field-increment object uses the full-width configuration. Advice
and committed-program objects retain their existing bounded configurations
and digit policies. The joint opening selects the configuration for each
object's role; it does not widen those objects' value bounds.

The field-increment polynomial is an independent dense object, committed in
stage 0 under a transparent setup and opened in the same heterogeneous batch
as the advice and one-hot trace objects. Its `PrecommittedRole` has order 2
and transcript label `field_inc`; the batch order is:

```text
[UntrustedAdvice, TrustedAdvice, FieldInc, OneHotTrace]
```

The role places a trace-derived object in the PCS batch; its presence does
not mean that field increments are fixed during program preprocessing.

`FieldIncLayout` in `crates/jolt-claims/src/protocols/field_inline/lattice/`
owns the physical arity, layout digest, and opening-point padding. For a
trace with `log_T` variables, the physical arity is the larger of `log_T` and
the dense schedule floor. The prover pads the evaluations with zeros to that
arity. The verifier prepends zero coordinates to the reduced opening point,
selecting the original trace polynomial without changing its evaluation.

In stage 8, the stage-6b claim `(v, r)` becomes a single dense opening at the
padded point with value `v`. The PCS binds it directly to the field-increment
commitment. There is no limb recomposition identity, selector challenge, or
additional reconstruction sumcheck. Prover and verifier share this claim
construction in `jolt-verifier/src/stages/stage8/packed.rs`.

The commitment is required on every Akita proof with field-inline enabled,
including traces that execute no field instructions. An all-zero polynomial
is legal because the dense schedule depends on shape rather than content.
The verifier rejects an absent commitment, incompatible layout metadata,
and duplicate field-increment roles in the final batch.

The schedule registry provisions the full-width dense object at the canonical
physical arity and includes it in every field-inline batch profile, alongside
each supported advice subset. Layout and schedule provisioning tests pin the
same sizing law on both sides of the PCS boundary.

## Field-specific execution encoding

Akita proves over fp128. Its feature chain selects the tracer's `ProofField`
as `jolt_field::Prime128OffsetA7F7` and `FieldValueEncoding::ACTIVE` as
`TWO_LIMB_128_CANONICAL`. Dory uses BN254 Fr with
`BN254_SCALAR_CANONICAL`. The tracer's `decode_field` and `encode_field` in
`tracer/src/instruction/field_inline/mod.rs` operate on the selected field.

The encoding is recorded in `FieldInlineBytecodeMetadata` and the instruction
profile fingerprint. A proof or preprocessing artifact using a different
encoding is rejected by the metadata equality check. Field-inline guests
are therefore configuration-specific.

Guest ingress and readout still use u64 limbs because the integer register
file is RV64; these instruction operands are independent of the commitment
representation:

- A full fp128 load uses zero initialization followed by two
  `FIELD_LOAD_ACCUMULATE_FROM_REGISTER` instructions, high limb first. Each
  accumulation computes `old_destination * 2^64 + limb` in the field.
- Canonical readout uses two in-place `FIELD_ADVICE_LIMB` instructions,
  `FIELD_ASSERT_ZERO` on the remaining quotient, and an integer check that
  the emitted value is below the modulus. Accumulating the emitted limbs
  high-to-low restores the consumed source without a scratch field register.

## Validation

`crates/jolt-prover/tests/akita_field_inline_e2e.rs` exercises:

- The `field-ops` guest on reference and optimized kernels, with identical
  proofs combining bounded advice and field increments whose centered
  representatives exceed 64 bits.
- `muldiv` with no field operations, an all-zero `FieldRdInc` commitment, and
  identical proofs on both backends.
- Rejection after changing the reduced increment claim, commitment layout
  digest, or joint opening proof, stripping the commitment, or duplicating
  the field-increment role in the batch.

The shared guest acceptance matrix runs field-inline in clear Dory, ZK Dory,
and Akita modes. Encoding-mismatch tests run under both tracer field
configurations.
