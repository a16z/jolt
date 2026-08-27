# Spec: Field-Inline Portability — Packed Commitments and Base-Field Instantiation

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Claude |
| Created | 2026-08-19 |
| Status | draft |
| PR | TBD |

## Purpose

Field-inline v1 is implemented end-to-end on the modular stack for the
homomorphic (Dory) commitment axis over BN254 Fr (see the status note in
[field-inline-protocol.md](field-inline-protocol.md)). Two upgrades are
anticipated by pending work: the Akita packed commitment mode moving to the
modular prover (#1718, #1732), and Jolt moving to a 128-bit base field. This
spec settles both upgrade designs now so the later PRs land against a decided
plan instead of re-deriving it.

## Scope

```text
in scope:
  packed (Akita) treatment of the field-inline committed surface
  instantiating field-inline over a smaller base field (e.g. fp128)
out of scope:
  extension-field sumcheck soundness (base fields ONLY: FR registers hold
    elements of the sumcheck field F itself; any base/extension split is a
    separate spec)
  any change to the Twist identities or the virtual/committed split
  zk over the packed axis (akita x zk stays mutually exclusive)
```

## Invariants that carry unchanged

- Native-field invariant: q = modulus(F). Field-inline accelerates whatever
  field the proof runs over; FMUL stays one guarded row plus one product lane,
  no quotient witnesses, under every instantiation in scope.
- The Twist memory-checking identities (`crates/jolt-claims/src/twist/`) are
  representation-agnostic; both upgrades reuse them verbatim.
- FR RA/WA/Val remain virtual and bytecode-anchored — no packed one-hot
  obligations arise from field-inline on any axis.
- The composition seams (per-stage `field_inline` modules, boundary whitelist
  tests, `suppress_field_operand_slots`) are commitment- and field-agnostic.

## Axis 1: packed (Akita) FieldRdInc

`FieldRdInc` is the extension's only committed polynomial. The packed mode's
inc machinery (balanced digits + carry + booleanity-style digit checks, per
the digit-zero work in #1731) requires small values; a field delta
(post − pre mod p) has no small representation. Design:

```text
commit:   the limb columns of FieldRdInc's canonical representative —
          limb_i in u64, i in 0..L (L = 4 for a 254-bit F, L = 2 for fp128)
each limb column is RdInc-shaped and rides the existing balanced-digit
          machinery verbatim; digit smallness enforcement doubles as the
          limb range check (limb_i < 2^64)
recompose: FieldRdInc = sum_i limb_i * 2^(64 i)   — one linear identity over F
          (exact: the canonical representative is < p, so no carries and no
          modular wraparound in the recomposition)
virtual:  full-width FieldRdInc becomes a virtual polynomial; every Twist
          relation consumes it unchanged
openings: the final opening opens the limb columns; the FieldRdInc claim is
          reconstructed linearly via the existing stage-8 reconstruction
          machinery (the pattern akita already uses for fused-inc cells)
reduction: FieldRegistersIncClaimReduction consumes the recomposed virtual
          instead of a committed opening (wiring change only)
```

Implementation-time checks (not design questions): lattice norm-budget
headroom for L limb columns; the reconstruction ordering in stage 8's packed
path. The `field-inline x akita` compile error in
`crates/jolt-verifier/src/config.rs` is removed only in the PR that lands
this design with accept/tamper fixtures — same discipline as the verifier
gate's removal.

## Axis 2: 128-bit base-field instantiation

Everything above the tracer is generic over `F`. The concrete work:

- Tracer genericization: `decode_field`/`encode_field` in
  `tracer/src/instruction/field_inline.rs` are pinned to BN254 `Fr`;
  parameterize over the active `F` (~150 LOC, per the v2-port estimate).
- Encoding version: `FieldValueEncoding` gains a two-limb 16-byte variant
  beside `BN254_SCALAR_CANONICAL`; `FieldInlineBytecodeMetadata.value_encoding`
  and the profile fingerprint already version this — a proof/preprocessing
  built under one encoding rejects under another fail-closed.
- Bridge economics improve: `FIELD_LOAD_FROM_X` covers half the field; a
  full-width load is one radix multiplication (single 2^64 constant) plus one
  add; `FIELD_STORE_TO_X`'s range-restricted semantics (< 2^64, trap
  otherwise) and `FIELD_LOAD_IMM` are unchanged.
- Generator budget: `MAX_BLINDFOLD_GENERATORS` is cfg-keyed today (32 FR-off /
  64 FR-on); the composed uniskip degrees do not change with the field, so no
  further action.
- Expectation reset (documentation, not protocol): software two-limb field
  multiplication costs tens of cycles, so the per-op native speedup drops from
  ~190x (BN254) to ~20-40x; the pinned-slot SDK matters relatively more.

## Ordering against pending PRs

#1718/#1732 (akita -> modular prover) touch the same seams as the landed
field-inline work (kernel backend slots, stage-0 commit path, stage-6b/8
per-mode test fixtures). The axes are compile-disjoint, so all conflicts are
textual adjacency, arbitrated by the boundary tests and both ratchets.
Recommended order: merge the field-inline branch first; the packed-FieldRdInc
slice (Axis 1) lands after #1718, as one slice-sized unit; the base-field
slice (Axis 2) lands with or after the field switch itself.

## Implementation steps

1. Axis 1 after #1718 merges: limb-column commit + digit rides + recomposition
   identity + stage-8 reconstruction + inc-reduction rewiring; accept/tamper
   fixtures on the packed path; remove the compile error last.
   Review gate: FR-off akita byte-identity; packed FR fixtures accept/tamper.
2. Axis 2 with the field switch: tracer parameterization + encoding variant +
   bridge fixture updates; the eq-MLE guest re-fixtured under the new
   encoding.
   Review gate: encoding-mismatch proofs reject fail-closed; e2e both modes.

## Status (2026-08-26): both axes landed under the fp128 ruling

The fp128 switch decision came down as a ruling: the packed (akita) axis
proves exclusively over fp128, and no BN254 akita configuration will ever
exist. FR execution is therefore configuration-selected — the akita feature
chain repoints the tracer's `ProofField` to `jolt_field::AkitaField` and
`FieldValueEncoding::ACTIVE` to `TWO_LIMB_128_CANONICAL` (inert while
field-inline is off); Dory keeps BN254 Fr. FR guests are
configuration-specific, and cross-configuration proofs reject fail-closed on
the metadata encoding-equality gate (intended behavior, tested in both
directions).

Everything above is implemented; the `field-inline x akita` compile error is
removed. Axis 1 as landed, with two dispositions the design left open:

- `FieldRegistersIncClaimReduction` stays a stage-6b member unchanged; on the
  packed axis its reduced claim feeds a stage-8 reconstruction member
  (`FieldIncLimbReconstruction`: per-column booleanity legs at a fresh
  reference point plus the balanced-digit decode leg) instead of the
  homomorphic RLC splice (`stage8/field_inline.rs` stays homomorphic-only;
  the packed seam is `stage8/field_inline_packed.rs` on both fronts).
- The limb object's presence is claim-gated: `FieldRdInc` identically zero
  means every limb column is empty, and the catalogued Akita fold schedules
  cannot open an all-zero one-hot object — so the object (commitment +
  opening) exists exactly when the stage-6b reduced `FieldRdInc` claim is
  nonzero, enforced both ways fail-closed at the stage-8 opening
  (Schwartz-Zippel over the reduction chain; the reconstruction member
  itself always runs).

Review gates met: FR-off akita byte-identity (the legacy byte-diff ratchets),
dory FR-on identity (the dory e2e's reference/optimized wire equality), and
the packed accept/tamper suite (`jolt-prover/tests/akita_field_inline_e2e.rs`:
eq-MLE re-fixtured at the 16-byte encoding via host-side fp128 evaluation,
the FR-inactive muldiv with the object absent, and five rejected tampers).
The packed reconstruction kernel is the naive reference tier on both
backends; a sparse optimized kernel is the noted follow-up.

## Status (2026-08-27): packed axis re-landed on dense-group batching

Upstream #1798 replaced the byte one-hot advice objects with dense u64-word
commitments opened through one heterogeneous batch; the packed `FieldRdInc`
treatment is re-landed on that mechanism, superseding the one-hot limb
columns, the `FieldIncLimbReconstruction` member, and the claim-gated limb
object above. The `field-inline x akita` compile error is removed again. The
fp128 ruling, the limb decomposition facts
(`canonical_limbs`/`limb_place_value`), and the Axis 2 encoding gates are
unchanged. The dense-group design:

- One independent dense precommitted group carries `FieldRdInc`'s two
  canonical u64 limb-word columns (fp128, L = 2), prefix-packed at `log_T`
  through the shared dense schedule floor
  (`protocols/field_inline/lattice/packing.rs`), committed in stage 0 beside
  advice under a transparent setup, and opened in the SAME native
  heterogeneous batch. Its frozen `PrecommittedRole` is order 2, transcript
  label `field_inc_limbs`: the canonical batch order is
  `[UntrustedAdvice, TrustedAdvice, FieldIncLimbs, OneHotTrace]`.
- The stage-6b reduced `FieldRdInc` claim `(v, r)` binds to the group by ONE
  explicit verifier equality BEFORE the packing reduction:
  `v == e0 + 2^64 * e1` over the proof-carried limb evaluations at `r`
  (typed `FieldIncLimbRecompositionMismatch` reject), which the batch then
  binds to the committed columns through the selector-reduced physical
  claim. No reconstruction sumcheck member, no booleanity legs.
- Presence is never claim-gated: on an FR-on packed build the group is
  ALWAYS present (all-zero content is legal — dense schedules are keyed by
  `(num_vars, num_polys)` shape, never content), enforced fail-closed both
  ways between `PrecommittedSchedule.field_inc_limbs` (always scheduled) and
  the proof's commitment/claims slots.
- Provisioning: `AdviceScheduleParams` carries the FR limb arity line
  (`jolt-akita` `FieldIncLimbScheduleParams`, law-derived data pinned to the
  jolt-claims packing law by the registry's FR provisioning tests), and the
  grouped schedule registry enumerates FR-present combinations only — every
  advice subset (including advice-absent) with the per-final-arity FR
  profile appended last, planned under the same u64-bounded dense fold
  policy as advice.

Review gates met: FR-off akita and dory byte-identity (the byte-diff
ratchets), dory FR-on fixtures unchanged, and the packed accept/tamper suite
(`jolt-prover/tests/akita_field_inline_e2e.rs`: eq-MLE accepted on both
kernel backends with wire equality, the FR-inactive muldiv accepted with the
group present and all-zero, and the tamper matrix — limb-evaluation offset,
layout-digest flip, batch-proof mutation, stripped group, spurious second
FR-role group — all rejected).
