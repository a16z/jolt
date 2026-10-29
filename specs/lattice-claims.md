# Spec: Lattice (Akita) Claims

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Claude |
| Created | 2026-07-02 |
| Updated | 2026-10-09 |
| Status | implemented |
| PR | [#1675](https://github.com/a16z/jolt/pull/1675) |

> **Advice update:** The byte-one-hot trusted/untrusted advice design in this
> document is historical. The active protocol commits dense word advice and
> opens each present advice object from the corresponding Stage 4
> `RamValCheck` advice contribution. Akita no longer runs Stage 6b/7
> `AdviceClaimReduction` members; those reductions remain Dory-only. The old
> advice IDs remain only as positional-codec tombstones. See
> [a16z/jolt#1798](https://github.com/a16z/jolt/pull/1798)
> for the dense advice format, and
> [a16z/jolt#1829](https://github.com/a16z/jolt/pull/1829)
> for the Stage 4 advice leaf-claim boundary. One-hot trace digit-zero
> reduction and direct committed-program openings remain active.
>
> **Field-inline update:** on Akita builds with field-inline enabled, the
> joint opening also carries an always-present dense commitment to the full
> `FieldRdInc` values, created from the execution witness in stage 0 and
> ordered after `TrustedAdvice`. Stage 8 opens the stage-6b reduced claim
> directly. The verifier rejects committed-program mode with field-inline
> enabled, so direct-program roles never share a batch with it. See
> [field-inline-portability.md](field-inline-portability.md).

## Purpose

Akita is a lattice PCS with no commitment homomorphism. Jolt therefore cannot
reuse Dory's commitment-level RLC at the final opening. The Akita protocol
instead has two layers:

1. Per-proof one-hot trace columns form one native `OneHotTrace` commitment
   group. Every column has `log_T + log_K` variables and one evaluation at
   the common `(cycle || address)` point; Stage 8 assembles these evaluations
   in canonical column order without a selector reduction.
2. Independently committed dense objects—advice, RV64 and field-register increments, direct
   whole bytecode, and the initial program image—join that trace in one
   native grouped Akita opening, according to the selected protocol.

This document defines the current Akita claim boundary, commitment layout, and
stage schedule. The direct committed-program design is introduced in
[LayerZero-Research/jolt#36](https://github.com/LayerZero-Research/jolt/pull/36).

## Scope

The Akita clear-mode protocol includes:

- one native `OneHotTrace` commitment group containing the actual trace columns;
- one bounded-dense fused RV64 increment polynomial;
- the lattice bytecode read-RAF and digit-zero claim-reduction chain;
- dense word commitments for trusted and untrusted advice;
- one bounded-dense `ProgramBytecode` object for the whole bytecode;
- one bounded-dense `ProgramImageInit` object;
- one grouped opening ordered as
  `[UntrustedAdvice?, TrustedAdvice?, ProgramBytecode, ProgramImageInit, FusedInc, OneHotTrace]`.

Akita and `zk` remain mutually exclusive. The Dory protocol, including its
clear and BlindFold modes, is unchanged.

## Boundary Contract

| Crate | Owns | Must not contain |
|-------|------|------------------|
| `jolt-claims` | ids, arities, symbolic relations, canonical layouts, final-opening sources | transcripts, witnesses, PCS code |
| `jolt-openings` | fixed-prefix packing, zero-prefix claim embedding, generic grouped-opening types | Jolt-specific relation semantics |
| `jolt-verifier` | input validation, stage schedule, claim assembly, grouped statement verification | duplicated relation algebra |
| `jolt-witness` / prover crates | witness materialization and the matching proving schedule | verifier-only policy |
| `jolt-akita` | Akita commitment/opening transport, grouped schedules, role binding | Jolt relation algebra |

The proof never chooses a layout, digest, group role, or group order. Those are
derived from preprocessing and protocol configuration on both sides.

## Module Layout

```text
crates/jolt-claims/src/protocols/jolt/lattice/
├── geometry.rs       lattice stage dimensions
├── increments.rs     dense RV64 increment layout and opening point
├── packing.rs        OneHotTrace, advice, and direct-program object plans
├── strategy.rs       OneHotTrace layout, point permutation, setup shape
└── relations/
    ├── read_raf.rs   lattice bytecode read-RAF with fused-inc stages
    └── digit_zero.rs stage-7 RA digit-zero reduction
```

The committed-program reconstruction relation files and their decomposed
polynomial families were deleted. Advice is also committed directly as dense
word polynomials; it has no byte-decomposition relation.

## Committed Polynomial Families

The Akita-specific committed families that survive are:

```rust
TrustedAdvice
UntrustedAdvice
ProgramBytecode
ProgramImageInit
```

`FusedInc` remains a virtual polynomial. `RdInc` and `RamInc` are not
separately committed in Akita mode: the bytecode read-RAF stages consume their
reduced claims and produce the fused evaluation opened directly against the
mandatory dense increment object. The virtual opening ID names this reduced
claim; `FusedIncLayout` and its group role bind its committed representation.
The old balanced-digit/carry enum ordinals remain reserved and are not trace columns.

Enum `Ord` is not protocol group order. Packing plans and
`CommitmentGroupRole` explicitly define order, and layout/statement transcripts
bind it.

## Commitment Layout

### OneHotTrace

`OneHotTrace` contains the per-proof one-hot columns:

```text
InstructionRa(0..I)
BytecodeRa(0..B)
RamRa(0..R)
```

Every semantic column has logical arity `log_K + log_T`. Instruction
and bytecode columns omit the digit-zero row.
RAM retains digit zero only on cycles with a remappable RAM access. Stage 8
requires exactly one final claim per canonical column and rejects missing
claims, point disagreement, arity disagreement, or noncanonical commitment
metadata before calling the PCS.

Each native polynomial uses the row-major `(cycle || address)` order. Relation
claims use `(address || cycle)`, so the canonical layout permutes the point.
There is no slot prefix. The versioned layout digest binds ordered column
identities, their count, arity, chunk width, trace length, and family dimensions.
`OneHotTraceLayout::plan` rejects more than 64 columns because the streaming
witness represents digit-zero presence with a `u64` mask; this is a witness
representation limit, not a PCS batching limit.

### Advice objects

Each present advice kind is one singleton dense word polynomial. Its logical
arity is the log of the padded word count. Its physical arity is
`max(14, logical_arity)`, implemented as a zero-prefix embedding. Advice
objects are optional at proof time and appear before committed-program objects
in canonical role order.

### Direct committed-program objects

Committed-program preprocessing creates:

- one singleton `ProgramBytecode` object;
- one singleton `ProgramImageInit` object.

The bytecode uses the shared canonical
`build_committed_bytecode_coeffs` grid. Its lane capacity is 512, so
with `R = log2(bytecode_len)` its logical arity is `9 + R`. The image
logical arity is `log2(padded_image_words)`. Both use physical arity
`max(14, logical_arity)` and reject arity above 34.

Bytecode coefficients are interleaved by the preprocessing
`TracePolynomialOrder`. That order is serialized, included in the
preprocessing digest and bytecode layout digest, and checked against the proof
before any proving or verification stage. Program immediates are validated by
the shared committed-bytecode helper: unsigned magnitude at most
`u64::MAX` is accepted; larger values are rejected before
`Field::from_i128`.

There is no byte decomposition and no reconstruction claim. Stage 6b/7 final
claims for `ProgramBytecode` and `ProgramImageInit` are the claims Stage 8
opens directly.

## Fused Increment Relation Chain

### Bytecode read-RAF

The lattice read-RAF address phase consumes the four reduced increment claims.
Its cycle phases carry the `FusedInc` factor and produce the fused opening at
the shared stage-6b cycle point. Store selection is substituted directly from
the bytecode Store flag; there is no separately committed selector.

### Direct dense increment

Commit `D(t) = FusedInc(t)` before drawing protocol challenges. On Store cycles
it is the RAM delta; otherwise it is the destination-register delta. Each honest
delta is the difference of two u64 values, hence `|D(t)| <= u64::MAX`, and padding
coefficients are zero. This fits the existing signed 65-bit `JoltDenseBounded`
source contract; no PCS security preset or source bound is weakened.

Stage 8 opens the Stage-6b `FusedInc` claim directly against that commitment.
`FusedIncLayout` binds the trace arity and pads only to the dense schedule floor;
any added leading coordinates are opened at zero. The verifier requires the
commitment even for an all-zero increment table and checks its dense backend,
singleton count, arity, and layout digest. Its transcript label and final group
role are `fused_inc`.

There are no increment digit/carry columns, Booleanity terms, or decode
sumcheck. The four original register/RAM increment claims still pass through
the same bytecode ReadRAF identities. Booleanity and the Stage-7 digit-zero
reduction now cover only RA columns. No arithmetic range-check obligation is
removed: the former digit/carry decode did not enforce the honest u64-difference
range, and the VM's register/memory constraints remain unchanged.

## Final Opening

Stage 8 first resolves one final evaluation for every semantic column:

- OneHotTrace columns come from the relation DAG's final outputs;
- the dense fused increment is opened at the Stage-6b ReadRAF cycle point;
- advice claims come directly from the Stage 4 `RamValCheck` advice
  contributions;
- the whole-bytecode claim come from the bytecode claim reduction;
- the program-image claim comes from the program-image claim reduction.

Auxiliary singleton object plans zero-prefix-embed their logical claims to
their physical arities. The native trace group keeps all ordered column
evaluations at its common point. Akita proves one heterogeneous grouped opening
statement whose groups use their own local evaluation points. Commitments
(including polynomial counts), group roles, points, and ordered evaluations
are absorbed before backend batching challenges are derived.
The canonical group order is:

```text
UntrustedAdvice?
TrustedAdvice?
ProgramBytecode
ProgramImageInit
FusedInc
OneHotTrace
```

Roles are bound by label. Duplicate,
missing, or permuted roles are rejected before backend verification. The
preamble separately absorbs the direct program commitments as
`program_bytecode_commitment` followed by
`program_image_init_commitment`.

Full-program mode omits the program objects but still opens the mandatory
`FusedInc` and `OneHotTrace` groups. Field-inline mode retains its separate
full-field `FieldRdInc` group before `FusedInc`.

## Schedule Provisioning

Grouped schedule rows are provisioned during preprocessing. Direct-program
profiles and the fused increment are mandatory in every committed-program combination.
Advice presence contributes only the reachable optional opening groups and no
longer participates in the shared precommitted claim-reduction scheduling
reference for bytecode or program-image reductions. Direct objects are not
power-set factors.

For `A` advice kinds with nonzero capacity and `M` native trace columns,
committed-program setup covers `4 + A` commitment groups and `3 + A + M`
polynomials: one whole bytecode, one image, one fused increment, `A` singleton advice objects,
and one `M`-polynomial trace group. The bounds `A <= 2` and `M <= 64`
allow at most 6 groups and 69 polynomials.
One setup plans only its final arity and column count, so at most four rows
cover the reachable advice-presence cases. The 128-row bound applies to one
provisioning request, not to the process cache.

## Protocol Invariants

- Every committed-program coefficient fits `JoltDenseBounded`'s centered
  `u64::MAX` bound.
- Logical and physical arity are distinct; padding is a zero-prefix embedding,
  not a change to the semantic polynomial.
- Prover and verifier derive identical object plans, role order, transcript
  labels, setup profiles, and trace order.
- `ProgramBytecode` and `ProgramImageInit` are opened directly. No
  virtualized reconstruction exception or auxiliary proof path exists.
- Akita preprocessing uses the versioned `akita-whole-bytecode/v1` digest
  domain; Dory uses `dory-whole-bytecode/v2`. Regenerate older preprocessing
  and proofs for both backends, including full-program mode.

## Testing Strategy

Required gates include:

- whole-bytecode object identity, arity floor/ceiling, and capacity tests;
- immediate boundary tests for both signs at `u64::MAX` and `2^64`;
- committed-program e2e tests across internal Akita response profiles;
- reordered direct-role and trace-order mismatch rejection;
- grouped provisioning and backend group-capacity tests;
- Akita catalog regeneration and Fiat-Shamir inventory checks;
- standard and ZK Dory suites, confirming the protocol change is Akita-only.

See [the verifier testing gates](../book/src/dev/testing-gates.md) for the
inventory and provisioning commands.
