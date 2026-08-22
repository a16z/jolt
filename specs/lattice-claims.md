# Spec: Lattice (Akita) Claims

| Field | Value |
|-------|-------|
| Author(s) | Markos Georghiades, Claude |
| Created | 2026-07-02 |
| Updated | 2026-08-20 |
| Status | implemented |
| PR | [#1675](https://github.com/a16z/jolt/pull/1675) |

> **Advice update:** The byte-one-hot trusted/untrusted advice design in this
> document is historical. The active protocol commits dense word advice and
> directly opens the final `AdviceClaimReduction` claims; both advice objects
> are precommitted groups of one joint Akita opening, in the canonical order
> `[UntrustedAdvice, TrustedAdvice, OneHotTrace]`. The old advice IDs remain
> only as positional-codec tombstones. One-hot trace and committed-program
> reconstruction remain active. See
> [a16z/jolt#1798](https://github.com/a16z/jolt/pull/1798)
> for the current advice format, batch-opening flow, and preprocessing-time
> schedule provisioning design.

## Purpose

Akita is a lattice PCS with no commitment homomorphism. Jolt therefore cannot
reuse Dory's commitment-level RLC at the final opening. The Akita protocol
instead has two layers:

1. Per-proof one-hot trace columns are packed into one physical
   `OneHotTrace` polynomial and selector-reduced to one evaluation.
2. Independently committed dense objects—advice, direct bytecode chunks, and
   the initial program image—join that trace in one native grouped Akita
   opening.

This document defines the current Akita claim boundary, commitment layout, and
stage schedule. The direct committed-program design is introduced in
[LayerZero-Research/jolt#36](https://github.com/LayerZero-Research/jolt/pull/36).

## Scope

The Akita clear-mode protocol includes:

- one physical `OneHotTrace` commitment;
- fused balanced-digit increment columns and their signed carry;
- the lattice bytecode read-RAF and digit-zero claim-reduction chain;
- dense word commitments for trusted and untrusted advice;
- one bounded-dense `BytecodeChunk(i)` object per committed bytecode chunk;
- one bounded-dense `ProgramImageInit` object;
- one grouped opening ordered as
  `[UntrustedAdvice?, TrustedAdvice?, BytecodeChunk(0..C), ProgramImageInit, OneHotTrace]`.

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
├── geometry.rs       balanced increment chunking and decode algebra
├── packing.rs        OneHotTrace, advice, and direct-program object plans
├── strategy.rs       OneHotTrace layout, point permutation, setup shape
└── relations/
    ├── read_raf.rs   lattice bytecode read-RAF with fused-inc stages
    ├── digit_zero.rs stage-7 digit-zero and increment decode reduction
    └── booleanity.rs lattice-mode Booleanity
```

The committed-program reconstruction relation files and their decomposed
polynomial families were deleted. Advice is also committed directly as dense
word polynomials; it has no byte-decomposition relation.

## Committed Polynomial Families

The Akita-specific committed families that survive are:

```rust
BalancedIncDigit(usize)
BalancedIncCarry
TrustedAdvice
UntrustedAdvice
BytecodeChunk(usize)
ProgramImageInit
```

`FusedInc` remains a virtual polynomial. `RdInc` and `RamInc` are not
separately committed in Akita mode: the bytecode read-RAF stages consume their
reduced claims and produce the fused value used by the balanced-digit chain.

Enum `Ord` is not protocol group order. Packing plans and
`PrecommittedRole` explicitly define order, and layout/statement transcripts
bind it.

## Commitment Layout

### OneHotTrace

`OneHotTrace` contains the per-proof one-hot columns:

```text
InstructionRa(0..I)
BalancedIncDigit(0..N)
BalancedIncCarry
BytecodeRa(0..B)
RamRa(0..R)
```

Every semantic column has logical arity `log_K + log_T`. Instruction,
bytecode, digit, and carry columns omit the digit-zero row; RAM retains it.
Unused physical slots are zero. Stage 8 rejects missing claims, point
disagreement, arity disagreement, or noncanonical commitment metadata before
calling the PCS.

The physical polynomial uses the layout's row-major order. Relation claims use
the protocol's logical address/cycle order, so the canonical layout performs
the required point permutation before adding the selector prefix. The layout
digest binds the ordered identities, capacity, dimensions, and trace order.

### Advice objects

Each present advice kind is one singleton dense word polynomial. Its logical
arity is the log of the padded word count. Its physical arity is
`max(14, logical_arity)`, implemented as a zero-prefix embedding. Advice
objects are optional at proof time and appear before committed-program objects
in canonical role order.

### Direct committed-program objects

Committed-program preprocessing creates:

- `C` singleton `BytecodeChunk(i)` objects, in increasing chunk index;
- one singleton `ProgramImageInit` object.

A chunk is the shared canonical
`build_committed_bytecode_chunk_coeffs` grid. Its lane capacity is 512, so
with `R = log2(bytecode_len / C)` its logical arity is `9 + R`. The image
logical arity is `log2(padded_image_words)`. Both use physical arity
`max(14, logical_arity)` and reject arity above 34.

Chunk coefficients are interleaved by the preprocessing
`TracePolynomialOrder`. That order is serialized, included in the
preprocessing digest and chunk layout digest, and checked against the proof
before any proving or verification stage. Program immediates are validated by
the shared committed-bytecode helper: unsigned magnitude at most
`u64::MAX` is accepted; larger values are rejected before
`Field::from_i128`.

There is no byte decomposition and no reconstruction claim. Stage 6b/7 final
claims for `BytecodeChunk(i)` and `ProgramImageInit` are the claims Stage 8
opens directly.

## Fused Increment Relation Chain

### Bytecode read-RAF

The lattice read-RAF address phase consumes the four reduced increment claims.
Its cycle phases carry the `FusedInc` factor and produce the fused opening at
the shared stage-6b cycle point. Store selection is substituted directly from
the bytecode Store flag; there is no separately committed selector.

### Balanced digits and carry

For chunk width `b = log_K`, the fused RV64 increment is represented by
`64 / b` balanced radix-`2^b` one-hot digit columns plus one signed carry
column. The carry uses the same `K × T` domain and encodes `-1`, `0`, or
`1` modulo the radix. This uniform shape is required by the shared final
point.

Lattice Booleanity proves the digit/carry cells boolean. Stage 7's digit-zero
claim reduction accounts for the omitted zero row and folds the balanced
decode against `FusedInc`:

```text
Σ_j 2^(b·j) · balanced_digit_j + 2^64 · carry = FusedInc
```

RAM retains its own Hamming/booleanity coverage. There is no standalone
increment reconstruction stage after Stage 7.

## Final Opening

Stage 8 first resolves one final evaluation for every semantic column:

- OneHotTrace columns come from the relation DAG's final outputs;
- advice claims come from the advice claim reductions;
- bytecode chunk claims come from the bytecode claim reduction;
- the program-image claim comes from the program-image claim reduction.

Each object plan zero-prefix-embeds its logical claim to the object's physical
arity and selector-reduces occupied slots. Akita then proves one grouped
same-point statement. The canonical group order is:

```text
UntrustedAdvice?
TrustedAdvice?
BytecodeChunk(0)
…
BytecodeChunk(C - 1)
ProgramImageInit
OneHotTrace
```

Roles are bound by label, and chunk roles also bind their index. Duplicate,
missing, or permuted roles are rejected before backend verification. The
preamble separately absorbs the direct program commitments as indexed
`bytecode_chunk_commitment` entries followed by
`program_image_init_commitment`.

Full-program mode has no direct-program suffix. If it also has no advice, the
ordinary single-group OneHotTrace opening remains valid.

## Schedule Provisioning

Grouped schedule rows are provisioned during preprocessing. Direct-program
profiles are a mandatory suffix of every committed-program combination.
Advice presence contributes only the reachable optional prefixes; direct
objects are not power-set factors.

For a program with `C` bytecode chunks and `A` advice kinds with nonzero
capacity, setup capacity is exactly:

```text
C + 2 + A
```

That is `C` chunks, one image, one main trace, and `A` advice objects. The
maximum supported shape is `C = 256`, `A = 2`: 260 total
groups/polynomials. The registered-row cap is 256; a K=256 family with 32 final
arities and four advice-presence cases reaches 128 rows for one preprocessing.

## Protocol Invariants

- Every committed-program coefficient fits `JoltDenseBounded`'s centered
  `u64::MAX` bound.
- Logical and physical arity are distinct; padding is a zero-prefix embedding,
  not a change to the semantic polynomial.
- Prover and verifier derive identical object plans, role order, transcript
  labels, setup profiles, and trace order.
- `BytecodeChunk(i)` and `ProgramImageInit` are opened directly. No
  virtualized reconstruction exception or auxiliary proof path exists.
- Modular and legacy Akita provers must remain byte-identical.

## Testing Strategy

Required gates include:

- direct-object arity floor/ceiling and 256-chunk capacity tests;
- immediate boundary tests for both signs at `u64::MAX` and `2^64`;
- committed-program e2e tests with one and two bytecode chunks;
- modular/legacy byte-parity tests with one and two chunks;
- reordered direct-role and trace-order mismatch rejection;
- 128-row provisioning and 260-group setup-capacity tests;
- Akita catalog regeneration and Fiat-Shamir inventory checks;
- standard and ZK Dory suites, confirming the protocol change is Akita-only.

See [the verifier testing gates](../book/src/dev/testing-gates.md) for the
inventory and provisioning commands.
