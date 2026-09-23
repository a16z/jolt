# Digit-zero virtualization for the packed one-hot trace

Scope 2026-08-16: digit-zero virtualization applies to the instruction,
bytecode, and balanced-increment families only. RAM stays on the base treatment
(all rows committed, `RamHammingBooleanity`, and the Hamming-weight leg) so it
can be reviewed in a separate change. This document makes no claim about the
note's nonconstant-activation RAM construction. Source: "Digit-Zero
Virtualization for Twist and Shout" (`~/akita-paper/ra-virtualization-note.tex`).

Base (Dory) mode is untouched: it commits full one-hot columns for every
family, keeps all Hamming-weight legs and `RamHammingBooleanity`, and must keep
producing bit-identical proofs.

## 1. Setting and notation

Fix a memory of size `K` with `d`-digit one-hot addressing over `T` cycles: an
address `k ∈ [K]` is the tuple of digits `(k_1, …, k_d) ∈ [K^{1/d}]^d` (radix
`K^{1/d}`). For digit position `i`, the coordinate read-address polynomial is

```
ra_i : [K^{1/d}] × [T] → F,       ra_i(k_i, j) = 1  iff cycle j selects digit value k_i,
ra(k, j) := Π_{i=1..d} ra_i(k_i, j),
```

and `r̃a_i` denotes the multilinear extension. `ẽq` is the multilinear equality
polynomial; evaluation points are written `(r_address, r_cycle)`.

The technique applies here to the committed coordinate read-address
polynomials of the **instruction and bytecode** memories, and (an extension
beyond the note, §6) to the balanced-increment value columns. Each has public
unit activation (§2). **RAM and registers are excluded from this change**: RAM
requires the note's separate nonconstant activation treatment, while
`Rs1Ra`, `Rs2Ra`, and `RdWa` are full-domain virtual polynomials already bound
by the bytecode read-RAF.

Notation to code: digit position `i` is a chunk index; digit value `k_i` is a
row index; `ra_i(0,·)` is the digit-zero row; and the nonzero-digit rows are
committed. Code comments use the paper's reconstruction term
`ẽq(r_address,0)·M̃_µ(r_cycle)` rather than the earlier `w(0)·A` shorthand.

## 2. Memory activation (Definition 1) — constant 1 for every virtualized family

For each virtualized family µ, the activation `M_µ : [T] → F` is the public
constant 1 (the memory is read every cycle; padding cycles select the all-zero
word, so the digit-zero row is hot and no committed row carries an entry —
padding rows are free):

| family | `M_µ` |
|---|---|
| `InstructionRa` | `≡ 1` (one lookup per cycle) |
| `BytecodeRa` | `≡ 1` (one fetch per cycle) |
| `BalancedIncDigit/Carry` | `≡ 1` (every cycle has a possibly-zero fused increment) |

Because every virtualized family has `M_µ ≡ 1`, reconstruction forces
`Σ_{k_i} ra_i(k_i,j) = 1` identically. Booleanity then gives an exactly-one-hot
column per cycle without a Hamming-weight leg. For balanced increments, read
`ra_i` as the corresponding digit or carry one-hot column. Public unit
activation is what lets this change avoid the additional RAM-specific proof
obligations.

## 3. Digit-zero virtualization (Definition 2)

For a virtualized family, the prover commits only the nonzero-digit rows
`r̃a_i(k_i, ·)`, `k_i ≠ 0` (`pack_one_hot_columns`,
`jolt-prover-legacy/src/zkvm/packed_witness.rs`, drops digit value 0 for these
families). The omitted row is **defined** on the hypercube by
`ra_i(0, j) := 1 − Σ_{k_i≥1} ra_i(k_i, j)`, which off the cube gives the
reconstruction identity

```
r̃a_i(r_address, r_cycle) = ẽq(r_address, 0)
    + Σ_{k_i≥1} ( ẽq(r_address, k_i) − ẽq(r_address, 0) )·r̃a_i(k_i, r_cycle).   (1)
```

An input claim `c` on a semantic (digit-zero-inclusive) column becomes a claim
on committed rows only:
`c − ẽq(r_address, 0) = Σ_{k_i≥1} (ẽq(r_address,k_i) − ẽq(r_address,0))·r̃a_i(k_i,r_cycle)`.
Both sides live in the stage-7 reduction
(`jolt-claims/src/protocols/jolt/lattice/relations/digit_zero.rs`): the
constant baseline `ẽq(r_address, 0)` folds into the input claim, the prover
zeroes the digit-zero row of each virtualized pushforward so the sumcheck runs
over committed rows alone, and the verifier supplies the `ẽq(·, 0)` values as
the `…AtDigitZero` publics. The RAM pushforwards remain unchanged.

Booleanity binds the digit-zero-*inclusive* semantic columns (digit-zero row
included; `subprotocols/booleanity.rs`), and the stage-7 recentering ties those
Boolean-checked columns to the committed rows and the constant activation.
There is no Hamming-weight check for these families — `Σ_k ra_i(k,j) = 1` holds
by construction of the reconstruction, so the weight is fixed with nothing to
check.

## 4. RAM: the base treatment, unchanged

`RamRa` is committed with **all** rows (digit-zero row included), exactly as
base mode does, and keeps:

- `RamHammingBooleanity` (stage 6b): proves the RAM access indicator
  `H = RamHammingWeight` is Boolean (`H² = H`) at the stage-1 cycle point;
- the stage-7 Hamming-weight leg (`γ^{3i}`): ties `Σ_k ra(k,·) = H` against the
  committed rows — a genuine tie because all rows are committed;
- the standard Booleanity and RA-virtualization legs.

So RAM's stage-7 legs are the base 3-leg form (Hamming, Booleanity,
virtualization) with **no digit-zero recentering**; the virtualized families
use the 2-leg recentered form (Booleanity, virtualization) plus, for the
increment columns, the decode leg. The stage-7 reduction is therefore
per-family hybrid.

## 5. Proof obligations

### Virtualized families

For instruction and bytecode, the note's theorem applies directly. Given
Booleanity of the reconstructed rows, their sum is one by definition, so each
`ra_i(·,j)` is a unit vector and `ra(·,j)` encodes a unique address. The same
argument applies column-by-column to the balanced-increment extension because
each digit and carry column also has public activation one.

### RAM is deferred

The note proposes `M_RAM = Load + Store`, which is not the public unit case
implemented here. Integrating that construction requires a separate mapping
of the activation claims, their evaluation points, and their binding to the
existing RAM relations. This patch deliberately does not settle those details.

For now `RamRa` commits every row and follows the existing base protocol:
`RamHammingBooleanity`, the Hamming-weight leg, Booleanity, and RAM
virtualization. This is a scope boundary, not a conclusion about whether a
later implementation of the note's RAM construction is possible.

## 6. The balanced-increment columns (extension beyond the note)

The balanced-increment value columns (`BalancedIncDigit(j)`,
`BalancedIncCarry`) are virtualized with `M ≡ 1`; the balanced encoding puts a
zero fused increment on the digit-zero row for every digit and the carry, so
padding rows are free. Their *decode* leg —
`Σ_j 2^{8j}·value(digit_j) + 2^64·value(carry) = FusedInc` at the 6b point — is
not a weight check and is unaffected: same γ placement, same `FusedInc` input
from the read-RAF cycle phase, same range caveat (one-hotness pins the
per-digit values only to `[−K/2, K/2)`; `specs/lattice-claims.md`, relation 3).

## 7. Delta versus base mode / prior drafts

- Instruction/bytecode/increment: digit-zero-virtualized (omit the digit-zero
  row, two reconstructed stage-7 legs, no Hamming leg).
- RAM: unchanged from base (all rows committed, `RamHammingBooleanity` + base
  Hamming leg, three-leg stage-7 reduction). The `Load+Store` construction is
  deferred.
- Notation renamed to digit-zero throughout the lattice path
  (`…AtDigitZero` publics, `LatticeDigitZeroClaimReduction`, digest
  `…/digit-zero-mu-one-full-ram/…`).
- Shared `JoltRelationId::HammingWeightClaimReduction` / `SumcheckId` names
  kept (base mode still performs a genuine Hamming-weight reduction under them,
  and now so does RAM on the packed path).

## 8. Gates

- `muldiv` both modes (base bit-identical); the four akita e2e tests;
  `jolt-claims` lattice-semantics + claim-graph; `jolt-verifier` tamper suites
  and fingerprint-keyed fixtures; both clippy modes; `cargo fmt`.
- Policy-specific: physical packing tests must show that digit zero is omitted
  for a virtualized column and retained for a RAM-style committed column; the
  stage-7 algebra test must show the mixed two-leg/three-leg gamma layout.
