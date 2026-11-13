# Digit-zero virtualization for the packed one-hot trace

Digit-zero virtualization applies to the instruction and bytecode families.
RV64 fused increments are committed directly as a bounded-dense polynomial. RAM
keeps all of its rows committed and is not digit-zero virtualized (§4). This
document makes no claim about the note's nonconstant-activation RAM
construction. Source: "Digit-Zero Virtualization for Twist and Shout"
(`~/akita-paper/ra-virtualization-note.tex`).

Base (Dory) mode commits full one-hot columns for every family and keeps the
Hamming-weight legs of the instruction and bytecode families.

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
polynomials of the **instruction and bytecode** memories. Each has public
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

Because every virtualized family has `M_µ ≡ 1`, reconstruction forces
`Σ_{k_i} ra_i(k_i,j) = 1` identically. Booleanity then gives an exactly-one-hot
column per cycle without a Hamming-weight leg. Public unit
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

## 4. RAM: committed digit-zero row, no recentering

`RamRa` is committed with **all** rows (digit-zero row included), as in base
mode. Its stage-7 legs are Booleanity (`γ^{2i}`) and RA virtualization
(`γ^{2i+1}`) with **no digit-zero recentering**; the virtualized families use
the same two legs, recentered.

Neither mode has a RAM Hamming-weight check: there is no `RamHammingBooleanity`
sumcheck and no RAM Hamming-weight leg. RAM read-write checking enforces the
per-cycle access count `Σ_k ra(k,j) = Load(j) + Store(j)` instead, which with
Booleanity of every committed `RamRa` chunk makes `ra` one-hot on memory rows
and zero on other rows (`specs/ram-guards.md`).

## 5. Proof obligations

### Virtualized families

For instruction and bytecode, the note's theorem applies directly. Given
Booleanity of the reconstructed rows, their sum is one by definition, so each
`ra_i(·,j)` is a unit vector and `ra(·,j)` encodes a unique address.

### RAM is deferred

The note proposes `M_RAM = Load + Store`, which is not the public unit case
implemented here. RAM read-write checking already enforces that activation on
the product selector: `Σ_k ra(k,j) = Load(j) + Store(j)` on every cycle. It
does not constrain each chunk's weight on non-memory rows, which the note's
per-digit reconstruction would. Virtualizing RAM's digit-zero row on top of it
remains future work: it needs a mapping of the activation claims, their
evaluation points, and their binding to the existing RAM relations.

`RamRa` commits every row and reduces Booleanity and RA virtualization. This is
a scope boundary, not a conclusion about whether a later implementation of the
note's RAM construction is possible.

## 6. Delta versus base mode / prior drafts

- Instruction/bytecode: digit-zero-virtualized (omit the digit-zero
  row, two reconstructed stage-7 legs, no Hamming leg).
- RAM: all rows committed, as in base mode; two unrecentered stage-7 legs
  (Booleanity, virtualization); no `RamHammingBooleanity` and no Hamming leg in
  either mode, since RAM read-write checking enforces the `Load + Store` access
  count. The note's `Load+Store` digit-zero construction is deferred.
- Notation renamed to digit-zero throughout the lattice path
  (`…AtDigitZero` publics, `LatticeDigitZeroClaimReduction`, digest
  `…/digit-zero-mu-one-full-ram/…`).
- Shared `JoltRelationId::HammingWeightClaimReduction` / `SumcheckId` names
  kept (base mode still performs a genuine Hamming-weight reduction under them
  for the instruction and bytecode families).

## 7. Gates

- `muldiv` both modes; the four akita e2e tests;
  `jolt-claims` lattice-semantics + claim-graph; `jolt-verifier` tamper suites
  and fingerprint-keyed fixtures; both clippy modes; `cargo fmt`.
- Policy-specific: physical packing tests must show that digit zero is omitted
  for a virtualized column and retained for a RAM-style committed column; the
  stage-7 algebra test must show two legs per polynomial, recentered for
  instruction and bytecode and unrecentered for RAM.
