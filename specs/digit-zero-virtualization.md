# Digit-zero virtualization for the packed one-hot trace

Status 2026-08-14: **Digit-zero virtualization applies to the instruction,
bytecode, and balanced-increment families only. RAM is NOT virtualized** — it
stays on the base treatment (all lanes committed, `RamHammingBooleanity` + the
base Hamming-weight leg). This was decided after an adversarial review
*confirmed* a soundness break in every virtualized-RAM form (§5.RAM). Source:
"Digit-Zero Virtualization for Twist and Shout"
(`~/akita-paper/ra-virtualization-note.pdf`); the note virtualizes RAM too, but
that is unsound in this system.

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

The technique applies to the committed coordinate read-address polynomials of
the **instruction and bytecode** memories, and (an extension beyond the note,
§6) to the balanced-increment value columns — every family whose activation
(§2) is the public constant 1. **RAM and registers are excluded**: RAM because
its activation is not constant and no sound in-protocol proof of it exists on
the virtualized path (§5.RAM); registers because `Rs1Ra`, `Rs2Ra`, `RdWa` are
full-domain virtual polynomials already bound by the bytecode read-RAF.

Notation ↔ code: digit position `i` = chunk index; digit value `k_i` = lane;
digit-zero row `ra_i(0,·)` = lane zero / "default"; nonzero-digit rows = the
committed lanes; digit-zero baseline `ẽq(r_address,0)·M̃_µ(r_cycle)` = the
`w(0)·A` recentering term.

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

Because every virtualized family has `M_µ ≡ 1` — a public constant plugged
directly into the recentering — the reconstruction forces `Σ_k L(k,j) = 1`
*identically*, and with Booleanity this gives exactly-one-hot per cycle with no
auxiliary check (Theorem 1, §5). This constant-activation property is the
entire reason the technique is sound here and not for RAM.

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
on committed rows only: `c − ẽq(r_address, 0) = Σ_{k_i≥1} (ẽq(k_i) − ẽq(0))·⟨committed openings⟩`.
Both sides live in the stage-7 reduction
(`jolt-claims/src/protocols/jolt/lattice/relations/digit_zero.rs`): the
constant baseline `ẽq(r_address, 0)` folds into the input claim, the prover
zeroes the digit-zero row of every pushforward so the sumcheck runs over
committed rows alone, and the verifier supplies the `ẽq(·, 0)` weights as the
`…AtDigitZero` publics.

Booleanity binds the digit-zero-*inclusive* semantic columns (digit-zero row
included; `subprotocols/booleanity.rs`), and the stage-7 recentering ties those
Boolean-checked columns to the committed rows and the constant activation.
There is no Hamming-weight check for these families — `Σ_k ra_i(k,j) = 1` holds
by construction of the reconstruction, so the weight is fixed with nothing to
check.

## 4. RAM: the base treatment, unchanged

`RamRa` is committed with **all** lanes (digit-zero row included), exactly as
base mode does, and keeps:

- `RamHammingBooleanity` (stage 6b): proves the RAM access indicator
  `H = RamHammingWeight` is Boolean (`H² = H`) at the stage-1 cycle point;
- the stage-7 Hamming-weight leg (`γ^{3i}`): ties `Σ_k ra(k,·) = H` against the
  committed rows — a genuine tie because all lanes are committed;
- the standard Booleanity and RA-virtualization legs.

So RAM's stage-7 legs are the base 3-leg form (Hamming, Booleanity,
virtualization) with **no digit-zero recentering**; the virtualized families
use the 2-leg recentered form (Booleanity, virtualization) plus, for the
increment columns, the decode leg. The stage-7 reduction is therefore
per-family hybrid.

## 5. Soundness

### Virtualized families (instruction, bytecode, increment) — Theorem 1

Given Booleanity of all reconstructed rows: each row's weight `w` satisfies
`w·1_F = M_µ(j) = 1`, and injectivity of `m ↦ m·1_F` below `char F` forces
`w = 1`. So every `ra_i(·,j)` is a unit vector and `ra(·,j)` encodes a unique
address on every cycle, padding included. The activation is a public constant,
so there is no activation to forge and nothing rests on a prover-supplied
value. Sound unconditionally, in full and committed-program modes alike.

### RAM — sound by full commitment; the virtualized path was a confirmed break

**Why the virtualized RAM path is unsound.** Virtualizing `RamRa` needs an
activation `M_RAM` (the per-cycle access indicator) that is not a public
constant, so `M_RAM ∈ {0,1}` must be *proved* in-protocol. Every form tried is
unsound for the same structural reason: `M_RAM` (whether the prover-supplied
`RamHammingWeight` claim of the original PR, pinned by a five-hop RAF chain, or
a `Load+Store` flag sum) is a virtual column the prover materializes at
stage 6b, while its booleanity check's reference point is the stage-1 Spartan
cycle binding, drawn five stages earlier. A booleanity zero-check
`Σ_j eq(r₁,j)·(M²−M)(j) = 0` forces `M` Boolean only when `r₁` postdates `M`;
here it predates it, so a prover that knows `r₁` can pass with a non-Boolean
`M`.

An adversarial review then *confirmed* (not conjectured) that nothing
downstream rescues it: the Twist read/write + val-check + output-check chain is
**linear in `ra` and contains no multiset or one-hot check**
(`relations/ram/{read_write_checking,val_check,output_check}.rs`; reference
kernels), so a weight-≥2 RAM tensor on an access cycle is accepted end-to-end.
RAF (`Σ_k unmap(k)·ra = RamAddress`) and Spartan rows 0/1 constrain only the
*sum* of the two addresses, not the weight. Concretely, a malicious committed
program can set `M_RAM = 2` on a load, produce a weight-2 tensor whose two
`unmap` values sum to the honest address, commit `inc = 0`, and make the load
return an attacker-chosen `T = Val(k1) + Val(k2)` — with Booleanity, RAF,
Spartan, read/write, val-check, and output-check all satisfied. This break is
latent in any virtualized-RAM form.

**Why full commitment is sound.** Committing all of `RamRa`'s lanes makes the
Hamming-weight leg `Σ_k ra(k,·) = H` a *real* tie of `H` to stage-0 committed
data (all lanes present, nothing reconstructed). `H` is then a fixed stage-0
polynomial, so `RamHammingBooleanity`'s `H² = H` at the stage-1 point is sound
(data fixed before the reference randomness), giving `H ∈ {0,1}`, hence
`Σ_k ra(k,j) ∈ {0,1}` — genuine one-hotness, cryptographic and trust-free,
exactly base Jolt's guarantee. The virtualized path broke this by omitting the
digit-zero lane, which made the Hamming leg vacuous (`Σ_k L = M_RAM` by
construction) and severed `H` from committed data. Not virtualizing RAM
restores the tie.

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
  lane, 2-leg recentered stage-7 reduction, no Hamming leg). Sound by Theorem 1.
- RAM: unchanged from base (all lanes committed, `RamHammingBooleanity` + base
  Hamming leg, 3-leg stage-7 reduction). No `RamActivationBooleanity`, no
  `Load+Store` activation — both removed as unsound.
- Notation renamed to digit-zero throughout the lattice path
  (`…AtDigitZero` publics, `LatticeDigitZeroClaimReduction`, digest
  `…/digit-zero-balanced-inc/…`).
- Shared `JoltRelationId::HammingWeightClaimReduction` / `SumcheckId` names
  kept (base mode still performs a genuine Hamming-weight reduction under them,
  and now so does RAM on the packed path).

## 8. Gates

- `muldiv` both modes (base bit-identical); the four akita e2e tests;
  `jolt-claims` lattice-semantics + claim-graph; `jolt-verifier` tamper suites
  and fingerprint-keyed fixtures; both clippy modes; `cargo fmt`.
- Soundness-specific: a negative/tamper test that a weight-≥2 RAM tensor (or a
  forged `RamHammingWeight`) is rejected — the regression guarding this
  decision.
