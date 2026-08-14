# Digit-zero virtualization for the packed one-hot trace

Status 2026-08-14: **SOUNDNESS REVIEW REQUIRED — do not treat the RAM
activation as sound.** Implemented on `perf/akita-protocol-opts` (PR #1731) and
adapted in the modular stack (PR #1732, held unpushed). An adversarial review
found the RAM one-hotness argument (§5) has an open gap: the
`RamActivationBooleanity` check's reference point predates the columns it
checks, so it does not by itself force the activation Boolean. See the
**OPEN SOUNDNESS QUESTION** box in §5. Instruction/bytecode/increment families
(activation ≡ 1, Theorem 1) are unaffected. This document replaces the "Where
the RAM activation is pinned" argument in `specs/lattice-claims.md` (five-hop
chain, one-free-bit closure) with the construction below. Source: "Digit-Zero
Virtualization for Twist and Shout" (`~/akita-paper/ra-virtualization-note.pdf`).

Base (Dory) mode is untouched by everything here: it commits full one-hot
columns, keeps its direct Hamming-weight legs and `RamHammingBooleanity`, and
must keep producing bit-identical proofs.

## 1. Setting and notation

Fix a memory of size `K` with `d`-digit one-hot addressing over `T` cycles:
an address `k ∈ [K]` is the tuple of digits `(k_1, …, k_d) ∈ [K^{1/d}]^d`
(radix `K^{1/d}`; `ram_d = ceil(log₂ ram_K / log_k_chunk)`,
`jolt-prover-legacy/src/zkvm/config.rs:267`). For digit position `i`, the
coordinate read-address polynomial is

```
ra_i : [K^{1/d}] × [T] → F,       ra_i(k_i, j) = 1  iff cycle j selects digit value k_i,
ra(k, j) := Π_{i=1..d} ra_i(k_i, j),
```

and `r̃a_i` denotes the multilinear extension. `ẽq` is the multilinear
equality polynomial; evaluation points are written `(r_address, r_cycle)`.

The technique applies to the committed coordinate read-address polynomials of
the **instruction, bytecode, and RAM** memories, and (an extension beyond the
note, §6) to the balanced-increment value columns. **Registers are excluded**:
`Rs1Ra`, `Rs2Ra`, `RdWa` are full-domain virtual polynomials whose claims are
already bound by the bytecode read-RAF relation.

Mapping to the previous vocabulary (the old names are being retired on this
path; see §7):

| this spec / the note | previous code term |
|---|---|
| digit position `i` | chunk index `i` |
| digit value `k_i` | lane |
| digit-zero row `ra_i(0, ·)` | lane zero / implicit zero / "default" |
| nonzero-digit rows `ra_i(k_i, ·), k_i ≠ 0` | committed lanes, `Q` |
| memory activation `M_µ` | activation `A` |
| digit-zero baseline `ẽq(r_address, 0)·M̃_µ(r_cycle)` | lane-zero baseline `w(0)·A` |
| reconstruction identity, Eq. (1) | implicit-zero recentering |

## 2. Memory activation polynomials (Definition 1)

For each family µ, the activation `M_µ : [T] → F` says whether the memory is
accessed at cycle `j`:

| family | `M_µ` | provenance |
|---|---|---|
| `InstructionRa` | `≡ 1` | public constant — one lookup per cycle |
| `BytecodeRa` | `≡ 1` | public constant — one fetch per cycle |
| `RamRa` | `M_RAM := L̃oad + S̃tore` | derived virtual — the MLEs of the two circuit flags |
| `BalancedIncDigit/Carry` | `≡ 1` | public constant — every cycle has a (possibly zero) fused increment |

`Load` and `Store` are the per-cycle Boolean circuit-flag columns
(`jolt-riscv/src/flags.rs:32,34`; witness `jolt-witness/src/witnesses/flags.rs`),
mutually exclusive per cycle (`Load(j)·Store(j) = 0`). Padding cycles are
no-ops: both flags are 0 and every constant-1 family selects the all-zero
word, i.e. every digit-zero row is hot and no committed row carries an entry
(padding rows are free).

`M_RAM` is **never committed and has no standalone identity test**. At each
point where a reconstruction needs `M̃_RAM(r_cycle)`, the protocol expands it
as `L̃oad(r_cycle) + S̃tore(r_cycle)` and the two flag claims join the
claim-reduction batch of the corresponding read-address claims (§4) — this is
footnote 1 of the note.

## 3. Digit-zero virtualization (Definition 2)

The prover commits only the nonzero-digit rows `r̃a_i(k_i, ·)`, `k_i ≠ 0`
(`pack_one_hot_columns`, `jolt-prover-legacy/src/zkvm/packed_witness.rs:23-45`,
drops digit value 0). The omitted row is **defined** on the hypercube by

```
ra_i(0, j) := M_µ(j) − Σ_{k_i=1..K^{1/d}−1} ra_i(k_i, j),
```

which off the cube is the reconstruction identity

```
r̃a_i(r_address, r_cycle) = ẽq(r_address, 0)·M̃_µ(r_cycle)
    + Σ_{k_i≥1} ( ẽq(r_address, k_i) − ẽq(r_address, 0) )·r̃a_i(k_i, r_cycle).   (1)
```

An input claim `c = r̃a_i(r_address, r_cycle)` on a semantic (digit-zero-
inclusive) column therefore becomes a claim on committed rows only:

```
c − ẽq(r_address, 0)·M̃_µ(r_cycle) = Σ_{k_i≥1} ( ẽq(k_i) − ẽq(0) )·⟨committed openings⟩.
```

Both sides of this move live in the stage-7 reduction: the baseline
`ẽq(r_address, 0)·M̃_µ(r_cycle)` folds into its input claim
(`jolt-claims/src/protocols/jolt/lattice/relations/hamming_weight.rs`
`input_expression`; legacy `input_claim`,
`jolt-prover-legacy/src/zkvm/claim_reductions/hamming_weight.rs:346-377`), the
prover zeroes the digit-zero row of every pushforward so the sumcheck runs
over committed rows alone (`initialize_lattice`, same file), and the verifier
supplies the `ẽq(·, 0)` weights as the `…AtDigitZero` publics
(`jolt-verifier/src/stages/stage7/hamming_weight_claim_reduction.rs`).

**Consequence.** `Σ_{k_i} ra_i(k_i, j) = M_µ(j)` holds identically by
construction, for every digit of every family. There is **no Hamming-weight
check anywhere on this path** — not as a sumcheck, not as a γ-leg (the
previously reserved-but-vacuous powers are renumbered away, §7). What remains
mandatory is **Booleanity over the reconstruction**: the Booleanity sumcheck
binds the digit-zero-inclusive semantic columns, digit-zero row included
(`jolt-prover-legacy/src/subprotocols/booleanity.rs:657-669` states this
contract), and the stage-7 recentering is what ties those Boolean-checked
columns to the committed rows and the activation.

## 4. Producing the RAM activation claims

Stage 7 consumes the activation at the shared stage-6b cycle point (all its
input legs — Booleanity and virtualization outputs — carry that cycle
coordinate). The existing `Load`/`Store` openings live at the stage-1 Spartan
outer point, and the read-RAF address fold (stage 6a) cannot consume a claim
at a point that is only bound during stage 6b. So the flag openings at the
6b point are produced the only place they can be: by a stage-6b batch member
that binds the flag columns directly.

**`RamActivationBooleanity`** (replaces `RamHammingBooleanity` in the akita
schedule; same slot, same rounds `log T`, same degree 3, zero input claim, no
challenges):

```
0 = Σ_j eq(r_cycle_ref, j) · ( (Load(j) + Store(j))² − (Load(j) + Store(j)) )
```

with output claims `OpFlags(Load)` and `OpFlags(Store)` at the bound stage-6b
point. It proves, in-protocol, that the activation
`M_RAM(j) = Load(j) + Store(j) ∈ {0, 1}` — a checked fact, not a structural
assumption about the bytecode. Stage 7's two RAM baselines then use
`M̃_RAM(r_6b) = Load@6b + Store@6b`.

WARNING — the check is deliberately a *single* booleanity on the sum, not a
γ-batch of per-flag legs (`γ⁰(L²−L) + γ¹(S²−S) + γ²LS`). The flag columns
bound here are virtual — never committed — so the prover may choose them
after every challenge is drawn; a γ-combination of independent legs over
prover-chosen columns has non-Boolean solutions for any fixed γ (pick
`L² − L = c`, `S² − S = −c/γ`), while `B² = B` for the single polynomial
`B := Load + Store` has only Boolean roots pointwise, no matter when the
columns are chosen. Individual flag booleanity and mutual exclusivity are
neither checked nor needed here: only the sum flows into the reconstruction,
and the split between the two openings is deliberately unconstrained (§5).
The Spartan-side flag columns remain bytecode-bound exactly as today.

The previous construction's `RamHammingWeight` claim, its `A² = A` sumcheck,
and every wire that carried them disappear from this path. Nothing else ever
consumed them (sole consumer was the stage-7 baseline). The relation and the
`RamHammingWeight` virtual polynomial remain in the tree for base mode.

Cost accounting: one degree-3 `log T` sumcheck replaced by one degree-3
`log T` sumcheck; one opening wire (`RamHammingWeight@6b`) replaced by two
(`Load@6b`, `Store@6b`); one sumcheck (`RamHammingBooleanity`) and one virtual
polynomial retired from the path.

## 5. Soundness — the dishonest-prover accounting

The prover controls: the committed nonzero-digit rows, the semantic columns
it feeds Booleanity, the flag columns it feeds `RamActivationBooleanity`, and
every claimed evaluation. The verifier-enforced facts are:

- (B) Booleanity output claims over the semantic columns, digit-zero row
  included, at the 6b point;
- (R) the stage-7 recentering equations — for each column, the Booleanity leg
  and the virtualization leg, at two independent address points, both against
  the same committed-row openings and the same activation claims;
- (X) `RamActivationBooleanity`: the sum-booleanity zero-check
  `0 = Σ_j eq(r₁, j)·(B(j)² − B(j))`, `B := Load + Store`, at the stage-1
  cycle point `r₁`, plus the two flag openings at the 6b point (individual
  columns unconstrained and unused beyond their sum). In isolation a
  zero-check over *virtual* columns at a pre-known point is not
  pointwise-binding — what makes it force `m(j) := B(j) ∈ {0, 1}` is the
  composite argument spelled out in "Why the zero-check binds" below;
- (F) the RAF identity `Σ_k unmap(k)·ra(k, j) = RamAddress(j)` with
  `unmap(k) = 8k + ℓ`, `ℓ = lowest remapped address`, enforced fail-closed to
  satisfy `ℓ > 8` (`validate_ram_remap_base`,
  `jolt-verifier/src/verifier.rs:862-881` — **kept**, rationale below);
- (S) Spartan rv64 rows 0/1 (`jolt-r1cs/src/constraints/rv64.rs:272-294`):
  `(Load+Store)·(RamAddress − Rs1 − Imm) = 0` and
  `(1 − Load − Store)·RamAddress = 0`, over the Spartan flag columns, which
  the read-RAF β-legs bind to the bytecode table.

All statements below are up to sumcheck/Schwartz–Zippel soundness error, as
usual.

**Step 1 — weights equal the activation.** By (R), each semantic column
agrees with the Eq. (1) reconstruction from the committed rows and the
activation claims at random points, so as polynomials
`Σ_{k_i} ra_i(k_i, j) = m(j)` per digit `i`, cycle `j`. By (B) each summand is
Boolean, so the sum is `w_i(j)·1_F` for an integer weight
`w_i(j) ∈ {0, …, K^{1/d}}`. The map `n ↦ n·1_F` is injective on that range
(`K^{1/d} < char F`), so `w_i(j) = m(j)` exactly — every digit of a family
carries the same weight, the activation value.

**Step 2 — unit-or-zero.** For the constant-1 families `m ≡ 1`: every
`ra_i(·, j)` is a unit vector and `ra(·, j)` encodes a unique address
`raf(j)` on every cycle, padding included. For RAM, (X) (via the binding
argument below) gives `m(j) ∈ {0, 1}`: `ra(·, j)` is a unit vector at a
unique `raf(j)` when `m(j) = 1` and identically zero when `m(j) = 0`. The
same one residual freedom as the previous construction remains at this point
in the argument: on a cycle where every committed RAM row is empty, both
`m(j) = 0` (no access) and `m(j) = 1` (digit-zero rows hot — an access at
remapped word 0) are representable; step 3 closes it.

**Step 3 — the activation matches the real access pattern.** The flag
columns bound in (X) are still prover-materialized; (F) + (S) pin their
values. Note (S) constrains the *Spartan* flag columns (bytecode-bound), and
(F) relates the one-hot tensor to `RamAddress`; together they leave the
activation no room:

- *Fabrication* (`m(j) = 1` on a cycle whose instruction is not a
  load/store): step 2 makes `ra(·, j)` a unit vector, so the RAF left side is
  `unmap(raf(j)) = 8·raf(j) + ℓ ≥ ℓ > 8`, a nonzero integer below
  `char F`. Row 1 of (S) forces `RamAddress(j) = 0`. Contradiction — this is
  the direction where `ℓ > 8` (equivalently `unmap(0) ≠ 0`) is load-bearing,
  which is why `validate_ram_remap_base` stays even though the activation no
  longer *depends* on it for its own pinning: fabricating an access **at
  remapped word 0** has RAF left side exactly `unmap(0) = ℓ`.
- *Suppression* (`m(j) = 0` on a genuine load/store): the tensor is zero, so
  the RAF left side is 0, while row 0 of (S) forces
  `RamAddress(j) = Rs1(j) + Imm(j)`, the access's address — nonzero for every
  tracer-reachable access (all live RAM sits at byte addresses `≥ ℓ > 8`).
  Contradiction.

**Step 4 — why the activation booleanity is checked in-protocol, precisely.**
Without (X), step 1 still forces all digit weights to a common integer
`m(j) ≥ 0`, and one may ask whether `m(j) ≥ 2` survives. On a non-access
cycle it cannot: the RAF left side is a sum of `m^d ≥ 1` values
`unmap(k) ≥ ℓ`, a positive integer below `char F` (at most
`m^d·(8(K−1)+ℓ) ≪ char F` for every supported geometry), while row 1 forces
`RamAddress = 0`. On an *access* cycle, however, positivity alone does not
exclude `m ≥ 2`: the `m^d` unmap values could in principle sum to
`RamAddress` as integers. The sum-booleanity (X) is what closes that case
(`m ≤ 1` always), via the value-set argument below. This matters most in committed-program
mode: without (X), "the activation is Boolean" would rest on "no bytecode row
sets both flags" — a property of the bytecode *table*, which a malicious
committed program controls (`Load = Store = 1`, or non-Boolean flag lanes).
The previous construction's `A² = A` gave the same `m ≤ 1` guarantee; (X)
preserves it under the flags-derived activation rather than assuming bytecode
validity. For such malformed rows the (S)-side analysis (a guard
`g ∉ {0, 1}` forces `RamAddress = 0 ∧ Rs1 + Imm = 0` via both rows) leaves
them provable only as no-access cycles. Full-program mode additionally
re-checks flag validity offline on the public bytecode, as today. Note (X)
constrains the activation columns this relation binds, not the Spartan flag
columns — the two are cross-pinned only through steps 1–3, which is exactly
enough: the activation's *value* is forced to the true access indicator, and
the Spartan flags are separately bytecode-bound.

**OPEN SOUNDNESS QUESTION (2026-08-14 adversarial review) — the activation
booleanity check is not pointwise-binding as implemented.** A booleanity
zero-check `Σ_j eq(ref, j)·(B² − B)(j) = 0` forces `B` Boolean only when `ref`
is drawn *after* `B` is fixed. Here `B = Load + Store` is a virtual column the
prover materializes at stage 6b, and `ref = r₁` is the **stage-1** Spartan
`LookupOutput` cycle binding — drawn five stages earlier. The two flag
openings are consumed by nothing but this gadget and the stage-7 baseline
(verified: no tie to the bytecode-bound Spartan flags or to committed data),
so the prover chooses `B` knowing `r₁` and can satisfy the single evaluation
`(B²−B)̃(r₁) = 0` with a non-Boolean `B`.

Base mode does **not** have this gap: its Hamming-weight leg pins
`H = Σ_k ra(k, ·)` to the stage-0 committed rows, so its identical stage-1
booleanity check is really about a fixed stage-0 polynomial that genuinely
predates `r₁`. Digit-zero virtualization deleted that leg (the weight identity
"holds by construction"), which removed exactly the stage-0 pin that made the
point ordering safe. An earlier draft of this section argued a three-move
Schwartz–Zippel closure "over `r₁`"; that is **invalid** — it assumed `r₁`
postdates the free digit-zero/flag choices, and it does not.

Consequence: for RAM, per-cycle one-hotness (`Σ_k ra(k,j) ≤ 1` on access
cycles) currently has **no valid proof**. Step 4 above explicitly delegates the
`m ≥ 2`-on-access-cycles case to this check, and RAF + Spartan rows 0/1 do not
independently exclude it. Whether a weight-≥2 committed RAM tensor can survive
the Twist read/write + val-check chain on a genuine access cycle (with a
malicious committed program) was not resolved; if that chain independently
pins the tensor one-hot, the construction is sound but this section's
justification is still wrong. **Do not ship the RAM activation on this
argument.** Instruction, bytecode, and increment families are unaffected
(activation ≡ 1, Theorem 1, no gadget).

Candidate hardenings (a design decision, not yet chosen):
- tie the `RamActivationBooleanity` flag openings to the bytecode-bound
  Spartan `OpFlags(Load)/OpFlags(Store)` (sound iff committed-program bytecode
  flags are validated Boolean+exclusive offline);
- commit the activation column at stage 0 and draw its booleanity reference
  afterward, restoring the base-mode "fixed before `r₁`" property;
- restore an explicit `Σ_k committed_ra(k,j) ≤ 1` bound for RAM whose
  randomness postdates the commitment (a real weight leg, partially undoing
  the RAM saving);
- or prove that Twist read/write + val-check already forbid weight-≥2 RAM
  tensors on access cycles, and rewrite §5 to rest on that instead.

**Completeness.** Honest traces are always provable: padding cycles have both
flags 0 and all-zero tensors (representable — every committed row empty,
digit-zero rows reconstruct to 0); an access to remapped word 0 has every
committed row empty and all digit-zero rows hot (representable — the
reconstruction carries the 1), with RAF satisfied through
`unmap(0) = ℓ = RamAddress`. A load/store whose computed address is raw 0 is
not representable as an access — but the tracer faults on it, so no honest
trace contains one, and its acceptance set (provable only as a no-access
cycle with `RamAddress = 0`) is unchanged from the previous construction.

## 6. The balanced-increment columns (extension beyond the note)

The note covers read-address families; the packed trace also digit-zero-
virtualizes the balanced-increment value columns (`BalancedIncDigit(j)`,
`BalancedIncCarry`) with `M ≡ 1`. The balanced encoding puts a zero fused
increment on digit-zero for every digit and the carry, so padding rows are
free here too. Their *decode* leg — `Σ_j 2^{8j}·value(digit_j) + 2^64·
value(carry) = FusedInc` at the 6b point — is not a weight check and is
untouched by this spec: same γ placement (after renumbering), same `FusedInc`
input from the read-RAF cycle phase, same range caveat (one-hotness pins the
per-digit values only to `[−K/2, K/2)`;
`specs/lattice-claims.md`, relation 3).

## 7. Delta versus the previous construction

Removed from the akita path:

- `RamHammingBooleanity` scheduling, the `RamHammingWeight` opening and its
  codec/tamper/clear-claims wires (base mode keeps all of it);
- the reserved-but-vacuous Hamming γ-powers in the stage-7 reduction: the
  layout renumbers to 2 powers per RA family (Booleanity, virtualization),
  1 per increment column (Booleanity), plus the decode power — the
  base-mode index alignment those powers preserved is intentionally given up;
- the five-hop pinning argument and the one-free-bit caveat in
  `specs/lattice-claims.md` (superseded by §5).

Added: `RamActivationBooleanity` (§4), the two flag openings at the 6b point,
in-protocol booleanity of the activation sum.

Renamed (notation alignment; lattice path only — shared
`JoltRelationId::HammingWeightClaimReduction` / `SumcheckId` variants keep
their names because base mode still performs a genuine Hamming-weight
reduction under them):

| old | new |
|---|---|
| implicit zero / lane zero (prose, flags, tests) | digit zero |
| `LatticeHammingWeightClaimReduction*` types, `lattice/relations/hamming_weight.rs` | `LatticeDigitZeroClaimReduction*`, `lattice/relations/digit_zero.rs` |
| `HammingWeightClaimReductionPublic::EqBooleanityAtDefault` | `…::EqBooleanityAtDigitZero` |
| `HammingWeightClaimReductionPublic::EqVirtualizationAtDefault(i)` | `…::EqVirtualizationAtDigitZero(i)` |
| `eq_at_default` (verifier) | `eq_at_digit_zero` |
| params `implicit_zero`, `eq_bool_at_zero`, `eq_virt_at_zero`, `claims_hw` | `digit_zero`, `eq_bool_at_digit_zero`, `eq_virt_at_digit_zero`, `activations` |
| `hamming_weight_claim()` activation selector (lattice use) | lattice-side `activation_claim()` (base keeps `hamming_weight_claim`) |
| layout digest `…/implicit-zero-balanced-inc/v5` | `…/digit-zero-balanced-inc/v6` |

Unchanged: everything about the commitment geometry (fixed-prefix layout,
slot packing, stage-8 selector order), Booleanity and virtualization
structure, the read-RAF fused-inc stages, advice and precommitted-program
prefix objects (they still commit digit zero), and base mode wholesale.

## 8. Gates

- `cargo nextest run -p jolt-prover-legacy muldiv --cargo-quiet --features host`
  and `--features host,zk` — base mode bit-identical;
- akita e2e: `muldiv_e2e_akita`, `muldiv_e2e_akita_forced_k256`,
  `muldiv_e2e_akita_committed_program`, `advice_e2e_akita`;
- `jolt-claims` lattice-semantics and claim-graph suites; `jolt-verifier`
  tamper suites (retargeted wires plus new flag-wire targets) and
  fingerprint-keyed fixtures (re-key);
- both workspace clippy modes and `cargo fmt -q`.
