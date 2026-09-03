# W4-R — verifier stage algebra as an R1CS (`jolt-wrapper::relation`)

Date: 2026-09-02. Branch: `wrap/spartan-hyperkzg`. Lane: W4-R (relation + profile).

## Result

- `profile.rs`: `WrapperProfile` (log_t, log_k_ram, log_k_bytecode, rw/one-hot configs, trace order,
  memory layout, program-image length, entry index) + `digest()` (blake3 over bincode). Rejects
  committed-program preprocessing, untrusted-advice proofs and ZK proofs.
- `relation/`: `build_relation(profile) -> Relation { matrices: ConstraintMatrices<Fr>, public:
  PublicLayout, link: LinkTable, rows: Vec<RowSpan> }` and `generate_witness(profile, preprocessing,
  public_io, proof) -> Witness { values, state_in, outsourced, native_parity }`. One walk in two
  modes: build mode allocates unknown wires; assign mode replays a `Recording<Blake3Transcript>` log
  of the native verifier (every `append_bytes` payload and squeeze), asserting the schedule
  event-for-event (labels, counts, lengths, verifier-computed values), then runs the native parity
  guard (`relation/native.rs`): every derived and challenge wire is recomputed through its native
  `ConcreteSumcheck` owner and compared (typed `RelationError::NativeMismatch`).
- Covered: stages 1–7, the stage-8 RLC, the Dory `Fr` scalar algebra (β/α/γ/d wires, inverses,
  u/v/χ/Δ scalars, s1/s2 accumulators, HT, pairing scalars, commitment weights), and the trailing
  `EvaluationClaim` absorb. Challenge wires are plain variables; binding them to the hash chain is
  the transcript table's (T1) job via `LinkTable::schedule`.
- Fibonacci 2^18 (L = 18, ram K = 2^13, bytecode 2^12, chunk 4, σ = 11): `check_witness` passes;
  `build_relation` 7 ms, `generate_witness` 136 ms (includes the native verify replay), on the M4 mini.

## Constraint table (fibonacci 2^18, exact)

| section | rows | what |
|---|---:|---|
| stage1/uniskip | 29 | degree-27 uni-skip: round-sum row + Horner |
| stage1/remainder | 58 | 19 rounds × (Horner 3 rows) + head |
| stage1/expected | 181 | 19 row weights, Az/Bz linear forms, tau kernel (Lagrange + eq), 2 products |
| stage2/uniskip | 14 | degree-6 uni-skip + Lagrange weights |
| stage2/batch | 108 | 31 rounds × 3 + 5 head folds |
| stage2/public | 13 | 13 public copies of the RAM address |
| stage2/expected | 230 | EqCycle, TauKernel, EqSpartan, EqAddress, IoMask, 5 lowered outputs |
| stage3/batch | 71 | 18 rounds × 3 + 3 folds |
| stage3/expected | 271 | 2 × EqPlusOne, EqProduct, EqSpartan, 3 lowered outputs |
| stage4/batch | 84 | 25 rounds × 3 + 2 folds + input folds |
| stage4/public | 7 | 7 register-address copies |
| stage4/expected | 121 | EqCycle, LtCycle+γ, 2 lowered outputs |
| stage5/batch | 447 | 128 address rounds × 2 + 18 cycle rounds × 10 + 3 folds |
| stage5/tables | 1,025 | eq_reduction + 54 table MLEs (971 shared rows) + 54 EqTableValue products |
| stage5/expected | 318 | EqRafConstant/Flag, 3 × EqCycle, LtCycle, 3 lowered outputs (incl. Σ_i EqTableValue_i·flag_i) |
| stage6a/batch | 208 | 6 gamma squeezes are free; 4 rounds × 3 + 8 rounds × 2 + 2 folds; input fold of ~90 stage-1..5 openings |
| stage6a/expected | 3 | two opening equalities + fold |
| stage6a/public | 18 | 12 bytecode-address + 6 gamma copies |
| stage6b/batch | 116 | 18 rounds × 5 + 6 folds + input folds |
| stage6b/expected | 858 | 5 StageValue products + eqs, Entry, SpartanOuter/ShiftRaf, EqAddressCycle, 8 cycle eqs, 6 lowered outputs (39 booleanity squares, 39 RA products) |
| stage7/batch | 207 | 4 rounds × 2 + input fold over 39 × 3 claims (γ powers) |
| stage7/expected | 588 | EqBooleanity + 39 EqVirtualization (4-coordinate eqs) + output fold |
| stage8/rlc | 84 | 2 embedding scales (3 rows), 41 absorbed values, 40 ρ powers, 41 products |
| stage8/dory | 194 | 24 inverses, 41 commitment weights, χ(σ) unit, per round: u·α, v·α⁻¹, χ (2), Δ1R, Δ2R, s1/s2 (4) |
| stage8/evaluation_claim | 1 | joint claim copy |
| **total** | **5,254** | 6,761 variables (incl. constant one), 45 public |

Per stage: 1: 268 · 2: 365 · 3: 342 · 4: 212 · 5: 1,790 · 6a: 229 · 6b: 974 · 7: 795 · 8: 278.

Fibonacci 2^20 (L = 20, σ = 12; proof generated in 12.6 s / 1.48 GiB RSS on the mini, cached):
**5,454** constraints, 7,031 variables, 45 public — per stage 278 · 383 · 376 · 230 · 1,834 · 229 ·
1,038 · 795 · 291; `build_relation` 9 ms, `generate_witness` 181 ms. Growth per unit of `log_t` is
≈ 100 rows (the `L`-round batches at degrees 3/3/3/10/5 plus the `L`-coordinate eq gadgets).

The plan's ≈9.7k estimate assumed the expanded 1,296-term Spartan-outer form and per-term products;
the factored `TauKernel·Az·Bz` form (19 row-weight wires, constant matrix entries), constant folding
in `Ctx::mul`, and sorted-prefix memoization in the expression lowering bring stages 1–7 to 4,975.

## Public IO layout (`PublicLayout`, `z[1..=45]`)

| slot | count | meaning |
|---|---:|---|
| `val_io` | 1 | outsourced input: `Π(1−r_hi)·sparse_segments_mle_msb(public IO, r_lo)` at the RAM address |
| `init_eval` | 1 | outsourced input: public initial RAM MLE at the RAM address |
| `stage_values[5]` | 5 | outsourced inputs: address-only bytecode-table folds (`read_raf_stage_values` · `eq(r_bytecode)`) |
| `outputs.ram_address` | 13 (= log K) | read-write / output-check / value-check RAM address (one wire set) |
| `outputs.bytecode_address` | 12 | stage-6a bytecode address |
| `outputs.bytecode_gammas` | 6 | γ, stage1..5 γ of the bytecode read-RAF |
| `outputs.register_address` | 7 | register read-write / value-evaluation address (one wire set) |

`outsourced_inputs(preprocessing, public_io, ram_address, StageValueInputs {..})` recomputes the
inputs from the outputs alone (the fixture test asserts equality with the witness).

## Link table

- `schedule`: 2,434 entries — 1,222 `Fr` wires (prover elements and verifier-computed absorbs),
  376 squeezes (310 `challenge()` 125-bit + 66 `challenge_scalar()` 128-bit), 137 opaque Dory
  elements (29,408 bytes: 68 GT × 384, 35 G1 × 32, 34 G2 × 64), 22,336 bytes of constant labels.
- `dory`: `num_vars 22, sigma 11`, 175 named scalars (`DoryScalar`): evaluation `y`, 41 commitment
  weights `ρ^i·β_0⁻¹`, β/β⁻¹/α/α⁻¹ × 11, γ, γ⁻¹, d, d⁻¹, d², β_0+d², and per round u, v, u·α,
  v·α⁻¹, χ, Δ1R, Δ2R, plus `Chi(σ)` (the unit alone), s1_acc, s2_acc, HT, −γd⁻¹s1, −γ⁻¹ds2.
  Index convention (native `process_round`, `num_rounds` counting down from σ): round `j` folds
  with `Δ1R[σ − j]`, `Δ2R[σ − j]` and `Δ1L = Δ2L = χ[σ − j − 1]`, so `Delta1R(k)`/`Delta2R(k)` carry
  the setup index `k = σ − j` and `Chi(k)` collects round `j = σ − 1 − k` plus the unit term; every
  base of the deferred right-hand side (`C_init`, `C_i`, `D2_init`, per-round `C±`, `D1L/R`, `D2L/R`,
  `χ[0..=σ]`, `Δ1R[1..=σ]`, `Δ2R[1..=σ]`, `HT`) has a named scalar.
- `state_in` (witness): the transcript state after the natively absorbed preamble and commitments —
  where the hidden segment begins.

## GLV split decision

Signed-digit (w = 5) mini-scalar bits are **not** R1CS wires. 174 Dory scalars (plus 41 RLC weights)
× ≈ 2 × 26 digits × 6 booleans ≈ 55k booleanity rows would dwarf the 5.3k relation; T2's limb
table verifies its own decomposition (a booleanity zero-check inside its sumcheck) against the
scalar wires named in `DoryLinks::scalars`. The relation exposes scalars only.

## Design notes

- Per-round degree profile: the prover trims batched round polynomials, so a batch's round degree
  is the max emitted degree of the members active in that round (stage 5: 128 rounds at 2, then
  `D + 2 = 10`; stage 6a: 8 rounds at 2 then 4 at 3 — the booleanity address phase is suffix-aligned).
- Aliased openings share wires (stage 2 instruction claim reduction → product remainder; stage 3
  instruction input → shift, registers → instruction input; stage 6b booleanity `bytecode_ra[0]` →
  bytecode read-RAF, detected structurally by wire-vector equality).
- The 54 table MLEs are hand-transcribed gadgets over the 128 interleaved wires sharing `x_i·y_i`, the
  equality prefix, the SRL/PEXT recurrence and the `Π(1−y)` chain (971 rows for all 54).
- Unsupported shapes (typed errors): committed program, advice, ZK, address-major trace order,
  chunk width larger than the bytecode address.

## Tests

```text
cargo nextest run -p jolt-wrapper --cargo-quiet -E 'test(relation::)'          # 54 tables × 200 random points vs evaluate_mle; gadget parity vs jolt-poly (eq, lt, eq+1, Lagrange, kernel, identity, operands, eq_index, range mask, point regrouping, chunks)
cargo nextest run -p jolt-wrapper --features prover-fixtures --cargo-quiet --test relation_fixture --no-capture
cargo nextest run -p jolt-wrapper --features prover-fixtures --cargo-quiet --test relation_fixture --run-ignored ignored-only  # 2^20 (proves on first run)
cargo nextest run -p jolt-wrapper --features prover-fixtures --cargo-quiet --test relation_dory_native  # named Dory scalars × real setup/proof bases == native pairing equation
cargo clippy -p jolt-wrapper --all-targets --features prover-fixtures -q --message-format=short -- -D warnings
```

`relation_fixture`: real proof → build, witness, `check_witness`; pinned total and per-stage row
counts (2^18 and, ignored by default, 2^20); outsourced inputs recomputed from public outputs; tampers: a stage-1 round coefficient fails
at `stage1/remainder` row 38, `ValIo` at `stage2/expected` row 626, Dory γ at `stage8/dory` row 5082.
`native_parity` (asserted in `relation_fixture` at 2^18): 214 derived ids and 19 challenge ids
compared with their native owners; the guard also fails if any registered wire has no owner.

`relation_dory_native`: the deferred Dory equation re-evaluated with every scalar taken from the
witness by `DoryScalar` name and applied to the real setup constants, commitments and proof elements
(four-pairing left-hand side vs GT right-hand side, the M1 bench closed form); the accept must hold,
and two negative controls (Delta links paired with the `σ − 1 − j` neighbour; Δ1R/Δ2R scalars swapped)
must break it.

## Native parity guard (`relation/native.rs`)

Runs inside `generate_witness` after the walk. `replay()` now also keeps the native stage outputs
(`Stage1Output`…`Stage7Output`), `CheckedInputs`, the formula dimensions and the squeeze values with
each stage's draw start. Per stage the guard rebuilds the verifier's own `Stage*Sumchecks` exactly
the way `stageN::verify` does (`OuterRemainder::new`, `Stage2BatchSumchecks {..}`, …,
`Stage6aSumchecks::build_from_parts`, `Stage6bSumchecks::build_from_parts`, `build_stage7_sumchecks`),
re-draws the stage challenges from the recorded squeezes through the generated `draw_challenges` /
`Stage6bDraws::draw`, takes the input points from the upstream helpers and the output points from
the native stage output, then for every `Source::Derived` / `Source::Challenge` factor of each
member's input and output expression compares `derive_input_term` / `derive_output_term` /
`resolve_challenge` with the wire value. Two owners expose their publics differently: the product
uni-skip (`ProductUniskip::derive_input_term` for the Lagrange weights) and the stage-6b bytecode
read-RAF, which folds its publics inside `expected_output`; there the same public functions it
calls (`read_raf_committed_public_values`, `stage_values_at_r_address`) give the per-id values.
Stage 1 now also registers the native `AzWeight(i)`/`BzWeight(i)`/`AzConstant`/`BzConstant`/`TauKernel`
wires (linear combinations of the row weights the circuit already multiplies), so the guard covers
the Spartan outer linear forms.

## Review #1 response

1. BLOCKER — `Delta1R`/`Delta2R` now carry the setup index `σ − j` (`Chi` stays at `σ − 1 − j`;
   `Chi(σ)` emitted as the unit); enum docs corrected; `relation_dory_native` is the test that would
   have caught it (it fails with the old index — the negative control is exactly that shift).
2. MAJOR — native parity guard as above, typed error, always on in assign mode; pinned coverage
   214/19 in `relation_fixture`. Finding it exposed nothing in stages 1–7 beyond the unregistered
   stage-1 linear-form weights (now registered).
3. MINOR — inventory corrected: 137 opaque elements = 68 GT, 35 G1, 34 G2 (29,408 bytes).
4. MINOR — `table_gadget_values` and `tests/relation_tables.rs` removed; the internal
   `relation::tables::tests::table_gadgets_match_native_mles` stays.

Row counts moved by the `Chi(σ)` wire only: stage 8 279 (2^18) / 292 (2^20); totals 5,254 / 5,455.

The 2^18 fixture is the transcript-table lane's cached triple
(`/Volumes/Dev/scratch/wrapper-fixtures/fibonacci_2_18_blake3.bin`).
