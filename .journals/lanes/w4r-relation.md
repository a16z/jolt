# W4-R — verifier stage algebra as an R1CS (`jolt-wrapper::relation`)

Date: 2026-09-02. Branch: `wrap/spartan-hyperkzg`. Lane: W4-R (relation + profile).

## Result

- `profile.rs`: `WrapperProfile` (log_t, log_k_ram, log_k_bytecode, rw/one-hot configs, trace order,
  memory layout, program-image length, entry index) + `digest()` (blake3 over bincode). Rejects
  committed-program preprocessing, untrusted-advice proofs and ZK proofs.
- `relation/`: `build_relation(profile) -> Relation { matrices: ConstraintMatrices<Fr>, public:
  PublicLayout, link: LinkTable, rows: Vec<RowSpan> }` and `generate_witness(profile, preprocessing,
  public_io, proof) -> Witness { values, state_in, outsourced }`. One walk in two modes: build mode
  allocates unknown wires; assign mode replays a `Recording<Blake3Transcript>` log of the native
  verifier (every `append_bytes` payload and squeeze), asserting the schedule event-for-event
  (labels, counts, lengths, verifier-computed values).
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
| stage8/dory | 193 | 24 inverses, 41 commitment weights, per round: u·α, v·α⁻¹, χ (2), Δ1R, Δ2R, s1/s2 (4) |
| stage8/evaluation_claim | 1 | joint claim copy |
| **total** | **5,253** | 6,760 variables (incl. constant one), 45 public |

Per stage: 1: 268 · 2: 365 · 3: 342 · 4: 212 · 5: 1,790 · 6a: 229 · 6b: 974 · 7: 795 · 8: 278.

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
  376 squeezes (310 `challenge()` 125-bit + 66 `challenge_scalar()` 128-bit), 89 opaque Dory
  elements (29,408 bytes: 68 GT × 384, 24 G1 × 32, 24 G2 × 64), 22,336 bytes of constant labels.
- `dory`: `num_vars 22, sigma 11`, 174 named scalars (`DoryScalar`): evaluation `y`, 41 commitment
  weights `ρ^i·β_0⁻¹`, β/β⁻¹/α/α⁻¹ × 11, γ, γ⁻¹, d, d⁻¹, d², β_0+d², and per round u, v, u·α,
  v·α⁻¹, χ, Δ1R, Δ2R, plus s1_acc, s2_acc, HT, −γd⁻¹s1, −γ⁻¹ds2.
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
cargo nextest run -p jolt-wrapper --cargo-quiet --test relation_tables          # 54 tables × 200 random points vs evaluate_mle
cargo nextest run -p jolt-wrapper --cargo-quiet -E 'test(relation::)'          # gadget parity vs jolt-poly (eq, lt, eq+1, Lagrange, kernel, identity, operands, eq_index, range mask, point regrouping, chunks)
cargo nextest run -p jolt-wrapper --features prover-fixtures --cargo-quiet --test relation_fixture --no-capture
cargo nextest run -p jolt-wrapper --features prover-fixtures --cargo-quiet --test relation_fixture --run-ignored ignored-only  # 2^20 (proves on first run)
cargo clippy -p jolt-wrapper --all-targets --features prover-fixtures -q --message-format=short -- -D warnings
```

`relation_fixture`: real proof → build, witness, `check_witness`; pinned total and per-stage row
counts; outsourced inputs recomputed from public outputs; tampers: a stage-1 round coefficient fails
at `stage1/remainder` row 38, `ValIo` at `stage2/expected` row 626, Dory γ at `stage8/dory` row 5082.
Per-derived-id parity is covered by the gadget unit tests (one per formula family) plus the real-proof
satisfaction, which exercises every derived id's gadget against the prover's actual claims.

The 2^18 fixture is the transcript-table lane's cached triple
(`/Volumes/Dev/scratch/wrapper-fixtures/fibonacci_2_18_blake3.bin`).
