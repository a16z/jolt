Draft. Design record and per-lane review journals live on branch `wrap/spartan-hyperkzg` (`.journals/`); this branch is that tree without the journals.

## What this is

`crates/jolt-wrapper`: a single-layer wrapper that proves "the Jolt verifier accepts this Dory-committed proof for (program digest, IO)" with **one R1CS proven by Spartan, one batched sumcheck stream over a common `2^18` row domain, and one HyperKZG opening**. Inner PCS stays Dory (commitments unchanged), inner transcript stays Blake3, no trusted setup beyond the universal KZG SRS, no Groth16, no Poseidon, CPU-only prover.

Measured on the real fibonacci `2^18` proof (Mac mini M4, 10 threads), default packing `k=32`:

| | k=32 | k=16 |
|---|---:|---:|
| proof payload / bincode | **7,488 B / 7,628 B** | 7,776 / 7,928 B |
| statement (11 Fr: program+profile digest, IO, T1 state/tail) | 352 B | 352 B |
| verifier ops (ecMul / ecAdd / pairing pairs / Fr mul / Fr inv / Keccak) | 234 / 233 / 8 / 127,884 / 10 / 857 | 247 / 246 / 8 / 172,364 / 10 / 864 |
| EVM gas, op-count model (Cancun unit costs) | **≈ 5.05 M** | ≈ 6.05 M |
| wrapper prover, online (excl. inner Jolt proof) | ≈ 27–38 s under load 10–20 (idle profile pending) | ≈ 33 s |
| native verify | 35 ms | 27 ms |

Bytes, itemized (k=32): wire commitments 672 (21 groups × 32) · Spartan outer 13 committed rounds 864 · Spartan inner 13 clear rounds 832 · stage A 18 committed rounds 1,184 · term stage 9 committed rounds 608 · shared BDFG + degree-shift 96 · 4 factor evaluations 128 · stage B 11 clear rounds 704 · reduced claims (opening + Az/Bz/Cz/W) 160 · HyperKZG (ℓ = 23) 2,240.

## Architecture

- **R — verifier algebra as R1CS, proven by Spartan** (`relation/`, `spartan.rs`): the Jolt verifier's Fr algebra (stages 1–8, 54 table-MLE gadgets, Dory scalar algebra) lowered to an exact R1CS of **5,254 constraints × 6,761 variables** (constant + 7 public statement fields + 6,753 private). The private witness W is one sparse Fr column committed in phase 1a; Spartan outer sumcheck (13 KZG-committed degree-3 rounds sharing the stream's BDFG opening) → Az/Bz/Cz claims → inner sumcheck (13 clear degree-2 rounds) → W(ry) carried into stage A by an 18-round eq member so the final HyperKZG opening also settles the R1CS witness claim. Ã/B̃/C̃(rx, ry) are evaluated natively by the verifier from the key's sparse matrices (34,945 nonzeros; 87,081 Fr multiplications with shared eq tables) — no SPARK, no committed matrices.
- **T1 — transcript table** (`hash_table/`): the inner Blake3 transcript replayed as G-step rows (163 committed bit columns + wired copies + 64 canonicality bits; 128-row cells; VK-pinned constants/padding; symbolic schedule fixed at key time from the profile). 232 terms, degree 2.
- **T2 — limb table** (`limb_table/`): the Dory deferred final check (GT multi-exp over 9σ+N+4 bases, G1/G2 Straus multi-exps, 4-pair Miller loop, final exponentiation) in 96-bit limbs / 16-bit chunks with grouped-inverse LogUp range checks, 16-row cells, committed radix-16 digits with unique recoding, transcript-derived Straus offsets θ, guarded EC adds where degeneracy is prover-influenceable, ψ-chain subgroup checks on the two proof-derived pairing inputs, GT norm-1 checks, Fq canonicality, sign-flag checks. 201,575 rows, 149 columns over 5 commitment phases, 176 terms, degree 4, verifier 9,973 Fr (execution-derived, compile-time constants). Decomposed data is committed, never sent.
- **Links** (`links/`): ten LogUp `CopyLink`s bind T1→W (376 squeeze outputs, 1,200 absorbed Fr words), T1→T2 (45,152 element bytes over 1,526 input rows, incl. sign flags and the commitment permutation); the 7 statement fields sit in Spartan's public segment; an occurrence-weighted scalar link binds W's 173 Dory scalars to T2's Straus digits. Every value the inner verifier consumes is bound by exactly one mechanism; the key checks the T1 state/tail and the program/profile digest.
- **Stream** (`stream/`): k-packed commitments (`packed[row·k + slot]`, canonical padding, VK groups off the wire), ordered commitment phases with Fiat–Shamir challenges drawn only after the last commitment they batch (1a: T1 + W → 38 T1 randomizers, θ · 1b: T2 chunks/signs/digits → ξ, α, ten CopyLink (β, γ), ρ · 2a → 1 · 2b → 3 · 2c + helpers → 232 row/member challenges), then stage A = ONE batched sumcheck (KZG-committed round polynomials: 1 G1 + 1 Fr per round, one batched `S(0)`, one BDFG20 opening + one shifted-commitment degree check shared by all committed stages, G1-side pairings only), then **claim compression**: no column evaluations are sent — the batched final claim Σ coeff(t)·Π L_{t,j}(v) over T = 510 terms is proven by a 9-round term-index sumcheck ending in 4 linear functionals, stage B reduces them to one packed evaluation over all groups, and HyperKZG opens the eq-weighted RLC once (P₀(r²)-only evaluation vector). Outer transcript: chained Keccak256 digests. `VerifierCost` is execution-derived (counting shims on the verifier path) and feeds the gas model.
- **Keys**: `WrapVerifierKey::new(profile, srs)` derives the R1CS, every link plan, pin and phase from the profile (one recorded reference run at key generation; trust assumption stated in `wrap.rs`); `verify_wrapped_with_key(&key, &statement, &proof)` takes nothing from the proof to build the relation.

Supporting changes: `jolt-hyperkzg` (restored + `shift_g2`, exact SRS powers, parallel evaluation/division, fixed-base setup), `jolt-crypto` (batch-affine Pippenger with skew census, small-scalar and bit-column MSM kernels, compressed GT), `jolt-transcript` (`Blake3Transcript`, `Keccak256` digest transcript), `jolt-r1cs` (Blake3/bit gadgets, column range contributions), `jolt-dory` (compression fix).

## Security argument (summary)

- Statement soundness reduces to the inner verifier: T1 fixes the transcript bytes and hence all inner challenges; R evaluates the verifier algebra on the linked values; T2 evaluates the Dory pairing equation on the linked elements/scalars; the links make the three views one proof.
- Fiat–Shamir order: every batching challenge (CopyLink β/γ, scalar-link ρ, LogUp ξ/α, Straus θ, Spartan τ, stage/term/member challenges) is derived after the last commitment on either side of the identity it batches; the real gate pins the per-phase challenge counts (39 / 23 / 1 / 3 / 232).
- Group-element hygiene: on-curve pins for all proof G1/G2 points; subgroup membership enforced only where it matters — on the G2 pairing inputs (torsion components of intermediate points either cancel in the full-group accumulators or reach a pairing input and are rejected); GT inputs are norm-1 (torus) so the cyclotomic group splits as G_T × H with gcd(r, |H|) = 1; every Fr/Fq byte-linked value is canonical (top-64-bit window checks, completeness loss ≈ 2^−60 per element, documented).
- EC degeneracy: affine adds are incomplete; Straus offsets R = [θ]G, Z0 = [θ]G′ are wrapper Fiat–Shamir challenges drawn after the commitments that bind the proof points, so any exceptional add is a one-root equation in θ; correction adds and ψ-chains use guarded adds (inv·(x₂−x₁) = 1); digits are uniquely recoded per occurrence (window check), so no digit is chosen after θ.
- Sumcheck/PCS: KZG-committed rounds bound in degree by the shifted-commitment check (common bound 5 — the outer rounds' declared degree 3 is not separately enforced, so their per-round soundness term is 5/|Fr|); Spartan's public-column contributions are verifier-computed and subtracted before the inner claim; term stage and stage B are standard sumchecks with verifier-computed weights; HyperKZG with exact SRS bound and G1-side pairing form.
- Process: every table and the assembly went through fresh adversarial reviews with scratch repro tests until 0 blockers (T1 ×4, T2 ×5, stream ×6, assembly ×4); every serialized proof field and every witness class has a tamper test that must reject (`tests/wrap_real_t1_r.rs`, feature `prover-fixtures`), including a changed T1 public-preamble word and program/profile mismatch.

## Known gaps / deferred

- **Prover time** is the open item: ≈ 27–38 s online at k=32 under load. Profile: HyperKZG open of the 2^23 packed polynomial ≈ 9 s, 96 full-width LogUp inverse columns (T2 phase 2a) ≈ 7.5 s, stage A ≈ 6 s, 20 full-width CopyLink helper columns ≈ 2.8 s, R/T2 adaptation ≈ 4.5 s. A prover-time campaign is in progress on the design branch; levers and floors are recorded there.
- Gas is an op-count model (ecMul 6k, pairing pairs, Fr mulmod, Keccak, calldata), no Solidity verifier yet; native matrix evaluation (87k Fr multiplications) is the largest EVM cost after the MSMs.
- Key generation records one reference run of the inner verifier to fix the symbolic transcript schedule; the verifier never reads it from the proof, but the reference must be honest (trust assumption at key generation).
- Completeness losses from canonicality/recoding windows (≈ 2^−60 per element) are accepted and documented.
- Zeromorph variant built and reviewed but parked (`wrap/zeromorph-archive`); PCS stays HyperKZG by decision.
