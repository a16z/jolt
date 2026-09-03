Draft. Not ready for review of correctness claims beyond what is stated below; design record and per-lane review journals live on branch `wrap/spartan-hyperkzg` (`.journals/`).

## What this is

`crates/jolt-wrapper`: a single-layer wrapper that proves "the Jolt verifier accepts this Dory-committed proof for (program digest, IO)" with **one batched sumcheck stream and one HyperKZG opening**. Inner PCS stays Dory (commitments unchanged), inner transcript stays Blake3, no trusted setup beyond the universal KZG SRS, no Groth16, no Poseidon, CPU-only prover.

Measured on the real fibonacci `2^18` proof (Mac mini M4, 10 threads), default packing `k=32`:

| | k=32 | k=16 |
|---|---:|---:|
| proof payload / bincode | **5,728 B / 5,836 B** | 6,016 / 6,136 B |
| statement (11 Fr: program+profile digest, IO, T1 state/tail) | 352 B | 352 B |
| verifier ops (ecMul / ecAdd / pairing pairs / Fr mul / Fr inv / Keccak) | 185 / 184 / 8 / 40,722 / 8 / 755 | 198 / 197 / 8 / 33,539 / 8 / 762 |
| EVM gas, op-count model (Cancun unit costs) | **≈ 2.88 M** | ≈ 2.85 M |
| wrapper prover, online (excl. inner Jolt proof) | ≈ 38 s (load ≈ 25) | ≈ 33 s |
| offline: SRS 2^23 / key commitments | 7.7 s / 1.2 s | 3.8 s / 0.8 s |
| native verify | 21 ms | 21 ms |

Bytes, itemized (k=32): wire commitments 672 (21 groups × 32) · stage A 18 committed rounds 1,184 · term stage 10 committed rounds 672 · shared BDFG + degree-shift 96 · 4 factor evaluations 128 · stage B 11 clear rounds 704 · reduced claim 32 · HyperKZG (ℓ = 23) 2,240.

## Architecture

Three row tables over one common `2^18` row domain, all proven inside ONE stage-A batched sumcheck (KZG-committed round polynomials: 1 G1 + 1 Fr per round, one batched `S(0)`, one BDFG20 opening + one shifted-commitment degree check for both committed stages, G1-side pairings only):

- **T1 — transcript table** (`hash_table/`): the inner Blake3 transcript replayed as G-step rows (163 committed bit columns + wired copies + 64 canonicality bits; 128-row cells; VK-pinned constants/padding; symbolic schedule fixed at key time from the profile). 232 terms, degree 2, verifier 4.9k Fr.
- **T2 — limb table** (`limb_table/`): the Dory deferred final check (GT multi-exp over 9σ+N+4 bases, G1/G2 Straus multi-exps, 4-pair Miller loop, final exponentiation) in 96-bit limbs / 16-bit chunks with grouped-inverse LogUp range checks, 16-row cells, committed radix-16 digits with unique recoding, transcript-derived Straus offsets θ, guarded EC adds where degeneracy is prover-influenceable, ψ-chain subgroup checks on the two proof-derived pairing inputs, GT norm-1 checks, Fq canonicality, sign-flag checks. 201,575 rows, 149 columns over 5 commitment phases, 177 terms, degree 4, verifier 9,973 Fr (execution-derived, compile-time constants).
- **R — verifier-algebra row table** (`relation_table/`, lowered from the R1CS in `relation/`): the Jolt verifier's Fr algebra (stages 1–8, 54 table-MLE gadgets, Dory scalar algebra) as 38,981 Plonkish gates with VK-committed selectors and a LogUp copy argument; 16 terms.
- **Links** (`relation_table/copy_link.rs`, `scalar_link.rs`): 11 generic LogUp `CopyLink`s bind T1→R (376 squeeze outputs, 1,199 absorbed Fr words), T1→T2 (45,152 element bytes over 1,526 input rows, incl. sign flags and the commitment permutation), statement→R (7 public fields); an occurrence-weighted digit link binds R's 173 Dory scalars to T2's Straus digits. Every value the inner verifier consumes is bound by exactly one mechanism; the key checks the T1 state/tail and the program/profile digest.
- **Stream** (`stream/`): k-packed commitments (`packed[row·k + slot]`, canonical padding, VK groups off the wire), ordered commitment phases with Fiat–Shamir challenges between them (1a → θ → 1b → LogUp challenges → 2a/2b/2c → helpers → member challenges), stage A, then **claim compression**: no column evaluations are sent — the batched final claim Σ coeff(t)·Π L_{t,j}(v) over T = 535 terms is proven by a 10-round term-index sumcheck ending in 4 linear functionals, stage B reduces them to one packed evaluation over all 36 groups, and HyperKZG opens the eq-weighted RLC once (P₀(r²)-only evaluation vector). Outer transcript: chained Keccak256 digests. `VerifierCost` is execution-derived (counting shims on the verifier path) and feeds the gas model.
- **Keys**: `WrapVerifierKey::new(profile, srs)` derives every relation, selector, link plan, pin and phase from the profile (one recorded reference run at key generation; trust assumption stated in `wrap.rs`); `verify_wrapped_with_key(&key, &statement, &proof)` takes nothing from the proof to build the relation.

Supporting changes: `jolt-hyperkzg` (restored + `shift_g2`, exact SRS powers, parallel evaluation/division, fixed-base setup), `jolt-crypto` (batch-affine Pippenger with skew census, small-scalar and bit-column MSM kernels, compressed GT), `jolt-transcript` (`Blake3Transcript`, `Keccak256` digest transcript), `jolt-r1cs` (Blake3/bit gadgets, column range contributions), `jolt-dory` (compression fix).

## Security argument (summary)

- Statement soundness reduces to the inner verifier: T1 fixes the transcript bytes and hence all inner challenges; R evaluates the verifier algebra on the linked values; T2 evaluates the Dory pairing equation on the linked elements/scalars; the links make the three views one proof.
- Group-element hygiene: on-curve pins for all proof G1/G2 points; subgroup membership enforced only where it matters — on the G2 pairing inputs (torsion components of intermediate points either cancel in the full-group accumulators or reach a pairing input and are rejected; the "cleaned" proof satisfies the same equations under the same challenges); GT inputs are norm-1 (torus) so conjugation is inversion and the cyclotomic group splits as G_T × H with gcd(r, |H|) = 1; every Fr/Fq byte-linked value is canonical (top-64-bit window checks, completeness loss ≈ 2^−60 per element, documented).
- EC degeneracy: affine adds are incomplete; Straus offsets R = [θ]G, Z0 = [θ]G′ are wrapper Fiat–Shamir challenges drawn after the commitments that bind the proof points, so any exceptional add is a one-root equation in θ; correction adds and ψ-chains use guarded adds (inv·(x₂−x₁) = 1); digits are uniquely recoded per occurrence (window check), so no digit is chosen after θ.
- Sumcheck/PCS: KZG-committed rounds bound in degree by the shifted-commitment check; term stage and stage B are standard sumchecks with verifier-computed weights; HyperKZG with exact SRS bound and G1-side pairing form.
- Process: every table and the assembly went through fresh adversarial reviews with scratch repro tests until 0 blockers (T1 ×4, T2 ×5, stream ×6, R + assembly ×2); every serialized proof field and every witness class has a tamper test that must reject (`tests/wrap_real_t1_r.rs`, feature `prover-fixtures`).

## Known gaps / deferred

- **Prover time**: ≈ 38 s online at k=32 on the mini; < 1 s is not reachable on CPU under the ≤ 6 KB budget (HyperKZG open = 2·2^23 full-width points ≈ 8.9 s; phase-2a commits 7.4 s; stage A 4.5 s dominated by CopyLink members binding full columns despite sparse selectors — shared batching is the next lever; helper commits; typed RLC). Ranked levers and measured floors in the design record.
- Gas is an op-count model (ecMul 6k, pairing pairs, Fr mulmod, Keccak, calldata), no Solidity verifier yet; the ~30k Fr multiplications of the term/link weights are the largest EVM cost after the MSMs.
- Key generation records one reference run of the inner verifier to fix the symbolic transcript schedule; the verifier never reads it from the proof, but the reference must be honest (trust assumption at key generation).
- Completeness losses from canonicality/recoding windows (≈ 2^−60 per element) are accepted and documented.
- `T2` operand columns are committed after the ξ combination (full-width); a signed-u128 MSM tier would need a typed export.
- Zeromorph variant built and reviewed but parked (`wrap/zeromorph-archive`); PCS stays HyperKZG by decision.
