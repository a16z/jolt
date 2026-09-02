# Plan v3 architecture — size/gas-ranked re-budget (lane M5, planner, read-only)

Date 2026-09-02 19:38–20:20 · tree c229590c4 (wrap/spartan-hyperkzg) · objective per USER DECISION 19:34 (`plan.md` tail):
rank designs by (1) proof bytes → (2) EVM verifier gas → (3) prover seconds (2.5 s fine, more if it buys KB).
Inputs (do not re-derive): `plan-relation.md` §2/§4/§5/§8 (R1CS side, public IO, witness replay), `lanes/dory-offload-study.md` §1–§3
(Dory closed form, 9σ+N+4 GT bases), `lanes/m1-gt-multiexp-counts.md` (exact rows), `lanes/m2-limb-relation.md` §6 (limb relation
measured), `lanes/m3-blake3-transcript.md` (C3, hash relation, microbench), `lanes/m6-bit-commit-kernel.md` (bit commits), `lanes/m4-blake3-gadget.md`
(Blake3 R1CS 15,536/continuation block). "exact" = counted from code/lanes; "est." marked. Profile fibonacci L=20 (K=16, σ=12, N=41 [42 real]);
L=18 (σ=11) in parentheses where it differs.

## 0. Verdict

| rank | design | proof bytes (wire) | EVM calldata | est. gas | prover s | verifier | setup |
|---|---|---:|---:|---:|---:|---|---|
| **1** | **D4** two layers, Groth16 final (layer 1 = sumcheck-offload wrapper, k=8 packing; layer 2 = Groth16 over Blake3-FS verification of layer 1, HyperKZG pairing deferred on-chain) | **≈1.2 KB** (128 Groth16 + 672 deferred G1 + 256 public Fr + Jolt IO) | ≈2.0 KB | **≈560k** (≈450k with an O(1)-proof KZG multilinear PCS, lever L6) | ≈11–14 s (layer 1 7.7 + Groth16 3.35M·g µs, g unmeasured → lane N2) | EVM or native | Groth16 circuit CRS per profile + universal 2^21 SRS |
| 2 | D2 single layer + KZG-committed round polys, k=16 | **8.8 KB** | — (native verifier only) | n/a on-chain (see D5) | 10.4 s (k=16) / 7.7 s (k=8, 9.2 KB) | native ≈0.4 s (O(rows) wiring) | universal 2^22 SRS |
| 3 | D3 two layers, Spartan+HyperKZG final | 8.3 KB | — | n/a (needs SPARK: +≈3 KB, ≥1M gas est.) | ≈23 s | native ≈0.15 s | universal 2^22 SRS |
| 4 | D1 single layer, size-optimized, k=16 / 8 / 1 | 11.3 / 11.7 / 17.7 KB | — | n/a | 10.4 / 7.7 / 5.3 s | native | universal |
| 5 | D0 current measured shape (lanes M2/M3, 2^17 tables) | 26 KB (≈30 KB for the full statement) | — | n/a | 2.5 s (≈5.3 s full statement) | native | universal |
| 6 | D5 D1 verified directly on-chain | ≈17 KB (D1 + committed wiring/SPARK) | 17 KB | **≈3M** (public-matrix MLEs ≈48k Fr ops) | ≈9 s | EVM | universal |

- Only D4 reaches "a couple of KB". Every single-layer design is floored by its sumcheck round polynomials: 3 stages × 18–28 rounds ×
  (deg 2–5 coefficients) × 32 B ≈ 7.0 KB (4.4 KB with KZG-committed rounds) — no column trick touches that term.
- D4's gas is 55% the deferred HyperKZG check (≈24 ecMul + 2 pairings ≈ 261k), 41% Groth16 (≈230k), 4% calldata/hashes. Layer-1 proof
  size is irrelevant in D4 (it is layer-2 witness); what matters is bytes *hashed* in-circuit (2.9M constraints for 12 KB) and the number of
  deferred G1 points (ℓ+1 = 21).
- Two facts the earlier lanes did not budget, both now in every number below: (a) with GT elements and commitments committed (not sent),
  their transcript absorption is in-table: C_tot = C3 + 290 (42 commitments) + 636 (Dory VMV/rounds/final, σ=12) = **1,980** compressions
  (1,893 @L=18), hash rows 116·C_tot = 229.7k → **2^18** (M3 measured the hidden segment only: 1,054 → 2^17); (b) M1's limb rows are 205k
  (σ=11) / 230.6k (σ=12) → **2^18**, not M2's 122k fused estimate. Both tables at 2^18 double every per-row cost of M2/M3.

## 1. Statement (all designs share layer 1)

Layer 1 proves: "the Jolt verifier (stages 1–8 incl. Dory's deferred check, `dory-offload-study.md` §1.4) accepts proof π for preprocessing digest
d_pp, program IO io, with transcript profile Blake3-streaming (`crates/jolt-transcript/src/blake3.rs`)".

| class | content | where |
|---|---|---|
| public input (verifier-computed) | `state_pre` = Blake3 state after the native preamble absorb (`verifier.rs:580-650`), `val_io`, `init_eval`, `stage_value[0..5]` (plan-relation §5), Dory verifier-key GT constants χ[k], Δ1R, Δ2R, HT as limb rows' public operands | x column of the R1CS + public operands of the limb table |
| public output (transmitted) | `r_ram_addr` (K), `r_bytecode_addr` (12) → 28 Fr = 896 B | proof |
| committed witness | 42 commitment GTs (as limb operand rows **and** as hash message bytes, 42×416 B), round polys/claims of stages 1–7 (R1CS witness W, mirrored as hash bytes), Dory proof: 6σ+2 GT, 3σ+2 G1, 3σ+1 G2 (limb rows + hash bytes), all GT/G1/G2 intermediates, hash-table bits, limb chunks, LogUp helpers | HyperKZG polys |
| decision: commitments | **committed, absorption proven in-table** (+290 compressions = 33.6k hash rows, +1,320 operand rows). Sending them = 42×128 = 5,376 B (lane J) — alone exceeds the D4 total | — |
| decision: final exponentiation | **in-table** (3,288 rows, M1). Native FE would need the pre-FE Fq12 sent (384 B) or exposed as 12 Fq public outputs (384 B) | — |
| decision: Dory challenges | derived in-table over the committed GT bytes (636 compressions). The "expose `state_rlc`, derive natively" variant needs the Dory proof public (12 KB) | — |

Linking (all as extra public-matrix terms of the wiring sumcheck, no new committed rows): L1 absorbed Fr in W ↔ hash message bits; L2 R1CS
challenge wires ↔ squeeze output words (125/128-bit decode is linear in bits); L3 `state_pre` (public) → first-compression key words,
transcript never exits the table again (Dory challenges are consumed by the limb table's public scalar arithmetic in the R1CS: 143 scalars,
≈600 constraints); L4 GT/G1/G2 limb chunks ↔ hash message bits (16 bits per chunk, linear); L5 Dory Fr scalars (R1CS) ↔ limb rows' public
scalar inputs (Straus digit selectors, §2 W).
