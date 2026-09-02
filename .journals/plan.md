# Plan — Spartan+HyperKZG wrapper for curve Jolt (card 135)

Inputs: .journals/discovery/{proof-size,verifier-structure,blindfold-and-r1cs-tooling,prior-art,prover-stack-prior-art,throughput,hyperkzg-perf}.md

## Constraints
- Proof size: lower as much as possible (parent target single-digit KB).
- Wrapper prover < 1 s wall-clock on this Mac mini (10 cores, 16 GiB, CPU). MEASURED budget: HyperKZG commit+open (after lane G) 2^18: 137+336 ms, 2^19: 334+669 ms; Spartan-side (lane F) 2^18 ≈ 156 ms, 2^19 ≈ 312 ms → total 2^18 ≈ 590 ms OK, 2^19 ≈ 1,250 ms FAIL. **Hard budget m, n ≤ 2^18 = 262,144.**
- Keep Dory as the inner PCS (user spec).

## Decision (2026-09-02 16:10)
In-circuit Dory verification is infeasible under the <1 s budget by ≥3 orders of magnitude: ≈3.2M Fq mults online (165 GT exps, 177 GT mults, 40 G1/G2 smuls, 1 four-way multipairing @L=20) → ≈3–5B plain-R1CS constraints (FqVar mul 2,716), ≈130–200M even with hypothetical lookup-based Fq muls; the 41-term GT RLC alone ≈0.7B. Deferring only the pairing saves <1%. 2-cycle (Grumpkin) designs need ≥3–8 s. ⇒ **Dory stays native.** The wrapper hides everything else: stages 1–7 sumcheck round polys (28–34 KB), opening claims (8.5 KB), the stage-8 RLC/reduction algebra, and the Fiat–Shamir transcript over the hidden data.

Resulting proof = wrapper (Spartan outer+inner + HyperKZG open + public IO ≈ 5–6 KB) + 41 GT commitments + Dory opening proof (6σ+2 GT, 3σ+2 G1, 3σ+1 G2; σ = ⌈(L+4)/2⌉).
- With today's 384-B GT encoding: ≈ 52 KB @2^18 … 59 KB @2^24 (vs 82–95 KB).
- With the fork's unused CompressedFq12 (128 B) codec on GT (commitments + Dory proof): ≈ 23 KB @2^18 … 27 KB @2^24 (−3.5×). Independent lane; lands with the wrapper.
Single-digit KB with Dory retained is NOT reachable under <1 s on this hardware. Alternatives (need user decision, not pursued): inner PCS → HyperKZG (≈10–11 KB, but Jolt prover +~20–40 s @2^20 for 2^24-coefficient opens); Grumpkin 2-cycle (≈7–10 KB, wrapper ≥3–8 s); Groth16-of-everything (minutes).

## Relation design principles
1. In-circuit only what must be hidden AND is cheap in Fr: sumcheck round algebra (deg+2 constraints/round; 290 rounds @L=20), uni-skip power-sum/Lagrange, eq/EqPlusOne, input/output-claim sum-of-products (Blindfold already has these declaratively), stage-1 Spartan fold (6.5k), stage-5 lookup-table MLEs (12–14k), RLC of 123 openings → 41. Total ≈ 25–30k.
2. Outsource via public IO everything the native verifier can compute from public data at a public point: (a) preprocessing/commitments/program-IO absorbs → initial sponge state (public input); (b) O(code_size)/O(image) MLE evaluations (stage-6b bytecode fold, stage-4 initial-RAM MLE, stage-2 IO MLE) → the circuit exposes the evaluation point(s) as public outputs and takes the values as public inputs; the native verifier computes them; (c) sponge state at the Dory boundary + batched opening claim/point → public outputs; the native verifier squeezes the 26 Dory challenges itself and runs Dory natively.
3. Transcript: field-native Poseidon sponge (absorb Fr as one rate slot; bytes/GT/G1 as packed limbs), ≥1 permutation per challenge point (~400 @L=20) → 140k (Poseidon2 t=8, 360 constr/perm; params must be generated) or ≈200–220k (light-poseidon circom t=9/t=13, in-tree params). Hidden absorbs ≈1,300 Fr + 371 squeezes.
4. Challenges: FULL-Fr in the wrapped profile (in-circuit 128-bit truncation = 254 constraints × 398 ≈ 101k — over budget). Measure Jolt prover delta vs 125-bit challenges; fallback only if budget allows.
5. Spartan verifier does O(nnz) A/B/C evaluation (≈2–5M Fr ops ≈ 50–100 ms) and rebuilds the R1CS from the profile (L, K, config) — no SPARK in v1 (deferred; on-chain not a goal).
Budget check: 25–30k algebra + 140–220k transcript + ~10k IO/RLC ≈ 175–260k ≤ 262k. Tight: the planner must produce an exact count and a go/no-go; 2^19 fallback costs ≈1.25 s (needs GLV + Spartan-side 2× to recover <1 s).

## Component ladder (dependency order)
0. [done] HyperKZG restored (992ad9d23) + fast MSM/open (abab852a5, cfa02939f, 6634e39f2).
1. Field-native Poseidon transcript in jolt-transcript (+ exact byte/field encoding spec for GT/G1/G2/u64 absorbs, full-Fr challenges); prove+verify a modular Jolt proof with it; measure Jolt prover delta.
2. CompressedFq12 GT codec on the wire for Dory commitments + proof (fork codec; verifier decompress + existing subgroup checks). Independent.
3. jolt-wrapper crate: R1CS with public-input column (reuse jolt_r1cs::R1csBuilder), gadgets: Poseidon permutation (bit-exact with #1), sumcheck round (Horner + sum), uni-skip, eq/EqPlusOne/Lagrange, RLC; constraint-count report.
4. Spartan (plain R1CS, public inputs) prover + verifier + HyperKZG binding, forked from jolt-blindfold's outer/inner sumchecks; proof struct + bincode; VK = deterministic rebuild from profile.
5. Jolt-verifier relation + witness generator (stages 1–7 + stage-8 algebra to the Dory boundary; public IO per principle 2). Strategy decided by the planner: Blindfold's declarative StageConfig/InputClaimConstraint walker + gadgets for challenge-derived values vs. making jolt-verifier stage code generic over a Var type.
6. wrap()/verify_wrapped() pipeline + e2e (fibonacci, sha2-chain @2^18–2^20; tamper tests) + size/time table (bytes + wrapper wall-clock, first-class).
7. Gates (clippy both modes, nextest, e2e) → draft PR (architecture, proof-size + wall-time table, security argument, deferred: SPARK, ZK, on-chain, inner-PCS alternatives).
