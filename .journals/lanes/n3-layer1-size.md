# Lane N3 — layer-1 proof size vs prover time: column packing, column sumcheck, row/column reshape, one batched stream

Date 2026-09-02 · tree wrap/spartan-hyperkzg (on c229590c4) · Apple M4 mini, 10 Rayon threads, 16 GiB.
Load averages 2.5–7 (own runs) for the hash/batched numbers; the limb k=8/k=16 timings quoted are from
the quiet first campaign (later reruns at load 11 were 1.5× slower and are not used). Every run
verifies (sumcheck final checks + HyperKZG); tamper tests of the limb bench still reject.
Code: `crates/jolt-wrapper-bench` (hash table, reshape, batched stream), `crates/jolt-limb-bench`
(now a lib + bin; `k=` packing), `crates/jolt-hyperkzg` (`setup_from_secret` parallel scalar muls).

## 0. Result — layer-1 core (hash 2^17 + limb 2^17 t=12 + Spartan 2^14 + ONE opening + IO 1 KB)

| config | commitments | claims | round polys | HyperKZG | IO | **total B** | commits s | sumchecks s | opening s | **prover s** | RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A baseline (k=1, per-column claims, 3 separate compressed streams, s=6) | 7,296 | 11,872 | 8,224 | 2,176 | 1,024 | **30,592** | 1.42 | 1.32 (+0.25 build) | 0.22 (+0.17 RLC) | **3.4** | 2.0 GB |
| B k=8 (pack only) | 992 | 11,872 | 8,224 | 2,560 | 1,024 | 24,672 | 1.61 | 1.28 (+0.24) | 1.63 (+0.16) | 4.9 | 2.3 GB |
| B′ k=8 + hash column sumcheck | 992 | 5,344 | 8,224 | 2,560 | 1,024 | 18,144 | 1.61 | 1.28 (+0.24) | 1.63 (+0.16) | 4.9 | 2.3 GB |
| C k=16 (pack only) | 544 | 11,872 | 8,224 | 2,688 | 1,024 | 24,352 | 1.6 | 1.3 (+0.25) | 3.1–3.3 (+0.2) | 6.5 | 3.0 GB |
| C′ k=16 + column sumcheck | 544 | 5,344 | 8,224 | 2,688 | 1,024 | 17,824 | 1.6 | 1.3 | 3.2 | 6.5 | 3.0 GB |
| D reshape: hash bits:1 @2^21 (col) + limb k=1, separate streams | 2,592 | 5,056 | 8,608 | 2,688 | 1,024 | 19,968 | 1.47 | 1.59 (+0.4) | 2.60 (+0.16) | 6.2 | 3.1 GB |
| E reshape + k=8: hash bits:2 @2^20 k=8 (opening 2^23) + limb k=8 | 448 | 5,152 | 8,512 | 2,944 | 1,024 | 18,080 | 1.97 | 1.86 | 10.4 | 14.5 | 5.1 GB |
| F + batched stream (word k=8, col, s=6) | 992 | 5,344 | 4,352 | 2,560 | 1,024 | 14,272 | 1.48–1.61 | 1.16–1.28 (+0.23) | 1.34–1.63 (+0.15) | 4.3–4.9 | 2.3 GB |
| **G + s=3 (word k=8, col, batched, s=3)** | 1,024 | 5,632 | 2,720 | 2,560 | 1,024 | **12,960** | 2.11 | 0.96 (+0.24) | 1.35–1.45 (+0.15) | **4.8–4.9** | 2.7 GB |
| G16 as G with k=16 | 576 | 5,632 | 2,720 | 2,688 | 1,024 | 12,640 | 2.52 | 1.20 (+0.26) | 3.32 (+0.20) | 7.5 | 3.0 GB |
| G1 as G with k=1 (no packing) | 7,584 | 5,632 | 2,720 | 2,176 | 1,024 | 19,136 | 1.98 | 0.95 (+0.23) | 0.19 (+0.17) | **3.5** | 2.35 GB |
| S1 as G with s=1 (degree-3 stream) | 1,152 | 6,784 | 1,632 | 2,560 | 1,024 | 13,152 | 5.00 | 0.90 (+0.28) | 1.35 (+0.17) | 7.7 | 3.1 GB |

Rows A–E are sums of separately measured components (§2), F–S1 are single measured runs of the
batched bench. "claims" = every evaluation sent (hash committed+wired or its column-sumcheck
replacement, limb 64/73 column claims + 72 operand-limb claims, Spartan Az/Bz/Cz/M/Z); "commits" =
hash + limb chunks + limb helpers + W. Not included: the wiring layer (S2 / limb wiring / S3
two-point reduction — modeled in §3.2 at ≈3.4–4.7 KB column-batched, ≈16 KB with per-column claims)
and the bench-only input claims (0 in production: satisfied relations sum to zero).

**Byte breakdown of G (12,960 B):** commitments 1,024 = 32 × (21 hash + 7 limb-chunk + 3 limb-helper
+ 1 W) · claims 5,632 = hash column sumcheck 832 (8 rounds × 3 coeffs + T1(s) + T2(s)) + limb column
claims 2,336 (54 chunks + 18 helpers + multiplicity) + operand-limb claims 2,304 (6t = 72) + Spartan
160 · round polys 2,720 = one stream, 17 rounds × degree 5 × 32 · HyperKZG 2,560 = ℓ = 20: 19 fold
commitments + witness + 60 evaluations · IO 1,024 (orchestrator's estimate, unchanged).
**Recursion-layer hash load:** every byte above is absorbed by the layer-1 verifier → 12,960 B
(11,936 without IO) ≈ 203 64-byte blocks, ≈ 260 Blake3 compressions with the ≈60 squeezes
(vs 30,592 B ≈ 480 blocks for A).

Frontier (bytes ↔ seconds): G1 19.1 KB @ 3.5 s → G 13.0 KB @ 4.9 s → G16 12.6 KB @ 7.5 s.
k = 16 buys 320 B for +2.6 s; the reshapes (D, E) are dominated by B′/G on both axes.

## 1. What was built (bench-only; one production change)

- **Column packing** (`jolt_limb_bench::pack`, `commit::commit_bit_columns`): `k` columns share one
  polynomial of `rows·k` entries, column index in the HIGH variables (`P_g[j·rows + row] =
  column_{g·k+j}[row]`), so the `⌈c/k⌉` packed polynomials all live over the same `rows·k` SRS
  prefix and the M6 shared-base bit kernel still applies (its per-base amortisation drops from 163 to
  21 columns: hash commit 0.08 → 0.23 s at k=8, 0.22 s at k=16). Limb columns are committed per
  column over the slot's SRS window `[j·rows, (j+1)·rows)` (u16 small-scalar MSM / full-width MSM
  unchanged) and summed per group — commit time unchanged. Opening: one HyperKZG open of
  `Σ_g w_g·P_g` (`rows·k` entries) at `(s_lo, r)`; verifier claim `Σ_g w_g Σ_j eq(s_lo, j)·claim_{g·k+j}`
  — the multilinear identity `P_g(s_lo, r) = Σ_j eq(s_lo, j)·c_{g·k+j}(r)`. Tables of different
  heights share the opening by zero-padding at the high end (natural commitment unchanged; the
  claim is scaled by `eq(r_high, 0)`): the Spartan witness W (2^14) rides in the same opening.
- **Claim reduction (b) — column sumcheck** (`sumcheck::ColumnInstance`): the hash relation is
  rewritten as an *aligned quadratic form* `Σ_j γ̃_j v_j² + γ̃'_j v_j w_j + L1_j v_j + L2_j w_j` over a
  committed vector `v` (163 entries, padded to 256) and a wired vector `w` whose XOR operands sit at
  the index of their committed partner (din_k at A'_k, bin_k at C'_k); the three wired words (a_in,
  c_in, and the parity-rotated D' input of the binary add — replacing M3's in-row `sel` selector,
  whose parity-dependent rotation moves into the public wiring matrix) sit at indices 163–165 where
  `v = 0`. After the row sumcheck the final check is `eq(τ,r)·Q(v(r), w(r))`; instead of sending the
  230 values, an 8-round degree-3 sumcheck over the column index reduces `Q` to `v(s), w(s)`
  (768 + 64 B). `v(s) = Σ_g eq(s_hi, g)·P_g(s_lo, r)` is exactly the packed opening with weights
  `eq(s_hi, ·)`; `w(s)` is the input claim of the wiring sumcheck S2 with weights `eq(s, ·)` (S2 not
  in the bench). Verifier work: four 256-entry MLE evaluations. Cost: < 5 ms.
- **Reshape** (`relation::Layout::Bits(b)`): bit-sliced G-step rows holding `b` bit positions: per
  position 16 committed bits (A1 D1 C1 B1 A2 D2 C2 B2, 6 carry bits — 2 per ternary add, 1 per
  binary add — and m0, m1) and 7 wired bits (a, b, c, d inputs; rot16 D1, rot12 B1, rot8 D2 as row
  shifts) plus 6 wired carry-ins for the row's first position; 24 constraints per position (16
  booleanity, 4 XOR, 4 linear adds), degree 3, same aligned-quadratic form. Rows = 32·G-steps/b:
  b = 1 → 2^21 (1.92 M rows, 92 % full at L=20), b = 2 → 2^20, b = 4 → 2^19, b = 8 → 2^18. Random
  data in the exact shape (as M3's bench): the sumcheck proves whatever the sum is; costs are exact.
- **Batched stream** (`sumcheck::prove_stream`, Jolt `prove_batch` conventions): members = hash row
  relation (17 rounds, deg 3), limb relation (17 rounds, deg s+2), synthetic Spartan outer
  `eq·(A·B − C)` (14 rounds, deg 3) and inner `M·Z` (14 rounds, deg 2), RLC with one β per member,
  Spartan members head-aligned (their polynomials carry the `2^(17−14)` padding scale; they bind the
  first 14 challenges = the low row bits, which is what lets W pad at the high end); round messages
  compressed to `deg` coefficients (the constant is implied by the running claim). Verified by a
  replaying verifier: stream → per-member expected finals → hash final via the column sumcheck.
- Production change: `HyperKZGScheme::setup_from_secret` computes the SRS powers with parallel
  scalar multiplications (2^21: 16 s, 2^23: 103 s; was serial). No API change.
- The bit-column commit uses lane M6's `g1_bit_columns_msm` (M3's bench still used the old kernel:
  1.10 s → 0.08 s at 2^17 × 163).

## 2. Measurements

### 2.1 Hash table, word layout (M3 shape: 2^17 rows, 163 committed bits, 67 wired, 229 constraints, degree 3)

| k | claims | commit | row sumcheck | reduction + combine | open | **prover** | RSS | commitments | rounds | claims | opening | **bytes** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | (a) 230 claims | 0.081 | 0.256 | 0.092 | 0.163 | **0.59** | 0.62 GB | 5,216 | 1,632 | 7,360 | 2,176 | 16,384 |
| 1 | (b) column sumcheck | 0.079 | 0.276 | 0.092 | 0.161 | 0.61 | 0.62 GB | 5,216 | 1,632 | 832 | 2,176 | 9,856 |
| 8 | (a) | 0.231 | 0.309 | 0.101 | 1.331 | 1.97 | 1.52 GB | 672 | 1,632 | 7,360 | 2,560 | 12,224 |
| 8 | (b) | 0.231 | 0.310 | 0.100 | 1.367 | **2.01** | 1.50 GB | 672 | 1,632 | 832 | 2,560 | **5,696** |
| 16 | (a) | 0.220 | 0.295 | 0.098 | 2.578 | 3.19 | 2.29 GB | 352 | 1,632 | 7,360 | 2,688 | 12,032 |
| 16 | (b) | 0.221 | 0.296 | 0.101 | 2.579 | 3.20 | 2.30 GB | 352 | 1,632 | 832 | 2,688 | 5,504 |

Rounds are compressed (17 × 3 × 32; M3 sent 4 coefficients = 2,176). M3's 11,733 B excluded the 163
commitments (5,216 B); like for like, S1 at k = 1 is 16,384 B (a). The M6 kernel makes the commit
14× cheaper than M3's measured 1.10 s; the row sumcheck is 0.26–0.31 s (M3: 0.35 s contended).

### 2.2 Hash table, bit-sliced reshape (column sumcheck on; k = 1 unless noted)

| layout | rows | committed / wired | commit | row sumcheck | reduction | open | **prover** | RSS | commitments | rounds | claims (b) / (a) | opening | **bytes (b)** |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bits:1 | 2^21 | 16 / 13 | 0.382 | 0.564 | 0.158 | 2.600 | 3.70 | 3.07 GB | 512 | 2,016 | 544 / 928 | 2,688 | 5,760 |
| bits:2 | 2^20 | 32 / 20 | 0.335 | 0.477 | 0.156 | 1.309 | 2.28 | 1.98 GB | 1,024 | 1,920 | 640 / 1,664 | 2,560 | 6,144 |
| bits:4 | 2^19 | 64 / 34 | 0.203 | 0.485 | 0.175 | 0.709 | 1.57 | 1.16 GB | 2,048 | 1,824 | 736 / 3,136 | 2,432 | 7,040 |
| bits:8 | 2^18 | 128 / 62 | 0.157 | 0.480 | 0.177 | 0.376 | 1.19 | 1.04 GB | 4,096 | 1,728 | 832 / 6,080 | 2,304 | 8,960 |
| bits:4, k=4 | 2^19 → 2^21 | 64 / 34 | 0.404 | 0.483 | 0.156 | 2.603 | 3.65 | 2.81 GB | 512 | 1,824 | 736 | 2,688 | 5,760 |
| bits:2, k=8 | 2^20 → 2^23 | 32 / 20 | 0.797 | 0.765 | 0.139 | 10.39 | 12.1 | 5.11 GB | 128 | 1,920 | 640 | 2,944 | 5,632 |

The reshape's sumcheck costs 0.48–0.56 s at every b (24·b constraints × 2^21/b rows ≈ 50 M
constraint evaluations vs 30 M for the word layout), its commit 0.16–0.40 s (fewer columns share each
base in the M6 kernel), and its opening follows the row count. Word k=8 (b) — 5,696 B at 2.0 s —
beats every reshape on both axes: same opening size class as bits:2, 3 fewer rounds than bits:1, and
packing + column sumcheck erase the column-count penalty (672 + 832 B vs 512 + 544 B). bits:1 k=8 was
not run (opening 2^24 ≈ 21 s by the law; ≈ −60 B).

### 2.3 Limb table (M2 shape: 2^17 rows, t = 12, 54 u16 chunk columns + ⌈54/s⌉ helpers + multiplicity)

| s | k | commit chunks | commit helpers | sumcheck | RLC + open | **prover** | RSS | commitments | rounds | column claims | operand claims | opening | **bytes** |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 1 | 0.374 | 0.694 | 1.017 | 0.234 | **2.32** | 1.65 GB | 2,048 | 4,352 | 2,048 | 2,304 | 2,176 | 12,928 |
| 6 | 8 | 0.413 | 0.762 | 1.087 | 1.403 | 3.67 | 1.93 GB | 288 | 4,352 | 2,048 | 2,304 | 2,560 | 11,552 |
| 6 | 16 | 0.447 | 0.824 | 1.219 | 3.058 | 5.55 | 2.22 GB | 160 | 4,352 | 2,048 | 2,304 | 2,688 | 11,552 |
| 3 | 8 | 0.485 | 1.646 | 0.966 | 1.552 | 4.65 | 2.31 GB | 320 | 2,720 | 2,336 | 2,304 | 2,560 | 10,240 |
| 1 | 8 | 0.410 | 4.260 | 0.861 | 1.402 | 6.94 | 2.74 GB | 448 | 1,632 | 3,488 | 2,304 | 2,560 | 10,432 |

Chunk columns and helper columns are packed in separate groups (α must be squeezed after the chunk
commitments, before the helpers exist): ⌈54/k⌉ + ⌈(⌈54/s⌉+1)/k⌉ commitments. Packing leaves the
limb commit and sumcheck untouched (same MSMs over shifted SRS windows) and only grows the opening.
M2's s = 6 result (2,115 ms) reproduces at 2.32 s (load 3–4).

### 2.4 One batched stream (hash word 2^17 + limb 2^17 t=12 + Spartan outer/inner 2^14, one opening)

| s | k | hash claims | commits (hash / chunks / helpers / W) | build | stream (hash / limb / Spartan) | reduction + combine | open | **prover** | RSS | commitments | rounds (separate) | hash cl. | limb cl. | Spartan | opening | **bytes** |
|---|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 6 | 1 | (a) | 1.42 (0.11 / 0.45 / 0.84 / 0.02) | 0.25 | 1.32 (0.30 / 1.01 / 0.01) | 0.17 | 0.22 | **3.37** | 1.97 GB | 7,296 | 4,352 (8,224) | 7,360 | 4,352 | 160 | 2,176 | 25,696 |
| 6 | 8 | (b) | 1.48–1.61 (0.23–0.27 / 0.42–0.46 / 0.81–0.86 / 0.02) | 0.23 | 1.16–1.28 (0.26–0.28 / 0.90–0.99 / 0.01) | 0.15 | 1.34–1.63 | 4.35–4.92 | 2.29 GB | 992 | 4,352 (8,224) | 832 | 4,352 | 160 | 2,560 | 13,248 |
| 3 | 8 | (b) | 2.11 (0.23 / 0.41 / 1.46 / 0.02) | 0.24 | 0.96 (0.25 / 0.70 / 0.01) | 0.15 | 1.35–1.45 | **4.81–4.91** | 2.70 GB | 1,024 | 2,720 (6,592) | 832 | 4,640 | 160 | 2,560 | **11,936** |
| 3 | 16 | (b) | 2.52 (0.34 / 0.49 / 1.67 / 0.02) | 0.26 | 1.20 (0.34 / 0.85 / 0.01) | 0.20 | 3.32 | 7.50 | 2.97 GB | 576 | 2,720 (6,592) | 832 | 4,640 | 160 | 2,688 | 11,616 |
| 3 | 1 | (b) | 1.98 (0.10 / 0.41 / 1.46 / 0.02) | 0.23 | 0.95 (0.25 / 0.69 / 0.01) | 0.17 | 0.19 | **3.52** | 2.35 GB | 7,584 | 2,720 (6,592) | 832 | 4,640 | 160 | 2,176 | 18,112 |
| 1 | 8 | (b) | 5.00 (0.23 / 0.41 / 4.35 / 0.02) | 0.28 | 0.90 (0.25 / 0.64 / 0.01) | 0.17 | 1.35 | 7.70 | 3.09 GB | 1,152 | 1,632 (5,504) | 832 | 5,792 | 160 | 2,560 | 12,128 |

Three runs of (s=3, k=8): 4.81 / 4.89 / 4.91 s. Batching saves 8,224 − 4,352 = 3,872 B at s = 6
and 6,592 − 2,720 = 3,872 B at s = 3 (the three shorter/lower-degree members hide under the limb
member's 17 × (s+2) envelope). s = 3 vs s = 6: −1,632 B of rounds, +288 B of helper claims/commit
(+0.3 s helper commits, −0.3 s sumcheck — a wash in time). s = 1 makes the stream degree 3 (1,632 B)
but adds 45 helper columns (+1,440 B claims, +2.9 s commits): worse on both axes unless the limb
claims are column-batched too (§4).

## 3. Derivation notes

### 3.1 Rows A–E of §0
A: components of the s=6 k=1 run with the three streams counted separately (hash 17×3, limb 17×8,
Spartan 14×3 + 14×2, all compressed); the batched run does the same work, so its time is the time.
B/C: hash k=8/16 (a) from §2.1 + limb k=8/16 s=6 from §2.3 + Spartan (W 32, claims 160, rounds
2,240) + one opening at 2^20 / 2^21; commits = hash + limb + W; sumchecks = hash + limb + 0.01.
D: bits:1 col (§2.2) + limb k=1 s=6 + Spartan + opening 2^21 (2,688 B, 2.60 s). E: bits:2 k=8 (§2.2)
+ limb k=8 s=6, opening 2^23 (2,944 B, 10.4 s). Opening law confirmed: 2^17 0.16–0.23 s, 2^20
1.31–1.63, 2^21 2.58–3.3, 2^23 10.4.

### 3.2 Wiring layer (not built; same for every column-batched row)
S2 (hash wiring, 17 rounds deg 2, compressed) 1,088 B + its end claims column-batched
(8 × 2 × 32 + 32 = 544) · limb wiring (17 rounds deg 2) 1,088 B + z-chunk source claims batched
(5 × 2 × 32 + 32 = 352) · S3 two-point → one-point reduction over the 25 (row, column) variables,
deg 2: 1,600 + 32. Sum ≈ 4.7 KB; as one second stream (S2 + limb wiring, max deg 3 for the chaining
terms) + S3: ≈ 3.4 KB. With per-column claims (M3 §B / M2 §4 style): 6,752 + 6,848 + 2,688 ≈ 16.3 KB.
Prover ≈ 0.3–0.5 s (S2 ≈ 30 M mults, limb wiring ≈ 19 M, S3 over 163 × 2^17 entries). The verifier's
native O(rows) matrix-MLE evaluations (≈ 20 M mults) are what a recursion layer would have to prove
via SPARK-style sparse evaluations — the dominant open item for on-chain cost, not for bytes.

### 3.3 What a recursion layer hashes
All proof bytes are absorbed (commitments, compressed round coefficients, claims, HyperKZG
com/w/v) plus IO. G: 11,936 + 1,024 = 12,960 B; with the modeled wiring layer ≈ 16.4 KB.

## 4. Findings and next levers

1. **Packing + column sumcheck is the lever, reshape is not.** k=8 with the column sumcheck takes the
   hash table from 16.4 KB to 5.7 KB for +1.4 s (all of it the 2^20 opening); bits:1 @2^21 reaches
   5.8 KB for +3.1 s. k=16 saves a further 190–320 B for +1.3–2.6 s: not worth it.
2. **One stream** saves 3.9 KB of round polynomials for free; **s=3** saves another 1.3 KB net at
   equal prover time. G = 12.0 KB core + 1 KB IO at 4.8–4.9 s (2.5 s budget: +2.3 s, all in the
   2^20 opening and the s=3 helper commits).
3. **Remaining bytes in G:** operand-limb claims 2,304 (∝ 6t; inherent to the wiring lane's
   products), limb column claims 2,336, opening 2,560, rounds 2,720, commitments 1,024, IO 1,024,
   hash 832, Spartan 160. Next levers, ranked by bytes per prover-second:
   - **Opening format:** HyperKZG sends `ℓ−1` fold commitments + 3ℓ evaluations (2,560 B at ℓ = 20).
     A Zeromorph/Shplonk-style multilinear-to-univariate opening sends ℓ commitments + one witness
     (≈ 700 B) at the same prover cost class → −1.9 KB. Not built; no change to the committed
     polynomials.
   - **Limb column sumcheck:** needs the limb relation in aligned-quadratic form, which only s = 1
     gives (`h_j·(α − c_j)` aligned, chunks linear once the operand claims are sent): 2,336 → ≈ 800 B
     and the stream degree drops to 3 (2,720 → 1,632), i.e. ≈ 9.4 KB core, but 54 full-width helper
     columns cost +2.9 s of commits (7.7 s total). A GPU/Metal MSM for the helpers is the only way to
     make s = 1 cheap.
   - **k=8 at 2^16 rows:** M1's realistic Dory row count (≈122 k) does not fit 2^16; the hash table
     (118–122 k rows) does not either. No row-count lever below 2^17 without splitting tables.
   - Bytes floor of this architecture (one opening, one stream, all claims batched): commitments
     ≈ 1 KB + rounds 1.6–2.7 KB + opening 0.7–2.6 KB + operand claims 2.3 KB + IO 1 KB ≈ 7–10 KB
     before a recursion layer; "a couple of KB" needs the layer-2 wrap.
4. Soundness bookkeeping for the two new arguments (kept in the bench verifiers): packed opening —
   `P_g(s_lo, r) = Σ_j eq(s_lo, j) c_{g·k+j}(r)` is a multilinear identity, the group weights are
   `eq(s_hi, ·)` (hash) / `ρ^g` (others) drawn after the claims; column sumcheck — the final claim
   `Q(v, w)` is a public quadratic form once γ and the row point are fixed, `v(s)` is bound by the
   opening and `w(s)` becomes S2's input claim; the wired words live at indices where `v = 0`.

## 5. Reproduce

```
export CARGO_TARGET_DIR=/Volumes/Dev/cargo-target/wrap-spartan-hyperkzg
cargo build --release -p jolt-wrapper-bench -p jolt-limb-bench
B=$CARGO_TARGET_DIR/release
$B/jolt-wrapper-bench 17                       # hash word, k=1, all claims
$B/jolt-wrapper-bench k=8 col 17               # packing + column sumcheck
$B/jolt-wrapper-bench layout=bits:1 col 21     # reshape (also bits:2 20, bits:4 19, bits:8 18)
$B/limb-relation 17 12 s=6 k=8                 # limb table, packed
$B/jolt-wrapper-bench batched s=3 k=8 col 17   # one stream + one opening (row G)
```
Peak RSS via `/usr/bin/time -l`. Logs of every run: /tmp/n3/*.log (all.log, rerun.log).
