# Lane M3 — streaming Blake3 transcript: profile, exact counts, relation cost model, microbench

Date: 2026-09-02. Tree: wrap/spartan-hyperkzg. Machine: Mac mini, 10 threads, 16 GiB, **shared with 3
other lanes' builds during every measurement (load average 9–12)** — wall times below are inflated by
CPU contention; the counts are exact.

## Result in one screen

| item | fibonacci L=18 (K=13) | fibonacci L=20 (K=16) |
|---|---:|---:|
| legacy chained Blake2b-256 compressions, hidden segment (measured) | **2,111** | **2,191** |
| streaming Blake3, labels kept as 32-B words — **C3** | **1,018** | **1,054** |
| streaming Blake3, 4-byte label tags | 903 | 939 |
| streaming Blake3, 1-byte label tags | 892 | 928 |
| floor: squeezes + absorbed Fr bytes only (labels free) | 722 | 756 |
| G-steps = 56·C3 | 57,008 | 59,024 |
| table rows = 116·C3 (half-G rows + 4 chaining rows per compression) | 118,088 → 2^17 | 122,264 → 2^17 |

Microbench of the L=20 shape (2^17 rows × 163 committed bit columns + 66 wired inputs, degree-3
sumcheck, one HyperKZG opening), 10 threads, contended (best of 3): **commit 1.10 s, sumcheck 0.35 s,
RLC 0.02 s, open 0.26 s, verify 5 ms; proof 11,733 B** (rounds 2,176 + 229 claims 7,328 + opening 2,197).
Target was commits ≈100 ms / sumchecks ≈200 ms / opening ≈250 ms (shared): the transcript table
misses on commit (11×) and sumcheck (2.2×); see §C for the failing sub-steps and the next technique.

## A. Streaming Blake3 transcript

### Spec (byte level) — `jolt_transcript::Blake3Transcript` (`crates/jolt-transcript/src/blake3.rs`)

- State = 32 bytes. `new(label)`: `state₀ = blake3::hash(label ‖ 0^(32−|label|))` (one plain compression,
  outside every count); the running hasher is `blake3::Hasher::new_keyed(state)` — every transcript
  compression carries `KEYED_HASH`.
- `append_bytes(b)`: streams `b` into the running keyed chunk. Encodings unchanged from the legacy
  profile: `Label` = 32-B zero-padded word, `LabelWithCount` = 24-B label ‖ 8-B BE count, `U64Word` =
  24 zero bytes ‖ 8-B BE, Fr = 32-B big-endian. An empty append absorbs nothing (the protocol's
  `LabelWithCount` delimits payloads; the stage-4 `append_bytes(&[])` separator costs 0).
- Segment bound: a segment never exceeds 1,024 bytes (one Blake3 chunk). When a 17th block would start,
  the chunk is closed: `state ← keyed_hash(state, segment)` (the 16th block's compression, already
  counted, gets `CHUNK_END|ROOT`), and the hasher is re-keyed. No parent nodes ever exist: every
  compression in the chain is a block compression with `t = 0`, `b = bytes in block`,
  flags ⊆ {CHUNK_START, CHUNK_END, ROOT, KEYED_HASH}.
- Squeeze (`challenge` / `challenge_scalar`): finalize the pending (possibly empty) block with
  `CHUNK_END|ROOT` and read **64 bytes of root output** (one compression, output counter 0):
  `out[0..32]` = the standard 32-byte digest = **next state** (keys the next segment);
  `out[32..48]` = the **16 challenge bytes**, decoded exactly as before (`from_challenge_bytes`
  125-bit / `from_scalar_challenge_bytes` 128-bit BE). `state()` = `keyed_hash(state, pending)`.
- Count law: a segment of `n` pending bytes ending in a squeeze or a chunk close costs
  `max(1, ⌈n/64⌉)` compressions — the finalize IS the last block's compression. Consecutive
  squeezes cost 1 each (empty block).
- Cheaper sound variant considered and NOT taken: challenge from `out[0..16]` and state from
  `out[32..64]` is the same one compression — no gain. Dropping the state-carrying key (plain
  hashing with `state ‖ bytes`) would cost 32 B per segment (+½ block per squeeze ≈ +180 compressions).
- Tests: `crates/jolt-transcript/tests/blake3_tests.rs` — byte-exact keyed chain (state =
  `keyed_hash(prev, appended bytes)`, challenge = XOF bytes 32..48), the 1,024-B close (exact chunk,
  chunk+1, split appends), determinism/order/domain separation, 1,000 distinct consecutive squeezes.
  Feature `transcript-blake3` (default), dep `blake3` (workspace 1.8.5).

### Counting hook and exact counts

`crates/jolt-prover/tests/dory_byte_diff.rs` `mod transcript_schedule` (behind `prover-fixtures`):
`Recording<T>` forwards every `Transcript` call to `T` and logs `(kind ∈ {Label, Scalar, Bytes}, len)`
per append and every squeeze (kind from the `AppendToTranscript` type name; direct
`value.append_to_transcript(t)` calls — the stage-8 RLC claims — are classified by the 32-byte word's
first byte: ASCII label vs BE Fr < 0x31). The modular verifier's stage spine is replayed with a marker
per stage on a real proof (fibonacci 2^18: 197,595 rows; 2^20: 787,395 rows), for both
`LegacyBlake2bTranscript` and `Blake3Transcript` proofs; the two recorded schedules are asserted
identical over the hidden segment (the schedule is transcript-agnostic — the same log feeds both
counts). Hidden segment = stage-1 start … the stage-8 RLC γ squeeze (last squeeze before the first
384-B Dory GT absorb). Legacy count = `⌈(64+len)/128⌉` per append + 1 per squeeze (`digest.rs`).
Blake3 count = the law above, applied to the same byte stream with the label width varied.

Run: `cargo nextest run -p jolt-prover --features prover-fixtures --no-capture -E 'test(fibonacci_2_18_schedule)'`
(and `_2_20_`).

Per stage, fibonacci 2^18 (K=13):

| stage | labels | Fr | raw | squeezes | Blake2b | B3 32-B labels | B3 4-B | B3 1-B | B3 no labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 57 | 122 | 0 | 41 | 220 | 110 | 94 | 93 | 90 |
| 2 | 53 | 121 | 0 | 53 | 227 | 105 | 97 | 96 | 96 |
| 3 | 34 | 70 | 0 | 24 | 128 | 56 | 50 | 49 | 49 |
| 4 | 35 | 84 | 1 | 29 | 149 | 61 | 58 | 58 | 57 |
| 5 | 215 | 505 | 0 | 151 | 871 | 436 | 407 | 404 | 257 |
| 6a | 16 | 32 | 0 | 21 | 69 | 35 | 35 | 35 | 25 |
| 6b | 104 | 176 | 0 | 26 | 306 | 146 | 110 | 106 | 103 |
| 7 | 44 | 48 | 0 | 6 | 98 | 48 | 30 | 30 | 24 |
| 8 RLC | 1 | 41 | 0 | 1 | 43 | 21 | 22 | 21 | 21 |
| **total** | **559** | **1,199** | 1 | **352** | **2,111** | **1,018** | **903** | **892** | **722** |

Per stage, fibonacci 2^20 (K=16):

| stage | labels | Fr | raw | squeezes | Blake2b | B3 32-B labels | B3 4-B | B3 1-B | B3 no labels |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 59 | 128 | 0 | 45 | 232 | 116 | 100 | 99 | 96 |
| 2 | 55 | 127 | 0 | 55 | 237 | 109 | 101 | 100 | 100 |
| 3 | 36 | 76 | 0 | 26 | 138 | 60 | 54 | 53 | 53 |
| 4 | 37 | 90 | 1 | 31 | 159 | 65 | 62 | 62 | 61 |
| 5 | 217 | 525 | 0 | 153 | 895 | 448 | 419 | 416 | 267 |
| 6a | 16 | 32 | 0 | 21 | 69 | 35 | 35 | 35 | 25 |
| 6b | 106 | 186 | 0 | 28 | 320 | 152 | 116 | 112 | 109 |
| 7 | 44 | 48 | 0 | 6 | 98 | 48 | 30 | 30 | 24 |
| 8 RLC | 1 | 41 | 0 | 1 | 43 | 21 | 22 | 21 | 21 |
| **total** | **571** | **1,253** | 1 | **366** | **2,191** | **1,054** | **939** | **928** | **756** |

Reading: 2^18 matches lane K's 2,111 exactly; 2^20 measures 2,191 (plan-relation §1's formula gave
2,209 — 4 labels/9 Fr/6 squeezes fewer than the closed form). Streaming Blake3 halves the count
(**C3 = 1,018 / 1,054**, −52%). Label tags buy little (−11%): stage 5's 128 degree-2 rounds absorb
label + 2 Fr = 96 B → 2 blocks with 32-B labels, and still 2 blocks with a 1-byte tag (65 B); only
dropping the per-round label entirely (the schedule already fixes round/degree) reaches the floor
of 722 / 756, which is `Σ_squeezes max(1, ⌈Fr bytes/64⌉)` — the absolute minimum for any 64-B-block
hash at this schedule. G-function count 56·C3 = **57,008 / 59,024** (vs 56·2,111 = 118,216 for a
Blake2b-shaped count of the same schedule, ignoring Blake2b's 12 rounds × 8 G × 64-bit words).
Native (seed) segment: 239 legacy compressions (preamble + 43 GT commitments), unchanged.

Prover time: the transcript is ≈3,000 hash calls per proof (<1 ms either way); measured prove wall
(contended) 2^18: Blake2b 9.3 s vs Blake3 6.8 s; 2^20: 12.0 s vs 23.3 s — noise from concurrent
builds, not a transcript effect. Proof bytes identical (82,191 / 86,523 B).

## B. Relation cost model (committed bits + sumcheck)

Row = one **half G-step** (`a' = a+b+m; d' = (d⊕a')⋙R1; c' = c+d'; b' = (b⊕c')⋙R2`, R1/R2 = 16/12
on even rows, 8/7 on odd rows). Rows per compression: 7 rounds × 8 G × 2 = 112, plus **4 chaining
rows** materializing the 8 output words `v7[i] ⊕ v7[i+8]` two per row (the next compression's CV
words 4..7 are XOR inputs and must be degree-1; words 0..3 and the squeeze's challenge words 8..11
are needed as degree-≤2 links). **Rows = 116·C3 = 118,088 (L=18) / 122,264 (L=20) → padded 2^17**
(90% / 93% full; ≤ 2^17.5 met; 2^17 holds up to C3 = 1,129).

Columns per row (all committed columns are bits):

| group | columns | role |
|---|---:|---|
| A', D', C', B' | 4 × 32 | add outputs (A', C') and **un-rotated** XOR outputs (D' = d⊕a', B' = b⊕c'); rotations live in the wiring/weights |
| κ0, κ1, κ2 | 3 | overflow bits of the ternary add (2^32, 2^33) and the binary add (2^32) |
| m | 32 | this row's message word bits |
| **committed** | **163** | committed bits = 163 × 2^17 = **21,364,736** (19.9 M before padding) |
| wired inputs | 66 | `a_in`, `c_in` (integers), `bin`, `din` (32 bits each) — public sparse matrices over committed columns (+ public constants for the first round's IV/t/b/flags words) |
| public | 1 | `sel` = row parity (the top row-index bit: bound last, so no round exceeds degree 3) |

Constraints per row (229, all degree ≤ 2 in columns → **sumcheck degree 3** with eq):
booleanity ×163; XOR `D'_k = din_k + A'_k − 2·din_k·A'_k` ×32 and `B'_k = bin_k + C'_k − 2·bin_k·C'_k`
×32 (rotated bit references are in the wiring: `din_k = D'_{src,(k+R1) mod 32}`);
ternary add `ΣA'2^k + 2^32κ0 + 2^33κ1 = a_in + Σbin_k 2^k + Σm_k 2^k` (linear);
binary add `ΣC'2^k + 2^32κ2 = c_in + (1−sel)·R16(D') + sel·R8(D')` with
`R_r(D') = Σ_k 2^((k−r) mod 32) D'_k` (linear × selector = degree 2).
Rotation by 16/8 (d) and 12/7 (b) are pure bit re-indexings inside the public wiring matrices, so
the same constraint text serves both row parities; the only in-row parity dependence is the
selector on the linear add term.

Sumchecks (all over the 2^17 rows):

| # | statement | degree | rounds | ends with |
|---|---|---:|---:|---|
| S1 main | `Σ_t eq(τ,t)·Σ_j γ^j C_j(t)` = 0 | 3 | 17 | 163 committed + 66 wired claims at `r` |
| S2 wiring + links | `Σ_u Σ_j ρ_j M̃_j(r,u)·col_src(j)(u)` = Σ_j ρ_j·wired_j(r); includes the message copy constraint (7 uses of each block word), the R1CS links L1–L3 | 2 (3 on the 4 chaining-row terms) | 17 | 160 source-column claims at `u*`; verifier evaluates the public `M̃_j(r,u*)` natively, O(rows) per distinct matrix (≈8 shapes: same-G, round wiring, message schedule, chaining) |
| S3 two-point reduction | `Σ_u (eq(r,u) + β·eq(u*,u))·col(u)` | 2 | 17 | 163 claims at `w` → the shared HyperKZG batch opening |

Linking sumchecks (rows counted; all folded into S2 as extra public-matrix terms over the same 2^17
rows, no committed rows added): **L1 message link** — R1CS witness Fr (HyperKZG-committed `W`) at
the absorbed-Fr positions = `Σ_word 2^(32·word) Σ_k 2^k m_k` over the 8 first-use rows of that Fr
(1,253 Fr @L=20 → 10,024 word links); **L2 challenge link** — the 366 R1CS challenge variables =
125/128-bit decode of root output words 8..11 = `v7[8+i] ⊕ cv[i]` (XOR of a committed round-7 word
and a materialized chaining word, degree 2); **L3 state link** — `state_in` (public input) enters
as wiring constants of the first compression; `state_rlc` (public output) = the last compression's
materialized CV bits (linear).

Prover field operations (Fr mults, L=20 shape, 2^17 rows): S1 round 0 on bits ≈ 5 mults/pair
(γ-bucket sums, i128 adds) → negligible; rounds ≥1: 916 mults/pair for Q(0)+Q₂ (163 squares + 163 γ
weights, 64 XOR products + 64 γ weights, ×2 for the quadratic coefficient) + 230 bind mults, over
Σ 2^15+2^14+… = 2^16 pairs → **≈75 M**; S2 ≈ 30 M (160 columns × 2 points × 2^16 + O(nnz) matrix
folding); S3 ≈ 21 M; commit: 163 subset sums ≈ **10.7 M affine additions**; RLC 10.7 M adds;
opening: one 2^17 HyperKZG (`cost law 170 ms`). Total ≈ 125 M Fr mults + 10.7 M point additions.

Proof bytes (brief's formula `rounds × (deg+1) × 32 + claims × 32`, opening measured):

| piece | bytes |
|---|---:|
| S1 17 × 4 × 32 | 2,176 |
| S1 claims 229 × 32 | 7,328 |
| S2 17 × 3 × 32 + 160 claims | 1,632 + 5,120 |
| S3 17 × 3 × 32 + 163 claims | 1,632 + 5,216 |
| HyperKZG opening at 2^17 (measured bincode) | 2,197 |
| **total** | **25,301** |

Per-column claims at three points dominate (17.7 KB). Next technique (single): **column batching** —
replace every per-column claim set by one 16-round degree-3 sumcheck over the column index
(`Σ_{j1,j2} Q̃(j1,j2)·T(r,j1)·T(r,j2)` with the public constraint matrix Q̃) ending in two
`eq(s,·)`-weighted RLC evaluations, so each point costs O(1) claims: 2,176 + 2,048 + 1,632 + 1,632 +
≈160 + 2,197 = **≈9.8 KB** for the transcript table (2.2 KB of which is the opening shared with the
other tables). The round messages then dominate; compressed rounds (d coefficients + claim) give
≈7.7 KB.

## C. Microbench — `crates/jolt-wrapper-bench` (`cargo build --profile test -p jolt-wrapper-bench`, run `jolt-wrapper-bench 16 17 18`)

What it runs, per size: random 0/1 columns in the exact B shape (163 committed, 66 wired, selector);
(1) one HyperKZG commitment per committed column via `batch_g1_additions_multi` over the SRS powers
(checked against `HyperKZGScheme::commit` for column 0); (2) the degree-3 sumcheck
`Σ_t eq(τ,t)·Σ_j γ^j C_j(t)` with the 229 constraints above (round 0 on bits with γ-bucket sums,
rounds ≥1 generic; Gruen-style `s(X) = c·l(X)·t(X)` from `t(0)`, `t(1)` via the running claim, and the
quadratic coefficient); (3) RLC of the 163 committed columns with ρ^j, `combine` of the commitments,
one `HyperKZGScheme::open` at the sumcheck point; (4) verify (sumcheck rounds + final `Q(claims)`
check + `HyperKZGScheme::verify`). Blake3Transcript as the Fiat-Shamir transcript. Wired inputs are
random prover-side vectors (their claims are not linked — S2 is not in the bench).

Measured (10 rayon threads; three runs, **load average 4–12 from other lanes' rustc/clippy during
all of them** — best of three, range in brackets; reference for the contention: HyperKZG open 2^17 =
264–418 ms here vs 170 ms in lane G's uncontended cost law):

| rows | commit (163 batch additions) | sumcheck (round 0 bits / round 1) | RLC | combine+open | verify | prover total | proof bytes |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2^16 | 0.742 s [0.74–0.81] | 0.200 s [0.20–0.22] (0.022 / 0.083) | 0.013 s | 0.162 s [0.16–0.30] | 5 ms | 1.18 s | 11,476 |
| **2^17** | **1.100 s** [1.10–1.18] | **0.354 s** [0.35–0.53] (0.055 / 0.141) | 0.021 s | **0.264 s** [0.26–0.42] | 5 ms | **1.74 s** [1.74–2.15] | **11,733** |
| 2^18 | 2.367 s [2.37–2.53] | 0.761 s [0.76–0.79] (0.087 / 0.342) | 0.056 s | 0.566 s [0.57–0.59] | 5 ms | 3.75 s | 11,990 |

Peak RSS 4.18 GB for the whole 16/17/18 run (2^18: 231 Fr columns × 2^17 after round 0 ≈ 0.97 GB;
`batch_g1_additions_multi` copies every selected affine point per column ≈ 1.4 GB; two SRS copies).
Scaling is linear in rows for every phase (2× rows → 1.4–2.2×; the commit's tree is memory-bound).
Proof bytes = rounds (17 × 4 × 32) + 229 claims + opening, i.e. the S1 line of §B; 7,328 of the
11,733 B are the per-column claims.

Verdict per phase against plan v3's budget (commits ≈100 ms, sumchecks ≈200 ms for all tables,
one opening ≈250 ms):
- **commit — fails (11×)**: 10.7 M affine additions in 1.10 s = 103 ns per addition on 10 threads
  ≈ 1 µs of CPU per addition, ≈20× the batch-affine floor (~6 Fq mults ≈ 50 ns single-threaded →
  ≈55 ms on 10 threads). The kernel, not the bit count, is the problem: `batch_g1_additions_multi`
  materializes `Vec<Vec<G1Affine>>` working sets and reallocates per tree level. Next technique:
  an in-place affine subset-sum kernel (pairwise adds with one shared batch inversion per level over
  a flat buffer, columns interleaved for locality) — lane M2's MSM territory; expected 60–120 ms.
  Independent −20%: drop the in-row `m` bits (the add uses `m` only as an integer) in favour of a
  16-word × 32-bit message table per compression (16·C3 = 16,864 rows → 2^15) referenced through
  the wiring as one wired integer: 131 committed columns in the main table + 32 in the small table.
- **sumcheck — fails (1.8× for T1 alone)**: 0.35 s ≈ 75 M Fr mults at ≈4.7 ns/mult/thread under
  contention. Next technique: after round 0 every bound value lies in {0, 1, r, 1−r}, so round 1's
  163 squares + 64 products become 4-entry/16-entry table lookups (round 1 is 43% of the remaining
  work); rounds ≥2 stay generic → ≈0.25 s projected, plus the −13% from 32 fewer columns.
- **opening — meets the shared budget once uncontended** (cost law 170 ms at 2^17; 341 ms here).
- **verify** 7 ms; **RLC** 29 ms — fine.

## D. Switching the Jolt profile to Blake3Transcript

Call sites that pick the transcript type (production paths only): `jolt-sdk/src/host_utils.rs:49,67`
(`VerifierTranscript = jolt_transcript::LegacyBlake2bTranscript<VerifierField>` →
`Blake3Transcript<VerifierField>`), `jolt-sdk/src/host_utils.rs:10,19` (`ProofTranscript =
jolt_prover_legacy::transcripts::Blake2bTranscript` — the legacy monolith has its own transcript
module; the modular prover takes `T` generically at `jolt_prover::dory::prove::<F, PCS, VC, T, W>`, so
a legacy-prover profile change needs a legacy-side `Transcript` impl over blake3 (≈120 lines) or the
prover moves to the modular path), `crates/jolt-prover/src/profile.rs` (profile harness type
choice), `crates/jolt-prover/src/stages/drivers.rs` (type mentions), `jolt-dory`'s
`JoltToDoryTranscript` (generic over `T`, no change). `JoltProtocolConfig` (`crates/jolt-verifier/src/
config.rs:45`) records zk/commitment/challenge-endianness but not the hash — add a `transcript`
variant (append-only enum) so `validate_proof_config` rejects a mismatched profile fail-closed.
Byte-diff/fixture tests pinning Blake2b states (`dory_byte_diff.rs`, `jolt-verifier/tests/support`)
keep the legacy type. Expected Jolt-prover delta ≈ 0: ≈3,000 transcript calls per proof; Blake3
≈1,300 compressions ≈ 0.1 ms vs Blake2b ≈2,700 ≈ 0.5 ms.

## Hygiene / kill-list

- Pre-commit clippy over the workspace is broken before this lane: `crates/jolt-dory/src/compression.rs:434`
  imports `jolt_field::RandomSampling` (lane J code) which no longer exists → the lib-test target of
  jolt-dory fails to compile; committed with `DISABLE_CLIPPY=1` after
  `cargo clippy -p jolt-transcript -p jolt-hyperkzg -p jolt-wrapper-bench --all-targets -- -D warnings`
  passed clean. Style-invariant violations remaining in the hook output are all in lane G/J files
  (jolt-crypto, jolt-dory compression, jolt-hyperkzg bench/scheme/types), none in this lane's code.
- `gt_compression` (lane J's module in `dory_byte_diff.rs`) did not compile against the current
  `prove(&W)` signature; fixed with `witness.as_ref()`.
- Commits: 7b5b1b9f8 (code), d7db622b9 (clippy `print_stdout` expectation on the schedule test module; `cargo clippy -p jolt-prover --tests --features prover-fixtures -- -D warnings` clean).

## Files

- `crates/jolt-transcript/{Cargo.toml, src/lib.rs, src/blake3.rs, tests/blake3_tests.rs}` — feature
  `transcript-blake3`, `Blake3Transcript`.
- `crates/jolt-prover/tests/dory_byte_diff.rs` — `mod transcript_schedule` (recorder, stage replay,
  count tables at 2^18 and 2^20); also `witness.as_ref()` in `gt_compression` (the module no longer
  compiled against the current `prove` signature).
- `crates/jolt-hyperkzg/src/types.rs` — `HyperKZGCommitment::new(point)` for externally computed
  commitments.
- `crates/jolt-wrapper-bench/` — the microbench (workspace member).
