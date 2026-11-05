# Whiteout overhead over clear Akita, after T0 (P1)

Parent: `specs/whiteout-optimization-campaign.md`. Method:
`specs/whiteout-o0-profile.md` and `specs/whiteout-p0b-idle-rerun.md`,
with prover spans added. This ticket measures; it changes no code.

**Code.** jolt `wo/t0` `81bf8dbe2`, whose `[pin]` commit builds against
Akita `~/akita-wo-t0` `deb13c039`. Release builds of `jolt-prover --features
profiling,akita` (clear) and `profiling,akita,zk` (Whiteout), each run as
`jolt-prover profile --name <w> --scale <s> --format chrome --backend
optimized`. Apple M4 Max (12P + 4E cores), shared with other users and a
running test bar.

## Verdict

1. **The prover overhead is the Σ tail.** Over clear Akita, Whiteout adds
   +0.84 / +1.35 / +2.11 s (muldiv / fib 2^16 / fib 2^20, with the
   in-process response plan excluded). Σ and the response proof take
   0.86 / 1.39 / 2.15 s. The pad commitment adds 27–32 ms. The masked
   prefix is 40–75 ms *cheaper* than the clear opening it replaces, and
   stages 1–7 cost the same in both builds.
2. **The fixed cost is the response witness `e`, and `e` is the
   residual.** `e` holds 15.0M / 23.3M / 42.7M digits (ν = 24 / 25 / 26).
   96–98% of them are the ten radix-16 digits of each response
   coefficient. Those coefficients span the residual `s`: 1.44M / 2.24M /
   4.18M coefficients, the witness at the level after the cut. That
   witness barely shrinks with T: muldiv (T = 2^12) has 64% of fib 2^16's
   residual. Every Σ and response step is linear in `|s|` or in 2^ν, so
   16× less trace buys only 1.6× less overhead.
3. **The verifier overhead is the response proof's `A(r)`.** Whiteout
   verifies in 28.7 / 34.2 / 44.6 ms on 16 threads, against clear Akita's
   4.1 / 4.9 / 6.2 ms. On one thread it takes 63.5 / 90.6 / 150.6 ms
   (clear: 5.0 / 7.3 / 11.0). On one thread the residual `N·z` products
   are 45 / 51 / 62% of the verify. They scale with `|s|` too, at about
   24 ms per million residual coefficients.
4. **Regression: muldiv's response row.** The ν = 24 response plan now
   has one recursive fold and a 166 KB terminal. A muldiv proof is
   therefore 193.9 KB (clear: 86.7 KB; P0b, before T0: 93.3 KB). ν = 23,
   25 and 26 plan 4–5 folds and a 49 KB terminal. Akita's `response_plan`
   example reproduces this on its own (below).
5. **The largest lever is a protocol change.** A deeper cut shrinks `s`,
   which shrinks `e`, Σ, the response proof and `N·z` together (an
   estimated −55 to −70% of the tail). Every engineering lever together is
   worth 40–60% of the tail if every estimate holds, and leaves the cost
   linear in `|s|`. The ranking is under "Candidates".

## Measured (main pass)

Seven rounds. Each round runs all three workloads, clear and Whiteout
interleaved, and alternates which build goes first. Every run is a fresh
process under `/usr/bin/time -l`. A run started only once the 1-minute load
was below 8 (`uptime`). Launch loads were 4.7–8.0, median 6.7. Cells give
the median [min–max] of 7.

The 1-minute load never fell below 2 during the pass: another session's
test bar was running. Two Whiteout runs hit a load spike mid-run (muldiv
round 1, fib 2^20 round 0; 3–5× slower in every span). The medians absorb
them, and the ranges show them.

| | muldiv (T = 2^12) | fib 2^16 | fib 2^20 |
|---|---:|---:|---:|
| clear `jolt_prover::prove` (ms) | 176.6 [172.6–186.5] | 275.9 [266.3–290.5] | 622.9 [617.1–884.2] |
| Whiteout prove (ms) | 1,134.0 [1,113.4–5,570.1] | 6,284.2 [6,140.3–6,508.0] | 5,142.2 [5,055.5–14,702.4] |
| — of which response plan, first use (ms) | 113.8 [110.4–290.1] | 4,541.7 [4,504.0–4,713.2] | 2,388.2 [2,350.2–3,092.3] |
| Whiteout prove minus the plan (ms) | 1,020.7 [999.5–5,280.1] | 1,626.6 [1,600.2–2,003.9] | 2,730.5 [2,705.3–11,610.1] |
| **prover overhead, plan excluded** | **+844 ms (5.8×)** | **+1,351 ms (5.9×)** | **+2,108 ms (4.4×)** |
| `whiteout_prove_tail` (ms) | 859.8 [838.2–3,143.1] | 1,389.6 [1,371.6–1,728.1] | 2,146.5 [2,126.5–7,125.6] |
| — response proof (`whiteout_prove_response`) | 605.1 [591.0–1,976.3] | 962.2 [948.8–1,209.3] | 1,511.2 [1,498.7–5,011.1] |
| — Σ (tail minus response) | 252.8 [247.2–1,166.8] | 424.3 [418.0–518.8] | 635.3 [621.1–2,114.5] |
| masked prefix (`whiteout_prove_masked`) | 59.9 | 81.1 | 189.2 |
| clear opening it replaces (`akita_main_batched_prove`) | 99.3 | 148.3 | 264.2 |
| pad commitment (`commit_pads` minus the plan) | 26.7 | 27.5 | 31.8 |
| stages 1–7 and glue, clear / Whiteout | 55.6 / 57.2 | 96.5 / 96.7 | 264.7 / 265.3 |
| stage 0, clear / Whiteout | 20.2 / 8.7 | 29.5 / 15.5 | 97.0 / 81.0 |
| Σ attempts (every run) | 1 | 1 | 1 |
| verify, 16 threads, clear (ms) | 4.1 [4.0–4.2] | 4.9 [4.8–5.1] | 6.2 [6.0–7.6] |
| verify, 16 threads, Whiteout (ms) | 28.7 [26.5–108.5] | 34.2 [32.7–35.8] | 44.6 [42.8–118.7] |
| verify, 1 thread, clear (ms) | 5.0 [4.8–5.1] | 7.3 [7.2–7.4] | 11.0 [11.0–11.3] |
| verify, 1 thread, Whiteout (ms) | 63.5 [63.3–98.9] | 90.6 [90.2–93.3] | 150.6 [146.8–180.6] |
| first verify, transported, clear (ms) | 56.1 [55.6–57.7] | 57.6 [57.0–58.1] | 61.0 [60.8–64.9] |
| first verify, transported, Whiteout (ms) | 53.7 [52.0–198.2] | 69.2 [66.9–107.8] | 84.7 [80.9–163.9] |
| loading the persisted plans, catalog + response (ms) | 6.7 + 3.8 | 11.1 + 4.8 | 13.4 + 3.8 |
| proof bytes, clear | 86,704 | 91,693 | 94,146 |
| proof bytes, Whiteout | 193,927 [193,895–193,957] | 102,741 [102,689–102,758] | 105,283 [105,249–105,326] |
| — of which the response proof | 170.2 K | 77.0 K | 77.4 K |
| peak RSS, clear / Whiteout (GiB) | 0.3 / 1.0 | 0.4 / 1.7 | 0.6 / 2.4 |

Notes on the table:

- **Prove.** The response plan runs inside every fresh-process prove
  (first use). With persisted plans (O1, D32) a prover would load it
  instead, so the overhead row excludes it.
- **The Whiteout stage 0** is cheaper than the clear one because the ZK
  catalog row commits differently. That is not a Whiteout cost.
- **The transported first verify.** Clear Akita spends 55–60 ms of it
  outside its stages; this harness does not split that time. Whiteout's
  includes the layout's public state (pad commitment, residual matrices,
  `A_D`): 20 / 23 / 26 ms in `whiteout_response_lookup`. The first-verify
  gap is −2 / +12 / +24 ms.
- **BlindFold** does not run in an `akita,zk` build; Whiteout replaces
  it. Nothing else outside the spans above is Whiteout-specific.

### Verify, by bucket

From the same runs (`scripts/whiteout_verify_profile.py report` on this
log), median ms on 16 threads / 1 thread.

| bucket | muldiv | fib 2^16 | fib 2^20 |
|---|---:|---:|---:|
| response proof: `N·z` | 7.0 / 35.6 | 8.0 / 52.9 | 14.4 / 100.5 |
| response proof: eq dual | 2.9 / 4.3 | 4.6 / 7.7 | 5.8 / 10.7 |
| response proof: lifted rows | 2.6 / 6.1 | 2.6 / 6.2 | 2.7 / 6.3 |
| response proof: row batch | 1.5 / 1.1 | 3.2 / 2.7 | 3.7 / 3.2 |
| response proof: sumchecks and opening `E(r)`, by difference | 2.7 / 3.3 | 3.6 / 5.1 | 4.0 / 5.7 |
| pad-layout replay | 3.1 / 4.0 | 3.3 / 4.8 | 4.7 / 8.4 |
| lift | 2.1 / 1.8 | 2.1 / 1.8 | 2.3 / 1.9 |
| masked prefix | 1.8 / 2.7 | 1.9 / 3.5 | 3.3 / 7.0 |
| seed, stages 1–7, Σ check, relation | 3.2 / 3.0 | 3.3 / 3.1 | 3.5 / 3.2 |
| residual products `A_a` + `[B_a\|B′_a]` (CPU) | 19.2 / 28.8 | 49.9 / 46.0 | 107.0 / 93.5 |
| their share of the 1-thread verify | 45% | 51% | 62% |

With 16 threads, the `A_a` product's CPU time exceeds its one-thread time
(fib 2^16: 45.0 against 35.3 ms). The parallel `N·z` loses about 30% to
overhead. The script's sumcheck and opening sub-buckets read 0 since T0, so
that row is the response proof minus `A(r)` and the row batch.

## Where the tail goes

**Probe.** A scratch build added spans around each step of
`prove_under_dit` (jolt-zk `whiteout/sigma.rs`: self-checks, the
`ρ`/`l_s` precompute, each attempt split into sampling, `linear_parts`,
`N·y`, the `V` commit and compression, and challenge/shift/rejection, then
`witness` and the statement). It also spanned each step of
`AkitaResponseProver::prove` (`akita-pcs/src/zk/mod.rs`: `check_witness`,
commit, statement rows, product sumcheck, row batch, digit range, fused
sumcheck, opening). It logged `e`'s layout at `build_layout_state`. The
probe is not committed.

The probe never got the main pass's load (< 8): the test bar held the load at
16–150. Below are the medians of three runs at launch load 23–30. Their
tails run 1.1–1.4× slower than the main pass's (muldiv 0.98 s against
0.86 s), so read the shares, not the absolute numbers. The shares move
with load: an earlier set at load 36–100 put sampling at 30% of Σ, not
60–64%. These runs are the closest to idle available.

| step (ms, load 23–30) | muldiv | fib 2^16 | fib 2^20 | what it is |
|---|---:|---:|---:|---|
| **Σ** (tail minus response) | ≈ 268 | ≈ 460 | ≈ 690 | |
| sampling `y` | 161.4 | 286.1 | 443.0 | constant-time discrete Gaussian, one sample per residual coefficient |
| `N·y`, `linear_parts(y)`, `V`, challenge, shift, rejection | 40 | 69 | 99 | one attempt, minus sampling |
| self-checks | 21.8 | 36.9 | 48.2 | relation rows, `N·s = t`, alphabet |
| witness (`z`, `θ`) | 25.7 | 43.1 | 65.6 | |
| `ρ` and `l_s` precompute | 9.4 | 12.8 | 13.6 | |
| **response proof** | 709 | 1,457 | 1,878 | |
| fused sumcheck | 243.5 | 427.3 | 771.4 | over 2^ν, degree 3 |
| row batch | 219.1 | 288.3 | 434.1 | `statement.weigh`: `Nᵀω` per column |
| opening `E(r)` | 87.8 | 472.6 [188.8–877.9] | 218.1 [199.0–498.7] | Akita's embedded batched opening |
| digit range | 79.6 | 116.9 | 172.8 | |
| `check_witness` | 44.5 | 68.3 | 96.0 | recomputes `N·z`, the compression and the `V` commitment |
| statement rows | 36.9 | 64.8 | 90.6 | |
| commit to `e` (i8 digits) | 19.1 | 27.7 | 49.8 | |
| product sumcheck | 0.2 | 0.2 | 0.2 | |

**`e`'s layout** (from the probe's log line):

| | muldiv | fib 2^16 | fib 2^20 |
|---|---:|---:|---:|
| residual coefficients `\|s\|` (Gaussian segments) | 1,438,976 | 2,243,840 | 4,175,104 |
| response digits (10 per coefficient) | 14,389,760 (96.0%) | 22,438,400 (96.2%) | 41,751,040 (97.7%) |
| `V` (32 digits per coefficient) | 251,904 | 530,432 | 620,544 |
| compression witnesses (u_a ×2 chunks, t_D) | 348,160 | 348,160 | 348,160 |
| `\|e\|`, ν, fill of 2^ν | 14.99M, 24, 0.89 | 23.32M, 25, 0.69 | 42.72M, 26, 0.64 |
| residual ring dimension | 128 | 128 | 128 |

**Thread scaling.** muldiv with `RAYON_NUM_THREADS` at 1, 4 and 16, two
runs each at load 15–22, gives tails of 4.6–5.2 s, 1.44–1.46 s and 0.90 s.
So 16 threads reach 5.5×, and only 1.6× over 4. The steps that stop scaling
past 4 threads hold most of the 16-thread tail:

| muldiv step | 1 thread | 4 | 16 | 1 → 16 |
|---|---:|---:|---:|---:|
| fused sumcheck | 1,543–1,555 | 426–434 | 193 | 8.0× |
| sampling | 872–1,327 | 250 | 153 | 5.7–8.7× |
| row batch | 765–769 | 281–282 | 196 | 3.9× |
| digit range | 323–330 | 96–98 | 70 | 4.6× |
| opening | 338–346 | 105–106 | 79 | 4.3× |
| `check_witness` | 143–173 | 56–69 | 44 | 3.5× |
| statement rows | 137–146 | 51 | 38 | 3.7× |
| commit | 93–108 | 30–31 | 18 | 5.5× |

(The 16-thread column is the run at load 18.9; the second run hit a spike.)

## Why the small-trace cost is what it is

- **`|s|` is set by the Akita row, not by T.** The cut sits at level 1
  (`cut_level = 1`, Akita `specs/whiteout-akita.md` §4.1; "there is no
  planning for k ∈ {3, 4}"). The residual is the witness entering level 2.
  Level 0 folds the one-hot trace, whose committed size has a floor (K =
  256 chunks, precommitted program and advice groups). So muldiv's
  residual is 64% of fib 2^16's at 1/16 of the trace.
- **Ten digits per coefficient.** `BINDING_RESPONSE_DIGITS = 10` and α =
  2^17 make the Gaussian width about 2^39, so each residual coefficient
  becomes ten radix-16 digits of `e`. That is 96–98% of `e`.
- **Padding to 2^ν.** The fused sumcheck, the commit and the opening run
  over 2^ν. At fib 2^16 and 2^20, 31% and 36% of that domain is padding.
  The digit range runs over the live length.
- **Not the attempt count.** Every run accepted Σ on the first attempt
  (rejection ≈ 2^−13.2), so σ tuning for attempts has nothing left to
  win.
- **Not setup re-derivation.** A fresh process builds the layout's public
  state once (27–32 ms in `commit_pads`) and loads the plans in 10–17 ms.
  Both are cached afterwards.
- **Not the commit to `e`.** It costs 21–50 ms; the i8 commit (M2) made it
  cheap. The mostly-zero padding does cost in the sumchecks (above).
- **Sampling is CPU-bound in constant-time fixed-point arithmetic.** On
  one thread it costs 0.6–0.9 µs per sample. A `sample` profile of fib
  2^16 puts 25% of the Σ/response pool's busy samples in
  `Fix192::ct_product`, called from `GaussianSampler::trial`.
- **The row batch converts out of CRT-NTT five times per column.**
  `PreparedMatrix::apply_transposed` splits each 128-bit weight into five
  30-bit limbs. For each limb and column it accumulates in CRT-NTT, then
  converts back with `to_ring_ct` (inverse NTT plus constant-time Garner
  over 6 primes). `to_ring_ct` was 28% of the pool's busy samples in the
  same profile, almost all under `batch_rows → matrix_adjoint`. The
  weights (η, α, the product point) and the matrices are public.

## Muldiv's response row (regression)

Akita's `response_plan` example at `deb13c039` (`--features
catalog-gen,parallel`, with a local print of the schedule's fold count
and estimate):

| ν | find_schedule | recursive folds | terminal payload | first-direct setup |
|---|---:|---:|---:|---:|
| 23 | 1,046 ms | 4 | 49,206 B | 262,144 |
| **24** | **72 ms** | **1** | **165,956 B** | 262,144 |
| 25 | 2,275 ms | 5 | 49,206 B | 352,256 |
| 26 | 940 ms | 5 | 49,206 B | 524,288 |

The ν = 24 row reproduces the 170 KB response proof measured in jolt. P0b
planned ν = 24 in 2.1 s for a proof about 100 KB smaller. The setup
column suggests a cause, which this ticket has not verified: the
setup-first selection keeps the 262,144-element first-direct setup at
ν = 24, and only a one-fold schedule fits it. Prove and verify times under
a fixed row were not measured.

## Candidates

Savings come from the main pass's medians (load 5–8), scaled by the probe's
shares. They are estimates unless marked measured. "Security" uses three
classes: none; parameter change (needs a paper delta, no new argument);
protocol change.

### Ranked

| # | candidate | prove saving, muldiv / fib 2^16 / fib 2^20 | verify saving | security | size | risk |
|---|---|---|---|---|---|---|
| 1 | **Deeper cut (k = 2 → 3)**: one more masked level, so the residual is the witness entering level 3 | −0.45 to −0.6 / −0.75 to −0.95 / −1.2 to −1.5 s | 1 thread: −20 to −27 / −30 to −40 / −60 to −75 ms; 16 threads: −4 / −5 / −9 ms | protocol: Def. source-cutoff at k = 3, δ_fold over 3 masked levels, admission, a paper delta | large (Akita planner, admission and masked prefix; jolt-zk shape) | high |
| 2 | **Faster constant-time Gaussian sampler** | −75 to −115 / −130 to −200 / −205 to −305 ms (2–4× on sampling) | — | none if the output distribution is unchanged; a new algorithm or precision needs a statistical-distance argument against ε = 2^−137 (paper delta) | medium (jolt-crypto `sample`) | medium: constant time, distribution tests |
| 3 | **Row batch `Nᵀω` on public data**: one CRT-NTT exit per column instead of five (a 7th CRT prime, or a field-side transposed product), and the non-CT conversion, since the inputs are public | −95 to −125 / −125 to −165 / −175 to −230 ms (2–3× on the row batch) | — (the verifier's `A(r)` does not use this path) | none | small (jolt-crypto `ajtai.rs`, jolt-zk `linear.rs`) | low: an exact identity, testable against the current output |
| 4 | **Parallel scaling of the steps that stop at 4 threads**: digit range, opening, statement rows, `check_witness`, commit (the row batch is #3) | −60 to −120 / −80 to −150 / −100 to −180 ms | — | none | medium (Akita digit-range and opening kernels, jolt-zk) | low |
| 5 | **Deduplicate the prover's self-checks**: `check_witness` recomputes `N·z` and the `V` commitment, and Σ's self-check recomputes `N·s` and the relation rows; keep the alphabet and shift checks, which protect hiding | −40 to −55 / −70 to −90 / −95 to −115 ms | — | none for soundness (the verifier checks); less defense in depth against prover bugs | small | low |
| 6 | **Residual at d = 256** (O6's second step) | ≈ 0 | 1 thread: −14 / −23 / −47 ms; 16 threads: −2 / −3 / −5 ms (half of the residual products, estimated) | parameter: a D35 extension and SIS re-estimates | medium to large (planner ceiling, presets, re-pinned rows) | medium |
| 7 | **Fix the ν = 24 response row** (selection or setup tier) | not measured | not measured | none: any admitted row has the same security; the response catalog is re-blessed | small once the cause is known | low to medium |
| 8 | **Padding-aware fused sumcheck, commit and opening**: skip the known-zero suffix of 2^ν in the early rounds | −20 / −110 to −150 / −200 to −250 ms | — | none (the live length is public) | medium (Akita fused and opening kernels) | medium: constant-time paths on `e` |
| 9 | **Persist the layout's public state** (pad commitment, residual matrices, `A_D`) beside the plans | −27 to −32 ms per fresh-process prove; 0 warm | first verify −20 / −23 / −26 ms | none (deterministic public data; validate as O1 validates plans) | small | low |

Notes on the estimates:

- **#1, the deeper cut.** In the checked-in ZK rows, the level after the
  cut shrinks the witness 2.6–3.9× (`fp128_onehot_recursive_zk`: level 2
  out/in = 0.38 and 0.25; level 3 = 0.32 and 0.24). Moving the cut one
  level down divides `|s|` by about that much, and with it Σ, the
  live-length steps and `N·z`. ν drops by 1–2 at every workload. Against
  that come one more masked level in the prefix (+30–70 ms prove, and
  bytes not estimated) and the residual commitment's new shape. This is
  the only candidate that changes the overhead's order of magnitude.
- **#2, the sampler.** The saving assumes a sampler 2–4× faster at the
  same distribution. Its 1-thread cost (0.6–0.9 µs per sample) leaves
  room for that, but none was prototyped.
- **#3, the row batch.** The 2–3× assumes the conversions are about 80%
  of the row batch, as the `sample` profile suggests (`matrix_adjoint` is
  85% of `batch_rows` on the coordinating thread).
- **#6, d = 256.** Its gate is now met. The campaign's rule ran O6's
  d = 256 step when the residual products reach 50% of the 1-thread
  verify. After O6 (d = 128) they read 40% at fib 2^20; now they read 51%
  at fib 2^16 and 62% at fib 2^20, since the rest of the verify shrank
  (O3b) and the residual did not. The prover gains nothing measurable:
  `N·y` and `N·s` are 3–6% of Σ.
- **#7, the response row.** This is a byte fix (−99 KB on muldiv). P0b's
  muldiv tail, with the multi-fold row, was 1.02 s against 0.86 s now, but
  P0b predates O6 and M2, so the time effect is unknown.
- **#8, padding.** The estimate takes the fused sumcheck and opening
  times × (1 − fill) × 0.8. At muldiv the fill is 0.89, so it buys little
  there.

### Already landed, rejected or queued (not re-proposed)

- **P9 (D = 9, α = 2^13), queued since O7.** It removes 10% of the
  response digits and crosses no power of two: `|e|` becomes about
  13.5M / 21.1M / 38.5M, which stays at ν = 24 / 25 / 26. So it saves only
  the live-length steps (Σ, row batch, digit range, rows): about −40 /
  −70 / −100 ms prove and −10% of `N·z`. The cost is one rerun per about
  595 proofs. The new numbers do not raise its case. It compounds with
  #1 and #6.
- **The κ₁ and extraction-radius levers** were dropped in O7 (≤ 0.24
  bits).
- **Fewer Σ attempts.** There is nothing left: one attempt in all 21
  Whiteout runs.
- **Overlapping the opening `E(r)` and the lifted rows with `A(r)`
  (O3b).** It is worth 3–6 ms on 16 threads now, and still blocked by
  the harness's per-thread span attribution.
- **A speculative pad-layout replay (O3b).** The replay is 3–5 ms on 16
  threads; not worth the speculation.
- **The O1/O2/O3/O3b/O4/O5/M1/M2 items** have all landed. The
  `A`-streaming, i8 commit and allocator items are in M2.

### Engineering against protocol

Engineering candidates #2–#5, #8 and #9 together are worth roughly
−0.3 to −0.45 s on muldiv and −0.55 to −0.8 s on fib 2^16, 40–60% of
the overhead if every estimate holds. They leave the overhead's shape alone: still linear in
`|s|`, still about 1 s at muldiv. #1 is a protocol change and the only
lever on `|s|` itself. #6 is a parameter change for the verifier only, and
#7 a parameter selection for bytes.

## Reproduce

- **Main pass.** A driver interleaved the two builds over the three
  workloads. It recorded the `summary.json` prover spans, the verify
  spans through `scripts/whiteout_verify_profile.py`'s `split`, the Σ and
  response-proof trace events (`attempts`, `response_bytes`) and peak RSS.
  `whiteout_verify_profile.py run --clear-bin … --zk-bin …` reproduces the
  verify half alone.
- **Probe.** Add `tracing::info_span!` around the steps listed under
  "Where the tail goes", and a `tracing::info!` of
  `sigma.response_layout(..)` in `WhiteoutTail::build_layout_state`.
- **Response row.** `cargo run --release -p akita-planner --features
  catalog-gen,parallel --example response_plan -- 23 24 25 26`, printing
  `planned.schedule.recursive_folds.len()` and `planned.estimate`.

## Not verified

- **No idle measurement.** The main pass ran at load 4.7–8.0, the probe at
  23–30 and the thread scaling at 15–22, all against a running test bar.
  The idle gate (< 2) of P0b never opened.
- **Every saving is an estimate.** Nothing was prototyped. The deeper
  cut's 2.6–3.9× comes from the checked-in ZK rows' level ratios, not
  from muldiv's or fib's own rows (which are planned in process).
- **The cause of the ν = 24 regression** is unverified. Neither the
  pre-T0 Akita nor a fixed row was run.
- **`sample` attributions** come from one fib 2^16 run, which sampling
  slowed about 3.5×. They are used only to name hot functions.
- **Proof validity** rests on the harness's own verify asserts (every
  measured run verified). No fixture or tamper test ran.
- **One machine** (M4 Max). x86_64 sampler and NTT costs differ.
