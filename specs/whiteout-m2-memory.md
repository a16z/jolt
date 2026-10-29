# Whiteout prover memory (M2)

Parent: `specs/whiteout-optimization-campaign.md`; predecessor
`specs/whiteout-m1-memory.md` ("What is left, and why"). Changes: jolt
`wo/m2-memory` (jolt-algebra, jolt-zk) and Akita `wo/akita-m2` (akita-pcs
`zk/`, akita-cpu-backend signed-block sources, akita-types compression
weights). One commit per lever on each side.

M1 left two tables of `e`'s length live through the response proof: the
committed polynomial of `e` over `F`, and the batched row weights `A`. At
`e` = 2^26 each is 1 GiB. M2 removes both.

- **L1.** The commitment holds `e` as `i8` digits (`SignedBlockPoly`), 64
  MiB at 2^26 instead of 1 GiB.
- **L2.** `A` is never built. jolt-zk returns it factored, and the fused
  sumcheck streams `A` and `E` for its first three rounds. It then holds
  both, bound, in 2·2^(ν−3) entries: 256 MiB at 2^26 instead of 1 GiB
  for `A` alone.

## Result

Peak memory and prover time of
`jolt-prover profile --name <w> --scale <s> --backend optimized --format chrome`,
built with `--features profiling,akita,zk` (release), under
`/usr/bin/time -l`. Base is jolt `ac5102b86` with Akita `9c6c7273f`
(`whiteout/dev` and `~/akita-whiteout`); M2 is the final heads of both
branches. Apple M4 Max, 16 cores, shared machine. One discarded warm-up
per binary and workload, then nine interleaved pairs: three base-first,
then six alternating which binary runs first. The 1-minute load average
was 8–20 during the runs.

| workload | peak RSS base → M2 | peak footprint base → M2 |
|---|---|---|
| muldiv (`e` 2^24) | 1.72–1.74 → 1.23–1.25 GiB (−29%) | 1.44–1.46 → 0.95–0.96 GiB (−34%) |
| fib 2^16 (`e` 2^25) | 2.74–2.76 → 1.75–1.79 GiB (−35%) | 2.39–2.41 → 1.42–1.43 GiB (−40%) |
| fib 2^20 (`e` 2^26) | 4.45–4.49 → 2.58–2.60 GiB (−42%) | 4.10–4.15 → 2.23–2.27 GiB (−46%) |

Each lever removes about one `F` table of `e`'s padded length: 0.25, 0.5
and 1 GiB at 2^24, 2^25 and 2^26. An earlier batch at the previous agent's
heads measured the levers one at a time (base → L1 → L1+L2, fib 2^20:
4.45–4.49 → 3.53–3.54 → 2.58–2.59 GiB). Base is lower than M1's "after"
column (2.19 / 3.44 / 5.28 GiB) because M1's figures predate the O6 merge.

Medians of the nine runs, seconds:

| workload | `jolt_prover::prove` base → M2 | `whiteout_prove_tail` | `whiteout_prove_response` |
|---|---|---|---|
| muldiv | 1.738 → 1.740 (+0.1%) | 0.968 → 0.971 | 0.683 → 0.697 (+2.0%) |
| fib 2^16 | 3.546 → 3.640 (+2.7%) | 1.370 → 1.446 | 0.966 → 1.033 (+6.9%) |
| fib 2^20 | 4.631 → 4.527 (−2.2%) | 2.393 → 2.350 | 1.668 → 1.656 (−0.7%) |

The prover time stays within the 5% budget. The run-to-run spread is
larger than the differences: fib 2^16 base ranged 3.34–4.23 s as the load
moved. A batch of six base-first pairs before the scratch-wipe fixup gave
medians 1.712 → 1.686, 3.383 → 3.349 and 4.104 → 3.894 s, with the
response span at +0.8% on fib 2^16. The fib 2^16 response figure above is
therefore not a stable regression. Akita's `zk_response` example times
the response prover alone in-process (scale 2000, `e` = 23.9 M digits,
ν = 25). In interleaved process pairs before that fixup, base took 815–888
ms (median 850) and M2 took 816–924 ms (median about 825).

Proofs stay valid. Every profile run verifies its proof, and the harness
prints the verifier timings only after verification succeeds (all 54
measured runs). Proof sizes vary by up to 0.1 KB between runs of the same
build, base included, because the hiding randomness is fresh. No run was
seeded, and nothing seeded was built.

### Why three streamed rounds

A streamed round writes `A` from its factors and lifts `E` from the
digits over all 2^ν entries, so it costs more than a held round, which
reads only the entries left after the binds so far. Each streamed round
also halves the buffer held afterwards. At fib 2^20:

- **Two rounds.** The 512 MiB buffer could not reuse the memory that the
  digit-range session had freed. Peak RSS was 3.31–3.32 GiB, against
  2.77–2.79 GiB with four rounds (load 54–62).
- **Three or four rounds.** The buffer fits in that freed memory. Peak
  RSS was 2.60–2.61 GiB with three rounds and 2.58–2.59 GiB with four
  (three interleaved pairs, load 8–13).
- **Four rounds.** On `zk_response`, the response prover was about 3.5%
  slower than base. With three rounds it is level.

So the proof streams three rounds (`STREAMED_ROUNDS`, Akita
`zk/rounds.rs`).

## What changed

### L1: the committed `e` as `i8` digits

- **Akita `zk/mod.rs`, `AkitaResponseProver::prove`.** The prover commits
  `SignedBlockPoly::i8(committed.to_vec(), ν)` in place of
  `DensePoly::from_field_evals` of `e`'s field lift. The backend owns the
  digits until the opening. `committed`, a second copy, stays with the
  fused sumcheck.
- **The signed-block source is hardened for secret values**
  (akita-cpu-backend `sources/`). Nothing in production used
  `SignedBlockPoly` before.
  - Ring values enter `F` through `lift_ct`, not `Field::from_i8` or
    `from_i16`, which branch on the sign. The sites are `RingValues` for
    `[i8; D]` and `[i16; D]`, and `convert`.
  - Signed readers never skip a zero. `RingValues::SKIPS_ZEROS` is `false`
    for signed values, and `evaluate_and_fold_base` reads that flag.
  - `SignedBlockPoly` wipes its owned values on drop. `SignedScratch`, the
    reader's staging buffers, wipes its own on drop (the fixup commit).
- **jolt-algebra owns `lift_ct`.** It moved there from jolt-zk's
  `segments.rs`, so jolt-zk, akita-pcs and akita-cpu-backend all import
  the same function.

### L2: `A` streamed

- **jolt-zk `ResponseStatement::weigh`** returns `ResponseWeights` instead
  of a dense vector.
  - A response segment of `n` coefficients keeps its `n` coefficient
    weights `w`. Its digit `p·n + k` weighs `16^p·w_k`.
  - V, θ and the compression witnesses stay dense. They are small.
  - `fill(start, out)` writes any range. Each value is the one
    `PlaneLayout::transpose` wrote before, `w_k·16^p` added to zero.
- **Akita `zk/rows.rs`: `batch_rows` returns `BatchWeights`.** It holds the
  jolt-zk weights, the chains' weights on the hull of their digit
  intervals, and the padding term's two half-tables of `eq(τ_pad, ·)`. Its
  `fill` writes `A` over any range of the padded `e`.
- **The chains' weights are computed on that window only.**
  `ReducedCompressionRelationWeights::accumulate_dense` (akita-types) now
  takes the window start and adds only the window's entries. An event
  outside the window is a typed error. The existing full-table callers
  pass 0.
- **Akita `zk/rounds.rs`: the `FusedKernel` streams its first rounds.**
  For the first `STREAMED_ROUNDS = 3` rounds it holds neither `A` nor `E`.
  - Each round works in chunks of 4,096 unbound entries. A chunk is
    written (`A` by `fill`, `E` by `lift_ct` of the digits), folded
    through the challenges bound so far by `fold_in_place`, and summed by
    `RoundB::sums`.
  - The bind that ends round 3 writes `A` and `E`, bound, into one
    `SecretTable` of 2·2^(ν−3) entries. From then on the kernel folds in
    place exactly as M1's did.
  - The sparse part `S` reads `E` at a pair by lifting and folding that
    pair's 2^(j+1) digits.
- **Values are unchanged.** A fold is local to a pair, and chunks start
  at multiples of 2^(j+1). Each streamed entry is therefore the same
  `Fold::fold_one` of the same operands as the held table's. The round
  sums are the same field elements added in another order, and field
  addition is exact. The proof's transcript does not depend on the
  representation.

## The constant-time argument for L1 (A45)

**Claim.** The response prover's reads of `e` that M2 adds or changes
take time and touch memory addresses independent of `e`'s digits. This
covers everything the signed-block source runs for the response schedule
and the streamed fused rounds. Two things are argued separately below:
comparisons whose outcome is the same for every honest `e`, and the
shared kernels that already consumed `e`'s digits before M2.

Line numbers are at Akita `wo/akita-m2` and jolt `wo/m2-memory` heads.

**What "honest" fixes.** `ResponseProver::prove_response` (Akita
`zk/mod.rs:537`) runs `ResponseStatement::check_witness` before proving.
Every digit is then in the balanced radix-16 alphabet `[−8, 7]`. The
honest padding past `|e|` is zero. Only the test-support
`UncheckedResponseProver` skips the check. The response schedule commits
one digit of 4 bits at `D` = 512. Its accepted reach is `(8, 7)`, and its
centering threshold is `q/2`, because `decompose_centering_threshold(1, 4,
q)` is `q/2` when 4 ≠ 128.

**1. Every access index is public.**

- The signed-block source reads `e` only by ring and block ranges that
  the schedule's geometry fixes: `SignedBlockDigits::source_rings`,
  `rows_i8` (`signed_blocks.rs:542`), `RingValueSource::values`, and the
  block loops of the commit and opening kernels.
- The fused kernel reads digits `[start, start + chunk)` at public chunk
  starts (`ChunkScratch::stream`, `FusedKernel::hold`) and pairs at
  `I_bin`'s public positions (`sparse_evaluations`, its `index % 2` test
  is on the public index). `bound_e` (`rounds.rs:352`) is the one reader.
- `fold_in_place` and `SecretTable::fold` index by loop counters; their
  `get(..).unwrap_or_default()` tests a public length.
- No index anywhere is computed from a digit.

**2. Every i8 → F lift is `lift_ct`.**

- `lift_ct` (`jolt-algebra/src/algebra.rs:633`) computes the magnitude
  by `(x ^ s) − s` and the sign by multiplying with `s & 1`. `Fp128`'s
  `from_u64` is `pack(v, 0)` (`solinas/fp128.rs:1040`). Nothing branches.
- The signed `RingValues::coeff` uses it
  (`akita-cpu-backend/src/sources/block_digits.rs:42,49`).
- So does `SignedBlockPoly`'s `convert` (`sources/signed_blocks/owned.rs:280`).
  The response schedule does not reach it (see the table), but it is
  hardened anyway.
- So does the fused kernel's `bound_e` (`akita-pcs/src/zk/rounds.rs:357`).
- jolt-zk's own lifts (`challenge.rs`, `lift.rs`, `linear.rs`,
  `response.rs`, `segments.rs`) import the same function.

**3. No secret-dependent branch or memory access.**

- **Folding happens after lifting, over `F`.** In the fused kernel `E` is
  folded only after `bound_e` has lifted it, by `fold_in_place`. That is
  `Fold::fold_one`, `even + r·(odd − even)`
  (`jolt-algebra/src/solinas/unreduced.rs:705`).
- **Summands are field arithmetic:** `RoundB::sums`, `fused_term` and
  `sparse_evaluations`.
- **The signed-block opening multiplies lifted values by public
  weights.** These are `seam_packing_partials` and
  `coefficient_packing_partials_from_position_source`
  (`sources/dense/kernels.rs:249` branches only on the public
  `flat_index < live_coeff_len`). So is `evaluate_and_fold_base`, which
  no longer skips zero signed values (`block_digits/evaluate.rs:39`).
- **Comparisons on `e`'s extrema have one outcome.** The signed-block
  path computes the least and greatest digit and compares them with
  bounds. Those comparisons branch, so the argument is that every honest
  `e` takes the same branch:
  1. **Admission** (`commitment/api.rs:341`, which calls
     `SignedBlockPoly::committed_centered_reach`, `owned.rs:188–195`).
     The extrema come from `signed_extrema` (`signed_blocks.rs:77`). The
     test `positive ≤ top && negative_abs < q − top`, with `top = q/2`,
     is true for every `i8`. The exceeds test against `(8, 7)` is false
     for every honest `e`.
  2. **Commit source** (`arithmetic/commitment.rs:280`, which calls
     `SignedBlockDigits::new`, `signed_blocks.rs:235`).
     `per_value_bounds()` is `None` for this schedule, because the
     accepted interval keeps the `i8` carrier centered
     (`preserves_carrier`, threshold `q/2 ≥ 127`). So the per-value,
     branching `AcceptedReach::centered` (`signed_blocks.rs:125`) never
     runs. The known extrema set `within_bounds`, which is true for every
     honest `e`. `single_fits(shape)` is true from the accepted bounds
     alone (8 ≤ 8, 7 < 8). Every commit request therefore borrows the
     digit rows as they are, with no scan (`rows_i8`).
  3. **Opening source** (`signed_blocks/kernels.rs`, `with_source!`, no
     accepted bounds). In `rows_i8` the per-request scan is a min/max
     fold (`track`, `signed_blocks.rs:106`). Its per-value branch (line
     117) needs `scan.accepted`, which is `None` here. `digits_fit` then
     compares the chunk's extrema with `[−8, 8)`. That is true for every
     honest `e`, so the borrowed rows are always returned.
- **The extrema are computed without branches.** In the release
  `jolt-prover` binary built at the previous agent's heads (aarch64), the
  `i8` instance of `slice_extrema`'s parallel fold compiles to NEON
  `smin.16b` / `smax.16b` / `sminv` / `smaxv`, with `csel` for the scalar
  tail and the reduce. It is inlined into rayon's
  `bridge_producer_consumer::helper`, and its conditional branches test
  lengths and counters only. The same holds for the signed-block
  `i8_rows` instances that contain the scan. M2's later changes do not
  touch these functions; the disassembly was not repeated at the final
  heads.

**Which signed-block functions the response reaches.** The previous
agent found these with temporary `Once`-guarded probes, run once on
muldiv and then removed. They were never committed.

| function | reached |
|---|---|
| `committed_centered_reach` | yes |
| `SignedBlockDigits::new` (commit: `D` = 512, accepted bounds and extrema given) | yes |
| `rows_i8`, borrowed path (shape 1 × 4 bits) | yes |
| `coefficient_packing_partials_batch` | yes |
| `decompose_fold_batch` (1 digit, 4 bits) | yes |
| `write_rows` (decomposition) | no |
| `evaluate_and_fold` | no |
| `source_coefficients` | no |

On aarch64 the commit takes the i8 NTT matvec. On x86_64 with AVX-512
IFMA, `commit_rows` (`arithmetic/commitment.rs:324`) can prefer
`commit_rows_i16`. That path reads `rows_i16` through `write_rows`, which
copies the digits into `SignedScratch`. Its reads are still at public
ranges, and since the fixup its buffers are wiped. That path was not run.

**4. Wiping.** Every buffer that holds `e`'s digits or their lifts:

| buffer | wiped by | location |
|---|---|---|
| `committed`, the padded digits (prover, then the fused kernel's `digits`) | `Zeroizing<Vec<i8>>` | `zk/mod.rs:253`, `rounds.rs` `FusedTables::Streamed` |
| `SignedBlockPoly`'s copy, owned by the backend until the opening | `Drop`: zeroes the whole capacity, in parallel | `owned.rs:127,142`; also on a rejected length (`owned.rs:45,53`) |
| `SignedScratch` (`raw`, `i8`, `i16` staging rows), new with this representation | `Drop`, and before a row buffer grows: zeroes the whole capacity | `signed_blocks.rs:435,455` |
| the fused kernel's `E` chunk scratch | `SecretTable` | `rounds.rs:67` (`ChunkScratch::e`) |
| the held `A`/`E` buffer | `SecretTable` | `FusedKernel::hold` |

The `A` scratch and `BatchWeights` are public and are not wiped. M1's one
exception, the committed polynomial's unwiped `F` copy, is gone, and so
is its digit cache.

Still unwiped:

- Stack arrays. Each holds at most one ring's `[i8; 512]`
  (`RingValueSource::values`, per position of the coefficient-packing
  pass) or at most 16 lifted field elements (`e_first`,
  `sparse_evaluations`).
- The cached extrema `(min, max)` in `SignedBlockPoly`'s `OnceLock`.
- `SignedBlockPoly::source_coefficients` returns its lifts in a plain
  `Vec<F>` (the trait returns a `Cow`). The response schedule never
  reaches it (see the table), and a `WARNING` at `owned.rs:293` says that a
  secret source must not.

**The shared kernels, which take the same inputs as before M2.**

- **Commit.** The commit matvec runs `I8CommitPolicy::Dense` with
  `check_zero = false` (`arithmetic/commitment.rs:296`), as the
  `DensePoly` path did (`:235`). Its CRT reconstruction is `to_ring`, not
  `to_ring_ct`, and it acts on the secret inner images.
- **Opening.** The opening runs the decompose-fold accumulators. At
  `D` = 512 these are the narrow and windowed sparse kernels, whose
  indices come from the public challenges. The rotated kernel, which
  skips zero digits (`rotated_accum.rs:63,95`), runs only at `D` = 64 (`should_use_rotated_challenge`, `rotated_accum.rs:8`).
- **Digit-to-Montgomery lift in the NTT.** The commit's NTT conversion
  (`assign_i8_with_lut` → `DigitMontLut::fill_ntt_limb`,
  `jolt-algebra/src/ring/crt_ntt_repr/lut.rs`, from Akita's
  `mat_vec_mul_blocks_i8_with_params_impl` under `Dense`) has a scalar
  fallback, `fill_limb`. At M2 it read `DigitMontLut::get`, which indexed
  a 16-entry table by the secret digit. The fallback runs whenever no
  vector transform applies:
  - aarch64 with `i16` primes, since the NEON path needs `i32`;
  - x86_64 without AVX2, and x86_64 at `D < 64`
    (`use_x86_transform_ntt`), including the response commit's
    compression levels at `D` = 16 and 8;
  - any target without NEON or AVX2.

  **Resolved (U1).** `DigitMontLut::get` now computes the digit's
  Montgomery form as one Montgomery product by `R^2`
  (`NttPrime::from_canonical`), with no table. The vector paths were
  already table-free: NEON (`centered_i8_to_mont_i32`,
  `forward_ntt_i8_i32`) and AVX2/AVX-512 (`forward_ntt_i8_*`) sign-extend
  the digits and multiply by `R^2` or `psi^i R^2`. The IFMA i16 kernel is
  not on this chain. Its scalar fallback (`Ifma52Prime::canonical_i16`,
  `mul` by `%`) branches and divides on its input; that is recorded as
  open in the U1 campaign entry.

  On the M4 the fallback is still not live for the response. A temporary
  probe (uncommitted, removed) logged every `fill_ntt_limb` call during
  the response commit on muldiv and fib 2^20. Each one had `i32` primes
  (`K` = 6) and the NEON plan, at `D` = 512, 64, 16 and 8, so each took
  `neon::forward_ntt_i8_i32` or its cyclic form.
- **Digit range.** The digit-range session (`DigitRangeProver`) reads its
  own unwiped copy of the digits (`Arc::from(digits.range)`, `zk/mod.rs:321`)
  and groups them by value class in `CompactDigitSource`.

These kernels received the same `i8` digit rows before M2. M2 does not
change their exposure. They remain requirements on Akita's shared
kernels, recorded in Akita `specs/whiteout-akita.md` A45, which M2
updates to this state.

## Where the peak is now

At fib 2^20 the footprint rises through the response proof. It peaks at
about 2.24 GiB during the digit-range reduction (`digit_range_*` spans,
about +0.42 GiB). The fused sumcheck's held buffer reuses memory that
session freed, and the opening adds almost nothing after it. The next
lever is Akita's `DigitRangeProver`, which copies `e`'s digits into its
own packed and class tables. It was not in scope here.

## Checks

All of these passed on the final trees: jolt `wo/m2-memory` rebased onto
`ac5102b86`, and Akita `wo/akita-m2` at `9be943a5e`.

- **Akita.**
  - `cargo fmt --all --check`.
  - clippy `-D warnings`, `--all --all-targets --release
    --no-default-features`, with `parallel,disk-persistence` and with
    `parallel,disk-persistence,dev-protocol`.
  - clippy on `-p akita-pcs` with `parallel,zk` and `parallel,zk-prover`,
    and on `-p akita-cpu-backend` with `zk` and `zk-prover`.
  - The CI nextest: `--lib --bins --tests --profile ci --cargo-profile
    ci-test --no-default-features --features
    parallel,disk-persistence,catalog-gen`. 1920 passed, 13 skipped.
  - `-p akita-cpu-backend --features zk-prover`: 549 passed.
  - `-p akita-pcs --no-default-features --features
    parallel,test-support,logging`: 226 passed.
  - `scripts/check-rust-file-lines.sh` and `check-doc-guardrails.sh` fail
    on files M2 does not touch: `commitment/executor.rs`,
    `setup_contribution/tests.rs` and book paths into `akita-algebra`. They
    fail the same way at base.
- **jolt.**
  - `cargo fmt --all --check`; `check_style_invariants.py --base
    ac5102b86`.
  - clippy `-D warnings` on `jolt-prover` with
    `akita,zk,prover-fixtures` and `akita,zk,profiling,prover-fixtures`;
    on `jolt-verifier` with `akita,zk,prover-fixtures,logging`; on
    `--all` with `host` and `host,zk`; and on `jolt-akita` with
    `zk-prover`.
  - nextest:
    - `jolt-prover --features akita,zk,prover-fixtures --test
      whiteout_e2e`: 6/6.
    - jolt-akita: 89 passed. With `zk-prover`: 94 passed.
    - `jolt-verifier --features akita,zk,prover-fixtures,logging
      --test-threads 1`: 88 passed.
    - `-p jolt-crypto -p jolt-zk -p jolt-dit -p jolt-sumcheck -p
      jolt-transcript --features jolt-zk/whiteout-prover`: 346 passed.
    - `-p jolt-algebra`: 45 passed.

## Not verified

- **Byte identity.** Base and M2 were not compared under a seeded
  randomness source. Identical values are argued above. The evidence is
  that every proof verifies, including the Akita zk lane's honest and
  cheating provers. The verifier evaluates `A(r)` independently of the
  prover's streamed `A`.
- **Small shapes.** No test proves a response with ν < 3. There the
  kernel never leaves the streamed state, and `e_first` folds the digits
  directly. The synthetic shapes in the tests are larger.
- **The scalar digit fill off the M4.** U1 made it table-free and
  checked it on the M4 with the scalar plan forced. On x86_64 it was only
  type-checked (`cargo check --target x86_64-apple-darwin`), not run.
- **Timing.** No dudect run. The constant-time argument is by code
  reading and by one compiler's output on aarch64, inspected at the
  previous agent's heads. x86_64 was not inspected, and its IFMA i16
  commit path was not run.
- **Linux memory and T = 2^24** were not measured.
- **The load.** All times come from a shared machine at a load average of
  8–20. Differences below about 5% are inside the run-to-run spread.
