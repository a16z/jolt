# Whiteout verifier profile and benchmark (O0)

Parent: `specs/whiteout-optimization-campaign.md` (O0),
`specs/whiteout-v6-costs.md` (baseline).

> **LOADED — not for decisions.** The measurement gate (1-minute load < 2)
> never opened: 37 polls, 5 minutes apart, from 07:59 to 11:04 on
> 2026-10-25 read 2.86–148, and the pass below ran at load 94–133. Absolute
> times are 5–8× V6's. Use the shares and the structure, not the
> milliseconds, and rerun the benchmark on an idle machine before any
> ticket reports against it.

## Verdict

1. **Per-verify recomputation dominates.** With one thread, 84–85% of the
   Whiteout verify rebuilds public state that is fixed by the statement's
   shape: the Σ setup twice, the pad commitment, the residual and pad
   matrices' CRT-NTT and its inverse, and the pad-layout replay. With 16
   threads it is 58–72%. This is O3's lever, and it is larger than O3
   assumed.
2. **The residual N·z products are small.** `A_a` plus `[B_a|B'_a]` is
   4–8% of the one-thread verify. Counting the residual's per-verify matrix
   setup and field-entry conversion too, it is 42–60%. Almost all of that
   setup goes away under O3.
3. **The O6 gate (residual ≥ 50% of verify time) is not met by the
   products.** It is met only by the residual's setup, which O3 removes
   without changing a parameter. P9's gating ("the residual's measured
   share is worth it") reads the same way.
4. **The Whiteout verifier does parallelize.** V6 found the 1- and
   16-thread times equal. That came from the harness: its
   "single-threaded" mode only routed Akita backend calls to a one-thread
   pool, and the stages and Whiteout's parallel loops ran on the global
   pool. O0 installs the pool around the whole verify, as the Dory harness
   does (`29cb287ea`). Measured correctly, one thread is 2.6–2.9× slower
   than 16 threads, under load.
5. **First use is planning.** As in V6, a fresh verifier spends 32–62 s
   (loaded) in `find_schedule` (response plan, under
   `whiteout_response_lookup`) and `find_zk_schedule` (catalog plan, under
   the stages bucket). This is O1/O2's lever. The masked prefix, Σ check
   and sumchecks are each ≤ 3%.

**Recommended order: O3 before O6.** O3 removes most of the steady verify
without touching a parameter. Rerun this benchmark after O3. Then decide
O6 d = 256 and P9 on the residual products' share of the post-O3 verify,
which these numbers put well under 50%.

## What is recomputed per verify

Everything below depends only on the pad layout and the Whiteout row,
never on the proof, except `u_a`. `u_a` enters `N` only as a target, not as
a matrix.

| work | where | one-thread share (fib 2^20) |
|---|---|---|
| pad-layout replay: stages 1–7 and the masked prefix rerun on zero bytes | `jolt_verifier::whiteout::pad_layout`, called by every `verify` | 1% |
| pad commitment: `public_stream` and `PreparedMatrix::new` (CRT-NTT) | `WhiteoutTail::pad_commitment`, built 3 times per verify (pad receipt, Σ setup, response lookup) | 5% + 6% + in lookup |
| residual matrices `A_a`, `[B_a\|B'_a]`: `PreparedMatrix::new` from Akita's setup stream | `ResidualShape::blocks`, built twice | 11% + in lookup |
| `A_D`: stream and `PreparedMatrix::new` | `WhiteoutTail::sigma`, built twice | 1% + in lookup |
| a whole Σ setup with a zero `u_a`, only to size `e` (`num_vars`) | `WhiteoutTail::response` | 22% |
| field entries: the inverse CRT-NTT of `A_a`, `[B_a\|B'_a]` and the pad matrix | `PreparedMatrix::field_entries`, lazily on the first `combine_columns` of each fresh matrix | 11% + 19% + 8% |

Two of these are pure waste, not just cacheable state:

- The verifier only reads matrix entries over F (`combine_columns`). It
  converts the stream to CRT-NTT in `PreparedMatrix::new` and back in
  `field_entries`. A verifier-side matrix built straight from the stream
  skips both transforms.
- `WhiteoutTail::response` builds a full Σ setup to compute one integer.

Holding the Σ setup's matrices, the pad commitment and the response
`num_vars` per (row, pad layout) in the verifier's derived state would
remove all of this from steady verify. That state sits next to the plans O1
persists.

## Results (LOADED, n = 1 per cell)

The time in ms, with the share of that verify in parentheses. "1 thread"
and "16 threads" run in the same process after proving, so plans are warm.
"Products (CPU)" are summed over worker threads, so with 16 threads they
can exceed their parent's wall time.

**fib 2^20 (T = 2^20).** Whiteout proof 117,491 B (load 116); clear Akita
88,685 B (load 115).

| bucket | 16 threads | 1 thread |
|---|---:|---:|
| clear Akita verify | 40.2 | 23.1 |
| Whiteout verify | 1,199.9 | 3,141.5 |
| seed, stages 1–7, stage-8 claims | 10.1 (1%) | 10.8 (0%) |
| pad-layout replay [recomputed] | 47.1 (4%) | 30.5 (1%) |
| pad receipt [recomputed] | 58.5 (5%) | 156.0 (5%) |
| masked prefix (levels 0, 1) | 40.7 (3%) | 12.2 (0%) |
| relation (auxiliary block) | 13.0 (1%) | 8.7 (0%) |
| Σ setup [recomputed] | 214.2 (18%) | 578.9 (18%) |
| response lookup [recomputed] | 275.9 (23%) | 693.8 (22%) |
| lift and Σ check | 2.6 (0%) | 4.2 (0%) |
| response proof | 536.8 (45%) | 1,645.5 (52%) |
| — row batch, digit range, sumchecks | 18.5 (2%) | 18.6 (1%) |
| — A(r): eq dual, A_D·V, lifted rows | 241.4 (20%) | 145.4 (5%) |
| — A(r): N·z field entries [recomputed] | 154.9 (13%) | 1,208.3 (38%) |
| — A(r): N·z products A_a / [B_a\|B'_a] / pad (CPU) | 746.6 / 54.4 / 50.4 | 217.0 / 23.1 / 20.7 |
| — opening E(r) | 7.4 (1%) | 6.2 (0%) |
| **residual N·z products** | 801.0 CPU (67%) | 240.1 (8%) |
| **residual all-in** (products, field entries, matrix setup) | 1,137.0 (95%) | 1,890.0 (60%) |
| **recomputed per verify** | 750.7 (63%) | 2,667.5 (85%) |

**Across workloads (Whiteout; clear Akita in the first row):**

| | muldiv (T = 2^12) | fib 2^16 | fib 2^20 |
|---|---|---|---|
| clear verify, 16 / 1 threads | 9.8 / 14.0 | 10.7 / 14.6 | 40.2 / 23.1 |
| Whiteout verify, 16 / 1 threads | 1,439 / 4,162 | 1,550 / 3,966 | 1,200 / 3,142 |
| Whiteout first verify, transported | 45.9 s | 62.5 s | 32.0 s |
| 1 thread: recomputed | 84% | 85% | 85% |
| 1 thread: residual products | 4% | 5% | 8% |
| 1 thread: residual all-in | 42% | 50% | 60% |
| 1 thread: response proof | 48% | 52% | 52% |
| 1 thread: Σ setup + response lookup | 42% | 41% | 40% |
| load (1-minute) | 94 | 133 | 116 |

The clear Akita control's stages 1–7 take 2.6–5.3 ms, and Whiteout's
`verify_stages_1_to_7` takes 4.0–7.1 ms. The masked stages cost about the
same as the clear ones.

## Thread scaling

Under load, 16 threads beat 1 thread by 2.6–2.9× overall. Per bucket, at
fib 2^20:

- **Matrix setup: 5–8×.** The residual NTT runs 348 → 67 ms and the
  field entries 1,208 → 155 ms. Both are already parallel.
- **Dual and lifted rows: none.** They are flat or slower with 16 threads
  (145 → 241 ms). These loops are serial or contended.
- **Residual products: none.** The per-block parallelism nests a parallel
  `combine_columns` inside parallel blocks. On a loaded machine, 16 threads
  spend more CPU (801 ms) than 1 thread spends wall time (240 ms).

Scaling on a loaded machine is not trustworthy. Rerun on an idle one.

## Instrumentation

All spans compile in the verifier features (`zk`, jolt-zk `whiteout`), not
in `zk-prover`. They are plain `tracing` spans: without a subscriber, each
costs one callsite-interest check.

- **jolt-verifier:** `whiteout_verify_{stages,prefix,relation}`,
  `whiteout_pad_layout`, and `verify_stages_1_to_7`. The last is shared
  with the clear Akita verifier, as the comparison bucket.
- **jolt-akita:**
  - the pad receipt and the tail: `whiteout_receive_pads` and
    `whiteout_verify_tail`;
  - the per-verify setup: `whiteout_sigma_setup`, `whiteout_residual_matrices`,
    `whiteout_v_matrix_setup` and `whiteout_pad_commitment_setup`;
  - the lookup, lift and Σ check: `whiteout_response_lookup`,
    `whiteout_lift` and `whiteout_verify_sigma`.
- **jolt-zk:** `A(r)` by item (`whiteout_weight_{dual,v_image,nz,lifted_rows}`),
  and N·z per block (`whiteout_nz_field_entries` and `whiteout_nz_product`,
  with `block` = `residual_a` | `residual_b` | `pad`).
- **Akita:** `AkitaVerifier::{verify,count}_masked_prefix`,
  `masked_prefix_level` (with `level`), `AkitaResponseProver::verify`, and
  `response_{product_sumcheck,row_batch,digit_range,fused_sumcheck,weight_at}`.
  The opening is the existing `AkitaVerifier::batched_verify`.

The labels are in `taxonomy::WHITEOUT_VERIFIER_SPANS`. `profiling_smoke`
asserts that each one fires under `akita,zk`. The FS inventory is unchanged
(`fs_obligations` passes without a bless).

## Benchmark (frozen)

**Machine.** Apple M4 Max, 16 cores, 128 GiB, macOS 26.6.2, shared.

**Code.**
- jolt: `wo/o0-profile`. The binaries were built at `e00e5c48a`, on
  whiteout/dev `8482916e5`; later commits touch only the script and this
  note.
- Akita: `wo/akita-o0` at `2808ff5b6`, on `299567ec6`. The `[pin]` commit
  `399a14c00` points the jolt patch at `~/akita-wo-o0`.

**Build**, from the jolt worktree root:

```bash
CARGO_PROFILE_RELEASE_DEBUG=0 cargo build --release -p jolt-prover --features profiling,akita,zk
cp target/release/jolt-prover bench-bin/jolt-prover-akita-zk
CARGO_PROFILE_RELEASE_DEBUG=0 cargo build --release -p jolt-prover --features profiling,akita
cp target/release/jolt-prover bench-bin/jolt-prover-akita
```

Dropping debuginfo saves disk; it does not change codegen.

**Per workload**, the harness command, the same as V6's (`<build>` is
`akita-zk` or `akita`):

```bash
bench-bin/jolt-prover-<build> profile --name muldiv    --scale 16 --format chrome --backend optimized
bench-bin/jolt-prover-<build> profile --name fibonacci --scale 16 --format chrome --backend optimized
bench-bin/jolt-prover-<build> profile --name fibonacci --scale 20 --format chrome --backend optimized
```

Each run reports the verify time three ways:

- steady, with 16 threads;
- steady, with 1 thread (whole verify on a one-thread pool);
- first verify on bincode-transported verifier preprocessing.

**The O0 benchmark.** Check `uptime` first: the 1-minute load must be
below 2. Then run:

```bash
python3 -I scripts/whiteout_verify_profile.py run \
  --clear-bin bench-bin/jolt-prover-akita --zk-bin bench-bin/jolt-prover-akita-zk \
  --runs 7 --out bench-out/<label>
python3 -I scripts/whiteout_verify_profile.py report bench-out/<label>/runs.jsonl [--all-spans]
```

**What the script does.**

- It runs 7 rounds over the three workloads. The clear and Whiteout builds
  are interleaved, and the one that goes first alternates by round.
- It records the 1-minute load before every run.
- It splits each `trace.json` by verify mode (`profile_verifier_*` window).
  It keys each span by label within its nearest context (the pad-layout
  replay, the response lookup, the response proof).
- `report` prints the median [min–max] of each bucket above.
