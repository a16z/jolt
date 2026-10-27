# Whiteout prover memory (M1)

Parent: `specs/whiteout-optimization-campaign.md`; baseline
`specs/whiteout-v6-costs.md` §Results item 3. Changes: jolt `wo/m1-memory`
(jolt-zk, jolt-akita, jolt-prover, jolt-kernels) and Akita `wo/akita-m1`
(akita-pcs `zk/`).

## Result

Peak memory of `jolt-prover profile --backend optimized --format chrome`
built with `--features profiling,allocative,akita,zk`, from
`/usr/bin/time -l`. Two interleaved runs per cell against a build of the
base commits (jolt `9e571d2e9`, Akita `b86b1f3ed`), load average 6–12:

| workload | peak RSS before → after | peak footprint before → after |
|---|---|---|
| muldiv (T = 2^12, `e` 2^24) | 2.91–2.93 → 2.15 GiB (−27%) | 2.56–2.58 → 1.80 GiB (−30%) |
| fib 2^16 (`e` 2^25) | 4.90 → 3.37 GiB (−31%) | 4.30 → 2.77 GiB (−36%) |
| fib 2^20 (`e` 2^26) | 8.04–8.06 → 5.13–5.14 GiB (−36%) | 7.40–7.41 → 4.48–4.49 GiB (−39%) |

`--format none` runs agree within 0.02 GiB. After rebasing onto post-O5 dev
(jolt `2cd1d6a02`, Akita `234b46afd`), `--format none`, two runs each,
load average 9–18:

| workload | peak RSS before → after | peak footprint before → after |
|---|---|---|
| muldiv | 2.93–2.94 → 2.18–2.20 GiB | 2.59–2.60 → 1.83–1.85 GiB |
| fib 2^16 | 4.88–4.91 → 3.44 GiB | 4.24–4.31 → 2.84–2.87 GiB |
| fib 2^20 | 7.97 → 5.27–5.28 GiB | 7.32–7.33 → 4.62–4.63 GiB |

Proof bytes are unchanged. With every randomness source seeded (a scratch
patch: the prover's `OsRng` and Akita's `HidingRng::os`), base and M1
produce byte-identical proofs on all three workloads (SHA-256 prefixes
`533dda22…`, `6a756336…`, `c60fe840…`; 93,309 / 95,579 / 97,933 B), and
each build reproduces its own bytes across runs.

## How it was measured

`allocative` does not reach the Whiteout tail. Its snapshots visit only
the stage-1–7 kernel sessions. The attribution below therefore used three
scratch instruments, none of them checked in:

- a counting global allocator in the profile binary. It is sampled every
  20 ms into the Chrome trace as `live_gib` (exact live heap bytes) and
  `livepeak_gib` (the exact maximum since the last sample);
- a dump of every live allocation of 16 MiB or more, with its backtrace,
  at four points of the tail;
- spans around the response proof's six steps.

Footprint is the kernel's `phys_footprint`, sampled by the profile
monitor. The RSS and footprint peaks come from `/usr/bin/time -l`.

## Attribution, fib 2^20

Live heap (exact, GiB) at each step of the tail. `e` has 39.5 M live
digits, padded to 2^26, so one `F` table of that length is 1 GiB.

| step | before: live peak / end | after: live peak / end |
|---|---|---|
| response plan (first use) | 1.15 / 0.80 | 1.01 / 0.75 |
| masked prefix | 1.22 / 1.22 | 1.18 / 1.07 |
| Σ (lift, attempt, witness), at the response proof's start | 1.79 | 1.65 |
| response: commit `e` | 3.82 / 2.85 | 2.61 / 2.61 |
| response: row batch | **5.03** / 4.03 | 3.69 / 3.61 |
| response: digit range | 4.38 / 4.23 | **3.96** / 3.82 |
| response: fused sumcheck | 4.60 / 2.91 | 3.82 / 3.57 |
| response: opening | 2.94 / 2.93 | 3.57 / 2.64 |
| peak footprint (monitor) | 7.41 | 4.47 |

The live peak drops 1.07 GiB (5.03 → 3.96). The footprint peak drops
2.94 GiB, because most of the old gap between footprint and live heap was
freed tables that macOS's allocator kept resident (see "Held during
verification" below). muldiv's live peak after the change is 1.42 GiB
(footprint 1.77 GiB), also at the digit range.

What each step held before, from the allocation dumps:

- **Commit.** `e` lifted to `F` as a wiped `SecretTable` (1 GiB), then
  copied into the committed `DensePoly` (1 GiB). The table was freed after
  the copy. The polynomial is held to the end, because the opening reads
  it.
- **Row batch.** `ResponseStatement::weigh` returned `A` with capacity
  `len`. Appending the padding weights reallocated it to twice `len`
  (1.18 GiB) while the old buffer (0.59 GiB) and a separate padding vector
  (0.44 GiB) were live.
- **Digit range.** The range session's tables (~0.2 GiB) stayed live
  through the fused sumcheck and the opening.
- **Fused sumcheck.** Each round allocated new `A` and `E` tables and
  freed the old ones: 2× the first `A` at round one, then a stream of
  frees.
- **Through the whole response proof.** The Σ attempt's `y`, `z` and shift
  buffers (3 × 29.5 MiB) stayed live. So did a second copy of the
  catalog's public setup matrix (147 MiB): `WhiteoutCatalog::build`
  generated one for the verifier and dropped the prover setup, and the
  first prove generated it again.

At the new peak (the digit range), the live heap holds:

- the committed polynomial (1 GiB) and `A` (1 GiB);
- the range session (0.35 GiB) and `e`'s digits (64 MiB);
- public caches that persist across proofs: Akita commitment NTT slots
  (0.22 GiB), the Σ matrices in `F` (0.22 GiB) and CRT+NTT (0.33 GiB), the
  pad matrix (0.06 + 0.10 GiB), the catalog setup (0.15 GiB) and the
  response setup (0.03 GiB);
- the trace witness and stage state.

## What changed

1. **Commit (Akita `zk/mod.rs`).** `e`'s field form is collected into a
   `Vec` and moved into the committed `DensePoly` instead of being copied
   out of a `SecretTable`. The backend already freed the polynomial
   unwiped. When the padded length is at least 1024 there is now one
   unwiped copy, where before there were a wiped and an unwiped one.
   Below 1024, `DensePoly::from_field_evals` pads its buffer to 1024 by
   reallocating, which frees one more unwiped copy; that resize existed
   before (it padded the copy) and is inherited. Every other `e`-derived table is still
   wiped, and the comment states the exception.
2. **Row batch.** `ResponseStatement::weigh` (jolt-zk) takes a capacity.
   Akita passes the padded length, and `PaddingTerm::extend_weights`
   appends the padding weights in place, with no intermediate vector.
3. **Digit range and fused sumcheck.** The range session and the fused
   kernel are dropped as soon as their outputs are read. The fused kernel
   keeps `A` and `E` in `A`'s buffer and folds both in place
   (`fold_in_place`). `A` occupies `[0, len)`. `E` occupies the upper half
   once the first challenge binds `e`'s digits.
   - **Why in-place folding is safe in parallel.** Outputs `[s, 2s)` read
     only `[2s, 4s)`. Blocks `s = 1, 2, 4, …` therefore run in order, each
     in parallel, and every input is read before it is overwritten.
   - **Memory.** The kernel allocates nothing after it is built: one `A`
     (1 GiB) for the whole sumcheck, against 2× at round one before.
   - **Wiping.** The buffer holds `E`, so it is a `SecretTable` and is
     wiped when freed.
   - **The sparse part of `B`** reuses its buffer between rounds the same
     way.
   - **Values are unchanged.** Folds compute `low + r·(high − low)` via
     `Fold::fold_one` on the same operands. `e`'s digits still enter `F`
     through `lift_ct`. Blocks under 4096 entries fold serially, as in
     `fold_evals_in_place`.

   An intermediate version ping-ponged between buffers and shrank `A`
   with `Vec::shrink_to` before reusing it. It lowered the live peak but
   grew the footprint by 0.5 GiB per proof at `e` = 2^25 when proving in
   a loop (Akita's `zk_response` example), because on macOS a large
   in-place shrink leaves the block unreusable. The in-place version is
   stable: about +0.03 GiB per proof, against +0.05 GiB for the base.
4. **Σ (jolt-zk).** `prove_sigma` drops the attempt's buffers, which
   wipes them, before the response proof.
5. **Catalog setup (jolt-akita).** A prover keeps the setup the verifier
   setup was derived from (they share the matrix `Arc`) and builds only
   the backend on first use. `zk`-only builds still drop the setup. A
   `zk-prover` build keeps the setup for the catalog's lifetime even when
   it only verifies; it shares the matrix with the verifier setup, so the
   cost is the prefix-slot registry.
6. **After the tail (jolt-prover).** `release_retained_memory` runs after
   `whiteout_prove_tail`. jolt-kernels makes it public for this call. See
   the next section.

## "Held during verification" was the allocator

V6 saw the response proof's memory still held while the harness
verified. The prover was not holding it: live heap after the tail is
1.46 GiB at fib 2^20.

- **Cause.** macOS's allocator keeps freed large blocks resident, and the
  stage-boundary purge (`purge_retained_memory`) only runs from
  T = 2^22.
- **Fix.** The response proof's tables are sized by `e` (2^24–2^26 even
  at T = 2^12), so the tail now releases unconditionally.
- **Effect.** fib 2^20's footprint is 1.6 GiB when the prove returns and
  while the harness verifies. Before, it stayed at 7.3 GiB.
- **Scope.** This is a macOS effect. glibc maps blocks this large and
  unmaps them on free.

The same retention explains why the footprint exceeds the live peak.
Allocations made after a large free do not reuse the freed block. The
in-place fused kernel removes the largest source of such frees.

## What is left, and why

At fib 2^20 the live peak is 3.96 GiB and the footprint peak 4.47 GiB.

- **Committed polynomial as `F`: 1 GiB.** `SignedBlockPoly::i8` would
  hold the same digits in 64 MiB. Akita's parity test
  (`tests/backend_ownership/signed_block_poly.rs`) shows identical
  commitment and proof bytes for it. It is not done here because the
  signed-block path scans the values for their extrema and branches per
  value (`AcceptedReach::centered`) to size its reach check. Whether
  that is acceptable for secret digits is a constant-time question for
  A45. The dense path also derives its reach from the values
  (`centered_reach_of_field_coeffs`), so the ruling covers both paths.
- **`A`: 1 GiB.** The fused sumcheck reads `A` densely. The verifier
  evaluates `A` factored (`batch_weight_at`). A prover that streams `A`
  from the factored rows would remove the table, but that is a kernel
  rewrite.
- **Digit-range session: 0.35 GiB.** This is Akita's `DigitRangeProver`,
  which copies the digits into its own packed and class tables. It is
  the step that now sets the peak.
- **Allocator retention: ~0.5 GiB at the peak, macOS only.** It comes
  from frees before the fused sumcheck: Σ and masked-prefix temporaries,
  commit internals in Akita's backend, and the range session. Releasing
  at those points would put allocator calls inside Akita.
- **Public matrices in both forms.** The prover reads the `F` form in
  `combine_columns` (the lift) and the CRT+NTT form in the constant-time
  products. Both forms are cached across proofs (Σ: 0.22 + 0.33 GiB;
  pads: 0.06 + 0.10 GiB). O3's lazily built NTT form adds nothing for a
  prover, which needs both forms anyway, and a verifier holds only `F`.
- **Planner memo (O2).** The response plan peaks at about 1.0 GiB live
  on first use and settles at 0.75 GiB. That is below the proof's peak,
  and O1 persists the plans, so it is not on the steady path.

## Prover time

Same interleaved runs as the Result table (load average 6–12). Each cell
gives the span totals of the two runs:

| span | muldiv | fib 2^16 | fib 2^20 |
|---|---|---|---|
| `whiteout_prove_tail` | 1.10 / 1.12 → 1.06 / 1.11 s | 1.54 / 1.56 → 1.51 / 1.51 s | 2.45 / 2.44 → 2.32 / 2.31 s |
| `whiteout_prove_response` | 0.80 / 0.80 → 0.75 / 0.78 s | 1.10 / 1.10 → 1.06 / 1.06 s | 1.76 / 1.76 → 1.64 / 1.63 s |
| `jolt_prover::prove` | 3.55 / 3.75 → 3.54 / 3.66 s | 5.86 / 6.11 → 5.85 / 6.03 s | 6.50 / 6.55 → 6.36 / 6.30 s |

Akita's `zk_response` example at `e` = 2^25 (scale 2000) measures the
response prover alone, in-process, as 3 interleaved processes × 7 proofs
(load average 3–7). The median goes from 0.822 s to 0.737 s (−10%). Its
live peak per proof goes from 2.03 to 1.48 GiB.

There is no regression; the tail is 2–5% faster, from the copies that
are gone.

## Checks

Byte identity: see Result (seeded scratch build, post-O5 base). All
checks below ran on the rebased trees.

- **jolt.** The bar passed on the final tree:
  - clippy `-D warnings` on the workspace (`host`, `host,zk`); jolt-prover
    and jolt-verifier with `akita,zk,prover-fixtures`, the `logging`,
    `field-inline` and `profiling` variants, and `akita` alone; jolt-akita
    with `zk` and `zk-prover`;
  - `cargo fmt --check` and `check_style_invariants.py --base
    whiteout/dev`;
  - nextest: `fs_obligations` with no bless, `whiteout_e2e` (6/6, and the
    field-inline variant 1/1), jolt-verifier
    `akita,zk,prover-fixtures,logging` (88/88), the field-inline census and
    `fs_attacks` (11/11), jolt-akita (89/89), jolt-akita `zk-prover`
    whiteout (5/5), `profiling_smoke` under `akita,zk,profiling`, the
    clear and BlindFold prover-fixture suites, and jolt-crypto, jolt-zk,
    jolt-dit, jolt-sumcheck and jolt-transcript with `whiteout-prover`
    (345/345).
- **Akita.** clippy (`parallel,disk-persistence`, `+dev-protocol`, the
  zk/test-support lane, the A46 akita-pcs and akita-cpu-backend
  `zk`/`zk-prover` lanes, the `zk_response` example), fmt, taplo; CI
  nextest 1919/1919; proof-dump identity equal to post-O5 dev's dump (the
  same 5 known ZK-row omissions against the old base, no new entries);
  no-disk-cache zk lanes 188/188 and 225/225, zk integration 37/37,
  akita-cpu-backend `zk-prover` 549/549; zk tree clean. The planner
  feature lanes were not rerun; M1 does not touch `akita-planner`.

## Not verified

- Linux memory. The retention analysis is macOS-specific, and the live
  heap figures are platform-independent.
- T = 2^24, the paper's headline shape. The tables scale with `e`, so the
  savings should scale with it, but this was not run.
- The constant-time behaviour of the in-place fold was not re-measured
  with dudect. The fold reads public indices only, as before.
