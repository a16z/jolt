# Whiteout verifier: serial cost centers (O3b)

Parent: `specs/whiteout-optimization-campaign.md`. Answers the P0b verdict
(`specs/whiteout-p0b-idle-rerun.md`): after O3, the 16-thread steady verify
was spread over the lifted rows (19–23%), the pad-layout replay (14–16%),
the response row batch (14–16%) and `N·z` (11–19%, O6's). O3b takes the
first three.

O3b changes no message, check, challenge or transcript operation, and adds
no cache. Every change computes the same field values as before with less
work. `N·z` is untouched.

## Where the time went

Profiled with temporary spans and macOS `sample` on fib 2^16, before any
change (16 / 1 threads, ms):

| item | cost | cause |
|---|---:|---|
| lifted rows: `aggregate` | 8.6 / 8.6 | 21.7k label weights (a product of up to 15 challenges each) and B-tree updates for the scalarity rows |
| lifted rows: `linear_parts` | 12.3 / 31.6 | 5,760 dense terms, each a full 64×64 ring product, products evaluated one after another |
| row batch: `aggregate` again | 7.0 / 6.8 | `constant` rebuilt the aggregation `weight_at` builds |
| row batch: rest of `constant` | 4.2 / 4.2 | 3 ring products per product row, to form a term in which the quadratic parts cancel |
| replay and relation: `SourceValues::value` | 28% of the replay's samples | linear scan of the public sources per lookup |
| relation: `AffineRow::add_scaled` | 77% of `append_rows`' samples | a fresh `BTreeMap` per addition, in loops that grow a running claim |

## What changed

1. **One aggregation per statement** (`jolt-zk`, `ResponseStatement::new`).
   The statement holds `AggregatedRelation` under its `ρ`. `constant`,
   `weight_at`, `rows`, `weigh` and `check_witness` read it, instead of
   each calling `LiftedRelation::aggregate`. The aggregation is a function
   of the relation and `ρ`, both fixed when the statement is built. Nothing
   outlives the statement, so nothing proof-derived outlives a verify.
2. **Scalarity weights factored** (`LiftedRelation::aggregate`). Label
   `64o + r` with `r < 64` has disjoint bits from `64o`, so
   `μ_{64o+r} = μ_{64o}·μ_r` exactly (`RowLabels::weight` is a product over
   set bits). 63 low weights are computed once; each digit ring takes one
   slot lookup instead of 63.
3. **Factored dense sums** (`SValues::dense_sum`, `FactoredS` override).
   On the verifier's factored dual, a ring inside one native element `E` is
   `S_E` times the ring `lo[o + i·stride]`, which depends only on
   `(o, stride)`. So `Σ_k σ(ℓ̂_k)·x_k = Σ_{(o,stride)} (Σ_k S_{E_k} σ(ℓ̂_k))·lo_{o,stride}`:
   one scalar multiple per term, one ring product per distinct
   `(o, stride)`. A ring straddling two elements (none does in a setup's
   layout) is multiplied directly. The prover's `[i64]` path keeps the
   default (ring products with the public weights on the receiving side,
   the constant-time WARNING moved with it).
4. **Products in parallel** (`AggregatedRelation::linear_parts`). The
   per-product parts are filled in place into `SecretVec`s, so no unwiped
   intermediate buffer holds a prover's secret parts.
5. **The constant's reduced target in closed form**
   (`AggregatedRelation::target_at_zero`). `reduce(Σ μ_i X_i Y_i) − Q_μ(u, ·)`
   is `u·(Σ μ_i Z_i − λ − g)` for any vector: the identity
   `target_linear` and `product_adjoint` already rest on. At zero,
   `Z_i = u·ẑ_i`, `λ = u·λ̂`, `g = 0`, so `target_at_zero` feeds those to
   `target_linear` instead of forming and cancelling every `X_i·Y_i`.
6. **Public sources in a map** (`jolt-verifier` residual `SourceValues`):
   a `BTreeMap` keyed by `VerifierPublicId` (now `Ord`, as is
   `JoltSpartanOuterPublic`). The inconsistent-value check is unchanged.
7. **Ordered row merge** (`AffineRow::add_scaled`). When both rows' terms
   are strictly ordered by variable, which every row `add_scaled`,
   `masked`, `variable` and `constant` build is, the sum is one linear
   merge. Otherwise the old `BTreeMap` path runs. Both produce the same
   terms in the same order, zero coefficients included.

**Why the proof cannot change.** Items 1, 2, 5 and 6 compute the same
field elements by exact identities; 3 regroups a sum of products in a
commutative ring; 4 and 7 reorder independent work and merge. No change
reads or writes the transcript. The prover runs items 1, 2, 4 and 7 too
(the statement, `linear_parts`, residual rows); the honest proofs of
`whiteout_e2e` and the jolt-verifier fixtures verify, and a seeded-RNG
probe (below) compares proof bytes.

**Caching.** None added. The brief's lever "cache a public-only part of
the replay keyed on the full `PadLayout`" has no target: the replay
*produces* the layout from the proof's seeded prefix (O3, "Not cached"),
and its cost is the stage verifiers, the residual build, Akita's masked
replay and the layout lowering, all of which read challenges. Items 6 and
7 cut the replay's and the relation's lowering instead.

## Results

**Harness.** `scripts/whiteout_verify_profile.py run --base-bin`, the A/B
mode this ticket adds: base (`48b4db315`) and O3b interleaved, the first
build alternating by round, 7 rounds, `--backend optimized --format
chrome`, both built with `CARGO_PROFILE_RELEASE_DEBUG=0 --release -p
jolt-prover --features profiling,akita,zk` from the same Akita
(`6ca22bd57`). Each binary waited for a 1-minute load below 16 (at most
300 s) and recorded its launch load. The machine is shared: launch loads
were 12.7–71.5, median 15–16.5; 4 of 42 launches hit the 300 s cap. Cells
give the median [min–max] in ms. A first pass at load 75–250 is
discarded; it agreed in direction only.

| | muldiv (T = 2^12) | fib 2^16 | fib 2^20 |
|---|---:|---:|---:|
| steady verify, 16 threads, base | 82.3 [81.3–432.9] | 91.7 [89.3–624.2] | 111.4 [105.2–149.3] |
| steady verify, 16 threads, O3b | **47.3** [43.5–123.8] | **54.9** [52.2–333.7] | **73.3** [69.3–80.6] |
| per-round speedup, 16 threads | 1.87× [1.68–8.48] | 1.70× [1.11–1.87] | 1.52× [1.40–1.85] |
| steady verify, 1 thread, base | 148.2 [147.4–181.8] | 186.2 [183.8–234.0] | 284.0 [282.3–308.0] |
| steady verify, 1 thread, O3b | **95.1** [94.3–115.4] | **133.9** [128.5–168.3] | **230.5** [226.2–232.3] |
| per-round speedup, 1 thread | 1.57× [1.54–1.72] | 1.40× [1.36–1.45] | 1.24× [1.23–1.34] |
| proof bytes, base / O3b (median) | 93,365 / 93,323 | 95,566 / 95,586 | 97,914 / 97,913 |

The base column reproduces P0b's idle medians within 2–6 ms.

**By bucket, 16 threads / 1 thread (ms, median), base → O3b:**

| bucket | muldiv | fib 2^16 | fib 2^20 |
|---|---|---|---|
| lifted rows (item 4) | 17.9 / 37.7 → 2.4 / 5.1 | 18.8 / 38.3 → 2.7 / 5.3 | 18.6 / 39.1 → 3.5 / 5.4 |
| row batch | 13.2 / 12.6 → 1.9 / 1.5 | 13.9 / 13.3 → 2.8 / 2.4 | 14.6 / 13.9 → 3.4 / 2.9 |
| pad-layout replay | 13.1 / 14.0 → 8.9 / 9.9 | 13.5 / 15.8 → 9.3 / 11.7 | 15.5 / 20.4 → 11.2 / 16.4 |
| relation (auxiliary block) | 6.3 / 6.3 → 1.6 / 1.6 | 6.9 / 6.9 → 1.7 / 1.7 | 7.5 / 7.6 → 1.8 / 1.8 |
| seed, stages 1–7, stage-8 claims | 7.8 / 7.7 → 5.9 / 5.9 | 7.8 / 8.0 → 6.1 / 5.9 | 7.8 / 7.7 → 6.0 / 6.0 |
| `N·z` (O6, unchanged) | 11.2 / 53.0 → 12.4 / 53.1 | 15.8 / 80.8 → 15.0 / 82.5 | 25.9 / 161.9 → 26.3 / 162.9 |

The aggregation now runs under `verify_sigma` before the response proof,
which is why "Σ check" reads 1.2–1.4 ms instead of 0.0. The 1-thread cut
is 53 ms at every workload; it is a smaller share at fib 2^20 because
`N·z` there is 163 of the remaining 231 ms.

**Target.** ≤ 60 ms on 16 threads holds for muldiv (47.3) and fib 2^16
(54.9), not for fib 2^20 (73.3). There `N·z` is 26.3, the eq dual 7.9 and
the replay 11.2.

**Proof bytes.** Whiteout proofs are randomized (pads, Σ attempts, Akita's
hiding draws), so the table's sizes vary by tens of bytes either way. For
an exact check, a local probe (never committed) replaced both
`OsRng` sources, `prove`'s and Akita's `HidingRng::os`, with one fixed-seed
generator, and hashed `proof.narg` of a muldiv proof. Two runs of each
build agreed, and base and O3b produced the same 93,333-byte argument
string (`DefaultHasher` `d8e330466c5c0dcd`).

## What is left

- **The replay is still serial.** At idle it was 12.5–15 ms on 16
  threads. What remains after O3b is the stage verifiers (with their
  symbolic member formulas), Akita's `count_masked_prefix` (whose
  stage-2 oracle weights wait on Rayon) and the residual build. Overlapping
  it with the real stages needs the layout before the pad commitment is
  read, so only a speculative verify on a guessed layout (re-run on a
  mismatch) would hide it. Not done: the guess would come from earlier
  proofs.
- **Opening `E(r)` and the lifted rows could overlap `A(r)`.** The opening
  reads the transcript after `v`, and `weight_at` does not; a `rayon::join`
  would hide 4–9 ms on 16 threads, nothing on one. Not done: the profile
  harness attributes spans by thread, and the hidden half would leave its
  bucket.
- **`N·z` and the eq dual** (O6) are now the largest items.

## Not verified

- One machine (M4 Max, 16 cores), shared: the gate held the median launch
  load near 16, not the < 2 of P0b. Absolute numbers carry that load. The
  comparison is paired by round.
- The byte-identity probe covers one muldiv proof with one seed, not
  fib 2^16 / 2^20 or a proof with advice.
- Akita's `akita-pcs` `zk::` unit tests were not rerun. No Akita source
  changed, but they call jolt-zk's `constant` and `weight_at`. An honest
  `constant` is exercised end to end by `whiteout_e2e` and the
  jolt-verifier fixtures.
- jolt-akita under `zk-prover`: 93 of 94 pass.
  `scheme::tests::legacy_single_setup_keeps_its_transcript` fails the
  same way on base `48b4db315` (a legacy setup decode, `UnexpectedEnd`).
  The Whiteout lane (`-E test(whiteout)`) passes 5/5.
- `cargo clippy --all` with `akita,…` features fails in `jolt-sdk`
  (`verifier_preprocessing`, `GroupSetupMetadata`). That crate is not
  touched here. The per-crate lanes CI runs pass.
- Constant time: the prover's `[i64]` path of `dense_sum` is the old code.
  `linear_parts` now runs products on Rayon. That changes no branch on
  secrets, but no dudect run checked it.
