# Akita Metal end-to-end prover polish

## Decision

Optimize the composed Akita/Metal prover, not isolated kernels. The hard milestone is
at least **5x complete `jolt_prover::prove` speedup** over the optimized CPU backend
for BTreeMap, Fibonacci, and SHA-2 chain at `T = 2^28`, maximizing the worst ratio
first. Once all three clear 5x with credible margin, continue while a bounded,
non-invasive candidate has at least 0.5 seconds or 1% of plausible T28 upside.

Every per-proof cost is charged: witness adaptation, hybrid CPU work, allocation,
transfers, synchronization, readback, and proof assembly. Every scored proof must
verify, silent fallback is forbidden, and peak RSS must not exceed 90 GiB. Public,
witness-independent preprocessing may be excluded only when it is genuinely reusable
across proofs.

## Accepted state and gap

Worktrees:

- Jolt: `feat/akita-metal` at
  `/Users/mgeorghiades/worktrees/jolt/bright-ridge/jolt`;
- Akita: `perf/metal-commit-eval-proof` at
  `/Users/mgeorghiades/worktrees/akita-metal-eval-proof`.

The accepted runtime parent is Jolt `6ec86d08a77d2210676c4f299d55cf7f0ab46892`
and Akita `8291c2dbcd75f413e9697b7cb7ff89942a0c9005`, with runtime tree IDs
`c109b3e925f58fe0e9553eca0a17439280cd02c8` and
`58523a7b0546b540c7636248a31906074ae1e136`. Jolt's documentation head is
`5cd8417ddd24c5597853dd78ba10c436d7394cf8`; its full tree ID is
`84388625f9e8699da824c1bbd07de57d37f5667c`. Audit revisions, tree IDs, and
runtime diffs before resuming. Jolt's modified `Cargo.lock` and untracked
`.cargo/config.toml` are intentional local Akita path overrides; do not commit or
remove them. Do not push.

Both worktrees have accepted runtime sources: Jolt has no diff under
`crates/jolt-kernels/src`, and Akita has no runtime diff. Akita's untracked
commit-analysis specs are historical evidence, not active runtime changes. The
current release evaluator has SHA-256
`f73bd8322232b2fc1f99b0dd17ef572e71705fc27d6ef0e239c14b5f38868e7f`; it contains
the rejected C12 candidate and must not be scored. Compilation is not part of an
experiment's time gate.

| Workload | Optimized CPU | Last credible Metal | Speedup | 5x target | Remaining gap |
|---|---:|---:|---:|---:|---:|
| BTreeMap | 166.548 s | 46.99 s | 3.544x | 33.310 s | 13.680 s |
| Fibonacci | 215.177 s | 45.719 s | 4.71x | 43.035 s | 2.684 s |
| SHA-2 chain | 213.703 s | 42.452 s | 5.03x | 42.741 s | clears by 0.289 s |

BTreeMap is the post-S1 score. Fibonacci and SHA-2 are the last credible results
from the preceding accepted parent; do not spend a matrix run merely to refresh
them. Remeasure them at the next material milestone.

The accepted post-S1 BTreeMap trace is 47.389 seconds: 14.149 seconds commit,
27.211 seconds PIOP, 6.001 seconds eval proof, and about 0.028 seconds other work.
S1 overlaps independent host and Metal Stage-6b members, cutting that stage from
7.043 to 4.994 seconds; its untraced confirmation is 46.99 seconds at 80.07 GiB.
Deleting all commit time would still leave roughly 32.84 seconds, leaving only
0.47 seconds of margin against the 5x target. The remaining plan therefore needs
both a major commit gain and non-commit critical-path savings.

## Main plan

Execute the campaign in this order:

1. Audit the evaluator and preserve the frozen controls, intentional path overrides,
   accepted parent, and append-only negative evidence.
2. Keep the measured dead ends closed. These include P2/P2b pair-major Booleanity,
   S7 compact-record RAM state, Stage-6b shared-row fusion, and the apparent
   0.826-second integrated-eval tax. The same-revision eval difference is only about
   87 ms. Reopen a family only when a new mechanism deletes work rather than changing
   its route or representation.
3. Keep C12 closed. Its exact balanced radix-`2^26` path improved the T25 root by
   11.45% but measured 1.3047 seconds against the fixed 1.25-second gate. It was
   reverted without a repeat or T28 run; same-state radix representations need an
   additional matrix-traffic or charged-owner deletion argument before reopening.
4. Rank the remaining non-commit owners only by complete critical-path saving: a
   table-major Stage-4/5 owner that removes the suffix traversal, a Stage-1 fused owner
   that deletes a scan or residency payment, and eval work that deletes a source
   scan, relation, or synchronization dependency. Each implemented tranche must
   predict at least 0.5 seconds; the portfolio should expose about 16.4 seconds of
   gross non-overlapping upside before broad implementation. If that cannot be done
   within the non-invasive boundary, write the smallest protocol delta that changes
   the bound instead of resuming local candidate grinding.
5. At material milestones, validate all three T28 workloads, then calibrate T25--T28
   and lower-scale crossovers. Once the hard gate is met, remove research scaffolding
   and continue only on bounded, obvious remaining inefficiencies.

The detailed evidence and admission rules for each step follow.

### 1. Freeze the evaluator and establish a feasible disjoint budget

Keep the CPU controls frozen unless the CPU implementation, protocol, workload,
compiler, machine, flags, or timed boundary changes. Preserve one accepted paired
parent and an append-only ignored ledger under
`benchmark-runs/akita-metal-e2e-polish/`.

Do not rerun a baseline unless a candidate needs a fresh paired control. The current
binary contains rejected C12 code and is invalid for scoring; rebuild only after the
next candidate passes correctness. Update the BTreeMap model from the accepted S1
trace and existing counters. The measured gap is 13.68 seconds; use about 16.4
seconds of gross, non-overlapping modeled upside as the planning target for noise and
interaction.

The currently quantified local ceilings do not yet constitute such a path. Even
optimistically adding 4.736 seconds from a traffic-perfect commit root with unchanged
arithmetic, 1.590 seconds from the previously isolated address-segmented sequence
preparation, 1.486 seconds of Stage-4 excess over its CPU reference, the entire
2.725-second Booleanity member, and the 0.824-second Stage-1 primer window yields only
about 11.36 seconds. The earlier 0.826-second integrated-eval term was a cross-revision
comparison and receives no credit. Several remaining terms are
unattainable in full and may overlap; S6b later showed that dense RAM preconstruction
does not realize its isolated ceiling end to end. The campaign therefore needs at
least one structural mechanism that reduces commit arithmetic or deletes additional
Stage-1, Stage-2, or Stage-6 work; a sequence of local scheduling tweaks cannot
establish 5x.

Maintain a disjoint ceiling ledger and a short ranked queue. Reject a candidate
before code when its complete-prover ceiling is below 0.5 seconds. Do not begin a
broad implementation tranche until the ledger contains a credible path to the
remaining target. Take a new Perfetto trace only when existing telemetry cannot
identify the owner.

### 2. Redesign the Akita commit root structurally

The accepted commit takes 14.149 seconds, including about 12.51 seconds in the D512
root. It uses one task per SIMDgroup, two coefficient bands, sixteen position
partials, and about forty persistent 32-bit accumulator values per lane. At T28 it
performs about 1.019 trillion fp128 coefficient additions. Modeled traffic is
1,810 GiB, with a 4.39-second bandwidth floor; the calibrated arithmetic term is
about 7.77 seconds.

The 7.77-second arithmetic term already exceeds the traffic floor. Perfect reuse
without changing that arithmetic can save at most about 4.736 seconds, which is
insufficient for the composed target alongside the bounded local queue. The next
commit design must reduce the number or attainable cost of coefficient additions,
change the matrix/partial ownership tradeoff, or use a bounded public schedule or
configuration change. Matrix-major and hierarchical designs must charge their full
partial or atomic traffic and memory rather than crediting matrix reuse alone.

The tested local variants are closed: two live tasks per SIMDgroup, wider carry-free
digits, interleaved carry chains, sign-quadrant specialization, and same-state
carry-save accumulation. A sequential second task wave is not an escape from the
two-task result: retaining one extra task wave costs 160 KiB of state across the 32
SIMDgroups, or 320 KiB per tile after spill and reload, to avoid only a 32-KiB matrix
load; threadgroup memory is already fully occupied. Reusing the same registers instead
requires another matrix stream. Do not code another reuse-factor or accumulator
micro-variant without a new work-elimination argument.

That gate selected C12, a five-vector balanced radix-`2^26` representation. It keeps
the accepted forty-u32 per-lane state and canonical 16-byte partial ABI while
predecoding the public matrix into five signed digits. Normalization every 62 selected
rows is exact modulo `2^128 - 0xffffa7f7`; radix `2^27` is dominated because it allows
only 30--31 additions between normalizations. The 25% wider public matrix raises the
modeled movement quotient from 4.388 to 5.474 seconds, while the calibrated hot-add
term falls from 7.774 to about 2.638 seconds. The resulting T28 root prediction is
8.5--9.8 seconds versus 12.510 accepted, or 2.2--3.8 seconds of complete-prover
saving. This is an implementation representation only: the field, transcript, proof,
verifier, planner, and Jolt adapter are unchanged.

C12 passed independent field round-trip tests, normalization-boundary CPU/Metal
parity, the full 41-test Akita Metal suite, and an integrated verified
`muldiv_e2e_akita` proof. Its single verified T25 sentinel measured 1.304687 seconds
of root GPU time against the fixed 1.25-second gate. That is an 11.45% improvement
over the accepted 1.473321 seconds, but proportional scaling predicts an 11.08-second
T28 root rather than the required 10.0 seconds. The wider matrix, extra tile barriers,
normalization, and packing consumed most of the source-level ALU reduction. C12 was
rejected without a repeat or T28 run and exactly reverted. Its trace artifact is
`/private/tmp/akita-btreemap-25-metal-c12-radix26-reject.json`.

C8 established that command boundaries are not the T28 limiter. Batching five
unchanged 32-task streams cut the T25 root by 17.96%, but cut the T28 root by only
1.82% (12.510 to 12.282 seconds) and regressed the complete traced proof to 49.275
seconds. It was exactly reverted. Do not sweep command-batch sizes; retain the T25
result only as evidence for a later lower-scale crossover.

C9 tested exact five-prime RNS accumulation. The five-prime product has 7.999x exact
capacity margin and the implementation passed focused parity, all Akita Metal tests,
and an integrated verified Jolt proof. It nevertheless expanded matrix, partial, and
scratch traffic by 25%, added reduction barriers and centered Garner reconstruction,
and produced a 1.5809-second T25 root versus the accepted 1.4733 seconds (+7.31%). It
failed its 1.25-second pre-registered gate, so no T28 run or tuning sweep was allowed;
the runtime code was exactly reverted. This closes independent modular accumulator
chains unless a later design also eliminates enough other work to pay their overhead.

Selector/frequency aggregation is also closed analytically. Within a root task, the
reduction key includes the output task, A position, row parity/sign, and selector. A
repeated one-hot value at another position selects a different public A row, while a
repeat in another task targets a different output coefficient. The existing
SIMDgroups already share each matrix tile. The exact summand multiplicity under the
required destination key is therefore one, so histogramming equal selectors cannot
reduce the 1.019-trillion-add count; it only adds grouping and reduction traffic.

Public root schedule geometry is now closed analytically. Constrained `nv41` K256
planner queries produced this family:

| Root D | `n_a` | `M` | Blocks | Matrix fields | `n_a * D` |
|---:|---:|---:|---:|---:|---:|
| 512 (accepted) | 1 | 262,144 | 16,384 | 134,217,728 | 512 |
| 256 | 2 | 524,288 | 16,384 | 268,435,456 | 512 |
| 128 | 3 | 524,288 | 32,768 | 201,326,592 | 384 |
| 64 | 6 | 1,048,576 | 32,768 | 402,653,184 | 384 |

D256 preserves the root arithmetic product while doubling the matrix. D128 reduces
coefficient arithmetic by at most 25%, but doubles the base streams and raises the
logical traffic proxy by 50%; charging barriers gives a calibrated 12.7--13.3-second
root prediction versus the accepted 12.51 seconds. D64 raises the traffic proxy by
about 3x. The proof remains 76,138 bytes, the root successor shrinks by only about
4.6%, and recursive rank changes do not recover the root loss. No schedule cleared
the 2-second complete-prover admission gate, so no runtime edit or performance run was
made. Do not reopen this family unless a change to the planner algebra, rather than a
new parameter point, changes these invariants.

This leaves the same hard constraint on future commit work: it must delete
coefficient additions or remove another charged owner. Re-encoding accumulators,
changing dispatch geometry, or trading inner dimension for rank is insufficient.
A bounded public protocol/configuration change remains in scope only after its full
root, recursion, eval-proof, proof-size, RSS, and verifier costs establish a disjoint
complete-prover path.

Generic fp128 kernels, matrix residency, and commit scheduling live in Akita.
Jolt owns workload geometry, adapters, and cross-stage orchestration.

### 3. Reopen Stage-2 RAM only with compact or final ownership

S4 retained Stage-1 access chunks and reduced source iterations from `2T` to `2R`,
but its record route took 44.3 ms at T25 against a 40-ms gate; the permitted
diagnostic repeat rose to 58.4 ms. The record census itself was 27.1 ms. Destination
first-touch and the initial-memory/application tail accounted for the missing time,
so S4 moved work instead of deleting it and was reverted without a T28 run.

S6 and S6b then tested schedule-only preconstruction using the unchanged dense RAM
source. S6b successfully started the address-segmented sequence during Stage 1 and
completed it before the Stage-2 join: Stage-2 wall fell by 2.748 seconds and the join
cost only 54 microseconds. The complete proof, however, improved by only 8
milliseconds. Stage 0 grew by 0.386 seconds, Stage 1 by 0.870 seconds, Stage 4 by
0.920 seconds, and smaller costs appeared later. Peak RSS rose from about 80.08 to
82.68 GiB, and swap grew by about 186 MiB. The candidate therefore fails both the
0.5-second promotion gate and the no-swapping invariant. It was exactly reverted.

This establishes the mechanism: the overlap is real, but co-residency of the roughly
5-GiB dense source and 7.63-GiB sequence merely displaces the memory cost. Do not
retry a different start time or another dense prebuild. Reopen RAM only with a compact
source or a final-destination owner that prevents those representations from
coexisting and removes a scan or first-touch pass.

S7 tested compact-record lazy first-cycle state. It retained the existing chunked
24-byte active-access records, released the dense pre/post arrays before resident
sequence allocation, and scattered raw 20-byte records into both address and cycle
order. It did not concatenate the chunks into a 1.46-GiB flat vector or build a
0.24-GiB permutation. The initial message derived `value = pre`, `ra = 1`, Hamming
`= 1`, and `increment = post - pre` directly from raw state. The first challenge
materialized the accepted fp128 steady-state layouts; all later rounds were unchanged.

The first bind does not halve the dominant state: 99.97% of address entries and
78.54% of cycle entries remain live after it. S7 therefore claims no allocation-size
win. Its mechanism is removal of 48 bytes per active access from each of sequence
preparation, the initial message, and the first bind: about 8.743 GiB of aggregate
early traffic at `R = 65,195,206`, plus about 2.543 GiB less source residency. The
counter-derived prediction is 0.28--0.48 seconds for preparation, 0.19--0.25 seconds
for the first two GPU operations, and 0.65--1.10 seconds of complete-proof saving.

The fixed T25 gates were preparation at most 45 ms, first-two GPU time at most 40 ms,
a verified proof at most 6.50 seconds, and no swapping. The T28 gates would have been
preparation at most 0.70 seconds, first-two GPU time at most 0.23 seconds, a verified
traced proof at most 46.89 seconds, RSS at most 90 GiB, and no swap growth. Messages,
challenges, variable order, claims, transcript, proof, verifier, and soundness were
unchanged.

The first sentinel exposed a Stage-1 ownership miss; after the route fix, the next
sentinel exposed premature destruction of the Stage-4 compatibility certificate.
Both failures gained exact route-level regression tests. The corrected candidate then
produced a valid, verified T25 proof through `metal_address_segmented_compact_v1`:
6.110903 seconds complete proving, 59.278 ms preparation, 31.584 ms first-two GPU
time, 16.81 GiB peak RSS, zero process swaps, and no system-swap growth. Against the
accepted 6.15-second observation it saved only 39.1 ms, or 0.64%, while missing the
45-ms preparation gate. The 59.3-ms result also reproduces S4's 58.4-ms record-route
diagnostic, showing that record census plus dual scatter rearranges rather than
deletes enough work. S7 was rejected without a T28 run and exactly reverted. Its
valid artifact is `/private/tmp/akita-btreemap-25-metal-s7-compact-reject.json`,
SHA-256 `a13ef911d6f30b66dfbc10e13ae1400a5a2959540d3080e0c0aa56ffa46e1677`.

### 4. Attack the remaining exposed owners in disjoint order

After each retained treatment, recompute the complete critical path and rank these
owners by expected saving per turnaround time:

1. **Standalone Booleanity structure.** The accepted member is 2.725 seconds, but
   early Bytecode/Booleanity shared-row fusion is closed: at most 8 GiB of common
   traffic, a 19-ms copy floor, and a deliberately loose 0.280-second observational
   ceiling. Seven earlier local shader variants are also closed. The live structural
   question is whether each Instruction-RA, Bytecode-RA, RAM-RA, and unsigned-increment
   Booleanity term can be absorbed into an existing relation that already scans the
   same polynomial. Before code, produce a coverage table containing the exact
   Booleanity identity, existing owner, variable order and evaluation point,
   transcript-derived batching coefficient, degree/round delta, added reads and
   arithmetic, required openings, and synchronized prover/verifier change. Random
   batching must retain the existing soundness error; packed witness representation
   is not a soundness argument. Full absorption has a plausible 1.8--2.4-second
   complete-proof ceiling after added relation work. Implement only if the exact map
   supports that mechanism and predicts at least 1.5 seconds; admit a partial family
   only if it independently clears 0.5 seconds and does not obstruct full absorption.

   The exact map now rejects this mechanism under the campaign's minor-protocol-change
   boundary. After the address variables are bound, an honest one-hot multilinear
   extension is generally not Boolean: for the one-bit column with values `(1, 0)`,
   `P(r) = 1 - r` and `P(1/2)^2 - P(1/2) = -1/4`. The standalone cycle phase is
   complete only because its input is the intermediate produced by the preceding
   Booleanity address sumcheck. It therefore cannot be replaced by checking an
   owner's independently address-folded table.

   All Stage-6b members already share the cycle point, but their address points do
   not alias. At the production bytecode geometry (`log_k = 14`, chunk width 8), let
   the shared Stage-6a batch challenges be `c0..c13`. The tail-aligned Booleanity
   member opens at `(c13..c6)`, while the two padded bytecode chunks are
   `(0, 0, c13..c8)` and `(c7..c0)`; neither is equal. Instruction and RAM chunks
   come from independent Stage-5 points. Head-aligning Booleanity would alias only
   the second bytecode chunk, at most one of the 30 checked columns. Making all owner
   chunks alias would collapse independent address randomness; splitting and moving
   the address reductions into many owner batches is a multi-stage protocol rewrite,
   not a minor schedule change. The modeled partial ceiling is below 0.5 seconds, so
   do not implement owner-point absorption in this campaign.

   P2 was a protocol-preserving kernel candidate. It targeted only branch
   widths 1, 2, and 4, which account for about 1.856 seconds of the accepted
   2.700-second Booleanity round time at T28. Each SIMDgroup covers eight cycle pairs
   and four polynomials; a 256-thread group covers 64 pairs, stages the four physical
   source words for each row in at most 16 KiB of threadgroup memory, groups branch
   gathers by polynomial, and preserves the existing two-lane messages, dense state,
   transcript, proof, and verifier. The per-eight-pair step proxy falls from roughly
   `72 + 16w` to `42 + 16w` operations at branch width `w`. The pre-registered
   prediction is 1.18--1.35 seconds for the first three rounds, or 0.51--0.68 seconds
   of complete-proof saving.

   P2 was correctness-qualified but not performance-accepted. Its initial mapping
   discarded the pair subtotals in SIMD lanes 1--7 because the inherited finalizer
   consumes only lane zero. A final SIMD fold over the eight pair lanes fixed the
   deterministic round failure. The intended width-1/2/4 route now passes the route
   invariant, exact resident-row parity against the optimized CPU oracle, and the
   integrated `muldiv_e2e_akita` verified-proof guard. The width-1 path also retains
   the registered exact `(h_1 - h_0)^2` replacement for the large leading-pair table.

   Full P2 passed T25 but failed T28: width 1 rose from 0.761 to 2.042 seconds,
   total Booleanity reached 3.567 seconds, and the verified proof took 50.12 seconds.
   The one permitted split, P2b, restored width 1 and retained pair-major widths 2
   and 4. Those changed rounds repeatedly saved 0.565--0.628 seconds, but unchanged
   round 0 and later rounds displaced enough work that total Booleanity was 2.318 and
   2.529 seconds in its two allowed T28 observations, missing the 2.20-second gate
   both times. The second proof cleared the complete gate at 46.29 seconds, but both
   gates were required. P2b was exactly reverted. Do not reopen pair-major
   Booleanity without a new work-elimination mechanism rather than another route or
   tile split.

   The fixed performance gates were: at T25, Booleanity at most 0.255 seconds and the
   first three rounds at most 0.170 seconds; at T28, Booleanity at most 2.20 seconds
   and a verified traced BTreeMap proof at most 46.89 seconds. A mixed result permits
   one diagnostic split, not a shape sweep. Keep P2 only if it clears the mechanism
   and complete-proof gates; otherwise exactly revert its three runtime files and
   route test while preserving the negative evidence.

2. **Stage-4/5 ownership and schedule.** Stage 4 is 5.409 seconds versus the 3.923-second
   CPU reference. Private grouped storage and direct cycle-major consumption are
   closed; the latter added 2.29 seconds of T28 random gathers. Admit only a table-major
   owner, a relocation that improves the combined Stage-4 plus Stage-5 critical path,
   or a design that deletes the suffix traversal.
3. **Stage-1 Product/Instruction/Shift chain.** S3 made Product output 1.476 seconds
   faster locally but made the complete proof 0.703 seconds slower by moving a shared
   44-GiB residency payment to later consumers. Standalone CPU openings are closed.
   The late primer has only a 0.824-second contention-free ceiling. Reopen only with a
   fused owner that deletes a scan or useful work that hides the residency payment.
4. **Eval cross-phase work elimination.** The earlier 0.826-second integrated-eval
   estimate compared the accepted integrated runtime with an older isolated runtime.
   At the same accepted Akita runtime (`a454c7575` and `8291c2db` have identical
   runtime sources), isolated O1 opening is 5.809657 seconds and integrated opening is
   5.896453 seconds. Integrated trace coefficient packing is 0.410 seconds slower, but
   ring-relation folding is about 0.332 seconds faster; the net integration-only
   ceiling is roughly 87 ms. Close integration-only tuning. Reopen eval only with a
   mechanism that removes a source scan, relation, synchronization dependency, or
   protocol work from the full 5.896-second span, and require at least 0.5 seconds of
   modeled complete-proof saving.
5. **Workload-specific islands.** Generalizing SHA-2's narrow bytecode carrier can
   remove up to the observed 1.34-second CPU island, but it follows work on the worst
   ratio unless SHA-2 loses its 5x margin.

The accepted two-lane Stage-6b scheduler leaves only about 9 ms of host work exposed;
do not count hidden host members again. The lazy Bytecode width-8 path is closed
because it improved T25 but regressed T28 Bytecode and Stage 6b.

Prefer mechanisms that help all three workloads and create margin for Fibonacci and
SHA-2. Route only on public geometry or activity, never workload names. Hybrid
execution is allowed when fully timed.

Minor public schedule, batching, or layout changes are in scope only when a written
ceiling shows that prover-only work cannot reach the target. They must preserve the
statement and soundness, update prover and verifier together, be independently
revertible, and be recorded in
[akita-metal-protocol-changes.md](akita-metal-protocol-changes.md). Invasive protocol
changes remain out of scope.

### 5. Validate milestones, calibrate crossovers, and clean up

Run the full T28 workload matrix only when the model predicts a material change to
the worst ratio. Once all three appear above 5x, run two order-reversed CPU/Metal
pairs per workload and score the worse valid ratio. Then run verified Metal guards at
T25--T28, fit CPU/Metal crossovers from public geometry/activity, and test T20 plus
the scales bracketing every threshold.

Before handoff, remove rejected variants, search-only switches, obsolete telemetry,
and raw artifacts. Keep the generic Metal backend and protocol-facing changes
reviewable as separate logical diffs. Run formatting, focused exact-parity tests,
relevant nextest suites, and both required clippy modes. Document remaining caveats
and anything not verified.

## Fast candidate loop

For each candidate:

1. State one mechanism, exact boundary, lower bound, predicted complete-prover
   saving, and numerical falsifier before code.
2. Add the smallest red exactness or parity test, then one scoped edit.
3. Run focused correctness and normally one warm T25 affected-workload sentinel.
4. Promote to one T28 run only when affected-span telemetry supports the mechanism.
5. Keep, exactly revert, or mark invalid. Repeat once only for threshold ambiguity,
   a surprising result, or final promotion, not to hunt a favorable sample.
6. Update the accepted parent, latency model, and negative-evidence ledger.

A routine execution gate is at most 120 seconds, excluding compilation. Do not run
repeated CPU controls, broad matrices, Criterion, or Perfetto during ordinary
iterations. Fail closed on wrong output, verifier failure, missing metrics, evaluator
drift, unexplained fallback, swapping, non-finite timing, or unrelated source edits.

Closed paths and their evidence live in
[akita-metal-high-activity-ram.md](akita-metal-high-activity-ram.md),
[akita-metal-stage4-stage5-prefetch.md](akita-metal-stage4-stage5-prefetch.md),
[akita-metal-perfetto-t28-analysis.md](akita-metal-perfetto-t28-analysis.md), and the
ignored experiment ledger. Do not retry a closed mechanism without a materially new
ownership or work-elimination argument.

## Fixed evaluator

Build:

```bash
cargo build --release -p jolt-prover --example modular_benchmark \
  --features prover-fixtures,metal
```

Score the reported `jolt_prover::prove` wall time and require
`PROOF_VERIFIED ... value=true`:

```bash
./target/release/examples/modular_benchmark \
  --name fibonacci --scale 28 --backend {optimized|metal}
./target/release/examples/modular_benchmark \
  --name sha2-chain --scale 28 --backend {optimized|metal}
./target/release/examples/modular_benchmark \
  --name btreemap --scale 28 --target-trace-size 150000000 \
  --backend {optimized|metal}
```

The BTreeMap trace-size override is part of the workload identity. Record proof
verification, route/fallback counters, affected-span telemetry, and peak RSS.

## Completion gate

Do not claim 5x or finish the goal until all of these hold:

- two order-reversed CPU/Metal pairs for each T28 workload, with the worse valid
  ratio above 5x and enough margin for observed noise;
- exact verification, no undocumented fallback, no swapping, and RSS at most 90 GiB
  for every scored run;
- verified Metal guards for all workloads at T25--T28 and tests bracketing every
  fitted lower-scale crossover, including T20 when practical;
- a production diff without rejected implementations, search machinery, raw logs, or
  obsolete instrumentation; and
- focused parity tests, formatting, relevant nextest suites, and both required clippy
  modes, with any pre-existing blocker separated from candidate diagnostics.

## Copy/paste goal prompt

```text
Create a persistent goal with this objective:

Execute specs/akita-metal-e2e-polish-goal.md through its completion gate: achieve and
credibly validate at least 5x complete jolt_prover::prove speedup over optimized CPU
for BTreeMap, Fibonacci, and SHA-2 chain at T=2^28, maximize the worst valid ratio,
preserve useful lower-scale behavior, productionize the retained Metal path, and then
continue only on bounded non-invasive candidates with at least 0.5 seconds or 1% of
plausible T28 upside.

Before acting, read the entire specification, both repositories' AGENTS.md files, and
the append-only campaign ledger. Treat the specification as canonical when this
prompt and live workspace state differ. Its accepted parent, numerical models,
closed paths, evaluator, candidate gates, and completion criteria are binding.
Continue across goal turns until the completion gate is actually met or a real
external blocker satisfies goal mode's blocked threshold. One fast component or
workload is not completion.

Work only in these existing worktrees:

- Jolt: /Users/mgeorghiades/worktrees/jolt/bright-ridge/jolt,
  branch feat/akita-metal
- Akita: /Users/mgeorghiades/worktrees/akita-metal-eval-proof,
  branch perf/metal-commit-eval-proof

Do not push. Preserve Jolt's intentional Cargo.lock and .cargo/config.toml path
overrides without committing them, and preserve unrelated user changes. Follow each
repository's AGENTS.md, use cargo nextest rather than cargo test, and use apply_patch
for source edits. Compilation may take as long as needed and is excluded from the
120-second routine execution gate.

First audit both heads, tree IDs, diffs, path overrides, evaluator binary hash, and
ledger against the specification. C12 is closed: its exact path improved the T25 root
by 11.45% but missed the fixed 1.25-second gate, received no T28 run, and was exactly
reverted. The current release binary still contains C12 and must not be scored. Do
not reopen same-state radix accumulation without an additional work-deletion
mechanism.

Resume by rebuilding the disjoint BTreeMap ceiling ledger and ranking the remaining
owners: a table-major Stage-4/5 design that deletes the suffix traversal, a Stage-1
fused owner that deletes a scan or residency payment, or eval work that deletes a
source scan, relation, or synchronization dependency. Admit only a candidate with at
least 0.5 seconds of disjoint complete-prover upside and a credible portfolio path to
the remaining target. If no non-invasive portfolio exists, write the smallest minor
protocol delta that changes the bound before implementing it.

For later candidates, keep the loop lean and sequential: one work-elimination
mechanism, an exact boundary and lower bound, a disjoint complete-prover prediction,
one numerical falsifier, the smallest red correctness test, one scoped edit, focused
parity, and normally one T25 sentinel. Promote to T28 only when the affected span
supports the model. Do not rerun frozen CPU controls, broad workload matrices,
Criterion, or Perfetto during ordinary iterations. Repeat a measurement only for a
threshold ambiguity, a surprising result, or final promotion. After each result,
keep or exactly revert the candidate and update the accepted parent, latency model,
ranked queue, and append-only ledger.

Charge all per-proof CPU/GPU work. Every scored proof must verify, use the qualified
Metal route without silent fallback, cause no swap growth, and remain at or below 90
GiB peak RSS. Route only on public geometry or activity. Generic fp128 kernels and
commit scheduling belong in Akita; Jolt owns workload geometry, adapters, PIOP, and
cross-stage orchestration. Minor protocol changes require the written necessity,
soundness, prover/verifier, and documentation checks in the specification. Invasive
protocol changes remain out of scope.

Run the three-workload T28 matrix only at material milestones. Do not claim 5x or
complete the goal until the specification's paired final measurements, verification,
memory, fallback, T25--T28, crossover, cleanup, nextest, formatting, and clippy gates
all pass. Separate pre-existing blockers from candidate failures.
```
