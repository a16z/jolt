# Akita Metal committed-symbol 5x goal, revision 4

Status: ready for user edits. The campaign is paused. Do not compile, benchmark, or
edit runtime code until the user launches the prompt at the end of this file.

## Decision

Stop ordinary kernel and unified-memory tuning. S14, S15, and S16 proved that local
traffic reductions can move first-touch and residency costs without improving the
complete proof. S15 nevertheless exposed a real 2.046-second complete-proof win that
the old local-span gate wrongly discarded. Revision 4 retains valid proof-level wins
even when a diagnostic sub-bound fails, then updates the model.

The present protocol has no conservative two-to-four-mechanism path to the target.
The next campaign is therefore a bounded study of one statement-preserving protocol
change: commit balanced radix-4 address digits instead of sparse one-hot extensions.
Raw bit planes and raw bytes are analytically closed. Radix-4 is the only surviving
representation because it can reuse Akita's small-coefficient range proof and replace
one-hot commitment, opening, and Booleanity work together.

Do not implement the production protocol immediately. First establish the exact
deletable Jolt owners and a conservative T28 roofline. If that analysis cannot cover
the measured BTreeMap gap plus one second, stop without code. If the interval straddles
the target, write one deciding prototype for the missing constants. Production work
is admitted only after that gate.

Minor changes may alter the public committed layout, batching, proof bytes, or a
domain-separated protocol version when prover and verifier change together and the
soundness target is preserved. Pause before changing the Jolt statement, witness
language, memory-consistency argument, security assumption, or independence of
unrelated Fiat-Shamir challenges.

## Frozen objective and parent

The score is the worst valid T28 ratio across the three workloads:

| Workload | Frozen optimized CPU | Accepted Metal | Speedup | 5x ceiling |
|---|---:|---:|---:|---:|
| BTreeMap | 166.548 s | 46.990 s | 3.544x | 33.310 s |
| Fibonacci | 215.177 s | 45.719 s | 4.706x | 43.035 s |
| SHA-2 chain | 213.703 s | 42.452 s | 5.034x | 42.741 s |

The diagnostic BTreeMap parent is the verified 47.388661-second trace. The
margin-bearing ceiling is 32.309600 seconds, so the campaign must conservatively
remove 15.079061 seconds. Its principal owners are:

| Owner | CPU | Metal | Metal excess over CPU / 5 |
|---|---:|---:|---:|
| Commit | 65.767 s | 14.149 s | 0.996 s |
| Stage 1 + 2 | 20.879 s | 12.140 s | 7.964 s |
| Stage 4 | 3.923 s | 5.409 s | 4.624 s |
| Other measured PIOP | 57.381 s | 9.662 s | -1.814 s |
| Eval proof | 16.735 s | 6.001 s | 2.654 s |

The runtime parents remain Jolt
`6ec86d08a77d2210676c4f299d55cf7f0ab46892` and Akita
`8291c2dbcd75f413e9697b7cb7ff89942a0c9005`. Jolt HEAD
`5cd8417ddd24c5597853dd78ba10c436d7394cf8` adds documentation only.
S16 was completed, verified, measured, rejected, and exactly restored. The only
intentional Jolt source delta is the independent benchmark-harness fix that permits
missing Metal opening telemetry only for an explicitly unqualified public shape.
Preserve local Cargo path overrides and unrelated user changes. Existing release
binaries contain rejected candidates and are not scorable.

## Evidence that closes ordinary tuning

- S14 saved 1.460--1.867 seconds but retained a temporary host flatten and later
  dense owners.
- S15 reduced the Product/Instruction terminal boundary from 2.045 to 0.427 seconds
  and verified a 45.342650-second proof, 2.046 seconds faster than the traced parent.
  Its full residual scan had also kept the borrowed Stage-3 destination resident;
  deleting the scan moved 0.810 seconds into Stage 3. Under revision 4 the valid
  complete-proof gain would be retained and the residency model corrected.
- S16 combined that terminal opening with three native InstructionInput binds. It
  deleted 1.758 seconds from Stage 2, but first-writing a 4-GiB frontier into recently
  unread pages added 1.210 seconds to Stage 3 and displaced another 0.794 seconds
  later. The complete proof regressed by 0.028 seconds.
- Moving RAM construction earlier, typed duplicate owners, purge/restore, page
  primers, cold scans, host-seeded register paths, root-radix variants, and local eval
  retuning are closed by the append-only evidence. Reopen one only with a new whole
  owner, traversal, relation, or authenticated-opening deletion.
- Even granting zero time to Stage 1, Stage 2, Stage 4, and eval yields only 15.243
  seconds of generally applicable effective saving, 0.164 seconds beyond the
  required amount. Partial deletion under the current protocol cannot provide the
  one-second margin.

## The one admitted structural candidate

There are 30 semantic one-hot members over a byte-sized address chunk. Represent each
address byte by four two-bit digits. Map unsigned digits `0,1,2,3` to the balanced
alphabet `0,1,-2,-1`, preserving zero, and commit four length-`T` digit
polynomials per member. Keep the 120 polynomials separate; do not pack member or digit
indices into variables across which a nonlinear equality product would be folded.

For a requested digit `a`, its equality selector is the cubic Lagrange polynomial

```text
L_a(c) = product over d != a of (c - d) / (a - d).
```

Byte equality is the product of four such factors. If
`S = c(c + 1)`, every cubic factor has the form

```text
L_a(c) = A_a + B_a c + C_a S + D_a cS.
```

Akita Stage 1 already range-checks the balanced basis-4 alphabet, and Stage 2 already
binds `S`. The smallest extension carries the additional virtual table `cS` and
authenticates the mapped factor. A low-degree product tree combines four factors;
do not use one monolithic degree-13 sumcheck. The changed layout and relation must be
versioned and transcript-domain-separated.

This does not change the Jolt statement or memory-consistency argument. It changes the
committed witness representation, the Booleanity/Hamming reduction used for those
one-hot members, the Akita mapped-opening relation, proof shape, and verifier
equations. The old one-hot protocol remains a supported config and fail-closed
fallback.

## Current analytical bounds

The exact T28 BTreeMap source has 253,779,321 populated rows, 2,023,057,407 current
sparse hot entries, and an estimated 6,238,628,051 one bits. The average populated
row has 7.971 nonzero byte lanes and 24.583 set bits.

Raw bit planes are closed: sparse commitment would require about 3.194 trillion
D512 ring additions, 3.13 times the current root work, while a straightforward
post-bind product frontier exceeds the 90-GiB cap. Raw bytes are closed because exact
byte equality is degree 255 unless another lookup argument is added; Akita's existing
quadratic range image is not a generic 256-entry lookup.

For grouped radix-4 at T28, the deterministic planner gives approximately:

- 120 compact digit polynomials, 30 GiB of i8 source;
- 62,914,560 D512 root rings;
- 256 MiB of setup fields and about 1.96 GiB of root-successor fields;
- roughly 724.8 billion five-prime NTT butterflies and 161 billion pointwise
  products; and
- about 600 GiB of public-matrix traffic without cross-block reuse, a 1.455-second
  floor at the calibrated 412.5 GiB/s.

Sparse radix-4 root commitment is also closed: it has at least half as many nonzero
digits as source bit ones, already more than the current hot-entry count, and needs
small scalar multiplies. A dense NTT root is the only credible route.

The mapped-factor/product proof has a separate arithmetic floor. Four map terms and
three product nodes cost about seven fp128 multiplications per logical member-row:
56.37 billion multiplications at T28, or 3.445 seconds at the measured
16.36-billion/s ideal throughput before folding, traffic, recursion, and transcript
cost. Materializing four fp128 factors for every member after the first bind would
take about 240 GiB, so the implementation must perform several compact-prefix cycle
binds before materializing. Three binds reduce that frontier to roughly 60 GiB, but
the complete live-set lifetime still has to be proven under 90 GiB.

Commit plus eval alone allow at most 5.071 seconds of replacement work at the
margin-bearing target, which the analytical floors already rule out. Feasibility
depends on deleting the corresponding Booleanity work and its Hamming reduction.
The accepted Stage-6b Booleanity member is 2.725 seconds; Stage 6a and Stage 7 contain
other members, so no credit may be taken for either complete stage. The exact
Booleanity-address and Hamming-reduction shares must be isolated before admission.
S15's observed 2.046-second Metal-only gain may be a separate portfolio tranche, but
it is not present in source and may not be counted until its overlap with the new
representation is proven and the result is revalidated.

## Phase 1: one bounded feasibility pass

Before runtime edits, append one candidate card that resolves only these
decision-changing quantities:

1. Enumerate exactly which committed members migrate and which Stage-6a,
   Stage-6b, Stage-7, Stage-8, commitment, and opening owners disappear. Do not
   credit whole stages that retain unrelated members.
2. Give an allocation-to-retirement table for the 30-GiB compact source, NTT
   buffers, root successor, compact-prefix factor state, materialized frontier, and
   every old owner removed. Prove a schedule at or below 90 GiB with no overlapping
   source/destination command.
3. Bound dense-root NTT compute, matrix traffic/reuse, digit extraction, mapped
   factor construction, product-tree rounds, recursive Akita opening, readback, and
   synchronization. Do not add ideal floors that execute on the same resource as
   though they overlap.
4. Use the best valid optimized CPU config as the denominator. If the structural
   protocol also improves that baseline, charge
   `Metal saving - CPU saving / 5`. It is valid for CPU to retain the old public
   protocol config if that is faster.
5. Specify the transcript dependency graph, verifier identities, proof-byte delta,
   soundness-error delta, fallback, and protocol/layout version.

Use

```text
required effective saving = 47.388661 - 32.309600 = 15.079061 s
structural effective saving =
    old owners actually deleted
  - complete replacement owner
  - CPU improvement / 5
portfolio effective saving =
    structural effective saving
  + disjoint retained Metal-only wins
```

The conservative portfolio must reach 15.079061 seconds. A point estimate is
insufficient: price compulsory work and use the high-cost end of every unresolved
interval. If the lower bound misses, reject the candidate and stop within the approved
scope. If the interval straddles the threshold, name at most two missing constants and
measure them with one deciding prototype.

## Phase 2: deciding prototype, only if needed

The prototype is analysis infrastructure, not production protocol code. It may
exercise:

1. grouped basis-4 D512 dense-root commitment with the real T28 geometry; and
2. compact-prefix cubic-map plus four-factor product-tree work with the intended
   memory schedule.

Use T20/T25 only for arithmetic parity and route checks. Use at most one T28 boundary
measurement for each missing constant, preferably in one process. Pre-register the
maximum combined replacement time and peak live bytes derived by Phase 1. Reject
immediately if the measured/projection interval crosses that bound. Do not run a full
Jolt proof, workload matrix, CPU control, Criterion suite, or Perfetto trace during
this phase.

Routine non-build commands have a 120-second limit. Compilation time is unrestricted.
Do not poll a running command when no new evidence can exist.

## Phase 3: production protocol, only after admission

Keep protocol ownership at the verifier boundary:

- Akita owns basis-4 digit range semantics, the `cS` mapped-opening relation,
  recursive checks, layout/version binding, and generic CPU/Metal APIs.
- Jolt claims/verifier own the committed radix-4 layout, selector-product relation,
  cross-stage claims, and proof configuration.
- `akita-metal` owns dense root, compact-prefix map/product, and recursive kernels,
  but is not the only specification of the protocol.

Implement prover and verifier together. Add independent tests for digit range,
all four Lagrange maps, product-tree claims, old/new protocol cross-rejection,
transcript order, tampered layout/version, fallback, CPU/Metal proof equality under
one version, and exact verification. Preserve the old one-hot route.

Use the smallest real-proof gate that exercises the changed protocol, then one
BTreeMap T28 treatment. A valid complete-proof improvement of at least 0.50 seconds
becomes the provisional search parent even if a local prediction fails. Repeat only a
result within 0.25 seconds of its retention threshold, a surprising result needed to
repair the model, or a finalist. Restore only rejected candidate paths.

Do not reconstruct S15 merely to obtain a small win. Once the structural mechanism
has a credible path to the target, S15 may be rebuilt as a separate candidate if its
owner set is disjoint; under revision 4 its prior valid 2.046-second result is strong
admission evidence, not an accepted source result.

## Lean campaign rules

For each candidate use one owner, one model, one numerical boundary falsifier, one
complete-proof falsifier, and one scoped implementation. Keep the frozen reference
parent for comparison and the best valid retained result as the search parent.

Hard promotion gates are:

- exact proof verification and focused arithmetic parity;
- qualified Metal route with no silent fallback;
- unchanged statement and security target within the approved scope;
- peak RSS at most 90 GiB, zero process swaps, and no system swap growth;
- immutable evaluator plus recorded binary/artifact hashes; and
- a worthwhile complete-proof improvement.

Local span predictions, queue-overlap estimates, and page-temperature assumptions are
diagnostic. A miss repairs the model; it does not reject an otherwise valid
proof-level winner.

Do not rerun frozen CPU controls, broad workload matrices, Criterion, or Perfetto
during ordinary search. Run the three-workload Metal T28 matrix after a retained
cumulative improvement of at least two seconds, after BTreeMap drops below 40 seconds,
or after a protocol change that can change workload ordering. Keep evidence
append-only in
[`analysis.md`](../benchmark-runs/akita-metal-e2e-polish/analysis.md) and
`events.jsonl`. Do not push.

## Acceptance

Completion requires two order-reversed CPU/Metal T28 pairs for BTreeMap, Fibonacci,
and SHA-2, with the worse valid ratio at least 5x for every workload; exact
verification; qualified Metal routes without fallback; no system swap growth; peak
RSS at most 90 GiB; relevant T25--T28 guards; calibrated lower-scale crossovers;
production cleanup; protocol documentation; focused tests; required nextest suites;
formatting; and both required clippy modes. Component or prototype speedups are not
completion.

After all three workloads clear 5x, continue only while a bounded non-invasive
candidate has at least 0.35 seconds or 1% of plausible T28 upside.

## Feedable Goal Mode prompt

```text
Create a persistent goal and execute revision 4 of:

/Users/mgeorghiades/worktrees/jolt/bright-ridge/jolt/specs/akita-metal-e2e-structural-5x-goal.md

Read the entire specification, both repositories' AGENTS.md files, the protocol-change
ledger, and the append-only analysis/events before acting. Revision 4 is the active
contract and supersedes the earlier e2e prompts.

The hard objective is a verified >=5x complete jolt_prover::prove speedup over the
best optimized CPU Akita prover for BTreeMap, Fibonacci, and SHA-2 at T=2^28, scored
by the worst valid ratio, followed by bounded production cleanup and lower-scale
crossover calibration. Continue across goal turns until every acceptance condition
passes or a real external blocker satisfies Goal Mode's blocked threshold.

Do not resume ordinary kernel or residency tuning. S16 was completed, measured,
rejected, and restored. First run the bounded analytical feasibility pass for the
balanced radix-4 committed-address representation described in the specification.
Audit the exact Jolt owners it removes, model every new allocation and complete
replacement owner, charge CPU effects, and require a conservative portfolio saving
of at least 15.079061 seconds. Do not credit an entire mixed stage or an ideal
overlap. If the conservative bound misses, stop within the approved scope without
production code.

Raw bits, direct bytes, a new memory argument, challenge reordering, and ordinary
one-hot tuning are out of scope. If no more than two constants keep the radix-4
interval from deciding admission, build one minimal analysis-only prototype for the
grouped dense root and compact-prefix mapped product, with focused parity and at most
one T28 boundary measurement per missing constant. No full proof or workload matrix
belongs in that gate.

Only after the structural bound clears, implement the smallest versioned protocol
slice in Akita and Jolt, updating prover and verifier together. Preserve the old
one-hot config, the Jolt statement, the memory-consistency argument, the soundness
target, and unrelated Fiat-Shamir independence. Domain-separate the new layout and
relation, add independent tamper/cross-rejection/parity tests, and fail closed on
unsupported routes.

Keep the loop lean: one owner, one candidate card, one cheap falsifier, focused
correctness, and normally one T28 treatment. Compilation is unrestricted; routine
non-build commands have a 120-second limit. Retain a valid worthwhile end-to-end
winner even if a diagnostic span moves, repair the model, and continue from the best
valid search parent. Do not rerun frozen controls, broad matrices, Criterion, or
Perfetto during ordinary iterations.

Preserve Cargo path overrides, unrelated changes, and append-only evidence. Use cargo
nextest, never cargo test. Do not push. Do not claim completion until the paired
three-workload final matrix, verification, memory, fallback, lower-scale, cleanup,
formatting, and clippy requirements in the specification all pass.
```
