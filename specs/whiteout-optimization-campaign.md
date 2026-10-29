# Whiteout optimization campaign: proof size and verifier

Parent: `specs/whiteout-v6-costs.md` (baseline), `specs/whiteout-paper-deltas.md`
(D9, D29), `akita:specs/whiteout-akita.md` §4.1, §5.5.2.

## Decision

Whiteout adds +29–31 KB to a Jolt+Akita proof (paper: +7.3 KB). It also takes
the verifier from 7–10 ms to 185–250 ms (paper: +0.8 ms), and a verifier in a
fresh process spends another 12–14 s re-planning. Prover overhead is
acceptable and is not a target.

The campaign runs in measurement-gated phases:

- **P0, measure.** Profile where the verifier's time goes; nobody has done
  this yet.
- **P1, low-risk wins.** Persist the plans, parallelize the verifier, and
  compress u_a: recover the paper's 128 B.
- **P2, structural.** Move the residual commitment to a larger ring, or lower
  the Σ collision radius. These change parameters the paper argues about.
- **P3, tune.** The response proof and the masked prefix.

Every ticket must leave every clear catalog unchanged (C0) and keep every
estimated SIS occurrence at 128 bits or more.

## Targets

| metric (fib 2^20; muldiv and fib 2^16 must not regress) | baseline | target | stretch |
|---|---|---|---|
| proof Δ over clear Akita | +28.8 KB | ≤ +10 KB | ≤ +7.3 KB (paper) |
| steady verify | 250 ms | ≤ 60 ms | ≤ 25 ms |
| first verify, transported preprocessing | 13.7 s | ≤ 0.5 s | ≈ steady verify |
| steady prover Δ | +2.47 s | no regression > 10% | — |

Non-goal: prover RSS (7.9 GiB at fib 2^20, dominated by the response proof).
Record it, but don't optimize it here.

## Where the cost is (V6)

**Bytes**, Δ ≈ 29–31 KB:

- **Raw u_a: 19.5 KB.** 19 × `R_64` at the Σ-binding rank (D29).
- **Rest of the masked prefix: 5.1 KB.**
- **Response proof minus the clear opening: +2.2 to +5.1 KB.** The response
  proof itself is 70.4–70.8 KB.
- **Pads: 384 B.** That is three 128 B chunks; the paper has 128 B.
- **Auxiliary blocks: 720 B.**
- **Σ: 160 B.** This matches the model.

**Verifier time.**

- **Not split yet.** Akita's own harness puts the growth in the rank-19
  residual matrices `A_a` and `[B_a | B′_a]` inside N (W5e: 4–5×).
- **Single-threaded.** The Whiteout verifier takes the same time with one
  thread as with 16, so its new work runs serially.
- **First-use cost.** It comes from planning, not setup expansion. The
  response plan (`find_schedule`) takes 10.8–11.3 s and the catalog plan
  (`find_zk_schedule`) 1.3–2.2 s. At V6 both were derived state skipped
  by serde; O1 persists them (paper delta D32).

## Tickets

Each ticket gives its expected gain, its risk, and the repository it lands
in.

### P0: measure first

- **O0. Verifier profile and a frozen benchmark.**
  - Add spans to the Whiteout verifier: masked levels, the Σ check, the two
    residual `N·z` products, response-proof verification (per sumcheck and
    opening), and any setup expansion inside `verify`.
  - Freeze one benchmark command per workload. It reuses V6's profile
    harness: release build, an idle machine (1-minute load < 2, recorded),
    and 7 runs, reporting median and range.
  - Every later ticket reports against this.
  - jolt plus Akita, instrumentation only.

### P1: low-risk, independent

- **O1. Persist the plans.** Expected: first verify 13.7 s → under 0.5 s;
  prover first use −12 s.
  - Serialize the catalog rows and the response schedule with the verifier
    preprocessing, or ship them as versioned artifacts.
  - On load, the verifier re-runs `ZkOpeningSchedule::admit` and the
    Σ-radius checks on the decoded plan. A plan is a public parameter that
    soundness depends on, so the verifier validates it rather than trusting
    it.
  - Test: tampered plan bytes are rejected, and a round-trip verify gives
    the same result.
  - jolt, plus Akita for plan serde.
  - **Outcome (`wo/o1-plans`, paper delta D32).** jolt only; Akita's
    trusted-artifact decoder already audits and admits rows. A fresh setup
    plans its catalog at construction and persists it; exported plans
    (`AkitaVerifierSetup::whiteout_plans`, response schedules included)
    enter a new prover or verifier setup only at construction
    (`preprocess_full_from_plans`), and a setup holding a response set
    refuses a length it lacks. The pads absorb the masked catalog's, the
    selected row's and the response schedule's digests. One profile run
    each on the rebased head (after O4), release, `--backend optimized`;
    the 1-minute load average was 30–36, so absolute times are inflated:

    | | muldiv (2^12) | fib (2^20) |
    |---|---|---|
    | planning in the same run (`whiteout_plan_catalog` at setup + `whiteout_plan_response`) | 1.91 + 11.5 s | 2.29 + 12.0 s |
    | loading and validating (`whiteout_load_catalog` + `whiteout_load_response`) | 0.010 + 0.001 s | 0.018 + 0.002 s |
    | verify, in process (parallel) | 0.160 s | 0.282 s |
    | verify, first use on transported preprocessing | 0.207 s (V6: 12.3 s) | 0.343 s (V6: 13.7 s) |

    The transported verifier is built from the prover's exported plans
    (`preprocess_full_from_plans`), so it plans nothing. Its excess over
    the in-process verify (47 ms and 61 ms) is not attributed here.
- **O2. Faster planning.** Expected: response plan 11–16 s down to about
  1 s.
  - Profile `find_schedule` on the response statement: candidate count and
    pricing cost.
  - Then memoize the pricing, prune dominated candidates, or seed the search
    from the stored plan's neighborhood.
  - This still matters after O1, for preprocessing time and for statements
    whose shape is new.
  - Akita.
- **O3. Parallel, precomputed verifier.** Expected: 2–8× on the share O0
  attributes to N and the response proof.
  - Parallelize the residual `N·z` products and the response-proof verifier's
    inner loops.
  - Hold N in the NTT domain in the verifier setup, not in each verify, if
    O0 shows per-verify conversion.
  - Akita.
- **O4. Compress u_a (paper Item 3).** Expected: proof Δ −19 KB, which is the
  largest size lever.
  - The cut successor sends u_a compressed: 1,216 coefficients, which is 3
    chunks of 128 B under W5c's split chain.
  - Σ's N holds the compressed target, and u_a's compression witness joins
    `e`. That costs at most one extra ν of the response proof, about
    +0.2 KB.
  - Admission must allow a `Compressed` payload at the cut (§4.1 requires
    `Raw` today). The compression of u_a is secret-dependent, so it must
    take A45's constant-time path.
  - This retires D29's deviation from the paper; record the delta change.
  - Akita (cut payload, admission, Σ statement) plus jolt-zk (statement
    blocks).
- **O5. Small byte items.** Expected: up to −0.9 KB.
  - Pads: 3 chunks down to 1. Shrink t_p below the 512-coefficient cap, or
    explain why it can't be.
  - Auxiliary blocks (720 B): audit what they carry and drop or compress any
    redundant part.
  - jolt.
  - **Outcome (`wo/o5-bytes` + Akita `wo/akita-o5`, paper delta D34).**
    −464 B deterministic on every workload; −534 to −575 B in total
    (muldiv, fib 2^16, fib 2^20, two runs; the rest is response-proof
    noise). Whiteout's
    compressed images take the fewest balanced chunks under the chain's
    SIS bound, 884 coefficients (the paper's `FCHAIN_MAX_IMAGE`), instead
    of the 8 KiB clear cap: t_p and cm(u_a) go from 3 chunks to 2 of 608
    (−128 B each). A repeated intermediate product takes one auxiliary
    slot: 45 → 32 slots (−208 B), all in stage 6's bytecode read-RAF
    cycle output; BlindFold committed-program proofs drop 38 → 27. One-chunk
    t_p is impossible: A_p must bind at R_Σ, which needs rank ≥ 18 at
    d = 64 (1,152 coefficients), above the 884 a two-map chain
    certifies. The 2 defining slots stay (−32 B for 38 more operand
    pads); the other slots are irreducible.

### P2: structural (needs P0 numbers, and decisions)

- **O6. Residual commitment at d = 128, then 256.** Expected: residual
  verifier work and setup about 3.5× smaller at d = 256 (Akita §5.5.2
  estimate). Proof bytes are roughly unchanged once O4 lands.
  - Add a zero-knowledge-only dimension domain for the after-cut level and a
    `d_B` ceiling exception there, keeping C0 for every clear catalog.
  - Before lifting the monotone `dimension_ceiling`, write the correctness
    argument that lifting it is safe; none has been found against it.
  - d = 128 needs only blocker (1) at the rows whose level-1 `d_B` is 128;
    land it first.
  - At d = 256 the headline offloads at the cut again, so a masked Stage 3
    runs end to end.
  - Akita (planner, presets); jolt only for re-pinned rows and the census.
- **O7. Lower the Σ collision radius.** Expected: a smaller residual rank,
  hence u_a and N shrink together. It compounds with O6.
  - `R_Σ = 8κ₁·B(D)`, with κ₁ = 53 and D = 10 binding digits. Each digit
    removed divides R_Σ by 16.
  - The levers are D (the Σ regime, `whiteout-sigma.md` §5.1, which trades
    against the rejection rate and soundness) and κ₁ (a D64 challenge family
    with a smaller ℓ1 norm, at equal challenge-space entropy).
  - First a paper-side analysis: how rank moves with log R_Σ in the
    Σ-binding table, and the effect on rejection and knowledge error. Then
    implement only if the gain beats O6's.
  - jolt-zk owns the formula; Akita reads the radius.

### P3: tuning

- **O8. Response proof (+2–5 KB over the clear opening).**
  - Search the response prover's schedule space: basis, digit planes, and
    proof-of-work on the product and fused sumchecks (A34), against bytes.
  - Akita.
- **O9. Masked prefix (5.1 KB).**
  - List the masked levels' messages by size.
  - Candidate: compress the public `OpeningPayload` D image (§4.2), which
    the masked path exchanges in full.
  - Akita.

## Ordering and process

- **O0 first.** Then O1, O3, O4 and O5 run in parallel; they touch disjoint
  owners, except that O1 and O4 both touch the Σ setup's serialized form, so
  O4 rebases on O1.
- **O2** goes in parallel with P1.
- **O6/O7** start after O0 attributes the verifier time and O4 has landed.
  The choice between them, or both, follows from the measured residual
  share.
- **O8/O9** come last.

**Per-ticket gate** (the existing merge bars):

- Akita: the full bar, with the dump identity showing exactly the 5 known
  ZK-row omissions and no new entries.
- jolt: the full bar, with whiteout_e2e passing and the census and FS
  inventory blessed with the diff reviewed.
- `catalog_security --check` at 128 bits or more for every SIS occurrence.
- A measured result on the O0 benchmark (all three workloads).
- A paper-delta entry for any change to what the protocol sends or checks
  (O4, O6, O7).

Measurement claims go through result-validation: interleaved runs, an idle
machine, and the spread reported.

## Decisions (owner, 2026-10-25)

The owner delegated the calls to the conductor, on two conditions: every
security argument must hold, and every change must be documented as a
paper delta.

- **O6:** d = 128 first. d = 256 follows if O0 attributes at least 50% of
  verifier time to the residual.
- **O7:** analysis is done (`specs/whiteout-o7-sigma-radius.md`).
  - **Dropped:** the κ₁ lever (worth at most 0.24 bits) and the
    extraction-radius lever (worth 0.09 bits).
  - **Queued: P9** (D = 9, α = 2^13) on top of O6. It cuts the residual by
    another 1.4× at d = 256 and the A_p chunks from 3 to 2, at the price of
    one rerun per about 595 proofs.
  - **Before implementing:**
    - certify P9's ranks with a `sigma_binding_width_table` run, since they
      are interpolated today;
    - wait for O0/O6 to show the residual's measured share is worth it.
  - P9 revisits §5.1's review decision (the reviewer chose D = 10 over
    D = 9), so its paper delta (D33; D32 is O1's persisted plans) records the new evidence: the
    Σ-binding ranks make D = 9 worth 24–30% of the residual.
- **O4:** in scope. The ticket writes the argument that Item 3's
  compressed-target check carries over to the cut successor, as a paper
  delta. A question for the paper authors stays open, but it does not block
  the work.

## Debt queued behind the campaign

- **S1. Style debt.** `scripts/check_style_invariants.py --base main` reports 78
  `nominal-imports` violations accumulated on the Whiteout line. CI runs this
  check, so it blocks upstreaming. It is mechanical; run it after O0, O1 and O4
  merge, since they touch the same files. Merge bars check new lines against
  the pre-merge dev head.
  **Resolved (S1):** 72 violations, all `nominal-imports`, fixed as
  import-only path shortening with matching `cfg` gates. Merged at
  3e2a3a48f; `check_style_invariants.py --base main` is clean, and the jolt
  bar is green.
- **S2. Legacy setup bytes under `zk`.** `jolt-akita`'s
  `legacy_single_setup_keeps_its_transcript` fails with `--features
  zk-prover` (`UnexpectedEnd`). Since W6b, a `zk` build's
  `AkitaVerifierSetup` ends with `#[serde(default)] whiteout:
  Option<WhiteoutSetup>`; bincode is not self-describing and ignores
  `serde(default)`, so pre-Whiteout setup bytes no longer decode once any
  crate unifies `zk` on. The wire format therefore depends on a feature.
  Fix before upstreaming: carry the Whiteout setup outside the clear setup's
  encoding (e.g. a wrapper or an appended artifact variant) so clear bytes
  are identical in every build. The merge bars only run that suite's
  `whiteout` subset, which is why they did not catch it.
  **Resolved (S2, `specs/whiteout-s2-setup-encoding.md`):** the Whiteout
  setup is the appended `OneHotWhiteout` artifact variant; the transcript
  absorbs the same values; the legacy test passes with and without
  `zk-prover`. Merged at da0a7aecb; jolt bar green.

## Review log

- **O1 (persisted plans).** Two reviews, no soundness blocker. In the fix round:
  - the masked catalog digest is absorbed at PADS (it was bound only from
    stage 8, though it fixes D and the commitment profiles);
  - plans enter only at construction, with no `adopt`;
  - serialized preprocessing is history-independent;
  - the style lint is fixed.
- **O4 (compressed u_a).** Two reviews, no blocker. The binding, the extractor,
  error terms, hiding and clear-path invariance hold by trace. In the fix
  round:
  - a chain-row-isolating tamper;
  - constant-time `recompose`;
  - a code-level ν ≤ 27 guard (D27);
  - dev-protocol masked-prefix coverage;
  - `BackendCompression` made test-only.

- **O1 merged** after a delta re-review that confirmed every fix. Its
  first-verify target (< 0.5 s) holds when plans are distributed through
  `whiteout_plans()`; plain serde transport still plans responses (D32),
  which O2 inherits.
- **O4 merged.** Combined bars are green; Akita's dump is unchanged apart from
  the 5 known ZK rows.
- **O0 merged** (LOADED pass only; `specs/whiteout-o0-profile.md`). About 85%
  of a one-thread steady verify was public state rebuilt per verify; the
  residual `N·z` products alone were 4–8%. Decision: O3 before O6; O6 d = 256
  and P9 wait for a post-O3 idle rerun.
- **O3 (verifier caches).** One review, no blocker: every cache is a
  deterministic function of public inputs keyed on the full input, nothing
  proof-derived is cached, failed builds are not stored, and the transcript
  did not move. Fix round: taxonomy v7 (a label was removed), profile script
  buckets, the tail owns its row, the layout cache is capped at 8, note
  clauses. Merged; LOADED steady verify 3.5–4.0× (16 threads) and 7–13×
  (1 thread) faster.
- **Intermittent failure after the O3 merge.**
  `whiteout_tamper_per_census_class_with_precommitted_groups` failed once in
  the merge bar; the bar dropped its message. A dedicated investigation found
  no reproduction in about 70 runs and 10,700 single-bit tampers (every
  message of three fresh proofs included), no unabsorbed or non-canonical
  byte, and no way for an O3 cache to skip a per-verify check. Root cause
  unknown; the bar now keeps each lane's full log so a recurrence names the
  accepted tamper or the panic site.
- **O2 (parallel planner).** One review, no planner blocker: memo values are
  functions of their key, errors are never memoized, and the original
  traversal makes every decision. Fix round: `ResponsePlans::get` plans
  outside its lock (a held lock across the now-parallel planner could
  deadlock concurrent verifies), claim bookkeeping gated to `parallel`, and
  the note records the extra states early completion explores and the stack
  headroom. Merged; bars green, dump identity exactly the 5 known rows.
  LOADED: response planning 3.5–7× faster; the ~1 s target is not met.
- **M1 (prover memory)** opened: peak RSS 0.60 → 7.94 GiB at fib 2^20, the
  response proof dominant.
- **O5 (compression and shared products, D34).** One review; every item was
  fixed. Compressed images are chunked at the chain's SIS bound and
  balanced (1216 → 608 + 608), lifting the first map to 141 bits under a new
  policy tag; `ProductMemo` shares repeated product slots (aux block 45 → 32
  slots). The fixed saving is −464 B/proof (pads −128, cut −128, aux −208),
  and totals are 93,325 / 95,535 / 97,883 B for muldiv / fib 2^16 / fib 2^20.
  The checked-in ZK fuzz bundles were already broken at base and were
  regenerated. Merged; combined bars are green, and Akita's dump identity is
  exactly the 5 known rows.
- **M1 (prover memory; `specs/whiteout-m1-memory.md`).** One review, no
  blocker: wiping holds on every path including unwind, the in-place fused
  fold is parallel-safe (outputs `[s, 2s)` read only `[2s, 4s)`, blocks in
  order) and field-identical, and nothing transcript-visible moved. Fix
  round: the backend takes the prover setup's matrix, the allocator release
  follows `drop(stage8)`, and the spec qualifies the single-unwiped-copy
  claim (padded length ≥ 1024). Seeded proofs are byte-identical to post-O5
  dev. Peak RSS, LOADED: 2.93 → 2.19, 4.9 → 3.44, 7.97 → 5.28 GiB for
  muldiv / fib 2^16 / fib 2^20. Merged; combined bars green, dump identity
  exactly the 5 known rows. Left: the committed `e` as `i8` (1 GiB → 64 MiB,
  needs an A45 constant-time ruling) and streaming `A` (1 GiB, a kernel
  rewrite).
- **P0b (idle rerun; `specs/whiteout-p0b-idle-rerun.md`).** Idle steady
  verify 80 / 88 / 105 ms (16 threads); the LOADED 0.53 s was load. The
  residual is 29 / 37 / 53% of 1-thread verify, so O6 runs at d = 128 and
  d = 256 (with P9) waits for a post-O6 rerun.
- **O3b (verifier serial paths; `specs/whiteout-o3b-verifier.md`).** One
  review, no blocker: every rewrite is an exact identity for all inputs
  (the row-batch constant's cancellation, `μ_{64o+r} = μ_{64o}·μ_r`, a
  sorted merge that falls back when inputs are not strictly sorted,
  first-wins public sources); the parallel `linear_parts` writes only into
  wiped buffers; no transcript change. Fix round: a straddling-ring test,
  the `target_at_zero` owner note. A/B (load ~16): 16-thread steady verify
  82 → 47, 92 → 55, 111 → 73 ms; nothing cached. Merged; jolt bars green.
- **O2b (response planning; `specs/whiteout-o2b-planner.md`).** One review,
  no blocker: prunes are strict against a sound lower bound, ties break on
  the full schedule descriptor, no lock is held across parallel work, and
  errors are never memoized, so the chosen plan is byte-identical in every
  visit order *whenever the search returns a plan*. Errors raised after a
  prune decision (`child_choice`, `PendingQueryEdge::new`, a successor's
  suffix derivation) may depend on visit order; by inspection they are
  unreachable for well-formed input, and the doc and spec say so. Fix
  round: the split replay is keyed on the request's opening, the
  live-length test covers chunked levels, and the memo-pressure `<=` is
  explained. LOADED response plan 13.6 → 2.1, 23.4 → 7.6, 14.2 → 2.1 s;
  plans and `artifacts/schedules` are unchanged. Merged (Akita 8c22267ed);
  Akita and jolt bars green, and the proof dump equals O2b's verified dump.
- **O6 (residual at d = 128; `specs/whiteout-o6-d128.md`, D35).** An
  independent security review found no blocker. The ceiling exemption is
  reachable only by a ZK search at `cut + 1`, the ring is bound through
  the row and catalog digests, and the SIS cells were re-estimated (A_p
  steps to rank 10 at 1,232 columns, 127.995 bits at rank 9). The
  LHL margin stays 512 bits, and the cut level's Stage-2 round count is
  unchanged for L ≥ 128, so D27 is unchanged. Its N3 coverage probe then
  found that the priced search offloaded into the cut level from 35
  variables, so the 36:1 headline's `e` needed ν = 29 > 27 (fail-closed
  `ResponseTooLarge`). Fix: the priced ZK search keeps the edge into the
  cut level direct, and `every_checked_in_zk_row_fits_the_response_bound`
  guards every ZK row through jolt-zk's own layout. The final 36:1 row
  proves, verifies and rejects a tamper (verify 75 ms, 10.1 GiB). A/B,
  load < 6: 1-thread verify −15 / −19 / −25%, 16-thread −4 / −4 / −9%,
  proof bytes flat. fib 2^20's residual share is now 40% < 50%, so
  d = 256 and P9 do not run. Merged (Akita 9c6c7273f, jolt 2d68df80c);
  bars green, dump has exactly the 5 known omissions.
- **M2 (prover memory; `specs/whiteout-m2-memory.md`).** The committed `e`
  is held as i8 digits and the fused sumcheck streams its row weights.
  Peak RSS 1.72 → 1.24, 2.75 → 1.78, 4.47 → 2.59 GiB (muldiv / fib 2^16 /
  fib 2^20); prover time within ±3% at load 8–20. An independent
  constant-time review found no blocker: every lift is `lift_ct`, the
  extrema branches are constant over every honest `e`, and every new
  digit buffer is wiped. Fix round: the scratch-growth and constructor
  error paths wipe, `source_coefficients` carries a WARNING, and A45
  lists the inherited `DigitMontLut` digit-indexed read. A runtime probe
  showed it is not reached on M4 (i32 NEON); it is reached on x86_64 at
  D < 64 and with i16 primes, which is unverified. Merged (Akita
  f0de039e2, jolt 3bc688951); bars green, and the dump has the 5 known
  omissions.

## Not verified

- **Verifier time split:** O0's split is LOADED; no idle-machine
  measurement exists yet.
- **The intermittent tamper-test failure above:** unexplained.
- **O4's ν growth:** an estimate.
