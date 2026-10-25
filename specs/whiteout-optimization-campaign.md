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
  (`find_zk_schedule`) 1.3–2.2 s. Both are derived state skipped by serde.

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

## Not verified

- **Verifier time split:** unmeasured until O0. Every verifier-side estimate
  above is Akita's harness or §5.5.2's scaling estimate, not a Jolt
  measurement.
- **O4's ν growth and O2's planning speedup:** estimates.
