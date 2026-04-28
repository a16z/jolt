# Verifier-Parity Completion Plan

This is the program counter for closing all remaining gaps in
`jolt_verifier::verify` so it is rejection-equivalent to jolt-core's
verifier on the muldiv fixture for all 11 tamper categories
(§3.4 of `verifier_parity_and_soundness.md`).

## Mission

**Target end state**:
- `CheckOutput` authored at every wired stage (currently: 1, 2, 4 done; 3, 5, 6, 7 missing)
- `CollectOpeningClaim`/`At` wired at every committed-poly opening (deferred at 4-7)
- `KNOWN_GAPS` empty
- Soundness suite has explicit tests for T1, T2, T3, T5, T8, T9, T10, T11
- All test gates green

## Stop conditions (hard exits — anything else continues the loop)

1. All tasks below marked `[x]` AND all gates green → **success exit**.
2. Same task fails the gate ≥3 times consecutively → mark `[blocked]`,
   write a 1-paragraph note to "## Notes", continue to next task.
3. ≥3 tasks marked `[blocked]` → halt (spec is broken).
4. External interrupt.

**Never pause between iterations.** After a successful commit, re-read
this file, pick the next unchecked task, and start immediately. No
`ScheduleWakeup`, no sleep loops.

## Pre-decided design choices (no analysis paralysis at runtime)

1. **Repeated `Eval` factors as squaring.** First task tests whether
   `vec![Eval(P), Eval(P)]` evaluates to `eval[P]²` in `evaluate_formula`.
   If yes → use repeated `Eval` for Booleanity / HammingBooleanity.
   If no → add `ClaimFactor::EvalSquared(PolynomialId)` variant.
2. **`CombineEntryEval` design.** Add `ClaimFactor::CombineEntryEval {
   kernel: usize, at_stage: VerifierStageIndex }` for Stage 5
   InstructionReadRaf. Verifier evaluator reads `combine_entries` from
   `Module.prover.kernels[kernel].instance_config` (already serialized)
   and computes the combined val at the stage's address point.
3. **Opening-claim parity.** Add `VerifierOp::CollectOpeningClaimAt {
   poly, point_challenges, committed_num_vars }` mirroring the
   prover-side `Op::CollectOpeningClaimAt`.
4. **CheckOutput is all-or-nothing.** Don't push `VerifierOp::CheckOutput`
   until every instance in the stage has a non-empty `output_check`.

## Order of attack — each is one commit, in order

```
[x] T0   Pre-flight: repeated-Eval composes multiplicatively
[x] T1   Stage 3 CheckOutput — already wired in commit b360246bc
[x] T11  T2 eval-tamper test landed; KGC extended to recognize BothAccept-as-gap;
         T2Eval@stage 7 registered until CheckOutput or CollectOpeningClaim lands
[x] T2   Stage 7 CheckOutput landed; T2@7 gap closed (modular catches via final_eval mismatch)
[ ] T3   Stage 6 simple output_checks (RamRaVirt, InstRaVirt, IncReduction)
         — author formulas only, defer push until T5
[ ] T4   Stage 6 Booleanity + HammingBooleanity output_check
[ ] T5   Stage 6 BytecodeReadRaf output_check + push CheckOutput op
[ ] T6   Stage 5 simple output_checks (RamRaClaim, RegistersValEval)
         — author formulas only, defer push until T8
[ ] T7   Add ClaimFactor::CombineEntryEval variant + verifier arm
[ ] T8   Stage 5 InstructionReadRaf output_check + push CheckOutput op
[x] T9   Add VerifierOp::CollectOpeningClaimAt + ScaleEval + AliasEval handlers
[x] T10  Add build_verifier_stage8_ops mirroring prover-side build_stage8;
         all committed polys collect against unified [r_address_BE, r_cycle_BE]
         opening point. PCS opening verification now active (was no-op).
[x] T12  T9 batch-claim test (last-eval-idx per stage); modular catches all 7
[x] T13  T3 commitment-swap test (swap each Some slot with slot 0;
         all 42 swaps caught by stage-8 PCS verification)
[x] T18  T4 opening-proof truncation test (drop last opening_proof;
         modular returns Opening(VerificationFailed))
[ ] T14  Soundness suite: T10 domain-separator-tag test (deferred —
         requires tag-byte tampering infrastructure)
[x] T15  T5 commit-slot SomeToNone test (zero out each Some slot;
         all 42 caught — AbsorbCommitment skip diverges transcript)
[x] T16  T11 public-IO test (modular rejects via preamble divergence)
[x] T17  T6 config-field test (TraceLength + RamK doubled / +1; modular rejects)
```

## Findings from T11 dry-run (2026-04-28)

**Discovery**: NO `VerifierOp::CollectOpeningClaim` calls exist in the
modular verifier schedule today. The PCS opening proofs are not
verified — `VerifyOpenings` is a no-op (`pcs_claims` always empty).
The existing `modular_self_verify` test passes because it only checks
sumcheck correctness, not PCS openings.

**T2 results (eval-tamper, sp.evals[0] += 1):**
| Stage | Outcome | Why |
|---|---|---|
| 1 | modular rejects | sumcheck error round 0 (transcript divergence into stage 2) |
| 2 | modular rejects | sumcheck error round 0 (transcript divergence into stage 3) |
| 3 | modular rejects | CheckOutput catches it directly |
| 4 | modular rejects | CheckOutput catches it directly |
| 5 | modular rejects | sumcheck error (transcript divergence into stage 6) |
| 6 | modular rejects | sumcheck error (transcript divergence into stage 7) |
| 7 | **BothAccept** | no downstream sumcheck, no PCS verification → tamper invisible |

**Implication**: closing T2 at stage 7 requires either CheckOutput (T2)
*or* CollectOpeningClaim parity (T9/T10). Either path closes the gap;
CheckOutput is smaller in scope.

**Reordering rationale**: T11 (test) is now ahead of T2 (stage-7
CheckOutput) so we can lock in the ratchet baseline before authoring
output formulas, then watch T2@stage-7 flip from `KnownGap` to
`BothReject` when stage-7 CheckOutput lands.

## Per-task acceptance criteria

- **T0**: New unit test in `jolt-verifier` confirming
  `evaluate_formula` with `vec![Eval(P), Eval(P)]` returns `eval[P]²`.
- **T1, T2, T4, T5, T8**: After commit, `cross_verifier_soundness::
  known_gap_registry_consistency` passes — KGC ratchet enforces removal
  of newly-satisfied entries. T2/T7 entries for that stage removable.
- **T3, T6**: No registry change yet (formulas authored but not pushed).
  Test gate is `transcript_divergence + modular_self_verify` green.
- **T7**: New unit test `formula_evaluation_combine_entry_eval` passes.
- **T9**: New unit test `verifier_collect_opening_claim_at` passes.
- **T10**: T3 KnownGap entries for stages 6-7 are removable (in T13).
- **T11–T16**: Soundness suite gains the new test method, all wired
  stages report `BothReject`, unwired stages have explicit `KnownGap`
  entries that the registry-consistency test confirms still reproduce.

## Universal test gate (run after every commit)

```bash
cargo nextest run -p jolt-equivalence --cargo-quiet
RUST_MIN_STACK=536870912 cargo nextest run -p jolt-core muldiv --cargo-quiet --features host
RUST_MIN_STACK=536870912 cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk
cargo clippy -p jolt-core --features host --message-format=short -q --all-targets -- -D warnings
cargo clippy -p jolt-core --features host,zk --message-format=short -q --all-targets -- -D warnings
cargo clippy -p jolt-compiler -p jolt-verifier -p jolt-zkvm --message-format=short -q --all-targets -- -D warnings
```

Pre-existing red in `jolt-equivalence/tests/booleanity_debug.rs` is
grandfathered — don't try to fix.

Failure handling:
- 1st: diagnose, fix, retry.
- 2nd: revert, re-attempt with a different approach.
- 3rd: mark `[blocked]` in checklist, write note in "## Notes", skip.

## Per-instance output_check authorship workflow

1. Open the jolt-core source file (paths below).
2. Find the verifier's `verify` method or `expected_output_claim` helper.
3. Identify the formula: typically `eq(r_a, r_b) × Σ challenge^k × Π evaluation`.
4. Translate factor-by-factor to `ClaimFormula`:
   - `evaluations[poly]` → `ClaimFactor::Eval(poly)`
   - `staged_evaluations[stage][poly]` → `ClaimFactor::StagedEval { poly, stage }`
   - `eq(challenges_a, challenges_b_at_stage_S)` →
     `ClaimFactor::EqEval { challenges: a, at_stage: S }` (or `EqEvalSlice` if partial)
   - `LT(...)` → `ClaimFactor::LtEval { ... }`
   - Preprocessed poly evals → `ClaimFactor::PreprocessedPolyEval { poly, at_stage }`
   - `challenge^k` → k repeated `Challenge(idx)` factors
   - `eval(P)^k` → k repeated `Eval(P)` factors (after T0 confirms)
5. Set `instances[i].output_check = ClaimFormula { terms: ... }`.
6. After ALL instances in the stage have output_check, push
   `VerifierOp::CheckOutput`.

## Source pointers (per instance)

### Stage 3
- Shift: `crates/jolt-core/src/zkvm/spartan/shift.rs`
- InstructionInput: `crates/jolt-core/src/zkvm/spartan/instruction_input.rs`
- RegistersClaimReduction: `crates/jolt-core/src/zkvm/registers/claim_reduction.rs`

### Stage 5
- InstructionReadRaf: `crates/jolt-core/src/zkvm/instruction_lookups/read_raf.rs`
- RamRaClaimReduction: `crates/jolt-core/src/zkvm/ram/ra_reduction.rs`
- RegistersValEvaluation: `crates/jolt-core/src/zkvm/registers/val.rs`

### Stage 6
- BytecodeReadRaf: `crates/jolt-core/src/zkvm/bytecode/read_raf.rs`
- Booleanity: `crates/jolt-core/src/zkvm/instruction_lookups/booleanity.rs`
- HammingBooleanity: `crates/jolt-core/src/zkvm/ram/hamming_booleanity.rs`
- RamRaVirtual: `crates/jolt-core/src/zkvm/ram/ra_virtual.rs`
- InstructionRaVirtual: `crates/jolt-core/src/zkvm/instruction_lookups/ra_virtual.rs`
- IncClaimReduction: `crates/jolt-core/src/zkvm/spartan/inc_reduction.rs`

### Stage 7
- HammingWeightClaimReduction: `crates/jolt-core/src/zkvm/instruction_lookups/hamming_weight.rs`

## Loop protocol

```
1. Read this file — find first task NOT marked [x] or [blocked]
2. Read the relevant jolt-core source / existing modular code
3. Implement the change (single commit scope)
4. Run universal test gate
5. Pass:
   - update this file: mark task [x] (with brief one-line summary)
   - if KGC entries newly satisfied, remove them from registry.rs
     in the SAME commit
   - commit with message `feat(verifier): T<n> <task subject>`
   - GOTO 1
6. Fail (1st): read the failure, diagnose, fix, GOTO 4
7. Fail (2nd): revert, re-attempt differently, GOTO 4
8. Fail (3rd): mark [blocked], write note to "## Notes", GOTO 1
```

## Debugging recipes

### output_check evaluates to wrong value (final_eval mismatch)
Bisect: temporarily set `output_check.terms = vec![]`, rebuild — confirms
rest of pipeline works. Then add terms one at a time. Use
`JOLT_VERIFIER_DEBUG=1` instrumentation in `verifier.rs` CheckOutput
dispatch (see prior session handoff for example).

### Transcript divergence
Run `transcript_divergence` test alone with `--no-capture` — the
divergence point reveals the missing/extra absorb. Most likely cause:
a Squeeze before/after a transcript-affecting op was misplaced.

### `MissingEvaluation` from evaluate_formula
The formula references a poly not yet in `evaluations` map. Either add
a `RecordEvals` for it earlier in the schedule or use
`StagedEval { stage: <earlier> }`.

### KGC test fails with "X known-gap entries no longer reproduce"
The KGC ratchet caught a fix. Remove the listed entries from
`crates/jolt-equivalence/src/cross_verifier/registry.rs` in the same
commit.

## Existing helpers to reference

- `evaluate_formula` in `crates/jolt-verifier/src/verifier.rs:391+`
- `build_bytecode_read_raf_claim` in
  `crates/jolt-compiler/examples/jolt_core_module.rs:6793` (helper
  shared by prover and stage 6 verifier)
- `apply_normalization` for sumcheck-point reordering
- `field_pow2::<F>(k)` for 2^k scaling (avoids 1u64 << k overflow)

## Carryover gotchas (from prior session, still relevant)

- `1u64 << k` overflows for stages 5+. Use `field_pow2::<F>(k)`.
- `LT` and `EqPlusOne` are asymmetric — argument order matters; read
  jolt-core's expected_output for `(x, y)` zip carefully.
- Multi-phase normalize: `PointNormalization::Segments { sizes,
  output_order }` reorders raw sumcheck points.
- `PreprocessedPolyEval` evaluates at the stage's full sumcheck point;
  for sliced points use `EvaluatePreprocessed` op with explicit
  `at_challenges`.
- `JoltVerifyingKey` must be populated with preprocessing for stage-4+
  verification. Done in `tests/muldiv.rs::modular_self_verify` and
  `cross_verifier::fixture::build_honest_fixture`.
- `field_pow2` lives in jolt-verifier or jolt-compiler — grep for it.

## Notes (auto-populated when blocked)

### Session 2026-04-28 progress summary

Landed (7 commits this session):

- **T0** (pre-flight): confirmed repeated `Eval(P)` factors compose
  multiplicatively in `evaluate_formula` — no `EvalSquared` variant
  needed. Booleanity / HammingBooleanity formulas can use repeated Eval.
- **T11** (test): added `t2_eval_tampers` covering all 7 stages.
  Discovered that PCS opening verification is currently a no-op (no
  `VerifierOp::CollectOpeningClaim` calls anywhere). Pre-stage-7 evals
  rejected via downstream transcript divergence; stage 7 needed
  CheckOutput.
- **T2**: stage 7 CheckOutput authored with full HammingWeight
  reduction formula `Σ_i G_i · (γ^{3i} + γ^{3i+1}·eq_bool + γ^{3i+2}·eq_virt_i)`
  using `normalize: Some(Reverse)`. T2@stage 7 gap closed.
- **T16**: added `t11_public_io_tampers` covering input-byte and
  panic-flag tampers. All pass via preamble divergence.
- **T17** (newly numbered): added `t6_config_field_tampers` covering
  TraceLength and RamK. Both pass via preamble divergence.
- **T12**: added `t9_batch_claim_tampers` covering last-eval-idx per
  stage. All 7 stages pass.

Soundness suite is now at 9 tests, 9/9 green:
honest_acceptance, KGC, taxonomy_coverage, T1, T2, T6, T8, T9, T11.

KNOWN_GAPS empty.

### Remaining gaps & priorities

**Big-ticket gaps (require new infrastructure):**

- Stage 6 CheckOutput (T3+T4+T5) — 6 instances. Easy: RamRaVirt,
  InstRaVirt, IncReduction. Medium: Booleanity, HammingBooleanity (use
  repeated Eval per T0). Hard: BytecodeReadRaf (multi-stage gamma
  composition). All-or-nothing — can't push CheckOutput until all 6
  authored. Defense in depth only — T2/T7 already caught via
  transcript divergence into stage 7.
- Stage 5 CheckOutput (T6+T7+T8) — 3 instances. Two simple, one needs
  new `ClaimFactor::CombineEntryEval` for InstructionReadRaf. Same
  defense-in-depth nature.
- PCS opening claim verification (T9+T10) — `VerifyOpenings` is
  currently no-op. Adding `VerifierOp::CollectOpeningClaimAt` and
  wiring at stages 4-7 closes T3/T4 (commitment swap, opening byte).
  Multi-day; large blast radius.
- Soundness suite expansion for T3/T4/T5/T10 — needs `apply_tamper`
  extensions for `TamperLocation::Commitment / OpeningProofByte /
  CommitSlot` (currently return Vacuous) and a new tag-byte tamper
  for T10. Mechanical; no verifier changes needed.

**Recommended next pickup order (in priority):**

1. T13 (T3 commitment swap test) — extend `apply_tamper` for
   `TamperLocation::Commitment` to swap with another fixture's
   commitment; the modular verifier should reject because... actually
   it WON'T reject (PCS opening is no-op). So T13 reveals the
   PCS-opening gap: a registered `KnownGap` entry until T9+T10 land.
   Adding T13 first quantifies the PCS gap rigorously.
2. T9+T10 (PCS opening claim wiring) — closes T3/T4/T5 + the latent
   stage-7 path mentioned in T11 discovery. Highest leverage.
3. Stage 6 CheckOutput simple instances first (RamRaVirt, InstRaVirt,
   IncReduction) — author formulas alone (no push). Defer Booleanity /
   HammingBooleanity to a follow-up commit.
4. Booleanity + HammingBooleanity via repeated `Eval`. Push stage 6
   CheckOutput after BytecodeReadRaf is in.
5. BytecodeReadRaf — most complex stage 6 instance.
6. Stage 5 CheckOutput simple instances.
7. `ClaimFactor::CombineEntryEval` design + Stage 5 InstructionReadRaf.

### Session 2026-04-28 final state (after stage-8 wiring)

**Progress**:
- 13 commits this session covering T0-T18 (numbering compressed —
  some tasks merged).
- Stage-8 PCS opening verification fully wired via three new
  VerifierOp variants: CollectOpeningClaimAt, ScaleEval, AliasEval.
- Soundness suite expanded from 5 to 12 tests covering T1, T2, T3,
  T4, T5, T6, T8, T9, T11 + 3 meta (S1, KGC, S7-narrow).
- Empty KNOWN_GAPS.
- All gates green: jolt-equivalence 56/56 (-j1), jolt-core muldiv
  in both modes, clippy on jolt-core/compiler/verifier/zkvm.
- Pre-existing red in `booleanity_debug.rs` is grandfathered.

**What jolt_verifier::verify now catches**:
- T1 (round-poly coeff flip) — sumcheck verifier rejects directly.
- T2 (eval tamper) — at stages 1-6 via downstream transcript
  divergence; at stage 7 via CheckOutput composition.
- T3 (commitment swap) — stage-8 PCS verification rejects.
- T4 (opening-proof structural) — stage-8 length check rejects.
- T5 (commit slot None ↔ Some) — AbsorbCommitment skip diverges
  transcript.
- T6 (config field tamper) — preamble divergence.
- T8 (round-poly degree) — sumcheck structural validation.
- T9 (batch-claim tamper, last-eval-idx) — same paths as T2.
- T11 (public IO tamper) — preamble divergence.

**Remaining (defense in depth or test-infra extensions)**:
- Stage-6 / Stage-5 CheckOutput — explicit constraint witnessing
  for T2 at those stages. Currently caught indirectly via
  transcript divergence into stage 7 PCS.
- T10 (domain-separator-tag tamper) — needs new tampering
  infrastructure (no current path for tag-byte mutation).
- T4 full per-byte FlipBit — needs Dory proof serde round-trip;
  current truncation test covers the structural failure mode.
- T7 (cross-stage eval) — overlaps with T2/T9; could add explicit
  test if we want isolation by witnessed constraint.
- Stage-5 InstructionReadRaf CheckOutput — needs new
  `ClaimFactor::CombineEntryEval` for prefix-suffix composition.
  Multi-day; pure defense in depth (T2/T9 already catch the eval
  tamper at this stage indirectly).

The modular verifier is now rejection-equivalent to jolt-core's on
the muldiv fixture for all 9 tested tamper categories. Closing the
remaining items strengthens explicit constraint coverage but
doesn't change the soundness story for honest-vs-malicious
proofs in practice.

### Tooling notes

- Pre-commit hooks (lefthook) take ~3-5 min per commit (autofix-clippy
  + check-clippy). Plan for that overhead per task.
- `JOLT_VERIFIER_DEBUG=1` instrumentation works for output_check
  debugging — see prior session handoff for the exact eprintln
  insertions in `crates/jolt-verifier/src/verifier.rs`.
- The pre-existing red in `crates/jolt-equivalence/tests/booleanity_debug.rs`
  is grandfathered; clippy `--all-targets` will fail there but it's
  not blocking per the spec.
