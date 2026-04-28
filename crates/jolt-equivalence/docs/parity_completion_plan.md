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
[ ] T2   Stage 7 CheckOutput (HammingWeightClaimReduction) → closes T2@7
[ ] T3   Stage 6 simple output_checks (RamRaVirt, InstRaVirt, IncReduction)
         — author formulas only, defer push until T5
[ ] T4   Stage 6 Booleanity + HammingBooleanity output_check
[ ] T5   Stage 6 BytecodeReadRaf output_check + push CheckOutput op
[ ] T6   Stage 5 simple output_checks (RamRaClaim, RegistersValEval)
         — author formulas only, defer push until T8
[ ] T7   Add ClaimFactor::CombineEntryEval variant + verifier arm
[ ] T8   Stage 5 InstructionReadRaf output_check + push CheckOutput op
[ ] T9   Add VerifierOp::CollectOpeningClaimAt + verifier handler
[ ] T10  Wire per-poly CollectOpeningClaim(At) at stages 4–7
[ ] T12  Soundness suite: T9 batch-claim-tamper test
[ ] T13  Soundness suite: T3 commitment-swap test
[ ] T14  Soundness suite: T10 domain-separator-tag test
[ ] T15  Soundness suite: T5 commit-slot-None↔Some test
[ ] T16  Soundness suite: T11 public-IO test
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

(empty)
