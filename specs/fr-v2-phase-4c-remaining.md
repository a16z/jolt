# Phase 4c.2 — Activate FieldRegRW in the Stage 4 Batched Sumcheck

This file is a working note for the next session, not a long-lived spec.
Delete after Phase 4c.2 lands.

## State at HEAD

- `9d919f85f wip(phase-4c): Stage 4 FieldRegRW kernel + MLIR scaffolding`
- `cb027fbba wip(phase-4c): wire field_reg_rw.gamma squeeze + batched_inputs slot`

What works now:
- Kernel side (`jolt-kernels/src/stage4.rs`): `Stage4Relation::FieldRegRW`,
  `Stage4KernelAbi::FieldRegRW`, `Stage4FieldRegWitness`, `field_reg_rw_state`
  (Dense path only — sparse phase-segmented logic deferred to Phase 5b for
  FR-active programs), `expected_field_reg_rw`. Prover + verifier dispatchers
  wired.
- MLIR side (`bolt/.../phases/stage4.rs`): `jolt.stage4.field_reg_rw` relation
  declared, FR oracles (FieldRdWriteValue/FieldRs1Value/FieldRs2Value virtual
  trace + FieldRegVal/FrRs1Ra/FrRs2Ra/FrdWa virtual on FR domain + FrdInc
  committed), `jolt.stage4_field_reg_rw_domain`, 3 FR opening inputs from
  Stage 3, `stage4.field_reg_rw.gamma` transcript squeeze, `field_reg_gamma`
  threaded into `Stage4BatchedSumcheckInputs`.

What's dormant: the FR relation is declared but no plan references it. The
gamma + opening inputs are dead-code-guarded.

## Source-branch protocol (verified against `feat/fr-coprocessor-v2`, commit `f37f9a98d`)

FR RW P1 (cycle phase, log_t rounds, degree 3, mirror of integer RegistersRW):
```
eq · frd_wa · field_reg_val
+ eq · frd_wa · frd_inc
+ γ_fr_rw · eq · frs1_ra · field_reg_val
+ γ_fr_rw² · eq · frs2_ra · field_reg_val
```

FR RW P2 (address phase, LOG_K_FR=4 rounds, degree 2, dense post-ScalarCapture):
After P1, the bound values of `eq` and `frd_inc` are captured as challenges
`ch_fr_eq_bound`, `ch_fr_inc_bound`. P2 then runs:
```
ch_fr_eq_bound · frd_wa · field_reg_val
+ ch_fr_eq_bound · frd_wa · ch_fr_inc_bound
+ γ · ch_fr_eq_bound · frs1_ra · field_reg_val
+ γ² · ch_fr_eq_bound · frs2_ra · field_reg_val
```

Total: `fr_rw_rounds = LOG_K_FR + log_t = 4 + log_t`. In modular-sdk's
batched sumcheck (max rounds = `register_log_k + log_t` = 11 for fixture),
FR enters at `first_active_round = register_log_k - LOG_K_FR = 3` (skipping
the first 3 address rounds).

For inert muldiv (zero FR cycles): all 5 polynomials are zero → all 4 terms
evaluate to zero → claim is zero → trivially satisfied. The 2-phase
segmented sumcheck collapses to a single dense path; my
`field_reg_rw_state` already handles this.

## Remaining work for Phase 4c.2

### 1. Mirror the registers claim_expr chain for FR in `append_stage4_batched_sumcheck`

Right after the registers claim_expr block (currently lines ~576-610), add:
```rust
let field_reg_gamma2 = append_field_pow(
    context, module,
    "stage4.field_reg_rw.gamma2",
    spec.field_reg_gamma, 2,
)?;
let fr_rs1_term = append_field_mul(
    context, module,
    "stage4.field_reg_rw.term.FieldRs1Value",
    spec.field_reg_gamma,
    inputs.field_rs1_value.eval,
)?;
let fr_rs2_term = append_field_mul(
    context, module,
    "stage4.field_reg_rw.term.FieldRs2Value",
    field_reg_gamma2,
    inputs.field_rs2_value.eval,
)?;
let fr_sum = append_field_add(
    context, module,
    "stage4.field_reg_rw.partial.FieldRdWriteValueFieldRs1Value",
    inputs.field_rd_write_value.eval,
    fr_rs1_term,
)?;
let field_reg_claim = append_field_add(
    context, module,
    "stage4.field_reg_rw.claim_expr",
    fr_sum,
    fr_rs2_term,
)?;
```

### 2. Append FR sumcheck claim to the `claims` array

Insert as 3rd entry (after ram_val_check):
```rust
append_sumcheck_claim(
    context, module,
    SumcheckClaimSpec {
        symbol: "stage4.field_reg_rw.input",
        stage: "stage4",
        domain: "jolt.stage4_field_reg_rw_domain",
        num_rounds: stage4_field_reg_rw_rounds(params),
        degree: FIELD_REG_RW_DEGREE,
        claim: "stage4.field_reg_rw.weighted_values",
        relation: "jolt.stage4.field_reg_rw",
    },
    field_reg_claim,
    &[
        inputs.field_rd_write_value.claim,
        inputs.field_rs1_value.claim,
        inputs.field_rs2_value.claim,
    ],
)?,
```

### 3. Update batch `ordered_claims` to include `"stage4.field_reg_rw.input"`

### 4. Add FR sumcheck instance result

```rust
let field_reg_rw = append_sumcheck_instance_result(
    context, module,
    SumcheckInstanceResultSpec {
        symbol: "stage4.field_reg_rw.instance",
        source: "stage4.sumcheck",
        claim: "stage4.field_reg_rw.input",
        relation: "jolt.stage4.field_reg_rw",
        index: 2,
        point_arity: stage4_field_reg_rw_rounds(params),
        num_rounds: stage4_field_reg_rw_rounds(params),
        round_offset: params.register_log_k - params.field_reg_log_k, // = 3
        point_order: "stage4_field_reg_rw", // see step 5
        degree: FIELD_REG_RW_DEGREE,
    },
    point,
    result_value,
)?;
```

### 5. Add `stage4_field_reg_rw` point_order kernel helper

Look at how `"stage4_registers_rw"` is dispatched. It's handled by
`normalize_stage4_registers_rw_point` in `jolt-kernels/src/stage4.rs` (~line
672). Add a parallel `normalize_stage4_field_reg_rw_point` and a parallel
`normalize_stage4_field_reg_rw_cycle_point` (used by
`expected_field_reg_rw`). The shape is: extract `(address: LOG_K_FR, cycle:
log_t)` from the point. Search for `"stage4_registers_rw"` in the dispatcher
(~line 672) and add the FR arm.

### 6. Extend `append_stage4_output_openings`

Pass `field_reg_rw: (Value, Value)` as a new parameter. Inside, add the 4
virtual evals (FieldRegVal/FrRs1Ra/FrRs2Ra/FrdWa, indices 0-3 of FR
instance) on `jolt.stage4_field_reg_rw_domain` with point_arity =
`stage4_field_reg_rw_rounds(params)`. Then FrdInc on `jolt.trace_domain`,
point_arity = `params.log_t`, via `append_point_slice` extracting the cycle
suffix.

### 7. Bump output_openings batch count

The existing count formula needs to add 5 (FR outputs).

### 8. KernelSpec resolve dispatcher

`crates/bolt/src/protocols/jolt/phases/stage1.rs`: add arm for
`"jolt.stage4.field_reg_rw"` → KernelSpec with
abi=`"jolt_stage4_field_reg_rw"`.

### 9. Emit template (`bolt/.../emit/rust/stage4.rs`)

- ABI dispatch arm: `"jolt.stage4.field_reg_rw" => "jolt_stage4_field_reg_rw"`
- Verifier `expected_batched_output_claim` dispatch arm calling
  `expected_field_reg_rw`
- Raw-string `fn expected_field_reg_rw` template using formula:
  `eq_eval(r_cycle, trace_point) * (frd_wa * (field_reg_val + frd_inc)
   + gamma * (frs1_ra * field_reg_val + gamma * frs2_ra * field_reg_val))`

### 10. Inert witness wiring

Stage 4 muldiv prover lives at:
- `crates/jolt-prover/src/prover.rs` (line ~1109): `Stage4ProverInputs::new(...).with_stage45_sparse_trace_witness(...)`
- `crates/bolt/src/protocols/jolt/artifacts.rs` (line ~1173): same

Both need an additional `.with_field_reg(Stage4FieldRegWitness {
field_reg_count: 16, trace_len, field_reg_val: &zeros_16T, frs1_ra:
&zeros_16T, frs2_ra: &zeros_16T, frd_wa: &zeros_16T, frd_inc: &zeros_T })`.

Cleanest: extend `Stage45SparseTraceWitness` (in `jolt-witness/src/lib.rs`)
to carry pre-allocated zero vecs for the FR slots, then add `with_field_reg`
into a new builder `with_stage4_fr_witness` that takes those.

### 11. Goldens regen + commitment_ir fixture bumps

```
JOLT_UPDATE_GOLDENS=1 cargo nextest run -p bolt --test commitment_ir \
    generated_jolt_artifacts_have_uniform_crate_layout_and_import_rules \
    --cargo-quiet
```

Then run full `cargo nextest run -p bolt --test commitment_ir --cargo-quiet
--no-fail-fast` and bump the fixtures: kernels.len() (3→4),
opening_inputs.len() (was N; +3), field_exprs.len() (+5),
opening_equalities.len() (unchanged), claims.len() (2→3),
opening_claims.len() (was 16; +5), ordered_claims (insert FR claim).

### 12. Gates

```
source ./.bolt-dev-env
cargo nextest run -p bolt --test commitment_ir --cargo-quiet --no-fail-fast  # 53/53
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host
cargo clippy -p jolt-witness -p jolt-r1cs -p jolt-kernels -p bolt \
    --message-format=short -q --all-targets -- -D warnings
```

### 13. Commit message

`feat(phase-4c.2): activate FieldRegRW in Stage 4 batched sumcheck`

---

## After Phase 4c.2 lands → Phase 4b

Stage 5 `FieldRegValEvaluation` mirrors `RegistersValEvaluation`. The
opening point that 4b consumes is `stage4.field_reg_rw.opening.FieldRegVal`
which 4c.2 publishes. Formula: degree-3 `frd_inc · frd_wa_at_addr · LT`.
Estimated ~650 LOC. Same layered touchpoints (kernel/MLIR/emit/dispatcher).

## After Phase 4b lands → Phase 5

Per `specs/fr-v2-port-plan.md` lines 113-122:
- 5a: poseidon2-external example (~150 LOC, low risk)
- 5b: FieldRegConfig replay materializers — converts inert all-zero FR polys
  into FR-event-driven replay (this is where the sparse phase-segmented
  Stage 4 kernel finally lands for FR-active programs)
- 5c: poseidon2-sdk example + e2e gate
- 5d: Audit fixes C1-C11
