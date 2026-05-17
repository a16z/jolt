#![expect(
    clippy::expect_used,
    clippy::print_stderr,
    reason = "verifier cleanup tests use explicit panic messages and print metrics for CI logs"
)]

use std::path::{Path, PathBuf};

/// S2.5 keeps only Jolt-specific verifier-program data in generated
/// `verifier.rs`; reusable program shape and dispatch live in
/// `crates/bolt-verifier-runtime`.
const GENERATED_VERIFIER_TARGET_LOC: usize = 6_500;
const GENERATED_VERIFIER_STRETCH_LOC: usize = 3_000;
const VERIFIER_RS_TARGET_LOC: usize = 850;
const VERIFIER_RS_STRETCH_LOC: usize = 350;
const STAGE6_STAGE7_TARGET_LOC: usize = 3_000;

const GENERATED_VERIFIER_BASELINE_LOC_CEILING: usize = 9_185;
const VERIFIER_RS_BASELINE_LOC_CEILING: usize = VERIFIER_RS_TARGET_LOC;
const STAGE6_STAGE7_BASELINE_LOC_CEILING: usize = STAGE6_STAGE7_TARGET_LOC;

/// Tier A ceiling inside the generated verifier crate. S2 moves generic Bolt
/// verifier scaffolding into `crates/bolt-verifier-runtime`, so this generated
/// surface should stay at zero after the extraction.
const BOLT_RUNTIME_BASELINE_LOC_CEILING: usize = 0;

/// Tier B ceiling: hand-written Jolt verifier math lives in
/// `stages/jolt_relations.rs`. Growth here is a *protocol-math* decision and
/// should be reviewed as such, not as emitter-LOC creep.
const JOLT_VERIFIER_CORE_BASELINE_LOC_CEILING: usize = 700;
const STAGE_LOCAL_PLAN_STRUCT_BASELINE_CEILING: usize = 18;
const FIELD_EXPR_OPERAND_CONSTANT_BASELINE_CEILING: usize = 0;
const BATCH_OPERAND_STRING_SITE_BASELINE_CEILING: usize = 0;
const CLAIM_INPUT_OPENING_STRING_SITE_BASELINE_CEILING: usize = 0;
const POINT_CONCAT_INPUT_STRING_SITE_BASELINE_CEILING: usize = 0;
const STAGE_LOCAL_MACRO_RULES_BASELINE_CEILING: usize = 0;
const STAGE_HELPER_FUNCTION_BASELINE_CEILING: usize = 38;
const RELATION_STRING_SITE_BASELINE_CEILING: usize = 0;
const SUMCHECK_POINT_ORDER_STRING_SITE_BASELINE_CEILING: usize = 0;
const RELATION_INDEXED_EVAL_PREFIX_SITE_BASELINE_CEILING: usize = 0;
const HANDWRITTEN_EXPECTED_OUTPUT_FUNCTION_BASELINE_CEILING: usize = 12;

const ALLOWED_JOLT_PROTOCOL_SYMBOLS: &[&str] = &[
    "jolt.commitment_phase",
    "jolt.main_witness_commit_domain",
    "jolt.main_witness_commitments",
    "jolt.main_witness_polys",
    "jolt.ram_address_domain",
    "jolt.stage1.outer.remaining",
    "jolt.stage1.outer.uniskip",
    "jolt.stage1_outer",
    "jolt.stage1_uniskip_domain",
    "jolt.stage2",
    "jolt.stage2.batched",
    "jolt.stage2.instruction_lookup.claim_reduction",
    "jolt.stage2.product_virtual.remainder",
    "jolt.stage2.product_virtual.uniskip",
    "jolt.stage2.ram.output_check",
    "jolt.stage2.ram.output_check.layout",
    "jolt.stage2.ram.raf_evaluation",
    "jolt.stage2.ram.read_write",
    "jolt.stage2_ram_rw_domain",
    "jolt.stage2_uniskip_domain",
    "jolt.stage3",
    "jolt.stage3.batched",
    "jolt.stage3.instruction_input",
    "jolt.stage3.registers_claim_reduction",
    "jolt.stage3.spartan_shift",
    "jolt.stage4",
    "jolt.stage4.batched",
    "jolt.stage4.ram_val_check",
    "jolt.stage4.registers_read_write",
    "jolt.stage4_registers_rw_domain",
    "jolt.stage5.batched",
    "jolt.stage5.instruction_read_raf",
    "jolt.stage5.ram_ra_claim_reduction",
    "jolt.stage5.registers_val_evaluation",
    "jolt.stage5_instruction_ra_chunk_domain",
    "jolt.stage5_instruction_read_raf_domain",
    "jolt.stage6.batched",
    "jolt.stage6.booleanity",
    "jolt.stage6.bytecode_read_raf",
    "jolt.stage6.hamming_booleanity",
    "jolt.stage6.inc_claim_reduction",
    "jolt.stage6.instruction_ra_virtual",
    "jolt.stage6.ram_ra_virtual",
    "jolt.stage6_booleanity_domain",
    "jolt.stage6_bytecode_read_raf_domain",
    "jolt.stage7.batched",
    "jolt.stage7.hamming_booleanity",
    "jolt.stage7.hamming_weight_claim_reduction",
    "jolt.stage7_hamming_weight_claim_reduction_domain",
    "jolt.stage8",
    "jolt.trace_domain",
    "jolt.trusted_advice_commitment",
    "jolt.untrusted_advice_commitment",
];

const GENERIC_COMPILER_JOLT_PATTERNS: &[&str] = &[
    "jolt.",
    "Jolt",
    "jolt_",
    "jolt-",
    "jolt_core",
    "stage1_outer",
    "stage1",
    "stage2",
    "stage3",
    "stage4",
    "stage5",
    "stage6",
    "stage7",
    "stage8",
    "uniskip",
    "spartan",
    "bytecode",
    "hamming",
    "instruction_read",
    "ram_val",
    "ram_ra",
    "registers_read",
    "lookup",
    "dory",
    "bn254",
];

#[derive(Debug, Default)]
struct VerifierCleanupMetrics {
    total_loc: usize,
    generated_surface_loc: usize,
    /// Tier A: generic Bolt verifier scaffolding still emitted inside
    /// `crates/jolt-verifier/src` after extraction.
    bolt_runtime_loc: usize,
    /// Tier B: audited Jolt verifier core
    /// (`crates/jolt-verifier/src/stages/jolt_relations.rs`).
    jolt_verifier_core_loc: usize,
    verifier_rs_loc: usize,
    stage6_stage7_loc: usize,
    stage_local_generic_plan_structs: usize,
    field_expr_operand_constants: usize,
    batch_operand_string_sites: usize,
    claim_input_opening_string_sites: usize,
    point_concat_input_string_sites: usize,
    stage_local_macro_rules: usize,
    stage_local_helper_functions: usize,
    relation_string_sites: usize,
    sumcheck_point_order_string_sites: usize,
    relation_indexed_eval_prefix_sites: usize,
    compute_poly_op_call_sites: usize,
    compute_point_op_call_sites: usize,
    value_graph_relation_outputs: usize,
    handwritten_expected_output_functions: usize,
}

#[test]
fn checked_in_generated_verifier_metrics_are_recorded_and_bounded() {
    let verifier_src = workspace_root().join("crates/jolt-verifier/src");
    if !verifier_src.exists() {
        return;
    }
    let metrics = verifier_cleanup_metrics(&verifier_src);

    eprintln!(
        "\nGenerated verifier cleanup metrics\n\
         generated_surface_loc: {generated_surface_loc} (target <= {target_loc}, stretch <= {stretch_loc})\n\
         tier_a_bolt_runtime_loc: {bolt_runtime_loc} (baseline ceiling <= {bolt_runtime_baseline}; Tier A: generic Bolt scaffolding)\n\
         tier_b_jolt_verifier_core_loc: {jolt_verifier_core_loc} (baseline ceiling <= {jolt_core_baseline}; Tier B: audited Jolt verifier math)\n\
         total_loc: {total_loc} (baseline ceiling <= {baseline_loc})\n\
         verifier_rs_loc: {verifier_rs_loc} (target <= {verifier_target}, stretch <= {verifier_stretch}, baseline ceiling <= {verifier_baseline})\n\
         stage6_stage7_loc: {stage6_stage7_loc} (target <= {stage67_target}, baseline ceiling <= {stage67_baseline})\n\
         stage_local_generic_plan_structs: {plan_structs} (baseline ceiling <= {plan_baseline})\n\
         field_expr_operand_constants: {operand_constants} (baseline ceiling <= {operand_baseline})\n\
         batch_operand_string_sites: {batch_operand_string_sites} (baseline ceiling <= {batch_operand_baseline})\n\
         claim_input_opening_string_sites: {claim_input_opening_string_sites} (baseline ceiling <= {claim_input_opening_baseline})\n\
         point_concat_input_string_sites: {point_concat_input_string_sites} (baseline ceiling <= {point_concat_input_baseline})\n\
         stage_local_macro_rules: {stage_local_macro_rules} (baseline ceiling <= {macro_rules_baseline})\n\
         stage_local_helper_functions: {helper_functions} (baseline ceiling <= {helper_baseline})\n\
         relation_string_sites: {relation_sites} (baseline ceiling <= {relation_baseline})\n\
         sumcheck_point_order_string_sites: {point_order_sites} (baseline ceiling <= {point_order_baseline})\n\
         relation_indexed_eval_prefix_sites: {indexed_eval_prefix_sites} (baseline ceiling <= {indexed_eval_prefix_baseline})\n\
         compute_poly_op_call_sites: {compute_poly_op_call_sites}\n\
         compute_point_op_call_sites: {compute_point_op_call_sites}\n\
         value_graph_relation_outputs: {value_graph_relation_outputs}\n\
         handwritten_expected_output_functions: {expected_output_functions} (baseline ceiling <= {expected_output_functions_baseline})",
        generated_surface_loc = metrics.generated_surface_loc,
        bolt_runtime_loc = metrics.bolt_runtime_loc,
        bolt_runtime_baseline = BOLT_RUNTIME_BASELINE_LOC_CEILING,
        jolt_verifier_core_loc = metrics.jolt_verifier_core_loc,
        jolt_core_baseline = JOLT_VERIFIER_CORE_BASELINE_LOC_CEILING,
        total_loc = metrics.total_loc,
        target_loc = GENERATED_VERIFIER_TARGET_LOC,
        stretch_loc = GENERATED_VERIFIER_STRETCH_LOC,
        baseline_loc = GENERATED_VERIFIER_BASELINE_LOC_CEILING,
        verifier_rs_loc = metrics.verifier_rs_loc,
        verifier_target = VERIFIER_RS_TARGET_LOC,
        verifier_stretch = VERIFIER_RS_STRETCH_LOC,
        verifier_baseline = VERIFIER_RS_BASELINE_LOC_CEILING,
        stage6_stage7_loc = metrics.stage6_stage7_loc,
        stage67_target = STAGE6_STAGE7_TARGET_LOC,
        stage67_baseline = STAGE6_STAGE7_BASELINE_LOC_CEILING,
        plan_structs = metrics.stage_local_generic_plan_structs,
        plan_baseline = STAGE_LOCAL_PLAN_STRUCT_BASELINE_CEILING,
        operand_constants = metrics.field_expr_operand_constants,
        operand_baseline = FIELD_EXPR_OPERAND_CONSTANT_BASELINE_CEILING,
        batch_operand_string_sites = metrics.batch_operand_string_sites,
        batch_operand_baseline = BATCH_OPERAND_STRING_SITE_BASELINE_CEILING,
        claim_input_opening_string_sites = metrics.claim_input_opening_string_sites,
        claim_input_opening_baseline = CLAIM_INPUT_OPENING_STRING_SITE_BASELINE_CEILING,
        point_concat_input_string_sites = metrics.point_concat_input_string_sites,
        point_concat_input_baseline = POINT_CONCAT_INPUT_STRING_SITE_BASELINE_CEILING,
        stage_local_macro_rules = metrics.stage_local_macro_rules,
        macro_rules_baseline = STAGE_LOCAL_MACRO_RULES_BASELINE_CEILING,
        helper_functions = metrics.stage_local_helper_functions,
        helper_baseline = STAGE_HELPER_FUNCTION_BASELINE_CEILING,
        relation_sites = metrics.relation_string_sites,
        relation_baseline = RELATION_STRING_SITE_BASELINE_CEILING,
        point_order_sites = metrics.sumcheck_point_order_string_sites,
        point_order_baseline = SUMCHECK_POINT_ORDER_STRING_SITE_BASELINE_CEILING,
        indexed_eval_prefix_sites = metrics.relation_indexed_eval_prefix_sites,
        indexed_eval_prefix_baseline = RELATION_INDEXED_EVAL_PREFIX_SITE_BASELINE_CEILING,
        compute_poly_op_call_sites = metrics.compute_poly_op_call_sites,
        compute_point_op_call_sites = metrics.compute_point_op_call_sites,
        value_graph_relation_outputs = metrics.value_graph_relation_outputs,
        expected_output_functions = metrics.handwritten_expected_output_functions,
        expected_output_functions_baseline =
            HANDWRITTEN_EXPECTED_OUTPUT_FUNCTION_BASELINE_CEILING,
    );

    assert!(
        metrics.generated_surface_loc <= GENERATED_VERIFIER_TARGET_LOC,
        "generated verifier surface grew to {} LOC; keep stage files thin (target <= {})",
        metrics.generated_surface_loc,
        GENERATED_VERIFIER_TARGET_LOC,
    );
    if metrics.generated_surface_loc <= GENERATED_VERIFIER_STRETCH_LOC {
        eprintln!(
            "[verifier_cleanup] notice: generated_surface_loc = {} reached stretch target {}; \
             tighten GENERATED_VERIFIER_TARGET_LOC",
            metrics.generated_surface_loc, GENERATED_VERIFIER_STRETCH_LOC,
        );
    }
    assert_eq!(
        metrics.bolt_runtime_loc, BOLT_RUNTIME_BASELINE_LOC_CEILING,
        "Tier A bolt verifier runtime grew to {} LOC (ceiling {}); generic Bolt scaffolding should ratchet down, not up",
        metrics.bolt_runtime_loc, BOLT_RUNTIME_BASELINE_LOC_CEILING
    );
    assert!(
        metrics.jolt_verifier_core_loc <= JOLT_VERIFIER_CORE_BASELINE_LOC_CEILING,
        "Tier B audited Jolt verifier core grew to {} LOC (ceiling {}); growth here must be reviewed as a protocol-math decision",
        metrics.jolt_verifier_core_loc, JOLT_VERIFIER_CORE_BASELINE_LOC_CEILING,
    );
    assert!(
        metrics.total_loc <= GENERATED_VERIFIER_BASELINE_LOC_CEILING,
        "generated verifier total grew to {} LOC (ceiling {})",
        metrics.total_loc,
        GENERATED_VERIFIER_BASELINE_LOC_CEILING,
    );
    assert!(
        metrics.verifier_rs_loc <= VERIFIER_RS_BASELINE_LOC_CEILING,
        "top-level verifier grew to {} LOC; keep orchestration small and readable",
        metrics.verifier_rs_loc
    );
    assert!(
        metrics.stage6_stage7_loc <= STAGE6_STAGE7_BASELINE_LOC_CEILING,
        "stage6 + stage7 grew to {} LOC (ceiling {})",
        metrics.stage6_stage7_loc,
        STAGE6_STAGE7_BASELINE_LOC_CEILING,
    );
    assert!(
        metrics.stage_local_generic_plan_structs <= STAGE_LOCAL_PLAN_STRUCT_BASELINE_CEILING,
        "stage-local generic plan struct count grew to {}; move shared plan types into common verifier runtime",
        metrics.stage_local_generic_plan_structs
    );
    assert!(
        metrics.field_expr_operand_constants == FIELD_EXPR_OPERAND_CONSTANT_BASELINE_CEILING,
        "field-expression operand constants grew to {}; compact field expression encoding",
        metrics.field_expr_operand_constants
    );
    assert!(
        metrics.batch_operand_string_sites == BATCH_OPERAND_STRING_SITE_BASELINE_CEILING,
        "batch operand string sites grew to {}; prefer structured claim slices",
        metrics.batch_operand_string_sites
    );
    assert!(
        metrics.claim_input_opening_string_sites
            == CLAIM_INPUT_OPENING_STRING_SITE_BASELINE_CEILING,
        "claim input-opening string sites grew to {}; prefer structured input-opening slices",
        metrics.claim_input_opening_string_sites
    );
    assert!(
        metrics.point_concat_input_string_sites == POINT_CONCAT_INPUT_STRING_SITE_BASELINE_CEILING,
        "point-concat input string sites grew to {}; prefer structured point input slices",
        metrics.point_concat_input_string_sites
    );
    assert!(
        metrics.stage_local_macro_rules == STAGE_LOCAL_MACRO_RULES_BASELINE_CEILING,
        "stage-local macro_rules sites grew to {}; prefer named constructors or shared runtime helpers",
        metrics.stage_local_macro_rules
    );
    assert!(
        metrics.stage_local_helper_functions <= STAGE_HELPER_FUNCTION_BASELINE_CEILING,
        "stage-local helper function count grew to {}; factor verifier mechanics into shared runtime",
        metrics.stage_local_helper_functions
    );
    assert!(
        metrics.relation_string_sites == RELATION_STRING_SITE_BASELINE_CEILING,
        "relation string sites grew to {}; prefer typed relation plan data or explicit allowlists",
        metrics.relation_string_sites
    );
    assert!(
        metrics.sumcheck_point_order_string_sites
            == SUMCHECK_POINT_ORDER_STRING_SITE_BASELINE_CEILING,
        "sumcheck point-order string sites grew to {}; prefer typed point-order plan data",
        metrics.sumcheck_point_order_string_sites
    );
    assert!(
        metrics.relation_indexed_eval_prefix_sites
            == RELATION_INDEXED_EVAL_PREFIX_SITE_BASELINE_CEILING,
        "relation indexed-eval prefix sites grew to {}; prefer typed eval-family plan data",
        metrics.relation_indexed_eval_prefix_sites
    );
    assert!(
        metrics.handwritten_expected_output_functions
            <= HANDWRITTEN_EXPECTED_OUTPUT_FUNCTION_BASELINE_CEILING,
        "handwritten expected-output helper count grew to {}; move output math into typed value-graph plan data",
        metrics.handwritten_expected_output_functions
    );
}

#[test]
fn checked_in_generated_verifier_respects_boundary_hygiene() {
    let verifier_root = workspace_root().join("crates/jolt-verifier");
    if !verifier_root.exists() {
        return;
    }
    let manifest =
        std::fs::read_to_string(verifier_root.join("Cargo.toml")).expect("read verifier manifest");
    for package in [
        "jolt-prover",
        "jolt-kernels",
        "jolt-core",
        "jolt-equivalence",
        "jolt-profiling",
        "tracer",
    ] {
        assert!(
            !manifest.contains(package),
            "generated verifier manifest depends on forbidden package `{package}`"
        );
    }

    for path in rust_files(&verifier_root.join("src")) {
        let source = std::fs::read_to_string(&path).expect("read verifier source");
        for pattern in [
            "use jolt_prover",
            "jolt_prover::",
            "use jolt_kernels",
            "jolt_kernels::",
            "use jolt_core",
            "jolt_core::",
            "use jolt_equivalence",
            "jolt_equivalence::",
            "use jolt_profiling",
            "jolt_profiling::",
            "use tracer",
            "tracer::",
        ] {
            assert!(
                !source.contains(pattern),
                "generated verifier source `{}` contains forbidden import/reference `{pattern}`",
                path.display()
            );
        }
        assert!(
            !source.contains("JoltField::Challenge")
                && !source.contains("Transcript<Challenge = Challenge>")
                && !source.contains("Challenge = <"),
            "generated verifier source `{}` drifted away from the full-field transcript path",
            path.display()
        );
    }
}

#[test]
fn checked_in_generated_verifier_uses_typed_top_level_program() {
    let verifier_rs = workspace_root().join("crates/jolt-verifier/src/verifier.rs");
    if !verifier_rs.exists() {
        return;
    }
    let source = std::fs::read_to_string(&verifier_rs).expect("read verifier.rs");
    for pattern in [
        "pub const VERIFIER_PROGRAM",
        "pub const JOLT_VERIFIER_STEPS",
        "pub enum JoltProofSlot",
        "pub enum JoltVerifierCheckpoint",
        "pub type JoltVerifierStepKind = bolt_verifier_runtime::VerifierProgramStepKind",
        "pub type JoltVerifierStepPlan = bolt_verifier_runtime::VerifierProgramStepPlan<JoltProofSlot>",
        "pub type JoltVerifierTargetPlan = bolt_verifier_runtime::VerifierTargetPlan<JoltVerifierCheckpoint>",
        "JoltVerifierStepKind::ReceiveCommitments",
        "JoltVerifierStepKind::VerifySumcheckStage",
        "JoltVerifierStepKind::VerifyPcsOpening",
        "step_count:",
        "bolt_verifier_runtime::execute_verifier_program",
        "fn execute_jolt_verifier_program",
        "fn execute_jolt_verifier_step",
        "struct JoltArtifactStore",
    ] {
        assert!(
            source.contains(pattern),
            "generated verifier.rs is missing typed top-level verifier-program pattern `{pattern}`"
        );
    }
    for stale_pattern in [
        "JoltVerifierTarget::ThroughStage",
        "fn verifies_stage6",
        "fn verifies_stage7",
        "fn verifies_evaluation",
        "fn allows_optional_evaluation",
        "target.verifies_stage",
        "target.allows_optional_evaluation",
    ] {
        assert!(
            !source.contains(stale_pattern),
            "generated verifier.rs still contains stale target-control-flow pattern `{stale_pattern}`"
        );
    }
}

#[test]
fn stage67_output_plan_cutover_removed_obsolete_relation_helpers() {
    let root = workspace_root();
    let generated_relations = root.join("crates/jolt-verifier/src/stages/jolt_relations.rs");
    if !generated_relations.exists() {
        return;
    }

    let relation_sources = [
        generated_relations,
        root.join("crates/bolt/src/protocols/jolt/verifier_jolt_relations.rs.template"),
    ];
    for path in relation_sources {
        let source = std::fs::read_to_string(&path).expect("read Jolt relation source");
        for stale in [
            "expected_stage67_bytecode_read_raf",
            "expected_stage67_booleanity",
            "expected_stage67_hamming_booleanity",
            "expected_stage67_ram_ra_virtual",
            "expected_stage67_instruction_ra_virtual",
            "expected_stage67_inc_claim_reduction",
        ] {
            assert!(
                !source.contains(stale),
                "`{}` still contains obsolete Stage 6/7 relation helper `{stale}`",
                path.display()
            );
        }
    }

    let stage6_source =
        std::fs::read_to_string(root.join("crates/jolt-verifier/src/stages/stage6.rs"))
            .expect("read generated Stage 6 verifier source");
    for stale_field in [
        "booleanity_point",
        "stage5_instruction_ra0",
        "booleanity_combined_point",
        "booleanity_instruction_ra_prefix",
        "booleanity_bytecode_ra_prefix",
        "booleanity_ram_ra_prefix",
        "hamming_weight_eval",
        "hamming_lookup_output",
        "ram_ra_virtual_cycle",
        "ram_ra_virtual_eval_prefix",
        "instruction_ra_virtual_cycle",
        "instruction_ra_virtual_eval_prefix",
        "instruction_ra_virtual_input_prefix",
        "instruction_ra_virtual_gamma",
        "inc_ram_stage2",
        "inc_ram_stage4",
        "inc_rd_stage4",
        "inc_rd_stage5",
        "inc_gamma",
        "inc_ram_eval",
        "inc_rd_eval",
    ] {
        assert!(
            !stage6_source.contains(stale_field),
            "generated Stage 6 relation symbol table still exposes obsolete field `{stale_field}`"
        );
    }
}

#[test]
fn verifier_runtime_has_no_indexed_eval_prefix_api() {
    let runtime = workspace_root().join("crates/bolt-verifier-runtime/src/lib.rs");
    if !runtime.exists() {
        return;
    }
    let source = std::fs::read_to_string(&runtime).expect("read verifier runtime source");
    for stale in ["indexed_evals_by_prefix", "eval_prefix"] {
        assert!(
            !source.contains(stale),
            "bolt-verifier-runtime still exposes indexed eval-prefix API `{stale}`"
        );
    }
}

#[test]
fn verifier_runtime_has_no_name_then_position_eval_fallback() {
    let runtime = workspace_root().join("crates/bolt-verifier-runtime/src/lib.rs");
    if !runtime.exists() {
        return;
    }
    let source = std::fs::read_to_string(&runtime).expect("read verifier runtime source");
    for stale in [
        ".or_else(|| output.evals.get(eval.index))",
        "output.evals.get(eval.index)",
    ] {
        assert!(
            !source.contains(stale),
            "bolt-verifier-runtime still accepts sumcheck evals by position fallback `{stale}`"
        );
    }
}

#[test]
fn verifier_runtime_has_no_jolt_specific_sumcheck_point_orders() {
    let runtime = workspace_root().join("crates/bolt-verifier-runtime/src/lib.rs");
    if !runtime.exists() {
        return;
    }
    let source = std::fs::read_to_string(&runtime).expect("read verifier runtime source");
    for stale in [
        "Stage4RegistersReadWrite",
        "InstructionReadRaf",
        "BytecodeReadRaf",
        "Stage6Booleanity",
        "stage4_registers_rw",
        "instruction_read_raf",
        "bytecode_read_raf",
        "stage6_booleanity",
    ] {
        assert!(
            !source.contains(stale),
            "bolt-verifier-runtime still exposes Jolt-specific point order `{stale}`"
        );
    }
}

#[test]
fn verifier_goal_doc_tracks_extracted_runtime_boundary() {
    let goal = workspace_root().join("crates/bolt/GOAL.md");
    if !goal.exists() {
        return;
    }
    let source = std::fs::read_to_string(&goal).expect("read verifier goal doc");
    assert!(
        source.contains("crates/bolt-verifier-runtime/src/lib.rs"),
        "GOAL.md should name the extracted Bolt verifier runtime crate as Tier A"
    );
    for stale in [
        "crates/jolt-verifier/src/stages/common.rs",
        "crates/bolt/src/protocols/jolt/verifier_common.rs.template",
        "shared verifier plan/runtime scaffolding exists in stages/common.rs",
    ] {
        assert!(
            !source.contains(stale),
            "GOAL.md still references deleted Tier A path `{stale}`"
        );
    }
}

#[test]
fn verifier_cpu_fixtures_are_kernel_free() {
    let fixtures = workspace_root().join("crates/bolt/tests/fixtures");
    if !fixtures.exists() {
        eprintln!("skipping optional verifier MLIR scratch fixture check; run commitment_ir with JOLT_UPDATE_GOLDENS=1 to materialize fixtures");
        return;
    }
    let mut checked = 0usize;
    for path in files_with_extension(&fixtures, "mlir") {
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("fixture file name");
        if !file_name.contains("verifier") {
            continue;
        }
        checked += 1;
        let source = std::fs::read_to_string(&path).expect("read verifier MLIR fixture");
        for pattern in ["kernel = @", "\"cpu.kernel\"", "\"compute.kernel\""] {
            assert!(
                !source.contains(pattern),
                "verifier MLIR fixture `{}` contains forbidden kernel marker `{pattern}`",
                path.display()
            );
        }
    }
    assert!(checked > 0, "no verifier MLIR fixtures were checked");
}

#[test]
fn checked_in_generated_verifier_protocol_symbols_are_allowlisted() {
    let verifier_root = workspace_root().join("crates/jolt-verifier/src");
    if !verifier_root.exists() {
        return;
    }
    let mut checked = 0usize;
    for path in rust_files(&verifier_root) {
        let source = std::fs::read_to_string(&path).expect("read verifier source");
        for symbol in quoted_jolt_protocol_symbols(&source) {
            checked += 1;
            assert_allowed_jolt_protocol_symbol(&path, symbol);
        }
    }
    assert!(
        checked > 0,
        "no generated verifier Jolt symbols were checked"
    );
}

#[test]
fn verifier_mlir_fixtures_protocol_symbols_are_allowlisted() {
    let fixtures = workspace_root().join("crates/bolt/tests/fixtures");
    if !fixtures.exists() {
        eprintln!("skipping optional verifier MLIR scratch symbol check; run commitment_ir with JOLT_UPDATE_GOLDENS=1 to materialize fixtures");
        return;
    }
    let mut checked = 0usize;
    for path in files_with_extension(&fixtures, "mlir") {
        let file_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .expect("fixture file name");
        if !file_name.contains("verifier") {
            continue;
        }
        let source = std::fs::read_to_string(&path).expect("read verifier MLIR fixture");
        for symbol in mlir_jolt_protocol_symbols(&source) {
            checked += 1;
            assert_allowed_jolt_protocol_symbol(&path, symbol);
        }
    }
    assert!(checked > 0, "no verifier MLIR Jolt symbols were checked");
}

#[test]
fn generic_compiler_rejects_jolt_protocol_strings() {
    let root = workspace_root();
    let mut offenders = Vec::new();
    for path in generic_compiler_source_files(&root) {
        let source = std::fs::read_to_string(&path).expect("read generic compiler source");
        let hits = count_generic_compiler_jolt_hits(&source);
        if hits == 0 {
            continue;
        }

        let relative = relative_workspace_path(&root, &path);
        offenders.push(format!("{relative}: {hits} hit(s)"));
    }
    assert!(
        offenders.is_empty(),
        "generic compiler source contains quarantined Jolt protocol strings:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn jolt_artifact_apis_are_quarantined_out_of_generic_exports() {
    let root = workspace_root();
    let artifact_source =
        std::fs::read_to_string(root.join("crates/bolt/src/emit/rust/artifacts.rs"))
            .expect("read generic artifact assembly");
    for pattern in [
        "JoltProtocolStage",
        "JoltArtifactCrate",
        "JoltRustArtifact",
        "JoltGeneratedCrate",
        "JoltGeneratedFile",
        "jolt_artifact_config",
        "jolt_rust_artifact",
        "assemble_jolt_generated_crates",
        "assemble_jolt_workspace_generated_crates",
        "write_jolt_generated_crates",
        "validate_jolt_rust_artifact_imports",
    ] {
        assert!(
            !artifact_source.contains(pattern),
            "generic artifact assembly still exposes quarantined Jolt API `{pattern}`"
        );
    }

    let rust_mod_source = std::fs::read_to_string(root.join("crates/bolt/src/emit/rust/mod.rs"))
        .expect("read Rust emitter exports");
    assert!(
        !rust_mod_source.contains("assemble_jolt_")
            && !rust_mod_source.contains("JoltProtocolStage")
            && !rust_mod_source.contains("jolt_artifact_config"),
        "generic Rust emitter exports still re-export Jolt artifact APIs"
    );

    let lib_source =
        std::fs::read_to_string(root.join("crates/bolt/src/lib.rs")).expect("read bolt lib");
    assert!(
        !lib_source.contains("pub use protocols::jolt"),
        "root bolt exports must keep Jolt APIs under bolt::protocols::jolt"
    );
}

fn verifier_cleanup_metrics(verifier_src: &Path) -> VerifierCleanupMetrics {
    let mut metrics = VerifierCleanupMetrics::default();
    for path in rust_files(verifier_src) {
        let source = std::fs::read_to_string(&path).expect("read verifier source");
        let relative = path
            .strip_prefix(verifier_src)
            .expect("relative verifier path");
        let line_count = source.lines().count();
        metrics.total_loc += line_count;
        if relative == Path::new("stages/common.rs") {
            metrics.bolt_runtime_loc += line_count;
        } else if relative == Path::new("stages/jolt_relations.rs") {
            metrics.jolt_verifier_core_loc += line_count;
        } else {
            metrics.generated_surface_loc += line_count;
        }
        if relative == Path::new("verifier.rs") {
            metrics.verifier_rs_loc = line_count;
        }
        if relative == Path::new("stages/stage6.rs") || relative == Path::new("stages/stage7.rs") {
            metrics.stage6_stage7_loc += line_count;
        }
        if relative.to_string_lossy().starts_with("stages/stage") {
            metrics.stage_local_macro_rules += count_stage_local_macro_rules(&source);
        }
        if relative.starts_with("stages") {
            metrics.stage_local_generic_plan_structs +=
                count_stage_local_generic_plan_structs(&source);
            metrics.field_expr_operand_constants += count_field_expr_operand_constants(&source);
            metrics.batch_operand_string_sites += count_batch_operand_string_sites(&source);
            metrics.claim_input_opening_string_sites +=
                count_claim_input_opening_string_sites(&source);
            metrics.point_concat_input_string_sites +=
                count_point_concat_input_string_sites(&source);
            metrics.stage_local_helper_functions += count_stage_local_helper_functions(&source);
            metrics.relation_string_sites += count_relation_string_sites(&source);
            metrics.sumcheck_point_order_string_sites +=
                count_sumcheck_point_order_string_sites(&source);
            metrics.relation_indexed_eval_prefix_sites +=
                count_relation_indexed_eval_prefix_sites(&source);
            metrics.compute_poly_op_call_sites += count_compute_poly_op_call_sites(&source);
            metrics.compute_point_op_call_sites += count_compute_point_op_call_sites(&source);
            metrics.value_graph_relation_outputs += count_value_graph_relation_outputs(&source);
            metrics.handwritten_expected_output_functions +=
                count_handwritten_expected_output_functions(&source);
        }
    }
    metrics
}

fn count_stage_local_generic_plan_structs(source: &str) -> usize {
    const PLAN_SUFFIXES: &[&str] = &[
        "FieldExprPlan",
        "OpeningClaimPlan",
        "OpeningClaimEqualityPlan",
        "SumcheckClaimPlan",
        "SumcheckDriverPlan",
        "SumcheckEvalPlan",
        "SumcheckInstanceResultPlan",
        "PointSlicePlan",
        "PointConcatPlan",
        "ProgramStepPlan",
        "TranscriptSqueezePlan",
        "TranscriptAbsorbBytesPlan",
        "CpuProgramPlan",
        "VerifierProgramPlan",
        "NamedEval",
    ];
    source
        .lines()
        .filter(|line| {
            let line = line.trim_start();
            if line.starts_with("pub type Stage") && line.contains("bolt_verifier_runtime::") {
                return false;
            }
            (line.starts_with("pub struct Stage") || line.starts_with("pub type Stage"))
                && PLAN_SUFFIXES.iter().any(|suffix| line.contains(suffix))
        })
        .count()
}

fn count_field_expr_operand_constants(source: &str) -> usize {
    source
        .lines()
        .filter(|line| line.contains("FIELD_EXPR_") && line.contains("OPERAND"))
        .count()
}

fn count_batch_operand_string_sites(source: &str) -> usize {
    source
        .lines()
        .filter(|line| line.contains("ordered_claims: \"") || line.contains("claim_operands: \""))
        .count()
}

fn count_claim_input_opening_string_sites(source: &str) -> usize {
    source
        .lines()
        .filter(|line| line.contains("input_openings: \""))
        .count()
}

fn count_point_concat_input_string_sites(source: &str) -> usize {
    source
        .lines()
        .filter(|line| line.contains("PointConcatPlan") && line.contains("inputs: \""))
        .count()
}

fn count_stage_local_macro_rules(source: &str) -> usize {
    source
        .lines()
        .filter(|line| line.trim_start().starts_with("macro_rules!"))
        .count()
}

fn count_stage_local_helper_functions(source: &str) -> usize {
    const HELPER_PREFIXES: &[&str] = &[
        "fn evaluate_stage",
        "fn verify_opening_equalities",
        "fn append_opening_claims",
        "fn find_",
        "fn expected_",
        "fn pow_field",
        "fn single_operand",
        "fn require_operand_count",
    ];
    source
        .lines()
        .filter(|line| {
            let line = line.trim_start();
            HELPER_PREFIXES
                .iter()
                .any(|prefix| line.starts_with(prefix))
        })
        .count()
}

fn count_relation_string_sites(source: &str) -> usize {
    source
        .lines()
        .filter(|line| {
            line.contains("relation: Some(\"jolt.")
                || line.contains("relation: \"jolt.")
                || line.contains("relation == \"jolt.")
        })
        .count()
}

fn count_sumcheck_point_order_string_sites(source: &str) -> usize {
    source
        .lines()
        .filter(|line| line.contains("point_order: \""))
        .count()
}

fn count_relation_indexed_eval_prefix_sites(source: &str) -> usize {
    source
        .lines()
        .filter(|line| line.contains("indexed_evals_by_prefix") || line.contains("eval_prefix"))
        .count()
}

fn count_compute_poly_op_call_sites(source: &str) -> usize {
    source
        .matches("ScalarExprKind::StructuredPolynomial")
        .count()
}

fn count_compute_point_op_call_sites(source: &str) -> usize {
    source
        .lines()
        .filter(|line| line.contains("PointExprKind::") && line.contains("PointExprPlan"))
        .count()
}

fn count_value_graph_relation_outputs(source: &str) -> usize {
    source
        .lines()
        .filter(|line| line.contains("RelationOutputPlan {") && line.contains("expected_output:"))
        .count()
}

fn count_handwritten_expected_output_functions(source: &str) -> usize {
    source
        .lines()
        .filter(|line| line.trim_start().starts_with("fn expected_"))
        .count()
}

fn assert_allowed_jolt_protocol_symbol(path: &Path, symbol: &str) {
    assert!(
        ALLOWED_JOLT_PROTOCOL_SYMBOLS.contains(&symbol),
        "`{}` contains unreviewed Jolt protocol symbol `{symbol}`",
        path.display()
    );
}

fn quoted_jolt_protocol_symbols(source: &str) -> Vec<&str> {
    let mut symbols = Vec::new();
    let mut rest = source;
    while let Some(offset) = rest.find("\"jolt.") {
        let after_quote = &rest[offset + 1..];
        if let Some(end) = after_quote.find('"') {
            symbols.push(&after_quote[..end]);
            rest = &after_quote[end + 1..];
        } else {
            break;
        }
    }
    symbols
}

fn mlir_jolt_protocol_symbols(source: &str) -> Vec<&str> {
    let mut symbols = Vec::new();
    let mut rest = source;
    while let Some(offset) = rest.find("@jolt") {
        let after_at = &rest[offset + 1..];
        let end = after_at
            .char_indices()
            .find_map(|(index, ch)| {
                (!matches!(ch, 'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '.')).then_some(index)
            })
            .unwrap_or(after_at.len());
        symbols.push(&after_at[..end]);
        rest = &after_at[end..];
    }
    symbols
}

fn generic_compiler_source_files(root: &Path) -> Vec<PathBuf> {
    let source_root = root.join("crates/bolt/src");
    let mut files = rust_files(&source_root)
        .into_iter()
        .filter(|path| {
            !relative_workspace_path(root, path).starts_with("crates/bolt/src/protocols/")
        })
        .collect::<Vec<_>>();
    files.sort();
    files
}

fn count_generic_compiler_jolt_hits(source: &str) -> usize {
    source
        .lines()
        .filter(|line| {
            GENERIC_COMPILER_JOLT_PATTERNS
                .iter()
                .any(|pattern| line.contains(pattern))
        })
        .count()
}

fn relative_workspace_path(root: &Path, path: &Path) -> String {
    path.strip_prefix(root)
        .expect("workspace-relative path")
        .to_string_lossy()
        .replace('\\', "/")
}

fn rust_files(root: &Path) -> Vec<PathBuf> {
    files_with_extension(root, "rs")
}

fn files_with_extension(root: &Path, extension: &str) -> Vec<PathBuf> {
    let mut files = Vec::new();
    collect_files_with_extension(root, extension, &mut files);
    files.sort();
    files
}

fn collect_files_with_extension(root: &Path, extension: &str, files: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(root).expect("read directory") {
        let entry = entry.expect("read directory entry");
        let path = entry.path();
        if path.is_dir() {
            collect_files_with_extension(&path, extension, files);
        } else if path.extension().and_then(|ext| ext.to_str()) == Some(extension) {
            files.push(path);
        }
    }
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}
