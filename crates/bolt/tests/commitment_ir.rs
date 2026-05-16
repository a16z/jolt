#![expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "integration tests use explicit panic messages"
)]

use bolt::protocols::jolt::{
    assemble_jolt_generated_crates, assemble_jolt_workspace_generated_crates,
    build_commitment_protocol, build_stage1_outer_protocol, build_stage2_protocol,
    build_stage3_protocol, build_stage4_protocol, build_stage5_protocol, build_stage6_protocol,
    build_stage7_protocol, build_stage8_protocol, commitment_cpu_program, emit_commitment_rust,
    emit_stage1_rust, emit_stage2_rust, emit_stage3_rust, emit_stage4_rust, emit_stage5_rust,
    emit_stage6_rust, emit_stage7_rust, emit_stage8_rust, jolt_artifact_config, jolt_rust_artifact,
    lower_commitment_to_compute, lower_compute_to_cpu, lower_stage1_to_compute,
    lower_stage2_to_compute, lower_stage3_to_compute, lower_stage4_to_compute,
    lower_stage5_to_compute, lower_stage6_to_compute, lower_stage7_to_compute,
    lower_stage8_to_compute, resolve_compute_kernels, stage1_cpu_program, stage2_cpu_program,
    stage3_cpu_program, stage4_cpu_program, stage5_cpu_program, stage6_cpu_program,
    stage7_cpu_program, stage8_cpu_program, validate_jolt_rust_artifact_imports,
    verify_jolt_protocol_schema, write_jolt_generated_crates, JoltGeneratedCrate,
    JoltProtocolParams, JoltProtocolStage,
};
use bolt::{
    assemble_generated_crates, lower_piop_and_fiat_shamir, project_prover_party,
    project_verifier_party, protocol_rust_artifact, validate_rust_artifact_imports,
    verify_compute_schema, verify_concrete_transcript, verify_cpu_schema, verify_protocol_schema,
    Concrete, Cpu, GeneratedFile, MeliorContext, ProtocolArtifactConfig, ProtocolRuntimeModule,
    ProtocolStage, ProtocolStageKind, ProtocolStandaloneDependency, Role, RustSourceFile,
    RustTypeRef, TextMlir,
};
use std::fmt::Write as _;
use std::path::Path;
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

#[test]
fn bolt_irdl_dialects_are_registered() {
    let context = MeliorContext::new();
    assert!(!context.context().allow_unregistered_dialects());

    let registered = r#"
module @registered {
  "field.define"() {modulus_bits = 254 : i64, role = "scalar", sym_name = "bn254_fr"} : () -> ()
}
"#;
    let _ = context
        .parse_module::<bolt::Protocol>(registered)
        .expect("registered dialect op parses");

    let unknown = r#"
module @unknown {
  "unknown.dialect_op"() : () -> ()
}
"#;
    let _ = context
        .parse_module::<bolt::Protocol>(unknown)
        .expect_err("unknown dialect rejected");
}

#[test]
fn commitment_protocol_uses_bolt_semantic_dialects() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let protocol = build_commitment_protocol(&context, &params).expect("build protocol");
    let text = protocol.to_text_mlir();

    assert!(text.contains("\"protocol.params\"()"));
    assert!(text.contains("sym_name = \"jolt.params\""));
    assert!(text.contains("trace_length = 65536"));
    assert!(text.contains("num_committed = 41"));
    assert!(text.contains("\"field.define\"()"));
    assert!(text.contains("sym_name = \"bn254_fr\""));
    assert!(text.contains("\"poly.domain\"()"));
    assert!(text.contains("sym_name = \"jolt.main_witness_commit_domain\""));
    assert!(text.contains("\"protocol.boundary\"()"));
    assert!(text.contains("sym_name = \"jolt.commitment_phase\""));
    assert!(text.contains("\"piop.oracle\"()"));
    assert!(text.contains("sym_name = \"InstructionRa_0\""));
    assert!(text.contains("\"piop.oracle_family\"()"));
    assert!(text.contains("sym_name = \"jolt.main_witness_polys\""));
    assert!(text.contains("ordered_oracles = [@RdInc, @RamInc, @InstructionRa_0"));
    assert!(text.contains("\"commit.publish_batch\"()"));
    assert!(text.contains("\"pcs.commit_batch\"(%"));
    assert!(!text.contains("commitment = @jolt.main_witness_commitments"));
    assert!(text.contains("\"transcript.absorb\"(%"));

    let parsed = context
        .parse_module::<bolt::Protocol>(&text)
        .expect("parse protocol MLIR");
    assert!(parsed.to_text_mlir().contains("\"protocol.boundary\""));
}

#[test]
fn concrete_commitment_phase_threads_transcript_state() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let protocol = build_commitment_protocol(&context, &params).expect("build protocol");
    let concrete =
        lower_piop_and_fiat_shamir(&context, &protocol).expect("lower Fiat-Shamir state");
    verify_concrete_transcript(&concrete).expect("valid transcript state threading");

    let text = concrete.to_text_mlir();
    assert!(text.contains("!transcript.state_type"));
    assert!(text.contains("\"transcript.state\"()"));
    assert!(text.contains("sym_name = \"fs0\""));
    assert!(text.contains("\"transcript.absorb\"(%"));
    assert!(text.contains("\"transcript.absorb_optional\"(%"));
    assert!(!text.contains("in = @fs"));
    assert!(!text.contains("out = @fs"));
}

#[test]
fn transcript_absorb_bytes_threads_and_lowers_to_cpu() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let protocol = context
        .parse_module::<bolt::Protocol>(&transcript_absorb_bytes_protocol(&params))
        .expect("parse absorb-bytes protocol");
    verify_protocol_schema(&protocol).expect("absorb-bytes protocol schema is valid");

    let concrete =
        lower_piop_and_fiat_shamir(&context, &protocol).expect("lower absorb-bytes protocol");
    verify_concrete_transcript(&concrete).expect("absorb-bytes threads transcript state");

    let prover = project_prover_party(&context, &concrete).expect("project prover party");
    // Stage4 does not have its own lowering entrypoint yet; this exercises the
    // shared operation mapping that each stage lowering uses.
    let compute = lower_stage3_to_compute(&context, &prover).expect("lower to compute");
    verify_compute_schema(&compute).expect("compute schema accepts absorb-bytes");
    assert!(compute
        .to_text_mlir()
        .contains("\"compute.transcript_absorb_bytes\"(%"));

    let kernelized = resolve_compute_kernels(&context, &compute).expect("kernelize compute");
    assert!(kernelized
        .to_text_mlir()
        .contains("\"compute.transcript_absorb_bytes\"(%"));

    let cpu = lower_compute_to_cpu(&context, &kernelized).expect("lower to CPU");
    verify_cpu_schema(&cpu).expect("CPU schema accepts absorb-bytes");
    let cpu_text = cpu.to_text_mlir();
    assert!(cpu_text.contains("\"cpu.transcript_absorb_bytes\"(%"));
    assert!(cpu_text.contains("label = \"ram_val_check_gamma\""));
    assert!(cpu_text.contains("payload = \"\""));
}

#[test]
fn concrete_projects_to_party_ir() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let protocol = build_commitment_protocol(&context, &params).expect("build protocol");
    let concrete =
        lower_piop_and_fiat_shamir(&context, &protocol).expect("lower Fiat-Shamir state");
    let prover = project_prover_party(&context, &concrete).expect("project prover party");
    let verifier = project_verifier_party(&context, &concrete).expect("project verifier party");
    let prover_text = prover.to_text_mlir();
    let verifier_text = verifier.to_text_mlir();

    assert!(prover_text.contains("bolt.phase = \"party\""));
    assert!(prover_text.contains("bolt.role = \"prover\""));
    assert!(prover_text.contains("\"party.function\"()"));
    assert!(prover_text.contains("role = \"prover\""));
    assert!(prover_text.contains("\"transcript.absorb\"(%"));
    assert!(!prover_text.contains("in = @fs"));
    assert!(verifier_text.contains("bolt.phase = \"party\""));
    assert!(verifier_text.contains("bolt.role = \"verifier\""));
    assert!(verifier_text.contains("\"party.function\"()"));
    assert!(verifier_text.contains("role = \"verifier\""));
    assert!(verifier_text.contains("\"transcript.absorb\"(%"));
    assert!(!verifier_text.contains("in = @fs"));
}

#[test]
fn commitment_compute_lowers_to_cpu_ir() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let protocol = build_commitment_protocol(&context, &params).expect("build protocol");
    let concrete =
        lower_piop_and_fiat_shamir(&context, &protocol).expect("lower Fiat-Shamir state");
    let prover = project_prover_party(&context, &concrete).expect("project prover party");
    let verifier = project_verifier_party(&context, &concrete).expect("project verifier party");
    let prover_compute = lower_commitment_to_compute(&context, &prover).expect("lower compute");
    let verifier_compute =
        lower_commitment_to_compute(&context, &verifier).expect("lower verifier compute");
    let prover_cpu = lower_compute_to_cpu(&context, &prover_compute).expect("lower to CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_compute).expect("lower verifier to CPU");
    let compute_text = prover_compute.to_text_mlir();
    let text = prover_cpu.to_text_mlir();
    let verifier_compute_text = verifier_compute.to_text_mlir();
    let verifier_text = verifier_cpu.to_text_mlir();

    assert!(compute_text.contains("\"compute.oracle_dense_trace\"()"));
    assert!(compute_text.contains("\"compute.oracle_one_hot_chunk\"()"));
    assert!(compute_text.contains("\"compute.oracle_family_append\"(%"));
    assert!(compute_text.contains("\"compute.pcs_commit_batch\"(%"));
    assert!(compute_text.contains("artifact = @jolt.main_witness_commitments"));
    assert!(compute_text.contains("ordered_oracles = [@RdInc, @RamInc, @InstructionRa_0"));
    assert!(compute_text.contains("\"compute.pcs_commit_optional\"(%"));
    assert!(compute_text.contains("skip_policy = \"missing_or_zero\""));
    assert!(compute_text.contains("!compute.transcript_state"));
    assert!(compute_text.contains("\"compute.transcript_absorb\"(%"));
    assert!(!compute_text.contains("in = @fs"));
    assert!(text.contains("\"cpu.function\"()"));
    assert!(!text.contains("\"compute.function\"()"));
    assert!(text.contains("\"cpu.oracle_family_append\"(%"));
    assert!(text.contains("\"cpu.pcs_commit_batch\"(%"));
    assert!(text.contains("\"cpu.pcs_commit_optional\"(%"));
    assert!(text.contains("skip_policy = \"missing_or_zero\""));
    assert!(text.contains("!cpu.transcript_state"));
    assert!(text.contains("\"cpu.transcript_absorb\"(%"));
    assert!(!text.contains("in = @fs"));
    assert!(verifier_compute_text.contains("\"compute.oracle_ref\"()"));
    assert!(verifier_compute_text.contains("\"compute.pcs_receive_batch\"(%"));
    assert!(verifier_compute_text.contains("\"compute.pcs_receive_optional\"(%"));
    assert!(!verifier_compute_text.contains("\"compute.pcs_commit_batch\"(%"));
    assert!(verifier_text.contains("\"cpu.pcs_receive_batch\"(%"));
    assert!(verifier_text.contains("\"cpu.pcs_receive_optional\"(%"));
    assert!(!verifier_text.contains("\"cpu.pcs_commit_batch\"(%"));

    let parsed = context
        .parse_module::<bolt::Cpu>(&text)
        .expect("parse CPU MLIR");
    assert!(parsed.to_text_mlir().contains("\"cpu.pcs_commit_batch\""));
    let parsed = context
        .parse_module::<bolt::Cpu>(&verifier_text)
        .expect("parse verifier CPU MLIR");
    assert!(parsed.to_text_mlir().contains("\"cpu.pcs_receive_batch\""));
}

#[test]
fn generic_protocol_schema_accepts_non_jolt_params() {
    let context = MeliorContext::new();
    let generic = context.new_module::<bolt::Protocol>("generic", None);
    context
        .append_op(
            &generic,
            "protocol.params",
            Some("generic.params"),
            &[
                ("field", "@some_field"),
                ("pcs", "@some_pcs"),
                ("transcript", "@some_transcript"),
            ],
        )
        .expect("append generic params");

    assert!(
        generic.verify(),
        "generic protocol params pass IRDL verification"
    );
    verify_protocol_schema(&generic).expect("generic schema does not require Jolt attrs");
}

#[test]
fn protocol_schema_rejects_bad_derived_params() {
    let context = MeliorContext::new();
    let bad = context.new_module::<bolt::Protocol>("bad", None);
    let mut attrs = JoltProtocolParams::fixture().attrs();
    for (name, value) in &mut attrs {
        if name == "num_committed" {
            *value = "40 : i64".to_owned();
        }
    }
    context
        .append_op_with_owned_attrs(&bad, "protocol.params", Some("jolt.params"), &attrs)
        .expect("append params");
    context
        .append_op(
            &bad,
            "piop.oracle_family",
            Some("jolt.main_witness_polys"),
            &[
                (
                    "ordered_oracles",
                    "[@RdInc, @RamInc, @InstructionRa_0, @RamRa_0, @BytecodeRa_0]",
                ),
                ("count", "40 : i64"),
                ("domain", "@jolt.trace_domain"),
                ("visibility", r#""committed""#),
            ],
        )
        .expect("append family");

    let error = verify_jolt_protocol_schema(&bad).expect_err("bad derived param rejected");
    assert!(error.to_string().contains("num_committed must be 41"));
}

#[test]
fn concrete_verifier_rejects_unthreaded_transcript_absorb() {
    let context = MeliorContext::new();
    let concrete = context.new_module::<Concrete>("bad", None);
    context
        .append_op(
            &concrete,
            "transcript.absorb",
            Some("bad_absorb"),
            &[
                ("label", r#""commitment""#),
                ("source", "@jolt.main_witness_commitments"),
            ],
        )
        .expect("append bad absorb");

    let error = verify_concrete_transcript(&concrete).expect_err("missing transcript state");
    assert!(error
        .to_string()
        .contains("requires a prior transcript.state result"));
}

#[test]
fn protocol_schema_accepts_explicit_sumcheck_and_opening_flow() {
    let context = MeliorContext::new();
    let protocol = context
        .parse_module::<bolt::Protocol>(explicit_sumcheck_protocol())
        .expect("parse explicit sumcheck protocol");

    verify_protocol_schema(&protocol).expect("explicit sumcheck protocol schema is valid");
    let concrete =
        lower_piop_and_fiat_shamir(&context, &protocol).expect("lower protocol copy to concrete");
    verify_concrete_transcript(&concrete).expect("sumcheck/opening ops thread transcript state");

    let text = concrete.to_text_mlir();
    assert!(text.contains("\"piop.sumcheck_batch\"(%"));
    assert!(text.contains("round_schedule = [2, 1, 1]"));
    assert!(text.contains("\"pcs.opening_claim\"(%"));
    assert!(text.contains("\"pcs.opening_batch\"(%"));
    assert!(text.contains("\"pcs.batch_open\"(%"));
}

#[test]
fn protocol_schema_rejects_eval_family_count_mismatch() {
    let context = MeliorContext::new();
    let protocol = context
        .parse_module::<bolt::Protocol>(&explicit_sumcheck_protocol_with_eval_family(
            2,
            "[@stage1.outer.eval]",
        ))
        .expect("parse protocol with mismatched eval family");

    let error = verify_protocol_schema(&protocol).expect_err("eval family count mismatch");
    assert!(error
        .to_string()
        .contains("piop.sumcheck_eval_family attr `evals` length 1 does not match count 2"));
}

#[test]
fn protocol_schema_rejects_empty_eval_family() {
    let context = MeliorContext::new();
    let protocol = context
        .parse_module::<bolt::Protocol>(&explicit_sumcheck_protocol_with_eval_family(0, "[]"))
        .expect("parse protocol with empty eval family");

    let error = verify_protocol_schema(&protocol).expect_err("empty eval family rejected");
    assert!(error
        .to_string()
        .contains("piop.sumcheck_eval_family attr `evals` must contain at least one symbol"));
}

#[test]
fn opening_batch_schema_rejects_hidden_or_reordered_claims() {
    let context = MeliorContext::new();
    let protocol = context
        .parse_module::<bolt::Protocol>(&explicit_sumcheck_protocol().replace(
            "ordered_claims = [@stage1.outer.opening]",
            "ordered_claims = [@wrong.opening]",
        ))
        .expect("parse explicit sumcheck protocol");

    let error = verify_protocol_schema(&protocol).expect_err("opening batch order mismatch");
    assert!(error
        .to_string()
        .contains("expected @wrong.opening, got @stage1.outer.opening"));
}

#[test]
fn opening_claim_equal_lowers_through_ssa_pipeline() {
    let context = MeliorContext::new();
    let protocol = context
        .parse_module::<bolt::Protocol>(&opening_claim_equal_protocol(
            "LeftInstructionInput",
            "LeftInstructionInput",
            "point_and_eval",
        ))
        .expect("parse opening equality protocol");
    verify_protocol_schema(&protocol).expect("opening equality protocol schema is valid");
    let concrete =
        lower_piop_and_fiat_shamir(&context, &protocol).expect("lower protocol to concrete");
    let prover = project_prover_party(&context, &concrete).expect("project prover party");
    let compute = lower_stage2_to_compute(&context, &prover).expect("lower equality to compute");
    verify_compute_schema(&compute).expect("compute equality schema is valid");
    let kernelized =
        resolve_compute_kernels(&context, &compute).expect("preserve equality through kernels");
    let cpu = lower_compute_to_cpu(&context, &kernelized).expect("lower equality to CPU");
    verify_cpu_schema(&cpu).expect("CPU equality schema is valid");

    let compute_text = kernelized.to_text_mlir();
    let cpu_text = cpu.to_text_mlir();
    assert!(compute_text.contains("\"compute.opening_claim_equal\"(%"));
    assert!(compute_text.contains("mode = \"point_and_eval\""));
    assert!(cpu_text.contains("\"cpu.opening_claim_equal\"(%"));
    assert!(cpu_text.contains("mode = \"point_and_eval\""));
}

#[test]
fn opening_claim_equal_rejects_incompatible_claim_metadata() {
    let context = MeliorContext::new();
    let protocol = context
        .parse_module::<bolt::Protocol>(&opening_claim_equal_protocol(
            "LeftInstructionInput",
            "RightInstructionInput",
            "point_and_eval",
        ))
        .expect("parse bad opening equality protocol");

    let error = verify_protocol_schema(&protocol).expect_err("mismatched claims are rejected");
    assert!(error.to_string().contains("compares incompatible claims"));
}

#[test]
fn opening_claim_equal_rejects_unsupported_mode() {
    let context = MeliorContext::new();
    let protocol = context
        .parse_module::<bolt::Protocol>(&opening_claim_equal_protocol(
            "LeftInstructionInput",
            "LeftInstructionInput",
            "eval_only",
        ))
        .expect("parse bad opening equality mode");

    let error = verify_protocol_schema(&protocol).expect_err("unsupported equality mode rejected");
    assert!(error.to_string().contains("expected \"point_and_eval\""));
}

#[test]
fn sumcheck_compute_lowers_to_cpu_kernel_ir() {
    let context = MeliorContext::new();
    let compute = context
        .parse_module::<bolt::Compute>(explicit_sumcheck_compute())
        .expect("parse explicit sumcheck compute");

    verify_compute_schema(&compute).expect("compute sumcheck schema is valid");
    let kernelized =
        resolve_compute_kernels(&context, &compute).expect("resolve sumcheck compute kernels");
    verify_compute_schema(&kernelized).expect("kernelized sumcheck schema is valid");
    let cpu = lower_compute_to_cpu(&context, &kernelized).expect("lower sumcheck compute to CPU");
    verify_cpu_schema(&cpu).expect("CPU sumcheck schema is valid");

    let text = cpu.to_text_mlir();
    assert!(text.contains("\"cpu.transcript_squeeze\"(%"));
    assert!(text.contains("\"cpu.sumcheck_batch\"(%"));
    assert!(text.contains("\"cpu.sumcheck_driver\"(%"));
    assert!(text.contains("\"cpu.sumcheck_eval\"(%"));
    assert!(text.contains("\"cpu.pcs_opening_claim\"(%"));
    assert!(text.contains("\"cpu.pcs_batch_open\"(%"));
    assert!(text.contains("!cpu.sumcheck_claim_type"));
}

#[test]
fn jolt_stage1_outer_protocol_defines_virtual_claim_flow() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let protocol =
        build_stage1_outer_protocol(&context, &params).expect("build stage1 outer protocol");
    verify_protocol_schema(&protocol).expect("stage1 protocol schema is valid");
    let concrete =
        lower_piop_and_fiat_shamir(&context, &protocol).expect("lower stage1 to concrete");
    verify_concrete_transcript(&concrete).expect("stage1 transcript is threaded");

    let text = protocol.to_text_mlir();
    assert!(text.contains("sym_name = \"stage1.uniskip.sumcheck\""));
    assert!(text.contains("sym_name = \"stage1.outer_remaining.sumcheck\""));
    assert!(text.contains("relation = @jolt.stage1.outer.uniskip"));
    assert!(!text.contains("kernel = @"));
    assert!(text.contains("\"piop.sumcheck_claim\"(%"));
    assert!(text.contains("\"piop.sumcheck_eval\"(%"));
    assert!(text.contains("\"piop.opening_claim\"(%"));
    assert!(text.contains("\"piop.opening_batch\"(%"));
    assert!(text.contains("count = 35 : i64"));
    assert!(text.contains("ordered_claims = [@stage1.outer_remaining.opening.LeftInstructionInput"));
    assert!(text.contains("oracle = @OpFlagIsLastInSequence"));
    assert!(!text.contains("\"pcs.opening_claim\""));
    assert_or_update_fixture("tests/fixtures/stage1_outer_protocol.mlir", &text);
}

#[test]
fn jolt_stage2_protocol_defines_product_ram_claim_flow() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let protocol = build_stage2_protocol(&context, &params).expect("build stage2 protocol");
    verify_protocol_schema(&protocol).expect("stage2 protocol schema is valid");
    let concrete =
        lower_piop_and_fiat_shamir(&context, &protocol).expect("lower stage2 to concrete");
    verify_concrete_transcript(&concrete).expect("stage2 transcript is threaded");

    let text = protocol.to_text_mlir();
    assert!(text.contains("sym_name = \"stage2.product_virtual.uniskip.sumcheck\""));
    assert!(text.contains("sym_name = \"stage2.sumcheck\""));
    assert!(text.contains("relation = @jolt.stage2.product_virtual.uniskip"));
    assert!(text.contains("relation = @jolt.stage2.batched"));
    assert!(text.contains("\"piop.opening_input\"()"));
    assert!(text.contains("\"field.add\"(%"));
    assert!(text.contains("\"poly.lagrange_basis_eval\"(%"));
    assert!(text.contains("sym_name = \"stage2.ram_read_write.claim_expr\""));
    assert!(text.contains("\"piop.sumcheck_instance_result\"(%"));
    assert!(text.contains("round_offset = 16 : i64"));
    assert!(text.contains("\"poly.point_slice\"(%"));
    assert!(text.contains("\"poly.point_concat\"(%"));
    assert!(text.contains(
        "ordered_claims = [@stage2.ram_read_write.input, @stage2.product_virtual.remainder.input"
    ));
    assert!(text.contains("ordered_claims = [@stage2.ram_read_write.opening.RamVal, @stage2.ram_read_write.opening.RamRa, @stage2.ram_read_write.opening.RamInc"));
    assert!(text.contains("claim_kind = \"committed\""));
    assert!(text.contains("source_claim = @stage1.outer_remaining.opening.RamAddress"));
    assert!(!text.contains("kernel = @"));
    assert!(!text.contains("\"compute."));
    assert_or_update_fixture("tests/fixtures/stage2_protocol.mlir", &text);
}

#[test]
fn jolt_stage2_lowers_to_compute_and_cpu_role_ir() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let protocol = build_stage2_protocol(&context, &params).expect("build stage2 protocol");
    let concrete =
        lower_piop_and_fiat_shamir(&context, &protocol).expect("lower stage2 to concrete");
    let prover = project_prover_party(&context, &concrete).expect("project prover party");
    let verifier = project_verifier_party(&context, &concrete).expect("project verifier party");
    let prover_compute = lower_stage2_to_compute(&context, &prover).expect("lower prover stage2");
    let verifier_compute =
        lower_stage2_to_compute(&context, &verifier).expect("lower verifier stage2");
    verify_compute_schema(&prover_compute).expect("prover stage2 compute schema is valid");
    verify_compute_schema(&verifier_compute).expect("verifier stage2 compute schema is valid");

    let prover_compute_text = prover_compute.to_text_mlir();
    let verifier_compute_text = verifier_compute.to_text_mlir();
    assert!(prover_compute_text.contains("\"compute.opening_input\"()"));
    assert!(prover_compute_text.contains("\"compute.field_add\"(%"));
    assert!(prover_compute_text.contains("\"compute.poly_lagrange_basis_eval\"(%"));
    assert!(prover_compute_text.contains("\"compute.point_slice\"(%"));
    assert!(prover_compute_text.contains("\"compute.point_concat\"(%"));
    assert!(prover_compute_text.contains("\"compute.sumcheck_claim\"(%"));
    assert!(prover_compute_text.contains("\"compute.sumcheck_driver\"(%"));
    assert!(!prover_compute_text.contains("kernel = @"));
    assert!(verifier_compute_text.contains("\"compute.sumcheck_verify_claim\""));
    assert!(verifier_compute_text.contains("\"compute.sumcheck_verify\""));
    assert!(!verifier_compute_text.contains("\"compute.kernel\""));
    assert!(!verifier_compute_text.contains("kernel = @"));

    let prover_kernel_compute =
        resolve_compute_kernels(&context, &prover_compute).expect("resolve prover kernels");
    let verifier_kernel_compute =
        resolve_compute_kernels(&context, &verifier_compute).expect("resolve verifier kernels");
    verify_compute_schema(&prover_kernel_compute)
        .expect("prover kernelized stage2 compute schema is valid");
    verify_compute_schema(&verifier_kernel_compute)
        .expect("verifier kernelized stage2 compute schema is valid");

    let prover_cpu =
        lower_compute_to_cpu(&context, &prover_kernel_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_kernel_compute).expect("lower verifier CPU");
    verify_cpu_schema(&prover_cpu).expect("prover stage2 CPU schema is valid");
    verify_cpu_schema(&verifier_cpu).expect("verifier stage2 CPU schema is valid");

    let prover_cpu_text = prover_cpu.to_text_mlir();
    let verifier_cpu_text = verifier_cpu.to_text_mlir();
    assert!(prover_cpu_text.contains("\"cpu.opening_input\"()"));
    assert!(prover_cpu_text.contains("\"cpu.field_add\"(%"));
    assert!(prover_cpu_text.contains("\"cpu.poly_lagrange_basis_eval\"(%"));
    assert!(prover_cpu_text.contains("\"cpu.point_slice\"(%"));
    assert!(prover_cpu_text.contains("\"cpu.point_concat\"(%"));
    assert!(prover_cpu_text.contains("\"cpu.kernel\"()"));
    assert!(prover_cpu_text.contains("kernel = @jolt.cpu.stage2.batched"));
    assert!(verifier_cpu_text.contains("\"cpu.opening_input\"()"));
    assert!(verifier_cpu_text.contains("\"cpu.sumcheck_verify_claim\""));
    assert!(verifier_cpu_text.contains("\"cpu.sumcheck_verify\""));
    assert!(!verifier_cpu_text.contains("\"cpu.kernel\""));
    assert!(!verifier_cpu_text.contains("kernel = @"));

    assert_or_update_fixture(
        "tests/fixtures/stage2_prover_compute.mlir",
        &prover_compute.to_text_mlir(),
    );
    assert_or_update_fixture(
        "tests/fixtures/stage2_verifier_compute.mlir",
        &verifier_compute.to_text_mlir(),
    );
    assert_or_update_fixture(
        "tests/fixtures/stage2_prover_kernel_compute.mlir",
        &prover_kernel_compute.to_text_mlir(),
    );
    assert_or_update_fixture(
        "tests/fixtures/stage2_verifier_kernel_compute.mlir",
        &verifier_kernel_compute.to_text_mlir(),
    );
    assert_or_update_fixture("tests/fixtures/stage2_prover_cpu.mlir", &prover_cpu_text);
    assert_or_update_fixture(
        "tests/fixtures/stage2_verifier_cpu.mlir",
        &verifier_cpu_text,
    );
}

#[test]
fn jolt_stage3_protocol_defines_shift_instruction_register_flow() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let protocol = build_stage3_protocol(&context, &params).expect("build stage3 protocol");
    verify_protocol_schema(&protocol).expect("stage3 protocol schema is valid");
    let concrete =
        lower_piop_and_fiat_shamir(&context, &protocol).expect("lower stage3 to concrete");
    verify_concrete_transcript(&concrete).expect("stage3 transcript is threaded");

    let text = protocol.to_text_mlir();
    assert!(text.contains("sym_name = \"stage3.spartan_shift.input\""));
    assert!(text.contains("sym_name = \"stage3.instruction_input.input\""));
    assert!(text.contains("sym_name = \"stage3.registers_claim_reduction.input\""));
    assert!(text.contains("\"piop.opening_claim_equal\"(%"));
    assert!(text.contains("\"field.add\"(%"));
    assert!(text.contains("\"field.mul\"(%"));
    assert!(text.contains("\"field.sub\"(%"));
    assert!(text.contains("policy = \"jolt_core_stage3_aligned\""));
    assert!(text.contains("point_order = \"reverse\""));
    assert!(text.contains("ordered_claims = [@stage3.spartan_shift.input, @stage3.instruction_input.input, @stage3.registers_claim_reduction.input]"));
    assert!(text.contains("ordered_claims = [@stage3.spartan_shift.opening.UnexpandedPC, @stage3.spartan_shift.opening.PC"));
    assert!(!text.contains("kernel = @"));
    assert!(!text.contains("\"compute."));
    assert_or_update_fixture("tests/fixtures/stage3_protocol.mlir", &text);
}

#[test]
fn jolt_stage3_lowers_to_compute_and_cpu_role_ir() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let protocol = build_stage3_protocol(&context, &params).expect("build stage3 protocol");
    let concrete =
        lower_piop_and_fiat_shamir(&context, &protocol).expect("lower stage3 to concrete");
    let prover = project_prover_party(&context, &concrete).expect("project prover party");
    let verifier = project_verifier_party(&context, &concrete).expect("project verifier party");
    let prover_compute = lower_stage3_to_compute(&context, &prover).expect("lower prover stage3");
    let verifier_compute =
        lower_stage3_to_compute(&context, &verifier).expect("lower verifier stage3");
    verify_compute_schema(&prover_compute).expect("prover stage3 compute schema is valid");
    verify_compute_schema(&verifier_compute).expect("verifier stage3 compute schema is valid");

    let prover_compute_text = prover_compute.to_text_mlir();
    let verifier_compute_text = verifier_compute.to_text_mlir();
    assert!(prover_compute_text.contains("\"compute.opening_input\"()"));
    assert!(prover_compute_text.contains("\"compute.opening_claim_equal\"(%"));
    assert!(prover_compute_text.contains("\"compute.field_add\"(%"));
    assert!(prover_compute_text.contains("\"compute.field_mul\"(%"));
    assert!(prover_compute_text.contains("\"compute.sumcheck_claim\"(%"));
    assert!(prover_compute_text.contains("\"compute.sumcheck_driver\"(%"));
    assert!(!prover_compute_text.contains("kernel = @"));
    assert!(verifier_compute_text.contains("\"compute.opening_claim_equal\"(%"));
    assert!(verifier_compute_text.contains("\"compute.sumcheck_verify_claim\""));
    assert!(verifier_compute_text.contains("\"compute.sumcheck_verify\""));
    assert!(!verifier_compute_text.contains("\"compute.kernel\""));
    assert!(!verifier_compute_text.contains("kernel = @"));

    let prover_kernel_compute =
        resolve_compute_kernels(&context, &prover_compute).expect("resolve prover kernels");
    let verifier_kernel_compute =
        resolve_compute_kernels(&context, &verifier_compute).expect("resolve verifier kernels");
    verify_compute_schema(&prover_kernel_compute)
        .expect("prover kernelized stage3 compute schema is valid");
    verify_compute_schema(&verifier_kernel_compute)
        .expect("verifier kernelized stage3 compute schema is valid");

    let prover_cpu =
        lower_compute_to_cpu(&context, &prover_kernel_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_kernel_compute).expect("lower verifier CPU");
    verify_cpu_schema(&prover_cpu).expect("prover stage3 CPU schema is valid");
    verify_cpu_schema(&verifier_cpu).expect("verifier stage3 CPU schema is valid");

    let prover_kernel_text = prover_kernel_compute.to_text_mlir();
    let verifier_kernel_text = verifier_kernel_compute.to_text_mlir();
    let prover_cpu_text = prover_cpu.to_text_mlir();
    let verifier_cpu_text = verifier_cpu.to_text_mlir();
    assert!(prover_kernel_text.contains("kernel = @jolt.cpu.stage3.batched"));
    assert!(!verifier_kernel_text.contains("kernel = @"));
    assert!(prover_cpu_text.contains("\"cpu.opening_claim_equal\"(%"));
    assert!(prover_cpu_text.contains("\"cpu.kernel\"()"));
    assert!(prover_cpu_text.contains("kernel = @jolt.cpu.stage3.batched"));
    assert!(verifier_cpu_text.contains("\"cpu.opening_claim_equal\"(%"));
    assert!(verifier_cpu_text.contains("\"cpu.sumcheck_verify_claim\""));
    assert!(verifier_cpu_text.contains("\"cpu.sumcheck_verify\""));
    assert!(!verifier_cpu_text.contains("\"cpu.kernel\""));
    assert!(!verifier_cpu_text.contains("kernel = @"));

    assert_or_update_fixture(
        "tests/fixtures/stage3_prover_compute.mlir",
        &prover_compute_text,
    );
    assert_or_update_fixture(
        "tests/fixtures/stage3_verifier_compute.mlir",
        &verifier_compute_text,
    );
    assert_or_update_fixture(
        "tests/fixtures/stage3_prover_kernel_compute.mlir",
        &prover_kernel_text,
    );
    assert_or_update_fixture(
        "tests/fixtures/stage3_verifier_kernel_compute.mlir",
        &verifier_kernel_text,
    );
    assert_or_update_fixture("tests/fixtures/stage3_prover_cpu.mlir", &prover_cpu_text);
    assert_or_update_fixture(
        "tests/fixtures/stage3_verifier_cpu.mlir",
        &verifier_cpu_text,
    );
}

#[test]
fn jolt_stage4_protocol_defines_registers_and_ram_val_flow() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let protocol = build_stage4_protocol(&context, &params).expect("build stage4 protocol");
    verify_protocol_schema(&protocol).expect("stage4 protocol schema is valid");
    let concrete =
        lower_piop_and_fiat_shamir(&context, &protocol).expect("lower stage4 to concrete");
    verify_concrete_transcript(&concrete).expect("stage4 transcript is threaded");

    let text = protocol.to_text_mlir();
    assert!(text.contains("sym_name = \"stage4.registers_read_write.input\""));
    assert!(text.contains("sym_name = \"stage4.ram_val_check.input\""));
    assert!(text.contains("\"transcript.absorb_bytes\"(%"));
    assert!(text.contains("label = \"ram_val_check_gamma\""));
    assert!(text.contains("payload = \"\""));
    assert!(text.contains("sym_name = \"stage4.input.initial_ram.RamValInit\""));
    assert!(text.contains("sym_name = \"stage4.registers.rs1_claim_consistency\""));
    assert!(text.contains("sym_name = \"stage4.registers.rs2_claim_consistency\""));
    assert!(text.contains(
        "ordered_claims = [@stage4.registers_read_write.input, @stage4.ram_val_check.input]"
    ));
    assert!(text.contains("ordered_claims = [@stage4.registers_read_write.opening.RegistersVal"));
    assert!(text.contains("@stage4.ram_val_check.opening.RamRa"));
    assert!(text.contains("@stage4.ram_val_check.opening.RamInc"));
    assert!(!text.contains("kernel = @"));
    assert!(!text.contains("\"compute."));
    assert_or_update_fixture("tests/fixtures/stage4_protocol.mlir", &text);
}

#[test]
fn jolt_stage4_lowers_to_compute_and_cpu_role_ir() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let protocol = build_stage4_protocol(&context, &params).expect("build stage4 protocol");
    let concrete =
        lower_piop_and_fiat_shamir(&context, &protocol).expect("lower stage4 to concrete");
    let prover = project_prover_party(&context, &concrete).expect("project prover party");
    let verifier = project_verifier_party(&context, &concrete).expect("project verifier party");
    let prover_compute = lower_stage4_to_compute(&context, &prover).expect("lower prover stage4");
    let verifier_compute =
        lower_stage4_to_compute(&context, &verifier).expect("lower verifier stage4");
    verify_compute_schema(&prover_compute).expect("prover stage4 compute schema is valid");
    verify_compute_schema(&verifier_compute).expect("verifier stage4 compute schema is valid");

    let prover_compute_text = prover_compute.to_text_mlir();
    let verifier_compute_text = verifier_compute.to_text_mlir();
    assert!(prover_compute_text.contains("\"compute.transcript_absorb_bytes\"(%"));
    assert!(prover_compute_text.contains("\"compute.opening_claim_equal\"(%"));
    assert!(prover_compute_text.contains("\"compute.field_sub\"(%"));
    assert!(prover_compute_text.contains("\"compute.sumcheck_claim\"(%"));
    assert!(prover_compute_text.contains("\"compute.sumcheck_driver\"(%"));
    assert!(verifier_compute_text.contains("\"compute.transcript_absorb_bytes\"(%"));
    assert!(verifier_compute_text.contains("\"compute.sumcheck_verify_claim\""));
    assert!(verifier_compute_text.contains("\"compute.sumcheck_verify\""));
    assert!(!verifier_compute_text.contains("kernel = @"));

    let prover_kernel_compute =
        resolve_compute_kernels(&context, &prover_compute).expect("resolve prover kernels");
    let verifier_kernel_compute =
        resolve_compute_kernels(&context, &verifier_compute).expect("resolve verifier kernels");
    verify_compute_schema(&prover_kernel_compute)
        .expect("prover kernelized stage4 compute schema is valid");
    verify_compute_schema(&verifier_kernel_compute)
        .expect("verifier kernelized stage4 compute schema is valid");

    let prover_cpu =
        lower_compute_to_cpu(&context, &prover_kernel_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_kernel_compute).expect("lower verifier CPU");
    verify_cpu_schema(&prover_cpu).expect("prover stage4 CPU schema is valid");
    verify_cpu_schema(&verifier_cpu).expect("verifier stage4 CPU schema is valid");

    let prover_kernel_text = prover_kernel_compute.to_text_mlir();
    let verifier_kernel_text = verifier_kernel_compute.to_text_mlir();
    let prover_cpu_text = prover_cpu.to_text_mlir();
    let verifier_cpu_text = verifier_cpu.to_text_mlir();
    assert!(prover_kernel_text.contains("kernel = @jolt.cpu.stage4.batched"));
    assert!(prover_cpu_text.contains("\"cpu.transcript_absorb_bytes\"(%"));
    assert!(prover_cpu_text.contains("kernel = @jolt.cpu.stage4.batched"));
    assert!(verifier_cpu_text.contains("\"cpu.transcript_absorb_bytes\"(%"));
    assert!(verifier_cpu_text.contains("\"cpu.sumcheck_verify_claim\""));
    assert!(!verifier_kernel_text.contains("kernel = @"));
    assert!(!verifier_cpu_text.contains("kernel = @"));

    assert_or_update_fixture(
        "tests/fixtures/stage4_prover_compute.mlir",
        &prover_compute_text,
    );
    assert_or_update_fixture(
        "tests/fixtures/stage4_verifier_compute.mlir",
        &verifier_compute_text,
    );
    assert_or_update_fixture(
        "tests/fixtures/stage4_prover_kernel_compute.mlir",
        &prover_kernel_text,
    );
    assert_or_update_fixture(
        "tests/fixtures/stage4_verifier_kernel_compute.mlir",
        &verifier_kernel_text,
    );
    assert_or_update_fixture("tests/fixtures/stage4_prover_cpu.mlir", &prover_cpu_text);
    assert_or_update_fixture(
        "tests/fixtures/stage4_verifier_cpu.mlir",
        &verifier_cpu_text,
    );
}

#[test]
fn jolt_stage5_protocol_defines_value_lookup_reduction_flow() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let protocol = build_stage5_protocol(&context, &params).expect("build stage5 protocol");
    verify_protocol_schema(&protocol).expect("stage5 protocol schema is valid");
    let concrete =
        lower_piop_and_fiat_shamir(&context, &protocol).expect("lower stage5 to concrete");
    verify_concrete_transcript(&concrete).expect("stage5 transcript is threaded");

    let text = protocol.to_text_mlir();
    assert!(text.contains("sym_name = \"stage5.instruction_read_raf.input\""));
    assert!(text.contains("sym_name = \"stage5.ram_ra_claim_reduction.input\""));
    assert!(text.contains("sym_name = \"stage5.registers_val_evaluation.input\""));
    assert!(text.contains("sym_name = \"stage5.instruction_read_raf.gamma\""));
    assert!(text.contains("sym_name = \"stage5.ram_ra_claim_reduction.gamma\""));
    assert!(text.contains("sym_name = \"stage5.instruction.lookup_output_claim_consistency\""));
    assert!(text.contains("round_schedule = [128, 16]"));
    assert!(text.contains("ordered_claims = [@stage5.instruction_read_raf.input, @stage5.ram_ra_claim_reduction.input, @stage5.registers_val_evaluation.input]"));
    assert!(text.contains("@stage5.instruction_read_raf.opening.LookupTableFlag_0"));
    assert!(text.contains("@stage5.instruction_read_raf.opening.InstructionRa_0"));
    assert!(text.contains("@stage5.instruction_read_raf.opening.InstructionRafFlag"));
    assert!(text.contains("@stage5.ram_ra_claim_reduction.opening.RamRa"));
    assert!(text.contains("@stage5.registers_val_evaluation.opening.RdInc"));
    assert!(text.contains("@stage5.registers_val_evaluation.opening.RdWa"));
    assert!(text.contains("sym_name = \"stage5.ram_ra_claim_reduction.output.eq.Raf\""));
    assert!(text.contains("sym_name = \"stage5.ram_ra_claim_reduction.output.claim\""));
    assert!(
        text.contains("sym_name = \"stage5.registers_val_evaluation.output.lt.RegistersValCycle\"")
    );
    assert!(text.contains("sym_name = \"stage5.registers_val_evaluation.output.claim\""));
    assert!(!text.contains("kernel = @"));
    assert!(!text.contains("\"compute."));
}

#[test]
fn jolt_stage5_lowers_to_compute_and_cpu_role_ir() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let protocol = build_stage5_protocol(&context, &params).expect("build stage5 protocol");
    let concrete =
        lower_piop_and_fiat_shamir(&context, &protocol).expect("lower stage5 to concrete");
    let prover = project_prover_party(&context, &concrete).expect("project prover party");
    let verifier = project_verifier_party(&context, &concrete).expect("project verifier party");
    let prover_compute = lower_stage5_to_compute(&context, &prover).expect("lower prover stage5");
    let verifier_compute =
        lower_stage5_to_compute(&context, &verifier).expect("lower verifier stage5");
    verify_compute_schema(&prover_compute).expect("prover stage5 compute schema is valid");
    verify_compute_schema(&verifier_compute).expect("verifier stage5 compute schema is valid");

    let prover_compute_text = prover_compute.to_text_mlir();
    let verifier_compute_text = verifier_compute.to_text_mlir();
    assert!(prover_compute_text.contains("\"compute.opening_claim_equal\"(%"));
    assert!(prover_compute_text.contains("\"compute.field_pow\"(%"));
    assert!(prover_compute_text.contains("\"compute.sumcheck_claim\"(%"));
    assert!(prover_compute_text.contains("\"compute.sumcheck_driver\"(%"));
    assert!(verifier_compute_text.contains("\"compute.sumcheck_verify_claim\""));
    assert!(verifier_compute_text.contains("\"compute.sumcheck_verify\""));
    assert!(!verifier_compute_text.contains("kernel = @"));

    let prover_kernel_compute =
        resolve_compute_kernels(&context, &prover_compute).expect("resolve prover kernels");
    let verifier_kernel_compute =
        resolve_compute_kernels(&context, &verifier_compute).expect("resolve verifier kernels");
    verify_compute_schema(&prover_kernel_compute)
        .expect("prover kernelized stage5 compute schema is valid");
    verify_compute_schema(&verifier_kernel_compute)
        .expect("verifier kernelized stage5 compute schema is valid");

    let prover_cpu =
        lower_compute_to_cpu(&context, &prover_kernel_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_kernel_compute).expect("lower verifier CPU");
    verify_cpu_schema(&prover_cpu).expect("prover stage5 CPU schema is valid");
    verify_cpu_schema(&verifier_cpu).expect("verifier stage5 CPU schema is valid");

    let prover_kernel_text = prover_kernel_compute.to_text_mlir();
    let verifier_kernel_text = verifier_kernel_compute.to_text_mlir();
    let prover_cpu_text = prover_cpu.to_text_mlir();
    let verifier_cpu_text = verifier_cpu.to_text_mlir();
    assert!(prover_kernel_text.contains("kernel = @jolt.cpu.stage5.batched"));
    assert!(prover_cpu_text.contains("kernel = @jolt.cpu.stage5.batched"));
    assert!(prover_cpu_text.contains("point_order = \"instruction_read_raf\""));
    assert!(verifier_cpu_text.contains("\"cpu.structured_polynomial_eval\""));
    assert!(verifier_cpu_text.contains("\"cpu.sumcheck_output_claim\""));
    assert!(verifier_cpu_text.contains("\"cpu.sumcheck_verify_claim\""));
    assert!(!verifier_kernel_text.contains("kernel = @"));
    assert!(!verifier_cpu_text.contains("kernel = @"));
}

#[test]
fn jolt_stage6_protocol_defines_bytecode_booleanity_and_virtualization_flow() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let protocol = build_stage6_protocol(&context, &params).expect("build stage6 protocol");
    verify_protocol_schema(&protocol).expect("stage6 protocol schema is valid");
    let concrete =
        lower_piop_and_fiat_shamir(&context, &protocol).expect("lower stage6 to concrete");
    verify_concrete_transcript(&concrete).expect("stage6 transcript is threaded");

    let text = protocol.to_text_mlir();
    assert!(text.contains("sym_name = \"stage6.bytecode_read_raf.input\""));
    assert!(text.contains("sym_name = \"stage6.booleanity.input\""));
    assert!(text.contains("sym_name = \"stage6.hamming_booleanity.input\""));
    assert!(text.contains("sym_name = \"stage6.ram_ra_virtual.input\""));
    assert!(text.contains("sym_name = \"stage6.instruction_ra_virtual.input\""));
    assert!(text.contains("sym_name = \"stage6.inc_claim_reduction.input\""));
    assert!(text.contains("sym_name = \"stage6.bytecode_read_raf.gamma\""));
    assert!(text.contains("sym_name = \"stage6.bytecode_read_raf.stage5_gamma\""));
    assert!(text.contains("sym_name = \"stage6.booleanity.gamma\""));
    assert!(text.contains("sym_name = \"stage6.instruction_ra_virtual.gamma\""));
    assert!(text.contains("sym_name = \"stage6.inc_claim_reduction.gamma\""));
    assert!(text.contains("source_claim = @stage2.ram_read_write.opening.RamInc"));
    assert!(text.contains("source_claim = @stage4.registers_read_write.opening.RdInc"));
    assert!(text.contains("source_claim = @stage5.registers_val_evaluation.opening.RdInc"));
    assert!(text.contains("round_schedule = [10, 16]"));
    assert!(text.contains("ordered_claims = [@stage6.bytecode_read_raf.input, @stage6.booleanity.input, @stage6.hamming_booleanity.input, @stage6.ram_ra_virtual.input, @stage6.instruction_ra_virtual.input, @stage6.inc_claim_reduction.input]"));
    assert!(text.contains("@stage6.bytecode_read_raf.opening.BytecodeRa_0"));
    assert!(text.contains("@stage6.booleanity.opening.InstructionRa_0"));
    assert!(text.contains("@stage6.hamming_booleanity.opening.HammingWeight"));
    assert!(text.contains("@stage6.ram_ra_virtual.opening.RamRa_0"));
    assert!(text.contains("@stage6.instruction_ra_virtual.opening.InstructionRa_0"));
    assert!(text.contains("@stage6.inc_claim_reduction.opening.RamInc"));
    assert!(text.contains("@stage6.inc_claim_reduction.opening.RdInc"));
    assert!(text.contains("sym_name = \"stage6.booleanity.output.claim\""));
    assert!(text.contains("sym_name = \"stage6.hamming_booleanity.output.claim\""));
    assert!(text.contains("sym_name = \"stage6.ram_ra_virtual.output.claim\""));
    assert!(text.contains("sym_name = \"stage6.instruction_ra_virtual.output.claim\""));
    assert!(text.contains("sym_name = \"stage6.inc_claim_reduction.output.claim\""));
    assert!(text.contains("sym_name = \"stage6.booleanity.output.family\""));
    assert!(text.contains("sym_name = \"stage6.hamming_booleanity.output.family\""));
    assert!(text.contains("sym_name = \"stage6.ram_ra_virtual.output.family\""));
    assert!(text.contains("sym_name = \"stage6.instruction_ra_virtual.output.family\""));
    assert!(text.contains("sym_name = \"stage6.inc_claim_reduction.output.family\""));
    assert!(text.contains("\"piop.sumcheck_output_function_family\""));
    assert!(text.contains("\"piop.sumcheck_output_product_family\""));
    assert!(text.contains("\"piop.sumcheck_output_eval_family\""));
    assert!(text.contains("sym_name = \"stage6.inc_claim_reduction.output.eq.RdIncStage5\""));
    assert!(text.contains("sym_name = \"stage6.booleanity.output.eq.InstructionRa0\""));
    assert!(!text.contains("kernel = @"));
    assert!(!text.contains("\"compute."));
}

#[test]
fn jolt_stage6_lowers_to_compute_and_cpu_role_ir() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let protocol = build_stage6_protocol(&context, &params).expect("build stage6 protocol");
    let concrete =
        lower_piop_and_fiat_shamir(&context, &protocol).expect("lower stage6 to concrete");
    let prover = project_prover_party(&context, &concrete).expect("project prover party");
    let verifier = project_verifier_party(&context, &concrete).expect("project verifier party");
    let prover_compute = lower_stage6_to_compute(&context, &prover).expect("lower prover stage6");
    let verifier_compute =
        lower_stage6_to_compute(&context, &verifier).expect("lower verifier stage6");
    verify_compute_schema(&prover_compute).expect("prover stage6 compute schema is valid");
    verify_compute_schema(&verifier_compute).expect("verifier stage6 compute schema is valid");

    let prover_compute_text = prover_compute.to_text_mlir();
    let verifier_compute_text = verifier_compute.to_text_mlir();
    assert!(prover_compute_text.contains("\"compute.field_pow\"(%"));
    assert!(prover_compute_text.contains("\"compute.field_zero\"()"));
    assert!(prover_compute_text.contains("\"compute.sumcheck_claim\"(%"));
    assert!(prover_compute_text.contains("\"compute.sumcheck_driver\"(%"));
    assert!(verifier_compute_text.contains("\"compute.sumcheck_verify_claim\""));
    assert!(verifier_compute_text.contains("\"compute.sumcheck_verify\""));
    assert!(!verifier_compute_text.contains("kernel = @"));

    let prover_kernel_compute =
        resolve_compute_kernels(&context, &prover_compute).expect("resolve prover kernels");
    let verifier_kernel_compute =
        resolve_compute_kernels(&context, &verifier_compute).expect("resolve verifier kernels");
    verify_compute_schema(&prover_kernel_compute)
        .expect("prover kernelized stage6 compute schema is valid");
    verify_compute_schema(&verifier_kernel_compute)
        .expect("verifier kernelized stage6 compute schema is valid");

    let prover_cpu =
        lower_compute_to_cpu(&context, &prover_kernel_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_kernel_compute).expect("lower verifier CPU");
    verify_cpu_schema(&prover_cpu).expect("prover stage6 CPU schema is valid");
    verify_cpu_schema(&verifier_cpu).expect("verifier stage6 CPU schema is valid");

    let prover_kernel_text = prover_kernel_compute.to_text_mlir();
    let verifier_kernel_text = verifier_kernel_compute.to_text_mlir();
    let prover_cpu_text = prover_cpu.to_text_mlir();
    let verifier_cpu_text = verifier_cpu.to_text_mlir();
    assert!(prover_kernel_text.contains("kernel = @jolt.cpu.stage6.batched"));
    assert!(prover_cpu_text.contains("kernel = @jolt.cpu.stage6.batched"));
    assert!(prover_cpu_text.contains("point_order = \"bytecode_read_raf\""));
    assert!(verifier_cpu_text.contains("\"cpu.sumcheck_verify_claim\""));
    assert!(verifier_cpu_text.contains("\"cpu.structured_polynomial_eval\""));
    assert!(verifier_cpu_text.contains("\"cpu.sumcheck_output_eval_family\""));
    assert!(verifier_cpu_text.contains("\"cpu.sumcheck_output_claim\""));
    assert!(!verifier_kernel_text.contains("kernel = @"));
    assert!(!verifier_cpu_text.contains("kernel = @"));
}

#[test]
fn jolt_stage7_protocol_defines_hamming_weight_claim_reduction_flow() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let protocol = build_stage7_protocol(&context, &params).expect("build stage7 protocol");
    verify_protocol_schema(&protocol).expect("stage7 protocol schema is valid");
    let concrete =
        lower_piop_and_fiat_shamir(&context, &protocol).expect("lower stage7 to concrete");
    verify_concrete_transcript(&concrete).expect("stage7 transcript is threaded");

    let text = protocol.to_text_mlir();
    assert!(text.contains("sym_name = \"jolt.stage7_hamming_weight_claim_reduction_domain\""));
    assert!(text.contains("sym_name = \"jolt.stage7.hamming_weight_claim_reduction\""));
    assert!(text.contains("sym_name = \"jolt.stage7.batched\""));
    assert!(text.contains("sym_name = \"stage7.hamming_weight_claim_reduction.gamma\""));
    assert!(text.contains("sym_name = \"stage7.field.one\""));
    assert!(text.contains("sym_name = \"stage7.hamming_weight_claim_reduction.input\""));
    assert!(text.contains("round_schedule = [4]"));
    assert!(text.contains("ordered_claims = [@stage7.hamming_weight_claim_reduction.input]"));
    assert!(text.contains("source_claim = @stage6.booleanity.opening.InstructionRa_0"));
    assert!(text.contains("source_claim = @stage6.instruction_ra_virtual.opening.InstructionRa_0"));
    assert!(text.contains("source_claim = @stage6.bytecode_read_raf.opening.BytecodeRa_0"));
    assert!(text.contains("source_claim = @stage6.ram_ra_virtual.opening.RamRa_0"));
    assert!(text.contains("source_claim = @stage6.hamming_booleanity.opening.HammingWeight"));
    assert!(text.contains("sym_name = \"stage7.hamming_weight_claim_reduction.point.cycle\""));
    assert!(text.contains("sym_name = \"stage7.hamming_weight_claim_reduction.point\""));
    assert!(
        text.contains("sym_name = \"stage7.hamming_weight_claim_reduction.output.eq.Booleanity\"")
    );
    assert!(text.contains(
        "sym_name = \"stage7.hamming_weight_claim_reduction.output.eq.InstructionRa_0.virtualization\""
    ));
    assert!(text.contains("sym_name = \"stage7.hamming_weight_claim_reduction.output.family\""));
    assert!(text.contains("\"piop.sumcheck_output_eval_family\""));
    assert!(text.contains("sym_name = \"stage7.hamming_weight_claim_reduction.output\""));
    assert!(text.contains("@stage7.hamming_weight_claim_reduction.opening.InstructionRa_0"));
    assert!(text.contains("@stage7.hamming_weight_claim_reduction.opening.BytecodeRa_0"));
    assert!(text.contains("@stage7.hamming_weight_claim_reduction.opening.RamRa_0"));
    assert!(!text.contains("kernel = @"));
    assert!(!text.contains("\"compute."));
}

#[test]
fn jolt_stage7_lowers_to_compute_and_cpu_role_ir() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let protocol = build_stage7_protocol(&context, &params).expect("build stage7 protocol");
    let concrete =
        lower_piop_and_fiat_shamir(&context, &protocol).expect("lower stage7 to concrete");
    let prover = project_prover_party(&context, &concrete).expect("project prover party");
    let verifier = project_verifier_party(&context, &concrete).expect("project verifier party");
    let prover_compute = lower_stage7_to_compute(&context, &prover).expect("lower prover stage7");
    let verifier_compute =
        lower_stage7_to_compute(&context, &verifier).expect("lower verifier stage7");
    verify_compute_schema(&prover_compute).expect("prover stage7 compute schema is valid");
    verify_compute_schema(&verifier_compute).expect("verifier stage7 compute schema is valid");

    let prover_compute_text = prover_compute.to_text_mlir();
    let verifier_compute_text = verifier_compute.to_text_mlir();
    assert!(prover_compute_text.contains("\"compute.field_one\"()"));
    assert!(prover_compute_text.contains("\"compute.field_pow\"(%"));
    assert!(prover_compute_text.contains("\"compute.sumcheck_claim\"(%"));
    assert!(prover_compute_text.contains("\"compute.sumcheck_driver\"(%"));
    assert!(prover_compute_text.contains("\"compute.point_concat\"(%"));
    assert!(verifier_compute_text.contains("\"compute.sumcheck_verify_claim\""));
    assert!(verifier_compute_text.contains("\"compute.sumcheck_verify\""));
    assert!(!verifier_compute_text.contains("kernel = @"));

    let prover_kernel_compute =
        resolve_compute_kernels(&context, &prover_compute).expect("resolve prover kernels");
    let verifier_kernel_compute =
        resolve_compute_kernels(&context, &verifier_compute).expect("resolve verifier kernels");
    verify_compute_schema(&prover_kernel_compute)
        .expect("prover kernelized stage7 compute schema is valid");
    verify_compute_schema(&verifier_kernel_compute)
        .expect("verifier kernelized stage7 compute schema is valid");

    let prover_cpu =
        lower_compute_to_cpu(&context, &prover_kernel_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_kernel_compute).expect("lower verifier CPU");
    verify_cpu_schema(&prover_cpu).expect("prover stage7 CPU schema is valid");
    verify_cpu_schema(&verifier_cpu).expect("verifier stage7 CPU schema is valid");

    let prover_kernel_text = prover_kernel_compute.to_text_mlir();
    let verifier_kernel_text = verifier_kernel_compute.to_text_mlir();
    let prover_cpu_text = prover_cpu.to_text_mlir();
    let verifier_cpu_text = verifier_cpu.to_text_mlir();
    assert!(prover_kernel_text.contains("kernel = @jolt.cpu.stage7.batched"));
    assert!(prover_cpu_text.contains("kernel = @jolt.cpu.stage7.batched"));
    assert!(prover_cpu_text.contains("point_order = \"reverse\""));
    assert!(verifier_cpu_text.contains("\"cpu.structured_polynomial_eval\""));
    assert!(verifier_cpu_text.contains("\"cpu.sumcheck_output_eval_family\""));
    assert!(verifier_cpu_text.contains("\"cpu.sumcheck_output_claim\""));
    assert!(verifier_cpu_text.contains("\"cpu.sumcheck_verify_claim\""));
    assert!(!verifier_kernel_text.contains("kernel = @"));
    assert!(!verifier_cpu_text.contains("kernel = @"));
}

#[test]
fn jolt_stage8_protocol_defines_evaluation_proof_flow() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let protocol = build_stage8_protocol(&context, &params).expect("build stage8 protocol");
    verify_protocol_schema(&protocol).expect("stage8 protocol schema is valid");
    let concrete =
        lower_piop_and_fiat_shamir(&context, &protocol).expect("lower stage8 to concrete");
    verify_concrete_transcript(&concrete).expect("stage8 transcript is threaded");

    let text = protocol.to_text_mlir();
    assert!(text.contains("sym_name = \"jolt.stage8\""));
    assert!(text.contains("name = \"evaluation_proof\""));
    assert!(text.contains("sym_name = \"stage8.evaluation.point_source\""));
    assert!(text.contains("source_claim = @stage7.input.stage6.booleanity.InstructionRa_0"));
    assert!(text.contains("sym_name = \"stage8.evaluation.opening.RamInc\""));
    assert!(text.contains("source_claim = @stage6.inc_claim_reduction.eval.RamInc"));
    assert!(text.contains("sym_name = \"stage8.evaluation.opening.InstructionRa_0\""));
    assert!(
        text.contains("source_claim = @stage7.hamming_weight_claim_reduction.eval.InstructionRa_0")
    );
    assert!(text.contains("\"pcs.opening_batch\"(%"));
    assert!(text.contains("policy = \"jolt_stage8_joint_rlc\""));
    assert!(text.contains("transcript_label = \"rlc_claims\""));
    assert!(text.contains("ordered_claims = [@stage8.evaluation.opening.RamInc, @stage8.evaluation.opening.RdInc, @stage8.evaluation.opening.InstructionRa_0"));
}

#[test]
fn jolt_stage8_lowers_to_compute_and_cpu_role_ir() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let protocol = build_stage8_protocol(&context, &params).expect("build stage8 protocol");
    let concrete =
        lower_piop_and_fiat_shamir(&context, &protocol).expect("lower stage8 to concrete");
    let prover = project_prover_party(&context, &concrete).expect("project prover party");
    let verifier = project_verifier_party(&context, &concrete).expect("project verifier party");
    let prover_compute = lower_stage8_to_compute(&context, &prover).expect("lower prover stage8");
    let verifier_compute =
        lower_stage8_to_compute(&context, &verifier).expect("lower verifier stage8");
    verify_compute_schema(&prover_compute).expect("prover stage8 compute schema is valid");
    verify_compute_schema(&verifier_compute).expect("verifier stage8 compute schema is valid");

    let prover_compute_text = prover_compute.to_text_mlir();
    let verifier_compute_text = verifier_compute.to_text_mlir();
    assert!(prover_compute_text.contains("\"compute.pcs_opening_claim\"(%"));
    assert!(prover_compute_text.contains("\"compute.pcs_opening_batch\"(%"));
    assert!(prover_compute_text.contains("\"compute.pcs_batch_open\"(%"));
    assert!(!prover_compute_text.contains("\"compute.pcs_batch_verify\"(%"));
    assert!(verifier_compute_text.contains("\"compute.pcs_batch_verify\"(%"));
    assert!(!verifier_compute_text.contains("\"compute.pcs_batch_open\"(%"));

    let prover_kernel_compute =
        resolve_compute_kernels(&context, &prover_compute).expect("resolve prover kernels");
    let verifier_kernel_compute =
        resolve_compute_kernels(&context, &verifier_compute).expect("resolve verifier kernels");
    verify_compute_schema(&prover_kernel_compute)
        .expect("prover kernelized stage8 compute schema is valid");
    verify_compute_schema(&verifier_kernel_compute)
        .expect("verifier kernelized stage8 compute schema is valid");

    let prover_cpu =
        lower_compute_to_cpu(&context, &prover_kernel_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_kernel_compute).expect("lower verifier CPU");
    verify_cpu_schema(&prover_cpu).expect("prover stage8 CPU schema is valid");
    verify_cpu_schema(&verifier_cpu).expect("verifier stage8 CPU schema is valid");

    let prover_cpu_text = prover_cpu.to_text_mlir();
    let verifier_cpu_text = verifier_cpu.to_text_mlir();
    assert!(prover_cpu_text.contains("\"cpu.pcs_batch_open\"(%"));
    assert!(verifier_cpu_text.contains("\"cpu.pcs_batch_verify\"(%"));
    assert!(!prover_cpu_text.contains("kernel = @"));
    assert!(!verifier_cpu_text.contains("kernel = @"));
}

#[test]
fn stage2_rust_targets_extract_and_compile() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let (prover_cpu, verifier_cpu) = build_stage2_pipeline_cpu(&context, &params);
    let prover_program = stage2_cpu_program(&prover_cpu).expect("extract prover stage2 program");
    let verifier_program =
        stage2_cpu_program(&verifier_cpu).expect("extract verifier stage2 program");

    assert_eq!(prover_program.role, Role::Prover);
    assert_eq!(verifier_program.role, Role::Verifier);
    assert_eq!(prover_program.kernels.len(), 7);
    assert!(verifier_program.kernels.is_empty());
    assert_eq!(prover_program.opening_inputs.len(), 11);
    assert_eq!(prover_program.field_exprs.len(), 21);
    assert_eq!(prover_program.field_constants.len(), 1);
    assert_eq!(prover_program.claims.len(), 6);
    assert_eq!(prover_program.drivers.len(), 2);
    assert_eq!(prover_program.point_slices.len(), 1);
    assert_eq!(prover_program.point_concats.len(), 1);
    assert!(prover_program
        .claims
        .iter()
        .any(|claim| claim.claim_value == "stage2.ram_read_write.claim_expr"));
    assert!(prover_program
        .drivers
        .iter()
        .any(|driver| driver.kernel.as_deref() == Some("jolt.cpu.stage2.batched")));
    assert!(verifier_program
        .claims
        .iter()
        .all(|claim| claim.kernel.is_none() && claim.relation.is_some()));
    assert!(verifier_program
        .drivers
        .iter()
        .all(|driver| driver.kernel.is_none() && driver.relation.is_some()));

    let prover_source = emit_stage2_rust(&prover_cpu).expect("emit stage2 prover rust");
    let verifier_source = emit_stage2_rust(&verifier_cpu).expect("emit stage2 verifier rust");
    assert_eq!(prover_source.filename, "prove_stage2.rs");
    assert_eq!(verifier_source.filename, "verify_stage2.rs");
    assert!(prover_source.source.contains("jolt_stage2_ram_read_write"));
    assert!(prover_source.source.contains("Stage2KernelExecutor"));
    assert!(!verifier_source.source.contains("jolt_kernels"));
    assert!(verifier_source.source.contains("Stage2VerifierProgramPlan"));
    assert!(verifier_source.source.contains("pub fn verify_stage2"));
    assert!(verifier_source
        .source
        .contains("SumcheckVerifier::verify_optimized"));
    assert_or_update_fixture("tests/fixtures/prove_stage2.rs", &prover_source.source);
    assert_or_update_fixture("tests/fixtures/verify_stage2.rs", &verifier_source.source);
    assert_rust_source_compiles(&prover_source.filename, &prover_source.source);
    assert_rust_source_compiles(&verifier_source.filename, &verifier_source.source);
}

#[test]
fn stage3_rust_targets_extract_and_compile() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let (prover_cpu, verifier_cpu) = build_stage3_pipeline_cpu(&context, &params);
    let prover_program = stage3_cpu_program(&prover_cpu).expect("extract prover stage3 program");
    let verifier_program =
        stage3_cpu_program(&verifier_cpu).expect("extract verifier stage3 program");

    assert_eq!(prover_program.role, Role::Prover);
    assert_eq!(verifier_program.role, Role::Verifier);
    assert_eq!(prover_program.kernels.len(), 4);
    assert!(verifier_program.kernels.is_empty());
    assert_eq!(prover_program.opening_inputs.len(), 12);
    assert_eq!(prover_program.field_exprs.len(), 19);
    assert_eq!(prover_program.field_constants.len(), 1);
    assert_eq!(prover_program.opening_equalities.len(), 2);
    assert_eq!(prover_program.claims.len(), 3);
    assert_eq!(prover_program.drivers.len(), 1);
    assert_eq!(prover_program.opening_claims.len(), 16);
    assert!(prover_program
        .drivers
        .iter()
        .any(|driver| driver.kernel.as_deref() == Some("jolt.cpu.stage3.batched")));
    assert!(verifier_program
        .claims
        .iter()
        .all(|claim| claim.kernel.is_none() && claim.relation.is_some()));
    assert!(verifier_program
        .drivers
        .iter()
        .all(|driver| driver.kernel.is_none() && driver.relation.is_some()));

    let prover_source = emit_stage3_rust(&prover_cpu).expect("emit stage3 prover rust");
    let verifier_source = emit_stage3_rust(&verifier_cpu).expect("emit stage3 verifier rust");
    assert_eq!(prover_source.filename, "prove_stage3.rs");
    assert_eq!(verifier_source.filename, "verify_stage3.rs");
    assert!(prover_source.source.contains("jolt_stage3_spartan_shift"));
    assert!(prover_source.source.contains("Stage3KernelExecutor"));
    assert!(prover_source
        .source
        .contains("Stage3OpeningClaimEqualityPlan"));
    assert!(!verifier_source.source.contains("jolt_kernels"));
    assert!(verifier_source.source.contains("Stage3VerifierProgramPlan"));
    assert!(verifier_source.source.contains("pub fn verify_stage3"));
    assert!(verifier_source
        .source
        .contains("bolt_verifier_runtime::verify_batched_sumcheck"));
    assert!(verifier_source
        .source
        .contains("Stage3OpeningClaimEqualityPlan"));
    assert_or_update_fixture("tests/fixtures/prove_stage3.rs", &prover_source.source);
    assert_or_update_fixture("tests/fixtures/verify_stage3.rs", &verifier_source.source);
    assert_rust_source_compiles(&prover_source.filename, &prover_source.source);
    assert_rust_source_compiles(&verifier_source.filename, &verifier_source.source);
}

#[test]
fn stage4_rust_targets_extract_and_compile() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let (prover_cpu, verifier_cpu) = build_stage4_pipeline_cpu(&context, &params);
    let prover_program = stage4_cpu_program(&prover_cpu).expect("extract prover stage4 program");
    let verifier_program =
        stage4_cpu_program(&verifier_cpu).expect("extract verifier stage4 program");

    assert_eq!(prover_program.role, Role::Prover);
    assert_eq!(verifier_program.role, Role::Verifier);
    assert_eq!(prover_program.kernels.len(), 3);
    assert!(verifier_program.kernels.is_empty());
    assert_eq!(prover_program.steps.len(), 4);
    assert_eq!(prover_program.transcript_squeezes.len(), 2);
    assert_eq!(prover_program.transcript_absorb_bytes.len(), 1);
    assert_eq!(prover_program.opening_inputs.len(), 8);
    assert_eq!(prover_program.field_exprs.len(), 9);
    assert!(prover_program.field_constants.is_empty());
    assert_eq!(prover_program.opening_equalities.len(), 2);
    assert_eq!(prover_program.claims.len(), 2);
    assert_eq!(prover_program.drivers.len(), 1);
    assert_eq!(prover_program.instance_results.len(), 2);
    assert_eq!(prover_program.evals.len(), 7);
    assert_eq!(prover_program.point_slices.len(), 2);
    assert_eq!(prover_program.point_concats.len(), 1);
    assert_eq!(prover_program.opening_claims.len(), 7);
    assert_eq!(prover_program.opening_batches.len(), 1);
    assert!(prover_program
        .transcript_absorb_bytes
        .iter()
        .any(
            |absorb| absorb.symbol == "stage4.ram_val_check.domain_separator"
                && absorb.label == "ram_val_check_gamma"
                && absorb.payload.is_empty()
        ));
    assert!(prover_program
        .drivers
        .iter()
        .any(|driver| driver.kernel.as_deref() == Some("jolt.cpu.stage4.batched")));
    assert!(verifier_program
        .claims
        .iter()
        .all(|claim| claim.kernel.is_none() && claim.relation.is_some()));
    assert!(verifier_program
        .drivers
        .iter()
        .all(|driver| driver.kernel.is_none() && driver.relation.is_some()));

    let prover_source = emit_stage4_rust(&prover_cpu).expect("emit stage4 prover rust");
    let verifier_source = emit_stage4_rust(&verifier_cpu).expect("emit stage4 verifier rust");
    assert_eq!(prover_source.filename, "prove_stage4.rs");
    assert_eq!(verifier_source.filename, "verify_stage4.rs");
    assert!(prover_source.source.contains("jolt_stage4_ram_val_check"));
    assert!(prover_source
        .source
        .contains("Stage4TranscriptAbsorbBytesPlan"));
    assert!(prover_source
        .source
        .contains("STAGE4_TRANSCRIPT_ABSORB_BYTES"));
    assert!(prover_source.source.contains("Stage4KernelExecutor"));
    assert!(prover_source.source.contains("execute_stage4_program"));
    assert!(prover_source.source.contains("execute_stage4_prover"));
    assert!(!verifier_source.source.contains("jolt_kernels"));
    assert!(verifier_source
        .source
        .contains("Stage4TranscriptAbsorbBytesPlan"));
    assert!(verifier_source
        .source
        .contains("relation: Some(Stage4RelationKind::Stage4Batched)"));
    assert!(verifier_source.source.contains("Stage4VerifierProgramPlan"));
    assert!(verifier_source.source.contains("pub fn verify_stage4"));
    assert!(verifier_source.source.contains("LabelWithCount"));
    assert!(verifier_source
        .source
        .contains("bolt_verifier_runtime::verify_batched_sumcheck"));
    assert!(verifier_source.source.contains("stage4_verifier_program"));
    assert_or_update_fixture("tests/fixtures/prove_stage4.rs", &prover_source.source);
    assert_or_update_fixture("tests/fixtures/verify_stage4.rs", &verifier_source.source);
    assert_rust_source_compiles(&prover_source.filename, &prover_source.source);
    assert_rust_source_compiles(&verifier_source.filename, &verifier_source.source);
}

#[test]
fn stage5_rust_targets_extract_and_compile() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let (prover_cpu, verifier_cpu) = build_stage5_pipeline_cpu(&context, &params);
    let prover_program = stage5_cpu_program(&prover_cpu).expect("extract prover stage5 program");
    let verifier_program =
        stage5_cpu_program(&verifier_cpu).expect("extract verifier stage5 program");

    assert_eq!(prover_program.role, Role::Prover);
    assert_eq!(verifier_program.role, Role::Verifier);
    assert_eq!(prover_program.kernels.len(), 4);
    assert!(verifier_program.kernels.is_empty());
    assert_eq!(prover_program.steps.len(), 3);
    assert_eq!(prover_program.transcript_squeezes.len(), 2);
    assert!(prover_program.transcript_absorb_bytes.is_empty());
    assert_eq!(prover_program.opening_inputs.len(), 8);
    assert_eq!(prover_program.field_exprs.len(), 10);
    assert!(prover_program.field_constants.is_empty());
    assert_eq!(prover_program.opening_equalities.len(), 1);
    assert_eq!(prover_program.claims.len(), 3);
    assert_eq!(prover_program.drivers.len(), 1);
    assert_eq!(prover_program.instance_results.len(), 3);
    assert_eq!(
        prover_program.evals.len(),
        params.lookup_table_count + params.instruction_ra_virtual_d + 4
    );
    assert_eq!(prover_program.relation_output_values.len(), 4);
    assert!(prover_program.relation_outputs.is_empty());
    assert_eq!(verifier_program.relation_output_values.len(), 5);
    assert_eq!(verifier_program.relation_outputs.len(), 3);
    assert_eq!(
        prover_program.point_slices.len(),
        params.instruction_ra_virtual_d + 3
    );
    assert_eq!(
        prover_program.point_concats.len(),
        params.instruction_ra_virtual_d + 2
    );
    assert_eq!(
        prover_program.opening_claims.len(),
        params.lookup_table_count + params.instruction_ra_virtual_d + 4
    );
    assert_eq!(prover_program.opening_batches.len(), 1);
    assert!(prover_program
        .drivers
        .iter()
        .any(|driver| driver.kernel.as_deref() == Some("jolt.cpu.stage5.batched")));
    assert!(prover_program.instance_results.iter().any(|instance| {
        instance.symbol == "stage5.instruction_read_raf.instance"
            && instance.point_order == "instruction_read_raf"
    }));
    assert!(verifier_program
        .claims
        .iter()
        .all(|claim| claim.kernel.is_none() && claim.relation.is_some()));
    assert!(verifier_program
        .drivers
        .iter()
        .all(|driver| driver.kernel.is_none() && driver.relation.is_some()));

    let prover_source = emit_stage5_rust(&prover_cpu).expect("emit stage5 prover rust");
    let verifier_source = emit_stage5_rust(&verifier_cpu).expect("emit stage5 verifier rust");
    assert_eq!(prover_source.filename, "prove_stage5.rs");
    assert_eq!(verifier_source.filename, "verify_stage5.rs");
    assert!(prover_source
        .source
        .contains("jolt_stage5_instruction_read_raf"));
    assert!(prover_source.source.contains("Stage5KernelExecutor"));
    assert!(prover_source.source.contains("execute_stage5_program"));
    assert!(prover_source.source.contains("execute_stage5_prover"));
    assert!(!verifier_source.source.contains("jolt_kernels"));
    assert!(verifier_source.source.contains("Stage5VerifierProgramPlan"));
    assert!(verifier_source.source.contains("pub fn verify_stage5"));
    assert!(verifier_source
        .source
        .contains("relation: Some(Stage5RelationKind::Stage5Batched)"));
    assert!(verifier_source
        .source
        .contains("evaluate_stage5_instruction_read_raf_local_scalars"));
    assert!(verifier_source
        .source
        .contains("Stage5RelationKind::Stage5InstructionReadRaf"));
    assert!(verifier_source
        .source
        .contains("Stage5InstructionReadRafLocalScalarKind::LookupTable"));
    assert!(!verifier_source
        .source
        .contains("expected_ram_ra_claim_reduction"));
    assert!(!verifier_source
        .source
        .contains("expected_registers_val_evaluation"));
    assert!(verifier_source.source.contains("Stage5RelationOutputPlan"));
    assert!(verifier_source
        .source
        .contains("stage5.instruction_read_raf.output.claim_expr"));
    assert!(verifier_source
        .source
        .contains("stage5.ram_ra_claim_reduction.output.eq.ReadWrite"));
    assert!(verifier_source
        .source
        .contains("stage5.registers_val_evaluation.output.lt.RegistersValCycle"));
    assert!(verifier_source
        .source
        .contains("bolt_verifier_runtime::evaluate_relation_output_batch"));
    assert!(verifier_source
        .source
        .contains("stage5_relation_output_inputs"));
    assert!(verifier_source
        .source
        .contains("Stage5RelationKind::Stage5RamRaClaimReduction"));
    assert!(verifier_source
        .source
        .contains("Stage5RelationKind::Stage5RegistersValEvaluation"));
    assert!(verifier_source.source.contains("LookupTableFlag_40"));
    assert!(!verifier_source.source.contains("LookupTableFlag_41"));
    assert!(verifier_source
        .source
        .contains("stage5.instruction_read_raf.eval.InstructionRa_7"));
    assert!(!verifier_source
        .source
        .contains("stage5.instruction_read_raf.eval.InstructionRa_8"));
    assert!(!verifier_source
        .source
        .contains("jolt.stage5.registers_read_write"));
    assert!(!verifier_source.source.contains("jolt.stage5.ram_val_check"));
    assert!(verifier_source
        .source
        .contains("bolt_verifier_runtime::verify_batched_sumcheck"));
    assert!(verifier_source.source.contains("stage5_verifier_program"));
    assert_rust_source_compiles(&prover_source.filename, &prover_source.source);
    assert_rust_source_compiles(&verifier_source.filename, &verifier_source.source);
}

#[test]
fn stage6_rust_targets_extract_and_compile() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let (prover_cpu, verifier_cpu) = build_stage6_pipeline_cpu(&context, &params);
    let prover_program = stage6_cpu_program(&prover_cpu).expect("extract prover stage6 program");
    let verifier_program =
        stage6_cpu_program(&verifier_cpu).expect("extract verifier stage6 program");

    assert_eq!(prover_program.role, Role::Prover);
    assert_eq!(verifier_program.role, Role::Verifier);
    assert_eq!(prover_program.kernels.len(), 7);
    assert!(verifier_program.kernels.is_empty());
    assert_eq!(prover_program.steps.len(), 10);
    assert_eq!(prover_program.transcript_squeezes.len(), 9);
    assert!(prover_program.transcript_absorb_bytes.is_empty());
    assert_eq!(prover_program.opening_inputs.len(), 91);
    assert!(prover_program.field_exprs.len() > 150);
    assert_eq!(prover_program.field_constants.len(), 1);
    assert!(prover_program.opening_equalities.is_empty());
    assert_eq!(prover_program.claims.len(), 6);
    assert_eq!(prover_program.drivers.len(), 1);
    assert_eq!(prover_program.instance_results.len(), 6);
    assert_eq!(
        prover_program.evals.len(),
        params.bytecode_d
            + params.instruction_d
            + params.bytecode_d
            + params.ram_d
            + 1
            + params.ram_d
            + params.instruction_d
            + 2
    );
    assert_eq!(prover_program.relation_output_values.len(), 8);
    assert!(prover_program.relation_outputs.is_empty());
    assert_eq!(verifier_program.relation_output_values.len(), 8);
    assert!(verifier_program.relation_output_eval_families.is_empty());
    assert!(verifier_program.relation_output_product_families.is_empty());
    assert!(verifier_program
        .relation_output_function_families
        .is_empty());
    assert_eq!(verifier_program.relation_outputs.len(), 6);
    let total_booleanity_ra = params.instruction_d + params.bytecode_d + params.ram_d;
    let booleanity_exprs = verifier_program
        .field_exprs
        .iter()
        .filter(|expr| expr.symbol.starts_with("stage6.booleanity.output."))
        .collect::<Vec<_>>();
    assert!(booleanity_exprs.iter().any(|expr| {
        expr.symbol == "stage6.booleanity.output.term0.boolean_zero.square"
            && expr.formula == "field.product"
            && expr.operands
                == vec![
                    "stage6.booleanity.eval.InstructionRa_0".to_owned(),
                    "stage6.booleanity.eval.InstructionRa_0".to_owned(),
                ]
    }));
    assert!(booleanity_exprs.iter().any(|expr| {
        expr.symbol == "stage6.booleanity.output.term0.boolean_zero"
            && expr.formula == "field.sub"
            && expr.operands
                == vec![
                    "stage6.booleanity.output.term0.boolean_zero.square".to_owned(),
                    "stage6.booleanity.eval.InstructionRa_0".to_owned(),
                ]
    }));
    assert!(booleanity_exprs.iter().any(|expr| {
        expr.symbol == "stage6.booleanity.output.term0"
            && expr.formula == "field.product"
            && expr.operands
                == vec![
                    "stage6.booleanity.output.term0.boolean_zero".to_owned(),
                    "stage6.booleanity.output.eq.InstructionRa0".to_owned(),
                ]
    }));
    assert!(booleanity_exprs.iter().any(|expr| {
        expr.symbol == "stage6.booleanity.output.gamma_pow_2"
            && expr.formula == "field.pow:2"
            && expr.operands == vec!["stage6.booleanity.gamma".to_owned()]
    }));
    assert!(booleanity_exprs.iter().any(|expr| {
        expr.symbol
            == format!(
                "stage6.booleanity.output.gamma_pow_{}",
                2 * (total_booleanity_ra - 1)
            )
    }));
    assert!(booleanity_exprs.iter().any(|expr| {
        expr.symbol == "stage6.booleanity.output.claim_expr" && expr.formula == "field.sum"
    }));
    let hamming_expr_symbols = verifier_program
        .field_exprs
        .iter()
        .filter(|expr| expr.symbol.starts_with("stage6.hamming_booleanity.output."))
        .map(|expr| expr.symbol.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        hamming_expr_symbols,
        vec![
            "stage6.hamming_booleanity.output.term0.boolean_zero.square",
            "stage6.hamming_booleanity.output.term0.boolean_zero",
            "stage6.hamming_booleanity.output.term0",
            "stage6.hamming_booleanity.output.claim_expr"
        ]
    );
    assert!(verifier_program
        .relation_output_product_families
        .iter()
        .all(|family| family.symbol != "stage6.ram_ra_virtual.output.family"));
    assert!(verifier_program
        .relation_output_product_families
        .iter()
        .all(|family| family.symbol != "stage6.instruction_ra_virtual.output.family"));
    let ram_expr_symbols = verifier_program
        .field_exprs
        .iter()
        .filter(|expr| expr.symbol.starts_with("stage6.ram_ra_virtual.output."))
        .map(|expr| expr.symbol.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        ram_expr_symbols,
        vec![
            "stage6.ram_ra_virtual.output.term0",
            "stage6.ram_ra_virtual.output.claim_expr"
        ]
    );
    let committed_per_virtual = params.instruction_d / params.instruction_ra_virtual_d;
    let instruction_exprs = verifier_program
        .field_exprs
        .iter()
        .filter(|expr| {
            expr.symbol
                .starts_with("stage6.instruction_ra_virtual.output.")
        })
        .collect::<Vec<_>>();
    assert!(instruction_exprs
        .iter()
        .any(|expr| expr.symbol == "stage6.instruction_ra_virtual.output.claim_expr"));
    assert!(instruction_exprs
        .iter()
        .any(|expr| expr.symbol == "stage6.instruction_ra_virtual.output.gamma_pow_7"));
    for virtual_index in 0..params.instruction_ra_virtual_d {
        let term = instruction_exprs
            .iter()
            .find(|expr| {
                expr.symbol == format!("stage6.instruction_ra_virtual.output.term{virtual_index}")
            })
            .expect("instruction RA virtual output term");
        if virtual_index == 0 {
            assert!(!term
                .operands
                .contains(&"stage6.instruction_ra_virtual.output.gamma_pow_0".to_owned()));
        } else {
            assert!(term.operands.contains(&format!(
                "stage6.instruction_ra_virtual.output.gamma_pow_{virtual_index}"
            )));
        }
        assert_eq!(
            term.operands
                .iter()
                .filter(|operand| operand.contains(".eval.InstructionRa_"))
                .cloned()
                .collect::<Vec<_>>(),
            (0..committed_per_virtual)
                .map(|chunk_index| {
                    let index = virtual_index * committed_per_virtual + chunk_index;
                    format!("stage6.instruction_ra_virtual.eval.InstructionRa_{index}")
                })
                .collect::<Vec<_>>()
        );
        assert!(term
            .operands
            .contains(&"stage6.instruction_ra_virtual.output.eq.Cycle".to_owned()));
    }
    let inc_expr_symbols = verifier_program
        .field_exprs
        .iter()
        .filter(|expr| {
            expr.symbol
                .starts_with("stage6.inc_claim_reduction.output.")
        })
        .map(|expr| expr.symbol.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        inc_expr_symbols,
        vec![
            "stage6.inc_claim_reduction.output.term0",
            "stage6.inc_claim_reduction.output.gamma_pow_1",
            "stage6.inc_claim_reduction.output.term1",
            "stage6.inc_claim_reduction.output.gamma_pow_2",
            "stage6.inc_claim_reduction.output.term2",
            "stage6.inc_claim_reduction.output.gamma_pow_3",
            "stage6.inc_claim_reduction.output.term3",
            "stage6.inc_claim_reduction.output.claim_expr"
        ]
    );
    let inc_claims = verifier_program
        .relation_outputs
        .iter()
        .filter(|claim| claim.expected_output == "stage6.inc_claim_reduction.output.claim_expr")
        .collect::<Vec<_>>();
    assert_eq!(inc_claims.len(), 1);
    let booleanity_claims = verifier_program
        .relation_outputs
        .iter()
        .filter(|claim| claim.expected_output == "stage6.booleanity.output.claim_expr")
        .collect::<Vec<_>>();
    assert_eq!(booleanity_claims.len(), 1);
    assert!(verifier_program
        .scalar_exprs
        .iter()
        .any(|expr| expr.symbol == "stage6.booleanity.output.eq.InstructionRa0"));
    let hamming_claims = verifier_program
        .relation_outputs
        .iter()
        .filter(|claim| claim.expected_output == "stage6.hamming_booleanity.output.claim_expr")
        .collect::<Vec<_>>();
    assert_eq!(hamming_claims.len(), 1);
    let ram_ra_claims = verifier_program
        .relation_outputs
        .iter()
        .filter(|claim| claim.expected_output == "stage6.ram_ra_virtual.output.claim_expr")
        .collect::<Vec<_>>();
    assert_eq!(ram_ra_claims.len(), 1);
    let instruction_ra_claims = verifier_program
        .relation_outputs
        .iter()
        .filter(|claim| claim.expected_output == "stage6.instruction_ra_virtual.output.claim_expr")
        .collect::<Vec<_>>();
    assert_eq!(instruction_ra_claims.len(), 1);
    let bytecode_claims = verifier_program
        .relation_outputs
        .iter()
        .filter(|claim| claim.expected_output == "stage6.bytecode_read_raf.output.claim_expr")
        .collect::<Vec<_>>();
    assert_eq!(bytecode_claims.len(), 1);
    let bytecode_output_scalar_exprs = verifier_program
        .scalar_exprs
        .iter()
        .filter(|expr| expr.symbol.starts_with("stage6.bytecode_read_raf.output."))
        .collect::<Vec<_>>();
    assert!(bytecode_output_scalar_exprs.iter().any(|expr| {
        expr.symbol == "stage6.bytecode_read_raf.output.product.BytecodeRa"
            && expr.formula == "field_vector.product"
            && expr.operands == vec!["stage6.bytecode_read_raf.eval.BytecodeRa".to_owned()]
    }));
    let bytecode_output_exprs = verifier_program
        .field_exprs
        .iter()
        .filter(|expr| expr.symbol.starts_with("stage6.bytecode_read_raf.output."))
        .collect::<Vec<_>>();
    assert!(bytecode_output_exprs.iter().any(|expr| {
        expr.symbol == "stage6.bytecode_read_raf.output.claim_expr"
            && expr.formula == "field.product"
            && expr.operands
                == vec![
                    "stage6.bytecode_read_raf.output.contribution".to_owned(),
                    "stage6.bytecode_read_raf.output.product.BytecodeRa".to_owned(),
                ]
    }));
    assert_eq!(prover_program.point_zeros.len(), 1);
    assert_eq!(
        prover_program.point_slices.len(),
        params.bytecode_d + 3 + params.ram_d + params.instruction_d
    );
    assert_eq!(
        prover_program.point_concats.len(),
        params.bytecode_d + 2 + params.ram_d + params.instruction_d
    );
    assert_eq!(
        prover_program.opening_claims.len(),
        prover_program.evals.len()
    );
    assert_eq!(prover_program.opening_batches.len(), 1);
    assert!(prover_program
        .drivers
        .iter()
        .any(|driver| driver.kernel.as_deref() == Some("jolt.cpu.stage6.batched")));
    assert!(prover_program.instance_results.iter().any(|instance| {
        instance.symbol == "stage6.bytecode_read_raf.instance"
            && instance.point_order == "bytecode_read_raf"
    }));
    assert!(prover_program.instance_results.iter().any(|instance| {
        instance.symbol == "stage6.booleanity.instance"
            && instance.point_order == "stage6_booleanity"
    }));
    assert!(verifier_program.opening_inputs.iter().any(|input| {
        input.symbol == "stage6.input.stage1.LookupOutput"
            && input.source_stage == "stage1"
            && input.source_claim == "stage1.outer_remaining.opening.LookupOutput"
    }));
    assert!(verifier_program.claims.iter().any(|claim| {
        claim.symbol == "stage6.hamming_booleanity.input"
            && claim
                .input_openings
                .contains(&"stage6.input.stage1.LookupOutput".to_owned())
    }));
    assert!(verifier_program.claims.iter().any(|claim| {
        claim.symbol == "stage6.booleanity.input" && claim.input_openings.is_empty()
    }));
    assert!(verifier_program
        .claims
        .iter()
        .all(|claim| claim.kernel.is_none() && claim.relation.is_some()));
    assert!(verifier_program
        .drivers
        .iter()
        .all(|driver| driver.kernel.is_none() && driver.relation.is_some()));

    let prover_source = emit_stage6_rust(&prover_cpu).expect("emit stage6 prover rust");
    let verifier_source = emit_stage6_rust(&verifier_cpu).expect("emit stage6 verifier rust");
    assert_eq!(prover_source.filename, "prove_stage6.rs");
    assert_eq!(verifier_source.filename, "verify_stage6.rs");
    assert!(prover_source
        .source
        .contains("jolt_stage6_bytecode_read_raf"));
    assert!(prover_source.source.contains("Stage6KernelExecutor"));
    assert!(prover_source.source.contains("execute_stage6_program"));
    assert!(prover_source.source.contains("execute_stage6_prover"));
    assert!(!verifier_source.source.contains("jolt_kernels"));
    assert!(verifier_source.source.contains("Stage6VerifierProgramPlan"));
    assert!(verifier_source.source.contains("pub fn verify_stage6"));
    assert!(verifier_source
        .source
        .contains("relation: Some(Stage6RelationKind::Stage6Batched)"));
    assert!(verifier_source
        .source
        .contains("Stage6RelationKind::Stage6BytecodeReadRaf"));
    assert!(verifier_source.source.contains("Stage6VerifierData"));
    assert!(verifier_source.source.contains("Stage6BytecodeReadRafData"));
    assert!(verifier_source.source.contains("Stage6BytecodeEntry"));
    assert!(!verifier_source
        .source
        .contains("expected_bytecode_read_raf"));
    assert!(verifier_source.source.contains("STAGE6_BYTECODE_PLAN"));
    assert!(verifier_source
        .source
        .contains("Stage67BytecodeReadRafPlan"));
    assert!(verifier_source
        .source
        .contains("Stage67BytecodeTermPlan::LookupTable"));
    assert!(verifier_source
        .source
        .contains("Stage67BytecodeTermPlan::RegisterEq"));
    assert!(verifier_source
        .source
        .contains("evaluate_stage67_bytecode_read_raf_output_scalars"));
    assert!(!verifier_source
        .source
        .contains("stage6.bytecode_read_raf.output.product.BytecodeReadRaf"));
    assert!(verifier_source
        .source
        .contains("stage6.bytecode_read_raf.output.claim_expr"));
    assert!(verifier_source
        .source
        .contains("Stage6ScalarExprKind::FieldVectorProduct"));
    assert!(!verifier_source
        .source
        .contains("expected_stage67_bytecode_read_raf"));
    assert!(!verifier_source.source.contains("Stage67BytecodeSymbols"));
    assert!(verifier_source
        .source
        .contains("stage6.bytecode_read_raf.data"));
    assert!(!verifier_source.source.contains("expected_booleanity"));
    assert!(!verifier_source
        .source
        .contains("expected_hamming_booleanity"));
    assert!(verifier_source
        .source
        .contains("bolt_verifier_runtime::evaluate_relation_output_batch"));
    assert!(verifier_source
        .source
        .contains("stage6_relation_output_inputs"));
    assert!(verifier_source
        .source
        .contains("stage6.booleanity.output.eq.InstructionRa0"));
    assert!(!verifier_source
        .source
        .contains("stage6.booleanity.output.family"));
    assert!(verifier_source
        .source
        .contains("stage6.hamming_booleanity.output.eq.LookupOutput"));
    assert!(verifier_source
        .source
        .contains("Stage6RelationKind::Stage6IncClaimReduction"));
    assert!(verifier_source
        .source
        .contains("stage6.input.stage1.LookupOutput"));
    assert!(!verifier_source.source.contains("expected_ram_ra_virtual"));
    assert!(!verifier_source
        .source
        .contains("expected_instruction_ra_virtual"));
    assert!(!verifier_source
        .source
        .contains("expected_inc_claim_reduction"));
    assert!(verifier_source.source.contains("Stage6RelationOutputPlan"));
    assert!(!verifier_source
        .source
        .contains("RelationOutputFunctionFamilyPlan"));
    assert!(!verifier_source
        .source
        .contains("STAGE6_RELATION_OUTPUT_0_FUNCTION_FAMILIES"));
    assert!(!verifier_source
        .source
        .contains("STAGE6_RELATION_OUTPUT_4_FAMILIES"));
    assert!(!verifier_source
        .source
        .contains("RelationOutputProductFamilyPlan"));
    assert!(!verifier_source
        .source
        .contains("STAGE6_RELATION_OUTPUT_2_PRODUCT_FAMILIES"));
    assert!(!verifier_source
        .source
        .contains("STAGE6_RELATION_OUTPUT_3_PRODUCT_FAMILIES"));
    assert!(!verifier_source
        .source
        .contains("STAGE6_RELATION_OUTPUT_5_PRODUCT_FAMILIES"));
    assert!(!verifier_source
        .source
        .contains("stage6.booleanity.output.family"));
    assert!(verifier_source
        .source
        .contains("stage6.booleanity.output.eq.InstructionRa0"));
    assert!(!verifier_source
        .source
        .contains("stage6.booleanity.output.gamma_sq_"));
    assert!(!verifier_source
        .source
        .contains("stage6.hamming_booleanity.output.family"));
    assert!(verifier_source
        .source
        .contains("stage6.hamming_booleanity.output.claim_expr"));
    assert!(verifier_source
        .source
        .contains("stage6.hamming_booleanity.output.term0.boolean_zero"));
    assert!(!verifier_source
        .source
        .contains("stage6.ram_ra_virtual.output.family"));
    assert!(!verifier_source
        .source
        .contains("stage6.instruction_ra_virtual.output.family"));
    assert!(verifier_source
        .source
        .contains("stage6.inc_claim_reduction.output.eq.RdIncStage5"));
    assert!(!verifier_source
        .source
        .contains("stage6.inc_claim_reduction.output.family"));
    assert!(verifier_source
        .source
        .contains("stage6.inc_claim_reduction.output.claim_expr"));
    assert!(verifier_source
        .source
        .contains("stage6.inc_claim_reduction.output.gamma_pow_3"));
    assert!(verifier_source.source.contains("Stage6FieldExprKind::Sum"));
    assert!(verifier_source
        .source
        .contains("Stage6FieldExprKind::Product"));
    assert!(!verifier_source
        .source
        .contains("stage6.hamming_booleanity.output.square.HammingWeight"));
    assert!(!verifier_source
        .source
        .contains("stage6.hamming_booleanity.output.neg.HammingWeight"));
    assert!(!verifier_source
        .source
        .contains("stage6.hamming_booleanity.output.gamma_identity"));
    assert!(!verifier_source
        .source
        .contains("stage6.ram_ra_virtual.output.product.RamRa"));
    assert!(verifier_source
        .source
        .contains("stage6.ram_ra_virtual.output.claim_expr"));
    assert!(!verifier_source
        .source
        .contains("stage6.ram_ra_virtual.output.gamma_identity"));
    assert!(!verifier_source
        .source
        .contains("stage6.instruction_ra_virtual.output.product.InstructionRa_0"));
    assert!(!verifier_source
        .source
        .contains("stage6.instruction_ra_virtual.output.weighted_sum"));
    assert!(!verifier_source
        .source
        .contains("stage6.instruction_ra_virtual.output.term.InstructionRa_"));
    assert!(verifier_source
        .source
        .contains("stage6.instruction_ra_virtual.output.gamma_pow_7"));
    assert!(verifier_source
        .source
        .contains("stage6.instruction_ra_virtual.output.claim_expr"));
    assert!(verifier_source
        .source
        .contains("stage6.inc_claim_reduction.output.term0"));
    assert!(verifier_source
        .source
        .contains("stage6.bytecode_read_raf.eval.BytecodeRa_0"));
    assert!(verifier_source
        .source
        .contains("stage6.booleanity.eval.InstructionRa_31"));
    assert!(verifier_source
        .source
        .contains("stage6.inc_claim_reduction.eval.RdInc"));
    assert!(verifier_source
        .source
        .contains("bolt_verifier_runtime::verify_batched_sumcheck"));
    assert!(verifier_source.source.contains("stage6_verifier_program"));
    assert_rust_source_compiles(&prover_source.filename, &prover_source.source);
    assert_rust_source_compiles(&verifier_source.filename, &verifier_source.source);
}

#[test]
fn stage7_rust_targets_extract_and_compile() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let total_ra = params.instruction_d + params.bytecode_d + params.ram_d;
    let (prover_cpu, verifier_cpu) = build_stage7_pipeline_cpu(&context, &params);
    let prover_program = stage7_cpu_program(&prover_cpu).expect("extract prover stage7 program");
    let verifier_program =
        stage7_cpu_program(&verifier_cpu).expect("extract verifier stage7 program");

    assert_eq!(prover_program.role, Role::Prover);
    assert_eq!(verifier_program.role, Role::Verifier);
    assert_eq!(prover_program.kernels.len(), 2);
    assert!(verifier_program.kernels.is_empty());
    assert_eq!(prover_program.steps.len(), 2);
    assert_eq!(prover_program.transcript_squeezes.len(), 1);
    assert!(prover_program.transcript_absorb_bytes.is_empty());
    assert_eq!(prover_program.opening_inputs.len(), 1 + 2 * total_ra);
    assert_eq!(prover_program.field_constants.len(), 1);
    assert!(prover_program.field_exprs.len() >= 3 * total_ra);
    assert_eq!(prover_program.claims.len(), 1);
    assert_eq!(prover_program.batches.len(), 1);
    assert_eq!(prover_program.drivers.len(), 1);
    assert_eq!(prover_program.instance_results.len(), 1);
    assert_eq!(prover_program.evals.len(), total_ra);
    assert_eq!(prover_program.relation_output_values.len(), total_ra + 1);
    assert!(prover_program.relation_outputs.is_empty());
    assert_eq!(verifier_program.relation_output_values.len(), total_ra + 1);
    assert!(verifier_program.relation_output_eval_families.is_empty());
    assert_eq!(verifier_program.relation_outputs.len(), 1);
    assert_eq!(
        verifier_program.relation_outputs[0].expected_output,
        "stage7.hamming_weight_claim_reduction.output.claim_expr"
    );
    let input_claim = verifier_program
        .claims
        .iter()
        .find(|claim| claim.symbol == "stage7.hamming_weight_claim_reduction.input")
        .expect("stage7 hamming input claim exists");
    assert_eq!(
        input_claim.claim_value,
        "stage7.hamming_weight_claim_reduction.input.claim_expr"
    );
    let input_expr = verifier_program
        .scalar_exprs
        .iter()
        .find(|expr| expr.symbol == "stage7.hamming_weight_claim_reduction.input.claim_expr")
        .expect("stage7 hamming input claim is lowered to a scalar expression");
    assert_eq!(
        input_expr.formula,
        format!("field.power_strided_weighted_sum:{total_ra}:3:_:_:0,1,2")
    );
    assert_eq!(input_expr.operands.len(), 1 + total_ra + 3 * total_ra);
    let output_expr = verifier_program
        .scalar_exprs
        .iter()
        .find(|expr| expr.symbol == "stage7.hamming_weight_claim_reduction.output.claim_expr")
        .expect("stage7 hamming output is lowered to a scalar expression");
    assert_eq!(
        output_expr.formula,
        format!("field.power_strided_weighted_sum:{total_ra}:3:0:1:2")
    );
    assert_eq!(output_expr.operands.len(), 2 * total_ra + 2);
    assert_eq!(
        output_expr.operands[0],
        "stage7.hamming_weight_claim_reduction.gamma"
    );
    assert_eq!(
        output_expr.operands[1 + total_ra],
        "stage7.hamming_weight_claim_reduction.output.eq.Booleanity"
    );
    assert!(prover_program.point_zeros.is_empty());
    assert_eq!(prover_program.point_slices.len(), 1);
    assert_eq!(prover_program.point_concats.len(), 1);
    assert_eq!(prover_program.opening_claims.len(), total_ra);
    assert_eq!(prover_program.opening_batches.len(), 1);
    assert!(prover_program
        .drivers
        .iter()
        .any(|driver| driver.kernel.as_deref() == Some("jolt.cpu.stage7.batched")));
    assert!(prover_program.claims.iter().any(|claim| {
        claim.symbol == "stage7.hamming_weight_claim_reduction.input"
            && claim.kernel.as_deref() == Some("jolt.cpu.stage7.hamming_weight_claim_reduction")
    }));
    assert!(prover_program.opening_claims.iter().any(|claim| {
        claim.symbol == "stage7.hamming_weight_claim_reduction.opening.InstructionRa_0"
            && claim.point_source == "stage7.hamming_weight_claim_reduction.point"
    }));
    assert!(verifier_program
        .claims
        .iter()
        .all(|claim| claim.kernel.is_none() && claim.relation.is_some()));
    assert!(verifier_program
        .drivers
        .iter()
        .all(|driver| driver.kernel.is_none() && driver.relation.is_some()));

    let prover_source = emit_stage7_rust(&prover_cpu).expect("emit stage7 prover rust");
    let verifier_source = emit_stage7_rust(&verifier_cpu).expect("emit stage7 verifier rust");
    assert_eq!(prover_source.filename, "prove_stage7.rs");
    assert_eq!(verifier_source.filename, "verify_stage7.rs");
    assert!(prover_source
        .source
        .contains("jolt_stage7_hamming_weight_claim_reduction"));
    assert!(prover_source.source.contains("Stage7KernelExecutor"));
    assert!(prover_source.source.contains("execute_stage7_program"));
    assert!(prover_source.source.contains("execute_stage7_prover"));
    assert!(!verifier_source.source.contains("jolt_kernels"));
    assert!(verifier_source.source.contains("Stage7VerifierProgramPlan"));
    assert!(verifier_source.source.contains("pub fn verify_stage7"));
    assert!(verifier_source
        .source
        .contains("relation: Some(Stage7RelationKind::Stage7Batched)"));
    assert!(verifier_source
        .source
        .contains("Stage7RelationKind::Stage7HammingWeightClaimReduction"));
    assert!(!verifier_source
        .source
        .contains("expected_hamming_weight_claim_reduction"));
    assert!(verifier_source.source.contains("Stage7RelationOutputPlan"));
    assert!(!verifier_source
        .source
        .contains("RelationOutputEvalFamilyPlan"));
    assert!(!verifier_source
        .source
        .contains("STAGE7_RELATION_OUTPUT_0_FAMILY_0_EVALS"));
    assert!(verifier_source
        .source
        .contains("Stage7ScalarExprKind::PowerStridedWeightedSum"));
    assert!(!verifier_source
        .source
        .contains("Stage7FieldExprKind::PowerStridedWeightedSum"));
    assert!(!verifier_source
        .source
        .contains("stage7.hamming_weight_claim_reduction.claim_expr.partial"));
    assert!(verifier_source
        .source
        .contains("stage7.hamming_weight_claim_reduction.output.eq.Booleanity"));
    assert!(verifier_source.source.contains(
        "stage7.hamming_weight_claim_reduction.output.eq.InstructionRa_0.virtualization"
    ));
    assert!(!verifier_source
        .source
        .contains("stage7.hamming_weight_claim_reduction.output.term"));
    assert!(verifier_source
        .source
        .contains("bolt_verifier_runtime::evaluate_relation_output_batch"));
    assert!(verifier_source
        .source
        .contains("stage7.input.stage6.booleanity.InstructionRa_0"));
    assert!(verifier_source
        .source
        .contains("stage7.hamming_weight_claim_reduction.eval.InstructionRa_0"));
    assert!(verifier_source
        .source
        .contains("bolt_verifier_runtime::verify_batched_sumcheck"));
    assert!(verifier_source.source.contains("stage7_verifier_program"));
    assert_rust_source_compiles(&prover_source.filename, &prover_source.source);
    assert_rust_source_compiles(&verifier_source.filename, &verifier_source.source);
}

#[test]
fn stage8_rust_targets_extract_and_compile() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let expected_claims = params.num_committed;
    let (prover_cpu, verifier_cpu) = build_stage8_pipeline_cpu(&context, &params);
    let prover_program = stage8_cpu_program(&prover_cpu).expect("extract prover stage8 program");
    let verifier_program =
        stage8_cpu_program(&verifier_cpu).expect("extract verifier stage8 program");

    assert_eq!(prover_program.role, Role::Prover);
    assert_eq!(verifier_program.role, Role::Verifier);
    assert_eq!(prover_program.opening_inputs.len(), expected_claims + 1);
    assert_eq!(prover_program.opening_claims.len(), expected_claims);
    assert_eq!(prover_program.opening_batches.len(), 1);
    assert_eq!(prover_program.pcs_proofs.len(), 1);
    assert_eq!(prover_program.pcs_proofs[0].mode, "open");
    assert_eq!(verifier_program.pcs_proofs[0].mode, "verify");
    assert_eq!(
        prover_program.opening_batches[0].ordered_claims,
        prover_program.opening_batches[0].claim_operands
    );
    assert!(prover_program.opening_claims.iter().any(|claim| {
        claim.symbol == "stage8.evaluation.opening.RamInc"
            && claim.source_claim == "stage6.inc_claim_reduction.eval.RamInc"
    }));
    assert!(prover_program.opening_claims.iter().any(|claim| {
        claim.symbol == "stage8.evaluation.opening.InstructionRa_0"
            && claim.source_claim == "stage7.hamming_weight_claim_reduction.eval.InstructionRa_0"
    }));

    let prover_source = emit_stage8_rust(&prover_cpu).expect("emit stage8 prover rust");
    let verifier_source = emit_stage8_rust(&verifier_cpu).expect("emit stage8 verifier rust");
    assert_eq!(prover_source.filename, "prove_stage8.rs");
    assert_eq!(verifier_source.filename, "verify_stage8.rs");
    assert!(prover_source.source.contains("pub const STAGE8_PROGRAM"));
    assert!(prover_source
        .source
        .contains("stage8.evaluation.point_source"));
    assert!(prover_source.source.contains("jolt_stage8_joint_rlc"));
    assert!(prover_source
        .source
        .contains("stage6.inc_claim_reduction.eval.RamInc"));
    assert!(prover_source
        .source
        .contains("stage7.hamming_weight_claim_reduction.eval.InstructionRa_0"));
    assert!(verifier_source
        .source
        .contains("mode: Stage8PcsProofMode::Verify"));
    assert_rust_source_compiles(&prover_source.filename, &prover_source.source);
    assert_rust_source_compiles(&verifier_source.filename, &verifier_source.source);
}

#[test]
fn stage4_generated_artifact_crates_compile_in_isolation() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let (prover_cpu, verifier_cpu) = build_stage4_pipeline_cpu(&context, &params);
    let stage = ProtocolStage::new("stage4", "stage4", 4, ProtocolStageKind::Proof);
    let config = jolt_artifact_config();
    let artifacts = vec![
        protocol_rust_artifact(
            &config,
            stage.clone(),
            Role::Prover,
            emit_stage4_rust(&prover_cpu).expect("emit stage4 prover"),
        ),
        protocol_rust_artifact(
            &config,
            stage,
            Role::Verifier,
            emit_stage4_rust(&verifier_cpu).expect("emit stage4 verifier"),
        ),
    ];
    for artifact in &artifacts {
        validate_jolt_rust_artifact_imports(artifact).expect("stage4 import policy");
    }
    if !generated_jolt_runtime_available() {
        return;
    }

    let output_root = new_temp_dir("bolt_stage4_generated_crates");
    let dependency_root = workspace_root().join("crates");
    let generated_crates =
        assemble_jolt_generated_crates(artifacts, &dependency_root.display().to_string())
            .expect("assemble stage4 crates");
    write_jolt_generated_crates(&generated_crates, &output_root)
        .expect("write stage4 generated crates");
    redirect_generated_prover_to_generated_verifier(&output_root, &dependency_root);
    for generated in &generated_crates {
        assert_generated_crate_manifest_compiles(&output_root, &generated.crate_name);
    }
    let _ = std::fs::remove_dir_all(output_root);
}

#[test]
fn generic_artifact_assembly_supports_non_jolt_protocol_config() {
    let config = non_jolt_artifact_config();
    let stage = ProtocolStage::new("alpha", "alpha", 1, ProtocolStageKind::Proof);
    let artifacts = vec![
        protocol_rust_artifact(
            &config,
            stage.clone(),
            Role::Prover,
            non_jolt_alpha_prover_source(),
        ),
        protocol_rust_artifact(
            &config,
            stage,
            Role::Verifier,
            non_jolt_alpha_verifier_source(),
        ),
    ];
    assert_eq!(
        artifacts[0].path, "acme-prover/src/stages/alpha.rs",
        "generic artifact path should derive from config and stage module"
    );
    assert_eq!(
        artifacts[1].path, "acme-verifier/src/stages/alpha.rs",
        "generic artifact path should derive from config and stage module"
    );
    for artifact in &artifacts {
        validate_rust_artifact_imports(&config, artifact).expect("generic import policy");
    }

    let generated =
        assemble_generated_crates(&config, artifacts, "../deps").expect("assemble generic crates");
    let prover = generated
        .iter()
        .find(|generated| generated.crate_name == "acme-prover")
        .expect("generated prover crate");
    let verifier = generated
        .iter()
        .find(|generated| generated.crate_name == "acme-verifier")
        .expect("generated verifier crate");

    let prover_manifest = prover
        .files
        .iter()
        .find(|file| file.path == "Cargo.toml")
        .expect("prover manifest")
        .source
        .as_str();
    assert!(prover_manifest.contains("name = \"acme-prover\""));
    assert!(prover_manifest.contains("acme-verifier = { path = \"../deps/acme-verifier\" }"));
    assert!(prover_manifest.contains("serde = { version = \"1\", default-features = false }"));
    assert!(!prover_manifest.contains("serde = { path = "));
    assert!(prover.files.iter().any(|file| file.path == "src/prover.rs"));
    assert!(prover
        .files
        .iter()
        .any(|file| file.path == "src/stages/alpha.rs"));

    let verifier_stages = verifier
        .files
        .iter()
        .find(|file| file.path == "src/stages/mod.rs")
        .expect("verifier stages module")
        .source
        .as_str();
    assert!(verifier_stages.contains("pub mod shared;"));
    assert!(verifier_stages.contains("pub mod alpha;"));
    assert!(verifier
        .files
        .iter()
        .any(|file| file.path == "src/verifier.rs"));
    assert!(verifier
        .files
        .iter()
        .any(|file| file.path == "src/stages/shared.rs"));

    let generated_surface = generated
        .iter()
        .flat_map(|generated| generated.files.iter())
        .map(|file| file.source.as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(
        !generated_surface.contains("jolt") && !generated_surface.contains("Jolt"),
        "generic artifact assembly leaked Jolt names into a non-Jolt protocol fixture"
    );
    assert!(
        !generated_surface.contains("ark-bn254") && !generated_surface.contains("arkworks-algebra"),
        "generic artifact assembly leaked Jolt/arkworks standalone manifest patches into a non-Jolt protocol fixture"
    );
    assert!(generated_surface.contains("pub const TRANSCRIPT_LABEL: &[u8] = b\"acme transcript\";"));
    assert!(generated_surface.contains("crate::stages::shared::StageProof"));
    assert!(generated_surface.contains("pub fn prove_acme"));
    assert!(generated_surface.contains("pub fn verify_acme"));
}

#[test]
fn generated_jolt_artifacts_have_uniform_crate_layout_and_import_rules() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let (commitment_prover_cpu, commitment_verifier_cpu) =
        build_commitment_pipeline_cpu(&context, &params);
    let (stage1_prover_cpu, stage1_verifier_cpu) = build_stage1_pipeline_cpu(&context, &params);
    let (stage2_prover_cpu, stage2_verifier_cpu) = build_stage2_pipeline_cpu(&context, &params);
    let (stage3_prover_cpu, stage3_verifier_cpu) = build_stage3_pipeline_cpu(&context, &params);
    let (stage4_prover_cpu, stage4_verifier_cpu) = build_stage4_pipeline_cpu(&context, &params);
    let (stage5_prover_cpu, stage5_verifier_cpu) = build_stage5_pipeline_cpu(&context, &params);
    let (stage6_prover_cpu, stage6_verifier_cpu) = build_stage6_pipeline_cpu(&context, &params);
    let (stage7_prover_cpu, stage7_verifier_cpu) = build_stage7_pipeline_cpu(&context, &params);
    let (stage8_prover_cpu, stage8_verifier_cpu) = build_stage8_pipeline_cpu(&context, &params);

    let emitted = [
        (
            JoltProtocolStage::Commitment,
            Role::Prover,
            emit_commitment_rust(&commitment_prover_cpu).expect("emit commitment prover"),
        ),
        (
            JoltProtocolStage::Commitment,
            Role::Verifier,
            emit_commitment_rust(&commitment_verifier_cpu).expect("emit commitment verifier"),
        ),
        (
            JoltProtocolStage::Stage1Outer,
            Role::Prover,
            emit_stage1_rust(&stage1_prover_cpu).expect("emit stage1 prover"),
        ),
        (
            JoltProtocolStage::Stage1Outer,
            Role::Verifier,
            emit_stage1_rust(&stage1_verifier_cpu).expect("emit stage1 verifier"),
        ),
        (
            JoltProtocolStage::Stage2,
            Role::Prover,
            emit_stage2_rust(&stage2_prover_cpu).expect("emit stage2 prover"),
        ),
        (
            JoltProtocolStage::Stage2,
            Role::Verifier,
            emit_stage2_rust(&stage2_verifier_cpu).expect("emit stage2 verifier"),
        ),
        (
            JoltProtocolStage::Stage3,
            Role::Prover,
            emit_stage3_rust(&stage3_prover_cpu).expect("emit stage3 prover"),
        ),
        (
            JoltProtocolStage::Stage3,
            Role::Verifier,
            emit_stage3_rust(&stage3_verifier_cpu).expect("emit stage3 verifier"),
        ),
        (
            JoltProtocolStage::Stage4,
            Role::Prover,
            emit_stage4_rust(&stage4_prover_cpu).expect("emit stage4 prover"),
        ),
        (
            JoltProtocolStage::Stage4,
            Role::Verifier,
            emit_stage4_rust(&stage4_verifier_cpu).expect("emit stage4 verifier"),
        ),
        (
            JoltProtocolStage::Stage5,
            Role::Prover,
            emit_stage5_rust(&stage5_prover_cpu).expect("emit stage5 prover"),
        ),
        (
            JoltProtocolStage::Stage5,
            Role::Verifier,
            emit_stage5_rust(&stage5_verifier_cpu).expect("emit stage5 verifier"),
        ),
        (
            JoltProtocolStage::Stage6,
            Role::Prover,
            emit_stage6_rust(&stage6_prover_cpu).expect("emit stage6 prover"),
        ),
        (
            JoltProtocolStage::Stage6,
            Role::Verifier,
            emit_stage6_rust(&stage6_verifier_cpu).expect("emit stage6 verifier"),
        ),
        (
            JoltProtocolStage::Stage7,
            Role::Prover,
            emit_stage7_rust(&stage7_prover_cpu).expect("emit stage7 prover"),
        ),
        (
            JoltProtocolStage::Stage7,
            Role::Verifier,
            emit_stage7_rust(&stage7_verifier_cpu).expect("emit stage7 verifier"),
        ),
        (
            JoltProtocolStage::Stage8,
            Role::Prover,
            emit_stage8_rust(&stage8_prover_cpu).expect("emit stage8 prover"),
        ),
        (
            JoltProtocolStage::Stage8,
            Role::Verifier,
            emit_stage8_rust(&stage8_verifier_cpu).expect("emit stage8 verifier"),
        ),
    ];
    let artifacts = emitted
        .into_iter()
        .map(|(stage, role, source)| {
            let artifact = jolt_rust_artifact(stage, role, source).expect("canonical artifact");
            validate_jolt_rust_artifact_imports(&artifact).expect("artifact import policy");
            artifact
        })
        .collect::<Vec<_>>();

    let paths = artifacts
        .iter()
        .map(|artifact| artifact.path.as_str())
        .collect::<Vec<_>>();
    assert_eq!(
        paths,
        vec![
            "jolt-prover/src/stages/commitment.rs",
            "jolt-verifier/src/stages/commitment.rs",
            "jolt-prover/src/stages/stage1_outer.rs",
            "jolt-verifier/src/stages/stage1_outer.rs",
            "jolt-prover/src/stages/stage2.rs",
            "jolt-verifier/src/stages/stage2.rs",
            "jolt-prover/src/stages/stage3.rs",
            "jolt-verifier/src/stages/stage3.rs",
            "jolt-prover/src/stages/stage4.rs",
            "jolt-verifier/src/stages/stage4.rs",
            "jolt-prover/src/stages/stage5.rs",
            "jolt-verifier/src/stages/stage5.rs",
            "jolt-prover/src/stages/stage6.rs",
            "jolt-verifier/src/stages/stage6.rs",
            "jolt-prover/src/stages/stage7.rs",
            "jolt-verifier/src/stages/stage7.rs",
            "jolt-prover/src/stages/stage8.rs",
            "jolt-verifier/src/stages/stage8.rs",
        ]
    );
    assert!(artifacts
        .iter()
        .filter(|artifact| artifact.crate_name == "jolt-verifier")
        .all(|artifact| !artifact.source.source.contains("jolt_kernels")));
    assert!(artifacts
        .iter()
        .filter(|artifact| { artifact.crate_name == "jolt-prover" && artifact.stage.is_proof() })
        .all(|artifact| artifact.source.source.contains("jolt_kernels")));
    let workspace_generated_crates = assemble_jolt_workspace_generated_crates(artifacts.clone())
        .expect("assemble workspace generated role crates");
    if std::env::var_os("JOLT_UPDATE_GOLDENS").is_some() {
        write_jolt_generated_crates(&workspace_generated_crates, workspace_root().join("crates"))
            .expect("update checked-in generated role crates");
    }
    if !checked_in_generated_role_crates_available() {
        return;
    }
    assert_checked_in_generated_role_crate_sources_match(&workspace_generated_crates);
    let dependency_root = workspace_root().join("crates").display().to_string();
    let generated_crates = assemble_jolt_generated_crates(artifacts, &dependency_root)
        .expect("assemble generated role crates");
    assert_eq!(
        generated_crates
            .iter()
            .map(|generated| generated.crate_name.as_str())
            .collect::<Vec<_>>(),
        vec!["jolt-prover", "jolt-verifier"]
    );
    for generated in &generated_crates {
        assert_generated_role_crate_compiles(generated);
    }
    let output_root = new_temp_dir("bolt_generated_crates");
    write_jolt_generated_crates(&workspace_generated_crates, &output_root)
        .expect("write generated role crates");
    for generated in &workspace_generated_crates {
        for file in &generated.files {
            assert!(output_root
                .join(&generated.crate_name)
                .join(&file.path)
                .exists());
        }
    }
    let _ = std::fs::remove_dir_all(output_root);
}

#[test]
fn jolt_stage1_outer_lowers_to_compute_and_cpu_kernel_ir() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let protocol =
        build_stage1_outer_protocol(&context, &params).expect("build stage1 outer protocol");
    let concrete =
        lower_piop_and_fiat_shamir(&context, &protocol).expect("lower stage1 to concrete");
    let prover = project_prover_party(&context, &concrete).expect("project prover party");
    let verifier = project_verifier_party(&context, &concrete).expect("project verifier party");
    let prover_compute = lower_stage1_to_compute(&context, &prover).expect("lower prover stage1");
    let verifier_compute =
        lower_stage1_to_compute(&context, &verifier).expect("lower verifier stage1");
    verify_compute_schema(&prover_compute).expect("prover stage1 compute schema is valid");
    verify_compute_schema(&verifier_compute).expect("verifier stage1 compute schema is valid");
    assert!(prover_compute
        .to_text_mlir()
        .contains("relation = @jolt.stage1.outer.uniskip"));
    assert!(!prover_compute.to_text_mlir().contains("kernel = @"));
    let verifier_compute_text = verifier_compute.to_text_mlir();
    assert!(verifier_compute_text.contains("\"compute.sumcheck_verify_claim\""));
    assert!(verifier_compute_text.contains("\"compute.sumcheck_verify\""));
    assert!(!verifier_compute_text.contains("\"compute.kernel\""));
    assert!(!verifier_compute_text.contains("kernel = @"));

    let prover_kernel_compute =
        resolve_compute_kernels(&context, &prover_compute).expect("resolve prover kernels");
    let verifier_kernel_compute =
        resolve_compute_kernels(&context, &verifier_compute).expect("resolve verifier kernels");
    verify_compute_schema(&prover_kernel_compute)
        .expect("prover kernelized stage1 compute schema is valid");
    verify_compute_schema(&verifier_kernel_compute)
        .expect("verifier kernelized stage1 compute schema is valid");

    let prover_cpu =
        lower_compute_to_cpu(&context, &prover_kernel_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_kernel_compute).expect("lower verifier CPU");
    verify_cpu_schema(&prover_cpu).expect("prover stage1 CPU schema is valid");
    verify_cpu_schema(&verifier_cpu).expect("verifier stage1 CPU schema is valid");
    let program = stage1_cpu_program(&prover_cpu).expect("extract prover stage1 CPU program");

    let cpu_text = prover_cpu.to_text_mlir();
    let verifier_cpu_text = verifier_cpu.to_text_mlir();
    assert!(cpu_text.contains("\"cpu.kernel\"()"));
    assert!(cpu_text.contains("kernel = @jolt.cpu.stage1.outer.uniskip"));
    assert!(cpu_text.contains("kernel = @jolt.cpu.stage1.outer.remaining"));
    assert!(cpu_text.contains("\"cpu.sumcheck_driver\"(%"));
    assert!(cpu_text.contains("\"cpu.sumcheck_eval\"(%"));
    assert!(cpu_text.contains("\"cpu.opening_claim\"(%"));
    assert!(cpu_text.contains("\"cpu.opening_batch\"(%"));
    assert!(cpu_text.contains("\"cpu.sumcheck_claim\"(%"));
    assert!(cpu_text.contains("count = 35 : i64"));
    assert!(!cpu_text.contains("\"cpu.pcs_opening_claim\""));
    assert!(verifier_cpu_text.contains("\"cpu.sumcheck_verify_claim\""));
    assert!(verifier_cpu_text.contains("\"cpu.sumcheck_verify\""));
    assert!(!verifier_cpu_text.contains("\"cpu.kernel\""));
    assert!(!verifier_cpu_text.contains("kernel = @"));
    assert_eq!(program.role, Role::Prover);
    assert_eq!(program.kernels.len(), 2);
    assert!(program.kernels.iter().any(|kernel| {
        kernel.symbol == "jolt.cpu.stage1.outer.uniskip"
            && kernel.relation == "jolt.stage1.outer.uniskip"
            && kernel.abi == "jolt_stage1_outer_uniskip"
    }));
    assert!(program.kernels.iter().any(|kernel| {
        kernel.symbol == "jolt.cpu.stage1.outer.remaining"
            && kernel.relation == "jolt.stage1.outer.remaining"
            && kernel.abi == "jolt_stage1_outer_remaining"
    }));
    assert_eq!(program.claims.len(), 2);
    assert_eq!(program.batches.len(), 2);
    assert_eq!(program.drivers.len(), 2);
    assert_eq!(program.opening_claims.len(), 36);
    assert_eq!(program.opening_batches.len(), 1);
    let uniskip = program
        .drivers
        .iter()
        .find(|driver| driver.symbol == "stage1.uniskip.sumcheck")
        .expect("uniskip driver");
    assert_eq!(
        uniskip.kernel.as_deref(),
        Some("jolt.cpu.stage1.outer.uniskip")
    );
    assert_eq!(uniskip.round_schedule, vec![1]);
    assert_eq!(uniskip.num_rounds, 1);
    assert_eq!(uniskip.degree, 27);
    let remaining = program
        .drivers
        .iter()
        .find(|driver| driver.symbol == "stage1.outer_remaining.sumcheck")
        .expect("remaining driver");
    assert_eq!(
        remaining.kernel.as_deref(),
        Some("jolt.cpu.stage1.outer.remaining")
    );
    assert_eq!(remaining.round_schedule, vec![params.log_t + 1]);
    assert_eq!(remaining.num_rounds, params.log_t + 1);
    assert_eq!(remaining.degree, 3);
    assert_eq!(
        program
            .evals
            .iter()
            .filter(|eval| eval.source == "stage1.outer_remaining.sumcheck")
            .count(),
        35
    );
    assert_eq!(program.opening_batches[0].count, 35);
    assert_eq!(
        program.opening_batches[0].ordered_claims,
        program.opening_batches[0].claim_operands
    );

    assert_or_update_fixture(
        "tests/fixtures/stage1_outer_prover_compute.mlir",
        &prover_compute.to_text_mlir(),
    );
    assert_or_update_fixture(
        "tests/fixtures/stage1_outer_verifier_compute.mlir",
        &verifier_compute.to_text_mlir(),
    );
    assert_or_update_fixture(
        "tests/fixtures/stage1_outer_prover_kernel_compute.mlir",
        &prover_kernel_compute.to_text_mlir(),
    );
    assert_or_update_fixture(
        "tests/fixtures/stage1_outer_verifier_kernel_compute.mlir",
        &verifier_kernel_compute.to_text_mlir(),
    );
    assert_or_update_fixture(
        "tests/fixtures/stage1_outer_prover_cpu.mlir",
        &prover_cpu.to_text_mlir(),
    );
    assert_or_update_fixture(
        "tests/fixtures/stage1_outer_verifier_cpu.mlir",
        &verifier_cpu.to_text_mlir(),
    );
}

#[test]
fn stage1_rust_emission_matches_golden_and_compiles() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let protocol =
        build_stage1_outer_protocol(&context, &params).expect("build stage1 outer protocol");
    let concrete =
        lower_piop_and_fiat_shamir(&context, &protocol).expect("lower stage1 to concrete");
    let prover = project_prover_party(&context, &concrete).expect("project prover party");
    let verifier = project_verifier_party(&context, &concrete).expect("project verifier party");
    let prover_compute = lower_stage1_to_compute(&context, &prover).expect("lower prover stage1");
    let verifier_compute =
        lower_stage1_to_compute(&context, &verifier).expect("lower verifier stage1");
    let prover_kernel_compute =
        resolve_compute_kernels(&context, &prover_compute).expect("resolve prover kernels");
    let verifier_kernel_compute =
        resolve_compute_kernels(&context, &verifier_compute).expect("resolve verifier kernels");
    let prover_cpu =
        lower_compute_to_cpu(&context, &prover_kernel_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_kernel_compute).expect("lower verifier CPU");
    let source = emit_stage1_rust(&prover_cpu).expect("emit prover stage1 rust");
    let verifier_source = emit_stage1_rust(&verifier_cpu).expect("emit verifier stage1 rust");

    assert_eq!(source.filename, "prove_stage1_outer.rs");
    assert_eq!(verifier_source.filename, "verify_stage1_outer.rs");
    assert!(source.source.contains("pub fn prove_stage1_outer"));
    assert!(verifier_source
        .source
        .contains("pub fn verify_stage1_outer"));
    assert!(source.source.contains("jolt_stage1_outer_uniskip"));
    assert!(source.source.contains("jolt_stage1_outer_remaining"));
    assert!(!verifier_source.source.contains("jolt_kernels"));
    assert!(verifier_source.source.contains("jolt_sumcheck"));
    assert_or_update_fixture("tests/fixtures/prove_stage1_outer.rs", &source.source);
    assert_or_update_fixture(
        "tests/fixtures/verify_stage1_outer.rs",
        &verifier_source.source,
    );
    assert_rust_source_compiles(&source.filename, &source.source);
    assert_rust_source_compiles(&verifier_source.filename, &verifier_source.source);
}

#[test]
fn generated_stage1_prover_shape_proof_verifier_accepts() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::new(2, 2, 2);
    let (prover_cpu, verifier_cpu) = build_stage1_pipeline_cpu(&context, &params);
    let prover_source = emit_stage1_rust(&prover_cpu).expect("emit stage1 prover rust");
    let verifier_source = emit_stage1_rust(&verifier_cpu).expect("emit stage1 verifier rust");

    assert_eq!(prover_source.filename, "prove_stage1_outer.rs");
    assert_eq!(verifier_source.filename, "verify_stage1_outer.rs");
    assert_generated_stage1_self_parity_runs(
        &prover_source,
        &verifier_source,
        &generated_stage1_shape_self_parity_main(),
    );
}

#[test]
fn generated_stage1_real_executor_reaches_kernel_dispatch() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::new(2, 2, 2);
    let (prover_cpu, verifier_cpu) = build_stage1_pipeline_cpu(&context, &params);
    let prover_source = emit_stage1_rust(&prover_cpu).expect("emit stage1 prover rust");
    let verifier_source = emit_stage1_rust(&verifier_cpu).expect("emit stage1 verifier rust");

    assert_generated_stage1_self_parity_runs(
        &prover_source,
        &verifier_source,
        generated_stage1_real_dispatch_main(),
    );
}

#[test]
fn generated_stage1_real_executor_self_verifies_synthetic_remaining() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::new(2, 2, 2);
    let (prover_cpu, verifier_cpu) = build_stage1_pipeline_cpu(&context, &params);
    let prover_source = emit_stage1_rust(&prover_cpu).expect("emit stage1 prover rust");
    let verifier_source = emit_stage1_rust(&verifier_cpu).expect("emit stage1 verifier rust");

    assert_generated_stage1_self_parity_runs(
        &prover_source,
        &verifier_source,
        &generated_stage1_synthetic_remaining_main(),
    );
}

#[test]
fn generated_stage1_real_executor_self_verifies_r1cs_data() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::new(2, 2, 2);
    let (prover_cpu, verifier_cpu) = build_stage1_pipeline_cpu(&context, &params);
    let prover_source = emit_stage1_rust(&prover_cpu).expect("emit stage1 prover rust");
    let verifier_source = emit_stage1_rust(&verifier_cpu).expect("emit stage1 verifier rust");

    assert_generated_stage1_self_parity_runs(
        &prover_source,
        &verifier_source,
        &generated_stage1_r1cs_data_main(),
    );
}

#[test]
fn jolt_protocol_chain_commitment_stage1_fixture_tracks_phase_order() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let chain = jolt_protocol_chain_commitment_stage1_fixture(&context, &params);

    assert_or_update_fixture(
        "tests/fixtures/jolt_protocol_chain_commitment_stage1.yaml",
        &chain,
    );
}

#[test]
fn generated_jolt_chain_commitment_then_stage1_self_parity_runs() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::new(2, 2, 2);
    let (commitment_prover_cpu, commitment_verifier_cpu) =
        build_commitment_pipeline_cpu(&context, &params);
    let (stage1_prover_cpu, stage1_verifier_cpu) = build_stage1_pipeline_cpu(&context, &params);
    let commitment_prover =
        emit_commitment_rust(&commitment_prover_cpu).expect("emit commitment prover rust");
    let commitment_verifier =
        emit_commitment_rust(&commitment_verifier_cpu).expect("emit commitment verifier rust");
    let stage1_prover = emit_stage1_rust(&stage1_prover_cpu).expect("emit stage1 prover rust");
    let stage1_verifier =
        emit_stage1_rust(&stage1_verifier_cpu).expect("emit stage1 verifier rust");

    assert_generated_jolt_chain_self_parity_runs(
        &[
            &commitment_prover,
            &commitment_verifier,
            &stage1_prover,
            &stage1_verifier,
        ],
        &generated_commitment_stage1_chain_main(),
    );
}

#[test]
fn commitment_pipeline_matches_golden_mlir_fixtures() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let protocol = build_commitment_protocol(&context, &params).expect("build protocol");
    let concrete =
        lower_piop_and_fiat_shamir(&context, &protocol).expect("lower Fiat-Shamir state");
    let prover = project_prover_party(&context, &concrete).expect("project prover party");
    let verifier = project_verifier_party(&context, &concrete).expect("project verifier party");
    let prover_compute = lower_commitment_to_compute(&context, &prover).expect("lower compute");
    let verifier_compute =
        lower_commitment_to_compute(&context, &verifier).expect("lower verifier compute");
    let prover_cpu = lower_compute_to_cpu(&context, &prover_compute).expect("lower to CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_compute).expect("lower verifier to CPU");

    assert_or_update_fixture(
        "tests/fixtures/commitment_protocol.mlir",
        &protocol.to_text_mlir(),
    );
    assert_or_update_fixture(
        "tests/fixtures/commitment_concrete.mlir",
        &concrete.to_text_mlir(),
    );
    assert_or_update_fixture(
        "tests/fixtures/commitment_prover_party.mlir",
        &prover.to_text_mlir(),
    );
    assert_or_update_fixture(
        "tests/fixtures/commitment_verifier_party.mlir",
        &verifier.to_text_mlir(),
    );
    assert_or_update_fixture(
        "tests/fixtures/commitment_prover_compute.mlir",
        &prover_compute.to_text_mlir(),
    );
    assert_or_update_fixture(
        "tests/fixtures/commitment_verifier_compute.mlir",
        &verifier_compute.to_text_mlir(),
    );
    assert_or_update_fixture(
        "tests/fixtures/commitment_prover_cpu.mlir",
        &prover_cpu.to_text_mlir(),
    );
    assert_or_update_fixture(
        "tests/fixtures/commitment_verifier_cpu.mlir",
        &verifier_cpu.to_text_mlir(),
    );
}

#[test]
fn commitment_rust_emission_matches_golden_and_compiles() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::fixture();
    let protocol = build_commitment_protocol(&context, &params).expect("build protocol");
    let concrete =
        lower_piop_and_fiat_shamir(&context, &protocol).expect("lower Fiat-Shamir state");
    let prover = project_prover_party(&context, &concrete).expect("project prover party");
    let verifier = project_verifier_party(&context, &concrete).expect("project verifier party");
    let prover_compute = lower_commitment_to_compute(&context, &prover).expect("lower compute");
    let verifier_compute =
        lower_commitment_to_compute(&context, &verifier).expect("lower verifier compute");
    let prover_cpu = lower_compute_to_cpu(&context, &prover_compute).expect("lower to CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_compute).expect("lower verifier to CPU");
    let source = emit_commitment_rust(&prover_cpu).expect("emit prover commitment rust");
    let verifier_source =
        emit_commitment_rust(&verifier_cpu).expect("emit verifier commitment rust");

    assert_eq!(source.filename, "prove_commitment_phase.rs");
    assert_eq!(verifier_source.filename, "verify_commitment_phase.rs");
    assert_or_update_fixture("tests/fixtures/prove_commitment_phase.rs", &source.source);
    assert_or_update_fixture(
        "tests/fixtures/verify_commitment_phase.rs",
        &verifier_source.source,
    );
    assert_rust_source_compiles(&source.filename, &source.source);
    assert_rust_source_compiles(&verifier_source.filename, &verifier_source.source);
}

#[test]
fn generated_commitment_prover_verifier_self_parity_runs() {
    let context = MeliorContext::new();
    let prover_cpu = build_small_commitment_cpu(&context, Role::Prover);
    let verifier_cpu = build_small_commitment_cpu(&context, Role::Verifier);
    let prover_source =
        emit_commitment_rust(&prover_cpu).expect("emit small prover commitment rust");
    let verifier_source =
        emit_commitment_rust(&verifier_cpu).expect("emit small verifier commitment rust");

    assert_eq!(prover_source.filename, "prove_commitment_phase.rs");
    assert_eq!(verifier_source.filename, "verify_commitment_phase.rs");
    assert_generated_commitment_self_parity_runs(
        &prover_source,
        &verifier_source,
        &generated_small_self_parity_main(),
    );
}

#[test]
fn pipeline_generated_commitment_prover_verifier_self_parity_runs() {
    let context = MeliorContext::new();
    let params = JoltProtocolParams::new(0, 0, 0);
    let (prover_cpu, verifier_cpu) = build_commitment_pipeline_cpu(&context, &params);
    let prover_source =
        emit_commitment_rust(&prover_cpu).expect("emit pipeline prover commitment rust");
    let verifier_source =
        emit_commitment_rust(&verifier_cpu).expect("emit pipeline verifier commitment rust");

    assert_eq!(prover_source.filename, "prove_commitment_phase.rs");
    assert_eq!(verifier_source.filename, "verify_commitment_phase.rs");
    assert_generated_commitment_self_parity_runs(
        &prover_source,
        &verifier_source,
        &generated_pipeline_self_parity_main(),
    );
}

#[test]
fn commitment_rust_emission_requires_cpu_target_params() {
    let context = MeliorContext::new();
    let cpu = context
        .parse_module::<Cpu>(
            r#"
module @bad attributes {bolt.phase = "cpu", bolt.role = "prover"} {
  %0 = "cpu.oracle_family_init"() {count = 1 : i64, family = @bad.family, sym_name = "bad.family"} : () -> !cpu.oracle_family
  %1 = "cpu.oracle_ref"() {domain = @bad.domain, num_vars = 1 : i64, oracle = @A, sym_name = "bad.A"} : () -> !cpu.oracle_buffer
  %2 = "cpu.oracle_family_append"(%0, %1) {family = @bad.family, index = 0 : i64, oracle = @A, sym_name = "bad.family.append0"} : (!cpu.oracle_family, !cpu.oracle_buffer) -> !cpu.oracle_family
  %3 = "cpu.pcs_commit_batch"(%2) {artifact = @bad.artifact, count = 1 : i64, domain = @bad.domain, label = "bad", num_vars = 1 : i64, oracle_family = @bad.family, ordered_oracles = [@A], pcs = @dory, sym_name = "bad.batch"} : (!cpu.oracle_family) -> !cpu.commitment_artifact
}
"#,
        )
        .expect("parse bad CPU module");

    let error = emit_commitment_rust(&cpu).expect_err("missing params rejected");
    assert!(error.to_string().contains("missing cpu.params"));
}

fn build_small_commitment_cpu(context: &MeliorContext, role: Role) -> bolt::BoltModule<'_, Cpu> {
    let (batch_op, optional_op) = match role {
        Role::Prover => ("cpu.pcs_commit_batch", "cpu.pcs_commit_optional"),
        Role::Verifier => ("cpu.pcs_receive_batch", "cpu.pcs_receive_optional"),
    };
    context
        .parse_module::<Cpu>(&format!(
            r#"
module @small.commitment_phase attributes {{bolt.phase = "cpu", bolt.role = "{}"}} {{
  "cpu.params"() {{field = @bn254_fr, pcs = @dory, sym_name = "small.params", transcript = @blake2b_transcript}} : () -> ()
  "cpu.function"() {{source = @small.commitment_phase, sym_name = "small.commitment_phase"}} : () -> ()
  %0 = "cpu.transcript_init"() {{scheme = @blake2b_transcript, sym_name = "fs0"}} : () -> !cpu.transcript_state
  %1 = "cpu.oracle_family_init"() {{count = 2 : i64, family = @small.main_polys, sym_name = "small.main_polys"}} : () -> !cpu.oracle_family
  %2 = "cpu.oracle_ref"() {{domain = @small.domain, num_vars = 2 : i64, oracle = @A, sym_name = "small.A"}} : () -> !cpu.oracle_buffer
  %3 = "cpu.oracle_family_append"(%1, %2) {{family = @small.main_polys, index = 0 : i64, oracle = @A, sym_name = "small.main_polys.append0"}} : (!cpu.oracle_family, !cpu.oracle_buffer) -> !cpu.oracle_family
  %4 = "cpu.oracle_ref"() {{domain = @small.domain, num_vars = 2 : i64, oracle = @B, sym_name = "small.B"}} : () -> !cpu.oracle_buffer
  %5 = "cpu.oracle_family_append"(%3, %4) {{family = @small.main_polys, index = 1 : i64, oracle = @B, sym_name = "small.main_polys.append1"}} : (!cpu.oracle_family, !cpu.oracle_buffer) -> !cpu.oracle_family
  %6 = "{batch_op}"(%5) {{artifact = @small.main, count = 2 : i64, domain = @small.domain, label = "commitment", num_vars = 2 : i64, oracle_family = @small.main_polys, ordered_oracles = [@A, @B], pcs = @dory, sym_name = "small.main"}} : (!cpu.oracle_family) -> !cpu.commitment_artifact
  %7 = "cpu.oracle_ref"() {{domain = @small.domain, num_vars = 2 : i64, oracle = @Advice, sym_name = "small.Advice"}} : () -> !cpu.oracle_buffer
  %8 = "{optional_op}"(%7) {{artifact = @small.advice, domain = @small.domain, label = "advice", num_vars = 2 : i64, oracle = @Advice, pcs = @dory, skip_policy = "missing_or_zero", sym_name = "small.advice"}} : (!cpu.oracle_buffer) -> !cpu.commitment_artifact
  %9 = "cpu.transcript_absorb"(%0, %6) {{label = "commitment", optional = false, sym_name = "small.absorb_main"}} : (!cpu.transcript_state, !cpu.commitment_artifact) -> !cpu.transcript_state
  %10 = "cpu.transcript_absorb"(%9, %8) {{label = "advice", optional = true, sym_name = "small.absorb_advice"}} : (!cpu.transcript_state, !cpu.commitment_artifact) -> !cpu.transcript_state
}}
"#,
            role.as_str()
        ))
        .expect("parse small CPU module")
}

fn build_commitment_pipeline_cpu<'c>(
    context: &'c MeliorContext,
    params: &JoltProtocolParams,
) -> (bolt::BoltModule<'c, Cpu>, bolt::BoltModule<'c, Cpu>) {
    let protocol = build_commitment_protocol(context, params).expect("build protocol");
    let concrete = lower_piop_and_fiat_shamir(context, &protocol).expect("lower Fiat-Shamir state");
    let prover = project_prover_party(context, &concrete).expect("project prover party");
    let verifier = project_verifier_party(context, &concrete).expect("project verifier party");
    let prover_compute =
        lower_commitment_to_compute(context, &prover).expect("lower prover compute");
    let verifier_compute =
        lower_commitment_to_compute(context, &verifier).expect("lower verifier compute");
    let prover_cpu = lower_compute_to_cpu(context, &prover_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(context, &verifier_compute).expect("lower verifier CPU");
    (prover_cpu, verifier_cpu)
}

fn build_stage1_pipeline_cpu<'c>(
    context: &'c MeliorContext,
    params: &JoltProtocolParams,
) -> (bolt::BoltModule<'c, Cpu>, bolt::BoltModule<'c, Cpu>) {
    let protocol = build_stage1_outer_protocol(context, params).expect("build stage1 protocol");
    let concrete = lower_piop_and_fiat_shamir(context, &protocol).expect("lower stage1 protocol");
    let prover = project_prover_party(context, &concrete).expect("project prover party");
    let verifier = project_verifier_party(context, &concrete).expect("project verifier party");
    let prover_compute = lower_stage1_to_compute(context, &prover).expect("lower prover stage1");
    let verifier_compute =
        lower_stage1_to_compute(context, &verifier).expect("lower verifier stage1");
    let prover_compute =
        resolve_compute_kernels(context, &prover_compute).expect("resolve prover kernels");
    let verifier_compute =
        resolve_compute_kernels(context, &verifier_compute).expect("resolve verifier kernels");
    let prover_cpu = lower_compute_to_cpu(context, &prover_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(context, &verifier_compute).expect("lower verifier CPU");
    (prover_cpu, verifier_cpu)
}

fn build_stage2_pipeline_cpu<'c>(
    context: &'c MeliorContext,
    params: &JoltProtocolParams,
) -> (bolt::BoltModule<'c, Cpu>, bolt::BoltModule<'c, Cpu>) {
    let protocol = build_stage2_protocol(context, params).expect("build stage2 protocol");
    let concrete = lower_piop_and_fiat_shamir(context, &protocol).expect("lower stage2 protocol");
    let prover = project_prover_party(context, &concrete).expect("project prover party");
    let verifier = project_verifier_party(context, &concrete).expect("project verifier party");
    let prover_compute = lower_stage2_to_compute(context, &prover).expect("lower prover stage2");
    let verifier_compute =
        lower_stage2_to_compute(context, &verifier).expect("lower verifier stage2");
    let prover_compute =
        resolve_compute_kernels(context, &prover_compute).expect("resolve prover kernels");
    let verifier_compute =
        resolve_compute_kernels(context, &verifier_compute).expect("resolve verifier kernels");
    let prover_cpu = lower_compute_to_cpu(context, &prover_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(context, &verifier_compute).expect("lower verifier CPU");
    (prover_cpu, verifier_cpu)
}

fn build_stage3_pipeline_cpu<'c>(
    context: &'c MeliorContext,
    params: &JoltProtocolParams,
) -> (bolt::BoltModule<'c, Cpu>, bolt::BoltModule<'c, Cpu>) {
    let protocol = build_stage3_protocol(context, params).expect("build stage3 protocol");
    let concrete = lower_piop_and_fiat_shamir(context, &protocol).expect("lower stage3 protocol");
    let prover = project_prover_party(context, &concrete).expect("project prover party");
    let verifier = project_verifier_party(context, &concrete).expect("project verifier party");
    let prover_compute = lower_stage3_to_compute(context, &prover).expect("lower prover stage3");
    let verifier_compute =
        lower_stage3_to_compute(context, &verifier).expect("lower verifier stage3");
    let prover_compute =
        resolve_compute_kernels(context, &prover_compute).expect("resolve prover kernels");
    let verifier_compute =
        resolve_compute_kernels(context, &verifier_compute).expect("resolve verifier kernels");
    let prover_cpu = lower_compute_to_cpu(context, &prover_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(context, &verifier_compute).expect("lower verifier CPU");
    (prover_cpu, verifier_cpu)
}

fn build_stage4_pipeline_cpu<'c>(
    context: &'c MeliorContext,
    params: &JoltProtocolParams,
) -> (bolt::BoltModule<'c, Cpu>, bolt::BoltModule<'c, Cpu>) {
    let protocol = build_stage4_protocol(context, params).expect("build stage4 protocol");
    let concrete = lower_piop_and_fiat_shamir(context, &protocol).expect("lower stage4 protocol");
    let prover = project_prover_party(context, &concrete).expect("project prover party");
    let verifier = project_verifier_party(context, &concrete).expect("project verifier party");
    let prover_compute = lower_stage4_to_compute(context, &prover).expect("lower prover stage4");
    let verifier_compute =
        lower_stage4_to_compute(context, &verifier).expect("lower verifier stage4");
    let prover_compute =
        resolve_compute_kernels(context, &prover_compute).expect("resolve prover kernels");
    let verifier_compute =
        resolve_compute_kernels(context, &verifier_compute).expect("resolve verifier kernels");
    let prover_cpu = lower_compute_to_cpu(context, &prover_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(context, &verifier_compute).expect("lower verifier CPU");
    (prover_cpu, verifier_cpu)
}

fn build_stage5_pipeline_cpu<'c>(
    context: &'c MeliorContext,
    params: &JoltProtocolParams,
) -> (bolt::BoltModule<'c, Cpu>, bolt::BoltModule<'c, Cpu>) {
    let protocol = build_stage5_protocol(context, params).expect("build stage5 protocol");
    let concrete = lower_piop_and_fiat_shamir(context, &protocol).expect("lower stage5 protocol");
    let prover = project_prover_party(context, &concrete).expect("project prover party");
    let verifier = project_verifier_party(context, &concrete).expect("project verifier party");
    let prover_compute = lower_stage5_to_compute(context, &prover).expect("lower prover stage5");
    let verifier_compute =
        lower_stage5_to_compute(context, &verifier).expect("lower verifier stage5");
    let prover_compute =
        resolve_compute_kernels(context, &prover_compute).expect("resolve prover kernels");
    let verifier_compute =
        resolve_compute_kernels(context, &verifier_compute).expect("resolve verifier kernels");
    let prover_cpu = lower_compute_to_cpu(context, &prover_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(context, &verifier_compute).expect("lower verifier CPU");
    (prover_cpu, verifier_cpu)
}

fn build_stage6_pipeline_cpu<'c>(
    context: &'c MeliorContext,
    params: &JoltProtocolParams,
) -> (bolt::BoltModule<'c, Cpu>, bolt::BoltModule<'c, Cpu>) {
    let protocol = build_stage6_protocol(context, params).expect("build stage6 protocol");
    let concrete = lower_piop_and_fiat_shamir(context, &protocol).expect("lower stage6 protocol");
    let prover = project_prover_party(context, &concrete).expect("project prover party");
    let verifier = project_verifier_party(context, &concrete).expect("project verifier party");
    let prover_compute = lower_stage6_to_compute(context, &prover).expect("lower prover stage6");
    let verifier_compute =
        lower_stage6_to_compute(context, &verifier).expect("lower verifier stage6");
    let prover_compute =
        resolve_compute_kernels(context, &prover_compute).expect("resolve prover kernels");
    let verifier_compute =
        resolve_compute_kernels(context, &verifier_compute).expect("resolve verifier kernels");
    let prover_cpu = lower_compute_to_cpu(context, &prover_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(context, &verifier_compute).expect("lower verifier CPU");
    (prover_cpu, verifier_cpu)
}

fn build_stage7_pipeline_cpu<'c>(
    context: &'c MeliorContext,
    params: &JoltProtocolParams,
) -> (bolt::BoltModule<'c, Cpu>, bolt::BoltModule<'c, Cpu>) {
    let protocol = build_stage7_protocol(context, params).expect("build stage7 protocol");
    let concrete = lower_piop_and_fiat_shamir(context, &protocol).expect("lower stage7 protocol");
    let prover = project_prover_party(context, &concrete).expect("project prover party");
    let verifier = project_verifier_party(context, &concrete).expect("project verifier party");
    let prover_compute = lower_stage7_to_compute(context, &prover).expect("lower prover stage7");
    let verifier_compute =
        lower_stage7_to_compute(context, &verifier).expect("lower verifier stage7");
    let prover_compute =
        resolve_compute_kernels(context, &prover_compute).expect("resolve prover kernels");
    let verifier_compute =
        resolve_compute_kernels(context, &verifier_compute).expect("resolve verifier kernels");
    let prover_cpu = lower_compute_to_cpu(context, &prover_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(context, &verifier_compute).expect("lower verifier CPU");
    (prover_cpu, verifier_cpu)
}

fn build_stage8_pipeline_cpu<'c>(
    context: &'c MeliorContext,
    params: &JoltProtocolParams,
) -> (bolt::BoltModule<'c, Cpu>, bolt::BoltModule<'c, Cpu>) {
    let protocol = build_stage8_protocol(context, params).expect("build stage8 protocol");
    let concrete = lower_piop_and_fiat_shamir(context, &protocol).expect("lower stage8 protocol");
    let prover = project_prover_party(context, &concrete).expect("project prover party");
    let verifier = project_verifier_party(context, &concrete).expect("project verifier party");
    let prover_compute = lower_stage8_to_compute(context, &prover).expect("lower prover stage8");
    let verifier_compute =
        lower_stage8_to_compute(context, &verifier).expect("lower verifier stage8");
    let prover_compute =
        resolve_compute_kernels(context, &prover_compute).expect("resolve prover kernels");
    let verifier_compute =
        resolve_compute_kernels(context, &verifier_compute).expect("resolve verifier kernels");
    let prover_cpu = lower_compute_to_cpu(context, &prover_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(context, &verifier_compute).expect("lower verifier CPU");
    (prover_cpu, verifier_cpu)
}

fn non_jolt_artifact_config() -> ProtocolArtifactConfig {
    ProtocolArtifactConfig {
        protocol_name: "Acme".to_owned(),
        type_prefix: "Acme".to_owned(),
        transcript_label: "acme transcript".to_owned(),
        repository: None,
        prover_crate_name: "acme-prover".to_owned(),
        verifier_crate_name: "acme-verifier".to_owned(),
        crates_io_patches: Vec::new(),
        standalone_dependency_overrides: vec![ProtocolStandaloneDependency::new(
            "serde",
            "serde = { version = \"1\", default-features = false }",
        )],
        common_dependencies: vec!["serde".to_owned()],
        prover_dependencies: Vec::new(),
        verifier_dependencies: Vec::new(),
        instrumentation_prefix: None,
        prover_forbidden_imports: vec!["forbidden_prover".to_owned()],
        verifier_forbidden_imports: vec!["forbidden_verifier".to_owned()],
        kernel_crate: None,
        field_type: RustTypeRef::new("std::primitive::u64"),
        default_transcript_type: RustTypeRef::new("crate::stages::alpha::DefaultTranscript"),
        transcript_trait: RustTypeRef::new("crate::stages::alpha::Transcript"),
        commitment_type: RustTypeRef::new("crate::stages::shared::Commitment"),
        prover_setup_type: RustTypeRef::new("crate::stages::alpha::ProverSetup"),
        role_api_extension: None,
        verifier_runtime_modules: vec![ProtocolRuntimeModule {
            module_name: "shared".to_owned(),
            file: GeneratedFile {
                path: "src/stages/shared.rs".to_owned(),
                source: non_jolt_verifier_common_source(),
            },
        }],
        verifier_named_eval_type: RustTypeRef::new("crate::stages::shared::StageNamedEval"),
        verifier_sumcheck_output_type: RustTypeRef::new(
            "crate::stages::shared::StageSumcheckOutput",
        ),
        verifier_stage_proof_type: RustTypeRef::new("crate::stages::shared::StageProof"),
    }
}

fn non_jolt_alpha_prover_source() -> RustSourceFile {
    RustSourceFile {
        filename: "prove_alpha.rs".to_owned(),
        source: r"
pub struct DefaultTranscript<F>(core::marker::PhantomData<F>);

pub trait Transcript {
    type Challenge;
}

pub struct ProverSetup;

#[derive(Clone, Debug)]
pub struct AlphaExecutionArtifacts<F> {
    pub sumchecks: Vec<AlphaSumcheckOutput<F>>,
}

#[derive(Clone, Debug)]
pub struct AlphaSumcheckOutput<F> {
    pub driver: &'static str,
    pub point: Vec<F>,
    pub evals: Vec<AlphaNamedEval<F>>,
    pub proof: (),
}

#[derive(Clone, Debug)]
pub struct AlphaNamedEval<F> {
    pub name: &'static str,
    pub oracle: &'static str,
    pub value: F,
}

#[derive(Debug)]
pub struct AlphaKernelError;

pub trait AlphaKernelExecutor<F> {}

pub fn execute_alpha<F, T, E>(
    _executor: &mut E,
    _transcript: &mut T,
) -> Result<AlphaExecutionArtifacts<F>, AlphaKernelError>
where
    E: AlphaKernelExecutor<F>,
{
    Ok(AlphaExecutionArtifacts {
        sumchecks: Vec::new(),
    })
}
"
        .trim_start()
        .to_owned(),
    }
}

fn non_jolt_alpha_verifier_source() -> RustSourceFile {
    RustSourceFile {
        filename: "verify_alpha.rs".to_owned(),
        source: r"
pub struct DefaultTranscript<F>(core::marker::PhantomData<F>);

pub trait Transcript {
    type Challenge;
}

pub type AlphaNamedEval<F> = super::shared::StageNamedEval<F>;
pub type AlphaSumcheckOutput<F> = super::shared::StageSumcheckOutput<F>;
pub type AlphaProof<F> = super::shared::StageProof<F>;

#[derive(Clone, Debug)]
pub struct AlphaExecutionArtifacts<F> {
    pub sumchecks: Vec<AlphaSumcheckOutput<F>>,
}

#[derive(Debug)]
pub enum VerifyAlphaError {}

pub fn verify_alpha<F, T>(
    _proof: &AlphaProof<F>,
    _transcript: &mut T,
) -> Result<AlphaExecutionArtifacts<F>, VerifyAlphaError> {
    Ok(AlphaExecutionArtifacts {
        sumchecks: Vec::new(),
    })
}
"
        .trim_start()
        .to_owned(),
    }
}

fn non_jolt_verifier_common_source() -> String {
    r"
#[derive(Clone, Debug)]
pub struct Commitment;

#[derive(Clone, Debug)]
pub struct StageNamedEval<F> {
    pub name: &'static str,
    pub oracle: &'static str,
    pub value: F,
}

#[derive(Clone, Debug)]
pub struct StageSumcheckOutput<F> {
    pub driver: &'static str,
    pub point: Vec<F>,
    pub evals: Vec<StageNamedEval<F>>,
    pub proof: (),
}

#[derive(Clone, Debug)]
pub struct StageProof<F> {
    pub sumchecks: Vec<StageSumcheckOutput<F>>,
}
"
    .trim_start()
    .to_owned()
}

fn jolt_protocol_chain_commitment_stage1_fixture(
    context: &MeliorContext,
    params: &JoltProtocolParams,
) -> String {
    let (commitment_prover_cpu, commitment_verifier_cpu) =
        build_commitment_pipeline_cpu(context, params);
    let (stage1_prover_cpu, stage1_verifier_cpu) = build_stage1_pipeline_cpu(context, params);
    let commitment_prover =
        commitment_cpu_program(&commitment_prover_cpu).expect("extract commitment prover program");
    let commitment_verifier = commitment_cpu_program(&commitment_verifier_cpu)
        .expect("extract commitment verifier program");
    let stage1_prover = stage1_cpu_program(&stage1_prover_cpu).expect("extract stage1 prover");
    let stage1_verifier =
        stage1_cpu_program(&stage1_verifier_cpu).expect("extract stage1 verifier");
    let commitment_prover_source =
        emit_commitment_rust(&commitment_prover_cpu).expect("emit commitment prover");
    let commitment_verifier_source =
        emit_commitment_rust(&commitment_verifier_cpu).expect("emit commitment verifier");
    let stage1_prover_source = emit_stage1_rust(&stage1_prover_cpu).expect("emit stage1 prover");
    let stage1_verifier_source =
        emit_stage1_rust(&stage1_verifier_cpu).expect("emit stage1 verifier");

    let mut text = String::new();
    writeln!(&mut text, "# Jolt protocol chain fixture").unwrap();
    writeln!(&mut text, "params:").unwrap();
    writeln!(&mut text, "  log_t: {}", params.log_t).unwrap();
    writeln!(&mut text, "  log_k_bytecode: {}", params.log_k_bytecode).unwrap();
    writeln!(&mut text, "  log_k_ram: {}", params.log_k_ram).unwrap();
    writeln!(&mut text, "  trace_length: {}", params.trace_length).unwrap();
    writeln!(&mut text, "phases:").unwrap();
    writeln!(&mut text, "  - name: commitment").unwrap();
    writeln!(
        &mut text,
        "    protocol_fixture: tests/fixtures/commitment_protocol.mlir"
    )
    .unwrap();
    writeln!(
        &mut text,
        "    concrete_fixture: tests/fixtures/commitment_concrete.mlir"
    )
    .unwrap();
    writeln!(
        &mut text,
        "    prover_cpu_fixture: tests/fixtures/commitment_prover_cpu.mlir"
    )
    .unwrap();
    writeln!(
        &mut text,
        "    verifier_cpu_fixture: tests/fixtures/commitment_verifier_cpu.mlir"
    )
    .unwrap();
    writeln!(
        &mut text,
        "    prover_rust_fixture: tests/fixtures/{}",
        commitment_prover_source.filename
    )
    .unwrap();
    writeln!(
        &mut text,
        "    verifier_rust_fixture: tests/fixtures/{}",
        commitment_verifier_source.filename
    )
    .unwrap();
    writeln!(
        &mut text,
        "    prover_batches: {}",
        commitment_prover.batch_plans.len()
    )
    .unwrap();
    writeln!(
        &mut text,
        "    verifier_batches: {}",
        commitment_verifier.batch_plans.len()
    )
    .unwrap();
    writeln!(
        &mut text,
        "    optional_commitments: {}",
        commitment_prover.optional_plans.len()
    )
    .unwrap();
    writeln!(
        &mut text,
        "    transcript_steps: {}",
        commitment_prover.transcript_steps.len()
    )
    .unwrap();
    writeln!(&mut text, "  - name: stage1_outer").unwrap();
    writeln!(&mut text, "    consumes_transcript_from: commitment").unwrap();
    writeln!(
        &mut text,
        "    protocol_fixture: tests/fixtures/stage1_outer_protocol.mlir"
    )
    .unwrap();
    writeln!(
        &mut text,
        "    prover_compute_fixture: tests/fixtures/stage1_outer_prover_compute.mlir"
    )
    .unwrap();
    writeln!(
        &mut text,
        "    verifier_compute_fixture: tests/fixtures/stage1_outer_verifier_compute.mlir"
    )
    .unwrap();
    writeln!(
        &mut text,
        "    prover_kernel_compute_fixture: tests/fixtures/stage1_outer_prover_kernel_compute.mlir"
    )
    .unwrap();
    writeln!(
        &mut text,
        "    verifier_kernel_compute_fixture: tests/fixtures/stage1_outer_verifier_kernel_compute.mlir"
    )
    .unwrap();
    writeln!(
        &mut text,
        "    prover_cpu_fixture: tests/fixtures/stage1_outer_prover_cpu.mlir"
    )
    .unwrap();
    writeln!(
        &mut text,
        "    verifier_cpu_fixture: tests/fixtures/stage1_outer_verifier_cpu.mlir"
    )
    .unwrap();
    writeln!(
        &mut text,
        "    prover_rust_fixture: tests/fixtures/{}",
        stage1_prover_source.filename
    )
    .unwrap();
    writeln!(
        &mut text,
        "    verifier_rust_fixture: tests/fixtures/{}",
        stage1_verifier_source.filename
    )
    .unwrap();
    writeln!(
        &mut text,
        "    transcript_squeezes: {}",
        stage1_prover.transcript_squeezes.len()
    )
    .unwrap();
    writeln!(
        &mut text,
        "    prover_sumcheck_drivers: {}",
        stage1_prover.drivers.len()
    )
    .unwrap();
    writeln!(
        &mut text,
        "    verifier_sumcheck_drivers: {}",
        stage1_verifier.drivers.len()
    )
    .unwrap();
    writeln!(
        &mut text,
        "    opening_claims: {}",
        stage1_prover.opening_claims.len()
    )
    .unwrap();
    writeln!(
        &mut text,
        "    opening_batches: {}",
        stage1_prover.opening_batches.len()
    )
    .unwrap();
    writeln!(&mut text, "    drivers:").unwrap();
    for driver in &stage1_prover.drivers {
        writeln!(
            &mut text,
            "      - {}: kernel={} rounds={} degree={} proof_slot={}",
            driver.symbol,
            driver.kernel.as_deref().unwrap_or("<none>"),
            driver.num_rounds,
            driver.degree,
            driver.proof_slot
        )
        .unwrap();
    }
    writeln!(&mut text, "parity_gates:").unwrap();
    writeln!(
        &mut text,
        "  - pipeline_generated_commitment_prover_verifier_self_parity_runs"
    )
    .unwrap();
    writeln!(
        &mut text,
        "  - generated_stage1_real_executor_self_verifies_synthetic_remaining"
    )
    .unwrap();
    writeln!(
        &mut text,
        "  - generated_jolt_chain_commitment_then_stage1_self_parity_runs"
    )
    .unwrap();
    text
}

fn opening_claim_equal_protocol(left_oracle: &str, right_oracle: &str, mode: &str) -> String {
    let right_oracle_def = if left_oracle == right_oracle {
        String::new()
    } else {
        format!(
            r#"  "piop.oracle"() {{commit_domain = @trace, domain = @trace, field = @bn254_fr, layout = "virtual", sym_name = "{right_oracle}", visibility = "virtual"}} : () -> ()
"#
        )
    };
    format!(
        r#"
module @opening.claim.equal attributes {{bolt.phase = "protocol"}} {{
  "field.define"() {{modulus_bits = 254 : i64, role = "scalar", sym_name = "bn254_fr"}} : () -> ()
  "hash.function"() {{algorithm = "blake2b", sym_name = "blake2b"}} : () -> ()
  "transcript.scheme"() {{hash = @blake2b, sym_name = "blake2b_transcript"}} : () -> ()
  "pcs.scheme"() {{field = @bn254_fr, sym_name = "dory"}} : () -> ()
  "poly.domain"() {{field = @bn254_fr, log_size = 16 : i64, sym_name = "trace"}} : () -> ()
  "protocol.params"() {{field = @bn254_fr, pcs = @dory, sym_name = "params", transcript = @blake2b_transcript}} : () -> ()
  "protocol.boundary"() {{roles = ["prover", "verifier"], sym_name = "opening.claim.equal"}} : () -> ()
  "piop.oracle"() {{commit_domain = @trace, domain = @trace, field = @bn254_fr, layout = "virtual", sym_name = "{left_oracle}", visibility = "virtual"}} : () -> ()
{right_oracle_def}
  %left:3 = "piop.opening_input"() {{claim_kind = "virtual", domain = @trace, oracle = @{left_oracle}, point_arity = 16 : i64, source_claim = @stage2.product_virtual.remainder.opening.{left_oracle}, source_stage = @stage2, sym_name = "stage3.input.stage2_left.{left_oracle}"}} : () -> (!poly.point, !field.scalar, !piop.opening_claim_type)
  %right:3 = "piop.opening_input"() {{claim_kind = "virtual", domain = @trace, oracle = @{right_oracle}, point_arity = 16 : i64, source_claim = @stage2.instruction_lookup.claim_reduction.opening.{right_oracle}, source_stage = @stage2, sym_name = "stage3.input.stage2_right.{right_oracle}"}} : () -> (!poly.point, !field.scalar, !piop.opening_claim_type)
  "piop.opening_claim_equal"(%left#2, %right#2) {{mode = "{mode}", sym_name = "stage3.instruction_input.left_claim_consistency"}} : (!piop.opening_claim_type, !piop.opening_claim_type) -> ()
}}
"#
    )
}

fn transcript_absorb_bytes_protocol(params: &JoltProtocolParams) -> String {
    format!(
        r#"
module @transcript.absorb.bytes attributes {{bolt.phase = "protocol"}} {{
  "field.define"() {{modulus_bits = 254 : i64, role = "scalar", sym_name = "bn254_fr"}} : () -> ()
  "hash.function"() {{algorithm = "blake2b", sym_name = "blake2b"}} : () -> ()
  "transcript.scheme"() {{hash = @blake2b, sym_name = "blake2b_transcript"}} : () -> ()
  "pcs.scheme"() {{field = @bn254_fr, sym_name = "dory"}} : () -> ()
  "protocol.params"() {{{params_attrs}, sym_name = "jolt.params"}} : () -> ()
  "protocol.boundary"() {{roles = ["prover", "verifier"], sym_name = "transcript.absorb.bytes"}} : () -> ()
  %0 = "transcript.state"() {{scheme = @blake2b_transcript, sym_name = "fs_after_stage3"}} : () -> !transcript.state_type
  %1 = "transcript.absorb_bytes"(%0) {{label = "ram_val_check_gamma", payload = "", sym_name = "stage4.ram_val_check.domain_separator"}} : (!transcript.state_type) -> !transcript.state_type
  %2:2 = "transcript.squeeze"(%1) {{count = 1 : i64, kind = "challenge_scalar", label = "ram_val_check_gamma", sym_name = "stage4.ram_val_check.gamma"}} : (!transcript.state_type) -> (!transcript.state_type, !field.scalar)
}}
"#,
        params_attrs = jolt_params_attrs_source(params)
    )
}

fn jolt_params_attrs_source(params: &JoltProtocolParams) -> String {
    params
        .attrs()
        .into_iter()
        .map(|(name, value)| format!("{name} = {value}"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn explicit_sumcheck_protocol() -> &'static str {
    r#"
module @explicit.sumcheck attributes {bolt.phase = "protocol"} {
  "field.define"() {modulus_bits = 254 : i64, role = "scalar", sym_name = "bn254_fr"} : () -> ()
  "hash.function"() {algorithm = "blake2b", sym_name = "blake2b"} : () -> ()
  "transcript.scheme"() {hash = @blake2b, sym_name = "blake2b_transcript"} : () -> ()
  "pcs.scheme"() {field = @bn254_fr, sym_name = "dory"} : () -> ()
  "poly.domain"() {field = @bn254_fr, log_size = 16 : i64, sym_name = "trace"} : () -> ()
  "piop.relation"() {degree = 3 : i64, domain = @trace, kind = "sumcheck", num_rounds = 4 : i64, output_count = 1 : i64, sym_name = "jolt.stage1.outer.remaining"} : () -> ()
  %0 = "transcript.state"() {scheme = @blake2b_transcript, sym_name = "fs0"} : () -> !transcript.state_type
  %1, %alpha = "transcript.squeeze"(%0) {count = 1 : i64, kind = "scalar", label = "sumcheck_claim", sym_name = "stage1.alpha"} : (!transcript.state_type) -> (!transcript.state_type, !field.scalar)
  %stage = "piop.stage"() {name = "stage1", order = 1 : i64, roles = ["prover", "verifier"], sym_name = "stage1"} : () -> !piop.stage_type
  %claim_value = "field.const"() {field = @bn254_fr, value = 0 : i64, sym_name = "stage1.outer.claim_value"} : () -> !field.scalar
  %claim = "piop.sumcheck_claim"(%claim_value) {claim = @stage1.outer.claim, degree = 3 : i64, domain = @trace, num_rounds = 4 : i64, relation = @jolt.stage1.outer.remaining, stage = @stage1, sym_name = "stage1.outer.claim"} : (!field.scalar) -> !piop.sumcheck_claim_type
  %batch = "piop.sumcheck_batch"(%stage, %claim) {claim_label = "sumcheck_claim", count = 1 : i64, ordered_claims = [@stage1.outer.claim], policy = "jolt_core_front_loaded", proof_slot = @stage1.sumcheck, round_label = "sumcheck_poly", round_schedule = [2, 1, 1], stage = @stage1, sym_name = "stage1.outer.batch"} : (!piop.stage_type, !piop.sumcheck_claim_type) -> !piop.sumcheck_batch_type
  %2, %point, %result, %proof = "piop.sumcheck"(%1, %batch) {claim_label = "sumcheck_claim", degree = 3 : i64, num_rounds = 4 : i64, policy = "jolt_core_front_loaded", proof_slot = @stage1.sumcheck, relation = @jolt.stage1.outer.remaining, round_label = "sumcheck_poly", round_schedule = [2, 1, 1], stage = @stage1, sym_name = "stage1.outer.sumcheck"} : (!transcript.state_type, !piop.sumcheck_batch_type) -> (!transcript.state_type, !poly.point, !piop.sumcheck_result_type, !piop.sumcheck_proof_type)
  %eval = "piop.sumcheck_eval"(%result) {index = 0 : i64, name = @stage1.outer.eval, oracle = @RdInc, source = @stage1.outer.sumcheck, sym_name = "stage1.outer.eval"} : (!piop.sumcheck_result_type) -> !field.scalar
  %opening = "pcs.opening_claim"(%point, %eval) {domain = @trace, family = @jolt.main_witness_polys, oracle = @RdInc, point_arity = 4 : i64, sym_name = "stage1.outer.opening"} : (!poly.point, !field.scalar) -> !pcs.opening_claim_type
  %openings = "pcs.opening_batch"(%opening) {count = 1 : i64, ordered_claims = [@stage1.outer.opening], policy = "jolt_core_order", proof_slot = @stage1.openings, sym_name = "stage1.opening_batch"} : (!pcs.opening_claim_type) -> !pcs.opening_batch_type
  %3, %opening_proof = "pcs.batch_open"(%2, %openings) {pcs = @dory, proof_slot = @stage1.openings, sym_name = "stage1.open", transcript_label = "opening_proof"} : (!transcript.state_type, !pcs.opening_batch_type) -> (!transcript.state_type, !pcs.opening_proof_type)
}
"#
}

fn explicit_sumcheck_protocol_with_eval_family(count: usize, evals: &str) -> String {
    explicit_sumcheck_protocol().replace(
        r#"  %opening = "pcs.opening_claim""#,
        &format!(
            r#"  "piop.sumcheck_eval_family"() {{count = {count} : i64, evals = {evals}, oracle_family = @RdInc, source = @stage1.outer.sumcheck, sym_name = "stage1.outer.eval.family"}} : () -> ()
  %opening = "pcs.opening_claim""#
        ),
    )
}

fn explicit_sumcheck_compute() -> &'static str {
    r#"
module @explicit.sumcheck attributes {bolt.phase = "compute", bolt.role = "prover"} {
  "compute.params"() {field = @bn254_fr, pcs = @dory, sym_name = "params", transcript = @blake2b_transcript} : () -> ()
  "compute.function"() {source = @explicit.sumcheck, sym_name = "explicit.sumcheck"} : () -> ()
  "compute.relation"() {degree = 3 : i64, domain = @trace, kind = "sumcheck", num_rounds = 4 : i64, output_count = 1 : i64, sym_name = "jolt.stage1.outer.remaining"} : () -> ()
  %0 = "compute.transcript_init"() {scheme = @blake2b_transcript, sym_name = "fs0"} : () -> !compute.transcript_state
  %1, %alpha = "compute.transcript_squeeze"(%0) {count = 1 : i64, kind = "scalar", label = "sumcheck_claim", sym_name = "stage1.alpha"} : (!compute.transcript_state) -> (!compute.transcript_state, !compute.field_value)
  %claim_value = "compute.field_const"() {field = @bn254_fr, value = 0 : i64, sym_name = "stage1.outer.claim_value"} : () -> !compute.field_value
  %claim = "compute.sumcheck_claim"(%claim_value) {claim = @stage1.outer.claim, degree = 3 : i64, domain = @trace, num_rounds = 4 : i64, relation = @jolt.stage1.outer.remaining, stage = @stage1, sym_name = "stage1.outer.claim"} : (!compute.field_value) -> !compute.sumcheck_claim_type
  %batch = "compute.sumcheck_batch"(%claim) {claim_label = "sumcheck_claim", count = 1 : i64, ordered_claims = [@stage1.outer.claim], policy = "jolt_core_front_loaded", proof_slot = @stage1.sumcheck, round_label = "sumcheck_poly", round_schedule = [2, 1, 1], stage = @stage1, sym_name = "stage1.outer.batch"} : (!compute.sumcheck_claim_type) -> !compute.sumcheck_batch_type
  %2, %point, %result, %proof = "compute.sumcheck_driver"(%1, %batch) {claim_label = "sumcheck_claim", degree = 3 : i64, num_rounds = 4 : i64, policy = "jolt_core_front_loaded", proof_slot = @stage1.sumcheck, relation = @jolt.stage1.outer.remaining, round_label = "sumcheck_poly", round_schedule = [2, 1, 1], stage = @stage1, sym_name = "stage1.outer.sumcheck"} : (!compute.transcript_state, !compute.sumcheck_batch_type) -> (!compute.transcript_state, !compute.point, !compute.sumcheck_result_type, !compute.sumcheck_proof_type)
  %eval = "compute.sumcheck_eval"(%result) {index = 0 : i64, name = @stage1.outer.eval, oracle = @RdInc, source = @stage1.outer.sumcheck, sym_name = "stage1.outer.eval"} : (!compute.sumcheck_result_type) -> !compute.field_value
  %opening = "compute.pcs_opening_claim"(%point, %eval) {domain = @trace, family = @jolt.main_witness_polys, oracle = @RdInc, point_arity = 4 : i64, sym_name = "stage1.outer.opening"} : (!compute.point, !compute.field_value) -> !compute.opening_claim_type
  %openings = "compute.pcs_opening_batch"(%opening) {count = 1 : i64, ordered_claims = [@stage1.outer.opening], policy = "jolt_core_order", proof_slot = @stage1.openings, sym_name = "stage1.opening_batch"} : (!compute.opening_claim_type) -> !compute.opening_batch_type
  %3, %opening_proof = "compute.pcs_batch_open"(%2, %openings) {pcs = @dory, proof_slot = @stage1.openings, sym_name = "stage1.open", transcript_label = "opening_proof"} : (!compute.transcript_state, !compute.opening_batch_type) -> (!compute.transcript_state, !compute.opening_proof_type)
}
"#
}

fn assert_or_update_fixture(path: &str, actual: &str) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
    if std::env::var_os("JOLT_UPDATE_GOLDENS").is_some() {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create golden fixture directory");
        }
        std::fs::write(&path, actual).expect("write golden fixture");
        return;
    }
    if !path.exists() {
        return;
    }
    let expected = std::fs::read_to_string(&path).expect("read golden fixture");
    assert_eq!(expected, actual);
}

fn assert_rust_source_compiles(_filename: &str, source: &str) {
    if !generated_jolt_runtime_available() {
        return;
    }
    let dir = new_temp_dir("bolt_emit");
    let workspace_root = workspace_root();
    std::fs::write(
        dir.join("Cargo.toml"),
        generated_crate_manifest(&workspace_root),
    )
    .expect("write generated cargo manifest");
    std::fs::create_dir_all(dir.join("src")).expect("create generated src dir");
    if source.contains("super::jolt_relations") {
        // Tier B: audited Jolt verifier core. Tier A is provided by the
        // bolt-verifier-runtime crate and is not staged as generated source.
        std::fs::write(
            dir.join("src/jolt_relations.rs"),
            generated_verifier_jolt_relations_source(&workspace_root),
        )
        .expect("write generated jolt_relations source");
        std::fs::write(dir.join("src/generated.rs"), source).expect("write generated source");
        std::fs::write(
            dir.join("src/lib.rs"),
            "pub mod jolt_relations;\n#[rustfmt::skip]\npub mod generated;\n",
        )
        .expect("write generated lib wrapper");
    } else {
        std::fs::write(dir.join("src/lib.rs"), source).expect("write generated source");
    }
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    let output = Command::new(cargo)
        .arg("check")
        .arg("--manifest-path")
        .arg(dir.join("Cargo.toml"))
        .arg("-q")
        .env_remove("RUSTFLAGS")
        .env("CARGO_TARGET_DIR", dir.join("target"))
        .output()
        .expect("run cargo check");
    assert!(
        output.status.success(),
        "generated rust did not compile\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let _ = std::fs::remove_dir_all(dir);
}

fn assert_generated_role_crate_compiles(generated: &JoltGeneratedCrate) {
    let dir = new_temp_dir(&generated.crate_name);
    for file in &generated.files {
        let path = dir.join(&file.path);
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent).expect("create generated crate dir");
        }
        std::fs::write(path, &file.source).expect("write generated crate file");
    }
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    let output = Command::new(cargo)
        .arg("check")
        .arg("--manifest-path")
        .arg(dir.join("Cargo.toml"))
        .arg("-q")
        .env("CARGO_TARGET_DIR", dir.join("target"))
        .output()
        .expect("run generated role crate check");
    assert!(
        output.status.success(),
        "generated role crate `{}` did not compile\nstdout:\n{}\nstderr:\n{}",
        generated.crate_name,
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let _ = std::fs::remove_dir_all(dir);
}

fn assert_generated_crate_manifest_compiles(output_root: &Path, crate_name: &str) {
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    let output = Command::new(cargo)
        .arg("check")
        .arg("--manifest-path")
        .arg(output_root.join(crate_name).join("Cargo.toml"))
        .arg("-q")
        .env(
            "CARGO_TARGET_DIR",
            output_root.join("target").join(crate_name),
        )
        .output()
        .expect("run generated crate check");
    assert!(
        output.status.success(),
        "generated crate `{crate_name}` did not compile\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
}

fn redirect_generated_prover_to_generated_verifier(output_root: &Path, dependency_root: &Path) {
    let manifest_path = output_root.join("jolt-prover").join("Cargo.toml");
    let workspace_verifier = format!(
        "jolt-verifier = {{ path = \"{}/jolt-verifier\" }}",
        dependency_root.display()
    );
    let generated_verifier = format!(
        "jolt-verifier = {{ path = \"{}\" }}",
        output_root.join("jolt-verifier").display()
    );
    let manifest = std::fs::read_to_string(&manifest_path).expect("read generated prover manifest");
    let manifest = manifest.replace(&workspace_verifier, &generated_verifier);
    std::fs::write(&manifest_path, manifest).expect("rewrite generated prover manifest");
}

fn assert_checked_in_generated_role_crate_sources_match(generated_crates: &[JoltGeneratedCrate]) {
    let crates_root = workspace_root().join("crates");
    for generated in generated_crates {
        for file in &generated.files {
            let checked_in_path = crates_root.join(&generated.crate_name).join(&file.path);
            let checked_in =
                std::fs::read_to_string(&checked_in_path).expect("read checked-in generated file");
            assert_eq!(
                checked_in,
                file.source,
                "checked-in generated crate file `{}` is stale; regenerate with the Bolt artifact writer",
                checked_in_path.display()
            );
            if generated.crate_name == "jolt-verifier" {
                assert!(
                    !checked_in.contains("use jolt_prover")
                        && !checked_in.contains("jolt_prover::")
                        && !checked_in.contains("use jolt_kernels")
                        && !checked_in.contains("jolt_kernels::")
                        && !checked_in.contains("use jolt_core")
                        && !checked_in.contains("jolt_core::"),
                    "generated verifier file `{}` imports non-audit role/runtime code",
                    checked_in_path.display()
                );
            }
            if generated.crate_name == "jolt-prover" {
                assert!(
                    !checked_in.contains("jolt_verifier::stages"),
                    "generated prover file `{}` imports verifier stage internals instead of only verifier-owned proof types",
                    checked_in_path.display()
                );
            }
        }
    }
}

fn assert_generated_commitment_self_parity_runs(
    prover_source: &RustSourceFile,
    verifier_source: &RustSourceFile,
    main_source: &str,
) {
    if !generated_jolt_runtime_available() {
        return;
    }
    let dir = new_temp_dir("bolt_self_parity");
    let workspace_root = workspace_root();
    std::fs::write(
        dir.join("Cargo.toml"),
        generated_crate_manifest(&workspace_root),
    )
    .expect("write generated cargo manifest");
    let src_dir = dir.join("src");
    std::fs::create_dir_all(&src_dir).expect("create generated src dir");
    std::fs::write(src_dir.join(&prover_source.filename), &prover_source.source)
        .expect("write generated prover source");
    std::fs::write(
        src_dir.join(&verifier_source.filename),
        &verifier_source.source,
    )
    .expect("write generated verifier source");
    std::fs::write(src_dir.join("main.rs"), main_source)
        .expect("write generated self-parity harness");

    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    let output = Command::new(cargo)
        .arg("run")
        .arg("--manifest-path")
        .arg(dir.join("Cargo.toml"))
        .arg("-q")
        .env_remove("RUSTFLAGS")
        .env("CARGO_TARGET_DIR", dir.join("target"))
        .output()
        .expect("run generated self-parity crate");
    assert!(
        output.status.success(),
        "generated commitment self-parity failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let _ = std::fs::remove_dir_all(dir);
}

fn assert_generated_stage1_self_parity_runs(
    prover_source: &RustSourceFile,
    verifier_source: &RustSourceFile,
    main_source: &str,
) {
    if !generated_jolt_runtime_available() {
        return;
    }
    let dir = new_temp_dir("bolt_stage1_self_parity");
    let workspace_root = workspace_root();
    std::fs::write(
        dir.join("Cargo.toml"),
        generated_crate_manifest(&workspace_root),
    )
    .expect("write generated cargo manifest");
    let src_dir = dir.join("src");
    std::fs::create_dir_all(&src_dir).expect("create generated src dir");
    let main_source = if verifier_source.source.contains("super::jolt_relations") {
        write_verifier_jolt_relations_module(&src_dir, &workspace_root);
        format!("mod jolt_relations;\n{main_source}")
    } else {
        main_source.to_owned()
    };
    std::fs::write(src_dir.join(&prover_source.filename), &prover_source.source)
        .expect("write generated stage1 prover source");
    std::fs::write(
        src_dir.join(&verifier_source.filename),
        &verifier_source.source,
    )
    .expect("write generated stage1 verifier source");
    std::fs::write(src_dir.join("main.rs"), main_source)
        .expect("write generated stage1 self-parity harness");

    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    let output = Command::new(cargo)
        .arg("run")
        .arg("--manifest-path")
        .arg(dir.join("Cargo.toml"))
        .arg("-q")
        .env_remove("RUSTFLAGS")
        .env("CARGO_TARGET_DIR", dir.join("target"))
        .output()
        .expect("run generated stage1 self-parity crate");
    assert!(
        output.status.success(),
        "generated stage1 self-parity failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let _ = std::fs::remove_dir_all(dir);
}

fn assert_generated_jolt_chain_self_parity_runs(files: &[&RustSourceFile], main_source: &str) {
    if !generated_jolt_runtime_available() {
        return;
    }
    let dir = new_temp_dir("bolt_chain_self_parity");
    let workspace_root = workspace_root();
    std::fs::write(
        dir.join("Cargo.toml"),
        generated_crate_manifest(&workspace_root),
    )
    .expect("write generated cargo manifest");
    let src_dir = dir.join("src");
    std::fs::create_dir_all(&src_dir).expect("create generated src dir");
    let main_source = if files
        .iter()
        .any(|file| file.source.contains("super::jolt_relations"))
    {
        write_verifier_jolt_relations_module(&src_dir, &workspace_root);
        format!("mod jolt_relations;\n{main_source}")
    } else {
        main_source.to_owned()
    };
    for file in files {
        std::fs::write(src_dir.join(&file.filename), &file.source)
            .expect("write generated chain source");
    }
    std::fs::write(src_dir.join("main.rs"), main_source)
        .expect("write generated chain self-parity harness");

    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    let output = Command::new(cargo)
        .arg("run")
        .arg("--manifest-path")
        .arg(dir.join("Cargo.toml"))
        .arg("-q")
        .env_remove("RUSTFLAGS")
        .env("CARGO_TARGET_DIR", dir.join("target"))
        .output()
        .expect("run generated chain self-parity crate");
    assert!(
        output.status.success(),
        "generated commitment+stage1 self-parity failed\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let _ = std::fs::remove_dir_all(dir);
}

fn write_verifier_jolt_relations_module(src_dir: &Path, workspace_root: &Path) {
    std::fs::write(
        src_dir.join("jolt_relations.rs"),
        generated_verifier_jolt_relations_source(workspace_root),
    )
    .expect("write generated jolt_relations source");
}

fn generated_verifier_jolt_relations_source(workspace_root: &Path) -> String {
    let jolt_relations = std::fs::read_to_string(
        workspace_root.join("crates/jolt-verifier/src/stages/jolt_relations.rs"),
    )
    .expect("read generated verifier jolt_relations stage source");
    format!(
        "#![allow(dead_code, unused_imports, unused_macros, reason = \"audited Jolt verifier core helpers are shared across generated stage subsets\")]\n{jolt_relations}"
    )
}

fn workspace_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn generated_jolt_runtime_available() -> bool {
    let workspace_root = workspace_root();
    workspace_root
        .join("crates/jolt-kernels/Cargo.toml")
        .exists()
        && workspace_root
            .join("crates/bolt-verifier-runtime/Cargo.toml")
            .exists()
        && workspace_root
            .join("crates/jolt-verifier/src/stages/jolt_relations.rs")
            .exists()
}

fn checked_in_generated_role_crates_available() -> bool {
    let workspace_root = workspace_root();
    generated_jolt_runtime_available()
        && workspace_root
            .join("crates/jolt-prover/Cargo.toml")
            .exists()
        && workspace_root
            .join("crates/jolt-verifier/Cargo.toml")
            .exists()
}

fn generated_crate_manifest(workspace_root: &Path) -> String {
    format!(
        r#"[package]
name = "generated-commitment-phase-check"
version = "0.0.0"
edition = "2021"

[patch.crates-io]
ark-bn254 = {{ git = "https://github.com/a16z/arkworks-algebra", branch = "dev/twist-shout" }}
ark-ec = {{ git = "https://github.com/a16z/arkworks-algebra", branch = "dev/twist-shout" }}
ark-ff = {{ git = "https://github.com/a16z/arkworks-algebra", branch = "dev/twist-shout" }}
ark-serialize = {{ git = "https://github.com/a16z/arkworks-algebra", branch = "dev/twist-shout" }}

[dependencies]
bolt-verifier-runtime = {{ path = "{}" }}
jolt-dory = {{ path = "{}" }}
jolt-field = {{ path = "{}" }}
jolt-kernels = {{ path = "{}" }}
jolt-lookup-tables = {{ path = "{}" }}
jolt-openings = {{ path = "{}" }}
jolt-poly = {{ path = "{}" }}
jolt-r1cs = {{ path = "{}" }}
jolt-sumcheck = {{ path = "{}" }}
jolt-transcript = {{ path = "{}" }}
jolt-witness = {{ path = "{}" }}
rayon = "1.12.0"
serde = {{ version = "1.0", default-features = false, features = ["derive"] }}
tracing = {{ version = "0.1.37", default-features = false, features = ["attributes"] }}
"#,
        workspace_root
            .join("crates/bolt-verifier-runtime")
            .display(),
        workspace_root.join("crates/jolt-dory").display(),
        workspace_root.join("crates/jolt-field").display(),
        workspace_root.join("crates/jolt-kernels").display(),
        workspace_root.join("crates/jolt-lookup-tables").display(),
        workspace_root.join("crates/jolt-openings").display(),
        workspace_root.join("crates/jolt-poly").display(),
        workspace_root.join("crates/jolt-r1cs").display(),
        workspace_root.join("crates/jolt-sumcheck").display(),
        workspace_root.join("crates/jolt-transcript").display(),
        workspace_root.join("crates/jolt-witness").display(),
    )
}

fn new_temp_dir(prefix: &str) -> std::path::PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after unix epoch")
        .as_nanos();
    let dir = std::env::temp_dir().join(format!("{}_{}_{}", prefix, std::process::id(), nonce));
    std::fs::create_dir_all(&dir).expect("create generated crate temp dir");
    dir
}

fn generated_small_self_parity_main() -> String {
    let mut source = r#"mod prove_commitment_phase;
mod verify_commitment_phase;

use std::borrow::Cow;

use jolt_dory::DoryScheme;
use jolt_field::Fr;
use jolt_transcript::{Blake2bTranscript, Transcript};

struct Inputs;

impl prove_commitment_phase::CommitmentInputProvider for Inputs {
    fn materialize(&mut self, oracle: &'static str) -> Option<Cow<'_, [Fr]>> {
        match oracle {
            "A" => Some(Cow::Owned(vec![Fr::from_u64(1), Fr::from_u64(2)])),
            "B" => Some(Cow::Owned(vec![
                Fr::from_u64(3),
                Fr::from_u64(4),
                Fr::from_u64(5),
                Fr::from_u64(6),
            ])),
            "Advice" => Some(Cow::Owned(vec![Fr::from_u64(0), Fr::from_u64(0)])),
            _ => None,
        }
    }
}

"#
    .to_owned();
    source.push_str(tracing_transcript_support());
    source.push_str(
        r#"
fn main() {
    let prover_setup =
        DoryScheme::setup_prover(prove_commitment_phase::COMMITMENT_BATCH_PLANS[0].num_vars);
    let mut inputs = Inputs;
    let mut prover_transcript = TracingTranscript::new(b"self");
    let prover = prove_commitment_phase::prove_commitment_phase(
        &mut inputs,
        &prover_setup,
        &mut prover_transcript,
    )
    .expect("prover commitment phase");

    assert_eq!(prover.commitments.len(), 3);
    assert!(prover.commitments[2].is_none());

    let mut verifier_transcript = TracingTranscript::new(b"self");
    let verifier = verify_commitment_phase::verify_commitment_phase(
        &prover.commitments,
        &mut verifier_transcript,
    )
    .expect("verifier commitment phase");

    assert_eq!(prover.commitments, verifier.commitments);
    assert_eq!(prover.records.len(), verifier.records.len());
    assert_transcript_step_parity(&prover_transcript, &verifier_transcript);
}
"#,
    );
    source
}

fn generated_pipeline_self_parity_main() -> String {
    let mut source = "mod prove_commitment_phase;
mod verify_commitment_phase;

use jolt_dory::DoryScheme;
use jolt_field::Fr;
use jolt_transcript::{Blake2bTranscript, Transcript};

"
    .to_owned();
    source.push_str(tracing_transcript_support());
    source.push_str(
        r#"
fn main() {
    let prover_setup =
        DoryScheme::setup_prover(prove_commitment_phase::COMMITMENT_BATCH_PLANS[0].num_vars);
    let inputs = prove_commitment_phase::CommitmentOracleInputs {
        rd_inc: &[1],
        ram_inc: &[2],
        instruction_keys: &[Some(0x1234_5678_9abc_def0_0123_4567_89ab_cdefu128)],
        ram_addresses: &[],
        bytecode_indices: &[],
        untrusted_advice: None,
        trusted_advice: None,
    };
    let mut oracles = prove_commitment_phase::build_commitment_oracles(&inputs)
        .expect("build commitment oracles");
    let mut prover_transcript = TracingTranscript::new(b"pipeline");
    let prover = prove_commitment_phase::prove_commitment_phase(
        &mut oracles,
        &prover_setup,
        &mut prover_transcript,
    )
    .expect("prover commitment phase");

    let expected_slots = prove_commitment_phase::COMMITMENT_BATCH_PLANS
        .iter()
        .map(|plan| plan.oracles.len())
        .sum::<usize>()
        + prove_commitment_phase::OPTIONAL_COMMITMENT_PLANS.len();
    assert_eq!(prover.commitments.len(), expected_slots);

    let mut verifier_transcript = TracingTranscript::new(b"pipeline");
    let verifier = verify_commitment_phase::verify_commitment_phase(
        &prover.commitments,
        &mut verifier_transcript,
    )
    .expect("verifier commitment phase");

    assert_eq!(prover.commitments, verifier.commitments);
    assert_eq!(prover.records.len(), verifier.records.len());
    for (prover_record, verifier_record) in prover.records.iter().zip(&verifier.records) {
        assert_eq!(prover_record.artifact, verifier_record.artifact);
        assert_eq!(prover_record.oracle, verifier_record.oracle);
        assert_eq!(prover_record.label, verifier_record.label);
        assert_eq!(prover_record.num_vars, verifier_record.num_vars);
    }
    assert_transcript_step_parity(&prover_transcript, &verifier_transcript);
}
"#,
    );
    source
}

fn generated_stage1_shape_self_parity_main() -> String {
    let mut source = r"mod prove_stage1_outer;
mod verify_stage1_outer;

use jolt_field::Fr;
use jolt_kernels::stage1::Stage1ShapeKernelExecutor;
use jolt_transcript::{Blake2bTranscript, Transcript};

"
    .to_owned();
    source.push_str(tracing_transcript_support());
    source.push_str(&stage1_verifier_proof_adapter(true));
    source.push_str(
        r#"
fn main() {
    let mut prover_executor = Stage1ShapeKernelExecutor;
    let mut prover_transcript = TracingTranscript::new(b"stage1");
    let prover = prove_stage1_outer::prove_stage1_outer(
        &mut prover_executor,
        &mut prover_transcript,
    )
    .expect("generated prover runs shape kernels");

    let proof = verifier_proof_from_prover_artifacts(&prover);
    let mut verifier_transcript = TracingTranscript::new(b"stage1");
    let verifier = verify_stage1_outer::verify_stage1_outer(
        &proof,
        &mut verifier_transcript,
    )
    .expect("generated verifier accepts shape proof");

    assert_eq!(
        prover.sumchecks.len(),
        prove_stage1_outer::STAGE1_SUMCHECK_DRIVERS.len()
    );
    assert_eq!(prover.sumchecks.len(), verifier.sumchecks.len());
    assert_eq!(prover.opening_batches.len(), verifier.opening_batches.len());
    for (prover_batch, verifier_batch) in prover.opening_batches.iter().zip(&verifier.opening_batches) {
        assert_eq!(prover_batch.symbol, verifier_batch.symbol);
        assert_eq!(prover_batch.count, verifier_batch.count);
    }
    for (prover_sumcheck, verifier_sumcheck) in prover.sumchecks.iter().zip(&verifier.sumchecks) {
        assert_eq!(prover_sumcheck.driver, verifier_sumcheck.driver);
        assert_eq!(prover_sumcheck.evals.len(), verifier_sumcheck.evals.len());
        for (prover_eval, verifier_eval) in prover_sumcheck.evals.iter().zip(&verifier_sumcheck.evals) {
            assert_eq!(prover_eval.name, verifier_eval.name);
            assert_eq!(prover_eval.oracle, verifier_eval.oracle);
            assert_eq!(prover_eval.value, verifier_eval.value);
        }
        assert_eq!(
            prover_sumcheck.proof.round_polynomials.len(),
            verifier_sumcheck.proof.round_polynomials.len()
        );
        for (prover_round, verifier_round) in prover_sumcheck
            .proof
            .round_polynomials
            .iter()
            .zip(&verifier_sumcheck.proof.round_polynomials)
        {
            assert_eq!(prover_round.coefficients(), verifier_round.coefficients());
        }
    }
    assert_ne!(prover_transcript.state(), verifier_transcript.state());
}
"#,
    );
    source
}

fn stage1_verifier_proof_adapter(clear_points: bool) -> String {
    let point_expr = if clear_points {
        "Vec::new()"
    } else {
        "sumcheck.point.clone()"
    };
    r"
fn verifier_proof_from_prover_artifacts(
    artifacts: &jolt_kernels::stage1::Stage1ExecutionArtifacts<Fr>,
) -> verify_stage1_outer::Stage1Proof<Fr> {
    verify_stage1_outer::Stage1Proof {
        sumchecks: artifacts
            .sumchecks
            .iter()
            .map(|sumcheck| verify_stage1_outer::Stage1SumcheckOutput {
                driver: sumcheck.driver,
                point: $POINT_EXPR,
                evals: sumcheck
                    .evals
                    .iter()
                    .map(|eval| verify_stage1_outer::Stage1NamedEval {
                        name: eval.name,
                        oracle: eval.oracle,
                        value: eval.value,
                    })
                    .collect(),
                proof: sumcheck.proof.clone(),
            })
            .collect(),
    }
}

"
    .replace("$POINT_EXPR", point_expr)
}

fn generated_stage1_real_dispatch_main() -> &'static str {
    r#"mod prove_stage1_outer;
mod verify_stage1_outer;

use jolt_field::Fr;
use jolt_kernels::stage1::{
    Stage1KernelError, Stage1ProverInputs, Stage1ProverKernelExecutor,
};
use jolt_sumcheck::SumcheckError;
use jolt_transcript::{Blake2bTranscript, Transcript};

fn main() {
    let inputs = Stage1ProverInputs::<Fr>::empty(2);
    let mut prover_executor = Stage1ProverKernelExecutor::new(inputs);
    let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"stage1");
    let prover_error = prove_stage1_outer::prove_stage1_outer(
        &mut prover_executor,
        &mut prover_transcript,
    )
    .expect_err("real prover requires uniskip extended evaluations");
    assert_eq!(
        prover_error,
        Stage1KernelError::MissingKernelInput {
            kernel: "jolt_stage1_outer_uniskip",
            input: "uniskip_extended_evals",
        }
    );

    let proof = verify_stage1_outer::Stage1Proof {
        sumchecks: vec![
            verify_stage1_outer::Stage1SumcheckOutput {
                driver: "stage1.uniskip.sumcheck",
                point: Vec::new(),
                evals: Vec::new(),
                proof: Default::default(),
            },
            verify_stage1_outer::Stage1SumcheckOutput {
                driver: "stage1.outer_remaining.sumcheck",
                point: Vec::new(),
                evals: Vec::new(),
                proof: Default::default(),
            },
        ],
    };
    let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage1");
    let verifier_error = verify_stage1_outer::verify_stage1_outer(
        &proof,
        &mut verifier_transcript,
    )
    .expect_err("real verifier rejects empty uniskip proof");
    assert!(matches!(
        verifier_error,
        verify_stage1_outer::VerifyStage1Error::Sumcheck {
            driver: "stage1.uniskip.sumcheck",
            error: SumcheckError::WrongNumberOfRounds { expected: 1, got: 0 },
        }
    ));
}
"#
}

fn generated_stage1_synthetic_remaining_main() -> String {
    let mut source = r"mod prove_stage1_outer;
mod verify_stage1_outer;

use jolt_field::Fr;
use jolt_kernels::stage1::{
    Stage1OuterRemainingContext, Stage1OuterRemainingEvaluator, Stage1ProverInputs,
    Stage1ProverKernelExecutor,
};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::SumcheckError;
use jolt_transcript::{Blake2bTranscript, Transcript};

struct SumZeroRemainingEvaluator;

impl Stage1OuterRemainingEvaluator<Fr> for SumZeroRemainingEvaluator {
    fn evaluate(&self, _context: Stage1OuterRemainingContext<'_, Fr>, point: &[Fr]) -> Fr {
        point[0] + point[0] - Fr::from_u64(1)
    }

    fn evaluate_virtual_oracle(
        &self,
        _context: Stage1OuterRemainingContext<'_, Fr>,
        _oracle: &str,
        point: &[Fr],
    ) -> Option<Fr> {
        Some(point.iter().copied().sum())
    }
}

"
    .to_owned();
    source.push_str(&stage1_verifier_proof_adapter(false));
    source.push_str(
        r#"
fn main() {
    let extended_evals = vec![Fr::from_u64(0); 9];
    let evaluator = SumZeroRemainingEvaluator;
    let inputs = Stage1ProverInputs::<Fr>::empty(2)
        .with_uniskip_extended_evals(&extended_evals)
        .with_outer_remaining_evaluator(&evaluator);
    let mut prover_executor = Stage1ProverKernelExecutor::new(inputs);
    let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"stage1");
    let prover_artifacts = prove_stage1_outer::prove_stage1_outer(
        &mut prover_executor,
        &mut prover_transcript,
    )
    .expect("generated real stage1 prover succeeds");

    let proof = verifier_proof_from_prover_artifacts(&prover_artifacts);
    let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage1");
    let verifier_artifacts = verify_stage1_outer::verify_stage1_outer(
        &proof,
        &mut verifier_transcript,
    )
    .expect("generated real stage1 verifier accepts prover proof");

    assert_eq!(prover_transcript.state(), verifier_transcript.state());
    assert_eq!(prover_artifacts.sumchecks.len(), 2);
    assert_eq!(verifier_artifacts.sumchecks.len(), 2);
    assert_eq!(
        prover_artifacts.sumchecks[1].point,
        verifier_artifacts.sumchecks[1].point
    );

    let mut extra_proof = proof.clone();
    extra_proof.sumchecks.push(proof.sumchecks[0].clone());
    let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage1");
    assert!(matches!(
        verify_stage1_outer::verify_stage1_outer(&extra_proof, &mut verifier_transcript),
        Err(verify_stage1_outer::VerifyStage1Error::UnexpectedProofCount {
            expected: 2,
            got: 3,
        })
    ));

    let mut wrong_driver = proof.clone();
    wrong_driver.sumchecks.swap(0, 1);
    let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage1");
    assert!(matches!(
        verify_stage1_outer::verify_stage1_outer(&wrong_driver, &mut verifier_transcript),
        Err(verify_stage1_outer::VerifyStage1Error::InvalidProof {
            driver: "stage1.uniskip.sumcheck",
            reason: "driver symbol mismatch",
        })
    ));

    let mut wrong_round = proof.clone();
    let mut coefficients = wrong_round.sumchecks[0].proof.round_polynomials[0]
        .coefficients()
        .to_vec();
    coefficients[0] += Fr::from_u64(1);
    wrong_round.sumchecks[0].proof.round_polynomials[0] = UnivariatePoly::new(coefficients);
    let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage1");
    assert!(matches!(
        verify_stage1_outer::verify_stage1_outer(&wrong_round, &mut verifier_transcript),
        Err(verify_stage1_outer::VerifyStage1Error::Sumcheck {
            driver: "stage1.uniskip.sumcheck",
            error: SumcheckError::RoundCheckFailed { .. },
        })
    ));

    let mut wrong_uniskip_eval = proof.clone();
    wrong_uniskip_eval.sumchecks[0].evals[0].value += Fr::from_u64(1);
    let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage1");
    assert!(matches!(
        verify_stage1_outer::verify_stage1_outer(&wrong_uniskip_eval, &mut verifier_transcript),
        Err(verify_stage1_outer::VerifyStage1Error::InvalidProof {
            driver: "stage1.uniskip.sumcheck",
            reason: "eval value mismatch",
        })
    ));

    let mut wrong_remaining_eval = proof.clone();
    wrong_remaining_eval.sumchecks[1].evals.swap(0, 1);
    let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage1");
    assert!(matches!(
        verify_stage1_outer::verify_stage1_outer(&wrong_remaining_eval, &mut verifier_transcript),
        Err(verify_stage1_outer::VerifyStage1Error::InvalidProof {
            driver: "stage1.outer_remaining.sumcheck",
            reason: "eval name mismatch",
        })
    ));

    let mut wrong_point = proof.clone();
    wrong_point.sumchecks[1].point[0] += Fr::from_u64(1);
    let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage1");
    assert!(matches!(
        verify_stage1_outer::verify_stage1_outer(&wrong_point, &mut verifier_transcript),
        Err(verify_stage1_outer::VerifyStage1Error::InvalidProof {
            driver: "stage1.outer_remaining.sumcheck",
            reason: "outer remaining point mismatch",
        })
    ));
}
"#,
    );
    source
}

fn generated_stage1_r1cs_data_main() -> String {
    let mut source = r"mod prove_stage1_outer;
mod verify_stage1_outer;

use jolt_field::Fr;
use jolt_kernels::stage1::{
    Stage1OuterR1csData, Stage1ProverInputs, Stage1ProverKernelExecutor,
};
use jolt_r1cs::{constraints::rv64, R1csKey};
use jolt_transcript::{Blake2bTranscript, Transcript};

"
    .to_owned();
    source.push_str(&stage1_verifier_proof_adapter(false));
    source.push_str(
        r#"
fn main() {
    let key = R1csKey::new(rv64::rv64_constraints::<Fr>(), 4);
    let mut witness = vec![Fr::from_u64(0); key.num_cycles * key.num_vars_padded];
    for cycle in 0..key.num_cycles {
        let base = cycle * key.num_vars_padded;
        witness[base + rv64::V_CONST] = Fr::from_u64(1);
        witness[base + rv64::V_FLAG_DO_NOT_UPDATE_UNEXPANDED_PC] = Fr::from_u64(1);
        key.matrices
            .check_witness(&witness[base..base + rv64::NUM_VARS_PER_CYCLE])
            .expect("noop cycle satisfies RV64 constraints");
    }
    let data = Stage1OuterR1csData::new(&key, &witness).expect("valid R1CS witness shape");
    let inputs = Stage1ProverInputs::<Fr>::empty(key.num_cycle_vars())
        .with_outer_remaining_evaluator(&data);
    let mut prover_executor = Stage1ProverKernelExecutor::new(inputs);
    let mut prover_transcript = Blake2bTranscript::<Fr>::new(b"stage1");
    let prover_artifacts = prove_stage1_outer::prove_stage1_outer(
        &mut prover_executor,
        &mut prover_transcript,
    )
    .expect("generated real stage1 prover succeeds with R1CS data");

    let proof = verifier_proof_from_prover_artifacts(&prover_artifacts);
    let mut verifier_transcript = Blake2bTranscript::<Fr>::new(b"stage1");
    let verifier_artifacts = verify_stage1_outer::verify_stage1_outer(
        &proof,
        &mut verifier_transcript,
    )
    .expect("generated real stage1 verifier accepts R1CS-backed proof");

    assert_eq!(prover_transcript.state(), verifier_transcript.state());
    assert_eq!(prover_artifacts.sumchecks.len(), 2);
    assert_eq!(verifier_artifacts.sumchecks.len(), 2);
    for (prover_sumcheck, verifier_sumcheck) in prover_artifacts
        .sumchecks
        .iter()
        .zip(verifier_artifacts.sumchecks.iter())
    {
        assert_eq!(prover_sumcheck.point, verifier_sumcheck.point);
        assert_eq!(prover_sumcheck.evals.len(), verifier_sumcheck.evals.len());
        for (prover_eval, verifier_eval) in prover_sumcheck.evals.iter().zip(&verifier_sumcheck.evals) {
            assert_eq!(prover_eval.oracle, verifier_eval.oracle);
            assert_eq!(prover_eval.value, verifier_eval.value);
        }
    }
}
"#,
    );
    source
}

fn generated_commitment_stage1_chain_main() -> String {
    let mut source = r"mod prove_commitment_phase;
mod prove_stage1_outer;
mod verify_commitment_phase;
mod verify_stage1_outer;

use jolt_dory::DoryScheme;
use jolt_field::Fr;
use jolt_kernels::stage1::{
    Stage1OuterRemainingContext, Stage1OuterRemainingEvaluator, Stage1ProverInputs,
    Stage1ProverKernelExecutor,
};
use jolt_transcript::{Blake2bTranscript, Transcript};

struct SumZeroRemainingEvaluator;

impl Stage1OuterRemainingEvaluator<Fr> for SumZeroRemainingEvaluator {
    fn evaluate(&self, _context: Stage1OuterRemainingContext<'_, Fr>, point: &[Fr]) -> Fr {
        point[0] + point[0] - Fr::from_u64(1)
    }

    fn evaluate_virtual_oracle(
        &self,
        _context: Stage1OuterRemainingContext<'_, Fr>,
        _oracle: &str,
        point: &[Fr],
    ) -> Option<Fr> {
        Some(point.iter().copied().sum())
    }
}

"
    .to_owned();
    source.push_str(tracing_transcript_support());
    source.push_str(&stage1_verifier_proof_adapter(false));
    source.push_str(
        r#"
fn main() {
    let prover_setup =
        DoryScheme::setup_prover(prove_commitment_phase::COMMITMENT_BATCH_PLANS[0].num_vars);
    let commitment_inputs = prove_commitment_phase::CommitmentOracleInputs {
        rd_inc: &[1, 0, 0, 0],
        ram_inc: &[2, 0, 0, 0],
        instruction_keys: &[
            Some(0x1234_5678_9abc_def0_0123_4567_89ab_cdefu128),
            Some(0),
            Some(0),
            Some(0),
        ],
        ram_addresses: &[Some(0), Some(1), Some(2), Some(3)],
        bytecode_indices: &[Some(0), Some(1), Some(2), Some(3)],
        untrusted_advice: None,
        trusted_advice: None,
    };
    let mut commitment_oracles = prove_commitment_phase::build_commitment_oracles(
        &commitment_inputs,
    )
    .expect("build commitment oracles");
    let mut prover_transcript = TracingTranscript::new(b"jolt-chain");
    let commitment = prove_commitment_phase::prove_commitment_phase(
        &mut commitment_oracles,
        &prover_setup,
        &mut prover_transcript,
    )
    .expect("prover commitment phase");

    let extended_evals = vec![Fr::from_u64(0); 9];
    let evaluator = SumZeroRemainingEvaluator;
    let stage1_inputs = Stage1ProverInputs::<Fr>::empty(2)
        .with_uniskip_extended_evals(&extended_evals)
        .with_outer_remaining_evaluator(&evaluator);
    let mut stage1_prover_executor = Stage1ProverKernelExecutor::new(stage1_inputs);
    let stage1 = prove_stage1_outer::prove_stage1_outer(
        &mut stage1_prover_executor,
        &mut prover_transcript,
    )
    .expect("stage1 prover phase");

    let mut verifier_transcript = TracingTranscript::new(b"jolt-chain");
    let verified_commitment = verify_commitment_phase::verify_commitment_phase(
        &commitment.commitments,
        &mut verifier_transcript,
    )
    .expect("verifier commitment phase");
    let stage1_proof = verifier_proof_from_prover_artifacts(&stage1);
    let verified_stage1 = verify_stage1_outer::verify_stage1_outer(
        &stage1_proof,
        &mut verifier_transcript,
    )
    .expect("stage1 verifier phase");

    assert_eq!(commitment.commitments, verified_commitment.commitments);
    assert_eq!(stage1.sumchecks.len(), 2);
    assert_eq!(verified_stage1.sumchecks.len(), 2);
    assert_eq!(stage1.sumchecks[1].point, verified_stage1.sumchecks[1].point);
    assert_transcript_step_parity(&prover_transcript, &verifier_transcript);
}
"#,
    );
    source
}

fn tracing_transcript_support() -> &'static str {
    r"#[derive(Clone, Debug, PartialEq, Eq)]
enum TranscriptEvent {
    Init([u8; 32]),
    Append { bytes: Vec<u8>, state: [u8; 32] },
    Challenge { state: [u8; 32] },
}

#[derive(Clone, Default)]
struct TracingTranscript {
    inner: Blake2bTranscript<Fr>,
    events: Vec<TranscriptEvent>,
}

impl Transcript for TracingTranscript {
    type Challenge = Fr;

    fn new(label: &'static [u8]) -> Self {
        let inner = Blake2bTranscript::<Fr>::new(label);
        let events = vec![TranscriptEvent::Init(*inner.state())];
        Self { inner, events }
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        self.inner.append_bytes(bytes);
        self.events.push(TranscriptEvent::Append {
            bytes: bytes.to_vec(),
            state: *self.inner.state(),
        });
    }

    fn challenge(&mut self) -> Fr {
        let challenge = self.inner.challenge();
        self.events.push(TranscriptEvent::Challenge {
            state: *self.inner.state(),
        });
        challenge
    }

    fn state(&self) -> &[u8; 32] {
        self.inner.state()
    }
}

#[allow(dead_code)]
fn assert_transcript_step_parity(prover: &TracingTranscript, verifier: &TracingTranscript) {
    assert_eq!(prover.events, verifier.events);
    assert_eq!(prover.state(), verifier.state());
}

"
}
