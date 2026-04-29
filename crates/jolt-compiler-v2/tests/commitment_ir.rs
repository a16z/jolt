use jolt_compiler_v2::{
    build_commitment_protocol, emit_commitment_rust, lower_commitment_to_compute,
    lower_compute_to_cpu, lower_piop_and_fiat_shamir, project_prover_party, project_verifier_party,
    verify_concrete_transcript, verify_jolt_protocol_schema, verify_protocol_schema, Concrete, Cpu,
    JoltProtocolParams, MeliorContext, Role, RustSourceFile, TextMlir,
};
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
        .parse_module::<jolt_compiler_v2::Protocol>(registered)
        .expect("registered dialect op parses");

    let unknown = r#"
module @unknown {
  "unknown.dialect_op"() : () -> ()
}
"#;
    let _ = context
        .parse_module::<jolt_compiler_v2::Protocol>(unknown)
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
        .parse_module::<jolt_compiler_v2::Protocol>(&text)
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
        .parse_module::<jolt_compiler_v2::Cpu>(&text)
        .expect("parse CPU MLIR");
    assert!(parsed.to_text_mlir().contains("\"cpu.pcs_commit_batch\""));
    let parsed = context
        .parse_module::<jolt_compiler_v2::Cpu>(&verifier_text)
        .expect("parse verifier CPU MLIR");
    assert!(parsed.to_text_mlir().contains("\"cpu.pcs_receive_batch\""));
}

#[test]
fn generic_protocol_schema_accepts_non_jolt_params() {
    let context = MeliorContext::new();
    let generic = context.new_module::<jolt_compiler_v2::Protocol>("generic", None);
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

    verify_protocol_schema(&generic).expect("generic schema does not require Jolt attrs");
}

#[test]
fn protocol_schema_rejects_bad_derived_params() {
    let context = MeliorContext::new();
    let bad = context.new_module::<jolt_compiler_v2::Protocol>("bad", None);
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

fn build_small_commitment_cpu(
    context: &MeliorContext,
    role: Role,
) -> jolt_compiler_v2::BoltModule<'_, Cpu> {
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
) -> (
    jolt_compiler_v2::BoltModule<'c, Cpu>,
    jolt_compiler_v2::BoltModule<'c, Cpu>,
) {
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

fn assert_or_update_fixture(path: &str, actual: &str) {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(path);
    if std::env::var_os("JOLT_UPDATE_GOLDENS").is_some() {
        std::fs::write(&path, actual).expect("write golden fixture");
        return;
    }
    let expected = std::fs::read_to_string(&path).expect("read golden fixture");
    assert_eq!(expected, actual);
}

fn assert_rust_source_compiles(_filename: &str, source: &str) {
    let dir = new_temp_dir("jolt_compiler_v2_emit");
    let workspace_root = workspace_root();
    std::fs::write(
        dir.join("Cargo.toml"),
        generated_crate_manifest(&workspace_root),
    )
    .expect("write generated cargo manifest");
    std::fs::create_dir_all(dir.join("src")).expect("create generated src dir");
    std::fs::write(dir.join("src/lib.rs"), source).expect("write generated source");
    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_owned());
    let output = Command::new(cargo)
        .arg("check")
        .arg("--manifest-path")
        .arg(dir.join("Cargo.toml"))
        .arg("-q")
        .env(
            "CARGO_TARGET_DIR",
            workspace_root.join("target/jolt-compiler-v2-generated"),
        )
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

fn assert_generated_commitment_self_parity_runs(
    prover_source: &RustSourceFile,
    verifier_source: &RustSourceFile,
    main_source: &str,
) {
    let dir = new_temp_dir("jolt_compiler_v2_self_parity");
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
        .env(
            "CARGO_TARGET_DIR",
            workspace_root.join("target/jolt-compiler-v2-generated"),
        )
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

fn workspace_root() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn generated_crate_manifest(workspace_root: &Path) -> String {
    format!(
        r#"[package]
name = "generated-commitment-phase-check"
version = "0.0.0"
edition = "2021"

[dependencies]
jolt-dory = {{ path = "{}" }}
jolt-field = {{ path = "{}" }}
jolt-openings = {{ path = "{}" }}
jolt-transcript = {{ path = "{}" }}
jolt-witness-v2 = {{ path = "{}" }}
"#,
        workspace_root.join("crates/jolt-dory").display(),
        workspace_root.join("crates/jolt-field").display(),
        workspace_root.join("crates/jolt-openings").display(),
        workspace_root.join("crates/jolt-transcript").display(),
        workspace_root.join("crates/jolt-witness-v2").display(),
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
use jolt_field::{Field, Fr};
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
use jolt_field::{Field, Fr};
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

fn assert_transcript_step_parity(prover: &TracingTranscript, verifier: &TracingTranscript) {
    assert_eq!(prover.events, verifier.events);
    assert_eq!(prover.state(), verifier.state());
}

"
}
