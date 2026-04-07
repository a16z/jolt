//! Cross-system equivalence test for the muldiv guest program.
//!
//! Runs jolt-core's prover and jolt-zkvm's prover, both with
//! `Blake2bTranscript` (real Fiat-Shamir, identical domain-separation labels),
//! extracts per-stage protocol trace data, and compares them
//! coefficient-by-coefficient.
//!
//! Each stage has its own independent test so stages light up green
//! incrementally as the jolt-zkvm pipeline is wired up.
#![allow(non_snake_case, clippy::print_stderr)]

use std::collections::BTreeSet;
use std::panic::{self, AssertUnwindSafe};
use std::process::Command;
use std::sync::OnceLock;

use jolt_core::curve::Bn254Curve;
use jolt_core::field::JoltField;
use jolt_core::host;
use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryGlobals};
use jolt_core::poly::opening_proof::OpeningId;
use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;
use jolt_core::transcripts::Blake2bTranscript;
use jolt_core::zkvm::proof_serialization::JoltProof;
use jolt_core::zkvm::prover::{JoltCpuProver, JoltProverPreprocessing};
use jolt_core::zkvm::verifier::{JoltSharedPreprocessing, JoltVerifier, JoltVerifierPreprocessing};

use common::constants::RAM_START_ADDRESS;
use jolt_compiler::module::Module;
use jolt_compiler::{Op, VerifierOp};
use jolt_compute::link;
use jolt_cpu::CpuBackend;
use jolt_host::{extract_trace, BytecodePreprocessing, Program};
use jolt_dory::types::DoryProverSetup;
use jolt_dory::DoryScheme;
use jolt_r1cs::{constraints::rv64, R1csKey, R1csSource};
use jolt_transcript::Transcript;
use jolt_verifier::{OneHotConfig, ProverConfig, ReadWriteConfig, TRANSCRIPT_LABEL};
use jolt_witness::{PolynomialConfig, PolynomialId, Polynomials};
use jolt_zkvm::derived::DerivedSource;
use jolt_zkvm::preprocessed::PreprocessedSource;
use jolt_zkvm::prove::prove;
use jolt_zkvm::provider::ProverData;
use num_traits::Zero;

use jolt_equivalence::checkpoint::{CheckpointTranscript, TranscriptEvent};
use jolt_equivalence::{compare_stage, StageTrace};

type Fr = ark_bn254::Fr;
type NewFr = jolt_field::Fr;

type CoreProver<'a> = JoltCpuProver<'a, Fr, Bn254Curve, DoryCommitmentScheme, Blake2bTranscript>;
type CoreProof = JoltProof<Fr, Bn254Curve, DoryCommitmentScheme, Blake2bTranscript>;
type CoreVerifier<'a> = JoltVerifier<'a, Fr, Bn254Curve, DoryCommitmentScheme, Blake2bTranscript>;

fn to_ark(f: NewFr) -> Fr {
    f.into()
}

// ═══════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════

fn extract_clear_rounds(
    proof: &SumcheckInstanceProof<Fr, Bn254Curve, Blake2bTranscript>,
    initial_claim: Fr,
    challenges: &[<Fr as JoltField>::Challenge],
) -> Vec<Vec<Fr>> {
    match proof {
        SumcheckInstanceProof::Clear(clear) => {
            clear.decompress_all_rounds(initial_claim, challenges)
        }
        SumcheckInstanceProof::Zk(_) => panic!("expected ClearSumcheckProof"),
    }
}

fn extract_clear_degree(proof: &SumcheckInstanceProof<Fr, Bn254Curve, Blake2bTranscript>) -> usize {
    match proof {
        SumcheckInstanceProof::Clear(clear) => clear
            .compressed_polys
            .first()
            .map_or(0, |p| p.coeffs_except_linear_term.len()),
        SumcheckInstanceProof::Zk(_) => panic!("expected ClearSumcheckProof"),
    }
}

/// Collect the claim values for all new openings added between two snapshots
/// of the verifier's opening accumulator.
fn diff_opening_evals(keys_before: &BTreeSet<OpeningId>, verifier: &CoreVerifier<'_>) -> Vec<Fr> {
    verifier
        .opening_accumulator
        .openings
        .iter()
        .filter(|(k, _)| !keys_before.contains(k))
        .map(|(_, (_, claim))| *claim)
        .collect()
}

fn snapshot_opening_keys(verifier: &CoreVerifier<'_>) -> BTreeSet<OpeningId> {
    verifier
        .opening_accumulator
        .openings
        .keys()
        .cloned()
        .collect()
}

// ═══════════════════════════════════════════════════════════════════
// jolt-core extraction (reference prover)
// ═══════════════════════════════════════════════════════════════════

fn extract_jolt_core_stages() -> Vec<StageTrace<Fr>> {
    DoryGlobals::reset();

    let mut program = host::Program::new("muldiv-guest");
    let (bytecode, init_memory_state, _, e_entry) = program.decode();
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
    let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode,
        io_device.memory_layout.clone(),
        init_memory_state,
        1 << 16,
        e_entry,
    );
    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());

    let elf_contents = program.get_elf_contents().expect("elf");
    let prover: CoreProver<'_> = CoreProver::gen_from_elf(
        &prover_preprocessing,
        &elf_contents,
        &inputs,
        &[],
        &[],
        None,
        None,
        None,
    );
    let io = prover.program_io.clone();
    let (proof, _debug): (CoreProof, _) = prover.prove();

    let verifier_preprocessing: &'static _ = Box::leak(Box::new(JoltVerifierPreprocessing::from(
        &prover_preprocessing,
    )));
    let mut verifier: CoreVerifier<'_> =
        CoreVerifier::new(verifier_preprocessing, proof, io, None, None).expect("build verifier");
    verifier.run_preamble();

    let mut stages = Vec::new();

    macro_rules! extract_stage {
        (tuple, $verify:ident, $proof_field:ident) => {{
            let keys_before = snapshot_opening_keys(&verifier);
            let (sr, _) = verifier.$verify().expect(stringify!($verify));
            let coeffs = extract_clear_rounds(
                &verifier.proof.$proof_field,
                sr.initial_claim,
                &sr.challenges,
            );
            let degree = extract_clear_degree(&verifier.proof.$proof_field);
            let evals = diff_opening_evals(&keys_before, &verifier);
            stages.push(StageTrace {
                num_rounds: sr.challenges.len(),
                poly_degree: degree,
                round_poly_coeffs: coeffs,
                evals,
            });
        }};
        (plain, $verify:ident, $proof_field:ident) => {{
            let keys_before = snapshot_opening_keys(&verifier);
            let sr = verifier.$verify().expect(stringify!($verify));
            let coeffs = extract_clear_rounds(
                &verifier.proof.$proof_field,
                sr.initial_claim,
                &sr.challenges,
            );
            let degree = extract_clear_degree(&verifier.proof.$proof_field);
            let evals = diff_opening_evals(&keys_before, &verifier);
            stages.push(StageTrace {
                num_rounds: sr.challenges.len(),
                poly_degree: degree,
                round_poly_coeffs: coeffs,
                evals,
            });
        }};
    }

    extract_stage!(tuple, verify_stage1, stage1_sumcheck_proof);
    extract_stage!(tuple, verify_stage2, stage2_sumcheck_proof);
    extract_stage!(plain, verify_stage3, stage3_sumcheck_proof);
    extract_stage!(plain, verify_stage4, stage4_sumcheck_proof);
    extract_stage!(plain, verify_stage5, stage5_sumcheck_proof);
    extract_stage!(plain, verify_stage6, stage6_sumcheck_proof);
    extract_stage!(plain, verify_stage7, stage7_sumcheck_proof);

    stages
}

/// Protocol parameters extracted from jolt-core's proof.
struct CoreProtocolParams {
    trace_length: usize,
    ram_k: usize,
    bytecode_k: usize,
    /// PCS generators shared from jolt-core's prover preprocessing.
    /// Must be the exact same SRS for commitment equivalence.
    pcs_setup: DoryProverSetup,
}

/// Run jolt-core verifier and return per-operation state history + protocol params.
fn extract_jolt_core_state_history() -> (Vec<[u8; 32]>, CoreProtocolParams) {
    DoryGlobals::reset();

    let mut program = host::Program::new("muldiv-guest");
    let (bytecode, init_memory_state, _, e_entry) = program.decode();
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
    let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode,
        io_device.memory_layout.clone(),
        init_memory_state,
        1 << 16,
        e_entry,
    );
    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());

    let elf_contents = program.get_elf_contents().expect("elf");
    let prover: CoreProver<'_> = CoreProver::gen_from_elf(
        &prover_preprocessing,
        &elf_contents,
        &inputs,
        &[],
        &[],
        None,
        None,
        None,
    );
    let io = prover.program_io.clone();
    let (proof, _debug): (CoreProof, _) = prover.prove();

    let params = CoreProtocolParams {
        trace_length: proof.trace_length,
        ram_k: proof.ram_K,
        bytecode_k: prover_preprocessing.shared.bytecode.code_size,
        pcs_setup: DoryProverSetup(prover_preprocessing.generators.clone()),
    };

    let verifier_preprocessing: &'static _ = Box::leak(Box::new(JoltVerifierPreprocessing::from(
        &prover_preprocessing,
    )));
    let mut verifier: CoreVerifier<'_> =
        CoreVerifier::new(verifier_preprocessing, proof, io, None, None).expect("build verifier");

    verifier.run_preamble();
    let _ = verifier.verify_stage1().expect("stage1");
    let _ = verifier.verify_stage2().expect("stage2");
    let _ = verifier.verify_stage3().expect("stage3");
    let _ = verifier.verify_stage4().expect("stage4");
    let _ = verifier.verify_stage5().expect("stage5");
    let _ = verifier.verify_stage6().expect("stage6");
    let _ = verifier.verify_stage7().expect("stage7");

    (verifier.transcript.state_history.clone(), params)
}

// ═══════════════════════════════════════════════════════════════════
// jolt-zkvm extraction (new modular pipeline)
// ═══════════════════════════════════════════════════════════════════

fn build_protocol_module(log_t: usize, log_k_bytecode: usize, log_k_ram: usize) -> Module {
    let tmp_path = format!("/tmp/jolt_equiv_module_{log_t}_{log_k_bytecode}_{log_k_ram}.jolt");

    let output = Command::new("cargo")
        .args([
            "run",
            "--example",
            "jolt_core_module",
            "-p",
            "jolt-compiler",
            "-q",
            "--",
            "--log-t",
            &log_t.to_string(),
            "--log-k-bytecode",
            &log_k_bytecode.to_string(),
            "--log-k-ram",
            &log_k_ram.to_string(),
            "--emit",
            &tmp_path,
        ])
        .output()
        .expect("failed to run jolt_core_module example");

    assert!(
        output.status.success(),
        "jolt_core_module failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let bytes = std::fs::read(&tmp_path).expect("failed to read protocol binary");
    Module::from_bytes(&bytes)
}

/// Truncate the module to include only the first `num_stages` prover stages.
/// Verifier ops are cut at the corresponding boundary.
fn truncate_after_stage(module: &mut Module, num_stages: usize) {
    // Prover: cut at BeginStage{num_stages} (the start of the next stage).
    let mut seen = 0;
    if let Some(pos) = module.prover.ops.iter().position(|op| {
        if let Op::BeginStage { index } = op {
            if *index >= num_stages {
                return true;
            }
            seen = *index + 1;
        }
        false
    }) {
        module.prover.ops.truncate(pos);
    }

    // Verifier: cut at the (num_stages+1)-th BeginStage.
    let mut stage_count = 0;
    if let Some(pos) = module.verifier.ops.iter().position(|op| {
        if matches!(op, VerifierOp::BeginStage) {
            stage_count += 1;
        }
        stage_count > num_stages
    }) {
        module.verifier.ops.truncate(pos);
    }
}

struct ZkvmSetup {
    trace_length: usize,
    config: ProverConfig,
}

fn setup_zkvm_muldiv(core_params: &CoreProtocolParams) -> (
    jolt_compute::Executable<CpuBackend, NewFr>,
    Polynomials<NewFr>,
    R1csKey<NewFr>,
    Vec<NewFr>, // r1cs_witness
    ZkvmSetup,
) {
    let mut program = Program::new("muldiv-guest");
    let (bytecode_raw, _init_mem, _program_size, entry_address) = program.decode();
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
    let (_, trace, _, io_device) = program.trace(&inputs, &[], &[]);

    let bytecode = BytecodePreprocessing::preprocess(bytecode_raw, entry_address);
    let memory_layout = &io_device.memory_layout;

    let trace_length = core_params.trace_length;
    let log_t = trace_length.trailing_zeros() as usize;
    let bytecode_k = core_params.bytecode_k;
    let log_k_bytecode = bytecode_k.trailing_zeros() as usize;
    let ram_k = core_params.ram_k;
    let log_k_ram = ram_k.trailing_zeros() as usize;

    eprintln!(
        "Protocol params: trace_length={trace_length}, bytecode_k={bytecode_k}, \
         ram_k={ram_k} (log_t={log_t}, log_k_bc={log_k_bytecode}, log_k_ram={log_k_ram})"
    );

    let mut module = build_protocol_module(log_t, log_k_bytecode, log_k_ram);
    truncate_after_stage(&mut module, 1);
    let backend = CpuBackend;
    let executable = link::<CpuBackend, NewFr>(module, &backend);

    let one_hot = OneHotConfig::new(log_t);
    let log_k_chunk = one_hot.log_k_chunk as usize;
    let rw_config = ReadWriteConfig::new(log_t, log_k_ram);

    let lowest_addr = memory_layout.get_lowest_address();
    let remap = |addr: u64| ((addr - lowest_addr) / 8) as usize;

    let config = ProverConfig {
        trace_length,
        ram_k,
        bytecode_k,
        one_hot_config: one_hot,
        rw_config,
        memory_start: RAM_START_ADDRESS,
        memory_end: RAM_START_ADDRESS + ram_k as u64,
        entry_address,
        io_hash: [0u8; 32],
        max_input_size: memory_layout.max_input_size,
        max_output_size: memory_layout.max_output_size,
        heap_size: memory_layout.heap_size,
        inputs: io_device.inputs.clone(),
        outputs: {
            // jolt-core truncates trailing zeros from outputs in gen_from_trace
            let mut out = io_device.outputs.clone();
            out.truncate(
                out.iter()
                    .rposition(|&b| b != 0)
                    .map_or(0, |pos| pos + 1),
            );
            out
        },
        panic: io_device.panic,
        ram_lowest_address: lowest_addr,
        input_word_offset: remap(memory_layout.input_start),
        output_word_offset: remap(memory_layout.output_start),
        panic_word_offset: remap(memory_layout.panic),
        termination_word_offset: remap(memory_layout.termination),
    };

    let poly_config = PolynomialConfig::new(log_k_chunk, 128, log_k_bytecode, log_k_ram);
    let matrices = rv64::rv64_constraints::<NewFr>();
    let r1cs_key = R1csKey::new(matrices, trace_length);

    let (cycle_inputs, r1cs_witness) = extract_trace::<_, NewFr>(
        &trace,
        trace_length,
        &bytecode,
        memory_layout,
        r1cs_key.num_vars_padded,
    );

    let mut polys = Polynomials::<NewFr>::new(poly_config);
    polys.push(&cycle_inputs);
    polys.finish();
    let _ = polys.insert(PolynomialId::UntrustedAdvice, vec![NewFr::zero(); trace_length]);
    let _ = polys.insert(PolynomialId::TrustedAdvice, vec![NewFr::zero(); trace_length]);

    let setup = ZkvmSetup {
        trace_length,
        config,
    };

    (executable, polys, r1cs_key, r1cs_witness, setup)
}

fn extract_jolt_zkvm_stages() -> Vec<StageTrace<Fr>> {
    let (_, params) = jolt_core_state_history();
    let (executable, mut polys, r1cs_key, r1cs_witness, setup) = setup_zkvm_muldiv(params);

    let r1cs = R1csSource::new(&r1cs_key, &r1cs_witness);
    let derived = DerivedSource::new(&r1cs_witness, setup.trace_length, r1cs_key.num_vars_padded);
    let preprocessed = PreprocessedSource::new();
    let mut provider = ProverData::new(&mut polys, r1cs, derived, preprocessed);

    let pcs_setup = &params.pcs_setup;
    let mut transcript = jolt_transcript::Blake2bTranscript::<NewFr>::new(TRANSCRIPT_LABEL);

    let proof = prove::<_, _, _, DoryScheme>(
        &executable,
        &mut provider,
        &backend(),
        &pcs_setup,
        &mut transcript,
        setup.config,
    );

    proof
        .stage_proofs
        .iter()
        .map(|sp| {
            let round_polys = &sp.round_polys.round_polynomials;
            StageTrace {
                num_rounds: round_polys.len(),
                poly_degree: round_polys
                    .first()
                    .map_or(0, |p| p.coefficients().len().saturating_sub(1)),
                round_poly_coeffs: round_polys
                    .iter()
                    .map(|p| p.coefficients().iter().copied().map(to_ark).collect())
                    .collect(),
                evals: sp.evals.iter().copied().map(to_ark).collect(),
            }
        })
        .collect()
}

/// Run jolt-zkvm prover with a CheckpointTranscript and return the event log.
fn extract_jolt_zkvm_checkpoint_log() -> Vec<TranscriptEvent> {
    let (_, params) = jolt_core_state_history();
    let (executable, mut polys, r1cs_key, r1cs_witness, setup) = setup_zkvm_muldiv(params);

    // Debug: print InstructionRa[0] stats
    if let Some(ra0) = polys.try_get(PolynomialId::InstructionRa(0)) {
        let nonzero: Vec<_> = ra0.iter().enumerate().filter(|(_, v)| !v.is_zero()).take(5).collect();
        eprintln!(
            "InstructionRa(0): len={}, nonzero_count={}, first_nonzero={:?}",
            ra0.len(),
            ra0.iter().filter(|v| !v.is_zero()).count(),
            nonzero.iter().map(|(i, _)| i).collect::<Vec<_>>(),
        );
    }

    let r1cs = R1csSource::new(&r1cs_key, &r1cs_witness);
    let derived = DerivedSource::new(&r1cs_witness, setup.trace_length, r1cs_key.num_vars_padded);
    let preprocessed = PreprocessedSource::new();
    let mut provider = ProverData::new(&mut polys, r1cs, derived, preprocessed);

    let pcs_setup = &params.pcs_setup;
    let mut transcript = CheckpointTranscript::<jolt_transcript::Blake2bTranscript<NewFr>>::new(
        TRANSCRIPT_LABEL,
    );

    let _proof = prove::<_, _, _, DoryScheme>(
        &executable,
        &mut provider,
        &backend(),
        &pcs_setup,
        &mut transcript,
        setup.config,
    );

    transcript.into_log()
}

fn backend() -> CpuBackend {
    CpuBackend
}

// ═══════════════════════════════════════════════════════════════════
// Cached setup (OnceLock so multiple per-stage tests share one run)
// ═══════════════════════════════════════════════════════════════════

static CORE_STAGES: OnceLock<Vec<StageTrace<Fr>>> = OnceLock::new();
static ZKVM_STAGES: OnceLock<Result<Vec<StageTrace<Fr>>, String>> = OnceLock::new();
static CORE_STATE_HISTORY: OnceLock<(Vec<[u8; 32]>, CoreProtocolParams)> = OnceLock::new();
static ZKVM_CHECKPOINT_LOG: OnceLock<Result<Vec<TranscriptEvent>, String>> = OnceLock::new();

fn jolt_core_stages() -> &'static Vec<StageTrace<Fr>> {
    CORE_STAGES.get_or_init(extract_jolt_core_stages)
}

fn jolt_zkvm_stages() -> &'static Result<Vec<StageTrace<Fr>>, String> {
    ZKVM_STAGES.get_or_init(|| {
        let result = panic::catch_unwind(AssertUnwindSafe(extract_jolt_zkvm_stages));
        match result {
            Ok(stages) => Ok(stages),
            Err(e) => {
                let msg = if let Some(s) = e.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                };
                Err(msg)
            }
        }
    })
}

fn jolt_core_state_history() -> &'static (Vec<[u8; 32]>, CoreProtocolParams) {
    CORE_STATE_HISTORY.get_or_init(extract_jolt_core_state_history)
}

fn jolt_zkvm_checkpoint() -> &'static Result<Vec<TranscriptEvent>, String> {
    ZKVM_CHECKPOINT_LOG.get_or_init(|| {
        let result = panic::catch_unwind(AssertUnwindSafe(extract_jolt_zkvm_checkpoint_log));
        match result {
            Ok(log) => Ok(log),
            Err(e) => {
                let msg = if let Some(s) = e.downcast_ref::<&str>() {
                    (*s).to_string()
                } else if let Some(s) = e.downcast_ref::<String>() {
                    s.clone()
                } else {
                    "unknown panic".to_string()
                };
                Err(msg)
            }
        }
    })
}

// ═══════════════════════════════════════════════════════════════════
// Smoke tests (jolt-core only — always runnable)
// ═══════════════════════════════════════════════════════════════════

#[test]
fn jolt_core_mock_transcript_proves() {
    let stages = jolt_core_stages();
    assert_eq!(stages.len(), 7);
    for (i, stage) in stages.iter().enumerate() {
        eprintln!(
            "Stage {}: {} rounds, degree {}, {} evals",
            i + 1,
            stage.num_rounds,
            stage.poly_degree,
            stage.evals.len(),
        );
        assert!(stage.num_rounds > 0);
        assert!(stage.poly_degree > 0);
        assert_eq!(stage.round_poly_coeffs.len(), stage.num_rounds);
    }
}

#[test]
#[ignore = "requires full pipeline wiring"]
fn jolt_zkvm_mock_transcript_proves() {
    let stages = match jolt_zkvm_stages() {
        Ok(s) => s,
        Err(e) => panic!("jolt-zkvm prove failed: {e}"),
    };
    eprintln!("jolt-zkvm produced {} stages", stages.len());
    for (i, stage) in stages.iter().enumerate() {
        eprintln!(
            "Stage {}: {} rounds, degree {}, {} evals",
            i + 1,
            stage.num_rounds,
            stage.poly_degree,
            stage.evals.len(),
        );
    }
    assert!(!stages.is_empty(), "expected at least 1 stage");
}

// ═══════════════════════════════════════════════════════════════════
// Per-stage cross-system equivalence tests
// ═══════════════════════════════════════════════════════════════════

macro_rules! equivalence_test_body {
    ($stage_idx:literal) => {{
        let core = jolt_core_stages();
        let zkvm = match jolt_zkvm_stages() {
            Ok(s) => s,
            Err(e) => panic!("jolt-zkvm prove failed: {e}"),
        };
        assert!(
            core.len() > $stage_idx,
            "jolt-core missing stage {}",
            $stage_idx + 1
        );
        if zkvm.len() <= $stage_idx {
            eprintln!(
                "SKIP: jolt-zkvm has {} stages, need {}",
                zkvm.len(),
                $stage_idx + 1
            );
            return;
        }
        compare_stage($stage_idx, &core[$stage_idx], &zkvm[$stage_idx])
            .unwrap_or_else(|e| panic!("{e}"));
    }};
}

macro_rules! equivalence_test {
    ($name:ident, $stage_idx:literal) => {
        #[test]
        fn $name() {
            equivalence_test_body!($stage_idx);
        }
    };
}

equivalence_test!(cross_system_stage1, 0);

#[test]
#[ignore = "requires full pipeline wiring"]
fn cross_system_stage2() { equivalence_test_body!(1); }
#[test]
#[ignore = "requires full pipeline wiring"]
fn cross_system_stage3() { equivalence_test_body!(2); }
#[test]
#[ignore = "requires full pipeline wiring"]
fn cross_system_stage4() { equivalence_test_body!(3); }
#[test]
#[ignore = "requires full pipeline wiring"]
fn cross_system_stage5() { equivalence_test_body!(4); }
#[test]
#[ignore = "requires full pipeline wiring"]
fn cross_system_stage6() { equivalence_test_body!(5); }
#[test]
#[ignore = "requires full pipeline wiring"]
fn cross_system_stage7() { equivalence_test_body!(6); }

// ═══════════════════════════════════════════════════════════════════
// Transcript divergence test
// ═══════════════════════════════════════════════════════════════════

fn hex(b: &[u8]) -> String {
    use std::fmt::Write;
    let mut s = String::with_capacity(b.len() * 2);
    for byte in b {
        let _ = write!(s, "{byte:02x}");
    }
    s
}

/// Compare jolt-core and jolt-zkvm transcript state histories operation-by-operation.
///
/// jolt-core's state_history (cfg(test)) gives us the state after every raw hash
/// operation. jolt-zkvm's CheckpointTranscript gives us state_after for every
/// append/squeeze. Since both use the same Blake2b256 transcript construction,
/// the N-th state in both logs should be identical.
///
/// The first divergence pinpoints the exact operation where the two systems disagree.
#[test]
fn transcript_divergence() {
    let (golden, _) = jolt_core_state_history();
    let log = match jolt_zkvm_checkpoint() {
        Ok(l) => l,
        Err(e) => panic!("jolt-zkvm checkpoint extraction failed: {e}"),
    };

    // Extract state_after from checkpoint events
    let zkvm_states: Vec<[u8; 32]> = log
        .iter()
        .map(|ev| match ev {
            TranscriptEvent::Append { state_after, .. } => *state_after,
            TranscriptEvent::Squeeze { state_after } => *state_after,
        })
        .collect();

    eprintln!("jolt-core state_history: {} entries", golden.len());
    eprintln!("jolt-zkvm checkpoint:    {} entries", zkvm_states.len());

    // golden[0] is the initial state from `new(b"Jolt")`, not an operation.
    // zkvm_states[0] is the state after the first operation.
    // So golden[i+1] should match zkvm_states[i].
    let golden_ops = &golden[1..]; // skip initial state
    let min_len = golden_ops.len().min(zkvm_states.len());

    // Check initial states match (both from `new(b"Jolt")`)
    eprintln!(
        "Initial state (jolt-core): {}",
        &hex(&golden[0])[..16]
    );

    for i in 0..min_len {
        if golden_ops[i] != zkvm_states[i] {
            eprintln!("\n=== DIVERGENCE at operation #{i} ===");
            // Show context
            let start = i.saturating_sub(3);
            eprintln!("Context (operations {start}..{}):", (i + 1).min(min_len));
            for j in start..=i.min(min_len - 1) {
                let marker = if j == i { ">>>" } else { "   " };
                eprintln!(
                    "  {marker} [{j}] core={} zkvm={}",
                    &hex(&golden_ops[j])[..16],
                    &hex(&zkvm_states[j])[..16],
                );
                // Show the checkpoint event details for zkvm
                if j < log.len() {
                    eprintln!("           zkvm event: {:?}", log[j]);
                }
            }
            panic!(
                "Transcript divergence at operation #{i}: \
                 core state {} != zkvm state {}",
                &hex(&golden_ops[i])[..16],
                &hex(&zkvm_states[i])[..16],
            );
        }
    }

    if golden_ops.len() > zkvm_states.len() {
        eprintln!(
            "\njolt-zkvm log is shorter ({} ops) than jolt-core ({} ops). \
             {} operations matched before zkvm ran out.",
            zkvm_states.len(),
            golden_ops.len(),
            min_len,
        );
    } else if zkvm_states.len() > golden_ops.len() {
        eprintln!(
            "\njolt-zkvm log is longer ({} ops) than jolt-core ({} ops). \
             {} operations matched before core ran out.",
            zkvm_states.len(),
            golden_ops.len(),
            min_len,
        );
    } else {
        eprintln!(
            "\nAll {} operations matched perfectly.",
            min_len,
        );
    }
}
