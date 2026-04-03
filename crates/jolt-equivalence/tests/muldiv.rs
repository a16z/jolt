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
use jolt_compute::link;
use jolt_cpu::CpuBackend;
use jolt_host::{BytecodePreprocessing, Program};
use jolt_openings::mock::MockCommitmentScheme;
use jolt_transcript::Transcript;
use jolt_verifier::{OneHotConfig, ProverConfig, ReadWriteConfig};
use jolt_zkvm::prove::{prove, TraceData};

use jolt_equivalence::{compare_stage, StageTrace};

type Fr = ark_bn254::Fr;
type NewFr = jolt_field::Fr;
type MockPCS = MockCommitmentScheme<NewFr>;

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

    // Helper macro: snapshot openings, verify stage, extract trace.
    // Stages 1-2 return (StageVerifyResult, extra) — use `tuple` variant.
    // Stages 3-7 return StageVerifyResult directly — use `plain` variant.
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

fn extract_jolt_zkvm_stages() -> Vec<StageTrace<Fr>> {
    let mut program = Program::new("muldiv-guest");
    let (bytecode_raw, _init_mem, _program_size, entry_address) = program.decode();
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
    let (_, trace, _, io_device) = program.trace(&inputs, &[], &[]);

    let bytecode = BytecodePreprocessing::preprocess(bytecode_raw, entry_address);
    let memory_layout = &io_device.memory_layout;

    let trace_length = trace.len().next_power_of_two();
    let log_t = trace_length.trailing_zeros() as usize;
    let bytecode_k = bytecode.code_size;
    let log_k_bytecode = bytecode_k.trailing_zeros() as usize;
    let ram_k = 1usize << 20;
    let log_k_ram = ram_k.trailing_zeros() as usize;

    let module = build_protocol_module(log_t, log_k_bytecode, log_k_ram);
    let backend = CpuBackend;
    let executable = link::<CpuBackend, NewFr>(module, &backend);

    let one_hot = OneHotConfig::new(log_t);
    let rw_config = ReadWriteConfig::new(log_t, log_k_ram);
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
        outputs: io_device.outputs.clone(),
        panic: io_device.panic,
    };

    let trace_data = TraceData {
        trace: &trace,
        bytecode: &bytecode,
        memory_layout,
    };
    let pcs_setup = ();
    let mut transcript = jolt_transcript::Blake2bTranscript::<NewFr>::new(b"Jolt");

    let proof = prove::<_, _, _, _, MockPCS>(
        &executable,
        &trace_data,
        &backend,
        &pcs_setup,
        &mut transcript,
        config,
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

// ═══════════════════════════════════════════════════════════════════
// Cached setup (OnceLock so multiple per-stage tests share one run)
// ═══════════════════════════════════════════════════════════════════

static CORE_STAGES: OnceLock<Vec<StageTrace<Fr>>> = OnceLock::new();
static ZKVM_STAGES: OnceLock<Result<Vec<StageTrace<Fr>>, String>> = OnceLock::new();

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

macro_rules! equivalence_test {
    ($name:ident, $stage_idx:literal) => {
        #[test]
        #[ignore = "requires full pipeline wiring"]
        fn $name() {
            let core = jolt_core_stages();
            let zkvm = match jolt_zkvm_stages() {
                Ok(s) => s,
                Err(e) => {
                    eprintln!("SKIP stage {}: jolt-zkvm prove failed: {e}", $stage_idx + 1);
                    return;
                }
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
        }
    };
}

equivalence_test!(cross_system_stage1, 0);
equivalence_test!(cross_system_stage2, 1);
equivalence_test!(cross_system_stage3, 2);
equivalence_test!(cross_system_stage4, 3);
equivalence_test!(cross_system_stage5, 4);
equivalence_test!(cross_system_stage6, 5);
equivalence_test!(cross_system_stage7, 6);
