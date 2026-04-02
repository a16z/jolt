//! Cross-system equivalence test for the muldiv guest program.
//!
//! Runs jolt-core's prover with `MockTranscript` (deterministic challenges),
//! extracts per-stage protocol trace data, and compares against jolt-zkvm's
//! output using the same deterministic transcript.
#![allow(non_snake_case, unused_results, clippy::print_stderr)]

use jolt_core::curve::Bn254Curve;
use jolt_core::field::JoltField;
use jolt_core::host;
use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryGlobals};
use jolt_core::subprotocols::sumcheck::SumcheckInstanceProof;
use jolt_core::transcripts::MockTranscript;
use jolt_core::zkvm::proof_serialization::JoltProof;
use jolt_core::zkvm::prover::{JoltCpuProver, JoltProverPreprocessing};
use jolt_core::zkvm::verifier::{JoltSharedPreprocessing, JoltVerifier, JoltVerifierPreprocessing};

use jolt_equivalence::{ProtocolTrace, StageTrace};

type Fr = ark_bn254::Fr;

type MockProver<'a> = JoltCpuProver<'a, Fr, Bn254Curve, DoryCommitmentScheme, MockTranscript>;
type MockProof = JoltProof<Fr, Bn254Curve, DoryCommitmentScheme, MockTranscript>;
type MockVerifier<'a> = JoltVerifier<'a, Fr, Bn254Curve, DoryCommitmentScheme, MockTranscript>;

fn extract_clear_rounds(
    proof: &SumcheckInstanceProof<Fr, Bn254Curve, MockTranscript>,
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

fn extract_clear_degree(proof: &SumcheckInstanceProof<Fr, Bn254Curve, MockTranscript>) -> usize {
    match proof {
        SumcheckInstanceProof::Clear(clear) => {
            // compressed stores [c0, c2, c3, ..., cd], so degree = len
            clear
                .compressed_polys
                .first()
                .map_or(0, |p| p.coeffs_except_linear_term.len())
        }
        SumcheckInstanceProof::Zk(_) => panic!("expected ClearSumcheckProof"),
    }
}

/// Run jolt-core's muldiv prover + verifier with MockTranscript,
/// extract per-stage trace data.
fn run_jolt_core() -> ProtocolTrace<Fr> {
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
    let prover: MockProver<'_> = MockProver::gen_from_elf(
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
    let (proof, _debug): (MockProof, _) = prover.prove();

    // Run verifier to extract per-stage initial claims + challenges
    let verifier_preprocessing: &'static _ = Box::leak(Box::new(JoltVerifierPreprocessing::from(
        &prover_preprocessing,
    )));
    let mut verifier: MockVerifier<'_> =
        MockVerifier::new(verifier_preprocessing, proof, io, None, None)
            .expect("build mock verifier");
    verifier.run_preamble();

    let mut stages = Vec::new();

    // Stage 1
    let (s1, _) = verifier.verify_stage1().expect("stage 1");
    let coeffs1 = extract_clear_rounds(
        &verifier.proof.stage1_sumcheck_proof,
        s1.initial_claim,
        &s1.challenges,
    );
    let degree1 = extract_clear_degree(&verifier.proof.stage1_sumcheck_proof);
    stages.push(StageTrace {
        num_rounds: s1.challenges.len(),
        poly_degree: degree1,
        round_poly_coeffs: coeffs1,
        evals: vec![],
    });

    // Stage 2
    let (s2, _) = verifier.verify_stage2().expect("stage 2");
    let coeffs2 = extract_clear_rounds(
        &verifier.proof.stage2_sumcheck_proof,
        s2.initial_claim,
        &s2.challenges,
    );
    let degree2 = extract_clear_degree(&verifier.proof.stage2_sumcheck_proof);
    stages.push(StageTrace {
        num_rounds: s2.challenges.len(),
        poly_degree: degree2,
        round_poly_coeffs: coeffs2,
        evals: vec![],
    });

    // Stage 3
    let s3 = verifier.verify_stage3().expect("stage 3");
    let coeffs3 = extract_clear_rounds(
        &verifier.proof.stage3_sumcheck_proof,
        s3.initial_claim,
        &s3.challenges,
    );
    let degree3 = extract_clear_degree(&verifier.proof.stage3_sumcheck_proof);
    stages.push(StageTrace {
        num_rounds: s3.challenges.len(),
        poly_degree: degree3,
        round_poly_coeffs: coeffs3,
        evals: vec![],
    });

    // Stage 4
    let s4 = verifier.verify_stage4().expect("stage 4");
    let coeffs4 = extract_clear_rounds(
        &verifier.proof.stage4_sumcheck_proof,
        s4.initial_claim,
        &s4.challenges,
    );
    let degree4 = extract_clear_degree(&verifier.proof.stage4_sumcheck_proof);
    stages.push(StageTrace {
        num_rounds: s4.challenges.len(),
        poly_degree: degree4,
        round_poly_coeffs: coeffs4,
        evals: vec![],
    });

    // Stage 5
    let s5 = verifier.verify_stage5().expect("stage 5");
    let coeffs5 = extract_clear_rounds(
        &verifier.proof.stage5_sumcheck_proof,
        s5.initial_claim,
        &s5.challenges,
    );
    let degree5 = extract_clear_degree(&verifier.proof.stage5_sumcheck_proof);
    stages.push(StageTrace {
        num_rounds: s5.challenges.len(),
        poly_degree: degree5,
        round_poly_coeffs: coeffs5,
        evals: vec![],
    });

    // Stage 6
    let s6 = verifier.verify_stage6().expect("stage 6");
    let coeffs6 = extract_clear_rounds(
        &verifier.proof.stage6_sumcheck_proof,
        s6.initial_claim,
        &s6.challenges,
    );
    let degree6 = extract_clear_degree(&verifier.proof.stage6_sumcheck_proof);
    stages.push(StageTrace {
        num_rounds: s6.challenges.len(),
        poly_degree: degree6,
        round_poly_coeffs: coeffs6,
        evals: vec![],
    });

    // Stage 7
    let s7 = verifier.verify_stage7().expect("stage 7");
    let coeffs7 = extract_clear_rounds(
        &verifier.proof.stage7_sumcheck_proof,
        s7.initial_claim,
        &s7.challenges,
    );
    let degree7 = extract_clear_degree(&verifier.proof.stage7_sumcheck_proof);
    stages.push(StageTrace {
        num_rounds: s7.challenges.len(),
        poly_degree: degree7,
        round_poly_coeffs: coeffs7,
        evals: vec![],
    });

    ProtocolTrace { stages }
}

#[test]
fn jolt_core_mock_transcript_proves() {
    let trace = run_jolt_core();
    assert_eq!(trace.stages.len(), 7);
    for (i, stage) in trace.stages.iter().enumerate() {
        eprintln!(
            "Stage {}: {} rounds, degree {}",
            i + 1,
            stage.num_rounds,
            stage.poly_degree,
        );
        assert!(stage.num_rounds > 0);
        assert!(stage.poly_degree > 0);
        assert_eq!(stage.round_poly_coeffs.len(), stage.num_rounds);
    }
}

// ═══════════════════════════════════════════════════════════════════
// jolt-zkvm side (new modular pipeline)
// ═══════════════════════════════════════════════════════════════════

use common::constants::RAM_START_ADDRESS;
use jolt_compiler::module::Module;
use jolt_compute::link;
use jolt_cpu::CpuBackend;
use jolt_host::{build_r1cs_witness, cycle_to_input, BytecodePreprocessing};
use jolt_openings::mock::MockCommitmentScheme;
use jolt_r1cs::{R1csKey, R1csProvider};
use jolt_verifier::config::{OneHotConfig, ReadWriteConfig};
use jolt_verifier::ProverConfig;
use jolt_witness::{CycleInput, PolynomialConfig, PolynomialId, Polynomials};

type NewFr = jolt_field::Fr;

/// Convert jolt_field::Fr → ark_bn254::Fr (repr(transparent) newtype).
fn to_ark(f: NewFr) -> Fr {
    f.into()
}

/// Map Module poly names to jolt-witness PolynomialId.
///
/// Covers all polynomials needed for stages 1–2. Panics on names
/// that only appear in later stages (acceptable for incremental testing).
fn name_to_poly_id(name: &str) -> PolynomialId {
    match name {
        // Committed
        "RdInc" => PolynomialId::RdInc,
        "RamInc" => PolynomialId::RamInc,
        "UntrustedAdvice" => PolynomialId::UntrustedAdvice,
        "TrustedAdvice" => PolynomialId::TrustedAdvice,
        // Spartan internals
        "Az" => PolynomialId::Az,
        "Bz" => PolynomialId::Bz,
        "SpartanEqTable" => PolynomialId::SpartanEq,
        "ProductLeft" => PolynomialId::ProductLeft,
        "ProductRight" => PolynomialId::ProductRight,
        "OuterUniskipEval" => PolynomialId::OuterUniskipEval,
        "ProductUniskipEval" => PolynomialId::ProductUniskipEval,
        // R1CS inputs
        "LeftInstructionInput" => PolynomialId::LeftInstructionInput,
        "RightInstructionInput" => PolynomialId::RightInstructionInput,
        "Product" => PolynomialId::Product,
        "ShouldBranch" => PolynomialId::ShouldBranch,
        "PC" => PolynomialId::ExpandedPc,
        "UnexpandedPC" => PolynomialId::UnexpandedPc,
        "Imm" => PolynomialId::Imm,
        "RamAddress" => PolynomialId::RamAddress,
        "Rs1Value" => PolynomialId::Rs1Value,
        "Rs2Value" => PolynomialId::Rs2Value,
        "RdWriteValue" => PolynomialId::RdWriteValue,
        "RamReadValue" => PolynomialId::RamReadValue,
        "RamWriteValue" => PolynomialId::RamWriteValue,
        "LeftLookupOperand" => PolynomialId::LeftLookupOperand,
        "RightLookupOperand" => PolynomialId::RightLookupOperand,
        "NextUnexpandedPC" => PolynomialId::NextUnexpandedPc,
        "NextPC" => PolynomialId::NextPc,
        "NextIsVirtual" => PolynomialId::NextIsVirtual,
        "NextIsFirstInSequence" => PolynomialId::NextIsFirstInSequence,
        "LookupOutput" => PolynomialId::LookupOutput,
        "ShouldJump" => PolynomialId::ShouldJump,
        "NextIsNoop" => PolynomialId::NextIsNoop,
        // OpFlags (14 entries, index matches CircuitFlags enum order)
        "OpFlag_AddOperands" => PolynomialId::OpFlag(0),
        "OpFlag_SubtractOperands" => PolynomialId::OpFlag(1),
        "OpFlag_MultiplyOperands" => PolynomialId::OpFlag(2),
        "OpFlag_Load" => PolynomialId::OpFlag(3),
        "OpFlag_Store" => PolynomialId::OpFlag(4),
        "OpFlag_Jump" => PolynomialId::OpFlag(5),
        "OpFlag_WriteLookupOutputToRD" => PolynomialId::OpFlag(6),
        "OpFlag_VirtualInstruction" => PolynomialId::OpFlag(7),
        "OpFlag_Assert" => PolynomialId::OpFlag(8),
        "OpFlag_DoNotUpdateUnexpandedPC" => PolynomialId::OpFlag(9),
        "OpFlag_Advice" => PolynomialId::OpFlag(10),
        "OpFlag_IsCompressed" => PolynomialId::OpFlag(11),
        "OpFlag_IsFirstInSequence" => PolynomialId::OpFlag(12),
        "OpFlag_IsLastInSequence" => PolynomialId::OpFlag(13),
        // Instruction flags
        "InstFlag_LeftOperandIsPC" => PolynomialId::LeftIsPc,
        "InstFlag_RightOperandIsImm" => PolynomialId::RightIsImm,
        "InstFlag_LeftOperandIsRs1Value" => PolynomialId::LeftIsRs1,
        "InstFlag_RightOperandIsRs2Value" => PolynomialId::RightIsRs2,
        "InstFlag_Branch" => PolynomialId::BranchFlag,
        // Registers
        "RdWA" => PolynomialId::RdWa,
        "RegRaRs1" => PolynomialId::Rs1Ra,
        "RegRaRs2" => PolynomialId::Rs2Ra,
        "RegVal" => PolynomialId::RegistersVal,
        // RAM
        "RamCombinedRa" => PolynomialId::RamCombinedRa,
        "RamVal" => PolynomialId::RamVal,
        "RamValFinal" => PolynomialId::RamValFinal,
        "HammingWeight" => PolynomialId::HammingWeight,
        // RAF
        "RamRafRa" => PolynomialId::RamRafRa,
        "InstructionRafFlag" => PolynomialId::InstructionRafFlag,
        // Public / preprocessed
        "IoMask" => PolynomialId::IoMask,
        "ValIo" => PolynomialId::ValIo,
        "RamUnmap" => PolynomialId::RamUnmap,
        // Parametric families
        s if s.starts_with("InstructionRa_") => {
            PolynomialId::InstructionRa(s["InstructionRa_".len()..].parse().unwrap())
        }
        s if s.starts_with("RamRa_") => {
            PolynomialId::RamRa(s["RamRa_".len()..].parse().unwrap())
        }
        s if s.starts_with("BytecodeRa_") => {
            PolynomialId::BytecodeRa(s["BytecodeRa_".len()..].parse().unwrap())
        }
        s if s.starts_with("LookupTableFlag_") => {
            PolynomialId::LookupTableFlag(s["LookupTableFlag_".len()..].parse().unwrap())
        }
        s if s.starts_with("BcReadVal_") => {
            PolynomialId::BytecodeReadRafVal(s["BcReadVal_".len()..].parse().unwrap())
        }
        // Polynomials only used in stages 3+ (not yet in Module).
        // Panicking is acceptable for incremental testing.
        other => panic!("unmapped Module poly name: {other}"),
    }
}

/// Build a Module<PolynomialId> for the given trace parameters.
///
/// Calls the jolt_core_module example as a subprocess to generate the
/// canonical reference Module (as a .jolt binary), then deserializes
/// and remaps from usize indices to PolynomialId.
fn build_module(log_t: usize, log_k_bytecode: usize, log_k_ram: usize) -> Module<PolynomialId> {
    use std::process::Command;

    let tmp_path = format!(
        "/tmp/jolt_equiv_module_{log_t}_{log_k_bytecode}_{log_k_ram}.jolt"
    );

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
        "jolt_core_module example failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let bytes = std::fs::read(&tmp_path).expect("failed to read Module binary");
    eprintln!("Module binary: {} bytes, first 40 hex: {:02x?}", bytes.len(), &bytes[..40.min(bytes.len())]);
    let module_usize: Module = Module::from_bytes(&bytes);

    // Build name→PolynomialId mapping from the poly table
    let names: Vec<String> = module_usize.polys.iter().map(|p| p.name.clone()).collect();
    module_usize.remap(|idx: usize| name_to_poly_id(&names[idx]))
}

/// Run jolt-zkvm's muldiv prover with MockTranscript.
///
/// Produces a ProtocolTrace<Fr> (ark_bn254::Fr) by converting from
/// jolt_field::Fr (which is a repr(transparent) newtype wrapper).
fn run_jolt_zkvm() -> ProtocolTrace<Fr> {
    // 1. Run muldiv trace via jolt-core host (same Program + inputs as run_jolt_core).
    //    jolt-core's decode returns entry_address which jolt-host's does not.
    let mut program = host::Program::new("muldiv-guest");
    let (bytecode_raw, _, _, e_entry) = program.decode();
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
    let (_, trace, _, io_device) = program.trace(&inputs, &[], &[]);

    let bytecode = BytecodePreprocessing::preprocess(bytecode_raw, e_entry);
    let memory_layout = &io_device.memory_layout;

    // 2. Derive trace parameters
    let trace_length = trace.len().next_power_of_two();
    let log_t = trace_length.trailing_zeros() as usize;
    let bytecode_k = bytecode.code_size;
    let log_k_bytecode = bytecode_k.trailing_zeros() as usize;
    let ram_k = 1usize << 20; // TODO: derive from memory_layout
    let log_k_ram = ram_k.trailing_zeros() as usize;

    eprintln!(
        "jolt-zkvm params: log_t={log_t}, log_k_bytecode={log_k_bytecode}, log_k_ram={log_k_ram}"
    );

    // 3. Build Module and link to CPU backend
    let module = build_module(log_t, log_k_bytecode, log_k_ram);
    let backend = CpuBackend;
    let executable = link::<PolynomialId, CpuBackend, NewFr>(module, &backend);

    // 4. Build witness polynomials
    let one_hot = OneHotConfig::new(log_t);
    let poly_config = PolynomialConfig::new(
        one_hot.log_k_chunk as usize,
        128, // LOG_K_INSTRUCTION
        log_k_bytecode,
        log_k_ram,
    );
    let cycle_inputs: Vec<CycleInput> = trace
        .iter()
        .map(|c| cycle_to_input(c, &bytecode, memory_layout))
        .collect();
    let mut polys = Polynomials::<NewFr>::new(poly_config);
    polys.push(&cycle_inputs);
    polys.finish();

    // 5. Build R1CS
    let matrices = jolt_r1cs::constraints::rv64::rv64_constraints::<NewFr>();
    let r1cs_key = R1csKey::new(matrices, trace_length);
    let r1cs_witness =
        build_r1cs_witness::<_, NewFr>(&trace, &bytecode, r1cs_key.num_vars_padded);
    let r1cs = R1csProvider::new(&r1cs_key, &r1cs_witness);

    // 6. Build ProverConfig
    let config = ProverConfig {
        trace_length,
        ram_k,
        bytecode_k,
        one_hot_config: one_hot,
        rw_config: ReadWriteConfig::new(log_t, log_k_ram),
        memory_start: RAM_START_ADDRESS,
        memory_end: RAM_START_ADDRESS + ram_k as u64,
        entry_address: e_entry,
        io_hash: [0u8; 32],
    };

    // 7. Prove with MockTranscript + MockCommitmentScheme
    let pcs_setup = ();
    let mut transcript = jolt_transcript::MockTranscript::<NewFr>::default();
    let mut buffers = jolt_zkvm::buffers::ProverBuffers::new(&mut polys, r1cs);

    let proof = jolt_zkvm::prove::prove_with_buffers::<
        PolynomialId,
        CpuBackend,
        NewFr,
        _,
        MockCommitmentScheme<NewFr>,
    >(
        &executable,
        &mut buffers,
        &backend,
        &pcs_setup,
        &mut transcript,
        config,
    );

    // 8. Extract stage proofs → ProtocolTrace<Fr>
    let stages = proof
        .stage_proofs
        .iter()
        .map(|sp| {
            let round_polys = &sp.round_polys.round_polynomials;
            StageTrace {
                num_rounds: round_polys.len(),
                poly_degree: round_polys.first().map_or(0, |p| p.coefficients().len().saturating_sub(1)),
                round_poly_coeffs: round_polys
                    .iter()
                    .map(|p| p.coefficients().iter().copied().map(to_ark).collect())
                    .collect(),
                evals: sp.evals.iter().copied().map(to_ark).collect(),
            }
        })
        .collect();

    ProtocolTrace { stages }
}

#[test]
#[ignore = "requires full pipeline wiring — run with --ignored"]
fn jolt_zkvm_mock_transcript_proves() {
    let trace = run_jolt_zkvm();
    eprintln!("jolt-zkvm produced {} stages", trace.stages.len());
    for (i, stage) in trace.stages.iter().enumerate() {
        eprintln!(
            "Stage {}: {} rounds, degree {}",
            i + 1,
            stage.num_rounds,
            stage.poly_degree,
        );
    }
    assert!(trace.stages.len() >= 2, "expected at least 2 stages");
}

#[test]
#[ignore = "requires full pipeline wiring — run with --ignored"]
fn cross_system_stages_match() {
    let core_trace = run_jolt_core();
    let zkvm_trace = run_jolt_zkvm();

    let n = core_trace.stages.len().min(zkvm_trace.stages.len());
    eprintln!(
        "comparing {n} stages (core={}, zkvm={})",
        core_trace.stages.len(),
        zkvm_trace.stages.len()
    );

    for i in 0..n {
        let a = &core_trace.stages[i];
        let b = &zkvm_trace.stages[i];
        assert_eq!(a.num_rounds, b.num_rounds, "stage {i}: num_rounds");
        assert_eq!(a.poly_degree, b.poly_degree, "stage {i}: poly_degree");
        assert_eq!(
            a.round_poly_coeffs.len(),
            b.round_poly_coeffs.len(),
            "stage {i}: round count",
        );
        for (j, (pa, pb)) in a
            .round_poly_coeffs
            .iter()
            .zip(&b.round_poly_coeffs)
            .enumerate()
        {
            assert_eq!(pa, pb, "stage {i} round {j}: poly coeffs diverge");
        }
    }
}
