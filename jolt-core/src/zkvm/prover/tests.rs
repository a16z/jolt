use ark_bn254::Fr;
use serial_test::serial;

use crate::host;
use crate::poly::commitment::dory::{DoryGlobals, DoryLayout};
#[cfg(feature = "zk")]
use crate::poly::commitment::pedersen::PedersenGenerators;
use crate::poly::{
    commitment::{
        commitment_scheme::CommitmentScheme,
        dory::{DoryCommitmentScheme, DoryContext},
    },
    multilinear_polynomial::MultilinearPolynomial,
    opening_proof::{OpeningAccumulator, SumcheckId},
};
use crate::zkvm::claim_reductions::AdviceKind;
use crate::zkvm::verifier::JoltSharedPreprocessing;
use crate::zkvm::witness::CommittedPolynomial;
use crate::zkvm::{
    prover::JoltProverPreprocessing,
    ram::populate_memory_states,
    verifier::{JoltVerifier, JoltVerifierPreprocessing},
    RV64IMACProver, RV64IMACVerifier,
};
#[cfg(feature = "zk")]
use crate::{curve::JoltCurve, field::JoltField};

#[cfg(feature = "zk")]
fn round_commitment_data<F: JoltField, C: JoltCurve, R: rand_core::RngCore>(
    gens: &PedersenGenerators<C>,
    stages: &[crate::subprotocols::blindfold::StageWitness<F>],
    rng: &mut R,
) -> (Vec<C::G1>, Vec<Vec<F>>, Vec<F>) {
    let mut commitments = Vec::new();
    let mut coeffs = Vec::new();
    let mut blindings = Vec::new();
    for stage in stages {
        for round in &stage.rounds {
            let blinding = F::random(rng);
            let commitment = gens.commit(&round.coeffs, &blinding);
            commitments.push(commitment);
            coeffs.push(round.coeffs.clone());
            blindings.push(blinding);
        }
    }
    (commitments, coeffs, blindings)
}

fn commit_trusted_advice_preprocessing_only(
    preprocessing: &JoltProverPreprocessing<Fr, DoryCommitmentScheme>,
    trusted_advice_bytes: &[u8],
) -> (
    <DoryCommitmentScheme as CommitmentScheme>::Commitment,
    <DoryCommitmentScheme as CommitmentScheme>::OpeningProofHint,
) {
    let max_trusted_advice_size = preprocessing.shared.memory_layout.max_trusted_advice_size;
    let mut trusted_advice_words = vec![0u64; (max_trusted_advice_size as usize) / 8];
    populate_memory_states(
        0,
        trusted_advice_bytes,
        Some(&mut trusted_advice_words),
        None,
    );

    let poly = MultilinearPolynomial::<Fr>::from(trusted_advice_words);
    let advice_len = poly.len().next_power_of_two().max(1);

    let _guard = DoryGlobals::initialize_context(1, advice_len, DoryContext::TrustedAdvice, None);
    let (commitment, hint) = {
        let _ctx = DoryGlobals::with_context(DoryContext::TrustedAdvice);
        DoryCommitmentScheme::commit(&poly, &preprocessing.generators)
    };
    (commitment, hint)
}

#[test]
#[serial]
fn fib_e2e_dory() {
    DoryGlobals::reset();
    let mut program = host::Program::new("fibonacci-guest");
    let inputs = postcard::to_stdvec(&100u32).unwrap();
    let (bytecode, init_memory_state, _) = program.decode();
    let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);
    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode.clone(),
        io_device.memory_layout.clone(),
        init_memory_state,
        1 << 16,
    );

    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
    let elf_contents_opt = program.get_elf_contents();
    let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
    let prover = RV64IMACProver::gen_from_elf(
        &prover_preprocessing,
        elf_contents,
        &inputs,
        &[],
        &[],
        None,
        None,
        None,
    );
    let io_device = prover.program_io.clone();
    let (jolt_proof, debug_info) = prover.prove();

    let verifier_preprocessing = JoltVerifierPreprocessing::new(
        shared_preprocessing,
        prover_preprocessing.generators.to_verifier_setup(),
    );
    let verifier = RV64IMACVerifier::new(
        &verifier_preprocessing,
        jolt_proof,
        io_device,
        None,
        debug_info,
    )
    .expect("Failed to create verifier");
    verifier.verify().expect("Failed to verify proof");
}

#[test]
#[serial]
fn small_trace_e2e_dory() {
    DoryGlobals::reset();
    let mut program = host::Program::new("fibonacci-guest");
    let inputs = postcard::to_stdvec(&5u32).unwrap();
    let (bytecode, init_memory_state, _) = program.decode();
    let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode.clone(),
        io_device.memory_layout.clone(),
        init_memory_state,
        8192,
    );

    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
    let elf_contents_opt = program.get_elf_contents();
    let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
    let log_chunk = 13;
    let prover = RV64IMACProver::gen_from_elf(
        &prover_preprocessing,
        elf_contents,
        &inputs,
        &[],
        &[],
        None,
        None,
        None,
    );

    assert!(
        prover.padded_trace_len <= (1 << log_chunk),
        "Test requires T <= chunk_size ({}), got T = {}",
        1 << log_chunk,
        prover.padded_trace_len
    );

    let io_device = prover.program_io.clone();
    let (jolt_proof, debug_info) = prover.prove();

    let verifier_preprocessing = JoltVerifierPreprocessing::new(
        prover_preprocessing.shared.clone(),
        prover_preprocessing.generators.to_verifier_setup(),
    );
    let verifier = RV64IMACVerifier::new(
        &verifier_preprocessing,
        jolt_proof,
        io_device,
        None,
        debug_info,
    )
    .expect("Failed to create verifier");
    verifier.verify().expect("Failed to verify proof");
}

#[test]
#[serial]
fn sha3_e2e_dory() {
    DoryGlobals::reset();
    #[cfg(feature = "host")]
    use jolt_inlines_keccak256 as _;

    let mut program = host::Program::new("sha3-guest");
    let (bytecode, init_memory_state, _) = program.decode();
    let inputs = postcard::to_stdvec(&[5u8; 32]).unwrap();
    let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode.clone(),
        io_device.memory_layout.clone(),
        init_memory_state,
        1 << 16,
    );

    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
    let elf_contents_opt = program.get_elf_contents();
    let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
    let prover = RV64IMACProver::gen_from_elf(
        &prover_preprocessing,
        elf_contents,
        &inputs,
        &[],
        &[],
        None,
        None,
        None,
    );
    let io_device = prover.program_io.clone();
    let (jolt_proof, debug_info) = prover.prove();

    let verifier_preprocessing = JoltVerifierPreprocessing::new(
        prover_preprocessing.shared.clone(),
        prover_preprocessing.generators.to_verifier_setup(),
    );
    let verifier = RV64IMACVerifier::new(
        &verifier_preprocessing,
        jolt_proof,
        io_device.clone(),
        None,
        debug_info,
    )
    .expect("Failed to create verifier");
    verifier.verify().expect("Failed to verify proof");
    assert_eq!(
        io_device.inputs, inputs,
        "Inputs mismatch: expected {:?}, got {:?}",
        inputs, io_device.inputs
    );
    let expected_output = &[
        0xd0, 0x3, 0x5c, 0x96, 0x86, 0x6e, 0xe2, 0x2e, 0x81, 0xf5, 0xc4, 0xef, 0xbd, 0x88, 0x33,
        0xc1, 0x7e, 0xa1, 0x61, 0x10, 0x81, 0xfc, 0xd7, 0xa3, 0xdd, 0xce, 0xce, 0x7f, 0x44, 0x72,
        0x4, 0x66,
    ];
    assert_eq!(io_device.outputs, expected_output, "Outputs mismatch",);
}

#[test]
#[serial]
fn sha2_e2e_dory() {
    DoryGlobals::reset();
    #[cfg(feature = "host")]
    use jolt_inlines_sha2 as _;
    let mut program = host::Program::new("sha2-guest");
    let (bytecode, init_memory_state, _) = program.decode();
    let inputs = postcard::to_stdvec(&[5u8; 32]).unwrap();
    let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode.clone(),
        io_device.memory_layout.clone(),
        init_memory_state,
        1 << 16,
    );

    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
    let elf_contents_opt = program.get_elf_contents();
    let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
    let prover = RV64IMACProver::gen_from_elf(
        &prover_preprocessing,
        elf_contents,
        &inputs,
        &[],
        &[],
        None,
        None,
        None,
    );
    let io_device = prover.program_io.clone();
    let (jolt_proof, debug_info) = prover.prove();

    let verifier_preprocessing = JoltVerifierPreprocessing::new(
        prover_preprocessing.shared.clone(),
        prover_preprocessing.generators.to_verifier_setup(),
    );
    let verifier = RV64IMACVerifier::new(
        &verifier_preprocessing,
        jolt_proof,
        io_device.clone(),
        None,
        debug_info,
    )
    .expect("Failed to create verifier");
    verifier.verify().expect("Failed to verify proof");
    let expected_output = &[
        0x28, 0x9b, 0xdf, 0x82, 0x9b, 0x4a, 0x30, 0x26, 0x7, 0x9a, 0x3e, 0xa0, 0x89, 0x73, 0xb1,
        0x97, 0x2d, 0x12, 0x4e, 0x7e, 0xaf, 0x22, 0x33, 0xc6, 0x3, 0x14, 0x3d, 0xc6, 0x3b, 0x50,
        0xd2, 0x57,
    ];
    assert_eq!(
        io_device.outputs, expected_output,
        "Outputs mismatch: expected {:?}, got {:?}",
        expected_output, io_device.outputs
    );
}

#[test]
#[serial]
fn sha2_e2e_dory_with_unused_advice() {
    DoryGlobals::reset();
    let mut program = host::Program::new("sha2-guest");
    let (bytecode, init_memory_state, _) = program.decode();
    let inputs = postcard::to_stdvec(&[5u8; 32]).unwrap();
    let trusted_advice = postcard::to_stdvec(&[7u8; 32]).unwrap();
    let untrusted_advice = postcard::to_stdvec(&[9u8; 32]).unwrap();

    let (_, _, _, io_device) = program.trace(&inputs, &untrusted_advice, &trusted_advice);

    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode.clone(),
        io_device.memory_layout.clone(),
        init_memory_state,
        1 << 16,
    );
    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
    let elf_contents = program.get_elf_contents().expect("elf contents is None");

    let (trusted_commitment, trusted_hint) =
        commit_trusted_advice_preprocessing_only(&prover_preprocessing, &trusted_advice);

    let prover = RV64IMACProver::gen_from_elf(
        &prover_preprocessing,
        &elf_contents,
        &inputs,
        &untrusted_advice,
        &trusted_advice,
        Some(trusted_commitment),
        Some(trusted_hint),
        None,
    );
    let io_device = prover.program_io.clone();
    let (jolt_proof, debug_info) = prover.prove();

    let verifier_preprocessing = JoltVerifierPreprocessing::from(&prover_preprocessing);
    RV64IMACVerifier::new(
        &verifier_preprocessing,
        jolt_proof,
        io_device.clone(),
        Some(trusted_commitment),
        debug_info,
    )
    .expect("Failed to create verifier")
    .verify()
    .expect("Failed to verify proof");

    let expected_output = &[
        0x28, 0x9b, 0xdf, 0x82, 0x9b, 0x4a, 0x30, 0x26, 0x7, 0x9a, 0x3e, 0xa0, 0x89, 0x73, 0xb1,
        0x97, 0x2d, 0x12, 0x4e, 0x7e, 0xaf, 0x22, 0x33, 0xc6, 0x3, 0x14, 0x3d, 0xc6, 0x3b, 0x50,
        0xd2, 0x57,
    ];
    assert_eq!(io_device.outputs, expected_output);
}

#[test]
#[serial]
fn max_advice_with_small_trace() {
    DoryGlobals::reset();
    let mut program = host::Program::new("fibonacci-guest");
    let inputs = postcard::to_stdvec(&5u32).unwrap();
    let trusted_advice = vec![7u8; 4096];
    let untrusted_advice = vec![9u8; 4096];

    let (bytecode, init_memory_state, _) = program.decode();
    let (lazy_trace, trace, final_memory_state, io_device) =
        program.trace(&inputs, &untrusted_advice, &trusted_advice);

    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode.clone(),
        io_device.memory_layout.clone(),
        init_memory_state,
        4096,
    );
    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
    tracing::info!(
        "preprocessing.memory_layout.max_trusted_advice_size: {}",
        shared_preprocessing.memory_layout.max_trusted_advice_size
    );

    let (trusted_commitment, trusted_hint) =
        commit_trusted_advice_preprocessing_only(&prover_preprocessing, &trusted_advice);

    let prover = RV64IMACProver::gen_from_trace(
        &prover_preprocessing,
        lazy_trace,
        trace,
        io_device,
        Some(trusted_commitment),
        Some(trusted_hint),
        final_memory_state,
    );

    assert!(prover.unpadded_trace_len < 8192);
    assert_eq!(prover.padded_trace_len, 4096);

    let io_device = prover.program_io.clone();
    let (jolt_proof, debug_info) = prover.prove();

    let verifier_preprocessing = JoltVerifierPreprocessing::from(&prover_preprocessing);
    RV64IMACVerifier::new(
        &verifier_preprocessing,
        jolt_proof,
        io_device,
        Some(trusted_commitment),
        debug_info,
    )
    .expect("Failed to create verifier")
    .verify()
    .expect("Verification failed");
}

#[test]
#[serial]
fn advice_e2e_dory() {
    DoryGlobals::reset();
    let mut program = host::Program::new("merkle-tree-guest");
    let (bytecode, init_memory_state, _) = program.decode();

    let inputs = postcard::to_stdvec(&[5u8; 32].as_slice()).unwrap();
    let untrusted_advice = postcard::to_stdvec(&[8u8; 32]).unwrap();
    let mut trusted_advice = postcard::to_stdvec(&[6u8; 32]).unwrap();
    trusted_advice.extend(postcard::to_stdvec(&[7u8; 32]).unwrap());

    let (_, _, _, io_device) = program.trace(&inputs, &untrusted_advice, &trusted_advice);
    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode.clone(),
        io_device.memory_layout.clone(),
        init_memory_state,
        1 << 16,
    );
    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
    let elf_contents = program.get_elf_contents().expect("elf contents is None");

    let (trusted_commitment, trusted_hint) =
        commit_trusted_advice_preprocessing_only(&prover_preprocessing, &trusted_advice);

    let prover = RV64IMACProver::gen_from_elf(
        &prover_preprocessing,
        &elf_contents,
        &inputs,
        &untrusted_advice,
        &trusted_advice,
        Some(trusted_commitment),
        Some(trusted_hint),
        None,
    );
    let io_device = prover.program_io.clone();
    let (jolt_proof, debug_info) = prover.prove();

    let verifier_preprocessing = JoltVerifierPreprocessing::from(&prover_preprocessing);
    RV64IMACVerifier::new(
        &verifier_preprocessing,
        jolt_proof,
        io_device.clone(),
        Some(trusted_commitment),
        debug_info,
    )
    .expect("Failed to create verifier")
    .verify()
    .expect("Verification failed");

    let expected_output = &[
        0xb4, 0x37, 0x0f, 0x3a, 0xb, 0x3d, 0x38, 0xa8, 0x7a, 0x6c, 0x4c, 0x46, 0x9, 0xe7, 0x83,
        0xb3, 0xcc, 0xb7, 0x1c, 0x30, 0x1f, 0xf8, 0x54, 0xd, 0xf7, 0xdd, 0xc8, 0x42, 0x32, 0xbb,
        0x16, 0xd7,
    ];
    assert_eq!(io_device.outputs, expected_output);
}

#[test]
#[serial]
fn advice_opening_point_derives_from_unified_point() {
    DoryGlobals::reset();
    let mut program = host::Program::new("fibonacci-guest");
    let inputs = postcard::to_stdvec(&5u32).unwrap();
    let trusted_advice = postcard::to_stdvec(&[7u8; 32]).unwrap();
    let untrusted_advice = postcard::to_stdvec(&[9u8; 32]).unwrap();

    let (bytecode, init_memory_state, _) = program.decode();
    let (lazy_trace, trace, final_memory_state, io_device) =
        program.trace(&inputs, &untrusted_advice, &trusted_advice);

    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode.clone(),
        io_device.memory_layout.clone(),
        init_memory_state,
        1 << 16,
    );
    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
    let (trusted_commitment, trusted_hint) =
        commit_trusted_advice_preprocessing_only(&prover_preprocessing, &trusted_advice);

    let prover = RV64IMACProver::gen_from_trace(
        &prover_preprocessing,
        lazy_trace,
        trace,
        io_device,
        Some(trusted_commitment),
        Some(trusted_hint),
        final_memory_state,
    );

    assert_eq!(prover.padded_trace_len, 4096, "test expects small trace");

    let io_device = prover.program_io.clone();
    let (jolt_proof, debug_info) = prover.prove();
    let debug_info = debug_info.expect("expected debug_info in tests");

    let (opening_point, _) = debug_info
        .opening_accumulator
        .get_committed_polynomial_opening(
            CommittedPolynomial::InstructionRa(0),
            SumcheckId::HammingWeightClaimReduction,
        );
    let mut point_dory_le = opening_point.r.clone();
    point_dory_le.reverse();

    let total_vars = point_dory_le.len();
    let (sigma_main, _nu_main) = DoryGlobals::balanced_sigma_nu(total_vars);
    let (sigma_a, nu_a) = DoryGlobals::advice_sigma_nu_from_max_bytes(
        prover_preprocessing
            .shared
            .memory_layout
            .max_trusted_advice_size as usize,
    );

    let mut expected_advice_le: Vec<_> = point_dory_le[0..sigma_a].to_vec();
    expected_advice_le.extend_from_slice(&point_dory_le[sigma_main..sigma_main + nu_a]);

    for (name, kind) in [
        ("trusted", AdviceKind::Trusted),
        ("untrusted", AdviceKind::Untrusted),
    ] {
        let get_fn = debug_info
            .opening_accumulator
            .get_advice_opening(kind, SumcheckId::AdviceClaimReduction);
        assert!(
            get_fn.is_some(),
            "{name} advice opening missing for AdviceClaimReductionPhase2"
        );
        let (point_be, _) = get_fn.unwrap();
        let mut point_le = point_be.r.clone();
        point_le.reverse();
        assert_eq!(point_le, expected_advice_le, "{name} advice point mismatch");
    }

    let verifier_preprocessing = JoltVerifierPreprocessing::from(&prover_preprocessing);
    RV64IMACVerifier::new(
        &verifier_preprocessing,
        jolt_proof,
        io_device,
        Some(trusted_commitment),
        Some(debug_info),
    )
    .expect("Failed to create verifier")
    .verify()
    .expect("Verification failed");
}

#[test]
#[serial]
fn memory_ops_e2e_dory() {
    DoryGlobals::reset();
    let mut program = host::Program::new("memory-ops-guest");
    let (bytecode, init_memory_state, _) = program.decode();
    let (_, _, _, io_device) = program.trace(&[], &[], &[]);

    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode.clone(),
        io_device.memory_layout.clone(),
        init_memory_state,
        1 << 16,
    );

    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
    let elf_contents_opt = program.get_elf_contents();
    let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
    let prover = RV64IMACProver::gen_from_elf(
        &prover_preprocessing,
        elf_contents,
        &[],
        &[],
        &[],
        None,
        None,
        None,
    );
    let io_device = prover.program_io.clone();
    let (jolt_proof, debug_info) = prover.prove();

    let verifier_preprocessing = JoltVerifierPreprocessing::new(
        prover_preprocessing.shared.clone(),
        prover_preprocessing.generators.to_verifier_setup(),
    );
    let verifier = RV64IMACVerifier::new(
        &verifier_preprocessing,
        jolt_proof,
        io_device,
        None,
        debug_info,
    )
    .expect("Failed to create verifier");
    verifier.verify().expect("Failed to verify proof");
}

#[test]
#[serial]
fn btreemap_e2e_dory() {
    DoryGlobals::reset();
    let mut program = host::Program::new("btreemap-guest");
    let (bytecode, init_memory_state, _) = program.decode();
    let inputs = postcard::to_stdvec(&50u32).unwrap();
    let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode.clone(),
        io_device.memory_layout.clone(),
        init_memory_state,
        1 << 16,
    );

    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
    let elf_contents_opt = program.get_elf_contents();
    let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
    let prover = RV64IMACProver::gen_from_elf(
        &prover_preprocessing,
        elf_contents,
        &inputs,
        &[],
        &[],
        None,
        None,
        None,
    );
    let io_device = prover.program_io.clone();
    let (jolt_proof, debug_info) = prover.prove();

    let verifier_preprocessing = JoltVerifierPreprocessing::new(
        prover_preprocessing.shared.clone(),
        prover_preprocessing.generators.to_verifier_setup(),
    );
    let verifier = RV64IMACVerifier::new(
        &verifier_preprocessing,
        jolt_proof,
        io_device,
        None,
        debug_info,
    )
    .expect("Failed to create verifier");
    verifier.verify().expect("Failed to verify proof");
}

#[test]
#[serial]
fn muldiv_e2e_dory() {
    DoryGlobals::reset();
    let mut program = host::Program::new("muldiv-guest");
    let (bytecode, init_memory_state, _) = program.decode();
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
    let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode.clone(),
        io_device.memory_layout.clone(),
        init_memory_state,
        1 << 16,
    );

    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
    let elf_contents_opt = program.get_elf_contents();
    let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
    let prover = RV64IMACProver::gen_from_elf(
        &prover_preprocessing,
        elf_contents,
        &inputs,
        &[],
        &[],
        None,
        None,
        None,
    );
    let io_device = prover.program_io.clone();
    let (jolt_proof, debug_info) = prover.prove();

    let verifier_preprocessing = JoltVerifierPreprocessing::new(
        prover_preprocessing.shared.clone(),
        prover_preprocessing.generators.to_verifier_setup(),
    );
    let verifier = RV64IMACVerifier::new(
        &verifier_preprocessing,
        jolt_proof,
        io_device,
        None,
        debug_info,
    )
    .expect("Failed to create verifier");
    verifier.verify().expect("Failed to verify proof");
}

#[cfg(feature = "zk")]
#[test]
#[serial]
fn blindfold_r1cs_satisfaction() {
    use crate::curve::Bn254Curve;
    use crate::subprotocols::blindfold::{
        BakedPublicInputs, BlindFoldWitness, RoundWitness, StageConfig, StageWitness,
        VerifierR1CSBuilder,
    };
    use crate::subprotocols::sumcheck::SumcheckInstanceProof;
    use crate::transcripts::{KeccakTranscript, Transcript};
    use crate::zkvm::verifier::JoltSharedPreprocessing;
    fn process_stage<ProofTranscript: Transcript>(
        _stage_name: &str,
        proof: &SumcheckInstanceProof<Fr, Bn254Curve, ProofTranscript>,
        transcript: &mut KeccakTranscript,
    ) -> Vec<(RoundWitness<Fr>, usize)> {
        match proof {
            SumcheckInstanceProof::Standard(std_proof) => {
                let compressed_polys = &std_proof.compressed_polys;
                let num_rounds = compressed_polys.len();

                if num_rounds == 0 {
                    return vec![];
                }

                let mut rounds = Vec::with_capacity(num_rounds);

                for compressed_poly in compressed_polys.iter() {
                    transcript.append_scalars(
                        b"sumcheck_poly",
                        &compressed_poly.coeffs_except_linear_term,
                    );
                    let challenge: Fr = transcript.challenge_scalar_optimized::<Fr>().into();

                    let compressed = &compressed_poly.coeffs_except_linear_term;
                    let degree = compressed.len();

                    let c0 = compressed[0];
                    let sum_higher_coeffs: Fr = compressed[1..].iter().copied().sum();

                    let claimed_sum = Fr::from(12345u64);
                    let c1 = claimed_sum - c0 - c0 - sum_higher_coeffs;

                    let mut coeffs = vec![c0, c1];
                    coeffs.extend_from_slice(&compressed[1..]);

                    let round_witness =
                        RoundWitness::with_claimed_sum(coeffs, challenge, claimed_sum);

                    rounds.push((round_witness, degree));
                }

                rounds
            }
            SumcheckInstanceProof::Zk(zk_proof) => {
                let num_rounds = zk_proof.round_commitments.len();

                if num_rounds == 0 {
                    return vec![];
                }

                let mut rounds = Vec::with_capacity(num_rounds);

                for (round_idx, commitment) in zk_proof.round_commitments.iter().enumerate() {
                    transcript.append_point(b"sumcheck_commitment", commitment);
                    let challenge: Fr = transcript.challenge_scalar_optimized::<Fr>().into();

                    let degree = zk_proof.poly_degrees[round_idx];

                    let claimed_sum = Fr::from(12345u64);

                    let c0 = Fr::from(1u64);
                    let num_higher_coeffs = degree.saturating_sub(1);
                    let sum_higher_coeffs = Fr::from(num_higher_coeffs as u64);
                    let c1 = claimed_sum - c0 - c0 - sum_higher_coeffs;

                    let mut coeffs = vec![c0, c1];
                    for _ in 0..num_higher_coeffs {
                        coeffs.push(Fr::from(1u64));
                    }

                    let round_witness =
                        RoundWitness::with_claimed_sum(coeffs, challenge, claimed_sum);

                    rounds.push((round_witness, degree));
                }

                rounds
            }
        }
    }

    let mut program = host::Program::new("muldiv-guest");
    let (bytecode, init_memory_state, _) = program.decode();
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
    let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode.clone(),
        io_device.memory_layout.clone(),
        init_memory_state,
        1 << 16,
    );
    let preprocessing = JoltProverPreprocessing::new(shared_preprocessing);
    let elf_contents_opt = program.get_elf_contents();
    let elf_contents = elf_contents_opt.as_deref().expect("elf contents is None");
    let prover = RV64IMACProver::gen_from_elf(
        &preprocessing,
        elf_contents,
        &[50],
        &[],
        &[],
        None,
        None,
        None,
    );
    let (jolt_proof, _) = prover.prove();

    println!("\n=== BlindFold R1CS Satisfaction Test (All 7 Stages) ===\n");

    let stage_proofs: Vec<(&str, &SumcheckInstanceProof<Fr, Bn254Curve, _>)> = vec![
        ("Stage 1 (Spartan Outer)", &jolt_proof.stage1_sumcheck_proof),
        (
            "Stage 2 (Product Virtual)",
            &jolt_proof.stage2_sumcheck_proof,
        ),
        ("Stage 3 (Instruction)", &jolt_proof.stage3_sumcheck_proof),
        ("Stage 4 (Registers+RAM)", &jolt_proof.stage4_sumcheck_proof),
        ("Stage 5 (Value+Lookup)", &jolt_proof.stage5_sumcheck_proof),
        (
            "Stage 6 (OneHot+Hamming)",
            &jolt_proof.stage6_sumcheck_proof,
        ),
        (
            "Stage 7 (HammingWeight+ClaimReduction)",
            &jolt_proof.stage7_sumcheck_proof,
        ),
    ];

    let mut total_rounds = 0;
    let mut total_constraints = 0;

    for (stage_name, proof) in &stage_proofs {
        let mut stage_transcript = KeccakTranscript::new(b"BlindFoldStageTest");

        let rounds = process_stage(stage_name, proof, &mut stage_transcript);

        if rounds.is_empty() {
            println!("  {stage_name} - 0 rounds, skipping");
            continue;
        }

        let mut stage_rounds = 0;
        let mut stage_constraints = 0;

        for (round_witness, degree) in rounds {
            let config = StageConfig::new(1, degree);
            let initial_claim = round_witness.claimed_sum;
            let baked = BakedPublicInputs {
                challenges: vec![round_witness.challenge],
                initial_claims: vec![initial_claim],
                ..Default::default()
            };
            let builder = VerifierR1CSBuilder::<Fr>::new(&[config.clone()], &baked);
            let r1cs = builder.build();
            let stage_witness = StageWitness::new(vec![round_witness]);
            let witness = BlindFoldWitness::new(initial_claim, vec![stage_witness]);

            let z = witness.assign(&r1cs);
            match r1cs.check_satisfaction(&z) {
                Ok(()) => {
                    stage_rounds += 1;
                    stage_constraints += r1cs.num_constraints;
                }
                Err(row) => {
                    panic!(
                        "{} (degree {}) - constraint {} failed (out of {})",
                        stage_name, degree, row, r1cs.num_constraints
                    );
                }
            }
        }

        println!(
            "  {stage_name} - {stage_rounds} rounds, {stage_constraints} constraints - SATISFIED"
        );
        total_rounds += stage_rounds;
        total_constraints += stage_constraints;
    }

    println!("\n=== Summary ===");
    println!("Total rounds across all stages: {total_rounds}");
    println!("Total constraints across all stages: {total_constraints}");
    println!("All 6 stages satisfied!\n");

    assert!(total_rounds > 0, "Expected at least some sumcheck rounds");
    assert!(
        total_constraints > 0,
        "Expected at least some R1CS constraints"
    );
}

#[test]
#[serial]
#[should_panic]
fn truncated_trace() {
    let mut program = host::Program::new("fibonacci-guest");
    let (bytecode, init_memory_state, _) = program.decode();
    let inputs = postcard::to_stdvec(&9u8).unwrap();
    let (lazy_trace, mut trace, final_memory_state, mut program_io) =
        program.trace(&inputs, &[], &[]);
    trace.truncate(100);
    program_io.outputs[0] = 0;

    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode.clone(),
        program_io.memory_layout.clone(),
        init_memory_state,
        1 << 16,
    );

    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());

    let prover = RV64IMACProver::gen_from_trace(
        &prover_preprocessing,
        lazy_trace,
        trace,
        program_io.clone(),
        None,
        None,
        final_memory_state,
    );

    let (proof, _) = prover.prove();

    let verifier_preprocessing = JoltVerifierPreprocessing::new(
        prover_preprocessing.shared.clone(),
        prover_preprocessing.generators.to_verifier_setup(),
    );
    let verifier =
        RV64IMACVerifier::new(&verifier_preprocessing, proof, program_io, None, None).unwrap();
    verifier.verify().unwrap();
}

#[test]
#[serial]
#[should_panic]
fn malicious_trace() {
    let mut program = host::Program::new("fibonacci-guest");
    let inputs = postcard::to_stdvec(&1u8).unwrap();
    let (bytecode, init_memory_state, _) = program.decode();
    let (lazy_trace, trace, final_memory_state, mut program_io) = program.trace(&inputs, &[], &[]);

    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode.clone(),
        program_io.memory_layout.clone(),
        init_memory_state,
        1 << 16,
    );
    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());

    program_io.memory_layout.output_start = program_io.memory_layout.input_start;
    program_io.memory_layout.output_end = program_io.memory_layout.input_end;
    program_io.memory_layout.termination = program_io.memory_layout.input_start;

    let prover = RV64IMACProver::gen_from_trace(
        &prover_preprocessing,
        lazy_trace,
        trace,
        program_io.clone(),
        None,
        None,
        final_memory_state,
    );
    let (proof, _) = prover.prove();

    let verifier_preprocessing = JoltVerifierPreprocessing::new(
        prover_preprocessing.shared.clone(),
        prover_preprocessing.generators.to_verifier_setup(),
    );
    let verifier =
        JoltVerifier::new(&verifier_preprocessing, proof, program_io, None, None).unwrap();
    verifier.verify().unwrap();
}

#[cfg(feature = "zk")]
#[test]
#[serial]
fn blindfold_protocol_e2e() {
    use crate::curve::Bn254Curve;
    use crate::subprotocols::blindfold::{
        BakedPublicInputs, BlindFoldProver, BlindFoldVerifier, BlindFoldVerifierInput,
        BlindFoldWitness, RelaxedR1CSInstance, RoundWitness, StageConfig, StageWitness,
        VerifierR1CSBuilder,
    };
    use crate::transcripts::{KeccakTranscript, Transcript};
    use rand::thread_rng;

    let mut rng = thread_rng();

    let configs = [StageConfig::new(2, 3)];

    let round1 = RoundWitness::new(
        vec![
            Fr::from(20u64),
            Fr::from(5u64),
            Fr::from(7u64),
            Fr::from(3u64),
        ],
        Fr::from(2u64),
    );
    let next1 = round1.evaluate(Fr::from(2u64));

    let c0_2 = Fr::from(30u64);
    let c2_2 = Fr::from(10u64);
    let c3_2 = Fr::from(5u64);
    let c1_2 = next1 - Fr::from(75u64);
    let round2 = RoundWitness::new(vec![c0_2, c1_2, c2_2, c3_2], Fr::from(4u64));

    let initial_claim = Fr::from(55u64);
    let blindfold_witness =
        BlindFoldWitness::new(initial_claim, vec![StageWitness::new(vec![round1, round2])]);

    let baked = BakedPublicInputs {
        challenges: vec![Fr::from(2u64), Fr::from(4u64)],
        initial_claims: vec![initial_claim],
        ..Default::default()
    };
    let builder = VerifierR1CSBuilder::<Fr>::new(&configs, &baked);
    let r1cs = builder.build();

    let gens = PedersenGenerators::<Bn254Curve>::deterministic(r1cs.hyrax.C + 1);

    let z = blindfold_witness.assign(&r1cs);
    assert!(r1cs.is_satisfied(&z));

    let witness: Vec<Fr> = z[1..].to_vec();

    let hyrax = &r1cs.hyrax;
    let hyrax_C = hyrax.C;
    let R_coeff = hyrax.R_coeff;
    let R_prime = hyrax.R_prime;

    let (round_commitments, _round_coefficients, round_blindings) =
        round_commitment_data(&gens, &blindfold_witness.stages, &mut rng);

    let noncoeff_rows = hyrax.noncoeff_rows();
    let mut noncoeff_row_commitments = Vec::new();
    let mut w_row_blindings = vec![Fr::from(0u64); R_prime];
    for (i, blinding) in round_blindings.iter().enumerate() {
        w_row_blindings[i] = *blinding;
    }
    let noncoeff_start = R_coeff * hyrax_C;
    for row in 0..noncoeff_rows {
        let start = noncoeff_start + row * hyrax_C;
        let end = (start + hyrax_C).min(witness.len());
        let mut row_data = vec![Fr::from(0u64); hyrax_C];
        row_data[..end - start].copy_from_slice(&witness[start..end]);
        let blinding = Fr::random(&mut rng);
        noncoeff_row_commitments.push(gens.commit(&row_data, &blinding));
        w_row_blindings[R_coeff + row] = blinding;
    }

    let (real_instance, real_witness) = RelaxedR1CSInstance::<Fr, Bn254Curve>::new_non_relaxed(
        &witness,
        r1cs.num_constraints,
        hyrax_C,
        round_commitments,
        noncoeff_row_commitments,
        Vec::new(),
        w_row_blindings,
    );

    let prover = BlindFoldProver::new(&gens, &r1cs, None);
    let verifier = BlindFoldVerifier::new(&gens, &r1cs, None);

    let mut prover_transcript = KeccakTranscript::new(b"BlindFold_E2E");
    let proof = prover.prove(&real_instance, &real_witness, &z, &mut prover_transcript);

    let verifier_input = BlindFoldVerifierInput {
        round_commitments: real_instance.round_commitments.clone(),
        eval_commitments: real_instance.eval_commitments.clone(),
    };

    let mut verifier_transcript = KeccakTranscript::new(b"BlindFold_E2E");
    let result = verifier.verify(&proof, &verifier_input, &mut verifier_transcript);

    assert!(
        result.is_ok(),
        "BlindFold protocol verification failed: {result:?}"
    );

    println!("\n=== BlindFold Protocol E2E Test ===");
    println!(
        "R1CS size: {} constraints, {} variables",
        r1cs.num_constraints, r1cs.num_vars
    );
    println!("Witness size: {} field elements", witness.len());
    println!("Spartan sumcheck rounds: {}", proof.spartan_proof.len());
    println!("Protocol verification: SUCCESS");
}

#[test]
#[serial]
fn fib_e2e_dory_address_major() {
    DoryGlobals::reset();
    DoryGlobals::set_layout(DoryLayout::AddressMajor);

    let mut program = host::Program::new("fibonacci-guest");
    let inputs = postcard::to_stdvec(&50u32).unwrap();
    let (bytecode, init_memory_state, _) = program.decode();
    let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode.clone(),
        io_device.memory_layout.clone(),
        init_memory_state,
        1 << 16,
    );
    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
    let elf_contents = program.get_elf_contents().expect("elf contents is None");
    let prover = RV64IMACProver::gen_from_elf(
        &prover_preprocessing,
        &elf_contents,
        &inputs,
        &[],
        &[],
        None,
        None,
        None,
    );
    let io_device = prover.program_io.clone();
    let (proof, debug_info) = prover.prove();

    let verifier_preprocessing = JoltVerifierPreprocessing::new(
        shared_preprocessing,
        prover_preprocessing.generators.to_verifier_setup(),
    );

    RV64IMACVerifier::new(&verifier_preprocessing, proof, io_device, None, debug_info)
        .expect("verifier creation failed")
        .verify()
        .expect("verification failed");
}

#[test]
#[serial]
fn advice_e2e_dory_address_major() {
    DoryGlobals::reset();
    DoryGlobals::set_layout(DoryLayout::AddressMajor);

    let mut program = host::Program::new("merkle-tree-guest");
    let (bytecode, init_memory_state, _) = program.decode();

    let inputs = postcard::to_stdvec(&[5u8; 32].as_slice()).unwrap();
    let untrusted_advice = postcard::to_stdvec(&[8u8; 32]).unwrap();
    let mut trusted_advice = postcard::to_stdvec(&[6u8; 32]).unwrap();
    trusted_advice.extend(postcard::to_stdvec(&[7u8; 32]).unwrap());

    let (_, _, _, io_device) = program.trace(&inputs, &untrusted_advice, &trusted_advice);
    let shared_preprocessing = JoltSharedPreprocessing::new(
        bytecode.clone(),
        io_device.memory_layout.clone(),
        init_memory_state,
        1 << 16,
    );
    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing.clone());
    let elf_contents = program.get_elf_contents().expect("elf contents is None");

    let (trusted_commitment, trusted_hint) =
        commit_trusted_advice_preprocessing_only(&prover_preprocessing, &trusted_advice);

    let prover = RV64IMACProver::gen_from_elf(
        &prover_preprocessing,
        &elf_contents,
        &inputs,
        &untrusted_advice,
        &trusted_advice,
        Some(trusted_commitment),
        Some(trusted_hint),
        None,
    );
    let io_device = prover.program_io.clone();
    let (jolt_proof, debug_info) = prover.prove();

    let verifier_preprocessing = JoltVerifierPreprocessing::from(&prover_preprocessing);
    RV64IMACVerifier::new(
        &verifier_preprocessing,
        jolt_proof,
        io_device.clone(),
        Some(trusted_commitment),
        debug_info,
    )
    .expect("Failed to create verifier")
    .verify()
    .expect("Verification failed");

    let expected_output = &[
        0xb4, 0x37, 0x0f, 0x3a, 0xb, 0x3d, 0x38, 0xa8, 0x7a, 0x6c, 0x4c, 0x46, 0x9, 0xe7, 0x83,
        0xb3, 0xcc, 0xb7, 0x1c, 0x30, 0x1f, 0xf8, 0x54, 0xd, 0xf7, 0xdd, 0xc8, 0x42, 0x32, 0xbb,
        0x16, 0xd7,
    ];
    assert_eq!(io_device.outputs, expected_output);
}
