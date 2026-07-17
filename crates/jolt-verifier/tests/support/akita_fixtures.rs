//! Akita verifier fixture cases: real packed-prover artifacts backing the
//! fixture-driven tamper/soundness tests on the akita path.
//!
//! Unlike the Dory fixtures there is no disk cache: the transparent akita
//! setup would have to be re-derived at load anyway, so each case is
//! generated once per test binary (`OnceLock`) and shared across every
//! tamper application.

#![expect(
    clippy::expect_used,
    reason = "fixture generation should fail loudly when prover artifact construction breaks"
)]

use std::sync::OnceLock;

use common::jolt_device::JoltDevice;
use jolt_verifier::{verify, JoltVerifierPreprocessing, VerifierError};

use jolt_openings::CommitmentScheme as VerifierCommitmentScheme;
use jolt_prover_legacy::host;
use jolt_prover_legacy::zkvm::packed::{
    akita_verifier_preprocessing, commit_trusted_advice_one_hot,
    shared_preprocessing_with_program_one_hot, AkitaField, AkitaJoltProof, AkitaPackedProver,
    AkitaPackedScheme, AkitaScheme, AkitaTranscript, AkitaVc,
};
use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
use jolt_prover_legacy::zkvm::program::ProgramPreprocessing;
use jolt_prover_legacy::zkvm::prover::{JoltCpuProver, JoltProverPreprocessing};

type AkitaCommitmentOutput = <AkitaScheme as jolt_crypto::Commitment>::Output;

pub struct AkitaFixtureCase {
    pub preprocessing: JoltVerifierPreprocessing<AkitaScheme, AkitaVc>,
    pub public_io: JoltDevice,
    pub proof: AkitaJoltProof,
    pub trusted_advice_commitment: Option<AkitaCommitmentOutput>,
}

impl AkitaFixtureCase {
    pub fn verify(&self) -> Result<(), VerifierError> {
        self.verify_proof(&self.proof)
    }

    pub fn verify_proof(&self, proof: &AkitaJoltProof) -> Result<(), VerifierError> {
        verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
            &self.preprocessing,
            &self.public_io,
            proof,
            self.trusted_advice_commitment.as_ref(),
        )
    }
}

/// The muldiv case: one `OneHotTrace` commitment object, no auxiliary objects.
pub fn akita_muldiv_case() -> &'static AkitaFixtureCase {
    static CASE: OnceLock<AkitaFixtureCase> = OnceLock::new();
    CASE.get_or_init(generate_muldiv)
}

/// The advice case: both advice kinds, three commitment objects
/// (`OneHotTrace`, `UntrustedAdviceOneHot`, `TrustedAdviceOneHot`) and an auxiliary joint opening.
pub fn akita_advice_case() -> &'static AkitaFixtureCase {
    static CASE: OnceLock<AkitaFixtureCase> = OnceLock::new();
    CASE.get_or_init(generate_advice)
}

/// The committed-program case: `ProgramOneHot` as the auxiliary packed object.
pub fn akita_committed_muldiv_case() -> &'static AkitaFixtureCase {
    static CASE: OnceLock<AkitaFixtureCase> = OnceLock::new();
    CASE.get_or_init(generate_committed_muldiv)
}

fn generate_muldiv() -> AkitaFixtureCase {
    let mut program = host::Program::new("muldiv-guest");
    let (bytecode, init_memory_state, _, e_entry) = program.decode();
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");
    let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

    let program_data = ProgramPreprocessing::preprocess(bytecode, init_memory_state, e_entry)
        .expect("program preprocessing");
    let shared: JoltSharedPreprocessing<AkitaPackedScheme> =
        JoltSharedPreprocessing::new(program_data, io_device.memory_layout.clone(), 1 << 16);
    let prover_preprocessing = JoltProverPreprocessing::new(shared);
    let elf_contents = program.get_elf_contents().expect("elf contents");
    let prover: AkitaPackedProver<'_> = JoltCpuProver::gen_from_elf(
        &prover_preprocessing,
        &elf_contents,
        &inputs,
        &[],
        &[],
        None,
        None,
        None,
    );
    let public_io = prover.program_io.clone();
    let (object_setup, verifier_setup) =
        <AkitaScheme as VerifierCommitmentScheme>::setup(prover.one_hot_trace_setup_params())
            .expect("transparent packed setup");
    let proof = prover
        .prove_packed(&object_setup, None, None)
        .expect("packed prover");
    let preprocessing = akita_verifier_preprocessing(&prover_preprocessing, verifier_setup, None);
    AkitaFixtureCase {
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment: None,
    }
}

fn generate_advice() -> AkitaFixtureCase {
    // The purpose-built advice guest asserts `trusted + untrusted == public`
    // (7 + 5 == 12), exercising both advice kinds without any exotic inline
    // instruction (unlike the merkle example, which fails Jolt expansion).
    let mut program = host::Program::new("advice-consumer-guest");
    let (bytecode, init_memory_state, _, e_entry) = program.decode();
    let inputs = postcard::to_stdvec(&12u64).expect("serialize inputs");
    let untrusted_advice = postcard::to_stdvec(&5u64).expect("serialize untrusted");
    let trusted_advice = postcard::to_stdvec(&7u64).expect("serialize trusted");
    let (_, _, _, io_device) = program.trace(&inputs, &untrusted_advice, &trusted_advice);

    let program_data = ProgramPreprocessing::preprocess(bytecode, init_memory_state, e_entry)
        .expect("program preprocessing");
    let shared: JoltSharedPreprocessing<AkitaPackedScheme> =
        JoltSharedPreprocessing::new(program_data, io_device.memory_layout.clone(), 1 << 16);
    let prover_preprocessing = JoltProverPreprocessing::new(shared);
    let elf_contents = program.get_elf_contents().expect("elf contents");
    let trusted_object = commit_trusted_advice_one_hot(
        &trusted_advice,
        io_device.memory_layout.max_trusted_advice_size as usize,
    )
    .expect("trusted advice object");
    let prover: AkitaPackedProver<'_> = JoltCpuProver::gen_from_elf(
        &prover_preprocessing,
        &elf_contents,
        &inputs,
        &untrusted_advice,
        &trusted_advice,
        None,
        None,
        None,
    );
    let public_io = prover.program_io.clone();
    let (object_setup, verifier_setup) =
        <AkitaScheme as VerifierCommitmentScheme>::setup(prover.one_hot_trace_setup_params())
            .expect("transparent packed setup");
    let trusted_commitment = trusted_object.commitment.clone();
    let proof = prover
        .prove_packed(&object_setup, Some(trusted_object), None)
        .expect("packed prover");
    let preprocessing = akita_verifier_preprocessing(&prover_preprocessing, verifier_setup, None);
    AkitaFixtureCase {
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment: Some(trusted_commitment),
    }
}

fn generate_committed_muldiv() -> AkitaFixtureCase {
    let mut program = host::Program::new("muldiv-guest");
    let (bytecode, init_memory_state, _, e_entry) = program.decode();
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");
    let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

    let program_data = ProgramPreprocessing::preprocess(bytecode, init_memory_state, e_entry)
        .expect("program preprocessing");
    let (shared, prover_data, program_one_hot) = shared_preprocessing_with_program_one_hot(
        program_data,
        io_device.memory_layout.clone(),
        1 << 16,
        2,
    )
    .expect("packed committed preprocessing");
    let prover_preprocessing =
        JoltProverPreprocessing::new_committed(shared, prover_data, AkitaPackedScheme);
    let elf_contents = program.get_elf_contents().expect("elf contents");
    let prover: AkitaPackedProver<'_> = JoltCpuProver::gen_from_elf(
        &prover_preprocessing,
        &elf_contents,
        &inputs,
        &[],
        &[],
        None,
        None,
        None,
    );
    let public_io = prover.program_io.clone();
    let (object_setup, verifier_setup) =
        <AkitaScheme as VerifierCommitmentScheme>::setup(prover.one_hot_trace_setup_params())
            .expect("transparent packed setup");
    let program_one_hot_commitment = program_one_hot.commitment.clone();
    let proof = prover
        .prove_packed(&object_setup, None, Some(program_one_hot))
        .expect("packed prover");
    let preprocessing = akita_verifier_preprocessing(
        &prover_preprocessing,
        verifier_setup,
        Some(program_one_hot_commitment),
    );
    AkitaFixtureCase {
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment: None,
    }
}
