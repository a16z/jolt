//! Akita prover artifacts backing verifier completeness and tamper tests.

#![expect(
    clippy::expect_used,
    reason = "fixture generation should fail loudly when prover artifact construction breaks"
)]

use std::sync::OnceLock;

use common::jolt_device::JoltDevice;
use jolt_akita::{AkitaCommitment, AkitaField, AkitaScheme};
use jolt_host::Program;
use jolt_program::execution::OwnedTrace;
use jolt_prover::akita::preprocessing::{self, AkitaProverPreprocessing, AkitaTranscript, AkitaVc};
use jolt_prover::akita::{self, JoltAkitaBackend};
use jolt_prover::ProverConfig;
use jolt_verifier::proof::JoltProof;
use jolt_verifier::{verify, JoltVerifierPreprocessing, VerifierError};
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

use super::guest_fixtures::{prepare_guest, PreparedGuest};

const MAX_PADDED_TRACE_LENGTH: usize = 1 << 16;

pub type AkitaJoltProof = JoltProof<AkitaScheme, AkitaVc>;

pub struct AkitaFixtureCase {
    pub preprocessing: JoltVerifierPreprocessing<AkitaScheme, AkitaVc>,
    pub public_io: JoltDevice,
    pub proof: AkitaJoltProof,
    pub trusted_advice_commitment: Option<AkitaCommitment>,
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

/// The muldiv case: one `OneHotTrace` commitment object and no precommitted objects.
pub fn akita_muldiv_case() -> &'static AkitaFixtureCase {
    static CASE: OnceLock<AkitaFixtureCase> = OnceLock::new();
    CASE.get_or_init(generate_muldiv)
}

/// The advice case: both advice kinds, three commitment objects
/// (`OneHotTrace`, `UntrustedAdvice`, `TrustedAdvice`) in one grouped opening.
pub fn akita_advice_case() -> &'static AkitaFixtureCase {
    static CASE: OnceLock<AkitaFixtureCase> = OnceLock::new();
    CASE.get_or_init(generate_advice)
}

/// The committed-program case: direct bytecode and program-image objects in
/// the native grouped opening.
pub fn akita_committed_muldiv_case() -> &'static AkitaFixtureCase {
    static CASE: OnceLock<AkitaFixtureCase> = OnceLock::new();
    CASE.get_or_init(generate_committed_muldiv)
}

fn generate_muldiv() -> AkitaFixtureCase {
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");
    let run = prepare_guest(Program::new("muldiv-guest"), &inputs, &[], &[]);
    let config = derive_config(&run);
    let preprocessing = preprocessing::preprocess_full(run.program_preprocessing.clone(), &config)
        .expect("Akita preprocessing");
    prove_prepared(run, config, preprocessing, &[])
}

fn generate_advice() -> AkitaFixtureCase {
    let inputs = postcard::to_stdvec(&12u64).expect("serialize inputs");
    let untrusted_advice = postcard::to_stdvec(&5u64).expect("serialize untrusted advice");
    let trusted_advice = postcard::to_stdvec(&7u64).expect("serialize trusted advice");
    let run = prepare_guest(
        Program::new("advice-consumer-guest"),
        &inputs,
        &untrusted_advice,
        &trusted_advice,
    );
    let config = derive_config(&run);
    let preprocessing = preprocessing::preprocess_full_with_advice(
        run.program_preprocessing.clone(),
        &config,
        true,
        true,
    )
    .expect("Akita advice preprocessing");
    prove_prepared(run, config, preprocessing, &trusted_advice)
}

fn generate_committed_muldiv() -> AkitaFixtureCase {
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");
    let run = prepare_guest(Program::new("muldiv-guest"), &inputs, &[], &[]);
    let config = derive_config(&run);
    let preprocessing =
        preprocessing::preprocess_committed(run.program_preprocessing.clone(), &config, 2)
            .expect("committed Akita preprocessing");
    prove_prepared(run, config, preprocessing, &[])
}

fn derive_config(run: &PreparedGuest) -> ProverConfig {
    ProverConfig::derive_compact::<AkitaField>(
        run.trace.trace.as_slice(),
        &run.program_preprocessing.memory_layout,
        run.program_preprocessing.ram.min_bytecode_address,
        run.program_preprocessing.ram.bytecode_words.len(),
        MAX_PADDED_TRACE_LENGTH,
    )
    .expect("derive Akita prover config")
}

fn prove_prepared(
    run: PreparedGuest,
    config: ProverConfig,
    preprocessing: AkitaProverPreprocessing,
    trusted_advice: &[u8],
) -> AkitaFixtureCase {
    let program_preprocessing = preprocessing
        .program_arc()
        .expect("full program retained by prover preprocessing");
    let public_io = run.trace.device.clone();
    let has_trusted_advice = !trusted_advice.is_empty();
    let witness = TraceBackend::<OwnedTrace>::from_compact(
        JoltVmWitnessConfig::new(
            config.trace_length.ilog2() as usize,
            config.ram_K,
            config.one_hot_config,
        )
        .include_untrusted_advice(!public_io.untrusted_advice.is_empty())
        .include_trusted_advice(has_trusted_advice),
        JoltVmWitnessInputs::new(&run.program, &program_preprocessing, run.trace),
    );
    let trusted = has_trusted_advice.then(|| {
        preprocessing::commit_trusted_advice(&preprocessing, trusted_advice)
            .expect("trusted advice commitment")
    });
    let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript, _>(
        &JoltAkitaBackend::optimized(),
        &preprocessing,
        &config,
        trusted.as_ref(),
        &witness,
        &public_io,
    )
    .expect("prove Akita verifier fixture");
    AkitaFixtureCase {
        preprocessing: preprocessing.verifier,
        public_io,
        proof,
        trusted_advice_commitment: trusted.map(|object| object.commitment),
    }
}
