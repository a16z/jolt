//! Akita field-inline proving shared by the acceptance matrix and the
//! specialized parity and soundness suite.

use common::jolt_device::JoltDevice;
use jolt_akita::{AkitaChunkProfile, AkitaField, AkitaScheduleArtifacts, AkitaScheme};
use jolt_akita::{AkitaCommitment, AkitaProverSetup};
use jolt_prover::akita::preprocessing::AkitaVc;
use jolt_prover::JoltBackend;
use jolt_prover::{akita, ProverConfig};
use jolt_transcript::{Keccak, ProverState, VerifierState};
use jolt_verifier::proof::JoltProof;
use jolt_verifier::{JoltVerifierPreprocessing, VerifierError};
use jolt_witness::field_inline::FieldInlineWitnessOracle;
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, JoltWitnessOracle, TraceBackend};

use crate::support::{GuestCase, PreparedGuest};

pub type Proof = JoltProof<AkitaScheme, AkitaVc>;

pub struct ProveOutput {
    pub verifier_preprocessing: JoltVerifierPreprocessing<AkitaScheme, AkitaVc>,
    pub public_io: JoltDevice,
    pub proof: Proof,
    trusted: Option<AkitaCommitment>,
}

/// Prepare and prove a guest case with the modular Akita prover. The grouped
/// setup includes the full-width field increment commitment, including all-zero traces.
/// The callback inspects the attached witness before proving.
pub fn prove<D>(
    case: &GuestCase,
    committed_profile: Option<AkitaChunkProfile>,
    backend: JoltBackend<AkitaField, AkitaScheme>,
    inspect: impl FnOnce(
        &ProverConfig,
        &dyn FieldInlineWitnessOracle<AkitaField>,
        &AkitaProverSetup,
    ) -> D,
) -> (ProveOutput, D) {
    let PreparedGuest {
        preprocessing: program_preprocessing,
        trace: trace_output,
        program,
    } = crate::support::prepare(case);
    let memory_layout = trace_output.device.memory_layout.clone();
    let public_io = trace_output.device.clone();
    let mut config = ProverConfig::derive_from_dimensions::<AkitaField>(
        trace_output.dimensions,
        &memory_layout,
        program_preprocessing.ram.min_bytecode_address,
        program_preprocessing.ram.bytecode_words.len(),
        program_preprocessing.max_padded_trace_length,
    )
    .expect("derive config");

    let log_t = config.trace_length.ilog2() as usize;
    let untrusted_advice = !case.untrusted_advice.is_empty();
    let trusted_advice = !case.trusted_advice.is_empty();
    let preprocess = if let Some(profile) = committed_profile {
        config.akita_chunk_profile = profile;
        akita::preprocessing::preprocess_committed_with_advice
    } else {
        akita::preprocessing::preprocess_full_with_advice
    };
    let prover_preprocessing = preprocess(
        &AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2)),
        program_preprocessing,
        &config,
        untrusted_advice,
        trusted_advice,
    )
    .expect("field-inline Akita preprocessing");

    let trusted = trusted_advice.then(|| {
        akita::preprocessing::commit_trusted_advice(&prover_preprocessing, &case.trusted_advice)
            .expect("trusted advice")
    });
    let witness_output = trace_output;
    let program_preprocessing = prover_preprocessing
        .program_arc()
        .expect("full program preprocessing");
    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config)
            .include_untrusted_advice(untrusted_advice)
            .include_trusted_advice(trusted_advice),
        JoltVmWitnessInputs::new(&program, &program_preprocessing, witness_output),
    )
    .with_field_inline()
    .expect("field-inline witness view");
    let diagnostics = inspect(
        &config,
        JoltWitnessOracle::<AkitaField>::field_inline(&witness).expect("field-inline oracle"),
        &prover_preprocessing.pcs_setup,
    );

    let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, ProverState<Keccak>>(
        backend.with_witness(&witness),
        &prover_preprocessing,
        &config,
        trusted.as_ref(),
        &public_io,
    )
    .expect("Akita field-inline prove");
    (
        ProveOutput {
            verifier_preprocessing: prover_preprocessing.verifier,
            public_io,
            proof,
            trusted: trusted.map(|object| object.commitment),
        },
        diagnostics,
    )
}

impl ProveOutput {
    pub fn verify(&self, public_io: &JoltDevice, proof: &Proof) -> Result<(), VerifierError> {
        jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, VerifierState<'_, Keccak>>(
            &self.verifier_preprocessing,
            public_io,
            proof,
            self.trusted.as_ref(),
        )
    }
}
