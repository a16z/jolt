//! Packed field-inline proving shared by the acceptance matrix and the
//! specialized parity and soundness suite.

use common::jolt_device::JoltDevice;
use jolt_akita::{AkitaField, AkitaScheduleArtifacts, AkitaScheme};
use jolt_program::execution::{OwnedTrace, TraceOutput, TraceRow};
use jolt_prover::akita::preprocessing::{AkitaTranscript, AkitaVc};
use jolt_prover::akita::JoltAkitaBackend;
use jolt_prover::{akita, ProverConfig};
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
}

/// Prepare and prove a guest case with the modular packed prover. The grouped
/// setup includes the full-width field increment commitment, including all-zero traces.
/// The callback inspects the attached witness before proving.
pub fn prove<D>(
    case: &GuestCase,
    backend: JoltAkitaBackend<AkitaField, AkitaScheme>,
    inspect: impl FnOnce(&ProverConfig, &dyn FieldInlineWitnessOracle<AkitaField>) -> D,
) -> (ProveOutput, D) {
    let PreparedGuest {
        preprocessing: program_preprocessing,
        trace: trace_output,
        program,
    } = crate::support::prepare(case);
    let memory_layout = trace_output.device.memory_layout.clone();
    let public_io = trace_output.device.clone();
    let config = ProverConfig::derive::<AkitaField>(
        trace_output.trace.rows(),
        &memory_layout,
        program_preprocessing.ram.min_bytecode_address,
        program_preprocessing.ram.bytecode_words.len(),
        program_preprocessing.max_padded_trace_length,
    )
    .expect("derive config");

    let log_t = config.trace_length.ilog2() as usize;
    let untrusted_advice = !case.untrusted_advice.is_empty();
    let prover_preprocessing = akita::preprocessing::preprocess_full_with_advice(
        &AkitaScheduleArtifacts::shared_from_default_directory(),
        program_preprocessing,
        &config,
        untrusted_advice,
        false,
    )
    .expect("field-inline packed preprocessing");

    let mut rows = trace_output.trace.into_rows();
    rows.resize(config.trace_length, TraceRow::default());
    let padded_output = TraceOutput::new(
        OwnedTrace::new(rows),
        trace_output.device,
        trace_output.final_memory,
        trace_output.advice_tape,
    );
    let program_preprocessing = prover_preprocessing
        .program_arc()
        .expect("full program preprocessing");
    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config)
            .include_untrusted_advice(untrusted_advice),
        JoltVmWitnessInputs::new(&program, &program_preprocessing, padded_output),
    )
    .with_field_inline()
    .expect("field-inline witness view");
    let diagnostics = inspect(
        &config,
        JoltWitnessOracle::<AkitaField>::field_inline(&witness).expect("field-inline oracle"),
    );

    let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript, _>(
        &backend,
        &prover_preprocessing,
        &config,
        None,
        &witness,
        &public_io,
    )
    .expect("packed field-inline prove");
    (
        ProveOutput {
            verifier_preprocessing: prover_preprocessing.verifier,
            public_io,
            proof,
        },
        diagnostics,
    )
}

pub fn verify_full(
    preprocessing: &JoltVerifierPreprocessing<AkitaScheme, AkitaVc>,
    public_io: &JoltDevice,
    proof: &Proof,
) -> Result<(), VerifierError> {
    jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
        preprocessing,
        public_io,
        proof,
        None,
    )
}
