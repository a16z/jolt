use common::jolt_device::JoltDevice;
use jolt_crypto::{Bn254G1, Pedersen};
use jolt_dory::DoryScheme;
use jolt_field::Fr;
use jolt_prover::{dory, JoltBackend, JoltSharedPreprocessing, ProverConfig};
use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;
use jolt_verifier::proof::JoltProof;
use jolt_verifier::{JoltVerifierPreprocessing, VerifierError};
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

use crate::support::{self, GuestCase, PreparedGuest};

pub type Proof = JoltProof<DoryScheme, Pedersen<Bn254G1>>;
pub type VerifierPreprocessing = JoltVerifierPreprocessing<DoryScheme, Pedersen<Bn254G1>>;

pub fn prove(
    case: &GuestCase,
    backend: JoltBackend<Fr, DoryScheme>,
) -> (VerifierPreprocessing, JoltDevice, Proof) {
    let PreparedGuest {
        program,
        preprocessing,
        trace,
    } = support::prepare(case);
    let config = ProverConfig::derive_from_dimensions::<Fr>(
        trace.dimensions,
        &preprocessing.memory_layout,
        preprocessing.ram.min_bytecode_address,
        preprocessing.ram.bytecode_words.len(),
        preprocessing.max_padded_trace_length,
    )
    .expect("derive config");
    let preprocessing = dory::from_shared(
        JoltSharedPreprocessing::new(preprocessing).expect("shared preprocessing"),
    )
    .expect("Dory preprocessing");
    let program_preprocessing = preprocessing
        .program_arc()
        .expect("full program preprocessing");
    let public_io = trace.device.clone();
    let witness_output = trace;
    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(
            config.trace_length.ilog2() as usize,
            config.ram_K,
            config.one_hot_config,
        ),
        JoltVmWitnessInputs::new(&program, &program_preprocessing, witness_output),
    )
    .with_field_inline()
    .expect("field-inline witness view");
    let proof = dory::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
        backend.with_witness(&witness),
        &preprocessing,
        &config,
        None,
        &public_io,
    )
    .expect("modular field-inline prove");
    (preprocessing.verifier, public_io, proof)
}

pub fn verify_full(
    preprocessing: &VerifierPreprocessing,
    public_io: &JoltDevice,
    proof: &Proof,
) -> Result<(), VerifierError> {
    jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
        preprocessing,
        public_io,
        proof,
        None,
    )
}
