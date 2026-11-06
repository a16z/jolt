#![expect(
    clippy::expect_used,
    reason = "fixture generation should fail loudly when prover artifact construction breaks"
)]

use std::sync::OnceLock;

#[cfg(all(feature = "zk", feature = "logging"))]
use akita_transcript::ProofMessageRange;
use common::jolt_device::JoltDevice;
use jolt_akita::{
    AkitaChunkProfile, AkitaCommitment, AkitaField, AkitaScheduleArtifacts, AkitaScheme,
};
use jolt_host::Program;
use jolt_prover::akita;
use jolt_prover::akita::preprocessing::{self, AkitaProverPreprocessing, AkitaVc};
#[cfg(feature = "zk")]
use jolt_prover::akita::whiteout_faults::Aborted;
use jolt_prover::JoltBackend;
use jolt_prover::ProverConfig;
use jolt_transcript::{Keccak, ProverState, VerifierState};
use jolt_verifier::proof::JoltProof;
use jolt_verifier::{verify, JoltVerifierPreprocessing, VerifierError};

use super::guest_fixtures::{fixture_witness, prepare_guest, PreparedGuest};

const MAX_PADDED_TRACE_LENGTH: usize = 1 << 16;

#[cfg(not(feature = "zk"))]
pub type AkitaJoltProof = JoltProof<AkitaScheme, AkitaVc>;
/// A Whiteout proof carries no clear opening proof and no ZK payload outside
/// its masked tape and argument string.
#[cfg(feature = "zk")]
pub type AkitaJoltProof = JoltProof<AkitaScheme, AkitaVc, (), ()>;

pub struct AkitaFixtureCase {
    pub preprocessing: JoltVerifierPreprocessing<AkitaScheme, AkitaVc>,
    pub public_io: JoltDevice,
    pub proof: AkitaJoltProof,
    pub trusted_advice_commitment: Option<AkitaCommitment>,
    /// Whiteout: the proof's pads, for probes that rebuild the residual
    /// relation as a cheating prover would.
    #[cfg(feature = "zk")]
    pub pads: Vec<AkitaField>,
    /// Whiteout: Akita's record of the argument string's message ranges, from
    /// the proving thread (`logging`).
    #[cfg(all(feature = "zk", feature = "logging"))]
    pub narg_ranges: Vec<ProofMessageRange>,
    /// Whiteout: the argument string's length at the end of each prover
    /// region (`jolt_prover::akita::whiteout_regions`).
    #[cfg(all(feature = "zk", feature = "logging"))]
    pub narg_regions: Vec<(&'static str, usize)>,
}

impl AkitaFixtureCase {
    pub fn verify(&self) -> Result<(), VerifierError> {
        self.verify_proof(&self.proof)
    }

    pub fn verify_proof(&self, proof: &AkitaJoltProof) -> Result<(), VerifierError> {
        verify::<AkitaField, AkitaScheme, AkitaVc, VerifierState<'_, Keccak>>(
            &self.preprocessing,
            &self.public_io,
            proof,
            self.trusted_advice_commitment.as_ref(),
        )
    }
}

/// The muldiv case: `OneHotTrace` and dense `FusedInc`, with no advice objects.
#[cfg(not(feature = "zk"))]
pub fn akita_muldiv_case() -> &'static AkitaFixtureCase {
    static CASE: OnceLock<AkitaFixtureCase> = OnceLock::new();
    CASE.get_or_init(generate_muldiv)
}

/// The advice case: both advice kinds alongside `OneHotTrace` and `FusedInc`
/// in one grouped opening.
#[cfg(not(feature = "zk"))]
pub fn akita_advice_case() -> &'static AkitaFixtureCase {
    static CASE: OnceLock<AkitaFixtureCase> = OnceLock::new();
    CASE.get_or_init(generate_advice)
}

/// The committed-program case: direct bytecode and program-image objects in
/// the native grouped opening.
#[cfg(not(feature = "zk"))]
pub fn akita_committed_muldiv_case() -> &'static AkitaFixtureCase {
    static CASE: OnceLock<AkitaFixtureCase> = OnceLock::new();
    CASE.get_or_init(generate_committed_muldiv)
}

#[cfg(not(feature = "zk"))]
fn generate_muldiv() -> AkitaFixtureCase {
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");
    let run = prepare_guest(Program::new("muldiv-guest"), &inputs, &[], &[]);
    let config = derive_config(&run);
    let preprocessing = preprocessing::preprocess_full(
        &AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2)),
        run.program_preprocessing.clone(),
        &config,
    )
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
        &AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2)),
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
    let preprocessing = preprocessing::preprocess_committed(
        &AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2)),
        run.program_preprocessing.clone(),
        &config,
    )
    .expect("committed Akita preprocessing");
    prove_prepared(run, config, preprocessing, &[])
}

/// Whiteout's muldiv case (`whiteout`): one hiding `OneHotTrace` group,
/// under the production setup.
#[cfg(feature = "zk")]
pub fn whiteout_muldiv_case() -> &'static AkitaFixtureCase {
    static CASE: OnceLock<AkitaFixtureCase> = OnceLock::new();
    CASE.get_or_init(generate_whiteout_muldiv)
}

/// Whiteout's advice case (`whiteout`): both advice kinds as hiding
/// precommitted groups beside the trace group.
#[cfg(feature = "zk")]
pub fn whiteout_advice_case() -> &'static AkitaFixtureCase {
    static CASE: OnceLock<AkitaFixtureCase> = OnceLock::new();
    CASE.get_or_init(generate_advice)
}

/// Whiteout's committed-program case (`whiteout`): the bytecode and
/// program-image objects as hiding precommitted groups beside the trace
/// group.
#[cfg(feature = "zk")]
pub fn whiteout_committed_case() -> &'static AkitaFixtureCase {
    static CASE: OnceLock<AkitaFixtureCase> = OnceLock::new();
    CASE.get_or_init(generate_committed_muldiv)
}

/// Whiteout's muldiv case proved with its first attempt aborted after the
/// masked levels (HT3), and what that attempt had produced.
#[cfg(feature = "zk")]
pub fn whiteout_rerun_case() -> (AkitaFixtureCase, Aborted) {
    akita::whiteout_faults::abort_next_attempt();
    let case = generate_whiteout_muldiv();
    let aborted = akita::whiteout_faults::take_aborted().expect("the first attempt aborted");
    (case, aborted)
}

#[cfg(feature = "zk")]
fn generate_whiteout_muldiv() -> AkitaFixtureCase {
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");
    let run = prepare_guest(Program::new("muldiv-guest"), &inputs, &[], &[]);
    let config = derive_config(&run);
    let preprocessing = preprocessing::preprocess_full(
        &AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2)),
        run.program_preprocessing.clone(),
        &config,
    )
    .expect("Whiteout preprocessing");
    prove_prepared(run, config, preprocessing, &[])
}

/// Akita's record of the last proof's argument string on this thread. Every
/// attempt writes a fresh string, so a rerun's ranges restart at offset 0:
/// the last attempt's are the suffix after the last backwards step.
#[cfg(all(feature = "zk", feature = "logging"))]
pub fn final_attempt_ranges() -> Vec<ProofMessageRange> {
    let ranges = akita_transcript::thread_proof_ranges();
    let first = ranges
        .windows(2)
        .rposition(|pair| pair[1].start < pair[0].start)
        .map_or(0, |step| step + 1);
    ranges[first..].to_vec()
}

/// The last Whiteout proof's region marks on this thread, with where its
/// response proof began (`response`).
#[cfg(all(feature = "zk", feature = "logging"))]
pub fn whiteout_regions() -> Vec<(&'static str, usize)> {
    let mut regions = akita::whiteout_regions::take();
    if let Some(start) = jolt_akita::whiteout_census::take_response_start() {
        regions.push(("response", start));
    }
    regions
}

fn derive_config(run: &PreparedGuest) -> ProverConfig {
    ProverConfig::derive_from_dimensions::<AkitaField>(
        run.trace.dimensions,
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
    let witness = fixture_witness(
        &run.program,
        &program_preprocessing,
        run.trace,
        &config,
        has_trusted_advice,
    );
    let trusted = has_trusted_advice.then(|| {
        preprocessing::commit_trusted_advice(&preprocessing, trusted_advice)
            .expect("trusted advice commitment")
    });
    #[cfg(all(feature = "zk", feature = "logging"))]
    akita_transcript::clear_thread_events();
    let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, ProverState<Keccak>>(
        JoltBackend::optimized().with_witness(&witness),
        &preprocessing,
        &config,
        trusted.as_ref(),
        &public_io,
    )
    .expect("prove Akita verifier fixture");
    AkitaFixtureCase {
        preprocessing: preprocessing.verifier,
        public_io,
        proof,
        trusted_advice_commitment: trusted.map(|object| object.commitment),
        #[cfg(feature = "zk")]
        pads: akita::whiteout_secrets::take().expect("the Whiteout proof's pads"),
        #[cfg(all(feature = "zk", feature = "logging"))]
        narg_ranges: final_attempt_ranges(),
        #[cfg(all(feature = "zk", feature = "logging"))]
        narg_regions: whiteout_regions(),
    }
}

/// The Akita field-inline case: the eq-MLE field-inline guest proven by the MODULAR Akita
/// prover (the only field-inline-capable one) over fp128, with the transparent grouped setup
/// carrying the field-increment limb arity line — the Akita twin of the Dory
/// `standard_field_inline_eqpoly_case`. Legacy-generated akita fixtures pin the field-inline
/// axis disabled and cannot verify with field-inline enabled, so this is the only Akita
/// fixture the akita verifier suites with field-inline enabled run over.
#[cfg(feature = "field-inline")]
pub fn akita_field_inline_eqpoly_case() -> &'static AkitaFixtureCase {
    static CASE: OnceLock<AkitaFixtureCase> = OnceLock::new();
    CASE.get_or_init(field_inline::generate_eqpoly)
}

#[cfg(feature = "field-inline")]
mod field_inline {
    use std::sync::Arc;

    use common::jolt_device::{MemoryConfig, MemoryLayout};
    use jolt_akita::{AkitaChunkProfile, AkitaField, AkitaScheduleArtifacts, AkitaScheme};
    use jolt_host::{JoltProgramSource, Program};
    use jolt_program::execution::{ExecutionBackend, JoltProgram, TraceInputs, TraceOutput};
    use jolt_program::preprocess::JoltProgramPreprocessing;
    use jolt_prover::akita::preprocessing::AkitaVc;
    use jolt_prover::JoltBackend;
    use jolt_prover::{akita, ProverConfig};
    use jolt_transcript::{Keccak, ProverState};
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
    use tracer::execution_backend::TracerBackend;

    use super::AkitaFixtureCase;

    const MAX_PADDED_TRACE_LENGTH: usize = 1 << 16;
    const EQ_PAIRS: [[u64; 2]; 4] = [
        [u64::MAX, u64::MAX - 1],
        [u64::MAX - 2, 2],
        [11, 13],
        [u64::MAX - 3, 9],
    ];

    /// `eq(r, x) = Π_i (r_i·x_i + (1 − r_i)(1 − x_i))` over the Akita axis's
    /// proof field, pinned as four canonical little-endian u64 limbs (the
    /// 16-byte fp128 form fills the low two; the guest Horner-recomposes them
    /// in whatever field it proves over).
    fn eqpoly_inputs() -> Vec<u8> {
        jolt_host::field_inline::eqpoly_inputs::<AkitaField>(EQ_PAIRS)
            .expect("field-ops input encoding")
    }

    fn trace_modular(
        program: &JoltProgram,
        memory_layout: &MemoryLayout,
        inputs: &[u8],
    ) -> TraceOutput {
        let memory_config = MemoryConfig {
            max_untrusted_advice_size: memory_layout.max_untrusted_advice_size,
            max_trusted_advice_size: memory_layout.max_trusted_advice_size,
            max_input_size: memory_layout.max_input_size,
            max_output_size: memory_layout.max_output_size,
            stack_size: memory_layout.stack_size,
            heap_size: memory_layout.heap_size,
            program_size: Some(memory_layout.program_size),
        };
        TracerBackend::new()
            .trace(
                program,
                TraceInputs {
                    inputs: inputs.to_vec(),
                    untrusted_advice: Vec::new(),
                    trusted_advice: Vec::new(),
                    memory_config,
                    advice_tape: None,
                },
            )
            .expect("modular trace")
    }

    pub(super) fn generate_eqpoly() -> AkitaFixtureCase {
        let inputs = eqpoly_inputs();
        let mut program = Program::new("field-ops-guest");
        program.enable_field_inline();
        let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);
        let jolt_program = Arc::new(
            program
                .build_jolt_program()
                .expect("build field-inline program"),
        );
        let program_preprocessing = JoltProgramPreprocessing::new(
            jolt_program.expanded_bytecode.clone(),
            jolt_program.memory_init.clone(),
            io_device.memory_layout.clone(),
            jolt_program.entry_address,
            MAX_PADDED_TRACE_LENGTH,
            program.instruction_profile(),
        )
        .expect("field-inline preprocessing");
        let memory_layout = io_device.memory_layout.clone();
        let trace_output = trace_modular(&jolt_program, &memory_layout, &inputs);
        let public_io = trace_output.device.clone();

        let config = ProverConfig::derive_from_dimensions::<AkitaField>(
            trace_output.dimensions,
            &memory_layout,
            program_preprocessing.ram.min_bytecode_address,
            program_preprocessing.ram.bytecode_words.len(),
            MAX_PADDED_TRACE_LENGTH,
        )
        .expect("derive config");
        let log_t = config.trace_length.ilog2() as usize;
        let prover_preprocessing = akita::preprocessing::preprocess_full(
            &AkitaScheduleArtifacts::shared_from_default_directory(AkitaChunkProfile::new(8, 2)),
            program_preprocessing,
            &config,
        )
        .expect("field-inline Akita preprocessing");

        let program_preprocessing = prover_preprocessing
            .program_arc()
            .expect("full program preprocessing");
        let witness = TraceBackend::try_new(
            JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
            JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, trace_output),
        )
        .expect("trace witness")
        .with_field_inline()
        .expect("field-inline witness view");
        #[cfg(all(feature = "zk", feature = "logging"))]
        akita_transcript::clear_thread_events();
        let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, ProverState<Keccak>>(
            JoltBackend::optimized().with_witness(&witness),
            &prover_preprocessing,
            &config,
            None,
            &public_io,
        )
        .expect("Akita field-inline prove");
        AkitaFixtureCase {
            preprocessing: prover_preprocessing.verifier,
            public_io,
            proof,
            trusted_advice_commitment: None,
            #[cfg(feature = "zk")]
            pads: akita::whiteout_secrets::take().expect("the Whiteout proof's pads"),
            #[cfg(all(feature = "zk", feature = "logging"))]
            narg_ranges: super::final_attempt_ranges(),
            #[cfg(all(feature = "zk", feature = "logging"))]
            narg_regions: super::whiteout_regions(),
        }
    }
}
