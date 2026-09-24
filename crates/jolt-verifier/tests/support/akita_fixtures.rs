//! Akita prover artifacts backing verifier completeness and tamper tests.

#![expect(
    clippy::expect_used,
    reason = "fixture generation should fail loudly when prover artifact construction breaks"
)]

use std::sync::OnceLock;

use common::jolt_device::JoltDevice;
use jolt_akita::{AkitaCommitment, AkitaField, AkitaScheduleArtifacts, AkitaScheme};
use jolt_host::Program;
use jolt_prover::akita::preprocessing::{self, AkitaProverPreprocessing, AkitaTranscript, AkitaVc};
use jolt_prover::akita::{self, JoltAkitaBackend};
use jolt_prover::ProverConfig;
use jolt_verifier::proof::JoltProof;
use jolt_verifier::{verify, JoltVerifierPreprocessing, VerifierError};

use super::guest_fixtures::{fixture_witness, prepare_guest, PreparedGuest};

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
    let preprocessing = preprocessing::preprocess_full(
        &AkitaScheduleArtifacts::shared_from_default_directory(),
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
        &AkitaScheduleArtifacts::shared_from_default_directory(),
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
        &AkitaScheduleArtifacts::shared_from_default_directory(),
        run.program_preprocessing.clone(),
        &config,
        2,
    )
    .expect("committed Akita preprocessing");
    prove_prepared(run, config, preprocessing, &[])
}

fn derive_config(run: &PreparedGuest) -> ProverConfig {
    #[cfg(not(feature = "field-inline"))]
    {
        ProverConfig::derive_compact::<AkitaField>(
            run.trace.trace.as_slice(),
            &run.program_preprocessing.memory_layout,
            run.program_preprocessing.ram.min_bytecode_address,
            run.program_preprocessing.ram.bytecode_words.len(),
            MAX_PADDED_TRACE_LENGTH,
        )
        .expect("derive Akita prover config")
    }
    #[cfg(feature = "field-inline")]
    {
        ProverConfig::derive::<AkitaField>(
            run.trace.trace.rows(),
            &run.program_preprocessing.memory_layout,
            run.program_preprocessing.ram.min_bytecode_address,
            run.program_preprocessing.ram.bytecode_words.len(),
            MAX_PADDED_TRACE_LENGTH,
        )
        .expect("derive Akita prover config")
    }
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

/// The FR-on packed case: the eq-MLE FR guest proven by the MODULAR packed
/// prover (the only FR-capable one) over fp128, with the transparent grouped
/// setup carrying the FR limb arity line — the packed twin of the Dory
/// `standard_field_inline_eqpoly_case`. Legacy-generated akita fixtures pin
/// the FR axis disabled and cannot verify FR-on, so this is the only packed
/// fixture the FR-on akita verifier suites run over.
#[cfg(feature = "field-inline")]
pub fn akita_field_inline_eqpoly_case() -> &'static AkitaFixtureCase {
    static CASE: OnceLock<AkitaFixtureCase> = OnceLock::new();
    CASE.get_or_init(field_inline::generate_eqpoly)
}

#[cfg(feature = "field-inline")]
mod field_inline {
    use std::sync::Arc;

    use common::jolt_device::{MemoryConfig, MemoryLayout};
    use jolt_akita::{AkitaField, AkitaScheduleArtifacts, AkitaScheme};
    use jolt_field::{CanonicalBytes, Ring};
    use jolt_host::{JoltProgramSource, Program};
    use jolt_program::execution::{
        ExecutionBackend, JoltProgram, OwnedTrace, TraceInputs, TraceOutput, TraceRow,
    };
    use jolt_program::preprocess::JoltProgramPreprocessing;
    use jolt_prover::akita::preprocessing::{AkitaTranscript, AkitaVc};
    use jolt_prover::akita::JoltAkitaBackend;
    use jolt_prover::{akita, ProverConfig};
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
    use tracer::execution_backend::TracerBackend;

    use super::AkitaFixtureCase;

    const MAX_PADDED_TRACE_LENGTH: usize = 1 << 16;
    const EQ_PAIRS: [[u64; 2]; 4] = [[3, 5], [7, 2], [11, 13], [1, 9]];

    /// `eq(r, x) = Π_i (r_i·x_i + (1 − r_i)(1 − x_i))` over the packed axis's
    /// proof field, pinned as four canonical little-endian u64 limbs (the
    /// 16-byte fp128 form fills the low two; the guest Horner-recomposes them
    /// in whatever field it proves over).
    fn eqpoly_inputs() -> Vec<u8> {
        let one = AkitaField::from_u64(1);
        let value = EQ_PAIRS.iter().fold(one, |acc, [r, x]| {
            let r = AkitaField::from_u64(*r);
            let x = AkitaField::from_u64(*x);
            acc * (r * x + (one - r) * (one - x))
        });
        let bytes = value.to_bytes_le_vec();
        let mut limbs = [0u64; 4];
        for (limb, chunk) in limbs.iter_mut().zip(bytes.chunks_exact(8)) {
            *limb = u64::from_le_bytes(chunk.try_into().expect("8-byte chunk"));
        }
        let mut inputs = postcard::to_stdvec(&EQ_PAIRS).expect("serialize pairs");
        inputs.extend(postcard::to_stdvec(&limbs).expect("serialize limbs"));
        inputs
    }

    fn trace_modular(
        program: &JoltProgram,
        memory_layout: &MemoryLayout,
        inputs: &[u8],
    ) -> TraceOutput<OwnedTrace> {
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

        let config = ProverConfig::derive::<AkitaField>(
            trace_output.trace.rows(),
            &memory_layout,
            program_preprocessing.ram.min_bytecode_address,
            program_preprocessing.ram.bytecode_words.len(),
            MAX_PADDED_TRACE_LENGTH,
        )
        .expect("derive config");
        let log_t = config.trace_length.ilog2() as usize;
        let prover_preprocessing = jolt_prover::akita::preprocessing::preprocess_full(
            &AkitaScheduleArtifacts::shared_from_default_directory(),
            program_preprocessing,
            &config,
        )
        .expect("field-inline packed preprocessing");

        let mut rows = trace_output.trace.rows().to_vec();
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
            JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
            JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded_output),
        )
        .with_field_inline()
        .expect("field-inline witness view");
        let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript, _>(
            &JoltAkitaBackend::optimized(),
            &prover_preprocessing,
            &config,
            None,
            &witness,
            &public_io,
        )
        .expect("packed FR prove");
        AkitaFixtureCase {
            preprocessing: prover_preprocessing.verifier,
            public_io,
            proof,
            trusted_advice_commitment: None,
        }
    }
}
