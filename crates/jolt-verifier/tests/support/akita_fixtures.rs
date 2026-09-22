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
    akita_verifier_preprocessing, commit_trusted_advice, shared_preprocessing_with_direct_program,
    AkitaField, AkitaJoltProof, AkitaPackedProver, AkitaPackedScheme, AkitaScheduleArtifacts,
    AkitaScheme, AkitaTranscript, AkitaVc,
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
    let schedule_artifacts = AkitaScheduleArtifacts::shared_from_default_directory();
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
    )
    .expect("legacy prover construction");
    let public_io = prover.program_io.clone();
    let (object_setup, verifier_setup) = <AkitaScheme as VerifierCommitmentScheme>::setup(
        prover.one_hot_trace_setup_params(schedule_artifacts),
    )
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
    let schedule_artifacts = AkitaScheduleArtifacts::shared_from_default_directory();
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
    let trusted_object = commit_trusted_advice(
        &schedule_artifacts,
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
    )
    .expect("legacy prover construction");
    let public_io = prover.program_io.clone();
    let (object_setup, verifier_setup) = <AkitaScheme as VerifierCommitmentScheme>::setup(
        prover.one_hot_trace_setup_params(schedule_artifacts),
    )
    .expect("transparent packed setup");
    let trusted_commitment = trusted_object.commitment.clone();
    let proof = prover
        .prove_packed(&object_setup, Some(&trusted_object), None)
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
    let schedule_artifacts = AkitaScheduleArtifacts::shared_from_default_directory();
    let mut program = host::Program::new("muldiv-guest");
    let (bytecode, init_memory_state, _, e_entry) = program.decode();
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");
    let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);

    let program_data = ProgramPreprocessing::preprocess(bytecode, init_memory_state, e_entry)
        .expect("program preprocessing");
    let (shared, prover_data, direct_program) = shared_preprocessing_with_direct_program(
        &schedule_artifacts,
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
    )
    .expect("legacy prover construction");
    let public_io = prover.program_io.clone();
    let (object_setup, verifier_setup) = <AkitaScheme as VerifierCommitmentScheme>::setup(
        prover.one_hot_trace_setup_params(schedule_artifacts),
    )
    .expect("transparent packed setup");
    let proof = prover
        .prove_packed(&object_setup, None, Some(&direct_program))
        .expect("packed prover");
    let preprocessing =
        akita_verifier_preprocessing(&prover_preprocessing, verifier_setup, Some(&direct_program));
    AkitaFixtureCase {
        preprocessing,
        public_io,
        proof,
        trusted_advice_commitment: None,
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
    use jolt_field::{CanonicalBytes, Ring};
    use jolt_openings::CommitmentScheme as VerifierCommitmentScheme;
    use jolt_program::execution::{
        ExecutionBackend, JoltProgram, OwnedTrace, TraceInputs, TraceOutput, TraceRow,
    };
    use jolt_prover::akita::JoltAkitaBackend;
    use jolt_prover::{akita, JoltProverPreprocessing, ProverConfig};
    use jolt_prover_legacy::field::akita::AkitaFp128;
    use jolt_prover_legacy::host::Program;
    use jolt_prover_legacy::zkvm::packed::{
        akita_verifier_preprocessing, field_inline_one_hot_trace_setup_params, AkitaField,
        AkitaNoCurve, AkitaPackedScheme, AkitaScheduleArtifacts, AkitaScheme, AkitaTranscript,
        AkitaVc,
    };
    use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
    use jolt_prover_legacy::zkvm::program::ProgramPreprocessing as LegacyProgramPreprocessing;
    use jolt_prover_legacy::zkvm::prover::JoltProverPreprocessing as LegacyProverPreprocessing;
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
        let mut program = Program::new("eqpoly-field-guest");
        program.enable_field_inline();
        let (bytecode, memory_init, _, entry_address) = program.decode();
        let (_, _, _, io_device) = program.trace(&inputs, &[], &[]);
        let elf_contents = program.get_elf_contents().expect("elf contents");
        let preprocessed =
            LegacyProgramPreprocessing::<AkitaPackedScheme>::preprocess_with_profile(
                bytecode,
                memory_init,
                entry_address,
                program.instruction_profile(),
            )
            .expect("FR-profile packed preprocess");
        let shared: JoltSharedPreprocessing<AkitaPackedScheme> = JoltSharedPreprocessing::new(
            preprocessed,
            io_device.memory_layout.clone(),
            MAX_PADDED_TRACE_LENGTH,
        );
        let legacy_preprocessing: LegacyProverPreprocessing<
            AkitaFp128,
            AkitaNoCurve,
            AkitaPackedScheme,
        > = LegacyProverPreprocessing::new(shared);
        let jolt_program = Arc::new(JoltProgram::from_elf_bytes_with_profile(
            elf_contents,
            program.instruction_profile(),
        ));
        let memory_layout = io_device.memory_layout.clone();
        let trace_output = trace_modular(&jolt_program, &memory_layout, &inputs);
        let public_io = trace_output.device.clone();

        let config = ProverConfig::derive::<AkitaField>(
            trace_output.trace.rows(),
            &memory_layout,
            legacy_preprocessing
                .shared
                .program_meta
                .min_bytecode_address,
            legacy_preprocessing
                .shared
                .program
                .program_image_len_words(),
            MAX_PADDED_TRACE_LENGTH,
        )
        .expect("derive config");
        let log_t = config.trace_length.ilog2() as usize;
        let (setup_shape, layout_digest, one_hot_k) =
            akita::one_hot_trace_setup_shape(&config, legacy_preprocessing.shared.bytecode_size())
                .expect("OneHotTrace setup shape");
        // The setup's final arity carries the FR limb group.
        let params = field_inline_one_hot_trace_setup_params(
            setup_shape.num_vars,
            setup_shape.num_polys,
            layout_digest,
            one_hot_k,
            AkitaScheduleArtifacts::shared_from_default_directory(),
        )
        .expect("FR-on packed setup params");
        let (object_setup, verifier_setup) =
            <AkitaScheme as VerifierCommitmentScheme>::setup(params)
                .expect("the transparent packed setup must derive");
        let verifier_preprocessing =
            akita_verifier_preprocessing(&legacy_preprocessing, verifier_setup, None);

        let mut rows = trace_output.trace.rows().to_vec();
        rows.resize(config.trace_length, TraceRow::default());
        let padded_output = TraceOutput::new(
            OwnedTrace::new(rows),
            trace_output.device,
            trace_output.final_memory,
            trace_output.advice_tape,
        );
        let program_preprocessing = verifier_preprocessing
            .program
            .as_full_arc()
            .expect("full program preprocessing");
        let witness = TraceBackend::new(
            JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
            JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded_output),
        )
        .with_field_inline()
        .expect("field-inline witness view");
        let prover_preprocessing = JoltProverPreprocessing::<AkitaScheme, AkitaVc> {
            verifier: verifier_preprocessing,
            pcs_setup: object_setup,
            committed_program: None,
        };
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
