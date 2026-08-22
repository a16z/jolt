#![expect(
    clippy::expect_used,
    reason = "measurement harness: fixture errors fail loudly"
)]

use std::sync::Arc;

use common::jolt_device::MemoryLayout;
use jolt_kernels::CommitmentGrid;
use jolt_program::execution::{JoltProgram, OwnedTrace, TraceOutput};
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_prover::profile::{pad_trace, trace_modular, Workload};
use jolt_prover::ProverConfig;
use jolt_prover_legacy::host;
use jolt_prover_legacy::poly::commitment::dory::DoryCommitmentScheme;
use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
use jolt_prover_legacy::zkvm::program::ProgramPreprocessing as LegacyProgramPreprocessing;
use jolt_prover_legacy::zkvm::proof::verifier_preprocessing_from_prover;
use jolt_prover_legacy::zkvm::prover::JoltProverPreprocessing as LegacyProverPreprocessing;
use jolt_verifier::stages::{CommittedProgramSchedule, PrecommittedSchedule};
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

const SAFETY_MARGIN: f64 = 0.9;

pub struct Fixture {
    pub program: Arc<JoltProgram>,
    pub program_preprocessing: Arc<JoltProgramPreprocessing>,
    pub config: ProverConfig,
    pub log_t: usize,
    pub memory_layout: MemoryLayout,
    pub min_bytecode_address: u64,
    pub program_image_len_words: usize,
    pub bytecode_chunk_count: usize,
    trace: TraceOutput<OwnedTrace>,
}

pub struct Parts {
    pub program_preprocessing: Arc<JoltProgramPreprocessing>,
    pub config: ProverConfig,
    pub log_t: usize,
}

impl Fixture {
    pub fn build(workload: Workload, scale: u32, bytecode_chunks: usize) -> Self {
        let bench_name = workload.as_str();
        let max_trace_length = 1usize << scale;
        let input = workload.input((max_trace_length as f64 * SAFETY_MARGIN) as usize);

        let mut program = host::Program::new(&format!("{bench_name}-guest"));
        let (bytecode, init_memory_state, _, entry_address) = program.decode();
        let (_, legacy_trace, _, io_device) = program.trace(&input, &[], &[]);
        drop(legacy_trace);
        let elf_contents = program.get_elf_contents().expect("elf contents");
        let memory_layout = io_device.memory_layout.clone();

        let program_data =
            LegacyProgramPreprocessing::preprocess(bytecode, init_memory_state, entry_address)
                .expect("legacy preprocess");
        let shared =
            JoltSharedPreprocessing::new(program_data, memory_layout.clone(), max_trace_length);
        let legacy = LegacyProverPreprocessing::<
            jolt_prover_legacy::ark_bn254::Fr,
            jolt_prover_legacy::curve::Bn254Curve,
            DoryCommitmentScheme,
        >::new(shared);
        let verifier_preprocessing = verifier_preprocessing_from_prover(&legacy);
        let program_preprocessing = Arc::new(
            verifier_preprocessing
                .program
                .as_full()
                .expect("full program preprocessing")
                .clone(),
        );
        let jolt_program = Arc::new(JoltProgram::from_elf_bytes(elf_contents));

        let trace_output = trace_modular(&jolt_program, &memory_layout, &input);
        let config = ProverConfig::derive::<jolt_field::Fr>(
            trace_output.trace.rows(),
            &memory_layout,
            verifier_preprocessing.program.min_bytecode_address(),
            verifier_preprocessing.program.program_image_len_words(),
            max_trace_length,
        )
        .expect("derive config");
        let padded = pad_trace(trace_output, config.trace_length);
        let log_t = config.trace_length.ilog2() as usize;

        Self {
            program: jolt_program,
            program_preprocessing,
            config,
            log_t,
            memory_layout,
            min_bytecode_address: verifier_preprocessing.program.min_bytecode_address(),
            program_image_len_words: verifier_preprocessing.program.program_image_len_words(),
            bytecode_chunk_count: bytecode_chunks,
            trace: padded,
        }
    }

    pub fn device(&self) -> &common::jolt_device::JoltDevice {
        &self.trace.device
    }

    pub fn parts(&self) -> Parts {
        Parts {
            program_preprocessing: Arc::clone(&self.program_preprocessing),
            config: self.config,
            log_t: self.log_t,
        }
    }

    fn clone_trace(&self) -> TraceOutput<OwnedTrace> {
        TraceOutput::new(
            OwnedTrace::new(self.trace.trace.rows().to_vec()),
            self.trace.device.clone(),
            self.trace.final_memory.clone(),
            self.trace.advice_tape.clone(),
        )
    }

    pub fn base_witness(&self) -> TraceBackend<OwnedTrace> {
        TraceBackend::new(
            JoltVmWitnessConfig::new(self.log_t, self.config.ram_K, self.config.one_hot_config),
            JoltVmWitnessInputs::new(
                &self.program,
                &self.program_preprocessing,
                self.clone_trace(),
            ),
        )
    }

    pub fn advice_witness(&self) -> TraceBackend<OwnedTrace> {
        TraceBackend::new(
            JoltVmWitnessConfig::new(self.log_t, self.config.ram_K, self.config.one_hot_config)
                .include_trusted_advice(true)
                .include_untrusted_advice(true),
            JoltVmWitnessInputs::new(
                &self.program,
                &self.program_preprocessing,
                self.clone_trace(),
            ),
        )
    }

    pub fn commitment_grid(&self) -> CommitmentGrid {
        CommitmentGrid {
            total_vars: self
                .config
                .commitment_total_vars(&self.memory_layout, false, false, None),
            log_t: self.log_t,
            log_k_chunk: self.config.one_hot_config.committed_chunk_bits(),
            order: self.config.trace_polynomial_order,
        }
    }

    pub fn precommitted_schedule(&self) -> PrecommittedSchedule {
        let start_index = self
            .memory_layout
            .remapped_word_address(self.min_bytecode_address)
            .expect("program image start index") as usize;
        PrecommittedSchedule::new(
            self.config.trace_polynomial_order,
            self.log_t,
            self.config.one_hot_config.committed_chunk_bits(),
            Some(self.memory_layout.max_trusted_advice_size as usize),
            Some(self.memory_layout.max_untrusted_advice_size as usize),
            Some(CommittedProgramSchedule {
                bytecode_len: self.program_preprocessing.bytecode.code_size,
                bytecode_chunk_count: self.bytecode_chunk_count,
                program_image_len_words: self.program_image_len_words,
                program_image_start_index: start_index,
            }),
        )
        .expect("precommitted schedule")
    }
}
