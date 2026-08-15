use std::time::{Duration, Instant};

use common::constants::XLEN as RISCV_XLEN;
use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::protocols::jolt::geometry::ram::RamValCheckInit;
use jolt_claims::protocols::jolt::relations::instruction::{
    InstructionReadRafChallenges, InstructionReadRafInputClaims,
};
use jolt_claims::protocols::jolt::relations::ram::{RamValCheckChallenges, RamValCheckInputClaims};
use jolt_claims::protocols::jolt::{JoltChallengeId, JoltRelationId, TraceDimensions};
use jolt_claims::NoChallenges;
use jolt_dory::DoryScheme;
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_kernels::{JoltBackend, ProofSession, ProverInputs};
use jolt_program::execution::JoltProgram;
use jolt_prover_legacy::host;
use jolt_prover_legacy::poly::commitment::dory::DoryCommitmentScheme;
use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
use jolt_prover_legacy::zkvm::program::ProgramPreprocessing as LegacyProgramPreprocessing;
use jolt_prover_legacy::zkvm::proof::verifier_preprocessing_from_prover;
use jolt_prover_legacy::zkvm::prover::JoltProverPreprocessing as LegacyProverPreprocessing;
use jolt_sumcheck::SumcheckError;
use jolt_verifier::stages::formula_dimensions_from_parts;
use jolt_verifier::stages::stage2::ram_read_write_checking::{
    RamReadWriteChallenges, RamReadWriteChecking, RamReadWriteInputClaims,
};
use jolt_verifier::stages::stage4::registers_read_write_checking::{
    RegistersReadWriteChallenges, RegistersReadWriteChecking, RegistersReadWriteInputClaims,
};
use jolt_verifier::stages::stage5::instruction_read_raf::InstructionReadRaf;
use jolt_verifier::stages::stage6b::instruction_ra_virtualization::{
    InstructionRaVirtualization, InstructionRaVirtualizationChallenges,
    InstructionRaVirtualizationInputClaims,
};
use jolt_verifier::stages::stage6b::ram_ra_virtualization::{
    RamRaVirtualization, RamRaVirtualizationInputClaims,
};
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

use crate::profile::{pad_trace, trace_modular, BackendKind, Workload};
use crate::ProverConfig;

const SAFETY_MARGIN: f64 = 0.9;

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum VerticalRelation {
    InstructionRaVirtualization,
    InstructionReadRaf,
    RamRaVirtualization,
    RamReadWrite,
    RamValCheck,
    RegistersReadWrite,
}

impl VerticalRelation {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InstructionRaVirtualization => "instruction-ra-virtualization",
            Self::InstructionReadRaf => "instruction-read-raf",
            Self::RamRaVirtualization => "ram-ra-virtualization",
            Self::RamReadWrite => "ram-read-write",
            Self::RamValCheck => "ram-val-check",
            Self::RegistersReadWrite => "registers-read-write",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RoundPhase {
    Address,
    Handoff,
    Cycle,
}

#[derive(Debug, clap::Args)]
pub struct VerticalArgs {
    #[clap(long, value_enum)]
    pub relation: VerticalRelation,

    #[clap(long, value_enum)]
    pub name: Workload,

    #[clap(long, value_delimiter = ',', default_values_t = [16u32, 20, 22])]
    pub scales: Vec<u32>,

    #[clap(long, value_enum, default_value = "reference")]
    pub backend: BackendKind,
}

#[derive(Clone, Copy, Debug)]
pub struct VerticalTiming {
    pub log_t: usize,
    pub prepare: Duration,
    pub address: Duration,
    pub handoff: Duration,
    pub cycle: Duration,
    pub claims: Duration,
}

impl VerticalTiming {
    pub fn total(&self) -> Duration {
        self.prepare + self.address + self.handoff + self.cycle + self.claims
    }
}

#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "measurement harness: kernel errors fail loudly"
)]
fn drive_rounds<F, R>(
    kernel: &mut dyn jolt_kernels::SumcheckKernel<F, Relation = R>,
    claims: &jolt_verifier::stages::relations::SumcheckInputClaims<F, R>,
    rounds: usize,
    log_t: usize,
    prepare: Duration,
    phase: impl Fn(usize) -> RoundPhase,
) -> VerticalTiming
where
    F: jolt_field::Field + FromPrimitiveInt,
    R: jolt_verifier::stages::relations::ConcreteSumcheck<F>,
    jolt_verifier::stages::relations::SumcheckInputClaims<F, R>: jolt_claims::InputClaims<F>,
    jolt_verifier::stages::relations::SumcheckOutputClaims<F, R>: jolt_claims::OutputClaims<F>,
    jolt_verifier::stages::relations::ConcreteSumcheckChallenges<F, R>:
        jolt_claims::SumcheckChallenges<F, JoltChallengeId>,
{
    let mut claim = F::from_u64(0);
    let mut bind = None;
    let mut address = Duration::ZERO;
    let mut handoff = Duration::ZERO;
    let mut cycle = Duration::ZERO;

    for round in 0..rounds {
        let challenge = F::from_u64(17 + round as u64);
        let start = Instant::now();
        let outcome = kernel.prove_round(bind, round, claim);
        let elapsed = start.elapsed();
        match phase(round) {
            RoundPhase::Address => address += elapsed,
            RoundPhase::Handoff => handoff += elapsed,
            RoundPhase::Cycle => cycle += elapsed,
        }
        claim = match outcome {
            Ok(poly) => poly.evaluate(challenge),
            Err(SumcheckError::RoundCheckFailed { actual, .. }) => actual,
            Err(error) => panic!("vertical round {round} failed: {error:?}"),
        };
        bind = Some(challenge);
    }
    kernel
        .finish_rounds(F::from_u64(17 + rounds as u64))
        .expect("finish the vertical rounds");

    let start = Instant::now();
    let _ = kernel
        .output_claims(claims)
        .expect("vertical output claims");
    let claims_time = start.elapsed();

    VerticalTiming {
        log_t,
        prepare,
        address,
        handoff,
        cycle,
        claims: claims_time,
    }
}

#[expect(
    clippy::print_stdout,
    reason = "measurement harness: reports to stdout like the profile subcommand"
)]
pub fn run(args: &VerticalArgs) -> Vec<VerticalTiming> {
    println!(
        "{} vertical — {} backend, {} workload",
        args.relation.as_str(),
        args.backend.as_str(),
        args.name.as_str(),
    );
    println!(
        "{:>6}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}",
        "log_T", "prepare", "address", "handoff", "cycle", "claims", "total",
    );
    let mut timings = Vec::new();
    for &scale in &args.scales {
        let timing = match args.relation {
            VerticalRelation::InstructionRaVirtualization => {
                measure_instruction_ra_virtualization(args.name, scale, args.backend)
            }
            VerticalRelation::InstructionReadRaf => {
                measure_instruction_read_raf(args.name, scale, args.backend)
            }
            VerticalRelation::RamRaVirtualization => {
                measure_ram_ra_virtualization(args.name, scale, args.backend)
            }
            VerticalRelation::RamReadWrite => {
                measure_ram_read_write(args.name, scale, args.backend)
            }
            VerticalRelation::RamValCheck => measure_ram_val_check(args.name, scale, args.backend),
            VerticalRelation::RegistersReadWrite => {
                measure_registers_read_write(args.name, scale, args.backend)
            }
        };
        println!(
            "{:>6}  {:>11.3?}  {:>11.3?}  {:>11.3?}  {:>11.3?}  {:>11.3?}  {:>11.3?}",
            timing.log_t,
            timing.prepare,
            timing.address,
            timing.handoff,
            timing.cycle,
            timing.claims,
            timing.total(),
        );
        timings.push(timing);
    }
    timings
}

#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
fn measure_instruction_read_raf(
    workload: Workload,
    scale: u32,
    backend: BackendKind,
) -> VerticalTiming {
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
    let program_preprocessing = verifier_preprocessing
        .program
        .as_full()
        .expect("full program preprocessing")
        .clone();
    let jolt_program = JoltProgram::from_elf_bytes(elf_contents);

    let trace_output = trace_modular(&jolt_program, &memory_layout, &input);
    let config = ProverConfig::derive::<Fr>(
        trace_output.trace.rows(),
        &memory_layout,
        verifier_preprocessing.program.min_bytecode_address(),
        verifier_preprocessing.program.program_image_len_words(),
        max_trace_length,
    )
    .expect("derive config");
    let padded = pad_trace(trace_output, config.trace_length);
    let log_t = config.trace_length.ilog2() as usize;

    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded),
    );

    let dimensions = formula_dimensions_from_parts(
        config.one_hot_config,
        log_t,
        verifier_preprocessing.program.bytecode_len(),
        config.ram_K,
        JoltRelationId::InstructionReadRaf,
    )
    .expect("formula dimensions")
    .instruction_read_raf;
    let relation = InstructionReadRaf::<Fr>::new(dimensions);

    let point = |offset: u64| -> Vec<Fr> {
        (0..log_t)
            .map(|i| Fr::from_u64(offset + 7 * i as u64 + 3))
            .collect()
    };
    let claims = InstructionReadRafInputClaims {
        lookup_output: Fr::from_u64(0),
        left_lookup_operand: Fr::from_u64(0),
        right_lookup_operand: Fr::from_u64(0),
    };
    let points = InstructionReadRafInputClaims {
        lookup_output: point(31),
        left_lookup_operand: point(131),
        right_lookup_operand: point(231),
    };
    let challenges = InstructionReadRafChallenges {
        gamma: Fr::from_u64(101),
    };
    let inputs = || ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    let start = Instant::now();
    let mut kernel = selected
        .instruction_read_raf
        .prepare(&mut session, &witness, inputs())
        .expect("prepare the stage-5 read-RAF kernel");
    let prepare = start.elapsed();

    let address_bits = 2 * RISCV_XLEN;
    drive_rounds(
        &mut *kernel,
        &claims,
        address_bits + log_t,
        log_t,
        prepare,
        |round| match round.cmp(&address_bits) {
            std::cmp::Ordering::Less => RoundPhase::Address,
            std::cmp::Ordering::Equal => RoundPhase::Handoff,
            std::cmp::Ordering::Greater => RoundPhase::Cycle,
        },
    )
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
fn measure_instruction_ra_virtualization(
    workload: Workload,
    scale: u32,
    backend: BackendKind,
) -> VerticalTiming {
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
    let program_preprocessing = verifier_preprocessing
        .program
        .as_full()
        .expect("full program preprocessing")
        .clone();
    let jolt_program = JoltProgram::from_elf_bytes(elf_contents);

    let trace_output = trace_modular(&jolt_program, &memory_layout, &input);
    let config = ProverConfig::derive::<Fr>(
        trace_output.trace.rows(),
        &memory_layout,
        verifier_preprocessing.program.min_bytecode_address(),
        verifier_preprocessing.program.program_image_len_words(),
        max_trace_length,
    )
    .expect("derive config");
    let padded = pad_trace(trace_output, config.trace_length);
    let log_t = config.trace_length.ilog2() as usize;

    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded),
    );

    let dimensions = formula_dimensions_from_parts(
        config.one_hot_config,
        log_t,
        verifier_preprocessing.program.bytecode_len(),
        config.ram_K,
        JoltRelationId::InstructionRaVirtualization,
    )
    .expect("formula dimensions")
    .instruction_ra_virtualization;
    let chunk_bits = config.one_hot_config.committed_chunk_bits();
    let instruction_address: Vec<Fr> = (0..dimensions.num_committed_ra_polys() * chunk_bits)
        .map(|i| Fr::from_u64(29 + 5 * i as u64))
        .collect();
    let instruction_read_raf_cycle: Vec<Fr> = (0..log_t)
        .map(|i| Fr::from_u64(37 + 7 * i as u64))
        .collect();
    let relation = InstructionRaVirtualization::<Fr>::new(
        dimensions,
        instruction_address,
        instruction_read_raf_cycle,
        chunk_bits,
    );

    let claims = InstructionRaVirtualizationInputClaims {
        instruction_ra: vec![Fr::from_u64(0); dimensions.num_virtual_ra_polys()],
    };
    let points = InstructionRaVirtualizationInputClaims {
        instruction_ra: (0..dimensions.num_virtual_ra_polys())
            .map(|virtual_index| {
                (0..dimensions.num_committed_per_virtual() * chunk_bits + log_t)
                    .map(|bit| Fr::from_u64(11 + 3 * (virtual_index * 97 + bit) as u64))
                    .collect()
            })
            .collect(),
    };
    let challenges = InstructionRaVirtualizationChallenges {
        gamma: Fr::from_u64(101),
    };
    let inputs = || ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    let start = Instant::now();
    let mut kernel = selected
        .instruction_ra_virtualization
        .prepare(&mut session, &witness, inputs())
        .expect("prepare the stage-6b RA virtualization kernel");
    let prepare = start.elapsed();

    drive_rounds(&mut *kernel, &claims, log_t, log_t, prepare, |_| {
        RoundPhase::Cycle
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
fn measure_ram_ra_virtualization(
    workload: Workload,
    scale: u32,
    backend: BackendKind,
) -> VerticalTiming {
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
    let program_preprocessing = verifier_preprocessing
        .program
        .as_full()
        .expect("full program preprocessing")
        .clone();
    let jolt_program = JoltProgram::from_elf_bytes(elf_contents);

    let trace_output = trace_modular(&jolt_program, &memory_layout, &input);
    let config = ProverConfig::derive::<Fr>(
        trace_output.trace.rows(),
        &memory_layout,
        verifier_preprocessing.program.min_bytecode_address(),
        verifier_preprocessing.program.program_image_len_words(),
        max_trace_length,
    )
    .expect("derive config");
    let padded = pad_trace(trace_output, config.trace_length);
    let log_t = config.trace_length.ilog2() as usize;

    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded),
    );

    let ram_log_k = config.ram_K.ilog2() as usize;
    let dimensions = formula_dimensions_from_parts(
        config.one_hot_config,
        log_t,
        verifier_preprocessing.program.bytecode_len(),
        config.ram_K,
        JoltRelationId::RamRaVirtualization,
    )
    .expect("formula dimensions")
    .ram_ra_virtualization;
    let chunk_bits = config.one_hot_config.committed_chunk_bits();
    let ram_reduced_address: Vec<Fr> = (0..ram_log_k)
        .map(|i| Fr::from_u64(29 + 5 * i as u64))
        .collect();
    let ram_reduced_cycle: Vec<Fr> = (0..log_t)
        .map(|i| Fr::from_u64(37 + 7 * i as u64))
        .collect();
    let relation = RamRaVirtualization::<Fr>::new(
        dimensions,
        ram_reduced_address.clone(),
        ram_reduced_cycle.clone(),
        chunk_bits,
    );

    let claims = RamRaVirtualizationInputClaims {
        ram_ra_reduced: Fr::from_u64(0),
    };
    let points = RamRaVirtualizationInputClaims {
        ram_ra_reduced: [ram_reduced_address.as_slice(), ram_reduced_cycle.as_slice()].concat(),
    };
    let challenges = NoChallenges::<Fr>::default();
    let inputs = || ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    let start = Instant::now();
    let mut kernel = selected
        .ram_ra_virtualization
        .prepare(&mut session, &witness, inputs())
        .expect("prepare the stage-6b RAM RA virtualization kernel");
    let prepare = start.elapsed();

    drive_rounds(&mut *kernel, &claims, log_t, log_t, prepare, |_| {
        RoundPhase::Cycle
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
fn measure_ram_val_check(workload: Workload, scale: u32, backend: BackendKind) -> VerticalTiming {
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
    let program_preprocessing = verifier_preprocessing
        .program
        .as_full()
        .expect("full program preprocessing")
        .clone();
    let jolt_program = JoltProgram::from_elf_bytes(elf_contents);

    let trace_output = trace_modular(&jolt_program, &memory_layout, &input);
    let config = ProverConfig::derive::<Fr>(
        trace_output.trace.rows(),
        &memory_layout,
        verifier_preprocessing.program.min_bytecode_address(),
        verifier_preprocessing.program.program_image_len_words(),
        max_trace_length,
    )
    .expect("derive config");
    let padded = pad_trace(trace_output, config.trace_length);
    let log_t = config.trace_length.ilog2() as usize;

    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded),
    );

    let ram_log_k = config.ram_K.ilog2() as usize;
    let relation = jolt_verifier::stages::stage4::ram_val_check::RamValCheck::<Fr>::new(
        TraceDimensions::new(log_t),
        ram_log_k,
        RamValCheckInit::full(Fr::from_u64(0)),
    );

    let point: Vec<Fr> = (0..ram_log_k + log_t)
        .map(|i| Fr::from_u64(31 + 7 * i as u64 + 3))
        .collect();
    let claims = RamValCheckInputClaims {
        ram_val: Fr::from_u64(0),
        ram_val_final: Fr::from_u64(0),
        untrusted_advice: None,
        trusted_advice: None,
        program_image: None,
    };
    let points = RamValCheckInputClaims {
        ram_val: point.clone(),
        ram_val_final: point,
        untrusted_advice: None,
        trusted_advice: None,
        program_image: None,
    };
    let challenges = RamValCheckChallenges {
        gamma: Fr::from_u64(101),
    };
    let inputs = || ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    let start = Instant::now();
    let mut kernel = selected
        .ram_val_check
        .prepare(&mut session, &witness, inputs())
        .expect("prepare the stage-4 RAM value-check kernel");
    let prepare = start.elapsed();

    drive_rounds(&mut *kernel, &claims, log_t, log_t, prepare, |_| {
        RoundPhase::Cycle
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
fn measure_ram_read_write(workload: Workload, scale: u32, backend: BackendKind) -> VerticalTiming {
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
    let program_preprocessing = verifier_preprocessing
        .program
        .as_full()
        .expect("full program preprocessing")
        .clone();
    let jolt_program = JoltProgram::from_elf_bytes(elf_contents);

    let trace_output = trace_modular(&jolt_program, &memory_layout, &input);
    let config = ProverConfig::derive::<Fr>(
        trace_output.trace.rows(),
        &memory_layout,
        verifier_preprocessing.program.min_bytecode_address(),
        verifier_preprocessing.program.program_image_len_words(),
        max_trace_length,
    )
    .expect("derive config");
    let padded = pad_trace(trace_output, config.trace_length);
    let log_t = config.trace_length.ilog2() as usize;
    let ram_log_k = config.ram_K.ilog2() as usize;

    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded),
    );

    let ram_dimensions = config.rw_config.ram_dimensions(log_t, ram_log_k);
    let tau_low: Vec<Fr> = (0..log_t)
        .map(|i| Fr::from_u64(53 + 11 * i as u64))
        .collect();
    let relation = RamReadWriteChecking::<Fr>::new(ram_dimensions, ram_log_k, tau_low);

    let point = |offset: u64| -> Vec<Fr> {
        (0..log_t)
            .map(|i| Fr::from_u64(offset + 7 * i as u64 + 3))
            .collect()
    };
    let claims = RamReadWriteInputClaims {
        ram_read_value: Fr::from_u64(0),
        ram_write_value: Fr::from_u64(0),
    };
    let points = RamReadWriteInputClaims {
        ram_read_value: point(41),
        ram_write_value: point(141),
    };
    let challenges = RamReadWriteChallenges {
        gamma: Fr::from_u64(103),
    };
    let inputs = || ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    let start = Instant::now();
    let mut kernel = selected
        .ram_read_write
        .prepare(&mut session, &witness, inputs())
        .expect("prepare the stage-2 RAM read-write kernel");
    let prepare = start.elapsed();

    let rounds = jolt_claims::SymbolicSumcheck::rounds(
        jolt_verifier::stages::relations::ConcreteSumcheck::symbolic(&relation),
    );
    drive_rounds(&mut *kernel, &claims, rounds, log_t, prepare, |round| {
        if round < log_t {
            RoundPhase::Cycle
        } else {
            RoundPhase::Address
        }
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
fn measure_registers_read_write(
    workload: Workload,
    scale: u32,
    backend: BackendKind,
) -> VerticalTiming {
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
    let program_preprocessing = verifier_preprocessing
        .program
        .as_full()
        .expect("full program preprocessing")
        .clone();
    let jolt_program = JoltProgram::from_elf_bytes(elf_contents);

    let trace_output = trace_modular(&jolt_program, &memory_layout, &input);
    let config = ProverConfig::derive::<Fr>(
        trace_output.trace.rows(),
        &memory_layout,
        verifier_preprocessing.program.min_bytecode_address(),
        verifier_preprocessing.program.program_image_len_words(),
        max_trace_length,
    )
    .expect("derive config");
    let padded = pad_trace(trace_output, config.trace_length);
    let log_t = config.trace_length.ilog2() as usize;

    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded),
    );

    let register_dimensions = config
        .rw_config
        .register_dimensions(log_t, REGISTER_ADDRESS_BITS);
    let relation = RegistersReadWriteChecking::<Fr>::new(register_dimensions);

    let point = |offset: u64| -> Vec<Fr> {
        (0..log_t)
            .map(|i| Fr::from_u64(offset + 7 * i as u64 + 3))
            .collect()
    };
    let claims = RegistersReadWriteInputClaims {
        rd_write_value: Fr::from_u64(0),
        rs1_value: Fr::from_u64(0),
        rs2_value: Fr::from_u64(0),
    };
    let points = RegistersReadWriteInputClaims {
        rd_write_value: point(31),
        rs1_value: point(131),
        rs2_value: point(231),
    };
    let challenges = RegistersReadWriteChallenges {
        gamma: Fr::from_u64(101),
    };
    let inputs = || ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    let start = Instant::now();
    let mut kernel = selected
        .registers_read_write
        .prepare(&mut session, &witness, inputs())
        .expect("prepare the stage-4 registers read-write kernel");
    let prepare = start.elapsed();

    let rounds = jolt_claims::SymbolicSumcheck::rounds(
        jolt_verifier::stages::relations::ConcreteSumcheck::symbolic(&relation),
    );
    drive_rounds(&mut *kernel, &claims, rounds, log_t, prepare, |round| {
        if round < log_t {
            RoundPhase::Cycle
        } else {
            RoundPhase::Address
        }
    })
}
