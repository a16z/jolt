use std::time::{Duration, Instant};

use common::constants::XLEN as RISCV_XLEN;
use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::NUM_BYTECODE_VAL_STAGES;
use jolt_claims::protocols::jolt::geometry::claim_reductions::hamming_weight::HammingWeightClaimReductionDimensions;
use jolt_claims::protocols::jolt::geometry::dimensions::REGISTER_ADDRESS_BITS;
use jolt_claims::protocols::jolt::geometry::ram::RamRafEvaluationDimensions;
use jolt_claims::protocols::jolt::geometry::ram::RamValCheckInit;
use jolt_claims::protocols::jolt::geometry::spartan::{
    SpartanOuterDimensions, SpartanProductDimensions,
};
use jolt_claims::protocols::jolt::relations::booleanity::{
    BooleanityAddressPhaseChallenges, BooleanityAddressPhaseInputClaims,
};
use jolt_claims::protocols::jolt::relations::bytecode::{
    BytecodeReadRafAddressPhaseChallenges, BytecodeReadRafAddressPhaseInputClaims,
};
use jolt_claims::protocols::jolt::relations::claim_reductions::hamming_weight::{
    HammingWeightClaimReductionChallenges, HammingWeightClaimReductionInputClaims,
};
use jolt_claims::protocols::jolt::relations::claim_reductions::increments::{
    IncClaimReductionChallenges, IncClaimReductionInputClaims,
};
use jolt_claims::protocols::jolt::relations::claim_reductions::instruction::{
    InstructionClaimReductionChallenges, InstructionClaimReductionInputClaims,
};
use jolt_claims::protocols::jolt::relations::claim_reductions::registers::{
    RegistersClaimReductionChallenges, RegistersClaimReductionInputClaims,
};
use jolt_claims::protocols::jolt::relations::instruction::{
    InstructionInputChallenges, InstructionInputInputClaims, InstructionReadRafChallenges,
    InstructionReadRafInputClaims,
};
use jolt_claims::protocols::jolt::relations::ram::{
    RamHammingBooleanityInputClaims, RamOutputCheckChallenges, RamOutputCheckInputClaims,
    RamRaClaimReductionChallenges, RamRaClaimReductionInputClaims, RamRafEvaluationInputClaims,
    RamValCheckChallenges, RamValCheckInputClaims,
};
use jolt_claims::protocols::jolt::relations::registers::RegistersValEvaluationInputClaims;
use jolt_claims::protocols::jolt::relations::spartan::{
    ProductRemainderInputClaims, SpartanShiftChallenges, SpartanShiftInputClaims,
};
use jolt_claims::protocols::jolt::{JoltChallengeId, JoltRelationId, TraceDimensions};
use jolt_claims::NoChallenges;
use jolt_dory::DoryScheme;
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_kernels::{JoltBackend, ProofSession, ProverInputs};
use jolt_program::execution::{JoltProgram, OwnedTrace, TraceOutput};
use jolt_program::preprocess::{JoltProgramPreprocessing, PublicIoMemory};
use jolt_prover_legacy::host;
use jolt_prover_legacy::poly::commitment::dory::DoryCommitmentScheme;
use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
use jolt_prover_legacy::zkvm::program::ProgramPreprocessing as LegacyProgramPreprocessing;
use jolt_prover_legacy::zkvm::proof::verifier_preprocessing_from_prover;
use jolt_prover_legacy::zkvm::prover::JoltProverPreprocessing as LegacyProverPreprocessing;
use jolt_sumcheck::SumcheckError;
use jolt_verifier::stages::formula_dimensions_from_parts;
use jolt_verifier::stages::stage1::outer_remainder::{
    outer_remainder_input_values_from_uniskip_output, OuterRemainder, OuterRemainderInputClaims,
};
use jolt_verifier::stages::stage2::instruction_claim_reduction::InstructionClaimReduction;
use jolt_verifier::stages::stage2::product_remainder::{
    product_remainder_input_values_from_uniskip_output, ProductRemainder,
};
use jolt_verifier::stages::stage2::ram_output_check::RamOutputCheck;
use jolt_verifier::stages::stage2::ram_raf_evaluation::RamRafEvaluation;
use jolt_verifier::stages::stage2::ram_read_write_checking::{
    RamReadWriteChallenges, RamReadWriteChecking, RamReadWriteInputClaims,
};
use jolt_verifier::stages::stage3::outputs::{
    InstructionInput, RegistersClaimReduction, SpartanShift,
};
use jolt_verifier::stages::stage4::registers_read_write_checking::{
    RegistersReadWriteChallenges, RegistersReadWriteChecking, RegistersReadWriteInputClaims,
};
use jolt_verifier::stages::stage5::instruction_read_raf::InstructionReadRaf;
use jolt_verifier::stages::stage5::ram_ra_claim_reduction::RamRaClaimReduction;
use jolt_verifier::stages::stage5::registers_val_evaluation::RegistersValEvaluation;
use jolt_verifier::stages::stage6a::booleanity::BooleanityAddressPhase;
use jolt_verifier::stages::stage6a::bytecode_read_raf::{
    BytecodeReadRafAddressPhase, BytecodeStagePoints,
};
use jolt_verifier::stages::stage6b::booleanity::{
    Booleanity, BooleanityCyclePhaseChallenges, BooleanityInputClaims,
};
use jolt_verifier::stages::stage6b::bytecode_read_raf::{
    BytecodeReadRafCommittedCycleInputs, BytecodeReadRafCycle,
    BytecodeReadRafCyclePhaseCommittedChallenges, BytecodeReadRafInputClaims,
    READ_RAF_CYCLE_STAGES,
};
use jolt_verifier::stages::stage6b::inc_claim_reduction::IncClaimReduction;
use jolt_verifier::stages::stage6b::instruction_ra_virtualization::{
    InstructionRaVirtualization, InstructionRaVirtualizationChallenges,
    InstructionRaVirtualizationInputClaims,
};
use jolt_verifier::stages::stage6b::ram_hamming_booleanity::RamHammingBooleanity;
use jolt_verifier::stages::stage6b::ram_ra_virtualization::{
    RamRaVirtualization, RamRaVirtualizationInputClaims,
};
use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::HammingWeightClaimReduction;
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

use crate::profile::{pad_trace, trace_modular, BackendKind, Workload};
use crate::ProverConfig;

const SAFETY_MARGIN: f64 = 0.9;

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum VerticalRelation {
    BooleanityAddress,
    BooleanityCycle,
    BytecodeReadRafAddress,
    BytecodeReadRafCycle,
    HammingWeightClaimReduction,
    IncClaimReduction,
    InstructionClaimReduction,
    InstructionInput,
    InstructionRaVirtualization,
    InstructionReadRaf,
    RamHammingBooleanity,
    RamOutputCheck,
    RamRaClaimReduction,
    RamRafEvaluation,
    RamRaVirtualization,
    RamReadWrite,
    RamValCheck,
    RegistersClaimReduction,
    RegistersReadWrite,
    RegistersValEvaluation,
    SpartanOuter,
    SpartanProduct,
    SpartanShift,
}

impl VerticalRelation {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::BooleanityAddress => "booleanity-address",
            Self::BooleanityCycle => "booleanity-cycle",
            Self::BytecodeReadRafAddress => "bytecode-read-raf-address",
            Self::BytecodeReadRafCycle => "bytecode-read-raf-cycle",
            Self::HammingWeightClaimReduction => "hamming-weight-claim-reduction",
            Self::IncClaimReduction => "inc-claim-reduction",
            Self::InstructionClaimReduction => "instruction-claim-reduction",
            Self::InstructionInput => "instruction-input",
            Self::InstructionRaVirtualization => "instruction-ra-virtualization",
            Self::InstructionReadRaf => "instruction-read-raf",
            Self::RamHammingBooleanity => "ram-hamming-booleanity",
            Self::RamOutputCheck => "ram-output-check",
            Self::RamRaClaimReduction => "ram-ra-claim-reduction",
            Self::RamRafEvaluation => "ram-raf-evaluation",
            Self::RamRaVirtualization => "ram-ra-virtualization",
            Self::RamReadWrite => "ram-read-write",
            Self::RamValCheck => "ram-val-check",
            Self::RegistersClaimReduction => "registers-claim-reduction",
            Self::RegistersReadWrite => "registers-read-write",
            Self::RegistersValEvaluation => "registers-val-evaluation",
            Self::SpartanOuter => "spartan-outer",
            Self::SpartanProduct => "spartan-product",
            Self::SpartanShift => "spartan-shift",
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

struct VerticalFixture {
    program: JoltProgram,
    program_preprocessing: JoltProgramPreprocessing,
    config: ProverConfig,
    log_t: usize,
    trace: TraceOutput<OwnedTrace>,
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture errors fail loudly"
)]
fn fixture(workload: Workload, scale: u32) -> VerticalFixture {
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

    VerticalFixture {
        program: jolt_program,
        program_preprocessing,
        config,
        log_t,
        trace: padded,
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
            VerticalRelation::BooleanityCycle => {
                measure_booleanity_cycle(args.name, scale, args.backend)
            }
            VerticalRelation::BytecodeReadRafCycle => {
                measure_bytecode_read_raf_cycle(args.name, scale, args.backend)
            }
            VerticalRelation::BooleanityAddress => {
                measure_booleanity_address(args.name, scale, args.backend)
            }
            VerticalRelation::BytecodeReadRafAddress => {
                measure_bytecode_read_raf_address(args.name, scale, args.backend)
            }
            VerticalRelation::HammingWeightClaimReduction => {
                measure_hamming_weight_claim_reduction(args.name, scale, args.backend)
            }
            VerticalRelation::IncClaimReduction => {
                measure_inc_claim_reduction(args.name, scale, args.backend)
            }
            VerticalRelation::InstructionClaimReduction => {
                measure_instruction_claim_reduction(args.name, scale, args.backend)
            }
            VerticalRelation::InstructionInput => {
                measure_instruction_input(args.name, scale, args.backend)
            }
            VerticalRelation::InstructionRaVirtualization => {
                measure_instruction_ra_virtualization(args.name, scale, args.backend)
            }
            VerticalRelation::RamHammingBooleanity => {
                measure_ram_hamming_booleanity(args.name, scale, args.backend)
            }
            VerticalRelation::RamOutputCheck => {
                measure_ram_output_check(args.name, scale, args.backend)
            }
            VerticalRelation::RamRaClaimReduction => {
                measure_ram_ra_claim_reduction(args.name, scale, args.backend)
            }
            VerticalRelation::RamRafEvaluation => {
                measure_ram_raf_evaluation(args.name, scale, args.backend)
            }
            VerticalRelation::RegistersValEvaluation => {
                measure_registers_val_evaluation(args.name, scale, args.backend)
            }
            VerticalRelation::RegistersClaimReduction => {
                measure_registers_claim_reduction(args.name, scale, args.backend)
            }
            VerticalRelation::SpartanShift => measure_spartan_shift(args.name, scale, args.backend),
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
            VerticalRelation::SpartanOuter => measure_spartan_outer(args.name, scale, args.backend),
            VerticalRelation::SpartanProduct => {
                measure_spartan_product(args.name, scale, args.backend)
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
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
fn measure_instruction_read_raf(
    workload: Workload,
    scale: u32,
    backend: BackendKind,
) -> VerticalTiming {
    let VerticalFixture {
        program: jolt_program,
        program_preprocessing,
        config,
        log_t,
        trace: padded,
    } = fixture(workload, scale);

    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded),
    );

    let dimensions = formula_dimensions_from_parts(
        config.one_hot_config,
        log_t,
        program_preprocessing.bytecode.code_size,
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
    let VerticalFixture {
        program: jolt_program,
        program_preprocessing,
        config,
        log_t,
        trace: padded,
    } = fixture(workload, scale);

    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded),
    );

    let dimensions = formula_dimensions_from_parts(
        config.one_hot_config,
        log_t,
        program_preprocessing.bytecode.code_size,
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
fn measure_booleanity_cycle(
    workload: Workload,
    scale: u32,
    backend: BackendKind,
) -> VerticalTiming {
    let VerticalFixture {
        program: jolt_program,
        program_preprocessing,
        config,
        log_t,
        trace: padded,
    } = fixture(workload, scale);

    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded),
    );

    let chunk_bits = config.one_hot_config.committed_chunk_bits();
    let layout = formula_dimensions_from_parts(
        config.one_hot_config,
        log_t,
        program_preprocessing.bytecode.code_size,
        config.ram_K,
        JoltRelationId::Booleanity,
    )
    .expect("formula dimensions")
    .ra_layout;
    let dimensions = BooleanityDimensions::new(layout, log_t, chunk_bits);

    let r_address: Vec<Fr> = (0..chunk_bits)
        .map(|i| Fr::from_u64(23 + 3 * i as u64))
        .collect();
    let reference_address: Vec<Fr> = (0..chunk_bits)
        .map(|i| Fr::from_u64(29 + 5 * i as u64))
        .collect();
    let reference_cycle: Vec<Fr> = (0..log_t)
        .map(|i| Fr::from_u64(37 + 7 * i as u64))
        .collect();
    let relation = Booleanity::<Fr>::new(
        dimensions,
        r_address.clone(),
        reference_address,
        reference_cycle.clone(),
    );

    let claims = BooleanityInputClaims {
        address_phase: Fr::from_u64(0),
    };
    let points = BooleanityInputClaims {
        address_phase: [r_address.as_slice(), reference_cycle.as_slice()].concat(),
    };
    let challenges = BooleanityCyclePhaseChallenges {
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
        .booleanity_cycle
        .prepare(&mut session, &witness, inputs())
        .expect("prepare the stage-6b booleanity cycle-phase kernel");
    let prepare = start.elapsed();

    drive_rounds(&mut *kernel, &claims, log_t, log_t, prepare, |_| {
        RoundPhase::Cycle
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
fn measure_bytecode_read_raf_cycle(
    workload: Workload,
    scale: u32,
    backend: BackendKind,
) -> VerticalTiming {
    let VerticalFixture {
        program: jolt_program,
        program_preprocessing,
        config,
        log_t,
        trace: padded,
    } = fixture(workload, scale);

    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded),
    );

    let chunk_bits = config.one_hot_config.committed_chunk_bits();
    let dimensions = formula_dimensions_from_parts(
        config.one_hot_config,
        log_t,
        program_preprocessing.bytecode.code_size,
        config.ram_K,
        JoltRelationId::BytecodeReadRaf,
    )
    .expect("formula dimensions")
    .bytecode_read_raf;
    let log_k = dimensions.log_k();

    let r_address: Vec<Fr> = (0..log_k)
        .map(|i| Fr::from_u64(23 + 3 * i as u64))
        .collect();
    let stage_cycle_points: [Vec<Fr>; READ_RAF_CYCLE_STAGES] = core::array::from_fn(|stage| {
        (0..log_t)
            .map(|i| Fr::from_u64(37 + 7 * i as u64 + 101 * stage as u64))
            .collect()
    });
    let relation = BytecodeReadRafCycle::<Fr>::committed(BytecodeReadRafCommittedCycleInputs {
        dimensions,
        r_address: r_address.clone(),
        stage_cycle_points,
        entry_bytecode_index: (1usize << log_k) - 2,
        committed_chunk_bits: chunk_bits,
        val_stages: (0..NUM_BYTECODE_VAL_STAGES)
            .map(|stage| Fr::from_u64(53 + 11 * stage as u64))
            .collect(),
    });

    let claims = BytecodeReadRafInputClaims {
        address_phase: Fr::from_u64(0),
    };
    let points = BytecodeReadRafInputClaims {
        address_phase: r_address,
    };
    let challenges = BytecodeReadRafCyclePhaseCommittedChallenges {
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
        .bytecode_read_raf_cycle
        .prepare(&mut session, &witness, inputs())
        .expect("prepare the stage-6b bytecode read-RAF cycle-phase kernel");
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
    let VerticalFixture {
        program: jolt_program,
        program_preprocessing,
        config,
        log_t,
        trace: padded,
    } = fixture(workload, scale);

    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded),
    );

    let ram_log_k = config.ram_K.ilog2() as usize;
    let dimensions = formula_dimensions_from_parts(
        config.one_hot_config,
        log_t,
        program_preprocessing.bytecode.code_size,
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
    let VerticalFixture {
        program: jolt_program,
        program_preprocessing,
        config,
        log_t,
        trace: padded,
    } = fixture(workload, scale);

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
    let VerticalFixture {
        program: jolt_program,
        program_preprocessing,
        config,
        log_t,
        trace: padded,
    } = fixture(workload, scale);

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

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
fn measure_spartan_outer(workload: Workload, scale: u32, backend: BackendKind) -> VerticalTiming {
    let VerticalFixture {
        program: jolt_program,
        program_preprocessing,
        config,
        log_t,
        trace: padded,
    } = fixture(workload, scale);

    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded),
    );

    let tau: Vec<Fr> = (0..log_t + 2)
        .map(|i| Fr::from_u64(37 + 7 * i as u64))
        .collect();
    let uniskip_challenge = Fr::from_u64(101);

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    let start = Instant::now();
    selected
        .spartan_outer_uniskip
        .prepare(&mut session, log_t, &tau, &witness)
        .expect("prepare the stage-1 Spartan outer uni-skip kernel");
    let prepare = start.elapsed();

    let start = Instant::now();
    let _ = selected
        .spartan_outer_uniskip
        .first_round_poly(&mut session, &[])
        .expect("the stage-1 Spartan outer uni-skip first-round polynomial");
    let uniskip_poly = start.elapsed();

    let relation = OuterRemainder::new(
        SpartanOuterDimensions::rv64(log_t),
        tau.clone(),
        uniskip_challenge,
    );
    let claims = outer_remainder_input_values_from_uniskip_output(Fr::from_u64(0));
    let points = OuterRemainderInputClaims {
        outer_uniskip: Vec::new(),
    };
    let challenges = NoChallenges::default();
    let inputs = ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let start = Instant::now();
    let mut kernel = selected
        .spartan_outer_remainder
        .prepare(&mut session, &witness, inputs)
        .expect("prepare the stage-1 Spartan outer remainder kernel");
    let remainder_prepare = start.elapsed();

    let rounds = jolt_claims::SymbolicSumcheck::rounds(
        jolt_verifier::stages::relations::ConcreteSumcheck::symbolic(&relation),
    );
    let mut timing = drive_rounds(&mut *kernel, &claims, rounds, log_t, prepare, |_| {
        RoundPhase::Cycle
    });
    timing.handoff = uniskip_poly;
    timing.address = remainder_prepare;
    timing
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
fn measure_spartan_shift(workload: Workload, scale: u32, backend: BackendKind) -> VerticalTiming {
    let VerticalFixture {
        program: jolt_program,
        program_preprocessing,
        config,
        log_t,
        trace: padded,
    } = fixture(workload, scale);

    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded),
    );

    let product_tau_low: Vec<Fr> = (0..log_t)
        .map(|i| Fr::from_u64(23 + 3 * i as u64))
        .collect();
    let product_remainder_point: Vec<Fr> = (0..log_t)
        .map(|i| Fr::from_u64(37 + 7 * i as u64))
        .collect();
    let relation = SpartanShift::<Fr>::new(
        TraceDimensions::new(log_t),
        product_tau_low,
        product_remainder_point,
    );

    let claims = SpartanShiftInputClaims::default();
    let points = SpartanShiftInputClaims::default();
    let challenges = SpartanShiftChallenges {
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
        .spartan_shift
        .prepare(&mut session, &witness, inputs())
        .expect("prepare the stage-3 Spartan shift kernel");
    let prepare = start.elapsed();

    drive_rounds(&mut *kernel, &claims, log_t, log_t, prepare, |_| {
        RoundPhase::Cycle
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
fn measure_instruction_input(
    workload: Workload,
    scale: u32,
    backend: BackendKind,
) -> VerticalTiming {
    let VerticalFixture {
        program: jolt_program,
        program_preprocessing,
        config,
        log_t,
        trace: padded,
    } = fixture(workload, scale);

    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded),
    );

    let product_remainder_point: Vec<Fr> = (0..log_t)
        .map(|i| Fr::from_u64(37 + 7 * i as u64))
        .collect();
    let relation =
        InstructionInput::<Fr>::new(TraceDimensions::new(log_t), product_remainder_point);

    let claims = InstructionInputInputClaims::default();
    let points = InstructionInputInputClaims::default();
    let challenges = InstructionInputChallenges {
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
        .instruction_input
        .prepare(&mut session, &witness, inputs())
        .expect("prepare the stage-3 instruction input-virtualization kernel");
    let prepare = start.elapsed();

    drive_rounds(&mut *kernel, &claims, log_t, log_t, prepare, |_| {
        RoundPhase::Cycle
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
fn measure_registers_claim_reduction(
    workload: Workload,
    scale: u32,
    backend: BackendKind,
) -> VerticalTiming {
    let VerticalFixture {
        program: jolt_program,
        program_preprocessing,
        config,
        log_t,
        trace: padded,
    } = fixture(workload, scale);

    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded),
    );

    let product_tau_low: Vec<Fr> = (0..log_t)
        .map(|i| Fr::from_u64(23 + 3 * i as u64))
        .collect();
    let relation = RegistersClaimReduction::<Fr>::new(TraceDimensions::new(log_t), product_tau_low);

    let claims = RegistersClaimReductionInputClaims::default();
    let points = RegistersClaimReductionInputClaims::default();
    let challenges = RegistersClaimReductionChallenges {
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
        .registers_claim_reduction
        .prepare(&mut session, &witness, inputs())
        .expect("prepare the stage-3 registers claim-reduction kernel");
    let prepare = start.elapsed();

    drive_rounds(&mut *kernel, &claims, log_t, log_t, prepare, |_| {
        RoundPhase::Cycle
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
fn measure_instruction_claim_reduction(
    workload: Workload,
    scale: u32,
    backend: BackendKind,
) -> VerticalTiming {
    let VerticalFixture {
        program: jolt_program,
        program_preprocessing,
        config,
        log_t,
        trace: padded,
    } = fixture(workload, scale);

    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded),
    );

    let tau_low: Vec<Fr> = (0..log_t)
        .map(|i| Fr::from_u64(23 + 3 * i as u64))
        .collect();
    let relation = InstructionClaimReduction::<Fr>::new(TraceDimensions::new(log_t), tau_low);

    let claims = InstructionClaimReductionInputClaims::default();
    let points = InstructionClaimReductionInputClaims::default();
    let challenges = InstructionClaimReductionChallenges {
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
        .instruction_claim_reduction
        .prepare(&mut session, &witness, inputs())
        .expect("prepare the stage-2 instruction claim-reduction kernel");
    let prepare = start.elapsed();

    drive_rounds(&mut *kernel, &claims, log_t, log_t, prepare, |_| {
        RoundPhase::Cycle
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
fn measure_inc_claim_reduction(
    workload: Workload,
    scale: u32,
    backend: BackendKind,
) -> VerticalTiming {
    let VerticalFixture {
        program: jolt_program,
        program_preprocessing,
        config,
        log_t,
        trace: padded,
    } = fixture(workload, scale);

    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded),
    );

    let cycle_point = |offset: u64| -> Vec<Fr> {
        (0..log_t)
            .map(|i| Fr::from_u64(23 + 3 * i as u64 + offset))
            .collect()
    };
    let relation = IncClaimReduction::<Fr>::new(
        TraceDimensions::new(log_t),
        cycle_point(0),
        cycle_point(101),
        cycle_point(202),
        cycle_point(303),
    );

    let claims = IncClaimReductionInputClaims::default();
    let points = IncClaimReductionInputClaims::default();
    let challenges = IncClaimReductionChallenges {
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
        .inc_claim_reduction
        .prepare(&mut session, &witness, inputs())
        .expect("prepare the stage-6b increment claim-reduction kernel");
    let prepare = start.elapsed();

    drive_rounds(&mut *kernel, &claims, log_t, log_t, prepare, |_| {
        RoundPhase::Cycle
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
fn measure_ram_hamming_booleanity(
    workload: Workload,
    scale: u32,
    backend: BackendKind,
) -> VerticalTiming {
    let VerticalFixture {
        program: jolt_program,
        program_preprocessing,
        config,
        log_t,
        trace: padded,
    } = fixture(workload, scale);

    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded),
    );

    let stage1_cycle_binding: Vec<Fr> = (0..log_t)
        .map(|i| Fr::from_u64(23 + 3 * i as u64))
        .collect();
    let relation =
        RamHammingBooleanity::<Fr>::new(TraceDimensions::new(log_t), stage1_cycle_binding);

    let claims = RamHammingBooleanityInputClaims::default();
    let points = RamHammingBooleanityInputClaims::default();
    let challenges = NoChallenges::default();
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
        .ram_hamming_booleanity
        .prepare(&mut session, &witness, inputs())
        .expect("prepare the stage-6b RAM Hamming-booleanity kernel");
    let prepare = start.elapsed();

    drive_rounds(&mut *kernel, &claims, log_t, log_t, prepare, |_| {
        RoundPhase::Cycle
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
fn measure_registers_val_evaluation(
    workload: Workload,
    scale: u32,
    backend: BackendKind,
) -> VerticalTiming {
    let VerticalFixture {
        program: jolt_program,
        program_preprocessing,
        config,
        log_t,
        trace: padded,
    } = fixture(workload, scale);

    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded),
    );

    let relation = RegistersValEvaluation::<Fr>::new(TraceDimensions::new(log_t));
    let point: Vec<Fr> = (0..REGISTER_ADDRESS_BITS + log_t)
        .map(|i| Fr::from_u64(31 + 7 * i as u64))
        .collect();

    let claims = RegistersValEvaluationInputClaims {
        registers_val: Fr::from_u64(0),
    };
    let points = RegistersValEvaluationInputClaims {
        registers_val: point,
    };
    let challenges = NoChallenges::default();
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
        .registers_val_evaluation
        .prepare(&mut session, &witness, inputs())
        .expect("prepare the stage-5 registers value-evaluation kernel");
    let prepare = start.elapsed();

    drive_rounds(&mut *kernel, &claims, log_t, log_t, prepare, |_| {
        RoundPhase::Cycle
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
fn measure_ram_ra_claim_reduction(
    workload: Workload,
    scale: u32,
    backend: BackendKind,
) -> VerticalTiming {
    let VerticalFixture {
        program: jolt_program,
        program_preprocessing,
        config,
        log_t,
        trace: padded,
    } = fixture(workload, scale);

    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded),
    );

    let ram_log_k = config.ram_K.ilog2() as usize;
    let relation = RamRaClaimReduction::<Fr>::new(TraceDimensions::new(log_t), ram_log_k);
    let point = |offset: u64| -> Vec<Fr> {
        (0..ram_log_k + log_t)
            .map(|i| Fr::from_u64(31 + 7 * i as u64 + offset))
            .collect()
    };

    let claims = RamRaClaimReductionInputClaims {
        raf: Fr::from_u64(0),
        read_write: Fr::from_u64(0),
        val_check: Fr::from_u64(0),
    };
    let points = RamRaClaimReductionInputClaims {
        raf: point(0),
        read_write: point(101),
        val_check: point(202),
    };
    let challenges = RamRaClaimReductionChallenges {
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
        .ram_ra_claim_reduction
        .prepare(&mut session, &witness, inputs())
        .expect("prepare the stage-5 RAM ra claim-reduction kernel");
    let prepare = start.elapsed();

    drive_rounds(&mut *kernel, &claims, log_t, log_t, prepare, |_| {
        RoundPhase::Cycle
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
fn measure_hamming_weight_claim_reduction(
    workload: Workload,
    scale: u32,
    backend: BackendKind,
) -> VerticalTiming {
    let VerticalFixture {
        program: jolt_program,
        program_preprocessing,
        config,
        log_t,
        trace: padded,
    } = fixture(workload, scale);

    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded),
    );

    let chunk_bits = config.one_hot_config.committed_chunk_bits();
    let layout = formula_dimensions_from_parts(
        config.one_hot_config,
        log_t,
        program_preprocessing.bytecode.code_size,
        config.ram_K,
        JoltRelationId::HammingWeightClaimReduction,
    )
    .expect("formula dimensions")
    .ra_layout;
    let dimensions = HammingWeightClaimReductionDimensions::new(layout, chunk_bits);

    let r_cycle: Vec<Fr> = (0..log_t)
        .map(|i| Fr::from_u64(37 + 7 * i as u64))
        .collect();
    let r_address: Vec<Fr> = (0..dimensions.log_k_chunk)
        .map(|i| Fr::from_u64(23 + 3 * i as u64))
        .collect();
    let virtualization_points: Vec<Vec<Fr>> = (0..dimensions.layout.total())
        .map(|leg| {
            (0..dimensions.log_k_chunk)
                .map(|i| Fr::from_u64(29 + 5 * i as u64 + 101 * leg as u64))
                .collect()
        })
        .collect();
    let relation = HammingWeightClaimReduction::<Fr>::new(
        dimensions,
        r_cycle,
        r_address,
        virtualization_points,
    );

    let claims = HammingWeightClaimReductionInputClaims::default();
    let points = HammingWeightClaimReductionInputClaims::default();
    let challenges = HammingWeightClaimReductionChallenges {
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
        .hamming_weight_claim_reduction
        .prepare(&mut session, &witness, inputs())
        .expect("prepare the stage-7 Hamming-weight claim-reduction kernel");
    let prepare = start.elapsed();

    let rounds = jolt_claims::SymbolicSumcheck::rounds(
        jolt_verifier::stages::relations::ConcreteSumcheck::symbolic(&relation),
    );
    drive_rounds(&mut *kernel, &claims, rounds, log_t, prepare, |_| {
        RoundPhase::Address
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
fn measure_ram_raf_evaluation(
    workload: Workload,
    scale: u32,
    backend: BackendKind,
) -> VerticalTiming {
    let VerticalFixture {
        program: jolt_program,
        program_preprocessing,
        config,
        log_t,
        trace: padded,
    } = fixture(workload, scale);

    let lowest_address = padded.device.memory_layout.get_lowest_address();
    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded),
    );

    let ram_log_k = config.ram_K.ilog2() as usize;
    let read_write_dimensions = config.rw_config.ram_dimensions(log_t, ram_log_k);
    let raf_dimensions = RamRafEvaluationDimensions::try_from(read_write_dimensions)
        .expect("RAM RAF evaluation dimensions");
    let tau_low: Vec<Fr> = (0..log_t)
        .map(|i| Fr::from_u64(23 + 3 * i as u64))
        .collect();
    let relation = RamRafEvaluation::<Fr>::new(
        read_write_dimensions,
        raf_dimensions,
        ram_log_k,
        lowest_address,
        tau_low,
    );

    let claims = RamRafEvaluationInputClaims {
        ram_address: Fr::from_u64(0),
    };
    let points = RamRafEvaluationInputClaims {
        ram_address: Vec::new(),
    };
    let challenges = NoChallenges::default();
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
        .ram_raf_evaluation
        .prepare(&mut session, &witness, inputs())
        .expect("prepare the stage-2 RAM RAF-evaluation kernel");
    let prepare = start.elapsed();

    let rounds = jolt_claims::SymbolicSumcheck::rounds(
        jolt_verifier::stages::relations::ConcreteSumcheck::symbolic(&relation),
    );
    drive_rounds(&mut *kernel, &claims, rounds, log_t, prepare, |_| {
        RoundPhase::Address
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
fn measure_ram_output_check(
    workload: Workload,
    scale: u32,
    backend: BackendKind,
) -> VerticalTiming {
    let VerticalFixture {
        program: jolt_program,
        program_preprocessing,
        config,
        log_t,
        trace: padded,
    } = fixture(workload, scale);

    let public_memory = PublicIoMemory::new(&padded.device).expect("public IO memory");
    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded),
    );

    let ram_log_k = config.ram_K.ilog2() as usize;
    let read_write_dimensions = config.rw_config.ram_dimensions(log_t, ram_log_k);
    let relation = RamOutputCheck::<Fr>::new(read_write_dimensions, public_memory);

    let claims = RamOutputCheckInputClaims::default();
    let points = RamOutputCheckInputClaims::default();
    let challenges = RamOutputCheckChallenges {
        output_address: (0..ram_log_k)
            .map(|i| Fr::from_u64(23 + 3 * i as u64))
            .collect(),
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
        .ram_output_check
        .prepare(&mut session, &witness, inputs())
        .expect("prepare the stage-2 RAM output-check kernel");
    let prepare = start.elapsed();

    let rounds = jolt_claims::SymbolicSumcheck::rounds(
        jolt_verifier::stages::relations::ConcreteSumcheck::symbolic(&relation),
    );
    drive_rounds(&mut *kernel, &claims, rounds, log_t, prepare, |_| {
        RoundPhase::Address
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
fn measure_booleanity_address(
    workload: Workload,
    scale: u32,
    backend: BackendKind,
) -> VerticalTiming {
    let VerticalFixture {
        program: jolt_program,
        program_preprocessing,
        config,
        log_t,
        trace: padded,
    } = fixture(workload, scale);

    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded),
    );

    let chunk_bits = config.one_hot_config.committed_chunk_bits();
    let layout = formula_dimensions_from_parts(
        config.one_hot_config,
        log_t,
        program_preprocessing.bytecode.code_size,
        config.ram_K,
        JoltRelationId::Booleanity,
    )
    .expect("formula dimensions")
    .ra_layout;
    let dimensions = BooleanityDimensions::new(layout, log_t, chunk_bits);

    let instruction_r_address: Vec<Fr> = (0..chunk_bits)
        .map(|i| Fr::from_u64(23 + 3 * i as u64))
        .collect();
    let instruction_r_cycle: Vec<Fr> = (0..log_t)
        .map(|i| Fr::from_u64(37 + 7 * i as u64))
        .collect();
    let relation =
        BooleanityAddressPhase::<Fr>::new(dimensions, instruction_r_address, instruction_r_cycle);

    let claims = BooleanityAddressPhaseInputClaims::default();
    let points = BooleanityAddressPhaseInputClaims::default();
    let challenges = BooleanityAddressPhaseChallenges {
        reference_address: (0..chunk_bits)
            .map(|i| Fr::from_u64(29 + 5 * i as u64))
            .collect(),
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
        .booleanity_address
        .prepare(&mut session, &witness, inputs())
        .expect("prepare the stage-6a booleanity address-phase kernel");
    let prepare = start.elapsed();

    let rounds = jolt_claims::SymbolicSumcheck::rounds(
        jolt_verifier::stages::relations::ConcreteSumcheck::symbolic(&relation),
    );
    drive_rounds(&mut *kernel, &claims, rounds, log_t, prepare, |_| {
        RoundPhase::Address
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
fn measure_bytecode_read_raf_address(
    workload: Workload,
    scale: u32,
    backend: BackendKind,
) -> VerticalTiming {
    let VerticalFixture {
        program: jolt_program,
        program_preprocessing,
        config,
        log_t,
        trace: padded,
    } = fixture(workload, scale);

    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded),
    );

    let dimensions = formula_dimensions_from_parts(
        config.one_hot_config,
        log_t,
        program_preprocessing.bytecode.code_size,
        config.ram_K,
        JoltRelationId::BytecodeReadRaf,
    )
    .expect("formula dimensions")
    .bytecode_read_raf;
    let log_k = dimensions.log_k();

    let stage_cycle_points: [Vec<Fr>; READ_RAF_CYCLE_STAGES] = core::array::from_fn(|stage| {
        (0..log_t)
            .map(|i| Fr::from_u64(37 + 7 * i as u64 + 101 * stage as u64))
            .collect()
    });
    let register_point = |offset: u64| -> Vec<Fr> {
        (0..REGISTER_ADDRESS_BITS + log_t)
            .map(|i| Fr::from_u64(31 + 7 * i as u64 + offset))
            .collect()
    };
    let relation = BytecodeReadRafAddressPhase::<Fr>::new(
        dimensions,
        false,
        BytecodeStagePoints {
            stage_cycle_points,
            register_read_write_point: register_point(0),
            register_val_evaluation_point: register_point(101),
        },
        (1usize << log_k) - 2,
    );

    let claims = BytecodeReadRafAddressPhaseInputClaims::default();
    let points = BytecodeReadRafAddressPhaseInputClaims::default();
    let challenges = BytecodeReadRafAddressPhaseChallenges {
        gamma: Fr::from_u64(101),
        stage1_gamma: Fr::from_u64(103),
        stage2_gamma: Fr::from_u64(107),
        stage3_gamma: Fr::from_u64(109),
        stage4_gamma: Fr::from_u64(113),
        stage5_gamma: Fr::from_u64(127),
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
        .bytecode_read_raf_address
        .prepare(&mut session, &witness, inputs())
        .expect("prepare the stage-6a bytecode read-RAF address-phase kernel");
    let prepare = start.elapsed();

    let rounds = jolt_claims::SymbolicSumcheck::rounds(
        jolt_verifier::stages::relations::ConcreteSumcheck::symbolic(&relation),
    );
    drive_rounds(&mut *kernel, &claims, rounds, log_t, prepare, |_| {
        RoundPhase::Address
    })
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
fn measure_spartan_product(workload: Workload, scale: u32, backend: BackendKind) -> VerticalTiming {
    let VerticalFixture {
        program: jolt_program,
        program_preprocessing,
        config,
        log_t,
        trace: padded,
    } = fixture(workload, scale);

    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded),
    );

    let tau_low: Vec<Fr> = (0..log_t)
        .map(|i| Fr::from_u64(23 + 3 * i as u64))
        .collect();
    let tau_high = Fr::from_u64(97);
    let uniskip_challenge = Fr::from_u64(101);

    let selected = match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut session = ProofSession::default();
    let start = Instant::now();
    selected
        .spartan_product_uniskip
        .prepare(&mut session, log_t, &tau_low, &witness)
        .expect("prepare the stage-2 Spartan product uni-skip kernel");
    let prepare = start.elapsed();

    let start = Instant::now();
    let _ = selected
        .spartan_product_uniskip
        .first_round_poly(&mut session, &[tau_high])
        .expect("the stage-2 Spartan product uni-skip first-round polynomial");
    let uniskip_poly = start.elapsed();

    let relation = ProductRemainder::new(
        SpartanProductDimensions::new(log_t),
        uniskip_challenge,
        tau_high,
        tau_low,
    );
    let claims = product_remainder_input_values_from_uniskip_output(Fr::from_u64(0));
    let points = ProductRemainderInputClaims {
        product_uniskip: Vec::new(),
    };
    let challenges = NoChallenges::default();
    let inputs = ProverInputs {
        relation: &relation,
        claims: &claims,
        points: &points,
        challenges: &challenges,
    };

    let start = Instant::now();
    let mut kernel = selected
        .spartan_product_remainder
        .prepare(&mut session, &witness, inputs)
        .expect("prepare the stage-2 Spartan product remainder kernel");
    let remainder_prepare = start.elapsed();

    let rounds = jolt_claims::SymbolicSumcheck::rounds(
        jolt_verifier::stages::relations::ConcreteSumcheck::symbolic(&relation),
    );
    let mut timing = drive_rounds(&mut *kernel, &claims, rounds, log_t, prepare, |_| {
        RoundPhase::Cycle
    });
    timing.handoff = uniskip_poly;
    timing.address = remainder_prepare;
    timing
}
