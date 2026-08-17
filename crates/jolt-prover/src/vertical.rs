use std::time::{Duration, Instant};

use common::constants::XLEN as RISCV_XLEN;
use common::jolt_device::MemoryLayout;
use jolt_claims::protocols::jolt::geometry::booleanity::BooleanityDimensions;
use jolt_claims::protocols::jolt::geometry::claim_reductions::bytecode::{
    COMMITTED_BYTECODE_LANE_CAPACITY, NUM_BYTECODE_VAL_STAGES,
};
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
use jolt_claims::protocols::jolt::relations::claim_reductions::advice::{
    TrustedAdviceAddressPhaseInputClaims, TrustedAdviceCyclePhaseInputClaims,
    UntrustedAdviceAddressPhaseInputClaims, UntrustedAdviceCyclePhaseInputClaims,
};
use jolt_claims::protocols::jolt::relations::claim_reductions::bytecode::{
    BytecodeReductionAddressPhaseInputClaims, BytecodeReductionCyclePhaseChallenges,
    BytecodeReductionCyclePhaseInputClaims,
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
use jolt_claims::protocols::jolt::relations::claim_reductions::program_image::{
    ProgramImageReductionAddressPhaseInputClaims, ProgramImageReductionCyclePhaseInputClaims,
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
use jolt_claims::protocols::jolt::JoltCommittedPolynomial;
use jolt_claims::protocols::jolt::{
    AdviceClaimReductionLayout, BytecodeClaimReductionLayout, JoltAdviceKind, JoltChallengeId,
    JoltRelationId, PrecommittedClaimReduction, PrecommittedReductionLayout,
    ProgramImageClaimReductionLayout, TraceDimensions,
};
use jolt_claims::NoChallenges;
use jolt_dory::DoryScheme;
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_kernels::{CommitmentGrid, JoltBackend, ProofSession, ProverInputs};
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
use jolt_verifier::stages::stage6b::committed_reduction_cycle_phase::{
    BytecodeReductionCyclePhase, ProgramImageReductionCyclePhase, TrustedAdviceCyclePhase,
    UntrustedAdviceCyclePhase,
};
use jolt_verifier::stages::stage6b::inc_claim_reduction::IncClaimReduction;
use jolt_verifier::stages::stage6b::instruction_ra_virtualization::{
    InstructionRaVirtualization, InstructionRaVirtualizationChallenges,
    InstructionRaVirtualizationInputClaims,
};
use jolt_verifier::stages::stage6b::outputs::BytecodeReductionWeights;
use jolt_verifier::stages::stage6b::ram_hamming_booleanity::RamHammingBooleanity;
use jolt_verifier::stages::stage6b::ram_ra_virtualization::{
    RamRaVirtualization, RamRaVirtualizationInputClaims,
};
use jolt_verifier::stages::stage7::advice_address_phase::{
    TrustedAdviceAddressPhase, UntrustedAdviceAddressPhase,
};
use jolt_verifier::stages::stage7::committed_reduction_address_phase::{
    BytecodeReductionAddressPhase, ProgramImageReductionAddressPhase,
};
use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::HammingWeightClaimReduction;
use jolt_verifier::stages::{CommittedProgramSchedule, PrecommittedSchedule};
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

use crate::profile::{pad_trace, trace_modular, BackendKind, Workload};
use crate::vertical_baseline::{
    advice_baseline, bytecode_baseline, program_image_baseline, LegacyPrecommittedInputs,
};
use crate::ProverConfig;
use jolt_kernels::committed_program::program_image_words_padded;
use jolt_prover_legacy::zkvm::claim_reductions::AdviceKind;

const SAFETY_MARGIN: f64 = 0.9;

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
pub enum VerticalRelation {
    AdviceOpening,
    BooleanityAddress,
    BooleanityCycle,
    BytecodeReadRafAddress,
    BytecodeReadRafCycle,
    BytecodeReductionAddress,
    BytecodeReductionCycle,
    Commit,
    HammingWeightClaimReduction,
    IncClaimReduction,
    InstructionClaimReduction,
    InstructionInput,
    InstructionRaVirtualization,
    InstructionReadRaf,
    JointOpening,
    ProgramImageReductionAddress,
    ProgramImageReductionCycle,
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
    TrustedAdviceAddress,
    TrustedAdviceCycle,
    UntrustedAdviceAddress,
    UntrustedAdviceCycle,
}

impl VerticalRelation {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AdviceOpening => "advice-opening",
            Self::BooleanityAddress => "booleanity-address",
            Self::BooleanityCycle => "booleanity-cycle",
            Self::BytecodeReadRafAddress => "bytecode-read-raf-address",
            Self::BytecodeReadRafCycle => "bytecode-read-raf-cycle",
            Self::BytecodeReductionAddress => "bytecode-reduction-address",
            Self::BytecodeReductionCycle => "bytecode-reduction-cycle",
            Self::Commit => "commit",
            Self::HammingWeightClaimReduction => "hamming-weight-claim-reduction",
            Self::IncClaimReduction => "inc-claim-reduction",
            Self::InstructionClaimReduction => "instruction-claim-reduction",
            Self::InstructionInput => "instruction-input",
            Self::InstructionRaVirtualization => "instruction-ra-virtualization",
            Self::InstructionReadRaf => "instruction-read-raf",
            Self::JointOpening => "joint-opening",
            Self::ProgramImageReductionAddress => "program-image-reduction-address",
            Self::ProgramImageReductionCycle => "program-image-reduction-cycle",
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
            Self::TrustedAdviceAddress => "trusted-advice-address",
            Self::TrustedAdviceCycle => "trusted-advice-cycle",
            Self::UntrustedAdviceAddress => "untrusted-advice-address",
            Self::UntrustedAdviceCycle => "untrusted-advice-cycle",
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

    #[clap(long, default_value_t = 2)]
    pub bytecode_chunks: usize,

    #[clap(long)]
    pub legacy: bool,
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
    memory_layout: MemoryLayout,
    min_bytecode_address: u64,
    program_image_len_words: usize,
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
        memory_layout,
        min_bytecode_address: verifier_preprocessing.program.min_bytecode_address(),
        program_image_len_words: verifier_preprocessing.program.program_image_len_words(),
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
        if args.legacy {
            "legacy"
        } else {
            args.backend.as_str()
        },
        args.name.as_str(),
    );
    println!(
        "{:>6}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}  {:>11}",
        "log_T", "prepare", "address", "handoff", "cycle", "claims", "total",
    );
    let mut timings = Vec::new();
    for &scale in &args.scales {
        if args.legacy {
            assert!(
                !matches!(
                    args.relation,
                    VerticalRelation::Commit
                        | VerticalRelation::JointOpening
                        | VerticalRelation::AdviceOpening
                ),
                "the Dory MSM slots have no in-harness legacy driver: legacy's commit method is \
                 private and takes `&mut self` on the whole prover, and its joint opening happens \
                 inside the external dory crate. Take the baseline from legacy's own trace instead \
                 — `cargo run --release -p jolt-prover-legacy -- benchmark --name {} --scale {} \
                 --format chrome` then `python3 scripts/legacy_relation_baseline.py \
                 benchmark-runs/perfetto_traces/<trace>.json commit commit-phases stage8`",
                args.name.as_str(),
                scale,
            );
            let timing =
                measure_legacy_precommitted(args.relation, args.name, scale, args.bytecode_chunks);
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
            continue;
        }
        let timing = match args.relation {
            VerticalRelation::Commit => measure_commit(args.name, scale, args.backend),
            VerticalRelation::JointOpening => measure_joint_opening(args.name, scale, args.backend),
            VerticalRelation::AdviceOpening => measure_advice_opening(
                args.name,
                scale,
                args.backend,
                args.bytecode_chunks,
                JoltAdviceKind::Trusted,
            ),
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
            VerticalRelation::TrustedAdviceCycle => measure_advice_cycle(
                args.name,
                scale,
                args.backend,
                args.bytecode_chunks,
                JoltAdviceKind::Trusted,
            ),
            VerticalRelation::UntrustedAdviceCycle => measure_advice_cycle(
                args.name,
                scale,
                args.backend,
                args.bytecode_chunks,
                JoltAdviceKind::Untrusted,
            ),
            VerticalRelation::TrustedAdviceAddress => measure_advice_address(
                args.name,
                scale,
                args.backend,
                args.bytecode_chunks,
                JoltAdviceKind::Trusted,
            ),
            VerticalRelation::UntrustedAdviceAddress => measure_advice_address(
                args.name,
                scale,
                args.backend,
                args.bytecode_chunks,
                JoltAdviceKind::Untrusted,
            ),
            VerticalRelation::BytecodeReductionCycle => measure_bytecode_reduction_cycle(
                args.name,
                scale,
                args.backend,
                args.bytecode_chunks,
            ),
            VerticalRelation::BytecodeReductionAddress => measure_bytecode_reduction_address(
                args.name,
                scale,
                args.backend,
                args.bytecode_chunks,
            ),
            VerticalRelation::ProgramImageReductionCycle => measure_program_image_reduction_cycle(
                args.name,
                scale,
                args.backend,
                args.bytecode_chunks,
            ),
            VerticalRelation::ProgramImageReductionAddress => {
                measure_program_image_reduction_address(
                    args.name,
                    scale,
                    args.backend,
                    args.bytecode_chunks,
                )
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
        ..
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
        ..
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
        ..
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
        ..
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
        ..
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
        ..
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
        ..
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
        ..
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
        ..
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
        ..
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
        ..
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
        ..
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
        ..
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
        ..
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
        ..
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
        ..
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
        ..
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
        ..
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
        ..
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
        ..
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
        ..
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
        ..
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

struct PrecommittedFixture {
    fixture: VerticalFixture,
    schedule: PrecommittedSchedule,
    bytecode_chunk_count: usize,
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture errors fail loudly"
)]
fn precommitted_fixture(
    workload: Workload,
    scale: u32,
    bytecode_chunks: usize,
) -> PrecommittedFixture {
    let fixture = fixture(workload, scale);
    let start_index = fixture
        .memory_layout
        .remapped_word_address(fixture.min_bytecode_address)
        .expect("program image start index") as usize;
    let schedule = PrecommittedSchedule::new(
        fixture.config.trace_polynomial_order,
        fixture.log_t,
        fixture.config.one_hot_config.committed_chunk_bits(),
        Some(fixture.memory_layout.max_trusted_advice_size as usize),
        Some(fixture.memory_layout.max_untrusted_advice_size as usize),
        Some(CommittedProgramSchedule {
            bytecode_len: fixture.program_preprocessing.bytecode.code_size,
            bytecode_chunk_count: bytecode_chunks,
            program_image_len_words: fixture.program_image_len_words,
            program_image_start_index: start_index,
        }),
    )
    .expect("precommitted schedule");
    PrecommittedFixture {
        fixture,
        schedule,
        bytecode_chunk_count: bytecode_chunks,
    }
}

fn with_precommitted_fixture<T>(
    workload: Workload,
    scale: u32,
    bytecode_chunks: usize,
    body: impl FnOnce(&TraceBackend<'_, OwnedTrace>, &PrecommittedSchedule, PrecommittedGeometry) -> T,
) -> T {
    let PrecommittedFixture {
        fixture,
        schedule,
        bytecode_chunk_count,
    } = precommitted_fixture(workload, scale, bytecode_chunks);
    let VerticalFixture {
        program: jolt_program,
        program_preprocessing,
        config,
        log_t,
        trace: padded,
        ..
    } = fixture;
    let geometry = PrecommittedGeometry {
        log_t,
        ram_log_k: config.ram_K.ilog2() as usize,
        bytecode_chunk_count,
    };
    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config)
            .include_trusted_advice(true)
            .include_untrusted_advice(true),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded),
    );
    body(&witness, &schedule, geometry)
}

#[derive(Clone, Copy)]
struct PrecommittedGeometry {
    log_t: usize,
    ram_log_k: usize,
    bytecode_chunk_count: usize,
}

fn with_commit_fixture<T>(
    workload: Workload,
    scale: u32,
    body: impl FnOnce(&TraceBackend<'_, OwnedTrace>, CommitmentGrid, &ProverConfig) -> T,
) -> T {
    let VerticalFixture {
        program: jolt_program,
        program_preprocessing,
        config,
        log_t,
        trace: padded,
        memory_layout,
        ..
    } = fixture(workload, scale);
    let grid = CommitmentGrid {
        total_vars: config.commitment_total_vars(&memory_layout, false, false, None),
        log_t,
        log_k_chunk: config.one_hot_config.committed_chunk_bits(),
        order: config.trace_polynomial_order,
    };
    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded),
    );
    body(&witness, grid, &config)
}

#[expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "measurement harness: kernel errors fail loudly and geometry is reported to stdout"
)]
fn measure_commit(workload: Workload, scale: u32, backend: BackendKind) -> VerticalTiming {
    with_commit_fixture(workload, scale, |witness, grid, _config| {
        let ids: Vec<JoltCommittedPolynomial> =
            jolt_witness::JoltWitnessOracle::<Fr>::committed_order(witness)
                .expect("committed order")
                .into_iter()
                .filter(|id| {
                    !matches!(
                        id,
                        JoltCommittedPolynomial::TrustedAdvice
                            | JoltCommittedPolynomial::UntrustedAdvice
                    )
                })
                .collect();
        let setup = DoryScheme::setup_prover(grid.total_vars);
        let selected = selected_backend(backend);
        let mut session = ProofSession::default();
        let start = Instant::now();
        let committed = selected
            .commit
            .commit_witness(
                &mut session,
                witness as &dyn jolt_witness::RowSource,
                &ids,
                grid,
                &setup,
            )
            .expect("commit the witness polynomials");
        let elapsed = start.elapsed();
        println!(
            "         {} committed columns, grid {} vars, {} columns per row",
            committed.len(),
            grid.total_vars,
            grid.num_columns(),
        );
        VerticalTiming {
            log_t: grid.log_t,
            prepare: Duration::ZERO,
            address: Duration::ZERO,
            handoff: Duration::ZERO,
            cycle: elapsed,
            claims: Duration::ZERO,
        }
    })
}

#[expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "measurement harness: kernel errors fail loudly and geometry is reported to stdout"
)]
fn measure_joint_opening(workload: Workload, scale: u32, backend: BackendKind) -> VerticalTiming {
    with_commit_fixture(workload, scale, |witness, grid, _config| {
        let order: Vec<JoltCommittedPolynomial> =
            jolt_witness::JoltWitnessOracle::<Fr>::committed_order(witness)
                .expect("committed order")
                .into_iter()
                .filter(|id| {
                    !matches!(
                        id,
                        JoltCommittedPolynomial::TrustedAdvice
                            | JoltCommittedPolynomial::UntrustedAdvice
                    )
                })
                .collect();
        let tables = std::collections::BTreeMap::new();
        let selected = selected_backend(backend);
        let mut session = ProofSession::default();

        let start = Instant::now();
        let polynomials = selected
            .joint_opening
            .prepare(&mut session, witness, &order, &tables, grid)
            .expect("prepare the joint-opening polynomials");
        let prepare = start.elapsed();

        let sigma = grid.total_vars.div_ceil(2);
        let left = synthetic_point(1usize << (grid.total_vars - sigma), 17);
        let start = Instant::now();
        for polynomial in &polynomials {
            let folded = polynomial.fold_rows(&left, sigma);
            assert_eq!(folded.len(), 1usize << sigma, "fold width");
        }
        let fold = start.elapsed();

        println!(
            "         {} polynomials, grid {} vars, sigma {}",
            polynomials.len(),
            grid.total_vars,
            sigma,
        );
        VerticalTiming {
            log_t: grid.log_t,
            prepare,
            address: Duration::ZERO,
            handoff: Duration::ZERO,
            cycle: fold,
            claims: Duration::ZERO,
        }
    })
}

#[expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "measurement harness: kernel errors fail loudly and geometry is reported to stdout"
)]
fn measure_advice_opening(
    workload: Workload,
    scale: u32,
    backend: BackendKind,
    bytecode_chunks: usize,
    kind: JoltAdviceKind,
) -> VerticalTiming {
    with_precommitted_fixture(
        workload,
        scale,
        bytecode_chunks,
        |witness, schedule, geometry| {
            let layout = match kind {
                JoltAdviceKind::Trusted => schedule.trusted_advice.as_ref(),
                JoltAdviceKind::Untrusted => schedule.untrusted_advice.as_ref(),
            }
            .expect("advice layout present");
            let vars = layout
                .precommitted()
                .poly_opening_round_permutation_be()
                .len();
            let point = synthetic_point(vars, 29);
            let selected = selected_backend(backend);
            let mut session = ProofSession::default();
            let start = Instant::now();
            let value = selected
                .advice_opening
                .evaluate(&mut session, kind, &point, witness)
                .expect("evaluate the advice opening");
            let elapsed = start.elapsed();
            println!(
                "         {kind:?} advice, {vars} vars, value nonzero = {}",
                value != Fr::from_u64(0)
            );
            VerticalTiming {
                log_t: geometry.log_t,
                prepare: Duration::ZERO,
                address: Duration::ZERO,
                handoff: Duration::ZERO,
                cycle: elapsed,
                claims: Duration::ZERO,
            }
        },
    )
}

fn selected_backend(backend: BackendKind) -> JoltBackend<Fr, DoryScheme> {
    match backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    }
}

fn synthetic_point(len: usize, seed: u64) -> Vec<Fr> {
    (0..len)
        .map(|i| Fr::from_u64(seed + 7 * i as u64 + 1))
        .collect()
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: schedule errors fail loudly"
)]
fn synthetic_cycle_variables(reduction: &PrecommittedClaimReduction) -> Vec<Fr> {
    let challenges = synthetic_point(reduction.cycle_phase_total_rounds(), 97);
    reduction
        .cycle_phase_variable_challenges(&challenges)
        .expect("cycle-phase variable challenges")
}

fn bytecode_reduction_weights_fixture(
    layout: &BytecodeClaimReductionLayout,
) -> BytecodeReductionWeights<Fr> {
    BytecodeReductionWeights {
        r_bc: synthetic_point(layout.log_bytecode_chunk_size(), 31),
        chunk_rbc_weights: synthetic_point(layout.chunk_count(), 53),
        lane_weights: synthetic_point(COMMITTED_BYTECODE_LANE_CAPACITY, 71),
    }
}

#[expect(
    clippy::print_stdout,
    reason = "measurement harness: reports to stdout like the surrounding arms"
)]
fn absent_address_phase(
    label: &str,
    reduction: &PrecommittedClaimReduction,
    geometry: PrecommittedGeometry,
) -> VerticalTiming {
    println!(
        "  {label}: NO ADDRESS PHASE at log_T={} — the polynomial has {} variables and \
         {}/{} active cycle rounds, so none of its variables land in the schedule's top \
         {} address rounds; the reduction finalizes in the cycle phase and stage 7 has no \
         member for it. Nothing to measure.",
        geometry.log_t,
        reduction.poly_opening_round_permutation_be().len(),
        reduction.cycle_phase_rounds().len(),
        reduction.cycle_phase_total_rounds(),
        reduction.address_phase_total_rounds(),
    );
    VerticalTiming {
        log_t: geometry.log_t,
        prepare: Duration::ZERO,
        address: Duration::ZERO,
        handoff: Duration::ZERO,
        cycle: Duration::ZERO,
        claims: Duration::ZERO,
    }
}

#[expect(
    clippy::print_stdout,
    reason = "measurement harness: reports to stdout like the surrounding arms"
)]
fn report_precommitted_geometry(
    label: &str,
    reduction: &PrecommittedClaimReduction,
    tables: usize,
) {
    let vars = reduction.poly_opening_round_permutation_be().len();
    println!(
        "  {label}: table 2^{vars} = {} coefficients x {tables} tables ({:.2} MiB), \
         cycle {}/{} active, address {}/{} active",
        1usize << vars,
        ((tables << vars) * 32) as f64 / (1024.0 * 1024.0),
        reduction.cycle_phase_rounds().len(),
        reduction.cycle_phase_total_rounds(),
        reduction.address_phase_rounds().len(),
        reduction.address_phase_total_rounds(),
    );
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
fn measure_advice_cycle(
    workload: Workload,
    scale: u32,
    backend: BackendKind,
    bytecode_chunks: usize,
    kind: JoltAdviceKind,
) -> VerticalTiming {
    with_precommitted_fixture(
        workload,
        scale,
        bytecode_chunks,
        |witness, schedule, geometry| {
            let layout: &AdviceClaimReductionLayout =
                schedule.advice(kind).expect("advice layout present");
            let reduction = layout.precommitted();
            let rounds = reduction.cycle_phase_total_rounds();
            let r_val = synthetic_point(reduction.poly_opening_round_permutation_be().len(), 11);
            let selected = selected_backend(backend);
            let challenges = NoChallenges::default();
            let mut session = ProofSession::default();
            report_precommitted_geometry(
                match kind {
                    JoltAdviceKind::Trusted => "trusted-advice-cycle",
                    JoltAdviceKind::Untrusted => "untrusted-advice-cycle",
                },
                reduction,
                2,
            );

            match kind {
                JoltAdviceKind::Trusted => {
                    let relation = TrustedAdviceCyclePhase::<Fr>::new(layout, Some(r_val));
                    let claims = TrustedAdviceCyclePhaseInputClaims::default();
                    let points = TrustedAdviceCyclePhaseInputClaims::default();
                    let start = Instant::now();
                    let mut kernel = selected
                        .trusted_advice_cycle
                        .prepare(
                            &mut session,
                            witness,
                            ProverInputs {
                                relation: &relation,
                                claims: &claims,
                                points: &points,
                                challenges: &challenges,
                            },
                        )
                        .expect("prepare the stage-6b trusted-advice cycle-phase kernel");
                    let prepare = start.elapsed();
                    drive_rounds(
                        &mut *kernel,
                        &claims,
                        rounds,
                        geometry.log_t,
                        prepare,
                        |_| RoundPhase::Cycle,
                    )
                }
                JoltAdviceKind::Untrusted => {
                    let relation = UntrustedAdviceCyclePhase::<Fr>::new(layout, Some(r_val));
                    let claims = UntrustedAdviceCyclePhaseInputClaims::default();
                    let points = UntrustedAdviceCyclePhaseInputClaims::default();
                    let start = Instant::now();
                    let mut kernel = selected
                        .untrusted_advice_cycle
                        .prepare(
                            &mut session,
                            witness,
                            ProverInputs {
                                relation: &relation,
                                claims: &claims,
                                points: &points,
                                challenges: &challenges,
                            },
                        )
                        .expect("prepare the stage-6b untrusted-advice cycle-phase kernel");
                    let prepare = start.elapsed();
                    drive_rounds(
                        &mut *kernel,
                        &claims,
                        rounds,
                        geometry.log_t,
                        prepare,
                        |_| RoundPhase::Cycle,
                    )
                }
            }
        },
    )
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
fn measure_advice_address(
    workload: Workload,
    scale: u32,
    backend: BackendKind,
    bytecode_chunks: usize,
    kind: JoltAdviceKind,
) -> VerticalTiming {
    with_precommitted_fixture(
        workload,
        scale,
        bytecode_chunks,
        |witness, schedule, geometry| {
            let layout: &AdviceClaimReductionLayout =
                schedule.advice(kind).expect("advice layout present");
            let reduction = layout.precommitted();
            let label = match kind {
                JoltAdviceKind::Trusted => "trusted-advice-address",
                JoltAdviceKind::Untrusted => "untrusted-advice-address",
            };
            if reduction.num_address_phase_rounds() == 0 {
                return absent_address_phase(label, reduction, geometry);
            }
            report_precommitted_geometry(label, reduction, 2);

            let cycle_rounds = reduction.cycle_phase_total_rounds();
            let address_rounds = reduction.address_phase_total_rounds();
            let cycle_variables = synthetic_cycle_variables(reduction);
            let r_val = synthetic_point(reduction.poly_opening_round_permutation_be().len(), 11);
            let selected = selected_backend(backend);
            let challenges = NoChallenges::default();
            let mut session = ProofSession::default();

            match kind {
                JoltAdviceKind::Trusted => {
                    let cycle_relation =
                        TrustedAdviceCyclePhase::<Fr>::new(layout, Some(r_val.clone()));
                    let cycle_claims = TrustedAdviceCyclePhaseInputClaims::default();
                    let cycle_points = TrustedAdviceCyclePhaseInputClaims::default();
                    let cycle_kernel = selected
                        .trusted_advice_cycle
                        .prepare(
                            &mut session,
                            witness,
                            ProverInputs {
                                relation: &cycle_relation,
                                claims: &cycle_claims,
                                points: &cycle_points,
                                challenges: &challenges,
                            },
                        )
                        .expect("prepare the stage-6b trusted-advice cycle-phase kernel");
                    let mut cycle_kernel = cycle_kernel;
                    let _ = drive_rounds(
                        &mut *cycle_kernel,
                        &cycle_claims,
                        cycle_rounds,
                        geometry.log_t,
                        Duration::ZERO,
                        |_| RoundPhase::Cycle,
                    );
                    cycle_kernel.park_residue(&mut session);

                    let relation =
                        TrustedAdviceAddressPhase::<Fr>::new(layout, Some(r_val), cycle_variables);
                    let claims = TrustedAdviceAddressPhaseInputClaims::default();
                    let points = TrustedAdviceAddressPhaseInputClaims::default();
                    let start = Instant::now();
                    let mut kernel = selected
                        .trusted_advice_address
                        .prepare(
                            &mut session,
                            witness,
                            ProverInputs {
                                relation: &relation,
                                claims: &claims,
                                points: &points,
                                challenges: &challenges,
                            },
                        )
                        .expect("prepare the stage-7 trusted-advice address-phase kernel");
                    let prepare = start.elapsed();
                    drive_rounds(
                        &mut *kernel,
                        &claims,
                        address_rounds,
                        geometry.log_t,
                        prepare,
                        |_| RoundPhase::Address,
                    )
                }
                JoltAdviceKind::Untrusted => {
                    let cycle_relation =
                        UntrustedAdviceCyclePhase::<Fr>::new(layout, Some(r_val.clone()));
                    let cycle_claims = UntrustedAdviceCyclePhaseInputClaims::default();
                    let cycle_points = UntrustedAdviceCyclePhaseInputClaims::default();
                    let mut cycle_kernel = selected
                        .untrusted_advice_cycle
                        .prepare(
                            &mut session,
                            witness,
                            ProverInputs {
                                relation: &cycle_relation,
                                claims: &cycle_claims,
                                points: &cycle_points,
                                challenges: &challenges,
                            },
                        )
                        .expect("prepare the stage-6b untrusted-advice cycle-phase kernel");
                    let _ = drive_rounds(
                        &mut *cycle_kernel,
                        &cycle_claims,
                        cycle_rounds,
                        geometry.log_t,
                        Duration::ZERO,
                        |_| RoundPhase::Cycle,
                    );
                    cycle_kernel.park_residue(&mut session);

                    let relation = UntrustedAdviceAddressPhase::<Fr>::new(
                        layout,
                        Some(r_val),
                        cycle_variables,
                    );
                    let claims = UntrustedAdviceAddressPhaseInputClaims::default();
                    let points = UntrustedAdviceAddressPhaseInputClaims::default();
                    let start = Instant::now();
                    let mut kernel = selected
                        .untrusted_advice_address
                        .prepare(
                            &mut session,
                            witness,
                            ProverInputs {
                                relation: &relation,
                                claims: &claims,
                                points: &points,
                                challenges: &challenges,
                            },
                        )
                        .expect("prepare the stage-7 untrusted-advice address-phase kernel");
                    let prepare = start.elapsed();
                    drive_rounds(
                        &mut *kernel,
                        &claims,
                        address_rounds,
                        geometry.log_t,
                        prepare,
                        |_| RoundPhase::Address,
                    )
                }
            }
        },
    )
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
fn measure_bytecode_reduction_cycle(
    workload: Workload,
    scale: u32,
    backend: BackendKind,
    bytecode_chunks: usize,
) -> VerticalTiming {
    with_precommitted_fixture(
        workload,
        scale,
        bytecode_chunks,
        |witness, schedule, geometry| {
            let layout: &BytecodeClaimReductionLayout = schedule
                .bytecode
                .as_ref()
                .expect("committed bytecode layout present");
            let reduction = layout.precommitted();
            report_precommitted_geometry(
                "bytecode-reduction-cycle",
                reduction,
                2 + geometry.bytecode_chunk_count,
            );

            let relation = BytecodeReductionCyclePhase::<Fr>::new(
                layout,
                bytecode_reduction_weights_fixture(layout),
            );
            let claims = BytecodeReductionCyclePhaseInputClaims::default();
            let points = BytecodeReductionCyclePhaseInputClaims::default();
            let challenges = BytecodeReductionCyclePhaseChallenges {
                eta: Fr::from_u64(101),
            };
            let selected = selected_backend(backend);
            let mut session = ProofSession::default();
            let start = Instant::now();
            let mut kernel = selected
                .bytecode_reduction_cycle
                .prepare(
                    &mut session,
                    witness,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &challenges,
                    },
                )
                .expect("prepare the stage-6b committed-bytecode cycle-phase kernel");
            let prepare = start.elapsed();
            drive_rounds(
                &mut *kernel,
                &claims,
                reduction.cycle_phase_total_rounds(),
                geometry.log_t,
                prepare,
                |_| RoundPhase::Cycle,
            )
        },
    )
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
fn measure_bytecode_reduction_address(
    workload: Workload,
    scale: u32,
    backend: BackendKind,
    bytecode_chunks: usize,
) -> VerticalTiming {
    with_precommitted_fixture(
        workload,
        scale,
        bytecode_chunks,
        |witness, schedule, geometry| {
            let layout: &BytecodeClaimReductionLayout = schedule
                .bytecode
                .as_ref()
                .expect("committed bytecode layout present");
            let reduction = layout.precommitted();
            if reduction.num_address_phase_rounds() == 0 {
                return absent_address_phase("bytecode-reduction-address", reduction, geometry);
            }
            report_precommitted_geometry(
                "bytecode-reduction-address",
                reduction,
                2 + geometry.bytecode_chunk_count,
            );

            let weights = bytecode_reduction_weights_fixture(layout);
            let cycle_relation = BytecodeReductionCyclePhase::<Fr>::new(layout, weights.clone());
            let cycle_claims = BytecodeReductionCyclePhaseInputClaims::default();
            let cycle_points = BytecodeReductionCyclePhaseInputClaims::default();
            let cycle_challenges = BytecodeReductionCyclePhaseChallenges {
                eta: Fr::from_u64(101),
            };
            let selected = selected_backend(backend);
            let mut session = ProofSession::default();
            let mut cycle_kernel = selected
                .bytecode_reduction_cycle
                .prepare(
                    &mut session,
                    witness,
                    ProverInputs {
                        relation: &cycle_relation,
                        claims: &cycle_claims,
                        points: &cycle_points,
                        challenges: &cycle_challenges,
                    },
                )
                .expect("prepare the stage-6b committed-bytecode cycle-phase kernel");
            let _ = drive_rounds(
                &mut *cycle_kernel,
                &cycle_claims,
                reduction.cycle_phase_total_rounds(),
                geometry.log_t,
                Duration::ZERO,
                |_| RoundPhase::Cycle,
            );
            cycle_kernel.park_residue(&mut session);

            let relation = BytecodeReductionAddressPhase::<Fr>::new(
                layout,
                Some(weights),
                synthetic_cycle_variables(reduction),
            );
            let claims = BytecodeReductionAddressPhaseInputClaims::default();
            let points = BytecodeReductionAddressPhaseInputClaims::default();
            let challenges = NoChallenges::default();
            let start = Instant::now();
            let mut kernel = selected
                .bytecode_reduction_address
                .prepare(
                    &mut session,
                    witness,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &challenges,
                    },
                )
                .expect("prepare the stage-7 committed-bytecode address-phase kernel");
            let prepare = start.elapsed();
            drive_rounds(
                &mut *kernel,
                &claims,
                reduction.address_phase_total_rounds(),
                geometry.log_t,
                prepare,
                |_| RoundPhase::Address,
            )
        },
    )
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
fn measure_program_image_reduction_cycle(
    workload: Workload,
    scale: u32,
    backend: BackendKind,
    bytecode_chunks: usize,
) -> VerticalTiming {
    with_precommitted_fixture(
        workload,
        scale,
        bytecode_chunks,
        |witness, schedule, geometry| {
            let layout: &ProgramImageClaimReductionLayout = schedule
                .program_image
                .as_ref()
                .expect("program image layout present");
            let reduction = layout.precommitted();
            report_precommitted_geometry("program-image-reduction-cycle", reduction, 2);

            let relation = ProgramImageReductionCyclePhase::<Fr>::new(
                layout,
                synthetic_point(geometry.ram_log_k, 13),
            );
            let claims = ProgramImageReductionCyclePhaseInputClaims::default();
            let points = ProgramImageReductionCyclePhaseInputClaims::default();
            let challenges = NoChallenges::default();
            let selected = selected_backend(backend);
            let mut session = ProofSession::default();
            let start = Instant::now();
            let mut kernel = selected
                .program_image_reduction_cycle
                .prepare(
                    &mut session,
                    witness,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &challenges,
                    },
                )
                .expect("prepare the stage-6b program-image cycle-phase kernel");
            let prepare = start.elapsed();
            drive_rounds(
                &mut *kernel,
                &claims,
                reduction.cycle_phase_total_rounds(),
                geometry.log_t,
                prepare,
                |_| RoundPhase::Cycle,
            )
        },
    )
}

#[expect(
    clippy::expect_used,
    reason = "measurement harness: fixture and kernel errors fail loudly"
)]
fn measure_program_image_reduction_address(
    workload: Workload,
    scale: u32,
    backend: BackendKind,
    bytecode_chunks: usize,
) -> VerticalTiming {
    with_precommitted_fixture(
        workload,
        scale,
        bytecode_chunks,
        |witness, schedule, geometry| {
            let layout: &ProgramImageClaimReductionLayout = schedule
                .program_image
                .as_ref()
                .expect("program image layout present");
            let reduction = layout.precommitted();
            if reduction.num_address_phase_rounds() == 0 {
                return absent_address_phase(
                    "program-image-reduction-address",
                    reduction,
                    geometry,
                );
            }
            report_precommitted_geometry("program-image-reduction-address", reduction, 2);

            let r_addr_rw = synthetic_point(geometry.ram_log_k, 13);
            let cycle_relation =
                ProgramImageReductionCyclePhase::<Fr>::new(layout, r_addr_rw.clone());
            let cycle_claims = ProgramImageReductionCyclePhaseInputClaims::default();
            let cycle_points = ProgramImageReductionCyclePhaseInputClaims::default();
            let challenges = NoChallenges::default();
            let selected = selected_backend(backend);
            let mut session = ProofSession::default();
            let mut cycle_kernel = selected
                .program_image_reduction_cycle
                .prepare(
                    &mut session,
                    witness,
                    ProverInputs {
                        relation: &cycle_relation,
                        claims: &cycle_claims,
                        points: &cycle_points,
                        challenges: &challenges,
                    },
                )
                .expect("prepare the stage-6b program-image cycle-phase kernel");
            let _ = drive_rounds(
                &mut *cycle_kernel,
                &cycle_claims,
                reduction.cycle_phase_total_rounds(),
                geometry.log_t,
                Duration::ZERO,
                |_| RoundPhase::Cycle,
            );
            cycle_kernel.park_residue(&mut session);

            let relation = ProgramImageReductionAddressPhase::<Fr>::new(
                layout,
                Some(r_addr_rw),
                synthetic_cycle_variables(reduction),
            );
            let claims = ProgramImageReductionAddressPhaseInputClaims::default();
            let points = ProgramImageReductionAddressPhaseInputClaims::default();
            let start = Instant::now();
            let mut kernel = selected
                .program_image_reduction_address
                .prepare(
                    &mut session,
                    witness,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &challenges,
                    },
                )
                .expect("prepare the stage-7 program-image address-phase kernel");
            let prepare = start.elapsed();
            drive_rounds(
                &mut *kernel,
                &claims,
                reduction.address_phase_total_rounds(),
                geometry.log_t,
                prepare,
                |_| RoundPhase::Address,
            )
        },
    )
}

#[expect(
    clippy::expect_used,
    clippy::panic,
    reason = "measurement harness: fixture errors and unsupported relations fail loudly"
)]
fn measure_legacy_precommitted(
    relation: VerticalRelation,
    workload: Workload,
    scale: u32,
    bytecode_chunks: usize,
) -> VerticalTiming {
    let PrecommittedFixture {
        fixture,
        schedule,
        bytecode_chunk_count,
    } = precommitted_fixture(workload, scale, bytecode_chunks);
    let program_image_words =
        program_image_words_padded(&fixture.program_preprocessing.ram.bytecode_words);
    let start_index = fixture
        .memory_layout
        .remapped_word_address(fixture.min_bytecode_address)
        .expect("program image start index") as usize;
    let inputs = LegacyPrecommittedInputs {
        log_t: fixture.log_t,
        log_k_chunk: fixture.config.one_hot_config.committed_chunk_bits(),
        trace_length: fixture.config.trace_length,
        ram_k: fixture.config.ram_K,
        bytecode: &fixture.program_preprocessing.bytecode.bytecode,
        bytecode_chunk_count,
        program_image_words: &program_image_words,
        program_image_start_index: start_index,
        max_trusted_advice_size: fixture.memory_layout.max_trusted_advice_size as usize,
        max_untrusted_advice_size: fixture.memory_layout.max_untrusted_advice_size as usize,
    };

    let absent = |reduction: &PrecommittedClaimReduction| {
        absent_address_phase(
            relation.as_str(),
            reduction,
            PrecommittedGeometry {
                log_t: fixture.log_t,
                ram_log_k: inputs.ram_k.ilog2() as usize,
                bytecode_chunk_count,
            },
        )
    };

    let (baseline, address) = match relation {
        VerticalRelation::TrustedAdviceCycle => (
            advice_baseline(&inputs, &schedule, AdviceKind::Trusted, false),
            false,
        ),
        VerticalRelation::UntrustedAdviceCycle => (
            advice_baseline(&inputs, &schedule, AdviceKind::Untrusted, false),
            false,
        ),
        VerticalRelation::TrustedAdviceAddress => {
            let layout = schedule
                .advice(JoltAdviceKind::Trusted)
                .expect("advice layout");
            if layout.precommitted().num_address_phase_rounds() == 0 {
                return absent(layout.precommitted());
            }
            (
                advice_baseline(&inputs, &schedule, AdviceKind::Trusted, true),
                true,
            )
        }
        VerticalRelation::UntrustedAdviceAddress => {
            let layout = schedule
                .advice(JoltAdviceKind::Untrusted)
                .expect("advice layout");
            if layout.precommitted().num_address_phase_rounds() == 0 {
                return absent(layout.precommitted());
            }
            (
                advice_baseline(&inputs, &schedule, AdviceKind::Untrusted, true),
                true,
            )
        }
        VerticalRelation::BytecodeReductionCycle => {
            (bytecode_baseline(&inputs, &schedule, false), false)
        }
        VerticalRelation::BytecodeReductionAddress => {
            let layout = schedule.bytecode.as_ref().expect("bytecode layout");
            if layout.precommitted().num_address_phase_rounds() == 0 {
                return absent(layout.precommitted());
            }
            (bytecode_baseline(&inputs, &schedule, true), true)
        }
        VerticalRelation::ProgramImageReductionCycle => {
            (program_image_baseline(&inputs, &schedule, false), false)
        }
        VerticalRelation::ProgramImageReductionAddress => {
            let layout = schedule
                .program_image
                .as_ref()
                .expect("program image layout");
            if layout.precommitted().num_address_phase_rounds() == 0 {
                return absent(layout.precommitted());
            }
            (program_image_baseline(&inputs, &schedule, true), true)
        }
        other => panic!(
            "--legacy covers only the eight precommitted claim-reduction arms; {} has a \
             trace-derived baseline via scripts/legacy_relation_baseline.py",
            other.as_str()
        ),
    };

    VerticalTiming {
        log_t: fixture.log_t,
        prepare: baseline.prepare,
        address: if address {
            baseline.rounds
        } else {
            Duration::ZERO
        },
        handoff: Duration::ZERO,
        cycle: if address {
            Duration::ZERO
        } else {
            baseline.rounds
        },
        claims: Duration::ZERO,
    }
}
