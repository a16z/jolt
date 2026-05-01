#![allow(clippy::print_stderr, clippy::print_stdout)]

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;

use clap::Parser;
use common::constants::{RAM_START_ADDRESS, XLEN};
use common::jolt_device::JoltDevice;
use jolt_bench::measure::{median, time_it};
use jolt_bench::programs::Program;
use jolt_compiler_v2::{
    build_commitment_protocol, build_stage1_outer_protocol, build_stage2_protocol,
    commitment_cpu_program, lower_commitment_to_compute, lower_compute_to_cpu,
    lower_piop_and_fiat_shamir, lower_stage1_to_compute, lower_stage2_to_compute,
    project_prover_party, project_verifier_party, resolve_compute_kernels, stage1_cpu_program,
    stage2_cpu_program, CommitmentCpuProgram, JoltProtocolParams, MeliorContext,
    OptionalSkipPolicy, OracleGeneration, Role, TranscriptStep,
};
use jolt_compiler_v2::{
    Stage1CpuProgram as CompilerStage1CpuProgram, Stage2CpuProgram as CompilerStage2CpuProgram,
};
use jolt_core::curve::Bn254Curve;
use jolt_core::host;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme as CoreCommitmentScheme;
use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryGlobals};
use jolt_core::poly::unipoly::UniPoly;
use jolt_core::subprotocols::univariate_skip::{
    UniSkipFirstRoundProof, UniSkipFirstRoundProofVariant,
};
use jolt_core::transcripts::Blake2bTranscript as CoreBlake2bTranscript;
use jolt_core::zkvm::instruction::LookupQuery;
use jolt_core::zkvm::proof_serialization::{Claims as CoreClaims, JoltProof as CoreJoltProof};
use jolt_core::zkvm::prover::{JoltCpuProver, JoltProverPreprocessing};
use jolt_core::zkvm::r1cs::inputs::ProductCycleInputs as CoreProductCycleInputs;
use jolt_core::zkvm::ram::remap_address;
use jolt_core::zkvm::verifier::{JoltSharedPreprocessing, JoltVerifier, JoltVerifierPreprocessing};
use jolt_dory::{DoryCommitment, DoryHint, DoryProverSetup, DoryScheme};
use jolt_field::signed::{S128, S64};
use jolt_field::{Field, Fr};
use jolt_host::{extract_trace, BytecodePreprocessing, CycleRow, Program as HostProgram};
use jolt_inlines_keccak256 as _;
use jolt_inlines_sha2 as _;
use jolt_instructions::flags::{CircuitFlags, InstructionFlags};
use jolt_kernels::stage1::{
    execute_stage1_program, Stage1CpuProgramPlan as KernelStage1CpuProgramPlan,
    Stage1ExecutionArtifacts, Stage1ExecutionMode, Stage1KernelPlan as KernelStage1KernelPlan,
    Stage1OpeningBatchPlan as KernelStage1OpeningBatchPlan,
    Stage1OpeningClaimPlan as KernelStage1OpeningClaimPlan, Stage1OuterRemainingEvaluator,
    Stage1OuterRv64Data, Stage1Params, Stage1Proof, Stage1ProverInputs, Stage1ProverKernelExecutor,
    Stage1Rv64Cycle, Stage1SumcheckBatchPlan as KernelStage1SumcheckBatchPlan,
    Stage1SumcheckClaimPlan as KernelStage1SumcheckClaimPlan,
    Stage1SumcheckDriverPlan as KernelStage1SumcheckDriverPlan,
    Stage1SumcheckEvalPlan as KernelStage1SumcheckEvalPlan,
    Stage1SumcheckInstanceResultPlan as KernelStage1SumcheckInstanceResultPlan,
    Stage1TranscriptSqueezePlan, Stage1VerifierKernelExecutor,
};
use jolt_kernels::stage2::{
    execute_stage2_program, product_virtual_uniskip_extended_evals,
    Stage2ChallengeExtractPlan as KernelStage2ChallengeExtractPlan,
    Stage2CpuProgramPlan as KernelStage2CpuProgramPlan, Stage2ExecutionArtifacts,
    Stage2ExecutionMode, Stage2FieldConstantPlan as KernelStage2FieldConstantPlan,
    Stage2FieldExprPlan as KernelStage2FieldExprPlan, Stage2InstructionLookupCycle,
    Stage2KernelPlan as KernelStage2KernelPlan,
    Stage2OpeningBatchPlan as KernelStage2OpeningBatchPlan,
    Stage2OpeningClaimPlan as KernelStage2OpeningClaimPlan,
    Stage2OpeningInputPlan as KernelStage2OpeningInputPlan, Stage2OpeningInputValue, Stage2Params,
    Stage2PointConcatPlan as KernelStage2PointConcatPlan,
    Stage2PointSlicePlan as KernelStage2PointSlicePlan, Stage2ProductVirtualCycle,
    Stage2ProgramStepPlan as KernelStage2ProgramStepPlan, Stage2Proof, Stage2ProverInputs,
    Stage2ProverKernelExecutor, Stage2RamAccess, Stage2RamData, Stage2RamOutputLayout,
    Stage2SumcheckBatchPlan as KernelStage2SumcheckBatchPlan,
    Stage2SumcheckClaimPlan as KernelStage2SumcheckClaimPlan,
    Stage2SumcheckDriverPlan as KernelStage2SumcheckDriverPlan,
    Stage2SumcheckEvalPlan as KernelStage2SumcheckEvalPlan,
    Stage2SumcheckInstanceResultPlan as KernelStage2SumcheckInstanceResultPlan,
    Stage2TranscriptSqueezePlan as KernelStage2TranscriptSqueezePlan, Stage2VerifierKernelExecutor,
};
use jolt_r1cs::{constraints::rv64, R1csKey};
use jolt_transcript::{
    AppendToTranscript, Blake2bTranscript, Label, LabelWithCount, Transcript, U64Word,
};
use jolt_verifier::TRANSCRIPT_LABEL;
use jolt_witness::CycleInput;
use jolt_witness_v2::{
    dense_i128_column_to_field, one_hot_chunk_address_major, optional_field_oracle,
};
use serde::Serialize;
use tracer::instruction::RAMAccess;

type CoreFr = ark_bn254::Fr;
type CoreCommitment = <DoryCommitmentScheme as CoreCommitmentScheme>::Commitment;
type CoreProver<'a> =
    JoltCpuProver<'a, CoreFr, Bn254Curve, DoryCommitmentScheme, CoreBlake2bTranscript>;
type CoreProof = CoreJoltProof<CoreFr, Bn254Curve, DoryCommitmentScheme, CoreBlake2bTranscript>;
type CoreVerifier<'a> =
    JoltVerifier<'a, CoreFr, Bn254Curve, DoryCommitmentScheme, CoreBlake2bTranscript>;
type CoreVerifierPreprocessing =
    JoltVerifierPreprocessing<CoreFr, Bn254Curve, DoryCommitmentScheme>;

#[derive(Debug, Parser)]
#[command(
    name = "bolt-stage",
    about = "Correctness-gated stage perf oracle for jolt-core vs Bolt"
)]
struct Cli {
    #[arg(long, default_value_t = Program::Sha2Chain)]
    program: Program,

    #[arg(long, value_enum, default_value_t = BenchStage::Stage1)]
    stage: BenchStage,

    #[arg(long, default_value_t = 16)]
    log_t: usize,

    #[arg(long, default_value_t = 1)]
    num_iters: u32,

    #[arg(long, default_value_t = 1)]
    iters: usize,

    #[arg(long, default_value_t = 0)]
    warmup: usize,

    #[arg(long, default_value_t = 10.0)]
    bolt_timeout_multiplier: f64,

    #[arg(long)]
    json: Option<PathBuf>,

    #[arg(long, value_name = "NAME")]
    trace_chrome: Option<String>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum BenchStage {
    Stage1,
    Stage2,
}

impl BenchStage {
    fn name(self) -> &'static str {
        match self {
            Self::Stage1 => "stage1",
            Self::Stage2 => "stage2",
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum TranscriptEvent {
    Append {
        bytes: Vec<u8>,
        state_after: [u8; 32],
    },
    Squeeze {
        state_after: [u8; 32],
    },
}

#[derive(Clone)]
struct RecordingTranscript<T: Transcript> {
    inner: T,
    log: Vec<TranscriptEvent>,
}

impl<T: Transcript> RecordingTranscript<T> {
    fn log(&self) -> &[TranscriptEvent] {
        &self.log
    }
}

impl<T: Transcript> Default for RecordingTranscript<T> {
    fn default() -> Self {
        Self {
            inner: T::default(),
            log: Vec::new(),
        }
    }
}

impl<T: Transcript> Transcript for RecordingTranscript<T> {
    type Challenge = T::Challenge;

    fn new(label: &'static [u8]) -> Self {
        Self {
            inner: T::new(label),
            log: Vec::new(),
        }
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        self.inner.append_bytes(bytes);
        self.log.push(TranscriptEvent::Append {
            bytes: bytes.to_vec(),
            state_after: *self.inner.state(),
        });
    }

    fn challenge(&mut self) -> Self::Challenge {
        let challenge = self.inner.challenge();
        self.log.push(TranscriptEvent::Squeeze {
            state_after: *self.inner.state(),
        });
        challenge
    }

    fn state(&self) -> &[u8; 32] {
        self.inner.state()
    }
}

#[derive(Clone, Debug)]
struct CommitmentRecord {
    artifact: String,
}

#[derive(Clone, Debug)]
struct BoltCommitmentTrace {
    commitments: Vec<Option<DoryCommitment>>,
    records: Vec<CommitmentRecord>,
}

struct CoreStage1Fixture {
    params: JoltProtocolParams,
    pcs_setup: DoryProverSetup,
    proof: CoreProof,
    verifier_preprocessing: &'static CoreVerifierPreprocessing,
    io: JoltDevice,
    entry_address: u64,
    cycle_inputs: Vec<CycleInput>,
    r1cs_witness: Vec<Fr>,
    rv64_cycles: Vec<Stage1Rv64Cycle>,
    product_virtual_cycles: Vec<Stage2ProductVirtualCycle>,
    instruction_lookup_cycles: Vec<Stage2InstructionLookupCycle>,
    ram_accesses: Vec<Stage2RamAccess>,
    initial_ram_state: Vec<u64>,
    final_ram_state: Vec<u64>,
    ram_start_address: u64,
    ram_output_layout: Stage2RamOutputLayout,
    commitments: Vec<CoreCommitment>,
}

#[derive(Serialize)]
struct Stage1BenchReport {
    program: String,
    stage: &'static str,
    max_log_t: usize,
    actual_log_t: usize,
    trace_length: usize,
    num_iters: u32,
    iters: usize,
    warmup: usize,
    #[serde(skip_serializing_if = "Option::is_none")]
    correctness: Option<CorrectnessReport>,
    runs: Vec<Stage1Run>,
}

#[derive(Serialize)]
struct CorrectnessReport {
    bolt_prover_verifier_transcript: bool,
    core_accepts_bolt_stage: bool,
    core_bolt_stage_transcript_states: bool,
    commitment_parity: bool,
}

#[derive(Serialize)]
struct Stage1Run {
    stack: &'static str,
    stage: &'static str,
    #[serde(skip_serializing_if = "Option::is_none")]
    stage_ms: Option<f64>,
    samples_ms: Vec<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    ratio_vs_core: Option<f64>,
    #[serde(skip_serializing_if = "std::ops::Not::not", default)]
    timed_out: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    timeout_ms: Option<f64>,
}

fn main() {
    let mut cli = Cli::parse();
    let _tracing_guards = cli.trace_chrome.as_deref().map(|name| {
        cli.iters = 1;
        cli.warmup = 0;
        jolt_profiling::setup_tracing(&[jolt_profiling::TracingFormat::Chrome], name)
    });
    assert!(cli.iters > 0, "--iters must be positive");
    assert!(
        cli.bolt_timeout_multiplier > 0.0,
        "--bolt-timeout-multiplier must be positive"
    );

    let fixture = core_stage1_fixture(cli.program, cli.log_t, cli.num_iters);
    let stage1_context = bolt_stage1_context(&fixture);
    let actual_log_t = fixture.proof.trace_length.trailing_zeros() as usize;
    let trace_length = fixture.proof.trace_length;

    let core_samples = measure_core_stage(
        cli.stage,
        cli.program,
        cli.log_t,
        cli.num_iters,
        cli.iters,
        cli.warmup,
    );
    let core_ms = median(&core_samples);
    let measured_bolt_runs = cli.iters + cli.warmup;
    let bolt_timeout_ms = core_ms * cli.bolt_timeout_multiplier * measured_bolt_runs as f64;
    let correctness = assert_stage_correctness(cli.stage, &fixture, &stage1_context);
    let bolt_result = run_bolt_measurement_with_timeout(
        cli.stage,
        fixture,
        stage1_context,
        cli.iters,
        cli.warmup,
        Duration::from_secs_f64(bolt_timeout_ms / 1000.0),
    );

    let (bolt_run, timed_out) = match bolt_result {
        Ok(bolt_samples) => {
            let bolt_ms = median(&bolt_samples);
            (
                Stage1Run {
                    stack: "bolt",
                    stage: cli.stage.name(),
                    stage_ms: Some(bolt_ms),
                    samples_ms: bolt_samples,
                    ratio_vs_core: Some(bolt_ms / core_ms),
                    timed_out: false,
                    timeout_ms: Some(bolt_timeout_ms),
                },
                false,
            )
        }
        Err(reason) => {
            eprintln!("{reason}");
            (
                Stage1Run {
                    stack: "bolt",
                    stage: cli.stage.name(),
                    stage_ms: None,
                    samples_ms: Vec::new(),
                    ratio_vs_core: None,
                    timed_out: true,
                    timeout_ms: Some(bolt_timeout_ms),
                },
                true,
            )
        }
    };

    let report = Stage1BenchReport {
        program: cli.program.cli_name().to_string(),
        stage: cli.stage.name(),
        max_log_t: cli.log_t,
        actual_log_t,
        trace_length,
        num_iters: cli.num_iters,
        iters: cli.iters,
        warmup: cli.warmup,
        correctness: Some(correctness),
        runs: vec![
            Stage1Run {
                stack: "core",
                stage: cli.stage.name(),
                stage_ms: Some(core_ms),
                samples_ms: core_samples,
                ratio_vs_core: None,
                timed_out: false,
                timeout_ms: None,
            },
            bolt_run,
        ],
    };

    let json = serde_json::to_string_pretty(&report).expect("serialize Stage 1 bench report");
    if let Some(path) = cli.json {
        std::fs::write(&path, json).expect("write Stage 1 bench JSON");
    } else {
        println!("{json}");
    }

    if timed_out {
        std::process::exit(2);
    }
}

struct BoltStage1Context {
    commitment_prover_program: CommitmentCpuProgram,
    commitment_verifier_program: CommitmentCpuProgram,
    commitment_prover_trace: BoltCommitmentTrace,
    commitment_verifier_trace: BoltCommitmentTrace,
    stage1_prover_plan: &'static KernelStage1CpuProgramPlan,
    stage1_verifier_plan: &'static KernelStage1CpuProgramPlan,
    stage2_prover_plan: &'static KernelStage2CpuProgramPlan,
    num_cycle_vars: usize,
}

fn run_bolt_measurement_with_timeout(
    stage: BenchStage,
    fixture: CoreStage1Fixture,
    context: BoltStage1Context,
    iters: usize,
    warmup: usize,
    timeout: Duration,
) -> Result<Vec<f64>, String> {
    let (sender, receiver) = mpsc::channel();
    let timeout_ms = timeout.as_secs_f64() * 1000.0;
    let _worker = std::thread::spawn(move || {
        let result = std::panic::catch_unwind(|| {
            measure_bolt_stage(stage, &fixture, &context, iters, warmup)
        });
        let _ = sender.send(result);
    });

    match receiver.recv_timeout(timeout) {
        Ok(Ok(result)) => Ok(result),
        Ok(Err(_panic)) => Err(format!("Bolt {} worker panicked", stage.name())),
        Err(mpsc::RecvTimeoutError::Timeout) => Err(format!(
            "Bolt {} exceeded timeout budget ({timeout_ms:.3}ms)",
            stage.name()
        )),
        Err(mpsc::RecvTimeoutError::Disconnected) => {
            Err("Bolt Stage 1 worker disconnected".to_string())
        }
    }
}

fn core_stage1_fixture(program: Program, log_t: usize, num_iters: u32) -> CoreStage1Fixture {
    DoryGlobals::reset();

    let max_trace_length = 1usize << log_t;
    let inputs = program.canonical_inputs_with(Some(num_iters));
    let mut core_program = host::Program::new(program.guest_name());
    let (core_bytecode, init_memory_state, _, entry_address) = core_program.decode();
    let (_, _, _, core_io_device) = core_program.trace(&inputs, &[], &[]);
    let shared_preprocessing = JoltSharedPreprocessing::new(
        core_bytecode,
        core_io_device.memory_layout.clone(),
        init_memory_state,
        max_trace_length,
        entry_address,
    );
    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing);
    let elf_contents = core_program.get_elf_contents().expect("guest ELF");
    let prover: CoreProver<'_> = CoreProver::gen_from_elf(
        &prover_preprocessing,
        &elf_contents,
        &inputs,
        &[],
        &[],
        None,
        None,
        None,
    );
    let io = prover.program_io.clone();
    let initial_ram_state = prover.initial_ram_state.clone();
    let final_ram_state = prover.final_ram_state.clone();
    let (proof, _debug): (CoreProof, _) = prover.prove();
    let commitments = proof.commitments.clone();
    let verifier_preprocessing: &'static _ = Box::leak(Box::new(JoltVerifierPreprocessing::from(
        &prover_preprocessing,
    )));

    let mut host_program = HostProgram::new(program.guest_name());
    let (bytecode_raw, _, _, host_entry_address) = host_program.decode();
    let (_, trace, _, host_io_device) = host_program.trace(&inputs, &[], &[]);
    let mut padded_trace = trace.clone();
    padded_trace.resize(proof.trace_length, jolt_host::Cycle::NoOp);
    let product_virtual_cycles = (0..proof.trace_length)
        .map(|index| {
            let row = CoreProductCycleInputs::from_trace::<CoreFr>(&padded_trace, index);
            Stage2ProductVirtualCycle {
                instruction_left_input: row.instruction_left_input,
                instruction_right_input: row.instruction_right_input,
                should_branch_lookup_output: row.should_branch_lookup_output,
                write_lookup_output_to_rd_flag: row.write_lookup_output_to_rd_flag,
                jump_flag: row.jump_flag,
                should_branch_flag: row.should_branch_flag,
                not_next_noop: row.not_next_noop,
                virtual_instruction_flag: row.virtual_instruction_flag,
            }
        })
        .collect();
    let instruction_lookup_cycles = padded_trace
        .iter()
        .map(|cycle| {
            let (left_instruction_input, right_instruction_input) =
                LookupQuery::<XLEN>::to_instruction_inputs(cycle);
            let (left_lookup_operand, right_lookup_operand) =
                LookupQuery::<XLEN>::to_lookup_operands(cycle);
            let lookup_output = LookupQuery::<XLEN>::to_lookup_output(cycle);
            Stage2InstructionLookupCycle {
                lookup_output,
                left_lookup_operand,
                right_lookup_operand,
                left_instruction_input,
                right_instruction_input,
            }
        })
        .collect();
    let ram_accesses = padded_trace
        .iter()
        .map(|cycle| match cycle.ram_access() {
            RAMAccess::Read(read) => Stage2RamAccess {
                remapped_address: remap_address(read.address, &host_io_device.memory_layout)
                    .map(|address| address as usize),
                read_value: read.value,
                write_value: read.value,
            },
            RAMAccess::Write(write) => Stage2RamAccess {
                remapped_address: remap_address(write.address, &host_io_device.memory_layout)
                    .map(|address| address as usize),
                read_value: write.pre_value,
                write_value: write.post_value,
            },
            RAMAccess::NoOp => Stage2RamAccess::noop(),
        })
        .collect();
    let ram_start_address = host_io_device.memory_layout.get_lowest_address();
    let ram_output_layout = Stage2RamOutputLayout {
        io_start: remap_address(
            host_io_device.memory_layout.input_start,
            &host_io_device.memory_layout,
        )
        .expect("input start remaps") as usize,
        io_end: remap_address(RAM_START_ADDRESS, &host_io_device.memory_layout)
            .expect("RAM start remaps") as usize,
    };
    let bytecode = BytecodePreprocessing::preprocess(bytecode_raw, host_entry_address);
    let r1cs_key = R1csKey::new(rv64::rv64_constraints::<Fr>(), proof.trace_length);
    let (cycle_inputs, r1cs_witness, _) = extract_trace::<_, Fr>(
        &trace,
        proof.trace_length,
        &bytecode,
        &host_io_device.memory_layout,
        r1cs_key.num_vars_padded,
    );
    let rv64_cycles = stage1_rv64_cycles(&trace, proof.trace_length, &bytecode);

    let log_t = proof.trace_length.trailing_zeros() as usize;
    let log_k_bytecode = prover_preprocessing
        .shared
        .bytecode
        .code_size
        .trailing_zeros() as usize;
    let log_k_ram = proof.ram_K.trailing_zeros() as usize;
    let params = JoltProtocolParams::new(log_t, log_k_bytecode, log_k_ram);

    CoreStage1Fixture {
        params,
        pcs_setup: DoryProverSetup(prover_preprocessing.generators.clone()),
        proof,
        verifier_preprocessing,
        io,
        entry_address,
        cycle_inputs,
        r1cs_witness,
        rv64_cycles,
        product_virtual_cycles,
        instruction_lookup_cycles,
        ram_accesses,
        initial_ram_state,
        final_ram_state,
        ram_start_address,
        ram_output_layout,
        commitments,
    }
}

fn stage1_rv64_cycles<C: CycleRow>(
    trace: &[C],
    size: usize,
    bytecode: &BytecodePreprocessing,
) -> Vec<Stage1Rv64Cycle> {
    (0..size)
        .map(|cycle| stage1_rv64_cycle(trace, cycle, bytecode))
        .collect()
}

fn stage1_rv64_cycle<C: CycleRow>(
    trace: &[C],
    cycle_index: usize,
    bytecode: &BytecodePreprocessing,
) -> Stage1Rv64Cycle {
    let Some(cycle) = trace.get(cycle_index) else {
        return Stage1Rv64Cycle::padding();
    };
    let next = trace.get(cycle_index + 1);
    if cycle.is_noop() {
        let mut row = Stage1Rv64Cycle::padding();
        fill_next_rv64_fields(&mut row, next, bytecode);
        return row;
    }

    let flags = cycle.circuit_flags();
    let instruction_flags = cycle.instruction_flags();
    let left_input = if instruction_flags[InstructionFlags::LeftOperandIsPC] {
        cycle.unexpanded_pc()
    } else if instruction_flags[InstructionFlags::LeftOperandIsRs1Value] {
        cycle.rs1_read().map_or(0, |(_, value)| value)
    } else {
        0
    };
    let right_i128 = if instruction_flags[InstructionFlags::RightOperandIsImm] {
        cycle.imm()
    } else if instruction_flags[InstructionFlags::RightOperandIsRs2Value] {
        cycle.rs2_read().map_or(0, |(_, value)| value as i128)
    } else {
        0
    };
    let right_input = s64_from_i128(right_i128);
    let product = S64::from_u64(left_input).mul_trunc::<2, 2>(&S128::from_i128(right_i128));
    let lookup_output = cycle.lookup_output();
    let (left_lookup, right_lookup) =
        lookup_operands_raw(left_input, right_i128, product, &flags, lookup_output);
    let next_is_noop = next.is_none_or(CycleRow::is_noop);

    let mut row = Stage1Rv64Cycle {
        left_input,
        right_input,
        product,
        left_lookup,
        right_lookup,
        lookup_output,
        rs1_read_value: cycle.rs1_read().map_or(0, |(_, value)| value),
        rs2_read_value: cycle.rs2_read().map_or(0, |(_, value)| value),
        rd_write_value: cycle.rd_write().map_or(0, |(_, _, post)| post),
        ram_addr: cycle.ram_access_address().unwrap_or(0),
        ram_read_value: cycle.ram_read_value().unwrap_or(0),
        ram_write_value: cycle.ram_write_value().unwrap_or(0),
        pc: bytecode.get_pc(cycle) as u64,
        next_pc: 0,
        unexpanded_pc: cycle.unexpanded_pc(),
        next_unexpanded_pc: 0,
        imm: s64_from_i128(cycle.imm()),
        flags,
        should_jump: flags[CircuitFlags::Jump] && !next_is_noop,
        should_branch: instruction_flags[InstructionFlags::Branch] && lookup_output == 1,
        next_is_virtual: false,
        next_is_first_in_sequence: false,
    };
    fill_next_rv64_fields(&mut row, next, bytecode);
    row
}

fn fill_next_rv64_fields<C: CycleRow>(
    row: &mut Stage1Rv64Cycle,
    next: Option<&C>,
    bytecode: &BytecodePreprocessing,
) {
    if let Some(next_cycle) = next {
        row.next_pc = bytecode.get_pc(next_cycle) as u64;
        row.next_unexpanded_pc = next_cycle.unexpanded_pc();
        let next_flags = next_cycle.circuit_flags();
        row.next_is_virtual = next_flags[CircuitFlags::VirtualInstruction];
        row.next_is_first_in_sequence = next_flags[CircuitFlags::IsFirstInSequence];
    }
}

fn s64_from_i128(value: i128) -> S64 {
    let magnitude = value.unsigned_abs();
    assert!(magnitude <= u64::MAX as u128, "S64 input overflow");
    S64::from_u64_with_sign(magnitude as u64, value >= 0)
}

fn lookup_operands_raw(
    left: u64,
    right: i128,
    product: S128,
    flags: &[bool; jolt_instructions::NUM_CIRCUIT_FLAGS],
    lookup_output: u64,
) -> (u64, u128) {
    if flags[CircuitFlags::AddOperands] {
        (0, (left as i128 + right) as u128)
    } else if flags[CircuitFlags::SubtractOperands] {
        (0, (left as i128 - right + (1i128 << 64)) as u128)
    } else if flags[CircuitFlags::MultiplyOperands] {
        (0, product.magnitude_as_u128())
    } else if flags[CircuitFlags::Advice] {
        (0, lookup_output as u128)
    } else {
        (left, right as u128)
    }
}

fn measure_core_stage(
    stage: BenchStage,
    program: Program,
    log_t: usize,
    num_iters: u32,
    iters: usize,
    warmup: usize,
) -> Vec<f64> {
    for _ in 0..warmup {
        let _ = core_stage_once(stage, program, log_t, num_iters);
    }
    (0..iters)
        .map(|_| core_stage_once(stage, program, log_t, num_iters))
        .collect()
}

fn core_stage_once(stage: BenchStage, program: Program, log_t: usize, num_iters: u32) -> f64 {
    DoryGlobals::reset();

    let max_trace_length = 1usize << log_t;
    let inputs = program.canonical_inputs_with(Some(num_iters));
    let mut core_program = host::Program::new(program.guest_name());
    let (core_bytecode, init_memory_state, _, entry_address) = core_program.decode();
    let (_, _, _, core_io_device) = core_program.trace(&inputs, &[], &[]);
    let shared_preprocessing = JoltSharedPreprocessing::new(
        core_bytecode,
        core_io_device.memory_layout.clone(),
        init_memory_state,
        max_trace_length,
        entry_address,
    );
    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing);
    let elf_contents = core_program.get_elf_contents().expect("guest ELF");
    let prover: CoreProver<'_> = CoreProver::gen_from_elf(
        &prover_preprocessing,
        &elf_contents,
        &inputs,
        &[],
        &[],
        None,
        None,
        None,
    );
    match stage {
        BenchStage::Stage1 => prover.benchmark_stage1_ms(),
        BenchStage::Stage2 => prover.benchmark_stage2_ms(),
    }
}

fn bolt_stage1_context(fixture: &CoreStage1Fixture) -> BoltStage1Context {
    let (commitment_prover_program, commitment_verifier_program) =
        bolt_commitment_programs_with_params(&fixture.params);
    let (stage1_prover_program, stage1_verifier_program) =
        bolt_stage1_programs_with_params(&fixture.params);
    let (stage2_prover_program, _) = bolt_stage2_programs_with_params(&fixture.params);
    let oracle_data = real_oracle_data(&commitment_prover_program, &fixture.cycle_inputs);
    let commitment_prover_trace = run_bolt_commitment_prover_with(
        &commitment_prover_program,
        &fixture.pcs_setup,
        |oracle, _num_vars| oracle_data.get(oracle).cloned().flatten(),
    );
    let commitment_verifier_trace = run_bolt_commitment_verifier(
        &commitment_verifier_program,
        &commitment_prover_trace.commitments,
    );

    BoltStage1Context {
        commitment_prover_program,
        commitment_verifier_program,
        commitment_prover_trace,
        commitment_verifier_trace,
        stage1_prover_plan: leak_stage1_program(&stage1_prover_program),
        stage1_verifier_plan: leak_stage1_program(&stage1_verifier_program),
        stage2_prover_plan: leak_stage2_program(&stage2_prover_program),
        num_cycle_vars: fixture.proof.trace_length.trailing_zeros() as usize,
    }
}

fn assert_stage_correctness(
    stage: BenchStage,
    fixture: &CoreStage1Fixture,
    context: &BoltStage1Context,
) -> CorrectnessReport {
    match stage {
        BenchStage::Stage1 => assert_stage1_correctness(fixture, context),
        BenchStage::Stage2 => assert_stage2_correctness(fixture, context),
    }
}

fn assert_stage1_correctness(
    fixture: &CoreStage1Fixture,
    context: &BoltStage1Context,
) -> CorrectnessReport {
    let r1cs_key = R1csKey::new(rv64::rv64_constraints::<Fr>(), fixture.proof.trace_length);
    let data = Stage1OuterRv64Data::new(&r1cs_key, &fixture.r1cs_witness, &fixture.rv64_cycles)
        .expect("valid R1CS witness shape");
    let mut prover_transcript = RecordingTranscript::<Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
    append_bolt_prefix(
        &mut prover_transcript,
        fixture,
        &context.commitment_prover_trace,
        &context.commitment_prover_program.transcript_steps,
    );
    let (prover_transcript, stage1_artifacts) = run_bolt_stage1_prover(
        context.stage1_prover_plan,
        context.num_cycle_vars,
        &data,
        prover_transcript,
    );
    let stage1_proof = Stage1Proof::from(stage1_artifacts.clone());

    let mut verifier_transcript =
        RecordingTranscript::<Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
    append_bolt_prefix(
        &mut verifier_transcript,
        fixture,
        &context.commitment_verifier_trace,
        &context.commitment_verifier_program.transcript_steps,
    );
    let (_, verifier_log) = run_bolt_stage1_verifier(
        context.stage1_verifier_plan,
        &stage1_proof,
        verifier_transcript,
    );
    assert_eq!(prover_transcript.log(), verifier_log.as_slice());

    assert_core_accepts_bolt_stage1(fixture, &stage1_artifacts);
    assert_core_states_match_bolt_stage1(fixture, prover_transcript.log());

    let bolt_core_commitments = context
        .commitment_prover_trace
        .commitments
        .iter()
        .filter_map(|commitment| commitment.as_ref())
        .take(fixture.commitments.len())
        .map(commitment_to_ark)
        .collect::<Vec<_>>();
    assert_eq!(bolt_core_commitments, fixture.commitments);

    CorrectnessReport {
        bolt_prover_verifier_transcript: true,
        core_accepts_bolt_stage: true,
        core_bolt_stage_transcript_states: true,
        commitment_parity: true,
    }
}

fn assert_stage2_correctness(
    fixture: &CoreStage1Fixture,
    context: &BoltStage1Context,
) -> CorrectnessReport {
    let r1cs_key = R1csKey::new(rv64::rv64_constraints::<Fr>(), fixture.proof.trace_length);
    let data = Stage1OuterRv64Data::new(&r1cs_key, &fixture.r1cs_witness, &fixture.rv64_cycles)
        .expect("valid R1CS witness shape");
    let mut prover_transcript = RecordingTranscript::<Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
    append_bolt_prefix(
        &mut prover_transcript,
        fixture,
        &context.commitment_prover_trace,
        &context.commitment_prover_program.transcript_steps,
    );
    let (prover_transcript, stage1_artifacts) = run_bolt_stage1_prover(
        context.stage1_prover_plan,
        context.num_cycle_vars,
        &data,
        prover_transcript,
    );
    let ram_data = stage2_ram_data(fixture);
    let (prover_transcript, stage2_artifacts) = run_bolt_stage2_prover(
        context.stage2_prover_plan,
        fixture,
        &stage1_artifacts,
        &ram_data,
        prover_transcript,
    );

    let stage1_proof = Stage1Proof::from(stage1_artifacts.clone());
    let stage2_proof = Stage2Proof::from(stage2_artifacts.clone());
    let mut verifier_transcript =
        RecordingTranscript::<Blake2bTranscript<Fr>>::new(TRANSCRIPT_LABEL);
    append_bolt_prefix(
        &mut verifier_transcript,
        fixture,
        &context.commitment_verifier_trace,
        &context.commitment_verifier_program.transcript_steps,
    );
    let mut stage1_verifier = Stage1VerifierKernelExecutor::new(&stage1_proof);
    let verified_stage1 = execute_stage1_program(
        context.stage1_verifier_plan,
        Stage1ExecutionMode::Verifier,
        &mut stage1_verifier,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 1 verifier accepts prover proof");
    let stage2_openings = stage2_opening_inputs(&verified_stage1);
    let mut stage2_verifier =
        Stage2VerifierKernelExecutor::new(&stage2_proof, &stage2_openings).with_ram_data(&ram_data);
    let verified_stage2 = execute_stage2_program(
        context.stage2_prover_plan,
        Stage2ExecutionMode::Verifier,
        &mut stage2_verifier,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 2 verifier accepts prover proof");

    assert_eq!(
        stage2_artifacts.sumchecks.len(),
        verified_stage2.sumchecks.len()
    );
    assert_eq!(prover_transcript.log(), verifier_transcript.log());
    assert_core_accepts_bolt_stage2(fixture, &stage1_artifacts, &stage2_artifacts);
    assert_core_states_match_bolt_stage2(fixture, prover_transcript.log());

    let bolt_core_commitments = context
        .commitment_prover_trace
        .commitments
        .iter()
        .filter_map(|commitment| commitment.as_ref())
        .take(fixture.commitments.len())
        .map(commitment_to_ark)
        .collect::<Vec<_>>();
    assert_eq!(bolt_core_commitments, fixture.commitments);

    CorrectnessReport {
        bolt_prover_verifier_transcript: true,
        core_accepts_bolt_stage: true,
        core_bolt_stage_transcript_states: true,
        commitment_parity: true,
    }
}

fn measure_bolt_stage(
    stage: BenchStage,
    fixture: &CoreStage1Fixture,
    context: &BoltStage1Context,
    iters: usize,
    warmup: usize,
) -> Vec<f64> {
    match stage {
        BenchStage::Stage1 => measure_bolt_stage1(fixture, context, iters, warmup),
        BenchStage::Stage2 => measure_bolt_stage2(fixture, context, iters, warmup),
    }
}

fn measure_bolt_stage1(
    fixture: &CoreStage1Fixture,
    context: &BoltStage1Context,
    iters: usize,
    warmup: usize,
) -> Vec<f64> {
    let r1cs_key = R1csKey::new(rv64::rv64_constraints::<Fr>(), fixture.proof.trace_length);
    let data = Stage1OuterRv64Data::new(&r1cs_key, &fixture.r1cs_witness, &fixture.rv64_cycles)
        .expect("valid R1CS witness shape");
    let mut prefix = Blake2bTranscript::<Fr>::new(TRANSCRIPT_LABEL);
    append_bolt_prefix(
        &mut prefix,
        fixture,
        &context.commitment_prover_trace,
        &context.commitment_prover_program.transcript_steps,
    );

    for _ in 0..warmup {
        let _ = run_bolt_stage1_prover(
            context.stage1_prover_plan,
            context.num_cycle_vars,
            &data,
            prefix.clone(),
        );
    }

    (0..iters)
        .map(|_| {
            time_it(|| {
                run_bolt_stage1_prover(
                    context.stage1_prover_plan,
                    context.num_cycle_vars,
                    &data,
                    prefix.clone(),
                )
            })
            .0
        })
        .collect()
}

fn measure_bolt_stage2(
    fixture: &CoreStage1Fixture,
    context: &BoltStage1Context,
    iters: usize,
    warmup: usize,
) -> Vec<f64> {
    let r1cs_key = R1csKey::new(rv64::rv64_constraints::<Fr>(), fixture.proof.trace_length);
    let data = Stage1OuterRv64Data::new(&r1cs_key, &fixture.r1cs_witness, &fixture.rv64_cycles)
        .expect("valid R1CS witness shape");
    let mut prefix = Blake2bTranscript::<Fr>::new(TRANSCRIPT_LABEL);
    append_bolt_prefix(
        &mut prefix,
        fixture,
        &context.commitment_prover_trace,
        &context.commitment_prover_program.transcript_steps,
    );
    let (stage2_prefix, stage1_artifacts) = run_bolt_stage1_prover(
        context.stage1_prover_plan,
        context.num_cycle_vars,
        &data,
        prefix,
    );
    let ram_data = stage2_ram_data(fixture);

    for _ in 0..warmup {
        let _ = run_bolt_stage2_prover(
            context.stage2_prover_plan,
            fixture,
            &stage1_artifacts,
            &ram_data,
            stage2_prefix.clone(),
        );
    }

    (0..iters)
        .map(|_| {
            time_it(|| {
                run_bolt_stage2_prover(
                    context.stage2_prover_plan,
                    fixture,
                    &stage1_artifacts,
                    &ram_data,
                    stage2_prefix.clone(),
                )
            })
            .0
        })
        .collect()
}

fn run_bolt_stage1_prover<T, E>(
    plan: &'static KernelStage1CpuProgramPlan,
    num_cycle_vars: usize,
    data: &E,
    mut transcript: T,
) -> (T, Stage1ExecutionArtifacts<Fr>)
where
    T: Transcript<Challenge = Fr>,
    E: Stage1OuterRemainingEvaluator<Fr>,
{
    let inputs = Stage1ProverInputs::empty(num_cycle_vars).with_outer_remaining_evaluator(data);
    let mut prover = Stage1ProverKernelExecutor::new(inputs);
    let artifacts = execute_stage1_program(
        plan,
        Stage1ExecutionMode::Prover,
        &mut prover,
        &mut transcript,
    )
    .expect("Bolt Stage 1 prover succeeds");
    (transcript, artifacts)
}

fn run_bolt_stage2_prover<T>(
    plan: &'static KernelStage2CpuProgramPlan,
    fixture: &CoreStage1Fixture,
    stage1_artifacts: &Stage1ExecutionArtifacts<Fr>,
    ram_data: &Stage2RamData<'_>,
    mut transcript: T,
) -> (T, Stage2ExecutionArtifacts<Fr>)
where
    T: Transcript<Challenge = Fr>,
{
    let opening_inputs = stage2_opening_inputs(stage1_artifacts);
    let tau_low = opening_inputs
        .iter()
        .find(|input| input.symbol == "stage2.input.stage1.Product")
        .expect("product opening")
        .point
        .as_slice();
    let extended_evals =
        product_virtual_uniskip_extended_evals(&fixture.product_virtual_cycles, tau_low)
            .expect("product virtual extended evals");
    let inputs = Stage2ProverInputs::new(&opening_inputs)
        .with_product_uniskip_extended_evals(&extended_evals)
        .with_product_virtual_cycles(&fixture.product_virtual_cycles)
        .with_instruction_lookup_cycles(&fixture.instruction_lookup_cycles)
        .with_ram_data(ram_data);
    let mut prover = Stage2ProverKernelExecutor::new(inputs);
    let artifacts = execute_stage2_program(
        plan,
        Stage2ExecutionMode::Prover,
        &mut prover,
        &mut transcript,
    )
    .expect("Bolt Stage 2 prover succeeds");
    (transcript, artifacts)
}

fn stage2_ram_data(fixture: &CoreStage1Fixture) -> Stage2RamData<'_> {
    Stage2RamData {
        log_k: fixture.params.log_k_ram,
        start_address: fixture.ram_start_address,
        initial_ram: &fixture.initial_ram_state,
        final_ram: &fixture.final_ram_state,
        accesses: &fixture.ram_accesses,
        output_layout: Some(fixture.ram_output_layout),
    }
}

fn stage2_opening_inputs(
    stage1_artifacts: &Stage1ExecutionArtifacts<Fr>,
) -> Vec<Stage2OpeningInputValue<Fr>> {
    [
        "Product",
        "ShouldBranch",
        "ShouldJump",
        "RamReadValue",
        "RamWriteValue",
        "LookupOutput",
        "LeftLookupOperand",
        "RightLookupOperand",
        "LeftInstructionInput",
        "RightInstructionInput",
        "RamAddress",
    ]
    .into_iter()
    .map(|oracle| {
        let source_claim = format!("stage1.outer_remaining.opening.{oracle}");
        let opening = stage1_artifacts
            .opening_value(&source_claim)
            .unwrap_or_else(|| panic!("missing Stage 1 opening {source_claim}"));
        Stage2OpeningInputValue {
            symbol: leak_str(&format!("stage2.input.stage1.{oracle}")),
            point: opening.point.clone(),
            eval: opening.eval,
        }
    })
    .collect()
}

fn run_bolt_stage1_verifier<T>(
    plan: &'static KernelStage1CpuProgramPlan,
    proof: &Stage1Proof<Fr>,
    mut transcript: T,
) -> (Stage1ExecutionArtifacts<Fr>, Vec<TranscriptEvent>)
where
    T: Transcript<Challenge = Fr> + IntoRecordingLog,
{
    let mut verifier = Stage1VerifierKernelExecutor::new(proof);
    let artifacts = execute_stage1_program(
        plan,
        Stage1ExecutionMode::Verifier,
        &mut verifier,
        &mut transcript,
    )
    .expect("Bolt Stage 1 verifier accepts prover proof");
    (artifacts, transcript.into_recording_log())
}

trait IntoRecordingLog {
    fn into_recording_log(self) -> Vec<TranscriptEvent>;
}

impl<T: Transcript> IntoRecordingLog for RecordingTranscript<T> {
    fn into_recording_log(self) -> Vec<TranscriptEvent> {
        self.log
    }
}

fn append_bolt_prefix<T>(
    transcript: &mut T,
    fixture: &CoreStage1Fixture,
    commitment_trace: &BoltCommitmentTrace,
    transcript_steps: &[TranscriptStep],
) where
    T: Transcript<Challenge = Fr>,
{
    append_bolt_preamble(
        transcript,
        &fixture.io,
        fixture.proof.ram_K,
        fixture.proof.trace_length,
        fixture.entry_address,
    );
    append_bolt_commitments_to_transcript(
        transcript,
        &commitment_trace.records,
        &commitment_trace.commitments,
        transcript_steps,
    );
}

fn bolt_commitment_programs_with_params(
    params: &JoltProtocolParams,
) -> (CommitmentCpuProgram, CommitmentCpuProgram) {
    let context = MeliorContext::new();
    let protocol = build_commitment_protocol(&context, params).expect("build protocol");
    let concrete =
        lower_piop_and_fiat_shamir(&context, &protocol).expect("lower Fiat-Shamir state");
    let prover_party = project_prover_party(&context, &concrete).expect("project prover");
    let verifier_party = project_verifier_party(&context, &concrete).expect("project verifier");
    let prover_compute =
        lower_commitment_to_compute(&context, &prover_party).expect("lower prover compute");
    let verifier_compute =
        lower_commitment_to_compute(&context, &verifier_party).expect("lower verifier compute");
    let prover_cpu = lower_compute_to_cpu(&context, &prover_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_compute).expect("lower verifier CPU");
    let prover_program = commitment_cpu_program(&prover_cpu).expect("extract prover CPU program");
    let verifier_program =
        commitment_cpu_program(&verifier_cpu).expect("extract verifier CPU program");
    (prover_program, verifier_program)
}

fn bolt_stage1_programs_with_params(
    params: &JoltProtocolParams,
) -> (CompilerStage1CpuProgram, CompilerStage1CpuProgram) {
    let context = MeliorContext::new();
    let protocol = build_stage1_outer_protocol(&context, params).expect("build stage1 protocol");
    let concrete = lower_piop_and_fiat_shamir(&context, &protocol).expect("lower Stage 1 protocol");
    let prover_party = project_prover_party(&context, &concrete).expect("project prover");
    let verifier_party = project_verifier_party(&context, &concrete).expect("project verifier");
    let prover_compute =
        lower_stage1_to_compute(&context, &prover_party).expect("lower prover Stage 1");
    let verifier_compute =
        lower_stage1_to_compute(&context, &verifier_party).expect("lower verifier Stage 1");
    let prover_compute =
        resolve_compute_kernels(&context, &prover_compute).expect("resolve prover kernels");
    let verifier_compute =
        resolve_compute_kernels(&context, &verifier_compute).expect("resolve verifier kernels");
    let prover_cpu = lower_compute_to_cpu(&context, &prover_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_compute).expect("lower verifier CPU");
    let prover_program = stage1_cpu_program(&prover_cpu).expect("extract prover Stage 1 CPU");
    let verifier_program = stage1_cpu_program(&verifier_cpu).expect("extract verifier Stage 1 CPU");
    (prover_program, verifier_program)
}

fn bolt_stage2_programs_with_params(
    params: &JoltProtocolParams,
) -> (CompilerStage2CpuProgram, CompilerStage2CpuProgram) {
    let context = MeliorContext::new();
    let protocol = build_stage2_protocol(&context, params).expect("build stage2 protocol");
    let concrete = lower_piop_and_fiat_shamir(&context, &protocol).expect("lower Stage 2 protocol");
    let prover_party = project_prover_party(&context, &concrete).expect("project prover");
    let verifier_party = project_verifier_party(&context, &concrete).expect("project verifier");
    let prover_compute =
        lower_stage2_to_compute(&context, &prover_party).expect("lower prover Stage 2");
    let verifier_compute =
        lower_stage2_to_compute(&context, &verifier_party).expect("lower verifier Stage 2");
    let prover_compute =
        resolve_compute_kernels(&context, &prover_compute).expect("resolve prover kernels");
    let verifier_compute =
        resolve_compute_kernels(&context, &verifier_compute).expect("resolve verifier kernels");
    let prover_cpu = lower_compute_to_cpu(&context, &prover_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_compute).expect("lower verifier CPU");
    let prover_program = stage2_cpu_program(&prover_cpu).expect("extract prover Stage 2 CPU");
    let verifier_program = stage2_cpu_program(&verifier_cpu).expect("extract verifier Stage 2 CPU");
    (prover_program, verifier_program)
}

fn real_oracle_data(
    program: &CommitmentCpuProgram,
    cycle_inputs: &[CycleInput],
) -> BTreeMap<String, Option<Vec<Fr>>> {
    let mut data = BTreeMap::new();
    for plan in &program.oracle_plans {
        let materialized = match &plan.generation {
            OracleGeneration::Reference => continue,
            OracleGeneration::DenseTrace { .. } => {
                let values = dense_source(cycle_inputs, &plan.source);
                Some(dense_i128_column_to_field(
                    &values,
                    target_len(plan.num_vars),
                ))
            }
            OracleGeneration::OneHotChunk {
                trace_num_vars,
                chunk,
                num_chunks,
                chunk_bits,
                padding,
                ..
            } => {
                let values = one_hot_source(cycle_inputs, &plan.source);
                Some(one_hot_chunk_address_major(
                    &values,
                    *chunk,
                    *num_chunks,
                    *chunk_bits,
                    target_len(*trace_num_vars),
                    padding_value(padding),
                ))
            }
            OracleGeneration::OptionalAdvice { .. } => {
                optional_field_oracle::<Fr>(None, target_len(plan.num_vars))
            }
        };
        let _ = data.insert(plan.oracle.clone(), materialized);
    }
    data
}

fn run_bolt_commitment_prover_with<F>(
    program: &CommitmentCpuProgram,
    setup: &DoryProverSetup,
    mut materialize: F,
) -> BoltCommitmentTrace
where
    F: FnMut(&str, usize) -> Option<Vec<Fr>>,
{
    assert_eq!(program.role, Role::Prover);
    let mut commitments = Vec::new();
    let mut records = Vec::new();

    for plan in &program.batch_plans {
        assert_eq!(plan.count, plan.oracles.len());
        for oracle in &plan.oracles {
            let oracle_num_vars = oracle_num_vars(program, oracle, plan.num_vars);
            let data = materialize(oracle, oracle_num_vars)
                .unwrap_or_else(|| panic!("missing batch oracle `{oracle}`"));
            let data = into_padded_oracle(data, oracle_num_vars);
            let (commitment, _) = commit_with_layout(&data, plan.num_vars, setup);
            records.push(CommitmentRecord {
                artifact: plan.artifact.clone(),
            });
            commitments.push(Some(commitment));
        }
    }
    for plan in &program.optional_plans {
        let oracle_num_vars = oracle_num_vars(program, &plan.oracle, plan.num_vars);
        let commitment = materialize(&plan.oracle, oracle_num_vars)
            .filter(|data| !should_skip_optional(&plan.skip_policy, data))
            .map(|data| {
                let data = into_padded_oracle(data, oracle_num_vars);
                commit_with_layout(&data, plan.num_vars, setup).0
            });
        records.push(CommitmentRecord {
            artifact: plan.artifact.clone(),
        });
        commitments.push(commitment);
    }

    BoltCommitmentTrace {
        commitments,
        records,
    }
}

fn run_bolt_commitment_verifier(
    program: &CommitmentCpuProgram,
    proof_commitments: &[Option<DoryCommitment>],
) -> BoltCommitmentTrace {
    assert_eq!(program.role, Role::Verifier);
    let mut commitments = Vec::new();
    let mut records = Vec::new();
    let mut cursor = 0;

    for plan in &program.batch_plans {
        assert_eq!(plan.count, plan.oracles.len());
        for _ in &plan.oracles {
            let commitment = proof_commitments
                .get(cursor)
                .expect("proof commitment slot")
                .clone();
            assert!(commitment.is_some(), "batch commitments cannot be skipped");
            cursor += 1;
            records.push(CommitmentRecord {
                artifact: plan.artifact.clone(),
            });
            commitments.push(commitment);
        }
    }
    for plan in &program.optional_plans {
        let commitment = proof_commitments
            .get(cursor)
            .expect("optional proof commitment slot")
            .clone();
        cursor += 1;
        records.push(CommitmentRecord {
            artifact: plan.artifact.clone(),
        });
        commitments.push(commitment);
    }
    assert_eq!(cursor, proof_commitments.len());

    BoltCommitmentTrace {
        commitments,
        records,
    }
}

fn append_bolt_preamble<T>(
    transcript: &mut T,
    program_io: &JoltDevice,
    ram_k: usize,
    trace_length: usize,
    entry_address: u64,
) where
    T: Transcript<Challenge = Fr>,
{
    append_u64(
        transcript,
        b"max_input_size",
        program_io.memory_layout.max_input_size,
    );
    append_u64(
        transcript,
        b"max_output_size",
        program_io.memory_layout.max_output_size,
    );
    append_u64(transcript, b"heap_size", program_io.memory_layout.heap_size);
    append_bytes(transcript, b"inputs", &program_io.inputs);
    append_bytes(transcript, b"outputs", &program_io.outputs);
    append_u64(transcript, b"panic", program_io.panic as u64);
    append_u64(transcript, b"ram_K", ram_k as u64);
    append_u64(transcript, b"trace_length", trace_length as u64);
    append_u64(transcript, b"entry_address", entry_address);
}

fn append_u64<T>(transcript: &mut T, label: &'static [u8], value: u64)
where
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&Label(label));
    transcript.append(&U64Word(value));
}

fn append_bytes<T>(transcript: &mut T, label: &'static [u8], bytes: &[u8])
where
    T: Transcript<Challenge = Fr>,
{
    transcript.append(&LabelWithCount(label, bytes.len() as u64));
    transcript.append_bytes(bytes);
}

fn append_bolt_commitments_to_transcript<T>(
    transcript: &mut T,
    records: &[CommitmentRecord],
    commitments: &[Option<DoryCommitment>],
    transcript_steps: &[TranscriptStep],
) where
    T: Transcript<Challenge = Fr>,
{
    for step in transcript_steps {
        let mut appended = false;
        for (record, commitment) in records.iter().zip(commitments) {
            if record.artifact != step.source {
                continue;
            }
            if let Some(commitment) = commitment {
                transcript.append(&LabelWithCount(
                    static_transcript_label(&step.label),
                    commitment.serialized_len(),
                ));
                commitment.append_to_transcript(transcript);
                appended = true;
            }
        }
        assert!(step.optional || appended, "missing transcript source");
    }
}

fn assert_core_accepts_bolt_stage1(
    fixture: &CoreStage1Fixture,
    artifacts: &Stage1ExecutionArtifacts<Fr>,
) {
    let mut proof = clone_core_proof(&fixture.proof);
    proof.stage1_uni_skip_first_round_proof = to_core_uniskip_proof(&artifacts.sumchecks[0]);
    proof.stage1_sumcheck_proof =
        to_core_sumcheck_proof(&artifacts.sumchecks[1].proof.round_polynomials);

    let mut verifier = CoreVerifier::new(
        fixture.verifier_preprocessing,
        proof,
        fixture.io.clone(),
        None,
        None,
    )
    .expect("construct core verifier");
    verifier.run_preamble();
    let _ = verifier
        .verify_stage1()
        .expect("jolt-core accepts Bolt Stage 1 proof");
}

fn assert_core_accepts_bolt_stage2(
    fixture: &CoreStage1Fixture,
    stage1_artifacts: &Stage1ExecutionArtifacts<Fr>,
    stage2_artifacts: &Stage2ExecutionArtifacts<Fr>,
) {
    let mut proof = clone_core_proof(&fixture.proof);
    proof.stage1_uni_skip_first_round_proof = to_core_uniskip_proof(&stage1_artifacts.sumchecks[0]);
    proof.stage1_sumcheck_proof =
        to_core_sumcheck_proof(&stage1_artifacts.sumchecks[1].proof.round_polynomials);
    proof.stage2_uni_skip_first_round_proof =
        to_core_stage2_uniskip_proof(&stage2_artifacts.sumchecks[0]);
    proof.stage2_sumcheck_proof =
        to_core_sumcheck_proof(&stage2_artifacts.sumchecks[1].proof.round_polynomials);

    let mut verifier = CoreVerifier::new(
        fixture.verifier_preprocessing,
        proof,
        fixture.io.clone(),
        None,
        None,
    )
    .expect("construct core verifier");
    verifier.run_preamble();
    let _ = verifier.verify_stage1().expect("core accepts Bolt Stage 1");
    let _ = verifier.verify_stage2().expect("core accepts Bolt Stage 2");
}

fn assert_core_states_match_bolt_stage1(fixture: &CoreStage1Fixture, bolt_log: &[TranscriptEvent]) {
    let mut verifier = CoreVerifier::new(
        fixture.verifier_preprocessing,
        clone_core_proof(&fixture.proof),
        fixture.io.clone(),
        None,
        None,
    )
    .expect("construct core verifier");
    verifier.run_preamble();
    let _ = verifier.verify_stage1().expect("core Stage 1 verifies");

    let core_states = verifier.transcript.state_history[1..].to_vec();
    let bolt_states = transcript_states(bolt_log);
    assert_eq!(core_states, bolt_states);
}

fn assert_core_states_match_bolt_stage2(fixture: &CoreStage1Fixture, bolt_log: &[TranscriptEvent]) {
    let mut verifier = CoreVerifier::new(
        fixture.verifier_preprocessing,
        clone_core_proof(&fixture.proof),
        fixture.io.clone(),
        None,
        None,
    )
    .expect("construct core verifier");
    verifier.run_preamble();
    let _ = verifier.verify_stage1().expect("core Stage 1 verifies");
    let _ = verifier.verify_stage2().expect("core Stage 2 verifies");

    let core_states = verifier.transcript.state_history[1..].to_vec();
    let bolt_states = transcript_states(bolt_log);
    assert_eq!(core_states, bolt_states);
}

fn to_core_uniskip_proof(
    output: &jolt_kernels::stage1::Stage1SumcheckOutput<Fr>,
) -> UniSkipFirstRoundProofVariant<CoreFr, Bn254Curve, CoreBlake2bTranscript> {
    assert_eq!(output.proof.round_polynomials.len(), 1);
    let coefficients = output.proof.round_polynomials[0]
        .coefficients()
        .iter()
        .copied()
        .map(to_ark)
        .collect();
    UniSkipFirstRoundProofVariant::Standard(UniSkipFirstRoundProof::new(UniPoly::from_coeff(
        coefficients,
    )))
}

fn to_core_stage2_uniskip_proof(
    output: &jolt_kernels::stage2::Stage2SumcheckOutput<Fr>,
) -> UniSkipFirstRoundProofVariant<CoreFr, Bn254Curve, CoreBlake2bTranscript> {
    assert_eq!(output.proof.round_polynomials.len(), 1);
    let coefficients = output.proof.round_polynomials[0]
        .coefficients()
        .iter()
        .copied()
        .map(to_ark)
        .collect();
    UniSkipFirstRoundProofVariant::Standard(UniSkipFirstRoundProof::new(UniPoly::from_coeff(
        coefficients,
    )))
}

fn to_core_sumcheck_proof(
    round_polys: &[jolt_poly::UnivariatePoly<Fr>],
) -> jolt_core::subprotocols::sumcheck::SumcheckInstanceProof<
    CoreFr,
    Bn254Curve,
    CoreBlake2bTranscript,
> {
    let compressed = round_polys
        .iter()
        .map(|poly| {
            let coeffs = poly.coefficients();
            let mut out = Vec::with_capacity(coeffs.len().saturating_sub(1));
            out.push(to_ark(coeffs[0]));
            for coeff in &coeffs[2..] {
                out.push(to_ark(*coeff));
            }
            jolt_core::poly::unipoly::CompressedUniPoly {
                coeffs_except_linear_term: out,
            }
        })
        .collect();
    jolt_core::subprotocols::sumcheck::SumcheckInstanceProof::Clear(
        jolt_core::subprotocols::sumcheck::ClearSumcheckProof::new(compressed),
    )
}

fn clone_core_proof(proof: &CoreProof) -> CoreProof {
    CoreJoltProof {
        commitments: proof.commitments.clone(),
        stage1_uni_skip_first_round_proof: proof.stage1_uni_skip_first_round_proof.clone(),
        stage1_sumcheck_proof: proof.stage1_sumcheck_proof.clone(),
        stage2_uni_skip_first_round_proof: proof.stage2_uni_skip_first_round_proof.clone(),
        stage2_sumcheck_proof: proof.stage2_sumcheck_proof.clone(),
        stage3_sumcheck_proof: proof.stage3_sumcheck_proof.clone(),
        stage4_sumcheck_proof: proof.stage4_sumcheck_proof.clone(),
        stage5_sumcheck_proof: proof.stage5_sumcheck_proof.clone(),
        stage6_sumcheck_proof: proof.stage6_sumcheck_proof.clone(),
        stage7_sumcheck_proof: proof.stage7_sumcheck_proof.clone(),
        joint_opening_proof: proof.joint_opening_proof.clone(),
        untrusted_advice_commitment: proof.untrusted_advice_commitment,
        opening_claims: CoreClaims(proof.opening_claims.0.clone()),
        trace_length: proof.trace_length,
        ram_K: proof.ram_K,
        rw_config: proof.rw_config.clone(),
        one_hot_config: proof.one_hot_config.clone(),
        dory_layout: proof.dory_layout,
    }
}

fn transcript_states(log: &[TranscriptEvent]) -> Vec<[u8; 32]> {
    log.iter()
        .map(|event| match event {
            TranscriptEvent::Append { state_after, .. }
            | TranscriptEvent::Squeeze { state_after } => *state_after,
        })
        .collect()
}

fn to_ark(value: Fr) -> CoreFr {
    value.into()
}

fn commitment_to_ark(commitment: &DoryCommitment) -> CoreCommitment {
    // SAFETY: both commitment types wrap the same arkworks G1 type; this oracle only
    // compares transcript-equivalent commitments across the modular and core crates.
    unsafe { std::mem::transmute_copy(&commitment.0) }
}

fn dense_source(cycle_inputs: &[CycleInput], source: &str) -> Vec<i128> {
    let slot = match source {
        "trace.rd_inc" => 0,
        "trace.ram_inc" => 1,
        _ => panic!("unsupported dense source `{source}`"),
    };
    cycle_inputs.iter().map(|cycle| cycle.dense[slot]).collect()
}

fn one_hot_source(cycle_inputs: &[CycleInput], source: &str) -> Vec<Option<u128>> {
    let slot = match source {
        "trace.instruction_keys" => 0,
        "trace.bytecode_indices" => 1,
        "trace.ram_addresses" => 2,
        _ => panic!("unsupported one-hot source `{source}`"),
    };
    cycle_inputs
        .iter()
        .map(|cycle| cycle.one_hot[slot])
        .collect()
}

fn padding_value(padding: &str) -> Option<u128> {
    match padding {
        "zero" => Some(0),
        "none" => None,
        _ => panic!("unsupported padding `{padding}`"),
    }
}

fn should_skip_optional(policy: &OptionalSkipPolicy, data: &[Fr]) -> bool {
    match policy {
        OptionalSkipPolicy::MissingOrZero => data.iter().all(|value| *value == Fr::from_u64(0)),
    }
}

fn oracle_num_vars(program: &CommitmentCpuProgram, oracle: &str, fallback: usize) -> usize {
    program
        .oracle_plans
        .iter()
        .find(|plan| plan.oracle == oracle)
        .map_or(fallback, |plan| plan.num_vars)
}

fn into_padded_oracle(mut data: Vec<Fr>, num_vars: usize) -> Vec<Fr> {
    let target_len = target_len(num_vars);
    assert!(
        data.len() <= target_len,
        "oracle has {} values, target length is {target_len}",
        data.len()
    );
    data.resize(target_len, Fr::from_u64(0));
    data
}

fn commit_with_layout(
    data: &[Fr],
    layout_num_vars: usize,
    setup: &DoryProverSetup,
) -> (DoryCommitment, DoryHint) {
    let row_len = target_len(layout_num_vars.div_ceil(2));
    DoryScheme::commit_evaluations_with_row_len(data, row_len, setup)
}

fn target_len(num_vars: usize) -> usize {
    1usize << num_vars
}

fn static_transcript_label(label: &str) -> &'static [u8] {
    match label {
        "commitment" => b"commitment",
        "untrusted_advice" => b"untrusted_advice",
        "trusted_advice" => b"trusted_advice",
        _ => panic!("unsupported transcript label `{label}`"),
    }
}

fn leak_stage1_program(program: &CompilerStage1CpuProgram) -> &'static KernelStage1CpuProgramPlan {
    let transcript_squeezes = leak_slice(
        program
            .transcript_squeezes
            .iter()
            .map(|plan| Stage1TranscriptSqueezePlan {
                symbol: leak_str(&plan.symbol),
                label: leak_str(&plan.label),
                kind: leak_str(&plan.kind),
                count: plan.count,
            })
            .collect(),
    );
    let kernels = if program.kernels.is_empty() {
        leak_slice(synthetic_stage1_kernels(program))
    } else {
        leak_slice(
            program
                .kernels
                .iter()
                .map(|plan| KernelStage1KernelPlan {
                    symbol: leak_str(&plan.symbol),
                    relation: leak_str(&plan.relation),
                    kind: leak_str(&plan.kind),
                    backend: leak_str(&plan.backend),
                    abi: leak_str(&plan.abi),
                })
                .collect(),
        )
    };
    let claims = leak_slice(
        program
            .claims
            .iter()
            .map(|plan| KernelStage1SumcheckClaimPlan {
                symbol: leak_str(&plan.symbol),
                stage: leak_str(&plan.stage),
                domain: leak_str(&plan.domain),
                num_rounds: plan.num_rounds,
                degree: plan.degree,
                claim: leak_str(&plan.claim),
                kernel: leak_str(stage1_kernel_symbol(
                    plan.kernel.as_deref(),
                    plan.relation.as_deref(),
                )),
                claim_value: leak_str(&plan.claim_value),
                input_openings: leak_str_slice(&plan.input_openings),
            })
            .collect(),
    );
    let batches = leak_slice(
        program
            .batches
            .iter()
            .map(|plan| KernelStage1SumcheckBatchPlan {
                symbol: leak_str(&plan.symbol),
                stage: leak_str(&plan.stage),
                proof_slot: leak_str(&plan.proof_slot),
                policy: leak_str(&plan.policy),
                count: plan.count,
                ordered_claims: leak_str_slice(&plan.ordered_claims),
                claim_operands: leak_str_slice(&plan.claim_operands),
                claim_label: leak_str(&plan.claim_label),
                round_label: leak_str(&plan.round_label),
                round_schedule: leak_usize_slice(&plan.round_schedule),
            })
            .collect(),
    );
    let drivers = leak_slice(
        program
            .drivers
            .iter()
            .map(|plan| KernelStage1SumcheckDriverPlan {
                symbol: leak_str(&plan.symbol),
                stage: leak_str(&plan.stage),
                proof_slot: leak_str(&plan.proof_slot),
                kernel: leak_str(stage1_kernel_symbol(
                    plan.kernel.as_deref(),
                    plan.relation.as_deref(),
                )),
                batch: leak_str(&plan.batch),
                policy: leak_str(&plan.policy),
                round_schedule: leak_usize_slice(&plan.round_schedule),
                claim_label: leak_str(&plan.claim_label),
                round_label: leak_str(&plan.round_label),
                num_rounds: plan.num_rounds,
                degree: plan.degree,
            })
            .collect(),
    );
    let instance_results = leak_slice(
        program
            .instance_results
            .iter()
            .map(|plan| KernelStage1SumcheckInstanceResultPlan {
                symbol: leak_str(&plan.symbol),
                source: leak_str(&plan.source),
                claim: leak_str(&plan.claim),
                relation: leak_str(&plan.relation),
                index: plan.index,
                point_arity: plan.point_arity,
                num_rounds: plan.num_rounds,
                round_offset: plan.round_offset,
                point_order: leak_str(&plan.point_order),
                degree: plan.degree,
            })
            .collect(),
    );
    let evals = leak_slice(
        program
            .evals
            .iter()
            .map(|plan| KernelStage1SumcheckEvalPlan {
                symbol: leak_str(&plan.symbol),
                source: leak_str(&plan.source),
                name: leak_str(&plan.name),
                index: plan.index,
                oracle: leak_str(&plan.oracle),
            })
            .collect(),
    );
    let opening_claims = leak_slice(
        program
            .opening_claims
            .iter()
            .map(|plan| KernelStage1OpeningClaimPlan {
                symbol: leak_str(&plan.symbol),
                oracle: leak_str(&plan.oracle),
                domain: leak_str(&plan.domain),
                point_arity: plan.point_arity,
                claim_kind: leak_str(&plan.claim_kind),
                point_source: leak_str(&plan.point_source),
                eval_source: leak_str(&plan.eval_source),
            })
            .collect(),
    );
    let opening_batches = leak_slice(
        program
            .opening_batches
            .iter()
            .map(|plan| KernelStage1OpeningBatchPlan {
                symbol: leak_str(&plan.symbol),
                stage: leak_str(&plan.stage),
                proof_slot: leak_str(&plan.proof_slot),
                policy: leak_str(&plan.policy),
                count: plan.count,
                ordered_claims: leak_str_slice(&plan.ordered_claims),
                claim_operands: leak_str_slice(&plan.claim_operands),
            })
            .collect(),
    );

    Box::leak(Box::new(KernelStage1CpuProgramPlan {
        params: Stage1Params {
            field: leak_str(&program.params.field),
            pcs: leak_str(&program.params.pcs),
            transcript: leak_str(&program.params.transcript),
        },
        transcript_squeezes,
        kernels,
        claims,
        batches,
        drivers,
        instance_results,
        evals,
        opening_claims,
        opening_batches,
    }))
}

fn synthetic_stage1_kernels(program: &CompilerStage1CpuProgram) -> Vec<KernelStage1KernelPlan> {
    let mut kernels: Vec<KernelStage1KernelPlan> = Vec::new();
    for driver in &program.drivers {
        let relation = driver
            .relation
            .as_deref()
            .expect("verifier driver relation");
        let kernel = synthetic_stage1_kernel(relation);
        if !kernels
            .iter()
            .any(|existing| existing.symbol == kernel.symbol)
        {
            kernels.push(kernel);
        }
    }
    kernels
}

fn stage1_kernel_symbol<'a>(kernel: Option<&'a str>, relation: Option<&str>) -> &'a str {
    if let Some(kernel) = kernel {
        return kernel;
    }
    synthetic_stage1_kernel(relation.expect("verifier relation")).symbol
}

fn synthetic_stage1_kernel(relation: &str) -> KernelStage1KernelPlan {
    match relation {
        "jolt.stage1.outer.uniskip" => KernelStage1KernelPlan {
            symbol: "jolt.cpu.stage1.outer.uniskip",
            relation: "jolt.stage1.outer.uniskip",
            kind: "sumcheck",
            backend: "cpu",
            abi: "jolt_stage1_outer_uniskip",
        },
        "jolt.stage1.outer.remaining" => KernelStage1KernelPlan {
            symbol: "jolt.cpu.stage1.outer.remaining",
            relation: "jolt.stage1.outer.remaining",
            kind: "sumcheck",
            backend: "cpu",
            abi: "jolt_stage1_outer_remaining",
        },
        relation => panic!("unsupported Stage 1 verifier relation `{relation}`"),
    }
}

fn leak_stage2_program(program: &CompilerStage2CpuProgram) -> &'static KernelStage2CpuProgramPlan {
    Box::leak(Box::new(KernelStage2CpuProgramPlan {
        params: Stage2Params {
            field: leak_str(&program.params.field),
            pcs: leak_str(&program.params.pcs),
            transcript: leak_str(&program.params.transcript),
        },
        steps: leak_slice(
            program
                .steps
                .iter()
                .map(|plan| KernelStage2ProgramStepPlan {
                    kind: leak_str(&plan.kind),
                    symbol: leak_str(&plan.symbol),
                })
                .collect(),
        ),
        transcript_squeezes: leak_slice(
            program
                .transcript_squeezes
                .iter()
                .map(|plan| KernelStage2TranscriptSqueezePlan {
                    symbol: leak_str(&plan.symbol),
                    label: leak_str(&plan.label),
                    kind: leak_str(&plan.kind),
                    count: plan.count,
                })
                .collect(),
        ),
        opening_inputs: leak_slice(
            program
                .opening_inputs
                .iter()
                .map(|plan| KernelStage2OpeningInputPlan {
                    symbol: leak_str(&plan.symbol),
                    source_stage: leak_str(&plan.source_stage),
                    source_claim: leak_str(&plan.source_claim),
                    oracle: leak_str(&plan.oracle),
                    domain: leak_str(&plan.domain),
                    point_arity: plan.point_arity,
                    claim_kind: leak_str(&plan.claim_kind),
                })
                .collect(),
        ),
        field_constants: leak_slice(
            program
                .field_constants
                .iter()
                .map(|plan| KernelStage2FieldConstantPlan {
                    symbol: leak_str(&plan.symbol),
                    field: leak_str(&plan.field),
                    value: plan.value,
                })
                .collect(),
        ),
        challenge_extracts: leak_slice(
            program
                .challenge_extracts
                .iter()
                .map(|plan| KernelStage2ChallengeExtractPlan {
                    symbol: leak_str(&plan.symbol),
                    source: leak_str(&plan.source),
                    index: plan.index,
                    challenge_source: leak_str(&plan.challenge_source),
                })
                .collect(),
        ),
        field_exprs: leak_slice(
            program
                .field_exprs
                .iter()
                .map(|plan| KernelStage2FieldExprPlan {
                    symbol: leak_str(&plan.symbol),
                    kind: leak_str(&plan.kind),
                    formula: leak_str(&plan.formula),
                    operand_names: leak_str_slice(&plan.operand_names),
                    operands: leak_str_slice(&plan.operands),
                })
                .collect(),
        ),
        kernels: leak_slice(
            program
                .kernels
                .iter()
                .map(|plan| KernelStage2KernelPlan {
                    symbol: leak_str(&plan.symbol),
                    relation: leak_str(&plan.relation),
                    kind: leak_str(&plan.kind),
                    backend: leak_str(&plan.backend),
                    abi: leak_str(&plan.abi),
                })
                .collect(),
        ),
        claims: leak_slice(
            program
                .claims
                .iter()
                .map(|plan| KernelStage2SumcheckClaimPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    domain: leak_str(&plan.domain),
                    num_rounds: plan.num_rounds,
                    degree: plan.degree,
                    claim: leak_str(&plan.claim),
                    kernel: leak_str(plan.kernel.as_deref().expect("Stage 2 claim kernel")),
                    claim_value: leak_str(&plan.claim_value),
                    input_openings: leak_str_slice(&plan.input_openings),
                })
                .collect(),
        ),
        batches: leak_slice(
            program
                .batches
                .iter()
                .map(|plan| KernelStage2SumcheckBatchPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    proof_slot: leak_str(&plan.proof_slot),
                    policy: leak_str(&plan.policy),
                    count: plan.count,
                    ordered_claims: leak_str_slice(&plan.ordered_claims),
                    claim_operands: leak_str_slice(&plan.claim_operands),
                    claim_label: leak_str(&plan.claim_label),
                    round_label: leak_str(&plan.round_label),
                    round_schedule: leak_usize_slice(&plan.round_schedule),
                })
                .collect(),
        ),
        drivers: leak_slice(
            program
                .drivers
                .iter()
                .map(|plan| KernelStage2SumcheckDriverPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    proof_slot: leak_str(&plan.proof_slot),
                    kernel: leak_str(plan.kernel.as_deref().expect("Stage 2 driver kernel")),
                    batch: leak_str(&plan.batch),
                    policy: leak_str(&plan.policy),
                    round_schedule: leak_usize_slice(&plan.round_schedule),
                    claim_label: leak_str(&plan.claim_label),
                    round_label: leak_str(&plan.round_label),
                    num_rounds: plan.num_rounds,
                    degree: plan.degree,
                })
                .collect(),
        ),
        instance_results: leak_slice(
            program
                .instance_results
                .iter()
                .map(|plan| KernelStage2SumcheckInstanceResultPlan {
                    symbol: leak_str(&plan.symbol),
                    source: leak_str(&plan.source),
                    claim: leak_str(&plan.claim),
                    relation: leak_str(&plan.relation),
                    index: plan.index,
                    point_arity: plan.point_arity,
                    num_rounds: plan.num_rounds,
                    round_offset: plan.round_offset,
                    point_order: leak_str(&plan.point_order),
                    degree: plan.degree,
                })
                .collect(),
        ),
        evals: leak_slice(
            program
                .evals
                .iter()
                .map(|plan| KernelStage2SumcheckEvalPlan {
                    symbol: leak_str(&plan.symbol),
                    source: leak_str(&plan.source),
                    name: leak_str(&plan.name),
                    index: plan.index,
                    oracle: leak_str(&plan.oracle),
                })
                .collect(),
        ),
        point_slices: leak_slice(
            program
                .point_slices
                .iter()
                .map(|plan| KernelStage2PointSlicePlan {
                    symbol: leak_str(&plan.symbol),
                    source: leak_str(&plan.source),
                    offset: plan.offset,
                    length: plan.length,
                    input: leak_str(&plan.input),
                })
                .collect(),
        ),
        point_concats: leak_slice(
            program
                .point_concats
                .iter()
                .map(|plan| KernelStage2PointConcatPlan {
                    symbol: leak_str(&plan.symbol),
                    layout: leak_str(&plan.layout),
                    arity: plan.arity,
                    inputs: leak_str_slice(&plan.inputs),
                })
                .collect(),
        ),
        opening_claims: leak_slice(
            program
                .opening_claims
                .iter()
                .map(|plan| KernelStage2OpeningClaimPlan {
                    symbol: leak_str(&plan.symbol),
                    oracle: leak_str(&plan.oracle),
                    domain: leak_str(&plan.domain),
                    point_arity: plan.point_arity,
                    claim_kind: leak_str(&plan.claim_kind),
                    point_source: leak_str(&plan.point_source),
                    eval_source: leak_str(&plan.eval_source),
                })
                .collect(),
        ),
        opening_batches: leak_slice(
            program
                .opening_batches
                .iter()
                .map(|plan| KernelStage2OpeningBatchPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    proof_slot: leak_str(&plan.proof_slot),
                    policy: leak_str(&plan.policy),
                    count: plan.count,
                    ordered_claims: leak_str_slice(&plan.ordered_claims),
                    claim_operands: leak_str_slice(&plan.claim_operands),
                })
                .collect(),
        ),
    }))
}

fn leak_str(value: &str) -> &'static str {
    Box::leak(value.to_owned().into_boxed_str())
}

fn leak_str_slice(values: &[String]) -> &'static [&'static str] {
    let leaked = values
        .iter()
        .map(|value| leak_str(value))
        .collect::<Vec<_>>();
    Box::leak(leaked.into_boxed_slice())
}

fn leak_usize_slice(values: &[usize]) -> &'static [usize] {
    Box::leak(values.to_vec().into_boxed_slice())
}

fn leak_slice<T>(values: Vec<T>) -> &'static [T] {
    Box::leak(values.into_boxed_slice())
}
