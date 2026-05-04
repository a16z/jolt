use std::borrow::Cow;
use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::PathBuf;
use std::sync::mpsc;
use std::time::{Duration, Instant};

use bolt::{
    build_commitment_protocol, build_stage1_outer_protocol, build_stage2_protocol,
    build_stage3_protocol, build_stage4_protocol, build_stage5_protocol, build_stage6_protocol,
    build_stage7_protocol, build_stage8_protocol, commitment_cpu_program,
    lower_commitment_to_compute, lower_compute_to_cpu, lower_piop_and_fiat_shamir,
    lower_stage1_to_compute, lower_stage2_to_compute, lower_stage3_to_compute,
    lower_stage4_to_compute, lower_stage5_to_compute, lower_stage6_to_compute,
    lower_stage7_to_compute, lower_stage8_to_compute, project_prover_party, project_verifier_party,
    resolve_compute_kernels, stage1_cpu_program, stage2_cpu_program, stage3_cpu_program,
    stage4_cpu_program, stage5_cpu_program, stage6_cpu_program, stage7_cpu_program,
    stage8_cpu_program, CommitmentCpuProgram, JoltProtocolParams, MeliorContext,
    OptionalSkipPolicy, OracleGeneration, OraclePlan, Role, TranscriptStep,
};
use bolt::{
    Stage1CpuProgram as CompilerStage1CpuProgram, Stage2CpuProgram as CompilerStage2CpuProgram,
    Stage3CpuProgram as CompilerStage3CpuProgram, Stage4CpuProgram as CompilerStage4CpuProgram,
    Stage5CpuProgram as CompilerStage5CpuProgram, Stage6CpuProgram as CompilerStage6CpuProgram,
    Stage7CpuProgram as CompilerStage7CpuProgram, Stage8CpuProgram as CompilerStage8CpuProgram,
};
use clap::Parser;
use common::constants::{RAM_START_ADDRESS, XLEN};
use common::jolt_device::JoltDevice;
use jolt_bench::measure::{median, time_it};
use jolt_bench::programs::Program;
use jolt_core::curve::Bn254Curve;
use jolt_core::host;
use jolt_core::poly::commitment::commitment_scheme::CommitmentScheme as CoreCommitmentScheme;
use jolt_core::poly::commitment::dory::{DoryCommitmentScheme, DoryGlobals};
use jolt_core::poly::opening_proof::{OpeningAccumulator, SumcheckId};
use jolt_core::poly::unipoly::UniPoly;
use jolt_core::subprotocols::univariate_skip::{
    UniSkipFirstRoundProof, UniSkipFirstRoundProofVariant,
};
use jolt_core::transcripts::Blake2bTranscript as CoreBlake2bTranscript;
use jolt_core::zkvm::instruction::{
    CircuitFlags as CoreCircuitFlags, Flags as _, InstructionFlags as CoreInstructionFlags,
    InstructionLookup, InterleavedBitsMarker, LookupQuery,
};
use jolt_core::zkvm::lookup_table::LookupTables as CoreLookupTables;
use jolt_core::zkvm::proof_serialization::{Claims as CoreClaims, JoltProof as CoreJoltProof};
use jolt_core::zkvm::prover::{JoltCpuProver, JoltProverPreprocessing};
use jolt_core::zkvm::r1cs::inputs::ProductCycleInputs as CoreProductCycleInputs;
use jolt_core::zkvm::ram::remap_address;
use jolt_core::zkvm::verifier::{JoltSharedPreprocessing, JoltVerifier, JoltVerifierPreprocessing};
use jolt_core::zkvm::witness::{CommittedPolynomial, VirtualPolynomial};
use jolt_dory::{DoryCommitment, DoryHint, DoryProverSetup, DoryScheme};
use jolt_field::signed::{S128, S64};
use jolt_field::{Field, Fr};
use jolt_host::{
    extract_trace, BytecodePreprocessing, CircuitFlagSet, CircuitFlags, CycleRow, InstructionFlags,
    NUM_CIRCUIT_FLAGS,
};
use jolt_inlines_keccak256 as _;
use jolt_inlines_sha2 as _;
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
use jolt_kernels::stage3::{
    execute_stage3_program, Stage3CpuProgramPlan as KernelStage3CpuProgramPlan, Stage3Cycle,
    Stage3ExecutionArtifacts, Stage3ExecutionMode,
    Stage3FieldConstantPlan as KernelStage3FieldConstantPlan,
    Stage3FieldExprPlan as KernelStage3FieldExprPlan, Stage3KernelPlan as KernelStage3KernelPlan,
    Stage3OpeningBatchPlan as KernelStage3OpeningBatchPlan,
    Stage3OpeningClaimEqualityPlan as KernelStage3OpeningClaimEqualityPlan,
    Stage3OpeningClaimPlan as KernelStage3OpeningClaimPlan, Stage3OpeningInputPlan,
    Stage3OpeningInputValue, Stage3Params, Stage3PointConcatPlan, Stage3PointSlicePlan,
    Stage3ProgramStepPlan as KernelStage3ProgramStepPlan, Stage3Proof, Stage3ProverInputs,
    Stage3ProverKernelExecutor, Stage3SumcheckBatchPlan as KernelStage3SumcheckBatchPlan,
    Stage3SumcheckClaimPlan as KernelStage3SumcheckClaimPlan,
    Stage3SumcheckDriverPlan as KernelStage3SumcheckDriverPlan,
    Stage3SumcheckEvalPlan as KernelStage3SumcheckEvalPlan, Stage3SumcheckInstanceResultPlan,
    Stage3TranscriptSqueezePlan as KernelStage3TranscriptSqueezePlan, Stage3VerifierKernelExecutor,
};
use jolt_kernels::stage4::{
    execute_stage4_program, Stage4CpuProgramPlan as KernelStage4CpuProgramPlan,
    Stage4ExecutionArtifacts, Stage4ExecutionMode,
    Stage4FieldConstantPlan as KernelStage4FieldConstantPlan,
    Stage4FieldExprPlan as KernelStage4FieldExprPlan, Stage4KernelPlan as KernelStage4KernelPlan,
    Stage4OpeningBatchPlan as KernelStage4OpeningBatchPlan,
    Stage4OpeningClaimEqualityPlan as KernelStage4OpeningClaimEqualityPlan,
    Stage4OpeningClaimPlan as KernelStage4OpeningClaimPlan, Stage4OpeningInputPlan,
    Stage4OpeningInputValue, Stage4Params, Stage4PointConcatPlan, Stage4PointSlicePlan,
    Stage4ProgramStepPlan as KernelStage4ProgramStepPlan, Stage4Proof, Stage4ProverInputs,
    Stage4ProverKernelExecutor, Stage4RamWitness, Stage4RegisterAccess, Stage4RegisterRead,
    Stage4RegisterWrite, Stage4RegistersWitness,
    Stage4SumcheckBatchPlan as KernelStage4SumcheckBatchPlan,
    Stage4SumcheckClaimPlan as KernelStage4SumcheckClaimPlan,
    Stage4SumcheckDriverPlan as KernelStage4SumcheckDriverPlan,
    Stage4SumcheckEvalPlan as KernelStage4SumcheckEvalPlan, Stage4SumcheckInstanceResultPlan,
    Stage4TranscriptAbsorbBytesPlan as KernelStage4TranscriptAbsorbBytesPlan,
    Stage4TranscriptSqueezePlan as KernelStage4TranscriptSqueezePlan, Stage4VerifierKernelExecutor,
};
use jolt_kernels::stage5 as kernel_stage5;
use jolt_kernels::stage6 as kernel_stage6;
use jolt_kernels::stage7 as kernel_stage7;
use jolt_openings::CommitmentScheme as _;
use jolt_poly::EqPolynomial;
use jolt_prover::stages::stage8 as generated_prover_stage8;
use jolt_r1cs::{constraints::rv64, R1csKey};
use jolt_transcript::{
    AppendToTranscript, Blake2bTranscript, Label, LabelWithCount, Transcript, U64Word,
};
use jolt_verifier::stages::stage8 as generated_stage8;
use jolt_verifier::TRANSCRIPT_LABEL;
use jolt_witness::CycleInput;
use jolt_witness::{
    dense_i128_column_to_field, one_hot_chunk_address_major, optional_field_oracle,
};
use serde::Serialize;
use strum::EnumCount;
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

    #[arg(long, default_value_t = 1.2)]
    max_ratio: f64,

    #[arg(long)]
    json: Option<PathBuf>,

    #[arg(long, value_name = "NAME")]
    trace_chrome: Option<String>,

    #[arg(long)]
    e2e: bool,

    #[arg(long, value_enum, default_value_t = E2eStack::Both)]
    e2e_stack: E2eStack,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum BenchStage {
    Stage1,
    Stage2,
    Stage3,
    Stage4,
    Stage5,
    Stage6,
    Stage7,
    Stage8,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, clap::ValueEnum)]
enum E2eStack {
    Both,
    Core,
    Bolt,
}

impl BenchStage {
    fn name(self) -> &'static str {
        match self {
            Self::Stage1 => "stage1",
            Self::Stage2 => "stage2",
            Self::Stage3 => "stage3",
            Self::Stage4 => "stage4",
            Self::Stage5 => "stage5",
            Self::Stage6 => "stage6",
            Self::Stage7 => "stage7",
            Self::Stage8 => "stage8",
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
    hints: Vec<jolt_prover::stages::commitment::OracleOpeningHint>,
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
    stage3_cycles: Vec<Stage3Cycle>,
    stage4_register_accesses: Vec<Stage4RegisterAccess>,
    stage5_lookup_indices: Vec<u128>,
    stage5_lookup_table_indices: Vec<Option<usize>>,
    stage5_is_interleaved_operands: Vec<bool>,
    padded_trace: Vec<jolt_host::Cycle>,
    stage6_bytecode_entries: Vec<kernel_stage6::Stage6BytecodeEntry<Fr>>,
    stage6_entry_bytecode_index: usize,
    ram_accesses: Vec<Stage2RamAccess>,
    initial_ram_state: Vec<u64>,
    final_ram_state: Vec<u64>,
    ram_start_address: u64,
    ram_output_layout: Stage2RamOutputLayout,
    commitments: Vec<CoreCommitment>,
}

#[derive(Clone, Debug)]
struct CoreStage6Data {
    opening_inputs: Vec<kernel_stage6::Stage6OpeningInputValue<Fr>>,
    transcript_states: Vec<[u8; 32]>,
}

#[derive(Clone, Debug)]
struct CoreStage7Data {
    opening_inputs: Vec<kernel_stage7::Stage7OpeningInputValue<Fr>>,
    transcript_states: Vec<[u8; 32]>,
}

#[derive(Clone, Debug)]
struct Stage6WitnessPolynomials {
    instruction_ra_indices: Vec<Vec<Option<u8>>>,
    bytecode_ra_indices: Vec<Vec<Option<u8>>>,
    ram_ra_indices: Vec<Vec<Option<u8>>>,
    instruction_ra_booleanity: Vec<Vec<Fr>>,
    bytecode_ra_booleanity: Vec<Vec<Fr>>,
    ram_ra_booleanity: Vec<Vec<Fr>>,
    bytecode_ra_read_raf: Vec<Vec<Fr>>,
    instruction_ra_virtual: Vec<Vec<Fr>>,
    ram_ra_virtual: Vec<Vec<Fr>>,
    hamming_weight: Vec<Fr>,
    ram_inc: Vec<Fr>,
    rd_inc: Vec<Fr>,
}

struct Stage8Prefix<T: Transcript<Challenge = Fr>> {
    transcript: T,
    commitment_artifacts: jolt_prover::stages::commitment::CommitmentArtifacts,
    stage6_artifacts: kernel_stage6::Stage6ExecutionArtifacts<Fr>,
    stage7_artifacts: kernel_stage7::Stage7ExecutionArtifacts<Fr>,
    stage7_opening_inputs: Vec<kernel_stage7::Stage7OpeningInputValue<Fr>>,
    oracle_data: BTreeMap<String, Option<Vec<Fr>>>,
}

struct BenchCommitmentInputs<'a> {
    data: &'a BTreeMap<String, Option<Vec<Fr>>>,
}

impl jolt_prover::stages::commitment::CommitmentInputProvider for BenchCommitmentInputs<'_> {
    fn materialize(&mut self, oracle: &'static str) -> Option<Cow<'_, [Fr]>> {
        self.data
            .get(oracle)
            .and_then(|values| values.as_ref())
            .map(|values| Cow::Borrowed(values.as_slice()))
    }
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
    max_ratio: f64,
    ratio_gate_passed: bool,
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

#[derive(Serialize)]
struct E2eBenchReport {
    program: String,
    max_log_t: usize,
    actual_log_t: usize,
    trace_length: usize,
    num_iters: u32,
    ratio_vs_core_prove: Option<f64>,
    core: Option<E2eRun>,
    bolt: Option<E2eRun>,
    setup: Option<E2eSetupReport>,
}

#[derive(Serialize)]
struct E2eRun {
    stack: &'static str,
    total_ms: f64,
    prove_ms: f64,
    verify_ms: f64,
    trace_length: usize,
}

#[derive(Serialize)]
struct E2eSetupReport {
    fixture_ms: f64,
    program_context_ms: f64,
    reference_openings_ms: f64,
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
    assert!(cli.max_ratio > 0.0, "--max-ratio must be positive");

    if cli.e2e {
        let report = run_e2e_trace(&cli);
        let json = serde_json::to_string_pretty(&report).expect("serialize E2E bench report");
        if let Some(path) = cli.json {
            std::fs::write(&path, json).expect("write E2E bench JSON");
        } else {
            println!("{json}");
        }
        return;
    }

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
    let ratio_gate_passed = bolt_run
        .ratio_vs_core
        .is_some_and(|ratio| ratio <= cli.max_ratio);

    let report = Stage1BenchReport {
        program: cli.program.cli_name().to_string(),
        stage: cli.stage.name(),
        max_log_t: cli.log_t,
        actual_log_t,
        trace_length,
        num_iters: cli.num_iters,
        iters: cli.iters,
        warmup: cli.warmup,
        max_ratio: cli.max_ratio,
        ratio_gate_passed,
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
    if !ratio_gate_passed {
        eprintln!(
            "Bolt {} exceeded ratio gate: max {:.3}x",
            cli.stage.name(),
            cli.max_ratio
        );
        std::process::exit(3);
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
    stage3_prover_plan: &'static KernelStage3CpuProgramPlan,
    stage4_prover_plan: &'static KernelStage4CpuProgramPlan,
    stage5_prover_plan: &'static kernel_stage5::Stage5CpuProgramPlan,
    stage6_prover_plan: &'static kernel_stage6::Stage6CpuProgramPlan,
    stage7_prover_plan: &'static kernel_stage7::Stage7CpuProgramPlan,
    stage8_prover_plan: &'static generated_prover_stage8::Stage8EvaluationProgramPlan,
    stage8_verifier_plan: &'static generated_stage8::Stage8EvaluationProgramPlan,
    num_cycle_vars: usize,
}

struct BoltE2eContext {
    commitment_prover_plan: &'static jolt_prover::stages::commitment::CommitmentProverProgramPlan,
    stage1_prover_plan: &'static KernelStage1CpuProgramPlan,
    stage2_prover_plan: &'static KernelStage2CpuProgramPlan,
    stage3_prover_plan: &'static KernelStage3CpuProgramPlan,
    stage4_prover_plan: &'static KernelStage4CpuProgramPlan,
    stage5_prover_plan: &'static kernel_stage5::Stage5CpuProgramPlan,
    stage6_prover_plan: &'static kernel_stage6::Stage6CpuProgramPlan,
    stage7_prover_plan: &'static kernel_stage7::Stage7CpuProgramPlan,
    stage8_prover_plan: &'static generated_prover_stage8::Stage8EvaluationProgramPlan,
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
    let core_bytecode_for_bolt = core_bytecode.clone();
    let (_, trace, _, core_io_device) = core_program.trace(&inputs, &[], &[]);
    let shared_preprocessing = JoltSharedPreprocessing::new(
        core_bytecode,
        core_io_device.memory_layout.clone(),
        init_memory_state,
        max_trace_length,
        entry_address,
    )
    .expect("shared preprocessing");
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
                remapped_address: remap_address(read.address, &core_io_device.memory_layout)
                    .map(|address| address as usize),
                read_value: read.value,
                write_value: read.value,
            },
            RAMAccess::Write(write) => Stage2RamAccess {
                remapped_address: remap_address(write.address, &core_io_device.memory_layout)
                    .map(|address| address as usize),
                read_value: write.pre_value,
                write_value: write.post_value,
            },
            RAMAccess::NoOp => Stage2RamAccess::noop(),
        })
        .collect();
    let ram_start_address = core_io_device.memory_layout.get_lowest_address();
    let ram_output_layout = Stage2RamOutputLayout {
        io_start: remap_address(
            core_io_device.memory_layout.input_start,
            &core_io_device.memory_layout,
        )
        .expect("input start remaps") as usize,
        io_end: remap_address(RAM_START_ADDRESS, &core_io_device.memory_layout)
            .expect("RAM start remaps") as usize,
    };
    let bytecode = BytecodePreprocessing::preprocess(core_bytecode_for_bolt, entry_address);
    let stage6_bytecode_entries = stage6_bytecode_entries(&bytecode);
    let stage6_entry_bytecode_index = bytecode.entry_bytecode_index();
    let r1cs_key = R1csKey::new(rv64::rv64_constraints::<Fr>(), proof.trace_length);
    let (cycle_inputs, r1cs_witness, _) = extract_trace::<_, Fr>(
        &trace,
        proof.trace_length,
        &bytecode,
        &core_io_device.memory_layout,
        r1cs_key.num_vars_padded,
    );
    let rv64_cycles = stage1_rv64_cycles(&trace, proof.trace_length, &bytecode);
    let stage3_cycles = stage3_cycles(&trace, proof.trace_length, &bytecode);
    let stage4_register_accesses = stage4_register_accesses(&trace, proof.trace_length);
    let stage5_lookup_indices = padded_trace
        .iter()
        .map(|cycle| LookupQuery::<XLEN>::to_lookup_index(cycle))
        .collect();
    let stage5_lookup_table_indices = padded_trace
        .iter()
        .map(|cycle| {
            InstructionLookup::<XLEN>::lookup_table(cycle)
                .map(|table| CoreLookupTables::<XLEN>::enum_index(&table))
        })
        .collect();
    let stage5_is_interleaved_operands = padded_trace
        .iter()
        .map(|cycle| {
            cycle
                .instruction()
                .circuit_flags()
                .is_interleaved_operands()
        })
        .collect();

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
        stage3_cycles,
        stage4_register_accesses,
        stage5_lookup_indices,
        stage5_lookup_table_indices,
        stage5_is_interleaved_operands,
        padded_trace,
        stage6_bytecode_entries,
        stage6_entry_bytecode_index,
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

fn stage3_cycles<C: CycleRow>(
    trace: &[C],
    size: usize,
    bytecode: &BytecodePreprocessing,
) -> Vec<Stage3Cycle> {
    (0..size)
        .map(|cycle| stage3_cycle(trace.get(cycle).copied(), bytecode))
        .collect()
}

fn stage3_cycle<C: CycleRow>(cycle: Option<C>, bytecode: &BytecodePreprocessing) -> Stage3Cycle {
    let Some(cycle) = cycle else {
        return Stage3Cycle::padding();
    };
    let circuit_flags = cycle.circuit_flags();
    let instruction_flags = cycle.instruction_flags();
    Stage3Cycle {
        unexpanded_pc: cycle.unexpanded_pc(),
        pc: bytecode.get_cycle_pc(&cycle) as u64,
        is_virtual: circuit_flags[CircuitFlags::VirtualInstruction],
        is_first_in_sequence: circuit_flags[CircuitFlags::IsFirstInSequence],
        is_noop: instruction_flags[InstructionFlags::IsNoop],
        left_operand_is_rs1: instruction_flags[InstructionFlags::LeftOperandIsRs1Value],
        rs1_value: cycle.rs1_read().map_or(0, |(_, value)| value),
        left_operand_is_pc: instruction_flags[InstructionFlags::LeftOperandIsPC],
        right_operand_is_rs2: instruction_flags[InstructionFlags::RightOperandIsRs2Value],
        rs2_value: cycle.rs2_read().map_or(0, |(_, value)| value),
        right_operand_is_imm: instruction_flags[InstructionFlags::RightOperandIsImm],
        imm: cycle.imm(),
        rd_write_value: cycle.rd_write().map_or(0, |(_, _, post)| post),
    }
}

fn stage4_register_accesses<C: CycleRow>(trace: &[C], size: usize) -> Vec<Stage4RegisterAccess> {
    (0..size)
        .map(|cycle| stage4_register_access(trace.get(cycle).copied()))
        .collect()
}

fn stage4_register_access<C: CycleRow>(cycle: Option<C>) -> Stage4RegisterAccess {
    let Some(cycle) = cycle else {
        return Stage4RegisterAccess {
            rs1: None,
            rs2: None,
            rd: None,
        };
    };
    Stage4RegisterAccess {
        rs1: cycle.rs1_read().map(|(address, value)| Stage4RegisterRead {
            address: address as usize,
            value,
        }),
        rs2: cycle.rs2_read().map(|(address, value)| Stage4RegisterRead {
            address: address as usize,
            value,
        }),
        rd: cycle
            .rd_write()
            .map(|(address, pre_value, post_value)| Stage4RegisterWrite {
                address: address as usize,
                pre_value,
                post_value,
            }),
    }
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
        lookup_operands_raw(left_input, right_i128, product, flags, lookup_output);
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
        pc: bytecode.get_cycle_pc(cycle) as u64,
        next_pc: 0,
        unexpanded_pc: cycle.unexpanded_pc(),
        next_unexpanded_pc: 0,
        imm: s64_from_i128(cycle.imm()),
        flags: stage1_rv64_flags(flags),
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
        row.next_pc = bytecode.get_cycle_pc(next_cycle) as u64;
        row.next_unexpanded_pc = next_cycle.unexpanded_pc();
        let next_flags = next_cycle.circuit_flags();
        row.next_is_virtual = next_flags[CircuitFlags::VirtualInstruction];
        row.next_is_first_in_sequence = next_flags[CircuitFlags::IsFirstInSequence];
    }
}

fn stage1_rv64_flags(flags: CircuitFlagSet) -> [bool; NUM_CIRCUIT_FLAGS] {
    [
        flags[CircuitFlags::AddOperands],
        flags[CircuitFlags::SubtractOperands],
        flags[CircuitFlags::MultiplyOperands],
        flags[CircuitFlags::Load],
        flags[CircuitFlags::Store],
        flags[CircuitFlags::Jump],
        flags[CircuitFlags::WriteLookupOutputToRD],
        flags[CircuitFlags::VirtualInstruction],
        flags[CircuitFlags::Assert],
        flags[CircuitFlags::DoNotUpdateUnexpandedPC],
        flags[CircuitFlags::Advice],
        flags[CircuitFlags::IsCompressed],
        flags[CircuitFlags::IsFirstInSequence],
        flags[CircuitFlags::IsLastInSequence],
    ]
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
    flags: CircuitFlagSet,
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
    )
    .expect("shared preprocessing");
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
        BenchStage::Stage3 => prover.benchmark_stage3_ms(),
        BenchStage::Stage4 => prover.benchmark_stage4_ms(),
        BenchStage::Stage5 => prover.benchmark_stage5_ms(),
        BenchStage::Stage6 => prover.benchmark_stage6_ms(),
        BenchStage::Stage7 => prover.benchmark_stage7_ms(),
        BenchStage::Stage8 => prover.benchmark_stage8_ms(),
    }
}

fn e2e_progress(label: &str) {
    let Some(path) = std::env::var_os("JOLT_E2E_PROGRESS") else {
        return;
    };
    let rss = memory_stats::memory_stats()
        .map(|stats| stats.physical_mem)
        .unwrap_or(0);
    let timestamp_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_secs_f64() * 1000.0)
        .unwrap_or(0.0);
    if let Ok(mut file) = OpenOptions::new().create(true).append(true).open(path) {
        let _ = writeln!(file, "{timestamp_ms:.3}\t{rss}\t{label}");
    }
}

fn run_e2e_trace(cli: &Cli) -> E2eBenchReport {
    let core = if matches!(cli.e2e_stack, E2eStack::Both | E2eStack::Core) {
        Some(run_core_e2e_once(cli.program, cli.log_t, cli.num_iters))
    } else {
        None
    };

    let (bolt, setup) = if matches!(cli.e2e_stack, E2eStack::Both | E2eStack::Bolt) {
        e2e_progress("bolt.fixture_setup_for_inputs.begin");
        let fixture_span =
            tracing::info_span!(target: "jolt_bench_e2e", "bolt.fixture_setup_for_inputs")
                .entered();
        let (fixture_ms, fixture) =
            time_it(|| core_stage1_fixture(cli.program, cli.log_t, cli.num_iters));
        drop(fixture_span);
        e2e_progress("bolt.fixture_setup_for_inputs.end");

        e2e_progress("bolt.compile_programs.begin");
        let context_span =
            tracing::info_span!(target: "jolt_bench_e2e", "bolt.compile_programs").entered();
        let (program_context_ms, context) = time_it(|| bolt_e2e_context(&fixture));
        drop(context_span);
        e2e_progress("bolt.compile_programs.end");

        e2e_progress("bolt.reference_opening_inputs.begin");
        let openings_span =
            tracing::info_span!(target: "jolt_bench_e2e", "bolt.reference_opening_inputs")
                .entered();
        let (reference_openings_ms, (core_stage6, core_stage7)) =
            time_it(|| (core_stage6_data(&fixture), core_stage7_data(&fixture)));
        drop(openings_span);
        e2e_progress("bolt.reference_opening_inputs.end");

        (
            Some(run_bolt_e2e_once(
                &fixture,
                &context,
                &core_stage6,
                &core_stage7,
            )),
            Some(E2eSetupReport {
                fixture_ms,
                program_context_ms,
                reference_openings_ms,
            }),
        )
    } else {
        (None, None)
    };

    let trace_length = core
        .as_ref()
        .map_or(0, |run| run.trace_length)
        .max(bolt.as_ref().map_or(0, |run| run.trace_length));
    E2eBenchReport {
        program: cli.program.cli_name().to_string(),
        max_log_t: cli.log_t,
        actual_log_t: trace_length.trailing_zeros() as usize,
        trace_length,
        num_iters: cli.num_iters,
        ratio_vs_core_prove: core
            .as_ref()
            .zip(bolt.as_ref())
            .map(|(core, bolt)| bolt.prove_ms / core.prove_ms),
        core,
        bolt,
        setup,
    }
}

fn run_core_e2e_once(program: Program, log_t: usize, num_iters: u32) -> E2eRun {
    e2e_progress("core.e2e.begin");
    let root_span = tracing::info_span!(
        target: "jolt_bench_e2e",
        "core.e2e",
        stack = "core",
        program = program.cli_name(),
        log_t,
        num_iters
    )
    .entered();
    DoryGlobals::reset();
    let total_start = Instant::now();
    let max_trace_length = 1usize << log_t;
    let inputs = program.canonical_inputs_with(Some(num_iters));

    e2e_progress("core.decode_and_trace.begin");
    let decode_span =
        tracing::info_span!(target: "jolt_bench_e2e", "core.decode_and_trace").entered();
    let mut core_program = host::Program::new(program.guest_name());
    let (core_bytecode, init_memory_state, _, entry_address) = core_program.decode();
    let (_, trace, _, core_io_device) = core_program.trace(&inputs, &[], &[]);
    assert!(
        trace.len().next_power_of_two() <= max_trace_length,
        "trace length {} exceeds requested 2^{log_t} capacity",
        trace.len()
    );
    drop(decode_span);
    e2e_progress("core.decode_and_trace.end");

    e2e_progress("core.preprocessing.begin");
    let preprocessing_span =
        tracing::info_span!(target: "jolt_bench_e2e", "core.preprocessing").entered();
    let shared_preprocessing = JoltSharedPreprocessing::new(
        core_bytecode,
        core_io_device.memory_layout.clone(),
        init_memory_state,
        max_trace_length,
        entry_address,
    )
    .expect("shared preprocessing");
    let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing);
    drop(preprocessing_span);
    e2e_progress("core.preprocessing.end");

    e2e_progress("core.gen_from_elf.begin");
    let gen_span = tracing::info_span!(target: "jolt_bench_e2e", "core.gen_from_elf").entered();
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
    let program_io = prover.program_io.clone();
    drop(gen_span);
    e2e_progress("core.gen_from_elf.end");

    e2e_progress("core.prove.begin");
    let prove_span = tracing::info_span!(target: "jolt_bench_e2e", "core.prove").entered();
    let prove_start = Instant::now();
    let (proof, _debug): (CoreProof, _) = prover.prove();
    let prove_ms = prove_start.elapsed().as_secs_f64() * 1000.0;
    let trace_length = proof.trace_length;
    drop(prove_span);
    e2e_progress("core.prove.end");
    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;

    e2e_progress("core.verify.begin");
    let verify_span = tracing::info_span!(target: "jolt_bench_e2e", "core.verify").entered();
    let verify_start = Instant::now();
    let verifier_preprocessing = CoreVerifierPreprocessing::from(&prover_preprocessing);
    let verifier: CoreVerifier<'_> =
        CoreVerifier::new(&verifier_preprocessing, proof, program_io, None, None)
            .expect("construct core verifier");
    verifier.verify().expect("core verifier accepts core proof");
    let verify_ms = verify_start.elapsed().as_secs_f64() * 1000.0;
    drop(verify_span);
    e2e_progress("core.verify.end");
    drop(root_span);
    e2e_progress("core.e2e.end");

    E2eRun {
        stack: "core",
        total_ms,
        prove_ms,
        verify_ms,
        trace_length,
    }
}

fn run_bolt_e2e_once(
    fixture: &CoreStage1Fixture,
    context: &BoltE2eContext,
    core_stage6: &CoreStage6Data,
    core_stage7: &CoreStage7Data,
) -> E2eRun {
    e2e_progress("bolt.e2e.begin");
    let root_span = tracing::info_span!(
        target: "jolt_bench_e2e",
        "bolt.e2e",
        stack = "bolt",
        log_t = fixture.params.log_t,
        trace_length = fixture.proof.trace_length
    )
    .entered();
    let total_start = Instant::now();

    let mut transcript = Blake2bTranscript::<Fr>::new(TRANSCRIPT_LABEL);
    let rd_inc_values = dense_source(&fixture.cycle_inputs, "trace.rd_inc");
    let ram_inc_values = dense_source(&fixture.cycle_inputs, "trace.ram_inc");
    let instruction_keys = one_hot_source(&fixture.cycle_inputs, "trace.instruction_keys");
    let ram_addresses = one_hot_source(&fixture.cycle_inputs, "trace.ram_addresses");
    let bytecode_indices = one_hot_source(&fixture.cycle_inputs, "trace.bytecode_indices");
    let mut commitment_inputs = jolt_prover::stages::commitment::SparseCommitmentInputs::new(
        jolt_prover::stages::commitment::CommitmentOracleInputs {
            rd_inc: &rd_inc_values,
            ram_inc: &ram_inc_values,
            instruction_keys: &instruction_keys,
            ram_addresses: &ram_addresses,
            bytecode_indices: &bytecode_indices,
            untrusted_advice: None,
            trusted_advice: None,
        },
    );
    e2e_progress("bolt.preamble.begin");
    let preamble_span = tracing::info_span!(target: "jolt_bench_e2e", "bolt.preamble").entered();
    append_bolt_preamble(&mut transcript, fixture);
    drop(preamble_span);
    e2e_progress("bolt.preamble.end");

    e2e_progress("bolt.commitment.begin");
    let commitment_span =
        tracing::info_span!(target: "jolt_bench_e2e", "bolt.commitment").entered();
    let commitment_artifacts =
        jolt_prover::stages::commitment::prove_commitment_phase_with_program(
            context.commitment_prover_plan,
            &mut commitment_inputs,
            &fixture.pcs_setup,
            &mut transcript,
        )
        .expect("Bolt commitment prover succeeds");
    drop(commitment_span);
    e2e_progress("bolt.commitment.end");

    e2e_progress("bolt.stage1.begin");
    let stage1_span = tracing::info_span!(target: "jolt_bench_e2e", "bolt.stage1").entered();
    let r1cs_key = R1csKey::new(rv64::rv64_constraints::<Fr>(), fixture.proof.trace_length);
    let data = Stage1OuterRv64Data::new(&r1cs_key, &fixture.r1cs_witness, &fixture.rv64_cycles)
        .expect("valid R1CS witness shape");
    let (transcript_after_stage1, stage1_artifacts) = run_bolt_stage1_prover(
        context.stage1_prover_plan,
        context.num_cycle_vars,
        &data,
        transcript,
    );
    let mut transcript = transcript_after_stage1;
    drop(data);
    drop(r1cs_key);
    drop(stage1_span);
    e2e_progress("bolt.stage1.end");

    e2e_progress("bolt.stage2.begin");
    let stage2_span = tracing::info_span!(target: "jolt_bench_e2e", "bolt.stage2").entered();
    let ram_data = stage2_ram_data(fixture);
    let stage2_openings = stage2_opening_inputs(&stage1_artifacts);
    let (transcript_after_stage2, stage2_artifacts) = run_bolt_stage2_prover(
        context.stage2_prover_plan,
        fixture,
        &stage1_artifacts,
        &ram_data,
        transcript,
    );
    transcript = transcript_after_stage2;
    drop(stage2_span);
    e2e_progress("bolt.stage2.end");

    e2e_progress("bolt.stage3.begin");
    let stage3_span = tracing::info_span!(target: "jolt_bench_e2e", "bolt.stage3").entered();
    let stage3_openings = stage3_opening_inputs(&stage1_artifacts, &stage2_artifacts);
    let (transcript_after_stage3, stage3_artifacts) = run_bolt_stage3_prover(
        context.stage3_prover_plan,
        fixture,
        &stage3_openings,
        transcript,
    );
    let mut transcript = transcript_after_stage3;
    drop(stage3_span);
    e2e_progress("bolt.stage3.end");

    e2e_progress("bolt.stage4.begin");
    let stage4_span = tracing::info_span!(target: "jolt_bench_e2e", "bolt.stage4").entered();
    let stage4_openings = stage4_opening_inputs(
        &fixture.params,
        &fixture.initial_ram_state,
        &stage2_artifacts,
        &stage3_artifacts,
    );
    let stage4_rd_inc = stage4_rd_inc(&fixture.stage4_register_accesses);
    let stage4_ram_addresses = stage4_ram_address_indices(&fixture.ram_accesses);
    let stage4_ram_inc = stage2_ram_inc(&fixture.ram_accesses);
    let stage4_artifacts = run_bolt_stage4_prover(
        context.stage4_prover_plan,
        fixture,
        &stage4_openings,
        &stage4_rd_inc,
        &stage4_ram_addresses,
        &stage4_ram_inc,
        &mut transcript,
    );
    drop(stage4_span);
    e2e_progress("bolt.stage4.end");

    e2e_progress("bolt.stage5.begin");
    let stage5_span = tracing::info_span!(target: "jolt_bench_e2e", "bolt.stage5").entered();
    let stage5_openings = stage5_opening_inputs(
        &fixture.params,
        &stage2_openings,
        &stage2_artifacts,
        &stage4_artifacts,
    );
    let stage5_rd_write_addresses = stage4_rd_write_addresses(&fixture.stage4_register_accesses);
    let _stage5_artifacts = run_bolt_stage5_prover(
        context.stage5_prover_plan,
        fixture,
        &stage5_openings,
        &stage4_rd_inc,
        &stage4_ram_addresses,
        &stage5_rd_write_addresses,
        &mut transcript,
    );
    drop(_stage5_artifacts);
    drop(stage5_rd_write_addresses);
    drop(stage5_openings);
    drop(stage4_ram_inc);
    drop(stage4_ram_addresses);
    drop(stage4_rd_inc);
    drop(stage4_openings);
    drop(stage4_artifacts);
    drop(stage3_openings);
    drop(stage3_artifacts);
    drop(stage2_openings);
    drop(stage2_artifacts);
    drop(stage1_artifacts);
    drop(stage5_span);
    e2e_progress("bolt.stage5.end");

    e2e_progress("bolt.stage6.begin");
    let stage6_span = tracing::info_span!(target: "jolt_bench_e2e", "bolt.stage6").entered();
    e2e_progress("bolt.stage6_witness.begin");
    let stage6_witness = stage6_witness_polynomials(fixture, &core_stage6.opening_inputs);
    e2e_progress("bolt.stage6_witness.end");
    e2e_progress("bolt.stage6_prover.begin");
    let stage6_artifacts = run_bolt_stage6_prover(
        context.stage6_prover_plan,
        fixture,
        &core_stage6.opening_inputs,
        &stage6_witness,
        &mut transcript,
    );
    e2e_progress("bolt.stage6_prover.end");
    drop(stage6_span);
    e2e_progress("bolt.stage6.end");

    e2e_progress("bolt.stage7.begin");
    let stage7_span = tracing::info_span!(target: "jolt_bench_e2e", "bolt.stage7").entered();
    let stage7_artifacts = run_bolt_stage7_prover(
        context.stage7_prover_plan,
        &core_stage7.opening_inputs,
        &stage6_witness,
        &mut transcript,
    );
    drop(stage7_span);
    drop(stage6_witness);
    e2e_progress("bolt.stage7.end");

    e2e_progress("bolt.stage8.begin");
    let stage8_span = tracing::info_span!(target: "jolt_bench_e2e", "bolt.stage8").entered();
    let evaluation = jolt_prover::prove_jolt_evaluation_proof(
        context.stage8_prover_plan,
        &mut commitment_inputs,
        &fixture.pcs_setup,
        &commitment_artifacts,
        &stage6_artifacts,
        &stage7_artifacts,
        &core_stage7.opening_inputs,
        &mut transcript,
    )
    .expect("Bolt Stage 8 prover succeeds");
    drop(stage8_span);
    drop(stage6_artifacts);
    drop(stage7_artifacts);
    drop(commitment_artifacts);
    e2e_progress("bolt.stage8.end");

    let total_ms = total_start.elapsed().as_secs_f64() * 1000.0;
    e2e_progress("bolt.core_acceptance_check.begin");
    let verify_span =
        tracing::info_span!(target: "jolt_bench_e2e", "bolt.core_acceptance_check").entered();
    let verify_start = Instant::now();
    assert_core_accepts_bolt_stage8(fixture, &evaluation);
    let verify_ms = verify_start.elapsed().as_secs_f64() * 1000.0;
    drop(verify_span);
    e2e_progress("bolt.core_acceptance_check.end");
    drop(root_span);
    e2e_progress("bolt.e2e.end");

    E2eRun {
        stack: "bolt",
        total_ms,
        prove_ms: total_ms,
        verify_ms,
        trace_length: fixture.proof.trace_length,
    }
}

fn bolt_stage1_context(fixture: &CoreStage1Fixture) -> BoltStage1Context {
    let (commitment_prover_program, commitment_verifier_program) =
        bolt_commitment_programs_with_params(&fixture.params);
    let (stage1_prover_program, stage1_verifier_program) =
        bolt_stage1_programs_with_params(&fixture.params);
    let (stage2_prover_program, _) = bolt_stage2_programs_with_params(&fixture.params);
    let (stage3_prover_program, _) = bolt_stage3_programs_with_params(&fixture.params);
    let (stage4_prover_program, _) = bolt_stage4_programs_with_params(&fixture.params);
    let (stage5_prover_program, _) = bolt_stage5_programs_with_params(&fixture.params);
    let (stage6_prover_program, _) = bolt_stage6_programs_with_params(&fixture.params);
    let (stage7_prover_program, _) = bolt_stage7_programs_with_params(&fixture.params);
    let (stage8_prover_program, stage8_verifier_program) =
        bolt_stage8_programs_with_params(&fixture.params);
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
        stage3_prover_plan: leak_stage3_program(&stage3_prover_program),
        stage4_prover_plan: leak_stage4_program(&stage4_prover_program),
        stage5_prover_plan: leak_stage5_program(&stage5_prover_program),
        stage6_prover_plan: leak_stage6_program(&stage6_prover_program),
        stage7_prover_plan: leak_stage7_program(&stage7_prover_program),
        stage8_prover_plan: leak_generated_stage8_prover_program(&stage8_prover_program),
        stage8_verifier_plan: leak_generated_stage8_verifier_program(&stage8_verifier_program),
        num_cycle_vars: fixture.proof.trace_length.trailing_zeros() as usize,
    }
}

fn bolt_e2e_context(fixture: &CoreStage1Fixture) -> BoltE2eContext {
    let (commitment_prover_program, _) = bolt_commitment_programs_with_params(&fixture.params);
    let (stage1_prover_program, _) = bolt_stage1_programs_with_params(&fixture.params);
    let (stage2_prover_program, _) = bolt_stage2_programs_with_params(&fixture.params);
    let (stage3_prover_program, _) = bolt_stage3_programs_with_params(&fixture.params);
    let (stage4_prover_program, _) = bolt_stage4_programs_with_params(&fixture.params);
    let (stage5_prover_program, _) = bolt_stage5_programs_with_params(&fixture.params);
    let (stage6_prover_program, _) = bolt_stage6_programs_with_params(&fixture.params);
    let (stage7_prover_program, _) = bolt_stage7_programs_with_params(&fixture.params);
    let (stage8_prover_program, _) = bolt_stage8_programs_with_params(&fixture.params);

    BoltE2eContext {
        commitment_prover_plan: leak_generated_commitment_prover_program(
            &commitment_prover_program,
        ),
        stage1_prover_plan: leak_stage1_program(&stage1_prover_program),
        stage2_prover_plan: leak_stage2_program(&stage2_prover_program),
        stage3_prover_plan: leak_stage3_program(&stage3_prover_program),
        stage4_prover_plan: leak_stage4_program(&stage4_prover_program),
        stage5_prover_plan: leak_stage5_program(&stage5_prover_program),
        stage6_prover_plan: leak_stage6_program(&stage6_prover_program),
        stage7_prover_plan: leak_stage7_program(&stage7_prover_program),
        stage8_prover_plan: leak_generated_stage8_prover_program(&stage8_prover_program),
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
        BenchStage::Stage3 => assert_stage3_correctness(fixture, context),
        BenchStage::Stage4 => assert_stage4_correctness(fixture, context),
        BenchStage::Stage5 => assert_stage5_correctness(fixture, context),
        BenchStage::Stage6 => assert_stage6_correctness(fixture, context),
        BenchStage::Stage7 => assert_stage7_correctness(fixture, context),
        BenchStage::Stage8 => assert_stage8_correctness(fixture, context),
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

    assert_commitment_parity(fixture, context);
    assert_core_accepts_bolt_stage1(fixture, &stage1_artifacts);
    assert_core_states_match_bolt_stage1(fixture, prover_transcript.log());

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
    assert_commitment_parity(fixture, context);
    assert_core_accepts_bolt_stage2(fixture, &stage1_artifacts, &stage2_artifacts);
    assert_core_states_match_bolt_stage2(fixture, prover_transcript.log());

    CorrectnessReport {
        bolt_prover_verifier_transcript: true,
        core_accepts_bolt_stage: true,
        core_bolt_stage_transcript_states: true,
        commitment_parity: true,
    }
}

fn assert_stage3_correctness(
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
    let (mut prover_transcript, stage2_artifacts) = run_bolt_stage2_prover(
        context.stage2_prover_plan,
        fixture,
        &stage1_artifacts,
        &ram_data,
        prover_transcript,
    );
    let stage3_openings = stage3_opening_inputs(&stage1_artifacts, &stage2_artifacts);
    let stage3_inputs =
        Stage3ProverInputs::new(&stage3_openings).with_cycles(&fixture.stage3_cycles);
    let mut stage3_prover = Stage3ProverKernelExecutor::new(stage3_inputs);
    let stage3_artifacts = execute_stage3_program(
        context.stage3_prover_plan,
        Stage3ExecutionMode::Prover,
        &mut stage3_prover,
        &mut prover_transcript,
    )
    .expect("Bolt Stage 3 prover succeeds");

    let stage1_proof = Stage1Proof::from(stage1_artifacts.clone());
    let stage2_proof = Stage2Proof::from(stage2_artifacts.clone());
    let stage3_proof = Stage3Proof::from(stage3_artifacts.clone());
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
    let stage3_verifier_openings = stage3_opening_inputs(&verified_stage1, &verified_stage2);
    let mut stage3_verifier =
        Stage3VerifierKernelExecutor::new(&stage3_proof, &stage3_verifier_openings);
    let verified_stage3 = execute_stage3_program(
        context.stage3_prover_plan,
        Stage3ExecutionMode::Verifier,
        &mut stage3_verifier,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 3 verifier accepts prover proof");

    assert_eq!(
        stage3_artifacts.sumchecks.len(),
        verified_stage3.sumchecks.len()
    );
    assert_eq!(prover_transcript.log(), verifier_transcript.log());
    assert_commitment_parity(fixture, context);
    assert_core_accepts_bolt_stage3(
        fixture,
        &stage1_artifacts,
        &stage2_artifacts,
        &stage3_artifacts,
    );
    assert_core_states_match_bolt_stage3(fixture, prover_transcript.log());

    CorrectnessReport {
        bolt_prover_verifier_transcript: true,
        core_accepts_bolt_stage: true,
        core_bolt_stage_transcript_states: true,
        commitment_parity: true,
    }
}

fn assert_stage4_correctness(
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
    let stage3_openings = stage3_opening_inputs(&stage1_artifacts, &stage2_artifacts);
    let (mut prover_transcript, stage3_artifacts) = run_bolt_stage3_prover(
        context.stage3_prover_plan,
        fixture,
        &stage3_openings,
        prover_transcript,
    );
    let stage4_openings = stage4_opening_inputs(
        &fixture.params,
        &fixture.initial_ram_state,
        &stage2_artifacts,
        &stage3_artifacts,
    );
    let stage4_rd_inc = stage4_rd_inc(&fixture.stage4_register_accesses);
    let stage4_ram_addresses = stage4_ram_address_indices(&fixture.ram_accesses);
    let stage4_ram_inc = stage2_ram_inc(&fixture.ram_accesses);
    let stage4_artifacts = run_bolt_stage4_prover(
        context.stage4_prover_plan,
        fixture,
        &stage4_openings,
        &stage4_rd_inc,
        &stage4_ram_addresses,
        &stage4_ram_inc,
        &mut prover_transcript,
    );

    let stage1_proof = Stage1Proof::from(stage1_artifacts.clone());
    let stage2_proof = Stage2Proof::from(stage2_artifacts.clone());
    let stage3_proof = Stage3Proof::from(stage3_artifacts.clone());
    let stage4_proof = Stage4Proof::from(stage4_artifacts.clone());
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
    let stage3_verifier_openings = stage3_opening_inputs(&verified_stage1, &verified_stage2);
    let mut stage3_verifier =
        Stage3VerifierKernelExecutor::new(&stage3_proof, &stage3_verifier_openings);
    let verified_stage3 = execute_stage3_program(
        context.stage3_prover_plan,
        Stage3ExecutionMode::Verifier,
        &mut stage3_verifier,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 3 verifier accepts prover proof");
    let stage4_verifier_openings = stage4_opening_inputs(
        &fixture.params,
        &fixture.initial_ram_state,
        &verified_stage2,
        &verified_stage3,
    );
    let mut stage4_verifier =
        Stage4VerifierKernelExecutor::new(&stage4_proof, &stage4_verifier_openings);
    let verified_stage4 = execute_stage4_program(
        context.stage4_prover_plan,
        Stage4ExecutionMode::Verifier,
        &mut stage4_verifier,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 4 verifier accepts prover proof");

    assert_eq!(
        stage4_artifacts.sumchecks.len(),
        verified_stage4.sumchecks.len()
    );
    assert_eq!(prover_transcript.log(), verifier_transcript.log());
    assert_commitment_parity(fixture, context);
    assert_core_accepts_bolt_stage4(
        fixture,
        &stage1_artifacts,
        &stage2_artifacts,
        &stage3_artifacts,
        &stage4_artifacts,
    );
    assert_core_states_match_bolt_stage4(fixture, prover_transcript.log());

    CorrectnessReport {
        bolt_prover_verifier_transcript: true,
        core_accepts_bolt_stage: true,
        core_bolt_stage_transcript_states: true,
        commitment_parity: true,
    }
}

fn assert_stage5_correctness(
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
    let stage2_openings = stage2_opening_inputs(&stage1_artifacts);
    let (prover_transcript, stage2_artifacts) = run_bolt_stage2_prover(
        context.stage2_prover_plan,
        fixture,
        &stage1_artifacts,
        &ram_data,
        prover_transcript,
    );
    let stage3_openings = stage3_opening_inputs(&stage1_artifacts, &stage2_artifacts);
    let (mut prover_transcript, stage3_artifacts) = run_bolt_stage3_prover(
        context.stage3_prover_plan,
        fixture,
        &stage3_openings,
        prover_transcript,
    );
    let stage4_openings = stage4_opening_inputs(
        &fixture.params,
        &fixture.initial_ram_state,
        &stage2_artifacts,
        &stage3_artifacts,
    );
    let stage4_rd_inc = stage4_rd_inc(&fixture.stage4_register_accesses);
    let stage4_ram_addresses = stage4_ram_address_indices(&fixture.ram_accesses);
    let stage4_ram_inc = stage2_ram_inc(&fixture.ram_accesses);
    let stage4_artifacts = run_bolt_stage4_prover(
        context.stage4_prover_plan,
        fixture,
        &stage4_openings,
        &stage4_rd_inc,
        &stage4_ram_addresses,
        &stage4_ram_inc,
        &mut prover_transcript,
    );
    let stage5_openings = stage5_opening_inputs(
        &fixture.params,
        &stage2_openings,
        &stage2_artifacts,
        &stage4_artifacts,
    );
    let stage5_rd_write_addresses = stage4_rd_write_addresses(&fixture.stage4_register_accesses);
    let stage5_artifacts = run_bolt_stage5_prover(
        context.stage5_prover_plan,
        fixture,
        &stage5_openings,
        &stage4_rd_inc,
        &stage4_ram_addresses,
        &stage5_rd_write_addresses,
        &mut prover_transcript,
    );

    let stage1_proof = Stage1Proof::from(stage1_artifacts.clone());
    let stage2_proof = Stage2Proof::from(stage2_artifacts.clone());
    let stage3_proof = Stage3Proof::from(stage3_artifacts.clone());
    let stage4_proof = Stage4Proof::from(stage4_artifacts.clone());
    let stage5_proof = kernel_stage5::Stage5Proof {
        sumchecks: stage5_artifacts.sumchecks.clone(),
    };
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
    let stage2_verifier_openings = stage2_opening_inputs(&verified_stage1);
    let mut stage2_verifier =
        Stage2VerifierKernelExecutor::new(&stage2_proof, &stage2_verifier_openings)
            .with_ram_data(&ram_data);
    let verified_stage2 = execute_stage2_program(
        context.stage2_prover_plan,
        Stage2ExecutionMode::Verifier,
        &mut stage2_verifier,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 2 verifier accepts prover proof");
    let stage3_verifier_openings = stage3_opening_inputs(&verified_stage1, &verified_stage2);
    let mut stage3_verifier =
        Stage3VerifierKernelExecutor::new(&stage3_proof, &stage3_verifier_openings);
    let verified_stage3 = execute_stage3_program(
        context.stage3_prover_plan,
        Stage3ExecutionMode::Verifier,
        &mut stage3_verifier,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 3 verifier accepts prover proof");
    let stage4_verifier_openings = stage4_opening_inputs(
        &fixture.params,
        &fixture.initial_ram_state,
        &verified_stage2,
        &verified_stage3,
    );
    let mut stage4_verifier =
        Stage4VerifierKernelExecutor::new(&stage4_proof, &stage4_verifier_openings);
    let verified_stage4 = execute_stage4_program(
        context.stage4_prover_plan,
        Stage4ExecutionMode::Verifier,
        &mut stage4_verifier,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 4 verifier accepts prover proof");
    let stage5_verifier_openings = stage5_opening_inputs(
        &fixture.params,
        &stage2_verifier_openings,
        &verified_stage2,
        &verified_stage4,
    );
    let mut stage5_verifier = kernel_stage5::Stage5ProofCarryingKernelExecutor::new(
        &stage5_proof,
        &stage5_verifier_openings,
    );
    let verified_stage5 = kernel_stage5::execute_stage5_program(
        context.stage5_prover_plan,
        kernel_stage5::Stage5ExecutionMode::Verifier,
        &mut stage5_verifier,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 5 verifier accepts prover proof");

    assert_eq!(
        stage5_artifacts.sumchecks.len(),
        verified_stage5.sumchecks.len()
    );
    assert_eq!(prover_transcript.log(), verifier_transcript.log());
    assert_commitment_parity(fixture, context);
    assert_core_accepts_bolt_stage5(
        fixture,
        &stage1_artifacts,
        &stage2_artifacts,
        &stage3_artifacts,
        &stage4_artifacts,
        &stage5_artifacts,
    );
    assert_core_states_match_bolt_stage5(fixture, prover_transcript.log());

    CorrectnessReport {
        bolt_prover_verifier_transcript: true,
        core_accepts_bolt_stage: true,
        core_bolt_stage_transcript_states: true,
        commitment_parity: true,
    }
}

fn assert_stage6_correctness(
    fixture: &CoreStage1Fixture,
    context: &BoltStage1Context,
) -> CorrectnessReport {
    let core_stage6 = core_stage6_data(fixture);
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
    let stage2_openings = stage2_opening_inputs(&stage1_artifacts);
    let (prover_transcript, stage2_artifacts) = run_bolt_stage2_prover(
        context.stage2_prover_plan,
        fixture,
        &stage1_artifacts,
        &ram_data,
        prover_transcript,
    );
    let stage3_openings = stage3_opening_inputs(&stage1_artifacts, &stage2_artifacts);
    let (mut prover_transcript, stage3_artifacts) = run_bolt_stage3_prover(
        context.stage3_prover_plan,
        fixture,
        &stage3_openings,
        prover_transcript,
    );
    let stage4_openings = stage4_opening_inputs(
        &fixture.params,
        &fixture.initial_ram_state,
        &stage2_artifacts,
        &stage3_artifacts,
    );
    let stage4_rd_inc = stage4_rd_inc(&fixture.stage4_register_accesses);
    let stage4_ram_addresses = stage4_ram_address_indices(&fixture.ram_accesses);
    let stage4_ram_inc = stage2_ram_inc(&fixture.ram_accesses);
    let stage4_artifacts = run_bolt_stage4_prover(
        context.stage4_prover_plan,
        fixture,
        &stage4_openings,
        &stage4_rd_inc,
        &stage4_ram_addresses,
        &stage4_ram_inc,
        &mut prover_transcript,
    );
    let stage5_openings = stage5_opening_inputs(
        &fixture.params,
        &stage2_openings,
        &stage2_artifacts,
        &stage4_artifacts,
    );
    let stage5_rd_write_addresses = stage4_rd_write_addresses(&fixture.stage4_register_accesses);
    let stage5_artifacts = run_bolt_stage5_prover(
        context.stage5_prover_plan,
        fixture,
        &stage5_openings,
        &stage4_rd_inc,
        &stage4_ram_addresses,
        &stage5_rd_write_addresses,
        &mut prover_transcript,
    );
    let stage6_witness = stage6_witness_polynomials(fixture, &core_stage6.opening_inputs);
    let stage6_artifacts = run_bolt_stage6_prover(
        context.stage6_prover_plan,
        fixture,
        &core_stage6.opening_inputs,
        &stage6_witness,
        &mut prover_transcript,
    );

    let stage1_proof = Stage1Proof::from(stage1_artifacts.clone());
    let stage2_proof = Stage2Proof::from(stage2_artifacts.clone());
    let stage3_proof = Stage3Proof::from(stage3_artifacts.clone());
    let stage4_proof = Stage4Proof::from(stage4_artifacts.clone());
    let stage5_proof = kernel_stage5::Stage5Proof {
        sumchecks: stage5_artifacts.sumchecks.clone(),
    };
    let stage6_proof = kernel_stage6::Stage6Proof {
        sumchecks: stage6_artifacts.sumchecks.clone(),
    };
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
    let stage2_verifier_openings = stage2_opening_inputs(&verified_stage1);
    let mut stage2_verifier =
        Stage2VerifierKernelExecutor::new(&stage2_proof, &stage2_verifier_openings)
            .with_ram_data(&ram_data);
    let verified_stage2 = execute_stage2_program(
        context.stage2_prover_plan,
        Stage2ExecutionMode::Verifier,
        &mut stage2_verifier,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 2 verifier accepts prover proof");
    let stage3_verifier_openings = stage3_opening_inputs(&verified_stage1, &verified_stage2);
    let mut stage3_verifier =
        Stage3VerifierKernelExecutor::new(&stage3_proof, &stage3_verifier_openings);
    let verified_stage3 = execute_stage3_program(
        context.stage3_prover_plan,
        Stage3ExecutionMode::Verifier,
        &mut stage3_verifier,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 3 verifier accepts prover proof");
    let stage4_verifier_openings = stage4_opening_inputs(
        &fixture.params,
        &fixture.initial_ram_state,
        &verified_stage2,
        &verified_stage3,
    );
    let mut stage4_verifier =
        Stage4VerifierKernelExecutor::new(&stage4_proof, &stage4_verifier_openings);
    let verified_stage4 = execute_stage4_program(
        context.stage4_prover_plan,
        Stage4ExecutionMode::Verifier,
        &mut stage4_verifier,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 4 verifier accepts prover proof");
    let stage5_verifier_openings = stage5_opening_inputs(
        &fixture.params,
        &stage2_verifier_openings,
        &verified_stage2,
        &verified_stage4,
    );
    let mut stage5_verifier = kernel_stage5::Stage5ProofCarryingKernelExecutor::new(
        &stage5_proof,
        &stage5_verifier_openings,
    );
    let verified_stage5 = kernel_stage5::execute_stage5_program(
        context.stage5_prover_plan,
        kernel_stage5::Stage5ExecutionMode::Verifier,
        &mut stage5_verifier,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 5 verifier accepts prover proof");
    let mut stage6_verifier = kernel_stage6::Stage6ProofCarryingKernelExecutor::new(
        &stage6_proof,
        &core_stage6.opening_inputs,
    )
    .with_bytecode_read_raf_data(stage6_bytecode_data(fixture));
    let verified_stage6 = kernel_stage6::execute_stage6_program(
        context.stage6_prover_plan,
        kernel_stage6::Stage6ExecutionMode::Verifier,
        &mut stage6_verifier,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 6 verifier accepts prover proof");

    assert_eq!(
        stage5_artifacts.sumchecks.len(),
        verified_stage5.sumchecks.len()
    );
    assert_eq!(
        stage6_artifacts.sumchecks.len(),
        verified_stage6.sumchecks.len()
    );
    assert_eq!(prover_transcript.log(), verifier_transcript.log());
    assert_commitment_parity(fixture, context);
    assert_core_accepts_bolt_stage6(
        fixture,
        &stage1_artifacts,
        &stage2_artifacts,
        &stage3_artifacts,
        &stage4_artifacts,
        &stage5_artifacts,
        &stage6_artifacts,
    );
    assert_core_states_match_bolt_stage6(fixture, prover_transcript.log());
    assert_eq!(
        core_stage6.transcript_states,
        transcript_states(prover_transcript.log())
    );

    CorrectnessReport {
        bolt_prover_verifier_transcript: true,
        core_accepts_bolt_stage: true,
        core_bolt_stage_transcript_states: true,
        commitment_parity: true,
    }
}

fn assert_stage7_correctness(
    fixture: &CoreStage1Fixture,
    context: &BoltStage1Context,
) -> CorrectnessReport {
    let core_stage6 = core_stage6_data(fixture);
    let core_stage7 = core_stage7_data(fixture);
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
    let stage2_openings = stage2_opening_inputs(&stage1_artifacts);
    let (prover_transcript, stage2_artifacts) = run_bolt_stage2_prover(
        context.stage2_prover_plan,
        fixture,
        &stage1_artifacts,
        &ram_data,
        prover_transcript,
    );
    let stage3_openings = stage3_opening_inputs(&stage1_artifacts, &stage2_artifacts);
    let (mut prover_transcript, stage3_artifacts) = run_bolt_stage3_prover(
        context.stage3_prover_plan,
        fixture,
        &stage3_openings,
        prover_transcript,
    );
    let stage4_openings = stage4_opening_inputs(
        &fixture.params,
        &fixture.initial_ram_state,
        &stage2_artifacts,
        &stage3_artifacts,
    );
    let stage4_rd_inc = stage4_rd_inc(&fixture.stage4_register_accesses);
    let stage4_ram_addresses = stage4_ram_address_indices(&fixture.ram_accesses);
    let stage4_ram_inc = stage2_ram_inc(&fixture.ram_accesses);
    let stage4_artifacts = run_bolt_stage4_prover(
        context.stage4_prover_plan,
        fixture,
        &stage4_openings,
        &stage4_rd_inc,
        &stage4_ram_addresses,
        &stage4_ram_inc,
        &mut prover_transcript,
    );
    let stage5_openings = stage5_opening_inputs(
        &fixture.params,
        &stage2_openings,
        &stage2_artifacts,
        &stage4_artifacts,
    );
    let stage5_rd_write_addresses = stage4_rd_write_addresses(&fixture.stage4_register_accesses);
    let stage5_artifacts = run_bolt_stage5_prover(
        context.stage5_prover_plan,
        fixture,
        &stage5_openings,
        &stage4_rd_inc,
        &stage4_ram_addresses,
        &stage5_rd_write_addresses,
        &mut prover_transcript,
    );
    let stage6_witness = stage6_witness_polynomials(fixture, &core_stage6.opening_inputs);
    let stage6_artifacts = run_bolt_stage6_prover(
        context.stage6_prover_plan,
        fixture,
        &core_stage6.opening_inputs,
        &stage6_witness,
        &mut prover_transcript,
    );
    let stage7_artifacts = run_bolt_stage7_prover(
        context.stage7_prover_plan,
        &core_stage7.opening_inputs,
        &stage6_witness,
        &mut prover_transcript,
    );

    let stage1_proof = Stage1Proof::from(stage1_artifacts.clone());
    let stage2_proof = Stage2Proof::from(stage2_artifacts.clone());
    let stage3_proof = Stage3Proof::from(stage3_artifacts.clone());
    let stage4_proof = Stage4Proof::from(stage4_artifacts.clone());
    let stage5_proof = kernel_stage5::Stage5Proof {
        sumchecks: stage5_artifacts.sumchecks.clone(),
    };
    let stage6_proof = kernel_stage6::Stage6Proof {
        sumchecks: stage6_artifacts.sumchecks.clone(),
    };
    let stage7_proof = kernel_stage7::Stage7Proof {
        sumchecks: stage7_artifacts.sumchecks.clone(),
    };
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
    let stage2_verifier_openings = stage2_opening_inputs(&verified_stage1);
    let mut stage2_verifier =
        Stage2VerifierKernelExecutor::new(&stage2_proof, &stage2_verifier_openings)
            .with_ram_data(&ram_data);
    let verified_stage2 = execute_stage2_program(
        context.stage2_prover_plan,
        Stage2ExecutionMode::Verifier,
        &mut stage2_verifier,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 2 verifier accepts prover proof");
    let stage3_verifier_openings = stage3_opening_inputs(&verified_stage1, &verified_stage2);
    let mut stage3_verifier =
        Stage3VerifierKernelExecutor::new(&stage3_proof, &stage3_verifier_openings);
    let verified_stage3 = execute_stage3_program(
        context.stage3_prover_plan,
        Stage3ExecutionMode::Verifier,
        &mut stage3_verifier,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 3 verifier accepts prover proof");
    let stage4_verifier_openings = stage4_opening_inputs(
        &fixture.params,
        &fixture.initial_ram_state,
        &verified_stage2,
        &verified_stage3,
    );
    let mut stage4_verifier =
        Stage4VerifierKernelExecutor::new(&stage4_proof, &stage4_verifier_openings);
    let verified_stage4 = execute_stage4_program(
        context.stage4_prover_plan,
        Stage4ExecutionMode::Verifier,
        &mut stage4_verifier,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 4 verifier accepts prover proof");
    let stage5_verifier_openings = stage5_opening_inputs(
        &fixture.params,
        &stage2_verifier_openings,
        &verified_stage2,
        &verified_stage4,
    );
    let mut stage5_verifier = kernel_stage5::Stage5ProofCarryingKernelExecutor::new(
        &stage5_proof,
        &stage5_verifier_openings,
    );
    let verified_stage5 = kernel_stage5::execute_stage5_program(
        context.stage5_prover_plan,
        kernel_stage5::Stage5ExecutionMode::Verifier,
        &mut stage5_verifier,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 5 verifier accepts prover proof");
    let mut stage6_verifier = kernel_stage6::Stage6ProofCarryingKernelExecutor::new(
        &stage6_proof,
        &core_stage6.opening_inputs,
    )
    .with_bytecode_read_raf_data(stage6_bytecode_data(fixture));
    let verified_stage6 = kernel_stage6::execute_stage6_program(
        context.stage6_prover_plan,
        kernel_stage6::Stage6ExecutionMode::Verifier,
        &mut stage6_verifier,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 6 verifier accepts prover proof");
    let mut stage7_verifier = kernel_stage7::Stage7ProofCarryingKernelExecutor::new(
        &stage7_proof,
        &core_stage7.opening_inputs,
    );
    let verified_stage7 = kernel_stage7::execute_stage7_program(
        context.stage7_prover_plan,
        kernel_stage7::Stage7ExecutionMode::Verifier,
        &mut stage7_verifier,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 7 verifier accepts prover proof");

    assert_eq!(
        stage5_artifacts.sumchecks.len(),
        verified_stage5.sumchecks.len()
    );
    assert_eq!(
        stage6_artifacts.sumchecks.len(),
        verified_stage6.sumchecks.len()
    );
    assert_eq!(
        stage7_artifacts.sumchecks.len(),
        verified_stage7.sumchecks.len()
    );
    assert_eq!(prover_transcript.log(), verifier_transcript.log());
    assert_commitment_parity(fixture, context);
    assert_core_accepts_bolt_stage7(
        fixture,
        &stage1_artifacts,
        &stage2_artifacts,
        &stage3_artifacts,
        &stage4_artifacts,
        &stage5_artifacts,
        &stage6_artifacts,
        &stage7_artifacts,
    );
    assert_core_states_match_bolt_stage7(fixture, prover_transcript.log());
    assert_eq!(
        core_stage7.transcript_states,
        transcript_states(prover_transcript.log())
    );

    CorrectnessReport {
        bolt_prover_verifier_transcript: true,
        core_accepts_bolt_stage: true,
        core_bolt_stage_transcript_states: true,
        commitment_parity: true,
    }
}

fn assert_stage8_correctness(
    fixture: &CoreStage1Fixture,
    context: &BoltStage1Context,
) -> CorrectnessReport {
    let _stage7_report = assert_stage7_correctness(fixture, context);
    let mut prefix =
        bolt_stage8_prefix::<RecordingTranscript<Blake2bTranscript<Fr>>>(fixture, context);
    let stage8_start = prefix.transcript.clone();
    let mut commitment_inputs = BenchCommitmentInputs {
        data: &prefix.oracle_data,
    };
    let evaluation = jolt_prover::prove_jolt_evaluation_proof(
        context.stage8_prover_plan,
        &mut commitment_inputs,
        &fixture.pcs_setup,
        &prefix.commitment_artifacts,
        &prefix.stage6_artifacts,
        &prefix.stage7_artifacts,
        &prefix.stage7_opening_inputs,
        &mut prefix.transcript,
    )
    .expect("Bolt Stage 8 prover succeeds");

    let commitment_artifacts = generated_commitment_artifacts(
        &context.commitment_prover_program,
        &context.commitment_prover_trace,
    );
    let stage6_proof = jolt_stage6_proof(&prefix.stage6_artifacts);
    let stage7_proof = jolt_stage7_proof(&prefix.stage7_artifacts);
    let verifier_stage7_openings = verifier_stage7_opening_inputs(&prefix.stage7_opening_inputs);
    let verifier_setup = DoryScheme::verifier_setup(&fixture.pcs_setup);
    let mut verifier_transcript = stage8_start;
    jolt_verifier::verify_jolt_evaluation_proof(
        context.stage8_verifier_plan,
        &evaluation,
        &commitment_artifacts,
        &stage6_proof,
        &stage7_proof,
        &verifier_stage7_openings,
        &verifier_setup,
        &mut verifier_transcript,
    )
    .expect("Bolt Stage 8 verifier accepts prover proof");
    assert_eq!(prefix.transcript.log(), verifier_transcript.log());
    assert_core_accepts_bolt_stage8(fixture, &evaluation);
    assert_core_states_match_bolt_stage8(fixture, prefix.transcript.log());

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
        BenchStage::Stage3 => measure_bolt_stage3(fixture, context, iters, warmup),
        BenchStage::Stage4 => measure_bolt_stage4(fixture, context, iters, warmup),
        BenchStage::Stage5 => measure_bolt_stage5(fixture, context, iters, warmup),
        BenchStage::Stage6 => measure_bolt_stage6(fixture, context, iters, warmup),
        BenchStage::Stage7 => measure_bolt_stage7(fixture, context, iters, warmup),
        BenchStage::Stage8 => measure_bolt_stage8(fixture, context, iters, warmup),
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

fn measure_bolt_stage3(
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
    let (stage3_prefix, stage2_artifacts) = run_bolt_stage2_prover(
        context.stage2_prover_plan,
        fixture,
        &stage1_artifacts,
        &ram_data,
        stage2_prefix,
    );
    let stage3_openings = stage3_opening_inputs(&stage1_artifacts, &stage2_artifacts);

    for _ in 0..warmup {
        let _ = run_bolt_stage3_prover(
            context.stage3_prover_plan,
            fixture,
            &stage3_openings,
            stage3_prefix.clone(),
        );
    }

    (0..iters)
        .map(|_| {
            time_it(|| {
                run_bolt_stage3_prover(
                    context.stage3_prover_plan,
                    fixture,
                    &stage3_openings,
                    stage3_prefix.clone(),
                )
            })
            .0
        })
        .collect()
}

fn measure_bolt_stage4(
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
    let (stage3_prefix, stage2_artifacts) = run_bolt_stage2_prover(
        context.stage2_prover_plan,
        fixture,
        &stage1_artifacts,
        &ram_data,
        stage2_prefix,
    );
    let stage3_openings = stage3_opening_inputs(&stage1_artifacts, &stage2_artifacts);
    let (stage4_prefix, stage3_artifacts) = run_bolt_stage3_prover(
        context.stage3_prover_plan,
        fixture,
        &stage3_openings,
        stage3_prefix,
    );
    let stage4_openings = stage4_opening_inputs(
        &fixture.params,
        &fixture.initial_ram_state,
        &stage2_artifacts,
        &stage3_artifacts,
    );
    let stage4_rd_inc = stage4_rd_inc(&fixture.stage4_register_accesses);
    let stage4_ram_addresses = stage4_ram_address_indices(&fixture.ram_accesses);
    let stage4_ram_inc = stage2_ram_inc(&fixture.ram_accesses);

    for _ in 0..warmup {
        let mut transcript = stage4_prefix.clone();
        let _ = run_bolt_stage4_prover(
            context.stage4_prover_plan,
            fixture,
            &stage4_openings,
            &stage4_rd_inc,
            &stage4_ram_addresses,
            &stage4_ram_inc,
            &mut transcript,
        );
    }

    (0..iters)
        .map(|_| {
            time_it(|| {
                let mut transcript = stage4_prefix.clone();
                run_bolt_stage4_prover(
                    context.stage4_prover_plan,
                    fixture,
                    &stage4_openings,
                    &stage4_rd_inc,
                    &stage4_ram_addresses,
                    &stage4_ram_inc,
                    &mut transcript,
                )
            })
            .0
        })
        .collect()
}

fn measure_bolt_stage5(
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
    let stage2_openings = stage2_opening_inputs(&stage1_artifacts);
    let (stage3_prefix, stage2_artifacts) = run_bolt_stage2_prover(
        context.stage2_prover_plan,
        fixture,
        &stage1_artifacts,
        &ram_data,
        stage2_prefix,
    );
    let stage3_openings = stage3_opening_inputs(&stage1_artifacts, &stage2_artifacts);
    let (mut stage4_prefix, stage3_artifacts) = run_bolt_stage3_prover(
        context.stage3_prover_plan,
        fixture,
        &stage3_openings,
        stage3_prefix,
    );
    let stage4_openings = stage4_opening_inputs(
        &fixture.params,
        &fixture.initial_ram_state,
        &stage2_artifacts,
        &stage3_artifacts,
    );
    let stage4_rd_inc = stage4_rd_inc(&fixture.stage4_register_accesses);
    let stage4_ram_addresses = stage4_ram_address_indices(&fixture.ram_accesses);
    let stage4_ram_inc = stage2_ram_inc(&fixture.ram_accesses);
    let stage4_artifacts = run_bolt_stage4_prover(
        context.stage4_prover_plan,
        fixture,
        &stage4_openings,
        &stage4_rd_inc,
        &stage4_ram_addresses,
        &stage4_ram_inc,
        &mut stage4_prefix,
    );
    let stage5_openings = stage5_opening_inputs(
        &fixture.params,
        &stage2_openings,
        &stage2_artifacts,
        &stage4_artifacts,
    );
    let stage5_rd_write_addresses = stage4_rd_write_addresses(&fixture.stage4_register_accesses);

    for _ in 0..warmup {
        let mut transcript = stage4_prefix.clone();
        let _ = run_bolt_stage5_prover(
            context.stage5_prover_plan,
            fixture,
            &stage5_openings,
            &stage4_rd_inc,
            &stage4_ram_addresses,
            &stage5_rd_write_addresses,
            &mut transcript,
        );
    }

    (0..iters)
        .map(|_| {
            time_it(|| {
                let mut transcript = stage4_prefix.clone();
                run_bolt_stage5_prover(
                    context.stage5_prover_plan,
                    fixture,
                    &stage5_openings,
                    &stage4_rd_inc,
                    &stage4_ram_addresses,
                    &stage5_rd_write_addresses,
                    &mut transcript,
                )
            })
            .0
        })
        .collect()
}

fn measure_bolt_stage6(
    fixture: &CoreStage1Fixture,
    context: &BoltStage1Context,
    iters: usize,
    warmup: usize,
) -> Vec<f64> {
    let core_stage6 = core_stage6_data(fixture);
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
    let stage2_openings = stage2_opening_inputs(&stage1_artifacts);
    let (stage3_prefix, stage2_artifacts) = run_bolt_stage2_prover(
        context.stage2_prover_plan,
        fixture,
        &stage1_artifacts,
        &ram_data,
        stage2_prefix,
    );
    let stage3_openings = stage3_opening_inputs(&stage1_artifacts, &stage2_artifacts);
    let (mut stage4_prefix, stage3_artifacts) = run_bolt_stage3_prover(
        context.stage3_prover_plan,
        fixture,
        &stage3_openings,
        stage3_prefix,
    );
    let stage4_openings = stage4_opening_inputs(
        &fixture.params,
        &fixture.initial_ram_state,
        &stage2_artifacts,
        &stage3_artifacts,
    );
    let stage4_rd_inc = stage4_rd_inc(&fixture.stage4_register_accesses);
    let stage4_ram_addresses = stage4_ram_address_indices(&fixture.ram_accesses);
    let stage4_ram_inc = stage2_ram_inc(&fixture.ram_accesses);
    let stage4_artifacts = run_bolt_stage4_prover(
        context.stage4_prover_plan,
        fixture,
        &stage4_openings,
        &stage4_rd_inc,
        &stage4_ram_addresses,
        &stage4_ram_inc,
        &mut stage4_prefix,
    );
    let stage5_openings = stage5_opening_inputs(
        &fixture.params,
        &stage2_openings,
        &stage2_artifacts,
        &stage4_artifacts,
    );
    let stage5_rd_write_addresses = stage4_rd_write_addresses(&fixture.stage4_register_accesses);
    let _stage5_artifacts = run_bolt_stage5_prover(
        context.stage5_prover_plan,
        fixture,
        &stage5_openings,
        &stage4_rd_inc,
        &stage4_ram_addresses,
        &stage5_rd_write_addresses,
        &mut stage4_prefix,
    );
    let stage6_witness = stage6_witness_polynomials(fixture, &core_stage6.opening_inputs);

    for _ in 0..warmup {
        let mut transcript = stage4_prefix.clone();
        let _ = run_bolt_stage6_prover(
            context.stage6_prover_plan,
            fixture,
            &core_stage6.opening_inputs,
            &stage6_witness,
            &mut transcript,
        );
    }

    (0..iters)
        .map(|_| {
            time_it(|| {
                let mut transcript = stage4_prefix.clone();
                run_bolt_stage6_prover(
                    context.stage6_prover_plan,
                    fixture,
                    &core_stage6.opening_inputs,
                    &stage6_witness,
                    &mut transcript,
                )
            })
            .0
        })
        .collect()
}

fn measure_bolt_stage7(
    fixture: &CoreStage1Fixture,
    context: &BoltStage1Context,
    iters: usize,
    warmup: usize,
) -> Vec<f64> {
    let core_stage6 = core_stage6_data(fixture);
    let core_stage7 = core_stage7_data(fixture);
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
    let stage2_openings = stage2_opening_inputs(&stage1_artifacts);
    let (stage3_prefix, stage2_artifacts) = run_bolt_stage2_prover(
        context.stage2_prover_plan,
        fixture,
        &stage1_artifacts,
        &ram_data,
        stage2_prefix,
    );
    let stage3_openings = stage3_opening_inputs(&stage1_artifacts, &stage2_artifacts);
    let (mut stage4_prefix, stage3_artifacts) = run_bolt_stage3_prover(
        context.stage3_prover_plan,
        fixture,
        &stage3_openings,
        stage3_prefix,
    );
    let stage4_openings = stage4_opening_inputs(
        &fixture.params,
        &fixture.initial_ram_state,
        &stage2_artifacts,
        &stage3_artifacts,
    );
    let stage4_rd_inc = stage4_rd_inc(&fixture.stage4_register_accesses);
    let stage4_ram_addresses = stage4_ram_address_indices(&fixture.ram_accesses);
    let stage4_ram_inc = stage2_ram_inc(&fixture.ram_accesses);
    let stage4_artifacts = run_bolt_stage4_prover(
        context.stage4_prover_plan,
        fixture,
        &stage4_openings,
        &stage4_rd_inc,
        &stage4_ram_addresses,
        &stage4_ram_inc,
        &mut stage4_prefix,
    );
    let stage5_openings = stage5_opening_inputs(
        &fixture.params,
        &stage2_openings,
        &stage2_artifacts,
        &stage4_artifacts,
    );
    let stage5_rd_write_addresses = stage4_rd_write_addresses(&fixture.stage4_register_accesses);
    let _stage5_artifacts = run_bolt_stage5_prover(
        context.stage5_prover_plan,
        fixture,
        &stage5_openings,
        &stage4_rd_inc,
        &stage4_ram_addresses,
        &stage5_rd_write_addresses,
        &mut stage4_prefix,
    );
    let stage6_witness = stage6_witness_polynomials(fixture, &core_stage6.opening_inputs);
    let _stage6_artifacts = run_bolt_stage6_prover(
        context.stage6_prover_plan,
        fixture,
        &core_stage6.opening_inputs,
        &stage6_witness,
        &mut stage4_prefix,
    );

    for _ in 0..warmup {
        let mut transcript = stage4_prefix.clone();
        let _ = run_bolt_stage7_prover(
            context.stage7_prover_plan,
            &core_stage7.opening_inputs,
            &stage6_witness,
            &mut transcript,
        );
    }

    (0..iters)
        .map(|_| {
            time_it(|| {
                let mut transcript = stage4_prefix.clone();
                run_bolt_stage7_prover(
                    context.stage7_prover_plan,
                    &core_stage7.opening_inputs,
                    &stage6_witness,
                    &mut transcript,
                )
            })
            .0
        })
        .collect()
}

fn measure_bolt_stage8(
    fixture: &CoreStage1Fixture,
    context: &BoltStage1Context,
    iters: usize,
    warmup: usize,
) -> Vec<f64> {
    let prefix = bolt_stage8_prefix::<Blake2bTranscript<Fr>>(fixture, context);

    for _ in 0..warmup {
        let mut transcript = prefix.transcript.clone();
        let mut commitment_inputs = BenchCommitmentInputs {
            data: &prefix.oracle_data,
        };
        let _ = jolt_prover::prove_jolt_evaluation_proof(
            context.stage8_prover_plan,
            &mut commitment_inputs,
            &fixture.pcs_setup,
            &prefix.commitment_artifacts,
            &prefix.stage6_artifacts,
            &prefix.stage7_artifacts,
            &prefix.stage7_opening_inputs,
            &mut transcript,
        )
        .expect("Bolt Stage 8 prover succeeds");
    }

    (0..iters)
        .map(|_| {
            time_it(|| {
                let mut transcript = prefix.transcript.clone();
                let mut commitment_inputs = BenchCommitmentInputs {
                    data: &prefix.oracle_data,
                };
                jolt_prover::prove_jolt_evaluation_proof(
                    context.stage8_prover_plan,
                    &mut commitment_inputs,
                    &fixture.pcs_setup,
                    &prefix.commitment_artifacts,
                    &prefix.stage6_artifacts,
                    &prefix.stage7_artifacts,
                    &prefix.stage7_opening_inputs,
                    &mut transcript,
                )
                .expect("Bolt Stage 8 prover succeeds")
            })
            .0
        })
        .collect()
}

fn bolt_stage8_prefix<T>(
    fixture: &CoreStage1Fixture,
    context: &BoltStage1Context,
) -> Stage8Prefix<T>
where
    T: Transcript<Challenge = Fr>,
{
    let core_stage6 = core_stage6_data(fixture);
    let core_stage7 = core_stage7_data(fixture);
    let r1cs_key = R1csKey::new(rv64::rv64_constraints::<Fr>(), fixture.proof.trace_length);
    let data = Stage1OuterRv64Data::new(&r1cs_key, &fixture.r1cs_witness, &fixture.rv64_cycles)
        .expect("valid R1CS witness shape");
    let mut prefix = T::new(TRANSCRIPT_LABEL);
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
    let stage2_openings = stage2_opening_inputs(&stage1_artifacts);
    let (stage3_prefix, stage2_artifacts) = run_bolt_stage2_prover(
        context.stage2_prover_plan,
        fixture,
        &stage1_artifacts,
        &ram_data,
        stage2_prefix,
    );
    let stage3_openings = stage3_opening_inputs(&stage1_artifacts, &stage2_artifacts);
    let (mut stage4_prefix, stage3_artifacts) = run_bolt_stage3_prover(
        context.stage3_prover_plan,
        fixture,
        &stage3_openings,
        stage3_prefix,
    );
    let stage4_openings = stage4_opening_inputs(
        &fixture.params,
        &fixture.initial_ram_state,
        &stage2_artifacts,
        &stage3_artifacts,
    );
    let stage4_rd_inc = stage4_rd_inc(&fixture.stage4_register_accesses);
    let stage4_ram_addresses = stage4_ram_address_indices(&fixture.ram_accesses);
    let stage4_ram_inc = stage2_ram_inc(&fixture.ram_accesses);
    let stage4_artifacts = run_bolt_stage4_prover(
        context.stage4_prover_plan,
        fixture,
        &stage4_openings,
        &stage4_rd_inc,
        &stage4_ram_addresses,
        &stage4_ram_inc,
        &mut stage4_prefix,
    );
    let stage5_openings = stage5_opening_inputs(
        &fixture.params,
        &stage2_openings,
        &stage2_artifacts,
        &stage4_artifacts,
    );
    let stage5_rd_write_addresses = stage4_rd_write_addresses(&fixture.stage4_register_accesses);
    let _stage5_artifacts = run_bolt_stage5_prover(
        context.stage5_prover_plan,
        fixture,
        &stage5_openings,
        &stage4_rd_inc,
        &stage4_ram_addresses,
        &stage5_rd_write_addresses,
        &mut stage4_prefix,
    );
    let stage6_witness = stage6_witness_polynomials(fixture, &core_stage6.opening_inputs);
    let stage6_artifacts = run_bolt_stage6_prover(
        context.stage6_prover_plan,
        fixture,
        &core_stage6.opening_inputs,
        &stage6_witness,
        &mut stage4_prefix,
    );
    let stage7_artifacts = run_bolt_stage7_prover(
        context.stage7_prover_plan,
        &core_stage7.opening_inputs,
        &stage6_witness,
        &mut stage4_prefix,
    );
    let oracle_data = real_oracle_data(&context.commitment_prover_program, &fixture.cycle_inputs);
    let commitment_artifacts = prover_commitment_artifacts(
        &context.commitment_prover_program,
        &context.commitment_prover_trace,
    );

    Stage8Prefix {
        transcript: stage4_prefix,
        commitment_artifacts,
        stage6_artifacts,
        stage7_artifacts,
        stage7_opening_inputs: core_stage7.opening_inputs,
        oracle_data,
    }
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

fn run_bolt_stage3_prover<T>(
    plan: &'static KernelStage3CpuProgramPlan,
    fixture: &CoreStage1Fixture,
    opening_inputs: &[Stage3OpeningInputValue<Fr>],
    mut transcript: T,
) -> (T, Stage3ExecutionArtifacts<Fr>)
where
    T: Transcript<Challenge = Fr>,
{
    let inputs = Stage3ProverInputs::new(opening_inputs).with_cycles(&fixture.stage3_cycles);
    let mut prover = Stage3ProverKernelExecutor::new(inputs);
    let artifacts = execute_stage3_program(
        plan,
        Stage3ExecutionMode::Prover,
        &mut prover,
        &mut transcript,
    )
    .expect("Bolt Stage 3 prover succeeds");
    (transcript, artifacts)
}

fn run_bolt_stage4_prover<T>(
    plan: &'static KernelStage4CpuProgramPlan,
    fixture: &CoreStage1Fixture,
    opening_inputs: &[Stage4OpeningInputValue<Fr>],
    rd_inc: &[Fr],
    ram_addresses: &[Option<usize>],
    ram_inc: &[Fr],
    transcript: &mut T,
) -> Stage4ExecutionArtifacts<Fr>
where
    T: Transcript<Challenge = Fr>,
{
    let registers = Stage4RegistersWitness {
        register_count: 1 << fixture.params.register_log_k,
        trace_len: fixture.proof.trace_length,
        registers_val: &[],
        rs1_ra: &[],
        rs2_ra: &[],
        rd_wa: &[],
        accesses: Some(&fixture.stage4_register_accesses),
        rd_inc,
    };
    let ram = Stage4RamWitness {
        ram_k: fixture.proof.ram_K,
        trace_len: fixture.proof.trace_length,
        ram_ra: &[],
        write_address_indices: Some(ram_addresses),
        ram_inc,
    };
    let inputs = Stage4ProverInputs::new(opening_inputs)
        .with_registers(registers)
        .with_ram(ram);
    let mut prover = Stage4ProverKernelExecutor::new(inputs);
    execute_stage4_program(plan, Stage4ExecutionMode::Prover, &mut prover, transcript)
        .expect("Bolt Stage 4 prover succeeds")
}

fn run_bolt_stage5_prover<T>(
    plan: &'static kernel_stage5::Stage5CpuProgramPlan,
    fixture: &CoreStage1Fixture,
    opening_inputs: &[kernel_stage5::Stage5OpeningInputValue<Fr>],
    rd_inc: &[Fr],
    ram_addresses: &[Option<usize>],
    rd_write_addresses: &[Option<usize>],
    transcript: &mut T,
) -> kernel_stage5::Stage5ExecutionArtifacts<Fr>
where
    T: Transcript<Challenge = Fr>,
{
    let inputs = kernel_stage5::Stage5ProverInputs::new(opening_inputs)
        .with_instruction_read_raf(kernel_stage5::Stage5InstructionReadRafWitness {
            trace_len: fixture.proof.trace_length,
            lookup_indices: &fixture.stage5_lookup_indices,
            lookup_table_indices: &fixture.stage5_lookup_table_indices,
            is_interleaved_operands: &fixture.stage5_is_interleaved_operands,
            ra_virtual_log_k_chunk: fixture.params.lookups_ra_virtual_log_k_chunk,
        })
        .with_ram_ra(kernel_stage5::Stage5RamRaWitness {
            ram_k: fixture.proof.ram_K,
            trace_len: fixture.proof.trace_length,
            ram_ra: &[],
            remapped_addresses: Some(ram_addresses),
        })
        .with_registers_val(kernel_stage5::Stage5RegistersValWitness {
            register_count: 1 << fixture.params.register_log_k,
            trace_len: fixture.proof.trace_length,
            rd_inc,
            rd_wa: &[],
            rd_write_addresses: Some(rd_write_addresses),
        });
    let mut prover = kernel_stage5::Stage5ProverKernelExecutor::new(inputs);
    kernel_stage5::execute_stage5_program(
        plan,
        kernel_stage5::Stage5ExecutionMode::Prover,
        &mut prover,
        transcript,
    )
    .expect("Bolt Stage 5 prover succeeds")
}

fn run_bolt_stage6_prover<T>(
    plan: &'static kernel_stage6::Stage6CpuProgramPlan,
    fixture: &CoreStage1Fixture,
    opening_inputs: &[kernel_stage6::Stage6OpeningInputValue<Fr>],
    witness: &Stage6WitnessPolynomials,
    transcript: &mut T,
) -> kernel_stage6::Stage6ExecutionArtifacts<Fr>
where
    T: Transcript<Challenge = Fr>,
{
    let mut booleanity_chunks = Vec::new();
    booleanity_chunks.extend(witness.instruction_ra_booleanity.iter().map(Vec::as_slice));
    booleanity_chunks.extend(witness.bytecode_ra_booleanity.iter().map(Vec::as_slice));
    booleanity_chunks.extend(witness.ram_ra_booleanity.iter().map(Vec::as_slice));
    let bytecode_ra_chunks = witness
        .bytecode_ra_read_raf
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let ram_ra_chunks = witness
        .ram_ra_virtual
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let instruction_ra_chunks = witness
        .instruction_ra_virtual
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let inputs = kernel_stage6::Stage6ProverInputs::new(opening_inputs)
        .with_bytecode_read_raf(kernel_stage6::Stage6BytecodeReadRafWitness {
            data: stage6_bytecode_data(fixture),
            bytecode_ra_chunks: &bytecode_ra_chunks,
        })
        .with_booleanity(kernel_stage6::Stage6BooleanityWitness {
            chunks: &booleanity_chunks,
        })
        .with_hamming_booleanity(kernel_stage6::Stage6HammingBooleanityWitness {
            hamming_weight: &witness.hamming_weight,
        })
        .with_ram_ra_virtual(kernel_stage6::Stage6RamRaVirtualWitness {
            ram_ra_chunks: &ram_ra_chunks,
        })
        .with_instruction_ra_virtual(kernel_stage6::Stage6InstructionRaVirtualWitness {
            instruction_ra_chunks: &instruction_ra_chunks,
            virtual_count: fixture.params.instruction_ra_virtual_d,
        })
        .with_inc_claim_reduction(kernel_stage6::Stage6IncClaimReductionWitness {
            ram_inc: &witness.ram_inc,
            rd_inc: &witness.rd_inc,
        });
    let mut prover = kernel_stage6::Stage6ProverKernelExecutor::new(inputs);
    kernel_stage6::execute_stage6_program(
        plan,
        kernel_stage6::Stage6ExecutionMode::Prover,
        &mut prover,
        transcript,
    )
    .expect("Bolt Stage 6 prover succeeds")
}

fn run_bolt_stage7_prover<T>(
    plan: &'static kernel_stage7::Stage7CpuProgramPlan,
    opening_inputs: &[kernel_stage7::Stage7OpeningInputValue<Fr>],
    witness: &Stage6WitnessPolynomials,
    transcript: &mut T,
) -> kernel_stage7::Stage7ExecutionArtifacts<Fr>
where
    T: Transcript<Challenge = Fr>,
{
    let instruction_ra_chunks = witness
        .instruction_ra_booleanity
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let bytecode_ra_chunks = witness
        .bytecode_ra_booleanity
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let ram_ra_chunks = witness
        .ram_ra_booleanity
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let instruction_ra_indices = witness
        .instruction_ra_indices
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let bytecode_ra_indices = witness
        .bytecode_ra_indices
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let ram_ra_indices = witness
        .ram_ra_indices
        .iter()
        .map(Vec::as_slice)
        .collect::<Vec<_>>();
    let inputs = kernel_stage7::Stage7ProverInputs::new(opening_inputs)
        .with_hamming_weight_claim_reduction(
            kernel_stage7::Stage7HammingWeightClaimReductionWitness {
                instruction_ra: kernel_stage7::Stage7RaChunks {
                    chunks: &instruction_ra_chunks,
                    layout: kernel_stage7::Stage7RaChunkLayout::CycleMajor,
                },
                bytecode_ra: kernel_stage7::Stage7RaChunks {
                    chunks: &bytecode_ra_chunks,
                    layout: kernel_stage7::Stage7RaChunkLayout::CycleMajor,
                },
                ram_ra: kernel_stage7::Stage7RaChunks {
                    chunks: &ram_ra_chunks,
                    layout: kernel_stage7::Stage7RaChunkLayout::CycleMajor,
                },
            },
        )
        .with_hamming_weight_claim_reduction_indices(
            kernel_stage7::Stage7HammingWeightClaimReductionIndexWitness {
                instruction_ra: kernel_stage7::Stage7RaIndexChunks {
                    chunks: &instruction_ra_indices,
                },
                bytecode_ra: kernel_stage7::Stage7RaIndexChunks {
                    chunks: &bytecode_ra_indices,
                },
                ram_ra: kernel_stage7::Stage7RaIndexChunks {
                    chunks: &ram_ra_indices,
                },
            },
        );
    let mut prover = kernel_stage7::Stage7ProverKernelExecutor::new(inputs);
    kernel_stage7::execute_stage7_program(
        plan,
        kernel_stage7::Stage7ExecutionMode::Prover,
        &mut prover,
        transcript,
    )
    .expect("Bolt Stage 7 prover succeeds")
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

fn stage3_opening_inputs(
    stage1_artifacts: &Stage1ExecutionArtifacts<Fr>,
    stage2_artifacts: &Stage2ExecutionArtifacts<Fr>,
) -> Vec<Stage3OpeningInputValue<Fr>> {
    let mut inputs = Vec::new();
    let trace_point_arity = stage1_artifacts
        .opening_value("stage1.outer_remaining.opening.NextUnexpandedPC")
        .expect("Stage 1 NextUnexpandedPC opening")
        .point
        .len();
    for (oracle, symbol) in [
        ("NextUnexpandedPC", "stage3.input.stage1.NextUnexpandedPC"),
        ("NextPC", "stage3.input.stage1.NextPC"),
        ("NextIsVirtual", "stage3.input.stage1.NextIsVirtual"),
        (
            "NextIsFirstInSequence",
            "stage3.input.stage1.NextIsFirstInSequence",
        ),
        ("RdWriteValue", "stage3.input.stage1.RdWriteValue"),
        ("Rs1Value", "stage3.input.stage1.Rs1Value"),
        ("Rs2Value", "stage3.input.stage1.Rs2Value"),
    ] {
        let source_claim = format!("stage1.outer_remaining.opening.{oracle}");
        let opening = stage1_artifacts
            .opening_value(&source_claim)
            .unwrap_or_else(|| panic!("missing Stage 1 opening {source_claim}"));
        inputs.push(Stage3OpeningInputValue {
            symbol,
            point: opening.point.clone(),
            eval: opening.eval,
        });
    }

    for (eval_name, symbol) in [
        (
            "stage2.product_virtual.remainder.eval.NextIsNoop",
            "stage3.input.stage2.product_virtual.NextIsNoop",
        ),
        (
            "stage2.product_virtual.remainder.eval.LeftInstructionInput",
            "stage3.input.stage2.product_virtual.LeftInstructionInput",
        ),
        (
            "stage2.product_virtual.remainder.eval.RightInstructionInput",
            "stage3.input.stage2.product_virtual.RightInstructionInput",
        ),
        (
            "stage2.instruction_lookup.claim_reduction.eval.LeftInstructionInput",
            "stage3.input.stage2.instruction_lookup.LeftInstructionInput",
        ),
        (
            "stage2.instruction_lookup.claim_reduction.eval.RightInstructionInput",
            "stage3.input.stage2.instruction_lookup.RightInstructionInput",
        ),
    ] {
        inputs.push(stage3_stage2_opening_input(
            stage2_artifacts,
            eval_name,
            symbol,
            trace_point_arity,
        ));
    }
    inputs
}

fn stage3_stage2_opening_input(
    artifacts: &Stage2ExecutionArtifacts<Fr>,
    eval_name: &'static str,
    symbol: &'static str,
    trace_point_arity: usize,
) -> Stage3OpeningInputValue<Fr> {
    let output = artifacts
        .sumchecks
        .iter()
        .find(|output| output.evals.iter().any(|eval| eval.name == eval_name))
        .unwrap_or_else(|| panic!("missing Stage 2 output for {eval_name}"));
    let eval = output
        .evals
        .iter()
        .find(|eval| eval.name == eval_name)
        .unwrap_or_else(|| panic!("missing Stage 2 eval {eval_name}"));
    let point_start = output
        .point
        .len()
        .checked_sub(trace_point_arity)
        .unwrap_or_else(|| panic!("Stage 2 point is shorter than trace point for {eval_name}"));
    let mut point = output.point[point_start..].to_vec();
    point.reverse();
    Stage3OpeningInputValue {
        symbol,
        point,
        eval: eval.value,
    }
}

fn stage4_opening_inputs(
    params: &JoltProtocolParams,
    initial_ram_state: &[u64],
    stage2_artifacts: &Stage2ExecutionArtifacts<Fr>,
    stage3_artifacts: &Stage3ExecutionArtifacts<Fr>,
) -> Vec<Stage4OpeningInputValue<Fr>> {
    let stage2 = stage2_artifacts
        .sumchecks
        .iter()
        .find(|output| output.driver == "stage2.sumcheck")
        .expect("Stage 2 batched output");
    let stage3 = stage3_artifacts
        .sumchecks
        .iter()
        .find(|output| output.driver == "stage3.sumcheck")
        .expect("Stage 3 batched output");
    let stage3_point = reverse_point(&stage3.point);
    let ram_val_point = reverse_point(&stage2.point);
    let ram_final_point = reversed_suffix(&stage2.point, params.log_k_ram);
    let mut inputs = Vec::new();
    for (eval_name, symbol) in [
        (
            "stage3.registers_claim_reduction.eval.RdWriteValue",
            "stage4.input.stage3.registers.RdWriteValue",
        ),
        (
            "stage3.registers_claim_reduction.eval.Rs1Value",
            "stage4.input.stage3.registers.Rs1Value",
        ),
        (
            "stage3.registers_claim_reduction.eval.Rs2Value",
            "stage4.input.stage3.registers.Rs2Value",
        ),
        (
            "stage3.instruction_input.eval.Rs1Value",
            "stage4.input.stage3.instruction.Rs1Value",
        ),
        (
            "stage3.instruction_input.eval.Rs2Value",
            "stage4.input.stage3.instruction.Rs2Value",
        ),
    ] {
        inputs.push(Stage4OpeningInputValue {
            symbol,
            point: stage3_point.clone(),
            eval: stage_eval_stage3(stage3, eval_name),
        });
    }
    inputs.push(Stage4OpeningInputValue {
        symbol: "stage4.input.stage2.RamVal",
        point: ram_val_point,
        eval: stage_eval_stage2(stage2, "stage2.ram_read_write.eval.RamVal"),
    });
    inputs.push(Stage4OpeningInputValue {
        symbol: "stage4.input.stage2.RamValFinal",
        point: ram_final_point.clone(),
        eval: stage_eval_stage2(stage2, "stage2.ram_output.eval.RamValFinal"),
    });
    inputs.push(Stage4OpeningInputValue {
        symbol: "stage4.input.initial_ram.RamValInit",
        point: ram_final_point.clone(),
        eval: mle_eval_u64(initial_ram_state, &ram_final_point),
    });
    inputs
}

fn stage5_opening_inputs(
    params: &JoltProtocolParams,
    stage2_inputs: &[Stage2OpeningInputValue<Fr>],
    stage2_artifacts: &Stage2ExecutionArtifacts<Fr>,
    stage4_artifacts: &Stage4ExecutionArtifacts<Fr>,
) -> Vec<kernel_stage5::Stage5OpeningInputValue<Fr>> {
    let stage2 = stage2_artifacts
        .sumchecks
        .iter()
        .find(|output| output.driver == "stage2.sumcheck")
        .expect("Stage 2 batched output");
    let stage4 = stage4_artifacts
        .sumchecks
        .iter()
        .find(|output| output.driver == "stage4.sumcheck")
        .expect("Stage 4 batched output");

    let instruction_point = reversed_suffix(&stage2.point, params.log_t);
    let stage2_ram_rw_point = reverse_point(&stage2.point);
    let stage2_ram_raf_address = reversed_suffix(&stage2.point, params.log_k_ram);
    let stage2_ram_address_input = stage2_inputs
        .iter()
        .find(|input| input.symbol == "stage2.input.stage1.RamAddress")
        .expect("Stage 2 RamAddress input");
    let mut stage2_ram_raf_point = stage2_ram_raf_address;
    stage2_ram_raf_point.extend_from_slice(&stage2_ram_address_input.point);

    let stage4_ram_address = stage2_ram_rw_point[..params.log_k_ram].to_vec();
    let mut stage4_ram_val_check_point = stage4_ram_address;
    stage4_ram_val_check_point.extend(reversed_suffix(&stage4.point, params.log_t));
    let stage4_registers_point = normalized_stage4_registers_rw_point(params, &stage4.point);

    vec![
        kernel_stage5::Stage5OpeningInputValue {
            symbol: "stage5.input.stage2.instruction.LookupOutput",
            point: instruction_point.clone(),
            eval: stage_eval_stage2(
                stage2,
                "stage2.instruction_lookup.claim_reduction.eval.LookupOutput",
            ),
        },
        kernel_stage5::Stage5OpeningInputValue {
            symbol: "stage5.input.stage2.product_virtual.LookupOutput",
            point: instruction_point.clone(),
            eval: stage_eval_stage2(stage2, "stage2.product_virtual.remainder.eval.LookupOutput"),
        },
        kernel_stage5::Stage5OpeningInputValue {
            symbol: "stage5.input.stage2.instruction.LeftLookupOperand",
            point: instruction_point.clone(),
            eval: stage_eval_stage2(
                stage2,
                "stage2.instruction_lookup.claim_reduction.eval.LeftLookupOperand",
            ),
        },
        kernel_stage5::Stage5OpeningInputValue {
            symbol: "stage5.input.stage2.instruction.RightLookupOperand",
            point: instruction_point,
            eval: stage_eval_stage2(
                stage2,
                "stage2.instruction_lookup.claim_reduction.eval.RightLookupOperand",
            ),
        },
        kernel_stage5::Stage5OpeningInputValue {
            symbol: "stage5.input.stage2.ram_raf.RamRa",
            point: stage2_ram_raf_point,
            eval: stage_eval_stage2(stage2, "stage2.ram_raf.eval.RamRa"),
        },
        kernel_stage5::Stage5OpeningInputValue {
            symbol: "stage5.input.stage2.ram_read_write.RamRa",
            point: stage2_ram_rw_point,
            eval: stage_eval_stage2(stage2, "stage2.ram_read_write.eval.RamRa"),
        },
        kernel_stage5::Stage5OpeningInputValue {
            symbol: "stage5.input.stage4.ram_val_check.RamRa",
            point: stage4_ram_val_check_point,
            eval: stage_eval_stage4(stage4, "stage4.ram_val_check.eval.RamRa"),
        },
        kernel_stage5::Stage5OpeningInputValue {
            symbol: "stage5.input.stage4.registers.RegistersVal",
            point: stage4_registers_point,
            eval: stage_eval_stage4(stage4, "stage4.registers_read_write.eval.RegistersVal"),
        },
    ]
}

fn stage_eval_stage2(
    output: &jolt_kernels::stage2::Stage2SumcheckOutput<Fr>,
    name: &'static str,
) -> Fr {
    output
        .evals
        .iter()
        .find(|eval| eval.name == name)
        .unwrap_or_else(|| panic!("missing eval {name}"))
        .value
}

fn stage_eval_stage3(
    output: &jolt_kernels::stage3::Stage3SumcheckOutput<Fr>,
    name: &'static str,
) -> Fr {
    output
        .evals
        .iter()
        .find(|eval| eval.name == name)
        .unwrap_or_else(|| panic!("missing eval {name}"))
        .value
}

fn stage_eval_stage4(
    output: &jolt_kernels::stage4::Stage4SumcheckOutput<Fr>,
    name: &'static str,
) -> Fr {
    output
        .evals
        .iter()
        .find(|eval| eval.name == name)
        .unwrap_or_else(|| panic!("missing eval {name}"))
        .value
}

fn reverse_point(point: &[Fr]) -> Vec<Fr> {
    point.iter().rev().copied().collect()
}

fn reversed_suffix(point: &[Fr], len: usize) -> Vec<Fr> {
    let start = point
        .len()
        .checked_sub(len)
        .unwrap_or_else(|| panic!("point is shorter than suffix length {len}"));
    point[start..].iter().rev().copied().collect()
}

fn normalized_stage4_registers_rw_point(params: &JoltProtocolParams, point: &[Fr]) -> Vec<Fr> {
    let expected = params.log_t + params.register_log_k;
    assert_eq!(
        point.len(),
        expected,
        "Stage 4 registers point length mismatch"
    );
    let (cycle, address) = point.split_at(params.log_t);
    address
        .iter()
        .rev()
        .copied()
        .chain(cycle.iter().rev().copied())
        .collect()
}

fn stage4_rd_inc(accesses: &[Stage4RegisterAccess]) -> Vec<Fr> {
    accesses
        .iter()
        .map(|access| {
            access.rd.map_or(Fr::from_u64(0), |rd| {
                field_delta_u64(rd.post_value, rd.pre_value)
            })
        })
        .collect()
}

fn stage4_rd_write_addresses(accesses: &[Stage4RegisterAccess]) -> Vec<Option<usize>> {
    accesses
        .iter()
        .map(|access| access.rd.map(|rd| rd.address))
        .collect()
}

fn stage4_ram_address_indices(accesses: &[Stage2RamAccess]) -> Vec<Option<usize>> {
    accesses
        .iter()
        .map(|access| access.remapped_address)
        .collect()
}

fn stage2_ram_inc(accesses: &[Stage2RamAccess]) -> Vec<Fr> {
    accesses
        .iter()
        .map(|access| field_delta_u64(access.write_value, access.read_value))
        .collect()
}

fn stage6_bytecode_data(
    fixture: &CoreStage1Fixture,
) -> kernel_stage6::Stage6BytecodeReadRafData<'_, Fr> {
    kernel_stage6::Stage6BytecodeReadRafData {
        entries: &fixture.stage6_bytecode_entries,
        entry_bytecode_index: fixture.stage6_entry_bytecode_index,
        num_lookup_tables: CoreLookupTables::<XLEN>::COUNT,
    }
}

fn stage6_bytecode_entries(
    bytecode: &BytecodePreprocessing,
) -> Vec<kernel_stage6::Stage6BytecodeEntry<Fr>> {
    bytecode
        .bytecode
        .iter()
        .map(|instruction| {
            let instr = instruction.normalize();
            let circuit_flags = instruction.circuit_flags();
            let instruction_flags = instruction.instruction_flags();
            let lookup_table = InstructionLookup::<XLEN>::lookup_table(instruction)
                .map(|table| CoreLookupTables::<XLEN>::enum_index(&table));
            kernel_stage6::Stage6BytecodeEntry {
                address: Fr::from_u64(instr.address as u64),
                imm: Fr::from_i128(instr.operands.imm),
                circuit_flags: stage6_circuit_flags(circuit_flags),
                rd: instr.operands.rd.map(usize::from),
                rs1: instr.operands.rs1.map(usize::from),
                rs2: instr.operands.rs2.map(usize::from),
                lookup_table,
                is_interleaved: circuit_flags.is_interleaved_operands(),
                is_branch: instruction_flags[CoreInstructionFlags::Branch],
                left_is_rs1: instruction_flags[CoreInstructionFlags::LeftOperandIsRs1Value],
                left_is_pc: instruction_flags[CoreInstructionFlags::LeftOperandIsPC],
                right_is_rs2: instruction_flags[CoreInstructionFlags::RightOperandIsRs2Value],
                right_is_imm: instruction_flags[CoreInstructionFlags::RightOperandIsImm],
                is_noop: instruction_flags[CoreInstructionFlags::IsNoop],
            }
        })
        .collect()
}

fn stage6_circuit_flags(flags: [bool; 14]) -> [bool; 14] {
    [
        flags[CoreCircuitFlags::AddOperands],
        flags[CoreCircuitFlags::SubtractOperands],
        flags[CoreCircuitFlags::MultiplyOperands],
        flags[CoreCircuitFlags::Load],
        flags[CoreCircuitFlags::Store],
        flags[CoreCircuitFlags::Jump],
        flags[CoreCircuitFlags::WriteLookupOutputToRD],
        flags[CoreCircuitFlags::VirtualInstruction],
        flags[CoreCircuitFlags::Assert],
        flags[CoreCircuitFlags::DoNotUpdateUnexpandedPC],
        flags[CoreCircuitFlags::Advice],
        flags[CoreCircuitFlags::IsCompressed],
        flags[CoreCircuitFlags::IsFirstInSequence],
        flags[CoreCircuitFlags::IsLastInSequence],
    ]
}

fn stage6_witness_polynomials(
    fixture: &CoreStage1Fixture,
    opening_inputs: &[kernel_stage6::Stage6OpeningInputValue<Fr>],
) -> Stage6WitnessPolynomials {
    let one_hot_params = jolt_core::zkvm::config::OneHotParams::from_config(
        &fixture.proof.one_hot_config,
        fixture.verifier_preprocessing.shared.bytecode.code_size,
        fixture.proof.ram_K,
    );
    let bytecode = fixture.verifier_preprocessing.shared.bytecode.as_ref();
    let memory_layout = &fixture.verifier_preprocessing.shared.memory_layout;
    let trace = fixture.padded_trace.as_slice();
    let trace_len = fixture.proof.trace_length;
    assert_eq!(trace.len(), trace_len, "padded trace length mismatch");

    let instruction_indices = (0..fixture.params.instruction_d)
        .map(|index| instruction_ra_chunk_indices(trace, &one_hot_params, index))
        .collect::<Vec<_>>();
    let bytecode_indices = (0..fixture.params.bytecode_d)
        .map(|index| bytecode_ra_chunk_indices(trace, bytecode, &one_hot_params, index))
        .collect::<Vec<_>>();
    let ram_indices = (0..fixture.params.ram_d)
        .map(|index| ram_ra_chunk_indices(trace, memory_layout, &one_hot_params, index))
        .collect::<Vec<_>>();

    let instruction_ra_booleanity = instruction_indices
        .iter()
        .map(|indices| one_hot_cycle_major_from_indices(indices, fixture.params.log_k_chunk))
        .collect::<Vec<_>>();
    let bytecode_ra_booleanity = bytecode_indices
        .iter()
        .map(|indices| one_hot_cycle_major_from_indices(indices, fixture.params.log_k_chunk))
        .collect::<Vec<_>>();
    let ram_ra_booleanity = ram_indices
        .iter()
        .map(|indices| one_hot_cycle_major_from_indices(indices, fixture.params.log_k_chunk))
        .collect::<Vec<_>>();

    let bytecode_ra_read_raf = bytecode_indices
        .iter()
        .zip(stage6_bytecode_chunk_lens(&fixture.params))
        .map(|(indices, chunk_len)| one_hot_address_major_from_indices(indices, chunk_len))
        .collect::<Vec<_>>();

    let ram_address_chunks = stage6_ram_virtual_address_chunks(&fixture.params, opening_inputs);
    assert_eq!(
        ram_address_chunks.len(),
        fixture.params.ram_d,
        "RAM Stage 6 address chunk count mismatch"
    );
    let ram_ra_virtual = ram_indices
        .iter()
        .zip(&ram_address_chunks)
        .map(|(indices, point)| one_hot_evals_at_chunk_point(indices, point))
        .collect::<Vec<_>>();

    let instruction_address_chunks =
        stage6_instruction_virtual_address_chunks(&fixture.params, opening_inputs);
    assert_eq!(
        instruction_address_chunks.len(),
        fixture.params.instruction_d,
        "instruction Stage 6 address chunk count mismatch"
    );
    let instruction_ra_virtual = instruction_indices
        .iter()
        .zip(&instruction_address_chunks)
        .map(|(indices, point)| one_hot_evals_at_chunk_point(indices, point))
        .collect::<Vec<_>>();

    let hamming_weight = trace
        .iter()
        .map(|cycle| {
            if cycle.ram_access().address() != 0 {
                Fr::from_u64(1)
            } else {
                Fr::from_u64(0)
            }
        })
        .collect();

    Stage6WitnessPolynomials {
        instruction_ra_indices: instruction_indices,
        bytecode_ra_indices: bytecode_indices,
        ram_ra_indices: ram_indices,
        instruction_ra_booleanity,
        bytecode_ra_booleanity,
        ram_ra_booleanity,
        bytecode_ra_read_raf,
        instruction_ra_virtual,
        ram_ra_virtual,
        hamming_weight,
        ram_inc: stage2_ram_inc(&fixture.ram_accesses),
        rd_inc: stage4_rd_inc(&fixture.stage4_register_accesses),
    }
}

fn instruction_ra_chunk_indices(
    trace: &[jolt_host::Cycle],
    one_hot_params: &jolt_core::zkvm::config::OneHotParams,
    chunk: usize,
) -> Vec<Option<u8>> {
    trace
        .iter()
        .map(|cycle| {
            let lookup_index = LookupQuery::<XLEN>::to_lookup_index(cycle);
            Some(one_hot_params.lookup_index_chunk(lookup_index, chunk))
        })
        .collect()
}

fn bytecode_ra_chunk_indices(
    trace: &[jolt_host::Cycle],
    bytecode: &jolt_core::zkvm::bytecode::BytecodePreprocessing,
    one_hot_params: &jolt_core::zkvm::config::OneHotParams,
    chunk: usize,
) -> Vec<Option<u8>> {
    trace
        .iter()
        .map(|cycle| {
            let pc = bytecode.get_pc(cycle);
            Some(one_hot_params.bytecode_pc_chunk(pc, chunk))
        })
        .collect()
}

fn ram_ra_chunk_indices(
    trace: &[jolt_host::Cycle],
    memory_layout: &common::jolt_device::MemoryLayout,
    one_hot_params: &jolt_core::zkvm::config::OneHotParams,
    chunk: usize,
) -> Vec<Option<u8>> {
    trace
        .iter()
        .map(|cycle| {
            remap_address(cycle.ram_access().address() as u64, memory_layout)
                .map(|address| one_hot_params.ram_address_chunk(address, chunk))
        })
        .collect()
}

fn one_hot_cycle_major_from_indices(indices: &[Option<u8>], chunk_len: usize) -> Vec<Fr> {
    let chunk_domain = 1usize << chunk_len;
    let mut output = vec![Fr::from_u64(0); chunk_domain * indices.len()];
    for (cycle, index) in indices.iter().enumerate() {
        if let Some(index) = index {
            let index = usize::from(*index);
            assert!(
                index < chunk_domain,
                "one-hot index {index} exceeds chunk domain {chunk_domain}"
            );
            output[cycle * chunk_domain + index] = Fr::from_u64(1);
        }
    }
    output
}

fn one_hot_address_major_from_indices(indices: &[Option<u8>], chunk_len: usize) -> Vec<Fr> {
    let chunk_domain = 1usize << chunk_len;
    let mut output = vec![Fr::from_u64(0); chunk_domain * indices.len()];
    for (cycle, index) in indices.iter().enumerate() {
        if let Some(index) = index {
            let index = usize::from(*index);
            assert!(
                index < chunk_domain,
                "one-hot index {index} exceeds chunk domain {chunk_domain}"
            );
            output[index * indices.len() + cycle] = Fr::from_u64(1);
        }
    }
    output
}

fn one_hot_evals_at_chunk_point(indices: &[Option<u8>], point: &[Fr]) -> Vec<Fr> {
    let eq_table = EqPolynomial::<Fr>::evals(point, None);
    indices
        .iter()
        .map(|index| {
            index.map_or(Fr::from_u64(0), |index| {
                eq_table
                    .get(usize::from(index))
                    .copied()
                    .expect("one-hot index is inside chunk point domain")
            })
        })
        .collect()
}

fn stage6_bytecode_chunk_lens(params: &JoltProtocolParams) -> Vec<usize> {
    let first_chunk_len = params.log_k_bytecode % params.log_k_chunk;
    let mut chunk_lens = Vec::with_capacity(params.bytecode_d);
    if first_chunk_len != 0 {
        chunk_lens.push(first_chunk_len);
    }
    while chunk_lens.len() < params.bytecode_d {
        chunk_lens.push(params.log_k_chunk);
    }
    chunk_lens
}

fn stage6_ram_virtual_address_chunks(
    params: &JoltProtocolParams,
    opening_inputs: &[kernel_stage6::Stage6OpeningInputValue<Fr>],
) -> Vec<Vec<Fr>> {
    let point = stage6_opening_point(
        opening_inputs,
        "stage6.input.stage5.ram_ra_claim_reduction.RamRa",
    );
    assert!(
        point.len() >= params.log_k_ram,
        "RAM RA opening point is shorter than the RAM address arity"
    );
    address_chunks_from_point(&point[..params.log_k_ram], params.log_k_chunk)
}

fn stage6_instruction_virtual_address_chunks(
    params: &JoltProtocolParams,
    opening_inputs: &[kernel_stage6::Stage6OpeningInputValue<Fr>],
) -> Vec<Vec<Fr>> {
    let mut address = Vec::with_capacity(params.instruction_log_k);
    for index in 0..params.instruction_ra_virtual_d {
        let symbol = format!("stage6.input.stage5.instruction_read_raf.InstructionRa_{index}");
        let point = stage6_opening_point(opening_inputs, &symbol);
        assert!(
            point.len() >= params.lookups_ra_virtual_log_k_chunk,
            "instruction RA opening point is shorter than the virtual address chunk arity"
        );
        address.extend_from_slice(&point[..params.lookups_ra_virtual_log_k_chunk]);
    }
    address_chunks_from_point(&address, params.log_k_chunk)
}

fn address_chunks_from_point(address: &[Fr], log_k_chunk: usize) -> Vec<Vec<Fr>> {
    let mut padded: Vec<Fr> = Vec::new();
    let remainder = address.len() % log_k_chunk;
    if remainder != 0 {
        padded.resize(log_k_chunk - remainder, Fr::from_u64(0));
    }
    padded.extend_from_slice(address);
    padded
        .chunks(log_k_chunk)
        .map(|chunk| chunk.to_vec())
        .collect()
}

fn stage6_opening_point<'a>(
    opening_inputs: &'a [kernel_stage6::Stage6OpeningInputValue<Fr>],
    symbol: &str,
) -> &'a [Fr] {
    opening_inputs
        .iter()
        .find(|input| input.symbol == symbol)
        .map(|input| input.point.as_slice())
        .unwrap_or_else(|| panic!("missing Stage 6 opening input `{symbol}`"))
}

fn field_delta_u64(post: u64, pre: u64) -> Fr {
    if post >= pre {
        Fr::from_u64(post - pre)
    } else {
        -Fr::from_u64(pre - post)
    }
}

fn mle_eval_u64(values: &[u64], point: &[Fr]) -> Fr {
    EqPolynomial::<Fr>::evals(point, None)
        .iter()
        .zip(values)
        .map(|(&weight, &value)| weight * Fr::from_u64(value))
        .sum()
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
    append_bolt_preamble(transcript, fixture);
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

fn bolt_stage3_programs_with_params(
    params: &JoltProtocolParams,
) -> (CompilerStage3CpuProgram, CompilerStage3CpuProgram) {
    let context = MeliorContext::new();
    let protocol = build_stage3_protocol(&context, params).expect("build stage3 protocol");
    let concrete = lower_piop_and_fiat_shamir(&context, &protocol).expect("lower Stage 3 protocol");
    let prover_party = project_prover_party(&context, &concrete).expect("project prover");
    let verifier_party = project_verifier_party(&context, &concrete).expect("project verifier");
    let prover_compute =
        lower_stage3_to_compute(&context, &prover_party).expect("lower prover Stage 3");
    let verifier_compute =
        lower_stage3_to_compute(&context, &verifier_party).expect("lower verifier Stage 3");
    let prover_compute =
        resolve_compute_kernels(&context, &prover_compute).expect("resolve prover kernels");
    let verifier_compute =
        resolve_compute_kernels(&context, &verifier_compute).expect("resolve verifier kernels");
    let prover_cpu = lower_compute_to_cpu(&context, &prover_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_compute).expect("lower verifier CPU");
    let prover_program = stage3_cpu_program(&prover_cpu).expect("extract prover Stage 3 CPU");
    let verifier_program = stage3_cpu_program(&verifier_cpu).expect("extract verifier Stage 3 CPU");
    (prover_program, verifier_program)
}

fn bolt_stage4_programs_with_params(
    params: &JoltProtocolParams,
) -> (CompilerStage4CpuProgram, CompilerStage4CpuProgram) {
    let context = MeliorContext::new();
    let protocol = build_stage4_protocol(&context, params).expect("build stage4 protocol");
    let concrete = lower_piop_and_fiat_shamir(&context, &protocol).expect("lower Stage 4 protocol");
    let prover_party = project_prover_party(&context, &concrete).expect("project prover");
    let verifier_party = project_verifier_party(&context, &concrete).expect("project verifier");
    let prover_compute =
        lower_stage4_to_compute(&context, &prover_party).expect("lower prover Stage 4");
    let verifier_compute =
        lower_stage4_to_compute(&context, &verifier_party).expect("lower verifier Stage 4");
    let prover_compute =
        resolve_compute_kernels(&context, &prover_compute).expect("resolve prover kernels");
    let verifier_compute =
        resolve_compute_kernels(&context, &verifier_compute).expect("resolve verifier kernels");
    let prover_cpu = lower_compute_to_cpu(&context, &prover_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_compute).expect("lower verifier CPU");
    let prover_program = stage4_cpu_program(&prover_cpu).expect("extract prover Stage 4 CPU");
    let verifier_program = stage4_cpu_program(&verifier_cpu).expect("extract verifier Stage 4 CPU");
    (prover_program, verifier_program)
}

fn bolt_stage5_programs_with_params(
    params: &JoltProtocolParams,
) -> (CompilerStage5CpuProgram, CompilerStage5CpuProgram) {
    let context = MeliorContext::new();
    let protocol = build_stage5_protocol(&context, params).expect("build stage5 protocol");
    let concrete = lower_piop_and_fiat_shamir(&context, &protocol).expect("lower Stage 5 protocol");
    let prover_party = project_prover_party(&context, &concrete).expect("project prover");
    let verifier_party = project_verifier_party(&context, &concrete).expect("project verifier");
    let prover_compute =
        lower_stage5_to_compute(&context, &prover_party).expect("lower prover Stage 5");
    let verifier_compute =
        lower_stage5_to_compute(&context, &verifier_party).expect("lower verifier Stage 5");
    let prover_compute =
        resolve_compute_kernels(&context, &prover_compute).expect("resolve prover kernels");
    let verifier_compute =
        resolve_compute_kernels(&context, &verifier_compute).expect("resolve verifier kernels");
    let prover_cpu = lower_compute_to_cpu(&context, &prover_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_compute).expect("lower verifier CPU");
    let prover_program = stage5_cpu_program(&prover_cpu).expect("extract prover Stage 5 CPU");
    let verifier_program = stage5_cpu_program(&verifier_cpu).expect("extract verifier Stage 5 CPU");
    (prover_program, verifier_program)
}

fn bolt_stage6_programs_with_params(
    params: &JoltProtocolParams,
) -> (CompilerStage6CpuProgram, CompilerStage6CpuProgram) {
    let context = MeliorContext::new();
    let protocol = build_stage6_protocol(&context, params).expect("build stage6 protocol");
    let concrete = lower_piop_and_fiat_shamir(&context, &protocol).expect("lower Stage 6 protocol");
    let prover_party = project_prover_party(&context, &concrete).expect("project prover");
    let verifier_party = project_verifier_party(&context, &concrete).expect("project verifier");
    let prover_compute =
        lower_stage6_to_compute(&context, &prover_party).expect("lower prover Stage 6");
    let verifier_compute =
        lower_stage6_to_compute(&context, &verifier_party).expect("lower verifier Stage 6");
    let prover_compute =
        resolve_compute_kernels(&context, &prover_compute).expect("resolve prover kernels");
    let verifier_compute =
        resolve_compute_kernels(&context, &verifier_compute).expect("resolve verifier kernels");
    let prover_cpu = lower_compute_to_cpu(&context, &prover_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_compute).expect("lower verifier CPU");
    let prover_program = stage6_cpu_program(&prover_cpu).expect("extract prover Stage 6 CPU");
    let verifier_program = stage6_cpu_program(&verifier_cpu).expect("extract verifier Stage 6 CPU");
    (prover_program, verifier_program)
}

fn bolt_stage7_programs_with_params(
    params: &JoltProtocolParams,
) -> (CompilerStage7CpuProgram, CompilerStage7CpuProgram) {
    let context = MeliorContext::new();
    let protocol = build_stage7_protocol(&context, params).expect("build stage7 protocol");
    let concrete = lower_piop_and_fiat_shamir(&context, &protocol).expect("lower Stage 7 protocol");
    let prover_party = project_prover_party(&context, &concrete).expect("project prover");
    let verifier_party = project_verifier_party(&context, &concrete).expect("project verifier");
    let prover_compute =
        lower_stage7_to_compute(&context, &prover_party).expect("lower prover Stage 7");
    let verifier_compute =
        lower_stage7_to_compute(&context, &verifier_party).expect("lower verifier Stage 7");
    let prover_compute =
        resolve_compute_kernels(&context, &prover_compute).expect("resolve prover kernels");
    let verifier_compute =
        resolve_compute_kernels(&context, &verifier_compute).expect("resolve verifier kernels");
    let prover_cpu = lower_compute_to_cpu(&context, &prover_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_compute).expect("lower verifier CPU");
    let prover_program = stage7_cpu_program(&prover_cpu).expect("extract prover Stage 7 CPU");
    let verifier_program = stage7_cpu_program(&verifier_cpu).expect("extract verifier Stage 7 CPU");
    (prover_program, verifier_program)
}

fn bolt_stage8_programs_with_params(
    params: &JoltProtocolParams,
) -> (CompilerStage8CpuProgram, CompilerStage8CpuProgram) {
    let context = MeliorContext::new();
    let protocol = build_stage8_protocol(&context, params).expect("build stage8 protocol");
    let concrete = lower_piop_and_fiat_shamir(&context, &protocol).expect("lower Stage 8 protocol");
    let prover_party = project_prover_party(&context, &concrete).expect("project prover");
    let verifier_party = project_verifier_party(&context, &concrete).expect("project verifier");
    let prover_compute =
        lower_stage8_to_compute(&context, &prover_party).expect("lower prover Stage 8");
    let verifier_compute =
        lower_stage8_to_compute(&context, &verifier_party).expect("lower verifier Stage 8");
    let prover_cpu = lower_compute_to_cpu(&context, &prover_compute).expect("lower prover CPU");
    let verifier_cpu =
        lower_compute_to_cpu(&context, &verifier_compute).expect("lower verifier CPU");
    let prover_program = stage8_cpu_program(&prover_cpu).expect("extract prover Stage 8 CPU");
    let verifier_program = stage8_cpu_program(&verifier_cpu).expect("extract verifier Stage 8 CPU");
    (prover_program, verifier_program)
}

fn real_oracle_data(
    program: &CommitmentCpuProgram,
    cycle_inputs: &[CycleInput],
) -> BTreeMap<String, Option<Vec<Fr>>> {
    let mut data = BTreeMap::new();
    for plan in &program.oracle_plans {
        let Some(materialized) = materialize_oracle_plan(plan, cycle_inputs) else {
            continue;
        };
        let _ = data.insert(plan.oracle.clone(), materialized);
    }
    data
}

fn materialize_oracle_plan(
    plan: &OraclePlan,
    cycle_inputs: &[CycleInput],
) -> Option<Option<Vec<Fr>>> {
    let materialized = match &plan.generation {
        OracleGeneration::Reference => return None,
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
    Some(materialized)
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
    let mut hints = Vec::new();

    for plan in &program.batch_plans {
        assert_eq!(plan.count, plan.oracles.len());
        for oracle in &plan.oracles {
            let oracle_num_vars = oracle_num_vars(program, oracle, plan.num_vars);
            let data = materialize(oracle, oracle_num_vars)
                .unwrap_or_else(|| panic!("missing batch oracle `{oracle}`"));
            let data = into_padded_oracle(data, oracle_num_vars);
            let (commitment, hint) = commit_with_layout(&data, plan.num_vars, setup);
            records.push(CommitmentRecord {
                artifact: plan.artifact.clone(),
            });
            hints.push(jolt_prover::stages::commitment::OracleOpeningHint {
                oracle: leak_str(oracle),
                hint,
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
                let (commitment, hint) = commit_with_layout(&data, plan.num_vars, setup);
                hints.push(jolt_prover::stages::commitment::OracleOpeningHint {
                    oracle: leak_str(&plan.oracle),
                    hint,
                });
                commitment
            });
        records.push(CommitmentRecord {
            artifact: plan.artifact.clone(),
        });
        commitments.push(commitment);
    }

    BoltCommitmentTrace {
        commitments,
        records,
        hints,
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
        hints: Vec::new(),
    }
}

fn generated_commitment_artifacts(
    program: &CommitmentCpuProgram,
    trace: &BoltCommitmentTrace,
) -> jolt_verifier::stages::commitment::CommitmentArtifacts {
    let mut records = Vec::new();
    for plan in &program.batch_plans {
        for oracle in &plan.oracles {
            records.push(jolt_verifier::stages::commitment::CommitmentRecord {
                artifact: leak_str(&plan.artifact),
                oracle: leak_str(oracle),
                label: leak_str(&plan.label),
                num_vars: plan.num_vars,
            });
        }
    }
    for plan in &program.optional_plans {
        records.push(jolt_verifier::stages::commitment::CommitmentRecord {
            artifact: leak_str(&plan.artifact),
            oracle: leak_str(&plan.oracle),
            label: leak_str(&plan.label),
            num_vars: plan.num_vars,
        });
    }
    assert_eq!(
        records.len(),
        trace.commitments.len(),
        "commitment record/proof count mismatch"
    );
    jolt_verifier::stages::commitment::CommitmentArtifacts {
        commitments: trace.commitments.clone(),
        records,
    }
}

fn prover_commitment_artifacts(
    program: &CommitmentCpuProgram,
    trace: &BoltCommitmentTrace,
) -> jolt_prover::stages::commitment::CommitmentArtifacts {
    let mut records = Vec::new();
    for plan in &program.batch_plans {
        for oracle in &plan.oracles {
            records.push(jolt_prover::stages::commitment::CommitmentRecord {
                artifact: leak_str(&plan.artifact),
                oracle: leak_str(oracle),
                label: leak_str(&plan.label),
                num_vars: plan.num_vars,
            });
        }
    }
    for plan in &program.optional_plans {
        records.push(jolt_prover::stages::commitment::CommitmentRecord {
            artifact: leak_str(&plan.artifact),
            oracle: leak_str(&plan.oracle),
            label: leak_str(&plan.label),
            num_vars: plan.num_vars,
        });
    }
    assert_eq!(
        records.len(),
        trace.commitments.len(),
        "commitment record/proof count mismatch"
    );
    jolt_prover::stages::commitment::CommitmentArtifacts {
        commitments: trace.commitments.clone(),
        records,
        hints: trace.hints.clone(),
    }
}

fn jolt_stage6_proof(
    artifacts: &kernel_stage6::Stage6ExecutionArtifacts<Fr>,
) -> jolt_verifier::JoltStageProof {
    jolt_verifier::JoltStageProof {
        sumchecks: artifacts
            .sumchecks
            .iter()
            .map(|output| jolt_verifier::JoltSumcheckOutput {
                driver: output.driver,
                point: output.point.clone(),
                evals: output
                    .evals
                    .iter()
                    .map(|eval| jolt_verifier::JoltNamedEval {
                        name: eval.name,
                        oracle: eval.oracle,
                        value: eval.value,
                    })
                    .collect(),
                proof: output.proof.clone(),
            })
            .collect(),
    }
}

fn jolt_stage7_proof(
    artifacts: &kernel_stage7::Stage7ExecutionArtifacts<Fr>,
) -> jolt_verifier::JoltStageProof {
    jolt_verifier::JoltStageProof {
        sumchecks: artifacts
            .sumchecks
            .iter()
            .map(|output| jolt_verifier::JoltSumcheckOutput {
                driver: output.driver,
                point: output.point.clone(),
                evals: output
                    .evals
                    .iter()
                    .map(|eval| jolt_verifier::JoltNamedEval {
                        name: eval.name,
                        oracle: eval.oracle,
                        value: eval.value,
                    })
                    .collect(),
                proof: output.proof.clone(),
            })
            .collect(),
    }
}

fn verifier_stage7_opening_inputs(
    inputs: &[kernel_stage7::Stage7OpeningInputValue<Fr>],
) -> Vec<jolt_verifier::stages::stage7::Stage7OpeningInputValue<Fr>> {
    inputs
        .iter()
        .map(
            |input| jolt_verifier::stages::stage7::Stage7OpeningInputValue {
                symbol: input.symbol,
                point: input.point.clone(),
                eval: input.eval,
            },
        )
        .collect()
}

fn append_bolt_preamble<T>(transcript: &mut T, fixture: &CoreStage1Fixture)
where
    T: Transcript<Challenge = Fr>,
{
    let program_io = &fixture.io;
    let preprocessing_digest = fixture.verifier_preprocessing.shared.digest();
    append_bytes(transcript, b"preprocessing_digest", &preprocessing_digest);
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
    append_u64(transcript, b"ram_K", fixture.proof.ram_K as u64);
    append_u64(
        transcript,
        b"trace_length",
        fixture.proof.trace_length as u64,
    );
    append_u64(transcript, b"entry_address", fixture.entry_address);
    append_u64(
        transcript,
        b"ram_rw_phase1_num_rounds",
        fixture.proof.rw_config.ram_rw_phase1_num_rounds as u64,
    );
    append_u64(
        transcript,
        b"ram_rw_phase2_num_rounds",
        fixture.proof.rw_config.ram_rw_phase2_num_rounds as u64,
    );
    append_u64(
        transcript,
        b"registers_rw_phase1_num_rounds",
        fixture.proof.rw_config.registers_rw_phase1_num_rounds as u64,
    );
    append_u64(
        transcript,
        b"registers_rw_phase2_num_rounds",
        fixture.proof.rw_config.registers_rw_phase2_num_rounds as u64,
    );
    append_u64(
        transcript,
        b"log_k_chunk",
        fixture.proof.one_hot_config.log_k_chunk as u64,
    );
    append_u64(
        transcript,
        b"lookups_ra_virtual_log_k_chunk",
        fixture.proof.one_hot_config.lookups_ra_virtual_log_k_chunk as u64,
    );
    append_u64(transcript, b"dory_layout", fixture.proof.dory_layout as u64);
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

fn assert_commitment_parity(fixture: &CoreStage1Fixture, context: &BoltStage1Context) {
    let bolt_core_commitments = context
        .commitment_prover_trace
        .commitments
        .iter()
        .filter_map(|commitment| commitment.as_ref())
        .take(fixture.commitments.len())
        .map(commitment_to_ark)
        .collect::<Vec<_>>();
    assert_eq!(bolt_core_commitments, fixture.commitments);
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

fn assert_core_accepts_bolt_stage3(
    fixture: &CoreStage1Fixture,
    stage1_artifacts: &Stage1ExecutionArtifacts<Fr>,
    stage2_artifacts: &Stage2ExecutionArtifacts<Fr>,
    stage3_artifacts: &Stage3ExecutionArtifacts<Fr>,
) {
    let mut proof = clone_core_proof(&fixture.proof);
    proof.stage1_uni_skip_first_round_proof = to_core_uniskip_proof(&stage1_artifacts.sumchecks[0]);
    proof.stage1_sumcheck_proof =
        to_core_sumcheck_proof(&stage1_artifacts.sumchecks[1].proof.round_polynomials);
    proof.stage2_uni_skip_first_round_proof =
        to_core_stage2_uniskip_proof(&stage2_artifacts.sumchecks[0]);
    proof.stage2_sumcheck_proof =
        to_core_sumcheck_proof(&stage2_artifacts.sumchecks[1].proof.round_polynomials);
    proof.stage3_sumcheck_proof =
        to_core_sumcheck_proof(&stage3_artifacts.sumchecks[0].proof.round_polynomials);

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
    let _ = verifier.verify_stage3().expect("core accepts Bolt Stage 3");
}

fn assert_core_accepts_bolt_stage4(
    fixture: &CoreStage1Fixture,
    stage1_artifacts: &Stage1ExecutionArtifacts<Fr>,
    stage2_artifacts: &Stage2ExecutionArtifacts<Fr>,
    stage3_artifacts: &Stage3ExecutionArtifacts<Fr>,
    stage4_artifacts: &Stage4ExecutionArtifacts<Fr>,
) {
    let mut proof = clone_core_proof(&fixture.proof);
    proof.stage1_uni_skip_first_round_proof = to_core_uniskip_proof(&stage1_artifacts.sumchecks[0]);
    proof.stage1_sumcheck_proof =
        to_core_sumcheck_proof(&stage1_artifacts.sumchecks[1].proof.round_polynomials);
    proof.stage2_uni_skip_first_round_proof =
        to_core_stage2_uniskip_proof(&stage2_artifacts.sumchecks[0]);
    proof.stage2_sumcheck_proof =
        to_core_sumcheck_proof(&stage2_artifacts.sumchecks[1].proof.round_polynomials);
    proof.stage3_sumcheck_proof =
        to_core_sumcheck_proof(&stage3_artifacts.sumchecks[0].proof.round_polynomials);
    proof.stage4_sumcheck_proof =
        to_core_sumcheck_proof(&stage4_artifacts.sumchecks[0].proof.round_polynomials);

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
    let _ = verifier.verify_stage3().expect("core accepts Bolt Stage 3");
    let _ = verifier.verify_stage4().expect("core accepts Bolt Stage 4");
}

fn assert_core_accepts_bolt_stage5(
    fixture: &CoreStage1Fixture,
    stage1_artifacts: &Stage1ExecutionArtifacts<Fr>,
    stage2_artifacts: &Stage2ExecutionArtifacts<Fr>,
    stage3_artifacts: &Stage3ExecutionArtifacts<Fr>,
    stage4_artifacts: &Stage4ExecutionArtifacts<Fr>,
    stage5_artifacts: &kernel_stage5::Stage5ExecutionArtifacts<Fr>,
) {
    let mut proof = clone_core_proof(&fixture.proof);
    proof.stage1_uni_skip_first_round_proof = to_core_uniskip_proof(&stage1_artifacts.sumchecks[0]);
    proof.stage1_sumcheck_proof =
        to_core_sumcheck_proof(&stage1_artifacts.sumchecks[1].proof.round_polynomials);
    proof.stage2_uni_skip_first_round_proof =
        to_core_stage2_uniskip_proof(&stage2_artifacts.sumchecks[0]);
    proof.stage2_sumcheck_proof =
        to_core_sumcheck_proof(&stage2_artifacts.sumchecks[1].proof.round_polynomials);
    proof.stage3_sumcheck_proof =
        to_core_sumcheck_proof(&stage3_artifacts.sumchecks[0].proof.round_polynomials);
    proof.stage4_sumcheck_proof =
        to_core_sumcheck_proof(&stage4_artifacts.sumchecks[0].proof.round_polynomials);
    proof.stage5_sumcheck_proof =
        to_core_sumcheck_proof(&stage5_artifacts.sumchecks[0].proof.round_polynomials);

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
    let _ = verifier.verify_stage3().expect("core accepts Bolt Stage 3");
    let _ = verifier.verify_stage4().expect("core accepts Bolt Stage 4");
    let _ = verifier.verify_stage5().expect("core accepts Bolt Stage 5");
}

fn assert_core_accepts_bolt_stage6(
    fixture: &CoreStage1Fixture,
    stage1_artifacts: &Stage1ExecutionArtifacts<Fr>,
    stage2_artifacts: &Stage2ExecutionArtifacts<Fr>,
    stage3_artifacts: &Stage3ExecutionArtifacts<Fr>,
    stage4_artifacts: &Stage4ExecutionArtifacts<Fr>,
    stage5_artifacts: &kernel_stage5::Stage5ExecutionArtifacts<Fr>,
    stage6_artifacts: &kernel_stage6::Stage6ExecutionArtifacts<Fr>,
) {
    let mut proof = clone_core_proof(&fixture.proof);
    proof.stage1_uni_skip_first_round_proof = to_core_uniskip_proof(&stage1_artifacts.sumchecks[0]);
    proof.stage1_sumcheck_proof =
        to_core_sumcheck_proof(&stage1_artifacts.sumchecks[1].proof.round_polynomials);
    proof.stage2_uni_skip_first_round_proof =
        to_core_stage2_uniskip_proof(&stage2_artifacts.sumchecks[0]);
    proof.stage2_sumcheck_proof =
        to_core_sumcheck_proof(&stage2_artifacts.sumchecks[1].proof.round_polynomials);
    proof.stage3_sumcheck_proof =
        to_core_sumcheck_proof(&stage3_artifacts.sumchecks[0].proof.round_polynomials);
    proof.stage4_sumcheck_proof =
        to_core_sumcheck_proof(&stage4_artifacts.sumchecks[0].proof.round_polynomials);
    proof.stage5_sumcheck_proof =
        to_core_sumcheck_proof(&stage5_artifacts.sumchecks[0].proof.round_polynomials);
    proof.stage6_sumcheck_proof =
        to_core_sumcheck_proof(&stage6_artifacts.sumchecks[0].proof.round_polynomials);

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
    let _ = verifier.verify_stage3().expect("core accepts Bolt Stage 3");
    let _ = verifier.verify_stage4().expect("core accepts Bolt Stage 4");
    let _ = verifier.verify_stage5().expect("core accepts Bolt Stage 5");
    let _ = verifier.verify_stage6().expect("core accepts Bolt Stage 6");
}

fn assert_core_accepts_bolt_stage7(
    fixture: &CoreStage1Fixture,
    stage1_artifacts: &Stage1ExecutionArtifacts<Fr>,
    stage2_artifacts: &Stage2ExecutionArtifacts<Fr>,
    stage3_artifacts: &Stage3ExecutionArtifacts<Fr>,
    stage4_artifacts: &Stage4ExecutionArtifacts<Fr>,
    stage5_artifacts: &kernel_stage5::Stage5ExecutionArtifacts<Fr>,
    stage6_artifacts: &kernel_stage6::Stage6ExecutionArtifacts<Fr>,
    stage7_artifacts: &kernel_stage7::Stage7ExecutionArtifacts<Fr>,
) {
    let mut proof = clone_core_proof(&fixture.proof);
    proof.stage1_uni_skip_first_round_proof = to_core_uniskip_proof(&stage1_artifacts.sumchecks[0]);
    proof.stage1_sumcheck_proof =
        to_core_sumcheck_proof(&stage1_artifacts.sumchecks[1].proof.round_polynomials);
    proof.stage2_uni_skip_first_round_proof =
        to_core_stage2_uniskip_proof(&stage2_artifacts.sumchecks[0]);
    proof.stage2_sumcheck_proof =
        to_core_sumcheck_proof(&stage2_artifacts.sumchecks[1].proof.round_polynomials);
    proof.stage3_sumcheck_proof =
        to_core_sumcheck_proof(&stage3_artifacts.sumchecks[0].proof.round_polynomials);
    proof.stage4_sumcheck_proof =
        to_core_sumcheck_proof(&stage4_artifacts.sumchecks[0].proof.round_polynomials);
    proof.stage5_sumcheck_proof =
        to_core_sumcheck_proof(&stage5_artifacts.sumchecks[0].proof.round_polynomials);
    proof.stage6_sumcheck_proof =
        to_core_sumcheck_proof(&stage6_artifacts.sumchecks[0].proof.round_polynomials);
    proof.stage7_sumcheck_proof =
        to_core_sumcheck_proof(&stage7_artifacts.sumchecks[0].proof.round_polynomials);

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
    let _ = verifier.verify_stage3().expect("core accepts Bolt Stage 3");
    let _ = verifier.verify_stage4().expect("core accepts Bolt Stage 4");
    let _ = verifier.verify_stage5().expect("core accepts Bolt Stage 5");
    let _ = verifier.verify_stage6().expect("core accepts Bolt Stage 6");
    let _ = verifier.verify_stage7().expect("core accepts Bolt Stage 7");
}

fn assert_core_accepts_bolt_stage8(
    fixture: &CoreStage1Fixture,
    evaluation: &jolt_verifier::JoltEvaluationProof,
) {
    let mut proof = clone_core_proof(&fixture.proof);
    proof.joint_opening_proof = evaluation.joint_opening_proof.0.clone();

    let mut verifier = CoreVerifier::new(
        fixture.verifier_preprocessing,
        proof,
        fixture.io.clone(),
        None,
        None,
    )
    .expect("construct core verifier");
    verifier.run_preamble();
    let _ = verifier.verify_stage1().expect("core accepts Stage 1");
    let _ = verifier.verify_stage2().expect("core accepts Stage 2");
    let _ = verifier.verify_stage3().expect("core accepts Stage 3");
    let _ = verifier.verify_stage4().expect("core accepts Stage 4");
    let _ = verifier.verify_stage5().expect("core accepts Stage 5");
    let _ = verifier.verify_stage6().expect("core accepts Stage 6");
    let _ = verifier.verify_stage7().expect("core accepts Stage 7");
    let _ = verifier.verify_stage8().expect("core accepts Bolt Stage 8");
}

fn core_stage6_data(fixture: &CoreStage1Fixture) -> CoreStage6Data {
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
    let _ = verifier.verify_stage3().expect("core Stage 3 verifies");
    let _ = verifier.verify_stage4().expect("core Stage 4 verifies");
    let _ = verifier.verify_stage5().expect("core Stage 5 verifies");
    let _ = verifier.verify_stage6().expect("core Stage 6 verifies");

    CoreStage6Data {
        opening_inputs: core_stage6_opening_inputs(&verifier.opening_accumulator, &fixture.params),
        transcript_states: verifier.transcript.state_history[1..].to_vec(),
    }
}

fn core_stage7_data(fixture: &CoreStage1Fixture) -> CoreStage7Data {
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
    let _ = verifier.verify_stage3().expect("core Stage 3 verifies");
    let _ = verifier.verify_stage4().expect("core Stage 4 verifies");
    let _ = verifier.verify_stage5().expect("core Stage 5 verifies");
    let _ = verifier.verify_stage6().expect("core Stage 6 verifies");
    let _ = verifier.verify_stage7().expect("core Stage 7 verifies");

    CoreStage7Data {
        opening_inputs: core_stage7_opening_inputs(&verifier.opening_accumulator, &fixture.params),
        transcript_states: verifier.transcript.state_history[1..].to_vec(),
    }
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

fn assert_core_states_match_bolt_stage3(fixture: &CoreStage1Fixture, bolt_log: &[TranscriptEvent]) {
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
    let _ = verifier.verify_stage3().expect("core Stage 3 verifies");

    let core_states = verifier.transcript.state_history[1..].to_vec();
    let bolt_states = transcript_states(bolt_log);
    assert_eq!(core_states, bolt_states);
}

fn assert_core_states_match_bolt_stage4(fixture: &CoreStage1Fixture, bolt_log: &[TranscriptEvent]) {
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
    let _ = verifier.verify_stage3().expect("core Stage 3 verifies");
    let _ = verifier.verify_stage4().expect("core Stage 4 verifies");

    let core_states = verifier.transcript.state_history[1..].to_vec();
    let bolt_states = transcript_states(bolt_log);
    assert_eq!(core_states, bolt_states);
}

fn assert_core_states_match_bolt_stage5(fixture: &CoreStage1Fixture, bolt_log: &[TranscriptEvent]) {
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
    let _ = verifier.verify_stage3().expect("core Stage 3 verifies");
    let _ = verifier.verify_stage4().expect("core Stage 4 verifies");
    let _ = verifier.verify_stage5().expect("core Stage 5 verifies");

    let core_states = verifier.transcript.state_history[1..].to_vec();
    let bolt_states = transcript_states(bolt_log);
    assert_eq!(core_states, bolt_states);
}

fn assert_core_states_match_bolt_stage6(fixture: &CoreStage1Fixture, bolt_log: &[TranscriptEvent]) {
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
    let _ = verifier.verify_stage3().expect("core Stage 3 verifies");
    let _ = verifier.verify_stage4().expect("core Stage 4 verifies");
    let _ = verifier.verify_stage5().expect("core Stage 5 verifies");
    let _ = verifier.verify_stage6().expect("core Stage 6 verifies");

    let core_states = verifier.transcript.state_history[1..].to_vec();
    let bolt_states = transcript_states(bolt_log);
    assert_eq!(core_states, bolt_states);
}

fn assert_core_states_match_bolt_stage7(fixture: &CoreStage1Fixture, bolt_log: &[TranscriptEvent]) {
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
    let _ = verifier.verify_stage3().expect("core Stage 3 verifies");
    let _ = verifier.verify_stage4().expect("core Stage 4 verifies");
    let _ = verifier.verify_stage5().expect("core Stage 5 verifies");
    let _ = verifier.verify_stage6().expect("core Stage 6 verifies");
    let _ = verifier.verify_stage7().expect("core Stage 7 verifies");

    let core_states = verifier.transcript.state_history[1..].to_vec();
    let bolt_states = transcript_states(bolt_log);
    assert_eq!(core_states, bolt_states);
}

fn assert_core_states_match_bolt_stage8(fixture: &CoreStage1Fixture, bolt_log: &[TranscriptEvent]) {
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
    let _ = verifier.verify_stage3().expect("core Stage 3 verifies");
    let _ = verifier.verify_stage4().expect("core Stage 4 verifies");
    let _ = verifier.verify_stage5().expect("core Stage 5 verifies");
    let _ = verifier.verify_stage6().expect("core Stage 6 verifies");
    let _ = verifier.verify_stage7().expect("core Stage 7 verifies");
    let _ = verifier.verify_stage8().expect("core Stage 8 verifies");

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

fn core_stage6_opening_inputs<A: OpeningAccumulator<CoreFr>>(
    accumulator: &A,
    params: &JoltProtocolParams,
) -> Vec<kernel_stage6::Stage6OpeningInputValue<Fr>> {
    let mut inputs = vec![
        core_stage6_virtual_opening_input(
            accumulator,
            VirtualPolynomial::UnexpandedPC,
            SumcheckId::SpartanOuter,
            "stage6.input.stage1.UnexpandedPC",
        ),
        core_stage6_virtual_opening_input(
            accumulator,
            VirtualPolynomial::Imm,
            SumcheckId::SpartanOuter,
            "stage6.input.stage1.Imm",
        ),
    ];

    for (oracle, flag) in [
        ("AddOperands", CoreCircuitFlags::AddOperands),
        ("SubtractOperands", CoreCircuitFlags::SubtractOperands),
        ("MultiplyOperands", CoreCircuitFlags::MultiplyOperands),
        ("Load", CoreCircuitFlags::Load),
        ("Store", CoreCircuitFlags::Store),
        ("Jump", CoreCircuitFlags::Jump),
        (
            "WriteLookupOutputToRD",
            CoreCircuitFlags::WriteLookupOutputToRD,
        ),
        ("VirtualInstruction", CoreCircuitFlags::VirtualInstruction),
        ("Assert", CoreCircuitFlags::Assert),
        (
            "DoNotUpdateUnexpandedPC",
            CoreCircuitFlags::DoNotUpdateUnexpandedPC,
        ),
        ("Advice", CoreCircuitFlags::Advice),
        ("IsCompressed", CoreCircuitFlags::IsCompressed),
        ("IsFirstInSequence", CoreCircuitFlags::IsFirstInSequence),
        ("IsLastInSequence", CoreCircuitFlags::IsLastInSequence),
    ] {
        inputs.push(core_stage6_virtual_opening_input(
            accumulator,
            VirtualPolynomial::OpFlags(flag),
            SumcheckId::SpartanOuter,
            leak_str(&format!("stage6.input.stage1.OpFlag{oracle}")),
        ));
    }

    for (symbol, polynomial) in [
        (
            "stage6.input.stage2.OpFlagJump",
            VirtualPolynomial::OpFlags(CoreCircuitFlags::Jump),
        ),
        (
            "stage6.input.stage2.InstructionFlagBranch",
            VirtualPolynomial::InstructionFlags(CoreInstructionFlags::Branch),
        ),
        (
            "stage6.input.stage2.OpFlagWriteLookupOutputToRD",
            VirtualPolynomial::OpFlags(CoreCircuitFlags::WriteLookupOutputToRD),
        ),
        (
            "stage6.input.stage2.OpFlagVirtualInstruction",
            VirtualPolynomial::OpFlags(CoreCircuitFlags::VirtualInstruction),
        ),
    ] {
        inputs.push(core_stage6_virtual_opening_input(
            accumulator,
            polynomial,
            SumcheckId::SpartanProductVirtualization,
            symbol,
        ));
    }

    inputs.push(core_stage6_virtual_opening_input(
        accumulator,
        VirtualPolynomial::Imm,
        SumcheckId::InstructionInputVirtualization,
        "stage6.input.stage3.instruction_input.Imm",
    ));
    inputs.push(core_stage6_virtual_opening_input(
        accumulator,
        VirtualPolynomial::UnexpandedPC,
        SumcheckId::SpartanShift,
        "stage6.input.stage3.spartan_shift.UnexpandedPC",
    ));
    for (symbol, flag) in [
        (
            "stage6.input.stage3.instruction_input.InstructionFlagLeftOperandIsRs1Value",
            CoreInstructionFlags::LeftOperandIsRs1Value,
        ),
        (
            "stage6.input.stage3.instruction_input.InstructionFlagLeftOperandIsPC",
            CoreInstructionFlags::LeftOperandIsPC,
        ),
        (
            "stage6.input.stage3.instruction_input.InstructionFlagRightOperandIsRs2Value",
            CoreInstructionFlags::RightOperandIsRs2Value,
        ),
        (
            "stage6.input.stage3.instruction_input.InstructionFlagRightOperandIsImm",
            CoreInstructionFlags::RightOperandIsImm,
        ),
    ] {
        inputs.push(core_stage6_virtual_opening_input(
            accumulator,
            VirtualPolynomial::InstructionFlags(flag),
            SumcheckId::InstructionInputVirtualization,
            symbol,
        ));
    }
    inputs.push(core_stage6_virtual_opening_input(
        accumulator,
        VirtualPolynomial::InstructionFlags(CoreInstructionFlags::IsNoop),
        SumcheckId::SpartanShift,
        "stage6.input.stage3.spartan_shift.InstructionFlagIsNoop",
    ));
    for (symbol, flag) in [
        (
            "stage6.input.stage3.spartan_shift.OpFlagVirtualInstruction",
            CoreCircuitFlags::VirtualInstruction,
        ),
        (
            "stage6.input.stage3.spartan_shift.OpFlagIsFirstInSequence",
            CoreCircuitFlags::IsFirstInSequence,
        ),
    ] {
        inputs.push(core_stage6_virtual_opening_input(
            accumulator,
            VirtualPolynomial::OpFlags(flag),
            SumcheckId::SpartanShift,
            symbol,
        ));
    }

    for (symbol, polynomial) in [
        ("stage6.input.stage4.RdWa", VirtualPolynomial::RdWa),
        ("stage6.input.stage4.Rs1Ra", VirtualPolynomial::Rs1Ra),
        ("stage6.input.stage4.Rs2Ra", VirtualPolynomial::Rs2Ra),
    ] {
        inputs.push(core_stage6_virtual_opening_input(
            accumulator,
            polynomial,
            SumcheckId::RegistersReadWriteChecking,
            symbol,
        ));
    }
    inputs.push(core_stage6_virtual_opening_input(
        accumulator,
        VirtualPolynomial::RdWa,
        SumcheckId::RegistersValEvaluation,
        "stage6.input.stage5.registers_val_evaluation.RdWa",
    ));
    inputs.push(core_stage6_virtual_opening_input(
        accumulator,
        VirtualPolynomial::InstructionRafFlag,
        SumcheckId::InstructionReadRaf,
        "stage6.input.stage5.InstructionRafFlag",
    ));
    for index in 0..params.lookup_table_count {
        inputs.push(core_stage6_virtual_opening_input(
            accumulator,
            VirtualPolynomial::LookupTableFlag(index),
            SumcheckId::InstructionReadRaf,
            leak_str(&format!("stage6.input.stage5.LookupTableFlag_{index}")),
        ));
    }
    inputs.push(core_stage6_virtual_opening_input(
        accumulator,
        VirtualPolynomial::PC,
        SumcheckId::SpartanOuter,
        "stage6.input.stage1.PC",
    ));
    inputs.push(core_stage6_virtual_opening_input(
        accumulator,
        VirtualPolynomial::PC,
        SumcheckId::SpartanShift,
        "stage6.input.stage3.spartan_shift.PC",
    ));
    inputs.push(core_stage6_virtual_opening_input(
        accumulator,
        VirtualPolynomial::RamRa,
        SumcheckId::RamRaClaimReduction,
        "stage6.input.stage5.ram_ra_claim_reduction.RamRa",
    ));
    for index in 0..params.instruction_ra_virtual_d {
        inputs.push(core_stage6_virtual_opening_input(
            accumulator,
            VirtualPolynomial::InstructionRa(index),
            SumcheckId::InstructionReadRaf,
            leak_str(&format!(
                "stage6.input.stage5.instruction_read_raf.InstructionRa_{index}"
            )),
        ));
    }
    inputs.push(core_stage6_virtual_opening_input(
        accumulator,
        VirtualPolynomial::LookupOutput,
        SumcheckId::SpartanOuter,
        "stage6.input.stage1.LookupOutput",
    ));
    inputs.push(core_stage6_committed_opening_input(
        accumulator,
        CommittedPolynomial::RamInc,
        SumcheckId::RamReadWriteChecking,
        "stage6.input.stage2.ram_read_write.RamInc",
    ));
    inputs.push(core_stage6_committed_opening_input(
        accumulator,
        CommittedPolynomial::RamInc,
        SumcheckId::RamValCheck,
        "stage6.input.stage4.ram_val_check.RamInc",
    ));
    inputs.push(core_stage6_committed_opening_input(
        accumulator,
        CommittedPolynomial::RdInc,
        SumcheckId::RegistersReadWriteChecking,
        "stage6.input.stage4.registers_read_write.RdInc",
    ));
    inputs.push(core_stage6_committed_opening_input(
        accumulator,
        CommittedPolynomial::RdInc,
        SumcheckId::RegistersValEvaluation,
        "stage6.input.stage5.registers_val_evaluation.RdInc",
    ));
    inputs
}

fn core_stage6_virtual_opening_input<A: OpeningAccumulator<CoreFr>>(
    accumulator: &A,
    polynomial: VirtualPolynomial,
    sumcheck: SumcheckId,
    symbol: &'static str,
) -> kernel_stage6::Stage6OpeningInputValue<Fr> {
    let (point, eval) = accumulator.get_virtual_polynomial_opening(polynomial, sumcheck);
    kernel_stage6::Stage6OpeningInputValue {
        symbol,
        point: point.r.into_iter().map(Fr::from).collect(),
        eval: Fr::from(eval),
    }
}

fn core_stage6_committed_opening_input<A: OpeningAccumulator<CoreFr>>(
    accumulator: &A,
    polynomial: CommittedPolynomial,
    sumcheck: SumcheckId,
    symbol: &'static str,
) -> kernel_stage6::Stage6OpeningInputValue<Fr> {
    let (point, eval) = accumulator.get_committed_polynomial_opening(polynomial, sumcheck);
    kernel_stage6::Stage6OpeningInputValue {
        symbol,
        point: point.r.into_iter().map(Fr::from).collect(),
        eval: Fr::from(eval),
    }
}

fn core_stage7_opening_inputs<A: OpeningAccumulator<CoreFr>>(
    accumulator: &A,
    params: &JoltProtocolParams,
) -> Vec<kernel_stage7::Stage7OpeningInputValue<Fr>> {
    let mut inputs = Vec::new();
    inputs.push(core_stage7_virtual_opening_input(
        accumulator,
        VirtualPolynomial::RamHammingWeight,
        SumcheckId::RamHammingBooleanity,
        "stage7.input.stage6.hamming_booleanity.HammingWeight",
    ));
    for index in 0..params.instruction_d {
        inputs.push(core_stage7_committed_opening_input(
            accumulator,
            CommittedPolynomial::InstructionRa(index),
            SumcheckId::Booleanity,
            leak_str(&format!(
                "stage7.input.stage6.booleanity.InstructionRa_{index}"
            )),
        ));
        inputs.push(core_stage7_committed_opening_input(
            accumulator,
            CommittedPolynomial::InstructionRa(index),
            SumcheckId::InstructionRaVirtualization,
            leak_str(&format!(
                "stage7.input.stage6.instruction_ra_virtual.InstructionRa_{index}"
            )),
        ));
    }
    for index in 0..params.bytecode_d {
        inputs.push(core_stage7_committed_opening_input(
            accumulator,
            CommittedPolynomial::BytecodeRa(index),
            SumcheckId::Booleanity,
            leak_str(&format!(
                "stage7.input.stage6.booleanity.BytecodeRa_{index}"
            )),
        ));
        inputs.push(core_stage7_committed_opening_input(
            accumulator,
            CommittedPolynomial::BytecodeRa(index),
            SumcheckId::BytecodeReadRaf,
            leak_str(&format!(
                "stage7.input.stage6.bytecode_read_raf.BytecodeRa_{index}"
            )),
        ));
    }
    for index in 0..params.ram_d {
        inputs.push(core_stage7_committed_opening_input(
            accumulator,
            CommittedPolynomial::RamRa(index),
            SumcheckId::Booleanity,
            leak_str(&format!("stage7.input.stage6.booleanity.RamRa_{index}")),
        ));
        inputs.push(core_stage7_committed_opening_input(
            accumulator,
            CommittedPolynomial::RamRa(index),
            SumcheckId::RamRaVirtualization,
            leak_str(&format!("stage7.input.stage6.ram_ra_virtual.RamRa_{index}")),
        ));
    }
    inputs
}

fn core_stage7_virtual_opening_input<A: OpeningAccumulator<CoreFr>>(
    accumulator: &A,
    polynomial: VirtualPolynomial,
    sumcheck: SumcheckId,
    symbol: &'static str,
) -> kernel_stage7::Stage7OpeningInputValue<Fr> {
    let (point, eval) = accumulator.get_virtual_polynomial_opening(polynomial, sumcheck);
    kernel_stage7::Stage7OpeningInputValue {
        symbol,
        point: point.r.into_iter().map(Fr::from).collect(),
        eval: Fr::from(eval),
    }
}

fn core_stage7_committed_opening_input<A: OpeningAccumulator<CoreFr>>(
    accumulator: &A,
    polynomial: CommittedPolynomial,
    sumcheck: SumcheckId,
    symbol: &'static str,
) -> kernel_stage7::Stage7OpeningInputValue<Fr> {
    let (point, eval) = accumulator.get_committed_polynomial_opening(polynomial, sumcheck);
    kernel_stage7::Stage7OpeningInputValue {
        symbol,
        point: point.r.into_iter().map(Fr::from).collect(),
        eval: Fr::from(eval),
    }
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
    let slot = one_hot_source_slot(source);
    cycle_inputs
        .iter()
        .map(|cycle| cycle.one_hot[slot])
        .collect()
}

fn one_hot_source_slot(source: &str) -> usize {
    match source {
        "trace.instruction_keys" => 0,
        "trace.bytecode_indices" => 1,
        "trace.ram_addresses" => 2,
        _ => panic!("unsupported one-hot source `{source}`"),
    }
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

fn leak_stage3_program(program: &CompilerStage3CpuProgram) -> &'static KernelStage3CpuProgramPlan {
    Box::leak(Box::new(KernelStage3CpuProgramPlan {
        params: Stage3Params {
            field: leak_str(&program.params.field),
            pcs: leak_str(&program.params.pcs),
            transcript: leak_str(&program.params.transcript),
        },
        steps: leak_slice(
            program
                .steps
                .iter()
                .map(|plan| KernelStage3ProgramStepPlan {
                    kind: leak_str(&plan.kind),
                    symbol: leak_str(&plan.symbol),
                })
                .collect(),
        ),
        transcript_squeezes: leak_slice(
            program
                .transcript_squeezes
                .iter()
                .map(|plan| KernelStage3TranscriptSqueezePlan {
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
                .map(|plan| Stage3OpeningInputPlan {
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
                .map(|plan| KernelStage3FieldConstantPlan {
                    symbol: leak_str(&plan.symbol),
                    field: leak_str(&plan.field),
                    value: plan.value,
                })
                .collect(),
        ),
        field_exprs: leak_slice(
            program
                .field_exprs
                .iter()
                .map(|plan| KernelStage3FieldExprPlan {
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
                .map(|plan| KernelStage3KernelPlan {
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
                .map(|plan| KernelStage3SumcheckClaimPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    domain: leak_str(&plan.domain),
                    num_rounds: plan.num_rounds,
                    degree: plan.degree,
                    claim: leak_str(&plan.claim),
                    kernel: leak_str(plan.kernel.as_deref().expect("Stage 3 claim kernel")),
                    claim_value: leak_str(&plan.claim_value),
                    input_openings: leak_str_slice(&plan.input_openings),
                })
                .collect(),
        ),
        batches: leak_slice(
            program
                .batches
                .iter()
                .map(|plan| KernelStage3SumcheckBatchPlan {
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
                .map(|plan| KernelStage3SumcheckDriverPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    proof_slot: leak_str(&plan.proof_slot),
                    kernel: leak_str(plan.kernel.as_deref().expect("Stage 3 driver kernel")),
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
                .map(|plan| Stage3SumcheckInstanceResultPlan {
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
                .map(|plan| KernelStage3SumcheckEvalPlan {
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
                .map(|plan| Stage3PointSlicePlan {
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
                .map(|plan| Stage3PointConcatPlan {
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
                .map(|plan| KernelStage3OpeningClaimPlan {
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
        opening_equalities: leak_slice(
            program
                .opening_equalities
                .iter()
                .map(|plan| KernelStage3OpeningClaimEqualityPlan {
                    symbol: leak_str(&plan.symbol),
                    mode: leak_str(&plan.mode),
                    lhs: leak_str(&plan.lhs),
                    rhs: leak_str(&plan.rhs),
                })
                .collect(),
        ),
        opening_batches: leak_slice(
            program
                .opening_batches
                .iter()
                .map(|plan| KernelStage3OpeningBatchPlan {
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

fn leak_stage4_program(program: &CompilerStage4CpuProgram) -> &'static KernelStage4CpuProgramPlan {
    Box::leak(Box::new(KernelStage4CpuProgramPlan {
        role: program.role.as_str(),
        params: Stage4Params {
            field: leak_str(&program.params.field),
            pcs: leak_str(&program.params.pcs),
            transcript: leak_str(&program.params.transcript),
        },
        steps: leak_slice(
            program
                .steps
                .iter()
                .map(|plan| KernelStage4ProgramStepPlan {
                    kind: leak_str(&plan.kind),
                    symbol: leak_str(&plan.symbol),
                })
                .collect(),
        ),
        transcript_squeezes: leak_slice(
            program
                .transcript_squeezes
                .iter()
                .map(|plan| KernelStage4TranscriptSqueezePlan {
                    symbol: leak_str(&plan.symbol),
                    label: leak_str(&plan.label),
                    kind: leak_str(&plan.kind),
                    count: plan.count,
                })
                .collect(),
        ),
        transcript_absorb_bytes: leak_slice(
            program
                .transcript_absorb_bytes
                .iter()
                .map(|plan| KernelStage4TranscriptAbsorbBytesPlan {
                    symbol: leak_str(&plan.symbol),
                    label: leak_str(&plan.label),
                    payload: leak_str(&plan.payload),
                })
                .collect(),
        ),
        opening_inputs: leak_slice(
            program
                .opening_inputs
                .iter()
                .map(|plan| Stage4OpeningInputPlan {
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
                .map(|plan| KernelStage4FieldConstantPlan {
                    symbol: leak_str(&plan.symbol),
                    field: leak_str(&plan.field),
                    value: plan.value,
                })
                .collect(),
        ),
        field_exprs: leak_slice(
            program
                .field_exprs
                .iter()
                .map(|plan| KernelStage4FieldExprPlan {
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
                .map(|plan| KernelStage4KernelPlan {
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
                .map(|plan| KernelStage4SumcheckClaimPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    domain: leak_str(&plan.domain),
                    num_rounds: plan.num_rounds,
                    degree: plan.degree,
                    claim: leak_str(&plan.claim),
                    kernel: plan.kernel.as_deref().map(leak_str),
                    relation: plan.relation.as_deref().map(leak_str),
                    claim_value: leak_str(&plan.claim_value),
                    input_openings: leak_str_slice(&plan.input_openings),
                })
                .collect(),
        ),
        batches: leak_slice(
            program
                .batches
                .iter()
                .map(|plan| KernelStage4SumcheckBatchPlan {
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
                .map(|plan| KernelStage4SumcheckDriverPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    proof_slot: leak_str(&plan.proof_slot),
                    kernel: plan.kernel.as_deref().map(leak_str),
                    relation: plan.relation.as_deref().map(leak_str),
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
                .map(|plan| Stage4SumcheckInstanceResultPlan {
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
                .map(|plan| KernelStage4SumcheckEvalPlan {
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
                .map(|plan| Stage4PointSlicePlan {
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
                .map(|plan| Stage4PointConcatPlan {
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
                .map(|plan| KernelStage4OpeningClaimPlan {
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
        opening_equalities: leak_slice(
            program
                .opening_equalities
                .iter()
                .map(|plan| KernelStage4OpeningClaimEqualityPlan {
                    symbol: leak_str(&plan.symbol),
                    mode: leak_str(&plan.mode),
                    lhs: leak_str(&plan.lhs),
                    rhs: leak_str(&plan.rhs),
                })
                .collect(),
        ),
        opening_batches: leak_slice(
            program
                .opening_batches
                .iter()
                .map(|plan| KernelStage4OpeningBatchPlan {
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

fn leak_stage5_program(
    program: &CompilerStage5CpuProgram,
) -> &'static kernel_stage5::Stage5CpuProgramPlan {
    Box::leak(Box::new(kernel_stage5::Stage5CpuProgramPlan {
        role: program.role.as_str(),
        params: kernel_stage5::Stage5Params {
            field: leak_str(&program.params.field),
            pcs: leak_str(&program.params.pcs),
            transcript: leak_str(&program.params.transcript),
        },
        steps: leak_slice(
            program
                .steps
                .iter()
                .map(|plan| kernel_stage5::Stage5ProgramStepPlan {
                    kind: leak_str(&plan.kind),
                    symbol: leak_str(&plan.symbol),
                })
                .collect(),
        ),
        transcript_squeezes: leak_slice(
            program
                .transcript_squeezes
                .iter()
                .map(|plan| kernel_stage5::Stage5TranscriptSqueezePlan {
                    symbol: leak_str(&plan.symbol),
                    label: leak_str(&plan.label),
                    kind: leak_str(&plan.kind),
                    count: plan.count,
                })
                .collect(),
        ),
        transcript_absorb_bytes: leak_slice(
            program
                .transcript_absorb_bytes
                .iter()
                .map(|plan| kernel_stage5::Stage5TranscriptAbsorbBytesPlan {
                    symbol: leak_str(&plan.symbol),
                    label: leak_str(&plan.label),
                    payload: leak_str(&plan.payload),
                })
                .collect(),
        ),
        opening_inputs: leak_slice(
            program
                .opening_inputs
                .iter()
                .map(|plan| kernel_stage5::Stage5OpeningInputPlan {
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
                .map(|plan| kernel_stage5::Stage5FieldConstantPlan {
                    symbol: leak_str(&plan.symbol),
                    field: leak_str(&plan.field),
                    value: plan.value,
                })
                .collect(),
        ),
        field_exprs: leak_slice(
            program
                .field_exprs
                .iter()
                .map(|plan| kernel_stage5::Stage5FieldExprPlan {
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
                .map(|plan| kernel_stage5::Stage5KernelPlan {
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
                .map(|plan| kernel_stage5::Stage5SumcheckClaimPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    domain: leak_str(&plan.domain),
                    num_rounds: plan.num_rounds,
                    degree: plan.degree,
                    claim: leak_str(&plan.claim),
                    kernel: plan.kernel.as_deref().map(leak_str),
                    relation: plan.relation.as_deref().map(leak_str),
                    claim_value: leak_str(&plan.claim_value),
                    input_openings: leak_str_slice(&plan.input_openings),
                })
                .collect(),
        ),
        batches: leak_slice(
            program
                .batches
                .iter()
                .map(|plan| kernel_stage5::Stage5SumcheckBatchPlan {
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
                .map(|plan| kernel_stage5::Stage5SumcheckDriverPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    proof_slot: leak_str(&plan.proof_slot),
                    kernel: plan.kernel.as_deref().map(leak_str),
                    relation: plan.relation.as_deref().map(leak_str),
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
                .map(|plan| kernel_stage5::Stage5SumcheckInstanceResultPlan {
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
                .map(|plan| kernel_stage5::Stage5SumcheckEvalPlan {
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
                .map(|plan| kernel_stage5::Stage5PointSlicePlan {
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
                .map(|plan| kernel_stage5::Stage5PointConcatPlan {
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
                .map(|plan| kernel_stage5::Stage5OpeningClaimPlan {
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
        opening_equalities: leak_slice(
            program
                .opening_equalities
                .iter()
                .map(|plan| kernel_stage5::Stage5OpeningClaimEqualityPlan {
                    symbol: leak_str(&plan.symbol),
                    mode: leak_str(&plan.mode),
                    lhs: leak_str(&plan.lhs),
                    rhs: leak_str(&plan.rhs),
                })
                .collect(),
        ),
        opening_batches: leak_slice(
            program
                .opening_batches
                .iter()
                .map(|plan| kernel_stage5::Stage5OpeningBatchPlan {
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

fn leak_stage6_program(
    program: &CompilerStage6CpuProgram,
) -> &'static kernel_stage6::Stage6CpuProgramPlan {
    Box::leak(Box::new(kernel_stage6::Stage6CpuProgramPlan {
        role: program.role.as_str(),
        params: kernel_stage6::Stage6Params {
            field: leak_str(&program.params.field),
            pcs: leak_str(&program.params.pcs),
            transcript: leak_str(&program.params.transcript),
        },
        steps: leak_slice(
            program
                .steps
                .iter()
                .map(|plan| kernel_stage6::Stage6ProgramStepPlan {
                    kind: leak_str(&plan.kind),
                    symbol: leak_str(&plan.symbol),
                })
                .collect(),
        ),
        transcript_squeezes: leak_slice(
            program
                .transcript_squeezes
                .iter()
                .map(|plan| kernel_stage6::Stage6TranscriptSqueezePlan {
                    symbol: leak_str(&plan.symbol),
                    label: leak_str(&plan.label),
                    kind: leak_str(&plan.kind),
                    count: plan.count,
                })
                .collect(),
        ),
        transcript_absorb_bytes: leak_slice(
            program
                .transcript_absorb_bytes
                .iter()
                .map(|plan| kernel_stage6::Stage6TranscriptAbsorbBytesPlan {
                    symbol: leak_str(&plan.symbol),
                    label: leak_str(&plan.label),
                    payload: leak_str(&plan.payload),
                })
                .collect(),
        ),
        opening_inputs: leak_slice(
            program
                .opening_inputs
                .iter()
                .map(|plan| kernel_stage6::Stage6OpeningInputPlan {
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
                .map(|plan| kernel_stage6::Stage6FieldConstantPlan {
                    symbol: leak_str(&plan.symbol),
                    field: leak_str(&plan.field),
                    value: plan.value,
                })
                .collect(),
        ),
        field_exprs: leak_slice(
            program
                .field_exprs
                .iter()
                .map(|plan| kernel_stage6::Stage6FieldExprPlan {
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
                .map(|plan| kernel_stage6::Stage6KernelPlan {
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
                .map(|plan| kernel_stage6::Stage6SumcheckClaimPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    domain: leak_str(&plan.domain),
                    num_rounds: plan.num_rounds,
                    degree: plan.degree,
                    claim: leak_str(&plan.claim),
                    kernel: plan.kernel.as_deref().map(leak_str),
                    relation: plan.relation.as_deref().map(leak_str),
                    claim_value: leak_str(&plan.claim_value),
                    input_openings: leak_str_slice(&plan.input_openings),
                })
                .collect(),
        ),
        batches: leak_slice(
            program
                .batches
                .iter()
                .map(|plan| kernel_stage6::Stage6SumcheckBatchPlan {
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
                .map(|plan| kernel_stage6::Stage6SumcheckDriverPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    proof_slot: leak_str(&plan.proof_slot),
                    kernel: plan.kernel.as_deref().map(leak_str),
                    relation: plan.relation.as_deref().map(leak_str),
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
                .map(|plan| kernel_stage6::Stage6SumcheckInstanceResultPlan {
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
                .map(|plan| kernel_stage6::Stage6SumcheckEvalPlan {
                    symbol: leak_str(&plan.symbol),
                    source: leak_str(&plan.source),
                    name: leak_str(&plan.name),
                    index: plan.index,
                    oracle: leak_str(&plan.oracle),
                })
                .collect(),
        ),
        point_zeros: leak_slice(
            program
                .point_zeros
                .iter()
                .map(|plan| kernel_stage6::Stage6PointZeroPlan {
                    symbol: leak_str(&plan.symbol),
                    field: leak_str(&plan.field),
                    arity: plan.arity,
                })
                .collect(),
        ),
        point_slices: leak_slice(
            program
                .point_slices
                .iter()
                .map(|plan| kernel_stage6::Stage6PointSlicePlan {
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
                .map(|plan| kernel_stage6::Stage6PointConcatPlan {
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
                .map(|plan| kernel_stage6::Stage6OpeningClaimPlan {
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
        opening_equalities: leak_slice(
            program
                .opening_equalities
                .iter()
                .map(|plan| kernel_stage6::Stage6OpeningClaimEqualityPlan {
                    symbol: leak_str(&plan.symbol),
                    mode: leak_str(&plan.mode),
                    lhs: leak_str(&plan.lhs),
                    rhs: leak_str(&plan.rhs),
                })
                .collect(),
        ),
        opening_batches: leak_slice(
            program
                .opening_batches
                .iter()
                .map(|plan| kernel_stage6::Stage6OpeningBatchPlan {
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

fn leak_stage7_program(
    program: &CompilerStage7CpuProgram,
) -> &'static kernel_stage7::Stage7CpuProgramPlan {
    Box::leak(Box::new(kernel_stage7::Stage7CpuProgramPlan {
        role: program.role.as_str(),
        params: kernel_stage7::Stage7Params {
            field: leak_str(&program.params.field),
            pcs: leak_str(&program.params.pcs),
            transcript: leak_str(&program.params.transcript),
        },
        steps: leak_slice(
            program
                .steps
                .iter()
                .map(|plan| kernel_stage7::Stage7ProgramStepPlan {
                    kind: leak_str(&plan.kind),
                    symbol: leak_str(&plan.symbol),
                })
                .collect(),
        ),
        transcript_squeezes: leak_slice(
            program
                .transcript_squeezes
                .iter()
                .map(|plan| kernel_stage7::Stage7TranscriptSqueezePlan {
                    symbol: leak_str(&plan.symbol),
                    label: leak_str(&plan.label),
                    kind: leak_str(&plan.kind),
                    count: plan.count,
                })
                .collect(),
        ),
        transcript_absorb_bytes: leak_slice(
            program
                .transcript_absorb_bytes
                .iter()
                .map(|plan| kernel_stage7::Stage7TranscriptAbsorbBytesPlan {
                    symbol: leak_str(&plan.symbol),
                    label: leak_str(&plan.label),
                    payload: leak_str(&plan.payload),
                })
                .collect(),
        ),
        opening_inputs: leak_slice(
            program
                .opening_inputs
                .iter()
                .map(|plan| kernel_stage7::Stage7OpeningInputPlan {
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
                .map(|plan| kernel_stage7::Stage7FieldConstantPlan {
                    symbol: leak_str(&plan.symbol),
                    field: leak_str(&plan.field),
                    value: plan.value,
                })
                .collect(),
        ),
        field_exprs: leak_slice(
            program
                .field_exprs
                .iter()
                .map(|plan| kernel_stage7::Stage7FieldExprPlan {
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
                .map(|plan| kernel_stage7::Stage7KernelPlan {
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
                .map(|plan| kernel_stage7::Stage7SumcheckClaimPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    domain: leak_str(&plan.domain),
                    num_rounds: plan.num_rounds,
                    degree: plan.degree,
                    claim: leak_str(&plan.claim),
                    kernel: plan.kernel.as_deref().map(leak_str),
                    relation: plan.relation.as_deref().map(leak_str),
                    claim_value: leak_str(&plan.claim_value),
                    input_openings: leak_str_slice(&plan.input_openings),
                })
                .collect(),
        ),
        batches: leak_slice(
            program
                .batches
                .iter()
                .map(|plan| kernel_stage7::Stage7SumcheckBatchPlan {
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
                .map(|plan| kernel_stage7::Stage7SumcheckDriverPlan {
                    symbol: leak_str(&plan.symbol),
                    stage: leak_str(&plan.stage),
                    proof_slot: leak_str(&plan.proof_slot),
                    kernel: plan.kernel.as_deref().map(leak_str),
                    relation: plan.relation.as_deref().map(leak_str),
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
                .map(|plan| kernel_stage7::Stage7SumcheckInstanceResultPlan {
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
                .map(|plan| kernel_stage7::Stage7SumcheckEvalPlan {
                    symbol: leak_str(&plan.symbol),
                    source: leak_str(&plan.source),
                    name: leak_str(&plan.name),
                    index: plan.index,
                    oracle: leak_str(&plan.oracle),
                })
                .collect(),
        ),
        point_zeros: leak_slice(
            program
                .point_zeros
                .iter()
                .map(|plan| kernel_stage7::Stage7PointZeroPlan {
                    symbol: leak_str(&plan.symbol),
                    field: leak_str(&plan.field),
                    arity: plan.arity,
                })
                .collect(),
        ),
        point_slices: leak_slice(
            program
                .point_slices
                .iter()
                .map(|plan| kernel_stage7::Stage7PointSlicePlan {
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
                .map(|plan| kernel_stage7::Stage7PointConcatPlan {
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
                .map(|plan| kernel_stage7::Stage7OpeningClaimPlan {
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
        opening_equalities: leak_slice(
            program
                .opening_equalities
                .iter()
                .map(|plan| kernel_stage7::Stage7OpeningClaimEqualityPlan {
                    symbol: leak_str(&plan.symbol),
                    mode: leak_str(&plan.mode),
                    lhs: leak_str(&plan.lhs),
                    rhs: leak_str(&plan.rhs),
                })
                .collect(),
        ),
        opening_batches: leak_slice(
            program
                .opening_batches
                .iter()
                .map(|plan| kernel_stage7::Stage7OpeningBatchPlan {
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

fn leak_generated_commitment_prover_program(
    program: &CommitmentCpuProgram,
) -> &'static jolt_prover::stages::commitment::CommitmentProverProgramPlan {
    Box::leak(Box::new(
        jolt_prover::stages::commitment::CommitmentProverProgramPlan {
            params: jolt_prover::stages::commitment::CommitmentParams {
                field: leak_str(&program.params.field),
                pcs: leak_str(&program.params.pcs),
                transcript: leak_str(&program.params.transcript),
            },
            oracle_plans: leak_slice(
                program
                    .oracle_plans
                    .iter()
                    .map(|plan| jolt_prover::stages::commitment::OraclePlan {
                        oracle: leak_str(&plan.oracle),
                        domain: leak_str(&plan.domain),
                        num_vars: plan.num_vars,
                    })
                    .collect(),
            ),
            batch_plans: leak_slice(
                program
                    .batch_plans
                    .iter()
                    .map(
                        |plan| jolt_prover::stages::commitment::CommitmentBatchPlan {
                            artifact: leak_str(&plan.artifact),
                            pcs: leak_str(&plan.pcs),
                            oracle_family: leak_str(&plan.oracle_family),
                            label: leak_str(&plan.label),
                            oracles: leak_str_slice(&plan.oracles),
                            count: plan.count,
                            domain: leak_str(&plan.domain),
                            num_vars: plan.num_vars,
                        },
                    )
                    .collect(),
            ),
            optional_plans: leak_slice(
                program
                    .optional_plans
                    .iter()
                    .map(
                        |plan| {
                            jolt_prover::stages::commitment::OptionalCommitmentPlan {
                        artifact: leak_str(&plan.artifact),
                        pcs: leak_str(&plan.pcs),
                        oracle: leak_str(&plan.oracle),
                        label: leak_str(&plan.label),
                        domain: leak_str(&plan.domain),
                        num_vars: plan.num_vars,
                        skip_policy: match plan.skip_policy {
                            OptionalSkipPolicy::MissingOrZero => {
                                jolt_prover::stages::commitment::OptionalSkipPolicy::MissingOrZero
                            }
                        },
                    }
                        },
                    )
                    .collect(),
            ),
            transcript_steps: leak_slice(
                program
                    .transcript_steps
                    .iter()
                    .map(|step| jolt_prover::stages::commitment::TranscriptStep {
                        label: leak_str(&step.label),
                        source: leak_str(&step.source),
                        optional: step.optional,
                    })
                    .collect(),
            ),
        },
    ))
}

fn leak_generated_stage8_prover_program(
    program: &CompilerStage8CpuProgram,
) -> &'static generated_prover_stage8::Stage8EvaluationProgramPlan {
    let evaluation_point_source = program
        .opening_inputs
        .iter()
        .find(|input| input.symbol == "stage8.evaluation.point_source")
        .expect("stage8 evaluation point source exists");
    Box::leak(Box::new(
        generated_prover_stage8::Stage8EvaluationProgramPlan {
            role: role_name(&program.role),
            function: leak_str(&program.function),
            params: generated_prover_stage8::Stage8Params {
                field: leak_str(&program.params.field),
                pcs: leak_str(&program.params.pcs),
                transcript: leak_str(&program.params.transcript),
            },
            evaluation_point_source: generated_prover_stage8::Stage8OpeningInputPlan {
                symbol: leak_str(&evaluation_point_source.symbol),
                source_stage: leak_str(&evaluation_point_source.source_stage),
                source_claim: leak_str(&evaluation_point_source.source_claim),
                oracle: leak_str(&evaluation_point_source.oracle),
                domain: leak_str(&evaluation_point_source.domain),
                point_arity: evaluation_point_source.point_arity,
                claim_kind: leak_str(&evaluation_point_source.claim_kind),
            },
            opening_inputs: leak_slice(
                program
                    .opening_inputs
                    .iter()
                    .map(|plan| generated_prover_stage8::Stage8OpeningInputPlan {
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
            opening_claims: leak_slice(
                program
                    .opening_claims
                    .iter()
                    .map(|plan| generated_prover_stage8::Stage8OpeningClaimPlan {
                        symbol: leak_str(&plan.symbol),
                        oracle: leak_str(&plan.oracle),
                        family: leak_str(&plan.family),
                        domain: leak_str(&plan.domain),
                        point_arity: plan.point_arity,
                        point_source: leak_str(&plan.point_source),
                        eval_source: leak_str(&plan.eval_source),
                        source_stage: leak_str(&plan.source_stage),
                        source_claim: leak_str(&plan.source_claim),
                    })
                    .collect(),
            ),
            opening_batch: generated_prover_stage8::Stage8OpeningBatchPlan {
                symbol: leak_str(&program.opening_batches[0].symbol),
                proof_slot: leak_str(&program.opening_batches[0].proof_slot),
                policy: leak_str(&program.opening_batches[0].policy),
                count: program.opening_batches[0].count,
                ordered_claims: leak_str_slice(&program.opening_batches[0].ordered_claims),
            },
            pcs_proof: generated_prover_stage8::Stage8PcsProofPlan {
                symbol: leak_str(&program.pcs_proofs[0].symbol),
                mode: leak_str(&program.pcs_proofs[0].mode),
                pcs: leak_str(&program.pcs_proofs[0].pcs),
                proof_slot: leak_str(&program.pcs_proofs[0].proof_slot),
                transcript_label: leak_str(&program.pcs_proofs[0].transcript_label),
                batch: leak_str(&program.pcs_proofs[0].batch),
            },
        },
    ))
}

fn leak_generated_stage8_verifier_program(
    program: &CompilerStage8CpuProgram,
) -> &'static generated_stage8::Stage8EvaluationProgramPlan {
    let evaluation_point_source = program
        .opening_inputs
        .iter()
        .find(|input| input.symbol == "stage8.evaluation.point_source")
        .expect("stage8 evaluation point source exists");
    Box::leak(Box::new(generated_stage8::Stage8EvaluationProgramPlan {
        role: role_name(&program.role),
        function: leak_str(&program.function),
        params: generated_stage8::Stage8Params {
            field: leak_str(&program.params.field),
            pcs: leak_str(&program.params.pcs),
            transcript: leak_str(&program.params.transcript),
        },
        evaluation_point_source: generated_stage8::Stage8OpeningInputPlan {
            symbol: leak_str(&evaluation_point_source.symbol),
            source_stage: leak_str(&evaluation_point_source.source_stage),
            source_claim: leak_str(&evaluation_point_source.source_claim),
            oracle: leak_str(&evaluation_point_source.oracle),
            domain: leak_str(&evaluation_point_source.domain),
            point_arity: evaluation_point_source.point_arity,
            claim_kind: leak_str(&evaluation_point_source.claim_kind),
        },
        opening_inputs: leak_slice(
            program
                .opening_inputs
                .iter()
                .map(|plan| generated_stage8::Stage8OpeningInputPlan {
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
        opening_claims: leak_slice(
            program
                .opening_claims
                .iter()
                .map(|plan| generated_stage8::Stage8OpeningClaimPlan {
                    symbol: leak_str(&plan.symbol),
                    oracle: leak_str(&plan.oracle),
                    family: leak_str(&plan.family),
                    domain: leak_str(&plan.domain),
                    point_arity: plan.point_arity,
                    point_source: leak_str(&plan.point_source),
                    eval_source: leak_str(&plan.eval_source),
                    source_stage: leak_str(&plan.source_stage),
                    source_claim: leak_str(&plan.source_claim),
                })
                .collect(),
        ),
        opening_batch: generated_stage8::Stage8OpeningBatchPlan {
            symbol: leak_str(&program.opening_batches[0].symbol),
            proof_slot: leak_str(&program.opening_batches[0].proof_slot),
            policy: leak_str(&program.opening_batches[0].policy),
            count: program.opening_batches[0].count,
            ordered_claims: leak_str_slice(&program.opening_batches[0].ordered_claims),
        },
        pcs_proof: generated_stage8::Stage8PcsProofPlan {
            symbol: leak_str(&program.pcs_proofs[0].symbol),
            mode: leak_str(&program.pcs_proofs[0].mode),
            pcs: leak_str(&program.pcs_proofs[0].pcs),
            proof_slot: leak_str(&program.pcs_proofs[0].proof_slot),
            transcript_label: leak_str(&program.pcs_proofs[0].transcript_label),
            batch: leak_str(&program.pcs_proofs[0].batch),
        },
    }))
}

fn role_name(role: &Role) -> &'static str {
    match role {
        Role::Prover => "prover",
        Role::Verifier => "verifier",
    }
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
