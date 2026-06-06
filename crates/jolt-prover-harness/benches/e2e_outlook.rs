use std::{
    env, fs,
    io::{self, Write},
    path::{Path, PathBuf},
    time::Instant,
};

#[cfg(all(feature = "core-fixtures", not(feature = "field-inline")))]
use std::sync::{Arc, Mutex};

#[cfg(feature = "zk")]
use common::constants::MAX_BLINDFOLD_GENERATORS;
use common::constants::{ONEHOT_CHUNK_THRESHOLD_LOG_T, REGISTER_COUNT};
use common::jolt_device::JoltDevice;
use jolt_backends::cpu::{CpuBackend, CpuBackendConfig};
use jolt_claims::protocols::jolt::{JoltFormulaDimensions, JoltOneHotConfig, JoltReadWriteConfig};
use jolt_crypto::{Bn254G1, Pedersen};
use jolt_dory::DoryScheme;
use jolt_field::Fr;
use jolt_openings::CommitmentScheme;
use jolt_program::execution::RamAccess;
use jolt_prover_harness::{
    evaluate_perf, trace_sdk_guest, PerfGate, RunMetrics, SdkGuestTraceRequest,
};
use jolt_transcript::Blake2bTranscript;
use jolt_verifier::{proof::TracePolynomialOrder, JoltVerifierPreprocessing};
use jolt_witness::protocols::jolt_vm::{
    JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackedJoltVmWitness, RV64_LOOKUP_ADDRESS_BITS,
};
use serde::{Deserialize, Serialize};

#[cfg(feature = "field-inline")]
use jolt_program::preprocess::JoltProgramPreprocessing;
#[cfg(feature = "field-inline")]
use jolt_riscv::JoltInstructionKind;

#[cfg(all(feature = "core-fixtures", not(feature = "field-inline")))]
use jolt_core::{
    curve::Bn254Curve,
    host,
    poly::commitment::dory::DoryCommitmentScheme,
    zkvm::{
        prover::JoltProverPreprocessing as CoreProverPreprocessing,
        verifier::{
            JoltSharedPreprocessing, JoltVerifierPreprocessing as CoreVerifierPreprocessing,
        },
        RV64IMACProver,
    },
};
#[cfg(all(feature = "core-fixtures", not(feature = "field-inline")))]
use jolt_dory::DoryVerifierSetup;
#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
use jolt_verifier::compat::convert::CoreCurveBridge;
#[cfg(all(feature = "core-fixtures", not(feature = "field-inline")))]
use jolt_verifier::compat::convert::ImportedCoreProof;
#[cfg(all(feature = "core-fixtures", not(feature = "field-inline")))]
use tracing::{Id, Subscriber};
#[cfg(all(feature = "core-fixtures", not(feature = "field-inline")))]
use tracing_subscriber::{
    layer::{Context, Layer},
    prelude::*,
    registry::LookupSpan,
};

#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
use jolt_crypto::PedersenSetup;

const SHA2_CHAIN_GUEST: &str = "sha2-chain-guest";
#[cfg(feature = "field-inline")]
const FIELD_INLINE_EQ_POLY_GUEST: &str = "field-inline-eq-poly-guest";
const DEFAULT_TRACE_POWER: u32 = 16;
const CYCLES_PER_SHA256: f64 = 3396.0;
const SAFETY_MARGIN: f64 = 0.9;
const PARITY_FAIL_RATIO: f64 = 1.15;

type BenchResult<T> = Result<T, String>;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
enum OutlookMode {
    Transparent,
    Zk,
    FieldInline,
    ZkFieldInline,
}

impl OutlookMode {
    const fn current() -> Self {
        match (cfg!(feature = "zk"), cfg!(feature = "field-inline")) {
            (false, false) => Self::Transparent,
            (true, false) => Self::Zk,
            (false, true) => Self::FieldInline,
            (true, true) => Self::ZkFieldInline,
        }
    }

    #[cfg(feature = "field-inline")]
    const fn baseline_without_field_inline(self) -> Option<Self> {
        match self {
            Self::FieldInline => Some(Self::Transparent),
            Self::ZkFieldInline => Some(Self::Zk),
            Self::Transparent | Self::Zk => None,
        }
    }

    const fn slug(self) -> &'static str {
        match self {
            Self::Transparent => "transparent",
            Self::Zk => "zk",
            Self::FieldInline => "field_inline",
            Self::ZkFieldInline => "zk_field_inline",
        }
    }
}

#[derive(Clone, Debug)]
struct BenchConfig {
    trace_power: u32,
    max_padded_trace_length: usize,
    samples: usize,
}

#[derive(Clone, Copy, Debug)]
struct ProofShape {
    trace_length: usize,
    ram_k: usize,
    bytecode_k: usize,
    instruction_ra_polynomials: usize,
    bytecode_ra_polynomials: usize,
    ram_ra_polynomials: usize,
    rw_config: JoltReadWriteConfig,
    one_hot_config: JoltOneHotConfig,
    trace_polynomial_order: TracePolynomialOrder,
}

impl ProofShape {
    const fn report(self) -> ProofShapeReport {
        ProofShapeReport {
            trace_length: self.trace_length,
            ram_k: self.ram_k,
            ram_log_k: self.ram_k.trailing_zeros() as usize,
            bytecode_k: self.bytecode_k,
            bytecode_log_k: self.bytecode_k.trailing_zeros() as usize,
            instruction_ra_polynomials: self.instruction_ra_polynomials,
            bytecode_ra_polynomials: self.bytecode_ra_polynomials,
            ram_ra_polynomials: self.ram_ra_polynomials,
            committed_polynomials: 2
                + self.instruction_ra_polynomials
                + self.bytecode_ra_polynomials
                + self.ram_ra_polynomials,
            log_k_chunk: self.one_hot_config.log_k_chunk,
            lookups_ra_virtual_log_k_chunk: self.one_hot_config.lookups_ra_virtual_log_k_chunk,
            ram_rw_phase1_num_rounds: self.rw_config.ram_rw_phase1_num_rounds,
            ram_rw_phase2_num_rounds: self.rw_config.ram_rw_phase2_num_rounds,
            registers_rw_phase1_num_rounds: self.rw_config.registers_rw_phase1_num_rounds,
            registers_rw_phase2_num_rounds: self.rw_config.registers_rw_phase2_num_rounds,
            trace_polynomial_order: self.trace_polynomial_order,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct E2eOutlookReport {
    scenario: String,
    mode: OutlookMode,
    fixture: FixtureReport,
    core: Option<RunReport>,
    modular: RunReport,
    comparisons: Vec<ComparisonReport>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct FixtureReport {
    guest: String,
    trace_power: Option<u32>,
    sha2_iterations: Option<u32>,
    samples: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RunReport {
    label: String,
    metrics: RunMetrics,
    proof_shape: Option<ProofShapeReport>,
    stage_timings: Vec<StageTimingReport>,
    verifier_accepted: bool,
    error: Option<String>,
    trace_rows: usize,
    padded_trace_length: usize,
    output_len: usize,
    output_hex: String,
    panic: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct StageTimingReport {
    label: String,
    time_ms: f64,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct ProofShapeReport {
    trace_length: usize,
    ram_k: usize,
    ram_log_k: usize,
    bytecode_k: usize,
    bytecode_log_k: usize,
    instruction_ra_polynomials: usize,
    bytecode_ra_polynomials: usize,
    ram_ra_polynomials: usize,
    committed_polynomials: usize,
    log_k_chunk: u8,
    lookups_ra_virtual_log_k_chunk: u8,
    ram_rw_phase1_num_rounds: u8,
    ram_rw_phase2_num_rounds: u8,
    registers_rw_phase1_num_rounds: u8,
    registers_rw_phase2_num_rounds: u8,
    trace_polynomial_order: TracePolynomialOrder,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ComparisonReport {
    name: String,
    ratio: Option<f64>,
    threshold: Option<f64>,
    passed: Option<bool>,
    note: String,
}

#[cfg(not(feature = "core-fixtures"))]
fn main() -> Result<(), String> {
    Err("e2e_outlook requires --features core-fixtures".to_owned())
}

#[cfg(feature = "core-fixtures")]
fn main() -> Result<(), String> {
    run_with_feature_stack()
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
fn run_with_feature_stack() -> BenchResult<()> {
    let pool = rayon::ThreadPoolBuilder::new()
        .thread_name(|index| format!("e2e-outlook-rayon-{index}"))
        .stack_size(128 * 1024 * 1024)
        .build()
        .map_err(|error| format!("build e2e_outlook Rayon pool: {error}"))?;
    let handle = std::thread::Builder::new()
        .name("e2e-outlook".to_owned())
        .stack_size(128 * 1024 * 1024)
        .spawn(move || pool.install(run))
        .map_err(|error| format!("spawn e2e_outlook: {error}"))?;
    handle.join().map_err(panic_payload_message)?
}

#[cfg(all(feature = "core-fixtures", not(feature = "zk")))]
fn run_with_feature_stack() -> BenchResult<()> {
    run()
}

#[cfg(feature = "core-fixtures")]
fn run() -> BenchResult<()> {
    use jolt_inlines_sha2 as _;

    let config = BenchConfig::from_env()?;
    let mut reports = Vec::new();

    #[cfg(not(feature = "field-inline"))]
    reports.push(run_sha2_chain_core_parity(&config)?);

    #[cfg(feature = "field-inline")]
    {
        reports.push(run_sha2_chain_field_inline_overhead(&config)?);
        reports.push(run_field_inline_eq_poly_native(&config)?);
    }

    let artifact_dir = artifact_dir();
    fs::create_dir_all(&artifact_dir)
        .map_err(|error| format!("create {}: {error}", artifact_dir.display()))?;

    let mut failed = false;
    for report in &reports {
        failed |= report_failed(report);
        let path = artifact_path(report.mode, report.fixture_name());
        write_report(report, &path)?;
        print_report_summary(report, &path);
    }

    if failed {
        return Err("one or more e2e outlook confirmations failed".to_owned());
    }

    Ok(())
}

impl BenchConfig {
    fn from_env() -> BenchResult<Self> {
        let trace_power = env_u32("JOLT_E2E_OUTLOOK_TRACE_POWER")?.unwrap_or(DEFAULT_TRACE_POWER);
        let max_padded_trace_length = 1usize
            .checked_shl(trace_power)
            .ok_or_else(|| format!("trace power {trace_power} overflows usize"))?;
        let samples = env_usize("JOLT_E2E_OUTLOOK_SAMPLES")?.unwrap_or(1).max(1);
        Ok(Self {
            trace_power,
            max_padded_trace_length,
            samples,
        })
    }

    fn sha2_iterations(&self) -> u32 {
        scale_to_target_ops(
            (self.max_padded_trace_length as f64 * SAFETY_MARGIN) as usize,
            CYCLES_PER_SHA256,
        )
    }
}

#[cfg(not(feature = "field-inline"))]
fn run_sha2_chain_core_parity(config: &BenchConfig) -> BenchResult<E2eOutlookReport> {
    let inputs = sha2_chain_inputs(config.sha2_iterations())?;
    let core = run_core_sha2_chain(config, inputs.clone())?;
    let modular = run_modular_sdk_guest(
        SHA2_CHAIN_GUEST,
        inputs,
        false,
        config.max_padded_trace_length,
        config.samples,
        Some(core.proof_shape),
    )?;

    let mut comparisons = Vec::new();
    comparisons.push(perf_comparison(
        "modular_vs_core_prover_time",
        &core.report.metrics,
        &modular.metrics,
        "modular prover time against converted core prover time",
    ));
    comparisons.push(boolean_comparison(
        "public_output_match",
        core.report.output_hex == modular.output_hex && core.report.panic == modular.panic,
        "core and modular public outputs/panic bit match",
    ));

    Ok(E2eOutlookReport {
        scenario: format!("sha2-chain-2^{} core parity", config.trace_power),
        mode: OutlookMode::current(),
        fixture: FixtureReport {
            guest: SHA2_CHAIN_GUEST.to_owned(),
            trace_power: Some(config.trace_power),
            sha2_iterations: Some(config.sha2_iterations()),
            samples: config.samples,
        },
        core: Some(core.report),
        modular,
        comparisons,
    })
}

#[cfg(feature = "field-inline")]
fn run_sha2_chain_field_inline_overhead(config: &BenchConfig) -> BenchResult<E2eOutlookReport> {
    let inputs = sha2_chain_inputs(config.sha2_iterations())?;
    let modular = run_modular_sdk_guest(
        SHA2_CHAIN_GUEST,
        inputs,
        true,
        config.max_padded_trace_length,
        config.samples,
        None,
    )?;
    let mut comparisons = Vec::new();
    comparisons.push(field_inline_baseline_comparison(
        OutlookMode::current(),
        "sha2_chain",
        &modular.metrics,
    ));

    Ok(E2eOutlookReport {
        scenario: format!(
            "sha2-chain-2^{} field-inline protocol overhead",
            config.trace_power
        ),
        mode: OutlookMode::current(),
        fixture: FixtureReport {
            guest: SHA2_CHAIN_GUEST.to_owned(),
            trace_power: Some(config.trace_power),
            sha2_iterations: Some(config.sha2_iterations()),
            samples: config.samples,
        },
        core: None,
        modular,
        comparisons,
    })
}

#[cfg(feature = "field-inline")]
fn run_field_inline_eq_poly_native(config: &BenchConfig) -> BenchResult<E2eOutlookReport> {
    let inputs = postcard::to_stdvec(&7_u32)
        .map_err(|error| format!("serialize field-inline input: {error}"))?;
    let modular = run_modular_sdk_guest(
        FIELD_INLINE_EQ_POLY_GUEST,
        inputs,
        true,
        1 << 16,
        config.samples,
        None,
    )?;
    let comparisons = vec![boolean_comparison(
        "native_verifier_accepts_real_field_inline_guest",
        modular.verifier_accepted && !modular.panic,
        "real field-inline instructions prove and verify natively",
    )];

    Ok(E2eOutlookReport {
        scenario: "field-inline-eq-poly native verifier".to_owned(),
        mode: OutlookMode::current(),
        fixture: FixtureReport {
            guest: FIELD_INLINE_EQ_POLY_GUEST.to_owned(),
            trace_power: None,
            sha2_iterations: None,
            samples: config.samples,
        },
        core: None,
        modular,
        comparisons,
    })
}

#[cfg(all(feature = "core-fixtures", not(feature = "field-inline")))]
struct CoreSha2Run {
    report: RunReport,
    proof_shape: ProofShape,
}

#[cfg(all(feature = "core-fixtures", not(feature = "field-inline")))]
fn run_core_sha2_chain(config: &BenchConfig, inputs: Vec<u8>) -> BenchResult<CoreSha2Run> {
    type CoreField = jolt_core::ark_bn254::Fr;
    type ConvertedProof = ImportedCoreProof<CoreField, Bn254Curve, DoryCommitmentScheme>;

    let mut program = host::Program::new(SHA2_CHAIN_GUEST);
    let jolt_program = program
        .jolt_program()
        .map_err(|error| format!("build core sha2-chain program: {error}"))?;
    let mut tracer_backend = tracer::TracerBackend::new();
    let trace = program
        .trace_with_backend(&mut tracer_backend, &inputs, &[], &[])
        .map_err(|error| format!("trace core sha2-chain guest: {error}"))?;
    let trace_rows = trace.trace.rows().len();
    let padded_trace_length = trace_rows.next_power_of_two().max(2);
    if padded_trace_length > config.max_padded_trace_length {
        return Err(format!(
            "sha2-chain trace pads to {padded_trace_length}, exceeding 2^{}",
            config.trace_power
        ));
    }

    let shared_preprocessing = JoltSharedPreprocessing::new(
        jolt_program.expanded_bytecode.clone(),
        trace.device.memory_layout.clone(),
        jolt_program.memory_init,
        config.max_padded_trace_length,
        jolt_program.entry_address,
    )
    .map_err(|error| format!("preprocess core sha2-chain: {error}"))?;
    let prover_preprocessing: CoreProverPreprocessing<CoreField, Bn254Curve, DoryCommitmentScheme> =
        CoreProverPreprocessing::new(shared_preprocessing);
    let elf_contents = program
        .get_elf_contents()
        .ok_or_else(|| "read core sha2-chain guest ELF: missing ELF contents".to_owned())?;
    let mut durations = Vec::with_capacity(config.samples);
    let mut converted_proof: Option<ConvertedProof> = None;
    let mut public_io: Option<JoltDevice> = None;
    for _ in 0..config.samples {
        let prover = RV64IMACProver::gen_from_elf(
            &prover_preprocessing,
            &elf_contents,
            &inputs,
            &[],
            &[],
            None,
            None,
            None,
        );
        public_io = Some(prover.program_io.clone());
        let start = Instant::now();
        let (proof, _) = prover.prove();
        durations.push(start.elapsed().as_secs_f64() * 1000.0);
        converted_proof = Some(
            proof
                .try_into()
                .map_err(|error| format!("convert core sha2-chain proof: {error}"))?,
        );
    }

    let core_stage_timings = if env_flag("JOLT_E2E_CAPTURE_CORE_TIMINGS")? {
        let prover = RV64IMACProver::gen_from_elf(
            &prover_preprocessing,
            &elf_contents,
            &inputs,
            &[],
            &[],
            None,
            None,
            None,
        );
        let (_proof_and_io, timings) = capture_core_stage_timings(|| prover.prove());
        timings
    } else {
        Vec::new()
    };

    let proof = converted_proof.ok_or_else(|| "core sha2-chain produced no proof".to_owned())?;
    let public_io = public_io.ok_or_else(|| "core sha2-chain produced no public IO".to_owned())?;
    let core_preprocessing = CoreVerifierPreprocessing::from(&prover_preprocessing);
    let preprocessing = convert_core_preprocessing(&core_preprocessing)?;
    jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript<Fr>>(
        &preprocessing,
        &public_io,
        &proof,
        None,
        cfg!(feature = "zk"),
    )
    .map_err(|error| format!("jolt-verifier rejected converted core sha2-chain proof: {error}"))?;

    let formula_dimensions = JoltFormulaDimensions::try_from(proof.one_hot_config.dimensions(
        proof.trace_length.trailing_zeros() as usize,
        RV64_LOOKUP_ADDRESS_BITS,
        preprocessing.program.bytecode.code_size,
        proof.ram_K,
    ))
    .map_err(|error| error.to_string())?;
    let proof_shape = ProofShape {
        trace_length: proof.trace_length,
        ram_k: proof.ram_K,
        bytecode_k: preprocessing.program.bytecode.code_size,
        instruction_ra_polynomials: formula_dimensions.ra_layout.instruction(),
        bytecode_ra_polynomials: formula_dimensions.ra_layout.bytecode(),
        ram_ra_polynomials: formula_dimensions.ra_layout.ram(),
        rw_config: proof.rw_config,
        one_hot_config: proof.one_hot_config,
        trace_polynomial_order: proof.trace_polynomial_order,
    };
    Ok(CoreSha2Run {
        report: run_report(
            "core",
            average_ms(&durations),
            trace_rows,
            padded_trace_length,
            &public_io,
        )
        .with_proof_shape(proof_shape)
        .with_stage_timings(core_stage_timings),
        proof_shape,
    })
}

#[cfg(all(feature = "core-fixtures", not(feature = "field-inline")))]
fn convert_core_preprocessing(
    preprocessing: &CoreVerifierPreprocessing<
        jolt_core::ark_bn254::Fr,
        Bn254Curve,
        DoryCommitmentScheme,
    >,
) -> BenchResult<JoltVerifierPreprocessing<DoryScheme, Pedersen<Bn254G1>>> {
    Ok(JoltVerifierPreprocessing::new(
        jolt_program::preprocess::JoltProgramPreprocessing {
            bytecode: preprocessing.shared.bytecode.as_ref().clone(),
            ram: preprocessing.shared.ram.clone(),
            memory_layout: preprocessing.shared.memory_layout.clone(),
            max_padded_trace_length: preprocessing.shared.max_padded_trace_length,
        },
        preprocessing.shared.digest(),
        DoryVerifierSetup(preprocessing.generators.clone()),
        convert_core_vc_setup(preprocessing)?,
    ))
}

#[cfg(all(
    feature = "core-fixtures",
    feature = "zk",
    not(feature = "field-inline")
))]
fn convert_core_vc_setup(
    preprocessing: &CoreVerifierPreprocessing<
        jolt_core::ark_bn254::Fr,
        Bn254Curve,
        DoryCommitmentScheme,
    >,
) -> BenchResult<Option<PedersenSetup<Bn254G1>>> {
    let setup = &preprocessing
        .blindfold_setup
        .as_ref()
        .ok_or_else(|| "missing core BlindFold setup".to_owned())?
        .0;
    Ok(Some(PedersenSetup::new(
        setup
            .message_generators
            .iter()
            .copied()
            .map(<Bn254Curve as CoreCurveBridge<jolt_core::ark_bn254::Fr>>::g1_into_verifier)
            .collect(),
        <Bn254Curve as CoreCurveBridge<jolt_core::ark_bn254::Fr>>::g1_into_verifier(
            setup.blinding_generator,
        ),
    )))
}

#[cfg(all(
    feature = "core-fixtures",
    not(feature = "zk"),
    not(feature = "field-inline")
))]
fn convert_core_vc_setup(
    _preprocessing: &CoreVerifierPreprocessing<
        jolt_core::ark_bn254::Fr,
        Bn254Curve,
        DoryCommitmentScheme,
    >,
) -> BenchResult<Option<jolt_crypto::PedersenSetup<Bn254G1>>> {
    Ok(None)
}

#[cfg(all(feature = "core-fixtures", not(feature = "field-inline")))]
#[derive(Clone)]
struct CoreTimingLayer {
    records: Arc<Mutex<Vec<StageTimingReport>>>,
}

#[cfg(all(feature = "core-fixtures", not(feature = "field-inline")))]
struct CoreSpanTiming {
    label: &'static str,
    start: Instant,
}

#[cfg(all(feature = "core-fixtures", not(feature = "field-inline")))]
impl<S> Layer<S> for CoreTimingLayer
where
    S: Subscriber + for<'lookup> LookupSpan<'lookup>,
{
    fn on_new_span(&self, attrs: &tracing::span::Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let Some(label) = core_span_label(attrs.metadata().name()) else {
            return;
        };
        let Some(span) = ctx.span(id) else {
            return;
        };
        span.extensions_mut().insert(CoreSpanTiming {
            label,
            start: Instant::now(),
        });
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let Some(span) = ctx.span(&id) else {
            return;
        };
        let Some(timing) = span.extensions_mut().remove::<CoreSpanTiming>() else {
            return;
        };
        let Ok(mut records) = self.records.lock() else {
            return;
        };
        records.push(StageTimingReport {
            label: timing.label.to_owned(),
            time_ms: timing.start.elapsed().as_secs_f64() * 1000.0,
        });
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "field-inline")))]
fn capture_core_stage_timings<R>(f: impl FnOnce() -> R) -> (R, Vec<StageTimingReport>) {
    let records = Arc::new(Mutex::new(Vec::new()));
    let layer = CoreTimingLayer {
        records: Arc::clone(&records),
    };
    let subscriber = tracing_subscriber::registry().with(layer);
    let result = tracing::subscriber::with_default(subscriber, f);
    let stage_timings = records
        .lock()
        .map(|records| records.clone())
        .unwrap_or_default();
    (result, stage_timings)
}

#[cfg(all(feature = "core-fixtures", not(feature = "field-inline")))]
fn core_span_label(name: &str) -> Option<&'static str> {
    match name {
        "generate_and_commit_witness_polynomials" => Some("core.stage0"),
        "prove_stage1" => Some("core.stage1"),
        "prove_stage2" => Some("core.stage2"),
        "prove_stage3" => Some("core.stage3"),
        "prove_stage4" => Some("core.stage4"),
        "prove_stage5" => Some("core.stage5"),
        "prove_stage6" => Some("core.stage6"),
        "prove_stage7" => Some("core.stage7"),
        "prove_stage8" => Some("core.stage8"),
        "prove_blindfold" => Some("core.blindfold_prove"),
        "DoryCommitmentScheme::combine_commitments" => Some("core.stage8.combine_commitments"),
        "DoryCommitmentScheme::combine_hints" => Some("core.stage8.combine_hints"),
        "DoryCommitmentScheme::prove" => Some("core.stage8.open_poly"),
        _ => None,
    }
}

fn run_modular_sdk_guest(
    guest: &'static str,
    inputs: Vec<u8>,
    field_inline: bool,
    max_padded_trace_length: usize,
    samples: usize,
    proof_shape: Option<ProofShape>,
) -> BenchResult<RunReport> {
    let fixture = trace_sdk_guest(
        SdkGuestTraceRequest::new(guest, inputs)
            .with_field_inline(field_inline)
            .with_max_padded_trace_length(max_padded_trace_length),
    )
    .map_err(|error| error.to_string())?;

    let proof_shape = match proof_shape {
        Some(proof_shape) => proof_shape,
        None => derive_proof_shape(&fixture)?,
    };
    let witness_config = JoltVmWitnessConfig::new(
        proof_shape.trace_length.trailing_zeros() as usize,
        proof_shape.ram_k,
        proof_shape.one_hot_config,
    );
    let witness = TraceBackedJoltVmWitness::new(
        witness_config,
        JoltVmWitnessInputs::new(&fixture.program, &fixture.preprocessing, fixture.trace),
    );
    let public_io = witness.trace.device.clone();
    let committed_chunk_bits = proof_shape.one_hot_config.committed_chunk_bits();
    let mut max_num_vars =
        proof_shape.trace_length.trailing_zeros() as usize + committed_chunk_bits;
    let vc_capacity = zk_vector_commitment_capacity();
    if let Some(vc_capacity) = vc_capacity {
        max_num_vars = max_num_vars.max(vc_capacity.ilog2() as usize);
    }
    let pcs_setup = DoryScheme::setup_prover(max_num_vars);
    let verifier_preprocessing =
        modular_verifier_preprocessing(&fixture.preprocessing, &pcs_setup, vc_capacity)?;
    let prover_preprocessing =
        jolt_prover::JoltProverPreprocessing::new(verifier_preprocessing.clone(), pcs_setup);
    let config =
        jolt_prover::ProverConfig::default().with_proof_shape(jolt_prover::ProverProofShape::new(
            proof_shape.trace_length,
            proof_shape.ram_k,
            proof_shape.rw_config,
            proof_shape.one_hot_config,
            proof_shape.trace_polynomial_order,
        ));

    #[cfg(feature = "field-inline")]
    let field_witness = witness
        .field_inline_witness()
        .map_err(|error| format!("build field-inline witness for {guest}: {error}"))?;

    let mut durations = Vec::with_capacity(samples);
    let mut verified = false;
    let mut last_stage_timings = Vec::new();
    for _ in 0..samples {
        let mut backend = CpuBackend::new(CpuBackendConfig {
            preserve_core_fast_path: true,
            commitment_chunk_size: 1 << 10,
        });
        jolt_prover::reset_stage_timings();
        #[cfg(feature = "core-fixtures")]
        jolt_backends::reset_backend_timings();
        let start = Instant::now();
        #[cfg(not(feature = "field-inline"))]
        let output_result = jolt_prover::prove_with_output::<DoryScheme, Pedersen<Bn254G1>, _, _>(
            &prover_preprocessing,
            &public_io,
            &witness,
            config,
            &mut backend,
        );
        #[cfg(feature = "field-inline")]
        let output_result = jolt_prover::prove_with_output::<DoryScheme, Pedersen<Bn254G1>, _, _, _>(
            &prover_preprocessing,
            &public_io,
            &witness,
            &field_witness,
            config,
            &mut backend,
        );
        durations.push(start.elapsed().as_secs_f64() * 1000.0);
        last_stage_timings = jolt_prover::take_stage_timings()
            .into_iter()
            .map(|timing| StageTimingReport {
                label: timing.label.to_owned(),
                time_ms: timing.time_ms,
            })
            .collect();
        #[cfg(feature = "core-fixtures")]
        last_stage_timings.extend(jolt_backends::take_backend_timings().into_iter().map(
            |timing| StageTimingReport {
                label: timing.label.to_owned(),
                time_ms: timing.time_ms,
            },
        ));

        let output = match output_result {
            Ok(output) => output,
            Err(error) => {
                return Ok(run_report(
                    "jolt-prover",
                    average_ms(&durations),
                    witness.trace.trace.rows().len(),
                    proof_shape.trace_length,
                    &public_io,
                )
                .with_proof_shape(proof_shape)
                .with_verifier_accepted(false)
                .with_stage_timings(last_stage_timings)
                .with_error(format!("prove modular {guest}: {error}")));
            }
        };

        if let Err(error) =
            jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript<Fr>>(
                &verifier_preprocessing,
                &public_io,
                &output.proof,
                output.trusted_advice_commitment.as_ref(),
                cfg!(feature = "zk"),
            )
        {
            return Ok(run_report(
                "jolt-prover",
                average_ms(&durations),
                witness.trace.trace.rows().len(),
                proof_shape.trace_length,
                &public_io,
            )
            .with_proof_shape(proof_shape)
            .with_verifier_accepted(false)
            .with_stage_timings(last_stage_timings)
            .with_error(format!("jolt-verifier rejected modular {guest}: {error}")));
        }
        verified = true;
    }

    Ok(run_report(
        "jolt-prover",
        average_ms(&durations),
        witness.trace.trace.rows().len(),
        proof_shape.trace_length,
        &public_io,
    )
    .with_proof_shape(proof_shape)
    .with_verifier_accepted(verified)
    .with_stage_timings(last_stage_timings))
}

fn derive_proof_shape(
    fixture: &jolt_prover_harness::SdkGuestTraceFixture,
) -> BenchResult<ProofShape> {
    let log_t = fixture.padded_trace_length.trailing_zeros() as usize;
    let one_hot_config = default_one_hot_config(log_t);
    let bytecode_start = fixture
        .preprocessing
        .memory_layout
        .remapped_word_address(fixture.preprocessing.ram.min_bytecode_address)
        .map_err(|error| error.to_string())? as usize;
    let bytecode_end = bytecode_start + fixture.preprocessing.ram.bytecode_words.len() + 1;
    let mut ram_rows = bytecode_end;
    for row in fixture.trace.trace.rows() {
        let Some(address) = ram_access_address(row.ram_access) else {
            continue;
        };
        let Some(remapped) = fixture
            .preprocessing
            .memory_layout
            .remap_word_address(address)
            .map_err(|error| error.to_string())?
        else {
            continue;
        };
        ram_rows = ram_rows.max(remapped as usize + 1);
    }
    let ram_k = ram_rows.next_power_of_two().max(1);
    let log_k = ram_k.trailing_zeros() as usize;
    let formula_dimensions = JoltFormulaDimensions::try_from(one_hot_config.dimensions(
        log_t,
        RV64_LOOKUP_ADDRESS_BITS,
        fixture.preprocessing.bytecode.code_size,
        ram_k,
    ))
    .map_err(|error| error.to_string())?;
    Ok(ProofShape {
        trace_length: fixture.padded_trace_length,
        ram_k,
        bytecode_k: fixture.preprocessing.bytecode.code_size,
        instruction_ra_polynomials: formula_dimensions.ra_layout.instruction(),
        bytecode_ra_polynomials: formula_dimensions.ra_layout.bytecode(),
        ram_ra_polynomials: formula_dimensions.ra_layout.ram(),
        rw_config: JoltReadWriteConfig {
            ram_rw_phase1_num_rounds: log_t as u8,
            ram_rw_phase2_num_rounds: log_k as u8,
            registers_rw_phase1_num_rounds: log_t as u8,
            registers_rw_phase2_num_rounds: REGISTER_COUNT.ilog2() as u8,
        },
        one_hot_config,
        trace_polynomial_order: TracePolynomialOrder::CycleMajor,
    })
}

fn default_one_hot_config(log_t: usize) -> JoltOneHotConfig {
    let log_k_chunk = if log_t < ONEHOT_CHUNK_THRESHOLD_LOG_T {
        4
    } else {
        8
    };
    let lookups_ra_virtual_log_k_chunk = if log_t < ONEHOT_CHUNK_THRESHOLD_LOG_T {
        RV64_LOOKUP_ADDRESS_BITS / 8
    } else {
        RV64_LOOKUP_ADDRESS_BITS / 4
    };
    JoltOneHotConfig {
        log_k_chunk: log_k_chunk as u8,
        lookups_ra_virtual_log_k_chunk: lookups_ra_virtual_log_k_chunk as u8,
    }
}

fn ram_access_address(access: RamAccess) -> Option<u64> {
    match access {
        RamAccess::Read(read) => Some(read.address),
        RamAccess::Write(write) => Some(write.address),
        RamAccess::NoOp => None,
    }
}

fn modular_verifier_preprocessing(
    preprocessing: &jolt_program::preprocess::JoltProgramPreprocessing,
    pcs_setup: &jolt_dory::DoryProverSetup,
    vc_capacity: Option<usize>,
) -> BenchResult<JoltVerifierPreprocessing<DoryScheme, Pedersen<Bn254G1>>> {
    let verifier_preprocessing = match vc_capacity {
        Some(capacity) => {
            JoltVerifierPreprocessing::<DoryScheme, Pedersen<Bn254G1>>::from_pcs_prover_setup(
                preprocessing.clone(),
                [7; 32],
                pcs_setup,
                capacity,
            )
        }
        None => JoltVerifierPreprocessing::<DoryScheme, Pedersen<Bn254G1>>::new(
            preprocessing.clone(),
            [7; 32],
            DoryScheme::verifier_setup(pcs_setup),
            None,
        ),
    };
    with_field_inline_bytecode(verifier_preprocessing, preprocessing)
}

#[cfg(feature = "field-inline")]
fn with_field_inline_bytecode(
    preprocessing: JoltVerifierPreprocessing<DoryScheme, Pedersen<Bn254G1>>,
    program: &JoltProgramPreprocessing,
) -> BenchResult<JoltVerifierPreprocessing<DoryScheme, Pedersen<Bn254G1>>> {
    Ok(preprocessing.with_field_inline_bytecode(verifier_field_inline_bytecode_rows(program)?))
}

#[cfg(not(feature = "field-inline"))]
fn with_field_inline_bytecode(
    preprocessing: JoltVerifierPreprocessing<DoryScheme, Pedersen<Bn254G1>>,
    _program: &jolt_program::preprocess::JoltProgramPreprocessing,
) -> BenchResult<JoltVerifierPreprocessing<DoryScheme, Pedersen<Bn254G1>>> {
    Ok(preprocessing)
}

#[cfg(feature = "field-inline")]
fn verifier_field_inline_bytecode_rows(
    preprocessing: &JoltProgramPreprocessing,
) -> BenchResult<
    Vec<jolt_claims::protocols::field_inline::formulas::bytecode::FieldInlineBytecodeRow>,
> {
    preprocessing
        .bytecode
        .bytecode
        .iter()
        .map(verifier_field_inline_bytecode_row)
        .collect()
}

#[cfg(feature = "field-inline")]
fn verifier_field_inline_bytecode_row(
    instruction: &jolt_riscv::JoltInstructionRow,
) -> BenchResult<jolt_claims::protocols::field_inline::formulas::bytecode::FieldInlineBytecodeRow> {
    use jolt_claims::protocols::field_inline::formulas::bytecode::{
        FieldInlineBytecodeFlags, FieldInlineBytecodeOperands, FieldInlineBytecodeRow,
    };

    let operands = FieldInlineBytecodeOperands {
        rd: instruction.operands.rd,
        rs1: instruction.operands.rs1,
        rs2: instruction.operands.rs2,
    };
    let mut row = FieldInlineBytecodeRow::default();
    match instruction.instruction_kind {
        JoltInstructionKind::NoOp => {}
        JoltInstructionKind::FIELD_ADD => {
            row.operands = operands;
            row.flags = FieldInlineBytecodeFlags {
                add: true,
                ..Default::default()
            };
        }
        JoltInstructionKind::FIELD_SUB => {
            row.operands = operands;
            row.flags = FieldInlineBytecodeFlags {
                sub: true,
                ..Default::default()
            };
        }
        JoltInstructionKind::FIELD_MUL => {
            row.operands = operands;
            row.flags = FieldInlineBytecodeFlags {
                mul: true,
                ..Default::default()
            };
        }
        JoltInstructionKind::FIELD_INV => {
            row.operands = FieldInlineBytecodeOperands {
                rd: instruction.operands.rd,
                rs1: instruction.operands.rs1,
                rs2: None,
            };
            row.flags = FieldInlineBytecodeFlags {
                inv: true,
                ..Default::default()
            };
        }
        JoltInstructionKind::FIELD_ASSERT_EQ => {
            row.operands = FieldInlineBytecodeOperands {
                rd: None,
                rs1: instruction.operands.rs1,
                rs2: instruction.operands.rs2,
            };
            row.flags = FieldInlineBytecodeFlags {
                assert_eq: true,
                ..Default::default()
            };
        }
        JoltInstructionKind::FIELD_LOAD_FROM_X => {
            row.operands = FieldInlineBytecodeOperands {
                rd: instruction.operands.rd,
                rs1: None,
                rs2: None,
            };
            row.flags = FieldInlineBytecodeFlags {
                load_from_x: true,
                ..Default::default()
            };
        }
        JoltInstructionKind::FIELD_STORE_TO_X => {
            row.operands = FieldInlineBytecodeOperands {
                rd: None,
                rs1: instruction.operands.rs1,
                rs2: None,
            };
            row.flags = FieldInlineBytecodeFlags {
                store_to_x: true,
                ..Default::default()
            };
        }
        JoltInstructionKind::FIELD_LOAD_IMM => {
            row.operands = FieldInlineBytecodeOperands {
                rd: instruction.operands.rd,
                rs1: None,
                rs2: None,
            };
            row.flags = FieldInlineBytecodeFlags {
                load_imm: true,
                ..Default::default()
            };
        }
        _ => {}
    }
    Ok(row)
}

#[cfg(feature = "zk")]
fn zk_vector_commitment_capacity() -> Option<usize> {
    Some(MAX_BLINDFOLD_GENERATORS.max(64))
}

#[cfg(not(feature = "zk"))]
fn zk_vector_commitment_capacity() -> Option<usize> {
    None
}

fn run_report(
    label: &str,
    time_ms: Option<f64>,
    trace_rows: usize,
    padded_trace_length: usize,
    public_io: &JoltDevice,
) -> RunReport {
    RunReport {
        label: label.to_owned(),
        metrics: RunMetrics::new(time_ms, None, None),
        proof_shape: None,
        stage_timings: Vec::new(),
        verifier_accepted: true,
        error: None,
        trace_rows,
        padded_trace_length,
        output_len: public_io.outputs.len(),
        output_hex: bytes_hex(&public_io.outputs),
        panic: public_io.panic,
    }
}

trait RunReportExt {
    fn with_proof_shape(self, proof_shape: ProofShape) -> Self;
    fn with_verifier_accepted(self, verifier_accepted: bool) -> Self;
    fn with_stage_timings(self, stage_timings: Vec<StageTimingReport>) -> Self;
    fn with_error(self, error: String) -> Self;
}

impl RunReportExt for RunReport {
    fn with_proof_shape(mut self, proof_shape: ProofShape) -> Self {
        self.proof_shape = Some(proof_shape.report());
        self
    }

    fn with_verifier_accepted(mut self, verifier_accepted: bool) -> Self {
        self.verifier_accepted = verifier_accepted;
        self
    }

    fn with_stage_timings(mut self, stage_timings: Vec<StageTimingReport>) -> Self {
        self.stage_timings = stage_timings;
        self
    }

    fn with_error(mut self, error: String) -> Self {
        self.error = Some(error);
        self
    }
}

fn perf_comparison(
    name: &str,
    baseline: &RunMetrics,
    candidate: &RunMetrics,
    note: &str,
) -> ComparisonReport {
    let gate = PerfGate {
        warn_ratio: 1.05,
        fail_ratio: PARITY_FAIL_RATIO,
        min_samples: 1,
        confirmation_size: None,
        require_time: true,
        require_peak_rss: false,
    };
    let evaluation = evaluate_perf(gate, baseline, candidate);
    ComparisonReport {
        name: name.to_owned(),
        ratio: evaluation.time_ratio,
        threshold: Some(PARITY_FAIL_RATIO),
        passed: evaluation
            .time_ratio
            .map(|ratio| ratio <= PARITY_FAIL_RATIO),
        note: format!("{note}; status={:?}", evaluation.status),
    }
}

fn boolean_comparison(name: &str, passed: bool, note: &str) -> ComparisonReport {
    ComparisonReport {
        name: name.to_owned(),
        ratio: None,
        threshold: None,
        passed: Some(passed),
        note: note.to_owned(),
    }
}

#[cfg(feature = "field-inline")]
fn field_inline_baseline_comparison(
    mode: OutlookMode,
    fixture_name: &str,
    candidate: &RunMetrics,
) -> ComparisonReport {
    let Some(baseline_mode) = mode.baseline_without_field_inline() else {
        return ComparisonReport {
            name: "field_inline_vs_baseline".to_owned(),
            ratio: None,
            threshold: Some(PARITY_FAIL_RATIO),
            passed: None,
            note: "mode has no non-field-inline baseline".to_owned(),
        };
    };
    let path = artifact_path(baseline_mode, fixture_name);
    let baseline = fs::read_to_string(&path)
        .map_err(|error| format!("read {}: {error}", path.display()))
        .and_then(|json| {
            serde_json::from_str::<E2eOutlookReport>(&json)
                .map_err(|error| format!("parse {}: {error}", path.display()))
        });

    match baseline {
        Ok(report) => perf_comparison(
            "field_inline_vs_non_field_inline_prover_time",
            &report.modular.metrics,
            candidate,
            &format!("compared against {}", path.display()),
        ),
        Err(error) => ComparisonReport {
            name: "field_inline_vs_non_field_inline_prover_time".to_owned(),
            ratio: None,
            threshold: Some(PARITY_FAIL_RATIO),
            passed: None,
            note: format!("baseline unavailable: {error}"),
        },
    }
}

fn sha2_chain_inputs(iterations: u32) -> BenchResult<Vec<u8>> {
    let mut inputs = postcard::to_stdvec(&[5_u8; 32])
        .map_err(|error| format!("serialize sha2 input: {error}"))?;
    inputs.extend(
        postcard::to_stdvec(&iterations)
            .map_err(|error| format!("serialize sha2 iterations: {error}"))?,
    );
    Ok(inputs)
}

fn scale_to_target_ops(target_cycles: usize, cycles_per_op: f64) -> u32 {
    u32::try_from((target_cycles as f64 / cycles_per_op) as usize)
        .unwrap_or(u32::MAX)
        .max(1)
}

fn average_ms(samples: &[f64]) -> Option<f64> {
    if samples.is_empty() {
        None
    } else {
        Some(samples.iter().sum::<f64>() / samples.len() as f64)
    }
}

fn artifact_dir() -> PathBuf {
    workspace_root()
        .join("target")
        .join("frontier-metrics")
        .join("e2e-outlook")
}

fn artifact_path(mode: OutlookMode, fixture_name: &str) -> PathBuf {
    artifact_dir().join(format!("{}_{}.json", mode.slug(), fixture_name))
}

fn workspace_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("jolt-prover-harness is nested two levels below workspace root")
        .to_path_buf()
}

fn write_report(report: &E2eOutlookReport, path: &Path) -> BenchResult<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|error| format!("create {}: {error}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(report)
        .map_err(|error| format!("serialize e2e outlook report: {error}"))?;
    fs::write(path, json).map_err(|error| format!("write {}: {error}", path.display()))
}

fn report_failed(report: &E2eOutlookReport) -> bool {
    !report.modular.verifier_accepted
        || report.modular.error.is_some()
        || report
            .core
            .as_ref()
            .is_some_and(|core| !core.verifier_accepted)
        || report
            .core
            .as_ref()
            .is_some_and(|core| core.error.is_some())
        || report
            .comparisons
            .iter()
            .any(|comparison| comparison.passed == Some(false))
}

fn print_report_summary(report: &E2eOutlookReport, path: &Path) {
    let mut summary = format!(
        "{} [{}] wrote {}\n",
        report.scenario,
        report.mode.slug(),
        path.display()
    );
    if let Some(core) = &report.core {
        summary.push_str(&format!(
            "  core:    {:>10.2?} ms, verifier={}, error={:?}\n",
            core.metrics.time_ms, core.verifier_accepted, core.error
        ));
    }
    summary.push_str(&format!(
        "  modular: {:>10.2?} ms, verifier={}, error={:?}\n",
        report.modular.metrics.time_ms, report.modular.verifier_accepted, report.modular.error
    ));
    for comparison in &report.comparisons {
        summary.push_str(&format!(
            "  {}: ratio={:?}, passed={:?} ({})\n",
            comparison.name, comparison.ratio, comparison.passed, comparison.note
        ));
    }
    let _ = emit_stdout(&summary);
}

impl E2eOutlookReport {
    fn fixture_name(&self) -> &'static str {
        if self.fixture.guest == SHA2_CHAIN_GUEST {
            "sha2_chain"
        } else {
            "field_inline_eq_poly"
        }
    }
}

fn bytes_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn env_u32(name: &str) -> BenchResult<Option<u32>> {
    match env::var(name) {
        Ok(value) => value
            .parse::<u32>()
            .map(Some)
            .map_err(|error| format!("parse {name}={value:?}: {error}")),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(format!("read {name}: {error}")),
    }
}

fn env_usize(name: &str) -> BenchResult<Option<usize>> {
    match env::var(name) {
        Ok(value) => value
            .parse::<usize>()
            .map(Some)
            .map_err(|error| format!("parse {name}={value:?}: {error}")),
        Err(env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(format!("read {name}: {error}")),
    }
}

#[cfg(all(feature = "core-fixtures", not(feature = "field-inline")))]
fn env_flag(name: &str) -> BenchResult<bool> {
    match env::var(name) {
        Ok(value) => match value.as_str() {
            "1" | "true" | "TRUE" | "yes" | "YES" => Ok(true),
            "0" | "false" | "FALSE" | "no" | "NO" => Ok(false),
            _ => Err(format!("parse {name}={value:?}: expected boolean flag")),
        },
        Err(env::VarError::NotPresent) => Ok(false),
        Err(error) => Err(format!("read {name}: {error}")),
    }
}

fn emit_stdout(message: &str) -> io::Result<()> {
    io::stdout().write_all(message.as_bytes())
}

#[cfg(all(feature = "core-fixtures", feature = "zk"))]
fn panic_payload_message(payload: Box<dyn std::any::Any + Send>) -> String {
    if let Some(message) = payload.downcast_ref::<&str>() {
        format!("e2e_outlook panicked: {message}")
    } else if let Some(message) = payload.downcast_ref::<String>() {
        format!("e2e_outlook panicked: {message}")
    } else {
        "e2e_outlook panicked with non-string payload".to_owned()
    }
}
