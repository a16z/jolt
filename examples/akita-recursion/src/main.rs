mod cycle_report;
use cycle_report::{Capture, CycleReport};

use std::{
    error::Error,
    fs,
    path::{Path, PathBuf},
    result::Result as StdResult,
    sync::Arc,
    thread::Builder,
};

use akita_types::setup_claims::{SetupBatchStatement, SetupPolynomialId, SetupQuery};
use clap::{Parser, Subcommand};
use common::jolt_device::{JoltDevice, MemoryConfig};
use jolt_akita::deferred::{InheritedSetupClaims, SetupClaimAdvice, SetupMode};
use jolt_akita::{
    AkitaChunkProfile, AkitaError, AkitaField, AkitaProverSetup, AkitaScheduleArtifacts,
    AkitaScheme,
};
use jolt_akita_recursion::{decode_pair, AnswerTape, InnerProof, ProofPair, Vc};
use jolt_host::{JoltProgramSource, Program};
use jolt_inlines_sha2 as _;
use jolt_openings::CommitmentScheme;
use jolt_program::execution::{ExecutionBackend, JoltProgram, TraceInputs, TraceOutput};
use jolt_program::preprocess::JoltProgramPreprocessing;
use jolt_prover::{
    akita::{
        self,
        preprocessing::{self, AkitaProverPreprocessing},
    },
    JoltBackend, ProverConfig,
};
use jolt_transcript::{Blake2b512, ProverState, VerifierState};
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
use serde::Serialize;
use tracer::execution_backend::TracerBackend;
use tracing_subscriber::EnvFilter;

type Result<T> = StdResult<T, Box<dyn Error>>;

#[derive(Parser)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Prove two different SHA-256 chains, requiring exactly the requested padded trace.
    Generate {
        #[arg(long, default_value_t = 26)]
        trace_log: u32,
        #[arg(long)]
        schedules: PathBuf,
        #[arg(long, default_value = "benchmark-runs/akita-recursion/pair.bin")]
        output: PathBuf,
        #[arg(long)]
        iterations: Option<u32>,
    },
    /// Re-embed the same direct-proof queries in a larger outer setup prefix.
    Rebind {
        #[arg(long)]
        pair: PathBuf,
        #[arg(long)]
        setup_log: u32,
        #[arg(long)]
        output: PathBuf,
    },
    /// Execute both recursion guests without retaining their trace rows.
    Trace {
        #[arg(long)]
        pair: PathBuf,
        #[arg(long)]
        report: PathBuf,
    },
}

struct NativeAdvice<'a> {
    setup: &'a AkitaProverSetup,
    polynomial: SetupPolynomialId,
    answers: Vec<AkitaField>,
}

impl SetupClaimAdvice<AkitaField> for NativeAdvice<'_> {
    fn answer(&mut self, query: &SetupQuery<AkitaField>) -> StdResult<AkitaField, AkitaError> {
        let value = self.setup.prepare_setup_answer(&self.polynomial, query)?;
        self.answers.push(value);
        Ok(value)
    }
}

fn artifacts(
    path: &Path,
    config: &ProverConfig,
    program: &JoltProgramPreprocessing,
    directory: &Path,
    direct: bool,
) -> Result<Arc<AkitaScheduleArtifacts>> {
    use akita_params::PolynomialGroupLayout;
    use akita_planner::emit::{
        publish_artifact_outputs, render_schedule_artifact_outputs_with_validation,
        MaterializationDiagnostics,
    };
    use jolt_akita::{schedules::emit::family_specs, AkitaOneHotScheduleArtifact};
    let (shape, _, k) = akita::one_hot_trace_setup_shape(config, program.bytecode.code_size)?;
    fs::create_dir_all(directory)?;
    let filename = format!("jolt-fp128-onehot-k{k}.aks");
    let mut fixture = family_specs(directory.to_path_buf())?
        .into_iter()
        .find(|fixture| {
            fixture
                .spec
                .output_path
                .file_name()
                .is_some_and(|name| name == filename.as_str())
        })
        .ok_or("missing one-hot family")?;
    fixture.spec.keys = vec![PolynomialGroupLayout::new(shape.num_vars, shape.num_polys)];
    let spec = if direct {
        fixture.prepare_direct()?
    } else {
        fixture.prepare()?
    };
    println!(
        "planning {}x{} {} catalog",
        shape.num_vars,
        shape.num_polys,
        if direct { "direct" } else { "recursive" }
    );
    let outputs = render_schedule_artifact_outputs_with_validation(
        &[spec],
        MaterializationDiagnostics {
            row_progress: true,
            ..Default::default()
        },
        |_, _| Ok(()),
    )?;
    publish_artifact_outputs(outputs)?;
    Ok(Arc::new(AkitaScheduleArtifacts::new(
        AkitaChunkProfile::new(8, 2),
        fs::read(path.join("jolt-fp128-dense-bounded.aks"))?,
        fs::read(path.join("jolt-fp128-dense-full.aks"))?,
        vec![AkitaOneHotScheduleArtifact {
            k,
            profile: config.akita_chunk_profile,
            bytes: fs::read(directory.join(filename))?,
        }],
    )?))
}

fn memory(source: &mut Program) -> MemoryConfig {
    let (_, _, program_size, _) = source.decode();
    source.memory_config_with_program_size(program_size)
}

fn prepare(
    source: &mut Program,
    input: Vec<u8>,
    bound: usize,
) -> Result<(
    Arc<JoltProgram>,
    JoltProgramPreprocessing,
    TraceOutput,
    ProverConfig,
)> {
    let memory = memory(source);
    let program = Arc::new(source.build_jolt_program()?);
    let trace = TracerBackend::new().trace(
        &program,
        TraceInputs::new(input, Vec::new(), Vec::new(), memory),
    )?;
    if trace.device.panic {
        return Err("guest panicked".into());
    }
    let program_preprocessing = JoltProgramPreprocessing::new(
        program.expanded_bytecode.clone(),
        program.memory_init.clone(),
        trace.device.memory_layout.clone(),
        program.entry_address,
        bound,
        source.instruction_profile(),
    )?;
    let config = ProverConfig::derive_from_dimensions::<AkitaField>(
        trace.dimensions,
        &trace.device.memory_layout,
        program_preprocessing.ram.min_bytecode_address,
        program_preprocessing.ram.bytecode_words.len(),
        bound,
    )?;
    Ok((program, program_preprocessing, trace, config))
}

fn prove(
    program: &Arc<JoltProgram>,
    trace: TraceOutput,
    config: &ProverConfig,
    preprocessing: &AkitaProverPreprocessing,
) -> Result<InnerProof> {
    let io = trace.device.clone();
    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(
            config.trace_length.ilog2() as usize,
            config.ram_K,
            config.one_hot_config,
        ),
        JoltVmWitnessInputs::new(
            program,
            &preprocessing
                .program_arc()
                .ok_or("full preprocessing required")?,
            trace,
        ),
    );
    let proof = akita::prove::<AkitaField, AkitaScheme, Vc, ProverState<Blake2b512>>(
        JoltBackend::optimized().with_witness(&witness),
        preprocessing,
        config,
        None,
        &io,
    )?;
    Ok(InnerProof { proof, io })
}

fn save<T: Serialize>(path: &Path, value: &T) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(
        path,
        bincode::serde::encode_to_vec(value, bincode::config::standard())?,
    )?;
    Ok(())
}

fn generate(
    trace_log: u32,
    schedules: &Path,
    output: &Path,
    iterations: Option<u32>,
) -> Result<()> {
    let bound = 1usize
        .checked_shl(trace_log)
        .ok_or("trace exponent too large")?;
    let iterations = iterations.unwrap_or(u32::try_from((bound * 9 / 10 / 3396).max(1))?);
    let mut source = Program::new("sha2-chain-guest");
    source.set_heap_size(32768);
    let mut preprocessing = None;
    let mut proofs = Vec::new();
    for seed in [5u8, 6] {
        let input = [
            postcard::to_stdvec(&[seed; 32])?,
            postcard::to_stdvec(&iterations)?,
        ]
        .concat();
        let (program, program_preprocessing, trace, config) = prepare(&mut source, input, bound)?;
        if config.trace_length != bound {
            return Err(format!(
                "actual padded trace {} differs from requested {bound}; adjust --iterations",
                config.trace_length
            )
            .into());
        }
        println!(
            "inner seed={seed} trace_rows={} padded_trace={}",
            trace.dimensions.trace_length, config.trace_length
        );
        if preprocessing.is_none() {
            let catalogs = artifacts(
                schedules,
                &config,
                &program_preprocessing,
                &output.with_extension("catalogs"),
                true,
            )?;
            let mut prepared =
                preprocessing::preprocess_full(&catalogs, program_preprocessing, &config)?;
            let domain_len = prepared.pcs_setup.required_setup_domain_len()?;
            prepared.pcs_setup = prepared.pcs_setup.with_setup_domain(domain_len)?;
            prepared.verifier.pcs_setup = AkitaScheme::verifier_setup(&prepared.pcs_setup);
            preprocessing = Some(prepared);
        }
        let prepared = preprocessing.as_ref().ok_or("missing preprocessing")?;
        let inner = prove(&program, trace, &config, prepared)?;
        if !prepared
            .verifier
            .pcs_setup
            .setup_offloads(&inner.proof.joint_opening_proof)?
            .is_empty()
        {
            return Err("inner proof uses setup offloading; supply a --direct catalog".into());
        }
        jolt_verifier::verify::<AkitaField, AkitaScheme, Vc, VerifierState<Blake2b512>>(
            &prepared.verifier,
            &inner.io,
            &inner.proof,
            None,
        )?;
        println!("inner seed={seed} verify OK");
        proofs.push(inner);
    }
    let prepared = preprocessing.ok_or("missing preprocessing")?;
    let domain_len = prepared.pcs_setup.required_setup_domain_len()?;
    let mut pair = ProofPair {
        preprocessing: prepared.verifier,
        inner: proofs.try_into().map_err(|_| "expected two proofs")?,
        setup_field_len: domain_len,
        answers: Vec::new(),
    };
    let mut advice = NativeAdvice {
        setup: &prepared.pcs_setup,
        polynomial: pair.preprocessing.pcs_setup.setup_polynomial(domain_len)?,
        answers: Vec::new(),
    };
    let statement = pair.replay(SetupMode::Defer, &mut advice, |_| {})?;
    println!(
        "deferred native replay OK; setup_fields={domain_len} claims={}",
        statement.claims().len()
    );
    pair.answers = advice.answers;
    save(output, &pair)
}

fn recursion_source(function: &str) -> Program {
    let mut source = Program::new("jolt-akita-recursion-guest");
    source.set_func(function);
    source.set_std(true);
    source.set_heap_size(1 << 31);
    source.set_stack_size(1 << 25);
    source.set_max_input_size(1 << 26);
    source.set_max_output_size(1 << 20);
    source.set_linker_memory_size(1 << 30);
    source
}

fn execute_pair(
    function: &str,
    bytes: &[u8],
    directory: &Path,
) -> Result<(CycleReport, JoltDevice)> {
    let mut source = recursion_source(function);
    let memory = memory(&mut source);
    let elf = source.elf.as_ref().ok_or("guest did not build")?;
    fs::write(directory.join(format!("{function}.elf")), elf.bytes())?;
    let inputs = postcard::to_stdvec(bytes)?;
    let capture = Capture::default();
    let (cycles, io, _) = tracing::subscriber::with_default(capture.subscriber(), || {
        tracer::execute(
            elf.bytes(),
            Some(elf.path()),
            &inputs,
            &[],
            &[],
            &memory,
            None,
        )
    });
    if io.panic {
        return Err(format!("{function} panicked").into());
    }
    let padded = ProverConfig::padded_trace_length::<AkitaField>(cycles)?;
    Ok((capture.finish(cycles, padded)?, io))
}

fn replay(pair: &ProofPair) -> Result<SetupBatchStatement<AkitaField>> {
    let mut advice = AnswerTape(pair.answers.iter());
    let statement = pair.replay(SetupMode::Defer, &mut advice, |_| {})?;
    if advice.0.next().is_some() {
        return Err("unused setup advice".into());
    }
    Ok(statement)
}

fn run() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .init();
    match Cli::parse().command {
        Command::Generate {
            trace_log,
            schedules,
            output,
            iterations,
        } => generate(trace_log, &schedules, &output, iterations),
        Command::Rebind {
            pair,
            setup_log,
            output,
        } => {
            let mut pair = decode_pair(&fs::read(pair)?);
            let domain_len = 1usize
                .checked_shl(setup_log)
                .ok_or("setup exponent too large")?;
            if domain_len < pair.setup_field_len {
                return Err("rebind requires a larger setup prefix".into());
            }
            pair.setup_field_len = domain_len;
            let _statement = replay(&pair)?;
            save(&output, &pair)
        }
        Command::Trace { pair, report } => {
            let bytes = fs::read(pair)?;
            let pair = decode_pair(&bytes);
            let statement = replay(&pair)?;
            let material = report.with_extension("material");
            fs::create_dir_all(&material)?;
            let (key, terminal) = pair
                .preprocessing
                .pcs_setup
                .prepare_recursion_material(&pair.inner[0].proof.joint_opening_proof)?;
            fs::write(material.join("key.bin"), key)?;
            fs::write(material.join("terminal.bin"), terminal)?;
            std::env::set_var("JOLT_AKITA_RECURSION_MATERIAL", material.canonicalize()?);
            let directory = report
                .parent()
                .filter(|path| !path.as_os_str().is_empty())
                .unwrap_or(Path::new("."));
            let (before, baseline_io) = execute_pair("verify_before", &bytes, directory)?;
            let (deferred, deferred_io) = execute_pair("verify_deferred", &bytes, directory)?;
            let baseline: Vec<u8> = postcard::from_bytes(&baseline_io.outputs)?;
            if !baseline.is_empty() {
                return Err("unexpected baseline output".into());
            }
            let descriptor: Vec<u8> = postcard::from_bytes(&deferred_io.outputs)?;
            let _inherited = InheritedSetupClaims::from_public_output(0, statement, &descriptor)?;
            let result = serde_json::json!({"inner_count": 2, "inner_padded_trace": pair.inner[0].proof.trace_length, "before": before, "deferred": deferred, "cycle_reduction": before.total_cycles as f64 / deferred.total_cycles as f64, "setup_domain_fields": pair.setup_field_len, "verifier_materialized_setup_fields": pair.preprocessing.pcs_setup.materialized_setup_field_len()?, "claims": pair.answers.len()});
            if let Some(parent) = report.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(report, serde_json::to_vec_pretty(&result)?)?;
            println!("{result}");
            Ok(())
        }
    }
}

fn main() -> Result<()> {
    let outcome = Builder::new()
        .name("akita-recursion".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(|| run().map_err(|error| error.to_string()))?
        .join()
        .map_err(|_| "recursion host thread panicked")?;
    outcome.map_err(Into::into)
}
