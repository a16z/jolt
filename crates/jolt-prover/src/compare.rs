#![expect(
    clippy::expect_used,
    clippy::panic,
    clippy::print_stdout,
    reason = "comparison harness: fail loudly and report to stdout"
)]

use std::time::{Duration, Instant};

use jolt_crypto::{Bn254G1, Pedersen};
use jolt_dory::DoryScheme;
use jolt_field::Fr;
use jolt_inlines_keccak256 as _;
use jolt_inlines_sha2 as _;
use jolt_program::execution::JoltProgram;
use jolt_prover_legacy::host;
use jolt_prover_legacy::poly::commitment::dory::DoryCommitmentScheme;
use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
use jolt_prover_legacy::zkvm::program::ProgramPreprocessing as LegacyProgramPreprocessing;
use jolt_prover_legacy::zkvm::proof::verifier_preprocessing_from_prover;
use jolt_prover_legacy::zkvm::prover::JoltProverPreprocessing as LegacyProverPreprocessing;
use jolt_prover_legacy::zkvm::RV64IMACProver;
use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;
use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

use crate::profile::{
    advice_vars, pad_trace, trace_modular, validate_scale, BackendKind, Workload, SAFETY_MARGIN,
};
use crate::{JoltBackend, JoltProverPreprocessing, ProverConfig};

#[derive(Debug, clap::Args)]
pub struct CompareArgs {
    #[clap(long, value_enum)]
    pub name: Workload,

    #[clap(long)]
    pub scale: Option<u32>,

    #[clap(long, value_enum, default_value = "reference")]
    pub backend: BackendKind,

    #[clap(long, default_value_t = 3)]
    pub runs: usize,

    #[clap(long)]
    pub skip_legacy: bool,

    #[clap(long)]
    pub skip_modular: bool,
}

#[derive(Debug)]
pub struct Timings {
    pub label: String,
    pub runs: Vec<Duration>,
}

impl Timings {
    pub fn median(&self) -> Duration {
        let mut sorted = self.runs.clone();
        sorted.sort_unstable();
        sorted[sorted.len() / 2]
    }
}

#[derive(Debug)]
pub struct Comparison {
    pub modular: Option<Timings>,
    pub legacy: Option<Timings>,
    pub bytes_match: Option<bool>,
}

impl Comparison {
    pub fn speedup(&self) -> Option<f64> {
        let legacy = self.legacy.as_ref()?.median().as_secs_f64();
        let modular = self.modular.as_ref()?.median().as_secs_f64();
        Some(legacy / modular)
    }

    fn report(&self) -> String {
        use std::fmt::Write as _;

        let line = |timings: &Timings| {
            let runs = timings
                .runs
                .iter()
                .map(|run| format!("{:.2}", run.as_secs_f64()))
                .collect::<Vec<_>>()
                .join(", ");
            format!(
                "  {:<10} median {:>7.2}s   runs [{runs}]",
                timings.label,
                timings.median().as_secs_f64(),
            )
        };
        let mut report = "Prover comparison:".to_owned();
        if let Some(modular) = &self.modular {
            report.push('\n');
            report.push_str(&line(modular));
        }
        if let Some(legacy) = &self.legacy {
            report.push('\n');
            report.push_str(&line(legacy));
        }
        if let Some(speedup) = self.speedup() {
            let verdict = if speedup >= 1.0 { "FASTER" } else { "SLOWER" };
            let _ = write!(
                report,
                "\n  speedup (legacy / modular): {speedup:.3}x  [{verdict} than legacy]"
            );
        }
        report.push_str(match self.bytes_match {
            Some(true) => "\n  proof bytes: identical to legacy",
            Some(false) => "\n  proof bytes: DIVERGED FROM LEGACY",
            None => "\n  proof bytes: one side skipped, not compared",
        });
        report
    }
}

pub fn run(args: &CompareArgs) -> Comparison {
    let scale = args.scale.unwrap_or_else(|| args.name.default_scale());
    validate_scale(scale);
    assert!(args.runs > 0, "--runs must be at least 1");

    let bench_name = args.name.as_str();
    let max_trace_length = 1usize << scale;
    let input = args
        .name
        .input((max_trace_length as f64 * SAFETY_MARGIN) as usize);

    println!(
        "=== {bench_name} at scale 2^{scale} ({} run(s) each) ===",
        args.runs
    );

    let mut program = host::Program::new(&format!("{bench_name}-guest"));
    let (bytecode, init_memory_state, _, entry_address) = program.decode();
    let (_, legacy_trace, _, io_device) = program.trace(&input, &[], &[]);
    assert!(
        legacy_trace.len().next_power_of_two() <= max_trace_length,
        "trace is longer than the requested scale"
    );
    drop(legacy_trace);
    let elf_contents = program.get_elf_contents().expect("elf contents");
    let memory_layout = io_device.memory_layout.clone();

    let program_data =
        LegacyProgramPreprocessing::preprocess(bytecode, init_memory_state, entry_address)
            .expect("legacy preprocess");
    let shared =
        JoltSharedPreprocessing::new(program_data, memory_layout.clone(), max_trace_length);
    let legacy_preprocessing = LegacyProverPreprocessing::<
        jolt_prover_legacy::ark_bn254::Fr,
        jolt_prover_legacy::curve::Bn254Curve,
        DoryCommitmentScheme,
    >::new(shared);
    let verifier_preprocessing = verifier_preprocessing_from_prover(&legacy_preprocessing);
    let program_preprocessing = verifier_preprocessing
        .program
        .as_full()
        .expect("full program preprocessing")
        .clone();

    let mut legacy_proof_bytes = None;
    let legacy = (!args.skip_legacy).then(|| {
        let mut runs = Vec::with_capacity(args.runs);
        for index in 0..args.runs {
            let prover = RV64IMACProver::gen_from_elf(
                &legacy_preprocessing,
                &elf_contents,
                &input,
                &[],
                &[],
                None,
                None,
                None,
            );
            let now = Instant::now();
            let (proof, _) = prover.prove().expect("legacy prove");
            runs.push(now.elapsed());
            println!(
                "  legacy   run {}: {:.2}s",
                index + 1,
                runs[index].as_secs_f64()
            );
            if legacy_proof_bytes.is_none() {
                legacy_proof_bytes = Some(
                    bincode::serde::encode_to_vec(&proof, bincode::config::standard())
                        .expect("serialize legacy proof"),
                );
            }
        }
        Timings {
            label: "legacy".to_owned(),
            runs,
        }
    });

    if args.skip_modular {
        let comparison = Comparison {
            modular: None,
            legacy,
            bytes_match: None,
        };
        println!("{}", comparison.report());
        return comparison;
    }

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
    let public_io = trace_output.device.clone();
    let padded_output = pad_trace(trace_output, config.trace_length);
    let witness = TraceBackend::new(
        JoltVmWitnessConfig::new(
            config.trace_length.ilog2() as usize,
            config.ram_K,
            config.one_hot_config,
        ),
        JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, padded_output),
    );
    let total_vars =
        config.one_hot_config.committed_chunk_bits() + config.trace_length.ilog2() as usize;
    let total_vars = total_vars
        .max(advice_vars(memory_layout.max_trusted_advice_size))
        .max(advice_vars(memory_layout.max_untrusted_advice_size));
    let prover_preprocessing = JoltProverPreprocessing::<DoryScheme, Pedersen<Bn254G1>> {
        verifier: verifier_preprocessing,
        pcs_setup: DoryScheme::setup_prover(total_vars),
        committed_program: None,
    };

    let backend = match args.backend {
        BackendKind::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
        #[cfg(feature = "cuda")]
        BackendKind::Cuda => JoltBackend::<Fr, DoryScheme>::cuda(),
    };

    let mut modular_proof_bytes = None;
    let mut runs = Vec::with_capacity(args.runs);
    for index in 0..args.runs {
        let now = Instant::now();
        let proof = crate::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript, _>(
            &backend,
            &prover_preprocessing,
            &config,
            None,
            &witness,
            &public_io,
        )
        .expect("modular prove");
        runs.push(now.elapsed());
        println!(
            "  {:<8} run {}: {:.2}s",
            args.backend.as_str(),
            index + 1,
            runs[index].as_secs_f64()
        );
        if modular_proof_bytes.is_none() {
            jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
                &prover_preprocessing.verifier,
                &public_io,
                &proof,
                None,
            )
            .expect("modular proof must verify");
            modular_proof_bytes = Some(
                bincode::serde::encode_to_vec(&proof, bincode::config::standard())
                    .expect("serialize modular proof"),
            );
        }
    }

    let comparison = Comparison {
        modular: Some(Timings {
            label: args.backend.as_str().to_owned(),
            runs,
        }),
        legacy,
        bytes_match: legacy_proof_bytes
            .as_ref()
            .map(|legacy| Some(legacy) == modular_proof_bytes.as_ref()),
    };
    println!("{}", comparison.report());
    if let Some(false) = comparison.bytes_match {
        panic!("modular proof bytes diverged from legacy — timings are meaningless");
    }
    comparison
}
