//! End-to-end benchmark harness for the MODULAR prover, mirroring the legacy
//! `jolt_prover_legacy benchmark` CLI so the two produce comparable numbers
//! and Perfetto traces.
//!
//! ```text
//! cargo run --release -p jolt-prover --example modular_benchmark \
//!     --features prover-fixtures -- \
//!     --name sha2-chain --scale 16 --format chrome
//! ```
//!
//! Pipeline: legacy-side guest compile/decode (the modular stack has no host
//! toolchain), legacy preprocessing → verifier preprocessing, modular trace
//! (`TracerBackend`), derived `ProverConfig`, `TraceBackend` witness,
//! `jolt_prover::prove` over the selected kernel backend, and a full
//! `jolt_verifier::verify` as the correctness gate. With `--features akita`,
//! the same harness uses the packed protocol and derives its transparent
//! setup directly from the modular prover geometry. Only `prove` is timed —
//! the same window the legacy harness times. Reports wall-clock + kHz,
//! process-lifetime peak RSS (`getrusage`), and the per-stage RSS table
//! collected from the `prove_stage*` spans.

#[cfg(not(feature = "akita"))]
#[expect(
    clippy::expect_used,
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "benchmark harness: fail loudly and report to stdout"
)]
mod dory_benchmark {

    use std::fs;
    use std::io::Write as _;
    use std::sync::Arc;
    use std::time::Instant;

    use clap::{Parser, ValueEnum};
    use common::jolt_device::MemoryConfig;
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    // Keep the inline libraries linked so their host-side registrations reach the
    // tracer (both the legacy and modular trace runs), exactly as the legacy
    // harness does.
    use jolt_inlines_keccak256 as _;
    use jolt_inlines_sha2 as _;
    use jolt_profiling::{
        format_memory_size, peak_rss_bytes, report_stage_memory, setup_tracing, TracingFormat,
        BYTES_PER_GIB,
    };
    use jolt_program::execution::{
        ExecutionBackend, JoltProgram, OwnedTrace, TraceInputs, TraceOutput,
    };
    use jolt_prover::{JoltBackend, JoltProverPreprocessing, ProverConfig};
    use jolt_prover_legacy::curve::Bn254Curve;
    use jolt_prover_legacy::host;
    use jolt_prover_legacy::poly::commitment::dory::DoryCommitmentScheme;
    use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
    use jolt_prover_legacy::zkvm::program::ProgramPreprocessing as LegacyProgramPreprocessing;
    use jolt_prover_legacy::zkvm::proof::verifier_preprocessing_from_prover;
    use jolt_prover_legacy::zkvm::prover::JoltProverPreprocessing as LegacyProverPreprocessing;
    use jolt_transcript::LegacyBlake2bTranscript as Blake2bTranscript;
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
    use tracer::execution_backend::TracerBackend;

    // Empirically measured cycles per operation for RV64IMAC — copied from the
    // legacy harness (`benches/e2e_profiling.rs`) so both harnesses construct
    // identical guest inputs for a given scale.
    const CYCLES_PER_SHA256: f64 = 3396.0;
    const CYCLES_PER_SHA3: f64 = 4330.0;
    const CYCLES_PER_BTREEMAP_OP: f64 = 1550.0;
    const CYCLES_PER_FIBONACCI_UNIT: f64 = 12.0;
    const SAFETY_MARGIN: f64 = 0.9; // Use 90% of max trace capacity

    fn scale_to_target_ops(target_cycles: usize, cycles_per_op: f64) -> u32 {
        std::cmp::max(1, (target_cycles as f64 / cycles_per_op) as u32)
    }

    #[derive(Debug, Copy, Clone, ValueEnum)]
    enum BenchName {
        Fibonacci,
        Sha2Chain,
        Sha3Chain,
        #[value(name = "btreemap")]
        BTreeMap,
    }

    impl BenchName {
        /// The canonical bench name, also the guest crate prefix (`{name}-guest`).
        const fn as_str(self) -> &'static str {
            match self {
                Self::Fibonacci => "fibonacci",
                Self::Sha2Chain => "sha2-chain",
                Self::Sha3Chain => "sha3-chain",
                Self::BTreeMap => "btreemap",
            }
        }

        /// The guest input targeting `target` trace cycles — the same mapping as
        /// the legacy harness's `master_benchmark`.
        fn input(self, target: usize) -> Vec<u8> {
            match self {
                Self::Fibonacci => {
                    postcard::to_stdvec(&scale_to_target_ops(target, CYCLES_PER_FIBONACCI_UNIT))
                        .expect("serialize input")
                }
                Self::Sha2Chain => [
                    postcard::to_stdvec(&[5u8; 32]).expect("serialize input"),
                    postcard::to_stdvec(&scale_to_target_ops(target, CYCLES_PER_SHA256))
                        .expect("serialize input"),
                ]
                .concat(),
                Self::Sha3Chain => [
                    postcard::to_stdvec(&[5u8; 32]).expect("serialize input"),
                    postcard::to_stdvec(&scale_to_target_ops(target, CYCLES_PER_SHA3))
                        .expect("serialize input"),
                ]
                .concat(),
                Self::BTreeMap => {
                    postcard::to_stdvec(&scale_to_target_ops(target, CYCLES_PER_BTREEMAP_OP))
                        .expect("serialize input")
                }
            }
        }
    }

    #[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
    enum Format {
        Default,
        Chrome,
    }

    #[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Default)]
    enum Backend {
        #[default]
        Reference,
        Optimized,
    }

    /// Benchmark the modular prover end to end, mirroring
    /// `jolt_prover_legacy benchmark` semantics.
    #[derive(Parser, Debug)]
    struct Cli {
        /// Benchmark to run.
        #[clap(long, value_enum)]
        name: BenchName,

        /// Max trace length as 2^scale (optional if target-trace-size is provided).
        #[clap(short, long)]
        scale: Option<usize>,

        /// Target specific cycle count (optional, defaults to 90% of 2^scale).
        #[clap(short, long)]
        target_trace_size: Option<usize>,

        /// Output formats.
        #[clap(short, long, value_enum)]
        format: Option<Vec<Format>>,

        /// Kernel backend to prove with.
        #[clap(short, long, value_enum, default_value = "reference")]
        backend: Backend,
    }

    pub fn run() {
        let cli = Cli::parse();
        let scale = match (cli.scale, cli.target_trace_size) {
            (Some(scale), _) => scale,
            (None, Some(target)) => target.next_power_of_two().trailing_zeros() as usize,
            (None, None) => {
                eprintln!("Error: Must provide either --scale or --target-trace-size");
                std::process::exit(1);
            }
        };
        let bench_name = cli.name.as_str();

        let formats: Vec<TracingFormat> = cli
            .format
            .unwrap_or_default()
            .iter()
            .map(|format| match format {
                Format::Default => TracingFormat::Default,
                Format::Chrome => TracingFormat::Chrome,
            })
            .collect();
        let backend_suffix = match cli.backend {
            Backend::Reference => "",
            Backend::Optimized => "_optimized",
        };
        let trace_name = format!(
            "modular_{}_{scale}{backend_suffix}",
            bench_name.replace('-', "_")
        );
        let _guards = setup_tracing(&formats, &trace_name);

        run_benchmark(cli.name, scale, cli.target_trace_size, cli.backend);
    }

    fn run_benchmark(
        bench: BenchName,
        scale: usize,
        target_trace_size: Option<usize>,
        backend_choice: Backend,
    ) {
        let bench_name = bench.as_str();
        let max_trace_length = 1usize << scale;
        let bench_target =
            target_trace_size.unwrap_or((max_trace_length as f64 * SAFETY_MARGIN) as usize);
        tracing::info!("Running modular {bench_name} benchmark at scale 2^{scale}");
        fs::create_dir_all("benchmark-runs/results").expect("create results directory");

        let input = bench.input(bench_target);

        // --- Guest + preprocessing (untimed, mirrors the legacy harness): the
        // guest is compiled/decoded through the legacy host toolchain, and the
        // verifier preprocessing (program view + digest) comes from the legacy
        // preprocessing exactly as in the byte-diff tests.
        let mut program = host::Program::new(&format!("{bench_name}-guest"));
        let (bytecode, init_memory_state, _, entry_address) = program.decode();
        let (_, legacy_trace, _, io_device) = program.trace(&input, &[], &[]);
        assert!(
            legacy_trace.len().next_power_of_two() <= max_trace_length,
            "Trace is longer than expected"
        );
        drop(legacy_trace);
        let elf_contents = program.get_elf_contents().expect("elf contents");
        let memory_layout = io_device.memory_layout.clone();

        let program_data =
            LegacyProgramPreprocessing::preprocess(bytecode, init_memory_state, entry_address)
                .expect("legacy preprocess");
        let shared_preprocessing =
            JoltSharedPreprocessing::new(program_data, memory_layout.clone(), max_trace_length);
        let legacy_preprocessing = LegacyProverPreprocessing::<
            jolt_prover_legacy::ark_bn254::Fr,
            Bn254Curve,
            DoryCommitmentScheme,
        >::new(shared_preprocessing);
        let verifier_preprocessing = verifier_preprocessing_from_prover(&legacy_preprocessing);
        let program_preprocessing = verifier_preprocessing
            .program
            .as_full()
            .expect("full program preprocessing")
            .clone();
        let jolt_program = JoltProgram::from_elf_bytes(elf_contents);

        let span = tracing::info_span!("E2E").entered();

        // --- Modular trace (untimed, like legacy's `gen_from_elf` emulation).
        let trace_output = trace_modular(&jolt_program, &memory_layout, &input);
        let trace_length = trace_output.trace.rows().len();

        let config = ProverConfig::derive::<Fr>(
            trace_output.trace.rows(),
            &memory_layout,
            verifier_preprocessing.program.min_bytecode_address(),
            verifier_preprocessing.program.program_image_len_words(),
            max_trace_length,
        )
        .expect("derive config");
        let public_io = trace_output.device.clone();
        let witness = Arc::new(TraceBackend::new(
            JoltVmWitnessConfig::new(
                config.trace_length.ilog2() as usize,
                config.ram_K,
                config.one_hot_config,
            ),
            JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, trace_output),
        ));

        // PCS setup sized like the byte-diff harness: the main one-hot matrix
        // maxed with both advice candidates (always included in setup sizing,
        // present or not — the SRS is prefix-stable).
        let total_vars = (config.one_hot_config.committed_chunk_bits()
            + config.trace_length.ilog2() as usize)
            .max(advice_vars(memory_layout.max_trusted_advice_size))
            .max(advice_vars(memory_layout.max_untrusted_advice_size));
        let prover_preprocessing = JoltProverPreprocessing::<DoryScheme, Pedersen<Bn254G1>> {
            verifier: verifier_preprocessing,
            pcs_setup: DoryScheme::setup_prover(total_vars),
            committed_program: None,
        };
        let backend = match backend_choice {
            Backend::Reference => JoltBackend::<Fr, DoryScheme>::reference(),
            Backend::Optimized => JoltBackend::<Fr, DoryScheme>::optimized(),
        };

        // --- The timed window: the full modular prove (witness materialization,
        // commitment, all sumcheck stages, joint opening) — the same window the
        // legacy harness times around `prover.prove()`.
        let now = Instant::now();
        let proof =
            jolt_prover::dory::prove::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript, _>(
                &backend,
                &prover_preprocessing,
                &config,
                None,
                Arc::clone(&witness),
                &public_io,
            )
            .expect("modular prove");
        let duration = now.elapsed();
        drop(span);

        let proof_size = bincode::serde::encode_to_vec(&proof, bincode::config::standard())
            .expect("serialize proof")
            .len();

        // --- Correctness gate (untimed): the proof must verify.
        jolt_verifier::verify::<Fr, DoryScheme, Pedersen<Bn254G1>, Blake2bTranscript>(
            &prover_preprocessing.verifier,
            &public_io,
            &proof,
            None,
        )
        .expect("modular proof verifies");

        let backend_label = match backend_choice {
            Backend::Reference => "reference",
            Backend::Optimized => "optimized",
        };
        let proving_hz = trace_length as f64 / duration.as_secs_f64();
        let padded_proving_hz = trace_length.next_power_of_two() as f64 / duration.as_secs_f64();
        println!(
        "modular {} (2^{}, {backend_label}): Prover completed in {:.2}s ({:.1} kHz / padded {:.1} kHz)",
        bench_name,
        scale,
        duration.as_secs_f64(),
        proving_hz / 1000.0,
        padded_proving_hz / 1000.0,
    );
        if let Some(peak) = peak_rss_bytes() {
            println!(
                "modular {} (2^{}, {backend_label}): Peak RSS {}",
                bench_name,
                scale,
                format_memory_size(peak as f64 / BYTES_PER_GIB),
            );
        }
        report_stage_memory();

        // The same 7-field CSV line the legacy harness writes, under a
        // modular-prefixed file name.
        let summary_line = format!(
            "{},{},{:.2},{},{:.2},{},{}\n",
            bench_name,
            scale,
            duration.as_secs_f64(),
            trace_length.next_power_of_two(),
            padded_proving_hz,
            proof_size,
            proof_size,
        );
        let individual_file = format!("benchmark-runs/results/modular_{bench_name}_{scale}.csv");
        if let Err(e) = fs::write(&individual_file, &summary_line) {
            eprintln!("Failed to write individual result file {individual_file}: {e}");
        }
        if let Err(e) = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("benchmark-runs/results/modular_timings.csv")
            .and_then(|mut f| f.write_all(summary_line.as_bytes()))
        {
            eprintln!("Failed to write consolidated timing: {e}");
        }
    }

    /// Trace the guest through the modular stack (`TracerBackend`), with the
    /// memory config mirrored off the legacy layout — the byte-diff wiring.
    fn trace_modular(
        program: &JoltProgram,
        memory_layout: &common::jolt_device::MemoryLayout,
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
                    advice_tape: None,
                    memory_config,
                },
            )
            .expect("modular trace")
    }

    /// A word-aligned advice buffer's balanced Dory matrix variable count.
    fn advice_vars(max_advice_size_bytes: u64) -> usize {
        ((max_advice_size_bytes / 8) as usize)
            .next_power_of_two()
            .max(1)
            .ilog2() as usize
    }
}

#[cfg(not(feature = "akita"))]
fn main() {
    dory_benchmark::run();
}

#[cfg(feature = "akita")]
#[expect(
    clippy::expect_used,
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "benchmark harness: fail loudly and report to stdout"
)]
mod akita_benchmark {
    use std::fs;
    use std::io::Write as _;
    use std::time::Instant;

    use clap::{Parser, ValueEnum};
    use common::jolt_device::MemoryConfig;
    use jolt_akita::AkitaSetupParams;
    use jolt_claims::protocols::jolt::lattice::{OneHotTraceShape, ONE_HOT_TRACE_LAYOUT};
    use jolt_claims::protocols::jolt::JoltRelationId;
    use jolt_inlines_keccak256 as _;
    use jolt_inlines_sha2 as _;
    use jolt_openings::CommitmentScheme;
    use jolt_profiling::{
        format_memory_size, peak_rss_bytes, report_stage_memory, setup_tracing, TracingFormat,
        BYTES_PER_GIB,
    };
    use jolt_program::execution::{
        ExecutionBackend, JoltProgram, OwnedTrace, TraceInputs, TraceOutput,
    };
    use jolt_prover::{akita, JoltProverPreprocessing, ProverConfig};
    use jolt_prover_legacy::host;
    use jolt_prover_legacy::zkvm::packed::{
        akita_verifier_preprocessing, AkitaField, AkitaPackedScheme, AkitaScheme, AkitaTranscript,
        AkitaVc,
    };
    use jolt_prover_legacy::zkvm::preprocessing::JoltSharedPreprocessing;
    use jolt_prover_legacy::zkvm::program::ProgramPreprocessing as LegacyProgramPreprocessing;
    use jolt_prover_legacy::zkvm::prover::JoltProverPreprocessing as LegacyProverPreprocessing;
    use jolt_verifier::stages::formula_dimensions_from_parts;
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
    use tracer::execution_backend::TracerBackend;

    const CYCLES_PER_SHA256: f64 = 3396.0;
    const CYCLES_PER_SHA3: f64 = 4330.0;
    const CYCLES_PER_BTREEMAP_OP: f64 = 1550.0;
    const CYCLES_PER_FIBONACCI_UNIT: f64 = 12.0;
    const SAFETY_MARGIN: f64 = 0.9;

    fn scale_to_target_ops(target_cycles: usize, cycles_per_op: f64) -> u32 {
        std::cmp::max(1, (target_cycles as f64 / cycles_per_op) as u32)
    }

    #[derive(Debug, Copy, Clone, ValueEnum)]
    enum BenchName {
        Fibonacci,
        Sha2Chain,
        Sha3Chain,
        #[value(name = "btreemap")]
        BTreeMap,
    }

    impl BenchName {
        const fn as_str(self) -> &'static str {
            match self {
                Self::Fibonacci => "fibonacci",
                Self::Sha2Chain => "sha2-chain",
                Self::Sha3Chain => "sha3-chain",
                Self::BTreeMap => "btreemap",
            }
        }

        fn input(self, target: usize) -> Vec<u8> {
            match self {
                Self::Fibonacci => {
                    postcard::to_stdvec(&scale_to_target_ops(target, CYCLES_PER_FIBONACCI_UNIT))
                        .expect("serialize input")
                }
                Self::Sha2Chain => [
                    postcard::to_stdvec(&[5u8; 32]).expect("serialize input"),
                    postcard::to_stdvec(&scale_to_target_ops(target, CYCLES_PER_SHA256))
                        .expect("serialize input"),
                ]
                .concat(),
                Self::Sha3Chain => [
                    postcard::to_stdvec(&[5u8; 32]).expect("serialize input"),
                    postcard::to_stdvec(&scale_to_target_ops(target, CYCLES_PER_SHA3))
                        .expect("serialize input"),
                ]
                .concat(),
                Self::BTreeMap => {
                    postcard::to_stdvec(&scale_to_target_ops(target, CYCLES_PER_BTREEMAP_OP))
                        .expect("serialize input")
                }
            }
        }
    }

    #[derive(Debug, Clone, Copy, ValueEnum, PartialEq)]
    enum Format {
        Default,
        Chrome,
    }

    #[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Default)]
    enum Backend {
        Reference,
        #[default]
        Optimized,
        #[cfg(all(feature = "metal", target_os = "macos"))]
        Metal,
    }

    #[derive(Parser, Debug)]
    struct Cli {
        #[clap(long, value_enum)]
        name: BenchName,

        #[clap(short, long)]
        scale: Option<usize>,

        #[clap(short, long)]
        target_trace_size: Option<usize>,

        #[clap(short, long, value_enum)]
        format: Option<Vec<Format>>,

        #[clap(short, long, value_enum, default_value = "optimized")]
        backend: Backend,

        /// Overrides the production threshold for the experimental Product/Instruction
        /// terminal cache. Intended for below-threshold validation runs.
        #[clap(long, hide = true)]
        metal_terminal_cache_cutoff_elements: Option<usize>,

        /// Forces full Outer storage preinitialization for controlled A/B runs.
        #[clap(long, hide = true)]
        metal_outer_full_initialization: bool,
    }

    pub fn run() {
        let cli = Cli::parse();
        let scale = match (cli.scale, cli.target_trace_size) {
            (Some(scale), _) => scale,
            (None, Some(target)) => target.next_power_of_two().trailing_zeros() as usize,
            (None, None) => {
                eprintln!("Error: Must provide either --scale or --target-trace-size");
                std::process::exit(1);
            }
        };
        let formats = cli
            .format
            .unwrap_or_default()
            .iter()
            .map(|format| match format {
                Format::Default => TracingFormat::Default,
                Format::Chrome => TracingFormat::Chrome,
            })
            .collect::<Vec<_>>();
        let backend_suffix = match cli.backend {
            Backend::Reference => "",
            Backend::Optimized => "_optimized",
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal => "_metal",
        };
        let trace_name = format!(
            "akita_{}_{}{backend_suffix}",
            cli.name.as_str().replace('-', "_"),
            scale,
        );
        let _guards = setup_tracing(&formats, &trace_name);

        run_benchmark(
            cli.name,
            scale,
            cli.target_trace_size,
            cli.backend,
            cli.metal_terminal_cache_cutoff_elements,
            cli.metal_outer_full_initialization,
        );
    }

    fn run_benchmark(
        bench: BenchName,
        scale: usize,
        target_trace_size: Option<usize>,
        backend_choice: Backend,
        metal_terminal_cache_cutoff_elements: Option<usize>,
        metal_outer_full_initialization: bool,
    ) {
        let bench_name = bench.as_str();
        let max_trace_length = 1usize << scale;
        let bench_target =
            target_trace_size.unwrap_or((max_trace_length as f64 * SAFETY_MARGIN) as usize);
        tracing::info!("Running modular Akita {bench_name} benchmark at scale 2^{scale}");
        fs::create_dir_all("benchmark-runs/results").expect("create results directory");

        let input = bench.input(bench_target);
        let mut program = host::Program::new(&format!("{bench_name}-guest"));
        let (bytecode, init_memory_state, _, entry_address) = program.decode();
        let (_, legacy_trace, _, io_device) = program.trace(&input, &[], &[]);
        assert!(
            legacy_trace.len().next_power_of_two() <= max_trace_length,
            "Trace is longer than expected"
        );
        drop(legacy_trace);
        let elf_contents = program.get_elf_contents().expect("elf contents");
        let memory_layout = io_device.memory_layout.clone();
        drop(program);

        let program_data =
            LegacyProgramPreprocessing::preprocess(bytecode, init_memory_state, entry_address)
                .expect("legacy preprocess");
        let program_meta = program_data.meta();
        let shared_preprocessing: JoltSharedPreprocessing<AkitaPackedScheme> =
            JoltSharedPreprocessing::new(program_data, memory_layout.clone(), max_trace_length);
        let legacy_preprocessing = LegacyProverPreprocessing::new(shared_preprocessing);
        let jolt_program = JoltProgram::from_elf_bytes(elf_contents);

        let span = tracing::info_span!("E2E").entered();
        let trace_output = trace_modular(&jolt_program, &memory_layout, &input);
        let trace_length = trace_output.trace.rows().len();
        let config = ProverConfig::derive::<AkitaField>(
            trace_output.trace.rows(),
            &memory_layout,
            program_meta.min_bytecode_address,
            program_meta.program_image_len_words,
            max_trace_length,
        )
        .expect("derive config");

        let log_t = config.trace_length.ilog2() as usize;
        let log_k_chunk = config.one_hot_config.committed_chunk_bits();
        let dimensions = formula_dimensions_from_parts(
            config.one_hot_config,
            log_t,
            program_meta.bytecode_len,
            config.ram_K,
            JoltRelationId::HammingWeightClaimReduction,
        )
        .expect("derive packed formula dimensions");
        let one_hot_shape = OneHotTraceShape {
            ra_layout: dimensions.ra_layout,
            log_t,
            log_k_chunk,
        };
        let bytecode_dimensions = dimensions.bytecode_read_raf;
        let backend_label = match backend_choice {
            Backend::Reference => "reference",
            Backend::Optimized => "optimized",
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal => "metal",
        };
        let effective_bytecode_algebra = if backend_choice == Backend::Reference
            || bytecode_dimensions.num_committed_ra_polys() != 2
        {
            "generic"
        } else {
            "q10"
        };
        println!(
            "BYTECODE_CYCLE_CONFIG requested=q10 effective={} log_t={} log_k={} chunk_bits={} num_ra={} degree={}",
            effective_bytecode_algebra,
            bytecode_dimensions.log_t(),
            bytecode_dimensions.log_k(),
            log_k_chunk,
            bytecode_dimensions.num_committed_ra_polys(),
            bytecode_dimensions.num_committed_ra_polys() + 2,
        );
        println!(
            "PIOP_EXECUTION_CONFIG rayon_threads={}",
            rayon::current_num_threads()
        );
        let setup_shape = ONE_HOT_TRACE_LAYOUT
            .setup_shape(&one_hot_shape)
            .expect("derive canonical packed setup shape");
        let trace_plan = ONE_HOT_TRACE_LAYOUT
            .plan(&one_hot_shape)
            .expect("derive canonical packed trace plan");
        let layout_digest = ONE_HOT_TRACE_LAYOUT
            .layout_digest(&one_hot_shape)
            .expect("derive canonical packed layout digest");
        let setup_params = AkitaSetupParams::one_hot_only(
            setup_shape.num_vars,
            setup_shape.num_polys,
            layout_digest,
            1usize << log_k_chunk,
        );
        let (pcs_setup, verifier_setup) =
            AkitaScheme::setup(setup_params).expect("the transparent packed setup must derive");
        let verifier_preprocessing =
            akita_verifier_preprocessing(&legacy_preprocessing, verifier_setup, None);
        drop(legacy_preprocessing);

        let program_preprocessing = verifier_preprocessing
            .program
            .as_full()
            .expect("full program preprocessing")
            .clone();
        let public_io = trace_output.device.clone();
        let witness = TraceBackend::new(
            JoltVmWitnessConfig::new(log_t, config.ram_K, config.one_hot_config),
            JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, trace_output),
        );
        let prover_preprocessing = JoltProverPreprocessing::<AkitaScheme, AkitaVc> {
            verifier: verifier_preprocessing,
            pcs_setup,
            committed_program: None,
        };
        let optimized_bytecode_algebra = jolt_kernels::optimized::BytecodeCycleAlgebra::Q10;
        let mut backend = match backend_choice {
            Backend::Reference => akita::JoltAkitaBackend::reference(),
            Backend::Optimized => akita::JoltAkitaBackend::optimized(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal => {
                let mut config = jolt_kernels::metal::MetalConfig::production();
                if let Some(cutoff) = metal_terminal_cache_cutoff_elements {
                    config
                        .spartan_product_remainder
                        .terminal_cache_cutoff_elements = cutoff;
                }
                if metal_outer_full_initialization {
                    config
                        .spartan_outer_remainder
                        .dispatch
                        .storage_initialization =
                        jolt_kernels::metal::solinas::OuterRemainderStorageInitialization::Full;
                }
                println!(
                    "AKITA_METAL_PROFILE value=production terminal_cache_cutoff_elements={} outer_storage_initialization={}",
                    config
                        .spartan_product_remainder
                        .terminal_cache_cutoff_elements,
                    config
                        .spartan_outer_remainder
                        .dispatch
                        .storage_initialization
                        .as_str(),
                );
                let metal = jolt_kernels::metal::MetalBackend::new(config)
                    .expect("Metal backend should initialize");
                akita::JoltAkitaBackend::optimized()
                    .with_metal_compute(&metal)
                    .expect("Metal trace commitment backend should initialize")
            }
        };
        if backend_choice == Backend::Optimized {
            backend.base.bytecode_read_raf_cycle =
                Box::new(jolt_kernels::optimized::OptimizedBytecodeReadRafCycle::new(
                    optimized_bytecode_algebra,
                ));
        }

        let prepare_start = Instant::now();
        backend
            .prepare_trace_commitment(
                &prover_preprocessing.pcs_setup,
                trace_plan.packing().slot_capacity(),
                trace_plan.packing().ids().len(),
                1usize << log_t,
            )
            .expect("trace commitment backend should prepare");
        println!(
            "AKITA_TRACE_COMMIT_PREPARE_MS value={:.6}",
            prepare_start.elapsed().as_secs_f64() * 1_000.0
        );

        let now = Instant::now();
        let proof = akita::prove::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript, _>(
            &backend,
            &prover_preprocessing,
            &config,
            None,
            None,
            &witness,
            &public_io,
        )
        .expect("modular Akita prove");
        let duration = now.elapsed();
        drop(span);

        #[cfg(all(feature = "metal", target_os = "macos"))]
        if backend_choice == Backend::Metal {
            let qualified = jolt_akita::TraceCommitmentBackend::shape_is_metal_qualified(
                prover_preprocessing.pcs_setup.one_hot_k(),
                prover_preprocessing.pcs_setup.max_num_vars(),
            );
            let metrics = backend
                .last_trace_commitment_metrics()
                .expect("Metal trace commitment metrics should be readable");
            if qualified {
                let metrics = metrics.expect("qualified trace commitment should dispatch Metal");
                assert!(metrics.input_zero_copy, "trace input must remain zero-copy");
                assert!(
                    metrics.matrix_cache_hit,
                    "explicit preparation must make the commit matrix-resident"
                );
                assert!(
                    metrics.metal_work_units > 0,
                    "qualified trace commitment must schedule Metal work"
                );
                println!(
                    "AKITA_TRACE_COMMIT_METRICS qualified=true used_metal=true \
                     input_zero_copy={} matrix_cache_hit={} cpu_work_units={} \
                     metal_work_units={} cpu_blocks={} columns={} metal_blocks={} hot_entries={} \
                     index_bytes={} matrix_bytes={} opening_index_bytes={} \
                     matrix_read_bytes={} lane_read_bytes={} scratch_bytes={} \
                     opening_index_ms={:.6} opening_index_gpu_ms={:.6} \
                     buffer_setup_ms={:.6} command_wall_ms={:.6} gpu_ms={} \
                     cpu_ms={:.6} readback_ms={:.6} reconstruction_ms={:.6} \
                     merge_ms={:.6} inner_total_ms={:.6} \
                     digit_rows_ms={:.6} digit_rows_gpu_ms={:.6} \
                     digit_rows_calls={} digit_rows_metal_calls={} \
                     digit_rows_max_rows={} digit_rows_max_columns={} \
                     digit_rows_max_batch={} compression_ms={:.6}",
                    metrics.input_zero_copy,
                    metrics.matrix_cache_hit,
                    metrics.cpu_work_units,
                    metrics.metal_work_units,
                    metrics.cpu_blocks,
                    metrics.metal_columns,
                    metrics.metal_blocks,
                    metrics.hot_entries,
                    metrics.index_bytes,
                    metrics.matrix_bytes,
                    metrics.opening_index_bytes,
                    metrics.modeled_matrix_read_bytes,
                    metrics.modeled_lane_read_bytes,
                    metrics.scratch_bytes,
                    metrics.opening_index_time.as_secs_f64() * 1_000.0,
                    metrics.opening_index_gpu_time.as_secs_f64() * 1_000.0,
                    metrics.buffer_setup_time.as_secs_f64() * 1_000.0,
                    metrics.command_wall_time.as_secs_f64() * 1_000.0,
                    metrics.gpu_time.map_or_else(
                        || "unavailable".to_string(),
                        |time| format!("{:.6}", time.as_secs_f64() * 1_000.0),
                    ),
                    metrics.cpu_time.as_secs_f64() * 1_000.0,
                    metrics.readback_copy_time.as_secs_f64() * 1_000.0,
                    metrics.output_reconstruction_time.as_secs_f64() * 1_000.0,
                    metrics.merge_time.as_secs_f64() * 1_000.0,
                    metrics.total_time.as_secs_f64() * 1_000.0,
                    metrics.digit_rows_time.as_secs_f64() * 1_000.0,
                    metrics.digit_rows_gpu_time.as_secs_f64() * 1_000.0,
                    metrics.digit_rows_calls,
                    metrics.digit_rows_metal_calls,
                    metrics.digit_rows_max_rows,
                    metrics.digit_rows_max_columns,
                    metrics.digit_rows_max_batch,
                    metrics.compression_time.as_secs_f64() * 1_000.0,
                );
            } else {
                assert!(
                    metrics.is_none(),
                    "unqualified trace commitment should remain on CPU"
                );
                println!("AKITA_TRACE_COMMIT_METRICS qualified=false used_metal=false route=cpu");
            }

            let opening = backend
                .last_trace_opening_metrics()
                .expect("Metal trace opening metrics should be readable");
            if let Some(opening) = opening {
                println!(
                    "AKITA_TRACE_OPENING_METRICS opening_index_bytes={} \
                     opening_index_ms={:.6} opening_index_gpu_ms={:.6} \
                     allocation_bytes={} command_wall_ms={:.6} gpu_ms={:.6} \
                     packed_indexed_calls={} cpu_fallback_calls={} \
                     planned_cpu_calls={} planned_cpu_work_units={} cpu_tail_work_units={}",
                    opening.opening_index_bytes,
                    opening.opening_index_time.as_secs_f64() * 1_000.0,
                    opening.opening_index_gpu_time.as_secs_f64() * 1_000.0,
                    opening.allocation_bytes,
                    opening.command_wall_time.as_secs_f64() * 1_000.0,
                    opening.gpu_active_time.as_secs_f64() * 1_000.0,
                    opening.packed_decompose_indexed_calls,
                    opening.cpu_fallback_calls,
                    opening.planned_cpu_calls,
                    opening.planned_cpu_work_units,
                    opening.cpu_tail_work_units,
                );
            } else {
                assert!(!qualified, "qualified trace opening should dispatch Metal");
                println!("AKITA_TRACE_OPENING_METRICS qualified=false route=cpu");
            }
        }

        // The Akita field uses its own canonical serializer, while the Jolt
        // proof envelope currently only exposes serde. Keep the legacy CSV
        // shape and report proof size as unavailable instead of measuring an
        // in-memory representation.
        let proof_size = 0;
        jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
            &prover_preprocessing.verifier,
            &public_io,
            &proof,
            None,
        )
        .expect("modular Akita proof verifies");
        println!("PROOF_VERIFIED backend={backend_label} value=true");

        let proving_hz = trace_length as f64 / duration.as_secs_f64();
        let padded_proving_hz = trace_length.next_power_of_two() as f64 / duration.as_secs_f64();
        println!(
            "modular Akita {} (2^{}, {backend_label}): Prover completed in {:.2}s \
             ({:.1} kHz / padded {:.1} kHz)",
            bench_name,
            scale,
            duration.as_secs_f64(),
            proving_hz / 1000.0,
            padded_proving_hz / 1000.0,
        );
        if let Some(peak) = peak_rss_bytes() {
            println!(
                "modular Akita {} (2^{}, {backend_label}): Peak RSS {}",
                bench_name,
                scale,
                format_memory_size(peak as f64 / BYTES_PER_GIB),
            );
        }
        report_stage_memory();

        let summary_line = format!(
            "{},{},{:.2},{},{:.2},{},{}\n",
            bench_name,
            scale,
            duration.as_secs_f64(),
            trace_length.next_power_of_two(),
            padded_proving_hz,
            proof_size,
            proof_size,
        );
        let backend_suffix = match backend_choice {
            Backend::Reference => "",
            Backend::Optimized => "_optimized",
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal => "_metal",
        };
        let individual_file =
            format!("benchmark-runs/results/akita_{bench_name}_{scale}{backend_suffix}.csv");
        if let Err(error) = fs::write(&individual_file, &summary_line) {
            eprintln!("Failed to write individual result file {individual_file}: {error}");
        }
        if let Err(error) = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open("benchmark-runs/results/akita_timings.csv")
            .and_then(|mut file| file.write_all(summary_line.as_bytes()))
        {
            eprintln!("Failed to write consolidated timing: {error}");
        }
    }

    fn trace_modular(
        program: &JoltProgram,
        memory_layout: &common::jolt_device::MemoryLayout,
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
                    advice_tape: None,
                    memory_config,
                },
            )
            .expect("modular trace")
    }
}

#[cfg(feature = "akita")]
fn main() {
    akita_benchmark::run();
}
