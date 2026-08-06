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
        let witness = TraceBackend::new(
            JoltVmWitnessConfig::new(
                config.trace_length.ilog2() as usize,
                config.ram_K,
                config.one_hot_config,
            ),
            JoltVmWitnessInputs::new(&jolt_program, &program_preprocessing, trace_output),
        );

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
                &witness,
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

    #[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Default)]
    enum InstructionRaMaterializeWidth {
        #[default]
        W16,
        W32,
        W64,
        W128,
        W256,
        W512,
    }

    #[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Default)]
    enum BytecodeCycleAlgebra {
        Generic,
        #[default]
        Q10,
        Q10Accum,
    }

    impl BytecodeCycleAlgebra {
        const fn as_str(self) -> &'static str {
            match self {
                Self::Generic => "generic",
                Self::Q10 => "q10",
                Self::Q10Accum => "q10-accum",
            }
        }
    }

    #[derive(Debug, Clone, Copy, ValueEnum, PartialEq, Default)]
    enum OuterRemainderBindingPlan {
        #[default]
        #[value(name = "b_only_v1")]
        BOnlyV1,
        #[value(name = "b_only_padded_56_v1")]
        BOnlyPadded56V1,
    }

    #[cfg(all(feature = "metal", target_os = "macos"))]
    impl OuterRemainderBindingPlan {
        const fn as_str(self) -> &'static str {
            match self {
                Self::BOnlyV1 => "b_only_v1",
                Self::BOnlyPadded56V1 => "b_only_padded_56_v1",
            }
        }
    }

    #[derive(Debug, Clone, Copy)]
    struct BytecodeMetalTuning {
        message_threads: usize,
        transition_threads: usize,
        max_threadgroups: usize,
        cutoff_log2: u32,
        trace_cutoff_log2: u32,
    }

    #[derive(Debug, Clone, Copy)]
    struct InstructionInputMetalTuning {
        native_message_threads: usize,
        native_transition_threads: usize,
        dense_transition_threads: usize,
        cutoff_log2: u32,
        trace_cutoff_log2: u32,
    }

    #[derive(Debug, Clone, Copy)]
    struct BooleanityAddressMetalTuning {
        inner_log2: usize,
        selectors_per_tile: usize,
        tile_threads: usize,
        finalize_threads: usize,
        trace_cutoff_log2: u32,
    }

    #[derive(Debug, Clone, Copy)]
    struct HammingWeightMetalTuning {
        inner_log2: usize,
        selectors_per_tile: usize,
        tile_threads: usize,
        finalize_threads: usize,
        trace_cutoff_log2: u32,
    }

    #[derive(Debug, Clone, Copy)]
    struct OuterRemainderMetalTuning {
        materialize_threads: usize,
        transition_threads: usize,
        output_threads: usize,
        cutoff_log2: u32,
        trace_cutoff_log2: u32,
        binding_plan: OuterRemainderBindingPlan,
    }

    #[derive(Debug, Clone, Copy)]
    struct BackendConfig {
        backend: Backend,
        instruction_ra_materialize_width: InstructionRaMaterializeWidth,
        instruction_ra_reuse_inverse: bool,
        bytecode_cycle_algebra: BytecodeCycleAlgebra,
        bytecode_metal: BytecodeMetalTuning,
        instruction_input_metal: InstructionInputMetalTuning,
        booleanity_address_metal: BooleanityAddressMetalTuning,
        hamming_weight_metal: HammingWeightMetalTuning,
        outer_remainder_metal: OuterRemainderMetalTuning,
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

        #[clap(long, value_enum, default_value = "w16")]
        instruction_ra_materialize_width: InstructionRaMaterializeWidth,

        #[clap(long)]
        instruction_ra_reuse_inverse: bool,

        #[clap(long, value_enum, default_value = "q10")]
        bytecode_cycle_algebra: BytecodeCycleAlgebra,

        #[clap(long, default_value_t = 256)]
        bytecode_metal_message_threads: usize,

        #[clap(long, default_value_t = 128)]
        bytecode_metal_transition_threads: usize,

        #[clap(long, default_value_t = 8192)]
        bytecode_metal_max_threadgroups: usize,

        #[clap(long, default_value_t = 16)]
        bytecode_metal_cutoff_log2: u32,

        #[clap(long, default_value_t = 18)]
        bytecode_metal_trace_cutoff_log2: u32,

        #[clap(long, default_value_t = 256)]
        instruction_input_metal_native_message_threads: usize,

        #[clap(long, default_value_t = 128)]
        instruction_input_metal_native_transition_threads: usize,

        #[clap(long, default_value_t = 128)]
        instruction_input_metal_dense_transition_threads: usize,

        #[clap(long, default_value_t = 16)]
        instruction_input_metal_cutoff_log2: u32,

        #[clap(long, default_value_t = 25)]
        instruction_input_metal_trace_cutoff_log2: u32,

        #[clap(long, default_value_t = 15)]
        booleanity_address_metal_inner_log2: usize,

        #[clap(long, default_value_t = 6)]
        booleanity_address_metal_selectors_per_tile: usize,

        #[clap(long, default_value_t = 512)]
        booleanity_address_metal_tile_threads: usize,

        #[clap(long, default_value_t = 1024)]
        booleanity_address_metal_finalize_threads: usize,

        #[clap(long, default_value_t = 18)]
        booleanity_address_metal_trace_cutoff_log2: u32,

        #[clap(long, default_value_t = 15)]
        hamming_weight_metal_inner_log2: usize,

        #[clap(long, default_value_t = 6)]
        hamming_weight_metal_selectors_per_tile: usize,

        #[clap(long, default_value_t = 512)]
        hamming_weight_metal_tile_threads: usize,

        #[clap(long, default_value_t = 1024)]
        hamming_weight_metal_finalize_threads: usize,

        #[clap(long, default_value_t = 18)]
        hamming_weight_metal_trace_cutoff_log2: u32,

        #[clap(long, default_value_t = 256)]
        outer_remainder_metal_materialize_threads: usize,

        #[clap(long, default_value_t = 128)]
        outer_remainder_metal_transition_threads: usize,

        #[clap(long, default_value_t = 256)]
        outer_remainder_metal_output_threads: usize,

        #[clap(long, default_value_t = 16)]
        outer_remainder_metal_cutoff_log2: u32,

        #[clap(long, default_value_t = 18)]
        outer_remainder_metal_trace_cutoff_log2: u32,

        #[clap(long, value_enum, default_value = "b_only_v1")]
        outer_remainder_metal_binding_plan: OuterRemainderBindingPlan,
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

        let backend_config = BackendConfig {
            backend: cli.backend,
            instruction_ra_materialize_width: cli.instruction_ra_materialize_width,
            instruction_ra_reuse_inverse: cli.instruction_ra_reuse_inverse,
            bytecode_cycle_algebra: cli.bytecode_cycle_algebra,
            bytecode_metal: BytecodeMetalTuning {
                message_threads: cli.bytecode_metal_message_threads,
                transition_threads: cli.bytecode_metal_transition_threads,
                max_threadgroups: cli.bytecode_metal_max_threadgroups,
                cutoff_log2: cli.bytecode_metal_cutoff_log2,
                trace_cutoff_log2: cli.bytecode_metal_trace_cutoff_log2,
            },
            instruction_input_metal: InstructionInputMetalTuning {
                native_message_threads: cli.instruction_input_metal_native_message_threads,
                native_transition_threads: cli.instruction_input_metal_native_transition_threads,
                dense_transition_threads: cli.instruction_input_metal_dense_transition_threads,
                cutoff_log2: cli.instruction_input_metal_cutoff_log2,
                trace_cutoff_log2: cli.instruction_input_metal_trace_cutoff_log2,
            },
            booleanity_address_metal: BooleanityAddressMetalTuning {
                inner_log2: cli.booleanity_address_metal_inner_log2,
                selectors_per_tile: cli.booleanity_address_metal_selectors_per_tile,
                tile_threads: cli.booleanity_address_metal_tile_threads,
                finalize_threads: cli.booleanity_address_metal_finalize_threads,
                trace_cutoff_log2: cli.booleanity_address_metal_trace_cutoff_log2,
            },
            hamming_weight_metal: HammingWeightMetalTuning {
                inner_log2: cli.hamming_weight_metal_inner_log2,
                selectors_per_tile: cli.hamming_weight_metal_selectors_per_tile,
                tile_threads: cli.hamming_weight_metal_tile_threads,
                finalize_threads: cli.hamming_weight_metal_finalize_threads,
                trace_cutoff_log2: cli.hamming_weight_metal_trace_cutoff_log2,
            },
            outer_remainder_metal: OuterRemainderMetalTuning {
                materialize_threads: cli.outer_remainder_metal_materialize_threads,
                transition_threads: cli.outer_remainder_metal_transition_threads,
                output_threads: cli.outer_remainder_metal_output_threads,
                cutoff_log2: cli.outer_remainder_metal_cutoff_log2,
                trace_cutoff_log2: cli.outer_remainder_metal_trace_cutoff_log2,
                binding_plan: cli.outer_remainder_metal_binding_plan,
            },
        };
        run_benchmark(cli.name, scale, cli.target_trace_size, backend_config);
    }

    fn run_benchmark(
        bench: BenchName,
        scale: usize,
        target_trace_size: Option<usize>,
        backend_config: BackendConfig,
    ) {
        let BackendConfig {
            backend: backend_choice,
            instruction_ra_materialize_width,
            instruction_ra_reuse_inverse,
            bytecode_cycle_algebra,
            bytecode_metal:
                BytecodeMetalTuning {
                    message_threads: bytecode_metal_message_threads,
                    transition_threads: bytecode_metal_transition_threads,
                    max_threadgroups: bytecode_metal_max_threadgroups,
                    cutoff_log2: bytecode_metal_cutoff_log2,
                    trace_cutoff_log2: bytecode_metal_trace_cutoff_log2,
                },
            instruction_input_metal:
                InstructionInputMetalTuning {
                    native_message_threads: instruction_input_metal_native_message_threads,
                    native_transition_threads: instruction_input_metal_native_transition_threads,
                    dense_transition_threads: instruction_input_metal_dense_transition_threads,
                    cutoff_log2: instruction_input_metal_cutoff_log2,
                    trace_cutoff_log2: instruction_input_metal_trace_cutoff_log2,
                },
            booleanity_address_metal:
                BooleanityAddressMetalTuning {
                    inner_log2: booleanity_address_metal_inner_log2,
                    selectors_per_tile: booleanity_address_metal_selectors_per_tile,
                    tile_threads: booleanity_address_metal_tile_threads,
                    finalize_threads: booleanity_address_metal_finalize_threads,
                    trace_cutoff_log2: booleanity_address_metal_trace_cutoff_log2,
                },
            hamming_weight_metal:
                HammingWeightMetalTuning {
                    inner_log2: hamming_weight_metal_inner_log2,
                    selectors_per_tile: hamming_weight_metal_selectors_per_tile,
                    tile_threads: hamming_weight_metal_tile_threads,
                    finalize_threads: hamming_weight_metal_finalize_threads,
                    trace_cutoff_log2: hamming_weight_metal_trace_cutoff_log2,
                },
            outer_remainder_metal:
                OuterRemainderMetalTuning {
                    materialize_threads: outer_remainder_metal_materialize_threads,
                    transition_threads: outer_remainder_metal_transition_threads,
                    output_threads: outer_remainder_metal_output_threads,
                    cutoff_log2: outer_remainder_metal_cutoff_log2,
                    trace_cutoff_log2: outer_remainder_metal_trace_cutoff_log2,
                    binding_plan: outer_remainder_metal_binding_plan,
                },
        } = backend_config;
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
            bytecode_cycle_algebra.as_str()
        };
        println!(
            "BYTECODE_CYCLE_CONFIG requested={} effective={} log_t={} log_k={} chunk_bits={} num_ra={} degree={}",
            bytecode_cycle_algebra.as_str(),
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
        #[cfg(not(all(feature = "metal", target_os = "macos")))]
        let _ = (
            instruction_ra_materialize_width,
            instruction_ra_reuse_inverse,
            bytecode_metal_message_threads,
            bytecode_metal_transition_threads,
            bytecode_metal_max_threadgroups,
            bytecode_metal_cutoff_log2,
            bytecode_metal_trace_cutoff_log2,
            instruction_input_metal_native_message_threads,
            instruction_input_metal_native_transition_threads,
            instruction_input_metal_dense_transition_threads,
            instruction_input_metal_cutoff_log2,
            instruction_input_metal_trace_cutoff_log2,
            booleanity_address_metal_inner_log2,
            booleanity_address_metal_selectors_per_tile,
            booleanity_address_metal_tile_threads,
            booleanity_address_metal_finalize_threads,
            booleanity_address_metal_trace_cutoff_log2,
            hamming_weight_metal_inner_log2,
            hamming_weight_metal_selectors_per_tile,
            hamming_weight_metal_tile_threads,
            hamming_weight_metal_finalize_threads,
            hamming_weight_metal_trace_cutoff_log2,
            outer_remainder_metal_materialize_threads,
            outer_remainder_metal_transition_threads,
            outer_remainder_metal_output_threads,
            outer_remainder_metal_cutoff_log2,
            outer_remainder_metal_trace_cutoff_log2,
            outer_remainder_metal_binding_plan,
        );
        let optimized_bytecode_algebra = match bytecode_cycle_algebra {
            BytecodeCycleAlgebra::Generic => jolt_kernels::optimized::BytecodeCycleAlgebra::Generic,
            BytecodeCycleAlgebra::Q10 => jolt_kernels::optimized::BytecodeCycleAlgebra::Q10,
            BytecodeCycleAlgebra::Q10Accum => {
                jolt_kernels::optimized::BytecodeCycleAlgebra::Q10Accum
            }
        };
        let mut backend = match backend_choice {
            Backend::Reference => akita::JoltAkitaBackend::reference(),
            Backend::Optimized => akita::JoltAkitaBackend::optimized(),
            #[cfg(all(feature = "metal", target_os = "macos"))]
            Backend::Metal => {
                let mut config = jolt_kernels::metal::MetalConfig::default();
                config
                    .instruction_ra_virtualization
                    .dispatch
                    .materialize_width = match instruction_ra_materialize_width {
                    InstructionRaMaterializeWidth::W16 => {
                        jolt_kernels::metal::solinas::InstructionRaMaterializeWidth::W16
                    }
                    InstructionRaMaterializeWidth::W32 => {
                        jolt_kernels::metal::solinas::InstructionRaMaterializeWidth::W32
                    }
                    InstructionRaMaterializeWidth::W64 => {
                        jolt_kernels::metal::solinas::InstructionRaMaterializeWidth::W64
                    }
                    InstructionRaMaterializeWidth::W128 => {
                        jolt_kernels::metal::solinas::InstructionRaMaterializeWidth::W128
                    }
                    InstructionRaMaterializeWidth::W256 => {
                        jolt_kernels::metal::solinas::InstructionRaMaterializeWidth::W256
                    }
                    InstructionRaMaterializeWidth::W512 => {
                        jolt_kernels::metal::solinas::InstructionRaMaterializeWidth::W512
                    }
                };
                config
                    .instruction_ra_virtualization
                    .dispatch
                    .reuse_inverse_for_dense = instruction_ra_reuse_inverse;
                config.bytecode_read_raf_cycle.cpu_tail_algebra = optimized_bytecode_algebra;
                config
                    .bytecode_read_raf_cycle
                    .dispatch
                    .message_threads_per_threadgroup = Some(bytecode_metal_message_threads);
                config
                    .bytecode_read_raf_cycle
                    .dispatch
                    .transition_threads_per_threadgroup = Some(bytecode_metal_transition_threads);
                config.bytecode_read_raf_cycle.dispatch.max_threadgroups =
                    bytecode_metal_max_threadgroups;
                config.bytecode_read_raf_cycle.cutoff_elements = 1usize
                    .checked_shl(bytecode_metal_cutoff_log2)
                    .expect("Bytecode Metal cutoff log2 must fit usize");
                config.bytecode_read_raf_cycle.trace_cutoff_elements = 1usize
                    .checked_shl(bytecode_metal_trace_cutoff_log2)
                    .expect("Bytecode Metal trace cutoff log2 must fit usize");
                config
                    .instruction_input
                    .dispatch
                    .native_message_threads_per_threadgroup =
                    Some(instruction_input_metal_native_message_threads);
                config
                    .instruction_input
                    .dispatch
                    .native_transition_threads_per_threadgroup =
                    Some(instruction_input_metal_native_transition_threads);
                config
                    .instruction_input
                    .dispatch
                    .dense_transition_threads_per_threadgroup =
                    Some(instruction_input_metal_dense_transition_threads);
                config.instruction_input.cutoff_elements = 1usize
                    .checked_shl(instruction_input_metal_cutoff_log2)
                    .expect("InstructionInput Metal cutoff log2 must fit usize");
                config.instruction_input.trace_cutoff_elements = 1usize
                    .checked_shl(instruction_input_metal_trace_cutoff_log2)
                    .expect("InstructionInput Metal trace cutoff log2 must fit usize");
                config.booleanity_address.dispatch.inner_log2 = booleanity_address_metal_inner_log2;
                config.booleanity_address.dispatch.selectors_per_tile =
                    booleanity_address_metal_selectors_per_tile;
                config
                    .booleanity_address
                    .dispatch
                    .tile_threads_per_threadgroup = Some(booleanity_address_metal_tile_threads);
                config
                    .booleanity_address
                    .dispatch
                    .finalize_threads_per_threadgroup =
                    Some(booleanity_address_metal_finalize_threads);
                config.booleanity_address.trace_cutoff_elements = 1usize
                    .checked_shl(booleanity_address_metal_trace_cutoff_log2)
                    .expect("Booleanity address Metal trace cutoff log2 must fit usize");
                config.hamming_weight_claim_reduction.dispatch.inner_log2 =
                    hamming_weight_metal_inner_log2;
                config
                    .hamming_weight_claim_reduction
                    .dispatch
                    .selectors_per_tile = hamming_weight_metal_selectors_per_tile;
                config
                    .hamming_weight_claim_reduction
                    .dispatch
                    .tile_threads_per_threadgroup = Some(hamming_weight_metal_tile_threads);
                config
                    .hamming_weight_claim_reduction
                    .dispatch
                    .finalize_threads_per_threadgroup = Some(hamming_weight_metal_finalize_threads);
                config.hamming_weight_claim_reduction.trace_cutoff_elements = 1usize
                    .checked_shl(hamming_weight_metal_trace_cutoff_log2)
                    .expect("Hamming-weight Metal trace cutoff log2 must fit usize");
                config
                    .spartan_outer_remainder
                    .dispatch
                    .materialize_threads_per_threadgroup =
                    Some(outer_remainder_metal_materialize_threads);
                config
                    .spartan_outer_remainder
                    .dispatch
                    .stream_bind_threads_per_threadgroup =
                    Some(outer_remainder_metal_transition_threads);
                config
                    .spartan_outer_remainder
                    .dispatch
                    .transition_threads_per_threadgroup =
                    Some(outer_remainder_metal_transition_threads);
                config
                    .spartan_outer_remainder
                    .dispatch
                    .opening_threads_per_threadgroup = Some(outer_remainder_metal_output_threads);
                config.spartan_outer_remainder.dispatch.binding_plan =
                    match outer_remainder_metal_binding_plan {
                        OuterRemainderBindingPlan::BOnlyV1 => {
                            jolt_kernels::metal::solinas::OuterBindingPlan::BOnlyV1
                        }
                        OuterRemainderBindingPlan::BOnlyPadded56V1 => {
                            jolt_kernels::metal::solinas::OuterBindingPlan::BOnlyPadded56V1
                        }
                    };
                config.spartan_outer_remainder.dispatch.cpu_tail_elements = 1usize
                    .checked_shl(outer_remainder_metal_cutoff_log2)
                    .expect("outer-remainder Metal cutoff log2 must fit usize");
                config.spartan_outer_remainder.trace_cutoff_elements = 1usize
                    .checked_shl(outer_remainder_metal_trace_cutoff_log2)
                    .expect("outer-remainder Metal trace cutoff log2 must fit usize");
                println!(
                    "BYTECODE_METAL_CONFIG backend=metal cpu_tail={} trace_cutoff={} cutoff={} message_threads={} transition_threads={} max_threadgroups={}",
                    bytecode_cycle_algebra.as_str(),
                    config.bytecode_read_raf_cycle.trace_cutoff_elements,
                    config.bytecode_read_raf_cycle.cutoff_elements,
                    bytecode_metal_message_threads,
                    bytecode_metal_transition_threads,
                    bytecode_metal_max_threadgroups,
                );
                println!(
                    "INSTRUCTION_INPUT_METAL_CONFIG backend=metal trace_cutoff={} cutoff={} native_message_threads={} native_transition_threads={} dense_transition_threads={} storage_initialization={} native_primer=async",
                    config.instruction_input.trace_cutoff_elements,
                    config.instruction_input.cutoff_elements,
                    instruction_input_metal_native_message_threads,
                    instruction_input_metal_native_transition_threads,
                    instruction_input_metal_dense_transition_threads,
                    config
                        .instruction_input
                        .dispatch
                        .storage_initialization
                        .as_str(),
                );
                println!(
                    "BOOLEANITY_ADDRESS_METAL_CONFIG backend=metal trace_cutoff={} inner_log2={} selectors_per_tile={} tile_threads={} finalize_threads={}",
                    config.booleanity_address.trace_cutoff_elements,
                    booleanity_address_metal_inner_log2,
                    booleanity_address_metal_selectors_per_tile,
                    booleanity_address_metal_tile_threads,
                    booleanity_address_metal_finalize_threads,
                );
                println!(
                    "HAMMING_WEIGHT_METAL_CONFIG backend=metal trace_cutoff={} inner_log2={} selectors_per_tile={} tile_threads={} finalize_threads={}",
                    config
                        .hamming_weight_claim_reduction
                        .trace_cutoff_elements,
                    hamming_weight_metal_inner_log2,
                    hamming_weight_metal_selectors_per_tile,
                    hamming_weight_metal_tile_threads,
                    hamming_weight_metal_finalize_threads,
                );
                println!(
                    "OUTER_REMAINDER_METAL_CONFIG backend=metal trace_cutoff={} cutoff={} materialize_threads={} transition_threads={} output_threads={} max_threadgroups={} binding_plan={} storage_initialization={}",
                    config.spartan_outer_remainder.trace_cutoff_elements,
                    config.spartan_outer_remainder.dispatch.cpu_tail_elements,
                    outer_remainder_metal_materialize_threads,
                    outer_remainder_metal_transition_threads,
                    outer_remainder_metal_output_threads,
                    config.spartan_outer_remainder.dispatch.max_threadgroups,
                    outer_remainder_metal_binding_plan.as_str(),
                    config
                        .spartan_outer_remainder
                        .dispatch
                        .storage_initialization
                        .as_str(),
                );
                akita::JoltAkitaBackend::metal(config).expect("Metal backend should initialize")
            }
        };
        if backend_choice == Backend::Optimized {
            backend.base.bytecode_read_raf_cycle =
                Box::new(jolt_kernels::optimized::OptimizedBytecodeReadRafCycle::new(
                    optimized_bytecode_algebra,
                ));
        }

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
