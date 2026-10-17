//! Compare CUDA and optimized clear-mode Dory proofs using the same guest inputs.

#[cfg(any(feature = "akita", feature = "zk", feature = "field-inline"))]
use std::error::Error;

#[cfg(any(feature = "akita", feature = "zk", feature = "field-inline"))]
fn main() -> Result<(), Box<dyn Error>> {
    Err("cuda_profile requires the ordinary clear-mode Dory profile".into())
}

#[cfg(not(any(feature = "akita", feature = "zk", feature = "field-inline")))]
fn main() {
    profile::run();
}

#[cfg(not(any(feature = "akita", feature = "zk", feature = "field-inline")))]
#[expect(
    clippy::expect_used,
    clippy::print_stdout,
    reason = "the profiling executable reports measurements and fails on invalid runs"
)]
mod profile {
    use std::sync::Arc;
    use std::time::Instant;

    use clap::{Parser, ValueEnum};
    use common::jolt_device::MemoryConfig;
    use jolt_crypto::{Bn254G1, Pedersen};
    use jolt_dory::DoryScheme;
    use jolt_field::Fr;
    use jolt_host::{JoltProgramSource, Program};
    use jolt_inlines_keccak256 as _;
    use jolt_inlines_sha2 as _;
    use jolt_kernels::cuda::CudaDoryScheme;
    use jolt_kernels::JoltBackend;
    use jolt_profiling::PeakMemory;
    use jolt_program::execution::{ExecutionBackend, TraceInputs};
    use jolt_program::preprocess::JoltProgramPreprocessing;
    use jolt_prover::{dory, JoltProverPreprocessing, JoltSharedPreprocessing, ProverConfig};
    use jolt_transcript::LegacyBlake2bTranscript;
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};
    use serde_json::json;
    use tracer::execution_backend::TracerBackend;

    #[derive(Clone, Copy, Debug, ValueEnum)]
    enum Workload {
        Fibonacci,
        Sha2Chain,
        Sha3Chain,
    }

    #[derive(Clone, Copy, Debug, PartialEq, ValueEnum)]
    enum Backend {
        Optimized,
        Cuda,
    }

    #[derive(Parser)]
    struct Args {
        #[arg(long, value_enum, default_value = "fibonacci")]
        name: Workload,
        /// Fibonacci iterations or chain hashes; the reported trace size is measured.
        #[arg(long, default_value_t = 4000)]
        iterations: u32,
        #[arg(long, value_enum, default_value = "cuda")]
        backend: Backend,
        #[arg(long, default_value_t = 2)]
        repetitions: usize,
        /// Run address rounds before cycle rounds in RAM/register read-write checks.
        #[arg(long)]
        address_first: bool,
    }

    pub fn run() {
        let args = Args::parse();
        assert!(args.iterations > 0 && args.repetitions > 0);
        let (name, inputs) = match args.name {
            Workload::Fibonacci => (
                "fibonacci-guest",
                postcard::to_stdvec(&args.iterations).expect("fibonacci input"),
            ),
            Workload::Sha2Chain | Workload::Sha3Chain => (
                if matches!(args.name, Workload::Sha2Chain) {
                    "sha2-chain-guest"
                } else {
                    "sha3-chain-guest"
                },
                postcard::to_stdvec(&([5u8; 32], args.iterations)).expect("chain input"),
            ),
        };
        let mut source = Program::new(name);
        let (_, sizing, _, device) = source.trace(&inputs, &[], &[]);
        assert!(!device.panic, "guest execution failed");
        let trace_bound = (sizing.len() + 1).next_power_of_two();
        drop(sizing);
        let layout = device.memory_layout;
        let program = Arc::new(source.build_jolt_program().expect("build Jolt program"));
        let preprocessing = JoltProgramPreprocessing::new(
            program.expanded_bytecode.clone(),
            program.memory_init.clone(),
            layout.clone(),
            program.entry_address,
            trace_bound,
            source.instruction_profile(),
        )
        .expect("program preprocessing");
        let trace = TracerBackend::new()
            .trace(
                &program,
                TraceInputs::new(
                    inputs,
                    Vec::new(),
                    Vec::new(),
                    MemoryConfig {
                        max_untrusted_advice_size: layout.max_untrusted_advice_size,
                        max_trusted_advice_size: layout.max_trusted_advice_size,
                        max_input_size: layout.max_input_size,
                        max_output_size: layout.max_output_size,
                        stack_size: layout.stack_size,
                        heap_size: layout.heap_size,
                        program_size: Some(layout.program_size),
                    },
                ),
            )
            .expect("modular trace");
        let mut config = ProverConfig::derive_from_dimensions::<Fr>(
            trace.dimensions,
            &layout,
            preprocessing.ram.min_bytecode_address,
            preprocessing.ram.bytecode_words.len(),
            trace_bound,
        )
        .expect("derive config");
        if args.address_first {
            config.rw_config.ram_rw_phase1_num_rounds = 0;
            config.rw_config.registers_rw_phase1_num_rounds = 0;
        }
        let setup_start = Instant::now();
        let cpu_preprocessing = dory::from_shared(
            JoltSharedPreprocessing::new(preprocessing).expect("shared preprocessing"),
        )
        .expect("Dory setup");
        let setup_seconds = setup_start.elapsed().as_secs_f64();
        let cuda_preprocessing = JoltProverPreprocessing::<CudaDoryScheme, Pedersen<Bn254G1>> {
            verifier: CudaDoryScheme::adopt_verifier_preprocessing(
                cpu_preprocessing.verifier.clone(),
            )
            .expect("CUDA verifier preprocessing"),
            pcs_setup: cpu_preprocessing.pcs_setup.clone(),
            committed_program: None,
        };
        let program_preprocessing = cpu_preprocessing.program_arc().expect("full program");
        let public_io = trace.device.clone();
        let witness = TraceBackend::new(
            JoltVmWitnessConfig::new(
                config.trace_length.ilog2() as usize,
                config.ram_K,
                config.one_hot_config,
            ),
            JoltVmWitnessInputs::new(&program, &program_preprocessing, trace),
        );
        let cpu = JoltBackend::<Fr, DoryScheme>::optimized();
        let cuda = JoltBackend::<Fr, CudaDoryScheme>::cuda();
        for repetition in 0..args.repetitions {
            macro_rules! prove {
                ($pcs:ty, $backend:expr, $preprocessing:expr) => {{
                    let start = Instant::now();
                    let proof =
                        dory::prove::<Fr, $pcs, Pedersen<Bn254G1>, LegacyBlake2bTranscript>(
                            $backend.with_witness(&witness),
                            $preprocessing,
                            &config,
                            None,
                            &public_io,
                        )
                        .expect("proof");
                    let seconds = start.elapsed().as_secs_f64();
                    let verify_start = Instant::now();
                    jolt_verifier::verify::<Fr, $pcs, Pedersen<Bn254G1>, LegacyBlake2bTranscript>(
                        &$preprocessing.verifier,
                        &public_io,
                        &proof,
                        None,
                    )
                    .expect("proof verifies");
                    let verify_seconds = verify_start.elapsed().as_secs_f64();
                    let size = bincode::serde::encode_to_vec(&proof, bincode::config::standard())
                        .expect("serialize proof")
                        .len();
                    (seconds, verify_seconds, size)
                }};
            }
            let (seconds, verify_seconds, proof_bytes) = match args.backend {
                Backend::Optimized => prove!(DoryScheme, cpu, &cpu_preprocessing),
                Backend::Cuda => prove!(CudaDoryScheme, cuda, &cuda_preprocessing),
            };
            println!(
                "{}",
                json!({
                    "workload": name,
                    "iterations": args.iterations,
                    "trace_length": config.trace_length,
                    "backend": format!("{:?}", args.backend).to_lowercase(),
                    "repetition": repetition,
                    "address_first": args.address_first,
                    "setup_seconds": setup_seconds,
                    "prove_seconds": seconds,
                    "verify_seconds": verify_seconds,
                    "proof_bytes": proof_bytes,
                    "process_peak_rss_bytes": PeakMemory::sample().rss_bytes,
                })
            );
        }
    }
}
