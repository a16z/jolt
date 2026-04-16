use ark_serialize::{CanonicalSerialize, Compress};
use jolt_core::host;
use jolt_core::poly::commitment::dory::DoryGlobals;
use jolt_core::zkvm::prover::JoltProverPreprocessing;
use jolt_core::zkvm::verifier::{JoltSharedPreprocessing, JoltVerifierPreprocessing};
use jolt_core::zkvm::{RV64IMACProver, RV64IMACVerifier};

use super::{IterMetrics, StackOutcome, StackRunner};
use crate::measure::{time_it, PeakRssSampler};
use crate::programs::Program;

/// `max_trace_length` passed to `JoltSharedPreprocessing`. Matches what the
/// `#[jolt::provable]` macro stamps on example guest programs.
const MAX_TRACE_LENGTH: usize = 1 << 16;

pub struct CoreStack;

impl CoreStack {
    fn run_once(program: Program) -> IterMetrics {
        // Each iteration is a fresh Dory layout + preprocessing. This matches
        // how the jolt-core CLI invokes prove on a cold process.
        DoryGlobals::reset();

        let mut host_program = host::Program::new(program.guest_name());
        let (bytecode, init_memory_state, _program_size, e_entry) = host_program.decode();
        let inputs = program.canonical_inputs();

        let (_, _, _, io_device) = host_program.trace(&inputs, &[], &[]);

        let shared_preprocessing = JoltSharedPreprocessing::new(
            bytecode,
            io_device.memory_layout.clone(),
            init_memory_state,
            MAX_TRACE_LENGTH,
            e_entry,
        );
        let prover_preprocessing = JoltProverPreprocessing::new(shared_preprocessing);
        let elf_contents = host_program
            .get_elf_contents()
            .expect("guest ELF should be available after decode()");

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
        let program_io = prover.program_io.clone();

        // Prove — sampled for peak RSS.
        let sampler = PeakRssSampler::start();
        let (prove_ms, (proof, _debug_info)) = time_it(|| prover.prove());
        let peak_rss_mb = sampler.finish();

        let proof_bytes = proof.serialized_size(Compress::Yes) as u64;

        let verifier_preprocessing = Box::leak(Box::new(JoltVerifierPreprocessing::from(
            &prover_preprocessing,
        )));
        let verifier = RV64IMACVerifier::new(verifier_preprocessing, proof, program_io, None, None)
            .expect("build jolt-core verifier");
        let (verify_ms, verify_result) = time_it(|| verifier.verify());
        verify_result.expect("jolt-core verify should succeed");

        IterMetrics {
            prove_ms,
            verify_ms,
            peak_rss_mb,
            proof_bytes,
        }
    }
}

impl StackRunner for CoreStack {
    fn run(&self, program: Program, iters: usize, warmup: usize) -> StackOutcome {
        for _ in 0..warmup {
            let _ = Self::run_once(program);
        }
        let measurements = (0..iters).map(|_| Self::run_once(program)).collect();
        StackOutcome::Metrics(measurements)
    }
}
