//! eq-MLE field-inline example host: compiles the field-inline guest, traces it through
//! the modular stack, proves with the modular prover (the only field-inline-capable
//! prover), and verifies through the full `jolt_verifier::verify` entry.
//!
//! The whole pipeline requires `--features field-inline`; the guest body is
//! raw field-inline instructions and cannot run under the classic profile.

#[cfg(feature = "field-inline")]
mod pipeline {
    use std::sync::Arc;

    use jolt::host::JoltProgramSource;
    use jolt::{
        JoltProgramPreprocessing, JoltProverPreprocessing, JoltSharedPreprocessing, MemoryConfig,
        OwnedTrace, TraceInputs, TraceOutput, TracerBackend, VerifierField, VerifierPCS,
        VerifierTranscript, VerifierVC,
    };
    use jolt_field::{CanonicalBytes, Ring};
    use jolt_program::execution::{ExecutionBackend, JoltProgram, TraceRow};
    use jolt_prover::{JoltBackend, ProverConfig};
    use jolt_witness::{JoltVmWitnessConfig, JoltVmWitnessInputs, TraceBackend};

    pub type Fr = VerifierField;

    /// Matches the guest's `max_trace_length` attribute.
    pub const MAX_PADDED_TRACE_LENGTH: usize = 1 << 16;

    pub const PAIRS: [[u64; 2]; 4] = [[3, 5], [7, 2], [11, 13], [1, 9]];

    /// eq(r, x) = prod_i (r_i·x_i + (1 − r_i)(1 − x_i)) over the pairs, the
    /// host-side reference the guest's FIELD_ASSERT_EQ checks against.
    pub fn eq_mle(pairs: &[[u64; 2]; 4]) -> Fr {
        let one = Fr::from_u64(1);
        pairs.iter().fold(one, |acc, [r, x]| {
            let r = Fr::from_u64(*r);
            let x = Fr::from_u64(*x);
            acc * (r * x + (one - r) * (one - x))
        })
    }

    /// Canonical little-endian u64 limbs, the form the guest recomposes with
    /// its repeated-squaring 2^64 radix.
    pub fn limbs(value: Fr) -> [u64; 4] {
        let mut bytes = [0u8; 32];
        value.to_bytes_le(&mut bytes);
        let mut limbs = [0u64; 4];
        for (limb, chunk) in limbs.iter_mut().zip(bytes.chunks_exact(8)) {
            *limb = u64::from_le_bytes(chunk.try_into().expect("8-byte chunk"));
        }
        limbs
    }

    /// The generated prover's input encoding: each argument postcard-encoded
    /// and concatenated.
    pub fn guest_inputs(pairs: &[[u64; 2]; 4]) -> Vec<u8> {
        let mut inputs = jolt::postcard::to_stdvec(pairs).expect("serialize pairs");
        inputs.extend(
            jolt::postcard::to_stdvec(&limbs(eq_mle(pairs))).expect("serialize expected limbs"),
        );
        inputs
    }

    pub struct TracedGuest {
        pub preprocessing: JoltProverPreprocessing,
        pub trace_output: TraceOutput<OwnedTrace>,
        pub program: Arc<JoltProgram>,
    }

    /// Compile the field-inline guest, build profile-aware modular preprocessing,
    /// and re-trace through the modular tracer backend.
    pub fn compile_and_trace(inputs: &[u8]) -> TracedGuest {
        let target_dir = "/tmp/jolt-guest-targets";
        // The guest's field-inline feature flows through `compile_eval_eq_mle`,
        // which switches the program to the field-inline instruction profile.
        let mut program = guest::compile_eval_eq_mle(target_dir);

        let (_, _, _, io_device) = program.trace(inputs, &[], &[]);
        let jolt_program = Arc::new(
            program
                .build_jolt_program()
                .expect("build field-inline program"),
        );
        let program_preprocessing = JoltProgramPreprocessing::new(
            jolt_program.expanded_bytecode.clone(),
            jolt_program.memory_init.clone(),
            io_device.memory_layout.clone(),
            jolt_program.entry_address,
            MAX_PADDED_TRACE_LENGTH,
            program.instruction_profile(),
        )
        .expect("field-inline preprocessing");
        let preprocessing = jolt_prover::dory::from_shared(
            JoltSharedPreprocessing::new(program_preprocessing).expect("shared preprocessing"),
        );
        let memory_layout = &io_device.memory_layout;
        let memory_config = MemoryConfig {
            max_untrusted_advice_size: memory_layout.max_untrusted_advice_size,
            max_trusted_advice_size: memory_layout.max_trusted_advice_size,
            max_input_size: memory_layout.max_input_size,
            max_output_size: memory_layout.max_output_size,
            stack_size: memory_layout.stack_size,
            heap_size: memory_layout.heap_size,
            program_size: Some(memory_layout.program_size),
        };
        let trace_output = TracerBackend::new()
            .trace(
                &jolt_program,
                TraceInputs {
                    inputs: inputs.to_vec(),
                    untrusted_advice: Vec::new(),
                    trusted_advice: Vec::new(),
                    memory_config,
                    advice_tape: None,
                },
            )
            .expect("modular trace");
        TracedGuest {
            preprocessing,
            trace_output,
            program: jolt_program,
        }
    }

    pub fn field_inline_rows(rows: &[TraceRow]) -> usize {
        rows.iter().filter(|row| row.field_inline.is_some()).count()
    }

    /// Prove with the modular prover and verify through the full verifier
    /// entry; returns the guest output.
    pub fn prove_and_verify(traced: TracedGuest) -> u64 {
        let TracedGuest {
            preprocessing,
            trace_output,
            program,
        } = traced;
        let memory_layout = trace_output.device.memory_layout.clone();
        let public_io = trace_output.device.clone();
        let config = ProverConfig::derive::<Fr>(
            trace_output.trace.rows(),
            &memory_layout,
            preprocessing.verifier.program.min_bytecode_address(),
            preprocessing.verifier.program.program_image_len_words(),
            MAX_PADDED_TRACE_LENGTH,
        )
        .expect("derive config");

        let mut rows = trace_output.trace.rows().to_vec();
        rows.resize(config.trace_length, TraceRow::default());
        let padded_output = TraceOutput::new(
            OwnedTrace::new(rows),
            trace_output.device,
            trace_output.final_memory,
            trace_output.advice_tape,
        );

        let program_preprocessing = preprocessing
            .program_arc()
            .expect("full program preprocessing");
        let witness = TraceBackend::new(
            JoltVmWitnessConfig::new(
                config.trace_length.ilog2() as usize,
                config.ram_K,
                config.one_hot_config,
            ),
            JoltVmWitnessInputs::new(&program, &program_preprocessing, padded_output),
        )
        // field-inline proving needs the field-inline witness view; classic-profile
        // guests are refused rather than silently proven without field-inline columns.
        .with_field_inline()
        .expect("field-inline witness view");
        let witness = Arc::new(witness);

        let prover_preprocessing = preprocessing;
        let backend = JoltBackend::<Fr, VerifierPCS>::reference();
        let proof = jolt_prover::prove::<Fr, VerifierPCS, VerifierVC, VerifierTranscript, _>(
            &backend,
            &prover_preprocessing,
            &config,
            None,
            witness.as_ref(),
            &public_io,
        )
        .expect("modular field-inline prove");

        jolt::jolt_verifier::verify::<Fr, VerifierPCS, VerifierVC, VerifierTranscript>(
            &prover_preprocessing.verifier,
            &public_io,
            &proof,
            None,
        )
        .expect("modular field-inline proof must verify");

        let (output, _) =
            jolt::postcard::take_from_bytes::<u64>(&public_io.outputs).expect("decode output");
        output
    }
}

#[cfg(feature = "field-inline")]
fn main() {
    use pipeline::{
        compile_and_trace, eq_mle, field_inline_rows, guest_inputs, prove_and_verify, PAIRS,
    };

    let inputs = guest_inputs(&PAIRS);
    println!("eq(r, x) = {:?}", eq_mle(&PAIRS));

    let traced = compile_and_trace(&inputs);
    let rows = traced.trace_output.trace.rows();
    println!(
        "trace: {} cycles, {} field-inline",
        rows.len(),
        field_inline_rows(rows)
    );

    let output = prove_and_verify(traced);
    println!("output: {output}");
    assert_eq!(
        output, 42,
        "the field-inline assert-eq path must bridge out 42"
    );
    println!("valid: true");
}

#[cfg(not(feature = "field-inline"))]
fn main() {
    eprintln!(
        "the field-ops example is field-inline-only: \
         cargo run -p field-ops --features field-inline"
    );
    std::process::exit(1);
}

#[cfg(all(test, feature = "field-inline"))]
mod tests {
    use super::pipeline::{compile_and_trace, field_inline_rows, guest_inputs, PAIRS};

    /// The guest's static field-inline instruction budget: 2 accumulator seeds, 8 per
    /// coordinate pair (2 bridge loads, 3 muls, 2 subs, 1 add), 7 for the
    /// 2^64 radix (LoadImm 2 + 6 squarings), 10 for the expected-value Horner
    /// recomposition (4 bridge loads, 3 muls, 3 adds), the FIELD_ASSERT_EQ,
    /// 3 for the result value (sub, LoadImm 42, add), and the StoreToX
    /// bridge.
    const EXPECTED_FIELD_INLINE_CYCLES: usize = 2 + 8 * PAIRS.len() + 7 + 10 + 1 + 3 + 1;

    /// Commit-A scope: the guest builds and traces field-active — the tracer
    /// executes the field-inline semantics (a failed FIELD_ASSERT_EQ or an
    /// out-of-range StoreToX traps at trace time), so a completed trace
    /// already pins the eq-MLE math. The full prove/verify e2e lives in
    /// jolt-prover's field_inline_e2e suite.
    #[test]
    fn guest_traces_field_inline_active() {
        let traced = compile_and_trace(&guest_inputs(&PAIRS));
        let rows = traced.trace_output.trace.rows();
        let field_rows = field_inline_rows(rows);
        assert_eq!(field_rows, EXPECTED_FIELD_INLINE_CYCLES);
        assert!(
            field_rows < rows.len(),
            "the trace must also carry ordinary rows"
        );
        let (output, _) =
            jolt::postcard::take_from_bytes::<u64>(&traced.trace_output.device.outputs)
                .expect("decode output");
        assert_eq!(output, 42, "the StoreToX bridge must return 42");
    }
}
