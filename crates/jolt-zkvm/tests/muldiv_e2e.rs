//! End-to-end test: real muldiv guest → protocol.jolt → prove → verify.
//!
//! Exercises the full pipeline with a real RISC-V program:
//!   1. Compile + trace the muldiv guest via jolt-host (with small memory)
//!   2. Generate protocol.jolt from the ground-truth example (matching params)
//!   3. Link the Module to the CPU backend
//!   4. Build all witness, derived, and preprocessed polynomial data
//!   5. `prove()` — full Stage 1 + Stage 2
//!   6. `verify()` the proof

use std::process::Command;

use common::constants::RAM_START_ADDRESS;
use jolt_compiler::module::Module;
use jolt_compute::link;
use jolt_cpu::CpuBackend;
use jolt_field::{Field, Fr};
use jolt_host::{extract_trace, BytecodePreprocessing, CycleRow, Program};
use jolt_openings::mock::MockCommitmentScheme;
use jolt_r1cs::{constraints::rv64, R1csKey, R1csSource};
use jolt_transcript::{Blake2bTranscript, Transcript};
use jolt_verifier::{
    verify, JoltVerifyingKey, OneHotConfig, ProverConfig, ReadWriteConfig, TRANSCRIPT_LABEL,
};
use jolt_witness::derived::{DerivedSource, InstructionFlags};
use jolt_witness::preprocessed::PreprocessedSource;
use jolt_witness::provider::ProverData;
use jolt_witness::{PolynomialConfig, PolynomialId, Polynomials};
use jolt_zkvm::prove::prove;
use num_traits::Zero;

type MockPCS = MockCommitmentScheme<Fr>;

fn build_protocol_module(log_t: usize, log_k_bytecode: usize, log_k_ram: usize) -> Module {
    let tmp_path = format!("/tmp/jolt_muldiv_e2e_{log_t}_{log_k_bytecode}_{log_k_ram}.jolt");

    let output = Command::new("cargo")
        .args([
            "run",
            "--example",
            "jolt_core_module",
            "-p",
            "jolt-compiler",
            "-q",
            "--",
            "--log-t",
            &log_t.to_string(),
            "--log-k-bytecode",
            &log_k_bytecode.to_string(),
            "--log-k-ram",
            &log_k_ram.to_string(),
            "--emit",
            &tmp_path,
        ])
        .output()
        .expect("failed to run jolt_core_module example");

    assert!(
        output.status.success(),
        "jolt_core_module failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let bytes = std::fs::read(&tmp_path).expect("failed to read protocol binary");
    Module::from_bytes(&bytes)
}

/// Build the K-element I/O mask: 1 for addresses in the I/O range.
fn build_io_mask(config: &ProverConfig, ram_k: usize) -> Vec<Fr> {
    let mut mask = vec![Fr::zero(); ram_k];
    let start = config.input_word_offset;
    let end = config.termination_word_offset + 1;
    for m in mask.iter_mut().take(end.min(ram_k)).skip(start) {
        *m = Fr::from_u64(1);
    }
    mask
}

/// Build the K-element address unmap: unmap[k] = k * 8 + lowest_address.
fn build_ram_unmap(lowest_addr: u64, ram_k: usize) -> Vec<Fr> {
    (0..ram_k)
        .map(|k| Fr::from_u64(k as u64 * 8 + lowest_addr))
        .collect()
}

/// Build the K-element I/O values polynomial.
///
/// Packs input/output bytes into 8-byte LE words at their remapped addresses.
fn build_val_io(config: &ProverConfig, ram_k: usize) -> Vec<Fr> {
    let mut val_io = vec![Fr::zero(); ram_k];

    // Pack inputs
    for (i, chunk) in config.inputs.chunks(8).enumerate() {
        let k = config.input_word_offset + i;
        if k < ram_k {
            let mut word = 0u64;
            for (j, &b) in chunk.iter().enumerate() {
                word |= (b as u64) << (j * 8);
            }
            val_io[k] = Fr::from_u64(word);
        }
    }

    // Pack outputs
    for (i, chunk) in config.outputs.chunks(8).enumerate() {
        let k = config.output_word_offset + i;
        if k < ram_k {
            let mut word = 0u64;
            for (j, &b) in chunk.iter().enumerate() {
                word |= (b as u64) << (j * 8);
            }
            val_io[k] = Fr::from_u64(word);
        }
    }

    // Panic bit
    if config.panic_word_offset < ram_k {
        val_io[config.panic_word_offset] = Fr::from_u64(config.panic as u64);
    }

    // Termination bit (1 = program terminated normally)
    if config.termination_word_offset < ram_k {
        val_io[config.termination_word_offset] = Fr::from_u64(1);
    }

    val_io
}

/// Full end-to-end: trace real muldiv → prove (Stage 1 + Stage 2) → verify.
#[test]
fn muldiv_prove_verify() {
    // Reduce heap to keep T×K polynomials manageable.
    // Defaults: advice=4096, input=4096, output=4096, stack=4096.
    // The guest ELF stores to advice addresses, so those must stay at defaults.
    let mut program = Program::new("muldiv-guest");
    let _ = program.set_heap_size(4096);

    let (bytecode_raw, mut init_mem, _program_size, entry_address) = program.decode();
    let inputs_bytes = postcard::to_stdvec(&[9u32, 5u32, 3u32]).unwrap();
    let (_, trace, _, io_device) = program.trace(&inputs_bytes, &[], &[]);

    let bytecode = BytecodePreprocessing::preprocess(bytecode_raw, entry_address);
    let memory_layout = &io_device.memory_layout;

    // Merge I/O data into init_mem: decode() only returns ELF sections; the
    // tracer writes inputs to device memory before execution. Without this,
    // ram_init and val_final miss the input bytes and the output check fails.
    for (i, &byte) in io_device.inputs.iter().enumerate() {
        init_mem.push((memory_layout.input_start + i as u64, byte));
    }

    let trace_length = trace.len().next_power_of_two();
    let log_t = trace_length.trailing_zeros() as usize;
    let bytecode_k = bytecode.code_size;
    let log_k_bytecode = bytecode_k.trailing_zeros() as usize;

    let lowest_addr = memory_layout.get_lowest_address();
    let remap = |addr: u64| ((addr - lowest_addr) / 8) as usize;

    // Compute ram_k from actual memory layout
    let highest_addr = RAM_START_ADDRESS + memory_layout.get_total_memory_size();
    let ram_k_words = ((highest_addr - lowest_addr) / 8) as usize;
    let ram_k = ram_k_words.next_power_of_two();
    let log_k_ram = ram_k.trailing_zeros() as usize;

    let module = build_protocol_module(log_t, log_k_bytecode, log_k_ram);

    let backend = CpuBackend;
    let executable = link::<CpuBackend, Fr>(module, &backend);

    let one_hot = OneHotConfig::new(log_t);
    let log_k_chunk = one_hot.log_k_chunk as usize;
    let rw_config = ReadWriteConfig::new(log_t, log_k_ram);
    let config = ProverConfig {
        trace_length,
        ram_k,
        bytecode_k,
        one_hot_config: one_hot,
        rw_config,
        memory_start: RAM_START_ADDRESS,
        memory_end: highest_addr,
        entry_address,
        io_hash: [0u8; 32],
        max_input_size: memory_layout.max_input_size,
        max_output_size: memory_layout.max_output_size,
        heap_size: memory_layout.heap_size,
        inputs: io_device.inputs.clone(),
        outputs: io_device.outputs.clone(),
        panic: io_device.panic,
        ram_lowest_address: lowest_addr,
        input_word_offset: remap(memory_layout.input_start),
        output_word_offset: remap(memory_layout.output_start),
        panic_word_offset: remap(memory_layout.panic),
        termination_word_offset: remap(memory_layout.termination),
    };

    // ── Witness polynomials ──
    let poly_config = PolynomialConfig::new(log_k_chunk, 128, log_k_bytecode, log_k_ram);
    let matrices = rv64::rv64_constraints::<Fr>();
    let r1cs_key = R1csKey::new(matrices, trace_length);
    let (cycle_inputs, r1cs_witness, instruction_flag_data) = extract_trace::<_, Fr>(
        &trace,
        trace_length,
        &bytecode,
        memory_layout,
        r1cs_key.num_vars_padded,
    );

    let mut polys = Polynomials::<Fr>::new(poly_config);
    polys.push(&cycle_inputs);
    polys.finish();
    let _ = polys.insert(
        PolynomialId::UntrustedAdvice,
        vec![Fr::zero(); trace_length],
    );
    let _ = polys.insert(PolynomialId::TrustedAdvice, vec![Fr::zero(); trace_length]);

    // ── Derived polynomials (Stage 2) ──
    // Build initial/final RAM state as u64 word arrays for RamConfig.
    let initial_state = {
        let mut words = vec![0u64; ram_k];
        for &(addr, byte_val) in &init_mem {
            if addr < lowest_addr {
                continue;
            }
            let word_idx = ((addr - lowest_addr) / 8) as usize;
            let byte_offset = ((addr - lowest_addr) % 8) as usize;
            if word_idx < ram_k {
                words[word_idx] |= (byte_val as u64) << (byte_offset * 8);
            }
        }
        words
    };
    let final_state = {
        let mut words = initial_state.clone();
        for cycle in &trace {
            if cycle.is_noop() {
                continue;
            }
            if let (Some(addr), Some(_read_val), Some(write_val)) = (
                cycle.ram_access_address(),
                cycle.ram_read_value(),
                cycle.ram_write_value(),
            ) {
                let k = ((addr - lowest_addr) / 8) as usize;
                if k < ram_k {
                    words[k] = write_val;
                }
            }
        }
        words
    };
    let derived = DerivedSource::new(&r1cs_witness, trace_length, r1cs_key.num_vars_padded)
        .with_instruction_flags(InstructionFlags {
            is_noop: instruction_flag_data.is_noop,
            left_is_rs1: instruction_flag_data.left_is_rs1,
            left_is_pc: instruction_flag_data.left_is_pc,
            right_is_rs2: instruction_flag_data.right_is_rs2,
            right_is_imm: instruction_flag_data.right_is_imm,
        })
        .with_ram(jolt_witness::derived::RamConfig {
            ram_k,
            lowest_addr,
            initial_state: initial_state.clone(),
            final_state,
        });

    // ── Preprocessed polynomials (Stage 2) ──
    let mut preprocessed = PreprocessedSource::new();
    preprocessed.insert(PolynomialId::IoMask, build_io_mask(&config, ram_k));
    preprocessed.insert(PolynomialId::RamUnmap, build_ram_unmap(lowest_addr, ram_k));
    preprocessed.insert(PolynomialId::ValIo, build_val_io(&config, ram_k));
    preprocessed.insert(
        PolynomialId::RamInit,
        initial_state.iter().map(|&v| Fr::from_u64(v)).collect(),
    );

    let r1cs = R1csSource::new(&r1cs_key, &r1cs_witness);
    let mut provider = ProverData::new(&mut polys, r1cs, derived, preprocessed);

    // ── Prove ──
    let pcs_setup = ();
    let mut transcript = Blake2bTranscript::<Fr>::new(TRANSCRIPT_LABEL);
    let proof = prove::<_, _, _, MockPCS>(
        &executable,
        &mut provider,
        &backend,
        &pcs_setup,
        &mut transcript,
        config,
        None,
        None,
    );

    // ── Verify ──
    let vk = JoltVerifyingKey::<Fr, MockPCS>::new(&executable.module, (), r1cs_key);
    verify(&vk, &proof, &[0u8; 32]).expect("proof should verify");
}
