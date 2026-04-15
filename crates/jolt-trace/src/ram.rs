//! RAM state construction.
//!
//! Builds K-element initial and final memory state arrays from:
//! - `init_mem`: byte-level address-value pairs from ELF decoding
//! - `Memory`: final tracer memory state
//! - `JoltDevice`: I/O data (inputs, outputs, advice, panic/termination)
//!
//! Mirrors jolt-core's `gen_ram_memory_states` but uses jolt-host types.

use common::constants::RAM_START_ADDRESS;
use common::jolt_device::JoltDevice;
use tracer::emulator::memory::Memory;

/// Build K-element initial and final RAM state arrays.
///
/// The initial state is populated from:
/// 1. ELF bytecode image (`init_mem` byte pairs → 8-byte LE words)
/// 2. I/O region: inputs, trusted/untrusted advice
///
/// The final state is populated from:
/// 1. Final tracer memory (addresses >= `RAM_START_ADDRESS`)
/// 2. I/O region: same as initial + outputs + panic/termination bits
///
/// Both arrays are indexed by remapped address `k = (addr - lowest) / 8`.
pub fn build_ram_states(
    init_mem: &[(u64, u8)],
    final_memory: &Memory,
    io_device: &JoltDevice,
    ram_k: usize,
) -> (Vec<u64>, Vec<u64>) {
    let layout = &io_device.memory_layout;
    let lowest = layout.get_lowest_address();
    let remap = |addr: u64| ((addr - lowest) / 8) as usize;

    let mut initial = vec![0u64; ram_k];
    let mut final_state = vec![0u64; ram_k];

    // 1. ELF bytecode image into initial state
    for &(addr, byte_val) in init_mem {
        if addr < lowest {
            continue;
        }
        let word_idx = remap(addr);
        let byte_offset = ((addr - lowest) % 8) as u32;
        if word_idx < ram_k {
            initial[word_idx] |= (byte_val as u64) << (byte_offset * 8);
        }
    }

    // 2. I/O region: inputs into both initial and final
    populate_words(
        remap(layout.input_start),
        &io_device.inputs,
        &mut initial,
        &mut final_state,
    );

    // 3. Trusted advice into both
    populate_words(
        remap(layout.trusted_advice_start),
        &io_device.trusted_advice,
        &mut initial,
        &mut final_state,
    );

    // 4. Untrusted advice into both
    populate_words(
        remap(layout.untrusted_advice_start),
        &io_device.untrusted_advice,
        &mut initial,
        &mut final_state,
    );

    // 5. Final memory (DRAM region, addresses >= RAM_START_ADDRESS)
    let dram_start = remap(RAM_START_ADDRESS);
    let num_words = final_memory
        .data
        .get_num_doublewords()
        .min(ram_k - dram_start);
    for k in 0..num_words {
        final_state[dram_start + k] = final_memory.get_doubleword(8 * k as u64);
    }

    // 6. Outputs into final state only
    populate_words_final(
        remap(layout.output_start),
        &io_device.outputs,
        &mut final_state,
    );

    // 7. Panic and termination bits
    let panic_idx = remap(layout.panic);
    final_state[panic_idx] = io_device.panic as u64;
    if !io_device.panic {
        let term_idx = remap(layout.termination);
        final_state[term_idx] = 1;
    }

    (initial, final_state)
}

/// Pack bytes into 8-byte LE words at `start_idx` in both arrays.
fn populate_words(start_idx: usize, bytes: &[u8], initial: &mut [u64], final_state: &mut [u64]) {
    for (i, chunk) in bytes.chunks(8).enumerate() {
        let idx = start_idx + i;
        if idx < initial.len() {
            let word = pack_le(chunk);
            initial[idx] = word;
            final_state[idx] = word;
        }
    }
}

/// Pack bytes into 8-byte LE words at `start_idx` in final state only.
fn populate_words_final(start_idx: usize, bytes: &[u8], final_state: &mut [u64]) {
    for (i, chunk) in bytes.chunks(8).enumerate() {
        let idx = start_idx + i;
        if idx < final_state.len() {
            final_state[idx] = pack_le(chunk);
        }
    }
}

fn pack_le(bytes: &[u8]) -> u64 {
    let mut word = 0u64;
    for (j, &b) in bytes.iter().enumerate() {
        word |= (b as u64) << (j * 8);
    }
    word
}
