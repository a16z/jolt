use crate::{CHAINING_VALUE_SIZE, IV, MSG_BLOCK_SIZE};
/// ------------------------------------------------------------------------------------------------
/// Rust implementation of Blake2b-256 on the host.
/// ------------------------------------------------------------------------------------------------
use tracer::{
    emulator::cpu::Cpu,
    instruction::{inline::INLINE, RISCVInstruction},
};

/// Load words from memory into the provided slice
/// Returns an error if any memory access fails
fn load_words_from_memory(cpu: &mut Cpu, base_addr: u64, state: &mut [u32]) -> Result<(), String> {
    for (i, word) in state.iter_mut().enumerate() {
        *word = cpu
            .mmu
            .load_word(base_addr.wrapping_add((i * 4) as u64))
            .map_err(|e| {
                format!(
                    "BLAKE3: Failed to load from memory at offset {}: {:?}",
                    i * 4,
                    e
                )
            })?
            .0;
    }
    Ok(())
}

/// Store words to memory from the provided slice
/// Returns an error if any memory access fails
fn store_words_to_memory(cpu: &mut Cpu, base_addr: u64, values: &[u32]) -> Result<(), String> {
    for (i, &value) in values.iter().enumerate() {
        cpu.mmu
            .store_word(base_addr.wrapping_add((i * 4) as u64), value)
            .map_err(|e| {
                format!(
                    "BLAKE3: Failed to store to memory at offset {}: {:?}",
                    i * 4,
                    e
                )
            })?;
    }
    Ok(())
}

/// Blake2b inline execution function for the emulator
pub fn blake3_exec(
    instr: &INLINE,
    cpu: &mut Cpu,
    _ram_access: &mut <INLINE as RISCVInstruction>::RAMAccess,
) {
    // Memory addresses
    let state_addr = cpu.x[instr.operands.rs1 as usize] as u64;
    let block_addr = cpu.x[instr.operands.rs2 as usize] as u64;

    // 1. Read the 8-word chaining value from memory
    let mut chaining_value = [0u32; (CHAINING_VALUE_SIZE as usize) * 2];
    load_words_from_memory(cpu, state_addr, &mut chaining_value)
        .expect("Failed to load chaining value");

    // 2. Read the 16-word message block from memory
    let mut message_words = [0u32; MSG_BLOCK_SIZE as usize];
    load_words_from_memory(cpu, block_addr, &mut message_words)
        .expect("Failed to load message block");

    // 3. Load counter values from memory (2 words after message block)
    let mut counter = [0u32; 2];
    load_words_from_memory(
        cpu,
        block_addr.wrapping_add(MSG_BLOCK_SIZE as u64 * 4),
        &mut counter,
    )
    .expect("Failed to load counter");

    // 4. Load input bytes length (1 word after counter)
    let mut input_bytes = [0u32; 1];
    load_words_from_memory(
        cpu,
        block_addr.wrapping_add(MSG_BLOCK_SIZE as u64 * 4 + 8),
        &mut input_bytes,
    )
    .expect("Failed to load input bytes length");

    // 5. Load flags (1 word after input bytes length)
    let mut flags = [0u32; 1];
    load_words_from_memory(
        cpu,
        block_addr.wrapping_add(MSG_BLOCK_SIZE as u64 * 4 + 12),
        &mut flags,
    )
    .expect("Failed to load flags");

    // 6. Execute Blake3 compression function
    execute_blake3_compression(
        &mut chaining_value,
        &message_words,
        &counter,
        input_bytes[0],
        flags[0],
    );

    // 7. Write the result back to memory
    // Blake3 compression returns 16 words, but we only store the first 8 as chaining value
    store_words_to_memory(cpu, state_addr, &chaining_value).expect("Failed to store result");
}
/// ------------------------------------------------------------------------------------------------
/// Rust implementation of Blake3 compression on the host.
/// ------------------------------------------------------------------------------------------------
// The following code is copied from reference Blake3 implementation (https://github.com/BLAKE3-team/BLAKE3/blob/master/reference_impl/reference_impl.rs)

/// Execute Blake3 compression with explicit counter values
pub fn execute_blake3_compression(
    chaining_value: &mut [u32; 16],
    block_words: &[u32; 16],
    counter: &[u32; 2],
    block_len: u32,
    flags: u32,
) {
    #[rustfmt::skip]
    let mut state = [
        chaining_value[0], chaining_value[1], chaining_value[2], chaining_value[3],
        chaining_value[4], chaining_value[5], chaining_value[6], chaining_value[7],
        IV[0],             IV[1],             IV[2],             IV[3],
        counter[0],        counter[1],        block_len,         flags,
    ];
    let mut block = *block_words;

    round(&mut state, &block); // round 1
    permute(&mut block);
    round(&mut state, &block); // round 2
    permute(&mut block);
    round(&mut state, &block); // round 3
    permute(&mut block);
    round(&mut state, &block); // round 4
    permute(&mut block);
    round(&mut state, &block); // round 5
    permute(&mut block);
    round(&mut state, &block); // round 6
    permute(&mut block);
    round(&mut state, &block); // round 7

    for i in 0..8 {
        state[i] ^= state[i + 8];
        state[i + 8] ^= chaining_value[i];
    }
    for i in 0..16 {
        chaining_value[i] = state[i];
    }
}

// The mixing function, G, which mixes either a column or a diagonal.
fn g(state: &mut [u32; 16], a: usize, b: usize, c: usize, d: usize, mx: u32, my: u32) {
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(mx);
    state[d] = (state[d] ^ state[a]).rotate_right(16);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(12);
    state[a] = state[a].wrapping_add(state[b]).wrapping_add(my);
    state[d] = (state[d] ^ state[a]).rotate_right(8);
    state[c] = state[c].wrapping_add(state[d]);
    state[b] = (state[b] ^ state[c]).rotate_right(7);
}

fn round(state: &mut [u32; 16], m: &[u32; 16]) {
    // Mix the columns.
    g(state, 0, 4, 8, 12, m[0], m[1]);
    g(state, 1, 5, 9, 13, m[2], m[3]);
    g(state, 2, 6, 10, 14, m[4], m[5]);
    g(state, 3, 7, 11, 15, m[6], m[7]);
    // Mix the diagonals.
    g(state, 0, 5, 10, 15, m[8], m[9]);
    g(state, 1, 6, 11, 12, m[10], m[11]);
    g(state, 2, 7, 8, 13, m[12], m[13]);
    g(state, 3, 4, 9, 14, m[14], m[15]);
}

const MSG_PERMUTATION: [usize; 16] = [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8];

fn permute(m: &mut [u32; 16]) {
    let mut permuted = [0; 16];
    for i in 0..16 {
        permuted[i] = m[MSG_PERMUTATION[i]];
    }
    *m = permuted;
}
