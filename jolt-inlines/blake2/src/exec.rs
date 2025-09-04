/// ------------------------------------------------------------------------------------------------
/// Rust implementation of Blake2b-256 on the host.
/// ------------------------------------------------------------------------------------------------
use tracer::{
    emulator::cpu::Cpu,
    instruction::{inline::INLINE, RISCVInstruction},
};

// Blake2b initialization vector (IV)
const BLAKE2B_IV: [u64; 8] = [
    0x6a09e667f3bcc908,
    0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b,
    0xa54ff53a5f1d36f1,
    0x510e527fade682d1,
    0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b,
    0x5be0cd19137e2179,
];

// Blake2b sigma permutation table for 12 rounds
const SIGMA: [[usize; 16]; 12] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
    [11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4],
    [7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8],
    [9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13],
    [2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9],
    [12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11],
    [13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10],
    [6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5],
    [10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0],
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3],
];

/// Blake2b inline execution function for the emulator
pub fn blake2b_exec(
    instr: &INLINE,
    cpu: &mut Cpu,
    _ram_access: &mut <INLINE as RISCVInstruction>::RAMAccess,
) {
    // This is the "fast path" for emulation without tracing.
    // It performs the Blake2b compression using a native Rust implementation.

    // 1. Read the 8-word (64-byte) state from memory pointed to by rs1.
    let mut state: [u64; 8] = [0u64; 8];
    let state_addr = cpu.x[instr.operands.rs1 as usize] as u64;
    for (i, lane) in state.iter_mut().enumerate() {
        *lane = cpu
            .mmu
            .load_doubleword(state_addr.wrapping_add((i * 8) as u64))
            .expect("BLAKE2B: Failed to load state from memory")
            .0;
    }

    // 2. Read the 18-word message (16 message words + counter + is_final flag) from memory pointed to by rs2.
    let mut message_words: [u64; 18] = [0u64; 18];
    let message_addr = cpu.x[instr.operands.rs2 as usize] as u64;
    for (i, word) in message_words.iter_mut().enumerate() {
        *word = cpu
            .mmu
            .load_doubleword(message_addr.wrapping_add((i * 8) as u64))
            .expect("BLAKE2B: Failed to load message from memory")
            .0;
    }

    // 3. Execute the Blake2b compression on the state.
    execute_blake2b_compression(&mut state, &message_words);

    // 4. Write the compressed state back to memory.
    for (i, &lane) in state.iter().enumerate() {
        cpu.mmu
            .store_doubleword(state_addr.wrapping_add((i * 8) as u64), lane)
            .expect("BLAKE2B: Failed to store state to memory");
    }
}

/// Execute Blake2b compression with explicit counter values
#[rustfmt::skip]
pub fn execute_blake2b_compression(
    state: &mut [u64; 8],
    message_words: &[u64; 18],
) {
    // Initialize working variables
    let mut v = [0u64; 16];
    v[0..8].copy_from_slice(state);
    v[8..16].copy_from_slice(&BLAKE2B_IV);

    // Blake2b counter handling: XOR counter values with v[12] and v[13]
    v[12] ^= message_words[16]; // counter_low
    // v[13] ^= counter.shr(64) as u64;  // counter_high (not used for 64-bit counter)

    // Set final block flag if this is the last block
    if message_words[17] != 0 {
        v[14] = !v[14]; // Invert v[14] for final block
    }

    // 12 rounds of mixing
    for s in SIGMA {
        // Column step
        g(&mut v, 0, 4, 8, 12, message_words[s[0]], message_words[s[1]]);
        g(&mut v, 1, 5, 9, 13, message_words[s[2]], message_words[s[3]]);
        g(&mut v, 2, 6, 10, 14, message_words[s[4]], message_words[s[5]]);
        g(&mut v, 3, 7, 11, 15, message_words[s[6]], message_words[s[7]]);

        // Diagonal step
        g(&mut v, 0, 5, 10, 15, message_words[s[8]], message_words[s[9]]);
        g(&mut v, 1, 6, 11, 12, message_words[s[10]], message_words[s[11]]);
        g(&mut v, 2, 7, 8, 13, message_words[s[12]], message_words[s[13]]);
        g(&mut v, 3, 4, 9, 14, message_words[s[14]], message_words[s[15]]);
    }

    // Finalize hash state
    for i in 0..8 {
        state[i] ^= v[i] ^ v[i + 8];
    }
}

// Blake2b G function
fn g(v: &mut [u64; 16], a: usize, b: usize, c: usize, d: usize, x: u64, y: u64) {
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(x);
    v[d] = (v[d] ^ v[a]).rotate_right(32);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(24);
    v[a] = v[a].wrapping_add(v[b]).wrapping_add(y);
    v[d] = (v[d] ^ v[a]).rotate_right(16);
    v[c] = v[c].wrapping_add(v[d]);
    v[b] = (v[b] ^ v[c]).rotate_right(63);
}
