use crate::{IV, SIGMA};
use tracer::{
    emulator::cpu::Cpu,
    instruction::{inline::INLINE, RISCVInstruction},
};

pub fn blake2b_exec(
    _instr: &INLINE,
    _cpu: &mut Cpu,
    _ram_access: &mut <INLINE as RISCVInstruction>::RAMAccess,
) {
}

/// Rust implementation of BLAKE2 compression on the host.
pub fn execute_blake2b_compression(state: &mut [u64; 8], message_words: &[u64; 18]) {
    let mut v = [0u64; 16];
    v[0..8].copy_from_slice(state);
    v[8..16].copy_from_slice(&IV);

    v[12] ^= message_words[16];
    // v[13] ^= counter.shr(64) as u64;  // not used for 64-bit counter

    if message_words[17] != 0 {
        v[14] = !v[14];
    }

    for s in SIGMA {
        // Column step
        g(
            &mut v,
            0,
            4,
            8,
            12,
            message_words[s[0]],
            message_words[s[1]],
        );
        g(
            &mut v,
            1,
            5,
            9,
            13,
            message_words[s[2]],
            message_words[s[3]],
        );
        g(
            &mut v,
            2,
            6,
            10,
            14,
            message_words[s[4]],
            message_words[s[5]],
        );
        g(
            &mut v,
            3,
            7,
            11,
            15,
            message_words[s[6]],
            message_words[s[7]],
        );

        // Diagonal step
        g(
            &mut v,
            0,
            5,
            10,
            15,
            message_words[s[8]],
            message_words[s[9]],
        );
        g(
            &mut v,
            1,
            6,
            11,
            12,
            message_words[s[10]],
            message_words[s[11]],
        );
        g(
            &mut v,
            2,
            7,
            8,
            13,
            message_words[s[12]],
            message_words[s[13]],
        );
        g(
            &mut v,
            3,
            4,
            9,
            14,
            message_words[s[14]],
            message_words[s[15]],
        );
    }

    for i in 0..8 {
        state[i] ^= v[i] ^ v[i + 8];
    }
}

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
