use crate::sequence_builder::{pi_destination, ROTATION_OFFSETS, ROUND_CONSTANTS};
#[cfg(test)]
use crate::RATE_IN_BYTES;
use crate::{Keccak256State, NUM_LANES, RATE_IN_U64};

// Host-side Keccak-256 implementation for reference and testing.
#[cfg(test)]
pub(crate) fn execute_keccak256(msg: &[u8]) -> [u8; 32] {
    // NUM_LANES × 64-bit state lanes initialised to zero.
    let mut state = [0u64; NUM_LANES];

    // 1. Absorb full RATE blocks.
    let mut blocks = msg.chunks_exact(RATE_IN_BYTES);
    for block in &mut blocks {
        execute_absorb_permute(&mut state, &block_words(block));
    }

    // 2. Absorb the final (possibly empty) partial block with padding.
    let mut block = [0u8; RATE_IN_BYTES];
    let remaining = blocks.remainder();
    block[..remaining.len()].copy_from_slice(remaining);

    // Domain separation / padding (Keccak: 0x01 .. 0x80).
    block[remaining.len()] ^= 0x01; // 0x01 delimiter after the message.
    block[RATE_IN_BYTES - 1] ^= 0x80; // Final bit of padding.
    execute_absorb_permute(&mut state, &block_words(&block));

    // 3. Squeeze the first 32 bytes of the state as the hash output.
    let mut hash = [0u8; 32];
    for (i, lane) in state.iter().take(4).enumerate() {
        // 4 lanes * 8 bytes/lane = 32 bytes
        hash[i * 8..(i + 1) * 8].copy_from_slice(&lane.to_le_bytes());
    }
    hash
}

/// The little-endian lanes of a `RATE_IN_BYTES`-byte block.
#[cfg(test)]
fn block_words(block: &[u8]) -> [u64; RATE_IN_U64] {
    core::array::from_fn(|i| u64::from_le_bytes(block[8 * i..8 * i + 8].try_into().unwrap()))
}

/// XORs a rate block into the state and applies Keccak-f[1600]: the model of
/// the absorb-and-permute inlines.
pub(crate) fn execute_absorb_permute(state: &mut Keccak256State, block: &[u64; RATE_IN_U64]) {
    for (lane, word) in state.iter_mut().zip(block) {
        *lane ^= word;
    }
    execute_keccak_f(state);
}

/// Absorbs a rate block into the zero state: the model of the INIT inlines.
pub(crate) fn execute_init_absorb_permute(block: &[u64; RATE_IN_U64]) -> Keccak256State {
    let mut state = [0; NUM_LANES];
    execute_absorb_permute(&mut state, block);
    state
}

/// Executes the 24-round Keccak-f[1600] permutation.
pub(crate) fn execute_keccak_f(state: &mut Keccak256State) {
    for rc in ROUND_CONSTANTS {
        execute_theta(state);
        execute_rho_and_pi(state);
        execute_chi(state);
        execute_iota(state, rc);
    }
}
/// The `theta` step of the Keccak-f permutation mixes columns to provide diffusion.
/// This step XORs each bit in the state with the parities of two columns in the state array.
pub(crate) fn execute_theta(state: &mut Keccak256State) {
    // 1. Compute the parity of each of the 5 columns (an array `C` of 5 lanes).
    let mut c = [0u64; 5];
    for x in 0..5 {
        c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
    }
    // 2. Compute `D[x] = C[x-1] ^ rotl64(C[x+1], 1)`
    let mut d = [0u64; 5];
    for x in 0..5 {
        d[x] = c[(x + 4) % 5] ^ c[(x + 1) % 5].rotate_left(1);
    }
    // 3. XOR `D[x]` into each lane in column `x`.
    for x in 0..5 {
        for y in 0..5 {
            state[x + 5 * y] ^= d[x];
        }
    }
}

/// The `rho` and `pi` steps of the Keccak-f permutation shuffles the state to provide diffusion.
/// `rho` rotates each lane by a different fixed offset. `pi` permutes positions of the lanes.
pub(crate) fn execute_rho_and_pi(state: &mut Keccak256State) {
    let mut b = [0u64; NUM_LANES];
    for x in 0..5 {
        for y in 0..5 {
            let (nx, ny) = pi_destination((x, y));
            b[nx + 5 * ny] = state[x + 5 * y].rotate_left(ROTATION_OFFSETS[x][y]);
        }
    }
    state.copy_from_slice(&b);
}

/// The `chi` step of the Keccak-f permutation introduces non-linearity (relationships between input and output).
pub(crate) fn execute_chi(state: &mut Keccak256State) {
    // For each row, it updates each lane as: lane[x] ^= (~lane[x+1] & lane[x+2])
    // This ensures output bit is a non-linear function of three input bits.
    for y in 0..5 {
        let mut row = [0u64; 5];
        for x in 0..5 {
            row[x] = state[x + 5 * y];
        }
        for x in 0..5 {
            state[x + 5 * y] = row[x] ^ (!row[(x + 1) % 5] & row[(x + 2) % 5]);
        }
    }
}

/// The `iota` step of Keccak-f breaks the symmetry of the rounds by injecting a round constant into the first lane.
pub(crate) fn execute_iota(state: &mut Keccak256State, round_constant: u64) {
    state[0] ^= round_constant; // Inject round constant.
}
