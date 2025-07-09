/// This file contains Keccak256-specific logic to be used in the Keccak256 inline:
/// 1) Prover: Keccak256SequenceBuilder expands the inline to a list of RV instructions.
/// 2) Host: Rust reference implementation to be called by jolt-sdk.
///
/// Keccak is a hash function that uses a sponge construction. The spone absorbs (and permutes) data. Each permutation has 24 rounds. Then squeezes out the hash.
/// Glossary:
///   - “Lane”  = one 64-bit word in the 5×5 state matrix (25 lanes total).
///   - “Round” = single application of θ ρ π χ ι to the state.
///   - “Rate”  = 1088 bits (136 B) that interact with the message/output.
///   - “Capacity” = 512 bits hidden from the attacker (1600 − 1088).
///   - “Permutation” = Keccak-f[1600] : 24 rounds, each θ→ρ→π→χ→ι.
/// Keccak256 refers to the specific variant where the rate is 1088 bits and the capacity is 512 bits.
/// Keccak256 differs from SHA3-256 (not implemented here) in the padding scheme.
use crate::instruction::RV32IMInstruction;

pub mod keccak256;

pub const NEEDED_REGISTERS: usize = 96;

/// The 24 round constants for the Keccak-f[1600] permutation.
/// These values are XORed into the state during the `iota` step of each round.
#[rustfmt::skip]
const ROUND_CONSTANTS: [u64; 24] = [
    0x0000000000000001, 0x0000000000008082,
    0x800000000000808a, 0x8000000080008000,
    0x000000000000808b, 0x0000000080000001,
    0x8000000080008081, 0x8000000000008009,
    0x000000000000008a, 0x0000000000000088,
    0x0000000080008009, 0x000000008000000a,
    0x000000008000808b, 0x800000000000008b,
    0x8000000000008089, 0x8000000000008003,
    0x8000000000008002, 0x8000000000000080,
    0x000000000000800a, 0x800000008000000a,
    0x8000000080008081, 0x8000000000008080,
    0x0000000080000001, 0x8000000080008008,
];

/// The rotation offsets for the `rho` step of the Keccak-f[1600] permutation.
/// The state is organized as a 5x5 matrix of 64-bit lanes, and `ROTATION_OFFSETS[y][x]`
/// specifies the left-rotation amount for the lane at `(x, y)`. Also known as rotation constants.
#[rustfmt::skip]
const ROTATION_OFFSETS: [[u32; 5]; 5] = [
    [ 0, 36,  3, 41, 18],
    [ 1, 44, 10, 45,  2],
    [62,  6, 43, 15, 61],
    [28, 55, 25, 21, 56],
    [27, 20, 39,  8, 14],
];

struct Keccak256SequenceBuilder {
    address: u64,
    sequence: Vec<RV32IMInstruction>,
    round: u32,
    vr: [usize; NEEDED_REGISTERS],
    operand_rs1: usize,
    operand_rs2: usize,
    initial: bool,
}

/// `Keccak256SequenceBuilder` is a helper struct for constructing the virtual instruction
/// sequence required to emulate the Keccak-256 hashing operation within the RISC-V
/// instruction set. This builder is responsible for generating the correct sequence of
/// `RV32IMInstruction` instances that together perform the Keccak-256 permutation and
/// hashing steps, using a set of virtual registers to hold intermediate state.
///
/// # Fields
/// - `address`: The starting program counter address for the sequence.
/// - `sequence`: The vector of generated instructions representing the Keccak-256 operation.
/// - `round`: The current round of the Keccak permutation (0..24).
/// - `vr`: An array of virtual register indices used for state and temporary values.
/// - `operand_rs1`: The source register index for the first operand (input state pointer).
/// - `operand_rs2`: The source register index for the second operand (input length or auxiliary).
/// - `initial`: Whether this is the initial invocation (affects state initialization).
///
/// # Usage
/// Typically, you construct a `Keccak256SequenceBuilder` with the required register mapping
/// and operands, then call `.build()` to obtain the full instruction sequence for the
/// Keccak-256 operation. This is used to inline the Keccak-256 hash logic into the
/// RISC-V instruction stream for tracing or emulation purposes.
///
/// ```ignore
/// let builder = Keccak256SequenceBuilder::new(
///     address,
///     vr,
///     operand_rs1,
///     operand_rs2,
///     initial,
/// );
/// let keccak_sequence = builder.build();
/// // `keccak_sequence` now contains the instructions to perform Keccak-256.
/// ```
///
/// # Note
/// The actual Keccak-256 logic is implemented in the `build` method, which generates
/// the appropriate instruction sequence. This struct is not intended for direct execution,
/// but rather for constructing instruction traces or emulation flows.

impl Keccak256SequenceBuilder {
    fn new(
        address: u64,
        vr: [usize; NEEDED_REGISTERS],
        operand_rs1: usize,
        operand_rs2: usize,
        initial: bool,
    ) -> Self {
        Keccak256SequenceBuilder {
            address,
            sequence: vec![],
            round: 0,
            vr,
            operand_rs1,
            operand_rs2,
            initial,
        }
    }

    fn build(mut self) -> Vec<RV32IMInstruction> {
        // TODO: Implement.
        vec![]
    }
}

/// ------------------------------------------------------------------------------------------------
/// Rust implementation of Keccak-256 on the host.
/// ------------------------------------------------------------------------------------------------

// Host-side Keccak-256 implementation for reference and testing.
pub fn execute_keccak256(msg: &[u8]) -> [u8; 32] {
    // Keccak-256 parameters.
    const RATE_IN_BYTES: usize = 136; // 1088-bit rate

    // 25 × 64-bit state lanes initialised to zero.
    let mut state = [0u64; 25];

    // 1. Absorb full RATE blocks.
    let mut offset = 0;
    while offset + RATE_IN_BYTES <= msg.len() {
        // XOR message block into the state.
        for (i, lane_bytes) in msg[offset..offset + RATE_IN_BYTES]
            .chunks_exact(8)
            .enumerate()
        {
            state[i] ^= u64::from_le_bytes(lane_bytes.try_into().unwrap());
        }

        // Apply the Keccak-f permutation after each full block.
        execute_keccak_f(&mut state);
        offset += RATE_IN_BYTES;
    }

    // 2. Absorb the final (possibly empty) partial block with padding.
    let mut block = [0u8; RATE_IN_BYTES];
    let remaining = &msg[offset..];
    block[..remaining.len()].copy_from_slice(remaining);

    // Domain separation / padding (Keccak: 0x01 .. 0x80).
    block[remaining.len()] ^= 0x01; // 0x01 delimiter after the message.
    block[RATE_IN_BYTES - 1] ^= 0x80; // Final bit of padding.

    // XOR padded block into the state and permute once more.
    for (i, lane_bytes) in block.chunks_exact(8).enumerate() {
        state[i] ^= u64::from_le_bytes(lane_bytes.try_into().unwrap());
    }
    execute_keccak_f(&mut state);

    // 3. Squeeze the first 32 bytes of the state as the hash output.
    let mut hash = [0u8; 32];
    for (i, lane) in state.iter().take(4).enumerate() {
        // 4 lanes * 8 bytes/lane = 32 bytes
        hash[i * 8..(i + 1) * 8].copy_from_slice(&lane.to_le_bytes());
    }
    hash
}

/// Executes the 24-round Keccak-f[1600] permutation.
pub fn execute_keccak_f(state: &mut [u64; 25]) {
    for rc in ROUND_CONSTANTS {
        execute_theta(state);
        execute_rho_and_pi(state);
        execute_chi(state);
        execute_iota(state, rc);
    }
}

/// The `theta` step of the Keccak-f permutation mixes columns to provide diffusion.
/// This step XORs each bit in the state with the parities of two columns in the state array.
fn execute_theta(state: &mut [u64; 25]) {
    // 1. Compute the parity of each of the 5 columns (an array `C` of 5 lanes).
    let mut c = [0u64; 5];
    for x in 0..5 {
        c[x] = state[x] ^ state[x + 5] ^ state[x + 10] ^ state[x + 15] ^ state[x + 20];
    }
    // 2. Compute `D[x] = C[x-1] ^ rotl64(C[x+1], 1)` for each column `x`.
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
fn execute_rho_and_pi(state: &mut [u64; 25]) {
    let mut b = [0u64; 25];
    for x in 0..5 {
        for y in 0..5 {
            let nx = y;
            let ny = (2 * x + 3 * y) % 5;
            b[nx + 5 * ny] = state[x + 5 * y].rotate_left(ROTATION_OFFSETS[x][y]);
        }
    }
    state.copy_from_slice(&b);
}

/// The `chi` step of the Keccak-f permutation introduces non-linearity (relationships between input and output).
fn execute_chi(state: &mut [u64; 25]) {
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
fn execute_iota(state: &mut [u64; 25], round_constant: u64) {
    state[0] ^= round_constant; // Inject round constant.
}

#[cfg(test)]
mod tests {
    use super::*;
    use hex_literal::hex;

    #[test]
    fn test_execute_keccak256() {
        let test_vectors: &[(&[u8], [u8; 32])] = &[
            (
                b"",
                hex!("c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470"),
            ),
            (
                b"abc",
                hex!("4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45"),
            ),
        ];

        for (input, expected_hash) in test_vectors {
            let hash = execute_keccak256(input);
            assert_eq!(&hash, expected_hash);
        }
    }
}
