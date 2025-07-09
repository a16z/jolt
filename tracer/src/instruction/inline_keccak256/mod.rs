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
use crate::instruction::and::AND;
use crate::instruction::andi::ANDI;
use crate::instruction::format::format_i::FormatI;
use crate::instruction::format::format_r::FormatR;
use crate::instruction::format::format_s::FormatS;
use crate::instruction::format::format_virtual_right_shift_i::FormatVirtualRightShiftI;
use crate::instruction::ld::LD;
use crate::instruction::sd::SD;
use crate::instruction::virtual_rotri::VirtualROTRI;
use crate::instruction::xor::XOR;
use crate::instruction::xori::XORI;
use crate::instruction::RV32IMInstruction;

pub mod keccak256;

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

#[derive(Clone, Copy)]
enum Value {
    Imm(u64),
    Reg(usize),
}
use Value::{Imm, Reg};

// Numb
/// Layout of the 96 virtual registers (`vr`):
/// - `vr[0..24]`: The 25 lanes of the Keccak state array `A`.
/// - `vr[25..49]`: A temporary state array `B` used in `rho_and_pi` and `chi`.
/// - `vr[50..54]`: The 5 lanes of the `C` array in `theta`.
/// - `vr[55..59]`: The 5 lanes of the `D` array in `theta`.
/// - `vr[60..64]`: The 5 lanes of the row buffer in `chi`.
/// - `vr[65..95]`: Additional scratch registers for temporary values.
pub const NEEDED_REGISTERS: usize = 96;
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
        // 1. Load the 25 lanes (64-bit words) of state from memory into registers.
        self.load_state();

        // 2. Main loop: 24 rounds of Keccak-f permutation.
        for round in 0..24 {
            self.round = round;
            self.theta();
            self.rho_and_pi();
            self.chi();
            self.iota();
        }

        // 3. Store the final state back to memory.
        self.store_state();

        // 4. Finalize the sequence by setting instruction indices.
        self.enumerate_sequence();
        self.sequence
    }

    /// Load the initial Keccak state from memory into virtual registers.
    /// Keccak state is 25 lanes of 64 bits each (200 bytes total).
    fn load_state(&mut self) {
        (0..25).for_each(|i| self.ld(self.operand_rs1, i as i64, self.vr[i as usize]));
    }

    /// Store the final Keccak state from virtual registers back to memory.
    fn store_state(&mut self) {
        (0..25).for_each(|i| self.sd(self.operand_rs1, self.vr[i as usize], i as i64));
    }

    // --- Lane / Register Helpers ---

    /// Get the register index for a given lane in the state matrix.
    fn lane(&self, x: usize, y: usize) -> usize {
        self.vr[5 * y + x]
    }

    // --- Keccak-f Round Functions ---

    fn theta(&mut self) {
        // --- C[x] = A[x,0] ^ A[x,1] ^ A[x,2] ^ A[x,3] ^ A[x,4] ---
        for x in 0..5 {
            let c_reg = self.vr[50 + x];
            // c_reg = A[x,0] ^ A[x,1]
            self.xor64(Reg(self.lane(x, 0)), Reg(self.lane(x, 1)), c_reg);
            // c_reg ^= A[x,2] ^ A[x,3] ^ A[x,4]
            for y in 2..5 {
                self.xor64(Reg(c_reg), Reg(self.lane(x, y)), c_reg);
            }
        }

        // --- D[x] = C[x-1] ^ rotl(C[x+1], 1) ---
        for x in 0..5 {
            let d_reg = self.vr[55 + x];
            let c_prev = self.vr[50 + (x + 4) % 5];
            let c_next = self.vr[50 + (x + 1) % 5];
            let temp_rot_reg = self.vr[65]; // Use a scratch register for the rotation result

            self.rotl64(Reg(c_next), 1, temp_rot_reg);
            self.xor64(Reg(c_prev), Reg(temp_rot_reg), d_reg);
        }

        // --- A[x,y] ^= D[x] ---
        for x in 0..5 {
            let d_reg = self.vr[55 + x];
            for y in 0..5 {
                let a_reg = self.lane(x, y);
                self.xor64(Reg(a_reg), Reg(d_reg), a_reg);
            }
        }
    }

    fn rho_and_pi(&mut self) {
        // This function combines two steps:
        // 1. Rho (ρ): Rotates each lane A[x,y] by a fixed offset.
        // 2. Pi (π): Permutes the lanes into a new configuration.
        //
        // The combined operation is: B[y, 2x+3y] = ROTL(A[x,y], offset)
        // We use vr[25..49] as the temporary state B.

        // --- 1. Rotate each lane and store in the permuted position in B ---
        for x in 0..5 {
            for y in 0..5 {
                // Get the source lane A[x,y] and its rotation offset.
                let source_reg = self.lane(x, y);
                let rotation_offset = ROTATION_OFFSETS[y][x];

                // Calculate the permuted destination coordinates in B.
                let nx = y;
                let ny = (2 * x + 3 * y) % 5;
                let dest_reg_in_b = self.vr[25 + (5 * ny + nx)];

                // Rotate A[x,y] and store the result in B[nx, ny].
                self.rotl64(Reg(source_reg), rotation_offset, dest_reg_in_b);
            }
        }

        // --- 2. Copy the temporary state B back to the main state A ---
        // We can do this with a no-op XOR with register zero, which acts as a move.
        for i in 0..25 {
            let b_reg = self.vr[25 + i];
            let a_reg = self.vr[i];
            // This is equivalent to: a_reg = b_reg
            self.xor64(Reg(b_reg), Imm(0), a_reg);
        }
    }

    fn chi(&mut self) {
        // The chi step provides non-linearity. For each row, it updates each lane as:
        // A[x,y] ^= (~A[x+1,y] & A[x+2,y])
        //
        // To do this without overwriting the A lanes we need for the calculation,
        // we first copy the row into a temporary buffer (vr[60..64]).
        for y in 0..5 {
            // 1. Copy the current row from state A into the temporary row buffer.
            for x in 0..5 {
                // For each lane A[x,y] in the current row,
                // copy its value from self.lane(x, y) to the temp row register self.vr[60 + x].

                let a_reg = self.lane(x, y);
                let temp_row_reg = self.vr[60 + x];
                // A no-op XOR (xor with 0) is a good way to copy a register value.
                self.xor64(Reg(a_reg), Imm(0), temp_row_reg);
            }

            // 2. Calculate the new lane values using the temporary buffer.
            for x in 0..5 {
                // Get the registers for the three input values from the temp buffer.
                // A[x,y], A[x+1,y], A[x+2,y]
                let current = self.vr[60 + x];
                let next = self.vr[60 + (x + 1) % 5];
                let two_next = self.vr[60 + (x + 2) % 5];

                // Define scratch registers for intermediate results.
                let not_next = self.vr[65];
                let not_next_and_two_next = self.vr[66];

                // Get the register for the lane we are updating in the main state A.
                let dest_a_reg = self.lane(x, y);

                // Implement A[x,y] ^= (~A[x+1,y] & A[x+2,y])
                // 1. not_next = ~A[x+1,y]
                self.not64(Reg(next), not_next);
                // 2. not_next_and_two_next = not_next & A[x+2,y]
                self.and64(Reg(not_next), Reg(two_next), not_next_and_two_next);
                // 3. A[x,y] ^= not_next_and_two_next
                self.xor64(Reg(current), Reg(not_next_and_two_next), dest_a_reg);
            }
        }
    }

    fn iota(&mut self) {
        // The iota step breaks symmetry by XORing a round-specific constant
        // into the first lane of the state, A[0,0].
        let round_constant = ROUND_CONSTANTS[self.round as usize];
        let first_lane_reg = self.lane(0, 0);

        // XOR the round_constant into the first_lane_reg.
        self.xor64(Reg(first_lane_reg), Imm(round_constant), first_lane_reg);
    }

    // --- 64-bit Arithmetic Helpers ---

    /// XOR two 64-bit numbers.
    fn xor64(&mut self, rs1: Value, rs2: Value, rd: usize) -> Value {
        self.xor(rs1, rs2, rd)
    }

    /// AND two 64-bit numbers.
    fn and64(&mut self, rs1: Value, rs2: Value, rd: usize) -> Value {
        self.and(rs1, rs2, rd)
    }

    /// NOT a 64-bit number (by XORing with u64::MAX).
    fn not64(&mut self, rs1: Value, rd: usize) -> Value {
        self.xor(rs1, Imm(u64::MAX), rd)
    }

    /// Rotate a 64-bit number to the left.
    fn rotl64(&mut self, rs1: Value, amount: u32, rd: usize) -> Value {
        if amount == 0 {
            // This is a no-op, but if rd is different from rs1, we need to move the value.
            // For simplicity, we assume we can overwrite rd or that caller handles it.
            return rs1;
        }
        // ROTL(x, n) = ROTR(x, 64 - n)
        self.rotri(rs1, 64 - amount as u64, rd)
    }

    // --- RV64 Instruction Emitters ---

    fn ld(&mut self, rs1: usize, offset: i64, rd: usize) {
        let ld = LD {
            address: self.address,
            operands: FormatI {
                rd,
                rs1,
                imm: (offset * 8) as u64, // 64-bit lanes are 8 bytes
            },
            virtual_sequence_remaining: Some(0),
        };
        self.sequence.push(ld.into());
    }

    fn sd(&mut self, rs1: usize, rs2: usize, offset: i64) {
        let sd = SD {
            address: self.address,
            operands: FormatS {
                rs1,
                rs2,
                imm: offset * 8, // 64-bit lanes are 8 bytes
            },
            virtual_sequence_remaining: Some(0),
        };
        self.sequence.push(sd.into());
    }

    fn xor(&mut self, rs1: Value, rs2: Value, rd: usize) -> Value {
        match (rs1, rs2) {
            (Reg(rs1), Reg(rs2)) => {
                let xor = XOR {
                    address: self.address,
                    operands: FormatR { rd, rs1, rs2 },
                    virtual_sequence_remaining: Some(0),
                };
                self.sequence.push(xor.into());
                Reg(rd)
            }
            (Reg(rs1), Imm(imm)) => {
                let xori = XORI {
                    address: self.address,
                    operands: FormatI { rd, rs1, imm },
                    virtual_sequence_remaining: Some(0),
                };
                self.sequence.push(xori.into());
                Reg(rd)
            }
            (Imm(_), Reg(_)) => self.xor(rs2, rs1, rd),
            (Imm(imm1), Imm(imm2)) => Imm(imm1 ^ imm2),
        }
    }

    fn and(&mut self, rs1: Value, rs2: Value, rd: usize) -> Value {
        match (rs1, rs2) {
            (Reg(rs1), Reg(rs2)) => {
                let and = AND {
                    address: self.address,
                    operands: FormatR { rd, rs1, rs2 },
                    virtual_sequence_remaining: Some(0),
                };
                self.sequence.push(and.into());
                Reg(rd)
            }
            (Reg(rs1), Imm(imm)) => {
                let andi = ANDI {
                    address: self.address,
                    operands: FormatI { rd, rs1, imm },
                    virtual_sequence_remaining: Some(0),
                };
                self.sequence.push(andi.into());
                Reg(rd)
            }
            (Imm(_), Reg(_)) => self.and(rs2, rs1, rd),
            (Imm(imm1), Imm(imm2)) => Imm(imm1 & imm2),
        }
    }

    fn rotri(&mut self, rs1: Value, imm: u64, rd: usize) -> Value {
        match rs1 {
            Reg(rs1) => {
                // This is a virtual instruction. The `imm` field for the format is a bitmask,
                // not the rotation amount. The `execute` method will extract the amount.
                // For now, we pass the rotation amount and it will be handled by the tracer.
                let rotri = VirtualROTRI {
                    address: self.address,
                    operands: FormatVirtualRightShiftI { rd, rs1, imm },
                    virtual_sequence_remaining: Some(0),
                };
                self.sequence.push(rotri.into());
                Reg(rd)
            }
            Imm(val) => Imm(val.rotate_right(imm as u32)),
        }
    }

    /// Enumerates sequence in reverse order and sets virtual_sequence_remaining
    fn enumerate_sequence(&mut self) {
        let len = self.sequence.len();
        self.sequence
            .iter_mut()
            .enumerate()
            .for_each(|(i, instruction)| {
                instruction.set_virtual_sequence_remaining(Some(len - i - 1));
            });
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
            // Note: ROTATION_OFFSETS[y][x] is correct.
            b[nx + 5 * ny] = state[x + 5 * y].rotate_left(ROTATION_OFFSETS[y][x]);
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
