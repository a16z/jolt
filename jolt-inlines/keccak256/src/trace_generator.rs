//! This file contains Keccak256-specific logic to be used in the Keccak256 inline:
//! 1) Prover: Keccak256SequenceBuilder expands the inline to a list of RV instructions.
//! 2) Host: Rust reference implementation to be called by jolt-sdk.
//!
//! Keccak is a hash function that uses a sponge construction. The spone absorbs (and permutes) data. Each permutation has 24 rounds. Then squeezes out the hash.
//! Glossary:
//!   - “Lane”  = one 64-bit word in the 5×5 state matrix (25 lanes total for Keccak256).
//!   - “Round” = single application of θ ρ π χ ι to the state.
//!   - “Rate”  = 1088 bits (136 B) that interact with the message/output.
//!   - “Capacity” = 512 bits hidden from the attacker (1600 − 1088).
//!   - “Permutation” = Keccak-f[1600] : 24 rounds, each θ→ρ→π→χ→ι.
//!
//! Keccak256 refers to the specific variant where the rate is 1088 bits and the capacity is 512 bits.
//! Keccak256 differs from SHA3-256 (not implemented here) in the padding scheme.

use tracer::emulator::cpu::Xlen;
use tracer::instruction::andn::ANDN;
use tracer::instruction::ld::LD;
use tracer::instruction::sd::SD;
use tracer::instruction::RV32IMInstruction;
use tracer::utils::inline_helpers::{
    InstrAssembler,
    Value::{Imm, Reg},
};
use tracer::utils::virtual_registers::allocate_virtual_register;

use crate::{Keccak256State, NUM_LANES};

/// The 24 round constants for the Keccak-f[1600] permutation.
/// These values are XORed into the state during the `iota` step of each round.
#[rustfmt::skip]
pub(crate) const ROUND_CONSTANTS: [u64; 24] = [
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

// Numb
/// Layout of the 96 virtual registers (`vr`).
///
/// Jolt requires the total number of registers (physical + virtual) to be a power of two.
/// With 32 physical registers, we need 96 virtual registers to reach 128.
///
/// While only 67 registers are actively used by this builder, we allocate 96
/// to satisfy the system requirement.
///
/// For NUM_LANES = 25, the layout is:
/// - `vr[0..24]`: The 25 lanes of the Keccak state array `A`.
/// - `vr[25..49]`: A temporary state array `B` used in `rho_and_pi`.
/// - `vr[50..54]`: The 5 lanes of the `C` array (column parities) in `theta`.
/// - `vr[55..59]`: The 5 lanes of the `D` array (theta effect) in `theta`.
/// - `vr[60..64]`: A 5-lane temporary buffer for the current row in `chi`.
/// - `vr[65..66]`: General-purpose scratch registers for intermediate values.
/// - `vr[67..95]`: Unused, allocated for padding to meet the power-of-two requirement.
pub(crate) const NEEDED_REGISTERS: usize = 96;
struct Keccak256SequenceBuilder {
    asm: InstrAssembler,
    round: u32,
    vr: [u8; NEEDED_REGISTERS],
    operand_rs1: u8,
    _operand_rs2: u8,
}

/// `Keccak256SequenceBuilder` is a helper struct for constructing the virtual instruction
/// sequence required to emulate the Keccak-256 hashing operation within the RISC-V
/// instruction set. This builder is responsible for generating the correct sequence of
/// `RV32IMInstruction` instances that together perform the Keccak-256 permutation and
/// hashing steps, using a set of virtual registers to hold intermediate state.
///
/// # Fields
/// - `address`: The starting program counter address for the sequence.
/// - `asm`: Builder for the vector of generated instructions representing the Keccak-256 operation.
/// - `round`: The current round of the Keccak permutation (0..24).
/// - `vr`: An array of virtual register indices used for state and temporary values.
/// - `operand_rs1`: The source register index for the first operand (input state pointer).
/// - `operand_rs2`: Unused.
///
/// # Usage
/// Typically, you construct a `Keccak256SequenceBuilder` with the required register mapping
/// and operands, then call `.build()` to obtain the full instruction sequence for the
/// Keccak-256 operation. This is used to inline the Keccak-256 hash logic into the
/// RISC-V instruction stream for tracing or emulation purposes.
///
/// # Note
/// The actual Keccak-256 logic is implemented in the `build` method, which generates
/// the appropriate instruction sequence. This struct is not intended for direct execution,
/// but rather for constructing instruction traces or emulation flows.
impl Keccak256SequenceBuilder {
    fn new(
        address: u64,
        is_compressed: bool,
        xlen: Xlen,
        vr: [u8; NEEDED_REGISTERS],
        operand_rs1: u8,
        operand_rs2: u8,
    ) -> Self {
        Keccak256SequenceBuilder {
            asm: InstrAssembler::new(address, is_compressed, xlen),
            round: 0,
            vr,
            operand_rs1,
            _operand_rs2: operand_rs2,
        }
    }

    fn build(mut self) -> Vec<RV32IMInstruction> {
        // 1. Load NUM_LANES lanes (64-bit words) of state from memory into registers.
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

        // 4. Finalize assembler and return instruction sequence.
        self.asm.finalize()
    }

    #[cfg(test)]
    /// Build sequence up to a specific round and step for testing
    fn build_up_to_step(mut self, target_round: u32, target_step: &str) -> Vec<RV32IMInstruction> {
        // Always start by loading state
        self.load_state();

        // Execute rounds up to target
        for round in 0..=target_round {
            self.round = round;

            // Execute steps within the round
            self.theta();
            if round == target_round && target_step == "theta" {
                break;
            }

            self.rho_and_pi();
            if round == target_round && target_step == "rho_and_pi" {
                break;
            }

            self.chi();
            if round == target_round && target_step == "chi" {
                break;
            }

            self.iota();
            if round == target_round && target_step == "iota" {
                break;
            }
        }

        self.asm.finalize()
    }

    /// Load the initial Keccak state from memory into virtual registers.
    /// Keccak state is NUM_LANES lanes of 64 bits each (200 bytes total).
    fn load_state(&mut self) {
        (0..NUM_LANES).for_each(|i| {
            self.asm
                .emit_ld::<LD>(self.vr[i], self.operand_rs1, (i * 8) as i64)
        });
    }

    /// Store the final Keccak state from virtual registers back to memory.
    fn store_state(&mut self) {
        (0..NUM_LANES).for_each(|i| {
            self.asm
                .emit_s::<SD>(self.operand_rs1, self.vr[i], (i * 8) as i64)
        });
    }

    /// Get the register index for a given lane in the state matrix.
    fn lane(&self, x: usize, y: usize) -> u8 {
        self.vr[5 * y + x]
    }

    // --- Keccak-f Round Functions ---

    fn theta(&mut self) {
        // --- C[x] = A[x,0] ^ A[x,1] ^ A[x,2] ^ A[x,3] ^ A[x,4] ---
        for x in 0..5 {
            let c_reg = self.vr[50 + x];
            // c_reg = A[x,0] ^ A[x,1]
            self.asm
                .xor(Reg(self.lane(x, 0)), Reg(self.lane(x, 1)), c_reg);
            // c_reg ^= A[x,2] ^ A[x,3] ^ A[x,4]
            for y in 2..5 {
                self.asm.xor(Reg(c_reg), Reg(self.lane(x, y)), c_reg);
            }
        }

        // --- D[x] = C[x-1] ^ rotl(C[x+1], 1) ---
        for x in 0..5 {
            let d_reg = self.vr[55 + x];
            let c_prev = self.vr[50 + (x + 4) % 5];
            let c_next = self.vr[50 + (x + 1) % 5];
            let temp_rot_reg = self.vr[65]; // Use a scratch register for the rotation result

            self.asm.rotl64(Reg(c_next), 1, temp_rot_reg);
            self.asm.xor(Reg(c_prev), Reg(temp_rot_reg), d_reg);
        }

        // --- A[x,y] ^= D[x] ---
        for x in 0..5 {
            let d_reg = self.vr[55 + x];
            for y in 0..5 {
                let a_reg = self.lane(x, y);
                self.asm.xor(Reg(a_reg), Reg(d_reg), a_reg);
            }
        }
    }

    fn rho_and_pi(&mut self) {
        // This function combines two steps:
        // 1. Rho (ρ): Rotates each lane A[x,y] by a fixed offset.
        // 2. Pi (π): Permutes the lanes into a new configuration.
        //
        // The combined operation is: B[y, 2x+3y] = ROTL(A[x,y], offset)
        // We use vr[NUM_LANES..NUM_LANES*2-1] as the temporary state B.

        // --- 1. Rotate each lane and store in the permuted position in B ---
        #[allow(clippy::needless_range_loop)] // This is clearer than enumerating
        for x in 0..5 {
            for y in 0..5 {
                // Get the source lane A[x,y] and its rotation offset.
                let source_reg = self.lane(x, y);
                // We have checked that this is [x][y].
                let rotation_offset = ROTATION_OFFSETS[x][y];

                // Calculate the permuted destination coordinates in B.
                let nx = y;
                let ny = (2 * x + 3 * y) % 5;
                let dest_reg_in_b = self.vr[NUM_LANES + (5 * ny + nx)];

                // Rotate A[x,y] and store the result in B[nx, ny].
                self.asm
                    .rotl64(Reg(source_reg), rotation_offset, dest_reg_in_b);
            }
        }
    }

    fn chi(&mut self) {
        // The chi step provides non-linearity. For each row, it updates each lane as:
        // A[x,y] ^= (~A[x+1,y] & A[x+2,y])
        for y in 0..5 {
            for x in 0..5 {
                // Get the registers for the three input values
                // A[x,y], A[x+1,y], A[x+2,y]
                let current = NUM_LANES as u8 + self.lane(x, y);
                let next = NUM_LANES as u8 + self.lane((x + 1) % 5, y);
                let two_next = NUM_LANES as u8 + self.lane((x + 2) % 5, y);

                // Define scratch registers for intermediate results.
                let not_next_and_two_next = self.vr[65]; // reuse scratch

                // Get the register for the lane we are updating in the main state A.
                let dest_a_reg = self.lane(x, y);

                // Implement A[x,y] ^= (~A[x+1,y] & A[x+2,y])
                // 1. not_next_and_two_next = A[x+2,y] & ~A[x+1,y] using ANDN
                self.asm
                    .emit_r::<ANDN>(not_next_and_two_next, two_next, next);
                // 2. A[x,y] ^= not_next_and_two_next
                self.asm
                    .xor(Reg(current), Reg(not_next_and_two_next), dest_a_reg);
            }
        }
    }

    fn iota(&mut self) {
        // The iota step breaks symmetry by XORing a round-specific constant
        // into the first lane of the state, A[0,0].
        let round_constant = ROUND_CONSTANTS[self.round as usize];
        let first_lane_reg = self.lane(0, 0);
        self.asm
            .xor(Reg(first_lane_reg), Imm(round_constant), first_lane_reg);
    }
}

#[cfg(all(test, feature = "host"))]
#[allow(clippy::too_many_arguments)]
pub(crate) fn keccak256_build_up_to_step(
    address: u64,
    is_compressed: bool,
    xlen: Xlen,
    vr: [u8; NEEDED_REGISTERS],
    operand_rs1: u8,
    operand_rs2: u8,
    target_round: u32,
    target_step: &str,
) -> Vec<RV32IMInstruction> {
    let builder =
        Keccak256SequenceBuilder::new(address, is_compressed, xlen, vr, operand_rs1, operand_rs2);
    builder.build_up_to_step(target_round, target_step)
}

// Host-side Keccak-256 implementation for reference and testing.
#[cfg(all(test, feature = "host"))]
pub(crate) fn execute_keccak256(msg: &[u8]) -> [u8; 32] {
    // Keccak-256 parameters.
    const RATE_IN_BYTES: usize = 136; // 1088-bit rate

    // NUM_LANES × 64-bit state lanes initialised to zero.
    let mut state = [0u64; NUM_LANES];

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
            let nx = y;
            let ny = (2 * x + 3 * y) % 5;
            // Definitely [x][y] here. That behavior allows the test to pass.
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

pub fn keccak256_inline_sequence_builder(
    address: u64,
    is_compressed: bool,
    xlen: Xlen,
    operand_rs1: u8,
    operand_rs2: u8,
    _rd: u8,
) -> Vec<RV32IMInstruction> {
    // Virtual registers used as a scratch space
    let guards: Vec<_> = (0..NEEDED_REGISTERS)
        .map(|_| allocate_virtual_register())
        .collect();
    let mut vr = [0; NEEDED_REGISTERS];
    for (i, guard) in guards.iter().enumerate() {
        vr[i] = **guard;
    }
    let builder =
        Keccak256SequenceBuilder::new(address, is_compressed, xlen, vr, operand_rs1, operand_rs2);
    builder.build()
}
