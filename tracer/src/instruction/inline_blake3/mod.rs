use crate::instruction::format::format_i::FormatI;
use crate::instruction::format::format_load::FormatLoad;
use crate::instruction::format::format_r::FormatR;
use crate::instruction::format::format_s::FormatS;
use crate::instruction::format::format_u::FormatU;
use crate::instruction::format::format_virtual_right_shift_i::FormatVirtualRightShiftI;
use crate::instruction::lui::LUI;
use crate::instruction::lw::LW;
use crate::instruction::sw::SW;
use crate::instruction::virtual_rotril::VirtualROTRIL;
use crate::instruction::virtual_xor_rot::{
    VirtualROTXOR12L, VirtualROTXOR16L, VirtualROTXOR7L, VirtualROTXOR8L,
};
use crate::instruction::xor::XOR;
use crate::instruction::xori::XORI;
use crate::instruction::RV32IMInstruction;
/// This file contains Blake3-specific logic to be used in the Blake3 inline:
/// 1) Prover: Blake3SequenceBuilder expands the inline to a list of RV instructions.
/// 2) Host: Rust reference implementation to be called by jolt-sdk.
///
/// Blake3 is a cryptographic hash function that operates on 32-bit words in a compression function.
/// Glossary:
///   - "Word" = one 32-bit value.
///   - "Round" = single application of G function to all columns and diagonals.
///   - "Block" = 64 bytes (16 words) of input data.
///   - "Compression" = Blake3 compression function: 7 rounds of G function.
use crate::instruction::{add::ADD, addi::ADDI};

pub mod blake3;
pub mod blake3_hash_modes;

// Re-export the Blake3 variants for public use
pub use blake3_hash_modes::{BLAKE3_128, BLAKE3_192, BLAKE3_256, BLAKE3_64};

/// Blake3 initialization vector (IV)
#[rustfmt::skip]
const BLAKE3_IV: [u32; 8] = [
    0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a,
    0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19,
];

/// Blake3 message scheduling constants for each round
/// Each round uses a different permutation of the input words
#[rustfmt::skip]
const MSG_SCHEDULE: [[usize; 16]; 7] = [
    [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
    [2, 6, 3, 10, 7, 0, 4, 13, 1, 11, 12, 5, 9, 14, 15, 8],
    [3, 4, 10, 12, 13, 2, 7, 14, 6, 5, 9, 0, 11, 15, 8, 1],
    [10, 7, 12, 9, 14, 3, 13, 15, 4, 0, 11, 2, 5, 8, 1, 6],
    [12, 13, 9, 11, 15, 10, 14, 8, 7, 2, 5, 3, 0, 1, 6, 4],
    [9, 14, 11, 5, 8, 12, 15, 1, 13, 3, 0, 10, 2, 6, 4, 7],
    [11, 15, 5, 0, 1, 9, 8, 6, 14, 10, 2, 12, 3, 4, 7, 13],
];

#[derive(Clone, Copy)]
enum Value {
    Imm(u32),
    Reg(usize),
}
use Value::{Imm, Reg};

/// Number of virtual registers needed for Blake3 implementation.
///
/// Jolt requires the total number of registers (physical + virtual) to be a power of two.
/// With 32 physical registers, we need 96 virtual registers to reach 128.
pub const NEEDED_REGISTERS: usize = 96;

/// Memory layout constants for Blake3 virtual registers
///
/// The virtual register layout is organized as follows:
/// - vr[0..15]:  Working state `v` (16 words) - computed during compression
/// - vr[16..31]: Message block `m` (16 words) - input data
/// - vr[32..39]: Chaining value `h` (8 words) - input/output hash state
/// - vr[40..41]: Counter values (2 words) - block counter
/// - vr[42]:     Input bytes length (1 word)
/// - vr[43]:     Flags (1 word) - final block flag, etc.
/// - vr[44]:     Temporary register for intermediate calculations
const VR_STATE_START: usize = 0;
const STATE_SIZE: usize = 16;
const VR_MESSAGE_BLOCK_START: usize = 16;
pub const MESSAGE_BLOCK_SIZE: usize = 16;
const VR_CHAINING_VALUE_START: usize = 32;
pub const CHAINING_VALUE_SIZE: usize = 8;
const VR_COUNTER_START: usize = 40;
const COUNTER_SIZE: usize = 2;
const VR_INPUT_BYTES_LEN_START: usize = 42;
const VR_FLAG_START: usize = 43;
const VR_TEMP: usize = 44;

pub enum BuilderMode {
    COMPRESSION,
    HASH(u32),
}

pub enum RotationAmount {
    ROT16,
    ROT12,
    ROT8,
    ROT7,
}

pub struct Blake3SequenceBuilder {
    address: u64,
    sequence: Vec<RV32IMInstruction>,
    round: u32,
    vr: [usize; NEEDED_REGISTERS],
    operand_rs1: usize,
    operand_rs2: usize,
    mode: BuilderMode,
}

/// `Blake3SequenceBuilder` is a helper struct for constructing the virtual instruction
/// sequence required to emulate the Blake3 hashing operation within the RISC-V
/// instruction set. This builder is responsible for generating the correct sequence of
/// `RV32IMInstruction` instances that together perform the Blake3 compression
/// function, using a set of virtual registers to hold intermediate state.
///
/// # Fields
/// - `address`: The starting program counter address for the sequence.
/// - `sequence`: The vector of generated instructions representing the Blake3 operation.
/// - `round`: The current round of the Blake3 compression (0..7).
/// - `vr`: An array of virtual register indices used for state and temporary values.
/// - `operand_rs1`: The source register index for the first operand (state pointer).
/// - `operand_rs2`: The source register index for the second operand (message block pointer).

impl Blake3SequenceBuilder {
    pub fn new(
        address: u64,
        vr: [usize; NEEDED_REGISTERS],
        operand_rs1: usize,
        operand_rs2: usize,
        mode: BuilderMode,
    ) -> Self {
        Blake3SequenceBuilder {
            address,
            sequence: vec![],
            round: 0,
            vr,
            operand_rs1,
            operand_rs2,
            mode,
        }
    }

    pub fn build(mut self) -> Vec<RV32IMInstruction> {
        match self.mode {
            BuilderMode::COMPRESSION => {
                return self.compress_build();
            }
            BuilderMode::HASH(block_num) => {
                // v[8..11] = IV[0..3] (first 4 words of initialization vector)
                for i in 0..8 {
                    self.load_32bit_immediate(BLAKE3_IV[i], self.vr[VR_CHAINING_VALUE_START + i]);
                }
                for i in 0..block_num {
                    // Load new message chunk
                    self.load_data_range(
                        self.operand_rs2,
                        MESSAGE_BLOCK_SIZE * i as usize,
                        VR_MESSAGE_BLOCK_START,
                        MESSAGE_BLOCK_SIZE,
                    );
                    // 2^0 for chunk_start -- 2^1 for chunk_end
                    let flag = (i == 0) as u32
                        | (((i == block_num - 1) as u32) << 1)
                        | (((i == block_num - 1) as u32) << 3);
                    self.initialize_working_state(flag);
                    // 3. Main loop: 7 rounds of Blake3 compression
                    for round in 0..7 {
                        self.round = round;
                        self.blake3_round();
                    }

                    // 4. Finalize the hash state: h[i] ^= v[i] ^ v[i+8]
                    let requires_truncate = i != block_num - 1;
                    self.finalize_state(requires_truncate);
                }
                self.store_state();
                // 6. Finalize the sequence by setting instruction indices
                self.enumerate_sequence();
                self.sequence
            }
        }
    }

    pub fn compress_build(mut self) -> Vec<RV32IMInstruction> {
        // Load the current hash state (8 words) from memory into virtual registers
        self.load_data_range(
            self.operand_rs1,
            0,
            VR_CHAINING_VALUE_START,
            CHAINING_VALUE_SIZE,
        );
        // Load the message block (16 words) from memory into virtual registers
        self.load_data_range(
            self.operand_rs2,
            0,
            VR_MESSAGE_BLOCK_START,
            MESSAGE_BLOCK_SIZE,
        );
        // Load the counter value from memory - stored after the message block
        self.load_data_range(
            self.operand_rs2,
            MESSAGE_BLOCK_SIZE,
            VR_COUNTER_START,
            COUNTER_SIZE,
        );
        // Load the input bytes length from memory - stored after the counter
        self.lw(
            self.operand_rs2,
            (MESSAGE_BLOCK_SIZE + COUNTER_SIZE) as i64,
            self.vr[VR_INPUT_BYTES_LEN_START],
        );
        // Load the final block flag from memory - stored after the input bytes length
        self.lw(
            self.operand_rs2,
            (MESSAGE_BLOCK_SIZE + COUNTER_SIZE + 1) as i64,
            self.vr[VR_FLAG_START],
        );

        // 2. Initialize the working state v[0..15]
        self.initialize_working_state(0);

        // 3. Main loop: 7 rounds of Blake3 compression
        for round in 0..7 {
            self.round = round;
            self.blake3_round();
        }

        // 4. Finalize the hash state: h[i] ^= v[i] ^ v[i+8]
        self.finalize_state(false);

        // 5. Store the final hash state back to memory
        self.store_state();

        // 6. Finalize the sequence by setting instruction indices
        self.enumerate_sequence();
        self.sequence
    }

    /// Load data from memory into virtual registers starting at a given offset
    fn load_data_range(
        &mut self,
        base_register: usize,
        memory_offset_start: usize,
        vr_start: usize,
        count: usize,
    ) {
        (0..count).for_each(|i| {
            self.lw(
                base_register,
                (memory_offset_start + i) as i64,
                self.vr[vr_start + i],
            )
        });
    }

    /// Initialize the working state v[0..15] according to Blake3 specification:
    fn initialize_working_state(&mut self, flag: u32) {
        // v[0..7] = h[0..7] (current hash state)
        for i in 0..CHAINING_VALUE_SIZE {
            self.xor32(
                Reg(self.vr[VR_CHAINING_VALUE_START + i]),
                Imm(0),
                self.vr[VR_STATE_START + i],
            );
        }

        // v[8..11] = IV[0..3] (first 4 words of initialization vector)
        for i in 0..4 {
            self.load_32bit_immediate(BLAKE3_IV[i], self.vr[CHAINING_VALUE_SIZE + i]);
        }

        match self.mode {
            BuilderMode::COMPRESSION => {
                // v[12..15] = counter values, input length, and flags
                self.xor32(
                    Reg(self.vr[VR_COUNTER_START]),
                    Imm(0),
                    self.vr[VR_STATE_START + 12],
                );
                self.xor32(
                    Reg(self.vr[VR_COUNTER_START + 1]),
                    Imm(0),
                    self.vr[VR_STATE_START + 13],
                );
                self.xor32(
                    Reg(self.vr[VR_INPUT_BYTES_LEN_START]),
                    Imm(0),
                    self.vr[VR_STATE_START + 14],
                );
                self.xor32(
                    Reg(self.vr[VR_FLAG_START]),
                    Imm(0),
                    self.vr[VR_STATE_START + 15],
                );
            }
            BuilderMode::HASH(_) => {
                for i in 12..14 {
                    self.load_32bit_immediate(0, self.vr[VR_STATE_START + i]);
                }
                self.load_32bit_immediate(64, self.vr[VR_STATE_START + 14]);
                self.load_32bit_immediate(flag, self.vr[VR_STATE_START + 15]);
            }
        }
    }

    /// Execute one round of Blake3 compression
    fn blake3_round(&mut self) {
        let msg_schedule_round = &MSG_SCHEDULE[self.round as usize];

        // Column step: apply G function to columns
        self.g_function(0, 4, 8, 12, msg_schedule_round[0], msg_schedule_round[1]);
        self.g_function(1, 5, 9, 13, msg_schedule_round[2], msg_schedule_round[3]);
        self.g_function(2, 6, 10, 14, msg_schedule_round[4], msg_schedule_round[5]);
        self.g_function(3, 7, 11, 15, msg_schedule_round[6], msg_schedule_round[7]);

        // Diagonal step: apply G function to diagonals
        self.g_function(0, 5, 10, 15, msg_schedule_round[8], msg_schedule_round[9]);
        self.g_function(1, 6, 11, 12, msg_schedule_round[10], msg_schedule_round[11]);
        self.g_function(2, 7, 8, 13, msg_schedule_round[12], msg_schedule_round[13]);
        self.g_function(3, 4, 9, 14, msg_schedule_round[14], msg_schedule_round[15]);
    }

    /// Blake3 G function: core mixing function
    /// Updates v[a], v[b], v[c], v[d] using message words m[x], m[y]
    fn g_function(&mut self, a: usize, b: usize, c: usize, d: usize, x: usize, y: usize) {
        let va = self.vr[a];
        let vb = self.vr[b];
        let vc = self.vr[c];
        let vd = self.vr[d];
        let mx = self.vr[VR_MESSAGE_BLOCK_START + x];
        let my = self.vr[VR_MESSAGE_BLOCK_START + y];
        let temp1 = self.vr[VR_TEMP];

        // v[a] = v[a] + v[b] + m[x]
        self.add32(Reg(va), Reg(vb), temp1);
        self.add32(Reg(temp1), Reg(mx), va);

        // // v[d] = rotr32(v[d] ^ v[a], 16)
        self.xor_rotate(vd, va, RotationAmount::ROT16, vd);

        // v[c] = v[c] + v[d]
        self.add32(Reg(vc), Reg(vd), vc);

        // v[b] = rotr32(v[b] ^ v[c], 12)
        self.xor_rotate(vb, vc, RotationAmount::ROT12, vb);

        // v[a] = v[a] + v[b] + m[y]
        self.add32(Reg(va), Reg(vb), temp1);
        self.add32(Reg(temp1), Reg(my), va);

        // v[d] = rotr32(v[d] ^ v[a], 8)
        self.xor_rotate(vd, va, RotationAmount::ROT8, vd);

        // v[c] = v[c] + v[d]
        self.add32(Reg(vc), Reg(vd), vc);

        // v[b] = rotr32(v[b] ^ v[c], 7)
        self.xor_rotate(vb, vc, RotationAmount::ROT7, vb);
    }

    fn finalize_state(&mut self, is_truncated: bool) {
        if is_truncated {
            for i in 0..STATE_SIZE / 2 {
                let hi = self.vr[VR_CHAINING_VALUE_START + i];
                let vi = self.vr[VR_STATE_START + i];
                let vi8 = self.vr[VR_STATE_START + i + 8];
                self.xor32(Reg(vi), Reg(vi8), hi);
            }
        } else {
            for i in 0..STATE_SIZE / 2 {
                let hi = self.vr[VR_STATE_START + i];
                let vi = self.vr[VR_STATE_START + i];
                let vi8 = self.vr[VR_STATE_START + i + 8];
                self.xor32(Reg(vi), Reg(vi8), hi);
            }
            for i in 0..STATE_SIZE / 2 {
                let hi_prime = self.vr[VR_STATE_START + i + 8];
                let vi = self.vr[VR_STATE_START + i + 8];
                let hi = self.vr[VR_CHAINING_VALUE_START + i];
                self.xor32(Reg(vi), Reg(hi), hi_prime);
            }
        }
    }

    /// Store the final hash state (chaining value) back to memory
    /// Blake3 compression outputs an 8-word chaining value
    fn store_state(&mut self) {
        for i in 0..STATE_SIZE {
            self.sw(self.operand_rs1, self.vr[VR_STATE_START + i], i as i64);
        }
    }

    fn add32(&mut self, rs1: Value, rs2: Value, rd: usize) -> Value {
        match (rs1, rs2) {
            (Reg(rs1), Reg(rs2)) => {
                let add = ADD {
                    address: self.address,
                    operands: FormatR { rd, rs1, rs2 },
                    virtual_sequence_remaining: Some(0),
                };
                self.sequence.push(add.into());
                Reg(rd)
            }
            (Reg(rs1), Imm(imm)) => {
                let addi = ADDI {
                    address: self.address,
                    operands: FormatI {
                        rd,
                        rs1,
                        imm: imm as u64,
                    },
                    virtual_sequence_remaining: Some(0),
                };
                self.sequence.push(addi.into());
                Reg(rd)
            }
            (Imm(_), Reg(_)) => self.add32(rs2, rs1, rd),
            (Imm(imm1), Imm(imm2)) => Imm((imm1).wrapping_add(imm2)),
        }
    }

    fn xor_rotate(&mut self, rs1: usize, rs2: usize, amount: RotationAmount, rd: usize) -> Value {
        match amount {
            RotationAmount::ROT16 => {
                let xor = VirtualROTXOR16L {
                    address: self.address,
                    operands: FormatR { rd, rs1, rs2 },
                    virtual_sequence_remaining: Some(0),
                };
                self.sequence.push(xor.into());
                Reg(rd)
            }
            RotationAmount::ROT12 => {
                let xor = VirtualROTXOR12L {
                    address: self.address,
                    operands: FormatR { rd, rs1, rs2 },
                    virtual_sequence_remaining: Some(0),
                };
                self.sequence.push(xor.into());
                Reg(rd)
            }
            RotationAmount::ROT8 => {
                let xor = VirtualROTXOR8L {
                    address: self.address,
                    operands: FormatR { rd, rs1, rs2 },
                    virtual_sequence_remaining: Some(0),
                };
                self.sequence.push(xor.into());
                Reg(rd)
            }
            RotationAmount::ROT7 => {
                let xor = VirtualROTXOR7L {
                    address: self.address,
                    operands: FormatR { rd, rs1, rs2 },
                    virtual_sequence_remaining: Some(0),
                };
                self.sequence.push(xor.into());
                Reg(rd)
            }
        }
    }

    fn xor32(&mut self, rs1: Value, rs2: Value, rd: usize) -> Value {
        self.xor(rs1, rs2, rd)
    }

    /// Right rotate a 32-bit number
    fn rotr32(&mut self, rs1: Value, amount: u32, rd: usize) -> Value {
        if amount == 0 {
            return self.xor(rs1, Imm(0), rd);
        }

        match rs1 {
            Reg(rs1_reg) => {
                // Construct bitmask: (32-imm) ones followed by imm zeros
                let ones = (1u64 << (32 - amount)) - 1;
                let imm = ones << amount;

                let rotri = VirtualROTRIL {
                    address: self.address,
                    operands: FormatVirtualRightShiftI {
                        rd,
                        rs1: rs1_reg,
                        imm,
                    },
                    virtual_sequence_remaining: Some(0),
                };
                self.sequence.push(rotri.into());
                Reg(rd)
            }
            Imm(val) => Imm(val.rotate_right(amount)),
        }
    }

    fn load_32bit_immediate(&mut self, value: u32, rd: usize) {
        let lui = LUI {
            address: self.address,
            operands: FormatU {
                rd,
                imm: value as i32 as u64,
            },
            virtual_sequence_remaining: Some(0),
        };
        self.sequence.push(lui.into());
    }

    // --- RV32 Instruction Emitters ---

    fn lw(&mut self, rs1: usize, offset: i64, rd: usize) {
        let lw = LW {
            address: self.address,
            operands: FormatLoad {
                rd,
                rs1,
                imm: offset * 4,
            },
            virtual_sequence_remaining: Some(0),
        };
        self.sequence.push(lw.into());
    }

    fn sw(&mut self, rs1: usize, rs2: usize, offset: i64) {
        let sw = SW {
            address: self.address,
            operands: FormatS {
                rs1,
                rs2,
                imm: offset * 4,
            },
            virtual_sequence_remaining: Some(0),
        };
        self.sequence.push(sw.into());
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
                    operands: FormatI {
                        rd,
                        rs1,
                        imm: imm as u64,
                    },
                    virtual_sequence_remaining: Some(0),
                };
                self.sequence.push(xori.into());
                Reg(rd)
            }
            (Imm(_), Reg(_)) => self.xor(rs2, rs1, rd),
            (Imm(imm1), Imm(imm2)) => Imm(imm1 ^ imm2),
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
        BLAKE3_IV[0],      BLAKE3_IV[1],      BLAKE3_IV[2],      BLAKE3_IV[3],
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
