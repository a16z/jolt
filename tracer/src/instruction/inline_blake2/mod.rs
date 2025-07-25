/// Blake2b inline implementation for Jolt zkVM.
/// 1) Prover: Blake2SequenceBuilder expands the inline to RISC-V-64 instructions.
/// 2) Host: Rust reference implementation called by jolt-sdk.
///
/// Blake2b is a cryptographic hash function operating on 64-bit words.


use crate::instruction::format::format_i::FormatI;
use crate::instruction::format::format_r::FormatR;
use crate::instruction::format::format_s::FormatS;
use crate::instruction::format::format_u::FormatU;
use crate::instruction::ld::LD;
use crate::instruction::lui::LUI;
use crate::instruction::sd::SD;
use crate::instruction::virtual_xor_rot::{VirtualROTXOR16, VirtualROTXOR24, VirtualROTXOR32, VirtualROTXOR63};
use crate::instruction::xor::XOR;
use crate::instruction::RV32IMInstruction;

use crate::instruction::{add::ADD, sub::SUB};

pub mod blake2;

/// Blake2b initialization vector (IV)
#[rustfmt::skip]
const BLAKE2B_IV: [u64; 8] = [
    0x6a09e667f3bcc908, 0xbb67ae8584caa73b,
    0x3c6ef372fe94f82b, 0xa54ff53a5f1d36f1,
    0x510e527fade682d1, 0x9b05688c2b3e6c1f,
    0x1f83d9abfb41bd6b, 0x5be0cd19137e2179,
];

/// Blake2b sigma permutation constants for message scheduling
/// Each round uses a different permutation of the input words
#[rustfmt::skip]
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

#[allow(dead_code)]
#[derive(Clone, Copy)]
enum Value {
    Imm(u64),
    Reg(usize),
}
#[allow(unused_imports)]
use Value::{Imm, Reg};

/// Number of virtual registers needed for Blake2 implementation.
///
/// Jolt requires the total number of registers (physical + virtual) to be a power of two.
/// With 32 physical registers, we need 96 virtual registers to reach 128.
pub const NEEDED_REGISTERS: usize = 96;

/// Memory layout constants for Blake2 virtual registers
///
/// Virtual register layout:
/// - vr[0..15]:  Working state `v` (16 words)
/// - vr[16..31]: Message block `m` (16 words)
/// - vr[32..39]: Hash state `h` (8 words)
/// - vr[40]:     Counter value `t`
/// - vr[41]:     Final block flag
/// - vr[42]:     Temporary register
/// - vr[43]:     Zero constant
const VR_WORKING_STATE_START: usize = 0;
const WORKING_STATE_SIZE: usize = 16;
const VR_MESSAGE_BLOCK_START: usize = 16;
pub const MESSAGE_BLOCK_SIZE: usize = 16;
const VR_HASH_STATE_START: usize = 32;
pub const HASH_STATE_SIZE: usize = 8;
const VR_T: usize = 40;
const VR_IS_FINAL: usize = 41;
const VR_TEMP: usize = 42;
const VR_ZERO: usize = 43;

// Rotation constants required in Blake2b
pub enum RotationAmount {
    ROT32,
    ROT24,
    ROT16,
    ROT63,
}

struct Blake2SequenceBuilder {
    address: u64,
    sequence: Vec<RV32IMInstruction>,
    round: u32,
    vr: [usize; NEEDED_REGISTERS],
    operand_rs1: usize,
    operand_rs2: usize,
}

/// `Blake2SequenceBuilder` generates RISC-V instruction sequences for Blake2b compression.
/// Uses virtual registers to hold intermediate state during the compression function.
///
/// # Fields
/// - `address`: Starting program counter address
/// - `sequence`: Generated instructions for Blake2 operation
/// - `round`: Current compression round (0..12)
/// - `vr`: Virtual register indices for state and temporary values
/// - `operand_rs1`: Source register for state pointer
/// - `operand_rs2`: Source register for message block pointer

impl Blake2SequenceBuilder {
    fn new(
        address: u64,
        vr: [usize; NEEDED_REGISTERS],
        operand_rs1: usize,
        operand_rs2: usize,
    ) -> Self {
        Blake2SequenceBuilder {
            address,
            sequence: vec![],
            round: 0,
            vr,
            operand_rs1,
            operand_rs2,
        }
    }

    fn build(mut self) -> Vec<RV32IMInstruction> {
        // Load inputs
        self.load_hash_state();
        self.load_message_blocks();
        self.load_counter_and_is_final_and_zero();

        // Initialize the working state v[0..15]
        self.initialize_working_state();

        // Cryptographic mixing for 12 rounds
        for round in 0..12 {
            self.round = round;
            self.blake2_round();
        }

        // Finalize the hash state
        self.finalize_state();

        // Store the final hash state back to memory
        self.store_state();

        self.enumerate_sequence();
        self.sequence
    }

    fn load_hash_state(&mut self) {
        self.load_data_range(self.operand_rs1, 0, VR_HASH_STATE_START, HASH_STATE_SIZE);
    }

    fn load_message_blocks(&mut self) {
        self.load_data_range(
            self.operand_rs2,
            0,
            VR_MESSAGE_BLOCK_START,
            MESSAGE_BLOCK_SIZE,
        );
    }

    fn load_counter_and_is_final_and_zero(&mut self) {
        self.ld(self.operand_rs2, MESSAGE_BLOCK_SIZE as i64, self.vr[VR_T]);
        self.ld(
            self.operand_rs2,
            MESSAGE_BLOCK_SIZE as i64 + 1,
            self.vr[VR_IS_FINAL],
        );
        self.load_64bit_immediate(
            0,
            self.vr[VR_ZERO],
        );
    }

    // Initialize the working state v[0..15] according to Blake2b specification:
    fn initialize_working_state(&mut self) {
        // v[0..7] = h[0..7] (current hash state)
        for i in 0..HASH_STATE_SIZE {
            self.xor64(
                Reg(self.vr[VR_HASH_STATE_START + i]),
                Reg(self.vr[VR_ZERO]),
                self.vr[VR_WORKING_STATE_START + i],
            );
        }

        // v[8..15] = IV[0..7] (initialization vector)
        for i in 0..WORKING_STATE_SIZE - HASH_STATE_SIZE {
            self.load_64bit_immediate(
                BLAKE2B_IV[i],
                self.vr[VR_WORKING_STATE_START + HASH_STATE_SIZE + i],
            );
        }

        // v[12] = v[12] ^ t (counter low)
        self.xor64(
            Reg(self.vr[VR_WORKING_STATE_START + 12]),
            Reg(self.vr[VR_T]),
            self.vr[VR_WORKING_STATE_START + 12],
        );

        // v[13] = IV[5] ^ (t >> 64) (counter high) - for 64-bit counter, high part is 0
        // Since we are using 64-bit counter, the high part is always 0, so v[13] remains unchanged

        // Handle final block flag: if is_final != 0, invert all bits of v[14]
        // We need to create a mask that is 0xFFFFFFFFFFFFFFFF if is_final != 0, or 0 if is_final == 0
        // Use the formula: mask = (0 - is_final) to convert 1 to 0xFFFFFFFFFFFFFFFF and 0 to 0
        let temp_mask = self.vr[VR_TEMP];
        // First, negate is_final (0 - is_final)
        self.negate64(Reg(self.vr[VR_IS_FINAL]), temp_mask);
        // XOR v[14] with the mask: inverts all bits if is_final=1, leaves unchanged if is_final=0
        self.xor64(
            Reg(self.vr[VR_WORKING_STATE_START + 14]),
            Reg(temp_mask),
            self.vr[VR_WORKING_STATE_START + 14],
        );
    }

    /// Execute one round of Blake2b compression
    fn blake2_round(&mut self) {
        let sigma_round = &SIGMA[self.round as usize];

        // Column step: apply G function to columns
        self.g_function(0, 4, 8, 12, sigma_round[0], sigma_round[1]);
        self.g_function(1, 5, 9, 13, sigma_round[2], sigma_round[3]);
        self.g_function(2, 6, 10, 14, sigma_round[4], sigma_round[5]);
        self.g_function(3, 7, 11, 15, sigma_round[6], sigma_round[7]);

        // Diagonal step: apply G function to diagonals
        self.g_function(0, 5, 10, 15, sigma_round[8], sigma_round[9]);
        self.g_function(1, 6, 11, 12, sigma_round[10], sigma_round[11]);
        self.g_function(2, 7, 8, 13, sigma_round[12], sigma_round[13]);
        self.g_function(3, 4, 9, 14, sigma_round[14], sigma_round[15]);
    }

    /// Blake2b G function: core mixing function
    /// Updates v[a], v[b], v[c], v[d] using message words m[x], m[y]
    fn g_function(&mut self, a: usize, b: usize, c: usize, d: usize, x: usize, y: usize) {
        let va = self.vr[VR_WORKING_STATE_START + a];
        let vb = self.vr[VR_WORKING_STATE_START + b];
        let vc = self.vr[VR_WORKING_STATE_START + c];
        let vd = self.vr[VR_WORKING_STATE_START + d];
        let mx = self.vr[VR_MESSAGE_BLOCK_START + x];
        let my = self.vr[VR_MESSAGE_BLOCK_START + y];
        let temp1 = self.vr[VR_TEMP];

        // v[a] = v[a] + v[b] + m[x]
        self.add64(Reg(va), Reg(vb), temp1);
        self.add64(Reg(temp1), Reg(mx), va);

        // v[d] = rotr64(v[d] ^ v[a], 32)
        self.xor_rotate(vd, va, RotationAmount::ROT32, vd);

        // v[c] = v[c] + v[d]
        self.add64(Reg(vc), Reg(vd), vc);

        // v[b] = rotr64(v[b] ^ v[c], 24)
        self.xor_rotate(vb, vc, RotationAmount::ROT24, vb);

        // v[a] = v[a] + v[b] + m[y]
        self.add64(Reg(va), Reg(vb), temp1);
        self.add64(Reg(temp1), Reg(my), va);

        // v[d] = rotr64(v[d] ^ v[a], 16)
        self.xor_rotate(vd, va, RotationAmount::ROT16, vd);

        // v[c] = v[c] + v[d]
        self.add64(Reg(vc), Reg(vd), vc);

        // v[b] = rotr64(v[b] ^ v[c], 63)
        self.xor_rotate(vb, vc, RotationAmount::ROT63, vb);
    }

    /// Finalize the hash state according to Blake2b specification:
    /// For i in 0..8: h[i] = h[i] ^ v[i] ^ v[i+8]
    /// This produces the final 8-word hash output in the hash state registers.
    fn finalize_state(&mut self) {
        let temp = self.vr[VR_TEMP];

        for i in 0..HASH_STATE_SIZE {
            let hi = self.vr[VR_HASH_STATE_START + i];
            let vi = self.vr[VR_WORKING_STATE_START + i];
            let vi8 = self.vr[VR_WORKING_STATE_START + i + HASH_STATE_SIZE];

            // temp = v[i] ^ v[i+8]
            self.xor64(Reg(vi), Reg(vi8), temp);
            // h[i] = h[i] ^ temp
            self.xor64(Reg(hi), Reg(temp), hi);
        }
    }

    /// Store the final hash state (8 words) back to memory
    /// Blake2b compression outputs an 8-word hash state
    fn store_state(&mut self) {
        for i in 0..HASH_STATE_SIZE {
            self.sd(self.operand_rs1, self.vr[VR_HASH_STATE_START + i], i as i64);
        }
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
            self.ld(
                base_register,
                (memory_offset_start + i) as i64,
                self.vr[vr_start + i],
            )
        });
    }

    // ADD two 64-bit registers
    fn add64(&mut self, rs1: Value, rs2: Value, rd: usize) -> Value {
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
            _ => unreachable!()
        }
    }

    // To negate register rs2, compute (0 - rs2)
    fn negate64(&mut self, rs2: Value, rd: usize) -> Value {
        match rs2 {
            Reg(rs2) => {
                let sub = SUB {
                    address: self.address,
                    operands: FormatR {
                        rd,
                        rs1: self.vr[VR_ZERO],
                        rs2,
                    },
                    virtual_sequence_remaining: Some(0),
                };
                self.sequence.push(sub.into());
                Reg(rd)
            }
            _ => unreachable!()
        }
    }

    // XOR two 64-bit registers
    fn xor64(&mut self, rs1: Value, rs2: Value, rd: usize) -> Value {
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
            _ => unreachable!()
        }
    }

    // xor two registers, and then rotate right by the given amount.
    fn xor_rotate(&mut self, rs1: usize, rs2: usize, amount: RotationAmount, rd: usize) -> Value {
        match amount {
            RotationAmount::ROT32 => {
                let xor = VirtualROTXOR32 {
                    address: self.address,
                    operands: FormatR { rd, rs1, rs2 },
                    virtual_sequence_remaining: Some(0),
                };
                self.sequence.push(xor.into());
                Reg(rd)
            }
            RotationAmount::ROT24 => {
                let xor = VirtualROTXOR24 {
                    address: self.address,
                    operands: FormatR { rd, rs1, rs2 },
                    virtual_sequence_remaining: Some(0),
                };
                self.sequence.push(xor.into());
                Reg(rd)
            }
            RotationAmount::ROT16 => {
                let xor = VirtualROTXOR16 {
                    address: self.address,
                    operands: FormatR { rd, rs1, rs2 },
                    virtual_sequence_remaining: Some(0),
                };
                self.sequence.push(xor.into());
                Reg(rd)
            }
            RotationAmount::ROT63 => {
                let xor = VirtualROTXOR63 {
                    address: self.address,
                    operands: FormatR { rd, rs1, rs2 },
                    virtual_sequence_remaining: Some(0),
                };
                self.sequence.push(xor.into());
                Reg(rd)
            }
        }
    }

    fn load_64bit_immediate(&mut self, value: u64, rd: usize) {
        let lui = LUI {
            address: self.address,
            operands: FormatU { rd, imm: value },
            virtual_sequence_remaining: Some(0),
        };
        self.sequence.push(lui.into());
    }

    fn ld(&mut self, rs1: usize, offset: i64, rd: usize) {
        let ld = LD {
            address: self.address,
            operands: FormatI {
                rd,
                rs1,
                imm: (offset * 8) as u64,
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
                imm: offset * 8,
            },
            virtual_sequence_remaining: Some(0),
        };
        self.sequence.push(sd.into());
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
/// Rust implementation of Blake2b-256 on the host.
/// ------------------------------------------------------------------------------------------------

/// Execute Blake2b compression with explicit counter values
#[rustfmt::skip]
pub fn execute_blake2b_compression(
    state: &mut [u64; 8],
    message_words: &[u64; 18],
) {
    // Use the host implementation for compression
    use crate::instruction::inline_blake2::{BLAKE2B_IV, SIGMA};

    // Initialize working variables
    let mut v = [0u64; 16];
    v[0..8].copy_from_slice(state);
    v[8..16].copy_from_slice(&BLAKE2B_IV);

    // Blake2b counter handling: XOR counter values with v[12] and v[13]
    v[12] ^= message_words[16]; // counter_low
                      // v[13] ^= counter.shr(64) as u64;  // counter_high

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
