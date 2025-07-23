use crate::instruction::format::format_i::FormatI;
use crate::instruction::format::format_r::FormatR;
use crate::instruction::format::format_s::FormatS;
use crate::instruction::format::format_u::FormatU;
use crate::instruction::format::format_virtual_right_shift_i::FormatVirtualRightShiftI;
use crate::instruction::ld::LD;
use crate::instruction::lui::LUI;
use crate::instruction::or::OR;
use crate::instruction::sd::SD;
use crate::instruction::slli::SLLI;
use crate::instruction::virtual_rotri::VirtualROTRI;
use crate::instruction::xor::XOR;
use crate::instruction::xori::XORI;
use crate::instruction::RV32IMInstruction;
/// This file contains Blake2-specific logic to be used in the Blake2 inline:
/// 1) Prover: Blake2SequenceBuilder expands the inline to a list of RV instructions.
/// 2) Host: Rust reference implementation to be called by jolt-sdk.
///
/// Blake2 is a cryptographic hash function that operates on 64-bit words in a compression function.
/// Glossary:
///   - "Word" = one 64-bit value in the 16-word state matrix.
///   - "Round" = single application of G function to all columns and diagonals.
///   - "Block" = 128 bytes (16 words) of input data.
///   - "Compression" = Blake2b compression function: 12 rounds of G function.
/// Blake2b-256 refers to Blake2b with 256-bit (32-byte) output.
use crate::instruction::{add::ADD, addi::ADDI, sub::SUB};

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

#[derive(Clone, Copy)]
enum Value {
    Imm(u64),
    Reg(usize),
}
use Value::{Imm, Reg};

/// Layout of the 96 virtual registers (`vr`) for Blake2.
///
/// Jolt requires the total number of registers (physical + virtual) to be a power of two.
/// With 32 physical registers, we need 96 virtual registers to reach 128.
///
pub const NEEDED_REGISTERS: usize = 96;

#[allow(dead_code)]
const VR_WORKING_STATE_START: usize = 0; // vr[0..15]: Working state `v` (16 words)
const WORKING_STATE_SIZE: usize = 16;
const VR_MESSAGE_BLOCK_START: usize = 16; // vr[16..31]: Message block `m` (16 words)
const MESSAGE_BLOCK_SIZE: usize = 16;
const VR_HASH_STATE_START: usize = 32; // vr[32..39]: Hash state `h` (8 words)
const HASH_STATE_SIZE: usize = 8;
const VR_T: usize = 40;
const VR_IS_FINAL: usize = 41;
const VR_TEMP: usize = 42;
// vr[40..79]: Temporary registers for G function (40 words)
// vr[80..95]: General-purpose scratch registers (16 words)

struct Blake2SequenceBuilder {
    address: u64,
    sequence: Vec<RV32IMInstruction>,
    round: u32,
    vr: [usize; NEEDED_REGISTERS],
    operand_rs1: usize,
    operand_rs2: usize,
}

/// `Blake2SequenceBuilder` is a helper struct for constructing the virtual instruction
/// sequence required to emulate the Blake2b hashing operation within the RISC-V
/// instruction set. This builder is responsible for generating the correct sequence of
/// `RV32IMInstruction` instances that together perform the Blake2b compression
/// function, using a set of virtual registers to hold intermediate state.
///
/// # Fields
/// - `address`: The starting program counter address for the sequence.
/// - `sequence`: The vector of generated instructions representing the Blake2 operation.
/// - `round`: The current round of the Blake2 compression (0..12).
/// - `vr`: An array of virtual register indices used for state and temporary values.
/// - `operand_rs1`: The source register index for the first operand (state pointer).
/// - `operand_rs2`: The source register index for the second operand (message block pointer).

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
        // 1. Load all required data from memory
        self.load_state(); // Load hash state (8 words)
        self.load_message(); // Load message block (16 words)
        self.load_t(); // Load counter value (1 word)
        self.load_is_final(); // Load final block flag (1 word)

        // 2. Initialize the working state v[0..15]
        self.initialize_working_state();

        // 3. Main loop: 12 rounds of Blake2b compression
        for round in 0..12 {
            self.round = round;
            self.blake2_round();
        }

        // 4. Finalize the hash state: h[i] ^= v[i] ^ v[i+8]
        self.finalize_state();

        // 5. Store the final hash state back to memory
        self.store_state();

        // 6. Finalize the sequence by setting instruction indices
        self.enumerate_sequence();
        self.sequence
    }

    /// Load the current hash state (8 words) from memory into virtual registers
    fn load_state(&mut self) {
        (0..HASH_STATE_SIZE)
            .for_each(|i| self.ld(self.operand_rs1, i as i64, self.vr[VR_HASH_STATE_START + i]));
    }

    /// Load the message block (16 words) from memory into virtual registers
    fn load_message(&mut self) {
        (0..MESSAGE_BLOCK_SIZE).for_each(|i| {
            self.ld(
                self.operand_rs2,
                i as i64,
                self.vr[VR_MESSAGE_BLOCK_START + i],
            )
        });
    }

    /// Load the counter value (t) from memory - stored after the message block
    fn load_t(&mut self) {
        self.ld(self.operand_rs2, MESSAGE_BLOCK_SIZE as i64, self.vr[VR_T]);
    }

    /// Load the final block flag (is_final) from memory - stored after the counter
    fn load_is_final(&mut self) {
        self.ld(
            self.operand_rs2,
            MESSAGE_BLOCK_SIZE as i64 + 1,
            self.vr[VR_IS_FINAL],
        );
    }

    /// Initialize the working state v[0..15] according to Blake2b specification
    fn initialize_working_state(&mut self) {
        // v[0..7] = h[0..7] (current hash state)
        for i in 0..HASH_STATE_SIZE {
            self.xor64(Reg(self.vr[VR_HASH_STATE_START + i]), Imm(0), self.vr[i]);
        }

        // v[8..15] = IV[0..7] (initialization vector)
        for i in 0..WORKING_STATE_SIZE - HASH_STATE_SIZE {
            self.load_64bit_immediate(BLAKE2B_IV[i], self.vr[HASH_STATE_SIZE + i]);
        }

        // Blake2b counter handling: XOR counter with v[12] and extract high part for v[13]
        // v[12] = vr[12] ^ t (counter low)
        self.xor64(Reg(self.vr[12]), Reg(self.vr[VR_T]), self.vr[12]);

        // v[13] = IV[5] ^ (t >> 64) (counter high) - for 64-bit counter, high part is 0
        // Since we're using 64-bit counter, the high part is always 0, so v[13] remains unchanged

        // Handle final block flag: if is_final != 0, invert all bits of v[14]
        // We need to create a mask that is 0xFFFFFFFFFFFFFFFF if is_final != 0, or 0 if is_final == 0
        // Use the formula: mask = (0 - is_final) to convert 1 to 0xFFFFFFFFFFFFFFFF and 0 to 0
        let temp_mask = self.vr[VR_TEMP];

        // First, negate is_final (0 - is_final)
        self.sub64(Imm(0), Reg(self.vr[VR_IS_FINAL]), temp_mask);

        // XOR v[14] with the mask: inverts all bits if is_final=1, leaves unchanged if is_final=0
        self.xor64(Reg(self.vr[14]), Reg(temp_mask), self.vr[14]);
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
        let va = self.vr[a];
        let vb = self.vr[b];
        let vc = self.vr[c];
        let vd = self.vr[d];
        let mx = self.vr[VR_MESSAGE_BLOCK_START + x];
        let my = self.vr[VR_MESSAGE_BLOCK_START + y];
        let temp1 = self.vr[VR_TEMP];

        // v[a] = v[a] + v[b] + m[x]
        self.add64(Reg(va), Reg(vb), temp1);
        self.add64(Reg(temp1), Reg(mx), va);

        // v[d] = rotr64(v[d] ^ v[a], 32)
        self.xor64(Reg(vd), Reg(va), temp1);
        self.rotr64(Reg(temp1), 32, vd);

        // v[c] = v[c] + v[d]
        self.add64(Reg(vc), Reg(vd), vc);

        // v[b] = rotr64(v[b] ^ v[c], 24)
        self.xor64(Reg(vb), Reg(vc), temp1);
        self.rotr64(Reg(temp1), 24, vb);

        // v[a] = v[a] + v[b] + m[y]
        self.add64(Reg(va), Reg(vb), temp1);
        self.add64(Reg(temp1), Reg(my), va);

        // v[d] = rotr64(v[d] ^ v[a], 16)
        self.xor64(Reg(vd), Reg(va), temp1);
        self.rotr64(Reg(temp1), 16, vd);

        // v[c] = v[c] + v[d]
        self.add64(Reg(vc), Reg(vd), vc);

        // v[b] = rotr64(v[b] ^ v[c], 63)
        self.xor64(Reg(vb), Reg(vc), temp1);
        self.rotr64(Reg(temp1), 63, vb);
    }

    /// Finalize the hash state: h[i] ^= v[i] ^ v[i+8]
    fn finalize_state(&mut self) {
        for i in 0..8 {
            let hi = self.vr[VR_HASH_STATE_START + i];
            let vi = self.vr[i];
            let vi8 = self.vr[i + 8];
            let temp = self.vr[VR_TEMP];

            self.xor64(Reg(vi), Reg(vi8), temp);
            self.xor64(Reg(hi), Reg(temp), hi);
        }
    }

    /// Store the final hash state back to memory
    fn store_state(&mut self) {
        for i in 0..HASH_STATE_SIZE {
            self.sd(self.operand_rs1, self.vr[VR_HASH_STATE_START + i], i as i64);
        }
    }

    // --- 64-bit Arithmetic Helpers ---

    /// ADD two 64-bit numbers (using lower 32 bits for RISC-V compatibility)
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
            (Reg(rs1), Imm(imm)) => {
                let addi = ADDI {
                    address: self.address,
                    operands: FormatI { rd, rs1, imm },
                    virtual_sequence_remaining: Some(0),
                };
                self.sequence.push(addi.into());
                Reg(rd)
            }
            (Imm(_), Reg(_)) => self.add64(rs2, rs1, rd),
            (Imm(imm1), Imm(imm2)) => Imm((imm1).wrapping_add(imm2)),
        }
    }

    /// SUB two 64-bit numbers (using lower 32 bits for RISC-V compatibility)
    fn sub64(&mut self, rs1: Value, rs2: Value, rd: usize) -> Value {
        match (rs1, rs2) {
            (Reg(rs1), Reg(rs2)) => {
                let sub = SUB {
                    address: self.address,
                    operands: FormatR { rd, rs1, rs2 },
                    virtual_sequence_remaining: Some(0),
                };
                self.sequence.push(sub.into());
                Reg(rd)
            }
            (Imm(imm), Reg(rs2)) => {
                // For immediate - register, we need to first load the immediate
                let temp_reg = self.vr[67]; // Use a temp register
                self.load_64bit_immediate(imm, temp_reg);
                let sub = SUB {
                    address: self.address,
                    operands: FormatR {
                        rd,
                        rs1: temp_reg,
                        rs2,
                    },
                    virtual_sequence_remaining: Some(0),
                };
                self.sequence.push(sub.into());
                Reg(rd)
            }
            (Reg(_), Imm(_)) => {
                // Register - immediate: convert to register + (-immediate)
                // This is complex, so let's use the first approach with temp register
                panic!("SUB with reg - imm not implemented, use imm - reg instead")
            }
            (Imm(imm1), Imm(imm2)) => Imm((imm1).wrapping_sub(imm2)),
        }
    }

    /// XOR two 64-bit numbers
    fn xor64(&mut self, rs1: Value, rs2: Value, rd: usize) -> Value {
        self.xor(rs1, rs2, rd)
    }

    /// Right rotate a 64-bit number
    fn rotr64(&mut self, rs1: Value, amount: u32, rd: usize) -> Value {
        if amount == 0 {
            return self.xor(rs1, Imm(0), rd);
        }

        match rs1 {
            Reg(rs1_reg) => {
                // Convert right rotation to left rotation: rotr(x, n) = rotl(x, 64-n)
                let left_amount = 64 - amount;
                let ones = (1u64 << left_amount) - 1;
                let imm = ones << (64 - left_amount);

                let rotri = VirtualROTRI {
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

    fn load_64bit_immediate(&mut self, value: u64, rd: usize) {
        let lui = LUI {
            address: self.address,
            operands: FormatU { rd, imm: value },
            virtual_sequence_remaining: Some(0),
        };
        self.sequence.push(lui.into());
    }
    // --- RV64 Instruction Emitters ---

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
pub fn execute_blake2b_compression(
    state: &mut [u64; 8],
    message_words: &[u64; 16],
    counter: u64,
    is_final: bool,
) {
    // Use the host implementation for compression
    use crate::instruction::inline_blake2::{BLAKE2B_IV, SIGMA};

    // Initialize working variables
    let mut v = [0u64; 16];
    v[0..8].copy_from_slice(state);
    v[8..16].copy_from_slice(&BLAKE2B_IV);

    // Blake2b counter handling: XOR counter values with v[12] and v[13]
    v[12] ^= counter; // counter_low
                      // v[13] ^= counter.shr(64) as u64;  // counter_high

    // Set final block flag if this is the last block
    if is_final {
        v[14] = !v[14]; // Invert v[14] for final block
    }

    // 12 rounds of mixing
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

    // Finalize hash state
    for i in 0..8 {
        state[i] ^= v[i] ^ v[i + 8];
    }
}

/// Blake2b G function
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

#[cfg(test)]
mod tests {
    use super::*;

    /// High-level Blake2 function following RFC specification
    ///
    /// Parameters:
    /// - data_blocks: Array of data blocks d[0..dd-1]
    /// - ll: Length of input in bytes
    /// - kk: Key length (0 for unkeyed)  
    /// - nn: Output length in bytes
    fn blake2(data_blocks: &[[u64; 16]], ll: u64, kk: u64, nn: u64) -> Vec<u8> {
        const BB: u64 = 128; // Block size in bytes

        // h[0..7] := IV[0..7] (Initialization Vector)
        let mut h = BLAKE2B_IV;

        // Parameter block p[0]: h[0] := h[0] ^ 0x01010000 ^ (kk << 8) ^ nn
        h[0] ^= 0x01010000 ^ (kk << 8) ^ nn;

        let dd = data_blocks.len();

        // Process padded key and data blocks
        if dd > 1 {
            for i in 0..(dd - 1) {
                let counter = (i as u64 + 1) * BB;
                execute_blake2b_compression(&mut h, &data_blocks[i], counter, false);
            }
        }

        // Final block
        let final_counter = if kk == 0 { ll } else { ll + BB };
        execute_blake2b_compression(&mut h, &data_blocks[dd - 1], final_counter, true);

        // Return first "nn" bytes from little-endian word array h[]
        let mut result = Vec::with_capacity(nn as usize);
        for &word in h.iter() {
            let bytes = word.to_le_bytes();
            for &byte in bytes.iter() {
                if result.len() < nn as usize {
                    result.push(byte);
                }
            }
        }
        result
    }

    #[test]
    fn test_blake2_rfc_abc() {
        // Test case from Blake2 RFC for "abc"
        let mut message_block = [0u64; 16];
        message_block[0] = 0x0000000000636261u64; // "abc" in little-endian
                                                  // All other words remain 0 (padding)

        let data_blocks = [message_block];
        let ll = 3u64; // Length of "abc" in bytes
        let kk = 0u64; // Key length (unkeyed)
        let nn = 64u64; // Output length (Blake2b-512)

        // Execute Blake2 function following RFC specification
        let result = blake2(&data_blocks, ll, kk, nn);

        assert_eq!(result.len(), 64);
        let expected = [
            0xba, 0x80, 0xa5, 0x3f, 0x98, 0x1c, 0x4d, 0x0d, 0x6a, 0x27, 0x97, 0xb6, 0x9f, 0x12,
            0xf6, 0xe9, 0x4c, 0x21, 0x2f, 0x14, 0x68, 0x5a, 0xc4, 0xb7, 0x4b, 0x12, 0xbb, 0x6f,
            0xdb, 0xff, 0xa2, 0xd1, 0x7d, 0x87, 0xc5, 0x39, 0x2a, 0xab, 0x79, 0x2d, 0xc2, 0x52,
            0xd5, 0xde, 0x45, 0x33, 0xcc, 0x95, 0x18, 0xd3, 0x8a, 0xa8, 0xdb, 0xf1, 0x92, 0x5a,
            0xb9, 0x23, 0x86, 0xed, 0xd4, 0x00, 0x99, 0x23,
        ];
        assert_eq!(result.as_slice(), &expected);
    }
}
