//! Blake2b is a cryptographic hash function operating on 64-bit words.
//! It uses a compression function that performs 12 rounds of mixing operations
//! on a 16-word working state derived from the hash state and message block.
//!
//! Glossary:
//!   - "Working state" = 16-word state array (v[0..15]) used during compression
//!   - "Hash state" = 8-word state array (h[0..7]) that holds the current hash value
//!   - "Message block" = 16-word input block (m[0..15]) to be compressed
//!   - "Round" = single application of G function mixing to the working state
//!   - "G function" = core mixing function that updates 4 state words using 2 message words

use tracer::emulator::cpu::Xlen;
use tracer::instruction::ld::LD;
use tracer::instruction::lui::LUI;
use tracer::instruction::sd::SD;
use tracer::instruction::sub::SUB;
#[allow(unused_imports)]
use tracer::instruction::virtual_xor_rot::{
    VirtualXORROT16, VirtualXORROT24, VirtualXORROT32, VirtualXORROT63,
};
use tracer::instruction::xor::XOR;
use tracer::instruction::RV32IMInstruction;
use tracer::utils::inline_helpers::{InstrAssembler, Value::Imm, Value::Reg};
use tracer::utils::virtual_registers::allocate_virtual_register_for_inline;

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

/// Virtual register layout:
/// - `vr[0..15]`:  Working state `v` (16 words)
/// - `vr[16..31]`: Message block `m` (16 words)
/// - `vr[32..39]`: Hash state `h` (8 words)
/// - `vr[40]`:     Counter value `t`
/// - `vr[41]`:     Final block flag
/// - `vr[42]`:     Temporary register
/// - `vr[43]`:     Zero constant
pub const NEEDED_REGISTERS: u8 = 43;

// Memory layout constants for Blake2 virtual registers
const VR_WORKING_STATE_START: usize = 0;
const WORKING_STATE_SIZE: usize = 16;
const VR_MESSAGE_BLOCK_START: usize = 16;
pub const MESSAGE_BLOCK_SIZE: usize = 16;
const VR_HASH_STATE_START: usize = 32;
pub const HASH_STATE_SIZE: usize = 8;
const VR_T: usize = 40;
const VR_IS_FINAL: usize = 41;
const VR_TEMP: usize = 42;

const BLAKE2_NUM_ROUNDS: u8 = 12;

// Rotation constants required in Blake2b
pub enum RotationAmount {
    ROT32,
    ROT24,
    ROT16,
    ROT63,
}

struct Blake2SequenceBuilder {
    asm: InstrAssembler,
    round: u8,
    vr: [u8; NEEDED_REGISTERS as usize],
    operand_rs1: u8,
    operand_rs2: u8,
}

/// # Fields
/// - `asm`: Builder for the vector of generated instructions representing the Blake2b operation.
/// - `round`: The current round of the Blake2b compression (0..12).
/// - `vr`: An array of virtual register indices used for state and temporary values.
/// - `operand_rs1`: The source register index for the hash state pointer.
/// - `operand_rs2`: The source register index for the message block pointer.
impl Blake2SequenceBuilder {
    fn new(
        address: u64,
        is_compressed: bool,
        xlen: Xlen,
        vr: [u8; NEEDED_REGISTERS as usize],
        operand_rs1: u8,
        operand_rs2: u8,
    ) -> Self {
        Blake2SequenceBuilder {
            asm: InstrAssembler::new_inline(address, is_compressed, xlen),
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
        self.load_counter_and_is_final();

        // Initialize the working state v[0..15]
        self.initialize_working_state();

        // Cryptographic mixing for 12 rounds
        for round in 0..BLAKE2_NUM_ROUNDS {
            self.round = round;
            self.blake2_round();
        }

        // Finalize the hash state
        self.finalize_state();

        // Store the final hash state back to memory
        self.store_state();

        self.asm.finalize_inline(NEEDED_REGISTERS)
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

    fn load_counter_and_is_final(&mut self) {
        self.asm.emit_ld::<LD>(
            self.vr[VR_T],
            self.operand_rs2,
            MESSAGE_BLOCK_SIZE as i64 * 8,
        );
        self.asm.emit_ld::<LD>(
            self.vr[VR_IS_FINAL],
            self.operand_rs2,
            (MESSAGE_BLOCK_SIZE as i64 + 1) * 8,
        );
    }

    // Initialize the working state v[0..15] according to Blake2b specification
    fn initialize_working_state(&mut self) {
        // v[0..7] = h[0..7]
        for i in 0..HASH_STATE_SIZE {
            self.asm.xor(
                Reg(self.vr[VR_HASH_STATE_START + i]),
                Imm(0),
                self.vr[VR_WORKING_STATE_START + i],
            );
        }

        // v[8..15] = IV[0..7]
        for (i, value) in BLAKE2B_IV
            .iter()
            .enumerate()
            .take(WORKING_STATE_SIZE - HASH_STATE_SIZE)
        {
            // Load Blake2b IV constants
            let rd = self.vr[VR_WORKING_STATE_START + HASH_STATE_SIZE + i];
            self.asm.emit_u::<LUI>(rd, *value);
        }

        // v[12] = v[12] ^ t (counter low)
        self.asm.xor(
            Reg(self.vr[VR_WORKING_STATE_START + 12]),
            Reg(self.vr[VR_T]),
            self.vr[VR_WORKING_STATE_START + 12],
        );

        // Since we are using 64-bit counter, the high part is always 0, so v[13] remains unchanged
        // v[13] = IV[5] ^ (t >> 64) (counter high) - for 64-bit counter, high part is 0

        // Handle final block flag: if is_final != 0, invert all bits of v[14]
        // We need to create a mask that is 0xFFFFFFFFFFFFFFFF if is_final != 0, or 0 if is_final == 0
        // Use the formula: mask = (0 - is_final) to convert 1 to 0xFFFFFFFFFFFFFFFF and 0 to 0
        // First, negate is_final (0 - is_final)
        // Using 0 as x0, which is always 0 in risc-v
        self.asm
            .emit_r::<SUB>(self.vr[VR_TEMP], 0, self.vr[VR_IS_FINAL]);
        // XOR v[14] with the mask: inverts all bits if is_final=1, leaves unchanged if is_final=0
        self.asm.xor(
            Reg(self.vr[VR_WORKING_STATE_START + 14]),
            Reg(self.vr[VR_TEMP]),
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
        self.asm.add(Reg(va), Reg(vb), temp1);
        self.asm.add(Reg(temp1), Reg(mx), va);

        // v[d] = rotr64(v[d] ^ v[a], 32)
        self.xor_rotate(vd, va, RotationAmount::ROT32, vd);

        // v[c] = v[c] + v[d]
        self.asm.add(Reg(vc), Reg(vd), vc);

        // v[b] = rotr64(v[b] ^ v[c], 24)
        self.xor_rotate(vb, vc, RotationAmount::ROT24, vb);

        // v[a] = v[a] + v[b] + m[y]
        self.asm.add(Reg(va), Reg(vb), temp1);
        self.asm.add(Reg(temp1), Reg(my), va);

        // v[d] = rotr64(v[d] ^ v[a], 16)
        self.xor_rotate(vd, va, RotationAmount::ROT16, vd);

        // v[c] = v[c] + v[d]
        self.asm.add(Reg(vc), Reg(vd), vc);

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
            self.asm.xor(Reg(vi), Reg(vi8), temp);
            // h[i] = h[i] ^ temp
            self.asm.xor(Reg(hi), Reg(temp), hi);
        }
    }

    /// Store the final hash state (8 words) back to memory
    /// Blake2b compression outputs an 8-word hash state
    fn store_state(&mut self) {
        for i in 0..HASH_STATE_SIZE {
            self.asm.emit_s::<SD>(
                self.operand_rs1,
                self.vr[VR_HASH_STATE_START + i],
                (i * 8) as i64,
            );
        }
    }

    /// Load data from memory into virtual registers starting at a given offset
    fn load_data_range(
        &mut self,
        base_register: u8,
        memory_offset_start: usize,
        vr_start: usize,
        count: usize,
    ) {
        (0..count).for_each(|i| {
            self.asm.emit_ld::<LD>(
                self.vr[vr_start + i],
                base_register,
                ((memory_offset_start + i) * 8) as i64,
            );
        });
    }

    /// XOR two registers, and then rotate right by the given amount.
    fn xor_rotate(&mut self, rs1: u8, rs2: u8, amount: RotationAmount, rd: u8) {
        // match amount {
        //     RotationAmount::ROT32 => {
        //         self.asm.emit_r::<VirtualXORROT32>(rd, rs1, rs2);
        //     }
        //     RotationAmount::ROT24 => {
        //         self.asm.emit_r::<VirtualXORROT24>(rd, rs1, rs2);
        //     }
        //     RotationAmount::ROT16 => {
        //         self.asm.emit_r::<VirtualXORROT16>(rd, rs1, rs2);
        //     }
        //     RotationAmount::ROT63 => {
        //         self.asm.emit_r::<VirtualXORROT63>(rd, rs1, rs2);
        //     }
        // }

        self.asm.emit_r::<XOR>(rd, rs1, rs2);
        match amount {
            RotationAmount::ROT32 => {
                self.asm.rotr64(Reg(rd), 32, rd);
            }
            RotationAmount::ROT24 => {
                self.asm.rotr64(Reg(rd), 24, rd);
            }
            RotationAmount::ROT16 => {
                self.asm.rotr64(Reg(rd), 16, rd);
            }
            RotationAmount::ROT63 => {
                self.asm.rotr64(Reg(rd), 63, rd);
            }
        }
    }
}

/// Build Blake2b inline sequence for the RISC-V instruction stream
pub fn blake2b_inline_sequence_builder(
    address: u64,
    is_compressed: bool,
    xlen: Xlen,
    operand_rs1: u8,
    operand_rs2: u8,
    _operand_rs3: u8,
) -> Vec<RV32IMInstruction> {
    // Virtual registers used as a scratch space
    let guards: Vec<_> = (0..NEEDED_REGISTERS)
        .map(|_| allocate_virtual_register_for_inline())
        .collect();
    let mut vr = [0; NEEDED_REGISTERS as usize];
    for (i, guard) in guards.iter().enumerate() {
        vr[i] = **guard;
    }
    let builder =
        Blake2SequenceBuilder::new(address, is_compressed, xlen, vr, operand_rs1, operand_rs2);
    builder.build()
}
