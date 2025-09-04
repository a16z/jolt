//! This file contains Blake3-specific logic to expand the inline instruction to a sequence of RISC-V instructions.
//!
//! Glossary:
//!   - "Working state" = 16-word state array (v[0..15]) used during compression
//!   - "Chaining value" = 8-word state array (h[0..7]) that holds the current hash value
//!   - "Message block" = 16-word input block (m[0..15]) to be compressed
//!   - "Round" = single application of G function mixing to the working state
//!   - "G function" = core mixing function that updates 4 state words using 2 message words

use crate::{NUM_ROUNDS, IV, MSG_SCHEDULE};
use tracer::emulator::cpu::Xlen;
use tracer::instruction::lw::LW;
use tracer::instruction::lui::LUI;
use tracer::instruction::sw::SW;
#[allow(unused_imports)]
use tracer::instruction::virtual_xor_rot::{VirtualXORROT16, VirtualXORROT24, VirtualXORROT32, VirtualXORROT63};
use tracer::instruction::xor::XOR;
use tracer::instruction::RV32IMInstruction;
use tracer::utils::inline_helpers::Value;
use tracer::utils::inline_helpers::{
    InstrAssembler,
    Value::Reg,
    Value::Imm,
};
use tracer::utils::virtual_registers::allocate_virtual_register_for_inline;

pub const NEEDED_REGISTERS: u8 = 45;

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

pub enum RotationAmount {
    ROT16,
    ROT12,
    ROT8,
    ROT7,
}

struct Blake3SequenceBuilder {
    asm: InstrAssembler,
    round: u8,
    vr: [u8; NEEDED_REGISTERS as usize],
    operand_rs1: u8,
    operand_rs2: u8,
}

impl Blake3SequenceBuilder {
    fn new(
        address: u64,
        is_compressed: bool,
        xlen: Xlen,
        vr: [u8; NEEDED_REGISTERS as usize],
        operand_rs1: u8,
        operand_rs2: u8,
    ) -> Self {
        Blake3SequenceBuilder {
            asm: InstrAssembler::new_inline(address, is_compressed, xlen),
            round: 0,
            vr,
            operand_rs1,
            operand_rs2,
        }
    }

    fn build(mut self) -> Vec<RV32IMInstruction> {
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
        self.asm.emit_ld::<LW>(self.vr[VR_INPUT_BYTES_LEN_START], self.operand_rs2, (MESSAGE_BLOCK_SIZE + COUNTER_SIZE) as i64 * 4);
        // Load the final block flag from memory - stored after the input bytes length
        self.asm.emit_ld::<LW>(self.vr[VR_FLAG_START], self.operand_rs2, (MESSAGE_BLOCK_SIZE + COUNTER_SIZE + 1) as i64 * 4);

        // 2. Initialize the internal state v[0..15]
        self.initialize_internal_state();

        // 3. Main loop: 7 rounds of Blake3 compression
        for round in 0..NUM_ROUNDS {
            self.round = round;
            self.blake3_round();
        }

        // 4. Finalize the hash state: h[i] ^= v[i] ^ v[i+8]
        self.finalize_state(false);

        // 5. Store the final hash state back to memory
        self.store_state();

        self.asm.finalize_inline(NEEDED_REGISTERS)
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
            self.asm.emit_ld::<LW>(self.vr[vr_start + i], base_register, (memory_offset_start + i) as i64 * 4);
        });
    }

    /// Initialize the working state v[0..15] according to Blake3 specification:
    fn initialize_internal_state(&mut self) {
        // v[0..7] = h[0..7] (current hash state)
        for i in 0..CHAINING_VALUE_SIZE {
            self.asm.xor(Reg(self.vr[VR_CHAINING_VALUE_START + i]), Imm(0), self.vr[VR_STATE_START + i]);
        }

        // v[8..11] = IV[0..3] (first 4 words of initialization vector)
        for i in 0..4 {
            self.asm.emit_u::<LUI>(self.vr[CHAINING_VALUE_SIZE + i], IV[i] as u64);
        }
        // v[12..15] = counter values, input length, and flags
        self.asm.xor(
            Reg(self.vr[VR_COUNTER_START]),
            Imm(0),
            self.vr[VR_STATE_START + 12],
        );
        self.asm.xor(
            Reg(self.vr[VR_COUNTER_START + 1]),
            Imm(0),
            self.vr[VR_STATE_START + 13],
        );
        self.asm.xor(
            Reg(self.vr[VR_INPUT_BYTES_LEN_START]),
            Imm(0),
            self.vr[VR_STATE_START + 14],
        );
        self.asm.xor(
            Reg(self.vr[VR_FLAG_START]),
            Imm(0),
            self.vr[VR_STATE_START + 15],
        );
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
         self.asm.add(Reg(va), Reg(vb), temp1);
         self.asm.add(Reg(temp1), Reg(mx), va);

        // // v[d] = rotr32(v[d] ^ v[a], 16)
        self.xor_rotate(vd, va, RotationAmount::ROT16, vd);

        // v[c] = v[c] + v[d]
         self.asm.add(Reg(vc), Reg(vd), vc);

        // v[b] = rotr32(v[b] ^ v[c], 12)
        self.xor_rotate(vb, vc, RotationAmount::ROT12, vb);

        // v[a] = v[a] + v[b] + m[y]
         self.asm.add(Reg(va), Reg(vb), temp1);
         self.asm.add(Reg(temp1), Reg(my), va);

        // v[d] = rotr32(v[d] ^ v[a], 8)
        self.xor_rotate(vd, va, RotationAmount::ROT8, vd);

        // v[c] = v[c] + v[d]
         self.asm.add(Reg(vc), Reg(vd), vc);

        // v[b] = rotr32(v[b] ^ v[c], 7)
        self.xor_rotate(vb, vc, RotationAmount::ROT7, vb);
    }

    fn finalize_state(&mut self, is_truncated: bool) {
        if is_truncated {
            for i in 0..STATE_SIZE / 2 {
                let hi = self.vr[VR_CHAINING_VALUE_START + i];
                let vi = self.vr[VR_STATE_START + i];
                let vi8 = self.vr[VR_STATE_START + i + 8];
                self.asm.xor(Reg(vi), Reg(vi8), hi);
            }
        } else {
            for i in 0..STATE_SIZE / 2 {
                let hi = self.vr[VR_STATE_START + i];
                let vi = self.vr[VR_STATE_START + i];
                let vi8 = self.vr[VR_STATE_START + i + 8];
                self.asm.xor(Reg(vi), Reg(vi8), hi);
            }
            for i in 0..STATE_SIZE / 2 {
                let hi_prime = self.vr[VR_STATE_START + i + 8];
                let vi = self.vr[VR_STATE_START + i + 8];
                let hi = self.vr[VR_CHAINING_VALUE_START + i];
                self.asm.xor(Reg(vi), Reg(hi), hi_prime);
            }
        }
    }

    /// Store the final hash state (chaining value) back to memory
    /// Blake3 compression outputs an 8-word chaining value
    fn store_state(&mut self) {
        for i in 0..STATE_SIZE {
            self.asm.emit_s::<SW>(self.operand_rs1, self.vr[VR_STATE_START + i], (i as i64) * 4);
        }
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
            RotationAmount::ROT16 => {
                self.asm.rotri32(Reg(rd), 16, rd);
            }
            RotationAmount::ROT12 => {
                self.asm.rotri32(Reg(rd), 12, rd);
            }
            RotationAmount::ROT8 => {
                self.asm.rotri32(Reg(rd), 8, rd);
            }
            RotationAmount::ROT7 => {
                self.asm.rotri32(Reg(rd), 7, rd);
            }
        }
    }
}

/// Build Blake2b inline sequence for the RISC-V instruction stream
pub fn blake3_inline_sequence_builder(
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
        Blake3SequenceBuilder::new(address, is_compressed, xlen, vr, operand_rs1, operand_rs2);
    builder.build()
}
