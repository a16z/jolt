//! This file contains BLAKE3-specific logic to expand the inline instruction to a sequence of RISC-V instructions.
//!
//! Glossary:
//!   - "Internal state" = 16-word state array (v[0..15]) used during compression
//!   - "Chaining value" = 8-word state array (h[0..7]) that holds the current hash value
//!   - "Message block" = 16-word input block (m[0..15]) to be compressed
//!   - "Round" = single application of G function mixing to the working state
//!   - "G function" = core mixing function that updates 4 state words using 2 message words

use crate::{CHAINING_VALUE_LEN, COUNTER_LEN, IV, MSG_BLOCK_LEN, MSG_SCHEDULE, NUM_ROUNDS};
use tracer::emulator::cpu::Xlen;
use tracer::instruction::lui::LUI;
use tracer::instruction::lw::LW;
use tracer::instruction::sw::SW;
use tracer::instruction::xor::XOR;
use tracer::instruction::RV32IMInstruction;
use tracer::utils::inline_helpers::{InstrAssembler, Value::Imm, Value::Reg};
use tracer::utils::virtual_registers::allocate_virtual_register_for_inline;

pub const NEEDED_REGISTERS: u8 = 45;

/// Virtual register layout:
/// - vr[0..15]:  Internal state `v`
/// - vr[16..31]: Message block `m`
/// - vr[32..39]: Chaining value `h`
/// - vr[40..41]: Counter values
/// - vr[42]:     Input bytes length
/// - vr[43]:     Flags
/// - vr[44]:     Temporary register
const INTERNAL_STATE_VR_START: usize = 0;
const MSG_BLOCK_START_VR: usize = 16;
const CV_START_VR: usize = 32;
const COUNTER_START_VR: usize = 40;
const INPUT_BYTES_VR: usize = 42;
const FLAG_VR: usize = 43;
const TEMP_VR: usize = 44;

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
        self.load_chaining_value();
        self.load_message_blocks();
        self.load_counter();
        self.load_input_len_and_flags();

        self.initialize_internal_state();

        for round in 0..NUM_ROUNDS {
            self.round = round;
            self.blake3_round();
        }

        self.finalize_state();
        self.store_state();
        self.asm.finalize_inline(NEEDED_REGISTERS)
    }

    fn initialize_internal_state(&mut self) {
        // v[0..7] = h[0..7]
        for i in 0..CHAINING_VALUE_LEN {
            self.asm.xor(
                Reg(self.vr[CV_START_VR + i]),
                Imm(0),
                self.vr[INTERNAL_STATE_VR_START + i],
            );
        }

        // v[8..11] = IV[0..3]
        for (i, val) in IV.iter().enumerate().take(4) {
            self.asm
                .emit_u::<LUI>(self.vr[CHAINING_VALUE_LEN + i], *val as u64);
        }
        // v[12..15] = counter values, input length, and flags
        self.asm.xor(
            Reg(self.vr[COUNTER_START_VR]),
            Imm(0),
            self.vr[INTERNAL_STATE_VR_START + 12],
        );
        self.asm.xor(
            Reg(self.vr[COUNTER_START_VR + 1]),
            Imm(0),
            self.vr[INTERNAL_STATE_VR_START + 13],
        );
        self.asm.xor(
            Reg(self.vr[INPUT_BYTES_VR]),
            Imm(0),
            self.vr[INTERNAL_STATE_VR_START + 14],
        );
        self.asm.xor(
            Reg(self.vr[FLAG_VR]),
            Imm(0),
            self.vr[INTERNAL_STATE_VR_START + 15],
        );
    }

    /// Execute one round of BLAKE3 compression
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

    fn g_function(&mut self, a: usize, b: usize, c: usize, d: usize, x: usize, y: usize) {
        let va = self.vr[a];
        let vb = self.vr[b];
        let vc = self.vr[c];
        let vd = self.vr[d];
        let mx = self.vr[MSG_BLOCK_START_VR + x];
        let my = self.vr[MSG_BLOCK_START_VR + y];
        let temp1 = self.vr[TEMP_VR];

        // v[a] = v[a] + v[b] + m[x]
        self.asm.add(Reg(va), Reg(vb), temp1);
        self.asm.add(Reg(temp1), Reg(mx), va);

        // v[d] = rotr32(v[d] ^ v[a], 16)
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

    fn finalize_state(&mut self) {
        for i in 0..CHAINING_VALUE_LEN {
            let hi = self.vr[CV_START_VR + i];
            let vi = self.vr[INTERNAL_STATE_VR_START + i];
            let vi8 = self.vr[INTERNAL_STATE_VR_START + i + 8];
            self.asm.xor(Reg(vi), Reg(vi8), hi);
        }
    }

    /// Update chaining value
    fn store_state(&mut self) {
        for i in 0..CHAINING_VALUE_LEN {
            self.asm
                .emit_s::<SW>(self.operand_rs1, self.vr[CV_START_VR + i], (i as i64) * 4);
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
            self.asm.emit_ld::<LW>(
                self.vr[vr_start + i],
                base_register,
                (memory_offset_start + i) as i64 * 4,
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

    fn load_chaining_value(&mut self) {
        self.load_data_range(self.operand_rs1, 0, CV_START_VR, CHAINING_VALUE_LEN);
    }

    fn load_message_blocks(&mut self) {
        self.load_data_range(self.operand_rs2, 0, MSG_BLOCK_START_VR, MSG_BLOCK_LEN);
    }

    fn load_counter(&mut self) {
        self.load_data_range(
            self.operand_rs2,
            MSG_BLOCK_LEN,
            COUNTER_START_VR,
            COUNTER_LEN,
        );
    }

    fn load_input_len_and_flags(&mut self) {
        // input length
        self.asm.emit_ld::<LW>(
            self.vr[INPUT_BYTES_VR],
            self.operand_rs2,
            (MSG_BLOCK_LEN + COUNTER_LEN) as i64 * 4,
        );
        // flag
        self.asm.emit_ld::<LW>(
            self.vr[FLAG_VR],
            self.operand_rs2,
            (MSG_BLOCK_LEN + COUNTER_LEN + 1) as i64 * 4,
        );
    }
}

pub fn blake3_inline_sequence_builder(
    address: u64,
    is_compressed: bool,
    xlen: Xlen,
    operand_rs1: u8,
    operand_rs2: u8,
    _operand_rs3: u8,
) -> Vec<RV32IMInstruction> {
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

#[cfg(test)]
mod tests {
    use crate::test_utils::{
        create_blake3_harness, helpers::*, instruction, load_blake3_data, read_output,
        ChainingValue, MessageBlock,
    };

    fn generate_trace_result(
        chaining_value: &ChainingValue,
        message: &MessageBlock,
        counter: &[u32; 2],
        block_len: u32,
        flags: u32,
    ) -> [u8; crate::OUTPUT_SIZE_IN_BYTES] {
        let mut harness = create_blake3_harness();
        load_blake3_data(
            &mut harness,
            chaining_value,
            message,
            counter,
            block_len,
            flags,
        );
        harness.execute_inline(instruction());
        let words = read_output(&mut harness);

        let mut bytes = [0u8; crate::OUTPUT_SIZE_IN_BYTES];
        for (i, w) in words.iter().enumerate() {
            let le = w.to_le_bytes();
            bytes[i * 4..(i + 1) * 4].copy_from_slice(&le);
        }
        bytes
    }

    #[test]
    fn test_trace_result_equals_blake3_compress_reference() {
        for _ in 0..1000 {
            let message_bytes = generate_random_bytes(crate::MSG_BLOCK_LEN * 4);
            // Convert bytes to message block (u32 words)
            assert_eq!(
                message_bytes.len(),
                crate::MSG_BLOCK_LEN * 4,
                "Message must be exactly {} bytes",
                crate::MSG_BLOCK_LEN * 4
            );
            let words_vec = bytes_to_u32_vec(&message_bytes);
            let mut message_words = [0u32; crate::MSG_BLOCK_LEN];
            message_words.copy_from_slice(&words_vec);

            let expected_hash_bytes = compute_expected_result(&message_bytes);
            let counter = [0u32, 0u32];
            let block_len = 64u32;
            let flags = crate::FLAG_CHUNK_START | crate::FLAG_CHUNK_END | crate::FLAG_ROOT;
            let trace_hash_bytes =
                generate_trace_result(&crate::IV, &message_words, &counter, block_len, flags);
            assert_eq!(
                trace_hash_bytes, expected_hash_bytes,
                "trace hash bytes mismatch"
            );
        }
    }

    #[test]
    fn test_trace_result_equals_blake3_keyed_compress_reference() {
        for _ in 0..1000 {
            // Generate random key
            let key_bytes = generate_random_bytes(crate::CHAINING_VALUE_LEN * 4);
            let mut key = [0u32; crate::CHAINING_VALUE_LEN];
            key.copy_from_slice(&bytes_to_u32_vec(&key_bytes));

            // Generate random message
            let message_bytes = generate_random_bytes(crate::MSG_BLOCK_LEN * 4);
            let words_vec = bytes_to_u32_vec(&message_bytes);
            let mut message_words = [0u32; crate::MSG_BLOCK_LEN];
            message_words.copy_from_slice(&words_vec);

            // Compute expected result using keyed hash
            let expected_hash_bytes = compute_keyed_expected_result(&message_bytes, key);

            // Generate trace result with keyed hash flag
            let counter = [0u32, 0u32];
            let block_len = 64u32;
            let flags = crate::FLAG_CHUNK_START
                | crate::FLAG_CHUNK_END
                | crate::FLAG_ROOT
                | crate::FLAG_KEYED_HASH;
            let trace_hash_bytes =
                generate_trace_result(&key, &message_words, &counter, block_len, flags);

            assert_eq!(
                trace_hash_bytes, expected_hash_bytes,
                "keyed trace hash bytes mismatch"
            );
        }
    }
}
