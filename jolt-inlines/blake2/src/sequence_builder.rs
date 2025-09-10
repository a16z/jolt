//! BLAKE2b-specific logic to expand the inline instruction into a sequence of RISC-V instructions.
//!
//! Glossary:
//!   - "Working state" = 16-word state array (v[0..15]) used during compression
//!   - "Hash state" = 8-word state array (h[0..7]) that holds the current hash value
//!   - "Message block" = 16-word input block (m[0..15]) to be compressed
//!   - "Round" = single application of G function mixing to the working state
//!   - "G function" = core mixing function that updates 4 state words using 2 message words

use core::array;

use crate::{IV, SIGMA};
use tracer::instruction::format::format_inline::FormatInline;
use tracer::instruction::ld::LD;
use tracer::instruction::lui::LUI;
use tracer::instruction::sd::SD;
use tracer::instruction::sub::SUB;
use tracer::instruction::xor::XOR;
use tracer::instruction::RV32IMInstruction;
use tracer::utils::inline_helpers::{InstrAssembler, Value::Imm, Value::Reg};
use tracer::utils::virtual_registers::VirtualRegisterGuard;

pub const NEEDED_REGISTERS: u8 = 43;

/// Virtual register layout:
/// - `vr[0..15]`:  Working state `v` (16 words)
/// - `vr[16..31]`: Message block `m` (16 words)
/// - `vr[32..39]`: Hash state `h` (8 words)
/// - `vr[40]`:     Counter value `t`
/// - `vr[41]`:     Final block flag
/// - `vr[42]`:     Temporary register
const VR_WORKING_STATE_START: usize = 0;
const WORKING_STATE_SIZE: usize = 16;
const VR_MESSAGE_BLOCK_START: usize = 16;
const VR_HASH_STATE_START: usize = 32;
const VR_T: usize = 40;
const VR_IS_FINAL: usize = 41;
const VR_TEMP: usize = 42;

const BLAKE2_NUM_ROUNDS: u8 = 12;

// Rotation constants required in BLAKE2b
pub enum RotationAmount {
    ROT32,
    ROT24,
    ROT16,
    ROT63,
}

struct Blake2SequenceBuilder {
    asm: InstrAssembler,
    round: u8,
    vr: [VirtualRegisterGuard; NEEDED_REGISTERS as usize],
    operands: FormatInline,
}

impl Blake2SequenceBuilder {
    fn new(asm: InstrAssembler, operands: FormatInline) -> Self {
        let vr = array::from_fn(|_| asm.allocator.allocate_for_inline());
        Blake2SequenceBuilder {
            asm,
            round: 0,
            vr,
            operands,
        }
    }

    fn build(mut self) -> Vec<RV32IMInstruction> {
        self.load_hash_state();
        self.load_message_blocks();
        self.load_counter_and_is_final();

        self.initialize_working_state();

        for round in 0..BLAKE2_NUM_ROUNDS {
            self.round = round;
            self.blake2_round();
        }

        self.finalize_state();
        self.store_state();
        self.asm.finalize_inline(NEEDED_REGISTERS)
    }

    fn load_hash_state(&mut self) {
        self.load_data_range(
            self.operands.rs1,
            0,
            VR_HASH_STATE_START,
            crate::STATE_VECTOR_LEN,
        );
    }

    fn load_message_blocks(&mut self) {
        self.load_data_range(
            self.operands.rs2,
            0,
            VR_MESSAGE_BLOCK_START,
            crate::MSG_BLOCK_LEN,
        );
    }

    fn load_counter_and_is_final(&mut self) {
        self.asm.emit_ld::<LD>(
            *self.vr[VR_T],
            self.operands.rs2,
            crate::MSG_BLOCK_LEN as i64 * 8,
        );
        self.asm.emit_ld::<LD>(
            *self.vr[VR_IS_FINAL],
            self.operands.rs2,
            (crate::MSG_BLOCK_LEN as i64 + 1) * 8,
        );
    }

    // Initialize the working state v[0..15] according to the BLAKE2b specification.
    fn initialize_working_state(&mut self) {
        // v[0..7] = h[0..7]
        for i in 0..crate::STATE_VECTOR_LEN {
            self.asm.xor(
                Reg(*self.vr[VR_HASH_STATE_START + i]),
                Imm(0),
                *self.vr[VR_WORKING_STATE_START + i],
            );
        }

        // v[8..15] = IV[0..7]
        for (i, value) in IV
            .iter()
            .enumerate()
            .take(WORKING_STATE_SIZE - crate::STATE_VECTOR_LEN)
        {
            // Load BLAKE2b IV constants.
            let rd = *self.vr[VR_WORKING_STATE_START + crate::STATE_VECTOR_LEN + i];
            self.asm.emit_u::<LUI>(rd, *value);
        }

        // v[12] = v[12] ^ t (counter low)
        self.asm.xor(
            Reg(*self.vr[VR_WORKING_STATE_START + 12]),
            Reg(*self.vr[VR_T]),
            *self.vr[VR_WORKING_STATE_START + 12],
        );

        // Since we are using 64-bit counter, the high part is always 0, so v[13] remains unchanged
        // v[13] = IV[5] ^ (t >> 64) (counter high) - for 64-bit counter, high part is 0.

        // Handle final block flag: if is_final != 0, invert all bits of v[14]
        // Create a mask that is 0xFFFFFFFFFFFFFFFF if is_final != 0, or 0 if is_final == 0.
        // Use the formula: mask = (0 - is_final) to convert 1 to 0xFFFFFFFFFFFFFFFF and 0 to 0
        // First, negate is_final (0 - is_final).
        // Using 0 as x0, which is always 0 in RISC-V.
        self.asm
            .emit_r::<SUB>(*self.vr[VR_TEMP], 0, *self.vr[VR_IS_FINAL]);
        // XOR v[14] with the mask: inverts all bits if is_final=1, leaves unchanged if is_final=0.
        self.asm.xor(
            Reg(*self.vr[VR_WORKING_STATE_START + 14]),
            Reg(*self.vr[VR_TEMP]),
            *self.vr[VR_WORKING_STATE_START + 14],
        );
    }

    /// Execute one round of BLAKE2b compression
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

    fn g_function(&mut self, a: usize, b: usize, c: usize, d: usize, x: usize, y: usize) {
        let va = *self.vr[VR_WORKING_STATE_START + a];
        let vb = *self.vr[VR_WORKING_STATE_START + b];
        let vc = *self.vr[VR_WORKING_STATE_START + c];
        let vd = *self.vr[VR_WORKING_STATE_START + d];
        let mx = *self.vr[VR_MESSAGE_BLOCK_START + x];
        let my = *self.vr[VR_MESSAGE_BLOCK_START + y];
        let temp1 = *self.vr[VR_TEMP];

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

    fn finalize_state(&mut self) {
        let temp = *self.vr[VR_TEMP];

        for i in 0..crate::STATE_VECTOR_LEN {
            let hi = *self.vr[VR_HASH_STATE_START + i];
            let vi = *self.vr[VR_WORKING_STATE_START + i];
            let vi8 = *self.vr[VR_WORKING_STATE_START + i + crate::STATE_VECTOR_LEN];

            // temp = v[i] ^ v[i+8]
            self.asm.xor(Reg(vi), Reg(vi8), temp);
            // h[i] = h[i] ^ temp
            self.asm.xor(Reg(hi), Reg(temp), hi);
        }
    }

    /// Store the final hash state
    fn store_state(&mut self) {
        for i in 0..crate::STATE_VECTOR_LEN {
            self.asm.emit_s::<SD>(
                self.operands.rs1,
                *self.vr[VR_HASH_STATE_START + i],
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
                *self.vr[vr_start + i],
                base_register,
                ((memory_offset_start + i) * 8) as i64,
            );
        });
    }

    /// XOR two registers, then rotate right by the given amount.
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

pub fn blake2b_inline_sequence_builder(
    asm: InstrAssembler,
    operands: FormatInline,
) -> Vec<RV32IMInstruction> {
    let builder = Blake2SequenceBuilder::new(asm, operands);
    builder.build()
}

#[cfg(test)]
mod tests {
    use crate::{
        test_utils::{create_blake2_harness, instruction, load_blake2_data, read_state},
        IV,
    };

    fn generate_default_input() -> ([u64; crate::MSG_BLOCK_LEN], u64) {
        // Message block with "abc" in little-endian
        let mut message = [0u64; crate::MSG_BLOCK_LEN];
        message[0] = 0x0000000000636261u64;
        (message, 3)
    }

    fn generate_random_input() -> ([u64; crate::MSG_BLOCK_LEN], u64) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        let mut input = [0u64; crate::MSG_BLOCK_LEN];
        for val in input.iter_mut() {
            *val = rng.gen();
        }
        (input, 128)
    }

    fn compute_reference_blake2b_hash(
        message: &[u64; crate::MSG_BLOCK_LEN],
        message_len: usize,
    ) -> [u64; crate::STATE_VECTOR_LEN] {
        use blake2::{Blake2b512, Digest};
        let mut message_bytes: Vec<u8> = message.iter().flat_map(|w| w.to_le_bytes()).collect();
        let effective_len = core::cmp::min(message_len, message_bytes.len());
        message_bytes.truncate(effective_len);

        let hash_result = Blake2b512::digest(&message_bytes);

        let mut state = [0u64; crate::STATE_VECTOR_LEN];
        for (i, chunk) in hash_result.chunks_exact(8).enumerate() {
            if i < crate::STATE_VECTOR_LEN {
                state[i] = u64::from_le_bytes(chunk.try_into().unwrap());
            }
        }
        state
    }

    fn generate_trace_result(
        state: &[u64; crate::STATE_VECTOR_LEN],
        message: &[u64; crate::MSG_BLOCK_LEN],
        counter: u64,
        is_final: bool,
    ) -> [u64; crate::STATE_VECTOR_LEN] {
        let mut harness = create_blake2_harness();
        load_blake2_data(&mut harness, state, message, counter, is_final);
        harness.execute_inline(instruction());
        read_state(&mut harness)
    }

    /// Helper function to test blake2b compression with given input
    fn verify_blake2b_compression(message_words: [u64; crate::MSG_BLOCK_LEN], message_len: u64) {
        let mut initial_state = IV;
        initial_state[0] ^= 0x01010000 ^ 64u64;

        let expected_state = compute_reference_blake2b_hash(&message_words, message_len as usize);
        let trace_result = generate_trace_result(&initial_state, &message_words, message_len, true);

        assert_eq!(
            &expected_state, &trace_result,
            "\n❌ BLAKE2b Trace Verification Failed!\n\
            Message: {message_words:016x?}"
        );
    }

    #[test]
    fn test_trace_result_with_default_input() {
        let input = generate_default_input();
        verify_blake2b_compression(input.0, input.1);
    }

    #[test]
    fn test_trace_result_with_random_inputs() {
        // Test with multiple random inputs
        for _ in 0..10 {
            let input = generate_random_input();
            verify_blake2b_compression(input.0, input.1);
        }
    }
}
