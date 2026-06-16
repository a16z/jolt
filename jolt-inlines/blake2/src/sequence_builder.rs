//! BLAKE2b-specific logic to expand the inline instruction into a sequence of RISC-V instructions.
//!
//! Glossary:
//!   - "Working state" = 16-word state array (v[0..15]) used during compression
//!   - "Hash state" = 8-word state array (h[0..7]) that holds the current hash value
//!   - "Message block" = 16-word input block (m[0..15]) to be compressed
//!   - "Round" = single application of G function mixing to the working state
//!   - "G function" = core mixing function that updates 4 state words using 2 message words

use crate::{IV, SIGMA};
use jolt_inlines_sdk::host::{
    instruction::{
        ld::LD,
        lui::LUI,
        sub::SUB,
        virtual_xor_rot::{VirtualXORROT16, VirtualXORROT24, VirtualXORROT32, VirtualXORROT63},
    },
    ExpandedInstructionSequence, ExpansionError, InlineBuilderExt, InlineExpansionBuilder,
    InlineOp, InlineOperands, InlineRegister,
    Value::{Imm, Reg},
};

pub const NEEDED_REGISTERS: usize = 43;

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

struct Blake2SequenceBuilder {
    asm: InlineExpansionBuilder,
    round: u8,
    vr: [InlineRegister; NEEDED_REGISTERS],
    operands: InlineOperands,
}

impl Blake2SequenceBuilder {
    fn new(
        mut asm: InlineExpansionBuilder,
        operands: InlineOperands,
    ) -> Result<Self, ExpansionError> {
        let vr = asm.allocate_inline_array::<NEEDED_REGISTERS>()?;
        Ok(Blake2SequenceBuilder {
            asm,
            round: 0,
            vr,
            operands,
        })
    }

    fn build(mut self) -> Result<ExpandedInstructionSequence, ExpansionError> {
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
        self.asm.release_many(self.vr);
        self.asm.finalize()
    }

    fn load_hash_state(&mut self) {
        self.asm.load_u64_range(
            self.operands.rs1,
            0,
            &self.vr[VR_HASH_STATE_START..VR_HASH_STATE_START + crate::STATE_VECTOR_LEN],
        );
    }

    fn load_message_blocks(&mut self) {
        self.asm.load_u64_range(
            self.operands.rs2,
            0,
            &self.vr[VR_MESSAGE_BLOCK_START..VR_MESSAGE_BLOCK_START + crate::MSG_BLOCK_LEN],
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
        self.asm.emit_r::<VirtualXORROT32>(vd, vd, va);

        // v[c] = v[c] + v[d]
        self.asm.add(Reg(vc), Reg(vd), vc);

        // v[b] = rotr64(v[b] ^ v[c], 24)
        self.asm.emit_r::<VirtualXORROT24>(vb, vb, vc);

        // v[a] = v[a] + v[b] + m[y]
        self.asm.add(Reg(va), Reg(vb), temp1);
        self.asm.add(Reg(temp1), Reg(my), va);

        // v[d] = rotr64(v[d] ^ v[a], 16)
        self.asm.emit_r::<VirtualXORROT16>(vd, vd, va);

        // v[c] = v[c] + v[d]
        self.asm.add(Reg(vc), Reg(vd), vc);

        // v[b] = rotr64(v[b] ^ v[c], 63)
        self.asm.emit_r::<VirtualXORROT63>(vb, vb, vc);
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
        self.asm.store_u64_range(
            self.operands.rs1,
            0,
            &self.vr[VR_HASH_STATE_START..VR_HASH_STATE_START + crate::STATE_VECTOR_LEN],
        );
    }
}

pub struct Blake2bCompression;

impl InlineOp for Blake2bCompression {
    type Advice = ();

    const OPCODE: u32 = crate::INLINE_OPCODE;
    const FUNCT3: u32 = crate::BLAKE2_FUNCT3;
    const FUNCT7: u32 = crate::BLAKE2_FUNCT7;
    const NAME: &'static str = crate::BLAKE2_NAME;

    fn build_sequence(
        asm: InlineExpansionBuilder,
        operands: InlineOperands,
    ) -> Result<ExpandedInstructionSequence, ExpansionError> {
        Blake2SequenceBuilder::new(asm, operands)?.build()
    }
}

#[cfg(test)]
mod tests {
    use super::Blake2bCompression;
    use crate::IV;
    use jolt_inlines_sdk::{assert_random_cases_match_reference, assert_reference_matches_harness};

    fn initial_state() -> [u64; crate::STATE_VECTOR_LEN] {
        let mut state = IV;
        state[0] ^= 0x01010000 ^ 64u64;
        state
    }

    fn default_input() -> (
        [u64; crate::STATE_VECTOR_LEN],
        [u64; crate::MSG_BLOCK_LEN],
        u64,
        bool,
    ) {
        let mut message = [0u64; crate::MSG_BLOCK_LEN];
        message[0] = 0x0000000000636261u64;
        (initial_state(), message, 3, true)
    }

    #[test]
    fn test_trace_result_with_default_input() {
        assert_reference_matches_harness::<Blake2bCompression>(&default_input());
    }

    #[test]
    fn test_trace_result_with_random_inputs() {
        assert_random_cases_match_reference::<Blake2bCompression>(0xB1A2E2, 10);
    }
}
