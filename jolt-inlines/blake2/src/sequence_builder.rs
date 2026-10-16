use crate::{IV, SIGMA};
use jolt_inlines_sdk::host::{
    ExpandedInstructionSequence, ExpansionError, InlineBuilderExt, InlineExpansionBuilder,
    InlineOp, InlineOperands, InlineRegister, Kind, NoAdvice,
    Value::{Imm, Reg},
};
use jolt_inlines_sdk::jolt_asm;

pub const NEEDED_REGISTERS: usize = 40;

const VR_WORKING_STATE_START: usize = 0;
const VR_MESSAGE_BLOCK_START: usize = 16;
const VR_HASH_STATE_START: usize = 32;

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
        self.load_tail_into_working_state();

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

    /// Load the counter `t` into v[12] and the final-block flag into v[14].
    /// `initialize_working_state` folds the IV constants into those slots.
    fn load_tail_into_working_state(&mut self) {
        jolt_asm!(self.asm, {
            ld *self.vr[VR_WORKING_STATE_START + 12], self.operands.rs2, crate::MSG_BLOCK_LEN as i64 * 8;
            ld *self.vr[VR_WORKING_STATE_START + 14], self.operands.rs2, (crate::MSG_BLOCK_LEN as i64 + 1) * 8;
        });
    }

    // Initialize the working state v[0..15] according to the BLAKE2b specification.
    fn initialize_working_state(&mut self) {
        for i in 0..crate::STATE_VECTOR_LEN {
            self.asm.emit_i(
                Kind::XORI,
                *self.vr[VR_WORKING_STATE_START + i],
                *self.vr[VR_HASH_STATE_START + i],
                0,
            );
        }

        // v[8..15] = IV[0..7], loading the BLAKE2b IV constants. v[12] and v[14] are
        // skipped: they already hold t / is_final and fold in IV[4] / IV[6] below.
        for i in [0, 1, 2, 3, 5, 7] {
            let rd = *self.vr[VR_WORKING_STATE_START + crate::STATE_VECTOR_LEN + i];
            self.asm.emit_u(Kind::LUI, rd, IV[i]);
        }

        // v[12] = IV[4] ^ t (counter low)
        self.asm.xor(
            Reg(*self.vr[VR_WORKING_STATE_START + 12]),
            Imm(IV[4]),
            *self.vr[VR_WORKING_STATE_START + 12],
        );

        // v[13] = IV[5] ^ (t >> 64) (counter high) - since we are using a 64-bit
        // counter, the high part is always 0, so v[13] keeps the plain IV[5].

        self.asm.emit_r(
            Kind::SUB,
            *self.vr[VR_WORKING_STATE_START + 14],
            0,
            *self.vr[VR_WORKING_STATE_START + 14],
        );
        // XOR the mask with IV[6]: v[14] = IV[6], bits inverted iff is_final = 1.
        self.asm.xor(
            Reg(*self.vr[VR_WORKING_STATE_START + 14]),
            Imm(IV[6]),
            *self.vr[VR_WORKING_STATE_START + 14],
        );
    }

    fn blake2_round(&mut self) {
        let sigma_round = &SIGMA[self.round as usize];

        self.g_function(0, 4, 8, 12, sigma_round[0], sigma_round[1]);
        self.g_function(1, 5, 9, 13, sigma_round[2], sigma_round[3]);
        self.g_function(2, 6, 10, 14, sigma_round[4], sigma_round[5]);
        self.g_function(3, 7, 11, 15, sigma_round[6], sigma_round[7]);

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
        jolt_asm!(self.asm, {
            add va, va, vb;
            add va, va, mx;
            xorrot32 vd, vd, va;
            add vc, vc, vd;
            xorrot24 vb, vb, vc;
            add va, va, vb;
            add va, va, my;
            xorrot16 vd, vd, va;
            add vc, vc, vd;
            xorrot63 vb, vb, vc;
        });
    }

    fn finalize_state(&mut self) {
        for i in 0..crate::STATE_VECTOR_LEN {
            let hi = *self.vr[VR_HASH_STATE_START + i];
            let vi = *self.vr[VR_WORKING_STATE_START + i];
            let vi8 = *self.vr[VR_WORKING_STATE_START + i + crate::STATE_VECTOR_LEN];

            jolt_asm!(self.asm, {
                xor vi, vi, vi8;
                xor hi, hi, vi;
            });
        }
    }

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
    type Advice = NoAdvice;

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
    use jolt_inlines_sdk::{
        assert_edge_cases_match_reference, assert_random_cases_match_reference,
    };

    #[test]
    fn test_trace_result_with_edge_cases() {
        assert_edge_cases_match_reference::<Blake2bCompression>();
    }

    #[test]
    fn test_trace_result_with_random_inputs() {
        assert_random_cases_match_reference::<Blake2bCompression>(0xB1A2E2, 10);
    }
}
