use jolt_inlines_sdk::host::{
    instruction::{addc::ADDC, ld::LD, mul::MUL, mulc::MULC, sd::SD},
    ExpandedInstructionSequence, ExpansionError, InlineExpansionBuilder, InlineOp, InlineOperands,
    InlineRegister, NoAdvice,
};

use super::{INPUT_LIMBS, OUTPUT_LIMBS};

/// Number of virtual registers needed for BigInt multiplication
/// Layout:
/// - a0..a3: First operand (4 u64 limbs)
/// - p0..p4: Partial products (5 u64 limbs)
///   Top limb is also used as the active limb of the second operand.
/// - r0..r4: Rolling result window (5 u64 limbs)
pub(crate) const NEEDED_REGISTERS: usize = INPUT_LIMBS + 2 * (INPUT_LIMBS + 1);

/// Builds assembly sequence for 256-bit × 256-bit multiplication
/// Expects first operand (4 u64 words) in RAM at location rs1
/// Expects second operand (4 u64 words) in RAM at location rs2
/// Output (8 u64 words) will be written to the memory rs3 points to
struct BigIntMulSequenceBuilder {
    asm: InlineExpansionBuilder,
    /// Virtual registers used by the sequence
    vr: [InlineRegister; NEEDED_REGISTERS],
    operands: InlineOperands,
}

impl BigIntMulSequenceBuilder {
    fn new(
        mut asm: InlineExpansionBuilder,
        operands: InlineOperands,
    ) -> Result<Self, ExpansionError> {
        let vr = asm.allocate_inline_array::<NEEDED_REGISTERS>()?;
        Ok(BigIntMulSequenceBuilder { asm, vr, operands })
    }

    /// Register indices for operands and temporaries
    // LHS
    fn a(&self, i: usize) -> u8 {
        *self.vr[i]
    }
    // RHS
    fn b(&self) -> u8 {
        *self.vr[INPUT_LIMBS + INPUT_LIMBS]
    }
    // Partial products
    fn p(&self, i: usize) -> u8 {
        *self.vr[INPUT_LIMBS + i]
    }
    // Rolling result window
    fn r(&self, i: usize) -> u8 {
        *self.vr[INPUT_LIMBS + INPUT_LIMBS + 1 + (i % (INPUT_LIMBS + 1))]
    }

    /// Builds the complete multiplication sequence
    fn build(mut self) -> Result<ExpandedInstructionSequence, ExpansionError> {
        for i in 0..INPUT_LIMBS {
            self.asm
                .emit_ld::<LD>(self.a(i), self.operands.rs1, i as i64 * 8);
        }

        // Seed the rolling window with a * b[0].
        self.asm.emit_ld::<LD>(self.b(), self.operands.rs2, 0);
        self.asm.emit_r::<MUL>(self.r(0), self.a(0), self.b());
        for i in 1..INPUT_LIMBS {
            self.asm.emit_r::<MULC>(self.r(i), self.a(i), self.b());
        }
        self.asm.emit_r::<ADDC>(self.r(INPUT_LIMBS), 0, 0);
        self.asm.emit_s::<SD>(self.operands.rs3, self.r(0), 0);

        // Each iteration adds p << (i * 64) into the rolling window and stores limb i.
        // The highest slot in the ring has already been written back to memory, so we
        // overwrite it on the final ADDC instead of clearing it ahead of time.
        for i in 1..INPUT_LIMBS {
            self.asm
                .emit_ld::<LD>(self.b(), self.operands.rs2, i as i64 * 8);
            self.asm.emit_r::<MUL>(self.p(0), self.a(0), self.b());
            for j in 1..INPUT_LIMBS {
                self.asm.emit_r::<MULC>(self.p(j), self.a(j), self.b());
            }
            self.asm.emit_r::<ADDC>(self.p(INPUT_LIMBS), 0, 0);

            for j in 0..INPUT_LIMBS {
                self.asm
                    .emit_r::<ADDC>(self.r(i + j), self.r(i + j), self.p(j));
            }
            self.asm
                .emit_r::<ADDC>(self.r(i + INPUT_LIMBS), 0, self.p(INPUT_LIMBS));

            self.asm
                .emit_s::<SD>(self.operands.rs3, self.r(i), i as i64 * 8);
        }

        for i in INPUT_LIMBS..OUTPUT_LIMBS {
            self.asm
                .emit_s::<SD>(self.operands.rs3, self.r(i), i as i64 * 8);
        }

        self.asm.release_many(self.vr);
        self.asm.finalize()
    }
}

pub struct BigintMul256;

impl InlineOp for BigintMul256 {
    type Advice = NoAdvice;

    const OPCODE: u32 = crate::INLINE_OPCODE;
    const FUNCT3: u32 = crate::BIGINT256_MUL_FUNCT3;
    const FUNCT7: u32 = crate::BIGINT256_MUL_FUNCT7;
    const NAME: &'static str = crate::BIGINT256_MUL_NAME;

    fn build_sequence(
        asm: InlineExpansionBuilder,
        operands: InlineOperands,
    ) -> Result<ExpandedInstructionSequence, ExpansionError> {
        BigIntMulSequenceBuilder::new(asm, operands)?.build()
    }
}
