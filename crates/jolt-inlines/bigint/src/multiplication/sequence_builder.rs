use jolt_inlines_sdk::host::{
    instruction::{add::ADD, ld::LD, mul::MUL, mulhu::MULHU, sd::SD, sltu::SLTU},
    ExpandedInstructionSequence, ExpansionError, InlineExpansionBuilder, InlineOp, InlineOperands,
    InlineRegister, NoAdvice,
};

use super::{INPUT_LIMBS, OUTPUT_LIMBS};

/// Number of virtual registers needed for BigInt multiplication
/// Layout:
/// - a0..a3: First operand (4 u64 limbs)
/// - a4..a7: Second operand (4 u64 limbs)
/// - s0..s1: Result limbs (2 u64 limbs, accumulator and carry)
/// - t0    : Temporary for handling carry propagation
pub(crate) const NEEDED_REGISTERS: usize = 11;

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
    fn b(&self, i: usize) -> u8 {
        *self.vr[INPUT_LIMBS + i]
    }
    // Results
    fn s(&self, i: usize) -> u8 {
        *self.vr[INPUT_LIMBS + INPUT_LIMBS + (i % 2)]
    }
    // Temporary for carry propagation
    fn t(&self) -> u8 {
        *self.vr[INPUT_LIMBS + INPUT_LIMBS + 2]
    }

    /// Builds the complete multiplication sequence
    fn build(mut self) -> Result<ExpandedInstructionSequence, ExpansionError> {
        for i in 0..INPUT_LIMBS {
            self.asm
                .emit_ld::<LD>(self.a(i), self.operands.rs1, i as i64 * 8);
        }

        for i in 0..INPUT_LIMBS {
            self.asm
                .emit_ld::<LD>(self.b(i), self.operands.rs2, i as i64 * 8);
        }

        // Inline finalization ensures that s0 and s1 start at zero
        // Thus no explicit initialization is needed

        // 0th limb is just a multiplication with no carry
        self.asm.emit_r::<MUL>(self.s(0), self.a(0), self.b(0));
        self.asm.emit_s::<SD>(self.operands.rs3, self.s(0), 0); // Store 0th limb immediately

        // 1st limb is 0 and doesn't receive a carry from the 0th limb
        // so initialize it with the upper half of A[0] * B[0]
        self.asm.emit_r::<MULHU>(self.s(1), self.a(0), self.b(0));

        // For each output limb R[k]
        for k in 1..OUTPUT_LIMBS {
            // alternate between s0 and s1 for accumulating results and carries to minimize register usage
            // overwrite carry register on first addition, then accumulate into it for subsequent additions
            let mut overwrite_carry = true;

            for i in 0..INPUT_LIMBS {
                for j in 0..INPUT_LIMBS {
                    if i == 0 && j == 0 {
                        continue; // skip the A[0] * B[0] term which is already handled
                    }
                    // add all lower(A[i] * B[j]) where i+j = k
                    if i + j == k {
                        // t = low part of A[i] * B[j]
                        self.asm.emit_r::<MUL>(self.t(), self.a(i), self.b(j));
                    // add all upper(A[i] * B[j]) where i+j = k-1
                    } else if i + j == k - 1 {
                        // t = high part of A[i] * B[j]
                        self.asm.emit_r::<MULHU>(self.t(), self.a(i), self.b(j));
                    }
                    // handle carry propagation
                    if i + j == k || i + j == k - 1 {
                        // add product to accumulator
                        self.asm.emit_r::<ADD>(self.s(k), self.s(k), self.t());
                        // test for a carry and either set or accumulate it
                        if overwrite_carry {
                            self.asm.emit_r::<SLTU>(self.s(k + 1), self.s(k), self.t());
                        } else {
                            self.asm.emit_r::<SLTU>(self.t(), self.s(k), self.t());
                            self.asm
                                .emit_r::<ADD>(self.s(k + 1), self.t(), self.s(k + 1));
                        }
                        // after the first addition, we need to accumulate carries instead of overwriting them
                        overwrite_carry = false;
                    }
                }
            }

            // store the accumulated result limb
            self.asm
                .emit_s::<SD>(self.operands.rs3, self.s(k), k as i64 * 8);
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
