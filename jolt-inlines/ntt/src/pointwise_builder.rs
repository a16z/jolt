use jolt_inlines_sdk::host::{
    ExpandedInstructionSequence, ExpansionError, InlineBuilderExt, InlineExpansionBuilder,
    InlineOp, InlineOperands, InlineRegister, NoAdvice, SourceKind as Kind,
};

use crate::pointwise::{DOT_FUNCT7, DOT_PRODUCTS};
use crate::sequence_builder::reduce;
use crate::{DEGREE, OPCODE};

pub struct PointwiseDot64;

struct DotBuilder {
    asm: InlineExpansionBuilder,
    operands: InlineOperands,
    sums: [InlineRegister; DEGREE / 2],
    pointers: [InlineRegister; 2 * DOT_PRODUCTS],
    primes: [InlineRegister; 2],
    scratch: [InlineRegister; 6],
}

impl DotBuilder {
    fn build(mut self) -> Result<ExpandedInstructionSequence, ExpansionError> {
        let [p, pinv] = self.primes.map(|r| *r);
        let [a_lo, a_hi, b_lo, b_hi, temp, mask] = self.scratch.map(|r| *r);
        for (index, pointer) in self.pointers.iter().enumerate() {
            self.asm
                .emit_ld(Kind::LD, **pointer, self.operands.rs2, index as i64 * 8);
        }
        self.asm.emit_ld(
            Kind::LD,
            p,
            self.operands.rs2,
            (2 * DOT_PRODUCTS) as i64 * 8,
        );
        self.asm.emit_i(Kind::SRAI, pinv, p, 32);
        self.asm.emit_i(Kind::ADDIW, p, p, 0);
        for block in (0..DEGREE).step_by(self.sums.len()) {
            for product in 0..DOT_PRODUCTS {
                for lane in (0..self.sums.len()).step_by(2) {
                    let offset = (block + lane) as i64 * 4;
                    self.asm
                        .emit_ld(Kind::LD, a_lo, *self.pointers[product], offset);
                    self.asm.emit_i(Kind::SRAI, a_hi, a_lo, 32);
                    self.asm.emit_i(Kind::ADDIW, a_lo, a_lo, 0);
                    self.asm.emit_ld(
                        Kind::LD,
                        b_lo,
                        *self.pointers[DOT_PRODUCTS + product],
                        offset,
                    );
                    self.asm.emit_i(Kind::SRAI, b_hi, b_lo, 32);
                    self.asm.emit_i(Kind::ADDIW, b_lo, b_lo, 0);
                    for (sum, a, b) in [
                        (*self.sums[lane], a_lo, b_lo),
                        (*self.sums[lane + 1], a_hi, b_hi),
                    ] {
                        if product == 0 {
                            self.asm.emit_r(Kind::MUL, sum, a, b);
                        } else {
                            self.asm.emit_r(Kind::MUL, temp, a, b);
                            self.asm.emit_r(Kind::ADD, sum, sum, temp);
                        }
                    }
                }
            }
            for lane in (0..self.sums.len()).step_by(2) {
                let lo = *self.sums[lane];
                let hi = *self.sums[lane + 1];
                for sum in [lo, hi] {
                    self.asm.emit_r(Kind::MULW, temp, sum, pinv);
                    self.asm.emit_r(Kind::MUL, temp, temp, p);
                    self.asm.emit_r(Kind::SUB, sum, sum, temp);
                    self.asm.emit_i(Kind::SRAI, sum, sum, 32);
                    reduce(&mut self.asm, sum, p, temp, mask);
                }
                let offset = (block + lane) as i64 * 4;
                self.asm.emit_ld(Kind::LD, a_lo, self.operands.rs1, offset);
                self.asm.emit_i(Kind::SRAI, a_hi, a_lo, 32);
                self.asm.emit_i(Kind::ADDIW, a_lo, a_lo, 0);
                self.asm.emit_r(Kind::ADDW, lo, lo, a_lo);
                self.asm.emit_r(Kind::ADDW, hi, hi, a_hi);
                reduce(&mut self.asm, lo, p, temp, mask);
                reduce(&mut self.asm, hi, p, temp, mask);
                self.asm.store_paired_u32(self.operands.rs1, offset, lo, hi);
            }
        }
        self.asm.release_many(self.sums);
        self.asm.release_many(self.pointers);
        self.asm.release_many(self.primes);
        self.asm.release_many(self.scratch);
        self.asm.finalize()
    }
}

impl InlineOp for PointwiseDot64 {
    type Advice = NoAdvice;
    const OPCODE: u32 = OPCODE;
    const FUNCT3: u32 = 0;
    const FUNCT7: u32 = DOT_FUNCT7;
    const NAME: &'static str = "NTT64_I32_POINTWISE_DOT6";

    fn build_sequence(
        mut asm: InlineExpansionBuilder,
        operands: InlineOperands,
    ) -> Result<ExpandedInstructionSequence, ExpansionError> {
        let sums = asm.allocate_inline_array::<{ DEGREE / 2 }>()?;
        let pointers = asm.allocate_inline_array::<{ 2 * DOT_PRODUCTS }>()?;
        let primes = asm.allocate_inline_array::<2>()?;
        let scratch = asm.allocate_inline_array::<6>()?;
        DotBuilder {
            asm,
            operands,
            sums,
            pointers,
            primes,
            scratch,
        }
        .build()
    }
}
