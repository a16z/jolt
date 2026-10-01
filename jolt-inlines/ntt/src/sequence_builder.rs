use jolt_inlines_sdk::host::{
    ExpandedInstructionSequence, ExpansionError, InlineBuilderExt, InlineExpansionBuilder,
    InlineOp, InlineOperands, InlineRegister, NoAdvice, SourceKind as Kind,
};

use crate::{DEGREE, FUNCT3, FUNCT7, OPCODE};

pub struct ForwardNtt64;

struct NttBuilder {
    asm: InlineExpansionBuilder,
    operands: InlineOperands,
    values: [InlineRegister; DEGREE],
    scratch: [InlineRegister; 8],
}

impl NttBuilder {
    fn build(mut self) -> Result<ExpandedInstructionSequence, ExpansionError> {
        let [psi, twiddles, p, pinv, temp, _, _, _] = self.scratch.map(|r| *r);
        self.asm.emit_ld(Kind::LD, psi, self.operands.rs2, 0);
        self.asm.emit_ld(Kind::LD, twiddles, self.operands.rs2, 8);
        self.asm.emit_ld(Kind::LD, p, self.operands.rs2, 16);
        self.asm.emit_i(Kind::SRAI, pinv, p, 32);
        self.asm.emit_i(Kind::ADDIW, p, p, 0);
        for i in (0..DEGREE).step_by(2) {
            let lo = *self.values[i];
            let hi = *self.values[i + 1];
            self.asm
                .emit_ld(Kind::LD, lo, self.operands.rs1, i as i64 * 4);
            self.asm.emit_i(Kind::SRAI, hi, lo, 32);
            self.asm.emit_i(Kind::ADDIW, lo, lo, 0);
        }
        // One table word supplies both twiddles; coefficients stay live across
        // every stage, using 72 of the 80 inline registers.
        for i in (0..DEGREE).step_by(2) {
            self.asm.emit_ld(Kind::LD, temp, psi, i as i64 * 4);
            self.asm.emit_i(Kind::ADDIW, psi, temp, 0);
            self.mont_mul(*self.values[i], *self.values[i], psi);
            self.asm.emit_i(Kind::SRAI, temp, temp, 32);
            self.mont_mul(*self.values[i + 1], *self.values[i + 1], temp);
            self.asm.emit_ld(Kind::LD, psi, self.operands.rs2, 0);
        }
        let mut len = DEGREE / 2;
        while len != 0 {
            // The stage starts at an odd i32 offset except at len=1; LW
            // expansion handles that alignment without loading outside the table.
            for j in 0..len {
                self.asm
                    .emit_ld(Kind::LW, temp, twiddles, (len - 1 + j) as i64 * 4);
                for start in (0..DEGREE).step_by(2 * len) {
                    let u = *self.values[start + j];
                    let v = *self.values[start + j + len];
                    self.asm.emit_r(Kind::SUBW, psi, u, v);
                    self.asm.emit_r(Kind::ADDW, u, u, v);
                    self.reduce(u);
                    self.mont_mul(v, psi, temp);
                }
            }
            len /= 2;
        }
        // Valid NTT parameters keep every stage output in (-p, p), so the
        // final pass only adds p to negative coefficients (see scalar_forward).
        for i in 0..DEGREE {
            let value = *self.values[i];
            self.asm.emit_i(Kind::SRAI, temp, value, 63);
            self.asm.emit_r(Kind::AND, temp, temp, p);
            self.asm.emit_r(Kind::ADDW, value, value, temp);
        }
        for i in (0..DEGREE).step_by(2) {
            self.asm.store_paired_u32(
                self.operands.rs1,
                i as i64 * 4,
                *self.values[i],
                *self.values[i + 1],
            );
        }
        self.asm.release_many(self.values);
        self.asm.release_many(self.scratch);
        self.asm.finalize()
    }

    fn mont_mul(&mut self, out: u8, a: u8, b: u8) {
        let [_, _, p, pinv, _, product, low, scratch] = self.scratch.map(|r| *r);
        self.asm.emit_r(Kind::MUL, product, a, b);
        self.asm.emit_r(Kind::MULW, low, product, pinv);
        self.asm.emit_r(Kind::MUL, scratch, low, p);
        self.asm.emit_r(Kind::SUB, product, product, scratch);
        self.asm.emit_i(Kind::SRAI, out, product, 32);
    }

    fn reduce(&mut self, value: u8) {
        let [_, _, p, _, _, diff, mask, _] = self.scratch.map(|r| *r);
        reduce(&mut self.asm, value, p, diff, mask);
    }
}

impl InlineOp for ForwardNtt64 {
    type Advice = NoAdvice;
    const OPCODE: u32 = OPCODE;
    const FUNCT3: u32 = FUNCT3;
    const FUNCT7: u32 = FUNCT7;
    const NAME: &'static str = "NTT64_I32_FORWARD";

    fn build_sequence(
        mut asm: InlineExpansionBuilder,
        operands: InlineOperands,
    ) -> Result<ExpandedInstructionSequence, ExpansionError> {
        let values = asm.allocate_inline_array::<DEGREE>()?;
        let scratch = asm.allocate_inline_array::<8>()?;
        NttBuilder {
            asm,
            operands,
            values,
            scratch,
        }
        .build()
    }
}

pub(super) fn reduce(asm: &mut InlineExpansionBuilder, value: u8, p: u8, diff: u8, mask: u8) {
    asm.emit_r(Kind::SUB, diff, value, p);
    asm.emit_i(Kind::SRAI, mask, diff, 63);
    asm.emit_r(Kind::AND, mask, mask, p);
    asm.emit_r(Kind::ADD, diff, diff, mask);
    asm.emit_i(Kind::SRAI, mask, diff, 63);
    asm.emit_r(Kind::AND, mask, mask, p);
    asm.emit_r(Kind::ADDW, value, diff, mask);
}
