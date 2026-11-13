use jolt_inlines_sdk::host::{
    ExpandedInstructionSequence, ExpansionError, InlineBuilderExt, InlineExpansionBuilder,
    InlineOp, InlineOperands, InlineRegister, NoAdvice, SourceKind as Kind,
};

use crate::{DEGREE, FUNCT3, FUNCT7, OPCODE};

/// Selector 8 requires doubleword-aligned state, twist and stage-table arrays.
/// Paired stage-table loads require 8-byte alignment even when the instruction
/// is emitted directly; `forward_ntt64` supplies aligned copies when needed.
/// Its expansion differs from the scalar-load schedule, so existing guest
/// ELFs require traces and preprocessing generated with this expansion.
pub struct ForwardNtt64;

struct NttBuilder {
    asm: InlineExpansionBuilder,
    operands: InlineOperands,
    values: [InlineRegister; DEGREE],
    scratch: [InlineRegister; 10],
}

impl NttBuilder {
    fn new(
        mut asm: InlineExpansionBuilder,
        operands: InlineOperands,
    ) -> Result<Self, ExpansionError> {
        let values = asm.allocate_inline_array::<DEGREE>()?;
        let scratch = asm.allocate_inline_array::<10>()?;
        Ok(Self {
            asm,
            operands,
            values,
            scratch,
        })
    }

    fn build(mut self) -> Result<ExpandedInstructionSequence, ExpansionError> {
        let [psi, twiddles, p, pinv, temp, _, _, _, weight, one] = self.scratch.map(|r| *r);
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
        // every stage, using 74 of the 80 inline registers.
        for i in (0..DEGREE).step_by(2) {
            self.asm.emit_ld(Kind::LD, temp, psi, i as i64 * 4);
            self.asm.emit_i(Kind::ADDIW, weight, temp, 0);
            self.mont_mul(*self.values[i], *self.values[i], weight);
            self.asm.emit_i(Kind::SRAI, temp, temp, 32);
            self.mont_mul(*self.values[i + 1], *self.values[i + 1], temp);
        }
        // Every stage starts with the Montgomery identity weight. A paired
        // load supplies it without an unaligned word-load expansion.
        self.asm.emit_ld(Kind::LD, one, twiddles, 0);
        self.asm.emit_i(Kind::ADDIW, one, one, 0);
        let mut len = DEGREE / 2;
        while len != 0 {
            let groups = len;
            for group in 0..groups {
                let twiddle = if len == 1 {
                    one
                } else {
                    let index = len - 1 + group;
                    // forward_ntt64 establishes doubleword alignment. Each
                    // pair stays inside the table; mont_mul preserves weight.
                    if group == 0 || index & 1 == 0 {
                        self.asm
                            .emit_ld(Kind::LD, weight, twiddles, (index & !1) as i64 * 4);
                    }
                    if index & 1 == 0 {
                        self.asm.emit_i(Kind::ADDIW, temp, weight, 0);
                    } else {
                        self.asm.emit_i(Kind::SRAI, temp, weight, 32);
                    }
                    temp
                };
                let pairs = DEGREE / (2 * len);
                for pair in 0..pairs {
                    let (start, j) = (pair * 2 * len, group);
                    let u = *self.values[start + j];
                    let v = *self.values[start + j + len];
                    self.asm.emit_r(Kind::SUBW, psi, u, v);
                    self.asm.emit_r(Kind::ADDW, u, u, v);
                    self.mont_mul(u, u, one);
                    self.mont_mul(v, psi, twiddle);
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
        let [_, _, p, pinv, _, product, low, scratch, _, _] = self.scratch.map(|r| *r);
        self.asm.emit_r(Kind::MUL, product, a, b);
        self.asm.emit_r(Kind::MULW, low, product, pinv);
        self.asm.emit_r(Kind::MUL, scratch, low, p);
        self.asm.emit_r(Kind::SUB, product, product, scratch);
        self.asm.emit_i(Kind::SRAI, out, product, 32);
    }
}

impl InlineOp for ForwardNtt64 {
    type Advice = NoAdvice;
    const OPCODE: u32 = OPCODE;
    const FUNCT3: u32 = FUNCT3;
    const FUNCT7: u32 = FUNCT7;
    const NAME: &'static str = "NTT64_I32_FORWARD";

    fn build_sequence(
        asm: InlineExpansionBuilder,
        operands: InlineOperands,
    ) -> Result<ExpandedInstructionSequence, ExpansionError> {
        NttBuilder::new(asm, operands)?.build()
    }
}
