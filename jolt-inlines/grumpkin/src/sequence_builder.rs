use ark_ff::{BigInt, Field, PrimeField};
use ark_grumpkin::{Fq, Fr};
use jolt_inlines_sdk::host::{
    ExpandedInstructionSequence, ExpansionError, FieldElementAdvice, FormatInline,
    GlvDecompositionAdvice, InlineAdviceContext, InlineBuilderExt, InlineExpansionBuilder,
    InlineOp, InlineOperands, InlineRegister,
};
struct GrumpkinDivAdv {
    asm: InlineExpansionBuilder,
    vr: InlineRegister, // only one register needed
    operands: InlineOperands,
}

impl GrumpkinDivAdv {
    fn new(
        mut asm: InlineExpansionBuilder,
        operands: InlineOperands,
    ) -> Result<Self, ExpansionError> {
        let vr = asm.allocate_for_inline()?;
        Ok(GrumpkinDivAdv { asm, vr, operands })
    }

    fn advice(
        operands: FormatInline,
        is_base_field: bool,
        ctx: &mut dyn InlineAdviceContext,
    ) -> FieldElementAdvice {
        let a_addr = ctx.register(operands.rs1 as usize);
        let a = [
            ctx.load_doubleword(a_addr).unwrap(),
            ctx.load_doubleword(a_addr + 8).unwrap(),
            ctx.load_doubleword(a_addr + 16).unwrap(),
            ctx.load_doubleword(a_addr + 24).unwrap(),
        ];
        let b_addr = ctx.register(operands.rs2 as usize);
        let b = [
            ctx.load_doubleword(b_addr).unwrap(),
            ctx.load_doubleword(b_addr + 8).unwrap(),
            ctx.load_doubleword(b_addr + 16).unwrap(),
            ctx.load_doubleword(b_addr + 24).unwrap(),
        ];
        let limbs = if is_base_field {
            let arr_to_fq = |a: &[u64; 4]| Fq::new_unchecked(BigInt(*a));
            (arr_to_fq(&b)
                .inverse()
                .expect("Attempted to invert zero in grumpkin base field")
                * arr_to_fq(&a))
            .0
             .0
        } else {
            let arr_to_fr = |a: &[u64; 4]| Fr::new_unchecked(BigInt(*a));
            (arr_to_fr(&b)
                .inverse()
                .expect("Attempted to invert zero in grumpkin scalar field")
                * arr_to_fr(&a))
            .0
             .0
        };
        FieldElementAdvice { limbs }
    }

    fn inline_sequence(mut self) -> Result<ExpandedInstructionSequence, ExpansionError> {
        self.asm.emit_advice_stores(*self.vr, self.operands.rs3, 4);
        self.asm.release(self.vr);
        self.asm.finalize()
    }
}

struct GlvrAdvBuilder {
    asm: InlineExpansionBuilder,
    vr: InlineRegister,
    operands: InlineOperands,
}

impl GlvrAdvBuilder {
    fn new(
        mut asm: InlineExpansionBuilder,
        operands: InlineOperands,
    ) -> Result<Self, ExpansionError> {
        let vr = asm.allocate_for_inline()?;
        Ok(GlvrAdvBuilder { asm, vr, operands })
    }

    fn advice(operands: FormatInline, ctx: &mut dyn InlineAdviceContext) -> GlvDecompositionAdvice {
        let k_addr = ctx.register(operands.rs1 as usize);
        let k_limbs = [
            ctx.load_doubleword(k_addr).unwrap(),
            ctx.load_doubleword(k_addr + 8).unwrap(),
            ctx.load_doubleword(k_addr + 16).unwrap(),
            ctx.load_doubleword(k_addr + 24).unwrap(),
        ];
        let k = Fr::new_unchecked(BigInt(k_limbs)).into_bigint().into();
        GlvDecompositionAdvice::from_sign_abs(crate::glv::decompose_scalar(k))
    }

    fn inline_sequence(mut self) -> Result<ExpandedInstructionSequence, ExpansionError> {
        self.asm.emit_advice_stores(*self.vr, self.operands.rs3, 6);
        self.asm.release(self.vr);
        self.asm.finalize()
    }
}

pub struct GrumpkinDivQAdv;

impl InlineOp for GrumpkinDivQAdv {
    type Advice = FieldElementAdvice;

    const OPCODE: u32 = crate::INLINE_OPCODE;
    const FUNCT3: u32 = crate::GRUMPKIN_DIVQ_ADV_FUNCT3;
    const FUNCT7: u32 = crate::GRUMPKIN_FUNCT7;
    const NAME: &'static str = crate::GRUMPKIN_DIVQ_ADV_NAME;

    fn build_sequence(
        asm: InlineExpansionBuilder,
        operands: InlineOperands,
    ) -> Result<ExpandedInstructionSequence, ExpansionError> {
        GrumpkinDivAdv::new(asm, operands)?.inline_sequence()
    }

    fn build_advice(operands: FormatInline, ctx: &mut dyn InlineAdviceContext) -> Self::Advice {
        GrumpkinDivAdv::advice(operands, true, ctx)
    }
}

pub struct GrumpkinDivRAdv;

impl InlineOp for GrumpkinDivRAdv {
    type Advice = FieldElementAdvice;

    const OPCODE: u32 = crate::INLINE_OPCODE;
    const FUNCT3: u32 = crate::GRUMPKIN_DIVR_ADV_FUNCT3;
    const FUNCT7: u32 = crate::GRUMPKIN_FUNCT7;
    const NAME: &'static str = crate::GRUMPKIN_DIVR_ADV_NAME;

    fn build_sequence(
        asm: InlineExpansionBuilder,
        operands: InlineOperands,
    ) -> Result<ExpandedInstructionSequence, ExpansionError> {
        GrumpkinDivAdv::new(asm, operands)?.inline_sequence()
    }

    fn build_advice(operands: FormatInline, ctx: &mut dyn InlineAdviceContext) -> Self::Advice {
        GrumpkinDivAdv::advice(operands, false, ctx)
    }
}

pub struct GrumpkinGlvrAdv;

impl InlineOp for GrumpkinGlvrAdv {
    type Advice = GlvDecompositionAdvice;

    const OPCODE: u32 = crate::INLINE_OPCODE;
    const FUNCT3: u32 = crate::GRUMPKIN_GLVR_ADV_FUNCT3;
    const FUNCT7: u32 = crate::GRUMPKIN_FUNCT7;
    const NAME: &'static str = crate::GRUMPKIN_GLVR_ADV_NAME;

    fn build_sequence(
        asm: InlineExpansionBuilder,
        operands: InlineOperands,
    ) -> Result<ExpandedInstructionSequence, ExpansionError> {
        GlvrAdvBuilder::new(asm, operands)?.inline_sequence()
    }

    fn build_advice(operands: FormatInline, ctx: &mut dyn InlineAdviceContext) -> Self::Advice {
        GlvrAdvBuilder::advice(operands, ctx)
    }
}
