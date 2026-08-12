use ark_ff::{BigInt, Field, PrimeField};
use ark_grumpkin::{Fq, Fr};
use jolt_inlines_sdk::host::{
    load_field_element_limbs, ExpandedInstructionSequence, ExpansionError, FieldElementAdvice,
    FormatInline, GlvDecompositionAdvice, InlineAdviceContext, InlineAdviceError, InlineBuilderExt,
    InlineExpansionBuilder, InlineOp, InlineOperands, InlineRegister,
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
    ) -> Result<FieldElementAdvice, InlineAdviceError> {
        let a_addr = ctx.register(operands.rs1 as usize);
        let a = load_field_element_limbs(ctx, a_addr)?;
        let b_addr = ctx.register(operands.rs2 as usize);
        let b = load_field_element_limbs(ctx, b_addr)?;
        // A zero divisor has no inverse, so no advice can satisfy `b * c == a`
        // for `a != 0`. Emit `c = 0` instead of aborting trace generation: the
        // guest-side check in `div_assume_nonzero` then spoils the proof, which
        // is the contract a guest calling `div` with a zero divisor expects.
        let limbs = if is_base_field {
            let arr_to_fq = |a: &[u64; 4]| Fq::new_unchecked(BigInt(*a));
            arr_to_fq(&b)
                .inverse()
                .map_or([0u64; 4], |b_inv| (b_inv * arr_to_fq(&a)).0 .0)
        } else {
            let arr_to_fr = |a: &[u64; 4]| Fr::new_unchecked(BigInt(*a));
            arr_to_fr(&b)
                .inverse()
                .map_or([0u64; 4], |b_inv| (b_inv * arr_to_fr(&a)).0 .0)
        };
        Ok(FieldElementAdvice { limbs })
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

    fn advice(
        operands: FormatInline,
        ctx: &mut dyn InlineAdviceContext,
    ) -> Result<GlvDecompositionAdvice, InlineAdviceError> {
        let k_addr = ctx.register(operands.rs1 as usize);
        let k_limbs = load_field_element_limbs(ctx, k_addr)?;
        let k = Fr::new_unchecked(BigInt(k_limbs)).into_bigint().into();
        Ok(GlvDecompositionAdvice::from_sign_abs(
            crate::glv::decompose_scalar(k),
        ))
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

    fn build_advice(
        operands: FormatInline,
        ctx: &mut dyn InlineAdviceContext,
    ) -> Result<Self::Advice, InlineAdviceError> {
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

    fn build_advice(
        operands: FormatInline,
        ctx: &mut dyn InlineAdviceContext,
    ) -> Result<Self::Advice, InlineAdviceError> {
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

    fn build_advice(
        operands: FormatInline,
        ctx: &mut dyn InlineAdviceContext,
    ) -> Result<Self::Advice, InlineAdviceError> {
        GlvrAdvBuilder::advice(operands, ctx)
    }
}
