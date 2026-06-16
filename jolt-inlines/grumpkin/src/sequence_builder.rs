use std::collections::VecDeque;

use ark_ff::{BigInt, Field, PrimeField};
use ark_grumpkin::{Fq, Fr};
use jolt_inlines_sdk::host::{
    Cpu, ExpandedInstructionSequence, ExpansionError, FormatInline, InlineBuilderExt,
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

    // Custom advice function
    fn advice(operands: FormatInline, is_base_field: bool, cpu: &mut Cpu) -> VecDeque<u64> {
        // read memory directly to get inputs
        let a_addr = cpu.x[operands.rs1 as usize] as u64;
        let a = [
            cpu.mmu.load_doubleword(a_addr).unwrap().0,
            cpu.mmu.load_doubleword(a_addr + 8).unwrap().0,
            cpu.mmu.load_doubleword(a_addr + 16).unwrap().0,
            cpu.mmu.load_doubleword(a_addr + 24).unwrap().0,
        ];
        let b_addr = cpu.x[operands.rs2 as usize] as u64;
        let b = [
            cpu.mmu.load_doubleword(b_addr).unwrap().0,
            cpu.mmu.load_doubleword(b_addr + 8).unwrap().0,
            cpu.mmu.load_doubleword(b_addr + 16).unwrap().0,
            cpu.mmu.load_doubleword(b_addr + 24).unwrap().0,
        ];
        // compute c = a / b and return limbs as VecDeque
        VecDeque::from(
            if is_base_field {
                let arr_to_fq = |a: &[u64; 4]| Fq::new_unchecked(BigInt(*a));
                (arr_to_fq(&b)
                    .inverse()
                    .expect("Attempted to invert zero in grumpkin base field")
                    * arr_to_fq(&a))
                .0
            } else {
                let arr_to_fr = |a: &[u64; 4]| Fr::new_unchecked(BigInt(*a));
                (arr_to_fr(&b)
                    .inverse()
                    .expect("Attempted to invert zero in grumpkin scalar field")
                    * arr_to_fr(&a))
                .0
            }
            .0
            .to_vec(),
        )
    }
    // inline sequence function
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

    fn advice(operands: FormatInline, cpu: &mut Cpu) -> VecDeque<u64> {
        let k_addr = cpu.x[operands.rs1 as usize] as u64;
        let k_limbs = [
            cpu.mmu.load_doubleword(k_addr).unwrap().0,
            cpu.mmu.load_doubleword(k_addr + 8).unwrap().0,
            cpu.mmu.load_doubleword(k_addr + 16).unwrap().0,
            cpu.mmu.load_doubleword(k_addr + 24).unwrap().0,
        ];
        let k = Fr::new_unchecked(BigInt(k_limbs)).into_bigint().into();
        let result = crate::glv::decompose_scalar_to_u64s(k);
        VecDeque::from(result.to_vec())
    }

    fn inline_sequence(mut self) -> Result<ExpandedInstructionSequence, ExpansionError> {
        self.asm.emit_advice_stores(*self.vr, self.operands.rs3, 6);
        self.asm.release(self.vr);
        self.asm.finalize()
    }
}

pub struct GrumpkinDivQAdv;

impl InlineOp for GrumpkinDivQAdv {
    type Advice = VecDeque<u64>;

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

    fn build_advice(operands: FormatInline, cpu: &mut Cpu) -> Self::Advice {
        GrumpkinDivAdv::advice(operands, true, cpu)
    }
}

pub struct GrumpkinDivRAdv;

impl InlineOp for GrumpkinDivRAdv {
    type Advice = VecDeque<u64>;

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

    fn build_advice(operands: FormatInline, cpu: &mut Cpu) -> Self::Advice {
        GrumpkinDivAdv::advice(operands, false, cpu)
    }
}

pub struct GrumpkinGlvrAdv;

impl InlineOp for GrumpkinGlvrAdv {
    type Advice = VecDeque<u64>;

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

    fn build_advice(operands: FormatInline, cpu: &mut Cpu) -> Self::Advice {
        GlvrAdvBuilder::advice(operands, cpu)
    }
}
