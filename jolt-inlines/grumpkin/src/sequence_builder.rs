use std::collections::VecDeque;

use ark_ff::{BigInt, Field};
use ark_grumpkin::{Fq, Fr};
use jolt_inlines_sdk::host::{
    Cpu, FormatInline, InlineOp, InstrAssembler, InstrAssemblerExt, Instruction,
    VirtualRegisterGuard,
};
struct GrumpkinDivAdv {
    asm: InstrAssembler,
    vr: VirtualRegisterGuard, // only one register needed
    operands: FormatInline,
    is_base_field: bool, // true if base field (Fq), false if scalar field (Fr)
}

impl GrumpkinDivAdv {
    fn new(asm: InstrAssembler, operands: FormatInline, is_base_field: bool) -> Self {
        let vr = asm.allocator.allocate_for_inline();
        GrumpkinDivAdv {
            asm,
            vr,
            operands,
            is_base_field,
        }
    }
    // Custom advice function
    fn advice(self, cpu: &mut Cpu) -> VecDeque<u64> {
        // read memory directly to get inputs
        let a_addr = cpu.x[self.operands.rs1 as usize] as u64;
        let a = [
            cpu.mmu.load_doubleword(a_addr).unwrap().0,
            cpu.mmu.load_doubleword(a_addr + 8).unwrap().0,
            cpu.mmu.load_doubleword(a_addr + 16).unwrap().0,
            cpu.mmu.load_doubleword(a_addr + 24).unwrap().0,
        ];
        let b_addr = cpu.x[self.operands.rs2 as usize] as u64;
        let b = [
            cpu.mmu.load_doubleword(b_addr).unwrap().0,
            cpu.mmu.load_doubleword(b_addr + 8).unwrap().0,
            cpu.mmu.load_doubleword(b_addr + 16).unwrap().0,
            cpu.mmu.load_doubleword(b_addr + 24).unwrap().0,
        ];
        // compute c = a / b and return limbs as VecDeque
        VecDeque::from(
            if self.is_base_field {
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
    fn inline_sequence(mut self) -> Vec<Instruction> {
        self.asm.emit_advice_stores(*self.vr, self.operands.rs3, 4);
        drop(self.vr);
        self.asm.finalize_inline()
    }
}

pub struct GrumpkinDivQAdv;

impl InlineOp for GrumpkinDivQAdv {
    const OPCODE: u32 = crate::INLINE_OPCODE;
    const FUNCT3: u32 = crate::GRUMPKIN_DIVQ_ADV_FUNCT3;
    const FUNCT7: u32 = crate::GRUMPKIN_FUNCT7;
    const NAME: &'static str = crate::GRUMPKIN_DIVQ_ADV_NAME;

    fn build_sequence(asm: InstrAssembler, operands: FormatInline) -> Vec<Instruction> {
        GrumpkinDivAdv::new(asm, operands, true).inline_sequence()
    }

    fn build_advice(
        asm: InstrAssembler,
        operands: FormatInline,
        cpu: &mut Cpu,
    ) -> Option<VecDeque<u64>> {
        Some(GrumpkinDivAdv::new(asm, operands, true).advice(cpu))
    }
}

pub struct GrumpkinDivRAdv;

impl InlineOp for GrumpkinDivRAdv {
    const OPCODE: u32 = crate::INLINE_OPCODE;
    const FUNCT3: u32 = crate::GRUMPKIN_DIVR_ADV_FUNCT3;
    const FUNCT7: u32 = crate::GRUMPKIN_FUNCT7;
    const NAME: &'static str = crate::GRUMPKIN_DIVR_ADV_NAME;

    fn build_sequence(asm: InstrAssembler, operands: FormatInline) -> Vec<Instruction> {
        GrumpkinDivAdv::new(asm, operands, false).inline_sequence()
    }

    fn build_advice(
        asm: InstrAssembler,
        operands: FormatInline,
        cpu: &mut Cpu,
    ) -> Option<VecDeque<u64>> {
        Some(GrumpkinDivAdv::new(asm, operands, false).advice(cpu))
    }
}
