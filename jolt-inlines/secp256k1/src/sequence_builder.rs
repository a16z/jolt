use ark_ff::{BigInt, Field};
use ark_secp256k1::Fq;
use tracer::{
    emulator::cpu::Cpu,
    instruction::{
        format::format_inline::FormatInline, sd::SD, virtual_advice::VirtualAdvice, Cycle,
        Instruction,
    },
    utils::{inline_helpers::InstrAssembler, virtual_registers::VirtualRegisterGuard},
};

// helper functions for custom traces

// wrapper to make it easier to create Fq from [u64; 4]
fn arr_to_fq(a: &[u64; 4]) -> Fq {
    Fq::new_unchecked(BigInt { 0: *a })
}

// inline functions

struct Secp256k1DivqAdv {
    asm: InstrAssembler,
    vr: VirtualRegisterGuard, // only one register needed
    operands: FormatInline,
}

impl Secp256k1DivqAdv {
    fn new(asm: InstrAssembler, operands: FormatInline) -> Self {
        let vr = asm.allocator.allocate_for_inline();
        Secp256k1DivqAdv { asm, vr, operands }
    }
    // custom trace function
    fn trace(self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
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
        // compute c = a / b
        let advice = (arr_to_fq(&b)
            .inverse()
            .expect("Attempted to invert zero in secp256k1 field")
            * arr_to_fq(&a))
        .0
         .0;
        // maintain counter for advice words
        let mut advice_counter = 0;
        // set up trace
        let mut trace = trace;
        let mut inline_sequence = self.inline_sequence();
        // execute remaining instructions, injecting advice where needed
        for instr in inline_sequence.iter_mut() {
            if let Instruction::VirtualAdvice(va) = instr {
                va.advice = advice[advice_counter];
                advice_counter += 1;
            }
            instr.trace(cpu, trace.as_deref_mut());
        }
    }
    // inline sequence function
    fn inline_sequence(mut self) -> Vec<Instruction> {
        for i in 0..4 {
            self.asm.emit_j::<VirtualAdvice>(*self.vr, 0);
            self.asm
                .emit_s::<SD>(self.operands.rs3, *self.vr, i as i64 * 8);
        }
        drop(self.vr);
        self.asm.finalize_inline()
    }
}

/// Virtual instruction builder for unchecked secp256k1 base field modular division
pub fn secp256k1_divq_adv_sequence_builder(
    asm: InstrAssembler,
    operands: FormatInline,
) -> Vec<Instruction> {
    let builder = Secp256k1DivqAdv::new(asm, operands);
    builder.inline_sequence()
}

/// Custom trace function for unchecked secp256k1 base field modular division
pub fn secp256k1_divq_adv_custom_trace(
    asm: InstrAssembler,
    operands: FormatInline,
    cpu: &mut Cpu,
    trace: Option<&mut Vec<Cycle>>,
) {
    let builder = Secp256k1DivqAdv::new(asm, operands);
    builder.trace(cpu, trace);
}
