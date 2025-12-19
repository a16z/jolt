use ark_ff::{BigInt, Field, PrimeField};
use ark_secp256k1::{Fq, Fr};
use num_bigint::BigInt as NBigInt;
use num_bigint::Sign;
use num_integer::Integer;
use tracer::{
    emulator::cpu::Cpu,
    instruction::{
        format::format_inline::FormatInline, sd::SD, virtual_advice::VirtualAdvice, Cycle,
        Instruction,
    },
    utils::{inline_helpers::InstrAssembler, virtual_registers::VirtualRegisterGuard},
};

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
        let arr_to_fq = |a: &[u64; 4]| Fq::new_unchecked(BigInt(*a));
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

struct Secp256k1DivrAdv {
    asm: InstrAssembler,
    vr: VirtualRegisterGuard, // only one register needed
    operands: FormatInline,
}

impl Secp256k1DivrAdv {
    fn new(asm: InstrAssembler, operands: FormatInline) -> Self {
        let vr = asm.allocator.allocate_for_inline();
        Secp256k1DivrAdv { asm, vr, operands }
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
        let arr_to_fr = |a: &[u64; 4]| Fr::new_unchecked(BigInt(*a));
        let advice = (arr_to_fr(&b)
            .inverse()
            .expect("Attempted to invert zero in secp256k1 scalar field")
            * arr_to_fr(&a))
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
pub fn secp256k1_divr_adv_sequence_builder(
    asm: InstrAssembler,
    operands: FormatInline,
) -> Vec<Instruction> {
    let builder = Secp256k1DivrAdv::new(asm, operands);
    builder.inline_sequence()
}

/// Custom trace function for unchecked secp256k1 base field modular division
pub fn secp256k1_divr_adv_custom_trace(
    asm: InstrAssembler,
    operands: FormatInline,
    cpu: &mut Cpu,
    trace: Option<&mut Vec<Cycle>>,
) {
    let builder = Secp256k1DivrAdv::new(asm, operands);
    builder.trace(cpu, trace);
}

struct Secp256k1GlvrAdv {
    asm: InstrAssembler,
    vr: VirtualRegisterGuard, // only one register needed
    operands: FormatInline,
}

impl Secp256k1GlvrAdv {
    fn new(asm: InstrAssembler, operands: FormatInline) -> Self {
        let vr = asm.allocator.allocate_for_inline();
        Secp256k1GlvrAdv { asm, vr, operands }
    }
    // custom trace function
    // heavily based on the implementation found in ec/src/scalar_mul/glv.rs in arkworks
    fn trace(self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // read memory directly to get inputs
        let k_addr = cpu.x[self.operands.rs1 as usize] as u64;
        let kr = [
            cpu.mmu.load_doubleword(k_addr).unwrap().0,
            cpu.mmu.load_doubleword(k_addr + 8).unwrap().0,
            cpu.mmu.load_doubleword(k_addr + 16).unwrap().0,
            cpu.mmu.load_doubleword(k_addr + 24).unwrap().0,
        ];
        // convert k from montgomery form to normal form
        let k: NBigInt = Fr::new_unchecked(BigInt(kr)).into_bigint().into();
        // constants for glv decomposition
        let r = NBigInt::from_bytes_le(
            Sign::Plus,
            &[
                65, 65, 54, 208, 140, 94, 210, 191, 59, 160, 72, 175, 230, 220, 174, 186, 254, 255,
                255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255, 255,
            ],
        );
        let a1 = NBigInt::from_bytes_le(
            Sign::Plus,
            &[
                21, 235, 132, 146, 228, 144, 108, 232, 205, 107, 212, 167, 33, 210, 134, 48,
            ],
        );
        let b1 = NBigInt::from_bytes_le(
            Sign::Plus,
            &[
                195, 228, 191, 10, 169, 127, 84, 111, 40, 136, 14, 1, 214, 126, 67, 228,
            ],
        );
        let a2 = NBigInt::from_bytes_le(
            Sign::Plus,
            &[
                216, 207, 68, 157, 141, 16, 193, 87, 246, 243, 226, 168, 247, 80, 202, 20, 1,
            ],
        );
        let beta_1 = {
            let (mut div, rem) = (&k * &a1).div_rem(&r);
            if (&rem + &rem) > r {
                div += NBigInt::from_bytes_le(Sign::Plus, &[1u8]);
            }
            div
        };
        let beta_2 = {
            let (mut div, rem) = (&k * &b1).div_rem(&r);
            if (&rem + &rem) > r {
                div += NBigInt::from_bytes_le(Sign::Plus, &[1u8]);
            }
            div
        };
        let k1 = &k - &beta_1 * &a1 - &beta_2 * &a2;
        let k2 = &beta_1 * &b1 - &beta_2 * &a1;
        // convert k1, k2 to absolute values and signs
        let serialize_k = |k: NBigInt| -> (u64, [u64; 2]) {
            let sign = if k.sign() == Sign::Minus { 1u64 } else { 0u64 };
            let abs_k = if sign == 1 { -k } else { k };
            let bytes = abs_k.to_bytes_le().1;
            let mut arr = [0u64; 2];
            for i in 0..bytes.len() {
                arr[i / 8] |= (bytes[i] as u64) << ((i % 8) * 8);
            }
            (sign, arr)
        };
        let (s1, k1_arr) = serialize_k(k1);
        let (s2, k2_arr) = serialize_k(k2);
        let advice = [s1, k1_arr[0], k1_arr[1], s2, k2_arr[0], k2_arr[1]];
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
        for i in 0..6 {
            self.asm.emit_j::<VirtualAdvice>(*self.vr, 0);
            self.asm
                .emit_s::<SD>(self.operands.rs2, *self.vr, i as i64 * 8);
        }
        drop(self.vr);
        self.asm.finalize_inline()
    }
}

/// Virtual instruction builder for unchecked secp256k1 base field modular division
pub fn secp256k1_glvr_adv_sequence_builder(
    asm: InstrAssembler,
    operands: FormatInline,
) -> Vec<Instruction> {
    let builder = Secp256k1GlvrAdv::new(asm, operands);
    builder.inline_sequence()
}

/// Custom trace function for unchecked secp256k1 base field modular division
pub fn secp256k1_glvr_adv_custom_trace(
    asm: InstrAssembler,
    operands: FormatInline,
    cpu: &mut Cpu,
    trace: Option<&mut Vec<Cycle>>,
) {
    let builder = Secp256k1GlvrAdv::new(asm, operands);
    builder.trace(cpu, trace);
}
