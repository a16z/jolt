use std::collections::VecDeque;

use ark_ff::{BigInt, BigInteger, Field, PrimeField};
use ark_grumpkin::{Fq, Fr};
use num_bigint::BigInt as NBigInt;
use num_bigint::Sign;
use num_integer::Integer;
use tracer::{
    emulator::cpu::Cpu,
    instruction::{
        format::format_inline::FormatInline, sd::SD, virtual_advice::VirtualAdvice, Instruction,
    },
    utils::{inline_helpers::InstrAssembler, virtual_registers::VirtualRegisterGuard},
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
        for i in 0..4 {
            self.asm.emit_j::<VirtualAdvice>(*self.vr, 0);
            self.asm
                .emit_s::<SD>(self.operands.rs3, *self.vr, i as i64 * 8);
        }
        drop(self.vr);
        self.asm.finalize_inline()
    }
}

/// Virtual instruction builder for unchecked grumpkin base field modular division
pub fn grumpkin_divq_adv_sequence_builder(
    asm: InstrAssembler,
    operands: FormatInline,
) -> Vec<Instruction> {
    let builder = GrumpkinDivAdv::new(asm, operands, true);
    builder.inline_sequence()
}

/// Custom trace function for unchecked grumpkin base field modular division
pub fn grumpkin_divq_adv_advice(
    asm: InstrAssembler,
    operands: FormatInline,
    cpu: &mut Cpu,
) -> VecDeque<u64> {
    let builder = GrumpkinDivAdv::new(asm, operands, true);
    builder.advice(cpu)
}

/// Virtual instruction builder for unchecked grumpkin scalar field modular division
pub fn grumpkin_divr_adv_sequence_builder(
    asm: InstrAssembler,
    operands: FormatInline,
) -> Vec<Instruction> {
    let builder = GrumpkinDivAdv::new(asm, operands, false);
    builder.inline_sequence()
}

/// Custom trace function for unchecked grumpkin scalar field modular division
pub fn grumpkin_divr_adv_advice(
    asm: InstrAssembler,
    operands: FormatInline,
    cpu: &mut Cpu,
) -> VecDeque<u64> {
    let builder = GrumpkinDivAdv::new(asm, operands, false);
    builder.advice(cpu)
}

struct GrumpkinGlvrAdv {
    asm: InstrAssembler,
    vr: VirtualRegisterGuard, // only one register needed
    operands: FormatInline,
}

impl GrumpkinGlvrAdv {
    fn new(asm: InstrAssembler, operands: FormatInline) -> Self {
        let vr = asm.allocator.allocate_for_inline();
        GrumpkinGlvrAdv { asm, vr, operands }
    }
    // Custom advice function
    // heavily based on the implementation found in ec/src/scalar_mul/glv.rs in arkworks
    fn advice(self, cpu: &mut Cpu) -> VecDeque<u64> {
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
        let r = NBigInt::from_bytes_le(Sign::Plus, &Fr::MODULUS.to_bytes_le());
        let n11 = NBigInt::from(147946756881789319000765030803803410729i128);
        let n12 = NBigInt::from(-9931322734385697762i128);
        let n21 = NBigInt::from(9931322734385697762i128);
        let n22 = NBigInt::from(147946756881789319010696353538189108491i128);
        let beta_1 = {
            let (mut div, rem) = (&k * &n22).div_rem(&r);
            if (&rem + &rem) > r {
                div += NBigInt::from(1u8);
            }
            div
        };
        let beta_2 = {
            let n12_neg = -n12.clone();
            let (mut div, rem) = (&k * &n12_neg).div_rem(&r);
            if (&rem + &rem) > r {
                div += NBigInt::from(1u8);
            }
            div
        };
        let k1 = &k - &beta_1 * &n11 - &beta_2 * &n21;
        let k2 = -(&beta_1 * &n12 + &beta_2 * &n22);
        // convert k1, k2 to absolute values and signs
        let serialize_k = |k: NBigInt| -> (u64, [u64; 2]) {
            let sign = if k.sign() == Sign::Minus { 1u64 } else { 0u64 };
            let abs_k = if sign == 1 { -k } else { k };
            let bytes = abs_k.to_bytes_le().1;
            assert!(
                bytes.len() <= 16,
                "GLV decomposition produced out-of-range half-scalar"
            );
            let mut arr = [0u64; 2];
            for i in 0..bytes.len() {
                arr[i / 8] |= (bytes[i] as u64) << ((i % 8) * 8);
            }
            (sign, arr)
        };
        let (s1, k1_arr) = serialize_k(k1);
        let (s2, k2_arr) = serialize_k(k2);
        let advice = vec![s1, k1_arr[0], k1_arr[1], s2, k2_arr[0], k2_arr[1]];
        VecDeque::from(advice)
    }
    // inline sequence function
    fn inline_sequence(mut self) -> Vec<Instruction> {
        for i in 0..6 {
            self.asm.emit_j::<VirtualAdvice>(*self.vr, 0);
            self.asm
                .emit_s::<SD>(self.operands.rs3, *self.vr, i as i64 * 8);
        }
        drop(self.vr);
        self.asm.finalize_inline()
    }
}

/// Virtual instruction builder for unchecked grumpkin GLV decomposition
pub fn grumpkin_glvr_adv_sequence_builder(
    asm: InstrAssembler,
    operands: FormatInline,
) -> Vec<Instruction> {
    let builder = GrumpkinGlvrAdv::new(asm, operands);
    builder.inline_sequence()
}

/// Custom trace function for unchecked grumpkin GLV decomposition
pub fn grumpkin_glvr_adv_advice(
    asm: InstrAssembler,
    operands: FormatInline,
    cpu: &mut Cpu,
) -> VecDeque<u64> {
    let builder = GrumpkinGlvrAdv::new(asm, operands);
    builder.advice(cpu)
}
