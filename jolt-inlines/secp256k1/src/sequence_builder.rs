use std::array;
use std::collections::VecDeque;

use ark_ff::{BigInt, Field, PrimeField};
use ark_secp256k1::{Fq, Fr};
use num_bigint::BigInt as NBigInt;
use num_bigint::BigUint as NBigUint;
use num_bigint::Sign;
use num_integer::Integer;
use tracer::instruction::virtual_assert_lte::VirtualAssertLTE;
use tracer::{
    emulator::cpu::Cpu,
    instruction::{
        add::ADD, format::format_inline::FormatInline, ld::LD, lui::LUI, mul::MUL, mulhu::MULHU,
        sd::SD, sltu::SLTU, virtual_advice::VirtualAdvice, virtual_assert_eq::VirtualAssertEQ,
        Instruction,
    },
    utils::{inline_helpers::InstrAssembler, virtual_registers::VirtualRegisterGuard},
};
struct Secp256k1DivAdv {
    asm: InstrAssembler,
    vr: VirtualRegisterGuard, // only one register needed
    operands: FormatInline,
    is_base_field: bool, // true if base field (Fq), false if scalar field (Fr)
}

impl Secp256k1DivAdv {
    fn new(asm: InstrAssembler, operands: FormatInline, is_base_field: bool) -> Self {
        let vr = asm.allocator.allocate_for_inline();
        Secp256k1DivAdv {
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
                let arr_to_fq = |a: &[u64; 4]| Fq::new(BigInt(*a));
                (arr_to_fq(&b)
                    .inverse()
                    .expect("Attempted to invert zero in secp256k1 base field")
                    * arr_to_fq(&a))
                .into_bigint()
            } else {
                let arr_to_fr = |a: &[u64; 4]| Fr::new_unchecked(BigInt(*a));
                (arr_to_fr(&b)
                    .inverse()
                    .expect("Attempted to invert zero in secp256k1 scalar field")
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

/// Virtual instruction builder for unchecked secp256k1 base field modular division
pub fn secp256k1_divq_adv_sequence_builder(
    asm: InstrAssembler,
    operands: FormatInline,
) -> Vec<Instruction> {
    let builder = Secp256k1DivAdv::new(asm, operands, true);
    builder.inline_sequence()
}

/// Custom trace function for unchecked secp256k1 base field modular division
pub fn secp256k1_divq_adv_advice(
    asm: InstrAssembler,
    operands: FormatInline,
    cpu: &mut Cpu,
) -> VecDeque<u64> {
    let builder = Secp256k1DivAdv::new(asm, operands, true);
    builder.advice(cpu)
}

/// Virtual instruction builder for unchecked secp256k1 scalar field modular division
pub fn secp256k1_divr_adv_sequence_builder(
    asm: InstrAssembler,
    operands: FormatInline,
) -> Vec<Instruction> {
    let builder = Secp256k1DivAdv::new(asm, operands, false);
    builder.inline_sequence()
}

/// Custom trace function for unchecked secp256k1 scalar field modular division
pub fn secp256k1_divr_adv_advice(
    asm: InstrAssembler,
    operands: FormatInline,
    cpu: &mut Cpu,
) -> VecDeque<u64> {
    let builder = Secp256k1DivAdv::new(asm, operands, false);
    builder.advice(cpu)
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

/// Virtual instruction builder for unchecked secp256k1 GLV decomposition
pub fn secp256k1_glvr_adv_sequence_builder(
    asm: InstrAssembler,
    operands: FormatInline,
) -> Vec<Instruction> {
    let builder = Secp256k1GlvrAdv::new(asm, operands);
    builder.inline_sequence()
}

/// Custom trace function for unchecked secp256k1 GLV decomposition
pub fn secp256k1_glvr_adv_advice(
    asm: InstrAssembler,
    operands: FormatInline,
    cpu: &mut Cpu,
) -> VecDeque<u64> {
    let builder = Secp256k1GlvrAdv::new(asm, operands);
    builder.advice(cpu)
}

// helper function to convert from vector of u64 limbs to NBigUint
fn limbs_to_nbiguint(limbs: &[u64]) -> NBigUint {
    let mut bytes = Vec::with_capacity(limbs.len() * 8);
    for &limb in limbs {
        for i in 0..8 {
            bytes.push(((limb >> (i * 8)) & 0xFF) as u8);
        }
    }
    NBigUint::from_bytes_le(&bytes)
}

// helper function to convert from NBigUint to vector of u64 limbs
fn nbiguint_to_limbs(n: &NBigUint) -> Vec<u64> {
    let bytes = n.to_bytes_le();
    let mut limbs = vec![0u64; (bytes.len() + 7) / 8];
    for (i, byte) in bytes.iter().enumerate() {
        limbs[i / 8] |= (*byte as u64) << ((i % 8) * 8);
    }
    limbs
}

// unnamed inline for secp256k1 base field multiplication helper
struct Secp256k1Unnamed {
    asm: InstrAssembler,
    vr: VirtualRegisterGuard, // only one register needed
    operands: FormatInline,
}

impl Secp256k1Unnamed {
    fn new(asm: InstrAssembler, operands: FormatInline) -> Self {
        let vr = asm.allocator.allocate_for_inline();
        Secp256k1Unnamed { asm, vr, operands }
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
        // convert inputs to bigints
        let a_big: NBigUint = limbs_to_nbiguint(&a);
        let b_big: NBigUint = limbs_to_nbiguint(&b);
        let q_big: NBigUint = Fq::MODULUS.into();
        // compute floor(a * b / q)
        let quotient = (a_big * b_big).div_floor(&q_big);
        // convert back to limbs
        let limbs = nbiguint_to_limbs(&quotient);
        /*println!("a: {:?}", a);
        println!("b: {:?}", b);
        println!("w: {:?}", limbs);*/
        // assert that limbs fits in 4 u64s
        assert!(limbs.len() <= 4, "Result does not fit in 4 limbs");
        // pad limbs to 4 u64s and return as VecDeque
        let mut padded_limbs = vec![0u64; 4];
        for i in 0..limbs.len() {
            padded_limbs[i] = limbs[i];
        }
        VecDeque::from(padded_limbs)
        // shortcut code below. TO DO, remove
        // simply computes a*b in fq and outputs the quotient limb-wise
        //VecDeque::from((Fq::new(BigInt(a)) * Fq::new(BigInt(b))).into_bigint().0)
    }
    // inline sequence function
    fn inline_sequence(mut self) -> Vec<Instruction> {
        // get advice and store to rs3
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
pub fn secp256k1_unnamed_sequence_builder(
    asm: InstrAssembler,
    operands: FormatInline,
) -> Vec<Instruction> {
    let builder = Secp256k1Unnamed::new(asm, operands);
    builder.inline_sequence()
}

/// Custom trace function for unchecked secp256k1 base field modular division
pub fn secp256k1_unnamed_advice(
    asm: InstrAssembler,
    operands: FormatInline,
    cpu: &mut Cpu,
) -> VecDeque<u64> {
    let builder = Secp256k1Unnamed::new(asm, operands);
    builder.advice(cpu)
}

enum MulqType {
    Mul,
    Square,
    Div,
}

// inline for secp256k1 base field multiplication/squaring/division
// does not handle checking that the result is canonical mod q,
// merely that it is correct and fits in 4 limbs
// multiplication followed by modulus can be represented as: ab = wq + c
// for some w and c in [0, q) where w is provided as advice
// here we do not explicitly check that c is in [0, q), but rather that c fits in 256 bits
// the fastest possible check uses branching and thus appears after invocations of this inline
// let p = (2^256 - q), the equation above can be rearranged to: ab + wp = 2^256 w + c
// Thus, for multiplication, squaring and division, this inline checks the following:
// For multiplication, this inline checks that ab + wp =  2^256 w + c
// For division, this inline checks that cb + wp = 2^256 w + a
// For squaring, this inline checks that a^2 + wp = 2^256 w + c
// The core structure is the same for all three operations, with only minor differences in loading inputs and storing outputs
// To minimize the use of virtual registers, the LHS is never fully constructed.
// Rather, the implementation considers 2 limbs at a time, one that is the main focus
// And one that accumulates carries. The algorithm ping-pongs between these two limbs as it computes the LHS.
struct Secp256k1Mulq {
    asm: InstrAssembler,
    a: [VirtualRegisterGuard; 4],
    b: Option<[VirtualRegisterGuard; 4]>, // only allocated if Mul or Div
    w: [VirtualRegisterGuard; 4],
    p: VirtualRegisterGuard,
    aux: VirtualRegisterGuard,
    aux2: Option<VirtualRegisterGuard>, // only allocated if Square
    r: [VirtualRegisterGuard; 2],
    operands: FormatInline,
    op_type: MulqType,
}

impl Secp256k1Mulq {
    fn new(asm: InstrAssembler, operands: FormatInline, op_type: MulqType) -> Self {
        let a = array::from_fn(|_| asm.allocator.allocate_for_inline());
        let b = match op_type {
            MulqType::Square => None,
            _ => Some(array::from_fn(|_| asm.allocator.allocate_for_inline())),
        };
        let w = array::from_fn(|_| asm.allocator.allocate_for_inline());
        let p = asm.allocator.allocate_for_inline();
        let aux = asm.allocator.allocate_for_inline();
        let aux2 = match op_type {
            MulqType::Square => Some(asm.allocator.allocate_for_inline()),
            _ => None,
        };
        let r = array::from_fn(|_| asm.allocator.allocate_for_inline());
        Secp256k1Mulq {
            asm,
            a,
            b,
            w,
            p,
            aux,
            aux2,
            r,
            operands,
            op_type,
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
        let b_addr = match self.op_type {
            MulqType::Square => a_addr,
            _ => cpu.x[self.operands.rs2 as usize] as u64,
        };
        let b = [
            cpu.mmu.load_doubleword(b_addr).unwrap().0,
            cpu.mmu.load_doubleword(b_addr + 8).unwrap().0,
            cpu.mmu.load_doubleword(b_addr + 16).unwrap().0,
            cpu.mmu.load_doubleword(b_addr + 24).unwrap().0,
        ];
        // convert inputs to bigints
        let a_big: NBigUint = limbs_to_nbiguint(&a);
        let b_big: NBigUint = limbs_to_nbiguint(&b);
        let q_big: NBigUint = Fq::MODULUS.into();
        // compute advice based on operation type
        match self.op_type {
            MulqType::Div => {
                // compute a / b in the field
                let arr_to_fq = |a: &[u64; 4]| Fq::new(BigInt(*a));
                let c_big = limbs_to_nbiguint(
                    &((arr_to_fq(&b)
                        .inverse()
                        .expect("Attempted to invert zero in secp256k1 base field")
                        * arr_to_fq(&a))
                    .into_bigint()
                    .0),
                );
                let c_limbs = nbiguint_to_limbs(&c_big);
                let quotient = (b_big * c_big).div_floor(&q_big);
                // convert back to limbs
                let quotient_limbs = nbiguint_to_limbs(&quotient);
                // assert that limbs fits in 4 u64s
                assert!(quotient_limbs.len() <= 4, "Result does not fit in 4 limbs");
                // pad limbs to 4 u64s each, interleave, and return as VecDeque
                let mut padded_limbs = vec![0u64; 8];
                for i in 0..c_limbs.len() {
                    padded_limbs[2 * i] = c_limbs[i];
                }
                for i in 0..quotient_limbs.len() {
                    padded_limbs[2 * i + 1] = quotient_limbs[i];
                }
                VecDeque::from(padded_limbs)
            }
            _ => {
                // compute floor(a * b / q)
                let quotient = (a_big * b_big).div_floor(&q_big);
                // convert back to limbs
                let limbs = nbiguint_to_limbs(&quotient);
                // assert that limbs fits in 4 u64s
                assert!(limbs.len() <= 4, "Result does not fit in 4 limbs");
                // pad limbs to 4 u64s and return as VecDeque
                let mut padded_limbs = vec![0u64; 4];
                for i in 0..limbs.len() {
                    padded_limbs[i] = limbs[i];
                }
                VecDeque::from(padded_limbs)
            }
        }
    }
    // inline sequence function
    fn inline_sequence(mut self) -> Vec<Instruction> {
        // load a, b, and w
        for i in 0..4 {
            match self.op_type {
                MulqType::Mul => {
                    // if mul, load a and b
                    self.asm
                        .emit_ld::<LD>(*self.a[i], self.operands.rs1, i as i64 * 8);
                    self.asm.emit_ld::<LD>(
                        *self.b.as_ref().unwrap()[i],
                        self.operands.rs2,
                        i as i64 * 8,
                    );
                }
                MulqType::Square => {
                    // if square load only a
                    self.asm
                        .emit_ld::<LD>(*self.a[i], self.operands.rs1, i as i64 * 8);
                }
                MulqType::Div => {
                    // if div load b and
                    self.asm.emit_ld::<LD>(
                        *self.b.as_ref().unwrap()[i],
                        self.operands.rs2,
                        i as i64 * 8,
                    );
                    // load c into a, immediately copy it to memory
                    // the inline will error out if a != b * c mod q later, ensuring correctness
                    self.asm.emit_j::<VirtualAdvice>(*self.a[i], 0);
                    self.asm
                        .emit_s::<SD>(self.operands.rs3, *self.a[i], i as i64 * 8);
                }
            }
            self.asm.emit_j::<VirtualAdvice>(*self.w[i], 0);
        }
        // constant (1u64 << 32) + 977 lives in [12]
        self.asm.emit_u::<LUI>(*self.p, (1u64 << 32) + 977);
        // compute ab + wp into [14..22]
        // special handling for bottom limb r[0]
        match self.op_type {
            MulqType::Square => {
                self.asm.emit_r::<MUL>(*self.r[0], *self.a[0], *self.a[0]);
            }
            _ => {
                self.asm
                    .emit_r::<MUL>(*self.r[0], *self.a[0], *self.b.as_ref().unwrap()[0]);
            }
        }
        self.mac_low(*self.r[1], *self.r[0], *self.w[0], *self.p, *self.aux);
        // if mul or square, store the lowest limb in rs3
        // if div, verify that the lowest limb matches the lowest limbs of the actual argument a
        match self.op_type {
            MulqType::Div => {
                self.asm.emit_ld::<LD>(*self.aux, self.operands.rs1, 0);
                self.asm.emit_b::<VirtualAssertEQ>(*self.r[0], *self.aux, 0);
            }
            _ => {
                self.asm.emit_s::<SD>(self.operands.rs3, *self.r[0], 0);
            }
        }
        // loop over output limbs 1 through 6
        // here we ping-pong between r[0] and r[1] as the main limb and carry limb
        for k in 1..7 {
            // For each output limb r[k]
            // add in relevant products
            // if r[k+1] has not be written to yet, the carry goes directly into it
            let mut first = true;
            let rk = *self.r[k % 2];
            let rk_next = *self.r[(k + 1) % 2];
            // add all lower(w[i] * p) where i = k
            if k < 4 {
                self.mac_low(rk_next, rk, *self.w[k], *self.p, *self.aux);
                first = false;
            }
            // add all upper(w[i] * p) where i = k-1
            if k > 0 && k - 1 < 4 {
                self.mac_high_conditional(!first, rk_next, rk, *self.w[k - 1], *self.p, *self.aux);
                first = false;
            }
            // add all lower(a[i] * b[j]) where i+j = k
            for i in 0..=k {
                let j = k - i;
                if i < 4 && j < 4 {
                    match self.op_type {
                        MulqType::Square => {
                            if i > j {
                                break;
                            } else if i == j {
                                self.mac_low_conditional(
                                    !first, rk_next, rk, *self.a[i], *self.a[j], *self.aux,
                                );
                                first = false;
                            } else {
                                self.m2ac_low_conditional(
                                    !first,
                                    rk_next,
                                    rk,
                                    *self.a[i],
                                    *self.a[j],
                                    *self.aux,
                                    **self.aux2.as_ref().unwrap(),
                                );
                                first = false;
                            }
                        }
                        _ => {
                            self.mac_low_conditional(
                                !first,
                                rk_next,
                                rk,
                                *self.a[i],
                                *self.b.as_ref().unwrap()[j],
                                *self.aux,
                            );
                            first = false;
                        }
                    }
                }
            }
            // add all upper(a[i] * b[j]) where i+j = k-1
            for i in 0..=k - 1 {
                let j = k - 1 - i;
                if i < 4 && j < 4 {
                    match self.op_type {
                        MulqType::Square => {
                            if i > j {
                                break;
                            } else if i == j {
                                self.mac_high_conditional(
                                    !first, rk_next, rk, *self.a[i], *self.a[j], *self.aux,
                                );
                                first = false;
                            } else {
                                self.m2ac_high_conditional(
                                    !first,
                                    rk_next,
                                    rk,
                                    *self.a[i],
                                    *self.a[j],
                                    *self.aux,
                                    **self.aux2.as_ref().unwrap(),
                                );
                                first = false;
                            }
                        }
                        _ => {
                            self.mac_high_conditional(
                                !first,
                                rk_next,
                                rk,
                                *self.a[i],
                                *self.b.as_ref().unwrap()[j],
                                *self.aux,
                            );
                            first = false;
                        }
                    }
                }
            }
            // handle the lower limbs
            if k < 4 {
                // if mul or square, store the limb in rs3
                // if div, verify that the lower limbs match the the actual argument a
                match self.op_type {
                    MulqType::Div => {
                        self.asm
                            .emit_ld::<LD>(*self.aux, self.operands.rs1, k as i64 * 8);
                        self.asm.emit_b::<VirtualAssertEQ>(rk, *self.aux, 0);
                    }
                    _ => {
                        self.asm.emit_s::<SD>(self.operands.rs3, rk, k as i64 * 8);
                    }
                }
                // verify that the upper limbs match w
            } else if k >= 4 {
                self.asm.emit_b::<VirtualAssertEQ>(rk, *self.w[k - 4], 0);
            }
        }
        // special handling for top limb
        match self.op_type {
            MulqType::Square => {
                self.asm.emit_r::<MULHU>(*self.aux, *self.a[3], *self.a[3]);
            }
            _ => {
                self.asm
                    .emit_r::<MULHU>(*self.aux, *self.a[3], *self.b.as_ref().unwrap()[3]);
            }
        }
        self.asm.emit_r::<ADD>(*self.r[1], *self.r[1], *self.aux);
        // verify that w[4] matches top limb
        self.asm
            .emit_b::<VirtualAssertEQ>(*self.r[1], *self.w[3], 0);
        // ensure no overflow
        self.asm
            .emit_b::<VirtualAssertLTE>(*self.aux, *self.r[1], 0);
        // clean up inline
        drop(self.a);
        match self.op_type {
            MulqType::Square => {}
            _ => {
                drop(self.b.unwrap());
            }
        }
        drop(self.w);
        drop(self.p);
        drop(self.aux);
        match self.op_type {
            MulqType::Square => drop(self.aux2.unwrap()),
            _ => {}
        }
        drop(self.r);
        self.asm.finalize_inline()
    }
    // (c2, c1) = lower(a * b) + c1
    // clobbers aux
    fn mac_low(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8) {
        // mul aux, a, b
        self.asm.emit_r::<MUL>(aux, a, b);
        // add c1, c1, aux
        self.asm.emit_r::<ADD>(c1, c1, aux);
        // sltu c2, c1, aux
        self.asm.emit_r::<SLTU>(c2, c1, aux);
    }
    // (c2, c1) = upper(a * b) + c1
    // clobbers aux
    fn mac_high(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8) {
        // mulhu aux, a, b
        self.asm.emit_r::<MULHU>(aux, a, b);
        // add c1, c1, aux
        self.asm.emit_r::<ADD>(c1, c1, aux);
        // sltu c2, c1, aux
        self.asm.emit_r::<SLTU>(c2, c1, aux);
    }
    // (c2, c1) += lower(a * b)
    // clobbers aux
    fn mac_low_w_carry(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8) {
        // mulhu aux, a, b
        self.asm.emit_r::<MUL>(aux, a, b);
        // add c1, c1, aux
        self.asm.emit_r::<ADD>(c1, c1, aux);
        // sltu aux, c1, aux
        self.asm.emit_r::<SLTU>(aux, c1, aux);
        // add c2, c2, aux
        self.asm.emit_r::<ADD>(c2, c2, aux);
    }
    // (c2, c1) += upper(a * b)
    // clobbers aux
    fn mac_high_w_carry(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8) {
        // mulhu aux, a, b
        self.asm.emit_r::<MULHU>(aux, a, b);
        // add c1, c1, aux
        self.asm.emit_r::<ADD>(c1, c1, aux);
        // sltu aux, c1, aux
        self.asm.emit_r::<SLTU>(aux, c1, aux);
        // add c2, c2, aux
        self.asm.emit_r::<ADD>(c2, c2, aux);
    }
    // if carry flag is true, mac_low_w_carry, otherwise mac_low
    fn mac_low_conditional(&mut self, carry_exists: bool, c2: u8, c1: u8, a: u8, b: u8, aux: u8) {
        if carry_exists {
            self.mac_low_w_carry(c2, c1, a, b, aux);
        } else {
            self.mac_low(c2, c1, a, b, aux);
        }
    }
    // if carry flag is true, mac_high_w_carry, otherwise mac_high
    fn mac_high_conditional(&mut self, carry_exists: bool, c2: u8, c1: u8, a: u8, b: u8, aux: u8) {
        if carry_exists {
            self.mac_high_w_carry(c2, c1, a, b, aux);
        } else {
            self.mac_high(c2, c1, a, b, aux);
        }
    }
    // mac-like functions for the square case where one wants to add 2*a*b
    // (c2, c1) = 2*lower(a * b) + c1
    // clobbers aux
    fn m2ac_low(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8) {
        // mul aux, a, b
        self.asm.emit_r::<MUL>(aux, a, b);
        // add c1, c1, aux
        self.asm.emit_r::<ADD>(c1, c1, aux);
        // sltu c2, c1, aux
        self.asm.emit_r::<SLTU>(c2, c1, aux);
        // add c1, c1, aux
        self.asm.emit_r::<ADD>(c1, c1, aux);
        // sltu aux, c1, aux
        self.asm.emit_r::<SLTU>(aux, c1, aux);
        // add c2, c2, aux
        self.asm.emit_r::<ADD>(c2, c2, aux);
    }
    // (c2, c1) = 2*upper(a * b) + c1
    // clobbers aux
    fn m2ac_high(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8) {
        // mulhu aux, a, b
        self.asm.emit_r::<MULHU>(aux, a, b);
        // add c1, c1, aux
        self.asm.emit_r::<ADD>(c1, c1, aux);
        // sltu c2, c1, aux
        self.asm.emit_r::<SLTU>(c2, c1, aux);
        // add c1, c1, aux
        self.asm.emit_r::<ADD>(c1, c1, aux);
        // sltu aux, c1, aux
        self.asm.emit_r::<SLTU>(aux, c1, aux);
        // add c2, c2, aux
        self.asm.emit_r::<ADD>(c2, c2, aux);
    }
    // (c2, c1) += 2*lower(a * b)
    // clobbers aux
    fn m2ac_low_w_carry(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8, aux2: u8) {
        // mulhu aux, a, b
        self.asm.emit_r::<MUL>(aux, a, b);
        // add c1, c1, aux
        self.asm.emit_r::<ADD>(c1, c1, aux);
        // sltu aux2, c1, aux2
        self.asm.emit_r::<SLTU>(aux2, c1, aux);
        // add c2, c2, aux2
        self.asm.emit_r::<ADD>(c2, c2, aux2);
        // add c1, c1, aux
        self.asm.emit_r::<ADD>(c1, c1, aux);
        // sltu aux, c1, aux2
        self.asm.emit_r::<SLTU>(aux2, c1, aux);
        // add c2, c2, aux2
        self.asm.emit_r::<ADD>(c2, c2, aux2);
    }
    // (c2, c1) += 2*upper(a * b)
    // clobbers aux
    fn m2ac_high_w_carry(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8, aux2: u8) {
        // mulhu aux, a, b
        self.asm.emit_r::<MULHU>(aux, a, b);
        // add c1, c1, aux
        self.asm.emit_r::<ADD>(c1, c1, aux);
        // sltu aux2, c1, aux
        self.asm.emit_r::<SLTU>(aux2, c1, aux);
        // add c2, c2, aux
        self.asm.emit_r::<ADD>(c2, c2, aux2);
        // add c1, c1, aux
        self.asm.emit_r::<ADD>(c1, c1, aux);
        // sltu aux2, c1, aux
        self.asm.emit_r::<SLTU>(aux2, c1, aux);
        // add c2, c2, aux
        self.asm.emit_r::<ADD>(c2, c2, aux2);
    }
    // if carry flag is true, m2ac_low_w_carry, otherwise m2ac_low
    fn m2ac_low_conditional(
        &mut self,
        carry_exists: bool,
        c2: u8,
        c1: u8,
        a: u8,
        b: u8,
        aux: u8,
        aux2: u8,
    ) {
        if carry_exists {
            self.m2ac_low_w_carry(c2, c1, a, b, aux, aux2);
        } else {
            self.m2ac_low(c2, c1, a, b, aux);
        }
    }
    // if carry flag is true, m2ac_high_w_carry, otherwise m2ac_high
    fn m2ac_high_conditional(
        &mut self,
        carry_exists: bool,
        c2: u8,
        c1: u8,
        a: u8,
        b: u8,
        aux: u8,
        aux2: u8,
    ) {
        if carry_exists {
            self.m2ac_high_w_carry(c2, c1, a, b, aux, aux2);
        } else {
            self.m2ac_high(c2, c1, a, b, aux);
        }
    }
}

/// Virtual instruction builder for unchecked secp256k1 base field modular multiplication
pub fn secp256k1_mulq_sequence_builder(
    asm: InstrAssembler,
    operands: FormatInline,
) -> Vec<Instruction> {
    let builder = Secp256k1Mulq::new(asm, operands, MulqType::Mul);
    builder.inline_sequence()
}

/// Custom trace function for unchecked secp256k1 base field modular multiplication
pub fn secp256k1_mulq_advice(
    asm: InstrAssembler,
    operands: FormatInline,
    cpu: &mut Cpu,
) -> VecDeque<u64> {
    let builder = Secp256k1Mulq::new(asm, operands, MulqType::Mul);
    builder.advice(cpu)
}

/// Virtual instruction builder for unchecked secp256k1 base field modular squaring
pub fn secp256k1_squareq_sequence_builder(
    asm: InstrAssembler,
    operands: FormatInline,
) -> Vec<Instruction> {
    let builder = Secp256k1Mulq::new(asm, operands, MulqType::Square);
    builder.inline_sequence()
}

/// Custom trace function for unchecked secp256k1 base field modular squaring
pub fn secp256k1_squareq_advice(
    asm: InstrAssembler,
    operands: FormatInline,
    cpu: &mut Cpu,
) -> VecDeque<u64> {
    let builder = Secp256k1Mulq::new(asm, operands, MulqType::Square);
    builder.advice(cpu)
}

/// Virtual instruction builder for unchecked secp256k1 base field modular division
pub fn secp256k1_divq_sequence_builder(
    asm: InstrAssembler,
    operands: FormatInline,
) -> Vec<Instruction> {
    let builder = Secp256k1Mulq::new(asm, operands, MulqType::Div);
    builder.inline_sequence()
}

/// Custom trace function for unchecked secp256k1 base field modular division
pub fn secp256k1_divq_advice(
    asm: InstrAssembler,
    operands: FormatInline,
    cpu: &mut Cpu,
) -> VecDeque<u64> {
    let builder = Secp256k1Mulq::new(asm, operands, MulqType::Div);
    builder.advice(cpu)
}
