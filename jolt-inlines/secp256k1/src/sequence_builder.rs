use std::array;
use std::collections::VecDeque;

use ark_ff::{BigInt, Field, PrimeField};
use ark_secp256k1::{Fq, Fr};
use jolt_inlines_sdk::host::{
    instruction::{
        add::ADD, ld::LD, lui::LUI, mul::MUL, mulhu::MULHU, sd::SD, sltu::SLTU,
        virtual_advice::VirtualAdvice, virtual_assert_eq::VirtualAssertEQ,
        virtual_assert_lte::VirtualAssertLTE,
    },
    Cpu, FormatInline, InlineOp, InstrAssembler, Instruction, VirtualRegisterGuard,
};
use num_bigint::BigUint as NBigUint;
use num_integer::Integer;

/// inline constructor for GLV decomposition in secp256k1 scalar field
struct GlvrAdvBuilder {
    asm: InstrAssembler,
    vr: VirtualRegisterGuard, // only one register needed
    operands: FormatInline,
}

impl GlvrAdvBuilder {
    fn new(asm: InstrAssembler, operands: FormatInline) -> Self {
        let vr = asm.allocator.allocate_for_inline();
        GlvrAdvBuilder { asm, vr, operands }
    }
    fn advice(self, cpu: &mut Cpu) -> VecDeque<u64> {
        let k_addr = cpu.x[self.operands.rs1 as usize] as u64;
        let kr = [
            cpu.mmu.load_doubleword(k_addr).unwrap().0,
            cpu.mmu.load_doubleword(k_addr + 8).unwrap().0,
            cpu.mmu.load_doubleword(k_addr + 16).unwrap().0,
            cpu.mmu.load_doubleword(k_addr + 24).unwrap().0,
        ];
        let k = Fr::new(BigInt(kr)).into_bigint().into();
        let result = crate::glv::decompose_scalar_to_u64s(k);
        VecDeque::from(result.to_vec())
    }
    fn build(mut self) -> Vec<Instruction> {
        for i in 0..6 {
            self.asm.emit_j::<VirtualAdvice>(*self.vr, 0);
            self.asm
                .emit_s::<SD>(self.operands.rs3, *self.vr, i as i64 * 8);
        }
        drop(self.vr);
        self.asm.finalize_inline()
    }
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
    let mut limbs = vec![0u64; bytes.len().div_ceil(8)];
    for (i, byte) in bytes.iter().enumerate() {
        limbs[i / 8] |= (*byte as u64) << ((i % 8) * 8);
    }
    limbs
}

/// Enum for type of multiplication-style operation
enum MulqType {
    Mul,
    Square,
    Div,
}

// inline for secp256k1 base and scalar field multiplication/squaring/division
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
// Additionally, to minimize the use of virtual registers, the LHS is never fully constructed.
// Rather, the implementation considers 2 limbs at a time, one that is the main focus
// And one that accumulates carries. The algorithm ping-pongs between these two limbs as it computes the LHS.
struct MulqBuilder {
    asm: InstrAssembler,
    a: [VirtualRegisterGuard; 4],
    b: Option<[VirtualRegisterGuard; 4]>, // only allocated if Mul or Div
    w: [VirtualRegisterGuard; 4],
    p: VirtualRegisterGuard,
    p2: Option<VirtualRegisterGuard>, // only allocated if scalar field
    aux: VirtualRegisterGuard,
    aux2: Option<VirtualRegisterGuard>, // only allocated if Square
    r: [VirtualRegisterGuard; 2],
    operands: FormatInline,
    op_type: MulqType,
    is_scalar_field: bool,
}

impl MulqBuilder {
    fn new(
        asm: InstrAssembler,
        operands: FormatInline,
        op_type: MulqType,
        is_scalar_field: bool,
    ) -> Self {
        let a = array::from_fn(|_| asm.allocator.allocate_for_inline());
        let b = match op_type {
            MulqType::Square => None,
            _ => Some(array::from_fn(|_| asm.allocator.allocate_for_inline())),
        };
        let w = array::from_fn(|_| asm.allocator.allocate_for_inline());
        let p = asm.allocator.allocate_for_inline();
        let p2 = if is_scalar_field {
            Some(asm.allocator.allocate_for_inline())
        } else {
            None
        };
        let aux = asm.allocator.allocate_for_inline();
        let aux2 = match op_type {
            MulqType::Square => Some(asm.allocator.allocate_for_inline()),
            _ => None,
        };
        let r = array::from_fn(|_| asm.allocator.allocate_for_inline());
        MulqBuilder {
            asm,
            a,
            b,
            w,
            p,
            p2,
            aux,
            aux2,
            r,
            operands,
            op_type,
            is_scalar_field,
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
        let q_big: NBigUint = if self.is_scalar_field {
            Fr::MODULUS.into()
        } else {
            Fq::MODULUS.into()
        };
        // compute advice based on operation type
        match self.op_type {
            MulqType::Div => {
                // compute a / b in the field
                let arr_to_fq = |a: &[u64; 4]| Fq::new(BigInt(*a));
                let arr_to_fr = |a: &[u64; 4]| Fr::new(BigInt(*a));
                let c_big = limbs_to_nbiguint(
                    &if self.is_scalar_field {
                        (arr_to_fr(&b)
                            .inverse()
                            .expect("Attempted to invert zero in secp256k1 scalar field")
                            * arr_to_fr(&a))
                        .into_bigint()
                    } else {
                        (arr_to_fq(&b)
                            .inverse()
                            .expect("Attempted to invert zero in secp256k1 base field")
                            * arr_to_fq(&a))
                        .into_bigint()
                    }
                    .0,
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
                padded_limbs[..limbs.len()].copy_from_slice(&limbs[..]);
                VecDeque::from(padded_limbs)
            }
        }
    }
    fn build(mut self) -> Vec<Instruction> {
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
        // load p (either as a single u64 or a pair of u64s
        if self.is_scalar_field {
            self.asm.emit_u::<LUI>(*self.p, 0x402da1732fc9bebf);
            self.asm
                .emit_u::<LUI>(**self.p2.as_ref().unwrap(), 0x4551231950b75fc4);
        } else {
            self.asm.emit_u::<LUI>(*self.p, (1u64 << 32) + 977);
        }
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
            if k - 1 < 4 {
                self.mac_high_conditional(!first, rk_next, rk, *self.w[k - 1], *self.p, *self.aux);
                first = false;
            }
            // if in the scalar field
            if self.is_scalar_field {
                // add all lower(w[i] * p2) where i = k-1
                if k > 0 && k - 1 < 4 {
                    self.mac_low_conditional(
                        !first,
                        rk_next,
                        rk,
                        *self.w[k - 1],
                        **self.p2.as_ref().unwrap(),
                        *self.aux,
                    );
                    first = false;
                }
                // add all upper(w[i] * p2) where i = k-2
                // and add an additional w[i]
                if k > 1 && k - 2 < 4 {
                    self.mac_high_conditional(
                        !first,
                        rk_next,
                        rk,
                        *self.w[k - 2],
                        **self.p2.as_ref().unwrap(),
                        *self.aux,
                    );
                    first = false;
                    self.asm.emit_r::<ADD>(rk, rk, *self.w[k - 2]);
                    self.asm.emit_r::<SLTU>(*self.aux, rk, *self.w[k - 2]);
                    self.asm.emit_r::<ADD>(rk_next, rk_next, *self.aux);
                }
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
                                if !first {
                                    self.m2ac_low_w_carry(
                                        rk_next,
                                        rk,
                                        *self.a[i],
                                        *self.a[j],
                                        *self.aux,
                                        **self.aux2.as_ref().unwrap(),
                                    );
                                } else {
                                    self.m2ac_low(rk_next, rk, *self.a[i], *self.a[j], *self.aux);
                                }
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
                                if !first {
                                    self.m2ac_high_w_carry(
                                        rk_next,
                                        rk,
                                        *self.a[i],
                                        *self.a[j],
                                        *self.aux,
                                        **self.aux2.as_ref().unwrap(),
                                    );
                                } else {
                                    self.m2ac_high(rk_next, rk, *self.a[i], *self.a[j], *self.aux);
                                }
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
        if self.is_scalar_field {
            drop(self.p2.unwrap());
        }
        drop(self.aux);
        if let MulqType::Square = self.op_type {
            drop(self.aux2.unwrap())
        }
        drop(self.r);
        self.asm.finalize_inline()
    }
    // (c2, c1) = lower(a * b) + c1
    // clobbers aux
    fn mac_low(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8) {
        self.asm.emit_r::<MUL>(aux, a, b);
        self.asm.emit_r::<ADD>(c1, c1, aux);
        self.asm.emit_r::<SLTU>(c2, c1, aux);
    }
    // (c2, c1) = upper(a * b) + c1
    // clobbers aux
    fn mac_high(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8) {
        self.asm.emit_r::<MULHU>(aux, a, b);
        self.asm.emit_r::<ADD>(c1, c1, aux);
        self.asm.emit_r::<SLTU>(c2, c1, aux);
    }
    // (c2, c1) += lower(a * b)
    // clobbers aux
    fn mac_low_w_carry(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8) {
        self.asm.emit_r::<MUL>(aux, a, b);
        self.asm.emit_r::<ADD>(c1, c1, aux);
        self.asm.emit_r::<SLTU>(aux, c1, aux);
        self.asm.emit_r::<ADD>(c2, c2, aux);
    }
    // (c2, c1) += upper(a * b)
    // clobbers aux
    fn mac_high_w_carry(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8) {
        self.asm.emit_r::<MULHU>(aux, a, b);
        self.asm.emit_r::<ADD>(c1, c1, aux);
        self.asm.emit_r::<SLTU>(aux, c1, aux);
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
        self.asm.emit_r::<MUL>(aux, a, b);
        self.asm.emit_r::<ADD>(c1, c1, aux);
        self.asm.emit_r::<SLTU>(c2, c1, aux);
        self.asm.emit_r::<ADD>(c1, c1, aux);
        self.asm.emit_r::<SLTU>(aux, c1, aux);
        self.asm.emit_r::<ADD>(c2, c2, aux);
    }
    // (c2, c1) = 2*upper(a * b) + c1
    // clobbers aux
    fn m2ac_high(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8) {
        self.asm.emit_r::<MULHU>(aux, a, b);
        self.asm.emit_r::<ADD>(c1, c1, aux);
        self.asm.emit_r::<SLTU>(c2, c1, aux);
        self.asm.emit_r::<ADD>(c1, c1, aux);
        self.asm.emit_r::<SLTU>(aux, c1, aux);
        self.asm.emit_r::<ADD>(c2, c2, aux);
    }
    // (c2, c1) += 2*lower(a * b)
    // clobbers aux
    fn m2ac_low_w_carry(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8, aux2: u8) {
        self.asm.emit_r::<MUL>(aux, a, b);
        self.asm.emit_r::<ADD>(c1, c1, aux);
        self.asm.emit_r::<SLTU>(aux2, c1, aux);
        self.asm.emit_r::<ADD>(c2, c2, aux2);
        self.asm.emit_r::<ADD>(c1, c1, aux);
        self.asm.emit_r::<SLTU>(aux2, c1, aux);
        self.asm.emit_r::<ADD>(c2, c2, aux2);
    }
    // (c2, c1) += 2*upper(a * b)
    // clobbers aux
    fn m2ac_high_w_carry(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8, aux2: u8) {
        self.asm.emit_r::<MULHU>(aux, a, b);
        self.asm.emit_r::<ADD>(c1, c1, aux);
        self.asm.emit_r::<SLTU>(aux2, c1, aux);
        self.asm.emit_r::<ADD>(c2, c2, aux2);
        self.asm.emit_r::<ADD>(c1, c1, aux);
        self.asm.emit_r::<SLTU>(aux2, c1, aux);
        self.asm.emit_r::<ADD>(c2, c2, aux2);
    }
}

pub struct Secp256k1MulQ;

impl InlineOp for Secp256k1MulQ {
    const OPCODE: u32 = crate::INLINE_OPCODE;
    const FUNCT3: u32 = crate::SECP256K1_MULQ_FUNCT3;
    const FUNCT7: u32 = crate::SECP256K1_FUNCT7;
    const NAME: &'static str = crate::SECP256K1_MULQ_NAME;
    type Advice = VecDeque<u64>;

    fn build_sequence(asm: InstrAssembler, operands: FormatInline) -> Vec<Instruction> {
        MulqBuilder::new(asm, operands, MulqType::Mul, false).build()
    }

    fn build_advice(asm: InstrAssembler, operands: FormatInline, cpu: &mut Cpu) -> VecDeque<u64> {
        MulqBuilder::new(asm, operands, MulqType::Mul, false).advice(cpu)
    }
}

pub struct Secp256k1SquareQ;

impl InlineOp for Secp256k1SquareQ {
    const OPCODE: u32 = crate::INLINE_OPCODE;
    const FUNCT3: u32 = crate::SECP256K1_SQUAREQ_FUNCT3;
    const FUNCT7: u32 = crate::SECP256K1_FUNCT7;
    const NAME: &'static str = crate::SECP256K1_SQUAREQ_NAME;
    type Advice = VecDeque<u64>;

    fn build_sequence(asm: InstrAssembler, operands: FormatInline) -> Vec<Instruction> {
        MulqBuilder::new(asm, operands, MulqType::Square, false).build()
    }

    fn build_advice(asm: InstrAssembler, operands: FormatInline, cpu: &mut Cpu) -> VecDeque<u64> {
        MulqBuilder::new(asm, operands, MulqType::Square, false).advice(cpu)
    }
}

pub struct Secp256k1DivQ;

impl InlineOp for Secp256k1DivQ {
    const OPCODE: u32 = crate::INLINE_OPCODE;
    const FUNCT3: u32 = crate::SECP256K1_DIVQ_FUNCT3;
    const FUNCT7: u32 = crate::SECP256K1_FUNCT7;
    const NAME: &'static str = crate::SECP256K1_DIVQ_NAME;
    type Advice = VecDeque<u64>;

    fn build_sequence(asm: InstrAssembler, operands: FormatInline) -> Vec<Instruction> {
        MulqBuilder::new(asm, operands, MulqType::Div, false).build()
    }

    fn build_advice(asm: InstrAssembler, operands: FormatInline, cpu: &mut Cpu) -> VecDeque<u64> {
        MulqBuilder::new(asm, operands, MulqType::Div, false).advice(cpu)
    }
}

pub struct Secp256k1MulR;

impl InlineOp for Secp256k1MulR {
    const OPCODE: u32 = crate::INLINE_OPCODE;
    const FUNCT3: u32 = crate::SECP256K1_MULR_FUNCT3;
    const FUNCT7: u32 = crate::SECP256K1_FUNCT7;
    const NAME: &'static str = crate::SECP256K1_MULR_NAME;
    type Advice = VecDeque<u64>;

    fn build_sequence(asm: InstrAssembler, operands: FormatInline) -> Vec<Instruction> {
        MulqBuilder::new(asm, operands, MulqType::Mul, true).build()
    }

    fn build_advice(asm: InstrAssembler, operands: FormatInline, cpu: &mut Cpu) -> VecDeque<u64> {
        MulqBuilder::new(asm, operands, MulqType::Mul, true).advice(cpu)
    }
}

pub struct Secp256k1SquareR;

impl InlineOp for Secp256k1SquareR {
    const OPCODE: u32 = crate::INLINE_OPCODE;
    const FUNCT3: u32 = crate::SECP256K1_SQUARER_FUNCT3;
    const FUNCT7: u32 = crate::SECP256K1_FUNCT7;
    const NAME: &'static str = crate::SECP256K1_SQUARER_NAME;
    type Advice = VecDeque<u64>;

    fn build_sequence(asm: InstrAssembler, operands: FormatInline) -> Vec<Instruction> {
        MulqBuilder::new(asm, operands, MulqType::Square, true).build()
    }

    fn build_advice(asm: InstrAssembler, operands: FormatInline, cpu: &mut Cpu) -> VecDeque<u64> {
        MulqBuilder::new(asm, operands, MulqType::Square, true).advice(cpu)
    }
}

pub struct Secp256k1DivR;

impl InlineOp for Secp256k1DivR {
    const OPCODE: u32 = crate::INLINE_OPCODE;
    const FUNCT3: u32 = crate::SECP256K1_DIVR_FUNCT3;
    const FUNCT7: u32 = crate::SECP256K1_FUNCT7;
    const NAME: &'static str = crate::SECP256K1_DIVR_NAME;
    type Advice = VecDeque<u64>;

    fn build_sequence(asm: InstrAssembler, operands: FormatInline) -> Vec<Instruction> {
        MulqBuilder::new(asm, operands, MulqType::Div, true).build()
    }

    fn build_advice(asm: InstrAssembler, operands: FormatInline, cpu: &mut Cpu) -> VecDeque<u64> {
        MulqBuilder::new(asm, operands, MulqType::Div, true).advice(cpu)
    }
}

pub struct Secp256k1GlvrAdv;

impl InlineOp for Secp256k1GlvrAdv {
    const OPCODE: u32 = crate::INLINE_OPCODE;
    const FUNCT3: u32 = crate::SECP256K1_GLVR_ADV_FUNCT3;
    const FUNCT7: u32 = crate::SECP256K1_FUNCT7;
    const NAME: &'static str = crate::SECP256K1_GLVR_ADV_NAME;
    type Advice = VecDeque<u64>;

    fn build_sequence(asm: InstrAssembler, operands: FormatInline) -> Vec<Instruction> {
        GlvrAdvBuilder::new(asm, operands).build()
    }

    fn build_advice(asm: InstrAssembler, operands: FormatInline, cpu: &mut Cpu) -> VecDeque<u64> {
        GlvrAdvBuilder::new(asm, operands).advice(cpu)
    }
}
