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
    limbs_to_nbiguint, mulq_advice, Cpu, FormatInline, InlineOp, InstrAssembler, InstrAssemblerExt,
    Instruction, MulqType, VirtualRegisterGuard,
};

/// inline constructor for GLV decomposition in secp256k1 scalar field
struct GlvrAdvBuilder {
    asm: InstrAssembler,
    vr: VirtualRegisterGuard,
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
    fn inline_sequence(mut self) -> Vec<Instruction> {
        self.asm.emit_advice_stores(*self.vr, self.operands.rs3, 6);
        drop(self.vr);
        self.asm.finalize_inline()
    }
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
    b: Option<[VirtualRegisterGuard; 4]>,
    w: [VirtualRegisterGuard; 4],
    p: VirtualRegisterGuard,
    p2: Option<VirtualRegisterGuard>,
    aux: VirtualRegisterGuard,
    aux2: Option<VirtualRegisterGuard>,
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
    fn advice(self, cpu: &mut Cpu) -> VecDeque<u64> {
        mulq_advice(
            &self.operands,
            cpu,
            self.is_scalar_field,
            &self.op_type,
            |is_scalar| {
                if is_scalar {
                    Fr::MODULUS.into()
                } else {
                    Fq::MODULUS.into()
                }
            },
            |b, a| {
                limbs_to_nbiguint(
                    &if self.is_scalar_field {
                        (Fr::new(BigInt(*b))
                            .inverse()
                            .expect("Attempted to invert zero in secp256k1 scalar field")
                            * Fr::new(BigInt(*a)))
                        .into_bigint()
                    } else {
                        (Fq::new(BigInt(*b))
                            .inverse()
                            .expect("Attempted to invert zero in secp256k1 base field")
                            * Fq::new(BigInt(*a)))
                        .into_bigint()
                    }
                    .0,
                )
            },
        )
    }
    fn inline_sequence(mut self) -> Vec<Instruction> {
        for i in 0..4 {
            match self.op_type {
                MulqType::Mul => {
                    self.asm
                        .emit_ld::<LD>(*self.a[i], self.operands.rs1, i as i64 * 8);
                    self.asm.emit_ld::<LD>(
                        *self.b.as_ref().unwrap()[i],
                        self.operands.rs2,
                        i as i64 * 8,
                    );
                }
                MulqType::Square => {
                    self.asm
                        .emit_ld::<LD>(*self.a[i], self.operands.rs1, i as i64 * 8);
                }
                MulqType::Div => {
                    self.asm.emit_ld::<LD>(
                        *self.b.as_ref().unwrap()[i],
                        self.operands.rs2,
                        i as i64 * 8,
                    );
                    self.asm.emit_j::<VirtualAdvice>(*self.a[i], 0);
                    self.asm
                        .emit_s::<SD>(self.operands.rs3, *self.a[i], i as i64 * 8);
                }
            }
            self.asm.emit_j::<VirtualAdvice>(*self.w[i], 0);
        }
        if self.is_scalar_field {
            self.asm.emit_u::<LUI>(*self.p, 0x402da1732fc9bebf);
            self.asm
                .emit_u::<LUI>(**self.p2.as_ref().unwrap(), 0x4551231950b75fc4);
        } else {
            self.asm.emit_u::<LUI>(*self.p, (1u64 << 32) + 977);
        }
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
        match self.op_type {
            MulqType::Div => {
                self.asm.emit_ld::<LD>(*self.aux, self.operands.rs1, 0);
                self.asm.emit_b::<VirtualAssertEQ>(*self.r[0], *self.aux, 0);
            }
            _ => {
                self.asm.emit_s::<SD>(self.operands.rs3, *self.r[0], 0);
            }
        }
        for k in 1..7 {
            let mut first = true;
            let rk = *self.r[k % 2];
            let rk_next = *self.r[(k + 1) % 2];
            if k < 4 {
                self.mac_low(rk_next, rk, *self.w[k], *self.p, *self.aux);
                first = false;
            }
            if k - 1 < 4 {
                self.mac_high_conditional(!first, rk_next, rk, *self.w[k - 1], *self.p, *self.aux);
                first = false;
            }
            if self.is_scalar_field {
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
            if k < 4 {
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
            } else if k >= 4 {
                self.asm.emit_b::<VirtualAssertEQ>(rk, *self.w[k - 4], 0);
            }
        }
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
        self.asm
            .emit_b::<VirtualAssertEQ>(*self.r[1], *self.w[3], 0);
        self.asm
            .emit_b::<VirtualAssertLTE>(*self.aux, *self.r[1], 0);
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
    fn mac_low(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8) {
        self.asm.emit_r::<MUL>(aux, a, b);
        self.asm.emit_r::<ADD>(c1, c1, aux);
        self.asm.emit_r::<SLTU>(c2, c1, aux);
    }
    fn mac_high(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8) {
        self.asm.emit_r::<MULHU>(aux, a, b);
        self.asm.emit_r::<ADD>(c1, c1, aux);
        self.asm.emit_r::<SLTU>(c2, c1, aux);
    }
    fn mac_low_w_carry(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8) {
        self.asm.emit_r::<MUL>(aux, a, b);
        self.asm.emit_r::<ADD>(c1, c1, aux);
        self.asm.emit_r::<SLTU>(aux, c1, aux);
        self.asm.emit_r::<ADD>(c2, c2, aux);
    }
    fn mac_high_w_carry(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8) {
        self.asm.emit_r::<MULHU>(aux, a, b);
        self.asm.emit_r::<ADD>(c1, c1, aux);
        self.asm.emit_r::<SLTU>(aux, c1, aux);
        self.asm.emit_r::<ADD>(c2, c2, aux);
    }
    fn mac_low_conditional(&mut self, carry_exists: bool, c2: u8, c1: u8, a: u8, b: u8, aux: u8) {
        if carry_exists {
            self.mac_low_w_carry(c2, c1, a, b, aux);
        } else {
            self.mac_low(c2, c1, a, b, aux);
        }
    }
    fn mac_high_conditional(&mut self, carry_exists: bool, c2: u8, c1: u8, a: u8, b: u8, aux: u8) {
        if carry_exists {
            self.mac_high_w_carry(c2, c1, a, b, aux);
        } else {
            self.mac_high(c2, c1, a, b, aux);
        }
    }
    fn m2ac_low(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8) {
        self.asm.emit_r::<MUL>(aux, a, b);
        self.asm.emit_r::<ADD>(c1, c1, aux);
        self.asm.emit_r::<SLTU>(c2, c1, aux);
        self.asm.emit_r::<ADD>(c1, c1, aux);
        self.asm.emit_r::<SLTU>(aux, c1, aux);
        self.asm.emit_r::<ADD>(c2, c2, aux);
    }
    fn m2ac_high(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8) {
        self.asm.emit_r::<MULHU>(aux, a, b);
        self.asm.emit_r::<ADD>(c1, c1, aux);
        self.asm.emit_r::<SLTU>(c2, c1, aux);
        self.asm.emit_r::<ADD>(c1, c1, aux);
        self.asm.emit_r::<SLTU>(aux, c1, aux);
        self.asm.emit_r::<ADD>(c2, c2, aux);
    }
    fn m2ac_low_w_carry(&mut self, c2: u8, c1: u8, a: u8, b: u8, aux: u8, aux2: u8) {
        self.asm.emit_r::<MUL>(aux, a, b);
        self.asm.emit_r::<ADD>(c1, c1, aux);
        self.asm.emit_r::<SLTU>(aux2, c1, aux);
        self.asm.emit_r::<ADD>(c2, c2, aux2);
        self.asm.emit_r::<ADD>(c1, c1, aux);
        self.asm.emit_r::<SLTU>(aux2, c1, aux);
        self.asm.emit_r::<ADD>(c2, c2, aux2);
    }
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

macro_rules! secp256k1_mulq_op {
    ($name:ident, funct3: $funct3:expr, name: $op_name:expr, mul_type: $mul_type:expr, is_scalar: $is_scalar:expr) => {
        pub struct $name;
        impl InlineOp for $name {
            const OPCODE: u32 = crate::INLINE_OPCODE;
            const FUNCT3: u32 = $funct3;
            const FUNCT7: u32 = crate::SECP256K1_FUNCT7;
            const NAME: &'static str = $op_name;
            fn build_sequence(asm: InstrAssembler, operands: FormatInline) -> Vec<Instruction> {
                MulqBuilder::new(asm, operands, $mul_type, $is_scalar).inline_sequence()
            }
            fn build_advice(
                asm: InstrAssembler,
                operands: FormatInline,
                cpu: &mut Cpu,
            ) -> Option<VecDeque<u64>> {
                Some(MulqBuilder::new(asm, operands, $mul_type, $is_scalar).advice(cpu))
            }
        }
    };
}

secp256k1_mulq_op!(Secp256k1MulQ,     funct3: crate::SECP256K1_MULQ_FUNCT3,    name: crate::SECP256K1_MULQ_NAME,    mul_type: MulqType::Mul,    is_scalar: false);
secp256k1_mulq_op!(Secp256k1SquareQ,  funct3: crate::SECP256K1_SQUAREQ_FUNCT3, name: crate::SECP256K1_SQUAREQ_NAME, mul_type: MulqType::Square, is_scalar: false);
secp256k1_mulq_op!(Secp256k1DivQ,     funct3: crate::SECP256K1_DIVQ_FUNCT3,    name: crate::SECP256K1_DIVQ_NAME,    mul_type: MulqType::Div,    is_scalar: false);
secp256k1_mulq_op!(Secp256k1MulR,     funct3: crate::SECP256K1_MULR_FUNCT3,    name: crate::SECP256K1_MULR_NAME,    mul_type: MulqType::Mul,    is_scalar: true);
secp256k1_mulq_op!(Secp256k1SquareR,  funct3: crate::SECP256K1_SQUARER_FUNCT3, name: crate::SECP256K1_SQUARER_NAME, mul_type: MulqType::Square, is_scalar: true);
secp256k1_mulq_op!(Secp256k1DivR,     funct3: crate::SECP256K1_DIVR_FUNCT3,    name: crate::SECP256K1_DIVR_NAME,    mul_type: MulqType::Div,    is_scalar: true);

pub struct Secp256k1GlvrAdv;
impl InlineOp for Secp256k1GlvrAdv {
    const OPCODE: u32 = crate::INLINE_OPCODE;
    const FUNCT3: u32 = crate::SECP256K1_GLVR_ADV_FUNCT3;
    const FUNCT7: u32 = crate::SECP256K1_FUNCT7;
    const NAME: &'static str = crate::SECP256K1_GLVR_ADV_NAME;
    fn build_sequence(asm: InstrAssembler, operands: FormatInline) -> Vec<Instruction> {
        GlvrAdvBuilder::new(asm, operands).inline_sequence()
    }
    fn build_advice(
        asm: InstrAssembler,
        operands: FormatInline,
        cpu: &mut Cpu,
    ) -> Option<VecDeque<u64>> {
        Some(GlvrAdvBuilder::new(asm, operands).advice(cpu))
    }
}
