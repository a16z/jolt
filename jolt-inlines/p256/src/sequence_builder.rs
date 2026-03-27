use std::array;
use std::collections::VecDeque;

use ark_ff::{BigInt, Field, PrimeField};
use ark_secp256r1::{Fq, Fr};
use jolt_inlines_sdk::host::{
    instruction::{
        add::ADD, ld::LD, lui::LUI, mul::MUL, mulhu::MULHU, sd::SD,
        virtual_advice::VirtualAdvice, virtual_assert_eq::VirtualAssertEQ,
        virtual_assert_lte::VirtualAssertLTE,
    },
    Cpu, FormatInline, InlineOp, InstrAssembler, Instruction, MulAccExt, VirtualRegisterGuard,
};
use num_bigint::{BigInt as NBigInt, BigUint as NBigUint};
use num_integer::Integer;

// p = 2^256 - q for base field:
//   p[0] = 0x0000000000000001  (special: equals 1, w[i]*p[0] = w[i], use ADD)
//   p[1] = 0xFFFFFFFF00000000  (loaded into p1)
//   p[2] = 0xFFFFFFFFFFFFFFFF  (loaded into p2; equals -1 mod 2^64, w[i]*p[2] = -w[i] for MUL)
//   p[3] = 0x00000000FFFFFFFE  (loaded into p3)
const P256_PQ: [u64; 4] = [
    0x0000000000000001,
    0xFFFFFFFF00000000,
    0xFFFFFFFFFFFFFFFF,
    0x00000000FFFFFFFE,
];

// p = 2^256 - n for scalar field:
//   p[0] = 0x0C46353D039CDAAF  (loaded into p1)
//   p[1] = 0x4319055258E8617B  (loaded into p2)
//   p[2] = 0x0000000000000000  (zero! skip all terms)
//   p[3] = 0x00000000FFFFFFFF  (loaded into p3)
const P256_NEG_N: [u64; 4] = [
    0x0C46353D039CDAAF,
    0x4319055258E8617B,
    0x0000000000000000,
    0x00000000FFFFFFFF,
];

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

// inline for P-256 base and scalar field multiplication/squaring/division
// does not handle checking that the result is canonical mod q,
// merely that it is correct and fits in 4 limbs
// multiplication followed by modulus can be represented as: ab = wq + c
// for some w and c in [0, q) where w is provided as advice
// here we do not explicitly check that c is in [0, q), but rather that c fits in 256 bits
// the fastest possible check uses branching and thus appears after invocations of this inline
// let p = (2^256 - q), the equation above can be rearranged to: ab + wp = 2^256 w + c
// Thus, for multiplication, squaring and division, this inline checks the following:
// For multiplication, this inline checks that ab + wp = 2^256 w + c
// For division, this inline checks that cb + wp = 2^256 w + a
// For squaring, this inline checks that a^2 + wp = 2^256 w + c
// The core structure is the same for all three operations, with only minor differences
// in loading inputs and storing outputs.
//
// P-256 differs from secp256k1 in that p = 2^256 - q has 4 nonzero limbs for the base field
// and 3 nonzero limbs for the scalar field, leading to more w*p cross-terms per output limb.
//
// Base field: p = [1, 0xFFFFFFFF00000000, 0xFFFFFFFFFFFFFFFF, 0x00000000FFFFFFFE]
//   p[0] = 1 is handled implicitly (ADD instead of MUL, high part = 0)
//   p[1], p[2], p[3] loaded into registers p1, p2, p3
//
// Scalar field: p = [0x0C46353D039CDAAF, 0x4319055258E8617B, 0, 0x00000000FFFFFFFF]
//   p[0], p[1], p[3] loaded into registers p1, p2, p3
//   p[2] = 0 so all terms involving p[2] are skipped
struct P256Mulq {
    asm: InstrAssembler,
    a: [VirtualRegisterGuard; 4],
    b: Option<[VirtualRegisterGuard; 4]>, // only allocated if Mul or Div
    w: [VirtualRegisterGuard; 4],
    p1: VirtualRegisterGuard,
    p2: VirtualRegisterGuard,
    p3: VirtualRegisterGuard,
    aux: VirtualRegisterGuard,
    aux2: Option<VirtualRegisterGuard>, // only allocated if Square
    r: [VirtualRegisterGuard; 2],
    operands: FormatInline,
    op_type: MulqType,
    is_scalar_field: bool,
}

impl P256Mulq {
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
        let p1 = asm.allocator.allocate_for_inline();
        let p2 = asm.allocator.allocate_for_inline();
        let p3 = asm.allocator.allocate_for_inline();
        let aux = asm.allocator.allocate_for_inline();
        let aux2 = match op_type {
            MulqType::Square => Some(asm.allocator.allocate_for_inline()),
            _ => None,
        };
        let r = array::from_fn(|_| asm.allocator.allocate_for_inline());
        P256Mulq {
            asm,
            a,
            b,
            w,
            p1,
            p2,
            p3,
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
                // compute a / b in the field using arkworks: c = b^{-1} * a mod q
                let arr_to_fq = |a: &[u64; 4]| Fq::new(BigInt(*a));
                let arr_to_fr = |a: &[u64; 4]| Fr::new(BigInt(*a));
                let c_big = limbs_to_nbiguint(
                    &if self.is_scalar_field {
                        (arr_to_fr(&b)
                            .inverse()
                            .expect("Attempted to invert zero in P-256 scalar field")
                            * arr_to_fr(&a))
                        .into_bigint()
                    } else {
                        (arr_to_fq(&b)
                            .inverse()
                            .expect("Attempted to invert zero in P-256 base field")
                            * arr_to_fq(&a))
                        .into_bigint()
                    }
                    .0,
                );
                let c_limbs = nbiguint_to_limbs(&c_big);
                let quotient = (&b_big * &c_big).div_floor(&q_big);
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

        // load p constants into p1, p2, p3
        if self.is_scalar_field {
            // Scalar field: p = [0x0C46353D039CDAAF, 0x4319055258E8617B, 0, 0x00000000FFFFFFFF]
            // p1 = p[0], p2 = p[1], p3 = p[3]
            self.asm.emit_u::<LUI>(*self.p1, P256_NEG_N[0]);
            self.asm.emit_u::<LUI>(*self.p2, P256_NEG_N[1]);
            self.asm.emit_u::<LUI>(*self.p3, P256_NEG_N[3]);
        } else {
            // Base field: p = [1, 0xFFFFFFFF00000000, 0xFFFFFFFFFFFFFFFF, 0x00000000FFFFFFFE]
            // p[0] = 1 is implicit (handled as ADD)
            // p1 = p[1], p2 = p[2], p3 = p[3]
            self.asm.emit_u::<LUI>(*self.p1, P256_PQ[1]);
            self.asm.emit_u::<LUI>(*self.p2, P256_PQ[2]);
            self.asm.emit_u::<LUI>(*self.p3, P256_PQ[3]);
        }

        // compute ab + wp into result limbs
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

        // add w[0]*p[0] contribution to limb 0, carry into r[1]
        if self.is_scalar_field {
            // scalar field: p[0] is in p1, use MAC
            self.asm.mac_low(*self.r[1], *self.r[0], *self.w[0], *self.p1, *self.aux);
        } else {
            // base field: p[0] = 1, so w[0]*p[0] = w[0], use ADD
            self.asm.adc(*self.r[1], *self.r[0], *self.w[0]);
        }

        // if mul or square, store the lowest limb in rs3
        // if div, verify that the lowest limb matches the lowest limb of the actual argument a
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
            let mut first = true;
            let rk = *self.r[k % 2];
            let rk_next = *self.r[(k + 1) % 2];

            // ============================================================
            // w*p terms: for each (i, j) where w[i]*p[j] contributes
            // Low terms: i+j = k, i in 0..3, j in 0..3
            // High terms: i+j = k-1, i in 0..3, j in 0..3
            // ============================================================

            if self.is_scalar_field {
                // Scalar field: p = [p1, p2, 0, p3]
                // p[0] in p1, p[1] in p2, p[2] = 0 (skip), p[3] in p3

                // --- LOW terms for w[i]*p[j] where i+j = k ---

                // j=0 (p1): i = k, need k < 4
                if k < 4 {
                    self.asm.mac_low_conditional(!first, rk_next, rk, *self.w[k], *self.p1, *self.aux);
                    first = false;
                }

                // j=1 (p2): i = k-1, need k-1 in 0..3 => k in 1..4
                if k >= 1 && k - 1 < 4 {
                    self.asm.mac_low_conditional(
                        !first,
                        rk_next,
                        rk,
                        *self.w[k - 1],
                        *self.p2,
                        *self.aux,
                    );
                    first = false;
                }

                // j=2: p[2] = 0, skip

                // j=3 (p3): i = k-3, need k-3 in 0..3 => k in 3..6
                if k >= 3 && k - 3 < 4 {
                    self.asm.mac_low_conditional(
                        !first,
                        rk_next,
                        rk,
                        *self.w[k - 3],
                        *self.p3,
                        *self.aux,
                    );
                    first = false;
                }

                // --- HIGH terms for w[i]*p[j] where i+j = k-1 ---

                // j=0 (p1): i = k-1, need k-1 in 0..3 => k in 1..4
                if k >= 1 && k - 1 < 4 {
                    self.asm.mac_high_conditional(
                        !first,
                        rk_next,
                        rk,
                        *self.w[k - 1],
                        *self.p1,
                        *self.aux,
                    );
                    first = false;
                }

                // j=1 (p2): i = k-2, need k-2 in 0..3 => k in 2..5
                if k >= 2 && k - 2 < 4 {
                    self.asm.mac_high_conditional(
                        !first,
                        rk_next,
                        rk,
                        *self.w[k - 2],
                        *self.p2,
                        *self.aux,
                    );
                    first = false;
                }

                // j=2: p[2] = 0, skip

                // j=3 (p3): i = k-4, need k-4 in 0..3 => k in 4..7, capped at k<=6
                if k >= 4 && k - 4 < 4 {
                    self.asm.mac_high_conditional(
                        !first,
                        rk_next,
                        rk,
                        *self.w[k - 4],
                        *self.p3,
                        *self.aux,
                    );
                    first = false;
                }
            } else {
                // Base field: p = [1, p1, p2, p3]
                // p[0] = 1 (implicit ADD, high = 0), p[1] in p1, p[2] in p2, p[3] in p3

                // --- LOW terms for w[i]*p[j] where i+j = k ---

                // j=0 (p[0]=1): i = k, need k < 4
                // low(w[k] * 1) = w[k], use ADD
                if k < 4 {
                    self.asm.add_conditional(!first, rk_next, rk, *self.w[k], *self.aux);
                    first = false;
                }

                // j=1 (p1): i = k-1, need k-1 in 0..3 => k in 1..4
                if k >= 1 && k - 1 < 4 {
                    self.asm.mac_low_conditional(
                        !first,
                        rk_next,
                        rk,
                        *self.w[k - 1],
                        *self.p1,
                        *self.aux,
                    );
                    first = false;
                }

                // j=2 (p2): i = k-2, need k-2 in 0..3 => k in 2..5
                if k >= 2 && k - 2 < 4 {
                    self.asm.mac_low_conditional(
                        !first,
                        rk_next,
                        rk,
                        *self.w[k - 2],
                        *self.p2,
                        *self.aux,
                    );
                    first = false;
                }

                // j=3 (p3): i = k-3, need k-3 in 0..3 => k in 3..6
                if k >= 3 && k - 3 < 4 {
                    self.asm.mac_low_conditional(
                        !first,
                        rk_next,
                        rk,
                        *self.w[k - 3],
                        *self.p3,
                        *self.aux,
                    );
                    first = false;
                }

                // --- HIGH terms for w[i]*p[j] where i+j = k-1 ---

                // j=0 (p[0]=1): i = k-1
                // high(w[k-1] * 1) = 0, skip entirely

                // j=1 (p1): i = k-2, need k-2 in 0..3 => k in 2..5
                if k >= 2 && k - 2 < 4 {
                    self.asm.mac_high_conditional(
                        !first,
                        rk_next,
                        rk,
                        *self.w[k - 2],
                        *self.p1,
                        *self.aux,
                    );
                    first = false;
                }

                // j=2 (p2): i = k-3, need k-3 in 0..3 => k in 3..6
                if k >= 3 && k - 3 < 4 {
                    self.asm.mac_high_conditional(
                        !first,
                        rk_next,
                        rk,
                        *self.w[k - 3],
                        *self.p2,
                        *self.aux,
                    );
                    first = false;
                }

                // j=3 (p3): i = k-4, need k-4 in 0..3 => k in 4..7, capped at k<=6
                if k >= 4 && k - 4 < 4 {
                    self.asm.mac_high_conditional(
                        !first,
                        rk_next,
                        rk,
                        *self.w[k - 4],
                        *self.p3,
                        *self.aux,
                    );
                    first = false;
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
                                self.asm.mac_low_conditional(
                                    !first, rk_next, rk, *self.a[i], *self.a[j], *self.aux,
                                );
                                first = false;
                            } else {
                                if !first {
                                    self.asm.m2ac_low_w_carry(
                                        rk_next,
                                        rk,
                                        *self.a[i],
                                        *self.a[j],
                                        *self.aux,
                                        **self.aux2.as_ref().unwrap(),
                                    );
                                } else {
                                    self.asm.m2ac_low(rk_next, rk, *self.a[i], *self.a[j], *self.aux);
                                }
                                first = false;
                            }
                        }
                        _ => {
                            self.asm.mac_low_conditional(
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
                                self.asm.mac_high_conditional(
                                    !first, rk_next, rk, *self.a[i], *self.a[j], *self.aux,
                                );
                                first = false;
                            } else {
                                if !first {
                                    self.asm.m2ac_high_w_carry(
                                        rk_next,
                                        rk,
                                        *self.a[i],
                                        *self.a[j],
                                        *self.aux,
                                        **self.aux2.as_ref().unwrap(),
                                    );
                                } else {
                                    self.asm.m2ac_high(rk_next, rk, *self.a[i], *self.a[j], *self.aux);
                                }
                                first = false;
                            }
                        }
                        _ => {
                            self.asm.mac_high_conditional(
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
                // if div, verify that the lower limbs match the actual argument a
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

        // Special handling for top limb (k=7).
        //
        // Unlike secp256k1 (where p[3]=0), P-256 has nonzero p[3] for both
        // fields (base: 0x00000000FFFFFFFE, scalar: 0x00000000FFFFFFFF).
        // This means high(w[3]*p[3]) contributes at k=7 and must be included.
        // The review proof that this cannot overflow is in the sequence builder
        // review document: since the true value at k=7 equals w[3] < 2^64,
        // and each addend is bounded, the intermediate sums stay below 2^64.
        self.asm.emit_r::<MULHU>(*self.aux, *self.w[3], *self.p3);
        self.asm.emit_r::<ADD>(*self.r[1], *self.r[1], *self.aux);

        // high(a[3]*b[3]) or high(a[3]*a[3])
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
        // verify that w[3] matches top limb
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
        drop(self.p1);
        drop(self.p2);
        drop(self.p3);
        drop(self.aux);
        if let MulqType::Square = self.op_type {
            drop(self.aux2.unwrap())
        }
        drop(self.r);
        self.asm.finalize_inline()
    }
}

/// Virtual instruction builder for P-256 base field modular multiplication
macro_rules! p256_mulq_op {
    ($name:ident, funct3: $funct3:expr, name: $op_name:expr, mul_type: $mul_type:expr, is_scalar: $is_scalar:expr) => {
        pub struct $name;

        impl InlineOp for $name {
            const OPCODE: u32 = crate::INLINE_OPCODE;
            const FUNCT3: u32 = $funct3;
            const FUNCT7: u32 = crate::P256_FUNCT7;
            const NAME: &'static str = $op_name;

            fn build_sequence(asm: InstrAssembler, operands: FormatInline) -> Vec<Instruction> {
                P256Mulq::new(asm, operands, $mul_type, $is_scalar).inline_sequence()
            }

            fn build_advice(
                asm: InstrAssembler,
                operands: FormatInline,
                cpu: &mut Cpu,
            ) -> Option<VecDeque<u64>> {
                Some(P256Mulq::new(asm, operands, $mul_type, $is_scalar).advice(cpu))
            }
        }
    };
}

p256_mulq_op!(P256MulQ,    funct3: crate::P256_MULQ_FUNCT3,    name: crate::P256_MULQ_NAME,    mul_type: MulqType::Mul,    is_scalar: false);
p256_mulq_op!(P256SquareQ, funct3: crate::P256_SQUAREQ_FUNCT3, name: crate::P256_SQUAREQ_NAME, mul_type: MulqType::Square, is_scalar: false);
p256_mulq_op!(P256DivQ,    funct3: crate::P256_DIVQ_FUNCT3,    name: crate::P256_DIVQ_NAME,    mul_type: MulqType::Div,    is_scalar: false);
p256_mulq_op!(P256MulR,    funct3: crate::P256_MULR_FUNCT3,    name: crate::P256_MULR_NAME,    mul_type: MulqType::Mul,    is_scalar: true);
p256_mulq_op!(P256SquareR, funct3: crate::P256_SQUARER_FUNCT3, name: crate::P256_SQUARER_NAME, mul_type: MulqType::Square, is_scalar: true);
p256_mulq_op!(P256DivR,    funct3: crate::P256_DIVR_FUNCT3,    name: crate::P256_DIVR_NAME,    mul_type: MulqType::Div,    is_scalar: true);

// Fake GLV advice inline: computes s*P and half-GCD decomposition off-circuit,
// outputs (R.x, R.y, a_lo, a_hi, b_lo, b_hi, sign_b) via VirtualAdvice.
// The guest SDK verifies correctness in-circuit.

/// Inline constructor for Fake GLV scalar multiplication advice
struct FakeGlvAdvBuilder {
    asm: InstrAssembler,
    vr: VirtualRegisterGuard,
    operands: FormatInline,
}

impl FakeGlvAdvBuilder {
    fn new(asm: InstrAssembler, operands: FormatInline) -> Self {
        let vr = asm.allocator.allocate_for_inline();
        FakeGlvAdvBuilder { asm, vr, operands }
    }

    /// Advice function: runs on the host at native speed.
    /// Reads scalar s from rs1 and point P from rs2.
    /// Computes R = s*P via arkworks, then half-GCD decomposition.
    /// Returns 14 u64 values: [R.x(4), R.y(4), a_lo, a_hi, a_sign, b_lo, b_hi, b_sign]
    fn advice(self, cpu: &mut Cpu) -> VecDeque<u64> {
        // Read scalar s from rs1
        let s_addr = cpu.x[self.operands.rs1 as usize] as u64;
        let s_limbs = [
            cpu.mmu.load_doubleword(s_addr).unwrap().0,
            cpu.mmu.load_doubleword(s_addr + 8).unwrap().0,
            cpu.mmu.load_doubleword(s_addr + 16).unwrap().0,
            cpu.mmu.load_doubleword(s_addr + 24).unwrap().0,
        ];

        // Read point P from rs2 (8 u64 limbs: x then y)
        let p_addr = cpu.x[self.operands.rs2 as usize] as u64;
        let px = [
            cpu.mmu.load_doubleword(p_addr).unwrap().0,
            cpu.mmu.load_doubleword(p_addr + 8).unwrap().0,
            cpu.mmu.load_doubleword(p_addr + 16).unwrap().0,
            cpu.mmu.load_doubleword(p_addr + 24).unwrap().0,
        ];
        let py = [
            cpu.mmu.load_doubleword(p_addr + 32).unwrap().0,
            cpu.mmu.load_doubleword(p_addr + 40).unwrap().0,
            cpu.mmu.load_doubleword(p_addr + 48).unwrap().0,
            cpu.mmu.load_doubleword(p_addr + 56).unwrap().0,
        ];

        // Compute R = s * P using arkworks
        use ark_ec::CurveGroup;
        use ark_secp256r1::Projective;
        let s_fr = Fr::new(BigInt(s_limbs));
        let p_fq_x = Fq::new(BigInt(px));
        let p_fq_y = Fq::new(BigInt(py));
        let p_affine = ark_secp256r1::Affine::new(p_fq_x, p_fq_y);
        let r_proj: Projective = p_affine.into();
        let r_result = (r_proj * s_fr).into_affine();

        let rx: [u64; 4] = r_result.x.into_bigint().0;
        let ry: [u64; 4] = r_result.y.into_bigint().0;

        // Half-GCD decomposition via shared module
        let s_big: NBigInt = Fr::new(BigInt(s_limbs)).into_bigint().into();
        let [a_lo, a_hi, a_sign, b_lo, b_hi, b_sign] = crate::fake_glv::decompose_to_u64s(&s_big);

        // Output: [R.x(4), R.y(4), a_lo, a_hi, a_sign, b_lo, b_hi, b_sign] = 14 values
        let mut advice = Vec::with_capacity(14);
        advice.extend_from_slice(&rx);
        advice.extend_from_slice(&ry);
        advice.push(a_lo);
        advice.push(a_hi);
        advice.push(a_sign);
        advice.push(b_lo);
        advice.push(b_hi);
        advice.push(b_sign);

        VecDeque::from(advice)
    }

    /// Inline sequence: just write 14 VirtualAdvice values to output memory
    fn inline_sequence(mut self) -> Vec<Instruction> {
        for i in 0..14 {
            self.asm.emit_j::<VirtualAdvice>(*self.vr, 0);
            self.asm
                .emit_s::<SD>(self.operands.rs3, *self.vr, i as i64 * 8);
        }
        drop(self.vr);
        self.asm.finalize_inline()
    }
}

pub struct P256FakeGlvAdv;

impl InlineOp for P256FakeGlvAdv {
    const OPCODE: u32 = crate::INLINE_OPCODE;
    const FUNCT3: u32 = crate::P256_FAKE_GLV_ADV_FUNCT3;
    const FUNCT7: u32 = crate::P256_FUNCT7;
    const NAME: &'static str = crate::P256_FAKE_GLV_ADV_NAME;

    fn build_sequence(asm: InstrAssembler, operands: FormatInline) -> Vec<Instruction> {
        FakeGlvAdvBuilder::new(asm, operands).inline_sequence()
    }

    fn build_advice(
        asm: InstrAssembler,
        operands: FormatInline,
        cpu: &mut Cpu,
    ) -> Option<VecDeque<u64>> {
        Some(FakeGlvAdvBuilder::new(asm, operands).advice(cpu))
    }
}
