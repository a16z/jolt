#![cfg(test)]
use crate::template_format;
use common::constants::{REGISTER_COUNT, RISCV_REGISTER_COUNT};
use std::env;
use std::fmt::Write;
use tracer::{
    instruction::{
        add::ADD,
        addi::ADDI,
        addiw::ADDIW,
        addw::ADDW,
        and::AND,
        andi::ANDI,
        andn::ANDN,
        div::DIV,
        divu::DIVU,
        divuw::DIVUW,
        divw::DIVW,
        format::{
            format_i::FormatI, format_load::FormatLoad, format_r::FormatR, format_s::FormatS,
            normalize_imm,
        },
        lb::LB,
        lbu::LBU,
        ld::LD,
        lh::LH,
        lhu::LHU,
        lui::LUI,
        lw::LW,
        lwu::LWU,
        mul::MUL,
        mulh::MULH,
        mulhsu::MULHSU,
        mulhu::MULHU,
        mulw::MULW,
        ori::ORI,
        rem::REM,
        remu::REMU,
        remuw::REMUW,
        remw::REMW,
        sb::SB,
        sd::SD,
        sh::SH,
        sll::SLL,
        slli::SLLI,
        slliw::SLLIW,
        sllw::SLLW,
        sltu::SLTU,
        sra::SRA,
        srai::SRAI,
        sraiw::SRAIW,
        sraw::SRAW,
        srl::SRL,
        srli::SRLI,
        srliw::SRLIW,
        srlw::SRLW,
        sub::SUB,
        subw::SUBW,
        sw::SW,
        virtual_advice::VirtualAdvice,
        virtual_align_addr::VirtualAlignAddr,
        virtual_assert_eq::VirtualAssertEQ,
        virtual_assert_halfword_alignment::VirtualAssertHalfwordAlignment,
        virtual_assert_lte::VirtualAssertLTE,
        virtual_assert_mulu_no_overflow::VirtualAssertMulUNoOverflow,
        virtual_assert_valid_div0::VirtualAssertValidDiv0,
        virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder,
        virtual_assert_word_alignment::VirtualAssertWordAlignment,
        virtual_movsign::VirtualMovsign,
        virtual_muli::VirtualMULI,
        virtual_muliw::VirtualMULIW,
        virtual_negate_if::VirtualNegateIf,
        virtual_pext::VirtualPext,
        virtual_pext_signed::VirtualPextSigned,
        virtual_pow2::VirtualPow2,
        virtual_pow2_w::VirtualPow2W,
        virtual_shift_data_b::VirtualShiftDataB,
        virtual_shift_data_h::VirtualShiftDataH,
        virtual_shift_data_w::VirtualShiftDataW,
        virtual_shift_right_bitmask::VirtualShiftRightBitmask,
        virtual_shift_right_bitmask_w::VirtualShiftRightBitmaskW,
        virtual_sign_extend_word::VirtualSignExtendWord,
        virtual_sra::VirtualSRA,
        virtual_srai::VirtualSRAI,
        virtual_sraiw::VirtualSRAIW,
        virtual_sraw::VirtualSRAW,
        virtual_srl::VirtualSRL,
        virtual_srli::VirtualSRLI,
        virtual_srliw::VirtualSRLIW,
        virtual_srlw::VirtualSRLW,
        virtual_window_mask_b::VirtualWindowMaskB,
        virtual_window_mask_h::VirtualWindowMaskH,
        virtual_window_mask_w::VirtualWindowMaskW,
        virtual_zero_extend_word::VirtualZeroExtendWord,
        xor::XOR,
        Cycle, Instruction, RISCVCycle, RISCVInstruction, RISCVTrace,
    },
    utils::virtual_registers::VirtualRegisterAllocator,
};
use z3::{
    ast::{Array, Bool, BV},
    Params, SatResult, Solver, Sort,
};

use crate::{Z3_RANDOM_SEED, Z3_TIMEOUT_MS};

const DEFAULT_BV_BITS: u32 = 64;
const BV_BITS_ENV: &str = "Z3_VERIFIER_BV_BITS";

fn verifier_bv_bits() -> u32 {
    match env::var(BV_BITS_ENV) {
        Ok(raw) => {
            let bits: u32 = raw.parse().unwrap_or_else(|_| {
                panic!("{BV_BITS_ENV} must be an integer in [8, 64] (power of two), got {raw:?}")
            });
            if !(8..=64).contains(&bits) || !bits.is_power_of_two() {
                panic!("{BV_BITS_ENV} must be a power-of-two in [8, 64], got {bits}");
            }
            bits
        }
        Err(_) => DEFAULT_BV_BITS,
    }
}

fn scale_imm_u64(imm: u64, cpu: &SymbolicCpu) -> u64 {
    // First check boundary values
    match imm {
        0x1f => return (cpu.word_bits - 1) as u64,
        0x3f => return (cpu.bv_bits - 1) as u64,
        32 => return cpu.word_bits as u64,
        64 => return cpu.bv_bits as u64,
        _ => {}
    }

    // For power-of-2 values, scale the exponent
    if imm != 0 && (imm & (imm - 1)) == 0 {
        let shift = imm.trailing_zeros();
        let scaled_shift = match shift {
            31 => cpu.word_bits - 1,
            32 => cpu.word_bits,
            63 => cpu.bv_bits - 1,
            _ => shift & (cpu.bv_bits - 1),
        };
        return 1u64 << scaled_shift;
    }

    // Otherwise return as-is
    imm
}

#[derive(Clone)]
struct SymbolicCpu {
    var_prefix: String,
    x: [BV; REGISTER_COUNT as usize],
    /// RAM at doubleword granularity: an array from (aligned) addresses to
    /// bv_bits-wide values. The memory sequences only access the containing
    /// aligned doubleword, so keys never partially alias.
    ///
    /// WARNING: the LD/SD arms key by the raw `rs1 + imm`; the
    /// no-partial-alias invariant is guaranteed by the expansions, not the
    /// arms. `expand_amo_d` loads from a raw `rs1`, so a future AMO.D entry
    /// must revisit this model.
    mem: Array,
    advice_vars: Vec<BV>,
    asserts: Vec<Bool>,
    bv_bits: u32,
    word_bits: u32,
}

impl SymbolicCpu {
    fn new(var_prefix: &str, bv_bits: u32) -> Self {
        assert!(bv_bits.is_power_of_two() && (8..=64).contains(&bv_bits));
        assert!(bv_bits.is_multiple_of(2));
        let word_bits = bv_bits / 2;
        let regs: [BV; REGISTER_COUNT as usize] = (0..REGISTER_COUNT)
            .map(|i| BV::new_const(format!("{var_prefix}_x{i}"), bv_bits))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let asserts = vec![regs[0].eq(BV::from_u64(0, bv_bits))];
        let mem = Array::new_const(
            format!("{var_prefix}_mem"),
            &Sort::bitvector(bv_bits),
            &Sort::bitvector(bv_bits),
        );
        SymbolicCpu {
            var_prefix: var_prefix.to_string(),
            x: regs,
            mem,
            advice_vars: Vec::new(),
            asserts, // x0 is always 0
            bv_bits,
            word_bits,
        }
    }

    fn bv_u64(&self, v: u64) -> BV {
        BV::from_u64(v, self.bv_bits)
    }

    fn bv_ones(&self) -> BV {
        BV::from_u64(u64::MAX, self.bv_bits)
    }

    fn bv_zero(&self) -> BV {
        BV::from_u64(0, self.bv_bits)
    }

    fn word_u64(&self, v: u64) -> BV {
        BV::from_u64(v, self.word_bits)
    }

    fn word_ones(&self) -> BV {
        BV::from_u64(u64::MAX, self.word_bits)
    }

    fn word_extract(&self, bv: &BV) -> BV {
        bv.extract(self.word_bits - 1, 0)
    }

    fn sign_ext_word(&self, bv: &BV) -> BV {
        bv.sign_ext(self.bv_bits - self.word_bits)
    }

    fn signed_min(bits: u32) -> BV {
        assert!((1..=64).contains(&bits));
        BV::from_u64(1u64 << (bits - 1), bits)
    }

    fn sign_extend(&self, bv: &BV) -> BV {
        bv.clone()
    }

    fn unsigned_data(&self, bv: &BV) -> BV {
        bv.clone()
    }
}

fn leading_zeros(bv: &BV, bitsz: u32) -> BV {
    fn lz_recursive(bv: &BV, curr_sz: u32, bitsz: u32) -> BV {
        if curr_sz == 1 {
            return bv
                .eq(BV::from_u64(0, 1))
                .ite(&BV::from_u64(1, bitsz), &BV::from_u64(0, bitsz));
        }
        let half = curr_sz / 2;
        let lower = bv.extract(half - 1, 0);
        let upper = bv.extract(curr_sz - 1, half);
        let upper_lz = lz_recursive(&upper, curr_sz - half, bitsz);
        let lower_lz = lz_recursive(&lower, half, bitsz);
        (upper.eq(BV::from_u64(0, curr_sz - half))).ite(
            &(lower_lz + BV::from_u64((curr_sz - half) as u64, bitsz)),
            &upper_lz,
        )
    }
    lz_recursive(bv, bitsz, bitsz)
}

fn trailing_zeros(bv: &BV, bitsz: u32) -> BV {
    fn tz_recursive(bv: &BV, curr_sz: u32, bitsz: u32) -> BV {
        if curr_sz == 1 {
            return bv
                .eq(BV::from_u64(0, 1))
                .ite(&BV::from_u64(1, bitsz), &BV::from_u64(0, bitsz));
        }
        let half = curr_sz / 2;
        let lower = bv.extract(half - 1, 0);
        let upper = bv.extract(curr_sz - 1, half);
        let upper_tz = tz_recursive(&upper, half, bitsz);
        let lower_tz = tz_recursive(&lower, half, bitsz);
        (lower.eq(BV::from_u64(0, half)))
            .ite(&(upper_tz + BV::from_u64(half as u64, bitsz)), &lower_tz)
    }
    tz_recursive(bv, bitsz, bitsz)
}

fn symbolic_exec(instr: &Instruction, cpu: &mut SymbolicCpu) {
    match instr {
        Instruction::ADD(ADD { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let rs2 = cpu.x[operands.rs2 as usize].clone();
            cpu.x[operands.rd as usize] = cpu.sign_extend(&(rs1 + rs2));
        }
        Instruction::ADDI(ADDI { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let imm = normalize_imm(operands.imm);
            cpu.x[operands.rd as usize] = cpu.sign_extend(&(rs1 + imm));
        }
        Instruction::AND(AND { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let rs2 = cpu.x[operands.rs2 as usize].clone();
            cpu.x[operands.rd as usize] = cpu.sign_extend(&(rs1 & rs2));
        }
        Instruction::ANDI(ANDI { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let imm = normalize_imm(scale_imm_u64(operands.imm, cpu));
            cpu.x[operands.rd as usize] = cpu.sign_extend(&(rs1 & imm));
        }
        Instruction::ANDN(ANDN { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let rs2 = cpu.x[operands.rs2 as usize].clone();
            cpu.x[operands.rd as usize] = cpu.sign_extend(&(rs1 & rs2.bvnot()));
        }
        Instruction::LUI(LUI { operands, .. }) => {
            let imm = normalize_imm(operands.imm);
            cpu.x[operands.rd as usize] = BV::from_i64(imm, cpu.bv_bits);
        }
        Instruction::MUL(MUL { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let rs2 = cpu.x[operands.rs2 as usize].clone();
            cpu.x[operands.rd as usize] = cpu.sign_extend(&(rs1 * rs2));
        }
        Instruction::MULHU(MULHU { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let rs2 = cpu.x[operands.rs2 as usize].clone();
            let product = rs1.zero_ext(cpu.bv_bits) * rs2.zero_ext(cpu.bv_bits);
            cpu.x[operands.rd as usize] = product.extract(cpu.bv_bits * 2 - 1, cpu.bv_bits)
        }
        Instruction::ORI(ORI { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let imm = normalize_imm(scale_imm_u64(operands.imm, cpu));
            cpu.x[operands.rd as usize] = cpu.sign_extend(&(rs1 | imm));
        }
        Instruction::SUB(SUB { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let rs2 = cpu.x[operands.rs2 as usize].clone();
            cpu.x[operands.rd as usize] = cpu.sign_extend(&(rs1 - rs2));
        }
        Instruction::ADDW(ADDW { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let rs2 = cpu.x[operands.rs2 as usize].clone();
            cpu.x[operands.rd as usize] = cpu.sign_ext_word(&cpu.word_extract(&(rs1 + rs2)));
        }
        Instruction::ADDIW(ADDIW { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let imm = normalize_imm(operands.imm);
            cpu.x[operands.rd as usize] = cpu.sign_ext_word(&cpu.word_extract(&(rs1 + imm)));
        }
        Instruction::SUBW(SUBW { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let rs2 = cpu.x[operands.rs2 as usize].clone();
            cpu.x[operands.rd as usize] = cpu.sign_ext_word(&cpu.word_extract(&(rs1 - rs2)));
        }
        Instruction::MULW(MULW { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let rs2 = cpu.x[operands.rs2 as usize].clone();
            cpu.x[operands.rd as usize] = cpu.sign_ext_word(&cpu.word_extract(&(rs1 * rs2)));
        }
        Instruction::VirtualAssertEQ(VirtualAssertEQ { operands, .. }) => {
            let val1 = cpu.x[operands.rs1 as usize].clone();
            let val2 = cpu.x[operands.rs2 as usize].clone();
            cpu.asserts.push(val1.eq(&val2));
        }
        Instruction::VirtualAssertLTE(VirtualAssertLTE { operands, .. }) => {
            let val1 = cpu.x[operands.rs1 as usize].clone();
            let val2 = cpu.x[operands.rs2 as usize].clone();
            cpu.asserts.push(val1.bvule(&val2));
        }
        Instruction::VirtualAssertHalfwordAlignment(VirtualAssertHalfwordAlignment {
            operands,
            ..
        }) => {
            let addr = &cpu.x[operands.rs1 as usize] + operands.imm;
            cpu.asserts.push(addr.extract(0, 0).eq(0))
        }
        Instruction::VirtualAssertMulUNoOverflow(VirtualAssertMulUNoOverflow {
            operands, ..
        }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let rs2 = cpu.x[operands.rs2 as usize].clone();
            cpu.asserts.push(rs1.bvmul_no_overflow(&rs2, false));
        }
        Instruction::VirtualAssertValidDiv0(VirtualAssertValidDiv0 { operands, .. }) => {
            let divisor = cpu.x[operands.rs1 as usize].clone();
            let quotient = cpu.x[operands.rs2 as usize].clone();
            let ones = cpu.bv_ones();
            cpu.asserts.push(divisor.eq(0).implies(quotient.eq(&ones)));
        }
        Instruction::VirtualAssertValidUnsignedRemainder(VirtualAssertValidUnsignedRemainder {
            operands,
            ..
        }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let rs2 = cpu.x[operands.rs2 as usize].clone();
            let remainder = rs1;
            let divisor = rs2;
            cpu.asserts.push(divisor.eq(0) | remainder.bvult(&divisor));
        }

        Instruction::VirtualAssertWordAlignment(VirtualAssertWordAlignment {
            operands, ..
        }) => {
            let addr = &cpu.x[operands.rs1 as usize] + operands.imm;
            cpu.asserts.push(addr.extract(1, 0).eq(0))
        }
        Instruction::VirtualNegateIf(VirtualNegateIf { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let rs2 = cpu.x[operands.rs2 as usize].clone();
            let sign = rs1.extract(cpu.bv_bits - 1, cpu.bv_bits - 1);
            cpu.x[operands.rd as usize] = sign.eq(1).ite(&rs2.bvneg(), &rs2)
        }
        Instruction::VirtualMovsign(VirtualMovsign { operands, .. }) => {
            let val = cpu.x[operands.rs1 as usize].clone();
            let ones = cpu.bv_ones();
            let zero = cpu.bv_zero();
            let sign_bit = val.extract(cpu.bv_bits - 1, cpu.bv_bits - 1);
            cpu.x[operands.rd as usize] = sign_bit.eq(1).ite(&ones, &zero);
        }
        Instruction::VirtualAdvice(VirtualAdvice { operands, .. }) => {
            let advice_var = BV::new_const(
                format!("{}_advice_{}", cpu.var_prefix, cpu.advice_vars.len()),
                cpu.bv_bits,
            );
            cpu.x[operands.rd as usize] = advice_var.clone();
            cpu.advice_vars.push(advice_var);
        }
        Instruction::VirtualPow2(VirtualPow2 { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            cpu.x[operands.rd as usize] = cpu
                .bv_u64(1)
                .bvshl(rs1 & cpu.bv_u64((cpu.bv_bits - 1) as u64));
        }
        Instruction::VirtualPow2W(VirtualPow2W { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            cpu.x[operands.rd as usize] = cpu
                .bv_u64(1)
                .bvshl(rs1 & cpu.bv_u64((cpu.word_bits - 1) as u64));
        }
        Instruction::VirtualSRA(VirtualSRA { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let rs2 = cpu.x[operands.rs2 as usize].clone();
            let shift = trailing_zeros(&rs2, cpu.bv_bits);
            cpu.x[operands.rd as usize] = rs1.bvashr(&shift);
        }
        Instruction::VirtualSRAW(VirtualSRAW { operands, .. }) => {
            let rs1 = cpu.word_extract(&cpu.x[operands.rs1 as usize]);
            let rs2 = cpu.word_extract(&cpu.x[operands.rs2 as usize]);
            let shift = trailing_zeros(&rs2, cpu.word_bits);
            cpu.x[operands.rd as usize] = cpu.sign_ext_word(&rs1.bvashr(&shift));
        }
        Instruction::VirtualSRL(VirtualSRL { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let rs2 = cpu.x[operands.rs2 as usize].clone();
            let shift = trailing_zeros(&rs2, cpu.bv_bits);
            cpu.x[operands.rd as usize] = rs1.bvlshr(&shift);
        }
        Instruction::VirtualSRLW(VirtualSRLW { operands, .. }) => {
            let rs1 = cpu.word_extract(&cpu.x[operands.rs1 as usize]);
            let rs2 = cpu.word_extract(&cpu.x[operands.rs2 as usize]);
            let shift = trailing_zeros(&rs2, cpu.word_bits);
            cpu.x[operands.rd as usize] = cpu.sign_ext_word(&rs1.bvlshr(&shift));
        }
        Instruction::VirtualShiftRightBitmask(VirtualShiftRightBitmask { operands, .. }) => {
            let shift = cpu.x[operands.rs1 as usize].clone() & cpu.bv_u64((cpu.bv_bits - 1) as u64);
            let inv_shift: BV = cpu.bv_u64(cpu.bv_bits as u64) - &shift;
            let ones =
                (BV::from_u64(1, cpu.bv_bits * 2).bvshl(inv_shift.zero_ext(cpu.bv_bits))) - 1;
            cpu.x[operands.rd as usize] = ones
                .bvshl(shift.zero_ext(cpu.bv_bits))
                .extract(cpu.bv_bits - 1, 0)
        }
        Instruction::VirtualShiftRightBitmaskW(VirtualShiftRightBitmaskW { operands, .. }) => {
            let shift =
                cpu.x[operands.rs1 as usize].clone() & cpu.bv_u64((cpu.word_bits - 1) as u64);
            let word_bound = cpu.bv_u64(1u64 << cpu.word_bits);
            let lower_bound = cpu.bv_u64(1).bvshl(&shift);
            cpu.x[operands.rd as usize] = word_bound - lower_bound;
        }
        Instruction::LD(LD { operands, .. }) => {
            // Tracer LD truncates the immediate through i32 (see ld.rs).
            let addr = cpu.x[operands.rs1 as usize].clone() + (operands.imm as i32) as i64;
            cpu.x[operands.rd as usize] = cpu.mem.select(&addr).as_bv().unwrap();
        }
        Instruction::SD(SD { operands, .. }) => {
            let addr = cpu.x[operands.rs1 as usize].clone() + operands.imm;
            let value = cpu.x[operands.rs2 as usize].clone();
            cpu.mem = cpu.mem.store(&addr, &value);
        }
        Instruction::VirtualAlignAddr(VirtualAlignAddr { operands, .. }) => {
            // (rs1 + imm) & !7: the containing doubleword address. The mask
            // constant truncates to bv_bits, so reduced widths stay faithful.
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let imm = normalize_imm(operands.imm);
            cpu.x[operands.rd as usize] = (rs1 + imm) & cpu.bv_u64(-8i64 as u64);
        }
        Instruction::VirtualWindowMaskW(VirtualWindowMaskW { operands, .. }) => {
            let ea = cpu.x[operands.rs1 as usize].clone() + normalize_imm(operands.imm);
            let bit2 = ea.extract(2, 2).zero_ext(cpu.bv_bits - 1);
            let shift = bit2 * cpu.bv_u64(cpu.word_bits as u64);
            let word_mask = cpu.word_ones().zero_ext(cpu.bv_bits - cpu.word_bits);
            cpu.x[operands.rd as usize] = word_mask.bvshl(shift);
        }
        Instruction::VirtualWindowMaskB(VirtualWindowMaskB { operands, .. }) => {
            let ea = cpu.x[operands.rs1 as usize].clone() + normalize_imm(operands.imm);
            let offset = ea.extract(2, 0).zero_ext(cpu.bv_bits - 3);
            // One of 8 lanes of bv_bits/8 bits each; scales with the reduced
            // solver widths.
            let byte_bits = (cpu.bv_bits / 8) as u64;
            let shift = offset * cpu.bv_u64(byte_bits);
            let byte_mask = cpu.bv_u64((1u64 << byte_bits) - 1);
            cpu.x[operands.rd as usize] = byte_mask.bvshl(shift);
        }
        Instruction::VirtualWindowMaskH(VirtualWindowMaskH { operands, .. }) => {
            let ea = cpu.x[operands.rs1 as usize].clone() + normalize_imm(operands.imm);
            let offset = ea.extract(2, 1).zero_ext(cpu.bv_bits - 2);
            // One of 4 lanes of bv_bits/4 bits each; scales with the reduced
            // solver widths.
            let half_bits = (cpu.bv_bits / 4) as u64;
            let shift = offset * cpu.bv_u64(half_bits);
            let half_mask = cpu.bv_u64((1u64 << half_bits) - 1);
            cpu.x[operands.rd as usize] = half_mask.bvshl(shift);
        }
        Instruction::VirtualShiftDataB(VirtualShiftDataB { operands, .. }) => {
            // One of 8 lanes of bv_bits/8 bits each; scales with the reduced
            // solver widths.
            let byte_bits = (cpu.bv_bits / 8) as u64;
            let data = cpu.x[operands.rs1 as usize].clone() & cpu.bv_u64((1u64 << byte_bits) - 1);
            let ea = cpu.x[operands.rs2 as usize].clone();
            let offset = ea.extract(2, 0).zero_ext(cpu.bv_bits - 3);
            let shift = offset * cpu.bv_u64(byte_bits);
            cpu.x[operands.rd as usize] = data.bvshl(shift);
        }
        Instruction::VirtualShiftDataH(VirtualShiftDataH { operands, .. }) => {
            // One of 4 lanes of bv_bits/4 bits each.
            let half_bits = (cpu.bv_bits / 4) as u64;
            let data = cpu.x[operands.rs1 as usize].clone() & cpu.bv_u64((1u64 << half_bits) - 1);
            let ea = cpu.x[operands.rs2 as usize].clone();
            let offset = ea.extract(2, 1).zero_ext(cpu.bv_bits - 2);
            let shift = offset * cpu.bv_u64(half_bits);
            cpu.x[operands.rd as usize] = data.bvshl(shift);
        }
        Instruction::VirtualShiftDataW(VirtualShiftDataW { operands, .. }) => {
            // One of 2 lanes of bv_bits/2 bits each.
            let word_bits = cpu.word_bits as u64;
            let data = cpu.x[operands.rs1 as usize].clone()
                & cpu.word_ones().zero_ext(cpu.bv_bits - cpu.word_bits);
            let ea = cpu.x[operands.rs2 as usize].clone();
            let offset = ea.extract(2, 2).zero_ext(cpu.bv_bits - 1);
            let shift = offset * cpu.bv_u64(word_bits);
            cpu.x[operands.rd as usize] = data.bvshl(shift);
        }
        Instruction::VirtualPext(VirtualPext { operands, .. }) => {
            // Zero-extending extract via shift-left then logical shift-right:
            // faithful for contiguous masks (including zero), the only shape
            // the window-mask instructions produce. Any other mask havocs rd
            // with a fresh unconstrained value, so a sequence relying on
            // non-contiguous behavior fails verification instead of being
            // certified against wrong semantics. (An assert would be wrong
            // here: `cpu.asserts` are solver assumptions and would vacuously
            // exclude exactly the misuse cases.)
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let rs2 = cpu.x[operands.rs2 as usize].clone();
            let tz = trailing_zeros(&rs2, cpu.bv_bits);
            let lz = leading_zeros(&rs2, cpu.bv_bits);
            // Contiguous (or zero) mask: shifting out the trailing zeros
            // leaves a value of the form 2^k − 1.
            let normalized = rs2.bvlshr(&tz);
            let extracted = rs1.bvshl(&lz).bvlshr(lz + tz);
            let contiguous = (normalized.clone() & (normalized + cpu.bv_u64(1))).eq(cpu.bv_zero());
            let havoc = BV::fresh_const(&format!("{}_pext_nc", cpu.var_prefix), cpu.bv_bits);
            cpu.x[operands.rd as usize] = contiguous.ite(&extracted, &havoc);
        }
        Instruction::VirtualPextSigned(VirtualPextSigned { operands, .. }) => {
            // Sign-extending extract via shift-left then arithmetic
            // shift-right: faithful for contiguous masks (including zero),
            // the only shape the window-mask instructions produce. Any other
            // mask havocs rd with a fresh unconstrained value, so a sequence
            // relying on non-contiguous behavior fails verification instead
            // of being certified against wrong semantics. (An assert would be
            // wrong here: `cpu.asserts` are solver assumptions and would
            // vacuously exclude exactly the misuse cases.)
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let rs2 = cpu.x[operands.rs2 as usize].clone();
            let tz = trailing_zeros(&rs2, cpu.bv_bits);
            let lz = leading_zeros(&rs2, cpu.bv_bits);
            // Contiguous (or zero) mask: shifting out the trailing zeros
            // leaves a value of the form 2^k − 1.
            let normalized = rs2.bvlshr(&tz);
            let extracted = rs1.bvshl(&lz).bvashr(lz + tz);
            let contiguous = (normalized.clone() & (normalized + cpu.bv_u64(1))).eq(cpu.bv_zero());
            let havoc = BV::fresh_const(&format!("{}_pext_nc", cpu.var_prefix), cpu.bv_bits);
            cpu.x[operands.rd as usize] = contiguous.ite(&extracted, &havoc);
        }
        Instruction::VirtualSignExtendWord(VirtualSignExtendWord { operands, .. }) => {
            let val = cpu.x[operands.rs1 as usize].clone();
            cpu.x[operands.rd as usize] = val
                .extract(cpu.word_bits - 1, 0)
                .sign_ext(cpu.bv_bits - cpu.word_bits)
        }
        Instruction::VirtualZeroExtendWord(VirtualZeroExtendWord { operands, .. }) => {
            let val = cpu.x[operands.rs1 as usize].clone();
            cpu.x[operands.rd as usize] = val
                .extract(cpu.word_bits - 1, 0)
                .zero_ext(cpu.bv_bits - cpu.word_bits)
        }
        Instruction::VirtualMULI(VirtualMULI { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let imm = scale_imm_u64(operands.imm, cpu);
            cpu.x[operands.rd as usize] = cpu.sign_extend(&(rs1 * imm));
        }
        Instruction::VirtualMULIW(VirtualMULIW { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let imm = scale_imm_u64(operands.imm, cpu);
            cpu.x[operands.rd as usize] = cpu.sign_ext_word(&cpu.word_extract(&(rs1 * imm)));
        }
        Instruction::VirtualSRLI(VirtualSRLI { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            // Bitmask immediate: compute trailing_zeros, then scale the shift amount
            let shift_amt = operands.imm.trailing_zeros();

            // Word instructions (SRLIW, SRAIW) encode as (base_shift + 32)
            // Decompose shifts >= 32 to handle this pattern
            let scaled_shift = if shift_amt >= 32 {
                let base = (shift_amt - 32) & (cpu.word_bits - 1);
                (cpu.word_bits + base) as u64
            } else {
                // Direct scaling for regular shifts
                match shift_amt {
                    31 => (cpu.word_bits - 1) as u64,
                    _ => shift_amt as u64 & (cpu.bv_bits - 1) as u64,
                }
            };

            cpu.x[operands.rd as usize] =
                cpu.sign_extend(&cpu.unsigned_data(&rs1).bvlshr(scaled_shift));
        }
        Instruction::VirtualSRLIW(VirtualSRLIW { operands, .. }) => {
            let rs1 = cpu.word_extract(&cpu.x[operands.rs1 as usize]);
            let shift = operands.imm.trailing_zeros() & (cpu.word_bits - 1);
            cpu.x[operands.rd as usize] = cpu.sign_ext_word(&rs1.bvlshr(shift as u64));
        }
        Instruction::VirtualSRAI(VirtualSRAI { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            // Bitmask immediate: compute trailing_zeros, then scale the shift amount
            let shift_amt = operands.imm.trailing_zeros();
            let scaled_shift = match shift_amt {
                31 => (cpu.word_bits - 1) as u64,
                32 => cpu.word_bits as u64,
                63 => (cpu.bv_bits - 1) as u64,
                _ => shift_amt as u64 & (cpu.bv_bits - 1) as u64,
            };
            cpu.x[operands.rd as usize] = cpu.sign_extend(&rs1.bvashr(scaled_shift));
        }
        Instruction::VirtualSRAIW(VirtualSRAIW { operands, .. }) => {
            let rs1 = cpu.word_extract(&cpu.x[operands.rs1 as usize]);
            let shift = operands.imm.trailing_zeros() & (cpu.word_bits - 1);
            cpu.x[operands.rd as usize] = cpu.sign_ext_word(&rs1.bvashr(shift as u64));
        }
        Instruction::XOR(XOR { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let rs2 = cpu.x[operands.rs2 as usize].clone();
            cpu.x[operands.rd as usize] = cpu.sign_extend(&rs1.bvxor(&rs2));
        }
        Instruction::SLTU(SLTU { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let rs2 = cpu.x[operands.rs2 as usize].clone();
            cpu.x[operands.rd as usize] = cpu
                .unsigned_data(&rs1)
                .bvult(cpu.unsigned_data(&rs2))
                .ite(&cpu.bv_u64(1), &cpu.bv_zero());
        }
        _ => panic!("Unsupported instruction {instr:?} in symbolic_exec"),
    }
}

fn test_correctness<I: RISCVInstruction + RISCVTrace>(
    expected: impl FnOnce(&I, &mut SymbolicCpu),
    instr: &I,
) where
    RISCVCycle<I>: Into<Cycle>,
{
    let mut solver_params = Params::default();
    solver_params.set_u32("timeout", Z3_TIMEOUT_MS);
    solver_params.set_u32("random_seed", Z3_RANDOM_SEED);

    let mut solver = Solver::new();
    solver.set_params(&solver_params);
    let allocator = VirtualRegisterAllocator::default();
    let bv_bits = verifier_bv_bits();
    let mut cpu = SymbolicCpu::new("cpu1", bv_bits);

    let cpu_initial = cpu.clone();
    let mut cpu_expected = cpu.clone();
    expected(instr, &mut cpu_expected);

    let instruction: Instruction = (*instr).into();
    let seq = instruction.inline_sequence(&allocator);
    for instr in seq {
        symbolic_exec(&instr, &mut cpu);
    }

    for assert in cpu.asserts {
        solver += assert;
    }
    // Guard against vacuous proofs: the assert assumptions alone must be
    // satisfiable, or the disequality below would be refuted trivially.
    assert!(
        matches!(solver.check(), SatResult::Sat),
        "assert assumptions are unsatisfiable; the correctness proof would be vacuous"
    );

    // We don't care if virtual registers differ; memory must match.
    let registers_differ = cpu.x[..RISCV_REGISTER_COUNT as usize]
        .iter()
        .zip(cpu_expected.x[..RISCV_REGISTER_COUNT as usize].iter())
        .map(|(x1, x2)| x1.ne(x2))
        .reduce(|acc, t| acc | t)
        .unwrap();
    solver += registers_differ | cpu.mem.ne(&cpu_expected.mem);

    match solver.check() {
        SatResult::Unsat => {}
        SatResult::Sat => {
            let mut msg = "Found incorrect outputs:\n".to_string();
            let model = solver.get_model().unwrap();
            let eval = |bv: &BV| model.eval(bv, true).unwrap().as_u64().unwrap();

            let rs1 = eval(&cpu_initial.x[2]);
            let rs2 = eval(&cpu_initial.x[3]);

            let rd_val = eval(&cpu.x[1]);
            let rd_expected = eval(&cpu_expected.x[1]);

            let _ = writeln!(msg, "rs1: {rs1:#x}");
            let _ = writeln!(msg, "rs2: {rs2:#x}");

            let _ = writeln!(msg, "rd: {rd_val:#x}");
            let _ = writeln!(msg, "rd expected: {rd_expected:#x}");

            if !cpu.advice_vars.is_empty() {
                let _ = writeln!(msg, "Using advice:");
                for (i, advice_var) in cpu.advice_vars.iter().enumerate() {
                    let _ = writeln!(msg, "  {}: {:#x}", i, eval(advice_var));
                }
            }

            panic!("{}", msg.trim());
        }
        SatResult::Unknown => panic!("Solver failed/timed out, result inconclusive"),
    }
}

fn test_consistency(instr: &Instruction) {
    let mut solver_params = Params::default();
    solver_params.set_u32("timeout", Z3_TIMEOUT_MS);
    solver_params.set_u32("random_seed", Z3_RANDOM_SEED);

    let mut solver = Solver::new();
    solver.set_params(&solver_params);
    let allocator = VirtualRegisterAllocator::default();
    let bv_bits = verifier_bv_bits();
    let (mut cpu1, mut cpu2) = (
        SymbolicCpu::new("cpu1", bv_bits),
        SymbolicCpu::new("cpu2", bv_bits),
    );
    let cpu1_initial = cpu1.clone();

    for (x1, x2) in cpu1.x.iter().zip(cpu2.x.iter()) {
        solver += &x1.eq(x2);
    }
    solver += &cpu1.mem.eq(&cpu2.mem);

    let seq = instr.inline_sequence(&allocator);
    for instr in &seq {
        symbolic_exec(instr, &mut cpu1);
        symbolic_exec(instr, &mut cpu2);
    }

    for assert in cpu1.asserts.iter().chain(cpu2.asserts.iter()) {
        solver += assert;
    }
    // Guard against vacuous proofs (see test_correctness).
    assert!(
        matches!(solver.check(), SatResult::Sat),
        "assert assumptions are unsatisfiable; the consistency proof would be vacuous"
    );

    // We don't care if virtual registers differ; memory must match.
    let registers_differ = cpu1.x[..RISCV_REGISTER_COUNT as usize]
        .iter()
        .zip(cpu2.x[..RISCV_REGISTER_COUNT as usize].iter())
        .map(|(x1, x2)| x1.ne(x2))
        .reduce(|acc, t| acc | t)
        .unwrap();
    solver += registers_differ | cpu1.mem.ne(&cpu2.mem);

    match solver.check() {
        SatResult::Unsat => {}
        SatResult::Sat => {
            let mut msg = "Found differing outputs:\n".to_string();
            let operands = instr
                .try_jolt_instruction_row()
                .expect("virtual sequence verifier only formats final Jolt instructions")
                .operands;
            let model = solver.get_model().unwrap();
            let eval = |bv: &BV| model.eval(bv, true).unwrap().as_u64().unwrap();
            for i in 0..RISCV_REGISTER_COUNT as usize {
                let val1 = eval(&cpu1.x[i]);
                let val2 = eval(&cpu2.x[i]);
                if val1 != val2 {
                    let reg = if Some(i as u8) == operands.rd {
                        format!("rd (x{})", operands.rd.unwrap())
                    } else {
                        format!("x{i}")
                    };
                    let _ = writeln!(msg, "  {reg}: {val1:#x} != {val2:#x}\n");
                }
            }
            let _ = writeln!(msg, "Using inputs:");
            if let Some(rs1) = operands.rs1 {
                let _ = writeln!(
                    msg,
                    "  rs1 (x{}): {:#x}",
                    rs1,
                    eval(&cpu1_initial.x[rs1 as usize])
                );
            }
            if let Some(rs2) = operands.rs2 {
                let _ = writeln!(
                    msg,
                    "  rs2 (x{}): {:#x}",
                    rs2,
                    eval(&cpu1_initial.x[rs2 as usize])
                );
            }
            let _ = writeln!(msg, "  imm: {:#x}\n", operands.imm);

            if !cpu1.advice_vars.is_empty() {
                let _ = writeln!(msg, "Using advice:");
                for (i, (advice_var1, advice_var2)) in cpu1
                    .advice_vars
                    .iter()
                    .zip(cpu2.advice_vars.iter())
                    .enumerate()
                {
                    let val1 = eval(advice_var1);
                    let val2 = eval(advice_var2);
                    let _ = writeln!(msg, "  {i}: {val1:#x}, {val2:#x}");
                }
            }

            panic!("{}", msg.trim());
        }
        SatResult::Unknown => panic!("Solver failed/timed out, result inconclusive"),
    }
}

/// The scaled sub-word load semantics: the `eighths`-byte lane of the
/// containing doubleword at `rs1 + imm` (lanes are `bv_bits/8` wide so
/// reduced solver widths stay faithful), sign- or zero-extended.
fn lane_load(cpu: &SymbolicCpu, rs1: u8, imm: i64, eighths: u32, signed: bool) -> BV {
    let ea = cpu.x[rs1 as usize].clone() + imm;
    let aligned = ea.clone() & cpu.bv_u64(-8i64 as u64);
    let dword = cpu.mem.select(&aligned).as_bv().unwrap();
    let byte_bits = cpu.bv_bits / 8;
    let lane_bits = byte_bits * eighths;
    let offset = ea & cpu.bv_u64(8 - eighths as u64);
    let shift = offset * cpu.bv_u64(byte_bits as u64);
    let lane = dword.bvlshr(&shift).extract(lane_bits - 1, 0);
    if signed {
        lane.sign_ext(cpu.bv_bits - lane_bits)
    } else {
        lane.zero_ext(cpu.bv_bits - lane_bits)
    }
}

/// The scaled sub-word store semantics: replace the `eighths`-byte lane of
/// the containing doubleword at `rs1 + imm` with the low lane of `rs2`.
fn lane_store(cpu: &mut SymbolicCpu, rs1: u8, rs2: u8, imm: i64, eighths: u32) {
    let ea = cpu.x[rs1 as usize].clone() + imm;
    let aligned = ea.clone() & cpu.bv_u64(-8i64 as u64);
    let old = cpu.mem.select(&aligned).as_bv().unwrap();
    let byte_bits = cpu.bv_bits / 8;
    let lane_bits = byte_bits * eighths;
    let lane_ones = cpu.bv_u64(u64::MAX >> (64 - lane_bits));
    let offset = ea & cpu.bv_u64(8 - eighths as u64);
    let shift = offset * cpu.bv_u64(byte_bits as u64);
    let mask = lane_ones.clone().bvshl(&shift);
    let data = (cpu.x[rs2 as usize].clone() & lane_ones).bvshl(&shift);
    let updated = (old & mask.bvnot()) | data;
    cpu.mem = cpu.mem.store(&aligned, &updated);
}

macro_rules! test_sequence {
    ($(#[$attr:meta])* $instr:ident, $operands:path, $expected:expr $(, $field:ident : $value:expr )* $(,)?) => {
        paste::paste! {
            $(#[$attr])*
            #[test]
            #[allow(nonstandard_style)]
            fn [<test_ $instr _correctness>]() {
                let instr = $instr {
                    operands: template_format!($operands),
                    $($field: $value,)*
                    // unused by solver
                    address: 8,
                    is_compressed: false,
                    is_first_in_sequence: false,
                    virtual_sequence_remaining: None,
                };
                test_correctness($expected, &instr);
            }

            $(#[$attr])*
            #[test]
            #[allow(nonstandard_style)]
            fn [<test_ $instr _consistency>]() {
                let instr = $instr {
                    operands: template_format!($operands),
                    $($field: $value,)*
                    // unused by solver
                    address: 8,
                    is_compressed: false,
                    is_first_in_sequence: false,
                    virtual_sequence_remaining: None,
                };
                test_consistency(&Instruction::$instr(instr));
            }
        }
    };
}

test_sequence!(ADDIW, FormatI, |instr: &ADDIW, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let imm = normalize_imm(instr.operands.imm);
    cpu.x[instr.operands.rd as usize] =
        cpu.sign_ext_word(&(rs1 + imm).extract(cpu.word_bits - 1, 0));
});
test_sequence!(ADDW, FormatR, |instr: &ADDW, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let rs2 = &cpu.x[instr.operands.rs2 as usize];
    cpu.x[instr.operands.rd as usize] =
        cpu.sign_ext_word(&((rs1 + rs2).extract(cpu.word_bits - 1, 0)));
});
test_sequence!(
    #[ignore = "solver-heavy under the default 64-bit Z3 model"]
    DIV,
    FormatR,
    |instr: &DIV, cpu| {
        let rs1 = &cpu.x[instr.operands.rs1 as usize];
        let rs2 = &cpu.x[instr.operands.rs2 as usize];
        let dividend = rs1;
        let divisor = rs2;
        let ones = cpu.bv_ones();
        let min = SymbolicCpu::signed_min(cpu.bv_bits);
        cpu.x[instr.operands.rd as usize] = divisor.eq(0).ite(
            &ones,
            &(dividend.eq(&min) & divisor.eq(&ones)).ite(dividend, &(dividend.bvsdiv(divisor))),
        );
    }
);
test_sequence!(
    #[ignore = "solver-heavy under the default 64-bit Z3 model"]
    DIVU,
    FormatR,
    |instr: &DIVU, cpu| {
        let rs1 = &cpu.x[instr.operands.rs1 as usize];
        let rs2 = &cpu.x[instr.operands.rs2 as usize];
        let dividend = rs1;
        let divisor = rs2;
        let ones = cpu.bv_ones();
        cpu.x[instr.operands.rd as usize] = divisor.eq(0).ite(&ones, &(dividend.bvudiv(divisor)));
    }
);
test_sequence!(
    #[ignore = "solver-heavy under the default 64-bit Z3 model"]
    DIVUW,
    FormatR,
    |instr: &DIVUW, cpu| {
        let rs1 = &cpu.x[instr.operands.rs1 as usize];
        let rs2 = &cpu.x[instr.operands.rs2 as usize];
        let dividend = cpu.word_extract(rs1);
        let divisor = cpu.word_extract(rs2);
        let q = divisor
            .eq(0)
            .ite(&cpu.word_ones(), &(dividend.bvudiv(&divisor)));
        cpu.x[instr.operands.rd as usize] = cpu.sign_ext_word(&q);
    }
);
test_sequence!(
    #[ignore = "solver-heavy under the default 64-bit Z3 model"]
    DIVW,
    FormatR,
    |instr: &DIVW, cpu| {
        let rs1 = &cpu.x[instr.operands.rs1 as usize];
        let rs2 = &cpu.x[instr.operands.rs2 as usize];
        let dividend = cpu.word_extract(rs1);
        let divisor = cpu.word_extract(rs2);
        let word_min = SymbolicCpu::signed_min(cpu.word_bits);
        let word_ones = cpu.word_ones();
        let q = divisor.eq(0).ite(
            &word_ones,
            &(dividend.eq(&word_min) & divisor.eq(&word_ones))
                .ite(&dividend, &(dividend.bvsdiv(&divisor))),
        );
        cpu.x[instr.operands.rd as usize] = cpu.sign_ext_word(&q);
    }
);
test_sequence!(LB, FormatLoad, |instr: &LB, cpu| {
    let v = lane_load(cpu, instr.operands.rs1, instr.operands.imm, 1, true);
    cpu.x[instr.operands.rd as usize] = v;
});
test_sequence!(LBU, FormatLoad, |instr: &LBU, cpu| {
    let v = lane_load(cpu, instr.operands.rs1, instr.operands.imm, 1, false);
    cpu.x[instr.operands.rd as usize] = v;
});
test_sequence!(LH, FormatLoad, |instr: &LH, cpu| {
    let v = lane_load(cpu, instr.operands.rs1, instr.operands.imm, 2, true);
    cpu.x[instr.operands.rd as usize] = v;
});
test_sequence!(LHU, FormatLoad, |instr: &LHU, cpu| {
    let v = lane_load(cpu, instr.operands.rs1, instr.operands.imm, 2, false);
    cpu.x[instr.operands.rd as usize] = v;
});
test_sequence!(LW, FormatLoad, |instr: &LW, cpu| {
    let v = lane_load(cpu, instr.operands.rs1, instr.operands.imm, 4, true);
    cpu.x[instr.operands.rd as usize] = v;
});
test_sequence!(LWU, FormatLoad, |instr: &LWU, cpu| {
    let v = lane_load(cpu, instr.operands.rs1, instr.operands.imm, 4, false);
    cpu.x[instr.operands.rd as usize] = v;
});
test_sequence!(
    #[ignore = "solver-heavy under the default 64-bit Z3 model"]
    MULH,
    FormatR,
    |instr: &MULH, cpu| {
        let rs1 = &cpu.x[instr.operands.rs1 as usize];
        let rs2 = &cpu.x[instr.operands.rs2 as usize];
        let lhs = rs1;
        let rhs = rs2;
        let product = lhs.sign_ext(cpu.bv_bits) * rhs.sign_ext(cpu.bv_bits);
        cpu.x[instr.operands.rd as usize] = product.extract(cpu.bv_bits * 2 - 1, cpu.bv_bits);
    }
);
test_sequence!(
    #[ignore = "solver-heavy under the default 64-bit Z3 model"]
    MULHSU,
    FormatR,
    |instr: &MULHSU, cpu| {
        let rs1 = &cpu.x[instr.operands.rs1 as usize];
        let rs2 = &cpu.x[instr.operands.rs2 as usize];
        let lhs = rs1;
        let rhs = rs2;
        let product = lhs.sign_ext(cpu.bv_bits) * rhs.zero_ext(cpu.bv_bits);
        cpu.x[instr.operands.rd as usize] = product.extract(cpu.bv_bits * 2 - 1, cpu.bv_bits);
    }
);
test_sequence!(MULW, FormatR, |instr: &MULW, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let rs2 = &cpu.x[instr.operands.rs2 as usize];
    cpu.x[instr.operands.rd as usize] =
        cpu.sign_ext_word(&(cpu.word_extract(rs1) * cpu.word_extract(rs2)));
});
test_sequence!(
    #[ignore = "solver-heavy under the default 64-bit Z3 model"]
    REM,
    FormatR,
    |instr: &REM, cpu| {
        let rs1 = &cpu.x[instr.operands.rs1 as usize];
        let rs2 = &cpu.x[instr.operands.rs2 as usize];
        let dividend = rs1;
        let divisor = rs2;
        let min = SymbolicCpu::signed_min(cpu.bv_bits);
        let ones = cpu.bv_ones();
        let zero = cpu.bv_zero();
        cpu.x[instr.operands.rd as usize] = divisor.eq(0).ite(
            dividend,
            &(dividend.eq(&min) & divisor.eq(&ones)).ite(&zero, &(dividend.bvsrem(divisor))),
        );
    }
);
test_sequence!(
    #[ignore = "solver-heavy under the default 64-bit Z3 model"]
    REMU,
    FormatR,
    |instr: &REMU, cpu| {
        let rs1 = &cpu.x[instr.operands.rs1 as usize];
        let rs2 = &cpu.x[instr.operands.rs2 as usize];
        let dividend = rs1;
        let divisor = rs2;
        cpu.x[instr.operands.rd as usize] =
            divisor.eq(0).ite(dividend, &(dividend.bvurem(divisor)));
    }
);
test_sequence!(
    #[ignore = "solver-heavy under the default 64-bit Z3 model"]
    REMUW,
    FormatR,
    |instr: &REMUW, cpu| {
        let rs1 = &cpu.x[instr.operands.rs1 as usize];
        let rs2 = &cpu.x[instr.operands.rs2 as usize];
        let dividend = cpu.word_extract(rs1);
        let divisor = cpu.word_extract(rs2);
        let r = divisor.eq(0).ite(&dividend, &(dividend.bvurem(&divisor)));
        cpu.x[instr.operands.rd as usize] = cpu.sign_ext_word(&r);
    }
);
test_sequence!(
    #[ignore = "solver-heavy under the default 64-bit Z3 model"]
    REMW,
    FormatR,
    |instr: &REMW, cpu| {
        let rs1 = &cpu.x[instr.operands.rs1 as usize];
        let rs2 = &cpu.x[instr.operands.rs2 as usize];
        let dividend = cpu.word_extract(rs1);
        let divisor = cpu.word_extract(rs2);
        let word_min = SymbolicCpu::signed_min(cpu.word_bits);
        let word_ones = cpu.word_ones();
        let word_zero = cpu.word_u64(0);
        let r = divisor.eq(0).ite(
            &dividend,
            &(dividend.eq(&word_min) & divisor.eq(&word_ones))
                .ite(&word_zero, &(dividend.bvsrem(&divisor))),
        );
        cpu.x[instr.operands.rd as usize] = cpu.sign_ext_word(&r);
    }
);
test_sequence!(SB, FormatS, |instr: &SB, cpu| {
    lane_store(
        cpu,
        instr.operands.rs1,
        instr.operands.rs2,
        instr.operands.imm,
        1,
    );
});
test_sequence!(SH, FormatS, |instr: &SH, cpu| {
    lane_store(
        cpu,
        instr.operands.rs1,
        instr.operands.rs2,
        instr.operands.imm,
        2,
    );
});
test_sequence!(SW, FormatS, |instr: &SW, cpu| {
    lane_store(
        cpu,
        instr.operands.rs1,
        instr.operands.rs2,
        instr.operands.imm,
        4,
    );
});

// Negative immediates exercise the sign-extension path through the
// expansions' immediate plumbing, which the templates' imm = 1234 cannot;
// one load and one store cover the shared mechanism (the store also lands
// in a different alignment residue class).
#[test]
#[allow(nonstandard_style)]
fn test_LB_negative_imm_correctness() {
    let instr = LB {
        operands: FormatLoad {
            rd: 1,
            rs1: 2,
            imm: -8,
        },
        address: 8,
        is_compressed: false,
        is_first_in_sequence: false,
        virtual_sequence_remaining: None,
    };
    test_correctness(
        |instr: &LB, cpu| {
            let v = lane_load(cpu, instr.operands.rs1, instr.operands.imm, 1, true);
            cpu.x[instr.operands.rd as usize] = v;
        },
        &instr,
    );
}

#[test]
#[allow(nonstandard_style)]
fn test_SH_negative_imm_correctness() {
    let instr = SH {
        operands: FormatS {
            rs1: 2,
            rs2: 3,
            imm: -6,
        },
        address: 8,
        is_compressed: false,
        is_first_in_sequence: false,
        virtual_sequence_remaining: None,
    };
    test_correctness(
        |instr: &SH, cpu| {
            lane_store(
                cpu,
                instr.operands.rs1,
                instr.operands.rs2,
                instr.operands.imm,
                2,
            );
        },
        &instr,
    );
}
test_sequence!(SLL, FormatR, |instr: &SLL, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let rs2 = &cpu.x[instr.operands.rs2 as usize];
    let shift = rs2 & cpu.bv_u64((cpu.bv_bits - 1) as u64);
    cpu.x[instr.operands.rd as usize] = cpu.sign_extend(&rs1.bvshl(&shift));
});
test_sequence!(SLLI, FormatI, |instr: &SLLI, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let mask = cpu.bv_bits - 1;
    let shift = BV::from_u64(instr.operands.imm & mask as u64, cpu.bv_bits);
    cpu.x[instr.operands.rd as usize] = cpu.sign_extend(&rs1.bvshl(&shift));
});
test_sequence!(SLLIW, FormatI, |instr: &SLLIW, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let shift = BV::from_u64(
        instr.operands.imm & (cpu.word_bits - 1) as u64,
        cpu.word_bits,
    );
    cpu.x[instr.operands.rd as usize] = cpu.sign_ext_word(&cpu.word_extract(rs1).bvshl(&shift));
});
test_sequence!(SLLW, FormatR, |instr: &SLLW, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let rs2 = &cpu.x[instr.operands.rs2 as usize];
    let shift = cpu.word_extract(rs2) & cpu.word_u64((cpu.word_bits - 1) as u64);
    cpu.x[instr.operands.rd as usize] = cpu.sign_ext_word(&cpu.word_extract(rs1).bvshl(&shift));
});
test_sequence!(SRA, FormatR, |instr: &SRA, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let rs2 = &cpu.x[instr.operands.rs2 as usize];
    let shift = rs2 & cpu.bv_u64((cpu.bv_bits - 1) as u64);
    cpu.x[instr.operands.rd as usize] = cpu.sign_extend(&rs1.bvashr(&shift));
});
test_sequence!(SRAI, FormatI, |instr: &SRAI, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let mask = cpu.bv_bits - 1;
    let shift = BV::from_u64(instr.operands.imm & mask as u64, cpu.bv_bits);
    cpu.x[instr.operands.rd as usize] = cpu.sign_extend(&rs1.bvashr(&shift));
});
test_sequence!(SRAIW, FormatI, |instr: &SRAIW, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let shift = BV::from_u64(
        instr.operands.imm & (cpu.word_bits - 1) as u64,
        cpu.word_bits,
    );
    cpu.x[instr.operands.rd as usize] = cpu.sign_ext_word(&cpu.word_extract(rs1).bvashr(&shift));
});
test_sequence!(SRAW, FormatR, |instr: &SRAW, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let rs2 = &cpu.x[instr.operands.rs2 as usize];
    let shift = cpu.word_extract(rs2) & cpu.word_u64((cpu.word_bits - 1) as u64);
    cpu.x[instr.operands.rd as usize] = cpu.sign_ext_word(&cpu.word_extract(rs1).bvashr(&shift));
});
test_sequence!(SRL, FormatR, |instr: &SRL, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let rs2 = &cpu.x[instr.operands.rs2 as usize];
    let shift = rs2 & cpu.bv_u64((cpu.bv_bits - 1) as u64);
    cpu.x[instr.operands.rd as usize] = cpu.sign_extend(&cpu.unsigned_data(rs1).bvlshr(&shift));
});
test_sequence!(SRLI, FormatI, |instr: &SRLI, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let mask = cpu.bv_bits - 1;
    let shift = BV::from_u64(instr.operands.imm & mask as u64, cpu.bv_bits);
    cpu.x[instr.operands.rd as usize] = cpu.sign_extend(&cpu.unsigned_data(rs1).bvlshr(&shift));
});
test_sequence!(SRLIW, FormatI, |instr: &SRLIW, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let shift = BV::from_u64(
        instr.operands.imm & (cpu.word_bits - 1) as u64,
        cpu.word_bits,
    );
    cpu.x[instr.operands.rd as usize] = cpu.sign_ext_word(&cpu.word_extract(rs1).bvlshr(&shift));
});
test_sequence!(SRLW, FormatR, |instr: &SRLW, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let rs2 = &cpu.x[instr.operands.rs2 as usize];
    let shift = cpu.word_extract(rs2) & cpu.word_u64((cpu.word_bits - 1) as u64);
    cpu.x[instr.operands.rd as usize] = cpu.sign_ext_word(&cpu.word_extract(rs1).bvlshr(&shift));
});
test_sequence!(SUBW, FormatR, |instr: &SUBW, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let rs2 = &cpu.x[instr.operands.rs2 as usize];
    cpu.x[instr.operands.rd as usize] =
        cpu.sign_ext_word(&(cpu.word_extract(rs1) - cpu.word_extract(rs2)));
});
//test_sequence!(SW, FormatS);
