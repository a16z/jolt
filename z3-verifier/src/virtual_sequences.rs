#![cfg(test)]
use crate::template_format;
use common::constants::{REGISTER_COUNT, RISCV_REGISTER_COUNT};
use std::fmt::Write;
use tracer::{
    emulator::cpu::Xlen,
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
        format::{format_i::FormatI, format_r::FormatR, normalize_imm},
        lui::LUI,
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
        virtual_advice::VirtualAdvice,
        virtual_assert_eq::VirtualAssertEQ,
        virtual_assert_halfword_alignment::VirtualAssertHalfwordAlignment,
        virtual_assert_lte::VirtualAssertLTE,
        virtual_assert_mulu_no_overflow::VirtualAssertMulUNoOverflow,
        virtual_assert_valid_div0::VirtualAssertValidDiv0,
        virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder,
        virtual_assert_word_alignment::VirtualAssertWordAlignment,
        virtual_change_divisor::VirtualChangeDivisor,
        virtual_change_divisor_w::VirtualChangeDivisorW,
        virtual_movsign::VirtualMovsign,
        virtual_muli::VirtualMULI,
        virtual_pow2::VirtualPow2,
        virtual_pow2_w::VirtualPow2W,
        virtual_shift_right_bitmask::VirtualShiftRightBitmask,
        virtual_sign_extend_word::VirtualSignExtendWord,
        virtual_sra::VirtualSRA,
        virtual_srai::VirtualSRAI,
        virtual_srl::VirtualSRL,
        virtual_srli::VirtualSRLI,
        virtual_zero_extend_word::VirtualZeroExtendWord,
        xor::XOR,
        Cycle, Instruction, RISCVCycle, RISCVInstruction, RISCVTrace,
    },
    utils::virtual_registers::VirtualRegisterAllocator,
};
use z3::{
    ast::{Bool, BV},
    Params, SatResult, Solver,
};

const _Z3_TIMEOUT_MS: u32 = 30_000;
const Z3_RANDOM_SEED: u32 = 42;

#[derive(Clone)]
struct SymbolicCpu {
    var_prefix: String,
    x: [BV; REGISTER_COUNT as usize],
    advice_vars: Vec<BV>,
    asserts: Vec<Bool>,
    xlen: Xlen,
}

impl SymbolicCpu {
    fn new(var_prefix: &str, xlen: Xlen) -> Self {
        let regs: [BV; REGISTER_COUNT as usize] = (0..REGISTER_COUNT)
            .map(|i| BV::new_const(format!("{var_prefix}_x{i}"), 64))
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        let asserts = vec![regs[0].eq(BV::from_u64(0, 64))];
        SymbolicCpu {
            var_prefix: var_prefix.to_string(),
            x: regs,
            advice_vars: Vec::new(),
            asserts, // x0 is always 0
            xlen,
        }
    }

    fn sign_extend(&self, bv: &BV) -> BV {
        match self.xlen {
            Xlen::Bit32 => bv.extract(31, 0).sign_ext(32),
            Xlen::Bit64 => bv.clone(),
        }
    }

    fn unsigned_data(&self, bv: &BV) -> BV {
        match self.xlen {
            Xlen::Bit32 => bv.extract(31, 0).zero_ext(32),
            Xlen::Bit64 => bv.clone(),
        }
    }
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
            let imm = normalize_imm(operands.imm, &cpu.xlen);
            cpu.x[operands.rd as usize] = cpu.sign_extend(&(rs1 + imm));
        }
        Instruction::AND(AND { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let rs2 = cpu.x[operands.rs2 as usize].clone();
            cpu.x[operands.rd as usize] = cpu.sign_extend(&(rs1 & rs2));
        }
        Instruction::ANDI(ANDI { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let imm = normalize_imm(operands.imm, &cpu.xlen);
            cpu.x[operands.rd as usize] = cpu.sign_extend(&(rs1 & imm));
        }
        Instruction::ANDN(ANDN { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let rs2 = cpu.x[operands.rs2 as usize].clone();
            cpu.x[operands.rd as usize] = cpu.sign_extend(&(rs1 & rs2.bvnot()));
        }
        Instruction::LUI(LUI { operands, .. }) => {
            let imm = normalize_imm(operands.imm, &cpu.xlen);
            cpu.x[operands.rd as usize] = BV::from_i64(imm, 64);
        }
        Instruction::MUL(MUL { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let rs2 = cpu.x[operands.rs2 as usize].clone();
            cpu.x[operands.rd as usize] = cpu.sign_extend(&(rs1 * rs2));
        }
        Instruction::MULHU(MULHU { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let rs2 = cpu.x[operands.rs2 as usize].clone();
            cpu.x[operands.rd as usize] = match cpu.xlen {
                Xlen::Bit32 => {
                    let lhs = rs1.extract(31, 0);
                    let rhs = rs2.extract(31, 0);
                    let product = lhs.zero_ext(32) * rhs.zero_ext(32);
                    cpu.sign_extend(&product.extract(63, 32))
                }
                Xlen::Bit64 => {
                    let product = rs1.zero_ext(64) * rs2.zero_ext(64);
                    product.extract(127, 64)
                }
            }
        }
        Instruction::ORI(ORI { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let imm = normalize_imm(operands.imm, &cpu.xlen);
            cpu.x[operands.rd as usize] = cpu.sign_extend(&(rs1 | imm));
        }
        Instruction::SUB(SUB { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let rs2 = cpu.x[operands.rs2 as usize].clone();
            cpu.x[operands.rd as usize] = cpu.sign_extend(&(rs1 - rs2));
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
            cpu.asserts.push(divisor.eq(0).implies(match cpu.xlen {
                Xlen::Bit32 => quotient.extract(31, 0).eq(u32::MAX),
                Xlen::Bit64 => quotient.eq(u64::MAX),
            }));
        }
        Instruction::VirtualAssertValidUnsignedRemainder(VirtualAssertValidUnsignedRemainder {
            operands,
            ..
        }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let rs2 = cpu.x[operands.rs2 as usize].clone();
            cpu.asserts.push(match cpu.xlen {
                Xlen::Bit32 => {
                    let remainder = rs1.extract(31, 0);
                    let divisor = rs2.extract(31, 0);
                    divisor.eq(0) | remainder.bvult(&divisor)
                }
                Xlen::Bit64 => {
                    let remainder = rs1;
                    let divisor = rs2;
                    divisor.eq(0) | remainder.bvult(&divisor)
                }
            });
        }

        Instruction::VirtualAssertWordAlignment(VirtualAssertWordAlignment {
            operands, ..
        }) => {
            let addr = &cpu.x[operands.rs1 as usize] + operands.imm;
            cpu.asserts.push(addr.extract(1, 0).eq(0))
        }
        Instruction::VirtualChangeDivisor(VirtualChangeDivisor { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let rs2 = cpu.x[operands.rs2 as usize].clone();
            cpu.x[operands.rd as usize] = match cpu.xlen {
                Xlen::Bit32 => {
                    let dividend = rs1.extract(31, 0);
                    let divisor = rs2.extract(31, 0);
                    (dividend.eq(i32::MIN) & divisor.eq(-1))
                        .ite(&BV::from_u64(1, 64), &divisor.sign_ext(32))
                }
                Xlen::Bit64 => {
                    let dividend = rs1;
                    let divisor = rs2;
                    (dividend.eq(i64::MIN) & divisor.eq(-1)).ite(&BV::from_u64(1, 64), &divisor)
                }
            }
        }
        Instruction::VirtualChangeDivisorW(VirtualChangeDivisorW { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let rs2 = cpu.x[operands.rs2 as usize].clone();
            cpu.x[operands.rd as usize] = match cpu.xlen {
                Xlen::Bit32 => {
                    panic!("VirtualChangeDivisorW is invalid in 32b mode");
                }
                Xlen::Bit64 => {
                    let dividend = rs1.extract(31, 0);
                    let divisor = rs2.extract(31, 0);
                    (dividend.eq(i32::MIN) & divisor.eq(-1))
                        .ite(&BV::from_u64(1, 64), &divisor.sign_ext(32))
                }
            }
        }
        Instruction::VirtualMovsign(VirtualMovsign { operands, .. }) => {
            let val = cpu.x[operands.rs1 as usize].clone();
            cpu.x[operands.rd as usize] = match cpu.xlen {
                Xlen::Bit32 => {
                    let sign_bit = val.extract(31, 31);
                    sign_bit
                        .eq(1)
                        .ite(&BV::from_u64(u32::MAX as u64, 64), &BV::from_u64(0, 64))
                }
                Xlen::Bit64 => {
                    let sign_bit = val.extract(63, 63);
                    sign_bit
                        .eq(1)
                        .ite(&BV::from_u64(u64::MAX, 64), &BV::from_u64(0, 64))
                }
            };
        }
        Instruction::VirtualAdvice(VirtualAdvice { operands, .. }) => {
            let advice_var = BV::new_const(
                format!("{}_advice_{}", cpu.var_prefix, cpu.advice_vars.len()),
                64,
            );
            cpu.x[operands.rd as usize] = advice_var.clone();
            cpu.advice_vars.push(advice_var);
        }
        Instruction::VirtualPow2(VirtualPow2 { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            cpu.x[operands.rd as usize] = match cpu.xlen {
                Xlen::Bit32 => BV::from_u64(1, 64).bvshl(rs1 & (32 - 1)),
                Xlen::Bit64 => BV::from_u64(1, 64).bvshl(rs1 & (64 - 1)),
            };
        }
        Instruction::VirtualPow2W(VirtualPow2W { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            cpu.x[operands.rd as usize] = match cpu.xlen {
                Xlen::Bit32 => panic!("VirtualPow2W is invalid in 32b mode"),
                Xlen::Bit64 => BV::from_u64(1, 64).bvshl(rs1 & (32 - 1)),
            };
        }
        Instruction::VirtualSRA(VirtualSRA { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let rs2 = cpu.x[operands.rs2 as usize].clone();
            let shift = trailing_zeros(&rs2, 64);
            cpu.x[operands.rd as usize] = rs1.bvashr(&shift);
        }
        Instruction::VirtualSRL(VirtualSRL { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let rs2 = cpu.x[operands.rs2 as usize].clone();
            let shift = trailing_zeros(&rs2, 64);
            cpu.x[operands.rd as usize] = rs1.bvlshr(&shift);
        }
        Instruction::VirtualShiftRightBitmask(VirtualShiftRightBitmask { operands, .. }) => {
            cpu.x[operands.rd as usize] = match cpu.xlen {
                Xlen::Bit32 => {
                    let shift = cpu.x[operands.rs1 as usize].clone() & (32 - 1);
                    let ones = (BV::from_u64(1, 64).bvshl(&(32 - &shift))) - 1;
                    ones.bvshl(&shift)
                }
                Xlen::Bit64 => {
                    let shift = cpu.x[operands.rs1 as usize].clone() & (64 - 1);
                    let inv_shift: BV = 64 - &shift;
                    let ones = (BV::from_u64(1, 128).bvshl(inv_shift.zero_ext(64))) - 1;
                    ones.bvshl(shift.zero_ext(64)).extract(63, 0)
                }
            }
        }
        Instruction::VirtualSignExtendWord(VirtualSignExtendWord { operands, .. }) => {
            cpu.x[operands.rd as usize] = match cpu.xlen {
                Xlen::Bit32 => panic!("VirtualSignExtendWord is not supported for 32-bit mode"),
                Xlen::Bit64 => {
                    let val = cpu.x[operands.rs1 as usize].clone();
                    val.extract(31, 0).sign_ext(32)
                }
            }
        }
        Instruction::VirtualZeroExtendWord(VirtualZeroExtendWord { operands, .. }) => {
            cpu.x[operands.rd as usize] = match cpu.xlen {
                Xlen::Bit32 => panic!("VirtualExtend is not supported for 32-bit mode"),
                Xlen::Bit64 => {
                    let val = cpu.x[operands.rs1 as usize].clone();
                    val.extract(31, 0).zero_ext(32)
                }
            }
        }
        Instruction::VirtualMULI(VirtualMULI { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            cpu.x[operands.rd as usize] = cpu.sign_extend(&(rs1 * operands.imm));
        }
        Instruction::VirtualSRLI(VirtualSRLI { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let shift = operands.imm.trailing_zeros();
            cpu.x[operands.rd as usize] = cpu.sign_extend(&cpu.unsigned_data(&rs1).bvlshr(shift));
        }
        Instruction::VirtualSRAI(VirtualSRAI { operands, .. }) => {
            let rs1 = cpu.x[operands.rs1 as usize].clone();
            let shift = operands.imm.trailing_zeros();
            cpu.x[operands.rd as usize] = cpu.sign_extend(&rs1.bvashr(shift));
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
                .ite(&BV::from_u64(1, 64), &BV::from_u64(0, 64));
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
    //solver_params.set_u32("timeout", Z3_TIMEOUT_MS);
    solver_params.set_u32("random_seed", Z3_RANDOM_SEED);

    let mut solver = Solver::new();
    solver.set_params(&solver_params);
    let allocator = VirtualRegisterAllocator::default();
    let xlen = Xlen::Bit64;
    let mut cpu = SymbolicCpu::new("cpu1", xlen);

    let cpu_initial = cpu.clone();
    let mut cpu_expected = cpu.clone();
    expected(instr, &mut cpu_expected);

    for instr in instr.inline_sequence(&allocator, xlen) {
        symbolic_exec(&instr, &mut cpu);
    }

    for assert in cpu.asserts {
        solver += assert;
    }

    // We don't care if virtual registers differ
    solver += cpu.x[..RISCV_REGISTER_COUNT as usize]
        .iter()
        .zip(cpu_expected.x[..RISCV_REGISTER_COUNT as usize].iter())
        .map(|(x1, x2)| x1.ne(x2))
        .reduce(|acc, t| acc | t)
        .unwrap();

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
    //solver_params.set_u32("timeout", Z3_TIMEOUT_MS);
    solver_params.set_u32("random_seed", Z3_RANDOM_SEED);

    let mut solver = Solver::new();
    solver.set_params(&solver_params);
    let allocator = VirtualRegisterAllocator::default();
    let xlen = Xlen::Bit64;
    let (mut cpu1, mut cpu2) = (
        SymbolicCpu::new("cpu1", xlen),
        SymbolicCpu::new("cpu2", xlen),
    );
    let cpu1_initial = cpu1.clone();

    for (x1, x2) in cpu1.x.iter().zip(cpu2.x.iter()) {
        solver += &x1.eq(x2);
    }

    for instr in instr.inline_sequence(&allocator, xlen) {
        symbolic_exec(&instr, &mut cpu1);
        symbolic_exec(&instr, &mut cpu2);
    }

    for assert in cpu1.asserts.iter().chain(cpu2.asserts.iter()) {
        solver += assert;
    }

    // We don't care if virtual registers differ
    solver += cpu1.x[..RISCV_REGISTER_COUNT as usize]
        .iter()
        .zip(cpu2.x[..RISCV_REGISTER_COUNT as usize].iter())
        .map(|(x1, x2)| x1.ne(x2))
        .reduce(|acc, t| acc | t)
        .unwrap();

    match solver.check() {
        SatResult::Unsat => {}
        SatResult::Sat => {
            let mut msg = "Found differing outputs:\n".to_string();
            let operands = instr.normalize().operands;
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

macro_rules! test_sequence {
    ($instr:ident, $operands:path, $expected:expr $(, $field:ident : $value:expr )* $(,)?) => {
        paste::paste! {
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
    let imm = normalize_imm(instr.operands.imm, &cpu.xlen);
    cpu.x[instr.operands.rd as usize] = (rs1 + imm).extract(31, 0).sign_ext(32);
});
test_sequence!(ADDW, FormatR, |instr: &ADDW, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let rs2 = &cpu.x[instr.operands.rs2 as usize];
    cpu.x[instr.operands.rd as usize] = ((rs1 + rs2).extract(31, 0)).sign_ext(32);
});
test_sequence!(DIV, FormatR, |instr: &DIV, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let rs2 = &cpu.x[instr.operands.rs2 as usize];
    cpu.x[instr.operands.rd as usize] = match cpu.xlen {
        Xlen::Bit32 => {
            todo!()
        }
        Xlen::Bit64 => {
            let dividend = rs1;
            let divisor = rs2;
            divisor.eq(0).ite(
                &BV::from_i64(-1, 64),
                &(dividend.eq(i64::MIN) & divisor.eq(-1))
                    .ite(dividend, &(dividend.bvsdiv(divisor))),
            )
        }
    };
});
test_sequence!(DIVU, FormatR, |instr: &DIVU, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let rs2 = &cpu.x[instr.operands.rs2 as usize];
    cpu.x[instr.operands.rd as usize] = match cpu.xlen {
        Xlen::Bit32 => {
            todo!()
        }
        Xlen::Bit64 => {
            let dividend = rs1;
            let divisor = rs2;
            divisor
                .eq(0)
                .ite(&BV::from_u64(u64::MAX, 64), &(dividend.bvudiv(divisor)))
        }
    };
});
test_sequence!(DIVUW, FormatR, |instr: &DIVUW, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let rs2 = &cpu.x[instr.operands.rs2 as usize];
    cpu.x[instr.operands.rd as usize] = match cpu.xlen {
        Xlen::Bit32 => {
            panic!("DIVUW is invalid in 32b mode");
        }
        Xlen::Bit64 => {
            let dividend = rs1.extract(31, 0);
            let divisor = rs2.extract(31, 0);
            divisor
                .eq(0)
                .ite(
                    &BV::from_u64(u32::MAX as u64, 32),
                    &(dividend.bvudiv(&divisor)),
                )
                .sign_ext(32)
        }
    };
});
test_sequence!(DIVW, FormatR, |instr: &DIVW, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let rs2 = &cpu.x[instr.operands.rs2 as usize];
    cpu.x[instr.operands.rd as usize] = match cpu.xlen {
        Xlen::Bit32 => {
            panic!("DIVW is invalid in 32b mode");
        }
        Xlen::Bit64 => {
            let dividend = rs1.extract(31, 0);
            let divisor = rs2.extract(31, 0);
            divisor
                .eq(0)
                .ite(
                    &BV::from_i64(-1, 32),
                    &(dividend.eq(i32::MIN as u32) & divisor.eq(-1))
                        .ite(&dividend, &(dividend.bvsdiv(&divisor))),
                )
                .sign_ext(32)
        }
    };
});
// Memory operations are not tested at the moment
// test_sequence!(LB, FormatLoad);
// test_sequence!(LBU, FormatLoad);
// test_sequence!(LH, FormatLoad);
// test_sequence!(LHU, FormatLoad);
// test_sequence!(LW, FormatLoad);
// test_sequence!(LWU, FormatLoad);
test_sequence!(MULH, FormatR, |instr: &MULH, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let rs2 = &cpu.x[instr.operands.rs2 as usize];
    cpu.x[instr.operands.rd as usize] = match cpu.xlen {
        Xlen::Bit32 => {
            let lhs = rs1.extract(31, 0);
            let rhs = rs2.extract(31, 0);
            let product = lhs.sign_ext(32) * rhs.sign_ext(32);
            cpu.sign_extend(&product.extract(63, 32))
        }
        Xlen::Bit64 => {
            let lhs = rs1;
            let rhs = rs2;
            let product = lhs.sign_ext(64) * rhs.sign_ext(64);
            product.extract(127, 64)
        }
    };
});
test_sequence!(MULHSU, FormatR, |instr: &MULHSU, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let rs2 = &cpu.x[instr.operands.rs2 as usize];
    cpu.x[instr.operands.rd as usize] = match cpu.xlen {
        Xlen::Bit32 => {
            let lhs = rs1.extract(31, 0);
            let rhs = rs2.extract(31, 0);
            let product = lhs.sign_ext(32) * rhs.zero_ext(32);
            cpu.sign_extend(&product.extract(63, 32))
        }
        Xlen::Bit64 => {
            let lhs = rs1;
            let rhs = rs2;
            let product = lhs.sign_ext(64) * rhs.zero_ext(64);
            product.extract(127, 64)
        }
    };
});
test_sequence!(MULW, FormatR, |instr: &MULW, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let rs2 = &cpu.x[instr.operands.rs2 as usize];
    cpu.x[instr.operands.rd as usize] = (rs1.extract(31, 0) * rs2.extract(31, 0)).sign_ext(32);
});
test_sequence!(REM, FormatR, |instr: &REM, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let rs2 = &cpu.x[instr.operands.rs2 as usize];
    cpu.x[instr.operands.rd as usize] = match cpu.xlen {
        Xlen::Bit32 => {
            todo!()
        }
        Xlen::Bit64 => {
            let dividend = rs1;
            let divisor = rs2;
            divisor.eq(0).ite(
                dividend,
                &(dividend.eq(i64::MIN) & divisor.eq(-1))
                    .ite(&BV::from_i64(0, 64), &(dividend.bvsrem(divisor))),
            )
        }
    };
});
test_sequence!(REMU, FormatR, |instr: &REMU, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let rs2 = &cpu.x[instr.operands.rs2 as usize];
    cpu.x[instr.operands.rd as usize] = match cpu.xlen {
        Xlen::Bit32 => {
            todo!()
        }
        Xlen::Bit64 => {
            let dividend = rs1;
            let divisor = rs2;
            divisor.eq(0).ite(dividend, &(dividend.bvurem(divisor)))
        }
    };
});
test_sequence!(REMUW, FormatR, |instr: &REMUW, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let rs2 = &cpu.x[instr.operands.rs2 as usize];
    cpu.x[instr.operands.rd as usize] = match cpu.xlen {
        Xlen::Bit32 => {
            panic!("REMUW is invalid in 32b mode");
        }
        Xlen::Bit64 => {
            let dividend = rs1.extract(31, 0);
            let divisor = rs2.extract(31, 0);
            divisor
                .eq(0)
                .ite(&dividend, &(dividend.bvurem(&divisor)))
                .sign_ext(32)
        }
    };
});
test_sequence!(REMW, FormatR, |instr: &REMW, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let rs2 = &cpu.x[instr.operands.rs2 as usize];
    cpu.x[instr.operands.rd as usize] = match cpu.xlen {
        Xlen::Bit32 => {
            panic!("REMW is invalid in 32b mode");
        }
        Xlen::Bit64 => {
            let dividend = rs1.extract(31, 0);
            let divisor = rs2.extract(31, 0);
            divisor
                .eq(0)
                .ite(
                    &dividend,
                    &(dividend.eq(i32::MIN as u32) & divisor.eq(-1))
                        .ite(&BV::from_i64(0, 32), &(dividend.bvsrem(&divisor))),
                )
                .sign_ext(32)
        }
    };
});
// test_sequence!(SB, FormatS);
// test_sequence!(SH, FormatS);
test_sequence!(SLL, FormatR, |instr: &SLL, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let rs2 = &cpu.x[instr.operands.rs2 as usize];
    let shift = rs2
        & match cpu.xlen {
            Xlen::Bit32 => BV::from_u64(32 - 1, 64),
            Xlen::Bit64 => BV::from_u64(64 - 1, 64),
        };
    cpu.x[instr.operands.rd as usize] = cpu.sign_extend(&rs1.bvshl(&shift));
});
test_sequence!(SLLI, FormatI, |instr: &SLLI, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let shift = BV::from_u64(
        instr.operands.imm
            & match cpu.xlen {
                Xlen::Bit32 => 32 - 1,
                Xlen::Bit64 => 64 - 1,
            },
        64,
    );
    cpu.x[instr.operands.rd as usize] = cpu.sign_extend(&rs1.bvshl(&shift));
});
test_sequence!(SLLIW, FormatI, |instr: &SLLIW, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let shift = BV::from_u64(instr.operands.imm & (32 - 1), 32);
    cpu.x[instr.operands.rd as usize] = (rs1.extract(31, 0).bvshl(&shift)).sign_ext(32);
});
test_sequence!(SLLW, FormatR, |instr: &SLLW, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let rs2 = &cpu.x[instr.operands.rs2 as usize];
    let shift = rs2.extract(31, 0) & BV::from_u64(32 - 1, 32);
    cpu.x[instr.operands.rd as usize] = (rs1.extract(31, 0).bvshl(&shift)).sign_ext(32);
});
test_sequence!(SRA, FormatR, |instr: &SRA, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let rs2 = &cpu.x[instr.operands.rs2 as usize];
    let shift = rs2
        & match cpu.xlen {
            Xlen::Bit32 => BV::from_u64(32 - 1, 64),
            Xlen::Bit64 => BV::from_u64(64 - 1, 64),
        };
    cpu.x[instr.operands.rd as usize] = cpu.sign_extend(&rs1.bvashr(&shift));
});
test_sequence!(SRAI, FormatI, |instr: &SRAI, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let shift = BV::from_u64(
        instr.operands.imm
            & match cpu.xlen {
                Xlen::Bit32 => 32 - 1,
                Xlen::Bit64 => 64 - 1,
            },
        64,
    );
    cpu.x[instr.operands.rd as usize] = cpu.sign_extend(&rs1.bvashr(&shift));
});
test_sequence!(SRAIW, FormatI, |instr: &SRAIW, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let shift = BV::from_u64(instr.operands.imm & (32 - 1), 32);
    cpu.x[instr.operands.rd as usize] = (rs1.extract(31, 0).bvashr(&shift)).sign_ext(32);
});
test_sequence!(SRAW, FormatR, |instr: &SRAW, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let rs2 = &cpu.x[instr.operands.rs2 as usize];
    let shift = rs2.extract(31, 0) & BV::from_u64(32 - 1, 32);
    cpu.x[instr.operands.rd as usize] = (rs1.extract(31, 0).bvashr(&shift)).sign_ext(32);
});
test_sequence!(SRL, FormatR, |instr: &SRL, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let rs2 = &cpu.x[instr.operands.rs2 as usize];
    let shift = rs2
        & match cpu.xlen {
            Xlen::Bit32 => BV::from_u64(32 - 1, 64),
            Xlen::Bit64 => BV::from_u64(64 - 1, 64),
        };
    cpu.x[instr.operands.rd as usize] = cpu.sign_extend(&cpu.unsigned_data(rs1).bvlshr(&shift));
});
test_sequence!(SRLI, FormatI, |instr: &SRLI, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let shift = BV::from_u64(
        instr.operands.imm
            & match cpu.xlen {
                Xlen::Bit32 => 32 - 1,
                Xlen::Bit64 => 64 - 1,
            },
        64,
    );
    cpu.x[instr.operands.rd as usize] = cpu.sign_extend(&cpu.unsigned_data(rs1).bvlshr(&shift));
});
test_sequence!(SRLIW, FormatI, |instr: &SRLIW, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let shift = BV::from_u64(instr.operands.imm & (32 - 1), 32);
    cpu.x[instr.operands.rd as usize] =
        (cpu.unsigned_data(&rs1.extract(31, 0)).bvlshr(&shift)).sign_ext(32);
});
test_sequence!(SRLW, FormatR, |instr: &SRLW, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let rs2 = &cpu.x[instr.operands.rs2 as usize];
    let shift = rs2.extract(31, 0) & BV::from_u64(32 - 1, 32);
    cpu.x[instr.operands.rd as usize] =
        (cpu.unsigned_data(&rs1.extract(31, 0)).bvlshr(&shift)).sign_ext(32);
});
test_sequence!(SUBW, FormatR, |instr: &SUBW, cpu| {
    let rs1 = &cpu.x[instr.operands.rs1 as usize];
    let rs2 = &cpu.x[instr.operands.rs2 as usize];
    cpu.x[instr.operands.rd as usize] = (rs1.extract(31, 0) - rs2.extract(31, 0)).sign_ext(32);
});
//test_sequence!(SW, FormatS);
