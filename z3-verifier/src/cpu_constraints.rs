#![cfg(test)]
#![allow(non_upper_case_globals)]

use jolt_core::zkvm::{
    instruction::{
        CircuitFlags, Flags, InstructionFlags, NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS,
    },
    r1cs::{
        constraints::{ProductFactorExpr, PRODUCT_CONSTRAINTS, R1CS_CONSTRAINTS},
        inputs::NUM_R1CS_INPUTS,
        ops::LC,
    },
    witness::VirtualPolynomial,
};
use std::{array, fmt::Write, str::FromStr};
use tracer::instruction::{
    add::ADD,
    addi::ADDI,
    and::AND,
    andi::ANDI,
    andn::ANDN,
    auipc::AUIPC,
    beq::BEQ,
    bge::BGE,
    bgeu::BGEU,
    blt::BLT,
    bltu::BLTU,
    bne::BNE,
    ecall::ECALL,
    fence::FENCE,
    format::{
        format_assert_align::AssertAlignFormat, format_b::FormatB, format_i::FormatI,
        format_j::FormatJ, format_load::FormatLoad, format_r::FormatR, format_s::FormatS,
        format_u::FormatU, format_virtual_right_shift_i::FormatVirtualRightShiftI,
        format_virtual_right_shift_r::FormatVirtualRightShiftR,
    },
    jal::JAL,
    jalr::JALR,
    ld::LD,
    lui::LUI,
    mul::MUL,
    mulhu::MULHU,
    or::OR,
    ori::ORI,
    sd::SD,
    slt::SLT,
    slti::SLTI,
    sltiu::SLTIU,
    sltu::SLTU,
    sub::SUB,
    virtual_advice::VirtualAdvice,
    virtual_assert_eq::VirtualAssertEQ,
    virtual_assert_halfword_alignment::VirtualAssertHalfwordAlignment,
    virtual_assert_lte::VirtualAssertLTE,
    virtual_assert_mulu_no_overflow::VirtualAssertMulUNoOverflow,
    virtual_assert_valid_div0::VirtualAssertValidDiv0,
    virtual_assert_valid_unsigned_remainder::VirtualAssertValidUnsignedRemainder,
    virtual_assert_word_alignment::VirtualAssertWordAlignment,
    virtual_change_divisor::VirtualChangeDivisor,
    virtual_movsign::VirtualMovsign,
    virtual_muli::VirtualMULI,
    virtual_pow2::VirtualPow2,
    virtual_pow2_w::VirtualPow2W,
    virtual_pow2i::VirtualPow2I,
    virtual_pow2i_w::VirtualPow2IW,
    virtual_rev8w::VirtualRev8W,
    virtual_rotriw::VirtualROTRIW,
    virtual_shift_right_bitmask::VirtualShiftRightBitmask,
    virtual_shift_right_bitmaski::VirtualShiftRightBitmaskI,
    virtual_sign_extend_word::VirtualSignExtendWord,
    virtual_sra::VirtualSRA,
    virtual_srai::VirtualSRAI,
    virtual_srl::VirtualSRL,
    virtual_srli::VirtualSRLI,
    Instruction,
};
use z3::{ast::Int, SatResult, Solver};

#[derive(Clone, Debug)]
struct JoltState<T = Int> {
    // R1CS cycle inputs
    left_input: T,
    right_input: T,
    product: T,
    left_lookup: T,
    right_lookup: T,
    lookup_output: T,
    rs1_value: T,
    rs2_value: T,
    rd_write_value: T,
    ram_addr: T,
    ram_read_value: T,
    ram_write_value: T,
    pc: T,
    next_pc: T,
    unexpanded_pc: T,
    next_unexpanded_pc: T,
    imm: T,
    flags: [T; NUM_CIRCUIT_FLAGS],
    instruction_flags: [T; NUM_INSTRUCTION_FLAGS],
    next_is_noop: T,
    should_jump: T,
    should_branch: T,
    write_lookup_output_to_rd: T,
    write_pc_to_rd: T,
    next_is_virtual: T,
    next_is_first_in_sequence: T,
}

impl JoltState {
    fn new(prefix: String) -> Self {
        JoltState {
            left_input: Int::new_const(format!("{prefix}_left_input")),
            right_input: Int::new_const(format!("{prefix}_right_input")),
            product: Int::new_const(format!("{prefix}_product")),
            left_lookup: Int::new_const(format!("{prefix}_left_lookup")),
            right_lookup: Int::new_const(format!("{prefix}_right_lookup")),
            lookup_output: Int::new_const(format!("{prefix}_lookup_output")),
            rs1_value: Int::new_const(format!("{prefix}_rs1_read_value")),
            rs2_value: Int::new_const(format!("{prefix}_rs2_read_value")),
            rd_write_value: Int::new_const(format!("{prefix}_rd_write_value")),
            ram_addr: Int::new_const(format!("{prefix}_ram_addr")),
            ram_read_value: Int::new_const(format!("{prefix}_ram_read_value")),
            ram_write_value: Int::new_const(format!("{prefix}_ram_write_value")),
            pc: Int::new_const(format!("{prefix}_pc")),
            next_pc: Int::new_const(format!("{prefix}_next_pc")),
            unexpanded_pc: Int::new_const(format!("{prefix}_unexpanded_pc")),
            next_unexpanded_pc: Int::new_const(format!("{prefix}_next_unexpanded_pc")),
            imm: Int::new_const(format!("{prefix}_imm")),
            flags: array::from_fn(|i| Int::new_const(format!("{prefix}_flag_{i}"))),
            instruction_flags: array::from_fn(|i| {
                Int::new_const(format!("{prefix}_instr_flag_{i}"))
            }),
            next_is_noop: Int::new_const(format!("{prefix}_next_is_noop")),
            should_jump: Int::new_const(format!("{prefix}_should_jump")),
            should_branch: Int::new_const(format!("{prefix}_should_branch")),
            write_lookup_output_to_rd: Int::new_const(format!(
                "{prefix}_write_lookup_output_to_rd_addr"
            )),
            write_pc_to_rd: Int::new_const(format!("{prefix}_write_pc_to_rd_addr")),
            next_is_virtual: Int::new_const(format!("{prefix}_next_is_virtual")),
            next_is_first_in_sequence: Int::new_const(format!(
                "{prefix}_next_is_first_in_sequence"
            )),
        }
    }

    fn r1cs_inputs(&self) -> [&Int; NUM_R1CS_INPUTS] {
        [
            &self.left_input,
            &self.right_input,
            &self.product,
            &self.write_lookup_output_to_rd,
            &self.write_pc_to_rd,
            &self.should_branch,
            &self.pc,
            &self.unexpanded_pc,
            &self.imm,
            &self.ram_addr,
            &self.rs1_value,
            &self.rs2_value,
            &self.rd_write_value,
            &self.ram_read_value,
            &self.ram_write_value,
            &self.left_lookup,
            &self.right_lookup,
            &self.next_unexpanded_pc,
            &self.next_pc,
            &self.next_is_virtual,
            &self.next_is_first_in_sequence,
            &self.lookup_output,
            &self.should_jump,
            &self.flags[CircuitFlags::AddOperands as usize],
            &self.flags[CircuitFlags::SubtractOperands as usize],
            &self.flags[CircuitFlags::MultiplyOperands as usize],
            &self.flags[CircuitFlags::Load as usize],
            &self.flags[CircuitFlags::Store as usize],
            &self.flags[CircuitFlags::Jump as usize],
            &self.flags[CircuitFlags::WriteLookupOutputToRD as usize],
            &self.flags[CircuitFlags::VirtualInstruction as usize],
            &self.flags[CircuitFlags::Assert as usize],
            &self.flags[CircuitFlags::DoNotUpdateUnexpandedPC as usize],
            &self.flags[CircuitFlags::Advice as usize],
            &self.flags[CircuitFlags::IsCompressed as usize],
            &self.flags[CircuitFlags::IsFirstInSequence as usize],
        ]
    }

    fn lc_to_int(&self, lc: &LC) -> Int {
        let mut result = lc
            .const_term()
            .map(|c| Int::from_str(&c.to_string()).unwrap())
            .unwrap_or(Int::from_i64(0));
        lc.for_each_term(|idx, coeff| {
            let coeff: Int = Int::from_str(&coeff.to_string()).unwrap();
            result += coeff * self.r1cs_inputs()[idx];
        });
        result
    }

    fn add_r1cs_constraints(&self, solver: &mut Solver) {
        R1CS_CONSTRAINTS.iter().for_each(|c| {
            let lhs = self.lc_to_int(&c.cons.a) * self.lc_to_int(&c.cons.b);
            *solver += lhs.eq(Int::from(0));
        });
    }

    fn virtpoly_to_int(&self, poly: &VirtualPolynomial) -> &Int {
        match poly {
            VirtualPolynomial::LeftInstructionInput => &self.left_input,
            VirtualPolynomial::RightInstructionInput => &self.right_input,
            VirtualPolynomial::Product => &self.product,
            VirtualPolynomial::InstructionFlags(InstructionFlags::IsRdNotZero) => {
                &self.instruction_flags[InstructionFlags::IsRdNotZero as usize]
            }
            VirtualPolynomial::OpFlags(CircuitFlags::WriteLookupOutputToRD) => {
                &self.flags[CircuitFlags::WriteLookupOutputToRD as usize]
            }
            VirtualPolynomial::WriteLookupOutputToRD => &self.write_lookup_output_to_rd,
            VirtualPolynomial::OpFlags(CircuitFlags::Jump) => {
                &self.flags[CircuitFlags::Jump as usize]
            }
            VirtualPolynomial::WritePCtoRD => &self.write_pc_to_rd,
            VirtualPolynomial::LookupOutput => &self.lookup_output,
            VirtualPolynomial::InstructionFlags(InstructionFlags::Branch) => {
                &self.instruction_flags[InstructionFlags::Branch as usize]
            }
            VirtualPolynomial::ShouldBranch => &self.should_branch,
            VirtualPolynomial::NextIsNoop => &self.next_is_noop,
            VirtualPolynomial::ShouldJump => &self.should_jump,
            _ => unreachable!(),
        }
    }

    fn prodfac_to_int(&self, pf: ProductFactorExpr) -> Int {
        match pf {
            ProductFactorExpr::Var(poly) => self.virtpoly_to_int(&poly).clone(),
            ProductFactorExpr::OneMinus(poly) => Int::from(1) - self.virtpoly_to_int(&poly),
        }
    }

    fn add_product_constraints(&self, solver: &mut Solver) {
        PRODUCT_CONSTRAINTS.iter().for_each(|c| {
            let lhs = self.prodfac_to_int(c.left);
            let rhs = self.prodfac_to_int(c.right);
            *solver += (lhs * rhs).eq(self.virtpoly_to_int(&c.output));
        });
    }

    fn add_input_constraints(&self, solver: &mut Solver) {
        *solver += (&self.instruction_flags[InstructionFlags::LeftOperandIsRs1Value as usize]
            * &self.rs1_value
            + &self.instruction_flags[InstructionFlags::LeftOperandIsPC as usize] * &self.pc)
            .eq(&self.left_input);

        *solver += (&self.instruction_flags[InstructionFlags::RightOperandIsRs2Value as usize]
            * &self.rs2_value
            + &self.instruction_flags[InstructionFlags::RightOperandIsImm as usize] * &self.imm)
            .eq(&self.right_input);
    }

    fn add_constraints(&self, solver: &mut Solver) {
        self.add_r1cs_constraints(solver);
        self.add_product_constraints(solver);
        self.add_input_constraints(solver);
    }

    fn assert_output_differs(&self, solver: &mut Solver, other: &Self) {
        let or_terms = vec![
            // we are currently missing constraints on next_pc
            //(&self.next_pc).ne(&other.next_pc),
            self.next_unexpanded_pc.ne(&other.next_unexpanded_pc),
            // write to rd differs
            self.instruction_flags[InstructionFlags::IsRdNotZero as usize]
                .ne(&other.instruction_flags[InstructionFlags::IsRdNotZero as usize]),
            &self.instruction_flags[InstructionFlags::IsRdNotZero as usize].eq(Int::from(1))
                & (self.rd_write_value.ne(&other.rd_write_value)),
            // lookup inputs differ
            self.left_lookup.ne(&other.left_lookup),
            self.right_lookup.ne(&other.right_lookup),
            // write to ram differs
            self.ram_addr.ne(&other.ram_addr),
            (&self.ram_addr.ne(Int::from(0))) & self.ram_write_value.ne(&other.ram_write_value),
        ];

        *solver += or_terms.into_iter().reduce(|acc, t| acc | t).unwrap();
    }

    fn assert_input_matches(&self, solver: &mut Solver, instr: &Instruction, other: &Self) {
        let flags = instr.circuit_flags();
        let instruction_flags = instr.instruction_flags();

        self.flags
            .iter()
            .zip(other.flags.iter())
            .zip(flags)
            .for_each(|((self_flag, other_flag), flag_value)| {
                let flag_value = Int::from(flag_value as i64);
                *solver += self_flag.eq(&flag_value);
                *solver += other_flag.eq(&flag_value);
            });

        self.instruction_flags
            .iter()
            .zip(other.instruction_flags.iter())
            .zip(instruction_flags)
            .for_each(|((self_flag, other_flag), flag_value)| {
                let flag_value = Int::from(flag_value as i64);
                *solver += self_flag.eq(&flag_value);
                *solver += other_flag.eq(&flag_value);
            });

        *solver += self.imm.eq(&other.imm);

        *solver += self.rs1_value.eq(&other.rs1_value);
        *solver += self.rs2_value.eq(&other.rs2_value);

        *solver += self.left_input.eq(&other.left_input);
        *solver += self.right_input.eq(&other.right_input);

        *solver += self.pc.eq(&other.pc);
        *solver += self.unexpanded_pc.eq(&other.unexpanded_pc);
        *solver += self.next_is_noop.eq(&other.next_is_noop);

        // for now we don't emulate the lookup table, just use addition arbitrarily for everything
        *solver += self
            .lookup_output
            .eq(&self.left_lookup + &self.right_lookup);
        *solver += other
            .lookup_output
            .eq(&other.left_lookup + &other.right_lookup);

        // Make an artificially memory, placing rv1 at address 8 and rv2 at address 16, rest is 0
        let rv1 = Int::new_const("rv1");
        let rv2 = Int::new_const("rv2");
        let ram_expr = |addr: &Int| {
            addr.eq(Int::from(8))
                .ite(&rv1, &addr.eq(Int::from(16)).ite(&rv2, &Int::from(0)))
        };
        *solver += self.ram_read_value.eq(ram_expr(&self.ram_addr));
        *solver += other.ram_read_value.eq(ram_expr(&other.ram_addr));
    }

    fn eval(self, model: &z3::Model) -> Option<JoltState<i64>> {
        let eval = |x: &Int| model.eval(x, true).and_then(|v| v.as_i64());
        let flags = self.flags.map(|f| eval(&f).unwrap());
        let instruction_flags = self.instruction_flags.map(|f| eval(&f).unwrap());

        Some(JoltState {
            left_input: eval(&self.left_input)?,
            right_input: eval(&self.right_input)?,
            product: eval(&self.product)?,
            left_lookup: eval(&self.left_lookup)?,
            right_lookup: eval(&self.right_lookup)?,
            lookup_output: eval(&self.lookup_output)?,
            rs1_value: eval(&self.rs1_value)?,
            rs2_value: eval(&self.rs2_value)?,
            rd_write_value: eval(&self.rd_write_value)?,
            ram_addr: eval(&self.ram_addr)?,
            ram_read_value: eval(&self.ram_read_value)?,
            ram_write_value: eval(&self.ram_write_value)?,
            pc: eval(&self.pc)?,
            next_pc: eval(&self.next_pc)?,
            unexpanded_pc: eval(&self.unexpanded_pc)?,
            next_unexpanded_pc: eval(&self.next_unexpanded_pc)?,
            imm: eval(&self.imm)?,
            flags,
            instruction_flags,
            next_is_noop: eval(&self.next_is_noop)?,
            should_jump: eval(&self.should_jump)?,
            should_branch: eval(&self.should_branch)?,
            write_lookup_output_to_rd: eval(&self.write_lookup_output_to_rd)?,
            write_pc_to_rd: eval(&self.write_pc_to_rd)?,
            next_is_virtual: eval(&self.next_is_virtual)?,
            next_is_first_in_sequence: eval(&self.next_is_first_in_sequence)?,
        })
    }
}

struct CompareResult {
    field: &'static str,
    lhs: i64,
    rhs: i64,
}

impl JoltState<i64> {
    /// Compare two states, returning lists of differing inputs and outputs
    fn compare(&self, other: &Self) -> (Vec<CompareResult>, Vec<CompareResult>) {
        macro_rules! cmp {
            ($vec:ident, $field:ident) => {
                if self.$field != other.$field {
                    $vec.push(CompareResult {
                        field: stringify!($field),
                        lhs: self.$field,
                        rhs: other.$field,
                    });
                }
            };
        }
        let mut input = vec![];
        let mut output = vec![];
        cmp!(input, rs1_value);
        cmp!(input, rs2_value);
        cmp!(input, left_input);
        cmp!(input, right_input);
        cmp!(input, imm);
        cmp!(input, ram_addr);
        cmp!(input, ram_read_value);
        cmp!(input, next_is_noop);

        cmp!(output, ram_write_value);
        cmp!(output, rd_write_value);
        cmp!(output, next_pc);
        cmp!(output, next_unexpanded_pc);
        (input, output)
    }
}

fn do_test(name: &str, instr: &Instruction) {
    let mut solver = Solver::new();

    let r1 = JoltState::new("r1".to_string());
    r1.add_constraints(&mut solver);

    let r2 = JoltState::new("r2".to_string());
    r2.add_constraints(&mut solver);

    r1.assert_input_matches(&mut solver, instr, &r2);

    assert!(matches!(solver.check(), SatResult::Sat));

    r1.assert_output_differs(&mut solver, &r2);
    match solver.check() {
        SatResult::Sat => {
            let mut msg = format!("Found differing outputs for {name}\n");
            let model = solver.get_model().unwrap();
            let r1 = r1.eval(&model).unwrap();
            let r2 = r2.eval(&model).unwrap();
            let (inputs, outputs) = r1.compare(&r2);
            if !inputs.is_empty() {
                let _ = writeln!(msg, "Inputs:");
                for input in inputs {
                    let _ = writeln!(msg, "    {}: {} != {}", input.field, input.lhs, input.rhs);
                }
            }
            if !outputs.is_empty() {
                let _ = writeln!(msg, "Outputs:");
                for output in outputs {
                    let _ = writeln!(
                        msg,
                        "    {}: {} != {}",
                        output.field, output.lhs, output.rhs
                    );
                }
            }
            panic!("{}", msg.trim());
        }
        SatResult::Unsat => (),
        SatResult::Unknown => panic!("Solver failed"),
    }
}

macro_rules! test_instruction_constraints {
    ($instr:ident, $operands:path $(, $field:ident : $value:expr )* $(,)?) => {
        paste::paste! {
            #[test]
            #[allow(nonstandard_style)]
            fn [<test_ $instr>]() {
                let instr = Instruction::$instr($instr {
                    operands: $crate::template_format!($operands),
                    $($field: $value,)*
                    // unused by solver
                    address: 8,
                    is_compressed: false,
                    is_first_in_sequence: false,
                    virtual_sequence_remaining: None,
                });
                do_test(stringify!(instr), &instr);
            }
        }
    };
}

test_instruction_constraints!(ADD, FormatR);
test_instruction_constraints!(ADDI, FormatI);
test_instruction_constraints!(AND, FormatR);
test_instruction_constraints!(ANDI, FormatI);
test_instruction_constraints!(ANDN, FormatR);
test_instruction_constraints!(AUIPC, FormatU);
test_instruction_constraints!(BEQ, FormatB);
test_instruction_constraints!(BGE, FormatB);
test_instruction_constraints!(BGEU, FormatB);
test_instruction_constraints!(BLT, FormatB);
test_instruction_constraints!(BLTU, FormatB);
test_instruction_constraints!(BNE, FormatB);
test_instruction_constraints!(ECALL, FormatI);
test_instruction_constraints!(FENCE, FormatI);
test_instruction_constraints!(JAL, FormatJ);
test_instruction_constraints!(JALR, FormatI);
test_instruction_constraints!(LD, FormatLoad);
test_instruction_constraints!(LUI, FormatU);
test_instruction_constraints!(MUL, FormatR);
test_instruction_constraints!(MULHU, FormatR);
test_instruction_constraints!(OR, FormatR);
test_instruction_constraints!(ORI, FormatI);
test_instruction_constraints!(SD, FormatS);
test_instruction_constraints!(SLT, FormatR);
test_instruction_constraints!(SLTI, FormatI);
test_instruction_constraints!(SLTIU, FormatI);
test_instruction_constraints!(SLTU, FormatR);
test_instruction_constraints!(SUB, FormatR);
test_instruction_constraints!(VirtualAdvice, FormatJ, advice: 0);
test_instruction_constraints!(VirtualAssertEQ, FormatB);
test_instruction_constraints!(VirtualAssertHalfwordAlignment, AssertAlignFormat);
test_instruction_constraints!(VirtualAssertLTE, FormatB);
test_instruction_constraints!(VirtualAssertMulUNoOverflow, FormatB);
test_instruction_constraints!(VirtualAssertValidDiv0, FormatB);
test_instruction_constraints!(VirtualAssertValidUnsignedRemainder, FormatB);
test_instruction_constraints!(VirtualAssertWordAlignment, AssertAlignFormat);
test_instruction_constraints!(VirtualChangeDivisor, FormatR);
test_instruction_constraints!(VirtualMovsign, FormatI);
test_instruction_constraints!(VirtualMULI, FormatI);
test_instruction_constraints!(VirtualPow2, FormatI);
test_instruction_constraints!(VirtualPow2I, FormatJ);
test_instruction_constraints!(VirtualPow2IW, FormatJ);
test_instruction_constraints!(VirtualPow2W, FormatI);
test_instruction_constraints!(VirtualRev8W, FormatI);
test_instruction_constraints!(VirtualROTRIW, FormatVirtualRightShiftI);
test_instruction_constraints!(VirtualShiftRightBitmask, FormatI);
test_instruction_constraints!(VirtualShiftRightBitmaskI, FormatJ);
test_instruction_constraints!(VirtualSignExtendWord, FormatI);
test_instruction_constraints!(VirtualSRA, FormatVirtualRightShiftR);
test_instruction_constraints!(VirtualSRAI, FormatVirtualRightShiftI);
test_instruction_constraints!(VirtualSRL, FormatVirtualRightShiftR);
test_instruction_constraints!(VirtualSRLI, FormatVirtualRightShiftI);
