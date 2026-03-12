#![cfg(test)]
#![allow(non_upper_case_globals)]

use jolt_instructions::{
    CircuitFlags, Flags, InstructionFlags, NUM_CIRCUIT_FLAGS, NUM_INSTRUCTION_FLAGS,
};
use jolt_ir::{zkvm::claims::r1cs::*, Z3Emitter};
use std::{array, fmt::Write};
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
        format_assert_align::FormatAssert, format_b::FormatB, format_fence::FormatFence,
        format_i::FormatI, format_j::FormatJ, format_load::FormatLoad, format_r::FormatR,
        format_s::FormatS, format_u::FormatU,
        format_virtual_right_shift_i::FormatVirtualRightShiftI,
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

/// Bridge from tracer's `Instruction` enum to jolt-instructions `Flags` trait.
///
/// The tracer crate defines the `Instruction` enum but does not implement
/// `Flags`. The jolt-instructions crate provides `Flags` on unit structs.
/// This function dispatches to the correct unit struct.
fn get_flags(instr: &Instruction) -> ([bool; NUM_CIRCUIT_FLAGS], [bool; NUM_INSTRUCTION_FLAGS]) {
    use jolt_instructions::rv::{arithmetic, branch, compare, jump, load, logic, store, system};
    use jolt_instructions::virtual_::{
        advice, arithmetic as v_arith, assert as v_assert, bitwise, byte, division, extension,
        shift as v_shift,
    };

    macro_rules! flags_of {
        ($t:expr) => {{
            let i = $t;
            (Flags::circuit_flags(&i), Flags::instruction_flags(&i))
        }};
    }

    match instr {
        Instruction::NoOp => {
            let mut cf = [false; NUM_CIRCUIT_FLAGS];
            cf[CircuitFlags::DoNotUpdateUnexpandedPC as usize] = true;
            let mut iflag = [false; NUM_INSTRUCTION_FLAGS];
            iflag[InstructionFlags::IsNoop as usize] = true;
            (cf, iflag)
        }
        // RV64I arithmetic
        Instruction::ADD(_) => flags_of!(arithmetic::Add),
        Instruction::ADDI(_) => flags_of!(arithmetic::Addi),
        Instruction::SUB(_) => flags_of!(arithmetic::Sub),
        Instruction::LUI(_) => flags_of!(arithmetic::Lui),
        Instruction::AUIPC(_) => flags_of!(arithmetic::Auipc),
        Instruction::MUL(_) => flags_of!(arithmetic::Mul),
        Instruction::MULHU(_) => flags_of!(arithmetic::MulHU),
        // RV64I logic
        Instruction::AND(_) => flags_of!(logic::And),
        Instruction::ANDI(_) => flags_of!(logic::AndI),
        Instruction::ANDN(_) => flags_of!(logic::Andn),
        Instruction::OR(_) => flags_of!(logic::Or),
        Instruction::ORI(_) => flags_of!(logic::OrI),
        // RV64I branch
        Instruction::BEQ(_) => flags_of!(branch::Beq),
        Instruction::BGE(_) => flags_of!(branch::Bge),
        Instruction::BGEU(_) => flags_of!(branch::BgeU),
        Instruction::BLT(_) => flags_of!(branch::Blt),
        Instruction::BLTU(_) => flags_of!(branch::BltU),
        Instruction::BNE(_) => flags_of!(branch::Bne),
        // RV64I compare
        Instruction::SLT(_) => flags_of!(compare::Slt),
        Instruction::SLTI(_) => flags_of!(compare::SltI),
        Instruction::SLTIU(_) => flags_of!(compare::SltIU),
        Instruction::SLTU(_) => flags_of!(compare::SltU),
        // RV64I jump
        Instruction::JAL(_) => flags_of!(jump::Jal),
        Instruction::JALR(_) => flags_of!(jump::Jalr),
        // RV64I load/store
        Instruction::LD(_) => flags_of!(load::Ld),
        Instruction::SD(_) => flags_of!(store::Sd),
        // RV64I system
        Instruction::ECALL(_) => flags_of!(system::Ecall),
        Instruction::FENCE(_) => flags_of!(system::Fence),
        // Virtual instructions
        Instruction::VirtualAdvice(_) => flags_of!(advice::VirtualAdvice),
        Instruction::VirtualAssertEQ(_) => flags_of!(v_assert::AssertEq),
        Instruction::VirtualAssertHalfwordAlignment(_) => {
            flags_of!(v_assert::AssertHalfwordAlignment)
        }
        Instruction::VirtualAssertLTE(_) => flags_of!(v_assert::AssertLte),
        Instruction::VirtualAssertMulUNoOverflow(_) => {
            flags_of!(v_assert::AssertMulUNoOverflow)
        }
        Instruction::VirtualAssertValidDiv0(_) => flags_of!(v_assert::AssertValidDiv0),
        Instruction::VirtualAssertValidUnsignedRemainder(_) => {
            flags_of!(v_assert::AssertValidUnsignedRemainder)
        }
        Instruction::VirtualAssertWordAlignment(_) => flags_of!(v_assert::AssertWordAlignment),
        Instruction::VirtualChangeDivisor(_) => flags_of!(division::VirtualChangeDivisor),
        Instruction::VirtualMovsign(_) => flags_of!(bitwise::MovSign),
        Instruction::VirtualMULI(_) => flags_of!(v_arith::MulI),
        Instruction::VirtualPow2(_) => flags_of!(v_arith::Pow2),
        Instruction::VirtualPow2I(_) => flags_of!(v_arith::Pow2I),
        Instruction::VirtualPow2IW(_) => flags_of!(v_arith::Pow2IW),
        Instruction::VirtualPow2W(_) => flags_of!(v_arith::Pow2W),
        Instruction::VirtualRev8W(_) => flags_of!(byte::VirtualRev8W),
        Instruction::VirtualROTRIW(_) => flags_of!(v_shift::VirtualRotriw),
        Instruction::VirtualShiftRightBitmask(_) => flags_of!(v_shift::VirtualShiftRightBitmask),
        Instruction::VirtualShiftRightBitmaskI(_) => {
            flags_of!(v_shift::VirtualShiftRightBitmaski)
        }
        Instruction::VirtualSignExtendWord(_) => flags_of!(extension::VirtualSignExtendWord),
        Instruction::VirtualSRA(_) => flags_of!(v_shift::VirtualSra),
        Instruction::VirtualSRAI(_) => flags_of!(v_shift::VirtualSrai),
        Instruction::VirtualSRL(_) => flags_of!(v_shift::VirtualSrl),
        Instruction::VirtualSRLI(_) => flags_of!(v_shift::VirtualSrli),
        _ => panic!("Unsupported instruction in z3-verifier: {instr:?}"),
    }
}

#[derive(Clone, Debug)]
struct JoltState<T = Int> {
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
    next_is_virtual: T,
    next_is_first_in_sequence: T,
    virtual_sequence_active: T,
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
            next_is_virtual: Int::new_const(format!("{prefix}_next_is_virtual")),
            next_is_first_in_sequence: Int::new_const(format!(
                "{prefix}_next_is_first_in_sequence"
            )),
            virtual_sequence_active: Int::new_const(format!("{prefix}_virtual_sequence_active")),
        }
    }

    /// Bind all JoltState fields as Opening variables in a Z3Emitter,
    /// mapping each field to its corresponding V_* index.
    fn bind_to_emitter(&self, emitter: &mut Z3Emitter) {
        emitter.bind_opening(V_LEFT_INSTRUCTION_INPUT as u32, self.left_input.clone());
        emitter.bind_opening(V_RIGHT_INSTRUCTION_INPUT as u32, self.right_input.clone());
        emitter.bind_opening(V_PRODUCT as u32, self.product.clone());
        emitter.bind_opening(V_SHOULD_BRANCH as u32, self.should_branch.clone());
        emitter.bind_opening(V_PC as u32, self.pc.clone());
        emitter.bind_opening(V_UNEXPANDED_PC as u32, self.unexpanded_pc.clone());
        emitter.bind_opening(V_IMM as u32, self.imm.clone());
        emitter.bind_opening(V_RAM_ADDRESS as u32, self.ram_addr.clone());
        emitter.bind_opening(V_RS1_VALUE as u32, self.rs1_value.clone());
        emitter.bind_opening(V_RS2_VALUE as u32, self.rs2_value.clone());
        emitter.bind_opening(V_RD_WRITE_VALUE as u32, self.rd_write_value.clone());
        emitter.bind_opening(V_RAM_READ_VALUE as u32, self.ram_read_value.clone());
        emitter.bind_opening(V_RAM_WRITE_VALUE as u32, self.ram_write_value.clone());
        emitter.bind_opening(V_LEFT_LOOKUP_OPERAND as u32, self.left_lookup.clone());
        emitter.bind_opening(V_RIGHT_LOOKUP_OPERAND as u32, self.right_lookup.clone());
        emitter.bind_opening(V_NEXT_UNEXPANDED_PC as u32, self.next_unexpanded_pc.clone());
        emitter.bind_opening(V_NEXT_PC as u32, self.next_pc.clone());
        emitter.bind_opening(V_NEXT_IS_VIRTUAL as u32, self.next_is_virtual.clone());
        emitter.bind_opening(
            V_NEXT_IS_FIRST_IN_SEQUENCE as u32,
            self.next_is_first_in_sequence.clone(),
        );
        emitter.bind_opening(V_LOOKUP_OUTPUT as u32, self.lookup_output.clone());
        emitter.bind_opening(V_SHOULD_JUMP as u32, self.should_jump.clone());

        // Circuit flags (V_FLAG_ADD_OPERANDS is the base offset)
        for i in 0..NUM_CIRCUIT_FLAGS {
            emitter.bind_opening((V_FLAG_ADD_OPERANDS + i) as u32, self.flags[i].clone());
        }

        // Product factor variables
        emitter.bind_opening(
            V_BRANCH as u32,
            self.instruction_flags[InstructionFlags::Branch as usize].clone(),
        );
        emitter.bind_opening(V_NEXT_IS_NOOP as u32, self.next_is_noop.clone());
    }

    /// Add all R1CS + product constraints using jolt-ir constraint expressions.
    fn add_ir_constraints(&self, solver: &mut Solver) {
        for c in &constraint_exprs() {
            let mut emitter = Z3Emitter::new();
            self.bind_to_emitter(&mut emitter);
            let z3_expr = c.expr.to_circuit(&mut emitter);
            *solver += z3_expr.eq(Int::from(0));
        }
    }

    /// Semantic constraints on instruction operand routing (not part of R1CS).
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
        self.add_ir_constraints(solver);
        self.add_input_constraints(solver);
    }

    fn assert_output_differs(&self, solver: &mut Solver, other: &Self) {
        let or_terms = vec![
            self.next_unexpanded_pc.ne(&other.next_unexpanded_pc),
            self.instruction_flags[InstructionFlags::IsRdNotZero as usize]
                .ne(&other.instruction_flags[InstructionFlags::IsRdNotZero as usize]),
            &self.instruction_flags[InstructionFlags::IsRdNotZero as usize].eq(Int::from(1))
                & (self.rd_write_value.ne(&other.rd_write_value)),
            self.left_lookup.ne(&other.left_lookup),
            self.right_lookup.ne(&other.right_lookup),
            self.ram_addr.ne(&other.ram_addr),
            (&self.ram_addr.ne(Int::from(0))) & self.ram_write_value.ne(&other.ram_write_value),
        ];

        *solver += or_terms.into_iter().reduce(|acc, t| acc | t).unwrap();
    }

    fn assert_input_matches(&self, solver: &mut Solver, instr: &Instruction, other: &Self) {
        let (circuit_flags, instruction_flags) = get_flags(instr);

        self.flags
            .iter()
            .zip(other.flags.iter())
            .zip(circuit_flags)
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

        // Simplified lookup table: addition for all instructions
        *solver += self
            .lookup_output
            .eq(&self.left_lookup + &self.right_lookup);
        *solver += other
            .lookup_output
            .eq(&other.left_lookup + &other.right_lookup);

        // Artificial memory: rv1 at address 8, rv2 at address 16, rest is 0
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
            next_is_virtual: eval(&self.next_is_virtual)?,
            next_is_first_in_sequence: eval(&self.next_is_first_in_sequence)?,
            virtual_sequence_active: eval(&self.virtual_sequence_active)?,
        })
    }
}

struct CompareResult {
    field: &'static str,
    lhs: i64,
    rhs: i64,
}

impl JoltState<i64> {
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
test_instruction_constraints!(FENCE, FormatFence);
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
test_instruction_constraints!(VirtualAssertHalfwordAlignment, FormatAssert);
test_instruction_constraints!(VirtualAssertLTE, FormatB);
test_instruction_constraints!(VirtualAssertMulUNoOverflow, FormatB);
test_instruction_constraints!(VirtualAssertValidDiv0, FormatB);
test_instruction_constraints!(VirtualAssertValidUnsignedRemainder, FormatB);
test_instruction_constraints!(VirtualAssertWordAlignment, FormatAssert);
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
