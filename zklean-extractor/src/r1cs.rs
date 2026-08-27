use jolt_field::{Fr, Ring};
use jolt_r1cs::{
    constraints::rv64::{self, NUM_CONSTRAINTS_PER_CYCLE, NUM_VARS_PER_CYCLE, V_CONST},
    ConstraintMatrices, SparseRow,
};
use jolt_riscv::CircuitFlags;

use crate::{
    constants::JoltParameterSet,
    modules::{AsModule, Module},
    util::indent,
};

const VARIABLE_NAMES: [&str; NUM_VARS_PER_CYCLE] = [
    "One",
    "LeftInstructionInput",
    "RightInstructionInput",
    "Product",
    "ShouldBranch",
    "PC",
    "UnexpandedPC",
    "Imm",
    "RamAddress",
    "Rs1Value",
    "Rs2Value",
    "RdWriteValue",
    "RamReadValue",
    "RamWriteValue",
    "LeftLookupOperand",
    "RightLookupOperand",
    "NextUnexpandedPC",
    "NextPC",
    "NextIsVirtual",
    "NextIsFirstInSequence",
    "LookupOutput",
    "ShouldJump",
    "OpFlags_AddOperands",
    "OpFlags_SubtractOperands",
    "OpFlags_MultiplyOperands",
    "OpFlags_Load",
    "OpFlags_Store",
    "OpFlags_Jump",
    "OpFlags_WriteLookupOutputToRD",
    "OpFlags_VirtualInstruction",
    "OpFlags_Assert",
    "OpFlags_DoNotUpdateUnexpandedPC",
    "OpFlags_Advice",
    "OpFlags_IsCompressed",
    "OpFlags_IsFirstInSequence",
    "OpFlags_IsLastInSequence",
    "Branch",
    "NextIsNoop",
];

const CONSTRAINT_NAMES: [&str; NUM_CONSTRAINTS_PER_CYCLE] = [
    "RamAddrEqRs1PlusImmIfLoadStore",
    "RamAddrEqZeroIfNotLoadStore",
    "RamReadEqRamWriteIfLoad",
    "RamReadEqRdWriteIfLoad",
    "Rs2EqRamWriteIfStore",
    "LeftLookupZeroUnlessAddSubMul",
    "LeftLookupEqLeftInputOtherwise",
    "RightLookupAdd",
    "RightLookupSub",
    "RightLookupEqProductIfMul",
    "RightLookupEqRightInputOtherwise",
    "AssertLookupOne",
    "RdWriteEqLookupIfWriteLookupToRd",
    "RdWriteEqPCPlusConstIfWritePCtoRD",
    "NextUnexpPCEqLookupIfShouldJump",
    "NextUnexpPCEqPCPlusImmIfShouldBranch",
    "NextUnexpPCUpdateOtherwise",
    "NextPCEqPCPlusOneIfInline",
    "MustStartSequenceFromBeginning",
    "Product",
    "ShouldBranch",
    "ShouldJump",
];

pub struct ZkLeanR1CSConstraints<J> {
    matrices: ConstraintMatrices<Fr>,
    phantom: std::marker::PhantomData<J>,
}

impl<J: JoltParameterSet> ZkLeanR1CSConstraints<J> {
    pub fn extract() -> Self {
        Self {
            matrices: rv64::rv64_trace_constraints(),
            phantom: std::marker::PhantomData,
        }
    }

    pub fn zklean_pretty_print(
        &self,
        f: &mut impl std::io::Write,
        mut indent_level: usize,
    ) -> std::io::Result<()> {
        let top_level_indent = indent_level;

        writeln!(
            f,
            "{}structure JoltR1CSInputs (f : Type) : Type where",
            indent(indent_level)
        )?;
        indent_level += 1;
        for field in &VARIABLE_NAMES[1..] {
            writeln!(f, "{}{field} : ZKExpr f", indent(indent_level))?;
        }
        writeln!(f)?;

        indent_level = top_level_indent;
        writeln!(
            f,
            "{}instance: Witnessable f (JoltR1CSInputs f) where",
            indent(indent_level)
        )?;
        indent_level += 1;
        writeln!(f, "{}witness := do", indent(indent_level))?;
        indent_level += 1;
        for field in &VARIABLE_NAMES[1..] {
            writeln!(
                f,
                "{}let {field} <- Witnessable.witness",
                indent(indent_level)
            )?;
        }
        writeln!(f)?;
        writeln!(f, "{}pure {{", indent(indent_level))?;
        indent_level += 1;
        for field in &VARIABLE_NAMES[1..] {
            writeln!(f, "{}{field} := {field}", indent(indent_level))?;
        }
        indent_level -= 1;
        writeln!(f, "{}}}", indent(indent_level))?;
        writeln!(f)?;

        indent_level = top_level_indent;
        writeln!(
            f,
            "{}def uniform_jolt_constraints [ZKField f] (jolt_inputs : JoltR1CSInputs f) : ZKBuilder f PUnit := do",
            indent(indent_level)
        )?;
        indent_level += 1;
        for (index, ((a, b), c)) in self
            .matrices
            .a
            .iter()
            .zip(&self.matrices.b)
            .zip(&self.matrices.c)
            .enumerate()
        {
            writeln!(f, "{}-- {}", indent(indent_level), CONSTRAINT_NAMES[index])?;
            writeln!(f, "{}ZKBuilder.constrainR1CS", indent(indent_level))?;
            indent_level += 1;
            writeln!(
                f,
                "{}{}",
                indent(indent_level),
                pretty_print_lc("jolt_inputs", a)
            )?;
            writeln!(
                f,
                "{}{}",
                indent(indent_level),
                pretty_print_lc("jolt_inputs", b)
            )?;
            writeln!(
                f,
                "{}{}",
                indent(indent_level),
                pretty_print_lc("jolt_inputs", c)
            )?;
            indent_level -= 1;
        }

        Ok(())
    }
}

impl<J: JoltParameterSet> AsModule for ZkLeanR1CSConstraints<J> {
    fn as_module(&self) -> std::io::Result<Module> {
        let mut contents = Vec::new();
        self.zklean_pretty_print(&mut contents, 0)?;

        Ok(Module {
            name: String::from("R1CS"),
            imports: vec![String::from("zkLean")],
            contents,
        })
    }
}

pub fn circuit_flag_field_name(flag: CircuitFlags) -> String {
    format!("OpFlags_{flag:?}")
}

fn signed_coefficient(value: Fr) -> i128 {
    for magnitude in [1_i128, 2, 4, 1_i128 << 64] {
        if value == Fr::from_i128(magnitude) {
            return magnitude;
        }
        if value == Fr::from_i128(-magnitude) {
            return -magnitude;
        }
    }
    panic!("RV64 R1CS introduced an unsupported coefficient: {value}")
}

fn pretty_print_term(inputs_struct: &str, variable: usize, coefficient: Fr) -> String {
    let coefficient = signed_coefficient(coefficient);
    if variable == V_CONST {
        return coefficient.to_string();
    }

    let variable = format!("{inputs_struct}.{}", VARIABLE_NAMES[variable]);
    match coefficient {
        1 => variable,
        coefficient => format!("({coefficient}*{variable})"),
    }
}

fn pretty_print_lc(inputs_struct: &str, row: &SparseRow<Fr>) -> String {
    let terms = row
        .iter()
        .map(|&(variable, coefficient)| pretty_print_term(inputs_struct, variable, coefficient))
        .collect::<Vec<_>>();

    match terms.as_slice() {
        [] => "0".to_string(),
        [term] => term.clone(),
        _ => format!("({})", terms.join(" + ")),
    }
}
