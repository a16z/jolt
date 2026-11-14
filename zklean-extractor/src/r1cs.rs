use jolt_claims::protocols::composed::r1cs::rv64::{
    self, RV64_CONSTRAINT_NAMES, RV64_VARIABLE_NAMES, V_CONST,
};
use jolt_field::{Fr, Ring};
use jolt_r1cs::{ConstraintMatrices, SparseRow};
use jolt_riscv::CircuitFlags;

use crate::{
    constants::JoltParameterSet,
    modules::{AsModule, Module},
    util::indent,
};

/// The RV64 constraints under each claimed panic bit; they differ only in
/// the jump lane's output (row 21's `C`), which the extracted Lean definition
/// selects with its `claimed_panic` parameter.
pub struct ZkLeanR1CSConstraints<J> {
    success: ConstraintMatrices<Fr>,
    panic: ConstraintMatrices<Fr>,
    phantom: std::marker::PhantomData<J>,
}

impl<J: JoltParameterSet> ZkLeanR1CSConstraints<J> {
    pub fn extract() -> Self {
        Self {
            success: rv64::rv64_trace_constraints(false),
            panic: rv64::rv64_trace_constraints(true),
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
        for field in &RV64_VARIABLE_NAMES[1..] {
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
        for field in &RV64_VARIABLE_NAMES[1..] {
            writeln!(
                f,
                "{}let {field} <- Witnessable.witness",
                indent(indent_level)
            )?;
        }
        writeln!(f)?;
        writeln!(f, "{}pure {{", indent(indent_level))?;
        indent_level += 1;
        for field in &RV64_VARIABLE_NAMES[1..] {
            writeln!(f, "{}{field} := {field}", indent(indent_level))?;
        }
        indent_level -= 1;
        writeln!(f, "{}}}", indent(indent_level))?;
        writeln!(f)?;

        indent_level = top_level_indent;
        writeln!(
            f,
            "{}def uniform_jolt_constraints [ZKField f] (claimed_panic : Bool) (jolt_inputs : JoltR1CSInputs f) : ZKBuilder f PUnit := do",
            indent(indent_level)
        )?;
        indent_level += 1;
        let rows = |matrices: &ConstraintMatrices<Fr>| {
            matrices
                .a
                .iter()
                .zip(&matrices.b)
                .zip(&matrices.c)
                .map(|((a, b), c)| [a.clone(), b.clone(), c.clone()])
                .collect::<Vec<_>>()
        };
        for (index, (success, panic)) in rows(&self.success)
            .into_iter()
            .zip(rows(&self.panic))
            .enumerate()
        {
            writeln!(
                f,
                "{}-- {}",
                indent(indent_level),
                RV64_CONSTRAINT_NAMES[index]
            )?;
            writeln!(f, "{}ZKBuilder.constrainR1CS", indent(indent_level))?;
            indent_level += 1;
            for (success, panic) in success.iter().zip(&panic) {
                let success = pretty_print_lc("jolt_inputs", success);
                let panic = pretty_print_lc("jolt_inputs", panic);
                if success == panic {
                    writeln!(f, "{}{success}", indent(indent_level))?;
                } else {
                    writeln!(
                        f,
                        "{}(if claimed_panic then {panic} else {success})",
                        indent(indent_level)
                    )?;
                }
            }
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

    let variable = format!("{inputs_struct}.{}", RV64_VARIABLE_NAMES[variable]);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::RV64IParameterSet;

    /// The extracted constraints carry both halt flags and branch on the
    /// claimed panic bit only in the jump lane.
    #[test]
    fn extracted_constraints_branch_only_the_jump_lane() {
        let mut output = Vec::new();
        ZkLeanR1CSConstraints::<RV64IParameterSet>::extract()
            .zklean_pretty_print(&mut output, 0)
            .unwrap();
        let output = String::from_utf8(output).unwrap();
        assert!(output.contains("OpFlags_Halt : ZKExpr f"));
        assert!(output.contains("OpFlags_HaltPanic : ZKExpr f"));
        assert_eq!(output.matches("if claimed_panic then").count(), 1);
        assert_eq!(
            output.matches("ZKBuilder.constrainR1CS").count(),
            RV64_CONSTRAINT_NAMES.len()
        );
    }
}
