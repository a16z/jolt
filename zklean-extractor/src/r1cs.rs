use common::rv_trace::MemoryLayout;
use jolt_core::r1cs::{
    builder::{Constraint, OffsetEqConstraint, OffsetLC, R1CSBuilder},
    constraints::R1CSConstraints,
    inputs::{ConstraintInput as _, JoltR1CSInputs},
    ops::{Term, Variable, LC},
};
use regex::{NoExpand, Regex};

use crate::{
    constants::JoltParameterSet,
    modules::{AsModule, Module},
    util::indent,
};

type F = ark_bn254::Fr;
type CS = jolt_core::r1cs::constraints::JoltRV32IMConstraints;

pub struct ZkLeanR1CSConstraints<J> {
    inputs: Vec<JoltR1CSInputs>,
    uniform_constraints: Vec<Constraint>,
    non_uniform_constraints: Vec<OffsetEqConstraint>,
    phantom: std::marker::PhantomData<J>,
}

impl<J: JoltParameterSet> ZkLeanR1CSConstraints<J>
where
    [(); J::C]:,
{
    pub fn extract() -> Self {
        let inputs = JoltR1CSInputs::flatten::<{ J::C }>();

        // XXX Make max input/output sizes configurable?
        let uniform_constraints = {
            let memory_layout = MemoryLayout::new(&J::MEMORY_CONFIG);

            let mut r1cs_builder = R1CSBuilder::<{ J::C }, F, JoltR1CSInputs>::new();
            CS::uniform_constraints(&mut r1cs_builder, memory_layout.input_start);

            r1cs_builder.get_constraints()
        };

        let non_uniform_constraints =
            <CS as R1CSConstraints<{ J::C }, F>>::cross_step_constraints();

        Self {
            inputs,
            uniform_constraints,
            non_uniform_constraints,
            phantom: std::marker::PhantomData,
        }
    }

    pub fn zklean_pretty_print(
        &self,
        f: &mut impl std::io::Write,
        mut indent_level: usize,
    ) -> std::io::Result<()> {
        let top_level_indent = indent_level;

        f.write_fmt(format_args!(
            "{}structure JoltR1CSInputs (f : Type) : Type where\n",
            indent(indent_level),
        ))?;
        indent_level += 1;
        for input in &self.inputs {
            let field = input_to_field_name(input);
            f.write_fmt(format_args!("{}{field} : ZKExpr f\n", indent(indent_level),))?;
        }
        f.write_all(b"\n")?;

        // for every input make it Witnessable following the pattern
        // ```
        // instance: Witnessable f (JoltR1CSInputs f) where
        //   witness := do
        //     let bytecode_a <- Witnessable.witness;
        //     let bytecode_elf_address <- Witnessable.witness;
        //     let bytecode_bitflags <- Witnessable.witness;
        //     ...
        //     pure {
        //       Bytecode_A := bytecode_a,
        //       Bytecode_ELFAddress := bytecode_elf_address,
        //       Bytecode_Bitflags := bytecode_bitflags,
        //       ...
        //     }
        // ```
        indent_level = top_level_indent;
        f.write_fmt(format_args!(
            "{}instance: Witnessable f (JoltR1CSInputs f) where\n",
            indent(indent_level),
        ))?;
        indent_level += 1;
        f.write_fmt(format_args!("{}witness := do\n", indent(indent_level),))?;
        indent_level += 1;
        for input in &self.inputs {
            let field = input_to_field_name(input);
            f.write_fmt(format_args!(
                "{}let {field} <- Witnessable.witness\n",
                indent(indent_level),
            ))?;
        }
        f.write_all(b"\n")?;
        f.write_fmt(format_args!("{}pure {{\n", indent(indent_level),))?;
        indent_level += 1;
        for input in &self.inputs {
            let field = input_to_field_name(input);
            f.write_fmt(format_args!("{}{field} := {field}\n", indent(indent_level),))?;
        }
        indent_level -= 1;
        f.write_fmt(format_args!("{}}}\n", indent(indent_level),))?;
        f.write_all(b"\n")?;

        indent_level = top_level_indent;
        f.write_fmt(format_args!(
                "{}def uniform_jolt_constraints [ZKField f] (jolt_inputs : JoltR1CSInputs f) : ZKBuilder f PUnit := do\n",
                indent(indent_level),
        ))?;
        indent_level += 1;
        for Constraint { a, b, c } in &self.uniform_constraints {
            f.write_fmt(format_args!("{}constrainR1CS\n", indent(indent_level),))?;
            indent_level += 1;
            f.write_fmt(format_args!(
                "{}{}\n",
                indent(indent_level),
                pretty_print_lc::<{ J::C }>("jolt_inputs", a),
            ))?;
            f.write_fmt(format_args!(
                "{}{}\n",
                indent(indent_level),
                pretty_print_lc::<{ J::C }>("jolt_inputs", b),
            ))?;
            f.write_fmt(format_args!(
                "{}{}\n",
                indent(indent_level),
                pretty_print_lc::<{ J::C }>("jolt_inputs", c),
            ))?;
            indent_level -= 1;
        }

        f.write_all(b"\n")?;
        indent_level = top_level_indent;
        f.write_fmt(format_args!(
                "{}def non_uniform_jolt_constraints [ZKField f] (jolt_inputs : JoltR1CSInputs f) (jolt_offset_inputs : JoltR1CSInputs f) : ZKBuilder f PUnit := do\n",
                indent(indent_level),
        ))?;
        indent_level += 1;
        for OffsetEqConstraint { cond, a, b } in &self.non_uniform_constraints {
            // NOTE: See comments on `materialize_offset_eq` and `OffsetLC`. An offset constraint is three
            // `OffsetLC`s, cond, a, and b. An `OffsetLC` is an `LC` and a `bool`. If the bool is true,
            // then the variables in the LC come from the *next* step. The cond, a, and b `LC`s resolve to
            // a constraint as
            // A: a - b
            // B: cond
            // C: 0
            f.write_fmt(format_args!("{}constrainR1CS\n", indent(indent_level),))?;
            indent_level += 1;
            f.write_fmt(format_args!(
                "{}({} - {})\n",
                indent(indent_level),
                pretty_print_offset_lc::<{ J::C }>("jolt_inputs", "jolt_offset_inputs", a),
                pretty_print_offset_lc::<{ J::C }>("jolt_inputs", "jolt_offset_inputs", b),
            ))?;
            f.write_fmt(format_args!(
                "{}{}\n",
                indent(indent_level),
                pretty_print_offset_lc::<{ J::C }>("jolt_inputs", "jolt_offset_inputs", cond),
            ))?;
            f.write_fmt(format_args!("{}0\n", indent(indent_level),))?;
            indent_level -= 1;
        }

        Ok(())
    }

    pub fn zklean_imports(&self) -> Vec<String> {
        vec![String::from("ZkLean")]
    }
}

impl<J: JoltParameterSet> AsModule for ZkLeanR1CSConstraints<J>
where
    [(); J::C]:,
{
    fn as_module(&self) -> std::io::Result<Module> {
        let mut contents: Vec<u8> = vec![];
        self.zklean_pretty_print(&mut contents, 0)?;

        Ok(Module {
            name: String::from("R1CS"),
            imports: self.zklean_imports(),
            contents,
        })
    }
}

pub fn input_to_field_name(input: &JoltR1CSInputs) -> String {
    let paren = Regex::new(r"\((.*)\)").unwrap();
    let comma = Regex::new(r", *").unwrap();

    let mut string: String = format!("{input:?}");

    string = comma
        .replace_all(string.as_str(), NoExpand("_"))
        .to_string();

    while paren.is_match(string.as_str()) {
        string = paren.replace(string.as_str(), "_$1").to_string();
    }

    string
}

fn input_index_to_field_name<const C: usize>(index: usize) -> String {
    input_to_field_name(&JoltR1CSInputs::from_index::<C>(index))
}

fn pretty_print_term<const C: usize>(
    inputs_struct: &str,
    Term(var, coeff): &Term,
) -> Option<String> {
    let var = match *var {
        Variable::Input(index) | Variable::Auxiliary(index) => {
            Some(input_index_to_field_name::<C>(index))
        }
        Variable::Constant => None,
    };
    match (coeff, var) {
        (0, _) => None,
        (1, Some(var)) => Some(format!("{inputs_struct}.{var}").to_string()),
        (_, Some(var)) => Some(format!("{coeff}*{inputs_struct}.{var}").to_string()),
        (_, None) => Some(format!("{coeff}").to_string()),
    }
}

fn pretty_print_lc<const C: usize>(inputs_struct: &str, lc: &LC) -> String {
    let terms = lc
        .terms()
        .iter()
        .filter_map(|term| pretty_print_term::<C>(inputs_struct, term))
        .collect::<Vec<_>>();
    match terms.len() {
        0 => "0".to_string(),
        1 => terms[0].clone(),
        _ => format!("({})", terms.join(" + ")).to_string(),
    }
}

fn pretty_print_offset_lc<const C: usize>(
    inputs_struct: &str,
    offset_inputs_struct: &str,
    (offset, lc): &OffsetLC,
) -> String {
    let inputs_struct = if *offset {
        offset_inputs_struct
    } else {
        inputs_struct
    };
    pretty_print_lc::<C>(inputs_struct, lc)
}
