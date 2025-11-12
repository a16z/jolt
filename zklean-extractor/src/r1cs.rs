use jolt_core::zkvm::r1cs::{
    constraints::{NamedR1CSConstraint, R1CSConstraint, R1CS_CONSTRAINTS},
    inputs::{JoltR1CSInputs, ALL_R1CS_INPUTS},
    ops::{Term, LC},
};
use regex::{NoExpand, Regex};

use crate::{
    constants::JoltParameterSet,
    modules::{AsModule, Module},
    util::indent,
};

pub struct ZkLeanR1CSConstraints<J> {
    inputs: Vec<JoltR1CSInputs>,
    uniform_constraints: Vec<NamedR1CSConstraint>,
    phantom: std::marker::PhantomData<J>,
}

impl<J: JoltParameterSet> ZkLeanR1CSConstraints<J> {
    pub fn extract() -> Self {
        let inputs = ALL_R1CS_INPUTS.to_vec();
        let uniform_constraints = R1CS_CONSTRAINTS.to_vec();

        Self {
            inputs,
            uniform_constraints,
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
        for constraint in &self.uniform_constraints {
            // Note that an R1CS constraint in Jolt is currently expressed as *just* the `a` and
            // `b` parts, with the `c` part omitted. See `zkvm/r1cs/constraints.rs`. That's because
            // all constraints are conditional equalities:
            //   if <condition> { <left> - <right> == 0 }
            // Thus `a` = <condition>, `b` = <left> - <right>, and (implicitly) `c` = 0.
            let R1CSConstraint { a, b } = &constraint.cons;
            let c = &LC::zero();
            let name = format!("{:?}", constraint.label);

            f.write_fmt(format_args!("{}-- {name}\n", indent(indent_level)))?;
            f.write_fmt(format_args!(
                "{}ZKBuilder.constrainR1CS\n",
                indent(indent_level),
            ))?;
            indent_level += 1;
            f.write_fmt(format_args!(
                "{}{}\n",
                indent(indent_level),
                pretty_print_lc("jolt_inputs", a),
            ))?;
            f.write_fmt(format_args!(
                "{}{}\n",
                indent(indent_level),
                pretty_print_lc("jolt_inputs", b),
            ))?;
            f.write_fmt(format_args!(
                "{}{}\n",
                indent(indent_level),
                pretty_print_lc("jolt_inputs", c),
            ))?;
            indent_level -= 1;
        }

        Ok(())
    }

    pub fn zklean_imports(&self) -> Vec<String> {
        vec![String::from("ZkLean")]
    }
}

impl<J: JoltParameterSet> AsModule for ZkLeanR1CSConstraints<J> {
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

fn input_index_to_field_name(index: usize) -> String {
    input_to_field_name(&ALL_R1CS_INPUTS[index])
}

fn pretty_print_term(inputs_struct: &str, Term { input_index, coeff }: &Term) -> String {
    let var = input_index_to_field_name(*input_index);
    match coeff {
        1 => format!("{inputs_struct}.{var}").to_string(),
        c => format!("({c}*{inputs_struct}.{var})").to_string(),
    }
}

fn pretty_print_lc(inputs_struct: &str, lc: &LC) -> String {
    let (var_terms, len, const_term) = LC::decompose(*lc);
    let const_term = match const_term {
        0 => None,
        c => Some(format!("{c}").to_string()),
    };
    let var_terms = var_terms[..len]
        .iter()
        .map(|t| pretty_print_term(inputs_struct, t));
    let terms = const_term.into_iter().chain(var_terms).collect::<Vec<_>>();

    match terms.len() {
        0 => "0".to_string(),
        1 => terms[0].clone(),
        _ => format!("({})", terms.join(" + ")).to_string(),
    }
}
