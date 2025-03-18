use jolt_core::r1cs::{
        builder::{Constraint, OffsetEqConstraint, OffsetLC, R1CSBuilder},
        constraints::R1CSConstraints,
        inputs::{ConstraintInput as _, JoltR1CSInputs},
        ops::{Term, Variable, LC},
    };
use common::rv_trace::MemoryLayout;
use regex::{NoExpand, Regex};

// NOTE: C=4 is taken from the invocation of the `impl_r1cs_input_lc_conversions` macro for
// `JoltR1CSInputs` in `jolt-core/src/r1cs/inputs.rs`.
// XXX Do we want to take `C` as a type parameter instead?
const C: usize = 4;

type F = ark_bn254::Fr;
type CS = jolt_core::r1cs::constraints::JoltRV32IMConstraints;

pub struct ZkLeanR1CSConstraints {
    inputs: Vec<JoltR1CSInputs>,
    uniform_constraints: Vec<Constraint>,
    non_uniform_constraints: Vec<OffsetEqConstraint>,
}

impl ZkLeanR1CSConstraints {
    pub fn extract(max_input_size: u64, max_output_size: u64) -> Self {
        let inputs = JoltR1CSInputs::flatten::<C>();

        // XXX Make max input/output sizes configurable?
        let uniform_constraints = {
            let memory_layout = MemoryLayout::new(max_input_size, max_output_size);

            let mut r1cs_builder = R1CSBuilder::<C, F, JoltR1CSInputs>::new();
            CS::uniform_constraints(&mut r1cs_builder, memory_layout.input_start);

            r1cs_builder.get_constraints()
        };

        let non_uniform_constraints = <CS as R1CSConstraints<C, F>>::cross_step_constraints();

        Self {
            inputs,
            uniform_constraints,
            non_uniform_constraints,
        }
    }

    pub fn zklean_pretty_print(&self, f: &mut impl std::io::Write) -> std::io::Result<()> {
        let tab = "  ";
        f.write(b"structure JoltR1CSInputs (f : Type) : Type where\n")?;
        for input in &self.inputs {
            let field = input_to_field_name(input);
            f.write(format!("{tab}{field} : ZKExpr f\n").as_bytes())?;
        }
        f.write(b"\n")?;
        f.write(b"def uniform_jolt_constraints [JoltField f] (jolt_inputs : JoltR1CSInputs f) : ZKBuilder f PUnit := do\n")?;
        for Constraint { a, b, c } in &self.uniform_constraints {
            f.write(format!("{tab}constrainR1CS\n").as_bytes())?;
            f.write(format!("{tab}{tab}{}\n", pretty_print_lc("jolt_inputs", a)).as_bytes())?;
            f.write(format!("{tab}{tab}{}\n", pretty_print_lc("jolt_inputs", b)).as_bytes())?;
            f.write(format!("{tab}{tab}{}\n", pretty_print_lc("jolt_inputs", c)).as_bytes())?;
        }
        f.write(b"\n")?;
        f.write(b"def non_uniform_jolt_constraints [JoltField f]")?;
        f.write(b" (jolt_inputs : JoltR1CSInputs f) (jolt_offset_inputs : JoltR1CSInputs f) : ZKBuilder f PUnit := do\n")?;
        for OffsetEqConstraint { cond, a, b } in &self.non_uniform_constraints {
            // NOTE: See comments on `materialize_offset_eq` and `OffsetLC`. An offset contraint is three
            // `OffsetLC`s, cond, a, and b. An `OffsetLC` is an `LC` and a `bool`. If the bool is true,
            // then the variables in the LC come from the *next* step. The cond, a, and b `LC`s resolve to
            // a constraint as
            // A: a - b
            // B: cond
            // C: 0
            f.write(format!("{tab}constrainR1CS\n").as_bytes())?;
            f.write(format!(
                    "{tab}{tab}({} - {})\n",
                    pretty_print_offset_lc("jolt_inputs", "jolt_offset_inputs", a),
                    pretty_print_offset_lc("jolt_inputs", "jolt_offset_inputs", b),
            ).as_bytes())?;
            f.write(format!("{tab}{tab}{}\n",
                    pretty_print_offset_lc("jolt_inputs", "jolt_offset_inputs", cond),
            ).as_bytes())?;
            f.write(format!("{tab}{tab}0\n").as_bytes())?;
        }

        Ok(())
    }
}

fn input_to_field_name(input: &JoltR1CSInputs) -> String {
    let paren = Regex::new(r"\((.*)\)").unwrap();
    let comma = Regex::new(r", *").unwrap();

    let mut string: String = format!("{:?}", input);

    string = comma.replace_all(&string.as_str(), NoExpand("_")).to_string();

    while paren.is_match(&string.as_str()) {
        string = paren.replace(&string.as_str(), "_$1").to_string();
    }

    string
}

fn input_index_to_field_name(index: usize) -> String {
    input_to_field_name(&JoltR1CSInputs::from_index::<C>(index))
}

fn pretty_print_term(inputs_struct: &str, Term(var, coeff): &Term) -> Option<String> {
    let var = match *var {
        Variable::Input(index) => Some(input_index_to_field_name(index)),
        Variable::Auxiliary(index) => Some(input_index_to_field_name(index)), // XXX What do we do differently for auxs?
        Variable::Constant => None,
    };
    match (coeff, var) {
        (0, _) => None,
        (1, Some(var)) => Some(format!("{inputs_struct}.{var}").to_string()),
        (_, Some(var)) => Some(format!("{coeff}*{inputs_struct}.{var}").to_string()),
        (_, None) => Some(format!("{coeff}").to_string()),
    }
}

fn pretty_print_lc(inputs_struct: &str, lc: &LC) -> String {
    let terms = lc.terms()
        .into_iter()
        .filter_map(|term| pretty_print_term(inputs_struct, term))
        .collect::<Vec<_>>();
    match terms.len() {
        0 => "0".to_string(),
        1 => terms[0].clone(),
        _ => format!("({})", terms.join(" + ")).to_string(),
    }
}

fn pretty_print_offset_lc(inputs_struct: &str, offset_inputs_struct: &str, (offset, lc): &OffsetLC) -> String {
    let inputs_struct = if *offset {
        offset_inputs_struct
    } else {
        inputs_struct
    };
    pretty_print_lc(inputs_struct, lc)
}
