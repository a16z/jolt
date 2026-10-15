use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::{Display, Formatter, Result as FmtResult},
};

use jolt_claims::protocols::jolt::{
    relations::{
        instruction::InputVirtualization, ram::ReadWriteChecking as RamReadWriteChecking,
        registers::ReadWriteChecking as RegistersReadWriteChecking, spartan::Shift,
    },
    JoltPolynomialId, JoltRelationId, JoltVirtualPolynomial, UnbatchedClaim as Claim,
    UnbatchedClaimExpr as ClaimExpr, UnbatchedRelation as RelationClaims,
};
use jolt_lookup_tables::LookupTableKind;
use regex::{NoExpand, Regex};

use crate::{
    modules::{AsModule, Module},
    util::indent,
};

/// The per-cycle identities underlying the four relations exported to ZKLean.
/// The runtime sumchecks fold this same metadata by their gamma challenges.
fn extracted_claims() -> Vec<RelationClaims> {
    vec![
        RamReadWriteChecking::unbatched_relation(),
        RegistersReadWriteChecking::unbatched_relation(),
        InputVirtualization::unbatched_relation(),
        Shift::unbatched_relation(),
    ]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ZkLeanVarRef {
    relation: JoltRelationId,
    polynomial: JoltPolynomialId,
}

impl Display for ZkLeanVarRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(
            f,
            "{}.{}",
            relation_ident(self.relation),
            polynomial_ident(self.polynomial)
        )
    }
}

impl ZkLeanVarRef {
    pub fn new(relation: JoltRelationId, polynomial: JoltPolynomialId) -> Self {
        Self {
            relation,
            polynomial,
        }
    }

    pub fn virtual_var(relation: JoltRelationId, polynomial: JoltVirtualPolynomial) -> Self {
        Self::new(relation, polynomial.into())
    }
}

#[derive(Debug, Clone)]
pub struct ZkLeanSumchecks {
    claims: Vec<RelationClaims>,
    vars: BTreeMap<JoltRelationId, BTreeSet<String>>,
}

impl ZkLeanSumchecks {
    pub fn extract<const XLEN: usize>() -> Self {
        let claims = extracted_claims();
        let mut vars = BTreeMap::<JoltRelationId, BTreeSet<String>>::new();

        for relation_claims in &claims {
            vars.entry(relation_claims.output_relation).or_default();
            for claim in &relation_claims.claims {
                claim.input.visit_polynomials(&mut |polynomial| {
                    vars.entry(claim.input_relation)
                        .or_default()
                        .insert(polynomial_ident(polynomial));
                });
                claim.output.visit_polynomials(&mut |polynomial| {
                    vars.entry(relation_claims.output_relation)
                        .or_default()
                        .insert(polynomial_ident(polynomial));
                });
            }
        }

        let instruction_read_raf = vars.entry(JoltRelationId::InstructionReadRaf).or_default();
        instruction_read_raf.insert(polynomial_ident(
            JoltVirtualPolynomial::InstructionRafFlag.into(),
        ));
        for index in 0..LookupTableKind::<XLEN>::COUNT {
            instruction_read_raf.insert(polynomial_ident(
                JoltVirtualPolynomial::LookupTableFlag(index).into(),
            ));
        }

        Self { claims, vars }
    }

    pub fn zklean_pretty_print(
        &self,
        f: &mut impl std::io::Write,
        mut indent_level: usize,
    ) -> std::io::Result<()> {
        let top_level_indent = indent_level;
        let vars_type = "SumcheckVars";

        for (relation, vars) in &self.vars {
            if *relation == JoltRelationId::SpartanOuter || vars.is_empty() {
                continue;
            }
            writeln!(
                f,
                "{}structure {} (f : Type) : Type where",
                indent(indent_level),
                relation_ident(*relation)
            )?;
            indent_level += 1;
            for var in vars {
                writeln!(f, "{}{var} : ZKExpr f", indent(indent_level))?;
            }
            indent_level -= 1;
            writeln!(f)?;
        }

        writeln!(
            f,
            "{}structure {vars_type} (f : Type) : Type where",
            indent(indent_level)
        )?;
        indent_level += 1;
        writeln!(
            f,
            "{}JoltR1CSInputs : JoltR1CSInputs f",
            indent(indent_level)
        )?;
        for (relation, vars) in &self.vars {
            if *relation == JoltRelationId::SpartanOuter || vars.is_empty() {
                continue;
            }
            let relation = relation_ident(*relation);
            writeln!(f, "{}{relation} : {relation} f", indent(indent_level))?;
        }
        indent_level = top_level_indent;
        writeln!(f)?;

        let mut uniform = Vec::new();
        let mut non_uniform = Vec::new();
        for relation in &self.claims {
            let (offset, same_cycle): (Vec<_>, Vec<_>) =
                relation.claims.iter().partition(|claim| claim.offset);
            if !same_cycle.is_empty() {
                let name = format!(
                    "{}.uniform_claims",
                    relation_ident(relation.output_relation)
                );
                uniform.push(name.clone());
                pretty_print_claims_fun(
                    f,
                    &name,
                    vars_type,
                    &same_cycle,
                    relation.output_relation,
                    false,
                    indent_level,
                )?;
                writeln!(f)?;
            }
            if !offset.is_empty() {
                let name = format!(
                    "{}.non_uniform_claims",
                    relation_ident(relation.output_relation)
                );
                non_uniform.push(name.clone());
                pretty_print_claims_fun(
                    f,
                    &name,
                    vars_type,
                    &offset,
                    relation.output_relation,
                    true,
                    indent_level,
                )?;
                writeln!(f)?;
            }
        }

        writeln!(
            f,
            "{}def uniform_claims [Field f] (cycle : {vars_type} f) : ZKBuilder f PUnit := do",
            indent(indent_level)
        )?;
        indent_level += 1;
        for name in uniform {
            writeln!(f, "{}{name} cycle", indent(indent_level))?;
        }
        indent_level -= 1;
        writeln!(f)?;

        writeln!(
            f,
            "{}def non_uniform_claims [Field f] (cycle next_cycle : {vars_type} f) : ZKBuilder f PUnit := do",
            indent(indent_level)
        )?;
        indent_level += 1;
        for name in non_uniform {
            writeln!(f, "{}{name} cycle next_cycle", indent(indent_level))?;
        }

        Ok(())
    }
}

fn remove_parens(mut string: String) -> String {
    let open = Regex::new(r"\(").expect("static regex is valid");
    let close = Regex::new(r"\)").expect("static regex is valid");
    string = open.replace_all(&string, NoExpand("_")).to_string();
    close.replace_all(&string, NoExpand("")).to_string()
}

fn relation_ident(relation: JoltRelationId) -> String {
    if relation == JoltRelationId::SpartanOuter {
        return String::from("JoltR1CSInputs");
    }
    remove_parens(format!("{relation:?}_Vars"))
}

fn polynomial_ident(polynomial: JoltPolynomialId) -> String {
    match polynomial {
        JoltPolynomialId::Committed(polynomial) => remove_parens(format!("{polynomial:?}")),
        JoltPolynomialId::Virtual(polynomial) => remove_parens(format!("{polynomial:?}")),
    }
}

fn pretty_print_claims_fun(
    f: &mut impl std::io::Write,
    name: &str,
    vars_type: &str,
    claims: &[&Claim],
    output_relation: JoltRelationId,
    offset: bool,
    mut indent_level: usize,
) -> std::io::Result<()> {
    let arguments = if offset { "cycle next_cycle" } else { "cycle" };
    writeln!(
        f,
        "{}def {name} [Field f] ({arguments} : {vars_type} f) : ZKBuilder f PUnit := do",
        indent(indent_level)
    )?;
    indent_level += 1;
    for claim in claims {
        writeln!(f, "{}ZKBuilder.constrainEq", indent(indent_level))?;
        indent_level += 1;
        write!(f, "{}", indent(indent_level))?;
        pretty_print_claim_expr(f, "cycle", claim.input_relation, &claim.input, true)?;
        writeln!(f)?;
        write!(f, "{}", indent(indent_level))?;
        pretty_print_claim_expr(
            f,
            if offset { "next_cycle" } else { "cycle" },
            output_relation,
            &claim.output,
            true,
        )?;
        writeln!(f)?;
        indent_level -= 1;
    }
    Ok(())
}

fn pretty_print_claim_expr(
    f: &mut impl std::io::Write,
    vars: &str,
    relation: JoltRelationId,
    expr: &ClaimExpr,
    group: bool,
) -> std::io::Result<()> {
    match expr {
        ClaimExpr::Constant(value) => write!(f, "{value}"),
        ClaimExpr::Polynomial(polynomial) => {
            write!(f, "{vars}.{}", ZkLeanVarRef::new(relation, *polynomial))
        }
        ClaimExpr::Add(lhs, rhs) | ClaimExpr::Mul(lhs, rhs) | ClaimExpr::Sub(lhs, rhs) => {
            if group {
                write!(f, "(")?;
            }
            pretty_print_claim_expr(f, vars, relation, lhs, !matches!(expr, ClaimExpr::Add(..)))?;
            let operator = match expr {
                ClaimExpr::Add(..) => " + ",
                ClaimExpr::Mul(..) => " * ",
                ClaimExpr::Sub(..) => " - ",
                _ => unreachable!(),
            };
            write!(f, "{operator}")?;
            pretty_print_claim_expr(f, vars, relation, rhs, !matches!(expr, ClaimExpr::Add(..)))?;
            if group {
                write!(f, ")")?;
            }
            Ok(())
        }
    }
}

impl AsModule for ZkLeanSumchecks {
    fn as_module(&self) -> std::io::Result<Module> {
        let mut contents = Vec::new();
        self.zklean_pretty_print(&mut contents, 0)?;
        Ok(Module {
            name: String::from("Sumchecks"),
            imports: vec![String::from("zkLean"), String::from("Jolt.R1CS")],
            contents,
        })
    }
}
