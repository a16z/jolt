use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::{Display, Formatter, Result as FmtResult},
};

use jolt_claims::protocols::jolt::{
    JoltCommittedPolynomial, JoltPolynomialId, JoltRelationId, JoltVirtualPolynomial,
};
#[cfg(test)]
use jolt_field::Ring;
use jolt_lookup_tables::LookupTableKind;
use jolt_riscv::{CircuitFlags, InstructionFlags};
use regex::{NoExpand, Regex};

use crate::{
    modules::{AsModule, Module},
    util::indent,
};

#[derive(Debug, Clone)]
enum ClaimExpr {
    Constant(i64),
    Var(JoltPolynomialId),
    Add(Box<Self>, Box<Self>),
    Mul(Box<Self>, Box<Self>),
    Sub(Box<Self>, Box<Self>),
}

impl ClaimExpr {
    fn var(polynomial: impl Into<JoltPolynomialId>) -> Self {
        Self::Var(polynomial.into())
    }

    fn add(self, rhs: Self) -> Self {
        Self::Add(Box::new(self), Box::new(rhs))
    }

    fn mul(self, rhs: Self) -> Self {
        Self::Mul(Box::new(self), Box::new(rhs))
    }

    fn sub(self, rhs: Self) -> Self {
        Self::Sub(Box::new(self), Box::new(rhs))
    }

    fn visit_vars(&self, visit: &mut impl FnMut(JoltPolynomialId)) {
        match self {
            Self::Constant(_) => {}
            Self::Var(polynomial) => visit(*polynomial),
            Self::Add(lhs, rhs) | Self::Mul(lhs, rhs) | Self::Sub(lhs, rhs) => {
                lhs.visit_vars(visit);
                rhs.visit_vars(visit);
            }
        }
    }

    #[cfg(test)]
    fn evaluate<F: Ring>(&self, resolve: &mut impl FnMut(JoltPolynomialId) -> F) -> F {
        match self {
            Self::Constant(value) => F::from_i64(*value),
            Self::Var(polynomial) => resolve(*polynomial),
            Self::Add(lhs, rhs) => lhs.evaluate(resolve) + rhs.evaluate(resolve),
            Self::Mul(lhs, rhs) => lhs.evaluate(resolve) * rhs.evaluate(resolve),
            Self::Sub(lhs, rhs) => lhs.evaluate(resolve) - rhs.evaluate(resolve),
        }
    }
}

#[derive(Debug, Clone)]
struct Claim {
    input_relation: JoltRelationId,
    input: ClaimExpr,
    output: ClaimExpr,
    offset: bool,
}

#[derive(Debug, Clone)]
struct RelationClaims {
    output_relation: JoltRelationId,
    claims: Vec<Claim>,
}

fn v(polynomial: JoltVirtualPolynomial) -> ClaimExpr {
    ClaimExpr::var(polynomial)
}

fn c(polynomial: JoltCommittedPolynomial) -> ClaimExpr {
    ClaimExpr::var(polynomial)
}

/// The per-cycle identities underlying the four relations exported to ZKLean.
/// Their gamma-folded forms are the modular relations in `jolt-claims`.
fn extracted_claims() -> Vec<RelationClaims> {
    use JoltRelationId as Relation;
    use JoltVirtualPolynomial as Virtual;

    vec![
        RelationClaims {
            output_relation: Relation::RamReadWriteChecking,
            claims: vec![
                Claim {
                    input_relation: Relation::SpartanOuter,
                    input: v(Virtual::RamReadValue),
                    output: v(Virtual::RamRa).mul(v(Virtual::RamVal)),
                    offset: false,
                },
                Claim {
                    input_relation: Relation::SpartanOuter,
                    input: v(Virtual::RamWriteValue),
                    output: v(Virtual::RamRa)
                        .mul(v(Virtual::RamVal).add(c(JoltCommittedPolynomial::RamInc))),
                    offset: false,
                },
            ],
        },
        RelationClaims {
            output_relation: Relation::RegistersReadWriteChecking,
            claims: vec![
                Claim {
                    input_relation: Relation::RegistersClaimReduction,
                    input: v(Virtual::RdWriteValue),
                    output: v(Virtual::RdWa)
                        .mul(v(Virtual::RegistersVal).add(c(JoltCommittedPolynomial::RdInc))),
                    offset: false,
                },
                Claim {
                    input_relation: Relation::RegistersClaimReduction,
                    input: v(Virtual::Rs1Value),
                    output: v(Virtual::Rs1Ra).mul(v(Virtual::RegistersVal)),
                    offset: false,
                },
                Claim {
                    input_relation: Relation::RegistersClaimReduction,
                    input: v(Virtual::Rs2Value),
                    output: v(Virtual::Rs2Ra).mul(v(Virtual::RegistersVal)),
                    offset: false,
                },
            ],
        },
        RelationClaims {
            output_relation: Relation::InstructionInputVirtualization,
            claims: vec![
                Claim {
                    input_relation: Relation::SpartanProductVirtualization,
                    input: v(Virtual::RightInstructionInput),
                    output: v(Virtual::InstructionFlags(
                        InstructionFlags::RightOperandIsRs2Value,
                    ))
                    .mul(v(Virtual::Rs2Value))
                    .add(
                        v(Virtual::InstructionFlags(
                            InstructionFlags::RightOperandIsImm,
                        ))
                        .mul(v(Virtual::Imm)),
                    ),
                    offset: false,
                },
                Claim {
                    input_relation: Relation::SpartanProductVirtualization,
                    input: v(Virtual::LeftInstructionInput),
                    output: v(Virtual::InstructionFlags(
                        InstructionFlags::LeftOperandIsRs1Value,
                    ))
                    .mul(v(Virtual::Rs1Value))
                    .add(
                        v(Virtual::InstructionFlags(InstructionFlags::LeftOperandIsPC))
                            .mul(v(Virtual::UnexpandedPC)),
                    ),
                    offset: false,
                },
            ],
        },
        RelationClaims {
            output_relation: Relation::SpartanShift,
            claims: vec![
                Claim {
                    input_relation: Relation::SpartanOuter,
                    input: v(Virtual::NextUnexpandedPC),
                    output: v(Virtual::UnexpandedPC),
                    offset: true,
                },
                Claim {
                    input_relation: Relation::SpartanOuter,
                    input: v(Virtual::NextPC),
                    output: v(Virtual::PC),
                    offset: true,
                },
                Claim {
                    input_relation: Relation::SpartanOuter,
                    input: v(Virtual::NextIsVirtual),
                    output: v(Virtual::OpFlags(CircuitFlags::VirtualInstruction)),
                    offset: true,
                },
                Claim {
                    input_relation: Relation::SpartanOuter,
                    input: v(Virtual::NextIsFirstInSequence),
                    output: v(Virtual::OpFlags(CircuitFlags::IsFirstInSequence)),
                    offset: true,
                },
                Claim {
                    input_relation: Relation::SpartanProductVirtualization,
                    input: ClaimExpr::Constant(1).sub(v(Virtual::NextIsNoop)),
                    output: ClaimExpr::Constant(1)
                        .sub(v(Virtual::InstructionFlags(InstructionFlags::IsNoop))),
                    offset: true,
                },
            ],
        },
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
                claim.input.visit_vars(&mut |polynomial| {
                    vars.entry(claim.input_relation)
                        .or_default()
                        .insert(polynomial_ident(polynomial));
                });
                claim.output.visit_vars(&mut |polynomial| {
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
        ClaimExpr::Var(polynomial) => {
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

#[cfg(test)]
mod tests {
    use super::*;
    use ark_std::{One, Zero};
    use jolt_claims::protocols::jolt::relations::{
        instruction::InputVirtualization, ram::ReadWriteChecking as RamReadWriteChecking,
        registers::ReadWriteChecking as RegistersReadWriteChecking, spartan::Shift,
    };
    use jolt_claims::protocols::jolt::{
        InstructionInputPublic, JoltChallengeId, JoltDerivedId, JoltOpeningId, RamReadWritePublic,
        ReadWriteDimensions, RegistersReadWritePublic, SpartanShiftPublic, TraceDimensions,
    };
    use jolt_claims::SymbolicSumcheck;
    use jolt_field::{Fr, Ring};

    fn polynomial_values(claims: &RelationClaims) -> BTreeMap<JoltPolynomialId, Fr> {
        let mut polynomials = BTreeSet::new();
        for claim in &claims.claims {
            claim.input.visit_vars(&mut |polynomial| {
                polynomials.insert(polynomial);
            });
            claim.output.visit_vars(&mut |polynomial| {
                polynomials.insert(polynomial);
            });
        }
        polynomials
            .into_iter()
            .enumerate()
            .map(|(index, polynomial)| (polynomial, Fr::from_u64(index as u64 + 2)))
            .collect()
    }

    fn assert_matches_runtime_relation<S>(
        extracted: &RelationClaims,
        runtime: &S,
        gamma: Fr,
        output_weight: impl Fn(JoltRelationId) -> Fr,
        derived_value: impl Fn(&JoltDerivedId) -> Fr,
    ) where
        S: SymbolicSumcheck<
            RelationId = JoltRelationId,
            OpeningId = JoltOpeningId,
            DerivedId = JoltDerivedId,
            ChallengeId = JoltChallengeId,
        >,
    {
        assert_eq!(extracted.output_relation, S::id());
        let values = polynomial_values(extracted);
        let resolve_polynomial = |polynomial| values[&polynomial];

        let runtime_input = runtime.input_expression::<Fr>().evaluate(
            |opening| resolve_polynomial(opening.polynomial_id()),
            |_| gamma,
            |_| Fr::zero(),
        );
        let runtime_output = runtime.output_expression::<Fr>().evaluate(
            |opening| resolve_polynomial(opening.polynomial_id()),
            |_| gamma,
            derived_value,
        );

        let mut gamma_power = Fr::one();
        let mut extracted_input = Fr::zero();
        let mut extracted_output = Fr::zero();
        for claim in &extracted.claims {
            extracted_input += gamma_power
                * claim
                    .input
                    .evaluate(&mut |polynomial| resolve_polynomial(polynomial));
            extracted_output += gamma_power
                * output_weight(claim.input_relation)
                * claim
                    .output
                    .evaluate(&mut |polynomial| resolve_polynomial(polynomial));
            gamma_power *= gamma;
        }

        assert_eq!(extracted_input, runtime_input);
        assert_eq!(extracted_output, runtime_output);
    }

    #[test]
    fn exported_unfused_claims_gamma_fold_to_runtime_relations() {
        let claims = extracted_claims();
        let get = |relation| {
            claims
                .iter()
                .find(|claims| claims.output_relation == relation)
                .expect("exported relation exists")
        };
        let gamma = Fr::from_u64(37);
        let eq = Fr::from_u64(41);
        let eq_product = Fr::from_u64(43);

        assert_matches_runtime_relation(
            get(JoltRelationId::RamReadWriteChecking),
            &RamReadWriteChecking::new(ReadWriteDimensions::new(5, 4, 2, 1)),
            gamma,
            |_| eq,
            |derived| match derived {
                JoltDerivedId::RamReadWrite(RamReadWritePublic::EqCycle) => eq,
                _ => Fr::zero(),
            },
        );
        assert_matches_runtime_relation(
            get(JoltRelationId::RegistersReadWriteChecking),
            &RegistersReadWriteChecking::new(ReadWriteDimensions::new(5, 7, 2, 1)),
            gamma,
            |_| eq,
            |derived| match derived {
                JoltDerivedId::RegistersReadWrite(RegistersReadWritePublic::EqCycle) => eq,
                _ => Fr::zero(),
            },
        );
        assert_matches_runtime_relation(
            get(JoltRelationId::InstructionInputVirtualization),
            &InputVirtualization::new(TraceDimensions::new(5)),
            gamma,
            |_| eq,
            |derived| match derived {
                JoltDerivedId::InstructionInput(InstructionInputPublic::EqProduct) => eq,
                _ => Fr::zero(),
            },
        );
        assert_matches_runtime_relation(
            get(JoltRelationId::SpartanShift),
            &Shift::new(TraceDimensions::new(5)),
            gamma,
            |input_relation| match input_relation {
                JoltRelationId::SpartanOuter => eq,
                JoltRelationId::SpartanProductVirtualization => eq_product,
                _ => Fr::zero(),
            },
            |derived| match derived {
                JoltDerivedId::SpartanShift(SpartanShiftPublic::EqPlusOneOuter) => eq,
                JoltDerivedId::SpartanShift(SpartanShiftPublic::EqPlusOneProduct) => eq_product,
                _ => Fr::zero(),
            },
        );
    }
}
