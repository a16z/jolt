use std::{
    collections::{BTreeMap, BTreeSet},
    fmt::{Display, Formatter, Result as FmtResult},
    io::{Error as IoError, Result as IoResult},
};

use jolt_claims::protocols::jolt::{
    relations::{
        instruction::InputVirtualization, ram::ReadWriteChecking as RamReadWriteChecking,
        registers::ReadWriteChecking as RegistersReadWriteChecking, spartan::Shift,
    },
    InstructionInputPublic, JoltDerivedId, JoltPolynomialId, JoltRelationId, JoltVirtualPolynomial,
    RamReadWritePublic, RegistersReadWritePublic, SpartanShiftPublic, UnbatchedClaim as Claim,
    UnbatchedClaimExpr as ClaimExpr, UnbatchedRelation as RelationClaims,
};
use jolt_lookup_tables::LookupTableKind;
use regex::{NoExpand, Regex};

use crate::{
    modules::{AsModule, Module},
    util::indent,
};

/// The `RamReadWriteChecking_Vars` field that `memory_step` (in the package
/// template's `Jolt/MemOps.lean`) binds to the write-protected-word mask C at
/// the doubleword a cycle accesses.
const WRITE_PROTECTED: &str = "WriteProtected";

/// A claim's public weight, lowered to the per-cycle model: cycle equality
/// states the identity at every cycle, successor-cycle equality links each
/// cycle to the next, and a further public factor becomes a per-cycle
/// variable that multiplies the claim's output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum CycleWeight {
    Cycle,
    NextCycle,
    WriteProtectedCycle,
}

impl CycleWeight {
    fn of(claim: &Claim) -> IoResult<Self> {
        let weight = match claim.output_weight {
            JoltDerivedId::RamReadWrite(public) => match public {
                RamReadWritePublic::EqCycle => Self::Cycle,
                RamReadWritePublic::WriteProtectedEqCycle => Self::WriteProtectedCycle,
                RamReadWritePublic::Reserved2(retired) => match retired {},
            },
            JoltDerivedId::RegistersReadWrite(public) => match public {
                RegistersReadWritePublic::EqCycle => Self::Cycle,
            },
            JoltDerivedId::InstructionInput(public) => match public {
                InstructionInputPublic::EqProduct => Self::Cycle,
            },
            JoltDerivedId::SpartanShift(public) => match public {
                SpartanShiftPublic::EqPlusOneOuter | SpartanShiftPublic::EqPlusOneProduct => {
                    Self::NextCycle
                }
            },
            weight => {
                return Err(IoError::other(format!(
                    "ZKLean has no lowering for the claim weight {weight:?}"
                )))
            }
        };
        if claim.offset != (weight == Self::NextCycle) {
            return Err(IoError::other(format!(
                "the claim weight {:?} disagrees with the claim's cycle offset",
                claim.output_weight
            )));
        }
        Ok(weight)
    }
}

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
        // The full-trace R1CS contains the jump product, so its successor bit is also
        // the witness read by the shift relation.
        let relation = if polynomial == JoltVirtualPolynomial::NextIsNoop.into() {
            JoltRelationId::SpartanOuter
        } else {
            relation
        };
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
        let mut declare = |relation, polynomial| {
            let field = ZkLeanVarRef::new(relation, polynomial);
            vars.entry(field.relation)
                .or_default()
                .insert(polynomial_ident(field.polynomial));
        };

        for relation_claims in &claims {
            for claim in &relation_claims.claims {
                claim
                    .input
                    .visit_polynomials(&mut |polynomial| declare(claim.input_relation, polynomial));
                claim.output.visit_polynomials(&mut |polynomial| {
                    declare(relation_claims.output_relation, polynomial)
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
    ) -> IoResult<()> {
        let top_level_indent = indent_level;
        let vars_type = "SumcheckVars";

        let mut declared = self.vars.clone();
        let mut lowered = Vec::with_capacity(self.claims.len());
        for relation in &self.claims {
            let mut claims = Vec::with_capacity(relation.claims.len());
            for claim in &relation.claims {
                let weight = CycleWeight::of(claim)?;
                if weight == CycleWeight::WriteProtectedCycle {
                    declared
                        .entry(relation.output_relation)
                        .or_default()
                        .insert(String::from(WRITE_PROTECTED));
                }
                claims.push((claim, weight));
            }
            lowered.push((relation.output_relation, claims));
        }

        for (relation, vars) in &declared {
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
        for (relation, vars) in &declared {
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
        for (output_relation, claims) in &lowered {
            let (offset, same_cycle): (Vec<_>, Vec<_>) = claims
                .iter()
                .partition(|(_, weight)| *weight == CycleWeight::NextCycle);
            if !same_cycle.is_empty() {
                let name = format!("{}.uniform_claims", relation_ident(*output_relation));
                uniform.push(name.clone());
                pretty_print_claims_fun(
                    f,
                    &name,
                    vars_type,
                    &same_cycle,
                    *output_relation,
                    false,
                    indent_level,
                )?;
                writeln!(f)?;
            }
            if !offset.is_empty() {
                let name = format!("{}.non_uniform_claims", relation_ident(*output_relation));
                non_uniform.push(name.clone());
                pretty_print_claims_fun(
                    f,
                    &name,
                    vars_type,
                    &offset,
                    *output_relation,
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
    claims: &[&(&Claim, CycleWeight)],
    output_relation: JoltRelationId,
    offset: bool,
    mut indent_level: usize,
) -> IoResult<()> {
    let arguments = if offset { "cycle next_cycle" } else { "cycle" };
    let output_vars = if offset { "next_cycle" } else { "cycle" };
    writeln!(
        f,
        "{}def {name} [Field f] ({arguments} : {vars_type} f) : ZKBuilder f PUnit := do",
        indent(indent_level)
    )?;
    indent_level += 1;
    for (claim, weight) in claims {
        writeln!(f, "{}ZKBuilder.constrainEq", indent(indent_level))?;
        indent_level += 1;
        write!(f, "{}", indent(indent_level))?;
        pretty_print_claim_expr(f, "cycle", claim.input_relation, &claim.input, true)?;
        writeln!(f)?;
        write!(f, "{}", indent(indent_level))?;
        match weight {
            CycleWeight::Cycle | CycleWeight::NextCycle => {
                pretty_print_claim_expr(f, output_vars, output_relation, &claim.output, true)?;
            }
            CycleWeight::WriteProtectedCycle => {
                write!(
                    f,
                    "({output_vars}.{}.{WRITE_PROTECTED} * ",
                    relation_ident(output_relation)
                )?;
                pretty_print_claim_expr(f, output_vars, output_relation, &claim.output, true)?;
                write!(f, ")")?;
            }
        }
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
) -> IoResult<()> {
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
    fn as_module(&self) -> IoResult<Module> {
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

    #[test]
    fn shift_binds_the_r1cs_successor_bit_to_the_next_rows_noop_flag() {
        let module = ZkLeanSumchecks::extract::<64>().as_module().unwrap();
        let lean = String::from_utf8(module.contents).unwrap();
        assert!(lean.contains(
            "(1 - cycle.JoltR1CSInputs.NextIsNoop)\n    (1 - next_cycle.SpartanShift_Vars.InstructionFlags_IsNoop)"
        ));
        assert!(!lean.contains("NextIsNoop : ZKExpr f"));
    }

    /// The write-protection claim keeps its mask C, so the model forbids a
    /// value-changing store only on a word `memory_step` marks protected.
    #[test]
    fn ram_write_protection_claim_keeps_its_mask() {
        let module = ZkLeanSumchecks::extract::<64>().as_module().unwrap();
        let lean = String::from_utf8(module.contents).unwrap();
        assert!(lean.contains("\n  WriteProtected : ZKExpr f\n"));
        assert!(lean.contains(
            "    0\n    (cycle.RamReadWriteChecking_Vars.WriteProtected * (cycle.RamReadWriteChecking_Vars.RamRa * cycle.RamReadWriteChecking_Vars.RamInc))"
        ));
    }
}
