use std::collections::HashMap;

use jolt_core::{
    field::JoltField,
    poly::opening_proof::SumcheckId,
    subprotocols::sumcheck_claim::{ClaimExpr, InputOutputClaims, SumcheckFrontend},
    zkvm::{
        ram::read_write_checking::RamReadWriteCheckingVerifier,
        registers::read_write_checking::RegistersReadWriteCheckingVerifier,
        spartan::{
            instruction_input::InstructionInputSumcheckVerifier, shift::ShiftSumcheckVerifier,
        },
    },
};

use crate::{
    modules::{AsModule, Module},
    util::indent,
};

// TODO Use EnumIter for this
fn all_sumcheck_claims<F: JoltField>() -> Vec<InputOutputClaims<F>> {
    vec![
        RamReadWriteCheckingVerifier::input_output_claims(),
        RegistersReadWriteCheckingVerifier::input_output_claims(),
        InstructionInputSumcheckVerifier::input_output_claims(),
        ShiftSumcheckVerifier::input_output_claims(),
    ]
}

#[derive(Debug, Clone)]
pub struct ZkLeanSumcheck<F: JoltField> {
    claims: Option<InputOutputClaims<F>>,
    vars: Vec<String>,
}

impl<F: JoltField> ZkLeanSumcheck<F> {
    fn insert_var(&mut self, var: &String) {
        // TODO Use something better than linear search?
        if self.vars.iter().position(|v| v == var).is_none() {
            self.vars.push(var.clone())
        }
    }
}

#[derive(Debug, Clone)]
pub struct ZkLeanSumchecks<F: JoltField> {
    sumchecks: HashMap<SumcheckId, ZkLeanSumcheck<F>>,
}

impl<F: JoltField> ZkLeanSumchecks<F> {
    fn empty() -> Self {
        Self {
            sumchecks: HashMap::new(),
        }
    }

    fn insert_vars_from_claim_expr(&mut self, sumcheck_id: SumcheckId, expr: &ClaimExpr<F>) {
        match expr {
            ClaimExpr::Add(e1, e2) | ClaimExpr::Mul(e1, e2) | ClaimExpr::Sub(e1, e2) => {
                self.insert_vars_from_claim_expr(sumcheck_id, &e1);
                self.insert_vars_from_claim_expr(sumcheck_id, &e2);
            }
            ClaimExpr::CommittedVar(committed_polynomial) => {
                let var_name = std::format!("{committed_polynomial:?}");
                self.sumchecks
                    .entry(sumcheck_id)
                    .and_modify(|s| s.insert_var(&var_name))
                    .or_insert_with(|| ZkLeanSumcheck {
                        claims: None,
                        vars: vec![var_name],
                    });
            }
            ClaimExpr::VirtualVar(virtual_polynomial) => {
                let var_name = std::format!("{virtual_polynomial:?}");
                self.sumchecks
                    .entry(sumcheck_id)
                    .and_modify(|s| s.insert_var(&var_name))
                    .or_insert_with(|| ZkLeanSumcheck {
                        claims: None,
                        vars: vec![var_name],
                    });
            }
            _ => (),
        }
    }

    fn insert_claims(&mut self, claims: &InputOutputClaims<F>) {
        self.sumchecks
            .entry(claims.output_sumcheck_id)
            .insert_entry(ZkLeanSumcheck {
                claims: Some(claims.clone()),
                vars: vec![],
            });
        for claim in &claims.claims {
            self.insert_vars_from_claim_expr(
                claims.output_sumcheck_id,
                &claim.expected_output_claim_expr,
            );
            self.insert_vars_from_claim_expr(claim.input_sumcheck_id, &claim.input_claim_expr);
        }
    }

    pub fn extract() -> Self {
        let mut res = Self::empty();
        for claims in all_sumcheck_claims() {
            res.insert_claims(&claims)
        }
        res
    }

    pub fn zklean_pretty_print(
        &self,
        f: &mut impl std::io::Write,
        mut indent_level: usize,
    ) -> std::io::Result<()> {
        writeln!(f, "{}/-", indent(indent_level))?;

        for (id, sumcheck) in &self.sumchecks {
            if sumcheck.vars.len() > 0 {
                writeln!(f, "{}struct {id:?} [f: field]", indent(indent_level))?;
                indent_level += 1;
                for var in &sumcheck.vars {
                    writeln!(f, "{}{var} : f", indent(indent_level))?;
                }
                indent_level -= 1;
            }
            writeln!(f, "")?;
        }

        writeln!(f, "{}def claims [f : field] : bool", indent(indent_level))?;
        indent_level += 1;
        for (_, sumcheck) in &self.sumchecks {
            match &sumcheck.claims {
                Some(claims) => {
                    for claim in &claims.claims {
                        write!(f, "{}", indent(indent_level))?;
                        pretty_print_claim_expr(
                            f,
                            claim.input_sumcheck_id,
                            &claim.input_claim_expr,
                        )?;
                        write!(f, " == ")?;
                        pretty_print_claim_expr(
                            f,
                            claim.input_sumcheck_id,
                            &claim.input_claim_expr,
                        )?;
                        writeln!(f, "")?;
                    }
                }
                None => (),
            }
        }
        indent_level -= 1;

        writeln!(f, "{}-/", indent(indent_level))
    }
}

fn pretty_print_claim_expr<F: JoltField>(
    f: &mut impl std::io::Write,
    sumcheck_id: SumcheckId,
    expr: &ClaimExpr<F>,
) -> std::io::Result<()> {
    match expr {
        ClaimExpr::Val(val) => {
            write!(f, "{val}")?;
        }
        ClaimExpr::Add(e1, e2) => {
            write!(f, "(")?;
            pretty_print_claim_expr(f, sumcheck_id, e1)?;
            write!(f, " + ")?;
            pretty_print_claim_expr(f, sumcheck_id, e2)?;
            write!(f, ")")?;
        }
        ClaimExpr::Mul(e1, e2) => {
            pretty_print_claim_expr(f, sumcheck_id, e1)?;
            write!(f, " * ")?;
            pretty_print_claim_expr(f, sumcheck_id, e2)?;
        }
        ClaimExpr::Sub(e1, e2) => {
            write!(f, "(")?;
            pretty_print_claim_expr(f, sumcheck_id, e1)?;
            write!(f, " - ")?;
            pretty_print_claim_expr(f, sumcheck_id, e2)?;
            write!(f, ")")?;
        }
        ClaimExpr::CommittedVar(committed_polynomial) => {
            let var_name = std::format!("{committed_polynomial:?}");
            write!(f, "{sumcheck_id:?}.{var_name}")?;
        }
        ClaimExpr::VirtualVar(virtual_polynomial) => {
            let var_name = std::format!("{virtual_polynomial:?}");
            write!(f, "{sumcheck_id:?}.{var_name}")?;
        }
    }

    Ok(())
}

impl<F: JoltField> AsModule for ZkLeanSumchecks<F> {
    fn as_module(&self) -> std::io::Result<Module> {
        let mut contents: Vec<u8> = vec![];
        self.zklean_pretty_print(&mut contents, 0)?;

        Ok(Module {
            name: String::from("Sumchecks"),
            imports: vec![],
            contents,
        })
    }
}
