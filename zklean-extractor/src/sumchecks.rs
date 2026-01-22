use std::collections::HashMap;

use jolt_core::{
    field::JoltField,
    poly::opening_proof::{PolynomialId, SumcheckId},
    subprotocols::sumcheck_claim::{
        Claim, ClaimExpr, InputOutputClaims, SumcheckFrontend, VerifierEvaluablePolynomial,
    },
    zkvm::{
        ram::read_write_checking::RamReadWriteCheckingVerifier,
        registers::read_write_checking::RegistersReadWriteCheckingVerifier,
        spartan::{
            instruction_input::InstructionInputSumcheckVerifier, shift::ShiftSumcheckVerifier,
        },
        witness::{CommittedPolynomial, VirtualPolynomial},
    },
};
use regex::{NoExpand, Regex};

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
        if !self.vars.iter().any(|v| v == var) {
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
                self.insert_vars_from_claim_expr(sumcheck_id, e1);
                self.insert_vars_from_claim_expr(sumcheck_id, e2);
            }
            ClaimExpr::Var(PolynomialId::Committed(committed_polynomial)) => {
                let var_name = committed_var_ident(committed_polynomial);
                self.sumchecks
                    .entry(sumcheck_id)
                    .and_modify(|s| s.insert_var(&var_name))
                    .or_insert_with(|| ZkLeanSumcheck {
                        claims: None,
                        vars: vec![var_name],
                    });
            }
            ClaimExpr::Var(PolynomialId::Virtual(virtual_polynomial)) => {
                let var_name = virtual_var_ident(virtual_polynomial);
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
        let sumcheck_vars_ty = "SumcheckVars";
        let single_step_claims_fun_name = "uniform_claims";
        let cross_step_claims_fun_name = "non_uniform_claims";

        // NOTE We special-case the SpartanOuter sumcheck to refer to the R1CS constraints, so we
        // don't need this entry
        let mut sumchecks = self.clone();
        sumchecks.sumchecks.remove(&SumcheckId::SpartanOuter);

        let mut vars_types: Vec<String> = vec![];
        for (id, sumcheck) in &sumchecks.sumchecks {
            if !sumcheck.vars.is_empty() {
                let typename = sumcheck_ident(id);
                vars_types.push(typename.clone());
                writeln!(
                    f,
                    "{}structure {typename} (f : Type) : Type where",
                    indent(indent_level)
                )?;

                indent_level += 1;
                for var in &sumcheck.vars {
                    writeln!(f, "{}{var} : ZKExpr f", indent(indent_level))?;
                }
                indent_level -= 1;
            }
            writeln!(f)?;
        }

        writeln!(
            f,
            "{}structure {sumcheck_vars_ty} (f : Type) : Type where",
            indent(indent_level)
        )?;
        indent_level += 1;
        writeln!(
            f,
            "{}JoltR1CSInputs : JoltR1CSInputs f",
            indent(indent_level)
        )?;
        for ty in vars_types {
            writeln!(f, "{}{ty} : {ty} f", indent(indent_level))?;
        }
        indent_level -= 1;
        writeln!(f)?;

        let mut single_step_claims_funs: Vec<String> = vec![];
        let mut cross_step_claims_funs: Vec<String> = vec![];
        for sumcheck in sumchecks.sumchecks.values() {
            if let Some(claims) = &sumcheck.claims {
                let (cross_step_claims, single_step_claims): (Vec<&Claim<F>>, Vec<&Claim<F>>) =
                    claims.claims.iter().partition(|c| {
                        // TODO This function currently only handles constraints quantified by Eq
                        // and EqPlusOne. We should handle the rest.
                        match c.batching_poly {
                            VerifierEvaluablePolynomial::Eq(_cached_point_ref) => false,
                            VerifierEvaluablePolynomial::EqPlusOne(_cached_point_ref) => true,
                            VerifierEvaluablePolynomial::Lt(_cached_point_ref) => todo!(),
                            VerifierEvaluablePolynomial::Identity => todo!(),
                            VerifierEvaluablePolynomial::UnmapRamAddress => todo!(),
                            VerifierEvaluablePolynomial::One => todo!(),
                        }
                    });
                if !single_step_claims.is_empty() {
                    let fun_name = format!(
                        "{}.{single_step_claims_fun_name}",
                        sumcheck_ident(&claims.output_sumcheck_id)
                    );
                    single_step_claims_funs.push(fun_name.clone());
                    pretty_print_claims_fun(
                        f,
                        &fun_name,
                        sumcheck_vars_ty,
                        &single_step_claims,
                        &claims.output_sumcheck_id,
                        false,
                        indent_level,
                    )?;
                    writeln!(f)?;
                }
                if !cross_step_claims.is_empty() {
                    let fun_name = format!(
                        "{}.{cross_step_claims_fun_name}",
                        sumcheck_ident(&claims.output_sumcheck_id)
                    );
                    cross_step_claims_funs.push(fun_name.clone());
                    pretty_print_claims_fun(
                        f,
                        &fun_name,
                        sumcheck_vars_ty,
                        &cross_step_claims,
                        &claims.output_sumcheck_id,
                        true,
                        indent_level,
                    )?;
                    writeln!(f)?;
                }
            }
        }

        writeln!(
            f,
            "{}def {single_step_claims_fun_name} [Field f] (cycle : {sumcheck_vars_ty} f) : ZKBuilder f PUnit := do",
            indent(indent_level)
        )?;
        indent_level += 1;
        for fun_name in &single_step_claims_funs {
            writeln!(f, "{}{fun_name} cycle", indent(indent_level))?;
        }
        indent_level -= 1;
        writeln!(f)?;

        writeln!(
            f,
            "{}def {cross_step_claims_fun_name} [Field f] (cycle next_cycle : {sumcheck_vars_ty} f) : ZKBuilder f PUnit := do",
            indent(indent_level)
        )?;
        indent_level += 1;
        for fun_name in &cross_step_claims_funs {
            writeln!(f, "{}{fun_name} cycle next_cycle", indent(indent_level))?;
        }

        Ok(())
    }
}

fn remove_parens(mut string: String) -> String {
    let open_paren = Regex::new(r"\(").unwrap();
    let close_paren = Regex::new(r"\)").unwrap();

    string = open_paren
        .replace_all(string.as_str(), NoExpand("_"))
        .to_string();
    string = close_paren
        .replace_all(string.as_str(), NoExpand(""))
        .to_string();

    string
}

fn sumcheck_ident(sumcheck_id: &SumcheckId) -> String {
    // NOTE We special-case the SpartanOuter sumcheck to refer to the R1CS constraints
    if *sumcheck_id == SumcheckId::SpartanOuter {
        return String::from("JoltR1CSInputs");
    }

    remove_parens(format!("{sumcheck_id:?}_Vars"))
}

fn committed_var_ident(var: &CommittedPolynomial) -> String {
    remove_parens(format!("{var:?}"))
}

fn virtual_var_ident(var: &VirtualPolynomial) -> String {
    remove_parens(format!("{var:?}"))
}

fn pretty_print_claims_fun<F: JoltField>(
    f: &mut impl std::io::Write,
    fun_name: &str,
    sumcheck_vars_ty: &str,
    claims: &[&Claim<F>],
    output_sumcheck_id: &SumcheckId,
    is_offset: bool,
    mut indent_level: usize,
) -> std::io::Result<()> {
    let cycle_vars = if is_offset {
        "cycle next_cycle"
    } else {
        "cycle"
    };
    writeln!(
        f,
        "{}def {fun_name} [Field f] ({cycle_vars} : {sumcheck_vars_ty} f) : ZKBuilder f PUnit := do",
        indent(indent_level)
    )?;
    indent_level += 1;
    for claim in claims {
        writeln!(f, "{}ZKBuilder.constrainEq ", indent(indent_level))?;
        indent_level += 1;
        write!(f, "{}", indent(indent_level))?;
        pretty_print_claim_expr(
            f,
            "cycle",
            &claim.input_sumcheck_id,
            &claim.input_claim_expr,
            true,
        )?;
        writeln!(f)?;
        write!(f, "{}", indent(indent_level))?;
        pretty_print_claim_expr(
            f,
            if is_offset { "next_cycle" } else { "cycle" },
            output_sumcheck_id,
            &claim.expected_output_claim_expr,
            true,
        )?;
        writeln!(f)?;
        indent_level -= 1;
    }

    Ok(())
}

fn pretty_print_opening_ref(
    f: &mut impl std::io::Write,
    vars_ident: &str,
    sumcheck_id: &SumcheckId,
    opening_ref: &PolynomialId,
) -> std::io::Result<()> {
    let vars_type = sumcheck_ident(sumcheck_id);
    match opening_ref {
        PolynomialId::Committed(committed_polynomial) => {
            let var_name = committed_var_ident(committed_polynomial);
            write!(f, "{vars_ident}.{vars_type}.{var_name}")?;
        }
        PolynomialId::Virtual(virtual_polynomial) => {
            let var_name = virtual_var_ident(virtual_polynomial);
            write!(f, "{vars_ident}.{vars_type}.{var_name}")?;
        }
    }

    Ok(())
}

fn pretty_print_claim_expr<F: JoltField>(
    f: &mut impl std::io::Write,
    vars_ident: &str,
    sumcheck_id: &SumcheckId,
    expr: &ClaimExpr<F>,
    group: bool,
) -> std::io::Result<()> {
    match expr {
        ClaimExpr::Constant(val) => {
            write!(f, "{val}")?;
        }
        ClaimExpr::Add(e1, e2) => {
            if group {
                write!(f, "(")?;
            }
            pretty_print_claim_expr(f, vars_ident, sumcheck_id, e1, false)?;
            write!(f, " + ")?;
            pretty_print_claim_expr(f, vars_ident, sumcheck_id, e2, false)?;
            if group {
                write!(f, ")")?;
            }
        }
        ClaimExpr::Mul(e1, e2) => {
            if group {
                write!(f, "(")?;
            }
            pretty_print_claim_expr(f, vars_ident, sumcheck_id, e1, true)?;
            write!(f, " * ")?;
            pretty_print_claim_expr(f, vars_ident, sumcheck_id, e2, true)?;
            if group {
                write!(f, ")")?;
            }
        }
        ClaimExpr::Sub(e1, e2) => {
            if group {
                write!(f, "(")?;
            }
            pretty_print_claim_expr(f, vars_ident, sumcheck_id, e1, false)?;
            write!(f, " - ")?;
            pretty_print_claim_expr(f, vars_ident, sumcheck_id, e2, true)?;
            if group {
                write!(f, ")")?;
            }
        }
        ClaimExpr::Var(opening_ref) => {
            pretty_print_opening_ref(f, vars_ident, sumcheck_id, opening_ref)?;
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
            imports: vec![String::from("ZkLean"), String::from("Jolt.R1CS")],
            contents,
        })
    }
}
