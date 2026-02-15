use std::collections::HashSet;

use crate::poly::opening_proof::OpeningId;

use super::{OutputClaimConstraint, StageConfig};

#[derive(Debug)]
pub enum LayoutStep<'a> {
    /// Chain start where the initial claim is a witness variable
    InitialClaimVar {
        chain_idx: usize,
    },
    /// Chain start where the initial claim is a baked constant (no variable allocated)
    ConstantInitialClaim {
        chain_idx: usize,
    },
    ConstraintVars {
        constraint: &'a OutputClaimConstraint,
        new_opening_count: usize,
        aux_var_count: usize,
        kind: ConstraintKind,
        stage_idx: usize,
    },
    CoeffRow {
        round_idx: usize,
        num_coeffs: usize,
        stage_idx: usize,
        round_in_stage: usize,
    },
    NextClaim {
        stage_idx: usize,
        round_in_stage: usize,
    },
    LinearFinalOutput {
        num_evaluations: usize,
        stage_idx: usize,
    },
    PlaceholderVars {
        num_vars: usize,
    },
    /// Extra constraint: openings(new_opening_count) + output(1) + aux(aux_var_count) + blinding(1)
    ExtraConstraintVars {
        constraint: &'a OutputClaimConstraint,
        new_opening_count: usize,
        aux_var_count: usize,
        extra_idx: usize,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConstraintKind {
    InitialInput,
    FinalOutput,
}

pub fn compute_witness_layout<'a>(
    stage_configs: &'a [StageConfig],
    extra_constraints: &'a [OutputClaimConstraint],
) -> Vec<LayoutStep<'a>> {
    let mut steps = Vec::new();
    let mut round_idx = 0usize;
    let mut chain_idx = 0usize;
    let mut seen_openings: HashSet<OpeningId> = HashSet::new();

    for (stage_idx, config) in stage_configs.iter().enumerate() {
        let is_chain_start = stage_idx == 0 || config.starts_new_chain;
        if stage_idx > 0 && config.starts_new_chain {
            chain_idx += 1;
        }

        if is_chain_start {
            if config.has_initial_claim_var() {
                steps.push(LayoutStep::InitialClaimVar { chain_idx });
            } else {
                steps.push(LayoutStep::ConstantInitialClaim { chain_idx });
            }
        }

        if let Some(ref ii) = config.initial_input {
            if let Some(ref constraint) = ii.constraint {
                let new_opening_count = count_new_openings(constraint, &mut seen_openings);
                let aux_var_count = constraint.estimate_aux_var_count();
                steps.push(LayoutStep::ConstraintVars {
                    constraint,
                    new_opening_count,
                    aux_var_count,
                    kind: ConstraintKind::InitialInput,
                    stage_idx,
                });
            }
        }

        for round_in_stage in 0..config.num_rounds {
            steps.push(LayoutStep::CoeffRow {
                round_idx,
                num_coeffs: config.poly_degree + 1,
                stage_idx,
                round_in_stage,
            });
            steps.push(LayoutStep::NextClaim {
                stage_idx,
                round_in_stage,
            });
            round_idx += 1;
        }

        if let Some(ref fout) = config.final_output {
            if let Some(exact) = fout.exact_num_witness_vars {
                steps.push(LayoutStep::PlaceholderVars { num_vars: exact });
            } else if let Some(ref constraint) = fout.constraint {
                let new_opening_count = count_new_openings(constraint, &mut seen_openings);
                let aux_var_count = constraint.estimate_aux_var_count();
                steps.push(LayoutStep::ConstraintVars {
                    constraint,
                    new_opening_count,
                    aux_var_count,
                    kind: ConstraintKind::FinalOutput,
                    stage_idx,
                });
            } else {
                steps.push(LayoutStep::LinearFinalOutput {
                    num_evaluations: fout.num_evaluations,
                    stage_idx,
                });
            }
        }
    }

    for (extra_idx, constraint) in extra_constraints.iter().enumerate() {
        let new_opening_count = count_new_openings(constraint, &mut seen_openings);
        let aux_var_count = constraint.estimate_aux_var_count();
        steps.push(LayoutStep::ExtraConstraintVars {
            constraint,
            new_opening_count,
            aux_var_count,
            extra_idx,
        });
    }

    steps
}

fn count_new_openings(constraint: &OutputClaimConstraint, seen: &mut HashSet<OpeningId>) -> usize {
    constraint
        .required_openings
        .iter()
        .filter(|id| seen.insert(**id))
        .count()
}
