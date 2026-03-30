//! Emit pass: transform [`Staging`] into [`CompilerOutput`].
//!
//! For each stage, simultaneously produces:
//! - [`KernelSpec`] + [`ProverStep`] sequence (prover schedule)
//! - [`VerifierStage`] (verifier script)

use crate::formula::{
    BindingOrder, CompositionFormula, Factor as FormulaFactor, ProductTerm,
};
use crate::ir::expr::Factor as ExprFactor;
use crate::ir::{PolyKind, PublicPoly, Vertex};
use crate::output::*;

use super::cost::CompileParams;
use super::stage::{StagePlan, Staging};

/// Emit the final compiler output from a staged protocol.
pub(crate) fn emit(staging: &Staging, params: &CompileParams) -> CompilerOutput {
    let protocol = &staging.protocol;
    let mut ctx = EmitCtx::new(protocol, params);

    // Preamble: committed polys
    let committed: Vec<usize> = protocol
        .polynomials
        .iter()
        .enumerate()
        .filter(|(_, p)| matches!(p.kind, PolyKind::Committed))
        .map(|(i, _)| i)
        .collect();
    if !committed.is_empty() {
        ctx.prover_steps
            .push(ProverStep::AppendCommitments { polys: committed.clone() });
    }

    // Per-stage Fiat-Shamir challenges allocated after each stage
    let stage_challenges: Vec<usize> = (0..staging.stages.len())
        .map(|si| {
            let ch_idx = ctx.challenges.len();
            ctx.challenges.push(ChallengeSpec {
                name: format!("alpha_s{si}"),
                source: ChallengeSource::FiatShamir { after_stage: si },
            });
            ch_idx
        })
        .collect();

    // External challenges from protocol
    let _external_challenges: Vec<usize> = protocol
        .challenge_names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let ch_idx = ctx.challenges.len();
            ctx.challenges.push(ChallengeSpec {
                name: name.clone(),
                source: ChallengeSource::External,
            });
            // Map protocol challenge index → output challenge index
            let _ = ctx.proto_challenge_map.insert(i, ch_idx);
            ch_idx
        })
        .collect();

    // Emit each stage
    for (si, plan) in staging.stages.iter().enumerate() {
        let verifier_commitments = if si == 0 { committed.clone() } else { vec![] };

        if !plan.vertices.is_empty() {
            emit_sumcheck_stage(&mut ctx, si, plan, &stage_challenges);
        }

        // Evaluate vertices
        let mut eval_specs = Vec::new();
        for &vi in &plan.evaluations {
            if let Vertex::Evaluate { poly, at_vertex, .. } = &protocol.vertices[vi] {
                ctx.prover_steps
                    .push(ProverStep::Evaluate { poly: *poly });
                eval_specs.push(EvalSpec {
                    poly: *poly,
                    at_vertex: *at_vertex,
                });
            }
        }

        if !plan.evaluations.is_empty() {
            let eval_indices: Vec<usize> = plan
                .evaluations
                .iter()
                .map(|&vi| {
                    if let Vertex::Evaluate { poly, .. } = &protocol.vertices[vi] {
                        *poly
                    } else {
                        0
                    }
                })
                .collect();
            ctx.prover_steps.push(ProverStep::AppendScalars {
                evals: eval_indices,
            });
        }

        // Post-stage challenges
        let mut post_squeeze = Vec::new();
        if si < stage_challenges.len() {
            ctx.prover_steps.push(ProverStep::Squeeze {
                challenge: stage_challenges[si],
            });
            post_squeeze.push(stage_challenges[si]);
        }

        // Build verifier stage (only for stages with sumcheck vertices)
        if !plan.vertices.is_empty() {
            let num_rounds = ctx.last_stage_rounds;
            let degree = ctx.last_stage_degree;
            ctx.verifier_stages.push(VerifierStage {
                commitments: verifier_commitments,
                input_claim: build_input_claim_formula(protocol, plan, &ctx),
                num_rounds,
                degree,
                evaluations: eval_specs,
                post_squeeze,
            });
        } else if !plan.evaluations.is_empty() {
            ctx.verifier_stages.push(VerifierStage {
                commitments: verifier_commitments,
                input_claim: ClaimFormula::zero(),
                num_rounds: 0,
                degree: 0,
                evaluations: eval_specs,
                post_squeeze,
            });
        }
    }

    // Emit opening stage if present
    if let Some(opening) = &staging.opening {
        let ch_idx = ctx.challenges.len();
        ctx.challenges.push(ChallengeSpec {
            name: opening.challenge_name.clone(),
            source: ChallengeSource::FiatShamir {
                after_stage: staging.stages.len(),
            },
        });
        ctx.prover_steps
            .push(ProverStep::Squeeze { challenge: ch_idx });
        ctx.verifier_stages.push(VerifierStage {
            commitments: vec![],
            input_claim: ClaimFormula::zero(),
            num_rounds: 0,
            degree: 0,
            evaluations: vec![],
            post_squeeze: vec![ch_idx],
        });
    }

    // Build poly specs
    let polys: Vec<PolySpec> = protocol
        .polynomials
        .iter()
        .map(|p| {
            let num_elements = poly_num_elements_from_dims(&p.dims, params);
            PolySpec {
                name: p.name.clone(),
                kind: p.kind.clone(),
                num_elements,
            }
        })
        .collect();

    CompilerOutput {
        polys,
        challenges: ctx.challenges,
        schedule: ProverSchedule {
            steps: ctx.prover_steps,
            kernels: ctx.kernels,
        },
        script: VerifierScript {
            stages: ctx.verifier_stages,
        },
    }
}

// ---------------------------------------------------------------------------
// Emit context
// ---------------------------------------------------------------------------

struct EmitCtx<'a> {
    protocol: &'a crate::ir::Protocol,
    params: &'a CompileParams,
    prover_steps: Vec<ProverStep>,
    kernels: Vec<KernelSpec>,
    challenges: Vec<ChallengeSpec>,
    verifier_stages: Vec<VerifierStage>,
    /// Protocol challenge index → output challenge index
    proto_challenge_map: std::collections::HashMap<usize, usize>,
    /// Bookkeeping for verifier stage construction
    last_stage_rounds: usize,
    last_stage_degree: usize,
}

impl<'a> EmitCtx<'a> {
    fn new(protocol: &'a crate::ir::Protocol, params: &'a CompileParams) -> Self {
        Self {
            protocol,
            params,
            prover_steps: Vec::new(),
            kernels: Vec::new(),
            challenges: Vec::new(),
            verifier_stages: Vec::new(),
            proto_challenge_map: std::collections::HashMap::new(),
            last_stage_rounds: 0,
            last_stage_degree: 0,
        }
    }
}

// ---------------------------------------------------------------------------
// Sumcheck stage emission
// ---------------------------------------------------------------------------

fn emit_sumcheck_stage(
    ctx: &mut EmitCtx<'_>,
    stage_idx: usize,
    plan: &StagePlan,
    _stage_challenges: &[usize],
) {
    let protocol = ctx.protocol;
    let params = ctx.params;

    let all_polys = collect_stage_polys(protocol, plan);

    // Materialize public polys
    for &pi in &all_polys {
        if should_materialize(&protocol.polynomials[pi].kind) {
            ctx.prover_steps
                .push(ProverStep::Materialize { poly: pi });
        }
    }

    // Union binding order → round count
    let union_dims = union_binding_dims(protocol, plan);
    let num_rounds: usize = union_dims
        .iter()
        .map(|&d| params.dim_sizes[d] as usize)
        .sum();

    let round_to_dim: Vec<usize> = union_dims
        .iter()
        .flat_map(|&d| std::iter::repeat_n(d, params.dim_sizes[d] as usize))
        .collect();

    let degree = plan
        .vertices
        .iter()
        .map(|&vi| vertex_degree(&protocol.vertices[vi]))
        .max()
        .unwrap_or(0);

    ctx.last_stage_rounds = num_rounds;
    ctx.last_stage_degree = degree;

    // Detect uni-skip first round
    let uniskip_domain = plan.vertices.iter().find_map(|&vi| {
        protocol.vertices[vi].domain_size()
    });

    // Build KernelSpec for this stage
    let (formula, input_mapping) = build_composition_formula(protocol, plan);
    let kernel_idx = ctx.kernels.len();

    // Determine eq mode: if the composition references a public eq poly
    // that's in the input list, it's AsInput. Otherwise Unit.
    let eq_input_idx = input_mapping
        .iter()
        .position(|&pi| matches!(protocol.polynomials[pi].kind, PolyKind::Public(PublicPoly::Eq(_))));
    let eq_mode = match eq_input_idx {
        Some(idx) => EqMode::AsInput(idx),
        None => EqMode::Unit,
    };

    ctx.kernels.push(KernelSpec {
        formula,
        inputs: input_mapping,
        eq_mode,
        binding_order: BindingOrder::LowToHigh,
        num_rounds,
        degree,
    });

    // Emit round steps
    for (r, &active_dim) in round_to_dim.iter().enumerate() {
        let dim_polys: Vec<usize> = all_polys
            .iter()
            .filter(|&&pi| protocol.polynomials[pi].dims.contains(&active_dim))
            .copied()
            .collect();

        let num_coeffs = if r == 0 {
            uniskip_domain.unwrap_or(degree + 1)
        } else {
            degree + 1
        };

        // Allocate round challenge
        let ch_idx = ctx.challenges.len();
        ctx.challenges.push(ChallengeSpec {
            name: format!("r_s{stage_idx}_r{r}"),
            source: ChallengeSource::SumcheckRound {
                stage: stage_idx,
                round: r,
            },
        });

        ctx.prover_steps.push(ProverStep::SumcheckRound {
            kernel: kernel_idx,
            round: r,
            num_vars_remaining: num_rounds - r,
        });
        ctx.prover_steps.push(ProverStep::AppendRoundPoly {
            kernel: kernel_idx,
            num_coeffs,
        });
        ctx.prover_steps
            .push(ProverStep::Squeeze { challenge: ch_idx });
        ctx.prover_steps.push(ProverStep::Bind {
            polys: dim_polys,
            challenge: ch_idx,
            order: BindingOrder::LowToHigh,
        });
    }
}

// ---------------------------------------------------------------------------
// Formula conversion: protocol Expr → CompositionFormula
// ---------------------------------------------------------------------------

/// Convert a stage's composition expressions to a single CompositionFormula.
/// Returns the formula and the poly index mapping (input_i → protocol poly index).
fn build_composition_formula(
    protocol: &crate::ir::Protocol,
    plan: &StagePlan,
) -> (CompositionFormula, Vec<usize>) {
    // Collect all poly indices across all vertices in this stage
    let mut all_poly_indices: Vec<usize> = Vec::new();
    for &vi in &plan.vertices {
        if let Some(comp) = protocol.vertices[vi].composition() {
            for term in &comp.0 {
                for f in &term.factors {
                    if let ExprFactor::Poly(idx) = f {
                        all_poly_indices.push(*idx);
                    }
                }
            }
        }
    }
    all_poly_indices.sort_unstable();
    all_poly_indices.dedup();

    // Build mapping: protocol poly index → formula input index
    let poly_to_input: std::collections::HashMap<usize, u32> = all_poly_indices
        .iter()
        .enumerate()
        .map(|(i, &pi)| (pi, i as u32))
        .collect();

    // Convert each vertex's composition terms
    let mut all_terms = Vec::new();
    for &vi in &plan.vertices {
        if let Some(comp) = protocol.vertices[vi].composition() {
            for term in &comp.0 {
                let mut factors = Vec::new();
                for f in &term.factors {
                    match f {
                        ExprFactor::Poly(idx) => {
                            factors.push(FormulaFactor::Input(poly_to_input[idx]));
                        }
                        ExprFactor::Challenge(idx) => {
                            factors.push(FormulaFactor::Challenge(*idx as u32));
                        }
                        ExprFactor::Claim(_) => {
                            // Claims are runtime values — they become challenge slots
                            // in the formula (resolved by the prover at runtime).
                            // For now, skip them — they appear in input_sum, not composition.
                        }
                    }
                }
                all_terms.push(ProductTerm {
                    coefficient: term.coeff as i128,
                    factors,
                });
            }
        }
    }

    (
        CompositionFormula::from_terms(all_terms),
        all_poly_indices,
    )
}

// ---------------------------------------------------------------------------
// Verifier claim formula construction
// ---------------------------------------------------------------------------

fn build_input_claim_formula(
    protocol: &crate::ir::Protocol,
    plan: &StagePlan,
    ctx: &EmitCtx<'_>,
) -> ClaimFormula {
    // For zero-check vertices (input_sum is 0), the claim is zero.
    // For non-zero input sums, build a symbolic formula over eval values and challenges.
    let mut terms = Vec::new();

    for &vi in &plan.vertices {
        if let Some(input_sum) = protocol.vertices[vi].input_sum() {
            for term in &input_sum.0 {
                let mut factors = Vec::new();
                for f in &term.factors {
                    match f {
                        ExprFactor::Claim(cid) => {
                            // The eval index is the claim's poly index for now.
                            // In a full implementation, this would map to the
                            // accumulated eval vector position.
                            factors.push(ClaimFactor::Eval(cid.0 as usize));
                        }
                        ExprFactor::Challenge(idx) => {
                            if let Some(&mapped) = ctx.proto_challenge_map.get(idx) {
                                factors.push(ClaimFactor::Challenge(mapped));
                            }
                        }
                        ExprFactor::Poly(_) => {
                            // Polys don't appear in input_sum expressions
                        }
                    }
                }
                terms.push(ClaimTerm {
                    coeff: term.coeff as i128,
                    factors,
                });
            }
        }
    }

    ClaimFormula { terms }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn collect_stage_polys(protocol: &crate::ir::Protocol, plan: &StagePlan) -> Vec<usize> {
    let mut polys = Vec::new();
    for &vi in &plan.vertices {
        if let Some(comp) = protocol.vertices[vi].composition() {
            for term in &comp.0 {
                for f in &term.factors {
                    if let ExprFactor::Poly(idx) = f {
                        polys.push(*idx);
                    }
                }
            }
        }
    }
    polys.sort_unstable();
    polys.dedup();
    polys
}

fn union_binding_dims(protocol: &crate::ir::Protocol, plan: &StagePlan) -> Vec<usize> {
    let mut dims: Vec<usize> = plan
        .vertices
        .iter()
        .filter_map(|&vi| protocol.vertices[vi].binding_order())
        .flat_map(|bo| bo.iter().copied())
        .collect();
    dims.sort_unstable();
    dims.dedup();
    dims
}

fn should_materialize(kind: &PolyKind) -> bool {
    match kind {
        PolyKind::Public(pp) => !matches!(pp, PublicPoly::Preprocessed | PublicPoly::Identity),
        _ => false,
    }
}

fn poly_num_elements_from_dims(dims: &[usize], params: &CompileParams) -> usize {
    let total_vars: u64 = dims.iter().map(|&d| params.dim_sizes[d]).sum();
    if total_vars >= 63 {
        usize::MAX
    } else {
        1usize << total_vars
    }
}

fn vertex_degree(vertex: &Vertex) -> usize {
    match vertex {
        Vertex::Sumcheck { composition, .. } => {
            composition
                .0
                .iter()
                .map(|term| {
                    term.factors
                        .iter()
                        .filter(|f| matches!(f, ExprFactor::Poly(_)))
                        .count()
                })
                .max()
                .unwrap_or(0)
        }
        Vertex::Evaluate { .. } => 0,
    }
}
