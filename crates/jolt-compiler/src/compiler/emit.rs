//! Emit pass: transform [`Staging`] into [`Module`].
//!
//! For each stage, simultaneously produces:
//! - [`KernelDef`] + [`Op`] sequence (prover schedule)
//! - [`VerifierOp`] sequence (verifier schedule)

use std::collections::HashSet;

use crate::formula::{BindingOrder, Factor as FormulaFactor, Formula, ProductTerm};
use crate::ir::expr::Factor as ExprFactor;
use crate::ir::{Density, PolyKind, PublicPoly, Vertex};
use crate::kernel_spec::{Iteration, KernelSpec};
use crate::module::{
    ChallengeDecl, ChallengeIdx, ChallengeSource, ClaimFactor, ClaimFormula, ClaimTerm,
    DomainSeparator, Evaluation, InputBinding, KernelDef, Module, Op, PolyDecl, Schedule,
    SumcheckInstance, VerifierOp, VerifierSchedule, VerifierStageIndex,
};
use crate::polynomial_id::PolynomialId;

use super::cost::CompileParams;
use super::stage::{StagePlan, Staging};

/// Emit the final compiled module from a staged protocol.
///
/// `poly_map` bridges protocol polynomial indices to concrete
/// [`PolynomialId`] values — every poly reference in the output
/// Module goes through this mapping.
pub(crate) fn emit(staging: &Staging, params: &CompileParams, poly_map: &[PolynomialId]) -> Module {
    let protocol = &staging.protocol;
    let mut ctx = EmitCtx::new(protocol, params, poly_map);

    // Preamble: committed polys
    let committed: Vec<usize> = protocol
        .polynomials
        .iter()
        .enumerate()
        .filter(|(_, p)| matches!(p.kind, PolyKind::Committed))
        .map(|(i, _)| i)
        .collect();
    if !committed.is_empty() {
        let num_vars = committed
            .iter()
            .map(|&i| {
                protocol.polynomials[i]
                    .dims
                    .iter()
                    .map(|&d| params.dim_sizes[d] as usize)
                    .sum::<usize>()
            })
            .max()
            .unwrap_or(0);
        ctx.ops.push(Op::Commit {
            polys: committed.iter().map(|&i| ctx.map_poly(i)).collect(),
            tag: DomainSeparator::Commitment,
            num_vars,
        });
    }

    // Verifier preamble
    ctx.verifier_ops.push(VerifierOp::Preamble);
    for &pi in &committed {
        ctx.verifier_ops.push(VerifierOp::AbsorbCommitment {
            poly: ctx.map_poly(pi),
            tag: DomainSeparator::Commitment,
        });
    }

    // Per-stage Fiat-Shamir challenges allocated after each stage
    let stage_challenges: Vec<ChallengeIdx> = (0..staging.stages.len())
        .map(|si| {
            let ch_idx = ChallengeIdx(ctx.challenges.len());
            ctx.challenges.push(ChallengeDecl {
                name: format!("alpha_s{si}"),
                source: ChallengeSource::FiatShamir { after_stage: si },
            });
            ch_idx
        })
        .collect();

    // External challenges from protocol
    let _external_challenges: Vec<ChallengeIdx> = protocol
        .challenge_names
        .iter()
        .enumerate()
        .map(|(i, name)| {
            let ch_idx = ChallengeIdx(ctx.challenges.len());
            ctx.challenges.push(ChallengeDecl {
                name: name.clone(),
                source: ChallengeSource::External,
            });
            let _ = ctx.proto_challenge_map.insert(i, ch_idx);
            ch_idx
        })
        .collect();

    // Build vertex → staging stage mapping for resolving evaluation points.
    let vertex_to_staging: std::collections::HashMap<usize, usize> = staging
        .stages
        .iter()
        .enumerate()
        .flat_map(|(si, plan)| plan.vertices.iter().map(move |&vi| (vi, si)))
        .collect();
    let mut staging_to_verifier: Vec<Option<usize>> = vec![None; staging.stages.len()];

    // Deferred output checks: (instances, stage_index) emitted after all
    // referenced evaluations are available.
    let mut pending_output_checks: Vec<(Vec<SumcheckInstance>, usize)> = Vec::new();

    // Emit each stage — verifier ops are pushed in forward order:
    // BeginStage → VerifySumcheck → RecordEvals → AbsorbEvals → CollectOpeningClaim → Squeeze
    for (si, plan) in staging.stages.iter().enumerate() {
        let has_sumcheck = !plan.vertices.is_empty();
        let has_evals = !plan.evaluations.is_empty();

        if has_sumcheck || has_evals {
            let verifier_stage_idx = ctx.verifier_stage_count;
            ctx.ops.push(Op::BeginStage {
                index: verifier_stage_idx,
            });
            ctx.verifier_ops.push(VerifierOp::BeginStage);
        }

        if has_sumcheck {
            let verifier_stage_idx = ctx.verifier_stage_count;
            staging_to_verifier[si] = Some(verifier_stage_idx);

            emit_sumcheck_stage(&mut ctx, si, plan, &staging.stages);

            let instances = vec![SumcheckInstance {
                input_claim: build_input_claim_formula(protocol, plan, &ctx),
                output_check: build_output_check_formula(protocol, plan, &ctx),
                num_rounds: ctx.last_stage_rounds,
                degree: ctx.last_stage_degree,
                normalize: None,
            }];

            ctx.verifier_ops.push(VerifierOp::VerifySumcheck {
                instances: instances.clone(),
                stage: verifier_stage_idx,
                batch_challenges: Vec::new(),
                claim_tag: None,
            });

            let has_output = instances.iter().any(|i| !i.output_check.terms.is_empty());
            if has_output {
                pending_output_checks.push((instances, verifier_stage_idx));
            }

            ctx.verifier_stage_count += 1;
        }

        let mut eval_descs = Vec::new();
        for &vi in &plan.evaluations {
            if let Vertex::Evaluate {
                poly, at_vertex, ..
            } = &protocol.vertices[vi]
            {
                let mapped = ctx.map_poly(*poly);
                ctx.ops.push(Op::Evaluate {
                    poly: mapped,
                    mode: crate::module::EvalMode::FullyBound,
                });
                let target_staging = vertex_to_staging[at_vertex];
                let at_stage = staging_to_verifier[target_staging]
                    .expect("evaluation target stage not yet emitted");
                eval_descs.push(Evaluation {
                    poly: mapped,
                    at_stage: VerifierStageIndex(at_stage),
                });
            }
        }

        if has_evals {
            let eval_polys: Vec<PolynomialId> = eval_descs.iter().map(|e| e.poly).collect();
            ctx.ops.push(Op::RecordEvals {
                polys: eval_polys.clone(),
            });
            ctx.ops.push(Op::AbsorbEvals {
                polys: eval_polys.clone(),
                tag: DomainSeparator::OpeningClaim,
            });

            ctx.verifier_ops.push(VerifierOp::RecordEvals {
                evals: eval_descs.clone(),
            });
            ctx.verifier_ops.push(VerifierOp::AbsorbEvals {
                polys: eval_polys,
                tag: DomainSeparator::OpeningClaim,
            });

            for &vi in &plan.evaluations {
                if let Vertex::Evaluate {
                    poly, at_vertex, ..
                } = &protocol.vertices[vi]
                {
                    if matches!(protocol.polynomials[*poly].kind, PolyKind::Committed) {
                        let mapped = ctx.map_poly(*poly);
                        let target_staging = vertex_to_staging[at_vertex];
                        let at_stage = staging_to_verifier[target_staging]
                            .expect("evaluation target stage not yet emitted");
                        ctx.ops.push(Op::CollectOpeningClaim {
                            poly: mapped,
                            at_stage: VerifierStageIndex(at_stage),
                        });
                        ctx.verifier_ops.push(VerifierOp::CollectOpeningClaim {
                            poly: mapped,
                            at_stage: VerifierStageIndex(at_stage),
                        });
                    }
                }
            }

            if !has_sumcheck {
                staging_to_verifier[si] = Some(ctx.verifier_stage_count);
                ctx.verifier_stage_count += 1;
            }
        }

        // Post-stage challenges
        if si < stage_challenges.len() {
            ctx.ops.push(Op::Squeeze {
                challenge: stage_challenges[si],
            });
            ctx.verifier_ops.push(VerifierOp::Squeeze {
                challenge: stage_challenges[si],
            });
        }
    }

    // Emit all output checks now that all evaluations are recorded.
    for (instances, stage) in pending_output_checks {
        ctx.verifier_ops.push(VerifierOp::CheckOutput {
            instances,
            stage,
            batch_challenges: Vec::new(),
        });
    }

    // Emit opening stage if present
    if let Some(opening) = &staging.opening {
        ctx.ops.push(Op::BeginStage {
            index: ctx.verifier_stage_count,
        });
        let ch_idx = ChallengeIdx(ctx.challenges.len());
        ctx.challenges.push(ChallengeDecl {
            name: opening.challenge_name.clone(),
            source: ChallengeSource::FiatShamir {
                after_stage: staging.stages.len(),
            },
        });
        ctx.ops.push(Op::Squeeze { challenge: ch_idx });

        ctx.verifier_ops.push(VerifierOp::BeginStage);
        ctx.verifier_ops
            .push(VerifierOp::Squeeze { challenge: ch_idx });
        ctx.verifier_stage_count += 1;
    }

    // PCS tail
    ctx.ops.push(Op::ReduceOpenings);
    ctx.ops.push(Op::Open);
    ctx.verifier_ops.push(VerifierOp::VerifyOpenings);

    if !committed.is_empty() {
        ctx.ops.push(Op::ReleaseHost {
            polys: committed.iter().map(|&i| ctx.map_poly(i)).collect(),
        });
    }

    insert_releases(&mut ctx.ops, &ctx.kernels);

    let polys: Vec<PolyDecl> = protocol
        .polynomials
        .iter()
        .map(|p| {
            let num_elements = poly_num_elements_from_dims(&p.dims, params);
            PolyDecl {
                name: p.name.clone(),
                kind: p.kind.clone(),
                num_elements,
                committed_num_vars: None,
            }
        })
        .collect();

    let num_challenges = ctx.challenges.len();
    let num_polys = polys.len();
    let num_stages = ctx.verifier_stage_count;

    Module {
        polys,
        challenges: ctx.challenges,
        prover: Schedule {
            ops: ctx.ops,
            kernels: ctx.kernels,
            batched_sumchecks: Vec::new(),
        },
        verifier: VerifierSchedule {
            ops: ctx.verifier_ops,
            num_challenges,
            num_polys,
            num_stages,
        },
    }
}

// Emit context

struct EmitCtx<'a> {
    protocol: &'a crate::ir::Protocol,
    params: &'a CompileParams,
    poly_map: &'a [PolynomialId],
    ops: Vec<Op>,
    kernels: Vec<KernelDef>,
    challenges: Vec<ChallengeDecl>,
    verifier_ops: Vec<VerifierOp>,
    verifier_stage_count: usize,
    /// Protocol challenge index → output challenge index
    proto_challenge_map: std::collections::HashMap<usize, ChallengeIdx>,
    /// Vertex index → round challenge indices for that vertex's sumcheck.
    vertex_challenges: std::collections::HashMap<usize, Vec<ChallengeIdx>>,
    last_stage_rounds: usize,
    last_stage_degree: usize,
}

impl<'a> EmitCtx<'a> {
    fn new(
        protocol: &'a crate::ir::Protocol,
        params: &'a CompileParams,
        poly_map: &'a [PolynomialId],
    ) -> Self {
        Self {
            protocol,
            params,
            poly_map,
            ops: Vec::new(),
            kernels: Vec::new(),
            challenges: Vec::new(),
            verifier_ops: Vec::new(),
            verifier_stage_count: 0,
            proto_challenge_map: std::collections::HashMap::new(),
            vertex_challenges: std::collections::HashMap::new(),
            last_stage_rounds: 0,
            last_stage_degree: 0,
        }
    }

    /// Map a protocol polynomial index to a concrete `PolynomialId`.
    #[inline]
    fn map_poly(&self, idx: usize) -> PolynomialId {
        self.poly_map[idx]
    }

    /// Resolve a `ClaimId` to the round challenge indices of the sumcheck
    /// that produced it. Returns `None` for claims whose vertex hasn't been
    /// emitted yet (shouldn't happen with correct stage ordering).
    fn claim_challenges(&self, claim_id: crate::ir::ClaimId) -> Option<&[ChallengeIdx]> {
        let claim = &self.protocol.claims[claim_id.0 as usize];
        self.vertex_challenges
            .get(&claim.produced_by)
            .map(|v| v.as_slice())
    }
}

// Sumcheck stage emission

fn emit_sumcheck_stage(
    ctx: &mut EmitCtx<'_>,
    stage_idx: usize,
    plan: &StagePlan,
    all_stages: &[StagePlan],
) {
    let protocol = ctx.protocol;
    let params = ctx.params;

    let all_polys = collect_stage_polys(protocol, plan);

    // Union binding order → round count
    let union_dims = union_binding_dims(protocol, plan);
    let num_rounds: usize = union_dims
        .iter()
        .map(|&d| params.dim_sizes[d] as usize)
        .sum();

    let degree = plan
        .vertices
        .iter()
        .map(|&vi| vertex_degree(&protocol.vertices[vi]))
        .max()
        .unwrap_or(0);

    ctx.last_stage_rounds = num_rounds;
    ctx.last_stage_degree = degree;

    // Detect uni-skip first round
    let uniskip_domain = plan
        .vertices
        .iter()
        .find_map(|&vi| protocol.vertices[vi].domain_size());

    // The verifier stage index for this stage (assigned by the caller
    // before calling emit_sumcheck_stage, via staging_to_verifier).
    let verifier_stage = VerifierStageIndex(ctx.verifier_stage_count);

    // Allocate all round challenges upfront so InputBindings can reference them
    let round_challenge_indices: Vec<ChallengeIdx> = (0..num_rounds)
        .map(|r| {
            let ch_idx = ChallengeIdx(ctx.challenges.len());
            ctx.challenges.push(ChallengeDecl {
                name: format!("r_s{}_r{r}", verifier_stage.0),
                source: ChallengeSource::SumcheckRound {
                    stage: verifier_stage,
                    round: r,
                },
            });
            ch_idx
        })
        .collect();

    // Record vertex → challenge mapping so later stages can resolve
    // PublicPoly::Eq(Some(claim_id)) to concrete challenge indices.
    for &vi in &plan.vertices {
        let _ = ctx
            .vertex_challenges
            .insert(vi, round_challenge_indices.clone());
    }

    // Build KernelDef for this stage
    let (formula, input_mapping) = build_formula(protocol, plan);
    let kernel_idx = ctx.kernels.len();

    // Convert poly indices → InputBinding with explicit data provenance.
    // Table-type bindings get their challenge indices resolved here.
    let input_bindings: Vec<InputBinding> = input_mapping
        .iter()
        .map(|&pi| {
            poly_to_input_binding(
                ctx.map_poly(pi),
                &protocol.polynomials[pi].kind,
                ctx,
                &round_challenge_indices,
            )
        })
        .collect();

    let is_sparse = plan
        .vertices
        .iter()
        .any(|&vi| match &protocol.vertices[vi] {
            Vertex::Sumcheck { density, .. } => *density == Density::Sparse,
            Vertex::Evaluate { .. } => false,
        });
    let iteration = if is_sparse {
        Iteration::Sparse
    } else {
        Iteration::Dense
    };

    ctx.kernels.push(KernelDef {
        spec: KernelSpec {
            formula,
            num_evals: degree + 1,
            iteration,
            binding_order: BindingOrder::LowToHigh,
        },
        inputs: input_bindings,
        num_rounds,
        instance_config: None,
    });

    // Materialize all kernel inputs before the first round.
    for binding in &ctx.kernels[kernel_idx].inputs {
        ctx.ops.push(Op::Materialize {
            binding: binding.clone(),
        });
    }

    // Emit round ops: round 0 has no bind, rounds 1+ fuse bind with reduce
    for r in 0..num_rounds {
        let num_coeffs = if r == 0 {
            uniskip_domain.unwrap_or(degree + 1)
        } else {
            degree + 1
        };

        let bind_challenge = if r > 0 {
            Some(round_challenge_indices[r - 1])
        } else {
            None
        };

        ctx.ops.push(Op::SumcheckRound {
            kernel: kernel_idx,
            round: r,
            bind_challenge,
        });
        let round_tag = if r == 0 && uniskip_domain.is_some() {
            DomainSeparator::UniskipPoly
        } else {
            DomainSeparator::SumcheckPoly
        };
        // Uniskip round 0 currently always uses Compressed encoding.
        ctx.ops.push(Op::AbsorbRoundPoly {
            num_coeffs,
            tag: round_tag,
            encoding: crate::module::RoundPolyEncoding::Compressed,
        });
        ctx.ops.push(Op::Squeeze {
            challenge: round_challenge_indices[r],
        });
    }

    // After all rounds: final bind at the last challenge, but only for polys
    // that survive beyond this stage (used in evaluations or later compositions).
    if !round_challenge_indices.is_empty() {
        let last_ch = *round_challenge_indices.last().unwrap();
        let future_polys = collect_future_polys(protocol, all_stages, stage_idx);
        let surviving: Vec<PolynomialId> = all_polys
            .into_iter()
            .filter(|p| future_polys.contains(p))
            .map(|p| ctx.map_poly(p))
            .collect();
        if !surviving.is_empty() {
            ctx.ops.push(Op::Bind {
                polys: surviving,
                challenge: last_ch,
                order: BindingOrder::LowToHigh,
            });
        }
    }
}

// Formula conversion: protocol Expr → Formula

/// Convert a stage's composition expressions to a single Formula.
/// Returns the formula and the poly index mapping (input_i → protocol poly index).
fn build_formula(protocol: &crate::ir::Protocol, plan: &StagePlan) -> (Formula, Vec<usize>) {
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

    (Formula::from_terms(all_terms), all_poly_indices)
}

// Verifier claim formula construction

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
                            // The eval index is the claim's poly index mapped
                            // through poly_map.
                            let proto_poly = protocol.claims[cid.0 as usize].poly;
                            factors.push(ClaimFactor::Eval(ctx.map_poly(proto_poly)));
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

/// Build the output check formula for a sumcheck stage.
///
/// After sumcheck verification the verifier has `(final_eval, challenges)`. This
/// formula, evaluated with the prover-provided polynomial evaluations and challenge
/// values, should equal `final_eval`. It encodes the composition polynomial
/// evaluated at the evaluation point.
fn build_output_check_formula(
    protocol: &crate::ir::Protocol,
    plan: &StagePlan,
    ctx: &EmitCtx<'_>,
) -> ClaimFormula {
    let mut terms = Vec::new();

    for &vi in &plan.vertices {
        if let Some(comp) = protocol.vertices[vi].composition() {
            for term in &comp.0 {
                let mut factors = Vec::new();
                for f in &term.factors {
                    match f {
                        ExprFactor::Poly(idx) => {
                            factors.push(ClaimFactor::Eval(ctx.map_poly(*idx)));
                        }
                        ExprFactor::Challenge(idx) => {
                            if let Some(&mapped) = ctx.proto_challenge_map.get(idx) {
                                factors.push(ClaimFactor::Challenge(mapped));
                            }
                        }
                        ExprFactor::Claim(_) => {
                            // Claims don't appear in output compositions
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

// Helpers

/// Polys referenced by evaluations in the current stage or by any later stage
/// (compositions + evaluations). A poly from the current stage's composition
/// only needs FinalBind if it appears in this set.
fn collect_future_polys(
    protocol: &crate::ir::Protocol,
    all_stages: &[StagePlan],
    current_stage: usize,
) -> HashSet<usize> {
    let mut polys = HashSet::new();

    // Current stage evaluations — these polys need to be fully bound
    for &vi in &all_stages[current_stage].evaluations {
        if let Vertex::Evaluate { poly, .. } = &protocol.vertices[vi] {
            let _ = polys.insert(*poly);
        }
    }

    // All later stages: compositions + evaluations
    for stage in &all_stages[current_stage + 1..] {
        for &vi in &stage.vertices {
            if let Some(comp) = protocol.vertices[vi].composition() {
                for term in &comp.0 {
                    for f in &term.factors {
                        if let ExprFactor::Poly(idx) = f {
                            let _ = polys.insert(*idx);
                        }
                    }
                }
            }
        }
        for &vi in &stage.evaluations {
            if let Vertex::Evaluate { poly, .. } = &protocol.vertices[vi] {
                let _ = polys.insert(*poly);
            }
        }
    }

    polys
}

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

/// Map a polynomial's kind to the appropriate [`InputBinding`] variant.
///
/// - Committed / Virtual / Preprocessed / Identity → `Provided` (loaded from BufferProvider)
/// - Eq / EqPlusOne / Lt → table built on-device from challenge points
///
/// For table variants with `Some(claim_id)`, the claim is resolved to
/// concrete challenge indices via the vertex→challenge map built during
/// earlier stage emissions.
///
/// `Eq(None)` means the eq table is anchored at the *current* sumcheck's
/// point — those are the round challenges being allocated right now.
/// The round challenge indices are already in `vertex_challenges` for the
/// current stage's vertices.
fn poly_to_input_binding(
    poly: PolynomialId,
    kind: &PolyKind,
    ctx: &EmitCtx<'_>,
    current_round_challenges: &[ChallengeIdx],
) -> InputBinding {
    match kind {
        PolyKind::Committed | PolyKind::Virtual => InputBinding::Provided { poly },
        PolyKind::Public(pp) => match pp {
            PublicPoly::Preprocessed | PublicPoly::Identity => InputBinding::Provided { poly },
            PublicPoly::Eq(claim_opt) => {
                let challenges =
                    resolve_table_challenges(claim_opt.as_ref(), ctx, current_round_challenges);
                InputBinding::EqTable { poly, challenges }
            }
            PublicPoly::EqPlusOne(claim_opt) => {
                let challenges =
                    resolve_table_challenges(claim_opt.as_ref(), ctx, current_round_challenges);
                InputBinding::EqPlusOneTable { poly, challenges }
            }
            PublicPoly::Lt(claim_opt) => {
                let challenges =
                    resolve_table_challenges(claim_opt.as_ref(), ctx, current_round_challenges);
                InputBinding::LtTable { poly, challenges }
            }
        },
    }
}

/// Resolve an optional `ClaimId` anchor to concrete challenge indices.
///
/// - `Some(claim_id)` → look up which vertex produced the claim, return
///   that vertex's round challenge indices.
/// - `None` → the table is anchored at the current sumcheck's own point.
///   Uses `current_round_challenges` (the round challenges allocated for
///   the stage being emitted).
fn resolve_table_challenges(
    claim: Option<&crate::ir::ClaimId>,
    ctx: &EmitCtx<'_>,
    current_round_challenges: &[ChallengeIdx],
) -> Vec<ChallengeIdx> {
    match claim {
        Some(cid) => ctx
            .claim_challenges(*cid)
            .map(|chs| chs.to_vec())
            .unwrap_or_default(),
        None => current_round_challenges.to_vec(),
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
        Vertex::Sumcheck { composition, .. } => composition
            .0
            .iter()
            .map(|term| {
                term.factors
                    .iter()
                    .filter(|f| matches!(f, ExprFactor::Poly(_)))
                    .count()
            })
            .max()
            .unwrap_or(0),
        Vertex::Evaluate { .. } => 0,
    }
}

// Release insertion via liveness analysis

/// Collect all poly identifiers referenced by an op (directly or via kernel inputs).
fn op_poly_refs(op: &Op, kernels: &[KernelDef]) -> Vec<PolynomialId> {
    match op {
        Op::SumcheckRound { kernel, .. } => {
            kernels[*kernel].inputs.iter().map(|b| b.poly()).collect()
        }
        Op::AbsorbRoundPoly { .. } => vec![],
        Op::Evaluate { poly, .. }
        | Op::CollectOpeningClaim { poly, .. }
        | Op::ScaleEval { poly, .. }
        | Op::CollectOpeningClaimAt { poly, .. } => vec![*poly],
        Op::Bind { polys, .. }
        | Op::LagrangeProject { polys, .. }
        | Op::DuplicateInterleave { polys }
        | Op::RegroupConstraints { polys, .. }
        | Op::Commit { polys, .. }
        | Op::CommitStreaming { polys, .. }
        | Op::RecordEvals { polys, .. }
        | Op::AbsorbEvals { polys, .. } => polys.clone(),
        Op::CaptureScalar { poly, .. } => vec![*poly],
        Op::Preamble
        | Op::BeginStage { .. }
        | Op::AbsorbInputClaim { .. }
        | Op::Squeeze { .. }
        | Op::ComputePower { .. }
        | Op::AppendDomainSeparator { .. }
        | Op::EvaluatePreprocessed { .. }
        | Op::ReleaseDevice { .. }
        | Op::ReleaseHost { .. }
        | Op::AliasEval { .. }
        | Op::ReduceOpenings
        | Op::Open
        | Op::BatchRoundBegin { .. }
        | Op::BatchInactiveContribution { .. }
        | Op::Materialize { .. }
        | Op::MaterializeUnlessFresh { .. }
        | Op::MaterializeIfAbsent { .. }
        | Op::MaterializeSegmentedOuterEq { .. }
        | Op::InstanceBindPreviousPhase { .. }
        | Op::InstanceReduce { .. }
        | Op::InstanceSegmentedReduce { .. }
        | Op::InstanceBind { .. }
        | Op::BindCarryBuffers { .. }
        | Op::BatchAccumulateInstance { .. }
        | Op::BatchRoundFinalize { .. }
        | Op::BindOpeningInputs { .. }
        | Op::ExpandingTableUpdate { .. }
        | Op::CheckpointEvalBatch { .. }
        | Op::InitInstanceWeights { .. }
        | Op::UpdateInstanceWeights { .. }
        | Op::SuffixScatter { .. }
        | Op::QBufferScatter { .. }
        | Op::MaterializePBuffers { .. }
        | Op::InitExpandingTable { .. }
        | Op::ReadCheckingReduce { .. }
        | Op::RafReduce { .. }
        | Op::MaterializeRA { .. }
        | Op::MaterializeCombinedVal { .. }
        | Op::WeightedSum { .. } => vec![],
    }
}

/// Insert `Op::ReleaseDevice` after the last use of each poly buffer.
///
/// Backward scan: the first time we see a poly (scanning from the end) is
/// its last use. We insert a Release immediately after that position.
/// Insertions are batched to avoid index shifting issues.
fn insert_releases(ops: &mut Vec<Op>, kernels: &[KernelDef]) {
    let mut seen = HashSet::new();
    // (insert_after_index, poly_id) — sorted by position descending for safe insertion
    let mut releases: Vec<(usize, PolynomialId)> = Vec::new();

    for i in (0..ops.len()).rev() {
        for poly in op_poly_refs(&ops[i], kernels) {
            if seen.insert(poly) {
                releases.push((i, poly));
            }
        }
    }

    // Sort by position descending so insertions don't shift earlier indices
    releases.sort_by(|a, b| b.0.cmp(&a.0));
    for (after_idx, poly) in releases {
        ops.insert(after_idx + 1, Op::ReleaseDevice { poly });
    }
}
