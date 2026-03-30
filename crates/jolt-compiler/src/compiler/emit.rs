//! Emit pass: transform [`Staging`] into [`Module`].
//!
//! For each stage, simultaneously produces:
//! - [`KernelDef`] + [`Op`] sequence (prover schedule)
//! - [`VerifierStage`] (verifier schedule)

use std::collections::HashSet;

use crate::formula::{BindingOrder, Factor as FormulaFactor, Formula, ProductTerm};
use crate::ir::expr::Factor as ExprFactor;
use crate::ir::{PolyKind, PublicPoly, Vertex};
use crate::module::{
    ChallengeDecl, ChallengeSource, ClaimFactor, ClaimFormula, ClaimTerm, Evaluation, InputBinding,
    KernelDef, Module, Op, PolyDecl, Schedule, VerifierSchedule, VerifierStage,
};

use super::cost::CompileParams;
use super::stage::{StagePlan, Staging};

/// Emit the final compiled module from a staged protocol.
pub(crate) fn emit(staging: &Staging, params: &CompileParams) -> Module {
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
        ctx.ops.push(Op::EmitCommitments {
            polys: committed.clone(),
        });
    }

    // Per-stage Fiat-Shamir challenges allocated after each stage
    let stage_challenges: Vec<usize> = (0..staging.stages.len())
        .map(|si| {
            let ch_idx = ctx.challenges.len();
            ctx.challenges.push(ChallengeDecl {
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
            ctx.challenges.push(ChallengeDecl {
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
            emit_sumcheck_stage(&mut ctx, si, plan, &staging.stages);
        }

        // Evaluate vertices
        let mut eval_ops = Vec::new();
        for &vi in &plan.evaluations {
            if let Vertex::Evaluate {
                poly, at_vertex, ..
            } = &protocol.vertices[vi]
            {
                ctx.ops.push(Op::Evaluate { poly: *poly });
                eval_ops.push(Evaluation {
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
            ctx.ops.push(Op::EmitScalars {
                evals: eval_indices,
            });
        }

        // Post-stage challenges
        let mut post_squeeze = Vec::new();
        if si < stage_challenges.len() {
            ctx.ops.push(Op::Squeeze {
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
                evaluations: eval_ops,
                post_squeeze,
            });
        } else if !plan.evaluations.is_empty() {
            ctx.verifier_stages.push(VerifierStage {
                commitments: verifier_commitments,
                input_claim: ClaimFormula::zero(),
                num_rounds: 0,
                degree: 0,
                evaluations: eval_ops,
                post_squeeze,
            });
        }
    }

    // Emit opening stage if present
    if let Some(opening) = &staging.opening {
        let ch_idx = ctx.challenges.len();
        ctx.challenges.push(ChallengeDecl {
            name: opening.challenge_name.clone(),
            source: ChallengeSource::FiatShamir {
                after_stage: staging.stages.len(),
            },
        });
        ctx.ops.push(Op::Squeeze { challenge: ch_idx });
        ctx.verifier_stages.push(VerifierStage {
            commitments: vec![],
            input_claim: ClaimFormula::zero(),
            num_rounds: 0,
            degree: 0,
            evaluations: vec![],
            post_squeeze: vec![ch_idx],
        });
    }

    // Insert Release ops via liveness analysis
    insert_releases(&mut ctx.ops, &ctx.kernels);

    // Build poly decls
    let polys: Vec<PolyDecl> = protocol
        .polynomials
        .iter()
        .map(|p| {
            let num_elements = poly_num_elements_from_dims(&p.dims, params);
            PolyDecl {
                name: p.name.clone(),
                kind: p.kind.clone(),
                num_elements,
            }
        })
        .collect();

    Module {
        polys,
        challenges: ctx.challenges,
        prover: Schedule {
            ops: ctx.ops,
            kernels: ctx.kernels,
        },
        verifier: VerifierSchedule {
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
    ops: Vec<Op>,
    kernels: Vec<KernelDef>,
    challenges: Vec<ChallengeDecl>,
    verifier_stages: Vec<VerifierStage>,
    /// Protocol challenge index → output challenge index
    proto_challenge_map: std::collections::HashMap<usize, usize>,
    /// Vertex index → round challenge indices for that vertex's sumcheck.
    /// Used to resolve `PublicPoly::Eq(Some(claim_id))` → challenge indices.
    vertex_challenges: std::collections::HashMap<usize, Vec<usize>>,
    /// Bookkeeping for verifier stage construction
    last_stage_rounds: usize,
    last_stage_degree: usize,
}

impl<'a> EmitCtx<'a> {
    fn new(protocol: &'a crate::ir::Protocol, params: &'a CompileParams) -> Self {
        Self {
            protocol,
            params,
            ops: Vec::new(),
            kernels: Vec::new(),
            challenges: Vec::new(),
            verifier_stages: Vec::new(),
            proto_challenge_map: std::collections::HashMap::new(),
            vertex_challenges: std::collections::HashMap::new(),
            last_stage_rounds: 0,
            last_stage_degree: 0,
        }
    }

    /// Resolve a `ClaimId` to the round challenge indices of the sumcheck
    /// that produced it. Returns `None` for claims whose vertex hasn't been
    /// emitted yet (shouldn't happen with correct stage ordering).
    fn claim_challenges(&self, claim_id: crate::ir::ClaimId) -> Option<&[usize]> {
        let claim = &self.protocol.claims[claim_id.0 as usize];
        self.vertex_challenges
            .get(&claim.produced_by)
            .map(|v| v.as_slice())
    }
}

// ---------------------------------------------------------------------------
// Sumcheck stage emission
// ---------------------------------------------------------------------------

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

    // Allocate all round challenges upfront so InputBindings can reference them
    let round_challenge_indices: Vec<usize> = (0..num_rounds)
        .map(|r| {
            let ch_idx = ctx.challenges.len();
            ctx.challenges.push(ChallengeDecl {
                name: format!("r_s{stage_idx}_r{r}"),
                source: ChallengeSource::SumcheckRound {
                    stage: stage_idx,
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
            poly_to_input_binding(pi, &protocol.polynomials[pi].kind, ctx, &round_challenge_indices)
        })
        .collect();

    ctx.kernels.push(KernelDef {
        formula,
        inputs: input_bindings,
        binding_order: BindingOrder::LowToHigh,
        num_rounds,
        degree,
    });

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
        ctx.ops.push(Op::EmitRoundPoly {
            kernel: kernel_idx,
            num_coeffs,
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
        let surviving: Vec<usize> = all_polys
            .into_iter()
            .filter(|p| future_polys.contains(p))
            .collect();
        if !surviving.is_empty() {
            ctx.ops.push(Op::FinalBind {
                polys: surviving,
                challenge: last_ch,
                order: BindingOrder::LowToHigh,
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Formula conversion: protocol Expr → Formula
// ---------------------------------------------------------------------------

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
    poly: usize,
    kind: &PolyKind,
    ctx: &EmitCtx<'_>,
    current_round_challenges: &[usize],
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
    current_round_challenges: &[usize],
) -> Vec<usize> {
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

// ---------------------------------------------------------------------------
// Release insertion via liveness analysis
// ---------------------------------------------------------------------------

/// Collect all poly indices referenced by an op (directly or via kernel inputs).
fn op_poly_refs(op: &Op, kernels: &[KernelDef]) -> Vec<usize> {
    match op {
        Op::SumcheckRound { kernel, .. } | Op::EmitRoundPoly { kernel, .. } => kernels[*kernel]
            .inputs
            .iter()
            .map(|b| b.poly())
            .collect(),
        Op::Evaluate { poly } => vec![*poly],
        Op::FinalBind { polys, .. } => polys.clone(),
        Op::EmitCommitments { polys } => polys.clone(),
        Op::EmitScalars { evals } => evals.clone(),
        Op::Squeeze { .. } | Op::Release { .. } => vec![],
    }
}

/// Insert `Op::Release` after the last use of each poly buffer.
///
/// Backward scan: the first time we see a poly (scanning from the end) is
/// its last use. We insert a Release immediately after that position.
/// Insertions are batched to avoid index shifting issues.
fn insert_releases(ops: &mut Vec<Op>, kernels: &[KernelDef]) {
    let mut seen = HashSet::new();
    // (insert_after_index, poly_index) — sorted by position descending for safe insertion
    let mut releases: Vec<(usize, usize)> = Vec::new();

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
        ops.insert(after_idx + 1, Op::Release { poly });
    }
}
