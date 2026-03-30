//! Staging pass: assign protocol vertices to batched sumcheck stages.
//!
//! 1. ASAP assignment: `stage(v) = depth(v)`
//! 2. Local search: delay vertices to reduce proof size
//! 3. Compaction: renumber stages to remove gaps
//!
//! Output is a [`Staging`] — an internal representation consumed by the
//! emit pass to produce [`Schedule`] and [`VerifierSchedule`].

use std::collections::{BTreeMap, HashSet};

use super::analyze::IRInfo;
use super::cost::{self, CompileParams, Cost, Objective, SolverConfig};
use super::CompileError;
use crate::ir::expr::{Challenge, Expr, Factor, Poly};
use crate::ir::{Claim, ClaimId, PolyKind, Protocol, PublicPoly, Vertex};

/// Result of the staging pass.
pub(crate) struct Staging {
    pub stages: Vec<StagePlan>,
    /// Extended protocol with any compiler-synthesized vertices (e.g. reduction sumchecks).
    pub protocol: Protocol,
    /// Opening RLC data for the final PCS batching stage, if applicable.
    pub opening: Option<OpeningData>,
}

/// Data for the opening RLC + PCS verification stage.
#[allow(dead_code)]
pub(crate) struct OpeningData {
    pub claims: Vec<ClaimId>,
    pub challenge_name: String,
}

/// Plan for a single stage.
pub(crate) struct StagePlan {
    /// Sumcheck vertex indices.
    pub vertices: Vec<usize>,
    /// Evaluate vertex indices resolved in this stage.
    pub evaluations: Vec<usize>,
}

/// Stage a validated protocol.
pub(crate) fn stage(
    protocol: &Protocol,
    info: &IRInfo,
    params: &CompileParams,
    config: &SolverConfig,
) -> Result<Staging, CompileError> {
    if protocol.vertices.is_empty() {
        return Ok(Staging {
            stages: vec![],
            protocol: protocol.clone(),
            opening: None,
        });
    }

    let terminal = find_terminal_claims(protocol);

    // Phase 1: ASAP baseline
    let mut assignment: Vec<usize> = info.depth.clone();
    let mut num_stages = assignment.iter().copied().max().unwrap() + 1;
    let mut costs = form_stage_costs(&assignment, num_stages, protocol, info, params, &terminal);
    let mut best_cost = evaluate_cost(&costs, params);

    // Phase 2: local search
    let ctx = SearchCtx {
        protocol,
        info,
        params,
        config,
        terminal: &terminal,
    };
    local_search(
        &mut assignment,
        &mut num_stages,
        &mut costs,
        &mut best_cost,
        &ctx,
    );

    // Phase 3: compact empty stages
    num_stages = compact(&mut assignment);
    costs = form_stage_costs(&assignment, num_stages, protocol, info, params, &terminal);
    best_cost = evaluate_cost(&costs, params);

    // Feasibility check
    if !cost::satisfies(&best_cost, config) {
        let violated = violated_list(config, &best_cost);
        return Err(CompileError::Infeasible {
            cost: best_cost,
            violated,
        });
    }

    Ok(assemble(&costs, &assignment, protocol, &terminal))
}

// ---------------------------------------------------------------------------
// Terminal claims
// ---------------------------------------------------------------------------

fn find_terminal_claims(protocol: &Protocol) -> HashSet<ClaimId> {
    let consumed: HashSet<ClaimId> = protocol
        .vertices
        .iter()
        .flat_map(|v| v.consumes().iter().copied())
        .collect();
    protocol
        .claims
        .iter()
        .map(|c| c.id)
        .filter(|id| !consumed.contains(id))
        .collect()
}

// ---------------------------------------------------------------------------
// Cost model
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct StageCostInfo {
    vertices: Vec<usize>,
    dims: Vec<usize>,
    degree: usize,
    num_polys: usize,
    num_variables: u64,
    has_terminal: bool,
    n_terminal_points: usize,
}

fn form_stage_costs(
    assignment: &[usize],
    num_stages: usize,
    protocol: &Protocol,
    info: &IRInfo,
    params: &CompileParams,
    terminal: &HashSet<ClaimId>,
) -> Vec<StageCostInfo> {
    (0..num_stages)
        .map(|s| {
            let verts: Vec<usize> = (0..assignment.len())
                .filter(|&v| assignment[v] == s && protocol.vertices[v].binding_order().is_some())
                .collect();
            make_stage_cost(verts, protocol, info, params, terminal)
        })
        .collect()
}

fn make_stage_cost(
    vertices: Vec<usize>,
    protocol: &Protocol,
    info: &IRInfo,
    params: &CompileParams,
    terminal: &HashSet<ClaimId>,
) -> StageCostInfo {
    let degree = vertices.iter().map(|&v| info.degree[v]).max().unwrap_or(0);

    let mut polys = HashSet::new();
    for &v in &vertices {
        if let Some(composition) = protocol.vertices[v].composition() {
            for term in &composition.0 {
                for f in &term.factors {
                    if let Factor::Poly(idx) = f {
                        let _ = polys.insert(*idx);
                    }
                }
            }
        }
    }

    let has_terminal = vertices.iter().any(|&v| {
        protocol.vertices[v]
            .produces()
            .iter()
            .any(|c| terminal.contains(c))
    });

    let terminal_bos: HashSet<Vec<usize>> = vertices
        .iter()
        .filter(|&&v| {
            protocol.vertices[v]
                .produces()
                .iter()
                .any(|c| terminal.contains(c))
        })
        .filter_map(|&v| protocol.vertices[v].binding_order().map(|bo| bo.to_vec()))
        .collect();

    let mut dims: Vec<usize> = vertices
        .iter()
        .filter_map(|&v| protocol.vertices[v].binding_order())
        .flat_map(|bo| bo.iter().copied())
        .collect();
    dims.sort_unstable();
    dims.dedup();
    let num_variables: u64 = dims.iter().map(|&d| params.dim_sizes[d]).sum();

    StageCostInfo {
        vertices,
        dims,
        degree,
        num_polys: polys.len(),
        num_variables,
        has_terminal,
        n_terminal_points: terminal_bos.len(),
    }
}

fn pow2_clamped(n: u64) -> u64 {
    if n >= 64 {
        u64::MAX
    } else {
        1u64 << n
    }
}

fn evaluate_cost(stages: &[StageCostInfo], params: &CompileParams) -> Cost {
    let sumcheck_fe: u64 = stages
        .iter()
        .map(|s| (s.degree as u64 + 1) * s.num_variables)
        .sum();

    let n_points: usize = stages.iter().map(|s| s.n_terminal_points).sum();

    let pcs_fe = if n_points > 0 {
        params.pcs_proof_size
    } else {
        0
    };

    let reduction_fe = if n_points > 1 {
        let mut dims: Vec<usize> = stages
            .iter()
            .filter(|s| s.has_terminal)
            .flat_map(|s| s.dims.iter().copied())
            .collect();
        dims.sort_unstable();
        dims.dedup();
        let max_vars: u64 = dims.iter().map(|&d| params.dim_sizes[d]).sum();
        3 * max_vars
    } else {
        0
    };

    let peak_mem = stages
        .iter()
        .map(|s| {
            (s.num_polys as u64)
                .saturating_mul(pow2_clamped(s.num_variables))
                .saturating_mul(params.field_size_bytes)
        })
        .max()
        .unwrap_or(0);

    let work: u64 = stages
        .iter()
        .map(|s| {
            (s.num_polys as u64)
                .saturating_mul(s.degree as u64 + 1)
                .saturating_mul(pow2_clamped(s.num_variables))
        })
        .fold(0u64, u64::saturating_add);

    Cost {
        proof_size: sumcheck_fe + reduction_fe + pcs_fe,
        eval_points: n_points,
        peak_memory: peak_mem,
        prover_time: work,
    }
}

// ---------------------------------------------------------------------------
// Local search
// ---------------------------------------------------------------------------

struct SearchCtx<'a> {
    protocol: &'a Protocol,
    info: &'a IRInfo,
    params: &'a CompileParams,
    config: &'a SolverConfig,
    terminal: &'a HashSet<ClaimId>,
}

fn local_search(
    assignment: &mut [usize],
    num_stages: &mut usize,
    stages: &mut Vec<StageCostInfo>,
    best: &mut Cost,
    ctx: &SearchCtx<'_>,
) {
    loop {
        let mut improved = false;

        for &v in ctx.info.topo_order.iter().rev() {
            if ctx.protocol.vertices[v].binding_order().is_none() {
                continue;
            }

            let cur = assignment[v];
            let ceiling = ctx.info.successors[v]
                .iter()
                .map(|&s| assignment[s])
                .min()
                .unwrap_or(*num_stages)
                .saturating_sub(1);

            if ceiling <= cur {
                continue;
            }

            for target in (cur + 1)..=ceiling {
                let saved = assignment[v];
                assignment[v] = target;

                let ns = assignment.iter().copied().max().unwrap() + 1;
                let new_stages = form_stage_costs(
                    assignment,
                    ns,
                    ctx.protocol,
                    ctx.info,
                    ctx.params,
                    ctx.terminal,
                );
                let new_cost = evaluate_cost(&new_stages, ctx.params);

                if cost::satisfies(&new_cost, ctx.config)
                    && cost::is_better(&new_cost, best, ctx.config)
                {
                    *num_stages = ns;
                    *stages = new_stages;
                    *best = new_cost;
                    improved = true;
                    break;
                }

                assignment[v] = saved;
            }
        }

        if !improved {
            break;
        }
    }
}

fn compact(assignment: &mut [usize]) -> usize {
    if assignment.is_empty() {
        return 0;
    }
    let mut used: Vec<usize> = assignment.to_vec();
    used.sort_unstable();
    used.dedup();
    for a in assignment.iter_mut() {
        *a = used.iter().position(|&u| u == *a).unwrap();
    }
    used.len()
}

fn violated_list(config: &SolverConfig, cost: &Cost) -> Vec<String> {
    let mut out = Vec::new();
    if let Objective::Bounded(limit) = config.proof_size {
        if cost.proof_size > limit {
            out.push(format!("proof_size {} > bound {limit}", cost.proof_size));
        }
    }
    if let Objective::Bounded(limit) = config.peak_memory {
        if cost.peak_memory > limit {
            out.push(format!("peak_memory {} > bound {limit}", cost.peak_memory));
        }
    }
    if let Objective::Bounded(limit) = config.prover_time {
        if cost.prover_time > limit {
            out.push(format!("prover_time {} > bound {limit}", cost.prover_time));
        }
    }
    out
}

// ---------------------------------------------------------------------------
// Assembly: build Staging from the final assignment
// ---------------------------------------------------------------------------

fn assemble(
    costs: &[StageCostInfo],
    assignment: &[usize],
    protocol: &Protocol,
    terminal: &HashSet<ClaimId>,
) -> Staging {
    let mut extended = protocol.clone();

    let mut stage_plans: Vec<StagePlan> = costs
        .iter()
        .enumerate()
        .map(|(s, sc)| {
            let evaluations: Vec<usize> = (0..protocol.vertices.len())
                .filter(|&v| assignment[v] == s && protocol.vertices[v].binding_order().is_none())
                .collect();
            StagePlan {
                vertices: sc.vertices.clone(),
                evaluations,
            }
        })
        .collect();

    let terminal_claims: Vec<ClaimId> = protocol
        .claims
        .iter()
        .filter(|c| terminal.contains(&c.id))
        .map(|c| c.id)
        .collect();

    if terminal_claims.is_empty() {
        return Staging {
            stages: stage_plans,
            protocol: extended,
            opening: None,
        };
    }

    let n_points: usize = costs.iter().map(|s| s.n_terminal_points).sum();
    let last_user_stage = stage_plans.len().saturating_sub(1);

    let opening_claims = if n_points > 1 {
        synthesize_reduction(
            &mut extended,
            &mut stage_plans,
            &terminal_claims,
            costs,
            last_user_stage,
        )
    } else {
        terminal_claims
    };

    Staging {
        stages: stage_plans,
        protocol: extended,
        opening: Some(OpeningData {
            claims: opening_claims,
            challenge_name: "rho_open".into(),
        }),
    }
}

fn synthesize_reduction(
    protocol: &mut Protocol,
    stages: &mut Vec<StagePlan>,
    terminal_claims: &[ClaimId],
    costs: &[StageCostInfo],
    _last_user_stage: usize,
) -> Vec<ClaimId> {
    let rho_proto_idx = protocol.challenge_names.len();
    protocol.challenge_names.push("rho_reduce".into());
    let rho = Challenge(rho_proto_idx);

    let mut binding_order: Vec<usize> = costs
        .iter()
        .filter(|s| s.has_terminal)
        .flat_map(|s| s.dims.iter().copied())
        .collect();
    binding_order.sort_unstable();
    binding_order.dedup();

    let mut groups: BTreeMap<usize, Vec<ClaimId>> = BTreeMap::new();
    for &cid in terminal_claims {
        let producing_vertex = protocol.claims[cid.0 as usize].produced_by;
        groups.entry(producing_vertex).or_default().push(cid);
    }

    let mut eq_polys: BTreeMap<usize, Poly> = BTreeMap::new();
    for (&producing_vertex, group_claims) in &groups {
        let anchor_claim = group_claims[0];
        let eq_poly = protocol.poly(
            &format!("_eq_reduce_v{producing_vertex}"),
            &binding_order,
            PolyKind::Public(PublicPoly::Eq(Some(anchor_claim))),
        );
        let _ = eq_polys.insert(producing_vertex, eq_poly);
    }

    let mut composition = Expr::from(0i64);
    let mut input_sum = Expr::from(0i64);
    let vertex_idx = protocol.vertices.len();

    let mut produced_claims = Vec::with_capacity(terminal_claims.len());
    let mut consumes = Vec::with_capacity(terminal_claims.len());

    for (i, &cid) in terminal_claims.iter().enumerate() {
        let producing_vertex = protocol.claims[cid.0 as usize].produced_by;
        let eq_poly = eq_polys[&producing_vertex];
        let poly_idx = protocol.claims[cid.0 as usize].poly;

        let rho_power = (0..i).fold(Expr::from(1i64), |acc, _| acc * rho);
        let term = rho_power.clone() * eq_poly * Poly(poly_idx);
        composition = composition + term;

        let claim_term = rho_power * cid;
        input_sum = input_sum + claim_term;

        consumes.push(cid);

        let new_claim_id = ClaimId(protocol.next_claim);
        protocol.next_claim += 1;
        protocol.claims.push(Claim {
            id: new_claim_id,
            poly: poly_idx,
            produced_by: vertex_idx,
        });
        produced_claims.push(new_claim_id);
    }

    protocol.vertices.push(Vertex::Sumcheck {
        composition,
        input_sum,
        produces: produced_claims.clone(),
        consumes,
        binding_order: binding_order.clone(),
        domain_size: None,
    });

    stages.push(StagePlan {
        vertices: vec![vertex_idx],
        evaluations: vec![],
    });

    produced_claims
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use crate::compiler::analyze;
    use crate::ir::{PolyKind, Protocol, PublicPoly};

    fn params() -> CompileParams {
        CompileParams {
            dim_sizes: vec![4, 3],
            field_size_bytes: 32,
            pcs_proof_size: 100,
        }
    }

    fn minimize_proof() -> SolverConfig {
        SolverConfig {
            proof_size: Objective::Minimize,
            peak_memory: Objective::Ignore,
            prover_time: Objective::Ignore,
        }
    }

    fn linear_chain() -> Protocol {
        let mut p = Protocol::new();
        let d = p.dim("d");
        let rho = p.challenge("rho");
        let eq = p.poly("eq", &[d], PolyKind::Public(PublicPoly::Eq(None)));
        let a = p.poly("a", &[d], PolyKind::Virtual);
        let b = p.poly("b", &[d], PolyKind::Virtual);
        let c = p.poly("c", &[d], PolyKind::Virtual);
        let c0 = p.sumcheck(eq * a, 0, &[d]);
        let c1 = p.sumcheck(eq * b, rho * c0[0], &[d]);
        let _ = p.sumcheck(eq * c, rho * c1[0], &[d]);
        p
    }

    fn diamond() -> Protocol {
        let mut p = Protocol::new();
        let d = p.dim("d");
        let rho = p.challenge("rho");
        let eq = p.poly("eq", &[d], PolyKind::Public(PublicPoly::Eq(None)));
        let a = p.poly("a", &[d], PolyKind::Virtual);
        let b = p.poly("b", &[d], PolyKind::Virtual);
        let c = p.poly("c", &[d], PolyKind::Virtual);
        let e = p.poly("e", &[d], PolyKind::Virtual);
        let c0 = p.sumcheck(eq * a, 0, &[d]);
        let c1 = p.sumcheck(eq * b, rho * c0[0], &[d]);
        let c2 = p.sumcheck(eq * c, rho * c0[0], &[d]);
        let _ = p.sumcheck(eq * e, rho * c1[0] + c2[0], &[d]);
        p
    }

    fn multi_dim() -> Protocol {
        let mut p = Protocol::new();
        let log_T = p.dim("log_T");
        let log_K = p.dim("log_K");
        let eq_T = p.poly("eq_T", &[log_T], PolyKind::Public(PublicPoly::Eq(None)));
        let eq_K = p.poly("eq_K", &[log_K], PolyKind::Public(PublicPoly::Eq(None)));
        let a = p.poly("a", &[log_T], PolyKind::Virtual);
        let b = p.poly("b", &[log_K], PolyKind::Virtual);
        let c = p.poly("c", &[log_T, log_K], PolyKind::Virtual);
        let _ = p.sumcheck(eq_T * a, 0, &[log_T]);
        let _ = p.sumcheck(eq_K * b, 0, &[log_K]);
        let _ = p.sumcheck(eq_T * c, 0, &[log_T, log_K]);
        p
    }

    #[test]
    fn linear_stages() {
        let p = linear_chain();
        let info = analyze::compute(&p);
        let s = stage(&p, &info, &params(), &minimize_proof()).unwrap();
        // 3 user stages
        assert_eq!(s.stages.len(), 3);
        assert!(s.opening.is_some());
    }

    #[test]
    fn diamond_parallel_batched() {
        let p = diamond();
        let info = analyze::compute(&p);
        let s = stage(&p, &info, &params(), &minimize_proof()).unwrap();
        // Should batch v1 and v2 together
        let batched = s.stages.iter().any(|sp| sp.vertices.len() == 2);
        assert!(batched);
    }

    #[test]
    fn empty_protocol() {
        let p = Protocol::new();
        let info = analyze::compute(&p);
        let params = CompileParams {
            dim_sizes: vec![],
            field_size_bytes: 32,
            pcs_proof_size: 100,
        };
        let s = stage(&p, &info, &params, &minimize_proof()).unwrap();
        assert!(s.stages.is_empty());
    }

    #[test]
    fn infeasible_returns_error() {
        let p = linear_chain();
        let info = analyze::compute(&p);
        let config = SolverConfig {
            proof_size: Objective::Bounded(0),
            peak_memory: Objective::Ignore,
            prover_time: Objective::Ignore,
        };
        let result = stage(&p, &info, &params(), &config);
        assert!(matches!(result, Err(CompileError::Infeasible { .. })));
    }

    #[test]
    fn multi_dim_has_reduction() {
        let p = multi_dim();
        let info = analyze::compute(&p);
        let s = stage(&p, &info, &params(), &minimize_proof()).unwrap();
        let has_reduction = s
            .protocol
            .polynomials
            .iter()
            .any(|p| p.name.starts_with("_eq_reduce_"));
        assert!(has_reduction);
    }

    #[test]
    fn compact_removes_gaps() {
        let mut a = vec![0, 2, 2, 4];
        let n = compact(&mut a);
        assert_eq!(a, vec![0, 1, 1, 2]);
        assert_eq!(n, 3);
    }

    #[test]
    fn evaluate_vertices_placed() {
        let mut p = Protocol::new();
        let d = p.dim("d");
        let rho = p.challenge("rho");
        let eq = p.poly("eq", &[d], PolyKind::Public(PublicPoly::Eq(None)));
        let a = p.poly("a", &[d], PolyKind::Virtual);
        let b = p.poly("b", &[d], PolyKind::Virtual);
        let c0 = p.sumcheck(eq * a, 0, &[d]);
        let ev = p.evaluate(b, c0[0]);
        let _ = p.sumcheck(eq * a, rho * ev, &[d]);

        let info = analyze::compute(&p);
        let s = stage(&p, &info, &params(), &minimize_proof()).unwrap();
        let has_eval_stage = s.stages.iter().any(|sp| !sp.evaluations.is_empty());
        assert!(has_eval_stage);
    }
}
