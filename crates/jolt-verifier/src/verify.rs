//! Graph-driven proof verification.
//!
//! [`verify_from_graph`] verifies a Jolt proof by walking the [`ProtocolGraph`]
//! -- no per-stage hand-coded logic. Challenge squeezing, sumcheck verification,
//! formula evaluation, and PCS opening checks are all driven by the graph.

use std::collections::HashMap;

use jolt_field::Field;
use jolt_ir::protocol::{
    ChallengeLabel, ChallengeSpec, ClaimId, InputClaim, ProtocolGraph, StageId, Symbol,
    SymbolicPoint, Vertex,
};
use jolt_openings::{AdditivelyHomomorphic, VerifierClaim};
use jolt_sumcheck::{BatchedSumcheckVerifier, SumcheckClaim};
use jolt_transcript::{AppendToTranscript, Transcript};

use crate::error::JoltError;
use crate::key::JoltVerifyingKey;
use crate::proof::JoltProof;

/// Caches polynomial evaluations produced by each stage, indexed by [`ClaimId`].
///
/// ClaimIds are dense consecutive integers (verified by protocol graph validation),
/// so a flat `Vec<Option<F>>` gives O(1) lookup with zero overhead.
struct EvalCache<F> {
    evals: Vec<Option<F>>,
    /// Challenge point produced by each stage (reversed to match MSB convention).
    points: HashMap<StageId, Vec<F>>,
}

impl<F: Clone> EvalCache<F> {
    fn new(num_claims: usize) -> Self {
        Self {
            evals: vec![None; num_claims],
            points: HashMap::new(),
        }
    }

    fn set(&mut self, id: ClaimId, val: F) {
        self.evals[id.0 as usize] = Some(val);
    }

    fn get(&self, id: ClaimId) -> Option<&F> {
        self.evals[id.0 as usize].as_ref()
    }

    fn set_point(&mut self, stage: StageId, point: Vec<F>) {
        let _ = self.points.insert(stage, point);
    }
}

/// Challenge values squeezed from the transcript for one stage.
struct StageChallenges<F> {
    /// Maps pre_squeeze label → base scalar value.
    /// For `Scalar` specs: the squeezed scalar.
    /// For `GammaPowers` specs: the base gamma.
    scalars: HashMap<&'static str, F>,
}

/// Builds a symbol table for resolving [`SymbolicExpr`](jolt_ir::protocol::SymbolicExpr)
/// dimensions.
pub fn build_symbol_table(
    log_t: usize,
    log_k: usize,
    log_rows: usize,
    log_cols: usize,
    d_instr: usize,
    d_bc: usize,
    d_ram: usize,
) -> HashMap<Symbol, usize> {
    let mut syms = HashMap::new();
    let _ = syms.insert(Symbol::LOG_T, log_t);
    let _ = syms.insert(Symbol::LOG_K, log_k);
    let _ = syms.insert(Symbol::LOG_ROWS, log_rows);
    let _ = syms.insert(Symbol::LOG_COLS, log_cols);
    let _ = syms.insert(Symbol::D_INSTR, d_instr);
    let _ = syms.insert(Symbol::D_BC, d_bc);
    let _ = syms.insert(Symbol::D_RAM, d_ram);
    let _ = syms.insert(Symbol::D_TOTAL, d_instr + d_bc + d_ram);
    syms
}

fn squeeze_stage_challenges<F: Field, T: Transcript<Challenge = F>>(
    specs: &[ChallengeSpec],
    symbols: &HashMap<Symbol, usize>,
    transcript: &mut T,
) -> StageChallenges<F> {
    let mut scalars = HashMap::new();

    for spec in specs {
        match spec {
            ChallengeSpec::Scalar { label } => {
                let val: F = transcript.challenge();
                let _ = scalars.insert(*label, val);
            }
            ChallengeSpec::Vector { label: _, dim } => {
                let n = dim.resolve(symbols).expect("unresolvable SymbolicExpr dim");
                // Squeeze n scalars from transcript (consumed but not stored as scalar)
                for _ in 0..n {
                    let _: F = transcript.challenge();
                }
            }
            ChallengeSpec::GammaPowers { label, count: _ } => {
                let base: F = transcript.challenge();
                let _ = scalars.insert(*label, base);
            }
        }
    }

    StageChallenges { scalars }
}

/// Compute `[1, γ, γ², …, γ^{count-1}]`.
pub fn gamma_powers<F: Field>(gamma: F, count: usize) -> Vec<F> {
    (0..count)
        .scan(F::one(), |g, _| {
            let v = *g;
            *g *= gamma;
            Some(v)
        })
        .collect()
}

/// Evaluate a Formula input claim using the eval cache and stage challenges.
fn evaluate_formula<F: Field>(
    formula: &jolt_ir::protocol::ClaimFormula,
    challenge_labels: &[ChallengeLabel],
    eval_cache: &EvalCache<F>,
    stage_challenges: &StageChallenges<F>,
    external_values: &HashMap<&str, F>,
) -> F {
    // Build openings array: indexed by formula var_id
    let max_opening_var = formula
        .definition
        .opening_bindings
        .iter()
        .map(|b| b.var_id + 1)
        .max()
        .unwrap_or(0) as usize;

    let mut openings = vec![F::zero(); max_opening_var];
    for binding in &formula.definition.opening_bindings {
        let claim_id = formula.opening_claims[&binding.var_id];
        openings[binding.var_id as usize] = eval_cache
            .get(claim_id)
            .copied()
            .expect("eval cache miss for formula opening");
    }

    // Build challenges array: indexed by formula var_id
    let max_challenge_var = formula.definition.num_challenges as usize;

    let mut challenges = vec![F::zero(); max_challenge_var];
    for (i, label) in challenge_labels.iter().enumerate() {
        challenges[i] = match label {
            ChallengeLabel::PreSqueeze(name) => {
                *stage_challenges.scalars.get(name).unwrap_or_else(|| {
                    panic!("challenge label {name:?} not found in stage pre_squeeze")
                })
            }
            ChallengeLabel::External(name) => *external_values
                .get(name)
                .unwrap_or_else(|| panic!("external value {name:?} not provided")),
        };
    }

    formula.definition.evaluate(&openings, &challenges)
}

/// Verify a Jolt proof by walking the protocol graph.
///
/// All stage logic is derived from the graph -- no per-stage hand-coded formulas.
///
/// # Parameters
///
/// - `graph`: The protocol graph (same one used by the prover).
/// - `proof`: The complete Jolt proof.
/// - `vk`: The verifying key (Spartan key + PCS setup).
/// - `symbols`: Symbol table for resolving symbolic dimensions.
/// - `external_values`: Externally-computed challenge values (e.g., RAM init eval).
#[allow(clippy::implicit_hasher)]
pub fn verify_from_graph<F, PCS>(
    graph: &ProtocolGraph,
    proof: &JoltProof<F, PCS>,
    vk: &JoltVerifyingKey<F, PCS>,
    symbols: &HashMap<Symbol, usize>,
    external_values: &HashMap<&str, F>,
) -> Result<(), JoltError>
where
    F: Field,
    PCS: AdditivelyHomomorphic<Field = F>,
{
    let mut transcript = jolt_transcript::Blake2bTranscript::<F>::new(b"jolt-v2");
    transcript.append_bytes(format!("{:?}", proof.witness_commitment).as_bytes());

    let num_claims = graph.claim_graph.claims.len();
    let mut cache = EvalCache::new(num_claims);

    // S1: Spartan (special-cased — opaque internal structure)
    let (_r_x, r_y) =
        crate::verifier::verify_spartan(&vk.spartan_key, &proof.spartan_proof, &mut transcript)?;

    // Populate S1 produced claims from the proof's spartan_evals.
    let s1_stage = &graph.staging.stages[0];
    let mut s1_eval_idx = 0;
    for &vid in &s1_stage.vertices {
        let vertex = graph.claim_graph.vertex(vid);
        for claim_id in vertex.all_produced_claims() {
            let eval = if s1_eval_idx < proof.spartan_evals.len() {
                proof.spartan_evals[s1_eval_idx]
            } else {
                F::zero()
            };
            cache.set(claim_id, eval);
            s1_eval_idx += 1;
        }
    }
    // Store full r_y (reversed) as S1's point. The graph's r_cycle() =
    // Slice(Challenges(S1), 0..log_T) extracts the first log_T elements.
    cache.set_point(s1_stage.id, r_y.iter().rev().copied().collect());

    // S2..S7: generic graph-driven verification loop
    for (stage_proof_idx, stage) in graph.staging.stages.iter().skip(1).enumerate() {
        // Squeeze pre-stage challenges
        let stage_challenges =
            squeeze_stage_challenges(&stage.pre_squeeze, symbols, &mut transcript);

        // Resolve num_vars for this stage
        let num_vars = stage
            .challenge_point
            .num_vars
            .resolve(symbols)
            .expect("unresolvable num_vars for stage");

        // Build sumcheck claims for each vertex
        let mut sumcheck_claims = Vec::new();
        for &vid in &stage.vertices {
            let vertex = graph.claim_graph.vertex(vid);
            if let Vertex::Sumcheck(sc) = vertex {
                let claimed_sum = match &sc.input {
                    InputClaim::Constant(c) => F::from_i64(*c),
                    InputClaim::Formula {
                        formula,
                        challenge_labels,
                    } => evaluate_formula(
                        formula,
                        challenge_labels,
                        &cache,
                        &stage_challenges,
                        external_values,
                    ),
                };

                sumcheck_claims.push(SumcheckClaim {
                    num_vars,
                    degree: sc.degree,
                    claimed_sum,
                });
            }
        }

        // Verify sumcheck
        if stage_proof_idx >= proof.stage_proofs.len() {
            return Err(JoltError::InvalidProof(format!(
                "missing stage proof for stage {:?}",
                stage.id
            )));
        }
        let stage_proof = &proof.stage_proofs[stage_proof_idx];

        let (final_eval, challenges, alpha) = BatchedSumcheckVerifier::verify_with_alpha(
            &sumcheck_claims,
            &stage_proof.round_polys,
            &mut transcript,
        )
        .map_err(|e| JoltError::StageVerification {
            stage: stage.id.0 as usize,
            reason: e.to_string(),
        })?;

        let eval_point: Vec<F> = challenges.iter().rev().copied().collect();

        // Flush evals to transcript (Fiat-Shamir binding)
        for &e in &stage_proof.evals {
            e.append_to_transcript(&mut transcript);
        }

        // Unpack evals into cache (includes side-effect claims)
        let mut eval_idx = 0;
        for &vid in &stage.vertices {
            let vertex = graph.claim_graph.vertex(vid);
            for claim_id in vertex.all_produced_claims() {
                if eval_idx >= stage_proof.evals.len() {
                    return Err(JoltError::InvalidProof(format!(
                        "stage {:?} has fewer evals than produced claims",
                        stage.id
                    )));
                }
                cache.set(claim_id, stage_proof.evals[eval_idx]);
                eval_idx += 1;
            }
        }

        // Output formula check: verify final_eval == Σ α^j · pad_j · g_j
        // The formula already includes the weighting (eq/lt/eq+1) as challenge values —
        // no separate weighting multiplication needed.
        let max_num_vars = sumcheck_claims
            .iter()
            .map(|c| c.num_vars)
            .max()
            .unwrap_or(0);
        let mut expected_final = F::zero();
        let mut alpha_power = F::one();

        let mut vertex_eval_offset = 0usize;
        for &vid in &stage.vertices {
            let vertex = graph.claim_graph.vertex(vid);
            if let Vertex::Sumcheck(sc) = vertex {
                let offset = max_num_vars - sc.num_vars.resolve(symbols).unwrap_or(num_vars);
                let pad = F::one().mul_pow_2(offset);

                let n_produced = sc.produces.len();
                let vertex_evals =
                    &stage_proof.evals[vertex_eval_offset..vertex_eval_offset + n_produced];

                // Compute output formula challenge values: weighting × gamma powers.
                // The formula's Derived challenges encode eq(eval_point, source_point) × gamma^i.
                let g_eval = evaluate_output_formula_with_challenges(
                    &sc.formula,
                    vertex_evals,
                    &sc.weighting,
                    &sc.deps,
                    &eval_point,
                    &cache,
                    &graph.claim_graph,
                    &stage_challenges,
                    external_values,
                );

                expected_final += alpha_power * pad * g_eval;
                alpha_power *= alpha;
                vertex_eval_offset += n_produced;
            }
        }

        if expected_final != final_eval {
            return Err(JoltError::EvaluationMismatch {
                stage: stage.id.0 as usize,
                reason: format!(
                    "output formula check failed: expected {expected_final:?}, got {final_eval:?}"
                ),
            });
        }

        cache.set_point(stage.id, eval_point);
    }

    // Point normalization
    for &vid in &graph.staging.opening.vertices {
        let vertex = graph.claim_graph.vertex(vid);
        if let Vertex::PointNormalization(pn) = vertex {
            let lagrange = compute_lagrange_factor(&pn.padding_source, &cache, symbols);

            for (&consumed, &produced) in pn.consumes.iter().zip(pn.produces.iter()) {
                let val = cache
                    .get(consumed)
                    .copied()
                    .expect("eval cache miss for normalization input");
                cache.set(produced, val * lagrange);
            }
        }
    }

    // PCS opening verification
    // Collect committed polynomial opening claims from Opening vertices.
    let mut pcs_claims: Vec<VerifierClaim<F, PCS::Output>> = Vec::new();

    let mut commitment_idx = 0;
    for &vid in &graph.staging.opening.vertices {
        let vertex = graph.claim_graph.vertex(vid);
        if let Vertex::Opening(ov) = vertex {
            let claim = graph.claim_graph.claim(ov.consumes);
            let eval = cache.get(ov.consumes).copied().ok_or_else(|| {
                JoltError::InvalidProof(format!("missing eval for opening claim {:?}", ov.consumes))
            })?;
            let point = resolve_point(&claim.point, &cache, symbols);
            if commitment_idx < proof.commitments.len() {
                pcs_claims.push(VerifierClaim {
                    commitment: proof.commitments[commitment_idx].clone(),
                    point,
                    eval,
                });
                commitment_idx += 1;
            }
        }
    }

    // Spartan witness commitment (opened internally by Spartan, not in the graph)
    pcs_claims.push(VerifierClaim {
        commitment: proof.witness_commitment.clone(),
        point: cache.points.get(&StageId(0)).cloned().unwrap_or_default(),
        eval: proof.spartan_proof.witness_eval,
    });

    crate::verifier::verify_openings::<PCS, _>(
        pcs_claims,
        &proof.opening_proofs,
        &vk.pcs_setup,
        &mut transcript,
    )?;

    Ok(())
}

/// Infer the weighting source point from a vertex's dependencies.
///
/// For most sumcheck instances, the eq/lt/eq+1 weighting is evaluated against a
/// prior stage's challenge point. We infer which stage by looking at the vertex's
/// dep claims and finding which stage produced them.
fn infer_weighting_source<F: Clone>(
    deps: &[ClaimId],
    graph: &jolt_ir::protocol::ClaimGraph,
    cache: &EvalCache<F>,
) -> Option<Vec<F>> {
    for &dep_id in deps {
        let claim = graph.claim(dep_id);
        if let SymbolicPoint::Challenges(sid) = &claim.point {
            if let Some(pt) = cache.points.get(sid) {
                return Some(pt.clone());
            }
        }
    }
    None
}

/// Evaluate the output formula with properly resolved challenge values.
///
/// The formula's Derived challenges encode weighting × gamma combinations.
/// We compute:
/// - Weighting eval (eq/lt/eq+1) at `eval_point` against the upstream source point
/// - Gamma powers from the stage's pre_squeeze
/// - Combined challenge values: weighting_eval × gamma^i for each challenge variable
#[allow(clippy::too_many_arguments)]
fn evaluate_output_formula_with_challenges<F: Field>(
    formula: &jolt_ir::protocol::ClaimFormula,
    evals: &[F],
    weighting: &jolt_ir::protocol::PublicPolynomial,
    deps: &[ClaimId],
    eval_point: &[F],
    cache: &EvalCache<F>,
    graph: &jolt_ir::protocol::ClaimGraph,
    stage_challenges: &StageChallenges<F>,
    _external_values: &HashMap<&str, F>,
) -> F {
    use jolt_ir::protocol::PublicPolynomial as PP;

    // Map formula opening variables to proof evaluations
    let max_var = formula
        .definition
        .opening_bindings
        .iter()
        .map(|b| b.var_id + 1)
        .max()
        .unwrap_or(0) as usize;
    let mut openings = vec![F::zero(); max_var];
    for (i, binding) in formula.definition.opening_bindings.iter().enumerate() {
        if i < evals.len() {
            openings[binding.var_id as usize] = evals[i];
        }
    }

    // Compute weighting evaluation at eval_point against inferred source point
    let source_point = infer_weighting_source(deps, graph, cache);
    let w_eval = match (weighting, &source_point) {
        (PP::Eq, Some(sp)) => jolt_poly::EqPolynomial::new(sp.clone()).evaluate(eval_point),
        (PP::EqPlusOne, Some(sp)) => {
            jolt_poly::EqPlusOnePolynomial::new(sp.clone()).evaluate(eval_point)
        }
        (PP::Lt, Some(sp)) => jolt_poly::LtPolynomial::evaluate(sp, eval_point),
        (PP::Derived, _) => F::one(), // Derived weights are absorbed into challenge values
        (_, None) => F::one(),
    };

    // Build challenge values: the first challenge gets the weighting eval,
    // subsequent challenges get gamma powers from stage pre_squeeze.
    // Common pattern: challenges = [w_eval, w_eval*γ, w_eval*γ², ...]
    // or [w_eval*(1+γ), w_eval*γ] for RamRW-style formulas.
    let n_challenges = formula.definition.num_challenges as usize;
    let mut challenges = vec![F::zero(); n_challenges];

    // Find the gamma base from any stage challenge (heuristic: first available scalar)
    let gamma_base = stage_challenges
        .scalars
        .values()
        .next()
        .copied()
        .unwrap_or(F::one());

    // Fill challenges: w_eval × gamma^i for each challenge variable
    let mut gamma_power = F::one();
    for c in challenges.iter_mut().take(n_challenges) {
        *c = w_eval * gamma_power;
        gamma_power *= gamma_base;
    }

    formula.definition.evaluate(&openings, &challenges)
}

/// Resolve a symbolic point to concrete field elements using the eval cache.
fn resolve_point<F: Clone>(
    point: &SymbolicPoint,
    cache: &EvalCache<F>,
    symbols: &HashMap<Symbol, usize>,
) -> Vec<F> {
    match point {
        SymbolicPoint::Challenges(sid) => cache.points.get(sid).cloned().unwrap_or_default(),
        SymbolicPoint::Concat(parts) => parts
            .iter()
            .flat_map(|p| resolve_point(p, cache, symbols))
            .collect(),
        SymbolicPoint::Slice { source, range } => {
            let full = resolve_point(source, cache, symbols);
            let start = range.start.resolve(symbols).expect("slice start");
            let end = range.end.resolve(symbols).expect("slice end");
            full[start..end].to_vec()
        }
    }
}

/// Compute `∏(1 − r_i)` from a padding source's challenge point.
fn compute_lagrange_factor<F: Field>(
    padding_source: &SymbolicPoint,
    cache: &EvalCache<F>,
    symbols: &HashMap<Symbol, usize>,
) -> F {
    let point = resolve_point(padding_source, cache, symbols);
    point.iter().fold(F::one(), |acc, r| acc * (F::one() - *r))
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_field::Fr;
    use num_traits::{One, Zero};

    #[test]
    fn eval_cache_basic() {
        let mut cache = EvalCache::<Fr>::new(5);
        cache.set(ClaimId(2), Fr::one());
        assert_eq!(cache.get(ClaimId(2)), Some(&Fr::one()));
        assert_eq!(cache.get(ClaimId(0)), None);
    }

    #[test]
    fn gamma_powers_correctness() {
        let base = Fr::from_u64(3);
        let powers = gamma_powers(base, 4);
        assert_eq!(powers[0], Fr::one());
        assert_eq!(powers[1], base);
        assert_eq!(powers[2], base * base);
        assert_eq!(powers[3], base * base * base);
    }

    #[test]
    fn squeeze_challenge_specs() {
        let specs = vec![
            ChallengeSpec::Scalar { label: "alpha" },
            ChallengeSpec::GammaPowers {
                label: "gamma",
                count: jolt_ir::protocol::SymbolicExpr::Concrete(3),
            },
        ];

        let symbols = HashMap::new();
        let mut t = jolt_transcript::Blake2bTranscript::<Fr>::new(b"test");

        let sc = squeeze_stage_challenges(&specs, &symbols, &mut t);

        assert!(sc.scalars.contains_key("alpha"));
        assert!(sc.scalars.contains_key("gamma"));
    }

    #[test]
    fn build_symbol_table_has_all_keys() {
        let syms = build_symbol_table(20, 8, 22, 10, 8, 4, 3);
        assert_eq!(syms[&Symbol::LOG_T], 20);
        assert_eq!(syms[&Symbol::LOG_K], 8);
        assert_eq!(syms[&Symbol::D_TOTAL], 15);
    }

    #[test]
    fn eval_cache_set_and_get_all() {
        let mut cache = EvalCache::<Fr>::new(3);
        cache.set(ClaimId(0), Fr::zero());
        cache.set(ClaimId(1), Fr::one());
        cache.set(ClaimId(2), Fr::from_u64(42));
        assert_eq!(cache.get(ClaimId(0)), Some(&Fr::zero()));
        assert_eq!(cache.get(ClaimId(1)), Some(&Fr::one()));
        assert_eq!(cache.get(ClaimId(2)), Some(&Fr::from_u64(42)));
    }

    #[test]
    fn formula_evaluation_simple() {
        use jolt_ir::protocol::ClaimFormula;
        use jolt_ir::{ClaimDefinition, ExprBuilder, OpeningBinding, PolynomialId};

        // Formula: gamma * opening_0 + gamma^2 * opening_1
        let b = ExprBuilder::new();
        let o0 = b.opening(0);
        let o1 = b.opening(1);
        let g = b.challenge(0);
        let expr = b.build(g * o0 + g * g * o1);

        let def = ClaimDefinition {
            expr,
            opening_bindings: vec![
                OpeningBinding {
                    var_id: 0,
                    polynomial: PolynomialId::RamInc,
                },
                OpeningBinding {
                    var_id: 1,
                    polynomial: PolynomialId::RdInc,
                },
            ],
            num_challenges: 1,
        };

        let mut opening_claims = HashMap::new();
        let _ = opening_claims.insert(0u32, ClaimId(0));
        let _ = opening_claims.insert(1u32, ClaimId(1));

        let formula = ClaimFormula {
            definition: def,
            opening_claims,
        };

        // Set up eval cache
        let mut cache = EvalCache::<Fr>::new(2);
        let val_0 = Fr::from_u64(5);
        let val_1 = Fr::from_u64(7);
        cache.set(ClaimId(0), val_0);
        cache.set(ClaimId(1), val_1);

        // Set up stage challenges
        let gamma = Fr::from_u64(3);
        let mut scalars = HashMap::new();
        let _ = scalars.insert("test_gamma", gamma);
        let stage_challenges = StageChallenges { scalars };

        let labels = vec![ChallengeLabel::PreSqueeze("test_gamma")];
        let external = HashMap::new();

        let result = evaluate_formula(&formula, &labels, &cache, &stage_challenges, &external);

        // Expected: gamma * 5 + gamma^2 * 7 = 3*5 + 9*7 = 15 + 63 = 78
        assert_eq!(result, Fr::from_u64(78));
    }
}
