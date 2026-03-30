//! Graph-driven prover loop.
//!
//! Walks the [`ProtocolGraph`] stage by stage, executing sumcheck vertices
//! with the appropriate witness builders and collecting proofs.
//!
//! Virtual polynomial data is computed on-the-fly from `CycleRow` during
//! sumcheck witness construction — no pre-materialized tables. Committed
//! polynomial tables (for PCS opening) come from `WitnessStore`.

use std::collections::HashMap;
use std::sync::Arc;

use jolt_compute::ComputeBackend;
use jolt_field::Field;
use jolt_ir::protocol::{
    ChallengeLabel, ChallengeSpec, ClaimId, InputClaim, ProtocolGraph, StageId, Symbol,
    SymbolicPoint, Vertex,
};
use jolt_ir::PolynomialId;
use jolt_openings::{AdditivelyHomomorphic, OpeningReduction, ProverClaim, RlcReduction};
use jolt_sumcheck::{BatchedSumcheckProver, CaptureHandler, SumcheckClaim};
use jolt_transcript::{AppendToTranscript, Transcript};
use jolt_witness::TracePolynomials;

use crate::evaluators::kernel::KernelEvaluator;
use crate::witness::store::WitnessStore;
use crate::witness_builder;

#[derive(Debug)]
pub enum ProveError {
    Spartan(jolt_spartan::SpartanError),
}

impl From<jolt_spartan::SpartanError> for ProveError {
    fn from(e: jolt_spartan::SpartanError) -> Self {
        Self::Spartan(e)
    }
}

impl std::fmt::Display for ProveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Spartan(e) => write!(f, "spartan: {e}"),
        }
    }
}

impl std::error::Error for ProveError {}

struct EvalCache<F> {
    evals: Vec<Option<F>>,
    points: HashMap<StageId, Vec<F>>,
    symbols: HashMap<Symbol, usize>,
}

impl<F: Copy> EvalCache<F> {
    fn new(n: usize, symbols: &HashMap<Symbol, usize>) -> Self {
        Self {
            evals: vec![None; n],
            points: HashMap::new(),
            symbols: symbols.clone(),
        }
    }

    fn set(&mut self, id: ClaimId, val: F) {
        self.evals[id.0 as usize] = Some(val);
    }

    fn get(&self, id: ClaimId) -> F {
        self.evals[id.0 as usize].unwrap_or_else(|| panic!("eval miss: {id:?}"))
    }

    fn set_point(&mut self, stage: StageId, point: Vec<F>) {
        let _ = self.points.insert(stage, point);
    }

    fn resolve_point(&self, sp: &SymbolicPoint) -> Vec<F> {
        match sp {
            SymbolicPoint::Challenges(sid) => self.points[sid].clone(),
            SymbolicPoint::Concat(parts) => {
                parts.iter().flat_map(|p| self.resolve_point(p)).collect()
            }
            SymbolicPoint::Slice { source, range } => {
                let full = self.resolve_point(source);
                let start = range.start.resolve(&self.symbols).expect("slice start");
                let end = range.end.resolve(&self.symbols).expect("slice end");
                full[start..end].to_vec()
            }
        }
    }
}

struct StageChallenges<F> {
    scalars: HashMap<&'static str, F>,
}

fn squeeze_challenges<F: Field, T: Transcript<Challenge = F>>(
    specs: &[ChallengeSpec],
    symbols: &HashMap<Symbol, usize>,
    transcript: &mut T,
) -> StageChallenges<F> {
    let mut scalars = HashMap::new();
    for spec in specs {
        match spec {
            ChallengeSpec::Scalar { label } => {
                let _ = scalars.insert(*label, transcript.challenge());
            }
            ChallengeSpec::Vector { dim, .. } => {
                for _ in 0..dim.resolve(symbols).expect("dim") {
                    let _: F = transcript.challenge();
                }
            }
            ChallengeSpec::GammaPowers { label, .. } => {
                let _ = scalars.insert(*label, transcript.challenge());
            }
        }
    }
    StageChallenges { scalars }
}

fn evaluate_input_claim<F: Field>(
    input: &InputClaim,
    cache: &EvalCache<F>,
    sc: &StageChallenges<F>,
    ext: &HashMap<&str, F>,
) -> F {
    match input {
        InputClaim::Constant(c) => F::from_i64(*c),
        InputClaim::Formula {
            formula,
            challenge_labels,
        } => {
            let max_o = formula
                .definition
                .opening_bindings
                .iter()
                .map(|b| b.var_id + 1)
                .max()
                .unwrap_or(0) as usize;
            let mut openings = vec![F::zero(); max_o];
            for b in &formula.definition.opening_bindings {
                openings[b.var_id as usize] = cache.get(formula.opening_claims[&b.var_id]);
            }

            let max_c = formula.definition.num_challenges as usize;
            let mut challenges = vec![F::zero(); max_c];
            for (i, label) in challenge_labels.iter().enumerate() {
                challenges[i] = match label {
                    ChallengeLabel::PreSqueeze(name) => {
                        *sc.scalars.get(name).unwrap_or_else(|| {
                            panic!(
                                "missing pre_squeeze label: {name:?} (available: {:?})",
                                sc.scalars.keys().collect::<Vec<_>>()
                            )
                        })
                    }
                    ChallengeLabel::External(name) => *ext
                        .get(name)
                        .unwrap_or_else(|| panic!("missing external value: {name:?}")),
                };
            }

            formula.definition.evaluate(&openings, &challenges)
        }
    }
}

/// Input bundle for [`prove_from_graph`].
pub struct GraphProverInput<'a, F: Field, PCS: jolt_openings::CommitmentScheme<Field = F>, B, R> {
    pub graph: &'a ProtocolGraph,
    /// Lazy polynomial data — computes virtual poly values on-the-fly from trace.
    pub trace_polys: &'a TracePolynomials<'a, R>,
    /// Committed polynomial tables (for PCS opening proofs).
    pub committed_store: &'a WitnessStore<F>,
    pub symbols: &'a HashMap<Symbol, usize>,
    pub external: &'a HashMap<&'a str, F>,
    pub spartan_key: &'a jolt_spartan::UniformSpartanKey<F>,
    pub flat_witness: &'a [F],
    pub pcs_setup: &'a PCS::ProverSetup,
    pub pcs_verifier_setup: PCS::VerifierSetup,
    pub config: jolt_verifier::ProverConfig,
    pub backend: Arc<B>,
}

/// Output of [`prove_from_graph`].
pub type GraphProveOutput<F, PCS> = (
    jolt_verifier::JoltProof<F, PCS>,
    jolt_verifier::JoltVerifyingKey<F, PCS>,
);

pub fn prove_from_graph<F, PCS, B, R>(
    input: GraphProverInput<'_, F, PCS, B, R>,
) -> Result<GraphProveOutput<F, PCS>, ProveError>
where
    F: Field,
    PCS: AdditivelyHomomorphic<Field = F>,
    B: ComputeBackend,
    R: jolt_host::CycleRow,
{
    let GraphProverInput {
        graph,
        trace_polys,
        committed_store,
        symbols,
        external,
        spartan_key,
        flat_witness,
        pcs_setup,
        pcs_verifier_setup,
        config,
        backend,
    } = input;
    let mut transcript = jolt_transcript::Blake2bTranscript::<F>::new(b"jolt-v2");

    let (witness_commitment, _) = PCS::commit(flat_witness, pcs_setup);
    transcript.append_bytes(format!("{witness_commitment:?}").as_bytes());

    let commitments = commit_committed_polys::<PCS>(committed_store, &config, pcs_setup);

    let mut cache = EvalCache::new(graph.claim_graph.claims.len(), symbols);

    // S1: Spartan
    let (spartan_proof, _r_x, r_y) =
        jolt_spartan::UniformSpartanProver::prove_dense_with_challenges(
            spartan_key,
            flat_witness,
            &mut transcript,
        )?;

    let s1 = &graph.staging.stages[0];
    let r_y_rev: Vec<F> = r_y.iter().rev().copied().collect();
    let log_t = symbols[&Symbol::LOG_T];
    let r_cycle = &r_y_rev[..log_t];

    // S1 claim evaluations: virtual polys from trace, committed polys from store.
    for &vid in &s1.vertices {
        for &cid in graph.claim_graph.vertex(vid).produced_claims() {
            let poly_id = graph.claim_graph.claim(cid).polynomial;
            let eval = if poly_id.is_committed() && poly_id != PolynomialId::SpartanWitness {
                let table = committed_store.get(poly_id);
                let n = table.len().trailing_zeros() as usize;
                let point = if r_cycle.len() > n {
                    &r_cycle[r_cycle.len() - n..]
                } else {
                    r_cycle
                };
                witness_builder::eval_poly(table, point)
            } else {
                trace_polys.eval_at_point(poly_id, r_cycle)
            };
            cache.set(cid, eval);
        }
    }
    cache.set_point(s1.id, r_y_rev);

    // S2–S7: generic loop
    let mut stage_proofs = Vec::new();
    for stage in graph.staging.stages.iter().skip(1) {
        let sc = squeeze_challenges(&stage.pre_squeeze, symbols, &mut transcript);
        let num_vars = stage
            .challenge_point
            .num_vars
            .resolve(symbols)
            .expect("num_vars");

        let mut claims = Vec::new();
        let mut witnesses: Vec<Box<dyn jolt_sumcheck::SumcheckCompute<F>>> = Vec::new();

        for &vid in &stage.vertices {
            if let Vertex::Sumcheck(sv) = graph.claim_graph.vertex(vid) {
                let claimed_sum = evaluate_input_claim(&sv.input, &cache, &sc, external);
                let eq_source = sv
                    .deps
                    .first()
                    .map(|&dep_id| graph.claim_graph.claim(dep_id).point.clone())
                    .unwrap_or(SymbolicPoint::Challenges(stage.id));
                let witness = build_witness(
                    sv,
                    trace_polys,
                    committed_store,
                    &cache,
                    symbols,
                    &backend,
                    &graph.claim_graph,
                    &eq_source,
                    &sc,
                    external,
                );
                let vertex_num_vars = sv.num_vars.resolve(symbols).unwrap_or(num_vars);
                claims.push(SumcheckClaim {
                    num_vars: vertex_num_vars,
                    degree: sv.degree,
                    claimed_sum,
                });
                witnesses.push(witness);
            }
        }

        let captured = BatchedSumcheckProver::prove_with_handler(
            &claims,
            &mut witnesses,
            &mut transcript,
            CaptureHandler::with_capacity(num_vars),
        );
        let ep: Vec<F> = captured.challenges.iter().rev().copied().collect();
        // Set stage point BEFORE resolving claim points (claims may reference this stage).
        cache.set_point(stage.id, ep.clone());

        // Collect evaluations: first from witness produced_evaluations (2D polys),
        // then compute remaining from trace on-the-fly.
        let witness_evals: HashMap<PolynomialId, F> = HashMap::new();
        // TODO: populate from witness.produced_evaluations() once PhasedEvaluator
        // is wired for 2D polys. For now, all evals come from trace on-the-fly.

        let mut evals = Vec::new();
        for &vid in &stage.vertices {
            for &cid in graph.claim_graph.vertex(vid).produced_claims() {
                let poly_id = graph.claim_graph.claim(cid).polynomial;
                // Resolve the claim's full evaluation point (may be multi-stage concat).
                let claim_point = cache.resolve_point(&graph.claim_graph.claim(cid).point);
                let eval = if let Some(&we) = witness_evals.get(&poly_id) {
                    we
                } else if poly_id.is_committed() && poly_id != PolynomialId::SpartanWitness {
                    let table = committed_store.get(poly_id);
                    let n = table.len().trailing_zeros() as usize;
                    let point = if claim_point.len() > n {
                        &claim_point[claim_point.len() - n..]
                    } else {
                        &claim_point
                    };
                    witness_builder::eval_poly(table, point)
                } else {
                    trace_polys.eval_at_point(poly_id, &claim_point)
                };
                cache.set(cid, eval);
                evals.push(eval);
            }
        }

        for &e in &evals {
            e.append_to_transcript(&mut transcript);
        }

        stage_proofs.push(jolt_verifier::StageProof {
            round_polys: captured.proof,
            evals,
        });
    }

    // Point normalization
    for v in &graph.claim_graph.vertices {
        if let Vertex::PointNormalization(pn) = v {
            let padding = cache.resolve_point(&pn.padding_source);
            let lagrange = padding
                .iter()
                .fold(F::one(), |acc, r| acc * (F::one() - *r));
            for (&c, &p) in pn.consumes.iter().zip(pn.produces.iter()) {
                cache.set(p, cache.get(c) * lagrange);
            }
        }
    }

    // PCS opening: committed polys from WitnessStore
    let mut pcs_claims: Vec<ProverClaim<F>> = Vec::new();
    for v in &graph.claim_graph.vertices {
        if let Vertex::Opening(ov) = v {
            let claim = graph.claim_graph.claim(ov.consumes);
            let poly_id = claim.polynomial;
            if poly_id == PolynomialId::SpartanWitness {
                continue; // Opened by Spartan internally
            }
            let eval = cache.get(ov.consumes);
            let point = cache.resolve_point(&claim.point);
            let raw_table = committed_store.get(poly_id);
            // Zero-pad to match point dimension (1D polys in multi-dim unified point).
            let target_size = 1usize << point.len();
            let mut table = raw_table.to_vec();
            table.resize(target_size, F::zero());
            pcs_claims.push(ProverClaim {
                evaluations: table,
                point,
                eval,
            });
        }
    }
    pcs_claims.push(ProverClaim {
        evaluations: flat_witness.to_vec(),
        point: r_y,
        eval: spartan_proof.witness_eval,
    });

    // RLC reduction groups claims by point. Currently produces 2 groups:
    // 1. All committed polys at the unified point
    // 2. Spartan witness at r_y
    // TODO: Dory batch opening can merge these into a single proof.
    let (reduced, ()) =
        <RlcReduction as OpeningReduction<PCS>>::reduce_prover(pcs_claims, &mut transcript);
    let opening_proofs = reduced
        .into_iter()
        .map(|c| {
            let poly: PCS::Polynomial = c.evaluations.into();
            PCS::open(&poly, &c.point, c.eval, pcs_setup, None, &mut transcript)
        })
        .collect();

    // S1 evals for proof
    let spartan_evals: Vec<F> = s1
        .vertices
        .iter()
        .flat_map(|&vid| graph.claim_graph.vertex(vid).produced_claims())
        .map(|&cid| cache.get(cid))
        .collect();

    let proof = jolt_verifier::JoltProof {
        config,
        spartan_proof,
        spartan_evals,
        stage_proofs,
        opening_proofs,
        witness_commitment,
        commitments,
    };
    let vk = jolt_verifier::JoltVerifyingKey {
        spartan_key: spartan_key.clone(),
        pcs_setup: pcs_verifier_setup,
    };

    Ok((proof, vk))
}

/// Commit all committed polynomials from the WitnessStore.
fn commit_committed_polys<PCS: AdditivelyHomomorphic>(
    store: &WitnessStore<PCS::Field>,
    config: &jolt_verifier::ProverConfig,
    setup: &PCS::ProverSetup,
) -> Vec<PCS::Output> {
    let c = |id: PolynomialId| PCS::commit(store.get(id), setup).0;
    let params = config.one_hot_params_from_config();
    let mut out = vec![c(PolynomialId::RamInc), c(PolynomialId::RdInc)];
    for i in 0..params.instruction_d {
        out.push(c(PolynomialId::InstructionRa(i)));
    }
    for i in 0..params.bytecode_d {
        out.push(c(PolynomialId::BytecodeRa(i)));
    }
    for i in 0..params.ram_d {
        out.push(c(PolynomialId::RamRa(i)));
    }
    out
}

/// Builds a sumcheck witness for a vertex, computing polynomial values
/// on-the-fly from the trace.
#[allow(clippy::too_many_arguments)]
fn build_witness<F: Field, B: ComputeBackend, R: jolt_host::CycleRow>(
    sv: &jolt_ir::protocol::SumcheckVertex,
    trace_polys: &TracePolynomials<'_, R>,
    committed_store: &WitnessStore<F>,
    cache: &EvalCache<F>,
    _symbols: &HashMap<Symbol, usize>,
    backend: &Arc<B>,
    graph: &jolt_ir::protocol::ClaimGraph,
    stage_point: &SymbolicPoint,
    stage_challenges: &StageChallenges<F>,
    external: &HashMap<&str, F>,
) -> Box<dyn jolt_sumcheck::SumcheckCompute<F>> {
    use jolt_ir::protocol::PublicPolynomial as PP;

    let target_vars = sv.num_vars.resolve(&cache.symbols).unwrap_or(0);
    let eq_point = if sv.deps.is_empty() {
        // Zero-check with no deps — use zero point (trivial weighting).
        // The sumcheck claimed_sum is zero, so weighting doesn't affect soundness.
        vec![F::zero(); target_vars]
    } else {
        let mut ep = cache.resolve_point(stage_point);
        if ep.len() < target_vars {
            let mut padded = vec![F::zero(); target_vars - ep.len()];
            padded.extend(ep);
            ep = padded;
        }
        ep
    };
    let w_table: Vec<F> = match &sv.weighting {
        PP::Eq => jolt_poly::EqPolynomial::new(eq_point).evaluations(),
        PP::EqPlusOne => jolt_poly::EqPlusOnePolynomial::evals(&eq_point, None).1,
        PP::Lt => jolt_poly::LtPolynomial::evaluations(&eq_point),
        PP::Derived => vec![F::one(); 1 << eq_point.len()],
    };

    // Materialize polynomial evaluation tables: committed from store, virtual from trace.
    // All tables must be padded to w_table.len() (= 2^num_vars for the stage).
    let domain_size = w_table.len();
    let formula = &sv.formula;
    let mut poly_tables: Vec<Vec<F>> = Vec::new();

    for binding in &formula.definition.opening_bindings {
        let claim_id = formula.opening_claims[&binding.var_id];
        let poly_id = graph.claim(claim_id).polynomial;
        let mut table = if poly_id.is_committed() && poly_id != PolynomialId::SpartanWitness {
            committed_store.get(poly_id).to_vec()
        } else {
            trace_polys.materialize(poly_id)
        };
        // Zero-pad to domain size if the table is shorter (1D poly in multi-dim stage).
        table.resize(domain_size, F::zero());
        poly_tables.push(table);
    }

    // Resolve challenge values for the composition formula
    let challenge_labels = match &sv.input {
        InputClaim::Formula {
            challenge_labels, ..
        } => challenge_labels,
        InputClaim::Constant(_) => &[] as &[_],
    };
    let num_challenges = formula.definition.num_challenges as usize;
    let mut challenges = vec![F::zero(); num_challenges];
    for (i, label) in challenge_labels.iter().enumerate() {
        if i < num_challenges {
            challenges[i] = match label {
                ChallengeLabel::PreSqueeze(name) => stage_challenges
                    .scalars
                    .get(name)
                    .copied()
                    .unwrap_or_else(F::zero),
                ChallengeLabel::External(name) => {
                    external.get(name).copied().unwrap_or_else(F::zero)
                }
            };
        }
    }

    let comp_formula = formula.definition.to_composition_formula();
    let order = binding_order_for(sv);

    // Upload polynomial tables to backend buffers.
    let buffers: Vec<B::Buffer<F>> = poly_tables
        .iter()
        .map(|table| backend.upload(table))
        .collect();

    if comp_formula.is_linear_combination() {
        // Pre-combine: g[i] = Σ_j weight[j] * poly_tables[j][i]
        let weights = comp_formula.linear_combination_weights(&challenges);
        let n = w_table.len();
        let mut combined = vec![F::zero(); n];
        for (table, &weight) in poly_tables.iter().zip(weights.iter()) {
            for (i, c) in combined.iter_mut().enumerate() {
                if i < table.len() {
                    *c += weight * table[i];
                }
            }
        }
        let eq_formula = crate::evaluators::catalog::eq_product();
        let eq_buf = backend.upload(&w_table);
        let g_buf = backend.upload(&combined);
        let kernel = backend.compile_kernel(&eq_formula);
        Box::new(
            KernelEvaluator::from_formula(
                &eq_formula,
                vec![eq_buf, g_buf],
                kernel,
                backend.clone(),
            )
            .with_binding_order(order),
        )
    } else if comp_formula.is_hamming_booleanity() {
        // Scale the weighting table by the eq scale factor.
        let scale = comp_formula.hamming_eq_scale(&challenges);
        let scaled_w: Vec<F> = w_table.iter().map(|&w| w * scale).collect();
        let ham_formula = crate::evaluators::catalog::hamming_booleanity();
        let w_buf = backend.upload(&scaled_w);
        let h_buf = buffers
            .into_iter()
            .next()
            .unwrap_or_else(|| backend.upload(&vec![F::zero(); scaled_w.len()]));
        let kernel = backend.compile_kernel(&ham_formula);
        Box::new(
            KernelEvaluator::from_formula(
                &ham_formula,
                vec![w_buf, h_buf],
                kernel,
                backend.clone(),
            )
            .with_binding_order(order),
        )
    } else if comp_formula.as_product_sum().is_some() {
        // Toom-Cook path: RA virtual and similar product sumchecks.
        let compiler_formula = jolt_cpu::from_ir_formula(&comp_formula);
        let kernel = backend.compile_kernel(&compiler_formula);
        let claimed_sum = F::zero(); // Set by set_claim() before first round
        Box::new(KernelEvaluator::from_formula_toom_cook(
            &compiler_formula,
            buffers,
            kernel,
            w_table,
            claimed_sum,
            order,
            backend.clone(),
        ))
    } else {
        // General path: compile kernel with baked-in challenge values.
        let compiler_formula = jolt_cpu::from_ir_formula(&comp_formula);
        let kernel = backend.compile_kernel_with_challenges(&compiler_formula, &challenges);
        let mut inputs = vec![backend.upload(&w_table)];
        inputs.extend(buffers);
        Box::new(
            KernelEvaluator::from_formula(&compiler_formula, inputs, kernel, backend.clone())
                .with_binding_order(order),
        )
    }
}

/// Derives the variable binding order from the vertex's first phase.
fn binding_order_for(sv: &jolt_ir::protocol::SumcheckVertex) -> jolt_compute::BindingOrder {
    use jolt_ir::protocol::VariableGroup;
    match sv.phases.first().map(|p| p.variable_group) {
        Some(VariableGroup::Address) => jolt_compute::BindingOrder::HighToLow,
        _ => jolt_compute::BindingOrder::LowToHigh,
    }
}
