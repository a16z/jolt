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
use jolt_witness::TracePolynomials;
use jolt_openings::{AdditivelyHomomorphic, OpeningReduction, ProverClaim, RlcReduction};
use jolt_sumcheck::{BatchedSumcheckProver, CaptureHandler, SumcheckClaim};
use jolt_transcript::{AppendToTranscript, Transcript};

use crate::evaluators::kernel::KernelEvaluator;
use crate::witness::store::WitnessStore;

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
                    ChallengeLabel::PreSqueeze(name) => sc.scalars[name],
                    ChallengeLabel::External(name) => ext[name],
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

    // S1 claim evaluations: computed lazily from trace via TracePolynomials.
    for &vid in &s1.vertices {
        for &cid in graph.claim_graph.vertex(vid).produced_claims() {
            let poly_id = graph.claim_graph.claim(cid).polynomial;
            cache.set(cid, trace_polys.eval_at_point(poly_id, r_cycle));
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
                    sv, trace_polys, &cache, symbols, &backend, &graph.claim_graph, &eq_source,
                    &sc, external,
                );
                claims.push(SumcheckClaim {
                    num_vars,
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

        // Collect evaluations: first from witness produced_evaluations (2D polys),
        // then compute remaining from trace on-the-fly.
        let witness_evals: HashMap<PolynomialId, F> = HashMap::new();
        // TODO: populate from witness.produced_evaluations() once PhasedEvaluator
        // is wired for 2D polys. For now, all evals come from trace on-the-fly.

        let mut evals = Vec::new();
        for &vid in &stage.vertices {
            for &cid in graph.claim_graph.vertex(vid).produced_claims() {
                let poly_id = graph.claim_graph.claim(cid).polynomial;
                let eval = if let Some(&we) = witness_evals.get(&poly_id) {
                    we
                } else {
                    trace_polys.eval_at_point(poly_id, &ep)
                };
                cache.set(cid, eval);
                evals.push(eval);
            }
        }

        // Side-effect claim evaluations
        for &vid in &stage.vertices {
            if let Vertex::Sumcheck(sv) = graph.claim_graph.vertex(vid) {
                for &cid in &sv.side_effect_claims {
                    let poly_id = graph.claim_graph.claim(cid).polynomial;
                    let eval = trace_polys.eval_at_point(poly_id, &ep);
                    cache.set(cid, eval);
                    evals.push(eval);
                }
            }
        }

        for &e in &evals {
            e.append_to_transcript(&mut transcript);
        }
        cache.set_point(stage.id, ep);

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
            let table = committed_store.get(poly_id);
            pcs_claims.push(ProverClaim {
                evaluations: table.to_vec(),
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
    cache: &EvalCache<F>,
    _symbols: &HashMap<Symbol, usize>,
    backend: &Arc<B>,
    graph: &jolt_ir::protocol::ClaimGraph,
    stage_point: &SymbolicPoint,
    stage_challenges: &StageChallenges<F>,
    external: &HashMap<&str, F>,
) -> Box<dyn jolt_sumcheck::SumcheckCompute<F>> {
    use jolt_ir::protocol::PublicPolynomial as PP;

    let eq_point = cache.resolve_point(stage_point);
    let w_table: Vec<F> = match &sv.weighting {
        PP::Eq => jolt_poly::EqPolynomial::new(eq_point).evaluations(),
        PP::EqPlusOne => jolt_poly::EqPlusOnePolynomial::evals(&eq_point, None).1,
        PP::Lt => jolt_poly::LtPolynomial::evaluations(&eq_point),
        PP::Derived => vec![F::one(); 1 << eq_point.len()],
    };

    // Materialize polynomial evaluation tables on-demand from trace.
    let formula = &sv.formula;
    let mut poly_tables: Vec<Vec<F>> = Vec::new();

    for binding in &formula.definition.opening_bindings {
        let claim_id = formula.opening_claims[&binding.var_id];
        let poly_id = graph.claim(claim_id).polynomial;
        poly_tables.push(trace_polys.materialize(poly_id));
    }

    // Resolve challenge values for compile_descriptor
    let challenge_labels = match &sv.input {
        InputClaim::Formula { challenge_labels, .. } => challenge_labels,
        InputClaim::Constant(_) => &[] as &[_],
    };
    let num_challenges = formula.definition.num_challenges as usize;
    let mut challenges = vec![F::zero(); num_challenges];
    for (i, label) in challenge_labels.iter().enumerate() {
        if i < num_challenges {
            challenges[i] = match label {
                ChallengeLabel::PreSqueeze(name) => {
                    stage_challenges.scalars.get(name).copied().unwrap_or_else(F::zero)
                }
                ChallengeLabel::External(name) => {
                    external.get(name).copied().unwrap_or_else(F::zero)
                }
            };
        }
    }

    // Compile the formula into a kernel descriptor + materialized challenge coefficients.
    let (desc, materialized) = formula.definition.compile_descriptor::<F>(&challenges);
    let order = binding_order_for(sv);

    // Upload polynomial tables to backend buffers.
    let buffers: Vec<B::Buffer<F>> = poly_tables
        .iter()
        .map(|table| backend.upload(table))
        .collect();

    use jolt_ir::KernelShape;
    match &desc.shape {
        KernelShape::EqProduct => {
            // Pre-combine: g[i] = Σ_j materialized[j] * poly_tables[j][i]
            let n = w_table.len();
            let mut combined = vec![F::zero(); n];
            for (j, (table, &weight)) in poly_tables.iter().zip(materialized.iter()).enumerate() {
                for (i, c) in combined.iter_mut().enumerate() {
                    if i < table.len() {
                        *c += weight * table[i];
                    }
                }
                let _ = j;
            }
            let eq_buf = backend.upload(&w_table);
            let g_buf = backend.upload(&combined);
            let kernel = backend.compile_kernel(&desc);
            Box::new(
                KernelEvaluator::from_descriptor(&desc, vec![eq_buf, g_buf], kernel, backend.clone())
                    .with_binding_order(order),
            )
        }
        KernelShape::HammingBooleanity => {
            // Scale the weighting table by the materialized eq scale factor.
            let scale = materialized.first().copied().unwrap_or(F::one());
            let scaled_w: Vec<F> = w_table.iter().map(|&w| w * scale).collect();
            let w_buf = backend.upload(&scaled_w);
            let h_buf = buffers.into_iter().next().unwrap_or_else(|| backend.upload(&vec![F::zero(); scaled_w.len()]));
            let kernel = backend.compile_kernel(&desc);
            Box::new(
                KernelEvaluator::from_descriptor(&desc, vec![w_buf, h_buf], kernel, backend.clone())
                    .with_binding_order(order),
            )
        }
        KernelShape::Custom { .. } | KernelShape::ProductSum { .. } => {
            // General path: compile kernel with baked-in challenge values, use all poly buffers.
            let kernel = backend.compile_kernel(&desc);
            let mut inputs = vec![backend.upload(&w_table)];
            inputs.extend(buffers);
            Box::new(
                KernelEvaluator::from_descriptor(&desc, inputs, kernel, backend.clone())
                    .with_binding_order(order),
            )
        }
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
