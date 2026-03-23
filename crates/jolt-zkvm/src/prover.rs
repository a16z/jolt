//! Graph-driven prover loop.
//!
//! Walks the [`ProtocolGraph`] stage by stage, executing sumcheck vertices
//! with the appropriate witness builders and collecting proofs.

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

use crate::tables::PolynomialTables;
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
}

impl<F: Copy> EvalCache<F> {
    fn new(n: usize) -> Self {
        Self { evals: vec![None; n], points: HashMap::new() }
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
            SymbolicPoint::Concat(parts) => parts.iter().flat_map(|p| self.resolve_point(p)).collect(),
            SymbolicPoint::Slice { source, .. } => self.resolve_point(source), // TODO: range
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
        InputClaim::Formula { formula, challenge_labels } => {
            let max_o = formula.definition.opening_bindings.iter()
                .map(|b| b.var_id + 1).max().unwrap_or(0) as usize;
            let mut openings = vec![F::zero(); max_o];
            for b in &formula.definition.opening_bindings {
                openings[b.var_id as usize] = cache.get(formula.opening_claims[&b.var_id]);
            }

            let max_c = formula.definition.challenge_bindings.iter()
                .map(|b| b.var_id + 1).max().unwrap_or(0) as usize;
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

fn eval_produced_claim<F: Field>(
    poly_id: PolynomialId,
    tables: &PolynomialTables<F>,
    eval_point: &[F],
) -> F {
    // Try to look up the polynomial table; return zero for virtual polys
    // that don't have stored tables.
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        witness_builder::eval_poly(tables.get(poly_id), eval_point)
    }))
    .unwrap_or(F::zero())
}

/// Input bundle for [`prove_from_graph`].
pub struct GraphProverInput<'a, F: Field, PCS: jolt_openings::CommitmentScheme<Field = F>, B> {
    pub graph: &'a ProtocolGraph,
    pub tables: &'a PolynomialTables<F>,
    pub symbols: &'a HashMap<Symbol, usize>,
    pub external: &'a HashMap<&'a str, F>,
    pub spartan_key: &'a jolt_spartan::UniformSpartanKey<F>,
    pub flat_witness: &'a [F],
    pub pcs_setup: &'a PCS::ProverSetup,
    pub pcs_verifier_setup: PCS::VerifierSetup,
    pub config: jolt_verifier::ProverConfig,
    pub backend: Arc<B>,
}

/// Proves a trace using the protocol graph.
/// Output of [`prove_from_graph`].
pub type GraphProveOutput<F, PCS> =
    (jolt_verifier::JoltProof<F, PCS>, jolt_verifier::JoltVerifyingKey<F, PCS>);

pub fn prove_from_graph<F, PCS, B>(
    input: GraphProverInput<'_, F, PCS, B>,
) -> Result<GraphProveOutput<F, PCS>, ProveError>
where
    F: Field,
    PCS: AdditivelyHomomorphic<Field = F>,
    B: ComputeBackend,
{
    let GraphProverInput {
        graph, tables, symbols, external, spartan_key, flat_witness,
        pcs_setup, pcs_verifier_setup, config, backend,
    } = input;
    let mut transcript = jolt_transcript::Blake2bTranscript::<F>::new(b"jolt-v2");

    let (witness_commitment, _) = PCS::commit(flat_witness, pcs_setup);
    transcript.append_bytes(format!("{witness_commitment:?}").as_bytes());

    let commitments = commit_polys::<PCS>(tables, pcs_setup);

    let mut cache = EvalCache::new(graph.claim_graph.claims.len());

    // S1: Spartan
    let (spartan_proof, _r_x, r_y) =
        jolt_spartan::UniformSpartanProver::prove_dense_with_challenges(
            spartan_key, flat_witness, &mut transcript,
        )?;

    let s1 = &graph.staging.stages[0];
    let r_cycle: Vec<F> = r_y.iter().rev().copied().collect();
    for &vid in &s1.vertices {
        for &cid in graph.claim_graph.vertex(vid).produced_claims() {
            let poly_id = graph.claim_graph.claim(cid).polynomial;
            cache.set(cid, eval_produced_claim(poly_id, tables, &r_cycle));
        }
    }
    cache.set_point(s1.id, r_cycle);

    // S2–S7: generic loop
    let mut stage_proofs = Vec::new();
    for stage in graph.staging.stages.iter().skip(1) {
        let sc = squeeze_challenges(&stage.pre_squeeze, symbols, &mut transcript);
        let num_vars = stage.challenge_point.num_vars.resolve(symbols).expect("num_vars");

        let mut claims = Vec::new();
        let mut witnesses: Vec<Box<dyn jolt_sumcheck::SumcheckCompute<F>>> = Vec::new();

        for &vid in &stage.vertices {
            if let Vertex::Sumcheck(sv) = graph.claim_graph.vertex(vid) {
                let claimed_sum = evaluate_input_claim(&sv.input, &cache, &sc, external);
                let witness = build_witness(
                    sv, tables, &cache, symbols, &backend,
                    &graph.claim_graph, &SymbolicPoint::Challenges(stage.id),
                    &sc, external,
                );
                claims.push(SumcheckClaim { num_vars, degree: sv.degree, claimed_sum });
                witnesses.push(witness);
            }
        }

        let captured = BatchedSumcheckProver::prove_with_handler(
            &claims, &mut witnesses, &mut transcript,
            CaptureHandler::with_capacity(num_vars),
        );
        let ep: Vec<F> = captured.challenges.iter().rev().copied().collect();

        let mut evals = Vec::new();
        for &vid in &stage.vertices {
            for &cid in graph.claim_graph.vertex(vid).produced_claims() {
                let poly_id = graph.claim_graph.claim(cid).polynomial;
                let eval = eval_produced_claim(poly_id, tables, &ep);
                cache.set(cid, eval);
                evals.push(eval);
            }
        }

        for &e in &evals {
            e.append_to_transcript(&mut transcript);
        }
        cache.set_point(stage.id, ep);

        stage_proofs.push(jolt_verifier::StageProof { round_polys: captured.proof, evals });
    }

    // Point normalization
    for v in &graph.claim_graph.vertices {
        if let Vertex::PointNormalization(pn) = v {
            let padding = cache.resolve_point(&pn.padding_source);
            let lagrange = padding.iter().fold(F::one(), |acc, r| acc * (F::one() - *r));
            for (&c, &p) in pn.consumes.iter().zip(pn.produces.iter()) {
                cache.set(p, cache.get(c) * lagrange);
            }
        }
    }

    // PCS opening
    let mut pcs_claims: Vec<ProverClaim<F>> = Vec::new();
    for v in &graph.claim_graph.vertices {
        if let Vertex::Opening(ov) = v {
            let claim = graph.claim_graph.claim(ov.consumes);
            let eval = cache.get(ov.consumes);
            let point = cache.resolve_point(&claim.point);
            if let Ok(table) = std::panic::catch_unwind(
                std::panic::AssertUnwindSafe(|| tables.get(claim.polynomial)),
            ) {
                pcs_claims.push(ProverClaim { evaluations: table.to_vec(), point, eval });
            }
        }
    }
    pcs_claims.push(ProverClaim {
        evaluations: flat_witness.to_vec(),
        point: r_y,
        eval: spartan_proof.witness_eval,
    });

    let (reduced, ()) =
        <RlcReduction as OpeningReduction<PCS>>::reduce_prover(pcs_claims, &mut transcript);
    let opening_proofs = reduced.into_iter().map(|c| {
        let poly: PCS::Polynomial = c.evaluations.into();
        PCS::open(&poly, &c.point, c.eval, pcs_setup, None, &mut transcript)
    }).collect();

    // Collect S1 virtual evals for the proof
    let spartan_evals: Vec<F> = s1.vertices.iter()
        .flat_map(|&vid| graph.claim_graph.vertex(vid).produced_claims())
        .map(|&cid| cache.get(cid))
        .collect();

    let proof = jolt_verifier::JoltProof {
        config, spartan_proof, spartan_evals, stage_proofs,
        opening_proofs, witness_commitment, commitments,
    };
    let vk = jolt_verifier::JoltVerifyingKey {
        spartan_key: spartan_key.clone(),
        pcs_setup: pcs_verifier_setup,
    };

    Ok((proof, vk))
}

fn commit_polys<PCS: AdditivelyHomomorphic>(
    tables: &PolynomialTables<PCS::Field>,
    setup: &PCS::ProverSetup,
) -> Vec<PCS::Output> {
    let c = |d: &[PCS::Field]| PCS::commit(d, setup).0;
    let mut out = vec![c(&tables.ram_inc), c(&tables.rd_inc)];
    for ra in &tables.instruction_ra { out.push(c(ra)); }
    for ra in &tables.bytecode_ra { out.push(c(ra)); }
    for ra in &tables.ram_ra { out.push(c(ra)); }
    out
}

/// Builds a sumcheck witness for a vertex.
///
/// Resolves the weighting polynomial, collects polynomial tables for the
/// formula's opening variables, evaluates challenge variables from the
/// stage's squeezed values, and delegates to `witness_builder::formula_witness`.
#[allow(clippy::too_many_arguments)]
fn build_witness<F: Field, B: ComputeBackend>(
    sv: &jolt_ir::protocol::SumcheckVertex,
    tables: &PolynomialTables<F>,
    cache: &EvalCache<F>,
    _symbols: &HashMap<Symbol, usize>,
    backend: &Arc<B>,
    graph: &jolt_ir::protocol::ClaimGraph,
    stage_point: &SymbolicPoint,
    stage_challenges: &StageChallenges<F>,
    external: &HashMap<&str, F>,
) -> Box<dyn jolt_sumcheck::SumcheckCompute<F>> {
    use jolt_ir::protocol::PublicPolynomial as PP;
    use jolt_ir::SopValue;

    let eq_point = cache.resolve_point(stage_point);
    let w_table: Vec<F> = match &sv.weighting {
        PP::Eq => jolt_poly::EqPolynomial::new(eq_point).evaluations(),
        PP::EqPlusOne => jolt_poly::EqPlusOnePolynomial::evals(&eq_point, None).1,
        PP::Lt => jolt_poly::LtPolynomial::evaluations(&eq_point),
        PP::Derived => vec![F::one(); 1 << eq_point.len()],
    };

    // Map formula opening var_ids → polynomial table slices.
    // Deduplicate: if the same poly appears in multiple bindings, reuse.
    let formula = &sv.formula;
    let mut opening_var_to_poly_idx: HashMap<u32, usize> = HashMap::new();
    let mut poly_tables: Vec<&[F]> = Vec::new();

    for binding in &formula.definition.opening_bindings {
        let claim_id = formula.opening_claims[&binding.var_id];
        let poly_id = graph.claim(claim_id).polynomial;
        if let Ok(table) = std::panic::catch_unwind(
            std::panic::AssertUnwindSafe(|| tables.get(poly_id)),
        ) {
            let idx = poly_tables.len();
            poly_tables.push(table);
            let _ = opening_var_to_poly_idx.insert(binding.var_id, idx);
        }
    }

    // Resolve challenge values for this formula.
    let challenge_labels = match &sv.input {
        InputClaim::Formula { challenge_labels, .. } => challenge_labels,
        InputClaim::Constant(_) => &[] as &[_],
    };
    let mut challenge_values: HashMap<u32, F> = HashMap::new();
    for (i, label) in challenge_labels.iter().enumerate() {
        let val = match label {
            jolt_ir::protocol::ChallengeLabel::PreSqueeze(name) => {
                stage_challenges.scalars.get(name).copied().unwrap_or(F::zero())
            }
            jolt_ir::protocol::ChallengeLabel::External(name) => {
                external.get(name).copied().unwrap_or(F::zero())
            }
        };
        let _ = challenge_values.insert(i as u32, val);
    }

    // Convert SoP terms into catalog::Term for formula_descriptor.
    // Each SoP term: coefficient × Π(opening_factors) × Π(challenge_factors)
    // Challenge factors fold into the term coefficient.
    let sop = formula.definition.expr.to_sum_of_products();
    let mut terms = Vec::new();

    for sop_term in &sop.terms {
        let mut coeff = F::from_i128(sop_term.coefficient);
        let mut factors = Vec::new();

        for factor in &sop_term.factors {
            match factor {
                SopValue::Opening(var_id) => {
                    if let Some(&idx) = opening_var_to_poly_idx.get(var_id) {
                        factors.push(idx);
                    }
                }
                SopValue::Challenge(var_id) => {
                    coeff *= challenge_values.get(var_id).copied().unwrap_or(F::one());
                }
                SopValue::Constant(c) => {
                    coeff *= F::from_i128(*c);
                }
            }
        }

        if !factors.is_empty() || coeff != F::zero() {
            terms.push(crate::evaluators::catalog::Term { coeff, factors });
        }
    }

    if terms.is_empty() || poly_tables.is_empty() {
        let n = w_table.len();
        let zero = vec![F::zero(); n];
        return witness_builder::formula_witness(
            &w_table, &[&zero],
            &[crate::evaluators::catalog::Term { coeff: F::one(), factors: vec![0] }],
            sv.degree + 1, backend,
        );
    }

    witness_builder::formula_witness(&w_table, &poly_tables, &terms, sv.degree + 1, backend)
}
