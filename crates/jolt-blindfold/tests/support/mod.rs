#![expect(
    dead_code,
    clippy::expect_used,
    reason = "shared integration-test harness is intentionally broader than each test file"
)]

use jolt_blindfold::{
    prove, BlindFoldProtocol, BlindFoldStage, BlindFoldStatement, BlindFoldWitness,
    CommittedClaimRows, FinalOpeningBinding, ProverError, RowDimensions, VerificationError,
    WitnessCoordinate,
};
use jolt_claims::r1cs::ClaimSourceTable;
use jolt_claims::{challenge, constant, derived, opening, Expr};
use jolt_crypto::{
    Bn254, Bn254G1, JoltGroup, Pedersen, PedersenSetup, VectorCommitment, VectorCommitmentOpening,
};
use jolt_field::{CanonicalBytes, Field, Fr, Ring};
use jolt_poly::{EqPolynomial, UnivariatePoly};
use jolt_r1cs::{ConstraintMatrices, R1csBuilder};
use jolt_sumcheck::{
    CommittedOutputClaims, CommittedSumcheckBuilder, CommittedSumcheckConsistency,
    CommittedSumcheckWitness, SumcheckDomainSpec, SumcheckR1csLayout, SumcheckStatement,
    SumcheckVerifier, VerifiedCommittedRound,
};
use jolt_transcript::{
    Blake2b512, Channel, ProtocolId, ProverTranscript, TranscriptError, VerifierTranscript,
};
use rand_chacha::ChaCha20Rng;
use rand_core::{RngCore, SeedableRng};

pub type F = Fr;
pub type VC = Pedersen<Bn254G1>;
pub type H = Blake2b512;
pub type TestExpr = Expr<F, Opening, Public, Challenge>;

pub const PROTOCOL: ProtocolId = ProtocolId::new::<H>("jolt-blindfold/tests");
pub const SESSION: &[u8] = b"blindfold-integration";

/// Compressed-round coefficient counts of the folded Spartan sumchecks.
const OUTER_ROUND_LEN: usize = 3;
const INNER_ROUND_LEN: usize = 2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Opening {
    Start,
    Final,
    Aux,
    Link,
    Mid,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Public {
    Offset,
    Multiplier,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Challenge {
    Scale,
    Bias,
    Mix,
}

/// One committed sumcheck stage written into a prover transcript, with the
/// commitments and challenges the verifier reads back and the prover-retained
/// openings.
#[derive(Clone, Debug)]
pub struct GeneratedStage {
    pub statement: SumcheckStatement,
    pub consistency: CommittedSumcheckConsistency<F, Bn254G1>,
    pub output_claims: CommittedOutputClaims<Bn254G1>,
    pub witness: CommittedSumcheckWitness<F>,
    pub input_claim: F,
    pub claim_outs: Vec<F>,
}

#[derive(Clone, Debug)]
pub struct DeepValues {
    pub start: F,
    pub aux: F,
    pub link: F,
    pub mid: F,
    pub final_value: F,
    pub scale: F,
    pub bias: F,
    pub mix: F,
    pub offset: F,
    pub multiplier: F,
}

#[derive(Clone, Debug)]
pub struct TestStageRelation<F, O = (), P = (), Ch = usize> {
    pub name: String,
    pub statement: SumcheckStatement,
    pub domain: SumcheckDomainSpec,
    pub input_claim: Expr<F, O, P, Ch>,
    pub output_claim: Expr<F, O, P, Ch>,
}

impl<F, O, P, Ch> TestStageRelation<F, O, P, Ch> {
    pub fn new(
        name: impl Into<String>,
        statement: SumcheckStatement,
        input_claim: Expr<F, O, P, Ch>,
        output_claim: Expr<F, O, P, Ch>,
    ) -> Self {
        Self {
            name: name.into(),
            statement,
            domain: SumcheckDomainSpec::BooleanHypercube,
            input_claim,
            output_claim,
        }
    }
}

pub fn f(value: u64) -> F {
    F::from_u64(value)
}

pub fn rng_field(rng: &mut impl RngCore) -> F {
    let mut bytes = [0u8; 32];
    rng.fill_bytes(&mut bytes);
    <F as jolt_field::CanonicalEncoding>::from_bytes_le_reduced(&bytes)
}

pub fn inverse(value: F) -> F {
    value.inverse().expect("test values are nonzero")
}

pub fn eval_poly(coefficients: &[F], point: F) -> F {
    let mut result = f(0);
    let mut power = f(1);
    for coefficient in coefficients {
        result += *coefficient * power;
        power *= point;
    }
    result
}

#[derive(Clone, Debug)]
pub struct StatisticalProjection {
    pub label: &'static str,
    pub values: Vec<u64>,
}

impl StatisticalProjection {
    pub fn new(label: &'static str, capacity: usize) -> Self {
        Self {
            label,
            values: Vec::with_capacity(capacity),
        }
    }

    pub fn push(&mut self, value: u64) {
        self.values.push(value);
    }
}

pub fn field_low_u64(value: F) -> u64 {
    let bytes = value.to_bytes_le_vec();
    u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

pub fn projection<A: CanonicalBytes>(label: &'static [u8], values: &[A]) -> u64 {
    let mut transcript = ProverTranscript::<H>::new(
        &ProtocolId::new::<H>("blindfold-statistical-projection"),
        label,
    );
    transcript.public_all(values);
    u64::from_le_bytes(transcript.challenge_bytes())
}

pub fn opening_projection(label: &'static [u8], opening: &VectorCommitmentOpening<F>) -> u64 {
    let mut values = opening.combined_vector.clone();
    values.push(opening.combined_blinding);
    projection(label, &values)
}

pub fn assert_empirical_distribution(projection: &StatisticalProjection) {
    assert!(
        projection.values.len() >= 128,
        "{} needs enough samples for empirical checks",
        projection.label
    );
    assert_high_unique_ratio(projection);
    assert_low_bit_balance(projection);
    assert_low_bucket_chi_square(projection);
    assert_lag_one_correlation(projection);
    assert_runs_around_median(projection);
}

pub fn assert_empirical_pairwise_independence(
    lhs: &StatisticalProjection,
    rhs: &StatisticalProjection,
) {
    let correlation = pearson_correlation(&lhs.values, &rhs.values);
    assert!(
        correlation.abs() < 0.25,
        "{} and {} have suspicious pairwise correlation: {correlation}",
        lhs.label,
        rhs.label
    );
}

fn assert_high_unique_ratio(projection: &StatisticalProjection) {
    let mut sorted = projection.values.clone();
    sorted.sort_unstable();
    sorted.dedup();
    let minimum_unique = projection.values.len() * 99 / 100;
    assert!(
        sorted.len() >= minimum_unique,
        "{} reused too many projected samples: {} unique out of {}",
        projection.label,
        sorted.len(),
        projection.values.len()
    );
}

fn assert_low_bit_balance(projection: &StatisticalProjection) {
    let ones = projection
        .values
        .iter()
        .map(|value| value.count_ones() as u64)
        .sum::<u64>();
    let bit_count = (projection.values.len() * u64::BITS as usize) as f64;
    let expected = bit_count / 2.0;
    let sigma = (bit_count / 4.0).sqrt();
    let z_score = ((ones as f64) - expected).abs() / sigma;
    assert!(
        z_score < 6.0,
        "{} low-bit balance failed: ones={ones}, z={z_score}",
        projection.label
    );
}

fn assert_low_bucket_chi_square(projection: &StatisticalProjection) {
    const BUCKETS: usize = 64;
    let mut buckets = [0usize; BUCKETS];
    for &value in &projection.values {
        buckets[(value & (BUCKETS as u64 - 1)) as usize] += 1;
    }
    let expected = projection.values.len() as f64 / BUCKETS as f64;
    let chi_square = buckets
        .iter()
        .map(|&count| {
            let delta = count as f64 - expected;
            delta * delta / expected
        })
        .sum::<f64>();
    assert!(
        chi_square < 155.0,
        "{} bucket chi-square too high: {chi_square}",
        projection.label
    );
}

fn assert_lag_one_correlation(projection: &StatisticalProjection) {
    let correlation = pearson_correlation(
        &projection.values[..projection.values.len() - 1],
        &projection.values[1..],
    );
    assert!(
        correlation.abs() < 0.25,
        "{} has suspicious lag-one correlation: {correlation}",
        projection.label
    );
}

fn assert_runs_around_median(projection: &StatisticalProjection) {
    let mut sorted = projection.values.clone();
    sorted.sort_unstable();
    let median = sorted[sorted.len() / 2];
    let signs = projection
        .values
        .iter()
        .map(|&value| value > median)
        .collect::<Vec<_>>();
    let high_count = signs.iter().filter(|&&sign| sign).count();
    let low_count = signs.len() - high_count;
    assert!(
        high_count > 0 && low_count > 0,
        "{} did not cross its sample median",
        projection.label
    );

    let runs = 1 + signs
        .windows(2)
        .filter(|window| window[0] != window[1])
        .count();
    let n = signs.len() as f64;
    let high = high_count as f64;
    let low = low_count as f64;
    let expected = 1.0 + 2.0 * high * low / n;
    let variance = 2.0 * high * low * (2.0 * high * low - n) / (n * n * (n - 1.0));
    let z_score = (runs as f64 - expected).abs() / variance.sqrt();
    assert!(
        z_score < 6.0,
        "{} median-runs test failed: runs={runs}, z={z_score}",
        projection.label
    );
}

fn pearson_correlation(lhs: &[u64], rhs: &[u64]) -> f64 {
    assert_eq!(lhs.len(), rhs.len());
    assert!(lhs.len() >= 2);
    let lhs_values = lhs.iter().map(|&value| value as f64).collect::<Vec<_>>();
    let rhs_values = rhs.iter().map(|&value| value as f64).collect::<Vec<_>>();
    let lhs_mean = lhs_values.iter().sum::<f64>() / lhs_values.len() as f64;
    let rhs_mean = rhs_values.iter().sum::<f64>() / rhs_values.len() as f64;
    let mut numerator = 0.0;
    let mut lhs_variance = 0.0;
    let mut rhs_variance = 0.0;
    for (&lhs_value, &rhs_value) in lhs_values.iter().zip(&rhs_values) {
        let lhs_delta = lhs_value - lhs_mean;
        let rhs_delta = rhs_value - rhs_mean;
        numerator += lhs_delta * rhs_delta;
        lhs_variance += lhs_delta * lhs_delta;
        rhs_variance += rhs_delta * rhs_delta;
    }
    let denominator = (lhs_variance * rhs_variance).sqrt();
    assert!(denominator > 0.0);
    numerator / denominator
}

pub fn pedersen_setup(capacity: usize) -> PedersenSetup<Bn254G1> {
    let generator = Bn254::g1_generator();
    let message_generators = (1..=capacity)
        .map(|i| generator.scalar_mul(&F::from_u64(i as u64)))
        .collect();
    PedersenSetup::new(message_generators, generator.scalar_mul(&f(99)))
}

pub fn coefficients_for_claim_with_rng(claim: F, degree: usize, rng: &mut impl RngCore) -> Vec<F> {
    let mut coefficients = vec![f(0); degree + 1];
    let mut nonconstant_sum = f(0);
    for coefficient in coefficients.iter_mut().skip(1) {
        *coefficient = rng_field(rng);
        nonconstant_sum += *coefficient;
    }
    coefficients[0] = (claim - nonconstant_sum) * inverse(f(2));
    coefficients
}

#[derive(Debug)]
pub struct SumcheckTestProver<R> {
    rng: R,
}

impl<R: RngCore> SumcheckTestProver<R> {
    pub fn new(rng: R) -> Self {
        Self { rng }
    }

    pub fn prove_stage(
        &mut self,
        setup: &PedersenSetup<Bn254G1>,
        transcript: &mut ProverTranscript<H>,
        statement: SumcheckStatement,
        input_claim: F,
    ) -> GeneratedStage {
        self.prove_stage_with_output_claims(setup, transcript, statement, input_claim, 0)
    }

    /// Proves a committed stage whose rounds sum to `input_claim`, followed by
    /// `output_claim_count` random output-claim rows of `degree + 1` values.
    pub fn prove_stage_with_output_claims(
        &mut self,
        setup: &PedersenSetup<Bn254G1>,
        transcript: &mut ProverTranscript<H>,
        statement: SumcheckStatement,
        input_claim: F,
        output_claim_count: usize,
    ) -> GeneratedStage {
        let blinding_rng = ChaCha20Rng::seed_from_u64(self.rng.next_u64());
        let mut builder = CommittedSumcheckBuilder::<F, VC, _>::new(setup, blinding_rng)
            .expect("setup has commitment capacity");
        let mut claim = input_claim;
        let mut challenges = Vec::with_capacity(statement.num_vars);
        let mut claim_outs = Vec::with_capacity(statement.num_vars);
        for _ in 0..statement.num_vars {
            let coefficients =
                coefficients_for_claim_with_rng(claim, statement.degree, &mut self.rng);
            let challenge = builder
                .commit_round(
                    &UnivariatePoly::new(coefficients.clone()),
                    statement.degree,
                    transcript,
                )
                .expect("round commits");
            claim = eval_poly(&coefficients, challenge);
            challenges.push(challenge);
            claim_outs.push(claim);
        }
        let output_claim_values = (0..output_claim_count * (statement.degree + 1))
            .map(|_| rng_field(&mut self.rng))
            .collect::<Vec<_>>();
        let witness = builder
            .finish(&output_claim_values, transcript)
            .expect("output claims commit");
        assert_eq!(witness.output_claim_rows.len(), output_claim_count);

        let consistency = CommittedSumcheckConsistency {
            rounds: witness
                .round_coefficients
                .iter()
                .zip(&witness.round_blindings)
                .zip(challenges)
                .map(
                    |((coefficients, blinding), challenge)| VerifiedCommittedRound {
                        commitment: VC::commit(setup, coefficients, blinding),
                        degree: statement.degree,
                        challenge,
                    },
                )
                .collect(),
        };
        let output_claims = CommittedOutputClaims {
            commitments: witness
                .output_claim_rows
                .iter()
                .zip(&witness.output_claim_blindings)
                .map(|(row, blinding)| VC::commit(setup, row, blinding))
                .collect(),
        };
        GeneratedStage {
            statement,
            consistency,
            output_claims,
            witness,
            input_claim,
            claim_outs,
        }
    }
}

pub fn blindfold_statement<O, P, Ch>(
    relations: &[TestStageRelation<F, O, P, Ch>],
    stages: &[&GeneratedStage],
    final_openings: Vec<FinalOpeningBinding<F, O, Bn254G1>>,
) -> BlindFoldStatement<F, O, Bn254G1, P, Ch>
where
    O: Clone,
    P: Clone,
    Ch: Clone,
{
    assert_eq!(
        relations.len(),
        stages.len(),
        "relations and generated stages must align"
    );
    let stages = relations
        .iter()
        .zip(stages)
        .map(|(relation, generated)| {
            BlindFoldStage::new(
                relation.name.clone(),
                relation.statement,
                relation.domain,
                generated.consistency.clone(),
                CommittedClaimRows::new(
                    Vec::new(),
                    relation.statement.degree + 1,
                    generated.output_claims.clone(),
                ),
                relation.input_claim.clone(),
                relation.output_claim.clone(),
            )
        })
        .collect();
    BlindFoldStatement::new(stages, final_openings)
}

pub fn assign_generated_stage(
    builder: &mut R1csBuilder<F>,
    layout: &SumcheckR1csLayout,
    generated: &GeneratedStage,
) {
    builder
        .assign(layout.input_claim, generated.input_claim)
        .expect("input claim assigns");
    for (round_layout, (round_coefficients, &claim_out)) in layout.rounds.iter().zip(
        generated
            .witness
            .round_coefficients
            .iter()
            .zip(&generated.claim_outs),
    ) {
        for (&variable, &coefficient) in round_layout.coefficients.iter().zip(round_coefficients) {
            builder
                .assign(variable, coefficient)
                .expect("coefficient assigns");
        }
        builder
            .assign(round_layout.claim_out, claim_out)
            .expect("claim out assigns");
    }
}

pub fn deep_stage1_input(values: &DeepValues) -> F {
    values.start * values.aux * values.scale + values.offset - values.bias
}

pub fn deep_stage2_input(values: &DeepValues) -> F {
    values.link * values.mix + values.multiplier
}

pub fn deep_stage3_input(values: &DeepValues) -> F {
    values.mid + values.start * values.bias
}

pub fn deep_values_without_links() -> DeepValues {
    DeepValues {
        start: f(6),
        aux: f(10),
        link: f(0),
        mid: f(0),
        final_value: f(0),
        scale: f(4),
        bias: f(12),
        mix: f(8),
        offset: f(18),
        multiplier: f(30),
    }
}

pub fn deep_values(
    stage1_final_claim: F,
    stage2_final_claim: F,
    stage3_final_claim: F,
) -> DeepValues {
    let mut values = deep_values_without_links();
    values.link = stage1_final_claim;
    values.mid = (stage2_final_claim - values.bias) * inverse(values.aux);
    values.final_value =
        (stage3_final_claim - values.mix * values.offset - values.link * values.mid)
            * inverse(values.aux * values.start);
    values
}

pub fn deep_claims() -> (TestExpr, TestExpr, TestExpr, TestExpr, TestExpr, TestExpr) {
    let stage1_input =
        opening(Opening::Start) * opening(Opening::Aux) * challenge(Challenge::Scale)
            + derived(Public::Offset)
            - challenge(Challenge::Bias);
    let stage1_output = opening(Opening::Link);
    let stage2_input =
        opening(Opening::Link) * challenge(Challenge::Mix) + derived(Public::Multiplier);
    let stage2_output = opening(Opening::Mid) * opening(Opening::Aux) + challenge(Challenge::Bias);
    let stage3_input = opening(Opening::Mid) + opening(Opening::Start) * challenge(Challenge::Bias);
    let stage3_output = opening(Opening::Final) * opening(Opening::Aux) * opening(Opening::Start)
        + challenge(Challenge::Mix) * derived(Public::Offset)
        + opening(Opening::Link) * opening(Opening::Mid);
    (
        stage1_input,
        stage1_output,
        stage2_input,
        stage2_output,
        stage3_input,
        stage3_output,
    )
}

pub fn build_deep_relation(
    stage1: &GeneratedStage,
    stage2: &GeneratedStage,
    stage3: &GeneratedStage,
    values: &DeepValues,
) -> Result<(), usize> {
    let (stage1_input, stage1_output, stage2_input, stage2_output, stage3_input, stage3_output) =
        deep_claims();
    let relations = vec![
        TestStageRelation::new(
            "deep-stage-1",
            stage1.statement,
            stage1_input,
            stage1_output,
        ),
        TestStageRelation::new(
            "deep-stage-2",
            stage2.statement,
            stage2_input,
            stage2_output,
        ),
        TestStageRelation::new(
            "deep-stage-3",
            stage3.statement,
            stage3_input,
            stage3_output,
        ),
    ];
    let statement = blindfold_statement(&relations, &[stage1, stage2, stage3], Vec::new());

    let mut builder = R1csBuilder::<F>::new();
    let mut sources = ClaimSourceTable::<F, Opening, Public, Challenge>::new();
    sources.insert_opening(Opening::Start, builder.alloc(values.start));
    sources.insert_opening(Opening::Aux, builder.alloc(values.aux));
    sources.insert_opening(Opening::Link, builder.alloc(values.link));
    sources.insert_opening(Opening::Mid, builder.alloc(values.mid));
    sources.insert_opening(Opening::Final, builder.alloc(values.final_value));
    sources.insert_challenge(Challenge::Scale, values.scale);
    sources.insert_challenge(Challenge::Bias, values.bias);
    sources.insert_challenge(Challenge::Mix, values.mix);
    sources.insert_public(Public::Offset, values.offset);
    sources.insert_public(Public::Multiplier, values.multiplier);

    let layout = statement
        .allocate_layout(&mut builder)
        .expect("layout allocates");
    statement
        .append(&mut builder, &layout, &mut sources)
        .expect("constraints append");
    assign_generated_stage(&mut builder, &layout.stages[0].sumcheck, stage1);
    assign_generated_stage(&mut builder, &layout.stages[1].sumcheck, stage2);
    assign_generated_stage(&mut builder, &layout.stages[2].sumcheck, stage3);

    let witness = builder.witness().expect("all witnesses assigned");
    builder.into_matrices().check_witness(&witness)
}

pub fn generated_deep_triple<R: RngCore>(
    prover: &mut SumcheckTestProver<R>,
) -> (GeneratedStage, GeneratedStage, GeneratedStage, DeepValues) {
    let setup = pedersen_setup(4);
    let statement = SumcheckStatement::new(4, 3);
    let mut values = deep_values_without_links();
    let mut transcript = ProverTranscript::<H>::new(&PROTOCOL, SESSION);
    let stage1 = prover.prove_stage(
        &setup,
        &mut transcript,
        statement,
        deep_stage1_input(&values),
    );
    values.link = *stage1
        .claim_outs
        .last()
        .expect("stage has at least one round");
    let stage2 = prover.prove_stage(
        &setup,
        &mut transcript,
        statement,
        deep_stage2_input(&values),
    );
    let stage2_final_claim = *stage2
        .claim_outs
        .last()
        .expect("stage has at least one round");
    values.mid = (stage2_final_claim - values.bias) * inverse(values.aux);
    let stage3 = prover.prove_stage(
        &setup,
        &mut transcript,
        statement,
        deep_stage3_input(&values),
    );
    let stage3_final_claim = *stage3
        .claim_outs
        .last()
        .expect("stage has at least one round");
    values = deep_values(values.link, stage2_final_claim, stage3_final_claim);
    (stage1, stage2, stage3, values)
}

/// The public description of a committed-stage protocol: everything the
/// verifier needs besides the proof bytes.
#[derive(Clone, Debug)]
pub struct StageTemplate {
    pub name: &'static str,
    pub statement: SumcheckStatement,
    pub output_claim_count: usize,
    pub opening_ids: Vec<usize>,
    pub input_claim: Expr<F, usize>,
    pub output_claim: Expr<F, usize>,
}

#[derive(Clone, Debug)]
pub struct ProtocolTemplate {
    pub stages: Vec<StageTemplate>,
    pub final_openings: Vec<FinalOpeningBinding<F, usize, Bn254G1>>,
}

impl ProtocolTemplate {
    pub fn statement(
        &self,
        committed: Vec<(
            CommittedSumcheckConsistency<F, Bn254G1>,
            CommittedOutputClaims<Bn254G1>,
        )>,
    ) -> BlindFoldStatement<F, usize, Bn254G1> {
        assert_eq!(self.stages.len(), committed.len());
        let stages = self
            .stages
            .iter()
            .zip(committed)
            .map(|(stage, (consistency, output_claims))| {
                BlindFoldStage::new(
                    stage.name,
                    stage.statement,
                    SumcheckDomainSpec::BooleanHypercube,
                    consistency,
                    CommittedClaimRows::new(
                        stage.opening_ids.clone(),
                        stage.statement.degree + 1,
                        output_claims,
                    ),
                    stage.input_claim.clone(),
                    stage.output_claim.clone(),
                )
            })
            .collect();
        BlindFoldStatement::new(stages, self.final_openings.clone())
    }

    /// Verifies a whole proof: reads every committed stage, rebuilds the
    /// BlindFold protocol from what was read, verifies BlindFold, and requires
    /// that nothing follows.
    pub fn verify(
        &self,
        setup: &PedersenSetup<Bn254G1>,
        narg: &[u8],
    ) -> Result<(), VerificationError<F>> {
        self.verify_with_session(setup, SESSION, narg)
    }

    pub fn verify_with_session(
        &self,
        setup: &PedersenSetup<Bn254G1>,
        session: &[u8],
        narg: &[u8],
    ) -> Result<(), VerificationError<F>> {
        let mut transcript = VerifierTranscript::<H>::new(&PROTOCOL, session, narg);
        let committed = self
            .stages
            .iter()
            .enumerate()
            .map(|(stage_index, stage)| {
                SumcheckVerifier::verify_committed(
                    stage.statement,
                    stage.output_claim_count,
                    &mut transcript,
                )
                .map_err(|source| VerificationError::Sumcheck {
                    stage_index,
                    source,
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let protocol = blindfold_protocol_from_statement(&self.statement(committed))?;
        protocol.verify::<VC, H>(setup, &mut transcript)?;
        Ok(transcript.finish()?)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage2Input {
    Constant,
    /// The product of two stage-1 output-claim openings, which makes the
    /// claim lowering allocate a product auxiliary.
    ProductOfStage1Openings,
}

/// Two committed stages, one final opening bound to the first stage-1
/// output-claim value, and the protocol the prover builds from them. The
/// stages are proved from `stage_seed`, so [`transcript`](Self::transcript)
/// rebuilds the prover transcript at the BlindFold boundary on demand: a
/// prover transcript cannot be copied.
#[derive(Clone)]
pub struct TwoStageFixture {
    pub setup: PedersenSetup<Bn254G1>,
    stage_seed: [u8; 32],
    stage2_input: Stage2Input,
    pub stages: Vec<GeneratedStage>,
    pub template: ProtocolTemplate,
    pub statement: BlindFoldStatement<F, usize, Bn254G1>,
    pub protocol: BlindFoldProtocol<F, Bn254G1>,
    pub eval_outputs: Vec<F>,
    pub eval_blindings: Vec<F>,
}

impl TwoStageFixture {
    /// The prover transcript after both committed stages.
    pub fn transcript(&self) -> ProverTranscript<H> {
        prove_stages(&self.setup, self.stage_seed, self.stage2_input).0
    }

    pub fn stage_witnesses(&self) -> Vec<&CommittedSumcheckWitness<F>> {
        self.stages.iter().map(|stage| &stage.witness).collect()
    }

    /// Runs the shipped BlindFold prover after the committed stages and
    /// returns the whole proof.
    pub fn prove(
        &self,
        rows: &[Vec<F>],
        blindings: &[F],
        rng: &mut impl RngCore,
    ) -> Result<Vec<u8>, ProverError<F>> {
        let mut transcript = self.transcript();
        prove::<F, VC, H, _>(
            &self.setup,
            &self.protocol,
            &mut transcript,
            BlindFoldWitness {
                rows,
                blindings,
                eval_outputs: &self.eval_outputs,
                eval_blindings: &self.eval_blindings,
            },
            rng,
        )?;
        Ok(transcript.finish())
    }

    pub fn verify(&self, narg: &[u8]) -> Result<(), VerificationError<F>> {
        self.template.verify(&self.setup, narg)
    }

    pub fn messages(&self, narg: &[u8]) -> ProofMessages {
        ProofMessages::parse(&self.protocol, self.transcript().narg().len(), narg)
            .expect("proof parses in transcript order")
    }
}

/// Proves the two committed stages from `seed` on a fresh transcript.
fn prove_stages(
    setup: &PedersenSetup<Bn254G1>,
    seed: [u8; 32],
    stage2_input: Stage2Input,
) -> (
    ProverTranscript<H>,
    GeneratedStage,
    GeneratedStage,
    Expr<F, usize>,
) {
    let mut rng = ChaCha20Rng::from_seed(seed);
    let mut transcript = ProverTranscript::<H>::new(&PROTOCOL, SESSION);
    let mut prover = SumcheckTestProver::new(&mut rng);
    let stage1 = prover.prove_stage_with_output_claims(
        setup,
        &mut transcript,
        SumcheckStatement::new(3, 3),
        f(37),
        2,
    );
    let (input2, input2_claim) = match stage2_input {
        Stage2Input::Constant => (f(89), constant(f(89))),
        Stage2Input::ProductOfStage1Openings => {
            let row = &stage1.witness.output_claim_rows[0];
            (row[0] * row[1], opening(0usize) * opening(1usize))
        }
    };
    let stage2 = prover.prove_stage_with_output_claims(
        setup,
        &mut transcript,
        SumcheckStatement::new(2, 3),
        input2,
        1,
    );
    (transcript, stage1, stage2, input2_claim)
}

pub fn two_stage_fixture<R: RngCore>(rng: &mut R, stage2_input: Stage2Input) -> TwoStageFixture {
    two_stage_fixture_with_bindings(rng, stage2_input, 1)
}

/// Like [`two_stage_fixture`], with `binding_count` (1 or 2) final-opening
/// bindings: the second opens stage 2's first output claim.
pub fn two_stage_fixture_with_bindings<R: RngCore>(
    rng: &mut R,
    stage2_input: Stage2Input,
    binding_count: usize,
) -> TwoStageFixture {
    let setup = pedersen_setup(4);
    let mut stage_seed = [0u8; 32];
    rng.fill_bytes(&mut stage_seed);
    let (_, stage1, stage2, input2_claim) = prove_stages(&setup, stage_seed, stage2_input);
    let statement1 = stage1.statement;
    let statement2 = stage2.statement;
    let input1 = f(37);
    let mut eval_outputs = vec![stage1.witness.output_claim_rows[0][0]];
    if binding_count == 2 {
        eval_outputs.push(stage2.witness.output_claim_rows[0][0]);
    }
    let eval_blindings: Vec<F> = eval_outputs.iter().map(|_| rng_field(rng)).collect();
    let eval_commitments: Vec<_> = eval_outputs
        .iter()
        .zip(&eval_blindings)
        .map(|(&output, blinding)| VC::commit(&setup, &[output], blinding))
        .collect();

    let row_len = statement1.degree + 1;
    let template = ProtocolTemplate {
        stages: vec![
            StageTemplate {
                name: "stage-1",
                statement: statement1,
                output_claim_count: 2,
                opening_ids: (0..2 * row_len).collect(),
                input_claim: constant(input1),
                output_claim: constant(*stage1.claim_outs.last().expect("stage has rounds")),
            },
            StageTemplate {
                name: "stage-2",
                statement: statement2,
                output_claim_count: 1,
                opening_ids: (100..100 + row_len).collect(),
                input_claim: input2_claim,
                output_claim: constant(*stage2.claim_outs.last().expect("stage has rounds")),
            },
        ],
        final_openings: [0usize, 100]
            .into_iter()
            .zip(&eval_commitments)
            .map(|(opening, &commitment)| {
                FinalOpeningBinding::new(vec![opening], vec![f(1)], commitment)
            })
            .collect(),
    };
    let statement = template.statement(
        [&stage1, &stage2]
            .iter()
            .map(|stage| (stage.consistency.clone(), stage.output_claims.clone()))
            .collect(),
    );
    let protocol =
        blindfold_protocol_from_statement(&statement).expect("protocol builds from stages");
    TwoStageFixture {
        setup,
        stage_seed,
        stage2_input,
        stages: vec![stage1, stage2],
        template,
        statement,
        protocol,
        eval_outputs,
        eval_blindings,
    }
}

/// Everything needed to drive a prover (harness or real) over the same
/// protocol-backed instance: the committed stages and protocol plus the
/// witness rows assembled independently of `assign_witness`.
#[derive(Clone)]
pub struct ProtocolBackedInstance {
    pub fixture: TwoStageFixture,
    pub rows: Vec<Vec<F>>,
    pub blindings: Vec<F>,
}

impl ProtocolBackedInstance {
    pub fn prove_real(&self, rng: &mut impl RngCore) -> Result<Vec<u8>, ProverError<F>> {
        self.fixture.prove(&self.rows, &self.blindings, rng)
    }
}

pub fn build_protocol_backed_instance<R: RngCore>(rng: &mut R) -> ProtocolBackedInstance {
    build_protocol_backed_instance_with_bindings(rng, 1)
}

/// [`build_protocol_backed_instance`] over
/// [`two_stage_fixture_with_bindings`].
pub fn build_protocol_backed_instance_with_bindings<R: RngCore>(
    rng: &mut R,
    binding_count: usize,
) -> ProtocolBackedInstance {
    let fixture = two_stage_fixture_with_bindings(rng, Stage2Input::Constant, binding_count);
    let (rows, blindings) = protocol_backed_witness(
        &fixture.protocol,
        &fixture.statement,
        &[&fixture.stages[0], &fixture.stages[1]],
        &fixture.eval_outputs,
        &fixture.eval_blindings,
        rng,
    );
    ProtocolBackedInstance {
        fixture,
        rows,
        blindings,
    }
}

/// A complete proof from the harness's reference BlindFold prover.
#[derive(Clone)]
pub struct BlindFoldTestProof {
    pub instance: ProtocolBackedInstance,
    pub narg: Vec<u8>,
}

pub fn prove_blindfold_protocol_pipeline<R: RngCore>(rng: &mut R) -> BlindFoldTestProof {
    let instance = build_protocol_backed_instance(rng);
    let fixture = &instance.fixture;
    let mut transcript = fixture.transcript();
    let witness = ProtocolWitness {
        rows: &instance.rows,
        blindings: &instance.blindings,
        eval_outputs: &fixture.eval_outputs,
        eval_blindings: &fixture.eval_blindings,
    };
    prove_from_protocol_witness(
        &fixture.setup,
        &fixture.protocol,
        &mut transcript,
        witness,
        rng,
    );
    BlindFoldTestProof {
        instance,
        narg: transcript.finish(),
    }
}

/// The BlindFold messages of a proof, read back in transcript order. Field
/// and commitment widths are fixed, so the offsets also locate each message.
#[derive(Clone, Debug)]
pub struct ProofMessages {
    pub auxiliary_rows: Vec<Bn254G1>,
    pub random_u: F,
    pub random_rounds: Vec<Bn254G1>,
    pub random_output_claim_rows: Vec<Bn254G1>,
    pub random_auxiliary_rows: Vec<Bn254G1>,
    pub random_error_rows: Vec<Bn254G1>,
    pub random_evals: Vec<Bn254G1>,
    pub cross_term_error_rows: Vec<Bn254G1>,
    pub folded_eval_outputs: Vec<F>,
    pub folded_eval_blindings: Vec<F>,
    pub eval_output_openings: Vec<VectorCommitmentOpening<F>>,
    pub eval_blinding_openings: Vec<VectorCommitmentOpening<F>>,
    pub outer_rounds: Vec<F>,
    pub abc: Vec<F>,
    pub error_opening: VectorCommitmentOpening<F>,
    pub inner_rounds: Vec<F>,
    pub witness_opening: VectorCommitmentOpening<F>,
}

impl ProofMessages {
    pub fn parse(
        protocol: &BlindFoldProtocol<F, Bn254G1>,
        prefix_len: usize,
        narg: &[u8],
    ) -> Result<Self, TranscriptError> {
        let dimensions = &protocol.dimensions;
        let eval_count = protocol.eval_commitments.len();
        let mut transcript = VerifierTranscript::<H>::new(&PROTOCOL, SESSION, narg);
        let _prefix = transcript.receive_bytes(prefix_len)?;
        let auxiliary_rows = transcript.receive_n(dimensions.auxiliary_rows)?;
        let random_u = transcript.receive()?;
        let random_rounds = transcript.receive_n(dimensions.coefficient_rows)?;
        let random_output_claim_rows = transcript.receive_n(dimensions.output_claim_rows)?;
        let random_auxiliary_rows = transcript.receive_n(dimensions.auxiliary_rows)?;
        let random_error_rows = transcript.receive_n(dimensions.error.row_count)?;
        let random_evals = transcript.receive_n(eval_count)?;
        let cross_term_error_rows = transcript.receive_n(dimensions.error.row_count)?;
        let folded_eval_outputs = transcript.receive_n(eval_count)?;
        let folded_eval_blindings = transcript.receive_n(eval_count)?;
        let witness_row_len = dimensions.witness.row_len;
        let mut eval_output_openings = Vec::new();
        let mut eval_blinding_openings = Vec::new();
        for coordinates in protocol
            .final_opening_witness_coordinates()
            .expect("final opening coordinates are in the witness layout")
        {
            if coordinates.evaluation.is_some() {
                eval_output_openings.push(receive_opening(witness_row_len, &mut transcript)?);
            }
            if coordinates.blinding.is_some() {
                eval_blinding_openings.push(receive_opening(witness_row_len, &mut transcript)?);
            }
        }
        let outer_rounds = transcript.receive_n(OUTER_ROUND_LEN * num_vars(dimensions.error))?;
        let abc = transcript.receive_n(3)?;
        let error_opening = receive_opening(dimensions.error.row_len, &mut transcript)?;
        let inner_rounds = transcript.receive_n(INNER_ROUND_LEN * num_vars(dimensions.witness))?;
        let witness_opening = receive_opening(witness_row_len, &mut transcript)?;
        transcript.finish()?;
        Ok(Self {
            auxiliary_rows,
            random_u,
            random_rounds,
            random_output_claim_rows,
            random_auxiliary_rows,
            random_error_rows,
            random_evals,
            cross_term_error_rows,
            folded_eval_outputs,
            folded_eval_blindings,
            eval_output_openings,
            eval_blinding_openings,
            outer_rounds,
            abc,
            error_opening,
            inner_rounds,
            witness_opening,
        })
    }
}

fn receive_opening(
    row_len: usize,
    transcript: &mut VerifierTranscript<'_, H>,
) -> Result<VectorCommitmentOpening<F>, TranscriptError> {
    Ok(VectorCommitmentOpening {
        combined_vector: transcript.receive_n(row_len)?,
        combined_blinding: transcript.receive()?,
    })
}

fn send_opening(opening: &VectorCommitmentOpening<F>, transcript: &mut ProverTranscript<H>) {
    transcript.send_all(&opening.combined_vector);
    transcript.send(&opening.combined_blinding);
}

fn num_vars(dimensions: RowDimensions) -> usize {
    log2(dimensions.row_count) + log2(dimensions.row_len)
}

pub fn blindfold_protocol_from_statement<O, P, Ch>(
    statement: &BlindFoldStatement<F, O, Bn254G1, P, Ch>,
) -> Result<BlindFoldProtocol<F, Bn254G1>, jolt_blindfold::VerificationError<F>>
where
    O: Clone + PartialEq,
    P: Clone + PartialEq,
    Ch: Clone + PartialEq,
{
    let mut builder = BlindFoldProtocol::<F, Bn254G1>::builder::<O, P, Ch>();
    for stage in &statement.stages {
        builder = builder
            .stage(stage.name.clone())
            .sumcheck(stage.statement)
            .domain(stage.domain)
            .consistency(stage.consistency.clone())
            .output_claim_rows(
                stage.output_claim_rows.opening_ids.clone(),
                stage.output_claim_rows.row_len,
                stage.output_claim_rows.commitments.clone(),
            )
            .input_claim(stage.input_claim.clone())
            .output_claim(stage.output_claim.clone())
            .finish_stage()
            .expect("test stage statement is complete");
    }
    for binding in &statement.final_openings {
        builder = builder.final_opening(
            binding.opening_ids.clone(),
            binding.coefficients.clone(),
            binding.evaluation_commitment,
        );
    }
    builder.build()
}

fn protocol_backed_witness<R: RngCore>(
    protocol: &BlindFoldProtocol<F, Bn254G1>,
    statement: &BlindFoldStatement<F, usize, Bn254G1>,
    stages: &[&GeneratedStage],
    eval_outputs: &[F],
    eval_blindings: &[F],
    rng: &mut R,
) -> (Vec<Vec<F>>, Vec<F>) {
    let mut builder = R1csBuilder::<F>::new();
    let mut sources = ClaimSourceTable::<F, usize, (), usize>::new();
    let layout = statement
        .allocate_layout(&mut builder)
        .expect("layout allocates");
    for (stage, stage_layout) in statement.stages.iter().zip(&layout.stages) {
        let variables = stage_layout
            .output_claim_rows
            .iter()
            .flat_map(|row| row.variables.iter().take(stage.output_claim_rows.row_len));
        for (opening_id, &variable) in stage.output_claim_rows.opening_ids.iter().zip(variables) {
            sources.insert_opening(*opening_id, variable);
        }
    }
    statement
        .append(&mut builder, &layout, &mut sources)
        .expect("constraints append");
    for (stage, (stage_layout, generated)) in statement
        .stages
        .iter()
        .zip(layout.stages.iter().zip(stages))
    {
        assign_generated_stage(&mut builder, &stage_layout.sumcheck, generated);
        let variables = stage_layout
            .output_claim_rows
            .iter()
            .flat_map(|row| row.variables.iter().take(stage.output_claim_rows.row_len));
        let values = generated
            .witness
            .output_claim_rows
            .iter()
            .flat_map(|row| row.iter().copied());
        for (&variable, value) in variables
            .zip(values)
            .take(stage.output_claim_rows.opening_ids.len())
        {
            builder
                .assign(variable, value)
                .expect("output claim opening assigns");
        }
    }
    for (index, final_opening) in layout.final_openings.iter().enumerate() {
        if let Some(evaluation) = final_opening.evaluation {
            builder
                .assign(evaluation, eval_outputs[index])
                .expect("final opening evaluation assigns");
        }
        if let Some(blinding) = final_opening.blinding {
            builder
                .assign(blinding, eval_blindings[index])
                .expect("final opening blinding assigns");
        }
    }
    let witness = builder.witness().expect("witness is assigned");
    assert!(builder.into_matrices().check_witness(&witness).is_ok());

    let row_len = protocol.dimensions.witness.row_len;
    let mut rows = witness[1..=protocol.dimensions.coefficient_values]
        .chunks(row_len)
        .map(<[F]>::to_vec)
        .collect::<Vec<_>>();
    assert_eq!(rows.len(), protocol.dimensions.coefficient_rows);

    for row in stages
        .iter()
        .flat_map(|stage| stage.witness.output_claim_rows.iter())
    {
        let mut row = row.clone();
        row.resize(row_len, f(0));
        rows.push(row);
    }
    assert_eq!(
        rows.len(),
        protocol.dimensions.witness_rows.output_claims.end
    );

    let output_claim_values = protocol
        .dimensions
        .output_claim_rows
        .checked_mul(row_len)
        .expect("output claim row value count fits");
    let auxiliary_values =
        &witness[1 + protocol.dimensions.coefficient_values + output_claim_values..];
    let mut auxiliary_rows = auxiliary_values
        .chunks(row_len)
        .map(|chunk| {
            let mut row = chunk.to_vec();
            row.resize(row_len, f(0));
            row
        })
        .collect::<Vec<_>>();
    auxiliary_rows.resize(protocol.dimensions.auxiliary_rows, vec![f(0); row_len]);
    rows.extend(auxiliary_rows);
    assert_eq!(rows.len(), protocol.dimensions.witness_rows.auxiliary.end);

    rows.resize(protocol.dimensions.witness.row_count, vec![f(0); row_len]);

    let mut blindings = stages
        .iter()
        .flat_map(|stage| stage.witness.round_blindings.iter().copied())
        .collect::<Vec<_>>();
    blindings.extend(
        stages
            .iter()
            .flat_map(|stage| stage.witness.output_claim_blindings.iter().copied()),
    );
    blindings.extend((0..protocol.dimensions.auxiliary_rows).map(|_| rng_field(rng)));
    blindings.resize(protocol.dimensions.witness.row_count, f(0));

    assert_eq!(rows.len(), protocol.dimensions.witness.row_count);
    assert_eq!(blindings.len(), protocol.dimensions.witness.row_count);
    (rows, blindings)
}

#[derive(Clone, Copy, Debug)]
struct ProtocolWitness<'a> {
    rows: &'a [Vec<F>],
    blindings: &'a [F],
    eval_outputs: &'a [F],
    eval_blindings: &'a [F],
}

#[derive(Clone, Debug)]
struct SumcheckTrace {
    point: Vec<F>,
}

fn prove_from_protocol_witness<R: RngCore>(
    setup: &PedersenSetup<Bn254G1>,
    protocol: &BlindFoldProtocol<F, Bn254G1>,
    transcript: &mut ProverTranscript<H>,
    witness: ProtocolWitness<'_>,
    rng: &mut R,
) {
    let auxiliary_range = protocol.dimensions.witness_rows.auxiliary.clone();
    let auxiliary_row_commitments = commit_rows(
        setup,
        &witness.rows[auxiliary_range.clone()],
        &witness.blindings[auxiliary_range],
    );
    let committed = protocol
        .committed_relaxed_instance(&auxiliary_row_commitments)
        .expect("committed relaxed instance builds");
    assert_eq!(
        committed.witness_row_commitments,
        commit_rows(setup, witness.rows, witness.blindings)
    );
    for ((commitment, &output), &blinding) in protocol
        .eval_commitments
        .iter()
        .zip(witness.eval_outputs)
        .zip(witness.eval_blindings)
    {
        assert!(VC::verify(setup, commitment, &[output], &blinding));
    }

    let random_u = rng_field(rng);
    let random_witness_rows = random_rows(
        protocol.dimensions.witness.row_count,
        protocol.dimensions.witness.row_len,
        rng,
    );
    let mut random_witness_rows = random_witness_rows;
    let mut random_witness_blindings = (0..protocol.dimensions.witness.row_count)
        .map(|_| rng_field(rng))
        .collect::<Vec<_>>();
    for row in protocol.dimensions.witness_rows.padding.clone() {
        random_witness_rows[row].fill(f(0));
        random_witness_blindings[row] = f(0);
    }
    let random_eval_outputs = (0..protocol.eval_commitments.len())
        .map(|_| rng_field(rng))
        .collect::<Vec<_>>();
    let random_eval_blindings = (0..protocol.eval_commitments.len())
        .map(|_| rng_field(rng))
        .collect::<Vec<_>>();
    let final_coordinates = protocol
        .final_opening_witness_coordinates()
        .expect("final opening coordinates are in witness layout");
    let mut dedicated_rows = Vec::new();
    for coordinates in &final_coordinates {
        if let Some(coordinate) = coordinates.evaluation {
            dedicated_rows.push(coordinate.row);
        }
        if let Some(coordinate) = coordinates.blinding {
            dedicated_rows.push(coordinate.row);
        }
    }
    dedicated_rows.sort_unstable();
    dedicated_rows.dedup();
    for row in dedicated_rows {
        random_witness_rows[row].fill(f(0));
    }
    for (index, coordinates) in final_coordinates.iter().enumerate() {
        if let Some(coordinate) = coordinates.evaluation {
            random_witness_rows[coordinate.row][coordinate.column] = random_eval_outputs[index];
        }
        if let Some(coordinate) = coordinates.blinding {
            random_witness_rows[coordinate.row][coordinate.column] = random_eval_blindings[index];
        }
    }
    let random_error_rows = error_rows_for(
        &protocol.r1cs,
        random_u,
        &flatten(&random_witness_rows),
        protocol.dimensions.error.row_len,
    );
    let random_error_blindings = (0..protocol.dimensions.error.row_count)
        .map(|_| rng_field(rng))
        .collect::<Vec<_>>();
    let coefficient_range = protocol.dimensions.witness_rows.coefficients.clone();
    let output_claim_range = protocol.dimensions.witness_rows.output_claims.clone();
    let auxiliary_range = protocol.dimensions.witness_rows.auxiliary.clone();
    let random_round_commitments = commit_rows(
        setup,
        &random_witness_rows[coefficient_range.clone()],
        &random_witness_blindings[coefficient_range],
    );
    let random_output_claim_row_commitments = commit_rows(
        setup,
        &random_witness_rows[output_claim_range.clone()],
        &random_witness_blindings[output_claim_range],
    );
    let random_auxiliary_row_commitments = commit_rows(
        setup,
        &random_witness_rows[auxiliary_range.clone()],
        &random_witness_blindings[auxiliary_range],
    );
    let random_error_row_commitments =
        commit_rows(setup, &random_error_rows, &random_error_blindings);
    let random_eval_commitments = random_eval_outputs
        .iter()
        .zip(&random_eval_blindings)
        .map(|(&output, blinding)| VC::commit(setup, &[output], blinding))
        .collect::<Vec<_>>();
    let random_instance = protocol
        .random_relaxed_instance(
            &random_round_commitments,
            &random_output_claim_row_commitments,
            &random_auxiliary_row_commitments,
            &random_error_row_commitments,
            &random_eval_commitments,
            random_u,
        )
        .expect("random relaxed instance builds");
    assert_eq!(
        random_instance.witness_row_commitments,
        commit_rows(setup, &random_witness_rows, &random_witness_blindings)
    );

    let cross_term_error_rows = cross_term_error_rows_for(
        &protocol.r1cs,
        f(1),
        &flatten(witness.rows),
        random_u,
        &flatten(&random_witness_rows),
        protocol.dimensions.error.row_len,
    );
    let cross_term_error_blindings = (0..protocol.dimensions.error.row_count)
        .map(|_| rng_field(rng))
        .collect::<Vec<_>>();
    let cross_term_error_row_commitments =
        commit_rows(setup, &cross_term_error_rows, &cross_term_error_blindings);

    transcript.send_all(&auxiliary_row_commitments);
    transcript.send(&random_u);
    transcript.send_all(&random_round_commitments);
    transcript.send_all(&random_output_claim_row_commitments);
    transcript.send_all(&random_auxiliary_row_commitments);
    transcript.send_all(&random_error_row_commitments);
    transcript.send_all(&random_eval_commitments);
    transcript.send_all(&cross_term_error_row_commitments);
    let folding_challenge: F = transcript.challenge_small();

    let folded_u = f(1) + folding_challenge * random_u;
    let folded_witness_rows = fold_rows(witness.rows, &random_witness_rows, folding_challenge);
    let folded_witness_blindings = fold_scalars(
        witness.blindings,
        &random_witness_blindings,
        folding_challenge,
    );
    let folded_error_rows = fold_error_rows(
        &zero_rows(
            protocol.dimensions.error.row_count,
            protocol.dimensions.error.row_len,
        ),
        &cross_term_error_rows,
        &random_error_rows,
        folding_challenge,
    );
    let folded_error_blindings = fold_error_scalars(
        &vec![f(0); protocol.dimensions.error.row_count],
        &cross_term_error_blindings,
        &random_error_blindings,
        folding_challenge,
    );
    let folded_eval_outputs = fold_scalars(
        witness.eval_outputs,
        &random_eval_outputs,
        folding_challenge,
    );
    let folded_eval_blindings = fold_scalars(
        witness.eval_blindings,
        &random_eval_blindings,
        folding_challenge,
    );
    transcript.send_all(&folded_eval_outputs);
    transcript.send_all(&folded_eval_blindings);
    let final_coordinates = protocol
        .final_opening_witness_coordinates()
        .expect("final opening coordinates are in witness layout");
    for (index, coordinates) in final_coordinates.iter().enumerate() {
        for (coordinate, expected) in [
            (coordinates.evaluation, folded_eval_outputs[index]),
            (coordinates.blinding, folded_eval_blindings[index]),
        ] {
            let Some(coordinate) = coordinate else {
                continue;
            };
            let (opening, opened) = open_witness_coordinate(
                &folded_witness_rows,
                &folded_witness_blindings,
                coordinate,
            );
            assert_eq!(opened, expected);
            send_opening(&opening, transcript);
        }
    }

    let outer_num_vars =
        log2(protocol.dimensions.error.row_count) + log2(protocol.dimensions.error.row_len);
    let tau: Vec<F> = transcript.challenges_small(outer_num_vars);
    let outer_trace = prove_slow_sumcheck(outer_num_vars, 3, f(0), transcript, |point| {
        outer_function(
            &protocol.r1cs,
            folded_u,
            &flatten(&folded_witness_rows),
            &folded_error_rows,
            &tau,
            point,
        )
    });

    let (az_rx, bz_rx, cz_rx) = abc_at_point(
        &protocol.r1cs,
        folded_u,
        &flatten(&folded_witness_rows),
        &outer_trace.point,
    );
    let (error_row_point, error_entry_point) = outer_trace
        .point
        .split_at(log2(protocol.dimensions.error.row_count));
    let (error_opening, _) = VC::open_committed_rows(
        &flatten(&folded_error_rows),
        &folded_error_blindings,
        protocol.dimensions.error.row_len,
        error_row_point,
        error_entry_point,
    )
    .expect("folded error rows open");

    transcript.send_all(&[az_rx, bz_rx, cz_rx]);
    send_opening(&error_opening, transcript);

    let ra: F = transcript.challenge_small();
    let rb: F = transcript.challenge_small();
    let rc: F = transcript.challenge_small();
    let inner_num_vars =
        log2(protocol.dimensions.witness.row_count) + log2(protocol.dimensions.witness.row_len);
    let row_weights = EqPolynomial::<F>::evals(&outer_trace.point, None);
    let public = protocol
        .r1cs
        .public_column_contributions(&row_weights, 0, folded_u)
        .expect("public column contributions evaluate");
    let inner_claim = ra * (az_rx - public.a) + rb * (bz_rx - public.b) + rc * (cz_rx - public.c);
    let inner_trace = prove_slow_sumcheck(inner_num_vars, 2, inner_claim, transcript, |point| {
        inner_function(
            &protocol.r1cs,
            &outer_trace.point,
            &folded_witness_rows,
            ra,
            rb,
            rc,
            point,
        )
    });
    let (witness_row_point, witness_entry_point) = inner_trace
        .point
        .split_at(log2(protocol.dimensions.witness.row_count));
    let (witness_opening, _) = VC::open_committed_rows(
        &flatten(&folded_witness_rows),
        &folded_witness_blindings,
        protocol.dimensions.witness.row_len,
        witness_row_point,
        witness_entry_point,
    )
    .expect("folded witness rows open");

    send_opening(&witness_opening, transcript);
}

fn commit_rows(setup: &PedersenSetup<Bn254G1>, rows: &[Vec<F>], blindings: &[F]) -> Vec<Bn254G1> {
    rows.iter()
        .zip(blindings)
        .map(|(row, blinding)| VC::commit(setup, row, blinding))
        .collect()
}

fn open_witness_coordinate(
    witness_rows: &[Vec<F>],
    witness_blindings: &[F],
    coordinate: WitnessCoordinate,
) -> (VectorCommitmentOpening<F>, F) {
    let row_vars = log2(witness_rows.len());
    let entry_vars = log2(witness_rows[0].len());
    VC::open_committed_rows(
        &flatten(witness_rows),
        witness_blindings,
        witness_rows[0].len(),
        &boolean_point(coordinate.row, row_vars),
        &boolean_point(coordinate.column, entry_vars),
    )
    .expect("folded witness coordinate opens")
}

fn boolean_point(index: usize, num_vars: usize) -> Vec<F> {
    (0..num_vars)
        .map(|bit| {
            let shift = num_vars - bit - 1;
            f(((index >> shift) & 1) as u64)
        })
        .collect()
}

fn zero_rows(row_count: usize, row_len: usize) -> Vec<Vec<F>> {
    vec![vec![f(0); row_len]; row_count]
}

fn random_rows<R: RngCore>(row_count: usize, row_len: usize, rng: &mut R) -> Vec<Vec<F>> {
    (0..row_count)
        .map(|_| (0..row_len).map(|_| rng_field(rng)).collect())
        .collect()
}

fn fold_rows(real: &[Vec<F>], random: &[Vec<F>], challenge: F) -> Vec<Vec<F>> {
    real.iter()
        .zip(random)
        .map(|(real_row, random_row)| {
            real_row
                .iter()
                .zip(random_row)
                .map(|(&real, &random)| real + challenge * random)
                .collect()
        })
        .collect()
}

fn fold_scalars(real: &[F], random: &[F], challenge: F) -> Vec<F> {
    real.iter()
        .zip(random)
        .map(|(&real, &random)| real + challenge * random)
        .collect()
}

fn fold_error_rows(
    real: &[Vec<F>],
    cross: &[Vec<F>],
    random: &[Vec<F>],
    challenge: F,
) -> Vec<Vec<F>> {
    let challenge_squared = challenge * challenge;
    real.iter()
        .zip(cross)
        .zip(random)
        .map(|((real_row, cross_row), random_row)| {
            real_row
                .iter()
                .zip(cross_row)
                .zip(random_row)
                .map(|((&real, &cross), &random)| {
                    real + challenge * cross + challenge_squared * random
                })
                .collect()
        })
        .collect()
}

fn fold_error_scalars(real: &[F], cross: &[F], random: &[F], challenge: F) -> Vec<F> {
    let challenge_squared = challenge * challenge;
    real.iter()
        .zip(cross)
        .zip(random)
        .map(|((&real, &cross), &random)| real + challenge * cross + challenge_squared * random)
        .collect()
}

fn error_rows_for(
    r1cs: &ConstraintMatrices<F>,
    u: F,
    witness: &[F],
    row_len: usize,
) -> Vec<Vec<F>> {
    let z = z_vector(u, witness);
    let mut errors = (0..r1cs.num_constraints)
        .map(|row_index| {
            dot(&r1cs.a[row_index], &z) * dot(&r1cs.b[row_index], &z)
                - u * dot(&r1cs.c[row_index], &z)
        })
        .collect::<Vec<_>>();
    pad_to_multiple(&mut errors, row_len);
    errors.chunks(row_len).map(<[F]>::to_vec).collect()
}

fn cross_term_error_rows_for(
    r1cs: &ConstraintMatrices<F>,
    real_u: F,
    real_witness: &[F],
    random_u: F,
    random_witness: &[F],
    row_len: usize,
) -> Vec<Vec<F>> {
    let real_z = z_vector(real_u, real_witness);
    let random_z = z_vector(random_u, random_witness);
    let mut errors = (0..r1cs.num_constraints)
        .map(|row_index| {
            dot(&r1cs.a[row_index], &real_z) * dot(&r1cs.b[row_index], &random_z)
                + dot(&r1cs.a[row_index], &random_z) * dot(&r1cs.b[row_index], &real_z)
                - real_u * dot(&r1cs.c[row_index], &random_z)
                - random_u * dot(&r1cs.c[row_index], &real_z)
        })
        .collect::<Vec<_>>();
    pad_to_multiple(&mut errors, row_len);
    errors.chunks(row_len).map(<[F]>::to_vec).collect()
}

fn pad_to_multiple(values: &mut Vec<F>, row_len: usize) {
    let remainder = values.len() % row_len;
    if remainder != 0 {
        values.resize(values.len() + row_len - remainder, f(0));
    }
}

fn z_vector(u: F, witness: &[F]) -> Vec<F> {
    let mut z = Vec::with_capacity(witness.len() + 1);
    z.push(u);
    z.extend_from_slice(witness);
    z
}

fn dot(row: &[(usize, F)], witness: &[F]) -> F {
    row.iter()
        .map(|&(column, coefficient)| coefficient * witness[column])
        .sum()
}

fn flatten(rows: &[Vec<F>]) -> Vec<F> {
    rows.iter().flat_map(|row| row.iter().copied()).collect()
}

fn abc_at_point(r1cs: &ConstraintMatrices<F>, u: F, witness: &[F], point: &[F]) -> (F, F, F) {
    let row_weights = EqPolynomial::<F>::evals(point, None);
    let z = z_vector(u, witness);
    let mut az = f(0);
    let mut bz = f(0);
    let mut cz = f(0);
    for (row_index, &row_weight) in row_weights.iter().enumerate().take(r1cs.num_constraints) {
        az += row_weight * dot(&r1cs.a[row_index], &z);
        bz += row_weight * dot(&r1cs.b[row_index], &z);
        cz += row_weight * dot(&r1cs.c[row_index], &z);
    }
    (az, bz, cz)
}

fn outer_function(
    r1cs: &ConstraintMatrices<F>,
    u: F,
    witness: &[F],
    error_rows: &[Vec<F>],
    tau: &[F],
    point: &[F],
) -> F {
    let (az, bz, cz) = abc_at_point(r1cs, u, witness, point);
    let error = mle_eval(&flatten(error_rows), point);
    EqPolynomial::<F>::mle(tau, point) * (az * bz - u * cz - error)
}

fn inner_function(
    r1cs: &ConstraintMatrices<F>,
    outer_point: &[F],
    witness_rows: &[Vec<F>],
    ra: F,
    rb: F,
    rc: F,
    point: &[F],
) -> F {
    let row_weights = EqPolynomial::<F>::evals(outer_point, None);
    let column_weights = EqPolynomial::<F>::evals(point, None);
    let l_w = r1cs
        .linear_form_bilinear_eval(
            &row_weights,
            &column_weights,
            1,
            column_weights.len(),
            [ra, rb, rc],
        )
        .expect("inner linear form dimensions match");
    l_w * mle_eval(&flatten(witness_rows), point)
}

fn mle_eval(values: &[F], point: &[F]) -> F {
    EqPolynomial::<F>::evals(point, None)
        .iter()
        .zip(values)
        .map(|(&weight, &value)| weight * value)
        .sum()
}

fn prove_slow_sumcheck(
    num_vars: usize,
    degree: usize,
    claim: F,
    transcript: &mut ProverTranscript<H>,
    eval: impl Fn(&[F]) -> F,
) -> SumcheckTrace {
    let mut running_sum = claim;
    let mut prefix = Vec::with_capacity(num_vars);

    for round in 0..num_vars {
        let remaining = num_vars - round - 1;
        let values = (0..=degree)
            .map(|point| {
                let mut sum = f(0);
                for suffix in 0..(1usize << remaining) {
                    let mut evaluation_point = prefix.clone();
                    evaluation_point.push(f(point as u64));
                    for bit in 0..remaining {
                        evaluation_point.push(f(((suffix >> bit) & 1) as u64));
                    }
                    sum += eval(&evaluation_point);
                }
                sum
            })
            .collect::<Vec<_>>();
        let coefficients = interpolate_zero_to_degree(&values);
        let round_sum = coefficients[0] + coefficients.iter().copied().sum::<F>();
        assert_eq!(round_sum, running_sum);
        let mut compressed = Vec::with_capacity(degree);
        compressed.push(coefficients[0]);
        compressed.extend_from_slice(&coefficients[2..]);
        transcript.send_all(&compressed);
        let challenge: F = transcript.challenge_small();
        running_sum = eval_poly(&coefficients, challenge);
        prefix.push(challenge);
    }

    SumcheckTrace { point: prefix }
}

fn interpolate_zero_to_degree(values: &[F]) -> Vec<F> {
    let degree = values.len() - 1;
    let mut result = vec![f(0); degree + 1];
    for (j, &value) in values.iter().enumerate() {
        let x_j = f(j as u64);
        let mut basis = vec![f(1)];
        let mut denominator = f(1);
        for m in 0..=degree {
            if m == j {
                continue;
            }
            let x_m = f(m as u64);
            basis = multiply_by_linear(&basis, -x_m, f(1));
            denominator *= x_j - x_m;
        }
        let scale = value * inverse(denominator);
        for (coefficient, basis_coefficient) in result.iter_mut().zip(basis) {
            *coefficient += scale * basis_coefficient;
        }
    }
    result
}

fn multiply_by_linear(poly: &[F], constant: F, linear: F) -> Vec<F> {
    let mut result = vec![f(0); poly.len() + 1];
    for (index, &coefficient) in poly.iter().enumerate() {
        result[index] += coefficient * constant;
        result[index + 1] += coefficient * linear;
    }
    result
}

fn log2(value: usize) -> usize {
    assert!(value.is_power_of_two());
    value.trailing_zeros() as usize
}
