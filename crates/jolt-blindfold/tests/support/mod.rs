#![expect(
    dead_code,
    clippy::expect_used,
    reason = "shared integration-test harness is intentionally broader than each test file"
)]

use jolt_blindfold::{
    BlindFoldProof, BlindFoldProtocol, BlindFoldStage, BlindFoldStatement, CommittedClaimRows,
    FinalOpeningBinding, WitnessCoordinate,
};
use jolt_claims::{challenge, constant, derived, opening, Expr};
use jolt_crypto::{
    Bn254, Bn254G1, JoltGroup, Pedersen, PedersenSetup, VectorCommitment, VectorCommitmentOpening,
};
use jolt_field::{FixedBytes, Fr, FromPrimitiveInt, Invertible};
use jolt_poly::{CompressedPoly, EqPolynomial};
use jolt_r1cs::{ClaimSourceTable, ConstraintMatrices, R1csBuilder};
use jolt_sumcheck::{
    CommittedOutputClaims, CommittedRound, CommittedRoundWitness, CommittedSumcheckConsistency,
    CommittedSumcheckProof, CompressedSumcheckProof, RoundMessage, SumcheckDomainSpec,
    SumcheckR1csLayout, SumcheckStatement, SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::{AppendToTranscript, Blake2bTranscript, Label, Transcript};
use rand_core::RngCore;

pub type F = Fr;
pub type VC = Pedersen<Bn254G1>;
pub type TestExpr = Expr<F, Opening, Public, Challenge>;

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

#[derive(Clone, Debug)]
pub struct GeneratedStage {
    pub statement: SumcheckStatement,
    pub proof: CommittedSumcheckProof<Bn254G1>,
    pub coefficients: Vec<Vec<F>>,
    pub blindings: Vec<F>,
    pub output_claim_rows: Vec<Vec<F>>,
    pub output_claim_blindings: Vec<F>,
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
    F::from_bytes_array(&bytes)
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
    let bytes = value.to_bytes_array();
    u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

pub fn transcript_projection<A: AppendToTranscript>(label: &'static [u8], value: &A) -> u64 {
    let mut transcript = Blake2bTranscript::<F>::new(b"blindfold-statistical-projection");
    transcript.append(&Label(label));
    value.append_to_transcript(&mut transcript);
    field_low_u64(transcript.challenge())
}

pub fn field_slice_projection(label: &'static [u8], values: &[F]) -> u64 {
    let mut transcript = Blake2bTranscript::<F>::new(b"blindfold-statistical-projection");
    transcript.append_values(label, values);
    field_low_u64(transcript.challenge())
}

pub fn compressed_sumcheck_projection(
    label: &'static [u8],
    proof: &CompressedSumcheckProof<F>,
) -> u64 {
    let mut values = Vec::new();
    for round in &proof.round_polynomials {
        values.extend_from_slice(round.coeffs_except_linear_term());
    }
    field_slice_projection(label, &values)
}

pub fn opening_projection(
    label: &'static [u8],
    opening: &jolt_crypto::VectorCommitmentOpening<F>,
) -> u64 {
    let mut values = opening.combined_vector.clone();
    values.push(opening.combined_blinding);
    field_slice_projection(label, &values)
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

pub fn commit_round_with_blinding(
    setup: &PedersenSetup<Bn254G1>,
    coefficients: Vec<F>,
    blinding: F,
) -> CommittedRound<Bn254G1> {
    CommittedRoundWitness {
        coefficients,
        blinding,
    }
    .commit::<VC>(setup)
    .expect("round witness commits")
}

pub fn commit_round(
    setup: &PedersenSetup<Bn254G1>,
    coefficients: Vec<F>,
    round: usize,
) -> CommittedRound<Bn254G1> {
    commit_round_with_blinding(setup, coefficients, f(round as u64 + 17))
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
        transcript: &mut Blake2bTranscript<F>,
        statement: SumcheckStatement,
        input_claim: F,
    ) -> GeneratedStage {
        self.prove_stage_with_output_claims(setup, transcript, statement, input_claim, 0)
    }

    pub fn prove_stage_with_output_claims(
        &mut self,
        setup: &PedersenSetup<Bn254G1>,
        transcript: &mut Blake2bTranscript<F>,
        statement: SumcheckStatement,
        input_claim: F,
        output_claim_count: usize,
    ) -> GeneratedStage {
        let mut claim = input_claim;
        let mut rounds = Vec::with_capacity(statement.num_vars);
        let mut coefficients = Vec::with_capacity(statement.num_vars);
        let mut blindings = Vec::with_capacity(statement.num_vars);
        let mut claim_outs = Vec::with_capacity(statement.num_vars);

        for _ in 0..statement.num_vars {
            let round_coefficients =
                coefficients_for_claim_with_rng(claim, statement.degree, &mut self.rng);
            let blinding = rng_field(&mut self.rng);
            let round = commit_round_with_blinding(setup, round_coefficients.clone(), blinding);
            round.append_to_transcript(transcript);
            let challenge = transcript.challenge();
            claim = eval_poly(&round_coefficients, challenge);

            rounds.push(round);
            coefficients.push(round_coefficients);
            blindings.push(blinding);
            claim_outs.push(claim);
        }
        let mut output_claim_rows = Vec::with_capacity(output_claim_count);
        let mut output_claim_blindings = Vec::with_capacity(output_claim_count);
        let mut output_commitments = Vec::with_capacity(output_claim_count);
        for _ in 0..output_claim_count {
            let row = (0..=statement.degree)
                .map(|_| rng_field(&mut self.rng))
                .collect::<Vec<_>>();
            let blinding = rng_field(&mut self.rng);
            output_commitments.push(VC::commit(setup, &row, &blinding));
            output_claim_rows.push(row);
            output_claim_blindings.push(blinding);
        }
        let output_claims = CommittedOutputClaims {
            commitments: output_commitments,
        };
        output_claims.append_to_transcript(transcript);

        GeneratedStage {
            statement,
            proof: CommittedSumcheckProof {
                rounds,
                output_claims,
            },
            coefficients,
            blindings,
            output_claim_rows,
            output_claim_blindings,
            input_claim,
            claim_outs,
        }
    }

    pub fn prove_stage_with_fresh_transcript(
        &mut self,
        setup: &PedersenSetup<Bn254G1>,
        transcript_label: &'static [u8],
        statement: SumcheckStatement,
        input_claim: F,
    ) -> GeneratedStage {
        let mut transcript = Blake2bTranscript::<F>::new(transcript_label);
        self.prove_stage(setup, &mut transcript, statement, input_claim)
    }
}

pub fn generate_zero_stage(setup: &PedersenSetup<Bn254G1>, num_vars: usize) -> GeneratedStage {
    let rounds = (0..num_vars)
        .map(|round| commit_round(setup, vec![f(0)], round))
        .collect();
    GeneratedStage {
        statement: SumcheckStatement::new(num_vars, 1),
        proof: CommittedSumcheckProof {
            rounds,
            output_claims: CommittedOutputClaims::default(),
        },
        coefficients: vec![vec![f(0)]; num_vars],
        blindings: (0..num_vars).map(|round| f(round as u64 + 17)).collect(),
        output_claim_rows: Vec::new(),
        output_claim_blindings: Vec::new(),
        input_claim: f(0),
        claim_outs: vec![f(0); num_vars],
    }
}

pub fn stage_consistency(
    statement: SumcheckStatement,
    proof: &CommittedSumcheckProof<Bn254G1>,
) -> CommittedSumcheckConsistency<F, Bn254G1> {
    let mut transcript = Blake2bTranscript::<F>::new(b"blindfold-r1cs-e2e");
    proof
        .verify_committed_consistency(statement, &mut transcript)
        .expect("committed proof transcript verifies")
}

pub fn stage_consistency_for_transcript(
    stages: &[&GeneratedStage],
) -> Vec<CommittedSumcheckConsistency<F, Bn254G1>> {
    stage_consistency_for_transcript_label(b"blindfold-r1cs-e2e", stages)
}

pub fn stage_consistency_for_transcript_label(
    transcript_label: &'static [u8],
    stages: &[&GeneratedStage],
) -> Vec<CommittedSumcheckConsistency<F, Bn254G1>> {
    let mut transcript = Blake2bTranscript::<F>::new(transcript_label);
    stages
        .iter()
        .map(|stage| {
            stage
                .proof
                .verify_committed_consistency(stage.statement, &mut transcript)
                .expect("committed proof transcript verifies")
        })
        .collect()
}

pub fn blindfold_statement_for_transcript_label<O, P, Ch>(
    transcript_label: &'static [u8],
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
    let mut transcript = Blake2bTranscript::<F>::new(transcript_label);
    let stages = relations
        .iter()
        .zip(stages)
        .map(|(relation, generated)| {
            let consistency = generated
                .proof
                .verify_committed_consistency(generated.statement, &mut transcript)
                .expect("committed proof transcript verifies");
            BlindFoldStage::new(
                relation.name.clone(),
                relation.statement,
                relation.domain,
                consistency,
                CommittedClaimRows::new(
                    Vec::new(),
                    relation.statement.degree + 1,
                    generated.proof.output_claims.clone(),
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
    for (round_layout, (round_coefficients, &claim_out)) in layout
        .rounds
        .iter()
        .zip(generated.coefficients.iter().zip(&generated.claim_outs))
    {
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
    let statement = blindfold_statement_for_transcript_label(
        b"blindfold-r1cs-e2e",
        &relations,
        &[stage1, stage2, stage3],
        Vec::new(),
    );

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
    let mut transcript = Blake2bTranscript::<F>::new(b"blindfold-r1cs-e2e");
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

#[derive(Clone, Debug)]
pub struct BlindFoldTestProof {
    pub protocol: BlindFoldProtocol<F, Bn254G1>,
    pub proof: BlindFoldProof<F, Bn254G1>,
    pub setup: PedersenSetup<Bn254G1>,
}

#[derive(Clone, Debug)]
struct SumcheckTrace {
    proof: CompressedSumcheckProof<F>,
    point: Vec<F>,
}

pub fn prove_blindfold_protocol_pipeline<R: RngCore>(rng: &mut R) -> BlindFoldTestProof {
    let setup = pedersen_setup(4);
    let transcript_label = b"protocol-backed-blindfold-proof";
    let statement1 = SumcheckStatement::new(3, 3);
    let statement2 = SumcheckStatement::new(2, 3);
    let input1 = f(37);
    let input2 = f(89);

    let (stage1, stage2) = {
        let mut prover = SumcheckTestProver::new(&mut *rng);
        let mut transcript = Blake2bTranscript::<F>::new(transcript_label);
        let stage1 =
            prover.prove_stage_with_output_claims(&setup, &mut transcript, statement1, input1, 2);
        let stage2 =
            prover.prove_stage_with_output_claims(&setup, &mut transcript, statement2, input2, 1);
        (stage1, stage2)
    };
    let stage1_output = *stage1
        .claim_outs
        .last()
        .expect("stage has at least one round");
    let stage2_output = *stage2
        .claim_outs
        .last()
        .expect("stage has at least one round");
    let real_eval_outputs = vec![stage1.output_claim_rows[0][0]];
    let real_eval_blindings = vec![rng_field(rng)];
    let eval_commitments = real_eval_outputs
        .iter()
        .zip(&real_eval_blindings)
        .map(|(&output, blinding)| VC::commit(&setup, &[output], blinding))
        .collect::<Vec<_>>();

    let mut transcript = Blake2bTranscript::<F>::new(transcript_label);
    let stage1_consistency = stage1
        .proof
        .verify_committed_consistency(stage1.statement, &mut transcript)
        .expect("stage 1 committed proof transcript verifies");
    let stage2_consistency = stage2
        .proof
        .verify_committed_consistency(stage2.statement, &mut transcript)
        .expect("stage 2 committed proof transcript verifies");
    let stages = vec![
        BlindFoldStage::new(
            "protocol-backed-stage-1",
            statement1,
            SumcheckDomainSpec::BooleanHypercube,
            stage1_consistency,
            CommittedClaimRows::new(
                (0..stage1.proof.output_claims.commitments.len() * (statement1.degree + 1))
                    .collect(),
                statement1.degree + 1,
                stage1.proof.output_claims.clone(),
            ),
            constant(input1),
            constant(stage1_output),
        ),
        BlindFoldStage::new(
            "protocol-backed-stage-2",
            statement2,
            SumcheckDomainSpec::BooleanHypercube,
            stage2_consistency,
            CommittedClaimRows::new(
                (100..100 + stage2.proof.output_claims.commitments.len() * (statement2.degree + 1))
                    .collect(),
                statement2.degree + 1,
                stage2.proof.output_claims.clone(),
            ),
            constant(input2),
            constant(stage2_output),
        ),
    ];
    let statement = BlindFoldStatement::new(
        stages,
        vec![FinalOpeningBinding::new(
            vec![0usize],
            vec![f(1)],
            eval_commitments[0],
        )],
    );
    let protocol = blindfold_protocol_from_statement(&statement)
        .expect("protocol builds from committed statement");
    let mut transcript = Blake2bTranscript::<F>::new(transcript_label);
    append_protocol_transcript_prefix(&protocol, &mut transcript);
    let (real_witness_rows, real_witness_blindings) = protocol_backed_witness(
        &protocol,
        &statement,
        &[&stage1, &stage2],
        &real_eval_outputs,
        &real_eval_blindings,
        rng,
    );
    let witness = ProtocolWitness {
        rows: &real_witness_rows,
        blindings: &real_witness_blindings,
        eval_outputs: &real_eval_outputs,
        eval_blindings: &real_eval_blindings,
    };
    let proof = prove_from_protocol_witness(&setup, &protocol, &mut transcript, witness, rng);

    BlindFoldTestProof {
        protocol,
        proof,
        setup,
    }
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

pub fn append_protocol_transcript_prefix(
    protocol: &BlindFoldProtocol<F, Bn254G1>,
    transcript: &mut Blake2bTranscript<F>,
) {
    for (stage, output_claims) in protocol
        .sumcheck_consistency
        .iter()
        .zip(&protocol.committed_output_claims)
    {
        for round in &stage.rounds {
            CommittedRound {
                commitment: round.commitment,
                degree: round.degree,
            }
            .append_to_transcript(transcript);
            let _ = transcript.challenge();
        }
        output_claims.append_to_transcript(transcript);
    }
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
        .flat_map(|stage| stage.output_claim_rows.iter())
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
        .flat_map(|stage| stage.blindings.iter().copied())
        .collect::<Vec<_>>();
    blindings.extend(
        stages
            .iter()
            .flat_map(|stage| stage.output_claim_blindings.iter().copied()),
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

fn prove_from_protocol_witness<R: RngCore>(
    setup: &PedersenSetup<Bn254G1>,
    protocol: &BlindFoldProtocol<F, Bn254G1>,
    transcript: &mut Blake2bTranscript<F>,
    witness: ProtocolWitness<'_>,
    rng: &mut R,
) -> BlindFoldProof<F, Bn254G1> {
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

    append_relaxed_instance_from_parts(
        transcript,
        RelaxedInstanceLabels {
            u: b"bf_committed_u",
            witness: b"bf_committed_w",
            error: b"bf_committed_e",
            eval: b"bf_committed_eval",
        },
        committed.u,
        &committed.witness_row_commitments,
        &committed.error_row_commitments,
        &committed.eval_commitments,
    );
    append_relaxed_instance_from_parts(
        transcript,
        RelaxedInstanceLabels {
            u: b"bf_random_u",
            witness: b"bf_random_w",
            error: b"bf_random_e",
            eval: b"bf_random_eval",
        },
        random_u,
        &random_instance.witness_row_commitments,
        &random_instance.error_row_commitments,
        &random_instance.eval_commitments,
    );
    transcript.append_values(b"bf_cross_e", &cross_term_error_row_commitments);
    let folding_challenge = transcript.challenge();

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
    let final_coordinates = protocol
        .final_opening_witness_coordinates()
        .expect("final opening coordinates are in witness layout");
    let mut folded_eval_output_openings = Vec::new();
    let mut folded_eval_blinding_openings = Vec::new();
    for (index, coordinates) in final_coordinates.iter().enumerate() {
        if let Some(coordinate) = coordinates.evaluation {
            let (opening, opened) = open_witness_coordinate(
                &folded_witness_rows,
                &folded_witness_blindings,
                coordinate,
            );
            assert_eq!(opened, folded_eval_outputs[index]);
            folded_eval_output_openings.push(opening);
        }
        if let Some(coordinate) = coordinates.blinding {
            let (opening, opened) = open_witness_coordinate(
                &folded_witness_rows,
                &folded_witness_blindings,
                coordinate,
            );
            assert_eq!(opened, folded_eval_blindings[index]);
            folded_eval_blinding_openings.push(opening);
        }
    }
    for opening in &folded_eval_output_openings {
        append_vector_opening(
            transcript,
            b"bf_eval_out_open",
            b"bf_eval_out_blind",
            opening,
        );
    }
    for opening in &folded_eval_blinding_openings {
        append_vector_opening(
            transcript,
            b"bf_eval_blind_open",
            b"bf_eval_blind_bl",
            opening,
        );
    }

    transcript.append(&Label(b"bf_spartan"));
    let outer_num_vars =
        log2(protocol.dimensions.error.row_count) + log2(protocol.dimensions.error.row_len);
    let tau = transcript.challenge_vector(outer_num_vars);
    let outer_trace = prove_slow_sumcheck(
        outer_num_vars,
        3,
        f(0),
        SUMCHECK_ROUND_TRANSCRIPT_LABEL,
        transcript,
        |point| {
            outer_function(
                &protocol.r1cs,
                folded_u,
                &flatten(&folded_witness_rows),
                &folded_error_rows,
                &tau,
                point,
            )
        },
    );

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

    transcript.append_values(b"bf_az_bz_cz", &[az_rx, bz_rx, cz_rx]);
    append_vector_opening(
        transcript,
        b"bf_error_opening",
        b"bf_error_blind",
        &error_opening,
    );

    let ra = transcript.challenge();
    let rb = transcript.challenge();
    let rc = transcript.challenge();
    let inner_num_vars =
        log2(protocol.dimensions.witness.row_count) + log2(protocol.dimensions.witness.row_len);
    let row_weights = EqPolynomial::<F>::evals(&outer_trace.point, None);
    let public = protocol
        .r1cs
        .public_column_contributions(&row_weights, 0, folded_u)
        .expect("public column contributions evaluate");
    let inner_claim = ra * (az_rx - public.a) + rb * (bz_rx - public.b) + rc * (cz_rx - public.c);
    let inner_trace = prove_slow_sumcheck(
        inner_num_vars,
        2,
        inner_claim,
        b"inner_sumcheck_poly",
        transcript,
        |point| {
            inner_function(
                &protocol.r1cs,
                &outer_trace.point,
                &folded_witness_rows,
                ra,
                rb,
                rc,
                point,
            )
        },
    );
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

    BlindFoldProof {
        auxiliary_row_commitments,
        random_round_commitments,
        random_output_claim_row_commitments,
        random_auxiliary_row_commitments,
        random_error_row_commitments,
        random_eval_commitments,
        random_u,
        cross_term_error_row_commitments,
        outer_sumcheck: outer_trace.proof,
        az_rx,
        bz_rx,
        cz_rx,
        inner_sumcheck: inner_trace.proof,
        witness_opening,
        error_opening,
        folded_eval_outputs,
        folded_eval_blindings,
        folded_eval_output_openings,
        folded_eval_blinding_openings,
    }
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

#[derive(Clone, Copy, Debug)]
struct RelaxedInstanceLabels {
    u: &'static [u8],
    witness: &'static [u8],
    error: &'static [u8],
    eval: &'static [u8],
}

fn append_relaxed_instance_from_parts(
    transcript: &mut Blake2bTranscript<F>,
    labels: RelaxedInstanceLabels,
    u: F,
    witness_commitments: &[Bn254G1],
    error_commitments: &[Bn254G1],
    eval_commitments: &[Bn254G1],
) {
    transcript.append(&Label(labels.u));
    u.append_to_transcript(transcript);
    transcript.append_values(labels.witness, witness_commitments);
    transcript.append_values(labels.error, error_commitments);
    transcript.append_values(labels.eval, eval_commitments);
}

fn append_vector_opening(
    transcript: &mut Blake2bTranscript<F>,
    row_label: &'static [u8],
    blinding_label: &'static [u8],
    opening: &jolt_crypto::VectorCommitmentOpening<F>,
) {
    transcript.append_values(row_label, &opening.combined_vector);
    transcript.append(&Label(blinding_label));
    opening.combined_blinding.append_to_transcript(transcript);
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
    label: &'static [u8],
    transcript: &mut Blake2bTranscript<F>,
    eval: impl Fn(&[F]) -> F,
) -> SumcheckTrace {
    let mut running_sum = claim;
    let mut prefix = Vec::with_capacity(num_vars);
    let mut rounds = Vec::with_capacity(num_vars);

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
        transcript.append_values(label, &compressed);
        let challenge = transcript.challenge();
        running_sum = eval_poly(&coefficients, challenge);
        prefix.push(challenge);
        rounds.push(CompressedPoly::new(compressed));
    }

    SumcheckTrace {
        proof: CompressedSumcheckProof {
            round_polynomials: rounds,
        },
        point: prefix,
    }
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
