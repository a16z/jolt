//! Spartan for plain R1CS with columns `[0, 1 + public_inputs.len())` public.

use jolt_crypto::Bn254;
use jolt_field::{CanonicalEncoding, Fr, One, Ring, Zero};
use jolt_hyperkzg::error::HyperKZGError;
use jolt_hyperkzg::{HyperKZGProverSetup, HyperKZGScheme, HyperKZGVerifierSetup};
use jolt_poly::{BindingOrder, EqPolynomial, Polynomial, UnivariatePoly};
use jolt_r1cs::{
    ConstraintMatrices, ConstraintMatrixEvalError, MatrixColumnContributions, SparseRow,
};
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;
use jolt_transcript::Transcript;
use rayon::prelude::*;
use thiserror::Error;

use crate::stream::{
    commit_packed, new_stream_transcript, prove_stage, verify_stage_with, Column, StageMember,
    StageMemberSpec, StreamError, WrapperProof,
};

const OUTER_DEGREE: usize = 3;
const INNER_DEGREE: usize = 2;

#[derive(Debug, Error)]
pub enum SpartanError {
    #[error("R1CS must have a nonzero power-of-two constraint count")]
    InvalidConstraintCount,
    #[error("public input range has {public} columns, but the R1CS has {total}")]
    PublicInputRange { public: usize, total: usize },
    #[error("witness has {actual} entries, expected {expected}")]
    WitnessLength { expected: usize, actual: usize },
    #[error("committed witness length must be a nonzero power of two")]
    InvalidWitnessLength,
    #[error("common row domain has {rows} rows, but the padded witness needs {minimum}")]
    InvalidSharedRowDomain { rows: usize, minimum: usize },
    #[error("common row point has {actual} coordinates, expected {expected}")]
    SharedPointDimension { expected: usize, actual: usize },
    #[error("R1CS witness fails at row {0}")]
    Unsatisfied(usize),
    #[error("malformed Spartan proof")]
    MalformedProof,
    #[error("public challenge does not fit the 128-bit outer-transcript encoding")]
    ChallengeOutOfRange,
    #[error("public challenge encoding is not canonical")]
    NonCanonicalChallenge,
    #[error("outer sumcheck final claim mismatch")]
    OuterFinalClaim,
    #[error("inner sumcheck input claim mismatch")]
    InnerInputClaim,
    #[error("inner sumcheck final claim mismatch")]
    InnerFinalClaim,
    #[error("stream: {0}")]
    Stream(#[from] StreamError),
    #[error("matrix evaluation: {0}")]
    Matrix(#[from] ConstraintMatrixEvalError),
    #[error("HyperKZG: {0}")]
    HyperKzg(#[from] HyperKZGError),
}

#[derive(Clone, Copy, Debug)]
pub struct SpartanPublicInputs<'a> {
    pub known: &'a [Fr],
    pub challenges: &'a [PublicChallenge],
}

#[derive(Clone, Copy, Debug)]
pub struct SpartanPublicInputStatement<'a> {
    pub known: &'a [Fr],
    pub challenge_decoders: &'a [ChallengeDecoder],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PublicChallenge {
    pub value: Fr,
    pub decoder: ChallengeDecoder,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChallengeDecoder {
    Challenge125,
    Scalar128,
}

/// Spartan's witness polynomial embedded in the wrapper's common row domain.
///
/// Each padded witness value is repeated across the suffix rows, so evaluating
/// at a common row point equals evaluating the short witness at its prefix.
/// The combined protocol can therefore head-align the 13-round inner sumcheck
/// and insert this column into its one packed opening.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SharedWitnessColumn {
    column: Column,
    inner_rounds: usize,
    common_row_vars: usize,
}

impl SharedWitnessColumn {
    pub fn new(witness: &[Fr], common_rows: usize) -> Result<Self, SpartanError> {
        let minimum = witness
            .len()
            .checked_next_power_of_two()
            .filter(|minimum| *minimum != 0)
            .ok_or(SpartanError::InvalidWitnessLength)?;
        if !common_rows.is_power_of_two() || common_rows < minimum {
            return Err(SpartanError::InvalidSharedRowDomain {
                rows: common_rows,
                minimum,
            });
        }
        let repetitions = common_rows / minimum;
        let mut evaluations = Vec::with_capacity(common_rows);
        for index in 0..minimum {
            let value = witness.get(index).copied().unwrap_or_else(Fr::zero);
            evaluations.extend(std::iter::repeat_n(value, repetitions));
        }
        Ok(Self {
            column: Column::Fr(evaluations),
            inner_rounds: minimum.trailing_zeros() as usize,
            common_row_vars: common_rows.trailing_zeros() as usize,
        })
    }

    pub fn inner_member(&self) -> StageMemberSpec {
        StageMemberSpec {
            rounds: self.inner_rounds,
            degree: INNER_DEGREE,
            offset: 0,
        }
    }

    pub fn inner_point<'a>(&self, common_row_point: &'a [Fr]) -> Result<&'a [Fr], SpartanError> {
        if common_row_point.len() != self.common_row_vars {
            return Err(SpartanError::SharedPointDimension {
                expected: self.common_row_vars,
                actual: common_row_point.len(),
            });
        }
        common_row_point
            .get(..self.inner_rounds)
            .ok_or(SpartanError::SharedPointDimension {
                expected: self.common_row_vars,
                actual: common_row_point.len(),
            })
    }

    pub fn into_column(self) -> Column {
        self.column
    }
}

impl ChallengeDecoder {
    pub fn pack(self, value: Fr) -> Result<[u8; 16], SpartanError> {
        let unshifted = match self {
            Self::Challenge125 => value * Fr::one().mul_pow_2(128),
            Self::Scalar128 => value,
        };
        let scalar = unshifted
            .to_u128_checked()
            .ok_or(SpartanError::ChallengeOutOfRange)?;
        let bytes = match self {
            Self::Challenge125 => scalar.to_le_bytes(),
            Self::Scalar128 => scalar.to_be_bytes(),
        };
        (self.decode(bytes)? == value)
            .then_some(bytes)
            .ok_or(SpartanError::ChallengeOutOfRange)
    }

    pub fn decode(self, bytes: [u8; 16]) -> Result<Fr, SpartanError> {
        if self == Self::Challenge125 && bytes[15] & 0xe0 != 0 {
            return Err(SpartanError::NonCanonicalChallenge);
        }
        Ok(match self {
            Self::Challenge125 => Fr::from_challenge_bytes(&bytes),
            Self::Scalar128 => Fr::from_scalar_challenge_bytes(&bytes),
        })
    }
}

pub fn prove_spartan(
    key_digest: &[u8; 32],
    r1cs: &ConstraintMatrices<Fr>,
    public: SpartanPublicInputs<'_>,
    witness: &[Fr],
    setup: &HyperKZGProverSetup<Bn254>,
) -> Result<WrapperProof, SpartanError> {
    let challenge_values: Vec<Fr> = public
        .challenges
        .iter()
        .map(|challenge| challenge.value)
        .collect();
    let public_inputs = materialize_public_inputs(public.known, &challenge_values);
    validate_dimensions(r1cs, public_inputs.len(), witness.len())?;
    let z = assignment(&public_inputs, witness);
    r1cs.check_witness(&z).map_err(SpartanError::Unsatisfied)?;

    let shared_witness = SharedWitnessColumn::new(witness, witness.len())?;
    let packed = commit_packed(&[shared_witness.into_column()], 1, setup)?;
    let mut transcript = new_stream_transcript(key_digest, &public_inputs, &packed.commitments);
    let tau = draw_point(
        r1cs.num_constraints.trailing_zeros() as usize,
        &mut transcript,
    );

    let mut outer = OuterSumcheck::new(r1cs, &z, &tau);
    let mut outer_members = [StageMember {
        prover: &mut outer,
        input_claim: Fr::zero(),
        degree: OUTER_DEGREE,
        offset: 0,
    }];
    let (outer_proof, outer_result) = prove_stage(&mut outer_members, &mut transcript)?;
    let [az, bz, cz] = outer.finals();
    let expected_outer = EqPolynomial::<Fr>::mle(&tau, &outer_result.point) * (az * bz - cz);
    if outer_result.output_claims != [expected_outer] {
        return Err(SpartanError::OuterFinalClaim);
    }
    for value in [az, bz, cz] {
        transcript.append(&value);
    }
    let matrix_weights = draw_matrix_weights(&mut transcript);
    let row_weights = EqPolynomial::<Fr>::evals(&outer_result.point, None);
    let contributions = public_contributions(r1cs, &row_weights, &public_inputs)?;
    let inner_claim = matrix_weights[0] * (az - contributions.a)
        + matrix_weights[1] * (bz - contributions.b)
        + matrix_weights[2] * (cz - contributions.c);
    let linear_form = project_witness_columns(
        r1cs,
        &row_weights,
        1 + public_inputs.len(),
        witness.len(),
        matrix_weights,
    );
    let mut inner = InnerSumcheck::new(linear_form, witness.to_vec(), inner_claim)?;
    let mut inner_members = [StageMember {
        prover: &mut inner,
        input_claim: inner_claim,
        degree: INNER_DEGREE,
        offset: 0,
    }];
    let (inner_proof, inner_result) = prove_stage(&mut inner_members, &mut transcript)?;
    let [linear_eval, witness_eval] = inner.finals();
    if inner_result.output_claims != [linear_eval * witness_eval] {
        return Err(SpartanError::InnerFinalClaim);
    }
    transcript.append(&witness_eval);
    let opening =
        HyperKZGScheme::<Bn254>::open(setup, witness, &inner_result.point, &mut transcript)
            .map_err(SpartanError::HyperKzg)?;

    Ok(WrapperProof {
        public_challenges: pack_challenges(public.challenges)?,
        commitments: packed.commitments,
        stages: vec![outer_proof, inner_proof],
        round_opening: None,
        stage_claims: vec![outer_result.output_claims, inner_result.output_claims],
        term_evaluations: Vec::new(),
        reduced_claims: vec![az, bz, cz, witness_eval],
        opening,
    })
}

pub fn verify_spartan(
    key_digest: &[u8; 32],
    r1cs: &ConstraintMatrices<Fr>,
    public: SpartanPublicInputStatement<'_>,
    proof: &WrapperProof,
    setup: &HyperKZGVerifierSetup<Bn254>,
) -> Result<(), SpartanError> {
    let challenges = unpack_challenges(&proof.public_challenges, public.challenge_decoders)?;
    let public_inputs = materialize_public_inputs(public.known, &challenges);
    let public_columns = 1 + public_inputs.len();
    if public_columns > r1cs.num_vars {
        return Err(SpartanError::PublicInputRange {
            public: public_columns,
            total: r1cs.num_vars,
        });
    }
    let witness_len = r1cs.num_vars - public_columns;
    validate_dimensions(r1cs, public_inputs.len(), witness_len)?;
    if proof.commitments.len() != 1
        || proof.stages.len() != 2
        || proof.round_opening.is_some()
        || !proof.term_evaluations.is_empty()
        || proof.stage_claims.len() != 2
        || proof.reduced_claims.len() != 4
    {
        return Err(SpartanError::MalformedProof);
    }
    let [az, bz, cz, witness_eval] = <[Fr; 4]>::try_from(proof.reduced_claims.as_slice())
        .map_err(|_| SpartanError::MalformedProof)?;
    let outer_proof = proof.stages.first().ok_or(SpartanError::MalformedProof)?;
    let inner_proof = proof.stages.get(1).ok_or(SpartanError::MalformedProof)?;
    let commitment = proof
        .commitments
        .first()
        .ok_or(SpartanError::MalformedProof)?;

    let mut transcript = new_stream_transcript(key_digest, &public_inputs, &proof.commitments);
    let tau = draw_point(
        r1cs.num_constraints.trailing_zeros() as usize,
        &mut transcript,
    );
    let outer_shape = [StageMemberSpec {
        rounds: tau.len(),
        degree: OUTER_DEGREE,
        offset: 0,
    }];
    let outer = verify_stage_with(
        outer_proof,
        &outer_shape,
        &[Fr::zero()],
        &mut transcript,
        |result| {
            checked_stage_claim(
                proof,
                0,
                EqPolynomial::<Fr>::mle(&tau, &result.point) * (az * bz - cz),
            )
        },
    )?;
    for value in [az, bz, cz] {
        transcript.append(&value);
    }
    let matrix_weights = draw_matrix_weights(&mut transcript);
    let row_weights = EqPolynomial::<Fr>::evals(&outer.point, None);
    let contributions = public_contributions(r1cs, &row_weights, &public_inputs)?;
    let inner_claim = matrix_weights[0] * (az - contributions.a)
        + matrix_weights[1] * (bz - contributions.b)
        + matrix_weights[2] * (cz - contributions.c);
    let inner_shape = [StageMemberSpec {
        rounds: witness_len.trailing_zeros() as usize,
        degree: INNER_DEGREE,
        offset: 0,
    }];
    let inner = verify_stage_with(
        inner_proof,
        &inner_shape,
        &[inner_claim],
        &mut transcript,
        |result| {
            let column_weights = EqPolynomial::<Fr>::evals(&result.point, None);
            let linear_eval = r1cs
                .linear_form_bilinear_eval(
                    &row_weights,
                    &column_weights,
                    public_columns,
                    witness_len,
                    matrix_weights,
                )
                .map_err(StreamError::Relation)?;
            checked_stage_claim(proof, 1, linear_eval * witness_eval)
        },
    )?;
    transcript.append(&witness_eval);
    HyperKZGScheme::<Bn254>::verify(
        setup,
        commitment,
        &inner.point,
        &witness_eval,
        &proof.opening,
        &mut transcript,
    )
    .map_err(SpartanError::HyperKzg)?;
    Ok(())
}

fn checked_stage_claim(
    proof: &WrapperProof,
    stage: usize,
    expected: Fr,
) -> Result<Vec<Fr>, StreamError> {
    if proof
        .stage_claims
        .get(stage)
        .and_then(|claims| claims.first())
        != Some(&expected)
        || proof.stage_claims.get(stage).map(Vec::len) != Some(1)
    {
        return Err(StreamError::StageOutputClaim);
    }
    Ok(vec![expected])
}

fn pack_challenges(challenges: &[PublicChallenge]) -> Result<Vec<[u8; 16]>, SpartanError> {
    challenges
        .iter()
        .map(|challenge| challenge.decoder.pack(challenge.value))
        .collect()
}

fn unpack_challenges(
    challenges: &[[u8; 16]],
    decoders: &[ChallengeDecoder],
) -> Result<Vec<Fr>, SpartanError> {
    if challenges.len() != decoders.len() {
        return Err(SpartanError::MalformedProof);
    }
    challenges
        .iter()
        .zip(decoders)
        .map(|(&bytes, &decoder)| decoder.decode(bytes))
        .collect()
}

fn materialize_public_inputs(known: &[Fr], challenges: &[Fr]) -> Vec<Fr> {
    let mut public_inputs = Vec::with_capacity(known.len() + challenges.len());
    public_inputs.extend_from_slice(known);
    public_inputs.extend_from_slice(challenges);
    public_inputs
}

fn validate_dimensions(
    r1cs: &ConstraintMatrices<Fr>,
    public_input_count: usize,
    witness_len: usize,
) -> Result<(), SpartanError> {
    if r1cs.num_constraints == 0 || !r1cs.num_constraints.is_power_of_two() {
        return Err(SpartanError::InvalidConstraintCount);
    }
    let public_columns = 1 + public_input_count;
    if public_columns > r1cs.num_vars {
        return Err(SpartanError::PublicInputRange {
            public: public_columns,
            total: r1cs.num_vars,
        });
    }
    let expected = r1cs.num_vars - public_columns;
    if witness_len != expected {
        return Err(SpartanError::WitnessLength {
            expected,
            actual: witness_len,
        });
    }
    if witness_len == 0 || !witness_len.is_power_of_two() {
        return Err(SpartanError::InvalidWitnessLength);
    }
    Ok(())
}

fn assignment(public_inputs: &[Fr], witness: &[Fr]) -> Vec<Fr> {
    let mut z = Vec::with_capacity(1 + public_inputs.len() + witness.len());
    z.push(Fr::one());
    z.extend_from_slice(public_inputs);
    z.extend_from_slice(witness);
    z
}

fn draw_point<T: Transcript<Challenge = Fr>>(rounds: usize, transcript: &mut T) -> Vec<Fr> {
    (0..rounds).map(|_| transcript.challenge()).collect()
}

fn draw_matrix_weights<T: Transcript<Challenge = Fr>>(transcript: &mut T) -> [Fr; 3] {
    [
        transcript.challenge(),
        transcript.challenge(),
        transcript.challenge(),
    ]
}

fn public_contributions(
    r1cs: &ConstraintMatrices<Fr>,
    row_weights: &[Fr],
    public_inputs: &[Fr],
) -> Result<MatrixColumnContributions<Fr>, ConstraintMatrixEvalError> {
    let mut values = Vec::with_capacity(1 + public_inputs.len());
    values.push(Fr::one());
    values.extend_from_slice(public_inputs);
    r1cs.column_range_contributions(row_weights, 0, &values)
}

fn project_witness_columns(
    r1cs: &ConstraintMatrices<Fr>,
    row_weights: &[Fr],
    start_column: usize,
    witness_len: usize,
    weights: [Fr; 3],
) -> Vec<Fr> {
    let mut projected = vec![Fr::zero(); witness_len];
    project_matrix(
        &mut projected,
        &r1cs.a,
        row_weights,
        start_column,
        weights[0],
    );
    project_matrix(
        &mut projected,
        &r1cs.b,
        row_weights,
        start_column,
        weights[1],
    );
    project_matrix(
        &mut projected,
        &r1cs.c,
        row_weights,
        start_column,
        weights[2],
    );
    projected
}

fn project_matrix(
    projected: &mut [Fr],
    rows: &[SparseRow<Fr>],
    row_weights: &[Fr],
    start_column: usize,
    matrix_weight: Fr,
) {
    if matrix_weight.is_zero() {
        return;
    }
    for (row, &row_weight) in rows.iter().zip(row_weights) {
        let scale = matrix_weight * row_weight;
        for &(column, coefficient) in row {
            if let Some(entry) = column
                .checked_sub(start_column)
                .and_then(|offset| projected.get_mut(offset))
            {
                *entry += scale * coefficient;
            }
        }
    }
}

fn matrix_vector_product(rows: &[SparseRow<Fr>], z: &[Fr]) -> Vec<Fr> {
    rows.par_iter()
        .map(|row| {
            row.iter()
                .map(|&(column, coefficient)| z[column] * coefficient)
                .sum()
        })
        .collect()
}

struct OuterSumcheck {
    eq_tau: Polynomial<Fr>,
    az: Polynomial<Fr>,
    bz: Polynomial<Fr>,
    cz: Polynomial<Fr>,
    rounds: usize,
}

impl OuterSumcheck {
    fn new(r1cs: &ConstraintMatrices<Fr>, z: &[Fr], tau: &[Fr]) -> Self {
        Self {
            eq_tau: Polynomial::new(EqPolynomial::<Fr>::evals(tau, None)),
            az: Polynomial::new(matrix_vector_product(&r1cs.a, z)),
            bz: Polynomial::new(matrix_vector_product(&r1cs.b, z)),
            cz: Polynomial::new(matrix_vector_product(&r1cs.c, z)),
            rounds: tau.len(),
        }
    }

    fn finals(&self) -> [Fr; 3] {
        [self.az.evals()[0], self.bz.evals()[0], self.cz.evals()[0]]
    }

    fn bind(&mut self, challenge: Fr) {
        for polynomial in [&mut self.eq_tau, &mut self.az, &mut self.bz, &mut self.cz] {
            polynomial.bind_with_order(challenge, BindingOrder::HighToLow);
        }
    }
}

impl ProveRounds<Fr> for OuterSumcheck {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        let half = self.az.len() / 2;
        let evaluations = (0..half)
            .into_par_iter()
            .map(|index| {
                let mut local = [Fr::zero(); OUTER_DEGREE + 1];
                for (x, evaluation) in local.iter_mut().enumerate() {
                    let x = Fr::from_u64(x as u64);
                    *evaluation = {
                        let eq = self.eq_tau.sumcheck_round_eval(index, x);
                        let az = self.az.sumcheck_round_eval(index, x);
                        let bz = self.bz.sumcheck_round_eval(index, x);
                        let cz = self.cz.sumcheck_round_eval(index, x);
                        eq * (az * bz - cz)
                    };
                }
                local
            })
            .reduce(
                || [Fr::zero(); OUTER_DEGREE + 1],
                |mut sum, local| {
                    for (sum, value) in sum.iter_mut().zip(local) {
                        *sum += value;
                    }
                    sum
                },
            );
        if evaluations[0] + evaluations[1] != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual: evaluations[0] + evaluations[1],
            });
        }
        Ok(UnivariatePoly::from_evals(&evaluations))
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.bind(bind);
        Ok(())
    }
}

struct InnerSumcheck {
    linear_form: Polynomial<Fr>,
    witness: Polynomial<Fr>,
    rounds: usize,
}

impl InnerSumcheck {
    fn new(linear_form: Vec<Fr>, witness: Vec<Fr>, claim: Fr) -> Result<Self, SpartanError> {
        let actual: Fr = linear_form
            .iter()
            .zip(&witness)
            .map(|(&left, &right)| left * right)
            .sum();
        if actual != claim {
            return Err(SpartanError::InnerInputClaim);
        }
        let rounds = witness.len().trailing_zeros() as usize;
        Ok(Self {
            linear_form: Polynomial::new(linear_form),
            witness: Polynomial::new(witness),
            rounds,
        })
    }

    fn finals(&self) -> [Fr; 2] {
        [self.linear_form.evals()[0], self.witness.evals()[0]]
    }

    fn bind(&mut self, challenge: Fr) {
        self.linear_form
            .bind_with_order(challenge, BindingOrder::HighToLow);
        self.witness
            .bind_with_order(challenge, BindingOrder::HighToLow);
    }
}

impl ProveRounds<Fr> for InnerSumcheck {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        let half = self.witness.len() / 2;
        let evaluations = (0..half)
            .into_par_iter()
            .map(|index| {
                let mut local = [Fr::zero(); INNER_DEGREE + 1];
                for (x, evaluation) in local.iter_mut().enumerate() {
                    let x = Fr::from_u64(x as u64);
                    *evaluation = self.linear_form.sumcheck_round_eval(index, x)
                        * self.witness.sumcheck_round_eval(index, x);
                }
                local
            })
            .reduce(
                || [Fr::zero(); INNER_DEGREE + 1],
                |mut sum, local| {
                    for (sum, value) in sum.iter_mut().zip(local) {
                        *sum += value;
                    }
                    sum
                },
            );
        if evaluations[0] + evaluations[1] != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual: evaluations[0] + evaluations[1],
            });
        }
        Ok(UnivariatePoly::from_evals(&evaluations))
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.bind(bind);
        Ok(())
    }
}
