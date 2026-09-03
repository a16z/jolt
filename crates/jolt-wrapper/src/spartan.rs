//! Spartan for plain R1CS with columns `[0, 1 + public_inputs.len())` public.

use jolt_crypto::Bn254;
use jolt_field::{Fr, One, Ring, Zero};
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
    #[error("R1CS witness fails at row {0}")]
    Unsatisfied(usize),
    #[error("malformed Spartan proof")]
    MalformedProof,
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

pub fn prove_spartan(
    key_digest: &[u8; 32],
    r1cs: &ConstraintMatrices<Fr>,
    public_inputs: &[Fr],
    witness: &[Fr],
    setup: &HyperKZGProverSetup<Bn254>,
) -> Result<WrapperProof, SpartanError> {
    validate_dimensions(r1cs, public_inputs.len(), witness.len())?;
    let z = assignment(public_inputs, witness);
    r1cs.check_witness(&z).map_err(SpartanError::Unsatisfied)?;

    let packed = commit_packed(&[Column::Fr(witness.to_vec())], 1, setup)?;
    let mut transcript = new_stream_transcript(key_digest, public_inputs, &packed.commitments);
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
    let public = public_contributions(r1cs, &row_weights, public_inputs)?;
    let inner_claim = matrix_weights[0] * (az - public.a)
        + matrix_weights[1] * (bz - public.b)
        + matrix_weights[2] * (cz - public.c);
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
        commitments: packed.commitments,
        stages: vec![outer_proof, inner_proof],
        stage_claims: vec![outer_result.output_claims, inner_result.output_claims],
        reduced_claims: vec![az, bz, cz, witness_eval],
        opening,
    })
}

pub fn verify_spartan(
    key_digest: &[u8; 32],
    r1cs: &ConstraintMatrices<Fr>,
    public_inputs: &[Fr],
    proof: &WrapperProof,
    setup: &HyperKZGVerifierSetup<Bn254>,
) -> Result<(), SpartanError> {
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

    let mut transcript = new_stream_transcript(key_digest, public_inputs, &proof.commitments);
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
    let public = public_contributions(r1cs, &row_weights, public_inputs)?;
    let inner_claim = matrix_weights[0] * (az - public.a)
        + matrix_weights[1] * (bz - public.b)
        + matrix_weights[2] * (cz - public.c);
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
