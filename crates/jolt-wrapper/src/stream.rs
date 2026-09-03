//! Shared clear-sumcheck stages and the final packed HyperKZG opening.

use jolt_crypto::ec::bn254::bit_columns::g1_bit_columns_msm;
use jolt_crypto::Bn254;
use jolt_field::{Field, Fr, One, Ring, Zero};
use jolt_hyperkzg::{HyperKZGProverSetup, HyperKZGScheme};
use jolt_openings::CommitmentScheme;
use jolt_poly::{BindingOrder, EqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::batch::{BatchMember, BatchPrelude};
use jolt_sumcheck::prover::{prove_batch, ProveRounds, SequentialRounds};
use jolt_sumcheck::recorder::{ClearSumcheckRecorder, SumcheckRecorder};
use jolt_sumcheck::{
    BooleanHypercube, ClearProof, SumcheckClaim, SumcheckError, OPENING_CLAIM_TRANSCRIPT_LABEL,
    SUMCHECK_ROUND_TRANSCRIPT_LABEL,
};
use jolt_transcript::Transcript;
use rayon::prelude::*;

mod types;
pub use types::*;
mod protocol;
pub use protocol::{new_stream_transcript, prove_stream, verify_stream};

const STREAM_LABEL: &[u8] = b"jolt-wrapper-v1";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Column {
    /// Values checked as 0/1 before commitment; the proved relation must enforce booleanity.
    Bits(Vec<u8>),
    /// Small-scalar commitment input; the proved relation must enforce its range requirement.
    U16(Vec<u16>),
    Fr(Vec<Fr>),
}

impl Column {
    pub fn len(&self) -> usize {
        match self {
            Self::Bits(values) => values.len(),
            Self::U16(values) => values.len(),
            Self::Fr(values) => values.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn value(&self, row: usize) -> Fr {
        match self {
            Self::Bits(values) => Fr::from_u64(u64::from(values[row])),
            Self::U16(values) => Fr::from_u64(u64::from(values[row])),
            Self::Fr(values) => values[row],
        }
    }
}

/// `ceil(columns / k)` polynomials with `packed[row * k + slot] = column[g*k+slot][row]`.
/// Thus the row variables precede the `log2(k)` low column-slot variables in an opening point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PackingLayout {
    pub rows: usize,
    pub column_count: usize,
    pub k: usize,
    pub group_count: usize,
    pub padded_group_count: usize,
    pub padded_column_count: usize,
}

impl PackingLayout {
    pub fn new(rows: usize, column_count: usize, k: usize) -> Result<Self, StreamError> {
        if column_count == 0 {
            return Err(StreamError::NoColumns);
        }
        if !k.is_power_of_two() {
            return Err(StreamError::InvalidPacking(k));
        }
        if rows == 0 || !rows.is_power_of_two() {
            return Err(StreamError::RowCount {
                column: 0,
                expected: rows.next_power_of_two(),
                actual: rows,
            });
        }
        let group_count = column_count.div_ceil(k);
        let padded_group_count = group_count.next_power_of_two();
        let padded_column_count = padded_group_count
            .checked_mul(k)
            .ok_or(StreamError::PackedLengthOverflow)?;
        Ok(Self {
            rows,
            column_count,
            k,
            group_count,
            padded_group_count,
            padded_column_count,
        })
    }

    pub fn row_vars(self) -> usize {
        self.rows.trailing_zeros() as usize
    }

    pub fn group_vars(self) -> usize {
        self.padded_group_count.trailing_zeros() as usize
    }

    pub fn slot_vars(self) -> usize {
        self.k.trailing_zeros() as usize
    }

    pub fn column_vars(self) -> usize {
        self.group_vars() + self.slot_vars()
    }

    pub fn packed_vars(self) -> usize {
        self.row_vars() + self.slot_vars()
    }

    pub fn split_column_point(self, point: &[Fr]) -> Result<(&[Fr], &[Fr]), StreamError> {
        if point.len() != self.column_vars() {
            return Err(StreamError::PointDimension {
                expected: self.column_vars(),
                actual: point.len(),
            });
        }
        Ok(point.split_at(self.group_vars()))
    }

    pub fn group_weights(self, column_point: &[Fr]) -> Result<Vec<Fr>, StreamError> {
        let (group_point, _) = self.split_column_point(column_point)?;
        Ok(EqPolynomial::<Fr>::evals(group_point, None)
            .into_iter()
            .take(self.group_count)
            .collect())
    }

    pub fn packed_point(
        self,
        row_point: &[Fr],
        column_point: &[Fr],
    ) -> Result<Vec<Fr>, StreamError> {
        if row_point.len() != self.row_vars() {
            return Err(StreamError::PointDimension {
                expected: self.row_vars(),
                actual: row_point.len(),
            });
        }
        let (_, slot_point) = self.split_column_point(column_point)?;
        let mut point = Vec::with_capacity(self.packed_vars());
        point.extend_from_slice(row_point);
        point.extend_from_slice(slot_point);
        Ok(point)
    }
}

#[derive(Clone, Debug)]
pub struct PackedColumns {
    pub layout: PackingLayout,
    pub evaluations: Vec<Vec<Fr>>,
    pub commitments: Vec<Commitment>,
}

impl PackedColumns {
    pub fn column_evaluations(&self, row_point: &[Fr]) -> Result<Vec<Fr>, StreamError> {
        let expected = self.layout.row_vars();
        if row_point.len() != expected {
            return Err(StreamError::PointDimension {
                expected,
                actual: row_point.len(),
            });
        }
        let row_weights = EqPolynomial::<Fr>::evals(row_point, None);
        let bound_groups: Vec<Vec<Fr>> = self
            .evaluations
            .par_iter()
            .map(|evaluations| {
                let mut values = vec![Fr::zero(); self.layout.k];
                for (&row_weight, row) in row_weights
                    .iter()
                    .zip(evaluations.chunks_exact(self.layout.k))
                {
                    for (value, &entry) in values.iter_mut().zip(row) {
                        *value += row_weight * entry;
                    }
                }
                values
            })
            .collect();
        let mut values: Vec<Fr> = (0..self.layout.column_count)
            .map(|column| bound_groups[column / self.layout.k][column % self.layout.k])
            .collect();
        values.resize(self.layout.padded_column_count, Fr::zero());
        Ok(values)
    }
}

pub fn commit_packed(
    columns: &[Column],
    k: usize,
    setup: &HyperKZGProverSetup<Bn254>,
) -> Result<PackedColumns, StreamError> {
    if columns.is_empty() {
        return Err(StreamError::NoColumns);
    }
    let rows = columns[0].len();
    let layout = PackingLayout::new(rows, columns.len(), k)?;
    for (column, values) in columns.iter().enumerate() {
        if values.len() != rows {
            return Err(StreamError::RowCount {
                column,
                expected: rows,
                actual: values.len(),
            });
        }
        if let Column::Bits(bits) = values {
            if let Some((row, &value)) = bits.iter().enumerate().find(|(_, bit)| **bit > 1) {
                return Err(StreamError::InvalidBit { column, row, value });
            }
        }
    }
    let packed_len = layout
        .rows
        .checked_mul(k)
        .ok_or(StreamError::PackedLengthOverflow)?;
    if setup.g1_powers().len() < packed_len {
        return Err(StreamError::SetupTooSmall {
            required: packed_len,
            actual: setup.g1_powers().len(),
        });
    }
    let groups = layout.group_count;
    let evaluations: Vec<Vec<Fr>> = (0..groups)
        .into_par_iter()
        .map(|group| {
            let mut packed = vec![Fr::zero(); packed_len];
            for row in 0..rows {
                for slot in 0..k {
                    if let Some(column) = columns.get(group * k + slot) {
                        packed[row * k + slot] = column.value(row);
                    }
                }
            }
            packed
        })
        .collect();

    let bit_groups: Vec<usize> = (0..groups)
        .filter(|&group| {
            columns
                .iter()
                .skip(group * k)
                .take(k)
                .all(|column| matches!(column, Column::Bits(_)))
        })
        .collect();
    let packed_bits: Vec<Vec<u8>> = bit_groups
        .par_iter()
        .map(|&group| {
            let mut packed = vec![0; packed_len];
            for row in 0..rows {
                for slot in 0..k {
                    if let Some(Column::Bits(column)) = columns.get(group * k + slot) {
                        packed[row * k + slot] = column[row];
                    }
                }
            }
            packed
        })
        .collect();
    let bit_refs: Vec<&[u8]> = packed_bits.iter().map(Vec::as_slice).collect();
    let mut indexed_commitments: Vec<(usize, Commitment)> = bit_groups
        .into_iter()
        .zip(
            g1_bit_columns_msm(&setup.g1_powers()[..packed_len], &bit_refs)
                .into_iter()
                .map(Commitment::new),
        )
        .collect();
    let mut other_commitments = (0..groups)
        .filter(|&group| {
            !columns
                .iter()
                .skip(group * k)
                .take(k)
                .all(|column| matches!(column, Column::Bits(_)))
        })
        .collect::<Vec<_>>()
        .into_par_iter()
        .map(|group| {
            let all_u16 = columns
                .iter()
                .skip(group * k)
                .take(k)
                .all(|column| matches!(column, Column::U16(_)));
            let commitment = if all_u16 {
                let mut packed = vec![0u16; packed_len];
                for row in 0..rows {
                    for slot in 0..k {
                        if let Some(Column::U16(column)) = columns.get(group * k + slot) {
                            packed[row * k + slot] = column[row];
                        }
                    }
                }
                Commitment::new(Bn254::g1_affine_msm_small(
                    &setup.g1_powers()[..packed_len],
                    &packed,
                ))
            } else {
                HyperKZGScheme::<Bn254>::commit(evaluations[group].as_slice(), setup)
                    .map(|(commitment, ())| commitment)
                    .map_err(StreamError::Commitment)?
            };
            Ok((group, commitment))
        })
        .collect::<Result<Vec<_>, StreamError>>()?;
    indexed_commitments.append(&mut other_commitments);
    indexed_commitments.sort_unstable_by_key(|(group, _)| *group);
    let commitments = indexed_commitments
        .into_iter()
        .map(|(_, commitment)| commitment)
        .collect();

    Ok(PackedColumns {
        layout,
        evaluations,
        commitments,
    })
}

struct ScaledRounds<'a> {
    inner: &'a mut dyn ProveRounds<Fr>,
    scale: Fr,
    scale_inverse: Fr,
}

impl ProveRounds<Fr> for ScaledRounds<'_> {
    fn num_rounds(&self) -> usize {
        self.inner.num_rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        let polynomial =
            self.inner
                .prove_round(bind, round, previous_claim * self.scale_inverse)?;
        Ok(UnivariatePoly::new(
            polynomial
                .coefficients()
                .iter()
                .map(|coefficient| *coefficient * self.scale)
                .collect(),
        ))
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.inner.finish_rounds(bind)
    }
}

pub fn prove_stage<T: Transcript<Challenge = Fr>>(
    members: &mut [StageMember<'_>],
    transcript: &mut T,
) -> Result<(StageProof, StageResult), StreamError> {
    let max_rounds = members
        .iter()
        .map(|member| member.offset + member.prover.num_rounds())
        .max()
        .ok_or(StreamError::EmptyStage)?;
    let max_degree = members
        .iter()
        .map(|member| member.degree)
        .max()
        .ok_or(StreamError::EmptyStage)?;
    let input_claims: Vec<Fr> = members.iter().map(|member| member.input_claim).collect();
    let mut recorder = ClearSumcheckRecorder::<Fr, Commitment>::new();
    recorder.absorb_input_claims(&input_claims, transcript);
    let coefficients: Vec<Fr> = members.iter().map(|_| transcript.challenge()).collect();
    let descriptions: Vec<BatchMember<Fr>> = members
        .iter()
        .zip(&coefficients)
        .map(|(member, &coefficient)| BatchMember {
            input_claim: member.input_claim,
            coefficient,
            rounds: member.prover.num_rounds(),
            offset: member.offset,
        })
        .collect();
    let prelude = BatchPrelude::new(descriptions, max_rounds, max_degree);
    let mut scaled: Vec<ScaledRounds<'_>> = members
        .iter_mut()
        .map(|member| {
            let scale =
                Fr::one().mul_pow_2(max_rounds - member.offset - member.prover.num_rounds());
            Ok(ScaledRounds {
                scale,
                scale_inverse: scale.inverse().ok_or(StreamError::StageScale)?,
                inner: &mut *member.prover,
            })
        })
        .collect::<Result<Vec<_>, StreamError>>()?;
    let mut provers: Vec<&mut dyn ProveRounds<Fr>> = scaled
        .iter_mut()
        .map(|member| member as &mut dyn ProveRounds<Fr>)
        .collect();
    let proved = prove_batch(
        &prelude,
        &mut provers,
        &mut SequentialRounds,
        &mut recorder,
        transcript,
    )?;
    let recorded = recorder.finish(&proved.member_claims, transcript)?;
    let round_polynomials = recorded
        .proof
        .as_clear()
        .and_then(|proof| match proof {
            ClearProof::Compressed(proof) => Some(proof.clone()),
            ClearProof::Full(_) => None,
        })
        .ok_or(StreamError::StageOutputClaim)?;
    let proof = StageProof { round_polynomials };
    Ok((
        proof,
        StageResult {
            point: proved.challenges,
            coefficients,
            output_claims: proved.member_claims,
            final_claim: proved.final_claim,
        },
    ))
}

pub fn verify_stage_with<T, F>(
    proof: &StageProof,
    members: &[StageMemberSpec],
    input_claims: &[Fr],
    transcript: &mut T,
    output_claims: F,
) -> Result<StageResult, StreamError>
where
    T: Transcript<Challenge = Fr>,
    F: FnOnce(&StageResult) -> Result<Vec<Fr>, StreamError>,
{
    let mut result = verify_stage_without_output(proof, members, input_claims, transcript)?;
    let output_claims = output_claims(&result)?;
    if output_claims.len() != members.len() {
        return Err(StreamError::StageMemberCount);
    }
    let expected: Fr = result
        .coefficients
        .iter()
        .zip(&output_claims)
        .map(|(&coefficient, &claim)| coefficient * claim)
        .sum();
    if result.final_claim != expected {
        return Err(StreamError::StageOutputClaim);
    }
    absorb_output_claims(&output_claims, transcript);
    result.output_claims = output_claims;
    Ok(result)
}

fn absorb_output_claims<T: Transcript>(claims: &[Fr], transcript: &mut T) {
    for claim in claims {
        transcript.append_labeled(OPENING_CLAIM_TRANSCRIPT_LABEL, claim);
    }
}

fn verify_stage_without_output<T: Transcript<Challenge = Fr>>(
    proof: &StageProof,
    members: &[StageMemberSpec],
    input_claims: &[Fr],
    transcript: &mut T,
) -> Result<StageResult, StreamError> {
    if input_claims.len() != members.len() {
        return Err(StreamError::StageMemberCount);
    }
    let max_rounds = members
        .iter()
        .map(|member| member.offset + member.rounds)
        .max()
        .ok_or(StreamError::EmptyStage)?;
    let max_degree = members
        .iter()
        .map(|member| member.degree)
        .max()
        .ok_or(StreamError::EmptyStage)?;
    let mut recorder = ClearSumcheckRecorder::<Fr, Commitment>::new();
    recorder.absorb_input_claims(input_claims, transcript);
    let coefficients: Vec<Fr> = members.iter().map(|_| transcript.challenge()).collect();
    let descriptions: Vec<BatchMember<Fr>> = members
        .iter()
        .zip(input_claims)
        .zip(&coefficients)
        .map(|((member, &input_claim), &coefficient)| BatchMember {
            input_claim,
            coefficient,
            rounds: member.rounds,
            offset: member.offset,
        })
        .collect();
    let prelude = BatchPrelude::new(descriptions, max_rounds, max_degree);
    let evaluation = proof.round_polynomials.verify(
        &SumcheckClaim::new(max_rounds, max_degree, prelude.claimed_sum),
        BooleanHypercube,
        SUMCHECK_ROUND_TRANSCRIPT_LABEL,
        transcript,
    )?;
    Ok(StageResult {
        point: evaluation.point.into_vec(),
        coefficients,
        output_claims: Vec::new(),
        final_claim: evaluation.value,
    })
}

pub struct ColumnBatching {
    terms: Vec<TensorTerm>,
    polynomials: Vec<Polynomial<Fr>>,
    eq_prefixes: Vec<Vec<Fr>>,
    log_columns: usize,
    arity: usize,
    round: usize,
    claim: Fr,
}

impl ColumnBatching {
    pub fn new(values: Vec<Fr>, terms: Vec<TensorTerm>) -> Result<Self, StreamError> {
        if values.is_empty() || !values.len().is_power_of_two() {
            return Err(StreamError::NoColumns);
        }
        let arity = terms
            .first()
            .map(|term| term.columns.len())
            .filter(|arity| *arity > 0)
            .ok_or(StreamError::EmptyTensor)?;
        for (term_index, term) in terms.iter().enumerate() {
            if term.columns.len() != arity {
                return Err(StreamError::TensorArity {
                    term: term_index,
                    expected: arity,
                    actual: term.columns.len(),
                });
            }
            if let Some(&column) = term.columns.iter().find(|&&column| column >= values.len()) {
                return Err(StreamError::ColumnOutOfRange {
                    column,
                    columns: values.len(),
                });
            }
        }
        let claim = terms
            .iter()
            .map(|term| {
                term.columns
                    .iter()
                    .fold(term.coefficient, |product, &column| {
                        product * values[column]
                    })
            })
            .sum();
        Ok(Self {
            eq_prefixes: vec![vec![Fr::one(); arity]; terms.len()],
            polynomials: (0..arity)
                .map(|_| Polynomial::new(values.clone()))
                .collect(),
            log_columns: values.len().trailing_zeros() as usize,
            terms,
            arity,
            round: 0,
            claim,
        })
    }

    pub fn input_claim(&self) -> Fr {
        self.claim
    }

    pub fn final_evaluations(&self) -> Vec<Fr> {
        self.polynomials
            .iter()
            .map(|polynomial| polynomial.evals()[0])
            .collect()
    }

    pub fn expected_final(
        column_count: usize,
        terms: &[TensorTerm],
        point: &[Fr],
        evaluations: &[Fr],
    ) -> Result<Fr, StreamError> {
        if column_count == 0 || !column_count.is_power_of_two() {
            return Err(StreamError::NoColumns);
        }
        let arity = terms
            .first()
            .map(|term| term.columns.len())
            .filter(|arity| *arity > 0)
            .ok_or(StreamError::EmptyTensor)?;
        let log_columns = column_count.trailing_zeros() as usize;
        if point.len() != arity * log_columns {
            return Err(StreamError::PointDimension {
                expected: arity * log_columns,
                actual: point.len(),
            });
        }
        if evaluations.len() != arity {
            return Err(StreamError::StageMemberCount);
        }
        let tensor_eval: Fr = terms
            .iter()
            .enumerate()
            .map(|(term_index, term)| {
                if term.columns.len() != arity {
                    return Err(StreamError::TensorArity {
                        term: term_index,
                        expected: arity,
                        actual: term.columns.len(),
                    });
                }
                let mut value = term.coefficient;
                for (factor, &column) in term.columns.iter().enumerate() {
                    if column >= column_count {
                        return Err(StreamError::ColumnOutOfRange {
                            column,
                            columns: column_count,
                        });
                    }
                    let start = factor * log_columns;
                    value *= EqPolynomial::<Fr>::mle(
                        &boolean_point(column, log_columns),
                        &point[start..start + log_columns],
                    );
                }
                Ok(value)
            })
            .sum::<Result<Fr, StreamError>>()?;
        Ok(evaluations
            .iter()
            .fold(tensor_eval, |value, evaluation| value * *evaluation))
    }

    fn bind_previous(&mut self, challenge: Fr) {
        let previous = self.round - 1;
        let block = previous / self.log_columns;
        let bit = previous % self.log_columns;
        for (term, prefixes) in self.terms.iter().zip(&mut self.eq_prefixes) {
            let column_bit = (term.columns[block] >> (self.log_columns - bit - 1)) & 1;
            prefixes[block] *= if column_bit == 0 {
                Fr::one() - challenge
            } else {
                challenge
            };
        }
        self.polynomials[block].bind_with_order(challenge, BindingOrder::HighToLow);
    }

    fn round_evaluation(&self, x: Fr) -> Fr {
        let block = self.round / self.log_columns;
        let bit = self.round % self.log_columns;
        let remaining = self.log_columns - bit - 1;
        self.terms
            .iter()
            .zip(&self.eq_prefixes)
            .map(|(term, prefixes)| {
                let mut product = term.coefficient;
                for (factor, (&prefix, &column)) in prefixes.iter().zip(&term.columns).enumerate() {
                    if factor == block {
                        let column_bit = (column >> remaining) & 1;
                        let eq = if column_bit == 0 { Fr::one() - x } else { x };
                        let suffix = column & ((1usize << remaining) - 1);
                        product *= prefix
                            * eq
                            * self.polynomials[factor].sumcheck_round_eval_with_order(
                                suffix,
                                x,
                                BindingOrder::HighToLow,
                            );
                    } else {
                        let suffix_bits = self.polynomials[factor].num_vars();
                        let suffix = column & ((1usize << suffix_bits) - 1);
                        product *= prefix * self.polynomials[factor].evals()[suffix];
                    }
                }
                product
            })
            .sum()
    }
}

impl ProveRounds<Fr> for ColumnBatching {
    fn num_rounds(&self) -> usize {
        self.arity * self.log_columns
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        self.round = round;
        if let Some(challenge) = bind {
            self.bind_previous(challenge);
        }
        let evaluations = [
            self.round_evaluation(Fr::zero()),
            self.round_evaluation(Fr::one()),
            self.round_evaluation(Fr::from_u64(2)),
        ];
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
        self.round = self.num_rounds();
        self.bind_previous(bind);
        Ok(())
    }
}

pub struct ClaimReduction {
    polynomials: Vec<Polynomial<Fr>>,
    weights: Vec<Polynomial<Fr>>,
    rounds: usize,
    claim: Fr,
}

impl ClaimReduction {
    pub fn new(
        polynomials: &[Vec<Fr>],
        claims: &[ReductionClaim],
        coefficients: &[Fr],
    ) -> Result<Self, StreamError> {
        if polynomials.is_empty() {
            return Err(StreamError::NoColumns);
        }
        if claims.len() != coefficients.len() {
            return Err(StreamError::CoefficientCount {
                claims: claims.len(),
                coefficients: coefficients.len(),
            });
        }
        let len = polynomials[0].len();
        if len == 0 || !len.is_power_of_two() {
            return Err(StreamError::PolynomialLength {
                polynomial: 0,
                expected: len.next_power_of_two(),
                actual: len,
            });
        }
        for (polynomial, evaluations) in polynomials.iter().enumerate().skip(1) {
            if evaluations.len() != len {
                return Err(StreamError::PolynomialLength {
                    polynomial,
                    expected: len,
                    actual: evaluations.len(),
                });
            }
        }
        let rounds = len.trailing_zeros() as usize;
        let mut weights = vec![vec![Fr::zero(); len]; polynomials.len()];
        for (claim_index, (claim, &coefficient)) in claims.iter().zip(coefficients).enumerate() {
            if claim.polynomial_weights.len() != polynomials.len() {
                return Err(StreamError::PolynomialWeightCount {
                    claim: claim_index,
                    expected: polynomials.len(),
                    actual: claim.polynomial_weights.len(),
                });
            }
            if claim.point.len() != rounds {
                return Err(StreamError::PointDimension {
                    expected: rounds,
                    actual: claim.point.len(),
                });
            }
            let eq = EqPolynomial::<Fr>::evals(&claim.point, None);
            for (polynomial_weights, &polynomial_coefficient) in
                weights.iter_mut().zip(&claim.polynomial_weights)
            {
                for (weight, &eq_value) in polynomial_weights.iter_mut().zip(&eq) {
                    *weight += coefficient * polynomial_coefficient * eq_value;
                }
            }
        }
        Ok(Self {
            polynomials: polynomials.iter().cloned().map(Polynomial::new).collect(),
            weights: weights.into_iter().map(Polynomial::new).collect(),
            rounds,
            claim: claims
                .iter()
                .zip(coefficients)
                .map(|(claim, &coefficient)| coefficient * claim.value)
                .sum(),
        })
    }

    pub fn input_claim(&self) -> Fr {
        self.claim
    }

    pub fn final_weights(&self) -> Vec<Fr> {
        self.weights
            .iter()
            .map(|polynomial| polynomial.evals()[0])
            .collect()
    }

    fn bind_previous(&mut self, challenge: Fr) {
        for polynomial in &mut self.polynomials {
            polynomial.bind_with_order(challenge, BindingOrder::HighToLow);
        }
        for weight in &mut self.weights {
            weight.bind_with_order(challenge, BindingOrder::HighToLow);
        }
    }
}

impl ProveRounds<Fr> for ClaimReduction {
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
            self.bind_previous(challenge);
        }
        let half = self.polynomials[0].len() / 2;
        let evaluations = (0..half)
            .into_par_iter()
            .map(|index| {
                let mut local = [Fr::zero(); 3];
                for (x, evaluation) in local.iter_mut().enumerate() {
                    let x = Fr::from_u64(x as u64);
                    *evaluation = self
                        .polynomials
                        .iter()
                        .zip(&self.weights)
                        .map(|(polynomial, weight)| {
                            polynomial.sumcheck_round_eval(index, x)
                                * weight.sumcheck_round_eval(index, x)
                        })
                        .sum();
                }
                local
            })
            .reduce(
                || [Fr::zero(); 3],
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
        self.bind_previous(bind);
        Ok(())
    }
}

pub(super) fn combine_evaluations(polynomials: &[Vec<Fr>], weights: &[Fr]) -> Vec<Fr> {
    (0..polynomials[0].len())
        .into_par_iter()
        .map(|index| {
            polynomials
                .iter()
                .zip(weights)
                .map(|(polynomial, &weight)| polynomial[index] * weight)
                .sum()
        })
        .collect()
}

fn boolean_point(index: usize, variables: usize) -> Vec<Fr> {
    (0..variables)
        .map(|bit| Fr::from_u64(((index >> (variables - bit - 1)) & 1) as u64))
        .collect()
}
