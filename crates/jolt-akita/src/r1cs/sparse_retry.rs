//! Fixed-capacity first norm acceptance on one contiguous indexed Blake tape.
use jolt_field::{Fr, Ring};
use jolt_r1cs::{
    bn254_bits::ByteVar, integer_bn254::SignedVar, LinearCombination, R1csBuilder, Variable,
};
use jolt_transcript::r1cs::Blake2bR1csError;
use thiserror::Error;

use super::sparse_routing::ReadTape;

use super::{
    operator_norm::{D64ShellVar, OperatorNormR1csError},
    AkitaSparseStreamVar, CandidateError, D64CandidateProfile,
};

type Expression = LinearCombination<Fr>;

#[derive(Debug, Error)]
pub enum SparseRetryError {
    #[error("invalid sparse retry profile")]
    Shape,
    #[error("no norm-accepted candidate within public retry capacity")]
    CapacityExceeded,
    #[error(transparent)]
    Candidate(#[from] CandidateError),
    #[error(transparent)]
    Norm(#[from] OperatorNormR1csError),
    #[error(transparent)]
    Blake(#[from] Blake2bR1csError),
}

/// Practical finite capacity, separately from native's 4096 candidate cap.
/// All activity, rejection counts, and cursor positions are private assignments.
#[derive(Clone, Copy, Debug)]
pub struct D64RetryProfile {
    candidate: D64CandidateProfile,
    rounds: usize,
    tape_len: usize,
}

impl D64RetryProfile {
    /// Select a public retry/trial capacity; does not promise negligible overflow.
    pub fn new(rounds: usize, trials: usize) -> Result<Self, SparseRetryError> {
        if rounds == 0 || rounds > 4096 {
            return Err(SparseRetryError::Shape);
        }
        let candidate = D64CandidateProfile::selective_l2(trials)?;
        let tape_len = candidate
            .tape_len()
            .checked_mul(rounds)
            .filter(|&n| n < usize::MAX)
            .ok_or(SparseRetryError::Shape)?;
        Ok(Self {
            candidate,
            rounds,
            tape_len,
        })
    }

    /// Constrain the native first accepted indexed challenge under public R/K limits.
    /// Root must be transcript-authenticated externally. ONE must be fixed, and
    /// all byte handles must belong to this builder. Errors leave partial rows.
    /// This authenticates the whole fixed tape but consumes only the private prefix.
    pub fn sample(
        &self,
        builder: &mut R1csBuilder<Fr>,
        root: &[ByteVar; 32],
        coordinate: u64,
    ) -> Result<D64AcceptedVar, SparseRetryError> {
        let mut stream = AkitaSparseStreamVar::new(builder, root, coordinate)?;
        let tape = stream.read(builder, self.tape_len)?;
        self.sample_tape(builder, &tape)
    }

    fn sample_tape(
        &self,
        builder: &mut R1csBuilder<Fr>,
        tape: &[ByteVar],
    ) -> Result<D64AcceptedVar, SparseRetryError> {
        if tape.len() != self.tape_len {
            return Err(SparseRetryError::Shape);
        }
        ReadTape::constrain(builder, tape, self.tape_len, |builder, reads| {
            let mut activity = Expression::one();
            let mut output: [Expression; 64] = std::array::from_fn(|_| Expression::zero());
            let mut selected = Vec::with_capacity(self.rounds);
            for _ in 0..self.rounds {
                let candidate = self.candidate.sample_at(builder, reads, activity.clone())?;
                let coefficients = candidate.dense().each_ref().map(SignedVar::variable);
                let values: Option<Vec<_>> = coefficients
                    .iter()
                    .map(|&v| {
                        builder
                            .evaluate(&Expression::variable(v))
                            .ok()
                            .and_then(|value| {
                                (-2i8..=2).find(|&n| {
                                    let magnitude = Fr::from_u64(n.unsigned_abs().into());
                                    value == if n < 0 { -magnitude } else { magnitude }
                                })
                            })
                    })
                    .collect();
                let values = values
                    .map(|v| v.try_into().map_err(|_| SparseRetryError::Shape))
                    .transpose()?;
                let shell = D64ShellVar::bind(builder, coefficients, values)?;
                let accepted = shell.acceptance(builder)?;
                let take = builder.multiply(activity.clone(), accepted);
                let next_activity = activity - take.clone();
                let witness = builder.evaluate(&next_activity).ok();
                let next = builder.alloc_witness(witness);
                builder.assert_equal(next, next_activity);
                activity = Expression::variable(next);
                for (output, coefficient) in output.iter_mut().zip(coefficients) {
                    *output = output.clone() + builder.multiply(take.clone(), coefficient);
                }
                selected.push(take);
            }
            builder.assert_zero(activity.clone());
            if builder
                .evaluate(&activity)
                .is_ok_and(|v| v != Fr::from_u64(0))
            {
                return Err(SparseRetryError::CapacityExceeded);
            }
            let coefficients = output.map(|value| {
                let assignment = builder.evaluate(&value).ok();
                let variable = builder.alloc_witness(assignment);
                builder.assert_equal(variable, value);
                variable
            });
            let consumed = reads.consumed();
            let cursor = reads.one_hot_cursor(builder);
            Ok(D64AcceptedVar {
                coefficients,
                selected,
                cursor,
                consumed,
            })
        })
    }
}

/// Dense coefficients of the first accepted candidate, with private progress.
/// Output variables are constrained to one selected native mixed-shell candidate.
pub struct D64AcceptedVar {
    coefficients: [Variable; 64],
    selected: Vec<Expression>,
    cursor: Vec<Expression>,
    consumed: Expression,
}
impl D64AcceptedVar {
    /// Dense coefficients, each constrained to the selected native candidate.
    pub fn coefficients(&self) -> &[Variable; 64] {
        &self.coefficients
    }
    /// One-hot first-accepted candidate selector; must remain private in a ZK wrapper.
    pub fn selected(&self) -> &[Expression] {
        &self.selected
    }
    /// One-hot absolute position within this coordinate's Blake stream prefix.
    pub fn end_cursor(&self) -> &[Expression] {
        &self.cursor
    }
    /// Private consumed prefix length, excluding inactive slots and padding.
    pub fn consumed_bytes(&self) -> Expression {
        self.consumed.clone()
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::indexing_slicing,
    reason = "native fixture parity and adversarial assignments"
)]
mod tests {
    use super::*;
    use akita_challenges::{
        d64_selective_l2_op_norm_table, FoldChallengeDrawDomain, FoldDraw, OperatorNormRejection,
        SparseChallenge, D64_SELECTIVE_L2_CHALLENGE_CONFIG, SPARSE_CHALLENGE_STREAM_DOMAIN,
    };
    use akita_transcript::blake2b_stream::Blake2bStream;

    struct FixedRoot(u8);
    impl FoldDraw for FixedRoot {
        fn absorb_and_squeeze(&mut self, _: &[u8], _: &[u8]) -> [u8; 32] {
            [self.0; 32]
        }
    }
    fn native(seed: u8, rejection: bool) -> SparseChallenge {
        FixedRoot(seed)
            .draw_folding_challenges_with_rejection(
                FoldChallengeDrawDomain::EvaluationTrace,
                64,
                0,
                1,
                1,
                &D64_SELECTIVE_L2_CHALLENGE_CONFIG,
                0,
                rejection.then_some(OperatorNormRejection::D64_SELECTIVE_L2),
            )
            .unwrap()
            .as_slice()[0]
            .clone()
    }
    fn tape(profile: D64RetryProfile, seed: u8) -> Vec<u8> {
        let mut context = vec![seed; 32];
        context.extend([0; 8]);
        let mut bytes = vec![0; profile.tape_len];
        Blake2bStream::new(SPARSE_CHALLENGE_STREAM_DOMAIN, &context)
            .unwrap()
            .read(&mut bytes)
            .unwrap();
        bytes
    }
    fn dense(candidate: &SparseChallenge) -> [Fr; 64] {
        let mut dense = [Fr::from_u64(0); 64];
        for (&p, &c) in candidate.positions.iter().zip(&candidate.coeffs) {
            let magnitude = Fr::from_u64(c.unsigned_abs().into());
            dense[p as usize] = if c < 0 { -magnitude } else { magnitude };
        }
        dense
    }

    #[test]
    fn native_first_rejection_then_acceptance_and_coherent_wrong_output() {
        let first = native(13, false);
        let expected = native(13, true);
        let table = d64_selective_l2_op_norm_table().unwrap();
        assert!(!table
            .accept_strict_parts(&first.positions, &first.coeffs, 18)
            .unwrap());
        assert!(table
            .accept_strict_parts(&expected.positions, &expected.coeffs, 18)
            .unwrap());
        assert_ne!(first, expected);
        let mut builder = R1csBuilder::new();
        let root = std::array::from_fn(|_| ByteVar::allocate(&mut builder, Some(13)));
        let result = D64RetryProfile::new(2, 4)
            .unwrap()
            .sample(&mut builder, &root, 0)
            .unwrap();
        for (i, selector) in result.selected().iter().enumerate() {
            assert_eq!(
                builder.evaluate(selector).unwrap(),
                Fr::from_u64(u64::from(i == 1))
            );
        }
        assert_eq!(
            builder.evaluate(&result.consumed_bytes()).unwrap(),
            Fr::from_u64(190)
        );
        for (&variable, expected) in result.coefficients().iter().zip(dense(&expected)) {
            assert_eq!(
                builder.evaluate(&Expression::variable(variable)).unwrap(),
                expected
            );
        }
        let witness = builder.witness().unwrap();
        let matrices = builder.into_matrices();
        assert!(matrices.check_witness(&witness).is_ok());
        // Replace the entire output coherently with the actual rejected first shell.
        let mut forged = witness.clone();
        for (&variable, value) in result.coefficients().iter().zip(dense(&first)) {
            forged[variable.index()] = value;
        }
        for (i, selector) in result.selected().iter().enumerate() {
            let variable = selector.terms[0].0;
            forged[variable.index()] = Fr::from_u64(u64::from(i == 0));
        }
        assert!(matrices.check_witness(&forged).is_err());
        let mut wrong_cursor = witness;
        let variable = result.end_cursor()[190].terms[0].0;
        wrong_cursor[variable.index()] += Fr::from_u64(1);
        assert!(matrices.check_witness(&wrong_cursor).is_err());
    }

    #[test]
    fn hidden_retry_count_unknown_shape_and_inactive_cursor() {
        let profile = D64RetryProfile::new(2, 4).unwrap();
        let emit = |seed: Option<u8>| {
            let values = seed.map(|s| tape(profile, s));
            let mut builder = R1csBuilder::new();
            let input: Vec<_> = (0..profile.tape_len)
                .map(|i| ByteVar::allocate(&mut builder, values.as_ref().map(|v| v[i])))
                .collect();
            let result = profile.sample_tape(&mut builder, &input).unwrap();
            if let Some(seed) = seed {
                let expected = native(seed, true);
                for (&variable, value) in result.coefficients().iter().zip(dense(&expected)) {
                    assert_eq!(
                        builder.evaluate(&Expression::variable(variable)).unwrap(),
                        value
                    );
                }
                assert_eq!(
                    builder.evaluate(&result.consumed_bytes()).unwrap(),
                    Fr::from_u64(if seed == 0 { 89 } else { 190 })
                );
                let witness = builder.witness().unwrap();
                let matrices = builder.into_matrices();
                assert!(matrices.check_witness(&witness).is_ok());
                matrices
            } else {
                builder.into_matrices()
            }
        };
        let known = emit(Some(0));
        for other in [Some(13), None] {
            let other = emit(other);
            assert_eq!(known.num_vars, other.num_vars);
            assert_eq!(known.a, other.a);
            assert_eq!(known.b, other.b);
            assert_eq!(known.c, other.c);
        }
    }

    #[test]
    fn retry_and_position_capacity_do_not_skip_rejections() {
        for (rounds, trials, seed, position_failure) in
            [(1, 4, 13, false), (2, 4, 5, false), (2, 3, 13, true)]
        {
            let profile = D64RetryProfile::new(rounds, trials).unwrap();
            let mut builder = R1csBuilder::new();
            let input: Vec<_> = tape(profile, seed)
                .into_iter()
                .map(|v| ByteVar::allocate(&mut builder, Some(v)))
                .collect();
            let result = profile.sample_tape(&mut builder, &input);
            if position_failure {
                assert!(matches!(
                    result,
                    Err(SparseRetryError::Candidate(
                        CandidateError::CapacityExceeded
                    ))
                ));
            } else {
                assert!(matches!(result, Err(SparseRetryError::CapacityExceeded)));
            }
            let witness = builder.witness().unwrap();
            assert!(builder.into_matrices().check_witness(&witness).is_err());
        }
        assert!(D64RetryProfile::new(4097, 4).is_err());
        assert!(D64RetryProfile::new(0, 4).is_err());
    }
}
