use jolt_field::Field;
use jolt_r1cs::{LinearCombination, R1csBuilder, Variable};
use thiserror::Error;

use crate::{SumcheckShape, VerifiedCommittedRound};

pub trait SumcheckR1csRound<F> {
    fn degree(&self) -> usize;
    fn challenge(&self) -> F;
}

impl<F: Copy, C> SumcheckR1csRound<F> for VerifiedCommittedRound<F, C> {
    fn degree(&self) -> usize {
        self.degree
    }

    fn challenge(&self) -> F {
        self.challenge
    }
}

#[derive(Clone, Debug, Error, PartialEq, Eq)]
pub enum SumcheckR1csError {
    #[error("sumcheck expects {expected} rounds but input has {actual}")]
    WrongNumberOfRounds { expected: usize, actual: usize },
    #[error("round {round_index} has degree {actual}, exceeding bound {bound}")]
    DegreeBoundExceeded {
        round_index: usize,
        bound: usize,
        actual: usize,
    },
    #[error("layout has {actual} rounds but sumcheck input has {expected}")]
    LayoutRoundCountMismatch { expected: usize, actual: usize },
    #[error("round {round_index} layout has no coefficient variables")]
    EmptyRoundLayout { round_index: usize },
    #[error(
        "round {round_index} layout has degree {actual} but sumcheck input has degree {expected}"
    )]
    LayoutRoundDegreeMismatch {
        round_index: usize,
        expected: usize,
        actual: usize,
    },
    #[error(
        "round {round_index} expected input claim variable {expected:?} but layout uses {actual:?}"
    )]
    RoundClaimLinkMismatch {
        round_index: usize,
        expected: Variable,
        actual: Variable,
    },
    #[error("expected output claim variable {expected:?} but layout uses {actual:?}")]
    OutputClaimLinkMismatch {
        expected: Variable,
        actual: Variable,
    },
    #[error("layout references variable {variable:?} but builder has {num_vars} variables")]
    LayoutVariableOutOfBounds { variable: Variable, num_vars: usize },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckR1csLayout {
    pub input_claim: Variable,
    pub rounds: Vec<SumcheckR1csRoundLayout>,
    pub output_claim: Variable,
}

impl SumcheckR1csLayout {
    pub fn round_count(&self) -> usize {
        self.rounds.len()
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SumcheckR1csRoundLayout {
    pub claim_in: Variable,
    pub coefficients: Vec<Variable>,
    pub claim_out: Variable,
}

impl SumcheckR1csRoundLayout {
    pub fn degree(&self) -> usize {
        self.coefficients.len().saturating_sub(1)
    }
}

pub fn allocate_sumcheck_r1cs_layout<F, R>(
    builder: &mut R1csBuilder<F>,
    shape: SumcheckShape,
    rounds: &[R],
) -> Result<SumcheckR1csLayout, SumcheckR1csError>
where
    F: Field,
    R: SumcheckR1csRound<F>,
{
    validate_rounds_shape(shape, rounds)?;

    let input_claim = builder.alloc_unknown();
    let mut claim_in = input_claim;
    let mut round_layouts = Vec::with_capacity(rounds.len());

    for round in rounds {
        let coefficients = (0..=round.degree())
            .map(|_| builder.alloc_unknown())
            .collect();
        let claim_out = builder.alloc_unknown();

        round_layouts.push(SumcheckR1csRoundLayout {
            claim_in,
            coefficients,
            claim_out,
        });
        claim_in = claim_out;
    }

    Ok(SumcheckR1csLayout {
        input_claim,
        rounds: round_layouts,
        output_claim: claim_in,
    })
}

pub fn append_sumcheck_r1cs_constraints<F, R>(
    builder: &mut R1csBuilder<F>,
    shape: SumcheckShape,
    rounds: &[R],
    layout: &SumcheckR1csLayout,
) -> Result<(), SumcheckR1csError>
where
    F: Field,
    R: SumcheckR1csRound<F>,
{
    validate_layout(builder.num_vars(), shape, rounds, layout)?;

    for (round_layout, round) in layout.rounds.iter().zip(rounds) {
        append_round_constraints(builder, round_layout, round.challenge());
    }

    Ok(())
}

fn validate_layout<F, R>(
    num_vars: usize,
    shape: SumcheckShape,
    rounds: &[R],
    layout: &SumcheckR1csLayout,
) -> Result<(), SumcheckR1csError>
where
    F: Field,
    R: SumcheckR1csRound<F>,
{
    validate_rounds_shape(shape, rounds)?;
    validate_variable(layout.input_claim, num_vars)?;
    validate_variable(layout.output_claim, num_vars)?;

    if rounds.len() != layout.rounds.len() {
        return Err(SumcheckR1csError::LayoutRoundCountMismatch {
            expected: rounds.len(),
            actual: layout.rounds.len(),
        });
    }

    let mut expected_claim_in = layout.input_claim;
    for (round_index, (round, round_layout)) in rounds.iter().zip(&layout.rounds).enumerate() {
        validate_variable(round_layout.claim_in, num_vars)?;
        validate_variable(round_layout.claim_out, num_vars)?;
        for &coefficient in &round_layout.coefficients {
            validate_variable(coefficient, num_vars)?;
        }

        if round_layout.coefficients.is_empty() {
            return Err(SumcheckR1csError::EmptyRoundLayout { round_index });
        }

        let layout_degree = round_layout.degree();
        if round.degree() != layout_degree {
            return Err(SumcheckR1csError::LayoutRoundDegreeMismatch {
                round_index,
                expected: round.degree(),
                actual: layout_degree,
            });
        }

        if round_layout.claim_in != expected_claim_in {
            return Err(SumcheckR1csError::RoundClaimLinkMismatch {
                round_index,
                expected: expected_claim_in,
                actual: round_layout.claim_in,
            });
        }
        expected_claim_in = round_layout.claim_out;
    }

    if layout.output_claim != expected_claim_in {
        return Err(SumcheckR1csError::OutputClaimLinkMismatch {
            expected: expected_claim_in,
            actual: layout.output_claim,
        });
    }

    Ok(())
}

fn validate_rounds_shape<F, R>(shape: SumcheckShape, rounds: &[R]) -> Result<(), SumcheckR1csError>
where
    R: SumcheckR1csRound<F>,
{
    if shape.num_vars != rounds.len() {
        return Err(SumcheckR1csError::WrongNumberOfRounds {
            expected: shape.num_vars,
            actual: rounds.len(),
        });
    }

    for (round_index, round) in rounds.iter().enumerate() {
        if round.degree() > shape.degree {
            return Err(SumcheckR1csError::DegreeBoundExceeded {
                round_index,
                bound: shape.degree,
                actual: round.degree(),
            });
        }
    }

    Ok(())
}

fn validate_variable(variable: Variable, num_vars: usize) -> Result<(), SumcheckR1csError> {
    if variable.0 >= num_vars {
        return Err(SumcheckR1csError::LayoutVariableOutOfBounds { variable, num_vars });
    }

    Ok(())
}

fn append_round_constraints<F: Field>(
    builder: &mut R1csBuilder<F>,
    round: &SumcheckR1csRoundLayout,
    challenge: F,
) {
    builder.assert_equal(check_round_sum_lc(round), round.claim_in);
    builder.assert_equal(
        polynomial_eval_lc(&round.coefficients, challenge),
        round.claim_out,
    );
}

fn check_round_sum_lc<F: Field>(round: &SumcheckR1csRoundLayout) -> LinearCombination<F> {
    LinearCombination::variable(round.coefficients[0])
        + polynomial_eval_lc(&round.coefficients, F::one())
}

fn polynomial_eval_lc<F: Field>(coefficients: &[Variable], point: F) -> LinearCombination<F> {
    let mut result = LinearCombination::zero();
    let mut power = F::one();

    for &coefficient in coefficients {
        result = result + LinearCombination::variable(coefficient).scale(power);
        power *= point;
    }

    result
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests may panic on assertion failures")]
mod tests {
    use super::*;
    use jolt_field::{Fr, FromPrimitiveInt};

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Round {
        degree: usize,
        challenge: Fr,
    }

    impl SumcheckR1csRound<Fr> for Round {
        fn degree(&self) -> usize {
            self.degree
        }

        fn challenge(&self) -> Fr {
            self.challenge
        }
    }

    fn round(degree: usize, challenge: u64) -> Round {
        Round {
            degree,
            challenge: Fr::from_u64(challenge),
        }
    }

    fn assign(builder: &mut R1csBuilder<Fr>, variable: Variable, value: u64) {
        builder
            .assign(variable, Fr::from_u64(value))
            .expect("assignment succeeds");
    }

    fn assign_round(
        builder: &mut R1csBuilder<Fr>,
        round: &SumcheckR1csRoundLayout,
        coefficients: &[u64],
        claim_out: u64,
    ) {
        for (&variable, &coefficient) in round.coefficients.iter().zip(coefficients) {
            assign(builder, variable, coefficient);
        }
        assign(builder, round.claim_out, claim_out);
    }

    #[test]
    fn emits_satisfied_round_constraints() {
        let shape = SumcheckShape::new(2, 1);
        let rounds = [round(1, 2), round(1, 3)];
        let mut builder = R1csBuilder::<Fr>::new();

        let layout = allocate_sumcheck_r1cs_layout(&mut builder, shape, &rounds)
            .expect("layout should allocate");
        append_sumcheck_r1cs_constraints(&mut builder, shape, &rounds, &layout)
            .expect("constraints should build");

        assign(&mut builder, layout.input_claim, 10);
        assign_round(&mut builder, &layout.rounds[0], &[3, 4], 11);
        assign_round(&mut builder, &layout.rounds[1], &[5, 1], 8);

        let witness = builder.witness().expect("witness is assigned");
        assert!(builder.into_matrices().check_witness(&witness).is_ok());
    }

    #[test]
    fn rejects_bad_round_sum() {
        let shape = SumcheckShape::new(1, 1);
        let rounds = [round(1, 2)];
        let mut builder = R1csBuilder::<Fr>::new();

        let layout = allocate_sumcheck_r1cs_layout(&mut builder, shape, &rounds)
            .expect("layout should allocate");
        append_sumcheck_r1cs_constraints(&mut builder, shape, &rounds, &layout)
            .expect("constraints should build");

        assign(&mut builder, layout.input_claim, 10);
        assign_round(&mut builder, &layout.rounds[0], &[3, 5], 13);

        let witness = builder.witness().expect("witness is assigned");
        assert!(builder.into_matrices().check_witness(&witness).is_err());
    }

    #[test]
    fn rejects_bad_challenge_transition() {
        let shape = SumcheckShape::new(1, 1);
        let rounds = [round(1, 2)];
        let mut builder = R1csBuilder::<Fr>::new();

        let layout = allocate_sumcheck_r1cs_layout(&mut builder, shape, &rounds)
            .expect("layout should allocate");
        append_sumcheck_r1cs_constraints(&mut builder, shape, &rounds, &layout)
            .expect("constraints should build");

        assign(&mut builder, layout.input_claim, 10);
        assign_round(&mut builder, &layout.rounds[0], &[3, 4], 12);

        let witness = builder.witness().expect("witness is assigned");
        assert!(builder.into_matrices().check_witness(&witness).is_err());
    }

    #[test]
    fn supports_constant_rounds_under_degree_bound() {
        let shape = SumcheckShape::new(1, 1);
        let rounds = [round(0, 99)];
        let mut builder = R1csBuilder::<Fr>::new();

        let layout = allocate_sumcheck_r1cs_layout(&mut builder, shape, &rounds)
            .expect("layout should allocate");
        append_sumcheck_r1cs_constraints(&mut builder, shape, &rounds, &layout)
            .expect("constraints should build");

        assign(&mut builder, layout.input_claim, 14);
        assign_round(&mut builder, &layout.rounds[0], &[7], 7);

        let witness = builder.witness().expect("witness is assigned");
        assert!(builder.into_matrices().check_witness(&witness).is_ok());
    }

    #[test]
    fn rejects_wrong_number_of_rounds() {
        let shape = SumcheckShape::new(2, 1);
        let rounds = [round(1, 2)];
        let mut builder = R1csBuilder::<Fr>::new();

        let error = allocate_sumcheck_r1cs_layout(&mut builder, shape, &rounds)
            .expect_err("round count differs");

        assert_eq!(
            error,
            SumcheckR1csError::WrongNumberOfRounds {
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn rejects_degree_above_shape_bound() {
        let shape = SumcheckShape::new(1, 1);
        let rounds = [round(2, 2)];
        let mut builder = R1csBuilder::<Fr>::new();

        let error = allocate_sumcheck_r1cs_layout(&mut builder, shape, &rounds)
            .expect_err("degree exceeds bound");

        assert_eq!(
            error,
            SumcheckR1csError::DegreeBoundExceeded {
                round_index: 0,
                bound: 1,
                actual: 2,
            }
        );
    }

    #[test]
    fn append_rejects_broken_claim_chain() {
        let shape = SumcheckShape::new(1, 1);
        let rounds = [round(1, 2)];
        let mut builder = R1csBuilder::<Fr>::new();

        let mut layout = allocate_sumcheck_r1cs_layout(&mut builder, shape, &rounds)
            .expect("layout should allocate");
        layout.rounds[0].claim_in = layout.rounds[0].claim_out;

        let error = append_sumcheck_r1cs_constraints(&mut builder, shape, &rounds, &layout)
            .expect_err("claim chain is broken");

        assert_eq!(
            error,
            SumcheckR1csError::RoundClaimLinkMismatch {
                round_index: 0,
                expected: layout.input_claim,
                actual: layout.rounds[0].claim_in,
            }
        );
    }

    #[test]
    fn append_rejects_out_of_bounds_layout_variable() {
        let shape = SumcheckShape::new(1, 1);
        let rounds = [round(1, 2)];
        let mut builder = R1csBuilder::<Fr>::new();

        let mut layout = allocate_sumcheck_r1cs_layout(&mut builder, shape, &rounds)
            .expect("layout should allocate");
        let num_vars = builder.num_vars();
        layout.rounds[0].coefficients[0] = Variable(num_vars);

        let error = append_sumcheck_r1cs_constraints(&mut builder, shape, &rounds, &layout)
            .expect_err("layout references an unknown variable");

        assert_eq!(
            error,
            SumcheckR1csError::LayoutVariableOutOfBounds {
                variable: Variable(num_vars),
                num_vars,
            }
        );
    }
}
