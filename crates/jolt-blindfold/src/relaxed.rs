use jolt_crypto::HomomorphicCommitment;
use jolt_field::Field;

use crate::RelaxedError;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelaxedInstance<F, Com> {
    pub u: F,
    pub witness_row_commitments: Vec<Com>,
    pub error_row_commitments: Vec<Com>,
    pub eval_commitments: Vec<Com>,
}

impl<F, Com> RelaxedInstance<F, Com> {
    pub fn new(
        u: F,
        witness_row_commitments: Vec<Com>,
        error_row_commitments: Vec<Com>,
        eval_commitments: Vec<Com>,
    ) -> Self {
        Self {
            u,
            witness_row_commitments,
            error_row_commitments,
            eval_commitments,
        }
    }
}

impl<F, Com> RelaxedInstance<F, Com>
where
    F: Field,
    Com: HomomorphicCommitment<F>,
{
    pub fn fold(
        &self,
        random: &Self,
        cross_term_error_row_commitments: &[Com],
        folding_challenge: F,
    ) -> Result<Self, RelaxedError> {
        ensure_len(
            "random witness row commitments",
            self.witness_row_commitments.len(),
            random.witness_row_commitments.len(),
        )?;
        ensure_len(
            "random error row commitments",
            self.error_row_commitments.len(),
            random.error_row_commitments.len(),
        )?;
        ensure_len(
            "cross-term error row commitments",
            self.error_row_commitments.len(),
            cross_term_error_row_commitments.len(),
        )?;
        ensure_len(
            "random eval commitments",
            self.eval_commitments.len(),
            random.eval_commitments.len(),
        )?;

        let r_squared = folding_challenge * folding_challenge;
        let u = self.u + folding_challenge * random.u;

        let witness_row_commitments = self
            .witness_row_commitments
            .iter()
            .zip(&random.witness_row_commitments)
            .map(|(real, random)| Com::linear_combine(real, random, &folding_challenge))
            .collect();
        let error_row_commitments = self
            .error_row_commitments
            .iter()
            .zip(cross_term_error_row_commitments)
            .zip(&random.error_row_commitments)
            .map(|((real, cross_term), random)| {
                let with_cross = Com::linear_combine(real, cross_term, &folding_challenge);
                Com::linear_combine(&with_cross, random, &r_squared)
            })
            .collect();
        let eval_commitments = self
            .eval_commitments
            .iter()
            .zip(&random.eval_commitments)
            .map(|(real, random)| Com::linear_combine(real, random, &folding_challenge))
            .collect();

        Ok(Self {
            u,
            witness_row_commitments,
            error_row_commitments,
            eval_commitments,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RelaxedWitness<F> {
    pub witness_rows: Vec<F>,
    pub witness_row_blindings: Vec<F>,
    pub error_rows: Vec<F>,
    pub error_row_blindings: Vec<F>,
}

impl<F> RelaxedWitness<F> {
    pub fn new(
        witness_rows: Vec<F>,
        witness_row_blindings: Vec<F>,
        error_rows: Vec<F>,
        error_row_blindings: Vec<F>,
    ) -> Self {
        Self {
            witness_rows,
            witness_row_blindings,
            error_rows,
            error_row_blindings,
        }
    }
}

impl<F: Field> RelaxedWitness<F> {
    pub fn fold(
        &self,
        random: &Self,
        cross_term_error_rows: &[F],
        cross_term_error_row_blindings: &[F],
        folding_challenge: F,
    ) -> Result<Self, RelaxedError> {
        ensure_len(
            "random witness rows",
            self.witness_rows.len(),
            random.witness_rows.len(),
        )?;
        ensure_len(
            "random witness row blindings",
            self.witness_row_blindings.len(),
            random.witness_row_blindings.len(),
        )?;
        ensure_len(
            "random error rows",
            self.error_rows.len(),
            random.error_rows.len(),
        )?;
        ensure_len(
            "cross-term error rows",
            self.error_rows.len(),
            cross_term_error_rows.len(),
        )?;
        ensure_len(
            "random error row blindings",
            self.error_row_blindings.len(),
            random.error_row_blindings.len(),
        )?;
        ensure_len(
            "cross-term error row blindings",
            self.error_row_blindings.len(),
            cross_term_error_row_blindings.len(),
        )?;

        let r_squared = folding_challenge * folding_challenge;
        let witness_rows = self
            .witness_rows
            .iter()
            .zip(&random.witness_rows)
            .map(|(&real, &random)| real + folding_challenge * random)
            .collect();
        let witness_row_blindings = self
            .witness_row_blindings
            .iter()
            .zip(&random.witness_row_blindings)
            .map(|(&real, &random)| real + folding_challenge * random)
            .collect();
        let error_rows = self
            .error_rows
            .iter()
            .zip(cross_term_error_rows)
            .zip(&random.error_rows)
            .map(|((&real, &cross_term), &random)| {
                real + folding_challenge * cross_term + r_squared * random
            })
            .collect();
        let error_row_blindings = self
            .error_row_blindings
            .iter()
            .zip(cross_term_error_row_blindings)
            .zip(&random.error_row_blindings)
            .map(|((&real, &cross_term), &random)| {
                real + folding_challenge * cross_term + r_squared * random
            })
            .collect();

        Ok(Self {
            witness_rows,
            witness_row_blindings,
            error_rows,
            error_row_blindings,
        })
    }
}

fn ensure_len(name: &'static str, expected: usize, actual: usize) -> Result<(), RelaxedError> {
    if expected != actual {
        return Err(RelaxedError::LengthMismatch {
            name,
            expected,
            actual,
        });
    }
    Ok(())
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "tests should fail loudly"
)]
mod tests {
    use super::*;
    use jolt_crypto::{Bn254, Bn254G1, JoltGroup, Pedersen, PedersenSetup, VectorCommitment};
    use jolt_field::{Fr, FromPrimitiveInt};

    fn f(value: u64) -> Fr {
        Fr::from_u64(value)
    }

    fn pedersen_setup() -> PedersenSetup<Bn254G1> {
        let generator = Bn254::g1_generator();
        PedersenSetup::new(vec![generator], generator.scalar_mul(&f(99)))
    }

    fn c(setup: &PedersenSetup<Bn254G1>, value: Fr, blinding: Fr) -> Bn254G1 {
        Pedersen::<Bn254G1>::commit(setup, &[value], &blinding)
    }

    #[test]
    fn folds_relaxed_instance() {
        let setup = pedersen_setup();
        let real = RelaxedInstance::new(
            f(2),
            vec![c(&setup, f(10), f(100)), c(&setup, f(20), f(200))],
            vec![c(&setup, f(30), f(300)), c(&setup, f(40), f(400))],
            vec![c(&setup, f(50), f(500))],
        );
        let random = RelaxedInstance::new(
            f(3),
            vec![c(&setup, f(1), f(10)), c(&setup, f(2), f(20))],
            vec![c(&setup, f(3), f(30)), c(&setup, f(4), f(40))],
            vec![c(&setup, f(5), f(50))],
        );
        let cross_terms = [c(&setup, f(7), f(70)), c(&setup, f(8), f(80))];
        let folded = real
            .fold(&random, &cross_terms, f(11))
            .expect("fold dimensions match");

        assert_eq!(folded.u, f(2) + f(11) * f(3));
        assert_eq!(
            folded.witness_row_commitments,
            vec![
                c(&setup, f(10) + f(11), f(100) + f(110)),
                c(&setup, f(20) + f(22), f(200) + f(220)),
            ]
        );
        assert_eq!(
            folded.error_row_commitments,
            vec![
                c(
                    &setup,
                    f(30) + f(11) * f(7) + f(121) * f(3),
                    f(300) + f(11) * f(70) + f(121) * f(30),
                ),
                c(
                    &setup,
                    f(40) + f(11) * f(8) + f(121) * f(4),
                    f(400) + f(11) * f(80) + f(121) * f(40),
                ),
            ]
        );
        assert_eq!(
            folded.eval_commitments,
            vec![c(&setup, f(50) + f(55), f(500) + f(550))]
        );
    }

    #[test]
    fn folds_relaxed_witness() {
        let real = RelaxedWitness::new(
            vec![f(10), f(20)],
            vec![f(30), f(40)],
            vec![f(50), f(60)],
            vec![f(70), f(80)],
        );
        let random = RelaxedWitness::new(
            vec![f(1), f(2)],
            vec![f(3), f(4)],
            vec![f(5), f(6)],
            vec![f(7), f(8)],
        );
        let folded = real
            .fold(&random, &[f(9), f(10)], &[f(11), f(12)], f(13))
            .expect("fold dimensions match");

        assert_eq!(folded.witness_rows, vec![f(10 + 13), f(20 + 26)]);
        assert_eq!(folded.witness_row_blindings, vec![f(30 + 39), f(40 + 52)]);
        assert_eq!(
            folded.error_rows,
            vec![
                f(50) + f(13) * f(9) + f(169) * f(5),
                f(60) + f(13) * f(10) + f(169) * f(6),
            ]
        );
        assert_eq!(
            folded.error_row_blindings,
            vec![
                f(70) + f(13) * f(11) + f(169) * f(7),
                f(80) + f(13) * f(12) + f(169) * f(8),
            ]
        );
    }

    #[test]
    fn rejects_instance_length_mismatch() {
        let setup = pedersen_setup();
        let real = RelaxedInstance::new(
            f(1),
            vec![c(&setup, f(1), f(10)), c(&setup, f(2), f(20))],
            vec![c(&setup, f(3), f(30))],
            vec![c(&setup, f(4), f(40))],
        );
        let random = RelaxedInstance::new(
            f(1),
            vec![c(&setup, f(1), f(10))],
            vec![c(&setup, f(3), f(30))],
            vec![c(&setup, f(4), f(40))],
        );

        let error = real
            .fold(&random, &[c(&setup, f(5), f(50))], f(2))
            .unwrap_err();

        assert_eq!(
            error,
            RelaxedError::LengthMismatch {
                name: "random witness row commitments",
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn rejects_cross_term_length_mismatch() {
        let setup = pedersen_setup();
        let real = RelaxedInstance::new(
            f(1),
            vec![c(&setup, f(1), f(10))],
            vec![c(&setup, f(2), f(20)), c(&setup, f(3), f(30))],
            vec![],
        );
        let random = RelaxedInstance::new(
            f(1),
            vec![c(&setup, f(1), f(10))],
            vec![c(&setup, f(2), f(20)), c(&setup, f(3), f(30))],
            vec![],
        );

        let error = real
            .fold(&random, &[c(&setup, f(5), f(50))], f(2))
            .unwrap_err();

        assert_eq!(
            error,
            RelaxedError::LengthMismatch {
                name: "cross-term error row commitments",
                expected: 2,
                actual: 1,
            }
        );
    }

    #[test]
    fn rejects_witness_length_mismatch() {
        let real = RelaxedWitness::new(vec![f(1)], vec![f(2)], vec![f(3), f(4)], vec![f(5)]);
        let random = RelaxedWitness::new(vec![f(1)], vec![f(2)], vec![f(3), f(4)], vec![f(5)]);

        let error = real.fold(&random, &[f(6)], &[f(7)], f(2)).unwrap_err();

        assert_eq!(
            error,
            RelaxedError::LengthMismatch {
                name: "cross-term error rows",
                expected: 2,
                actual: 1,
            }
        );
    }
}
