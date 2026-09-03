use jolt_field::{Fr, Ring, Zero};
use jolt_poly::{BindingOrder, EqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;
use rayon::prelude::*;

use super::{
    gate_value, grouped_relation, RelationTable, RelationTableWitness, ACTIVE, DEGREE,
    FIXED_COLUMNS, H_ID, H_SIGMA, SIGMA_A, TOTAL_COLUMNS,
};

pub struct RelationTableProver {
    columns: [Polynomial<Fr>; TOTAL_COLUMNS],
    identity: Polynomial<Fr>,
    eq: Polynomial<Fr>,
    beta: Fr,
    gamma: Fr,
    relation_weights: [Fr; 3],
    rounds: usize,
    input_claim: Fr,
}

impl RelationTableProver {
    pub fn new(
        table: &RelationTable,
        witness: &RelationTableWitness,
        tau: Vec<Fr>,
        beta: Fr,
        gamma: Fr,
        relation_weights: [Fr; 3],
    ) -> Self {
        assert_eq!(tau.len(), table.rows.trailing_zeros() as usize);
        let evaluations: [Vec<Fr>; TOTAL_COLUMNS] = std::array::from_fn(|column| {
            if column < FIXED_COLUMNS {
                table.fixed[column].clone()
            } else {
                witness.columns[column - FIXED_COLUMNS].clone()
            }
        });
        let identity: Vec<Fr> = (0..table.rows)
            .map(|row| Fr::from_u64(row as u64))
            .collect();
        let eq = EqPolynomial::<Fr>::evals(&tau, None);
        let input_claim = (0..table.rows)
            .into_par_iter()
            .map(|row| {
                let columns = std::array::from_fn(|column| evaluations[column][row]);
                relation_value(
                    &columns,
                    identity[row],
                    eq[row],
                    table.rows,
                    beta,
                    gamma,
                    relation_weights,
                )
            })
            .sum();
        Self {
            columns: evaluations.map(Polynomial::new),
            identity: Polynomial::new(identity),
            eq: Polynomial::new(eq),
            beta,
            gamma,
            relation_weights,
            rounds: tau.len(),
            input_claim,
        }
    }

    pub fn input_claim(&self) -> Fr {
        self.input_claim
    }

    pub fn claims(&self) -> [Fr; TOTAL_COLUMNS] {
        std::array::from_fn(|column| self.columns[column].evals()[0])
    }

    fn bind(&mut self, challenge: Fr) {
        for polynomial in self
            .columns
            .iter_mut()
            .chain(std::iter::once(&mut self.identity))
            .chain(std::iter::once(&mut self.eq))
        {
            polynomial.bind_with_order(challenge, BindingOrder::HighToLow);
        }
    }
}

impl ProveRounds<Fr> for RelationTableProver {
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
        let half = self.eq.len() / 2;
        let evaluations = (0..half)
            .into_par_iter()
            .map(|index| {
                std::array::from_fn::<_, { DEGREE + 1 }, _>(|x| {
                    let x = Fr::from_u64(x as u64);
                    let columns = std::array::from_fn(|column| {
                        self.columns[column].sumcheck_round_eval_with_order(
                            index,
                            x,
                            BindingOrder::HighToLow,
                        )
                    });
                    relation_value(
                        &columns,
                        self.identity.sumcheck_round_eval_with_order(
                            index,
                            x,
                            BindingOrder::HighToLow,
                        ),
                        self.eq
                            .sumcheck_round_eval_with_order(index, x, BindingOrder::HighToLow),
                        1 << self.rounds,
                        self.beta,
                        self.gamma,
                        self.relation_weights,
                    )
                })
            })
            .reduce(
                || [Fr::zero(); DEGREE + 1],
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

fn relation_value(
    columns: &[Fr; TOTAL_COLUMNS],
    identity: Fr,
    eq: Fr,
    rows: usize,
    beta: Fr,
    gamma: Fr,
    relation_weights: [Fr; 3],
) -> Fr {
    let fixed: [Fr; FIXED_COLUMNS] = columns[..FIXED_COLUMNS]
        .try_into()
        .unwrap_or_else(|_| unreachable!("fixed column count"));
    let wires = [
        columns[FIXED_COLUMNS],
        columns[FIXED_COLUMNS + 1],
        columns[FIXED_COLUMNS + 2],
    ];
    let ids = std::array::from_fn(|wire| Fr::from_u64((wire * rows) as u64) + identity);
    let sigmas = [fixed[SIGMA_A], fixed[SIGMA_A + 1], fixed[SIGMA_A + 2]];
    eq * gate_value(
        fixed[..5]
            .try_into()
            .unwrap_or_else(|_| unreachable!("gate selector count")),
        wires,
    ) + relation_weights[0]
        * eq
        * grouped_relation(wires, ids, fixed[ACTIVE], columns[H_ID], beta, gamma)
        + relation_weights[1]
            * eq
            * grouped_relation(wires, sigmas, fixed[ACTIVE], columns[H_SIGMA], beta, gamma)
        + relation_weights[2] * (columns[H_ID] - columns[H_SIGMA])
}
