use ark_bn254::Fr as ArkFr;
use ark_ff::batch_inversion;
use jolt_field::{Fr, Ring, Zero};
use jolt_hyperkzg::{NoopVerifierObserver, VerifierObserver};
use jolt_poly::{BindingOrder, EqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;
use rayon::prelude::*;

use super::{RelationTableError, DEGREE, WIRES};

/// VK columns describing three link slots per row. `ids` are logical edge
/// identifiers shared by the two sides; selectors disable unused slots.
#[derive(Clone, Debug)]
pub struct CopyLinkSide {
    pub selectors: [Vec<Fr>; WIRES],
    pub ids: [Vec<Fr>; WIRES],
}

impl CopyLinkSide {
    pub fn new(
        selectors: [Vec<Fr>; WIRES],
        ids: [Vec<Fr>; WIRES],
    ) -> Result<Self, RelationTableError> {
        let rows = selectors[0].len();
        if !rows.is_power_of_two()
            || selectors
                .iter()
                .chain(&ids)
                .any(|column| column.len() != rows)
        {
            return Err(RelationTableError::RowDomain {
                minimum: rows.next_power_of_two(),
                actual: rows,
            });
        }
        Ok(Self { selectors, ids })
    }
}

#[derive(Clone, Debug)]
pub struct CopyLink {
    pub left: CopyLinkSide,
    pub right: CopyLinkSide,
    rows: usize,
}

impl CopyLink {
    pub fn new(left: CopyLinkSide, right: CopyLinkSide) -> Result<Self, RelationTableError> {
        let rows = left.selectors[0].len();
        if right.selectors[0].len() != rows {
            return Err(RelationTableError::RowDomain {
                minimum: rows,
                actual: right.selectors[0].len(),
            });
        }
        Ok(Self { left, right, rows })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn witness(
        &self,
        left_values: [Vec<Fr>; WIRES],
        right_values: [Vec<Fr>; WIRES],
        beta: Fr,
        gamma: Fr,
    ) -> Result<CopyLinkWitness, RelationTableError> {
        if left_values
            .iter()
            .chain(&right_values)
            .any(|column| column.len() != self.rows)
        {
            return Err(RelationTableError::Claims);
        }
        let mut denominators = Vec::new();
        let mut positions = Vec::new();
        for row in 0..self.rows {
            for (wire, ((values, ids), selectors)) in left_values
                .iter()
                .zip(&self.left.ids)
                .zip(&self.left.selectors)
                .enumerate()
            {
                if !selectors[row].is_zero() {
                    denominators.push(gamma + values[row] + beta * ids[row]);
                    positions.push((0, row, wire));
                }
            }
            for (wire, ((values, ids), selectors)) in right_values
                .iter()
                .zip(&self.right.ids)
                .zip(&self.right.selectors)
                .enumerate()
            {
                if !selectors[row].is_zero() {
                    denominators.push(gamma + values[row] + beta * ids[row]);
                    positions.push((1, row, wire));
                }
            }
        }
        if denominators.iter().any(Zero::is_zero) {
            return Err(RelationTableError::ZeroDenominator);
        }
        let mut inverses: Vec<ArkFr> = denominators.iter().copied().map(ArkFr::from).collect();
        batch_inversion(&mut inverses);
        let mut helpers = [vec![Fr::zero(); self.rows], vec![Fr::zero(); self.rows]];
        for (&(side, row, wire), inverse) in positions.iter().zip(inverses) {
            let selectors = if side == 0 {
                &self.left.selectors
            } else {
                &self.right.selectors
            };
            helpers[side][row] += selectors[wire][row] * Fr::from(inverse);
        }
        Ok(CopyLinkWitness {
            left_values,
            right_values,
            helpers,
        })
    }

    pub fn check(
        &self,
        witness: &CopyLinkWitness,
        beta: Fr,
        gamma: Fr,
    ) -> Result<(), RelationTableError> {
        let mut sum = Fr::zero();
        for row in 0..self.rows {
            for (side, values, helper) in [
                (&self.left, &witness.left_values, witness.helpers[0][row]),
                (&self.right, &witness.right_values, witness.helpers[1][row]),
            ] {
                let values = std::array::from_fn(|wire| values[wire][row]);
                let ids = std::array::from_fn(|wire| side.ids[wire][row]);
                let selectors = std::array::from_fn(|wire| side.selectors[wire][row]);
                if grouped_selected_relation(values, ids, selectors, helper, beta, gamma)
                    != Fr::zero()
                {
                    return Err(RelationTableError::Copy);
                }
            }
            sum += witness.helpers[0][row] - witness.helpers[1][row];
        }
        if sum.is_zero() {
            Ok(())
        } else {
            Err(RelationTableError::Copy)
        }
    }

    pub fn prover(
        &self,
        witness: &CopyLinkWitness,
        tau: Vec<Fr>,
        beta: Fr,
        gamma: Fr,
        weights: [Fr; 3],
    ) -> CopyLinkProver {
        CopyLinkProver::new(self, witness, tau, beta, gamma, weights)
    }

    #[cfg(test)]
    pub fn final_value(
        beta: Fr,
        gamma: Fr,
        weights: [Fr; 3],
        eq: Fr,
        claims: &CopyLinkClaims,
    ) -> Fr {
        copy_link_value_observed(beta, gamma, weights, eq, claims, &mut NoopVerifierObserver)
    }

    #[cfg(test)]
    pub fn final_value_observed<O: VerifierObserver>(
        beta: Fr,
        gamma: Fr,
        weights: [Fr; 3],
        eq: Fr,
        claims: &CopyLinkClaims,
        observer: &mut O,
    ) -> Fr {
        copy_link_value_observed(beta, gamma, weights, eq, claims, observer)
    }
}

pub struct CopyLinkWitness {
    pub left_values: [Vec<Fr>; WIRES],
    pub right_values: [Vec<Fr>; WIRES],
    pub helpers: [Vec<Fr>; 2],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CopyLinkClaims {
    pub left_selectors: [Fr; WIRES],
    pub left_ids: [Fr; WIRES],
    pub left_values: [Fr; WIRES],
    pub right_selectors: [Fr; WIRES],
    pub right_ids: [Fr; WIRES],
    pub right_values: [Fr; WIRES],
    pub helpers: [Fr; 2],
}

const COPY_COLUMNS: usize = 20;

pub struct CopyLinkProver {
    columns: [Polynomial<Fr>; COPY_COLUMNS],
    eq: Polynomial<Fr>,
    beta: Fr,
    gamma: Fr,
    weights: [Fr; 3],
    rounds: usize,
    input_claim: Fr,
}

impl CopyLinkProver {
    fn new(
        link: &CopyLink,
        witness: &CopyLinkWitness,
        tau: Vec<Fr>,
        beta: Fr,
        gamma: Fr,
        weights: [Fr; 3],
    ) -> Self {
        assert_eq!(tau.len(), link.rows.trailing_zeros() as usize);
        let values: [Vec<Fr>; COPY_COLUMNS] = std::array::from_fn(|column| match column {
            0..=2 => link.left.selectors[column].clone(),
            3..=5 => link.left.ids[column - 3].clone(),
            6..=8 => witness.left_values[column - 6].clone(),
            9..=11 => link.right.selectors[column - 9].clone(),
            12..=14 => link.right.ids[column - 12].clone(),
            15..=17 => witness.right_values[column - 15].clone(),
            18..=19 => witness.helpers[column - 18].clone(),
            _ => unreachable!("copy column count"),
        });
        let eq = EqPolynomial::<Fr>::evals(&tau, None);
        let input_claim = (0..link.rows)
            .into_par_iter()
            .map(|row| {
                let claims = claims_at(&values, row);
                copy_link_value(beta, gamma, weights, eq[row], &claims)
            })
            .sum();
        Self {
            columns: values.map(Polynomial::new),
            eq: Polynomial::new(eq),
            beta,
            gamma,
            weights,
            rounds: tau.len(),
            input_claim,
        }
    }

    pub fn input_claim(&self) -> Fr {
        self.input_claim
    }

    pub fn claims(&self) -> CopyLinkClaims {
        let values: [Fr; COPY_COLUMNS] =
            std::array::from_fn(|column| self.columns[column].evals()[0]);
        claims_from_values(&values)
    }

    fn bind(&mut self, challenge: Fr) {
        for column in &mut self.columns {
            column.bind_with_order(challenge, BindingOrder::HighToLow);
        }
        self.eq.bind_with_order(challenge, BindingOrder::HighToLow);
    }
}

impl ProveRounds<Fr> for CopyLinkProver {
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
        let evaluations = (0..self.eq.len() / 2)
            .into_par_iter()
            .map(|index| {
                let mut local = [Fr::zero(); DEGREE + 1];
                for (x, value) in local.iter_mut().enumerate() {
                    let x = Fr::from_u64(x as u64);
                    let columns: [Fr; COPY_COLUMNS] = std::array::from_fn(|column| {
                        self.columns[column].sumcheck_round_eval_with_order(
                            index,
                            x,
                            BindingOrder::HighToLow,
                        )
                    });
                    let eq =
                        self.eq
                            .sumcheck_round_eval_with_order(index, x, BindingOrder::HighToLow);
                    *value = copy_link_value(
                        self.beta,
                        self.gamma,
                        self.weights,
                        eq,
                        &claims_from_values(&columns),
                    );
                }
                local
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

fn claims_at(values: &[Vec<Fr>; COPY_COLUMNS], row: usize) -> CopyLinkClaims {
    let row_values: [Fr; COPY_COLUMNS] = std::array::from_fn(|column| values[column][row]);
    claims_from_values(&row_values)
}

fn claims_from_values(values: &[Fr; COPY_COLUMNS]) -> CopyLinkClaims {
    CopyLinkClaims {
        left_selectors: [values[0], values[1], values[2]],
        left_ids: [values[3], values[4], values[5]],
        left_values: [values[6], values[7], values[8]],
        right_selectors: [values[9], values[10], values[11]],
        right_ids: [values[12], values[13], values[14]],
        right_values: [values[15], values[16], values[17]],
        helpers: [values[18], values[19]],
    }
}

fn copy_link_value(beta: Fr, gamma: Fr, weights: [Fr; 3], eq: Fr, claims: &CopyLinkClaims) -> Fr {
    copy_link_value_observed(beta, gamma, weights, eq, claims, &mut NoopVerifierObserver)
}

fn copy_link_value_observed<O: VerifierObserver>(
    beta: Fr,
    gamma: Fr,
    weights: [Fr; 3],
    eq: Fr,
    claims: &CopyLinkClaims,
    observer: &mut O,
) -> Fr {
    let left = grouped_selected_relation_observed(
        claims.left_values,
        claims.left_ids,
        claims.left_selectors,
        claims.helpers[0],
        beta,
        gamma,
        observer,
    );
    let right = grouped_selected_relation_observed(
        claims.right_values,
        claims.right_ids,
        claims.right_selectors,
        claims.helpers[1],
        beta,
        gamma,
        observer,
    );
    let eq_left = observer.fr_mul(eq, left);
    let left = observer.fr_mul(weights[0], eq_left);
    let eq_right = observer.fr_mul(eq, right);
    let right = observer.fr_mul(weights[1], eq_right);
    left + right + observer.fr_mul(weights[2], claims.helpers[0] - claims.helpers[1])
}

fn grouped_selected_relation(
    values: [Fr; WIRES],
    ids: [Fr; WIRES],
    selectors: [Fr; WIRES],
    helper: Fr,
    beta: Fr,
    gamma: Fr,
) -> Fr {
    grouped_selected_relation_observed(
        values,
        ids,
        selectors,
        helper,
        beta,
        gamma,
        &mut NoopVerifierObserver,
    )
}

fn grouped_selected_relation_observed<O: VerifierObserver>(
    values: [Fr; WIRES],
    ids: [Fr; WIRES],
    selectors: [Fr; WIRES],
    helper: Fr,
    beta: Fr,
    gamma: Fr,
    observer: &mut O,
) -> Fr {
    let denominators: [Fr; WIRES] =
        std::array::from_fn(|i| gamma + values[i] + observer.fr_mul(beta, ids[i]));
    let product01 = observer.fr_mul(denominators[0], denominators[1]);
    let product = observer.fr_mul(product01, denominators[2]);
    let pair12 = observer.fr_mul(denominators[1], denominators[2]);
    let selected0 = observer.fr_mul(selectors[0], pair12);
    let pair02 = observer.fr_mul(denominators[0], denominators[2]);
    let selected1 = observer.fr_mul(selectors[1], pair02);
    let pair01 = observer.fr_mul(denominators[0], denominators[1]);
    let selected2 = observer.fr_mul(selectors[2], pair01);
    observer.fr_mul(helper, product) - selected0 - selected1 - selected2
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests fail on invalid synthetic links")]
mod tests {
    use jolt_field::{Fr, One, Zero};
    use jolt_poly::EqPolynomial;
    use jolt_sumcheck::prover::ProveRounds;

    use super::*;
    use crate::relation_table::{
        evaluate_terms_observed, AffineForm, CopyLinkTermExporter, CopyLinkTermSide,
        CopyLinkTermsContext, TermContext, TermExporter,
    };
    use crate::stream::VerifierCost;

    #[test]
    fn synthetic_link_accepts_permutation_and_rejects_value_change() {
        let rows = 8;
        let selectors = std::array::from_fn(|wire| {
            (0..rows)
                .map(|row| Fr::from_u64(u64::from(row < 3 && wire == 0)))
                .collect()
        });
        let left_ids =
            std::array::from_fn(|_| (0..rows).map(|row| Fr::from_u64(row as u64)).collect());
        let right_ids = left_ids.clone();
        let left = CopyLinkSide::new(selectors.clone(), left_ids).unwrap();
        let right = CopyLinkSide::new(selectors, right_ids).unwrap();
        let link = CopyLink::new(left, right).unwrap();
        let values = std::array::from_fn(|wire| {
            (0..rows)
                .map(|row| Fr::from_u64((10 * wire + row) as u64))
                .collect()
        });
        let beta = Fr::from_u64(19);
        let gamma = Fr::from_u64(23);
        let witness = link
            .witness(values.clone(), values.clone(), beta, gamma)
            .unwrap();
        link.check(&witness, beta, gamma).unwrap();
        let tau = vec![Fr::from_u64(3), Fr::from_u64(5), Fr::from_u64(7)];
        let weights = [Fr::one(), Fr::from_u64(11), Fr::from_u64(13)];
        let mut prover = link.prover(&witness, tau.clone(), beta, gamma, weights);
        assert_eq!(prover.input_claim(), Fr::zero());
        let mut claim = prover.input_claim();
        let mut point = Vec::new();
        let mut bind = None;
        for round in 0..prover.num_rounds() {
            let polynomial = prover.prove_round(bind, round, claim).unwrap();
            let challenge = Fr::from_u64((29 + round) as u64);
            claim = polynomial.evaluate(challenge);
            point.push(challenge);
            bind = Some(challenge);
        }
        prover.finish_rounds(bind.unwrap()).unwrap();
        let claims = prover.claims();
        let final_claim = CopyLink::final_value(
            beta,
            gamma,
            weights,
            EqPolynomial::<Fr>::mle(&tau, &point),
            &claims,
        );
        assert_eq!(claim, final_claim);
        let mut cost = VerifierCost::default();
        assert_eq!(
            final_claim,
            CopyLink::final_value_observed(
                beta,
                gamma,
                weights,
                EqPolynomial::<Fr>::mle(&tau, &point),
                &claims,
                &mut cost,
            )
        );
        assert_eq!(cost.fr_mul, 29);

        let column_claims = [
            claims.left_selectors.as_slice(),
            claims.left_ids.as_slice(),
            claims.left_values.as_slice(),
            claims.right_selectors.as_slice(),
            claims.right_ids.as_slice(),
            claims.right_values.as_slice(),
            claims.helpers.as_slice(),
        ]
        .concat();
        let column = |slot| crate::relation_table::ColumnId { group: 0, slot };
        let form_columns = |base| {
            std::array::from_fn(|wire| AffineForm {
                constant: Fr::zero(),
                weights: vec![(column(base + wire), Fr::one())],
            })
        };
        let term_context = CopyLinkTermsContext {
            left: CopyLinkTermSide {
                selectors: [column(0), column(1), column(2)],
                ids: form_columns(3),
                values: form_columns(6),
                helper: column(18),
            },
            right: CopyLinkTermSide {
                selectors: [column(9), column(10), column(11)],
                ids: form_columns(12),
                values: form_columns(15),
                helper: column(19),
            },
            beta,
            gamma,
            eq: EqPolynomial::<Fr>::mle(&tau, &point),
            relation_weights: weights,
            stage_coefficient: Fr::from_u64(31),
        };
        let exporter = CopyLinkTermExporter {
            link: &link,
            left: term_context.left.clone(),
            right: term_context.right.clone(),
            tau: &tau,
            beta,
            gamma,
            relation_weights: weights,
            member_index: 0,
        };
        let export_context = TermContext {
            row_point: &point,
            batching_coefficients: &[term_context.stage_coefficient],
            challenges: &[],
        };
        let mut term_cost = VerifierCost::default();
        let terms = exporter.terms_observed(&export_context, &mut term_cost);
        assert_eq!(exporter.terms(&export_context), terms);
        assert_eq!(link.terms(&term_context), terms);
        assert_eq!(term_cost.fr_mul, 13);
        assert_eq!(terms.len(), crate::relation_table::COPY_LINK_TERM_COUNT);
        assert_eq!(
            terms.iter().map(|term| term.factors.len()).max(),
            Some(crate::relation_table::MAX_FACTORS)
        );
        assert_eq!(
            term_context.stage_coefficient * final_claim,
            evaluate_terms_observed(
                &terms,
                &|column| {
                    column_claims
                        .get(column.slot)
                        .copied()
                        .ok_or(RelationTableError::Claims)
                },
                &mut term_cost,
            )
            .unwrap()
        );
        assert_eq!(term_cost.fr_mul, 59);

        let mut right_bad = values.clone();
        right_bad[0][1] += Fr::one();
        let bad = link.witness(values, right_bad, beta, gamma).unwrap();
        assert!(link.check(&bad, beta, gamma).is_err());
    }
}
