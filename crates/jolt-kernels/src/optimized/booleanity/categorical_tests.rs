use crate::optimized::parity::ExceptionalEq;
#[cfg(feature = "akita")]
use jolt_field::Prime128Offset6A23801F;
use jolt_field::{Fr, JoltField, Ring, Zero};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial};

use super::{CategoricalProducts, ChunkIndexSource, LazyFoldedRa};
use crate::reference::views::eq_table;

struct Source(Vec<Vec<Option<usize>>>, usize);

impl ChunkIndexSource for Source {
    fn constant_suffix_start(&self) -> usize {
        self.1
    }
    fn num_columns(&self) -> usize {
        self.0.len()
    }
    fn cycles(&self) -> usize {
        self.0[0].len()
    }
    fn index(&self, i: usize, j: usize) -> Option<usize> {
        self.0[i][j]
    }
}

fn check_categorical_cubics<F: JoltField>(gamma: F) {
    for prefix in [0, 1, 7, 9, 17, 31, 32, 127, 128] {
        check_categorical_case(gamma, None, prefix);
        for case in ExceptionalEq::ALL {
            check_categorical_case(gamma, Some(case), prefix);
        }
    }
}

fn check_categorical_case<F: JoltField>(
    gamma: F,
    exceptional: Option<ExceptionalEq>,
    prefix_len: usize,
) {
    // Keep the 16-address fixture lazy through its first bind.
    let log_t = 7;
    for addresses in [2, 16, 256] {
        for bind in [F::zero(), F::one(), F::from_u64(13)] {
            // All-cold, alternating hot/cold, and fully hot families.
            // In particular None is not address zero, and a folded pair
            // can have only its low or only its high branch cold.
            let source = Source(
                (0..3)
                    .map(|family| {
                        (0..1 << log_t)
                            .map(|row| {
                                if row >= prefix_len {
                                    (family != 0).then_some(family % addresses)
                                } else if family == 0 || (family == 1 && row % 3 == 0) {
                                    None
                                } else {
                                    Some((row * row + 7 * row + family) % addresses)
                                }
                            })
                            .collect()
                    })
                    .collect(),
                prefix_len,
            );
            let rho: Vec<F> = [F::one(), gamma, gamma * gamma].to_vec();
            let unscaled: Vec<Vec<F>> = (0..3)
                .map(|_| {
                    (0..addresses)
                        .map(|i| F::from_u64((i * i + 3 * i + 11) as u64))
                        .collect()
                })
                .collect();
            let tables: Vec<Vec<F>> = unscaled
                .iter()
                .zip(&rho)
                .map(|(table, rho)| {
                    let scale = if rho.is_zero() { F::one() } else { *rho };
                    table.iter().map(|value| *value * scale).collect()
                })
                .collect();
            let mut dense: Vec<Vec<F>> = source
                .0
                .iter()
                .zip(&unscaled)
                .map(|(column, table)| {
                    column
                        .iter()
                        .map(|i| i.map_or(F::zero(), |i| table[i]))
                        .collect()
                })
                .collect();
            let mut lazy = LazyFoldedRa::new(tables, source);
            let reference: Vec<F> = exceptional.map_or_else(
                || {
                    (0..log_t)
                        .map(|i| F::from_u64((7 + 3 * i) as u64))
                        .collect()
                },
                |case| case.point(log_t, bind),
            );
            let mut prefix = F::from_u64(19);
            let mut eq = GruenSplitEqPolynomial::new_with_scaling(
                &reference,
                BindingOrder::LowToHigh,
                Some(prefix),
            );
            for round in 0..if addresses <= 16 { 2 } else { 1 } {
                let suffix = eq_table(&reference[..log_t - 1 - round]);
                let bit = reference[log_t - 1 - round];
                // Direct evaluations of the defining weighted cubic;
                // neither Gruen reconstruction nor coefficient caching.
                let direct: Vec<F> = (0..4)
                    .map(|x| {
                        let x = F::from_u64(x);
                        let mut sum = F::zero();
                        for (row, weight) in suffix.iter().enumerate() {
                            for (column, r) in dense.iter().zip(&rho) {
                                let v =
                                    column[2 * row] + x * (column[2 * row + 1] - column[2 * row]);
                                sum += *weight * *r * *r * v * (v - F::one());
                            }
                        }
                        prefix * (bit * x + (F::one() - bit) * (F::one() - x)) * sum
                    })
                    .collect();
                let LazyFoldedRa::Lazy {
                    tables,
                    width,
                    factors: 1,
                    source,
                    ..
                } = &lazy
                else {
                    unreachable!()
                };
                let prefix_pairs = prefix_len.div_ceil(1 << (round + 1));
                let tail_value: F = dense
                    .iter()
                    .zip(&rho)
                    .map(|(column, rho)| {
                        let value = column[column.len() - 1];
                        *rho * *rho * value * (value - F::one())
                    })
                    .sum();
                let tail_mass = suffix[prefix_pairs..].iter().copied().sum::<F>() * tail_value;
                let polynomials = [
                    CategoricalProducts::<F, _, false>::new(tables, *width, source, &rho)
                        .lookup(&eq, round, direct[0] + direct[1], prefix_pairs, tail_mass)
                        .unwrap(),
                    CategoricalProducts::<F, _, true>::new(tables, *width, source, &rho)
                        .lookup(&eq, round, direct[0] + direct[1], prefix_pairs, tail_mass)
                        .unwrap(),
                ];
                for polynomial in polynomials {
                    for (x, value) in direct.iter().enumerate() {
                        assert_eq!(polynomial.evaluate(F::from_u64(x as u64)), *value);
                    }
                }
                for column in &mut dense {
                    for row in 0..column.len() / 2 {
                        column[row] =
                            column[2 * row] + bind * (column[2 * row + 1] - column[2 * row]);
                    }
                    column.truncate(column.len() / 2);
                }
                lazy.bind(bind);
                eq.round_mut().unwrap().bind(bind);
                prefix *= bit * bind + (F::one() - bit) * (F::one() - bind);
            }
        }
    }
}

#[test]
fn categorical_booleanity_matches_direct_cubic_dory() {
    check_categorical_cubics::<Fr>(Fr::zero());
    check_categorical_cubics::<Fr>(Fr::from_u64(3));
}

#[test]
#[cfg(feature = "akita")]
fn categorical_booleanity_matches_direct_cubic_akita() {
    check_categorical_cubics::<Prime128Offset6A23801F>(Prime128Offset6A23801F::zero());
    check_categorical_cubics::<Prime128Offset6A23801F>(Prime128Offset6A23801F::from_u64(3));
}
