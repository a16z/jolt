use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial},
        unipoly::UniPoly,
    },
    utils::math::Math,
};

#[inline]
fn mle_eval_diamond_optimized_single_index<
    'a,
    F: JoltField,
    I: IntoParallelRefIterator<'a, Item = &'a MultilinearPolynomial<F>> + Sync + ?Sized,
>(
    polys: &'a I,
    binding_order: BindingOrder,
    length: usize,
    degree: usize,
    index: usize,
) -> Vec<F> {
    assert!(
        length.is_power_of_two(),
        "Length must be a power of two, length: {length}",
    );
    assert!(
        degree.is_power_of_two(),
        "Degree must be a power of two, degree: {degree}",
    );

    let c: usize = degree.log_2();

    let mut table: Vec<Vec<F>> = polys
        .par_iter()
        .map(|poly| {
            // TODO: clean this up.
            match binding_order {
                BindingOrder::LowToHigh => {
                    vec![
                        poly.get_bound_coeff(2 * index),
                        poly.get_bound_coeff(2 * index + 1),
                    ]
                }
                BindingOrder::HighToLow => {
                    vec![
                        poly.get_bound_coeff(index),
                        poly.get_bound_coeff(index + length / 2),
                    ]
                }
            }
        })
        .collect();

    for i in 0..c {
        // Extrapolate everything in the list from length 2^i + 1 -> 2^(i+1) + 1
        table.iter_mut().for_each(|row| {
            let univariate_poly = UniPoly::from_evals(row);
            row.extend(
                (i.pow2() + 1..(i + 1).pow2() + 1)
                    .map(|val| univariate_poly.evaluate(&F::from_u64(val as u64))),
            );

            #[cfg(test)]
            {
                assert_eq!(row.len(), (i + 1).pow2() + 1);
            }
        });

        // For each pair of polynomials — there will be 2^(c-i-1) pairs — multiply them together pointwise
        for idx in 0..(table.len() / 2) {
            table[idx] = table[2 * idx]
                .iter()
                .zip(table[2 * idx + 1].iter())
                .map(|(a, b)| *a * b)
                .collect();
        }
        table.truncate(table.len() / 2);
    }

    #[cfg(test)]
    {
        assert_eq!(table.len(), 1);
    }

    let evals = table
        .into_iter()
        .next()
        .expect("Should contain one element.");

    assert_eq!(evals.len(), degree + 1);
    evals
}

/// Implement the MLE evaluation optimization described in https://hackmd.io/@benediamond/Sye_bB1wJl.
/// Computes g_i(X) in the paper.
/// Arguments:
/// - `polys`: Input polynomials.
/// - `r_j`: The value to evaluate the polynomials at.
/// - `binding_order`: The binding order of the polynomials.
/// - `length`: The length of each input polynomial.
/// - `degree`: The degree of the result univariate polynomial to evaluate, which is the number of input polynomials.
pub fn mle_eval_diamond_optimized_pow2<
    'a,
    F: JoltField,
    I: IntoParallelRefIterator<'a, Item = &'a MultilinearPolynomial<F>> + Sync,
>(
    polys: &'a I,
    binding_order: BindingOrder,
    length: usize,
    degree: usize,
) -> UniPoly<F> {
    let evals = (0..length / 2)
        .into_par_iter()
        .map(|j| mle_eval_diamond_optimized_single_index(polys, binding_order, length, degree, j))
        .reduce(
            || vec![F::zero(); degree + 1],
            |running, new| {
                running
                    .iter()
                    .zip(new.iter())
                    .map(|(a, b)| *a + b)
                    .collect()
            },
        );

    UniPoly::from_evals(&evals)
}

pub fn mle_eval_diamond_optimized<
    'a,
    F: JoltField,
    I: IntoParallelRefIterator<'a, Item = &'a MultilinearPolynomial<F>> + Sync + 'a + ?Sized,
    IterI: IntoParallelRefIterator<'a, Item = &'a (&'a I, usize)> + Sync + 'a + ?Sized,
>(
    polys: &'a IterI,
    binding_order: BindingOrder,
    length: usize,
    degree: usize,
) -> UniPoly<F> {
    let evals = (0..length / 2)
        .into_par_iter()
        .map(|j| {
            polys
                .par_iter()
                .map(|(polys, sub_deg)| {
                    let mut evals = mle_eval_diamond_optimized_single_index(
                        *polys,
                        binding_order,
                        length,
                        *sub_deg,
                        j,
                    );
                    // Each polynomial has sub_deg < degree. We need to first extrapolate the polynomials to the highest degree.
                    // TODO: can do some optimization if sub_deg is two.
                    let univariate_poly = UniPoly::from_evals(&evals);
                    evals.extend(
                        (evals.len()..degree + 1)
                            .map(|j| univariate_poly.evaluate(&F::from_u32(j as u32))),
                    );
                    evals
                })
                .reduce(
                    || vec![F::one(); degree + 1],
                    |running, new| {
                        running
                            .iter()
                            .zip(new.iter())
                            .map(|(a, b)| *a * b)
                            .collect()
                    },
                )
        })
        .reduce(
            || vec![F::zero(); degree + 1],
            |running, new| {
                running
                    .iter()
                    .zip(new.iter())
                    .map(|(a, b)| *a + b)
                    .collect()
            },
        );

    UniPoly::from_evals(&evals)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_core::RngCore;
    use rayon::iter::{
        IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
        IntoParallelRefMutIterator, ParallelIterator,
    };

    use crate::{
        field::JoltField,
        poly::{
            multilinear_polynomial::{
                BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
            },
            unipoly::UniPoly,
        },
        subprotocols::mle_eval::{mle_eval_diamond_optimized, mle_eval_diamond_optimized_pow2},
        utils::math::Math,
    };

    /// Computes g_i(X) the univariate polynomial in the i-th round of a sumcheck protocol from product of input MLEs in the naive way.
    /// Arguments:
    /// - `polys`: Input polynomials.
    /// - `r_j`: The value to evaluate the polynomials at.
    /// - `binding_order`: The binding order of the polynomials.
    /// - `length`: The length of each input polynomial.
    /// - `degree`: The degree of the result univariate polynomial to evaluate, which is the number of input polynomials.
    fn mle_eval_naive<
        'a,
        F: JoltField,
        I: IntoParallelRefIterator<'a, Item = &'a MultilinearPolynomial<F>> + Sync,
    >(
        polys: &'a I,
        prev_claim: F,
        binding_order: BindingOrder,
        length: usize,
        degree: usize,
    ) -> UniPoly<F> {
        assert_ne!(degree, 0, "No polynomials to evaluate");

        let mut g_i = (0..length / 2)
            .into_par_iter()
            .map(|i| {
                let res = polys
                    .par_iter()
                    .map(|poly| {
                        let evals = poly.sumcheck_evals(i, degree, binding_order);
                        evals
                    })
                    .reduce(
                        || vec![F::one(); degree],
                        |running, new| {
                            running
                                .iter()
                                .zip(new.iter())
                                .map(|(a, b)| *a * *b)
                                .collect()
                        },
                    );
                res
            })
            .reduce(
                || vec![F::zero(); degree],
                |running, new| {
                    running
                        .iter()
                        .zip(new.iter())
                        .map(|(a, b)| *a + *b)
                        .collect()
                },
            );

        g_i.insert(1, prev_claim - g_i[0]);
        assert_eq!(g_i.len(), degree + 1);

        UniPoly::from_evals(&g_i)
    }

    fn test_data(num_vars: usize, num_polys: usize, rng: &mut impl RngCore) -> Vec<Vec<Fr>> {
        (0..num_polys)
            .map(|_| {
                (0..num_vars.pow2())
                    .map(|_| Fr::from_u32(rng.next_u32()))
                    .collect()
            })
            .collect()
    }

    fn test_polys_and_prev_claim(test_data: &[Vec<Fr>]) -> (Vec<MultilinearPolynomial<Fr>>, Fr) {
        let test_polys = test_data
            .iter()
            .map(|data| MultilinearPolynomial::<Fr>::from(data.clone()))
            .collect::<Vec<_>>();

        let prev_claim = (0..test_polys[0].len())
            .into_par_iter()
            .map(|i| {
                test_polys
                    .par_iter()
                    .map(|poly| poly.get_coeff(i))
                    .reduce(|| Fr::from_u32(1), |running, new| running * new)
            })
            .reduce(|| Fr::from_u32(0), |running, new| running + new);

        (test_polys, prev_claim)
    }

    #[test]
    fn diamond_optimization_correctness_pow2() {
        // Create test data.
        let num_vars = 3;
        let num_polys = 4;
        let mut rng = test_rng();
        let test_data = test_data(num_vars, num_polys, &mut rng);

        for binding_order in [BindingOrder::LowToHigh, BindingOrder::HighToLow] {
            let (mut test_polys, mut prev_claim) = test_polys_and_prev_claim(&test_data);
            for round in 0..num_vars {
                let length = test_polys[0].len();
                let r_j = Fr::from(rng.next_u32());
                let unipoly =
                    mle_eval_naive(&test_polys, prev_claim, binding_order, length, num_polys);
                let unipoly_optimized =
                    mle_eval_diamond_optimized_pow2(&test_polys, binding_order, length, num_polys);
                assert_eq!(
                    unipoly.evaluate(&r_j),
                    unipoly_optimized.evaluate(&r_j),
                    "Fails in round: {:?}, binding order: {:?}",
                    round,
                    binding_order
                );

                test_polys.par_iter_mut().for_each(|poly| {
                    poly.bind_parallel(r_j, binding_order);
                });
                prev_claim = unipoly.evaluate(&r_j);
            }
        }
    }

    #[test]
    fn diamond_optimization_correctness_non_pow2() {
        let num_vars = 3;
        let num_polys = 6;
        let mut rng = test_rng();
        let test_data = test_data(num_vars, num_polys, &mut rng);

        for binding_order in [BindingOrder::LowToHigh, BindingOrder::HighToLow] {
            let (mut test_polys, mut prev_claim) = test_polys_and_prev_claim(&test_data);
            for round in 0..num_vars {
                let length = test_polys[0].len();
                let r_j = Fr::from(rng.next_u32());

                let now = Instant::now();
                let unipoly =
                    mle_eval_naive(&test_polys, prev_claim, binding_order, length, num_polys);
                let duration = now.elapsed();
                println!("Naive time: {:?}", duration);

                let (arr1, arr2) = test_polys.split_at(4);
                assert_eq!(arr1.len(), 4);
                assert_eq!(arr2.len(), num_polys - 4);

                let now = Instant::now();
                let unipoly_optimized = mle_eval_diamond_optimized(
                    &[(arr1, arr1.len()), (arr2, arr2.len())],
                    binding_order,
                    length,
                    num_polys,
                );
                let duration = now.elapsed();
                println!("Optimized time: {:?}", duration);

                assert_eq!(
                    unipoly.evaluate(&r_j),
                    unipoly_optimized.evaluate(&r_j),
                    "Fails in round: {:?}, binding order: {:?}",
                    round,
                    binding_order
                );

                test_polys.par_iter_mut().for_each(|poly| {
                    poly.bind_parallel(r_j, binding_order);
                });
                prev_claim = unipoly.evaluate(&r_j);
            }
        }
    }
}
