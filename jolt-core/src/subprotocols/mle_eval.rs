use std::time::{Duration, Instant};

use rayon::iter::{
    IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial},
        unipoly::UniPoly,
    },
    utils::math::Math,
};

pub fn mle_eval_diamond_optimized_single_idx<
    'a,
    F: JoltField,
    I: IntoParallelRefIterator<'a, Item = &'a MultilinearPolynomial<F>> + Sync,
>(
    polys: &'a I,
    binding_order: BindingOrder,
    length: usize,
    degree: usize,
    j: usize,
) {
    assert!(length.is_power_of_two(), "Length must be a power of two");
    assert!(degree.is_power_of_two(), "Degree must be a power of two");
    let c: usize = degree.log_2();

    let now = Instant::now();

    let mut table: Vec<Vec<F>> = polys
        .par_iter()
        .map(|poly| {
            // TODO: clean this up.
            match binding_order {
                BindingOrder::LowToHigh => {
                    vec![poly.get_bound_coeff(2 * j), poly.get_bound_coeff(2 * j + 1)]
                }
                BindingOrder::HighToLow => {
                    vec![
                        poly.get_bound_coeff(j),
                        poly.get_bound_coeff(j + length / 2),
                    ]
                }
            }
        })
        .collect();

    let mut map_time = Duration::from_secs(0);
    let mut reduced_time = Duration::from_secs(0);

    for i in 0..c {
        let temp = Instant::now();

        // TODO: This is the bottleneck.                
        // Extrapolate everything in the list from length 2^i + 1 -> 2^(i+1) + 1
        // Eventually each row will have 2^(i+1) + 1 elements.
        table.par_iter_mut().for_each(|row: &mut Vec<F>| {
            row.extend(
                (i.pow2() + 1..(i + 1).pow2() + 1)
                    .into_par_iter()
                    .map(|val| UniPoly::eval_with_coeffs(row, &F::from_u64(val as u64)))
                    .collect::<Vec<F>>(),
            ); //.map(|val| univariate_poly.evaluate(&F::from_u64(val as u64))),

            #[cfg(test)]
            {
                assert_eq!(row.len(), (i + 1).pow2() + 1);
            }
        });

        map_time += temp.elapsed();

        let temp = Instant::now();
        table = (0..(table.len() / 2))
            .into_par_iter()
            .map(|idx| {
                table[2 * idx]
                    .iter()
                    .zip(table[2 * idx + 1].iter())
                    .map(|(a, b)| *a * b)
                    .collect::<Vec<F>>()
            })
            .collect::<Vec<Vec<F>>>();
        reduced_time += temp.elapsed();

        // // For each pair of polynomials — there will be 2^(c-i-1) pairs — multiply them together pointwise
        // for idx in 0..(table.len() / 2) {
        //     table[idx] = table[2 * idx]
        //         .iter()
        //         .zip(table[2 * idx + 1].iter())
        //         .map(|(a, b)| *a * b)
        //         .collect();
        // }
        // table.truncate(table.len() / 2);
    }

    let duration = now.elapsed();
    println!("Diamond optimized single idx time: {:?}", duration);
    println!("Map time: {:?}", map_time);
    println!("Reduced time: {:?}", reduced_time);
}

/// Implement the MLE evaluation optimization described in https://hackmd.io/@benediamond/Sye_bB1wJl.
/// Computes g_i(X) in the paper.
/// Arguments:
/// - `polys`: Input polynomials.
/// - `r_j`: The value to evaluate the polynomials at.
/// - `binding_order`: The binding order of the polynomials.
/// - `length`: The length of each input polynomial.
/// - `degree`: The degree of the result univariate polynomial to evaluate, which is the number of input polynomials.
pub fn mle_eval_diamond_optimized<
    'a,
    F: JoltField,
    I: IntoParallelRefIterator<'a, Item = &'a MultilinearPolynomial<F>> + Sync,
>(
    polys: &'a I,
    r_j: F,
    binding_order: BindingOrder,
    length: usize,
    degree: usize,
) -> F {
    assert!(length.is_power_of_two(), "Length must be a power of two");
    assert!(degree.is_power_of_two(), "Degree must be a power of two");
    let c: usize = degree.log_2();

    let evals = (0..length / 2)
        .into_par_iter()
        .map(|j| {
            let mut table: Vec<Vec<F>> = polys
                .par_iter()
                .map(|poly| {
                    // TODO: clean this up.
                    match binding_order {
                        BindingOrder::LowToHigh => {
                            vec![poly.get_bound_coeff(2 * j), poly.get_bound_coeff(2 * j + 1)]
                        }
                        BindingOrder::HighToLow => {
                            vec![
                                poly.get_bound_coeff(j),
                                poly.get_bound_coeff(j + length / 2),
                            ]
                        }
                    }
                })
                .collect();

            for i in 0..c {
                // Extrapolate everything in the list from length 2^i + 1 -> 2^(i+1) + 1
                table.par_iter_mut().for_each(|row: &mut Vec<F>| {
                    let univariate_poly = UniPoly::from_evals(row);
                    row.extend(
                        (i.pow2() + 1..(i + 1).pow2() + 1)
                            .into_par_iter()
                            .map(|val| {
                                let res = univariate_poly.evaluate(&F::from_u64(val as u64));
                                res
                            })
                            .collect::<Vec<F>>(),
                    ); //.map(|val| univariate_poly.evaluate(&F::from_u64(val as u64))),

                    #[cfg(test)]
                    {
                        assert_eq!(row.len(), (i + 1).pow2() + 1);
                    }
                });

                table = (0..(table.len() / 2))
                    .into_par_iter()
                    .map(|idx| {
                        table[2 * idx]
                            .iter()
                            .zip(table[2 * idx + 1].iter())
                            .map(|(a, b)| *a * b)
                            .collect::<Vec<F>>()
                    })
                    .collect::<Vec<Vec<F>>>();

                // // For each pair of polynomials — there will be 2^(c-i-1) pairs — multiply them together pointwise
                // for idx in 0..(table.len() / 2) {
                //     table[idx] = table[2 * idx]
                //         .iter()
                //         .zip(table[2 * idx + 1].iter())
                //         .map(|(a, b)| *a * b)
                //         .collect();
                // }
                // table.truncate(table.len() / 2);
            }

            #[cfg(test)]
            {
                assert_eq!(table.len(), 1);
            }

            table
                .into_iter()
                .next()
                .expect("Should contain one element.")
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

    #[cfg(test)]
    {
        assert_eq!(evals.len(), degree + 1);
    }

    let univariate_poly = UniPoly::from_evals(&evals);
    univariate_poly.evaluate(&r_j)
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use ark_bn254::Fr;
    use ark_std::test_rng;
    use rand_core::RngCore;
    use rayon::iter::{
        IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
    };

    use crate::{
        field::JoltField,
        poly::{
            multilinear_polynomial::{
                BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
            },
            unipoly::UniPoly,
        },
        subprotocols::mle_eval::{
            mle_eval_diamond_optimized, mle_eval_diamond_optimized_single_idx,
        },
        utils::math::Math,
    };

    fn mle_eval_naive_single_idx<
        'a,
        F: JoltField,
        I: IntoParallelRefIterator<'a, Item = &'a MultilinearPolynomial<F>> + Sync,
    >(
        polys: &'a I,
        r_j: F,
        prev_claim: F,
        binding_order: BindingOrder,
        length: usize,
        degree: usize,
        idx: usize,
    ) {
        assert_ne!(degree, 0, "No polynomials to evaluate");

        let now = Instant::now();
        let res = polys
            .par_iter()
            .map(|poly| {
                let evals = poly.sumcheck_evals(idx, degree, binding_order);
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
        let duration = now.elapsed();
        println!("Sumcheck naive evals time: {:?}", duration);
    }

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
        r_j: F,
        prev_claim: F,
        binding_order: BindingOrder,
        length: usize,
        degree: usize,
    ) -> F {
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

        let univariate_poly = UniPoly::from_evals(&g_i);
        univariate_poly.evaluate(&r_j)
    }

    #[test]
    fn diamond_optimization_correctness() {
        // Create test data.
        let num_vars = 4;
        let num_polys = 16;
        let mut rng = test_rng();

        let test_data: Vec<Vec<Fr>> = (0..num_polys)
            .map(|_| {
                (0..num_vars.pow2())
                    .map(|_| Fr::from_u32(rng.next_u32()))
                    .collect()
            })
            .collect();

        for binding_order in [BindingOrder::LowToHigh, BindingOrder::HighToLow] {
            let mut test_polys = test_data
                .iter()
                .map(|data| MultilinearPolynomial::<Fr>::from(data.clone()))
                .collect::<Vec<_>>();

            let mut prev_claim = (0..test_polys[0].len())
                .into_par_iter()
                .map(|i| {
                    test_polys
                        .par_iter()
                        .map(|poly| poly.get_coeff(i))
                        .reduce(|| Fr::from_u32(1), |running, new| running * new)
                })
                .reduce(|| Fr::from_u32(0), |running, new| running + new);

            for round in 0..num_vars {
                let length = test_polys[0].len();
                let r_j = Fr::from(rng.next_u32());

                mle_eval_naive_single_idx(
                    &test_polys,
                    r_j,
                    prev_claim,
                    binding_order,
                    length,
                    num_polys,
                    0,
                );

                let now = Instant::now();
                let result_optimized = mle_eval_diamond_optimized_single_idx(
                    &test_polys,
                    binding_order,
                    length,
                    num_polys,
                    0,
                );
                let _duration = now.elapsed();
                println!("Optimized time: {:?}", _duration);

                // let now = Instant::now();
                // let result_optimized =
                //     mle_eval_diamond_optimized(&test_polys, r_j, binding_order, length, num_polys);
                // let _duration = now.elapsed();
                // println!("Optimized time: {:?}", _duration);

                // let naive_start = Instant::now();
                // let result = mle_eval_naive(
                //     &test_polys,
                //     r_j,
                //     prev_claim,
                //     binding_order,
                //     length,
                //     num_polys,
                // );
                // let _duration = naive_start.elapsed();
                // println!("Naive time: {:?}", _duration);

                panic!();

                // assert_eq!(
                //     result, result_optimized,
                //     "Fails in round: {:?}, binding order: {:?}",
                //     round, binding_order
                // );

                // test_polys.par_iter_mut().for_each(|poly| {
                //     poly.bind_parallel(r_j, binding_order);
                // });
                // prev_claim = result;
            }
        }
    }
}
