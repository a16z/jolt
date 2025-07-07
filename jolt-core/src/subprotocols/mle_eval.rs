use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::{
    field::JoltField,
    poly::{
        multilinear_polynomial::{BindingOrder, MultilinearPolynomial, PolynomialEvaluation},
        unipoly::UniPoly,
    },
    utils::math::Math,
};

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
    polys: &'a mut I,
    r_j: F,
    prev_claim: F,
    binding_order: BindingOrder,
    length: usize,
    degree: usize,
) -> F {
    assert!(length.is_power_of_two(), "Length must be a power of two");
    assert!(degree.is_power_of_two(), "Degree must be a power of two");
    let c: usize = degree.log_2();

    let mut evals = (0..length / 2)
        .into_par_iter()
        .map(|j| {
            let mut table: Vec<Vec<F>> = polys
                .par_iter()
                .map(|poly| {
                    // TODO: clean this up.
                    vec![poly.get_bound_coeff(2 * j), poly.get_bound_coeff(2 * j + 1)]
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
        assert_eq!(evals[0] + evals[1], prev_claim);
    }

    let univariate_poly = UniPoly::from_evals(&evals);
    univariate_poly.evaluate(&r_j)
}

#[cfg(test)]
mod tests {
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
        subprotocols::mle_eval::mle_eval_diamond_optimized,
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
        polys: &'a mut I,
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
        let num_vars = 3;
        let num_polys = 4;
        let mut rng = test_rng();

        let test_data: Vec<Vec<Fr>> = (0..num_polys)
            .map(|_| {
                (0..num_vars.pow2())
                    .map(|_| Fr::from_u32(rng.next_u32()))
                    .collect()
            })
            .collect();

        let mut test_polys = test_data
            .iter()
            .map(|data| MultilinearPolynomial::<Fr>::from(data.clone()))
            .collect::<Vec<_>>();
        let length = test_polys[0].len();

        let prev_claim = (0..length)
            .into_par_iter()
            .map(|i| {
                test_polys
                    .par_iter()
                    .map(|poly| poly.get_coeff(i))
                    .reduce(|| Fr::from_u32(1), |running, new| running * new)
            })
            .reduce(|| Fr::from_u32(0), |running, new| running + new);

        let r_j = Fr::from(rng.next_u32());
        let binding_order = BindingOrder::LowToHigh;

        let result = mle_eval_naive(
            &mut test_polys,
            r_j,
            prev_claim,
            binding_order,
            length,
            num_polys,
        );
        let result_optimized = mle_eval_diamond_optimized(
            &mut test_polys,
            r_j,
            prev_claim,
            binding_order,
            length,
            num_polys,
        );
        assert_eq!(result, result_optimized);

        test_polys.par_iter_mut().for_each(|poly| {
            poly.bind_parallel(r_j, binding_order);
        });
    }
}
