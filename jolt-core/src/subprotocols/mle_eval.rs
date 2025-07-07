/// Implement the MLE evaluation optimization described in https://hackmd.io/@benediamond/Sye_bB1wJl.
/// Computes g_i(X) in the paper.

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
        }, utils::math::Math,
    };

    fn mle_eval_naive<
        'a,
        F: JoltField,
        I: IntoParallelRefIterator<'a, Item = &'a MultilinearPolynomial<F>> + Sync,
    >(
        polys: &'a mut I,
        r_j: F,
        binding_order: BindingOrder,
        length: usize,
        degree: usize,
    ) -> F {
        assert_ne!(degree, 0, "No polynomials to evaluate");
        let g_i = (0..length / 2)
            .into_par_iter()
            .map(|i| {
                let res = polys
                    .par_iter()
                    .map(|poly| {
                        let evals = poly.sumcheck_evals(i, degree, binding_order);
                        evals
                    })
                    .reduce(
                        || vec![F::zero(); degree],
                        |running, new| {
                            running
                                .iter()
                                .zip(new.iter())
                                .map(|(a, b)| *a * b)
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
                        .map(|(a, b)| *a + b)
                        .collect()
                },
            );
        let univariate_poly = UniPoly::from_evals(&g_i);
        univariate_poly.evaluate(&r_j)
    }

    #[test]
    fn naive_evaluation() {
        // Create test data.
        const D: usize = 3;
        let num_polys = 3;
        let mut rng = test_rng();

        let test_data: Vec<Vec<Fr>> = (0..num_polys)
            .map(|_| {
                (0..D.pow2())
                    .map(|_| Fr::from_u32(rng.next_u32()))
                    .collect()
            })
            .collect();

        let mut test_polys = test_data
            .iter()
            .map(|data| MultilinearPolynomial::<Fr>::from(data.clone()))
            .collect::<Vec<_>>();

        let r_j = Fr::from(rng.next_u32());
        let binding_order = BindingOrder::LowToHigh;
        let length = test_polys[0].len();

        let result = mle_eval_naive(&mut test_polys, r_j, binding_order, length, D);
        test_polys.par_iter_mut().for_each(|poly| {
            poly.bind_parallel(r_j, binding_order);
        });
    }
}
