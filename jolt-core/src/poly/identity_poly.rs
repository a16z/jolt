use crate::field::JoltField;

use crate::utils::math::Math;

use super::multilinear_polynomial::{BindingOrder, PolynomialBinding, PolynomialEvaluation};

pub struct IdentityPolynomial<F: JoltField> {
    num_vars: usize,
    num_bound_vars: usize,
    bound_value: F,
}

impl<F: JoltField> IdentityPolynomial<F> {
    pub fn new(num_vars: usize) -> Self {
        IdentityPolynomial {
            num_vars,
            num_bound_vars: 0,
            bound_value: F::zero(),
        }
    }
}

impl<F: JoltField> PolynomialBinding<F> for IdentityPolynomial<F> {
    fn is_bound(&self) -> bool {
        self.num_bound_vars != 0
    }

    fn bind(&mut self, r: F, order: BindingOrder) {
        debug_assert!(self.num_bound_vars < self.num_vars);
        debug_assert_eq!(
            order,
            BindingOrder::LowToHigh,
            "IdentityPolynomial only supports low-to-high binding"
        );

        self.bound_value += F::from_u32(1u32 << self.num_bound_vars) * r;
        self.num_bound_vars += 1;
    }

    fn bind_parallel(&mut self, r: F, order: BindingOrder) {
        // Binding is constant time, no parallelism necessary
        self.bind(r, order);
    }

    fn final_sumcheck_claim(&self) -> F {
        debug_assert_eq!(self.num_vars, self.num_bound_vars);
        self.bound_value
    }
}

impl<F: JoltField> PolynomialEvaluation<F> for IdentityPolynomial<F> {
    fn evaluate(&self, r: &[F]) -> F {
        let len = r.len();
        assert_eq!(len, self.num_vars);
        (0..len)
            .map(|i| F::from_u64((len - i - 1).pow2() as u64) * r[i])
            .sum()
    }

    fn batch_evaluate(_polys: &[&Self], _r: &[F]) -> (Vec<F>, Vec<F>) {
        unimplemented!("Currently unused")
    }

    fn sumcheck_evals(&self, index: usize, degree: usize, order: BindingOrder) -> Vec<F> {
        debug_assert!(degree > 0);
        debug_assert!(index < self.num_vars.pow2() / 2);
        debug_assert_eq!(
            order,
            BindingOrder::LowToHigh,
            "IdentityPolynomial only supports low-to-high binding"
        );

        let mut evals = vec![F::zero(); degree];
        evals[0] = self.bound_value + F::from_u64((index as u64) << (1 + self.num_bound_vars));
        let m = F::from_u32(1 << self.num_bound_vars);
        let mut eval = evals[0] + m;
        for i in 1..degree {
            eval += m;
            evals[i] = eval;
        }
        evals
    }
}

#[cfg(test)]
mod tests {
    use crate::poly::multilinear_polynomial::MultilinearPolynomial;

    use super::*;
    use ark_bn254::Fr;
    use ark_std::test_rng;

    #[test]
    fn identity_poly() {
        const NUM_VARS: usize = 10;

        let mut rng = test_rng();
        let mut identity_poly: IdentityPolynomial<Fr> = IdentityPolynomial::new(NUM_VARS);
        let mut reference_poly: MultilinearPolynomial<Fr> =
            MultilinearPolynomial::from((0..(1 << NUM_VARS)).map(|i| i as u32).collect::<Vec<_>>());

        for j in 0..reference_poly.len() / 2 {
            let identity_poly_evals = identity_poly.sumcheck_evals(j, 3, BindingOrder::LowToHigh);
            let reference_poly_evals = reference_poly.sumcheck_evals(j, 3, BindingOrder::LowToHigh);
            assert_eq!(identity_poly_evals, reference_poly_evals);
        }

        for _ in 0..NUM_VARS {
            let r = Fr::random(&mut rng);
            identity_poly.bind(r, BindingOrder::LowToHigh);
            reference_poly.bind(r, BindingOrder::LowToHigh);
            for j in 0..reference_poly.len() / 2 {
                let identity_poly_evals =
                    identity_poly.sumcheck_evals(j, 3, BindingOrder::LowToHigh);
                let reference_poly_evals =
                    reference_poly.sumcheck_evals(j, 3, BindingOrder::LowToHigh);
                assert_eq!(identity_poly_evals, reference_poly_evals);
            }
        }

        assert_eq!(
            identity_poly.final_sumcheck_claim(),
            reference_poly.final_sumcheck_claim()
        );
    }
}
