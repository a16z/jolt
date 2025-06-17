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

/// Polynomial that unmaps RAM addresses: k -> (k-1)*4 + input_start for k > 0, and 0 for k = 0
pub struct UnmapRamAddressPolynomial<F: JoltField> {
    num_vars: usize,
    num_bound_vars: usize,
    input_start: u64,
    /// Accumulator for the linear part: 4 * sum_{i=0}^{bound_vars-1} 2^i * r_i
    linear_term: F,
    /// Accumulator for the product part: prod_{i=0}^{bound_vars-1} (1 - r_i)
    product_term: F,
}

impl<F: JoltField> UnmapRamAddressPolynomial<F> {
    pub fn new(num_vars: usize, input_start: u64) -> Self {
        UnmapRamAddressPolynomial {
            num_vars,
            num_bound_vars: 0,
            input_start,
            linear_term: F::zero(),
            product_term: F::one(),
        }
    }
}

impl<F: JoltField> PolynomialBinding<F> for UnmapRamAddressPolynomial<F> {
    fn is_bound(&self) -> bool {
        self.num_bound_vars != 0
    }

    fn bind(&mut self, r: F, order: BindingOrder) {
        debug_assert!(self.num_bound_vars < self.num_vars);
        debug_assert_eq!(
            order,
            BindingOrder::LowToHigh,
            "UnmapAddressPolynomial only supports low-to-high binding"
        );

        // Update the linear term: add 4 * 2^i * r_i
        self.linear_term += F::from_u64(4 * (1u64 << self.num_bound_vars)) * r;

        // Update the product term: multiply by (1 - r_i)
        self.product_term *= F::one() - r;

        self.num_bound_vars += 1;
    }

    fn bind_parallel(&mut self, r: F, order: BindingOrder) {
        // Binding is constant time, no parallelism necessary
        self.bind(r, order);
    }

    fn final_sumcheck_claim(&self) -> F {
        debug_assert_eq!(self.num_vars, self.num_bound_vars);
        self.linear_term + F::from_u64(self.input_start - 4) * (F::one() - self.product_term)
    }
}

impl<F: JoltField> PolynomialEvaluation<F> for UnmapRamAddressPolynomial<F> {
    fn evaluate(&self, r: &[F]) -> F {
        let len = r.len();
        assert_eq!(len, self.num_vars);

        // Compute sum_{i=0}^{len-1} 2^i * r_i
        let mut sum = F::zero();
        let mut power_of_two = F::one();
        for &r_i in r.iter() {
            sum += power_of_two * r_i;
            power_of_two = power_of_two + power_of_two;
        }

        // Compute prod_{i=0}^{len-1} (1 - r_i)
        let mut prod = F::one();
        for &r_i in r.iter() {
            prod *= F::one() - r_i;
        }

        // Return 4 * sum + (input_start - 4) * (1 - prod)
        F::from_u64(4) * sum + F::from_u64(self.input_start - 4) * (F::one() - prod)
    }

    fn batch_evaluate(_polys: &[&Self], _r: &[F]) -> (Vec<F>, Vec<F>) {
        unimplemented!("Unused")
    }

    fn sumcheck_evals(&self, index: usize, degree: usize, order: BindingOrder) -> Vec<F> {
        debug_assert!(degree > 0);
        debug_assert!(index < self.num_vars.pow2() / 2);
        debug_assert_eq!(
            order,
            BindingOrder::LowToHigh,
            "UnmapAddressPolynomial only supports low-to-high binding"
        );

        let mut linear_evals = vec![F::zero(); degree];
        let mut product_evals = vec![F::one(); degree];

        linear_evals[0] =
            self.linear_term + F::from_u64((4 * index as u64) << (self.num_bound_vars + 1));
        product_evals[0] = if index == 0 {
            self.product_term
        } else {
            // if at least one bit of index is non-zero => whole prod is 0
            F::zero()
        };

        let m = F::from_u32(4 * (1 << self.num_bound_vars));
        let mut linear_eval = linear_evals[0] + m;
        for i in 1..degree {
            // Evaluate at point i+1
            linear_eval += m;
            linear_evals[i] = linear_eval;
            product_evals[i] = if index == 0 {
                self.product_term * (-F::from_u32(i as u32))
            } else {
                F::zero()
            };
        }

        linear_evals
            .into_iter()
            .zip(product_evals)
            .map(|(l, p)| l + F::from_u64(self.input_start - 4) * (F::one() - p))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use crate::poly::multilinear_polynomial::MultilinearPolynomial;

    use super::*;
    use ark_bn254::Fr;
    use ark_ec::AdditiveGroup;
    use ark_ff::Field;
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

    #[test]
    fn unmap_address_poly_evaluate() {
        const NUM_VARS: usize = 10;
        const INPUT_START: u64 = 0x80000000;

        // Test that UnmapAddressPolynomial evaluates correctly on boolean hypercube
        let unmap_poly = UnmapRamAddressPolynomial::<Fr>::new(NUM_VARS, INPUT_START);

        // Test a few specific points
        // k=0 should map to 0
        let point_0 = vec![Fr::ZERO; NUM_VARS];
        assert_eq!(unmap_poly.evaluate(&point_0), Fr::ZERO);

        // k=1 should map to input_start
        let mut point_1 = vec![Fr::ZERO; NUM_VARS];
        point_1[0] = Fr::ONE;
        assert_eq!(unmap_poly.evaluate(&point_1), Fr::from(INPUT_START));

        // k=2 should map to input_start + 4
        let mut point_2 = vec![Fr::ZERO; NUM_VARS];
        point_2[1] = Fr::ONE;
        assert_eq!(unmap_poly.evaluate(&point_2), Fr::from(INPUT_START + 4));

        // k=3 should map to input_start + 8
        let mut point_3 = vec![Fr::ZERO; NUM_VARS];
        point_3[0] = Fr::ONE;
        point_3[1] = Fr::ONE;
        assert_eq!(unmap_poly.evaluate(&point_3), Fr::from(INPUT_START + 8));
    }

    #[test]
    fn unmap_address_poly() {
        const NUM_VARS: usize = 4;
        const INPUT_START: u64 = 0x80000000;

        let mut unmap_poly = UnmapRamAddressPolynomial::<Fr>::new(NUM_VARS, INPUT_START);

        let K = 1 << NUM_VARS;
        let unmap_evals: Vec<Fr> = (0..K)
            .map(|k| {
                if k == 0 {
                    Fr::ZERO
                } else {
                    Fr::from((k as u64 - 1) * 4 + INPUT_START)
                }
            })
            .collect();
        let mut reference_poly = MultilinearPolynomial::from(unmap_evals.clone());

        for round in 0..NUM_VARS {
            let num_evals = 1 << (NUM_VARS - round - 1);

            for i in 0..num_evals {
                let unmap_evals = unmap_poly.sumcheck_evals(i, 3, BindingOrder::LowToHigh);
                let reference_evals = reference_poly.sumcheck_evals(i, 3, BindingOrder::LowToHigh);

                assert_eq!(
                    unmap_evals, reference_evals,
                    "Round {round}, index {i}: sumcheck_evals mismatch",
                );
            }

            let r = Fr::from(0x12345678u64 + round as u64);
            unmap_poly.bind(r, BindingOrder::LowToHigh);
            reference_poly.bind(r, BindingOrder::LowToHigh);
        }

        assert_eq!(
            unmap_poly.final_sumcheck_claim(),
            reference_poly.final_sumcheck_claim(),
            "Final sumcheck claims don't match"
        );
    }
}
