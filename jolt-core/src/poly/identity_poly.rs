use crate::field::JoltField;
use crate::poly::multilinear_polynomial::{
    MultilinearPolynomial, PrefixPolynomial, SuffixPolynomial,
};
use crate::utils::math::Math;
use crate::utils::uninterleave_bits;

use super::multilinear_polynomial::{BindingOrder, PolynomialBinding, PolynomialEvaluation};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Endianness {
    Little,
    Big,
}

pub struct IdentityPolynomial<F: JoltField> {
    num_vars: usize,
    num_bound_vars: usize,
    bound_value: F,
    endianness: Endianness,
}

impl<F: JoltField> IdentityPolynomial<F> {
    pub fn new(num_vars: usize) -> Self {
        Self::new_with_endianness(num_vars, Endianness::Little)
    }

    pub fn new_with_endianness(num_vars: usize, endianness: Endianness) -> Self {
        IdentityPolynomial {
            num_vars,
            num_bound_vars: 0,
            bound_value: F::zero(),
            endianness,
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

        let pow = match self.endianness {
            Endianness::Little => self.num_bound_vars,
            Endianness::Big => self.num_vars - 1 - self.num_bound_vars,
        };
        self.bound_value += F::from_u64(1u64 << pow) * r;
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
        match self.endianness {
            Endianness::Little => (0..len).map(|i| F::from_u64(i.pow2() as u64) * r[i]).sum(),
            Endianness::Big => (0..len)
                .map(|i| F::from_u64((len - 1 - i).pow2() as u64) * r[i])
                .sum(),
        }
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
        let m = match self.endianness {
            Endianness::Little => F::from_u32(1 << self.num_bound_vars),
            Endianness::Big => F::from_u32(1 << (self.num_vars - 1 - self.num_bound_vars)),
        };
        evals[0] = self.bound_value + (m + m).mul_u64(index as u64);
        let mut eval = evals[0] + m;
        for i in 1..degree {
            eval += m;
            evals[i] = eval;
        }
        evals
    }
}

impl<F: JoltField> PrefixPolynomial<F> for IdentityPolynomial<F> {
    type PrefixCheckpoints = F;

    fn prefix_polynomial(&self, prefix_len: usize) -> MultilinearPolynomial<F> {
        assert!(prefix_len % 2 == 0);
        assert_eq!(self.endianness, Endianness::Big);
        let bound_value = self.bound_value.mul_u64(1 << prefix_len);
        MultilinearPolynomial::from(
            (0..prefix_len.pow2())
                .map(|i| bound_value + F::from_u64(i as u64))
                .collect::<Vec<F>>(),
        )
    }

    fn update_checkpoints(&mut self, checkpoints: F) {
        self.bound_value = checkpoints;
    }
}

impl<F: JoltField> SuffixPolynomial<F> for IdentityPolynomial<F> {
    fn suffix_mle(&self, index: u64, suffix_len: usize) -> F {
        assert!(suffix_len % 2 == 0);
        assert_eq!(self.endianness, Endianness::Big);
        F::from_u64(index)
    }
}

pub struct ShiftSuffixPolynomial;
impl<F: JoltField> SuffixPolynomial<F> for ShiftSuffixPolynomial {
    fn suffix_mle(&self, _index: u64, suffix_len: usize) -> F {
        assert!(suffix_len % 2 == 0);
        F::from_u64(1 << suffix_len)
    }
}

/// BatchedUninterleavePolynomial evaluates
/// sum_{i=0}^{num_vars/2-1} (r[2i]  + z * r[2i + 1]) * 2^(num_vars/2 - 1 - i)
pub struct BatchedUninterleavePolynomial<F: JoltField> {
    num_vars: usize,
    num_bound_vars: usize,
    z: F,
    bound_value: F,
    /// Stores the odd variable that's been bound but hasn't updated bound_value yet
    pending_odd_bind: Option<F>,
}

impl<F: JoltField> BatchedUninterleavePolynomial<F> {
    pub fn new(num_vars: usize, z: F) -> Self {
        assert_eq!(num_vars % 2, 0, "num_vars must be divisible by 2");
        BatchedUninterleavePolynomial {
            num_vars,
            num_bound_vars: 0,
            z,
            bound_value: F::zero(),
            pending_odd_bind: None,
        }
    }
}

impl<F: JoltField> PolynomialBinding<F> for BatchedUninterleavePolynomial<F> {
    fn is_bound(&self) -> bool {
        self.num_bound_vars != 0
    }

    fn bind(&mut self, r: F, order: BindingOrder) {
        debug_assert!(self.num_bound_vars < self.num_vars);
        debug_assert_eq!(
            order,
            BindingOrder::HighToLow,
            "BatchedUninterleavePolynomial only supports high-to-low binding"
        );

        if self.num_bound_vars % 2 == 0 {
            // Binding an even variable (r[2i])
            self.pending_odd_bind = Some(r);
        } else {
            // Binding an odd variable (r[2i+1])
            let odd_r = self.pending_odd_bind.take().unwrap();
            self.bound_value += self.bound_value; // multiply by 2
            self.bound_value += odd_r + self.z * r;
        }
        self.num_bound_vars += 1;
    }

    fn bind_parallel(&mut self, r: F, order: BindingOrder) {
        // Binding is constant time, no parallelism necessary
        self.bind(r, order);
    }

    fn final_sumcheck_claim(&self) -> F {
        debug_assert_eq!(self.num_vars, self.num_bound_vars);
        debug_assert!(self.pending_odd_bind.is_none());
        self.bound_value
    }
}

impl<F: JoltField> PolynomialEvaluation<F> for BatchedUninterleavePolynomial<F> {
    fn evaluate(&self, r: &[F]) -> F {
        let len = r.len();
        assert_eq!(len, self.num_vars);
        assert_eq!(len % 2, 0);

        (0..len / 2)
            .map(|i| {
                F::from_u64(1u64 << (self.num_vars / 2 - 1 - i))
                    * (r[2 * i] + self.z * r[2 * i + 1])
            })
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
            BindingOrder::HighToLow,
            "BatchedUninterleavePolynomial only supports high-to-low binding"
        );

        let mut evals = vec![F::zero(); degree];

        let (right, left) = uninterleave_bits(index as u64);
        let comb = F::from_u32(right) + self.z.mul_u64(left as u64);

        if self.num_bound_vars % 2 == 0 {
            assert!(self.pending_odd_bind.is_none());
            let unbound_pairs = (self.num_vars - self.num_bound_vars) / 2;
            assert!(unbound_pairs > 0);
            evals[0] = self.bound_value.mul_u64(1 << unbound_pairs) + comb;
            let m = F::from_u32(1 << (unbound_pairs - 1));
            let mut eval = evals[0] + m;
            for i in 1..degree {
                eval += m;
                evals[i] = eval;
            }
            evals
        } else {
            assert!(self.pending_odd_bind.is_some());
            let odd_r = self.pending_odd_bind.unwrap();
            let unbound_pairs = (self.num_vars - self.num_bound_vars).div_ceil(2);
            evals[0] = self.bound_value.mul_u64(1 << unbound_pairs)
                + comb
                + odd_r.mul_u64(1 << (unbound_pairs - 1));
            let m = self.z.mul_u64(1 << (unbound_pairs - 1));
            let mut eval = evals[0] + m;
            for i in 1..degree {
                eval += m;
                evals[i] = eval;
            }
            evals
        }
    }
}

pub struct ShiftHalfSuffixPolynomial;
impl<F: JoltField> SuffixPolynomial<F> for ShiftHalfSuffixPolynomial {
    fn suffix_mle(&self, _index: u64, suffix_len: usize) -> F {
        assert!(suffix_len % 2 == 0);
        F::from_u64(1 << (suffix_len / 2))
    }
}

impl<F: JoltField> SuffixPolynomial<F> for BatchedUninterleavePolynomial<F> {
    fn suffix_mle(&self, index: u64, suffix_len: usize) -> F {
        assert!(suffix_len % 2 == 0);
        assert!(self.num_bound_vars % 2 == 0);
        let (right, left) = uninterleave_bits(index);
        F::from_u32(right) + self.z.mul_u64(left as u64)
    }
}

impl<F: JoltField> PrefixPolynomial<F> for BatchedUninterleavePolynomial<F> {
    type PrefixCheckpoints = F;

    fn prefix_polynomial(&self, prefix_len: usize) -> MultilinearPolynomial<F> {
        assert!(prefix_len % 2 == 0);
        assert!(self.num_bound_vars % 2 == 0);
        let bound_value = self.bound_value.mul_u64(1 << (prefix_len / 2));
        MultilinearPolynomial::from(
            (0..prefix_len.pow2())
                .map(|i| {
                    let (right, left) = uninterleave_bits(i as u64);
                    let comb = F::from_u32(right) + self.z.mul_u64(left as u64);
                    bound_value + comb
                })
                .collect::<Vec<F>>(),
        )
    }

    fn update_checkpoints(&mut self, checkpoints: F) {
        self.bound_value = checkpoints;
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
        self.linear_term += r.mul_u64(4 * (1u64 << self.num_bound_vars));

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
        let mut product_eval = F::zero();
        for i in 1..degree {
            // Evaluate at point i+1
            linear_eval += m;
            linear_evals[i] = linear_eval;
            product_evals[i] = if index == 0 {
                product_eval -= self.product_term;
                product_eval
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
    fn identity_poly_prefix_suffix_decomposition() {
        const NUM_VARS: usize = 8;
        const PREFIX_LEN: usize = 2;
        const SUFFIX_LEN: usize = NUM_VARS - PREFIX_LEN;

        let identity_poly: IdentityPolynomial<Fr> =
            IdentityPolynomial::new_with_endianness(NUM_VARS, Endianness::Big);
        let prefix_poly = identity_poly.prefix_polynomial(PREFIX_LEN);
        let shift_suffix = ShiftSuffixPolynomial;

        // Test over the entire boolean hypercube that:
        // IdentityPolynomial::evaluate(x_0, ..., x_7) =
        //     IdentityPolynomial::prefix_polynomial().evaluate(x_0, x_1) * shift_suffix_mle(x_2, .., x_7) +
        //     IdentityPolynomial::suffix_mle(x_2, ..., x_7)

        for i in 0..(1 << NUM_VARS) {
            let mut eval_point = vec![Fr::ZERO; NUM_VARS];
            for j in 0..NUM_VARS {
                if (i >> j) & 1 == 1 {
                    eval_point[j] = Fr::ONE;
                }
            }

            let direct_eval = identity_poly.evaluate(&eval_point);

            let prefix_eval_point = &eval_point[0..PREFIX_LEN];
            let suffix_eval_point = &eval_point[PREFIX_LEN..];

            let mut suffix_index = 0u64;
            for (j, &bit) in suffix_eval_point.iter().rev().enumerate() {
                if bit == Fr::ONE {
                    suffix_index |= 1 << j;
                }
            }

            let prefix_eval = prefix_poly.evaluate(prefix_eval_point);
            let shift_suffix_eval: Fr = shift_suffix.suffix_mle(suffix_index, SUFFIX_LEN);
            let suffix_eval = identity_poly.suffix_mle(suffix_index, SUFFIX_LEN);

            let decomposed_eval = prefix_eval * shift_suffix_eval + suffix_eval;

            assert_eq!(
                direct_eval, decomposed_eval,
                "IdentityPolynomial decomposition failed at index {i}: direct={direct_eval}, decomposed={decomposed_eval}"
            );
        }
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
    fn batched_uninterleave_poly_boolean_hypercube() {
        const NUM_VARS: usize = 8;

        let mut rng = test_rng();
        let z = Fr::random(&mut rng);
        let batched_poly = BatchedUninterleavePolynomial::<Fr>::new(NUM_VARS, z);

        // Test evaluation on boolean hypercube
        for i in 0..(1 << NUM_VARS) {
            let mut eval_point = vec![Fr::ZERO; NUM_VARS];
            for j in 0..NUM_VARS {
                if (i >> j) & 1 == 1 {
                    eval_point[j] = Fr::ONE;
                }
            }
            eval_point.reverse();

            // Use uninterleave_bits to match the polynomial's implementation
            let (right, left) = uninterleave_bits(i as u64);
            let expected = Fr::from_u32(right) + z * Fr::from_u32(left);

            assert_eq!(
                batched_poly.evaluate(&eval_point),
                expected,
                "Boolean hypercube evaluation failed at index {i}"
            );
        }
    }

    #[test]
    fn batched_uninterleave_poly_prefix_suffix_decomposition() {
        const NUM_VARS: usize = 8;
        const PREFIX_LEN: usize = 2;
        const SUFFIX_LEN: usize = NUM_VARS - PREFIX_LEN;

        let mut rng = test_rng();
        let z = Fr::random(&mut rng);
        let batched_poly = BatchedUninterleavePolynomial::<Fr>::new(NUM_VARS, z);
        let prefix_poly = batched_poly.prefix_polynomial(PREFIX_LEN);
        let shift_suffix = ShiftHalfSuffixPolynomial;

        // Test over the entire boolean hypercube that:
        // BatchedUninterleavePolynomial::evaluate(x_0, ..., x_7) =
        //     BatchedUninterleavePolynomial::prefix_polynomial().evaluate(x_0, x_1) * shift_suffix_mle(x_2, .., x_7) +
        //     BatchedUninterleavePolynomial::suffix_mle(x_2, ..., x_7)

        for i in 0..(1 << NUM_VARS) {
            let mut eval_point = vec![Fr::ZERO; NUM_VARS];
            for j in 0..NUM_VARS {
                if (i >> j) & 1 == 1 {
                    eval_point[j] = Fr::ONE;
                }
            }

            let direct_eval = batched_poly.evaluate(&eval_point);

            let prefix_eval_point = &eval_point[0..PREFIX_LEN];
            let suffix_eval_point = &eval_point[PREFIX_LEN..];

            let mut suffix_index = 0u64;
            for (j, &bit) in suffix_eval_point.iter().rev().enumerate() {
                if bit == Fr::ONE {
                    suffix_index |= 1 << j;
                }
            }

            let prefix_eval = prefix_poly.evaluate(prefix_eval_point);
            let shift_suffix_eval: Fr = shift_suffix.suffix_mle(suffix_index, SUFFIX_LEN);
            let suffix_eval = batched_poly.suffix_mle(suffix_index, SUFFIX_LEN);

            let decomposed_eval = prefix_eval * shift_suffix_eval + suffix_eval;

            assert_eq!(
                direct_eval, decomposed_eval,
                "BatchedUninterleavePolynomial decomposition failed at index {i}: direct={direct_eval}, decomposed={decomposed_eval}"
            );
        }
    }

    #[test]
    fn batched_uninterleave_poly() {
        const NUM_VARS: usize = 8;

        let mut rng = test_rng();
        let z = Fr::random(&mut rng);
        println!("z = {z}");
        let mut batched_poly = BatchedUninterleavePolynomial::<Fr>::new(NUM_VARS, z);

        // Create reference polynomial with evaluations
        let reference_evals: Vec<Fr> = (0u64..(1 << NUM_VARS))
            .map(|i| {
                let (right, left) = uninterleave_bits(i);
                Fr::from_u32(right) + z * Fr::from_u32(left)
            })
            .collect();
        let mut reference_poly = MultilinearPolynomial::from(reference_evals);

        // Verify that both polynomials agree on the entire boolean hypercube
        for i in 0..(1 << NUM_VARS) {
            let mut eval_point = vec![Fr::ZERO; NUM_VARS];
            for j in 0..NUM_VARS {
                if (i >> j) & 1 == 1 {
                    eval_point[j] = Fr::ONE;
                }
            }
            let batched_eval = batched_poly.evaluate(&eval_point);
            let reference_eval = reference_poly.evaluate(&eval_point);
            assert_eq!(
                batched_eval, reference_eval,
                "Evaluation mismatch at index {i}: batched={batched_eval}, reference={reference_eval}"
            );
        }

        for round in 0..NUM_VARS {
            let num_evals = 1 << (NUM_VARS - round - 1);

            for i in 0..num_evals {
                let batched_evals = batched_poly.sumcheck_evals(i, 3, BindingOrder::HighToLow);
                let reference_evals = reference_poly.sumcheck_evals(i, 3, BindingOrder::HighToLow);

                assert_eq!(
                    batched_evals, reference_evals,
                    "Round {round}, index {i}: sumcheck_evals mismatch"
                );
            }

            let r = Fr::random(&mut rng);
            batched_poly.bind(r, BindingOrder::HighToLow);
            reference_poly.bind(r, BindingOrder::HighToLow);
        }

        assert_eq!(
            batched_poly.final_sumcheck_claim(),
            reference_poly.final_sumcheck_claim(),
            "Final sumcheck claims don't match"
        );
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
