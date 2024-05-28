use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use sha3::Sha3_256;

use crate::poly::field::JoltField;

use super::{builder::{CombinedUniformBuilder, UniformR1CS}, ops::ConstraintInput};
use digest::Digest;

use crate::utils::math::Math;

use rayon::prelude::*;



#[derive(CanonicalSerialize, CanonicalDeserialize)]
pub struct UniformSpartanKey<F: JoltField> {
    pub uniform_r1cs: UniformR1CS<F>,

    /// Number of constraints across all steps padded to nearest power of 2
    pub num_cons_total: usize,

    /// Number of steps padded to the nearest power of 2
    pub num_steps: usize,

    /// Digest of verifier key
    pub(crate) vk_digest: F,
}

impl<F: JoltField> UniformSpartanKey<F> {
    pub fn from_builder<I: ConstraintInput>(constraint_builder: &CombinedUniformBuilder<F, I>) -> Self {
        let uniform_r1cs = constraint_builder.materialize_uniform();

        let total_rows = constraint_builder.constraint_rows().next_power_of_two();
        let num_steps = constraint_builder.uniform_repeat().next_power_of_two();

        // TODO(sragss): Need to digest non-uniform constraints as well.
        let vk_digest = Self::digest(&uniform_r1cs, num_steps);

        Self {
            uniform_r1cs,
            num_cons_total: total_rows,
            num_steps,
            vk_digest
        }
    }

    fn full_z_len(&self) -> usize {
        2 * self.num_steps * self.uniform_r1cs.num_vars
    }

    /// Number of variables across all steps padded to next power of two. 
    pub fn num_vars_total(&self) -> usize {
        self.num_steps * self.uniform_r1cs.num_vars.next_power_of_two()
    }
    
    // TODO(sragss): rm in favor of from_builder
    // pub fn construct(
    //     uniform: (Vec<(usize, usize, F)>, Vec<(usize, usize, F)>, Vec<(usize, usize, F)>), 
    //     num_constraints_total: usize,
    //     num_vars_total: usize,
    //     num_steps: usize,
    //     uniform_num_vars: usize,
    //     uniform_num_rows: usize,
    // ) -> Self {
    //     assert!(num_constraints_total.is_power_of_two());
    //     assert!(num_vars_total.is_power_of_two());
    //     assert!(num_steps.is_power_of_two());

    //     let vk_digest = Self::digest(&single_step_shape, num_steps);

    //     Self {
    //         uniform_r1cs: UniformR1CS { 
    //             a: single_step_shape.0, 
    //             b: single_step_shape.1, 
    //             c: single_step_shape.2, 
    //             num_vars: uniform_num_vars, 
    //             num_rows: uniform_num_rows,
    //         },
    //         num_cons_total: num_constraints_total.next_power_of_two(),
    //         // num_vars_total: num_vars_total.next_power_of_two(),
    //         num_steps: num_steps.next_power_of_two(),
    //         vk_digest,
    //     }
    // }

    /// Returns the digest of the r1cs shape
    fn digest(
            uniform_r1cs: &UniformR1CS<F>, 
            num_steps: usize) -> F {
        let mut compressed_bytes = Vec::new();
        uniform_r1cs 
            .serialize_compressed(&mut compressed_bytes)
            .unwrap();
        compressed_bytes.extend(num_steps.to_be_bytes().to_vec());
        let mut hasher = Sha3_256::new();
        hasher.input(compressed_bytes);

        let map_to_field = |digest: &[u8]| -> F {
            let bv = (0..250).map(|i| {
                let (byte_pos, bit_pos) = (i / 8, i % 8);
                let bit = (digest[byte_pos] >> bit_pos) & 1;
                bit == 1
            });

            // turn the bit vector into a scalar
            let mut digest = F::zero();
            let mut coeff = F::one();
            for bit in bv {
                if bit {
                    digest += coeff;
                }
                coeff += coeff;
            }
            digest
        };
        map_to_field(&hasher.result())
    }

    pub fn evaluate_z_mle(&self, segment_evals: &[F], r: &[F]) -> F {
        assert_eq!(self.uniform_r1cs.num_vars, segment_evals.len());
        assert_eq!(r.len(), self.full_z_len().log_2());

        // Z can be computed in two halves, [Variables, (constant) 1, 0 , ...] indexed by the first bit.
        let r_const = r[0];
        let r_rest = &r[1..];
        assert_eq!(r_rest.len(), self.num_vars_total().log_2());

        // Don't need the last log2(num_steps) bits, they've been evaluated already.
        let var_bits = self.uniform_r1cs.num_vars.next_power_of_two().log_2();
        let r_var= &r_rest[..var_bits];

        let eval_variables = (0..self.uniform_r1cs.num_vars)
            .map(|var_index| {
                // write var_index in binary
                let bin = format!("{var_index:0width$b}", width = var_bits); 

                let product = bin.chars().enumerate().fold(F::one(), |acc, (j, bit)| {
                    acc * if bit == '0' {
                        F::one() - r_var[j]
                    } else {
                        r_var[j]
                    }
                });

                product * segment_evals[var_index]
            })
            .sum::<F>();
        let const_poly = SparsePolynomial::new(self.num_vars_total().log_2(), vec![(0, F::one())]);
        let eval_const = const_poly.evaluate(r_rest);

        (F::one() - r_const) * eval_variables + r_const * eval_const
    }
}

struct SparsePolynomial<F: JoltField> {
    num_vars: usize,
    Z: Vec<(usize, F)>,
}

impl<Scalar: JoltField> SparsePolynomial<Scalar> {
    pub fn new(num_vars: usize, Z: Vec<(usize, Scalar)>) -> Self {
        SparsePolynomial { num_vars, Z }
    }

    /// Computes the $\tilde{eq}$ extension polynomial.
    /// return 1 when a == r, otherwise return 0.
    fn compute_chi(a: &[bool], r: &[Scalar]) -> Scalar {
        assert_eq!(a.len(), r.len());
        let mut chi_i = Scalar::one();
        for j in 0..r.len() {
            if a[j] {
                chi_i *= r[j];
            } else {
                chi_i *= Scalar::one() - r[j];
            }
        }
        chi_i
    }

    // Takes O(n log n)
    pub fn evaluate(&self, r: &[Scalar]) -> Scalar {
        assert_eq!(self.num_vars, r.len());

        (0..self.Z.len())
            .into_par_iter()
            .map(|i| {
                let bits = get_bits(self.Z[0].0, r.len());
                SparsePolynomial::compute_chi(&bits, r) * self.Z[i].1
            })
            .sum()
    }
}

/// Returns the `num_bits` from n in a canonical order
fn get_bits(operand: usize, num_bits: usize) -> Vec<bool> {
    (0..num_bits)
        .map(|shift_amount| ((operand & (1 << (num_bits - shift_amount - 1))) > 0))
        .collect::<Vec<bool>>()
}

#[cfg(test)]
mod test {
    use super::*;
    use ark_bn254::Fr;

    use crate::{poly::dense_mlpoly::DensePolynomial, r1cs::new_r1cs::{builder::{R1CSBuilder, R1CSConstraintBuilder}, test::TestInputs}, utils::math::Math};
    use strum::EnumCount;

    #[test]
    fn z_mle() {
        let mut uniform_builder = R1CSBuilder::<Fr, TestInputs>::new();
        // OpFlags0 * OpFlags1 == 12
        struct TestConstraints();
        impl<F: JoltField> R1CSConstraintBuilder<F> for TestConstraints {
            type Inputs = TestInputs;
            fn build_constraints(&self, builder: &mut R1CSBuilder<F, Self::Inputs>) {
                builder.constrain_prod(TestInputs::OpFlags0, TestInputs::OpFlags1, 12);
            }
        }

        let constraints = TestConstraints();
        constraints.build_constraints(&mut uniform_builder);
        let num_steps: usize = 3;
        let num_steps_pad = 4;
        let combined_builder = CombinedUniformBuilder::construct(uniform_builder, num_steps, vec![]);
        let mut inputs = vec![vec![Fr::zero(); num_steps]; TestInputs::COUNT];

        inputs[TestInputs::OpFlags0 as usize][0] = Fr::from(1);
        inputs[TestInputs::OpFlags1 as usize][0] = Fr::from(12);

        inputs[TestInputs::OpFlags0 as usize][1] = Fr::from(2);
        inputs[TestInputs::OpFlags1 as usize][1] = Fr::from(6);

        inputs[TestInputs::OpFlags0 as usize][2] = Fr::from(3);
        inputs[TestInputs::OpFlags1 as usize][2] = Fr::from(4);

        // Confirms validity of constraints
        let (_az, _bz, _cz) = combined_builder.compute_spartan(&inputs, &vec![]);

        let key = UniformSpartanKey::from_builder(&combined_builder);

        // Z's full padded length is 2 * (num_vars * num_steps.next_power_of_two()) 
        let z_pad_len = 2 * num_steps_pad * TestInputs::COUNT.next_power_of_two();
        let z_bits = z_pad_len.log_2();
        assert_eq!(z_bits, 8);

        // 1 bit to index const
        // 5 bits to index variable
        // 2 bits to index step
        let r_const = vec![Fr::from(100)];
        let r_var = vec![Fr::from(200), Fr::from(300), Fr::from(400), Fr::from(500), Fr::from(600)];
        let r_step = vec![Fr::from(100), Fr::from(200)];
        let r = [r_const, r_var, r_step.clone()].concat();

        let z_segment_evals: Vec<Fr> = inputs.iter().map(|input_vec| {
            let poly = DensePolynomial::new_padded(input_vec.clone());
            assert_eq!(poly.len(), num_steps_pad);
            poly.evaluate(&r_step)
        }).collect();

        // Construct the fully materialized version of 'z'
        // Expecting form of Z
        // [TestInputs::PCIn[0], ... PcIn[num_steps.next_pow_2 - 1],
        //  TestInputs::PCOut[0], ... PcOut[num_steps.next_pow_2 - 1],
        //  0 padding to num_vars.next_pow_2 * num_steps.next_pow_2
        //  1
        //  0 padding to 2 * num_vars.next_pow_2 * num_steps.next_pow_2
        // ]
        // 
        let mut z = Vec::with_capacity(z_pad_len);
        for var_across_steps in inputs {
            let new_padded_len = z.len() + num_steps_pad;
            z.extend(var_across_steps);
            z.resize(new_padded_len, Fr::zero());
        }
        let const_index = z_pad_len / 2;
        z.resize(const_index, Fr::zero());
        z.push(Fr::one());
        z.resize(z_pad_len, Fr::zero());


        let actual = key.evaluate_z_mle(&z_segment_evals, &r);
        let expected = DensePolynomial::new(z).evaluate(&r);
        assert_eq!(expected, actual);
    }
}