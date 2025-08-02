use jolt_core::poly::dense_mlpoly::DensePolynomial;
use jolt_core::poly::{
    multilinear_polynomial::MultilinearPolynomial, sparse_interleaved_poly::SparseCoefficient,
};
use jolt_core::{field::JoltField, r1cs::builder::Constraint};

#[derive(Default, Debug, Clone)]
pub struct SpartanInterleavedPolynomial<F: JoltField> {
    /// A sparse vector representing the (interleaved) coefficients in the Az, Bz, Cz
    /// polynomials used in the first Spartan sumcheck. Before the polynomial is bound
    /// the first time, all the coefficients can be represented by `i128`s.
    pub unbound_coeffs: Vec<SparseCoefficient<i128>>,
    /// A sparse vector representing the (interleaved) coefficients in the Az, Bz, Cz
    /// polynomials used in the first Spartan sumcheck. Once the polynomial has been
    /// bound, we switch to using `bound_coeffs` instead of `unbound_coeffs`, because
    /// coefficients will be full-width field elements rather than `i128`s.
    pub bound_coeffs: Vec<SparseCoefficient<F>>,
    /// A reused buffer where bound values are written to during `bind`.
    /// With every bind, `coeffs` and `binding_scratch_space` are swapped.
    pub binding_scratch_space: Vec<SparseCoefficient<F>>,
    /// The length of one of the Az, Bz, or Cz polynomials if it were represented by
    /// a single dense vector.
    pub dense_len: usize,
}

impl<F: JoltField> SpartanInterleavedPolynomial<F> {
    /// Computes the matrix-vector products Az, Bz, and Cz as a single interleaved sparse vector
    #[tracing::instrument(skip_all, name = "SpartanInterleavedPolynomial::new")]
    pub fn new(
        uniform_constraints: &[Constraint],
        flattened_polynomials: &[MultilinearPolynomial<F>], // N variables of (S steps)
        padded_num_constraints: usize,
    ) -> Self {
        let num_steps = flattened_polynomials[0].len();

        let num_chunks = 1;
        let chunk_size = num_steps;

        let unbound_coeffs: Vec<SparseCoefficient<i128>> = (0..num_chunks)
            .flat_map(|chunk_index| {
            let mut coeffs = Vec::with_capacity(chunk_size * padded_num_constraints * 3);
            for step_index in chunk_size * chunk_index..chunk_size * (chunk_index + 1) {
                // Uniform constraints
                for (constraint_index, constraint) in uniform_constraints.iter().enumerate() {
                let global_index =
                    3 * (step_index * padded_num_constraints + constraint_index);

                // Az
                let mut az_coeff = 0;
                if !constraint.a.terms().is_empty() {
                    az_coeff = constraint
                    .a
                    .evaluate_row(flattened_polynomials, step_index);
                    if az_coeff != 0 {
                    coeffs.push((global_index, az_coeff).into());
                    }
                }
                // Bz
                let mut bz_coeff = 0;
                if !constraint.b.terms().is_empty() {
                    bz_coeff = constraint
                    .b
                    .evaluate_row(flattened_polynomials, step_index);
                    if bz_coeff != 0 {
                    coeffs.push((global_index + 1, bz_coeff).into());
                    }
                }
                #[cfg(test)]
                {
                    let cz_coeff = az_coeff * bz_coeff;
                    if cz_coeff != constraint
                    .c
                    .evaluate_row(flattened_polynomials, step_index) {
                        use crate::jolt::r1cs::builder::R1CSConstraintFormatter;

                        let mut constraint_string = String::new();
                        let _ = constraint
                        .format_constraint::<F>(
                            &mut constraint_string,
                            flattened_polynomials,
                            step_index,
                        );
                        println!("{constraint_string}");
                        panic!(
                        "Uniform constraint {constraint_index} violated at step {step_index}",
                        );
                    }
                }
                // Cz = Az ⊙ Cz
                if az_coeff != 0 && bz_coeff != 0 {
                    let cz_coeff = az_coeff * bz_coeff;
                    coeffs.push((global_index + 2, cz_coeff).into());
                }
                }
            }
            coeffs
            })
            .collect();

        #[cfg(test)]
        {
            // Check that indices are monotonically increasing
            let mut prev_index = unbound_coeffs[0].index;
            for coeff in unbound_coeffs[1..].iter() {
                assert!(coeff.index > prev_index);
                prev_index = coeff.index;
            }
        }

        Self {
            unbound_coeffs,
            bound_coeffs: vec![],
            binding_scratch_space: vec![],
            dense_len: num_steps * padded_num_constraints,
        }
    }

    fn _uninterleave(&self) -> (DensePolynomial<F>, DensePolynomial<F>, DensePolynomial<F>) {
        let mut az = vec![F::zero(); self.dense_len];
        let mut bz = vec![F::zero(); self.dense_len];
        let mut cz = vec![F::zero(); self.dense_len];

        if !self.is_bound() {
            for coeff in &self.unbound_coeffs {
                match coeff.index % 3 {
                    0 => az[coeff.index / 3] = F::from_i128(coeff.value),
                    1 => bz[coeff.index / 3] = F::from_i128(coeff.value),
                    2 => cz[coeff.index / 3] = F::from_i128(coeff.value),
                    _ => unreachable!(),
                }
            }
        } else {
            for coeff in &self.bound_coeffs {
                match coeff.index % 3 {
                    0 => az[coeff.index / 3] = coeff.value,
                    1 => bz[coeff.index / 3] = coeff.value,
                    2 => cz[coeff.index / 3] = coeff.value,
                    _ => unreachable!(),
                }
            }
        }
        (
            DensePolynomial::new(az),
            DensePolynomial::new(bz),
            DensePolynomial::new(cz),
        )
    }

    pub fn is_bound(&self) -> bool {
        !self.bound_coeffs.is_empty()
    }
}
