//! Batches constraints into a single indexed polynomial F(z, x) = Σ_i eq(z, i) * C_i(x)

use crate::{
    field::JoltField,
    poly::{
        commitment::{
            commitment_scheme::RecursionExt,
            dory::{
                recursion::JoltGtExpWitness, ArkDoryProof, ArkworksVerifierSetup,
                DoryCommitmentScheme,
            },
        },
        dense_mlpoly::DensePolynomial,
        eq_poly::EqPolynomial,
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};
use ark_bn254::{Fq, Fr};
use ark_ff::{One, Zero};
use dory::backends::arkworks::ArkGT;
use jolt_optimizations::{fq12_to_multilinear_evals, get_g_mle};

/// Convert index to binary representation as field elements (little-endian)
pub fn index_to_binary(index: usize, num_vars: usize) -> Vec<Fq> {
    let mut binary = Vec::with_capacity(num_vars);
    let mut idx = index;

    for _ in 0..num_vars {
        binary.push(if idx & 1 == 1 { Fq::one() } else { Fq::zero() });
        idx >>= 1;
    }

    // binary.reverse();
    binary
}

/// Compute constraint formula: ρ_curr - ρ_prev² × base^{b} - quotient × g
pub fn compute_constraint_formula(
    rho_curr: Fq,
    rho_prev: Fq,
    base: Fq,
    quotient: Fq,
    g_val: Fq,
    bit: bool,
) -> Fq {
    let base_power = if bit { base } else { Fq::one() };
    rho_curr - rho_prev.square() * base_power - quotient * g_val
}

/// Polynomial types stored in the matrix
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum PolyType {
    Base = 0,
    RhoPrev = 1,
    RhoCurr = 2,
    Quotient = 3,
}

impl PolyType {
    pub const NUM_TYPES: usize = 4;

    pub fn all() -> [PolyType; 4] {
        [
            PolyType::Base,
            PolyType::RhoPrev,
            PolyType::RhoCurr,
            PolyType::Quotient,
        ]
    }

    /// Get polynomial type from row index
    pub fn from_row_index(row_idx: usize, num_constraints: usize) -> Self {
        match row_idx / num_constraints {
            0 => PolyType::Base,
            1 => PolyType::RhoPrev,
            2 => PolyType::RhoCurr,
            3 => PolyType::Quotient,
            _ => panic!("Invalid row index"),
        }
    }
}

/// Giant multilinear matrix M(s, x) that stores all Dory polynomials in a single structure.
///
/// Layout: M(s, x) where s is the row index and x are the constraint variables
/// Physical layout: rows are organized as [all base] [all rho_prev] [all rho_curr] [all quotient]
/// Row index = poly_type * num_constraints_padded + constraint_index
pub struct DoryMultilinearMatrix {
    /// Number of s variables (log2(num_rows))
    pub num_s_vars: usize,

    /// Number of constraint variables (x) - fixed at 4 for Fq12
    pub num_constraint_vars: usize,

    /// Number of constraint index variables (bits needed to index constraints)
    pub num_constraint_index_vars: usize,

    /// Number of constraints (before padding)
    pub num_constraints: usize,

    /// Number of constraints padded to power of 2
    pub num_constraints_padded: usize,

    /// Total number of rows: 4 * num_constraints_padded
    pub num_rows: usize,

    /// Total M variables: num_s_vars + num_constraint_vars
    pub num_vars: usize,

    /// Flattened storage: rows concatenated together
    /// Each row contains 2^num_constraint_vars evaluations
    /// Total size: num_rows * 2^num_constraint_vars
    pub evaluations: Vec<Fq>,
}

impl DoryMultilinearMatrix {
    /// Get row index for a given polynomial type and constraint index
    pub fn row_index(&self, poly_type: PolyType, constraint_idx: usize) -> usize {
        (poly_type as usize) * self.num_constraints_padded + constraint_idx
    }

    /// Get the storage offset for accessing a specific row's polynomial
    pub fn storage_offset(&self, row_index: usize) -> usize {
        row_index * (1 << self.num_constraint_vars)
    }

    /// Evaluate a specific row's polynomial at point x
    pub fn evaluate_row(&self, row: usize, constraint_vars: &[Fq]) -> Fq {
        let offset = self.storage_offset(row);
        let row_evals = &self.evaluations[offset..offset + (1 << self.num_constraint_vars)];

        let poly = DensePolynomial::new(row_evals.to_vec());
        poly.evaluate(constraint_vars)
    }

    /// Evaluate M(s, x) where s selects the row and x is the evaluation point
    pub fn evaluate(&self, s_vars: &[Fq], constraint_vars: &[Fq]) -> Fq {
        assert_eq!(s_vars.len(), self.num_s_vars);
        assert_eq!(constraint_vars.len(), self.num_constraint_vars);

        let mut result = Fq::zero();
        for row in 0..self.num_rows {
            let row_binary = index_to_binary(row, self.num_s_vars);
            let eq_eval = EqPolynomial::mle(&row_binary, s_vars);

            let row_poly_eval = self.evaluate_row(row, constraint_vars);
            result += eq_eval * row_poly_eval;
        }
        result
    }
}

/// Builder for constructing the giant multilinear matrix.
///
/// Physical layout: rows are organized as [all base] [all rho_prev] [all rho_curr] [all quotient]
/// Row index = poly_type * num_constraints_padded + constraint_index
pub struct DoryMatrixBuilder {
    num_constraint_vars: usize,
    /// Rows grouped by polynomial type: [base_rows, rho_prev_rows, rho_curr_rows, quotient_rows]
    rows_by_type: [Vec<Vec<Fq>>; 4],
    /// Bits for each constraint
    bits: Vec<bool>,
}

impl DoryMatrixBuilder {
    pub fn new(num_constraint_vars: usize) -> Self {
        Self {
            num_constraint_vars,
            rows_by_type: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
            bits: Vec::new(),
        }
    }

    /// Add constraints from a GT exponentiation witness.
    /// Each step j creates one constraint using:
    /// - base: the base element a (replicated for each constraint)
    /// - rho_prev: ρ_j
    /// - rho_curr: ρ_{j+1}
    /// - quotient: Q_j
    pub fn add_gt_exp_witness(&mut self, witness: &JoltGtExpWitness) {
        let base_mle = fq12_to_multilinear_evals(&witness.base);
        assert_eq!(base_mle.len(), 1 << self.num_constraint_vars);

        let n = witness.bits.len();

        for step in 0..n {
            // Base: always store a(x), independent of bit
            let base_row = base_mle.clone();
            self.rows_by_type[PolyType::Base as usize].push(base_row);

            // RhoPrev: ρ_step
            let rho_prev = witness.rho_mles[step].clone();
            assert_eq!(rho_prev.len(), 1 << self.num_constraint_vars);
            self.rows_by_type[PolyType::RhoPrev as usize].push(rho_prev);

            // RhoCurr: ρ_{step+1}
            let rho_curr = witness.rho_mles[step + 1].clone();
            assert_eq!(rho_curr.len(), 1 << self.num_constraint_vars);
            self.rows_by_type[PolyType::RhoCurr as usize].push(rho_curr);

            // Quotient: Q_step
            let quotient = witness.quotient_mles[step].clone();
            assert_eq!(quotient.len(), 1 << self.num_constraint_vars);
            self.rows_by_type[PolyType::Quotient as usize].push(quotient);

            // Store bit for this constraint
            self.bits.push(witness.bits[step]);
        }
    }

    pub fn build(self) -> (DoryMultilinearMatrix, Vec<MatrixConstraint>) {
        let num_constraints = self.rows_by_type[0].len();
        assert!(num_constraints > 0, "No constraints added");

        // Verify all row types have the same number of constraints
        for poly_type in PolyType::all() {
            assert_eq!(
                self.rows_by_type[poly_type as usize].len(),
                num_constraints,
                "Row type {:?} has wrong number of constraints",
                poly_type
            );
        }
        assert_eq!(
            self.bits.len(),
            num_constraints,
            "Number of bits must match number of constraints"
        );

        // Pad constraints to power of 2
        let num_constraints_bits = (num_constraints as f64).log2().ceil() as usize;
        let num_constraints_padded = 1 << num_constraints_bits;

        // Total rows = 4 * num_constraints_padded
        let num_rows = PolyType::NUM_TYPES * num_constraints_padded;

        // Calculate number of s variables needed
        let num_s_vars = (num_rows as f64).log2().ceil() as usize;
        assert_eq!(1 << num_s_vars, num_rows); // Should be exact

        let row_size = 1 << self.num_constraint_vars;
        let zero_row = vec![Fq::zero(); row_size];

        let mut evaluations = Vec::with_capacity(num_rows * row_size);

        // Layout: rows are organized as [all base] [all rho_prev] [all rho_curr] [all quotient]
        for poly_type in PolyType::all() {
            let rows = &self.rows_by_type[poly_type as usize];

            // For all row types, use as-is (no baking)
            for row in rows {
                evaluations.extend_from_slice(row);
            }

            // Pad this section to num_constraints_padded
            for _ in rows.len()..num_constraints_padded {
                evaluations.extend_from_slice(&zero_row);
            }
        }

        let matrix = DoryMultilinearMatrix {
            num_s_vars,
            num_constraint_vars: self.num_constraint_vars,
            num_constraint_index_vars: num_constraints_bits,
            num_constraints,
            num_constraints_padded,
            num_rows,
            num_vars: num_s_vars + self.num_constraint_vars,
            evaluations,
        };

        // Create constraint metadata
        let constraints: Vec<MatrixConstraint> = self
            .bits
            .into_iter()
            .enumerate()
            .map(|(idx, bit)| MatrixConstraint {
                constraint_index: idx,
                bit,
            })
            .collect();

        (matrix, constraints)
    }
}

/// Constraint metadata for matrix-based evaluation.
/// Row indices are computed from constraint_index using the matrix layout.
#[derive(Clone, Debug)]
pub struct MatrixConstraint {
    /// Index of this constraint (0 to num_constraints-1)
    pub constraint_index: usize,
    /// Bit value for this step (exponent bit)
    pub bit: bool,
}

/// Constraint system using a giant multilinear matrix for all witness polynomials
pub struct ConstraintSystem {
    /// The giant matrix M(s, x)
    pub matrix: DoryMultilinearMatrix,

    /// g(x) polynomial - precomputed as DensePolynomial
    pub g_poly: DensePolynomial<Fq>,

    /// Constraint metadata: maps constraint index to matrix rows it references
    pub constraints: Vec<MatrixConstraint>,

    /// Public multilinear extension of the exponent bits b_i over index bits
    /// Domain: {0,1}^{num_constraint_index_vars}
    pub exponent_mle: DensePolynomial<Fq>,
}

impl ConstraintSystem {
    pub fn new<T>(
        proof: &ArkDoryProof,
        setup: &ArkworksVerifierSetup,
        transcript: &mut T,
        point: &[<Fr as JoltField>::Challenge],
        evaluation: &Fr,
        commitment: &ArkGT,
    ) -> Result<(Self, <DoryCommitmentScheme as RecursionExt<Fr>>::Hint), ProofVerifyError>
    where
        T: Transcript,
    {
        // Extract witnesses
        let (witnesses, hints) = <DoryCommitmentScheme as RecursionExt<Fr>>::witness_gen(
            proof, setup, transcript, point, evaluation, commitment,
        )?;

        // Build matrix with new interleaved layout
        let mut builder = DoryMatrixBuilder::new(4); // 4 vars for Fq12

        for (_op_id, witness) in witnesses.gt_exp.iter() {
            builder.add_gt_exp_witness(witness);
        }

        let (matrix, constraints) = builder.build();
        let g_poly = DensePolynomial::new(get_g_mle());

        // Build exponent MLE from constraint bits.
        //
        // exponent_mle is the MLE over the index bits:
        //   exp(z) = Σ_i b_i * eq(z, i)
        //
        // But we store it as values on {0,1}^{num_constraint_index_vars}, i.e. a dense table:
        //   values[j] = b_j for 0 <= j < num_constraints
        //             = 0    for padding indices
        let num_i_bits = matrix.num_constraint_index_vars;
        let size = 1 << num_i_bits;
        let mut exponent_values = vec![Fq::zero(); size];

        for constraint in &constraints {
            if constraint.bit {
                exponent_values[constraint.constraint_index] = Fq::one();
            }
            // else it stays zero
        }

        let exponent_mle = DensePolynomial::new(exponent_values);

        Ok((
            Self {
                matrix,
                g_poly,
                constraints,
                exponent_mle,
            },
            hints,
        ))
    }

    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Number of s variables (for binding in Phase 2)
    pub fn num_s_vars(&self) -> usize {
        self.matrix.num_s_vars
    }

    /// Total variables: s + x
    pub fn num_vars(&self) -> usize {
        self.matrix.num_vars
    }

    /// Extract constraint polynomials for square-and-multiply sumcheck (Phase 1)
    pub fn extract_constraint_polynomials(&self) -> Vec<crate::subprotocols::square_and_multiply::ConstraintPolynomials> {
        let mut polys = Vec::new();
        let num_constraint_vars = self.matrix.num_constraint_vars;
        let row_size = 1 << num_constraint_vars;

        for (idx, constraint) in self.constraints.iter().enumerate() {
            // Extract polynomial data from the matrix for each constraint
            let base = self.extract_row_poly(PolyType::Base, idx, row_size);
            let rho_prev = self.extract_row_poly(PolyType::RhoPrev, idx, row_size);
            let rho_curr = self.extract_row_poly(PolyType::RhoCurr, idx, row_size);
            let quotient = self.extract_row_poly(PolyType::Quotient, idx, row_size);

            polys.push(crate::subprotocols::square_and_multiply::ConstraintPolynomials {
                base,
                rho_prev,
                rho_curr,
                quotient,
                bit: constraint.bit,
                constraint_index: constraint.constraint_index,
            });
        }

        polys
    }

    /// Helper to extract a row polynomial from the matrix
    fn extract_row_poly(&self, poly_type: PolyType, constraint_idx: usize, row_size: usize) -> Vec<Fq> {
        let type_start = (poly_type as usize) * self.matrix.num_constraints_padded * row_size;
        let row_start = type_start + constraint_idx * row_size;
        let row_end = row_start + row_size;
        self.matrix.evaluations[row_start..row_end].to_vec()
    }

    /// Evaluate μ at a given point s
    /// This is used for testing/debugging - the actual evaluation should use the multilinear polynomial
    pub fn evaluate_mu_at_binary_point(
        base_claim: Fq,
        rho_prev_claim: Fq,
        rho_curr_claim: Fq,
        quotient_claim: Fq,
        s_binary: &[Fq],
        num_constraints_padded: usize,
    ) -> Fq {
        // Convert binary point to index (for testing only)
        let mut s_index = 0usize;
        for (i, &bit) in s_binary.iter().enumerate() {
            if bit == Fq::one() {
                s_index |= 1 << i;
            }
        }

        // Determine which polynomial type this row index corresponds to
        let poly_type = PolyType::from_row_index(s_index, num_constraints_padded);

        match poly_type {
            PolyType::Base => base_claim,
            PolyType::RhoPrev => rho_prev_claim,
            PolyType::RhoCurr => rho_curr_claim,
            PolyType::Quotient => quotient_claim,
        }
    }

    /// Evaluate the constraint system for Phase 1 sumcheck
    ///
    /// Takes only x variables and evaluates F(x) = Σ_i γ^i * C_i(x)
    /// This is used in the square-and-multiply sumcheck.
    pub fn evaluate_constraints_batched(&self, x_vars: &[Fq], gamma: Fq) -> Fq {
        assert_eq!(x_vars.len(), self.matrix.num_constraint_vars);

        let mut result = Fq::zero();
        let mut gamma_power = gamma;

        for constraint in self.constraints.iter() {
            let constraint_eval = self.evaluate_constraint(constraint, x_vars);
            result += gamma_power * constraint_eval;
            gamma_power *= gamma;
        }

        result
    }

    /// Evaluate the full constraint system at a point (for testing)
    ///
    /// Point structure: [x_vars, s_vars]
    /// Returns F(x, s) where s selects which constraint to evaluate
    pub fn evaluate(&self, point: &[Fq]) -> Fq {
        let num_x_vars = self.matrix.num_constraint_vars;
        let num_s_vars = self.matrix.num_s_vars;

        assert_eq!(point.len(), num_x_vars + num_s_vars);

        // Split point: [x_vars, s_vars]
        let (x_vars, s_vars) = point.split_at(num_x_vars);

        // We only evaluate constraints here, not the matrix rows
        // So we need to map s to constraint index
        let num_constraint_bits = (self.num_constraints() as f64).log2().ceil() as usize;
        let num_constraints_padded = 1 << num_constraint_bits;

        let mut result = Fq::zero();

        // For each constraint, check if s selects it
        for constraint in self.constraints.iter() {
            // s encodes both poly type and constraint index
            // For constraint evaluation, we treat all poly types of a constraint the same
            let constraint_padded_idx = constraint.constraint_index;

            // Check all 4 row indices that correspond to this constraint
            for poly_type in PolyType::all() {
                let row_idx = (poly_type as usize) * num_constraints_padded + constraint_padded_idx;
                let row_binary = index_to_binary(row_idx, num_s_vars);
                let eq_eval = EqPolynomial::mle(&row_binary, s_vars);

                let constraint_eval = self.evaluate_constraint(constraint, x_vars);
                result += eq_eval * constraint_eval;
            }
        }

        result
    }

    /// Evaluate a single constraint C_i(x) using the matrix layout.
    fn evaluate_constraint(&self, constraint: &MatrixConstraint, x: &[Fq]) -> Fq {
        let idx = constraint.constraint_index;

        // Get row indices using the layout
        let base_row = self.matrix.row_index(PolyType::Base, idx);
        let rho_prev_row = self.matrix.row_index(PolyType::RhoPrev, idx);
        let rho_curr_row = self.matrix.row_index(PolyType::RhoCurr, idx);
        let quotient_row = self.matrix.row_index(PolyType::Quotient, idx);

        // Evaluate each row polynomial at x
        let base_eval = self.matrix.evaluate_row(base_row, x); // a(x)
        let rho_prev = self.matrix.evaluate_row(rho_prev_row, x);
        let rho_curr = self.matrix.evaluate_row(rho_curr_row, x);
        let quotient = self.matrix.evaluate_row(quotient_row, x);
        let g_eval = self.g_poly.evaluate(x);

        let bit = constraint.bit;
        let bit_f = if bit { Fq::one() } else { Fq::zero() };

        // a(x)^{b_i} = 1 + (a(x) - 1) * b_i
        let base_power = Fq::one() + (base_eval - Fq::one()) * bit_f;

        rho_curr - rho_prev.square() * base_power - quotient * g_eval
    }

    /// Evaluate the exponent MLE at a constraint index point
    pub fn exponent_eval_at(&self, constraint_index_point: &[Fq]) -> Fq {
        self.exponent_mle.evaluate(constraint_index_point)
    }

    #[cfg(test)]
    pub fn verify_constraints_are_zero(&self) {
        // Verify that each constraint evaluates to 0 over the entire hypercube
        let num_x_points = 1 << self.matrix.num_constraint_vars;

        for constraint in &self.constraints {
            let idx = constraint.constraint_index;

            // Check constraint over all x values in the hypercube
            for x_val in 0..num_x_points {
                // Convert x_val to binary representation
                let mut x_binary = Vec::with_capacity(self.matrix.num_constraint_vars);
                let mut x = x_val;
                for _ in 0..self.matrix.num_constraint_vars {
                    x_binary.push(if x & 1 == 1 { Fq::one() } else { Fq::zero() });
                    x >>= 1;
                }

                // Evaluate constraint C_i(x)
                let base_row = self.matrix.row_index(PolyType::Base, idx);
                let rho_prev_row = self.matrix.row_index(PolyType::RhoPrev, idx);
                let rho_curr_row = self.matrix.row_index(PolyType::RhoCurr, idx);
                let quotient_row = self.matrix.row_index(PolyType::Quotient, idx);

                let base_eval = self.matrix.evaluate_row(base_row, &x_binary);
                let rho_prev = self.matrix.evaluate_row(rho_prev_row, &x_binary);
                let rho_curr = self.matrix.evaluate_row(rho_curr_row, &x_binary);
                let quotient = self.matrix.evaluate_row(quotient_row, &x_binary);
                let g_eval = self.g_poly.evaluate(&x_binary);

                // Get the bit for this constraint
                let bit = constraint.bit;
                let bit_f = if bit { Fq::one() } else { Fq::zero() };

                // Compute: ρ_curr(x) - ρ_prev(x)² × base(x)^{b_i} - quotient(x) × g(x)
                // where base(x)^{b_i} = 1 + (base(x) - 1) * b_i
                let base_power = Fq::one() + (base_eval - Fq::one()) * bit_f;
                let constraint_eval = rho_curr - rho_prev.square() * base_power - quotient * g_eval;

                assert!(
                    constraint_eval == Fq::zero(),
                    "Constraint {} failed at x={:?}: got {}, expected 0",
                    idx,
                    x_binary,
                    constraint_eval
                );
            }
        }

        println!(
            "All {} constraints verified to be 0 over the hypercube!",
            self.constraints.len()
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::poly::{
        commitment::{
            commitment_scheme::CommitmentScheme,
            dory::{DoryCommitmentScheme, DoryGlobals},
        },
        dense_mlpoly::DensePolynomial,
        multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
    };
    use ark_bn254::Fr;
    use rayon::prelude::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_dory_witness_constraint_evaluation() {
        use ark_ff::UniformRand;
        use rand::thread_rng;

        DoryGlobals::reset();
        DoryGlobals::initialize(1 << 2, 1 << 2);
        let num_vars = 4;
        let mut rng = thread_rng();

        let prover_setup = DoryCommitmentScheme::setup_prover(num_vars);
        let verifier_setup = DoryCommitmentScheme::setup_verifier(&prover_setup);

        let coefficients: Vec<Fr> = (0..(1 << num_vars)).map(|_| Fr::rand(&mut rng)).collect();
        let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coefficients));
        let (commitment, hint) = DoryCommitmentScheme::commit(&poly, &prover_setup);

        let point: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
            .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
            .collect();

        let mut prover_transcript = crate::transcripts::Blake2bTranscript::new(b"test");
        let proof = DoryCommitmentScheme::prove(
            &prover_setup,
            &poly,
            &point,
            Some(hint),
            &mut prover_transcript,
        );

        let evaluation = PolynomialEvaluation::evaluate(&poly, &point);
        let mut extract_transcript = crate::transcripts::Blake2bTranscript::new(b"test");
        let (system, hints) = ConstraintSystem::new(
            &proof,
            &verifier_setup,
            &mut extract_transcript,
            &point,
            &evaluation,
            &commitment,
        )
        .expect("System creation should succeed");
        let total_vars = system.num_vars();
        let num_points = 1 << total_vars;

        if num_points <= (1 << 16) {
            let all_zero = (0..num_points).into_par_iter().all(|i| {
                let mut point = Vec::with_capacity(total_vars);
                for j in 0..total_vars {
                    point.push(if (i >> j) & 1 == 1 {
                        Fq::one()
                    } else {
                        Fq::zero()
                    });
                }
                system.evaluate(&point) == Fq::zero()
            });

            assert!(
                all_zero,
                "Constraint system should evaluate to 0 on all points of the boolean hypercube"
            );
        } else {
            use rand::{rngs::StdRng, Rng, SeedableRng};

            let all_zero = (0..1000).into_par_iter().all(|i| {
                let mut rng = StdRng::seed_from_u64(i as u64);
                let point: Vec<Fq> = (0..total_vars)
                    .map(|_| {
                        if rng.gen::<bool>() {
                            Fq::one()
                        } else {
                            Fq::zero()
                        }
                    })
                    .collect();
                system.evaluate(&point) == Fq::zero()
            });

            assert!(
                all_zero,
                "Constraint system should evaluate to 0 on sampled boolean points"
            );
        }
        let mut verify_transcript = crate::transcripts::Blake2bTranscript::new(b"test");
        DoryCommitmentScheme::verify_with_hint(
            &proof,
            &verifier_setup,
            &mut verify_transcript,
            &point,
            &evaluation,
            &commitment,
            &hints,
        )
        .expect("Verification with hint should succeed");

        let mut verify_transcript_no_hint = crate::transcripts::Blake2bTranscript::new(b"test");
        DoryCommitmentScheme::verify(
            &proof,
            &verifier_setup,
            &mut verify_transcript_no_hint,
            &point,
            &evaluation,
            &commitment,
        )
        .expect("Verification without hint should also succeed");
    }
}
