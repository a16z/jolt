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

/// Row type offsets for the interleaved matrix layout.
/// Layout: row = offset * num_constraints_padded + constraint_index
/// This puts offset bits HIGH-order so they remain unbound after Phase 2 of recursion sum-check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum RowOffset {
    Base = 0,
    RhoPrev = 1,
    RhoCurr = 2,
    Quotient = 3,
}

impl RowOffset {
    pub const NUM_OFFSETS: usize = 4;
    pub const NUM_OFFSET_BITS: usize = 2;

    pub fn all() -> [RowOffset; 4] {
        [
            RowOffset::Base,
            RowOffset::RhoPrev,
            RowOffset::RhoCurr,
            RowOffset::Quotient,
        ]
    }

    pub fn to_bits(self) -> [Fq; 2] {
        let val = self as usize;
        [
            if val & 1 == 1 { Fq::one() } else { Fq::zero() },
            if val & 2 == 2 { Fq::one() } else { Fq::zero() },
        ]
    }
}

/// Giant multilinear matrix M(s, x) that stores all Dory polynomials in a single structure.
///
/// Layout: M(offset_bits, constraint_bits, x_bits)
/// - offset_bits: 2 bits (high-order in row dimension) for row type [Base, RhoPrev, RhoCurr, Quotient]
/// - constraint_bits: log2(num_constraints_padded) bits for constraint index
/// - x_bits: 4 bits (low-order) for Fq12 constraint variables
///
/// Row index = offset * num_constraints_padded + constraint_index
/// This layout ensures offset bits are high-order, so they remain unbound after Phase 2.
pub struct DoryMultilinearMatrix {
    /// Number of constraint index variables (log2(num_constraints_padded))
    pub num_constraint_index_vars: usize,

    /// Number of constraint variables (x) - fixed at 4 for Fq12
    pub num_constraint_vars: usize,

    /// Number of constraints (before padding)
    pub num_constraints: usize,

    /// Number of constraints padded to power of 2
    pub num_constraints_padded: usize,

    /// Total number of rows: 4 * num_constraints_padded
    pub num_rows: usize,

    /// Total M variables: num_constraint_index_vars + 2 (offset) + num_constraint_vars
    pub num_vars: usize,

    /// Flattened storage: rows concatenated together
    /// Each row contains 2^num_constraint_vars evaluations
    /// Total size: num_rows * 2^num_constraint_vars
    pub evaluations: Vec<Fq>,
}

impl DoryMultilinearMatrix {
    /// Get row index for a given offset and constraint index
    pub fn row_index(&self, offset: RowOffset, constraint_idx: usize) -> usize {
        (offset as usize) * self.num_constraints_padded + constraint_idx
    }

    /// Total number of row index variables (constraint_index_vars + offset_bits)
    pub fn num_row_vars(&self) -> usize {
        self.num_constraint_index_vars + RowOffset::NUM_OFFSET_BITS
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

    /// Evaluate M(row_vars, x) where row_vars select the row and x is the evaluation point
    pub fn evaluate(&self, row_vars: &[Fq], constraint_vars: &[Fq]) -> Fq {
        let num_row_vars = self.num_row_vars();
        assert_eq!(row_vars.len(), num_row_vars);
        assert_eq!(constraint_vars.len(), self.num_constraint_vars);

        let mut result = Fq::zero();
        for row in 0..self.num_rows {
            let row_binary = index_to_binary(row, num_row_vars);
            let eq_eval = EqPolynomial::mle(&row_binary, row_vars);

            let row_poly_eval = self.evaluate_row(row, constraint_vars);
            result += eq_eval * row_poly_eval;
        }
        result
    }
}

/// Builder for constructing the giant multilinear matrix with interleaved layout.
///
/// Layout: row = offset * num_constraints_padded + constraint_index
/// - Rows [0, n): base polynomials (offset=0)
/// - Rows [n, 2n): rho_prev polynomials (offset=1)
/// - Rows [2n, 3n): rho_curr polynomials (offset=2)
/// - Rows [3n, 4n): quotient polynomials (offset=3)
pub struct DoryMatrixBuilder {
    num_constraint_vars: usize,
    /// Rows grouped by offset type: [base_rows, rho_prev_rows, rho_curr_rows, quotient_rows]
    rows_by_offset: [Vec<Vec<Fq>>; 4],
    /// Bits for each constraint
    bits: Vec<bool>,
}

impl DoryMatrixBuilder {
    pub fn new(num_constraint_vars: usize) -> Self {
        Self {
            num_constraint_vars,
            rows_by_offset: [Vec::new(), Vec::new(), Vec::new(), Vec::new()],
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
            // Base: replicated for each constraint
            self.rows_by_offset[RowOffset::Base as usize].push(base_mle.clone());

            // RhoPrev: ρ_step
            let rho_prev = witness.rho_mles[step].clone();
            assert_eq!(rho_prev.len(), 1 << self.num_constraint_vars);
            self.rows_by_offset[RowOffset::RhoPrev as usize].push(rho_prev);

            // RhoCurr: ρ_{step+1}
            let rho_curr = witness.rho_mles[step + 1].clone();
            assert_eq!(rho_curr.len(), 1 << self.num_constraint_vars);
            self.rows_by_offset[RowOffset::RhoCurr as usize].push(rho_curr);

            // Quotient: Q_step
            let quotient = witness.quotient_mles[step].clone();
            assert_eq!(quotient.len(), 1 << self.num_constraint_vars);
            self.rows_by_offset[RowOffset::Quotient as usize].push(quotient);

            // Store bit for this constraint
            self.bits.push(witness.bits[step]);
        }
    }

    pub fn build(self) -> (DoryMultilinearMatrix, Vec<MatrixConstraint>) {
        let num_constraints = self.rows_by_offset[0].len();
        assert!(num_constraints > 0, "No constraints added");

        // Pad to power of 2
        let num_constraint_index_vars = (num_constraints as f64).log2().ceil() as usize;
        let num_constraints_padded = 1 << num_constraint_index_vars;

        // Total rows = 4 * num_constraints_padded
        let num_rows = RowOffset::NUM_OFFSETS * num_constraints_padded;
        let row_size = 1 << self.num_constraint_vars;
        let zero_row = vec![Fq::zero(); row_size];

        let mut evaluations = Vec::with_capacity(num_rows * row_size);

        // Layout: rows are organized as [all base] [all rho_prev] [all rho_curr] [all quotient]
        for offset in RowOffset::all() {
            let rows = &self.rows_by_offset[offset as usize];
            for row in rows {
                evaluations.extend_from_slice(row);
            }
            // Pad this offset section to num_constraints_padded
            for _ in rows.len()..num_constraints_padded {
                evaluations.extend_from_slice(&zero_row);
            }
        }

        let matrix = DoryMultilinearMatrix {
            num_constraint_index_vars,
            num_constraint_vars: self.num_constraint_vars,
            num_constraints,
            num_constraints_padded,
            num_rows,
            num_vars: num_constraint_index_vars
                + RowOffset::NUM_OFFSET_BITS
                + self.num_constraint_vars,
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

        Ok((
            Self {
                matrix,
                g_poly,
                constraints,
            },
            hints,
        ))
    }

    pub fn num_constraints(&self) -> usize {
        self.constraints.len()
    }

    /// Number of constraint index variables (for binding in Phase 2)
    pub fn num_constraint_index_vars(&self) -> usize {
        self.matrix.num_constraint_index_vars
    }

    /// Total variables: x (4) + constraint_index + offset (2)
    pub fn num_vars(&self) -> usize {
        self.matrix.num_vars
    }

    /// Evaluate the constraint system at a point in MLE variable order
    ///
    /// Point structure (little-endian): [x_vars (4), i_vars (num_constraint_index_vars), offset_bits (2)]
    ///
    /// F(x, i, offset) = Σ_j eq(i, j) * C_j(x)
    /// where C_j(x) = ρ_curr(x) - ρ_prev(x)² × base(x)^{b_j} - quotient(x) × g(x)
    ///
    /// Note: offset bits are not used in constraint evaluation - they only affect which
    /// row polynomials are accessed in PCS openings. For constraint verification,
    /// we use all 4 row types (base, rho_prev, rho_curr, quotient) for each constraint.
    pub fn evaluate(&self, point: &[Fq]) -> Fq {
        let num_x_vars = self.matrix.num_constraint_vars;
        let num_i_vars = self.num_constraint_index_vars();

        // Split point: [x_vars, i_vars, offset_bits]
        let (x_vars, rest) = point.split_at(num_x_vars);
        let (i_vars, _offset_bits) = rest.split_at(num_i_vars);

        let mut result = Fq::zero();

        for constraint in self.constraints.iter() {
            let i_binary = index_to_binary(constraint.constraint_index, num_i_vars);
            let eq_eval = EqPolynomial::mle(&i_binary, i_vars);

            let constraint_eval = self.evaluate_constraint(constraint, x_vars);
            result += eq_eval * constraint_eval;
        }

        result
    }

    /// Evaluate a single constraint C_i(x) using the new matrix layout.
    /// Row indices are computed from constraint_index.
    fn evaluate_constraint(&self, constraint: &MatrixConstraint, x: &[Fq]) -> Fq {
        let idx = constraint.constraint_index;

        // Get row indices using the new layout
        let base_row = self.matrix.row_index(RowOffset::Base, idx);
        let rho_prev_row = self.matrix.row_index(RowOffset::RhoPrev, idx);
        let rho_curr_row = self.matrix.row_index(RowOffset::RhoCurr, idx);
        let quotient_row = self.matrix.row_index(RowOffset::Quotient, idx);

        // Evaluate each row polynomial at x
        let base_eval = self.matrix.evaluate_row(base_row, x);
        let rho_prev = self.matrix.evaluate_row(rho_prev_row, x);
        let rho_curr = self.matrix.evaluate_row(rho_curr_row, x);
        let quotient = self.matrix.evaluate_row(quotient_row, x);
        let g_eval = self.g_poly.evaluate(x);

        // Compute: ρ_curr(x) - ρ_prev(x)² × base(x)^{b_i} - quotient(x) × g(x)
        let base_power = if constraint.bit { base_eval } else { Fq::one() };
        rho_curr - rho_prev.square() * base_power - quotient * g_eval
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
        println!(
            "Created system with {} constraints",
            system.num_constraints()
        );
        let total_vars = system.num_vars();
        let num_points = 1 << total_vars;

        if num_points <= (1 << 16) {
            println!("Testing all {} points on the boolean hypercube", num_points);
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
            println!(
                "✓ Constraint system correctly evaluates to 0 on all {} boolean hypercube points",
                num_points
            );
        } else {
            println!(
                "Testing 1000 random boolean points ({} vars = {} total points)",
                total_vars, num_points
            );
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
            println!("✓ Constraint system correctly evaluates to 0 on sampled boolean points");
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
