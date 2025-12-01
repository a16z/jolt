//! SNARK Composition for Dory Verifier
//!
//! This module batches constraints from GT exp, GT mul, and scalar muls
//! into a single indexed polynomial F(z, x) = Σ_i eq(z, i) * C_i(x)

use crate::{
    field::JoltField,
    poly::{
        commitment::{
            commitment_scheme::RecursionExt,
            dory::{
                recursion::{JoltGtExpWitness, JoltWitness},
                ArkDoryProof, ArkworksVerifierSetup, DoryCommitmentScheme,
            },
        },
        eq_poly::EqPolynomial,
    },
    transcripts::Transcript,
    utils::errors::ProofVerifyError,
};
use ark_bn254::{Fq, Fr};
use ark_ff::{One, Zero};
use dory::{backends::arkworks::ArkGT, recursion::WitnessCollection};
use jolt_optimizations::{fq12_to_multilinear_evals, get_g_mle};
use std::{marker::PhantomData, sync::Arc};

/// A single constraint captured from a step of a Dory verifier computation
pub trait Constraint<F: JoltField> {
    fn evaluate(&self, x: &[F]) -> F;
}

// Implement Constraint for references to trait objects
impl Constraint<Fq> for &dyn Constraint<Fq> {
    fn evaluate(&self, x: &[Fq]) -> Fq {
        (*self).evaluate(x)
    }
}

/// The batched constraint system
pub struct BatchedConstraintSystem<F: JoltField> {
    /// Total number of constraints
    pub num_constraints: usize,
    /// Number of index variables (log of num_constraints)
    pub num_index_vars: usize,
    /// Number of constraint variables (in this case, 4)
    pub num_constraint_vars: usize,
    _phantom: PhantomData<F>,
}

impl<F: JoltField> BatchedConstraintSystem<F> {
    pub fn new(num_constraints: usize, num_constraint_vars: usize) -> Self {
        let num_index_vars = (num_constraints as f64).log2().ceil() as usize;
        Self {
            num_constraints,
            num_index_vars,
            num_constraint_vars,
            _phantom: PhantomData,
        }
    }

    /// Total number of variables in F(z, x)
    pub fn num_vars(&self) -> usize {
        self.num_index_vars + self.num_constraint_vars
    }

    /// Evaluate F(z, x) = Σ_i eq(z, i) * C_i(x)
    ///
    /// - z: index variables (first num_index_vars of point)
    /// - x: constraint variables (remaining vars of point)
    /// - constraints: the C_i constraints
    pub fn evaluate(&self, point: &[F], constraints: &[Box<dyn Constraint<F>>]) -> F {
        let (index_vars, constraint_vars) = point.split_at(self.num_index_vars);

        let mut result = F::zero();

        // For each constraint i
        for (i, constraint) in constraints.iter().enumerate() {
            // Convert i to binary for eq evaluation
            let i_binary: Vec<F> = self.index_to_binary(i);

            // eq(z, i)
            let eq_eval = EqPolynomial::mle(&i_binary, index_vars);

            // C_i(x)
            let constraint_eval = constraint.evaluate(constraint_vars);

            // Add eq(z, i) * C_i(x) to sum
            result += eq_eval * constraint_eval;
        }

        result
    }

    /// Convert constraint index to binary representation
    pub fn index_to_binary(&self, index: usize) -> Vec<F> {
        let mut binary = Vec::with_capacity(self.num_index_vars);
        let mut idx = index;

        for _ in 0..self.num_index_vars {
            binary.push(if idx & 1 == 1 { F::one() } else { F::zero() });
            idx >>= 1;
        }

        binary
    }
}

/// GT exponentiation constraint for a specific step
pub struct GtExpStepConstraint {
    /// Reference to the witness data
    witness: Arc<JoltGtExpWitness>,
    /// Which step/bit this constraint is for (0 to 253)
    step: usize,
}

impl Constraint<Fq> for GtExpStepConstraint {
    fn evaluate(&self, x: &[Fq]) -> Fq {
        // The constraint formula: ρ_i(x) - ρ_{i-1}(x)^2 * A(x)^{b_i} - Q_i(x) * g(x)

        // Evaluate MLEs at point x
        let rho_prev = eval_mle_at_point(&self.witness.rho_mles[self.step], x);
        let rho_curr = eval_mle_at_point(&self.witness.rho_mles[self.step + 1], x);
        let quotient = eval_mle_at_point(&self.witness.quotient_mles[self.step], x);

        // Get base MLE and evaluate
        let base_mle = fq12_to_multilinear_evals(&self.witness.base);
        let base_eval = eval_mle_at_point(&base_mle, x);

        // Get g(x) polynomial MLE and evaluate
        let g_mle = get_g_mle(); // From jolt-optimizations
        let g_eval = eval_mle_at_point(&g_mle, x);

        // Apply constraint formula
        let bit = self.witness.bits[self.step];
        let base_power = if bit { base_eval } else { Fq::one() };

        rho_curr - rho_prev.square() * base_power - quotient * g_eval
    }
}

/// Helper function to evaluate MLE at arbitrary point
fn eval_mle_at_point(mle: &[Fq], point: &[Fq]) -> Fq {
    // Direct MLE evaluation, no field conversion needed
    let mut result = Fq::zero();
    for (i, &coeff) in mle.iter().enumerate() {
        let mut term = coeff;
        for (j, &p) in point.iter().enumerate() {
            let bit = (i >> j) & 1;
            term *= if bit == 1 { p } else { Fq::one() - p };
        }
        result += term;
    }
    result
}

/// Build all constraints from a witness collection
pub fn build_gt_exp_constraints(
    witnesses: &WitnessCollection<JoltWitness>,
) -> Vec<Box<dyn Constraint<Fq>>> {
    let mut constraints = Vec::new();

    // For each GT exp witness (tuple of OpId and witness)
    for (_op_id, witness) in witnesses.gt_exp.iter() {
        let witness_arc = Arc::new(witness.clone());

        // Create constraints for each bit (step)
        for step in 0..witness.bits.len() {
            let constraint = GtExpStepConstraint {
                witness: witness_arc.clone(),
                step,
            };
            constraints.push(Box::new(constraint) as Box<dyn Constraint<Fq>>);
        }
    }

    constraints
}

/// Create batched constraint system with flat indexing
pub fn create_batched_system(
    witnesses: &WitnessCollection<JoltWitness>,
) -> (BatchedConstraintSystem<Fq>, Vec<Box<dyn Constraint<Fq>>>) {
    let constraints = build_gt_exp_constraints(witnesses);
    let num_constraints = constraints.len();

    // Create system with 4 constraint variables (for MLE evaluation)
    let system = BatchedConstraintSystem::new(num_constraints, 4);

    (system, constraints)
}

/// Extract constraints from Dory proof
/// Note: Witness generation still uses Fr, but constraint evaluation uses Fq
pub fn extract_constraints_from_proof<T>(
    proof: &ArkDoryProof,
    setup: &ArkworksVerifierSetup,
    transcript: &mut T,
    point: &[<Fr as JoltField>::Challenge],
    evaluation: &Fr,
    commitment: &ArkGT,
) -> Result<(BatchedConstraintSystem<Fq>, Vec<Box<dyn Constraint<Fq>>>), ProofVerifyError>
where
    T: Transcript,
{
    // Generate witnesses using existing Fr-based flow
    let (witnesses, _hints) = <DoryCommitmentScheme as RecursionExt<Fr>>::witness_gen(
        proof, setup, transcript, point, evaluation, commitment,
    )?;

    // Build constraint system in Fq
    let (system, constraints) = create_batched_system(&witnesses);

    Ok((system, constraints))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;

    /// Helper to create evaluation point for specific constraint index
    fn create_point_at_index(system: &BatchedConstraintSystem<Fq>, index: usize) -> Vec<Fq> {
        let mut point = vec![];
        // Index vars: binary representation of index
        point.extend(system.index_to_binary(index));
        // Constraint vars: some test values
        point.extend(vec![Fq::from(1u64), Fq::from(2u64), Fq::from(3u64), Fq::from(4u64)]);
        point
    }

    #[test]
    #[serial]
    fn test_dory_witness_constraint_evaluation() {
        use crate::{
            field::JoltField,
            poly::{
                commitment::commitment_scheme::CommitmentScheme,
                dense_mlpoly::DensePolynomial,
                multilinear_polynomial::{MultilinearPolynomial, PolynomialEvaluation},
            },
        };
        use ark_ff::UniformRand;
        use rand::thread_rng;

        // 1. Setup Dory and create proof
        use crate::poly::commitment::dory::DoryGlobals;
        DoryGlobals::reset();
        let K = 1 << 2; // 2^2 = 4
        let T = 1 << 2; // 2^2 = 4
        DoryGlobals::initialize(K, T);

        // Setup
        let num_vars = 4;
        let prover_setup = DoryCommitmentScheme::setup_prover(num_vars);
        let verifier_setup = DoryCommitmentScheme::setup_verifier(&prover_setup);

        // Create polynomial
        let mut rng = thread_rng();
        let size = 1 << num_vars; // 2^4 = 16
        let coefficients: Vec<Fr> = (0..size).map(|_| Fr::rand(&mut rng)).collect();
        let poly = MultilinearPolynomial::LargeScalars(DensePolynomial::new(coefficients));

        // Commit
        let (commitment, hint) = DoryCommitmentScheme::commit(&poly, &prover_setup);

        // Create evaluation point
        let point: Vec<<Fr as JoltField>::Challenge> = (0..num_vars)
            .map(|_| <Fr as JoltField>::Challenge::random(&mut rng))
            .collect();

        // Generate proof
        let mut prover_transcript = crate::transcripts::Blake2bTranscript::new(b"test");
        let proof = DoryCommitmentScheme::prove(
            &prover_setup,
            &poly,
            &point,
            Some(hint),
            &mut prover_transcript,
        );

        // Evaluate polynomial
        let evaluation = PolynomialEvaluation::evaluate(&poly, &point);

        // 2. Extract witness and constraints
        let mut extract_transcript = crate::transcripts::Blake2bTranscript::new(b"test");
        let (witnesses, hints) = <DoryCommitmentScheme as RecursionExt<Fr>>::witness_gen(
            &proof,
            &verifier_setup,
            &mut extract_transcript,
            &point,
            &evaluation,
            &commitment,
        )
        .expect("Witness generation should succeed");

        let (system, constraints) = create_batched_system(&witnesses);
        println!("Created system with {} constraints", constraints.len());

        // 3. Test evaluation at specific points
        let test_indices = vec![0, 2, 3, 4];

        for &idx in &test_indices {
            if idx < constraints.len() {
                let eval_point = create_point_at_index(&system, idx);
                let result = system.evaluate(&eval_point, &constraints);
                println!("Evaluation at index {}: {:?}", idx, result);

                // Also verify this gives us the constraint evaluation
                let constraint_vars = &eval_point[system.num_index_vars..];
                let direct_constraint_eval = constraints[idx].evaluate(constraint_vars);
                println!("Direct constraint {} evaluation: {:?}", idx, direct_constraint_eval);
            }
        }

        // 4. Verify with hint
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
    }
}