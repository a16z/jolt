//! Spartan Sumcheck for BlindFold R1CS
//!
//! Proves relaxed R1CS satisfaction: (Az) ∘ (Bz) = u·(Cz) + E
//! via a sumcheck over: Σ_x eq(τ,x) · [Az(x)·Bz(x) - u·Cz(x) - E(x)] = 0
//!
//! The sumcheck polynomial has degree 2 (Az·Bz is the quadratic term).
//! After sumcheck, the verifier needs evaluation claims for:
//! - Witness contributions to Az, Bz, Cz at the derived point
//! - E at the sumcheck challenge point

use crate::field::JoltField;
use crate::poly::dense_mlpoly::DensePolynomial;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::multilinear_polynomial::BindingOrder;
use crate::poly::opening_proof::{
    OpeningAccumulator, OpeningPoint, ProverOpeningAccumulator, VerifierOpeningAccumulator,
    BIG_ENDIAN,
};
use crate::poly::unipoly::UniPoly;
use crate::subprotocols::blindfold::{InputClaimConstraint, OutputClaimConstraint};
use crate::subprotocols::sumcheck_prover::SumcheckInstanceProver;
use crate::subprotocols::sumcheck_verifier::{SumcheckInstanceParams, SumcheckInstanceVerifier};
use crate::transcripts::Transcript;
use crate::utils::math::Math;

use super::r1cs::VerifierR1CS;

/// Spartan sumcheck degree bound.
/// eq(τ,x) * Az(x) * Bz(x) is degree 3 in each variable.
const SPARTAN_DEGREE_BOUND: usize = 3;

/// Parameters for the BlindFold Spartan sumcheck
#[derive(Clone)]
pub struct BlindFoldSpartanParams<F: JoltField> {
    /// Random row selector τ (length = num_constraint_vars)
    pub tau: Vec<F::Challenge>,
    /// Folded scalar u (1 for non-relaxed, u' for folded)
    pub u: F,
    /// Number of constraint variables (ceil(log2(num_constraints)))
    pub num_vars: usize,
    /// Folded public inputs x
    pub x: Vec<F>,
}

impl<F: JoltField> BlindFoldSpartanParams<F> {
    pub fn new(tau: Vec<F::Challenge>, u: F, x: Vec<F>, num_constraints: usize) -> Self {
        let num_vars = num_constraints.next_power_of_two().log_2();
        Self {
            tau,
            u,
            num_vars,
            x,
        }
    }
}

impl<F: JoltField> SumcheckInstanceParams<F> for BlindFoldSpartanParams<F> {
    fn degree(&self) -> usize {
        SPARTAN_DEGREE_BOUND
    }

    fn num_rounds(&self) -> usize {
        self.num_vars
    }

    fn input_claim(&self, _accumulator: &dyn OpeningAccumulator<F>) -> F {
        // Spartan starts with claim = 0 (we're proving Σ eq(τ,x)·[...] = 0)
        F::zero()
    }

    fn normalize_opening_point(&self, challenges: &[F::Challenge]) -> OpeningPoint<BIG_ENDIAN, F> {
        challenges.to_vec().into()
    }

    fn input_claim_constraint(&self) -> InputClaimConstraint {
        // No input constraint - Spartan starts at 0
        InputClaimConstraint::default()
    }

    fn input_constraint_challenge_values(
        &self,
        _accumulator: &dyn OpeningAccumulator<F>,
    ) -> Vec<F> {
        Vec::new()
    }

    fn output_claim_constraint(&self) -> Option<OutputClaimConstraint> {
        // The output claim is verified by BlindFold's final check
        // We don't need a general constraint here
        None
    }

    fn output_constraint_challenge_values(&self, _sumcheck_challenges: &[F::Challenge]) -> Vec<F> {
        Vec::new()
    }
}

/// Prover for BlindFold Spartan sumcheck
///
/// Proves: Σ_x eq(τ,x) · [Az(x)·Bz(x) - u·Cz(x) - E(x)] = 0
pub struct BlindFoldSpartanProver<'a, F: JoltField> {
    /// Parameters
    params: BlindFoldSpartanParams<F>,
    /// A·Z evaluated at each constraint row, padded to power of 2
    az: DensePolynomial<F>,
    /// B·Z evaluated at each constraint row, padded to power of 2
    bz: DensePolynomial<F>,
    /// C·Z evaluated at each constraint row, padded to power of 2
    cz: DensePolynomial<F>,
    /// Error vector E, padded to power of 2
    e: DensePolynomial<F>,
    /// eq(τ, ·) polynomial as expandable table
    eq_tau: DensePolynomial<F>,
    /// Reference to R1CS for opening computation
    r1cs: &'a VerifierR1CS<F>,
    /// Full Z vector (for opening proofs)
    z: Vec<F>,
}

impl<'a, F: JoltField> BlindFoldSpartanProver<'a, F> {
    /// Create a new Spartan prover for the folded R1CS instance
    ///
    /// # Arguments
    /// * `r1cs` - The verifier R1CS
    /// * `u` - Folded scalar
    /// * `z` - Full folded Z vector [u, x..., W...]
    /// * `e` - Folded error vector E
    /// * `tau` - Random row selector from transcript
    pub fn new(
        r1cs: &'a VerifierR1CS<F>,
        u: F,
        z: Vec<F>,
        e: Vec<F>,
        tau: Vec<F::Challenge>,
    ) -> Self {
        let num_constraints = r1cs.num_constraints;
        let padded_size = num_constraints.next_power_of_two();

        // Compute A·Z, B·Z, C·Z
        let az_raw = r1cs.a.mul_vector(&z);
        let bz_raw = r1cs.b.mul_vector(&z);
        let cz_raw = r1cs.c.mul_vector(&z);

        // Pad to power of 2
        let mut az_padded = az_raw;
        let mut bz_padded = bz_raw;
        let mut cz_padded = cz_raw;
        let mut e_padded = e.clone();

        az_padded.resize(padded_size, F::zero());
        bz_padded.resize(padded_size, F::zero());
        cz_padded.resize(padded_size, F::zero());
        e_padded.resize(padded_size, F::zero());

        // Build eq(τ, ·) table
        let tau_f: Vec<F> = tau.iter().map(|c| (*c).into()).collect();
        let eq_tau = EqPolynomial::evals(&tau_f);

        let x = z[1..1 + r1cs.num_public_inputs].to_vec();

        let params = BlindFoldSpartanParams::new(tau, u, x, num_constraints);

        Self {
            params,
            az: DensePolynomial::new(az_padded),
            bz: DensePolynomial::new(bz_padded),
            cz: DensePolynomial::new(cz_padded),
            e: DensePolynomial::new(e_padded),
            eq_tau: DensePolynomial::new(eq_tau),
            r1cs,
            z,
        }
    }

    /// Compute the sumcheck polynomial for the current round
    ///
    /// g(X) = Σ_{x_i=0,1} eq(τ, prefix || X || suffix) · [Az·Bz - u·Cz - E]
    ///
    /// This is degree 3 in X (eq is linear, Az*Bz is quadratic).
    ///
    /// Uses big-endian indexing: pairs (i, i+half) differ in the MSB,
    /// corresponding to the first sumcheck variable.
    pub fn compute_round_polynomial(&self, _previous_claim: F) -> UniPoly<F> {
        let u = self.params.u;
        let n = self.az.len();
        let half = n / 2;

        // Compute evaluations at X = 0, 1, 2, 3
        let mut evals = vec![F::zero(); 4];

        // Iterate over pairs (i, i+half) which differ in the MSB (first variable)
        for i in 0..half {
            let eq_lo = self.eq_tau[i];
            let eq_hi = self.eq_tau[i + half];

            let az_lo = self.az[i];
            let az_hi = self.az[i + half];

            let bz_lo = self.bz[i];
            let bz_hi = self.bz[i + half];

            let cz_lo = self.cz[i];
            let cz_hi = self.cz[i + half];

            let e_lo = self.e[i];
            let e_hi = self.e[i + half];

            // Deltas for interpolation
            let eq_delta = eq_hi - eq_lo;
            let az_delta = az_hi - az_lo;
            let bz_delta = bz_hi - bz_lo;
            let cz_delta = cz_hi - cz_lo;
            let e_delta = e_hi - e_lo;

            // At X = 0: use _lo values
            let term_0 = eq_lo * (az_lo * bz_lo - u * cz_lo - e_lo);
            evals[0] += term_0;

            // At X = 1: use _hi values
            let term_1 = eq_hi * (az_hi * bz_hi - u * cz_hi - e_hi);
            evals[1] += term_1;

            // At X = 2: f(2) = f_lo + 2*delta
            let eq_2 = eq_lo + eq_delta + eq_delta;
            let az_2 = az_lo + az_delta + az_delta;
            let bz_2 = bz_lo + bz_delta + bz_delta;
            let cz_2 = cz_lo + cz_delta + cz_delta;
            let e_2 = e_lo + e_delta + e_delta;

            let term_2 = eq_2 * (az_2 * bz_2 - u * cz_2 - e_2);
            evals[2] += term_2;

            // At X = 3: f(3) = f_lo + 3*delta
            let eq_3 = eq_2 + eq_delta;
            let az_3 = az_2 + az_delta;
            let bz_3 = bz_2 + bz_delta;
            let cz_3 = cz_2 + cz_delta;
            let e_3 = e_2 + e_delta;

            let term_3 = eq_3 * (az_3 * bz_3 - u * cz_3 - e_3);
            evals[3] += term_3;
        }

        // Interpolate to get coefficients of degree-3 polynomial
        // p(X) = c0 + c1*X + c2*X^2 + c3*X^3
        UniPoly::from_evals(&evals)
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceProver<F, T> for BlindFoldSpartanProver<'_, F> {
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn compute_message(&mut self, _round: usize, previous_claim: F) -> UniPoly<F> {
        self.compute_round_polynomial(previous_claim)
    }

    fn ingest_challenge(&mut self, r_j: F::Challenge, _round: usize) {
        // Bind all polynomials to the challenge
        self.az.bind_parallel(r_j, BindingOrder::HighToLow);
        self.bz.bind_parallel(r_j, BindingOrder::HighToLow);
        self.cz.bind_parallel(r_j, BindingOrder::HighToLow);
        self.e.bind_parallel(r_j, BindingOrder::HighToLow);
        self.eq_tau.bind_parallel(r_j, BindingOrder::HighToLow);
    }

    fn cache_openings(
        &self,
        _accumulator: &mut ProverOpeningAccumulator<F>,
        _transcript: &mut T,
        _sumcheck_challenges: &[F::Challenge],
    ) {
        // The opening proofs for W and E contributions are handled by BlindFold
        // after verifying the final claim. We cache the final values here.
        //
        // After sumcheck completes, the bound polynomials contain the final evaluations:
        // - self.az[0] = Az(r)
        // - self.bz[0] = Bz(r)
        // - self.cz[0] = Cz(r)
        // - self.e[0] = E(r)
        // - self.eq_tau[0] = eq(τ, r)
        //
        // The verifier will reconstruct the public contributions and verify:
        // eq(τ, r) * (Az(r)*Bz(r) - u*Cz(r) - E(r)) = final_sumcheck_claim
    }

    #[cfg(feature = "allocative")]
    fn update_flamegraph(&self, _flamegraph: &mut allocative::FlameGraphBuilder) {
        // No-op
    }
}

/// Final claims from the Spartan sumcheck prover
#[derive(Clone, Debug)]
pub struct SpartanFinalClaims<F> {
    /// eq(τ, r) where r is the sumcheck challenge point
    pub eq_tau_r: F,
    /// Az(r) = A·Z evaluated at the challenge point
    pub az_r: F,
    /// Bz(r) = B·Z evaluated at the challenge point
    pub bz_r: F,
    /// Cz(r) = C·Z evaluated at the challenge point
    pub cz_r: F,
    /// E(r) = error vector evaluated at the challenge point
    pub e_r: F,
}

impl<F: JoltField> BlindFoldSpartanProver<'_, F> {
    /// Get the number of sumcheck rounds
    pub fn num_vars(&self) -> usize {
        self.params.num_vars
    }

    /// Bind all polynomials to a challenge value
    pub fn bind_challenge(&mut self, r_j: F::Challenge) {
        self.az.bind_parallel(r_j, BindingOrder::HighToLow);
        self.bz.bind_parallel(r_j, BindingOrder::HighToLow);
        self.cz.bind_parallel(r_j, BindingOrder::HighToLow);
        self.e.bind_parallel(r_j, BindingOrder::HighToLow);
        self.eq_tau.bind_parallel(r_j, BindingOrder::HighToLow);
    }

    /// Get the final claims after sumcheck completes
    pub fn final_claims(&self) -> SpartanFinalClaims<F> {
        SpartanFinalClaims {
            eq_tau_r: self.eq_tau[0],
            az_r: self.az[0],
            bz_r: self.bz[0],
            cz_r: self.cz[0],
            e_r: self.e[0],
        }
    }

    /// Compute the witness contribution to Az(r), Bz(r), Cz(r)
    ///
    /// Returns (w_az, w_bz, w_cz) where:
    /// - w_az = Σ_{j ∈ witness} A'[j] · W[j]
    /// - w_bz = Σ_{j ∈ witness} B'[j] · W[j]
    /// - w_cz = Σ_{j ∈ witness} C'[j] · W[j]
    ///
    /// And A'[j] = Σ_i A[i,j] · eq(r, i)
    pub fn witness_contributions(&self, sumcheck_challenges: &[F::Challenge]) -> (F, F, F) {
        let r: Vec<F> = sumcheck_challenges.iter().map(|c| (*c).into()).collect();
        let padded_size = self.r1cs.num_constraints.next_power_of_two();

        // Compute eq(r, ·) for all constraint rows
        let eq_r: Vec<F> = EqPolynomial::evals(&r);

        // Compute A'[j], B'[j], C'[j] for witness columns only
        let witness_start = 1 + self.r1cs.num_public_inputs;
        let witness_len = self.z.len() - witness_start;

        let mut a_prime = vec![F::zero(); witness_len];
        let mut b_prime = vec![F::zero(); witness_len];
        let mut c_prime = vec![F::zero(); witness_len];

        // Process sparse matrix A
        for &(row, col, ref val) in &self.r1cs.a.entries {
            if col >= witness_start && row < padded_size {
                let eq_val = eq_r[row];
                a_prime[col - witness_start] += eq_val * *val;
            }
        }

        // Process sparse matrix B
        for &(row, col, ref val) in &self.r1cs.b.entries {
            if col >= witness_start && row < padded_size {
                let eq_val = eq_r[row];
                b_prime[col - witness_start] += eq_val * *val;
            }
        }

        // Process sparse matrix C
        for &(row, col, ref val) in &self.r1cs.c.entries {
            if col >= witness_start && row < padded_size {
                let eq_val = eq_r[row];
                c_prime[col - witness_start] += eq_val * *val;
            }
        }

        // Compute inner products with witness
        let w = &self.z[witness_start..];
        let w_az: F = a_prime.iter().zip(w.iter()).map(|(a, w)| *a * *w).sum();
        let w_bz: F = b_prime.iter().zip(w.iter()).map(|(b, w)| *b * *w).sum();
        let w_cz: F = c_prime.iter().zip(w.iter()).map(|(c, w)| *c * *w).sum();

        (w_az, w_bz, w_cz)
    }
}

/// Verifier for BlindFold Spartan sumcheck
pub struct BlindFoldSpartanVerifier<'a, F: JoltField> {
    /// Parameters
    params: BlindFoldSpartanParams<F>,
    /// Reference to R1CS
    r1cs: &'a VerifierR1CS<F>,
}

impl<'a, F: JoltField> BlindFoldSpartanVerifier<'a, F> {
    pub fn new(r1cs: &'a VerifierR1CS<F>, tau: Vec<F::Challenge>, u: F, x: Vec<F>) -> Self {
        let params = BlindFoldSpartanParams::new(tau, u, x, r1cs.num_constraints);
        Self { params, r1cs }
    }

    /// Compute the public contribution to Az(r), Bz(r), Cz(r)
    ///
    /// Returns (pub_az, pub_bz, pub_cz) where:
    /// - pub_az = u·A'[0] + Σ_{j ∈ public} A'[j] · x[j-1]
    /// - pub_bz = u·B'[0] + Σ_{j ∈ public} B'[j] · x[j-1]
    /// - pub_cz = u·C'[0] + Σ_{j ∈ public} C'[j] · x[j-1]
    ///
    /// Where A'[j] = Σ_i A[i,j] · eq(r, i)
    pub fn public_contributions(&self, sumcheck_challenges: &[F::Challenge]) -> (F, F, F) {
        let r: Vec<F> = sumcheck_challenges.iter().map(|c| (*c).into()).collect();
        let padded_size = self.r1cs.num_constraints.next_power_of_two();

        // Compute eq(r, ·) for all constraint rows
        let eq_r: Vec<F> = EqPolynomial::evals(&r);

        // Public columns: 0 (u scalar) and 1..1+num_public_inputs (x)
        let public_len = 1 + self.r1cs.num_public_inputs;

        let mut a_prime_pub = vec![F::zero(); public_len];
        let mut b_prime_pub = vec![F::zero(); public_len];
        let mut c_prime_pub = vec![F::zero(); public_len];

        // Process sparse matrix A
        for &(row, col, ref val) in &self.r1cs.a.entries {
            if col < public_len && row < padded_size {
                let eq_val = eq_r[row];
                a_prime_pub[col] += eq_val * *val;
            }
        }

        // Process sparse matrix B
        for &(row, col, ref val) in &self.r1cs.b.entries {
            if col < public_len && row < padded_size {
                let eq_val = eq_r[row];
                b_prime_pub[col] += eq_val * *val;
            }
        }

        // Process sparse matrix C
        for &(row, col, ref val) in &self.r1cs.c.entries {
            if col < public_len && row < padded_size {
                let eq_val = eq_r[row];
                c_prime_pub[col] += eq_val * *val;
            }
        }

        // Compute public contributions: u·A'[0] + Σ_{j>0} A'[j]·x[j-1]
        let u = self.params.u;
        let x = &self.params.x;

        let pub_az = a_prime_pub[0] * u
            + a_prime_pub[1..]
                .iter()
                .zip(x.iter())
                .map(|(a, x)| *a * *x)
                .sum::<F>();

        let pub_bz = b_prime_pub[0] * u
            + b_prime_pub[1..]
                .iter()
                .zip(x.iter())
                .map(|(b, x)| *b * *x)
                .sum::<F>();

        let pub_cz = c_prime_pub[0] * u
            + c_prime_pub[1..]
                .iter()
                .zip(x.iter())
                .map(|(c, x)| *c * *x)
                .sum::<F>();

        (pub_az, pub_bz, pub_cz)
    }

    /// Compute eq(τ, r)
    pub fn eq_tau_at_r(&self, sumcheck_challenges: &[F::Challenge]) -> F {
        let tau: Vec<F> = self.params.tau.iter().map(|c| (*c).into()).collect();
        let r: Vec<F> = sumcheck_challenges.iter().map(|c| (*c).into()).collect();
        EqPolynomial::mle(&tau, &r)
    }

    /// Verify the final sumcheck claim given witness and error contributions
    ///
    /// The final claim should be:
    /// final_claim = eq(τ, r) · [Az(r)·Bz(r) - u·Cz(r) - E(r)]
    ///
    /// Where:
    /// - Az(r) = pub_az + w_az
    /// - Bz(r) = pub_bz + w_bz
    /// - Cz(r) = pub_cz + w_cz
    pub fn expected_claim(
        &self,
        sumcheck_challenges: &[F::Challenge],
        w_az: F,
        w_bz: F,
        w_cz: F,
        e_r: F,
    ) -> F {
        let eq_tau_r = self.eq_tau_at_r(sumcheck_challenges);
        let (pub_az, pub_bz, pub_cz) = self.public_contributions(sumcheck_challenges);

        let az_r = pub_az + w_az;
        let bz_r = pub_bz + w_bz;
        let cz_r = pub_cz + w_cz;

        eq_tau_r * (az_r * bz_r - self.params.u * cz_r - e_r)
    }
}

impl<F: JoltField, T: Transcript> SumcheckInstanceVerifier<F, T>
    for BlindFoldSpartanVerifier<'_, F>
{
    fn get_params(&self) -> &dyn SumcheckInstanceParams<F> {
        &self.params
    }

    fn expected_output_claim(
        &self,
        _accumulator: &VerifierOpeningAccumulator<F>,
        _sumcheck_challenges: &[F::Challenge],
    ) -> F {
        // The BlindFold verifier handles the final claim verification
        // after receiving witness and error contributions
        unimplemented!(
            "BlindFold Spartan verifier uses expected_claim() with explicit witness contributions"
        )
    }

    fn cache_openings(
        &self,
        _accumulator: &mut VerifierOpeningAccumulator<F>,
        _transcript: &mut T,
        _sumcheck_challenges: &[F::Challenge],
    ) {
        // Opening verification is handled by BlindFold after sumcheck
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::subprotocols::blindfold::r1cs::VerifierR1CSBuilder;
    use crate::subprotocols::blindfold::witness::{BlindFoldWitness, RoundWitness, StageWitness};
    use crate::subprotocols::blindfold::StageConfig;
    use crate::transcripts::KeccakTranscript;
    use ark_bn254::Fr;
    use ark_std::{One, Zero};

    #[test]
    fn test_eq_polynomial_binding_vs_mle() {
        type F = Fr;

        // Generate tau and challenges from transcript (using proper Challenge type)
        let mut transcript = KeccakTranscript::new(b"test_eq_binding");
        let tau: Vec<_> = transcript.challenge_vector_optimized::<F>(3);
        let challenges: Vec<_> = transcript.challenge_vector_optimized::<F>(3);

        // Convert tau to F for building eq table
        let tau_f: Vec<F> = tau.iter().map(|c| (*c).into()).collect();

        // Method 1: Build eq table and bind
        let eq_evals: Vec<F> = EqPolynomial::evals(&tau_f);
        let mut eq_table = DensePolynomial::new(eq_evals);
        for &c in &challenges {
            eq_table.bind_parallel(c, BindingOrder::HighToLow);
        }
        let bound_result: F = eq_table[0];

        // Method 2: Direct mle computation (convert Challenge to F first)
        let challenges_f: Vec<F> = challenges.iter().map(|c| (*c).into()).collect();
        let mle_result: F = EqPolynomial::mle(&tau_f, &challenges_f);

        assert_eq!(
            bound_result, mle_result,
            "EqPolynomial binding mismatch: bound={:?}, mle={:?}",
            bound_result, mle_result
        );
    }

    #[test]
    fn test_spartan_sumcheck_correctness() {
        type F = Fr;

        // Create R1CS
        let configs = [StageConfig::new(2, 3)];
        let builder = VerifierR1CSBuilder::<F>::new(&configs);
        let r1cs = builder.build();

        // Create valid witness
        let round1 = RoundWitness::new(
            vec![
                F::from_u64(20),
                F::from_u64(5),
                F::from_u64(7),
                F::from_u64(3),
            ],
            F::from_u64(2),
        );
        let next1 = round1.evaluate(F::from_u64(2));

        let c0_2 = F::from_u64(85);
        let c2_2 = F::from_u64(2);
        let c3_2 = F::from_u64(1);
        let c1_2 = next1 - F::from_u64(173);
        let round2 = RoundWitness::new(vec![c0_2, c1_2, c2_2, c3_2], F::from_u64(3));

        let initial_claim = F::from_u64(55);
        let blindfold_witness =
            BlindFoldWitness::new(initial_claim, vec![StageWitness::new(vec![round1, round2])]);
        let z = blindfold_witness.assign(&r1cs);

        // Verify R1CS is satisfied
        assert!(r1cs.is_satisfied(&z));

        // For non-relaxed R1CS, E = 0
        let e = vec![F::zero(); r1cs.num_constraints];

        // Create transcript and derive τ
        let mut transcript = KeccakTranscript::new(b"test_spartan");
        let tau: Vec<_> = transcript
            .challenge_vector_optimized::<F>(r1cs.num_constraints.next_power_of_two().log_2());

        // Create Spartan prover
        let mut prover = BlindFoldSpartanProver::new(&r1cs, F::one(), z.clone(), e, tau.clone());

        // Run sumcheck manually
        let num_rounds = prover.num_vars();
        let mut claim = F::zero(); // Spartan starts at 0
        let mut challenges = Vec::new();

        for round in 0..num_rounds {
            let poly = prover.compute_round_polynomial(claim);

            // Verify sum check property: p(0) + p(1) = claim
            let sum = poly.evaluate(&F::zero()) + poly.evaluate(&F::one());
            assert_eq!(sum, claim, "Round {round}: p(0) + p(1) != claim");

            // Sample challenge
            let r_j = transcript.challenge_scalar_optimized::<F>();
            challenges.push(r_j);

            // Update claim and bind polynomials
            claim = poly.evaluate(&r_j);
            prover.bind_challenge(r_j);
        }

        // Get final claims
        let final_claims = prover.final_claims();

        // Verify final claim matches expected value
        let expected_final = final_claims.eq_tau_r
            * (final_claims.az_r * final_claims.bz_r
                - F::one() * final_claims.cz_r
                - final_claims.e_r);
        assert_eq!(claim, expected_final, "Final claim mismatch");

        // Verify using verifier
        let verifier = BlindFoldSpartanVerifier::new(
            &r1cs,
            tau.clone(),
            F::one(),
            z[1..1 + r1cs.num_public_inputs].to_vec(),
        );

        // Check eq(tau, r) computation
        let verifier_eq_tau_r = verifier.eq_tau_at_r(&challenges);
        assert_eq!(
            final_claims.eq_tau_r, verifier_eq_tau_r,
            "eq(tau, r) mismatch: prover has bound eq, verifier recomputes"
        );

        // Check Az/Bz/Cz decomposition
        let (w_az, w_bz, w_cz) = prover.witness_contributions(&challenges);
        let (pub_az, pub_bz, pub_cz) = verifier.public_contributions(&challenges);

        assert_eq!(
            final_claims.az_r,
            pub_az + w_az,
            "Az(r) mismatch: prover={:?}, pub+w={:?}+{:?}",
            final_claims.az_r,
            pub_az,
            w_az
        );
        assert_eq!(
            final_claims.bz_r,
            pub_bz + w_bz,
            "Bz(r) mismatch: prover={:?}, pub+w={:?}+{:?}",
            final_claims.bz_r,
            pub_bz,
            w_bz
        );
        assert_eq!(
            final_claims.cz_r,
            pub_cz + w_cz,
            "Cz(r) mismatch: prover={:?}, pub+w={:?}+{:?}",
            final_claims.cz_r,
            pub_cz,
            w_cz
        );

        let verifier_expected =
            verifier.expected_claim(&challenges, w_az, w_bz, w_cz, final_claims.e_r);
        assert_eq!(claim, verifier_expected, "Verifier expected claim mismatch");
    }

    #[test]
    fn test_spartan_with_relaxed_r1cs() {
        type F = Fr;

        // Create R1CS
        let configs = [StageConfig::new(1, 3)];
        let builder = VerifierR1CSBuilder::<F>::new(&configs);
        let r1cs = builder.build();

        // Create valid witness
        let round = RoundWitness::new(
            vec![
                F::from_u64(40),
                F::from_u64(5),
                F::from_u64(10),
                F::from_u64(5),
            ],
            F::from_u64(3),
        );
        let initial_claim = F::from_u64(100);
        let blindfold_witness =
            BlindFoldWitness::new(initial_claim, vec![StageWitness::new(vec![round])]);
        let z = blindfold_witness.assign(&r1cs);

        // For relaxed R1CS with u != 1, compute E = Az ∘ Bz - u·Cz
        let u = F::from_u64(7); // Non-trivial u
        let mut z_relaxed = z.clone();
        z_relaxed[0] = u; // Set u scalar

        let az = r1cs.a.mul_vector(&z_relaxed);
        let bz = r1cs.b.mul_vector(&z_relaxed);
        let cz = r1cs.c.mul_vector(&z_relaxed);

        let e: Vec<F> = (0..r1cs.num_constraints)
            .map(|i| az[i] * bz[i] - u * cz[i])
            .collect();

        // Create transcript and derive τ
        let mut transcript = KeccakTranscript::new(b"test_spartan_relaxed");
        let tau: Vec<_> = transcript
            .challenge_vector_optimized::<F>(r1cs.num_constraints.next_power_of_two().log_2());

        // Create Spartan prover with relaxed instance
        let mut prover =
            BlindFoldSpartanProver::new(&r1cs, u, z_relaxed.clone(), e.clone(), tau.clone());

        // Run sumcheck
        let num_rounds = prover.num_vars();
        let mut claim = F::zero();
        let mut challenges = Vec::new();

        for round in 0..num_rounds {
            let poly = prover.compute_round_polynomial(claim);

            let sum = poly.evaluate(&F::zero()) + poly.evaluate(&F::one());
            assert_eq!(sum, claim, "Round {round}: sum check failed");

            let r_j = transcript.challenge_scalar_optimized::<F>();
            challenges.push(r_j);
            claim = poly.evaluate(&r_j);
            prover.bind_challenge(r_j);
        }

        // Final claim should be 0 (relaxed R1CS is satisfied)
        let final_claims = prover.final_claims();
        let expected_final = final_claims.eq_tau_r
            * (final_claims.az_r * final_claims.bz_r - u * final_claims.cz_r - final_claims.e_r);

        // Since (Az)∘(Bz) = u·(Cz) + E by construction, the expression should be 0
        assert_eq!(
            expected_final, claim,
            "Relaxed R1CS final claim should be 0"
        );
    }
}
