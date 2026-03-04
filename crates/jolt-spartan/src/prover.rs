//! Spartan prover: produces a proof of R1CS satisfiability.
//!
//! The prover pipeline:
//! 1. Compute $Az$, $Bz$, $Cz$ from the R1CS and witness.
//! 2. Check constraint satisfaction: $Az \circ Bz = Cz$.
//! 3. Commit to the witness polynomial.
//! 4. Run the outer sumcheck proving
//!    $\sum_x \widetilde{eq}(x, \tau) \cdot (\widetilde{Az}(x) \cdot \widetilde{Bz}(x) - \widetilde{Cz}(x)) = 0$.
//! 5. Provide an opening proof for the witness polynomial.

use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_poly::{DensePolynomial, EqPolynomial, MultilinearPolynomial, UnivariatePoly};
use jolt_sumcheck::{SumcheckClaim, SumcheckProver, SumcheckWitness};
use jolt_transcript::Transcript;
use num_traits::Zero;

use crate::error::SpartanError;
use crate::key::SpartanKey;
use crate::proof::SpartanProof;
use crate::r1cs::R1CS;

/// Stateless Spartan prover.
///
/// Orchestrates the full proving pipeline: constraint checking, witness
/// commitment, outer sumcheck, and opening proofs.
pub struct SpartanProver;

impl SpartanProver {
    /// Generates a Spartan proof that `witness` satisfies the R1CS encoded in `key`.
    ///
    /// # Protocol
    ///
    /// 1. Compute $Az$, $Bz$, $Cz$ and verify $Az \circ Bz = Cz$.
    /// 2. Commit to $\tilde{z}$ (the witness MLE).
    /// 3. Sample $\tau \in \mathbb{F}^{\log m}$ via Fiat-Shamir.
    /// 4. Run sumcheck on
    ///    $\sum_x \widetilde{eq}(x, \tau) \cdot (\widetilde{Az}(x) \cdot \widetilde{Bz}(x) - \widetilde{Cz}(x)) = 0$.
    /// 5. Sample a witness evaluation point $r_y$ and produce an opening proof.
    ///
    /// # Type parameters
    ///
    /// * `PCS` - Polynomial commitment scheme.
    /// * `T` - Fiat-Shamir transcript (must produce `u128` challenges).
    pub fn prove<PCS, T>(
        r1cs: &impl R1CS<PCS::Field>,
        key: &SpartanKey<PCS::Field>,
        witness: &[PCS::Field],
        pcs_setup: &PCS::ProverSetup,
        transcript: &mut T,
    ) -> Result<SpartanProof<PCS::Field, PCS>, SpartanError>
    where
        PCS: CommitmentScheme,
        T: Transcript<Challenge = u128>,
    {
        let (az, bz, cz) = r1cs.multiply_witness(witness);

        for i in 0..az.len() {
            if az[i] * bz[i] != cz[i] {
                return Err(SpartanError::ConstraintViolation(i));
            }
        }

        let witness_poly = pad_to_power_of_two(witness, key.num_variables_padded);
        let az_poly = pad_to_power_of_two(&az, key.num_constraints_padded);
        let bz_poly = pad_to_power_of_two(&bz, key.num_constraints_padded);
        let cz_poly = pad_to_power_of_two(&cz, key.num_constraints_padded);

        let witness_commitment = PCS::commit(&witness_poly, pcs_setup);

        // Absorb commitment into transcript for Fiat-Shamir binding
        transcript.append_bytes(format!("{witness_commitment:?}").as_bytes());

        // Sample the random evaluation point tau for the eq polynomial
        let num_sc_vars = key.num_sumcheck_vars();
        let tau: Vec<PCS::Field> = (0..num_sc_vars)
            .map(|_| PCS::Field::from_u128(transcript.challenge()))
            .collect();

        let eq_poly = DensePolynomial::new(EqPolynomial::new(tau).evaluations());
        let mut sc_witness = OuterSumcheckWitness {
            eq: eq_poly,
            az: az_poly,
            bz: bz_poly,
            cz: cz_poly,
        };

        // Claimed sum is zero because Az*Bz = Cz for a satisfying witness
        let claim = SumcheckClaim {
            num_vars: num_sc_vars,
            degree: 3,
            claimed_sum: PCS::Field::zero(),
        };

        let sumcheck_proof =
            SumcheckProver::prove(&claim, &mut sc_witness, transcript, |c: u128| {
                PCS::Field::from_u128(c)
            });

        // After sumcheck, all polynomials are bound to single values
        let az_eval = sc_witness.az.evaluations()[0];
        let bz_eval = sc_witness.bz.evaluations()[0];
        let cz_eval = sc_witness.cz.evaluations()[0];

        // Sample a fresh evaluation point for the witness polynomial
        let witness_point: Vec<PCS::Field> = (0..key.num_witness_vars())
            .map(|_| PCS::Field::from_u128(transcript.challenge()))
            .collect();
        let witness_eval = witness_poly.evaluate(&witness_point);

        let witness_opening_proof = PCS::prove(
            &witness_poly,
            &witness_point,
            witness_eval,
            pcs_setup,
            transcript,
        );

        Ok(SpartanProof {
            witness_commitment,
            sumcheck_proof,
            witness_eval,
            az_eval,
            bz_eval,
            cz_eval,
            witness_opening_proof,
        })
    }
}

/// Pads `data` with zeros to `target_len` and wraps it as a [`DensePolynomial`].
///
/// R1CS matrices and witness vectors are not necessarily power-of-two sized,
/// but multilinear polynomials require $2^n$ evaluations. This zero-pads the
/// evaluation table so that unused hypercube entries contribute nothing.
fn pad_to_power_of_two<F: Field>(data: &[F], target_len: usize) -> DensePolynomial<F> {
    let mut evals = vec![F::zero(); target_len];
    let copy_len = data.len().min(target_len);
    evals[..copy_len].copy_from_slice(&data[..copy_len]);
    DensePolynomial::new(evals)
}

/// Witness for the outer Spartan sumcheck.
///
/// Represents the polynomial:
/// $$g(x) = \widetilde{eq}(x, \tau) \cdot (\widetilde{Az}(x) \cdot \widetilde{Bz}(x) - \widetilde{Cz}(x))$$
///
/// The round polynomial is computed by evaluating $g$ at $X \in \{0, 1, 2, 3\}$
/// (summing over the remaining Boolean hypercube) and interpolating the
/// resulting degree-3 univariate.
struct OuterSumcheckWitness<F: Field> {
    eq: DensePolynomial<F>,
    az: DensePolynomial<F>,
    bz: DensePolynomial<F>,
    cz: DensePolynomial<F>,
}

impl<F: Field> SumcheckWitness<F> for OuterSumcheckWitness<F> {
    fn round_polynomial(&self) -> UnivariatePoly<F> {
        let half = self.eq.evaluations().len() / 2;

        // Evaluate g(X) = sum_{x'} eq(X, x') * (az(X, x') * bz(X, x') - cz(X, x'))
        // at X in {0, 1, 2, 3} using the linear extension p(X) = lo + X*(hi - lo).
        let mut evals_at_points = [F::zero(); 4];

        let eq_evals = self.eq.evaluations();
        let az_evals = self.az.evaluations();
        let bz_evals = self.bz.evaluations();
        let cz_evals = self.cz.evaluations();

        for i in 0..half {
            let eq_lo = eq_evals[i];
            let eq_hi = eq_evals[i + half];
            let az_lo = az_evals[i];
            let az_hi = az_evals[i + half];
            let bz_lo = bz_evals[i];
            let bz_hi = bz_evals[i + half];
            let cz_lo = cz_evals[i];
            let cz_hi = cz_evals[i + half];

            let eq_delta = eq_hi - eq_lo;
            let az_delta = az_hi - az_lo;
            let bz_delta = bz_hi - bz_lo;
            let cz_delta = cz_hi - cz_lo;

            for (t, eval) in evals_at_points.iter_mut().enumerate() {
                let x = F::from_u64(t as u64);
                let eq_val = eq_lo + x * eq_delta;
                let az_val = az_lo + x * az_delta;
                let bz_val = bz_lo + x * bz_delta;
                let cz_val = cz_lo + x * cz_delta;

                *eval += eq_val * (az_val * bz_val - cz_val);
            }
        }

        let points: Vec<(F, F)> = (0..4)
            .map(|t| (F::from_u64(t as u64), evals_at_points[t]))
            .collect();

        UnivariatePoly::interpolate(&points)
    }

    fn bind(&mut self, challenge: F) {
        self.eq.bind_in_place(challenge);
        self.az.bind_in_place(challenge);
        self.bz.bind_in_place(challenge);
        self.cz.bind_in_place(challenge);
    }
}
