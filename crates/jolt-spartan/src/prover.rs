//! Spartan prover: produces a proof of R1CS satisfiability.
//!
//! The prover pipeline:
//! 1. Compute $Az$, $Bz$, $Cz$ from the R1CS and witness.
//! 2. Check constraint satisfaction: $Az \circ Bz = Cz$.
//! 3. Commit to the witness polynomial.
//! 4. Run the outer sumcheck proving
//!    $\sum_x \widetilde{eq}(x, \tau) \cdot (\widetilde{Az}(x) \cdot \widetilde{Bz}(x) - \widetilde{Cz}(x)) = 0$.
//! 5. Run the inner sumcheck proving
//!    $\sum_y M(r_x, y) \cdot \tilde{z}(y) = \rho_A \cdot \widetilde{Az}(r_x) + \rho_B \cdot \widetilde{Bz}(r_x) + \rho_C \cdot \widetilde{Cz}(r_x)$.
//! 6. Provide an opening proof for the witness polynomial at $r_y$.

use jolt_field::Field;
use jolt_openings::CommitmentScheme;
use jolt_poly::{EqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::proof::SumcheckProof;
use jolt_sumcheck::{
    ClearRoundHandler, RoundHandler, SumcheckClaim, SumcheckCompute, SumcheckProver,
};
use jolt_transcript::Transcript;
use num_traits::Zero;

use crate::error::SpartanError;
use crate::key::SpartanKey;
use crate::proof::{RelaxedSpartanProof, SpartanProof};
use crate::r1cs::R1CS;
use crate::uni_skip::FirstRoundStrategy;

/// Below this threshold the overhead of Rayon work-stealing exceeds the
/// benefit. Matches jolt-poly's threshold.
#[cfg(feature = "parallel")]
pub(crate) const PAR_THRESHOLD: usize = 1024;

/// Stateless Spartan prover.
///
/// Orchestrates the full proving pipeline: constraint checking, witness
/// commitment, outer sumcheck, inner sumcheck, and opening proofs.
pub struct SpartanProver;

impl SpartanProver {
    /// Generates a Spartan proof that `witness` satisfies the R1CS encoded in `key`.
    ///
    /// # Protocol
    ///
    /// 1. Compute $Az$, $Bz$, $Cz$ and verify $Az \circ Bz = Cz$.
    /// 2. Commit to $\tilde{z}$ (the witness MLE).
    /// 3. Sample $\tau \in \mathbb{F}^{\log m}$ via Fiat-Shamir.
    /// 4. Run outer sumcheck on
    ///    $\sum_x \widetilde{eq}(x, \tau) \cdot (\widetilde{Az}(x) \cdot \widetilde{Bz}(x) - \widetilde{Cz}(x)) = 0$.
    /// 5. Absorb $\widetilde{Az}(r_x), \widetilde{Bz}(r_x), \widetilde{Cz}(r_x)$ and sample $\rho_A, \rho_B, \rho_C$.
    /// 6. Run inner sumcheck on $\sum_y M(r_x, y) \cdot \tilde{z}(y)$.
    /// 7. Produce an opening proof for the witness polynomial at $r_y$.
    #[tracing::instrument(skip_all, name = "SpartanProver::prove")]
    pub fn prove<PCS, T>(
        r1cs: &impl R1CS<PCS::Field>,
        key: &SpartanKey<PCS::Field>,
        witness: &[PCS::Field],
        pcs_setup: &PCS::ProverSetup,
        transcript: &mut T,
        strategy: FirstRoundStrategy,
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

        let (witness_commitment, _hint) = PCS::commit(witness_poly.evaluations(), pcs_setup);

        // Absorb commitment into transcript for Fiat-Shamir binding
        transcript.append_bytes(format!("{witness_commitment:?}").as_bytes());

        // Sample the random evaluation point tau for the eq polynomial
        let num_sc_vars = key.num_sumcheck_vars();
        let tau: Vec<PCS::Field> = (0..num_sc_vars)
            .map(|_| PCS::Field::from_u128(transcript.challenge()))
            .collect();

        // Save tau[0] for potential univariate skip before tau is consumed
        let tau_1 = tau.first().copied();

        let eq_poly = Polynomial::new(EqPolynomial::new(tau).evaluations());
        let mut outer_witness = OuterSumcheckCompute {
            eq: eq_poly,
            az: az_poly,
            bz: bz_poly,
            cz: cz_poly,
        };

        // Claimed sum is zero because Az*Bz = Cz for a satisfying witness
        let outer_claim = SumcheckClaim {
            num_vars: num_sc_vars,
            degree: 3,
            claimed_sum: PCS::Field::zero(),
        };

        let (outer_sumcheck_proof, r_x) = match strategy {
            FirstRoundStrategy::UnivariateSkip if num_sc_vars > 0 => {
                prove_outer_uniskip(&outer_claim, &mut outer_witness, transcript, tau_1.unwrap())
            }
            _ => {
                let handler = TrackingHandler::new(num_sc_vars);
                SumcheckProver::prove_with_handler(
                    &outer_claim,
                    &mut outer_witness,
                    transcript,
                    |c: u128| PCS::Field::from_u128(c),
                    handler,
                )
            }
        };

        // After outer sumcheck, all polynomials are bound to single values
        let az_eval = outer_witness.az.evaluations()[0];
        let bz_eval = outer_witness.bz.evaluations()[0];
        let cz_eval = outer_witness.cz.evaluations()[0];

        // Absorb evaluation claims into transcript
        transcript.append(&az_eval);
        transcript.append(&bz_eval);
        transcript.append(&cz_eval);

        // Sample random linear combination coefficients for inner sumcheck
        let rho_a = PCS::Field::from_u128(transcript.challenge());
        let rho_b = PCS::Field::from_u128(transcript.challenge());
        let rho_c = PCS::Field::from_u128(transcript.challenge());

        // Fuse partial-evaluation of all three matrix MLEs into a single combined
        // row: M(r_x, ·) = ρ_A·A(r_x,·) + ρ_B·B(r_x,·) + ρ_C·C(r_x,·).
        // This avoids allocating three intermediate polynomials.
        let combined_row = combined_partial_evaluate(
            key.a_mle(),
            key.b_mle(),
            key.c_mle(),
            &r_x,
            rho_a,
            rho_b,
            rho_c,
        );

        let num_witness_vars = key.num_witness_vars();
        let inner_claim = SumcheckClaim {
            num_vars: num_witness_vars,
            degree: 2,
            claimed_sum: rho_a * az_eval + rho_b * bz_eval + rho_c * cz_eval,
        };

        // The inner sumcheck consumes its witness copy via bind(). We evaluate
        // the original witness polynomial at r_y afterwards for the opening proof,
        // avoiding a full clone during witness construction.
        let mut inner_witness = InnerSumcheckCompute::new(combined_row, &witness_poly);

        let inner_handler = TrackingHandler::new(num_witness_vars);
        let (inner_sumcheck_proof, r_y) = SumcheckProver::prove_with_handler(
            &inner_claim,
            &mut inner_witness,
            transcript,
            |c: u128| PCS::Field::from_u128(c),
            inner_handler,
        );

        // Evaluate the original (unconsumed) witness polynomial at r_y
        let witness_eval = witness_poly.evaluate(&r_y);

        let pcs_poly: PCS::Polynomial = witness_poly.evaluations().to_vec().into();
        let witness_opening_proof =
            PCS::open(&pcs_poly, &r_y, witness_eval, pcs_setup, None, transcript);

        Ok(SpartanProof {
            witness_commitment,
            outer_sumcheck_proof,
            az_eval,
            bz_eval,
            cz_eval,
            inner_sumcheck_proof,
            witness_eval,
            witness_opening_proof,
        })
    }

    /// Generates a relaxed Spartan proof that `witness` and `error` satisfy
    /// $Az \circ Bz = u \cdot Cz + E$.
    ///
    /// Used by BlindFold after Nova folding. Witness and error commitments
    /// are passed in rather than computed here, since the caller (BlindFold)
    /// manages commitment operations.
    #[allow(clippy::too_many_arguments)]
    #[tracing::instrument(skip_all, name = "SpartanProver::prove_relaxed")]
    pub fn prove_relaxed<PCS, T>(
        r1cs: &impl R1CS<PCS::Field>,
        key: &SpartanKey<PCS::Field>,
        u: PCS::Field,
        witness: &[PCS::Field],
        error: &[PCS::Field],
        w_commitment: &PCS::Output,
        e_commitment: &PCS::Output,
        pcs_setup: &PCS::ProverSetup,
        transcript: &mut T,
    ) -> Result<RelaxedSpartanProof<PCS::Field, PCS>, SpartanError>
    where
        PCS: CommitmentScheme,
        T: Transcript<Challenge = u128>,
    {
        let (az, bz, cz) = r1cs.multiply_witness(witness);

        for i in 0..az.len() {
            if az[i] * bz[i] != u * cz[i] + error[i] {
                return Err(SpartanError::RelaxedConstraintViolation(i));
            }
        }

        let witness_poly = pad_to_power_of_two(witness, key.num_variables_padded);
        let az_poly = pad_to_power_of_two(&az, key.num_constraints_padded);
        let bz_poly = pad_to_power_of_two(&bz, key.num_constraints_padded);
        let cz_poly = pad_to_power_of_two(&cz, key.num_constraints_padded);
        let error_poly = pad_to_power_of_two(error, key.num_constraints_padded);

        // Absorb commitments into transcript
        transcript.append_bytes(format!("{w_commitment:?}").as_bytes());
        transcript.append_bytes(format!("{e_commitment:?}").as_bytes());

        // Sample tau for the eq polynomial
        let num_sc_vars = key.num_sumcheck_vars();
        let tau: Vec<PCS::Field> = (0..num_sc_vars)
            .map(|_| PCS::Field::from_u128(transcript.challenge()))
            .collect();

        let eq_poly = Polynomial::new(EqPolynomial::new(tau).evaluations());
        let mut outer_witness = RelaxedOuterSumcheckCompute {
            eq: eq_poly,
            az: az_poly,
            bz: bz_poly,
            cz: cz_poly,
            e: error_poly,
            u,
        };

        let outer_claim = SumcheckClaim {
            num_vars: num_sc_vars,
            degree: 3,
            claimed_sum: PCS::Field::zero(),
        };

        let handler = TrackingHandler::new(num_sc_vars);
        let (outer_sumcheck_proof, r_x) = SumcheckProver::prove_with_handler(
            &outer_claim,
            &mut outer_witness,
            transcript,
            |c: u128| PCS::Field::from_u128(c),
            handler,
        );

        let az_eval = outer_witness.az.evaluations()[0];
        let bz_eval = outer_witness.bz.evaluations()[0];
        let cz_eval = outer_witness.cz.evaluations()[0];
        let e_eval = outer_witness.e.evaluations()[0];

        transcript.append(&az_eval);
        transcript.append(&bz_eval);
        transcript.append(&cz_eval);
        transcript.append(&e_eval);

        let rho_a = PCS::Field::from_u128(transcript.challenge());
        let rho_b = PCS::Field::from_u128(transcript.challenge());
        let rho_c = PCS::Field::from_u128(transcript.challenge());

        let combined_row = combined_partial_evaluate(
            key.a_mle(),
            key.b_mle(),
            key.c_mle(),
            &r_x,
            rho_a,
            rho_b,
            rho_c,
        );

        let num_witness_vars = key.num_witness_vars();
        let inner_claim = SumcheckClaim {
            num_vars: num_witness_vars,
            degree: 2,
            claimed_sum: rho_a * az_eval + rho_b * bz_eval + rho_c * cz_eval,
        };

        let mut inner_witness = InnerSumcheckCompute::new(combined_row, &witness_poly);

        let inner_handler = TrackingHandler::new(num_witness_vars);
        let (inner_sumcheck_proof, r_y) = SumcheckProver::prove_with_handler(
            &inner_claim,
            &mut inner_witness,
            transcript,
            |c: u128| PCS::Field::from_u128(c),
            inner_handler,
        );

        let witness_eval = witness_poly.evaluate(&r_y);

        let pcs_witness: PCS::Polynomial = witness_poly.evaluations().to_vec().into();
        let witness_opening_proof = PCS::open(
            &pcs_witness,
            &r_y,
            witness_eval,
            pcs_setup,
            None,
            transcript,
        );

        // Open error polynomial at r_x
        let error_eval_at_rx = outer_witness.e.evaluations()[0];
        debug_assert_eq!(error_eval_at_rx, e_eval);
        let error_poly_full = pad_to_power_of_two(error, key.num_constraints_padded);
        let pcs_error: PCS::Polynomial = error_poly_full.evaluations().to_vec().into();
        let error_opening_proof = PCS::open(&pcs_error, &r_x, e_eval, pcs_setup, None, transcript);

        Ok(RelaxedSpartanProof {
            outer_sumcheck_proof,
            az_eval,
            bz_eval,
            cz_eval,
            e_eval,
            inner_sumcheck_proof,
            witness_eval,
            witness_opening_proof,
            error_opening_proof,
        })
    }
}

/// Pads `data` with zeros to `target_len` and wraps it as a [`Polynomial`].
///
/// R1CS matrices and witness vectors are not necessarily power-of-two sized,
/// but multilinear polynomials require $2^n$ evaluations. This zero-pads the
/// evaluation table so that unused hypercube entries contribute nothing.
fn pad_to_power_of_two<F: Field>(data: &[F], target_len: usize) -> Polynomial<F> {
    let mut evals = vec![F::zero(); target_len];
    let copy_len = data.len().min(target_len);
    evals[..copy_len].copy_from_slice(&data[..copy_len]);
    Polynomial::new(evals)
}

/// Builds the combined row polynomial $M(r_x, \cdot) = \rho_A \cdot A(r_x, \cdot) + \rho_B \cdot B(r_x, \cdot) + \rho_C \cdot C(r_x, \cdot)$
/// directly as a size-$n$ vector, avoiding any $O(m \cdot n)$ intermediate allocations.
///
/// Uses the identity: for Boolean $y$,
/// $$M(r_x, y) = \sum_{i \in \{0,1\}^{\log m}} \widetilde{eq}(i, r_x) \cdot [\rho_A A_{ij} + \rho_B B_{ij} + \rho_C C_{ij}]$$
/// where $j$ is the integer index of Boolean point $y$.
///
/// Total allocation: $O(m + n)$ instead of the naive $O(m \cdot n)$.
fn combined_partial_evaluate<F: Field>(
    a_mle: &Polynomial<F>,
    b_mle: &Polynomial<F>,
    c_mle: &Polynomial<F>,
    r_x: &[F],
    rho_a: F,
    rho_b: F,
    rho_c: F,
) -> Polynomial<F> {
    let m_padded = 1usize << r_x.len();
    let n_padded = a_mle.len() / m_padded;
    debug_assert_eq!(a_mle.len(), m_padded * n_padded);
    debug_assert_eq!(b_mle.len(), m_padded * n_padded);
    debug_assert_eq!(c_mle.len(), m_padded * n_padded);

    let eq_evals = EqPolynomial::new(r_x.to_vec()).evaluations();

    let a = a_mle.evaluations();
    let b = b_mle.evaluations();
    let c = c_mle.evaluations();

    let mut combined = vec![F::zero(); n_padded];

    #[cfg(feature = "parallel")]
    {
        if n_padded >= PAR_THRESHOLD {
            use rayon::prelude::*;
            combined.par_iter_mut().enumerate().for_each(|(j, cj)| {
                for (i, &eq_val) in eq_evals.iter().enumerate() {
                    let idx = i * n_padded + j;
                    *cj += eq_val * (rho_a * a[idx] + rho_b * b[idx] + rho_c * c[idx]);
                }
            });
        } else {
            for (i, &eq_val) in eq_evals.iter().enumerate() {
                let row = i * n_padded;
                for (j, cj) in combined.iter_mut().enumerate() {
                    let idx = row + j;
                    *cj += eq_val * (rho_a * a[idx] + rho_b * b[idx] + rho_c * c[idx]);
                }
            }
        }
    }

    #[cfg(not(feature = "parallel"))]
    {
        for (i, &eq_val) in eq_evals.iter().enumerate() {
            let row = i * n_padded;
            for (j, cj) in combined.iter_mut().enumerate() {
                let idx = row + j;
                *cj += eq_val * (rho_a * a[idx] + rho_b * b[idx] + rho_c * c[idx]);
            }
        }
    }

    Polynomial::new(combined)
}

/// Round handler that wraps [`ClearRoundHandler`] and records challenges.
///
/// Used to extract the sumcheck challenge vector ($r_x$ or $r_y$) alongside
/// the proof, since the standard handler discards challenges.
struct TrackingHandler<F: Field> {
    inner: ClearRoundHandler<F>,
    challenges: Vec<F>,
}

impl<F: Field> TrackingHandler<F> {
    fn new(capacity: usize) -> Self {
        Self {
            inner: ClearRoundHandler::with_capacity(capacity),
            challenges: Vec::with_capacity(capacity),
        }
    }
}

impl<F: Field> RoundHandler<F> for TrackingHandler<F> {
    type Proof = (SumcheckProof<F>, Vec<F>);

    fn absorb_round_poly(&mut self, poly: &UnivariatePoly<F>, transcript: &mut impl Transcript) {
        self.inner.absorb_round_poly(poly, transcript);
    }

    fn on_challenge(&mut self, challenge: F) {
        self.challenges.push(challenge);
    }

    fn finalize(self) -> (SumcheckProof<F>, Vec<F>) {
        (self.inner.finalize(), self.challenges)
    }
}

/// Runs the outer sumcheck with univariate skip for the first round.
///
/// The first round polynomial is computed analytically via the factored
/// identity $t_1(0) = t_1(1) = 0$, requiring a single evaluation at $X=2$
/// instead of four evaluations. Remaining rounds use standard enumeration.
fn prove_outer_uniskip<F: Field>(
    claim: &SumcheckClaim<F>,
    witness: &mut OuterSumcheckCompute<F>,
    transcript: &mut impl Transcript<Challenge = u128>,
    tau_1: F,
) -> (SumcheckProof<F>, Vec<F>) {
    use crate::uni_skip::uniskip_first_round;

    let mut handler = TrackingHandler::new(claim.num_vars);

    // First round: univariate skip
    let first_round_poly = uniskip_first_round(
        witness.eq.evaluations(),
        witness.az.evaluations(),
        witness.bz.evaluations(),
        witness.cz.evaluations(),
        tau_1,
    );
    handler.absorb_round_poly(&first_round_poly, transcript);
    let challenge = F::from_u128(transcript.challenge());
    handler.on_challenge(challenge);
    witness.bind(challenge);

    // Remaining rounds: standard
    for _round in 1..claim.num_vars {
        let round_poly = <OuterSumcheckCompute<F> as SumcheckCompute<F>>::round_polynomial(witness);
        handler.absorb_round_poly(&round_poly, transcript);
        let challenge = F::from_u128(transcript.challenge());
        handler.on_challenge(challenge);
        witness.bind(challenge);
    }

    handler.finalize()
}

/// Sequential outer round evaluation shared by both cfg paths.
fn outer_round_sequential<F: Field>(
    half: usize,
    eq_evals: &[F],
    az_evals: &[F],
    bz_evals: &[F],
    cz_evals: &[F],
) -> [F; 4] {
    let mut evals = [F::zero(); 4];
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

        for (t, eval) in evals.iter_mut().enumerate() {
            let x = F::from_u64(t as u64);
            let eq_val = eq_lo + x * eq_delta;
            let az_val = az_lo + x * az_delta;
            let bz_val = bz_lo + x * bz_delta;
            let cz_val = cz_lo + x * cz_delta;
            *eval += eq_val * (az_val * bz_val - cz_val);
        }
    }
    evals
}

/// Sequential inner round evaluation shared by both cfg paths.
fn inner_round_sequential<F: Field>(half: usize, row_evals: &[F], w_evals: &[F]) -> [F; 3] {
    let mut evals = [F::zero(); 3];
    for i in 0..half {
        let row_lo = row_evals[i];
        let row_hi = row_evals[i + half];
        let w_lo = w_evals[i];
        let w_hi = w_evals[i + half];

        let row_delta = row_hi - row_lo;
        let w_delta = w_hi - w_lo;

        for (t, eval) in evals.iter_mut().enumerate() {
            let x = F::from_u64(t as u64);
            let row_val = row_lo + x * row_delta;
            let w_val = w_lo + x * w_delta;
            *eval += row_val * w_val;
        }
    }
    evals
}

/// Witness for the outer Spartan sumcheck.
///
/// Represents the polynomial:
/// $$g(x) = \widetilde{eq}(x, \tau) \cdot (\widetilde{Az}(x) \cdot \widetilde{Bz}(x) - \widetilde{Cz}(x))$$
///
/// The round polynomial is computed by evaluating $g$ at $X \in \{0, 1, 2, 3\}$
/// (summing over the remaining Boolean hypercube) and interpolating the
/// resulting degree-3 univariate.
struct OuterSumcheckCompute<F: Field> {
    eq: Polynomial<F>,
    az: Polynomial<F>,
    bz: Polynomial<F>,
    cz: Polynomial<F>,
}

impl<F: Field> SumcheckCompute<F> for OuterSumcheckCompute<F> {
    fn round_polynomial(&self) -> UnivariatePoly<F> {
        let half = self.eq.evaluations().len() / 2;

        let eq_evals = self.eq.evaluations();
        let az_evals = self.az.evaluations();
        let bz_evals = self.bz.evaluations();
        let cz_evals = self.cz.evaluations();

        // Evaluate g(X) = sum_{x'} eq(X, x') * (az(X, x') * bz(X, x') - cz(X, x'))
        // at X in {0, 1, 2, 3} using the linear extension p(X) = lo + X*(hi - lo).
        let evals_at_points;

        #[cfg(feature = "parallel")]
        {
            if half >= PAR_THRESHOLD {
                use rayon::prelude::*;
                evals_at_points = (0..half)
                    .into_par_iter()
                    .fold(
                        || [F::zero(); 4],
                        |mut local, i| {
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

                            for (t, eval) in local.iter_mut().enumerate() {
                                let x = F::from_u64(t as u64);
                                let eq_val = eq_lo + x * eq_delta;
                                let az_val = az_lo + x * az_delta;
                                let bz_val = bz_lo + x * bz_delta;
                                let cz_val = cz_lo + x * cz_delta;
                                *eval += eq_val * (az_val * bz_val - cz_val);
                            }
                            local
                        },
                    )
                    .reduce(
                        || [F::zero(); 4],
                        |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]],
                    );
            } else {
                evals_at_points =
                    outer_round_sequential(half, eq_evals, az_evals, bz_evals, cz_evals);
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            evals_at_points = outer_round_sequential(half, eq_evals, az_evals, bz_evals, cz_evals);
        }

        let points: Vec<(F, F)> = (0..4)
            .map(|t| (F::from_u64(t as u64), evals_at_points[t]))
            .collect();

        UnivariatePoly::interpolate(&points)
    }

    fn bind(&mut self, challenge: F) {
        self.eq.bind(challenge);
        self.az.bind(challenge);
        self.bz.bind(challenge);
        self.cz.bind(challenge);
    }
}

/// Witness for the inner Spartan sumcheck.
///
/// Represents the polynomial:
/// $$h(y) = M(r_x, y) \cdot \tilde{z}(y)$$
///
/// where $M(r_x, y) = \rho_A \cdot \tilde{A}(r_x, y) + \rho_B \cdot \tilde{B}(r_x, y) + \rho_C \cdot \tilde{C}(r_x, y)$.
///
/// The round polynomial is degree 2 (product of two multilinear polynomials),
/// evaluated at $Y \in \{0, 1, 2\}$ and interpolated.
struct InnerSumcheckCompute<F: Field> {
    combined_row: Polynomial<F>,
    witness: Polynomial<F>,
}

impl<F: Field> InnerSumcheckCompute<F> {
    /// Creates a new inner sumcheck witness by cloning the witness polynomial.
    ///
    /// The clone is necessary because `bind()` mutates in place. The caller
    /// retains the original for `PCS::open` and `evaluate` at the derived `r_y`.
    fn new(combined_row: Polynomial<F>, witness: &Polynomial<F>) -> Self {
        Self {
            combined_row,
            witness: witness.clone(),
        }
    }
}

impl<F: Field> SumcheckCompute<F> for InnerSumcheckCompute<F> {
    fn round_polynomial(&self) -> UnivariatePoly<F> {
        let half = self.combined_row.evaluations().len() / 2;

        let row_evals = self.combined_row.evaluations();
        let w_evals = self.witness.evaluations();

        let evals_at_points;

        #[cfg(feature = "parallel")]
        {
            if half >= PAR_THRESHOLD {
                use rayon::prelude::*;
                evals_at_points = (0..half)
                    .into_par_iter()
                    .fold(
                        || [F::zero(); 3],
                        |mut local, i| {
                            let row_lo = row_evals[i];
                            let row_hi = row_evals[i + half];
                            let w_lo = w_evals[i];
                            let w_hi = w_evals[i + half];

                            let row_delta = row_hi - row_lo;
                            let w_delta = w_hi - w_lo;

                            for (t, eval) in local.iter_mut().enumerate() {
                                let x = F::from_u64(t as u64);
                                let row_val = row_lo + x * row_delta;
                                let w_val = w_lo + x * w_delta;
                                *eval += row_val * w_val;
                            }
                            local
                        },
                    )
                    .reduce(
                        || [F::zero(); 3],
                        |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
                    );
            } else {
                evals_at_points = inner_round_sequential(half, row_evals, w_evals);
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            evals_at_points = inner_round_sequential(half, row_evals, w_evals);
        }

        let points: Vec<(F, F)> = (0..3)
            .map(|t| (F::from_u64(t as u64), evals_at_points[t]))
            .collect();

        UnivariatePoly::interpolate(&points)
    }

    fn bind(&mut self, challenge: F) {
        self.combined_row.bind(challenge);
        self.witness.bind(challenge);
    }
}

/// Witness for the relaxed outer Spartan sumcheck.
///
/// $$g(x) = \widetilde{eq}(x, \tau) \cdot (\widetilde{Az}(x) \cdot \widetilde{Bz}(x) - u \cdot \widetilde{Cz}(x) - \widetilde{E}(x))$$
///
/// Degree is still 3 (Az·Bz dominates). The additional `-u·Cz - E` terms are
/// degree 1 each (linear in the sumcheck variable), so the maximum degree is
/// unchanged from the standard case.
struct RelaxedOuterSumcheckCompute<F: Field> {
    eq: Polynomial<F>,
    az: Polynomial<F>,
    bz: Polynomial<F>,
    cz: Polynomial<F>,
    e: Polynomial<F>,
    u: F,
}

/// Sequential evaluation of the relaxed outer round polynomial.
fn relaxed_outer_round_sequential<F: Field>(
    half: usize,
    eq_evals: &[F],
    az_evals: &[F],
    bz_evals: &[F],
    cz_evals: &[F],
    e_evals: &[F],
    u: F,
) -> [F; 4] {
    let mut evals = [F::zero(); 4];
    for i in 0..half {
        let eq_lo = eq_evals[i];
        let eq_hi = eq_evals[i + half];
        let az_lo = az_evals[i];
        let az_hi = az_evals[i + half];
        let bz_lo = bz_evals[i];
        let bz_hi = bz_evals[i + half];
        let cz_lo = cz_evals[i];
        let cz_hi = cz_evals[i + half];
        let e_lo = e_evals[i];
        let e_hi = e_evals[i + half];

        let eq_delta = eq_hi - eq_lo;
        let az_delta = az_hi - az_lo;
        let bz_delta = bz_hi - bz_lo;
        let cz_delta = cz_hi - cz_lo;
        let e_delta = e_hi - e_lo;

        for (t, eval) in evals.iter_mut().enumerate() {
            let x = F::from_u64(t as u64);
            let eq_val = eq_lo + x * eq_delta;
            let az_val = az_lo + x * az_delta;
            let bz_val = bz_lo + x * bz_delta;
            let cz_val = cz_lo + x * cz_delta;
            let e_val = e_lo + x * e_delta;
            *eval += eq_val * (az_val * bz_val - u * cz_val - e_val);
        }
    }
    evals
}

impl<F: Field> SumcheckCompute<F> for RelaxedOuterSumcheckCompute<F> {
    fn round_polynomial(&self) -> UnivariatePoly<F> {
        let half = self.eq.evaluations().len() / 2;

        let eq_evals = self.eq.evaluations();
        let az_evals = self.az.evaluations();
        let bz_evals = self.bz.evaluations();
        let cz_evals = self.cz.evaluations();
        let e_evals = self.e.evaluations();
        let u = self.u;

        let evals_at_points;

        #[cfg(feature = "parallel")]
        {
            if half >= PAR_THRESHOLD {
                use rayon::prelude::*;
                evals_at_points = (0..half)
                    .into_par_iter()
                    .fold(
                        || [F::zero(); 4],
                        |mut local, i| {
                            let eq_lo = eq_evals[i];
                            let eq_hi = eq_evals[i + half];
                            let az_lo = az_evals[i];
                            let az_hi = az_evals[i + half];
                            let bz_lo = bz_evals[i];
                            let bz_hi = bz_evals[i + half];
                            let cz_lo = cz_evals[i];
                            let cz_hi = cz_evals[i + half];
                            let e_lo = e_evals[i];
                            let e_hi = e_evals[i + half];

                            let eq_delta = eq_hi - eq_lo;
                            let az_delta = az_hi - az_lo;
                            let bz_delta = bz_hi - bz_lo;
                            let cz_delta = cz_hi - cz_lo;
                            let e_delta = e_hi - e_lo;

                            for (t, eval) in local.iter_mut().enumerate() {
                                let x = F::from_u64(t as u64);
                                let eq_val = eq_lo + x * eq_delta;
                                let az_val = az_lo + x * az_delta;
                                let bz_val = bz_lo + x * bz_delta;
                                let cz_val = cz_lo + x * cz_delta;
                                let e_val = e_lo + x * e_delta;
                                *eval += eq_val * (az_val * bz_val - u * cz_val - e_val);
                            }
                            local
                        },
                    )
                    .reduce(
                        || [F::zero(); 4],
                        |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]],
                    );
            } else {
                evals_at_points = relaxed_outer_round_sequential(
                    half, eq_evals, az_evals, bz_evals, cz_evals, e_evals, u,
                );
            }
        }

        #[cfg(not(feature = "parallel"))]
        {
            evals_at_points = relaxed_outer_round_sequential(
                half, eq_evals, az_evals, bz_evals, cz_evals, e_evals, u,
            );
        }

        let points: Vec<(F, F)> = (0..4)
            .map(|t| (F::from_u64(t as u64), evals_at_points[t]))
            .collect();

        UnivariatePoly::interpolate(&points)
    }

    fn bind(&mut self, challenge: F) {
        self.eq.bind(challenge);
        self.az.bind(challenge);
        self.bz.bind(challenge);
        self.cz.bind(challenge);
        self.e.bind(challenge);
    }
}
