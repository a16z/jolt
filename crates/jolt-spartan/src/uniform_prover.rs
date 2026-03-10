//! Prover for uniform (repeated-constraint) Spartan.
//!
//! Provides two proving modes:
//!
//! 1. **Dense mode** — materializes `Az`, `Bz`, `Cz` fully and runs the
//!    standard outer/inner sumcheck. Suitable for testing and small circuits.
//!
//! 2. **Streaming mode** — uses the `StreamingSumcheck` engine with
//!    pluggable `StreamingSumcheckWindow` and `LinearSumcheckStage`
//!    implementations. The concrete implementations are provided by
//!    jolt-zkvm for the RISC-V circuit.
//!
//! Both modes produce the same [`UniformSpartanProof`] structure.

use jolt_field::Field;
use jolt_poly::{EqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::proof::SumcheckProof;
use jolt_sumcheck::{
    ClearRoundHandler, RoundHandler, SumcheckClaim, SumcheckCompute, SumcheckProver,
};
use jolt_transcript::Transcript;

use crate::error::SpartanError;
use crate::uniform_key::UniformSpartanKey;

/// Proof structure for uniform Spartan (pure PIOP).
///
/// Contains the two sumcheck proofs and evaluation claims. The witness
/// commitment and opening proof are NOT included — the caller handles
/// PCS operations externally.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
#[serde(bound = "")]
pub struct UniformSpartanProof<F: Field> {
    pub outer_sumcheck_proof: SumcheckProof<F>,
    pub az_eval: F,
    pub bz_eval: F,
    pub cz_eval: F,
    pub inner_sumcheck_proof: SumcheckProof<F>,
    pub witness_eval: F,
}

/// Stateless uniform Spartan prover.
pub struct UniformSpartanProver;

impl UniformSpartanProver {
    /// Dense-mode proving: materializes full Az, Bz, Cz polynomials.
    ///
    /// The caller must commit to the witness and append the commitment to the
    /// transcript BEFORE calling this function. After this returns, the caller
    /// opens the witness polynomial at `r_y` via PCS.
    ///
    /// # Arguments
    ///
    /// * `key` — Uniform Spartan key with sparse constraint matrices.
    /// * `witness` — Flat interleaved witness vector of length `total_cols_padded`.
    ///   Layout: `witness[c * num_vars_padded + v]` is variable `v` in cycle `c`.
    /// * `transcript` — Fiat-Shamir transcript (with witness commitment already appended).
    #[tracing::instrument(skip_all, name = "UniformSpartanProver::prove_dense")]
    pub fn prove_dense<F, T>(
        key: &UniformSpartanKey<F>,
        witness: &[F],
        transcript: &mut T,
    ) -> Result<UniformSpartanProof<F>, SpartanError>
    where
        F: Field,
        T: Transcript<Challenge = u128>,
    {
        let (proof, _r_x, _r_y) = Self::prove_dense_with_challenges(key, witness, transcript)?;
        Ok(proof)
    }

    /// Like [`prove_dense`](Self::prove_dense) but also returns the outer and
    /// inner sumcheck challenge vectors `(r_x, r_y)`.
    ///
    /// Downstream stages need `r_x` and `r_y` to construct eq-weighted
    /// sumcheck claims and evaluate witness polynomials at the correct points.
    #[tracing::instrument(skip_all, name = "UniformSpartanProver::prove_dense_with_challenges")]
    #[allow(clippy::type_complexity)]
    pub fn prove_dense_with_challenges<F, T>(
        key: &UniformSpartanKey<F>,
        witness: &[F],
        transcript: &mut T,
    ) -> Result<(UniformSpartanProof<F>, Vec<F>, Vec<F>), SpartanError>
    where
        F: Field,
        T: Transcript<Challenge = u128>,
    {
        let total_rows = key.total_rows();
        let total_cols = key.total_cols();
        let total_rows_padded = total_rows.next_power_of_two();
        let total_cols_padded = total_cols.next_power_of_two();

        assert_eq!(
            witness.len(),
            total_cols_padded,
            "witness length {} != total_cols_padded {total_cols_padded}",
            witness.len(),
        );

        let (az, bz, cz) = {
            let _span = tracing::info_span!("materialize_Az_Bz_Cz").entered();
            let mut az = vec![F::zero(); total_rows_padded];
            let mut bz = vec![F::zero(); total_rows_padded];
            let mut cz = vec![F::zero(); total_rows_padded];

            for c in 0..key.num_cycles {
                let cycle_base = c * key.num_vars_padded;
                for (k, (a_row, (b_row, c_row))) in key
                    .a_sparse
                    .iter()
                    .zip(key.b_sparse.iter().zip(key.c_sparse.iter()))
                    .enumerate()
                {
                    let row = c * key.num_constraints_padded + k;

                    let mut a_val = F::zero();
                    for &(j, coeff) in a_row {
                        a_val += coeff * witness[cycle_base + j];
                    }

                    let mut b_val = F::zero();
                    for &(j, coeff) in b_row {
                        b_val += coeff * witness[cycle_base + j];
                    }

                    let mut c_val = F::zero();
                    for &(j, coeff) in c_row {
                        c_val += coeff * witness[cycle_base + j];
                    }

                    az[row] = a_val;
                    bz[row] = b_val;
                    cz[row] = c_val;
                }
            }
            (az, bz, cz)
        };

        {
            let _span = tracing::info_span!("constraint_check").entered();
            for i in 0..total_rows {
                if az[i] * bz[i] != cz[i] {
                    return Err(SpartanError::ConstraintViolation(i));
                }
            }
        }

        let witness_poly = Polynomial::new(witness.to_vec());

        let num_row_vars = log2_padded(total_rows_padded);
        let tau: Vec<F> = (0..num_row_vars)
            .map(|_| F::from_u128(transcript.challenge()))
            .collect();

        let eq_poly = Polynomial::new(EqPolynomial::new(tau).evaluations());
        let az_poly = Polynomial::new(az);
        let bz_poly = Polynomial::new(bz);
        let cz_poly = Polynomial::new(cz);

        let mut outer_witness = UniformOuterSumcheckCompute {
            eq: eq_poly,
            az: az_poly,
            bz: bz_poly,
            cz: cz_poly,
        };

        let outer_claim = SumcheckClaim {
            num_vars: num_row_vars,
            degree: 3,
            claimed_sum: F::zero(),
        };

        let handler = TrackingHandler::new(num_row_vars);
        let (outer_sumcheck_proof, r_x) = {
            let _span = tracing::info_span!("outer_sumcheck", num_vars = num_row_vars).entered();
            SumcheckProver::prove_with_handler(
                &outer_claim,
                &mut outer_witness,
                transcript,
                |c: u128| F::from_u128(c),
                handler,
            )
        };

        let az_eval = outer_witness.az.evaluations()[0];
        let bz_eval = outer_witness.bz.evaluations()[0];
        let cz_eval = outer_witness.cz.evaluations()[0];

        transcript.append(&az_eval);
        transcript.append(&bz_eval);
        transcript.append(&cz_eval);

        let rho_a = F::from_u128(transcript.challenge());
        let rho_b = F::from_u128(transcript.challenge());
        let rho_c = F::from_u128(transcript.challenge());

        let num_col_vars = log2_padded(total_cols_padded);
        let combined_row = {
            let _span = tracing::info_span!("combined_partial_evaluate").entered();
            combined_partial_evaluate_uniform(key, &r_x, rho_a, rho_b, rho_c, total_cols_padded)
        };

        let inner_claim = SumcheckClaim {
            num_vars: num_col_vars,
            degree: 2,
            claimed_sum: rho_a * az_eval + rho_b * bz_eval + rho_c * cz_eval,
        };

        let mut inner_witness = InnerSumcheckCompute {
            combined_row,
            witness: witness_poly.clone(),
        };

        let inner_handler = TrackingHandler::new(num_col_vars);
        let (inner_sumcheck_proof, r_y) = {
            let _span = tracing::info_span!("inner_sumcheck", num_vars = num_col_vars).entered();
            SumcheckProver::prove_with_handler(
                &inner_claim,
                &mut inner_witness,
                transcript,
                |c: u128| F::from_u128(c),
                inner_handler,
            )
        };

        let witness_eval = witness_poly.evaluate(&r_y);

        let proof = UniformSpartanProof {
            outer_sumcheck_proof,
            az_eval,
            bz_eval,
            cz_eval,
            inner_sumcheck_proof,
            witness_eval,
        };

        Ok((proof, r_x, r_y))
    }
}

/// Computes the combined row polynomial `M(r_x, ·)` for the inner sumcheck
/// using the uniform key's sparse structure.
///
/// $$M(r_x, y) = \rho_A \cdot A(r_x, y) + \rho_B \cdot B(r_x, y) + \rho_C \cdot C(r_x, y)$$
///
/// For a uniform R1CS, `r_x = (r_cycle, r_constraint)` and the matrix factors
/// as `M(r_x, y) = eq(r_cycle, y_cycle) · M_local(r_constraint, y_var)`.
///
/// We materialize this as a dense polynomial over all `total_cols_padded`
/// column indices.
fn combined_partial_evaluate_uniform<F: Field>(
    key: &UniformSpartanKey<F>,
    r_x: &[F],
    rho_a: F,
    rho_b: F,
    rho_c: F,
    total_cols_padded: usize,
) -> Polynomial<F> {
    let cycle_vars = key.num_cycle_vars();
    let (r_x_cycle, r_x_constraint) = r_x.split_at(cycle_vars);

    let eq_constraint = EqPolynomial::new(r_x_constraint.to_vec()).evaluations();
    let eq_cycle = EqPolynomial::new(r_x_cycle.to_vec()).evaluations();

    // Build the combined local row: M_local(r_constraint, v) for each variable v
    let mut local_row = vec![F::zero(); key.num_vars_padded];
    for (k, (a_row, (b_row, c_row))) in key
        .a_sparse
        .iter()
        .zip(key.b_sparse.iter().zip(key.c_sparse.iter()))
        .enumerate()
    {
        let w = eq_constraint[k];
        if w.is_zero() {
            continue;
        }

        for &(j, coeff) in a_row {
            local_row[j] += w * rho_a * coeff;
        }
        for &(j, coeff) in b_row {
            local_row[j] += w * rho_b * coeff;
        }
        for &(j, coeff) in c_row {
            local_row[j] += w * rho_c * coeff;
        }
    }

    // Expand to full column space: M(r_x, y) = eq(r_cycle, y_cycle) * local_row[y_var]
    let mut combined = vec![F::zero(); total_cols_padded];
    for (c, &eq_c) in eq_cycle.iter().enumerate() {
        if eq_c.is_zero() {
            continue;
        }
        let base = c * key.num_vars_padded;
        for (v, &local_val) in local_row.iter().enumerate() {
            if !local_val.is_zero() {
                combined[base + v] = eq_c * local_val;
            }
        }
    }

    Polynomial::new(combined)
}

/// Outer sumcheck witness for uniform Spartan (dense mode).
struct UniformOuterSumcheckCompute<F: Field> {
    eq: Polynomial<F>,
    az: Polynomial<F>,
    bz: Polynomial<F>,
    cz: Polynomial<F>,
}

impl<F: Field> SumcheckCompute<F> for UniformOuterSumcheckCompute<F> {
    fn round_polynomial(&self) -> UnivariatePoly<F> {
        let half = self.eq.evaluations().len() / 2;
        let eq_evals = self.eq.evaluations();
        let az_evals = self.az.evaluations();
        let bz_evals = self.bz.evaluations();
        let cz_evals = self.cz.evaluations();

        let mut evals_at_points = [F::zero(); 4];
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
        self.eq.bind(challenge);
        self.az.bind(challenge);
        self.bz.bind(challenge);
        self.cz.bind(challenge);
    }
}

/// Inner sumcheck witness for uniform Spartan.
struct InnerSumcheckCompute<F: Field> {
    combined_row: Polynomial<F>,
    witness: Polynomial<F>,
}

impl<F: Field> SumcheckCompute<F> for InnerSumcheckCompute<F> {
    fn round_polynomial(&self) -> UnivariatePoly<F> {
        let half = self.combined_row.evaluations().len() / 2;
        let row_evals = self.combined_row.evaluations();
        let w_evals = self.witness.evaluations();

        let mut evals_at_points = [F::zero(); 3];
        for i in 0..half {
            let row_lo = row_evals[i];
            let row_hi = row_evals[i + half];
            let w_lo = w_evals[i];
            let w_hi = w_evals[i + half];

            let row_delta = row_hi - row_lo;
            let w_delta = w_hi - w_lo;

            for (t, eval) in evals_at_points.iter_mut().enumerate() {
                let x = F::from_u64(t as u64);
                let row_val = row_lo + x * row_delta;
                let w_val = w_lo + x * w_delta;
                *eval += row_val * w_val;
            }
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

/// Round handler that records challenges alongside the proof.
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

#[inline]
fn log2_padded(n: usize) -> usize {
    n.trailing_zeros() as usize
}
