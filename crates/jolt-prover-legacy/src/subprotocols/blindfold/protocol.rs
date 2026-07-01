//! BlindFold Protocol Implementation
//!
//! The BlindFold protocol makes sumcheck proofs zero-knowledge by:
//! 1. Encoding verifier checks into a small R1CS
//! 2. Using Nova folding to hide the witness
//! 3. Using Spartan sumcheck to prove R1CS satisfaction without revealing the witness
//! 4. Hyrax-style openings to verify W(ry) and E(rx) evaluations

use crate::curve::{JoltCurve, JoltGroupElement};
use crate::field::JoltField;
use crate::poly::commitment::hyrax::{self as hyrax, HyraxOpeningProof};
use crate::poly::commitment::pedersen::PedersenGenerators;
use crate::poly::eq_poly::EqPolynomial;
use crate::poly::unipoly::CompressedUniPoly;
use crate::transcript_msgs::{FsAbsorb, FsChallenge, FsNargRead, FsNargWrite};
use crate::utils::math::Math;

use super::folding::{commit_cross_term_rows, compute_cross_term, sample_random_satisfying_pair};
use super::r1cs::VerifierR1CS;
use super::relaxed_r1cs::{RelaxedR1CSInstance, RelaxedR1CSWitness};
use super::spartan::{INNER_SUMCHECK_DEGREE_BOUND, SPARTAN_DEGREE_BOUND};

pub struct BlindFoldProver<'a, F: JoltField, C: JoltCurve<F = F>> {
    gens: &'a PedersenGenerators<C>,
    r1cs: &'a VerifierR1CS<F>,
    eval_commitment_gens: Option<(C::G1, C::G1)>,
}

impl<'a, F: JoltField, C: JoltCurve<F = F>> BlindFoldProver<'a, F, C> {
    pub fn new(
        gens: &'a PedersenGenerators<C>,
        r1cs: &'a VerifierR1CS<F>,
        eval_commitment_gens: Option<(C::G1, C::G1)>,
    ) -> Self {
        Self {
            gens,
            r1cs,
            eval_commitment_gens,
        }
    }

    #[tracing::instrument(skip_all, name = "BlindFoldProver::prove")]
    pub fn prove<T>(
        &self,
        real_instance: &RelaxedR1CSInstance<F, C>,
        real_witness: &RelaxedR1CSWitness<F>,
        real_z: &[F],
        transcript: &mut T,
    ) where
        T: FsChallenge<F> + FsAbsorb + FsNargWrite,
    {
        use super::spartan::{BlindFoldInnerSumcheckProver, BlindFoldSpartanProver};

        let mut rng = rand::thread_rng();

        write_instance_to_transcript(real_instance, transcript, InstanceRole::Real);

        let (random_instance, random_witness, random_z) = sample_random_satisfying_pair(
            self.gens,
            self.r1cs,
            self.eval_commitment_gens,
            &mut rng,
        );

        write_instance_to_transcript(&random_instance, transcript, InstanceRole::Random);

        let T = compute_cross_term(
            self.r1cs,
            real_z,
            real_instance.u,
            &random_z,
            random_instance.u,
        );

        let (R_E, C_E) = self.r1cs.hyrax.e_grid(self.r1cs.num_constraints);

        let (t_row_commitments, t_row_blindings) =
            commit_cross_term_rows(self.gens, &T, R_E, C_E, &mut rng);

        transcript.write_slice(&t_row_commitments);

        let r: F::Challenge = transcript.challenge_optimized();
        let r_field: F = r.into();

        let folded_instance = real_instance
            .fold(&random_instance, &t_row_commitments, r_field)
            .expect("prover-controlled fold inputs");
        let folded_witness = real_witness.fold(&random_witness, &T, &t_row_blindings, r_field);

        let RelaxedR1CSWitness {
            E: folded_E,
            W: folded_W,
            w_row_blindings,
            e_row_blindings,
        } = folded_witness;

        let folded_eval_outputs: Vec<F> = self
            .r1cs
            .extra_output_vars
            .iter()
            .map(|var| folded_W[var.index() - 1])
            .collect();
        let folded_eval_blindings: Vec<F> = self
            .r1cs
            .extra_blinding_vars
            .iter()
            .map(|var| folded_W[var.index() - 1])
            .collect();
        let folded_eval_output_openings: Vec<_> = self
            .r1cs
            .extra_output_vars
            .iter()
            .map(|&variable| {
                open_witness_variable(self.r1cs, &folded_W, &w_row_blindings, variable)
            })
            .collect();
        let folded_eval_blinding_openings: Vec<_> = self
            .r1cs
            .extra_blinding_vars
            .iter()
            .map(|&variable| {
                open_witness_variable(self.r1cs, &folded_W, &w_row_blindings, variable)
            })
            .collect();

        // Safe to reveal: each folded value is real + r * random, where the
        // random value is a one-time pad sampled by BlindFold.
        transcript.write_slice(&folded_eval_outputs);
        transcript.write_slice(&folded_eval_blindings);

        for opening in &folded_eval_output_openings {
            write_hyrax_opening(transcript, opening);
        }
        for opening in &folded_eval_blinding_openings {
            write_hyrax_opening(transcript, opening);
        }

        let mut folded_z = Vec::with_capacity(self.r1cs.num_vars);
        folded_z.push(folded_instance.u);
        folded_z.extend_from_slice(&folded_W);

        let padded_e_len = self.r1cs.num_constraints.next_power_of_two();
        let mut e_padded = folded_E;
        e_padded.resize(padded_e_len, F::zero());
        let e_for_hyrax = e_padded.clone();

        let num_vars = padded_e_len.log_2();
        let tau: Vec<_> = transcript.challenge_optimized_vec(num_vars);

        let mut spartan_prover =
            BlindFoldSpartanProver::new(self.r1cs, folded_instance.u, folded_z, e_padded, tau);

        let mut spartan_challenges: Vec<F::Challenge> = Vec::with_capacity(num_vars);
        let mut claim = F::zero();

        for _ in 0..num_vars {
            let poly = spartan_prover.compute_round_polynomial(claim);
            let compressed = poly.compress();
            transcript.write_slice(&compressed.coeffs_except_linear_term);

            let r_j = transcript.challenge_optimized();
            claim = poly.evaluate(&r_j);
            spartan_prover.bind_challenge(r_j);
            spartan_challenges.push(r_j);
        }

        let final_claims = spartan_prover.final_claims();
        let az_r = final_claims.az_r;
        let bz_r = final_claims.bz_r;
        let cz_r = final_claims.cz_r;

        let hyrax = &self.r1cs.hyrax;
        let log_R_E = R_E.log_2();
        let rx: Vec<F> = spartan_challenges.iter().map(|c| (*c).into()).collect();
        let e_combined_row = hyrax::combined_row(&e_for_hyrax, C_E, &rx[..log_R_E]);
        let e_combined_blinding = hyrax::combined_blinding(&e_row_blindings, &rx[..log_R_E]);
        let e_opening = HyraxOpeningProof {
            combined_row: e_combined_row,
            combined_blinding: e_combined_blinding,
        };

        transcript.write_slice(&[az_r, bz_r, cz_r]);
        write_hyrax_opening(transcript, &e_opening);

        let ra: F = transcript.challenge_optimized().into();
        let rb: F = transcript.challenge_optimized().into();
        let rc: F = transcript.challenge_optimized().into();

        let w_for_inner = folded_W.clone();
        let mut inner_prover = BlindFoldInnerSumcheckProver::new(
            self.r1cs,
            &spartan_challenges,
            w_for_inner,
            ra,
            rb,
            rc,
        );

        let inner_num_vars = inner_prover.num_vars();
        let mut inner_challenges: Vec<F::Challenge> = Vec::with_capacity(inner_num_vars);

        let (w_az, w_bz, w_cz) = spartan_prover.witness_contributions(&spartan_challenges);
        let mut inner_claim = ra * w_az + rb * w_bz + rc * w_cz;

        for _ in 0..inner_num_vars {
            let poly = inner_prover.compute_round_polynomial();
            debug_assert_eq!(
                poly.coeffs[0] + poly.coeffs.iter().sum::<F>(),
                inner_claim,
                "Inner sumcheck round polynomial p(0)+p(1) != claim"
            );

            let compressed = poly.compress();
            transcript.write_slice(&compressed.coeffs_except_linear_term);

            let r_j = transcript.challenge_optimized();
            inner_claim = poly.evaluate(&r_j);
            inner_prover.bind_challenge(r_j);
            inner_challenges.push(r_j);
        }

        // W opening at ry_w
        let log_R_prime = hyrax.log_R_prime();
        let ry_w: Vec<F> = inner_challenges.iter().map(|c| (*c).into()).collect();
        let w_combined_row = hyrax::combined_row(&folded_W, hyrax.C, &ry_w[..log_R_prime]);
        let w_combined_blinding = hyrax::combined_blinding(&w_row_blindings, &ry_w[..log_R_prime]);
        let w_opening = HyraxOpeningProof {
            combined_row: w_combined_row,
            combined_blinding: w_combined_blinding,
        };
        write_hyrax_opening(transcript, &w_opening);
    }
}

fn open_witness_variable<F: JoltField>(
    r1cs: &VerifierR1CS<F>,
    witness: &[F],
    row_blindings: &[F],
    variable: super::Variable,
) -> HyraxOpeningProof<F> {
    let witness_index = variable.index() - 1;
    let row = witness_index / r1cs.hyrax.C;
    let row_start = row * r1cs.hyrax.C;
    HyraxOpeningProof {
        combined_row: witness[row_start..row_start + r1cs.hyrax.C].to_vec(),
        combined_blinding: row_blindings[row],
    }
}

fn write_hyrax_opening<F: JoltField, T>(transcript: &mut T, opening: &HyraxOpeningProof<F>)
where
    T: FsChallenge<F> + FsAbsorb + FsNargWrite,
{
    transcript.write_slice(&opening.combined_row);
    transcript.write_slice(std::slice::from_ref(&opening.combined_blinding));
}

fn read_vec<F, T, U>(transcript: &mut T) -> Result<Vec<U>, BlindFoldVerifyError>
where
    F: JoltField,
    T: FsChallenge<F> + FsAbsorb + FsNargRead,
    U: ark_serialize::CanonicalDeserialize,
{
    transcript
        .read_slice()
        .map_err(|_| BlindFoldVerifyError::MalformedProof)
}

fn read_one<F, T, U>(transcript: &mut T) -> Result<U, BlindFoldVerifyError>
where
    F: JoltField,
    T: FsChallenge<F> + FsAbsorb + FsNargRead,
    U: ark_serialize::CanonicalDeserialize,
{
    let values: Vec<U> = read_vec(transcript)?;
    match <[U; 1]>::try_from(values) {
        Ok([value]) => Ok(value),
        Err(_) => Err(BlindFoldVerifyError::MalformedProof),
    }
}

fn read_hyrax_opening<F: JoltField, T>(
    transcript: &mut T,
    expected_len: usize,
) -> Result<HyraxOpeningProof<F>, BlindFoldVerifyError>
where
    T: FsChallenge<F> + FsAbsorb + FsNargRead,
{
    let combined_row: Vec<F> = read_vec(transcript)?;
    if combined_row.len() != expected_len {
        return Err(BlindFoldVerifyError::MalformedProof);
    }
    let combined_blinding: F = read_one(transcript)?;
    Ok(HyraxOpeningProof {
        combined_row,
        combined_blinding,
    })
}

fn verify_folded_eval_witness_bindings<F: JoltField, C: JoltCurve<F = F>>(
    r1cs: &VerifierR1CS<F>,
    gens: &PedersenGenerators<C>,
    folded_instance: &RelaxedR1CSInstance<F, C>,
    folded_eval_outputs: &[F],
    folded_eval_blindings: &[F],
    folded_eval_output_openings: &[HyraxOpeningProof<F>],
    folded_eval_blinding_openings: &[HyraxOpeningProof<F>],
) -> Result<(), BlindFoldVerifyError> {
    if folded_eval_output_openings.len() != r1cs.extra_output_vars.len()
        || folded_eval_blinding_openings.len() != r1cs.extra_blinding_vars.len()
        || folded_eval_outputs.len() != r1cs.extra_output_vars.len()
        || folded_eval_blindings.len() != r1cs.extra_blinding_vars.len()
    {
        return Err(BlindFoldVerifyError::MalformedProof);
    }

    for (index, (&variable, opening)) in r1cs
        .extra_output_vars
        .iter()
        .zip(folded_eval_output_openings)
        .enumerate()
    {
        let opened =
            verify_witness_variable_opening(r1cs, gens, folded_instance, variable, opening)?;
        if opened != folded_eval_outputs[index] {
            return Err(BlindFoldVerifyError::EvalWitnessMismatch);
        }
    }

    for (index, (&variable, opening)) in r1cs
        .extra_blinding_vars
        .iter()
        .zip(folded_eval_blinding_openings)
        .enumerate()
    {
        let opened =
            verify_witness_variable_opening(r1cs, gens, folded_instance, variable, opening)?;
        if opened != folded_eval_blindings[index] {
            return Err(BlindFoldVerifyError::EvalWitnessMismatch);
        }
    }

    Ok(())
}

fn verify_witness_variable_opening<F: JoltField, C: JoltCurve<F = F>>(
    r1cs: &VerifierR1CS<F>,
    gens: &PedersenGenerators<C>,
    folded_instance: &RelaxedR1CSInstance<F, C>,
    variable: super::Variable,
    opening: &HyraxOpeningProof<F>,
) -> Result<F, BlindFoldVerifyError> {
    let witness_index = variable.index() - 1;
    let row = witness_index / r1cs.hyrax.C;
    let column = witness_index % r1cs.hyrax.C;
    if opening.combined_row.len() != r1cs.hyrax.C {
        return Err(BlindFoldVerifyError::MalformedProof);
    }

    for (slot, value) in opening.combined_row.iter().enumerate() {
        if slot != column && *value != F::zero() {
            return Err(BlindFoldVerifyError::EvalWitnessMismatch);
        }
    }

    let witness_commitments =
        folded_instance.all_w_row_commitments(r1cs.hyrax.R_coeff, r1cs.hyrax.R_prime)?;
    if row >= witness_commitments.len()
        || !gens.verify(
            &witness_commitments[row],
            &opening.combined_row,
            &opening.combined_blinding,
        )
    {
        return Err(BlindFoldVerifyError::EvalWitnessOpeningFailed);
    }

    Ok(opening.combined_row[column])
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BlindFoldVerifyError {
    SpartanSumcheckFailed(usize),
    DegreeBoundExceeded { expected: usize, got: usize },
    MalformedProof,
    EOpeningFailed,
    OuterClaimMismatch,
    InnerSumcheckFailed(usize),
    WOpeningFailed,
    FinalClaimMismatch,
    EvalCommitmentMismatch,
    EvalWitnessMismatch,
    EvalWitnessOpeningFailed,
}

pub struct BlindFoldVerifierInput<C: JoltCurve> {
    pub round_commitments: Vec<C::G1>,
    /// Hyrax OC row commitments, extracted from stage proofs.
    pub output_claims_row_commitments: Vec<C::G1>,
    pub eval_commitments: Vec<C::G1>,
}

pub struct BlindFoldVerifier<'a, F: JoltField, C: JoltCurve<F = F>> {
    gens: &'a PedersenGenerators<C>,
    r1cs: &'a VerifierR1CS<F>,
    eval_commitment_gens: Option<(C::G1, C::G1)>,
}

impl<'a, F: JoltField, C: JoltCurve<F = F>> BlindFoldVerifier<'a, F, C> {
    pub fn new(
        gens: &'a PedersenGenerators<C>,
        r1cs: &'a VerifierR1CS<F>,
        eval_commitment_gens: Option<(C::G1, C::G1)>,
    ) -> Self {
        Self {
            gens,
            r1cs,
            eval_commitment_gens,
        }
    }

    #[tracing::instrument(skip_all, name = "BlindFoldVerifier::verify")]
    pub fn verify<T>(
        &self,
        input: BlindFoldVerifierInput<C>,
        transcript: &mut T,
    ) -> Result<(), BlindFoldVerifyError>
    where
        T: FsChallenge<F> + FsAbsorb + FsNargRead,
    {
        use super::spartan::{compute_L_w_at_ry, BlindFoldSpartanVerifier};

        let hyrax = &self.r1cs.hyrax;
        let (R_E, C_E) = hyrax.e_grid(self.r1cs.num_constraints);
        let expected_noncoeff_rows = hyrax.regular_noncoeff_rows();
        let expected_oc_rows = hyrax.output_claims_rows;

        if input.round_commitments.len() != hyrax.R_coeff {
            return Err(BlindFoldVerifyError::MalformedProof);
        }
        if input.output_claims_row_commitments.len() != expected_oc_rows {
            return Err(BlindFoldVerifyError::MalformedProof);
        }

        let real_instance = read_real_instance_from_transcript::<F, C, T>(
            F::one(),
            input.round_commitments,
            input.output_claims_row_commitments,
            vec![C::G1::zero(); R_E],
            input.eval_commitments,
            transcript,
        )?;
        if real_instance.noncoeff_row_commitments.len() != expected_noncoeff_rows {
            return Err(BlindFoldVerifyError::MalformedProof);
        }

        let random_instance = read_random_instance_from_transcript::<F, C, T>(transcript)?;
        if random_instance.noncoeff_row_commitments.len() != expected_noncoeff_rows
            || random_instance.round_commitments.len() != hyrax.R_coeff
            || random_instance.output_claims_row_commitments.len() != expected_oc_rows
            || random_instance.e_row_commitments.len() != R_E
        {
            return Err(BlindFoldVerifyError::MalformedProof);
        }

        let cross_term_row_commitments: Vec<C::G1> = read_vec(transcript)?;
        let r: F::Challenge = transcript.challenge_optimized();
        let r_field: F = r.into();

        let folded_instance =
            real_instance.fold(&random_instance, &cross_term_row_commitments, r_field)?;

        let folded_eval_outputs: Vec<F> = read_vec(transcript)?;
        let folded_eval_blindings: Vec<F> = read_vec(transcript)?;

        if let Some((g1_0, h1)) = self.eval_commitment_gens {
            if folded_eval_outputs.len() != folded_instance.eval_commitments.len()
                || folded_eval_blindings.len() != folded_instance.eval_commitments.len()
            {
                return Err(BlindFoldVerifyError::MalformedProof);
            }
            for (i, eval_com) in folded_instance.eval_commitments.iter().enumerate() {
                let expected = g1_0.scalar_mul(&folded_eval_outputs[i])
                    + h1.scalar_mul(&folded_eval_blindings[i]);
                if *eval_com != expected {
                    return Err(BlindFoldVerifyError::EvalCommitmentMismatch);
                }
            }
        }

        let mut folded_eval_output_openings = Vec::with_capacity(self.r1cs.extra_output_vars.len());
        for _ in 0..self.r1cs.extra_output_vars.len() {
            folded_eval_output_openings.push(read_hyrax_opening(transcript, self.r1cs.hyrax.C)?);
        }
        let mut folded_eval_blinding_openings =
            Vec::with_capacity(self.r1cs.extra_blinding_vars.len());
        for _ in 0..self.r1cs.extra_blinding_vars.len() {
            folded_eval_blinding_openings.push(read_hyrax_opening(transcript, self.r1cs.hyrax.C)?);
        }

        verify_folded_eval_witness_bindings(
            self.r1cs,
            self.gens,
            &folded_instance,
            &folded_eval_outputs,
            &folded_eval_blindings,
            &folded_eval_output_openings,
            &folded_eval_blinding_openings,
        )?;

        let num_vars = self.r1cs.num_constraints.next_power_of_two().log_2();

        let tau: Vec<_> = transcript.challenge_optimized_vec(num_vars);
        let mut claim = F::zero();
        let mut challenges: Vec<F::Challenge> = Vec::with_capacity(num_vars);

        for round in 0..num_vars {
            let coeffs_except_linear_term: Vec<F> = read_vec(transcript)?;
            if coeffs_except_linear_term.len() != SPARTAN_DEGREE_BOUND {
                return Err(BlindFoldVerifyError::DegreeBoundExceeded {
                    expected: SPARTAN_DEGREE_BOUND,
                    got: coeffs_except_linear_term.len(),
                });
            }
            let compressed_poly = CompressedUniPoly {
                coeffs_except_linear_term,
            };

            let poly = compressed_poly.decompress(&claim);
            let sum = poly.coeffs[0] + poly.coeffs.iter().sum::<F>();
            if sum != claim {
                return Err(BlindFoldVerifyError::SpartanSumcheckFailed(round));
            }

            let r_j = transcript.challenge_optimized();
            challenges.push(r_j);
            claim = poly.evaluate(&r_j);
        }

        let azbzcz: Vec<F> = read_vec(transcript)?;
        if azbzcz.len() != 3 {
            return Err(BlindFoldVerifyError::MalformedProof);
        }
        let az_r = azbzcz[0];
        let bz_r = azbzcz[1];
        let cz_r = azbzcz[2];

        let e_opening = read_hyrax_opening(transcript, C_E)?;

        let ra: F = transcript.challenge_optimized().into();
        let rb: F = transcript.challenge_optimized().into();
        let rc: F = transcript.challenge_optimized().into();

        let spartan_verifier = BlindFoldSpartanVerifier::new(self.r1cs, tau, folded_instance.u);

        let (pub_az, pub_bz, pub_cz) = spartan_verifier.public_contributions(&challenges);
        let mut inner_claim = ra * (az_r - pub_az) + rb * (bz_r - pub_bz) + rc * (cz_r - pub_cz);

        let w_padded_len = hyrax.R_prime * hyrax.C;
        let inner_num_vars = w_padded_len.log_2();

        let mut inner_challenges: Vec<F::Challenge> = Vec::with_capacity(inner_num_vars);

        for round in 0..inner_num_vars {
            let coeffs_except_linear_term: Vec<F> = read_vec(transcript)?;
            if coeffs_except_linear_term.len() != INNER_SUMCHECK_DEGREE_BOUND {
                return Err(BlindFoldVerifyError::DegreeBoundExceeded {
                    expected: INNER_SUMCHECK_DEGREE_BOUND,
                    got: coeffs_except_linear_term.len(),
                });
            }
            let compressed_poly = CompressedUniPoly {
                coeffs_except_linear_term,
            };

            let poly = compressed_poly.decompress(&inner_claim);
            let sum = poly.coeffs[0] + poly.coeffs.iter().sum::<F>();
            if sum != inner_claim {
                return Err(BlindFoldVerifyError::InnerSumcheckFailed(round));
            }

            let r_j = transcript.challenge_optimized();
            inner_challenges.push(r_j);
            inner_claim = poly.evaluate(&r_j);
        }

        let log_R_E = R_E.log_2();
        let rx: Vec<F> = challenges.iter().map(|c| (*c).into()).collect();
        let (rx_row, rx_col) = rx.split_at(log_R_E);

        let eq_rx_row: Vec<F> = EqPolynomial::evals(rx_row);
        let c_combined_e = C::g1_msm(&folded_instance.e_row_commitments, &eq_rx_row);
        let expected_e_com = self
            .gens
            .commit(&e_opening.combined_row, &e_opening.combined_blinding);
        if c_combined_e != expected_e_com {
            return Err(BlindFoldVerifyError::EOpeningFailed);
        }
        let e_r = hyrax::evaluate(&e_opening.combined_row, rx_col);

        let expected_outer = spartan_verifier.expected_claim(&challenges, az_r, bz_r, cz_r, e_r);
        if claim != expected_outer {
            return Err(BlindFoldVerifyError::OuterClaimMismatch);
        }

        let log_R_prime = hyrax.log_R_prime();
        let ry_w: Vec<F> = inner_challenges.iter().map(|c| (*c).into()).collect();
        let (ry_row, ry_col) = ry_w.split_at(log_R_prime);

        let all_w_rows = folded_instance.all_w_row_commitments(hyrax.R_coeff, hyrax.R_prime)?;
        let eq_ry_row: Vec<F> = EqPolynomial::evals(ry_row);
        let c_combined_w = C::g1_msm(&all_w_rows, &eq_ry_row);
        let w_opening = read_hyrax_opening(transcript, hyrax.C)?;
        let expected_w_com = self
            .gens
            .commit(&w_opening.combined_row, &w_opening.combined_blinding);
        if c_combined_w != expected_w_com {
            return Err(BlindFoldVerifyError::WOpeningFailed);
        }
        let w_ry = hyrax::evaluate(&w_opening.combined_row, ry_col);

        let l_w_at_ry = compute_L_w_at_ry(self.r1cs, &challenges, &inner_challenges, ra, rb, rc);
        let expected_inner_final = l_w_at_ry * w_ry;
        if inner_claim != expected_inner_final {
            return Err(BlindFoldVerifyError::FinalClaimMismatch);
        }

        Ok(())
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum InstanceRole {
    Real,
    Random,
}

fn write_instance_to_transcript<F: JoltField, C: JoltCurve<F = F>, T>(
    instance: &RelaxedR1CSInstance<F, C>,
    transcript: &mut T,
    role: InstanceRole,
) where
    T: FsChallenge<F> + FsAbsorb + FsNargWrite,
{
    match role {
        InstanceRole::Real => {
            transcript.absorb(&instance.u);
            transcript.absorb(&instance.round_commitments);
            transcript.absorb(&instance.output_claims_row_commitments);
            transcript.write_slice(&instance.noncoeff_row_commitments);
            transcript.absorb(&instance.e_row_commitments);
            transcript.absorb(&instance.eval_commitments);
        }
        InstanceRole::Random => {
            transcript.write_slice(std::slice::from_ref(&instance.u));
            transcript.write_slice(&instance.round_commitments);
            transcript.write_slice(&instance.output_claims_row_commitments);
            transcript.write_slice(&instance.noncoeff_row_commitments);
            transcript.write_slice(&instance.e_row_commitments);
            transcript.write_slice(&instance.eval_commitments);
        }
    }
}

fn read_real_instance_from_transcript<F: JoltField, C: JoltCurve<F = F>, T>(
    u: F,
    round_commitments: Vec<C::G1>,
    output_claims_row_commitments: Vec<C::G1>,
    e_row_commitments: Vec<C::G1>,
    eval_commitments: Vec<C::G1>,
    transcript: &mut T,
) -> Result<RelaxedR1CSInstance<F, C>, BlindFoldVerifyError>
where
    T: FsChallenge<F> + FsAbsorb + FsNargRead,
{
    transcript.absorb(&u);
    transcript.absorb(&round_commitments);
    transcript.absorb(&output_claims_row_commitments);
    let noncoeff_row_commitments: Vec<C::G1> = read_vec(transcript)?;
    transcript.absorb(&e_row_commitments);
    transcript.absorb(&eval_commitments);
    Ok(RelaxedR1CSInstance {
        u,
        round_commitments,
        output_claims_row_commitments,
        noncoeff_row_commitments,
        e_row_commitments,
        eval_commitments,
    })
}

fn read_random_instance_from_transcript<F: JoltField, C: JoltCurve<F = F>, T>(
    transcript: &mut T,
) -> Result<RelaxedR1CSInstance<F, C>, BlindFoldVerifyError>
where
    T: FsChallenge<F> + FsAbsorb + FsNargRead,
{
    let u: F = read_one(transcript)?;
    let round_commitments: Vec<C::G1> = read_vec(transcript)?;
    let output_claims_row_commitments: Vec<C::G1> = read_vec(transcript)?;
    let noncoeff_row_commitments: Vec<C::G1> = read_vec(transcript)?;
    let e_row_commitments: Vec<C::G1> = read_vec(transcript)?;
    let eval_commitments: Vec<C::G1> = read_vec(transcript)?;
    Ok(RelaxedR1CSInstance {
        u,
        round_commitments,
        output_claims_row_commitments,
        noncoeff_row_commitments,
        e_row_commitments,
        eval_commitments,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::curve::Bn254Curve;
    use crate::subprotocols::blindfold::r1cs::VerifierR1CSBuilder;
    use crate::subprotocols::blindfold::witness::{BlindFoldWitness, RoundWitness, StageWitness};
    use crate::subprotocols::blindfold::{BakedPublicInputs, StageConfig};
    use ark_bn254::Fr;
    use ark_std::Zero;
    use jolt_transcript::{Blake2b512, ProverState, VerifierState};
    use rand::rngs::StdRng;
    use rand::thread_rng;

    const TEST_INSTANCE: [u8; 32] = [0u8; 32];

    /// Prover transcript for the BlindFold tests (fixed test session/instance/sponge).
    fn bf_prover(label: &'static [u8]) -> ProverState<Blake2b512, StdRng> {
        jolt_transcript::prover_transcript(label, TEST_INSTANCE, Blake2b512::default())
    }

    fn bf_verifier<'a>(label: &'static [u8], narg: &'a [u8]) -> VerifierState<'a, Blake2b512> {
        jolt_transcript::verifier_transcript(label, TEST_INSTANCE, Blake2b512::default(), narg)
    }

    /// The verifier's public input is fully derived from the (honest) real instance.
    fn make_verifier_input(
        real_instance: &RelaxedR1CSInstance<Fr, Bn254Curve>,
    ) -> BlindFoldVerifierInput<Bn254Curve> {
        BlindFoldVerifierInput {
            round_commitments: real_instance.round_commitments.clone(),
            output_claims_row_commitments: real_instance.output_claims_row_commitments.clone(),
            eval_commitments: real_instance.eval_commitments.clone(),
        }
    }

    type TestInstance = (
        RelaxedR1CSInstance<Fr, Bn254Curve>,
        RelaxedR1CSWitness<Fr>,
        VerifierR1CS<Fr>,
        PedersenGenerators<Bn254Curve>,
        Vec<Fr>,
    );

    fn make_test_instance(
        configs: &[StageConfig],
        blindfold_witness: &BlindFoldWitness<Fr>,
    ) -> TestInstance {
        type F = Fr;
        let mut rng = thread_rng();

        let baked = BakedPublicInputs::from_witness(blindfold_witness, configs);
        let builder = VerifierR1CSBuilder::<F>::new(configs, &baked);
        let r1cs = builder.build();
        let gens = PedersenGenerators::<Bn254Curve>::deterministic(r1cs.hyrax.C + 1);

        let z = blindfold_witness.assign(&r1cs);
        r1cs.check_satisfaction(&z).unwrap();

        let witness: Vec<F> = z[1..].to_vec();

        let hyrax = &r1cs.hyrax;
        let hyrax_C = hyrax.C;
        let R_coeff = hyrax.R_coeff;
        let R_prime = hyrax.R_prime;

        let mut round_commitments = Vec::new();
        let mut w_row_blindings = vec![F::zero(); R_prime];

        for round_idx in 0..hyrax.total_rounds {
            let row_start = round_idx * hyrax_C;
            let blinding = F::random(&mut rng);
            let commitment = gens.commit(&witness[row_start..row_start + hyrax_C], &blinding);
            w_row_blindings[round_idx] = blinding;
            round_commitments.push(commitment);
        }

        let noncoeff_rows_count = hyrax.total_noncoeff_rows();
        let mut noncoeff_row_commitments = Vec::new();
        for row in 0..noncoeff_rows_count {
            let start = R_coeff * hyrax_C + row * hyrax_C;
            let end = (start + hyrax_C).min(witness.len());
            let blinding = F::random(&mut rng);
            noncoeff_row_commitments.push(gens.commit(&witness[start..end], &blinding));
            w_row_blindings[R_coeff + row] = blinding;
        }

        let (real_instance, real_witness) = RelaxedR1CSInstance::<F, Bn254Curve>::new_non_relaxed(
            &witness,
            r1cs.num_constraints,
            hyrax_C,
            round_commitments,
            Vec::new(),
            noncoeff_row_commitments,
            Vec::new(),
            w_row_blindings,
        );

        (real_instance, real_witness, r1cs, gens, z)
    }

    fn prove_to_narg(
        r1cs: &VerifierR1CS<Fr>,
        gens: &PedersenGenerators<Bn254Curve>,
        real_instance: &RelaxedR1CSInstance<Fr, Bn254Curve>,
        real_witness: &RelaxedR1CSWitness<Fr>,
        z: &[Fr],
        label: &'static [u8],
    ) -> Vec<u8> {
        let prover = BlindFoldProver::new(gens, r1cs, None);
        let mut prover_transcript = bf_prover(label);
        prover.prove(real_instance, real_witness, z, &mut prover_transcript);
        prover_transcript.narg_string().to_vec()
    }

    fn verify_narg(
        r1cs: &VerifierR1CS<Fr>,
        gens: &PedersenGenerators<Bn254Curve>,
        verifier_input: BlindFoldVerifierInput<Bn254Curve>,
        narg: &[u8],
        label: &'static [u8],
    ) -> Result<(), BlindFoldVerifyError> {
        let verifier = BlindFoldVerifier::new(gens, r1cs, None);
        let mut verifier_transcript = bf_verifier(label, narg);
        verifier.verify(verifier_input, &mut verifier_transcript)
    }

    fn prove_and_verify(
        r1cs: &VerifierR1CS<Fr>,
        gens: &PedersenGenerators<Bn254Curve>,
        real_instance: &RelaxedR1CSInstance<Fr, Bn254Curve>,
        real_witness: &RelaxedR1CSWitness<Fr>,
        z: &[Fr],
        label: &'static [u8],
    ) -> Result<(), BlindFoldVerifyError> {
        let narg = prove_to_narg(r1cs, gens, real_instance, real_witness, z, label);
        let verifier_input = make_verifier_input(real_instance);
        verify_narg(r1cs, gens, verifier_input, &narg, label)
    }

    #[test]
    fn test_blindfold_protocol_completeness() {
        type F = Fr;

        let configs = [StageConfig::new(1, 3)];
        let round = RoundWitness::new(
            vec![
                F::from_u64(40),
                F::from_u64(5),
                F::from_u64(10),
                F::from_u64(5),
            ],
            F::from_u64(3),
        );
        let blindfold_witness =
            BlindFoldWitness::new(F::from_u64(100), vec![StageWitness::new(vec![round])]);

        let (real_instance, real_witness, r1cs, gens, z) =
            make_test_instance(&configs, &blindfold_witness);

        let result = prove_and_verify(
            &r1cs,
            &gens,
            &real_instance,
            &real_witness,
            &z,
            b"BlindFold_test",
        );
        assert!(result.is_ok(), "Verification should succeed: {result:?}");
    }

    #[test]
    fn test_blindfold_protocol_soundness_wrong_proof_length() {
        type F = Fr;

        let configs = [StageConfig::new(1, 3)];
        let round = RoundWitness::new(
            vec![
                F::from_u64(40),
                F::from_u64(5),
                F::from_u64(10),
                F::from_u64(5),
            ],
            F::from_u64(3),
        );
        let blindfold_witness =
            BlindFoldWitness::new(F::from_u64(100), vec![StageWitness::new(vec![round])]);

        let (real_instance, real_witness, r1cs, gens, z) =
            make_test_instance(&configs, &blindfold_witness);

        let narg = prove_to_narg(
            &r1cs,
            &gens,
            &real_instance,
            &real_witness,
            &z,
            b"BlindFold_test",
        );
        let mut truncated = narg.clone();
        truncated.truncate(truncated.len() / 2);

        let verifier_input = make_verifier_input(&real_instance);
        let result = verify_narg(&r1cs, &gens, verifier_input, &truncated, b"BlindFold_test");

        assert!(
            result.is_err(),
            "Verification should fail on a truncated NARG: {result:?}"
        );
    }

    #[test]
    fn test_blindfold_rejects_malformed_random_round_commitments() {
        type F = Fr;

        let configs = [StageConfig::new(1, 3)];
        let round = RoundWitness::new(
            vec![
                F::from_u64(40),
                F::from_u64(5),
                F::from_u64(10),
                F::from_u64(5),
            ],
            F::from_u64(3),
        );
        let blindfold_witness =
            BlindFoldWitness::new(F::from_u64(100), vec![StageWitness::new(vec![round])]);

        let (real_instance, real_witness, r1cs, gens, z) =
            make_test_instance(&configs, &blindfold_witness);

        let narg = prove_to_narg(
            &r1cs,
            &gens,
            &real_instance,
            &real_witness,
            &z,
            b"BlindFold_test",
        );
        let mut verifier_input = make_verifier_input(&real_instance);
        let extra = verifier_input.round_commitments[0];
        verifier_input.round_commitments.push(extra);

        let result = verify_narg(&r1cs, &gens, verifier_input, &narg, b"BlindFold_test");

        assert_eq!(result, Err(BlindFoldVerifyError::MalformedProof));
    }

    #[test]
    fn test_blindfold_protocol_multi_round() {
        type F = Fr;

        let configs = [StageConfig::new(3, 3)];

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

        let c0_2 = F::from_u64(30);
        let c2_2 = F::from_u64(10);
        let c3_2 = F::from_u64(5);
        let c1_2 = next1 - F::from_u64(75);
        let round2 = RoundWitness::new(vec![c0_2, c1_2, c2_2, c3_2], F::from_u64(4));
        let next2 = round2.evaluate(F::from_u64(4));

        let c0_3 = F::from_u64(50);
        let c2_3 = F::from_u64(8);
        let c3_3 = F::from_u64(2);
        let c1_3 = next2 - F::from_u64(110);
        let round3 = RoundWitness::new(vec![c0_3, c1_3, c2_3, c3_3], F::from_u64(6));

        let initial_claim = F::from_u64(55);
        let blindfold_witness = BlindFoldWitness::new(
            initial_claim,
            vec![StageWitness::new(vec![round1, round2, round3])],
        );

        let (real_instance, real_witness, r1cs, gens, z) =
            make_test_instance(&configs, &blindfold_witness);

        let result = prove_and_verify(
            &r1cs,
            &gens,
            &real_instance,
            &real_witness,
            &z,
            b"BlindFold_multi",
        );
        assert!(
            result.is_ok(),
            "Multi-round verification should succeed: {result:?}"
        );
    }

    #[test]
    fn test_blindfold_soundness_tampered_az_r() {
        type F = Fr;

        let configs = [StageConfig::new(1, 3)];
        let round = RoundWitness::new(
            vec![
                F::from_u64(40),
                F::from_u64(5),
                F::from_u64(10),
                F::from_u64(5),
            ],
            F::from_u64(3),
        );
        let blindfold_witness =
            BlindFoldWitness::new(F::from_u64(100), vec![StageWitness::new(vec![round])]);

        let (real_instance, real_witness, r1cs, gens, z) =
            make_test_instance(&configs, &blindfold_witness);

        let mut narg = prove_to_narg(
            &r1cs,
            &gens,
            &real_instance,
            &real_witness,
            &z,
            b"BlindFold_test",
        );
        let i = narg.len() * 3 / 4;
        narg[i] ^= 0x01;

        let verifier_input = make_verifier_input(&real_instance);
        let result = verify_narg(&r1cs, &gens, verifier_input, &narg, b"BlindFold_test");
        assert!(
            result.is_err(),
            "Tampered az_r NARG region should cause verification failure"
        );
    }

    #[test]
    fn test_blindfold_soundness_tampered_w_combined_row() {
        type F = Fr;

        let configs = [StageConfig::new(1, 3)];
        let round = RoundWitness::new(
            vec![
                F::from_u64(40),
                F::from_u64(5),
                F::from_u64(10),
                F::from_u64(5),
            ],
            F::from_u64(3),
        );
        let blindfold_witness =
            BlindFoldWitness::new(F::from_u64(100), vec![StageWitness::new(vec![round])]);

        let (real_instance, real_witness, r1cs, gens, z) =
            make_test_instance(&configs, &blindfold_witness);

        let mut narg = prove_to_narg(
            &r1cs,
            &gens,
            &real_instance,
            &real_witness,
            &z,
            b"BlindFold_test",
        );
        let i = narg.len().saturating_sub(40);
        narg[i] ^= 0x01;

        let verifier_input = make_verifier_input(&real_instance);
        let result = verify_narg(&r1cs, &gens, verifier_input, &narg, b"BlindFold_test");
        assert!(
            result.is_err(),
            "Tampered W opening NARG region should fail: {result:?}"
        );
    }

    #[test]
    fn test_blindfold_soundness_tampered_e_combined_row() {
        type F = Fr;

        let configs = [StageConfig::new(1, 3)];
        let round = RoundWitness::new(
            vec![
                F::from_u64(40),
                F::from_u64(5),
                F::from_u64(10),
                F::from_u64(5),
            ],
            F::from_u64(3),
        );
        let blindfold_witness =
            BlindFoldWitness::new(F::from_u64(100), vec![StageWitness::new(vec![round])]);

        let (real_instance, real_witness, r1cs, gens, z) =
            make_test_instance(&configs, &blindfold_witness);

        let mut narg = prove_to_narg(
            &r1cs,
            &gens,
            &real_instance,
            &real_witness,
            &z,
            b"BlindFold_test",
        );
        let i = narg.len() * 7 / 10;
        narg[i] ^= 0x01;

        let verifier_input = make_verifier_input(&real_instance);
        let result = verify_narg(&r1cs, &gens, verifier_input, &narg, b"BlindFold_test");
        assert!(
            result.is_err(),
            "Tampered E opening NARG region should fail: {result:?}"
        );
    }
}
