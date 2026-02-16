use std::collections::HashMap;

use crate::curve::JoltCurve;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::{CommitmentScheme, ZkEvalCommitment};
use crate::poly::commitment::dory::{bind_opening_inputs, DoryContext, DoryGlobals, DoryLayout};
use crate::poly::opening_proof::{
    compute_advice_lagrange_factor, DoryOpeningState, OpeningAccumulator, OpeningId, SumcheckId,
};
use crate::transcripts::Transcript;
use crate::utils::errors::ProofVerifyError;
use crate::zkvm::claim_reductions::AdviceKind;
use crate::zkvm::stage8_opening_ids;
use crate::zkvm::witness::{all_committed_polynomials, CommittedPolynomial};

#[cfg(feature = "zk")]
use crate::poly::commitment::dory::bind_opening_inputs_zk;

use super::JoltVerifier;

#[derive(Clone, Debug)]
#[cfg_attr(not(feature = "zk"), allow(dead_code))]
pub(super) struct Stage8VerifyData<F: JoltField> {
    pub(super) opening_ids: Vec<OpeningId>,
    pub(super) constraint_coeffs: Vec<F>,
}

impl<
        'a,
        F: JoltField,
        C: JoltCurve,
        PCS: CommitmentScheme<Field = F> + ZkEvalCommitment<C>,
        ProofTranscript: Transcript,
    > JoltVerifier<'a, F, C, PCS, ProofTranscript>
{
    pub(super) fn verify_stage8(&mut self) -> Result<Stage8VerifyData<F>, ProofVerifyError> {
        let _guard = DoryGlobals::initialize_context(
            1 << self.one_hot_params.log_k_chunk,
            self.proof.trace_length.next_power_of_two(),
            DoryContext::Main,
            Some(self.proof.dory_layout),
        );

        let (opening_point, _) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::InstructionRa(0),
            SumcheckId::HammingWeightClaimReduction,
        );
        let log_k_chunk = self.one_hot_params.log_k_chunk;
        let r_address_stage7 = &opening_point.r[..log_k_chunk];

        let mut polynomial_claims = Vec::new();
        let mut scaling_factors = Vec::new();

        let (_, ram_inc_claim) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RamInc,
            SumcheckId::IncClaimReduction,
        );
        let (_, rd_inc_claim) = self.opening_accumulator.get_committed_polynomial_opening(
            CommittedPolynomial::RdInc,
            SumcheckId::IncClaimReduction,
        );

        // In AddressMajor, dense coefficients occupy every K-th column (sparse embedding),
        // so the Dory VMV includes a factor eq(r_addr, 0) = ∏(1 − r_addr_i).
        // In CycleMajor, dense rows are replicated K times, and the streaming VMV
        // sums row_factors = Σ_addr eq(r_addr, addr) = 1, so no correction is needed.
        let lagrange_factor: F = if DoryGlobals::get_layout() == DoryLayout::AddressMajor {
            r_address_stage7.iter().map(|r| F::one() - r).product()
        } else {
            F::one()
        };
        polynomial_claims.push((CommittedPolynomial::RamInc, ram_inc_claim * lagrange_factor));
        scaling_factors.push(lagrange_factor);
        polynomial_claims.push((CommittedPolynomial::RdInc, rd_inc_claim * lagrange_factor));
        scaling_factors.push(lagrange_factor);

        for i in 0..self.one_hot_params.instruction_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::InstructionRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            polynomial_claims.push((CommittedPolynomial::InstructionRa(i), claim));
            scaling_factors.push(F::one());
        }
        for i in 0..self.one_hot_params.bytecode_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::BytecodeRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            polynomial_claims.push((CommittedPolynomial::BytecodeRa(i), claim));
            scaling_factors.push(F::one());
        }
        for i in 0..self.one_hot_params.ram_d {
            let (_, claim) = self.opening_accumulator.get_committed_polynomial_opening(
                CommittedPolynomial::RamRa(i),
                SumcheckId::HammingWeightClaimReduction,
            );
            polynomial_claims.push((CommittedPolynomial::RamRa(i), claim));
            scaling_factors.push(F::one());
        }

        let mut include_trusted_advice = false;
        let mut include_untrusted_advice = false;

        if let Some((advice_point, advice_claim)) = self
            .opening_accumulator
            .get_advice_opening(AdviceKind::Trusted, SumcheckId::AdviceClaimReduction)
        {
            let lagrange_factor =
                compute_advice_lagrange_factor::<F>(&opening_point.r, &advice_point.r);
            polynomial_claims.push((
                CommittedPolynomial::TrustedAdvice,
                advice_claim * lagrange_factor,
            ));
            scaling_factors.push(lagrange_factor);
            include_trusted_advice = true;
        }

        if let Some((advice_point, advice_claim)) = self
            .opening_accumulator
            .get_advice_opening(AdviceKind::Untrusted, SumcheckId::AdviceClaimReduction)
        {
            let lagrange_factor =
                compute_advice_lagrange_factor::<F>(&opening_point.r, &advice_point.r);
            polynomial_claims.push((
                CommittedPolynomial::UntrustedAdvice,
                advice_claim * lagrange_factor,
            ));
            scaling_factors.push(lagrange_factor);
            include_untrusted_advice = true;
        }

        let gamma_powers: Vec<F> = self
            .transcript
            .challenge_scalar_powers(polynomial_claims.len());
        let constraint_coeffs: Vec<F> = gamma_powers
            .iter()
            .zip(&scaling_factors)
            .map(|(gamma, scale)| *gamma * *scale)
            .collect();

        let opening_ids = stage8_opening_ids(
            &self.one_hot_params,
            include_trusted_advice,
            include_untrusted_advice,
        );

        let claims: Vec<F> = polynomial_claims.iter().map(|(_, c)| *c).collect();
        let joint_claim: F = gamma_powers
            .iter()
            .zip(claims.iter())
            .map(|(gamma, claim)| *gamma * claim)
            .sum();

        let state = DoryOpeningState {
            opening_point: opening_point.r.clone(),
            gamma_powers: gamma_powers.clone(),
            polynomial_claims,
        };

        let mut commitments_map = HashMap::new();
        let expected_polynomials = all_committed_polynomials(&self.one_hot_params);
        if expected_polynomials.len() != self.proof.commitments.len() {
            return Err(ProofVerifyError::InvalidInputLength(
                expected_polynomials.len(),
                self.proof.commitments.len(),
            ));
        }
        for (polynomial, commitment) in expected_polynomials
            .into_iter()
            .zip(&self.proof.commitments)
        {
            commitments_map.insert(polynomial, commitment.clone());
        }

        if let Some(ref commitment) = self.trusted_advice_commitment {
            if state
                .polynomial_claims
                .iter()
                .any(|(p, _)| *p == CommittedPolynomial::TrustedAdvice)
            {
                commitments_map.insert(CommittedPolynomial::TrustedAdvice, commitment.clone());
            }
        }
        if let Some(ref commitment) = self.proof.untrusted_advice_commitment {
            if state
                .polynomial_claims
                .iter()
                .any(|(p, _)| *p == CommittedPolynomial::UntrustedAdvice)
            {
                commitments_map.insert(CommittedPolynomial::UntrustedAdvice, commitment.clone());
            }
        }

        let joint_commitment = self.compute_joint_commitment(&mut commitments_map, &state)?;

        let zk_mode = self.opening_accumulator.zk_mode;
        if zk_mode {
            PCS::verify(
                &self.proof.joint_opening_proof,
                &self.preprocessing.generators,
                &mut self.transcript,
                &opening_point.r,
                &F::zero(),
                &joint_commitment,
            )?;

            #[cfg(feature = "zk")]
            {
                let y_com: C::G1 = PCS::eval_commitment(&self.proof.joint_opening_proof)
                    .ok_or(ProofVerifyError::InvalidOpeningProof)?;
                bind_opening_inputs_zk::<F, C, _>(&mut self.transcript, &opening_point.r, &y_com);
            }
            #[cfg(not(feature = "zk"))]
            {
                return Err(ProofVerifyError::ZkFeatureRequired);
            }
        } else {
            PCS::verify(
                &self.proof.joint_opening_proof,
                &self.preprocessing.generators,
                &mut self.transcript,
                &opening_point.r,
                &joint_claim,
                &joint_commitment,
            )?;

            bind_opening_inputs::<F, _>(&mut self.transcript, &opening_point.r, &joint_claim);
        }

        Ok(Stage8VerifyData {
            opening_ids,
            constraint_coeffs,
        })
    }

    fn compute_joint_commitment(
        &self,
        commitment_map: &mut HashMap<CommittedPolynomial, PCS::Commitment>,
        state: &DoryOpeningState<F>,
    ) -> Result<PCS::Commitment, ProofVerifyError> {
        let mut rlc_map = HashMap::new();
        for (gamma, (poly, _claim)) in state
            .gamma_powers
            .iter()
            .zip(state.polynomial_claims.iter())
        {
            *rlc_map.entry(*poly).or_insert(F::zero()) += *gamma;
        }

        let (coeffs, commitments): (Vec<F>, Vec<PCS::Commitment>) = rlc_map
            .into_iter()
            .map(|(k, v)| {
                commitment_map
                    .remove(&k)
                    .map(|c| (v, c))
                    .ok_or(ProofVerifyError::InternalError)
            })
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .unzip();

        Ok(PCS::combine_commitments(&commitments, &coeffs))
    }
}
