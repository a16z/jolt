use std::collections::HashMap;
use std::sync::Arc;

use crate::curve::JoltCurve;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::{StreamingCommitmentScheme, ZkEvalCommitment};
use crate::poly::commitment::dory::{DoryContext, DoryGlobals, DoryLayout};
use crate::poly::opening_proof::{
    compute_advice_lagrange_factor, DoryOpeningState, OpeningAccumulator, SumcheckId,
};
use crate::poly::rlc_polynomial::{RLCStreamingData, TraceSource};
use crate::transcripts::Transcript;
use crate::zkvm::claim_reductions::AdviceKind;
use crate::zkvm::witness::CommittedPolynomial;

#[cfg(not(feature = "zk"))]
use crate::poly::commitment::dory::bind_opening_inputs;
#[cfg(feature = "zk")]
use crate::poly::commitment::dory::bind_opening_inputs_zk;
#[cfg(feature = "zk")]
use crate::zkvm::stage8_opening_ids;

use super::JoltCpuProver;

impl<
        'a,
        F: JoltField,
        C: JoltCurve,
        PCS: StreamingCommitmentScheme<Field = F> + ZkEvalCommitment<C>,
        ProofTranscript: Transcript,
    > JoltCpuProver<'a, F, C, PCS, ProofTranscript>
{
    #[tracing::instrument(skip_all)]
    pub(super) fn prove_stage8(
        &mut self,
        opening_proof_hints: HashMap<CommittedPolynomial, PCS::OpeningProofHint>,
    ) -> PCS::Proof {
        tracing::info!("Stage 8 proving (Dory batch opening)");

        let _guard = DoryGlobals::initialize_context(
            self.one_hot_params.k_chunk,
            self.padded_trace_len,
            DoryContext::Main,
            Some(DoryGlobals::get_layout()),
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

        #[cfg(feature = "zk")]
        let mut include_trusted_advice = false;
        #[cfg(feature = "zk")]
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
            #[cfg(feature = "zk")]
            {
                include_trusted_advice = true;
            }
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
            #[cfg(feature = "zk")]
            {
                include_untrusted_advice = true;
            }
        }

        let claims: Vec<F> = polynomial_claims.iter().map(|(_, c)| *c).collect();
        let gamma_powers: Vec<F> = self.transcript.challenge_scalar_powers(claims.len());
        #[cfg(feature = "zk")]
        let constraint_coeffs: Vec<F> = gamma_powers
            .iter()
            .zip(&scaling_factors)
            .map(|(gamma, scale)| *gamma * *scale)
            .collect();
        let joint_claim: F = gamma_powers
            .iter()
            .zip(claims.iter())
            .map(|(gamma, claim)| *gamma * claim)
            .sum();

        #[cfg(feature = "zk")]
        let opening_ids = stage8_opening_ids(
            &self.one_hot_params,
            include_trusted_advice,
            include_untrusted_advice,
        );

        let state = DoryOpeningState {
            opening_point: opening_point.r.clone(),
            gamma_powers,
            polynomial_claims,
        };

        let streaming_data = Arc::new(RLCStreamingData {
            bytecode: Arc::clone(&self.preprocessing.shared.bytecode),
            memory_layout: self.preprocessing.shared.memory_layout.clone(),
        });

        let mut advice_polys = HashMap::new();
        if let Some(poly) = self.advice.trusted_advice_polynomial.take() {
            advice_polys.insert(CommittedPolynomial::TrustedAdvice, poly);
        }
        if let Some(poly) = self.advice.untrusted_advice_polynomial.take() {
            advice_polys.insert(CommittedPolynomial::UntrustedAdvice, poly);
        }

        let (joint_poly, hint) = state.build_streaming_rlc::<PCS>(
            self.one_hot_params.clone(),
            TraceSource::Materialized(Arc::clone(&self.trace)),
            streaming_data,
            opening_proof_hints,
            advice_polys,
        );

        let (proof, _y_blinding) = PCS::prove(
            &self.preprocessing.generators,
            &joint_poly,
            &opening_point.r,
            Some(hint),
            &mut self.transcript,
        );

        #[cfg(feature = "zk")]
        {
            let y_com: C::G1 = PCS::eval_commitment(&proof).expect("ZK proof must have y_com");
            bind_opening_inputs_zk::<F, C, _>(&mut self.transcript, &opening_point.r, &y_com);
            self.stage8_zk_data = Some(super::Stage8ZkData {
                opening_ids,
                constraint_coeffs,
                joint_claim,
                y_blinding: _y_blinding.expect("ZK mode requires y_blinding"),
            });
        }
        #[cfg(not(feature = "zk"))]
        {
            bind_opening_inputs::<F, _>(&mut self.transcript, &opening_point.r, &joint_claim);
        }

        proof
    }
}
