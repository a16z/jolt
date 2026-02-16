use crate::curve::JoltCurve;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::{CommitmentScheme, ZkEvalCommitment};
use crate::poly::commitment::pedersen::PedersenGenerators;
use crate::poly::lagrange_poly::LagrangeHelper;
use crate::subprotocols::blindfold::{
    pedersen_generator_count_for_r1cs, BakedPublicInputs, BlindFoldVerifier,
    BlindFoldVerifierInput, ClaimBindingConfig, InputClaimConstraint, OutputClaimConstraint,
    StageConfig, ValueSource, VerifierR1CSBuilder,
};
use crate::subprotocols::sumcheck::SumcheckInstanceProof;
use crate::subprotocols::univariate_skip::UniSkipFirstRoundProofVariant;
use crate::transcripts::Transcript;
use crate::utils::errors::ProofVerifyError;
use crate::zkvm::r1cs::constraints::{
    OUTER_FIRST_ROUND_POLY_NUM_COEFFS, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
    PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS, PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
};

use super::opening::Stage8VerifyData;
use super::JoltVerifier;

impl<
        'a,
        F: JoltField,
        C: JoltCurve,
        PCS: CommitmentScheme<Field = F> + ZkEvalCommitment<C>,
        ProofTranscript: Transcript,
    > JoltVerifier<'a, F, C, PCS, ProofTranscript>
{
    #[allow(clippy::too_many_arguments)]
    pub(super) fn verify_blindfold(
        &mut self,
        sumcheck_challenges: &[Vec<F::Challenge>; 7],
        uniskip_challenges: [F::Challenge; 2],
        stage_output_constraints: &[Option<OutputClaimConstraint>; 7],
        output_constraint_challenge_values: &[Vec<F>; 7],
        stage_input_constraints: &[InputClaimConstraint; 7],
        input_constraint_challenge_values: &[Vec<F>; 7],
        stage1_batched_input: &InputClaimConstraint,
        stage2_batched_input: &InputClaimConstraint,
        stage1_batched_input_values: &[F],
        stage2_batched_input_values: &[F],
        stage8_data: &Stage8VerifyData<F>,
    ) -> Result<(), ProofVerifyError> {
        let stage_proofs = [
            &self.proof.stage1_sumcheck_proof,
            &self.proof.stage2_sumcheck_proof,
            &self.proof.stage3_sumcheck_proof,
            &self.proof.stage4_sumcheck_proof,
            &self.proof.stage5_sumcheck_proof,
            &self.proof.stage6_sumcheck_proof,
            &self.proof.stage7_sumcheck_proof,
        ];

        let outer_power_sums = LagrangeHelper::power_sums::<
            OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
            OUTER_FIRST_ROUND_POLY_NUM_COEFFS,
        >();
        let product_power_sums = LagrangeHelper::power_sums::<
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
            PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS,
        >();

        let mut stage_configs = Vec::new();
        let mut uniskip_indices: Vec<usize> = Vec::new();
        let mut regular_first_round_indices: Vec<usize> = Vec::new();
        let mut last_round_indices: Vec<usize> = Vec::new();

        for (stage_idx, proof) in stage_proofs.iter().enumerate() {
            if stage_idx < 2 {
                let uniskip_proof = if stage_idx == 0 {
                    &self.proof.stage1_uni_skip_first_round_proof
                } else {
                    &self.proof.stage2_uni_skip_first_round_proof
                };
                let poly_degree = uniskip_proof.poly_degree();

                let power_sums: Vec<i128> = if stage_idx == 0 {
                    outer_power_sums.to_vec()
                } else {
                    product_power_sums.to_vec()
                };

                uniskip_indices.push(stage_configs.len());

                let config = if stage_idx == 0 {
                    StageConfig::new_uniskip(poly_degree, power_sums)
                } else {
                    StageConfig::new_uniskip_chain(poly_degree, power_sums)
                };
                stage_configs.push(config);
            }

            regular_first_round_indices.push(stage_configs.len());

            let num_rounds = proof.num_rounds();
            for round_idx in 0..num_rounds {
                let poly_degree = match proof {
                    SumcheckInstanceProof::Standard(std_proof) => std_proof.compressed_polys
                        [round_idx]
                        .coeffs_except_linear_term
                        .len(),
                    SumcheckInstanceProof::Zk(zk_proof) => zk_proof.poly_degrees[round_idx],
                };
                let starts_new_chain = round_idx == 0;
                let config = if starts_new_chain {
                    StageConfig::new_chain(1, poly_degree)
                } else {
                    StageConfig::new(1, poly_degree)
                };
                stage_configs.push(config);
            }

            last_round_indices.push(stage_configs.len() - 1);
        }

        for (stage_idx, constraint) in stage_output_constraints.iter().enumerate() {
            if let Some(batched) = constraint {
                let last_round_idx = last_round_indices[stage_idx];
                stage_configs[last_round_idx].final_output =
                    Some(ClaimBindingConfig::with_constraint(batched.clone()));
            }
        }

        let uniskip_constraints = [
            stage_input_constraints[0].clone(),
            stage_input_constraints[1].clone(),
        ];
        for (i, constraint) in uniskip_constraints.iter().enumerate() {
            let idx = uniskip_indices[i];
            stage_configs[idx].initial_input =
                Some(ClaimBindingConfig::with_constraint(constraint.clone()));
        }

        let regular_constraints = [
            stage1_batched_input.clone(),
            stage2_batched_input.clone(),
            stage_input_constraints[2].clone(),
            stage_input_constraints[3].clone(),
            stage_input_constraints[4].clone(),
            stage_input_constraints[5].clone(),
            stage_input_constraints[6].clone(),
        ];
        for (i, constraint) in regular_constraints.iter().enumerate() {
            let idx = regular_first_round_indices[i];
            stage_configs[idx].initial_input =
                Some(ClaimBindingConfig::with_constraint(constraint.clone()));
        }

        let extra_constraint_terms: Vec<(ValueSource, ValueSource)> = stage8_data
            .opening_ids
            .iter()
            .enumerate()
            .map(|(i, id)| (ValueSource::challenge(i), ValueSource::opening(*id)))
            .collect();
        let extra_constraint = OutputClaimConstraint::linear(extra_constraint_terms);
        let extra_constraints = vec![extra_constraint];

        let mut baked_challenges: Vec<F> = Vec::new();
        for (stage_idx, stage_challenges) in sumcheck_challenges.iter().enumerate() {
            if stage_idx < 2 {
                baked_challenges.push(uniskip_challenges[stage_idx].into());
            }
            for challenge in stage_challenges.iter() {
                baked_challenges.push((*challenge).into());
            }
        }

        let all_input_challenge_values: [&[F]; 9] = [
            &input_constraint_challenge_values[0],
            stage1_batched_input_values,
            &input_constraint_challenge_values[1],
            stage2_batched_input_values,
            &input_constraint_challenge_values[2],
            &input_constraint_challenge_values[3],
            &input_constraint_challenge_values[4],
            &input_constraint_challenge_values[5],
            &input_constraint_challenge_values[6],
        ];
        let mut baked_input_challenges: Vec<F> = Vec::new();
        for expected_values in all_input_challenge_values.iter() {
            baked_input_challenges.extend_from_slice(expected_values);
        }

        let mut baked_output_challenges: Vec<F> = Vec::new();
        for expected_values in output_constraint_challenge_values.iter() {
            baked_output_challenges.extend_from_slice(expected_values);
        }

        let baked = BakedPublicInputs {
            challenges: baked_challenges,
            initial_claims: Vec::new(),
            batching_coefficients: Vec::new(),
            output_constraint_challenges: baked_output_challenges,
            input_constraint_challenges: baked_input_challenges,
            extra_constraint_challenges: stage8_data.constraint_coeffs.clone(),
        };

        let builder =
            VerifierR1CSBuilder::new_with_extra(&stage_configs, &extra_constraints, &baked);
        let r1cs = builder.build();

        let mut round_commitments: Vec<C::G1> = Vec::new();
        for (stage_idx, proof) in stage_proofs.iter().enumerate() {
            if stage_idx < 2 {
                let uniskip_proof = if stage_idx == 0 {
                    &self.proof.stage1_uni_skip_first_round_proof
                } else {
                    &self.proof.stage2_uni_skip_first_round_proof
                };
                if let UniSkipFirstRoundProofVariant::Zk(zk_uniskip) = uniskip_proof {
                    round_commitments.push(zk_uniskip.commitment);
                }
            }
            if let SumcheckInstanceProof::Zk(zk_proof) = proof {
                round_commitments.extend(zk_proof.round_commitments.iter().cloned());
            }
        }

        let eval_commitment = PCS::eval_commitment(&self.proof.joint_opening_proof)
            .ok_or(ProofVerifyError::InvalidOpeningProof)?;
        let eval_commitments = vec![eval_commitment];

        let verifier_input = BlindFoldVerifierInput {
            round_commitments,
            eval_commitments,
        };

        let pedersen_generator_count = pedersen_generator_count_for_r1cs(&r1cs);
        let pedersen_generators = PedersenGenerators::<C>::deterministic(pedersen_generator_count);
        let eval_commitment_gens =
            PCS::eval_commitment_gens_verifier(&self.preprocessing.generators);
        let verifier =
            BlindFoldVerifier::<_, _>::new(&pedersen_generators, &r1cs, eval_commitment_gens);
        let mut blindfold_transcript = ProofTranscript::new(b"BlindFold");

        verifier
            .verify(
                &self.proof.blindfold_proof,
                &verifier_input,
                &mut blindfold_transcript,
            )
            .map_err(|e| ProofVerifyError::BlindFoldError(format!("{e:?}")))?;

        tracing::debug!(
            "BlindFold verification passed: {} R1CS constraints",
            r1cs.num_constraints
        );

        Ok(())
    }
}
