use ark_serialize::CanonicalDeserialize;

use crate::curve::JoltCurve;
use crate::field::JoltField;
use crate::poly::commitment::commitment_scheme::{StreamingCommitmentScheme, ZkEvalCommitment};
use crate::poly::commitment::pedersen::PedersenGenerators;
use crate::poly::lagrange_poly::LagrangeHelper;
use crate::subprotocols::blindfold::{
    pedersen_generator_count_for_r1cs, BakedPublicInputs, BlindFoldProof, BlindFoldProver,
    BlindFoldWitness, ExtraConstraintWitness, FinalOutputWitness, InputClaimConstraint,
    OutputClaimConstraint, RelaxedR1CSInstance, RoundWitness, StageConfig, StageWitness,
    ValueSource, VerifierR1CSBuilder,
};
use crate::transcripts::Transcript;
use crate::zkvm::r1cs::constraints::{
    OUTER_FIRST_ROUND_POLY_NUM_COEFFS, OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
    PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS, PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
};

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
    pub(super) fn prove_blindfold(
        &mut self,
        joint_opening_proof: &PCS::Proof,
    ) -> BlindFoldProof<F, C> {
        let stage8_data = self
            .stage8_zk_data
            .as_ref()
            .expect("stage8_zk_data must be populated before prove_blindfold");
        tracing::info!("BlindFold proving");

        let mut rng = rand::thread_rng();

        let uniskip_stages = self.opening_accumulator.take_uniskip_stage_data();
        assert_eq!(
            uniskip_stages.len(),
            2,
            "Expected 2 uni-skip stages, got {}",
            uniskip_stages.len()
        );

        let zk_stages = self.opening_accumulator.take_zk_stage_data();
        assert_eq!(
            zk_stages.len(),
            7,
            "Expected 7 ZK stages, got {}",
            zk_stages.len()
        );

        let outer_power_sums = LagrangeHelper::power_sums::<
            OUTER_UNIVARIATE_SKIP_DOMAIN_SIZE,
            OUTER_FIRST_ROUND_POLY_NUM_COEFFS,
        >();
        let product_power_sums = LagrangeHelper::power_sums::<
            PRODUCT_VIRTUAL_UNIVARIATE_SKIP_DOMAIN_SIZE,
            PRODUCT_VIRTUAL_FIRST_ROUND_POLY_NUM_COEFFS,
        >();

        let mut stage_configs = Vec::new();
        let mut stage_witnesses = Vec::new();
        let mut initial_claims = Vec::new();

        for (stage_idx, zk_data) in zk_stages.iter().enumerate() {
            if stage_idx < 2 {
                let uniskip = &uniskip_stages[stage_idx];
                let coeffs = &uniskip.poly_coeffs;
                let challenge: F = uniskip.challenge.into();
                let poly_degree = coeffs.len() - 1;
                let claimed_sum = uniskip.input_claim;

                initial_claims.push(claimed_sum);

                let power_sums: Vec<i128> = if stage_idx == 0 {
                    outer_power_sums.to_vec()
                } else {
                    product_power_sums.to_vec()
                };

                let input_constraint = uniskip.input_constraint.clone();
                let input_challenge_values = uniskip.input_constraint_challenge_values.clone();
                let input_opening_values: Vec<F> = input_constraint
                    .required_openings
                    .iter()
                    .map(|id| self.opening_accumulator.get_opening(*id))
                    .collect();
                let initial_input =
                    FinalOutputWitness::general(input_challenge_values, input_opening_values);

                let config = if stage_idx == 0 {
                    StageConfig::new_uniskip(poly_degree, power_sums)
                        .with_input_constraint(input_constraint)
                } else {
                    StageConfig::new_uniskip_chain(poly_degree, power_sums)
                        .with_input_constraint(input_constraint)
                };
                stage_configs.push(config);
                stage_witnesses.push(StageWitness::with_initial_input(
                    vec![RoundWitness::with_claimed_sum(
                        coeffs.clone(),
                        challenge,
                        claimed_sum,
                    )],
                    initial_input,
                ));
            } else {
                initial_claims.push(zk_data.initial_claim);
            }

            if stage_idx < 2 {
                initial_claims.push(zk_data.initial_claim);
            }

            let mut current_claim = zk_data.initial_claim;
            let stage_challenges = &zk_data.challenges;
            let num_rounds = zk_data.poly_coeffs.len();

            for (round_idx, coeffs) in zk_data.poly_coeffs.iter().enumerate() {
                let challenge: F = stage_challenges[round_idx].into();
                let poly_degree = coeffs.len() - 1;
                let claimed_sum = current_claim;

                let mut next_claim = coeffs[coeffs.len() - 1];
                for i in (0..coeffs.len() - 1).rev() {
                    next_claim = coeffs[i] + challenge * next_claim;
                }

                let starts_new_chain = round_idx == 0;
                let is_last_round = round_idx == num_rounds - 1;
                let is_first_round = round_idx == 0;

                let config = if starts_new_chain {
                    StageConfig::new_chain(1, poly_degree)
                } else {
                    StageConfig::new(1, poly_degree)
                };

                let (config, initial_input_witness) = if is_first_round {
                    let batched_constraint = InputClaimConstraint::batch_required(
                        &zk_data.input_constraints,
                        zk_data.batching_coefficients.len(),
                    );

                    let mut challenge_values: Vec<F> = zk_data
                        .batching_coefficients
                        .iter()
                        .zip(&zk_data.input_claim_scaling_exponents)
                        .map(|(alpha, &scale)| alpha.mul_pow_2(scale))
                        .collect();
                    for cv in &zk_data.input_constraint_challenge_values {
                        challenge_values.extend(cv.iter().cloned());
                    }

                    let opening_values: Vec<F> = batched_constraint
                        .required_openings
                        .iter()
                        .map(|id| self.opening_accumulator.get_opening(*id))
                        .collect();

                    let initial_input =
                        FinalOutputWitness::general(challenge_values, opening_values);
                    let config_with_input = config.with_input_constraint(batched_constraint);
                    (config_with_input, Some(initial_input))
                } else {
                    (config, None)
                };

                let (config, final_output_witness) = if is_last_round {
                    let batched = OutputClaimConstraint::batch(
                        &zk_data.output_constraints,
                        zk_data.batching_coefficients.len(),
                    );

                    if let Some(batched_constraint) = batched {
                        let mut challenge_values: Vec<F> = zk_data.batching_coefficients.clone();
                        for cv in &zk_data.constraint_challenge_values {
                            challenge_values.extend(cv.iter().cloned());
                        }

                        let opening_values: Vec<F> = batched_constraint
                            .required_openings
                            .iter()
                            .map(|id| self.opening_accumulator.get_opening(*id))
                            .collect();

                        let final_output =
                            FinalOutputWitness::general(challenge_values, opening_values);
                        let config_with_fout = config.with_constraint(batched_constraint);
                        (config_with_fout, Some(final_output))
                    } else {
                        (config, None)
                    }
                } else {
                    (config, None)
                };

                stage_configs.push(config);
                let round_witness =
                    RoundWitness::with_claimed_sum(coeffs.clone(), challenge, claimed_sum);

                let stage_witness = match (initial_input_witness, final_output_witness) {
                    (Some(ii), Some(fout)) => {
                        StageWitness::with_both(vec![round_witness], ii, fout)
                    }
                    (Some(ii), None) => StageWitness::with_initial_input(vec![round_witness], ii),
                    (None, Some(fout)) => {
                        StageWitness::with_final_output(vec![round_witness], fout)
                    }
                    (None, None) => StageWitness::new(vec![round_witness]),
                };
                stage_witnesses.push(stage_witness);

                current_claim = next_claim;
            }
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
        let mut baked_output_challenges: Vec<F> = Vec::new();
        let mut baked_input_challenges: Vec<F> = Vec::new();

        for (stage_idx, zk_data) in zk_stages.iter().enumerate() {
            if stage_idx < 2 {
                let uniskip = &uniskip_stages[stage_idx];
                baked_input_challenges
                    .extend(uniskip.input_constraint_challenge_values.iter().cloned());
                baked_challenges.push(uniskip.challenge.into());
            }

            let num_rounds = zk_data.poly_coeffs.len();
            for round_idx in 0..num_rounds {
                if round_idx == 0 {
                    let mut cv: Vec<F> = zk_data
                        .batching_coefficients
                        .iter()
                        .zip(&zk_data.input_claim_scaling_exponents)
                        .map(|(alpha, &scale)| alpha.mul_pow_2(scale))
                        .collect();
                    for cv_inner in &zk_data.input_constraint_challenge_values {
                        cv.extend(cv_inner.iter().cloned());
                    }
                    baked_input_challenges.extend(cv);
                }

                baked_challenges.push(zk_data.challenges[round_idx].into());

                if round_idx == num_rounds - 1 {
                    let batched = OutputClaimConstraint::batch(
                        &zk_data.output_constraints,
                        zk_data.batching_coefficients.len(),
                    );
                    if batched.is_some() {
                        let mut cv: Vec<F> = zk_data.batching_coefficients.clone();
                        for cv_inner in &zk_data.constraint_challenge_values {
                            cv.extend(cv_inner.iter().cloned());
                        }
                        baked_output_challenges.extend(cv);
                    }
                }
            }
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
            VerifierR1CSBuilder::<F>::new_with_extra(&stage_configs, &extra_constraints, &baked);
        let r1cs = builder.build();

        let extra_opening_values: Vec<F> = stage8_data
            .opening_ids
            .iter()
            .map(|id| self.opening_accumulator.get_opening(*id))
            .collect();
        let extra_blinding = stage8_data.y_blinding;
        let extra_witness = ExtraConstraintWitness {
            output_value: stage8_data.joint_claim,
            blinding: extra_blinding,
            challenge_values: stage8_data.constraint_coeffs.clone(),
            opening_values: extra_opening_values,
        };

        let blindfold_witness = BlindFoldWitness::with_extra_constraints(
            initial_claims,
            stage_witnesses,
            vec![extra_witness],
        );

        let z = blindfold_witness.assign(&r1cs);

        #[cfg(test)]
        {
            if let Err(row) = r1cs.check_satisfaction(&z) {
                panic!(
                    "BlindFold R1CS not satisfied at constraint row {row} (total constraints: {}, total vars: {})",
                    r1cs.num_constraints, r1cs.num_vars
                );
            }
        }

        let witness: Vec<F> = z[1..].to_vec();

        let mut round_commitments: Vec<C::G1> = Vec::new();
        let mut round_blindings: Vec<F> = Vec::new();

        for (stage_idx, zk_data) in zk_stages.iter().enumerate() {
            if stage_idx < 2 {
                let uniskip = &uniskip_stages[stage_idx];
                let commitment =
                    C::G1::deserialize_compressed(&uniskip.commitment_bytes[..]).unwrap();
                round_commitments.push(commitment);
                round_blindings.push(uniskip.blinding_factor);
            }

            for (commitment_bytes, blinding) in zk_data
                .round_commitments
                .iter()
                .zip(&zk_data.blinding_factors)
            {
                let commitment = C::G1::deserialize_compressed(&commitment_bytes[..]).unwrap();
                round_commitments.push(commitment);
                round_blindings.push(*blinding);
            }
        }

        let pedersen_generator_count = pedersen_generator_count_for_r1cs(&r1cs);
        let pedersen_generators = PedersenGenerators::<C>::deterministic(pedersen_generator_count);
        let eval_commitments =
            vec![PCS::eval_commitment(joint_opening_proof).expect("missing eval commitment")];

        let hyrax = &r1cs.hyrax;
        let hyrax_C = hyrax.C;
        let R_coeff = hyrax.R_coeff;
        let R_prime = hyrax.R_prime;
        let noncoeff_rows = hyrax.noncoeff_rows();

        let noncoeff_rows_start = R_coeff * hyrax_C;
        let mut noncoeff_row_commitments = Vec::with_capacity(noncoeff_rows);
        let mut noncoeff_row_blindings = Vec::with_capacity(noncoeff_rows);
        for row_idx in 0..noncoeff_rows {
            let row_start = noncoeff_rows_start + row_idx * hyrax_C;
            let mut row_data = vec![F::zero(); hyrax_C];
            for k in 0..hyrax_C {
                if row_start + k < witness.len() {
                    row_data[k] = witness[row_start + k];
                }
            }
            let blinding = F::random(&mut rng);
            noncoeff_row_commitments.push(pedersen_generators.commit(&row_data, &blinding));
            noncoeff_row_blindings.push(blinding);
        }

        let mut w_row_blindings = Vec::with_capacity(R_prime);
        w_row_blindings.extend_from_slice(&round_blindings);
        w_row_blindings.resize(R_coeff, F::zero());
        w_row_blindings.extend_from_slice(&noncoeff_row_blindings);
        w_row_blindings.resize(R_prime, F::zero());

        let (real_instance, real_witness) = RelaxedR1CSInstance::<F, C>::new_non_relaxed(
            &witness,
            r1cs.num_constraints,
            hyrax_C,
            round_commitments,
            noncoeff_row_commitments,
            eval_commitments,
            w_row_blindings,
        );

        let eval_commitment_gens = PCS::eval_commitment_gens(&self.preprocessing.generators);
        let prover =
            BlindFoldProver::<_, _>::new(&pedersen_generators, &r1cs, eval_commitment_gens);
        let mut blindfold_transcript = ProofTranscript::new(b"BlindFold");

        prover.prove(&real_instance, &real_witness, &z, &mut blindfold_transcript)
    }
}
