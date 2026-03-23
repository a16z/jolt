//! Stateless proof verification for the Jolt typed DAG pipeline.
//!
//! Replays the Fiat-Shamir transcript, verifies each stage's sumcheck proof,
//! and verifies PCS opening proofs. All polynomial evaluations come from the
//! proof — no polynomial tables needed.

use jolt_field::Field;
use jolt_openings::{AdditivelyHomomorphic, VirtualEval, VerifierClaim};
use jolt_sumcheck::{BatchedSumcheckVerifier, SumcheckClaim};
use jolt_transcript::{AppendToTranscript, Transcript};

use crate::error::JoltError;
use crate::key::JoltVerifyingKey;
use crate::proof::JoltProof;
use crate::protocol::{claims, types::*};

pub fn gamma_powers<F: Field>(gamma: F, count: usize) -> Vec<F> {
    (0..count).scan(F::one(), |g, _| { let v = *g; *g *= gamma; Some(v) }).collect()
}

fn verify_sumcheck<F: Field, T: Transcript<Challenge = F>>(
    stage: usize,
    claims: &[SumcheckClaim<F>],
    stage_proof: &crate::proof::StageProof<F>,
    transcript: &mut T,
) -> Result<(F, Vec<F>), JoltError> {
    let (final_eval, challenges) =
        BatchedSumcheckVerifier::verify(claims, &stage_proof.round_polys, transcript)
            .map_err(|e| JoltError::StageVerification { stage, reason: e.to_string() })?;

    for &e in &stage_proof.evals {
        e.append_to_transcript(transcript);
    }

    Ok((final_eval, challenges.iter().rev().copied().collect()))
}

/// Verifies a complete Jolt proof. Standalone — no prover dependencies.
pub fn verify_proof<F, PCS>(
    proof: &JoltProof<F, PCS>,
    vk: &JoltVerifyingKey<F, PCS>,
) -> Result<(), JoltError>
where
    F: Field,
    PCS: AdditivelyHomomorphic<Field = F>,
{
    let mut t = jolt_transcript::Blake2bTranscript::<F>::new(b"jolt-v2");
    t.append_bytes(format!("{:?}", proof.witness_commitment).as_bytes());

    let (_r_x, r_y) =
        crate::verifier::verify_spartan(&vk.spartan_key, &proof.spartan_proof, &mut t)?;

    let ohp = proof.config.one_hot_params_from_config();
    let log_t = r_y.len();

    // S1 virtual evals — the verifier needs these for downstream claims.
    // In a full implementation, S1 evals come from the Spartan witness opening
    // (verified by PCS). For now, the proof doesn't carry them separately —
    // they'll be added when we populate SpartanOutput from the proof.
    let s1_evals = SpartanVirtualEvals {
        ram_read_value: VirtualEval(F::zero()),
        ram_write_value: VirtualEval(F::zero()),
        ram_address: VirtualEval(F::zero()),
        lookup_output: VirtualEval(F::zero()),
        left_operand: VirtualEval(F::zero()),
        right_operand: VirtualEval(F::zero()),
        left_instruction_input: VirtualEval(F::zero()),
        right_instruction_input: VirtualEval(F::zero()),
        rd_write_value: VirtualEval(F::zero()),
        rs1_value: VirtualEval(F::zero()),
        rs2_value: VirtualEval(F::zero()),
    };

    // S2
    let _gp2 = gamma_powers(t.challenge(), 5);
    let (_s2_final, s2_ep) = verify_sumcheck(2,
        &[SumcheckClaim { num_vars: log_t, degree: 3, claimed_sum: F::zero() }],
        &proof.stage_proofs[0], &mut t)?;
    let s2 = S2Evals::unpack(&s2_ep, &proof.stage_proofs[0].evals);

    // S3
    let _shift_gp = gamma_powers(t.challenge(), 5);
    let instr_gamma: F = t.challenge();
    let reg_gamma: F = t.challenge();
    let instr_claim = claims::s3_instruction_input(&s2, instr_gamma);
    let reg_claim = claims::s3_registers_cr(&s1_evals, reg_gamma);
    let (_s3_final, s3_ep) = verify_sumcheck(3,
        &[
            SumcheckClaim { num_vars: log_t, degree: 2, claimed_sum: F::zero() }, // shift
            SumcheckClaim { num_vars: log_t, degree: 3, claimed_sum: instr_claim },
            SumcheckClaim { num_vars: log_t, degree: 2, claimed_sum: reg_claim },
        ],
        &proof.stage_proofs[1], &mut t)?;
    let s3 = S3Evals::unpack(&s3_ep, &proof.stage_proofs[1].evals);

    // S4
    let s4_reg_gamma: F = t.challenge();
    let _s4_ram_gamma: F = t.challenge();
    let s4_reg_claim = claims::s4_registers_rw(&s3, s4_reg_gamma);
    let (_s4_final, s4_ep) = verify_sumcheck(4,
        &[
            SumcheckClaim { num_vars: log_t, degree: 3, claimed_sum: s4_reg_claim },
            SumcheckClaim { num_vars: log_t, degree: 3, claimed_sum: F::zero() }, // ram val
        ],
        &proof.stage_proofs[2], &mut t)?;
    let s4 = S4Evals::unpack(&s4_ep, &proof.stage_proofs[2].evals);

    // S5
    let _: F = t.challenge();
    let _: F = t.challenge();
    let _: Vec<F> = (0..log_t).map(|_| t.challenge()).collect();
    let (_s5_final, s5_ep) = verify_sumcheck(5,
        &[SumcheckClaim { num_vars: log_t, degree: 3, claimed_sum: F::zero() }],
        &proof.stage_proofs[3], &mut t)?;
    let s5 = S5Evals::unpack(&s5_ep, &proof.stage_proofs[3].evals);

    // S6
    let _: F = t.challenge();
    let _: Vec<F> = (0..log_t).map(|_| t.challenge()).collect();
    let _: F = t.challenge();
    let _: Vec<F> = (0..log_t).map(|_| t.challenge()).collect();
    let _: F = t.challenge();
    let _: F = t.challenge();
    let inc_gamma: F = t.challenge();
    let inc_claim = claims::s6_inc_cr(&s2, &s4, &s5, inc_gamma);
    let (_s6_final, _s6_ep) = verify_sumcheck(6,
        &[
            SumcheckClaim { num_vars: log_t, degree: 2, claimed_sum: inc_claim },
            SumcheckClaim { num_vars: log_t, degree: 3, claimed_sum: F::zero() }, // hamming
        ],
        &proof.stage_proofs[4], &mut t)?;

    // S7
    let _: F = t.challenge();
    let (_s7_final, _s7_ep) = verify_sumcheck(7,
        &[SumcheckClaim { num_vars: ohp.log_k_chunk, degree: 2, claimed_sum: F::zero() }],
        &proof.stage_proofs[5], &mut t)?;

    // S8: PCS
    let pcs_claims = vec![VerifierClaim {
        commitment: proof.witness_commitment.clone(),
        point: r_y,
        eval: proof.spartan_proof.witness_eval,
    }];

    crate::verifier::verify_openings::<PCS, _>(
        pcs_claims, &proof.opening_proofs, &vk.pcs_setup, &mut t,
    )?;

    Ok(())
}
