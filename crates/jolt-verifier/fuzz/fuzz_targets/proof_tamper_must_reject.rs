#![no_main]

//! Structured mutation of accepted transparent proofs.
//!
//! Each input selects an accepted fixture and exactly one semantic mutation.
//! The mutation families cover preamble inputs, commitments, every sumcheck
//! stage, clear claims, advice, and the final Dory opening.

use std::sync::OnceLock;

use common::jolt_device::JoltDevice;
use jolt_crypto::{Bn254G1, Pedersen};
use jolt_dory::{DoryCommitment, DoryScheme};
use jolt_field::{Fr, FromPrimitiveInt};
use jolt_poly::{CompressedPoly, UnivariatePoly};
use jolt_sumcheck::{ClearProof, SumcheckProof};
use jolt_transcript::LegacyBlake2bTranscript;
use jolt_verifier::{
    verify, JoltProof, JoltProofClaims, JoltVerifierPreprocessing, ZkConfig,
};
use libfuzzer_sys::fuzz_target;

type Preprocessing = JoltVerifierPreprocessing<DoryScheme, Pedersen<Bn254G1>>;
type Proof = JoltProof<DoryScheme, Pedersen<Bn254G1>>;
type Bundle = (Preprocessing, JoltDevice, Proof, Option<DoryCommitment>);
type StageProof = SumcheckProof<Fr, Bn254G1>;

static FIXTURES: [&[u8]; 3] = [
    include_bytes!("../fixtures/muldiv-bundle.bin"),
    include_bytes!("../fixtures/advice-consumer-bundle.bin"),
    include_bytes!("../fixtures/committed-muldiv-bundle.bin"),
];

fn bundles() -> &'static [Bundle] {
    static BUNDLES: OnceLock<Vec<Bundle>> = OnceLock::new();
    BUNDLES.get_or_init(|| {
        FIXTURES
            .iter()
            .map(|bytes| {
                let (bundle, consumed): (Bundle, usize) =
                    bincode::serde::decode_from_slice(bytes, bincode::config::standard())
                        .expect("fixture decodes");
                assert_eq!(consumed, bytes.len(), "fixture has trailing bytes");
                let (preprocessing, public_io, proof, advice) = &bundle;
                verify::<Fr, DoryScheme, Pedersen<Bn254G1>, LegacyBlake2bTranscript>(
                    preprocessing,
                    public_io,
                    proof,
                    advice.as_ref(),
                )
                .expect("honest fixture proof must verify before tampering");
                bundle
            })
            .collect()
    })
}

fn stage_proof(proof: &mut Proof, index: usize) -> &mut StageProof {
    match index % 10 {
        0 => &mut proof.stages.stage1_uni_skip_first_round_proof,
        1 => &mut proof.stages.stage1_sumcheck_proof,
        2 => &mut proof.stages.stage2_uni_skip_first_round_proof,
        3 => &mut proof.stages.stage2_sumcheck_proof,
        4 => &mut proof.stages.stage3_sumcheck_proof,
        5 => &mut proof.stages.stage4_sumcheck_proof,
        6 => &mut proof.stages.stage5_sumcheck_proof,
        7 => &mut proof.stages.stage6a_sumcheck_proof,
        8 => &mut proof.stages.stage6b_sumcheck_proof,
        _ => &mut proof.stages.stage7_sumcheck_proof,
    }
}

fn mutate_clear_sumcheck(proof: &mut StageProof, operation: u8, index: usize) -> bool {
    let before = proof.clone();
    match proof {
        SumcheckProof::Clear(ClearProof::Full(clear)) => match operation % 3 {
            0 => {
                let _ = clear.round_polynomials.pop();
            }
            1 => clear
                .round_polynomials
                .push(UnivariatePoly::new(vec![Fr::from_u64(1)])),
            _ => {
                if clear.round_polynomials.is_empty() {
                    return false;
                }
                let round = index % clear.round_polynomials.len();
                clear.round_polynomials[round] =
                    UnivariatePoly::new(vec![Fr::from_u64(index as u64 + 17)]);
            }
        },
        SumcheckProof::Clear(ClearProof::Compressed(clear)) => match operation % 3 {
            0 => {
                let _ = clear.round_polynomials.pop();
            }
            1 => clear
                .round_polynomials
                .push(CompressedPoly::new(vec![Fr::from_u64(1)])),
            _ => {
                if clear.round_polynomials.is_empty() {
                    return false;
                }
                let round = index % clear.round_polynomials.len();
                clear.round_polynomials[round] =
                    CompressedPoly::new(vec![Fr::from_u64(index as u64 + 17)]);
            }
        },
        SumcheckProof::Committed(_) => return false,
    }
    *proof != before
}

fn mutate_clear_claim(proof: &mut Proof, stage: usize) -> bool {
    let JoltProofClaims::Clear(claims) = &mut proof.claims else {
        return false;
    };
    let delta = Fr::from_u64(1);
    match stage % 8 {
        0 => claims.stage1.uniskip_output_claim += delta,
        1 => claims.stage2.product_uniskip_output_claim += delta,
        2 => claims.stage3.shift.pc += delta,
        3 => claims.stage4.ram_val_check.ram_ra += delta,
        4 => claims.stage5.registers_val_evaluation.rd_inc += delta,
        5 => claims.stage6a.booleanity.intermediate += delta,
        6 => claims.stage6b.ram_hamming_booleanity.ram_hamming_weight += delta,
        _ => {
            let Some(value) = claims
                .stage7
                .hamming_weight_claim_reduction
                .instruction_ra
                .first_mut()
            else {
                return false;
            };
            *value += delta;
        }
    }
    true
}

fuzz_target!(|data: &[u8]| {
    if data.len() < 4 {
        return;
    }
    let fixture = &bundles()[data[0] as usize % bundles().len()];
    let mut preprocessing = fixture.0.clone();
    let mut public_io = fixture.1.clone();
    let mut proof = fixture.2.clone();
    let mut advice = fixture.3.clone();
    let mutation = data[1] % 16;
    let index = data[2] as usize;
    let operation = data[3];

    let changed = match mutation {
        0 => {
            proof.trace_length = 3;
            true
        }
        1 => {
            proof.ram_K = 3;
            true
        }
        2 => {
            proof.protocol.zk = ZkConfig::BlindFold;
            true
        }
        3 => {
            if public_io.inputs.is_empty() {
                return;
            }
            let position = index % public_io.inputs.len();
            public_io.inputs[position] ^= operation.max(1);
            true
        }
        4 => {
            public_io.memory_layout.heap_size =
                public_io.memory_layout.heap_size.saturating_add(1);
            true
        }
        5 => {
            if proof.commitments.rd_inc == proof.commitments.ram_inc {
                return;
            }
            std::mem::swap(
                &mut proof.commitments.rd_inc,
                &mut proof.commitments.ram_inc,
            );
            true
        }
        6 => proof.commitments.bytecode_ra.pop().is_some(),
        7 => {
            if proof.commitments.instruction_ra.is_empty() {
                return;
            }
            let position = index % proof.commitments.instruction_ra.len();
            let before = proof.commitments.instruction_ra[position].clone();
            proof.commitments.instruction_ra[position] = DoryCommitment::default();
            proof.commitments.instruction_ra[position] != before
        }
        8 => {
            proof.untrusted_advice_commitment =
                match proof.untrusted_advice_commitment.take() {
                    Some(_) => None,
                    None => Some(DoryCommitment::default()),
                };
            true
        }
        9 => {
            advice = match advice.take() {
                Some(_) => None,
                None => Some(DoryCommitment::default()),
            };
            true
        }
        10 => mutate_clear_sumcheck(stage_proof(&mut proof, index), operation, index),
        11 => mutate_clear_claim(&mut proof, index),
        12 => proof.joint_opening_proof.0.final_message.take().is_some(),
        13 => {
            preprocessing.program = bundles()[(data[0] as usize + 1) % bundles().len()]
                .0
                .program
                .clone();
            preprocessing.program != fixture.0.program
        }
        14 => {
            if proof.commitments.ram_ra.is_empty() {
                return;
            }
            let position = index % proof.commitments.ram_ra.len();
            let before = proof.commitments.ram_ra[position].clone();
            proof.commitments.ram_ra[position] = DoryCommitment::default();
            proof.commitments.ram_ra[position] != before
        }
        _ => {
            if proof.commitments.bytecode_ra.len() < 2
                || proof.commitments.bytecode_ra[0] == proof.commitments.bytecode_ra[1]
            {
                return;
            }
            proof.commitments.bytecode_ra.swap(0, 1);
            true
        }
    };
    if !changed {
        return;
    }

    let result = verify::<Fr, DoryScheme, Pedersen<Bn254G1>, LegacyBlake2bTranscript>(
        &preprocessing,
        &public_io,
        &proof,
        advice.as_ref(),
    );
    assert!(
        result.is_err(),
        "verifier accepted a tampered transparent proof (mutation class {mutation})"
    );
});
