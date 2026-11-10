#![expect(clippy::unwrap_used, reason = "soundness fixtures fail loudly")]

use jolt_claims::protocols::jolt::relations::ram::{
    RamReadWriteChallenges, RamReadWriteInputClaims, RamReadWriteOutputClaims,
};
use jolt_claims::protocols::jolt::ReadWriteDimensions;
use jolt_crypto::Bn254G1;
use jolt_field::{Fr, JoltField, Ring, Zero};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ClearSumcheckRecorder, CompressedSumcheckProof, SumcheckProof};
use jolt_transcript::{prover_transcript, Keccak};
use jolt_verifier::stages::relations::{ConcreteSumcheck, SumcheckBatch};
use jolt_verifier::stages::stage2::ram_read_write_checking::RamReadWriteChecking;
use jolt_verifier::VerifierError;

#[derive(SumcheckBatch)]
struct RamGuardSumchecks<F: JoltField> {
    ram: RamReadWriteChecking<F>,
}

/// A two-cycle witness bypasses all tracer admission. The first write changes
/// a word; the second restores it. Every produced opening is its actual MLE.
fn verify_witness(
    code: bool,
    read: [i64; 2],
    increments: [i64; 2],
    selected: [i64; 2],
    memory_rows: [i64; 2],
) -> Result<(), VerifierError> {
    let tau = Fr::from_u64(7);
    let lerp = |values: [i64; 2], point: Fr| {
        let a = Fr::from_i64(values[0]);
        a + point * (Fr::from_i64(values[1]) - a)
    };
    let sumchecks = RamGuardSumchecks {
        ram: RamReadWriteChecking::new(
            ReadWriteDimensions::new(1, 0, 1, 0),
            0,
            vec![tau],
            if code { vec![(0, 1)] } else { Vec::new() },
        ),
    };
    let input_points = RamGuardInputPoints {
        ram: RamReadWriteInputClaims::default(),
    };
    let input = RamGuardInputClaims {
        ram: RamReadWriteInputClaims {
            ram_read_value: lerp(read, tau),
            ram_write_value: lerp([read[0] + increments[0], read[1] + increments[1]], tau),
            load: Fr::zero(),
            store: lerp(memory_rows, tau),
        },
    };
    let challenges = RamGuardChallenges {
        ram: RamReadWriteChallenges {
            gamma: Fr::from_u64(11),
        },
    };
    let outputs_at = |point| RamReadWriteOutputClaims {
        val: lerp(read, point),
        ra: lerp(selected, point),
        inc: lerp(increments, point),
    };
    let transcript = || {
        prover_transcript::<Keccak>(
            &jolt_transcript::protocol_tag(b"ram-guard-malicious-witness"),
            b"",
        )
    };
    let mut probe = transcript();
    let mut recorder = ClearSumcheckRecorder::new();
    let (batch, coefficients) =
        sumchecks.begin_batch(&input, &challenges, &mut recorder, &mut probe)?;
    let samples: Vec<_> = (0..4)
        .map(|sample| {
            let point = Fr::from_u64(sample);
            let output_points = sumchecks
                .ram
                .derive_opening_points(&[point], &input_points.ram)
                .unwrap();
            sumchecks
                .ram
                .expected_output(
                    &input_points.ram,
                    &outputs_at(point),
                    &output_points,
                    &challenges.ram,
                )
                .unwrap()
                * coefficients.ram
        })
        .collect();
    let proof = SumcheckProof::<Fr, Bn254G1>::from(CompressedSumcheckProof {
        round_polynomials: vec![UnivariatePoly::from_evals(&samples).compress()],
        pow_nonces: Vec::new(),
    });
    let reduction = proof
        .verify_compressed_boolean(
            batch.max_num_vars,
            batch.max_degree,
            batch.claimed_sum,
            &mut probe,
        )
        .unwrap();
    let output = RamGuardOutputClaims {
        ram: outputs_at(reduction.point[0]),
    };
    sumchecks
        .verify_clear(
            &input,
            &input_points,
            &challenges,
            &output,
            &proof,
            &mut transcript(),
            2,
        )
        .map(|_| ())
}

#[test]
fn verifier_rejects_code_change_followed_by_restoration() {
    assert!(verify_witness(false, [41, 50], [9, -9], [1, 1], [1, 1]).is_ok());
    assert!(matches!(
        verify_witness(true, [41, 50], [9, -9], [1, 1], [1, 1]),
        Err(VerifierError::StageClaimOutputMismatch { stage: 2 }),
    ));
    assert!(verify_witness(true, [41, 41], [0, 0], [1, 1], [1, 1]).is_ok());
}

#[test]
fn verifier_requires_selection_only_on_memory_rows() {
    assert!(verify_witness(false, [0, 0], [0, 0], [0, 0], [0, 0]).is_ok());
    assert!(matches!(
        verify_witness(false, [0, 0], [0, 0], [0, 0], [1, 0]),
        Err(VerifierError::StageClaimOutputMismatch { stage: 2 }),
    ));
}
