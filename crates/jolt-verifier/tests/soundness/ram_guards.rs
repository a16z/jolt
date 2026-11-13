#![expect(clippy::unwrap_used, reason = "soundness fixtures fail loudly")]

use common::jolt_device::{MemoryConfig, MemoryLayout};
use jolt_claims::protocols::jolt::relations::ram::{
    RamReadWriteChallenges, RamReadWriteInputClaims, RamReadWriteOutputClaims,
};
use jolt_claims::protocols::jolt::ReadWriteDimensions;
use jolt_crypto::Bn254G1;
use jolt_field::{Fr, JoltField, Ring, Zero};
use jolt_poly::UnivariatePoly;
use jolt_program::preprocess::ram::write_protected_word_intervals;
use jolt_sumcheck::{ClearSumcheckRecorder, SumcheckProof, SumcheckRecorder};
use jolt_transcript::{prover_transcript, Keccak};
use jolt_verifier::stages::relations::{ConcreteSumcheck, SumcheckBatch};
use jolt_verifier::stages::stage2::ram_read_write_checking::RamReadWriteChecking;
use jolt_verifier::VerifierError;

#[derive(SumcheckBatch)]
struct RamGuardSumchecks<F: JoltField> {
    ram: RamReadWriteChecking<F>,
}

/// A two-cycle witness bypasses all tracer admission. Every produced opening
/// is its actual MLE.
fn verify_witness(
    code: bool,
    read: [i64; 2],
    increments: [i64; 2],
    selected: [i64; 2],
    memory_rows: [i64; 2],
) -> Result<(), VerifierError> {
    verify_address_witness(
        RamGuardDomain {
            intervals: if code { vec![(0, 1)] } else { Vec::new() },
            log_k: 0,
            addresses: vec![0],
        },
        read,
        increments,
        selected,
        memory_rows,
    )
}

struct RamGuardDomain {
    intervals: Vec<(usize, usize)>,
    log_k: usize,
    /// The words a selecting cycle selects, all at once.
    addresses: Vec<usize>,
}

fn verify_address_witness(
    domain: RamGuardDomain,
    read: [i64; 2],
    increments: [i64; 2],
    selected: [i64; 2],
    memory_rows: [i64; 2],
) -> Result<(), VerifierError> {
    let RamGuardDomain {
        intervals,
        log_k,
        addresses,
    } = domain;
    let tau = Fr::from_u64(7);
    let lerp = |values: [i64; 2], point: Fr| {
        let a = Fr::from_i64(values[0]);
        a + point * (Fr::from_i64(values[1]) - a)
    };
    let sumchecks = RamGuardSumchecks {
        ram: RamReadWriteChecking::new(
            ReadWriteDimensions::new(1, log_k, 1, log_k),
            log_k,
            vec![tau],
            intervals,
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
    let outputs_at = |point: &[Fr]| {
        let cycle = point[log_k];
        let address_eq: Fr = addresses
            .iter()
            .map(|address| {
                point[..log_k]
                    .iter()
                    .enumerate()
                    .map(|(bit, r)| {
                        if address & (1 << (log_k - bit - 1)) == 0 {
                            Fr::from_u64(1) - *r
                        } else {
                            *r
                        }
                    })
                    .product::<Fr>()
            })
            .sum();
        RamReadWriteOutputClaims {
            val: lerp(read, cycle),
            ra: address_eq * lerp(selected, cycle),
            inc: lerp(increments, cycle),
        }
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
    let mut bindings = Vec::new();
    for round in 0..batch.max_num_vars {
        let samples: Vec<_> =
            (0..4)
                .map(|sample| {
                    (0..1usize << (batch.max_num_vars - round - 1))
                        .map(|vertex| {
                            let mut point = bindings.clone();
                            point.push(Fr::from_u64(sample));
                            point.extend((round + 1..batch.max_num_vars).map(|bit| {
                                Fr::from_u64(((vertex >> (bit - round - 1)) & 1) as u64)
                            }));
                            let output_points = sumchecks
                                .ram
                                .derive_opening_points(&point, &input_points.ram)
                                .unwrap();
                            sumchecks
                                .ram
                                .expected_output(
                                    &input_points.ram,
                                    &outputs_at(&output_points.val),
                                    &output_points,
                                    &challenges.ram,
                                )
                                .unwrap()
                                * coefficients.ram
                        })
                        .sum()
                })
                .collect();
        let poly = UnivariatePoly::from_evals(&samples);
        bindings.push(
            recorder
                .absorb_round(&poly, batch.max_degree, &mut probe)
                .unwrap(),
        );
    }
    let proof = SumcheckProof::<Fr, Bn254G1>::from(recorder.finish(&[], &mut probe).unwrap().proof);
    let output_points = sumchecks
        .ram
        .derive_opening_points(&bindings, &input_points.ram)
        .unwrap();
    let output = RamGuardOutputClaims {
        ram: outputs_at(&output_points.val),
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
    assert!(matches!(
        verify_witness(false, [0, 0], [0, 0], [1, 0], [0, 0]),
        Err(VerifierError::StageClaimOutputMismatch { stage: 2 }),
    ));
}

#[test]
fn verifier_rejects_two_selected_words_on_a_memory_row() {
    let domain = |addresses| RamGuardDomain {
        intervals: Vec::new(),
        log_k: 1,
        addresses,
    };
    assert!(verify_address_witness(domain(vec![1]), [0, 0], [0, 0], [1, 1], [1, 1]).is_ok());
    assert!(matches!(
        verify_address_witness(domain(vec![0, 1]), [0, 0], [0, 0], [1, 1], [1, 1]),
        Err(VerifierError::StageClaimOutputMismatch { stage: 2 }),
    ));
}

#[test]
fn verifier_rejects_protected_io_and_canary_changes_with_or_without_restoration() {
    let layout = MemoryLayout::new(&MemoryConfig {
        program_size: Some(8),
        max_input_size: 8,
        max_trusted_advice_size: 8,
        max_untrusted_advice_size: 8,
        max_output_size: 8,
        stack_size: 0,
        heap_size: 0,
    });
    let ram_k = layout.max_ram_k().unwrap();
    let intervals = write_protected_word_intervals(&[], &layout, ram_k).unwrap();
    let log_k = ram_k.trailing_zeros() as usize;
    for address in [
        layout.input_start,
        layout.trusted_advice_start,
        layout.untrusted_advice_start,
        layout.stack_end,
    ] {
        let word = layout.remapped_word_address(address).unwrap() as usize;
        for (read, increments) in [([41, 50], [9, 0]), ([41, 50], [9, -9])] {
            assert!(verify_address_witness(
                RamGuardDomain {
                    intervals: Vec::new(),
                    log_k,
                    addresses: vec![word]
                },
                read,
                increments,
                [1, 1],
                [1, 1],
            )
            .is_ok());
            assert!(matches!(
                verify_address_witness(
                    RamGuardDomain {
                        intervals: intervals.clone(),
                        log_k,
                        addresses: vec![word]
                    },
                    read,
                    increments,
                    [1, 1],
                    [1, 1],
                ),
                Err(VerifierError::StageClaimOutputMismatch { stage: 2 }),
            ));
        }
        assert!(verify_address_witness(
            RamGuardDomain {
                intervals: intervals.clone(),
                log_k,
                addresses: vec![word]
            },
            [41, 41],
            [0, 0],
            [1, 1],
            [1, 1],
        )
        .is_ok());
    }
}
