use jolt_field::AkitaField;

use super::*;
use crate::metal::solinas::Fp128;

const MAX_BUFFER_LENGTH: u64 = u64::MAX;

fn akita_mul_u64(coefficient: Fp128, scalar: u64) -> Fp128 {
    let coefficient = coefficient.into_jolt_field::<AkitaField>();
    Fp128::from_jolt_field(&(coefficient * AkitaField::from_u64(scalar)))
}

fn akita_mul_signed_u64(coefficient: Fp128, magnitude: u64, negative: bool) -> Fp128 {
    let coefficient = coefficient.into_jolt_field::<AkitaField>();
    let magnitude = AkitaField::from_u64(magnitude);
    let factor = if negative { -magnitude } else { magnitude };
    Fp128::from_jolt_field(&(coefficient * factor))
}

#[test]
fn abi_is_stable_and_probe_names_are_unique() {
    assert_eq!(core::mem::size_of::<HalfWidthOperand>(), 16);
    assert_eq!(core::mem::align_of::<HalfWidthOperand>(), 16);
    assert_eq!(core::mem::size_of::<HalfWidthProbeParams>(), 8);
    assert_eq!(core::mem::align_of::<HalfWidthProbeParams>(), 4);

    for (index, probe) in HalfWidthProbe::ALL.iter().enumerate() {
        assert!(HalfWidthProbe::ALL[..index]
            .iter()
            .all(|previous| previous.name() != probe.name()));
        assert!(SOURCE.contains(probe.name()));
    }
}

#[test]
fn operand_domains_validate_without_narrowing() {
    let unsigned = HalfWidthOperand::unsigned(u64::MAX);
    assert_eq!(unsigned.words(), [u64::MAX, 0]);
    assert_eq!(unsigned.validate(HalfWidthDomain::Unsigned), Ok(()));

    let positive_zero = HalfWidthOperand::signed_magnitude(0, true);
    assert_eq!(positive_zero.words(), [0, 0]);
    assert_eq!(
        positive_zero.validate(HalfWidthDomain::SignedMagnitude),
        Ok(())
    );
    assert_eq!(
        HalfWidthOperand::from_words([0, 1]).validate(HalfWidthDomain::SignedMagnitude),
        Err(HalfWidthOperandError::NegativeZero)
    );
    assert_eq!(
        HalfWidthOperand::from_words([7, 2]).validate(HalfWidthDomain::SignedMagnitude),
        Err(HalfWidthOperandError::InvalidSignWord(2))
    );
    assert_eq!(
        HalfWidthOperand::from_words([7, 1]).validate(HalfWidthDomain::Unsigned),
        Err(HalfWidthOperandError::NonzeroUnsignedPadding(1))
    );

    assert_eq!(HalfWidthOperand::delta(0, u64::MAX).words(), [0, u64::MAX]);
}

#[test]
fn checked_shapes_separate_allocation_from_semantic_traffic() {
    let coefficients = vec![Fp128::ONE; 8];
    let unsigned = vec![HalfWidthOperand::unsigned(3); 8];
    let shape = checked_probe_shape(
        HalfWidthProbe::ChainU64Ilp4,
        &coefficients,
        &unsigned,
        7,
        HALF_WIDTH_AKITA_OFFSET,
        MAX_BUFFER_LENGTH,
    )
    .unwrap();
    assert_eq!(shape.params().elements, 8);
    assert_eq!(shape.params().iterations, 7);
    assert_eq!(shape.grid_threads(), 2);
    assert_eq!(shape.field_buffer_bytes(), 128);
    assert_eq!(shape.operand_buffer_bytes(), 128);
    assert_eq!(shape.allocated_bytes(), 392);
    assert_eq!(shape.semantic_bytes(), 320);
    assert_eq!(shape.operation_count(), 56);

    let signed = vec![HalfWidthOperand::signed_magnitude(3, true); 8];
    let signed_shape = checked_probe_shape(
        HalfWidthProbe::ChainSignedU64Ilp4,
        &coefficients,
        &signed,
        7,
        HALF_WIDTH_AKITA_OFFSET,
        MAX_BUFFER_LENGTH,
    )
    .unwrap();
    assert_eq!(signed_shape.allocated_bytes(), 392);
    assert_eq!(signed_shape.semantic_bytes(), 328);

    let delta = vec![HalfWidthOperand::delta(3, 5); 8];
    let delta_shape = checked_probe_shape(
        HalfWidthProbe::ChainU64DeltaIlp4,
        &coefficients,
        &delta,
        7,
        HALF_WIDTH_AKITA_OFFSET,
        MAX_BUFFER_LENGTH,
    )
    .unwrap();
    assert_eq!(delta_shape.allocated_bytes(), 392);
    assert_eq!(delta_shape.semantic_bytes(), 384);
}

#[test]
fn checked_shapes_reject_invalid_inputs() {
    let coefficient = [Fp128::ONE];
    assert_eq!(
        checked_probe_shape(
            HalfWidthProbe::MulU64,
            &coefficient,
            &[HalfWidthOperand::from_words([1, 9])],
            1,
            HALF_WIDTH_AKITA_OFFSET,
            MAX_BUFFER_LENGTH,
        ),
        Err(HalfWidthProbeError::NonzeroUnsignedOperand { index: 0, value: 9 })
    );
    assert_eq!(
        checked_probe_shape(
            HalfWidthProbe::MulSignedU64,
            &coefficient,
            &[HalfWidthOperand::from_words([1, 9])],
            1,
            HALF_WIDTH_AKITA_OFFSET,
            MAX_BUFFER_LENGTH,
        ),
        Err(HalfWidthProbeError::InvalidSignedOperand { index: 0, value: 9 })
    );
    assert_eq!(
        checked_probe_shape(
            HalfWidthProbe::MulSignedU64,
            &coefficient,
            &[HalfWidthOperand::from_words([0, 1])],
            1,
            HALF_WIDTH_AKITA_OFFSET,
            MAX_BUFFER_LENGTH,
        ),
        Err(HalfWidthProbeError::NegativeZeroOperand { index: 0 })
    );
    assert_eq!(
        checked_probe_shape(
            HalfWidthProbe::MulU64,
            &[Fp128::from_u128(HALF_WIDTH_AKITA_MODULUS)],
            &[HalfWidthOperand::unsigned(1)],
            1,
            HALF_WIDTH_AKITA_OFFSET,
            MAX_BUFFER_LENGTH,
        ),
        Err(HalfWidthProbeError::NonCanonicalCoefficient {
            index: 0,
            offset: HALF_WIDTH_AKITA_OFFSET,
        })
    );
    assert_eq!(
        checked_probe_shape(
            HalfWidthProbe::ChainU64Ilp8,
            &[Fp128::ONE; 7],
            &[HalfWidthOperand::unsigned(1); 7],
            1,
            HALF_WIDTH_AKITA_OFFSET,
            MAX_BUFFER_LENGTH,
        ),
        Err(HalfWidthProbeError::MisalignedElementCount {
            probe: HalfWidthProbe::ChainU64Ilp8.name(),
            ilp: 8,
        })
    );
}

#[test]
fn akita_constants_and_reduction_budget_are_exact() {
    assert_eq!(HALF_WIDTH_AKITA_OFFSET, 0xffff_a7f7);
    assert_eq!(
        HALF_WIDTH_AKITA_MODULUS,
        u128::MAX - u128::from(HALF_WIDTH_AKITA_OFFSET) + 1
    );
    assert_eq!(AKITA_REDUCTION_BOUNDS.high_bits, 64);
    assert_eq!(AKITA_REDUCTION_BOUNDS.high_times_offset_bits, 96);
    assert_eq!(AKITA_REDUCTION_BOUNDS.first_fold_carry_max, 1);
    assert_eq!(AKITA_REDUCTION_BOUNDS.second_fold_carry_max, 0);
    assert_eq!(AKITA_REDUCTION_BOUNDS.canonical_subtractions, 1);

    assert_eq!(HalfWidthInstructionBudget::half_width().total_products, 11);
    assert_eq!(HalfWidthInstructionBudget::full_width().total_products, 21);
}

#[test]
fn reduction_exercises_both_nontrivial_paths() {
    let first_carry_coefficient =
        Fp128::from_u128(340_282_366_920_938_463_444_927_863_358_058_659_838);
    let (first_carry_result, first_carry_trace) =
        reduce_u192_akita(product_u64_limbs(first_carry_coefficient, u64::MAX));
    assert_eq!(first_carry_trace.first_fold_carry, 1);
    assert_eq!(first_carry_trace.carry_fold_carry, 0);
    assert!(!first_carry_trace.canonical_subtracted);
    assert_eq!(
        first_carry_result,
        akita_mul_u64(first_carry_coefficient, u64::MAX)
    );

    let canonical_correction_coefficient = Fp128::from_u128((1u128 << 64) + 1);
    let (corrected_result, corrected_trace) = reduce_u192_akita(product_u64_limbs(
        canonical_correction_coefficient,
        u64::MAX,
    ));
    assert_eq!(corrected_trace.first_fold_carry, 0);
    assert_eq!(corrected_trace.carry_fold_carry, 0);
    assert!(corrected_trace.canonical_subtracted);
    assert_eq!(
        corrected_result,
        akita_mul_u64(canonical_correction_coefficient, u64::MAX)
    );
}

#[test]
fn limb_oracle_matches_akita_on_boundaries_and_random_inputs() {
    let coefficients = [
        0,
        1,
        u128::from(u32::MAX),
        u128::from(u32::MAX) + 1,
        u128::from(u64::MAX),
        u128::from(u64::MAX) + 1,
        1u128 << 127,
        HALF_WIDTH_AKITA_MODULUS - 1,
    ];
    let scalars = [
        0,
        1,
        u64::from(u32::MAX),
        u64::from(u32::MAX) + 1,
        1u64 << 63,
        u64::MAX - 1,
        u64::MAX,
    ];
    for coefficient in coefficients.map(Fp128::from_u128) {
        for scalar in scalars {
            assert_eq!(
                mul_u64_oracle(coefficient, scalar),
                akita_mul_u64(coefficient, scalar)
            );
            for negative in [false, true] {
                assert_eq!(
                    mul_signed_u64_oracle(coefficient, scalar, negative),
                    akita_mul_signed_u64(coefficient, scalar, negative)
                );
            }
        }
    }

    let mut state = 0x6a09_e667_f3bc_c909u64;
    for _ in 0..1_024 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let high = state;
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let low = state;
        let coefficient = Fp128::from_u128(
            ((u128::from(high) << 64) | u128::from(low)) % HALF_WIDTH_AKITA_MODULUS,
        );
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        let scalar = state;
        assert_eq!(
            mul_u64_oracle(coefficient, scalar),
            akita_mul_u64(coefficient, scalar)
        );
    }
}

#[test]
fn signed_and_endpoint_delta_domains_cover_full_u64_magnitude() {
    let coefficient = Fp128::from_u128(HALF_WIDTH_AKITA_MODULUS - 1);
    let endpoints = [
        (0, 0),
        (u64::MAX, u64::MAX),
        (u64::MAX, 0),
        (0, u64::MAX),
        (u64::MAX, 1),
        (1, u64::MAX),
        ((1u64 << 32) - 1, 1u64 << 32),
        ((1u64 << 63) - 1, 1u64 << 63),
    ];
    for (minuend, subtrahend) in endpoints {
        let expected = AkitaField::from_u64(minuend) - AkitaField::from_u64(subtrahend);
        let expected = coefficient.into_jolt_field::<AkitaField>() * expected;
        assert_eq!(
            mul_u64_delta_oracle(coefficient, minuend, subtrahend),
            Fp128::from_jolt_field(&expected)
        );
    }
}

#[test]
fn reference_outputs_cover_all_domains_and_chain_iterations() {
    let coefficients = vec![Fp128::from_u128(3); 8];
    for probe in HalfWidthProbe::ALL {
        let operands = (0..8)
            .map(|index| match probe.domain() {
                HalfWidthDomain::Unsigned => HalfWidthOperand::unsigned(u64::MAX - index as u64),
                HalfWidthDomain::SignedMagnitude => {
                    HalfWidthOperand::signed_magnitude(u64::MAX - index as u64, index % 2 != 0)
                }
                HalfWidthDomain::UnsignedDelta if index % 2 == 0 => {
                    HalfWidthOperand::delta(u64::MAX - index as u64, index as u64)
                }
                HalfWidthDomain::UnsignedDelta => {
                    HalfWidthOperand::delta(index as u64, u64::MAX - index as u64)
                }
            })
            .collect::<Vec<_>>();
        let iteration_counts: &[u32] = if probe.is_chain() { &[1, 3, 7] } else { &[1] };
        for &iterations in iteration_counts {
            let expected = reference_outputs(probe, &coefficients, &operands, iterations).unwrap();
            let oracle = coefficients
                .iter()
                .copied()
                .zip(operands.iter().copied())
                .map(|(mut accumulator, operand)| {
                    for _ in 0..iterations {
                        accumulator = match probe.domain() {
                            HalfWidthDomain::Unsigned => {
                                mul_u64_oracle(accumulator, operand.primary)
                            }
                            HalfWidthDomain::SignedMagnitude => mul_signed_u64_oracle(
                                accumulator,
                                operand.primary,
                                operand.secondary != 0,
                            ),
                            HalfWidthDomain::UnsignedDelta => mul_u64_delta_oracle(
                                accumulator,
                                operand.primary,
                                operand.secondary,
                            ),
                        };
                    }
                    accumulator
                })
                .collect::<Vec<_>>();
            assert_eq!(
                oracle,
                expected,
                "probe={} iterations={iterations}",
                probe.name()
            );
        }
    }
}

#[test]
fn occupancy_model_is_a_floor_not_a_residency_claim() {
    assert_eq!(
        HalfWidthRegisterFloor::for_probe(HalfWidthProbe::ChainU64Ilp8).minimum_live_words,
        60
    );
    assert_eq!(
        HalfWidthRegisterFloor::for_probe(HalfWidthProbe::ChainSignedU64Ilp8).minimum_live_words,
        68
    );
    assert_eq!(
        HalfWidthRegisterFloor::for_probe(HalfWidthProbe::ChainU64DeltaIlp8).minimum_live_words,
        68
    );
}

#[test]
fn promotion_gate_fails_closed_and_enforces_both_speed_floors() {
    assert_eq!(
        gate_status(HalfWidthGateEvidence::default()),
        HalfWidthGateStatus::ParityMissing
    );
    let passing = HalfWidthGateEvidence {
        parity_passed: true,
        candidate_compiler_shape_passed: Some(true),
        control_compiler_shape_passed: Some(true),
        candidate_spills_detected: Some(false),
        control_spills_detected: Some(false),
        candidate_resident_threadgroups: Some(MINIMUM_RESIDENT_THREADGROUPS),
        control_resident_threadgroups: Some(MINIMUM_RESIDENT_THREADGROUPS),
        full_width_products_per_second: Some(FULL_WIDTH_CONTROL_PRODUCTS_PER_SECOND),
        half_width_products_per_second: Some(MINIMUM_HALF_WIDTH_PRODUCTS_PER_SECOND),
        relative_mad_bps: Some(MAXIMUM_RELATIVE_MAD_BPS),
    };
    assert_eq!(gate_status(passing), HalfWidthGateStatus::Pass);
    assert_eq!(
        gate_status(HalfWidthGateEvidence {
            half_width_products_per_second: Some(MINIMUM_HALF_WIDTH_PRODUCTS_PER_SECOND - 1),
            ..passing
        }),
        HalfWidthGateStatus::AbsoluteThroughputFailed
    );
    assert_eq!(
        gate_status(HalfWidthGateEvidence {
            full_width_products_per_second: Some(MINIMUM_HALF_WIDTH_PRODUCTS_PER_SECOND),
            ..passing
        }),
        HalfWidthGateStatus::RelativeThroughputFailed
    );
    assert_eq!(
        gate_status(HalfWidthGateEvidence {
            candidate_spills_detected: Some(true),
            ..passing
        }),
        HalfWidthGateStatus::CandidateSpillDetected
    );
    assert_eq!(
        gate_status(HalfWidthGateEvidence {
            control_spills_detected: Some(true),
            ..passing
        }),
        HalfWidthGateStatus::ControlSpillDetected
    );
    assert_eq!(
        maximum_active_ns(
            TARGET_CHAIN_OPERATIONS,
            MINIMUM_HALF_WIDTH_PRODUCTS_PER_SECOND
        ),
        TARGET_MAX_GPU_ACTIVE_NS
    );
}

#[test]
fn candidate_policies_preserve_required_fallbacks() {
    assert_eq!(
        candidate_policy(HalfWidthCandidate::SpartanShiftNativeU64),
        HalfWidthCandidatePolicy::MayUseAfterPromotion
    );
    assert_eq!(
        candidate_policy(HalfWidthCandidate::BytecodeRawIncrementFirstMessage),
        HalfWidthCandidatePolicy::HybridOnly
    );
    assert_eq!(
        candidate_policy(HalfWidthCandidate::RegistersClaimUnreducedAccumulator),
        HalfWidthCandidatePolicy::RetainDeferredAccumulator
    );
    assert_eq!(
        candidate_policy(HalfWidthCandidate::BoundMultilinearState),
        HalfWidthCandidatePolicy::FullWidthRequired
    );
}
