use jolt_field::{AkitaField, Field};
use jolt_poly::UnivariatePoly;

use super::model::{
    admission_decision, work_plan, AdmissionDecision, CompiledEvidence, Geometry, ReductionOwner,
    Schedule, TimingEvidence, WeightSource,
};
use super::oracle::{
    chunk_partials_device_weights, chunk_partials_host_weights, direct_weight, fold_native_blocks,
    low_binding_weights, DenseOracle, OracleError, SuccessorTail,
};
use super::*;

fn field(value: u64) -> AkitaField {
    AkitaField::from_u64(value)
}

fn range(storage_id: u64, offset_bytes: u64, length_bytes: u64) -> BufferRange {
    BufferRange {
        storage_id,
        offset_bytes,
        length_bytes,
    }
}

fn target_ranges() -> RamOutputSuccessorRanges {
    RamOutputSuccessorRanges {
        source: range(7, 0, 65_536),
        coefficients: range(7, 65_536, 16_384),
        partials: range(7, 81_920, 1_024),
        output: range(7, 82_944, 128),
        status: range(7, 83_072, 4),
    }
}

#[test]
fn target_abi_accepts_only_the_frozen_geometry() {
    let params = RamOutputSuccessorParams::target(WeightMode::HostTable);
    assert_eq!(params.validate(), Ok(params));
    assert_eq!(params.coefficient_elements(), TARGET_WEIGHTS);
    let dispatch = DispatchShape {
        threadgroups: TARGET_PARTIALS,
        threads_per_threadgroup: TARGET_THREADS,
        initial_status: 0,
    };
    assert_eq!(
        target_ranges().validate_partials(params, dispatch),
        Ok(target_ranges())
    );

    let mut bad = params;
    bad.reserved[0] = 1;
    assert_eq!(bad.validate(), Err(AbiError::InvalidParams));

    let mut overlap = target_ranges();
    overlap.partials.offset_bytes = overlap.coefficients.offset_bytes;
    assert_eq!(
        overlap.validate_partials(params, dispatch),
        Err(AbiError::OverlappingRanges {
            left: "coefficients",
            right: "partials",
        })
    );
}

#[test]
fn resident_source_requires_public_io_and_device_provenance() {
    let metadata = ResidentRamFinalMetadata {
        range: target_ranges().source,
        device_registry_id: 19,
        allocation_identity: 23,
        elements: TARGET_ADDRESSES,
        stride_bytes: NATIVE_WORD_BYTES as u32,
        public_io_certified: true,
        host_readable: true,
    };
    assert_eq!(metadata.validate(19), Ok(metadata));
    assert_eq!(
        ResidentRamFinalMetadata {
            public_io_certified: false,
            ..metadata
        }
        .validate(19),
        Err(AbiError::InvalidResidentSource)
    );
    assert_eq!(metadata.validate(29), Err(AbiError::InvalidResidentSource));
}

#[test]
fn work_model_reproduces_the_registered_target_counts() {
    let selected = work_plan(Geometry::target(), Schedule::selected()).unwrap();
    assert_eq!(selected.host_full_products, 1_023);
    assert_eq!(selected.device_full_products, 0);
    assert_eq!(selected.device_half_width_products, 8_192);
    assert_eq!(selected.device_reduction_additions, 8_128);
    assert_eq!(selected.host_reduction_additions, 56);
    assert_eq!(selected.threadgroups, 64);
    assert_eq!(selected.simdgroups, 256);
    assert_eq!(selected.dispatches, 1);
    assert_eq!(selected.perfect_cache_bytes().unwrap(), 100_352);
    assert_eq!(selected.shader_requested_bytes().unwrap(), 215_040);
    assert_eq!(selected.arithmetic_floor_ns().unwrap(), 95);
    assert_eq!(selected.perfect_cache_traffic_floor_ns().unwrap(), 223);
    assert_eq!(selected.optimistic_device_floor_ns().unwrap(), 223);

    let device_weights = work_plan(
        Geometry::target(),
        Schedule {
            weights: WeightSource::DeviceChallenges,
            reduction: ReductionOwner::Device,
        },
    )
    .unwrap();
    assert_eq!(device_weights.host_full_products, 0);
    assert_eq!(device_weights.device_full_products, 73_728);
    assert_eq!(device_weights.device_half_width_products, 8_192);
    assert_eq!(device_weights.device_reduction_additions, 8_184);
    assert_eq!(device_weights.host_reduction_additions, 0);
    assert_eq!(device_weights.dispatches, 2);
    assert_eq!(device_weights.perfect_cache_bytes().unwrap(), 68_160);
    assert_eq!(device_weights.shader_requested_bytes().unwrap(), 1_378_720);
    assert_eq!(device_weights.arithmetic_floor_ns().unwrap(), 1_708);
    assert_eq!(
        device_weights.perfect_cache_traffic_floor_ns().unwrap(),
        151
    );
}

#[test]
fn admission_rejects_a_new_command_and_requires_eight_x_confirmation() {
    let work = work_plan(Geometry::target(), Schedule::selected()).unwrap();
    let compiled = CompiledEvidence {
        thread_execution_width: Some(SIMD_WIDTH),
        max_threads_per_threadgroup: Some(TARGET_THREADS),
        spills_detected: Some(false),
        resident_threadgroups_per_core: Some(2),
    };
    let passing = TimingEvidence {
        parity_passed: true,
        same_parent_command_control: true,
        counter_delimited_auxiliary: true,
        new_command_buffers: 0,
        new_waits: 0,
        empty_auxiliary_service_ns: Some(15_000),
        host_weights_ns: Some(6_000),
        partial_dispatch_ns: Some(18_000),
        host_tail_ns: Some(8_000),
        complete_incremental_ns: Some(32_000),
        resident_cpu_ns: Some(60_000),
        comparison_noise_ns: Some(1_000),
        five_alternating_pairs: true,
    };
    assert_eq!(
        admission_decision(work, compiled, passing),
        Ok(AdmissionDecision::EightXCandidate)
    );
    assert_eq!(
        admission_decision(
            work,
            compiled,
            TimingEvidence {
                new_command_buffers: 1,
                ..passing
            },
        ),
        Ok(AdmissionDecision::StandaloneTopologyRejected)
    );
    assert_eq!(
        admission_decision(
            work,
            compiled,
            TimingEvidence {
                complete_incremental_ns: Some(40_000),
                ..passing
            },
        ),
        Ok(AdmissionDecision::FiveXOnlyNeedsCeilingReview)
    );
    assert_eq!(
        admission_decision(
            work,
            compiled,
            TimingEvidence {
                resident_cpu_ns: Some(32_500),
                ..passing
            },
        ),
        Ok(AdmissionDecision::ResidentCpuWins)
    );
}

#[test]
fn direct_and_table_weights_produce_the_same_chunk_partials() {
    let challenges = (0..TARGET_CHALLENGES)
        .map(|index| field(7 + 19 * u64::from(index)))
        .collect::<Vec<_>>();
    let weights = low_binding_weights(&challenges);
    for (index, &weight) in weights.iter().enumerate() {
        assert_eq!(direct_weight(index, &challenges), weight);
    }
    let values = (0..TARGET_ADDRESSES)
        .map(|index| match index % 7 {
            0 => u64::MAX,
            1 => 0,
            _ => u64::from(index).wrapping_mul(0x9e37_79b9_7f4a_7c15),
        })
        .collect::<Vec<_>>();
    assert_eq!(
        chunk_partials_host_weights(&values, &weights),
        chunk_partials_device_weights(&values, &challenges)
    );
}

#[test]
fn certified_fold_and_tail_match_the_dense_relation() {
    let output_address = (0..13)
        .map(|index| field(31 + 11 * index))
        .collect::<Vec<_>>();
    let round_challenges = (0..13)
        .map(|index| field(101 + 17 * index))
        .collect::<Vec<_>>();
    let val_final = (0..TARGET_ADDRESSES)
        .map(|index| match index % 11 {
            0 => u64::MAX,
            1 => 0,
            _ => u64::from(index).wrapping_mul(0xbf58_476d_1ce4_e5b9),
        })
        .collect::<Vec<_>>();
    let mut val_io = vec![0_u64; TARGET_ADDRESSES as usize];
    val_io[(1 << 10)..(1 << 12)].copy_from_slice(&val_final[(1 << 10)..(1 << 12)]);

    let mut dense =
        DenseOracle::new(&output_address, 1 << 10, 1 << 12, &val_io, &val_final).unwrap();
    dense
        .defer_zero_prefix(&round_challenges[..TARGET_CHALLENGES as usize])
        .unwrap();
    let folded =
        fold_native_blocks(&val_final, &round_challenges[..TARGET_CHALLENGES as usize]).unwrap();
    assert_eq!(dense.val_final_table(), folded.as_slice());
    let mut tail = SuccessorTail::new(
        &output_address,
        &round_challenges[..TARGET_CHALLENGES as usize],
        folded,
        true,
    )
    .unwrap();

    let mut claim = AkitaField::zero();
    for &challenge in &round_challenges[TARGET_CHALLENGES as usize..] {
        let dense_evals = dense.checked_message(claim).unwrap();
        let tail_evals = tail.checked_message(claim).unwrap();
        assert_eq!(tail_evals, dense_evals);
        claim = UnivariatePoly::from_evals(&dense_evals).evaluate(challenge);
        dense.bind(challenge).unwrap();
        tail.bind(challenge).unwrap();
    }
    assert_eq!(tail.bound_values(), dense.bound_values());
}

#[test]
fn an_inside_mask_mutation_invalidates_zero_deferral() {
    let output_address = (0..13)
        .map(|index| field(13 + 7 * index))
        .collect::<Vec<_>>();
    let challenges = (0..10)
        .map(|index| field(71 + 5 * index))
        .collect::<Vec<_>>();
    let mut val_io = vec![0_u64; TARGET_ADDRESSES as usize];
    let mut val_final = vec![0_u64; TARGET_ADDRESSES as usize];
    val_io[(1 << 10)..(1 << 12)].fill(9);
    val_final[(1 << 10)..(1 << 12)].fill(9);
    val_final[(1 << 10) + 7] = 10;
    let mut dense =
        DenseOracle::new(&output_address, 1 << 10, 1 << 12, &val_io, &val_final).unwrap();
    assert!(matches!(
        dense.defer_zero_prefix(&challenges),
        Err(OracleError::NonZeroDeferredMessage { .. })
    ));
    assert!(matches!(
        SuccessorTail::new(&output_address, &challenges, [field(0); 8], false),
        Err(OracleError::UncertifiedPublicIo)
    ));
}
