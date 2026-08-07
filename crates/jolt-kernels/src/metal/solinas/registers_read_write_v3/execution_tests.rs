use core::mem::{align_of, size_of};

use super::execution_abi::{
    Arena, BufferBinding, DenseRoundParams, DenseState, HistogramParams, HostSchedule,
    LifetimeDisposition, PipelineReadiness, RawCoefficientParams, RawReplayParams,
    RawRoundZeroParams, ReductionParams, SequencePoint, ARENA_LIFETIMES, DENSE_BINDINGS,
    DENSE_BIND_MESSAGE_PIPELINE, DENSE_DESCRIPTOR, HISTOGRAM_BINDINGS, HISTOGRAM_DESCRIPTOR,
    HISTOGRAM_PIPELINE, PHASES, PIPELINES, RAW_BASIS_GAMMA, RAW_BASIS_GAMMA_SQUARED, RAW_BASIS_ONE,
    RAW_COEFFICIENT_BINDINGS, RAW_COEFFICIENT_DESCRIPTOR, RAW_COEFFICIENT_PIPELINE,
    RAW_FLAG_MATERIALIZE_DENSE, RAW_REPLAY_BINDINGS, RAW_REPLAY_DESCRIPTOR, RAW_REPLAY_PIPELINE,
    RAW_ROUND_ZERO_DESCRIPTOR, RAW_ROUND_ZERO_PIPELINE, REDUCE_COLUMNS_PIPELINE,
    REDUCTION_BINDINGS, REDUCTION_DESCRIPTOR, REGISTERS_VAL_HANDOFF, ROUND_ZERO_BINDINGS, SOURCE,
};
use super::execution_model::{
    trace_peak_logical_bytes, Log26ExecutionModel, TraceExecutionPlan,
    ANALYTICAL_EXECUTION_HIGH_NS, ANALYTICAL_EXECUTION_LOW_NS, CPU_BASELINE_NS,
    CURRENT_CPU_FALLBACK_NS, FIXED_COEFFICIENT_ARENA_BYTES, FIXED_LOCAL_WEIGHT_ARENA_BYTES,
    LATEST_DIAGNOSTIC_CPU_NS, LOG26_DENSE_CACHE_BYTES, LOG26_DENSE_REQUESTED_BYTES,
    LOG26_EXECUTION_CACHE_BYTES, LOG26_EXECUTION_REQUESTED_BYTES, LOG26_HISTOGRAM_CACHE_BYTES,
    LOG26_HISTOGRAM_REQUESTED_BYTES, LOG26_LIFECYCLE_CACHE_BYTES, LOG26_LIFECYCLE_REQUESTED_BYTES,
    LOG26_PEAK_LOGICAL_BYTES, LOG26_PRODUCER_ALLOCATION_BYTES,
    LOG26_PRODUCER_INITIALIZED_WRITE_BYTES, LOG26_RAW_CACHE_BYTES, LOG26_RAW_REQUESTED_BYTES,
    LOG26_RAW_ROUND_PRODUCTS, LOG26_REGISTERS_VAL_BYTES, LOG26_TOPOLOGY_ALLOCATION_BYTES,
    LOG26_TOPOLOGY_INITIALIZED_BYTES, PRODUCER_PURSUIT_CAP_NS, TIME_BUDGETS,
};

#[test]
fn shader_abi_layouts_are_fixed() {
    assert_eq!(size_of::<DenseState>(), 48);
    assert_eq!(align_of::<DenseState>(), 16);
    assert_eq!(size_of::<RawRoundZeroParams>(), 32);
    assert_eq!(size_of::<RawCoefficientParams>(), 32);
    assert_eq!(size_of::<RawReplayParams>(), 48);
    assert_eq!(size_of::<DenseRoundParams>(), 32);
    assert_eq!(size_of::<ReductionParams>(), 16);
    assert_eq!(size_of::<HistogramParams>(), 32);

    for bindings in [
        ROUND_ZERO_BINDINGS.as_slice(),
        RAW_COEFFICIENT_BINDINGS.as_slice(),
        RAW_REPLAY_BINDINGS.as_slice(),
        DENSE_BINDINGS.as_slice(),
        REDUCTION_BINDINGS.as_slice(),
        HISTOGRAM_BINDINGS.as_slice(),
    ] {
        assert_contiguous(bindings);
    }
}

#[test]
fn raw_and_dense_phase_geometry_matches_the_frozen_sequence() {
    let round_zero = RawRoundZeroParams::target_shard();
    assert_eq!(round_zero.cycles, 1 << 26);
    assert_eq!(round_zero.blocks, 1 << 18);
    assert_eq!(round_zero.blocks_per_outer, 32);
    assert_eq!(round_zero.e_in_length, 4_096);
    assert_eq!(round_zero.e_out_length, 8_192);

    let coefficient = RawCoefficientParams::for_round(8).unwrap();
    assert_eq!(coefficient.width, 256);
    assert_eq!(coefficient.basis_weight_fields, 768);
    assert_eq!(coefficient.strict_suffix_fields, 256);
    assert_eq!(coefficient.local_weight_fields, 1_024);
    assert_eq!(coefficient.coefficient_fields, 196_608);
    assert_eq!(coefficient.logical_products, 196_608);
    assert_eq!(coefficient.basis_weight_offset(RAW_BASIS_ONE), Some(0));
    assert_eq!(coefficient.basis_weight_offset(RAW_BASIS_GAMMA), Some(256));
    assert_eq!(
        coefficient.basis_weight_offset(RAW_BASIS_GAMMA_SQUARED),
        Some(512)
    );
    assert_eq!(coefficient.strict_suffix_offset(), 768);
    assert_eq!(coefficient.coefficient_index(2, 255, 255), Some(196_607));

    let replay = RawReplayParams::target_shard(8, 1).unwrap();
    assert_eq!(replay.remaining_cycles, 1 << 18);
    assert_eq!(replay.flags, RAW_FLAG_MATERIALIZE_DENSE);
    assert_eq!(RawReplayParams::target_shard(7, 1).unwrap().flags, 0);
    assert_eq!(replay.replay_e_in_length, 16);

    let round9 = DenseRoundParams::target_shard(9).unwrap();
    assert_eq!(round9.source_rows, 1 << 18);
    assert_eq!(round9.destination_rows, 1 << 17);
    assert_eq!(round9.pair_count, 1 << 16);
    assert_eq!(round9.e_in_length, 8);
    assert_eq!(round9.e_out_length, 8_192);
    assert_eq!(DenseRoundParams::dynamic_threadgroup_bytes(), 128);

    let round25 = DenseRoundParams::target_shard(25).unwrap();
    assert_eq!(round25.source_rows, 4);
    assert_eq!(round25.destination_rows, 2);
    assert_eq!(round25.pair_count, 1);
    assert_eq!(round25.e_in_length, 1);
    assert_eq!(round25.e_out_length, 1);

    let histogram = HistogramParams::target_shard();
    assert_eq!(histogram.blocks_per_hi, 32);
    assert_eq!(histogram.e_hi_length, 8_192);
    assert_eq!(histogram.e_lo_length, 8_192);
}

#[test]
fn raw_local_coefficients_use_strict_write_suffixes() {
    let lambda = [2i64, -4, -3, 6];
    let mut strict_suffix = [0i64; 4];
    let mut suffix_sum = 0;
    for position in (0..lambda.len()).rev() {
        strict_suffix[position] = suffix_sum;
        suffix_sum += lambda[position];
    }
    assert_eq!(strict_suffix, [-1, 3, 6, 0]);

    let even_value = 10 + 3 * strict_suffix[0] - 2 * strict_suffix[2];
    let odd_value = 11 + 4 * strict_suffix[1] - strict_suffix[3];
    assert_eq!((even_value, odd_value), (-5, 23));

    let even_complement = [
        1 - strict_suffix[0],
        1 - strict_suffix[1],
        1 - strict_suffix[2],
        1 - strict_suffix[3],
    ];
    assert_eq!(even_complement, [2, -2, -5, 1]);
    assert_eq!(3 * even_complement[0] - 2 * even_complement[2], 16);
    assert_eq!(4 * strict_suffix[1] - strict_suffix[3], 12);
    assert_eq!(odd_value - even_value, 28);

    let q_0 = -10 * even_value + 4 * 5;
    let q_infinity = -54 * (odd_value - even_value) - 7 * 4;
    assert_eq!(q_0, 70);
    assert_eq!(q_infinity, -1_540);
}

#[test]
fn dense_rounds_require_exactly_thirty_three_reductions() {
    let mut reductions = 0;
    for round in 9..=25 {
        let mut count = DenseRoundParams::target_shard(round).unwrap().e_out_length;
        while count > 1 {
            count = ReductionParams::new(count, 2).unwrap().output_count;
            reductions += 1;
        }
    }
    assert_eq!(reductions, 33);
}

#[test]
fn phase_schedule_keeps_the_junction_and_async_histogram_visible() {
    assert_eq!(PHASES.len(), 5);
    assert_eq!(PHASES[2].name, "round-8 dense junction");
    assert_eq!(PHASES[2].first_round, Some(8));
    assert_eq!(PHASES[2].last_round, Some(8));
    assert_eq!(PHASES[4].name, "read histograms");
    assert_eq!(
        PHASES[4].host_schedule,
        HostSchedule::OverlapAddressTailThenJoin
    );
    assert_eq!(
        PHASES
            .iter()
            .map(|phase| phase.dispatches as u32)
            .sum::<u32>(),
        98
    );
    assert_eq!(
        PHASES
            .iter()
            .map(|phase| phase.barriers as u32)
            .sum::<u32>(),
        71
    );
    assert_eq!(
        PHASES
            .iter()
            .map(|phase| phase.command_buffers as u32)
            .sum::<u32>(),
        27
    );
}

#[test]
fn only_algebraically_closed_pipelines_are_shader_ready() {
    assert_eq!(PIPELINES.len(), 6);
    for descriptor in [
        RAW_ROUND_ZERO_DESCRIPTOR,
        RAW_COEFFICIENT_DESCRIPTOR,
        RAW_REPLAY_DESCRIPTOR,
    ] {
        assert_eq!(descriptor.readiness, PipelineReadiness::AbiOnly);
        assert!(!SOURCE.contains(&format!("kernel void {}", descriptor.name)));
    }
    for descriptor in [DENSE_DESCRIPTOR, REDUCTION_DESCRIPTOR, HISTOGRAM_DESCRIPTOR] {
        assert_eq!(descriptor.readiness, PipelineReadiness::ExactShader);
        assert!(SOURCE.contains(&format!("kernel void {}", descriptor.name)));
    }
    assert_eq!(SOURCE.matches("threadgroup_barrier").count(), 1);
    assert!(SOURCE.contains("x_in = simdgroup"));
    assert!(SOURCE.contains("column = lane + bank * REGISTERS_RW_V3_SIMD_WIDTH"));
    assert!(SOURCE.contains("device const ushort* rs1_offsets [[buffer(0)]]"));
    assert!(SOURCE.contains("rs1_positions[position_base + event]"));
    assert!(SOURCE.contains(DENSE_BIND_MESSAGE_PIPELINE));
    assert!(SOURCE.contains(REDUCE_COLUMNS_PIPELINE));
    assert!(SOURCE.contains(HISTOGRAM_PIPELINE));
    assert!(!SOURCE.contains(RAW_ROUND_ZERO_PIPELINE));
    assert!(!SOURCE.contains(RAW_COEFFICIENT_PIPELINE));
    assert!(!SOURCE.contains(RAW_REPLAY_PIPELINE));
}

#[test]
fn lifetime_map_preserves_the_registers_val_handoff() {
    assert_eq!(
        REGISTERS_VAL_HANDOFF.completion_point,
        SequencePoint::AfterRawRound1
    );
    assert_eq!(REGISTERS_VAL_HANDOFF.planes, [Arena::RdIndex, Arena::RdInc]);
    for arena in REGISTERS_VAL_HANDOFF.planes {
        let lifetime = ARENA_LIFETIMES
            .iter()
            .find(|lifetime| lifetime.arena == arena)
            .unwrap();
        assert_eq!(lifetime.last_use, SequencePoint::AfterRawRound1);
        assert_eq!(lifetime.disposition, LifetimeDisposition::RegistersVal);
    }
    for arena in [Arena::DenseStateA, Arena::DenseStateB] {
        let lifetime = ARENA_LIFETIMES
            .iter()
            .find(|lifetime| lifetime.arena == arena)
            .unwrap();
        assert_eq!(lifetime.last_use, SequencePoint::AfterDenseRound25);
        assert_eq!(lifetime.disposition, LifetimeDisposition::HistogramScratch);
    }
}

#[test]
fn log26_execution_census_reconstructs_exactly() {
    let model = Log26ExecutionModel::checked().unwrap();
    assert_eq!(model.raw.products.full, 184_336_380);
    assert_eq!(model.raw.products.half, 1_119_975_750);
    assert_eq!(model.raw.cache_unique_bytes, LOG26_RAW_CACHE_BYTES);
    assert_eq!(model.raw.requested_bytes, LOG26_RAW_REQUESTED_BYTES);
    assert_eq!(model.dense.products.full, 135_085_048);
    assert_eq!(model.dense.cache_unique_bytes, LOG26_DENSE_CACHE_BYTES);
    assert_eq!(model.dense.requested_bytes, LOG26_DENSE_REQUESTED_BYTES);
    assert_eq!(model.histogram.products.full, 2_097_152);
    assert_eq!(
        model.histogram.cache_unique_bytes,
        LOG26_HISTOGRAM_CACHE_BYTES
    );
    assert_eq!(
        model.histogram.requested_bytes,
        LOG26_HISTOGRAM_REQUESTED_BYTES
    );
    assert_eq!(model.execution.products.full, 321_518_580);
    assert_eq!(model.execution.products.half, 1_119_975_750);
    assert_eq!(
        model.execution.cache_unique_bytes,
        LOG26_EXECUTION_CACHE_BYTES
    );
    assert_eq!(
        model.execution.requested_bytes,
        LOG26_EXECUTION_REQUESTED_BYTES
    );
    assert_eq!(model.launches.dispatches, 98);
    assert_eq!(model.launches.barriers, 71);
    assert_eq!(model.launches.command_buffers, 27);
    assert_eq!(model.launches.host_waits, 27);
}

#[test]
fn allocation_and_traffic_accounting_do_not_mix() {
    assert_eq!(
        LOG26_TOPOLOGY_ALLOCATION_BYTES + LOG26_REGISTERS_VAL_BYTES,
        LOG26_PRODUCER_ALLOCATION_BYTES
    );
    assert_eq!(
        LOG26_TOPOLOGY_INITIALIZED_BYTES + LOG26_REGISTERS_VAL_BYTES,
        LOG26_PRODUCER_INITIALIZED_WRITE_BYTES
    );
    assert_eq!(
        LOG26_EXECUTION_CACHE_BYTES + LOG26_PRODUCER_INITIALIZED_WRITE_BYTES,
        LOG26_LIFECYCLE_CACHE_BYTES
    );
    assert_eq!(
        LOG26_EXECUTION_REQUESTED_BYTES + LOG26_PRODUCER_INITIALIZED_WRITE_BYTES,
        LOG26_LIFECYCLE_REQUESTED_BYTES
    );
}

#[test]
fn logical_peak_scales_through_the_supported_trace_range() {
    assert_eq!(
        trace_peak_logical_bytes(26).unwrap(),
        LOG26_PEAK_LOGICAL_BYTES
    );
    assert_eq!(trace_peak_logical_bytes(27).unwrap(), 11_147_411_520);
    assert_eq!(trace_peak_logical_bytes(28).unwrap(), 22_290_653_312);
    assert!(trace_peak_logical_bytes(25).is_err());
    assert!(trace_peak_logical_bytes(29).is_err());

    assert_eq!(
        TraceExecutionPlan::for_log_t(26).unwrap(),
        TraceExecutionPlan {
            log_t: 26,
            metal_shards: 1,
            metal_cycle_rounds: 26,
            cpu_high_cycle_rounds: 0,
            cpu_address_rounds: 7,
        }
    );
    assert_eq!(TraceExecutionPlan::for_log_t(28).unwrap().metal_shards, 4);
    assert_eq!(
        TraceExecutionPlan::for_log_t(28)
            .unwrap()
            .cpu_high_cycle_rounds,
        2
    );
}

#[test]
fn durable_denominator_and_pursuit_budgets_are_explicit() {
    assert_eq!(CPU_BASELINE_NS, 934_665_875);
    assert_eq!(LATEST_DIAGNOSTIC_CPU_NS, 971_178_000);
    assert_eq!(CURRENT_CPU_FALLBACK_NS, 948_053_000);
    assert_eq!(PRODUCER_PURSUIT_CAP_NS, 45_000_000);
    assert_eq!(ANALYTICAL_EXECUTION_LOW_NS, 89_100_000);
    assert_eq!(ANALYTICAL_EXECUTION_HIGH_NS, 118_500_000);
    assert_eq!(FIXED_COEFFICIENT_ARENA_BYTES, 3_145_728);
    assert_eq!(FIXED_LOCAL_WEIGHT_ARENA_BYTES, 16_384);
    assert_eq!(TIME_BUDGETS[0].complete_cap_ns, 186_933_175);
    assert_eq!(
        TIME_BUDGETS[0].execution_cap_at_producer_pursuit_ns,
        141_933_175
    );
    assert_eq!(TIME_BUDGETS[1].complete_cap_ns, 155_777_645);
    assert_eq!(TIME_BUDGETS[2].complete_cap_ns, 133_523_696);
    assert_eq!(TIME_BUDGETS[3].complete_cap_ns, 116_833_234);
    assert_eq!(
        LOG26_RAW_ROUND_PRODUCTS.map(|round| round.products.full),
        [
            49_152,
            100_679_692,
            41_959_472,
            20_988_096,
            10_502_912,
            5_262_336,
            2_650_112,
            1_376_256,
            868_352,
        ]
    );

    let _model = Log26ExecutionModel::checked().unwrap();
    assert_eq!(
        Log26ExecutionModel::projected_complete_ns(),
        (134_100_000, 163_500_000)
    );
    let (lower, upper) = Log26ExecutionModel::projected_speedup();
    assert!(lower > 5.71 && lower < 5.72);
    assert!(upper > 6.96 && upper < 6.98);
}

fn assert_contiguous(bindings: &[BufferBinding]) {
    for (slot, binding) in bindings.iter().enumerate() {
        assert_eq!(binding.slot, slot as u64);
    }
}
