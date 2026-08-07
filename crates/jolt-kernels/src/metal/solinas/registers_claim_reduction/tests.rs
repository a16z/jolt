use core::mem::{align_of, size_of};

use jolt_field::AkitaField;

use super::super::registers_read_write_v3::{
    RegisterBcsr256, RegisterRead, RegisterRow, RegisterWrite, REGISTER_CSR_COLUMNS,
};
use super::bcsr_runtime::RegistersClaimBcsrRuntimeError;
use super::*;

fn field(value: u64) -> AkitaField {
    AkitaField::from_u64(value)
}

fn splitmix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9e37_79b9_7f4a_7c15);
    value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    value ^ (value >> 31)
}

fn native_plane(rows: usize, seed: u64) -> Vec<u64> {
    const EDGES: [u64; 6] = [
        0,
        1,
        (1_u64 << 32) - 1,
        1_u64 << 32,
        (1_u64 << 32) + 1,
        u64::MAX,
    ];
    (0..rows)
        .map(|index| {
            if index < EDGES.len() {
                EDGES[(index + seed as usize) % EDGES.len()]
            } else {
                splitmix64(seed.wrapping_add(index as u64))
            }
        })
        .collect()
}

fn bcsr_fixture(cycles: usize) -> (RegisterBcsr256, Vec<RegisterRow>) {
    let mut initial_values = core::array::from_fn(|register| register as u64 * 1_003 + 17);
    initial_values[0] = u64::MAX;
    let mut state = initial_values;
    let mut rows = Vec::with_capacity(cycles);
    for cycle in 0..cycles {
        let local = cycle % 256;
        let rs1_register = if local == 7 {
            0
        } else if local == 0 && cycle != 0 {
            1
        } else {
            ((3 * cycle + 7) % REGISTER_CSR_COLUMNS) as u8
        };
        let rs2_register = ((5 * cycle + 11) % REGISTER_CSR_COLUMNS) as u8;
        let rs1 = (cycle.is_multiple_of(2) || local == 0 || local == 7)
            .then(|| RegisterRead::new(rs1_register, state[usize::from(rs1_register)]));
        let rs2 = (cycle.is_multiple_of(3) || local == 7)
            .then(|| RegisterRead::new(rs2_register, state[usize::from(rs2_register)]));
        let rd_register = if local == 7 {
            0
        } else if local == 255 {
            1
        } else {
            ((7 * cycle + 13) % REGISTER_CSR_COLUMNS) as u8
        };
        let rd = (cycle.is_multiple_of(5) || local == 7 || local == 255).then(|| {
            let register = usize::from(rd_register);
            let pre_value = state[register];
            let post_value = if local == 7 {
                pre_value
            } else {
                pre_value.wrapping_add((cycle % 23 + 1) as u64)
            };
            state[register] = post_value;
            RegisterWrite::new(rd_register, pre_value, post_value)
        });
        rows.push(RegisterRow::new(rs1, rs2, rd));
    }
    let (bcsr, _) = RegisterBcsr256::from_rows(&rows, &initial_values).unwrap();
    (bcsr, rows)
}

fn dense_planes_from_rows(rows: &[RegisterRow]) -> (Vec<u64>, Vec<u64>, Vec<u64>) {
    let rd = rows
        .iter()
        .map(|row| row.rd().map_or(0, RegisterWrite::post_value))
        .collect();
    let rs1 = rows
        .iter()
        .map(|row| row.rs1().map_or(0, RegisterRead::value))
        .collect();
    let rs2 = rows
        .iter()
        .map(|row| row.rs2().map_or(0, RegisterRead::value))
        .collect();
    (rd, rs1, rs2)
}

fn read_index_planes_from_rows(rows: &[RegisterRow]) -> (Vec<u8>, Vec<u8>) {
    let rs1 = rows
        .iter()
        .map(|row| row.rs1().map_or(u8::MAX, |read| read.register()))
        .collect();
    let rs2 = rows
        .iter()
        .map(|row| row.rs2().map_or(u8::MAX, |read| read.register()))
        .collect();
    (rs1, rs2)
}

fn challenge_point(log_t: usize) -> Vec<AkitaField> {
    let minus_one = -field(1);
    (0..log_t)
        .map(|index| match index % 5 {
            0 => minus_one,
            1 => field(0),
            2 => field(1),
            3 => field(2),
            _ => field(splitmix64(index as u64)),
        })
        .collect()
}

fn bcsr_challenge_point(log_t: usize) -> Vec<AkitaField> {
    (0..log_t)
        .map(|index| field(splitmix64(index as u64 + 0x7a11_ce55)))
        .collect()
}

fn assert_unfactored_oracle(rows: usize, gamma: AkitaField) {
    let geometry = RegistersClaimGeometry::new(rows).unwrap();
    let rd = native_plane(rows, 0);
    let rs1 = native_plane(rows, 1);
    let rs2 = native_plane(rows, 2);
    let planes = RegisterValuePlanes::new(geometry, &rd, &rs1, &rs2).unwrap();
    let tau = challenge_point(geometry.log_t());

    let factorized = build_linear_q(geometry, planes, &tau, gamma).unwrap();
    let unfactored = build_dense_reference_q(geometry, planes, &tau, gamma).unwrap();
    assert_eq!(factorized, unfactored);

    let prefix_claim = unfactored
        .p
        .iter()
        .zip(&unfactored.q)
        .fold(field(0), |sum, (p, q)| sum + *p * *q);
    let openings = dense_register_openings(geometry, planes, &tau).unwrap();
    assert_eq!(prefix_claim, output_combination(openings, gamma));
}

#[test]
fn resident_bcsr_log26_contract_is_exact() {
    assert_eq!(size_of::<RegistersClaimBcsrComponentParams>(), 32);
    assert_eq!(size_of::<RegistersClaimBcsrReduceParams>(), 16);
    assert_eq!(size_of::<RegistersClaimBcsrMidpointParams>(), 32);
    assert_eq!(BCSR_COMPONENT_REPLAY_BYTES, 6_144);
    assert_eq!(BCSR_COMPONENT_THREADGROUP_BYTES, 6_160);
    assert_eq!(BCSR_INDEXED_EVENT_BYTES, 512);
    assert_eq!(BCSR_INDEXED_THREADGROUP_BYTES, 528);
    assert_eq!(BCSR_COMPONENT_THREADGROUPS, 8_192);
    assert_eq!(BCSR_COMPONENT_REDUCE_THREADGROUPS, 96);
    assert_eq!(BCSR_MIDPOINT_THREADGROUPS, 8_192);

    let plan = RegistersClaimBcsrPlan::log26().unwrap();
    assert_eq!(plan.component_partial_bytes, 100_663_296);
    assert_eq!(plan.component_carrier_bytes, 393_216);
    assert_eq!(plan.midpoint_output_bytes, 131_072);
    assert_eq!(plan.roof.optimistic_floor_ns, 6_519_528);
    assert_eq!(plan.roof.no_cache_floor_ns, 7_937_740);

    assert_eq!(
        REGISTERS_CLAIM_LOG26_SOURCE_COSTS.map(RegistersClaimSourceCost::charged_source_bytes),
        [3_758_096_384, 2_872_865_208, 1_560_514_104]
    );
}

#[test]
fn linear_q_abi_and_slots_are_fixed() {
    assert_eq!(REGISTERS_CLAIM_AKITA_OFFSET, 0xffff_a7f7);
    assert_eq!(size_of::<RegistersClaimParams>(), 16);
    assert_eq!(align_of::<RegistersClaimParams>(), 4);
    assert_eq!(
        [
            LINEAR_Q_RD_WRITE_VALUE_SLOT,
            LINEAR_Q_RS1_VALUE_SLOT,
            LINEAR_Q_RS2_VALUE_SLOT,
            LINEAR_Q_GAMMA_POWERS_SLOT,
            LINEAR_Q_EQ_SUFFIX_SLOT,
            LINEAR_Q_OUTPUT_SLOT,
            LINEAR_Q_PARAMS_SLOT,
        ],
        [0, 1, 2, 3, 4, 5, 6]
    );
    assert_eq!(
        [
            DIRECT_FOLD_RD_WRITE_VALUE_SLOT,
            DIRECT_FOLD_RS1_VALUE_SLOT,
            DIRECT_FOLD_RS2_VALUE_SLOT,
            DIRECT_FOLD_EQ_PREFIX_SLOT,
            DIRECT_FOLD_OUTPUT_SLOT,
            DIRECT_FOLD_PARAMS_SLOT,
            DIRECT_FOLD_THREADGROUP_SLOT,
        ],
        [0, 1, 2, 3, 4, 5, 0]
    );
    assert!(SOURCE.contains("kernel void solinas_registers_claim_build_linear_q"));
    assert!(SOURCE.contains("kernel void solinas_registers_claim_build_linear_q_canonical"));
    assert!(SOURCE.contains("kernel void solinas_registers_claim_fold_direct"));
    assert!(BCSR_SOURCE.contains("kernel void solinas_registers_claim_bcsr_components"));
    assert!(BCSR_SOURCE.contains("kernel void solinas_registers_claim_bcsr_indexed_components"));
    assert!(BCSR_SOURCE.contains("kernel void solinas_registers_claim_bcsr_reduce_components"));
    assert!(BCSR_SOURCE.contains("RegistersClaimBcsrWorkspace"));
    assert!(BCSR_SOURCE.contains("RegistersClaimBcsrIndexedWorkspace"));
}

#[test]
fn log26_linear_q_accounting_is_exact() {
    let rows = 1usize << REGISTERS_CLAIM_TARGET_LOG_T;
    let plan =
        RegistersClaimLinearQPlan::new(rows, 1usize << 31, RegistersClaimKernelConfig::default())
            .unwrap();
    assert_eq!(plan.geometry.prefix_elements(), 8192);
    assert_eq!(plan.geometry.suffix_elements(), 8192);
    assert_eq!(plan.params.reserved, 0);
    assert_eq!(
        u64::from(plan.params.rows),
        u64::from(plan.params.prefix_elements) * u64::from(plan.params.suffix_elements)
    );
    assert_eq!(plan.storage.native_plane_bytes, 536_870_912);
    assert_eq!(plan.storage.resident_input_bytes, 1_610_612_736);
    assert_eq!(plan.storage.gamma_powers_bytes, 32);
    assert_eq!(plan.storage.eq_suffix_bytes, 131_072);
    assert_eq!(plan.storage.output_bytes, 131_072);
    assert_eq!(plan.storage.private_bytes, 262_176);
    assert_eq!(plan.storage.total_resident_bytes, 1_610_874_912);
    assert_eq!(plan.storage.roof_compulsory_bytes, 1_610_874_880);
    assert_eq!(plan.storage.shader_issued_bytes, 2_684_485_632);
    assert_eq!(plan.useful_threads(), 8192);
    assert_eq!(plan.threadgroups(), 64);
    assert_eq!(plan.dispatched_threads().unwrap(), 8192);

    let work = plan.work().unwrap();
    assert_eq!(work.half_width_terms, 201_326_592);
    assert_eq!(work.full_products, 16_384);
    assert_eq!(work.compulsory_bytes, 1_610_874_880);
    let ceiling = work
        .calibrated_ceiling(RegistersClaimRoofRates::CONSERVATIVE, 80)
        .unwrap();
    assert_eq!(ceiling.half_width_floor_ns, 7_663_162);
    assert_eq!(ceiling.full_product_floor_ns, 906);
    assert_eq!(ceiling.arithmetic_floor_ns, 7_664_068);
    assert_eq!(ceiling.traffic_floor_ns, 3_566_236);
    assert_eq!(ceiling.roof_floor_ns, 7_664_068);
    assert_eq!(ceiling.utilization_cap_ns, 9_580_085);
}

#[test]
fn linear_q_plan_rejects_an_undersized_plane_limit() {
    assert!(matches!(
        RegistersClaimLinearQPlan::new(
            1usize << REGISTERS_CLAIM_TARGET_LOG_T,
            (1usize << 29) - 1,
            RegistersClaimKernelConfig::default(),
        ),
        Err(RegistersClaimPlanError::BufferTooLarge {
            name: "linear-q native plane",
            bytes: 536_870_912,
            max_buffer_length: 536_870_911,
        })
    ));
}

#[test]
fn unfactored_oracle_matches_even_and_odd_splits() {
    for rows in [1usize << 12, 1usize << 13] {
        for gamma in [field(0), field(1), -field(1), field(0xfeed_face_cafe_beef)] {
            assert_unfactored_oracle(rows, gamma);
        }
    }
}

#[test]
fn unfactored_oracle_covers_long_maximal_carry_chains() {
    let rows = 1usize << 18;
    let geometry = RegistersClaimGeometry::new(rows).unwrap();
    let rd = vec![u64::MAX; rows];
    let rs1 = vec![(1_u64 << 32) + 1; rows];
    let rs2 = vec![(1_u64 << 32) - 1; rows];
    let planes = RegisterValuePlanes::new(geometry, &rd, &rs1, &rs2).unwrap();
    let tau = vec![-field(1); geometry.log_t()];
    let gamma = -field(1);

    assert_eq!(
        build_linear_q(geometry, planes, &tau, gamma).unwrap(),
        build_dense_reference_q(geometry, planes, &tau, gamma).unwrap()
    );
}

#[test]
fn q_checksum_is_canonical_and_order_sensitive() {
    let values = [field(0), field(1), -field(1), field(u64::MAX)];
    let checksum = registers_claim_q_checksum(&values);
    assert_eq!(checksum, 0xf9a2_fb60_670c_1980);

    let mut reversed = values;
    reversed.reverse();
    assert_ne!(checksum, registers_claim_q_checksum(&reversed));
}

#[test]
fn metal_linear_q_matches_the_unfactored_oracle() {
    let context = match super::super::SolinasMetal::for_akita() {
        Ok(context) => context,
        Err(super::super::MetalError::DeviceUnavailable) => return,
        Err(error) => panic!("Akita Metal library failed to compile: {error:?}"),
    };
    let rows = 1usize << 12;
    let geometry = RegistersClaimGeometry::new(rows).unwrap();
    let rd = vec![u64::MAX; rows];
    let rs1 = vec![(1_u64 << 32) + 1; rows];
    let rs2 = vec![(1_u64 << 32) - 1; rows];
    let planes = RegisterValuePlanes::new(geometry, &rd, &rs1, &rs2).unwrap();
    let resident = context
        .prepare_registers_claim_resident_planes(&rd, &rs1, &rs2)
        .unwrap();
    let tau = vec![-field(1); geometry.log_t()];
    assert!(matches!(
        context.prepare_registers_claim_linear_q(
            &resident,
            &tau[..tau.len() - 1],
            -field(1),
            RegistersClaimKernelConfig::default(),
        ),
        Err(RegistersClaimLinearQError::WrongPointLength {
            expected: 12,
            actual: 11,
        })
    ));
    for gamma in [field(0), field(1), -field(1), field(0xfeed_face_cafe_beef)] {
        let expected = build_dense_reference_q(geometry, planes, &tau, gamma).unwrap();
        for accumulator in [
            RegistersClaimAccumulator::Deferred224,
            RegistersClaimAccumulator::Canonical128,
        ] {
            let invocation = context
                .prepare_registers_claim_linear_q(
                    &resident,
                    &tau,
                    gamma,
                    RegistersClaimKernelConfig {
                        accumulator,
                        ..RegistersClaimKernelConfig::default()
                    },
                )
                .unwrap();

            assert_eq!(invocation.execute_device_buffer_allocations(), 0);
            assert_eq!(
                invocation.source_allocation_identities(),
                resident.allocation_identities()
            );
            assert!(!invocation
                .source_allocation_identities()
                .contains(&invocation.output_allocation_identity()));
            let observation = invocation.execute_timed().unwrap();
            assert_eq!(observation.q, expected.q);
            assert_eq!(
                observation.checksum,
                registers_claim_q_checksum(&expected.q)
            );
            assert_eq!(observation.useful_half_width_terms, 3 * rows as u64);
            assert_eq!(
                observation.full_products,
                2 * geometry.prefix_elements() as u64
            );
        }
    }

    let prefix_challenges = challenge_point(geometry.prefix_vars());
    assert!(matches!(
        context.prepare_registers_claim_direct_fold(
            &resident,
            &prefix_challenges[..prefix_challenges.len() - 1],
            RegistersClaimKernelConfig::default(),
        ),
        Err(RegistersClaimLinearQError::WrongPrefixChallengeCount {
            expected: 6,
            actual: 5,
        })
    ));
    let expected = fold_direct(geometry, planes, &prefix_challenges).unwrap();
    let invocation = context
        .prepare_registers_claim_direct_fold(
            &resident,
            &prefix_challenges,
            RegistersClaimKernelConfig::default(),
        )
        .unwrap();
    assert_eq!(invocation.execute_device_buffer_allocations(), 0);
    let observation = invocation.execute_timed().unwrap();
    assert_eq!(observation.outputs, expected);
    assert_eq!(observation.useful_half_width_terms, 3 * rows as u64);
    assert_eq!(observation.threadgroups, geometry.suffix_elements());
}

#[test]
fn metal_bcsr_components_match_dense_factorization() {
    let context = match super::super::SolinasMetal::for_akita() {
        Ok(context) => context,
        Err(super::super::MetalError::DeviceUnavailable) => return,
        Err(error) => panic!("Akita Metal library failed to compile: {error:?}"),
    };
    let cycles = 1usize << 16;
    let geometry = RegistersClaimGeometry::new(cycles).unwrap();
    let (bcsr, rows) = bcsr_fixture(cycles);
    let (rd, rs1, rs2) = dense_planes_from_rows(&rows);
    let planes = RegisterValuePlanes::new(geometry, &rd, &rs1, &rs2).unwrap();
    let tau = bcsr_challenge_point(geometry.log_t());
    let expected = build_linear_components(geometry, planes, &tau).unwrap();

    assert!(matches!(
        context.prepare_registers_claim_bcsr_components(
            &bcsr,
            &tau,
            RegistersClaimBcsrKernelConfig {
                partial_blocks: 3,
                ..RegistersClaimBcsrKernelConfig::default()
            },
        ),
        Err(RegistersClaimBcsrRuntimeError::InvalidState(_))
    ));

    let (rs1_index, rs2_index) = read_index_planes_from_rows(&rows);
    for replay in [
        RegistersClaimBcsrReplayStrategy::ColumnReplay,
        RegistersClaimBcsrReplayStrategy::IndexedPredecessor,
    ] {
        for partial_blocks in [8, 32, 64, 128, 256] {
            let config = RegistersClaimBcsrKernelConfig {
                partial_blocks,
                replay,
            };
            let invocation = match replay {
                RegistersClaimBcsrReplayStrategy::ColumnReplay => context
                    .prepare_registers_claim_bcsr_components(&bcsr, &tau, config)
                    .unwrap(),
                RegistersClaimBcsrReplayStrategy::IndexedPredecessor => context
                    .prepare_registers_claim_bcsr_indexed_components(
                        &bcsr, &rs1_index, &rs2_index, &tau, config,
                    )
                    .unwrap(),
            };
            let observation = invocation.execute_timed().unwrap();
            assert_eq!(observation.components, expected);
            assert_eq!(observation.dispatches, 2);
        }
    }
}

#[test]
fn synthetic_bcsr_benchmark_preserves_the_checked_boundary() {
    let context = match super::super::SolinasMetal::for_akita() {
        Ok(context) => context,
        Err(super::super::MetalError::DeviceUnavailable) => return,
        Err(error) => panic!("Akita Metal library failed to compile: {error:?}"),
    };
    let cycles = 1usize << 16;
    let tau = bcsr_challenge_point(cycles.trailing_zeros() as usize);
    let mut expected_checksum = None;
    for (replay, source_bytes) in [
        (RegistersClaimBcsrReplayStrategy::ColumnReplay, 1_181_184),
        (
            RegistersClaimBcsrReplayStrategy::IndexedPredecessor,
            1_049_088,
        ),
    ] {
        let invocation = context
            .prepare_registers_claim_bcsr_benchmark(
                cycles,
                &tau,
                RegistersClaimBcsrKernelConfig {
                    partial_blocks: 8,
                    replay,
                },
            )
            .unwrap();
        let cold = invocation.execute_timed().unwrap();
        let warm = invocation.execute_timed().unwrap();
        assert_eq!(cold.checksum, warm.checksum);
        if let Some(expected_checksum) = expected_checksum {
            assert_eq!(warm.checksum, expected_checksum);
        } else {
            expected_checksum = Some(warm.checksum);
        }
        assert_eq!(warm.event_counts, [58_368, 54_528, 49_152]);
        assert_eq!(warm.dispatches, 2);
        assert_eq!(warm.partial_blocks, 8);
        assert_eq!(warm.replay, replay);
        assert_eq!(warm.component_threadgroups, 8);
        assert_eq!(warm.source_bytes, source_bytes);
        assert_eq!(warm.partial_bytes, 98_304);
    }
}
