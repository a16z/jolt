#![cfg(all(feature = "metal", target_os = "macos"))]
#![expect(clippy::expect_used, reason = "integration test")]

use jolt_field::{AkitaField, FromPrimitiveInt};
use jolt_kernels::metal::solinas::{
    dense_pushforward_oracle, product_remainder_reference, ram_raf_split_equality,
    ram_val_check_oracle, split_pushforward_oracle,
};
use jolt_kernels::metal::solinas::{
    BooleanityRow, BooleanitySelector, BooleanitySequenceConfig, DispatchConfig, Fp128, MetalError,
    Probe, Product5SequenceConfig, ProductRemainderRow, ProductRemainderSequenceConfig,
    RamRafConfig, RamValCheckConfig, RamValCheckDenseRow, RamValCheckNativeRow, RamValCheckPlan,
    RegistersValDenseConfig, RegistersValFirstMessageConfig, RegistersValTransitionConfig,
    SolinasMetal, AKITA_OFFSET_FFFFA7F7, OFFSET_275, RAM_RAF_ADDRESS_DOMAIN, RAM_RAF_NO_ACCESS,
};
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, LtPolynomial};

mod support;

use support::{
    expected_field_for_offset, expected_u32_mad, inputs, inputs_for_offset,
    product5_fused_transition, product5_message, values, PRODUCT5_FACTORS,
};

#[test]
fn akita_field_uses_the_metal_abi() {
    for value in [
        0,
        1,
        u64::MAX as u128,
        (u64::MAX as u128) << 64,
        u128::MAX - AKITA_OFFSET_FFFFA7F7 as u128,
    ] {
        let field = AkitaField::from_u128(value);
        let encoded = Fp128::from_jolt_field(&field);
        assert_eq!(encoded.to_u128(), value);
        assert_eq!(encoded.into_jolt_field::<AkitaField>(), field);
    }
}

#[test]
fn akita_shader_matches_jolt_field() {
    let context = SolinasMetal::for_akita().expect("Akita Metal context should compile");
    assert_eq!(context.device_info().offset, AKITA_OFFSET_FFFFA7F7);
    let (lhs, rhs) = inputs_for_offset(4099, AKITA_OFFSET_FFFFA7F7);

    for probe in [Probe::Copy, Probe::Add, Probe::Sub, Probe::MulWide] {
        let invocation = context
            .prepare(probe, &lhs, &rhs, DispatchConfig::default())
            .expect("Akita probe should compile");
        invocation.execute().expect("Akita probe should execute");
        let actual = invocation.read_output().expect("Akita output should read");
        let expected = lhs
            .iter()
            .zip(&rhs)
            .map(|(&lhs, &rhs)| {
                let lhs = lhs.into_jolt_field::<AkitaField>();
                let rhs = rhs.into_jolt_field::<AkitaField>();
                let value = match probe {
                    Probe::Copy => lhs,
                    Probe::Add => lhs + rhs,
                    Probe::Sub => lhs - rhs,
                    Probe::MulWide => lhs * rhs,
                    _ => unreachable!("probe list contains only pointwise operations"),
                };
                Fp128::from_jolt_field(&value)
            })
            .collect::<Vec<_>>();
        assert_eq!(actual, expected, "{}", probe.name());
    }
}

#[test]
fn product_remainder_sequence_matches_cpu_at_every_boundary() {
    for rows in [1 << 8, 1 << 9] {
        assert_product_remainder_sequence(rows);
    }
}

#[test]
fn ram_val_check_sequence_matches_cpu_at_every_boundary() {
    let context = SolinasMetal::for_akita().expect("Akita Metal context should compile");
    let log_t = 12;
    let log_k = 5;
    let cycles = 1usize << log_t;
    let address_domain = 1usize << log_k;
    let config = RamValCheckConfig {
        first_message_threads: 32,
        native_transition_threads: 32,
        dense_transition_threads: 64,
        cpu_tail_elements: 1 << 7,
    };
    let plan =
        RamValCheckPlan::new(log_t, log_k, config).expect("RAM value-check plan should be valid");
    let rows = (0..cycles)
        .map(|index| {
            let (address, increment) = match index % 11 {
                0 => (None, 0),
                1 => (Some((index % address_domain) as u32), u64::MAX as i128),
                2 => (Some((index % address_domain) as u32), -(u64::MAX as i128)),
                _ => (
                    Some(((37 * index + 19) % address_domain) as u32),
                    (index as i128 % 2001) - 1000,
                ),
            };
            RamValCheckNativeRow::new(address, increment)
                .expect("synthetic RAM value-check row should be valid")
        })
        .collect::<Vec<_>>();
    let r_address = (0..log_k)
        .map(|index| AkitaField::from_u64(0x101 + 17 * index as u64))
        .collect::<Vec<_>>();
    let r_cycle = (0..log_t)
        .map(|index| AkitaField::from_u64(0x1001 + 29 * index as u64))
        .collect::<Vec<_>>();
    let eq_address = EqPolynomial::<AkitaField>::evals(&r_address, None);
    let (r_hi, r_lo) = r_cycle.split_at(log_t - log_t / 2);
    let mut lt_lo = LtPolynomial::<AkitaField>::evaluations(r_lo);
    let gamma = AkitaField::from_u64(0xfeed_beef_cafe_babe);
    let lt_hi = LtPolynomial::<AkitaField>::evaluations(r_hi)
        .into_iter()
        .map(|value| value + gamma)
        .collect::<Vec<_>>();
    let eq_hi = EqPolynomial::<AkitaField>::evals(r_hi, None);
    let expected_first =
        ram_val_check_oracle::first_message(&rows, &eq_address, &lt_lo, &lt_hi, &eq_hi)
            .expect("CPU first message should be well-shaped");

    let resident_rows = context
        .prepare_ram_val_check_rows(&rows, address_domain)
        .expect("RAM value-check rows should prepare");
    let mut sequence = context
        .prepare_ram_val_check_sequence(
            resident_rows.clone(),
            &eq_address,
            &lt_lo,
            &lt_hi,
            &eq_hi,
            plan,
        )
        .expect("RAM value-check sequence should prepare");
    assert_eq!(sequence.round_device_buffer_allocations(), 0);
    assert_eq!(
        sequence.row_allocation_identity(),
        resident_rows.allocation_identity()
    );
    assert_eq!(
        sequence
            .replay_first_message_timed()
            .expect("first message should replay")
            .0,
        expected_first
    );
    assert_eq!(
        sequence.message().expect("first message should execute"),
        expected_first
    );

    let modulus_minus_one = AkitaField::from_u128(u128::MAX - AKITA_OFFSET_FFFFA7F7 as u128);
    let challenges = [
        AkitaField::from_u64(0),
        AkitaField::from_u64(1),
        modulus_minus_one,
        AkitaField::from_u64(0x0123_4567_89ab_cdef),
        AkitaField::from_u64(0x0ddc_0ffe_e15e_cafe),
    ];
    let mut expected_state: Option<Vec<RamValCheckDenseRow<AkitaField>>> = None;
    for (round, challenge) in challenges
        .into_iter()
        .take(plan.gpu_bind_rounds())
        .enumerate()
    {
        lt_lo = lt_lo
            .chunks_exact(2)
            .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
            .collect();
        let expected = if let Some(state) = expected_state.as_ref() {
            ram_val_check_oracle::dense_bind_and_message(state, challenge, &lt_lo, &lt_hi, &eq_hi)
                .expect("CPU dense transition should be well-shaped")
        } else {
            ram_val_check_oracle::native_bind_and_message(
                &rows,
                &eq_address,
                challenge,
                &lt_lo,
                &lt_hi,
                &eq_hi,
            )
            .expect("CPU native transition should be well-shaped")
        };
        assert_eq!(
            sequence
                .replay_current_bind_and_message_timed(challenge, &lt_lo)
                .expect("RAM value-check transition should replay")
                .0,
            expected.evals,
            "replay round {round}"
        );
        assert_eq!(
            sequence
                .bind_and_message(challenge, &lt_lo)
                .expect("RAM value-check transition should execute"),
            expected.evals,
            "round {round}"
        );
        assert_eq!(
            sequence
                .read_current_state()
                .expect("RAM value-check dense state should read"),
            expected.state,
            "state round {round}"
        );
        expected_state = Some(expected.state);
    }
    assert!(sequence.at_cpu_handoff());
    assert_eq!(sequence.current_elements(), config.cpu_tail_elements);
    assert_eq!(sequence.current_lt_lo_length(), lt_lo.len());
}

#[test]
fn ram_raf_pushforward_matches_independent_dense_oracles() {
    let context = SolinasMetal::for_akita().expect("Akita Metal context should compile");
    let rows = 1usize << 16;
    let point = (0..rows.ilog2() as usize)
        .map(|index| {
            AkitaField::from_u64(
                0x9e37_79b9_7f4a_7c15_u64.wrapping_mul(index as u64 + 3) & ((1 << 56) - 1),
            )
        })
        .collect::<Vec<_>>();
    let addresses = (0..rows)
        .map(|index| match index % 19 {
            0 => RAM_RAF_NO_ACCESS,
            1..=9 => 0,
            _ => ((index * 37 + index / 11) % RAM_RAF_ADDRESS_DOMAIN) as u32,
        })
        .collect::<Vec<_>>();
    let config = RamRafConfig {
        trace_cutoff: 1 << 15,
        ..RamRafConfig::default()
    };
    let plane = context
        .prepare_ram_raf_addresses(&addresses, config)
        .expect("RAM RAF address plane should prepare");
    let storage_id = plane.storage_id();
    let sequence = context
        .prepare_ram_raf_sequence(plane, &point, config)
        .expect("RAM RAF sequence should prepare");
    assert_eq!(sequence.address_storage_id(), storage_id);
    assert_eq!(sequence.round_device_buffer_allocations(), 0);

    let observation = sequence
        .execute_timed()
        .expect("RAM RAF pushforward should execute");
    let dense_eq = EqPolynomial::<AkitaField>::evals(&point, None);
    let dense = dense_pushforward_oracle(&addresses, &dense_eq, RAM_RAF_ADDRESS_DOMAIN)
        .expect("dense RAM RAF oracle should execute");
    let (e_lo, e_hi) = ram_raf_split_equality(&point).expect("split equality should prepare");
    let split = split_pushforward_oracle(&addresses, &e_lo, &e_hi, RAM_RAF_ADDRESS_DOMAIN)
        .expect("split RAM RAF oracle should execute");
    assert_eq!(split, dense);
    assert_eq!(observation.masses, dense);
    assert_eq!(observation.counters.invalid_rows, 0);
    assert_eq!(observation.counters.unsupported_dispatches, 0);
    assert_eq!(
        observation.counters.accessed_rows as usize,
        addresses
            .iter()
            .filter(|&&address| address != RAM_RAF_NO_ACCESS)
            .count()
    );

    let replay = sequence
        .execute_timed()
        .expect("RAM RAF pushforward should replay");
    assert_eq!(replay.masses, observation.masses);
    assert_eq!(replay.counters, observation.counters);
}

fn assert_product_remainder_sequence(row_count: usize) {
    let context = SolinasMetal::for_akita().expect("Akita Metal context should compile");
    let modulus_minus_one = AkitaField::from_u128(u128::MAX - AKITA_OFFSET_FFFFA7F7 as u128);
    let rows = (0..row_count)
        .map(|index| {
            let right_input = match index % 5 {
                0 => i128::MIN,
                1 => i128::MAX,
                2 => -1,
                3 => 0,
                _ => index as i128 * 0x1_0000_0001 - 0x1234_5678,
            };
            ProductRemainderRow::new(
                (index as u64)
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .rotate_left((index % 63) as u32),
                right_input,
                index & 1 != 0,
                index % 3 == 0,
                (!(index as u64)).wrapping_mul(0xbf58_476d_1ce4_e5b9),
                index % 5 == 0,
                index % 7 == 0,
                index % 11 == 0,
            )
        })
        .collect::<Vec<_>>();
    let lagrange = [
        AkitaField::from_u64(0),
        AkitaField::from_u64(0x0123_4567_89ab_cdef),
        modulus_minus_one,
    ];
    let weights = |length: usize, salt: usize| {
        (0..length)
            .map(|index| match (index + salt) % 4 {
                0 => AkitaField::from_u64(0),
                1 => AkitaField::from_u64(1),
                2 => modulus_minus_one,
                _ => AkitaField::from_u64(
                    (index as u64)
                        .wrapping_mul(0x94d0_49bb_1331_11eb)
                        .wrapping_add(salt as u64),
                ),
            })
            .collect::<Vec<_>>()
    };

    let mut sequence = context
        .prepare_product_remainder_sequence(
            &rows,
            lagrange,
            32,
            16,
            ProductRemainderSequenceConfig::default(),
        )
        .expect("product remainder sequence should prepare");
    assert_eq!(sequence.resident_buffer_count(), 8);
    assert_eq!(sequence.round_device_buffer_allocations(), 0);

    let materialize_e_in = weights(16, 1);
    let materialize_e_out = weights(row_count / 32, 2);
    let mut expected = product_remainder_reference::materialize_message(
        &rows,
        lagrange,
        &materialize_e_in,
        &materialize_e_out,
    )
    .expect("CPU materialization should be well-shaped");
    assert_eq!(
        sequence
            .message(&materialize_e_in, &materialize_e_out)
            .expect("product remainder first message should execute"),
        expected.endpoints
    );
    assert_eq!(
        sequence
            .replay_materialize_message_timed(&materialize_e_in, &materialize_e_out)
            .expect("product remainder first message should replay")
            .0,
        expected.endpoints
    );
    let (actual_left, actual_right) = sequence
        .read_current_state()
        .expect("materialized state should read");
    assert_eq!([actual_left, actual_right].concat(), expected.state);

    let challenges = [
        AkitaField::from_u64(0),
        AkitaField::from_u64(1),
        modulus_minus_one,
        AkitaField::from_u64(0xfeed_beef_cafe_babe),
        AkitaField::from_u64(0x0123_4567_89ab_cdef),
        AkitaField::from_u64(2),
        AkitaField::from_u64(0xdead_beef),
        AkitaField::from_u64(0x0ddc_0ffe_e15e_cafe),
    ];
    for (round, challenge) in challenges
        .into_iter()
        .take(row_count.ilog2() as usize - 1)
        .enumerate()
    {
        let source_elements = sequence.current_elements();
        let weighted_pairs = source_elements / 4;
        let e_in_length = 1 << (weighted_pairs.ilog2() as usize / 2);
        let e_out_length = weighted_pairs / e_in_length;
        let e_in = weights(e_in_length, 3 + 2 * round);
        let e_out = weights(e_out_length, 4 + 2 * round);
        let next = product_remainder_reference::bind_and_message(
            &expected.state,
            source_elements,
            challenge,
            &e_in,
            &e_out,
        )
        .expect("CPU transition should be well-shaped");
        let replay = sequence
            .replay_current_bind_and_message_timed(challenge, &e_in, &e_out)
            .expect("product remainder transition should replay")
            .0;
        assert_eq!(replay, next.endpoints, "replay round {round}");
        assert_eq!(
            sequence
                .bind_and_message(challenge, &e_in, &e_out)
                .expect("product remainder transition should execute"),
            replay,
            "round {round}"
        );
        let (actual_left, actual_right) = sequence
            .read_current_state()
            .expect("bound state should read");
        assert_eq!(
            [actual_left, actual_right].concat(),
            next.state,
            "round {round}"
        );
        expected.state = next.state;
    }
    assert_eq!(sequence.current_elements(), 2);

    let opening_e_in = weights(row_count / 16, 19);
    let opening_e_out = weights(16, 20);
    let expected_openings =
        product_remainder_reference::openings(&rows, &opening_e_in, &opening_e_out)
            .expect("CPU openings should be well-shaped");
    assert_eq!(
        sequence
            .openings(&opening_e_in, &opening_e_out)
            .expect("product remainder openings should execute"),
        expected_openings
    );
    assert_eq!(
        sequence
            .restart_message_timed(&materialize_e_in, &materialize_e_out)
            .expect("product remainder sequence should restart")
            .0,
        expected.endpoints
    );
    assert_eq!(sequence.current_elements(), row_count);
    assert_eq!(sequence.round_device_buffer_allocations(), 0);
}

#[test]
fn registers_val_first_message_matches_dense_cpu() {
    let context = SolinasMetal::for_akita().expect("Akita Metal context should compile");
    let cycles = 1_usize << 10;
    let inc = (0..cycles)
        .map(|index| {
            AkitaField::from_u64(
                (index as u64)
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .rotate_left((index % 63) as u32),
            )
        })
        .collect::<Vec<_>>();
    let rd = (0..cycles)
        .map(|index| {
            if index % 11 == 0 {
                u8::MAX
            } else {
                ((37 * index + 19) & 127) as u8
            }
        })
        .collect::<Vec<_>>();
    let r_address = (0..7)
        .map(|index| AkitaField::from_u64(0x101 + 17 * index as u64))
        .collect::<Vec<_>>();
    let r_cycle = (0..cycles.ilog2() as usize)
        .map(|index| AkitaField::from_u64(0x1001 + 29 * index as u64))
        .collect::<Vec<_>>();
    let eq_address = EqPolynomial::<AkitaField>::evals(&r_address, None);
    let lt = LtPolynomial::<AkitaField>::evaluations(&r_cycle);
    let mut expected = [AkitaField::from_u64(0); 3];
    for pair in 0..cycles / 2 {
        let indices = [2 * pair, 2 * pair + 1];
        let inc_pair = [inc[indices[0]], inc[indices[1]]];
        let wa_pair = indices.map(|index| {
            let register = rd[index];
            if register == u8::MAX {
                AkitaField::from_u64(0)
            } else {
                eq_address[register as usize]
            }
        });
        let lt_pair = [lt[indices[0]], lt[indices[1]]];
        for (sample, t) in [0_u64, 2, 3].into_iter().enumerate() {
            let t = AkitaField::from_u64(t);
            let interpolate = |pair: [AkitaField; 2]| pair[0] + t * (pair[1] - pair[0]);
            expected[sample] += interpolate(inc_pair) * interpolate(wa_pair) * interpolate(lt_pair);
        }
    }
    let challenge = AkitaField::from_u64(0xfeed_beef_cafe_babe);
    let bind = |low: AkitaField, high: AkitaField| low + challenge * (high - low);
    let mid = r_cycle.len() / 2;
    let (_, r_lo) = r_cycle.split_at(r_cycle.len() - mid);
    let lt_lo = LtPolynomial::<AkitaField>::evaluations(r_lo);
    let bound_lt_lo = lt_lo
        .chunks_exact(2)
        .map(|pair| bind(pair[0], pair[1]))
        .collect::<Vec<_>>();
    let bound_lt = lt
        .chunks_exact(2)
        .map(|pair| bind(pair[0], pair[1]))
        .collect::<Vec<_>>();
    let expected_dense = (0..cycles / 2)
        .map(|pair| {
            let first = 2 * pair;
            let wa = [rd[first], rd[first + 1]].map(|register| {
                if register == u8::MAX {
                    AkitaField::from_u64(0)
                } else {
                    eq_address[register as usize]
                }
            });
            [bind(inc[first], inc[first + 1]), bind(wa[0], wa[1])]
        })
        .collect::<Vec<_>>();
    let mut next_expected = [AkitaField::from_u64(0); 3];
    for pair in 0..expected_dense.len() / 2 {
        let low = expected_dense[2 * pair];
        let high = expected_dense[2 * pair + 1];
        for (sample, t) in [0_u64, 2, 3].into_iter().enumerate() {
            let t = AkitaField::from_u64(t);
            let interpolate = |low: AkitaField, high: AkitaField| low + t * (high - low);
            next_expected[sample] += interpolate(low[0], high[0])
                * interpolate(low[1], high[1])
                * interpolate(bound_lt[2 * pair], bound_lt[2 * pair + 1]);
        }
    }

    for threads_per_threadgroup in [32, 128] {
        let invocation = context
            .prepare_registers_val_first_message(
                &inc,
                &rd,
                &r_address,
                &r_cycle,
                RegistersValFirstMessageConfig {
                    threads_per_threadgroup: Some(threads_per_threadgroup),
                },
            )
            .expect("registers value first message should prepare");
        assert_eq!(invocation.execute_device_buffer_allocations(), 0);
        invocation
            .execute()
            .expect("registers value first message should execute");
        assert_eq!(
            invocation
                .read_message()
                .expect("registers value first message should be readable"),
            expected,
            "threads_per_threadgroup={threads_per_threadgroup}"
        );
        let transition = invocation
            .into_first_transition(
                &bound_lt_lo,
                RegistersValTransitionConfig {
                    threads_per_threadgroup: Some(threads_per_threadgroup),
                },
            )
            .expect("registers value first transition should prepare");
        assert_eq!(transition.execute_device_buffer_allocations(), 0);
        transition
            .execute(challenge)
            .expect("registers value first transition should execute");
        assert_eq!(
            transition
                .read_message()
                .expect("registers value second message should be readable"),
            next_expected,
            "threads_per_threadgroup={threads_per_threadgroup}"
        );
        assert_eq!(
            transition
                .read_dense_state()
                .expect("registers value dense state should be readable"),
            expected_dense,
            "threads_per_threadgroup={threads_per_threadgroup}"
        );
    }
}

#[test]
fn registers_val_transition_handles_odd_even_splits_and_edge_challenges() {
    let context = SolinasMetal::for_akita().expect("Akita Metal context should compile");
    let modulus_minus_one = AkitaField::from_u128(u128::MAX - AKITA_OFFSET_FFFFA7F7 as u128);
    for (log_cycles, challenge) in [
        (2, AkitaField::from_u64(0)),
        (3, modulus_minus_one),
        (4, AkitaField::from_u64(0)),
        (4, AkitaField::from_u64(1)),
        (5, modulus_minus_one),
        (8, AkitaField::from_u64(1)),
        (9, AkitaField::from_u64(0x0123_4567_89ab_cdef)),
    ] {
        let cycles = 1_usize << log_cycles;
        let inc = (0..cycles)
            .map(|index| match index % 4 {
                0 => AkitaField::from_u64(0),
                1 => AkitaField::from_u64(1),
                2 => modulus_minus_one,
                _ => AkitaField::from_u64(index as u64),
            })
            .collect::<Vec<_>>();
        let rd = (0..cycles)
            .map(|index| match index % 4 {
                0 => u8::MAX,
                1 => 0,
                2 => 127,
                _ => ((31 * index + 7) & 127) as u8,
            })
            .collect::<Vec<_>>();
        let r_address = (0..7)
            .map(|index| match index % 3 {
                0 => AkitaField::from_u64(0),
                1 => AkitaField::from_u64(1),
                _ => modulus_minus_one,
            })
            .collect::<Vec<_>>();
        let r_cycle = (0..log_cycles)
            .map(|index| match index % 3 {
                0 => AkitaField::from_u64(0),
                1 => AkitaField::from_u64(1),
                _ => modulus_minus_one,
            })
            .collect::<Vec<_>>();
        let eq_address = EqPolynomial::<AkitaField>::evals(&r_address, None);
        let wa = rd
            .iter()
            .map(|&register| {
                if register == u8::MAX {
                    AkitaField::from_u64(0)
                } else {
                    eq_address[register as usize]
                }
            })
            .collect::<Vec<_>>();
        let lt = LtPolynomial::<AkitaField>::evaluations(&r_cycle);
        let first_expected = registers_val_dense_message(&inc, &wa, &lt);

        for threads_per_threadgroup in [32, 64] {
            let invocation = context
                .prepare_registers_val_first_message(
                    &inc,
                    &rd,
                    &r_address,
                    &r_cycle,
                    RegistersValFirstMessageConfig {
                        threads_per_threadgroup: Some(threads_per_threadgroup),
                    },
                )
                .expect("registers value edge case should prepare");
            invocation
                .execute()
                .expect("registers value edge case should execute");
            assert_eq!(
                invocation
                    .read_message()
                    .expect("registers value edge message should be readable"),
                first_expected,
                "log={log_cycles}, tg={threads_per_threadgroup}"
            );
            if log_cycles < 4 {
                continue;
            }

            let bind = |values: &[AkitaField], challenge: AkitaField| {
                values
                    .chunks_exact(2)
                    .map(|pair| pair[0] + challenge * (pair[1] - pair[0]))
                    .collect::<Vec<_>>()
            };
            let mut dense_inc = bind(&inc, challenge);
            let mut dense_wa = bind(&wa, challenge);
            let mut dense_lt = bind(&lt, challenge);
            let mid = r_cycle.len() / 2;
            let (_, r_lo) = r_cycle.split_at(r_cycle.len() - mid);
            let mut bound_lt_lo = bind(&LtPolynomial::<AkitaField>::evaluations(r_lo), challenge);
            let next_expected = registers_val_dense_message(&dense_inc, &dense_wa, &dense_lt);
            let transition = invocation
                .into_first_transition(
                    &bound_lt_lo,
                    RegistersValTransitionConfig {
                        threads_per_threadgroup: Some(threads_per_threadgroup),
                    },
                )
                .expect("registers value edge transition should prepare");
            transition
                .execute(challenge)
                .expect("registers value edge transition should execute");
            assert_eq!(
                transition
                    .read_message()
                    .expect("registers value edge second message should be readable"),
                next_expected,
                "log={log_cycles}, tg={threads_per_threadgroup}"
            );
            let expected_state = dense_inc
                .iter()
                .copied()
                .zip(dense_wa.iter().copied())
                .map(|(inc, wa)| [inc, wa])
                .collect::<Vec<_>>();
            assert_eq!(
                transition
                    .read_dense_state()
                    .expect("registers value edge dense state should be readable"),
                expected_state,
                "log={log_cycles}, tg={threads_per_threadgroup}"
            );

            let mut sequence = transition
                .into_sequence(RegistersValDenseConfig {
                    threads_per_threadgroup: Some(threads_per_threadgroup),
                })
                .expect("registers value dense sequence should prepare");
            assert_eq!(sequence.round_device_buffer_allocations(), 0);
            let dense_challenges = if log_cycles % 2 == 0 {
                [AkitaField::from_u64(0), AkitaField::from_u64(1)]
            } else {
                [
                    modulus_minus_one,
                    AkitaField::from_u64(0x0fed_cba9_8765_4321),
                ]
            };
            for (round, dense_challenge) in dense_challenges.into_iter().enumerate() {
                if bound_lt_lo.len() < 4 {
                    break;
                }
                dense_inc = bind(&dense_inc, dense_challenge);
                dense_wa = bind(&dense_wa, dense_challenge);
                dense_lt = bind(&dense_lt, dense_challenge);
                bound_lt_lo = bind(&bound_lt_lo, dense_challenge);
                let expected = registers_val_dense_message(&dense_inc, &dense_wa, &dense_lt);
                let actual = sequence
                    .bind_and_message(dense_challenge, &bound_lt_lo)
                    .expect("registers value dense transition should execute");
                assert_eq!(
                    actual, expected,
                    "log={log_cycles}, tg={threads_per_threadgroup}, dense_round={round}"
                );
                let expected_state = dense_inc
                    .iter()
                    .copied()
                    .zip(dense_wa.iter().copied())
                    .map(|(inc, wa)| [inc, wa])
                    .collect::<Vec<_>>();
                assert_eq!(
                    sequence
                        .read_current_dense_state()
                        .expect("registers value resident state should be readable"),
                    expected_state,
                    "log={log_cycles}, tg={threads_per_threadgroup}, dense_round={round}"
                );
            }
            if bound_lt_lo.len() == 2 {
                let exhausted_lt = bind(&bound_lt_lo, AkitaField::from_u64(2));
                assert!(matches!(
                    sequence.bind_and_message(AkitaField::from_u64(2), &exhausted_lt),
                    Err(MetalError::RegistersValSplitLtExhausted(2))
                ));
            }
        }
    }
}

fn registers_val_dense_message(
    inc: &[AkitaField],
    wa: &[AkitaField],
    lt: &[AkitaField],
) -> [AkitaField; 3] {
    let mut message = [AkitaField::from_u64(0); 3];
    for pair in 0..inc.len() / 2 {
        for (sample, t) in [0_u64, 2, 3].into_iter().enumerate() {
            let t = AkitaField::from_u64(t);
            let interpolate = |values: &[AkitaField]| {
                values[2 * pair] + t * (values[2 * pair + 1] - values[2 * pair])
            };
            message[sample] += interpolate(inc) * interpolate(wa) * interpolate(lt);
        }
    }
    message
}

#[test]
fn product5_sequence_reuses_resident_buffers_across_rounds() {
    let context = SolinasMetal::for_akita().expect("Akita Metal context should compile");
    let elements = 256;
    let tables = values(PRODUCT5_FACTORS * elements);
    let point = values(elements.trailing_zeros() as usize)
        .into_iter()
        .map(Fp128::into_jolt_field::<AkitaField>)
        .collect::<Vec<_>>();
    let mut gruen = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
    let mut sequence = context
        .prepare_product5_sequence(
            &tables
                .iter()
                .copied()
                .map(Fp128::into_jolt_field::<AkitaField>)
                .collect::<Vec<_>>(),
            elements,
            gruen.e_in_current(),
            gruen.e_out_current(),
            Product5SequenceConfig::default(),
        )
        .expect("product5 sequence should compile");
    let allocations = sequence.resident_buffer_count();

    let message = sequence
        .message(gruen.e_in_current(), gruen.e_out_current())
        .expect("initial sequence message should execute");
    let expected = product5_message(
        &tables,
        elements,
        &gruen
            .e_in_current()
            .iter()
            .map(Fp128::from_jolt_field)
            .collect::<Vec<_>>(),
        &gruen
            .e_out_current()
            .iter()
            .map(Fp128::from_jolt_field)
            .collect::<Vec<_>>(),
        AKITA_OFFSET_FFFFA7F7,
    );
    assert_eq!(
        message.map(|value| Fp128::from_jolt_field(&value)),
        expected
    );

    let challenge_1 = AkitaField::from_u64(17);
    gruen.bind(challenge_1);
    let message = sequence
        .bind_and_message(challenge_1, gruen.e_in_current(), gruen.e_out_current())
        .expect("first sequence transition should execute");
    let (bound, expected) = product5_fused_transition(
        &tables,
        elements,
        Fp128::from_jolt_field(&challenge_1),
        &gruen
            .e_in_current()
            .iter()
            .map(Fp128::from_jolt_field)
            .collect::<Vec<_>>(),
        &gruen
            .e_out_current()
            .iter()
            .map(Fp128::from_jolt_field)
            .collect::<Vec<_>>(),
        AKITA_OFFSET_FFFFA7F7,
    );
    assert_eq!(
        message.map(|value| Fp128::from_jolt_field(&value)),
        expected
    );

    let challenge_2 = AkitaField::from_u64(29);
    gruen.bind(challenge_2);
    let message = sequence
        .bind_and_message(challenge_2, gruen.e_in_current(), gruen.e_out_current())
        .expect("second sequence transition should execute");
    let (bound, expected) = product5_fused_transition(
        &bound,
        elements / 2,
        Fp128::from_jolt_field(&challenge_2),
        &gruen
            .e_in_current()
            .iter()
            .map(Fp128::from_jolt_field)
            .collect::<Vec<_>>(),
        &gruen
            .e_out_current()
            .iter()
            .map(Fp128::from_jolt_field)
            .collect::<Vec<_>>(),
        AKITA_OFFSET_FFFFA7F7,
    );
    assert_eq!(
        message.map(|value| Fp128::from_jolt_field(&value)),
        expected
    );

    let mut resident = vec![AkitaField::from_u64(0); bound.len()];
    sequence
        .read_current_tables(&mut resident)
        .expect("resident sequence state should read");
    assert_eq!(
        resident
            .iter()
            .map(Fp128::from_jolt_field)
            .collect::<Vec<_>>(),
        bound
    );
    assert_eq!(sequence.resident_buffer_count(), allocations);
    assert_eq!(sequence.round_device_buffer_allocations(), 0);
}

#[test]
fn gpu_field_probes_match_biguint() {
    let context = SolinasMetal::for_offset_275().expect("Metal context should compile");
    let (chain_lhs, chain_rhs) = inputs(4096);
    let (tail_lhs, tail_rhs) = inputs(4103);

    let noop = context
        .prepare_noop()
        .expect("noop pipeline should compile");
    noop.execute().expect("noop should execute");
    assert!(noop
        .read_output()
        .expect("noop output should read")
        .is_empty());

    for probe in [Probe::Copy, Probe::Add, Probe::Sub, Probe::MulWide] {
        assert_probe(&context, probe, &tail_lhs, &tail_rhs, 1, OFFSET_275);
    }

    for probe in [
        Probe::ChainWide1,
        Probe::ChainWide2,
        Probe::ChainWide4,
        Probe::ChainWide8,
    ] {
        assert_probe(&context, probe, &chain_lhs, &chain_rhs, 3, OFFSET_275);
    }
}

#[test]
fn runtime_specialization_supports_offset_edges() {
    assert!(matches!(
        SolinasMetal::new(0),
        Err(MetalError::InvalidOffset)
    ));

    for offset in [1, u32::MAX] {
        let context = SolinasMetal::new(offset).expect("specialized Metal context should compile");
        let (lhs, rhs) = inputs_for_offset(256, offset);
        for probe in [Probe::Add, Probe::Sub, Probe::MulWide] {
            assert_probe(&context, probe, &lhs, &rhs, 1, offset);
        }
    }
}

#[test]
fn rust_bindings_reject_invalid_dispatches() {
    let context = SolinasMetal::for_offset_275().expect("Metal context should compile");
    let valid = [Fp128::ONE];
    let noncanonical = [Fp128::from_u128(u128::MAX - OFFSET_275 as u128 + 1)];

    assert!(matches!(
        context.prepare(Probe::Add, &noncanonical, &valid, DispatchConfig::default()),
        Err(MetalError::NonCanonicalInput { side: "lhs", .. })
    ));
    assert!(matches!(
        context.prepare(Probe::Add, &valid, &[], DispatchConfig::default()),
        Err(MetalError::LengthMismatch { .. })
    ));
    assert!(matches!(
        context.prepare(Probe::Noop, &valid, &valid, DispatchConfig::default()),
        Err(MetalError::NoopPreparation)
    ));

    let invocation = context
        .prepare(Probe::Copy, &valid, &valid, DispatchConfig::default())
        .expect("copy pipeline should compile");
    assert!(matches!(
        invocation.read_output(),
        Err(MetalError::NotExecuted)
    ));
}

#[test]
fn raw_integer_probe_matches_wrapping_u32_arithmetic() {
    let context = SolinasMetal::for_offset_275().expect("Metal context should compile");
    let (lhs, rhs) = inputs(256);
    let invocation = context
        .prepare(
            Probe::U32MadIlp8,
            &lhs,
            &rhs,
            DispatchConfig {
                iterations: 7,
                threads_per_threadgroup: None,
            },
        )
        .expect("raw integer pipeline should compile");

    invocation
        .execute()
        .expect("raw integer probe should execute");
    let actual = invocation.read_output().expect("raw output should read");
    let expected = lhs
        .iter()
        .zip(&rhs)
        .map(|(&lhs, &rhs)| expected_u32_mad(lhs, rhs, 7))
        .collect::<Vec<_>>();
    assert_eq!(actual, expected);
}

#[test]
fn booleanity_sequence_matches_dense_cpu_at_every_round() {
    const LOG_T: usize = 10;
    const K: usize = 256;

    let rows = booleanity_rows(1 << LOG_T);
    let selectors = booleanity_selectors();
    assert_eq!(selectors.len(), 29);

    let gamma = AkitaField::from_u64(7);
    let mut gamma_power = AkitaField::from_u64(1);
    let mut rho = Vec::with_capacity(selectors.len());
    for _ in &selectors {
        rho.push(gamma_power);
        gamma_power *= gamma;
    }
    let eq_address = (0..K)
        .map(|index| AkitaField::from_u64((17 * index + 3) as u64))
        .collect::<Vec<_>>();
    let base_tables = rho
        .iter()
        .flat_map(|rho| eq_address.iter().map(|value| *rho * *value))
        .collect::<Vec<_>>();
    let point = (0..LOG_T)
        .map(|round| AkitaField::from_u64((19 * round + 11) as u64))
        .collect::<Vec<_>>();
    let mut eq = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
    let mut cpu_tables = dense_booleanity_tables(&rows, &selectors, &base_tables, K);

    let context = SolinasMetal::for_akita().expect("Akita Metal context should compile");
    let mut sequence = context
        .prepare_booleanity_sequence(
            &rows,
            &selectors,
            &base_tables,
            &rho,
            K,
            eq.e_in_current().len(),
            eq.e_out_current().len(),
            BooleanitySequenceConfig::default(),
        )
        .expect("booleanity pipelines should compile");
    assert_eq!(sequence.round_device_buffer_allocations(), 0);

    let expected =
        dense_booleanity_message(&cpu_tables, &rho, eq.e_in_current(), eq.e_out_current());
    let actual = sequence
        .message(eq.e_in_current(), eq.e_out_current())
        .expect("initial Booleanity message should execute");
    assert_eq!(actual, expected, "round 0");

    for round in 1..LOG_T {
        let challenge = AkitaField::from_u64((31 * round + 5) as u64);
        bind_dense_tables(&mut cpu_tables, challenge);
        eq.bind(challenge);
        let expected =
            dense_booleanity_message(&cpu_tables, &rho, eq.e_in_current(), eq.e_out_current());
        let actual = sequence
            .bind_and_message(challenge, eq.e_in_current(), eq.e_out_current())
            .expect("bound Booleanity message should execute");
        assert_eq!(actual, expected, "round {round}");

        if sequence.is_dense() {
            let mut resident = vec![AkitaField::from_u64(0); selectors.len() * cpu_tables[0].len()];
            sequence
                .read_current_tables(&mut resident)
                .expect("resident dense tables should read");
            assert_eq!(
                resident,
                flatten_tables(&cpu_tables),
                "round {round} tables"
            );
        }
    }

    assert!(sequence.is_dense());
    assert_eq!(sequence.current_elements(), 2);
    let final_challenge = AkitaField::from_u64(101);
    bind_dense_tables(&mut cpu_tables, final_challenge);
    assert_eq!(
        cpu_tables.iter().map(Vec::len).collect::<Vec<_>>(),
        vec![1; 29]
    );
}

fn booleanity_selectors() -> Vec<BooleanitySelector> {
    let mut selectors = (0..16)
        .map(|index| BooleanitySelector::Lookup { shift: 8 * index })
        .collect::<Vec<_>>();
    selectors.push(BooleanitySelector::Bytecode { shift: 0 });
    selectors.extend([0, 8, 56].map(|shift| BooleanitySelector::Ram { shift }));
    selectors.extend((0..8).map(|index| BooleanitySelector::FusedInc { shift: 8 * index }));
    selectors.push(BooleanitySelector::FusedIncMsb);
    selectors
}

fn booleanity_rows(rows: usize) -> Vec<BooleanityRow> {
    let mut state = 0x0123_4567_89ab_cdef_fedc_ba98_7654_3210u128;
    (0..rows)
        .map(|row| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 43;
            let mapped_pc = (row % 7 != 0).then_some(((state >> 49) as u64) & ((1 << 13) - 2));
            let ram_address = (row % 11 != 0).then_some((state as u64) & (u64::from(u32::MAX) - 1));
            let fused_inc = match row % 6 {
                0 => -(u64::MAX as i128),
                1 => -((1i128 << 63) + row as i128),
                2 => u64::MAX as i128 - row as i128,
                3 => (1i128 << 63) + row as i128,
                4 => row as i128,
                _ => -(row as i128),
            };
            BooleanityRow::new(state, mapped_pc, ram_address, fused_inc)
                .expect("synthetic row should fit the Metal ABI")
        })
        .collect()
}

fn dense_booleanity_tables(
    rows: &[BooleanityRow],
    selectors: &[BooleanitySelector],
    base_tables: &[AkitaField],
    k: usize,
) -> Vec<Vec<AkitaField>> {
    selectors
        .iter()
        .copied()
        .enumerate()
        .map(|(poly, selector)| {
            rows.iter()
                .copied()
                .map(|row| {
                    booleanity_hot_index(row, selector, k).map_or_else(
                        || AkitaField::from_u64(0),
                        |hot| base_tables[poly * k + hot],
                    )
                })
                .collect()
        })
        .collect()
}

fn booleanity_hot_index(
    row: BooleanityRow,
    selector: BooleanitySelector,
    k: usize,
) -> Option<usize> {
    let words = row.words();
    let width = k.ilog2() as usize;
    let mask = k - 1;
    match selector {
        BooleanitySelector::Lookup { shift } => {
            let lookup = u128::from(words[0]) | (u128::from(words[1]) << 64);
            Some(((lookup >> shift) as usize) & mask)
        }
        BooleanitySelector::Bytecode { shift } => {
            let pc_plus_one = words[4] & ((1 << 56) - 1);
            pc_plus_one
                .checked_sub(1)
                .map(|pc| ((pc >> shift) as usize) & mask)
        }
        BooleanitySelector::Ram { shift } => words[2]
            .checked_sub(1)
            .map(|address| ((address >> shift) as usize) & mask),
        BooleanitySelector::FusedInc { shift } => {
            let biased = biased_fused_inc(words, width);
            let standard = ((biased >> shift) as usize) & mask;
            Some((standard + k / 2) & mask)
        }
        BooleanitySelector::FusedIncMsb => {
            let carry = biased_fused_inc(words, width) >> 64;
            Some(carry.rem_euclid(k as i128) as usize)
        }
    }
}

fn biased_fused_inc(words: [u64; 5], width: usize) -> i128 {
    let magnitude = i128::from(words[3]);
    let value = if words[4] >> 63 == 0 {
        magnitude
    } else {
        -magnitude
    };
    let radix = 1i128 << width;
    let bias = (radix / 2) * (((1i128 << 64) - 1) / (radix - 1));
    value + bias
}

fn dense_booleanity_message(
    tables: &[Vec<AkitaField>],
    rho: &[AkitaField],
    e_in: &[AkitaField],
    e_out: &[AkitaField],
) -> [AkitaField; 2] {
    let pairs = tables[0].len() / 2;
    assert_eq!(pairs, e_in.len() * e_out.len());
    let mut message = [AkitaField::from_u64(0); 2];
    for (x_out, outer_weight) in e_out.iter().copied().enumerate() {
        for (x_in, inner_weight) in e_in.iter().copied().enumerate() {
            let pair = x_out * e_in.len() + x_in;
            let mut relation = [AkitaField::from_u64(0); 2];
            for (table, rho) in tables.iter().zip(rho) {
                let h_0 = table[2 * pair];
                let h_1 = table[2 * pair + 1];
                let delta = h_1 - h_0;
                relation[0] += h_0 * (h_0 - *rho);
                relation[1] += delta * delta;
            }
            let weight = outer_weight * inner_weight;
            message[0] += weight * relation[0];
            message[1] += weight * relation[1];
        }
    }
    message
}

fn bind_dense_tables(tables: &mut [Vec<AkitaField>], challenge: AkitaField) {
    for table in tables {
        let bound_len = table.len() / 2;
        for index in 0..bound_len {
            let lo = table[2 * index];
            let hi = table[2 * index + 1];
            table[index] = lo + challenge * (hi - lo);
        }
        table.truncate(bound_len);
    }
}

fn flatten_tables(tables: &[Vec<AkitaField>]) -> Vec<AkitaField> {
    tables.iter().flatten().copied().collect()
}

fn assert_probe(
    context: &SolinasMetal,
    probe: Probe,
    lhs: &[Fp128],
    rhs: &[Fp128],
    iterations: u32,
    offset: u32,
) {
    let invocation = context
        .prepare(
            probe,
            lhs,
            rhs,
            DispatchConfig {
                iterations,
                threads_per_threadgroup: None,
            },
        )
        .expect("pipeline should compile");
    let limits = invocation.pipeline_limits();
    assert!(limits.thread_execution_width > 0);
    assert!(invocation.threads_per_threadgroup() <= limits.max_total_threads_per_threadgroup);

    invocation.execute().expect("probe should execute");
    let actual = invocation.read_output().expect("output should read");
    let expected = lhs
        .iter()
        .zip(rhs)
        .map(|(&lhs, &rhs)| expected_field_for_offset(probe, lhs, rhs, iterations, offset))
        .collect::<Result<Vec<_>, _>>()
        .expect("probe should have a field oracle");
    assert_eq!(actual, expected, "{}", probe.name());
}
