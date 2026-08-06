#![cfg(all(feature = "metal", target_os = "macos"))]
#![expect(clippy::expect_used, reason = "integration test")]

use jolt_field::{AkitaField, FromPrimitiveInt};
use jolt_kernels::metal::solinas::{
    BooleanityRow, BooleanitySelector, BooleanitySequenceConfig, DispatchConfig, Fp128,
    InstructionRaFirstMessageConfig, MetalError, Probe, Product5Config, Product5SequenceConfig,
    RegisterAccessRow, RegistersReadWriteMessageConfig, SolinasMetal, AKITA_OFFSET_FFFFA7F7,
    OFFSET_275,
};
use jolt_poly::{BindingOrder, GruenSplitEqPolynomial};

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
fn instruction_ra_first_message_matches_cpu() {
    const LOG_T: usize = 12;
    const ROWS: usize = 1 << LOG_T;
    const FACTORS: usize = 16;
    const BINS: usize = 256;

    let mut cycle_lookups = (0..ROWS)
        .map(|row| {
            let lo = (row as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15);
            let hi = (!(row as u64)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            u128::from(lo) | (u128::from(hi) << 64)
        })
        .collect::<Vec<_>>();
    cycle_lookups[0] = 0x0001_0203_0405_0607_0809_0a0b_0c0d_0e0f;
    cycle_lookups[1] = 0xf0e1_d2c3_b4a5_9687_7869_5a4b_3c2d_1e0f;
    cycle_lookups[2] = 0xff00_aa55_cc33_9966_1234_5678_9abc_def0;

    let mut table_major_lookups = vec![0u128; ROWS];
    let mut cycle_to_table_major = vec![0u32; ROWS];
    for (cycle, &lookup) in cycle_lookups.iter().enumerate() {
        let slot = (5 * cycle + 3) & (ROWS - 1);
        table_major_lookups[slot] = lookup;
        cycle_to_table_major[cycle] = slot as u32;
    }
    let chunk_tables = (0..FACTORS)
        .flat_map(|factor| {
            (0..BINS).map(move |bin| AkitaField::from_u64((2 + 17 * factor + 31 * bin) as u64))
        })
        .collect::<Vec<_>>();
    let point = (0..LOG_T)
        .map(|round| AkitaField::from_u64((1009 + 37 * round) as u64))
        .collect::<Vec<_>>();
    let gruen = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
    let expected = instruction_ra_message_cpu(
        &cycle_lookups,
        &chunk_tables,
        gruen.e_in_current(),
        gruen.e_out_current(),
    );

    let context = SolinasMetal::for_akita().expect("Akita Metal context should compile");
    let invocation = context
        .prepare_instruction_ra_first_message(
            &table_major_lookups,
            &cycle_to_table_major,
            &chunk_tables,
            gruen.e_in_current(),
            gruen.e_out_current(),
            InstructionRaFirstMessageConfig::default(),
        )
        .expect("Instruction RA pipelines should compile");
    assert_eq!(invocation.threads_per_threadgroup(), 128);
    assert_eq!(invocation.dynamic_threadgroup_memory_bytes(), 256);
    assert_eq!(
        invocation.useful_multiplications(),
        44 * (ROWS / 2) as u64 + 4 * gruen.e_out_current().len() as u64
    );
    assert_eq!(invocation.logical_lookup_plane_bytes(), 20 * ROWS as u64);
    assert_eq!(invocation.logical_branch_bytes(), 256 * ROWS as u64);
    assert_eq!(
        invocation.logical_weight_bytes(),
        8 * ROWS as u64 + 16 * gruen.e_out_current().len() as u64
    );
    assert!(matches!(
        invocation.read_message(),
        Err(MetalError::NotExecuted)
    ));

    invocation
        .execute()
        .expect("Instruction RA first message should execute");
    assert_eq!(
        invocation
            .read_message()
            .expect("Instruction RA message should read"),
        expected
    );
}

fn registers_rw_cell(
    row: RegisterAccessRow,
    column: u8,
    gamma: AkitaField,
) -> Option<(AkitaField, u64, AkitaField, bool, bool)> {
    let rs1 = row.rs1().filter(|(index, _)| *index == column);
    let rs2 = row.rs2().filter(|(index, _)| *index == column);
    let rd = row.rd().filter(|(index, ..)| *index == column);
    if rs1.is_none() && rs2.is_none() && rd.is_none() {
        return None;
    }
    let value = rs1
        .map(|(_, value)| value)
        .or_else(|| rs2.map(|(_, value)| value))
        .unwrap_or_else(|| rd.expect("present cell has an access").1);
    let mut ra = AkitaField::from_u64(0);
    if rs1.is_some() {
        ra += gamma;
    }
    if rs2.is_some() {
        ra += gamma * gamma;
    }
    Some((
        AkitaField::from_u64(value),
        rd.map_or(value, |(_, _, post)| post),
        ra,
        rs1.is_some() || rs2.is_some(),
        rd.is_some(),
    ))
}

fn registers_rw_first_message_cpu(
    rows: &[RegisterAccessRow],
    inc: &[AkitaField],
    gamma: AkitaField,
    e_in: &[AkitaField],
    e_out: &[AkitaField],
) -> [AkitaField; 2] {
    let mut result = [AkitaField::from_u64(0); 2];
    for (x_out, &outer_weight) in e_out.iter().enumerate() {
        for (x_in, &inner_weight) in e_in.iter().enumerate() {
            let pair = x_out * e_in.len() + x_in;
            let lo = rows[2 * pair];
            let hi = rows[2 * pair + 1];
            let inc_zero = inc[2 * pair];
            let inc_infinity = inc[2 * pair + 1] - inc_zero;
            let mut q = [AkitaField::from_u64(0); 2];
            for column in 0..128u8 {
                let even = registers_rw_cell(lo, column, gamma);
                let odd = registers_rw_cell(hi, column, gamma);
                if even.is_none() && odd.is_none() {
                    continue;
                }
                if let Some((value, _, ra, read, write)) = even {
                    if read {
                        q[0] += ra * value;
                    }
                    if write {
                        q[0] += value + inc_zero;
                    }
                }
                let value_infinity = match (even, odd) {
                    (Some((even_value, ..)), Some((odd_value, ..))) => odd_value - even_value,
                    (Some((even_value, next, ..)), None) => AkitaField::from_u64(next) - even_value,
                    (None, Some(_)) => AkitaField::from_u64(0),
                    (None, None) => unreachable!("untouched column was skipped"),
                };
                let even_ra = even.map_or(AkitaField::from_u64(0), |cell| cell.2);
                let odd_ra = odd.map_or(AkitaField::from_u64(0), |cell| cell.2);
                if even.is_some_and(|cell| cell.3) || odd.is_some_and(|cell| cell.3) {
                    q[1] += (odd_ra - even_ra) * value_infinity;
                }
                let even_write = even.is_some_and(|cell| cell.4);
                let odd_write = odd.is_some_and(|cell| cell.4);
                let write_term = value_infinity + inc_infinity;
                if odd_write && !even_write {
                    q[1] += write_term;
                } else if even_write && !odd_write {
                    q[1] -= write_term;
                }
            }
            result[0] += outer_weight * inner_weight * q[0];
            result[1] += outer_weight * inner_weight * q[1];
        }
    }
    result
}

fn registers_rw_second_message_cpu(
    rows: &[RegisterAccessRow],
    inc: &[AkitaField],
    gamma: AkitaField,
    first_challenge: AkitaField,
    e_in: &[AkitaField],
    e_out: &[AkitaField],
) -> [AkitaField; 2] {
    const REGISTERS: usize = 128;
    let mut state = [0u64; REGISTERS];
    let mut val = vec![AkitaField::from_u64(0); rows.len() * REGISTERS];
    let mut ra = vec![AkitaField::from_u64(0); rows.len() * REGISTERS];
    let mut wa = vec![AkitaField::from_u64(0); rows.len() * REGISTERS];
    for (cycle, row) in rows.iter().copied().enumerate() {
        for register in 0..REGISTERS {
            val[cycle * REGISTERS + register] = AkitaField::from_u64(state[register]);
        }
        if let Some((register, _)) = row.rs1() {
            ra[cycle * REGISTERS + register as usize] += gamma;
        }
        if let Some((register, _)) = row.rs2() {
            ra[cycle * REGISTERS + register as usize] += gamma * gamma;
        }
        if let Some((register, _, post)) = row.rd() {
            wa[cycle * REGISTERS + register as usize] = AkitaField::from_u64(1);
            state[register as usize] = post;
        }
    }

    let blocks = rows.len() / 2;
    let mut bound_val = vec![AkitaField::from_u64(0); blocks * REGISTERS];
    let mut bound_ra = vec![AkitaField::from_u64(0); blocks * REGISTERS];
    let mut bound_wa = vec![AkitaField::from_u64(0); blocks * REGISTERS];
    for block in 0..blocks {
        for register in 0..REGISTERS {
            let bind = |table: &[AkitaField]| {
                let lo = table[(2 * block) * REGISTERS + register];
                let hi = table[(2 * block + 1) * REGISTERS + register];
                lo + first_challenge * (hi - lo)
            };
            bound_val[block * REGISTERS + register] = bind(&val);
            bound_ra[block * REGISTERS + register] = bind(&ra);
            bound_wa[block * REGISTERS + register] = bind(&wa);
        }
    }
    let bound_inc = inc
        .chunks_exact(2)
        .map(|pair| pair[0] + first_challenge * (pair[1] - pair[0]))
        .collect::<Vec<_>>();

    let mut result = [AkitaField::from_u64(0); 2];
    for (x_out, &outer_weight) in e_out.iter().enumerate() {
        for (x_in, &inner_weight) in e_in.iter().enumerate() {
            let pair = x_out * e_in.len() + x_in;
            let lo_block = 2 * pair;
            let hi_block = lo_block + 1;
            let inc_zero = bound_inc[lo_block];
            let inc_infinity = bound_inc[hi_block] - inc_zero;
            let mut q = [AkitaField::from_u64(0); 2];
            for register in 0..REGISTERS {
                let lo = lo_block * REGISTERS + register;
                let hi = hi_block * REGISTERS + register;
                q[0] += bound_ra[lo] * bound_val[lo] + bound_wa[lo] * (bound_val[lo] + inc_zero);
                let val_infinity = bound_val[hi] - bound_val[lo];
                q[1] += (bound_ra[hi] - bound_ra[lo]) * val_infinity
                    + (bound_wa[hi] - bound_wa[lo]) * (val_infinity + inc_infinity);
            }
            result[0] += outer_weight * inner_weight * q[0];
            result[1] += outer_weight * inner_weight * q[1];
        }
    }
    result
}

#[test]
fn registers_read_write_first_message_matches_cpu() {
    const LOG_T: usize = 11;
    const ROWS: usize = 1 << LOG_T;

    let mut state = [0u64; 128];
    let rows = (0..ROWS)
        .map(|cycle| {
            let rs1_index = ((13 * cycle) & 127) as u8;
            let rs2_index = if cycle % 7 == 0 {
                rs1_index
            } else {
                ((29 * cycle + 5) & 127) as u8
            };
            let rd_index = if cycle % 5 == 0 {
                rs2_index
            } else {
                ((47 * cycle + 11) & 127) as u8
            };
            let rs1 = (cycle % 9 != 0).then_some((rs1_index, state[rs1_index as usize]));
            let rs2 = (cycle % 6 != 0).then_some((rs2_index, state[rs2_index as usize]));
            let rd = if cycle % 4 == 0 {
                None
            } else {
                let pre = state[rd_index as usize];
                let post = pre.wrapping_add((cycle as u64) | 1);
                state[rd_index as usize] = post;
                Some((rd_index, pre, post))
            };
            RegisterAccessRow::new(rs1, rs2, rd)
        })
        .collect::<Vec<_>>();
    let inc = (0..ROWS)
        .map(|row| AkitaField::from_u64((17 * row + 3) as u64))
        .collect::<Vec<_>>();
    let gamma = AkitaField::from_u64(0x1234_5678_9abc_def0);
    let point = (0..LOG_T)
        .map(|round| AkitaField::from_u64((1009 + 37 * round) as u64))
        .collect::<Vec<_>>();
    let mut gruen = GruenSplitEqPolynomial::new(&point, BindingOrder::LowToHigh);
    let expected = registers_rw_first_message_cpu(
        &rows,
        &inc,
        gamma,
        gruen.e_in_current(),
        gruen.e_out_current(),
    );

    let context = SolinasMetal::for_akita().expect("Akita Metal context should compile");
    let invocation = context
        .prepare_registers_read_write_first_message(
            &rows,
            &inc,
            gamma,
            gruen.e_in_current(),
            gruen.e_out_current(),
            RegistersReadWriteMessageConfig::default(),
        )
        .expect("registers read/write pipelines should compile");
    assert_eq!(invocation.rows(), ROWS);
    assert_eq!(invocation.threads_per_threadgroup(), 128);
    assert_eq!(invocation.dynamic_threadgroup_memory_bytes(), 128);
    assert_eq!(invocation.logical_row_bytes(), 40 * ROWS as u64);
    assert_eq!(invocation.logical_inc_bytes(), 16 * ROWS as u64);
    assert_eq!(invocation.execute_device_buffer_allocations(), 0);
    assert!(matches!(
        invocation.read_message(),
        Err(MetalError::NotExecuted)
    ));

    invocation
        .execute()
        .expect("registers read/write first message should execute");
    assert_eq!(
        invocation
            .read_message()
            .expect("registers read/write message should read"),
        expected
    );

    let first_challenge = AkitaField::from_u64(0xfeed_beef_cafe_babe);
    gruen.bind(first_challenge);
    let second_expected = registers_rw_second_message_cpu(
        &rows,
        &inc,
        gamma,
        first_challenge,
        gruen.e_in_current(),
        gruen.e_out_current(),
    );
    let second = invocation
        .prepare_second_message(
            gruen.e_in_current(),
            gruen.e_out_current(),
            RegistersReadWriteMessageConfig::default(),
        )
        .expect("second registers read/write pipeline should compile");
    assert_eq!(second.rows(), ROWS);
    assert_eq!(second.threads_per_threadgroup(), 128);
    assert_eq!(second.dynamic_threadgroup_memory_bytes(), 128);
    assert_eq!(second.execute_device_buffer_allocations(), 0);
    assert!(matches!(
        second.read_message(),
        Err(MetalError::NotExecuted)
    ));
    second
        .execute(first_challenge)
        .expect("second registers read/write message should execute");
    assert_eq!(
        second
            .read_message()
            .expect("second registers read/write message should read"),
        second_expected
    );
}

#[test]
fn product5_message_matches_biguint() {
    let context = SolinasMetal::for_offset_275().expect("Metal context should compile");
    let elements = 256;
    let tables = values(PRODUCT5_FACTORS * elements);
    let (e_in, _) = inputs(8);
    let (e_out, _) = inputs(elements / 2 / e_in.len());
    let invocation = context
        .prepare_product5_message(&tables, elements, &e_in, &e_out, Product5Config::default())
        .expect("product5 message pipeline should compile");
    assert_eq!(invocation.threads_per_threadgroup(), 128);
    assert_eq!(invocation.dynamic_threadgroup_memory_bytes(), 320);
    assert_eq!(invocation.useful_multiplications(), 11 * 256 + 5 * 16);
    assert_eq!(invocation.logical_factor_bytes(), 80 * 256);

    invocation
        .execute()
        .expect("product5 message should execute");
    assert_eq!(
        invocation
            .read_message()
            .expect("product5 message should read"),
        product5_message(&tables, elements, &e_in, &e_out, OFFSET_275)
    );
    assert!(invocation
        .read_bound_tables()
        .expect("message-only bound output should read")
        .is_none());
}

#[test]
fn product5_fused_transition_matches_biguint() {
    let context = SolinasMetal::for_offset_275().expect("Metal context should compile");
    let elements = 256;
    let tables = values(PRODUCT5_FACTORS * elements);
    let (e_in, _) = inputs(8);
    let (e_out, _) = inputs(elements / 4 / e_in.len());
    let challenge = Fp128::from_u128(0x1234_5678_9abc_def0);
    let (expected_bound, expected_message) =
        product5_fused_transition(&tables, elements, challenge, &e_in, &e_out, OFFSET_275);
    let invocation = context
        .prepare_product5_fused_transition(
            &tables,
            elements,
            challenge,
            &e_in,
            &e_out,
            Product5Config::default(),
        )
        .expect("product5 transition pipeline should compile");
    assert_eq!(invocation.threads_per_threadgroup(), 64);
    assert_eq!(invocation.dynamic_threadgroup_memory_bytes(), 160);
    assert_eq!(invocation.useful_multiplications(), 8 * 256 + 5 * 8);
    assert_eq!(invocation.logical_factor_bytes(), 120 * 256);

    invocation
        .execute()
        .expect("product5 transition should execute");
    assert_eq!(
        invocation
            .read_message()
            .expect("product5 message should read"),
        expected_message
    );
    assert_eq!(
        invocation
            .read_bound_tables()
            .expect("product5 bound output should read")
            .expect("fused transition should have bound output"),
        expected_bound
    );
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

    let tables = values(PRODUCT5_FACTORS * 8);
    let e_in = [Fp128::ONE];
    let e_out = [Fp128::ONE; 4];
    assert!(matches!(
        context.prepare_product5_message(&tables, 6, &e_in, &e_out, Product5Config::default()),
        Err(MetalError::InvalidProduct5TableLength { .. })
    ));
    assert!(matches!(
        context.prepare_product5_message(
            &tables[..tables.len() - 1],
            8,
            &e_in,
            &e_out,
            Product5Config::default()
        ),
        Err(MetalError::Product5StorageLength { .. })
    ));
    assert!(matches!(
        context.prepare_product5_message(&tables, 8, &e_in, &e_out[..3], Product5Config::default()),
        Err(MetalError::Product5WeightShape { .. })
    ));
    let product5 = context
        .prepare_product5_message(&tables, 8, &e_in, &e_out, Product5Config::default())
        .expect("product5 pipeline should compile");
    assert!(matches!(
        product5.read_message(),
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
            let mapped_pc = (row % 7 != 0).then_some(((state >> 49) as u64) & ((1 << 55) - 2));
            let ram_address = (row % 11 != 0).then_some((state as u64) & (u64::MAX - 1));
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

fn instruction_ra_message_cpu(
    lookups: &[u128],
    chunk_tables: &[AkitaField],
    e_in: &[AkitaField],
    e_out: &[AkitaField],
) -> [AkitaField; 4] {
    const GROUPS: usize = 4;
    const FACTORS_PER_GROUP: usize = 4;
    const FACTORS: usize = GROUPS * FACTORS_PER_GROUP;
    const BINS: usize = 256;

    assert_eq!(lookups.len() / 2, e_in.len() * e_out.len());
    assert_eq!(chunk_tables.len(), FACTORS * BINS);
    let zero = AkitaField::from_u64(0);
    let mut output = [zero; 4];
    for (x_out, &outer_weight) in e_out.iter().enumerate() {
        let mut outer = [zero; 4];
        for (x_in, &inner_weight) in e_in.iter().enumerate() {
            let pair = x_out * e_in.len() + x_in;
            let mut groups = [zero; 4];
            for group in 0..GROUPS {
                let mut finite = [AkitaField::from_u64(1); 3];
                let mut infinity = AkitaField::from_u64(1);
                for offset in 0..FACTORS_PER_GROUP {
                    let factor = group * FACTORS_PER_GROUP + offset;
                    let shift = 8 * (FACTORS - 1 - factor);
                    let lo_index = ((lookups[2 * pair] >> shift) & 0xff) as usize;
                    let hi_index = ((lookups[2 * pair + 1] >> shift) & 0xff) as usize;
                    let lo = chunk_tables[factor * BINS + lo_index];
                    let hi = chunk_tables[factor * BINS + hi_index];
                    let step = hi - lo;
                    for (sample, value) in finite.iter_mut().enumerate() {
                        *value *= lo + AkitaField::from_u64((sample + 1) as u64) * step;
                    }
                    infinity *= step;
                }
                for sample in 0..3 {
                    groups[sample] += finite[sample];
                }
                groups[3] += infinity;
            }
            for (outer, group) in outer.iter_mut().zip(groups) {
                *outer += inner_weight * group;
            }
        }
        for (output, outer) in output.iter_mut().zip(outer) {
            *output += outer_weight * outer;
        }
    }
    output
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
