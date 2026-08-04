#![cfg(all(feature = "metal", target_os = "macos"))]
#![expect(clippy::expect_used, reason = "integration test")]

use jolt_field::{AkitaField, FromPrimitiveInt};
use jolt_kernels::metal::solinas::{
    DispatchConfig, Fp128, MetalError, Probe, Product5Config, Product5SequenceConfig, SolinasMetal,
    AKITA_OFFSET_FFFFA7F7, OFFSET_275,
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
