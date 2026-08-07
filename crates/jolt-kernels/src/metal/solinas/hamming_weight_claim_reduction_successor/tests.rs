use core::mem::{align_of, offset_of, size_of};

use jolt_field::AkitaField;
use jolt_poly::EqPolynomial;

use super::super::{BooleanityRow, SolinasMetal};
use super::*;

#[test]
fn abi_geometry_and_dispatch_are_exact() {
    assert_eq!(size_of::<HammingWeightRetainedParams>(), 32);
    assert_eq!(align_of::<HammingWeightRetainedParams>(), 4);
    assert_eq!(offset_of!(HammingWeightRetainedParams, rows), 0);
    assert_eq!(offset_of!(HammingWeightRetainedParams, e_in_length), 4);
    assert_eq!(offset_of!(HammingWeightRetainedParams, e_out_length), 8);
    assert_eq!(offset_of!(HammingWeightRetainedParams, selector_offset), 12);
    assert_eq!(
        offset_of!(HammingWeightRetainedParams, selectors_in_tile),
        16
    );
    assert_eq!(offset_of!(HammingWeightRetainedParams, bins), 20);
    assert_eq!(offset_of!(HammingWeightRetainedParams, reserved), 24);

    let geometry =
        HammingWeightRetainedGeometry::new(1 << 26, HammingWeightRetainedConfig::default())
            .unwrap();
    let lengths = geometry.buffer_lengths().unwrap();
    assert_eq!(lengths.hot_bytes, 1_946_157_056);
    assert_eq!(lengths.e_in_fields, 32_768);
    assert_eq!(lengths.e_out_fields, 2_048);
    assert_eq!(lengths.partial_fields, 3_145_728);
    assert_eq!(lengths.output_fields, 7_424);
    assert_eq!(lengths.owned_bytes().unwrap(), 51_007_488);

    let offsets = (0..HAMMING_RETAINED_TILES)
        .map(|tile| geometry.params(tile).unwrap().selector_offset)
        .collect::<Vec<_>>();
    assert_eq!(offsets, [0, 6, 12, 18, 24]);
    let dispatch = geometry.dispatch_plan();
    assert_eq!(dispatch.command_buffers, 1);
    assert_eq!(dispatch.encoders, 10);
    assert_eq!(dispatch.dispatches, 10);
    assert_eq!(dispatch.completion_waits, 1);
    assert_eq!(dispatch.readbacks, 1);
    assert_eq!(dispatch.tile_threadgroups, 10_240);
    assert_eq!(dispatch.finalize_threadgroups, 29);
}

#[test]
fn fixed_geometry_rejects_retuning() {
    for config in [
        HammingWeightRetainedConfig {
            inner_log2: 14,
            ..Default::default()
        },
        HammingWeightRetainedConfig {
            accumulator_threads_per_threadgroup: 256,
            ..Default::default()
        },
        HammingWeightRetainedConfig {
            finalize_threads_per_threadgroup: 512,
            ..Default::default()
        },
    ] {
        assert!(HammingWeightRetainedGeometry::new(1 << 15, config).is_err());
    }
    assert!(HammingWeightRetainedGeometry::new(
        (1 << 15) - 1,
        HammingWeightRetainedConfig::default()
    )
    .is_err());
}

#[test]
fn retained_projection_matches_scalar_hamming_oracle() {
    let rows = fixture_rows(1 << 15);
    let point = point(1 << 15);
    let weights = EqPolynomial::evals(&point, None);
    let expected = recentered_oracle(&rows, &weights);
    let context = SolinasMetal::for_akita().unwrap();
    let resident = context.prepare_booleanity_rows(&rows).unwrap();
    let source_rows_storage_id = resident.allocation_identity();
    let producer = context
        .prepare_booleanity_address_successor(
            resident,
            &point,
            super::super::BooleanityAddressSuccessorConfig::default(),
        )
        .unwrap();
    let _producer_gpu_active = producer.execute_timed().unwrap();
    let hot_rows = producer.completed_hot_rows().unwrap();
    let hot_rows_storage_id = hot_rows.allocation_identity();

    let invocation = context
        .prepare_hamming_weight_retained(hot_rows, &point, HammingWeightRetainedConfig::default())
        .unwrap();
    assert_eq!(invocation.source_rows_storage_id(), source_rows_storage_id);
    assert_eq!(invocation.hot_rows_storage_id(), hot_rows_storage_id);
    assert!(matches!(
        invocation.read_masses(),
        Err(HammingWeightRetainedRuntimeError::InvalidState(_))
    ));
    assert!(invocation.tile_pipeline_limits().iter().all(|limits| {
        limits.thread_execution_width == HAMMING_RETAINED_SIMD_WIDTH
            && limits.max_total_threads_per_threadgroup >= HAMMING_RETAINED_ACCUMULATOR_THREADS
    }));
    assert_eq!(
        invocation.finalize_pipeline_limits().thread_execution_width,
        HAMMING_RETAINED_SIMD_WIDTH
    );

    let _consumer_gpu_active = invocation.execute_timed().unwrap();
    let actual = invocation.read_masses().unwrap();
    assert_eq!(actual, expected);
    for selector in 0..HAMMING_RETAINED_SELECTORS {
        assert_eq!(actual[selector * HAMMING_RETAINED_BINS], AkitaField::zero());
    }
}

fn point(rows: usize) -> Vec<AkitaField> {
    (0..rows.ilog2())
        .map(|index| AkitaField::from_u64(0x1234 + 37 * u64::from(index)))
        .collect()
}

fn fixture_rows(rows: usize) -> Vec<BooleanityRow> {
    (0..rows)
        .map(|index| {
            let lo = (index as u64).wrapping_mul(0x0102_0304_0506_0708);
            let hi = (!(index as u64)).rotate_left(17);
            let lookup = u128::from(lo) | (u128::from(hi) << 64);
            let mapped_pc = match index % 5 {
                0 => None,
                1 => Some(0),
                _ => Some(((index * 7) & 0xffff) as u64),
            };
            let ram = match index % 3 {
                0 => None,
                1 => Some(0),
                _ => Some(((index * 11) & 0xffff) as u64),
            };
            let magnitude = (index as u64).wrapping_mul(0x1_0001) as i128;
            let fused_inc = match index % 7 {
                0 => 0,
                1 => magnitude,
                2 => -magnitude,
                3 => u64::MAX as i128,
                4 => -(u64::MAX as i128),
                5 => (1u64 << 63) as i128,
                _ => -((1u64 << 63) as i128),
            };
            BooleanityRow::new(lookup, mapped_pc, ram, fused_inc).unwrap()
        })
        .collect()
}

fn recentered_oracle(rows: &[BooleanityRow], weights: &[AkitaField]) -> Vec<AkitaField> {
    let mut output = vec![AkitaField::zero(); HAMMING_RETAINED_SELECTORS * HAMMING_RETAINED_BINS];
    for (row, weight) in rows.iter().copied().zip(weights.iter().copied()) {
        for selector in 0..HAMMING_RETAINED_SELECTORS {
            if let Some(hot) = hot(row, selector).filter(|hot| *hot != 0) {
                output[selector * HAMMING_RETAINED_BINS + hot] += weight;
            }
        }
    }
    output
}

fn hot(row: BooleanityRow, selector: usize) -> Option<usize> {
    const PC_MASK: u64 = 0x00ff_ffff_ffff_ffff;
    const INC_BIAS: u64 = 0x8080_8080_8080_8080;
    let [lookup_lo, lookup_hi, ram_plus_one, magnitude, packed_pc] = row.words();
    match selector {
        0..=7 => Some(((lookup_hi >> (8 * (7 - selector))) & 0xff) as usize),
        8..=15 => Some(((lookup_lo >> (8 * (15 - selector))) & 0xff) as usize),
        16..=17 => {
            let plus_one = packed_pc & PC_MASK;
            (plus_one != 0).then(|| (((plus_one - 1) >> (8 * (17 - selector))) & 0xff) as usize)
        }
        18..=19 => (ram_plus_one != 0)
            .then(|| (((ram_plus_one - 1) >> (8 * (19 - selector))) & 0xff) as usize),
        20..=27 => {
            let biased = if packed_pc >> 63 != 0 {
                INC_BIAS.wrapping_sub(magnitude)
            } else {
                INC_BIAS.wrapping_add(magnitude)
            };
            let byte = ((biased >> (8 * (selector - 20))) & 0xff) as u8;
            Some(byte.wrapping_add(128) as usize)
        }
        28 => {
            let carry: i8 = if packed_pc >> 63 != 0 {
                if magnitude > INC_BIAS {
                    -1
                } else {
                    0
                }
            } else {
                i8::from(INC_BIAS.wrapping_add(magnitude) < INC_BIAS)
            };
            Some(carry as u8 as usize)
        }
        _ => None,
    }
}
