#![expect(clippy::unwrap_used)]

use std::mem::{align_of, size_of};

use super::*;

fn selector(table: Option<usize>, raf: bool) -> ProducerSelector {
    ProducerSelector::new(table, raf).unwrap()
}

fn shard(rows: usize) -> ProducerShardPlan {
    ProducerGeometry::new(rows).unwrap().shard(0).unwrap()
}

#[test]
fn selector_convention_covers_all_82_segments() {
    assert_eq!(selector(None, false).segment(), 0);
    assert_eq!(selector(None, true).segment(), 1);
    assert_eq!(selector(None, false).claim(), 0x00);
    assert_eq!(selector(None, true).claim(), 0x80);

    for table in 0..LOOKUP_TABLES {
        for raf in [false, true] {
            let value = selector(Some(table), raf);
            assert_eq!(value.segment(), 2 * (table + 1) + raf as usize);
            assert_eq!(decode_claim(value.claim()).unwrap(), value);
        }
    }
    assert_eq!(selector(Some(LOOKUP_TABLES - 1), true).segment(), 81);
    assert_eq!(
        ProducerSelector::new(Some(LOOKUP_TABLES), false),
        Err(ProducerLayoutError::InvalidTableIndex(LOOKUP_TABLES))
    );
    assert_eq!(decode_claim(41), Err(ProducerLayoutError::InvalidClaim(41)));
}

#[test]
fn log28_geometry_is_four_checked_log26_shards() {
    let geometry = ProducerGeometry::new(1 << 28).unwrap();
    let shards = geometry.shards().unwrap();
    assert_eq!(geometry.shard_count(), 4);
    assert_eq!(shards.len(), 4);

    for (index, shard) in shards.into_iter().enumerate() {
        assert_eq!(shard.shard_index(), index);
        assert_eq!(shard.absolute_row_start(), index * (1 << 26));
        assert_eq!(shard.absolute_row_end().unwrap(), (index + 1) * (1 << 26));
        assert_eq!(shard.rows(), 1 << 26);
        assert_eq!(shard.chunks(), 1 << 14);
        for shape in shard.buffer_shapes().unwrap() {
            assert!(shape.bytes() <= MAX_BUFFER_BYTES);
        }
        assert_eq!(
            shard
                .buffer_shape(PlaneRole::CycleLookupLo)
                .unwrap()
                .bytes(),
            1 << 29
        );
        assert_eq!(
            shard.buffer_shape(PlaneRole::CycleClaims).unwrap().bytes(),
            1 << 26
        );
        assert_eq!(
            shard
                .buffer_shape(PlaneRole::CycleToGroupedLocal)
                .unwrap()
                .bytes(),
            1 << 28
        );
    }
}

#[test]
fn traffic_model_counts_compact_inputs_and_no_grouped_claim_write() {
    let log26 = ScatterTraffic::for_geometry(ProducerGeometry::new(1 << 26).unwrap()).unwrap();
    assert_eq!(log26.input_bytes(), 1_140_850_688);
    assert_eq!(log26.output_bytes(), 1_342_177_280);
    assert_eq!(log26.payload_bytes(), 2_483_027_968);
    assert_eq!(log26.layout_upload_bytes(), 5_374_284);

    let log28 = ScatterTraffic::for_geometry(ProducerGeometry::new(1 << 28).unwrap()).unwrap();
    assert_eq!(log28.rows(), 1 << 28);
    assert_eq!(log28.input_bytes(), 4_563_402_752);
    assert_eq!(log28.output_bytes(), 5_368_709_120);
    assert_eq!(log28.payload_bytes(), 9_932_111_872);
    assert_eq!(log28.layout_upload_bytes(), 21_497_136);
    assert_eq!(PRODUCER_PAYLOAD_BYTES_PER_ROW, 37);
}

#[test]
fn host_scatter_preserves_raw_lookup_limbs_and_cycle_claims() {
    let facts = [
        (u128::MAX, selector(Some(2), true)),
        (11, selector(None, false)),
        (12, selector(Some(2), true)),
        (13, selector(Some(0), false)),
        (14, selector(None, false)),
        (15, selector(Some(0), true)),
        (16, selector(Some(2), true)),
        (17, selector(Some(0), false)),
    ];
    let lookup_lo = facts.map(|(lookup, _)| lookup as u64);
    let lookup_hi = facts.map(|(lookup, _)| (lookup >> 64) as u64);
    let cycle_claims = facts.map(|(_, selector)| selector.claim());
    let scatter =
        HostScatter::from_cycle_planes(shard(8), &lookup_lo, &lookup_hi, &cycle_claims).unwrap();

    assert_eq!(scatter.layout().segment_offsets()[0], 0);
    assert_eq!(scatter.layout().segment_offsets()[1], 2);
    assert_eq!(scatter.layout().segment_offsets()[3], 4);
    assert_eq!(scatter.layout().segment_offsets()[4], 5);
    assert_eq!(scatter.layout().segment_offsets()[7], 5);
    assert_eq!(scatter.layout().segment_offsets()[8], 8);
    assert_eq!(scatter.layout().segment_offsets()[GROUPED_SEGMENTS], 8);
    assert_eq!(cycle_claims[0], selector(Some(2), true).claim());

    let range = scatter.layout().segment_offsets()[7] as usize
        ..scatter.layout().segment_offsets()[8] as usize;
    let lookups: Vec<u128> = range
        .map(|grouped| {
            scatter.grouped_lookup_lo()[grouped] as u128
                | ((scatter.grouped_lookup_hi()[grouped] as u128) << 64)
        })
        .collect();
    assert_eq!(lookups, [u128::MAX, 12, 16]);
}

#[test]
fn checked_oracle_accepts_unordered_scatter_within_a_chunk_segment() {
    let lookup_lo: Vec<u64> = (0..8).collect();
    let lookup_hi: Vec<u64> = (100..108).collect();
    let cycle_claims = vec![selector(Some(3), true).claim(); 8];
    let scatter =
        HostScatter::from_cycle_planes(shard(8), &lookup_lo, &lookup_hi, &cycle_claims).unwrap();
    let mut grouped_lo = scatter.grouped_lookup_lo().to_vec();
    let mut grouped_hi = scatter.grouped_lookup_hi().to_vec();
    let mut inverse = scatter.cycle_to_grouped_local().to_vec();
    let first = inverse[0] as usize;
    let second = inverse[1] as usize;
    grouped_lo.swap(first, second);
    grouped_hi.swap(first, second);
    inverse.swap(0, 1);

    let unordered = HostScatter::from_checked_parts(
        scatter.layout(),
        grouped_lo,
        grouped_hi,
        inverse,
        &lookup_lo,
        &lookup_hi,
        &cycle_claims,
    )
    .unwrap();
    assert_ne!(unordered.cycle_to_grouped_local()[0], first as u32);
}

#[test]
fn chunk_bases_reserve_disjoint_unordered_ranges() {
    let lookup_lo: Vec<u64> = (0..8192).collect();
    let lookup_hi: Vec<u64> = (8192..16_384).collect();
    let cycle_claims: Vec<u8> = (0..8192)
        .map(|cycle| {
            let table = match cycle % 5 {
                0 => None,
                value => Some(value - 1),
            };
            selector(table, cycle % 2 == 1).claim()
        })
        .collect();
    let scatter =
        HostScatter::from_cycle_planes(shard(8192), &lookup_lo, &lookup_hi, &cycle_claims).unwrap();

    assert_eq!(scatter.layout().chunk_segment_bases().len(), 2);
    for segment in 0..GROUPED_SEGMENTS {
        let first = scatter.layout().chunk_segment_count(0, segment).unwrap();
        let second = scatter.layout().chunk_segment_count(1, segment).unwrap();
        assert_eq!(
            first + second,
            scatter.layout().segment_offsets()[segment + 1]
                - scatter.layout().segment_offsets()[segment]
        );
        assert_eq!(
            scatter.layout().chunk_segment_bases()[1][segment],
            scatter.layout().chunk_segment_bases()[0][segment] + first
        );
    }
}

#[test]
fn predispatch_freezes_absolute_bounds_and_plane_capacities() {
    let cycle_claims = vec![selector(None, false).claim(); 8];
    let shard_plan = shard(8);
    let layout = ScatterLayout::from_cycle_claims(shard_plan, &cycle_claims).unwrap();
    let dispatch = ScatterDispatchPlan::new(shard_plan, &layout).unwrap();
    let words = dispatch.params().words();

    assert_eq!(&words[..6], &[8, 0, 8, 1, 82, 4096]);
    assert_eq!(&words[6..12], &[8, 8, 8, 8, 8, 8]);
    assert_eq!(words[12], 82);
    assert_eq!(words[13], 83);
    assert_eq!(words[14], 1);
    assert_eq!(words[15], 0);
    assert_eq!(dispatch.threadgroups(), 1);
    assert_eq!(dispatch.threads_per_group(), 1024);
    assert_eq!(
        dispatch.required_buffers().map(BufferShape::role),
        SCATTER_BUFFER_ROLES
    );
    for (slot, role) in SCATTER_BUFFER_ROLES.into_iter().enumerate() {
        assert_eq!(role.metal_buffer_slot(), slot);
    }

    let other_shard = shard(16);
    let other_layout = ScatterLayout::from_cycle_claims(other_shard, &[0; 16]).unwrap();
    assert_eq!(
        ScatterDispatchPlan::new(shard_plan, &other_layout),
        Err(ProducerLayoutError::ShardMismatch)
    );
}

#[test]
fn malformed_planes_claims_and_layouts_are_rejected() {
    let shard = shard(8);
    let lookup_lo = [0u64; 8];
    let lookup_hi = [0u64; 8];
    let mut claims = [0u8; 8];
    assert!(matches!(
        HostScatter::from_cycle_planes(shard, &lookup_lo[..7], &lookup_hi, &claims),
        Err(ProducerLayoutError::PlaneElements {
            plane: PlaneRole::CycleLookupLo,
            ..
        })
    ));

    claims[3] = 41;
    assert_eq!(
        ScatterLayout::from_cycle_claims(shard, &claims),
        Err(ProducerLayoutError::InvalidClaim(41))
    );
    claims[3] = 0;

    let layout = ScatterLayout::from_cycle_claims(shard, &claims).unwrap();
    let mut offsets = *layout.segment_offsets();
    offsets[GROUPED_SEGMENTS] -= 1;
    assert_eq!(
        ScatterLayout::from_checked_parts(
            shard,
            &claims,
            &offsets,
            layout.chunk_segment_bases().to_vec(),
        ),
        Err(ProducerLayoutError::InvalidLayout(
            "offsets and chunk bases must match the cycle claim counts"
        ))
    );
    assert_eq!(
        ProducerGeometry::new(1 << 29),
        Err(ProducerLayoutError::InvalidRowCount(1 << 29))
    );
}

#[test]
fn rust_and_shader_abis_are_fixed() {
    assert_eq!(size_of::<ScatterParams>(), 64);
    assert_eq!(align_of::<ScatterParams>(), 4);
    assert_eq!(PRODUCER_THREADGROUP_BYTES, 328);
    assert!(METAL_SOURCE.contains("params.shard_row_start + local_cycle"));
    assert!(METAL_SOURCE.contains("atomic_fetch_add_explicit"));
    assert!(METAL_SOURCE.contains("cycle_to_grouped_local"));
    assert!(!METAL_SOURCE.contains("grouped_claims"));
}

#[test]
fn runtime_binding_checks_fail_closed_without_a_device() {
    use super::runtime::{
        validate_recorded_binding, validate_source_aliases, BindingFacts, BindingRecord,
    };

    let record = BindingRecord {
        shard: 2,
        role: PlaneRole::GroupedLookupHi,
        bytes: 512,
        device_registry_id: 17,
        allocation_identity: 0x4000,
        source_generation: 9,
    };
    let facts = BindingFacts {
        bytes: 512,
        device_registry_id: 17,
        allocation_identity: 0x4000,
    };
    validate_recorded_binding(record, facts, 9, 1024).unwrap();

    assert!(matches!(
        validate_recorded_binding(
            record,
            BindingFacts {
                bytes: 256,
                ..facts
            },
            9,
            1024
        ),
        Err(ProducerRuntimeError::BufferLength {
            shard: 2,
            role: PlaneRole::GroupedLookupHi,
            ..
        })
    ));
    assert!(matches!(
        validate_recorded_binding(
            record,
            BindingFacts {
                device_registry_id: 18,
                ..facts
            },
            9,
            1024
        ),
        Err(ProducerRuntimeError::BufferDevice { .. })
    ));
    assert!(matches!(
        validate_recorded_binding(
            record,
            BindingFacts {
                allocation_identity: 0x5000,
                ..facts
            },
            9,
            1024
        ),
        Err(ProducerRuntimeError::BufferIdentity { .. })
    ));
    assert!(matches!(
        validate_recorded_binding(record, facts, 10, 1024),
        Err(ProducerRuntimeError::SourceGeneration {
            expected: 10,
            got: 9,
            ..
        })
    ));
    assert!(matches!(
        validate_recorded_binding(record, facts, 9, 511),
        Err(ProducerRuntimeError::DeviceBufferLimit {
            bytes: 512,
            maximum: 511,
            ..
        })
    ));
    assert!(matches!(
        validate_source_aliases(2, [0x1000, 0x2000, 0x1000]),
        Err(ProducerRuntimeError::AliasedBuffers {
            first_role: PlaneRole::CycleLookupLo,
            second_role: PlaneRole::CycleClaims,
            identity: 0x1000,
            ..
        })
    ));
}

#[test]
fn resident_runtime_matches_the_unordered_host_oracle_and_reuses_outputs() {
    use metal::{MTLCommandBufferStatus, MTLResourceOptions};

    let context = super::super::SolinasMetal::for_akita().unwrap();
    let geometry = ProducerGeometry::new(8192).unwrap();
    let shard = geometry.shard(0).unwrap();
    let lookup_lo = (0..shard.rows())
        .map(|cycle| (cycle as u64).wrapping_mul(0x9e37_79b9))
        .collect::<Vec<_>>();
    let lookup_hi = (0..shard.rows())
        .map(|cycle| !(cycle as u64).rotate_left(17))
        .collect::<Vec<_>>();
    let claims = (0..shard.rows())
        .map(|cycle| {
            let table = (cycle % 7 != 0).then_some(cycle % LOOKUP_TABLES);
            selector(table, cycle % 3 == 0).claim()
        })
        .collect::<Vec<_>>();
    let layout = ScatterLayout::from_cycle_claims(shard, &claims).unwrap();
    let source = context
        .attach_instruction_read_raf_producer_source_shard(
            shard,
            7,
            11,
            super::super::buffer_from_slice(&context.device, &lookup_lo),
            super::super::buffer_from_slice(&context.device, &lookup_hi),
            super::super::buffer_from_slice(&context.device, &claims),
        )
        .unwrap();
    let source_identities = source.receipt().allocation_identities();
    let mut owner = context
        .prepare_instruction_read_raf_producer(
            geometry,
            vec![ProducerShardInput::new(source, layout).unwrap()],
        )
        .unwrap();
    assert_eq!(owner.source_copy_bytes(), 0);
    assert_eq!(owner.source_resident_bytes(), 8192 * 17);
    assert_eq!(owner.execute_device_buffer_allocations(), 0);

    let completed = owner.execute().unwrap();
    assert_eq!(completed.receipt().dispatch_serial(), 1);
    assert_eq!(completed.receipt().source_generation(), 7);
    assert_eq!(completed.receipt().source_completion_serial(), 11);
    let output = completed.shard(0).unwrap();
    assert_eq!(
        output.cycle_lookup_lo().receipt().allocation_identity(),
        source_identities[0]
    );
    assert_eq!(
        output.cycle_claims().receipt().initialization(),
        ProducerPlaneInitialization::Source {
            completion_serial: 11
        }
    );
    assert_eq!(
        output.segment_offsets().receipt().initialization(),
        ProducerPlaneInitialization::FrozenLayout
    );
    assert_eq!(
        output.grouped_lookup_lo().receipt().initialization(),
        ProducerPlaneInitialization::ScatterDispatch { serial: 1 }
    );
    let output_identities = [
        output.grouped_lookup_lo().receipt().allocation_identity(),
        output.grouped_lookup_hi().receipt().allocation_identity(),
        output
            .cycle_to_grouped_local()
            .receipt()
            .allocation_identity(),
    ];

    let grouped_lo = context.device.new_buffer(
        output.grouped_lookup_lo().receipt().bytes(),
        MTLResourceOptions::StorageModeShared,
    );
    let grouped_hi = context.device.new_buffer(
        output.grouped_lookup_hi().receipt().bytes(),
        MTLResourceOptions::StorageModeShared,
    );
    let inverse = context.device.new_buffer(
        output.cycle_to_grouped_local().receipt().bytes(),
        MTLResourceOptions::StorageModeShared,
    );
    let command = context.queue.new_command_buffer();
    let blit = command.new_blit_command_encoder();
    blit.copy_from_buffer(
        output.grouped_lookup_lo().buffer(),
        0,
        &grouped_lo,
        0,
        grouped_lo.length(),
    );
    blit.copy_from_buffer(
        output.grouped_lookup_hi().buffer(),
        0,
        &grouped_hi,
        0,
        grouped_hi.length(),
    );
    blit.copy_from_buffer(
        output.cycle_to_grouped_local().buffer(),
        0,
        &inverse,
        0,
        inverse.length(),
    );
    blit.end_encoding();
    command.commit();
    command.wait_until_completed();
    assert_eq!(command.status(), MTLCommandBufferStatus::Completed);

    // SAFETY: the blit completed and each shared allocation has exactly the
    // element count passed to from_checked_parts.
    let grouped_lo_values = unsafe {
        std::slice::from_raw_parts(grouped_lo.contents().cast::<u64>(), shard.rows()).to_vec()
    };
    // SAFETY: the same completed blit and allocation-size argument applies.
    let grouped_hi_values = unsafe {
        std::slice::from_raw_parts(grouped_hi.contents().cast::<u64>(), shard.rows()).to_vec()
    };
    // SAFETY: the same completed blit and allocation-size argument applies.
    let inverse_values = unsafe {
        std::slice::from_raw_parts(inverse.contents().cast::<u32>(), shard.rows()).to_vec()
    };
    let _ = HostScatter::from_checked_parts(
        output.layout(),
        grouped_lo_values,
        grouped_hi_values,
        inverse_values,
        &lookup_lo,
        &lookup_hi,
        &claims,
    )
    .unwrap();

    let second = owner.execute().unwrap();
    assert_eq!(second.receipt().dispatch_serial(), 2);
    let second = second.shard(0).unwrap();
    assert_eq!(
        [
            second.grouped_lookup_lo().receipt().allocation_identity(),
            second.grouped_lookup_hi().receipt().allocation_identity(),
            second
                .cycle_to_grouped_local()
                .receipt()
                .allocation_identity(),
        ],
        output_identities
    );
}
