use jolt_field::AkitaField;

use super::abi::{
    CarrierProducer, MidpointUpcLease, PartialCarrierHeader, PartialCarrierLease,
    ResidentBufferDescriptor, SpartanShiftSuccessorAbiError, SpartanShiftSuccessorGeometry,
    OUTER_COMPONENT_TABLES,
};
use super::model::{
    target_plan, AttributionBoundary, MidpointPlan, HALF_WIDTH_PROMOTION_FLOOR_PER_SECOND,
    RETAINED_MATCHED_HALF_TERMS_PER_SECOND,
};
use super::oracle::{
    attach_midpoint_upc, combine_q, direct_trace, factorized_initial_claim, fold_all_columns,
    fold_residual_columns, outer_component_tables, product_component_tables,
    SpartanShiftSuccessorRow,
};

#[test]
fn target_model_freezes_exact_work_and_roofs() {
    let primary = target_plan(MidpointPlan::BorrowInstructionInputUpc).unwrap();
    assert_eq!(primary.gross_half_width_terms().unwrap(), 335_527_936);
    assert_eq!(primary.incremental_half_width_terms().unwrap(), 201_326_592);
    assert_eq!(primary.full_products().unwrap(), 352_213);
    assert_eq!(
        primary.kernel_logical_device_bytes().unwrap(),
        1_665_007_616
    );
    assert_eq!(primary.transient_upc_projection_bytes, 536_870_912);
    assert_eq!(primary.producer_projection_bytes, 1_098_907_648);
    assert_eq!(primary.logical_device_bytes().unwrap(), 2_763_915_264);
    assert_eq!(primary.host_handoff_bytes, 1_179_648);

    let roof = primary
        .roof(
            AttributionBoundary::GrossCoMaterializedCompact,
            RETAINED_MATCHED_HALF_TERMS_PER_SECOND,
        )
        .unwrap();
    assert_eq!(roof.compute_floor_ns, 10_116_014);
    assert_eq!(roof.traffic_floor_ns, 6_118_895);
    assert_eq!(roof.binding_floor_ns, roof.compute_floor_ns);
    assert_eq!(roof.eighty_percent_bar_ns, 12_645_018);
    assert_eq!(primary.projection_write_floor_ns().unwrap(), 2_432_818);
    assert_eq!(
        primary
            .nonoverlapped_projection_compute_floor_ns(RETAINED_MATCHED_HALF_TERMS_PER_SECOND,)
            .unwrap(),
        12_548_832
    );

    let promotion_floor = primary
        .roof(
            AttributionBoundary::GrossCoMaterializedCompact,
            HALF_WIDTH_PROMOTION_FLOOR_PER_SECOND,
        )
        .unwrap();
    assert_eq!(promotion_floor.compute_floor_ns, 12_771_314);

    let incremental = primary
        .roof(
            AttributionBoundary::ResidentPiopIncremental,
            RETAINED_MATCHED_HALF_TERMS_PER_SECOND,
        )
        .unwrap();
    assert_eq!(incremental.compute_floor_ns, 6_069_905);
    assert_eq!(incremental.traffic_floor_ns, 0);
    assert_eq!(
        primary.fresh_scan_compulsory_device_bytes(false).unwrap(),
        11_873_943_552
    );
    assert_eq!(
        primary.fresh_scan_compulsory_device_bytes(true).unwrap(),
        15_095_169_024
    );
    assert_eq!(
        primary
            .roof(
                AttributionBoundary::GrossFreshFusedOuter,
                RETAINED_MATCHED_HALF_TERMS_PER_SECOND,
            )
            .unwrap()
            .traffic_floor_ns,
        26_287_135
    );
    assert_eq!(
        primary
            .roof(
                AttributionBoundary::GrossFreshSplitOuter,
                RETAINED_MATCHED_HALF_TERMS_PER_SECOND,
            )
            .unwrap()
            .traffic_floor_ns,
        33_418_446
    );

    let full = target_plan(MidpointPlan::SelfContained).unwrap();
    assert_eq!(full.gross_half_width_terms().unwrap(), 402_636_800);
    assert_eq!(full.kernel_logical_device_bytes().unwrap(), 2_202_009_600);
    assert_eq!(full.producer_projection_bytes, 1_098_907_648);
    assert_eq!(full.logical_device_bytes().unwrap(), 3_300_917_248);
    assert_eq!(
        full.roof(
            AttributionBoundary::GrossCoMaterializedCompact,
            RETAINED_MATCHED_HALF_TERMS_PER_SECOND,
        )
        .unwrap()
        .compute_floor_ns,
        12_139_316
    );
}

#[test]
fn abi_rejects_shader_shapes_that_would_misindex_or_underallocate() {
    let geometry = SpartanShiftSuccessorGeometry::new(1 << 12).unwrap();
    assert_eq!(
        geometry.reduction_params(geometry.suffix_elements, 3),
        Err(SpartanShiftSuccessorAbiError::InvalidOutputColumns)
    );
    assert_eq!(
        geometry
            .outer_partial_params(geometry.suffix_elements)
            .unwrap()
            .output_columns,
        4
    );
    assert_eq!(
        geometry
            .product_partial_params(geometry.suffix_elements)
            .unwrap()
            .output_columns,
        2
    );
    assert_eq!(
        SpartanShiftSuccessorGeometry::fold_threadgroup_bytes(48, 4),
        Err(SpartanShiftSuccessorAbiError::InvalidThreadgroupWidth)
    );
    assert_eq!(
        SpartanShiftSuccessorGeometry::fold_threadgroup_bytes(64, 4),
        Ok(128)
    );
    assert_eq!(
        SpartanShiftSuccessorGeometry::fold_threadgroup_bytes(128, 5),
        Ok(320)
    );
    let oversized = SpartanShiftSuccessorGeometry::new(1 << 30).unwrap();
    assert_eq!(
        oversized.outer_partial_params(1),
        Err(SpartanShiftSuccessorAbiError::ShaderIndexOverflow)
    );
}

#[test]
fn component_carriers_equal_the_unfactored_dense_claim() {
    let rows = sample_rows(4);
    let r_outer = values(4, 0xA110_0001);
    let r_product = values(4, 0xB220_0002);
    let challenges = values(4, 0xC330_0003);
    let gamma = AkitaField::from_u64(0xD440_0004);

    let direct = direct_trace(&rows, &r_outer, &r_product, gamma, &challenges).unwrap();
    let outer = outer_component_tables(&rows, &r_outer).unwrap();
    let product = product_component_tables(&rows, &r_product).unwrap();
    let q = combine_q(&outer, &product, gamma).unwrap();
    let factorized = factorized_initial_claim(&q, &r_outer, &r_product).unwrap();
    assert_eq!(factorized, direct.initial_claim);

    assert_ne!(q.outer_current, q.outer_successor);
    assert_ne!(q.product_current, q.product_successor);
    assert_ne!(direct.round_endpoints[0][0], direct.round_endpoints[0][1]);
}

#[test]
fn midpoint_alias_reconstructs_every_dense_table() {
    let rows = sample_rows(6);
    let prefix_challenges = values(3, 0xE550_0005);
    let full = fold_all_columns(&rows, &prefix_challenges).unwrap();
    let residual = fold_residual_columns(&rows, &prefix_challenges).unwrap();
    let attached = attach_midpoint_upc(residual, full.unexpanded_pc.clone()).unwrap();
    assert_eq!(attached, full);

    let mut wrong = full.unexpanded_pc;
    wrong.pop();
    let residual = fold_residual_columns(&rows, &prefix_challenges).unwrap();
    assert!(attach_midpoint_upc(residual, wrong).is_err());
}

#[test]
fn carrier_metadata_rejects_wrong_point_and_aliases() {
    let geometry = SpartanShiftSuccessorGeometry::new(1 << 10).unwrap();
    let point_digest = [7; 32];
    let mut next_id = 11u64;
    let tables = core::array::from_fn(|_| {
        let descriptor = ResidentBufferDescriptor {
            storage_id: next_id,
            byte_len: geometry.table_bytes().unwrap(),
        };
        next_id += 1;
        descriptor
    });
    let lease = PartialCarrierLease::<OUTER_COMPONENT_TABLES> {
        header: PartialCarrierHeader {
            producer: CarrierProducer::Stage1Outer,
            witness_generation: 3,
            device_registry_id: 5,
            rows: geometry.rows,
            table_elements: geometry.prefix_elements,
            point_digest,
        },
        tables,
    };
    assert_eq!(
        lease.validate(geometry, CarrierProducer::Stage1Outer, 3, 5, point_digest,),
        Ok(lease)
    );
    assert_eq!(
        lease.validate(geometry, CarrierProducer::Stage1Outer, 3, 5, [8; 32],),
        Err(SpartanShiftSuccessorAbiError::WrongPointDigest)
    );

    let mut aliased = lease;
    aliased.tables[1].storage_id = aliased.tables[0].storage_id;
    assert_eq!(
        aliased.validate(geometry, CarrierProducer::Stage1Outer, 3, 5, point_digest,),
        Err(SpartanShiftSuccessorAbiError::DuplicateBufferIdentity)
    );
}

#[test]
fn midpoint_metadata_binds_the_ordered_challenges() {
    let geometry = SpartanShiftSuccessorGeometry::new(1 << 12).unwrap();
    let lease = MidpointUpcLease {
        producer: CarrierProducer::Stage3InstructionInput,
        witness_generation: 13,
        device_registry_id: 17,
        rows: geometry.rows,
        table_elements: geometry.suffix_elements,
        ordered_challenge_digest: [19; 32],
        table: ResidentBufferDescriptor {
            storage_id: 23,
            byte_len: geometry.dense_table_bytes().unwrap(),
        },
    };
    assert_eq!(lease.validate(geometry, 13, 17, [19; 32]), Ok(lease));
    assert_eq!(
        lease.validate(geometry, 13, 17, [20; 32]),
        Err(SpartanShiftSuccessorAbiError::WrongChallengeDigest)
    );
}

fn sample_rows(log_t: usize) -> Vec<SpartanShiftSuccessorRow> {
    let rows = 1usize << log_t;
    (0..rows)
        .map(|index| SpartanShiftSuccessorRow {
            unexpanded_pc: splitmix(index as u64 ^ 0x243F_6A88_85A3_08D3),
            pc: splitmix(index as u64 ^ 0x1319_8A2E_0370_7344),
            is_virtual: index % 5 == 1,
            is_first_in_sequence: index % 7 == 3,
            is_noop: index == 0 || index + 1 == rows || index % 11 == 4,
        })
        .collect()
}

fn values(count: usize, seed: u64) -> Vec<AkitaField> {
    (0..count)
        .map(|index| AkitaField::from_u64(splitmix(seed ^ index as u64)))
        .collect()
}

fn splitmix(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}
