use core::mem::{align_of, size_of};

use jolt_field::{Fr, FromPrimitiveInt};

use super::abi::{
    IncrementAccessRow, IncrementAccessSource, RamValBufferRange, RamValFirstMessageBufferLengths,
    RamValFirstMessageParams, RamValLaunch, RamValReductionBuffers, RamValReductionParams,
    RamValSuccessorDispatchError, RamValSuccessorRowError, FIRST_MESSAGE_PIPELINE, MESSAGE_COLUMNS,
    NO_RAM_ADDRESS, SIMD_WIDTH, STATUS_INVALID_ROW, STATUS_UNSUPPORTED,
};
use super::model::{
    admission_decision, heuristic_first_message_ns, heuristic_hybrid_ns, median_of_five,
    sparse_screen_class, speed_screen_decision, target_work_plan, ActivityProvenance,
    ActivityProvenanceRejection, AdmissionDecision, CandidateEvidence, CompiledCaptureEvidence,
    CompiledCaptureRejection, CompiledPhaseResources, FirstMessageActivity, Geometry,
    PhaseLatencySamples, PhaseRoofRejection, ProducerEvidence, ProducerKind, ProducerRejection,
    SparseScreenClass, SuccessorPhase, EIGHT_X_SCREEN_CAP_NS, FIVE_X_SCREEN_CAP_NS,
    FROZEN_CPU_ARTIFACT, FROZEN_CPU_ARTIFACT_SHA256, FROZEN_CPU_MEDIAN_NS, FROZEN_CPU_REVISION,
    FROZEN_CPU_SAMPLES_NS, FROZEN_CPU_SAMPLE_SELECTOR, RETAINED_METAL_EVIDENCE,
    RETAINED_METAL_EVIDENCE_SHA256, SCREENING_EVIDENCE_JSON, SPARSE_SCREEN_PROXY_PRIORITY_PERMILLE,
    SPARSE_SCREEN_TARGET_PRIORITY_PERMILLE, TARGET_CPU_CUTOFF, TARGET_ROWS,
};
use super::oracle::{
    dense_lt_from_split, direct_first_message, factorized_sparse_first_message, final_claim,
    input_claim, InitContribution,
};

fn field(value: u64) -> Fr {
    Fr::from_u64(value)
}

fn row(
    remapped_ram_address: Option<u64>,
    store: bool,
    ram_increment: i128,
    rd_increment: i128,
) -> IncrementAccessRow {
    IncrementAccessRow::from_source(
        IncrementAccessSource {
            remapped_ram_address,
            store,
            ram_increment,
            rd_increment,
        },
        4,
    )
    .unwrap()
}

fn fixture_rows() -> Vec<IncrementAccessRow> {
    vec![
        IncrementAccessRow::default(),
        row(Some(1), false, 0, 7),
        row(Some(2), true, 5, 0),
        row(Some(1), true, -3, 0),
        row(Some(3), false, 0, -4),
        row(Some(0), true, 0, 0),
        row(None, false, 0, 9),
        row(None, true, 2, 0),
    ]
}

#[test]
fn producer_row_abi_is_exact_and_round_trips() {
    assert_eq!(size_of::<IncrementAccessRow>(), 16);
    assert_eq!(align_of::<IncrementAccessRow>(), 8);
    assert_eq!(size_of::<RamValFirstMessageParams>(), 32);
    assert_eq!(size_of::<RamValReductionParams>(), 16);
    assert_eq!(STATUS_UNSUPPORTED, 1);
    assert_eq!(STATUS_INVALID_ROW, 2);

    for row in fixture_rows() {
        assert_eq!(IncrementAccessRow::try_from_words(row.words(), 4), Ok(row));
    }
}

#[test]
fn composite_producer_checks_exclusivity_in_release_logic() {
    let store_with_rd = IncrementAccessSource {
        remapped_ram_address: Some(1),
        store: true,
        ram_increment: 3,
        rd_increment: 4,
    };
    assert_eq!(
        IncrementAccessRow::from_source(store_with_rd, 4),
        Err(RamValSuccessorRowError::IncrementExclusivity {
            store: true,
            ram_increment: 3,
            rd_increment: 4,
        })
    );

    let non_store_with_ram = IncrementAccessSource {
        remapped_ram_address: Some(2),
        store: false,
        ram_increment: -7,
        rd_increment: 0,
    };
    assert_eq!(
        IncrementAccessRow::from_source(non_store_with_ram, 4),
        Err(RamValSuccessorRowError::IncrementExclusivity {
            store: false,
            ram_increment: -7,
            rd_increment: 0,
        })
    );
}

#[test]
fn producer_row_rejects_noncanonical_encodings() {
    let negative_zero = [0, u64::from(NO_RAM_ADDRESS)];
    assert_eq!(
        IncrementAccessRow::try_from_words(negative_zero, 4),
        Err(RamValSuccessorRowError::NegativeZero)
    );

    let reserved = [1, u64::from(NO_RAM_ADDRESS) | ((1_u64 << 7) << 32)];
    assert_eq!(
        IncrementAccessRow::try_from_words(reserved, 4),
        Err(RamValSuccessorRowError::ReservedFlags(1 << 7))
    );
    assert_eq!(
        IncrementAccessRow::from_source(
            IncrementAccessSource {
                remapped_ram_address: Some(u64::from(NO_RAM_ADDRESS)),
                store: false,
                ram_increment: 0,
                rd_increment: 0,
            },
            4,
        ),
        Err(RamValSuccessorRowError::SentinelCollision)
    );
    assert!(matches!(
        IncrementAccessRow::from_source(
            IncrementAccessSource {
                remapped_ram_address: Some(0),
                store: true,
                ram_increment: i128::from(u64::MAX) + 1,
                rd_increment: 0,
            },
            4,
        ),
        Err(RamValSuccessorRowError::IncrementOutOfRange(_))
    ));
    let outside_domain = IncrementAccessSource {
        remapped_ram_address: Some(4),
        store: false,
        ram_increment: 0,
        rd_increment: 0,
    };
    assert_eq!(
        IncrementAccessRow::from_source(outside_domain, 4),
        Err(RamValSuccessorRowError::AddressOutOfDomain {
            address: 4,
            domain: 4,
        })
    );
    let too_wide = IncrementAccessSource {
        remapped_ram_address: Some(u64::from(u32::MAX) + 1),
        ..outside_domain
    };
    assert_eq!(
        IncrementAccessRow::from_source(too_wide, 4),
        Err(RamValSuccessorRowError::RemappedAddressOutOfRange(
            u64::from(u32::MAX) + 1
        ))
    );
    assert_eq!(
        IncrementAccessRow::from_source(IncrementAccessSource::default(), 3),
        Err(RamValSuccessorRowError::InvalidAddressDomain(3))
    );
}

#[test]
fn zero_address_store_is_valid_and_has_zero_ram_ra() {
    let zero_address_store = row(None, true, -11, 0);
    let eq = [field(3), field(5), field(7), field(9)];
    assert!(zero_address_store.is_ram_increment());
    assert_eq!(
        zero_address_store.ram_increment::<Fr>(),
        field(0) - field(11)
    );
    assert_eq!(zero_address_store.ram_ra(&eq), Ok(field(0)));

    let raw = zero_address_store.words();
    assert_eq!(
        IncrementAccessRow::try_from_words(raw, 4),
        Ok(zero_address_store)
    );
}

#[test]
fn load_keeps_ra_but_contributes_zero_ram_increment() {
    let load = row(Some(2), false, 0, -11);
    let eq = [field(3), field(5), field(7), field(9)];
    assert_eq!(load.ram_increment::<Fr>(), field(0));
    assert_eq!(load.ram_ra(&eq), Ok(eq[2]));
}

#[test]
fn sparse_factorization_matches_direct_relation() {
    let rows = fixture_rows();
    let eq_address = [field(2), field(3), field(5), field(7)];
    let lt_low = [field(11), field(13), field(17), field(19)];
    let lt_high = [field(23), field(29)];
    let eq_high = [field(31), field(37)];
    let dense = dense_lt_from_split(rows.len(), &lt_low, &lt_high, &eq_high).unwrap();
    let direct = direct_first_message(&rows, &eq_address, &dense).unwrap();
    let sparse =
        factorized_sparse_first_message(&rows, &eq_address, &lt_low, &lt_high, &eq_high).unwrap();
    assert_eq!(sparse, direct);
}

#[test]
fn zero_address_store_preserves_neighbor_cross_terms() {
    let rows = [
        row(None, true, 5, 0),
        row(Some(2), false, 0, 0),
        IncrementAccessRow::default(),
        IncrementAccessRow::default(),
    ];
    let eq_address = [field(2), field(3), field(5), field(7)];
    let lt_low = [field(11), field(13)];
    let lt_high = [field(17), field(19)];
    let eq_high = [field(23), field(29)];
    let dense = dense_lt_from_split(rows.len(), &lt_low, &lt_high, &eq_high).unwrap();
    assert_eq!(
        factorized_sparse_first_message(&rows, &eq_address, &lt_low, &lt_high, &eq_high),
        direct_first_message(&rows, &eq_address, &dense)
    );
}

#[test]
fn input_claim_matches_the_symbolic_init_decomposition() {
    let ram_val = field(11);
    let ram_val_final = field(13);
    let init_eval = field(17);
    let gamma = field(19);
    let contributions = [
        InitContribution {
            selector: field(23),
            opening: field(29),
        },
        InitContribution {
            selector: field(31),
            opening: field(37),
        },
    ];
    let init = init_eval
        - contributions[0].selector * contributions[0].opening
        - contributions[1].selector * contributions[1].opening;
    assert_eq!(
        input_claim(ram_val, ram_val_final, init_eval, &contributions, gamma,),
        ram_val + gamma * ram_val_final - (field(1) + gamma) * init
    );
}

#[test]
fn final_claim_is_the_verifier_output_product() {
    let rows = fixture_rows();
    let eq_address = [field(2), field(3), field(5), field(7)];
    let lt_low = [field(11), field(13), field(17), field(19)];
    let lt_high = [field(23), field(29)];
    let eq_high = [field(31), field(37)];
    let dense = dense_lt_from_split(rows.len(), &lt_low, &lt_high, &eq_high).unwrap();
    let claim = final_claim(
        &rows,
        &eq_address,
        &dense,
        &[field(41), field(43), field(47)],
    )
    .unwrap();
    assert_eq!(
        claim.product,
        claim.ram_inc * claim.ram_ra * claim.lt_cycle_plus_gamma
    );
}

fn first_message_params() -> RamValFirstMessageParams {
    RamValFirstMessageParams {
        rows: 8,
        high_blocks: 2,
        low_length: 4,
        address_domain: 4,
        threads: SIMD_WIDTH as u32,
        no_address: NO_RAM_ADDRESS,
        reserved: [0; 2],
    }
}

#[test]
fn host_launch_contract_checks_geometry_status_and_buffers() {
    let launch = RamValLaunch {
        threadgroups: 2,
        threads_per_threadgroup: SIMD_WIDTH as u32,
        initial_status: 0,
    };
    let buffers = RamValFirstMessageBufferLengths {
        rows: 8,
        eq_address: 4,
        lt_low: 4,
        lt_high: 2,
        eq_high: 2,
        partials: 2 * MESSAGE_COLUMNS,
        status_words: 1,
    };
    assert_eq!(
        first_message_params().validate_launch(launch, buffers),
        Ok(())
    );

    let uncleared = RamValLaunch {
        initial_status: STATUS_INVALID_ROW,
        ..launch
    };
    assert_eq!(
        first_message_params().validate_launch(uncleared, buffers),
        Err(RamValSuccessorDispatchError::StatusNotCleared(
            STATUS_INVALID_ROW
        ))
    );
    let short_partials = RamValFirstMessageBufferLengths {
        partials: 2 * MESSAGE_COLUMNS - 1,
        ..buffers
    };
    assert_eq!(
        first_message_params().validate_launch(launch, short_partials),
        Err(RamValSuccessorDispatchError::BufferTooShort {
            name: "partials",
            required: 2 * MESSAGE_COLUMNS,
            got: 2 * MESSAGE_COLUMNS - 1,
        })
    );
}

#[test]
fn reduction_contract_rejects_overdispatch_and_short_buffers() {
    let params = RamValReductionParams {
        input_count: 8_192,
        output_count: 256,
        columns: MESSAGE_COLUMNS as u32,
        reserved: 0,
    };
    let launch = RamValLaunch {
        threadgroups: 256,
        threads_per_threadgroup: SIMD_WIDTH as u32,
        initial_status: 0,
    };
    let input_bytes = 3 * 8_192 * 16;
    let output_bytes = 3 * 256 * 16;
    let buffers = RamValReductionBuffers {
        input: RamValBufferRange {
            storage_id: 11,
            offset_bytes: 0,
            length_bytes: input_bytes,
        },
        output: RamValBufferRange {
            storage_id: 12,
            offset_bytes: 0,
            length_bytes: output_bytes,
        },
        status: RamValBufferRange {
            storage_id: 13,
            offset_bytes: 0,
            length_bytes: 4,
        },
    };
    assert_eq!(params.validate_launch(launch, buffers), Ok(()));

    let overdispatch = RamValLaunch {
        threadgroups: 257,
        ..launch
    };
    assert_eq!(
        params.validate_launch(overdispatch, buffers),
        Err(RamValSuccessorDispatchError::WrongThreadgroups {
            expected: 256,
            got: 257,
        })
    );

    let short_output = RamValReductionBuffers {
        output: RamValBufferRange {
            length_bytes: output_bytes - 16,
            ..buffers.output
        },
        ..buffers
    };
    assert_eq!(
        params.validate_launch(launch, short_output),
        Err(RamValSuccessorDispatchError::BufferRangeTooShort {
            name: "output",
            required_bytes: output_bytes,
            got_bytes: output_bytes - 16,
        })
    );
}

#[test]
fn reduction_contract_rejects_data_and_status_aliasing() {
    let params = RamValReductionParams {
        input_count: 8_192,
        output_count: 256,
        columns: MESSAGE_COLUMNS as u32,
        reserved: 0,
    };
    let launch = RamValLaunch {
        threadgroups: 256,
        threads_per_threadgroup: SIMD_WIDTH as u32,
        initial_status: 0,
    };
    let input_bytes = 3 * 8_192 * 16;
    let output_bytes = 3 * 256 * 16;
    let adjacent = RamValReductionBuffers {
        input: RamValBufferRange {
            storage_id: 21,
            offset_bytes: 0,
            length_bytes: input_bytes,
        },
        output: RamValBufferRange {
            storage_id: 21,
            offset_bytes: input_bytes,
            length_bytes: output_bytes,
        },
        status: RamValBufferRange {
            storage_id: 21,
            offset_bytes: input_bytes + output_bytes,
            length_bytes: 4,
        },
    };
    assert_eq!(params.validate_launch(launch, adjacent), Ok(()));

    let overlapping_output = RamValReductionBuffers {
        output: RamValBufferRange {
            offset_bytes: input_bytes - 16,
            ..adjacent.output
        },
        ..adjacent
    };
    assert_eq!(
        params.validate_launch(launch, overlapping_output),
        Err(RamValSuccessorDispatchError::OverlappingBufferRanges {
            left: "input",
            right: "output",
        })
    );

    let overlapping_status = RamValReductionBuffers {
        status: RamValBufferRange {
            offset_bytes: input_bytes,
            ..adjacent.status
        },
        ..adjacent
    };
    assert_eq!(
        params.validate_launch(launch, overlapping_status),
        Err(RamValSuccessorDispatchError::OverlappingBufferRanges {
            left: "output",
            right: "status",
        })
    );

    let missing_identity = RamValReductionBuffers {
        status: RamValBufferRange {
            storage_id: 0,
            ..adjacent.status
        },
        ..adjacent
    };
    assert_eq!(
        params.validate_launch(launch, missing_identity),
        Err(RamValSuccessorDispatchError::MissingBufferIdentity { name: "status" })
    );
}

#[test]
fn target_dense_work_and_traffic_are_exact() {
    assert_eq!(Geometry::target().cpu_cutoff(), TARGET_CPU_CUTOFF);
    let plan = target_work_plan(FirstMessageActivity::dense(Geometry::target())).unwrap();
    assert_eq!(plan.phases[0].logical_products, 201_375_744);
    assert_eq!(plan.phases[1].logical_products, 167_821_312);
    assert_eq!(plan.phases[2].logical_products, 167_886_848);
    assert_eq!(plan.logical_products(), 537_083_904);
    assert_eq!(plan.lane_zero_products(), 540_672);
    assert_eq!(plan.phases[0].simd_equivalent_product_slots, 202_899_456);
    assert_eq!(plan.phases[1].simd_equivalent_product_slots, 169_345_024);
    assert_eq!(plan.phases[2].simd_equivalent_product_slots, 187_170_816);
    assert_eq!(plan.simd_equivalent_product_slots(), 559_415_296);

    assert_eq!(plan.phases[0].large_state_bytes, 1_073_741_824);
    assert_eq!(plan.phases[1].large_state_bytes, 2_147_483_648);
    assert_eq!(plan.phases[2].large_state_bytes, 3_214_934_016);
    assert_eq!(plan.large_state_bytes(), 6_436_159_488);
    assert_eq!(plan.partial_per_message_bytes, 811_824);
    assert_eq!(plan.partial_global_bytes, 8_930_064);
    assert_eq!(plan.first_message_accounted_bytes(), Ok(1_074_553_656));
    assert_eq!(plan.challenge_table_initial_write_bytes, 524_288);
    assert_eq!(plan.challenge_table_bind_write_bytes, 130_944);
    assert_eq!(plan.cpu_tail_handoff_bytes, 2_097_152);
    assert_eq!(plan.host_message_read_bytes, 528);
    assert_eq!(plan.status_per_message_bytes, 8);
    assert_eq!(plan.status_host_io_bytes, 88);
    assert_eq!(plan.accounted_compulsory_bytes(), 6_447_842_552);
    assert_eq!(plan.sequence_resident_bytes, 2_685_665_284);
    assert_eq!(plan.producer_diagnostic_write_bytes, 1_073_741_824);

    assert_eq!(plan.phase_message_counts, [1, 1, 9]);
    assert_eq!(plan.phase_bind_write_bytes, [0, 65_536, 65_408]);
    assert_eq!(plan.phase_handoff_bytes, [0, 0, 2_097_152]);
    assert_eq!(
        plan.phase_accounted_bytes(SuccessorPhase::FirstMessage),
        Ok(1_074_553_704)
    );
    assert_eq!(
        plan.phase_accounted_bytes(SuccessorPhase::NativeBindAndMessage),
        Ok(2_148_361_064)
    );
    assert_eq!(
        plan.phase_accounted_bytes(SuccessorPhase::DenseTransitions),
        Ok(3_224_403_496)
    );
    let roofs = plan.phase_roofs().unwrap();
    assert_eq!(roofs[0].eighty_percent_roof_bar_ns, 7_844_860);
    assert_eq!(roofs[1].eighty_percent_roof_bar_ns, 6_547_519);
    assert_eq!(roofs[2].eighty_percent_roof_bar_ns, 8_922_934);
}

#[test]
fn sparse_work_separates_logical_products_from_simd_slots() {
    let activity = FirstMessageActivity {
        active_pairs: 10,
        active_simd_iterations: 2,
    };
    let plan = target_work_plan(activity).unwrap();
    assert_eq!(plan.phases[0].logical_products, 49_212);
    assert_eq!(plan.phases[0].lane_zero_products, 49_152);
    assert_eq!(plan.phases[0].simd_equivalent_product_slots, 1_573_248);
    assert_eq!(activity.active_simd_permille(Geometry::target()), Ok(1));
    assert!(target_work_plan(FirstMessageActivity {
        active_pairs: 33,
        active_simd_iterations: 1,
    })
    .is_err());
}

#[test]
fn sparse_roofs_are_independent_of_the_heuristic_interpolation() {
    let total_iterations = TARGET_ROWS / (2 * SIMD_WIDTH as u64);
    let active_iterations =
        (total_iterations * SPARSE_SCREEN_TARGET_PRIORITY_PERMILLE + 999) / 1_000;
    let plan = target_work_plan(FirstMessageActivity {
        active_pairs: active_iterations,
        active_simd_iterations: active_iterations,
    })
    .unwrap();
    assert_eq!(active_iterations, 629_146);
    assert_eq!(plan.phases[0].simd_equivalent_product_slots, 122_368_896);
    assert_eq!(plan.simd_equivalent_product_slots(), 478_884_736);

    let first = plan.first_message_roof().unwrap();
    assert_eq!(first.compute_floor_ns, 3_784_996);
    assert_eq!(first.traffic_floor_ns, 2_378_902);
    assert_eq!(first.optimistic_floor_ns, 3_784_996);
    assert_eq!(first.eighty_percent_roof_bar_ns, 4_731_245);

    let prefix = plan.prefix_roof().unwrap();
    assert_eq!(prefix.compute_floor_ns, 14_812_396);
    assert_eq!(prefix.traffic_floor_ns, 14_274_559);
    assert_eq!(prefix.optimistic_floor_ns, 14_812_396);
    assert_eq!(prefix.eighty_percent_roof_bar_ns, 18_515_495);
}

#[test]
fn frozen_denominator_is_screening_provenance_not_promotion_input() {
    assert_eq!(
        median_of_five(FROZEN_CPU_SAMPLES_NS),
        Some(FROZEN_CPU_MEDIAN_NS)
    );
    assert_eq!(FIVE_X_SCREEN_CAP_NS, 46_931_375);
    assert_eq!(EIGHT_X_SCREEN_CAP_NS, 29_332_109);
    for value in [
        FROZEN_CPU_ARTIFACT,
        FROZEN_CPU_ARTIFACT_SHA256,
        FROZEN_CPU_REVISION,
        RETAINED_METAL_EVIDENCE,
        RETAINED_METAL_EVIDENCE_SHA256,
    ] {
        assert!(SCREENING_EVIDENCE_JSON.contains(value));
    }
    assert!(FROZEN_CPU_SAMPLE_SELECTOR.contains("RamValCheck"));
    assert!(SCREENING_EVIDENCE_JSON.contains("RamValCheck"));
    assert!(SCREENING_EVIDENCE_JSON.contains("screening_only"));
}

#[test]
fn sparse_interpolation_only_prioritizes_experiments() {
    assert_eq!(heuristic_first_message_ns(0).unwrap(), 2_377_104);
    assert_eq!(
        heuristic_hybrid_ns(SPARSE_SCREEN_TARGET_PRIORITY_PERMILLE).unwrap(),
        28_889_325
    );
    assert_eq!(
        sparse_screen_class(SPARSE_SCREEN_TARGET_PRIORITY_PERMILLE),
        Ok(SparseScreenClass::TargetScalePriority)
    );
    assert_eq!(
        sparse_screen_class(SPARSE_SCREEN_TARGET_PRIORITY_PERMILLE + 1),
        Ok(SparseScreenClass::ProxyFirst)
    );
    assert!(
        heuristic_hybrid_ns(SPARSE_SCREEN_PROXY_PRIORITY_PERMILLE).unwrap() > EIGHT_X_SCREEN_CAP_NS
    );
    assert_eq!(
        sparse_screen_class(SPARSE_SCREEN_PROXY_PRIORITY_PERMILLE),
        Ok(SparseScreenClass::LowPriority)
    );
}

fn valid_producer() -> ProducerEvidence {
    let dense = FirstMessageActivity::dense(Geometry::target());
    ProducerEvidence {
        kind: ProducerKind::SharedIncrementAccess,
        rows: TARGET_ROWS,
        row_bytes: 16,
        allocations: 1,
        rows_written: TARGET_ROWS,
        row_upload_bytes: 0,
        full_domain_copy_bytes: 0,
        full_domain_temporary_row_bytes: 0,
        streaming_scratch_peak_bytes: 0,
        prepare_storage_id: 17,
        ram_val_storage_id: 17,
        terminal_storage_id: 17,
        produced_before_piop: true,
        retained_through_stage7: true,
        semantics_checked: true,
        active_pairs: dense.active_pairs,
        active_simd_iterations: dense.active_simd_iterations,
    }
}

const PAIRED_SOURCE_REVISION: [u8; 20] = [0x17; 20];

fn speed_candidate(candidate_ns: u64) -> CandidateEvidence {
    CandidateEvidence {
        producer: Some(valid_producer()),
        independent_oracle_parity: true,
        output_claim_parity: true,
        clear_and_zk_proofs_verified: true,
        host_fiat_shamir_preserved: true,
        same_artifact_boundary: true,
        alternating_pair_order: true,
        paired_artifact_provenance_recorded: true,
        paired_piop_validation: true,
        paired_cpu_samples_ns: Some(FROZEN_CPU_SAMPLES_NS),
        paired_candidate_samples_ns: Some([candidate_ns; 5]),
        paired_source_revision: PAIRED_SOURCE_REVISION,
        activity_provenance: None,
        phase_latency_samples: None,
        compiled_capture: None,
    }
}

fn correct_candidate(candidate_ns: u64) -> CandidateEvidence {
    let producer = valid_producer();
    let roofs = target_work_plan(FirstMessageActivity::dense(Geometry::target()))
        .unwrap()
        .phase_roofs()
        .unwrap();
    CandidateEvidence {
        activity_provenance: Some(ActivityProvenance {
            source_revision: PAIRED_SOURCE_REVISION,
            artifact_sha256: [0x21; 32],
            trace_sha256: [0x31; 32],
            storage_id: producer.prepare_storage_id,
            rows: producer.rows,
            active_pairs: producer.active_pairs,
            active_simd_iterations: producer.active_simd_iterations,
        }),
        phase_latency_samples: Some(PhaseLatencySamples {
            first_message_ns: [roofs[0].eighty_percent_roof_bar_ns; 5],
            native_bind_and_message_ns: [roofs[1].eighty_percent_roof_bar_ns; 5],
            dense_transitions_ns: [roofs[2].eighty_percent_roof_bar_ns; 5],
        }),
        compiled_capture: Some(CompiledCaptureEvidence {
            source_revision: PAIRED_SOURCE_REVISION,
            binary_sha256: [0x41; 32],
            capture_sha256: [0x51; 32],
            phases: [CompiledPhaseResources {
                allocated_registers_per_thread: 64,
                resident_simdgroups_per_core: 4,
                required_resident_simdgroups_per_core: 4,
                spill_bytes: 0,
            }; 3],
        }),
        ..speed_candidate(candidate_ns)
    }
}

#[test]
fn admission_fails_closed_on_the_current_missing_producer() {
    assert_eq!(
        admission_decision(CandidateEvidence::default()),
        AdmissionDecision::RejectMissingProducer
    );

    let mut producer = valid_producer();
    producer.kind = ProducerKind::DedicatedRamValPack;
    let mut evidence = correct_candidate(40_000_000);
    evidence.producer = Some(producer);
    assert_eq!(
        admission_decision(evidence),
        AdmissionDecision::RejectProducer(ProducerRejection::WrongKind)
    );

    let mut producer = valid_producer();
    producer.full_domain_temporary_row_bytes = 16 * TARGET_ROWS;
    let mut evidence = correct_candidate(40_000_000);
    evidence.producer = Some(producer);
    assert_eq!(
        admission_decision(evidence),
        AdmissionDecision::RejectProducer(ProducerRejection::RepackedOrCopied)
    );

    let mut producer = valid_producer();
    producer.streaming_scratch_peak_bytes = 16 * TARGET_ROWS;
    let mut evidence = correct_candidate(40_000_000);
    evidence.producer = Some(producer);
    assert_eq!(
        admission_decision(evidence),
        AdmissionDecision::RejectProducer(ProducerRejection::RepackedOrCopied)
    );
}

#[test]
fn speed_screen_and_complete_promotion_are_distinct() {
    let screened = speed_candidate(29_000_000);
    assert_eq!(
        speed_screen_decision(screened),
        AdmissionDecision::PassEightX
    );
    assert_eq!(
        admission_decision(screened),
        AdmissionDecision::RejectMissingActivityProvenance
    );
}

#[test]
fn promotion_derives_the_first_phase_bar_from_recorded_activity() {
    let total_iterations = TARGET_ROWS / (2 * SIMD_WIDTH as u64);
    let active_iterations =
        (total_iterations * SPARSE_SCREEN_TARGET_PRIORITY_PERMILLE + 999) / 1_000;
    let mut evidence = correct_candidate(29_000_000);
    let producer = evidence.producer.as_mut().unwrap();
    producer.active_pairs = active_iterations;
    producer.active_simd_iterations = active_iterations;
    let provenance = evidence.activity_provenance.as_mut().unwrap();
    provenance.active_pairs = active_iterations;
    provenance.active_simd_iterations = active_iterations;

    let sparse_bar_ns = target_work_plan(FirstMessageActivity {
        active_pairs: active_iterations,
        active_simd_iterations: active_iterations,
    })
    .unwrap()
    .phase_roof(SuccessorPhase::FirstMessage)
    .unwrap()
    .eighty_percent_roof_bar_ns;
    let dense_median_ns = evidence.phase_latency_samples.unwrap().first_message_ns[2];
    assert_eq!(sparse_bar_ns, 4_731_245);
    assert_eq!(
        admission_decision(evidence),
        AdmissionDecision::RejectPhaseRoof(PhaseRoofRejection::ExceedsBar {
            phase: SuccessorPhase::FirstMessage,
            median_ns: dense_median_ns,
            bar_ns: sparse_bar_ns,
        })
    );

    evidence
        .phase_latency_samples
        .as_mut()
        .unwrap()
        .first_message_ns = [sparse_bar_ns; 5];
    assert_eq!(admission_decision(evidence), AdmissionDecision::PassEightX);
}

#[test]
fn promotion_fails_closed_on_activity_roofs_and_compiled_resources() {
    let mut evidence = correct_candidate(29_000_000);
    evidence.activity_provenance.as_mut().unwrap().active_pairs -= 1;
    assert_eq!(
        admission_decision(evidence),
        AdmissionDecision::RejectActivityProvenance(ActivityProvenanceRejection::WrongActivity)
    );

    let mut evidence = correct_candidate(29_000_000);
    evidence.phase_latency_samples = None;
    assert_eq!(
        admission_decision(evidence),
        AdmissionDecision::RejectMissingPhaseLatencyEvidence
    );

    let mut evidence = correct_candidate(29_000_000);
    let plan = target_work_plan(FirstMessageActivity::dense(Geometry::target())).unwrap();
    let bar_ns = plan
        .phase_roof(SuccessorPhase::NativeBindAndMessage)
        .unwrap()
        .eighty_percent_roof_bar_ns;
    evidence
        .phase_latency_samples
        .as_mut()
        .unwrap()
        .native_bind_and_message_ns = [bar_ns + 1; 5];
    assert_eq!(
        admission_decision(evidence),
        AdmissionDecision::RejectPhaseRoof(PhaseRoofRejection::ExceedsBar {
            phase: SuccessorPhase::NativeBindAndMessage,
            median_ns: bar_ns + 1,
            bar_ns,
        })
    );

    let mut evidence = correct_candidate(29_000_000);
    evidence.compiled_capture = None;
    assert_eq!(
        admission_decision(evidence),
        AdmissionDecision::RejectMissingCompiledCapture
    );

    let mut evidence = correct_candidate(29_000_000);
    evidence.compiled_capture.as_mut().unwrap().phases[0].allocated_registers_per_thread = 0;
    assert_eq!(
        admission_decision(evidence),
        AdmissionDecision::RejectCompiledCapture(CompiledCaptureRejection::MissingRegisterCount {
            phase: SuccessorPhase::FirstMessage,
        })
    );

    let mut evidence = correct_candidate(29_000_000);
    evidence.compiled_capture.as_mut().unwrap().phases[2].resident_simdgroups_per_core = 3;
    assert_eq!(
        admission_decision(evidence),
        AdmissionDecision::RejectCompiledCapture(
            CompiledCaptureRejection::ResidencyBelowRequirement {
                phase: SuccessorPhase::DenseTransitions,
                required: 4,
                got: 3,
            }
        )
    );

    let mut evidence = correct_candidate(29_000_000);
    evidence.compiled_capture.as_mut().unwrap().phases[1].spill_bytes = 16;
    assert_eq!(
        admission_decision(evidence),
        AdmissionDecision::RejectCompiledCapture(CompiledCaptureRejection::DeviceMemorySpill {
            phase: SuccessorPhase::NativeBindAndMessage,
            bytes: 16,
        })
    );
}

#[test]
fn promotion_uses_current_paired_medians_after_complete_gates() {
    assert_eq!(
        admission_decision(correct_candidate(40_000_000)),
        AdmissionDecision::PassFiveXPursueEightX
    );
    assert_eq!(
        admission_decision(correct_candidate(29_000_000)),
        AdmissionDecision::PassEightX
    );
    assert_eq!(
        admission_decision(correct_candidate(50_000_000)),
        AdmissionDecision::RejectBelowFiveX
    );

    let mut current_pair = correct_candidate(29_000_000);
    current_pair.paired_cpu_samples_ns = Some([100_000_000; 5]);
    assert_eq!(
        admission_decision(current_pair),
        AdmissionDecision::RejectBelowFiveX
    );
}

#[test]
fn first_shader_slice_is_fixed_width_sparse_and_fail_closed() {
    let shader = include_str!("shader.metal");
    assert!(shader.contains(FIRST_MESSAGE_PIPELINE));
    assert!(shader.contains("simd_any(pair_active)"));
    assert!(shader.contains("RAM_VAL_SUCCESSOR_STATUS_INVALID_ROW"));
    assert!(shader.contains("threads == RAM_VAL_SUCCESSOR_SIMD_WIDTH"));
    assert!(shader.contains("output_index >= params.output_count"));
    assert!(!shader.contains("ram_address_present"));
    assert!(!shader.contains("threadgroup SolinasFp128* shared"));
}
