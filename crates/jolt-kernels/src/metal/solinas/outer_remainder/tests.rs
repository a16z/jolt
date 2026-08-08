use std::mem::size_of;

use super::{
    api::{OuterRemainderPhase, OuterRemainderSequenceConfig, OuterRemainderStorageInitialization},
    plan::{
        field_bytes, message_threadgroup_bytes, opening_layout, opening_threadgroup_memory_lengths,
        outer_remainder_sequence_max_buffer_bytes_with_config,
        outer_remainder_sequence_storage_bytes_with_config, storage_geometry,
    },
    sequence::{OpeningParams, PhaseParams, ReduceParams},
    OuterBindingPlan,
};

#[test]
fn default_schedule_reaches_the_log_26_tail_in_nine_transitions() {
    let config = OuterRemainderSequenceConfig::default();
    let mut elements = 1usize << 27;
    let mut transitions = 0;
    while elements > config.cpu_tail_elements {
        elements /= 2;
        transitions += 1;
    }
    assert_eq!(elements, 1 << 18);
    assert_eq!(transitions, 9);
}

#[test]
fn initial_log_26_gruen_shape_excludes_the_active_variable() {
    let current_elements = 1usize << 27;
    let e_in = 1usize << 13;
    let e_out = 1usize << 13;
    assert_eq!(e_in * e_out * 2, current_elements);
}

#[test]
fn opening_layouts_close_their_dynamic_threadgroup_memory() {
    let legacy_layout = opening_layout(OuterBindingPlan::BOnlyV1);
    assert_eq!(legacy_layout.tile_rows, 64);
    assert_eq!(legacy_layout.source_row_words, 20);
    assert_eq!(legacy_layout.row_stride_words, 20);
    assert!(legacy_layout.shard_sums);

    let padded_layout = opening_layout(OuterBindingPlan::BOnlyPadded56V1);
    assert_eq!(padded_layout.tile_rows, 56);
    assert_eq!(padded_layout.source_row_words, 20);
    assert_eq!(padded_layout.row_stride_words, 21);
    assert!(!padded_layout.shard_sums);

    let legacy =
        opening_threadgroup_memory_lengths(OuterBindingPlan::BOnlyV1, 256, false, false).unwrap();
    let padded =
        opening_threadgroup_memory_lengths(OuterBindingPlan::BOnlyPadded56V1, 256, false, false)
            .unwrap();

    assert_eq!(legacy, [10_240, 1_024, 3_920]);
    assert_eq!(legacy.into_iter().sum::<u64>(), 15_184);
    assert_eq!(padded, [9_408, 896, 0]);
    assert_eq!(padded.into_iter().sum::<u64>(), 10_304);

    let carrier =
        opening_threadgroup_memory_lengths(OuterBindingPlan::BOnlyV1, 256, true, false).unwrap();
    assert_eq!(carrier, [10_240, 1_024, 3_552]);

    let registers =
        opening_threadgroup_memory_lengths(OuterBindingPlan::BOnlyV1, 256, true, true).unwrap();
    assert_eq!(registers, carrier);
}

#[test]
fn default_sequence_keeps_the_legacy_binding_plan() {
    assert_eq!(
        OuterRemainderSequenceConfig::default().binding_plan,
        OuterBindingPlan::BOnlyV1,
    );
}

#[test]
fn abi_params_have_stable_sizes() {
    assert_eq!(size_of::<PhaseParams>(), 16);
    assert_eq!(size_of::<OpeningParams>(), 32);
    assert_eq!(size_of::<ReduceParams>(), 16);
}

#[test]
fn log_26_storage_has_two_two_gib_state_buffers() {
    let config = OuterRemainderSequenceConfig::default();
    let geometry = storage_geometry(1 << 26, config).unwrap();

    assert_eq!(geometry.current_elements, 1 << 27);
    assert_eq!(geometry.weight_capacity, 1 << 13);
    assert_eq!(geometry.max_threadgroups, 8192);
    assert_eq!(
        geometry.element_counts,
        [
            1 << 27,
            1 << 27,
            1 << 13,
            1 << 13,
            202,
            16_384,
            2,
            286_720,
            35
        ],
    );
    assert_eq!(geometry.owned_bytes, 4_300_082_928);
    assert_eq!(field_bytes(1 << 27).unwrap(), 1 << 31);
    assert_eq!(
        outer_remainder_sequence_storage_bytes_with_config(1 << 26, config).unwrap(),
        4_300_082_928,
    );
    assert_eq!(
        outer_remainder_sequence_max_buffer_bytes_with_config(1 << 26, config).unwrap(),
        1 << 31,
    );
    assert_eq!(message_threadgroup_bytes(256), 256);
}

#[test]
fn log_25_screen_retains_the_saturated_threadgroup_cap() {
    let config = OuterRemainderSequenceConfig::default();
    let geometry = storage_geometry(1 << 25, config).unwrap();

    assert_eq!(geometry.current_elements, 1 << 26);
    assert_eq!(geometry.weight_capacity, 1 << 13);
    assert_eq!(geometry.max_threadgroups, 8192);
    assert_eq!(geometry.owned_bytes, 2_152_599_280);
    assert_eq!(
        outer_remainder_sequence_max_buffer_bytes_with_config(1 << 25, config).unwrap(),
        1 << 30,
    );
}

#[test]
fn phase_names_are_stable_diagnostics() {
    assert_eq!(
        OuterRemainderPhase::BeforeMaterialize.name(),
        "before materialization",
    );
    assert_eq!(OuterRemainderPhase::BOnly.name(), "B-only");
    assert_eq!(OuterRemainderPhase::Interleaved.name(), "interleaved");
    assert_eq!(OuterRemainderPhase::Exported.name(), "CPU tail exported");
    assert_eq!(
        OuterRemainderPhase::OpeningsComplete.name(),
        "openings complete",
    );
    assert_eq!(OuterRemainderPhase::Poisoned.name(), "poisoned");
    assert_eq!(OuterRemainderStorageInitialization::Lazy.as_str(), "lazy");
    assert_eq!(OuterRemainderStorageInitialization::Full.as_str(), "full");
}
