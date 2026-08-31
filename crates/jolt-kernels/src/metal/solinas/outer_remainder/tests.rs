use std::mem::size_of;

use super::{
    api::{OuterRemainderPhase, OuterRemainderSequenceConfig, OuterRemainderStorageInitialization},
    plan::{
        field_bytes, message_threadgroup_bytes, opening_layout, opening_threadgroup_memory_lengths,
        outer_remainder_sequence_max_buffer_bytes_with_config,
        outer_remainder_sequence_storage_bytes_with_config, storage_geometry,
    },
    sequence::{OpeningParams, PhaseParams, ReduceParams},
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
    let layout = opening_layout();
    assert_eq!(layout.tile_rows, 64);
    assert_eq!(layout.source_row_words, 20);
    assert_eq!(layout.row_stride_words, 20);
    assert!(layout.shard_sums);

    let lengths = opening_threadgroup_memory_lengths(256, false).unwrap();
    assert_eq!(lengths, [10_240, 1_024, 3_920]);
    assert_eq!(lengths.into_iter().sum::<u64>(), 15_184);

    let carrier_lengths = opening_threadgroup_memory_lengths(256, true).unwrap();
    assert_eq!(carrier_lengths, [10_240, 1_024, 3_552]);
}

#[test]
fn abi_params_have_stable_sizes() {
    assert_eq!(size_of::<PhaseParams>(), 16);
    assert_eq!(size_of::<OpeningParams>(), 32);
    assert_eq!(size_of::<ReduceParams>(), 16);
}

#[test]
fn log_26_storage_uses_in_place_stream_bind_capacity() {
    let config = OuterRemainderSequenceConfig::default();
    let geometry = storage_geometry(1 << 26, config).unwrap();

    assert_eq!(geometry.current_elements, 1 << 27);
    assert_eq!(geometry.weight_capacity, 1 << 13);
    assert_eq!(geometry.max_threadgroups, 8192);
    assert_eq!(
        geometry.element_counts,
        [
            1 << 27,
            1 << 26,
            1 << 13,
            1 << 13,
            230,
            16_384,
            2,
            286_720,
            35
        ],
    );
    assert_eq!(geometry.owned_bytes, 3_226_341_552);
    assert_eq!(field_bytes(1 << 27).unwrap(), 1 << 31);
    assert_eq!(
        outer_remainder_sequence_storage_bytes_with_config(1 << 26, config).unwrap(),
        3_226_341_552,
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
    assert_eq!(geometry.owned_bytes, 1_615_728_816);
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
