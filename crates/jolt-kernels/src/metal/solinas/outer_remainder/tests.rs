use std::mem::size_of;

use super::super::Fp128;
use super::{
    api::{
        OuterRemainderPhase, OuterRemainderSequenceConfig, OuterRemainderStorageInitialization,
        OUTER_REMAINDER_OPENINGS,
    },
    plan::{
        field_bytes, message_threadgroup_bytes,
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
fn opening_tile_memory_is_below_16_kib_at_256_threads() {
    let shards = 256 / OUTER_REMAINDER_OPENINGS;
    let bytes = 64 * 20 * size_of::<u64>()
        + 64 * size_of::<Fp128>()
        + OUTER_REMAINDER_OPENINGS * shards * size_of::<Fp128>();
    assert_eq!(shards, 7);
    assert!(bytes < 16 * 1024);
}

#[test]
fn abi_params_are_four_words() {
    assert_eq!(size_of::<PhaseParams>(), 16);
    assert_eq!(size_of::<OpeningParams>(), 16);
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
            10,
            16_384,
            2,
            286_720,
            35
        ],
    );
    assert_eq!(geometry.owned_bytes, 4_300_079_856);
    assert_eq!(field_bytes(1 << 27).unwrap(), 1 << 31);
    assert_eq!(
        outer_remainder_sequence_storage_bytes_with_config(1 << 26, config).unwrap(),
        4_300_079_856,
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
    assert_eq!(geometry.owned_bytes, 2_152_596_208);
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
