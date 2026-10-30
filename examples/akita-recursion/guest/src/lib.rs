#![cfg(feature = "guest")]

use jolt_akita::deferred::SetupMode;
use jolt_akita::AkitaField;
use jolt_akita::TrustedBytes;
use jolt_akita_recursion::{decode_pair, AnswerTape, ReplayEvent};

include!(concat!(env!("OUT_DIR"), "/material.rs"));

fn observe(event: ReplayEvent) {
    match event {
        ReplayEvent::PrepareStart => jolt::start_cycle_tracking("prepare_akita_setup"),
        ReplayEvent::PrepareEnd => jolt::end_cycle_tracking("prepare_akita_setup"),
        ReplayEvent::InnerStart(index) => {
            jolt::start_cycle_tracking(["inner0_jolt", "inner1_jolt"][index])
        }
        ReplayEvent::InnerEnd(index) => {
            jolt::end_cycle_tracking(["inner0_jolt", "inner1_jolt"][index])
        }
        ReplayEvent::AkitaStart(index) => {
            jolt::start_cycle_tracking(["inner0_akita", "inner1_akita"][index])
        }
        ReplayEvent::AkitaEnd(index) => {
            jolt::end_cycle_tracking(["inner0_akita", "inner1_akita"][index])
        }
    }
}

#[expect(
    clippy::expect_used,
    reason = "the benchmark guest panics on rejected artifacts and the host checks its panic flag"
)]
fn execute(bytes: &[u8], mode: SetupMode) -> Vec<u8> {
    jolt::start_cycle_tracking("decode_pair");
    let pair = decode_pair(bytes);
    jolt::end_cycle_tracking("decode_pair");
    jolt::start_cycle_tracking("verify_pair");
    jolt::start_cycle_tracking("install_prepared_setup");
    assert!(
        !KEY.0.is_empty(),
        "prepare and embed the verifier material before tracing"
    );
    // SAFETY: both payloads are exported by the native seed-derived setup and
    // embedded by build.rs, so canonical field words are fixed by this image.
    let (key, terminal) = unsafe { (TrustedBytes::new(&KEY.0), TrustedBytes::new(&TERMINAL.0)) };
    pair.preprocessing
        .pcs_setup
        .install_recursion_material(&pair.inner[0].proof.joint_opening_proof, key, terminal)
        .expect("install program-bound verifier material");
    jolt::end_cycle_tracking("install_prepared_setup");
    let statement = {
        let mut advice = AnswerTape(pair.answers.iter());
        let statement = pair
            .replay(mode, &mut advice, observe)
            .expect("inner proof replay");
        if mode == SetupMode::Defer {
            assert!(advice.0.next().is_none(), "unused setup advice");
        }
        statement
    };
    jolt::end_cycle_tracking("verify_pair");
    jolt::start_cycle_tracking("export_setup_claims");
    let output = match mode {
        SetupMode::Defer => statement
            .descriptor_bytes::<AkitaField>()
            .expect("setup statement"),
        SetupMode::Check => {
            assert!(statement.claims().is_empty());
            Vec::new()
        }
    };
    jolt::end_cycle_tracking("export_setup_claims");
    jolt::start_cycle_tracking("drop_pair");
    drop(statement);
    drop(pair);
    jolt::end_cycle_tracking("drop_pair");
    output
}

#[jolt::provable(
    max_input_size = 67108864,
    max_output_size = 1048576,
    heap_size = 2147483648,
    stack_size = 33554432,
    max_trace_length = 68719476736
)]
fn verify_deferred(bytes: &[u8]) -> Vec<u8> {
    execute(bytes, SetupMode::Defer)
}

#[jolt::provable(
    max_input_size = 67108864,
    max_output_size = 1048576,
    heap_size = 2147483648,
    stack_size = 33554432,
    max_trace_length = 68719476736
)]
fn verify_before(bytes: &[u8]) -> Vec<u8> {
    execute(bytes, SetupMode::Check)
}
