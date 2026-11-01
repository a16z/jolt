#![cfg(feature = "guest")]
use jolt::UntrustedAdvice;
use jolt_akita::deferred::SetupMode;
use jolt_akita::{AkitaField, TrustedBytes};
use jolt_akita_external::{decode_exact, Event, OuterAdvice, Tape};
include!(concat!(env!("OUT_DIR"), "/material.rs"));
fn observe(event: Event) {
    match event {
        Event::PrepareStart => jolt::start_cycle_tracking("prepare_inherited_batch"),
        Event::PrepareEnd => jolt::end_cycle_tracking("prepare_inherited_batch"),
        Event::JoltStart => jolt::start_cycle_tracking("outer_jolt"),
        Event::JoltEnd => jolt::end_cycle_tracking("outer_jolt"),
        Event::AkitaStart => jolt::start_cycle_tracking("outer_akita"),
        Event::AkitaEnd => jolt::end_cycle_tracking("outer_akita"),
        Event::RebindStart => jolt::start_cycle_tracking("rebind_pending_claims"),
        Event::RebindEnd => jolt::end_cycle_tracking("rebind_pending_claims"),
    }
}
#[jolt::provable(
    max_input_size = 4096,
    max_untrusted_advice_size = 67108864,
    max_output_size = 1048576,
    heap_size = 268435456,
    stack_size = 33554432,
    max_trace_length = 68719476736
)]
fn verify_external(bytes: UntrustedAdvice<Vec<u8>>) -> Vec<u8> {
    jolt::start_cycle_tracking("decode_external");
    let advice: OuterAdvice = decode_exact(&bytes).expect("external advice");
    let verifier = decode_exact(&VERIFIER.0).expect("fixed outer verifier");
    let input = advice.with_verifier(verifier, INHERITED_FIELDS);
    jolt::end_cycle_tracking("decode_external");
    jolt::start_cycle_tracking("install_prepared_setup");
    assert!(!KEY.0.is_empty(), "missing prepared verifier material");
    // SAFETY: these canonical field bytes are exported from the seed-derived
    // verifier setup and are authenticated by this guest's committed image.
    let (key, terminal) = unsafe { (TrustedBytes::new(&KEY.0), TrustedBytes::new(&TERMINAL.0)) };
    input
        .preprocessing
        .pcs_setup
        .install_recursion_material(&input.outer.proof.joint_opening_proof, key, terminal)
        .expect("outer verifier material");
    jolt::end_cycle_tracking("install_prepared_setup");
    let mut tape = Tape(input.answers.iter());
    let statement = input
        .replay(SetupMode::Defer, &mut tape, observe)
        .expect("saved outer deferred verification");
    assert!(tape.0.next().is_none(), "unused setup advice");
    assert_eq!(
        statement.claims().len(),
        EXPECTED_PENDING,
        "pending direct-fold claims"
    );
    jolt::start_cycle_tracking("export_setup_claims");
    let output = statement
        .descriptor_bytes::<AkitaField>()
        .expect("external pending statement");
    jolt::end_cycle_tracking("export_setup_claims");
    jolt::start_cycle_tracking("drop_external");
    drop(statement);
    drop(input);
    jolt::end_cycle_tracking("drop_external");
    output
}
