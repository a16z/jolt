#![cfg(feature = "guest")]
use jolt_akita::deferred::SetupMode;
use jolt_akita::{AkitaField, TrustedBytes};
use jolt_akita_external::pair_api::{AnswerTape, ReplayEvent};
use jolt_akita_external::{decode_exact, PairAdvice};
include!(concat!(env!("OUT_DIR"), "/material.rs"));
fn observe(event: ReplayEvent) {
    match event {
        ReplayEvent::PrepareStart => jolt::start_cycle_tracking("prepare_akita_setup"),
        ReplayEvent::PrepareEnd => jolt::end_cycle_tracking("prepare_akita_setup"),
        ReplayEvent::InnerStart(0) => jolt::start_cycle_tracking("inner0_jolt"),
        ReplayEvent::InnerEnd(0) => jolt::end_cycle_tracking("inner0_jolt"),
        ReplayEvent::AkitaStart(0) => jolt::start_cycle_tracking("inner0_akita"),
        ReplayEvent::AkitaEnd(0) => jolt::end_cycle_tracking("inner0_akita"),
        ReplayEvent::InnerStart(1) => jolt::start_cycle_tracking("inner1_jolt"),
        ReplayEvent::InnerEnd(1) => jolt::end_cycle_tracking("inner1_jolt"),
        ReplayEvent::AkitaStart(1) => jolt::start_cycle_tracking("inner1_akita"),
        ReplayEvent::AkitaEnd(1) => jolt::end_cycle_tracking("inner1_akita"),
        _ => unreachable!(),
    }
}
#[jolt::provable(
    max_input_size = 4096,
    max_untrusted_advice_size = 67108864,
    max_output_size = 1048576,
    heap_size = 1073741824,
    stack_size = 33554432,
    max_trace_length = 68719476736
)]
fn verify_advice(bytes: jolt::UntrustedAdvice<Vec<u8>>) -> Vec<u8> {
    let _ = INHERITED_FIELDS;
    jolt::start_cycle_tracking("decode_pair");
    let advice: PairAdvice = decode_exact(&bytes).expect("pair advice");
    let verifier = decode_exact(&VERIFIER.0).expect("fixed inner verifier");
    let pair = advice.with_verifier(verifier);
    jolt::end_cycle_tracking("decode_pair");
    jolt::start_cycle_tracking("verify_pair");
    jolt::start_cycle_tracking("install_prepared_setup");
    // SAFETY: canonical seed-derived material and the verifier are bound by this committed guest.
    let (key, terminal) = unsafe { (TrustedBytes::new(&KEY.0), TrustedBytes::new(&TERMINAL.0)) };
    pair.preprocessing
        .pcs_setup
        .install_recursion_material(&pair.inner[0].proof.joint_opening_proof, key, terminal)
        .expect("inner material");
    jolt::end_cycle_tracking("install_prepared_setup");
    let mut answers = AnswerTape(pair.answers.iter());
    let statement = pair
        .replay(SetupMode::Defer, &mut answers, observe)
        .expect("inner deferred verification");
    assert!(answers.0.next().is_none(), "unused answers");
    assert_eq!(statement.claims().len(), EXPECTED_PENDING, "complete A");
    jolt::end_cycle_tracking("verify_pair");
    jolt::start_cycle_tracking("export_setup_claims");
    let output = statement
        .descriptor_bytes::<AkitaField>()
        .expect("canonical A");
    jolt::end_cycle_tracking("export_setup_claims");
    jolt::start_cycle_tracking("drop_pair");
    drop(statement);
    drop(pair);
    jolt::end_cycle_tracking("drop_pair");
    output
}
