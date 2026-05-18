use crate::{
    soundness::tampering,
    support,
    support::dory_pedersen,
    support::{soundness_expectation, HarnessExpectation},
};

#[test]
fn public_io_memory_layout_mismatch_rejects_now() {
    let mut case = dory_pedersen::standard_case();
    case.public_io.memory_layout.heap_size += 1;

    support::assert_rejects_at_or_before_current_frontier(case.verify());
}

#[test]
fn invalid_trace_length_rejects_now() {
    let mut case = dory_pedersen::standard_case();
    case.proof.trace_length = 3;

    support::assert_rejects_at_or_before_current_frontier(case.verify());
}

#[test]
fn excessive_trace_length_rejects_now() {
    let mut case = dory_pedersen::standard_case();
    case.proof.trace_length = case.preprocessing.program.max_padded_trace_length * 2;

    support::assert_rejects_at_or_before_current_frontier(case.verify());
}

#[test]
fn invalid_ram_domain_rejects_now() {
    let mut case = dory_pedersen::standard_case();
    case.proof.ram_K = 3;

    support::assert_rejects_at_or_before_current_frontier(case.verify());
}

#[test]
fn zero_ram_domain_rejects_now() {
    let mut case = dory_pedersen::standard_case();
    case.proof.ram_K = 0;

    support::assert_rejects_at_or_before_current_frontier(case.verify());
}

#[test]
fn oversized_public_input_rejects_now() {
    let mut case = dory_pedersen::standard_case();
    case.public_io.inputs =
        vec![0; case.preprocessing.program.memory_layout.max_input_size as usize + 1];

    support::assert_rejects_at_or_before_current_frontier(case.verify());
}

#[test]
fn oversized_public_output_rejects_now() {
    let mut case = dory_pedersen::standard_case();
    case.public_io.outputs =
        vec![0; case.preprocessing.program.memory_layout.max_output_size as usize + 1];

    support::assert_rejects_at_or_before_current_frontier(case.verify());
}

#[test]
#[ignore = "direct verifier-input tampering fixtures are not wired yet"]
fn tampered_public_input_bytes_reject() {
    assert_eq!(
        soundness_expectation(tampering::PUBLIC_INPUT_BYTES),
        HarnessExpectation::FutureCheckpoint,
    );
}
