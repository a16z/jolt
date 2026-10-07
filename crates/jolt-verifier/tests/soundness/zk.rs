//! ZK Dory fixtures: every stage sends hiding commitments, so the clear
//! stage checks move to the Stage 8 opening and the BlindFold tail.

#![expect(
    clippy::expect_used,
    reason = "tamper tests fail loudly when a fixture breaks an assumption"
)]

use super::checks;
use crate::support::narg::{Region, TracedCase};
use crate::support::verifier_fixtures::{
    fresh_zk_muldiv_case, zk_advice_consumer_case, zk_committed_muldiv_case, zk_muldiv_case,
};
use jolt_verifier::VerifierError;

const SAMPLE: Option<usize> = Some(300);

#[test]
fn zk_muldiv_message_sweep() {
    checks::message_sweep(&zk_muldiv_case(), None);
}

#[test]
fn zk_muldiv_structural_tampers() {
    checks::structural_tampers(&zk_muldiv_case(), Some(128));
}

#[test]
fn zk_muldiv_statement_tampers() {
    let case = zk_muldiv_case();
    checks::statement_tampers(&case);
    checks::header_equivocations(&case);
    checks::commitment_substitutions(&case);
    checks::preprocessing_swap(&case, &zk_committed_muldiv_case().preprocessing);

    let mut preprocessing = case.preprocessing.clone();
    preprocessing.vc_setup = None;
    assert!(matches!(
        case.verify_statement_with(&preprocessing),
        Err(VerifierError::MissingVectorCommitmentSetup)
    ));
}

#[test]
fn zk_advice_message_sweep() {
    let case = zk_advice_consumer_case();
    checks::message_sweep(&case, SAMPLE);
    checks::trusted_advice_tampers(&case, &checks::sent_commitment(&case, 0));
}

#[test]
fn zk_committed_message_sweep() {
    let case = zk_committed_muldiv_case();
    checks::message_sweep(&case, SAMPLE);
    checks::committed_program_tampers(
        &case,
        &[("swapped bytecode chunk commitments", |preprocessing| {
            let committed = checks::committed_mut(preprocessing);
            assert!(committed.bytecode_chunk_commitments.len() >= 2);
            committed.bytecode_chunk_commitments.swap(0, 1);
        })],
    );
}

/// Two honest ZK proofs of one statement share every message width, so a
/// region transplanted from one into the other is a well-formed argument
/// string whose messages are each valid but bound to another transcript.
#[test]
fn zk_region_transplants_reject() {
    let base = zk_muldiv_case();
    let donor = fresh_zk_muldiv_case();
    let base_messages = base.honest_trace().messages();
    assert_eq!(
        base_messages,
        donor.honest_trace().messages(),
        "fresh proofs of one statement have one message layout"
    );
    for region in Region::expected() {
        if region == Region::Preamble {
            continue;
        }
        let deadline = base_messages
            .iter()
            .filter(|m| m.region == region)
            .map(|m| m.deadline)
            .max()
            .expect("every region sends a message");
        let mut narg = base.proof.narg.clone();
        for message in base_messages.iter().filter(|m| m.region == region) {
            narg[message.range.clone()].copy_from_slice(&donor.proof.narg[message.range.clone()]);
        }
        assert_ne!(
            narg, base.proof.narg,
            "{region:?} transplant changed nothing"
        );
        let trace = base.trace(&narg);
        assert!(trace.result.is_err(), "{region:?} transplant accepted");
        assert!(
            trace.stop_region() <= deadline,
            "{region:?} transplant rejected in {:?}: {:?}",
            trace.stop_region(),
            trace.result
        );
    }
}
