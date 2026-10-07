//! Retained-statement parity of the byte link (spec §4): a flag-off and a
//! flag-on build must both reproduce the frozen record. Over the same tape the
//! flag-on build also checks that the final opening binds the histogram
//! commitments. Regenerate the record on a flag-off build only:
//!
//! ```text
//! JOLT_PARITY_BLESS=1 cargo nextest run -p jolt-verifier \
//!   --features akita,prover-fixtures,fs-audit --test byte_link_parity
//! ```

#![cfg(all(feature = "akita", feature = "prover-fixtures", feature = "fs-audit"))]
#![expect(
    dead_code,
    clippy::expect_used,
    reason = "the shared support module is only partially used per feature configuration; \
              a missing or unwritable record must fail loudly"
)]

mod support;

use std::{env, fs, path::PathBuf};

#[cfg(feature = "akita-byte-link")]
use jolt_akita::AkitaCommitment;
#[cfg(feature = "akita-byte-link")]
use jolt_verifier::VerifierError;
use support::parity::parity_record;
#[cfg(feature = "akita-byte-link")]
use support::parity::{tape_case, verify_over_tape};

const RECORD: &str = "tests/byte_link_parity.record";

#[test]
fn retained_s1_to_s7_statements_match_the_frozen_record() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(RECORD);
    let actual = parity_record().lines();
    let bless = env::var_os("JOLT_PARITY_BLESS").is_some();
    assert!(
        !(bless && cfg!(feature = "akita-byte-link")),
        "the parity record is blessed from the flag-off build only"
    );
    if bless {
        let header = "# Byte-link parity record (spec §4); regenerate with JOLT_PARITY_BLESS=1 \
                      on a flag-off build (tests/byte_link_parity.rs).\n";
        fs::write(&path, header.to_owned() + &actual.join("\n") + "\n")
            .expect("write the parity record");
        return;
    }
    let frozen = fs::read_to_string(&path).expect("read the parity record");
    let expected = frozen
        .lines()
        .filter(|line| !line.starts_with('#'))
        .collect::<Vec<_>>();
    assert_eq!(
        actual.len(),
        expected.len(),
        "the parity record covers a different stage set"
    );
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(actual, expected, "retained parity broken");
    }
}

/// Over the tape no challenge depends on an absorbed byte, so a histogram
/// commitment that no longer matches the link's `W` evaluations can only fail
/// at the final opening.
#[cfg(feature = "akita-byte-link")]
#[test]
fn the_final_opening_binds_the_histogram_commitments() {
    let case = tape_case();
    verify_over_tape(&case, &case.proof).expect("the tape proof verifies");
    for group in 0..2 {
        let mut proof = case.proof.clone();
        let commitment = &mut proof.byte_link.histogram_commitments[group];
        *commitment = with_first_coefficient_bit_flipped(commitment);
        assert!(
            matches!(
                verify_over_tape(&case, &proof),
                Err(VerifierError::FinalOpeningVerificationFailed { .. })
            ),
            "histogram group {group}"
        );
    }
}

#[cfg(feature = "akita-byte-link")]
fn with_first_coefficient_bit_flipped(commitment: &AkitaCommitment) -> AkitaCommitment {
    let mut value = serde_json::to_value(commitment).expect("serialize the commitment");
    let byte = &mut value["serialized_backend_bytes"][0];
    *byte = (byte.as_u64().expect("a backend commitment byte") ^ 1).into();
    serde_json::from_value(value).expect("decode the perturbed commitment")
}
