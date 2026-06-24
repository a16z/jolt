// #![cfg_attr(feature = "guest", no_std)]

use jolt_sdk::{self as jolt};

extern crate alloc;

use jolt::{JoltDevice, JoltVerifierPreprocessing, RV64IMACProof};
use serde::de::DeserializeOwned;

use jolt::{end_cycle_tracking, start_cycle_tracking};

mod embedded_bytes {
    include!("./embedded_bytes.rs");
}

include!("./provable_macro.rs");

fn read_record<T: DeserializeOwned>(buffer: &[u8], offset: &mut usize) -> T {
    assert!(
        buffer.len().saturating_sub(*offset) >= 8,
        "missing record length prefix"
    );
    let mut len_bytes = [0u8; 8];
    len_bytes.copy_from_slice(&buffer[*offset..*offset + 8]);
    *offset += 8;

    let len = usize::try_from(u64::from_le_bytes(len_bytes)).unwrap();
    assert!(
        buffer.len().saturating_sub(*offset) >= len,
        "truncated serialized record"
    );
    let end = *offset + len;
    let (value, consumed) =
        bincode::serde::decode_from_slice(&buffer[*offset..end], bincode::config::standard())
            .unwrap();
    assert_eq!(consumed, len, "record decoder left trailing bytes");
    *offset = end;
    value
}

provable_with_config! {
fn verify(bytes: &[u8]) -> u32 {
    let use_embedded = !embedded_bytes::EMBEDDED_BYTES.is_empty();
    let data_bytes = if use_embedded {
        embedded_bytes::EMBEDDED_BYTES
    } else {
        bytes
    };

    let mut offset = 0;

    start_cycle_tracking("deserialize preprocessing");
    let verifier_preprocessing: JoltVerifierPreprocessing = read_record(data_bytes, &mut offset);
    end_cycle_tracking("deserialize preprocessing");

    start_cycle_tracking("deserialize count of proofs");
    // Deserialize number of proofs to verify
    let n: u32 = read_record(data_bytes, &mut offset);
    end_cycle_tracking("deserialize count of proofs");

    let mut all_valid = true;
    for _ in 0..n {
        start_cycle_tracking("deserialize proof");
        let proof: RV64IMACProof = read_record(data_bytes, &mut offset);
        end_cycle_tracking("deserialize proof");

        start_cycle_tracking("deserialize device");
        let device: JoltDevice = read_record(data_bytes, &mut offset);
        end_cycle_tracking("deserialize device");

        start_cycle_tracking("verification");
        let is_valid = jolt::jolt_verifier::verify::<
            jolt::VerifierField,
            jolt::VerifierPCS,
            jolt::VerifierVC,
            jolt::VerifierTranscript,
        >(&verifier_preprocessing, &device, &proof, None, false)
        .is_ok();
        end_cycle_tracking("verification");
        all_valid = all_valid && is_valid;
    }

    all_valid as u32
}
}
