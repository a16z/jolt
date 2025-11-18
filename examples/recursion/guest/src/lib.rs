// #![cfg_attr(feature = "guest", no_std)]

use jolt_sdk::{self as jolt};

extern crate alloc;

use ark_serialize::{CanonicalDeserialize, Compress, Validate};
use jolt::{JoltDevice, JoltVerifierPreprocessing, RV64IMACProof, RV64IMACVerifier, F, PCS};

use jolt::{end_cycle_tracking, start_cycle_tracking};

mod embedded_bytes {
    include!("./embedded_bytes.rs");
}

include!("./provable_macro.rs");

provable_with_config! {
fn verify(bytes: &[u8]) -> u32 {
    let use_embedded = !embedded_bytes::EMBEDDED_BYTES.is_empty();
    let data_bytes = if use_embedded {
        embedded_bytes::EMBEDDED_BYTES
    } else {
        bytes
    };

    let mut cursor = std::io::Cursor::new(data_bytes);

    start_cycle_tracking("deserialize preprocessing");
    let verifier_preprocessing: JoltVerifierPreprocessing<F, PCS> =
        JoltVerifierPreprocessing::<F, PCS>::deserialize_with_mode(&mut cursor, Compress::Yes, Validate::No).unwrap();
    end_cycle_tracking("deserialize preprocessing");

    start_cycle_tracking("deserialize count of proofs");
    // Deserialize number of proofs to verify
    let n: u32 = u32::deserialize_with_mode(&mut cursor, Compress::Yes, Validate::No).unwrap();
    end_cycle_tracking("deserialize count of proofs");

    let mut all_valid = true;
    for _ in 0..n {
        start_cycle_tracking("deserialize proof");
        let proof = RV64IMACProof::deserialize_with_mode(&mut cursor, Compress::Yes, Validate::No).unwrap();
        end_cycle_tracking("deserialize proof");

        start_cycle_tracking("deserialize device");
        let device = JoltDevice::deserialize_with_mode(&mut cursor, Compress::Yes, Validate::No).unwrap();
        end_cycle_tracking("deserialize device");

        start_cycle_tracking("verification");
        let verifier = RV64IMACVerifier::new(&verifier_preprocessing, proof, device, None, None);
        let is_valid = verifier.is_ok_and(|verifier| verifier.verify().is_ok());
        end_cycle_tracking("verification");
        all_valid = all_valid && is_valid;
    }

    all_valid as u32
}
}
