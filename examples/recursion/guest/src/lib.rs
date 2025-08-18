// #![cfg_attr(feature = "guest", no_std)]

use jolt_sdk::{self as jolt};

extern crate alloc;

use ark_serialize::CanonicalDeserialize;
use jolt::Jolt;
use jolt::{JoltDevice, JoltRV32IM, JoltVerifierPreprocessing, RV32IMJoltProof, F, PCS};

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
    start_cycle_tracking("deserialize");
    let _ = u32::deserialize_compressed(&mut cursor).unwrap();
    end_cycle_tracking("deserialize");

    start_cycle_tracking("preprocessing");
    let verifier_preprocessing: JoltVerifierPreprocessing<F, PCS> =
        JoltVerifierPreprocessing::<F, PCS>::deserialize_compressed(&mut cursor).unwrap();
    end_cycle_tracking("preprocessing");

    start_cycle_tracking("n");
    let n: u32 = u32::deserialize_compressed(&mut cursor).unwrap();
    end_cycle_tracking("n");

    let mut all_valid = true;
    for _ in 0..n {
        start_cycle_tracking("proof");
        let proof = RV32IMJoltProof::deserialize_compressed(&mut cursor).unwrap();
        end_cycle_tracking("proof");

        start_cycle_tracking("device");
        let device = JoltDevice::deserialize_compressed(&mut cursor).unwrap();
        end_cycle_tracking("device");

        start_cycle_tracking("verification");
        let is_valid = JoltRV32IM::verify(&verifier_preprocessing, proof, device, None).is_ok();
        end_cycle_tracking("verification");
        all_valid = all_valid && is_valid;
    }

    end_cycle_tracking("verify_proofs");

    all_valid as u32
}
}
