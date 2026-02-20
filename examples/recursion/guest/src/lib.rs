// #![cfg_attr(feature = "guest", no_std)]

use jolt_sdk::{self as jolt};

extern crate alloc;

use ark_serialize::{CanonicalDeserialize, Compress, Validate};
use jolt::transport;
use jolt::{JoltDevice, JoltVerifierPreprocessing, RV64IMACProof, RV64IMACVerifier, F, PCS};

use jolt::transport::{
    BUNDLE_SIGNATURE, BUNDLE_TAG_PREPROCESSING, BUNDLE_TAG_RECORD, RECORD_TAG_DEVICE,
    RECORD_TAG_PROOF,
};
use jolt::{end_cycle_tracking, start_cycle_tracking};
use std::io::Read;

mod embedded_bytes {
    include!("./embedded_bytes.rs");
}

include!("./provable_macro.rs");

provable_with_config! {
fn verify(bytes: &[u8]) -> u32 {
    // Force single-threaded execution in std-mode guests.
    // We avoid spawning worker threads (which relies on OS/thread-local plumbing) by using
    // a local Rayon pool installed on the current thread.
    let rayon_pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .use_current_thread()
        .build()
        .ok();

    // Treat embedded bytes as an optimization for the "embed mode" host path, where the host
    // passes empty input bytes. If non-empty input bytes are provided, prefer them even if the
    // embedded blob exists (so iterative testing doesn't accidentally use stale embedded data).
    let use_embedded = bytes.is_empty() && !embedded_bytes::EMBEDDED_BYTES.is_empty();
    let data_bytes = if use_embedded {
        embedded_bytes::EMBEDDED_BYTES
    } else {
        // Non-embedded mode: `#[jolt::provable]` already deserialized `bytes` from the input tape
        // using `postcard::take_from_bytes::<&[u8]>()`, so `bytes` is the raw bundle.
        bytes
    };

    let mut cursor = std::io::Cursor::new(data_bytes);

    // Framed bundle decode (strict).
    transport::signature_check(&mut cursor, BUNDLE_SIGNATURE).unwrap();

    let mut verifier_preprocessing: Option<JoltVerifierPreprocessing<F, PCS>> = None;

    let mut all_valid = true;
    while let Some((tag, len)) = transport::read_frame_header(&mut cursor, 1_u64 << 32).unwrap() {
        let mut limited = (&mut cursor).take(len);
        match tag {
            BUNDLE_TAG_PREPROCESSING => {
                if verifier_preprocessing.is_some() {
                    panic!("duplicate preprocessing");
                }
                // IMPORTANT: this is a verifier path; do not skip validation on attacker-controlled bytes.
                // (For performance, a trusted host pipeline can pre-validate and provide a guest-optimized encoding.)
                start_cycle_tracking("deserialize preprocessing");
                let prep = JoltVerifierPreprocessing::<F, PCS>::deserialize_with_mode(
                    &mut limited,
                    Compress::Yes,
                    Validate::Yes,
                )
                .unwrap();
                end_cycle_tracking("deserialize preprocessing");
                if limited.limit() != 0 {
                    panic!("trailing bytes in preprocessing frame");
                }
                verifier_preprocessing = Some(prep);
            }
            BUNDLE_TAG_RECORD => {
                let prep = verifier_preprocessing.as_ref().expect("missing preprocessing");

                let mut proof: Option<RV64IMACProof> = None;
                let mut device: Option<JoltDevice> = None;

                while let Some((rtag, rlen)) =
                    transport::read_frame_header(&mut limited, 1_u64 << 32).unwrap()
                {
                    let mut rlimited = (&mut limited).take(rlen);
                    match rtag {
                        RECORD_TAG_PROOF => {
                            if proof.is_some() {
                                panic!("duplicate proof in record");
                            }
                            start_cycle_tracking("deserialize proof");
                            let p = RV64IMACProof::deserialize_with_mode(
                                &mut rlimited,
                                Compress::Yes,
                                Validate::Yes,
                            )
                            .unwrap();
                            end_cycle_tracking("deserialize proof");
                            if rlimited.limit() != 0 {
                                panic!("trailing bytes in proof subframe");
                            }
                            proof = Some(p);
                        }
                        RECORD_TAG_DEVICE => {
                            if device.is_some() {
                                panic!("duplicate device in record");
                            }
                            start_cycle_tracking("deserialize device");
                            let d = JoltDevice::deserialize_with_mode(
                                &mut rlimited,
                                Compress::Yes,
                                Validate::Yes,
                            )
                            .unwrap();
                            end_cycle_tracking("deserialize device");
                            if rlimited.limit() != 0 {
                                panic!("trailing bytes in device subframe");
                            }
                            device = Some(d);
                        }
                        _ => panic!("unknown record tag"),
                    }
                }

                if limited.limit() != 0 {
                    panic!("trailing bytes in record frame");
                }

                let proof = proof.expect("missing proof in record");
                let device = device.expect("missing device in record");

                jolt::eprintln!("starting record verification");
                start_cycle_tracking("verification");
                let verifier = RV64IMACVerifier::new(prep, proof, device, None, None);
                jolt::eprintln!("verifier_new_ok={}", verifier.is_ok());
                let is_valid = match verifier {
                    Ok(verifier) => {
                        jolt::eprintln!("calling verifier.verify()");
                        let run = || match verifier.verify() {
                            Ok(()) => true,
                            Err(e) => {
                                jolt::eprintln!("verifier.verify error: {e}");
                                false
                            }
                        };
                        match &rayon_pool {
                            Some(pool) => pool.install(run),
                            None => run(),
                        }
                    }
                    Err(_) => false,
                };
                end_cycle_tracking("verification");
                jolt::eprintln!("finished record verification: {is_valid}");
                all_valid = all_valid && is_valid;
            }
            _ => panic!("unknown bundle tag"),
        }
    }

    all_valid as u32
}
}
