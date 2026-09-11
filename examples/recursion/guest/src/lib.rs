// #![cfg_attr(feature = "guest", no_std)]

use jolt_sdk::{self as jolt};

extern crate alloc;
use alloc::vec::Vec;
use embedded_bytes::EMBEDDED_BYTES;
#[cfg(feature = "akita")]
use jolt::jolt_verifier::{JoltProof, JoltVerifierPreprocessing as GenericVerifierPreprocessing};
#[cfg(feature = "akita")]
use jolt_crypto::NoVectorCommitment;
#[cfg(feature = "akita")]
use jolt_transcript::LegacyBlake2bTranscript;

use jolt::JoltDevice;
#[cfg(not(feature = "akita"))]
use jolt::{JoltVerifierPreprocessing, RV64IMACProof};
#[cfg(feature = "akita")]
use jolt_akita::{AkitaField, AkitaScheme};
#[cfg(feature = "akita")]
type AkitaVc = NoVectorCommitment<AkitaField>;
#[cfg(feature = "akita")]
type AkitaTranscript = LegacyBlake2bTranscript<AkitaField>;
#[cfg(feature = "akita")]
type RV64IMACProof = JoltProof<AkitaScheme, AkitaVc>;
#[cfg(feature = "akita")]
type JoltVerifierPreprocessing = GenericVerifierPreprocessing<AkitaScheme, AkitaVc>;
use serde::de::DeserializeOwned;

use jolt::{end_cycle_tracking, start_cycle_tracking};

mod embedded_bytes {
    include!("./embedded_bytes.rs");
}

include!("./provable_macro.rs");

/// Reader over the host's record stream: `[u64 length][body][zero padding to
/// 8 bytes]` per record behind one leading pad record that lands every body
/// 8-byte aligned in guest memory (see the host's `frame_guest_input`). Raw
/// bodies are used where they lie.
struct Records<'a> {
    buffer: &'a [u8],
    offset: usize,
    /// Start of the padded section (after the pad record): record padding is
    /// relative to it, and the host lands it on an 8-byte address.
    base: usize,
}

impl<'a> Records<'a> {
    fn new(buffer: &'a [u8]) -> Self {
        let mut records = Self {
            buffer,
            offset: 0,
            base: 0,
        };
        let pad = records.len_prefix();
        records.offset += pad;
        records.base = records.offset;
        assert_eq!(
            (buffer.as_ptr() as usize + records.base) % 8,
            0,
            "record bodies are not 8-byte aligned"
        );
        records
    }

    fn len_prefix(&mut self) -> usize {
        assert!(
            self.buffer.len().saturating_sub(self.offset) >= 8,
            "missing record length prefix"
        );
        let mut len_bytes = [0u8; 8];
        len_bytes.copy_from_slice(&self.buffer[self.offset..self.offset + 8]);
        self.offset += 8;
        usize::try_from(u64::from_le_bytes(len_bytes)).unwrap()
    }

    fn raw(&mut self) -> &'a [u8] {
        let len = self.len_prefix();
        assert!(
            self.buffer.len().saturating_sub(self.offset) >= len,
            "truncated record"
        );
        let end = self.offset + len;
        let bytes = &self.buffer[self.offset..end];
        self.offset = (self.base + (end - self.base).next_multiple_of(8)).min(self.buffer.len());
        bytes
    }

    fn record<T: DeserializeOwned>(&mut self) -> T {
        let bytes = self.raw();
        let (value, consumed) =
            bincode::serde::decode_from_slice(bytes, bincode::config::standard()).unwrap();
        assert_eq!(consumed, bytes.len(), "record decoder left trailing bytes");
        value
    }
}

/// The guest's input region and its own image are mapped for the whole run
/// and never written or freed, so a slice of either outlives every use.
#[cfg(feature = "akita")]
fn assume_static(bytes: &[u8]) -> &'static [u8] {
    // SAFETY: see above; the pointer and length are unchanged.
    unsafe { core::slice::from_raw_parts(bytes.as_ptr(), bytes.len()) }
}

provable_with_config! {
fn verify(bytes: &[u8]) -> u32 {
    let mut input = Records::new(bytes);
    // The verifier setup comes from this image when the host baked it in
    // (then the input carries only the proofs), else from the input.
    let mut embedded = (!EMBEDDED_BYTES.is_empty())
        .then(|| Records::new(EMBEDDED_BYTES));
    let setup = embedded.as_mut().unwrap_or(&mut input);

    start_cycle_tracking("deserialize preprocessing");
    #[cfg_attr(not(feature = "akita"), expect(unused_mut))]
    let mut verifier_preprocessing: JoltVerifierPreprocessing = setup.record();
    // Setup payloads the host detached from the record (Akita's expanded
    // verifier keys), attached back as views of where they lie.
    let payload_count: u32 = setup.record();
    let payloads: Vec<&[u8]> = (0..payload_count).map(|_| setup.raw()).collect();
    #[cfg(feature = "akita")]
    {
        let payloads: Vec<&'static [u8]> =
            payloads.iter().map(|payload| assume_static(payload)).collect();
        verifier_preprocessing
            .pcs_setup
            .attach_prepared_payloads(&payloads)
            .expect("attach prepared Akita payloads");
    }
    #[cfg(not(feature = "akita"))]
    assert!(payloads.is_empty(), "unexpected setup payloads");
    end_cycle_tracking("deserialize preprocessing");

    start_cycle_tracking("deserialize count of proofs");
    let n: u32 = input.record();
    end_cycle_tracking("deserialize count of proofs");

    let mut all_valid = true;
    for _ in 0..n {
        start_cycle_tracking("deserialize proof");
        let proof: RV64IMACProof = input.record();
        end_cycle_tracking("deserialize proof");

        start_cycle_tracking("deserialize device");
        let device: JoltDevice = input.record();
        end_cycle_tracking("deserialize device");

        start_cycle_tracking("verification");
        #[cfg(not(feature = "akita"))]
        let is_valid = jolt::jolt_verifier::verify::<
            jolt::VerifierField,
            jolt::VerifierPCS,
            jolt::VerifierVC,
            jolt::VerifierTranscript,
        >(&verifier_preprocessing, &device, &proof, None)
        .is_ok();
        #[cfg(feature = "akita")]
        let is_valid = jolt::jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
            &verifier_preprocessing,
            &device,
            &proof,
            None,
        )
        .inspect_err(|error| eprintln!("verification failed: {error:?}"))
        .is_ok();
        end_cycle_tracking("verification");
        all_valid = all_valid && is_valid;
    }

    all_valid as u32
}
}
