use jolt_sdk::{self as jolt};

extern crate alloc;
use alloc::sync::Arc;
use alloc::vec::Vec;
use embedded_bytecode::EMBEDDED_BYTECODE;
use embedded_bytes::{EMBEDDED_BYTES, SETUP_DIGEST};
use jolt::jolt_verifier::preprocessing::ProgramPreprocessing;
use jolt::jolt_verifier::VerifierPreprocessingWire;
#[cfg(feature = "akita")]
use jolt::jolt_verifier::{JoltProof, JoltVerifierPreprocessing as GenericVerifierPreprocessing};
#[cfg(feature = "akita")]
use jolt_akita::TrustedBytes;
#[cfg(feature = "akita")]
use jolt_crypto::NoVectorCommitment;

use jolt::JoltDevice;
#[cfg(not(feature = "akita"))]
use jolt::{JoltVerifierPreprocessing, RV64IMACProof};
#[cfg(feature = "akita")]
use jolt_akita::{AkitaField, AkitaScheme};
#[cfg(feature = "akita")]
type AkitaVc = NoVectorCommitment<AkitaField>;
#[cfg(feature = "akita")]
type RV64IMACProof = JoltProof<AkitaScheme, AkitaVc>;
#[cfg(feature = "akita")]
type JoltVerifierPreprocessing = GenericVerifierPreprocessing<AkitaScheme, AkitaVc>;
#[cfg(feature = "akita")]
type PreprocessingWire = VerifierPreprocessingWire<AkitaScheme, AkitaVc>;
#[cfg(not(feature = "akita"))]
type PreprocessingWire = VerifierPreprocessingWire<jolt::VerifierPCS, jolt::VerifierVC>;

use jolt::{end_cycle_tracking, start_cycle_tracking};

mod embedded_bytecode {
    include!("./embedded_bytecode.rs");
}

mod embedded_bytes {
    include!("./embedded_bytes.rs");
}

include!("./provable_macro.rs");

mod records;
use records::{RecordError, Records};

impl Records<'_> {
    /// The baked setup's preprocessing with its compiled bytecode rows.
    ///
    /// The record carries the program without its rows, followed by the
    /// preprocessing digest the host computed over the complete program. The
    /// image fixes both, so the guest assembles the preprocessing from them
    /// instead of re-encoding and hashing the program.
    #[inline(never)]
    fn preprocessing(&mut self) -> Result<JoltVerifierPreprocessing, RecordError> {
        let PreprocessingWire {
            mut program,
            pcs_setup,
            vc_setup,
        } = self.record()?;
        let preprocessing_digest: [u8; 32] = self.record()?;
        if !EMBEDDED_BYTECODE.is_empty() {
            let ProgramPreprocessing::Full(program) = &mut program else {
                panic!("compiled bytecode requires full program preprocessing");
            };
            let program = Arc::make_mut(program);
            assert!(
                program.bytecode.bytecode.is_empty(),
                "compiled bytecode placeholder is not empty"
            );
            assert_eq!(
                program.bytecode.code_size,
                EMBEDDED_BYTECODE.len(),
                "compiled bytecode shape mismatch"
            );
            program.bytecode.bytecode = EMBEDDED_BYTECODE.to_vec();
        }
        Ok(JoltVerifierPreprocessing {
            program,
            preprocessing_digest,
            pcs_setup,
            vc_setup,
        })
    }
}

/// Extend a slice of the guest's input region or its own image to `'static`.
///
/// # Safety
///
/// `bytes` must lie in the input region or the image, which are mapped for
/// the whole run and never written or freed.
#[cfg(feature = "akita")]
unsafe fn assume_static(bytes: &[u8]) -> &'static [u8] {
    // SAFETY: the caller's contract; the pointer and length are unchanged.
    unsafe { core::slice::from_raw_parts(bytes.as_ptr(), bytes.len()) }
}

provable_with_config! {
fn verify(bytes: &[u8]) -> u32 {
    verify_records(bytes).unwrap_or(0)
}
}

fn setup_result<T>(result: Result<T, RecordError>) -> Result<T, RecordError> {
    if EMBEDDED_BYTES.is_empty() {
        result
    } else {
        Ok(result.expect("corrupt embedded setup"))
    }
}

fn verify_records(bytes: &[u8]) -> Result<u32, RecordError> {
    let mut input = Records::new(bytes)?;
    if EMBEDDED_BYTES.is_empty() {
        input.authenticate_setup(SETUP_DIGEST.as_ref())?;
    }
    // The verifier setup comes from this image when the host baked it in
    // (then the input carries only the proofs), else from the input.
    let mut embedded = (!EMBEDDED_BYTES.is_empty())
        .then(|| Records::new(EMBEDDED_BYTES).expect("corrupt embedded framing"));
    let setup = embedded.as_mut().unwrap_or(&mut input);

    start_cycle_tracking("deserialize preprocessing");
    #[cfg_attr(not(feature = "akita"), expect(unused_mut))]
    let mut verifier_preprocessing = if EMBEDDED_BYTES.is_empty() {
        setup.record()?
    } else {
        setup.preprocessing().expect("corrupt embedded preprocessing")
    };

    // Setup payloads the host detached from the record (Akita's expanded
    // verifier keys), attached back as views of where they lie.
    let payload_count: u32 = setup_result(setup.record())?;
    setup_result(setup.check_count(payload_count, 1))?;
    let payloads: Vec<&[u8]> = setup_result(
        (0..payload_count).map(|_| setup.raw()).collect::<Result<_, _>>(),
    )?;
    #[cfg(feature = "akita")]
    {
        let payloads: Vec<TrustedBytes> = payloads
            .iter()
            // SAFETY: every payload slice lies in the image or input region,
            // and is bound by the image: embedded bytes directly, or the
            // complete input setup authenticated against SETUP_DIGEST above.
            .map(|payload| unsafe { TrustedBytes::new(assume_static(payload)) })
            .collect();
        setup_result(
            verifier_preprocessing
                .pcs_setup
                .attach_prepared_payloads(&payloads)
                .map_err(|_| RecordError),
        )?;
    }
    #[cfg(not(feature = "akita"))]
    if !payloads.is_empty() {
        return setup_result(Err(RecordError));
    }
    end_cycle_tracking("deserialize preprocessing");

    start_cycle_tracking("deserialize count of proofs");
    let n: u32 = input.record()?;
    input.check_count(n, 2)?;
    end_cycle_tracking("deserialize count of proofs");

    let mut all_valid = true;
    for _ in 0..n {
        start_cycle_tracking("deserialize proof");
        let proof: RV64IMACProof = input.record()?;
        end_cycle_tracking("deserialize proof");

        start_cycle_tracking("deserialize device");
        let device: JoltDevice = input.record()?;
        end_cycle_tracking("deserialize device");

        start_cycle_tracking("verification");
        #[cfg(not(feature = "akita"))]
        let is_valid = jolt::jolt_verifier::verify::<
            jolt::VerifierField,
            jolt::VerifierPCS,
            jolt::VerifierVC,
            jolt::VerifierTranscript<'_>,
        >(&verifier_preprocessing, &device, &proof, None)
        .is_ok();
        #[cfg(feature = "akita")]
        let is_valid = jolt::jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, jolt::VerifierTranscript<'_>>(
            &verifier_preprocessing,
            &device,
            &proof,
            None,
        )
        .is_ok();
        end_cycle_tracking("verification");
        all_valid = all_valid && is_valid;
    }

    Ok(all_valid as u32)
}
