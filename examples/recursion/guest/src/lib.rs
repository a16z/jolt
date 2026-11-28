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
use jolt::{TrustedAdvice, UntrustedAdvice};
#[cfg(feature = "akita")]
use jolt_akita::TrustedBytes;
#[cfg(feature = "akita")]
use jolt_crypto::NoVectorCommitment;

use jolt::JoltDevice;
#[cfg(not(feature = "akita"))]
use jolt::{JoltVerifierPreprocessing, RV64IMACProof};
#[cfg(feature = "akita")]
use jolt_akita::{AkitaField, AkitaScheme, AkitaVerifierTranscript};
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

mod bounded;
#[cfg(feature = "akita")]
mod compiled_rows;
mod records;
use records::{RecordError, Records};

/// Where a setup whose record omits the bytecode rows keeps them.
enum CompiledRows {
    /// The static table compiled into this image.
    Image,
    /// A fixed-width record after the digest, in the setup stream.
    #[cfg(feature = "akita")]
    Stream,
}

impl Records<'_> {
    /// A setup's preprocessing with its compiled bytecode rows.
    ///
    /// The record carries the program without its rows, followed by the
    /// preprocessing digest the host computed over the complete program. The
    /// image or the committed trusted advice fixes both, so the guest
    /// assembles the preprocessing from them instead of re-encoding and
    /// hashing the program.
    #[inline(never)]
    fn compiled_preprocessing(
        &mut self,
        rows: CompiledRows,
    ) -> Result<JoltVerifierPreprocessing, RecordError> {
        let PreprocessingWire {
            mut program,
            pcs_setup,
            vc_setup,
        } = self.record()?;
        let preprocessing_digest: [u8; 32] = self.record()?;
        let rows = match rows {
            CompiledRows::Image => EMBEDDED_BYTECODE.to_vec(),
            #[cfg(feature = "akita")]
            CompiledRows::Stream => {
                let words = bytemuck::try_cast_slice(self.raw()?).map_err(|_| RecordError)?;
                compiled_rows::decode(words).ok_or(RecordError)?
            }
        };
        if !rows.is_empty() {
            let ProgramPreprocessing::Full(program) = &mut program else {
                return Err(RecordError);
            };
            let program = Arc::make_mut(program);
            if !program.bytecode.bytecode.is_empty() || program.bytecode.code_size != rows.len() {
                return Err(RecordError);
            }
            program.bytecode.bytecode = rows;
        }
        Ok(JoltVerifierPreprocessing {
            program,
            preprocessing_digest,
            pcs_setup,
            vc_setup,
        })
    }

    /// Attach the bodies `detach_prepared_payloads` moved out of the setup
    /// record, in its order, as views of where they lie in this stream.
    ///
    /// # Safety
    ///
    /// The stream must lie in the image, the input region, or the
    /// trusted-advice region, which stay mapped and unwritten for the whole
    /// run, and its bytes must be the intended setup, bound to the statement.
    unsafe fn attach_payloads(
        &mut self,
        #[cfg_attr(not(feature = "akita"), expect(unused_variables))]
        preprocessing: &mut JoltVerifierPreprocessing,
    ) -> Result<(), RecordError> {
        let payload_count: u32 = self.record()?;
        self.check_count(payload_count, 1)?;
        let payloads: Vec<&[u8]> = (0..payload_count)
            .map(|_| self.raw())
            .collect::<Result<_, _>>()?;
        #[cfg(feature = "akita")]
        {
            let payloads: Vec<TrustedBytes> = payloads
                .iter()
                // SAFETY: the caller's contract.
                .map(|payload| unsafe { TrustedBytes::new(assume_static(payload)) })
                .collect();
            preprocessing
                .pcs_setup
                .attach_prepared_payloads(&payloads)
                .map_err(|_| RecordError)?;
        }
        #[cfg(not(feature = "akita"))]
        if !payloads.is_empty() {
            return Err(RecordError);
        }
        Ok(())
    }

    fn proof(&mut self) -> Result<RV64IMACProof, RecordError> {
        start_cycle_tracking("deserialize proof");
        let proof = self.record();
        end_cycle_tracking("deserialize proof");
        proof
    }

    fn device(&mut self) -> Result<JoltDevice, RecordError> {
        start_cycle_tracking("deserialize device");
        let device = self.record();
        end_cycle_tracking("deserialize device");
        device
    }
}

/// Extend a slice of the guest's input region, trusted advice, or own image
/// to `'static`.
///
/// # Safety
///
/// `bytes` must lie in the input region, the trusted-advice region, or the
/// image, which are mapped for the whole run and never written or freed.
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

#[cfg(feature = "akita")]
provable_with_config! {
/// Advice mode: the setup is trusted advice, the proofs are untrusted
/// advice, and the public input carries only the statements: the proof count
/// and one I/O device per proof.
fn verify_advice(
    statements: &[u8],
    setup: TrustedAdvice<&[u8]>,
    proofs: UntrustedAdvice<&[u8]>,
) -> u32 {
    verify_advice_records(statements, *setup, *proofs).unwrap_or(0)
}
}

fn setup_result<T>(result: Result<T, RecordError>) -> Result<T, RecordError> {
    if EMBEDDED_BYTES.is_empty() {
        result
    } else {
        Ok(result.expect("corrupt embedded setup"))
    }
}

fn verify_inner(
    preprocessing: &JoltVerifierPreprocessing,
    proof: &RV64IMACProof,
    device: &JoltDevice,
) -> bool {
    start_cycle_tracking("verification");
    #[cfg(not(feature = "akita"))]
    let is_valid = jolt::jolt_verifier::verify::<
        jolt::VerifierField,
        jolt::VerifierPCS,
        jolt::VerifierVC,
        jolt::VerifierTranscript<'_>,
    >(preprocessing, device, proof, None)
    .is_ok();
    #[cfg(feature = "akita")]
    let is_valid = jolt::jolt_verifier::verify::<
        AkitaField,
        AkitaScheme,
        AkitaVc,
        AkitaVerifierTranscript<'_>,
    >(preprocessing, device, proof, None)
    .is_ok();
    end_cycle_tracking("verification");
    is_valid
}

fn verify_records(bytes: &[u8]) -> Result<u32, RecordError> {
    let mut input = Records::new(bytes, RECORD_LIMITS)?;
    if EMBEDDED_BYTES.is_empty() {
        input.authenticate_setup(SETUP_DIGEST.as_ref())?;
    }
    // The verifier setup comes from this image when the host baked it in
    // (then the input carries only the proofs), else from the input.
    let mut embedded = (!EMBEDDED_BYTES.is_empty())
        .then(|| Records::new(EMBEDDED_BYTES, RECORD_LIMITS).expect("corrupt embedded framing"));
    let setup = embedded.as_mut().unwrap_or(&mut input);

    start_cycle_tracking("deserialize preprocessing");
    let mut verifier_preprocessing = if EMBEDDED_BYTES.is_empty() {
        setup.record()?
    } else {
        setup
            .compiled_preprocessing(CompiledRows::Image)
            .expect("corrupt embedded preprocessing")
    };
    // SAFETY: the setup stream is this image or the input region. The image
    // binds embedded bytes directly; the complete input setup was
    // authenticated against SETUP_DIGEST above.
    setup_result(unsafe { setup.attach_payloads(&mut verifier_preprocessing) })?;
    end_cycle_tracking("deserialize preprocessing");

    start_cycle_tracking("deserialize count of proofs");
    let n = input.proof_count(2)?;
    end_cycle_tracking("deserialize count of proofs");

    let mut all_valid = true;
    for _ in 0..n {
        let proof = input.proof()?;
        let device = input.device()?;
        all_valid &= verify_inner(&verifier_preprocessing, &proof, &device);
    }

    input.finish()?;
    Ok(all_valid as u32)
}

#[cfg(feature = "akita")]
fn verify_advice_records(
    statements: &[u8],
    setup: &[u8],
    proofs: &[u8],
) -> Result<u32, RecordError> {
    let mut setup = Records::new(setup, RECORD_LIMITS)?;
    start_cycle_tracking("deserialize preprocessing");
    let mut verifier_preprocessing = setup.compiled_preprocessing(CompiledRows::Stream)?;
    // SAFETY: the setup stream is the trusted-advice region. An outer
    // verifier checks that region against the `trusted_advice_commitment` it
    // is given, which the consumer pins to the published setup, so these are
    // the published setup bytes.
    unsafe { setup.attach_payloads(&mut verifier_preprocessing) }?;
    setup.finish()?;
    end_cycle_tracking("deserialize preprocessing");

    let mut statements = Records::new(statements, RECORD_LIMITS)?;
    let mut proofs = Records::new(proofs, RECORD_LIMITS)?;
    start_cycle_tracking("deserialize count of proofs");
    let n = statements.proof_count(1)?;
    proofs.check_count(n, 1)?;
    end_cycle_tracking("deserialize count of proofs");

    let mut all_valid = true;
    for _ in 0..n {
        let proof = proofs.proof()?;
        let device = statements.device()?;
        all_valid &= verify_inner(&verifier_preprocessing, &proof, &device);
    }

    statements.finish()?;
    proofs.finish()?;
    Ok(all_valid as u32)
}

#[cfg(feature = "kernel-tests")]
mod kernel_tests;
