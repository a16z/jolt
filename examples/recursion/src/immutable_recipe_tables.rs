//! In-process immutable recipe tables for the verifier-guest build.
//!
//! Before the guest is built, the host verifies every proof of the stream it
//! is about to embed, natively and inside one recording session. Only if every
//! proof verifies and the capture closes does it render the tables, write them
//! under the guest target directory by content digest, and name them in
//! `JOLT_IMMUTABLE_RECIPES_RS` for that build. Otherwise it names a file of
//! three empty tables: every lookup misses and the guest keeps the canonical
//! expression path. Tables are never taken from a proof that did not verify,
//! and the caller never has to set the variable.

use crate::{
    read_raw, read_record, AkitaField, GuestProof, GuestVerifierPreprocessing, JoltDevice,
};
use jolt_akita::{AkitaScheme, TrustedBytes};
use jolt_claims::immutable_recipes::record::Session;
use jolt_field::Ring;
use jolt_sdk::jolt_prover::akita::preprocessing::{AkitaTranscript, AkitaVc};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::info;

const VARIABLE: &str = "JOLT_IMMUTABLE_RECIPES_RS";
/// Three empty tables, in the generated file's form: every lookup misses.
const EMPTY: &str = "pub static JOLT:JoltTable=JoltTable::empty();\n\
pub static COMPOSED:ComposedTable=ComposedTable::empty();\n\
pub static FIELD:FieldTable=FieldTable::empty();\n";
/// `RECURSION_IMMUTABLE_RECIPES=off` skips generation: the guest gets empty tables.
const SWITCH: &str = "RECURSION_IMMUTABLE_RECIPES";

/// Names the tables for the next guest build: the generated ones, or empty
/// tables when generation is switched off or does not close. A caller-set
/// variable is kept. Returns whether the variable was set here, for [`release`].
pub(crate) fn prepare(stream: &[u8], target_dir: &str) -> bool {
    if std::env::var_os(VARIABLE).is_some() {
        info!("  immutable recipe tables: caller-supplied {VARIABLE}");
        return false;
    }
    let source = if std::env::var(SWITCH).as_deref() == Ok("off") {
        info!("  immutable recipe tables: off; empty tables, canonical path");
        EMPTY.to_owned()
    } else {
        match generate(stream) {
            Ok(source) => source,
            Err(reason) => {
                info!("  immutable recipe tables: none ({reason}); empty tables, canonical path");
                EMPTY.to_owned()
            }
        }
    };
    match write(&source, target_dir) {
        Ok(path) => {
            info!("  immutable recipe tables file: {}", path.display());
            std::env::set_var(VARIABLE, &path);
            true
        }
        Err(reason) => {
            info!("  immutable recipe tables file not written: {reason}");
            false
        }
    }
}

/// Removes the variable [`prepare`] set, so it does not outlive this build.
pub(crate) fn release(set_here: bool) {
    if set_here {
        std::env::remove_var(VARIABLE);
    }
}

fn generate(stream: &[u8]) -> Result<String, String> {
    if AkitaField::immutable_recipe_native_field().is_none() {
        return Err("the proof field has no native recipe form".to_owned());
    }
    let mut offset = 0;
    let verifier_preprocessing: GuestVerifierPreprocessing = read_record(stream, &mut offset)?;
    let payload_count: u32 = read_record(stream, &mut offset)?;
    let mut payloads = Vec::new();
    for _ in 0..payload_count {
        payloads.push(read_raw(stream, &mut offset)?.to_vec());
    }
    let proof_count: u32 = read_record(stream, &mut offset)?;
    let mut records = Vec::new();
    for _ in 0..proof_count {
        let proof: GuestProof = read_record(stream, &mut offset)?;
        let public_io: JoltDevice = read_record(stream, &mut offset)?;
        records.push((proof, public_io));
    }
    if offset != stream.len() || records.is_empty() {
        return Err("the proof stream is not one setup and at least one proof".to_owned());
    }
    let mut attached = verifier_preprocessing;
    let bodies: Vec<TrustedBytes> = payloads
        .into_iter()
        // SAFETY: these are the setup bodies of the stream the guest embeds as
        // its trusted constant (written by `generate` from the host's own
        // prepared setup); the host extends them the same trust as the guest.
        .map(|payload| unsafe { TrustedBytes::new(Box::leak(payload.into_boxed_slice())) })
        .collect();
    attached
        .pcs_setup
        .attach_prepared_payloads(&bodies)
        .map_err(|error| format!("prepared payloads did not attach: {error:?}"))?;
    let session = Session::begin().map_err(|error| error.to_string())?;
    for (proof, public_io) in &records {
        jolt_sdk::jolt_verifier::verify::<AkitaField, AkitaScheme, AkitaVc, AkitaTranscript>(
            &attached, public_io, proof, None,
        )
        .map_err(|error| format!("native verification rejected a proof: {error:?}"))?;
    }
    let mut capture = session.finish().map_err(|error| error.to_string())?;
    capture.source().map_err(|error| error.to_string())
}

/// Writes `source` under the guest target directory by content digest.
fn write(source: &str, target_dir: &str) -> Result<PathBuf, String> {
    let digest = hex::encode(Sha256::digest(source.as_bytes()));
    let directory = Path::new(target_dir).join("immutable-recipes");
    fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    let path = directory.join(format!("{digest}.rs"));
    if !path.is_file() {
        let temporary = directory.join(format!("{digest}.rs.tmp-{}", std::process::id()));
        fs::write(&temporary, source.as_bytes()).map_err(|error| error.to_string())?;
        fs::rename(&temporary, &path).map_err(|error| error.to_string())?;
    }
    Ok(path)
}
