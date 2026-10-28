//! Prevalidated one-hot catalog view for the verifier-guest build (feature
//! `prevalidated-catalog`, default on with `akita`).
//!
//! Before the guest is built, the host decodes the setup section of the
//! stream it is about to embed (never a proof), attaches its payloads and
//! loads the prepared one-hot catalog view through Akita's canonical loader,
//! recording what loading derived. It writes the generated file under the
//! guest target directory by content digest and names it in
//! `JOLT_PREVALIDATED_CATALOG_RS` for that build. If anything fails, or
//! `RECURSION_PREVALIDATED_CATALOG=off`, it names a file with no view: every
//! lookup misses and the guest keeps the canonical path. A caller-set variable
//! is kept.

use crate::{read_raw, read_record, GuestVerifierPreprocessing};
use jolt_akita::{TrustedBytes, EMPTY_PREVALIDATED_CATALOG_SOURCE, PREVALIDATED_CATALOG_VARIABLE};
use sha2::{Digest, Sha256};
use std::fs;
use std::path::{Path, PathBuf};
use tracing::info;

const SWITCH: &str = "RECURSION_PREVALIDATED_CATALOG";

/// Names the prevalidated catalog file for the next guest build. `setup` is
/// the stream's setup section. Returns whether the variable was set here.
pub(crate) fn prepare(setup: &[u8], target_dir: &str) -> bool {
    if std::env::var_os(PREVALIDATED_CATALOG_VARIABLE).is_some() {
        info!("  prevalidated catalog: caller-supplied {PREVALIDATED_CATALOG_VARIABLE}");
        return false;
    }
    let source = if std::env::var(SWITCH).as_deref() == Ok("off") {
        info!("  prevalidated catalog: off; no view, canonical path");
        EMPTY_PREVALIDATED_CATALOG_SOURCE.to_owned()
    } else {
        match generate(setup) {
            Ok(source) => source,
            Err(reason) => {
                info!("  prevalidated catalog: none ({reason}); canonical path");
                EMPTY_PREVALIDATED_CATALOG_SOURCE.to_owned()
            }
        }
    };
    match write(&source, target_dir) {
        Ok(path) => {
            info!("  prevalidated catalog file: {}", path.display());
            std::env::set_var(PREVALIDATED_CATALOG_VARIABLE, &path);
            true
        }
        Err(reason) => {
            info!("  prevalidated catalog file not written: {reason}");
            false
        }
    }
}

/// Removes the variable [`prepare`] set, so it does not outlive this build.
pub(crate) fn release(set_here: bool) {
    if set_here {
        std::env::remove_var(PREVALIDATED_CATALOG_VARIABLE);
    }
}

fn generate(setup: &[u8]) -> Result<String, String> {
    let mut offset = 0;
    let verifier_preprocessing: GuestVerifierPreprocessing = read_record(setup, &mut offset)?;
    let payload_count: u32 = read_record(setup, &mut offset)?;
    let mut payloads = Vec::new();
    for _ in 0..payload_count {
        payloads.push(read_raw(setup, &mut offset)?.to_vec());
    }
    if offset != setup.len() {
        return Err("the setup section has trailing bytes".to_owned());
    }
    let mut attached = verifier_preprocessing;
    let bodies: Vec<TrustedBytes> = payloads
        .into_iter()
        // SAFETY: these are the setup bodies of the section the guest embeds
        // as its trusted constant; the host extends them the same trust.
        .map(|payload| unsafe { TrustedBytes::new(Box::leak(payload.into_boxed_slice())) })
        .collect();
    attached
        .pcs_setup
        .attach_prepared_payloads(&bodies)
        .map_err(|error| format!("prepared payloads did not attach: {error:?}"))?;
    attached.pcs_setup.prevalidated_catalog_source()
}

/// Writes `source` under the guest target directory by content digest.
fn write(source: &str, target_dir: &str) -> Result<PathBuf, String> {
    let digest = hex::encode(Sha256::digest(source.as_bytes()));
    let directory = Path::new(target_dir).join("prevalidated-catalog");
    fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    let path = directory.join(format!("{digest}.rs"));
    if !path.is_file() {
        let temporary = directory.join(format!("{digest}.rs.tmp-{}", std::process::id()));
        fs::write(&temporary, source.as_bytes()).map_err(|error| error.to_string())?;
        fs::rename(&temporary, &path).map_err(|error| error.to_string())?;
    }
    Ok(path)
}
