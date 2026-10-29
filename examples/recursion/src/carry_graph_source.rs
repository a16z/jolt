//! Compiled carry-state graphs for the verifier-guest build (feature
//! `carry-graph`, default on with `akita`).
//!
//! Before the guest is built, the host reads the padded trace length of the
//! proofs in the stream it is about to verify and emits Akita's fixed
//! carry-state graph source for that size. It writes the source under the
//! guest target directory and names it in `JOLT_CARRY_GRAPH_RS` for that
//! build. If the proofs do not share one supported size, or
//! `RECURSION_CARRY_GRAPH=off`, it names a file with no graph: every lookup
//! misses and the guest keeps the recurrence path. A caller-set variable is
//! kept. The graphs are fixed per size and never taken from a proof's values.

use crate::{read_raw, read_record, GuestProof, JoltDevice};
use jolt_akita::carry_graph::emit_fixed_source;
use std::fs;
use std::path::{Path, PathBuf};
use tracing::info;

const VARIABLE: &str = "JOLT_CARRY_GRAPH_RS";
/// `RECURSION_CARRY_GRAPH=off` skips emission: the guest gets no graph.
const SWITCH: &str = "RECURSION_CARRY_GRAPH";
/// No graph, in the emitted file's form: every lookup misses.
const EMPTY: &str = "pub(super) static GRAPHS: &[Graph<'static>] = &[\n];\n";

/// Names the carry graph source for the next guest build. Returns whether the
/// variable was set here, for [`release`].
pub(crate) fn prepare(stream: &[u8], target_dir: &str) -> bool {
    if std::env::var_os(VARIABLE).is_some() {
        info!("  carry graph: caller-supplied {VARIABLE}");
        return false;
    }
    let (name, source) = if std::env::var(SWITCH).as_deref() == Ok("off") {
        info!("  carry graph: off; no graph, recurrence path");
        ("none".to_owned(), EMPTY.to_owned())
    } else {
        match generate(stream) {
            Ok((exponent, source)) => (exponent.to_string(), source),
            Err(reason) => {
                info!("  carry graph: none ({reason}); recurrence path");
                ("none".to_owned(), EMPTY.to_owned())
            }
        }
    };
    match write(&name, &source, target_dir) {
        Ok(path) => {
            info!("  carry graph file: {}", path.display());
            std::env::set_var(VARIABLE, &path);
            true
        }
        Err(reason) => {
            info!("  carry graph file not written: {reason}");
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

/// The padded trace exponent shared by every proof of the stream, and the
/// fixed graph source Akita emits for it.
fn generate(stream: &[u8]) -> Result<(u32, String), String> {
    let mut offset = 0;
    // The verifier preprocessing record and the setup payloads are skipped.
    read_raw(stream, &mut offset)?;
    let payload_count: u32 = read_record(stream, &mut offset)?;
    for _ in 0..payload_count {
        read_raw(stream, &mut offset)?;
    }
    let proof_count: u32 = read_record(stream, &mut offset)?;
    let mut shared = None;
    for _ in 0..proof_count {
        let proof: GuestProof = read_record(stream, &mut offset)?;
        let _: JoltDevice = read_record(stream, &mut offset)?;
        if shared.is_some_and(|length| length != proof.trace_length) {
            return Err("the proofs do not share one trace length".to_owned());
        }
        shared = Some(proof.trace_length);
    }
    let length = shared.ok_or("the stream holds no proof")?;
    if offset != stream.len() || !length.is_power_of_two() {
        return Err("the stream is not one setup and padded proofs".to_owned());
    }
    let exponent = length.trailing_zeros();
    let source = emit_fixed_source(exponent).map_err(|error| format!("{error:?}"))?;
    Ok((exponent, source))
}

/// Writes `source` under the guest target directory, one file per size, and
/// rewrites it only when its bytes change.
fn write(name: &str, source: &str, target_dir: &str) -> Result<PathBuf, String> {
    let directory = Path::new(target_dir).join("carry-graph");
    fs::create_dir_all(&directory).map_err(|error| error.to_string())?;
    let path = directory.join(format!("graph-{name}.rs"));
    if fs::read(&path).ok().as_deref() != Some(source.as_bytes()) {
        let temporary = directory.join(format!("graph-{name}.rs.tmp-{}", std::process::id()));
        fs::write(&temporary, source.as_bytes()).map_err(|error| error.to_string())?;
        fs::rename(&temporary, &path).map_err(|error| error.to_string())?;
    }
    Ok(path)
}
