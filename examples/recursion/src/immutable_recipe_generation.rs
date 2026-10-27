use crate::{
    AkitaField, GuestProgram, GuestProof, GuestVerifierPreprocessing, JoltDevice,
    ProgramPreprocessingMode,
};
use jolt_claims::immutable_recipes::record::{Capture, Session};
use jolt_field::Ring;
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::error::Error;
use std::fs::{self, OpenOptions};
use std::io::{self, Write};
use std::path::PathBuf;

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[derive(Serialize)]
struct Context {
    status: &'static str,
    guest: &'static str,
    proof_count: u32,
    requested_scalar: u32,
    requested_input_sha256: String,
    max_trace_length: usize,
    actual_trace_length: usize,
    padded_trace_length: usize,
    input_sha256: String,
    accepted_proof_encoding_sha256: String,
    accepted_public_io_encoding_sha256: String,
    prepared_setup_encoding_sha256: String,
    detached_payload_sha256: Vec<String>,
    host_executable_sha256: String,
    interpretation: &'static str,
}

pub(crate) struct Request {
    output: PathBuf,
    scalar: u32,
    input: Vec<u8>,
    input_hash: String,
    trace_length: Option<usize>,
}
impl Request {
    pub(crate) fn observe_trace(&mut self, trace_length: usize) -> Result<()> {
        if trace_length == 0 || self.trace_length.replace(trace_length).is_some() {
            return Err(
                io::Error::other("exactly one nonempty canonical trace is required").into(),
            );
        }
        Ok(())
    }
}
pub(crate) struct Pending {
    request: Request,
    max_trace_length: usize,
    session: Result<Session>,
}
pub(crate) struct Ready {
    output: PathBuf,
    input: Vec<u8>,
    context: Context,
    capture: Capture,
}

fn hash(bytes: &[u8]) -> String {
    hex::encode(Sha256::digest(bytes))
}

/// Request validation and input construction occur before any trace/proof work.
/// Capture itself remains inactive until the original verification loop.
pub(crate) fn request(
    guest: GuestProgram,
    proofs: usize,
    mode: ProgramPreprocessingMode,
) -> Result<Option<Request>> {
    let Some(output) = std::env::var_os("JOLT_IMMUTABLE_RECIPE_OUTPUT") else {
        return Ok(None);
    };
    let output = PathBuf::from(output);
    if !output.is_absolute() || output.exists() {
        return Err(io::Error::other("recipe output must be a fresh absolute directory").into());
    }
    if mode != ProgramPreprocessingMode::Full || guest != GuestProgram::Fibonacci || proofs != 1 {
        return Err(io::Error::other(
            "recipe generation requires exactly one Full Fibonacci proof",
        )
        .into());
    }
    if AkitaField::immutable_recipe_native_field().is_none() {
        return Err(
            io::Error::other("native recipe generation requires the admitted Fp128 type").into(),
        );
    }
    let scalar: u32 = std::env::var("RECURSION_FIB_N")?.parse()?;
    let input = postcard::to_stdvec(&scalar)?;
    let input_hash = hash(&input);
    Ok(Some(Request {
        output,
        scalar,
        input,
        input_hash,
        trace_length: None,
    }))
}

pub(crate) fn inputs(
    request: &Option<Request>,
    canonical: impl FnOnce() -> Vec<Vec<u8>>,
) -> Vec<Vec<u8>> {
    match request {
        Some(request) => vec![request.input.clone()],
        None => canonical(),
    }
}

/// A capture-start error is deferred until canonical verification has returned.
pub(crate) fn begin(
    request: Option<Request>,
    n: u32,
    actual_count: usize,
    max_trace_length: usize,
) -> Option<Pending> {
    request.map(|request| {
        let session = if n != 1 || actual_count != 1 {
            Err(io::Error::other("canonical generation proof count changed").into())
        } else {
            Session::begin().map_err(Into::into)
        };
        Pending {
            request,
            max_trace_length,
            session,
        }
    })
}

impl Pending {
    pub(crate) fn finish(
        self,
        records: &[(GuestProof, JoltDevice)],
        setup: &GuestVerifierPreprocessing,
        payloads: &[Vec<u8>],
    ) -> Result<Ready> {
        let capture = self.session?.finish()?;
        let [(proof, public_io)] = records else {
            return Err(io::Error::other("canonical generation proof count changed").into());
        };
        if !matches!(proof.trace_length, 4096 | 1048576) {
            return Err(io::Error::other(
                "canonical generation is outside the two admitted padded sizes",
            )
            .into());
        }
        let actual_trace_length = self
            .request
            .trace_length
            .ok_or_else(|| io::Error::other("canonical unpadded trace observation missing"))?;
        let proof_bytes = bincode::serde::encode_to_vec(proof, bincode::config::standard())?;
        let public_io_bytes =
            bincode::serde::encode_to_vec(public_io, bincode::config::standard())?;
        let setup_bytes = bincode::serde::encode_to_vec(setup, bincode::config::standard())?;
        if public_io.inputs != self.request.input {
            return Err(io::Error::other(
                "accepted public input differs from the pre-trace request",
            )
            .into());
        }
        let executable = fs::read(std::env::current_exe()?)?;
        Ok(Ready {
            output: self.request.output, input: self.request.input,
            context: Context {
                status: "CANONICAL_NATIVE_RECORDING_ONLY",
                guest: "fibonacci-guest", proof_count: 1, max_trace_length: self.max_trace_length,
                requested_scalar: self.request.scalar, requested_input_sha256: self.request.input_hash,
                actual_trace_length, padded_trace_length: proof.trace_length,
                input_sha256: hash(&public_io.inputs),
                accepted_proof_encoding_sha256: hash(&proof_bytes),
                accepted_public_io_encoding_sha256: hash(&public_io_bytes),
                prepared_setup_encoding_sha256: hash(&setup_bytes),
                detached_payload_sha256: payloads.iter().map(|payload| hash(payload)).collect(),
                host_executable_sha256: hash(&executable),
                interpretation: "Canonical encodings of the same typed values accepted by the original single verification. Native verification did not parse the final stream. No HP05 byte identity or source/guest qualification is inferred.",
            }, capture,
        })
    }
}

struct Staging {
    path: PathBuf,
    published: bool,
}
impl Drop for Staging {
    fn drop(&mut self) {
        if !self.published {
            for name in [
                "recipes.rs",
                "capture.json",
                "context.json",
                "accepted-stream.bin",
                "input.bin",
                "SHA256SUMS",
            ] {
                let _ = fs::remove_file(self.path.join(name));
            }
            let _ = fs::remove_dir(&self.path);
        }
    }
}
impl Staging {
    fn write(&self, name: &str, bytes: &[u8]) -> Result<String> {
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(self.path.join(name))?;
        file.write_all(bytes)?;
        file.sync_all()?;
        Ok(format!("{}  {name}\n", hash(bytes)))
    }
}

/// Publication follows the existing serialization of the accepted typed values.
/// No source file is written while capture/coverage/serialization can still fail.
pub(crate) fn publish(pending: Option<Result<Ready>>, stream: &[u8]) -> Result<()> {
    let Some(pending) = pending else {
        return Ok(());
    };
    let mut ready = pending?;
    let source = ready.capture.source()?;
    let identities = ready.capture.identities()?;
    let capture = serde_json::to_vec_pretty(&serde_json::json!({
        "status": "CLOSED_CANONICAL_RECIPES_GENERATED_NOT_QUALIFIED",
        "logical_capture_bytes": ready.capture.logical_bytes,
        "factor_events": ready.capture.factor_events,
        "records": identities,
        "fallback_keys": &ready.capture.fallbacks,
        "serialized_stream_sha256": hash(stream),
        "source_sha256": hash(source.as_bytes()),
        "source_bytes": source.len(),
    }))?;
    let context = serde_json::to_vec_pretty(&ready.context)?;
    if ready.output.exists() {
        return Err(io::Error::other("recipe output already exists").into());
    }
    let parent = ready
        .output
        .parent()
        .ok_or_else(|| io::Error::other("recipe output has no parent"))?;
    let name = ready
        .output
        .file_name()
        .ok_or_else(|| io::Error::other("recipe output has no name"))?;
    let mut temporary_name = name.to_os_string();
    temporary_name.push(format!(".staging-{}", std::process::id()));
    let temporary = parent.join(temporary_name);
    fs::create_dir(&temporary)?;
    let mut staging = Staging {
        path: temporary,
        published: false,
    };
    let mut manifest = String::new();
    manifest.push_str(&staging.write("recipes.rs", source.as_bytes())?);
    manifest.push_str(&staging.write("capture.json", &capture)?);
    manifest.push_str(&staging.write("context.json", &context)?);
    manifest.push_str(&staging.write("accepted-stream.bin", stream)?);
    manifest.push_str(&staging.write("input.bin", &ready.input)?);
    let _ = staging.write("SHA256SUMS", manifest.as_bytes())?;
    if ready.output.exists() {
        return Err(io::Error::other("recipe output appeared during staging").into());
    }
    fs::rename(&staging.path, &ready.output)?;
    staging.published = true;
    Ok(())
}
