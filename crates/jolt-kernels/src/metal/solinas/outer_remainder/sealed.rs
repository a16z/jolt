use std::{
    collections::BTreeSet,
    fmt::Write as _,
    fs::{self, File},
    io::Read,
    os::unix::fs::MetadataExt,
    path::Path,
};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{
    api::{OuterRemainderSequenceConfig, OuterRemainderStorageInitialization},
    artifact::{OuterBindingPlan, OuterKernelArtifact},
    plan::opening_layout,
    shader::pipeline_names,
};
use crate::metal::solinas::{MetalError, SolinasMetal};

const ARTIFACT_SCHEMA: &str = "jolt_outer_artifact_v2";
const ARTIFACT_SCHEMA_VERSION: u32 = 2;
const SOURCE_FILE: &str = "outer.metal";
const MANIFEST_FILE: &str = "manifest.json";
const MAX_SOURCE_BYTES: usize = 2 * 1024 * 1024;
const MAX_MANIFEST_BYTES: usize = 64 * 1024;

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Entrypoints {
    dense_bind: String,
    materialize: String,
    opening: String,
    reduction: String,
    stream_bind: String,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct OpeningLayout {
    row_stride_words: usize,
    source_row_words: usize,
    tile_rows: usize,
    uses_shard_sums: bool,
}

#[derive(Debug, Serialize)]
struct OuterAbi<'a> {
    opening_layout: OpeningLayout,
    required_entrypoints: &'a Entrypoints,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Dispatch {
    cpu_tail_elements: usize,
    materialize_threads: usize,
    max_threadgroups: usize,
    opening_threads: usize,
    storage_initialization: StorageInitialization,
    stream_bind_threads: usize,
    trace_cutoff_elements: usize,
    transition_threads: usize,
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
enum StorageInitialization {
    #[serde(rename = "full")]
    Full,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Manifest {
    schema: String,
    schema_version: u32,
    source_file: String,
    source_bytes: usize,
    outer_source_sha256: String,
    dispatch: Dispatch,
    dispatch_sha256: String,
    binding_plan: String,
    binding_plan_sha256: String,
    opening_layout: OpeningLayout,
    opening_layout_sha256: String,
    outer_abi_sha256: String,
    required_entrypoints: Entrypoints,
}

pub struct SealedOuterArtifact {
    kernel: OuterKernelArtifact,
    artifact_sha256: String,
    outer_source_sha256: String,
    dispatch: Dispatch,
}

impl SealedOuterArtifact {
    pub fn load(path: &Path) -> Result<Self, MetalError> {
        validate_directory(path)?;
        let manifest_bytes =
            read_regular_file(&path.join(MANIFEST_FILE), MAX_MANIFEST_BYTES, "manifest")?;
        let value: serde_json::Value = serde_json::from_slice(&manifest_bytes)
            .map_err(|error| invalid(format!("manifest JSON is invalid: {error}")))?;
        let canonical = serde_json::to_vec(&value)
            .map_err(|error| invalid(format!("manifest cannot be canonicalized: {error}")))?;
        if canonical != manifest_bytes {
            return Err(invalid("manifest encoding is not canonical"));
        }
        let manifest: Manifest = serde_json::from_value(value)
            .map_err(|error| invalid(format!("manifest contract is invalid: {error}")))?;
        let source = read_regular_file(&path.join(SOURCE_FILE), MAX_SOURCE_BYTES, "source")?;
        validate_manifest(&manifest, &source)?;

        let artifact_sha256 = digest_parts(&[&manifest_bytes, b"\0", &source]);
        let directory_name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| invalid("artifact directory name is invalid"))?;
        if directory_name != artifact_sha256 {
            return Err(invalid("artifact directory does not match its digest"));
        }
        let binding_plan = OuterBindingPlan::from_id(&manifest.binding_plan)
            .ok_or_else(|| invalid("binding plan is unsupported"))?;
        let source_text = String::from_utf8(source)
            .map_err(|error| invalid(format!("source is not UTF-8: {error}")))?;
        let kernel = OuterKernelArtifact::new(source_text, binding_plan)?;
        Ok(Self {
            kernel,
            artifact_sha256,
            outer_source_sha256: manifest.outer_source_sha256,
            dispatch: manifest.dispatch,
        })
    }

    pub fn compile_akita(&self) -> Result<SolinasMetal, MetalError> {
        SolinasMetal::for_akita_outer_only_with_artifact(&self.kernel)
    }

    pub fn sequence_config(&self) -> OuterRemainderSequenceConfig {
        OuterRemainderSequenceConfig {
            binding_plan: self.kernel.binding_plan(),
            materialize_threads_per_threadgroup: Some(self.dispatch.materialize_threads),
            stream_bind_threads_per_threadgroup: Some(self.dispatch.stream_bind_threads),
            transition_threads_per_threadgroup: Some(self.dispatch.transition_threads),
            opening_threads_per_threadgroup: Some(self.dispatch.opening_threads),
            max_threadgroups: self.dispatch.max_threadgroups,
            cpu_tail_elements: self.dispatch.cpu_tail_elements,
            storage_initialization: OuterRemainderStorageInitialization::Full,
        }
    }

    pub fn artifact_sha256(&self) -> &str {
        &self.artifact_sha256
    }

    pub fn outer_source_sha256(&self) -> &str {
        &self.outer_source_sha256
    }

    pub const fn trace_cutoff_elements(&self) -> usize {
        self.dispatch.trace_cutoff_elements
    }

    pub const fn binding_plan(&self) -> OuterBindingPlan {
        self.kernel.binding_plan()
    }
}

fn validate_directory(path: &Path) -> Result<(), MetalError> {
    let metadata = fs::symlink_metadata(path)
        .map_err(|error| invalid(format!("artifact directory cannot be read: {error}")))?;
    if !metadata.file_type().is_dir() || metadata.file_type().is_symlink() {
        return Err(invalid("artifact path must be a real directory"));
    }
    let mut entries = BTreeSet::new();
    for entry in fs::read_dir(path)
        .map_err(|error| invalid(format!("artifact directory cannot be listed: {error}")))?
    {
        let entry = entry
            .map_err(|error| invalid(format!("artifact directory entry is invalid: {error}")))?;
        let name = entry
            .file_name()
            .into_string()
            .map_err(|_| invalid("artifact directory contains a non-UTF-8 name"))?;
        let _ = entries.insert(name);
    }
    let expected = BTreeSet::from([MANIFEST_FILE.to_owned(), SOURCE_FILE.to_owned()]);
    if entries != expected {
        return Err(invalid("artifact directory has unexpected files"));
    }
    Ok(())
}

fn read_regular_file(
    path: &Path,
    maximum_bytes: usize,
    description: &str,
) -> Result<Vec<u8>, MetalError> {
    let before = fs::symlink_metadata(path)
        .map_err(|error| invalid(format!("{description} cannot be inspected: {error}")))?;
    if !before.file_type().is_file() || before.file_type().is_symlink() {
        return Err(invalid(format!("{description} must be a regular file")));
    }
    let size = usize::try_from(before.size())
        .map_err(|_| invalid(format!("{description} size does not fit usize")))?;
    if size == 0 || size > maximum_bytes {
        return Err(invalid(format!("{description} size is invalid")));
    }

    let mut file = File::open(path)
        .map_err(|error| invalid(format!("{description} cannot be opened: {error}")))?;
    let opened = file
        .metadata()
        .map_err(|error| invalid(format!("{description} metadata cannot be read: {error}")))?;
    if !same_file(&before, &opened) {
        return Err(invalid(format!(
            "{description} changed while it was admitted"
        )));
    }
    let mut bytes = Vec::with_capacity(size);
    let bytes_read = file
        .read_to_end(&mut bytes)
        .map_err(|error| invalid(format!("{description} cannot be read: {error}")))?;
    let after = file
        .metadata()
        .map_err(|error| invalid(format!("{description} metadata cannot be reread: {error}")))?;
    if bytes_read != size || bytes.len() != size || !same_file(&opened, &after) {
        return Err(invalid(format!("{description} changed while it was read")));
    }
    Ok(bytes)
}

fn same_file(left: &fs::Metadata, right: &fs::Metadata) -> bool {
    left.file_type().is_file()
        && right.file_type().is_file()
        && left.dev() == right.dev()
        && left.ino() == right.ino()
        && left.size() == right.size()
        && left.mtime() == right.mtime()
        && left.mtime_nsec() == right.mtime_nsec()
        && left.ctime() == right.ctime()
        && left.ctime_nsec() == right.ctime_nsec()
}

fn validate_manifest(manifest: &Manifest, source: &[u8]) -> Result<(), MetalError> {
    if manifest.schema != ARTIFACT_SCHEMA
        || manifest.schema_version != ARTIFACT_SCHEMA_VERSION
        || manifest.source_file != SOURCE_FILE
        || manifest.source_bytes != source.len()
        || manifest.outer_source_sha256 != digest_parts(&[source])
    {
        return Err(invalid("manifest does not match its source"));
    }
    let binding_plan = OuterBindingPlan::from_id(&manifest.binding_plan)
        .ok_or_else(|| invalid("binding plan is unsupported"))?;
    let entrypoints = expected_entrypoints(binding_plan);
    let layout = expected_opening_layout(binding_plan);
    let abi = OuterAbi {
        opening_layout: layout,
        required_entrypoints: &entrypoints,
    };
    if manifest.binding_plan_sha256 != digest_parts(&[manifest.binding_plan.as_bytes()])
        || manifest.required_entrypoints != entrypoints
        || manifest.opening_layout != layout
        || manifest.opening_layout_sha256
            != canonical_digest(&manifest.opening_layout, "opening layout")?
        || manifest.outer_abi_sha256 != canonical_digest(&abi, "entrypoint ABI")?
        || manifest.dispatch_sha256 != canonical_digest(&manifest.dispatch, "dispatch")?
    {
        return Err(invalid("manifest ABI or dispatch digest is invalid"));
    }
    validate_dispatch(manifest.dispatch)
}

fn expected_opening_layout(plan: OuterBindingPlan) -> OpeningLayout {
    let layout = opening_layout(plan);
    OpeningLayout {
        row_stride_words: layout.row_stride_words,
        source_row_words: layout.source_row_words,
        tile_rows: layout.tile_rows,
        uses_shard_sums: layout.shard_sums,
    }
}

fn expected_entrypoints(plan: OuterBindingPlan) -> Entrypoints {
    let names = pipeline_names(plan);
    Entrypoints {
        dense_bind: names.transition.to_owned(),
        materialize: names.materialize.to_owned(),
        opening: names.opening.to_owned(),
        reduction: names.reduction.to_owned(),
        stream_bind: names.stream_bind.to_owned(),
    }
}

fn validate_dispatch(dispatch: Dispatch) -> Result<(), MetalError> {
    for threads in [
        dispatch.materialize_threads,
        dispatch.stream_bind_threads,
        dispatch.transition_threads,
        dispatch.opening_threads,
    ] {
        if !(32..=1024).contains(&threads) || !threads.is_power_of_two() {
            return Err(invalid("dispatch threadgroup width is invalid"));
        }
    }
    if dispatch.stream_bind_threads != dispatch.transition_threads
        || dispatch.max_threadgroups != 8192
        || dispatch.cpu_tail_elements < 2
        || !dispatch.cpu_tail_elements.is_power_of_two()
        || dispatch.trace_cutoff_elements < 2
        || !dispatch.trace_cutoff_elements.is_power_of_two()
    {
        return Err(invalid("dispatch geometry is invalid"));
    }
    Ok(())
}

fn canonical_digest(value: &impl Serialize, description: &str) -> Result<String, MetalError> {
    let value = serde_json::to_value(value)
        .map_err(|error| invalid(format!("{description} cannot be encoded: {error}")))?;
    let bytes = serde_json::to_vec(&value)
        .map_err(|error| invalid(format!("{description} cannot be canonicalized: {error}")))?;
    Ok(digest_parts(&[&bytes]))
}

fn digest_parts(parts: &[&[u8]]) -> String {
    let mut digest = Sha256::new();
    for part in parts {
        digest.update(part);
    }
    let mut encoded = String::with_capacity(64);
    for byte in digest.finalize() {
        let _ = write!(encoded, "{byte:02x}");
    }
    encoded
}

fn invalid(message: impl Into<String>) -> MetalError {
    MetalError::InvalidSealedOuterArtifact(message.into())
}
