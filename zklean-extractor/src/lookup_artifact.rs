//! Deterministic standalone artifacts for generated lookup tables.

use std::{
    collections::BTreeMap,
    fs, io,
    path::{Path, PathBuf},
    process::Command,
};

use serde::Serialize;
use sha2::{Digest, Sha256};

use crate::{
    lookups::ZkLeanLookupTables,
    modules::{AsModule, Module},
};

const FORMAT_VERSION: u32 = 1;
const MANIFEST_PATH: &str = "lookup-artifact.json";
const SOURCE_REPOSITORY: &str = "https://github.com/a16z/jolt";

const RUNTIME_FILES: [(&str, &[u8]); 6] = [
    (
        "Jolt/LookupExpression.lean",
        include_bytes!("../package-template/Jolt/LookupExpression.lean"),
    ),
    (
        "Jolt/LookupGraph.lean",
        include_bytes!("../package-template/Jolt/LookupGraph.lean"),
    ),
    (
        "Jolt/LookupGraphExpression.lean",
        include_bytes!("../package-template/Jolt/LookupGraphExpression.lean"),
    ),
    (
        "Jolt/MaterializerGraph.lean",
        include_bytes!("../package-template/Jolt/MaterializerGraph.lean"),
    ),
    (
        "Jolt/LookupAC.lean",
        include_bytes!("../package-template/Jolt/LookupAC.lean"),
    ),
    (
        "Jolt/LookupProgram.lean",
        include_bytes!("../package-template/Jolt/LookupProgram.lean"),
    ),
];

#[derive(Debug, Serialize)]
struct ArtifactFile {
    path: String,
    sha256: String,
}

#[derive(Debug, Serialize)]
struct ArtifactManifest {
    format_version: u32,
    source_repository: &'static str,
    source_revision: String,
    generator: &'static str,
    xlen: usize,
    artifact_sha256: String,
    files: Vec<ArtifactFile>,
}

/// A deterministic set of generated Lean lookup files and its provenance manifest.
#[derive(Debug, PartialEq, Eq)]
pub struct LookupArtifact {
    files: BTreeMap<String, Vec<u8>>,
    manifest: Vec<u8>,
}

impl LookupArtifact {
    /// Extract all lookup tables and attach the exact Jolt revision they represent.
    pub fn extract<const XLEN: usize>(source_revision: &str) -> Result<Self, String> {
        let source_revision = validate_source_revision(source_revision)?;
        let lookup_tables = ZkLeanLookupTables::<XLEN>::extract()?;
        let lookup_module = lookup_tables
            .as_module()
            .map_err(|error| error.to_string())?;

        let mut files = RUNTIME_FILES
            .into_iter()
            .map(|(path, contents)| (path.to_string(), contents.to_vec()))
            .collect::<BTreeMap<_, _>>();
        files.insert(
            "Jolt/LookupTables.lean".to_string(),
            render_module(lookup_module),
        );

        let file_entries = files
            .iter()
            .map(|(path, contents)| ArtifactFile {
                path: path.clone(),
                sha256: sha256_hex(contents),
            })
            .collect::<Vec<_>>();
        let manifest = ArtifactManifest {
            format_version: FORMAT_VERSION,
            source_repository: SOURCE_REPOSITORY,
            source_revision,
            generator: env!("CARGO_PKG_VERSION"),
            xlen: XLEN,
            artifact_sha256: hash_file_set(&files),
            files: file_entries,
        };
        let mut manifest =
            serde_json::to_vec_pretty(&manifest).map_err(|error| error.to_string())?;
        manifest.push(b'\n');

        Ok(Self { files, manifest })
    }

    /// Write the artifact without deleting unrelated files from an existing directory.
    pub fn write_to(&self, root: &Path, overwrite: bool) -> io::Result<()> {
        if root.exists() && !overwrite {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("artifact directory already exists: {root:?}"),
            ));
        }

        for (relative_path, contents) in &self.files {
            write_file(root, relative_path, contents)?;
        }
        write_file(root, MANIFEST_PATH, &self.manifest)
    }
}

/// Check that an artifact revision names the clean Jolt checkout used to generate it.
pub fn verify_checkout_revision(source_revision: &str) -> Result<(), String> {
    verify_checkout_revision_at(Path::new("."), source_revision)
}

fn verify_checkout_revision_at(checkout: &Path, source_revision: &str) -> Result<(), String> {
    let source_revision = validate_source_revision(source_revision)?;
    let head = git_output(checkout, ["rev-parse", "HEAD"])?;
    if head != source_revision {
        return Err(format!(
            "source revision {source_revision} does not match checked out Jolt revision {head}"
        ));
    }

    let tracked_changes = git_output(checkout, ["status", "--porcelain", "--untracked-files=no"])?;
    if !tracked_changes.is_empty() {
        return Err(
            "tracked Jolt files are dirty; commit them before exporting an artifact".into(),
        );
    }
    Ok(())
}

fn git_output<const N: usize>(checkout: &Path, arguments: [&str; N]) -> Result<String, String> {
    let output = Command::new("git")
        .current_dir(checkout)
        .args(arguments)
        .output()
        .map_err(|error| format!("failed to run git: {error}"))?;
    if !output.status.success() {
        return Err(format!(
            "git command failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        ));
    }
    String::from_utf8(output.stdout)
        .map(|value| value.trim().to_string())
        .map_err(|error| format!("git output was not UTF-8: {error}"))
}

fn validate_source_revision(source_revision: &str) -> Result<String, String> {
    if source_revision.len() != 40 || !source_revision.bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err("source revision must contain exactly 40 hexadecimal characters".to_string());
    }
    Ok(source_revision.to_ascii_lowercase())
}

fn render_module(module: Module) -> Vec<u8> {
    module
        .imports
        .into_iter()
        .flat_map(|import| format!("import {import}\n").into_bytes())
        .chain([b'\n'])
        .chain(module.contents)
        .collect()
}

fn write_file(root: &Path, relative_path: &str, contents: &[u8]) -> io::Result<()> {
    let path = root.join(PathBuf::from(relative_path));
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, contents)
}

fn sha256_hex(contents: &[u8]) -> String {
    hex_encode(Sha256::digest(contents).as_ref())
}

fn hash_file_set(files: &BTreeMap<String, Vec<u8>>) -> String {
    let mut hasher = Sha256::new();
    for (path, contents) in files {
        hasher.update((path.len() as u64).to_be_bytes());
        hasher.update(path.as_bytes());
        hasher.update((contents.len() as u64).to_be_bytes());
        hasher.update(contents);
    }
    hex_encode(hasher.finalize().as_ref())
}

fn hex_encode(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    bytes
        .iter()
        .flat_map(|byte| {
            [
                DIGITS[(byte >> 4) as usize] as char,
                DIGITS[(byte & 0x0f) as usize] as char,
            ]
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use serde_json::Value;
    use tempfile::TempDir;

    use super::*;

    const REVISION: &str = "0123456789abcdef0123456789abcdef01234567";

    #[test]
    fn artifact_is_deterministic_and_records_provenance() {
        let first = LookupArtifact::extract::<8>(REVISION).unwrap();
        let second = LookupArtifact::extract::<8>(REVISION).unwrap();
        assert_eq!(first, second);

        let manifest: Value = serde_json::from_slice(&first.manifest).unwrap();
        assert_eq!(manifest["format_version"], 1);
        assert_eq!(manifest["source_revision"], REVISION);
        assert_eq!(manifest["xlen"], 8);
        assert_eq!(manifest["files"].as_array().unwrap().len(), 7);
        assert!(first.files.contains_key("Jolt/LookupTables.lean"));
    }

    #[test]
    fn artifact_rejects_ambiguous_source_revisions() {
        assert!(LookupArtifact::extract::<8>("main").is_err());
        assert!(LookupArtifact::extract::<8>("0123456789abcdef0123456789abcdef0123456g").is_err());
    }

    fn git(checkout: &Path, arguments: &[&str]) -> String {
        let output = Command::new("git")
            .current_dir(checkout)
            .args(arguments)
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "git failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        String::from_utf8(output.stdout).unwrap().trim().to_string()
    }

    fn clean_checkout() -> (TempDir, String) {
        let checkout = tempfile::tempdir().unwrap();
        git(checkout.path(), &["init", "--quiet"]);
        git(checkout.path(), &["config", "user.name", "Jolt test"]);
        git(
            checkout.path(),
            &["config", "user.email", "jolt-test@example.com"],
        );
        fs::write(checkout.path().join("tracked"), "committed\n").unwrap();
        git(checkout.path(), &["add", "tracked"]);
        git(checkout.path(), &["commit", "--quiet", "-m", "initial"]);
        let head = git(checkout.path(), &["rev-parse", "HEAD"]);
        (checkout, head)
    }

    #[test]
    fn checkout_revision_accepts_exact_clean_head() {
        let (checkout, head) = clean_checkout();
        assert_eq!(verify_checkout_revision_at(checkout.path(), &head), Ok(()));
    }

    #[test]
    fn checkout_revision_rejects_wrong_head() {
        let (checkout, head) = clean_checkout();
        let wrong_head = if head.starts_with('0') {
            format!("1{}", &head[1..])
        } else {
            format!("0{}", &head[1..])
        };
        let error = verify_checkout_revision_at(checkout.path(), &wrong_head).unwrap_err();
        assert!(error.contains("does not match checked out Jolt revision"));
    }

    #[test]
    fn checkout_revision_rejects_dirty_tracked_files() {
        let (checkout, head) = clean_checkout();
        fs::write(checkout.path().join("tracked"), "modified\n").unwrap();
        let error = verify_checkout_revision_at(checkout.path(), &head).unwrap_err();
        assert_eq!(
            error,
            "tracked Jolt files are dirty; commit them before exporting an artifact"
        );
    }
}
