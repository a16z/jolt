use std::{
    fs,
    path::{Path, PathBuf},
};

#[test]
fn prover_stages_use_linear_input_request_prove_output_shape() -> Result<(), String> {
    let stages = workspace_root()?.join("crates/jolt-prover/src/stages");
    for entry in fs::read_dir(&stages).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let name = path
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| "invalid stage directory name".to_owned())?;
        if !name.starts_with("stage") {
            continue;
        }

        assert_file(&path, "input.rs")?;
        assert_file(&path, "request.rs")?;
        assert_file(&path, "prove.rs")?;
        assert_file(&path, "output.rs")?;
        assert_absent(&path, "assembly.rs")?;
    }
    Ok(())
}

#[test]
fn backend_commitment_code_is_split_by_contract_and_compute() -> Result<(), String> {
    let root = workspace_root()?.join("crates/jolt-backends/src");

    for family in ["commitments", "sumcheck", "openings"] {
        assert_file(&root.join(family), "request.rs")?;
        assert_file(&root.join(family), "result.rs")?;
        assert_file(&root.join(family), "CONTRACT.md")?;
    }
    assert_file(&root.join("cpu/commitments"), "mod.rs")?;
    assert_file(&root.join("cpu/commitments"), "stream.rs")?;
    assert_file(&root.join("cpu/sumcheck"), "mod.rs")?;
    assert_file(&root.join("cpu/openings"), "mod.rs")?;
    assert_absent(&root, "commitments.rs")?;
    assert_absent(&root.join("cpu"), "commitments.rs")?;
    Ok(())
}

#[test]
fn zk_and_field_inline_backend_code_is_cfg_isolated() -> Result<(), String> {
    let root = workspace_root()?.join("crates/jolt-backends/src");

    assert_file(&root.join("blindfold"), "request.rs")?;
    assert_file(&root.join("blindfold"), "result.rs")?;
    assert_file(&root.join("blindfold"), "CONTRACT.md")?;
    assert_file(&root.join("cpu/blindfold"), "mod.rs")?;
    assert_file(&root.join("cpu/field_inline"), "mod.rs")?;
    Ok(())
}

#[test]
fn cpu_backend_keeps_family_modules_nested() -> Result<(), String> {
    let cpu = workspace_root()?.join("crates/jolt-backends/src/cpu");
    let allowed_flat_files = ["backend.rs", "config.rs", "mod.rs", "state.rs", "tests.rs"];
    let mut offenders = Vec::new();

    for entry in fs::read_dir(&cpu).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        let is_rust_file = path
            .extension()
            .and_then(|extension| extension.to_str())
            .is_some_and(|extension| extension.eq_ignore_ascii_case("rs"));
        if is_rust_file && !allowed_flat_files.contains(&file_name) {
            offenders.push(file_name.to_owned());
        }
    }

    if offenders.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "CPU backend family code should live in nested modules: {}",
            offenders.join(", ")
        ))
    }
}

#[test]
fn prover_features_forward_to_backend_capability_features() -> Result<(), String> {
    let cargo_toml = fs::read_to_string(workspace_root()?.join("crates/jolt-prover/Cargo.toml"))
        .map_err(|error| error.to_string())?;

    assert!(cargo_toml.contains("\"jolt-backends/field-inline\""));
    assert!(cargo_toml.contains("\"jolt-backends/zk\""));
    Ok(())
}

fn workspace_root() -> Result<PathBuf, String> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .map(Path::to_path_buf)
        .ok_or_else(|| "failed to locate workspace root".to_owned())
}

fn assert_file(dir: &Path, file_name: &str) -> Result<(), String> {
    let path = dir.join(file_name);
    if path.is_file() {
        Ok(())
    } else {
        Err(format!("missing required file {}", path.display()))
    }
}

fn assert_absent(dir: &Path, file_name: &str) -> Result<(), String> {
    let path = dir.join(file_name);
    if path.exists() {
        Err(format!("unexpected file {}", path.display()))
    } else {
        Ok(())
    }
}
