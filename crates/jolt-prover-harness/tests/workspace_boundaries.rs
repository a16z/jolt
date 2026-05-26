use std::{
    fs,
    path::{Path, PathBuf},
};

use toml_edit::{DocumentMut, Item, Table};

#[test]
fn production_dependencies_do_not_use_prover_harness() -> Result<(), String> {
    let workspace = workspace_root()?;
    let mut manifests = Vec::new();
    collect_manifests(&workspace, &mut manifests)?;

    let mut offenders = Vec::new();
    for manifest in manifests {
        let relative = manifest
            .strip_prefix(&workspace)
            .map_err(|error| error.to_string())?;
        if relative == Path::new("Cargo.toml")
            || relative == Path::new("crates/jolt-prover-harness/Cargo.toml")
        {
            continue;
        }

        let content = fs::read_to_string(&manifest).map_err(|error| error.to_string())?;
        let document = content
            .parse::<DocumentMut>()
            .map_err(|error| format!("{}: {error}", relative.display()))?;
        scan_dependency_tables(
            document.as_table(),
            &mut Vec::new(),
            relative,
            &mut offenders,
        );
    }

    if offenders.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "jolt-prover-harness is dev/test-only; production dependency offenders: {}",
            offenders.join(", ")
        ))
    }
}

#[test]
fn backend_does_not_depend_on_protocol_or_core_crates() -> Result<(), String> {
    assert_manifest_has_no_production_deps(
        Path::new("crates/jolt-backends/Cargo.toml"),
        &["jolt-core", "jolt-verifier", "jolt-claims"],
    )
}

#[test]
fn prover_does_not_depend_on_core_or_tracer() -> Result<(), String> {
    assert_manifest_has_no_production_deps(
        Path::new("crates/jolt-prover/Cargo.toml"),
        &["jolt-core", "tracer"],
    )
}

fn workspace_root() -> Result<PathBuf, String> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .map(Path::to_path_buf)
        .ok_or_else(|| "failed to locate workspace root".to_owned())
}

fn collect_manifests(root: &Path, manifests: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in fs::read_dir(root).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let path = entry.path();
        let file_name = entry.file_name();
        let file_name = file_name.to_string_lossy();

        if path.is_dir() {
            if should_skip_dir(&file_name) {
                continue;
            }
            collect_manifests(&path, manifests)?;
        } else if file_name == "Cargo.toml" {
            manifests.push(path);
        }
    }
    Ok(())
}

fn should_skip_dir(file_name: &str) -> bool {
    matches!(
        file_name,
        ".git" | ".jj" | "target" | ".nextest" | "node_modules"
    )
}

fn scan_dependency_tables(
    table: &Table,
    path: &mut Vec<String>,
    manifest: &Path,
    offenders: &mut Vec<String>,
) {
    for (key, item) in table {
        let Some(child) = item.as_table() else {
            continue;
        };
        path.push(key.to_owned());
        if is_production_dependency_table(path) && child.contains_key("jolt-prover-harness") {
            offenders.push(format!("{}:[{}]", manifest.display(), path.join(".")));
        }
        scan_inline_tables(item, path, manifest, offenders);
        scan_dependency_tables(child, path, manifest, offenders);
        let _ = path.pop();
    }
}

fn scan_inline_tables(
    item: &Item,
    path: &mut Vec<String>,
    manifest: &Path,
    offenders: &mut Vec<String>,
) {
    let Some(inline_table) = item.as_inline_table() else {
        return;
    };
    for (key, value) in inline_table {
        let Some(child) = value.as_inline_table() else {
            continue;
        };
        path.push(key.to_owned());
        if is_production_dependency_table(path) && child.contains_key("jolt-prover-harness") {
            offenders.push(format!("{}:[{}]", manifest.display(), path.join(".")));
        }
        let _ = path.pop();
    }
}

fn is_production_dependency_table(path: &[String]) -> bool {
    matches!(
        path.last().map(String::as_str),
        Some("dependencies" | "build-dependencies")
    )
}

fn assert_manifest_has_no_production_deps(
    relative_manifest: &Path,
    forbidden: &[&str],
) -> Result<(), String> {
    let workspace = workspace_root()?;
    let manifest = workspace.join(relative_manifest);
    let content = fs::read_to_string(&manifest).map_err(|error| error.to_string())?;
    let document = content
        .parse::<DocumentMut>()
        .map_err(|error| format!("{}: {error}", relative_manifest.display()))?;
    let mut offenders = Vec::new();
    scan_for_forbidden_dependencies(
        document.as_table(),
        &mut Vec::new(),
        relative_manifest,
        forbidden,
        &mut offenders,
    );

    if offenders.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "forbidden production dependencies: {}",
            offenders.join(", ")
        ))
    }
}

fn scan_for_forbidden_dependencies(
    table: &Table,
    path: &mut Vec<String>,
    manifest: &Path,
    forbidden: &[&str],
    offenders: &mut Vec<String>,
) {
    for (key, item) in table {
        let Some(child) = item.as_table() else {
            continue;
        };
        path.push(key.to_owned());
        if is_production_dependency_table(path) {
            for dependency in forbidden {
                if child.contains_key(dependency) {
                    offenders.push(format!(
                        "{}:[{}].{}",
                        manifest.display(),
                        path.join("."),
                        dependency
                    ));
                }
            }
        }
        scan_for_forbidden_dependencies(child, path, manifest, forbidden, offenders);
        let _ = path.pop();
    }
}
