use std::{
    fs,
    path::{Path, PathBuf},
};

#[test]
fn prover_does_not_import_concrete_cpu_backend() -> Result<(), String> {
    assert_no_patterns(
        &[workspace_root()?.join("crates/jolt-prover/src")],
        &["jolt_backends::cpu", "CpuBackend"],
    )
}

#[test]
fn backends_do_not_use_transcripts() -> Result<(), String> {
    assert_no_patterns(
        &[workspace_root()?.join("crates/jolt-backends/src")],
        &["jolt_transcript", "Transcript", "transcript"],
    )
}

#[test]
fn prover_and_witness_do_not_consume_tracer_internals() -> Result<(), String> {
    assert_no_patterns(
        &[
            workspace_root()?.join("crates/jolt-prover/src"),
            workspace_root()?.join("crates/jolt-witness/src"),
        ],
        &[
            "tracer::",
            "use tracer::",
            "tracer::emulator",
            "tracer::Cycle",
        ],
    )
}

#[test]
fn prover_and_backends_do_not_own_lookup_semantics() -> Result<(), String> {
    assert_no_patterns(
        &[
            workspace_root()?.join("crates/jolt-prover/src"),
            workspace_root()?.join("crates/jolt-backends/src"),
        ],
        &["combine_lookup", "JoltLookup"],
    )
}

#[test]
fn prover_docs_do_not_use_retired_architecture_terms() -> Result<(), String> {
    let workspace = workspace_root()?;
    let roots = [
        workspace.join("specs/jolt-prover-model-crate.md"),
        workspace.join("specs/jolt-prover-cpu-backend-port.md"),
        workspace.join("specs/jolt-prover-frontier-harness.md"),
        workspace.join("specs/jolt-witness-crate.md"),
        workspace.join("specs/field-inline-program-tracer.md"),
        workspace.join("specs/jolt-core-prover-optimization-inventory.md"),
        workspace.join("crates/jolt-prover/review.md"),
        workspace.join("crates/jolt-backends/README.md"),
        workspace.join("crates/jolt-prover-harness/README.md"),
        workspace.join("crates/jolt-backends/src"),
    ];

    assert_no_patterns_in_markdown(
        &roots,
        &[
            "backend traits/plans",
            "protocol-resolved plans",
            "selected plans",
            "opening plans",
            "commitment plan",
            "witness/prover plan",
            "proof assembly",
            "graft",
            "placeholder",
            "prelim",
            "fake final",
        ],
    )
}

#[test]
fn north_star_context_points_to_binding_specs_and_ledgers() -> Result<(), String> {
    let workspace = workspace_root()?;
    let required = [
        "specs/jolt-prover-frontier-harness.md",
        "specs/jolt-prover-cpu-backend-port.md",
        "specs/jolt-core-prover-optimization-inventory.md",
        "crates/jolt-prover-harness/src/optimization.rs",
    ];

    for file in [
        workspace.join("AGENTS.md"),
        workspace.join("CLAUDE.md"),
        workspace.join("crates/jolt-prover-harness/README.md"),
        workspace.join("crates/jolt-backends/README.md"),
    ] {
        assert_file_contains_all(&file, &required)?;
    }

    assert_file_contains_all(
        &workspace.join("specs/jolt-prover-frontier-harness.md"),
        &[
            "backend kernel ledger",
            "`CorePerformanceParity` is never optional",
            "Correctness-only implementations are not",
            "before wiring the algorithm through `jolt-prover`",
            "isolated backend microbenchmark result",
            "validate_frontier_replacement_ready",
            "validate_parity_certified_kernel_evidence_files",
            "validate_global_cpu_backend_inventory_coverage",
            "KernelBenchmarkEvidence::write_canonical_json",
            "Iteration Ladder",
            "<= 15% regression",
        ],
    )?;
    assert_file_contains_all(
        &workspace.join("specs/jolt-prover-cpu-backend-port.md"),
        &[
            "Microbench first",
            "before spending time debugging it through",
            "KernelBenchmarkEvidence",
            "15% failure threshold",
            "Fast Iteration Order",
            "validate_global_cpu_backend_inventory_coverage",
        ],
    )
}

fn workspace_root() -> Result<PathBuf, String> {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .ancestors()
        .nth(2)
        .map(Path::to_path_buf)
        .ok_or_else(|| "failed to locate workspace root".to_owned())
}

fn assert_file_contains_all(path: &Path, patterns: &[&str]) -> Result<(), String> {
    let content = fs::read_to_string(path).map_err(|error| error.to_string())?;
    let missing = patterns
        .iter()
        .copied()
        .filter(|pattern| !content.contains(pattern))
        .collect::<Vec<_>>();
    if missing.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "{} missing required north-star reference(s): {}",
            path.display(),
            missing.join(", ")
        ))
    }
}

fn assert_no_patterns(roots: &[PathBuf], patterns: &[&str]) -> Result<(), String> {
    let mut files = Vec::new();
    for root in roots {
        collect_rust_files(root, &mut files)?;
    }

    let mut offenders = Vec::new();
    for file in files {
        let content = fs::read_to_string(&file).map_err(|error| error.to_string())?;
        for (line_index, line) in content.lines().enumerate() {
            if patterns.iter().any(|pattern| line.contains(pattern)) {
                offenders.push(format!("{}:{}", file.display(), line_index + 1));
            }
        }
    }

    if offenders.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "forbidden architecture pattern(s): {}",
            offenders.join(", ")
        ))
    }
}

fn assert_no_patterns_in_markdown(roots: &[PathBuf], patterns: &[&str]) -> Result<(), String> {
    let mut files = Vec::new();
    for root in roots {
        collect_markdown_files(root, &mut files)?;
    }

    let mut offenders = Vec::new();
    for file in files {
        let content = fs::read_to_string(&file).map_err(|error| error.to_string())?;
        for (line_index, line) in content.lines().enumerate() {
            if patterns.iter().any(|pattern| line.contains(pattern)) {
                offenders.push(format!("{}:{}", file.display(), line_index + 1));
            }
        }
    }

    if offenders.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "retired prover architecture terminology: {}",
            offenders.join(", ")
        ))
    }
}

fn collect_rust_files(root: &Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in fs::read_dir(root).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let path = entry.path();
        if path.is_dir() {
            collect_rust_files(&path, files)?;
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("rs")
            && path.file_name().is_none_or(|name| name != "tests.rs")
        {
            files.push(path);
        }
    }
    Ok(())
}

fn collect_markdown_files(root: &Path, files: &mut Vec<PathBuf>) -> Result<(), String> {
    if root.is_file() {
        if root.extension().and_then(|ext| ext.to_str()) == Some("md") {
            files.push(root.to_path_buf());
        }
        return Ok(());
    }

    for entry in fs::read_dir(root).map_err(|error| error.to_string())? {
        let entry = entry.map_err(|error| error.to_string())?;
        let path = entry.path();
        if path.is_dir() {
            collect_markdown_files(&path, files)?;
        } else if path.extension().and_then(|ext| ext.to_str()) == Some("md") {
            files.push(path);
        }
    }
    Ok(())
}
