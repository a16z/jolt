//! Repo-hygiene boundary check for the field-inline seams (dependency-free;
//! reads the source tree at test time, so it enforces the boundary in the
//! default FR-off suite on every PR).
//!
//! The architectural rule: the field-inline protocol is a completely separate
//! codepath, and every FR divergence in this crate lives either in a dedicated
//! `field_inline` seam module or at an explicitly whitelisted interaction
//! point (a flagged one-line seam call, a carrier field, a module
//! registration, or a relation's appendage shell). Each whitelist entry below
//! says why its file legitimately carries `feature = "field-inline"` text, and
//! caps how much of it the file may carry — moving FR logic back inline blows
//! the cap and fails here.

#![expect(clippy::expect_used, reason = "test-only source-tree walking")]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

const GATE: &str = "feature = \"field-inline\"";

/// The whitelisted file set, each with WHY it is a legitimate seam and the
/// maximum number of `feature = "field-inline"` occurrences it may carry
/// (production and test text alike — the cap is the ratchet).
const WHITELIST: &[(&str, usize, &str)] = &[
    // Compile-time protocol selection: the FR config constant pair.
    ("config.rs", 2, "compile-time protocol config"),
    // The FR commitment payload is proof shape: carrier struct/field,
    // constructor default, the homomorphic commitment slot's attach builder,
    // and the packed limb-group commitment/claims slots with their carry-over
    // lines.
    ("proof.rs", 9, "FR commitment payload carriers"),
    // The payload presence check, the FR commitment absorb seams, and the
    // mode-specific test fixtures. (The fail-closed
    // require_field_inline_slices gate lived here until the FR prover
    // fixtures landed.)
    ("verifier.rs", 15, "commitment absorb seams + test fixtures"),
    // Module registration of the shared FR bytecode side-table seam, plus
    // the packed schedule's FR presence-marker field and its constructor.
    ("stages/mod.rs", 4, "seam registration + FR schedule marker"),
    // Per-stage seam-module and FR-twin module registrations.
    ("stages/stage1/mod.rs", 1, "seam module registration"),
    ("stages/stage2/mod.rs", 2, "seam module registrations"),
    ("stages/stage4/mod.rs", 3, "seam/twin module registrations"),
    ("stages/stage5/mod.rs", 3, "seam/twin module registrations"),
    ("stages/stage6a/mod.rs", 1, "seam module registration"),
    ("stages/stage6b/mod.rs", 2, "seam/twin module registrations"),
    ("stages/stage8/mod.rs", 2, "seam module registrations"),
    // Stage verify.rs files: exactly the flagged one-line divergences calling
    // their stage's field_inline seam (struct-literal fields keep the flag on
    // the field line — the uniform impossible-case shape).
    ("stages/stage1/verify.rs", 4, "flagged seam calls"),
    ("stages/stage2/verify.rs", 7, "flagged seam calls"),
    ("stages/stage4/verify.rs", 3, "flagged seam calls"),
    ("stages/stage5/verify.rs", 3, "flagged seam calls"),
    ("stages/stage6a/verify.rs", 2, "flagged seam calls"),
    (
        "stages/stage6b/verify.rs",
        11,
        "flagged seam calls + test fixtures",
    ),
    (
        "stages/stage8/verify.rs",
        6,
        "flagged seam calls + FR plan test gate",
    ),
    // The packed batch assembly: the FR proof-slot parameters and the flagged
    // block calling the stage-8 packed FR seam.
    (
        "stages/stage8/packed.rs",
        4,
        "FR slot params + flagged seam call",
    ),
    // The FR recomposition-mismatch reject is a typed error variant.
    ("error.rs", 1, "FR typed error variant"),
    // outputs.rs carrier fields are proof shape: FR batch-member slots,
    // output-claim carrier fields, point accessors, re-exports, and the
    // mode-specific test fixtures that construct them.
    ("stages/stage1/outputs.rs", 4, "FR carrier fields"),
    (
        "stages/stage2/outputs.rs",
        18,
        "FR carrier fields + test fixtures",
    ),
    (
        "stages/stage4/outputs.rs",
        15,
        "FR carrier fields + test fixtures",
    ),
    (
        "stages/stage5/outputs.rs",
        9,
        "FR carrier fields + test fixtures",
    ),
    ("stages/stage6b/outputs.rs", 6, "FR carrier fields"),
    // Relation files that carry an FR appendage: the OnceLock carrier field,
    // its setter, and the composed input/expected-output override shells
    // (trait items cannot move out of the impl; their FR math lives in the
    // stage's field_inline seam or jolt-claims composed-lane helpers).
    (
        "stages/stage1/outer_remainder.rs",
        8,
        "FR appendage carrier + accessor + override",
    ),
    (
        "stages/stage2/product_uniskip.rs",
        5,
        "FR appendage carrier + override",
    ),
    (
        "stages/stage2/product_remainder.rs",
        6,
        "FR appendage carrier + accessor + override",
    ),
    (
        "stages/stage6a/bytecode_read_raf.rs",
        9,
        "FR appendage carriers (input values + kernel geometry) + override shell",
    ),
    (
        "stages/stage6b/bytecode_read_raf.rs",
        10,
        "FR fold constructor leg + composed publics + kernel fold accessor + \
         ordinary-fold operand masking",
    ),
    // The stage-6b batch build: FR draw slot, build-parts leg, and the flagged
    // seam calls assembling the FR members (struct fields cannot move).
    (
        "stages/stage6b/batch.rs",
        15,
        "FR batch legs + flagged seam calls",
    ),
    // The BlindFold lowering: the composite VerifierPublicId FR arms (type
    // shape), the flagged seam calls into blindfold/field_inline.rs, and the
    // FR value-parity test gates.
    (
        "stages/zk/blindfold/mod.rs",
        16,
        "FR id arms + flagged seam calls + ordinary-fold operand masking",
    ),
    (
        "stages/zk/blindfold/stage1.rs",
        6,
        "flagged seam calls + tests",
    ),
    (
        "stages/zk/blindfold/stage2.rs",
        22,
        "flagged seam calls + tests",
    ),
    (
        "stages/zk/blindfold/stage4.rs",
        8,
        "flagged seam calls + tests",
    ),
    (
        "stages/zk/blindfold/stage5.rs",
        8,
        "flagged seam calls + tests",
    ),
    (
        "stages/zk/blindfold/stage6a.rs",
        2,
        "flagged seam call + test gate",
    ),
    (
        "stages/zk/blindfold/stage6b.rs",
        7,
        "flagged seam calls + tests",
    ),
];

/// Files that ARE the field-inline seams: whole modules cfg-gated at their
/// registration, so their content is FR by definition and carries no gate
/// text of its own (a gate inside one would be redundant but harmless).
const SEAM_MODULES: &[&str] = &[
    "stages/field_inline_bytecode.rs",
    "stages/stage1/field_inline.rs",
    "stages/stage2/field_inline.rs",
    "stages/stage4/field_inline.rs",
    "stages/stage5/field_inline.rs",
    "stages/stage6a/field_inline.rs",
    "stages/stage6b/field_inline.rs",
    "stages/stage8/field_inline.rs",
    "stages/stage8/field_inline_packed.rs",
    "stages/zk/blindfold/field_inline.rs",
    // The FR ConcreteSumcheck twins: separate types per the protocol ruling,
    // cfg-gated at their module registrations.
    "stages/stage2/field_registers_claim_reduction.rs",
    "stages/stage4/field_registers_read_write_checking.rs",
    "stages/stage5/field_registers_val_evaluation.rs",
    "stages/stage6b/field_registers_inc_claim_reduction.rs",
];

fn rust_sources(dir: &Path) -> Vec<PathBuf> {
    let mut files = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in fs::read_dir(&dir).expect("source directory is readable") {
            let path = entry.expect("directory entry is readable").path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|extension| extension == "rs") {
                files.push(path);
            }
        }
    }
    files.sort();
    files
}

#[test]
fn field_inline_gates_stay_in_the_whitelisted_seams() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let caps: BTreeMap<&str, usize> = WHITELIST
        .iter()
        .map(|(file, cap, _why)| (*file, *cap))
        .collect();

    let mut violations = Vec::new();
    for file in rust_sources(&src) {
        let relative = file
            .strip_prefix(&src)
            .expect("sources live under src")
            .to_string_lossy()
            .replace('\\', "/");
        let source = fs::read_to_string(&file).expect("source file is readable");
        let count = source.matches(GATE).count();
        if count == 0 {
            continue;
        }
        if SEAM_MODULES.contains(&relative.as_str()) {
            continue;
        }
        match caps.get(relative.as_str()) {
            None => violations.push(format!(
                "{relative}: {count} `{GATE}` occurrence(s) in a non-whitelisted file — move \
                 the FR logic into that stage's field_inline seam module (or whitelist the new \
                 seam here with a why-comment)"
            )),
            Some(cap) if count > *cap => violations.push(format!(
                "{relative}: {count} `{GATE}` occurrences exceed the whitelisted cap of {cap} — \
                 new FR divergences belong in the stage's field_inline seam module"
            )),
            Some(_) => {}
        }
    }
    assert!(
        violations.is_empty(),
        "field-inline seam boundary violated:\n{}",
        violations.join("\n")
    );
}

/// Whitelisted files must keep existing (a rename silently drops its cap), and
/// every seam module must stay registered where the convention says it lives.
#[test]
fn whitelisted_seam_files_exist() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut missing = Vec::new();
    for (file, _cap, _why) in WHITELIST {
        if !src.join(file).is_file() {
            missing.push(*file);
        }
    }
    for file in SEAM_MODULES {
        if !src.join(file).is_file() {
            missing.push(*file);
        }
    }
    assert!(
        missing.is_empty(),
        "whitelisted seam files missing (update the whitelist alongside renames):\n{}",
        missing.join("\n")
    );
}
