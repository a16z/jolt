//! Repo-hygiene boundary checks for the protocol split (dependency-free; reads
//! the source tree at test time).
//!
//! The architectural rule: `protocols/jolt` and `protocols/field_inline` are
//! separate protocol families. They share algebra through the id-free framework
//! modules and compose only in `jolt-verifier`: neither protocol module may import
//! the other. Both families compile unconditionally. The common Jolt flag carriers
//! and their geometry mirror the feature-gated ISA flags in `jolt-riscv`; this does
//! not introduce field-register protocol ids into the Jolt protocol.

#![expect(clippy::expect_used, reason = "test-only source-tree walking")]

use std::fs;
use std::path::{Path, PathBuf};

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

/// The file's source text with line comments, block comments, and string
/// literals blanked, so the boundary greps below match code only (doc comments
/// may legitimately mention the sibling family by name).
fn code_text(path: &Path) -> String {
    let source = fs::read_to_string(path).expect("source file is readable");
    let mut out = String::with_capacity(source.len());
    let mut chars = source.chars().peekable();
    let mut state = State::Code;
    #[derive(PartialEq)]
    enum State {
        Code,
        LineComment,
        BlockComment(usize),
        Str,
    }
    while let Some(c) = chars.next() {
        match state {
            State::Code => match c {
                '/' if chars.peek() == Some(&'/') => {
                    state = State::LineComment;
                }
                '/' if chars.peek() == Some(&'*') => {
                    let _ = chars.next();
                    state = State::BlockComment(1);
                }
                '"' => {
                    out.push(' ');
                    state = State::Str;
                }
                _ => out.push(c),
            },
            State::LineComment => {
                if c == '\n' {
                    out.push('\n');
                    state = State::Code;
                }
            }
            State::BlockComment(depth) => match c {
                '*' if chars.peek() == Some(&'/') => {
                    let _ = chars.next();
                    state = if depth == 1 {
                        State::Code
                    } else {
                        State::BlockComment(depth - 1)
                    };
                }
                '/' if chars.peek() == Some(&'*') => {
                    let _ = chars.next();
                    state = State::BlockComment(depth + 1);
                }
                '\n' => out.push('\n'),
                _ => {}
            },
            State::Str => match c {
                '\\' => {
                    let _ = chars.next();
                }
                '"' => state = State::Code,
                _ => {}
            },
        }
    }
    out
}

fn src_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src")
}

/// Neither protocol module imports (or otherwise names, outside comments and
/// strings) the other: the only sanctioned sharing is the id-free
/// `twist` algebra, and composition happens only in `jolt-verifier`.
#[test]
fn protocol_modules_are_import_disjoint() {
    let mut violations = Vec::new();
    for file in rust_sources(&src_dir().join("protocols").join("jolt")) {
        let code = code_text(&file);
        if code.contains("field_inline") || code.contains("FieldInline") {
            violations.push(format!(
                "{} references the field_inline protocol module",
                file.display()
            ));
        }
    }
    for file in rust_sources(&src_dir().join("protocols").join("field_inline")) {
        let code = code_text(&file);
        if code.contains("protocols::jolt") || code.contains("protocols :: jolt") {
            violations.push(format!(
                "{} imports the jolt protocol module",
                file.display()
            ));
        }
    }
    assert!(
        violations.is_empty(),
        "protocol modules must stay import-disjoint (share algebra via \
         twist, compose in jolt-verifier):\n{}",
        violations.join("\n")
    );
}

/// The shared Twist-identity module carries no protocol ids: it must not reference
/// either protocol module.
#[test]
fn twist_reference_no_protocol_module() {
    let mut violations = Vec::new();
    for file in rust_sources(&src_dir().join("twist")) {
        let code = code_text(&file);
        for needle in ["protocols::", "FieldInline", "field_inline", "Jolt"] {
            if code.contains(needle) {
                violations.push(format!("{} references `{needle}`", file.display()));
            }
        }
    }
    assert!(
        violations.is_empty(),
        "twist must stay protocol-id-free:\n{}",
        violations.join("\n")
    );
}

/// The shared balanced-digit algebra is id-free like `twist`: both packed
/// protocol families ride it, so it must not reference either protocol
/// module.
#[test]
fn lattice_algebra_references_no_protocol_module() {
    let mut violations = Vec::new();
    let lattice = src_dir().join("lattice.rs");
    let code = code_text(&lattice);
    for needle in ["protocols::", "FieldInline", "field_inline", "Jolt"] {
        if code.contains(needle) {
            violations.push(format!("{} references `{needle}`", lattice.display()));
        }
    }
    assert!(
        violations.is_empty(),
        "the shared lattice algebra must stay protocol-id-free:\n{}",
        violations.join("\n")
    );
}

/// Field-register relations and shared algebra remain unconditional. The three
/// ordinary Jolt flag carriers mirror `jolt-riscv::CircuitFlags`, whose field
/// instruction variants exist only with the `field-inline` ISA feature.
#[test]
fn field_inline_feature_gates_are_confined_to_common_flag_carriers() {
    let source_dir = src_dir();
    let flag_carriers = [
        "protocols/jolt/geometry/spartan.rs",
        "protocols/jolt/relations/spartan/outer_remainder.rs",
        "protocols/jolt/relations/bytecode/read_raf_address_phase.rs",
    ]
    .map(|path| source_dir.join(path));
    // The feature name is a string literal, which `code_text` would blank.
    let mut violations = Vec::new();
    for file in rust_sources(&source_dir) {
        let source = fs::read_to_string(&file).expect("source file is readable");
        if source.contains("feature = \"field-inline\"") && !flag_carriers.contains(&file) {
            violations.push(file.display().to_string());
        }
    }
    assert!(
        violations.is_empty(),
        "field-inline feature gates belong only in the common ISA flag carriers:\n{}",
        violations.join("\n")
    );
}
