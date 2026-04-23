//! Dump a Mermaid + DOT visualization of an AST that includes transcript ops.
//!
//! Run with:
//!   cargo run -q -p jolt-verifier-backend --example transcript_ast_dump
//!
//! Emits two artifacts under `target/transcript_ast/`:
//!   - `transcript_ast.mmd` — paste into a Mermaid renderer (or VS Code preview)
//!   - `transcript_ast.dot` — render with `dot -Tpng transcript_ast.dot > graph.png`
//!
//! The graph is intentionally tiny (one transcript init, two absorbs, two
//! squeezes, plus a couple of arithmetic ops on the squeezed challenges) so
//! the result fits on a single screen and shows every node kind that the
//! verifier produces in practice.

#![expect(
    clippy::print_stderr,
    clippy::print_stdout,
    reason = "this is a developer-facing example binary that prints diagnostics"
)]

use std::fs;
use std::path::PathBuf;

use jolt_field::{Field, Fr};
use jolt_openings::mock::MockCommitmentScheme;
use jolt_transcript::Transcript;
use jolt_verifier_backend::{to_dot, to_mermaid, FieldBackend, Tracing};

/// AST-side `Tracing` is generic over the [`CommitmentScheme`]. This example
/// only uses field-side ops, so we instantiate against
/// `MockCommitmentScheme<Fr>` (whose `Field = Fr`) to keep the snippet
/// self-contained without pulling in a real PCS.
type Mock = MockCommitmentScheme<Fr>;

fn main() {
    let mut tracer = Tracing::<Mock>::new();
    let mut transcript = tracer.new_transcript(b"jolt_demo");

    // Absorb a labeled domain (mimics what the verifier does before squeezing
    // the first batched-sumcheck challenge).
    transcript.append_bytes(b"opening_claim");
    let (alpha_f, alpha_w) = tracer.squeeze(&mut transcript, "alpha");

    // Absorb the squeezed challenge as proof data and squeeze a second one
    // (this happens inside batched sumcheck verification).
    transcript.append_bytes(&alpha_f.to_bytes());
    let (_beta_f, beta_w) = tracer.squeeze(&mut transcript, "beta");

    // A trivial arithmetic sub-expression that uses both challenges, so the
    // diagram shows the bridge from transcript squeeze -> field arithmetic.
    let two = tracer.const_i128(2);
    let alpha_doubled = tracer.mul(&alpha_w, &two);
    let combined = tracer.add(&alpha_doubled, &beta_w);
    let _square = tracer.square(&combined);

    let graph = tracer.snapshot();
    let dot = to_dot(&graph);
    let mer = to_mermaid(&graph);

    let dir: PathBuf = ["target", "transcript_ast"].iter().collect();
    fs::create_dir_all(&dir).expect("create target/transcript_ast");
    fs::write(dir.join("transcript_ast.dot"), &dot).expect("write dot");
    fs::write(dir.join("transcript_ast.mmd"), &mer).expect("write mermaid");

    eprintln!(
        "Wrote {} nodes / {} assertions to {}",
        graph.node_count(),
        graph.assertion_count(),
        dir.display()
    );
    println!("=== MERMAID ===\n{mer}");
}
