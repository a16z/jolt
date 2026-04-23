//! Pretty-printers for an [`AstGraph`].
//!
//! [`to_dot`] emits a Graphviz DOT diagram suitable for `dot -Tpng`. Nodes
//! are colored by op kind and wrap nodes carry their [`ScalarOrigin`] +
//! label. Assertions are rendered as dashed bidirectional edges.
//!
//! [`to_mermaid`] emits a Mermaid graph that GitHub-flavored markdown
//! renders inline.
//!
//! These are intentionally text-only utilities so they can be used in
//! tests, examples, and design notes without pulling in a renderer.
//!
//! Both renderers are generic over the [`CommitmentScheme`] so the AST
//! type they walk stays type-honest. Commitment-shaped variants
//! ([`AstOp::CommitmentWrap`], [`AstOp::TranscriptAbsorbCommitment`],
//! [`AstOp::OpeningCheck`]) print metadata about the inlined commitment
//! / proof rather than the raw payload.

use std::fmt::Write;

use jolt_openings::CommitmentScheme;
use jolt_transcript::AppendToTranscript;

use jolt_openings::{CommitmentOrigin, ScalarOrigin};
use crate::tracing::{AstAssertion, AstGraph, AstOp};

/// Render `graph` as a Graphviz DOT digraph.
///
/// Layout is top-down with one node per [`AstOp`]. Op-kind colour mapping
/// is meant to make a glance enough to distinguish proof inputs from
/// transcript challenges from intermediate arithmetic. Assertions appear
/// as dashed red edges between the two participating nodes.
pub fn to_dot<PCS: CommitmentScheme>(graph: &AstGraph<PCS>) -> String
where
    PCS::Output: AppendToTranscript,
{
    let mut out = String::new();
    out.push_str("digraph AstGraph {\n");
    out.push_str("  rankdir=TB;\n");
    out.push_str(
        "  node [shape=box, style=\"filled,rounded\", fontname=\"Helvetica\", fontsize=10];\n",
    );
    out.push_str("  edge [arrowsize=0.6, color=\"#4d4d4d\"];\n");

    for (idx, op) in graph.nodes.iter().enumerate() {
        let (label, attrs) = describe_node(op);
        let label = escape_dot(&label);
        writeln!(out, "  n{idx} [label=\"#{idx}\\n{label}\", {attrs}];").unwrap();
    }

    for (idx, op) in graph.nodes.iter().enumerate() {
        for (operand_idx, edge_label) in operands(op) {
            if let Some(lbl) = edge_label {
                writeln!(
                    out,
                    "  n{operand_idx} -> n{idx} [label=\"{lbl}\", fontsize=8];"
                )
                .unwrap();
            } else {
                writeln!(out, "  n{operand_idx} -> n{idx};").unwrap();
            }
        }
    }

    for (a_idx, assertion) in graph.assertions.iter().enumerate() {
        let ctx = escape_dot(assertion.ctx());
        match assertion {
            AstAssertion::Equality { lhs, rhs, .. } => {
                writeln!(
                    out,
                    "  a{a_idx} [label=\"assert\\n{ctx}\", shape=diamond, fillcolor=\"#fff5f5\", color=\"#c92a2a\", fontcolor=\"#c92a2a\"];",
                )
                .unwrap();
                let lhs = lhs.0;
                writeln!(
                    out,
                    "  n{lhs} -> a{a_idx} [color=\"#c92a2a\", style=dashed, arrowhead=none];",
                )
                .unwrap();
                let rhs = rhs.0;
                writeln!(
                    out,
                    "  a{a_idx} -> n{rhs} [color=\"#c92a2a\", style=dashed, arrowhead=none];",
                )
                .unwrap();
            }
            AstAssertion::OpeningHolds { check, .. } => {
                writeln!(
                    out,
                    "  a{a_idx} [label=\"opening holds\\n{ctx}\", shape=diamond, fillcolor=\"#f3f0ff\", color=\"#5f3dc4\", fontcolor=\"#5f3dc4\"];",
                )
                .unwrap();
                let check = check.0;
                writeln!(
                    out,
                    "  n{check} -> a{a_idx} [color=\"#5f3dc4\", style=dashed, arrowhead=none];",
                )
                .unwrap();
            }
        }
    }

    out.push_str("}\n");
    out
}

/// Render `graph` as a Mermaid `graph TD` block (no fences).
///
/// Suitable for inlining in markdown design docs. Uses Mermaid `classDef`
/// styling so node colour matches [`to_dot`].
pub fn to_mermaid<PCS: CommitmentScheme>(graph: &AstGraph<PCS>) -> String
where
    PCS::Output: AppendToTranscript,
{
    let mut out = String::new();
    out.push_str("graph TD\n");
    out.push_str("  classDef wrapPub fill:#d3f9d8,stroke:#2f9e44,color:#1b4332;\n");
    out.push_str("  classDef wrapProof fill:#dbe4ff,stroke:#3b5bdb,color:#1c3050;\n");
    out.push_str("  classDef wrapChall fill:#ffe8cc,stroke:#e8590c,color:#5c2a02;\n");
    out.push_str("  classDef constant fill:#f1f3f5,stroke:#868e96,color:#212529;\n");
    out.push_str("  classDef arith fill:#fff3bf,stroke:#f08c00,color:#5c3700;\n");
    out.push_str("  classDef multi fill:#ffd8a8,stroke:#e8590c,color:#5c2a02;\n");
    out.push_str("  classDef inv fill:#ffe3e3,stroke:#c92a2a,color:#5c0000;\n");
    out.push_str("  classDef assertion fill:#fff5f5,stroke:#c92a2a,color:#c92a2a;\n");
    out.push_str("  classDef tInit fill:#e7f5ff,stroke:#1971c2,color:#0b3a5c;\n");
    out.push_str("  classDef tAbsorb fill:#d0ebff,stroke:#1971c2,color:#0b3a5c;\n");
    out.push_str("  classDef tSqueezeState fill:#a5d8ff,stroke:#1864ab,color:#0b3a5c;\n");
    out.push_str("  classDef tSqueezeValue fill:#74c0fc,stroke:#1864ab,color:#0b3a5c;\n");
    out.push_str("  classDef commitPub fill:#d0bfff,stroke:#5f3dc4,color:#3d1c7a;\n");
    out.push_str("  classDef commitProof fill:#b197fc,stroke:#5f3dc4,color:#3d1c7a;\n");
    out.push_str("  classDef commitAbsorb fill:#e5dbff,stroke:#5f3dc4,color:#3d1c7a;\n");
    out.push_str("  classDef openingCheck fill:#f3f0ff,stroke:#5f3dc4,color:#5f3dc4;\n");
    out.push_str("  classDef openingHolds fill:#f3f0ff,stroke:#5f3dc4,color:#5f3dc4;\n");

    for (idx, op) in graph.nodes.iter().enumerate() {
        let (label, class) = mermaid_node(op);
        let label = escape_mermaid(&label);
        writeln!(out, "  n{idx}[\"#{idx} {label}\"]:::{class}").unwrap();
    }

    for (idx, op) in graph.nodes.iter().enumerate() {
        for (operand_idx, edge_label) in operands(op) {
            if let Some(lbl) = edge_label {
                writeln!(out, "  n{operand_idx} -- {lbl} --> n{idx}").unwrap();
            } else {
                writeln!(out, "  n{operand_idx} --> n{idx}").unwrap();
            }
        }
    }

    for (a_idx, assertion) in graph.assertions.iter().enumerate() {
        let ctx = escape_mermaid(assertion.ctx());
        match assertion {
            AstAssertion::Equality { lhs, rhs, .. } => {
                writeln!(out, "  a{a_idx}{{{{assert: {ctx}}}}}:::assertion").unwrap();
                let lhs = lhs.0;
                writeln!(out, "  n{lhs} -.-> a{a_idx}").unwrap();
                let rhs = rhs.0;
                writeln!(out, "  a{a_idx} -.-> n{rhs}").unwrap();
            }
            AstAssertion::OpeningHolds { check, .. } => {
                writeln!(out, "  a{a_idx}{{{{opening holds: {ctx}}}}}:::openingHolds").unwrap();
                let check = check.0;
                writeln!(out, "  n{check} -.-> a{a_idx}").unwrap();
            }
        }
    }

    out
}

fn describe_node<PCS: CommitmentScheme>(op: &AstOp<PCS>) -> (String, &'static str)
where
    PCS::Output: AppendToTranscript,
{
    match op {
        AstOp::Wrap { origin, label } => {
            let kind = match origin {
                ScalarOrigin::Public => "public",
                ScalarOrigin::Proof => "proof",
                ScalarOrigin::Challenge => "challenge",
            };
            let attrs = match origin {
                ScalarOrigin::Public => {
                    "fillcolor=\"#d3f9d8\", color=\"#2f9e44\", fontcolor=\"#1b4332\""
                }
                ScalarOrigin::Proof => {
                    "fillcolor=\"#dbe4ff\", color=\"#3b5bdb\", fontcolor=\"#1c3050\""
                }
                ScalarOrigin::Challenge => {
                    "fillcolor=\"#ffe8cc\", color=\"#e8590c\", fontcolor=\"#5c2a02\""
                }
            };
            (format!("{kind}: {label}"), attrs)
        }
        AstOp::Constant(v) => (
            format!("const {v}"),
            "fillcolor=\"#f1f3f5\", color=\"#868e96\", fontcolor=\"#212529\"",
        ),
        AstOp::Neg(_) => (
            "neg".to_owned(),
            "fillcolor=\"#fff3bf\", color=\"#f08c00\", fontcolor=\"#5c3700\"",
        ),
        AstOp::Add(..) => (
            "+".to_owned(),
            "fillcolor=\"#fff3bf\", color=\"#f08c00\", fontcolor=\"#5c3700\"",
        ),
        AstOp::Sub(..) => (
            "-".to_owned(),
            "fillcolor=\"#fff3bf\", color=\"#f08c00\", fontcolor=\"#5c3700\"",
        ),
        AstOp::Mul(..) => (
            "*".to_owned(),
            "fillcolor=\"#ffd8a8\", color=\"#e8590c\", fontcolor=\"#5c2a02\"",
        ),
        AstOp::Square(_) => (
            "^2".to_owned(),
            "fillcolor=\"#ffd8a8\", color=\"#e8590c\", fontcolor=\"#5c2a02\"",
        ),
        AstOp::Inverse { ctx, .. } => (
            format!("inv ({ctx})"),
            "fillcolor=\"#ffe3e3\", color=\"#c92a2a\", fontcolor=\"#5c0000\"",
        ),
        AstOp::TranscriptInit { label } => (
            format!("transcript init({})", display_label(label)),
            "fillcolor=\"#e7f5ff\", color=\"#1971c2\", fontcolor=\"#0b3a5c\"",
        ),
        AstOp::TranscriptAbsorbBytes { bytes, .. } => (
            format!("absorb {} B {}", bytes.len(), display_bytes_summary(bytes)),
            "fillcolor=\"#d0ebff\", color=\"#1971c2\", fontcolor=\"#0b3a5c\"",
        ),
        AstOp::TranscriptChallengeState { .. } => (
            "transcript squeeze (state)".to_owned(),
            "fillcolor=\"#a5d8ff\", color=\"#1864ab\", fontcolor=\"#0b3a5c\"",
        ),
        AstOp::TranscriptChallengeValue { .. } => (
            "transcript squeeze (F)".to_owned(),
            "fillcolor=\"#74c0fc\", color=\"#1864ab\", fontcolor=\"#0b3a5c\"",
        ),
        AstOp::CommitmentWrap { origin, label, .. } => {
            let kind = match origin {
                CommitmentOrigin::Public => "vk-commit",
                CommitmentOrigin::Proof => "proof-commit",
            };
            let attrs = match origin {
                CommitmentOrigin::Public => {
                    "fillcolor=\"#d0bfff\", color=\"#5f3dc4\", fontcolor=\"#3d1c7a\""
                }
                CommitmentOrigin::Proof => {
                    "fillcolor=\"#b197fc\", color=\"#5f3dc4\", fontcolor=\"#3d1c7a\""
                }
            };
            (format!("{kind}: {label}"), attrs)
        }
        AstOp::TranscriptAbsorbCommitment { label, .. } => (
            format!("absorb commitment({})", display_label(label)),
            "fillcolor=\"#e5dbff\", color=\"#5f3dc4\", fontcolor=\"#3d1c7a\"",
        ),
        AstOp::OpeningCheck { point, .. } => (
            format!("opening check (|z|={})", point.len()),
            "fillcolor=\"#f3f0ff\", color=\"#5f3dc4\", fontcolor=\"#5f3dc4\"",
        ),
    }
}

fn mermaid_node<PCS: CommitmentScheme>(op: &AstOp<PCS>) -> (String, &'static str)
where
    PCS::Output: AppendToTranscript,
{
    match op {
        AstOp::Wrap { origin, label } => match origin {
            ScalarOrigin::Public => (format!("public: {label}"), "wrapPub"),
            ScalarOrigin::Proof => (format!("proof: {label}"), "wrapProof"),
            ScalarOrigin::Challenge => (format!("chall: {label}"), "wrapChall"),
        },
        AstOp::Constant(v) => (format!("const {v}"), "constant"),
        AstOp::Neg(_) => ("neg".to_owned(), "arith"),
        AstOp::Add(..) => ("+".to_owned(), "arith"),
        AstOp::Sub(..) => ("-".to_owned(), "arith"),
        AstOp::Mul(..) => ("*".to_owned(), "multi"),
        AstOp::Square(_) => ("^2".to_owned(), "multi"),
        AstOp::Inverse { ctx, .. } => (format!("inv ({ctx})"), "inv"),
        AstOp::TranscriptInit { label } => (
            format!("transcript init({})", display_label(label)),
            "tInit",
        ),
        AstOp::TranscriptAbsorbBytes { bytes, .. } => (
            format!("absorb {}B {}", bytes.len(), display_bytes_summary(bytes)),
            "tAbsorb",
        ),
        AstOp::TranscriptChallengeState { .. } => {
            ("transcript squeeze (state)".to_owned(), "tSqueezeState")
        }
        AstOp::TranscriptChallengeValue { .. } => {
            ("transcript squeeze (F)".to_owned(), "tSqueezeValue")
        }
        AstOp::CommitmentWrap { origin, label, .. } => match origin {
            CommitmentOrigin::Public => (format!("vk-commit: {label}"), "commitPub"),
            CommitmentOrigin::Proof => (format!("proof-commit: {label}"), "commitProof"),
        },
        AstOp::TranscriptAbsorbCommitment { label, .. } => (
            format!("absorb commitment({})", display_label(label)),
            "commitAbsorb",
        ),
        AstOp::OpeningCheck { point, .. } => (
            format!("opening check (|z|={})", point.len()),
            "openingCheck",
        ),
    }
}

fn operands<PCS: CommitmentScheme>(op: &AstOp<PCS>) -> Vec<(u32, Option<&'static str>)>
where
    PCS::Output: AppendToTranscript,
{
    match op {
        AstOp::Wrap { .. }
        | AstOp::Constant(_)
        | AstOp::TranscriptInit { .. }
        | AstOp::CommitmentWrap { .. } => Vec::new(),
        AstOp::Neg(a) | AstOp::Square(a) => vec![(a.0, None)],
        AstOp::Inverse { operand, .. } => vec![(operand.0, None)],
        AstOp::Add(a, b) => vec![(a.0, Some("a")), (b.0, Some("b"))],
        AstOp::Sub(a, b) => vec![(a.0, Some("a")), (b.0, Some("b"))],
        AstOp::Mul(a, b) => vec![(a.0, Some("a")), (b.0, Some("b"))],
        AstOp::TranscriptAbsorbBytes { prev_state, .. }
        | AstOp::TranscriptChallengeState { prev_state } => vec![(prev_state.0, Some("state"))],
        AstOp::TranscriptChallengeValue { state } => vec![(state.0, Some("squeeze"))],
        AstOp::TranscriptAbsorbCommitment {
            prev_state,
            commitment,
            ..
        } => vec![
            (prev_state.0, Some("state")),
            (commitment.0, Some("commit")),
        ],
        AstOp::OpeningCheck {
            prev_state,
            commitment,
            point,
            claim,
            ..
        } => {
            let mut ops = Vec::with_capacity(3 + point.len());
            ops.push((prev_state.0, Some("state")));
            ops.push((commitment.0, Some("commit")));
            ops.push((claim.0, Some("claim")));
            for (i, p) in point.iter().enumerate() {
                let label: &'static str = match i {
                    0 => "z[0]",
                    1 => "z[1]",
                    2 => "z[2]",
                    3 => "z[3]",
                    _ => "z[..]",
                };
                ops.push((p.0, Some(label)));
            }
            ops
        }
    }
}

fn display_label(label: &[u8]) -> String {
    match std::str::from_utf8(label) {
        Ok(s) if !s.is_empty() => s.to_owned(),
        _ => format!("{label:02x?}"),
    }
}

/// Renders a short ASCII-prefix preview of a transcript byte payload, fenced
/// in single quotes if printable and otherwise dropped. The verifier's
/// labeled-domain encodings (`Label`, `LabelWithCount`) live at the start of
/// each 32-byte buffer, so this preview almost always surfaces them.
fn display_bytes_summary(bytes: &[u8]) -> String {
    let prefix_len = bytes
        .iter()
        .take(24)
        .take_while(|b| b.is_ascii_graphic() || **b == b' ')
        .count();
    if prefix_len >= 3 {
        let printable = std::str::from_utf8(&bytes[..prefix_len]).unwrap_or_default();
        format!("'{printable}'")
    } else {
        String::new()
    }
}

fn escape_dot(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

fn escape_mermaid(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    #![expect(clippy::unwrap_used, reason = "tests")]

    use super::*;
    use jolt_openings::FieldBackend;
    use crate::tracing::Tracing;
    use jolt_field::{Field, Fr};
    use jolt_openings::mock::MockCommitmentScheme;

    type Mock = MockCommitmentScheme<Fr>;

    fn small_graph() -> AstGraph<Mock> {
        let mut t = Tracing::<Mock>::new();
        let a = t.wrap_proof(Fr::from_u64(2), "a");
        let b = t.wrap_challenge(Fr::from_u64(3), "b");
        let c = t.const_i128(7);
        let ab = t.mul(&a, &b);
        let ab_plus_c = t.add(&ab, &c);
        let sq = t.square(&a);
        t.assert_eq(&ab_plus_c, &sq, "demo").unwrap();
        t.snapshot()
    }

    #[test]
    fn dot_contains_every_node_and_assertion() {
        let g = small_graph();
        let dot = to_dot(&g);
        assert!(dot.starts_with("digraph AstGraph"));
        for i in 0..g.node_count() {
            assert!(dot.contains(&format!("n{i}")), "missing n{i} in dot");
        }
        for i in 0..g.assertion_count() {
            assert!(dot.contains(&format!("a{i}")), "missing a{i} in dot");
        }
    }

    #[test]
    fn mermaid_contains_every_node_and_assertion() {
        let g = small_graph();
        let mer = to_mermaid(&g);
        assert!(mer.starts_with("graph TD"));
        for i in 0..g.node_count() {
            assert!(mer.contains(&format!("n{i}[")), "missing n{i} in mermaid");
        }
        for i in 0..g.assertion_count() {
            assert!(mer.contains(&format!("a{i}{{")), "missing a{i} in mermaid");
        }
    }
}
