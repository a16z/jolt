//! Claim-graph analysis, Akita PIOP, plus the Dory/Akita cross-PIOP
//! correspondence (both graphs are constructible in this build). See
//! `specs/claim-graph-analysis.md`.
#![expect(
    clippy::expect_used,
    reason = "the analysis harness fails loudly with full context"
)]

#[path = "claim_graph/mod.rs"]
mod claim_graph;

use std::collections::{BTreeMap, BTreeSet};

use claim_graph::graph::ClaimGraph;
use claim_graph::registry::{all_aliases, all_vertices};
use claim_graph::{assert_graph_snapshot, Piop, ProtocolConfig, VertexRecord};
use jolt_claims::protocols::jolt::JoltRelationId;

fn assert_well_formed(context: &str, config: &ProtocolConfig) {
    let records = all_vertices(config);
    let graph = ClaimGraph::build(Piop::Akita, &records, config, all_aliases());
    let violations = graph.check();
    assert!(
        violations.is_empty(),
        "[{context}] claim-graph violations:\n  {}\n\ngraph:\n{graph}",
        violations.join("\n  ")
    );
}

#[test]
fn akita_clear_graph_is_well_formed() {
    assert_well_formed("akita, clear, small", &ProtocolConfig::small());
}

#[test]
fn akita_clear_graph_is_well_formed_with_advice() {
    assert_well_formed(
        "akita, clear, small + trusted/untrusted advice",
        &ProtocolConfig::small_with_advice(),
    );
}

#[test]
fn akita_clear_graph_is_well_formed_with_committed_program() {
    assert_well_formed(
        "akita, clear, small + committed program",
        &ProtocolConfig::small_committed_program(),
    );
}

/// The Dory side of the documented substitution set under the small
/// (full-program, no-advice) configuration — `specs/lattice-claims.md`,
/// "Verifier and Prover Schedule": "The Dory build instantiates the original
/// Booleanity, increment claim reduction, HammingWeightClaimReduction, and
/// homomorphic final opening." The base bytecode read-RAF phases are the
/// increment claim reduction's counterpart on the Dory side: relation 1 of
/// the spec discharges the four reduced inc claims *inside* the lattice
/// bytecode read-RAF, replacing both the standalone `IncClaimReduction` and
/// the base read-RAF phases.
const DORY_ONLY_VERTICES: &[(&str, JoltRelationId)] = &[
    ("BooleanityCyclePhase", JoltRelationId::Booleanity),
    ("ReadRafAddressPhase", JoltRelationId::BytecodeReadRaf),
    ("ReadRafCyclePhase", JoltRelationId::BytecodeReadRaf),
    ("ClaimReduction", JoltRelationId::IncClaimReduction),
    (
        "ClaimReduction",
        JoltRelationId::HammingWeightClaimReduction,
    ),
];

/// The Akita side of the same substitution set: the lattice booleanity cycle
/// phase (spec relation 2, same `Booleanity` id), the lattice read-RAF phases
/// carrying the fused-inc consumer stages and the `FusedInc` cycle factor
/// (relation 1), and the extended hamming-weight claim reduction (relation 3).
/// The reconstruction relations (5-8) instantiate only with advice or a
/// committed program, so they do not appear under the small configuration.
const AKITA_ONLY_VERTICES: &[(&str, JoltRelationId)] = &[
    ("LatticeBooleanityCyclePhase", JoltRelationId::Booleanity),
    (
        "LatticeReadRafAddressPhase",
        JoltRelationId::BytecodeReadRaf,
    ),
    ("LatticeReadRafCyclePhase", JoltRelationId::BytecodeReadRaf),
    (
        "LatticeHammingWeightClaimReduction",
        JoltRelationId::HammingWeightClaimReduction,
    ),
];

/// Vertices keyed by `(type name, relation id)` — unique per graph under the
/// small configuration (asserted, so a future multi-instance shape cannot
/// silently alias two vertices).
fn vertex_keys(graph: &ClaimGraph) -> BTreeMap<(&'static str, JoltRelationId), &VertexRecord> {
    let mut keys = BTreeMap::new();
    for vertex in &graph.vertices {
        let previous = keys.insert((vertex.name, vertex.relation), vertex);
        assert!(
            previous.is_none(),
            "duplicate vertex key ({}, {:?})",
            vertex.name,
            vertex.relation
        );
    }
    keys
}

#[test]
fn dory_akita_graphs_differ_exactly_by_the_documented_substitution_set() {
    let config = ProtocolConfig::small();
    let records = all_vertices(&config);
    let dory = ClaimGraph::build(Piop::Dory, &records, &config, all_aliases());
    let akita = ClaimGraph::build(Piop::Akita, &records, &config, all_aliases());
    let dory_keys = vertex_keys(&dory);
    let akita_keys = vertex_keys(&akita);

    // Identical shared subgraph: every vertex present in both graphs carries
    // the same shape and the same expression edges.
    for (key, dory_vertex) in &dory_keys {
        let Some(akita_vertex) = akita_keys.get(key) else {
            continue;
        };
        assert_eq!(
            dory_vertex.in_edges, akita_vertex.in_edges,
            "shared vertex {key:?} consumes different ids across PIOPs"
        );
        assert_eq!(
            dory_vertex.out_edges, akita_vertex.out_edges,
            "shared vertex {key:?} produces different ids across PIOPs"
        );
        assert_eq!(
            (dory_vertex.rounds, dory_vertex.degree),
            (akita_vertex.rounds, akita_vertex.degree),
            "shared vertex {key:?} has different rounds/degree across PIOPs"
        );
    }

    // The symmetric difference is exactly the documented substitution set,
    // asserted by name in both directions.
    let dory_only: BTreeSet<(&str, JoltRelationId)> = dory_keys
        .keys()
        .filter(|key| !akita_keys.contains_key(*key))
        .copied()
        .collect();
    assert_eq!(
        dory_only,
        DORY_ONLY_VERTICES.iter().copied().collect(),
        "Dory-only vertices diverge from the lattice-claims.md substitution set"
    );
    let akita_only: BTreeSet<(&str, JoltRelationId)> = akita_keys
        .keys()
        .filter(|key| !dory_keys.contains_key(*key))
        .copied()
        .collect();
    assert_eq!(
        akita_only,
        AKITA_ONLY_VERTICES.iter().copied().collect(),
        "Akita-only vertices diverge from the lattice-claims.md substitution set"
    );
}

/// On-demand terminal rendering:
/// `cargo nextest run -p jolt-claims --features akita dump_claim_graph \
///  --run-ignored all --no-capture`
#[test]
#[ignore = "on-demand graph dump"]
#[expect(clippy::print_stdout, reason = "the dump test exists to print")]
fn dump_claim_graph() {
    let config = ProtocolConfig::small();
    let records = all_vertices(&config);
    println!(
        "{}",
        ClaimGraph::build(Piop::Akita, &records, &config, all_aliases())
    );
}

#[test]
fn akita_graph_snapshots() {
    for (name, config) in [
        ("akita_small", ProtocolConfig::small()),
        ("akita_small_advice", ProtocolConfig::small_with_advice()),
        (
            "akita_small_committed",
            ProtocolConfig::small_committed_program(),
        ),
    ] {
        let records = all_vertices(&config);
        let graph = ClaimGraph::build(Piop::Akita, &records, &config, all_aliases());
        assert_graph_snapshot(name, &graph);
    }
}
