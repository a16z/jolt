//! Claim-graph analysis over the symbolic protocol layer.
//!
//! Builds the whole-protocol claim-flow DAG from two independent sources —
//! the derive-emitted [`ClaimAdjacency`] metadata and the relations' symbolic
//! input/output expressions — and checks it: unique production, edge
//! resolution, no dangling claims, acyclicity, struct/expression
//! cross-validation. See `specs/claim-graph-analysis.md`.

pub mod graph;
pub mod registry;

use std::collections::BTreeSet;

use jolt_claims::protocols::jolt::{
    JoltExpr, JoltFormulaDimensions, JoltOneHotDimensions, JoltOpeningId, JoltReadWriteConfig,
    JoltRelationId, TracePolynomialOrder,
};
use jolt_claims::{ClaimAdjacency, ClaimEdge, Source, SymbolicSumcheck};
use jolt_field::Fr;

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Piop {
    Dory,
    Akita,
}

/// The configuration a claim graph is expanded under: formula dimensions plus
/// the presence flags that gate conditional (Optional) openings and sinks.
pub struct ProtocolConfig {
    pub formula: JoltFormulaDimensions,
    pub log_t: usize,
    /// Committed one-hot chunk bits (`JoltOneHotConfig::committed_chunk_bits`
    /// on the verifier); feeds the booleanity / hamming-weight shapes and the
    /// precommitted scheduling reference.
    pub log_k_chunk: usize,
    /// RAM address bits (`checked.ram_K.ilog2()` on the verifier).
    pub ram_log_k: usize,
    /// Read-write phase splits (`proof.rw_config` on the verifier).
    pub read_write: JoltReadWriteConfig,
    /// Trace grid interleaving (`proof.trace_polynomial_order`).
    pub trace_order: TracePolynomialOrder,
    pub trusted_advice: bool,
    pub untrusted_advice: bool,
    /// Max advice bytes used for any present advice kind (the verifier carries
    /// a per-kind max; one knob suffices for shape expansion). Read only when
    /// an advice flag is set.
    pub advice_max_bytes: usize,
    /// `Some(chunk_count)` in committed-program mode.
    pub committed_program_chunks: Option<usize>,
    /// Program-image length in words; read only in committed-program mode.
    pub program_image_len_words: usize,
}

impl ProtocolConfig {
    /// Small full-program configuration without advice.
    pub fn small() -> Self {
        let log_t = 20;
        let formula = JoltFormulaDimensions::try_from(JoltOneHotDimensions {
            log_t,
            instruction_address_bits: 128,
            bytecode_k: 1024,
            ram_k: 4096,
            committed_chunk_bits: 8,
            lookup_virtual_chunk_bits: 32,
        })
        .expect("small fixture dimensions are valid");
        Self {
            formula,
            log_t,
            log_k_chunk: 8,
            ram_log_k: 12,
            read_write: JoltReadWriteConfig {
                ram_rw_phase1_num_rounds: 2,
                ram_rw_phase2_num_rounds: 1,
                registers_rw_phase1_num_rounds: 2,
                registers_rw_phase2_num_rounds: 1,
            },
            trace_order: TracePolynomialOrder::CycleMajor,
            trusted_advice: false,
            untrusted_advice: false,
            advice_max_bytes: 2048,
            committed_program_chunks: None,
            program_image_len_words: 0,
        }
    }
}

/// A relation's place in the claim graph: how it instantiates into concrete
/// vertices under a configuration. Which PIOP graphs contain it is declared by
/// the section it is listed under in `claim_graph_vertices!`.
pub trait ProtocolVertex: SymbolicSumcheck {
    /// One `Shape` per vertex; phase-split relations yield several.
    fn instances(config: &ProtocolConfig) -> Vec<Self::Shape>;
}

/// One vertex of the claim graph, as owned data: the relation's expression
/// edges (expanded under a configuration) plus the derive-emitted struct
/// adjacency for cross-validation.
#[derive(Clone, Debug)]
pub struct VertexRecord {
    pub name: &'static str,
    pub relation: JoltRelationId,
    pub graphs: &'static [Piop],
    pub rounds: usize,
    pub degree: usize,
    pub in_edges: BTreeSet<JoltOpeningId>,
    pub out_edges: BTreeSet<JoltOpeningId>,
    pub struct_in_edges: &'static [ClaimEdge<JoltOpeningId>],
    pub struct_out_edges: &'static [ClaimEdge<JoltOpeningId>],
}

/// Extracts one relation type's vertices. Erases the non-object-safe
/// `SymbolicSumcheck` into plain records; everything downstream is ordinary
/// graph code.
pub fn vertices<S>(config: &ProtocolConfig, graphs: &'static [Piop]) -> Vec<VertexRecord>
where
    S: ProtocolVertex<
        RelationId = JoltRelationId,
        OpeningId = JoltOpeningId,
        DerivedId = jolt_claims::protocols::jolt::JoltDerivedId,
        ChallengeId = jolt_claims::protocols::jolt::JoltChallengeId,
    >,
    S::Inputs<()>: ClaimAdjacency<Id = JoltOpeningId>,
    S::Outputs<()>: ClaimAdjacency<Id = JoltOpeningId>,
{
    S::instances(config)
        .into_iter()
        .map(|shape| {
            let relation = S::new(shape);
            let struct_in_edges = <S::Inputs<()> as ClaimAdjacency>::EDGES;
            let struct_out_edges = <S::Outputs<()> as ClaimAdjacency>::EDGES;
            VertexRecord {
                name: short_type_name::<S>(),
                relation: S::id(),
                graphs,
                rounds: relation.rounds(),
                degree: relation.degree(),
                in_edges: edge_union(&relation.input_expression::<Fr>(), struct_in_edges),
                out_edges: edge_union(&relation.output_expression::<Fr>(), struct_out_edges),
                struct_in_edges,
                struct_out_edges,
            }
        })
        .collect()
}

/// A manifest-declaration record for a relation type with no instances under
/// the current configuration (advice / committed-program conditionals): keeps
/// the exhaustiveness backstop covered while contributing to no PIOP graph —
/// `graphs` is empty, so `ClaimGraph::build` filters it out of every graph,
/// and `rounds`/`degree` are never consulted.
pub fn declaration_only<S>() -> VertexRecord
where
    S: ProtocolVertex<
        RelationId = JoltRelationId,
        OpeningId = JoltOpeningId,
        DerivedId = jolt_claims::protocols::jolt::JoltDerivedId,
        ChallengeId = jolt_claims::protocols::jolt::JoltChallengeId,
    >,
    S::Inputs<()>: ClaimAdjacency<Id = JoltOpeningId>,
    S::Outputs<()>: ClaimAdjacency<Id = JoltOpeningId>,
{
    VertexRecord {
        name: short_type_name::<S>(),
        relation: S::id(),
        graphs: &[],
        rounds: 0,
        degree: 0,
        in_edges: BTreeSet::new(),
        out_edges: BTreeSet::new(),
        struct_in_edges: <S::Inputs<()> as ClaimAdjacency>::EDGES,
        struct_out_edges: <S::Outputs<()> as ClaimAdjacency>::EDGES,
    }
}

/// A vertex's produced/consumed set: the expression's opening ids plus the
/// struct-declared scalar and optional openings the expression omits. The
/// latter are *forwarded* openings — opened at the relation's point but
/// constrained downstream (see `wire_output_openings` in jolt-verifier's
/// `stages/relations.rs`, which documents the product remainder's flags as
/// the canonical example).
fn edge_union(
    expr: &JoltExpr<Fr>,
    struct_edges: &'static [ClaimEdge<JoltOpeningId>],
) -> BTreeSet<JoltOpeningId> {
    let mut ids = opening_ids(expr);
    for edge in struct_edges {
        if edge.arity != jolt_claims::ClaimArity::Family {
            let _ = ids.insert(edge.id);
        }
    }
    ids
}

/// The opening ids referenced by an expression, in factor position.
pub fn opening_ids(expr: &JoltExpr<Fr>) -> BTreeSet<JoltOpeningId> {
    expr.terms
        .iter()
        .flat_map(|term| &term.factors)
        .filter_map(|factor| match factor {
            Source::Opening(id) => Some(*id),
            Source::Challenge(_) | Source::Derived(_) => None,
        })
        .collect()
}

fn short_type_name<S>() -> &'static str {
    let full = std::any::type_name::<S>();
    full.rsplit("::").next().unwrap_or(full)
}
