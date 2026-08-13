//! Const adjacency metadata for the claim graph.
//!
//! The `InputClaims`/`OutputClaims` derives emit a [`ClaimAdjacency`] impl for
//! every claim struct, generated from the same `#[opening(..)]` tokens as the
//! resolution code, so the metadata cannot drift from the struct's semantics.
//! Consumed-claim structs carry their in-edges, produced-claim structs their
//! out-edges. Hand-written aggregate claim structs implement the trait
//! manually. Indexed (`Vec`) families are recorded once at index 0 with
//! [`ClaimArity::Family`]; analyses collapse expression-level ids to that
//! representative. See `specs/claim-graph-analysis.md`.

/// Multiplicity of one edge as declared by a claim-struct field's type.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ClaimArity {
    /// A plain `C` field: exactly one opening.
    Scalar,
    /// An `Option<C>` field: an opening present only in some proof
    /// configurations.
    Optional,
    /// A `Vec<C>` field: an indexed family, represented by its index-0 id.
    Family,
}

/// One edge of the claim graph, as declared by a claim-struct field.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ClaimEdge<Id: 'static> {
    /// Representative opening id (indexed families use index 0).
    pub id: Id,
    pub arity: ClaimArity,
}

/// Claim-graph adjacency contributed by one claim struct: in-edges for a
/// consumed-claim struct, out-edges for a produced-claim struct.
pub trait ClaimAdjacency {
    type Id: 'static;
    const EDGES: &'static [ClaimEdge<Self::Id>];
}
