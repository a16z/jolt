//! The byte link's wire.

use jolt_field::JoltField;
use serde::{Deserialize, Serialize};

/// Everything the link sends besides derived values (spec §2): the two `W`
/// commitments, the fraction-tree roots, each GKR layer's rounds and bound
/// children, and the three degree-two reductions with their final
/// evaluations. Round polynomials omit their linear coefficient, which the
/// verifier recovers from the running claim.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(
    serialize = "F: Serialize, C: Serialize",
    deserialize = "F: for<'a> Deserialize<'a>, C: for<'a> Deserialize<'a>"
))]
pub struct ByteLinkProof<F: JoltField, C> {
    /// `W` of the triple histograms, then of the RAM histogram.
    pub histogram_commitments: [C; 2],
    /// `(P, B)` roots of the seven trace trees, in pack order.
    pub trace_roots: [[F; 2]; 7],
    /// `(P, B)` roots of the seven table trees, in pack order.
    pub table_roots: [[F; 2]; 7],
    pub trace: Vec<GkrLayerProof<F>>,
    pub triples: Vec<GkrLayerProof<F>>,
    pub ram: Vec<GkrLayerProof<F>>,
    pub triple_query: ReductionProof<F>,
    pub ram_query: ReductionProof<F>,
    pub source: ReductionProof<F>,
}

/// One layer of a batched fraction-tree GKR: a cubic round per variable of
/// the parent index, then `[P_0, B_0, P_1, B_1]` of every tree's bound
/// children.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: for<'a> Deserialize<'a>"))]
pub struct GkrLayerProof<F: JoltField> {
    pub rounds: Vec<[F; 3]>,
    pub children: Vec<[F; 4]>,
}

/// A degree-two reduction: its rounds and the reduced polynomials' values at
/// the final point.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound(serialize = "F: Serialize", deserialize = "F: for<'a> Deserialize<'a>"))]
pub struct ReductionProof<F: JoltField> {
    pub rounds: Vec<[F; 2]>,
    pub finals: Vec<F>,
}
