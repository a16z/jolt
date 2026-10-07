//! The byte link (akita phase-2 spec §2, W-only): authenticates the twenty
//! one-hot claims and `F(r)` stage 6b leaves on the byte trace `Q` against
//! `Q` itself. Each pack's eq-weighted tuple histogram `W` is committed after
//! stage 6b; `Σ_t eq(r, t) / (β − γ·D(t)) = Σ_h W(h) / (β − γ·h)` is checked
//! as the cross product of two fraction-tree roots, each proved by a batched
//! binary GKR, and degree-two reductions take the histogram queries and every
//! claim on `Q` to one point per committed object for stage 8.
//!
//! A layer's sumcheck binds the low bit of the parent index first, so its
//! challenges `s` name the parent point `reverse(s)` and the child point
//! `(reverse(s), µ)`; every point here is MSB-first.

mod proof;
pub mod transcript;
mod verify;

pub use proof::{ByteLinkProof, GkrLayerProof, ReductionProof};
pub use verify::verify;

/// The compression challenges drawn after the `W` commitments.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ByteLinkCompression<F> {
    /// `γ` of each pack column, in pack order.
    pub gamma: [[F; 3]; 7],
    pub beta: F,
}

/// The `Q` reduction's batching weights.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SourceWeights<F> {
    /// One per pack's trace denominator leaf.
    pub denominators: [F; 7],
    pub fused_inc: F,
    pub zero_slots: [F; 2],
}

/// Every polynomial of one committed object at one point (MSB-first).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ByteLinkOpening<F> {
    pub point: Vec<F>,
    pub values: Vec<F>,
}

/// The claims the link leaves for stage 8: `W` of the six triple packs, `W`
/// of the RAM pack, and every column of `Q` in slot order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ByteLinkOpenings<F> {
    pub triples: ByteLinkOpening<F>,
    pub ram: ByteLinkOpening<F>,
    pub source: ByteLinkOpening<F>,
}
