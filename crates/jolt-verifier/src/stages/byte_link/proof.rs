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

#[cfg(test)]
#[expect(
    clippy::arithmetic_side_effects,
    reason = "tests count wire elements with plain arithmetic"
)]
mod tests {
    use jolt_claims::protocols::jolt::lattice::byte_link::{
        ByteLinkBatch, HistogramGroup, BYTE_LINK_PACKS,
    };
    use jolt_field::{Fr, Zero};

    use super::*;

    fn field_elements<F: JoltField, C>(proof: &ByteLinkProof<F, C>) -> usize {
        let ByteLinkProof {
            histogram_commitments: _,
            trace_roots,
            table_roots,
            trace,
            triples,
            ram,
            triple_query,
            ram_query,
            source,
        } = proof;
        let gkr = |layers: &[GkrLayerProof<F>]| {
            layers
                .iter()
                .map(|GkrLayerProof { rounds, children }| 3 * rounds.len() + 4 * children.len())
                .sum::<usize>()
        };
        let reduction =
            |ReductionProof { rounds, finals }: &ReductionProof<F>| 2 * rounds.len() + finals.len();
        2 * (trace_roots.len() + table_roots.len())
            + gkr(trace)
            + gkr(triples)
            + gkr(ram)
            + reduction(triple_query)
            + reduction(ram_query)
            + reduction(source)
    }

    /// At 2^29 cycles the link sends 4,117 field elements beside its two `W`
    /// commitments: 65,872 bytes of Fp128, the spec §2 budget.
    #[test]
    fn byte_link_wire_at_2_29_is_the_counted_size() {
        const LOG_T: usize = 29;
        let gkr = |batch: ByteLinkBatch, trees: usize| {
            (0..batch.num_vars(LOG_T))
                .map(|layer| GkrLayerProof {
                    rounds: vec![[Fr::zero(); 3]; layer],
                    children: vec![[Fr::zero(); 4]; trees],
                })
                .collect()
        };
        let reduction = |num_vars: usize, finals: usize| ReductionProof {
            rounds: vec![[Fr::zero(); 2]; num_vars],
            finals: vec![Fr::zero(); finals],
        };
        let proof = ByteLinkProof {
            histogram_commitments: [(), ()],
            trace_roots: [[Fr::zero(); 2]; 7],
            table_roots: [[Fr::zero(); 2]; 7],
            trace: gkr(ByteLinkBatch::Trace, BYTE_LINK_PACKS.len()),
            triples: gkr(
                ByteLinkBatch::Triples,
                HistogramGroup::Triples.packs().len(),
            ),
            ram: gkr(ByteLinkBatch::Ram, HistogramGroup::Ram.packs().len()),
            triple_query: reduction(
                HistogramGroup::Triples.num_vars(),
                HistogramGroup::Triples.packs().len(),
            ),
            ram_query: reduction(
                HistogramGroup::Ram.num_vars(),
                HistogramGroup::Ram.packs().len(),
            ),
            source: reduction(LOG_T, 32),
        };
        assert_eq!(16 * field_elements(&proof), 65_872);
    }
}
