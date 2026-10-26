use jolt_crypto::VectorCommitmentOpening;
use jolt_field::JoltField;
use jolt_sumcheck::{CompressedSumcheckProof, Nonced, NoncedRef};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlindFoldProof<F: JoltField, Com> {
    pub auxiliary_row_commitments: Vec<Com>,
    pub random_round_commitments: Vec<Com>,
    pub random_output_claim_row_commitments: Vec<Com>,
    pub random_auxiliary_row_commitments: Vec<Com>,
    pub random_error_row_commitments: Vec<Com>,
    pub random_eval_commitments: Vec<Com>,
    pub random_u: F,
    pub cross_term_error_row_commitments: Vec<Com>,
    /// Proof-of-work nonces for the folding challenge, the outer zero-check
    /// point and the inner combination, in that order, one for each of these
    /// draws whose difficulty is positive.
    pub pow_nonces: Vec<u64>,
    pub outer_sumcheck: CompressedSumcheckProof<F>,
    pub az_rx: F,
    pub bz_rx: F,
    pub cz_rx: F,
    pub inner_sumcheck: CompressedSumcheckProof<F>,
    pub witness_opening: VectorCommitmentOpening<F>,
    pub error_opening: VectorCommitmentOpening<F>,
    pub folded_eval_outputs: Vec<F>,
    pub folded_eval_blindings: Vec<F>,
    pub folded_eval_output_openings: Vec<VectorCommitmentOpening<F>>,
    pub folded_eval_blinding_openings: Vec<VectorCommitmentOpening<F>>,
}

/// The fields in declaration order, with the nonces carried by the
/// cross-term commitments' [`Nonced`] header, so a proof with no nonces costs
/// nothing for them.
impl<F, Com> Serialize for BlindFoldProof<F, Com>
where
    F: JoltField + Serialize,
    Com: Serialize,
{
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let instances = (
            &self.auxiliary_row_commitments,
            &self.random_round_commitments,
            &self.random_output_claim_row_commitments,
            &self.random_auxiliary_row_commitments,
            &self.random_error_row_commitments,
            &self.random_eval_commitments,
            &self.random_u,
        );
        let cross_terms = NoncedRef {
            messages: &self.cross_term_error_row_commitments,
            pow_nonces: &self.pow_nonces,
        };
        let folded = (
            &self.outer_sumcheck,
            &self.az_rx,
            &self.bz_rx,
            &self.cz_rx,
            &self.inner_sumcheck,
            &self.witness_opening,
            &self.error_opening,
            &self.folded_eval_outputs,
            &self.folded_eval_blindings,
            &self.folded_eval_output_openings,
            &self.folded_eval_blinding_openings,
        );
        (instances, cross_terms, folded).serialize(serializer)
    }
}

impl<'de, F, Com> Deserialize<'de> for BlindFoldProof<F, Com>
where
    F: JoltField + for<'a> Deserialize<'a>,
    Com: Deserialize<'de>,
{
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        type Instances<F, Com> = (
            Vec<Com>,
            Vec<Com>,
            Vec<Com>,
            Vec<Com>,
            Vec<Com>,
            Vec<Com>,
            F,
        );
        type Folded<F> = (
            CompressedSumcheckProof<F>,
            F,
            F,
            F,
            CompressedSumcheckProof<F>,
            VectorCommitmentOpening<F>,
            VectorCommitmentOpening<F>,
            Vec<F>,
            Vec<F>,
            Vec<VectorCommitmentOpening<F>>,
            Vec<VectorCommitmentOpening<F>>,
        );
        let (
            (
                auxiliary_row_commitments,
                random_round_commitments,
                random_output_claim_row_commitments,
                random_auxiliary_row_commitments,
                random_error_row_commitments,
                random_eval_commitments,
                random_u,
            ),
            cross_terms,
            (
                outer_sumcheck,
                az_rx,
                bz_rx,
                cz_rx,
                inner_sumcheck,
                witness_opening,
                error_opening,
                folded_eval_outputs,
                folded_eval_blindings,
                folded_eval_output_openings,
                folded_eval_blinding_openings,
            ),
        ) = <(Instances<F, Com>, Nonced<Com>, Folded<F>)>::deserialize(deserializer)?;
        Ok(Self {
            auxiliary_row_commitments,
            random_round_commitments,
            random_output_claim_row_commitments,
            random_auxiliary_row_commitments,
            random_error_row_commitments,
            random_eval_commitments,
            random_u,
            cross_term_error_row_commitments: cross_terms.messages,
            pow_nonces: cross_terms.pow_nonces,
            outer_sumcheck,
            az_rx,
            bz_rx,
            cz_rx,
            inner_sumcheck,
            witness_opening,
            error_opening,
            folded_eval_outputs,
            folded_eval_blindings,
            folded_eval_output_openings,
            folded_eval_blinding_openings,
        })
    }
}
