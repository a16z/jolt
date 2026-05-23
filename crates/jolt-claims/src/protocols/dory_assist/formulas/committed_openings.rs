//! Dory-assist committed-polynomial opening order used by the Hyrax check.

use super::super::{DoryAssistCommittedPolynomial, DoryAssistOpeningId, DoryAssistRelationId};

pub fn final_opening_polynomial_order() -> [DoryAssistCommittedPolynomial; 1] {
    [DoryAssistCommittedPolynomial::DenseWitness]
}

pub fn final_opening_ids() -> [DoryAssistOpeningId; 1] {
    [DoryAssistOpeningId::dense_witness(
        DoryAssistRelationId::PrefixPacking,
    )]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn final_opening_order_is_single_dense_witness() {
        assert_eq!(
            final_opening_polynomial_order(),
            [DoryAssistCommittedPolynomial::DenseWitness]
        );
        assert_eq!(
            final_opening_ids(),
            [DoryAssistOpeningId::dense_witness(
                DoryAssistRelationId::PrefixPacking
            )]
        );
    }
}
