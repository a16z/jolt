use crate::util::extend_unique;

use super::super::super::DoryAssistOpeningId;
use super::super::miller_loop::{
    accumulator_input_openings, accumulator_output_openings, boundary_output_openings,
    line_evaluation_output_openings, line_step_output_openings, pair_product_output_openings,
};

pub fn prefix_packing_openings() -> Vec<DoryAssistOpeningId> {
    let mut openings = Vec::new();
    extend_unique(&mut openings, &line_step_output_openings());
    extend_unique(&mut openings, &line_evaluation_output_openings());
    extend_unique(&mut openings, &pair_product_output_openings());
    extend_unique(&mut openings, &accumulator_input_openings());
    extend_unique(&mut openings, &accumulator_output_openings());
    extend_unique(&mut openings, &boundary_output_openings());
    openings
}

#[cfg(test)]
mod tests {
    use super::super::super::super::{
        DoryAssistRelationId, DoryAssistVirtualPolynomial, MillerLoopPolynomial,
    };
    use super::*;

    #[test]
    fn prefix_packing_openings_include_all_miller_loop_subrelations() {
        let openings = prefix_packing_openings();

        assert!(openings.iter().any(|opening| matches!(
            opening,
            DoryAssistOpeningId::Polynomial { relation, .. }
                if *relation == DoryAssistRelationId::MillerLoopLineStep
        )));
        assert!(openings.iter().any(|opening| matches!(
            opening,
            DoryAssistOpeningId::Polynomial { relation, .. }
                if *relation == DoryAssistRelationId::MillerLoopLineEvaluation
        )));
        assert!(openings.iter().any(|opening| matches!(
            opening,
            DoryAssistOpeningId::Polynomial { relation, .. }
                if *relation == DoryAssistRelationId::MillerLoopPairProduct
        )));
        assert!(openings.iter().any(|opening| matches!(
            opening,
            DoryAssistOpeningId::Polynomial { relation, .. }
                if *relation == DoryAssistRelationId::MillerLoopAccumulator
        )));
        assert!(openings.iter().any(|opening| matches!(
            opening,
            DoryAssistOpeningId::Polynomial { relation, .. }
                if *relation == DoryAssistRelationId::MillerLoopBoundary
        )));
        assert!(openings.iter().any(|opening| matches!(
            opening,
            DoryAssistOpeningId::Polynomial {
                polynomial: super::super::super::super::DoryAssistPolynomialId::Virtual(
                    DoryAssistVirtualPolynomial::MillerLoop(
                        MillerLoopPolynomial::PairLineProductCoeff(0)
                    )
                ),
                relation: DoryAssistRelationId::MillerLoopPairProduct,
            }
        )));
    }
}
