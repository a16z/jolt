//! Akita (packed lattice) fixtures. Legacy-shape fixtures run with
//! field-inline disabled; the packed field-inline eq-MLE fixture runs with
//! it enabled.

use super::checks;

#[cfg(not(feature = "field-inline"))]
mod base {
    use jolt_claims::protocols::jolt::TracePolynomialOrder;
    use jolt_verifier::VerifierError;

    use super::checks;
    use crate::support::akita_fixtures::{
        akita_advice_case, akita_committed_muldiv_case, akita_muldiv_case,
    };
    use crate::support::narg::TracedCase;

    #[test]
    fn akita_muldiv_message_sweep() {
        let case = akita_muldiv_case();
        checks::message_sweep(case, None);
        checks::structural_tampers(case, None);
    }

    #[test]
    fn akita_muldiv_statement_tampers() {
        let case = akita_muldiv_case();
        checks::statement_tampers(case);
        checks::header_equivocations(case);
        checks::preprocessing_swap(case, &akita_committed_muldiv_case().preprocessing);
    }

    #[test]
    fn akita_advice_message_sweep() {
        let case = akita_advice_case();
        checks::message_sweep(case, None);
        checks::trusted_advice_tampers(case, &checks::sent_commitment(case, 1));
    }

    #[test]
    fn akita_committed_message_sweep() {
        let case = akita_committed_muldiv_case();
        checks::message_sweep(case, None);
        checks::committed_program_tampers(
            case,
            &[("swapped direct program commitments", |preprocessing| {
                let committed = checks::committed_mut(preprocessing);
                assert!(committed.direct_program_commitments.len() >= 2);
                committed.direct_program_commitments.swap(0, 1);
            })],
        );

        let mut preprocessing = case.preprocessing.clone();
        let committed = checks::committed_mut(&mut preprocessing);
        committed.trace_order = match committed.trace_order {
            TracePolynomialOrder::CycleMajor => TracePolynomialOrder::AddressMajor,
            TracePolynomialOrder::AddressMajor => TracePolynomialOrder::CycleMajor,
        };
        assert!(matches!(
            case.verify_statement_with(&preprocessing),
            Err(VerifierError::InvalidCommittedProgram { .. })
        ));
    }
}

#[cfg(feature = "field-inline")]
#[test]
fn akita_field_inline_eqpoly_message_sweep() {
    use crate::support::akita_fixtures::akita_field_inline_eqpoly_case;

    let case = akita_field_inline_eqpoly_case();
    checks::message_sweep(case, None);
    checks::structural_tampers(case, Some(64));
    checks::statement_tampers(case);
}
