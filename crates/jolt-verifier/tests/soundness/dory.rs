//! Clear Dory fixtures. Legacy-shape fixtures run with field-inline
//! disabled; the field-inline eq-MLE fixture runs with it enabled.

use super::checks;

#[cfg(not(feature = "field-inline"))]
mod base {
    use super::checks;
    use crate::support::verifier_fixtures::{
        standard_advice_consumer_case, standard_committed_muldiv_case, standard_muldiv_case,
    };

    /// Tamper cap for the larger fixtures; muldiv runs every tamper.
    const SAMPLE: Option<usize> = Some(400);

    #[test]
    fn standard_muldiv_message_sweep() {
        checks::message_sweep(&standard_muldiv_case(), None);
    }

    #[test]
    fn standard_muldiv_structural_tampers() {
        checks::structural_tampers(&standard_muldiv_case(), None);
    }

    #[test]
    fn standard_muldiv_statement_tampers() {
        let case = standard_muldiv_case();
        checks::statement_tampers(&case);
        checks::header_equivocations(&case);
        checks::commitment_substitutions(&case);
        checks::trusted_advice_tampers(&case, &checks::sent_commitment(&case, 0));
        checks::preprocessing_swap(&case, &standard_committed_muldiv_case().preprocessing);
    }

    #[test]
    fn standard_advice_message_sweep() {
        let case = standard_advice_consumer_case();
        checks::message_sweep(&case, SAMPLE);
        checks::structural_tampers(&case, Some(32));
    }

    #[test]
    fn standard_advice_statement_tampers() {
        let case = standard_advice_consumer_case();
        assert!(case.trusted_advice_commitment.is_some());
        checks::statement_tampers(&case);
        checks::trusted_advice_tampers(&case, &checks::sent_commitment(&case, 0));
    }

    #[test]
    fn standard_committed_message_sweep() {
        let case = standard_committed_muldiv_case();
        checks::message_sweep(&case, SAMPLE);
        checks::structural_tampers(&case, Some(32));
    }

    #[test]
    fn standard_committed_program_tampers() {
        checks::committed_program_tampers(
            &standard_committed_muldiv_case(),
            &[
                ("swapped bytecode chunk commitments", |preprocessing| {
                    let committed = checks::committed_mut(preprocessing);
                    assert!(committed.bytecode_chunk_commitments.len() >= 2);
                    committed.bytecode_chunk_commitments.swap(0, 1);
                }),
                (
                    "program image replaced by a chunk commitment",
                    |preprocessing| {
                        let committed = checks::committed_mut(preprocessing);
                        committed.program_image_commitment =
                            committed.bytecode_chunk_commitments[0].clone();
                    },
                ),
            ],
        );
    }
}

#[cfg(feature = "field-inline")]
#[test]
fn field_inline_eqpoly_message_sweep() {
    use crate::support::verifier_fixtures::standard_field_inline_eqpoly_case;

    let case = standard_field_inline_eqpoly_case();
    checks::message_sweep(&case, Some(600));
    checks::structural_tampers(&case, Some(64));
}

#[cfg(feature = "field-inline")]
#[test]
fn field_inline_eqpoly_statement_tampers() {
    use crate::support::verifier_fixtures::standard_field_inline_eqpoly_case;

    let case = standard_field_inline_eqpoly_case();
    checks::statement_tampers(&case);
    checks::header_equivocations(&case);
    checks::commitment_substitutions(&case);
}
