//! The optimized field-registers value-evaluation (stage 5) kernel: the
//! integer-register kernel ([`super::registers_val_evaluation`]) at the FR
//! geometry, byte-parity twin of
//! [`crate::reference::field_registers_val_evaluation`].
//!
//! The reference folds the `(2^4 × T)` one-hot `FieldRdWa` grid into a dense
//! cycle table at prepare time and binds three dense `T`-sized tables per
//! round. This kernel keeps the sibling's technique set:
//!
//! - **Lazy one-hot `wa`** ([`WaState`]): round 0 serves
//!   `wa(j) = eq(r_address)[rd_j]` straight from the per-cycle FR write slots
//!   and the K = 16 eq table — the grid is never materialized. The write
//!   slots are reclaimed from the stage-4 FR kernel's session carry
//!   ([`SharedFieldRdWrites`]), avoiding a second oracle-row walk.
//! - **Split LT** ([`SplitLt`]): `LT(j, r_cycle)` from three ~√T tables.
//! - **Eval-at-{0,2,3} sampling** with the engine hint supplying s(1).
//! - **Deferred-reduction accumulation** of the triple products.
//!
//! The increment column is the FR oracle's single committed polynomial
//! (`FieldRdInc`), materialized dense at prepare — there is no slice-backed
//! deferral like the integer sibling's because the FR oracle serves whole
//! tables only, and the column is all-zero exactly when the trace is
//! FR-inactive (a cheap bind).

use jolt_claims::protocols::field_inline::{
    FieldInlineCommittedPolynomial, FieldInlineDerivedId, FieldInlinePolynomialId,
    FieldRegistersValEvaluationPublic, FIELD_REGISTERS_LOG_K,
};
use jolt_field::JoltField;
use jolt_poly::{BindingOrder, EqPolynomial, Polynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck as _, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage5::field_registers_val_evaluation::FieldRegistersValEvaluation;
use jolt_verifier::VerifierError;
use jolt_witness::{JoltWitnessPlane, WitnessError};

use super::field_registers_read_write::SharedFieldRdWrites;
use super::registers_val_evaluation::WaState;
use super::support::{triple_product_round_evals, RoundProgress, SplitLt};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

pub struct OptimizedFieldRegistersValEvaluation;

impl<F: JoltField> PrepareKernel<F, FieldRegistersValEvaluation<F>>
    for OptimizedFieldRegistersValEvaluation
{
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, FieldRegistersValEvaluation<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = FieldRegistersValEvaluation<F>>>, KernelError<F>>
    {
        let relation = inputs.relation;
        let log_t = relation.trace_dimensions().log_t();
        if log_t == 0 {
            return Err(KernelError::Unsupported {
                reason: "optimized FR val-evaluation requires at least one cycle round",
            });
        }
        let registers_val_point: &[F] = &inputs.points.registers_val;
        if registers_val_point.len() != FIELD_REGISTERS_LOG_K + log_t {
            return Err(KernelError::InvariantViolation {
                reason: "FR value-evaluation input point has the wrong variable count",
            });
        }
        let (r_address, r_cycle) = registers_val_point.split_at(FIELD_REGISTERS_LOG_K);
        let cycles = 1usize << log_t;

        let field_inline =
            witness
                .field_inline()
                .ok_or(KernelError::Witness(WitnessError::UnavailableView {
                    label: "field-registers value-evaluation field-inline oracle",
                }))?;
        let inc_table = field_inline.oracle_table(FieldInlinePolynomialId::Committed(
            FieldInlineCommittedPolynomial::FieldRdInc,
        ))?;
        if inc_table.len() != cycles {
            return Err(KernelError::TableSizeMismatch {
                table: "FieldRdInc".to_owned(),
                expected: cycles,
                got: inc_table.len(),
            });
        }

        // Reclaim the FR write slots the stage-4 kernel parked; collect them
        // from the oracle rows otherwise (reference-only stage 4, tests).
        let writes = match session.take::<SharedFieldRdWrites>() {
            Some(SharedFieldRdWrites(writes))
                if writes
                    .last()
                    .is_none_or(|&(cycle, _)| (cycle as usize) < cycles) =>
            {
                writes
            }
            _ => field_inline
                .field_inline_register_read_write_rows()?
                .iter()
                .enumerate()
                .filter_map(|(cycle, row)| row.rd.map(|write| (cycle as u32, write.register)))
                .collect(),
        };
        let mut rd: Vec<Option<u8>> = vec![None; cycles];
        for (cycle, register) in writes {
            let slot = rd
                .get_mut(cycle as usize)
                .ok_or(KernelError::InvariantViolation {
                    reason: "FR write slot outside the cycle domain",
                })?;
            if usize::from(register) >= (1usize << FIELD_REGISTERS_LOG_K) {
                return Err(KernelError::InvariantViolation {
                    reason: "FR register index outside the field-register domain",
                });
            }
            *slot = Some(register);
        }

        Ok(Box::new(FieldValEvaluationKernel {
            progress: RoundProgress::new(log_t),
            inc: Polynomial::new(inc_table),
            wa: WaState::Indices {
                rd,
                eq_address: EqPolynomial::<F>::evals(r_address, None),
            },
            lt: SplitLt::new(r_cycle),
        }))
    }
}

#[cfg_attr(
    feature = "allocative",
    derive(allocative::Allocative),
    allocative(bound = "F: JoltField")
)]
struct FieldValEvaluationKernel<F: JoltField> {
    progress: RoundProgress,
    inc: Polynomial<F>,
    wa: WaState<F>,
    lt: SplitLt<F>,
}

impl<F: JoltField> FieldValEvaluationKernel<F> {
    fn bind(&mut self, challenge: F) {
        self.inc.bind_with_order(challenge, BindingOrder::LowToHigh);
        self.wa.bind(challenge);
        self.lt.bind(challenge);
        self.progress.advance();
    }
}

impl<F: JoltField> ProveRounds<F> for FieldValEvaluationKernel<F> {
    fn num_rounds(&self) -> usize {
        self.progress.total()
    }

    fn prove_round(
        &mut self,
        bind: Option<F>,
        _round: usize,
        previous_claim: F,
    ) -> Result<UnivariatePoly<F>, SumcheckError<F>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        let half = self.inc.len() / 2;
        let inc = self.inc.evals();
        let evals = triple_product_round_evals(
            half,
            |y| (inc[2 * y], inc[2 * y + 1]),
            |y| self.wa.pair(y),
            |y| self.lt.pair(y),
        );
        Ok(UnivariatePoly::from_evals_and_hint(previous_claim, &evals))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: JoltField> SumcheckKernel<F> for FieldValEvaluationKernel<F> {
    type Relation = FieldRegistersValEvaluation<F>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<F, Self::Relation>, SumcheckKernelError<F>> {
        use jolt_claims::protocols::field_inline::relations::registers::FieldRegistersValEvaluationOutputClaims;

        self.progress.require_complete()?;
        Ok(FieldRegistersValEvaluationOutputClaims {
            rd_inc: self.inc.evals()[0],
            rd_wa: self.wa.final_value(),
        })
    }

    /// Pin the split-LT tables to the verifier's scalar path: the fully bound
    /// LT value must equal `derive_output_term(LtCycle)` (the reference
    /// kernel's tie-down on the table it materializes).
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        let expected = relation.derive_output_term(
            &FieldInlineDerivedId::from(FieldRegistersValEvaluationPublic::LtCycle),
            input_points,
            output_points,
            challenges,
        )?;
        let got = self.lt.final_value();
        if got != expected {
            return Err(SumcheckKernelError::Verifier(
                VerifierError::StageClaimSumcheckFailed {
                    stage: "FieldRegistersValEvaluation".to_string(),
                    reason: format!(
                        "bound LT value {got:?}, but derive_output_term gives {expected:?}"
                    ),
                },
            ));
        }
        Ok(())
    }
}

/// Byte parity against the reference kernel on register-consistent FR
/// traces, covering both index sources (parked by stage 4 vs collected from
/// the oracle rows) and the FR-inactive degenerate case.
#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::field_inline::relations::registers::FieldRegistersValEvaluationInputClaims;
    use jolt_claims::protocols::field_inline::FieldRegistersTraceDimensions;
    use jolt_claims::NoChallenges;
    use jolt_field::{Fr, Ring};
    use jolt_witness::JoltWitnessOracle;

    use super::*;
    use crate::optimized::field_registers_testing::{
        inactive_fr_fixture, structured_fr_fixture, FrTraceFixture,
    };
    use crate::optimized::parity::{
        probe_input_claim, run_lockstep, run_lockstep_degenerate, synthetic_point,
    };
    use crate::ReferenceBackend;

    enum IndexSource {
        Collect,
        Parked,
        StaleParked,
    }

    fn run_parity(
        fixture: FrTraceFixture,
        log_t: usize,
        seed: u64,
        expect_active: bool,
        source: &IndexSource,
    ) {
        fixture.with_plane(log_t, |backend| {
            let relation =
                FieldRegistersValEvaluation::<Fr>::new(FieldRegistersTraceDimensions::new(log_t));
            let point = synthetic_point(FIELD_REGISTERS_LOG_K + log_t, seed);
            let claims = FieldRegistersValEvaluationInputClaims {
                registers_val: Fr::from_u64(0),
            };
            let points = FieldRegistersValEvaluationInputClaims {
                registers_val: point,
            };
            let challenges = NoChallenges::default();
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };

            let mut session = ProofSession::default();
            match source {
                IndexSource::Collect => {}
                IndexSource::Parked => {
                    let oracle: &dyn jolt_witness::field_inline::FieldInlineWitnessOracle<Fr> =
                        backend.field_inline().unwrap();
                    let writes = oracle
                        .field_inline_register_read_write_rows()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .filter_map(|(cycle, row)| {
                            row.rd.map(|write| (cycle as u32, write.register))
                        })
                        .collect();
                    session.park(super::SharedFieldRdWrites(writes));
                }
                IndexSource::StaleParked => {
                    // An out-of-domain cycle: prepare must fall back to
                    // collecting from the oracle rows.
                    session.park(super::SharedFieldRdWrites(vec![(1 << log_t, 0)]));
                }
            }

            let mut reference = <ReferenceBackend as PrepareKernel<
                Fr,
                FieldRegistersValEvaluation<Fr>,
            >>::prepare(
                &ReferenceBackend, &mut session, backend, inputs()
            )
            .unwrap();
            let mut optimized = OptimizedFieldRegistersValEvaluation
                .prepare(&mut session, backend, inputs())
                .unwrap();

            let claim = probe_input_claim(reference.as_mut());
            let round_challenges =
                synthetic_point(relation.rounds(), seed.wrapping_mul(0x9E37_79B9));
            if expect_active {
                assert!(claim != Fr::from_u64(0), "FR-active fixture degenerated");
                run_lockstep(
                    reference.as_mut(),
                    optimized.as_mut(),
                    claim,
                    &round_challenges,
                );
            } else {
                assert_eq!(claim, Fr::from_u64(0), "FR-inactive claim must be zero");
                run_lockstep_degenerate(
                    reference.as_mut(),
                    optimized.as_mut(),
                    claim,
                    &round_challenges,
                );
            }
            assert_eq!(
                reference.output_claims(&claims).unwrap(),
                optimized.output_claims(&claims).unwrap()
            );
            let output_points = relation
                .derive_opening_points(&round_challenges, &points)
                .unwrap();
            reference
                .validate_derived_tables(&relation, &points, &output_points, &challenges)
                .unwrap();
            optimized
                .validate_derived_tables(&relation, &points, &output_points, &challenges)
                .unwrap();
        });
    }

    #[test]
    fn parity_structured_collected_indices() {
        run_parity(
            structured_fr_fixture(16),
            4,
            211,
            true,
            &IndexSource::Collect,
        );
    }

    #[test]
    fn parity_structured_parked_indices() {
        run_parity(structured_fr_fixture(8), 3, 223, true, &IndexSource::Parked);
    }

    #[test]
    fn parity_stale_parked_indices_fall_back() {
        run_parity(
            structured_fr_fixture(8),
            3,
            227,
            true,
            &IndexSource::StaleParked,
        );
    }

    #[test]
    fn parity_inactive_trace_is_degenerate() {
        run_parity(inactive_fr_fixture(4), 3, 229, false, &IndexSource::Collect);
    }
}
