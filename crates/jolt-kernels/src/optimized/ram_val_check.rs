//! Optimized RAM value check (stage 4).
//!
//! Summand: `inc(j) · ra(j) · (LT(j, r_cycle) + γ)`. `inc` and the
//! address-folded `ra(j) = eq(r_address, address(j))` vanish off the
//! [`RamEventLog`]'s events, so each binds as a row-sorted
//! [`SparseCycleColumn`]: `inc` over the nonzero increments, `ra` over the
//! remapped accesses. Neither support contains the other, and a row pair can
//! hold an increment in one half and an access in the other: its product
//! vanishes at `t ∈ {0, 1}` but not at the sampled `t ∈ {2, 3}`, so a round
//! sums over the pairs where `inc` has an entry, never only over the rows
//! both columns share. [`SplitLt`] stores `LT + γ` in ~√T space; the rest of
//! the kernel's state scales with the event count, not `T`.

use jolt_claims::protocols::jolt::{JoltDerivedId, RamValCheckPublic};
use jolt_field::{Accumulator, JoltField};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage4::ram_val_check::{RamValCheck, RamValCheckOutputClaims};
use jolt_witness::JoltWitnessPlane;
#[cfg(feature = "parallel")]
use rayon::prelude::*;

use super::ram_trace::RamEventLog;
use super::rw_matrix::SparseCycleColumn;
use super::support::{accumulate_triple_product, pin_derived_term, RoundProgress, SplitLt};
use super::OptimizedBackend;
use crate::reference::views::eq_table;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

impl<F: JoltField> PrepareKernel<F, RamValCheck<F>> for OptimizedBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<F>,
        inputs: ProverInputs<'_, F, RamValCheck<F>>,
    ) -> Result<Box<dyn SumcheckKernel<F, Relation = RamValCheck<F>>>, KernelError<F>> {
        let relation = inputs.relation;
        let log_t = relation.trace_dimensions().log_t();
        let ram_log_k = relation.ram_log_k();
        let ram_val_point: &[F] = &inputs.points.ram_val;
        if ram_val_point.len() != ram_log_k + log_t {
            return Err(KernelError::InvariantViolation {
                reason: "RAM value-check input point has the wrong variable count",
            });
        }
        let (r_address, r_cycle) = ram_val_point.split_at(ram_log_k);

        // Shared, not taken: the stage-5 RA claim reduction consumes the log.
        let log = RamEventLog::shared(session, witness, log_t, 1usize << ram_log_k)?;
        let eq_address = eq_table(r_address);

        Ok(Box::new(RamValCheckKernel {
            progress: RoundProgress::new(log_t),
            inc: SparseCycleColumn::increments(log),
            ra: SparseCycleColumn::folded_ra(log, &eq_address),
            lt: SplitLt::new_plus_constant(r_cycle, inputs.challenges.gamma),
        }))
    }
}

#[cfg_attr(feature = "allocative", derive(allocative::Allocative))]
struct RamValCheckKernel<F: JoltField> {
    progress: RoundProgress,
    inc: SparseCycleColumn<F>,
    ra: SparseCycleColumn<F>,
    lt: SplitLt<F>,
}

impl<F: JoltField> RamValCheckKernel<F> {
    fn bind(&mut self, challenge: F) {
        self.inc.bind(challenge);
        self.ra.bind(challenge);
        self.lt.bind(challenge);
        self.progress.advance();
    }

    /// `[s(0), s(2), s(3)]` of `Σ_y inc_y(t) · ra_y(t) · lt_y(t)` over the
    /// pairs where `inc` has an entry; `ra.pair` is zero on a pair without an
    /// access.
    fn round_evals(&self) -> [F; 3] {
        let accumulate = |mut acc: [F::Accumulator; 3], (pair, inc): (usize, [F; 2])| {
            let [inc_0, inc_1] = inc;
            let [ra_0, ra_1] = self.ra.pair(pair);
            let lt = self.lt.pair(pair);
            accumulate_triple_product(&mut acc, (inc_0, inc_1), (ra_0, ra_1), lt);
            acc
        };
        #[cfg(feature = "parallel")]
        {
            self.inc
                .pairs()
                .fold(|| [F::Accumulator::default(); 3], accumulate)
                .map(|acc| acc.map(F::Accumulator::reduce))
                .reduce(
                    || [F::zero(); 3],
                    |a, b| [a[0] + b[0], a[1] + b[1], a[2] + b[2]],
                )
        }
        #[cfg(not(feature = "parallel"))]
        {
            self.inc
                .pairs()
                .fold([F::Accumulator::default(); 3], accumulate)
                .map(F::Accumulator::reduce)
        }
    }
}

impl<F: JoltField> ProveRounds<F> for RamValCheckKernel<F> {
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
        Ok(UnivariatePoly::from_evals_and_hint(
            previous_claim,
            &self.round_evals(),
        ))
    }

    fn finish_rounds(&mut self, bind: F) -> Result<(), SumcheckError<F>> {
        self.bind(bind);
        Ok(())
    }
}

impl<F: JoltField> SumcheckKernel<F> for RamValCheckKernel<F> {
    type Relation = RamValCheck<F>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<F, Self::Relation>,
    ) -> Result<RamValCheckOutputClaims<F>, SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        // Advice outputs echo their input claims; they are not bound here.
        Ok(RamValCheckOutputClaims {
            untrusted_advice: inputs.untrusted_advice,
            trusted_advice: inputs.trusted_advice,
            program_image: inputs.program_image,
            ram_ra: self.ra.bound_value(),
            ram_inc: self.inc.bound_value(),
        })
    }

    /// Check the bound split-LT value against the verifier's scalar path.
    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<F, Self::Relation>,
        output_points: &SumcheckOutputPoints<F, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<F, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<F>> {
        self.progress.require_complete()?;
        let id = JoltDerivedId::from(RamValCheckPublic::LtCyclePlusGamma);
        pin_derived_term(
            relation,
            id,
            input_points,
            output_points,
            challenges,
            self.lt.final_value(),
        )
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module: fail loudly")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::dimensions::TraceDimensions;
    use jolt_claims::protocols::jolt::geometry::ram::{
        ram_inc_val_check, ram_ra_val_check, RamValCheckInit, RamValCheckInitContribution,
    };
    use jolt_claims::protocols::jolt::relations::ram::{
        RamValCheckChallenges, RamValCheckInputClaims,
    };
    #[cfg(feature = "akita")]
    use jolt_field::Prime128OffsetA7F7;
    use jolt_field::{Fr, Ring};
    use jolt_poly::LtPolynomial;
    use jolt_verifier::stages::relations::ConcreteSumcheck;
    use jolt_witness::{JoltWitnessOracle, TraceBackend, WitnessError};

    use super::super::parity::run_lockstep;
    use super::super::ram_trace::SharedRamAddresses;
    use super::super::registers_read_write::test_support::challenge_sequence;
    use super::super::testing::{ram_fixture, replayed, silent_ram_fixture, FixtureShape, RamOp};
    use super::*;
    use crate::reference::views::address_fold;
    use crate::ReferenceBackend;

    /// Bind challenges for the rounds of a case without exceptional ones.
    const BINDS: [u64; 5] = [
        0x9E37_79B9_7F4A_7C15,
        0xBF58_476D_1CE4_E5B9,
        0x94D0_49BB_1331_11EB,
        0x2545_F491_4F6C_DD1D,
        0x6A09_E667_F3BC_C909,
    ];

    /// The optimized kernel over `witness` against the reference tier over the
    /// resident rows, through `binds`: equal round polynomials and output
    /// claims, the staged advice and program-image claims echoed, and both
    /// LT pins. The kernel leaves the event log parked for stage 5 and
    /// publishes no dense address column.
    fn parity_in_field<F: JoltField>(
        resident: &TraceBackend,
        witness: &TraceBackend,
        shape: FixtureShape,
        binds: &[u64],
        seed: u64,
    ) {
        let r_address = challenge_sequence::<F>(shape.log_k(), seed);
        let r_cycle = challenge_sequence::<F>(shape.log_t, seed ^ 0x11);
        let gamma = challenge_sequence::<F>(1, seed ^ 0x22)[0];
        let relation = RamValCheck::<F>::new(
            TraceDimensions::new(shape.log_t),
            shape.log_k(),
            RamValCheckInit::decomposed(
                F::from_u64(3),
                [
                    RamValCheckInitContribution::program_image(F::from_u64(5)),
                    RamValCheckInitContribution::untrusted(F::from_u64(7)),
                    RamValCheckInitContribution::trusted(F::from_u64(11)),
                ],
            ),
        );
        let claims = RamValCheckInputClaims {
            ram_val: F::zero(),
            ram_val_final: F::zero(),
            untrusted_advice: Some(F::from_u64(13)),
            trusted_advice: Some(F::from_u64(17)),
            program_image: Some(F::from_u64(19)),
        };
        let points = RamValCheckInputClaims::<Vec<F>> {
            ram_val: [r_address.clone(), r_cycle.clone()].concat(),
            ram_val_final: r_address.clone(),
            untrusted_advice: Some(r_address.clone()),
            trusted_advice: Some(r_address.clone()),
            program_image: Some(r_address.clone()),
        };
        let challenges = RamValCheckChallenges { gamma };
        let inputs = || ProverInputs {
            relation: &relation,
            claims: &claims,
            points: &points,
            challenges: &challenges,
        };

        let mut reference_session = ProofSession::default();
        let mut reference = PrepareKernel::<F, _>::prepare(
            &ReferenceBackend,
            &mut reference_session,
            resident,
            inputs(),
        )
        .unwrap();
        let mut session = ProofSession::default();
        let mut optimized =
            PrepareKernel::<F, _>::prepare(&OptimizedBackend, &mut session, witness, inputs())
                .unwrap();
        assert!(session.state::<RamEventLog>().is_some(), "log not parked");
        assert!(
            session.state::<SharedRamAddresses>().is_none(),
            "dense address column published"
        );

        let ra = address_fold::<F>(resident, ram_ra_val_check(), shape.log_t, &r_address).unwrap();
        let inc =
            JoltWitnessOracle::<F>::oracle_table(resident, ram_inc_val_check().polynomial_id())
                .unwrap();
        let lt = LtPolynomial::evaluations(&r_cycle);
        let input_claim = (0..1usize << shape.log_t)
            .map(|j| inc[j] * ra[j] * (lt[j] + gamma))
            .sum();
        let binds: Vec<F> = binds.iter().map(|&bind| F::from_u64(bind)).collect();
        run_lockstep(reference.as_mut(), optimized.as_mut(), input_claim, &binds);

        assert_eq!(
            reference.output_claims(&claims).unwrap(),
            optimized.output_claims(&claims).unwrap(),
            "output claims diverged"
        );
        let output_points = relation.derive_opening_points(&binds, &points).unwrap();
        for kernel in [&reference, &optimized] {
            kernel
                .validate_derived_tables(&relation, &points, &output_points, &challenges)
                .unwrap();
        }
    }

    /// [`parity_in_field`] in each field, on the resident rows and on the same
    /// rows replayed from two-row checkpoints.
    fn run_parity(resident: &TraceBackend, shape: FixtureShape, binds: &[u64], seed: u64) {
        let replay = replayed(resident, 2, None);
        for witness in [resident, &replay] {
            parity_in_field::<Fr>(resident, witness, shape, binds, seed);
            #[cfg(feature = "akita")]
            parity_in_field::<Prime128OffsetA7F7>(resident, witness, shape, binds, seed);
        }
    }

    fn mixed_ops() -> Vec<RamOp> {
        vec![
            RamOp::Write { word: 3, post: 5 },
            RamOp::Read { word: 3 },
            RamOp::None,
            RamOp::Write { word: 9, post: 4 },
            RamOp::Read { word: 9 },
            RamOp::Write { word: 3, post: 8 },
            RamOp::Read { word: 15 },
        ]
    }

    #[test]
    fn matches_reference_on_mixed_traffic() {
        let shape = FixtureShape {
            log_t: 4,
            ram_k: 16,
        };
        let resident = ram_fixture(shape, Vec::new(), mixed_ops());
        run_parity(&resident, shape, &BINDS[..4], 31);
    }

    #[test]
    fn matches_reference_on_odd_log_t() {
        // Five rounds: the split LT collapses its lo tables mid-protocol.
        let mut ops = mixed_ops();
        ops.extend([
            RamOp::Write { word: 7, post: 2 },
            RamOp::Read { word: 7 },
            RamOp::None,
            RamOp::Read { word: 3 },
        ]);
        let shape = FixtureShape {
            log_t: 5,
            ram_k: 16,
        };
        run_parity(&ram_fixture(shape, Vec::new(), ops), shape, &BINDS, 37);
    }

    #[test]
    fn matches_reference_on_single_round() {
        let shape = FixtureShape { log_t: 1, ram_k: 8 };
        let resident = ram_fixture(shape, Vec::new(), vec![RamOp::Write { word: 1, post: 4 }]);
        run_parity(&resident, shape, &BINDS[..1], 41);
    }

    /// Row pairs holding an increment in one half and an access in the other,
    /// in both orders: their products vanish at 0 and 1 but not at 2 and 3.
    /// Also zero-increment accesses, a store that is no event, increment pairs
    /// (2, 1) that the challenge 2 cancels, the challenges 0 and 1, and an event
    /// on the last cycle (the fixture's termination write, increment 1).
    #[test]
    fn matches_reference_on_opposite_half_increments_and_accesses() {
        let shape = FixtureShape {
            log_t: 4,
            ram_k: 16,
        };
        let ops = vec![
            RamOp::UnmappedWrite { pre: 0, post: 2 },
            RamOp::UnmappedWrite { pre: 0, post: 1 },
            RamOp::Read { word: 3 },
            RamOp::UnmappedWrite { pre: 1, post: 4 },
            RamOp::UnmappedWrite { pre: 5, post: 2 },
            RamOp::Read { word: 4 },
            RamOp::Write { word: 3, post: 0 },
            RamOp::Write { word: 5, post: 9 },
            RamOp::None,
            RamOp::Write { word: 5, post: 9 },
            RamOp::UnmappedWrite { pre: 6, post: 6 },
            RamOp::Write { word: 6, post: 7 },
            RamOp::Read { word: 6 },
            RamOp::UnmappedWrite { pre: 0, post: 3 },
            RamOp::UnmappedWrite { pre: 0, post: 2 },
        ];
        let resident = ram_fixture(shape, Vec::new(), ops);
        let schedules: [[u64; 4]; 3] = [[2, 0, 1, 5], [1, 2, 0, 7], [0, 1, 2, 3]];
        for binds in schedules {
            run_parity(&resident, shape, &binds, 43);
        }
    }

    #[test]
    fn matches_reference_without_ram_events() {
        let shape = FixtureShape { log_t: 4, ram_k: 8 };
        run_parity(&silent_ram_fixture(shape), shape, &BINDS[..4], 47);
    }

    /// The event log's failures reach `prepare` typed; nothing else is
    /// collected or published in their place.
    #[test]
    fn prepare_propagates_event_log_failures() {
        let shape = FixtureShape { log_t: 3, ram_k: 8 };
        let resident = ram_fixture(
            shape,
            Vec::new(),
            vec![RamOp::Write { word: 3, post: 4 }, RamOp::Read { word: 3 }],
        );
        let prepare = |witness: &TraceBackend, log_t: usize, session: &mut ProofSession| {
            let relation = RamValCheck::<Fr>::new(
                TraceDimensions::new(log_t),
                shape.log_k(),
                RamValCheckInit::full(Fr::from_u64(0)),
            );
            let claims = RamValCheckInputClaims {
                ram_val: Fr::from_u64(0),
                ram_val_final: Fr::from_u64(0),
                untrusted_advice: None,
                trusted_advice: None,
                program_image: None,
            };
            let points = RamValCheckInputClaims::<Vec<Fr>> {
                ram_val: vec![Fr::from_u64(3); shape.log_k() + log_t],
                ram_val_final: vec![Fr::from_u64(3); shape.log_k()],
                untrusted_advice: None,
                trusted_advice: None,
                program_image: None,
            };
            let challenges = RamValCheckChallenges {
                gamma: Fr::from_u64(5),
            };
            let inputs = ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };
            PrepareKernel::<Fr, _>::prepare(&OptimizedBackend, session, witness, inputs).map(|_| ())
        };

        // A configured domain shorter than the trace leaves no valid handle.
        let mut shortened = replayed(&resident, 2, None);
        shortened.config.log_t = 1;
        let mut session = ProofSession::default();
        assert!(matches!(
            prepare(&shortened, 1, &mut session),
            Err(KernelError::Witness(WitnessError::UnavailableView { .. }))
        ));
        assert!(session.state::<RamEventLog>().is_none());
        assert!(session.state::<SharedRamAddresses>().is_none());

        let mut session = ProofSession::default();
        assert!(matches!(
            prepare(&replayed(&resident, 2, Some(1)), shape.log_t, &mut session),
            Err(KernelError::Witness(WitnessError::Replay { .. }))
        ));
        assert!(session.state::<RamEventLog>().is_none());
        assert!(session.state::<SharedRamAddresses>().is_none());

        let mut session = ProofSession::default();
        let _ = RamEventLog::shared::<Fr>(&mut session, &resident, shape.log_t, 2 * shape.ram_k)
            .unwrap();
        assert!(matches!(
            prepare(&resident, shape.log_t, &mut session),
            Err(KernelError::TableSizeMismatch {
                expected: 8,
                got: 16,
                ..
            })
        ));
        assert!(session.state::<SharedRamAddresses>().is_none());
    }
}
