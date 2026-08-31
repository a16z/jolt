use std::mem;

use jolt_claims::protocols::jolt::{JoltDerivedId, SpartanShiftPublic};
use jolt_field::AkitaField;
use jolt_poly::{EqPlusOnePrefixSuffix, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputPoints,
};
use jolt_verifier::stages::stage3::spartan_shift::{SpartanShift, SpartanShiftOutputClaims};
use jolt_witness::JoltWitnessPlane;

use super::backend::MetalBackend;
use super::solinas::spartan_shift::{
    bind_dense_state, bind_prefix_tables, build_dense_state, dense_round, final_outputs,
    prefix_round, PendingSpartanShiftFold, PendingSpartanShiftPrefix, SpartanShiftDenseState,
    SpartanShiftGeometry, SpartanShiftKernelConfig, SpartanShiftPrefixTables,
    SpartanShiftResidentRows,
};
use super::solinas::SolinasMetal;
use super::spartan_dense::SpartanDenseResidentOwner;
use crate::optimized::spartan_shift::OptimizedSpartanShift;
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

#[cfg(feature = "test-utils")]
mod evaluation;
#[cfg(feature = "test-utils")]
pub use evaluation::{
    SpartanShiftCpuMetalEvalFixture, SpartanShiftEvalError, SpartanShiftEvalResult,
    SpartanShiftEvalSample, SpartanShiftRoundTiming, SpartanShiftShapeSnapshot,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpartanShiftMetalConfig {
    pub trace_cutoff_elements: usize,
    pub dispatch: SpartanShiftKernelConfig,
}

impl Default for SpartanShiftMetalConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: 1 << 25,
            dispatch: SpartanShiftKernelConfig::default(),
        }
    }
}

impl PrepareKernel<AkitaField, SpartanShift<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, SpartanShift<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = SpartanShift<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let rounds = inputs.relation.rounds();
        let cycles = 1usize
            .checked_shl(rounds as u32)
            .ok_or(KernelError::InvariantViolation {
                reason: "Spartan shift trace domain overflows usize",
            })?;
        let r_outer = inputs.relation.product_uniskip_tau_low();
        let r_product = inputs.relation.product_remainder_opening_point();
        if rounds == 0 || r_outer.len() != rounds || r_product.len() != rounds {
            return Err(KernelError::InvariantViolation {
                reason: "Spartan shift relation has invalid geometry",
            });
        }

        let metal_config = self.config.spartan_shift;
        let owner_lease = if let Some(mut owner) = session.take::<SpartanDenseResidentOwner>() {
            let lease = owner.take_shift_lease();
            session.park(owner);
            lease
        } else {
            None
        };
        let (rows, resident_source) = if let Some(lease) = owner_lease {
            (
                lease
                    .into_rows(cycles, self.context.device_registry_id())
                    .map_err(metal_prepare_error)?,
                "spartan_dense_owner",
            )
        } else {
            return OptimizedSpartanShift.prepare(session, witness, inputs);
        };
        if cycles < metal_config.trace_cutoff_elements
            || rows.len() != cycles
            || rows.device_registry_id() != self.context.device_registry_id()
        {
            return OptimizedSpartanShift.prepare(session, witness, inputs);
        }
        let _resident_span = tracing::info_span!(
            "MetalSpartanShift::resident_rows_consume",
            cycles,
            source = resident_source,
            resident_bytes = rows.resident_bytes(),
        )
        .entered();

        let geometry = SpartanShiftGeometry::new(cycles).map_err(metal_prepare_error)?;
        let config = metal_config.dispatch;
        let invocation = match self.context.prepare_spartan_shift_prefix(
            &rows,
            r_outer,
            r_product,
            inputs.challenges.gamma,
            config,
        ) {
            Ok(invocation) => invocation,
            Err(error) => {
                tracing::warn!(
                    target: "jolt::metal",
                    error = %error,
                    "Spartan shift prefix preparation failed; using optimized CPU"
                );
                return OptimizedSpartanShift.prepare(session, witness, inputs);
            }
        };
        #[cfg(feature = "allocative")]
        let plan = invocation.plan();
        let pending = match invocation.submit() {
            Ok(pending) => pending,
            Err(error) => {
                tracing::warn!(
                    target: "jolt::metal",
                    error = %error,
                    "Spartan shift prefix submission failed; using optimized CPU"
                );
                return OptimizedSpartanShift.prepare(session, witness, inputs);
            }
        };
        #[cfg(any(test, feature = "test-utils"))]
        let _ = self
            .test_counters
            .spartan_shift_sequences
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        let outer = EqPlusOnePrefixSuffix::new(r_outer);
        let product = EqPlusOnePrefixSuffix::new(r_product);
        let p = [
            outer.prefix_0,
            outer.prefix_1,
            product.prefix_0,
            product.prefix_1,
        ];
        Ok(Box::new(MetalSpartanShiftKernel {
            context: self.context.clone(),
            rows: Some(rows),
            geometry,
            config,
            #[cfg(feature = "allocative")]
            plan,
            gamma: inputs.challenges.gamma,
            r_outer: r_outer.to_vec(),
            r_product: r_product.to_vec(),
            bound_challenges: Vec::with_capacity(rounds),
            cursor: RoundCursor::new(rounds, geometry.prefix_vars()),
            phase: MetalSpartanShiftPhase::PrefixPending { pending, p },
            source_retained: true,
        }))
    }
}

enum MetalSpartanShiftPhase {
    PrefixPending {
        pending: PendingSpartanShiftPrefix,
        p: [Vec<AkitaField>; 4],
    },
    Prefix(SpartanShiftPrefixTables<AkitaField>),
    FoldPending(PendingSpartanShiftFold),
    Dense(SpartanShiftDenseState<AkitaField>),
    Poisoned,
}

struct MetalSpartanShiftKernel {
    context: std::sync::Arc<SolinasMetal>,
    rows: Option<SpartanShiftResidentRows>,
    geometry: SpartanShiftGeometry,
    config: SpartanShiftKernelConfig,
    #[cfg(feature = "allocative")]
    plan: super::solinas::spartan_shift::SpartanShiftPlan,
    gamma: AkitaField,
    r_outer: Vec<AkitaField>,
    r_product: Vec<AkitaField>,
    bound_challenges: Vec<AkitaField>,
    cursor: RoundCursor,
    phase: MetalSpartanShiftPhase,
    source_retained: bool,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for MetalSpartanShiftKernel {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        use crate::backend::vec_heap_bytes;

        let mut visitor = visitor.enter_self_sized::<Self>();
        for (name, values) in [
            ("r_outer", &self.r_outer),
            ("r_product", &self.r_product),
            ("bound_challenges", &self.bound_challenges),
        ] {
            visitor.visit_simple(allocative::Key::new(name), vec_heap_bytes(values));
        }
        if self.source_retained {
            visitor.visit_simple(
                allocative::Key::new("device_rows"),
                self.plan.storage.native_value_bytes + self.plan.storage.native_flag_bytes,
            );
        }
        let (host_phase, device_phase) = match &self.phase {
            MetalSpartanShiftPhase::PrefixPending { p, .. } => (
                p.iter().map(vec_heap_bytes).sum(),
                self.plan.storage.high_weight_bytes
                    + self.plan.storage.partial_bytes
                    + self.plan.storage.q_bytes,
            ),
            MetalSpartanShiftPhase::Prefix(tables) => (
                tables
                    .p
                    .iter()
                    .chain(tables.q.iter())
                    .map(vec_heap_bytes)
                    .sum(),
                0,
            ),
            MetalSpartanShiftPhase::FoldPending(_) => (
                0,
                self.plan.storage.low_weight_bytes + self.plan.storage.dense_output_bytes,
            ),
            MetalSpartanShiftPhase::Dense(state) => (
                [
                    &state.eq_plus_one_outer,
                    &state.eq_plus_one_product,
                    &state.unexpanded_pc,
                    &state.pc,
                    &state.is_virtual,
                    &state.is_first_in_sequence,
                    &state.is_noop,
                ]
                .into_iter()
                .map(vec_heap_bytes)
                .sum(),
                0,
            ),
            MetalSpartanShiftPhase::Poisoned => (0, 0),
        };
        visitor.visit_simple(allocative::Key::new("host_phase"), host_phase);
        visitor.visit_simple(allocative::Key::new("device_phase"), device_phase);
        visitor.exit();
    }
}

impl MetalSpartanShiftKernel {
    fn ensure_prefix_ready(&mut self) -> Result<(), SumcheckError<AkitaField>> {
        let phase = mem::replace(&mut self.phase, MetalSpartanShiftPhase::Poisoned);
        match phase {
            MetalSpartanShiftPhase::PrefixPending { pending, p } => {
                let span = tracing::info_span!(
                    "MetalSpartanShift::prefix",
                    gpu_active_ns = tracing::field::Empty,
                );
                let _entered = span.enter();
                let (_invocation, observation) = pending.join().map_err(metal_round_error)?;
                let _ = span.record("gpu_active_ns", duration_nanos(observation.gpu_active));
                self.phase = MetalSpartanShiftPhase::Prefix(SpartanShiftPrefixTables {
                    p,
                    q: observation.q,
                });
                Ok(())
            }
            MetalSpartanShiftPhase::Prefix(tables) => {
                self.phase = MetalSpartanShiftPhase::Prefix(tables);
                Ok(())
            }
            _ => Err(round_state_error(
                "Spartan shift prefix was requested after its phase ended",
            )),
        }
    }

    fn transition_to_fold(&mut self) -> Result<(), SumcheckError<AkitaField>> {
        let phase = mem::replace(&mut self.phase, MetalSpartanShiftPhase::Poisoned);
        if !matches!(phase, MetalSpartanShiftPhase::Prefix(_)) {
            return Err(round_state_error(
                "Spartan shift midpoint transition requires prefix tables",
            ));
        }
        let rows = self.rows.as_ref().ok_or_else(|| {
            round_state_error("Spartan shift resident rows disappeared before the midpoint fold")
        })?;
        let invocation = self
            .context
            .prepare_spartan_shift_fold(rows, &self.bound_challenges, self.config)
            .map_err(metal_round_error)?;
        let pending = invocation.submit().map_err(metal_round_error)?;
        self.rows = None;
        self.phase = MetalSpartanShiftPhase::FoldPending(pending);
        Ok(())
    }

    fn ensure_dense_ready(&mut self) -> Result<(), SumcheckError<AkitaField>> {
        let phase = mem::replace(&mut self.phase, MetalSpartanShiftPhase::Poisoned);
        match phase {
            MetalSpartanShiftPhase::FoldPending(pending) => {
                let span = tracing::info_span!(
                    "MetalSpartanShift::fold",
                    gpu_active_ns = tracing::field::Empty,
                );
                let _entered = span.enter();
                let (_invocation, observation) = pending.join().map_err(metal_round_error)?;
                let _ = span.record("gpu_active_ns", duration_nanos(observation.gpu_active));
                let state = build_dense_state(
                    self.geometry,
                    observation.outputs,
                    &self.r_outer,
                    &self.r_product,
                    &self.bound_challenges,
                )
                .map_err(metal_round_error)?;
                self.source_retained = false;
                self.phase = MetalSpartanShiftPhase::Dense(state);
                Ok(())
            }
            MetalSpartanShiftPhase::Dense(state) => {
                self.phase = MetalSpartanShiftPhase::Dense(state);
                Ok(())
            }
            _ => Err(round_state_error(
                "Spartan shift dense phase was requested before the midpoint fold",
            )),
        }
    }

    fn apply_bind(
        &mut self,
        action: BindAction,
        challenge: AkitaField,
    ) -> Result<(), SumcheckError<AkitaField>> {
        match action {
            BindAction::None => Ok(()),
            BindAction::Prefix => {
                self.ensure_prefix_ready()?;
                let MetalSpartanShiftPhase::Prefix(tables) = &mut self.phase else {
                    return Err(round_state_error(
                        "Spartan shift prefix bind has no prefix tables",
                    ));
                };
                bind_prefix_tables(tables, challenge).map_err(metal_round_error)
            }
            BindAction::Transition => self.transition_to_fold(),
            BindAction::Dense => {
                self.ensure_dense_ready()?;
                let MetalSpartanShiftPhase::Dense(state) = &mut self.phase else {
                    return Err(round_state_error(
                        "Spartan shift dense bind has no dense tables",
                    ));
                };
                bind_dense_state(state, challenge).map_err(metal_round_error)
            }
        }
    }

    fn prove_prefix(
        &mut self,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        self.ensure_prefix_ready()?;
        let MetalSpartanShiftPhase::Prefix(tables) = &self.phase else {
            return Err(round_state_error(
                "Spartan shift prefix round has no prefix tables",
            ));
        };
        prefix_round(previous_claim, tables).map_err(metal_round_error)
    }

    fn prove_dense(
        &mut self,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        self.ensure_dense_ready()?;
        let MetalSpartanShiftPhase::Dense(state) = &self.phase else {
            return Err(round_state_error(
                "Spartan shift dense round has no dense tables",
            ));
        };
        dense_round(previous_claim, state, self.gamma).map_err(metal_round_error)
    }

    fn dense_state(
        &self,
    ) -> Result<&SpartanShiftDenseState<AkitaField>, SumcheckKernelError<AkitaField>> {
        if !self.cursor.finished() {
            return Err(SumcheckKernelError::NotFullyBound {
                remaining: self.cursor.rounds() - self.bound_challenges.len(),
            });
        }
        let MetalSpartanShiftPhase::Dense(state) = &self.phase else {
            return Err(SumcheckKernelError::InvariantViolation {
                reason: "Spartan shift finished without dense output tables",
            });
        };
        Ok(state)
    }
}

impl ProveRounds<AkitaField> for MetalSpartanShiftKernel {
    fn num_rounds(&self) -> usize {
        self.cursor.rounds()
    }

    fn prove_round(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        let step = self
            .cursor
            .start_round(round, bind.is_some())
            .map_err(round_state_error)?;
        if let Some(challenge) = bind {
            self.bound_challenges.push(challenge);
            self.apply_bind(step.bind, challenge)?;
        }
        match step.prove {
            ProvePhase::Prefix => self.prove_prefix(previous_claim),
            ProvePhase::Dense => self.prove_dense(previous_claim),
        }
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        let action = self.cursor.finish().map_err(round_state_error)?;
        self.bound_challenges.push(bind);
        self.apply_bind(action, bind)?;
        if action == BindAction::Transition {
            self.ensure_dense_ready()?;
        }
        if self.bound_challenges.len() != self.cursor.rounds() {
            return Err(round_state_error(
                "Spartan shift bound challenge count differs from its round count",
            ));
        }
        Ok(())
    }
}

impl SumcheckKernel<AkitaField> for MetalSpartanShiftKernel {
    type Relation = SpartanShift<AkitaField>;

    fn output_claims(
        &mut self,
        _inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<SpartanShiftOutputClaims<AkitaField>, SumcheckKernelError<AkitaField>> {
        let outputs = final_outputs(self.dense_state()?).map_err(metal_output_error)?;
        Ok(SpartanShiftOutputClaims {
            unexpanded_pc: outputs.unexpanded_pc,
            pc: outputs.pc,
            is_virtual: outputs.is_virtual,
            is_first_in_sequence: outputs.is_first_in_sequence,
            is_noop: outputs.is_noop,
        })
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<AkitaField, Self::Relation>,
        output_points: &SumcheckOutputPoints<AkitaField, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<AkitaField, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<AkitaField>> {
        let state = self.dense_state()?;
        for (public, got) in [
            (
                SpartanShiftPublic::EqPlusOneOuter,
                state.eq_plus_one_outer[0],
            ),
            (
                SpartanShiftPublic::EqPlusOneProduct,
                state.eq_plus_one_product[0],
            ),
        ] {
            let id = JoltDerivedId::from(public);
            let expected =
                relation.derive_output_term(&id, input_points, output_points, challenges)?;
            if got != expected {
                return Err(SumcheckKernelError::DerivedTableDrift { id, expected, got });
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BindAction {
    None,
    Prefix,
    Transition,
    Dense,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum ProvePhase {
    Prefix,
    Dense,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RoundStep {
    bind: BindAction,
    prove: ProvePhase,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct RoundCursor {
    rounds: usize,
    prefix_rounds: usize,
    next_round: usize,
    finished: bool,
}

impl RoundCursor {
    const fn new(rounds: usize, prefix_rounds: usize) -> Self {
        Self {
            rounds,
            prefix_rounds,
            next_round: 0,
            finished: false,
        }
    }

    const fn rounds(self) -> usize {
        self.rounds
    }

    const fn finished(self) -> bool {
        self.finished
    }

    fn start_round(&mut self, round: usize, has_bind: bool) -> Result<RoundStep, &'static str> {
        if self.finished || round != self.next_round || round >= self.rounds {
            return Err("Spartan shift round calls are out of order");
        }
        if has_bind != (round != 0) {
            return Err("Spartan shift round has the wrong bind argument");
        }
        let bind = if round == 0 {
            BindAction::None
        } else if round < self.prefix_rounds {
            BindAction::Prefix
        } else if round == self.prefix_rounds {
            BindAction::Transition
        } else {
            BindAction::Dense
        };
        let prove = if round < self.prefix_rounds {
            ProvePhase::Prefix
        } else {
            ProvePhase::Dense
        };
        self.next_round += 1;
        Ok(RoundStep { bind, prove })
    }

    fn finish(&mut self) -> Result<BindAction, &'static str> {
        if self.finished || self.next_round != self.rounds {
            return Err("Spartan shift cannot finish before every round polynomial");
        }
        self.finished = true;
        if self.prefix_rounds == self.rounds {
            Ok(BindAction::Transition)
        } else {
            Ok(BindAction::Dense)
        }
    }
}

fn metal_prepare_error(error: impl ToString) -> KernelError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: error.to_string(),
    }
    .into()
}

fn metal_round_error(error: impl ToString) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: error.to_string(),
    }
}

fn metal_output_error(error: impl ToString) -> SumcheckKernelError<AkitaField> {
    SumcheckKernelError::ComputeBackend {
        backend: "metal",
        message: error.to_string(),
    }
}

fn round_state_error(reason: &'static str) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: reason.to_owned(),
    }
}

fn duration_nanos(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use super::{BindAction, ProvePhase, RoundCursor, RoundStep};

    #[test]
    fn even_geometry_transitions_on_the_first_suffix_round() {
        let mut cursor = RoundCursor::new(26, 13);
        assert_eq!(
            cursor.start_round(0, false),
            Ok(RoundStep {
                bind: BindAction::None,
                prove: ProvePhase::Prefix,
            })
        );
        for round in 1..13 {
            assert_eq!(
                cursor.start_round(round, true),
                Ok(RoundStep {
                    bind: BindAction::Prefix,
                    prove: ProvePhase::Prefix,
                })
            );
        }
        assert_eq!(
            cursor.start_round(13, true),
            Ok(RoundStep {
                bind: BindAction::Transition,
                prove: ProvePhase::Dense,
            })
        );
        for round in 14..26 {
            assert_eq!(
                cursor.start_round(round, true),
                Ok(RoundStep {
                    bind: BindAction::Dense,
                    prove: ProvePhase::Dense,
                })
            );
        }
        assert_eq!(cursor.finish(), Ok(BindAction::Dense));
        assert!(cursor.finished());
    }

    #[test]
    fn odd_geometry_keeps_the_larger_half_in_prefix() {
        let mut cursor = RoundCursor::new(25, 13);
        for round in 0..13 {
            let step = cursor.start_round(round, round != 0).unwrap();
            assert_eq!(step.prove, ProvePhase::Prefix);
        }
        assert_eq!(
            cursor.start_round(13, true),
            Ok(RoundStep {
                bind: BindAction::Transition,
                prove: ProvePhase::Dense,
            })
        );
    }

    #[test]
    fn one_round_geometry_joins_the_fold_during_finish() {
        let mut cursor = RoundCursor::new(1, 1);
        assert_eq!(
            cursor.start_round(0, false),
            Ok(RoundStep {
                bind: BindAction::None,
                prove: ProvePhase::Prefix,
            })
        );
        assert_eq!(cursor.finish(), Ok(BindAction::Transition));
    }

    #[test]
    fn cursor_rejects_missing_binds_reordering_and_early_finish() {
        let mut cursor = RoundCursor::new(4, 2);
        assert!(cursor.finish().is_err());
        assert!(cursor.start_round(1, true).is_err());
        assert!(cursor.start_round(0, true).is_err());
        assert!(cursor.start_round(0, false).is_ok());
        assert!(cursor.start_round(1, false).is_err());
        assert!(cursor.start_round(1, true).is_ok());
        assert!(cursor.start_round(1, true).is_err());
    }
}
