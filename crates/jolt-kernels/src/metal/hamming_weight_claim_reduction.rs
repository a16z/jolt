use std::mem::size_of;

use jolt_field::Prime128OffsetA7F7 as AkitaField;
use jolt_sumcheck::SumcheckError;
use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::HammingWeightClaimReduction;
use jolt_witness::JoltWitnessPlane;

use super::backend::MetalBackend;
use super::booleanity::booleanity_address_can_fallback;
use super::solinas::{
    BooleanityAddressPushforwardConfig, BooleanityRows, BOOLEANITY_SOURCE_ROW_BYTES,
};
use crate::optimized::hamming_weight_claim_reduction::{
    HammingWeightPreparePlan, OptimizedHammingWeightClaimReduction,
};
use crate::{KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct HammingWeightMetalConfig {
    pub trace_cutoff_elements: usize,
    pub dispatch: BooleanityAddressPushforwardConfig,
}

impl Default for HammingWeightMetalConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: 1 << 18,
            dispatch: BooleanityAddressPushforwardConfig::default(),
        }
    }
}

impl HammingWeightMetalConfig {
    pub(super) fn admits(self, trace_elements: usize, log_t: usize, log_k_chunk: usize) -> bool {
        trace_elements >= self.trace_cutoff_elements
            && log_k_chunk == 8
            && self.dispatch.inner_log2 <= log_t
            && self.dispatch.inner_log2 <= 16
            && (1..=6).contains(&self.dispatch.selectors_per_tile)
            && self
                .dispatch
                .tile_threads_per_threadgroup
                .is_none_or(|threads| threads > 0 && threads.is_multiple_of(32))
            && self
                .dispatch
                .finalize_threads_per_threadgroup
                .is_none_or(|threads| matches!(threads, 256 | 512 | 768 | 1024))
    }
}

impl PrepareKernel<AkitaField, HammingWeightClaimReduction<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, HammingWeightClaimReduction<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = HammingWeightClaimReduction<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let cpu_inputs = || ProverInputs {
            relation: inputs.relation,
            claims: inputs.claims,
            points: inputs.points,
            challenges: inputs.challenges,
        };
        let cpu = |session: &mut ProofSession| {
            let _ = session.take::<BooleanityRows>();
            OptimizedHammingWeightClaimReduction.prepare(session, witness, cpu_inputs())
        };
        let dimensions = inputs.relation.dimensions();
        let log_t = inputs.relation.r_cycle().len();
        let trace_elements = 1usize << log_t;
        let config = self.config.hamming_weight_claim_reduction;
        if !config.admits(trace_elements, log_t, dimensions.log_k_chunk) {
            return cpu(session);
        }

        let plan = match HammingWeightPreparePlan::new(inputs.relation, inputs.challenges) {
            Ok(plan) => plan,
            Err(error) => {
                let _ = session.take::<BooleanityRows>();
                return Err(error);
            }
        };
        let selectors = plan.metal_selectors();
        let resident_rows = match session.state::<BooleanityRows>().cloned() {
            Some(rows)
                if rows.len() == trace_elements
                    && self.context.validate_booleanity_rows(&rows).is_ok() =>
            {
                rows
            }
            _ => return cpu(session),
        };
        let resident_row_identity = resident_rows.allocation_identity();
        let resident_row_bytes = BOOLEANITY_SOURCE_ROW_BYTES;
        let e_in_elements = 1usize << config.dispatch.inner_log2;
        let e_out_elements = trace_elements / e_in_elements;
        let selector_bytes = selectors.len() * size_of::<[u32; 2]>();
        let e_in_bytes = e_in_elements * size_of::<AkitaField>();
        let e_out_bytes = e_out_elements * size_of::<AkitaField>();
        let partial_bytes =
            e_out_elements * config.dispatch.selectors_per_tile * 256 * size_of::<AkitaField>();
        let output_bytes = selectors.len() * 256 * size_of::<AkitaField>();
        let planned_device_bytes =
            selector_bytes + e_in_bytes + e_out_bytes + partial_bytes + output_bytes;
        let device = self.context.device_info();
        let requested_tile_threads = config.dispatch.tile_threads_per_threadgroup.unwrap_or(0);
        let requested_finalize_threads = config
            .dispatch
            .finalize_threads_per_threadgroup
            .unwrap_or(0);

        let prepare_guard =
            tracing::info_span!("MetalHammingWeightClaimReduction::prepare").entered();
        let sequence_span = tracing::info_span!(
            "MetalHammingWeightClaimReduction::sequence_prepare",
            resident_rows_storage_id = resident_row_identity,
            resident_rows = trace_elements,
            resident_row_bytes,
            row_upload_bytes = 0u64,
            polys = selectors.len(),
            k = 256usize,
            e_in_elements,
            e_out_elements,
            requested_inner_log2 = config.dispatch.inner_log2,
            effective_inner_log2 = config.dispatch.inner_log2,
            requested_selectors_per_tile = config.dispatch.selectors_per_tile,
            effective_selectors_per_tile = tracing::field::Empty,
            requested_tile_threads,
            effective_tile_threads = tracing::field::Empty,
            requested_finalize_threads,
            effective_finalize_threads = tracing::field::Empty,
            selector_tiles = tracing::field::Empty,
            production_specialized = tracing::field::Empty,
        );
        let sequence_guard = sequence_span.enter();
        let allocation_span = tracing::info_span!(
            "MetalHammingWeightClaimReduction::allocation_plan",
            device_buffers = 5u64,
            planned_device_bytes,
            current_device_bytes = device.current_allocated_size,
            recommended_device_bytes = device.recommended_max_working_set_size,
        );
        let allocation_guard = allocation_span.enter();
        let invocation = match self.context.prepare_booleanity_address_pushforward(
            resident_rows,
            &selectors,
            plan.reference_cycle(),
            config.dispatch,
        ) {
            Ok(invocation) => invocation,
            Err(error) if booleanity_address_can_fallback(&error) => {
                tracing::warn!(
                    error = %error,
                    "Hamming-weight Metal preparation unavailable; using the optimized CPU kernel"
                );
                drop(allocation_guard);
                drop(sequence_guard);
                drop(prepare_guard);
                return cpu(session);
            }
            Err(error) => {
                let _ = session.take::<BooleanityRows>();
                return Err(metal_error(error.to_string()).into());
            }
        };
        drop(allocation_guard);
        let _ = sequence_span.record(
            "effective_selectors_per_tile",
            invocation.selectors_per_tile(),
        );
        let _ = sequence_span.record(
            "effective_tile_threads",
            invocation.tile_threads_per_threadgroup(),
        );
        let _ = sequence_span.record(
            "effective_finalize_threads",
            invocation.finalize_threads_per_threadgroup(),
        );
        let _ = sequence_span.record("selector_tiles", invocation.selector_tiles());
        let _ = sequence_span.record(
            "production_specialized",
            invocation.uses_production_specialization(),
        );
        drop(sequence_guard);

        let consumed_rows =
            session
                .take::<BooleanityRows>()
                .ok_or(KernelError::InvariantViolation {
                    reason: "Hamming-weight Metal preparation lost its resident row owner",
                })?;
        if consumed_rows.allocation_identity() != resident_row_identity {
            return Err(KernelError::InvariantViolation {
                reason: "Hamming-weight Metal preparation changed resident row allocations",
            });
        }
        let terminal_carry_removed = session.state::<BooleanityRows>().is_none();
        if !terminal_carry_removed {
            return Err(KernelError::InvariantViolation {
                reason: "Hamming-weight Metal preparation left a resident row owner",
            });
        }
        let lifecycle_guard = tracing::info_span!(
            "MetalBooleanityRows::stage7_hamming_use",
            resident_rows_storage_id = resident_row_identity,
            resident_rows = consumed_rows.len(),
            resident_row_bytes,
            device_registry_id = consumed_rows.device_registry_id(),
            row_allocations = 0u64,
            row_upload_bytes = 0u64,
            terminal_consumer = true,
            terminal_carry_removed,
        )
        .entered();

        let dispatch_span = tracing::info_span!(
            "MetalHammingWeightClaimReduction::dispatch",
            command_buffers = 1u64,
            tile_dispatches = invocation.selector_tiles(),
            finalize_dispatches = invocation.selector_tiles(),
            command_completed = tracing::field::Empty,
            gpu_active_ns = tracing::field::Empty,
            resident_rows_storage_id = resident_row_identity,
        );
        let dispatch_guard = dispatch_span.enter();
        let gpu_active = invocation
            .execute_timed()
            .map_err(|error| metal_error(error.to_string()))?;
        let gpu_active_ns = u64::try_from(gpu_active.as_nanos()).unwrap_or(u64::MAX);
        let _ = dispatch_span.record("command_completed", true);
        let _ = dispatch_span.record("gpu_active_ns", gpu_active_ns);
        drop(dispatch_guard);

        let readback_span = tracing::info_span!(
            "MetalHammingWeightClaimReduction::readback",
            elements = invocation.output_elements(),
            bytes = invocation.output_elements() * size_of::<AkitaField>(),
            readbacks = 1u64,
        );
        let readback_guard = readback_span.enter();
        let masses = invocation
            .read_masses()
            .map_err(|error| metal_error(error.to_string()))?;
        drop(readback_guard);
        drop(lifecycle_guard);
        drop(consumed_rows);
        let kernel = plan.finish_flat(masses)?;
        #[cfg(any(test, feature = "test-utils"))]
        let _ = self
            .test_counters
            .hamming_dispatches
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        Ok(kernel)
    }
}

fn metal_error(message: impl Into<String>) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: message.into(),
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "Metal parity test setup")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::ra::JoltRaPolynomialLayout;
    use jolt_field::Prime128OffsetA7F7 as AkitaField;
    use jolt_field::{Ring as _, Zero as _};
    use jolt_verifier::stages::stage7::hamming_weight_claim_reduction::{
        HammingWeightClaimReduction, HammingWeightClaimReductionChallenges,
        HammingWeightClaimReductionDimensions, HammingWeightClaimReductionInputClaims,
    };

    use super::*;
    use crate::metal::solinas::{BooleanityRows, BooleanitySelector};
    use crate::optimized::booleanity::testing::with_booleanity_backend;
    use crate::optimized::instruction_read_raf::{
        collect_instruction_cycle_rows, InstructionCycleRow,
    };
    use crate::optimized::OptimizedHammingWeightClaimReduction;
    use crate::{PrepareKernel, ProofSession, ProverInputs, SumcheckKernel};

    fn point(seed: u64, len: usize) -> Vec<AkitaField> {
        (0..len as u64)
            .map(|index| AkitaField::from_u64(seed + 37 * index + 5))
            .collect()
    }

    fn metal_config(
        trace_cutoff_elements: usize,
        tile_threads_per_threadgroup: Option<usize>,
    ) -> super::super::MetalConfig {
        super::super::MetalConfig {
            hamming_weight_claim_reduction: HammingWeightMetalConfig {
                trace_cutoff_elements,
                dispatch: BooleanityAddressPushforwardConfig {
                    inner_log2: 8,
                    selectors_per_tile: 6,
                    tile_threads_per_threadgroup,
                    finalize_threads_per_threadgroup: Some(256),
                },
            },
            ..Default::default()
        }
    }

    fn run_lockstep(
        expected: &mut dyn SumcheckKernel<
            AkitaField,
            Relation = HammingWeightClaimReduction<AkitaField>,
        >,
        actual: &mut dyn SumcheckKernel<
            AkitaField,
            Relation = HammingWeightClaimReduction<AkitaField>,
        >,
        claims: &HammingWeightClaimReductionInputClaims<AkitaField>,
    ) {
        let mut claim = AkitaField::zero();
        let mut bind = None;
        let mut round_challenges = Vec::new();
        for round in 0..expected.num_rounds() {
            let expected_poly = expected.prove_round(bind, round, claim).unwrap();
            let actual_poly = actual.prove_round(bind, round, claim).unwrap();
            assert_eq!(actual_poly, expected_poly, "round {round}");
            let challenge = AkitaField::from_u64(0x1234_5678 + 1000 * round as u64 + 7);
            claim = expected_poly.evaluate(challenge);
            round_challenges.push(challenge);
            bind = Some(challenge);
        }
        let final_bind = *round_challenges.last().unwrap();
        expected.finish_rounds(final_bind).unwrap();
        actual.finish_rounds(final_bind).unwrap();
        assert_eq!(
            actual.output_claims(claims).unwrap(),
            expected.output_claims(claims).unwrap()
        );
    }

    #[test]
    fn prepare_matches_optimized_cpu_and_consumes_resident_rows() {
        let log_t = 10;
        with_booleanity_backend(log_t, 8, |witness, base_dimensions| {
            let dimensions = HammingWeightClaimReductionDimensions::new(
                base_dimensions.layout,
                base_dimensions.log_k_chunk,
            )
            .unwrap();
            let relation = HammingWeightClaimReduction::new(
                dimensions,
                point(300, log_t),
                point(500, dimensions.log_k_chunk),
                (0..dimensions.layout.total())
                    .map(|index| point(700 + index as u64, dimensions.log_k_chunk))
                    .collect(),
            );
            let challenges = HammingWeightClaimReductionChallenges {
                gamma: AkitaField::from_u64(23),
            };
            let claims = HammingWeightClaimReductionInputClaims::<AkitaField>::default();
            let points = HammingWeightClaimReductionInputClaims::<Vec<AkitaField>>::default();
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };

            let mut expected = OptimizedHammingWeightClaimReduction
                .prepare(&mut ProofSession::default(), witness, inputs())
                .unwrap();
            let metal = super::super::MetalBackend::new(metal_config(2, Some(256))).unwrap();
            let packed = collect_instruction_cycle_rows::<AkitaField>(witness, 1 << log_t).unwrap();
            let resident = metal
                .context
                .prepare_booleanity_rows(InstructionCycleRow::metal_booleanity_rows(&packed))
                .unwrap();
            let mut session = ProofSession::default();
            session.park(resident);

            let mut actual = metal.prepare(&mut session, witness, inputs()).unwrap();
            assert!(session.state::<BooleanityRows>().is_none());
            assert_eq!(metal.hamming_dispatches(), 1);

            run_lockstep(expected.as_mut(), actual.as_mut(), &claims);
        });
    }

    #[test]
    fn production_plan_uses_the_specialized_selector_schedule() {
        let layout = JoltRaPolynomialLayout::new(16, 2, 2).unwrap();
        let dimensions = HammingWeightClaimReductionDimensions::new(layout, 8).unwrap();
        let relation = HammingWeightClaimReduction::new(
            dimensions,
            point(300, 10),
            point(500, 8),
            (0..layout.total())
                .map(|index| point(700 + index as u64, 8))
                .collect(),
        );
        let plan = HammingWeightPreparePlan::new(
            &relation,
            &HammingWeightClaimReductionChallenges {
                gamma: AkitaField::from_u64(23),
            },
        )
        .unwrap();
        let expected = (0..16)
            .map(|index| BooleanitySelector::Lookup {
                shift: (120 - 8 * index) as u32,
            })
            .chain((0..2).map(|index| BooleanitySelector::Bytecode {
                shift: (8 - 8 * index) as u32,
            }))
            .chain((0..2).map(|index| BooleanitySelector::Ram {
                shift: (8 - 8 * index) as u32,
            }))
            .chain((0..8).map(|index| BooleanitySelector::FusedInc {
                shift: (8 * index) as u32,
            }))
            .chain(std::iter::once(BooleanitySelector::FusedIncMsb))
            .collect::<Vec<_>>();
        assert_eq!(plan.metal_selectors(), expected);
    }

    #[test]
    fn cpu_fallbacks_match_and_consume_resident_rows() {
        let log_t = 10;
        with_booleanity_backend(log_t, 8, |witness, base_dimensions| {
            let dimensions = HammingWeightClaimReductionDimensions::new(
                base_dimensions.layout,
                base_dimensions.log_k_chunk,
            )
            .unwrap();
            let relation = HammingWeightClaimReduction::new(
                dimensions,
                point(300, log_t),
                point(500, dimensions.log_k_chunk),
                (0..dimensions.layout.total())
                    .map(|index| point(700 + index as u64, dimensions.log_k_chunk))
                    .collect(),
            );
            let challenges = HammingWeightClaimReductionChallenges {
                gamma: AkitaField::from_u64(23),
            };
            let claims = HammingWeightClaimReductionInputClaims::<AkitaField>::default();
            let points = HammingWeightClaimReductionInputClaims::<Vec<AkitaField>>::default();
            let inputs = || ProverInputs {
                relation: &relation,
                claims: &claims,
                points: &points,
                challenges: &challenges,
            };
            let packed = collect_instruction_cycle_rows::<AkitaField>(witness, 1 << log_t).unwrap();

            let mut invalid_finalize = metal_config(2, Some(256));
            invalid_finalize
                .hamming_weight_claim_reduction
                .dispatch
                .finalize_threads_per_threadgroup = Some(128);
            let cases = [
                (metal_config(2, Some(256)), None),
                (metal_config(1 << 12, Some(256)), Some(1 << log_t)),
                (metal_config(2, Some(256)), Some(1 << (log_t - 1))),
                (metal_config(2, Some(usize::MAX)), Some(1 << log_t)),
                (invalid_finalize, Some(1 << log_t)),
            ];
            for (config, resident_len) in cases {
                let metal = super::super::MetalBackend::new(config).unwrap();
                let mut session = ProofSession::default();
                if let Some(resident_len) = resident_len {
                    let resident = metal
                        .context
                        .prepare_booleanity_rows(InstructionCycleRow::metal_booleanity_rows(
                            &packed[..resident_len],
                        ))
                        .unwrap();
                    session.park(resident);
                }
                let mut expected = OptimizedHammingWeightClaimReduction
                    .prepare(&mut ProofSession::default(), witness, inputs())
                    .unwrap();
                let mut actual = metal.prepare(&mut session, witness, inputs()).unwrap();
                assert_eq!(metal.hamming_dispatches(), 0);
                assert!(session.state::<BooleanityRows>().is_none());
                run_lockstep(expected.as_mut(), actual.as_mut(), &claims);
            }
        });
    }
}
