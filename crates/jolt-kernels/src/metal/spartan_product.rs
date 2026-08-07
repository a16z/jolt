use std::{collections::BTreeMap, time::Instant};

use jolt_claims::protocols::jolt::geometry::dimensions::PRODUCT_UNISKIP_DOMAIN_SIZE;
use jolt_claims::protocols::jolt::geometry::spartan::{
    branch_flag_product, jump_flag_product, left_instruction_input_product, lookup_output_product,
    next_is_noop_product, right_instruction_input_product, virtual_instruction_product,
    write_lookup_output_to_rd_product,
};
use jolt_claims::protocols::jolt::{
    JoltDerivedId, JoltOpeningId, SpartanProductVirtualizationPublic,
};
use jolt_claims::{InputClaims as _, OutputClaims as _};
use jolt_field::AkitaField;
use jolt_poly::lagrange::{centered_lagrange_evals, centered_lagrange_kernel};
use jolt_poly::{BindingOrder, EqPolynomial, GruenSplitEqPolynomial, UnivariatePoly};
use jolt_sumcheck::{ProveRounds, SumcheckError};
use jolt_verifier::stages::relations::{
    ConcreteSumcheck as _, ConcreteSumcheckChallenges, SumcheckInputClaims, SumcheckInputPoints,
    SumcheckOutputClaims, SumcheckOutputPoints,
};
use jolt_verifier::stages::stage2::product_remainder::ProductRemainder;
use jolt_verifier::VerifierError;
use jolt_witness::{collect_bundles, JoltWitnessPlane};

use super::backend::MetalBackend;
use super::solinas::{
    MetalError, ProductRemainderRow, ProductRemainderRows, ProductRemainderSequence,
    ProductRemainderSequenceConfig,
};
use crate::optimized::spartan_product::{
    discard_product_uniskip_carry, product_uniskip_carry_metadata, OptimizedProductRemainder,
    SpartanProductRow,
};
use crate::{
    KernelError, PrepareKernel, ProofSession, ProverInputs, SumcheckKernel, SumcheckKernelError,
};

const DOMAIN: usize = PRODUCT_UNISKIP_DOMAIN_SIZE;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct SpartanProductRemainderMetalConfig {
    pub trace_cutoff_elements: usize,
    pub dispatch: ProductRemainderSequenceConfig,
}

impl Default for SpartanProductRemainderMetalConfig {
    fn default() -> Self {
        Self {
            trace_cutoff_elements: 1 << 18,
            dispatch: ProductRemainderSequenceConfig::default(),
        }
    }
}

impl MetalBackend {
    pub(super) fn prepare_product_remainder_witness(
        &self,
        session: &mut ProofSession,
        log_t: usize,
        witness: &dyn JoltWitnessPlane<AkitaField>,
    ) -> Result<(), KernelError<AkitaField>> {
        let cycles = 1usize
            .checked_shl(log_t as u32)
            .ok_or(KernelError::InvariantViolation {
                reason: "Spartan product trace length overflows usize",
            })?;
        if cycles < self.config.spartan_product_remainder.trace_cutoff_elements {
            return Ok(());
        }

        let span = tracing::info_span!(
            "MetalProductRemainder::witness_prepare",
            cycles,
            row_bytes = cycles.saturating_mul(std::mem::size_of::<ProductRemainderRow>()),
            collect_wall_ns = tracing::field::Empty,
            upload_wall_ns = tracing::field::Empty,
            resident_rows_storage_id = tracing::field::Empty,
            admitted = tracing::field::Empty,
            fallback_reason = tracing::field::Empty,
        );
        let _entered = span.enter();
        let started = Instant::now();
        let rows: Vec<SpartanProductRow> = collect_bundles(witness, cycles)?;
        let packed = rows
            .iter()
            .map(ProductRemainderRow::from)
            .collect::<Vec<_>>();
        drop(rows);
        let _ = span.record("collect_wall_ns", duration_nanos(started.elapsed()));

        let started = Instant::now();
        match self.context.prepare_product_remainder_rows(&packed) {
            Ok(rows) => {
                let _ = span.record("upload_wall_ns", duration_nanos(started.elapsed()));
                let _ = span.record(
                    "resident_rows_storage_id",
                    rows.allocation_identity() as u64,
                );
                let _ = span.record("admitted", true);
                let _ = span.record("fallback_reason", "none");
                session.park(rows);
                Ok(())
            }
            Err(error) if error.is_capacity_error() => {
                let _ = span.record("upload_wall_ns", duration_nanos(started.elapsed()));
                let _ = span.record("admitted", false);
                let _ = span.record("fallback_reason", "capacity");
                tracing::warn!(
                    target: "jolt::metal",
                    error = %error,
                    "product-remainder resident rows were not admitted; using optimized CPU"
                );
                Ok(())
            }
            Err(error) => Err(metal_prepare_error(error)),
        }
    }
}

impl PrepareKernel<AkitaField, ProductRemainder<AkitaField>> for MetalBackend {
    fn prepare(
        &self,
        session: &mut ProofSession,
        witness: &dyn JoltWitnessPlane<AkitaField>,
        inputs: ProverInputs<'_, AkitaField, ProductRemainder<AkitaField>>,
    ) -> Result<
        Box<dyn SumcheckKernel<AkitaField, Relation = ProductRemainder<AkitaField>>>,
        KernelError<AkitaField>,
    > {
        let rounds = inputs.relation.rounds();
        let cycles = 1usize
            .checked_shl(rounds as u32)
            .ok_or(KernelError::InvariantViolation {
                reason: "Spartan product trace length overflows usize",
            })?;
        let use_metal = cycles >= self.config.spartan_product_remainder.trace_cutoff_elements
            && session.state::<ProductRemainderRows>().is_some();
        if !use_metal {
            drop(session.take::<ProductRemainderRows>());
            return OptimizedProductRemainder.prepare(session, witness, inputs);
        }

        let (carry_log_t, tau_low) = product_uniskip_carry_metadata(session)?;
        if carry_log_t != rounds || tau_low.len() != rounds {
            return Err(KernelError::InvariantViolation {
                reason: "product uni-skip carry disagrees with the remainder relation",
            });
        }
        let host = MetalProductRemainderHost::new(
            &tau_low,
            inputs.relation.uniskip_challenge(),
            inputs.relation.tau_high(),
        )?;
        let rows = session.state::<ProductRemainderRows>().cloned().ok_or(
            KernelError::InvariantViolation {
                reason: "Metal product remainder lost its resident row owner",
            },
        )?;
        if rows.len() != cycles || rows.device_registry_id() != self.context.device_registry_id() {
            return Err(KernelError::InvariantViolation {
                reason: "Metal product-remainder rows have the wrong shape or device",
            });
        }
        let row_storage_id = rows.allocation_identity();
        let (e_in, e_out) = host.current_weights();
        let e_in_capacity = 1usize << (rounds / 2);
        let e_out_capacity = cycles / e_in_capacity;
        let prepare_span = tracing::info_span!(
            "MetalProductRemainder::prepare",
            cycles,
            rounds,
            resident_rows_storage_id = row_storage_id as u64,
            row_upload_bytes = 0u64,
            round_device_buffer_allocations = 0u64,
            sequence_prepare_wall_ns = tracing::field::Empty,
            materialize_wall_ns = tracing::field::Empty,
            materialize_gpu_active_ns = tracing::field::Empty,
        );
        let _entered = prepare_span.enter();
        let started = Instant::now();
        let sequence = self.context.prepare_product_remainder_sequence_with_rows(
            rows,
            host.lagrange_weights,
            e_in_capacity,
            e_out_capacity,
            self.config.spartan_product_remainder.dispatch,
        );
        let _ = prepare_span.record(
            "sequence_prepare_wall_ns",
            duration_nanos(started.elapsed()),
        );
        let mut sequence = match sequence {
            Ok(sequence) => sequence,
            Err(error) if product_prepare_fallback_reason(&error).is_some() => {
                tracing::warn!(
                    target: "jolt::metal",
                    error = %error,
                    "product-remainder Metal sequence preparation failed; using optimized CPU"
                );
                drop(session.take::<ProductRemainderRows>());
                return OptimizedProductRemainder.prepare(session, witness, inputs);
            }
            Err(error) => return Err(metal_prepare_error(error)),
        };
        let started = Instant::now();
        let first_message = sequence.message_timed(e_in, e_out);
        let materialize_wall = started.elapsed();
        let (first_message, materialize_gpu_active) = match first_message {
            Ok(result) => result,
            Err(error) if product_prepare_fallback_reason(&error).is_some() => {
                tracing::warn!(
                    target: "jolt::metal",
                    error = %error,
                    "product-remainder Metal materialization failed; using optimized CPU"
                );
                drop(session.take::<ProductRemainderRows>());
                return OptimizedProductRemainder.prepare(session, witness, inputs);
            }
            Err(error) => return Err(metal_prepare_error(error)),
        };
        let _ = prepare_span.record("materialize_wall_ns", duration_nanos(materialize_wall));
        let _ = prepare_span.record(
            "materialize_gpu_active_ns",
            duration_nanos(materialize_gpu_active),
        );
        if sequence.row_allocation_identity() != row_storage_id {
            return Err(KernelError::InvariantViolation {
                reason: "product-remainder sequence changed the resident row allocation",
            });
        }
        drop(session.take::<ProductRemainderRows>());
        discard_product_uniskip_carry(session, carry_log_t, &tau_low)?;
        #[cfg(any(test, feature = "test-utils"))]
        let _ = self
            .product_remainder_sequences
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        Ok(Box::new(MetalProductRemainderKernel {
            host,
            sequence,
            pending_endpoints: Some(first_message),
            row_storage_id,
        }))
    }
}

fn product_prepare_fallback_reason(error: &MetalError) -> Option<&'static str> {
    if error.is_capacity_error() {
        return Some("capacity");
    }
    match error {
        MetalError::CommandFailed(_) => Some("command_failed"),
        MetalError::GpuTimestampLookup { .. } => Some("gpu_timestamp_lookup"),
        MetalError::InvalidGpuTimestamps { .. } => Some("invalid_gpu_timestamps"),
        _ => None,
    }
}

fn metal_prepare_error(error: MetalError) -> KernelError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: error.to_string(),
    }
    .into()
}

fn metal_round_error(error: MetalError) -> SumcheckError<AkitaField> {
    SumcheckError::ComputeBackend {
        backend: "metal",
        message: error.to_string(),
    }
}

fn metal_output_error(error: MetalError) -> SumcheckKernelError<AkitaField> {
    SumcheckKernelError::ComputeBackend {
        backend: "metal",
        message: error.to_string(),
    }
}

fn duration_nanos(duration: std::time::Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

struct MetalProductRemainderHost {
    rounds: usize,
    split_eq: GruenSplitEqPolynomial<AkitaField>,
    challenges: Vec<AkitaField>,
    lagrange_weights: [AkitaField; DOMAIN],
}

impl MetalProductRemainderHost {
    fn new(
        tau_low: &[AkitaField],
        uniskip_challenge: AkitaField,
        tau_high: AkitaField,
    ) -> Result<Self, KernelError<AkitaField>> {
        let lagrange_weights = centered_lagrange_evals(DOMAIN, uniskip_challenge)?
            .try_into()
            .map_err(|_| KernelError::InvariantViolation {
                reason: "product-remainder Lagrange vector has the wrong length",
            })?;
        let scale = centered_lagrange_kernel(DOMAIN, tau_high, uniskip_challenge)?;
        Ok(Self {
            rounds: tau_low.len(),
            split_eq: GruenSplitEqPolynomial::new_with_scaling(
                tau_low,
                BindingOrder::LowToHigh,
                Some(scale),
            ),
            challenges: Vec::with_capacity(tau_low.len()),
            lagrange_weights,
        })
    }

    fn current_weights(&self) -> (&[AkitaField], &[AkitaField]) {
        (self.split_eq.e_in_current(), self.split_eq.e_out_current())
    }

    fn bind(&mut self, challenge: AkitaField) {
        self.split_eq.bind(challenge);
        self.challenges.push(challenge);
    }

    fn polynomial(
        &self,
        endpoints: [AkitaField; 2],
        previous_claim: AkitaField,
    ) -> UnivariatePoly<AkitaField> {
        self.split_eq
            .gruen_poly_deg_3(endpoints[0], endpoints[1], previous_claim)
    }

    fn opening_weights(&self) -> (Vec<AkitaField>, Vec<AkitaField>) {
        let point = self.challenges.iter().rev().copied().collect::<Vec<_>>();
        let split = point.len() / 2;
        let (out_point, in_point) = point.split_at(split);
        (
            EqPolynomial::evals(in_point, None),
            EqPolynomial::evals(out_point, None),
        )
    }
}

struct MetalProductRemainderKernel {
    host: MetalProductRemainderHost,
    sequence: ProductRemainderSequence,
    pending_endpoints: Option<[AkitaField; 2]>,
    row_storage_id: usize,
}

#[cfg(feature = "allocative")]
impl allocative::Allocative for MetalProductRemainderKernel {
    fn visit<'a, 'b: 'a>(&self, visitor: &'a mut allocative::Visitor<'b>) {
        let mut visitor = visitor.enter_self_sized::<Self>();
        visitor.visit_field(allocative::Key::new("sequence"), &self.sequence);
        visitor.exit();
    }
}

impl ProveRounds<AkitaField> for MetalProductRemainderKernel {
    fn num_rounds(&self) -> usize {
        self.host.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<AkitaField>,
        round: usize,
        previous_claim: AkitaField,
    ) -> Result<UnivariatePoly<AkitaField>, SumcheckError<AkitaField>> {
        let endpoints = if let Some(challenge) = bind {
            self.host.bind(challenge);
            if self.host.challenges.len() != round {
                return Err(SumcheckError::ComputeBackend {
                    backend: "metal",
                    message: "product-remainder round order drifted".to_string(),
                });
            }
            let source_elements = self.sequence.current_elements();
            let (e_in, e_out) = self.host.current_weights();
            let span = tracing::info_span!(
                "MetalProductRemainder::bind_and_message",
                round,
                source_elements,
                resident_rows_storage_id = self.row_storage_id as u64,
                dispatch_wall_ns = tracing::field::Empty,
                gpu_active_ns = tracing::field::Empty,
            );
            let _entered = span.enter();
            let started = Instant::now();
            let (message, gpu_active) = self
                .sequence
                .bind_and_message_timed(challenge, e_in, e_out)
                .map_err(metal_round_error)?;
            let _ = span.record("dispatch_wall_ns", duration_nanos(started.elapsed()));
            let _ = span.record("gpu_active_ns", duration_nanos(gpu_active));
            message
        } else {
            if round != 0 || !self.host.challenges.is_empty() {
                return Err(SumcheckError::ComputeBackend {
                    backend: "metal",
                    message: "product-remainder first message was requested out of order"
                        .to_string(),
                });
            }
            self.pending_endpoints
                .take()
                .ok_or_else(|| SumcheckError::ComputeBackend {
                    backend: "metal",
                    message: "product-remainder first message was already consumed".to_string(),
                })?
        };
        Ok(self.host.polynomial(endpoints, previous_claim))
    }

    fn finish_rounds(&mut self, bind: AkitaField) -> Result<(), SumcheckError<AkitaField>> {
        self.host.bind(bind);
        if self.host.challenges.len() != self.host.rounds || self.sequence.current_elements() != 2 {
            return Err(SumcheckError::ComputeBackend {
                backend: "metal",
                message: "product-remainder sequence did not reach its terminal state".to_string(),
            });
        }
        Ok(())
    }
}

impl SumcheckKernel<AkitaField> for MetalProductRemainderKernel {
    type Relation = ProductRemainder<AkitaField>;

    fn output_claims(
        &mut self,
        inputs: &SumcheckInputClaims<AkitaField, Self::Relation>,
    ) -> Result<SumcheckOutputClaims<AkitaField, Self::Relation>, SumcheckKernelError<AkitaField>>
    {
        let remaining = self.host.rounds.saturating_sub(self.host.challenges.len());
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let (e_in, e_out) = self.host.opening_weights();
        let span = tracing::info_span!(
            "MetalProductRemainder::output_claims",
            resident_rows_storage_id = self.row_storage_id as u64,
            row_upload_bytes = 0u64,
            dispatch_wall_ns = tracing::field::Empty,
            gpu_active_ns = tracing::field::Empty,
        );
        let _entered = span.enter();
        let started = Instant::now();
        let (values, gpu_active) = self
            .sequence
            .openings_timed(&e_in, &e_out)
            .map_err(metal_output_error)?;
        let _ = span.record("dispatch_wall_ns", duration_nanos(started.elapsed()));
        let _ = span.record("gpu_active_ns", duration_nanos(gpu_active));
        let ids = [
            left_instruction_input_product(),
            right_instruction_input_product(),
            jump_flag_product(),
            write_lookup_output_to_rd_product(),
            lookup_output_product(),
            branch_flag_product(),
            next_is_noop_product(),
            virtual_instruction_product(),
        ];
        let claims: BTreeMap<JoltOpeningId, AkitaField> = ids.into_iter().zip(values).collect();
        SumcheckOutputClaims::<AkitaField, Self::Relation>::from_opening_values(|id| {
            claims.get(id).copied().or_else(|| inputs.resolve_input(id))
        })
        .map_err(SumcheckKernelError::from)
    }

    fn validate_derived_tables(
        &self,
        relation: &Self::Relation,
        input_points: &SumcheckInputPoints<AkitaField, Self::Relation>,
        output_points: &SumcheckOutputPoints<AkitaField, Self::Relation>,
        challenges: &ConcreteSumcheckChallenges<AkitaField, Self::Relation>,
    ) -> Result<(), SumcheckKernelError<AkitaField>> {
        let remaining = self.host.rounds.saturating_sub(self.host.challenges.len());
        if remaining != 0 {
            return Err(SumcheckKernelError::NotFullyBound { remaining });
        }
        let ids = std::iter::once(SpartanProductVirtualizationPublic::TauKernel)
            .chain((0..DOMAIN).map(SpartanProductVirtualizationPublic::LagrangeWeight));
        for public_id in ids {
            let id = JoltDerivedId::from(public_id);
            let expected =
                match relation.derive_output_term(&id, input_points, output_points, challenges) {
                    Ok(value) => value,
                    Err(VerifierError::MissingStageClaimDerived { .. }) => continue,
                    Err(error) => return Err(error.into()),
                };
            let got = match public_id {
                SpartanProductVirtualizationPublic::TauKernel => {
                    self.host.split_eq.current_scalar()
                }
                SpartanProductVirtualizationPublic::LagrangeWeight(index) => {
                    self.host.lagrange_weights[index]
                }
                SpartanProductVirtualizationPublic::UniskipLagrangeWeight(_) => continue,
            };
            if got != expected {
                return Err(SumcheckKernelError::DerivedTableDrift { id, expected, got });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "test module")]
mod tests {
    use jolt_claims::protocols::jolt::geometry::spartan::SpartanProductDimensions;
    use jolt_claims::NoChallenges;
    use jolt_verifier::stages::stage2::product_remainder::{
        product_remainder_input_values_from_uniskip_output, ProductRemainderInputClaims,
    };
    use jolt_witness::testing::with_sample_backend_at_log_t;

    use super::*;
    use crate::optimized::spartan_product::OptimizedProductUniskip;
    use crate::uniskip::UniskipKernel;

    fn true_input_claim(
        rows: &[ProductRemainderRow],
        tau_low: &[AkitaField],
        tau_high: AkitaField,
        uniskip_challenge: AkitaField,
    ) -> AkitaField {
        let eq = EqPolynomial::<AkitaField>::evals(tau_low, None);
        let weights: [AkitaField; DOMAIN] = centered_lagrange_evals(DOMAIN, uniskip_challenge)
            .unwrap()
            .try_into()
            .unwrap();
        let scale = centered_lagrange_kernel(DOMAIN, tau_high, uniskip_challenge).unwrap();
        scale
            * rows
                .iter()
                .zip(eq)
                .map(|(&row, eq)| {
                    let (left, right) = row.relation_values(&weights);
                    eq * left * right
                })
                .sum::<AkitaField>()
    }

    #[test]
    fn resident_product_remainder_matches_optimized_cpu() {
        let log_t = 4usize;
        with_sample_backend_at_log_t(log_t, 4, |witness| {
            let tau_low = (0..log_t)
                .map(|index| AkitaField::from_u64(19 + 7 * index as u64))
                .collect::<Vec<_>>();
            let tau_high = AkitaField::from_u64(313);
            let uniskip_challenge = AkitaField::from_u64(911);
            let rows = collect_bundles::<SpartanProductRow>(witness, 1 << log_t)
                .unwrap()
                .iter()
                .map(ProductRemainderRow::from)
                .collect::<Vec<_>>();
            let input_claim = true_input_claim(&rows, &tau_low, tau_high, uniskip_challenge);
            let relation = ProductRemainder::new(
                SpartanProductDimensions::new(log_t),
                uniskip_challenge,
                tau_high,
                tau_low.clone(),
            );
            let claims = product_remainder_input_values_from_uniskip_output(input_claim);
            let points = ProductRemainderInputClaims::<Vec<AkitaField>>::default();
            let no_challenges = NoChallenges::<AkitaField>::default();

            let mut optimized_session = ProofSession::default();
            OptimizedProductUniskip
                .prepare(&mut optimized_session, log_t, &tau_low, witness)
                .unwrap();
            let _ = OptimizedProductUniskip
                .first_round_poly(&mut optimized_session, &[tau_high])
                .unwrap();
            let mut optimized = OptimizedProductRemainder
                .prepare(
                    &mut optimized_session,
                    witness,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &no_challenges,
                    },
                )
                .unwrap();

            let mut config = super::super::MetalConfig::default();
            config.spartan_product_remainder.trace_cutoff_elements = 2;
            let metal = MetalBackend::new(config).unwrap();
            let mut metal_session = ProofSession::default();
            metal
                .prepare_product_remainder_witness(&mut metal_session, log_t, witness)
                .unwrap();
            OptimizedProductUniskip
                .prepare(&mut metal_session, log_t, &tau_low, witness)
                .unwrap();
            let _ = OptimizedProductUniskip
                .first_round_poly(&mut metal_session, &[tau_high])
                .unwrap();
            let mut actual =
                <MetalBackend as PrepareKernel<AkitaField, ProductRemainder<AkitaField>>>::prepare(
                    &metal,
                    &mut metal_session,
                    witness,
                    ProverInputs {
                        relation: &relation,
                        claims: &claims,
                        points: &points,
                        challenges: &no_challenges,
                    },
                )
                .unwrap();
            assert_eq!(metal.product_remainder_sequences(), 1);

            let challenges = (0..log_t)
                .map(|index| AkitaField::from_u64(1201 + 43 * index as u64))
                .collect::<Vec<_>>();
            let mut bind = None;
            let mut previous_claim = input_claim;
            for (round, &challenge) in challenges.iter().enumerate() {
                let expected = optimized.prove_round(bind, round, previous_claim).unwrap();
                let got = actual.prove_round(bind, round, previous_claim).unwrap();
                assert_eq!(got, expected, "round {round}");
                previous_claim = expected.evaluate(challenge);
                bind = Some(challenge);
            }
            let final_challenge = *challenges.last().unwrap();
            optimized.finish_rounds(final_challenge).unwrap();
            actual.finish_rounds(final_challenge).unwrap();
            assert_eq!(
                actual.output_claims(&claims).unwrap(),
                optimized.output_claims(&claims).unwrap()
            );

            let output_points = relation
                .derive_opening_points(&challenges, &points)
                .unwrap();
            optimized
                .validate_derived_tables(&relation, &points, &output_points, &no_challenges)
                .unwrap();
            actual
                .validate_derived_tables(&relation, &points, &output_points, &no_challenges)
                .unwrap();
        });
    }
}
