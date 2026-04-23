use std::collections::{HashMap, HashSet};

use rayon::prelude::*;

use jolt_compiler::module::{
    ChallengeIdx, CheckpointEvalAction, DomainSeparator, EvalMode, Op, RoundPolyEncoding,
};
use jolt_compiler::PolynomialId;
use jolt_compute::{Buf, BufferProvider, ComputeBackend, DeviceBuffer, Executable};
use jolt_crypto::HomomorphicCommitment;
use jolt_field::Field;
use jolt_openings::{AdditivelyHomomorphic, ProverClaim};
use jolt_poly::UnivariatePoly;
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript};

use super::helpers::{
    bind_kernel_inputs, build_outer_eq, materialize_binding, materialize_pending_claims,
    PendingClaim,
};

/// Enter a per-variant `info_span!` for the op currently being dispatched.
///
/// The span name is the op variant name — chrome/Perfetto groups self-time by
/// span name, so this gives a per-op-kind wall-clock breakdown. Every arm is
/// one line; new `Op` variants must be added here or they'll be invisible.
fn op_span(op: &Op) -> tracing::span::EnteredSpan {
    match op {
        Op::SumcheckRound { .. } => tracing::info_span!("SumcheckRound").entered(),
        Op::Evaluate { .. } => tracing::info_span!("Evaluate").entered(),
        Op::Bind { .. } => tracing::info_span!("Bind").entered(),
        Op::LagrangeProject { .. } => tracing::info_span!("LagrangeProject").entered(),
        Op::DuplicateInterleave { .. } => tracing::info_span!("DuplicateInterleave").entered(),
        Op::RegroupConstraints { .. } => tracing::info_span!("RegroupConstraints").entered(),
        Op::Commit { .. } => tracing::info_span!("Commit").entered(),
        Op::CommitStreaming { .. } => tracing::info_span!("CommitStreaming").entered(),
        Op::ProveBatch => tracing::info_span!("ProveBatch").entered(),
        Op::Preamble => tracing::info_span!("Preamble").entered(),
        Op::BeginStage { index } => tracing::info_span!("BeginStage", index = index).entered(),
        Op::AbsorbRoundPoly { .. } => tracing::info_span!("AbsorbRoundPoly").entered(),
        Op::RecordEvals { .. } => tracing::info_span!("RecordEvals").entered(),
        Op::AbsorbEvals { .. } => tracing::info_span!("AbsorbEvals").entered(),
        Op::AbsorbInputClaim { .. } => tracing::info_span!("AbsorbInputClaim").entered(),
        Op::Squeeze { .. } => tracing::info_span!("Squeeze").entered(),
        Op::ComputePower { .. } => tracing::info_span!("ComputePower").entered(),
        Op::AppendDomainSeparator { .. } => tracing::info_span!("AppendDomainSeparator").entered(),
        Op::EvaluatePreprocessed { .. } => tracing::info_span!("EvaluatePreprocessed").entered(),
        Op::AliasEval { .. } => tracing::info_span!("AliasEval").entered(),
        Op::CollectOpeningClaim { .. } => tracing::info_span!("CollectOpeningClaim").entered(),
        Op::ScaleEval { .. } => tracing::info_span!("ScaleEval").entered(),
        Op::CollectOpeningClaimAt { .. } => tracing::info_span!("CollectOpeningClaimAt").entered(),
        Op::BindOpeningInputs { .. } => tracing::info_span!("BindOpeningInputs").entered(),
        Op::ReleaseDevice { .. } => tracing::info_span!("ReleaseDevice").entered(),
        Op::ReleaseHost { .. } => tracing::info_span!("ReleaseHost").entered(),
        Op::BatchRoundBegin { batch, round, .. } => {
            tracing::info_span!("BatchRoundBegin", batch = batch.0, round = round).entered()
        }
        Op::BatchInactiveContribution { .. } => {
            tracing::info_span!("BatchInactiveContribution").entered()
        }
        Op::Materialize { .. } => tracing::info_span!("Materialize").entered(),
        Op::MaterializeUnlessFresh { .. } => {
            tracing::info_span!("MaterializeUnlessFresh").entered()
        }
        Op::MaterializeIfAbsent { .. } => tracing::info_span!("MaterializeIfAbsent").entered(),
        Op::MaterializeSegmentedOuterEq { .. } => {
            tracing::info_span!("MaterializeSegmentedOuterEq").entered()
        }
        Op::InstanceBindPreviousPhase { .. } => {
            tracing::info_span!("InstanceBindPreviousPhase").entered()
        }
        Op::CaptureScalar { .. } => tracing::info_span!("CaptureScalar").entered(),
        Op::InstanceReduce {
            batch,
            instance,
            kernel,
        } => tracing::info_span!(
            "InstanceReduce",
            batch = batch.0,
            instance = instance.0,
            kernel = kernel
        )
        .entered(),
        Op::InstanceSegmentedReduce {
            batch,
            instance,
            kernel,
            round_within_phase,
            ..
        } => tracing::info_span!(
            "InstanceSegmentedReduce",
            batch = batch.0,
            instance = instance.0,
            kernel = kernel,
            round = round_within_phase
        )
        .entered(),
        Op::InstanceBind {
            batch,
            instance,
            kernel,
            ..
        } => tracing::info_span!(
            "InstanceBind",
            batch = batch.0,
            instance = instance.0,
            kernel = kernel
        )
        .entered(),
        Op::BindCarryBuffers { .. } => tracing::info_span!("BindCarryBuffers").entered(),
        Op::BatchAccumulateInstance { .. } => {
            tracing::info_span!("BatchAccumulateInstance").entered()
        }
        Op::BatchRoundFinalize { .. } => tracing::info_span!("BatchRoundFinalize").entered(),
        Op::ExpandingTableUpdate { .. } => tracing::info_span!("ExpandingTableUpdate").entered(),
        Op::CheckpointEvalBatch { .. } => tracing::info_span!("CheckpointEvalBatch").entered(),
        Op::InitInstanceWeights { .. } => tracing::info_span!("InitInstanceWeights").entered(),
        Op::UpdateInstanceWeights { .. } => tracing::info_span!("UpdateInstanceWeights").entered(),
        Op::SuffixScatter { .. } => tracing::info_span!("SuffixScatter").entered(),
        Op::QBufferScatter { .. } => tracing::info_span!("QBufferScatter").entered(),
        Op::MaterializePBuffers { .. } => tracing::info_span!("MaterializePBuffers").entered(),
        Op::InitExpandingTable { .. } => tracing::info_span!("InitExpandingTable").entered(),
        Op::ReadCheckingReduce { .. } => tracing::info_span!("ReadCheckingReduce").entered(),
        Op::RafReduce { .. } => tracing::info_span!("RafReduce").entered(),
        Op::MaterializeRA { .. } => tracing::info_span!("MaterializeRA").entered(),
        Op::MaterializeCombinedVal { .. } => {
            tracing::info_span!("MaterializeCombinedVal").entered()
        }
        Op::WeightedSum { .. } => tracing::info_span!("WeightedSum").entered(),
    }
}

/// Dispatch a single Op from the schedule.
#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_op<B, F, T, PCS>(
    op: &Op,
    state: &mut super::RuntimeState<F, PCS>,
    device_buffers: &mut HashMap<PolynomialId, Buf<B, F>>,
    executable: &Executable<B, F>,
    provider: &mut impl BufferProvider<F>,
    backend: &B,
    pcs_setup: &PCS::ProverSetup,
    transcript: &mut T,
    stage_point_indices: &[Vec<usize>],
) where
    B: ComputeBackend,
    F: Field,
    T: Transcript<Challenge = F>,
    PCS: AdditivelyHomomorphic<Field = F>,
    PCS::Output: AppendToTranscript + HomomorphicCommitment<F>,
{
    let _op_span = op_span(op);
    let module = &executable.module;

    match op {
        Op::SumcheckRound {
            kernel,
            round: _,
            bind_challenge,
        } => {
            let kdef = &module.prover.kernels[*kernel];
            let compiled_kernel = &executable.kernels[*kernel];

            if let Some(ch) = bind_challenge {
                let scalar = state.challenges[ch.0];
                bind_kernel_inputs(device_buffers, backend, compiled_kernel, kdef, scalar);
            }

            let input_refs: Vec<&Buf<B, F>> = kdef
                .inputs
                .iter()
                .map(|b| {
                    device_buffers.get(&b.poly()).unwrap_or_else(|| {
                        panic!(
                            "SumcheckRound: missing buffer {:?} (kernel={kernel})",
                            b.poly()
                        )
                    })
                })
                .collect();

            state.last_round_coeffs =
                backend.reduce(compiled_kernel, &input_refs, &state.challenges);
        }

        Op::Evaluate { poly, mode } => {
            if let Some(buf) = device_buffers.get(poly) {
                let data = backend.download(buf.as_field());
                let val = match data.len() {
                    0 => return,
                    1 => data[0],
                    2 => {
                        let r = state.last_squeezed;
                        data[0] + r * (data[1] - data[0])
                    }
                    n => panic!("Evaluate: {poly:?} has {n}-element buffer; expected 1 or 2"),
                };
                let _ = state.evaluations.insert(*poly, val);
                let _ = state
                    .staged_evals
                    .insert((*poly, state.current_stage_idx), val);
            } else if matches!(mode, EvalMode::RoundPoly) {
                let round_poly = state
                    .last_round_poly
                    .as_ref()
                    .expect("RoundPoly: no round polynomial available");
                let val = round_poly.evaluate(state.last_squeezed);
                let _ = state.evaluations.insert(*poly, val);
                let _ = state
                    .staged_evals
                    .insert((*poly, state.current_stage_idx), val);
            } else if let Some(round_poly) = &state.last_round_poly {
                let val = round_poly.evaluate(state.last_squeezed);
                let _ = state.evaluations.insert(*poly, val);
                let _ = state
                    .staged_evals
                    .insert((*poly, state.current_stage_idx), val);
            }
        }

        Op::Bind {
            polys,
            challenge,
            order,
        } => {
            let scalar = state.challenges[challenge.0];
            for pi in polys {
                if !device_buffers.contains_key(pi) {
                    let data = provider.materialize(*pi);
                    let buf = DeviceBuffer::Field(backend.upload(&data));
                    let _ = device_buffers.insert(*pi, buf);
                }
                if let Some(DeviceBuffer::Field(buf)) = device_buffers.get_mut(pi) {
                    backend.interpolate_inplace(buf, scalar, *order);
                }
            }
        }

        Op::LagrangeProject {
            polys,
            challenge,
            domain_size,
            domain_start,
            stride,
            group_offsets,
            kernel_tau,
        } => {
            let r = state.challenges[challenge.0];

            // Lagrange kernel scale: L(τ, r) = Σ_k L_k(τ) · L_k(r)
            let scale = if let Some(tau_idx) = kernel_tau {
                let tau = state.challenges[tau_idx.0];
                let basis = jolt_poly::lagrange::lagrange_evals(*domain_start, *domain_size, r);
                let tau_basis =
                    jolt_poly::lagrange::lagrange_evals(*domain_start, *domain_size, tau);
                basis
                    .iter()
                    .zip(tau_basis.iter())
                    .map(|(&lk_r, &lk_tau)| lk_r * lk_tau)
                    .sum::<F>()
            } else {
                F::one()
            };

            for pi in polys {
                let buf = device_buffers
                    .remove(pi)
                    .expect("LagrangeProject: buffer missing");
                // Scale applies to first poly only (avoids squaring the factor).
                let poly_scale = if *pi == polys[0] { scale } else { F::one() };
                let projected = backend.lagrange_project(
                    buf.as_field(),
                    r,
                    *domain_start,
                    *domain_size,
                    *stride,
                    group_offsets,
                    poly_scale,
                );
                let _ = device_buffers.insert(*pi, DeviceBuffer::Field(projected));
            }
        }

        Op::DuplicateInterleave { polys } => {
            for pi in polys {
                let buf = device_buffers
                    .remove(pi)
                    .expect("DuplicateInterleave: buffer missing");
                let expanded = DeviceBuffer::Field(backend.duplicate_interleave(buf.as_field()));
                let _ = device_buffers.insert(*pi, expanded);
            }
        }

        Op::RegroupConstraints {
            polys,
            group_indices,
            old_stride,
            new_stride,
            num_cycles,
        } => {
            for pi in polys {
                // Auto-materialize if not yet on device.
                if !device_buffers.contains_key(pi) {
                    let data = provider.materialize(*pi);
                    let buf = DeviceBuffer::Field(backend.upload(&data));
                    let _ = device_buffers.insert(*pi, buf);
                }
                let buf = device_buffers
                    .remove(pi)
                    .expect("RegroupConstraints: buffer missing");
                let regrouped = DeviceBuffer::Field(backend.regroup_constraints(
                    buf.as_field(),
                    group_indices,
                    *old_stride,
                    *new_stride,
                    *num_cycles,
                ));
                let _ = device_buffers.insert(*pi, regrouped);
            }
        }

        Op::Commit {
            polys,
            tag,
            num_vars,
        }
        | Op::CommitStreaming {
            polys,
            tag,
            num_vars,
            ..
        } => {
            // jolt-core skips advice commits when data is empty/zero.
            // Push `None` per poly so the verifier's AbsorbCommitment
            // schedule stays aligned; the verifier also skips the
            // transcript append on None. (Transcript parity vs jolt-core
            // is checked by transcript_divergence.)
            let skip = matches!(
                tag,
                DomainSeparator::UntrustedAdvice | DomainSeparator::TrustedAdvice
            ) && polys.iter().all(|pi| {
                let raw = provider.materialize(*pi);
                raw.iter().all(|v| *v == F::zero())
            });
            if skip {
                for _ in polys {
                    state.commitments.push(None);
                }
                return;
            }

            let target_len = 1 << num_vars;
            // Materialize sequentially (BufferProvider's &mut receiver is shared here).
            let data: Vec<(PolynomialId, Vec<F>)> = polys
                .iter()
                .map(|pi| {
                    let raw = provider.materialize(*pi);
                    let mut v = raw.into_owned();
                    if v.len() < target_len {
                        v.resize(target_len, F::zero());
                    }
                    (*pi, v)
                })
                .collect();
            // Parallel: run Dory PCS::commit per polynomial (each call is tier-1 G1::msm
            // chunks + tier-2 Pedersen). Trace showed 42 serial commits at 12 ms avg
            // = 508 ms wall with only 1.4× effective internal parallelism.
            let results: Vec<(PolynomialId, PCS::Output, PCS::OpeningHint)> = data
                .into_par_iter()
                .map(|(pi, data)| {
                    let (commitment, hint) = PCS::commit(&data[..], pcs_setup);
                    (pi, commitment, hint)
                })
                .collect();
            // Sequential: append to transcript in the same order as the serial loop.
            for (pi, commitment, hint) in results {
                transcript.append(&LabelWithCount(tag.as_bytes(), commitment.serialized_len()));
                commitment.append_to_transcript(transcript);
                let _ = state.hints.insert(pi, hint);
                state.commitments.push(Some(commitment));
            }
        }

        Op::ProveBatch => {
            let pending = std::mem::take(&mut state.pending_claims);
            let hints = std::mem::take(&mut state.pending_hints);

            let claims: Vec<ProverClaim<F>> =
                materialize_pending_claims(pending, provider, &mut state.padded_poly_data);

            let (batch_proof, joint_evals) = PCS::prove_batch(claims, hints, pcs_setup, transcript);

            state.opening_proof = Some(batch_proof);
            state.binding_evals = joint_evals;
        }

        Op::Preamble => {
            transcript.append(&state.config);
        }

        Op::BeginStage { index } => {
            if let Some(builder) = state.current_stage.take() {
                state.stage_proofs.push(builder.finalize());
            }
            state.current_stage = Some(super::StageBuilder::new());
            state.current_stage_idx = *index;
        }

        Op::AbsorbRoundPoly {
            num_coeffs,
            tag,
            encoding,
        } => {
            let coeffs = match encoding {
                RoundPolyEncoding::Uniskip {
                    domain_size,
                    domain_start,
                    tau_challenge,
                    zero_base,
                } => {
                    let mut raw_evals = std::mem::take(&mut state.last_round_coeffs);
                    debug_assert_eq!(raw_evals.len(), 2 * *domain_size - 1);
                    let tau_high = state.challenges[tau_challenge.0];
                    backend.uniskip_encode(
                        &mut raw_evals,
                        *domain_size,
                        *domain_start,
                        tau_high,
                        *zero_base,
                        *num_coeffs,
                    )
                }
                RoundPolyEncoding::Compressed => {
                    backend.compressed_encode(&state.last_round_coeffs[..*num_coeffs])
                }
            };

            match encoding {
                RoundPolyEncoding::Uniskip { .. } => {
                    transcript.append(&LabelWithCount(tag.as_bytes(), coeffs.len() as u64));
                    for c in &coeffs {
                        transcript.append(c);
                    }
                }
                RoundPolyEncoding::Compressed => {
                    let compressed_len = coeffs.len() - 1;
                    transcript.append(&LabelWithCount(tag.as_bytes(), compressed_len as u64));
                    transcript.append(&coeffs[0]);
                    for c in &coeffs[2..] {
                        transcript.append(c);
                    }
                }
            }
            let round_poly = UnivariatePoly::new(coeffs);
            state.last_round_poly = Some(round_poly.clone());
            if let Some(stage) = &mut state.current_stage {
                stage.round_polys.push(round_poly);
            }
        }

        Op::RecordEvals { polys } => {
            if let Some(stage) = &mut state.current_stage {
                for pi in polys {
                    if let Some(&val) = state.evaluations.get(pi) {
                        stage.evals.push(val);
                    }
                }
            }
        }

        Op::AbsorbEvals { polys, tag } => {
            for pi in polys {
                if let Some(&val) = state.evaluations.get(pi) {
                    transcript.append(&Label(tag.as_bytes()));
                    transcript.append(&val);
                }
            }
        }

        Op::AbsorbInputClaim {
            formula,
            tag,
            batch,
            instance,
            inactive_scale_bits,
        } => {
            let val = backend.evaluate_claim(
                formula,
                &state.evaluations,
                &state.staged_evals,
                &state.challenges,
            );
            transcript.append(&Label(tag.as_bytes()));
            transcript.append(&val);
            // Pre-scale claim by 2^inactive_scale_bits so that the
            // inactive-round halving lands on the correct value.
            let mut scaled = val;
            let two = F::from_u64(2);
            for _ in 0..*inactive_scale_bits {
                scaled *= two;
            }
            state.batch_instance_claims[batch.0][instance.0] = scaled;
        }

        Op::Squeeze { challenge } => {
            let val = transcript.challenge();
            state.challenges[challenge.0] = val;
            state.last_squeezed = val;
        }

        Op::ComputePower {
            target,
            base,
            exponent,
        } => {
            let base_val = state.challenges[base.0];
            let mut result = F::one();
            let mut b = base_val;
            let mut exp = *exponent;
            while exp > 0 {
                if exp & 1 == 1 {
                    result *= b;
                }
                b = b.square();
                exp >>= 1;
            }
            state.challenges[target.0] = result;
        }

        Op::AppendDomainSeparator { tag } => {
            let label = tag.as_bytes();
            let mut packed = [0u8; 32];
            packed[..label.len()].copy_from_slice(label);
            transcript.append_bytes(&packed);
            transcript.append_bytes(&[]);
        }

        Op::EvaluatePreprocessed {
            source,
            at_challenges,
            store_as,
        } => {
            let data = provider.materialize(*source);
            let point: Vec<F> = at_challenges
                .iter()
                .map(|&ci| state.challenges[ci.0])
                .collect();
            let eval = backend.evaluate_mle(&data, &point);
            let _ = state.evaluations.insert(*store_as, eval);
        }

        Op::AliasEval { from, to } => {
            if let Some(&val) = state.evaluations.get(from) {
                let _ = state.evaluations.insert(*to, val);
            }
        }

        Op::CollectOpeningClaim { poly, at_stage } => {
            if let Some(&eval) = state.evaluations.get(poly) {
                let point: Vec<F> = stage_point_indices[at_stage.0]
                    .iter()
                    .map(|&ci| state.challenges[ci])
                    .collect();
                state.pending_claims.push(PendingClaim {
                    poly: *poly,
                    point,
                    eval,
                });
                let hint = state.hints.get(poly).cloned().unwrap_or_default();
                state.pending_hints.push(hint);
            }
        }

        Op::ScaleEval {
            poly,
            factor_challenges,
        } => {
            if let Some(eval) = state.evaluations.get_mut(poly) {
                let factor: F = factor_challenges
                    .iter()
                    .map(|&ci| F::one() - state.challenges[ci.0])
                    .product();
                *eval *= factor;
            }
        }

        Op::CollectOpeningClaimAt {
            poly,
            point_challenges,
            committed_num_vars,
        } => {
            if let Some(&eval) = state.evaluations.get(poly) {
                let point: Vec<F> = point_challenges
                    .iter()
                    .map(|&ci| state.challenges[ci.0])
                    .collect();
                if let Some(nv) = committed_num_vars {
                    let target_len = 1 << nv;
                    let data = provider.materialize(*poly);
                    if data.len() < target_len {
                        let mut v = data.to_vec();
                        v.resize(target_len, F::zero());
                        let _ = state.padded_poly_data.insert(*poly, v);
                    }
                }
                state.pending_claims.push(PendingClaim {
                    poly: *poly,
                    point,
                    eval,
                });
                let hint = state.hints.get(poly).cloned().unwrap_or_default();
                state.pending_hints.push(hint);
            }
        }

        Op::BindOpeningInputs { point_challenges } => {
            let point: Vec<F> = point_challenges
                .iter()
                .map(|&ci| state.challenges[ci.0])
                .collect();
            let joint_eval = state.binding_evals.first().copied().unwrap_or(F::zero());
            PCS::bind_opening_inputs(transcript, &point, &joint_eval);
        }

        Op::ReleaseDevice { poly } => {
            let _ = device_buffers.remove(poly);
        }

        Op::ReleaseHost { polys } => {
            for pi in polys {
                provider.release(*pi);
            }
        }

        Op::BatchRoundBegin {
            batch,
            round,
            max_evals,
            bind_challenge,
        } => {
            state.current_batch_round = *round;
            state.batch_combined = vec![F::zero(); *max_evals];
            state.bound_this_round.clear();
            if let Some(ch) = bind_challenge {
                let r = state.challenges[ch.0];
                for (inst_idx, evals) in state.last_round_instance_evals.iter().enumerate() {
                    if !evals.is_empty() {
                        state.batch_instance_claims[batch.0][inst_idx] =
                            backend.interpolate_evaluate(evals, r);
                    }
                }
            }
            let num_instances = state.batch_instance_claims[batch.0].len();
            state.last_round_instance_evals = vec![Vec::new(); num_instances];
        }

        Op::BatchInactiveContribution { batch, instance } => {
            let bdef = &module.prover.batched_sumchecks[batch.0];
            let coeff = state.challenges[bdef.instances[instance.0].batch_coeff.0];
            let two_inv = F::from_u64(2).inverse().unwrap();
            let half_claim = state.batch_instance_claims[batch.0][instance.0] * two_inv;
            for slot in &mut state.batch_combined {
                *slot += coeff * half_claim;
            }
            state.batch_instance_claims[batch.0][instance.0] = half_claim;
        }

        Op::Materialize { binding } => {
            let pi = binding.poly();
            let buf = materialize_binding(binding, &state.challenges, provider, backend);
            let _ = device_buffers.insert(pi, buf);
        }

        Op::MaterializeUnlessFresh {
            binding,
            expected_size,
        } => {
            let pi = binding.poly();
            if let Some(existing) = device_buffers.get(&pi) {
                if backend.len(existing.as_field()) == *expected_size {
                    return;
                }
            }
            let buf = materialize_binding(binding, &state.challenges, provider, backend);
            let _ = device_buffers.insert(pi, buf);
        }

        Op::MaterializeIfAbsent { binding } => {
            let pi = binding.poly();
            if device_buffers.contains_key(&pi) {
                return;
            }
            let buf = materialize_binding(binding, &state.challenges, provider, backend);
            let _ = device_buffers.insert(pi, buf);
        }

        Op::MaterializeSegmentedOuterEq {
            batch,
            instance,
            segmented,
        } => {
            let outer_eq = build_outer_eq(&state.challenges, segmented, backend);
            let _ = state
                .segmented_outer_eqs
                .insert((batch.0, instance.0), outer_eq);
        }

        Op::InstanceBindPreviousPhase {
            batch: _,
            instance: _,
            kernel,
            challenge,
        } => {
            let kdef = &module.prover.kernels[*kernel];
            let scalar = state.challenges[challenge.0];
            let order = kdef.spec.binding_order;
            let mut seen = HashSet::new();
            for b in &kdef.inputs {
                let pid = b.poly();
                if state.bound_this_round.contains(&pid) || !seen.insert(pid) {
                    continue;
                }
                if let Some(buf) = device_buffers.get_mut(&pid) {
                    backend.interpolate_inplace(buf.as_field_mut(), scalar, order);
                }
                let _ = state.bound_this_round.insert(pid);
            }
        }

        Op::CaptureScalar { poly, challenge } => {
            let buf = device_buffers
                .get(poly)
                .expect("CaptureScalar: buffer not found");
            let data = backend.download(buf.as_field());
            assert!(
                data.len() == 1,
                "CaptureScalar: expected 1-element buffer for {:?}, got {}",
                poly,
                data.len()
            );
            state.challenges[challenge.0] = data[0];
        }

        Op::InstanceReduce {
            batch: _,
            instance,
            kernel,
        } => {
            let kdef = &module.prover.kernels[*kernel];
            let compiled_kernel = &executable.kernels[*kernel];
            let input_refs: Vec<&Buf<B, F>> = kdef
                .inputs
                .iter()
                .map(|b| {
                    device_buffers.get(&b.poly()).unwrap_or_else(|| {
                        panic!(
                            "InstanceReduce: missing buffer {:?} (inst={}, kernel={kernel})",
                            b.poly(),
                            instance.0
                        )
                    })
                })
                .collect();
            let inst_evals = backend.reduce(compiled_kernel, &input_refs, &state.challenges);
            state.last_round_instance_evals[instance.0].clone_from(&inst_evals);
        }

        Op::InstanceSegmentedReduce {
            batch,
            instance,
            kernel,
            round_within_phase,
            segmented,
        } => {
            let kdef = &module.prover.kernels[*kernel];
            let compiled_kernel = &executable.kernels[*kernel];
            let outer_eq = state
                .segmented_outer_eqs
                .get(&(batch.0, instance.0))
                .expect("InstanceSegmentedReduce: outer eq missing");
            let inner_size = 1usize << (segmented.inner_num_vars - round_within_phase);
            let input_bufs: Vec<&B::Buffer<F>> = kdef
                .inputs
                .iter()
                .map(|b| {
                    device_buffers
                        .get(&b.poly())
                        .expect("InstanceSegmentedReduce: input missing")
                        .as_field()
                })
                .collect();
            let inst_evals = if kdef.spec.gruen_hint.is_some() {
                let prev_claim = state.batch_instance_claims[batch.0][instance.0];
                backend.gruen_segmented_reduce(
                    compiled_kernel,
                    &input_bufs,
                    outer_eq,
                    &segmented.inner_only,
                    inner_size,
                    &state.challenges,
                    prev_claim,
                    *round_within_phase,
                )
            } else {
                backend.segmented_reduce(
                    compiled_kernel,
                    &input_bufs,
                    outer_eq,
                    &segmented.inner_only,
                    inner_size,
                    &state.challenges,
                )
            };
            state.last_round_instance_evals[instance.0].clone_from(&inst_evals);
        }

        Op::InstanceBind {
            batch: _,
            instance: _,
            kernel,
            challenge,
        } => {
            let kdef = &module.prover.kernels[*kernel];
            let scalar = state.challenges[challenge.0];
            let order = kdef.spec.binding_order;
            let mut seen = HashSet::new();
            for b in &kdef.inputs {
                let pid = b.poly();
                if state.bound_this_round.contains(&pid) || !seen.insert(pid) {
                    continue;
                }
                if let Some(buf) = device_buffers.get_mut(&pid) {
                    backend.interpolate_inplace(buf.as_field_mut(), scalar, order);
                }
                let _ = state.bound_this_round.insert(pid);
            }
        }

        Op::BindCarryBuffers {
            polys,
            challenge,
            order,
        } => {
            let scalar = state.challenges[challenge.0];
            for pid in polys {
                if state.bound_this_round.contains(pid) {
                    continue;
                }
                if let Some(buf) = device_buffers.get_mut(pid) {
                    backend.interpolate_inplace(buf.as_field_mut(), scalar, *order);
                }
                let _ = state.bound_this_round.insert(*pid);
            }
        }

        Op::BatchAccumulateInstance {
            batch,
            instance,
            max_evals,
            num_evals,
        } => {
            let bdef = &module.prover.batched_sumchecks[batch.0];
            let coeff = state.challenges[bdef.instances[instance.0].batch_coeff.0];
            let evals = &state.last_round_instance_evals[instance.0];
            debug_assert_eq!(evals.len(), *num_evals);
            let extended;
            let full_evals = if *num_evals < *max_evals {
                extended = backend.extend_evals(evals, *max_evals);
                &extended
            } else {
                evals.as_slice()
            };
            for (i, &v) in full_evals.iter().enumerate() {
                state.batch_combined[i] += coeff * v;
            }
        }

        Op::BatchRoundFinalize { batch: _ } => {
            state.last_round_coeffs = std::mem::take(&mut state.batch_combined);
        }

        Op::ExpandingTableUpdate {
            table,
            challenge,
            current_len,
        } => {
            let r = state.challenges[challenge.0];
            let buf = device_buffers
                .get(table)
                .expect("ExpandingTableUpdate: buffer missing");
            let mut data = backend.download(buf.as_field());
            for i in (0..*current_len).rev() {
                let v_i = data[i];
                let eval_1 = r * v_i;
                data[2 * i] = v_i - eval_1;
                data[2 * i + 1] = eval_1;
            }
            let new_buf = backend.upload(&data);
            let _ = device_buffers.insert(*table, DeviceBuffer::Field(new_buf));
        }

        Op::CheckpointEvalBatch { updates } => {
            let snapshot: Vec<Option<F>> = state.instance_checkpoints.clone();
            let empty_buffers: std::collections::HashMap<PolynomialId, &[F]> =
                std::collections::HashMap::new();
            for (idx, action) in updates {
                match action {
                    CheckpointEvalAction::Set(expr) => {
                        let v = crate::scalar_expr::eval_scalar_expr(
                            expr,
                            &state.challenges,
                            &snapshot,
                            0,
                            &empty_buffers,
                        );
                        state.instance_checkpoints[*idx] = Some(v);
                    }
                    CheckpointEvalAction::Clear => {
                        state.instance_checkpoints[*idx] = None;
                    }
                }
            }
        }

        Op::InitInstanceWeights {
            r_reduction,
            num_prefixes,
        } => {
            let point: Vec<F> = r_reduction
                .iter()
                .map(|ci| state.challenges[ci.0])
                .collect();
            state.instance_weights = jolt_poly::EqPolynomial::<F>::evals(&point, None);
            state.instance_checkpoints = vec![None; *num_prefixes];
        }

        Op::UpdateInstanceWeights {
            expanding_table,
            chunk_bits,
            num_phases,
            phase,
        } => {
            let trace = provider.lookup_trace().unwrap();
            let prev_data = backend.download(device_buffers[expanding_table].as_field());
            let m_mask = (1usize << chunk_bits) - 1;
            let suffix_len = (num_phases - phase) * chunk_bits;
            for (j, &key) in trace.lookup_keys.iter().enumerate() {
                state.instance_weights[j] *= prev_data[((key >> suffix_len) as usize) & m_mask];
            }
        }

        Op::SuffixScatter { kernel, phase } => {
            let config = executable.module.prover.kernels[*kernel]
                .instance_config
                .as_ref()
                .unwrap();
            let trace = provider.lookup_trace().unwrap();
            let m = 1usize << config.chunk_bits;
            let suffix_len = (config.num_phases - 1 - phase) * config.chunk_bits;
            let suffix_mask = (1u128 << suffix_len).wrapping_sub(1);
            let mut all_polys: Vec<Vec<Vec<F>>> = (0..config.num_tables)
                .map(|t| vec![vec![F::zero(); m]; config.suffixes_per_table[t]])
                .collect();
            for (j, &key) in trace.lookup_keys.iter().enumerate() {
                let Some(t) = trace.table_kind_indices[j] else {
                    continue;
                };
                let idx = ((key >> suffix_len) as usize) & (m - 1);
                let u = state.instance_weights[j];
                for (poly, op) in all_polys[t].iter_mut().zip(config.suffix_ops[t].iter()) {
                    let v = op.eval(key & suffix_mask, suffix_len);
                    match v {
                        0 => {}
                        1 => poly[idx] += u,
                        _ => poly[idx] += u.mul_u64(v),
                    }
                }
            }
            for (t, table_polys) in all_polys.into_iter().enumerate() {
                for (s, p) in table_polys.into_iter().enumerate() {
                    let _ = device_buffers.insert(
                        PolynomialId::InstanceSuffix(t, s),
                        DeviceBuffer::Field(backend.upload(&p)),
                    );
                }
            }
        }

        Op::QBufferScatter { kernel, phase } => {
            let config = executable.module.prover.kernels[*kernel]
                .instance_config
                .as_ref()
                .unwrap();
            let trace = provider.lookup_trace().unwrap();
            let m = 1usize << config.chunk_bits;
            let suffix_len = (config.num_phases - 1 - phase) * config.chunk_bits;
            let suffix_mask = (1u128 << suffix_len).wrapping_sub(1);
            let shift_half_f = F::from_u128(if suffix_len >= 2 {
                1u128 << (suffix_len / 2)
            } else {
                1
            });
            let shift_full_f = F::from_u128(if suffix_len > 0 {
                1u128 << suffix_len
            } else {
                1
            });
            let (mut sh, mut l, mut r, mut sf, mut id) = (
                vec![F::zero(); m],
                vec![F::zero(); m],
                vec![F::zero(); m],
                vec![F::zero(); m],
                vec![F::zero(); m],
            );
            for (j, &key) in trace.lookup_keys.iter().enumerate() {
                let idx = ((key >> suffix_len) as usize) & (m - 1);
                let u = state.instance_weights[j];
                if trace.is_interleaved[j] {
                    sh[idx] += u;
                    let (lo, ro) = uninterleave_u128(key & suffix_mask);
                    if lo != 0 {
                        l[idx] += u.mul_u64(lo);
                    }
                    if ro != 0 {
                        r[idx] += u.mul_u64(ro);
                    }
                } else {
                    sf[idx] += u;
                    let v = key & suffix_mask;
                    if v != 0 {
                        id[idx] += u.mul_u128(v);
                    }
                }
            }
            for v in &mut sh {
                *v *= shift_half_f;
            }
            for v in &mut sf {
                *v *= shift_full_f;
            }
            for (c, q) in [[sh.clone(), l], [sh, r], [sf, id]].into_iter().enumerate() {
                let _ = device_buffers.insert(
                    PolynomialId::InstanceQ(c, 0),
                    DeviceBuffer::Field(backend.upload(&q[0])),
                );
                let _ = device_buffers.insert(
                    PolynomialId::InstanceQ(c, 1),
                    DeviceBuffer::Field(backend.upload(&q[1])),
                );
            }
        }

        Op::MaterializePBuffers { kernel } => {
            let config = executable.module.prover.kernels[*kernel]
                .instance_config
                .as_ref()
                .unwrap();
            let m = 1usize << config.chunk_bits;
            let half_m = 1usize << (config.chunk_bits / 2);
            let cp = |i: usize| {
                let v = state.challenges[config.registry_checkpoint_slots[i].0];
                if v != F::zero() {
                    Some(v)
                } else {
                    None
                }
            };
            let id_base = cp(2).unwrap_or(F::zero()) * F::from_u64(m as u64);
            let p_identity: Vec<F> = (0..m).map(|i| id_base + F::from_u64(i as u64)).collect();
            let left_base = cp(1).unwrap_or(F::zero()) * F::from_u64(half_m as u64);
            let right_base = cp(0).unwrap_or(F::zero()) * F::from_u64(half_m as u64);
            let mut p_left = vec![F::zero(); m];
            let mut p_right = vec![F::zero(); m];
            for i in 0..m {
                let (lo, ro) = uninterleave_u128(i as u128);
                p_left[i] = left_base + F::from_u64(lo);
                p_right[i] = right_base + F::from_u64(ro);
            }
            let _ = device_buffers.insert(
                PolynomialId::InstanceP(0, 0),
                DeviceBuffer::Field(backend.upload(&p_left)),
            );
            let _ = device_buffers.insert(
                PolynomialId::InstanceP(1, 0),
                DeviceBuffer::Field(backend.upload(&p_right)),
            );
            let _ = device_buffers.insert(
                PolynomialId::InstanceP(2, 0),
                DeviceBuffer::Field(backend.upload(&p_identity)),
            );
        }

        Op::InitExpandingTable { table, size } => {
            let mut expanding = vec![F::zero(); *size];
            expanding[0] = F::one();
            let _ = device_buffers.insert(*table, DeviceBuffer::Field(backend.upload(&expanding)));
        }

        Op::ReadCheckingReduce {
            kernel,
            round,
            r_x_challenge,
        } => {
            let config = executable.module.prover.kernels[*kernel]
                .instance_config
                .as_ref()
                .unwrap();
            let suffix_polys: Vec<Vec<Vec<F>>> = (0..config.num_tables)
                .map(|t| {
                    (0..config.suffixes_per_table[t])
                        .map(|s| {
                            backend.download(
                                device_buffers[&PolynomialId::InstanceSuffix(t, s)].as_field(),
                            )
                        })
                        .collect()
                })
                .collect();
            let checkpoints: Vec<Option<F>> = state.instance_checkpoints.clone();
            let r_x: Option<F> = r_x_challenge.map(|ci| state.challenges[ci.0]);
            state.read_checking_evals =
                crate::runtime::prefix_suffix::compute_read_checking_from_lowered(
                    &config.prefix_lowered[*round],
                    &config.combine_entries,
                    &suffix_polys,
                    &checkpoints,
                    r_x,
                );
        }

        Op::RafReduce {
            batch: ps_batch,
            instance,
            kernel,
        } => {
            let config = executable.module.prover.kernels[*kernel]
                .instance_config
                .as_ref()
                .unwrap();
            let gamma = state.challenges[config.gamma.0];
            let gamma_sqr = gamma * gamma;
            let q_bufs: Vec<Vec<F>> = (0..3)
                .flat_map(|c| (0..2).map(move |h| (c, h)))
                .map(|(c, h)| {
                    backend.download(device_buffers[&PolynomialId::InstanceQ(c, h)].as_field())
                })
                .collect();
            let p_bufs: Vec<Option<Vec<F>>> = (0..3)
                .flat_map(|c| (0..2).map(move |h| (c, h)))
                .map(|(c, h)| {
                    device_buffers
                        .get(&PolynomialId::InstanceP(c, h))
                        .map(|b| backend.download(b.as_field()))
                })
                .collect();
            let half = q_bufs[0].len() / 2;
            let (mut l0, mut l2, mut r0, mut r2) = (F::zero(), F::zero(), F::zero(), F::zero());
            for b in 0..half {
                for (comp, (qb, pb)) in [(0usize, 0usize), (2, 2), (1, 1)].into_iter().enumerate() {
                    let (mut e0, mut e2l, mut e2r) = (F::zero(), F::zero(), F::zero());
                    for i in 0..2 {
                        let (p0, p2) = match p_bufs[pb * 2 + i] {
                            Some(ref d) => {
                                let pl = d[b];
                                (pl, d[b + half] + d[b + half] - pl)
                            }
                            None => (F::one(), F::one()),
                        };
                        e0 += p0 * q_bufs[qb * 2 + i][b];
                        e2l += p2 * q_bufs[qb * 2 + i][b];
                        e2r += p2 * q_bufs[qb * 2 + i][b + half];
                    }
                    if comp == 0 {
                        l0 += e0;
                        l2 += e2r + e2r - e2l;
                    } else {
                        r0 += e0;
                        r2 += e2r + e2r - e2l;
                    }
                }
            }
            let eval_0 = state.read_checking_evals[0] + gamma * l0 + gamma_sqr * r0;
            let eval_2 = state.read_checking_evals[1] + gamma * l2 + gamma_sqr * r2;
            let eval_1 = state.batch_instance_claims[ps_batch.0][instance.0] - eval_0;
            state.last_round_instance_evals[instance.0] = vec![eval_0, eval_1, eval_2];
        }

        Op::MaterializeRA { kernel } => {
            let config = executable.module.prover.kernels[*kernel]
                .instance_config
                .as_ref()
                .unwrap();
            let trace = provider.lookup_trace().unwrap();
            let (chunk_bits, num_phases) = (config.chunk_bits, config.num_phases);
            let m_mask = (1usize << chunk_bits) - 1;
            let tables: Vec<Vec<F>> = (0..num_phases)
                .map(|p| {
                    backend.download(device_buffers[&PolynomialId::ExpandingTable(p)].as_field())
                })
                .collect();
            let n_vra = 128 / config.ra_virtual_log_k_chunk;
            let chunk_size = num_phases / n_vra;
            for chunk_i in 0..n_vra {
                let off = chunk_i * chunk_size;
                let ra: Vec<F> = trace
                    .lookup_keys
                    .iter()
                    .map(|&key| {
                        let mut shift = (num_phases - 1 - off) * chunk_bits;
                        let mut acc = tables[off][((key >> shift) as usize) & m_mask];
                        for et in &tables[(off + 1)..(off + chunk_size)] {
                            shift -= chunk_bits;
                            acc *= et[((key >> shift) as usize) & m_mask];
                        }
                        acc
                    })
                    .collect();
                let _ = device_buffers.insert(
                    config.output_ra_polys[chunk_i],
                    DeviceBuffer::Field(backend.upload(&ra)),
                );
            }
        }

        Op::MaterializeCombinedVal { kernel } => {
            let config = executable.module.prover.kernels[*kernel]
                .instance_config
                .as_ref()
                .unwrap();
            let trace = provider.lookup_trace().unwrap();
            let prefix_vals: Vec<F> = state
                .instance_checkpoints
                .iter()
                .map(|v| v.unwrap_or(F::zero()))
                .collect();
            let mut table_values = vec![F::zero(); config.suffix_at_empty.len()];
            for e in &config.combine_entries {
                let p = e.prefix_idx.map_or(F::one(), |i| prefix_vals[i]);
                let s = F::from_u64(config.suffix_at_empty[e.table_idx][e.suffix_local_idx]);
                table_values[e.table_idx] += F::from_i128(e.coefficient) * p * s;
            }
            let gamma = state.challenges[config.gamma.0];
            let gsqr = gamma * gamma;
            let left = state.challenges[config.registry_checkpoint_slots[1].0];
            let right = state.challenges[config.registry_checkpoint_slots[0].0];
            let ident = state.challenges[config.registry_checkpoint_slots[2].0];
            let raf_inter = gamma * left + gsqr * right;
            let raf_ident = gsqr * ident;
            let combined: Vec<F> = (0..trace.lookup_keys.len())
                .map(|j| {
                    trace.table_kind_indices[j].map_or(F::zero(), |t| table_values[t])
                        + if trace.is_interleaved[j] {
                            raf_inter
                        } else {
                            raf_ident
                        }
                })
                .collect();
            let _ = device_buffers.insert(
                config.output_combined_val,
                DeviceBuffer::Field(backend.upload(&combined)),
            );
        }

        Op::WeightedSum {
            output,
            terms,
            identity_term,
            overall_scale,
        } => {
            let first_src = resolve_source(&terms[0].0, device_buffers, provider, backend);
            let n = first_src.len();
            let mut result = vec![F::zero(); n];
            accumulate(
                &mut result,
                &first_src,
                challenge_power(&state.challenges, &terms[0].1, terms[0].2),
            );
            for &(ref src, ref ch, pow) in &terms[1..] {
                let data = resolve_source(src, device_buffers, provider, backend);
                accumulate(
                    &mut result,
                    &data,
                    challenge_power(&state.challenges, ch, pow),
                );
            }
            if let Some((ch, pow)) = identity_term {
                let scale = challenge_power(&state.challenges, ch, *pow);
                for (i, r) in result.iter_mut().enumerate() {
                    *r += scale * F::from_u64(i as u64);
                }
            }
            if let Some((ch, pow)) = overall_scale {
                let scale = challenge_power(&state.challenges, ch, *pow);
                for r in &mut result {
                    *r *= scale;
                }
            }
            let _ = device_buffers.insert(*output, DeviceBuffer::Field(backend.upload(&result)));
        }
    }
}

/// Compute `base ^ power` from challenge slots.
fn challenge_power<F: Field>(challenges: &[F], ch: &ChallengeIdx, power: u8) -> F {
    let base = challenges[ch.0];
    let mut result = F::one();
    for _ in 0..power {
        result *= base;
    }
    result
}

/// Resolve a polynomial source from device buffers (if present) or provider.
fn resolve_source<B: ComputeBackend, F: Field>(
    poly: &PolynomialId,
    device_buffers: &HashMap<PolynomialId, Buf<B, F>>,
    provider: &impl BufferProvider<F>,
    backend: &B,
) -> Vec<F> {
    if let Some(buf) = device_buffers.get(poly) {
        backend.download(buf.as_field())
    } else {
        provider.materialize(*poly).into_owned()
    }
}

/// Accumulate `scale × data[i]` into `result[i]`.
fn accumulate<F: Field>(result: &mut [F], data: &[F], scale: F) {
    for (r, &d) in result.iter_mut().zip(data.iter()) {
        *r += scale * d;
    }
}

/// Separate interleaved x/y bits: odd positions → x (lo), even positions → y (ro).
fn uninterleave_u128(bits: u128) -> (u64, u64) {
    let mut x: u64 = 0;
    let mut y: u64 = 0;
    for i in 0..64 {
        y |= (((bits >> (2 * i)) & 1) as u64) << i;
        x |= (((bits >> (2 * i + 1)) & 1) as u64) << i;
    }
    (x, y)
}
