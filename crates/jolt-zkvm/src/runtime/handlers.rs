use std::collections::{HashMap, HashSet};

use jolt_compiler::module::{DomainSeparator, EvalMode, Op, RoundPolyEncoding};
use jolt_compiler::PolynomialId;
use jolt_compute::{
    Buf, BufferProvider, ComputeBackend, DeviceBuffer, Executable, LookupTraceData,
};
use jolt_crypto::HomomorphicCommitment;
use jolt_field::Field;
use jolt_openings::AdditivelyHomomorphic;
use jolt_poly::UnivariatePoly;
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript};

use super::helpers::{
    bind_kernel_inputs, build_outer_eq, fused_rlc_reduce, materialize_binding, PendingClaim,
};

/// Dispatch a single Op from the schedule.
#[allow(clippy::too_many_arguments)]
pub(super) fn dispatch_op<B, F, T, PCS>(
    op: &Op,
    state: &mut super::RuntimeState<B, F, PCS>,
    device_buffers: &mut HashMap<PolynomialId, Buf<B, F>>,
    executable: &Executable<B, F>,
    provider: &mut impl BufferProvider<F>,
    backend: &B,
    pcs_setup: &PCS::ProverSetup,
    transcript: &mut T,
    stage_point_indices: &[Vec<usize>],
    bytecode_data: Option<&jolt_witness::bytecode_raf::BytecodeData<F>>,
    lookup_trace: Option<&LookupTraceData>,
) where
    B: ComputeBackend,
    F: Field,
    T: Transcript<Challenge = F>,
    PCS: AdditivelyHomomorphic<Field = F>,
    PCS::Output: AppendToTranscript + HomomorphicCommitment<F>,
{
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
            // jolt-core skips advice commits when data is empty/zero
            let skip = matches!(
                tag,
                DomainSeparator::UntrustedAdvice | DomainSeparator::TrustedAdvice
            ) && polys.iter().all(|pi| {
                let raw = provider.materialize(*pi);
                raw.iter().all(|v| *v == F::zero())
            });
            if skip {
                return;
            }

            let target_len = 1 << num_vars;
            for pi in polys {
                let raw = provider.materialize(*pi);
                let data = if raw.len() < target_len {
                    let mut v = raw.into_owned();
                    v.resize(target_len, F::zero());
                    std::borrow::Cow::Owned(v)
                } else {
                    raw
                };
                let (commitment, hint) = PCS::commit(&*data, pcs_setup);
                // Match jolt-core's append_serializable: LabelWithCount header + body
                transcript.append(&LabelWithCount(tag.as_bytes(), commitment.serialized_len()));
                commitment.append_to_transcript(transcript);
                let _ = state.hints.insert(*pi, hint);
                state.commitments.push(commitment);
            }
        }

        Op::ReduceOpenings => {
            let pending = std::mem::take(&mut state.pending_claims);
            let hints = std::mem::take(&mut state.pending_hints);

            let (claims, combined_hints) = fused_rlc_reduce::<_, PCS>(
                pending,
                hints,
                provider,
                &state.padded_poly_data,
                transcript,
            );

            state.reduced_claims = claims;
            state.reduced_hints = combined_hints;
        }

        Op::Open => {
            for (claim, hint) in state.reduced_claims.iter().zip(state.reduced_hints.iter()) {
                let poly: PCS::Polynomial = claim.polynomial.evaluations().to_vec().into();
                let proof = PCS::open(
                    &poly,
                    &claim.point,
                    claim.eval,
                    pcs_setup,
                    Some(hint.clone()),
                    transcript,
                );
                state.opening_proofs.push(proof);
            }
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
            let joint_eval = state
                .reduced_claims
                .first()
                .map_or_else(F::zero, |c| c.eval);
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
            let buf =
                materialize_binding(binding, &state.challenges, provider, backend, bytecode_data);
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
            let buf =
                materialize_binding(binding, &state.challenges, provider, backend, bytecode_data);
            let _ = device_buffers.insert(pi, buf);
        }

        Op::MaterializeIfAbsent { binding } => {
            let pi = binding.poly();
            if device_buffers.contains_key(&pi) {
                return;
            }
            let buf =
                materialize_binding(binding, &state.challenges, provider, backend, bytecode_data);
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
            let inst_evals = backend.segmented_reduce(
                compiled_kernel,
                &input_bufs,
                outer_eq,
                &segmented.inner_only,
                inner_size,
                &state.challenges,
            );
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

        Op::UnifiedInstanceInit {
            batch,
            instance,
            config,
        } => {
            let is = backend.instance_init(
                config,
                &state.challenges,
                provider,
                lookup_trace,
                &module.prover.kernels,
            );
            let _ = state.instance_states.insert((batch.0, instance.0), is);
        }

        Op::UnifiedInstanceBind {
            batch,
            instance,
            challenge,
        } => {
            let scalar = state.challenges[challenge.0];
            let is = state
                .instance_states
                .get_mut(&(batch.0, instance.0))
                .expect("UnifiedInstanceBind: state missing");
            backend.instance_bind(is, scalar);
        }

        Op::UnifiedInstanceReduce { batch, instance } => {
            let is = state
                .instance_states
                .get(&(batch.0, instance.0))
                .expect("UnifiedInstanceReduce: state missing");
            let previous_claim = state.batch_instance_claims[batch.0][instance.0];
            let evals = backend.instance_reduce(is, previous_claim);
            state.last_round_instance_evals[instance.0].clone_from(&evals);
        }

        Op::UnifiedInstanceFinalize {
            batch,
            instance,
            output_buffers,
            output_evals,
        } => {
            let is = state
                .instance_states
                .remove(&(batch.0, instance.0))
                .expect("UnifiedInstanceFinalize: state missing");
            let output = backend.instance_finalize(is);
            assert_eq!(output.buffers.len(), output_buffers.len());
            assert_eq!(output.evaluations.len(), output_evals.len());
            for (poly_id, buf) in output_buffers.iter().zip(output.buffers) {
                let _ = device_buffers.insert(*poly_id, DeviceBuffer::Field(buf));
            }
            for (poly_id, val) in output_evals.iter().zip(output.evaluations) {
                let _ = state.evaluations.insert(*poly_id, val);
            }
        }
    }
}
