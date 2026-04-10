//! Prover runtime: execute a linked schedule to produce a proof.
//!
//! [`execute`] walks the complete [`Op`] sequence — compute, PCS, and
//! orchestration ops — and returns a [`JoltProof`]. It dispatches to:
//! - [`ComputeBackend`] for polynomial arithmetic (sumcheck, bind, evaluate)
//! - [`CommitmentScheme`] for cryptographic ops (commit, reduce, open)
//! - Direct calls for orchestration (transcript absorb/squeeze, lifecycle)
//!
//! ```text
//! Protocol → compile() → Module → link(backend) → Executable<B,F>
//!                                                       │
//!                                          execute(exe, provider, backend, pcs, transcript)
//!                                                       │
//!                                                       ▼
//!                                                 JoltProof<F, PCS>
//! ```

use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use jolt_compiler::module::{
    ChallengeSource, ClaimFactor, ClaimFormula, DomainSeparator, InputBinding, Op,
    SegmentedConfig, VerifierStageIndex,
};
use jolt_compiler::{Iteration, KernelDef, PolynomialId};
use jolt_compute::{Buf, BufferProvider, ComputeBackend, DeviceBuffer, Executable};
use jolt_field::Field;
use jolt_crypto::HomomorphicCommitment;
use jolt_openings::{AdditivelyHomomorphic, ProverClaim};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::proof::SumcheckProof;
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript};
use jolt_verifier::proof::{JoltProof, StageProof};
use jolt_verifier::ProverConfig;

use crate::prefix_suffix::{LookupTraceData, PrefixSuffixState};

/// Per-stage proof being incrementally built.
struct StageBuilder<F: Field> {
    round_polys: Vec<UnivariatePoly<F>>,
    evals: Vec<F>,
}

impl<F: Field> StageBuilder<F> {
    fn new() -> Self {
        Self {
            round_polys: Vec::new(),
            evals: Vec::new(),
        }
    }

    fn finalize(self) -> StageProof<F> {
        StageProof {
            round_polys: SumcheckProof {
                round_polynomials: self.round_polys,
            },
            evals: self.evals,
        }
    }
}

/// Lightweight opening claim — defers the expensive evaluation table copy
/// until `ReduceOpenings` where it's actually needed for RLC combination.
struct PendingClaim<F: Field> {
    poly: PolynomialId,
    point: Vec<F>,
    eval: F,
}

/// Mutable state accumulated during schedule execution.
struct RuntimeState<F: Field, PCS: AdditivelyHomomorphic<Field = F>>
where
    PCS::Output: HomomorphicCommitment<F>,
{
    // ── Compute state ──
    challenges: Vec<F>,
    evaluations: HashMap<PolynomialId, F>,
    last_round_coeffs: Vec<F>,
    /// Last interpolated round polynomial (for scalar poly evaluation).
    last_round_poly: Option<UnivariatePoly<F>>,
    /// Most recently squeezed challenge value.
    last_squeezed: F,

    // ── Batched sumcheck state ──
    /// Per-batch → per-instance running claims.
    batch_instance_claims: Vec<Vec<F>>,
    /// Per-instance evaluations from the last round, used to update claims
    /// when the next round's challenge arrives. `[inst_idx] → [eval_0, eval_1, ...]`
    last_round_instance_evals: Vec<Vec<F>>,
    /// Outer eq tables for segmented instances: `(batch, inst) → eq_outer`.
    /// Built once at phase start, used for weighting during segmented reduce.
    segmented_outer_eqs: HashMap<(usize, usize), Vec<F>>,
    /// PrefixSuffix state for instances using PrefixSuffix iteration.
    /// Key: `(batch_idx, instance_idx)`.
    prefix_suffix_states: HashMap<(usize, usize), PrefixSuffixState<F>>,

    // ── Proof assembly (incremental) ──
    current_stage: Option<StageBuilder<F>>,
    stage_proofs: Vec<StageProof<F>>,

    // ── PCS state ──
    commitments: Vec<PCS::Output>,
    hints: HashMap<PolynomialId, PCS::OpeningHint>,
    pending_claims: Vec<PendingClaim<F>>,
    pending_hints: Vec<PCS::OpeningHint>,
    reduced_claims: Vec<ProverClaim<F>>,
    reduced_hints: Vec<PCS::OpeningHint>,
    opening_proofs: Vec<PCS::Proof>,
}

/// Remove kernel input buffers from cache, bind them at `scalar`, and reinsert.
fn bind_kernel_inputs<B: ComputeBackend, F: Field>(
    device_buffers: &mut HashMap<PolynomialId, Buf<B, F>>,
    backend: &B,
    compiled_kernel: &B::CompiledKernel<F>,
    kdef: &KernelDef,
    scalar: F,
) {
    let mut input_bufs: Vec<Buf<B, F>> = kdef
        .inputs
        .iter()
        .map(|b| {
            device_buffers
                .remove(&b.poly())
                .expect("bind_kernel_inputs: input buffer missing")
        })
        .collect();
    backend.bind(compiled_kernel, &mut input_bufs, scalar);
    for (buf, binding) in input_bufs.into_iter().zip(&kdef.inputs) {
        let _ = device_buffers.insert(binding.poly(), buf);
    }
}

/// Like [`bind_kernel_inputs`] but skips poly IDs already in `bound`.
/// After binding, newly-bound poly IDs are added to `bound`.
///
/// When multiple instances in a batched sumcheck share the same underlying
/// polynomial buffer, each buffer must only be bound (halved) once per round.
/// Without tracking, instance 0's bind halves a shared buffer, then instance 1
/// halves it again — causing an index-out-of-bounds crash on subsequent rounds.
fn bind_kernel_inputs_tracked<B: ComputeBackend, F: Field>(
    device_buffers: &mut HashMap<PolynomialId, Buf<B, F>>,
    backend: &B,
    kdef: &KernelDef,
    scalar: F,
    bound: &mut HashSet<PolynomialId>,
) {
    let order = kdef.spec.binding_order;
    let mut seen = HashSet::new();
    for b in &kdef.inputs {
        let pid = b.poly();
        if bound.contains(&pid) || !seen.insert(pid) {
            continue;
        }
        if let Some(buf) = device_buffers.get_mut(&pid) {
            backend.interpolate_inplace(buf.as_field_mut(), scalar, order);
        }
        let _ = bound.insert(pid);
    }
}

/// Execute the full prover schedule and return a complete proof.
///
/// Walks every op in the schedule, dispatching compute ops to `backend`,
/// PCS ops to the commitment scheme, and orchestration ops directly.
#[allow(clippy::print_stderr, clippy::too_many_arguments)]
pub(crate) fn execute<B, F, T, PCS>(
    executable: &Executable<B, F>,
    provider: &mut impl BufferProvider<F>,
    backend: &B,
    pcs_setup: &PCS::ProverSetup,
    transcript: &mut T,
    config: ProverConfig,
    lookup_trace: Option<LookupTraceData>,
    bytecode_data: Option<crate::bytecode_raf::BytecodeData<F>>,
) -> JoltProof<F, PCS>
where
    B: ComputeBackend,
    F: Field,
    T: Transcript<Challenge = F>,
    PCS: AdditivelyHomomorphic<Field = F>,
    PCS::Output: AppendToTranscript + HomomorphicCommitment<F>,
{
    let module = &executable.module;

    // Pre-allocate per-batch instance claim vectors.
    let batch_instance_claims: Vec<Vec<F>> = module
        .prover
        .batched_sumchecks
        .iter()
        .map(|b| vec![F::zero(); b.instances.len()])
        .collect();

    let mut state = RuntimeState::<F, PCS> {
        challenges: vec![F::zero(); module.challenges.len()],
        evaluations: HashMap::new(),
        last_round_coeffs: Vec::new(),
        last_round_poly: None,
        last_squeezed: F::zero(),
        batch_instance_claims,
        last_round_instance_evals: Vec::new(),
        segmented_outer_eqs: HashMap::new(),
        prefix_suffix_states: HashMap::new(),
        current_stage: None,
        stage_proofs: Vec::new(),
        commitments: Vec::new(),
        hints: HashMap::new(),
        pending_claims: Vec::new(),
        pending_hints: Vec::new(),
        reduced_claims: Vec::new(),
        reduced_hints: Vec::new(),
        opening_proofs: Vec::new(),
    };

    // Precompute: verifier stage → sorted round challenge indices.
    let stage_point_indices: Vec<Vec<usize>> = precompute_stage_points(module);

    // Device buffer cache — compute ops work on backend buffers.
    let mut device_buffers: HashMap<PolynomialId, Buf<B, F>> = HashMap::new();

    for op in &executable.ops {
        match op {
            // ── Compute ──
            Op::SumcheckRound {
                kernel,
                round: _,
                bind_challenge,
            } => {
                let kdef = &module.prover.kernels[*kernel];
                let compiled_kernel = &executable.kernels[*kernel];

                resolve_inputs(
                    &mut device_buffers,
                    &state.challenges,
                    kdef,
                    provider,
                    backend,
                    false,
                    bytecode_data.as_ref(),
                );

                // Bind (rounds 1+): take buffers out, bind via kernel-aware
                // dispatch, put back. The compiled kernel determines the
                // binding strategy (dense, tensor, sparse).
                if let Some(ch) = bind_challenge {
                    let scalar = state.challenges[*ch];
                    bind_kernel_inputs(
                        &mut device_buffers,
                        backend,
                        compiled_kernel,
                        kdef,
                        scalar,
                    );
                }

                let input_refs: Vec<&Buf<B, F>> = kdef
                    .inputs
                    .iter()
                    .filter_map(|b| device_buffers.get(&b.poly()))
                    .collect();

                #[cfg(debug_assertions)]
                if matches!(kdef.spec.iteration, Iteration::Dense)
                    && bind_challenge.is_none()
                {
                    // Dump buffer stats for the first Dense round (remaining round 0)
                    for (j, b) in input_refs.iter().enumerate() {
                        let data = backend.download(b.as_field());
                        let s: F = data.iter().copied().sum();
                        eprintln!(
                            "[remaining r0] buf[{j}] len={} sum={s:?} first4=[{:?}, {:?}, {:?}, {:?}]",
                            data.len(),
                            data.first().unwrap_or(&F::zero()),
                            data.get(1).unwrap_or(&F::zero()),
                            data.get(2).unwrap_or(&F::zero()),
                            data.get(3).unwrap_or(&F::zero()),
                        );
                    }
                }

                state.last_round_coeffs =
                    backend.reduce(compiled_kernel, &input_refs, &state.challenges);
            }

            Op::BatchedSumcheckRound {
                batch,
                round,
                bind_challenge,
            } => {
                let bdef = &module.prover.batched_sumchecks[*batch];
                let max_evals = bdef.max_degree + 1;
                let mut combined = vec![F::zero(); max_evals];
                let mut bound_this_round: HashSet<PolynomialId> = HashSet::new();

                // Update per-instance claims from the previous round's evals
                // using the bind challenge (= previous round's squeezed challenge).
                if let Some(ch) = bind_challenge {
                    let r = state.challenges[*ch];
                    for (inst_idx, evals) in state.last_round_instance_evals.iter().enumerate() {
                        if !evals.is_empty() {
                            let points: Vec<(F, F)> = evals
                                .iter()
                                .enumerate()
                                .map(|(s, &v)| (F::from_u64(s as u64), v))
                                .collect();
                            let poly = UnivariatePoly::interpolate(&points);
                            state.batch_instance_claims[*batch][inst_idx] = poly.evaluate(r);
                        }
                    }
                }
                state.last_round_instance_evals = vec![Vec::new(); bdef.instances.len()];

                for (inst_idx, inst) in bdef.instances.iter().enumerate() {
                    let coeff = state.challenges[inst.batch_coeff];

                    if *round < inst.first_active_round {
                        // Inactive: constant contribution claim/2 to all eval slots.
                        let two_inv = F::from_u64(2).inverse().unwrap();
                        let half_claim =
                            state.batch_instance_claims[*batch][inst_idx] * two_inv;
                        #[cfg(debug_assertions)]
                        eprintln!(
                            "  [zkvm round={round} inst={inst_idx} INACTIVE] half_claim={half_claim:?}, coeff={coeff:?}"
                        );
                        for slot in &mut combined {
                            *slot += coeff * half_claim;
                        }
                        state.batch_instance_claims[*batch][inst_idx] = half_claim;
                        continue;
                    }

                    // Determine the current phase within this instance.
                    let instance_round = *round - inst.first_active_round;
                    let (phase_idx, phase_start) = inst.phase_for_round(instance_round);
                    let phase = &inst.phases[phase_idx];
                    let kdef = &module.prover.kernels[phase.kernel];
                    let compiled_kernel = &executable.kernels[phase.kernel];

                    let is_prefix_suffix =
                        matches!(kdef.spec.iteration, Iteration::PrefixSuffix { .. });

                    if instance_round == 0 || instance_round == phase_start {
                        // At a phase boundary, bind previous phase's buffers.
                        if instance_round > 0 {
                            if let Some(ch) = bind_challenge {
                                let scalar = state.challenges[*ch];
                                let prev_phase = &inst.phases[phase_idx - 1];
                                let prev_kdef =
                                    &module.prover.kernels[prev_phase.kernel];
                                let prev_is_ps = matches!(
                                    prev_kdef.spec.iteration,
                                    Iteration::PrefixSuffix { .. }
                                );
                                if prev_is_ps {
                                    // PrefixSuffix→next phase transition:
                                    // ingest last address challenge, then materialize outputs.
                                    let ps_key = (*batch, inst_idx);
                                    if let Some(ps) = state.prefix_suffix_states.get_mut(&ps_key) {
                                        let addr_round = ps.total_round();
                                        ps.ingest_challenge(scalar, addr_round);
                                        let outputs = ps.materialize_outputs();
                                        for (poly_id, data) in outputs {
                                            let buf = DeviceBuffer::Field(backend.upload(&data));
                                            let _ = device_buffers.insert(poly_id, buf);
                                        }
                                    }
                                    let _ = state.prefix_suffix_states.remove(&ps_key);
                                } else {
                                    bind_kernel_inputs_tracked(
                                        &mut device_buffers,
                                        backend,
                                        prev_kdef,
                                        scalar,
                                        &mut bound_this_round,
                                    );
                                }
                            }
                        }

                        for cap in &phase.scalar_captures {
                            let val = device_buffers
                                .get(&cap.poly)
                                .map(|buf| {
                                    let data = backend.download(buf.as_field());
                                    assert!(
                                        data.len() == 1,
                                        "ScalarCapture: expected 1-element buffer for {:?}, got {}",
                                        cap.poly,
                                        data.len()
                                    );
                                    data[0]
                                })
                                .expect("ScalarCapture: buffer not found");
                            state.challenges[cap.challenge] = val;
                        }

                        if is_prefix_suffix {
                            // Create PrefixSuffix state on activation.
                            let trace = lookup_trace.as_ref().expect(
                                "PrefixSuffix iteration requires lookup_trace data"
                            );
                            let ps = PrefixSuffixState::new(
                                &kdef.spec.iteration,
                                &state.challenges,
                                trace,
                            );
                            let _ = state.prefix_suffix_states
                                .insert((*batch, inst_idx), ps);
                        } else {
                            // Build outer eq table for segmented phases.
                            if let Some(seg) = &phase.segmented {
                                let outer_eq =
                                    build_outer_eq(&state.challenges, seg, backend);
                                let _ = state
                                    .segmented_outer_eqs
                                    .insert((*batch, inst_idx), outer_eq);
                            }

                            // Force-refresh on instance activation (first active round)
                            // to replace stale buffers from prior stages.
                            let force = instance_round == 0;
                            resolve_inputs(
                                &mut device_buffers,
                                &state.challenges,
                                kdef,
                                provider,
                                backend,
                                force,
                                bytecode_data.as_ref(),
                            );
                        }
                    } else if let Some(ch) = bind_challenge {
                        let scalar = state.challenges[*ch];
                        if is_prefix_suffix {
                            // PrefixSuffix bind: ingest challenge into PS state.
                            let ps_key = (*batch, inst_idx);
                            if let Some(ps) = state.prefix_suffix_states.get_mut(&ps_key) {
                                let addr_round = ps.total_round();
                                ps.ingest_challenge(scalar, addr_round);
                            }
                        } else {
                            bind_kernel_inputs_tracked(
                                &mut device_buffers,
                                backend,
                                kdef,
                                scalar,
                                &mut bound_this_round,
                            );
                        }
                    }

                    // Reduce: PrefixSuffix, segmented, or standard.
                    let round_within_phase = instance_round - phase_start;
                    let inst_evals = if is_prefix_suffix {
                        let ps_key = (*batch, inst_idx);
                        let ps = state.prefix_suffix_states.get_mut(&ps_key)
                            .expect("PrefixSuffix state missing");
                        let [eval_0, eval_2] = ps.compute_address_round();
                        // PrefixSuffix is degree 2 → 3 evals at {0, 1, 2}
                        // eval_1 = previous_claim - eval_0 (from sumcheck relation)
                        // We return [eval_0, eval_2] and let the caller build the UniPoly.
                        // Actually, the runtime expects raw evaluation sums at grid points
                        // {0, 1, ..., num_evals-1}. For degree 2, num_evals = 3.
                        // eval_1 = previous_claim - eval_0
                        let previous_claim =
                            state.batch_instance_claims[*batch][inst_idx];
                        let eval_1 = previous_claim - eval_0;
                        vec![eval_0, eval_1, eval_2]
                    } else if let Some(seg) = &phase.segmented {
                        let outer_eq = state
                            .segmented_outer_eqs
                            .get(&(*batch, inst_idx))
                            .expect("segmented outer eq missing");
                        segmented_reduce(
                            &device_buffers,
                            outer_eq,
                            seg,
                            kdef,
                            compiled_kernel,
                            &state.challenges,
                            backend,
                            round_within_phase,
                        )
                    } else {
                        let input_refs: Vec<&Buf<B, F>> = kdef
                            .inputs
                            .iter()
                            .filter_map(|b| device_buffers.get(&b.poly()))
                            .collect();
                        backend.reduce(compiled_kernel, &input_refs, &state.challenges)
                    };

                    #[cfg(debug_assertions)]
                    {
                        eprintln!(
                            "  [zkvm round={round} inst={inst_idx} ACTIVE] raw_evals={inst_evals:?}"
                        );
                        // Dump first 8 elements of each input at first active round.
                        if instance_round == 0 && inst_idx == 1 {
                            for (j, binding) in kdef.inputs.iter().enumerate() {
                                if let Some(buf) = device_buffers.get(&binding.poly()) {
                                    let data = backend.download(buf.as_field());
                                    let n = data.len().min(8);
                                    eprintln!(
                                        "  [zkvm DUMP inst={inst_idx} input={j} ({:?})] len={}, first {n}: {:?}",
                                        binding.poly(), data.len(), &data[..n]
                                    );
                                }
                            }
                        }
                    }

                    // Save per-instance evals for claim update in next round.
                    state.last_round_instance_evals[inst_idx].clone_from(&inst_evals);

                    // Extrapolate: if this instance has fewer evals than
                    // max_evals (lower-degree kernel), fill the remaining
                    // slots by polynomial interpolation so the combined
                    // round polynomial is correct at ALL evaluation points.
                    let mut full_evals = inst_evals.clone();
                    if full_evals.len() < max_evals {
                        let points: Vec<(F, F)> = full_evals
                            .iter()
                            .enumerate()
                            .map(|(s, &v)| (F::from_u64(s as u64), v))
                            .collect();
                        let poly = UnivariatePoly::interpolate(&points);
                        for s in full_evals.len()..max_evals {
                            full_evals.push(poly.evaluate(F::from_u64(s as u64)));
                        }
                    }

                    for (i, &v) in full_evals.iter().enumerate() {
                        combined[i] += coeff * v;
                    }
                }
                #[cfg(debug_assertions)]
                {
                    eprintln!("[zkvm batch={batch} round={round}] combined evals:");
                    for (i, v) in combined.iter().enumerate() {
                        eprintln!("  combined[{i}] = {v:?}");
                    }
                }
                state.last_round_coeffs = combined;
            }

            Op::Evaluate { poly } => {
                if let Some(buf) = device_buffers.get(poly) {
                    let data = backend.download(buf.as_field());
                    #[cfg(debug_assertions)]
                    eprintln!("[zkvm eval] {poly:?} buf_len={}", data.len());
                    let val = match data.len() {
                        0 => continue,
                        1 => data[0],
                        // After a sumcheck with n rounds, n-1 binds leave
                        // 2-element buffers. Final evaluation = linear
                        // interpolation at the last squeezed challenge.
                        2 => {
                            let r = state.last_squeezed;
                            data[0] + r * (data[1] - data[0])
                        }
                        n => {
                            panic!(
                                "Evaluate: {poly:?} has {n}-element buffer; \
                                 expected 1 (fully bound) or 2 (final interpolation)"
                            )
                        }
                    };
                    let _ = state.evaluations.insert(*poly, val);
                } else if let Some(round_poly) = &state.last_round_poly {
                    let val = round_poly.evaluate(state.last_squeezed);
                    let _ = state.evaluations.insert(*poly, val);
                }
            }

            Op::Bind {
                polys,
                challenge,
                order,
            } => {
                let scalar = state.challenges[*challenge];
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
                let r = state.challenges[*challenge];
                let basis = jolt_poly::lagrange::lagrange_evals(*domain_start, *domain_size, r);
                let num_groups = group_offsets.len();

                // Lagrange kernel scale: L(τ, r) = Σ_k L_k(τ) · L_k(r)
                let scale = if let Some(tau_idx) = kernel_tau {
                    let tau = state.challenges[*tau_idx];
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
                    let data = backend.download(buf.as_field());
                    let num_cycles = data.len() / stride;

                    let mut projected = vec![F::zero(); num_cycles * num_groups];
                    for c in 0..num_cycles {
                        for (g, &offset) in group_offsets.iter().enumerate() {
                            let mut acc = F::zero();
                            for (k, &lk) in basis.iter().enumerate() {
                                let idx = c * stride + offset + k;
                                if idx < data.len() {
                                    acc += lk * data[idx];
                                }
                            }
                            projected[c * num_groups + g] = acc;
                        }
                    }

                    // Apply kernel scale to first poly only (avoids squaring the factor).
                    if *pi == polys[0] {
                        for v in &mut projected {
                            *v *= scale;
                        }
                    }

                    let new_buf = backend.upload(&projected);
                    let _ = device_buffers.insert(*pi, DeviceBuffer::Field(new_buf));
                }
            }

            Op::DuplicateInterleave { polys } => {
                for pi in polys {
                    let buf = device_buffers
                        .remove(pi)
                        .expect("DuplicateInterleave: buffer missing");
                    let expanded = DeviceBuffer::Field(
                        backend.duplicate_interleave(buf.as_field()),
                    );
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

            // ── PCS ──
            Op::Commit { polys, tag, num_vars }
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
                    continue;
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
                    transcript.append(&LabelWithCount(
                        tag.as_bytes(),
                        commitment.serialized_len(),
                    ));
                    commitment.append_to_transcript(transcript);
                    let _ = state.hints.insert(*pi, hint);
                    state.commitments.push(commitment);
                }
            }

            Op::ReduceOpenings => {
                let pending = std::mem::take(&mut state.pending_claims);
                let hints = std::mem::take(&mut state.pending_hints);

                let (claims, combined_hints) =
                    fused_rlc_reduce::<_, PCS>(pending, hints, provider, transcript);

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

            // ── Orchestration ──
            Op::Preamble => {
                transcript.append(&config);
            }

            Op::BeginStage { .. } => {
                if let Some(builder) = state.current_stage.take() {
                    state.stage_proofs.push(builder.finalize());
                }
                state.current_stage = Some(StageBuilder::new());
            }

            Op::AbsorbRoundPoly {
                kernel,
                num_coeffs,
                tag,
            } => {
                let kdef = &module.prover.kernels[*kernel];
                let coeffs = if let Iteration::Domain {
                    domain_size,
                    domain_start,
                    tau_challenge,
                    zero_base,
                    ..
                } = &kdef.spec.iteration
                {
                    // Uniskip post-processing: the kernel returned 2K-1
                    // evaluations of the composition polynomial t1(Y).
                    // We need to interpolate, convolve with the Lagrange
                    // kernel polynomial, and extract the final coefficients.
                    let k = *domain_size;
                    let mut raw_evals = state.last_round_coeffs.clone();
                    debug_assert_eq!(raw_evals.len(), 2 * k - 1);

                    #[cfg(debug_assertions)]
                    {
                        eprintln!("[uniskip] raw_evals BEFORE zeroing (K={k}, domain_start={domain_start}, zero_base={zero_base}):");
                        for (i, v) in raw_evals.iter().enumerate() {
                            let y = *domain_start + i as i64;
                            eprintln!("  raw[{i}] (y={y}): {v:?}");
                        }
                    }

                    // Only zero base evaluations when the formula guarantees
                    // vanishing (R1CS: Az*Bz = Cz). Product/other uniskip
                    // rounds have non-zero base evaluations.
                    if *zero_base {
                        for v in raw_evals.iter_mut().take(k) {
                            *v = F::zero();
                        }
                    }

                    // 1. Interpolate t1(Y) from 2K-1 evaluations on
                    //    {domain_start, ..., domain_start + 2K - 2}.
                    let t1_coeffs =
                        jolt_poly::lagrange::interpolate_to_coeffs(*domain_start, &raw_evals);
                    #[cfg(debug_assertions)]
                    {
                        eprintln!("[uniskip] t1 coefficients ({} terms):", t1_coeffs.len());
                        for (i, c) in t1_coeffs.iter().enumerate() {
                            eprintln!("  t1[{i}]: {c:?}");
                        }
                    }

                    // 2. Build Lagrange kernel L(τ_high, Y) over the base
                    //    domain {domain_start, ..., domain_start + K - 1}.
                    let tau_high = state.challenges[*tau_challenge];
                    let basis_at_tau =
                        jolt_poly::lagrange::lagrange_evals(*domain_start, k, tau_high);
                    let lagrange_coeffs =
                        jolt_poly::lagrange::interpolate_to_coeffs(*domain_start, &basis_at_tau);
                    #[cfg(debug_assertions)]
                    {
                        eprintln!("[uniskip] Lagrange kernel coefficients ({} terms):", lagrange_coeffs.len());
                        for (i, c) in lagrange_coeffs.iter().enumerate() {
                            eprintln!("  L[{i}]: {c:?}");
                        }
                    }

                    // 3. Convolve: s1(Y) = t1(Y) × L(τ_high, Y).
                    //    Degree 2(K-1) + (K-1) = 3(K-1), so 3K-2 coefficients.
                    let mut s1 = jolt_poly::lagrange::poly_mul(&t1_coeffs, &lagrange_coeffs);
                    s1.resize(*num_coeffs, F::zero());
                    #[cfg(debug_assertions)]
                    {
                        eprintln!("[uniskip] s1 coefficients ({} terms, num_coeffs={num_coeffs}):", s1.len());
                        for (i, c) in s1.iter().enumerate() {
                            eprintln!("  s1[{i}]: {c:?}");
                        }
                    }
                    s1
                } else {
                    // Standard: evaluations at {0, 1, ..., num_coeffs - 1}
                    // → interpolate to monomial coefficients.
                    let evals = &state.last_round_coeffs[..*num_coeffs];
                    #[cfg(debug_assertions)]
                    {
                        eprintln!("[remaining round] kernel={kernel} raw evals:");
                        for (i, v) in evals.iter().enumerate() {
                            eprintln!("  eval[{i}] = {v:?}");
                        }
                        eprintln!("  sum(0+1) = {:?}", evals[0] + evals[1]);
                    }
                    let points: Vec<(F, F)> = evals
                        .iter()
                        .enumerate()
                        .map(|(slot, &val)| (F::from_u64(slot as u64), val))
                        .collect();
                    UnivariatePoly::interpolate(&points).into_coefficients()
                };

                let is_uniskip = matches!(
                    &module.prover.kernels[*kernel].spec.iteration,
                    Iteration::Domain { .. }
                );

                if is_uniskip {
                    // Uniskip: send full polynomial (matches jolt-core's uniskip_poly)
                    transcript.append(&LabelWithCount(tag.as_bytes(), coeffs.len() as u64));
                    for c in &coeffs {
                        transcript.append(c);
                    }
                } else {
                    // Standard sumcheck: send compressed polynomial (skip c1)
                    // to match jolt-core's CompressedUniPoly encoding.
                    // compressed = [c0, c2, c3, ...] (coeffs.len() - 1 elements)
                    let compressed_len = coeffs.len() - 1;
                    transcript.append(&LabelWithCount(tag.as_bytes(), compressed_len as u64));
                    transcript.append(&coeffs[0]); // c0
                    for c in &coeffs[2..] {
                        transcript.append(c); // c2, c3, ...
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
            } => {
                let val = evaluate_claim(formula, &state.evaluations, &state.challenges);
                #[cfg(debug_assertions)]
                {
                    eprintln!("[AbsorbInputClaim batch={batch} inst={instance}] val={val:?}");
                    for term in &formula.terms {
                        let mut prod = F::from_i128(term.coeff);
                        for f in &term.factors {
                            match f {
                                ClaimFactor::Eval(poly) => {
                                    let e = state.evaluations.get(poly).copied().unwrap_or(F::zero());
                                    eprintln!("    Eval({poly:?}) = {e:?}");
                                    prod *= e;
                                }
                                ClaimFactor::Challenge(i) => {
                                    eprintln!("    Challenge({i}) = {:?}", state.challenges[*i]);
                                    prod *= state.challenges[*i];
                                }
                                _ => {}
                            }
                        }
                        eprintln!("    term product = {prod:?}");
                    }
                }
                transcript.append(&Label(tag.as_bytes()));
                transcript.append(&val);
                // Store input claim for inactive-round scaling.
                // The runtime halves this each inactive round, so it must
                // start at input_claim * 2^(max_rounds - num_rounds).
                let bdef = &module.prover.batched_sumchecks[*batch];
                let inst = &bdef.instances[*instance];
                let offset = bdef.max_rounds - inst.num_rounds();
                // Multiply by 2^offset using repeated doubling (safe for large offsets).
                let mut scaled = val;
                let two = F::from_u64(2);
                for _ in 0..offset {
                    scaled *= two;
                }
                state.batch_instance_claims[*batch][*instance] = scaled;
            }

            Op::Squeeze { challenge } => {
                let val = transcript.challenge();
                state.challenges[*challenge] = val;
                state.last_squeezed = val;
            }

            Op::AppendDomainSeparator { tag } => {
                // Match jolt-core's `append_bytes(label, &[])` which calls:
                //   1. raw_append_label_with_len: pack label (24b) + BE length 0 (8b) = 32b → update_state
                //   2. raw_append_bytes(&[]): hash empty bytes → update_state (increments n_rounds)
                let label = tag.as_bytes();
                let mut packed = [0u8; 32];
                packed[..label.len()].copy_from_slice(label);
                transcript.append_bytes(&packed);
                transcript.append_bytes(&[]);
            }

            Op::EvaluatePreprocessed { source, at_challenges, store_as } => {
                let data = provider.materialize(*source);
                let point: Vec<F> = at_challenges.iter().map(|&ci| state.challenges[ci]).collect();
                let eval = evaluate_mle(&data, &point);
                let _ = state.evaluations.insert(*store_as, eval);
            }

            Op::SnapshotEval { from, to } => {
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

            Op::ReleaseDevice { poly } => {
                let _ = device_buffers.remove(poly);
            }

            Op::ReleaseHost { polys } => {
                for pi in polys {
                    provider.release(*pi);
                }
            }
        }
    }

    // Finalize the last stage
    if let Some(builder) = state.current_stage.take() {
        state.stage_proofs.push(builder.finalize());
    }

    JoltProof {
        config,
        stage_proofs: state.stage_proofs,
        opening_proofs: state.opening_proofs,
        commitments: state.commitments,
    }
}

/// Fused RLC reduction: groups pending claims by point, draws one rho per group
/// from the transcript, and produces combined (claim, hint) pairs in a single pass.
///
/// Reads polynomial data directly from the provider via `as_slice` — no
/// intermediate `ProverClaim` copies. Only the final RLC-combined evaluation
/// table is allocated per group.
fn fused_rlc_reduce<F, PCS>(
    pending: Vec<PendingClaim<F>>,
    hints: Vec<PCS::OpeningHint>,
    provider: &impl BufferProvider<F>,
    transcript: &mut impl Transcript<Challenge = F>,
) -> (Vec<ProverClaim<F>>, Vec<PCS::OpeningHint>)
where
    F: Field,
    PCS: AdditivelyHomomorphic<Field = F>,
    PCS::Output: HomomorphicCommitment<F>,
{
    if pending.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Domain separation + count prefix (matches jolt-core's append_scalars(b"rlc_claims", &all))
    transcript.append(&LabelWithCount(b"rlc_claims", pending.len() as u64));
    for pc in &pending {
        pc.eval.append_to_transcript(transcript);
    }

    // Group by point (preserving insertion order).
    struct PointGroup<'a, F, H> {
        point: &'a Vec<F>,
        poly_ids: Vec<PolynomialId>,
        evals: Vec<F>,
        hints: Vec<H>,
    }

    let mut groups: Vec<PointGroup<'_, F, PCS::OpeningHint>> = Vec::new();

    for (pc, hint) in pending.iter().zip(hints) {
        if let Some(g) = groups.iter_mut().find(|g| *g.point == pc.point) {
            g.poly_ids.push(pc.poly);
            g.evals.push(pc.eval);
            g.hints.push(hint);
        } else {
            groups.push(PointGroup {
                point: &pc.point,
                poly_ids: vec![pc.poly],
                evals: vec![pc.eval],
                hints: vec![hint],
            });
        }
    }

    let mut reduced_claims = Vec::with_capacity(groups.len());
    let mut reduced_hints = Vec::with_capacity(groups.len());

    for PointGroup {
        point,
        poly_ids,
        evals,
        hints: group_hints,
    } in groups
    {
        let rho: F = transcript.challenge();

        // RLC-combine evaluation tables from materialized provider data.
        let materialized: Vec<Cow<'_, [F]>> =
            poly_ids.iter().map(|&pi| provider.materialize(pi)).collect();
        let slices: Vec<&[F]> = materialized.iter().map(|c| &**c).collect();
        let combined_evals = jolt_openings::rlc_combine(&slices, rho);
        let combined_eval = jolt_openings::rlc_combine_scalars(&evals, rho);

        // Combine hints with the same rho powers.
        let powers: Vec<F> = std::iter::successors(Some(F::from_u64(1)), |prev| Some(*prev * rho))
            .take(group_hints.len())
            .collect();
        let combined_hint = PCS::combine_hints(group_hints, &powers);

        reduced_claims.push(ProverClaim {
            polynomial: combined_evals.into(),
            point: point.clone(),
            eval: combined_eval,
        });
        reduced_hints.push(combined_hint);
    }

    (reduced_claims, reduced_hints)
}

/// Precompute verifier-stage → round challenge indices mapping.
///
/// For each verifier stage, collects the challenge indices corresponding
/// to its sumcheck rounds, sorted by round number. This avoids per-op
/// scanning of the challenge declarations during `CollectOpeningClaim`.
fn precompute_stage_points(module: &jolt_compiler::module::Module) -> Vec<Vec<usize>> {
    (0..module.verifier.num_stages)
        .map(|si| {
            let mut pairs: Vec<(usize, usize)> = module
                .challenges
                .iter()
                .enumerate()
                .filter_map(|(ci, decl)| {
                    if let ChallengeSource::SumcheckRound { stage, round } = &decl.source {
                        if *stage == VerifierStageIndex(si) {
                            return Some((*round, ci));
                        }
                    }
                    None
                })
                .collect();
            pairs.sort_unstable_by_key(|(r, _)| *r);
            pairs.iter().map(|&(_, ci)| ci).collect()
        })
        .collect()
}

/// Ensure all inputs for a kernel are loaded on-device.
fn resolve_inputs<B, F>(
    device_buffers: &mut HashMap<PolynomialId, Buf<B, F>>,
    challenges: &[F],
    kdef: &KernelDef,
    provider: &impl BufferProvider<F>,
    backend: &B,
    force_refresh: bool,
    bytecode_data: Option<&crate::bytecode_raf::BytecodeData<F>>,
) where
    B: ComputeBackend,
    F: Field,
{
    for binding in &kdef.inputs {
        let pi = binding.poly();
        // When force_refresh is set (instance activation), always reload
        // challenge-dependent inputs (EqTable, EqProject, etc.). For
        // Provided inputs, check if the buffer size matches the kernel's
        // expected round count: if the buffer has exactly 2^num_rounds
        // elements, it was freshly populated by a compute op (e.g.
        // LagrangeProject) and should be preserved. Otherwise (stale
        // remnant from a prior stage), re-materialize.
        let skip = if let Some(buf) = device_buffers.get(&pi) {
            match binding {
                InputBinding::Provided { .. } => {
                    if force_refresh {
                        let data = backend.download(buf.as_field());
                        let expected = 1usize << kdef.num_rounds;
                        data.len() == expected
                    } else {
                        true
                    }
                }
                _ => !force_refresh,
            }
        } else {
            false
        };
        if skip {
            continue;
        }
        let buf: Buf<B, F> = match binding {
            InputBinding::Provided { .. } => {
                let data = provider.materialize(pi);
                DeviceBuffer::Field(backend.upload(&data))
            }
            InputBinding::EqTable {
                challenges: chs, ..
            } => {
                let point: Vec<F> = chs.iter().map(|&ci| challenges[ci]).collect();
                DeviceBuffer::Field(backend.eq_table(&point))
            }
            InputBinding::EqPlusOneTable {
                challenges: chs, ..
            } => {
                let point: Vec<F> = chs.iter().map(|&ci| challenges[ci]).collect();
                let (_eq, eq_plus_one) = backend.eq_plus_one_table(&point);
                DeviceBuffer::Field(eq_plus_one)
            }
            InputBinding::LtTable {
                challenges: chs, ..
            } => {
                let point: Vec<F> = chs.iter().map(|&ci| challenges[ci]).collect();
                DeviceBuffer::Field(backend.lt_table(&point))
            }
            InputBinding::EqProject {
                source,
                challenges: chs,
                inner_size,
                outer_size,
                ..
            } => {
                // Source layout: inner_size rows × outer_size cols = source[t * outer_size + k].
                // The challenge point determines projection direction:
                //   - If challenges span the inner dim (len=log(inner_size)):
                //       project[k] = Σ_t eq(r, t) * source[t * outer + k]   → outer_size result
                //   - If challenges span the outer dim (len=log(outer_size)):
                //       project[t] = Σ_k eq(r, k) * source[t * outer + k]   → inner_size result
                let point: Vec<F> = chs.iter().map(|&ci| challenges[ci]).collect();
                let eq_table = jolt_poly::EqPolynomial::<F>::evals(&point, None);

                let src_data = provider.materialize(*source);
                if eq_table.len() == *inner_size {
                    // Project out the inner (cycle) dimension → outer_size result
                    let mut projected = vec![F::zero(); *outer_size];
                    for (t, &eq_val) in eq_table.iter().enumerate() {
                        if eq_val.is_zero() {
                            continue;
                        }
                        let base = t * outer_size;
                        for k in 0..*outer_size {
                            projected[k] += eq_val * src_data[base + k];
                        }
                    }
                    DeviceBuffer::Field(backend.upload(&projected))
                } else {
                    // Project out the outer (address) dimension → inner_size result
                    let mut projected = vec![F::zero(); *inner_size];
                    for (t, proj) in projected.iter_mut().enumerate() {
                        let base = t * outer_size;
                        for (k, &eq_val) in eq_table.iter().enumerate() {
                            if !eq_val.is_zero() {
                                *proj += eq_val * src_data[base + k];
                            }
                        }
                    }
                    DeviceBuffer::Field(backend.upload(&projected))
                }
            }
            InputBinding::EqGather {
                eq_challenges: chs,
                indices,
                ..
            } => {
                let point: Vec<F> = chs.iter().map(|&ci| challenges[ci]).collect();
                let eq_table = jolt_poly::EqPolynomial::<F>::evals(&point, None);
                let idx_data = provider.materialize(*indices);
                let t = idx_data.len();
                let mut gathered = Vec::with_capacity(t);
                for j in 0..t {
                    let val = match idx_data[j].to_u64() {
                        Some(k) if (k as usize) < eq_table.len() => eq_table[k as usize],
                        _ => F::zero(),
                    };
                    gathered.push(val);
                }
                DeviceBuffer::Field(backend.upload(&gathered))
            }
            InputBinding::EqPushforward {
                eq_challenges: chs,
                indices,
                output_size,
                ..
            } => {
                let point: Vec<F> = chs.iter().map(|&ci| challenges[ci]).collect();
                let eq_table = jolt_poly::EqPolynomial::<F>::evals(&point, None);
                let idx_data = provider.materialize(*indices);
                let mut result = vec![F::zero(); *output_size];
                for (j, &eq_val) in eq_table.iter().enumerate() {
                    if j < idx_data.len() {
                        if let Some(k) = idx_data[j].to_u64() {
                            let k = k as usize;
                            if k < *output_size {
                                result[k] += eq_val;
                            }
                        }
                    }
                }
                DeviceBuffer::Field(backend.upload(&result))
            }
            InputBinding::ScaleByChallenge {
                source,
                challenge,
                power,
                ..
            } => {
                let base = challenges[*challenge];
                let mut scale = F::one();
                for _ in 0..*power {
                    scale *= base;
                }
                let src = provider.materialize(*source);
                let scaled: Vec<F> = src.iter().map(|&v| scale * v).collect();
                DeviceBuffer::Field(backend.upload(&scaled))
            }
            InputBinding::BytecodeVal {
                stage,
                stage_gamma_base,
                stage_gamma_count,
                gamma_base,
                raf_gamma_power,
                register_eq_challenges: reg_chs,
                ..
            } => {
                let bc = bytecode_data.expect(
                    "BytecodeVal binding requires bytecode_data"
                );
                // Build stage gamma powers from single base: [1, γ, γ², ...]
                let sg_base = challenges[*stage_gamma_base];
                let mut gammas = Vec::with_capacity(*stage_gamma_count);
                let mut pow = F::one();
                for _ in 0..*stage_gamma_count {
                    gammas.push(pow);
                    pow *= sg_base;
                }
                let eq_r_register = if reg_chs.is_empty() {
                    Vec::new()
                } else {
                    let point: Vec<F> = reg_chs.iter().map(|&ci| challenges[ci]).collect();
                    jolt_poly::EqPolynomial::<F>::evals(&point, None)
                };
                let mut val = crate::bytecode_raf::compute_val_stage(
                    &bc.entries,
                    *stage,
                    &gammas,
                    &eq_r_register,
                );
                let gamma = challenges[*gamma_base];
                // Add RAF identity contribution: gamma^raf_power × k
                if let Some(p) = raf_gamma_power {
                    let mut raf_g = F::one();
                    for _ in 0..*p {
                        raf_g *= gamma;
                    }
                    for (k, v) in val.iter_mut().enumerate() {
                        *v += raf_g * F::from_u64(k as u64);
                    }
                }
                // Multiply by gamma^stage
                let mut overall = F::one();
                for _ in 0..*stage {
                    overall *= gamma;
                }
                for v in &mut val {
                    *v *= overall;
                }
                DeviceBuffer::Field(backend.upload(&val))
            }
        };
        let _ = device_buffers.insert(pi, buf);
    }
}

/// Build the outer eq table for a segmented phase.
///
/// When `outer_eq_challenges` is empty, returns uniform weights (all 1.0),
/// giving an unweighted sum over the outer dimension. This is used by
/// RamRW where the address dimension has no eq polynomial.
fn build_outer_eq<B, F>(challenges: &[F], seg: &SegmentedConfig, backend: &B) -> Vec<F>
where
    B: ComputeBackend,
    F: Field,
{
    if seg.outer_eq_challenges.is_empty() {
        vec![F::one(); 1 << seg.outer_num_vars]
    } else {
        let point: Vec<F> = seg
            .outer_eq_challenges
            .iter()
            .map(|&ci| challenges[ci])
            .collect();
        let buf = backend.eq_table(&point);
        backend.download(&buf)
    }
}

/// Segmented reduce: iterate over outer positions, extract inner columns
/// from mixed inputs, run the Dense kernel per column, accumulate with
/// outer eq weights.
///
/// For inner-only inputs, the same buffer is reused across all outer
/// positions. For mixed inputs (inner × outer elements), a column is
/// extracted per outer position.
#[allow(clippy::too_many_arguments)]
fn segmented_reduce<B, F>(
    device_buffers: &HashMap<PolynomialId, Buf<B, F>>,
    outer_eq: &[F],
    seg: &SegmentedConfig,
    kdef: &KernelDef,
    compiled_kernel: &B::CompiledKernel<F>,
    challenges: &[F],
    backend: &B,
    round_within_phase: usize,
) -> Vec<F>
where
    B: ComputeBackend,
    F: Field,
{
    let inner_size = 1usize << (seg.inner_num_vars - round_within_phase);
    let num_inputs = kdef.inputs.len();

    // Download all input data once.
    let input_data: Vec<Vec<F>> = kdef
        .inputs
        .iter()
        .map(|b| {
            let buf = device_buffers
                .get(&b.poly())
                .expect("segmented reduce: input buffer missing");
            backend.download(buf.as_field())
        })
        .collect();

    // Pre-allocate column buffer for mixed inputs.
    let mut col_buf = vec![F::zero(); inner_size];
    let mut total_evals: Option<Vec<F>> = None;

    for (a, &weight) in outer_eq.iter().enumerate() {
        if weight.is_zero() {
            continue;
        }

        // Build per-column input buffers.
        let mut col_bufs: Vec<Buf<B, F>> = Vec::with_capacity(num_inputs);
        for (j, data) in input_data.iter().enumerate() {
            if seg.inner_only[j] {
                // Inner-only: use the full buffer directly (T elements).
                col_bufs.push(DeviceBuffer::Field(backend.upload(data)));
            } else {
                // Mixed: extract column a (elements a*inner_size .. (a+1)*inner_size).
                let start = a * inner_size;
                col_buf.copy_from_slice(&data[start..start + inner_size]);
                col_bufs.push(DeviceBuffer::Field(backend.upload(&col_buf)));
            }
        }

        let col_refs: Vec<&Buf<B, F>> = col_bufs.iter().collect();
        let evals = backend.reduce(compiled_kernel, &col_refs, challenges);

        match &mut total_evals {
            Some(total) => {
                for (t, &e) in total.iter_mut().zip(&evals) {
                    *t += weight * e;
                }
            }
            None => {
                total_evals = Some(evals.iter().map(|&e| weight * e).collect());
            }
        }
    }

    total_evals.unwrap_or_else(|| vec![F::zero(); kdef.spec.num_evals])
}

/// Evaluate a [`ClaimFormula`] against the prover's current state.
///
/// Input claim formulas use only `Eval` and `Challenge` factors (no
/// `EqEval`, `StageEval`, or `PreprocessedPolyEval`).
fn evaluate_claim<F: Field>(
    formula: &ClaimFormula,
    evaluations: &HashMap<PolynomialId, F>,
    challenges: &[F],
) -> F {
    let mut sum = F::zero();
    for term in &formula.terms {
        let mut product = F::from_i128(term.coeff);
        for factor in &term.factors {
            product *= match factor {
                ClaimFactor::Eval(poly) => *evaluations
                    .get(poly)
                    .unwrap_or_else(|| panic!("evaluate_claim: {poly:?} not available")),
                ClaimFactor::Challenge(i) => challenges[*i],
                ClaimFactor::EqChallengePair { a, b } => {
                    let (ra, rb) = (challenges[*a], challenges[*b]);
                    ra * rb + (F::one() - ra) * (F::one() - rb)
                }
                other => panic!("evaluate_claim: unsupported factor {other:?}"),
            };
        }
        sum += product;
    }
    sum
}

/// Evaluate a multilinear extension at a point.
///
/// Standard MLE evaluation: for each variable, fold adjacent pairs
/// `f[2i], f[2i+1]` via `(1-r)*f[2i] + r*f[2i+1]`, halving the table.
fn evaluate_mle<F: Field>(evals: &[F], point: &[F]) -> F {
    let mut buf = evals.to_vec();
    for r in point.iter().rev() {
        let half = buf.len() / 2;
        for i in 0..half {
            buf[i] = buf[2 * i] + *r * (buf[2 * i + 1] - buf[2 * i]);
        }
        buf.truncate(half);
    }
    buf[0]
}
