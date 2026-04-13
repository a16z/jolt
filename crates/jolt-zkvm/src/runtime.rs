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
    ChallengeSource, DomainSeparator, EvalMode, InputBinding, Op, RoundPolyEncoding,
    SegmentedConfig, VerifierStageIndex,
};
use jolt_compiler::{KernelDef, PolynomialId};
use jolt_compute::{
    Buf, BufferProvider, ComputeBackend, DeviceBuffer, Executable, LookupTraceData,
};
use jolt_crypto::HomomorphicCommitment;
use jolt_field::Field;
use jolt_openings::{AdditivelyHomomorphic, ProverClaim};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::proof::SumcheckProof;
use jolt_transcript::{AppendToTranscript, Label, LabelWithCount, Transcript};
use jolt_verifier::proof::{JoltProof, StageProof};
use jolt_verifier::ProverConfig;

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
struct RuntimeState<B: ComputeBackend, F: Field, PCS: AdditivelyHomomorphic<Field = F>>
where
    PCS::Output: HomomorphicCommitment<F>,
{
    challenges: Vec<F>,
    evaluations: HashMap<PolynomialId, F>,
    last_round_coeffs: Vec<F>,
    /// Last interpolated round polynomial (for scalar poly evaluation).
    last_round_poly: Option<UnivariatePoly<F>>,
    /// Most recently squeezed challenge value.
    last_squeezed: F,

    /// Per-batch → per-instance running claims.
    batch_instance_claims: Vec<Vec<F>>,
    /// Per-instance evaluations from the last round, used to update claims
    /// when the next round's challenge arrives. `[inst_idx] → [eval_0, eval_1, ...]`
    last_round_instance_evals: Vec<Vec<F>>,
    /// In-progress combined round polynomial for the current batched round.
    /// Set by `BatchRoundBegin`, accumulated by `BatchAccumulateInstance`,
    /// finalized by `BatchRoundFinalize`.
    batch_combined: Vec<F>,
    /// Tracks which poly IDs have been bound this round to prevent double-binding
    /// when multiple instances share polynomial buffers.
    bound_this_round: HashSet<PolynomialId>,
    /// Current batch round index (set by BatchRoundBegin).
    current_batch_round: usize,
    /// Outer eq tables for segmented instances: `(batch, inst) → eq_outer`.
    /// Built once at phase start, used for weighting during segmented reduce.
    segmented_outer_eqs: HashMap<(usize, usize), Vec<F>>,
    /// Stateful sumcheck instances (unified InstanceState).
    instance_states: HashMap<(usize, usize), B::InstanceState<F>>,

    current_stage: Option<StageBuilder<F>>,
    stage_proofs: Vec<StageProof<F>>,

    commitments: Vec<PCS::Output>,
    hints: HashMap<PolynomialId, PCS::OpeningHint>,
    pending_claims: Vec<PendingClaim<F>>,
    pending_hints: Vec<PCS::OpeningHint>,
    reduced_claims: Vec<ProverClaim<F>>,
    reduced_hints: Vec<PCS::OpeningHint>,
    opening_proofs: Vec<PCS::Proof>,
    /// Zero-padded polynomial data for dense polys in the opening proof.
    /// Populated by `CollectOpeningClaimAt` when `committed_num_vars` requires padding.
    padded_poly_data: HashMap<PolynomialId, Vec<F>>,
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
    bytecode_data: Option<jolt_witness::bytecode_raf::BytecodeData<F>>,
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

    let mut state = RuntimeState::<B, F, PCS> {
        challenges: vec![F::zero(); module.challenges.len()],
        evaluations: HashMap::new(),
        last_round_coeffs: Vec::new(),
        last_round_poly: None,
        last_squeezed: F::zero(),
        batch_instance_claims,
        last_round_instance_evals: Vec::new(),
        batch_combined: Vec::new(),
        bound_this_round: HashSet::new(),
        current_batch_round: 0,
        segmented_outer_eqs: HashMap::new(),
        instance_states: HashMap::new(),
        current_stage: None,
        stage_proofs: Vec::new(),
        commitments: Vec::new(),
        hints: HashMap::new(),
        pending_claims: Vec::new(),
        pending_hints: Vec::new(),
        reduced_claims: Vec::new(),
        reduced_hints: Vec::new(),
        opening_proofs: Vec::new(),
        padded_poly_data: HashMap::new(),
    };

    // Precompute: verifier stage → sorted round challenge indices.
    let stage_point_indices: Vec<Vec<usize>> = precompute_stage_points(module);

    // Device buffer cache — compute ops work on backend buffers.
    let mut device_buffers: HashMap<PolynomialId, Buf<B, F>> = HashMap::new();

    let mut t_ops: usize = 0;
    for op in &executable.ops {
        match op {
            Op::SumcheckRound {
                kernel,
                round: _,
                bind_challenge,
            } => {
                let kdef = &module.prover.kernels[*kernel];
                let compiled_kernel = &executable.kernels[*kernel];

                if let Some(ch) = bind_challenge {
                    let scalar = state.challenges[*ch];
                    bind_kernel_inputs(&mut device_buffers, backend, compiled_kernel, kdef, scalar);
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
                        0 => continue,
                        1 => data[0],
                        2 => {
                            let r = state.last_squeezed;
                            data[0] + r * (data[1] - data[0])
                        }
                        n => panic!("Evaluate: {poly:?} has {n}-element buffer; expected 1 or 2"),
                    };
                    let _ = state.evaluations.insert(*poly, val);
                } else if matches!(mode, EvalMode::RoundPoly) {
                    let round_poly = state
                        .last_round_poly
                        .as_ref()
                        .expect("RoundPoly: no round polynomial available");
                    let val = round_poly.evaluate(state.last_squeezed);
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

                // Lagrange kernel scale: L(τ, r) = Σ_k L_k(τ) · L_k(r)
                let scale = if let Some(tau_idx) = kernel_tau {
                    let tau = state.challenges[*tau_idx];
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
                    let expanded =
                        DeviceBuffer::Field(backend.duplicate_interleave(buf.as_field()));
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
                    transcript.append(&LabelWithCount(tag.as_bytes(), commitment.serialized_len()));
                    t_ops += 1;
                    commitment.append_to_transcript(transcript);
                    t_ops += 1;
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
                transcript.append(&config);
                t_ops += 1;
            }

            Op::BeginStage { index } => {
                eprintln!("[zkvm] BeginStage {index} at transcript_op={t_ops}");
                if let Some(builder) = state.current_stage.take() {
                    state.stage_proofs.push(builder.finalize());
                }
                state.current_stage = Some(StageBuilder::new());
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
                        let tau_high = state.challenges[*tau_challenge];
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
                        t_ops += 1;
                        for c in &coeffs {
                            transcript.append(c);
                            t_ops += 1;
                        }
                    }
                    RoundPolyEncoding::Compressed => {
                        let compressed_len = coeffs.len() - 1;
                        let label_op = t_ops;
                        transcript.append(&LabelWithCount(tag.as_bytes(), compressed_len as u64));
                        t_ops += 1;
                        transcript.append(&coeffs[0]);
                        t_ops += 1;
                        for c in &coeffs[2..] {
                            transcript.append(c);
                            t_ops += 1;
                        }
                        {
                            let absorbed: Vec<_> = std::iter::once(&coeffs[0])
                                .chain(coeffs[2..].iter())
                                .collect();
                            eprintln!("[zkvm] AbsorbRoundPoly compressed: ops={label_op}..{t_ops} num_coeffs={num_coeffs} absorbed={absorbed:?}");
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
                        t_ops += 1;
                        transcript.append(&val);
                        t_ops += 1;
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
                let val = backend.evaluate_claim(formula, &state.evaluations, &state.challenges);
                transcript.append(&Label(tag.as_bytes()));
                t_ops += 1;
                transcript.append(&val);
                t_ops += 1;
                // Pre-scale claim by 2^inactive_scale_bits so that the
                // inactive-round halving lands on the correct value.
                let mut scaled = val;
                let two = F::from_u64(2);
                for _ in 0..*inactive_scale_bits {
                    scaled *= two;
                }
                state.batch_instance_claims[*batch][*instance] = scaled;
            }

            Op::Squeeze { challenge } => {
                let val = transcript.challenge();
                t_ops += 1;
                state.challenges[*challenge] = val;
                state.last_squeezed = val;
            }

            Op::ComputePower {
                target,
                base,
                exponent,
            } => {
                let base_val = state.challenges[*base];
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
                state.challenges[*target] = result;
            }

            Op::AppendDomainSeparator { tag } => {
                let label = tag.as_bytes();
                let mut packed = [0u8; 32];
                packed[..label.len()].copy_from_slice(label);
                transcript.append_bytes(&packed);
                t_ops += 1;
                transcript.append_bytes(&[]);
                t_ops += 1;
            }

            Op::EvaluatePreprocessed {
                source,
                at_challenges,
                store_as,
            } => {
                let data = provider.materialize(*source);
                let point: Vec<F> = at_challenges
                    .iter()
                    .map(|&ci| state.challenges[ci])
                    .collect();
                let eval = backend.evaluate_mle(&data, &point);
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

            Op::ScaleEval {
                poly,
                factor_challenges,
            } => {
                if let Some(eval) = state.evaluations.get_mut(poly) {
                    let factor: F = factor_challenges
                        .iter()
                        .map(|&ci| F::one() - state.challenges[ci])
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
                        .map(|&ci| state.challenges[ci])
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
                    .map(|&ci| state.challenges[ci])
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
                if *batch == 4 && *round == 0 {
                    eprintln!(
                        "[zkvm batch4 initial_claims] {:?}",
                        &state.batch_instance_claims[*batch]
                    );
                }
                if let Some(ch) = bind_challenge {
                    let r = state.challenges[*ch];
                    for (inst_idx, evals) in state.last_round_instance_evals.iter().enumerate() {
                        if !evals.is_empty() {
                            state.batch_instance_claims[*batch][inst_idx] =
                                backend.interpolate_evaluate(evals, r);
                        }
                    }
                }
                let num_instances = state.batch_instance_claims[*batch].len();
                state.last_round_instance_evals = vec![Vec::new(); num_instances];
            }

            Op::BatchInactiveContribution { batch, instance } => {
                let bdef = &module.prover.batched_sumchecks[*batch];
                let coeff = state.challenges[bdef.instances[*instance].batch_coeff];
                let two_inv = F::from_u64(2).inverse().unwrap();
                let half_claim = state.batch_instance_claims[*batch][*instance] * two_inv;
                if *batch == 4 && state.current_batch_round == 13 {
                    eprintln!(
                        "[zkvm inactive] batch={batch} round=13 inst={instance} half_claim={half_claim:?} coeff={coeff:?}",
                    );
                }
                for slot in &mut state.batch_combined {
                    *slot += coeff * half_claim;
                }
                state.batch_instance_claims[*batch][*instance] = half_claim;
            }

            Op::Materialize { binding } => {
                let pi = binding.poly();
                #[cfg(debug_assertions)]
                if matches!(pi, PolynomialId::BatchEq(70) | PolynomialId::BooleanityG(_)) {
                    eprintln!(
                        "[zkvm Materialize] {:?} round={}",
                        pi, state.current_batch_round
                    );
                }
                let buf = materialize_binding(
                    binding,
                    &state.challenges,
                    provider,
                    backend,
                    bytecode_data.as_ref(),
                );
                #[cfg(debug_assertions)]
                if matches!(pi, PolynomialId::BatchEq(70)) {
                    eprintln!(
                        "[zkvm Materialize] BatchEq(70) buffer size={}",
                        backend.download(buf.as_field()).len()
                    );
                }
                #[cfg(debug_assertions)]
                if matches!(binding, InputBinding::EqProject { .. }) {
                    let vals = backend.download(buf.as_field());
                    let sum: F = vals.iter().copied().sum();
                    let show: Vec<_> = vals.iter().take(8).collect();
                    eprintln!(
                        "[zkvm EqProject] {:?} len={} sum={:?} first8={:?}",
                        pi,
                        vals.len(),
                        sum,
                        show
                    );
                }
                let _ = device_buffers.insert(pi, buf);
            }

            Op::MaterializeUnlessFresh {
                binding,
                expected_size,
            } => {
                let pi = binding.poly();
                if let Some(existing) = device_buffers.get(&pi) {
                    if backend.len(existing.as_field()) == *expected_size {
                        continue;
                    }
                }
                let buf = materialize_binding(
                    binding,
                    &state.challenges,
                    provider,
                    backend,
                    bytecode_data.as_ref(),
                );
                let _ = device_buffers.insert(pi, buf);
            }

            Op::MaterializeIfAbsent { binding } => {
                let pi = binding.poly();
                #[cfg(debug_assertions)]
                if matches!(pi, PolynomialId::BatchEq(70) | PolynomialId::BooleanityG(_)) {
                    eprintln!(
                        "[zkvm MaterializeIfAbsent] {:?} present={} round={}",
                        pi,
                        device_buffers.contains_key(&pi),
                        state.current_batch_round
                    );
                }
                if device_buffers.contains_key(&pi) {
                    continue;
                }
                let buf = materialize_binding(
                    binding,
                    &state.challenges,
                    provider,
                    backend,
                    bytecode_data.as_ref(),
                );
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
                    .insert((*batch, *instance), outer_eq);
            }

            Op::InstanceBindPreviousPhase {
                batch: _,
                instance: _,
                kernel,
                challenge,
            } => {
                let kdef = &module.prover.kernels[*kernel];
                let scalar = state.challenges[*challenge];
                let order = kdef.spec.binding_order;
                #[cfg(debug_assertions)]
                eprintln!(
                    "[zkvm BindPrevPhase] kernel={} ch_idx={} scalar={:?}",
                    kernel, challenge, scalar
                );
                let mut seen = HashSet::new();
                for b in &kdef.inputs {
                    let pid = b.poly();
                    if state.bound_this_round.contains(&pid) || !seen.insert(pid) {
                        continue;
                    }
                    if let Some(buf) = device_buffers.get_mut(&pid) {
                        #[cfg(debug_assertions)]
                        {
                            let before_len = backend.download(buf.as_field()).len();
                            backend.interpolate_inplace(buf.as_field_mut(), scalar, order);
                            let after_len = backend.download(buf.as_field()).len();
                            eprintln!(
                                "[zkvm BindPrevPhase]   {:?}: {} -> {}",
                                pid, before_len, after_len
                            );
                        }
                        #[cfg(not(debug_assertions))]
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
                #[cfg(debug_assertions)]
                eprintln!(
                    "[zkvm CaptureScalar] poly={:?} ch_idx={} value={:?}",
                    poly, challenge, data[0]
                );
                state.challenges[*challenge] = data[0];
            }

            Op::InstanceReduce {
                batch,
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
                                "InstanceReduce: missing buffer {:?} (inst={instance}, kernel={kernel})",
                                b.poly()
                            )
                        })
                    })
                    .collect();
                let inst_evals = backend.reduce(compiled_kernel, &input_refs, &state.challenges);
                // Diagnostic: print booleanity evals at first active round (batch 5 = stage 6)
                #[cfg(debug_assertions)]
                if *instance == 1 && state.current_batch_round == 9 {
                    let buf_len = backend.len(input_refs[0].as_field());
                    eprintln!(
                        "[zkvm booleanity] batch={batch} round={} instance={instance} evals={inst_evals:?} n_inputs={} buf_len={buf_len}",
                        state.current_batch_round, input_refs.len(),
                    );
                }
                // Diagnostic: print evals for BytecodeReadRaf addr kernel (12 inputs, 3 evals)
                if input_refs.len() == 12 && inst_evals.len() == 3 {
                    let buf_len = backend.len(input_refs[0].as_field());
                    eprintln!(
                        "[zkvm bc_raf_addr] round={} kernel={kernel} instance={instance} evals={inst_evals:?} buf_len={buf_len}",
                        state.current_batch_round,
                    );
                    // Dump first 4 values of each input at rounds 9-10
                    if state.current_batch_round >= 9 && state.current_batch_round <= 10 {
                        for (idx, buf) in input_refs.iter().enumerate() {
                            let vals = backend.download(buf.as_field());
                            let show: Vec<_> = vals.iter().take(4).collect();
                            eprintln!(
                                "  [zkvm bc_raf_addr r={}] input[{idx}] len={} first4={show:?}",
                                state.current_batch_round,
                                vals.len()
                            );
                        }
                    }
                }
                state.last_round_instance_evals[*instance].clone_from(&inst_evals);
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
                    .get(&(*batch, *instance))
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
                state.last_round_instance_evals[*instance].clone_from(&inst_evals);
            }

            Op::InstanceBind {
                batch,
                instance,
                kernel,
                challenge,
            } => {
                let kdef = &module.prover.kernels[*kernel];
                let scalar = state.challenges[*challenge];
                let order = kdef.spec.binding_order;
                // BytecodeReadRaf addr kernel: 12 inputs, num_evals=3
                if kdef.inputs.len() == 12
                    && kdef.spec.num_evals == 3
                    && state.current_batch_round >= 9
                {
                    eprintln!(
                        "[zkvm InstanceBind bc_raf] round={} batch={batch} inst={instance} ch_idx={challenge} scalar={scalar:?}",
                        state.current_batch_round,
                    );
                }
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
                let scalar = state.challenges[*challenge];
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
                let bdef = &module.prover.batched_sumchecks[*batch];
                let coeff = state.challenges[bdef.instances[*instance].batch_coeff];
                let evals = &state.last_round_instance_evals[*instance];
                debug_assert_eq!(evals.len(), *num_evals);
                let extended;
                let full_evals = if *num_evals < *max_evals {
                    extended = backend.extend_evals(evals, *max_evals);
                    &extended
                } else {
                    evals.as_slice()
                };
                if *batch == 4 && state.current_batch_round >= 9 && state.current_batch_round <= 13
                {
                    eprintln!(
                        "[zkvm accum] batch={batch} round={} inst={instance} num_evals={num_evals} coeff={coeff:?} evals={evals:?} full_evals={full_evals:?}",
                        state.current_batch_round,
                    );
                }
                for (i, &v) in full_evals.iter().enumerate() {
                    state.batch_combined[i] += coeff * v;
                }
            }

            Op::BatchRoundFinalize { batch } => {
                eprintln!(
                    "[zkvm batch_final] batch={batch} len={} evals={:?}",
                    state.batch_combined.len(),
                    &state.batch_combined
                );
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
                    lookup_trace.as_ref(),
                    &module.prover.kernels,
                );
                let _ = state.instance_states.insert((*batch, *instance), is);
            }

            Op::UnifiedInstanceBind {
                batch,
                instance,
                challenge,
            } => {
                let scalar = state.challenges[*challenge];
                let is = state
                    .instance_states
                    .get_mut(&(*batch, *instance))
                    .expect("UnifiedInstanceBind: state missing");
                backend.instance_bind(is, scalar);
            }

            Op::UnifiedInstanceReduce { batch, instance } => {
                let is = state
                    .instance_states
                    .get(&(*batch, *instance))
                    .expect("UnifiedInstanceReduce: state missing");
                let previous_claim = state.batch_instance_claims[*batch][*instance];
                let evals = backend.instance_reduce(is, previous_claim);
                state.last_round_instance_evals[*instance].clone_from(&evals);
            }

            Op::UnifiedInstanceFinalize {
                batch,
                instance,
                output_buffers,
                output_evals,
            } => {
                let is = state
                    .instance_states
                    .remove(&(*batch, *instance))
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
    padded: &HashMap<PolynomialId, Vec<F>>,
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

        // RLC-combine evaluation tables, using padded data for dense polys.
        let materialized: Vec<Cow<'_, [F]>> = poly_ids
            .iter()
            .map(|&pi| {
                if let Some(p) = padded.get(&pi) {
                    Cow::Borrowed(p.as_slice())
                } else {
                    provider.materialize(pi)
                }
            })
            .collect();
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

fn materialize_binding<B, F>(
    binding: &InputBinding,
    challenges: &[F],
    provider: &impl BufferProvider<F>,
    backend: &B,
    bytecode_data: Option<&jolt_witness::bytecode_raf::BytecodeData<F>>,
) -> Buf<B, F>
where
    B: ComputeBackend,
    F: Field,
{
    match binding {
        InputBinding::Provided { poly, .. } => {
            let data = provider.materialize(*poly);
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
            poly: _,
            source,
            challenges: chs,
            inner_size,
            outer_size,
        } => {
            let point: Vec<F> = chs.iter().map(|&ci| challenges[ci]).collect();
            let src_data = provider.materialize(*source);
            DeviceBuffer::Field(backend.eq_project(&src_data, &point, *inner_size, *outer_size))
        }
        InputBinding::Transpose {
            source, rows, cols, ..
        } => {
            let src_data = provider.materialize(*source);
            DeviceBuffer::Field(backend.transpose_from_host(&src_data, *rows, *cols))
        }
        InputBinding::EqGather {
            eq_challenges: chs,
            indices,
            ..
        } => {
            let point: Vec<F> = chs.iter().map(|&ci| challenges[ci]).collect();
            let idx_data = provider.materialize(*indices);
            DeviceBuffer::Field(backend.eq_gather(&point, &idx_data))
        }
        InputBinding::EqPushforward {
            eq_challenges: chs,
            indices,
            output_size,
            ..
        } => {
            let point: Vec<F> = chs.iter().map(|&ci| challenges[ci]).collect();
            let idx_data = provider.materialize(*indices);
            DeviceBuffer::Field(backend.eq_pushforward(&point, &idx_data, *output_size))
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
            DeviceBuffer::Field(backend.scale_from_host(&src, scale))
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
            let bc = bytecode_data.expect("BytecodeVal binding requires bytecode_data");
            let val = bc.materialize_val(
                challenges,
                *stage,
                *stage_gamma_base,
                *stage_gamma_count,
                *gamma_base,
                *raf_gamma_power,
                reg_chs,
            );
            DeviceBuffer::Field(backend.upload(&val))
        }
    }
}
