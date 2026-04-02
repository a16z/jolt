//! Prover runtime: execute a linked schedule to produce a proof.
//!
//! [`execute`] walks the complete [`Op`] sequence — compute, PCS, and
//! orchestration ops — and returns a [`JoltProof`]. It dispatches to:
//! - [`ComputeBackend`] for polynomial arithmetic (sumcheck, bind, evaluate)
//! - [`CommitmentScheme`] for cryptographic ops (commit, reduce, open)
//! - Direct calls for orchestration (transcript absorb/squeeze, lifecycle)
//!
//! ```text
//! Protocol → compile() → Module → link(backend) → Executable<P,B,F>
//!                                                       │
//!                                          execute(exe, provider, backend, pcs, transcript)
//!                                                       │
//!                                                       ▼
//!                                                 JoltProof<F, PCS>
//! ```

use std::collections::HashMap;

use jolt_compiler::module::{ChallengeSource, InputBinding, Op, VerifierStageIndex};
use jolt_compiler::{KernelDef, PolyId};
use jolt_compute::{Buf, BufferProvider, ComputeBackend, DeviceBuffer, Executable};
use jolt_field::Field;
use jolt_openings::{AdditivelyHomomorphic, ProverClaim};
use jolt_poly::UnivariatePoly;
use jolt_sumcheck::proof::SumcheckProof;
use jolt_transcript::{AppendToTranscript, Transcript};
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
struct PendingClaim<P: PolyId, F: Field> {
    poly: P,
    point: Vec<F>,
    eval: F,
}

/// Mutable state accumulated during schedule execution.
struct RuntimeState<P: PolyId, F: Field, PCS: AdditivelyHomomorphic<Field = F>> {
    // ── Compute state ──
    challenges: Vec<F>,
    evaluations: HashMap<P, F>,
    last_round_coeffs: Vec<F>,
    /// Last interpolated round polynomial (for scalar poly evaluation).
    last_round_poly: Option<UnivariatePoly<F>>,
    /// Most recently squeezed challenge value.
    last_squeezed: F,

    // ── Proof assembly (incremental) ──
    current_stage: Option<StageBuilder<F>>,
    stage_proofs: Vec<StageProof<F>>,

    // ── PCS state ──
    commitments: Vec<PCS::Output>,
    hints: HashMap<P, PCS::OpeningHint>,
    pending_claims: Vec<PendingClaim<P, F>>,
    pending_hints: Vec<PCS::OpeningHint>,
    reduced_claims: Vec<ProverClaim<F>>,
    reduced_hints: Vec<PCS::OpeningHint>,
    opening_proofs: Vec<PCS::Proof>,
}

/// Execute the full prover schedule and return a complete proof.
///
/// Walks every op in the schedule, dispatching compute ops to `backend`,
/// PCS ops to the commitment scheme, and orchestration ops directly.
pub(crate) fn execute<P, B, F, T, PCS>(
    executable: &Executable<P, B, F>,
    provider: &mut impl BufferProvider<P, B, F>,
    backend: &B,
    pcs_setup: &PCS::ProverSetup,
    transcript: &mut T,
    config: ProverConfig,
) -> JoltProof<F, PCS>
where
    P: PolyId,
    B: ComputeBackend,
    F: Field,
    T: Transcript<Challenge = F>,
    PCS: AdditivelyHomomorphic<Field = F>,
    PCS::Output: AppendToTranscript,
{
    let module = &executable.module;

    let mut state = RuntimeState::<P, F, PCS> {
        challenges: vec![F::zero(); module.challenges.len()],
        evaluations: HashMap::new(),
        last_round_coeffs: Vec::new(),
        last_round_poly: None,
        last_squeezed: F::zero(),
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
    let mut device_buffers: HashMap<P, Buf<B, F>> = HashMap::new();

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
                );

                // Bind (rounds 1+): take buffers out, bind via kernel-aware
                // dispatch, put back. The compiled kernel determines the
                // binding strategy (dense, tensor, sparse).
                if let Some(ch) = bind_challenge {
                    let scalar = state.challenges[*ch];
                    let mut input_bufs: Vec<Buf<B, F>> = kdef
                        .inputs
                        .iter()
                        .map(|b| {
                            device_buffers
                                .remove(&b.poly())
                                .expect("kernel input buffer missing")
                        })
                        .collect();
                    backend.bind(compiled_kernel, &mut input_bufs, scalar);
                    for (buf, binding) in input_bufs.into_iter().zip(&kdef.inputs) {
                        let _ = device_buffers.insert(binding.poly(), buf);
                    }
                }

                let input_refs: Vec<&Buf<B, F>> = kdef
                    .inputs
                    .iter()
                    .filter_map(|b| device_buffers.get(&b.poly()))
                    .collect();

                state.last_round_coeffs =
                    backend.reduce(compiled_kernel, &input_refs, &state.challenges);
            }

            Op::Evaluate { poly } => {
                if let Some(buf) = device_buffers.get(poly) {
                    let data = backend.download(buf.as_field());
                    if !data.is_empty() {
                        let _ = state.evaluations.insert(*poly, data[0]);
                    }
                } else if let Some(round_poly) = &state.last_round_poly {
                    // Scalar evaluation: evaluate last round polynomial at
                    // the most recently squeezed challenge.
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
                    if let Some(DeviceBuffer::Field(buf)) = device_buffers.get_mut(pi) {
                        backend.interpolate_inplace(buf, scalar, *order);
                    }
                }
            }

            // ── PCS ──
            Op::Commit { polys, .. } | Op::CommitStreaming { polys, .. } => {
                for pi in polys {
                    let data = provider.as_slice(*pi);
                    let (commitment, hint) = PCS::commit(data, pcs_setup);
                    commitment.append_to_transcript(transcript);
                    let _ = state.hints.insert(*pi, hint);
                    state.commitments.push(commitment);
                }
            }

            Op::ReduceOpenings => {
                let pending = std::mem::take(&mut state.pending_claims);
                let hints = std::mem::take(&mut state.pending_hints);

                let (claims, combined_hints) =
                    fused_rlc_reduce::<_, _, _, PCS>(pending, hints, provider, transcript);

                state.reduced_claims = claims;
                state.reduced_hints = combined_hints;
            }

            Op::Open => {
                for (claim, hint) in state.reduced_claims.iter().zip(state.reduced_hints.iter()) {
                    let poly: PCS::Polynomial = claim.evaluations.clone().into();
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

            Op::AbsorbRoundPoly { num_coeffs, .. } => {
                let evals = &state.last_round_coeffs[..*num_coeffs];
                let points: Vec<(F, F)> = evals
                    .iter()
                    .enumerate()
                    .map(|(slot, &val)| (F::from_u64(slot as u64), val))
                    .collect();
                let poly = UnivariatePoly::interpolate(&points);
                let coeffs = poly.into_coefficients();
                for c in &coeffs {
                    transcript.append(c);
                }
                let round_poly = UnivariatePoly::new(coeffs);
                state.last_round_poly = Some(round_poly.clone());
                if let Some(stage) = &mut state.current_stage {
                    stage.round_polys.push(round_poly);
                }
            }

            Op::AbsorbEvals { polys, .. } => {
                let mut batch = Vec::with_capacity(polys.len());
                for pi in polys {
                    if let Some(&val) = state.evaluations.get(pi) {
                        transcript.append(&val);
                        batch.push(val);
                    }
                }
                if let Some(stage) = &mut state.current_stage {
                    stage.evals.extend(batch);
                }
            }

            Op::Squeeze { challenge } => {
                let val = transcript.challenge();
                state.challenges[*challenge] = val;
                state.last_squeezed = val;
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
fn fused_rlc_reduce<P, B, F, PCS>(
    pending: Vec<PendingClaim<P, F>>,
    hints: Vec<PCS::OpeningHint>,
    provider: &impl BufferProvider<P, B, F>,
    transcript: &mut impl Transcript<Challenge = F>,
) -> (Vec<ProverClaim<F>>, Vec<PCS::OpeningHint>)
where
    P: PolyId,
    B: ComputeBackend,
    F: Field,
    PCS: AdditivelyHomomorphic<Field = F>,
{
    if pending.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Absorb all claim evaluations before drawing any RLC challenge rho.
    for pc in &pending {
        pc.eval.append_to_transcript(transcript);
    }

    // Group by point (preserving insertion order).
    struct PointGroup<'a, P, F, H> {
        point: &'a Vec<F>,
        poly_ids: Vec<P>,
        evals: Vec<F>,
        hints: Vec<H>,
    }

    let mut groups: Vec<PointGroup<'_, P, F, PCS::OpeningHint>> = Vec::new();

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

        // RLC-combine evaluation tables directly from provider slices.
        // One allocation for the combined result — no per-poly copies.
        let slices: Vec<&[F]> = poly_ids.iter().map(|&pi| provider.as_slice(pi)).collect();
        let combined_evals = jolt_openings::rlc_combine(&slices, rho);
        let combined_eval = jolt_openings::rlc_combine_scalars(&evals, rho);

        // Combine hints with the same rho powers.
        let powers: Vec<F> = std::iter::successors(Some(F::from_u64(1)), |prev| Some(*prev * rho))
            .take(group_hints.len())
            .collect();
        let combined_hint = PCS::combine_hints(group_hints, &powers);

        reduced_claims.push(ProverClaim {
            evaluations: combined_evals,
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
fn precompute_stage_points<P: PolyId>(
    module: &jolt_compiler::module::Module<P>,
) -> Vec<Vec<usize>> {
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
fn resolve_inputs<P, B, F>(
    device_buffers: &mut HashMap<P, Buf<B, F>>,
    challenges: &[F],
    kdef: &KernelDef<P>,
    provider: &mut impl BufferProvider<P, B, F>,
    backend: &B,
) where
    P: PolyId,
    B: ComputeBackend,
    F: Field,
{
    for binding in &kdef.inputs {
        let pi = binding.poly();
        if device_buffers.contains_key(&pi) {
            continue;
        }
        let buf: Buf<B, F> = match binding {
            InputBinding::Provided { .. } => provider.load(pi, backend),
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
        };
        let _ = device_buffers.insert(pi, buf);
    }
}
