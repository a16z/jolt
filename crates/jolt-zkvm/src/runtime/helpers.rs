use std::borrow::Cow;
use std::collections::HashMap;

use jolt_compiler::module::{ChallengeSource, InputBinding, SegmentedConfig, VerifierStageIndex};
use jolt_compiler::PolynomialId;
use jolt_compute::{Buf, BufferProvider, ComputeBackend, DeviceBuffer};
use jolt_crypto::HomomorphicCommitment;
use jolt_field::Field;
use jolt_openings::{AdditivelyHomomorphic, ProverClaim};
use jolt_transcript::{AppendToTranscript, LabelWithCount, Transcript};

/// Lightweight opening claim — defers the expensive evaluation table copy
/// until `ReduceOpenings` where it's actually needed for RLC combination.
pub(crate) struct PendingClaim<F: Field> {
    pub poly: PolynomialId,
    pub point: Vec<F>,
    pub eval: F,
}

/// Remove kernel input buffers from cache, bind them at `scalar`, and reinsert.
pub(super) fn bind_kernel_inputs<B: ComputeBackend, F: Field>(
    device_buffers: &mut HashMap<PolynomialId, Buf<B, F>>,
    backend: &B,
    compiled_kernel: &B::CompiledKernel<F>,
    kdef: &jolt_compiler::KernelDef,
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

/// Precompute verifier-stage → round challenge indices mapping.
pub(super) fn precompute_stage_points(module: &jolt_compiler::module::Module) -> Vec<Vec<usize>> {
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

pub(super) fn build_outer_eq<B, F>(challenges: &[F], seg: &SegmentedConfig, backend: &B) -> Vec<F>
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
            .map(|&ci| challenges[ci.0])
            .collect();
        let buf = backend.eq_table(&point);
        backend.download(&buf)
    }
}

pub(super) fn materialize_binding<B, F>(
    binding: &InputBinding,
    challenges: &[F],
    provider: &impl BufferProvider<F>,
    backend: &B,
) -> Buf<B, F>
where
    B: ComputeBackend,
    F: Field,
{
    match binding {
        InputBinding::Provided { poly, .. } => {
            let _s = tracing::info_span!("mb::Provided").entered();
            let data = provider.materialize(*poly);
            DeviceBuffer::Field(backend.upload(&data))
        }
        InputBinding::EqTable {
            challenges: chs, ..
        } => {
            let _s = tracing::info_span!("mb::EqTable").entered();
            let point: Vec<F> = chs.iter().map(|&ci| challenges[ci.0]).collect();
            DeviceBuffer::Field(backend.eq_table(&point))
        }
        InputBinding::EqPlusOneTable {
            challenges: chs, ..
        } => {
            let _s = tracing::info_span!("mb::EqPlusOneTable").entered();
            let point: Vec<F> = chs.iter().map(|&ci| challenges[ci.0]).collect();
            let (_eq, eq_plus_one) = backend.eq_plus_one_table(&point);
            DeviceBuffer::Field(eq_plus_one)
        }
        InputBinding::LtTable {
            challenges: chs, ..
        } => {
            let _s = tracing::info_span!("mb::LtTable").entered();
            let point: Vec<F> = chs.iter().map(|&ci| challenges[ci.0]).collect();
            DeviceBuffer::Field(backend.lt_table(&point))
        }
        InputBinding::EqProject {
            source,
            challenges: chs,
            inner_size,
            outer_size,
            ..
        } => {
            let _s = tracing::info_span!("mb::EqProject").entered();
            let point: Vec<F> = chs.iter().map(|&ci| challenges[ci.0]).collect();
            let src_data = provider.materialize(*source);
            DeviceBuffer::Field(backend.eq_project(&src_data, &point, *inner_size, *outer_size))
        }
        InputBinding::Transpose {
            source, rows, cols, ..
        } => {
            let _s = tracing::info_span!("mb::Transpose").entered();
            let src_data = provider.materialize(*source);
            DeviceBuffer::Field(backend.transpose_from_host(&src_data, *rows, *cols))
        }
        InputBinding::EqGather {
            eq_challenges: chs,
            indices,
            ..
        } => {
            let _s = tracing::info_span!("mb::EqGather").entered();
            let point: Vec<F> = chs.iter().map(|&ci| challenges[ci.0]).collect();
            let idx_data = provider.materialize(*indices);
            DeviceBuffer::Field(backend.eq_gather(&point, &idx_data))
        }
        InputBinding::EqPushforward {
            eq_challenges: chs,
            indices,
            output_size,
            ..
        } => {
            let _s = tracing::info_span!("mb::EqPushforward").entered();
            let point: Vec<F> = chs.iter().map(|&ci| challenges[ci.0]).collect();
            let idx_data = provider.materialize(*indices);
            DeviceBuffer::Field(backend.eq_pushforward(&point, &idx_data, *output_size))
        }
        InputBinding::ScaleByChallenge {
            source,
            challenge,
            power,
            ..
        } => {
            let _s = tracing::info_span!("mb::ScaleByChallenge").entered();
            let base = challenges[challenge.0];
            let mut scale = F::one();
            for _ in 0..*power {
                scale *= base;
            }
            let src = provider.materialize(*source);
            DeviceBuffer::Field(backend.scale_from_host(&src, scale))
        }
    }
}

/// Fused RLC reduction: groups pending claims by point, draws one rho per group
/// from the transcript, and produces combined (claim, hint) pairs in a single pass.
pub(super) fn fused_rlc_reduce<F, PCS>(
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

    transcript.append(&LabelWithCount(b"rlc_claims", pending.len() as u64));
    for pc in &pending {
        pc.eval.append_to_transcript(transcript);
    }

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

    let group_count = groups.len();
    let total_claims = pending.len();
    let mut reduced_claims = Vec::with_capacity(group_count);
    let mut reduced_hints = Vec::with_capacity(group_count);
    let mut group_stats: Vec<(usize, usize, u128, u128, u128)> = Vec::with_capacity(group_count);

    for PointGroup {
        point,
        poly_ids,
        evals,
        hints: group_hints,
    } in groups
    {
        let rho: F = transcript.challenge();

        let poly_count = poly_ids.len();
        let t_mat_start = std::time::Instant::now();
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
        let t_mat_us = t_mat_start.elapsed().as_micros();

        let slices: Vec<&[F]> = materialized.iter().map(|c| &**c).collect();
        let total_elems = slices.iter().map(|s| s.len()).sum::<usize>();

        let t_rlc_start = std::time::Instant::now();
        let combined_evals = jolt_openings::rlc_combine(&slices, rho);
        let combined_eval = jolt_openings::rlc_combine_scalars(&evals, rho);
        let t_rlc_us = t_rlc_start.elapsed().as_micros();

        let t_hint_start = std::time::Instant::now();
        let powers: Vec<F> = std::iter::successors(Some(F::from_u64(1)), |prev| Some(*prev * rho))
            .take(group_hints.len())
            .collect();
        let combined_hint = PCS::combine_hints(group_hints, &powers);
        let t_hint_us = t_hint_start.elapsed().as_micros();

        group_stats.push((poly_count, total_elems, t_mat_us, t_rlc_us, t_hint_us));

        reduced_claims.push(ProverClaim {
            polynomial: combined_evals.into(),
            point: point.clone(),
            eval: combined_eval,
        });
        reduced_hints.push(combined_hint);
    }

    for (gi, (pc, te, tm, tr, th)) in group_stats.iter().enumerate() {
        tracing::info!(
            target: "perf_rlc",
            group = gi,
            poly_count = pc,
            total_elems = te,
            mat_us = *tm as u64,
            rlc_us = *tr as u64,
            hint_us = *th as u64,
            "rlc_group"
        );
    }
    tracing::info!(
        target: "perf_rlc",
        groups = group_count,
        total_claims = total_claims,
        "rlc_summary"
    );

    (reduced_claims, reduced_hints)
}
