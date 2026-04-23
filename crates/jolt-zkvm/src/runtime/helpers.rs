use std::collections::HashMap;

use jolt_compiler::module::{ChallengeSource, InputBinding, SegmentedConfig, VerifierStageIndex};
use jolt_compiler::PolynomialId;
use jolt_compute::{Buf, BufferProvider, ComputeBackend, DeviceBuffer};
use jolt_field::Field;
use jolt_openings::ProverClaim;

/// Lightweight opening claim — defers the expensive evaluation table copy
/// until `Op::ProveBatch` where it's needed to construct a `ProverClaim`.
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
            let _s_upload = tracing::info_span!("mb::upload").entered();
            // Cow::Owned → pass-through on CpuBackend (no memcpy);
            // Cow::Borrowed → into_owned() does a single to_vec() for the rare
            // borrow cases (Witness / Preprocessed polys are already ≤ 1% of mb::upload).
            DeviceBuffer::Field(backend.upload_vec(data.into_owned()))
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

/// Materialize each pending claim's polynomial evaluations into a
/// `ProverClaim` ready for `PCS::prove_batch`. Drains entries from
/// `padded` to avoid cloning when a polynomial was pre-padded
/// (e.g. by `Op::Commit`); falls back to `provider.materialize`.
#[tracing::instrument(skip_all, name = "materialize_pending_claims")]
pub(super) fn materialize_pending_claims<F: Field>(
    pending: Vec<PendingClaim<F>>,
    provider: &impl BufferProvider<F>,
    padded: &mut HashMap<PolynomialId, Vec<F>>,
) -> Vec<ProverClaim<F>> {
    pending
        .into_iter()
        .map(|pc| {
            let evals: Vec<F> = if let Some(p) = padded.remove(&pc.poly) {
                p
            } else {
                provider.materialize(pc.poly).into_owned()
            };
            ProverClaim {
                polynomial: evals.into(),
                point: pc.point,
                eval: pc.eval,
            }
        })
        .collect()
}
