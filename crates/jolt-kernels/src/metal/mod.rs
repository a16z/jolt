//! Apple Metal device tier (feature `metal`, macOS only).
//!
//! W1 infrastructure: the process-global runtime ([`runtime`]), BN254
//! Montgomery arithmetic in MSL (`shaders/`, constants generated off the
//! [`jolt_field::MontgomeryConstants`] seam by [`field`]), unified-memory
//! buffer wrappers ([`buffers`]), and the [`JoltBackend::metal`]
//! constructor. No prover slot runs on the device yet — `metal()` is
//! [`JoltBackend::optimized`] plus a prewarmed, fail-closed device context;
//! later waves overwrite slots with `Metal*` kernels that fall back to their
//! optimized twins under [`metal_gate`].
//!
//! # Environment convention (introduced here)
//!
//! Device paths are threshold-gated: a slot asks
//! `metal_gate(kind, work_items)` and runs on the GPU only when the answer
//! is true. Dispatch has a fixed latency floor (~97 µs per synchronous
//! command-buffer round trip on an M4 — see the `metal_microbench`
//! example), so small instances always stay on the CPU.
//!
//! - `JOLT_METAL_DISABLE=1` — kill every device path (any value other than
//!   `0`/empty counts).
//! - `JOLT_METAL_MIN_TERMS=N` — global minimum work-item count for device
//!   dispatch (default [`DEFAULT_MIN_TERMS`]).
//! - `JOLT_METAL_MIN_TERMS_<SLOT>=N` — per-slot override; `<SLOT>` is the
//!   gate's `kind` uppercased with non-alphanumerics mapped to `_`
//!   (e.g. kind `"bind"` → `JOLT_METAL_MIN_TERMS_BIND`).
//!
//! Environment is read per call (gates fire once per slot per proof —
//! nothing hot).

mod buffers;
mod commitment;
mod error;
mod field;
mod g1;
mod runtime;
mod slots;
pub mod testing;

pub use buffers::{DeviceBuffer, OwnedDeviceBuffer, PageAlignedVec, PAGE_SIZE};
pub use error::MetalError;
pub use field::{
    fr_as_u32s, fr_as_u32s_mut, fr_from_u32_limbs, fr_to_u32_limbs, FR_U32_LIMBS,
    G1_AFFINE_U32_STRIDE,
};
pub use g1::{bases_as_u32s, g1_seg_sum_dispatch, g1_seg_sums, jac_from_device_limbs, JAC_U32S};
pub use runtime::{
    ComputePass, KernelId, MetalContext, PendingPass, MAX_EVAL_POINTS, THREADGROUP_SIZE,
};
pub use slots::{
    MetalHammingWeightClaimReduction, MetalIncClaimReduction, MetalInstructionReadRaf,
    MetalRamHammingBooleanity, MetalRamRafEvaluation,
};

use jolt_field::Fr;
use jolt_openings::{CommitmentScheme, StreamingCommitment};

use crate::optimized::hamming_weight_claim_reduction::OptimizedHammingWeightClaimReduction;
use crate::optimized::inc_claim_reduction::OptimizedIncClaimReduction;
use crate::optimized::ram_hamming_booleanity::OptimizedRamHammingBooleanity;
use crate::optimized::OptimizedBackend;
use crate::JoltBackend;

/// Default [`metal_gate`] threshold, from `metal_microbench` on the target
/// M4 (2026-07-29): a synchronous dispatch round trip floors at ~97 µs
/// (D1), and the pure-streaming bind — the LEAST compute-dense kernel —
/// crosses over against the all-core CPU between 2^18 (device 12% behind)
/// and 2^19 (device 15% ahead) input elements (D2b). Real sumcheck kernels
/// do strictly more arithmetic per byte than a bare fold, which moves their
/// cutover below bind's, so 2^18 is the break-even-or-better default;
/// bind-shaped outliers can be raised per slot via
/// `JOLT_METAL_MIN_TERMS_<SLOT>`.
pub const DEFAULT_MIN_TERMS: usize = 1 << 18;

/// Should `kind` run on the device for `work_items` elements? See the
/// module docs for the environment convention.
pub fn metal_gate(kind: &str, work_items: usize) -> bool {
    if std::env::var("JOLT_METAL_DISABLE").is_ok_and(|v| !v.is_empty() && v != "0") {
        return false;
    }
    work_items >= min_terms(kind)
}

fn min_terms(kind: &str) -> usize {
    let suffix: String = kind
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect();
    parse_env(&format!("JOLT_METAL_MIN_TERMS_{suffix}"))
        .or_else(|| parse_env("JOLT_METAL_MIN_TERMS"))
        .unwrap_or(DEFAULT_MIN_TERMS)
}

fn parse_env(name: &str) -> Option<usize> {
    std::env::var(name).ok()?.trim().parse().ok()
}

impl<PCS> JoltBackend<Fr, PCS>
where
    PCS: CommitmentScheme<Field = Fr>,
{
    /// The Metal backend: [`JoltBackend::optimized`] with device kernels
    /// installed over the converted slots, each gated by [`metal_gate`] and
    /// falling back to its optimized twin below threshold or on any device
    /// failure. Fail-closed: no Metal device, a shader that fails to
    /// compile, or a pipeline that cannot be built errors HERE, never
    /// mid-proof.
    ///
    /// `Fr`-concrete (unlike the other constructors): the device tier is
    /// BN254 Montgomery arithmetic — the shaders' limb layout IS `Fr`'s.
    ///
    /// Installed device slots: the W2 sumcheck four (inc / hamming-weight
    /// claim reductions, ram hamming booleanity, ram RAF evaluation) plus
    /// `commit` — Dory one-hot tier-1 G1 accumulation on the device,
    /// pipelined with CPU tier-2 pairings (W3a; `(Fr, DoryScheme)`
    /// instantiations only — everything else keeps the optimized slot).
    pub fn metal() -> Result<Self, MetalError>
    where
        PCS: StreamingCommitment + 'static,
    {
        let context = MetalContext::global()?;
        let mut backend = Self::optimized();
        backend.inc_claim_reduction = Box::new(MetalIncClaimReduction {
            fallback: OptimizedIncClaimReduction,
        });
        backend.hamming_weight_claim_reduction = Box::new(MetalHammingWeightClaimReduction {
            fallback: OptimizedHammingWeightClaimReduction,
        });
        backend.ram_hamming_booleanity = Box::new(MetalRamHammingBooleanity {
            fallback: OptimizedRamHammingBooleanity,
        });
        backend.ram_raf_evaluation = Box::new(MetalRamRafEvaluation {
            fallback: OptimizedBackend,
        });
        backend.instruction_read_raf = Box::new(MetalInstructionReadRaf);
        if let Some(slot) = commitment::dory_commit_slot::<Fr, PCS>() {
            backend.commit = slot;
            tracing::info!(device = %context.device_name(), "Metal backend ready (commit on device)");
        } else {
            tracing::info!(device = %context.device_name(), "Metal backend ready");
        }
        Ok(backend)
    }
}
