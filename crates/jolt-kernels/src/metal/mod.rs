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
mod dory_folds;
mod dory_reduce;
mod error;
mod field;
mod g1;
mod g2;
mod hint_combine;
pub mod miller;
pub mod montgomery;
mod runtime;
mod slots;
pub mod testing;

pub use buffers::{DeviceBuffer, OwnedDeviceBuffer, PageAlignedVec, PAGE_SIZE};
#[cfg(feature = "bench-utils")]
pub use commitment::{G1SegBenchCase, G1SegBenchFixture, G1SegBenchSample};
pub use dory_folds::{
    g1_scalar_mul_add_device, g2_fixed_base_mul_device, g2_scalar_mul_add_device,
};
pub use error::MetalError;
pub use field::{
    fr_as_u32s, fr_as_u32s_mut, fr_from_u32_limbs, fr_to_u32_limbs, FR_U32_LIMBS,
    G1_AFFINE_U32_STRIDE, G2_AFFINE_U32_STRIDE,
};
pub use g1::{bases_as_u32s, g1_seg_sum_dispatch, g1_seg_sums, jac_from_device_limbs, JAC_U32S};
pub use g2::{g2_bases_as_u32s, g2_jac_from_device_limbs, G2_JAC_U32S};
#[cfg(feature = "bench-utils")]
pub use runtime::VariantPipeline;
pub use runtime::{
    ComputePass, KernelId, MetalContext, PendingPass, MAX_EVAL_POINTS, THREADGROUP_SIZE,
};
#[cfg(feature = "bench-utils")]
pub use slots::{IrrPhaseScanFixture, IrrSuffixScanFixture};
pub use slots::{
    MetalBooleanityCycle, MetalBytecodeReadRafCycle, MetalHammingWeightClaimReduction,
    MetalIncClaimReduction, MetalInstructionClaimReduction, MetalInstructionInput,
    MetalInstructionRaVirtualization, MetalInstructionReadRaf, MetalJointOpening,
    MetalOuterRemainder, MetalOuterUniskip, MetalProductRemainder, MetalProductUniskip,
    MetalRamHammingBooleanity, MetalRamRaVirtualization, MetalRamRafEvaluation,
    MetalRamReadWriteChecking, MetalRegistersReadWriteChecking, MetalRegistersValEvaluation,
};

use jolt_field::Fr;
use jolt_openings::{CommitmentScheme, StreamingCommitment};

use crate::optimized::hamming_weight_claim_reduction::OptimizedHammingWeightClaimReduction;
use crate::optimized::inc_claim_reduction::OptimizedIncClaimReduction;
use crate::optimized::instruction_claim_reduction::OptimizedInstructionClaimReduction;
use crate::optimized::instruction_input::OptimizedInstructionInput;
use crate::optimized::ram_hamming_booleanity::OptimizedRamHammingBooleanity;
use crate::optimized::OptimizedBackend;
use crate::JoltBackend;

/// Default [`metal_gate`] threshold. W1's `metal_microbench` put the
/// pure-streaming bind's cutover at 2^18–2^19 elements (D2b, ~97 µs
/// dispatch floor), the original default. With every W2–W5 slot live, the
/// W5b end-to-end sweep (sha2-chain @2^16/2^18/2^20, modular_benchmark,
/// same-arm pairs) moved the optimum DOWN: 2^16 beat 2^18 at every scale
/// (−2-3% e2e — real slots are more compute-dense than a bare bind, and
/// the big slots' shrinking tail rounds stay on device two rounds longer),
/// while 2^14 overshot the dispatch floor and 2^20 lost outright. No slot
/// regressed at small scales, so no per-slot override ships;
/// `JOLT_METAL_MIN_TERMS_<SLOT>` remains for outliers.
pub const DEFAULT_MIN_TERMS: usize = 1 << 16;

/// Should `kind` run on the device for `work_items` elements? See the
/// module docs for the environment convention.
pub fn metal_gate(kind: &str, work_items: usize) -> bool {
    if std::env::var("JOLT_METAL_DISABLE").is_ok_and(|v| !v.is_empty() && v != "0") {
        return false;
    }
    work_items >= min_terms(kind)
}

fn env_suffix(kind: &str) -> String {
    kind.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() {
                c.to_ascii_uppercase()
            } else {
                '_'
            }
        })
        .collect()
}

fn min_terms(kind: &str) -> usize {
    parse_env(&format!("JOLT_METAL_MIN_TERMS_{}", env_suffix(kind)))
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
        backend.spartan_outer_uniskip = Box::new(MetalOuterUniskip::new());
        backend.spartan_outer_remainder = Box::new(MetalOuterRemainder::new());
        backend.spartan_product_uniskip = Box::new(MetalProductUniskip::new());
        backend.spartan_product_remainder = Box::new(MetalProductRemainder::new());
        backend.inc_claim_reduction = Box::new(MetalIncClaimReduction {
            fallback: OptimizedIncClaimReduction,
        });
        backend.hamming_weight_claim_reduction = Box::new(MetalHammingWeightClaimReduction {
            fallback: OptimizedHammingWeightClaimReduction,
        });
        backend.instruction_claim_reduction = Box::new(MetalInstructionClaimReduction {
            fallback: OptimizedInstructionClaimReduction,
        });
        backend.ram_hamming_booleanity = Box::new(MetalRamHammingBooleanity {
            fallback: OptimizedRamHammingBooleanity,
        });
        backend.instruction_input = Box::new(MetalInstructionInput {
            fallback: OptimizedInstructionInput,
        });
        backend.ram_raf_evaluation = Box::new(MetalRamRafEvaluation {
            fallback: OptimizedBackend,
        });
        backend.ram_read_write = Box::new(MetalRamReadWriteChecking::new());
        backend.registers_read_write = Box::new(MetalRegistersReadWriteChecking::new());
        backend.registers_val_evaluation = Box::new(MetalRegistersValEvaluation);
        backend.joint_opening = Box::new(MetalJointOpening {
            fallback: OptimizedBackend,
        });
        backend.instruction_read_raf = Box::new(MetalInstructionReadRaf);
        backend.bytecode_read_raf_cycle = Box::new(MetalBytecodeReadRafCycle);
        backend.booleanity_cycle = Box::new(MetalBooleanityCycle);
        backend.instruction_ra_virtualization = Box::new(MetalInstructionRaVirtualization);
        backend.ram_ra_virtualization = Box::new(MetalRamRaVirtualization);
        if let Some(slot) = commitment::dory_commit_slot::<Fr, PCS>() {
            backend.commit = slot;
            tracing::info!(device = %context.device_name(), "Metal backend ready (commit on device)");
        } else {
            tracing::info!(device = %context.device_name(), "Metal backend ready");
        }
        Ok(backend)
    }
}
