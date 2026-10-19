//! CUDA backend selection using optimized compute kernels and the host witness plane.
//!
//! [`JoltBackend::cuda`] uses the optimized registry. [`CudaDoryScheme`] selects
//! the PCS independently and shares its proof and setup types with `DoryScheme`.
//! Both entry points execute entirely on the CPU.
//!
//! The shared CUDA runtime owns process-wide device contexts and compiled
//! modules. [`DeviceFrVec`] stores BN254 Montgomery limbs on its owning stream;
//! [`CudaKernelContext::fr_identity`] copies those elements on the device.
//! Elementwise field arithmetic uses CUDA kernels over that representation
//! and synchronizes the stream before returning each output buffer.
//! Polynomial binding writes a fresh device buffer for either variable order;
//! summation reduces on the device and downloads only the final field element.
//! Equality tables use separate input/output buffers at each level to avoid races;
//! less-than tables expand in place. Levels run in order on the owning stream.
//! Integer inputs upload as compact limbs and convert to Montgomery form on the
//! device. Host `i128` inputs use separate signs and magnitudes; device `i128`
//! inputs use two's-complement limbs.
//! `u32` prefix scans combine per-block scans with recursively scanned block
//! sums; the host-input API transfers those sums and offsets between levels.
//! [`DeviceLtPolynomial`] keeps its shift and split factors separate throughout
//! binding; optimized stores the shift in its high table and later combines the factors.
//! Weighted half-fold kernels promote packed integers when loading them.
//! Contiguous rows use a block reduction; strided columns use one thread per output.
//! Sum-of-products rounds evaluate finite points and leading coefficients on
//! device, then reduce the per-block results for each evaluation separately.
//! Gruen-weighted rounds cache the host's split-equality levels on device and
//! return both endpoints and the leading coefficient to the shared host
//! interpolation routines.
//! Lazy one-hot selectors gather from hot indices and challenge-scaled address
//! tables for both binding orders. CUDA materializes a dense table at the third
//! bind (`T/8` entries); optimized's shared lazy-RA path normally waits until `T/16`.
//! Dense-product rounds combine dense factors with lazy one-hot and split
//! less-than factors. Shards sum their round evaluations on the host, then
//! gather their final scalars onto one device to bind the remaining shard bits.
//! Integer-weighted scatters accumulate unreduced products in split 32-bit
//! lanes using atomic additions, then perform one field reduction per bucket.
//! One-hot cycle folds use shared-memory buckets when the address domain fits
//! and otherwise accumulate into global slots before field reduction.
//! BN254 base-field arithmetic supports curve coordinates. Batch inversion
//! assigns independent chunks to GPU threads and preserves zero entries.
//! G1 kernels double and add Jacobian points, including mixed affine addends.
//! Batched affine-pair addition uses those inversions for finite, distinct-x pairs.
//! G1 normalization batch-inverts Jacobian z-coordinates and keeps the resulting
//! affine base table on the device.
//! Segmented G1 sums select thread, warp, or block reductions according to segment
//! sizes. One-hot chunks build their segment plan on the host and sum on device.

use jolt_field::JoltField;
use jolt_openings::CommitmentScheme;

use crate::{JoltBackend, ModeStreamingCommitment};

mod common;
mod dory;

pub use common::context::{
    context_for, device_count, device_memory_used, enter_device, request_devices, shared_context,
    CudaKernelContext, DEVICE_COUNT_VARIABLE,
};
pub use common::device::{as_fr_slice, fr_into, DeviceFrVec, LIMBS};
pub use common::error::CudaError;
pub use common::lt_poly::DeviceLtPolynomial;
pub use common::xfer_stats;
pub use dory::CudaDoryScheme;

/// Initializes the device pool on first use and reports whether its selected
/// device has a usable kernel context.
pub fn device_available() -> bool {
    shared_context().is_some()
}

impl<F, PCS> JoltBackend<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    /// Select the optimized kernel slots through the CUDA backend entry point.
    pub fn cuda() -> Self
    where
        PCS: ModeStreamingCommitment,
    {
        Self::optimized()
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use cudarc::driver::CudaContext;
    use jolt_field::{Fr, Ring};

    use super::shared_context;

    macro_rules! require_device {
        () => {
            match shared_context() {
                Some(context) => context,
                None => return,
            }
        };
    }

    fn sample(count: usize) -> Vec<Fr> {
        (0..count as u64).map(|i| Fr::from_u64(i * 7 + 3)).collect()
    }

    #[test]
    fn kernel_source_compiles_when_a_device_is_present() {
        if CudaContext::new(0).is_err() {
            return;
        }
        assert!(
            shared_context().is_some(),
            "a CUDA device is present but the kernel module failed to build; every \
             device test would otherwise skip and report success",
        );
    }

    #[test]
    fn upload_launch_download_round_trips() {
        let context = require_device!();
        for count in [1usize, 5, 256, 1000] {
            let values = sample(count);
            let device = context.upload(&values).expect("upload");
            assert_eq!(device.len(), count);
            let copied = context.fr_identity(&device).expect("launch identity");
            assert_eq!(copied.to_host().expect("download"), values);
        }
    }

    #[test]
    fn empty_upload_round_trips() {
        let context = require_device!();
        let device = context.upload(&[]).expect("upload empty");
        assert!(device.is_empty());
        assert!(device.to_host().expect("download empty").is_empty());
    }

    #[test]
    fn first_reads_element_zero() {
        let context = require_device!();
        let values = sample(64);
        let device = context.upload(&values).expect("upload");
        assert_eq!(device.first().expect("first"), values[0]);
    }

    #[test]
    fn device_clone_copies_contents() {
        let context = require_device!();
        let values = sample(32);
        let device = context.upload(&values).expect("upload");
        let clone = device.try_clone().expect("clone");
        assert_eq!(clone.to_host().expect("download"), values);
    }

    #[test]
    fn pool_reuse_does_not_leak_stale_limbs() {
        let context = require_device!();
        let large = sample(512);
        let large_device = context.upload(&large).expect("upload large");
        assert_eq!(large_device.to_host().expect("download large"), large);
        let small: Vec<Fr> = sample(8).into_iter().map(|v| v + Fr::from_u64(1)).collect();
        let small_device = context.upload(&small).expect("upload small");
        assert_eq!(small_device.to_host().expect("download small"), small);
    }
}
