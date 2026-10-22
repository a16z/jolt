//! CUDA precommitted, increment, register and instruction claim reductions.
//! Also provides instruction-input checking, RAM output and value checking,
//! RAM hamming-weight booleanity, Spartan outer constraints, product virtualization,
//! shift checking, register-value evaluation, and RAM RAF evaluation.
//!
//! [`JoltBackend::cuda`] replaces the advice, bytecode, and program-image
//! cycle/address slots, RAM/register increment reduction, register-value
//! reduction, instruction operand reduction, instruction-input checking,
//! RAM output and value checking, hamming-weight booleanity, Spartan shift
//! checking, register-value evaluation, and RAM RAF evaluation in the optimized registry.
//! Spartan outer and product uni-skip and remainder slots use CUDA for RV64;
//! field-inline retains the optimized implementations.
//! [`CudaDoryScheme`] selects the PCS separately and currently forwards to CPU Dory
//! with shared proof and setup types.
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
//! sums; the host-input API transfers those sums and offsets between levels,
//! while the device-input API keeps them on the device.
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
//! Weighted G1 bucket sums split across blocks and merge their partials per row;
//! window folding combines those results from the most significant window down.
//! Fr row MSMs extract scalar windows and build buckets on the device. Windows
//! batch within a memory budget and otherwise accumulate sequentially.
//! Signed row MSMs reuse those buckets with 64-bit magnitudes and separate signs;
//! host staging rejects inputs outside `[-u64::MAX, u64::MAX]`.
//! One-hot row MSMs count and scatter indices on device, using shared-memory
//! tallies when they fit. The host scans counts before the segmented G1 sums.
//! For fully covered chunks, dominant buckets can use the total base sum minus
//! the other buckets; cold cycles disable this complement shortcut.
//! G1 GLV kernels split scalars into two signed components and emit matching
//! original/endomorphism point spans. Magnitudes stay on device; signs return
//! to the host for scheduling.
//! Shared-scalar row MSMs reuse one coefficient vector across a matrix of points.
//! Larger batches share a window schedule when the device memory budget permits;
//! other batches use a GLV block reduction per row.
//! Resident G1 MSMs read a device point span and store the sum back on device;
//! their GLV-expanded affine bases and Pippenger accumulators stay on the GPU.
//! G2 sums use Jacobian coordinates over Fq2 and signed base indices. Segments
//! run independently unless a large segment benefits from warp cooperation.
//! G2 scalar decomposition sums precomputed power-of-two entries on device.
//! Four component arrays retain their magnitudes there and return sign bytes
//! to the host, with `1` meaning negative rather than the CPU's positive flag.
//! G2 Frobenius expansion retains Jacobian coordinates and emits four point
//! spans matching those components, using coefficients derived from Arkworks.
//! Resident G2 MSMs build signed buckets, merge weighted partial sums and fold
//! windows on device. They decline shapes beyond the batch budget or segment
//! estimate; their output stays in the device buffer.
//! Fq6 and Fq12 arithmetic use Montgomery Fq2 coefficients, with specialized
//! sparse products for pairing line evaluations.
//! Line evaluation scales the coefficients by G1 coordinates and multiplies
//! the sparse result into each thread's Fq12 accumulator.
//! G2 Miller steps use homogeneous Fq2 coordinates and emit line coefficients.
//! Each host thread caches curve and Frobenius constants per device.
//! Fq12 products reduce independent segments in bounded chunks, repeating on
//! device until only one product per segment remains for download.
//! The warp Miller loop assigns one warp per G1/G2 pair. Lanes cooperate on
//! the six Fq2 accumulator coefficients; normalization and loop steps stay on
//! device, and identity pairs contribute one to the segment product.
//! The block Miller loop assigns one pair per thread, keeping its full Fq12
//! accumulator local. Batches with at most 256 total pairs use the warp loop;
//! larger batches use the block loop.
//! When multiple block-loop segments share a G2 span, CUDA prepares its line
//! coefficients once on device, using at most 512 MiB. The prepared loop combines
//! four strided pairs per thread to share accumulator squarings. Larger coefficient
//! tables use the direct block loop; identity pairs skip coefficient reads.
//! Prefix/suffix sumchecks first bind compact weight and accumulator tables,
//! then fold the original field columns on device to build the suffix phase.
//! Equality and shifted-equality weights share this engine; final column claims
//! are read from the fully bound suffix tables.
//! Windowed prefix/suffix rounds fold each resident window on its owning GPU.
//! Prefix accumulators are summed on the host; suffix folds are concatenated
//! in window order and uploaded to the device running the round loop.
//! Lookup suffixes evaluate masked bit strings with integer arithmetic on
//! device, matching the optimized lookup tables' `u64` results. One thread
//! handles each input; field promotion is left to the consuming sumcheck.
//! Prefix phase tables use the canonical host definitions for every prefix;
//! optimized materializes only prefixes used by present lookup tables. CUDA
//! uploads the phase tables, evaluates them at 0 and 2, and binds their rows
//! in batched device launches before downloading the next phase's checkpoints.
//! Sparse read/write matrices keep native value checkpoints and merge adjacent
//! cycle rows on device. Coefficient lookup tables grow with each bind, as in
//! optimized. CUDA writes merged rows to fresh arrays; optimized's indexed and
//! direct stages compact existing entries in place.
//! Cycle shards sum round coefficients on the host and gather their remaining
//! rows on GPU 0 before binding the shard variables.
//! Address-first rounds sort entry indices on the host and permute their device
//! columns, then merge adjacent addresses with initial-value checkpoints.
//! After the last address bind, dense cycle columns use split-eq reductions.
//! CUDA uses this sparse matrix path for both RAM and registers; optimized
//! registers instead scan compact access chunks with raw increment checkpoints.
//! Advice reductions upload oracle values and permute their rows on device into
//! the shared opening schedule. Inactive rounds halve the running scale; active
//! rounds reduce the product at 0 and 2 and interpolate with the incoming claim.
//! The cycle phase parks its device rows for the address phase; optimized parks
//! host polynomial tables across that same boundary.
//! Committed bytecode uses the canonical host coefficient builder. CUDA forms
//! the lane-weighted equality grid on device in the requested polynomial order,
//! then permutes both rows for the shared cycle/address reduction engine.
//! Program-image reductions upload padded words as field elements and compute
//! shifted RAM equality weights per entry on device. Optimized builds those
//! weights in aligned blocks and permutes words before field conversion.
//! Dense booleanity rounds reduce both endpoint values and the quadratic
//! coefficient against split equality tables. RAM hamming weights use a dense
//! field table for these rounds; optimized starts with bit-pattern histograms
//! before materializing its table.

use jolt_field::JoltField;
use jolt_openings::CommitmentScheme;

use crate::{JoltBackend, KernelError, ModeStreamingCommitment};

mod advice_claim_reduction;
mod bytecode_claim_reduction;
mod common;
mod dory;
mod hamming_weight_claim_reduction;
mod inc_claim_reduction;
mod instruction_claim_reduction;
mod instruction_input;
#[cfg(test)]
mod instruction_read_raf;
mod program_image_claim_reduction;
mod ram_hamming_booleanity;
mod ram_output_check;
mod ram_raf_evaluation;
mod ram_val_check;
mod registers_claim_reduction;
#[cfg(test)]
mod registers_read_write;
mod registers_val_evaluation;
#[cfg(not(feature = "field-inline"))]
mod spartan_outer;
#[cfg(not(feature = "field-inline"))]
mod spartan_product;
mod spartan_shift;
mod witness;

pub use common::context::{
    context_for, device_count, device_memory_used, enter_device, request_devices, shared_context,
    CudaKernelContext, DEVICE_COUNT_VARIABLE,
};
pub use common::device::{as_fr_slice, fr_into, DeviceFrVec, LIMBS};
pub use common::error::CudaError;
pub use common::lt_poly::DeviceLtPolynomial;
pub use common::xfer_stats;
pub use dory::CudaDoryScheme;

/// Stateless slot provider; witness-dependent device state belongs to the
/// bound proof session rather than the reusable backend registry.
pub struct CudaBackend;

/// Initializes the device pool on first use and reports whether its selected
/// device has a usable kernel context.
pub fn device_available() -> bool {
    shared_context().is_some()
}

pub(crate) fn require_context<F: JoltField>() -> Result<&'static CudaKernelContext, KernelError<F>>
{
    shared_context().ok_or(KernelError::Unsupported {
        reason: "no CUDA device is present",
    })
}

impl<F, PCS> JoltBackend<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    /// Replace supported claim reductions, instruction-input checking, RAM output
    /// and value checking, RAM hamming booleanity, hamming-weight claim reduction,
    /// register-value evaluation, Spartan shift, RAM RAF evaluation, and RV64 Spartan
    /// outer and product kernels with CUDA implementations.
    ///
    /// Construction is lazy: the first device operation initializes CUDA.
    pub fn cuda() -> Self
    where
        PCS: ModeStreamingCommitment,
    {
        let mut backend = Self::optimized();
        backend.trusted_advice_cycle = Box::new(CudaBackend);
        backend.untrusted_advice_cycle = Box::new(CudaBackend);
        backend.bytecode_reduction_cycle = Box::new(CudaBackend);
        backend.program_image_reduction_cycle = Box::new(CudaBackend);
        backend.trusted_advice_address = Box::new(CudaBackend);
        backend.untrusted_advice_address = Box::new(CudaBackend);
        backend.bytecode_reduction_address = Box::new(CudaBackend);
        backend.program_image_reduction_address = Box::new(CudaBackend);
        backend.ram_hamming_booleanity = Box::new(CudaBackend);
        backend.hamming_weight_claim_reduction = Box::new(CudaBackend);
        backend.inc_claim_reduction = Box::new(CudaBackend);
        backend.registers_claim_reduction = Box::new(CudaBackend);
        backend.instruction_claim_reduction = Box::new(CudaBackend);
        backend.instruction_input = Box::new(CudaBackend);
        backend.ram_output_check = Box::new(CudaBackend);
        backend.ram_raf_evaluation = Box::new(CudaBackend);
        backend.ram_val_check = Box::new(CudaBackend);
        backend.registers_val_evaluation = Box::new(CudaBackend);
        backend.spartan_shift = Box::new(CudaBackend);
        #[cfg(not(feature = "field-inline"))]
        {
            backend.spartan_outer_uniskip = Box::new(CudaBackend);
            backend.spartan_outer_remainder = Box::new(CudaBackend);
            backend.spartan_product_uniskip = Box::new(CudaBackend);
            backend.spartan_product_remainder = Box::new(CudaBackend);
        }
        backend
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
