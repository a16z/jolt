//! BN254 CUDA kernels behind the same relation slots as [`crate::optimized`].
//!
//! The host owns relation geometry, challenges, interpolation, and the proof
//! transcript. Device kernels reduce round evaluations and bind resident
//! columns; exact field arithmetic preserves the optimized round polynomials.
//!
//! Shared implementation choices:
//!
//! - Packed trace windows and derived columns live in the [`ProofSession`],
//!   so successive relations can reuse uploads. Device contexts and compiled
//!   modules are process-wide; witness data is not.
//! - Cycle work is partitioned into aligned windows. Sharded round kernels
//!   combine local messages on the host and move the remaining tables to GPU 0
//!   once the local cycle variables are bound. Small domains use one device.
//! - Split-eq and prefix/suffix factorizations follow the optimized algorithms,
//!   with GPU reductions in place of Rayon walks. Materialization thresholds
//!   differ where device gathers and dense arithmetic have different costs.
//!
//! [`JoltBackend::cuda`] starts from the optimized registry. Field-inline
//! Spartan, bytecode read/RAF, and field-register slots retain optimized
//! implementations. [`CudaDoryScheme`] selects CUDA PCS operations separately
//! from the relation registry; using `DoryScheme` retains its CPU opening path.

use crate::KernelError;
use crate::ProofSession;
use jolt_field::JoltField;
use jolt_openings::CommitmentScheme;
use jolt_witness::JoltWitnessPlane;

use crate::JoltBackend;

mod advice_claim_reduction;
mod booleanity;
mod bytecode_claim_reduction;
#[cfg(not(feature = "field-inline"))]
mod bytecode_read_raf;
mod commitment;
mod common;
mod dory;
mod hamming_weight_claim_reduction;
mod inc_claim_reduction;
mod instruction_claim_reduction;
mod instruction_input;
mod instruction_ra_virtualization;
mod instruction_read_raf;
mod opening;
mod program_image_claim_reduction;
mod ram_hamming_booleanity;
mod ram_output_check;
mod ram_ra_claim_reduction;
mod ram_ra_virtualization;
mod ram_raf_evaluation;
mod ram_read_write;
mod ram_val_check;
mod registers_claim_reduction;
mod registers_read_write;
mod registers_val_evaluation;
#[cfg(not(feature = "field-inline"))]
mod spartan_outer;
#[cfg(not(feature = "field-inline"))]
mod spartan_product;
mod spartan_shift;
mod witness;

pub use commitment::DeviceTier1Commitment;
pub use common::context::{
    context_for, device_count, device_memory_used, enter_device, request_devices, shared_context,
    CudaKernelContext, DEVICE_COUNT_VARIABLE,
};
pub use common::device::{as_fr_slice, fr_into, fr_vec_into, DeviceFrVec, LIMBS};
pub use common::error::CudaError;
pub use common::lt_poly::DeviceLtPolynomial;
pub use common::msm::{AffineLimbs, DeviceSegments, JacobianLimbs, SegmentMode};
pub use common::one_hot_fold::{DeviceOneHotColumns, FoldTuning, LANES, SHARED_BUDGET};
pub use common::xfer_stats;
pub use dory::CudaDoryScheme;
pub use instruction_read_raf::address_driver::DeviceAddressPhase;
pub use instruction_read_raf::address_phase::{init_raf_buckets, init_suffix_buckets, DeviceRows};

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
    /// Replace the supported optimized slots with CUDA implementations.
    ///
    /// Construction is lazy: the first device operation initializes CUDA.
    /// The PCS must accept device-produced row commitments; choose
    /// [`CudaDoryScheme`] to also accelerate clear Dory openings.
    pub fn cuda() -> Self
    where
        PCS: DeviceTier1Commitment,
    {
        let mut backend = Self::optimized();
        #[cfg(not(feature = "field-inline"))]
        {
            backend.bytecode_read_raf_address = Box::new(CudaBackend);
            backend.bytecode_read_raf_cycle = Box::new(CudaBackend);
            backend.spartan_outer_uniskip = Box::new(CudaBackend);
            backend.spartan_outer_remainder = Box::new(CudaBackend);
            backend.spartan_product_uniskip = Box::new(CudaBackend);
            backend.spartan_product_remainder = Box::new(CudaBackend);
        }
        backend.booleanity_address = Box::new(CudaBackend);
        backend.booleanity_cycle = Box::new(CudaBackend);
        backend.instruction_ra_virtualization = Box::new(CudaBackend);
        backend.instruction_read_raf = Box::new(CudaBackend);
        backend.registers_val_evaluation = Box::new(CudaBackend);
        backend.hamming_weight_claim_reduction = Box::new(CudaBackend);
        backend.inc_claim_reduction = Box::new(CudaBackend);
        backend.instruction_claim_reduction = Box::new(CudaBackend);
        backend.instruction_input = Box::new(CudaBackend);
        backend.registers_claim_reduction = Box::new(CudaBackend);
        backend.ram_hamming_booleanity = Box::new(CudaBackend);
        backend.ram_output_check = Box::new(CudaBackend);
        backend.ram_raf_evaluation = Box::new(CudaBackend);
        backend.ram_ra_claim_reduction = Box::new(CudaBackend);
        backend.ram_ra_virtualization = Box::new(CudaBackend);
        backend.ram_val_check = Box::new(CudaBackend);
        backend.registers_read_write = Box::new(CudaBackend);
        backend.ram_read_write = Box::new(CudaBackend);
        backend.spartan_shift = Box::new(CudaBackend);
        backend.trusted_advice_cycle = Box::new(CudaBackend);
        backend.untrusted_advice_cycle = Box::new(CudaBackend);
        backend.bytecode_reduction_cycle = Box::new(CudaBackend);
        backend.program_image_reduction_cycle = Box::new(CudaBackend);
        backend.trusted_advice_address = Box::new(CudaBackend);
        backend.untrusted_advice_address = Box::new(CudaBackend);
        backend.bytecode_reduction_address = Box::new(CudaBackend);
        backend.program_image_reduction_address = Box::new(CudaBackend);
        backend.commit = Box::new(CudaBackend);
        backend.ram_initial_openings = Box::new(CudaBackend);
        backend.joint_opening = Box::new(CudaBackend);
        backend
    }
}

/// Preload packed trace windows and their atom columns into a proof session.
///
/// Unlike optimized's host-row access, CUDA uploads each device's window,
/// including any successor row needed at its boundary. All uploads and
/// extraction kernels finish before this returns successfully. Subsequent
/// CUDA relations reuse the residency with the same session and witness.
///
/// The caller must supply a representable `1usize << log_t` cycle count
/// covered by the witness.
pub fn warm_shared_witness<F: JoltField>(
    session: &mut ProofSession,
    witness: &dyn JoltWitnessPlane<F>,
    log_t: usize,
) -> Result<(), KernelError<F>> {
    let cycles = 1usize << log_t;
    let _ = require_context::<F>()?;
    for (ordinal, window) in self::common::devices::witness_windows(cycles)
        .iter()
        .enumerate()
    {
        let _device = enter_device(ordinal);
        let device = context_for(ordinal).ok_or(KernelError::InvariantViolation {
            reason: "a witness warm-up window names an absent device",
        })?;
        let _ = self::witness::session_window_residency(device, session, witness, cycles, window)?;
        device.stream().synchronize().map_err(CudaError::from)?;
    }
    Ok(())
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use cudarc::driver::CudaContext;
    use jolt_field::{Fr, Ring};

    use std::sync::Arc;

    use super::{as_fr_slice, fr_into, fr_vec_into, shared_context, DeviceFrVec, LIMBS};

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

    #[test]
    fn raw_limb_readback_respects_the_logical_length() {
        let context = require_device!();
        let values = [
            Fr::from_u64(0),
            Fr::from_u64(1),
            -Fr::from_u64(1),
            Fr::from_u128(u128::MAX),
            Fr::from_u64(0x0123_4567_89ab_cdef),
        ];
        let words: Vec<_> = values
            .iter()
            .flat_map(|value| value.inner_limbs())
            .collect();
        for len in [0usize, 1, 3, values.len()] {
            let buffer = context.upload_u64_slice(&words).expect("upload limbs");
            let device = DeviceFrVec::from_parts(Arc::clone(context.stream()), buffer, len);
            assert_eq!(
                device.to_limbs().expect("download raw limbs"),
                words[..len * LIMBS],
            );
            assert_eq!(
                device.to_host().expect("download field values"),
                values[..len],
            );
        }
    }

    #[test]
    fn field_reinterprets_round_trip_for_fr() {
        let values = sample(4);
        assert_eq!(as_fr_slice::<Fr>(&values).expect("Fr slice"), &values[..]);
        assert_eq!(fr_vec_into::<Fr>(values.clone()).expect("Fr vec"), values,);
        assert_eq!(fr_into::<Fr>(values[0]).expect("Fr scalar"), values[0]);
    }
}
