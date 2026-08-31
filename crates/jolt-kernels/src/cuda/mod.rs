use jolt_field::Field;
use jolt_openings::CommitmentScheme;

use crate::JoltBackend;

mod advice_claim_reduction;
mod booleanity;
mod bytecode_claim_reduction;
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
mod spartan_outer;
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

pub struct CudaBackend;

pub fn device_available() -> bool {
    shared_context().is_some()
}

pub(crate) fn require_context<F: jolt_field::FieldCore>(
) -> Result<&'static CudaKernelContext, crate::KernelError<F>> {
    shared_context().ok_or(crate::KernelError::Unsupported {
        reason: "no CUDA device is present",
    })
}

impl<F, PCS> JoltBackend<F, PCS>
where
    F: Field,
    PCS: CommitmentScheme<Field = F>,
{
    pub fn cuda() -> Self
    where
        PCS: DeviceTier1Commitment,
    {
        let mut backend = Self::reference();
        if !device_available() {
            return backend;
        }
        backend.booleanity_address = Box::new(CudaBackend);
        backend.booleanity_cycle = Box::new(CudaBackend);
        backend.bytecode_read_raf_address = Box::new(CudaBackend);
        backend.bytecode_read_raf_cycle = Box::new(CudaBackend);
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
        backend.spartan_outer_uniskip = Box::new(CudaBackend);
        backend.spartan_outer_remainder = Box::new(CudaBackend);
        backend.spartan_shift = Box::new(CudaBackend);
        backend.spartan_product_uniskip = Box::new(CudaBackend);
        backend.spartan_product_remainder = Box::new(CudaBackend);
        backend.trusted_advice_cycle = Box::new(CudaBackend);
        backend.untrusted_advice_cycle = Box::new(CudaBackend);
        backend.bytecode_reduction_cycle = Box::new(CudaBackend);
        backend.program_image_reduction_cycle = Box::new(CudaBackend);
        backend.trusted_advice_address = Box::new(CudaBackend);
        backend.untrusted_advice_address = Box::new(CudaBackend);
        backend.bytecode_reduction_address = Box::new(CudaBackend);
        backend.program_image_reduction_address = Box::new(CudaBackend);
        backend.commit = Box::new(CudaBackend);
        backend.advice_opening = Box::new(CudaBackend);
        backend.joint_opening = Box::new(CudaBackend);
        backend
    }
}

pub fn warm_shared_witness<F: jolt_field::Field>(
    session: &mut crate::ProofSession,
    witness: &dyn jolt_witness::JoltWitnessPlane<F>,
    log_t: usize,
) -> Result<(), crate::KernelError<F>> {
    let cycles = 1usize << log_t;
    let _ = require_context::<F>()?;
    for (ordinal, window) in self::common::devices::witness_windows(cycles)
        .iter()
        .enumerate()
    {
        let _device = common::context::enter_device(ordinal);
        let device = common::context::context_for(ordinal).ok_or(
            crate::KernelError::InvariantViolation {
                reason: "a witness warm-up window names an absent device",
            },
        )?;
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
    use jolt_field::{Fr, FromPrimitiveInt};

    use super::{as_fr_slice, fr_into, fr_vec_into, shared_context};

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
        if cudarc::driver::CudaContext::new(0).is_err() {
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
    fn field_reinterprets_round_trip_for_fr() {
        let values = sample(4);
        assert_eq!(as_fr_slice::<Fr>(&values).expect("Fr slice"), &values[..]);
        assert_eq!(fr_vec_into::<Fr>(values.clone()).expect("Fr vec"), values,);
        assert_eq!(fr_into::<Fr>(values[0]).expect("Fr scalar"), values[0]);
    }
}
