use jolt_field::Field;
use jolt_openings::CommitmentScheme;

use crate::commitment::ModeStreamingCommitment;
use crate::JoltBackend;

mod address_driver;
mod address_phase;
mod combine;
mod context;
mod dense_product;
mod device;
mod error;
mod instruction_read_raf;
mod lt_poly;
mod prefix_suffix;
mod prefixes;
mod primitives;
mod ra_poly;
mod ram_ra_claim_reduction;
mod ram_ra_reduction;
mod registers_val_evaluation;
mod staging;
mod suffixes;
#[cfg(test)]
mod testing;
pub mod xfer_stats;

pub use address_driver::DeviceAddressPhase;
pub use address_phase::{init_raf_buckets, init_suffix_buckets, DeviceRows};
pub use context::{shared_context, CudaKernelContext};
pub use device::{as_fr_slice, fr_into, fr_vec_into, DeviceFrVec, LIMBS};
pub use error::CudaError;
pub use lt_poly::DeviceLtPolynomial;

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
        PCS: ModeStreamingCommitment,
    {
        let mut backend = Self::reference();
        if !device_available() {
            return backend;
        }
        backend.instruction_read_raf = Box::new(CudaBackend);
        backend.registers_val_evaluation = Box::new(CudaBackend);
        backend.ram_ra_claim_reduction = Box::new(CudaBackend);
        backend
    }
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
