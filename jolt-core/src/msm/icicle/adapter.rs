use crate::field::JoltField;
use crate::msm::{GpuBaseType, VariableBaseMSM};
use ark_bn254::G1Projective;
use ark_ec::{CurveGroup, ScalarMul};
use ark_ff::{BigInteger, Field, PrimeField};
use icicle_bn254::curve::CurveCfg as IcicleBn254;
use icicle_core::curve::{Affine, Curve, Projective};
use icicle_core::{
    msm::{msm, MSMConfig, MSM},
    traits::FieldImpl,
};
use icicle_runtime::memory::HostOrDeviceSlice;
use icicle_runtime::stream::IcicleStreamHandle;
use icicle_runtime::{
    memory::{DeviceVec, HostSlice},
    stream::IcicleStream,
};
use rayon::prelude::*;
use std::os::raw::c_void;

impl Icicle for G1Projective {
    type C = IcicleBn254;

    fn to_ark_projective(point: &Projective<Self::C>) -> Self {
        let proj_x =
            <Self as CurveGroup>::BaseField::from_random_bytes(&point.x.to_bytes_le()).unwrap();
        let proj_y =
            <Self as CurveGroup>::BaseField::from_random_bytes(&point.y.to_bytes_le()).unwrap();
        let proj_z =
            <Self as CurveGroup>::BaseField::from_random_bytes(&point.z.to_bytes_le()).unwrap();

        let proj_x = proj_x * proj_z;
        let proj_y = proj_y * proj_z * proj_z;
        Self::new_unchecked(proj_x, proj_y, proj_z)
    }

    fn from_ark_affine(point: &Self::MulBase) -> Affine<Self::C> {
        let x_bytes: Vec<u8> = point
            .x
            .to_base_prime_field_elements()
            .flat_map(|x| x.into_bigint().to_bytes_le())
            .collect();
        let y_bytes: Vec<u8> = point
            .y
            .to_base_prime_field_elements()
            .flat_map(|x| x.into_bigint().to_bytes_le())
            .collect();
        let x = <Self::C as Curve>::BaseField::from_bytes_le(&x_bytes);
        let y = <Self::C as Curve>::BaseField::from_bytes_le(&y_bytes);
        Affine::<Self::C> { x, y }
    }
}

pub trait Icicle: ScalarMul {
    type C: Curve<ScalarField: > + MSM<Self::C>;

    // Note: To prevent excessive trait the arkworks conversion functions within icicle are reimplemented
    fn to_ark_projective(point: &Projective<Self::C>) -> Self;

    fn from_ark_affine(point: &Self::MulBase) -> Affine<Self::C>;
}

#[tracing::instrument(skip_all, name = "icicle_msm")]
pub fn icicle_msm<V>(bases: &[GpuBaseType<V>], scalars: &[V::ScalarField], max_num_bits: usize) -> V
where
    V: VariableBaseMSM,
    V::ScalarField: JoltField,
{
    assert!(scalars.len() <= bases.len());

    let mut bases_slice = DeviceVec::<GpuBaseType<V>>::device_malloc(bases.len()).unwrap();

    let span = tracing::span!(tracing::Level::INFO, "convert_scalars");
    let _guard = span.enter();

    let mut scalars_slice =
        DeviceVec::<<<V as Icicle>::C as Curve>::ScalarField>::device_malloc(scalars.len())
            .unwrap();
    let scalars_mont =
        unsafe { &*(scalars as *const _ as *const [<<V as Icicle>::C as Curve>::ScalarField]) };

    drop(_guard);
    drop(span);

    let mut stream = IcicleStream::create().unwrap();

    let span = tracing::span!(tracing::Level::INFO, "copy_to_gpu");
    let _guard = span.enter();
    bases_slice
        .copy_from_host_async(HostSlice::from_slice(bases), &stream)
        .unwrap();
    scalars_slice
        .copy_from_host_async(HostSlice::from_slice(scalars_mont), &stream)
        .unwrap();
    drop(_guard);
    drop(span);

    let mut msm_result = DeviceVec::<Projective<V::C>>::device_malloc(1).unwrap();
    let mut cfg = MSMConfig::default();
    cfg.stream_handle = IcicleStreamHandle::from(&stream);
    cfg.is_async = false;
    cfg.are_scalars_montgomery_form = true;
    cfg.bitsize = max_num_bits as i32;

    let span = tracing::span!(tracing::Level::INFO, "gpu_msm");
    let _guard = span.enter();

    msm(
        &scalars_slice,
        &bases_slice[..scalars.len()],
        &cfg,
        &mut msm_result,
    )
    .unwrap();

    drop(_guard);
    drop(span);

    let mut msm_host_result = [Projective::<V::C>::zero(); 1];

    let span = tracing::span!(tracing::Level::INFO, "copy_msm_result");
    let _guard = span.enter();
    msm_result
        .copy_to_host(HostSlice::from_mut_slice(&mut msm_host_result))
        .unwrap();
    drop(_guard);
    drop(span);

    stream.synchronize().unwrap();
    stream.destroy().unwrap();
    V::to_ark_projective(&msm_host_result[0])
}

/// Batch process msms - assumes batches are equal in size
/// Variable Batch sizes is not currently supported by icicle
#[tracing::instrument(skip_all)]
pub fn icicle_batch_msm<V>(
    bases: &[GpuBaseType<V>],
    scalar_batches: &[&[V::ScalarField]],
    max_num_bits: usize,
) -> Vec<V>
where
    V: VariableBaseMSM,
    V::ScalarField: JoltField,
{
    let bases_len = bases.len();
    let batch_size = scalar_batches.len();
    assert!(scalar_batches.par_iter().all(|s| s.len() == bases_len));

    let mut stream = IcicleStream::create().unwrap();
    icicle_runtime::warmup(&stream).unwrap();

    let mut bases_slice =
        DeviceVec::<GpuBaseType<V>>::device_malloc_async(bases_len, &stream).unwrap();
    let span = tracing::span!(tracing::Level::INFO, "copy_bases_to_gpu");
    let _guard = span.enter();
    bases_slice
        .copy_from_host_async(HostSlice::from_slice(bases), &stream)
        .unwrap();
    drop(_guard);
    drop(span);

    let mut msm_result =
        DeviceVec::<Projective<V::C>>::device_malloc_async(batch_size, &stream).unwrap();
    let mut msm_host_results = vec![Projective::<V::C>::zero(); batch_size];
    let total_len: usize = scalar_batches.par_iter().map(|batch| batch.len()).sum();
    let mut scalars_slice =
        DeviceVec::<<<V as Icicle>::C as Curve>::ScalarField>::device_malloc_async(
            total_len, &stream,
        )
        .unwrap();

    let span = tracing::span!(tracing::Level::INFO, "copy_scalars_to_gpu");
    let _guard = span.enter();

    let mut offset = 0;
    for batch in scalar_batches {
        let scalars_mont = unsafe {
            &*(&batch[..] as *const _ as *const [<<V as Icicle>::C as Curve>::ScalarField])
        };
        copy_offset_from_host_async(
            &mut scalars_slice,
            HostSlice::from_slice(scalars_mont),
            offset,
            &stream,
        )
        .unwrap();
        offset += batch.len();
    }

    drop(_guard);
    drop(span);

    //TODO(sagar) why doesn't the GPU always go to 100% clock speeds
    let mut cfg = MSMConfig::default();
    cfg.stream_handle = IcicleStreamHandle::from(&stream);
    cfg.is_async = true;
    cfg.are_scalars_montgomery_form = true;
    cfg.batch_size = batch_size as i32;
    cfg.bitsize = max_num_bits as i32;
    cfg.ext
        .set_int(icicle_core::msm::CUDA_MSM_LARGE_BUCKET_FACTOR, 5);

    let span = tracing::span!(tracing::Level::INFO, "msm_batch_gpu");
    let _guard = span.enter();
    msm(&scalars_slice, &bases_slice, &cfg, &mut msm_result).unwrap();
    drop(_guard);
    drop(span);

    let span = tracing::span!(tracing::Level::INFO, "synchronize");
    let _guard = span.enter();
    stream.synchronize().unwrap();
    drop(_guard);
    drop(span);

    let span = tracing::span!(tracing::Level::INFO, "copy_msm_result");
    let _guard = span.enter();
    msm_result
        .copy_to_host(HostSlice::from_mut_slice(&mut msm_host_results))
        .unwrap();
    drop(_guard);
    drop(span);

    let span = tracing::span!(tracing::Level::INFO, "converting_results");
    let _guard = span.enter();
    stream.destroy().unwrap();
    msm_host_results
        .into_par_iter()
        .map(|res| V::to_ark_projective(&res))
        .collect()
}

pub fn copy_offset_from_host_async<T>(
    dest: &mut DeviceVec<T>,
    src: &HostSlice<T>,
    offset: usize,
    stream: &IcicleStream,
) -> Result<(), icicle_runtime::errors::eIcicleError> {
    if dest.is_empty() {
        return Ok(());
    }

    if !dest.is_on_active_device() {
        panic!("not allocated on an active device");
    }

    if (src.len() + offset) > dest.len() {
        panic!(
            "offset {} + HostSlice.len() {} exceeds the size of the destination DeviceVec {}",
            offset,
            src.len(),
            dest.len()
        );
    }

    let size = size_of::<T>() * src.len();
    unsafe {
        icicle_runtime::icicle_copy_to_device_async(
            dest.as_mut_ptr().add(offset) as *mut c_void,
            src.as_ptr() as *const c_void,
            size,
            stream.handle,
        )
        .wrap()
    }
}

pub fn icicle_from_ark<T, I>(ark: &T) -> I
where
    T: PrimeField,
    I: FieldImpl,
{
    let mut ark_bytes =
        Vec::with_capacity(T::BigInt::NUM_LIMBS * 8 * T::extension_degree() as usize);
    for base_elem in ark.to_base_prime_field_elements() {
        ark_bytes.extend_from_slice(&base_elem.into_bigint().to_bytes_le());
    }
    I::from_bytes_le(&ark_bytes)
}

pub fn icicle_to_ark<T, I>(icicle: &I) -> T
where
    T: PrimeField,
    I: FieldImpl,
{
    T::from_random_bytes(&icicle.to_bytes_le()).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::msm::total_memory_bits;
    use ark_bn254::{Fr, G1Affine, G1Projective};
    use ark_ec::VariableBaseMSM as ark_VariableBaseMSM;
    use ark_std::UniformRand;
    use icicle_bn254::curve::ScalarField as GPUScalar;
    use rand_core::SeedableRng;

    #[test]
    fn test_icicle_msm_consistency() {
        let pow = 10;
        let n = 1 << pow;
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(n as u64);
        for _ in 0..10 {
            let scalars: Vec<Fr> = std::iter::repeat_with(|| Fr::rand(&mut rng))
                .take(n)
                .collect();
            let bases: Vec<G1Affine> = std::iter::repeat_with(|| G1Affine::rand(&mut rng))
                .take(n)
                .collect();

            let gpu_bases = bases
                .par_iter()
                .map(|base| <G1Projective as Icicle>::from_ark_affine(base))
                .collect::<Vec<_>>();
            let icicle_res = icicle_msm::<G1Projective>(&gpu_bases, &scalars, 256);
            let arkworks_res: G1Projective = ark_VariableBaseMSM::msm(&bases, &scalars).unwrap();
            let no_gpu_res: G1Projective =
                VariableBaseMSM::msm_field_elements(&bases, None, &scalars, None, false).unwrap();

            assert_eq!(icicle_res, arkworks_res);
            assert_eq!(icicle_res, no_gpu_res);
        }
    }

    #[test]
    fn test_icicle_batch_msm_consistency() {
        let pow = 10;
        let n = 1 << pow;
        let mut rng = rand_chacha::ChaCha20Rng::seed_from_u64(n as u64);
        for _ in 0..10 {
            let scalars: Vec<Fr> = std::iter::repeat_with(|| Fr::rand(&mut rng))
                .take(n)
                .collect();
            let scalar_batches = [scalars.as_slice(); 20];

            let bases: Vec<G1Affine> = std::iter::repeat_with(|| G1Affine::rand(&mut rng))
                .take(n)
                .collect();

            let gpu_bases = bases
                .par_iter()
                .map(|base| <G1Projective as Icicle>::from_ark_affine(base))
                .collect::<Vec<_>>();
            let icicle_res = icicle_batch_msm::<G1Projective>(&gpu_bases, &scalar_batches, 256);
            let arkworks_res: Vec<G1Projective> = (0..20)
                .into_iter()
                .map(|_| ark_VariableBaseMSM::msm(&bases, &scalars).unwrap())
                .collect();
            let no_gpu_res: Vec<G1Projective> = (0..20)
                .into_iter()
                .map(|_| {
                    VariableBaseMSM::msm_field_elements(&bases, None, &scalars, None, false)
                        .unwrap()
                })
                .collect();

            assert_eq!(icicle_res, arkworks_res);
            assert_eq!(icicle_res, no_gpu_res);
        }
    }

    #[test]
    fn test_casting() {
        let ark = Fr::from(100);
        let gpu: GPUScalar = icicle_from_ark(&ark);

        let ark_bytes: [u8; 32] = unsafe { std::mem::transmute(ark) };
        let gpu_bytes: [u8; 32] =
            unsafe { std::mem::transmute(icicle_to_ark::<Fr, GPUScalar>(&gpu)) };
        assert_eq!(ark_bytes, gpu_bytes);
    }

    #[test]
    fn test_total_memory() {
        let total = total_memory_bits();
        assert!(total > 0);
    }
}
