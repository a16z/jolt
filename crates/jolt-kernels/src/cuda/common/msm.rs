//! BN254 base-field and G1 arithmetic on the device.
//!
//! Inputs and outputs use four Montgomery limbs per field element. Each GPU
//! thread inverts a chunk using its output range for prefix products; zero
//! entries remain zero.
//!
//! G1 doubling and addition keep Jacobian coordinates; mixed addition consumes
//! affine addends with all-zero device coordinates representing infinity.
//! Batched affine-pair addition instead inverts x-coordinate differences.
//! Callers must supply finite points with distinct x-coordinates in each pair;
//! this precondition is not checked by the wrapper or kernels.

use cudarc::driver::{CudaFunction, CudaSlice, PushKernelArg};

use super::context::CudaKernelContext;
use super::error::CudaError;

pub const FQ_LIMBS: usize = 4;

pub const AFFINE_LIMBS: usize = 2 * FQ_LIMBS;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct AffineLimbs {
    pub x: [u64; FQ_LIMBS],
    pub y: [u64; FQ_LIMBS],
    pub infinity: bool,
}

impl AffineLimbs {
    pub const IDENTITY: Self = Self {
        x: [0; FQ_LIMBS],
        y: [0; FQ_LIMBS],
        infinity: true,
    };
}

#[derive(Clone, Copy, Debug)]
pub struct JacobianLimbs {
    pub x: [u64; FQ_LIMBS],
    pub y: [u64; FQ_LIMBS],
    pub z: [u64; FQ_LIMBS],
}

impl JacobianLimbs {
    pub const IDENTITY: Self = Self {
        x: [0; FQ_LIMBS],
        y: [0; FQ_LIMBS],
        z: [0; FQ_LIMBS],
    };

    pub const fn is_identity(&self) -> bool {
        self.z[0] == 0 && self.z[1] == 0 && self.z[2] == 0 && self.z[3] == 0
    }
}

const BATCH_INVERSE_TARGET_THREADS: usize = 8192;

const BATCH_INVERSE_MAX_CHUNK: usize = 128;

fn flatten_affine(points: &[AffineLimbs]) -> Vec<u64> {
    let mut flat = Vec::with_capacity(points.len() * AFFINE_LIMBS);
    for point in points {
        if point.infinity {
            flat.extend_from_slice(&[0u64; AFFINE_LIMBS]);
        } else {
            flat.extend_from_slice(&point.x);
            flat.extend_from_slice(&point.y);
        }
    }
    flat
}

fn flatten_jacobian(points: &[JacobianLimbs]) -> Vec<u64> {
    let mut flat = Vec::with_capacity(points.len() * 3 * FQ_LIMBS);
    for point in points {
        flat.extend_from_slice(&point.x);
        flat.extend_from_slice(&point.y);
        flat.extend_from_slice(&point.z);
    }
    flat
}

fn take_limbs(chunk: &[u64]) -> [u64; FQ_LIMBS] {
    let mut limbs = [0u64; FQ_LIMBS];
    limbs.copy_from_slice(chunk);
    limbs
}

fn unflatten_jacobian(flat: &[u64]) -> Vec<JacobianLimbs> {
    flat.chunks_exact(3 * FQ_LIMBS)
        .map(|point| JacobianLimbs {
            x: take_limbs(&point[..FQ_LIMBS]),
            y: take_limbs(&point[FQ_LIMBS..2 * FQ_LIMBS]),
            z: take_limbs(&point[2 * FQ_LIMBS..]),
        })
        .collect()
}

fn unflatten_affine(flat: &[u64]) -> Vec<AffineLimbs> {
    flat.chunks_exact(AFFINE_LIMBS)
        .map(|point| {
            let x = take_limbs(&point[..FQ_LIMBS]);
            let y = take_limbs(&point[FQ_LIMBS..]);
            let infinity = x == [0; FQ_LIMBS] && y == [0; FQ_LIMBS];
            AffineLimbs { x, y, infinity }
        })
        .collect()
}

impl CudaKernelContext {
    fn fq_binary(
        &self,
        kernel: &CudaFunction,
        left: &[[u64; FQ_LIMBS]],
        right: &[[u64; FQ_LIMBS]],
    ) -> Result<Vec<[u64; FQ_LIMBS]>, CudaError> {
        if left.len() != right.len() {
            return Err(CudaError::LengthMismatch {
                expected: left.len(),
                got: right.len(),
            });
        }
        if left.is_empty() {
            return Ok(Vec::new());
        }
        let count = Self::count_of(left.len())?;
        let device_left = self.upload_u64_slice(left.as_flattened())?;
        let device_right = self.upload_u64_slice(right.as_flattened())?;
        let mut output = self.alloc_u64(left.len() * FQ_LIMBS)?;
        let mut builder = self.stream().launch_builder(kernel);
        let _ = builder.arg(&device_left);
        let _ = builder.arg(&device_right);
        let _ = builder.arg(&mut output);
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` reads only `left[i]`/`right[i]` and writes
        // only `out[i]` (4 limbs each); all three buffers hold
        // `count * FQ_LIMBS` u64s, the length equality is checked above, and
        // `out` is a fresh allocation distinct from both inputs. Threads with
        // `i >= count` return before any access.
        let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
        self.stream().synchronize()?;
        Ok(self
            .download_u64(&output)?
            .chunks_exact(FQ_LIMBS)
            .map(take_limbs)
            .collect())
    }

    fn g1_binary(
        &self,
        kernel: &CudaFunction,
        left: &[JacobianLimbs],
        right_flat: &[u64],
        right_stride: usize,
    ) -> Result<Vec<JacobianLimbs>, CudaError> {
        if left.len() * right_stride != right_flat.len() {
            return Err(CudaError::LengthMismatch {
                expected: left.len() * right_stride,
                got: right_flat.len(),
            });
        }
        if left.is_empty() {
            return Ok(Vec::new());
        }
        let count = Self::count_of(left.len())?;
        let device_left = self.upload_u64_slice(&flatten_jacobian(left))?;
        let device_right = self.upload_u64_slice(right_flat)?;
        let mut output = self.alloc_u64(left.len() * 3 * FQ_LIMBS)?;
        let mut builder = self.stream().launch_builder(kernel);
        let _ = builder.arg(&device_left);
        let _ = builder.arg(&device_right);
        let _ = builder.arg(&mut output);
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` reads only its own point at `left[i]`
        // (12 limbs) and `right[i]` (`right_stride` limbs), and writes only
        // `out[i]` (12 limbs). The buffers hold `count * 12`,
        // `count * right_stride` and `count * 12` u64s respectively — the
        // right-hand length is checked above — and `out` is a fresh allocation
        // distinct from both inputs. Threads with `i >= count` return first.
        let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
        self.stream().synchronize()?;
        Ok(unflatten_jacobian(&self.download_u64(&output)?))
    }

    pub fn fq_add(
        &self,
        left: &[[u64; FQ_LIMBS]],
        right: &[[u64; FQ_LIMBS]],
    ) -> Result<Vec<[u64; FQ_LIMBS]>, CudaError> {
        self.fq_binary(self.msm_fq_add(), left, right)
    }

    pub fn fq_sub(
        &self,
        left: &[[u64; FQ_LIMBS]],
        right: &[[u64; FQ_LIMBS]],
    ) -> Result<Vec<[u64; FQ_LIMBS]>, CudaError> {
        self.fq_binary(self.msm_fq_sub(), left, right)
    }

    pub fn fq_mul(
        &self,
        left: &[[u64; FQ_LIMBS]],
        right: &[[u64; FQ_LIMBS]],
    ) -> Result<Vec<[u64; FQ_LIMBS]>, CudaError> {
        self.fq_binary(self.msm_fq_mul(), left, right)
    }

    pub fn fq_batch_inverse(
        &self,
        values: &[[u64; FQ_LIMBS]],
    ) -> Result<Vec<[u64; FQ_LIMBS]>, CudaError> {
        if values.is_empty() {
            return Ok(Vec::new());
        }
        let device = self.upload_u64_slice(values.as_flattened())?;
        let mut output = self.alloc_u64(values.len() * FQ_LIMBS)?;
        self.launch_batch_inverse(&device, &mut output, values.len())?;
        Ok(self
            .download_u64(&output)?
            .chunks_exact(FQ_LIMBS)
            .map(take_limbs)
            .collect())
    }

    pub(crate) fn launch_batch_inverse(
        &self,
        values: &CudaSlice<u64>,
        output: &mut CudaSlice<u64>,
        len: usize,
    ) -> Result<(), CudaError> {
        if len == 0 {
            return Ok(());
        }
        let chunk = len
            .div_ceil(BATCH_INVERSE_TARGET_THREADS)
            .clamp(1, BATCH_INVERSE_MAX_CHUNK);
        let threads = Self::count_of(len.div_ceil(chunk))?;
        let count = Self::count_of(len)?;
        let chunk = Self::count_of(chunk)?;
        let mut builder = self.stream().launch_builder(self.msm_fq_batch_inverse());
        let _ = builder.arg(values);
        let _ = builder.arg(output);
        let _ = builder.arg(&count);
        let _ = builder.arg(&chunk);
        // SAFETY: thread `t` owns the half-open element range
        // `[t * chunk, min((t+1) * chunk, count))`, disjoint across threads and
        // clipped to `count`. It reads only `values` and read-modify-writes
        // only `out` inside that range (4 limbs per element; both buffers hold
        // `count * FQ_LIMBS` u64s), using `out` as its own prefix-product
        // scratch. `out` is a distinct allocation from `values`. Threads whose
        // range starts at or past `count` return before any access.
        let _ = unsafe { builder.launch(Self::launch_config(threads)) }?;
        self.stream().synchronize()?;
        Ok(())
    }

    pub fn g1_double(&self, points: &[JacobianLimbs]) -> Result<Vec<JacobianLimbs>, CudaError> {
        if points.is_empty() {
            return Ok(Vec::new());
        }
        let count = Self::count_of(points.len())?;
        let device = self.upload_u64_slice(&flatten_jacobian(points))?;
        let mut output = self.alloc_u64(points.len() * 3 * FQ_LIMBS)?;
        let mut builder = self.stream().launch_builder(self.msm_g1_double());
        let _ = builder.arg(&device);
        let _ = builder.arg(&mut output);
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` reads only `points[i]` and writes only
        // `out[i]` (12 limbs each); both buffers hold `count * 12` u64s and
        // `out` is a fresh allocation distinct from the input. Threads with
        // `i >= count` return before any access.
        let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
        self.stream().synchronize()?;
        Ok(unflatten_jacobian(&self.download_u64(&output)?))
    }

    pub fn g1_add(
        &self,
        left: &[JacobianLimbs],
        right: &[JacobianLimbs],
    ) -> Result<Vec<JacobianLimbs>, CudaError> {
        self.g1_binary(
            self.msm_g1_add(),
            left,
            &flatten_jacobian(right),
            3 * FQ_LIMBS,
        )
    }

    pub fn g1_add_affine(
        &self,
        left: &[JacobianLimbs],
        right: &[AffineLimbs],
    ) -> Result<Vec<JacobianLimbs>, CudaError> {
        self.g1_binary(
            self.msm_g1_add_affine(),
            left,
            &flatten_affine(right),
            AFFINE_LIMBS,
        )
    }

    pub fn g1_batch_add_affine_pairs(
        &self,
        left: &[AffineLimbs],
        right: &[AffineLimbs],
    ) -> Result<Vec<AffineLimbs>, CudaError> {
        if left.len() != right.len() {
            return Err(CudaError::LengthMismatch {
                expected: left.len(),
                got: right.len(),
            });
        }
        if left.is_empty() {
            return Ok(Vec::new());
        }
        let count = Self::count_of(left.len())?;
        let device_left = self.upload_u64_slice(&flatten_affine(left))?;
        let device_right = self.upload_u64_slice(&flatten_affine(right))?;
        let mut denominators = self.alloc_u64(left.len() * FQ_LIMBS)?;
        let mut builder = self.stream().launch_builder(self.msm_affine_denominators());
        let _ = builder.arg(&device_left);
        let _ = builder.arg(&device_right);
        let _ = builder.arg(&mut denominators);
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` reads the x-coordinate of `left[i]` and
        // `right[i]` (limb group `2i` of buffers holding
        // `count * AFFINE_LIMBS` u64s) and writes only `denominators[i]` of a
        // `count * FQ_LIMBS` fresh allocation distinct from both inputs.
        // Threads with `i >= count` return before any access.
        let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
        self.stream().synchronize()?;

        let mut inverses = self.alloc_u64(left.len() * FQ_LIMBS)?;
        self.launch_batch_inverse(&denominators, &mut inverses, left.len())?;

        let mut output = self.alloc_u64(left.len() * AFFINE_LIMBS)?;
        let mut builder = self.stream().launch_builder(self.msm_affine_combine());
        let _ = builder.arg(&device_left);
        let _ = builder.arg(&device_right);
        let _ = builder.arg(&inverses);
        let _ = builder.arg(&mut output);
        let _ = builder.arg(&count);
        // SAFETY: thread `i < count` reads `left[i]`, `right[i]`
        // (`AFFINE_LIMBS` u64s each) and `inverses[i]` (`FQ_LIMBS` u64s), and
        // writes only `out[i]` of a `count * AFFINE_LIMBS` fresh allocation
        // distinct from all three inputs. Threads with `i >= count` return
        // before any access.
        let _ = unsafe { builder.launch(Self::launch_config(count)) }?;
        self.stream().synchronize()?;
        Ok(unflatten_affine(&self.download_u64(&output)?))
    }
}

#[cfg(test)]
pub(crate) mod testing {
    use ark_bn254::{Fq, Fr as ArkFr, G1Affine, G1Projective};
    use ark_ec::{CurveGroup, PrimeGroup};
    use ark_ff::{BigInt, PrimeField};
    use proptest::prelude::*;

    use super::{AffineLimbs, JacobianLimbs, FQ_LIMBS};

    pub fn fq_from_limbs(limbs: [u64; FQ_LIMBS]) -> Fq {
        Fq::new_unchecked(BigInt(limbs))
    }

    pub fn fq_limbs(value: Fq) -> [u64; FQ_LIMBS] {
        value.0 .0
    }

    pub fn affine_limbs(point: G1Affine) -> AffineLimbs {
        if point.infinity {
            return AffineLimbs::IDENTITY;
        }
        AffineLimbs {
            x: fq_limbs(point.x),
            y: fq_limbs(point.y),
            infinity: false,
        }
    }

    pub fn jacobian_limbs(point: G1Projective) -> JacobianLimbs {
        JacobianLimbs {
            x: fq_limbs(point.x),
            y: fq_limbs(point.y),
            z: fq_limbs(point.z),
        }
    }

    pub fn projective(point: JacobianLimbs) -> G1Projective {
        G1Projective::new_unchecked(
            fq_from_limbs(point.x),
            fq_from_limbs(point.y),
            fq_from_limbs(point.z),
        )
    }

    pub fn projectives(points: &[JacobianLimbs]) -> Vec<G1Projective> {
        points.iter().copied().map(projective).collect()
    }

    pub fn point(k: u64) -> G1Projective {
        G1Projective::generator() * ArkFr::from(k)
    }

    pub fn affine(k: u64) -> G1Affine {
        point(k).into_affine()
    }

    pub fn arb_fq() -> impl Strategy<Value = Fq> {
        any::<[u64; FQ_LIMBS]>().prop_map(|limbs| {
            let mut bytes = [0u8; 32];
            for (chunk, limb) in bytes.chunks_exact_mut(8).zip(limbs) {
                chunk.copy_from_slice(&limb.to_le_bytes());
            }
            Fq::from_le_bytes_mod_order(&bytes)
        })
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use ark_bn254::{Fq, G1Affine, G1Projective};
    use ark_ec::{AdditiveGroup, AffineRepr, CurveGroup};
    use ark_ff::{PrimeField, Zero};
    use proptest::collection::vec;
    use proptest::prelude::*;

    use super::testing::{
        affine, affine_limbs, arb_fq, fq_from_limbs, fq_limbs, jacobian_limbs, point, projective,
        projectives,
    };
    use super::{AffineLimbs, JacobianLimbs, FQ_LIMBS};
    use crate::cuda::common::context::{shared_context, CudaKernelContext};

    fn device() -> Option<&'static CudaKernelContext> {
        shared_context()
    }

    fn limbs(values: &[Fq]) -> Vec<[u64; FQ_LIMBS]> {
        values.iter().copied().map(fq_limbs).collect()
    }

    fn affine_pairs(count: usize, offset: u64) -> (Vec<G1Affine>, Vec<G1Affine>) {
        let left = (0..count)
            .map(|i| affine(offset + 2 * i as u64 + 1))
            .collect();
        let right = (0..count)
            .map(|i| affine(offset + 2 * i as u64 + 2))
            .collect();
        (left, right)
    }

    fn special_pairs() -> Vec<(G1Projective, G1Projective)> {
        let p = point(7);
        let q = point(11);
        vec![
            (p, q),
            (p, p),
            (p, -p),
            (p, G1Projective::zero()),
            (G1Projective::zero(), p),
            (G1Projective::zero(), G1Projective::zero()),
            (q, -q),
            (point(1), point(1)),
        ]
    }

    #[test]
    fn special_pairs_cover_every_addition_branch() {
        let pairs = special_pairs();
        assert!(
            pairs.iter().any(|&(a, b)| a == b && !a.is_zero()),
            "no pair has equal nonzero inputs, so a formula missing the doubling branch \
             would pass",
        );
        assert!(
            pairs
                .iter()
                .any(|&(a, b)| !a.is_zero() && (a + b).is_zero()),
            "no pair sums to infinity, so a formula that cannot produce the identity would pass",
        );
        assert!(
            pairs.iter().any(|&(a, _)| a.is_zero()),
            "no pair has the accumulator at infinity",
        );
        assert!(
            pairs.iter().any(|&(_, b)| b.is_zero()),
            "no pair has the addend at infinity",
        );
        assert!(
            pairs
                .iter()
                .any(|&(a, b)| !a.is_zero() && !b.is_zero() && a != b && !(a + b).is_zero()),
            "no pair is the ordinary generic case",
        );
    }

    #[test]
    fn batch_affine_pairs_respect_the_distinct_x_precondition() {
        let (left, right) = affine_pairs(16, 5);
        assert!(
            left.iter().zip(&right).all(|(a, b)| a.x != b.x),
            "a pair shares an x-coordinate, which makes the batch-inversion denominator zero \
             and yields a silently off-curve point",
        );
        assert!(
            left.iter().chain(&right).all(|point| !point.infinity),
            "an input is at infinity, which affine pair addition cannot represent",
        );
        assert!(
            special_pairs()
                .iter()
                .any(|&(a, b)| !a.is_zero() && !b.is_zero() && a == b),
            "the cases affine pair addition excludes (equal, inverse, infinity) must be covered \
             by the general Jacobian addition instead, and they are not",
        );
    }

    #[test]
    fn identity_limbs_read_back_as_the_group_identity() {
        assert!(JacobianLimbs::IDENTITY.is_identity());
        assert_eq!(projective(JacobianLimbs::IDENTITY), G1Projective::zero());
        assert_eq!(
            affine_limbs(<G1Affine as AffineRepr>::zero()),
            AffineLimbs::IDENTITY,
        );
    }

    #[test]
    fn g1_add_matches_arkworks_at_special_cases() {
        let Some(context) = device() else {
            return;
        };
        let pairs = special_pairs();
        let left: Vec<JacobianLimbs> = pairs.iter().map(|&(a, _)| jacobian_limbs(a)).collect();
        let right: Vec<JacobianLimbs> = pairs.iter().map(|&(_, b)| jacobian_limbs(b)).collect();
        let expected: Vec<G1Projective> = pairs.iter().map(|&(a, b)| a + b).collect();
        let got = projectives(&context.g1_add(&left, &right).expect("device g1_add"));
        assert_eq!(got, expected);

        let right_affine: Vec<AffineLimbs> = pairs
            .iter()
            .map(|&(_, b)| affine_limbs(b.into_affine()))
            .collect();
        let got_affine = projectives(
            &context
                .g1_add_affine(&left, &right_affine)
                .expect("device g1_add_affine"),
        );
        assert_eq!(got_affine, expected);
    }

    #[test]
    fn fq_arithmetic_carries_across_limb_and_modulus_boundaries() {
        let Some(context) = device() else {
            return;
        };
        let mut values = vec![Fq::zero(), fq_from_limbs([1, 0, 0, 0])];
        let mut near_modulus = Fq::MODULUS.0;
        near_modulus[0] -= 1;
        values.push(fq_from_limbs(near_modulus));
        for end in 1..FQ_LIMBS {
            let mut below = [0; FQ_LIMBS];
            below[..end].fill(u64::MAX);
            let mut above = [0; FQ_LIMBS];
            above[end] = 1;
            values.extend([fq_from_limbs(below), fq_from_limbs(above)]);
        }
        let pairs: Vec<_> = values
            .iter()
            .flat_map(|&a| values.iter().map(move |&b| (a, b)))
            .collect();
        let left: Vec<_> = pairs.iter().map(|&(a, _)| fq_limbs(a)).collect();
        let right: Vec<_> = pairs.iter().map(|&(_, b)| fq_limbs(b)).collect();
        for (actual, expected) in [
            (
                context.fq_add(&left, &right).expect("add"),
                pairs
                    .iter()
                    .map(|&(a, b)| fq_limbs(a + b))
                    .collect::<Vec<_>>(),
            ),
            (
                context.fq_sub(&left, &right).expect("sub"),
                pairs
                    .iter()
                    .map(|&(a, b)| fq_limbs(a - b))
                    .collect::<Vec<_>>(),
            ),
            (
                context.fq_mul(&left, &right).expect("mul"),
                pairs
                    .iter()
                    .map(|&(a, b)| fq_limbs(a * b))
                    .collect::<Vec<_>>(),
            ),
        ] {
            assert_eq!(actual, expected);
        }
    }

    proptest! {
        #[test]
        fn fq_add_matches_arkworks(left in vec(arb_fq(), 1..200)) {
            let Some(context) = device() else { return Ok(()); };
            let right: Vec<_> = left.iter().rev().copied().collect();
            let expected: Vec<_> = left.iter().zip(&right).map(|(&a, &b)| a + b).collect();
            let got: Vec<_> = context
                .fq_add(&limbs(&left), &limbs(&right))
                .expect("device fq_add")
                .into_iter()
                .map(fq_from_limbs)
                .collect();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn fq_sub_matches_arkworks(left in vec(arb_fq(), 1..200)) {
            let Some(context) = device() else { return Ok(()); };
            let right: Vec<_> = left.iter().rev().copied().collect();
            let expected: Vec<_> = left.iter().zip(&right).map(|(&a, &b)| a - b).collect();
            let got: Vec<_> = context
                .fq_sub(&limbs(&left), &limbs(&right))
                .expect("device fq_sub")
                .into_iter()
                .map(fq_from_limbs)
                .collect();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn fq_batch_inverse_matches_arkworks(values in vec(arb_fq(), 1..200)) {
            let Some(context) = device() else { return Ok(()); };
            let mut values = values;
            for (index, value) in values.iter_mut().enumerate() {
                if index % 7 == 0 {
                    *value = ark_bn254::Fq::zero();
                }
            }
            prop_assert!(
                values.iter().any(|value| value.is_zero()),
                "no zero in the input, so the pass-through-zero convention is untested",
            );
            let mut expected = values.clone();
            ark_ff::fields::batch_inversion(&mut expected);
            let got: Vec<_> = context
                .fq_batch_inverse(&limbs(&values))
                .expect("device fq_batch_inverse")
                .into_iter()
                .map(fq_from_limbs)
                .collect();
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn fq_mul_matches_arkworks(left in vec(arb_fq(), 1..200)) {
            let Some(context) = device() else { return Ok(()); };
            let right: Vec<_> = left.iter().rev().copied().collect();
            let expected: Vec<_> = left.iter().zip(&right).map(|(&a, &b)| a * b).collect();
            let got: Vec<_> = context
                .fq_mul(&limbs(&left), &limbs(&right))
                .expect("device fq_mul")
                .into_iter()
                .map(fq_from_limbs)
                .collect();
            prop_assert_eq!(got, expected);
        }
    }

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(8))]

        #[test]
        fn g1_double_matches_arkworks(scalars in vec(1u64..1_000_000, 1..16)) {
            let Some(context) = device() else { return Ok(()); };
            let points: Vec<G1Projective> = scalars
                .iter()
                .copied()
                .map(point)
                .chain(std::iter::once(G1Projective::zero()))
                .collect();
            let expected: Vec<G1Projective> = points.iter().map(AdditiveGroup::double).collect();
            let inputs: Vec<JacobianLimbs> =
                points.iter().copied().map(jacobian_limbs).collect();
            let got = projectives(&context.g1_double(&inputs).expect("device g1_double"));
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn g1_add_matches_arkworks(scalars in vec(1u64..1_000_000, 2..16)) {
            let Some(context) = device() else { return Ok(()); };
            let left: Vec<G1Projective> = scalars.iter().copied().map(point).collect();
            let right: Vec<G1Projective> = scalars.iter().rev().copied().map(point).collect();
            let expected: Vec<G1Projective> =
                left.iter().zip(&right).map(|(&a, &b)| a + b).collect();
            let got = projectives(
                &context
                    .g1_add(
                        &left.iter().copied().map(jacobian_limbs).collect::<Vec<_>>(),
                        &right.iter().copied().map(jacobian_limbs).collect::<Vec<_>>(),
                    )
                    .expect("device g1_add"),
            );
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn g1_add_affine_matches_arkworks(scalars in vec(1u64..1_000_000, 2..16)) {
            let Some(context) = device() else { return Ok(()); };
            let left: Vec<G1Projective> = scalars.iter().copied().map(point).collect();
            let right: Vec<G1Affine> = scalars.iter().rev().copied().map(affine).collect();
            let expected: Vec<G1Projective> =
                left.iter().zip(&right).map(|(&a, &b)| a + b).collect();
            let got = projectives(
                &context
                    .g1_add_affine(
                        &left.iter().copied().map(jacobian_limbs).collect::<Vec<_>>(),
                        &right.iter().copied().map(affine_limbs).collect::<Vec<_>>(),
                    )
                    .expect("device g1_add_affine"),
            );
            prop_assert_eq!(got, expected);
        }

        #[test]
        fn g1_batch_add_affine_pairs_matches_arkworks(offset in 1u64..1_000) {
            let Some(context) = device() else { return Ok(()); };
            let (left, right) = affine_pairs(16, offset);
            let expected: Vec<G1Affine> = left
                .iter()
                .zip(&right)
                .map(|(&a, &b)| (G1Projective::from(a) + b).into_affine())
                .collect();
            let got: Vec<G1Affine> = context
                .g1_batch_add_affine_pairs(
                    &left.iter().copied().map(affine_limbs).collect::<Vec<_>>(),
                    &right.iter().copied().map(affine_limbs).collect::<Vec<_>>(),
                )
                .expect("device g1_batch_add_affine_pairs")
                .into_iter()
                .map(|point| {
                    G1Affine::new_unchecked(fq_from_limbs(point.x), fq_from_limbs(point.y))
                })
                .collect();
            prop_assert_eq!(got, expected);
        }
    }
}
