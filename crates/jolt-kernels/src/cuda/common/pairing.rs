use std::cell::RefCell;

use ark_bn254::{Fq, Fq2};
use ark_ec::bn::BnConfig;
use ark_ec::short_weierstrass::SWCurveConfig;
use ark_ff::{AdditiveGroup, Field, One};
use cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

use super::context::CudaKernelContext;
use super::error::CudaError;
use super::msm::FQ_LIMBS;

pub const FQ12_LIMBS: usize = 12 * FQ_LIMBS;

const PC_WORDS: usize = 28;

const PRODUCT_BLOCK: u32 = 32;

const WARP: u32 = 32;

const MILLER_WARP_WARPS: u32 = 4;

const MILLER_WARP_MAX_PAIRS: usize = 256;

struct Constants {
    values: CudaSlice<u64>,
    ate: CudaSlice<u64>,
    ate_len: usize,
}

thread_local! {
    static CONSTANTS: RefCell<Option<Constants>> = const { RefCell::new(None) };
}

fn fq_words(value: &Fq) -> [u64; FQ_LIMBS] {
    value.0 .0
}

fn push_fq2(out: &mut Vec<u64>, value: &Fq2) {
    out.extend_from_slice(&fq_words(&value.c0));
    out.extend_from_slice(&fq_words(&value.c1));
}

#[expect(
    clippy::expect_used,
    reason = "two is invertible in Fq; the alternative is a Result on an infallible constant"
)]
fn constant_words() -> Vec<u64> {
    let mut out = Vec::with_capacity(PC_WORDS);
    push_fq2(&mut out, &<ark_bn254::Config as BnConfig>::TWIST_MUL_BY_Q_X);
    push_fq2(&mut out, &<ark_bn254::Config as BnConfig>::TWIST_MUL_BY_Q_Y);
    push_fq2(&mut out, &<ark_bn254::g2::Config as SWCurveConfig>::COEFF_B);
    let two_inv = Fq::one().double().inverse().expect("two is invertible");
    out.extend_from_slice(&fq_words(&two_inv));
    out
}

fn ate_words() -> Vec<u64> {
    <ark_bn254::Config as BnConfig>::ATE_LOOP_COUNT
        .iter()
        .map(|bit| match bit {
            1 => 1,
            -1 => 2,
            _ => 0,
        })
        .collect()
}

impl CudaKernelContext {
    fn with_pairing_constants<R>(
        &self,
        act: impl FnOnce(&Constants) -> Result<R, CudaError>,
    ) -> Result<R, CudaError> {
        CONSTANTS.with_borrow_mut(|slot| {
            if slot.is_none() {
                let words = constant_words();
                if words.len() != PC_WORDS {
                    return Err(CudaError::LengthMismatch {
                        expected: PC_WORDS,
                        got: words.len(),
                    });
                }
                let ate = ate_words();
                *slot = Some(Constants {
                    values: self.upload_raw_u64(&words)?,
                    ate: self.upload_raw_u64(&ate)?,
                    ate_len: ate.len(),
                });
            }
            match slot.as_ref() {
                Some(constants) => act(constants),
                None => Err(CudaError::InvariantViolation {
                    reason: "the pairing constants did not initialise",
                }),
            }
        })
    }

    pub fn multi_miller_batch(
        &self,
        g1: &CudaSlice<u64>,
        g2: &CudaSlice<u64>,
        segments: &[(usize, usize)],
        count: usize,
    ) -> Result<Vec<u64>, CudaError> {
        if segments.len().saturating_mul(count) <= MILLER_WARP_MAX_PAIRS {
            return self.multi_miller_warp_batch(g1, g2, segments, count);
        }
        if count == 0 || segments.is_empty() {
            return Err(CudaError::InvariantViolation {
                reason: "a multi-Miller batch needs at least one segment and one pair",
            });
        }
        let mut g1_offsets = Vec::with_capacity(segments.len());
        let mut g2_offsets = Vec::with_capacity(segments.len());
        for &(g1_offset, g2_offset) in segments {
            if (g1_offset + count) * 3 * FQ_LIMBS > g1.len() {
                return Err(CudaError::LengthMismatch {
                    expected: g1.len(),
                    got: (g1_offset + count) * 3 * FQ_LIMBS,
                });
            }
            if (g2_offset + count) * 6 * FQ_LIMBS > g2.len() {
                return Err(CudaError::LengthMismatch {
                    expected: g2.len(),
                    got: (g2_offset + count) * 6 * FQ_LIMBS,
                });
            }
            g1_offsets.push(Self::count_of(g1_offset)?);
            g2_offsets.push(Self::count_of(g2_offset)?);
        }

        let lanes_len = segments
            .len()
            .checked_mul(count)
            .and_then(|pairs| pairs.checked_mul(FQ12_LIMBS))
            .ok_or(CudaError::InvariantViolation {
                reason: "a multi-Miller batch overflowed its lane buffer",
            })?;
        let mut lanes = self.alloc_u64(lanes_len)?;
        let pairs = Self::count_of(count)?;
        let lanes_of = Self::count_of(segments.len())?;
        let device_g1_offsets = self.upload_u32_slice(&g1_offsets)?;
        let device_g2_offsets = self.upload_u32_slice(&g2_offsets)?;

        self.with_pairing_constants(|constants| {
            let ate_len = Self::count_of(constants.ate_len)?;
            let mut builder = self.stream().launch_builder(self.pairing_miller());
            let _ = builder.arg(g1);
            let _ = builder.arg(g2);
            let _ = builder.arg(&constants.values);
            let _ = builder.arg(&constants.ate);
            let _ = builder.arg(&ate_len);
            let _ = builder.arg(&device_g1_offsets);
            let _ = builder.arg(&device_g2_offsets);
            let _ = builder.arg(&pairs);
            let _ = builder.arg(&mut lanes);
            // The grid is `(ceil(pairs / BLOCK), segments)`, so
            // `blockIdx.y` indexes `g1_offsets`/`g2_offsets`, both uploaded with
            // exactly `segments` entries. Thread `(segment, pair < pairs)` reads
            // the `3 * FQ_LIMBS` limbs of G1 point `g1_offsets[segment] + pair`
            // and the `6 * FQ_LIMBS` limbs of G2 point `g2_offsets[segment] +
            // pair`, every such span checked above to end inside its buffer,
            // plus the `PC_WORDS` constants and `ate_len` loop digits, whose
            // lengths are fixed at upload. It writes only the `FQ12_LIMBS` limbs
            // at `(segment * count + pair) * FQ12_LIMBS` of the freshly
            // allocated `lanes`, which holds `segments * count` such slots and
            // is distinct from every input, so no thread reads what another
            // writes. Threads with `pair >= pairs` return first.
            let mut config = Self::launch_config(pairs);
            config.grid_dim.1 = lanes_of;
            // SAFETY: as argued above.
            let _ = unsafe { builder.launch(config) }?;
            Ok(())
        })?;

        let mut product = self.alloc_u64(segments.len() * FQ12_LIMBS)?;
        let shared = PRODUCT_BLOCK * FQ12_LIMBS as u32 * size_of::<u64>() as u32;
        let mut builder = self.stream().launch_builder(self.pairing_fq12_product());
        let _ = builder.arg(&lanes);
        let _ = builder.arg(&pairs);
        let _ = builder.arg(&mut product);
        // SAFETY: one block per segment, so `blockIdx.x < segments` selects the
        // `count` Fq12 values at `blockIdx.x * count` of `lanes`, which holds
        // `segments * count` of them; each thread strides by `blockDim.x` from
        // `threadIdx.x` and so stays below `count`. Shared memory is
        // `PRODUCT_BLOCK * FQ12_LIMBS` u64s, matching `shared_mem_bytes`, and
        // `PRODUCT_BLOCK` is a power of two so the halving tree covers the
        // block; every thread reaches each `__syncthreads()` because the strided
        // loop and the tree sit outside any early return. Only thread 0 writes,
        // to slot `blockIdx.x` of the freshly allocated `product`, which is
        // distinct from `lanes`.
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (lanes_of, 1, 1),
                block_dim: (PRODUCT_BLOCK, 1, 1),
                shared_mem_bytes: shared,
            })
        }?;

        self.download_u64(&product)
    }

    pub fn multi_miller_warp_batch(
        &self,
        g1: &CudaSlice<u64>,
        g2: &CudaSlice<u64>,
        segments: &[(usize, usize)],
        count: usize,
    ) -> Result<Vec<u64>, CudaError> {
        if count == 0 || segments.is_empty() {
            return Err(CudaError::InvariantViolation {
                reason: "a multi-Miller batch needs at least one segment and one pair",
            });
        }
        let mut g1_offsets = Vec::with_capacity(segments.len());
        let mut g2_offsets = Vec::with_capacity(segments.len());
        for &(g1_offset, g2_offset) in segments {
            if (g1_offset + count) * 3 * FQ_LIMBS > g1.len() {
                return Err(CudaError::LengthMismatch {
                    expected: g1.len(),
                    got: (g1_offset + count) * 3 * FQ_LIMBS,
                });
            }
            if (g2_offset + count) * 6 * FQ_LIMBS > g2.len() {
                return Err(CudaError::LengthMismatch {
                    expected: g2.len(),
                    got: (g2_offset + count) * 6 * FQ_LIMBS,
                });
            }
            g1_offsets.push(Self::count_of(g1_offset)?);
            g2_offsets.push(Self::count_of(g2_offset)?);
        }

        let lanes_len = segments
            .len()
            .checked_mul(count)
            .and_then(|pairs| pairs.checked_mul(FQ12_LIMBS))
            .ok_or(CudaError::InvariantViolation {
                reason: "a multi-Miller batch overflowed its lane buffer",
            })?;
        let mut lanes = self.alloc_u64(lanes_len)?;
        let pairs = Self::count_of(count)?;
        let lanes_of = Self::count_of(segments.len())?;
        let device_g1_offsets = self.upload_u32_slice(&g1_offsets)?;
        let device_g2_offsets = self.upload_u32_slice(&g2_offsets)?;

        self.with_pairing_constants(|constants| {
            let ate_len = Self::count_of(constants.ate_len)?;
            let mut builder = self.stream().launch_builder(self.pairing_miller_warp());
            let _ = builder.arg(g1);
            let _ = builder.arg(g2);
            let _ = builder.arg(&constants.values);
            let _ = builder.arg(&constants.ate);
            let _ = builder.arg(&ate_len);
            let _ = builder.arg(&device_g1_offsets);
            let _ = builder.arg(&device_g2_offsets);
            let _ = builder.arg(&pairs);
            let _ = builder.arg(&mut lanes);
            // SAFETY: the block is `(WARP, MILLER_WARP_WARPS)` and the grid is
            // `(ceil(pairs / MILLER_WARP_WARPS), segments)`, so warp
            // `blockIdx.x * blockDim.y + threadIdx.y` owns one pair and
            // `blockIdx.y` indexes `g1_offsets`/`g2_offsets`, both uploaded with
            // exactly `segments` entries. That warp reads the `3 * FQ_LIMBS`
            // limbs of G1 point `g1_offsets[segment] + pair` and the
            // `6 * FQ_LIMBS` limbs of G2 point `g2_offsets[segment] + pair`,
            // every such span checked above to end inside its buffer, plus the
            // `PC_WORDS` constants and `ate_len` loop digits, whose lengths are
            // fixed at upload. It writes only the `FQ12_LIMBS` limbs at
            // `(segment * count + pair) * FQ12_LIMBS` of the freshly allocated
            // `lanes`, which holds `segments * count` such slots and is distinct
            // from every input, so no warp reads what another writes. Warps with
            // `pair >= pairs` return before any access, and they return as a
            // whole warp, so no lane of a live warp is lost to divergence at a
            // shuffle.
            let _ = unsafe {
                builder.launch(LaunchConfig {
                    grid_dim: (pairs.div_ceil(MILLER_WARP_WARPS), lanes_of, 1),
                    block_dim: (WARP, MILLER_WARP_WARPS, 1),
                    shared_mem_bytes: 0,
                })
            }?;
            Ok(())
        })?;

        let mut product = self.alloc_u64(segments.len() * FQ12_LIMBS)?;
        let shared = PRODUCT_BLOCK * FQ12_LIMBS as u32 * size_of::<u64>() as u32;
        let mut builder = self.stream().launch_builder(self.pairing_fq12_product());
        let _ = builder.arg(&lanes);
        let _ = builder.arg(&pairs);
        let _ = builder.arg(&mut product);
        // SAFETY: one block per segment, so `blockIdx.x < segments` selects the
        // `count` Fq12 values at `blockIdx.x * count` of `lanes`, which holds
        // `segments * count` of them; each thread strides by `blockDim.x` from
        // `threadIdx.x` and so stays below `count`. Shared memory is
        // `PRODUCT_BLOCK * FQ12_LIMBS` u64s, matching `shared_mem_bytes`, and
        // `PRODUCT_BLOCK` is a power of two so the halving tree covers the
        // block; every thread reaches each `__syncthreads()` because the strided
        // loop and the tree sit outside any early return. Only thread 0 writes,
        // to slot `blockIdx.x` of the freshly allocated `product`, which is
        // distinct from `lanes`.
        let _ = unsafe {
            builder.launch(LaunchConfig {
                grid_dim: (lanes_of, 1, 1),
                block_dim: (PRODUCT_BLOCK, 1, 1),
                shared_mem_bytes: shared,
            })
        }?;

        self.download_u64(&product)
    }
}

#[cfg(test)]
#[expect(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "test module: device operations fail loudly"
)]
mod tests {
    use ark_bn254::{Bn254, Fq12, Fq6, G1Projective, G2Projective};
    use ark_ec::pairing::Pairing;
    use ark_ec::CurveGroup;
    use ark_ff::{BigInt, UniformRand};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use super::*;
    use crate::cuda::common::context::shared_context;

    fn g1_words(point: &G1Projective) -> Vec<u64> {
        let mut out = Vec::with_capacity(3 * FQ_LIMBS);
        out.extend_from_slice(&point.x.0 .0);
        out.extend_from_slice(&point.y.0 .0);
        out.extend_from_slice(&point.z.0 .0);
        out
    }

    fn g2_words(point: &G2Projective) -> Vec<u64> {
        let mut out = Vec::with_capacity(6 * FQ_LIMBS);
        for value in [&point.x, &point.y, &point.z] {
            out.extend_from_slice(&value.c0.0 .0);
            out.extend_from_slice(&value.c1.0 .0);
        }
        out
    }

    fn fq(words: &[u64]) -> Fq {
        Fq::new_unchecked(BigInt([words[0], words[1], words[2], words[3]]))
    }

    fn fq2(words: &[u64]) -> Fq2 {
        Fq2::new(fq(&words[..FQ_LIMBS]), fq(&words[FQ_LIMBS..]))
    }

    fn fq12(words: &[u64]) -> Fq12 {
        let fq6 = |w: &[u64]| {
            Fq6::new(
                fq2(&w[..2 * FQ_LIMBS]),
                fq2(&w[2 * FQ_LIMBS..4 * FQ_LIMBS]),
                fq2(&w[4 * FQ_LIMBS..]),
            )
        };
        Fq12::new(fq6(&words[..6 * FQ_LIMBS]), fq6(&words[6 * FQ_LIMBS..]))
    }

    fn device_miller(ps: &[G1Projective], qs: &[G2Projective]) -> Fq12 {
        let context = shared_context().expect("a CUDA device");
        let g1: Vec<u64> = ps.iter().flat_map(g1_words).collect();
        let g2: Vec<u64> = qs.iter().flat_map(g2_words).collect();
        let device_g1 = context.upload_raw_u64(&g1).unwrap();
        let device_g2 = context.upload_raw_u64(&g2).unwrap();
        let limbs = context
            .multi_miller_batch(&device_g1, &device_g2, &[(0, 0)], ps.len())
            .unwrap();
        fq12(&limbs)
    }

    fn device_miller_warp(ps: &[G1Projective], qs: &[G2Projective]) -> Fq12 {
        let context = shared_context().expect("a CUDA device");
        let g1: Vec<u64> = ps.iter().flat_map(g1_words).collect();
        let g2: Vec<u64> = qs.iter().flat_map(g2_words).collect();
        let device_g1 = context.upload_raw_u64(&g1).unwrap();
        let device_g2 = context.upload_raw_u64(&g2).unwrap();
        let limbs = context
            .multi_miller_warp_batch(&device_g1, &device_g2, &[(0, 0)], ps.len())
            .unwrap();
        fq12(&limbs)
    }

    fn device_miller_warp_segments(
        ps: &[G1Projective],
        qs: &[G2Projective],
        count: usize,
    ) -> Vec<Fq12> {
        let context = shared_context().expect("a CUDA device");
        let g1: Vec<u64> = ps.iter().flat_map(g1_words).collect();
        let g2: Vec<u64> = qs.iter().flat_map(g2_words).collect();
        let device_g1 = context.upload_raw_u64(&g1).unwrap();
        let device_g2 = context.upload_raw_u64(&g2).unwrap();
        let segments: Vec<(usize, usize)> = (0..ps.len() / count)
            .map(|segment| (segment * count, segment * count))
            .collect();
        let limbs = context
            .multi_miller_warp_batch(&device_g1, &device_g2, &segments, count)
            .unwrap();
        limbs.chunks_exact(FQ12_LIMBS).map(fq12).collect()
    }

    fn shapes(seed: u64) -> Vec<(Vec<G1Projective>, Vec<G2Projective>)> {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut cases = Vec::new();
        for count in [1usize, 2, 3, 5, 64, 129] {
            let ps = (0..count)
                .map(|_| G1Projective::rand(&mut rng))
                .collect::<Vec<_>>();
            let qs = (0..count)
                .map(|_| G2Projective::rand(&mut rng))
                .collect::<Vec<_>>();
            cases.push((ps, qs));
        }

        let affine_p: Vec<G1Projective> = (0..4)
            .map(|_| G1Projective::rand(&mut rng).into_affine().into())
            .collect();
        let affine_q: Vec<G2Projective> = (0..4)
            .map(|_| G2Projective::rand(&mut rng).into_affine().into())
            .collect();
        cases.push((affine_p, affine_q));

        let mut ps = (0..4)
            .map(|_| G1Projective::rand(&mut rng))
            .collect::<Vec<_>>();
        let mut qs = (0..4)
            .map(|_| G2Projective::rand(&mut rng))
            .collect::<Vec<_>>();
        ps[1] = G1Projective::default();
        qs[2] = G2Projective::default();
        cases.push((ps, qs));

        cases
    }

    #[test]
    fn multi_miller_warp_matches_arkworks() {
        if shared_context().is_none() {
            return;
        }
        for (index, (ps, qs)) in shapes(5_400).into_iter().enumerate() {
            let expected = Bn254::multi_miller_loop(ps.clone(), qs.clone()).0;
            let got = device_miller_warp(&ps, &qs);
            assert_eq!(
                got,
                expected,
                "the warp Miller loop diverged for shape {index} of {} pairs",
                ps.len()
            );
        }
    }

    #[test]
    fn multi_miller_warp_segments_match_arkworks() {
        if shared_context().is_none() {
            return;
        }
        let mut rng = ChaCha20Rng::seed_from_u64(5_500);
        for (count, segments) in [(1usize, 2usize), (1, 4), (3, 3), (5, 2), (33, 2)] {
            let ps: Vec<G1Projective> = (0..count * segments)
                .map(|_| G1Projective::rand(&mut rng))
                .collect();
            let qs: Vec<G2Projective> = (0..count * segments)
                .map(|_| G2Projective::rand(&mut rng))
                .collect();
            let expected: Vec<Fq12> = (0..segments)
                .map(|segment| {
                    let span = segment * count..(segment + 1) * count;
                    Bn254::multi_miller_loop(ps[span.clone()].to_vec(), qs[span].to_vec()).0
                })
                .collect();
            let got = device_miller_warp_segments(&ps, &qs, count);
            assert_eq!(
                got, expected,
                "the warp Miller loop diverged for {segments} segments of {count} pairs"
            );
        }
    }

    #[test]
    fn device_pairing_matches_arkworks() {
        if shared_context().is_none() {
            return;
        }
        for (index, (ps, qs)) in shapes(5_200).into_iter().enumerate() {
            let expected = Bn254::multi_pairing(ps.clone(), qs.clone());
            let got = Bn254::final_exponentiation(ark_ec::pairing::MillerLoopOutput(
                device_miller(&ps, &qs),
            ))
            .expect("the Miller output is non-degenerate");
            assert_eq!(got, expected, "the pairing diverged for shape {index}");
        }
    }
}
