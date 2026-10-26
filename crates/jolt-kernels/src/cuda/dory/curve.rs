//! Dory curve adapter for resident group handles and pairing-result decoding.
//!
//! Contiguous multi-pairing inputs become deferred device requests; individual
//! pairings and other input layouts use the CPU curve.

use ark_bn254::{Fq, Fq12, Fq2, Fq6};
use ark_ff::BigInt;
#[cfg(test)]
use dory::backends::arkworks::{ArkG1, ArkG2, BN254};
#[cfg(test)]
use dory::primitives::arithmetic::PairingCurve;

#[cfg(test)]
use super::gt::DeviceGT;
#[cfg(test)]
use super::handle::{load_all, load_all_g2, span, span_g2, DeviceG1, DeviceG2};
use crate::cuda::common::msm::FQ_LIMBS;

#[cfg(test)]
#[derive(Clone)]
pub(super) struct CudaBN254;

#[cfg(test)]
fn ark(points: &[DeviceG1]) -> Vec<ArkG1> {
    load_all(points).into_iter().map(ArkG1).collect()
}

#[cfg(test)]
fn ark_g2(points: &[DeviceG2]) -> Vec<ArkG2> {
    load_all_g2(points).into_iter().map(ArkG2).collect()
}

fn fq(words: &[u64]) -> Fq {
    Fq::new_unchecked(BigInt([words[0], words[1], words[2], words[3]]))
}

fn fq2(words: &[u64]) -> Fq2 {
    Fq2::new(fq(&words[..FQ_LIMBS]), fq(&words[FQ_LIMBS..]))
}

pub(super) fn fq12(words: &[u64]) -> Fq12 {
    let fq6 = |chunk: &[u64]| {
        Fq6::new(
            fq2(&chunk[..2 * FQ_LIMBS]),
            fq2(&chunk[2 * FQ_LIMBS..4 * FQ_LIMBS]),
            fq2(&chunk[4 * FQ_LIMBS..]),
        )
    };
    Fq12::new(fq6(&words[..6 * FQ_LIMBS]), fq6(&words[6 * FQ_LIMBS..]))
}

#[cfg(test)]
#[tracing::instrument(skip_all, name = "cuda_multi_pair", fields(pairs = ps.len()))]
fn deferred_multi_pair(ps: &[DeviceG1], qs: &[DeviceG2]) -> Option<DeviceGT> {
    if ps.len() != qs.len() || ps.is_empty() {
        return None;
    }
    let (Some(g1_offset), Some(g2_offset)) = (span(ps), span_g2(qs)) else {
        return None;
    };
    Some(DeviceGT::enqueue(g1_offset, g2_offset, ps.len()))
}

#[cfg(test)]
impl PairingCurve for CudaBN254 {
    type G1 = DeviceG1;
    type G2 = DeviceG2;
    type GT = DeviceGT;

    fn pair(p: &Self::G1, q: &Self::G2) -> Self::GT {
        DeviceGT::from(BN254::pair(&ArkG1(p.load()), &ArkG2(q.load())))
    }

    fn multi_pair(ps: &[Self::G1], qs: &[Self::G2]) -> Self::GT {
        deferred_multi_pair(ps, qs)
            .unwrap_or_else(|| DeviceGT::from(BN254::multi_pair(&ark(ps), &ark_g2(qs))))
    }

    fn multi_pair_g1_setup(ps: &[Self::G1], qs: &[Self::G2]) -> Self::GT {
        deferred_multi_pair(ps, qs)
            .unwrap_or_else(|| DeviceGT::from(BN254::multi_pair_g1_setup(&ark(ps), &ark_g2(qs))))
    }

    fn multi_pair_g2_setup(ps: &[Self::G1], qs: &[Self::G2]) -> Self::GT {
        deferred_multi_pair(ps, qs)
            .unwrap_or_else(|| DeviceGT::from(BN254::multi_pair_g2_setup(&ark(ps), &ark_g2(qs))))
    }
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests fail loudly on setup errors")]
mod tests {
    use ark_bn254::{Bn254, G1Projective, G2Projective};
    use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
    use ark_ff::{Field, UniformRand};
    use dory::backends::arkworks::ArkGT;
    use dory::primitives::arithmetic::Group;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use super::super::{
        arena, gt,
        handle::{store_all, store_frozen_g2},
    };
    use super::*;
    use crate::cuda::common::context::shared_context;

    type MultiPair = fn(&[DeviceG1], &[DeviceG2]) -> DeviceGT;
    const MULTI_PAIRS: [MultiPair; 3] = [
        CudaBN254::multi_pair,
        CudaBN254::multi_pair_g1_setup,
        CudaBN254::multi_pair_g2_setup,
    ];

    fn points(count: usize) -> (Vec<G1Projective>, Vec<G2Projective>) {
        let mut rng = ChaCha20Rng::seed_from_u64(21_113);
        let ps = (0..count)
            .map(|i| {
                if i % 17 == 0 {
                    return G1Projective::default();
                }
                let mut point = G1Projective::rand(&mut rng).into_affine().into_group();
                let z = Fq::from((i % 13 + 2) as u64);
                point.x *= z.square();
                point.y *= z.square() * z;
                point.z *= z;
                point
            })
            .collect();
        let qs = (0..count)
            .map(|i| {
                if i % 19 == 0 {
                    return G2Projective::default();
                }
                let mut point = G2Projective::rand(&mut rng).into_affine().into_group();
                let z = Fq2::new(Fq::from((i % 11 + 2) as u64), Fq::from(3u64));
                point.x *= z.square();
                point.y *= z.square() * z;
                point.z *= z;
                point
            })
            .collect();
        (ps, qs)
    }

    #[test]
    fn individual_pairings_are_eager_and_match_arkworks() {
        if shared_context().is_none() {
            return;
        }
        let guard = arena::open(4, 4).expect("arena");
        gt::reset();
        let (ps, qs) = points(4);
        let hp = store_all(&ps);
        let hq = store_frozen_g2(&qs);
        let indices = [(0, 0), (0, 1), (1, 0), (1, 2), (3, 1)];
        let values: Vec<_> = indices
            .iter()
            .map(|&(p, q)| CudaBN254::pair(&hp[p], &hq[q]))
            .collect();
        drop(guard);
        for (&(p, q), value) in indices.iter().zip(values) {
            assert_eq!(value.value(), ArkGT(Bn254::pairing(ps[p], qs[q])));
        }
        assert!(!arena::poisoned());
        gt::reset();
    }

    #[test]
    fn contiguous_multi_pairings_match_arkworks() {
        if shared_context().is_none() {
            return;
        }
        let _guard = arena::open(133, 133).expect("arena");
        gt::reset();
        let (ps, qs) = points(133);
        let hp = store_all(&ps);
        let hq = store_frozen_g2(&qs);
        for (p, q, count) in [(0, 0, 1), (1, 2, 1), (2, 3, 17), (1, 4, 129)] {
            let expected = ArkGT(Bn254::multi_pairing(
                ps[p..p + count].to_vec(),
                qs[q..q + count].to_vec(),
            ));
            let deferred = deferred_multi_pair(&hp[p..p + count], &hq[q..q + count])
                .expect("contiguous handles use the device queue");
            let values = MULTI_PAIRS.map(|pair| pair(&hp[p..p + count], &hq[q..q + count]));
            assert_eq!(deferred.value(), expected);
            for value in values {
                assert_eq!(value.value(), expected);
            }
        }
        assert_eq!(load_all(&hp), ps);
        assert_eq!(load_all_g2(&hq), qs);
        assert!(!arena::poisoned());
        gt::reset();
    }

    #[test]
    fn noncontiguous_multi_pairings_preserve_input_order() {
        if shared_context().is_none() {
            return;
        }
        let guard = arena::open(8, 8).expect("arena");
        gt::reset();
        let (ps, qs) = points(8);
        let hp = store_all(&ps);
        let hq = store_frozen_g2(&qs);
        let mut results = Vec::new();
        for (p_indices, q_indices) in [
            ([1, 3, 5, 7], [1, 2, 3, 4]),
            ([1, 2, 3, 4], [7, 5, 3, 1]),
            ([3, 2, 1, 0], [0, 1, 2, 3]),
            ([2, 2, 0, 5], [1, 3, 3, 0]),
        ] {
            let p_handles = p_indices.map(|i| hp[i]);
            let q_handles = q_indices.map(|i| hq[i]);
            assert!(deferred_multi_pair(&p_handles, &q_handles).is_none());
            let expected = ArkGT(Bn254::multi_pairing(
                p_indices.map(|i| ps[i]),
                q_indices.map(|i| qs[i]),
            ));
            for pair in MULTI_PAIRS {
                results.push((pair(&p_handles, &q_handles), expected));
            }
        }
        assert_eq!(load_all(&hp), ps);
        assert_eq!(load_all_g2(&hq), qs);
        drop(guard);
        for (value, expected) in results {
            assert_eq!(value.value(), expected);
        }
        assert!(!arena::poisoned());
        gt::reset();
    }

    #[test]
    fn empty_multi_pairings_are_identity_without_an_arena() {
        gt::reset();
        assert!(deferred_multi_pair(&[], &[]).is_none());
        for pair in MULTI_PAIRS {
            assert_eq!(pair(&[], &[]).value(), ArkGT::identity());
        }
        gt::reset();
    }

    #[test]
    fn mismatched_multi_pairings_are_rejected() {
        if shared_context().is_none() {
            return;
        }
        let _guard = arena::open(2, 2).expect("arena");
        gt::reset();
        let (ps, qs) = points(2);
        let hp = store_all(&ps);
        let hq = store_frozen_g2(&qs);
        for (ps, qs) in [
            (&hp[..1], &hq[..2]),
            (&hp[..2], &hq[..1]),
            (&hp[..0], &hq[..1]),
            (&hp[..1], &hq[..0]),
        ] {
            assert!(deferred_multi_pair(ps, qs).is_none());
            for pair in MULTI_PAIRS {
                assert!(std::panic::catch_unwind(|| pair(ps, qs)).is_err());
            }
        }
        assert!(!arena::poisoned());
        gt::reset();
    }
}
