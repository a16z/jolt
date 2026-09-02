//! Wide-parallel host tail for a device-resident Dory reduce loop.

#![expect(
    clippy::expect_used,
    reason = "mirrors the stock dory-pcs driver's invertibility expects"
)]
#![expect(
    clippy::indexing_slicing,
    reason = "prover-only reduce tail over fixed-shape halves; every index is \
              pinned by the round geometry asserted at entry"
)]

use ark_bn254::{Bn254, G1Affine, G2Affine};
use ark_ec::bn::G2Prepared as BnG2Prepared;
use ark_ec::pairing::{MillerLoopOutput, Pairing, PairingOutput};
use ark_ec::CurveGroup;
use ark_ff::One;
use rayon::prelude::*;

use dory::backends::arkworks::{ArkFr, ArkG1, ArkG2, ArkGT};
use dory::messages::{FirstReduceMessage, SecondReduceMessage};
use dory::primitives::arithmetic::DoryRoutines;

use crate::routines::{JoltG1Routines, JoltG2Routines};

type RawG1 = ark_bn254::G1Projective;
type RawG2 = ark_bn254::G2Projective;
type G2Prepared = BnG2Prepared<ark_bn254::Config>;

#[inline]
fn g1_raw(points: &[ArkG1]) -> &[RawG1] {
    // SAFETY: ArkG1 is repr(transparent) over G1Projective.
    unsafe { std::slice::from_raw_parts(points.as_ptr().cast::<RawG1>(), points.len()) }
}

#[inline]
fn g2_raw(points: &[ArkG2]) -> &[RawG2] {
    // SAFETY: ArkG2 is repr(transparent) over G2Projective.
    unsafe { std::slice::from_raw_parts(points.as_ptr().cast::<RawG2>(), points.len()) }
}

const MILLER_CHUNK: usize = 8;

fn par_miller_prepared(g1: &[G1Affine], g2: &[&G2Prepared]) -> MillerLoopOutput<Bn254> {
    debug_assert_eq!(g1.len(), g2.len());
    g1.par_chunks(MILLER_CHUNK)
        .zip(g2.par_chunks(MILLER_CHUNK))
        .map(|(ps, qs)| {
            Bn254::multi_miller_loop(ps.iter().copied(), qs.iter().map(|q| (*q).clone()))
        })
        .reduce(
            || MillerLoopOutput(<Bn254 as Pairing>::TargetField::one()),
            |a, b| MillerLoopOutput(a.0 * b.0),
        )
}

fn par_miller(g1: &[G1Affine], g2: &[G2Affine]) -> MillerLoopOutput<Bn254> {
    debug_assert_eq!(g1.len(), g2.len());
    g1.par_chunks(MILLER_CHUNK)
        .zip(g2.par_chunks(MILLER_CHUNK))
        .map(|(ps, qs)| {
            Bn254::multi_miller_loop(ps.iter().copied(), qs.iter().map(|q| G2Prepared::from(*q)))
        })
        .reduce(
            || MillerLoopOutput(<Bn254 as Pairing>::TargetField::one()),
            |a, b| MillerLoopOutput(a.0 * b.0),
        )
}

#[inline]
#[expect(
    clippy::large_types_passed_by_value,
    reason = "MillerLoopOutput is consumed by final exponentiation"
)]
fn final_exp(miller: MillerLoopOutput<Bn254>) -> ArkGT {
    let output: PairingOutput<Bn254> =
        Bn254::final_exponentiation(miller).expect("final exponentiation of a Miller value");
    ArkGT(output)
}

/// Value-exact host implementation of the shrinking transparent rounds.
pub struct FastTail {
    v1: Vec<ArkG1>,
    v2: Vec<ArkG2>,
    g1: Vec<ArkG1>,
    g2: Vec<ArkG2>,
    g2_prepared: Vec<G2Prepared>,
    n: usize,
}

impl FastTail {
    /// Batch-normalize and prepare the largest setup prefix once.
    pub fn new(v1: Vec<ArkG1>, v2: Vec<ArkG2>, g1: Vec<ArkG1>, g2: Vec<ArkG2>) -> Self {
        let n = v1.len();
        debug_assert_eq!(v2.len(), n);
        debug_assert_eq!(g1.len(), n);
        debug_assert_eq!(g2.len(), n);
        let g2_prepared = if n >= 2 {
            let affines = RawG2::normalize_batch(g2_raw(&g2[..n / 2]));
            affines.into_par_iter().map(G2Prepared::from).collect()
        } else {
            Vec::new()
        };
        Self {
            v1,
            v2,
            g1,
            g2,
            g2_prepared,
            n,
        }
    }

    /// Stock transparent first message with tail-wide parallelism.
    pub fn compute_first_message(
        &self,
        s1: &[ArkFr],
        s2: &[ArkFr],
    ) -> FirstReduceMessage<ArkG1, ArkG2, ArkGT> {
        let n = self.n;
        let n2 = n / 2;
        let (v1_affine, v2_affine) = rayon::join(
            || RawG1::normalize_batch(g1_raw(&self.v1)),
            || RawG2::normalize_batch(g2_raw(&self.v2)),
        );
        let g1_affine = RawG1::normalize_batch(g1_raw(&self.g1[..n2]));
        let v2_prepared: Vec<G2Prepared> =
            v2_affine.into_par_iter().map(G2Prepared::from).collect();
        let g2_refs: Vec<_> = self.g2_prepared[..n2].iter().collect();
        let v2_left: Vec<_> = v2_prepared[..n2].iter().collect();
        let v2_right: Vec<_> = v2_prepared[n2..].iter().collect();

        let (((d1_left, d1_right), (d2_left, d2_right)), (e1_beta, e2_beta)) = rayon::join(
            || {
                rayon::join(
                    || {
                        rayon::join(
                            || final_exp(par_miller_prepared(&v1_affine[..n2], &g2_refs)),
                            || final_exp(par_miller_prepared(&v1_affine[n2..], &g2_refs)),
                        )
                    },
                    || {
                        rayon::join(
                            || final_exp(par_miller_prepared(&g1_affine, &v2_left)),
                            || final_exp(par_miller_prepared(&g1_affine, &v2_right)),
                        )
                    },
                )
            },
            || {
                rayon::join(
                    || JoltG1Routines::msm(&self.g1[..n], s2),
                    || JoltG2Routines::msm(&self.g2[..n], s1),
                )
            },
        );
        FirstReduceMessage {
            d1_left,
            d1_right,
            d2_left,
            d2_right,
            e1_beta,
            e2_beta,
        }
    }

    /// Apply the stock first challenge using its already-computed inverse.
    pub fn apply_first_challenge(&mut self, beta: &ArkFr, beta_inv: &ArkFr) {
        let n = self.n;
        JoltG1Routines::host_fixed_scalar_mul_bases_then_add(&self.g1[..n], &mut self.v1, beta);
        JoltG2Routines::host_fixed_scalar_mul_bases_then_add(&self.g2[..n], &mut self.v2, beta_inv);
    }

    /// Stock transparent second message with tail-wide parallelism.
    pub fn compute_second_message(
        &self,
        s1: &[ArkFr],
        s2: &[ArkFr],
    ) -> SecondReduceMessage<ArkG1, ArkG2, ArkGT> {
        let n2 = self.n / 2;
        let (v1_affine, v2_affine) = rayon::join(
            || RawG1::normalize_batch(g1_raw(&self.v1)),
            || RawG2::normalize_batch(g2_raw(&self.v2)),
        );
        let (s1_left, s1_right) = s1.split_at(n2);
        let (s2_left, s2_right) = s2.split_at(n2);
        let ((c_plus, c_minus), ((e1_plus, e1_minus), (e2_plus, e2_minus))) = rayon::join(
            || {
                rayon::join(
                    || final_exp(par_miller(&v1_affine[..n2], &v2_affine[n2..])),
                    || final_exp(par_miller(&v1_affine[n2..], &v2_affine[..n2])),
                )
            },
            || {
                rayon::join(
                    || {
                        rayon::join(
                            || JoltG1Routines::msm(&self.v1[..n2], s2_right),
                            || JoltG1Routines::msm(&self.v1[n2..], s2_left),
                        )
                    },
                    || {
                        rayon::join(
                            || JoltG2Routines::msm(&self.v2[n2..], s1_left),
                            || JoltG2Routines::msm(&self.v2[..n2], s1_right),
                        )
                    },
                )
            },
        );
        SecondReduceMessage {
            c_plus,
            c_minus,
            e1_plus,
            e1_minus,
            e2_plus,
            e2_minus,
        }
    }

    /// Apply the stock second challenge and halve the point vectors.
    pub fn apply_second_challenge(&mut self, alpha: &ArkFr, alpha_inv: &ArkFr) {
        let n2 = self.n / 2;
        let (v1_left, v1_right) = self.v1.split_at_mut(n2);
        JoltG1Routines::host_fixed_scalar_mul_vs_then_add(v1_left, v1_right, alpha);
        self.v1.truncate(n2);
        let (v2_left, v2_right) = self.v2.split_at_mut(n2);
        JoltG2Routines::host_fixed_scalar_mul_vs_then_add(v2_left, v2_right, alpha_inv);
        self.v2.truncate(n2);
        self.n = n2;
    }

    /// Current live vectors.
    pub fn vectors(&self) -> (&[ArkG1], &[ArkG2]) {
        (&self.v1, &self.v2)
    }

    /// Return the fully reduced point vectors to the vendored state machine.
    pub fn into_vectors(self) -> (Vec<ArkG1>, Vec<ArkG2>) {
        (self.v1, self.v2)
    }
}

#[cfg(test)]
mod tests {
    use ark_bn254::{Fr, G1Projective, G2Projective};
    use ark_ff::{Field, UniformRand};
    use dory::backends::arkworks::BN254 as InnerBN254;
    use dory::primitives::arithmetic::PairingCurve;
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use super::*;

    #[test]
    fn messages_and_folds_match_stock_routines() {
        let mut rng = ChaCha20Rng::seed_from_u64(0xfa57_7a11);
        let n = 64;
        let g1: Vec<_> = (0..n)
            .map(|_| ArkG1(G1Projective::rand(&mut rng)))
            .collect();
        let g2: Vec<_> = (0..n)
            .map(|_| ArkG2(G2Projective::rand(&mut rng)))
            .collect();
        let v1: Vec<_> = (0..n)
            .map(|_| ArkG1(G1Projective::rand(&mut rng)))
            .collect();
        let v2: Vec<_> = (0..n)
            .map(|_| ArkG2(G2Projective::rand(&mut rng)))
            .collect();
        let s1: Vec<_> = (0..n).map(|_| ArkFr(Fr::rand(&mut rng))).collect();
        let s2: Vec<_> = (0..n).map(|_| ArkFr(Fr::rand(&mut rng))).collect();
        let n2 = n / 2;
        let mut tail = FastTail::new(v1.clone(), v2.clone(), g1.clone(), g2.clone());

        let first = tail.compute_first_message(&s1, &s2);
        assert_eq!(
            first,
            FirstReduceMessage {
                d1_left: InnerBN254::multi_pair_g2_setup(&v1[..n2], &g2[..n2]),
                d1_right: InnerBN254::multi_pair_g2_setup(&v1[n2..], &g2[..n2]),
                d2_left: InnerBN254::multi_pair_g1_setup(&g1[..n2], &v2[..n2]),
                d2_right: InnerBN254::multi_pair_g1_setup(&g1[..n2], &v2[n2..]),
                e1_beta: JoltG1Routines::msm(&g1, &s2),
                e2_beta: JoltG2Routines::msm(&g2, &s1),
            }
        );

        let beta = ArkFr(Fr::rand(&mut rng));
        let beta_inv = ArkFr(beta.0.inverse().expect("random beta is invertible"));
        tail.apply_first_challenge(&beta, &beta_inv);
        let mut expected_v1 = v1;
        let mut expected_v2 = v2;
        JoltG1Routines::fixed_scalar_mul_bases_then_add(&g1, &mut expected_v1, &beta);
        JoltG2Routines::fixed_scalar_mul_bases_then_add(&g2, &mut expected_v2, &beta_inv);

        let second = tail.compute_second_message(&s1, &s2);
        let (s1_left, s1_right) = s1.split_at(n2);
        let (s2_left, s2_right) = s2.split_at(n2);
        assert_eq!(
            second,
            SecondReduceMessage {
                c_plus: InnerBN254::multi_pair(&expected_v1[..n2], &expected_v2[n2..]),
                c_minus: InnerBN254::multi_pair(&expected_v1[n2..], &expected_v2[..n2]),
                e1_plus: JoltG1Routines::msm(&expected_v1[..n2], s2_right),
                e1_minus: JoltG1Routines::msm(&expected_v1[n2..], s2_left),
                e2_plus: JoltG2Routines::msm(&expected_v2[n2..], s1_left),
                e2_minus: JoltG2Routines::msm(&expected_v2[..n2], s1_right),
            }
        );

        let alpha = ArkFr(Fr::rand(&mut rng));
        let alpha_inv = ArkFr(alpha.0.inverse().expect("random alpha is invertible"));
        tail.apply_second_challenge(&alpha, &alpha_inv);
        let (expected_v1_left, expected_v1_right) = expected_v1.split_at_mut(n2);
        JoltG1Routines::fixed_scalar_mul_vs_then_add(expected_v1_left, expected_v1_right, &alpha);
        let (expected_v2_left, expected_v2_right) = expected_v2.split_at_mut(n2);
        JoltG2Routines::fixed_scalar_mul_vs_then_add(
            expected_v2_left,
            expected_v2_right,
            &alpha_inv,
        );
        assert_eq!(tail.vectors().0, &expected_v1[..n2]);
        assert_eq!(tail.vectors().1, &expected_v2[..n2]);
    }
}
