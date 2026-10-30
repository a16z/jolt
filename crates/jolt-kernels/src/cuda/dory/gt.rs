//! Deferred target-group values for Dory's multi-pairings.
//!
//! Pending values record arena spans instead of immediately pairing them.
//! Reading a pending value, including for transcript serialization, batches
//! outstanding requests with equal pair counts into GPU Miller loops. Final
//! exponentiation and subsequent target-group arithmetic use host Arkworks
//! values. The queue belongs to the current opening's host thread.
//! Pending spans must remain unchanged until resolution. Slot handles must stay
//! on their owning thread and must not outlive a queue reset.

use std::cell::RefCell;
use std::ops::{Add, Mul, Neg, Sub};

use ark_bn254::Bn254;
use ark_ec::pairing::{MillerLoopOutput, Pairing};
use ark_serialize::{Read, Write};
use dory::backends::arkworks::{ArkFr, ArkGT};
use dory::primitives::arithmetic::Group as DoryGroup;
use dory::primitives::{Compress, DoryDeserialize, DorySerialize, SerializationError, Validate};

use super::arena;
use crate::cuda::common::pairing::FQ12_LIMBS;

#[derive(Clone, Copy)]
struct Request {
    g1_offset: usize,
    g2_offset: usize,
    count: usize,
}

#[expect(
    clippy::large_enum_variant,
    reason = "ready GT values are kept inline to avoid a separate allocation per slot"
)]
enum Slot {
    Pending(Request),
    Ready(ArkGT),
}

thread_local! {
    static SLOTS: RefCell<Vec<Slot>> = const { RefCell::new(Vec::new()) };
}

pub(super) fn reset() {
    SLOTS.with_borrow_mut(Vec::clear);
}

#[derive(Clone, Copy, Debug)]
pub struct DeviceGT {
    index: u32,
}

fn push(slot: Slot) -> DeviceGT {
    SLOTS.with_borrow_mut(|slots| {
        slots.push(slot);
        let index = u32::try_from(slots.len() - 1).unwrap_or(u32::MAX);
        DeviceGT { index }
    })
}

#[expect(
    clippy::large_types_passed_by_value,
    reason = "ArkGT is Copy and is moved straight into the slot; a reference would only add a deref"
)]
fn host(value: ArkGT) -> DeviceGT {
    push(Slot::Ready(value))
}

fn lane_words(limbs: &[u64], lane: usize) -> Option<&[u64]> {
    limbs.get(lane * FQ12_LIMBS..(lane + 1) * FQ12_LIMBS)
}

#[cfg(feature = "parallel")]
fn final_exponentiations(limbs: &[u64], lanes: usize) -> Vec<Option<ArkGT>> {
    use rayon::prelude::*;

    (0..lanes)
        .into_par_iter()
        .map(|lane| lane_words(limbs, lane).and_then(final_exponentiation))
        .collect()
}

#[cfg(not(feature = "parallel"))]
fn final_exponentiations(limbs: &[u64], lanes: usize) -> Vec<Option<ArkGT>> {
    (0..lanes)
        .map(|lane| lane_words(limbs, lane).and_then(final_exponentiation))
        .collect()
}

fn final_exponentiation(limbs: &[u64]) -> Option<ArkGT> {
    <Bn254 as Pairing>::final_exponentiation(MillerLoopOutput(super::curve::fq12(limbs))).map(ArkGT)
}

fn flush() {
    let batches = SLOTS.with_borrow(|slots| {
        let mut grouped: Vec<(usize, Vec<(usize, Request)>)> = Vec::new();
        for (index, slot) in slots.iter().enumerate() {
            if let Slot::Pending(request) = slot {
                match grouped
                    .iter_mut()
                    .find(|(count, _)| *count == request.count)
                {
                    Some((_, members)) => members.push((index, *request)),
                    None => grouped.push((request.count, vec![(index, *request)])),
                }
            }
        }
        grouped
    });

    for (count, members) in batches {
        let span = tracing::info_span!("cuda_miller_batch", pairs = count, lanes = members.len());
        let _entered = span.enter();
        let spans: Vec<(usize, usize)> = members
            .iter()
            .map(|(_, request)| (request.g1_offset, request.g2_offset))
            .collect();
        let limbs = match arena::multi_miller_batch(&spans, count) {
            Ok(limbs) => limbs,
            Err(error) => {
                tracing::error!(?error, "the batched multi-Miller loop failed");
                arena::poison("a batched multi-Miller loop failed");
                continue;
            }
        };
        let values = final_exponentiations(&limbs, members.len());
        for ((index, _), value) in members.iter().zip(values) {
            let Some(value) = value else {
                arena::poison("a batched Miller output was degenerate");
                continue;
            };
            SLOTS.with_borrow_mut(|slots| {
                if let Some(slot) = slots.get_mut(*index) {
                    *slot = Slot::Ready(value);
                }
            });
        }
    }
}

impl DeviceGT {
    pub(super) fn enqueue(g1_offset: usize, g2_offset: usize, count: usize) -> Self {
        push(Slot::Pending(Request {
            g1_offset,
            g2_offset,
            count,
        }))
    }

    pub(super) fn value(self) -> ArkGT {
        let ready = SLOTS.with_borrow(|slots| match slots.get(self.index as usize) {
            Some(Slot::Ready(value)) => Some(*value),
            _ => None,
        });
        if let Some(value) = ready {
            return value;
        }
        flush();
        SLOTS
            .with_borrow(|slots| match slots.get(self.index as usize) {
                Some(Slot::Ready(value)) => Some(*value),
                _ => None,
            })
            .unwrap_or_else(|| {
                arena::poison("a deferred GT value never materialised");
                ArkGT::default()
            })
    }
}

impl From<ArkGT> for DeviceGT {
    fn from(value: ArkGT) -> Self {
        host(value)
    }
}

impl PartialEq for DeviceGT {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index || self.value() == other.value()
    }
}

impl Add for DeviceGT {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        host(self.value() + rhs.value())
    }
}

impl Add<&Self> for DeviceGT {
    type Output = Self;
    fn add(self, rhs: &Self) -> Self {
        host(self.value() + rhs.value())
    }
}

impl Sub for DeviceGT {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        host(self.value() - rhs.value())
    }
}

impl Sub<&Self> for DeviceGT {
    type Output = Self;
    fn sub(self, rhs: &Self) -> Self {
        host(self.value() - rhs.value())
    }
}

impl Neg for DeviceGT {
    type Output = Self;
    fn neg(self) -> Self {
        host(-self.value())
    }
}

impl Mul<DeviceGT> for ArkFr {
    type Output = DeviceGT;
    fn mul(self, rhs: DeviceGT) -> DeviceGT {
        host(self * rhs.value())
    }
}

impl Mul<&DeviceGT> for ArkFr {
    type Output = DeviceGT;
    fn mul(self, rhs: &DeviceGT) -> DeviceGT {
        host(self * rhs.value())
    }
}

impl DorySerialize for DeviceGT {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.value().serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.value().serialized_size(compress)
    }
}

impl DoryDeserialize for DeviceGT {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(host(ArkGT::deserialize_with_mode(
            reader, compress, validate,
        )?))
    }
}

impl DoryGroup for DeviceGT {
    type Scalar = ArkFr;

    fn identity() -> Self {
        host(ArkGT::identity())
    }

    fn add(&self, rhs: &Self) -> Self {
        host(DoryGroup::add(&self.value(), &rhs.value()))
    }

    fn neg(&self) -> Self {
        host(DoryGroup::neg(&self.value()))
    }

    fn scale(&self, k: &Self::Scalar) -> Self {
        host(DoryGroup::scale(&self.value(), k))
    }

    fn random() -> Self {
        host(ArkGT::random())
    }
}

#[cfg(test)]
#[expect(
    clippy::expect_used,
    clippy::op_ref,
    reason = "tests exercise owned and borrowed group operations and fail loudly"
)]
mod tests {
    use ark_bn254::{Fq, Fq2, Fr, G1Projective, G2Projective};
    use ark_ec::{AffineRepr, CurveGroup, PrimeGroup};
    use ark_ff::{Field, UniformRand};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use super::super::handle::{store_all, store_frozen_g2};
    use super::*;
    use crate::cuda::common::context::shared_context;

    fn points(count: usize) -> (Vec<G1Projective>, Vec<G2Projective>) {
        let mut rng = ChaCha20Rng::seed_from_u64(21_107);
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
    fn pending_batches_resolve_together_and_cache_values() {
        if shared_context().is_none() {
            return;
        }
        let guard = arena::open(2064, 2064).expect("arena");
        reset();
        let (ps, qs) = points(2064);
        let _ = store_all(&ps);
        let _ = store_frozen_g2(&qs);
        let hosted = DeviceGT::identity();
        let shapes = [
            (2, 3, 2),
            (1, 4, 2048),
            (3, 2, 129),
            (4, 4, 2048),
            (5, 6, 2),
            (7, 4, 2048),
            (9, 4, 2048),
        ];
        let pending: Vec<_> = shapes
            .iter()
            .map(|&(p, q, count)| DeviceGT::enqueue(p, q, count))
            .collect();
        assert_eq!(hosted.value(), ArkGT::identity());
        assert!(SLOTS.with_borrow(|slots| slots
            .iter()
            .skip(1)
            .all(|slot| matches!(slot, Slot::Pending(_)))));
        let expected: Vec<_> = shapes
            .iter()
            .map(|&(p, q, count)| {
                ArkGT(Bn254::multi_pairing(
                    ps[p..p + count].to_vec(),
                    qs[q..q + count].to_vec(),
                ))
            })
            .collect();
        assert_eq!(pending[0].value(), expected[0]);
        assert!(SLOTS.with_borrow(|slots| slots.iter().all(|slot| matches!(slot, Slot::Ready(_)))));
        for (handle, expected) in pending.iter().zip(&expected) {
            assert_eq!(handle.value(), *expected);
        }
        assert!(!arena::poisoned());
        drop(guard);
        for (handle, expected) in pending.iter().zip(&expected) {
            assert_eq!(handle.value(), *expected);
        }
        reset();
        assert!(SLOTS.with_borrow(Vec::is_empty));
    }

    #[test]
    fn group_operations_match_arkworks() {
        reset();
        let a = ArkGT(Bn254::pairing(
            G1Projective::generator(),
            G2Projective::generator(),
        ));
        let b = ArkGT(Bn254::pairing(
            G1Projective::generator() * Fr::from(7u64),
            G2Projective::generator(),
        ));
        let x = DeviceGT::from(a);
        let y = DeviceGT::from(b);
        assert_eq!(x, DeviceGT::from(a));
        assert_ne!(x, y);
        assert_eq!((x + y).value(), a + b);
        assert_eq!((x + &y).value(), a + &b);
        assert_eq!((x - y).value(), a - b);
        assert_eq!((x - &y).value(), a - &b);
        assert_eq!((-x).value(), -a);
        assert_eq!(DoryGroup::add(&x, &y).value(), DoryGroup::add(&a, &b));
        assert_eq!(DoryGroup::neg(&x).value(), DoryGroup::neg(&a));
        let mut rng = ChaCha20Rng::seed_from_u64(21_109);
        for scalar in [
            Fr::from(0u64),
            Fr::from(1u64),
            -Fr::from(1u64),
            Fr::rand(&mut rng),
        ] {
            let scalar = ArkFr(scalar);
            assert_eq!((scalar * x).value(), scalar * a);
            assert_eq!((scalar * &x).value(), scalar * &a);
            assert_eq!(
                DoryGroup::scale(&x, &scalar).value(),
                DoryGroup::scale(&a, &scalar)
            );
        }
        let identity = DeviceGT::identity();
        assert_eq!((x + identity).value(), a);
        assert_eq!((x - x).value(), ArkGT::identity());
        let random = DeviceGT::random();
        assert_eq!((random + -random).value(), ArkGT::identity());
        reset();
    }

    #[test]
    fn serialization_resolves_pending_values_and_matches_host_bytes() {
        if shared_context().is_none() {
            return;
        }
        let guard = arena::open(16, 16).expect("arena");
        let (ps, qs) = points(16);
        let _ = store_all(&ps);
        let _ = store_frozen_g2(&qs);
        let expected = ArkGT(Bn254::multi_pairing(ps[1..6].to_vec(), qs[2..7].to_vec()));
        for compress in [Compress::Yes, Compress::No] {
            reset();
            let handle = DeviceGT::enqueue(1, 2, 5);
            let companion = DeviceGT::enqueue(3, 4, 5);
            let mut bytes = Vec::new();
            handle
                .serialize_with_mode(&mut bytes, compress)
                .expect("serialize pending GT");
            assert!(
                SLOTS.with_borrow(|slots| slots.iter().all(|slot| matches!(slot, Slot::Ready(_))))
            );
            let mut host_bytes = Vec::new();
            expected
                .serialize_with_mode(&mut host_bytes, compress)
                .expect("serialize host GT");
            assert_eq!(bytes, host_bytes);
            assert_eq!(handle.serialized_size(compress), bytes.len());
            for validate in [Validate::Yes, Validate::No] {
                let decoded = DeviceGT::deserialize_with_mode(bytes.as_slice(), compress, validate)
                    .expect("deserialize GT");
                assert_eq!(decoded.value(), expected);
            }
            assert_eq!(
                companion.value(),
                ArkGT(Bn254::multi_pairing(ps[3..8].to_vec(), qs[4..9].to_vec()))
            );
            for truncated in [0, 1, bytes.len() - 1] {
                assert!(DeviceGT::deserialize_with_mode(
                    &bytes[..truncated],
                    compress,
                    Validate::Yes
                )
                .is_err());
            }
            assert!(DeviceGT::deserialize_with_mode(
                vec![0xff; bytes.len()].as_slice(),
                compress,
                Validate::Yes
            )
            .is_err());
            let mut short_writer = [0u8; 1];
            assert!(handle
                .serialize_with_mode(&mut short_writer[..], compress)
                .is_err());
        }
        assert!(!arena::poisoned());
        reset();
        drop(guard);
    }

    #[test]
    fn failed_batches_poison_and_preserve_other_pair_counts() {
        if shared_context().is_none() {
            return;
        }
        let guard = arena::open(8, 8).expect("arena");
        reset();
        let (ps, qs) = points(8);
        let _ = store_all(&ps);
        let _ = store_frozen_g2(&qs);
        let bad = DeviceGT::enqueue(8, 0, 2);
        let same_count = DeviceGT::enqueue(1, 2, 2);
        let good = DeviceGT::enqueue(1, 2, 1);
        let expected = ArkGT(Bn254::pairing(ps[1], qs[2]));
        assert_eq!(good.value(), expected);
        assert!(arena::poisoned());
        assert!(SLOTS.with_borrow(|slots| matches!(slots[bad.index as usize], Slot::Pending(_))));
        assert!(
            SLOTS.with_borrow(|slots| matches!(slots[same_count.index as usize], Slot::Pending(_)))
        );
        assert_eq!(good.value(), expected);
        assert!(final_exponentiations(&[], 0).is_empty());
        assert_eq!(final_exponentiations(&[], 1), vec![None]);
        assert_eq!(final_exponentiations(&[0; FQ12_LIMBS], 1), vec![None]);
        reset();
        drop(guard);
        let guard = arena::open(1, 1).expect("fresh arena");
        assert!(!arena::poisoned());
        let pending = DeviceGT::enqueue(0, 0, 1);
        drop(guard);
        let _ = pending.value();
        assert!(arena::poisoned());
        reset();
    }

    #[test]
    fn independent_thread_queues_preserve_host_values() {
        reset();
        let a = ArkGT(Bn254::pairing(
            G1Projective::generator(),
            G2Projective::generator(),
        ));
        let b = ArkGT(Bn254::pairing(
            G1Projective::generator() * Fr::from(3u64),
            G2Projective::generator(),
        ));
        let parent = DeviceGT::from(a);
        let worker = std::thread::spawn(move || {
            reset();
            let own = DeviceGT::from(b);
            assert_eq!(own.value(), b);
            reset();
            b
        });
        assert_eq!(parent.value(), a);
        assert_eq!(worker.join().expect("worker"), b);
        assert_eq!(parent.value(), a);
        reset();
    }
}
