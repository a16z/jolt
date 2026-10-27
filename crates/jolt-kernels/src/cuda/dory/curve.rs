//! Dory curve and transcript adapters for resident group handles.
//!
//! Contiguous multi-pairing inputs become deferred device requests; individual
//! pairings and other input layouts use the CPU curve. Transcript operations
//! delegate to `JoltToDoryTranscript`, resolving handles through their ordinary
//! group serialization. The returned transparent proof is converted back to
//! the shared host proof type before the arena closes.

use ark_bn254::Fq;
use ark_bn254::Fq12;
use ark_bn254::Fq2;
use ark_bn254::Fq6;
use ark_ff::BigInt;
use dory::backends::arkworks::{ArkFr, ArkG1, ArkG2, ArkGT, BN254};
use dory::messages::FirstReduceMessage;
use dory::messages::ScalarProductMessage;
use dory::messages::SecondReduceMessage;
use dory::messages::VMVMessage;
use dory::primitives::arithmetic::Group;
use dory::primitives::arithmetic::PairingCurve;
use dory::primitives::transcript::Transcript as DoryTranscript;
use dory::primitives::DorySerialize;
use dory::proof::DoryProof;
use jolt_dory::JoltToDoryTranscript;
use jolt_field::Fr;
use jolt_transcript::Transcript;

use super::gt::DeviceGT;
use super::handle::{load_all, load_all_g2, span, span_g2, DeviceG1, DeviceG2};
use crate::cuda::common::msm::FQ_LIMBS;

#[derive(Clone)]
pub(super) struct CudaBN254;

fn ark(points: &[DeviceG1]) -> Vec<ArkG1> {
    load_all(points).into_iter().map(ArkG1).collect()
}

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

pub(super) struct CudaDoryTranscript<'a, T: Transcript<Challenge = Fr>> {
    inner: JoltToDoryTranscript<'a, T>,
}

impl<'a, T: Transcript<Challenge = Fr>> CudaDoryTranscript<'a, T> {
    pub(super) fn new(transcript: &'a mut T) -> Self {
        Self {
            inner: JoltToDoryTranscript::new(transcript),
        }
    }
}

impl<T: Transcript<Challenge = Fr>> DoryTranscript for CudaDoryTranscript<'_, T> {
    type Curve = CudaBN254;

    fn append_bytes(&mut self, label: &[u8], bytes: &[u8]) {
        self.inner.append_bytes(label, bytes);
    }

    fn append_field(&mut self, label: &[u8], x: &ArkFr) {
        self.inner.append_field(label, x);
    }

    fn append_group<G: Group>(&mut self, label: &[u8], g: &G) {
        self.inner.append_group(label, g);
    }

    fn append_serde<S: DorySerialize>(&mut self, label: &[u8], s: &S) {
        self.inner.append_serde(label, s);
    }

    fn challenge_scalar(&mut self, label: &[u8]) -> ArkFr {
        self.inner.challenge_scalar(label)
    }

    fn reset(&mut self, domain_label: &[u8]) {
        self.inner.reset(domain_label);
    }
}

type ResidentProof = DoryProof<DeviceG1, DeviceG2, DeviceGT>;
type HostProof = DoryProof<ArkG1, ArkG2, ArkGT>;

pub(super) fn rebind_proof(proof: ResidentProof) -> Result<HostProof, &'static str> {
    if proof.sigma1_proof.is_some()
        || proof.sigma2_proof.is_some()
        || proof.scalar_product_proof.is_some()
    {
        return Err("CudaDoryScheme::open is the transparent path; a sigma proof cannot appear");
    }
    let g1 = |handle: DeviceG1| ArkG1(handle.load());
    let g2 = |handle: DeviceG2| ArkG2(handle.load());
    let gt = DeviceGT::value;
    Ok(HostProof {
        vmv_message: VMVMessage {
            c: gt(proof.vmv_message.c),
            d2: gt(proof.vmv_message.d2),
            e1: g1(proof.vmv_message.e1),
        },
        first_messages: proof
            .first_messages
            .into_iter()
            .map(|message| FirstReduceMessage {
                d1_left: gt(message.d1_left),
                d1_right: gt(message.d1_right),
                d2_left: gt(message.d2_left),
                d2_right: gt(message.d2_right),
                e1_beta: g1(message.e1_beta),
                e2_beta: g2(message.e2_beta),
            })
            .collect(),
        second_messages: proof
            .second_messages
            .into_iter()
            .map(|message| SecondReduceMessage {
                c_plus: gt(message.c_plus),
                c_minus: gt(message.c_minus),
                e1_plus: g1(message.e1_plus),
                e1_minus: g1(message.e1_minus),
                e2_plus: g2(message.e2_plus),
                e2_minus: g2(message.e2_minus),
            })
            .collect(),
        final_message: proof.final_message.map(|message| ScalarProductMessage {
            e1: g1(message.e1),
            e2: g2(message.e2),
        }),
        nu: proof.nu,
        sigma: proof.sigma,
        e2: proof.e2.map(g2),
        y_com: proof.y_com.map(g1),
        sigma1_proof: None,
        sigma2_proof: None,
        scalar_product_proof: None,
    })
}

#[cfg(test)]
#[expect(clippy::expect_used, reason = "tests fail loudly on setup errors")]
mod tests {
    use std::panic::AssertUnwindSafe;

    use ark_bn254::{Bn254, Fr as ArkBn254Fr, G1Projective, G2Projective};
    use ark_ec::{pairing::Pairing, AffineRepr, CurveGroup};
    use ark_ff::{Field, UniformRand};
    use ark_serialize::{
        CanonicalDeserialize, CanonicalSerialize, Compress as ArkCompress, Validate as ArkValidate,
        Write,
    };
    use dory::messages::{ScalarProductProof, Sigma1Proof, Sigma2Proof};
    use dory::primitives::{Compress, SerializationError};
    use jolt_dory::DoryProof as JoltDoryProof;
    use jolt_transcript::{Blake2bTranscript, LegacyBlake2bTranscript};
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

    fn host_transcript_operations<T: Transcript<Challenge = Fr>>() {
        let mut actual = T::new(b"cuda-dory-transcript");
        let mut expected = T::new(b"cuda-dory-transcript");
        actual.append_bytes(b"prior Jolt state");
        expected.append_bytes(b"prior Jolt state");
        for payload in [b"".as_slice(), b"one", b"some longer transcript payload"] {
            CudaDoryTranscript::new(&mut actual).append_bytes(b"cuda-label", payload);
            JoltToDoryTranscript::new(&mut expected).append_bytes(b"host-label", payload);
            assert_eq!(actual.state(), expected.state());
        }
        let mut rng = ChaCha20Rng::seed_from_u64(21_127);
        for scalar in [
            ArkBn254Fr::from(0u64),
            ArkBn254Fr::from(1u64),
            -ArkBn254Fr::from(1u64),
            ArkBn254Fr::rand(&mut rng),
        ] {
            let value = ArkFr(scalar);
            CudaDoryTranscript::new(&mut actual).append_field(b"field", &value);
            JoltToDoryTranscript::new(&mut expected).append_field(b"field", &value);
            assert_eq!(actual.state(), expected.state());
            CudaDoryTranscript::new(&mut actual).append_serde(b"scalar", &value);
            JoltToDoryTranscript::new(&mut expected).append_serde(b"scalar", &value);
            assert_eq!(actual.state(), expected.state());
            for _ in 0..3 {
                let before = actual.state();
                assert_eq!(
                    CudaDoryTranscript::new(&mut actual).challenge_scalar(b"challenge"),
                    JoltToDoryTranscript::new(&mut expected).challenge_scalar(b"challenge"),
                );
                assert_eq!(actual.state(), expected.state());
                assert_ne!(actual.state(), before);
            }
        }
    }

    #[test]
    fn transcript_host_operations_match_cpu_without_an_arena() {
        host_transcript_operations::<Blake2bTranscript>();
        host_transcript_operations::<LegacyBlake2bTranscript>();
    }

    fn resident_transcript_messages<T: Transcript<Challenge = Fr>>() {
        let guard = arena::open(17, 17).expect("arena");
        gt::reset();
        let (ps, qs) = points(17);
        let hp = store_all(&ps);
        let hq = store_frozen_g2(&qs);
        let mut actual = T::new(b"cuda-dory-resident");
        let mut expected = T::new(b"cuda-dory-resident");
        for index in [0, 1, 3] {
            CudaDoryTranscript::new(&mut actual).append_group(b"g1", &hp[index]);
            JoltToDoryTranscript::new(&mut expected).append_group(b"g1", &ArkG1(ps[index]));
            assert_eq!(actual.state(), expected.state());
            CudaDoryTranscript::new(&mut actual).append_group(b"g2", &hq[index]);
            JoltToDoryTranscript::new(&mut expected).append_group(b"g2", &ArkG2(qs[index]));
            assert_eq!(actual.state(), expected.state());
        }
        let target = CudaBN254::multi_pair(&hp[1..6], &hq[2..7]);
        let host_target = ArkGT(Bn254::multi_pairing(ps[1..6].to_vec(), qs[2..7].to_vec()));
        CudaDoryTranscript::new(&mut actual).append_group(b"gt", &target);
        JoltToDoryTranscript::new(&mut expected).append_group(b"gt", &host_target);
        assert_eq!(actual.state(), expected.state());
        let targets = vec![
            CudaBN254::multi_pair(&hp[1..6], &hq[2..7]),
            CudaBN254::multi_pair(&hp[3..8], &hq[4..9]),
        ];
        let host_targets = vec![
            host_target,
            ArkGT(Bn254::multi_pairing(ps[3..8].to_vec(), qs[4..9].to_vec())),
        ];
        CudaDoryTranscript::new(&mut actual).append_serde(b"targets", &targets);
        JoltToDoryTranscript::new(&mut expected).append_serde(b"targets", &host_targets);
        assert_eq!(actual.state(), expected.state());
        drop(guard);
        for (value, host_value) in targets.into_iter().zip(host_targets) {
            CudaDoryTranscript::new(&mut actual).append_group(b"resolved", &value);
            JoltToDoryTranscript::new(&mut expected).append_group(b"resolved", &host_value);
            assert_eq!(
                CudaDoryTranscript::new(&mut actual).challenge_scalar(b"challenge"),
                JoltToDoryTranscript::new(&mut expected).challenge_scalar(b"challenge"),
            );
            assert_eq!(actual.state(), expected.state());
        }
        assert!(!arena::poisoned());
        gt::reset();
    }

    #[test]
    fn transcript_resident_messages_match_cpu() {
        if shared_context().is_none() {
            return;
        }
        resident_transcript_messages::<Blake2bTranscript>();
        resident_transcript_messages::<LegacyBlake2bTranscript>();
    }

    #[test]
    fn transcript_reset_is_rejected_without_changing_state() {
        let mut actual = LegacyBlake2bTranscript::new(b"cuda-dory-reset");
        let mut expected = LegacyBlake2bTranscript::new(b"cuda-dory-reset");
        actual.append_bytes(b"prior state");
        expected.append_bytes(b"prior state");
        let before = actual.state();
        assert!(std::panic::catch_unwind(AssertUnwindSafe(|| {
            CudaDoryTranscript::new(&mut actual).reset(b"new domain");
        }))
        .is_err());
        assert!(std::panic::catch_unwind(AssertUnwindSafe(|| {
            JoltToDoryTranscript::new(&mut expected).reset(b"new domain");
        }))
        .is_err());
        assert_eq!(actual.state(), before);
        assert_eq!(expected.state(), before);
        assert_eq!(
            CudaDoryTranscript::new(&mut actual).challenge_scalar(b"continued"),
            JoltToDoryTranscript::new(&mut expected).challenge_scalar(b"continued"),
        );
    }

    #[test]
    fn transcript_serialization_failure_does_not_absorb_partial_data() {
        struct Unserializable;
        impl DorySerialize for Unserializable {
            fn serialize_with_mode<W: Write>(
                &self,
                mut writer: W,
                _compress: Compress,
            ) -> Result<(), SerializationError> {
                writer.write_all(&[1, 2, 3])?;
                Err(SerializationError::UnexpectedData)
            }

            fn serialized_size(&self, _compress: Compress) -> usize {
                3
            }
        }
        let mut actual = LegacyBlake2bTranscript::new(b"cuda-dory-error");
        let mut expected = LegacyBlake2bTranscript::new(b"cuda-dory-error");
        let before = actual.state();
        assert!(std::panic::catch_unwind(AssertUnwindSafe(|| {
            CudaDoryTranscript::new(&mut actual).append_serde(b"failing", &Unserializable);
        }))
        .is_err());
        assert!(std::panic::catch_unwind(AssertUnwindSafe(|| {
            JoltToDoryTranscript::new(&mut expected).append_serde(b"failing", &Unserializable);
        }))
        .is_err());
        assert_eq!(actual.state(), before);
        assert_eq!(expected.state(), before);
    }

    fn proof_fixture<G1: Group + Copy, G2: Copy, GT: Copy>(
        ps: &[G1],
        qs: &[G2],
        targets: &[GT],
    ) -> DoryProof<G1, G2, GT> {
        DoryProof {
            vmv_message: VMVMessage {
                c: targets[0],
                d2: targets[1],
                e1: ps[0],
            },
            first_messages: (0..2)
                .map(|i| FirstReduceMessage {
                    d1_left: targets[2 + 4 * i],
                    d1_right: targets[3 + 4 * i],
                    d2_left: targets[4 + 4 * i],
                    d2_right: targets[5 + 4 * i],
                    e1_beta: ps[1 + i],
                    e2_beta: qs[i],
                })
                .collect(),
            second_messages: (0..2)
                .map(|i| SecondReduceMessage {
                    c_plus: targets[10 + 2 * i],
                    c_minus: targets[11 + 2 * i],
                    e1_plus: ps[3 + 2 * i],
                    e1_minus: ps[4 + 2 * i],
                    e2_plus: qs[3 + 2 * i],
                    e2_minus: qs[4 + 2 * i],
                })
                .collect(),
            final_message: Some(ScalarProductMessage {
                e1: ps[7],
                e2: qs[7],
            }),
            nu: 2,
            sigma: 3,
            e2: None,
            y_com: None,
            sigma1_proof: None,
            sigma2_proof: None,
            scalar_product_proof: None,
        }
    }

    #[test]
    fn proof_rebinding_preserves_messages_and_wire_bytes() {
        if shared_context().is_none() {
            return;
        }
        let (ps, qs) = points(14);
        let host_ps: Vec<_> = ps.iter().copied().map(ArkG1).collect();
        let host_qs: Vec<_> = qs.iter().copied().map(ArkG2).collect();
        let host_targets: Vec<_> = ps
            .iter()
            .zip(&qs)
            .map(|(p, q)| ArkGT(Bn254::pairing(*p, *q)))
            .collect();
        for rounds in [0, 2] {
            for parity in [0, 1] {
                let guard = arena::open(14, 14).expect("arena");
                gt::reset();
                let hp = store_all(&ps);
                let hq = store_frozen_g2(&qs);
                let targets: Vec<_> = (0..14)
                    .map(|i| {
                        if i % 2 == parity {
                            CudaBN254::multi_pair(&hp[i..=i], &hq[i..=i])
                        } else {
                            DeviceGT::from(host_targets[i])
                        }
                    })
                    .collect();
                let mut resident = proof_fixture(&hp, &hq, &targets);
                let mut expected = proof_fixture(&host_ps, &host_qs, &host_targets);
                resident.first_messages.truncate(rounds);
                resident.second_messages.truncate(rounds);
                resident.nu = rounds;
                expected.first_messages.truncate(rounds);
                expected.second_messages.truncate(rounds);
                expected.nu = rounds;
                let actual = rebind_proof(resident).expect("convert transparent proof");
                assert!(!arena::poisoned());
                drop(guard);
                gt::reset();
                assert_eq!(actual, expected);
                assert!(actual.mode().is_ok());
                for compress in [ArkCompress::Yes, ArkCompress::No] {
                    let mut actual_bytes = Vec::new();
                    let mut expected_bytes = Vec::new();
                    actual
                        .serialize_with_mode(&mut actual_bytes, compress)
                        .expect("serialize converted proof");
                    expected
                        .serialize_with_mode(&mut expected_bytes, compress)
                        .expect("serialize host fixture");
                    assert_eq!(actual_bytes, expected_bytes);
                    assert_eq!(actual.serialized_size(compress), actual_bytes.len());
                    let decoded = HostProof::deserialize_with_mode(
                        actual_bytes.as_slice(),
                        compress,
                        ArkValidate::Yes,
                    )
                    .expect("deserialize converted proof");
                    assert_eq!(decoded, expected);
                }
                let actual = JoltDoryProof(actual);
                let bytes = serde_json::to_vec(&actual).expect("serialize shared proof type");
                let decoded: JoltDoryProof =
                    serde_json::from_slice(&bytes).expect("deserialize shared proof type");
                assert_eq!(decoded, JoltDoryProof(expected));
            }
        }
    }

    #[test]
    fn proof_rebinding_rejects_sigma_proofs_before_loading_handles() {
        if shared_context().is_none() {
            return;
        }
        let guard = arena::open(1, 1).expect("arena");
        gt::reset();
        let p = DeviceG1::store(&G1Projective::default());
        let q = DeviceG2::store(&G2Projective::default());
        let target = DeviceGT::identity();
        let base = proof_fixture(&[p; 8], &[q; 8], &[target; 14]);
        let scalar = ArkFr(ArkBn254Fr::from(3u64));
        let sigma1 = Sigma1Proof {
            a1: q,
            a2: p,
            z1: scalar,
            z2: scalar,
            z3: scalar,
        };
        let sigma2 = Sigma2Proof {
            a: target,
            z1: scalar,
            z2: scalar,
        };
        let scalar_product = ScalarProductProof {
            p1: target,
            p2: target,
            q: target,
            r: target,
            e1: p,
            e2: q,
            r1: scalar,
            r2: scalar,
            r3: scalar,
        };
        drop(guard);
        gt::reset();
        for mask in 1..8 {
            let mut proof = base.clone();
            proof.sigma1_proof = (mask & 1 != 0).then(|| sigma1.clone());
            proof.sigma2_proof = (mask & 2 != 0).then(|| sigma2.clone());
            proof.scalar_product_proof = (mask & 4 != 0).then(|| scalar_product.clone());
            assert_eq!(
                rebind_proof(proof).expect_err("sigma proofs are unsupported"),
                "CudaDoryScheme::open is the transparent path; a sigma proof cannot appear"
            );
            assert!(!arena::poisoned());
        }
    }
}
