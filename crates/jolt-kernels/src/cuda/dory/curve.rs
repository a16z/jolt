use dory::backends::arkworks::{ArkFr, ArkG1, ArkG2, ArkGT, BN254};
use dory::primitives::arithmetic::PairingCurve;
use dory::primitives::transcript::Transcript as DoryTranscript;
use dory::primitives::DorySerialize;
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

fn fq(words: &[u64]) -> ark_bn254::Fq {
    ark_bn254::Fq::new_unchecked(ark_ff::BigInt([words[0], words[1], words[2], words[3]]))
}

fn fq2(words: &[u64]) -> ark_bn254::Fq2 {
    ark_bn254::Fq2::new(fq(&words[..FQ_LIMBS]), fq(&words[FQ_LIMBS..]))
}

pub(super) fn fq12(words: &[u64]) -> ark_bn254::Fq12 {
    let fq6 = |chunk: &[u64]| {
        ark_bn254::Fq6::new(
            fq2(&chunk[..2 * FQ_LIMBS]),
            fq2(&chunk[2 * FQ_LIMBS..4 * FQ_LIMBS]),
            fq2(&chunk[4 * FQ_LIMBS..]),
        )
    };
    ark_bn254::Fq12::new(fq6(&words[..6 * FQ_LIMBS]), fq6(&words[6 * FQ_LIMBS..]))
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

    fn append_group<G: dory::primitives::arithmetic::Group>(&mut self, label: &[u8], g: &G) {
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

type ResidentProof = dory::proof::DoryProof<DeviceG1, DeviceG2, DeviceGT>;
type HostProof = dory::proof::DoryProof<ArkG1, ArkG2, ArkGT>;

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
        vmv_message: dory::messages::VMVMessage {
            c: gt(proof.vmv_message.c),
            d2: gt(proof.vmv_message.d2),
            e1: g1(proof.vmv_message.e1),
        },
        first_messages: proof
            .first_messages
            .into_iter()
            .map(|message| dory::messages::FirstReduceMessage {
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
            .map(|message| dory::messages::SecondReduceMessage {
                c_plus: gt(message.c_plus),
                c_minus: gt(message.c_minus),
                e1_plus: g1(message.e1_plus),
                e1_minus: g1(message.e1_minus),
                e2_plus: g2(message.e2_plus),
                e2_minus: g2(message.e2_minus),
            })
            .collect(),
        final_message: proof
            .final_message
            .map(|message| dory::messages::ScalarProductMessage {
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
