//! Decode Montgomery-limb pairing results into Arkworks extension-field values.

use ark_bn254::{Fq, Fq12, Fq2, Fq6};
use ark_ff::BigInt;

use crate::cuda::common::msm::FQ_LIMBS;

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
