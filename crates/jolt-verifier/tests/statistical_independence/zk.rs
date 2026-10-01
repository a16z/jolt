//! Statistical independence of ZK proofs, over the argument string.
//!
//! Every message after the proof header is a hiding commitment, a masked
//! Dory message, or a BlindFold message over a randomized folded instance.
//! Two honest proofs of one statement must therefore differ in every such
//! message, the messages' low bytes must look uniform, and no commitment or
//! masked message may repeat within a proof (a repeat means reused
//! randomness).

#![expect(
    clippy::cast_precision_loss,
    reason = "chi-squared statistics are computed in floating point"
)]

use crate::support::narg::{Message, Region, TracedCase};
use crate::support::verifier_fixtures::{fresh_zk_muldiv_case, zk_muldiv_case};
use std::collections::BTreeMap;

const NUM_BUCKETS: usize = 16;
/// Chi-squared critical value for 15 degrees of freedom at p ~ 1.2e-5.
const CHI2_CRITICAL: f64 = 50.0;

#[test]
fn zk_proofs_are_independent_over_hiding_messages() {
    let cached = zk_muldiv_case();
    let fresh = fresh_zk_muldiv_case();
    let messages = cached.honest_trace().messages();
    assert_eq!(
        messages,
        fresh.honest_trace().messages(),
        "proofs of one statement share one message layout"
    );
    let (header, hiding): (Vec<_>, Vec<_>) = messages
        .into_iter()
        .partition(|message| message.region == Region::Preamble);
    let hiding: Vec<Message> = hiding
        .into_iter()
        .filter(|message| !message.range.is_empty())
        .collect();

    for message in &header {
        assert_eq!(
            cached.proof.narg[message.range.clone()],
            fresh.proof.narg[message.range.clone()],
            "the header is fixed by the statement"
        );
    }
    let repeated: Vec<_> = hiding
        .iter()
        .filter(|m| cached.proof.narg[m.range.clone()] == fresh.proof.narg[m.range.clone()])
        .collect();
    assert!(
        repeated.is_empty(),
        "hiding messages identical across fresh proofs: {repeated:?}"
    );

    let cached_buckets = low_byte_buckets(&cached.proof.narg, &hiding);
    let fresh_buckets = low_byte_buckets(&fresh.proof.narg, &hiding);
    assert!(
        hiding.len() >= NUM_BUCKETS * 4,
        "too few hiding messages ({}) for a bucketed test",
        hiding.len()
    );
    for (label, buckets) in [("cached", &cached_buckets), ("fresh", &fresh_buckets)] {
        let chi2 = uniform_chi_squared(buckets);
        assert!(
            chi2 < CHI2_CRITICAL,
            "{label} proof: low-byte chi2 {chi2:.2} >= {CHI2_CRITICAL} over {buckets:?}"
        );
    }
    let chi2 = two_sample_chi_squared(&cached_buckets, &fresh_buckets);
    assert!(
        chi2 < CHI2_CRITICAL,
        "cached and fresh low-byte distributions differ: chi2 {chi2:.2}"
    );

    // BlindFold is exempt: it reveals the folded witness, and each folded
    // evaluation output and blinding is sent once as a scalar and again as
    // the dedicated row's opened coordinate, which the verifier checks equal.
    for (label, narg) in [("cached", &cached.proof.narg), ("fresh", &fresh.proof.narg)] {
        let mut first_seen = BTreeMap::new();
        let repeats: Vec<_> = hiding
            .iter()
            .filter(|message| message.region != Region::BlindFold)
            .filter_map(|message| {
                let range = &message.range;
                let prefix = &narg[range.start..range.start + 8.min(range.len())];
                first_seen
                    .insert(prefix, message)
                    .map(|earlier| (earlier, message))
            })
            .collect();
        assert!(
            repeats.is_empty(),
            "{label} proof repeats a hiding message's leading bytes: {repeats:?}"
        );
    }
}

/// Buckets each message's first byte, the low byte of its first canonical
/// coordinate or scalar, by its low nibble.
fn low_byte_buckets(narg: &[u8], messages: &[Message]) -> [usize; NUM_BUCKETS] {
    let mut buckets = [0; NUM_BUCKETS];
    for message in messages {
        buckets[usize::from(narg[message.range.start]) % NUM_BUCKETS] += 1;
    }
    buckets
}

fn uniform_chi_squared(buckets: &[usize; NUM_BUCKETS]) -> f64 {
    let expected = buckets.iter().sum::<usize>() as f64 / NUM_BUCKETS as f64;
    buckets
        .iter()
        .map(|&observed| (observed as f64 - expected).powi(2) / expected)
        .sum()
}

fn two_sample_chi_squared(a: &[usize; NUM_BUCKETS], b: &[usize; NUM_BUCKETS]) -> f64 {
    let n_a = a.iter().sum::<usize>() as f64;
    let n_b = b.iter().sum::<usize>() as f64;
    let n = n_a + n_b;
    a.iter()
        .zip(b)
        .filter(|(&x, &y)| x + y > 0)
        .map(|(&x, &y)| {
            let pooled = (x + y) as f64;
            let expected_a = pooled * n_a / n;
            let expected_b = pooled * n_b / n;
            (x as f64 - expected_a).powi(2) / expected_a
                + (y as f64 - expected_b).powi(2) / expected_b
        })
        .sum()
}
