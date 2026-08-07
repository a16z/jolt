//! Review evidence for PR #1750: do the acceptance-suite guests actually
//! EXECUTE the fused word-arithmetic kinds? Counts fused rows in the traced
//! muldiv-guest (the primary e2e/byte-diff guest).

#![cfg(feature = "host")]
#![expect(clippy::expect_used)]

use jolt_prover_legacy::host;
use std::collections::BTreeMap;

fn census_for(guest: &str, inputs: &[u8], untrusted: &[u8], trusted: &[u8]) {
    let mut program = host::Program::new(guest);
    let (_, trace, _, _) = program.trace(inputs, untrusted, trusted);

    let mut census: BTreeMap<&'static str, usize> = BTreeMap::new();
    for cycle in &trace {
        let name = format!("{cycle:?}");
        for kind in ["ADDW", "ADDIW", "SUBW", "MULW", "SLLIW", "SLLW", "Pow2W"] {
            // Debug names are unambiguous prefixes like `ADDIW(..)`;
            // guard against substring collisions (e.g. ADDW inside ADDIW).
            if name.starts_with(&format!("{kind}(")) {
                *census.entry(kind).or_default() += 1;
            }
        }
    }
    let mut all_kinds: BTreeMap<String, usize> = BTreeMap::new();
    for cycle in &trace {
        let name = format!("{cycle:?}");
        let kind = name.split(['(', ' ']).next().unwrap_or("?").to_string();
        *all_kinds.entry(kind).or_default() += 1;
    }
    println!("{guest} trace length: {}", trace.len());
    println!("{guest} fused word-op census: {census:?}");
    println!("{guest} full kind census: {all_kinds:?}");
    // Evidence only; do not fail the build on guest codegen drift.
}

#[test]
fn muldiv_guest_word_op_census() {
    let inputs = postcard::to_stdvec(&[9u32, 5u32, 3u32]).expect("serialize inputs");
    census_for("muldiv-guest", &inputs, &[], &[]);
}

#[test]
fn fibonacci_guest_word_op_census() {
    let inputs = postcard::to_stdvec(&50u32).expect("serialize inputs");
    census_for("fibonacci-guest", &inputs, &[], &[]);
}

#[test]
fn advice_consumer_guest_word_op_census() {
    let inputs = postcard::to_stdvec(&12u64).expect("serialize inputs");
    let untrusted = postcard::to_stdvec(&5u64).expect("serialize untrusted advice");
    let trusted = postcard::to_stdvec(&7u64).expect("serialize trusted advice");
    census_for("advice-consumer-guest", &inputs, &untrusted, &trusted);
}

#[test]
fn sha2_chain_guest_word_op_census() {
    // Anchor the sha2 inline's inventory registration (dead-stripped otherwise).
    extern crate jolt_inlines_sha2;
    let inputs = [
        postcard::to_stdvec(&[5u8; 32]).expect("serialize seed"),
        postcard::to_stdvec(&30u32).expect("serialize iterations"),
    ]
    .concat();
    census_for("sha2-chain-guest", &inputs, &[], &[]);
}
