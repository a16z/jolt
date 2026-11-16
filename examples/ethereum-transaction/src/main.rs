//! Builds the guest's witness from a capture written by `fetch.py`: keeps the
//! trie nodes and contract codes the transaction resolves, and refuses a
//! capture whose receipt the guest's logic does not reproduce.

use std::cell::RefCell;
use std::{env, fs};

use alloy_consensus::TxEnvelope;
use alloy_eips::eip2718::Decodable2718 as _;
use alloy_primitives::map::{B256Map, B256Set};
use alloy_primitives::{hex, keccak256, Bytes, B256};
use guest::{execute, run, Preimages, Witness};
use serde::Deserialize;

#[derive(Deserialize)]
struct Capture {
    header: Bytes,
    parent_header: Bytes,
    transaction: Bytes,
    receipt: Bytes,
    state: Vec<Bytes>,
    codes: Vec<Bytes>,
}

/// Records which preimages a run resolves.
struct Recorder {
    preimages: B256Map<Bytes>,
    resolved: RefCell<B256Set>,
}

impl Preimages for Recorder {
    fn preimage(&self, digest: &B256) -> Option<&[u8]> {
        let preimage = self.preimages.get(digest)?;
        let _ = self.resolved.borrow_mut().insert(*digest);
        Some(preimage)
    }
}

fn main() {
    let [_, capture, fixture] = &env::args().collect::<Vec<_>>()[..] else {
        panic!("usage: ethereum-transaction <capture.json> <fixture.rlp>");
    };
    let capture: Capture =
        serde_json::from_slice(&fs::read(capture).expect("read the capture")).expect("capture");

    let transaction =
        TxEnvelope::decode_2718(&mut capture.transaction.as_ref()).expect("transaction");
    let key = transaction
        .signature()
        .recover_from_prehash(&transaction.signature_hash())
        .expect("recover the signer");
    let mut witness = Witness {
        header: capture.header,
        parent_header: capture.parent_header,
        transaction: capture.transaction,
        public_key: Bytes::copy_from_slice(&key.to_encoded_point(false).as_bytes()[1..]),
        preimages: Vec::new(),
    };

    let recorder = Recorder {
        preimages: capture
            .state
            .into_iter()
            .chain(capture.codes)
            .map(|preimage| (keccak256(&preimage), preimage))
            .collect(),
        resolved: RefCell::default(),
    };
    let outcome = run(&witness, &recorder).expect("execute the transaction");
    println!(
        "success: {}, gas used: {}",
        outcome.success, outcome.gas_used
    );
    assert_eq!(
        outcome.receipt_hash,
        keccak256(&capture.receipt).0,
        "the guest's receipt differs from the chain's"
    );

    let mut resolved: Vec<_> = recorder.resolved.into_inner().into_iter().collect();
    resolved.sort_unstable();
    witness.preimages = resolved
        .iter()
        .map(|digest| recorder.preimages[digest].clone())
        .collect();
    let encoded = alloy_rlp::encode(&witness);
    assert_eq!(
        execute(&encoded),
        outcome,
        "the pruned witness changes the outcome"
    );
    fs::write(fixture, &encoded).expect("write the fixture");

    println!(
        "fixture: {} bytes, {} preimages",
        encoded.len(),
        resolved.len()
    );
    println!("block hash: {}", hex::encode_prefixed(outcome.block_hash));
    println!(
        "transaction hash: {}",
        hex::encode_prefixed(outcome.transaction_hash)
    );
    println!("sender: {}", hex::encode_prefixed(outcome.sender));
    println!(
        "receipt hash: {}",
        hex::encode_prefixed(outcome.receipt_hash)
    );
    println!(
        "post-state root: {}",
        hex::encode_prefixed(outcome.post_state_root)
    );
}
