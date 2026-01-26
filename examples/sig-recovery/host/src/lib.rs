//! sig-recovery host library
//!
//! Utility functions for generating and serializing test transactions.

use alloy_consensus::SignableTransaction;
use alloy_eips::eip2718::Encodable2718;
use alloy_primitives::{Address, U256};
use reth_ethereum_primitives::{Transaction, TransactionSigned};
use secp256k1::{Keypair, Message, Secp256k1, SecretKey};

/// Generate a batch of signed transactions for testing
pub fn generate_test_transactions(count: usize) -> Vec<TransactionSigned> {
    let secp = Secp256k1::new();
    let mut rng = rand::thread_rng();

    (0..count)
        .map(|i| {
            let keypair = Keypair::new(&secp, &mut rng);
            let secret_key = SecretKey::from_keypair(&keypair);

            let tx = alloy_consensus::TxLegacy {
                nonce: i as u64,
                gas_price: 20_000_000_000u128,
                gas_limit: 21000,
                to: alloy_primitives::TxKind::Call(Address::ZERO),
                value: U256::from(1000000000000000000u64),
                input: Default::default(),
                chain_id: Some(1),
            };

            let signature_hash = tx.signature_hash();
            let msg = Message::from_digest(signature_hash.0);
            let sig = secp.sign_ecdsa_recoverable(&msg, &secret_key);
            let (recovery_id, sig_bytes) = sig.serialize_compact();

            let signature = alloy_primitives::Signature::new(
                U256::from_be_slice(&sig_bytes[..32]),
                U256::from_be_slice(&sig_bytes[32..]),
                i32::from(recovery_id) % 2 != 0,
            );

            TransactionSigned::new_unhashed(Transaction::Legacy(tx), signature)
        })
        .collect()
}

/// Serialize transactions to postcard format (RLP-encoded bytes)
pub fn serialize_transactions(txs: &[TransactionSigned]) -> Vec<u8> {
    let rlp_txs: Vec<Vec<u8>> = txs
        .iter()
        .map(|tx| {
            let mut buf = Vec::new();
            tx.encode_2718(&mut buf);
            buf
        })
        .collect();

    postcard::to_stdvec(&rlp_txs).expect("Failed to serialize transactions")
}
