//! sig-recovery guest program
//!
//! This crate contains the provable functions that run inside the Jolt zkVM.
//! The main function is `verify_txs` which recovers signer addresses from
//! serialized Ethereum transactions using parallel recovery via rayon.

use alloy_eips::eip2718::Decodable2718;
use reth_ethereum_primitives::TransactionSigned;
use reth_primitives_traits::transaction::recover::recover_signers;
use serde::{Deserialize, Serialize};

/// Result of transaction verification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    /// Number of transactions processed
    pub tx_count: u32,
    /// Number of successfully recovered signers
    pub recovered_count: u32,
    /// Recovered signer addresses (in order)
    pub signers: Vec<[u8; 20]>,
}

/// Verify transactions by recovering their signer addresses.
///
/// This function is marked with `#[jolt::provable]` to make it provable
/// inside the Jolt zkVM. It uses rayon-based parallel signature recovery
/// via reth's `recover_signers` function.
///
/// # Arguments
/// * `txs_bytes` - Postcard-serialized vector of RLP-encoded transactions
///
/// # Returns
/// * `VerificationResult` containing the recovered signers
#[jolt::provable(
    max_input_size = 1048576,   // 1MB input
    max_output_size = 65536,    // 64KB output
    heap_size = 33554432,     // 32MB memory
    stack_size = 131072,        // 128KB stack
    max_trace_length = 33554432 // 32M trace length
)]
pub fn verify_txs(txs_bytes: &[u8]) -> VerificationResult {
    jolt::start_cycle_tracking("deserialize");

    let rlp_txs: Vec<Vec<u8>> = match postcard::from_bytes(txs_bytes) {
        Ok(txs) => txs,
        Err(_) => {
            return VerificationResult {
                tx_count: 0,
                recovered_count: 0,
                signers: vec![],
            };
        }
    };

    let tx_count = rlp_txs.len() as u32;
    let mut signers = vec![[0u8; 20]; rlp_txs.len()];

    let txs: Vec<TransactionSigned> = rlp_txs
        .iter()
        .filter_map(|rlp| TransactionSigned::decode_2718(&mut rlp.as_slice()).ok())
        .collect();

    jolt::end_cycle_tracking("deserialize");

    let mut recovered_count = 0u32;
    if !txs.is_empty() {
        jolt::start_cycle_tracking("recover_signers");

        if let Ok(addresses) = recover_signers(&txs) {
            for (i, addr) in addresses.into_iter().enumerate() {
                signers[i] = addr.0 .0;
                recovered_count += 1;
            }
        }

        jolt::end_cycle_tracking("recover_signers");
    }

    VerificationResult {
        tx_count,
        recovered_count,
        signers,
    }
}
