#![cfg_attr(feature = "guest", no_std)]

extern crate alloc;

mod mpt;
mod state;

use alloc::format;
use alloc::string::String;
use alloc::vec::Vec;

use alloy_consensus::{
    Eip658Value, Header, Receipt, ReceiptEnvelope, Transaction as _, TxEnvelope, TxType,
};
use alloy_eips::eip2718::{Decodable2718 as _, Eip2718Error, Encodable2718 as _};
use alloy_eips::eip7840::BlobParams;
use alloy_primitives::map::B256Map;
use alloy_primitives::{keccak256, Address, Bytes, B256, U256};
use alloy_rlp::{Decodable as _, Error as RlpError, RlpDecodable, RlpEncodable};
use jolt_inlines_secp256k1::{ecdsa_verify, Secp256k1Fr, Secp256k1Point};
use revm::context::either::Either;
use revm::context::result::ResultAndState;
use revm::context::{BlockEnv, TxEnv};
use revm::context_interface::block::BlobExcessGasAndPrice;
use revm::database_interface::DBErrorMarker;
use revm::primitives::hardfork::SpecId;
use revm::{Context, ExecuteEvm as _, MainBuilder as _, MainContext as _};
use serde::{Deserialize, Serialize};

use crate::state::State;

/// Mainnet's second blob-parameter-only fork, 2026-01-07 01:01:11 UTC
/// (EIP-7892). [`execute`] applies Osaka rules with this fork's blob
/// parameters and rejects earlier blocks; it does not know later forks.
const BPO2_TIMESTAMP: u64 = 1_767_747_671;

const SECP256K1_ORDER: U256 = U256::from_limbs([
    0xbfd2_5e8c_d036_4141,
    0xbaae_dce6_af48_a03b,
    0xffff_ffff_ffff_fffe,
    0xffff_ffff_ffff_ffff,
]);

/// What [`execute`] needs to run one transaction as the first of its block.
#[derive(RlpEncodable, RlpDecodable)]
pub struct Witness {
    /// The block header, RLP.
    pub header: Bytes,
    /// The parent block's header, RLP. Its state root is the pre-state.
    pub parent_header: Bytes,
    /// The signed transaction, EIP-2718.
    pub transaction: Bytes,
    /// The signer's public key: big-endian `x` then `y`.
    pub public_key: Bytes,
    /// Trie nodes and contract codes, each looked up by its keccak256.
    pub preimages: Vec<Bytes>,
}

/// A source of keccak256 preimages.
pub trait Preimages {
    fn preimage(&self, digest: &B256) -> Option<&[u8]>;
}

impl Preimages for B256Map<Bytes> {
    fn preimage(&self, digest: &B256) -> Option<&[u8]> {
        self.get(digest).map(AsRef::as_ref)
    }
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Outcome {
    pub block_hash: [u8; 32],
    pub transaction_hash: [u8; 32],
    pub sender: [u8; 20],
    pub success: bool,
    pub gas_used: u64,
    /// keccak256 of the EIP-2718 receipt, as the block's receipt trie stores it.
    pub receipt_hash: [u8; 32],
    /// The state root after this transaction alone: the parent state without
    /// the block's system calls, which run before its first transaction.
    pub post_state_root: [u8; 32],
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("malformed RLP: {0}")]
    Rlp(#[from] RlpError),
    #[error("malformed transaction: {0}")]
    Transaction(#[from] Eip2718Error),
    #[error("malformed trie node")]
    Trie,
    #[error("the witness has no preimage of {0}")]
    MissingPreimage(B256),
    #[error("the parent header does not hash to the header's parent hash")]
    ParentHeader,
    #[error("the block predates the supported fork")]
    UnsupportedFork,
    #[error("the signature does not verify under the supplied public key")]
    Signature,
    #[error("BLOCKHASH of block {0}, which the witness does not carry")]
    BlockHash(u64),
    #[error("the transaction is invalid: {0}")]
    Invalid(String),
}

impl DBErrorMarker for Error {}

/// Executes the witness's transaction on the parent block's state.
///
/// Panics unless the parent header hashes to the header's parent hash, the
/// signature verifies under the supplied public key, every account, storage
/// slot and contract code read or written resolves by hash from the parent
/// state root, and the transaction is valid there.
#[jolt::provable(stack_size = 1048576, max_input_size = 262144)]
pub fn execute(witness: &[u8]) -> Outcome {
    let witness = Witness::decode(&mut &*witness).unwrap();
    let preimages: B256Map<Bytes> = witness
        .preimages
        .iter()
        .map(|preimage| (keccak256(preimage), preimage.clone()))
        .collect();
    run(&witness, &preimages).unwrap()
}

pub fn run(witness: &Witness, preimages: &impl Preimages) -> Result<Outcome, Error> {
    let header = Header::decode(&mut witness.header.as_ref())?;
    let parent = Header::decode(&mut witness.parent_header.as_ref())?;
    if keccak256(&witness.parent_header) != header.parent_hash {
        return Err(Error::ParentHeader);
    }
    if header.timestamp < BPO2_TIMESTAMP {
        return Err(Error::UnsupportedFork);
    }
    let transaction = TxEnvelope::decode_2718(&mut witness.transaction.as_ref())?;
    let sender = sender(&transaction, &witness.public_key)?;

    let block = BlockEnv {
        number: U256::from(header.number),
        beneficiary: header.beneficiary,
        timestamp: U256::from(header.timestamp),
        gas_limit: header.gas_limit,
        basefee: header.base_fee_per_gas.unwrap_or_default(),
        difficulty: header.difficulty,
        prevrandao: Some(header.mix_hash),
        blob_excess_gas_and_price: header.excess_blob_gas.map(|excess| {
            BlobExcessGasAndPrice::new(excess, BlobParams::bpo2().update_fraction as u64)
        }),
        ..BlockEnv::default()
    };
    let tx = TxEnv {
        tx_type: transaction.tx_type().into(),
        caller: sender,
        gas_limit: transaction.gas_limit(),
        gas_price: transaction.max_fee_per_gas(),
        kind: transaction.kind(),
        value: transaction.value(),
        data: transaction.input().clone(),
        nonce: transaction.nonce(),
        chain_id: transaction.chain_id(),
        access_list: transaction.access_list().cloned().unwrap_or_default(),
        gas_priority_fee: transaction.max_priority_fee_per_gas(),
        blob_hashes: transaction
            .blob_versioned_hashes()
            .unwrap_or_default()
            .to_vec(),
        max_fee_per_blob_gas: transaction.max_fee_per_blob_gas().unwrap_or_default(),
        authorization_list: transaction
            .authorization_list()
            .unwrap_or_default()
            .iter()
            .cloned()
            .map(Either::Left)
            .collect(),
    };

    let mut state = State::new(&parent, &header, preimages);
    let ResultAndState {
        result,
        state: changes,
    } = Context::mainnet()
        .with_db(&mut state)
        .modify_cfg_chained(|cfg| cfg.set_spec_and_mainnet_gas_params(SpecId::OSAKA))
        .with_block(block)
        .build_mainnet()
        .transact(tx)
        .map_err(|error| Error::Invalid(format!("{error}")))?;

    let success = result.is_success();
    let gas_used = result.tx_gas_used();
    let receipt = Receipt {
        status: Eip658Value::Eip658(success),
        cumulative_gas_used: gas_used,
        logs: result.into_logs(),
    }
    .with_bloom();
    let receipt = match transaction.tx_type() {
        TxType::Legacy => ReceiptEnvelope::Legacy(receipt),
        TxType::Eip2930 => ReceiptEnvelope::Eip2930(receipt),
        TxType::Eip1559 => ReceiptEnvelope::Eip1559(receipt),
        TxType::Eip4844 => ReceiptEnvelope::Eip4844(receipt),
        TxType::Eip7702 => ReceiptEnvelope::Eip7702(receipt),
    };

    Ok(Outcome {
        block_hash: keccak256(&witness.header).0,
        transaction_hash: transaction.tx_hash().0,
        sender: sender.0 .0,
        success,
        gas_used,
        receipt_hash: keccak256(receipt.encoded_2718()).0,
        post_state_root: state.commit(changes)?.0,
    })
}

/// The address of `public_key` once the transaction's signature verifies under
/// it. Ethereum instead recovers the key from the signature and its parity
/// bit; this check does not read the parity bit, so of the two keys a
/// signature verifies under, it accepts either.
fn sender(transaction: &TxEnvelope, public_key: &[u8]) -> Result<Address, Error> {
    let signature = transaction.signature();
    // EIP-2: a high `s` is invalid.
    if signature.normalize_s().is_some() || public_key.len() != 64 {
        return Err(Error::Signature);
    }
    let mut limbs = [0; 8];
    limbs[..4].copy_from_slice(U256::from_be_slice(&public_key[..32]).as_limbs());
    limbs[4..].copy_from_slice(U256::from_be_slice(&public_key[32..]).as_limbs());
    let point = Secp256k1Point::from_u64_arr(&limbs).map_err(|_| Error::Signature)?;
    let mut digest = U256::from_be_bytes(transaction.signature_hash().0);
    if digest >= SECP256K1_ORDER {
        digest -= SECP256K1_ORDER;
    }
    let scalar =
        |value: U256| Secp256k1Fr::from_u64_arr(value.as_limbs()).map_err(|_| Error::Signature);
    ecdsa_verify(
        scalar(digest)?,
        scalar(signature.r())?,
        scalar(signature.s())?,
        point,
    )
    .map_err(|_| Error::Signature)?;
    Ok(Address::from_slice(&keccak256(public_key)[12..]))
}

/// `alloy-primitives`' `native-keccak` hook: every keccak256 in the guest's
/// dependencies runs on the Jolt inline.
///
/// # Safety
///
/// `bytes` is valid for reads of `len` bytes and `output` for writes of 32.
#[cfg(target_arch = "riscv64")]
#[no_mangle]
unsafe extern "C" fn native_keccak256(bytes: *const u8, len: usize, output: *mut u8) {
    use jolt_inlines_keccak256::Keccak256;

    // SAFETY: the caller upholds this function's contract.
    unsafe {
        let digest = Keccak256::digest(core::slice::from_raw_parts(bytes, len));
        output.copy_from_nonoverlapping(digest.as_ptr(), 32);
    }
}

#[cfg(test)]
mod tests {
    use alloy_primitives::{address, b256};

    use super::{execute, Outcome};

    /// Every field but the post-state root is the chain's record of block
    /// 26429593's first transaction; that root is no on-chain value.
    #[test]
    fn fixture_reproduces_the_mainnet_receipt() {
        assert_eq!(
            execute(include_bytes!("../../fixtures/mainnet-26429593-0.rlp")),
            Outcome {
                block_hash: b256!(
                    "0xa09b582ab29a09cb17205032e7bbbaf62678bdb0a8263480b38327ec7cbc862c"
                )
                .0,
                transaction_hash: b256!(
                    "0xf1a82aee7fdc4effd94853a2a7062c528dae7a3326216aab8ffc27809f82294f"
                )
                .0,
                sender: address!("0x00003f0b7045822689ce3a6118846218a2c22cfd").0 .0,
                success: true,
                gas_used: 106_489,
                receipt_hash: b256!(
                    "0xc5e4a88f00abd49be76cd333afcceb842648a0f162f4a738d99a6b8ccab4c4b4"
                )
                .0,
                post_state_root: b256!(
                    "0xab955130bb9c385f91bee54543d93a0b2b34eb66918d92beb5b3d659772fbd2c"
                )
                .0,
            }
        );
    }
}
