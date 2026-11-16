//! The parent block's world state, read through a witness and updated with a
//! transaction's changes.

use alloy_consensus::Header;
use alloy_primitives::map::AddressMap;
use alloy_primitives::{keccak256, Address, Bytes, B256, U256};
use alloy_rlp::Decodable as _;
use alloy_trie::{TrieAccount, EMPTY_ROOT_HASH};
use revm::bytecode::Bytecode;
use revm::state::{AccountInfo, EvmState};
use revm::Database;

use crate::mpt::{path, Node};
use crate::{Error, Preimages};

pub(crate) struct State<'a, P> {
    accounts: Node,
    /// The storage tries of the accounts whose storage has been read.
    storage: AddressMap<Node>,
    parent_hash: B256,
    number: u64,
    preimages: &'a P,
}

impl<'a, P: Preimages> State<'a, P> {
    pub(crate) fn new(parent: &Header, header: &Header, preimages: &'a P) -> Self {
        Self {
            accounts: Node::from_root(parent.state_root),
            storage: AddressMap::default(),
            parent_hash: header.parent_hash,
            number: header.number,
            preimages,
        }
    }

    fn account(&mut self, address: Address) -> Result<Option<TrieAccount>, Error> {
        let leaf = self
            .accounts
            .get(&path(keccak256(address)), self.preimages)?;
        Ok(leaf
            .map(|mut rlp| TrieAccount::decode(&mut rlp))
            .transpose()?)
    }

    fn storage_root(&mut self, address: Address) -> Result<B256, Error> {
        let account = self.account(address)?;
        Ok(account.map_or(EMPTY_ROOT_HASH, |account| account.storage_root))
    }

    /// Applies a transaction's changes and returns the new state root.
    pub(crate) fn commit(mut self, changes: EvmState) -> Result<B256, Error> {
        for (address, account) in changes {
            if !account.is_touched() {
                continue;
            }
            let key = path(keccak256(address));
            // EIP-161: a touched account that ends empty leaves the state.
            if account.is_selfdestructed() || account.is_empty() {
                let _ = self.accounts.remove(&key, self.preimages)?;
                continue;
            }
            let mut slots = account.changed_storage_slots().peekable();
            let storage_root = if account.is_created() || slots.peek().is_some() {
                let mut trie = match self.storage.remove(&address) {
                    Some(trie) if !account.is_created() => trie,
                    _ => Node::Empty,
                };
                for (slot, value) in slots {
                    let key = path(keccak256(B256::from(*slot)));
                    let value = value.present_value();
                    if value.is_zero() {
                        let _ = trie.remove(&key, self.preimages)?;
                    } else {
                        trie.insert(&key, alloy_rlp::encode(value).into(), self.preimages)?;
                    }
                }
                trie.root()
            } else {
                self.storage_root(address)?
            };
            let leaf = TrieAccount {
                nonce: account.info.nonce,
                balance: account.info.balance,
                storage_root,
                code_hash: account.info.code_hash,
            };
            self.accounts
                .insert(&key, alloy_rlp::encode(leaf).into(), self.preimages)?;
        }
        Ok(self.accounts.root())
    }
}

impl<P: Preimages> Database for State<'_, P> {
    type Error = Error;

    fn basic(&mut self, address: Address) -> Result<Option<AccountInfo>, Error> {
        Ok(self.account(address)?.map(|account| AccountInfo {
            balance: account.balance,
            nonce: account.nonce,
            code_hash: account.code_hash,
            account_id: None,
            code: None,
        }))
    }

    fn code_by_hash(&mut self, code_hash: B256) -> Result<Bytecode, Error> {
        let code = self
            .preimages
            .preimage(&code_hash)
            .ok_or(Error::MissingPreimage(code_hash))?;
        Ok(Bytecode::new_raw(Bytes::copy_from_slice(code)))
    }

    fn storage(&mut self, address: Address, index: U256) -> Result<U256, Error> {
        if !self.storage.contains_key(&address) {
            let root = self.storage_root(address)?;
            let _ = self.storage.insert(address, Node::from_root(root));
        }
        let trie = self.storage.get_mut(&address).ok_or(Error::Trie)?;
        let value = trie.get(&path(keccak256(B256::from(index))), self.preimages)?;
        Ok(value
            .map(|mut rlp| U256::decode(&mut rlp))
            .transpose()?
            .unwrap_or_default())
    }

    fn block_hash(&mut self, number: u64) -> Result<B256, Error> {
        if number + 1 == self.number {
            Ok(self.parent_hash)
        } else {
            Err(Error::BlockHash(number))
        }
    }
}
