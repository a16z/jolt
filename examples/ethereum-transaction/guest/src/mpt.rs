//! A sparse Merkle-Patricia trie over keccak-hashed keys, for Ethereum's
//! state and storage tries. Only the paths a transaction touches are held in
//! memory; every other subtree stays a digest. A digest is expanded by looking
//! its preimage up by hash, which is what authenticates the witness.

use alloc::boxed::Box;
use alloc::vec::Vec;

use alloy_primitives::{keccak256, Bytes, B256};
use alloy_rlp::{Encodable, Header, EMPTY_STRING_CODE};
use alloy_trie::EMPTY_ROOT_HASH;

use crate::{Error, Preimages};

/// A key's 64 nibbles, most significant first.
pub(crate) type Path = [u8; 64];

pub(crate) fn path(key: B256) -> Path {
    let mut nibbles = [0; 64];
    for (pair, byte) in nibbles.as_chunks_mut().0.iter_mut().zip(key.0) {
        *pair = [byte >> 4, byte & 0xf];
    }
    nibbles
}

pub(crate) enum Node {
    Empty,
    Digest(B256),
    Leaf(Vec<u8>, Bytes),
    Extension(Vec<u8>, Box<Node>),
    /// Keys have one fixed length, so a branch never holds a value.
    Branch(Box<[Node; 16]>),
}

impl Node {
    pub(crate) fn from_root(root: B256) -> Self {
        if root == EMPTY_ROOT_HASH {
            Self::Empty
        } else {
            Self::Digest(root)
        }
    }

    pub(crate) fn root(&self) -> B256 {
        match self {
            Self::Empty => EMPTY_ROOT_HASH,
            Self::Digest(digest) => *digest,
            node => keccak256(node.rlp()),
        }
    }

    pub(crate) fn get(
        &mut self,
        key: &[u8],
        preimages: &impl Preimages,
    ) -> Result<Option<&[u8]>, Error> {
        self.expand(preimages)?;
        match self {
            Self::Empty => Ok(None),
            Self::Leaf(path, value) => Ok((path.as_slice() == key).then_some(&value[..])),
            Self::Extension(path, child) => match key.strip_prefix(path.as_slice()) {
                Some(rest) => child.get(rest, preimages),
                None => Ok(None),
            },
            Self::Branch(children) => children[usize::from(key[0])].get(&key[1..], preimages),
            Self::Digest(_) => unreachable!("expanded above"),
        }
    }

    /// Sets `key` to a nonempty `value`.
    pub(crate) fn insert(
        &mut self,
        key: &[u8],
        value: Bytes,
        preimages: &impl Preimages,
    ) -> Result<(), Error> {
        self.expand(preimages)?;
        match self {
            Self::Empty => *self = Self::Leaf(key.to_vec(), value),
            Self::Branch(children) => {
                children[usize::from(key[0])].insert(&key[1..], value, preimages)?;
            }
            Self::Leaf(path, old) if path.as_slice() == key => *old = value,
            Self::Extension(path, child) if key.starts_with(path) => {
                child.insert(&key[path.len()..], value, preimages)?;
            }
            Self::Leaf(path, _) | Self::Extension(path, _) => {
                let shared = path.iter().zip(key).take_while(|(a, b)| a == b).count();
                let slot = usize::from(path[shared]);
                let suffix = path.split_off(shared + 1);
                path.truncate(shared);
                let prefix = core::mem::take(path);
                let mut children = Box::new([const { Self::Empty }; 16]);
                children[slot] = match core::mem::replace(self, Self::Empty) {
                    Self::Leaf(_, old) => Self::Leaf(suffix, old),
                    Self::Extension(_, child) => child.prefixed(suffix),
                    _ => unreachable!("matched a leaf or an extension"),
                };
                children[usize::from(key[shared])] = Self::Leaf(key[shared + 1..].to_vec(), value);
                *self = Self::Branch(children).prefixed(prefix);
            }
            Self::Digest(_) => unreachable!("expanded above"),
        }
        Ok(())
    }

    /// Removes `key`, returning whether it was present.
    pub(crate) fn remove(&mut self, key: &[u8], preimages: &impl Preimages) -> Result<bool, Error> {
        self.expand(preimages)?;
        let removed = match self {
            Self::Empty => false,
            Self::Leaf(path, _) => path.as_slice() == key,
            Self::Extension(path, child) => match key.strip_prefix(path.as_slice()) {
                Some(rest) => child.remove(rest, preimages)?,
                None => false,
            },
            Self::Branch(children) => children[usize::from(key[0])].remove(&key[1..], preimages)?,
            Self::Digest(_) => unreachable!("expanded above"),
        };
        if !removed {
            return Ok(false);
        }
        *self = match core::mem::replace(self, Self::Empty) {
            Self::Extension(path, child) => child.prefixed(path),
            Self::Branch(mut children) => {
                let mut occupied = (0..16).filter(|&slot| !matches!(children[slot], Self::Empty));
                match (occupied.next(), occupied.next()) {
                    (Some(slot), None) => {
                        let mut only = core::mem::replace(&mut children[slot], Self::Empty);
                        only.expand(preimages)?;
                        only.prefixed(alloc::vec![slot as u8])
                    }
                    _ => Self::Branch(children),
                }
            }
            _ => Self::Empty,
        };
        Ok(true)
    }

    /// This node behind `prefix` more nibbles of path.
    fn prefixed(self, mut prefix: Vec<u8>) -> Self {
        if prefix.is_empty() {
            return self;
        }
        match self {
            Self::Leaf(path, value) => {
                prefix.extend(path);
                Self::Leaf(prefix, value)
            }
            Self::Extension(path, child) => {
                prefix.extend(path);
                Self::Extension(prefix, child)
            }
            node => Self::Extension(prefix, Box::new(node)),
        }
    }

    fn expand(&mut self, preimages: &impl Preimages) -> Result<(), Error> {
        if let Self::Digest(digest) = self {
            let rlp = preimages
                .preimage(digest)
                .ok_or(Error::MissingPreimage(*digest))?;
            *self = Self::decode(rlp)?;
        }
        Ok(())
    }

    fn decode(mut rlp: &[u8]) -> Result<Self, Error> {
        let mut items = Header::decode_bytes(&mut rlp, true)?;
        let mut fields = Vec::with_capacity(17);
        while !items.is_empty() {
            let (start, header) = (items, Header::decode(&mut items)?);
            let (payload, rest) = items
                .split_at_checked(header.payload_length)
                .ok_or(Error::Trie)?;
            fields.push((header.list, payload, &start[..start.len() - rest.len()]));
            items = rest;
        }
        let child = |&(list, payload, item): &(bool, &[u8], &[u8])| match (list, payload.len()) {
            (true, _) => Self::decode(item),
            (false, 0) => Ok(Self::Empty),
            (false, 32) => Ok(Self::Digest(B256::from_slice(payload))),
            _ => Err(Error::Trie),
        };
        match fields.as_slice() {
            [(false, encoded_path, _), target] => {
                let (&flags, rest) = encoded_path.split_first().ok_or(Error::Trie)?;
                let mut path = Vec::with_capacity(2 * encoded_path.len());
                if flags & 0x10 != 0 {
                    path.push(flags & 0xf);
                }
                path.extend(rest.iter().flat_map(|byte| [byte >> 4, byte & 0xf]));
                match (flags & 0x20 != 0, target) {
                    (true, (false, value, _)) => {
                        Ok(Self::Leaf(path, Bytes::copy_from_slice(value)))
                    }
                    (false, target) => Ok(Self::Extension(path, Box::new(child(target)?))),
                    _ => Err(Error::Trie),
                }
            }
            [children @ .., (false, [], _)] if children.len() == 16 => {
                let mut nodes = Box::new([const { Self::Empty }; 16]);
                for (node, field) in nodes.iter_mut().zip(children) {
                    *node = child(field)?;
                }
                Ok(Self::Branch(nodes))
            }
            _ => Err(Error::Trie),
        }
    }

    fn rlp(&self) -> Vec<u8> {
        let mut payload = Vec::new();
        match self {
            Self::Leaf(path, value) => {
                compact(path, true).as_slice().encode(&mut payload);
                value.encode(&mut payload);
            }
            Self::Extension(path, child) => {
                compact(path, false).as_slice().encode(&mut payload);
                child.reference(&mut payload);
            }
            Self::Branch(children) => {
                for child in children.iter() {
                    child.reference(&mut payload);
                }
                payload.push(EMPTY_STRING_CODE);
            }
            Self::Empty | Self::Digest(_) => unreachable!("only resolved nodes are encoded"),
        }
        let mut rlp = Vec::with_capacity(payload.len() + 3);
        Header {
            list: true,
            payload_length: payload.len(),
        }
        .encode(&mut rlp);
        rlp.extend(payload);
        rlp
    }

    /// How a parent refers to this node: inline below 32 bytes, else by hash.
    fn reference(&self, out: &mut Vec<u8>) {
        match self {
            Self::Empty => out.push(EMPTY_STRING_CODE),
            Self::Digest(digest) => digest.encode(out),
            node => {
                let rlp = node.rlp();
                if rlp.len() < 32 {
                    out.extend(rlp);
                } else {
                    keccak256(rlp).encode(out);
                }
            }
        }
    }
}

/// Hex-prefix encoding of a nibble path (Yellow Paper, appendix C).
fn compact(path: &[u8], leaf: bool) -> Vec<u8> {
    let flag = if leaf { 0x20 } else { 0 };
    let (first, rest) = if path.len() % 2 == 1 {
        (flag | 0x10 | path[0], &path[1..])
    } else {
        (flag, path)
    };
    let mut bytes = Vec::with_capacity(1 + rest.len() / 2);
    bytes.push(first);
    bytes.extend(rest.as_chunks().0.iter().map(|[high, low]| high << 4 | low));
    bytes
}

#[cfg(test)]
mod tests {
    use alloy_primitives::map::B256Map;
    use alloy_primitives::{b256, hex, keccak256, Bytes, B256};

    use super::{path, Node};

    /// The root after `operations` in order; `None` deletes the key.
    fn root(operations: &[(&[u8], Option<&[u8]>)]) -> B256 {
        let preimages = B256Map::<Bytes>::default();
        let mut trie = Node::Empty;
        for &(key, value) in operations {
            let key = path(keccak256(key));
            match value {
                Some(value) => trie
                    .insert(&key, Bytes::copy_from_slice(value), &preimages)
                    .unwrap(),
                None => assert!(trie.remove(&key, &preimages).unwrap()),
            }
        }
        trie.root()
    }

    /// `emptyValues` and `jeff` of `TrieTests/trietest_secureTrie.json` in
    /// ethereum/tests.
    #[test]
    fn matches_the_ethereum_secure_trie_vectors() {
        assert_eq!(
            root(&[
                (b"do", Some(b"verb")),
                (b"ether", Some(b"wookiedoo")),
                (b"horse", Some(b"stallion")),
                (b"shaman", Some(b"horse")),
                (b"doge", Some(b"coin")),
                (b"ether", None),
                (b"dog", Some(b"puppy")),
                (b"shaman", None),
            ]),
            b256!("0x29b235a58c3c25ab83010c327d5932bcf05324b7d6b1185e650798034783ca9d")
        );

        let word = |hex: &str| hex::decode(format!("{hex:0>64}")).unwrap();
        let padded = |hex: &str| hex::decode(format!("{hex:0<64}")).unwrap();
        let (owner, registrar, name) = (
            hex!("697c7b8c961b56f675d570498424ac8de1a918f6"),
            hex!("7ef9e639e2733cb34e4dfc576d4b23f72db776b2"),
            hex!("ec4f34c97e43fbb2816cfd95e388353c7181dab1"),
        );
        assert_eq!(
            root(&[
                (
                    &word("45"),
                    Some(&hex!("22b224a1420a802ab51d326e29fa98e34c4f24ea"))
                ),
                (&word("46"), Some(&padded("67706c207633"))),
                (&word("1234567890"), Some(&owner)),
                (&word(&hex::encode(owner)), Some(&hex!("1234567890"))),
                (
                    &word(&hex::encode(registrar)),
                    Some(&padded("465547415600"))
                ),
                (&word(&hex::encode(name)), Some(&padded("4e616d65526567"))),
                (&padded("465547415600"), Some(&registrar)),
                (&padded("4e616d65526567"), Some(&name)),
                (&word("1234567890"), None),
                (
                    &word(&hex::encode(owner)),
                    Some(&padded("6f6f6f6820736f2067726561742c207265616c6c6c793f")),
                ),
                (
                    &padded("6f6f6f6820736f2067726561742c207265616c6c6c793f"),
                    Some(&owner)
                ),
            ]),
            b256!("0x72adb52e9d9428f808e3e8045be18d3baa77881d0cfab89a17a2bcbacee2f320")
        );
    }
}
