//! Thread-local tunneling for transcript ↔ `SymbolicField` communication.
//!
//! The `AppendToTranscript` blanket impl calls `to_bytes()` then
//! `append_bytes(reversed)`. Since `to_bytes()` returns raw bytes with no
//! symbolic context, we tunnel the symbolic value through a thread-local:
//!
//! 1. `SymbolicField::to_bytes()` stores its `Edge` in `PENDING_APPEND`
//! 2. `PoseidonSymbolicTranscript::append_bytes()` retrieves it
//!
//! Similarly for challenges:
//!
//! 1. `PoseidonSymbolicTranscript::challenge()` stores a `Challenge` edge in
//!    `PENDING_CHALLENGE`
//! 2. `SymbolicField::from_u128()` retrieves it (since `Transcript::Challenge = u128`)

use crate::arena::Edge;

use std::cell::RefCell;

thread_local! {
    /// Set by `SymbolicField::to_bytes()`, consumed by
    /// `PoseidonSymbolicTranscript::append_bytes()`.
    static PENDING_APPEND: RefCell<Option<Edge>> = const { RefCell::new(None) };

    /// Set by `PoseidonSymbolicTranscript::challenge()`, consumed by
    /// `SymbolicField::from_u128()`.
    static PENDING_CHALLENGE: RefCell<Option<Edge>> = const { RefCell::new(None) };

    /// Set by a symbolic commitment's `AppendToTranscript` impl, consumed by
    /// `PoseidonSymbolicTranscript::append_bytes()`.
    ///
    /// Commitments serialize to multiple field-element-sized chunks (e.g.,
    /// Dory: 12 × 32-byte chunks from a 384-byte G1 point, hash-based PCS:
    /// 1 chunk from a 32-byte digest). The chunk count is PCS-determined —
    /// this tunnel just forwards whatever the caller provides.
    static PENDING_COMMITMENT_CHUNKS: RefCell<Option<Vec<Edge>>> = const { RefCell::new(None) };
}

/// Store a symbolic edge for the pending `append_bytes` call.
pub fn set_pending_append(edge: Edge) {
    PENDING_APPEND.with(|cell| {
        *cell.borrow_mut() = Some(edge);
    });
}

/// Take the pending symbolic edge (returns `None` if not set or already consumed).
pub fn take_pending_append() -> Option<Edge> {
    PENDING_APPEND.with(|cell| cell.borrow_mut().take())
}

/// Store a symbolic edge for the pending `from_u128` call.
pub fn set_pending_challenge(edge: Edge) {
    PENDING_CHALLENGE.with(|cell| {
        *cell.borrow_mut() = Some(edge);
    });
}

/// Take the pending challenge edge (returns `None` if not set or already consumed).
pub fn take_pending_challenge() -> Option<Edge> {
    PENDING_CHALLENGE.with(|cell| cell.borrow_mut().take())
}

/// Store symbolic edges for a multi-chunk commitment append.
///
/// Called by a symbolic commitment type's `AppendToTranscript` implementation
/// before it triggers `transcript.append_bytes()`. The transcript retrieves
/// these chunks and hashes them with Poseidon chaining.
///
/// The number of chunks is PCS-determined: Dory produces 12 (384 bytes / 32),
/// a hash-based PCS produces 1, lattice-based might produce more.
pub fn set_pending_commitment_chunks(chunks: Vec<Edge>) {
    PENDING_COMMITMENT_CHUNKS.with(|cell| {
        *cell.borrow_mut() = Some(chunks);
    });
}

/// Take the pending commitment chunks (returns `None` if not set or already consumed).
pub fn take_pending_commitment_chunks() -> Option<Vec<Edge>> {
    PENDING_COMMITMENT_CHUNKS.with(|cell| cell.borrow_mut().take())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{ArenaSession, Atom, NodeId};
    use crate::scalar_ops;

    #[test]
    fn append_roundtrip() {
        let _session = ArenaSession::new();
        let edge = Atom::Scalar(scalar_ops::from_u64(42));
        set_pending_append(edge);
        assert_eq!(take_pending_append(), Some(edge));
        // Second take returns None
        assert_eq!(take_pending_append(), None);
    }

    #[test]
    fn challenge_roundtrip() {
        let _session = ArenaSession::new();
        let edge = Atom::Node(NodeId(0));
        set_pending_challenge(edge);
        assert_eq!(take_pending_challenge(), Some(edge));
        assert_eq!(take_pending_challenge(), None);
    }

    #[test]
    fn independent_channels() {
        let _session = ArenaSession::new();
        let edge_a = Atom::Scalar(scalar_ops::from_u64(1));
        let edge_b = Atom::Scalar(scalar_ops::from_u64(2));
        set_pending_append(edge_a);
        set_pending_challenge(edge_b);
        // They don't interfere
        assert_eq!(take_pending_append(), Some(edge_a));
        assert_eq!(take_pending_challenge(), Some(edge_b));
    }

    #[test]
    fn commitment_chunks_roundtrip() {
        let _session = ArenaSession::new();
        let chunks = vec![
            Atom::Scalar(scalar_ops::from_u64(1)),
            Atom::Scalar(scalar_ops::from_u64(2)),
            Atom::Scalar(scalar_ops::from_u64(3)),
        ];
        set_pending_commitment_chunks(chunks.clone());
        let taken = take_pending_commitment_chunks();
        assert_eq!(taken, Some(chunks));
        // Second take returns None
        assert_eq!(take_pending_commitment_chunks(), None);
    }
}
