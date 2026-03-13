//! Symbolic Poseidon transcript for verifier tracing.
//!
//! `PoseidonSymbolicTranscript` implements [`Transcript`] by recording all
//! append/challenge operations as arena nodes instead of computing hashes.
//! The resulting AST captures the exact Fiat-Shamir structure that a circuit
//! verifier would need to reproduce.

use jolt_transcript::Transcript;

use crate::arena::{self, Atom, Edge, Node};
use crate::scalar_ops;
use crate::tunneling;

/// A symbolic transcript that records Poseidon operations in the arena.
///
/// Instead of hashing, each `append_bytes` and `challenge` call creates arena
/// nodes representing the corresponding Poseidon operations. The challenge
/// values are tunneled to `SymbolicField::from_u128()` so the caller gets
/// back a symbolic field element tied to the arena graph.
///
/// # Multi-element appends
///
/// PCS commitments may serialize to multiple field-element-sized chunks. These
/// are tunneled via `PENDING_COMMITMENT_CHUNKS` and hashed with Poseidon
/// chaining: first chunk includes `n_rounds` for domain separation, remaining
/// chunks chain as `poseidon(prev, 0, chunk_i)`. The chunk count is
/// PCS-determined — this transcript handles any number generically.
#[derive(Clone)]
pub struct PoseidonSymbolicTranscript {
    /// Current symbolic state — initially zero, updated by each operation.
    state: Edge,
    /// Round counter for generating unique challenge IDs.
    n_rounds: u64,
}

impl Default for PoseidonSymbolicTranscript {
    fn default() -> Self {
        Self {
            state: Atom::Scalar(scalar_ops::ZERO),
            n_rounds: 0,
        }
    }
}

impl std::fmt::Debug for PoseidonSymbolicTranscript {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PoseidonSymbolicTranscript")
            .field("state", &self.state)
            .field("n_rounds", &self.n_rounds)
            .finish()
    }
}

impl PoseidonSymbolicTranscript {
    /// Append multiple symbolic edges with Poseidon chaining.
    ///
    /// First element: `poseidon(state, n_rounds, elem)` — domain separation.
    /// Remaining elements: `poseidon(prev, 0, elem)` — chained without
    /// redundant `n_rounds`.
    ///
    /// This matches the concrete `PoseidonTranscript::raw_append_bytes` chunking
    /// semantics, generalized for an arbitrary number of symbolic elements.
    pub fn append_edges(&mut self, edges: &[Edge]) {
        let n_rounds_edge = Atom::Scalar(scalar_ops::from_u64(self.n_rounds));
        let zero_edge = Atom::Scalar(scalar_ops::ZERO);

        let mut iter = edges.iter();

        // First element includes n_rounds for domain separation
        let first_data = iter.next().copied().unwrap_or(zero_edge);
        let mut current = arena::alloc(Node::Poseidon {
            state: self.state,
            n_rounds: n_rounds_edge,
            data: first_data,
        });

        // Remaining elements chain without n_rounds
        for &edge in iter {
            current = arena::alloc(Node::Poseidon {
                state: Atom::Node(current),
                n_rounds: zero_edge,
                data: edge,
            });
        }

        self.state = Atom::Node(current);
        self.n_rounds += 1;
    }
}

impl Transcript for PoseidonSymbolicTranscript {
    type Challenge = u128;

    fn new(label: &'static [u8]) -> Self {
        assert!(label.len() < 33, "label must be less than 33 bytes");

        // Initialize state from label — this is a constant, so we fold it
        let mut padded = [0u8; 32];
        padded[..label.len()].copy_from_slice(label);
        let state_val = scalar_ops::from_bytes_le(&padded);

        Self {
            state: Atom::Scalar(state_val),
            n_rounds: 0,
        }
    }

    fn append_bytes(&mut self, bytes: &[u8]) {
        // Check for multi-chunk commitment tunneled from AppendToTranscript
        if let Some(chunks) = tunneling::take_pending_commitment_chunks() {
            self.append_edges(&chunks);
            return;
        }

        // Check for single symbolic value from SymbolicField::to_bytes()
        let data_edge = if let Some(edge) = tunneling::take_pending_append() {
            edge
        } else {
            // Raw bytes (e.g., from a &[u8] or non-field append) — embed as constant
            Atom::Scalar(scalar_ops::from_bytes_le(bytes))
        };

        // Record a Poseidon hash: state = Poseidon(state, n_rounds, data)
        let n_rounds_edge = Atom::Scalar(scalar_ops::from_u64(self.n_rounds));
        let new_state = arena::alloc(Node::Poseidon {
            state: self.state,
            n_rounds: n_rounds_edge,
            data: data_edge,
        });
        self.state = Atom::Node(new_state);
        self.n_rounds += 1;
    }

    fn challenge(&mut self) -> Self::Challenge {
        // Record a challenge node derived from current state
        let challenge_id = self.n_rounds;
        let challenge_node = arena::alloc(Node::Challenge { id: challenge_id });
        self.n_rounds += 1;

        // Tunnel the challenge edge so SymbolicField::from_u128() picks it up
        tunneling::set_pending_challenge(Atom::Node(challenge_node));

        // Return a dummy u128 — the real symbolic value is in the tunnel
        0u128
    }

    fn state(&self) -> &[u8; 32] {
        // We can't return a meaningful state for symbolic transcripts.
        // This is used for debugging/testing; return a zero state.
        static ZERO_STATE: [u8; 32] = [0u8; 32];
        &ZERO_STATE
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::{self, ArenaSession};

    #[test]
    fn new_transcript_is_constant() {
        let _session = ArenaSession::new();
        let transcript = PoseidonSymbolicTranscript::new(b"test");
        // State should be a constant derived from the label
        assert!(matches!(transcript.state, Atom::Scalar(_)));
        assert_eq!(transcript.n_rounds, 0);
    }

    #[test]
    fn append_creates_poseidon_node() {
        let _session = ArenaSession::new();
        let mut transcript = PoseidonSymbolicTranscript::new(b"test");

        // Append raw bytes
        transcript.append_bytes(b"hello");

        // State should now be a node (Poseidon hash)
        assert!(matches!(transcript.state, Atom::Node(_)));
        assert_eq!(transcript.n_rounds, 1);
        assert_eq!(arena::node_count(), 1); // one Poseidon node
    }

    #[test]
    fn challenge_creates_node_and_tunnels() {
        let _session = ArenaSession::new();
        let mut transcript = PoseidonSymbolicTranscript::new(b"test");

        let _challenge_val = transcript.challenge();

        // A Challenge node should exist in the arena
        assert_eq!(arena::node_count(), 1);

        // The challenge should have been tunneled
        // (it was already consumed by the challenge() call, but we can verify
        // by checking that from_u128 with pending challenge works)
        let mut transcript2 = PoseidonSymbolicTranscript::new(b"test2");
        let _ = transcript2.challenge();

        // The tunnel should be set for the next from_u128 call
        use crate::symbolic::SymbolicField;
        use jolt_field::Field;
        let symbolic = SymbolicField::from_u128(0);
        // Should be a Node (from tunnel), not a Scalar
        assert!(!symbolic.is_constant());
    }

    #[test]
    fn append_tunneled_symbolic_value() {
        let _session = ArenaSession::new();
        use crate::symbolic::SymbolicField;

        let x = SymbolicField::variable(0, "x");
        let mut transcript = PoseidonSymbolicTranscript::new(b"test");

        // Append symbolic value through the blanket impl path
        transcript.append::<SymbolicField>(&x);

        // Should have created a Poseidon node with the symbolic var as data
        assert_eq!(transcript.n_rounds, 1);
        let nodes = arena::snapshot();
        // Node 0: Var(x), Node 1: Poseidon
        assert_eq!(nodes.len(), 2);
        assert!(matches!(nodes[1], Node::Poseidon { .. }));
    }

    #[test]
    fn multiple_appends_chain_state() {
        let _session = ArenaSession::new();
        let mut transcript = PoseidonSymbolicTranscript::new(b"test");

        transcript.append_bytes(b"a");
        transcript.append_bytes(b"b");
        transcript.append_bytes(b"c");

        assert_eq!(transcript.n_rounds, 3);
        // Each Poseidon node chains to the previous state
        assert!(matches!(transcript.state, Atom::Node(_)));
        assert_eq!(arena::node_count(), 3);
    }

    #[test]
    fn append_edges_multi_element() {
        let _session = ArenaSession::new();
        let mut transcript = PoseidonSymbolicTranscript::new(b"test");

        let edges = vec![
            Atom::Scalar(scalar_ops::from_u64(1)),
            Atom::Scalar(scalar_ops::from_u64(2)),
            Atom::Scalar(scalar_ops::from_u64(3)),
        ];
        transcript.append_edges(&edges);

        // 3 chained Poseidon nodes: first with n_rounds, rest with 0
        assert_eq!(arena::node_count(), 3);
        assert_eq!(transcript.n_rounds, 1); // single domain separation increment
    }

    #[test]
    fn commitment_chunks_via_tunnel() {
        let _session = ArenaSession::new();
        let mut transcript = PoseidonSymbolicTranscript::new(b"test");

        // Simulate a commitment AppendToTranscript: set chunks, then call append_bytes
        let chunks = vec![
            Atom::Scalar(scalar_ops::from_u64(10)),
            Atom::Scalar(scalar_ops::from_u64(20)),
        ];
        crate::tunneling::set_pending_commitment_chunks(chunks);
        transcript.append_bytes(b"ignored_because_chunks_take_priority");

        // 2 chained Poseidon nodes
        assert_eq!(arena::node_count(), 2);
        assert_eq!(transcript.n_rounds, 1);
    }

    #[test]
    fn clone_independence() {
        let _session = ArenaSession::new();
        let mut transcript = PoseidonSymbolicTranscript::new(b"test");
        transcript.append_bytes(b"shared");

        let mut fork = transcript.clone();
        transcript.append_bytes(b"branch_a");
        fork.append_bytes(b"branch_b");

        assert_ne!(transcript.state, fork.state);
    }
}
