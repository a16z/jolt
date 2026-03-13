//! Global AST arena for symbolic field operations.
//!
//! The arena stores an append-only DAG of arithmetic operations. Because `Field`
//! requires `Copy`, `SymbolicField` is a 4-byte `NodeId` index into this arena.
//! `Field` methods have no arena parameter, so the arena must be globally
//! accessible.
//!
//! # Lifecycle
//!
//! Create an [`ArenaSession`] RAII guard to initialize the arena before symbolic
//! execution. The arena is cleared when the guard drops.
//!
//! ```
//! use jolt_wrapper::arena::ArenaSession;
//!
//! let _session = ArenaSession::new();
//! // ... symbolic execution ...
//! // arena is cleared on drop
//! ```
//!
//! # Thread safety
//!
//! The arena uses `OnceLock<RwLock<Arena>>`. Multiple readers can access
//! concurrently; writers (new node allocation) take a brief write lock.

use std::sync::{OnceLock, RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::scalar_ops;

/// Index into the global arena. This is the runtime representation of `SymbolicField`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct NodeId(pub u32);

/// A scalar constant or a reference to an arena node.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Atom {
    /// An inline BN254 scalar value (constant folding).
    Scalar([u64; 4]),
    /// A reference to a node in the arena.
    Node(NodeId),
}

/// An edge in the AST: either a constant or a node reference.
///
/// Computations involving only constants are folded at construction time — no
/// arena node is created. When at least one operand is a `Node`, the operation
/// produces a new `Node`.
pub type Edge = Atom;

/// An operation recorded in the arena.
#[derive(Debug, Clone)]
pub enum Node {
    /// A named input variable (e.g., opening or challenge).
    Var { index: u32, name: String },
    /// Additive inverse.
    Neg(Edge),
    /// Multiplicative inverse.
    Inv(Edge),
    /// Addition.
    Add(Edge, Edge),
    /// Subtraction.
    Sub(Edge, Edge),
    /// Multiplication.
    Mul(Edge, Edge),
    /// Division.
    Div(Edge, Edge),
    /// Fiat-Shamir challenge absorbed from transcript.
    Challenge { id: u64 },
    /// Poseidon hash operation (transcript primitive).
    Poseidon {
        state: Edge,
        n_rounds: Edge,
        data: Edge,
    },
    /// Byte-reverse a 32-byte field element (LE ↔ BE).
    ByteReverse(Edge),
    /// Truncate to low 128 bits.
    Truncate128(Edge),
    /// Multiply by 2^192.
    MulTwoPow192(Edge),
}

/// The global arena storage.
struct Arena {
    nodes: Vec<Node>,
}

impl Arena {
    fn new() -> Self {
        Self {
            nodes: Vec::with_capacity(1 << 16),
        }
    }
}

static ARENA: OnceLock<RwLock<Arena>> = OnceLock::new();

fn get_arena() -> &'static RwLock<Arena> {
    ARENA
        .get()
        .expect("ArenaSession not active — create an ArenaSession before using SymbolicField")
}

fn read_arena() -> RwLockReadGuard<'static, Arena> {
    get_arena().read().expect("arena read lock poisoned")
}

fn write_arena() -> RwLockWriteGuard<'static, Arena> {
    get_arena().write().expect("arena write lock poisoned")
}

pub fn alloc(node: Node) -> NodeId {
    let mut arena = write_arena();
    let id = NodeId(arena.nodes.len() as u32);
    arena.nodes.push(node);
    id
}

/// # Panics
///
/// Panics if the `NodeId` is out of bounds.
pub fn get_node(id: NodeId) -> Node {
    let arena = read_arena();
    arena.nodes[id.0 as usize].clone()
}

pub fn node_count() -> usize {
    let arena = read_arena();
    arena.nodes.len()
}

pub fn snapshot() -> Vec<Node> {
    let arena = read_arena();
    arena.nodes.clone()
}

/// Performs constant folding for a binary operation if both operands are scalars.
/// Returns `Some(Atom::Scalar(...))` if folded, `None` if at least one operand is a node.
pub fn try_fold_binary(
    lhs: Edge,
    rhs: Edge,
    fold: fn([u64; 4], [u64; 4]) -> [u64; 4],
    make_node: fn(Edge, Edge) -> Node,
) -> Edge {
    match (lhs, rhs) {
        (Atom::Scalar(a), Atom::Scalar(b)) => Atom::Scalar(fold(a, b)),
        _ => Atom::Node(alloc(make_node(lhs, rhs))),
    }
}

/// Performs constant folding for a unary operation.
pub fn try_fold_unary(
    inner: Edge,
    fold: fn([u64; 4]) -> [u64; 4],
    make_node: fn(Edge) -> Node,
) -> Edge {
    match inner {
        Atom::Scalar(a) => Atom::Scalar(fold(a)),
        Atom::Node(_) => Atom::Node(alloc(make_node(inner))),
    }
}

/// RAII guard that initializes and cleans up the global arena.
///
/// Create one at the start of symbolic execution. The arena is cleared when the
/// guard drops.
///
/// # Panics
///
/// Panics if an `ArenaSession` is already active (nested sessions are not
/// supported).
pub struct ArenaSession {
    _private: (),
}

impl ArenaSession {
    pub fn new() -> Self {
        // Initialize the arena. If it already exists, clear it.
        match ARENA.get() {
            Some(lock) => {
                let mut arena = lock.write().expect("arena write lock poisoned");
                arena.nodes.clear();
            }
            None => {
                let _ = ARENA.set(RwLock::new(Arena::new()));
            }
        }
        Self { _private: () }
    }

    pub fn node_count(&self) -> usize {
        node_count()
    }

    pub fn snapshot(&self) -> Vec<Node> {
        snapshot()
    }
}

impl Default for ArenaSession {
    fn default() -> Self {
        Self::new()
    }
}

impl Drop for ArenaSession {
    fn drop(&mut self) {
        if let Some(lock) = ARENA.get() {
            if let Ok(mut arena) = lock.write() {
                arena.nodes.clear();
            }
        }
    }
}

pub fn scalar_edge(val: [u64; 4]) -> Edge {
    Atom::Scalar(val)
}

pub fn node_edge(id: NodeId) -> Edge {
    Atom::Node(id)
}

pub fn add_edges(lhs: Edge, rhs: Edge) -> Edge {
    // Identity: x + 0 = x
    if let Atom::Scalar(s) = rhs {
        if scalar_ops::is_zero(s) {
            return lhs;
        }
    }
    if let Atom::Scalar(s) = lhs {
        if scalar_ops::is_zero(s) {
            return rhs;
        }
    }
    try_fold_binary(lhs, rhs, scalar_ops::add, Node::Add)
}

pub fn sub_edges(lhs: Edge, rhs: Edge) -> Edge {
    // Identity: x - 0 = x
    if let Atom::Scalar(s) = rhs {
        if scalar_ops::is_zero(s) {
            return lhs;
        }
    }
    try_fold_binary(lhs, rhs, scalar_ops::sub, Node::Sub)
}

pub fn mul_edges(lhs: Edge, rhs: Edge) -> Edge {
    // x * 0 = 0
    if let Atom::Scalar(s) = lhs {
        if scalar_ops::is_zero(s) {
            return Atom::Scalar(scalar_ops::ZERO);
        }
    }
    if let Atom::Scalar(s) = rhs {
        if scalar_ops::is_zero(s) {
            return Atom::Scalar(scalar_ops::ZERO);
        }
    }
    // x * 1 = x
    if let Atom::Scalar(s) = lhs {
        if scalar_ops::is_one(s) {
            return rhs;
        }
    }
    if let Atom::Scalar(s) = rhs {
        if scalar_ops::is_one(s) {
            return lhs;
        }
    }
    try_fold_binary(lhs, rhs, scalar_ops::mul, Node::Mul)
}

pub fn div_edges(lhs: Edge, rhs: Edge) -> Edge {
    if let (Atom::Scalar(a), Atom::Scalar(b)) = (lhs, rhs) {
        if let Some(result) = scalar_ops::div(a, b) {
            return Atom::Scalar(result);
        }
    }
    Atom::Node(alloc(Node::Div(lhs, rhs)))
}

pub fn neg_edge(inner: Edge) -> Edge {
    try_fold_unary(inner, scalar_ops::neg, Node::Neg)
}

pub fn inv_edge(inner: Edge) -> Edge {
    match inner {
        Atom::Scalar(a) => {
            if let Some(result) = scalar_ops::inv(a) {
                Atom::Scalar(result)
            } else {
                Atom::Node(alloc(Node::Inv(inner)))
            }
        }
        Atom::Node(_) => Atom::Node(alloc(Node::Inv(inner))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scalar_ops;

    #[test]
    fn session_lifecycle() {
        let session = ArenaSession::new();
        assert_eq!(session.node_count(), 0);

        let id = alloc(Node::Var {
            index: 0,
            name: "x".into(),
        });
        assert_eq!(id.0, 0);
        assert_eq!(session.node_count(), 1);

        drop(session);
        // After drop, arena is cleared
        let session2 = ArenaSession::new();
        assert_eq!(session2.node_count(), 0);
    }

    #[test]
    fn constant_folding_add() {
        let _session = ArenaSession::new();
        let a = scalar_edge(scalar_ops::from_u64(3));
        let b = scalar_edge(scalar_ops::from_u64(5));
        let result = add_edges(a, b);
        assert_eq!(result, Atom::Scalar(scalar_ops::from_u64(8)));
        assert_eq!(node_count(), 0); // no node allocated
    }

    #[test]
    fn constant_folding_mul() {
        let _session = ArenaSession::new();
        let a = scalar_edge(scalar_ops::from_u64(7));
        let b = scalar_edge(scalar_ops::from_u64(6));
        let result = mul_edges(a, b);
        assert_eq!(result, Atom::Scalar(scalar_ops::from_u64(42)));
    }

    #[test]
    fn identity_optimizations() {
        let _session = ArenaSession::new();
        let var = Atom::Node(alloc(Node::Var {
            index: 0,
            name: "x".into(),
        }));
        let zero = scalar_edge(scalar_ops::ZERO);
        let one = scalar_edge(scalar_ops::ONE);

        // x + 0 = x
        assert_eq!(add_edges(var, zero), var);
        // 0 + x = x
        assert_eq!(add_edges(zero, var), var);
        // x * 1 = x
        assert_eq!(mul_edges(var, one), var);
        // 1 * x = x
        assert_eq!(mul_edges(one, var), var);
        // x * 0 = 0
        assert_eq!(mul_edges(var, zero), Atom::Scalar(scalar_ops::ZERO));
        // x - 0 = x
        assert_eq!(sub_edges(var, zero), var);
    }

    #[test]
    fn non_constant_creates_node() {
        let _session = ArenaSession::new();
        let var = Atom::Node(alloc(Node::Var {
            index: 0,
            name: "x".into(),
        }));
        let scalar = scalar_edge(scalar_ops::from_u64(3));
        let result = add_edges(var, scalar);
        // Should be a node, not a scalar
        assert!(matches!(result, Atom::Node(_)));
        assert_eq!(node_count(), 2); // var + add
    }

    #[test]
    fn neg_constant_folds() {
        let _session = ArenaSession::new();
        let a = scalar_edge(scalar_ops::from_u64(5));
        let result = neg_edge(a);
        let expected = scalar_ops::neg(scalar_ops::from_u64(5));
        assert_eq!(result, Atom::Scalar(expected));
    }

    #[test]
    fn snapshot_captures_state() {
        let session = ArenaSession::new();
        let _ = alloc(Node::Var {
            index: 0,
            name: "a".into(),
        });
        let _ = alloc(Node::Var {
            index: 1,
            name: "b".into(),
        });
        let nodes = session.snapshot();
        assert_eq!(nodes.len(), 2);
    }
}
