// =============================================================================
// Imports
// =============================================================================

use std::cell::RefCell;
use std::cmp::max;
use std::collections::HashMap;
use std::fmt::{self};
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::{OnceLock, RwLock};

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError, Valid};
use ark_std::{One, Zero};
use serde::{Deserialize, Serialize};

use jolt_core::field::{FieldOps, JoltField};
use jolt_core::transcripts::AppendToTranscript;

#[cfg(test)]
use crate::util::Environment;
use crate::util::LetBinderIndex;

// =============================================================================
// Constants
// =============================================================================

/// BN254 scalar field modulus: 21888242871839275222246405745257275088548364400416034343698204186575808495617
const BN254_MODULUS: [u64; 4] = [
    0x43E1F593F0000001,
    0x2833E84879B97091,
    0xB85045B68181585D,
    0x30644E72E131A029,
];

/// Zero scalar: [0, 0, 0, 0]
const SCALAR_ZERO: Scalar = [0, 0, 0, 0];
/// One scalar: [1, 0, 0, 0]
const SCALAR_ONE: Scalar = [1, 0, 0, 0];

const CSE_PREFIX: &str = "cse";

/// This guides the extractor as to the granularity at which to output definitions for common
/// sub-expressions.
/// A low depth will produce many more intermediate definitions, but each definition will be for a
/// smaller term.  There may be repeated sub-terms of depth less than the threshold.
///
/// For instance, starting with the expression ((a + b) * (a + b)) * ((a + b) / (a + b)) might yield
///   v1     = a + b
///   v2     = v1 * v1
///   v3     = v1 / v1
///   result = v2 * v3
/// at threshold 1, versus:
///   v1     = (a + b) * (a + b)
///   v2     = (a + b) / (a + b)
///   result = v1 * v2
/// at threshold 2, fewer intermediates, but small terms get repeated.
// NOTE: For gnark-transpiler, CSE doesn't help much because constraints don't share
// actual AST nodes (they're constructed separately). For zkLean/Lean output where
// there's a single large AST, CSE with threshold 4 helps.
const CSE_DEPTH_THRESHOLD: usize = 4;

// =============================================================================
// Type aliases
// =============================================================================

/// A 256-bit scalar value represented as 4 u64 limbs in little-endian order.
/// Value = limb0 + limb1*2^64 + limb2*2^128 + limb3*2^192
pub type Scalar = [u64; 4];

type Index = u16;

type NodeId = usize;

pub type DefaultMleAst = MleAst;

/// This maps a given `Node` hash to a list of the nodes we've already seen with the same hash, and
/// what let-binder index they were given.  When no collisions happen, each bucket should have
/// exactly one element in the vector.  The vector deals with collisions.
///
/// To find what binder to use, you should hash your node, then traverse the vector at that key
/// checking for equality against the nodes there.
pub type Bindings = HashMap<u64, Vec<(Node, LetBinderIndex)>>;

// =============================================================================
// Global arena
// =============================================================================

static NODE_ARENA: OnceLock<RwLock<Vec<Node>>> = OnceLock::new();

fn node_arena() -> &'static RwLock<Vec<Node>> {
    NODE_ARENA.get_or_init(|| RwLock::new(Vec::new()))
}

pub fn insert_node(node: Node) -> NodeId {
    let arena = node_arena();
    let mut guard = arena.write().expect("node arena poisoned");
    let id = guard.len();
    guard.push(node);
    id
}

pub fn get_node(id: NodeId) -> Node {
    let arena = node_arena();
    let guard = arena.read().expect("node arena poisoned");
    guard.get(id).copied().expect("invalid node reference")
}

fn edge_for_root(root: NodeId) -> Edge {
    match get_node(root) {
        Node::Atom(atom) => Edge::Atom(atom),
        _ => Edge::NodeRef(root),
    }
}

// =============================================================================
// Thread-local storage for Transcript trait integration
// =============================================================================
//
// These thread-locals enable MleAst to work with jolt-core's generic Transcript trait.
// The Transcript trait uses `F: JoltField` with methods like:
//   - `append_scalar<F>(&mut self, scalar: &F)` - calls F::serialize
//   - `challenge_scalar<F>(&mut self) -> F` - calls F::from_bytes
//
// Since MleAst implements JoltField but serialize/from_bytes don't make semantic sense
// for ASTs, we use thread-local storage to tunnel the actual MleAst values through
// these trait boundaries.
//
// This will be used by PoseidonAstTranscript (in gnark-transpiler) to build symbolic
// AST nodes for Poseidon hash operations during verifier transpilation.
// =============================================================================

thread_local! {
    static PENDING_CHALLENGE: RefCell<Option<MleAst>> = const { RefCell::new(None) };
}

/// Set a pending challenge that will be returned by the next MleAst::from_bytes call.
/// Called by PoseidonAstTranscript::challenge_scalar before returning.
pub fn set_pending_challenge(challenge: MleAst) {
    PENDING_CHALLENGE.with(|cell| {
        *cell.borrow_mut() = Some(challenge);
    });
}

/// Take the pending challenge (if any).
fn take_pending_challenge() -> Option<MleAst> {
    PENDING_CHALLENGE.with(|cell| cell.borrow_mut().take())
}

thread_local! {
    static PENDING_APPEND: RefCell<Option<MleAst>> = const { RefCell::new(None) };
}

/// Set a pending MleAst value that will be retrieved by PoseidonAstTranscript::append_scalar.
/// Called by MleAst::serialize_with_mode.
pub fn set_pending_append(value: MleAst) {
    PENDING_APPEND.with(|cell| {
        *cell.borrow_mut() = Some(value);
    });
}

/// Take the pending append value (if any).
/// Called by PoseidonAstTranscript::append_scalar to get the actual MleAst.
pub fn take_pending_append() -> Option<MleAst> {
    PENDING_APPEND.with(|cell| cell.borrow_mut().take())
}

thread_local! {
    static PENDING_COMMITMENT_CHUNKS: RefCell<Option<Vec<MleAst>>> = const { RefCell::new(None) };
}

/// Set pending commitment chunks for PoseidonAstTranscript::append_serializable.
/// Called by AstCommitment::serialize_with_mode.
pub fn set_pending_commitment_chunks(chunks: Vec<MleAst>) {
    PENDING_COMMITMENT_CHUNKS.with(|cell| {
        *cell.borrow_mut() = Some(chunks);
    });
}

/// Take the pending commitment chunks (if any).
/// Called by PoseidonAstTranscript::append_serializable to get the 12 MleAst chunks.
pub fn take_pending_commitment_chunks() -> Option<Vec<MleAst>> {
    PENDING_COMMITMENT_CHUNKS.with(|cell| cell.borrow_mut().take())
}

// =============================================================================
// Symbolic constraint accumulation for transpilation
// =============================================================================

thread_local! {
    /// Accumulated constraints during symbolic execution.
    /// Each constraint is an MleAst that should equal zero.
    static SYMBOLIC_CONSTRAINTS: RefCell<Vec<MleAst>> = const { RefCell::new(Vec::new()) };

    /// Flag to enable constraint accumulation mode.
    /// When true, PartialEq comparisons register constraints instead of comparing NodeIds.
    static CONSTRAINT_MODE: RefCell<bool> = const { RefCell::new(false) };
}

/// Enable constraint accumulation mode.
/// In this mode, `MleAst == MleAst` registers `(lhs - rhs) == 0` as a constraint
/// and returns `true` to allow verification to continue.
pub fn enable_constraint_mode() {
    CONSTRAINT_MODE.with(|cell| {
        *cell.borrow_mut() = true;
    });
}

/// Disable constraint accumulation mode.
/// After this, `MleAst == MleAst` will panic instead of registering constraints.
pub fn disable_constraint_mode() {
    CONSTRAINT_MODE.with(|cell| {
        *cell.borrow_mut() = false;
    });
}

/// Check if constraint mode is enabled.
pub fn is_constraint_mode() -> bool {
    CONSTRAINT_MODE.with(|cell| *cell.borrow())
}

/// Return the number of accumulated constraints without taking them.
pub fn num_constraints() -> usize {
    SYMBOLIC_CONSTRAINTS.with(|cell| cell.borrow().len())
}

/// Take all accumulated constraints, clearing the list.
pub fn take_constraints() -> Vec<MleAst> {
    SYMBOLIC_CONSTRAINTS.with(|cell| cell.borrow_mut().drain(..).collect())
}

/// Add a constraint that should equal zero.
fn add_constraint(constraint: MleAst) {
    SYMBOLIC_CONSTRAINTS.with(|cell| {
        cell.borrow_mut().push(constraint);
    });
}

// =============================================================================
// Core types
// =============================================================================

/// An atomic (var or const) AST element
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum Atom {
    /// A constant value.
    Scalar(Scalar),
    /// A variable, represented by an index into a register of variables
    Var(Index),
    /// A let-bound variable, used for common sub-expression elimination
    NamedVar(LetBinderIndex),
}

impl Atom {
    #[cfg(test)]
    fn evaluate<F: JoltField>(&self, env: &Environment<F>) -> F {
        match self {
            Self::Scalar(value) => {
                // Convert [u64; 4] to F
                // value = limb0 + limb1*2^64 + limb2*2^128 + limb3*2^192
                // For test purposes, we only support values that fit in u128
                assert!(
                    value[2] == 0 && value[3] == 0,
                    "Scalar too large for test evaluation"
                );
                let val_u128 = value[0] as u128 + ((value[1] as u128) << 64);
                F::from_u128(val_u128)
            }
            Self::Var(index) => env.vars[*index as usize],
            Self::NamedVar(index) => *env
                .let_bindings
                .get(index)
                .expect("unregistered let-bound variable"),
        }
    }
}

/// Either an index into the arena, or an atomic (var or const) element.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum Edge {
    /// An atomic (var or const) AST element.
    Atom(Atom),
    /// A reference to a node in the arena.
    NodeRef(NodeId),
}

/// A node for a polynomial AST. Children are represented by node IDs into the global arena.
#[derive(Debug, Hash, PartialEq, Eq, Clone, Copy, Serialize, Deserialize)]
pub enum Node {
    /// An atomic (var or const) AST element. This should only be used for MLE's with a single
    /// node.
    Atom(Atom),
    /// The negation of a node (from zklean base, unused by Jolt transpiler)
    Neg(Edge),
    /// The multiplicative inverse of a node
    Inv(Edge),
    /// The sum of two nodes
    Add(Edge, Edge),
    /// The product of two nodes
    Mul(Edge, Edge),
    /// The difference between the first and second nodes
    Sub(Edge, Edge),
    /// The quotient between the first and second nodes (from zklean base, unused by Jolt transpiler)
    /// NOTE: No div-by-zero checks are performed here
    Div(Edge, Edge),
    /// Poseidon hash with 3 inputs (state, n_rounds, data).
    /// Matches jolt-core PoseidonTranscript which uses width-3 Poseidon.
    Poseidon(Edge, Edge, Edge),
    /// Byte-reverse a field element.
    /// Transforms: serialize(x) as LE bytes -> reverse -> from_le_bytes_mod_order.
    /// This matches PoseidonTranscript::append_scalar which reverses bytes for EVM compatibility.
    ByteReverse(Edge),
    /// Truncate to 128 bits and byte-reverse, then shift by 2^128.
    /// Used for challenge_scalar_optimized which produces F::Challenge (MontU128Challenge).
    /// Transforms: take low 16 bytes (LE) -> reverse -> from_bytes -> multiply by 2^128.
    /// The 2^128 shift matches MontU128Challenge internal layout.
    Truncate128Reverse(Edge),
    /// Truncate to 128 bits and byte-reverse WITHOUT shifting.
    /// Used for challenge_scalar which produces F (raw field element).
    /// Transforms: take low 16 bytes (LE) -> reverse -> from_bytes.
    Truncate128(Edge),
    /// Transform for PoseidonTranscript::append_u64.
    ///
    /// Computes: bswap64(x) * 2^192
    ///
    /// This matches PoseidonTranscript::raw_append_u64 which:
    /// 1. Packs u64 x into bytes 24-31 of a 32-byte array using x.to_be_bytes()
    /// 2. Interprets the 32 bytes as a little-endian field element
    ///
    /// The result is bswap64(x) * 2^192, not just x * 2^192.
    AppendU64Transform(Edge),
}

/// An AST intended for representing an MLE computation (although it will actually work for any
/// multivariate polynomial). The nodes are stored in a global arena, which allows each AST handle
/// to remain [`Copy`] and [`Sized`] while supporting unbounded growth of the underlying graph.
#[derive(Debug, PartialOrd, Ord, Clone, Copy)]
pub struct MleAst {
    /// Index of the root node in the arena.
    /// nodes: [ ]
    root: NodeId,
    /// Name of the register this MLE is evaluated over.
    // TODO: Support multiple registers?
    reg_name: Option<char>,
}

// =============================================================================
// impl MleAst (inherent methods)
// =============================================================================

impl MleAst {
    fn new_scalar(scalar: Scalar) -> Self {
        let root = insert_node(Node::Atom(Atom::Scalar(scalar)));
        Self {
            root,
            reg_name: None,
        }
    }

    fn new_var(name: char, index: Index) -> Self {
        let root = insert_node(Node::Atom(Atom::Var(index)));
        Self {
            root,
            reg_name: Some(name),
        }
    }

    fn merge_reg_name(&mut self, other: Option<char>) {
        if let (Some(lhs), Some(rhs)) = (self.reg_name, other) {
            assert_eq!(lhs, rhs, "Multiple registers not supported");
        }
        if self.reg_name.is_none() {
            self.reg_name = other;
        }
    }

    /// Create a new root node in the form of a unitary operator.
    fn unop(&mut self, constructor: impl FnOnce(Edge) -> Node) {
        let edge = edge_for_root(self.root);
        self.root = insert_node(constructor(edge));
    }

    /// Create a new root node in the form of a binary operator.
    fn binop(&mut self, constructor: impl FnOnce(Edge, Edge) -> Node, rhs: &Self) {
        self.merge_reg_name(rhs.reg_name);
        let lhs_edge = edge_for_root(self.root);
        let rhs_edge = edge_for_root(rhs.root);
        self.root = insert_node(constructor(lhs_edge, rhs_edge));
    }

    /// Create a variable from an index (for symbolic execution).
    pub fn from_var(index: u16) -> Self {
        Self::new_var('v', index)
    }

    /// Create an MleAst from an existing node ID.
    ///
    /// This is useful for codegen when you have a node ID but need to use
    /// MleAst methods like `is_constant()` or `try_evaluate_constant()`.
    pub fn from_node_id(node_id: NodeId) -> Self {
        Self {
            root: node_id,
            reg_name: None,
        }
    }

    /// Get the root node ID for this AST.
    pub fn root(&self) -> NodeId {
        self.root
    }

    /// Poseidon hash with 3 inputs (state, n_rounds, data).
    /// Matches jolt-core PoseidonTranscript structure.
    pub fn poseidon(state: &Self, n_rounds: &Self, data: &Self) -> Self {
        let state_edge = edge_for_root(state.root);
        let rounds_edge = edge_for_root(n_rounds.root);
        let data_edge = edge_for_root(data.root);
        let root = insert_node(Node::Poseidon(state_edge, rounds_edge, data_edge));
        Self {
            root,
            reg_name: state.reg_name.or(n_rounds.reg_name).or(data.reg_name),
        }
    }

    /// Byte-reverse a field element.
    /// Transforms: serialize(x) as LE bytes -> reverse -> from_le_bytes_mod_order.
    pub fn byte_reverse(input: &Self) -> Self {
        let edge = edge_for_root(input.root);
        let root = insert_node(Node::ByteReverse(edge));
        Self {
            root,
            reg_name: input.reg_name,
        }
    }

    /// Truncate to 128 bits and byte-reverse, then shift by 2^128.
    /// Used for challenge_scalar_optimized (produces MontU128Challenge).
    /// Transforms: take low 16 bytes (LE) -> reverse -> from_bytes -> multiply by 2^128.
    pub fn truncate_128_reverse(input: &Self) -> Self {
        let edge = edge_for_root(input.root);
        let root = insert_node(Node::Truncate128Reverse(edge));
        Self {
            root,
            reg_name: input.reg_name,
        }
    }

    /// Truncate to 128 bits and byte-reverse WITHOUT shifting.
    /// Used for challenge_scalar (produces raw F field element).
    /// Transforms: take low 16 bytes (LE) -> reverse -> from_bytes.
    pub fn truncate_128(input: &Self) -> Self {
        let edge = edge_for_root(input.root);
        let root = insert_node(Node::Truncate128(edge));
        Self {
            root,
            reg_name: input.reg_name,
        }
    }

    /// Transform for PoseidonTranscript::append_u64.
    ///
    /// Computes: bswap64(x) * 2^192
    ///
    /// This matches PoseidonTranscript::raw_append_u64 which:
    /// 1. Packs u64 x into bytes 24-31 of a 32-byte array using x.to_be_bytes()
    /// 2. Interprets the 32 bytes as a little-endian field element
    ///
    /// The result is bswap64(x) * 2^192, not just x * 2^192.
    pub fn append_u64_transform(input: &Self) -> Self {
        let edge = edge_for_root(input.root);
        let root = insert_node(Node::AppendU64Transform(edge));
        Self {
            root,
            reg_name: input.reg_name,
        }
    }

    /// Returns true if this AST represents a constant value (no variables).
    ///
    /// An expression is constant if it contains only scalars and operations on scalars,
    /// with no `Var` or `NamedVar` atoms anywhere in the tree.
    pub fn is_constant(&self) -> bool {
        is_node_constant(self.root)
    }

    /// If this AST is constant, evaluate it and return the scalar value.
    ///
    /// Returns `None` if the AST contains any variables.
    /// Returns `Some([u64; 4])` with the computed constant value.
    ///
    /// Note: This uses modular arithmetic with the BN254 scalar field modulus.
    pub fn try_evaluate_constant(&self) -> Option<Scalar> {
        if !self.is_constant() {
            return None;
        }
        Some(evaluate_constant_node(self.root))
    }
}

// =============================================================================
// Scalar arithmetic helpers
// =============================================================================

/// Convert a Scalar to a decimal string.
fn scalar_to_decimal_string(limbs: &Scalar) -> String {
    // Handle zero case
    if *limbs == [0, 0, 0, 0] {
        return "0".to_string();
    }

    // Convert [u64; 4] to BigUint for decimal formatting
    // Value = limb0 + limb1*2^64 + limb2*2^128 + limb3*2^192
    use num_bigint::BigUint;

    let mut value = BigUint::from(limbs[3]);
    value = (value << 64) + limbs[2];
    value = (value << 64) + limbs[1];
    value = (value << 64) + limbs[0];

    value.to_string()
}

/// Add two 256-bit numbers with modular reduction
pub fn scalar_add_mod(a: Scalar, b: Scalar) -> Scalar {
    let mut result = [0u64; 4];
    let mut carry = 0u128;

    for i in 0..4 {
        let sum = a[i] as u128 + b[i] as u128 + carry;
        result[i] = sum as u64;
        carry = sum >> 64;
    }

    // Reduce mod p if needed
    if carry > 0 || scalar_ge(&result, &BN254_MODULUS) {
        scalar_sub_no_borrow(&result, &BN254_MODULUS)
    } else {
        result
    }
}

/// Subtract two 256-bit numbers with modular reduction
pub fn scalar_sub_mod(a: Scalar, b: Scalar) -> Scalar {
    // a - b mod p = a + (p - b) mod p
    let neg_b = scalar_neg_mod(b);
    scalar_add_mod(a, neg_b)
}

/// Negate a scalar: -a mod p = p - a
pub fn scalar_neg_mod(a: Scalar) -> Scalar {
    if a == SCALAR_ZERO {
        return SCALAR_ZERO;
    }
    scalar_sub_no_borrow(&BN254_MODULUS, &a)
}

/// Multiply two 256-bit numbers with modular reduction (simplified)
pub fn scalar_mul_mod(a: Scalar, b: Scalar) -> Scalar {
    // For full correctness, we'd need Montgomery multiplication
    // For now, use a simple approach via BigUint
    use num_bigint::BigUint;

    let a_big = BigUint::from_slice(&[
        a[0] as u32,
        (a[0] >> 32) as u32,
        a[1] as u32,
        (a[1] >> 32) as u32,
        a[2] as u32,
        (a[2] >> 32) as u32,
        a[3] as u32,
        (a[3] >> 32) as u32,
    ]);
    let b_big = BigUint::from_slice(&[
        b[0] as u32,
        (b[0] >> 32) as u32,
        b[1] as u32,
        (b[1] >> 32) as u32,
        b[2] as u32,
        (b[2] >> 32) as u32,
        b[3] as u32,
        (b[3] >> 32) as u32,
    ]);
    let p_big = BigUint::from_slice(&[
        BN254_MODULUS[0] as u32,
        (BN254_MODULUS[0] >> 32) as u32,
        BN254_MODULUS[1] as u32,
        (BN254_MODULUS[1] >> 32) as u32,
        BN254_MODULUS[2] as u32,
        (BN254_MODULUS[2] >> 32) as u32,
        BN254_MODULUS[3] as u32,
        (BN254_MODULUS[3] >> 32) as u32,
    ]);

    let result = (a_big * b_big) % p_big;
    let digits = result.to_u64_digits();

    let mut out = [0u64; 4];
    for (i, &d) in digits.iter().take(4).enumerate() {
        out[i] = d;
    }
    out
}

/// Compare two 256-bit numbers: returns true if a >= b
fn scalar_ge(a: &Scalar, b: &Scalar) -> bool {
    for i in (0..4).rev() {
        if a[i] > b[i] {
            return true;
        }
        if a[i] < b[i] {
            return false;
        }
    }
    true // equal
}

/// Subtract b from a, assuming a >= b (no modular reduction)
fn scalar_sub_no_borrow(a: &Scalar, b: &Scalar) -> Scalar {
    let mut result = [0u64; 4];
    let mut borrow = 0i128;

    for i in 0..4 {
        let diff = a[i] as i128 - b[i] as i128 - borrow;
        if diff < 0 {
            result[i] = (diff + (1i128 << 64)) as u64;
            borrow = 1;
        } else {
            result[i] = diff as u64;
            borrow = 0;
        }
    }

    result
}

// =============================================================================
// Constant evaluation helpers
// =============================================================================

/// Check if an edge is constant (contains no variables)
fn is_edge_constant(edge: Edge) -> bool {
    match edge {
        Edge::Atom(Atom::Scalar(_)) => true,
        Edge::Atom(Atom::Var(_)) => false,
        Edge::Atom(Atom::NamedVar(_)) => false, // Named vars could be constants, but we can't know statically
        Edge::NodeRef(id) => is_node_constant(id),
    }
}

/// Check if a node is constant (contains no variables)
fn is_node_constant(node_id: NodeId) -> bool {
    match get_node(node_id) {
        Node::Atom(Atom::Scalar(_)) => true,
        Node::Atom(Atom::Var(_)) => false,
        Node::Atom(Atom::NamedVar(_)) => false,
        Node::Neg(e)
        | Node::Inv(e)
        | Node::ByteReverse(e)
        | Node::Truncate128Reverse(e)
        | Node::Truncate128(e)
        | Node::AppendU64Transform(e) => is_edge_constant(e),
        Node::Add(e1, e2) | Node::Mul(e1, e2) | Node::Sub(e1, e2) | Node::Div(e1, e2) => {
            is_edge_constant(e1) && is_edge_constant(e2)
        }
        Node::Poseidon(e1, e2, e3) => {
            is_edge_constant(e1) && is_edge_constant(e2) && is_edge_constant(e3)
        }
    }
}

/// Evaluate a constant edge to its scalar value
fn evaluate_constant_edge(edge: Edge) -> Scalar {
    match edge {
        Edge::Atom(Atom::Scalar(s)) => s,
        Edge::Atom(Atom::Var(_)) | Edge::Atom(Atom::NamedVar(_)) => {
            panic!("Cannot evaluate non-constant edge")
        }
        Edge::NodeRef(id) => evaluate_constant_node(id),
    }
}

/// Evaluate a constant node to its scalar value
fn evaluate_constant_node(node_id: NodeId) -> Scalar {
    match get_node(node_id) {
        Node::Atom(Atom::Scalar(s)) => s,
        Node::Atom(Atom::Var(_)) | Node::Atom(Atom::NamedVar(_)) => {
            panic!("Cannot evaluate non-constant node")
        }
        Node::Neg(e) => scalar_neg_mod(evaluate_constant_edge(e)),
        Node::Add(e1, e2) => scalar_add_mod(evaluate_constant_edge(e1), evaluate_constant_edge(e2)),
        Node::Sub(e1, e2) => scalar_sub_mod(evaluate_constant_edge(e1), evaluate_constant_edge(e2)),
        Node::Mul(e1, e2) => scalar_mul_mod(evaluate_constant_edge(e1), evaluate_constant_edge(e2)),
        Node::Inv(_) | Node::Div(_, _) => {
            panic!("Modular inverse not implemented for constant evaluation")
        }
        Node::Poseidon(_, _, _)
        | Node::ByteReverse(_)
        | Node::Truncate128Reverse(_)
        | Node::Truncate128(_)
        | Node::AppendU64Transform(_) => {
            panic!("Hash/transform operations cannot be evaluated as constants")
        }
    }
}

// =============================================================================
// Test-only evaluation helpers
// =============================================================================

#[cfg(test)]
fn evaluate_edge<F: JoltField>(edge: Edge, env: &Environment<F>) -> F {
    match edge {
        Edge::Atom(atom) => atom.evaluate(env),
        Edge::NodeRef(node) => evaluate_node(node, env),
    }
}

#[cfg(test)]
fn evaluate_node<F: JoltField>(node: NodeId, env: &Environment<F>) -> F {
    match get_node(node) {
        Node::Atom(atom) => atom.evaluate(env),
        Node::Neg(edge) => -evaluate_edge(edge, env),
        Node::Inv(edge) => F::one() / evaluate_edge(edge, env),
        Node::Add(e1, e2) => evaluate_edge(e1, env) + evaluate_edge(e2, env),
        Node::Mul(e1, e2) => evaluate_edge(e1, env) * evaluate_edge(e2, env),
        Node::Sub(e1, e2) => evaluate_edge(e1, env) - evaluate_edge(e2, env),
        Node::Div(e1, e2) => evaluate_edge(e1, env) / evaluate_edge(e2, env),
        Node::Poseidon(_, _, _)
        | Node::ByteReverse(_)
        | Node::Truncate128Reverse(_)
        | Node::Truncate128(_)
        | Node::AppendU64Transform(_) => {
            // Hash/transform nodes are for circuit generation only, not field evaluation
            unreachable!("Hash/transform nodes should not appear in zklean-extractor tests")
        }
    }
}

// =============================================================================
// Formatting
// =============================================================================

struct FormattingData<'a> {
    prefix: &'a String,
    reg_name: Option<char>,
}

fn fmt_atom(f: &mut fmt::Formatter<'_>, fmt_data: &FormattingData<'_>, atom: Atom) -> fmt::Result {
    match atom {
        Atom::Scalar(value) => write!(f, "{}", scalar_to_decimal_string(&value))?,
        Atom::Var(index) => {
            let name = fmt_data
                .reg_name
                .expect("unreachable: register name missing in var");
            write!(f, "{name}[{index}]")?;
        }
        Atom::NamedVar(index) => write!(f, "{}{index} x", fmt_data.prefix)?,
    }

    Ok(())
}

fn fmt_edge(
    f: &mut fmt::Formatter<'_>,
    fmt_data: &FormattingData<'_>,
    edge: Edge,
    group: bool,
) -> fmt::Result {
    match edge {
        Edge::Atom(atom) => fmt_atom(f, fmt_data, atom),
        Edge::NodeRef(node) => fmt_node(f, fmt_data, node, group),
    }
}

fn fmt_node(
    f: &mut fmt::Formatter<'_>,
    fmt_data: &FormattingData<'_>,
    node: NodeId,
    group: bool,
) -> fmt::Result {
    match get_node(node) {
        Node::Atom(atom) => fmt_atom(f, fmt_data, atom),
        Node::Neg(edge) => {
            write!(f, "-")?;
            fmt_edge(f, fmt_data, edge, true)
        }
        Node::Inv(edge) => {
            write!(f, "1 / ")?;
            fmt_edge(f, fmt_data, edge, true)
        }
        Node::Add(e1, e2) => {
            if group {
                write!(f, "(")?;
            }
            fmt_edge(f, fmt_data, e1, false)?;
            write!(f, " + ")?;
            fmt_edge(f, fmt_data, e2, false)?;
            if group {
                write!(f, ")")?;
            }
            Ok(())
        }
        Node::Mul(e1, e2) => {
            fmt_edge(f, fmt_data, e1, true)?;
            write!(f, " * ")?;
            fmt_edge(f, fmt_data, e2, true)
        }
        Node::Sub(e1, e2) => {
            if group {
                write!(f, "(")?;
            }
            fmt_edge(f, fmt_data, e1, false)?;
            write!(f, " - ")?;
            fmt_edge(f, fmt_data, e2, true)?;
            if group {
                write!(f, ")")?;
            }
            Ok(())
        }
        Node::Div(e1, e2) => {
            fmt_edge(f, fmt_data, e1, true)?;
            write!(f, " / ")?;
            fmt_edge(f, fmt_data, e2, true)
        }
        Node::Poseidon(e1, e2, e3) => {
            write!(f, "poseidon(")?;
            fmt_edge(f, fmt_data, e1, false)?;
            write!(f, ", ")?;
            fmt_edge(f, fmt_data, e2, false)?;
            write!(f, ", ")?;
            fmt_edge(f, fmt_data, e3, false)?;
            write!(f, ")")
        }
        Node::ByteReverse(edge) => {
            write!(f, "byte_reverse(")?;
            fmt_edge(f, fmt_data, edge, false)?;
            write!(f, ")")
        }
        Node::Truncate128Reverse(edge) => {
            write!(f, "truncate_128_reverse(")?;
            fmt_edge(f, fmt_data, edge, false)?;
            write!(f, ")")
        }
        Node::Truncate128(edge) => {
            write!(f, "truncate_128(")?;
            fmt_edge(f, fmt_data, edge, false)?;
            write!(f, ")")
        }
        Node::AppendU64Transform(edge) => {
            write!(f, "append_u64_transform(")?;
            fmt_edge(f, fmt_data, edge, false)?;
            write!(f, ")")
        }
    }
}

// =============================================================================
// Common Subexpression Elimination (CSE)
// =============================================================================

pub fn compute_hash<T: Hash>(value: &T) -> u64 {
    let mut h = DefaultHasher::new();
    value.hash(&mut h);
    h.finish()
}

fn node_depth(node: Node) -> usize {
    fn edge_depth(edge: Edge) -> usize {
        match edge {
            Edge::Atom(_) => 0,
            Edge::NodeRef(n) => node_depth(get_node(n)),
        }
    }
    match node {
        Node::Atom(_) => 0,
        Node::Neg(e)
        | Node::Inv(e)
        | Node::ByteReverse(e)
        | Node::Truncate128Reverse(e)
        | Node::Truncate128(e)
        | Node::AppendU64Transform(e) => 1 + edge_depth(e),
        Node::Add(e1, e2) | Node::Mul(e1, e2) | Node::Sub(e1, e2) | Node::Div(e1, e2) => {
            1 + max(edge_depth(e1), edge_depth(e2))
        }
        Node::Poseidon(e1, e2, e3) => 1 + max(edge_depth(e1), max(edge_depth(e2), edge_depth(e3))),
    }
}

/// Perform common subexpression elimination on an AST.
///
/// Returns a tuple of (bindings, new_root) where:
/// - bindings: Vec of nodes that should be hoisted as named variables (cse_0, cse_1, ...)
/// - new_root: The transformed root node with common subexpressions replaced by NamedVar references
///
/// The CSE_DEPTH_THRESHOLD controls granularity - subexpressions below this depth
/// are not hoisted (to avoid excessive small definitions).
pub fn common_subexpression_elimination(node: Node) -> (Vec<Node>, Node) {
    /// Assumption: the sub-nodes have already been CSE-d
    fn register(bindings: &mut Bindings, nodes: &mut Vec<Node>, node: Node) -> Node {
        let node_hash = compute_hash(&node);
        let depth = node_depth(node);

        if let Some(v) = bindings.get(&node_hash) {
            if let Some((_, i)) = v.iter().find(|(n, _)| n == &node) {
                return Node::Atom(Atom::NamedVar(*i));
            }
        }
        if depth < CSE_DEPTH_THRESHOLD {
            return node;
        }
        // Registering a new node
        let index = nodes.len();
        bindings
            .entry(node_hash)
            .and_modify(|v| {
                v.push((node, index));
            })
            .or_insert(vec![(node, index)]);
        nodes.push(node);
        Node::Atom(Atom::NamedVar(index))
    }

    fn aux_node(bindings: &mut Bindings, nodes: &mut Vec<Node>, node: Node) -> Node {
        match node {
            Node::Atom(_) => node,
            Node::Neg(e) => {
                let cse_e = aux_edge(bindings, nodes, e);
                register(bindings, nodes, Node::Neg(cse_e))
            }
            Node::Inv(e) => {
                let cse_e = aux_edge(bindings, nodes, e);
                register(bindings, nodes, Node::Inv(cse_e))
            }
            Node::Add(e1, e2) => {
                let cse_e1 = aux_edge(bindings, nodes, e1);
                let cse_e2 = aux_edge(bindings, nodes, e2);
                register(bindings, nodes, Node::Add(cse_e1, cse_e2))
            }
            Node::Mul(e1, e2) => {
                let cse_e1 = aux_edge(bindings, nodes, e1);
                let cse_e2 = aux_edge(bindings, nodes, e2);
                register(bindings, nodes, Node::Mul(cse_e1, cse_e2))
            }
            Node::Sub(e1, e2) => {
                let cse_e1 = aux_edge(bindings, nodes, e1);
                let cse_e2 = aux_edge(bindings, nodes, e2);
                register(bindings, nodes, Node::Sub(cse_e1, cse_e2))
            }
            Node::Div(e1, e2) => {
                let cse_e1 = aux_edge(bindings, nodes, e1);
                let cse_e2 = aux_edge(bindings, nodes, e2);
                register(bindings, nodes, Node::Div(cse_e1, cse_e2))
            }
            Node::Poseidon(e1, e2, e3) => {
                let cse_e1 = aux_edge(bindings, nodes, e1);
                let cse_e2 = aux_edge(bindings, nodes, e2);
                let cse_e3 = aux_edge(bindings, nodes, e3);
                register(bindings, nodes, Node::Poseidon(cse_e1, cse_e2, cse_e3))
            }
            Node::ByteReverse(e) => {
                let cse_e = aux_edge(bindings, nodes, e);
                register(bindings, nodes, Node::ByteReverse(cse_e))
            }
            Node::Truncate128Reverse(e) => {
                let cse_e = aux_edge(bindings, nodes, e);
                register(bindings, nodes, Node::Truncate128Reverse(cse_e))
            }
            Node::Truncate128(e) => {
                let cse_e = aux_edge(bindings, nodes, e);
                register(bindings, nodes, Node::Truncate128(cse_e))
            }
            Node::AppendU64Transform(e) => {
                let cse_e = aux_edge(bindings, nodes, e);
                register(bindings, nodes, Node::AppendU64Transform(cse_e))
            }
        }
    }

    fn aux_edge(bindings: &mut Bindings, nodes: &mut Vec<Node>, edge: Edge) -> Edge {
        match edge {
            Edge::Atom(_) => edge,
            Edge::NodeRef(node) => {
                Edge::NodeRef(insert_node(aux_node(bindings, nodes, get_node(node))))
            }
        }
    }

    let mut bindings = HashMap::new();
    let mut nodes = Vec::new();
    let new_node = aux_node(&mut bindings, &mut nodes, node);
    (nodes, new_node)
}

// =============================================================================
// Trait implementations for MleAst
// =============================================================================

impl crate::util::ZkLeanReprField for MleAst {
    fn register(name: char, size: usize) -> Vec<Self> {
        (0..size).map(|i| Self::new_var(name, i as Index)).collect()
    }

    /// Evaluate the computation represented by the AST over another [`JoltField`], starting at
    /// `root`, and using the variable assignments in `vars`.
    #[cfg(test)]
    fn evaluate<F: JoltField>(&self, env: &Environment<F>) -> F {
        evaluate_node(self.root, env)
    }

    /// Note: This performs common subexpression elimination for the output formula, as it tends to
    /// have many large repeated subterms.  We have found experimentally that Lean struggles with
    /// this, even when we define the common subterms as let-bindings.  Performance degrades
    /// exponentially in the number of let-bindings, which, when we moved to 64-bit formulae, made
    /// it unwieldy.
    ///
    /// Instead, we now output shared sub-formulae as top-level definitions, recovering type
    /// checking linear in the number of definitions.  See `CSE_DEPTH_THRESHOLD` to control the size
    /// of each definition.
    fn format_for_lean(
        &self,
        f: &mut fmt::Formatter<'_>,
        name: &str,
        num_variables: usize,
    ) -> fmt::Result {
        let (bindings, node) = common_subexpression_elimination(get_node(self.root));
        let node_id = insert_node(node);
        let fmt_data = FormattingData {
            prefix: &format!("{name}_{CSE_PREFIX}_"),
            reg_name: self.reg_name,
        };
        for (index, binding) in bindings.iter().enumerate() {
            write!(
                f,
                "def {}{index} [Field f] (x : Vector f {num_variables}) : f := ",
                fmt_data.prefix,
            )?;
            fmt_node(f, &fmt_data, insert_node(*binding), false)?;
            writeln!(f)?;
        }
        write!(
            f,
            "def {name} [Field f] (x : Vector f {num_variables}) : f := "
        )?;
        fmt_node(f, &fmt_data, node_id, false)?;
        writeln!(f)?;
        writeln!(f)?;
        Ok(())
    }
}

impl PartialEq for MleAst {
    fn eq(&self, other: &Self) -> bool {
        // If both are the same node, they're trivially equal
        if self.root == other.root {
            return true;
        }

        // In constraint mode, register constraint and return true
        if is_constraint_mode() {
            // Constraint: (self - other) == 0
            let diff = *self - *other;
            add_constraint(diff);
            return true;
        }

        // Normal mode: compare NodeIds (original behavior)
        false
    }
}

impl Eq for MleAst {}

impl Zero for MleAst {
    fn zero() -> Self {
        Self::new_scalar(SCALAR_ZERO)
    }

    fn is_zero(&self) -> bool {
        matches!(
            get_node(self.root),
            Node::Atom(Atom::Scalar(value)) if value == SCALAR_ZERO
        )
    }
}

impl One for MleAst {
    fn one() -> Self {
        Self::new_scalar(SCALAR_ONE)
    }

    /// Check if this MleAst represents the constant 1.
    ///
    /// IMPORTANT: This implementation checks the node structure directly
    /// instead of using `*self == Self::one()`. The default `is_one()`
    /// would trigger `PartialEq::eq`, which in constraint mode registers
    /// a constraint `(self - 1) = 0`. This caused spurious constraints
    /// involving io values (10, 55) when optimized multiplication checked
    /// if coefficients were 1.
    fn is_one(&self) -> bool {
        matches!(
            get_node(self.root),
            Node::Atom(Atom::Scalar(value)) if value == SCALAR_ONE
        )
    }
}

impl std::ops::Neg for MleAst {
    type Output = Self;

    fn neg(self) -> Self::Output {
        // Negation is 0 - x (Jolt transpiler doesn't use Node::Neg)
        Self::zero() - self
    }
}

impl std::ops::Add for MleAst {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.add(&rhs)
    }
}

impl std::ops::Sub for MleAst {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.sub(&rhs)
    }
}

impl std::ops::Mul for MleAst {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.mul(&rhs)
    }
}

impl std::ops::Div for MleAst {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self.div(&rhs)
    }
}

impl FieldOps for MleAst {}

impl std::ops::Add<&Self> for MleAst {
    type Output = Self;

    fn add(mut self, rhs: &Self) -> Self::Output {
        // Optimization: x + 0 = x, 0 + x = x
        // This prevents constant-vs-constant additions in generated code
        if self.is_zero() {
            return *rhs;
        }
        if rhs.is_zero() {
            return self;
        }
        self.binop(Node::Add, rhs);
        self
    }
}

impl std::ops::Sub<&Self> for MleAst {
    type Output = Self;

    fn sub(mut self, rhs: &Self) -> Self::Output {
        // Optimization: x - 0 = x
        // This prevents constant-vs-constant subtractions in generated code
        if rhs.is_zero() {
            return self;
        }
        // Optimization: 0 - x = -x
        if self.is_zero() {
            return -*rhs;
        }
        self.binop(Node::Sub, rhs);
        self
    }
}

impl std::ops::Mul<&Self> for MleAst {
    type Output = Self;

    fn mul(mut self, rhs: &Self) -> Self::Output {
        // Optimization: x * 0 = 0, 0 * x = 0
        // This prevents constant-vs-constant assertions in Gnark when
        // EqPolynomial::evals expands to terms multiplied by zero coefficients.
        if self.is_zero() || rhs.is_zero() {
            return Self::zero();
        }
        self.binop(Node::Mul, rhs);
        self
    }
}

impl std::ops::Div<&Self> for MleAst {
    type Output = Self;

    fn div(mut self, rhs: &Self) -> Self::Output {
        self.binop(Node::Div, rhs);
        self
    }
}

impl FieldOps<&Self, Self> for MleAst {}

impl std::ops::AddAssign for MleAst {
    fn add_assign(&mut self, rhs: Self) {
        // Optimization: x += 0 is a no-op
        if rhs.is_zero() {
            return;
        }
        // Optimization: 0 += x => self = x
        if self.is_zero() {
            *self = rhs;
            return;
        }
        self.binop(Node::Add, &rhs);
    }
}

impl<'a> std::ops::AddAssign<&'a Self> for MleAst {
    fn add_assign(&mut self, rhs: &'a Self) {
        // Optimization: x += 0 is a no-op
        if rhs.is_zero() {
            return;
        }
        // Optimization: 0 += x => self = x
        if self.is_zero() {
            *self = *rhs;
            return;
        }
        self.binop(Node::Add, rhs);
    }
}

impl std::ops::SubAssign for MleAst {
    fn sub_assign(&mut self, rhs: Self) {
        // Optimization: x -= 0 is a no-op
        if rhs.is_zero() {
            return;
        }
        // Optimization: 0 -= x => self = -x
        if self.is_zero() {
            *self = -rhs;
            return;
        }
        self.binop(Node::Sub, &rhs);
    }
}

impl<'a> std::ops::SubAssign<&'a Self> for MleAst {
    fn sub_assign(&mut self, rhs: &'a Self) {
        // Optimization: x -= 0 is a no-op
        if rhs.is_zero() {
            return;
        }
        // Optimization: 0 -= x => self = -x
        if self.is_zero() {
            *self = -*rhs;
            return;
        }
        self.binop(Node::Sub, rhs);
    }
}

impl std::ops::MulAssign for MleAst {
    fn mul_assign(&mut self, rhs: Self) {
        // Optimization: x *= 0 => x = 0, 0 *= x => stays 0
        if self.is_zero() || rhs.is_zero() {
            *self = Self::zero();
            return;
        }
        self.binop(Node::Mul, &rhs);
    }
}

impl<'a> std::ops::MulAssign<&'a Self> for MleAst {
    fn mul_assign(&mut self, rhs: &'a Self) {
        // Optimization: x *= 0 => x = 0, 0 *= x => stays 0
        if self.is_zero() || rhs.is_zero() {
            *self = Self::zero();
            return;
        }
        self.binop(Node::Mul, rhs);
    }
}

impl core::iter::Sum for MleAst {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|sum, term| sum + term)
            .unwrap_or_else(Self::zero)
    }
}

impl<'a> core::iter::Sum<&'a Self> for MleAst {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.cloned().sum()
    }
}

impl core::iter::Product for MleAst {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|product, factor| product * factor)
            .unwrap_or_else(Self::one)
    }
}

impl<'a> core::iter::Product<&'a Self> for MleAst {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.cloned().product()
    }
}

// Note: this instance prints the whole MLE as a single expression.  It can be very large, e.g. for
// 64-bit.  If you extract for Lean, you might want to use `format_for_lean` to get separate
// definitions for repeated sub-expressions.
impl fmt::Display for MleAst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let fmt_data = FormattingData {
            prefix: &String::from(""),
            reg_name: self.reg_name,
        };
        fmt_node(f, &fmt_data, self.root, false)
    }
}

impl Default for MleAst {
    fn default() -> Self {
        Self::zero()
    }
}

impl std::hash::Hash for MleAst {
    fn hash<H: std::hash::Hasher>(&self, _state: &mut H) {
        unimplemented!("hash unimplemented for MleAst")
    }
}

impl From<u128> for MleAst {
    fn from(value: u128) -> Self {
        Self::from_u128(value)
    }
}

impl<const N: usize> From<ark_ff::BigInt<N>> for MleAst {
    fn from(_value: ark_ff::BigInt<N>) -> Self {
        unimplemented!("hash unimplemented for MleAst")
    }
}

impl allocative::Allocative for MleAst {
    fn visit<'a, 'b: 'a>(&self, _visitor: &'a mut allocative::Visitor<'b>) {
        unimplemented!("Not needed for constructing ASTs");
    }
}

impl ark_std::rand::prelude::Distribution<MleAst> for ark_std::rand::distributions::Standard {
    fn sample<R: ark_std::rand::Rng + ?Sized>(&self, _rng: &mut R) -> MleAst {
        unimplemented!("Not needed for constructing ASTs");
    }
}

impl From<ark_ff::biginteger::signed_hi_32::SignedBigIntHi32<3>> for MleAst {
    fn from(_value: ark_ff::biginteger::signed_hi_32::SignedBigIntHi32<3>) -> Self {
        unimplemented!("Not needed for constructing ASTs");
    }
}

// Required for JoltField::Unreduced<N> - handles conversion from [u64; N] arrays
impl<const N: usize> From<[u64; N]> for MleAst {
    fn from(value: [u64; N]) -> Self {
        // Convert [u64; N] to [u64; 4] by copying available limbs and padding with zeros
        let mut limbs = [0u64; 4];
        let copy_len = N.min(4);
        limbs[..copy_len].copy_from_slice(&value[..copy_len]);
        Self::new_scalar(limbs)
    }
}

impl jolt_core::field::UnreducedInteger for MleAst {}

impl JoltField for MleAst {
    const NUM_BYTES: usize = 0;
    const NUM_LIMBS: usize = 0;

    const MONTGOMERY_R: Self = todo!();
    const MONTGOMERY_R_SQUARE: Self = todo!();

    type UnreducedElem = Self;
    type UnreducedMulU64 = Self;
    type UnreducedMulU128 = Self;
    type UnreducedMulU128Accum = Self;
    type UnreducedProduct = Self;
    type UnreducedProductAccum = Self;

    type Challenge = Self;
    type SmallValueLookupTables = ();

    fn random<R: rand_core::RngCore>(_rng: &mut R) -> Self {
        unimplemented!("Not needed for constructing ASTs");
    }

    fn from_bool(val: bool) -> Self {
        Self::new_scalar([val as u64, 0, 0, 0])
    }

    fn from_u8(n: u8) -> Self {
        Self::new_scalar([n as u64, 0, 0, 0])
    }

    fn from_u16(n: u16) -> Self {
        Self::new_scalar([n as u64, 0, 0, 0])
    }

    fn from_u32(n: u32) -> Self {
        Self::new_scalar([n as u64, 0, 0, 0])
    }

    fn from_u64(n: u64) -> Self {
        Self::new_scalar([n, 0, 0, 0])
    }

    fn from_i64(n: i64) -> Self {
        if n >= 0 {
            Self::new_scalar([n as u64, 0, 0, 0])
        } else {
            // For negative numbers in BN254 field: compute p - |n|
            // BN254 scalar field modulus:
            // p = 21888242871839275222246405745257275088548364400416034343698204186575808495617
            const BN254_MODULUS: [u64; 4] = [
                0x43e1f593f0000001,
                0x2833e84879b97091,
                0xb85045b68181585d,
                0x30644e72e131a029,
            ];

            let abs_n = n.unsigned_abs();
            let (l0, b0) = BN254_MODULUS[0].overflowing_sub(abs_n);
            let (l1, b1) = BN254_MODULUS[1].overflowing_sub(b0 as u64);
            let (l2, b2) = BN254_MODULUS[2].overflowing_sub(b1 as u64);
            let (l3, _) = BN254_MODULUS[3].overflowing_sub(b2 as u64);

            Self::new_scalar([l0, l1, l2, l3])
        }
    }

    fn from_u128(n: u128) -> Self {
        let low = n as u64;
        let high = (n >> 64) as u64;
        Self::new_scalar([low, high, 0, 0])
    }

    fn from_i128(n: i128) -> Self {
        if n >= 0 {
            let low = n as u64;
            let high = (n >> 64) as u64;
            Self::new_scalar([low, high, 0, 0])
        } else {
            // For negative numbers in BN254 field: compute p - |n|
            // BN254 scalar field modulus
            const BN254_MODULUS: [u64; 4] = [
                0x43e1f593f0000001,
                0x2833e84879b97091,
                0xb85045b68181585d,
                0x30644e72e131a029,
            ];

            let abs_n = n.unsigned_abs();
            let abs_low = abs_n as u64;
            let abs_high = (abs_n >> 64) as u64;

            // Compute p - |n| with borrow propagation
            let (l0, b0) = BN254_MODULUS[0].overflowing_sub(abs_low);
            let (l1, b1) = BN254_MODULUS[1].overflowing_sub(abs_high + b0 as u64);
            let (l2, b2) = BN254_MODULUS[2].overflowing_sub(b1 as u64);
            let (l3, _) = BN254_MODULUS[3].overflowing_sub(b2 as u64);

            Self::new_scalar([l0, l1, l2, l3])
        }
    }

    fn square(&self) -> Self {
        *self * self
    }

    fn from_bytes(_bytes: &[u8]) -> Self {
        // Check if there's a pending challenge from PoseidonAstTranscript
        if let Some(challenge) = take_pending_challenge() {
            return challenge;
        }
        // Fallback: create constant from bytes (for non-transpilation use)
        MleAst::from_i128(0)
    }

    fn inverse(&self) -> Option<Self> {
        if self.is_zero() {
            None
        } else {
            let mut res = *self;
            res.unop(Node::Inv);
            Some(res)
        }
    }

    fn to_unreduced(&self) -> Self::UnreducedElem {
        *self
    }

    fn mul_u64_unreduced(self, other: u64) -> Self::UnreducedMulU64 {
        self * Self::from_u64(other)
    }

    fn mul_u128_unreduced(self, other: u128) -> Self::UnreducedMulU128 {
        self * Self::from_u128(other)
    }

    fn mul_to_product(self, other: Self) -> Self::UnreducedProduct {
        self * other
    }

    fn mul_to_product_accum(self, other: Self) -> Self::UnreducedProductAccum {
        self * other
    }

    fn unreduced_mul_u64(a: &Self::UnreducedElem, b: u64) -> Self::UnreducedMulU64 {
        *a * Self::from_u64(b)
    }

    fn unreduced_mul_to_product_accum(
        a: &Self::UnreducedElem,
        b: &Self::UnreducedElem,
    ) -> Self::UnreducedProductAccum {
        *a * b
    }

    fn mul_to_accum_mag<const M: usize>(
        &self,
        _mag: &ark_ff::BigInt<M>,
    ) -> Self::UnreducedMulU128Accum {
        unimplemented!("Not needed for constructing ASTs")
    }

    fn mul_to_product_mag<const M: usize>(
        &self,
        _mag: &ark_ff::BigInt<M>,
    ) -> Self::UnreducedProduct {
        unimplemented!("Not needed for constructing ASTs")
    }

    fn reduce_mul_u64(x: Self::UnreducedMulU64) -> Self {
        x
    }
    fn reduce_mul_u128(x: Self::UnreducedMulU128) -> Self {
        x
    }
    fn reduce_mul_u128_accum(x: Self::UnreducedMulU128Accum) -> Self {
        x
    }
    fn reduce_product(x: Self::UnreducedProduct) -> Self {
        x
    }
    fn reduce_product_accum(x: Self::UnreducedProductAccum) -> Self {
        x
    }
}

/**********************************************************************
 * NOTE: We probably never need to serialize MleAst, so these are stubs
 */

impl CanonicalSerialize for MleAst {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        _writer: W,
        _compress: ark_serialize::Compress,
    ) -> Result<(), SerializationError> {
        // Store self in thread-local so PoseidonAstTranscript::append_scalar can retrieve it.
        // This is how we pass the MleAst through the generic Transcript trait.
        set_pending_append(*self);
        Ok(())
    }

    fn serialized_size(&self, _compress: ark_serialize::Compress) -> usize {
        // Return 32 bytes (standard field element size) so append_scalar works
        32
    }
}

impl CanonicalDeserialize for MleAst {
    fn deserialize_with_mode<R: std::io::Read>(
        _reader: R,
        _compress: ark_serialize::Compress,
        _validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        unimplemented!("Not needed for constructing ASTs")
    }
}

impl Valid for MleAst {
    fn check(&self) -> Result<(), SerializationError> {
        unimplemented!("Not needed for constructing ASTs")
    }
}

// =============================================================================
// Serialization stubs for Atom, Edge, Node
// =============================================================================

impl CanonicalSerialize for Atom {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        _writer: W,
        _compress: ark_serialize::Compress,
    ) -> Result<(), SerializationError> {
        unimplemented!("Not needed for constructing ASTs")
    }

    fn serialized_size(&self, _compress: ark_serialize::Compress) -> usize {
        unimplemented!("Not needed for constructing ASTs")
    }
}

impl CanonicalDeserialize for Atom {
    fn deserialize_with_mode<R: std::io::Read>(
        _reader: R,
        _compress: ark_serialize::Compress,
        _validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        unimplemented!("Not needed for constructing ASTs")
    }
}

impl Valid for Atom {
    fn check(&self) -> Result<(), SerializationError> {
        unimplemented!("Not needed for constructing ASTs")
    }
}

impl CanonicalSerialize for Edge {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        _writer: W,
        _compress: ark_serialize::Compress,
    ) -> Result<(), SerializationError> {
        unimplemented!("Not needed for constructing ASTs")
    }

    fn serialized_size(&self, _compress: ark_serialize::Compress) -> usize {
        unimplemented!("Not needed for constructing ASTs")
    }
}

impl CanonicalDeserialize for Edge {
    fn deserialize_with_mode<R: std::io::Read>(
        _reader: R,
        _compress: ark_serialize::Compress,
        _validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        unimplemented!("Not needed for constructing ASTs")
    }
}

impl Valid for Edge {
    fn check(&self) -> Result<(), SerializationError> {
        unimplemented!("Not needed for constructing ASTs")
    }
}

impl CanonicalSerialize for Node {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        _writer: W,
        _compress: ark_serialize::Compress,
    ) -> Result<(), SerializationError> {
        unimplemented!("Not needed for constructing ASTs")
    }

    fn serialized_size(&self, _compress: ark_serialize::Compress) -> usize {
        unimplemented!("Not needed for constructing ASTs")
    }
}

impl CanonicalDeserialize for Node {
    fn deserialize_with_mode<R: std::io::Read>(
        _reader: R,
        _compress: ark_serialize::Compress,
        _validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        unimplemented!("Not needed for constructing ASTs")
    }
}

impl Valid for Node {
    fn check(&self) -> Result<(), SerializationError> {
        unimplemented!("Not needed for constructing ASTs")
    }
}

/**********************************************************************/

// =============================================================================
// AstBundle: Serializable IR for transpilation and recursion
// =============================================================================

/// The kind of input variable - determines how it's treated in circuit generation.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InputKind {
    /// Public statement data (constant in the circuit).
    /// This includes things like: program bytecode hash, memory layout params,
    /// input/output hashes, etc. These are absorbed into the transcript during
    /// fiat_shamir_preamble but are fixed for a given program.
    PublicStatement,
    /// Proof data (variable in the circuit).
    /// This includes everything that comes from the proof: commitments,
    /// sumcheck coefficients, opening claims, etc. These vary per proof.
    ProofData,
}

/// Describes an input variable in the AST.
/// Maps `Var(i)` to its semantic meaning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputVar {
    /// The index into the vars register (matches `Atom::Var(index)`).
    pub index: u16,
    /// Human-readable name for debugging and codegen (e.g., "r_sumcheck_0", "claimed_output").
    pub name: String,
    /// Whether this is a public statement or proof data.
    pub kind: InputKind,
}

/// What assertion a constraint represents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Assertion {
    /// The expression must equal zero: `expr == 0`
    EqualZero,
    /// The expression must equal a public input by name: `expr == public_input[name]`
    EqualPublicInput { name: String },
    /// The expression must equal another node in the AST: `expr == other_node`
    EqualNode(NodeId),
}

/// A named constraint with its root expression and assertion type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    /// Human-readable name for the constraint (e.g., "stage1_sumcheck_final").
    pub name: String,
    /// The root node of the expression.
    pub root: NodeId,
    /// What assertion this constraint represents.
    pub assertion: Assertion,
}

/// Complete bundle of AST data for transpilation and recursion.
///
/// This structure contains everything needed to:
/// 1. Generate gnark circuits
/// 2. Generate other SNARK circuits for recursion
/// 3. Serialize/deserialize the AST (via JSON)
///
/// The `nodes` vec is the arena - all nodes are stored here and referenced by index.
/// The `bindings` vec contains NodeIds of hoisted subexpressions from CSE.
/// The `constraints` vec contains the actual assertions to be verified.
/// The `inputs` vec describes what each `Var(i)` means semantically.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstBundle {
    /// The node arena - all nodes in the AST(s).
    pub nodes: Vec<Node>,
    /// CSE bindings - NodeIds of hoisted common subexpressions.
    /// These map to `NamedVar(i)` references in the nodes.
    pub bindings: Vec<NodeId>,
    /// The constraints to be verified.
    pub constraints: Vec<Constraint>,
    /// Input variable descriptions.
    pub inputs: Vec<InputVar>,
}

impl AstBundle {
    /// Create a new empty bundle.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            bindings: Vec::new(),
            constraints: Vec::new(),
            inputs: Vec::new(),
        }
    }

    /// Add an input variable description.
    pub fn add_input(&mut self, index: u16, name: impl Into<String>, kind: InputKind) {
        self.inputs.push(InputVar {
            index,
            name: name.into(),
            kind,
        });
    }

    /// Add a constraint that asserts an expression equals zero.
    pub fn add_constraint_eq_zero(&mut self, name: impl Into<String>, root: NodeId) {
        self.constraints.push(Constraint {
            name: name.into(),
            root,
            assertion: Assertion::EqualZero,
        });
    }

    /// Add a constraint that asserts an expression equals a public input.
    pub fn add_constraint_eq_public(
        &mut self,
        name: impl Into<String>,
        root: NodeId,
        public_input_name: impl Into<String>,
    ) {
        self.constraints.push(Constraint {
            name: name.into(),
            root,
            assertion: Assertion::EqualPublicInput {
                name: public_input_name.into(),
            },
        });
    }

    /// Add a constraint that asserts two expressions are equal.
    pub fn add_constraint_eq_node(&mut self, name: impl Into<String>, root: NodeId, other: NodeId) {
        self.constraints.push(Constraint {
            name: name.into(),
            root,
            assertion: Assertion::EqualNode(other),
        });
    }

    /// Snapshot the current global arena into this bundle's nodes vec.
    /// Call this after all AST construction is complete.
    pub fn snapshot_arena(&mut self) {
        let arena = node_arena();
        let guard = arena.read().expect("node arena poisoned");
        self.nodes = guard.clone();
    }

    /// Get the number of public statement inputs.
    pub fn num_public_inputs(&self) -> usize {
        self.inputs
            .iter()
            .filter(|i| i.kind == InputKind::PublicStatement)
            .count()
    }

    /// Get the number of proof data inputs.
    pub fn num_proof_inputs(&self) -> usize {
        self.inputs
            .iter()
            .filter(|i| i.kind == InputKind::ProofData)
            .count()
    }

    /// Serialize to JSON string.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Serialize to pretty-printed JSON string.
    pub fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON string.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Write to a JSON file.
    pub fn write_json(&self, path: &std::path::Path) -> std::io::Result<()> {
        let json = self
            .to_json_pretty()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        std::fs::write(path, json)
    }

    /// Read from a JSON file.
    pub fn read_json(path: &std::path::Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Self::from_json(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
    }
}

impl Default for AstBundle {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// AstCommitment: Wrapper for commitments in symbolic execution
// =============================================================================

/// Wrapper type for a commitment represented as 12 MleAst chunks.
///
/// In the real verifier, commitments are `PCS::Commitment` (e.g., G1Affine, 384 bytes).
/// When `append_serializable` is called, it serializes to 384 bytes, reverses them,
/// and calls `append_bytes` which chunks into 12 × 32-byte pieces and hashes them
/// with proper chaining.
///
/// For symbolic execution, we represent each chunk as an MleAst variable.
/// When `AstCommitment` is serialized, it stores the 12 chunks in the
/// `PENDING_COMMITMENT_CHUNKS` thread-local. `PoseidonAstTranscript::append_serializable`
/// then retrieves them and performs the same 12-hash chaining operation symbolically.
#[derive(Clone, Debug)]
pub struct AstCommitment {
    /// The 12 MleAst chunks representing this commitment
    pub chunks: Vec<MleAst>,
}

impl AstCommitment {
    /// Create a new AstCommitment from 12 chunks.
    ///
    /// # Panics
    /// Panics if `chunks.len() != 12`.
    pub fn new(chunks: Vec<MleAst>) -> Self {
        assert_eq!(
            chunks.len(),
            12,
            "AstCommitment must have exactly 12 chunks"
        );
        Self { chunks }
    }
}

impl CanonicalSerialize for AstCommitment {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        _writer: W,
        _compress: ark_serialize::Compress,
    ) -> Result<(), SerializationError> {
        // Store chunks in thread-local for PoseidonAstTranscript::append_serializable to retrieve
        set_pending_commitment_chunks(self.chunks.clone());
        Ok(())
    }

    fn serialized_size(&self, _compress: ark_serialize::Compress) -> usize {
        384 // 12 chunks × 32 bytes = 384 bytes (same as G1Affine)
    }
}

impl CanonicalDeserialize for AstCommitment {
    fn deserialize_with_mode<R: std::io::Read>(
        _reader: R,
        _compress: ark_serialize::Compress,
        _validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        unimplemented!("AstCommitment deserialization not needed for transpilation")
    }
}

impl Valid for AstCommitment {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl Default for AstCommitment {
    fn default() -> Self {
        // Create 12 zero chunks
        Self {
            chunks: vec![MleAst::zero(); 12],
        }
    }
}

impl PartialEq for AstCommitment {
    fn eq(&self, other: &Self) -> bool {
        // Compare by root indices - this is sufficient for symbolic equality
        self.chunks.len() == other.chunks.len()
            && self
                .chunks
                .iter()
                .zip(other.chunks.iter())
                .all(|(a, b)| a.root() == b.root())
    }
}

impl AppendToTranscript for AstCommitment {
    fn append_to_transcript<T: jolt_core::transcripts::Transcript>(
        &self,
        label: &'static [u8],
        transcript: &mut T,
    ) {
        // Store chunks in thread-local for transcript to retrieve
        set_pending_commitment_chunks(self.chunks.clone());
        // The transcript's append_serializable will handle the actual hashing
        transcript.append_serializable(label, self);
    }
}
