use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError, Valid};
use ark_std::{One, Zero};
use jolt_core::field::{FieldOps, JoltField};

use std::fmt::{self, Write};
use std::sync::{OnceLock, RwLock};

type Scalar = i128;

type Index = u16;

type NodeId = usize;

static NODE_ARENA: OnceLock<RwLock<Vec<Node>>> = OnceLock::new();

fn node_arena() -> &'static RwLock<Vec<Node>> {
    NODE_ARENA.get_or_init(|| RwLock::new(Vec::new()))
}

fn insert_node(node: Node) -> NodeId {
    let arena = node_arena();
    let mut guard = arena.write().expect("node arena poisoned");
    let id = guard.len();
    guard.push(node);
    id
}

fn get_node(id: NodeId) -> Node {
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

pub type DefaultMleAst = MleAst;

/// An atomic (var or const) AST element
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Atom {
    /// A constant value.
    Scalar(Scalar),
    /// A variable, represented by an index into a register of variables
    Var(Index),
}

impl Atom {
    #[cfg(test)]
    fn evaluate<F: JoltField>(&self, vars: &[F]) -> F {
        match self {
            Self::Scalar(value) => F::from_i128(*value),
            Self::Var(index) => vars[*index as usize], // TODO: handle multiple registers?
        }
    }
}

/// Either an index into the arena, or an atomic (var or const) element.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Edge {
    /// An atomic (var or const) AST element.
    Atom(Atom),
    /// A reference to a node in the arena.
    NodeRef(NodeId),
}

/// A node for a polynomial AST. Children are represented by node IDs into the global arena.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Node {
    /// An atomic (var or const) AST element. This should only be used for MLE's with a single
    /// node.
    Atom(Atom),
    /// The negation of a node
    Neg(Edge),
    /// The multiplicative inverse of a node
    Inv(Edge),
    /// The sum of two nodes
    Add(Edge, Edge),
    /// The product of two nodes
    Mul(Edge, Edge),
    /// The difference between the first and second nodes
    Sub(Edge, Edge),
    /// The quotient between the first and second nodes
    /// NOTE: No div-by-zero checks are performed here
    Div(Edge, Edge),
}

/// An AST intended for representing an MLE computation (although it will actually work for any
/// multivariate polynomial). The nodes are stored in a global arena, which allows each AST handle
/// to remain [`Copy`] and [`Sized`] while supporting unbounded growth of the underlying graph.
#[derive(Debug, PartialEq, Eq, PartialOrd, Clone, Copy)]
pub struct MleAst {
    /// Index of the root node in the arena.
    /// nodes: [ ]
    root: NodeId,
    /// Name of the register this MLE is evaluated over.
    // TODO: Support multiple registers?
    reg_name: Option<char>,
}

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
}

#[cfg(test)]
fn evaluate_edge<F: JoltField>(edge: Edge, vars: &[F]) -> F {
    match edge {
        Edge::Atom(atom) => atom.evaluate(vars),
        Edge::NodeRef(node) => evaluate_node(node, vars),
    }
}

#[cfg(test)]
fn evaluate_node<F: JoltField>(node: NodeId, vars: &[F]) -> F {
    match get_node(node) {
        Node::Atom(atom) => atom.evaluate(vars),
        Node::Neg(edge) => -evaluate_edge(edge, vars),
        Node::Inv(edge) => F::one() / evaluate_edge(edge, vars),
        Node::Add(e1, e2) => evaluate_edge(e1, vars) + evaluate_edge(e2, vars),
        Node::Mul(e1, e2) => evaluate_edge(e1, vars) * evaluate_edge(e2, vars),
        Node::Sub(e1, e2) => evaluate_edge(e1, vars) - evaluate_edge(e2, vars),
        Node::Div(e1, e2) => evaluate_edge(e1, vars) / evaluate_edge(e2, vars),
    }
}

fn fmt_atom(f: &mut fmt::Formatter<'_>, atom: Atom, reg_name: Option<char>) -> fmt::Result {
    match atom {
        Atom::Scalar(value) => write!(f, "{value}")?,
        Atom::Var(index) => {
            let name = reg_name.expect("unreachable: register name missing in var");
            write!(f, "{name}[{index}]")?;
        }
    }

    Ok(())
}

fn fmt_edge(
    f: &mut fmt::Formatter<'_>,
    edge: Edge,
    reg_name: Option<char>,
    group: bool,
) -> fmt::Result {
    match edge {
        Edge::Atom(atom) => fmt_atom(f, atom, reg_name),
        Edge::NodeRef(node) => fmt_node(f, node, reg_name, group),
    }
}

fn fmt_node(
    f: &mut fmt::Formatter<'_>,
    node: NodeId,
    reg_name: Option<char>,
    group: bool,
) -> fmt::Result {
    match get_node(node) {
        Node::Atom(atom) => fmt_atom(f, atom, reg_name),
        Node::Neg(edge) => {
            write!(f, "-")?;
            fmt_edge(f, edge, reg_name, true)
        }
        Node::Inv(edge) => {
            write!(f, "1 / ")?;
            fmt_edge(f, edge, reg_name, true)
        }
        Node::Add(e1, e2) => {
            if group {
                write!(f, "(")?;
            }
            fmt_edge(f, e1, reg_name, false)?;
            write!(f, " + ")?;
            fmt_edge(f, e2, reg_name, false)?;
            if group {
                write!(f, ")")?;
            }
            Ok(())
        }
        Node::Mul(e1, e2) => {
            fmt_edge(f, e1, reg_name, true)?;
            write!(f, " * ")?;
            fmt_edge(f, e2, reg_name, true)
        }
        Node::Sub(e1, e2) => {
            if group {
                write!(f, "(")?;
            }
            fmt_edge(f, e1, reg_name, false)?;
            write!(f, " - ")?;
            fmt_edge(f, e2, reg_name, true)?;
            if group {
                write!(f, ")")?;
            }
            Ok(())
        }
        Node::Div(e1, e2) => {
            fmt_edge(f, e1, reg_name, true)?;
            write!(f, " / ")?;
            fmt_edge(f, e2, reg_name, true)
        }
    }
}

impl crate::util::ZkLeanReprField for MleAst {
    fn register(name: char, size: usize) -> Vec<Self> {
        (0..size).map(|i| Self::new_var(name, i as Index)).collect()
    }

    fn as_computation(&self) -> String {
        let mut res = String::new();
        write!(res, "{self}").unwrap();
        res
    }

    /// Evaluate the computation represented by the AST over another [`JoltField`], starting at
    /// `root`, and using the variable assignments in `vars`.
    #[cfg(test)]
    fn evaluate<F: JoltField>(&self, vars: &[F]) -> F {
        evaluate_node(self.root, vars)
    }
}

impl Zero for MleAst {
    fn zero() -> Self {
        Self::new_scalar(0)
    }

    fn is_zero(&self) -> bool {
        matches!(
            get_node(self.root),
            Node::Atom(Atom::Scalar(value)) if value == 0
        )
    }
}

impl One for MleAst {
    fn one() -> Self {
        Self::new_scalar(1)
    }
}

impl std::ops::Neg for MleAst {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.unop(Node::Neg);
        self
    }
}

impl std::ops::Add for MleAst {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl std::ops::Sub for MleAst {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self - &rhs
    }
}

impl std::ops::Mul for MleAst {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self * &rhs
    }
}

impl std::ops::Div for MleAst {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self / &rhs
    }
}

impl FieldOps for MleAst {}

impl std::ops::Add<&Self> for MleAst {
    type Output = Self;

    fn add(mut self, rhs: &Self) -> Self::Output {
        self.binop(Node::Add, rhs);
        self
    }
}

impl std::ops::Sub<&Self> for MleAst {
    type Output = Self;

    fn sub(mut self, rhs: &Self) -> Self::Output {
        self.binop(Node::Sub, rhs);
        self
    }
}

impl std::ops::Mul<&Self> for MleAst {
    type Output = Self;

    fn mul(mut self, rhs: &Self) -> Self::Output {
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
        self.binop(Node::Add, &rhs);
    }
}

impl<'a> std::ops::AddAssign<&'a Self> for MleAst {
    fn add_assign(&mut self, rhs: &'a Self) {
        self.binop(Node::Add, rhs);
    }
}

impl std::ops::SubAssign for MleAst {
    fn sub_assign(&mut self, rhs: Self) {
        self.binop(Node::Sub, &rhs);
    }
}

impl<'a> std::ops::SubAssign<&'a Self> for MleAst {
    fn sub_assign(&mut self, rhs: &'a Self) {
        self.binop(Node::Sub, &rhs);
    }
}

impl std::ops::MulAssign for MleAst {
    fn mul_assign(&mut self, rhs: Self) {
        self.binop(Node::Mul, &rhs);
    }
}

impl<'a> std::ops::MulAssign<&'a Self> for MleAst {
    fn mul_assign(&mut self, rhs: &'a Self) {
        self.binop(Node::Mul, &rhs);
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

impl fmt::Display for MleAst {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt_node(f, self.root, self.reg_name, false)
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

impl jolt_core::field::MulTrunc for MleAst {
    type Other<const M: usize> = Self;

    type Output<const P: usize> = Self;

    fn mul_trunc<const M: usize, const P: usize>(&self, other: &Self::Other<M>) -> Self::Output<P> {
        *self * other
    }
}

impl jolt_core::field::FmaddTrunc for MleAst {
    type Other<const M: usize> = Self;

    type Acc<const P: usize> = Self;

    fn fmadd_trunc<const M: usize, const P: usize>(
        &self,
        _other: &Self::Other<M>,
        _acc: &mut Self::Acc<P>,
    ) {
        unimplemented!("Not needed for constructing ASTs");
    }
}

impl jolt_core::field::MulU64WithCarry for MleAst {
    type Output<const NPLUS1: usize> = Self;

    fn mul_u64_w_carry<const NPLUS1: usize>(&self, _other: u64) -> Self::Output<NPLUS1> {
        unimplemented!("Not needed for constructing ASTs");
    }
}

impl From<ark_ff::biginteger::signed_hi_32::SignedBigIntHi32<3>> for MleAst {
    fn from(_value: ark_ff::biginteger::signed_hi_32::SignedBigIntHi32<3>) -> Self {
        unimplemented!("Not needed for constructing ASTs");
    }
}

impl<const N: usize> From<[u64; N]> for MleAst {
    fn from(_value: [u64; N]) -> Self {
        unimplemented!("Not needed for constructing ASTs");
    }
}

impl Ord for MleAst {
    fn cmp(&self, _other: &Self) -> std::cmp::Ordering {
        unimplemented!("Not needed for constructing ASTs");
    }
}

impl JoltField for MleAst {
    const NUM_BYTES: usize = 0;

    const MONTGOMERY_R: Self = todo!();

    const MONTGOMERY_R_SQUARE: Self = todo!();

    type Unreduced<const N: usize> = Self;

    type Challenge = Self;

    type SmallValueLookupTables = ();

    fn random<R: rand_core::RngCore>(_rng: &mut R) -> Self {
        unimplemented!("Not needed for constructing ASTs");
    }

    fn from_bool(val: bool) -> Self {
        Self::new_scalar(val as Scalar)
    }

    fn from_u8(n: u8) -> Self {
        Self::new_scalar(n as Scalar)
    }

    fn from_u16(n: u16) -> Self {
        Self::new_scalar(n as Scalar)
    }

    fn from_u32(n: u32) -> Self {
        Self::new_scalar(n as Scalar)
    }

    fn from_u64(n: u64) -> Self {
        Self::new_scalar(n as Scalar)
    }

    fn from_i64(n: i64) -> Self {
        Self::new_scalar(n as Scalar)
    }

    fn from_u128(n: u128) -> Self {
        Self::new_scalar(n as Scalar)
    }

    fn from_i128(n: i128) -> Self {
        Self::new_scalar(n as Scalar)
    }

    fn square(&self) -> Self {
        *self * self
    }

    fn from_bytes(_bytes: &[u8]) -> Self {
        unimplemented!("Not needed for constructing ASTs");
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

    fn as_unreduced_ref(&self) -> &Self::Unreduced<4> {
        self
    }

    fn mul_unreduced<const N: usize>(self, other: Self) -> Self::Unreduced<N> {
        self * other
    }

    fn mul_u64_unreduced(self, other: u64) -> Self::Unreduced<5> {
        self * Self::from_u64(other)
    }

    fn mul_u128_unreduced(self, other: u128) -> Self::Unreduced<6> {
        self * Self::from_u128(other)
    }

    fn from_montgomery_reduce<const N: usize>(unreduced: Self::Unreduced<N>) -> Self {
        unreduced
    }

    fn from_barrett_reduce<const N: usize>(unreduced: Self::Unreduced<N>) -> Self {
        unreduced
    }
}

/**********************************************************************
 * NOTE: We probably never need to serialize MleAst, so these are stubs
 */

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

impl CanonicalSerialize for MleAst {
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

/**********************************************************************/
