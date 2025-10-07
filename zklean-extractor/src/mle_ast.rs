use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError, Valid};
use ark_std::{One, Zero};
use jolt_core::field::{FieldOps, JoltField};

use std::fmt::Write;

/// Type used to represent scalars. This needs to be large enought to avoid losing information when
/// we convert to field elements. We use i128 here in order to support negative scalars.
type Scalar = u32;
type Index = u16;

/// An atomic (var or const) AST element
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Atom {
    /// A constant value.
    Scalar(Scalar),
    /// A variable, represented by an index into a register of variables
    Var(Index),
}

/// Either an index into the AST's node array, or a an atomic (var or const) element
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Edge {
    /// An atomic (var or const) AST element
    Atom(Atom),
    /// A reference to a node in the AST's `nodes` array
    NodeRef(Index),
}

/// A node for a polynomial AST, where children are represented by de Bruijn indices (negative
/// relative to the parent) into an array of nodes.
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
/// multivariate polynomial). The nodes are stored in a statically sized array, which allows the
/// data structure to be [`Copy`] and [`Sized`]. The size of the array (i.e., the max number of
/// nodes that may be stored) is given by the const-generic `NUM_NODES` type argument.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct MleAst<const NUM_NODES: usize> {
    /// Collection of nodes; the indices in each [`MleAstNode`] are indices into this array
    nodes: [Option<Node>; NUM_NODES],
    /// Index of the root of the AST (should always be the last used node)
    root: usize,
    /// Name of the register this MLE is evaluated over. We use a single char because this type
    /// needs to be `Sized`.
    /// TODO: Support multiple registers?
    reg_name: Option<char>,
}

impl<const NUM_NODES: usize> MleAst<NUM_NODES> {
    fn empty() -> Self {
        Self {
            nodes: [None; NUM_NODES],
            root: 0,
            reg_name: None,
        }
    }

    fn new_scalar(scalar: Scalar) -> Self {
        let mut res = Self::empty();
        res.nodes[0] = Some(Node::Atom(Atom::Scalar(scalar)));
        res
    }

    fn new_var(name: char, index: Index) -> Self {
        let mut res = Self::empty();
        res.nodes[0] = Some(Node::Atom(Atom::Var(index)));
        res.reg_name = Some(name);
        res
    }

    /// Return the number of nodes used so far in this AST
    fn nodes_used(&self) -> usize {
        // The root is always the index of the last node, so the number of nodes used is one more.
        self.root + 1
    }

    /// Append the nodes of `other` onto the array of nodes in `self`. The root of `self` (the left
    /// tree) will be the root the appended copy of `other` (the right tree). Return the indices of
    /// the left and right trees in the node array of `self`.
    fn concatenate(&mut self, other: Self) -> (usize, usize) {
        let shift = self.root + 1;
        for i in 0..=other.root {
            self.nodes[shift + i] = other.nodes[i];
            self.root += 1;
        }

        // de Bruijn indices, i.e., negative relative to the new root
        let left_root = other.root + 2;
        let right_root = 1;

        (left_root, right_root)
    }

    /// Create a new root node in the form of a unitary operator.
    fn unop(&mut self, constructor: impl FnOnce(Edge) -> Node) {
        match self.nodes[self.root].expect("unop called on AST with empty root") {
            Node::Atom(a) => {
                self.nodes[self.root] = Some(constructor(Edge::Atom(a)));
            }
            _ => {
                let required_nodes = self.nodes_used() + 1;
                assert!(required_nodes <= NUM_NODES,
                    "Ran out of space for nodes. Try increasing NUM_NODES from {NUM_NODES} to at least {required_nodes}.");
                self.nodes[self.root + 1] = Some(constructor(Edge::NodeRef(1)));
                self.root += 1;
            }
        }
    }

    /// Create a new root node in the form of a binary operator.
    fn binop(&mut self, constructor: impl FnOnce(Edge, Edge) -> Node, rhs: Self) {
        if let (Some(n), Some(m)) = (self.reg_name, rhs.reg_name) {
             assert_eq!(n, m, "multiple registers not supported");
        }
        let reg_name = self.reg_name.or(rhs.reg_name);

        let lhs_root_node = self.nodes[self.root].expect("binop called on lhs AST with empty root");
        let rhs_root_node = rhs.nodes[rhs.root].expect("binop called on rhs AST with empty root");

        match (lhs_root_node, rhs_root_node) {
            (Node::Atom(a1), Node::Atom(a2)) => {
                self.nodes[self.root] = Some(constructor(Edge::Atom(a1), Edge::Atom(a2)));
            }
            (Node::Atom(a), _) => {
                *self = rhs;
                self.unop(|i| constructor(Edge::Atom(a), i));
            }
            (_, Node::Atom(a)) => {
                self.unop(|i| constructor(i, Edge::Atom(a)));
            }
            _ => {
                let required_nodes = self.nodes_used() + rhs.nodes_used() + 1;
                assert!(required_nodes <= NUM_NODES,
                    "Ran out of space for nodes. Try increasing NUM_NODES from {NUM_NODES} to at least {required_nodes}.");
                let (lhs_root, rhs_root) = self.concatenate(rhs);
                self.nodes[self.root + 1] = Some(constructor(Edge::NodeRef(lhs_root as Index), Edge::NodeRef(rhs_root as Index)));
                self.root += 1;
            }
        }

        self.reg_name = reg_name;
    }
}

impl Atom {
    #[cfg(test)]
    fn evaluate<F: JoltField>(&self, vars: &[F]) -> F {
        match self {
            Self::Scalar(f) => F::from_u64(*f as u64), // TODO: handle negative scalars?
            Self::Var(var) => vars[*var as usize], // TODO: handle multiple registers?
        }
    }
}

impl<const NUM_NODES: usize> crate::util::ZkLeanReprField for MleAst<NUM_NODES> {
    fn register(name: char, size: usize) -> Vec<Self> {
        (0..size)
            .map(|i| Self::new_var(name, i as Index))
            .collect()
    }

    fn as_computation(&self) -> String {
        let mut res = "".to_string();
        write!(res, "{self}").unwrap();
        res
    }

    /// Evaluate the computation represented by the AST over another [`JoltField`], starting at
    /// `root`, and using the variable assignments in `vars`.
    #[cfg(test)]
    fn evaluate<F: JoltField>(&self, vars: &[F]) -> F {
        println!("{self}");
        // Reversed nodes; indices are now reverse de Bruijn indices (positive relative to root)
        let nodes = self.nodes[..=self.root].iter().flatten().rev().collect::<Vec<_>>();

        fn visit_index<F: JoltField>(index: &Edge, tree: &[&Node], vars: &[F]) -> F{
            match index {
                Edge::Atom(a) => a.evaluate(vars),
                Edge::NodeRef(i) => visit_subtree(&tree[*i as usize..], vars),
            }
        }

        fn visit_subtree<F: JoltField>(tree: &[&Node], vars: &[F]) -> F {
            match tree[0] {
                Node::Atom(a) => a.evaluate(vars),
                Node::Neg(i) => -visit_index(i, tree, vars),
                Node::Inv(i) => F::one() / visit_index(i, tree, vars),
                Node::Add(i1, i2) => visit_index(i1, tree, vars) + visit_index(i2, tree, vars),
                Node::Mul(i1, i2) => visit_index(i1, tree, vars) * visit_index(i2, tree, vars),
                Node::Sub(i1, i2) => visit_index(i1, tree, vars) - visit_index(i2, tree, vars),
                Node::Div(i1, i2) => visit_index(i1, tree, vars) / visit_index(i2, tree, vars),
            }
        }

        visit_subtree(&nodes, vars)
    }
}

impl<const NUM_NODES: usize> Zero for MleAst<NUM_NODES> {
    fn zero() -> Self {
        Self::new_scalar(0)
    }

    fn is_zero(&self) -> bool {
        self.nodes[self.root] == Some(Node::Atom(Atom::Scalar(0)))
    }
}

impl<const NUM_NODES: usize> One for MleAst<NUM_NODES> {
    fn one() -> Self {
        Self::new_scalar(1)
    }
}

impl<const NUM_NODES: usize> std::ops::Neg for MleAst<NUM_NODES> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.unop(Node::Neg);
        self
    }
}

impl<const NUM_NODES: usize> std::ops::Add for MleAst<NUM_NODES> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.binop(Node::Add, rhs);
        self
    }
}

impl<const NUM_NODES: usize> std::ops::Sub for MleAst<NUM_NODES> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self.binop(Node::Sub, rhs);
        self
    }
}

impl<const NUM_NODES: usize> std::ops::Mul for MleAst<NUM_NODES> {
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self::Output {
        self.binop(Node::Mul, rhs);
        self
    }
}

impl<const NUM_NODES: usize> std::ops::Div for MleAst<NUM_NODES> {
    type Output = Self;

    fn div(mut self, rhs: Self) -> Self::Output {
        self.binop(Node::Div, rhs);
        self
    }
}

impl<const NUM_NODES: usize> FieldOps for MleAst<NUM_NODES> {}

impl<const NUM_NODES: usize> std::ops::Add<&Self> for MleAst<NUM_NODES> {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        self + *rhs
    }
}

impl<const NUM_NODES: usize> std::ops::Sub<&Self> for MleAst<NUM_NODES> {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        self - *rhs
    }
}

impl<const NUM_NODES: usize> std::ops::Mul<&Self> for MleAst<NUM_NODES> {
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self::Output {
        self * *rhs
    }
}

impl<const NUM_NODES: usize> std::ops::Div<&Self> for MleAst<NUM_NODES> {
    type Output = Self;

    fn div(self, rhs: &Self) -> Self::Output {
        self / *rhs
    }
}

impl<const NUM_NODES: usize> FieldOps<&Self, Self> for MleAst<NUM_NODES> {}

impl<const NUM_NODES: usize> std::ops::AddAssign for MleAst<NUM_NODES> {
    fn add_assign(&mut self, rhs: Self) {
        self.binop(Node::Add, rhs);
    }
}

impl<const NUM_NODES: usize> std::ops::SubAssign for MleAst<NUM_NODES> {
    fn sub_assign(&mut self, rhs: Self) {
        self.binop(Node::Sub, rhs);
    }
}

impl<const NUM_NODES: usize> std::ops::MulAssign for MleAst<NUM_NODES> {
    fn mul_assign(&mut self, rhs: Self) {
        self.binop(Node::Mul, rhs);
    }
}

impl<const NUM_NODES: usize> core::iter::Sum for MleAst<NUM_NODES> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|sum, term| sum + term).unwrap_or(Self::zero())
    }
}

impl<'a, const NUM_NODES: usize> core::iter::Sum<&'a Self> for MleAst<NUM_NODES> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.cloned().sum()
    }
}

impl<const NUM_NODES: usize> core::iter::Product for MleAst<NUM_NODES> {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|product, factor| product * factor)
            .unwrap_or(Self::one())
    }
}

impl<'a, const NUM_NODES: usize> core::iter::Product<&'a Self> for MleAst<NUM_NODES> {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.cloned().product()
    }
}

/// Displays the AST as an algebraic formula. Variables are displayed as `name[index]`.
impl<const NUM_NODES: usize> std::fmt::Display for MleAst<NUM_NODES> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Reversed nodes; indices are now reverse de Bruijn indices (positive relative to root)
        let nodes = self.nodes[..=self.root].iter().flatten().rev().collect::<Vec<_>>();

        fn visit_atom(
            f: &mut std::fmt::Formatter<'_>,
            atom: &Atom,
            reg_name: Option<char>,
        ) -> std::fmt::Result {
            match atom {
                Atom::Scalar(scalar) => write!(f, "{scalar}")?,
                Atom::Var(index) => {
                    let name = reg_name.expect("unreachable: register name missing in var");
                    write!(f, "{name}[{index}]")?;
                }
            }

            Ok(())
        }

        fn visit_index(
            f: &mut std::fmt::Formatter<'_>,
            index: &Edge,
            tree: &[&Node],
            reg_name: Option<char>,
            group: bool, // whether to parenthesize this expression
        ) -> std::fmt::Result {
            match index {
                Edge::Atom(a) => visit_atom(f, a, reg_name)?,
                Edge::NodeRef(i) => visit_subtree(f, &tree[*i as usize..], reg_name, group)?,
            }

            Ok(())
        }

        fn visit_subtree(
            f: &mut std::fmt::Formatter<'_>,
            tree: &[&Node],
            reg_name: Option<char>,
            group: bool, // whether to parenthesize this expression
        ) -> std::fmt::Result {
            match tree[0] {
                Node::Atom(a) => visit_atom(f, a, reg_name)?,
                Node::Neg(i) => {
                    write!(f, "-")?;
                    visit_index(f, i, tree, reg_name, true)?;
                }
                Node::Inv(i) => {
                    write!(f, "1/")?;
                    visit_index(f, i, tree, reg_name, true)?;
                }
                Node::Add(i1, i2) => {
                    if group {
                        write!(f, "(")?;
                    }
                    visit_index(f, i1, tree, reg_name, false)?;
                    write!(f, " + ")?;
                    visit_index(f, i2, tree, reg_name, false)?;
                    if group {
                        write!(f, ")")?;
                    }
                }
                Node::Mul(i1, i2) => {
                    visit_index(f, i1, tree, reg_name, true)?;
                    write!(f, " * ")?;
                    visit_index(f, i2, tree, reg_name, true)?;
                }
                Node::Sub(i1, i2) => {
                    if group {
                        write!(f, "(")?;
                    }
                    visit_index(f, i1, tree, reg_name, false)?;
                    write!(f, " - ")?;
                    visit_index(f, i2, tree, reg_name, true)?;
                    if group {
                        write!(f, ")")?;
                    }
                }
                Node::Div(i1, i2) => {
                    visit_index(f, i1, tree, reg_name, true)?;
                    write!(f, " / ")?;
                    visit_index(f, i2, tree, reg_name, true)?;
                }
            }

            Ok(())
        }

        visit_subtree(f, &nodes, self.reg_name, false)
    }
}

impl<const NUM_NODES: usize> Default for MleAst<NUM_NODES> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const NUM_NODES: usize> std::hash::Hash for MleAst<NUM_NODES> {
    fn hash<H: std::hash::Hasher>(&self, _state: &mut H) {
        unimplemented!("hash unimplemented for MleAst<NUM_NODES>")
    }
}

impl<const NUM_NODES: usize> JoltField for MleAst<NUM_NODES> {
    const NUM_BYTES: usize = 0;

    type SmallValueLookupTables = ();

    fn random<R: rand_core::RngCore>(_rng: &mut R) -> Self {
        unimplemented!("Not needed for constructing ASTs");
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

    fn from_i128(n: i128) -> Self {
        Self::new_scalar(n as Scalar)
    }

    fn square(&self) -> Self {
        *self * *self
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

impl<const NUM_NODES: usize> CanonicalSerialize for MleAst<NUM_NODES> {
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

impl<const NUM_NODES: usize> CanonicalDeserialize for MleAst<NUM_NODES> {
    fn deserialize_with_mode<R: std::io::Read>(
        _reader: R,
        _compress: ark_serialize::Compress,
        _validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        unimplemented!("Not needed for constructing ASTs")
    }
}

impl<const NUM_NODES: usize> Valid for MleAst<NUM_NODES> {
    fn check(&self) -> Result<(), SerializationError> {
        unimplemented!("Not needed for constructing ASTs")
    }
}

/**********************************************************************/
