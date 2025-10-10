use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError, Valid};
use ark_std::{One, Zero};
use jolt_core::field::{FieldOps, JoltField};

use std::fmt::Write;

/// Type used to represent scalars. This needs to be large enought to avoid losing information when
/// we convert to field elements. We use i128 here in order to support negative scalars.
type Scalar = i128;

/// Type used to represent an index into the node, scalar, or variable arrays. Using an integer
/// smaller than `usize` (8 bytes, usually) significantly decreases the size of a `Node` enum,
/// since we need space for 2 per node.
type Index = u16;

pub type DefaultMleAst = MleAst<6080, 128>;

/// An atomic (var or const) AST element
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum Atom {
    /// A constant value.
    Scalar(Index),
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
/// `NUM_SCALARS` dictates the number of *unique* scalars that can be stored in the AST.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct MleAst<const NUM_NODES: usize, const NUM_SCALARS: usize> {
    /// Collection of nodes; the indices in each [`MleAstNode`] are indices into this array
    nodes: [Option<Node>; NUM_NODES],
    /// Index of the root of the AST (should always be the last used node)
    root: usize,
    /// Name of the register this MLE is evaluated over. We use a single char because this type
    /// needs to be `Sized`.
    // TODO: Support multiple registers?
    reg_name: Option<char>,
    scalars: [Option<Scalar>; NUM_SCALARS],
}

impl<const NUM_NODES: usize, const NUM_SCALARS: usize> MleAst<NUM_NODES, NUM_SCALARS> {
    fn empty() -> Self {
        Self {
            nodes: [None; NUM_NODES],
            root: 0,
            reg_name: None,
            scalars: [None; NUM_SCALARS],
        }
    }

    fn new_scalar(scalar: Scalar) -> Self {
        let mut res = Self::empty();
        res.nodes[0] = Some(Node::Atom(res.make_scalar_atom(scalar)));
        res
    }

    fn new_var(name: char, index: Index) -> Self {
        let mut res = Self::empty();
        res.nodes[0] = Some(Node::Atom(Atom::Var(index)));
        res.reg_name = Some(name);
        res
    }

    /// Construct a new scalar atom by either adding a new scalar to the table or reusing it, if it
    /// exists. Return an [`Atom::Scalar`] with the relevant index.
    fn make_scalar_atom(&mut self, scalar: Scalar) -> Atom {
        // TODO: Use something better than linear search here? Scales with the number of unique
        // scalars, which is generally small, so it's not noticably slow.
        let pos = self
            .scalars
            .iter()
            .position(|s| *s == Some(scalar) || s.is_none())
            .expect("Ran out of space for scalars");

        if self.scalars[pos].is_none() {
            self.scalars[pos] = Some(scalar);
        }
        Atom::Scalar(pos as Index)
    }

    /// Duplicate an atom from another AST into this AST. All scalar references will be changed to
    /// point to the scalar array in this AST, adding the scalar to the array, if necessary.
    fn duplicate_atom(&mut self, other: &Self, atom: Atom) -> Atom {
        match atom {
            Atom::Scalar(i) => {
                self.make_scalar_atom(other.scalars[i as usize].expect("Invalid scalar ref"))
            }
            Atom::Var(i) => {
                if let (Some(n), Some(m)) = (self.reg_name, other.reg_name) {
                    assert_eq!(n, m, "Multiple registers not supported");
                }
                self.reg_name = self.reg_name.or(other.reg_name);
                Atom::Var(i)
            }
        }
    }

    /// Duplicate an edge from another AST into this AST. All scalar references will be changed to
    /// point to the scalar array in this AST. The de Bruijn indices referring to other nodes will
    /// be preserved, so that the nodes can be copied in order.
    fn duplicate_edge(&mut self, other: &Self, edge: Edge) -> Edge {
        match edge {
            Edge::Atom(a) => Edge::Atom(self.duplicate_atom(other, a)),
            e => e,
        }
    }

    /// Duplicate a node from another AST into this AST. All scalar references will be changed to
    /// point to the scalar array in this AST. The de Bruijn indices referring to other nodes will
    /// be preserved, so that the nodes can be copied in order.
    fn duplicate_node(&mut self, other: &Self, node: Node) -> Node {
        match node {
            Node::Atom(a) => Node::Atom(self.duplicate_atom(other, a)),
            Node::Neg(e) => Node::Neg(self.duplicate_edge(other, e)),
            Node::Inv(e) => Node::Inv(self.duplicate_edge(other, e)),
            Node::Add(e1, e2) => Node::Add(
                self.duplicate_edge(other, e1),
                self.duplicate_edge(other, e2),
            ),
            Node::Mul(e1, e2) => Node::Mul(
                self.duplicate_edge(other, e1),
                self.duplicate_edge(other, e2),
            ),
            Node::Sub(e1, e2) => Node::Sub(
                self.duplicate_edge(other, e1),
                self.duplicate_edge(other, e2),
            ),
            Node::Div(e1, e2) => Node::Div(
                self.duplicate_edge(other, e1),
                self.duplicate_edge(other, e2),
            ),
        }
    }

    /// Return the number of nodes used so far in this AST
    fn nodes_used(&self) -> usize {
        // The root is always the index of the last node, so the number of nodes used is one more.
        self.root + 1
    }

    fn assert_have_space(&self, required_nodes: usize) {
        assert!(required_nodes <= NUM_NODES,
            "Ran out of space for nodes. Try increasing NUM_NODES from {NUM_NODES} to at least {required_nodes}.");
    }

    /// Append the nodes of `other` onto the array of nodes in `self`. The root of `self` (the left
    /// tree) will be the root the appended copy of `other` (the right tree). Return the indices of
    /// the left and right trees in the node array of `self`.
    fn concatenate(&mut self, other: &Self) -> (usize, usize) {
        let shift = self.root + 1;
        for i in 0..=other.root {
            let new_node = self.duplicate_node(other, other.nodes[i].expect("Invalid AST"));
            self.nodes[shift + i] = Some(new_node);
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
                self.assert_have_space(self.nodes_used() + 1);
                self.nodes[self.root + 1] = Some(constructor(Edge::NodeRef(1)));
                self.root += 1;
            }
        }
    }

    /// Create a new root node in the form of a binary operator.
    fn binop(&mut self, constructor: impl FnOnce(Edge, Edge) -> Node, rhs: &Self) {
        let lhs_root_node = self.nodes[self.root].expect("binop called on lhs AST with empty root");
        let rhs_root_node = rhs.nodes[rhs.root].expect("binop called on rhs AST with empty root");

        match (lhs_root_node, rhs_root_node) {
            (Node::Atom(a1), Node::Atom(a2)) => {
                let e1 = Edge::Atom(a1);
                let e2 = Edge::Atom(self.duplicate_atom(rhs, a2));
                self.nodes[self.root] = Some(constructor(e1, e2));
            }
            (Node::Atom(a), _) => {
                self.assert_have_space(rhs.nodes_used() + 1);
                let copy = *self;
                *self = *rhs;
                let e1 = Edge::Atom(self.duplicate_atom(&copy, a));
                let e2 = Edge::NodeRef(1);
                self.nodes[self.root + 1] = Some(constructor(e1, e2));
                self.root += 1;
            }
            (_, Node::Atom(a)) => {
                self.assert_have_space(self.nodes_used() + 1);
                let e1 = Edge::NodeRef(1);
                let e2 = Edge::Atom(self.duplicate_atom(rhs, a));
                self.nodes[self.root + 1] = Some(constructor(e1, e2));
                self.root += 1;
            }
            _ => {
                self.assert_have_space(self.nodes_used() + rhs.nodes_used() + 1);
                let (lhs_root, rhs_root) = self.concatenate(rhs);
                let e1 = Edge::NodeRef(lhs_root as Index);
                let e2 = Edge::NodeRef(rhs_root as Index);
                self.nodes[self.root + 1] = Some(constructor(e1, e2));
                self.root += 1;
            }
        };
    }
}

impl Atom {
    #[cfg(test)]
    fn evaluate<F: JoltField>(&self, vars: &[F], scalars: &[Option<Scalar>]) -> F {
        match self {
            Self::Scalar(i) => {
                let scalar = scalars[*i as usize].expect("Invalid scalar ref");
                F::from_u64(scalar as u64) // TODO: handle negative scalars?
            }
            Self::Var(i) => vars[*i as usize], // TODO: handle multiple registers?
        }
    }
}

impl<const NUM_NODES: usize, const NUM_SCALARS: usize> crate::util::ZkLeanReprField
    for MleAst<NUM_NODES, NUM_SCALARS>
{
    fn register(name: char, size: usize) -> Vec<Self> {
        (0..size).map(|i| Self::new_var(name, i as Index)).collect()
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
        // Reversed nodes; indices are now reverse de Bruijn indices (positive relative to root)
        let nodes = self.nodes[..=self.root]
            .iter()
            .flatten()
            .rev()
            .collect::<Vec<_>>();

        fn visit_index<F: JoltField>(
            index: &Edge,
            tree: &[&Node],
            vars: &[F],
            scalars: &[Option<Scalar>],
        ) -> F {
            match index {
                Edge::Atom(a) => a.evaluate(vars, scalars),
                Edge::NodeRef(i) => visit_subtree(&tree[*i as usize..], vars, scalars),
            }
        }

        fn visit_subtree<F: JoltField>(
            tree: &[&Node],
            vars: &[F],
            scalars: &[Option<Scalar>],
        ) -> F {
            match tree[0] {
                Node::Atom(a) => a.evaluate(vars, scalars),
                Node::Neg(i) => -visit_index(i, tree, vars, scalars),
                Node::Inv(i) => F::one() / visit_index(i, tree, vars, scalars),
                Node::Add(i1, i2) => {
                    visit_index(i1, tree, vars, scalars) + visit_index(i2, tree, vars, scalars)
                }
                Node::Mul(i1, i2) => {
                    visit_index(i1, tree, vars, scalars) * visit_index(i2, tree, vars, scalars)
                }
                Node::Sub(i1, i2) => {
                    visit_index(i1, tree, vars, scalars) - visit_index(i2, tree, vars, scalars)
                }
                Node::Div(i1, i2) => {
                    visit_index(i1, tree, vars, scalars) / visit_index(i2, tree, vars, scalars)
                }
            }
        }

        visit_subtree(&nodes, vars, &self.scalars)
    }
}

impl<const NUM_NODES: usize, const NUM_SCALARS: usize> Zero for MleAst<NUM_NODES, NUM_SCALARS> {
    fn zero() -> Self {
        Self::new_scalar(0)
    }

    fn is_zero(&self) -> bool {
        self.nodes[self.root] == Some(Node::Atom(Atom::Scalar(0)))
    }
}

impl<const NUM_NODES: usize, const NUM_SCALARS: usize> One for MleAst<NUM_NODES, NUM_SCALARS> {
    fn one() -> Self {
        Self::new_scalar(1)
    }
}

impl<const NUM_NODES: usize, const NUM_SCALARS: usize> std::ops::Neg
    for MleAst<NUM_NODES, NUM_SCALARS>
{
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.unop(Node::Neg);
        self
    }
}

impl<const NUM_NODES: usize, const NUM_SCALARS: usize> std::ops::Add
    for MleAst<NUM_NODES, NUM_SCALARS>
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self + &rhs
    }
}

impl<const NUM_NODES: usize, const NUM_SCALARS: usize> std::ops::Sub
    for MleAst<NUM_NODES, NUM_SCALARS>
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self - &rhs
    }
}

impl<const NUM_NODES: usize, const NUM_SCALARS: usize> std::ops::Mul
    for MleAst<NUM_NODES, NUM_SCALARS>
{
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self * &rhs
    }
}

impl<const NUM_NODES: usize, const NUM_SCALARS: usize> std::ops::Div
    for MleAst<NUM_NODES, NUM_SCALARS>
{
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        self / &rhs
    }
}

impl<const NUM_NODES: usize, const NUM_SCALARS: usize> FieldOps for MleAst<NUM_NODES, NUM_SCALARS> {}

impl<const NUM_NODES: usize, const NUM_SCALARS: usize> std::ops::Add<&Self>
    for MleAst<NUM_NODES, NUM_SCALARS>
{
    type Output = Self;

    fn add(mut self, rhs: &Self) -> Self::Output {
        self.binop(Node::Add, rhs);
        self
    }
}

impl<const NUM_NODES: usize, const NUM_SCALARS: usize> std::ops::Sub<&Self>
    for MleAst<NUM_NODES, NUM_SCALARS>
{
    type Output = Self;

    fn sub(mut self, rhs: &Self) -> Self::Output {
        self.binop(Node::Sub, rhs);
        self
    }
}

impl<const NUM_NODES: usize, const NUM_SCALARS: usize> std::ops::Mul<&Self>
    for MleAst<NUM_NODES, NUM_SCALARS>
{
    type Output = Self;

    fn mul(mut self, rhs: &Self) -> Self::Output {
        self.binop(Node::Mul, rhs);
        self
    }
}

impl<const NUM_NODES: usize, const NUM_SCALARS: usize> std::ops::Div<&Self>
    for MleAst<NUM_NODES, NUM_SCALARS>
{
    type Output = Self;

    fn div(mut self, rhs: &Self) -> Self::Output {
        self.binop(Node::Div, rhs);
        self
    }
}

impl<const NUM_NODES: usize, const NUM_SCALARS: usize> FieldOps<&Self, Self>
    for MleAst<NUM_NODES, NUM_SCALARS>
{
}

impl<const NUM_NODES: usize, const NUM_SCALARS: usize> std::ops::AddAssign
    for MleAst<NUM_NODES, NUM_SCALARS>
{
    fn add_assign(&mut self, rhs: Self) {
        self.binop(Node::Add, &rhs);
    }
}

impl<const NUM_NODES: usize, const NUM_SCALARS: usize> std::ops::SubAssign
    for MleAst<NUM_NODES, NUM_SCALARS>
{
    fn sub_assign(&mut self, rhs: Self) {
        self.binop(Node::Sub, &rhs);
    }
}

impl<const NUM_NODES: usize, const NUM_SCALARS: usize> std::ops::MulAssign
    for MleAst<NUM_NODES, NUM_SCALARS>
{
    fn mul_assign(&mut self, rhs: Self) {
        self.binop(Node::Mul, &rhs);
    }
}

impl<const NUM_NODES: usize, const NUM_SCALARS: usize> core::iter::Sum
    for MleAst<NUM_NODES, NUM_SCALARS>
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|sum, term| sum + term).unwrap_or(Self::zero())
    }
}

impl<'a, const NUM_NODES: usize, const NUM_SCALARS: usize> core::iter::Sum<&'a Self>
    for MleAst<NUM_NODES, NUM_SCALARS>
{
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.cloned().sum()
    }
}

impl<const NUM_NODES: usize, const NUM_SCALARS: usize> core::iter::Product
    for MleAst<NUM_NODES, NUM_SCALARS>
{
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|product, factor| product * factor)
            .unwrap_or(Self::one())
    }
}

impl<'a, const NUM_NODES: usize, const NUM_SCALARS: usize> core::iter::Product<&'a Self>
    for MleAst<NUM_NODES, NUM_SCALARS>
{
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.cloned().product()
    }
}

/// Displays the AST as an algebraic formula. Variables are displayed as `name[index]`.
impl<const NUM_NODES: usize, const NUM_SCALARS: usize> std::fmt::Display
    for MleAst<NUM_NODES, NUM_SCALARS>
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Reversed nodes; indices are now reverse de Bruijn indices (positive relative to root)
        let nodes = self.nodes[..=self.root]
            .iter()
            .flatten()
            .rev()
            .collect::<Vec<_>>();

        fn visit_atom(
            f: &mut std::fmt::Formatter<'_>,
            atom: &Atom,
            reg_name: Option<char>,
            scalars: &[Option<Scalar>],
        ) -> std::fmt::Result {
            match atom {
                Atom::Scalar(index) => {
                    let scalar = scalars[*index as usize].expect("Invalid scalar ref");
                    write!(f, "{scalar}")?;
                }
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
            scalars: &[Option<Scalar>],
            group: bool, // whether to parenthesize this expression
        ) -> std::fmt::Result {
            match index {
                Edge::Atom(a) => visit_atom(f, a, reg_name, scalars)?,
                Edge::NodeRef(i) => {
                    visit_subtree(f, &tree[*i as usize..], reg_name, scalars, group)?
                }
            }

            Ok(())
        }

        fn visit_subtree(
            f: &mut std::fmt::Formatter<'_>,
            tree: &[&Node],
            reg_name: Option<char>,
            scalars: &[Option<Scalar>],
            group: bool, // whether to parenthesize this expression
        ) -> std::fmt::Result {
            match tree[0] {
                Node::Atom(a) => visit_atom(f, a, reg_name, scalars)?,
                Node::Neg(i) => {
                    write!(f, "-")?;
                    visit_index(f, i, tree, reg_name, scalars, true)?;
                }
                Node::Inv(i) => {
                    write!(f, "1 / ")?;
                    visit_index(f, i, tree, reg_name, scalars, true)?;
                }
                Node::Add(i1, i2) => {
                    if group {
                        write!(f, "(")?;
                    }
                    visit_index(f, i1, tree, reg_name, scalars, false)?;
                    write!(f, " + ")?;
                    visit_index(f, i2, tree, reg_name, scalars, false)?;
                    if group {
                        write!(f, ")")?;
                    }
                }
                Node::Mul(i1, i2) => {
                    visit_index(f, i1, tree, reg_name, scalars, true)?;
                    write!(f, " * ")?;
                    visit_index(f, i2, tree, reg_name, scalars, true)?;
                }
                Node::Sub(i1, i2) => {
                    if group {
                        write!(f, "(")?;
                    }
                    visit_index(f, i1, tree, reg_name, scalars, false)?;
                    write!(f, " - ")?;
                    visit_index(f, i2, tree, reg_name, scalars, true)?;
                    if group {
                        write!(f, ")")?;
                    }
                }
                Node::Div(i1, i2) => {
                    visit_index(f, i1, tree, reg_name, scalars, true)?;
                    write!(f, " / ")?;
                    visit_index(f, i2, tree, reg_name, scalars, true)?;
                }
            }

            Ok(())
        }

        visit_subtree(f, &nodes, self.reg_name, &self.scalars, false)
    }
}

impl<const NUM_NODES: usize, const NUM_SCALARS: usize> Default for MleAst<NUM_NODES, NUM_SCALARS> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const NUM_NODES: usize, const NUM_SCALARS: usize> std::hash::Hash
    for MleAst<NUM_NODES, NUM_SCALARS>
{
    fn hash<H: std::hash::Hasher>(&self, _state: &mut H) {
        unimplemented!("hash unimplemented for MleAst<NUM_NODES>")
    }
}

impl<const NUM_NODES: usize, const NUM_SCALARS: usize> JoltField
    for MleAst<NUM_NODES, NUM_SCALARS>
{
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

impl<const NUM_NODES: usize, const NUM_SCALARS: usize> CanonicalSerialize
    for MleAst<NUM_NODES, NUM_SCALARS>
{
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

impl<const NUM_NODES: usize, const NUM_SCALARS: usize> CanonicalDeserialize
    for MleAst<NUM_NODES, NUM_SCALARS>
{
    fn deserialize_with_mode<R: std::io::Read>(
        _reader: R,
        _compress: ark_serialize::Compress,
        _validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        unimplemented!("Not needed for constructing ASTs")
    }
}

impl<const NUM_NODES: usize, const NUM_SCALARS: usize> Valid for MleAst<NUM_NODES, NUM_SCALARS> {
    fn check(&self) -> Result<(), SerializationError> {
        unimplemented!("Not needed for constructing ASTs")
    }
}

/**********************************************************************/
