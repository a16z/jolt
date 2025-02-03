use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError, Valid};
use ark_std::{One, Zero};
use jolt_core::field::{FieldOps, JoltField};

use std::fmt::Write;

/// A node for a polynomial AST, where children are represented by de Bruijn indices (negative
/// relative to the parent) into an array of nodes.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum MleAstNode {
    /// A constant value. We use i128 here in order to support negative scalars.
    Scalar(i128),
    /// A variable, represented by a one-character register name and an index into that register
    // XXX: We use a one-char name here because this data structure needs to be Copy and Sized,
    // limiting our use of string-like objects.
    Var(char, usize),
    /// The negation of a node
    Neg(usize),
    /// The multiplicative inverse of a node
    Inv(usize),
    /// The sum of two nodes
    Add(usize, usize),
    /// The product of two nodes
    Mul(usize, usize),
    /// The difference between the first and second nodes
    Sub(usize, usize),
    /// The quotient between the first and second nodes
    /// NOTE: No div-by-zero checks are performed here
    Div(usize, usize),
}

/// An AST intended for representing an MLE computation (although it will actually work for any
/// multivariate polynomial). The nodes are stored in a statically sized array, which allows the
/// data structure to be [`Copy`] and [`Sized`]. The size of the array (i.e., the max number of
/// nodes that may be stored) is given by the const-generic `NUM_NODES` type argument.
#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub struct MleAst<const NUM_NODES: usize> {
    /// Collection of nodes; the indices in each [`MleAstNode`] are indices into this array
    nodes: [Option<MleAstNode>; NUM_NODES],
    /// Index of the root of the AST (should always be the last used node)
    root: usize,
}

impl<const NUM_NODES: usize> MleAst<NUM_NODES> {
    /// Construct a new AST with the given node as the root.
    fn new_with_root(root: MleAstNode) -> Self {
        let nodes = std::array::from_fn(|i| {
            match i {
                0 => Some(root),
                _ => None,
            }
        });
        let root = 0;
        Self { nodes, root }
    }

    /// Append the nodes of `other` onto the array of nodes in `self`. The root of `self` will be
    /// the root the appended copy of `other` (the right tree). Return the indices of the left and
    /// right trees in the node array of `self`.
    fn concatenate(&mut self, other: Self) -> (usize, usize) {
        let shift = self.root+1;
        for i in 0..=other.root {
            self.nodes[shift+i] = other.nodes[i];
            self.root += 1;
        }

        // de Bruijn indices, i.e., negative relative to the new root
        let left_root = other.root + 2;
        let right_root = 1;

        (left_root, right_root)
    }

    /// Create a new root node in the form of a unitary operator.
    fn unop(&mut self, constructor: impl FnOnce(usize) -> MleAstNode) {
        let required_nodes = self.root + 1;
        assert!(required_nodes < NUM_NODES,
            "Ran out of space for nodes. Try increasing NUM_NODES from {NUM_NODES} to at least {required_nodes}.");
        self.nodes[self.root+1] = Some(constructor(self.root));
        self.root += 1;
    }

    /// Create a new root node in the form of a binary operator.
    fn binop(
        &mut self,
        constructor: impl FnOnce(usize, usize) -> MleAstNode,
        rhs: Self,
    ) {
        let required_nodes = self.root + rhs.root + 1;
        assert!(required_nodes < NUM_NODES,
            "Ran out of space for nodes. Try increasing NUM_NODES from {NUM_NODES} to at least {required_nodes}.");
        let (lhs_root, rhs_root) = self.concatenate(rhs);
        self.nodes[self.root+1] = Some(constructor(lhs_root, rhs_root));
        self.root += 1;
    }
}

impl<const NUM_NODES: usize> crate::util::ZkLeanReprField for MleAst<NUM_NODES> {
    fn register(name: char, size: usize) -> Vec<Self> {
        (0..size).map(|i| Self::new_with_root(MleAstNode::Var(name, i))).collect()
    }

    fn as_computation(&self) -> String {
        let mut res = "".to_string();
        write!(res, "{self}").unwrap();
        res
    }
}

impl<const NUM_NODES: usize> Zero for MleAst<NUM_NODES> {
    fn zero() -> Self {
        Self::new_with_root(MleAstNode::Scalar(0))
    }

    fn is_zero(&self) -> bool {
        self.nodes[0] == Some(MleAstNode::Scalar(0))
    }
}

impl<const NUM_NODES: usize> One for MleAst<NUM_NODES> {
    fn one() -> Self {
        Self::new_with_root(MleAstNode::Scalar(1))
    }
}

impl<const NUM_NODES: usize> std::ops::Neg for MleAst<NUM_NODES> {
    type Output = Self;

    fn neg(mut self) -> Self::Output {
        self.unop(MleAstNode::Neg);
        self
    }
}

impl<const NUM_NODES: usize> std::ops::Add for MleAst<NUM_NODES> {
    type Output = Self;

    fn add(mut self, rhs: Self) -> Self::Output {
        self.binop(MleAstNode::Add, rhs);
        self
    }
}

impl<const NUM_NODES: usize> std::ops::Sub for MleAst<NUM_NODES> {
    type Output = Self;

    fn sub(mut self, rhs: Self) -> Self::Output {
        self.binop(MleAstNode::Sub, rhs);
        self
    }
}

impl<const NUM_NODES: usize> std::ops::Mul for MleAst<NUM_NODES> {
    type Output = Self;

    fn mul(mut self, rhs: Self) -> Self::Output {
        self.binop(MleAstNode::Mul, rhs);
        self
    }
}

impl<const NUM_NODES: usize> std::ops::Div for MleAst<NUM_NODES> {
    type Output = Self;

    fn div(mut self, rhs: Self) -> Self::Output {
        self.binop(MleAstNode::Div, rhs);
        self
    }
}

impl<const NUM_NODES: usize> FieldOps for MleAst<NUM_NODES> {}

impl<'a, const NUM_NODES: usize> std::ops::Add<&'a Self> for MleAst<NUM_NODES> {
    type Output = Self;

    fn add(self, rhs: &Self) -> Self::Output {
        self + *rhs
    }
}

impl<'a, const NUM_NODES: usize> std::ops::Sub<&'a Self> for MleAst<NUM_NODES> {
    type Output = Self;

    fn sub(self, rhs: &Self) -> Self::Output {
        self - *rhs
    }
}

impl<'a, const NUM_NODES: usize> std::ops::Mul<&'a Self> for MleAst<NUM_NODES> {
    type Output = Self;

    fn mul(self, rhs: &Self) -> Self::Output {
        self * *rhs
    }
}

impl<'a, const NUM_NODES: usize> std::ops::Div<&'a Self> for MleAst<NUM_NODES> {
    type Output = Self;

    fn div(self, rhs: &Self) -> Self::Output {
        self / *rhs
    }
}

impl<'a, const NUM_NODES: usize> FieldOps<&'a Self, Self> for MleAst<NUM_NODES> {}

impl<const NUM_NODES: usize> std::ops::AddAssign for MleAst<NUM_NODES> {
    fn add_assign(&mut self, rhs: Self) {
        self.binop(MleAstNode::Add, rhs);
    }
}

impl<const NUM_NODES: usize> std::ops::SubAssign for MleAst<NUM_NODES> {
    fn sub_assign(&mut self, rhs: Self) {
        self.binop(MleAstNode::Sub, rhs);
    }
}

impl<const NUM_NODES: usize> std::ops::MulAssign for MleAst<NUM_NODES> {
    fn mul_assign(&mut self, rhs: Self) {
        self.binop(MleAstNode::Mul, rhs);
    }
}

impl<const NUM_NODES: usize> core::iter::Sum for MleAst<NUM_NODES> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|sum, term| sum + term)
            .unwrap_or(Self::zero())
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
            .unwrap_or(Self::zero())
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
        fn helper(
            f: &mut std::fmt::Formatter<'_>,
            nodes: &[Option<MleAstNode>],
            root: usize,
            group: bool, // whether to group a low-precedence op (+ or -) with parentheses
        ) -> std::fmt::Result {
            match nodes[root] {
                Some(MleAstNode::Scalar(scalar)) => write!(f, "{scalar}")?,
                Some(MleAstNode::Var(name, index)) => write!(f, "{name}[{index}]")?,
                Some(MleAstNode::Neg(index)) => {
                    write!(f, "-")?;
                    helper(f, nodes, root - index, true)?;
                }
                Some(MleAstNode::Inv(index)) => {
                    write!(f, "1/")?;
                    helper(f, nodes, root - index, true)?;
                }
                Some(MleAstNode::Add(lhs_index, rhs_index)) => {
                    if group {
                        write!(f, "(")?;
                    }
                    helper(f, nodes, root - lhs_index, false)?;
                    write!(f, " + ")?;
                    helper(f, nodes, root - rhs_index, false)?;
                    if group {
                        write!(f, ")")?;
                    }
                }
                Some(MleAstNode::Mul(lhs_index, rhs_index)) => {
                    helper(f, nodes, root - lhs_index, true)?;
                    write!(f, "*")?;
                    helper(f, nodes, root - rhs_index, true)?;
                }
                Some(MleAstNode::Sub(lhs_index, rhs_index)) => {
                    if group {
                        write!(f, "(")?;
                    }
                    helper(f, nodes, root - lhs_index, false)?;
                    write!(f, " - ")?;
                    helper(f, nodes, root - rhs_index, false)?;
                    if group {
                        write!(f, ")")?;
                    }
                }
                Some(MleAstNode::Div(lhs_index, rhs_index)) => {
                    helper(f, nodes, root - lhs_index, true)?;
                    write!(f, "/")?;
                    helper(f, nodes, root - rhs_index, true)?;
                }
                None => panic!("uninitialized node"),
            }
            Ok(())
        }

        helper(f, &self.nodes, self.root, false)
    }
}

impl<const NUM_NODES: usize> Default for MleAst<NUM_NODES> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<const NUM_NODES: usize> CanonicalSerialize for MleAst<NUM_NODES> {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        _writer: W,
        _compress: ark_serialize::Compress,
    ) -> Result<(), ark_serialize::SerializationError> {
        unimplemented!("ark serializer unimplemented for MleAst<NUM_NODES>")
    }

    fn serialized_size(&self, _compress: ark_serialize::Compress) -> usize {
        unimplemented!("ark serializer unimplemented for MleAst<NUM_NODES>")
    }
}

impl<const NUM_NODES: usize> Valid for MleAst<NUM_NODES> {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl<const NUM_NODES: usize> CanonicalDeserialize for MleAst<NUM_NODES> {
    fn deserialize_with_mode<R: std::io::Read>(
        _reader: R,
        _compress: ark_serialize::Compress,
        _validate: ark_serialize::Validate,
    ) -> Result<Self, ark_serialize::SerializationError> {
        unimplemented!("ark deserializer unimplemented for MleAst<NUM_NODES>")
    }
}

impl<const NUM_NODES: usize> std::hash::Hash for MleAst<NUM_NODES> {
    fn hash<H: std::hash::Hasher>(&self, _state: &mut H) {
        unimplemented!("hash unimplemented for MleAst<NUM_NODES>")
    }
}

impl<const NUM_NODES: usize> JoltField for MleAst<NUM_NODES> {
    const NUM_BYTES: usize = 0;

    //type SmallValueLookupTables = ();

    fn random<R: rand_core::RngCore>(_rng: &mut R) -> Self {
        unimplemented!("Not needed for constructing ASTs");
    }

    fn from_u8(n: u8) -> Self {
        Self::new_with_root(MleAstNode::Scalar(n as i128))
    }

    fn from_u16(n: u16) -> Self {
        Self::new_with_root(MleAstNode::Scalar(n as i128))
    }

    fn from_u32(n: u32) -> Self {
        Self::new_with_root(MleAstNode::Scalar(n as i128))
    }

    fn from_u64(n: u64) -> Self {
        Self::new_with_root(MleAstNode::Scalar(n as i128))
    }

    fn from_i64(val: i64) -> Self {
        Self::new_with_root(MleAstNode::Scalar(val as i128))
    }

    fn from_i128(val: i128) -> Self {
        Self::new_with_root(MleAstNode::Scalar(val as i128))
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
            res.unop(MleAstNode::Inv);
            Some(res)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::util::ZkLeanReprField;
    use crate::util::test::{Evaluatable, make_subtable_test_module};
    use binius_field::BinaryField128b;
    use jolt_core::field::binius::BiniusField;
    use proptest::prelude::*;

    type ReferenceField = BiniusField<BinaryField128b>;
    type TestField = MleAst<2048>;

    /// Evaluate the computation represented by the AST over another [`JoltField`], starting at
    /// `root`, and using the variable assignments in `vars`.
    fn evaluate_helper<F: JoltField>(
        vars: &[F],
        nodes: &[Option<MleAstNode>],
        root: usize,
    ) -> F {
        match nodes[root] {
            Some(MleAstNode::Scalar(f)) => F::from_u64(f),
            Some(MleAstNode::Var(_, var)) =>
                vars[var], // TODO: handle multiple registers?
            Some(MleAstNode::Neg(next_root)) =>
                -evaluate_helper(vars, nodes, root - next_root),
            Some(MleAstNode::Inv(next_root)) =>
                evaluate_helper(vars, nodes, root - next_root).inverse().expect("division by 0"),
            Some(MleAstNode::Add(lhs_root, rhs_root)) =>
                evaluate_helper(vars, nodes, root - lhs_root) + evaluate_helper(vars, nodes, root - rhs_root),
            Some(MleAstNode::Mul(lhs_root, rhs_root)) =>
                evaluate_helper(vars, nodes, root - lhs_root) * evaluate_helper(vars, nodes, root - rhs_root),
            Some(MleAstNode::Sub(lhs_root, rhs_root)) =>
                evaluate_helper(vars, nodes, root - lhs_root) - evaluate_helper(vars, nodes, root - rhs_root),
            Some(MleAstNode::Div(lhs_root, rhs_root)) =>
                evaluate_helper(vars, nodes, root - lhs_root) / evaluate_helper(vars, nodes, root - rhs_root),
            None => panic!("unreachable")
        }
    }

    impl<const NUM_NODES: usize> Evaluatable for MleAst<NUM_NODES> {
        fn evaluate<F: JoltField>(&self, vars: &[F]) -> F {
            evaluate_helper(vars, &self.nodes, self.root)
        }

        fn x_register(size: usize) -> Vec<Self> {
            Self::register('x', size)
        }
    }

    make_subtable_test_module!(TestField, ReferenceField);
}
