//! `SymbolicField` — a symbolic representation of BN254 field elements.
//!
//! `SymbolicField` is a `Copy` type (4 bytes) that records all arithmetic into
//! the global AST arena. Constants are folded eagerly; operations involving at
//! least one symbolic value produce arena nodes.
//!
//! Implements [`jolt_field::Field`] so it can be used anywhere a concrete field
//! element would be used — the verifier becomes a symbolic execution trace.

use std::fmt::{self, Debug, Display};
use std::hash::{Hash, Hasher};
use std::ops::{Add, AddAssign, Div, Mul, MulAssign, Neg, Sub, SubAssign};

use num_traits::{One, Zero};
use rand_core::RngCore;
use serde::{Deserialize, Serialize};

use jolt_field::Field;

use crate::arena::{self, Atom, Edge, Node, NodeId};
use crate::scalar_ops;
use crate::tunneling;

/// A symbolic BN254 field element.
///
/// Either an inline constant (`Scalar`) or a reference to an arena node. All
/// arithmetic operations produce new symbolic values, recording the computation
/// graph in the global arena.
///
/// # Copy semantics
///
/// `Field` requires `Copy`, so this is a 40-byte value type (1 byte tag + 32
/// bytes payload for `[u64; 4]`, or 1 byte tag + 4 bytes for `NodeId`).
/// The actual `Edge` enum is small enough to be efficiently copied.
#[derive(Clone, Copy)]
pub struct SymbolicField {
    pub(crate) edge: Edge,
}

impl SymbolicField {
    pub fn constant(val: [u64; 4]) -> Self {
        Self {
            edge: Atom::Scalar(val),
        }
    }

    pub fn from_node(id: NodeId) -> Self {
        Self {
            edge: Atom::Node(id),
        }
    }

    pub fn variable(index: u32, name: impl Into<String>) -> Self {
        let id = arena::alloc(Node::Var {
            index,
            name: name.into(),
        });
        Self {
            edge: Atom::Node(id),
        }
    }

    pub fn from_edge(edge: Edge) -> Self {
        Self { edge }
    }

    pub fn into_edge(self) -> Edge {
        self.edge
    }

    pub fn is_constant(&self) -> bool {
        matches!(self.edge, Atom::Scalar(_))
    }

    pub fn as_constant(&self) -> Option<[u64; 4]> {
        match self.edge {
            Atom::Scalar(val) => Some(val),
            Atom::Node(_) => None,
        }
    }
}

impl Add for SymbolicField {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            edge: arena::add_edges(self.edge, rhs.edge),
        }
    }
}

impl<'a> Add<&'a Self> for SymbolicField {
    type Output = Self;
    fn add(self, rhs: &'a Self) -> Self {
        self + *rhs
    }
}

impl Sub for SymbolicField {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self {
            edge: arena::sub_edges(self.edge, rhs.edge),
        }
    }
}

impl<'a> Sub<&'a Self> for SymbolicField {
    type Output = Self;
    fn sub(self, rhs: &'a Self) -> Self {
        self - *rhs
    }
}

impl Mul for SymbolicField {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            edge: arena::mul_edges(self.edge, rhs.edge),
        }
    }
}

impl<'a> Mul<&'a Self> for SymbolicField {
    type Output = Self;
    fn mul(self, rhs: &'a Self) -> Self {
        self * *rhs
    }
}

impl Div for SymbolicField {
    type Output = Self;
    fn div(self, rhs: Self) -> Self {
        Self {
            edge: arena::div_edges(self.edge, rhs.edge),
        }
    }
}

impl<'a> Div<&'a Self> for SymbolicField {
    type Output = Self;
    fn div(self, rhs: &'a Self) -> Self {
        self / *rhs
    }
}

impl Neg for SymbolicField {
    type Output = Self;
    fn neg(self) -> Self {
        Self {
            edge: arena::neg_edge(self.edge),
        }
    }
}

impl AddAssign for SymbolicField {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl SubAssign for SymbolicField {
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl MulAssign for SymbolicField {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl std::iter::Sum for SymbolicField {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::zero(), |acc, x| acc + x)
    }
}

impl<'a> std::iter::Sum<&'a Self> for SymbolicField {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.copied().sum()
    }
}

impl std::iter::Product for SymbolicField {
    fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::one(), |acc, x| acc * x)
    }
}

impl<'a> std::iter::Product<&'a Self> for SymbolicField {
    fn product<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.copied().product()
    }
}

impl PartialEq for SymbolicField {
    fn eq(&self, other: &Self) -> bool {
        match (self.edge, other.edge) {
            (Atom::Scalar(a), Atom::Scalar(b)) => a == b,
            (Atom::Node(a), Atom::Node(b)) => a == b,
            _ => false,
        }
    }
}

impl Eq for SymbolicField {}

impl Hash for SymbolicField {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.edge.hash(state);
    }
}

impl Debug for SymbolicField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.edge {
            Atom::Scalar(val) => write!(f, "Const({})", scalar_ops::to_decimal_string(val)),
            Atom::Node(NodeId(id)) => write!(f, "Node({id})"),
        }
    }
}

impl Display for SymbolicField {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Debug::fmt(self, f)
    }
}

impl Default for SymbolicField {
    fn default() -> Self {
        Self::zero()
    }
}

impl Zero for SymbolicField {
    fn zero() -> Self {
        Self::constant(scalar_ops::ZERO)
    }

    fn is_zero(&self) -> bool {
        matches!(self.edge, Atom::Scalar(val) if scalar_ops::is_zero(val))
    }
}

impl One for SymbolicField {
    fn one() -> Self {
        Self::constant(scalar_ops::ONE)
    }

    fn is_one(&self) -> bool {
        matches!(self.edge, Atom::Scalar(val) if scalar_ops::is_one(val))
    }
}

impl Serialize for SymbolicField {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self.edge {
            Atom::Scalar(val) => {
                let bytes = scalar_ops::to_bytes_le(val);
                bytes.serialize(serializer)
            }
            Atom::Node(NodeId(id)) => {
                // Encode as negative to distinguish from scalar bytes
                let marker: i64 = -(id as i64) - 1;
                marker.serialize(serializer)
            }
        }
    }
}

impl<'de> Deserialize<'de> for SymbolicField {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // For deserialization, we only support scalars (used for constant inputs)
        let bytes: Vec<u8> = Vec::deserialize(deserializer)?;
        let val = scalar_ops::from_bytes_le(&bytes);
        Ok(Self::constant(val))
    }
}

impl Field for SymbolicField {
    type Accumulator = jolt_field::NaiveAccumulator<Self>;

    const NUM_BYTES: usize = 32;

    fn to_bytes(&self) -> [u8; 32] {
        // Store the symbolic edge in thread-local for the transcript tunnel
        tunneling::set_pending_append(self.edge);

        // Return the constant bytes if available, otherwise dummy bytes
        match self.edge {
            Atom::Scalar(val) => scalar_ops::to_bytes_le(val),
            Atom::Node(_) => [0u8; 32],
        }
    }

    fn random<R: RngCore>(_rng: &mut R) -> Self {
        // Symbolic execution doesn't use random values — this shouldn't be called
        // during symbolic tracing. Return zero as a safe fallback.
        Self::zero()
    }

    fn from_bytes(bytes: &[u8]) -> Self {
        Self::constant(scalar_ops::from_bytes_le(bytes))
    }

    fn to_u64(&self) -> Option<u64> {
        match self.edge {
            Atom::Scalar(val) => {
                if val[1] == 0 && val[2] == 0 && val[3] == 0 {
                    Some(val[0])
                } else {
                    None
                }
            }
            Atom::Node(_) => None,
        }
    }

    fn num_bits(&self) -> u32 {
        match self.edge {
            Atom::Scalar(val) => scalar_ops::num_bits(val),
            Atom::Node(_) => 254, // assume worst case for symbolic
        }
    }

    fn square(&self) -> Self {
        *self * *self
    }

    fn inverse(&self) -> Option<Self> {
        match self.edge {
            Atom::Scalar(val) => scalar_ops::inv(val).map(Self::constant),
            Atom::Node(_) => Some(Self {
                edge: arena::inv_edge(self.edge),
            }),
        }
    }

    fn from_u64(n: u64) -> Self {
        Self::constant(scalar_ops::from_u64(n))
    }

    fn from_i64(val: i64) -> Self {
        Self::constant(scalar_ops::from_i64(val))
    }

    fn from_i128(val: i128) -> Self {
        Self::constant(scalar_ops::from_i128(val))
    }

    fn from_u128(val: u128) -> Self {
        // Check if a challenge was tunneled
        if let Some(edge) = tunneling::take_pending_challenge() {
            return Self { edge };
        }
        Self::constant(scalar_ops::from_u128(val))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::ArenaSession;

    #[test]
    fn constant_arithmetic() {
        let _session = ArenaSession::new();

        let a = SymbolicField::from_u64(10);
        let b = SymbolicField::from_u64(7);

        let sum = a + b;
        assert_eq!(sum.as_constant().unwrap(), scalar_ops::from_u64(17));

        let diff = a - b;
        assert_eq!(diff.as_constant().unwrap(), scalar_ops::from_u64(3));

        let prod = a * b;
        assert_eq!(prod.as_constant().unwrap(), scalar_ops::from_u64(70));

        let quot = prod / b;
        assert_eq!(quot.as_constant().unwrap(), scalar_ops::from_u64(10));
    }

    #[test]
    fn symbolic_arithmetic_creates_nodes() {
        let _session = ArenaSession::new();

        let x = SymbolicField::variable(0, "x");
        let y = SymbolicField::variable(1, "y");

        assert!(!x.is_constant());
        assert!(!y.is_constant());

        let sum = x + y;
        assert!(!sum.is_constant());

        // 2 vars + 1 add = 3 nodes
        assert_eq!(arena::node_count(), 3);
    }

    #[test]
    fn mixed_constant_symbolic() {
        let _session = ArenaSession::new();

        let x = SymbolicField::variable(0, "x");
        let three = SymbolicField::from_u64(3);

        let result = x * three;
        assert!(!result.is_constant());

        // x * 1 should just return x (identity optimization)
        let one = SymbolicField::one();
        let result2 = x * one;
        assert_eq!(result2.edge, x.edge);
    }

    #[test]
    fn zero_one_identity() {
        let _session = ArenaSession::new();

        assert!(SymbolicField::zero().is_zero());
        assert!(SymbolicField::one().is_one());
        assert!(!SymbolicField::zero().is_one());
        assert!(!SymbolicField::one().is_zero());
    }

    #[test]
    fn negation() {
        let _session = ArenaSession::new();

        let five = SymbolicField::from_u64(5);
        let neg_five = -five;
        let sum = five + neg_five;
        assert!(sum.is_zero());
    }

    #[test]
    fn from_i64_negative() {
        let _session = ArenaSession::new();

        let neg = SymbolicField::from_i64(-1);
        let one = SymbolicField::one();
        let sum = neg + one;
        assert!(sum.is_zero());
    }

    #[test]
    fn tunneling_challenge() {
        let _session = ArenaSession::new();

        let challenge_edge = Atom::Node(arena::alloc(Node::Challenge { id: 42 }));
        tunneling::set_pending_challenge(challenge_edge);

        // from_u128 should pick up the tunneled challenge
        let field_val = SymbolicField::from_u128(12345);
        assert_eq!(field_val.edge, challenge_edge);
    }

    #[test]
    fn tunneling_append() {
        let _session = ArenaSession::new();

        let x = SymbolicField::variable(0, "x");
        let _bytes = x.to_bytes();

        // The append tunnel should have been set
        let pending = tunneling::take_pending_append();
        assert_eq!(pending, Some(x.edge));
    }

    #[test]
    fn sum_product_iterators() {
        let _session = ArenaSession::new();

        let vals: Vec<SymbolicField> = (1..=5).map(SymbolicField::from_u64).collect();

        let sum: SymbolicField = vals.iter().sum();
        assert_eq!(sum.as_constant().unwrap(), scalar_ops::from_u64(15));

        let product: SymbolicField = vals.iter().product();
        assert_eq!(product.as_constant().unwrap(), scalar_ops::from_u64(120));
    }

    #[test]
    fn display_debug() {
        let _session = ArenaSession::new();

        let c = SymbolicField::from_u64(42);
        assert_eq!(format!("{c:?}"), "Const(42)");

        let x = SymbolicField::variable(0, "x");
        assert!(format!("{x:?}").starts_with("Node("));
    }
}
