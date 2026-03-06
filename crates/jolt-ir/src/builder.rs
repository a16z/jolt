use std::cell::RefCell;
use std::ops;

use crate::expr::{Expr, ExprArena, ExprId, ExprNode, Var};

/// Ergonomic builder for constructing `Expr` trees.
///
/// Uses interior mutability so that multiple `ExprHandle`s can coexist without
/// borrow conflicts. This matters because developers write claim definitions
/// as natural arithmetic expressions (`gamma * (h * h - h)`) where several
/// handles must be live simultaneously.
///
/// # Example
///
/// ```
/// use jolt_ir::ExprBuilder;
///
/// let b = ExprBuilder::new();
/// let h = b.opening(0);
/// let gamma = b.challenge(0);
/// let expr = b.build(gamma * (h * h - h));
/// assert_eq!(expr.len(), 5); // h, gamma, h*h, h*h-h, gamma*(h*h-h)
/// ```
pub struct ExprBuilder {
    arena: RefCell<ExprArena>,
}

impl ExprBuilder {
    pub fn new() -> Self {
        Self {
            arena: RefCell::new(ExprArena::new()),
        }
    }

    fn push(&self, node: ExprNode) -> ExprId {
        self.arena.borrow_mut().push(node)
    }

    /// Reference a polynomial opening variable.
    pub fn opening(&self, id: u32) -> ExprHandle<'_> {
        let expr_id = self.push(ExprNode::Var(Var::Opening(id)));
        ExprHandle {
            builder: self,
            id: expr_id,
        }
    }

    /// Reference a verifier challenge variable.
    pub fn challenge(&self, id: u32) -> ExprHandle<'_> {
        let expr_id = self.push(ExprNode::Var(Var::Challenge(id)));
        ExprHandle {
            builder: self,
            id: expr_id,
        }
    }

    /// A constant value.
    pub fn constant(&self, val: i128) -> ExprHandle<'_> {
        let expr_id = self.push(ExprNode::Constant(val));
        ExprHandle {
            builder: self,
            id: expr_id,
        }
    }

    /// Shorthand for `constant(0)`.
    pub fn zero(&self) -> ExprHandle<'_> {
        self.constant(0)
    }

    /// Shorthand for `constant(1)`.
    pub fn one(&self) -> ExprHandle<'_> {
        self.constant(1)
    }

    /// Finalize the builder and produce an `Expr` rooted at the given handle.
    ///
    /// Takes `&self` (not `self`) so handles can remain live during the call.
    /// The arena is moved out via `RefCell::replace`; the builder is left empty
    /// and should not be reused.
    pub fn build(&self, root: ExprHandle<'_>) -> Expr {
        let arena = self.arena.replace(ExprArena::new());
        Expr {
            arena,
            root: root.id,
        }
    }
}

impl Default for ExprBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// A live reference to a node in an `ExprBuilder`'s arena.
///
/// Implements `Add`, `Sub`, `Mul`, and `Neg` so claim formulas read as
/// natural arithmetic. Each operator creates a new node in the arena and
/// returns a fresh handle.
///
/// Integer literals are supported on the right-hand side of `Add`, `Sub`,
/// and `Mul` (and left-hand side of `Mul`), so `2 * h` and `h + 1` work
/// without calling `b.constant(...)`.
#[derive(Clone, Copy)]
pub struct ExprHandle<'a> {
    pub(crate) builder: &'a ExprBuilder,
    pub(crate) id: ExprId,
}

impl ExprHandle<'_> {
    /// The underlying `ExprId` in the arena.
    #[inline]
    pub fn expr_id(self) -> ExprId {
        self.id
    }
}

impl<'a> ops::Add for ExprHandle<'a> {
    type Output = ExprHandle<'a>;

    fn add(self, rhs: Self) -> Self::Output {
        let id = self.builder.push(ExprNode::Add(self.id, rhs.id));
        ExprHandle {
            builder: self.builder,
            id,
        }
    }
}

impl<'a> ops::Sub for ExprHandle<'a> {
    type Output = ExprHandle<'a>;

    fn sub(self, rhs: Self) -> Self::Output {
        let id = self.builder.push(ExprNode::Sub(self.id, rhs.id));
        ExprHandle {
            builder: self.builder,
            id,
        }
    }
}

impl<'a> ops::Mul for ExprHandle<'a> {
    type Output = ExprHandle<'a>;

    fn mul(self, rhs: Self) -> Self::Output {
        let id = self.builder.push(ExprNode::Mul(self.id, rhs.id));
        ExprHandle {
            builder: self.builder,
            id,
        }
    }
}

impl<'a> ops::Neg for ExprHandle<'a> {
    type Output = ExprHandle<'a>;

    fn neg(self) -> Self::Output {
        let id = self.builder.push(ExprNode::Neg(self.id));
        ExprHandle {
            builder: self.builder,
            id,
        }
    }
}

impl<'a> ops::Add<i128> for ExprHandle<'a> {
    type Output = ExprHandle<'a>;

    fn add(self, rhs: i128) -> Self::Output {
        let rhs_id = self.builder.push(ExprNode::Constant(rhs));
        let id = self.builder.push(ExprNode::Add(self.id, rhs_id));
        ExprHandle {
            builder: self.builder,
            id,
        }
    }
}

impl<'a> ops::Sub<i128> for ExprHandle<'a> {
    type Output = ExprHandle<'a>;

    fn sub(self, rhs: i128) -> Self::Output {
        let rhs_id = self.builder.push(ExprNode::Constant(rhs));
        let id = self.builder.push(ExprNode::Sub(self.id, rhs_id));
        ExprHandle {
            builder: self.builder,
            id,
        }
    }
}

impl<'a> ops::Mul<i128> for ExprHandle<'a> {
    type Output = ExprHandle<'a>;

    fn mul(self, rhs: i128) -> Self::Output {
        let rhs_id = self.builder.push(ExprNode::Constant(rhs));
        let id = self.builder.push(ExprNode::Mul(self.id, rhs_id));
        ExprHandle {
            builder: self.builder,
            id,
        }
    }
}

impl<'a> ops::Mul<ExprHandle<'a>> for i128 {
    type Output = ExprHandle<'a>;

    fn mul(self, rhs: ExprHandle<'a>) -> Self::Output {
        let lhs_id = rhs.builder.push(ExprNode::Constant(self));
        let id = rhs.builder.push(ExprNode::Mul(lhs_id, rhs.id));
        ExprHandle {
            builder: rhs.builder,
            id,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::ExprNode;

    #[test]
    fn builder_basic_construction() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let c = b.constant(5);
        let expr = b.build(a + c);

        assert_eq!(expr.len(), 3);
        assert_eq!(expr.get(ExprId(0)), ExprNode::Var(Var::Opening(0)));
        assert_eq!(expr.get(ExprId(1)), ExprNode::Constant(5));
        match expr.get(expr.root()) {
            ExprNode::Add(l, r) => {
                assert_eq!(l, ExprId(0));
                assert_eq!(r, ExprId(1));
            }
            other => panic!("expected Add, got {other:?}"),
        }
    }

    #[test]
    fn builder_booleanity_formula() {
        // gamma * (h^2 - h)
        // h is Copy so h*h reuses the same ExprId — 5 nodes total
        let b = ExprBuilder::new();
        let h = b.opening(0);
        let gamma = b.challenge(0);
        let expr = b.build(gamma * (h * h - h));

        assert_eq!(expr.len(), 5);
        match expr.get(expr.root()) {
            ExprNode::Mul(_, _) => {}
            other => panic!("expected Mul at root, got {other:?}"),
        }
    }

    #[test]
    fn builder_negation() {
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let expr = b.build(-a);

        assert_eq!(expr.len(), 2);
        assert_eq!(expr.get(expr.root()), ExprNode::Neg(ExprId(0)));
    }

    #[test]
    fn builder_complex_expression() {
        // (a + b) * (c - d)
        let b = ExprBuilder::new();
        let a = b.opening(0);
        let bv = b.opening(1);
        let c = b.opening(2);
        let d = b.opening(3);
        let expr = b.build((a + bv) * (c - d));

        assert_eq!(expr.len(), 7);
    }

    #[test]
    fn builder_zero_and_one() {
        let b = ExprBuilder::new();
        let z = b.zero();
        let o = b.one();
        let expr = b.build(z + o);

        assert_eq!(expr.get(ExprId(0)), ExprNode::Constant(0));
        assert_eq!(expr.get(ExprId(1)), ExprNode::Constant(1));
    }

    #[test]
    fn integer_literal_mul_rhs() {
        // h * 2
        let b = ExprBuilder::new();
        let h = b.opening(0);
        let expr = b.build(h * 2);

        assert_eq!(expr.len(), 3);
        match expr.get(expr.root()) {
            ExprNode::Mul(l, r) => {
                assert_eq!(expr.get(l), ExprNode::Var(Var::Opening(0)));
                assert_eq!(expr.get(r), ExprNode::Constant(2));
            }
            other => panic!("expected Mul, got {other:?}"),
        }
    }

    #[test]
    fn integer_literal_mul_lhs() {
        // 3 * h
        let b = ExprBuilder::new();
        let h = b.opening(0);
        let expr = b.build(3i128 * h);

        assert_eq!(expr.len(), 3);
        match expr.get(expr.root()) {
            ExprNode::Mul(l, r) => {
                assert_eq!(expr.get(l), ExprNode::Constant(3));
                assert_eq!(expr.get(r), ExprNode::Var(Var::Opening(0)));
            }
            other => panic!("expected Mul, got {other:?}"),
        }
    }

    #[test]
    fn integer_literal_add_sub() {
        // h + 1 - 2
        let b = ExprBuilder::new();
        let h = b.opening(0);
        let expr = b.build(h + 1 - 2);

        assert_eq!(expr.len(), 5);
        match expr.get(expr.root()) {
            ExprNode::Sub(_, r) => {
                assert_eq!(expr.get(r), ExprNode::Constant(2));
            }
            other => panic!("expected Sub, got {other:?}"),
        }
    }
}
