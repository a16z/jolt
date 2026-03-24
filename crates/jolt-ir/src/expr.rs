/// Variable reference in a claim expression.
///
/// Variables are either polynomial openings (bound at evaluation time) or
/// verifier challenges (Fiat-Shamir derived). The `u32` index is scoped to a
/// single `ClaimDefinition` — downstream code maps it to concrete polynomial
/// or challenge identifiers via binding metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Var {
    Opening(u32),
    Challenge(u32),
}

/// Expression node in the arena.
///
/// Each node is a single operation referencing child nodes by `ExprId`.
///
/// # Why constants are `i128`, not `F`
///
/// The IR is **field-agnostic** — it represents symbolic structure, not field
/// arithmetic. Constants are stored as `i128` and promoted to `F` only at the
/// backend boundary (e.g., `F::from_i128(val)` in the evaluate visitor).
///
/// This works because claim formula constants are always small structural
/// integers: `0`, `1`, `-1`, register counts, chunk sizes, etc. Actual
/// field-sized values (gamma powers, eq polynomial evaluations, batching
/// coefficients) enter the expression as `Var::Challenge` variables, resolved
/// to `F` at evaluation time via challenge value arrays.
///
/// If a future claim formula needs a compile-time field constant larger than
/// `i128`, model it as a `Var::Challenge` with a derived challenge value.
///
/// No `Div` or `Inv` variants — claim expressions never require division.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ExprNode {
    Constant(i128),
    Var(Var),
    Neg(ExprId),
    Add(ExprId, ExprId),
    Sub(ExprId, ExprId),
    Mul(ExprId, ExprId),
}

/// Stable index into an `ExprArena`. Small, `Copy`, hashable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ExprId(pub(crate) u32);

impl ExprId {
    #[inline]
    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// Arena storage for expression nodes.
///
/// Instance-local — no global mutable state. Each `ExprBuilder` owns one arena
/// which is moved into the final `Expr` on `build()`.
#[derive(Debug, Clone)]
pub struct ExprArena {
    nodes: Vec<ExprNode>,
}

impl ExprArena {
    pub(crate) fn new() -> Self {
        Self {
            nodes: Vec::with_capacity(16),
        }
    }

    pub(crate) fn push(&mut self, node: ExprNode) -> ExprId {
        let id = ExprId(self.nodes.len() as u32);
        self.nodes.push(node);
        id
    }

    #[inline]
    pub fn get(&self, id: ExprId) -> ExprNode {
        self.nodes[id.index()]
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

/// A complete expression tree: an arena of nodes plus a root reference.
///
/// Constructed via `ExprBuilder::build()`. Immutable after construction.
#[derive(Debug, Clone)]
pub struct Expr {
    pub(crate) arena: ExprArena,
    pub(crate) root: ExprId,
}

impl Expr {
    /// The root node id.
    #[inline]
    pub fn root(&self) -> ExprId {
        self.root
    }

    /// The underlying arena.
    #[inline]
    pub fn arena(&self) -> &ExprArena {
        &self.arena
    }

    /// Number of nodes in the expression.
    #[inline]
    pub fn len(&self) -> usize {
        self.arena.len()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.arena.is_empty()
    }

    /// Look up a node by id.
    #[inline]
    pub fn get(&self, id: ExprId) -> ExprNode {
        self.arena.get(id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arena_push_and_get() {
        let mut arena = ExprArena::new();
        let c = arena.push(ExprNode::Constant(42));
        let v = arena.push(ExprNode::Var(Var::Opening(0)));
        let add = arena.push(ExprNode::Add(c, v));

        assert_eq!(arena.len(), 3);
        assert_eq!(arena.get(c), ExprNode::Constant(42));
        assert_eq!(arena.get(v), ExprNode::Var(Var::Opening(0)));
        assert_eq!(arena.get(add), ExprNode::Add(c, v));
    }

    #[test]
    fn expr_id_index() {
        assert_eq!(ExprId(0).index(), 0);
        assert_eq!(ExprId(7).index(), 7);
    }

    #[test]
    fn expr_id_equality() {
        assert_eq!(ExprId(3), ExprId(3));
        assert_ne!(ExprId(0), ExprId(1));
    }
}
