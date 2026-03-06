//! Backend-agnostic AST bundle — captured constraint graph + named inputs.
//!
//! An [`AstBundle`] is the serializable intermediate representation between
//! symbolic execution (which fills the arena) and codegen (which emits via
//! [`AstEmitter`]).

use serde::{Deserialize, Serialize};

use crate::arena::{self, Atom, Edge, Node, NodeId};
use crate::ast_emitter::AstEmitter;

/// Whether an input variable is fixed for a given program or varies per proof.
///
/// Codegen backends use this to decide gnark struct tag annotations
/// (public vs. secret) or analogous distinctions in other targets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum WitnessKind {
    /// Varies per proof: commitments, sumcheck coefficients, opening claims.
    #[default]
    ProofData,
    /// Fixed for a given program: bytecode hash, memory layout, I/O hashes.
    PublicStatement,
}

/// A named input variable in the constraint system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputVar {
    /// Variable index (unique within the bundle).
    pub index: u32,
    /// Human-readable name for the variable.
    pub name: String,
    /// Whether this variable is proof data or a public statement.
    #[serde(default)]
    pub witness_kind: WitnessKind,
    /// Optional tag for non-native field variables.
    ///
    /// `None` means native scalar field. Backends interpret this tag to
    /// select appropriate arithmetic: e.g., gnark uses `std/math/emulated`
    /// for non-native fields, hash-based PCS may not need this at all.
    ///
    /// Examples: `Some("fq")` for BN254 base field (Dory commitment
    /// coordinates), `Some("ring_q")` for lattice ring elements.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub field_tag: Option<String>,
}

/// An assertion captured during symbolic execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Assertion {
    /// Assert that the edge evaluates to zero.
    Zero(SerializedEdge),
    /// Assert that two edges are equal.
    Equal(SerializedEdge, SerializedEdge),
}

/// Serializable representation of an arena edge.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SerializedEdge {
    Scalar([u64; 4]),
    Node(u32),
}

impl From<Edge> for SerializedEdge {
    fn from(edge: Edge) -> Self {
        match edge {
            Atom::Scalar(val) => SerializedEdge::Scalar(val),
            Atom::Node(NodeId(id)) => SerializedEdge::Node(id),
        }
    }
}

/// Serializable representation of an arena node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SerializedNode {
    Var {
        index: u32,
        name: String,
    },
    Neg(SerializedEdge),
    Inv(SerializedEdge),
    Add(SerializedEdge, SerializedEdge),
    Sub(SerializedEdge, SerializedEdge),
    Mul(SerializedEdge, SerializedEdge),
    Div(SerializedEdge, SerializedEdge),
    Challenge {
        id: u64,
    },
    Poseidon {
        state: SerializedEdge,
        n_rounds: SerializedEdge,
        data: SerializedEdge,
    },
    ByteReverse(SerializedEdge),
    Truncate128(SerializedEdge),
    MulTwoPow192(SerializedEdge),
}

impl From<&Node> for SerializedNode {
    fn from(node: &Node) -> Self {
        match node {
            Node::Var { index, name } => SerializedNode::Var {
                index: *index,
                name: name.clone(),
            },
            Node::Neg(e) => SerializedNode::Neg((*e).into()),
            Node::Inv(e) => SerializedNode::Inv((*e).into()),
            Node::Add(l, r) => SerializedNode::Add((*l).into(), (*r).into()),
            Node::Sub(l, r) => SerializedNode::Sub((*l).into(), (*r).into()),
            Node::Mul(l, r) => SerializedNode::Mul((*l).into(), (*r).into()),
            Node::Div(l, r) => SerializedNode::Div((*l).into(), (*r).into()),
            Node::Challenge { id } => SerializedNode::Challenge { id: *id },
            Node::Poseidon {
                state,
                n_rounds,
                data,
            } => SerializedNode::Poseidon {
                state: (*state).into(),
                n_rounds: (*n_rounds).into(),
                data: (*data).into(),
            },
            Node::ByteReverse(e) => SerializedNode::ByteReverse((*e).into()),
            Node::Truncate128(e) => SerializedNode::Truncate128((*e).into()),
            Node::MulTwoPow192(e) => SerializedNode::MulTwoPow192((*e).into()),
        }
    }
}

/// Captured constraint graph from a symbolic execution session.
///
/// Contains the complete node graph, named inputs, and assertions. This is
/// the backend-agnostic intermediate form that can be emitted to any target
/// via [`emit`](AstBundle::emit).
///
/// Can be serialized to JSON for external tooling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstBundle {
    /// All arena nodes in allocation order.
    pub nodes: Vec<SerializedNode>,
    /// Named input variables.
    pub inputs: Vec<InputVar>,
    /// Assertions (constraints).
    pub assertions: Vec<Assertion>,
}

/// Builder for constructing an `AstBundle` from the current arena state.
pub struct VarAllocator {
    inputs: Vec<InputVar>,
    assertions: Vec<Assertion>,
    next_index: u32,
}

impl VarAllocator {
    pub fn new() -> Self {
        Self {
            inputs: Vec::new(),
            assertions: Vec::new(),
            next_index: 0,
        }
    }

    /// Register a named input variable with default metadata (ProofData, native field).
    pub fn input(&mut self, name: impl Into<String>) -> u32 {
        self.input_with_metadata(name, WitnessKind::default(), None)
    }

    /// Register a named input variable with explicit metadata.
    ///
    /// # Arguments
    ///
    /// * `name` — Human-readable name for codegen.
    /// * `witness_kind` — Whether this is proof data or a public statement.
    /// * `field_tag` — `None` for native field, `Some("fq")` for base field, etc.
    pub fn input_with_metadata(
        &mut self,
        name: impl Into<String>,
        witness_kind: WitnessKind,
        field_tag: Option<String>,
    ) -> u32 {
        let index = self.next_index;
        self.inputs.push(InputVar {
            index,
            name: name.into(),
            witness_kind,
            field_tag,
        });
        self.next_index += 1;
        index
    }

    /// Add an assertion that `edge == 0`.
    pub fn assert_zero(&mut self, edge: Edge) {
        self.assertions.push(Assertion::Zero(edge.into()));
    }

    /// Add an assertion that `lhs == rhs`.
    pub fn assert_equal(&mut self, lhs: Edge, rhs: Edge) {
        self.assertions
            .push(Assertion::Equal(lhs.into(), rhs.into()));
    }

    /// Finalize: snapshot the arena and produce an `AstBundle`.
    pub fn finish(self) -> AstBundle {
        let arena_nodes = arena::snapshot();
        let nodes = arena_nodes.iter().map(SerializedNode::from).collect();
        AstBundle {
            nodes,
            inputs: self.inputs,
            assertions: self.assertions,
        }
    }
}

impl Default for VarAllocator {
    fn default() -> Self {
        Self::new()
    }
}

impl AstBundle {
    /// Emit this bundle through an [`AstEmitter`] backend.
    ///
    /// Walks the node graph in allocation order, emitting each node exactly
    /// once. Assertions are emitted at the end.
    pub fn emit<E: AstEmitter>(&self, emitter: &mut E)
    where
        E::Wire: Clone,
    {
        let mut wires: Vec<Option<E::Wire>> = vec![None; self.nodes.len()];

        // Helper to resolve an edge to a wire
        let resolve =
            |edge: &SerializedEdge, wires: &[Option<E::Wire>], emitter: &mut E| -> E::Wire {
                match edge {
                    SerializedEdge::Scalar(val) => emitter.constant(*val),
                    SerializedEdge::Node(id) => wires[*id as usize]
                        .as_ref()
                        .expect("forward reference in arena — nodes must be in topological order")
                        .clone(),
                }
            };

        // Emit all nodes in order
        for (i, node) in self.nodes.iter().enumerate() {
            let wire = match node {
                SerializedNode::Var { index, name } => emitter.variable(*index, name),
                SerializedNode::Neg(inner) => {
                    let w = resolve(inner, &wires, emitter);
                    emitter.neg(w)
                }
                SerializedNode::Inv(inner) => {
                    let w = resolve(inner, &wires, emitter);
                    emitter.inv(w)
                }
                SerializedNode::Add(l, r) => {
                    let wl = resolve(l, &wires, emitter);
                    let wr = resolve(r, &wires, emitter);
                    emitter.add(wl, wr)
                }
                SerializedNode::Sub(l, r) => {
                    let wl = resolve(l, &wires, emitter);
                    let wr = resolve(r, &wires, emitter);
                    emitter.sub(wl, wr)
                }
                SerializedNode::Mul(l, r) => {
                    let wl = resolve(l, &wires, emitter);
                    let wr = resolve(r, &wires, emitter);
                    emitter.mul(wl, wr)
                }
                SerializedNode::Div(l, r) => {
                    let wl = resolve(l, &wires, emitter);
                    let wr = resolve(r, &wires, emitter);
                    emitter.div(wl, wr)
                }
                SerializedNode::Challenge { .. } => {
                    // Challenges are treated as variables with a special name
                    emitter.variable(i as u32, &format!("challenge_{i}"))
                }
                SerializedNode::Poseidon {
                    state,
                    n_rounds,
                    data,
                } => {
                    let ws = resolve(state, &wires, emitter);
                    let wn = resolve(n_rounds, &wires, emitter);
                    let wd = resolve(data, &wires, emitter);
                    emitter.poseidon(ws, wn, wd)
                }
                SerializedNode::ByteReverse(inner) => {
                    let w = resolve(inner, &wires, emitter);
                    emitter.byte_reverse(w)
                }
                SerializedNode::Truncate128(inner) => {
                    let w = resolve(inner, &wires, emitter);
                    emitter.truncate_128(w)
                }
                SerializedNode::MulTwoPow192(inner) => {
                    let w = resolve(inner, &wires, emitter);
                    emitter.mul_two_pow_192(w)
                }
            };
            wires[i] = Some(wire);
        }

        // Emit assertions
        for assertion in &self.assertions {
            match assertion {
                Assertion::Zero(edge) => {
                    let w = resolve(edge, &wires, emitter);
                    emitter.assert_zero(w);
                }
                Assertion::Equal(l, r) => {
                    let wl = resolve(l, &wires, emitter);
                    let wr = resolve(r, &wires, emitter);
                    emitter.assert_equal(wl, wr);
                }
            }
        }
    }

    /// Serialize to JSON.
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON.
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::ArenaSession;
    use crate::symbolic::SymbolicField;
    use jolt_field::Field;

    /// Mock emitter that records operations as strings for testing.
    struct MockEmitter {
        ops: Vec<String>,
        asserts: Vec<String>,
        next_id: usize,
    }

    impl MockEmitter {
        fn new() -> Self {
            Self {
                ops: Vec::new(),
                asserts: Vec::new(),
                next_id: 0,
            }
        }

        fn alloc(&mut self) -> String {
            let id = format!("w{}", self.next_id);
            self.next_id += 1;
            id
        }
    }

    impl AstEmitter for MockEmitter {
        type Wire = String;

        fn constant(&mut self, val: [u64; 4]) -> String {
            let w = self.alloc();
            self.ops.push(format!("{w} = const({:?})", val[0]));
            w
        }

        fn variable(&mut self, index: u32, name: &str) -> String {
            let w = self.alloc();
            self.ops.push(format!("{w} = var({index}, {name})"));
            w
        }

        fn neg(&mut self, inner: String) -> String {
            let w = self.alloc();
            self.ops.push(format!("{w} = neg({inner})"));
            w
        }

        fn inv(&mut self, inner: String) -> String {
            let w = self.alloc();
            self.ops.push(format!("{w} = inv({inner})"));
            w
        }

        fn add(&mut self, lhs: String, rhs: String) -> String {
            let w = self.alloc();
            self.ops.push(format!("{w} = add({lhs}, {rhs})"));
            w
        }

        fn sub(&mut self, lhs: String, rhs: String) -> String {
            let w = self.alloc();
            self.ops.push(format!("{w} = sub({lhs}, {rhs})"));
            w
        }

        fn mul(&mut self, lhs: String, rhs: String) -> String {
            let w = self.alloc();
            self.ops.push(format!("{w} = mul({lhs}, {rhs})"));
            w
        }

        fn div(&mut self, lhs: String, rhs: String) -> String {
            let w = self.alloc();
            self.ops.push(format!("{w} = div({lhs}, {rhs})"));
            w
        }

        fn poseidon(&mut self, state: String, n_rounds: String, data: String) -> String {
            let w = self.alloc();
            self.ops
                .push(format!("{w} = poseidon({state}, {n_rounds}, {data})"));
            w
        }

        fn byte_reverse(&mut self, inner: String) -> String {
            let w = self.alloc();
            self.ops.push(format!("{w} = byte_reverse({inner})"));
            w
        }

        fn truncate_128(&mut self, inner: String) -> String {
            let w = self.alloc();
            self.ops.push(format!("{w} = truncate_128({inner})"));
            w
        }

        fn mul_two_pow_192(&mut self, inner: String) -> String {
            let w = self.alloc();
            self.ops.push(format!("{w} = mul_two_pow_192({inner})"));
            w
        }

        fn assert_zero(&mut self, expr: String) {
            self.asserts.push(format!("assert_zero({expr})"));
        }

        fn assert_equal(&mut self, lhs: String, rhs: String) {
            self.asserts.push(format!("assert_eq({lhs}, {rhs})"));
        }
    }

    #[test]
    fn bundle_from_symbolic_execution() {
        let _session = ArenaSession::new();

        let x = SymbolicField::variable(0, "x");
        let y = SymbolicField::variable(1, "y");
        let result = x * y + SymbolicField::from_u64(1);

        let mut allocator = VarAllocator::new();
        let _ = allocator.input("x");
        let _ = allocator.input("y");
        allocator.assert_zero(result.into_edge());

        let bundle = allocator.finish();

        assert_eq!(bundle.inputs.len(), 2);
        assert_eq!(bundle.assertions.len(), 1);
        // 2 vars + 1 mul + 1 add = 4 nodes (constant 1 is folded as Scalar edge)
        assert!(bundle.nodes.len() >= 3);
    }

    #[test]
    fn bundle_emit_to_mock() {
        let _session = ArenaSession::new();

        let x = SymbolicField::variable(0, "x");
        let y = SymbolicField::variable(1, "y");
        let sum = x + y;

        let mut allocator = VarAllocator::new();
        let _ = allocator.input("x");
        let _ = allocator.input("y");
        allocator.assert_zero(sum.into_edge());

        let bundle = allocator.finish();

        let mut emitter = MockEmitter::new();
        bundle.emit(&mut emitter);

        // Should have var, var, add operations
        assert!(emitter.ops.iter().any(|op| op.contains("var(0, x)")));
        assert!(emitter.ops.iter().any(|op| op.contains("var(1, y)")));
        assert!(emitter.ops.iter().any(|op| op.contains("add(")));
        assert_eq!(emitter.asserts.len(), 1);
        assert!(emitter.asserts[0].starts_with("assert_zero("));
    }

    #[test]
    fn bundle_json_roundtrip() {
        let _session = ArenaSession::new();

        let x = SymbolicField::variable(0, "x");
        let neg_x = -x;

        let mut allocator = VarAllocator::new();
        let _ = allocator.input("x");
        allocator.assert_zero(neg_x.into_edge());

        let bundle = allocator.finish();
        let json = bundle.to_json().unwrap();
        let restored = AstBundle::from_json(&json).unwrap();

        assert_eq!(restored.nodes.len(), bundle.nodes.len());
        assert_eq!(restored.inputs.len(), bundle.inputs.len());
        assert_eq!(restored.assertions.len(), bundle.assertions.len());
    }

    #[test]
    fn var_allocator_indexing() {
        let mut allocator = VarAllocator::new();
        assert_eq!(allocator.input("a"), 0);
        assert_eq!(allocator.input("b"), 1);
        assert_eq!(allocator.input("c"), 2);
    }
}
