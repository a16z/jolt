use std::collections::HashMap;
use std::fmt::{self, Write as _};

use crate::mle_ast::{
    get_node, scalar_to_decimal_string, Atom, Edge, MleAst, Node, NodeId, Scalar,
};

#[cfg(test)]
use jolt_prover_legacy::field::JoltField;

/// One operation in a shared algebraic lookup graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LookupGraphNode {
    Constant(Scalar),
    Input(usize),
    Add(usize, usize),
    Sub(usize, usize),
    Mul(usize, usize),
}

/// A compact, topologically ordered view of the `MleAst` reachable from one root.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LookupGraph {
    nodes: Vec<LookupGraphNode>,
    root: usize,
}

const LEAN_NODE_CHUNK_SIZE: usize = 32;

impl LookupGraph {
    /// Preserve arena sharing while converting the lookup subset of `MleAst`.
    pub fn from_mle_ast(ast: &MleAst, num_inputs: usize) -> Result<Self, String> {
        struct Builder {
            num_inputs: usize,
            nodes: Vec<LookupGraphNode>,
            node_indices: HashMap<NodeId, usize>,
            atom_indices: HashMap<Atom, usize>,
        }

        impl Builder {
            fn push(&mut self, node: LookupGraphNode) -> usize {
                let index = self.nodes.len();
                self.nodes.push(node);
                index
            }

            fn visit_atom(&mut self, atom: Atom) -> Result<usize, String> {
                if let Some(index) = self.atom_indices.get(&atom) {
                    return Ok(*index);
                }

                let node = match atom {
                    Atom::Scalar(value) => LookupGraphNode::Constant(value),
                    Atom::Var(index) => {
                        let index = usize::from(index);
                        if index >= self.num_inputs {
                            return Err(format!(
                                "lookup variable {index} is outside graph arity {}",
                                self.num_inputs
                            ));
                        }
                        LookupGraphNode::Input(index)
                    }
                    Atom::NamedVar(index) => {
                        return Err(format!(
                            "CSE variable {index} cannot occur in the source lookup graph"
                        ));
                    }
                };

                let index = self.push(node);
                self.atom_indices.insert(atom, index);
                Ok(index)
            }

            fn visit_edge(&mut self, edge: Edge) -> Result<usize, String> {
                match edge {
                    Edge::Atom(atom) => self.visit_atom(atom),
                    Edge::NodeRef(node) => self.visit_node(node),
                }
            }

            fn visit_node(&mut self, node_id: NodeId) -> Result<usize, String> {
                if let Some(index) = self.node_indices.get(&node_id) {
                    return Ok(*index);
                }

                let node = match get_node(node_id) {
                    Node::Atom(atom) => {
                        let index = self.visit_atom(atom)?;
                        self.node_indices.insert(node_id, index);
                        return Ok(index);
                    }
                    Node::Add(left, right) => {
                        LookupGraphNode::Add(self.visit_edge(left)?, self.visit_edge(right)?)
                    }
                    Node::Sub(left, right) => {
                        LookupGraphNode::Sub(self.visit_edge(left)?, self.visit_edge(right)?)
                    }
                    Node::Mul(left, right) => {
                        LookupGraphNode::Mul(self.visit_edge(left)?, self.visit_edge(right)?)
                    }
                    unsupported => {
                        return Err(format!(
                            "unsupported node in algebraic lookup graph: {unsupported:?}"
                        ));
                    }
                };

                let index = self.push(node);
                self.node_indices.insert(node_id, index);
                Ok(index)
            }
        }

        let mut builder = Builder {
            num_inputs,
            nodes: Vec::new(),
            node_indices: HashMap::new(),
            atom_indices: HashMap::new(),
        };
        let root = builder.visit_node(ast.root())?;

        Ok(Self {
            nodes: builder.nodes,
            root,
        })
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    #[cfg(test)]
    pub fn structurally_equivalent(&self, other: &Self) -> bool {
        #[derive(Clone, Debug, PartialEq, Eq, Hash)]
        enum Key {
            Constant(Scalar),
            Input(usize),
            Add(Vec<usize>),
            Sub(usize, usize),
            Mul(Vec<usize>),
        }

        fn intern_graph(
            graph: &LookupGraph,
            interner: &mut HashMap<Key, usize>,
            keys: &mut Vec<Key>,
        ) -> usize {
            let mut ids: Vec<usize> = Vec::with_capacity(graph.nodes.len());
            for node in &graph.nodes {
                let key = match node {
                    LookupGraphNode::Constant(value) => Key::Constant(*value),
                    LookupGraphNode::Input(index) => Key::Input(*index),
                    LookupGraphNode::Add(left, right) => {
                        let mut terms = Vec::new();
                        for id in [ids[*left], ids[*right]] {
                            match &keys[id] {
                                Key::Add(children) => terms.extend(children),
                                _ => terms.push(id),
                            }
                        }
                        terms.retain(|id| {
                            !matches!(&keys[*id], Key::Constant(value)
                                if value.iter().all(|limb| *limb == 0))
                        });
                        terms.sort_unstable();
                        match terms.as_slice() {
                            [] => Key::Constant([0; 4]),
                            [id] => keys[*id].clone(),
                            _ => Key::Add(terms),
                        }
                    }
                    LookupGraphNode::Sub(left, right) => Key::Sub(ids[*left], ids[*right]),
                    LookupGraphNode::Mul(left, right) => {
                        let mut factors = Vec::new();
                        for id in [ids[*left], ids[*right]] {
                            match &keys[id] {
                                Key::Mul(children) => factors.extend(children),
                                _ => factors.push(id),
                            }
                        }
                        if factors.iter().any(|id| {
                            matches!(&keys[*id], Key::Constant(value)
                                if value.iter().all(|limb| *limb == 0))
                        }) {
                            Key::Constant([0; 4])
                        } else {
                            factors.retain(|id| {
                                !matches!(&keys[*id], Key::Constant(value)
                                    if value[0] == 1 && value[1..].iter().all(|limb| *limb == 0))
                            });
                            factors.sort_unstable();
                            match factors.as_slice() {
                                [] => Key::Constant([1, 0, 0, 0]),
                                [id] => keys[*id].clone(),
                                _ => Key::Mul(factors),
                            }
                        }
                    }
                };
                let id = if let Some(id) = interner.get(&key) {
                    *id
                } else {
                    let id = keys.len();
                    interner.insert(key.clone(), id);
                    keys.push(key);
                    id
                };
                ids.push(id);
            }
            ids[graph.root]
        }

        let mut interner = HashMap::new();
        let mut keys = Vec::new();
        intern_graph(self, &mut interner, &mut keys)
            == intern_graph(other, &mut interner, &mut keys)
    }

    /// Evaluate the serialized graph independently of `MleAst` in tests.
    #[cfg(test)]
    pub fn evaluate<F: JoltField>(&self, point: &[F]) -> F {
        let radix = F::from_u128(1_u128 << 64);
        let mut values = Vec::with_capacity(self.nodes.len());

        for node in &self.nodes {
            let value = match node {
                LookupGraphNode::Constant(limbs) => limbs
                    .iter()
                    .rev()
                    .fold(F::zero(), |value, limb| value * radix + F::from_u64(*limb)),
                LookupGraphNode::Input(index) => point[*index],
                LookupGraphNode::Add(left, right) => values[*left] + values[*right],
                LookupGraphNode::Sub(left, right) => values[*left] - values[*right],
                LookupGraphNode::Mul(left, right) => values[*left] * values[*right],
            };
            values.push(value);
        }

        values[self.root]
    }

    /// Format the graph as data for the static Lean interpreter.
    pub fn format_for_lean(&self) -> Result<String, fmt::Error> {
        let mut output = String::new();
        writeln!(output, "{{ nodeChunks := [")?;
        for chunk in self.nodes.chunks(LEAN_NODE_CHUNK_SIZE) {
            writeln!(output, "      [")?;
            for node in chunk {
                match node {
                    LookupGraphNode::Constant(value) => {
                        writeln!(
                            output,
                            "        .constant {},",
                            scalar_to_decimal_string(value)
                        )?;
                    }
                    LookupGraphNode::Input(index) => {
                        writeln!(output, "        .input {index},")?;
                    }
                    LookupGraphNode::Add(left, right) => {
                        writeln!(output, "        .add {left} {right},")?;
                    }
                    LookupGraphNode::Sub(left, right) => {
                        writeln!(output, "        .sub {left} {right},")?;
                    }
                    LookupGraphNode::Mul(left, right) => {
                        writeln!(output, "        .mul {left} {right},")?;
                    }
                }
            }
            writeln!(output, "      ],")?;
        }
        write!(output, "    ]\n    root := {} }}", self.root)?;
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preserves_shared_subexpressions() {
        let left = MleAst::from_var(0);
        let right = MleAst::from_var(1);
        let shared = left + right;
        let expression = shared * shared;

        let graph = LookupGraph::from_mle_ast(&expression, 2).unwrap();

        assert_eq!(graph.len(), 4);
        assert!(matches!(graph.nodes[2], LookupGraphNode::Add(0, 1)));
        assert!(matches!(graph.nodes[3], LookupGraphNode::Mul(2, 2)));
    }

    #[test]
    fn rejects_variables_outside_the_declared_arity() {
        let expression = MleAst::from_var(2);
        let error = LookupGraph::from_mle_ast(&expression, 2).unwrap_err();

        assert_eq!(error, "lookup variable 2 is outside graph arity 2");
    }
}
