use std::fmt::{self, Write as _};

use std::collections::HashMap;

use crate::correspondence::{CanonicalizedGraph, Canonicalizer};
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
    fn write_node_for_lean(
        output: &mut String,
        node: &LookupGraphNode,
        indent: &str,
    ) -> fmt::Result {
        match node {
            LookupGraphNode::Constant(value) => writeln!(
                output,
                "{indent}.constant {},",
                scalar_to_decimal_string(value)
            ),
            LookupGraphNode::Input(index) => writeln!(output, "{indent}.input {index},"),
            LookupGraphNode::Add(left, right) => {
                writeln!(output, "{indent}.add {left} {right},")
            }
            LookupGraphNode::Sub(left, right) => {
                writeln!(output, "{indent}.sub {left} {right},")
            }
            LookupGraphNode::Mul(left, right) => {
                writeln!(output, "{indent}.mul {left} {right},")
            }
        }
    }

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

    /// Return the root node's index in the emitted topological order.
    pub fn root_index(&self) -> usize {
        self.root
    }

    /// Hash-cons this graph into a shared associative-commutative certificate DAG.
    pub fn canonicalize(&self, canonicalizer: &mut Canonicalizer) -> CanonicalizedGraph {
        let mut ids = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            let id = match node {
                LookupGraphNode::Constant(value) => canonicalizer.constant(*value),
                LookupGraphNode::Input(index) => canonicalizer.input(*index),
                LookupGraphNode::Add(left, right) => canonicalizer.add(ids[*left], ids[*right]),
                LookupGraphNode::Sub(left, right) => canonicalizer.sub(ids[*left], ids[*right]),
                LookupGraphNode::Mul(left, right) => canonicalizer.mul(ids[*left], ids[*right]),
            };
            ids.push(id);
        }
        CanonicalizedGraph {
            root_id: ids[self.root],
            ids,
        }
    }

    /// Compute compact support summaries for every graph node.
    pub fn multilinearity_summaries(&self, arity: usize) -> Result<Vec<(u128, bool)>, String> {
        if arity > u128::BITS as usize {
            return Err(format!(
                "multilinearity certificates support at most {} inputs, got {arity}",
                u128::BITS
            ));
        }

        let mut summaries: Vec<(u128, bool)> = Vec::with_capacity(self.nodes.len());
        for chunk in self.nodes.chunks(LEAN_NODE_CHUNK_SIZE) {
            for node in chunk {
                let summary = match node {
                    LookupGraphNode::Constant(_) => (0, true),
                    LookupGraphNode::Input(index) => {
                        if *index >= arity {
                            return Err(format!(
                                "lookup input {index} is outside certificate arity {arity}"
                            ));
                        }
                        (1_u128 << *index, true)
                    }
                    LookupGraphNode::Add(left, right) | LookupGraphNode::Sub(left, right) => {
                        let left = summaries
                            .get(*left)
                            .ok_or_else(|| format!("invalid certificate reference {left}"))?;
                        let right = summaries
                            .get(*right)
                            .ok_or_else(|| format!("invalid certificate reference {right}"))?;
                        (left.0 | right.0, left.1 && right.1)
                    }
                    LookupGraphNode::Mul(left, right) => {
                        let left = summaries
                            .get(*left)
                            .ok_or_else(|| format!("invalid certificate reference {left}"))?;
                        let right = summaries
                            .get(*right)
                            .ok_or_else(|| format!("invalid certificate reference {right}"))?;
                        (left.0 | right.0, left.1 && right.1 && left.0 & right.0 == 0)
                    }
                };
                summaries.push(summary);
            }
        }
        Ok(summaries)
    }

    /// Format each emitted node chunk independently for certificate obligations.
    pub fn chunks_for_lean(&self) -> Result<Vec<(usize, String)>, fmt::Error> {
        self.nodes
            .chunks(LEAN_NODE_CHUNK_SIZE)
            .map(|chunk| {
                let mut output = String::from("[");
                if !chunk.is_empty() {
                    output.push('\n');
                }
                for node in chunk {
                    Self::write_node_for_lean(&mut output, node, "    ")?;
                }
                output.push(']');
                Ok((chunk.len(), output))
            })
            .collect()
    }

    #[cfg(test)]
    pub fn structurally_equivalent(&self, other: &Self) -> bool {
        let mut canonicalizer = Canonicalizer::default();
        self.canonicalize(&mut canonicalizer).root_id
            == other.canonicalize(&mut canonicalizer).root_id
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
                Self::write_node_for_lean(&mut output, node, "        ")?;
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

    #[test]
    fn multilinearity_summaries_detect_reused_support() {
        let left = MleAst::from_var(0);
        let right = MleAst::from_var(1);
        let shared = left + right;
        let graph = LookupGraph::from_mle_ast(&(shared * shared), 2).unwrap();

        assert_eq!(
            graph.multilinearity_summaries(2).unwrap().last(),
            Some(&(3, false))
        );
    }

    #[test]
    fn multilinearity_summaries_accept_disjoint_support() {
        let expression = MleAst::from_var(0) * MleAst::from_var(1);
        let graph = LookupGraph::from_mle_ast(&expression, 2).unwrap();

        assert_eq!(
            graph.multilinearity_summaries(2).unwrap().last(),
            Some(&(3, true))
        );
        assert!(graph.multilinearity_summaries(129).is_err());
    }
}
