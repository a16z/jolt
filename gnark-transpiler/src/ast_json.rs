//! JSON serialization for MLE AST
//!
//! Exports the AST to a JSON format that can be loaded by other tools.
//!
//! Note: Functions that depend on `stage1_only_verifier` are gated behind the
//! `verifier-transpilation` feature, as they require code from the verifier-transpilation branch.

use serde::{Deserialize, Serialize};
use zklean_extractor::mle_ast::{get_node, Atom, Edge, Node};

/// JSON-serializable representation of an Atom
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum AtomJson {
    /// Scalar value as decimal string (to handle values > 128 bits)
    Scalar { value: String },
    Var { index: u16 },
    NamedVar { index: usize },
}

/// Convert [u64; 4] limbs to decimal string
fn limbs_to_decimal(limbs: [u64; 4]) -> String {
    use num_bigint::BigUint;

    if limbs == [0, 0, 0, 0] {
        return "0".to_string();
    }

    let mut value = BigUint::from(limbs[3]);
    value = (value << 64) + limbs[2];
    value = (value << 64) + limbs[1];
    value = (value << 64) + limbs[0];

    value.to_string()
}

impl From<Atom> for AtomJson {
    fn from(atom: Atom) -> Self {
        match atom {
            Atom::Scalar(value) => AtomJson::Scalar { value: limbs_to_decimal(value) },
            Atom::Var(index) => AtomJson::Var { index },
            Atom::NamedVar(index) => AtomJson::NamedVar { index },
        }
    }
}

/// JSON-serializable representation of an Edge
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum EdgeJson {
    Atom { atom: AtomJson },
    NodeRef { node_id: usize },
}

impl From<Edge> for EdgeJson {
    fn from(edge: Edge) -> Self {
        match edge {
            Edge::Atom(atom) => EdgeJson::Atom { atom: atom.into() },
            Edge::NodeRef(node_id) => EdgeJson::NodeRef { node_id },
        }
    }
}

/// JSON-serializable representation of a Node
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "op")]
pub enum NodeJson {
    Atom {
        atom: AtomJson,
    },
    Neg {
        child: EdgeJson,
    },
    Inv {
        child: EdgeJson,
    },
    Add {
        left: EdgeJson,
        right: EdgeJson,
    },
    Mul {
        left: EdgeJson,
        right: EdgeJson,
    },
    Sub {
        left: EdgeJson,
        right: EdgeJson,
    },
    Div {
        left: EdgeJson,
        right: EdgeJson,
    },
    /// Poseidon hash with 3 inputs: (state, n_rounds, data)
    Poseidon {
        state: EdgeJson,
        n_rounds: EdgeJson,
        data: EdgeJson,
    },
    Keccak256 {
        input: EdgeJson,
    },
    ByteReverse {
        input: EdgeJson,
    },
    Truncate128Reverse {
        input: EdgeJson,
    },
    Truncate128 {
        input: EdgeJson,
    },
    MulTwoPow192 {
        input: EdgeJson,
    },
}

impl From<Node> for NodeJson {
    fn from(node: Node) -> Self {
        match node {
            Node::Atom(atom) => NodeJson::Atom { atom: atom.into() },
            Node::Neg(child) => NodeJson::Neg {
                child: child.into(),
            },
            Node::Inv(child) => NodeJson::Inv {
                child: child.into(),
            },
            Node::Add(left, right) => NodeJson::Add {
                left: left.into(),
                right: right.into(),
            },
            Node::Mul(left, right) => NodeJson::Mul {
                left: left.into(),
                right: right.into(),
            },
            Node::Sub(left, right) => NodeJson::Sub {
                left: left.into(),
                right: right.into(),
            },
            Node::Div(left, right) => NodeJson::Div {
                left: left.into(),
                right: right.into(),
            },
            Node::Poseidon(state, n_rounds, data) => NodeJson::Poseidon {
                state: state.into(),
                n_rounds: n_rounds.into(),
                data: data.into(),
            },
            Node::Keccak256(input) => NodeJson::Keccak256 {
                input: input.into(),
            },
            Node::ByteReverse(input) => NodeJson::ByteReverse {
                input: input.into(),
            },
            Node::Truncate128Reverse(input) => NodeJson::Truncate128Reverse {
                input: input.into(),
            },
            Node::Truncate128(input) => NodeJson::Truncate128 {
                input: input.into(),
            },
            Node::MulTwoPow192(input) => NodeJson::MulTwoPow192 {
                input: input.into(),
            },
        }
    }
}

/// A constraint in the circuit
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConstraintJson {
    pub name: String,
    pub description: String,
    pub root_node_id: usize,
}

/// Complete AST export for Stage 1 circuit
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stage1AstJson {
    /// All nodes in the arena (indexed by node_id)
    pub nodes: Vec<NodeJson>,
    /// The constraints
    pub constraints: Vec<ConstraintJson>,
    /// Variables used in the circuit
    pub variables: Vec<u16>,
    /// Metadata
    pub trace_length: usize,
    pub num_rounds: usize,
}

/// Collect all nodes reachable from a root node
fn collect_nodes_from_root(root_id: usize, max_id: &mut usize) {
    *max_id = (*max_id).max(root_id);

    let node = get_node(root_id);
    match node {
        Node::Atom(_) => {}
        Node::Neg(edge) | Node::Inv(edge) | Node::ByteReverse(edge) | Node::Truncate128Reverse(edge) | Node::Truncate128(edge) | Node::MulTwoPow192(edge) => {
            if let Edge::NodeRef(id) = edge {
                collect_nodes_from_root(id, max_id);
            }
        }
        Node::Add(left, right)
        | Node::Mul(left, right)
        | Node::Sub(left, right)
        | Node::Div(left, right) => {
            if let Edge::NodeRef(id) = left {
                collect_nodes_from_root(id, max_id);
            }
            if let Edge::NodeRef(id) = right {
                collect_nodes_from_root(id, max_id);
            }
        }
        Node::Poseidon(state, n_rounds, data) => {
            for edge in [state, n_rounds, data] {
                if let Edge::NodeRef(id) = edge {
                    collect_nodes_from_root(id, max_id);
                }
            }
        }
        Node::Keccak256(input) => {
            if let Edge::NodeRef(id) = input {
                collect_nodes_from_root(id, max_id);
            }
        }
    }
}

/// Export Stage1TranscriptVerificationResult to JSON
pub fn export_stage1_ast(
    result: &jolt_core::zkvm::stage1_only_verifier::Stage1TranscriptVerificationResult<
        zklean_extractor::mle_ast::MleAst,
    >,
    trace_length: usize,
) -> Stage1AstJson {
    use std::collections::BTreeSet;

    // Find the maximum node ID we need
    let mut max_id = 0usize;
    collect_nodes_from_root(result.final_claim.root(), &mut max_id);
    collect_nodes_from_root(result.power_sum_check.root(), &mut max_id);
    for check in &result.sumcheck_consistency_checks {
        collect_nodes_from_root(check.root(), &mut max_id);
    }

    // Export all nodes up to max_id
    let nodes: Vec<NodeJson> = (0..=max_id).map(|id| get_node(id).into()).collect();

    // Collect variables
    let mut vars = BTreeSet::new();
    collect_vars_from_node(result.final_claim.root(), &mut vars);
    collect_vars_from_node(result.power_sum_check.root(), &mut vars);
    for check in &result.sumcheck_consistency_checks {
        collect_vars_from_node(check.root(), &mut vars);
    }

    // Build constraints
    let mut constraints = vec![ConstraintJson {
        name: "power_sum_check".to_string(),
        description: "Sum over symmetric domain must equal 0".to_string(),
        root_node_id: result.power_sum_check.root(),
    }];

    for (i, check) in result.sumcheck_consistency_checks.iter().enumerate() {
        constraints.push(ConstraintJson {
            name: format!("consistency_check_{}", i),
            description: format!("Sumcheck round {}: poly(0) + poly(1) - claim == 0", i),
            root_node_id: check.root(),
        });
    }

    constraints.push(ConstraintJson {
        name: "final_claim".to_string(),
        description: "Final claim must match expected value".to_string(),
        root_node_id: result.final_claim.root(),
    });

    let num_rounds = (trace_length as f64).log2() as usize;

    Stage1AstJson {
        nodes,
        constraints,
        variables: vars.into_iter().collect(),
        trace_length,
        num_rounds,
    }
}

fn collect_vars_from_node(node_id: usize, vars: &mut std::collections::BTreeSet<u16>) {
    let node = get_node(node_id);
    match node {
        Node::Atom(Atom::Var(index)) => {
            vars.insert(index);
        }
        Node::Atom(_) => {}
        Node::Neg(edge) | Node::Inv(edge) | Node::ByteReverse(edge) | Node::Truncate128Reverse(edge) | Node::Truncate128(edge) | Node::MulTwoPow192(edge) => {
            collect_vars_from_edge(edge, vars);
        }
        Node::Add(left, right)
        | Node::Mul(left, right)
        | Node::Sub(left, right)
        | Node::Div(left, right) => {
            collect_vars_from_edge(left, vars);
            collect_vars_from_edge(right, vars);
        }
        Node::Poseidon(state, n_rounds, data) => {
            for edge in [state, n_rounds, data] {
                collect_vars_from_edge(edge, vars);
            }
        }
        Node::Keccak256(input) => {
            collect_vars_from_edge(input, vars);
        }
    }
}

fn collect_vars_from_edge(edge: Edge, vars: &mut std::collections::BTreeSet<u16>) {
    match edge {
        Edge::Atom(Atom::Var(index)) => {
            vars.insert(index);
        }
        Edge::Atom(_) => {}
        Edge::NodeRef(node_id) => {
            collect_vars_from_node(node_id, vars);
        }
    }
}

impl Stage1AstJson {
    /// Serialize to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Serialize to compact JSON string
    pub fn to_json_compact(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string(self)
    }

    /// Generate a Mermaid diagram for each constraint
    pub fn to_mermaid(&self) -> String {
        let mut output = String::new();

        for constraint in &self.constraints {
            output.push_str(&format!("## {}\n\n", constraint.name));
            output.push_str(&format!("{}\n\n", constraint.description));
            output.push_str("```mermaid\ngraph TD\n");

            let mut visited = std::collections::HashSet::new();
            self.render_node_mermaid(constraint.root_node_id, &mut output, &mut visited);

            output.push_str("```\n\n");
        }

        output
    }

    fn render_node_mermaid(
        &self,
        node_id: usize,
        output: &mut String,
        visited: &mut std::collections::HashSet<usize>,
    ) {
        if visited.contains(&node_id) {
            return;
        }
        visited.insert(node_id);

        let node = &self.nodes[node_id];
        let node_label = self.node_to_label(node);

        output.push_str(&format!("    N{}[\"{}\"]\n", node_id, node_label));

        match node {
            NodeJson::Atom { .. } => {}
            NodeJson::Neg { child }
            | NodeJson::Inv { child }
            | NodeJson::ByteReverse { input: child }
            | NodeJson::Truncate128Reverse { input: child }
            | NodeJson::Truncate128 { input: child }
            | NodeJson::MulTwoPow192 { input: child } => {
                if let Some(child_id) = self.edge_to_id(child) {
                    output.push_str(&format!("    N{} --> N{}\n", node_id, child_id));
                    self.render_node_mermaid(child_id, output, visited);
                } else {
                    let leaf_id = format!("L{}_{}", node_id, 0);
                    output.push_str(&format!(
                        "    {}[\"{}\"]\n",
                        leaf_id,
                        self.edge_to_label(child)
                    ));
                    output.push_str(&format!("    N{} --> {}\n", node_id, leaf_id));
                }
            }
            NodeJson::Add { left, right }
            | NodeJson::Mul { left, right }
            | NodeJson::Sub { left, right }
            | NodeJson::Div { left, right } => {
                for (i, edge) in [left, right].iter().enumerate() {
                    if let Some(child_id) = self.edge_to_id(edge) {
                        output.push_str(&format!("    N{} --> N{}\n", node_id, child_id));
                        self.render_node_mermaid(child_id, output, visited);
                    } else {
                        let leaf_id = format!("L{}_{}", node_id, i);
                        output.push_str(&format!(
                            "    {}[\"{}\"]\n",
                            leaf_id,
                            self.edge_to_label(edge)
                        ));
                        output.push_str(&format!("    N{} --> {}\n", node_id, leaf_id));
                    }
                }
            }
            NodeJson::Poseidon {
                state,
                n_rounds,
                data,
            } => {
                for (i, edge) in [state, n_rounds, data].iter().enumerate() {
                    if let Some(child_id) = self.edge_to_id(edge) {
                        output.push_str(&format!("    N{} --> N{}\n", node_id, child_id));
                        self.render_node_mermaid(child_id, output, visited);
                    } else {
                        let leaf_id = format!("L{}_{}", node_id, i);
                        output.push_str(&format!(
                            "    {}[\"{}\"]\n",
                            leaf_id,
                            self.edge_to_label(edge)
                        ));
                        output.push_str(&format!("    N{} --> {}\n", node_id, leaf_id));
                    }
                }
            }
            NodeJson::Keccak256 { input } => {
                if let Some(child_id) = self.edge_to_id(input) {
                    output.push_str(&format!("    N{} --> N{}\n", node_id, child_id));
                    self.render_node_mermaid(child_id, output, visited);
                } else {
                    let leaf_id = format!("L{}_0", node_id);
                    output.push_str(&format!(
                        "    {}[\"{}\"]\n",
                        leaf_id,
                        self.edge_to_label(input)
                    ));
                    output.push_str(&format!("    N{} --> {}\n", node_id, leaf_id));
                }
            }
        }
    }

    fn node_to_label(&self, node: &NodeJson) -> String {
        match node {
            NodeJson::Atom { atom } => self.atom_to_label(atom),
            NodeJson::Neg { .. } => "NEG".to_string(),
            NodeJson::Inv { .. } => "INV".to_string(),
            NodeJson::Add { .. } => "+".to_string(),
            NodeJson::Mul { .. } => "×".to_string(),
            NodeJson::Sub { .. } => "-".to_string(),
            NodeJson::Div { .. } => "÷".to_string(),
            NodeJson::Poseidon { .. } => "POSEIDON".to_string(),
            NodeJson::Keccak256 { .. } => "KECCAK256".to_string(),
            NodeJson::ByteReverse { .. } => "BYTE_REV".to_string(),
            NodeJson::Truncate128Reverse { .. } => "TRUNC128_REV".to_string(),
            NodeJson::Truncate128 { .. } => "TRUNC128".to_string(),
            NodeJson::MulTwoPow192 { .. } => "MUL_2^192".to_string(),
        }
    }

    fn atom_to_label(&self, atom: &AtomJson) -> String {
        match atom {
            AtomJson::Scalar { value } => format!("{}", value),
            AtomJson::Var { index } => format!("X_{}", index),
            AtomJson::NamedVar { index } => format!("t_{}", index),
        }
    }

    fn edge_to_label(&self, edge: &EdgeJson) -> String {
        match edge {
            EdgeJson::Atom { atom } => self.atom_to_label(atom),
            EdgeJson::NodeRef { node_id } => format!("N{}", node_id),
        }
    }

    fn edge_to_id(&self, edge: &EdgeJson) -> Option<usize> {
        match edge {
            EdgeJson::Atom { .. } => None,
            EdgeJson::NodeRef { node_id } => Some(*node_id),
        }
    }
}

/// Alias for export_stage1_ast (kept for backwards compatibility)
pub fn export_stage1_poseidon_ast(
    result: &jolt_core::zkvm::stage1_only_verifier::Stage1TranscriptVerificationResult<
        zklean_extractor::mle_ast::MleAst,
    >,
    trace_length: usize,
) -> Stage1AstJson {
    export_stage1_ast(result, trace_length)
}
