use std::fmt::{self, Write as _};

use jolt_lookup_tables::MaterializerBackend;

use crate::correspondence::{CanonicalizedMaterializer, Canonicalizer};

const LEAN_NODE_CHUNK_SIZE: usize = 32;

/// A reference to one Boolean node in a materializer graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BoolRef(usize);

/// A reference to one natural-number node in a materializer graph.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct NatRef(usize);

/// One Boolean operation in a shared materializer graph.
#[derive(Clone, Debug, PartialEq, Eq)]
enum BoolNode {
    Input(usize),
    And(BoolRef, BoolRef),
    Not(BoolRef),
}

/// One natural-number operation in a shared materializer graph.
#[derive(Clone, Debug, PartialEq, Eq)]
enum NatNode {
    Constant(u128),
    OfBit(BoolRef),
    Add(NatRef, NatRef),
    Mul(NatRef, NatRef),
}

/// A compact, topologically ordered materializer extracted from shared Rust semantics.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MaterializerGraph {
    bool_nodes: Vec<BoolNode>,
    nat_nodes: Vec<NatNode>,
    root: NatRef,
}

impl MaterializerGraph {
    fn write_bool_node_for_lean(output: &mut String, node: &BoolNode, indent: &str) -> fmt::Result {
        match node {
            BoolNode::Input(index) => writeln!(output, "{indent}.input {index},"),
            BoolNode::And(left, right) => {
                writeln!(output, "{indent}.conj {} {},", left.0, right.0)
            }
            BoolNode::Not(value) => writeln!(output, "{indent}.neg {},", value.0),
        }
    }

    fn write_nat_node_for_lean(output: &mut String, node: &NatNode, indent: &str) -> fmt::Result {
        match node {
            NatNode::Constant(value) => writeln!(output, "{indent}.constant {value},"),
            NatNode::OfBit(value) => writeln!(output, "{indent}.ofBit {},", value.0),
            NatNode::Add(left, right) => {
                writeln!(output, "{indent}.add {} {},", left.0, right.0)
            }
            NatNode::Mul(left, right) => {
                writeln!(output, "{indent}.mul {} {},", left.0, right.0)
            }
        }
    }

    pub(crate) fn canonicalize(
        &self,
        canonicalizer: &mut Canonicalizer,
    ) -> CanonicalizedMaterializer {
        let mut bool_ids = Vec::with_capacity(self.bool_nodes.len());
        for node in &self.bool_nodes {
            let id = match node {
                BoolNode::Input(index) => canonicalizer.input(*index),
                BoolNode::And(left, right) => {
                    canonicalizer.mul(bool_ids[left.0], bool_ids[right.0])
                }
                BoolNode::Not(value) => {
                    let one = canonicalizer.one_id();
                    canonicalizer.sub(one, bool_ids[value.0])
                }
            };
            bool_ids.push(id);
        }

        let mut nat_ids = Vec::with_capacity(self.nat_nodes.len());
        for node in &self.nat_nodes {
            let id = match node {
                NatNode::Constant(value) => canonicalizer.constant_u128(*value),
                NatNode::OfBit(value) => bool_ids[value.0],
                NatNode::Add(left, right) => canonicalizer.add(nat_ids[left.0], nat_ids[right.0]),
                NatNode::Mul(left, right) => canonicalizer.mul(nat_ids[left.0], nat_ids[right.0]),
            };
            nat_ids.push(id);
        }

        CanonicalizedMaterializer {
            root_id: nat_ids[self.root.0],
            bool_ids,
            nat_ids,
        }
    }

    pub(crate) fn bool_chunks_for_lean(&self) -> Result<Vec<(usize, String, bool)>, fmt::Error> {
        self.bool_nodes
            .chunks(LEAN_NODE_CHUNK_SIZE)
            .map(|chunk| {
                let mut output = String::from("[");
                if !chunk.is_empty() {
                    output.push('\n');
                }
                for node in chunk {
                    Self::write_bool_node_for_lean(&mut output, node, "    ")?;
                }
                output.push(']');
                Ok((
                    chunk.len(),
                    output,
                    chunk.iter().any(|node| matches!(node, BoolNode::And(..))),
                ))
            })
            .collect()
    }

    pub(crate) fn nat_chunks_for_lean(&self) -> Result<Vec<(usize, String)>, fmt::Error> {
        self.nat_nodes
            .chunks(LEAN_NODE_CHUNK_SIZE)
            .map(|chunk| {
                let mut output = String::from("[");
                if !chunk.is_empty() {
                    output.push('\n');
                }
                for node in chunk {
                    Self::write_nat_node_for_lean(&mut output, node, "    ")?;
                }
                output.push(']');
                Ok((chunk.len(), output))
            })
            .collect()
    }

    pub fn evaluate(&self, point: &[bool]) -> u128 {
        let mut bool_values: Vec<bool> = Vec::with_capacity(self.bool_nodes.len());
        for node in &self.bool_nodes {
            let value = match node {
                BoolNode::Input(index) => point[*index],
                BoolNode::And(left, right) => bool_values[left.0] && bool_values[right.0],
                BoolNode::Not(value) => !bool_values[value.0],
            };
            bool_values.push(value);
        }

        let mut nat_values: Vec<u128> = Vec::with_capacity(self.nat_nodes.len());
        for node in &self.nat_nodes {
            let value = match node {
                NatNode::Constant(value) => *value,
                NatNode::OfBit(value) => u128::from(bool_values[value.0]),
                NatNode::Add(left, right) => nat_values[left.0] + nat_values[right.0],
                NatNode::Mul(left, right) => nat_values[left.0] * nat_values[right.0],
            };
            nat_values.push(value);
        }
        nat_values[self.root.0]
    }

    pub fn format_for_lean(&self) -> Result<String, fmt::Error> {
        let mut output = String::new();
        writeln!(output, "{{ boolNodeChunks := [")?;
        for chunk in self.bool_nodes.chunks(LEAN_NODE_CHUNK_SIZE) {
            writeln!(output, "      [")?;
            for node in chunk {
                Self::write_bool_node_for_lean(&mut output, node, "        ")?;
            }
            writeln!(output, "      ],")?;
        }

        writeln!(output, "    ]")?;
        writeln!(output, "    natNodeChunks := [")?;
        for chunk in self.nat_nodes.chunks(LEAN_NODE_CHUNK_SIZE) {
            writeln!(output, "      [")?;
            for node in chunk {
                Self::write_nat_node_for_lean(&mut output, node, "        ")?;
            }
            writeln!(output, "      ],")?;
        }
        write!(output, "    ]\n    root := {} }}", self.root.0)?;
        Ok(output)
    }

    #[cfg(test)]
    pub fn to_mle_ast(&self) -> crate::DefaultMleAst {
        let mut bool_values: Vec<crate::DefaultMleAst> = Vec::with_capacity(self.bool_nodes.len());
        for node in &self.bool_nodes {
            let value = match node {
                BoolNode::Input(index) => {
                    crate::DefaultMleAst::from_var((*index).try_into().unwrap())
                }
                BoolNode::And(left, right) => bool_values[left.0] * bool_values[right.0],
                BoolNode::Not(value) => crate::DefaultMleAst::from(1) - bool_values[value.0],
            };
            bool_values.push(value);
        }

        let mut nat_values: Vec<crate::DefaultMleAst> = Vec::with_capacity(self.nat_nodes.len());
        for node in &self.nat_nodes {
            let value = match node {
                NatNode::Constant(value) => crate::DefaultMleAst::from(*value),
                NatNode::OfBit(value) => bool_values[value.0],
                NatNode::Add(left, right) => nat_values[left.0] + nat_values[right.0],
                NatNode::Mul(left, right) => nat_values[left.0] * nat_values[right.0],
            };
            nat_values.push(value);
        }
        nat_values[self.root.0]
    }
}

/// Symbolic backend that records a materializer without expanding shared values.
pub struct MaterializerAst {
    num_inputs: usize,
    bool_nodes: Vec<BoolNode>,
    nat_nodes: Vec<NatNode>,
}

impl MaterializerAst {
    pub fn new(num_inputs: usize) -> Self {
        Self {
            num_inputs,
            bool_nodes: Vec::new(),
            nat_nodes: Vec::new(),
        }
    }

    pub fn finish(self, root: NatRef) -> MaterializerGraph {
        MaterializerGraph {
            bool_nodes: self.bool_nodes,
            nat_nodes: self.nat_nodes,
            root,
        }
    }

    fn push_bool(&mut self, node: BoolNode) -> BoolRef {
        let reference = BoolRef(self.bool_nodes.len());
        self.bool_nodes.push(node);
        reference
    }

    fn push_nat(&mut self, node: NatNode) -> NatRef {
        let reference = NatRef(self.nat_nodes.len());
        self.nat_nodes.push(node);
        reference
    }
}

impl MaterializerBackend for MaterializerAst {
    type Bit = BoolRef;
    type Nat = NatRef;
    type Output = NatRef;

    fn input_bit(&mut self, index: usize) -> Self::Bit {
        assert!(
            index < self.num_inputs,
            "materializer input is out of range"
        );
        self.push_bool(BoolNode::Input(index))
    }

    fn and(&mut self, left: Self::Bit, right: Self::Bit) -> Self::Bit {
        self.push_bool(BoolNode::And(left, right))
    }

    fn not(&mut self, value: Self::Bit) -> Self::Bit {
        self.push_bool(BoolNode::Not(value))
    }

    fn bit_to_nat(&mut self, value: Self::Bit) -> Self::Nat {
        self.push_nat(NatNode::OfBit(value))
    }

    fn nat_constant(&mut self, value: u128) -> Self::Nat {
        self.push_nat(NatNode::Constant(value))
    }

    fn nat_add(&mut self, left: Self::Nat, right: Self::Nat) -> Self::Nat {
        if matches!(self.nat_nodes[left.0], NatNode::Constant(0)) {
            return right;
        }
        if matches!(self.nat_nodes[right.0], NatNode::Constant(0)) {
            return left;
        }
        self.push_nat(NatNode::Add(left, right))
    }

    fn nat_mul(&mut self, left: Self::Nat, right: Self::Nat) -> Self::Nat {
        if matches!(self.nat_nodes[left.0], NatNode::Constant(0)) {
            return left;
        }
        if matches!(self.nat_nodes[right.0], NatNode::Constant(0)) {
            return right;
        }
        if matches!(self.nat_nodes[left.0], NatNode::Constant(1)) {
            return right;
        }
        if matches!(self.nat_nodes[right.0], NatNode::Constant(1)) {
            return left;
        }
        self.push_nat(NatNode::Mul(left, right))
    }

    fn bits_be<const N: usize>(&mut self, bits: [Self::Bit; N]) -> Self::Nat {
        let mut value = None;
        for (index, bit) in bits.into_iter().enumerate() {
            let bit = self.bit_to_nat(bit);
            let weight = self.nat_constant(1u128 << (N - 1 - index));
            let term = self.nat_mul(bit, weight);
            value = Some(match value {
                Some(value) => self.nat_add(value, term),
                None => term,
            });
        }
        value.unwrap_or_else(|| self.nat_constant(0))
    }

    fn output(&mut self, value: Self::Nat) -> Self::Output {
        value
    }
}

#[cfg(test)]
mod tests {
    use jolt_lookup_tables::{
        tables::{and::AndTable, virtual_rotr::VirtualROTRTable},
        LookupMaterializer, LookupTable,
    };

    use super::*;

    fn extract<const XLEN: usize>(table: &impl LookupMaterializer<XLEN>) -> MaterializerGraph {
        let mut backend = MaterializerAst::new(2 * XLEN);
        let root = table.materialize(&mut backend);
        backend.finish(root)
    }

    #[test]
    fn extracts_and_from_shared_materializer() {
        let graph = extract::<2>(&AndTable::<2>);
        let lean = graph.format_for_lean().unwrap();
        assert!(lean.contains(".conj 0 1"));
        assert!(lean.contains(".conj 3 4"));
        assert!(lean.contains("root := 5"));
    }

    #[test]
    fn extracted_and_interpreter_matches_concrete_backend() {
        const XLEN: usize = 4;
        let graph = extract::<XLEN>(&AndTable::<XLEN>);

        for index in 0..(1u128 << (2 * XLEN)) {
            let point = (0..2 * XLEN)
                .map(|i| (index >> (2 * XLEN - 1 - i)) & 1 == 1)
                .collect::<Vec<_>>();
            assert_eq!(
                graph.evaluate(&point),
                u128::from(AndTable::<XLEN>.materialize_entry(index))
            );
        }
    }

    #[test]
    fn extracted_virtual_rotr_interpreter_matches_concrete_backend() {
        const XLEN: usize = 4;
        let graph = extract::<XLEN>(&VirtualROTRTable::<XLEN>);

        for index in 0..(1u128 << (2 * XLEN)) {
            let point = (0..2 * XLEN)
                .map(|i| (index >> (2 * XLEN - 1 - i)) & 1 == 1)
                .collect::<Vec<_>>();
            assert_eq!(
                graph.evaluate(&point),
                u128::from(VirtualROTRTable::<XLEN>.materialize_entry(index))
            );
        }
    }
}
