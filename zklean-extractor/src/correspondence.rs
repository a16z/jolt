use std::collections::HashMap;

use crate::mle_ast::{scalar_to_decimal_string, Scalar};

/// One node in the hash-consed associative-commutative certificate DAG.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CanonicalNode {
    Constant(Scalar),
    Input(usize),
    Add(Vec<usize>),
    Sub(usize, usize),
    Mul(Vec<usize>),
}

impl CanonicalNode {
    pub fn format_for_lean(&self) -> String {
        match self {
            Self::Constant(value) => {
                format!(".constant {}", scalar_to_decimal_string(value))
            }
            Self::Input(index) => format!(".input {index}"),
            Self::Add(terms) => format!(
                ".add [{}]",
                terms
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
            Self::Sub(left, right) => format!(".sub {left} {right}"),
            Self::Mul(factors) => format!(
                ".mul [{}]",
                factors
                    .iter()
                    .map(usize::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            ),
        }
    }
}

#[derive(Clone, Debug)]
pub struct CanonicalizedGraph {
    pub ids: Vec<usize>,
    pub root_id: usize,
}

#[derive(Clone, Debug)]
pub struct CanonicalizedMaterializer {
    pub bool_ids: Vec<usize>,
    pub nat_ids: Vec<usize>,
    pub root_id: usize,
}

/// Shared hash-consing state used for both sides of one correspondence certificate.
#[derive(Default)]
pub struct Canonicalizer {
    nodes: Vec<CanonicalNode>,
    ids: HashMap<CanonicalNode, usize>,
}

impl Canonicalizer {
    fn intern(&mut self, node: CanonicalNode) -> usize {
        if let Some(index) = self.ids.get(&node) {
            return *index;
        }
        let index = self.nodes.len();
        self.ids.insert(node.clone(), index);
        self.nodes.push(node);
        index
    }

    pub fn constant(&mut self, value: Scalar) -> usize {
        self.intern(CanonicalNode::Constant(value))
    }

    pub fn constant_u128(&mut self, value: u128) -> usize {
        self.constant([value as u64, (value >> 64) as u64, 0, 0])
    }

    pub fn input(&mut self, index: usize) -> usize {
        self.intern(CanonicalNode::Input(index))
    }

    pub fn sub(&mut self, left: usize, right: usize) -> usize {
        self.intern(CanonicalNode::Sub(left, right))
    }

    pub fn add(&mut self, left: usize, right: usize) -> usize {
        let mut terms = Vec::new();
        for index in [left, right] {
            match &self.nodes[index] {
                CanonicalNode::Constant(value) if is_scalar(value, 0) => {}
                CanonicalNode::Add(children) => terms.extend(children),
                _ => terms.push(index),
            }
        }
        terms.sort_unstable();
        match terms.as_slice() {
            [] => self.constant_u128(0),
            [index] => *index,
            _ => self.intern(CanonicalNode::Add(terms)),
        }
    }

    pub fn mul(&mut self, left: usize, right: usize) -> usize {
        if matches!(&self.nodes[left], CanonicalNode::Constant(value) if is_scalar(value, 0))
            || matches!(&self.nodes[right], CanonicalNode::Constant(value) if is_scalar(value, 0))
        {
            return self.constant_u128(0);
        }

        let mut factors = Vec::new();
        for index in [left, right] {
            match &self.nodes[index] {
                CanonicalNode::Constant(value) if is_scalar(value, 1) => {}
                CanonicalNode::Mul(children) => factors.extend(children),
                _ => factors.push(index),
            }
        }
        factors.sort_unstable();
        match factors.as_slice() {
            [] => self.constant_u128(1),
            [index] => *index,
            _ => self.intern(CanonicalNode::Mul(factors)),
        }
    }

    pub fn nodes(&self) -> &[CanonicalNode] {
        &self.nodes
    }

    pub fn one_id(&mut self) -> usize {
        self.constant_u128(1)
    }
}

fn is_scalar(value: &Scalar, expected: u64) -> bool {
    value[0] == expected && value[1..].iter().all(|limb| *limb == 0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonicalizes_associative_commutative_identities() {
        let mut canonicalizer = Canonicalizer::default();
        let zero = canonicalizer.constant_u128(0);
        let one = canonicalizer.constant_u128(1);
        let x = canonicalizer.input(0);
        let y = canonicalizer.input(1);

        let y_plus_zero = canonicalizer.add(y, zero);
        let left_add = canonicalizer.add(x, y_plus_zero);
        let zero_plus_y = canonicalizer.add(zero, y);
        let right_add = canonicalizer.add(zero_plus_y, x);
        assert_eq!(left_add, right_add);

        let y_times_one = canonicalizer.mul(y, one);
        let left_mul = canonicalizer.mul(x, y_times_one);
        let one_times_y = canonicalizer.mul(one, y);
        let right_mul = canonicalizer.mul(one_times_y, x);
        assert_eq!(left_mul, right_mul);
    }

    #[test]
    fn preserves_subtraction_order_and_topological_references() {
        let mut canonicalizer = Canonicalizer::default();
        let x = canonicalizer.input(0);
        let y = canonicalizer.input(1);
        assert_ne!(canonicalizer.sub(x, y), canonicalizer.sub(y, x));

        for (index, node) in canonicalizer.nodes().iter().enumerate() {
            let children: Vec<_> = match node {
                CanonicalNode::Constant(_) | CanonicalNode::Input(_) => Vec::new(),
                CanonicalNode::Add(children) | CanonicalNode::Mul(children) => children.clone(),
                CanonicalNode::Sub(left, right) => vec![*left, *right],
            };
            assert!(children.into_iter().all(|child| child < index));
        }
    }
}
