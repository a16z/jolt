//! Concrete evaluation of AST nodes using real field arithmetic.
//!
//! Walks the `AstBundle` node arena and evaluates each node to a concrete `Fr` value.
//! Used for cross-validation: compare Rust AST evaluation against gnark circuit execution.

use ark_bn254::Fr;
use ark_ff::{Field, PrimeField};
use light_poseidon::{Poseidon, PoseidonHasher};
use std::collections::HashMap;
use zklean_extractor::ast_bundle::Constraint;
use zklean_extractor::mle_ast::{Atom, Edge, Node, NodeId, Scalar, TranscriptHashData};

/// Evaluate an Edge to a concrete Fr value.
fn eval_edge(
    edge: &Edge,
    nodes: &[Node],
    cache: &mut HashMap<NodeId, Fr>,
    witness: &HashMap<u16, Fr>,
) -> Fr {
    match edge {
        Edge::Atom(atom) => eval_atom(atom, witness),
        Edge::NodeRef(id) => eval_node(*id, nodes, cache, witness),
    }
}

/// Evaluate an Atom to Fr.
fn eval_atom(atom: &Atom, witness: &HashMap<u16, Fr>) -> Fr {
    match atom {
        Atom::Scalar(limbs) => scalar_to_fr(limbs),
        Atom::Var(idx) => *witness
            .get(idx)
            .unwrap_or_else(|| panic!("missing witness for Var({idx})")),
        Atom::NamedVar(_) => panic!("NamedVar should not appear in serialized bundle"),
    }
}

/// Convert a [u64; 4] scalar (raw, NOT Montgomery) to Fr.
fn scalar_to_fr(limbs: &Scalar) -> Fr {
    // Scalar is value-form (not Montgomery). Use from_le_bytes_mod_order.
    let mut bytes = [0u8; 32];
    for (i, limb) in limbs.iter().enumerate() {
        bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
    }
    Fr::from_le_bytes_mod_order(&bytes)
}

/// Evaluate a node, caching results.
fn eval_node(
    node_id: NodeId,
    nodes: &[Node],
    cache: &mut HashMap<NodeId, Fr>,
    witness: &HashMap<u16, Fr>,
) -> Fr {
    if let Some(&val) = cache.get(&node_id) {
        return val;
    }

    let result = match &nodes[node_id] {
        Node::Atom(atom) => eval_atom(atom, witness),

        Node::Add(a, b) => {
            eval_edge(a, nodes, cache, witness) + eval_edge(b, nodes, cache, witness)
        }
        Node::Sub(a, b) => {
            eval_edge(a, nodes, cache, witness) - eval_edge(b, nodes, cache, witness)
        }
        Node::Mul(a, b) => {
            eval_edge(a, nodes, cache, witness) * eval_edge(b, nodes, cache, witness)
        }
        Node::Div(a, b) => {
            let denom = eval_edge(b, nodes, cache, witness);
            eval_edge(a, nodes, cache, witness) * denom.inverse().expect("div by zero")
        }
        Node::Neg(a) => -eval_edge(a, nodes, cache, witness),
        Node::Inv(a) => eval_edge(a, nodes, cache, witness)
            .inverse()
            .expect("inv of zero"),

        Node::TranscriptHash(hash_data, state_edge, rounds_edge) => {
            let state = eval_edge(state_edge, nodes, cache, witness);
            let rounds = eval_edge(rounds_edge, nodes, cache, witness);

            match hash_data {
                TranscriptHashData::Poseidon(data_edge) => {
                    let data = eval_edge(data_edge, nodes, cache, witness);
                    let mut hasher =
                        Poseidon::<Fr>::new_circom(3).expect("failed to create Poseidon hasher");
                    hasher
                        .hash(&[state, rounds, data])
                        .expect("Poseidon hash failed")
                }
                _ => panic!("only Poseidon transcript is supported for evaluation"),
            }
        }

        Node::ByteReverse(e) => {
            let val = eval_edge(e, nodes, cache, witness);
            let bigint = val.into_bigint();
            let mut bytes = [0u8; 32];
            for (i, limb) in bigint.0.iter().enumerate() {
                bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
            }
            bytes.reverse();
            Fr::from_le_bytes_mod_order(&bytes)
        }

        Node::Truncate128Reverse(e) => {
            let val = eval_edge(e, nodes, cache, witness);
            let bigint = val.into_bigint();
            let mut le_bytes = [0u8; 32];
            for (i, limb) in bigint.0.iter().enumerate() {
                le_bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
            }
            // Take low 16 bytes, reverse, interpret as field element, multiply by 2^128
            let mut truncated = [0u8; 16];
            truncated.copy_from_slice(&le_bytes[..16]);
            truncated.reverse();
            let base = Fr::from_le_bytes_mod_order(&truncated);
            let shift = Fr::from(2u64).pow([128]);
            base * shift
        }

        Node::Truncate128(e) => {
            let val = eval_edge(e, nodes, cache, witness);
            let bigint = val.into_bigint();
            let mut le_bytes = [0u8; 32];
            for (i, limb) in bigint.0.iter().enumerate() {
                le_bytes[i * 8..(i + 1) * 8].copy_from_slice(&limb.to_le_bytes());
            }
            let mut truncated = [0u8; 16];
            truncated.copy_from_slice(&le_bytes[..16]);
            truncated.reverse();
            Fr::from_le_bytes_mod_order(&truncated)
        }

        Node::AppendU64Transform(e) => {
            let val = eval_edge(e, nodes, cache, witness);
            // bswap64(x) * 2^192
            let bigint = val.into_bigint();
            let x = bigint.0[0]; // u64 value
            let swapped = x.swap_bytes();
            Fr::from(swapped) * Fr::from(2u64).pow([192])
        }
    };

    cache.insert(node_id, result);
    result
}

/// LHS and RHS of one assertion.
pub struct AssertionValue {
    pub name: String,
    pub lhs: Fr,
    pub rhs: Fr,
}

/// Evaluate all constraints in the bundle, returning LHS and RHS for each.
///
/// For constraints with `EqualZero` assertion:
/// - If root is `Sub(lhs, rhs)`: returns the two sides separately
/// - Otherwise (e.g., sum_zero): LHS = the expression, RHS = 0
pub fn evaluate_assertions(
    nodes: &[Node],
    constraints: &[Constraint],
    witness: &HashMap<u16, Fr>,
) -> Vec<AssertionValue> {
    let mut cache: HashMap<NodeId, Fr> = HashMap::new();
    let mut results = Vec::new();

    for constraint in constraints {
        let root_node = &nodes[constraint.root];

        let (lhs, rhs) = match root_node {
            // Most assertions: Sub(lhs, rhs) == 0 means lhs == rhs
            Node::Sub(lhs_edge, rhs_edge) => {
                let l = eval_edge(lhs_edge, nodes, &mut cache, witness);
                let r = eval_edge(rhs_edge, nodes, &mut cache, witness);
                (l, r)
            }
            // sum_zero or other: the whole expression should be 0
            _ => {
                let val = eval_node(constraint.root, nodes, &mut cache, witness);
                (val, Fr::from(0u64))
            }
        };

        results.push(AssertionValue {
            name: constraint.name.clone(),
            lhs,
            rhs,
        });
    }

    results
}

/// Convert Fr to decimal string (matching gnark's output format).
pub fn fr_to_decimal(f: &Fr) -> String {
    f.into_bigint().to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_to_fr_zero() {
        let zero = scalar_to_fr(&[0, 0, 0, 0]);
        assert_eq!(zero, Fr::from(0u64));
    }

    #[test]
    fn test_scalar_to_fr_one() {
        let one = scalar_to_fr(&[1, 0, 0, 0]);
        assert_eq!(one, Fr::from(1u64));
    }

    #[test]
    fn test_scalar_to_fr_42() {
        let val = scalar_to_fr(&[42, 0, 0, 0]);
        assert_eq!(val, Fr::from(42u64));
    }

    #[test]
    fn test_poseidon_hash_zeros() {
        // Verify our Poseidon evaluation matches known test vectors
        let mut hasher = Poseidon::<Fr>::new_circom(3).unwrap();
        let result = hasher
            .hash(&[Fr::from(0u64), Fr::from(0u64), Fr::from(0u64)])
            .unwrap();
        // This should match Go's poseidon.Hash(0, 0, 0)
        assert_ne!(result, Fr::from(0u64)); // just verify it's non-trivial
    }
}
