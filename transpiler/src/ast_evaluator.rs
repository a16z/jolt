//! Concrete evaluation of AST nodes using real field arithmetic.
//!
//! Walks the `AstBundle` node arena and evaluates each node to a concrete `Fr` value.
//! Used for cross-validation: compare Rust AST evaluation against gnark circuit execution.

use ark_bn254::Fr;
use ark_ff::{Field, PrimeField};
use light_poseidon::{Poseidon, PoseidonHasher};
use std::cell::RefCell;
use std::collections::HashMap;
use zklean_extractor::ast_bundle::Constraint;
use zklean_extractor::mle_ast::{Atom, Edge, Node, NodeId, Scalar, TranscriptHashData};

thread_local! {
    /// Reused width-4 Circom Poseidon hasher for `TranscriptHash` evaluation.
    /// `compute_node` runs once per hash node over thousands of nodes per
    /// challenge AST, so rebuilding the round-constant tables on each call was a
    /// large, avoidable cost. `hash` resets its internal state per call (it is
    /// driven statefully by `ConcreteFieldSponge`), so reuse is value-identical.
    static POSEIDON_HASHER: RefCell<Poseidon<Fr>> = RefCell::new(
        Poseidon::<Fr>::new_circom(3).expect("failed to create Poseidon hasher"),
    );
}

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

/// Evaluate a single AST root to a concrete `Fr` given a witness map. Used by the
/// field-aligned sponge differential gate (`field_aligned_layout_matches_native_sponge`):
/// evaluate the symbolic challenge AST and compare against the native `PoseidonSponge`.
pub fn eval_root(nodes: &[Node], root: NodeId, witness: &HashMap<u16, Fr>) -> Fr {
    let mut cache = HashMap::new();
    eval_node(root, nodes, &mut cache, witness)
}

/// [`eval_root`] over many roots sharing ONE memo cache. Fiat-Shamir challenge
/// ASTs embed the whole sponge chain of every earlier challenge, so evaluating
/// `k` challenges with per-root fresh caches is quadratic in the transcript
/// length; the shared cache makes it linear. Used by the in-CI real-proof
/// challenge differential (`tests/symbolic_pipeline.rs`).
pub fn eval_roots(nodes: &[Node], roots: &[NodeId], witness: &HashMap<u16, Fr>) -> Vec<Fr> {
    let mut cache = HashMap::new();
    roots
        .iter()
        .map(|&root| eval_node(root, nodes, &mut cache, witness))
        .collect()
}

/// Uncached `NodeRef` children of `node`, pushed onto `out`. Returns `true`
/// if any were pending (i.e. `node` is not ready to compute yet).
fn push_pending_children(node: &Node, cache: &HashMap<NodeId, Fr>, out: &mut Vec<NodeId>) -> bool {
    let before = out.len();
    let mut push = |e: &Edge| {
        if let Edge::NodeRef(id) = e {
            if !cache.contains_key(id) {
                out.push(*id);
            }
        }
    };
    match node {
        Node::Atom(_) => {}
        Node::Neg(e) | Node::Inv(e) => push(e),
        Node::Add(a, b) | Node::Sub(a, b) | Node::Mul(a, b) | Node::Div(a, b) => {
            push(a);
            push(b);
        }
        Node::TranscriptHash(hash_data, state, rate_unit_a) => {
            push(state);
            push(rate_unit_a);
            for e in hash_data.as_slice() {
                push(e);
            }
        }
    }
    out.len() > before
}

/// Evaluate a node, caching results. Iterative (explicit work stack): the
/// sponge state chain alone is thousands of nodes deep, so a recursive walk
/// overflows the default test-thread stack on unoptimized builds.
fn eval_node(
    node_id: NodeId,
    nodes: &[Node],
    cache: &mut HashMap<NodeId, Fr>,
    witness: &HashMap<u16, Fr>,
) -> Fr {
    let mut stack: Vec<NodeId> = vec![node_id];
    while let Some(&id) = stack.last() {
        if cache.contains_key(&id) {
            let _ = stack.pop();
            continue;
        }
        if push_pending_children(&nodes[id], cache, &mut stack) {
            continue; // children first; `id` stays on the stack below them
        }
        let value = compute_node(&nodes[id], cache, witness);
        let _ = cache.insert(id, value);
        let _ = stack.pop();
    }
    cache[&node_id]
}

/// Compute one node whose `NodeRef` children are all cached.
fn compute_node(node: &Node, cache: &HashMap<NodeId, Fr>, witness: &HashMap<u16, Fr>) -> Fr {
    let val = |e: &Edge| -> Fr {
        match e {
            Edge::Atom(atom) => eval_atom(atom, witness),
            Edge::NodeRef(id) => cache[id],
        }
    };

    match node {
        Node::Atom(atom) => eval_atom(atom, witness),

        Node::Add(a, b) => val(a) + val(b),
        Node::Sub(a, b) => val(a) - val(b),
        Node::Mul(a, b) => val(a) * val(b),
        Node::Div(a, b) => val(a) * val(b).inverse().expect("div by zero"),
        Node::Neg(a) => -val(a),
        Node::Inv(a) => val(a).inverse().expect("inv of zero"),

        Node::TranscriptHash(hash_data, state_edge, rate_unit_a_edge) => {
            let state = val(state_edge);
            let rate_unit_a = val(rate_unit_a_edge);

            let TranscriptHashData::Poseidon(data_edge) = hash_data;
            let data = val(data_edge);
            POSEIDON_HASHER.with(|hasher| {
                hasher
                    .borrow_mut()
                    .hash(&[state, rate_unit_a, data])
                    .expect("Poseidon hash failed")
            })
        }
    }
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
