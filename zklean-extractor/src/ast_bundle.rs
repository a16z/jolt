//! AST Bundle and Commitment types for transpilation.
//!
//! This module contains the serializable Intermediate Representation types used for transpilation:
//! - `AstBundle`: Complete AST data for code generation
//! - `AstCommitment`: Symbolic representation of PCS commitments
//!

use ark_serialize::{CanonicalDeserialize, CanonicalSerialize, SerializationError, Valid};
use ark_std::Zero;
use serde::{Deserialize, Serialize};

use std::collections::{HashMap, HashSet};

use crate::mle_ast::{
    node_arena, set_pending_commitment_chunks, symbolize_read_bytes, Atom, Edge, MleAst, Node,
    NodeId, Scalar, TranscriptHashData,
};

// =============================================================================
// Input and Constraint Types
// =============================================================================

/// The witness type for an input variable, which drives its **gnark visibility**:
/// - `PublicStatement`: emitted as a `gnark:",public"` input — the program statement
///   (IO) plus the stage-8 binding values (opening claims, commitments) the on-chain
///   PCS check must see.
/// - `ProofData`: emitted as a `gnark:",secret"` witness — proof bytes the circuit
///   re-derives Fiat-Shamir from in-circuit, so they need not be public.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WitnessType {
    /// Public circuit input: the program IO statement and the stage-8 binding values
    /// (opening claims, polynomial/advice/trusted commitments). Kept public so an
    /// on-chain wrapper can bind them to the statement and to the deferred PCS check.
    PublicStatement,
    /// Secret circuit witness: sumcheck round polynomials, uni-skip coefficients, and
    /// other per-proof bytes. Self-binding via the in-circuit sponge, so not public.
    ProofData,
}

/// The target field for code generation.
///
/// This discriminator tells the transpilation pipeline which field a variable
/// belongs to, so codegen can emit the appropriate code (native vs emulated).
///
/// # Background
///
/// In the BN254/Grumpkin 2-cycle:
/// - Fr (BN254 scalar) = Fq (Grumpkin base): native in circuit
/// - Fq (BN254 base) = Fr (Grumpkin scalar): requires emulated arithmetic
///
/// # Extensibility
///
/// Currently only BN254 Fr/Fq are supported. Future variants could include
/// other curves (BLS12-381, Goldilocks, etc.) or extension fields.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum TargetField {
    /// BN254 scalar field (Fr). Native in circuit.
    /// Modulus: 21888242871839275222246405745257275088548364400416034343698204186575808495617
    #[default]
    Fr,

    /// BN254 base field (Fq). Requires emulated arithmetic in circuit.
    /// Modulus: 21888242871839275222246405745257275088696311157297823662689037894645226208583
    ///
    /// NOTE: Fq codegen is not yet implemented. Using Fq variables will
    /// panic at codegen time with a clear error message.
    Fq,
}

impl TargetField {
    /// Human-readable name for error messages and debugging.
    #[inline]
    pub const fn name(&self) -> &'static str {
        match self {
            Self::Fr => "Fr (BN254 scalar)",
            Self::Fq => "Fq (BN254 base)",
        }
    }

    /// Whether this field requires non-native (emulated) arithmetic.
    ///
    /// Returns true for fields that are not the native scalar field of the
    /// target circuit. Currently: Fr is native, Fq requires emulation.
    #[inline]
    pub const fn is_non_native(&self) -> bool {
        matches!(self, Self::Fq)
    }
}

/// Describes an input variable in the AST.
/// Maps `Var(i)` to its semantic meaning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputVar {
    /// The index into the vars register (matches `Atom::Var(index)`).
    pub index: u16,
    /// Human-readable name for debugging and codegen (e.g., "r_sumcheck_0", "claimed_output").
    pub name: String,
    /// Whether this is a public statement or proof data.
    pub witness_type: WitnessType,
    /// The target field for this variable (Fr or Fq).
    /// Defaults to Fr for backward compatibility with existing serialized bundles.
    #[serde(default)]
    pub target_field: TargetField,
}

/// What assertion a constraint represents.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Assertion {
    /// The expression must equal zero: `expr == 0`
    EqualZero,
    /// The expression must equal a public input by name: `expr == public_input[name]`
    EqualPublicInput { name: String },
    /// The expression must equal another node in the AST: `expr == other_node`
    EqualNode(NodeId),
}

/// A named constraint with its root expression and assertion type.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Constraint {
    /// Human-readable name for the constraint (e.g., "stage1_sumcheck_final").
    pub name: String,
    /// The root node of the expression.
    pub root: NodeId,
    /// What assertion this constraint represents.
    pub assertion: Assertion,
}

// =============================================================================
// AstBundle
// =============================================================================

/// CSE (Common Subexpression Elimination) bindings for a single constraint.
///
/// Each constraint has its own isolated CSE context to prevent aliasing bugs
/// (where structurally identical nodes from different constraints would merge).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ConstraintCse {
    /// NodeIds of hoisted subexpressions, in order.
    /// These become named variables: `cse_{constraint_idx}_{i}` for element `i`.
    pub bindings: Vec<NodeId>,
}

/// Global CSE bindings: nodes shared across ≥2 constraints, hoisted to a single
/// computation block. Avoids re-computing expensive nodes (especially TranscriptHash
/// chains) independently in each constraint function.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GlobalCse {
    /// NodeIds hoisted globally, in topological (post-order) order.
    /// These become `gcse[i]` in generated code.
    pub bindings: Vec<NodeId>,
}

/// Complete bundle of AST data for transpilation.
///
/// This structure contains everything needed to:
/// 1. Generate code for target backends
/// 2. Serialize/deserialize the AST (via JSON)
///
/// The `nodes` vec is the arena: all nodes are stored here and referenced by index.
/// The `constraint_cse` vec contains per-constraint CSE bindings (computed during `run_cse()`).
/// The `constraints` vec contains the actual assertions to be verified.
/// The `inputs` vec describes what each `Var(i)` means semantically.
///
/// ## CSE Architecture
///
/// CSE is performed at the AST level (not during codegen) for several reasons:
/// 1. **Single pass**: CSE runs once, shared across all codegen targets
/// 2. **Per-constraint isolation**: Each constraint has isolated CSE to prevent aliasing bugs
/// 3. **Simpler codegen**: Target code generators just read pre-computed bindings
/// 4. **Consistent results**: Same CSE decisions across all target backends
///
/// Call `run_cse()` after `snapshot_arena()` to compute CSE bindings.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AstBundle {
    /// The node arena - all nodes in the AST(s).
    pub nodes: Vec<Node>,
    /// Global CSE bindings: nodes shared across ≥2 constraints.
    /// Computed by `run_global_cse()` before `run_cse()`.
    #[serde(default)]
    pub global_cse: GlobalCse,
    /// Per-constraint CSE bindings, indexed by constraint index.
    /// Each constraint has its own isolated CSE context.
    #[serde(default)]
    pub constraint_cse: Vec<ConstraintCse>,
    /// The constraints to be verified.
    pub constraints: Vec<Constraint>,
    /// Input variable descriptions.
    pub inputs: Vec<InputVar>,
}

impl AstBundle {
    /// Create a new empty bundle.
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            global_cse: GlobalCse::default(),
            constraint_cse: Vec::new(),
            constraints: Vec::new(),
            inputs: Vec::new(),
        }
    }

    /// Add an input variable description with default target field (Fr).
    pub fn add_input(&mut self, index: u16, name: impl Into<String>, witness_type: WitnessType) {
        self.add_input_with_field(index, name, witness_type, TargetField::default())
    }

    /// Add an input variable description with explicit target field.
    ///
    /// Use this when the variable's field differs from the default (native) field.
    ///
    /// # Arguments
    /// * `index`: Variable index matching `Atom::Var(index)`
    /// * `name`: Human-readable name for codegen
    /// * `witness_type`: Public statement or proof data
    /// * `target_field`: Which field this variable belongs to
    pub fn add_input_with_field(
        &mut self,
        index: u16,
        name: impl Into<String>,
        witness_type: WitnessType,
        target_field: TargetField,
    ) {
        self.inputs.push(InputVar {
            index,
            name: name.into(),
            witness_type,
            target_field,
        });
    }

    /// Iterate over inputs for a specific target field.
    ///
    /// Useful for codegen to separate native vs non-native variables.
    pub fn inputs_for_field(&self, target_field: TargetField) -> impl Iterator<Item = &InputVar> {
        self.inputs
            .iter()
            .filter(move |i| i.target_field == target_field)
    }

    /// Check if any inputs use the specified target field.
    ///
    /// Use `has_inputs_for_field(TargetField::Fq)` to check if non-native
    /// arithmetic support is needed.
    pub fn has_inputs_for_field(&self, field: TargetField) -> bool {
        self.inputs.iter().any(|i| i.target_field == field)
    }

    /// Add a constraint that asserts an expression equals zero.
    pub fn add_constraint_eq_zero(&mut self, name: impl Into<String>, root: NodeId) {
        self.constraints.push(Constraint {
            name: name.into(),
            root,
            assertion: Assertion::EqualZero,
        });
    }

    /// Add a constraint that asserts an expression equals a public input.
    pub fn add_constraint_eq_public(
        &mut self,
        name: impl Into<String>,
        root: NodeId,
        public_input_name: impl Into<String>,
    ) {
        self.constraints.push(Constraint {
            name: name.into(),
            root,
            assertion: Assertion::EqualPublicInput {
                name: public_input_name.into(),
            },
        });
    }

    /// Add a constraint that asserts two expressions are equal.
    pub fn add_constraint_eq_node(&mut self, name: impl Into<String>, root: NodeId, other: NodeId) {
        self.constraints.push(Constraint {
            name: name.into(),
            root,
            assertion: Assertion::EqualNode(other),
        });
    }

    /// Snapshot the current global arena into this bundle's nodes vec.
    /// Call this after all AST construction is complete.
    pub fn snapshot_arena(&mut self) {
        let arena = node_arena();
        let guard = arena.read().expect("node arena poisoned");
        self.nodes = guard.clone();
    }

    /// Structural hash-consing + dead-node sweep (transpiler-optimization spec §5.1/§5.3).
    ///
    /// A POST-HOC pass over the snapshotted arena (deliberately NOT arena-time
    /// interning: replay must stay byte-identical and NodeId assignment deterministic;
    /// see the spec's §5.1 rationale). Two effects, one remap:
    ///
    /// 1. **Hash-consing**: nodes are interned by (kind, canonicalized children), with
    ///    commutative canonicalization for `Add`/`Mul` (operands sorted in the intern
    ///    key only — the surviving node keeps its original operand order, which is
    ///    value-identical). Edges to `Node::Atom` nodes are normalized to inline
    ///    `Edge::Atom`s so the two representations of the same atom unify. Duplicates
    ///    map to the *minimum* NodeId representative — the arena is topological
    ///    (children strictly precede parents), so min-id representatives preserve that
    ///    invariant and a single forward scan sees final canonical children.
    /// 2. **Dead-node sweep**: nodes unreachable from the constraint roots (incl.
    ///    `EqualNode` targets) and `extra_roots` are dropped, and surviving nodes are
    ///    compacted to dense NodeIds (in original order, preserving topology).
    ///
    /// All edge kinds are remapped: plain children, `TranscriptHash` data/state/rounds
    /// edges, constraint roots, `EqualNode` targets, and the caller's `extra_roots`
    /// (remapped in place — used by the pipeline for squeezed-challenge ASTs that live
    /// outside the constraint set). Inputs are Var-indexed and unaffected.
    ///
    /// Must run BEFORE `run_global_cse()`/`run_cse()`: CSE bindings are NodeId lists
    /// and are invalidated by the compaction (they are reset here defensively).
    pub fn canonicalize_and_sweep(&mut self, extra_roots: &mut [NodeId]) -> CanonicalizeStats {
        let n = self.nodes.len();

        // Pass 1: forward hash-consing scan. Children precede parents, so canon[] is
        // final for every child when its parent is visited.
        let mut canon: Vec<NodeId> = (0..n).collect();
        let mut intern: HashMap<Node, NodeId> = HashMap::with_capacity(n);
        let mut duplicates_merged = 0usize;
        for i in 0..n {
            let remapped = map_node_edges(self.nodes[i].clone(), &mut |e| match e {
                Edge::NodeRef(id) => {
                    debug_assert!(id < i, "arena must be topological (children < parent)");
                    let c = canon[id];
                    // Normalize references-to-atom-nodes to inline atoms so both
                    // representations of the same atom unify under one key.
                    match &self.nodes[c] {
                        Node::Atom(a) => Edge::Atom(*a),
                        _ => Edge::NodeRef(c),
                    }
                }
                atom => atom,
            });
            self.nodes[i] = remapped.clone();
            match intern.entry(commutative_key(remapped)) {
                std::collections::hash_map::Entry::Occupied(entry) => {
                    canon[i] = *entry.get();
                    duplicates_merged += 1;
                }
                std::collections::hash_map::Entry::Vacant(entry) => {
                    entry.insert(i);
                }
            }
        }

        // Pass 2: reachability from (canonicalized) constraint roots + extra roots.
        let mut reachable = vec![false; n];
        let mut stack: Vec<NodeId> = Vec::new();
        for constraint in &self.constraints {
            stack.push(canon[constraint.root]);
            if let Assertion::EqualNode(other) = &constraint.assertion {
                stack.push(canon[*other]);
            }
        }
        stack.extend(extra_roots.iter().map(|r| canon[*r]));
        while let Some(id) = stack.pop() {
            if reachable[id] {
                continue;
            }
            reachable[id] = true;
            // Children of a rewritten node already point at canonical ids.
            stack.extend(self.node_children(id));
        }

        // Pass 3: compaction in original id order (preserves topological order).
        let mut new_id: Vec<NodeId> = vec![usize::MAX; n];
        let mut new_nodes: Vec<Node> = Vec::with_capacity(reachable.iter().filter(|b| **b).count());
        for i in 0..n {
            if !reachable[i] {
                continue;
            }
            new_id[i] = new_nodes.len();
            let node = map_node_edges(self.nodes[i].clone(), &mut |e| match e {
                Edge::NodeRef(id) => Edge::NodeRef(new_id[id]),
                atom => atom,
            });
            new_nodes.push(node);
        }
        let nodes_after = new_nodes.len();
        let dead_nodes_dropped = n - duplicates_merged - nodes_after;
        self.nodes = new_nodes;

        // Pass 4: remap constraint roots, EqualNode targets, and extra roots.
        for constraint in &mut self.constraints {
            constraint.root = new_id[canon[constraint.root]];
            if let Assertion::EqualNode(other) = &mut constraint.assertion {
                *other = new_id[canon[*other]];
            }
        }
        for root in extra_roots.iter_mut() {
            *root = new_id[canon[*root]];
        }

        // CSE bindings are NodeId lists; any previously computed ones are now stale.
        self.global_cse = GlobalCse::default();
        self.constraint_cse.clear();

        CanonicalizeStats {
            nodes_before: n,
            duplicates_merged,
            dead_nodes_dropped,
            nodes_after,
        }
    }

    /// Run global CSE: identify nodes shared across ≥2 constraints and hoist them into
    /// a global block computed once, instead of being duplicated in each constraint
    /// function.
    ///
    /// EVERY non-atom node reachable from ≥2 distinct constraints is hoisted — not just
    /// `TranscriptHash` nodes. Shared compound nodes (eq-poly products, challenge
    /// powers, etc.) sit *above* the hash nodes in the DAG, so a hash-only filter
    /// (with a downward-only dependency walk) would leave them to be re-emitted in
    /// every consuming constraint. Atoms stay inline (free); only multi-constraint
    /// compound nodes hoist.
    ///
    /// Call this after `snapshot_arena()` and before `run_cse()`.
    pub fn run_global_cse(&mut self) {
        // Phase 1: For each constraint, collect the set of reachable NodeIds
        let mut node_to_constraints: HashMap<NodeId, Vec<usize>> = HashMap::new();
        for (idx, constraint) in self.constraints.iter().enumerate() {
            let refs = self.count_refs(constraint.root);
            for node_id in refs.keys() {
                node_to_constraints.entry(*node_id).or_default().push(idx);
            }
            // Also include EqualNode targets
            if let Assertion::EqualNode(other_id) = &constraint.assertion {
                let other_refs = self.count_refs(*other_id);
                for node_id in other_refs.keys() {
                    node_to_constraints.entry(*node_id).or_default().push(idx);
                }
            }
        }

        // Phase 2: Hoist every non-atom node reachable from ≥2 distinct constraints.
        // `node_to_constraints` already enumerates all such nodes, so the old downward
        // dependency expansion is unnecessary (children of a hoisted node that are
        // themselves multi-constraint are caught here directly).
        let mut global_nodes: HashSet<NodeId> = HashSet::new();
        for (&node_id, constraints) in &node_to_constraints {
            if matches!(self.nodes[node_id], Node::Atom(_)) {
                continue;
            }
            let mut unique: Vec<usize> = constraints.clone();
            unique.sort_unstable();
            unique.dedup();
            if unique.len() >= 2 {
                global_nodes.insert(node_id);
            }
        }

        if global_nodes.is_empty() {
            self.global_cse = GlobalCse::default();
            return;
        }

        // Phase 3: Topological sort (post-order) of the global set so each node's
        // children are computed before it.
        let bindings = self.topological_sort_subset(&global_nodes);

        self.global_cse = GlobalCse { bindings };
    }

    /// Topological sort a subset of nodes in post-order (children before parents).
    ///
    /// Roots are visited in ascending `NodeId` order (not `HashSet` iteration order) so
    /// the output — and therefore the `gcse[i]` index assignment derived from it — is
    /// DETERMINISTIC across runs. The arena assigns NodeIds in a fixed construction
    /// order, so this yields reproducible generated circuits (hence a reproducible
    /// Groth16 proving/verifying key) for the same proof.
    fn topological_sort_subset(&self, subset: &HashSet<NodeId>) -> Vec<NodeId> {
        let mut result = Vec::new();
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut stack: Vec<(NodeId, bool)> = Vec::new();

        let mut roots: Vec<NodeId> = subset.iter().copied().collect();
        roots.sort_unstable();
        for node_id in roots {
            if visited.contains(&node_id) {
                continue;
            }
            stack.push((node_id, false));

            while let Some((nid, children_processed)) = stack.pop() {
                if children_processed {
                    if subset.contains(&nid) {
                        result.push(nid);
                    }
                    continue;
                }

                if visited.contains(&nid) {
                    continue;
                }
                visited.insert(nid);

                stack.push((nid, true));

                // Only traverse children that are in the subset
                for child_id in self.node_children(nid).into_iter().rev() {
                    if subset.contains(&child_id) && !visited.contains(&child_id) {
                        stack.push((child_id, false));
                    }
                }
            }
        }

        result
    }

    /// Run CSE (Common Subexpression Elimination) on all constraints.
    ///
    /// This performs ref-counting CSE: any node referenced more than once
    /// within a constraint's expression tree is hoisted to a named variable.
    ///
    /// Nodes already in `global_cse` are excluded (they're computed globally).
    ///
    /// Call this after `run_global_cse()` and before code generation.
    ///
    /// # Example
    /// ```ignore
    /// bundle.snapshot_arena();
    /// bundle.run_global_cse();
    /// bundle.run_cse();
    /// // Now generate code. CSE bindings are pre-computed
    /// ```
    pub fn run_cse(&mut self) {
        let global_set: HashSet<NodeId> = self.global_cse.bindings.iter().copied().collect();
        self.constraint_cse = self
            .constraints
            .iter()
            .map(|constraint| self.compute_cse_for_constraint(constraint.root, &global_set))
            .collect();
    }

    /// Compute CSE bindings for a single constraint.
    ///
    /// Uses ref-counting: nodes with ref_count > 1 are hoisted.
    /// Nodes in `global_set` are excluded (already hoisted globally).
    fn compute_cse_for_constraint(
        &self,
        root: NodeId,
        global_set: &HashSet<NodeId>,
    ) -> ConstraintCse {
        // Phase 1: Count references to each node
        let ref_counts = self.count_refs(root);

        // Phase 2: Build post-order traversal and collect hoisted nodes
        let post_order = self.build_post_order(root);

        // Phase 3: Collect nodes that should be hoisted (ref_count > 1, not atoms, not global)
        let mut bindings = Vec::new();

        for node_id in post_order {
            let ref_count = ref_counts.get(&node_id).copied().unwrap_or(1);

            // Skip atoms, since they're always inlined
            if matches!(self.nodes[node_id], Node::Atom(_)) {
                continue;
            }

            // Skip nodes already in global CSE
            if global_set.contains(&node_id) {
                continue;
            }

            // Hoist if referenced more than once
            if ref_count > 1 {
                bindings.push(node_id);
            }
        }

        ConstraintCse { bindings }
    }

    /// Count how many times each node is referenced in a constraint's tree.
    fn count_refs(&self, root: NodeId) -> HashMap<NodeId, usize> {
        let mut ref_counts: HashMap<NodeId, usize> = HashMap::new();
        let mut stack = vec![root];

        while let Some(node_id) = stack.pop() {
            *ref_counts.entry(node_id).or_insert(0) += 1;

            // Only traverse children on first visit
            if ref_counts[&node_id] == 1 {
                stack.extend(self.node_children(node_id));
            }
        }

        ref_counts
    }

    /// Build post-order traversal of a constraint's tree.
    fn build_post_order(&self, root: NodeId) -> Vec<NodeId> {
        let mut post_order = Vec::new();
        let mut visited: HashSet<NodeId> = HashSet::new();
        let mut stack: Vec<(NodeId, bool)> = vec![(root, false)];

        while let Some((node_id, children_processed)) = stack.pop() {
            if children_processed {
                post_order.push(node_id);
                continue;
            }

            if visited.contains(&node_id) {
                continue;
            }
            visited.insert(node_id);

            stack.push((node_id, true));

            // Push children in reverse order for left-to-right processing
            for child_id in self.node_children(node_id).into_iter().rev() {
                if !visited.contains(&child_id) {
                    stack.push((child_id, false));
                }
            }
        }

        post_order
    }

    /// Get child NodeIds for a node.
    fn node_children(&self, node_id: NodeId) -> Vec<NodeId> {
        fn edge_to_node_id(edge: Edge) -> Option<NodeId> {
            match edge {
                Edge::NodeRef(id) => Some(id),
                Edge::Atom(_) => None,
            }
        }

        match &self.nodes[node_id] {
            Node::Atom(_) => vec![],
            Node::Neg(e) | Node::Inv(e) => edge_to_node_id(*e).into_iter().collect(),
            Node::Add(l, r) | Node::Mul(l, r) | Node::Sub(l, r) | Node::Div(l, r) => {
                [edge_to_node_id(*l), edge_to_node_id(*r)]
                    .into_iter()
                    .flatten()
                    .collect()
            }
            Node::TranscriptHash(hash_data, state, rate_unit_a) => {
                let mut children: Vec<NodeId> = hash_data
                    .as_slice()
                    .iter()
                    .filter_map(|e| edge_to_node_id(*e))
                    .collect();
                children.extend(edge_to_node_id(*state));
                children.extend(edge_to_node_id(*rate_unit_a));
                children
            }
        }
    }

    /// Check if CSE has been computed for this bundle.
    ///
    /// Returns true if CSE bindings exist for all constraints.
    pub fn has_cse(&self) -> bool {
        !self.constraint_cse.is_empty() && self.constraint_cse.len() == self.constraints.len()
    }

    /// Get CSE bindings for a specific constraint.
    ///
    /// Returns `None` if CSE hasn't been run or constraint index is out of bounds.
    pub fn get_cse_bindings(&self, constraint_idx: usize) -> Option<&[NodeId]> {
        self.constraint_cse
            .get(constraint_idx)
            .map(|cse| cse.bindings.as_slice())
    }

    /// Get the number of public statement inputs.
    pub fn num_public_inputs(&self) -> usize {
        self.inputs
            .iter()
            .filter(|i| i.witness_type == WitnessType::PublicStatement)
            .count()
    }

    /// Get the number of proof data inputs.
    pub fn num_proof_inputs(&self) -> usize {
        self.inputs
            .iter()
            .filter(|i| i.witness_type == WitnessType::ProofData)
            .count()
    }

    /// Deserialize from JSON string (used by read_json).
    fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Write to a JSON file.
    ///
    /// Streams compact JSON straight to a buffered file handle rather than
    /// materializing a pretty-printed `String` of the whole arena first — the arena can
    /// hold millions of nodes, so the intermediate `String` (several× the on-disk size)
    /// was a large, avoidable peak-memory spike.
    pub fn write_json(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let mut file = std::io::BufWriter::new(std::fs::File::create(path)?);
        serde_json::to_writer(&mut file, self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        file.flush()
    }

    /// Read from a JSON file.
    pub fn read_json(path: &std::path::Path) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        Self::from_json(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))
    }
}

impl Default for AstBundle {
    fn default() -> Self {
        Self::new()
    }
}

/// Counters returned by [`AstBundle::canonicalize_and_sweep`].
#[derive(Debug, Default, Clone, Copy)]
pub struct CanonicalizeStats {
    pub nodes_before: usize,
    /// Structural duplicates merged into an earlier identical node.
    pub duplicates_merged: usize,
    /// Canonical nodes unreachable from any constraint/extra root.
    pub dead_nodes_dropped: usize,
    pub nodes_after: usize,
}

/// Rebuild a node with every child edge passed through `f` (plain children,
/// `TranscriptHash` data/state/rate_unit_a edges included). Non-edge payloads
/// (atoms) are preserved.
fn map_node_edges(node: Node, f: &mut impl FnMut(Edge) -> Edge) -> Node {
    match node {
        Node::Atom(a) => Node::Atom(a),
        Node::Neg(e) => Node::Neg(f(e)),
        Node::Inv(e) => Node::Inv(f(e)),
        Node::Add(a, b) => Node::Add(f(a), f(b)),
        Node::Mul(a, b) => Node::Mul(f(a), f(b)),
        Node::Sub(a, b) => Node::Sub(f(a), f(b)),
        Node::Div(a, b) => Node::Div(f(a), f(b)),
        Node::TranscriptHash(data, state, rate_unit_a) => {
            let TranscriptHashData::Poseidon(e) = data;
            let data = TranscriptHashData::Poseidon(f(e));
            Node::TranscriptHash(data, f(state), f(rate_unit_a))
        }
    }
}

/// Total order on edges used for commutative canonicalization of `Add`/`Mul`
/// intern keys (any fixed total order works; it only has to be deterministic).
fn edge_sort_key(e: &Edge) -> (u8, u128, Scalar) {
    match e {
        Edge::Atom(Atom::Scalar(s)) => (0, 0, *s),
        Edge::Atom(Atom::Var(v)) => (1, *v as u128, [0; 4]),
        Edge::Atom(Atom::NamedVar(v)) => (2, *v as u128, [0; 4]),
        Edge::NodeRef(id) => (3, *id as u128, [0; 4]),
    }
}

/// Intern key for hash-consing: the node itself, with `Add`/`Mul` operands sorted
/// (field + and × are commutative, so `Mul(a, b)` and `Mul(b, a)` are
/// value-identical). `Sub`/`Div` are NOT commutative and keep operand order.
fn commutative_key(node: Node) -> Node {
    match node {
        Node::Add(a, b) if edge_sort_key(&b) < edge_sort_key(&a) => Node::Add(b, a),
        Node::Mul(a, b) if edge_sort_key(&b) < edge_sort_key(&a) => Node::Mul(b, a),
        other => other,
    }
}

// =============================================================================
// AstCommitment
// =============================================================================

/// Wrapper type for a commitment represented as MleAst chunks.
///
/// In the real verifier, commitments are `PCS::Commitment` (a Dory GT
/// element, 384 canonical bytes). The field-aligned Poseidon transcript
/// (specs/transpiler-optimization-spec.md §4.2) absorbs each commitment as
/// the byte rule over its serialization: [`COMMITMENT_CHUNKS`] = 13
/// little-endian chunks of [`BYTES_PER_CHUNK`] = 31 bytes (the last chunk is
/// the 12-byte remainder; 12×31 + 12 = 384). Each chunk is < 2²⁴⁸ < r, so
/// chunk ↦ `Fr` is injective and byte-reconstructible — this re-chunking is
/// what dissolved the review-spec "GT bytes as 32-byte reductions" blocker.
///
/// For symbolic execution, each chunk is an MleAst witness variable. When
/// `AstCommitment` is serialized, it stores the chunks in the
/// `PENDING_COMMITMENT_CHUNKS` thread-local; the transpiler's
/// `SymbolicVerifierFs::absorb_commitment` retrieves them and feeds the
/// sponge layout's commitment hook.
#[derive(Clone, Debug)]
pub struct AstCommitment {
    /// The MleAst chunks representing this commitment (13 byte-rule chunks
    /// per Dory GT: 12×31B + 1×12B).
    pub chunks: Vec<MleAst>,
}

/// Bytes per byte-rule chunk (31 = the largest whole-byte count with
/// chunk < 2²⁴⁸ < r, so every chunk embeds injectively in a BN254 `Fr`).
/// Mirrors `jolt_transcript::BYTE_RULE_CHUNK`.
pub const BYTES_PER_CHUNK: usize = 31;

/// Canonical byte length of one Dory GT commitment (Fq12, compressed ==
/// uncompressed = 384 bytes) — the only commitment type the transpiler
/// handles today.
pub const COMMITMENT_BYTES: usize = 384;

/// Byte-rule chunk count per commitment: ceil(384 / 31) = 13
/// (12 full 31-byte chunks + one 12-byte tail).
pub const COMMITMENT_CHUNKS: usize = COMMITMENT_BYTES.div_ceil(BYTES_PER_CHUNK);

impl AstCommitment {
    /// Create a new AstCommitment from chunks.
    ///
    /// The number of chunks should match `ceil(serialized_size / 31)` of the
    /// concrete commitment being symbolized ([`COMMITMENT_CHUNKS`] for Dory).
    ///
    /// # Panics
    /// Panics if `chunks` is empty.
    pub fn new(chunks: Vec<MleAst>) -> Self {
        assert!(
            !chunks.is_empty(),
            "AstCommitment must have at least one chunk"
        );
        Self { chunks }
    }
}

impl CanonicalSerialize for AstCommitment {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        _writer: W,
        _compress: ark_serialize::Compress,
    ) -> Result<(), SerializationError> {
        // Store chunks in thread-local for SymbolicVerifierFs::absorb_commitment to retrieve
        set_pending_commitment_chunks(self.chunks.clone());
        Ok(())
    }

    fn serialized_size(&self, _compress: ark_serialize::Compress) -> usize {
        // The 31-byte chunking is not length-invertible, so this is fixed to the
        // Dory GT width the deserializer consumes.
        COMMITMENT_BYTES
    }
}

/// Reads one commitment's worth of REAL proof bytes ([`COMMITMENT_BYTES`] =
/// Dory GT — an immutable const, not a mutable thread-local, so there is no
/// set-without-restore hazard; code-review #4) and symbolizes each byte-rule
/// chunk (12×31B + 1×12B, zero-padded into the 32-byte hook buffer — LE, so
/// the witness value equals `Fr::from_le_bytes_mod_order(chunk)`) into a
/// fresh witness variable via the `set_read_symbolizer` hook. Used by the
/// transpiler's `SymbolicVerifierFs::read_commitments` on the
/// commitment/advice frames.
impl CanonicalDeserialize for AstCommitment {
    fn deserialize_with_mode<R: std::io::Read>(
        mut reader: R,
        _compress: ark_serialize::Compress,
        _validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        let mut chunks = Vec::with_capacity(COMMITMENT_CHUNKS);
        let mut remaining = COMMITMENT_BYTES;
        while remaining > 0 {
            let take = remaining.min(BYTES_PER_CHUNK);
            let mut bytes = [0u8; 32];
            reader
                .read_exact(&mut bytes[..take])
                .map_err(|_| SerializationError::InvalidData)?;
            chunks.push(symbolize_read_bytes(&bytes).ok_or(SerializationError::InvalidData)?);
            remaining -= take;
        }
        debug_assert_eq!(chunks.len(), COMMITMENT_CHUNKS);
        Ok(Self { chunks })
    }
}

impl Valid for AstCommitment {
    fn check(&self) -> Result<(), SerializationError> {
        Ok(())
    }
}

impl Default for AstCommitment {
    fn default() -> Self {
        // Create a single zero chunk (minimal valid commitment).
        // Real commitments will have chunk count derived from serialized_size().
        Self {
            chunks: vec![MleAst::zero()],
        }
    }
}

impl PartialEq for AstCommitment {
    fn eq(&self, other: &Self) -> bool {
        // Compare arena indices directly to avoid MleAst::eq which may register constraints
        self.chunks.len() == other.chunks.len()
            && self
                .chunks
                .iter()
                .zip(other.chunks.iter())
                .all(|(a, b)| a.root() == b.root())
    }
}

#[cfg(test)]
mod canonicalize_tests {
    use super::*;

    fn var(i: u16) -> Edge {
        Edge::Atom(Atom::Var(i))
    }

    fn scalar(x: u64) -> Edge {
        Edge::Atom(Atom::Scalar([x, 0, 0, 0]))
    }

    /// Bundle built directly on local nodes (no global arena involvement).
    fn bundle_with(nodes: Vec<Node>) -> AstBundle {
        AstBundle {
            nodes,
            ..AstBundle::new()
        }
    }

    #[test]
    fn merges_structurally_identical_subtrees() {
        // Two identical subtrees (v0 * v1) feeding one Add: exactly one Mul survives.
        let mut bundle = bundle_with(vec![
            Node::Mul(var(0), var(1)),
            Node::Mul(var(0), var(1)),
            Node::Add(Edge::NodeRef(0), Edge::NodeRef(1)),
        ]);
        bundle.add_constraint_eq_zero("c0", 2);

        let stats = bundle.canonicalize_and_sweep(&mut []);

        assert_eq!(stats.duplicates_merged, 1);
        assert_eq!(bundle.nodes.len(), 2);
        let root = bundle.constraints[0].root;
        match &bundle.nodes[root] {
            Node::Add(Edge::NodeRef(a), Edge::NodeRef(b)) => {
                assert_eq!(a, b, "both edges must point at the single surviving Mul");
                assert!(matches!(bundle.nodes[*a], Node::Mul(_, _)));
            }
            other => panic!("expected Add of two NodeRefs, got {other:?}"),
        }
    }

    #[test]
    fn commutative_canonicalization_merges_swapped_mul_but_not_sub() {
        let mut bundle = bundle_with(vec![
            Node::Mul(var(0), var(1)),
            Node::Mul(var(1), var(0)), // dup of 0 under commutativity
            Node::Sub(var(0), var(1)),
            Node::Sub(var(1), var(0)), // NOT a dup: Sub is order-sensitive
            Node::Add(Edge::NodeRef(0), Edge::NodeRef(1)),
            Node::Add(Edge::NodeRef(2), Edge::NodeRef(3)),
            Node::Add(Edge::NodeRef(4), Edge::NodeRef(5)),
        ]);
        bundle.add_constraint_eq_zero("c0", 6);

        let stats = bundle.canonicalize_and_sweep(&mut []);

        assert_eq!(stats.duplicates_merged, 1, "only the swapped Mul merges");
        let n_subs = bundle
            .nodes
            .iter()
            .filter(|n| matches!(n, Node::Sub(_, _)))
            .count();
        assert_eq!(n_subs, 2, "both Sub orientations must survive");
        let n_muls = bundle
            .nodes
            .iter()
            .filter(|n| matches!(n, Node::Mul(_, _)))
            .count();
        assert_eq!(n_muls, 1);
    }

    #[test]
    fn sweeps_dead_nodes_and_remaps_equal_node_targets() {
        let mut bundle = bundle_with(vec![
            Node::Atom(Atom::Scalar([7, 0, 0, 0])), // dead (only ever atom-inlined)
            Node::Mul(var(0), scalar(3)),           // constraint root
            Node::Mul(var(9), var(9)),              // dead: unreachable
            Node::Add(var(1), scalar(2)),           // EqualNode target
        ]);
        bundle.add_constraint_eq_node("c0", 1, 3);

        let stats = bundle.canonicalize_and_sweep(&mut []);

        assert_eq!(stats.dead_nodes_dropped, 2);
        assert_eq!(bundle.nodes.len(), 2);
        let root = bundle.constraints[0].root;
        assert!(matches!(bundle.nodes[root], Node::Mul(_, _)));
        match &bundle.constraints[0].assertion {
            Assertion::EqualNode(other) => {
                assert!(matches!(bundle.nodes[*other], Node::Add(_, _)));
            }
            other => panic!("expected EqualNode, got {other:?}"),
        }
    }

    #[test]
    fn remaps_transcript_hash_data_edges_through_duplicates() {
        let mut bundle = bundle_with(vec![
            Node::Add(var(0), var(1)),
            Node::Add(var(0), var(1)), // dup of 0
            Node::TranscriptHash(
                TranscriptHashData::Poseidon(Edge::NodeRef(1)),
                var(2),
                scalar(1),
            ),
            Node::TranscriptHash(
                TranscriptHashData::Poseidon(Edge::NodeRef(0)),
                var(2),
                scalar(1),
            ),
            Node::Sub(Edge::NodeRef(2), Edge::NodeRef(3)),
        ]);
        bundle.add_constraint_eq_zero("c0", 4);

        let stats = bundle.canonicalize_and_sweep(&mut []);

        // The Add dup merges; the two hashes then become identical and merge too.
        assert_eq!(stats.duplicates_merged, 2);
        let n_hashes = bundle
            .nodes
            .iter()
            .filter(|n| matches!(n, Node::TranscriptHash(_, _, _)))
            .count();
        assert_eq!(n_hashes, 1);
        let root = bundle.constraints[0].root;
        match &bundle.nodes[root] {
            Node::Sub(Edge::NodeRef(a), Edge::NodeRef(b)) => {
                assert_eq!(a, b);
                match &bundle.nodes[*a] {
                    Node::TranscriptHash(TranscriptHashData::Poseidon(Edge::NodeRef(d)), _, _) => {
                        assert!(matches!(bundle.nodes[*d], Node::Add(_, _)));
                    }
                    other => panic!("expected Poseidon hash with NodeRef data edge, got {other:?}"),
                }
            }
            other => panic!("expected Sub of two NodeRefs, got {other:?}"),
        }
    }

    #[test]
    fn extra_roots_stay_alive_and_are_remapped() {
        let mut bundle = bundle_with(vec![
            Node::Mul(var(0), var(1)),
            Node::Mul(var(0), var(1)), // dup of 0; an extra root points here
            Node::Add(Edge::NodeRef(0), scalar(5)), // constraint root
            Node::Mul(var(7), var(8)), // extra root, NOT constraint-reachable
        ]);
        bundle.add_constraint_eq_zero("c0", 2);
        let mut extra_roots = vec![1, 3];

        bundle.canonicalize_and_sweep(&mut extra_roots);

        // Extra root 1 collapses onto the canonical Mul(v0, v1); extra root 3 survives
        // the sweep despite being unreachable from constraints.
        let r0 = extra_roots[0];
        let r1 = extra_roots[1];
        assert!(matches!(bundle.nodes[r0], Node::Mul(var0, var1)
            if var0 == var(0) && var1 == var(1)));
        assert!(matches!(bundle.nodes[r1], Node::Mul(var7, var8)
            if var7 == var(7) && var8 == var(8)));
        // And the constraint root's child edge points at the same canonical Mul.
        match &bundle.nodes[bundle.constraints[0].root] {
            Node::Add(Edge::NodeRef(a), _) => assert_eq!(*a, r0),
            other => panic!("expected Add(NodeRef, _), got {other:?}"),
        }
    }

    #[test]
    fn non_commutative_kinds_never_merge() {
        // Div swaps, Sub swaps, and cross-kind (Add vs Mul) must all stay
        // distinct — only the literal Add/Mul operand swap is canonicalized.
        let mut bundle = bundle_with(vec![
            Node::Div(var(0), var(1)),
            Node::Div(var(1), var(0)), // NOT a dup: Div is order-sensitive
            Node::Sub(var(0), var(1)),
            Node::Sub(var(1), var(0)), // NOT a dup: Sub is order-sensitive
            Node::Add(var(0), var(1)),
            Node::Mul(var(0), var(1)), // NOT a dup of the Add: different kind
            Node::Add(Edge::NodeRef(0), Edge::NodeRef(1)),
            Node::Add(Edge::NodeRef(2), Edge::NodeRef(3)),
            Node::Add(Edge::NodeRef(4), Edge::NodeRef(5)),
            Node::Add(Edge::NodeRef(6), Edge::NodeRef(7)),
            Node::Add(Edge::NodeRef(8), Edge::NodeRef(9)),
        ]);
        bundle.add_constraint_eq_zero("c0", 10);

        let stats = bundle.canonicalize_and_sweep(&mut []);

        assert_eq!(stats.duplicates_merged, 0);
        assert_eq!(stats.dead_nodes_dropped, 0);
        assert_eq!(bundle.nodes.len(), 11);
    }

    #[test]
    fn transcript_hash_chains_do_not_collapse() {
        // Sponge chain h1 = H(d, s0, r), h2 = H(d, h1, r): same data + rounds but the
        // state edge differs by construction — h2 must NOT merge with h1. A second
        // hash with the same data + state but a different round count must also stay
        // distinct. Only a hash with ALL THREE of (data, state, rounds) identical
        // merges (pure-function dedup).
        let d = var(0);
        let s0 = var(1);
        let mut bundle = bundle_with(vec![
            Node::TranscriptHash(TranscriptHashData::Poseidon(d), s0, scalar(1)), // h1
            Node::TranscriptHash(TranscriptHashData::Poseidon(d), Edge::NodeRef(0), scalar(1)), // h2: state = h1
            Node::TranscriptHash(TranscriptHashData::Poseidon(d), s0, scalar(2)), // h3: rounds differ
            Node::TranscriptHash(TranscriptHashData::Poseidon(d), s0, scalar(1)), // h4: true dup of h1
            Node::Add(Edge::NodeRef(1), Edge::NodeRef(2)),
            Node::Add(Edge::NodeRef(4), Edge::NodeRef(3)),
        ]);
        bundle.add_constraint_eq_zero("c0", 5);

        let stats = bundle.canonicalize_and_sweep(&mut []);

        assert_eq!(
            stats.duplicates_merged, 1,
            "only the (data,state,rounds)-identical h4 merges"
        );
        let n_hashes = bundle
            .nodes
            .iter()
            .filter(|n| matches!(n, Node::TranscriptHash(_, _, _)))
            .count();
        assert_eq!(
            n_hashes, 3,
            "h1, h2 (chained state), h3 (different rounds) all survive"
        );
        // The chain edge survives intact: some hash's state edge points at another hash.
        let chained = bundle.nodes.iter().any(|n| {
            matches!(n, Node::TranscriptHash(_, Edge::NodeRef(s), _)
                if matches!(bundle.nodes[*s], Node::TranscriptHash(_, _, _)))
        });
        assert!(chained, "h2's state edge must still reference h1");
    }

    #[test]
    fn atom_node_references_unify_with_inline_atoms() {
        // Mul(NodeRef -> Atom(v0), v1) must merge with Mul(inline v0, v1).
        let mut bundle = bundle_with(vec![
            Node::Atom(Atom::Var(0)),
            Node::Mul(Edge::NodeRef(0), var(1)),
            Node::Mul(var(0), var(1)),
            Node::Add(Edge::NodeRef(1), Edge::NodeRef(2)),
        ]);
        bundle.add_constraint_eq_zero("c0", 3);

        let stats = bundle.canonicalize_and_sweep(&mut []);

        assert_eq!(stats.duplicates_merged, 1);
        let n_muls = bundle
            .nodes
            .iter()
            .filter(|n| matches!(n, Node::Mul(_, _)))
            .count();
        assert_eq!(n_muls, 1);
    }
}
