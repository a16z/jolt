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

use crate::mle_ast::{node_arena, set_pending_commitment_chunks, Edge, MleAst, Node, NodeId};

// =============================================================================
// Input and Constraint Types
// =============================================================================

/// The witness type for an input variable.
///
/// Determines how it's treated in circuit generation:
/// - `PublicStatement`: Fixed for a given program (constant in circuit)
/// - `ProofData`: Varies per proof (variable witness in circuit)
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum WitnessType {
    /// Public statement data (constant in the circuit).
    /// This includes things like: program bytecode hash, memory layout params,
    /// input/output hashes, etc. These are absorbed into the transcript during
    /// fiat_shamir_preamble but are fixed for a given program.
    PublicStatement,
    /// Proof data (variable in the circuit).
    /// This includes everything that comes from the proof: commitments,
    /// sumcheck coefficients, opening claims, etc. These vary per proof.
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
    pub index: u32,
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
            constraint_cse: Vec::new(),
            constraints: Vec::new(),
            inputs: Vec::new(),
        }
    }

    /// Add an input variable description with default target field (Fr).
    pub fn add_input(&mut self, index: u32, name: impl Into<String>, witness_type: WitnessType) {
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
        index: u32,
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

    /// Count inputs for a specific target field.
    pub fn count_inputs_for_field(&self, field: TargetField) -> usize {
        self.inputs
            .iter()
            .filter(|i| i.target_field == field)
            .count()
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

    /// Run CSE (Common Subexpression Elimination) on all constraints.
    ///
    /// This performs ref-counting CSE: any node referenced more than once
    /// within a constraint's expression tree is hoisted to a named variable.
    ///
    /// Each constraint gets isolated CSE bindings (stored in `constraint_cse`).
    /// This prevents aliasing bugs where structurally identical nodes from
    /// different constraints would incorrectly share variables.
    ///
    /// Call this after `snapshot_arena()` and before code generation.
    ///
    /// # Example
    /// ```ignore
    /// bundle.snapshot_arena();
    /// bundle.run_cse();
    /// // Now generate code. CSE bindings are pre-computed
    /// ```
    pub fn run_cse(&mut self) {
        self.constraint_cse = self
            .constraints
            .iter()
            .map(|constraint| self.compute_cse_for_constraint(constraint.root))
            .collect();
    }

    /// Compute CSE bindings for a single constraint.
    ///
    /// Uses ref-counting: nodes with ref_count > 1 are hoisted.
    fn compute_cse_for_constraint(&self, root: NodeId) -> ConstraintCse {
        // Phase 1: Count references to each node
        let ref_counts = self.count_refs(root);

        // Phase 2: Build post-order traversal and collect hoisted nodes
        let post_order = self.build_post_order(root);

        // Phase 3: Collect nodes that should be hoisted (ref_count > 1, not atoms)
        let mut bindings = Vec::new();

        for node_id in post_order {
            let ref_count = ref_counts.get(&node_id).copied().unwrap_or(1);

            // Skip atoms, since they're always inlined
            if matches!(self.nodes[node_id], Node::Atom(_)) {
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
            Node::ByteReverse(e)
            | Node::Truncate128Reverse(e)
            | Node::Truncate128(e)
            | Node::AppendU64Transform(e) => edge_to_node_id(*e).into_iter().collect(),
            Node::Add(l, r) | Node::Mul(l, r) | Node::Sub(l, r) | Node::Div(l, r) => {
                [edge_to_node_id(*l), edge_to_node_id(*r)]
                    .into_iter()
                    .flatten()
                    .collect()
            }
            Node::TranscriptHash(hash_data, state, n_rounds) => {
                let mut children: Vec<NodeId> = hash_data
                    .as_slice()
                    .iter()
                    .filter_map(|e| edge_to_node_id(*e))
                    .collect();
                children.extend(edge_to_node_id(*state));
                children.extend(edge_to_node_id(*n_rounds));
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

    /// Serialize to pretty-printed JSON string (used by write_json).
    fn to_json_pretty(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON string (used by read_json).
    fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Write to a JSON file.
    pub fn write_json(&self, path: &std::path::Path) -> std::io::Result<()> {
        let json = self
            .to_json_pretty()
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e.to_string()))?;
        std::fs::write(path, json)
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

// =============================================================================
// AstCommitment
// =============================================================================

/// Wrapper type for a commitment represented as MleAst chunks.
///
/// In the real verifier, commitments are `PCS::Commitment` (e.g., G1Affine for Dory).
/// When `append_serializable` is called, it serializes to bytes and calls `append_bytes`
/// which chunks into 32-byte pieces and hashes them with proper chaining.
///
/// For symbolic execution, we represent each chunk as an MleAst variable.
/// When `AstCommitment` is serialized, it stores the chunks in the
/// `PENDING_COMMITMENT_CHUNKS` thread-local. `PoseidonAstTranscript::append_serializable`
/// then retrieves them and performs the same hash chaining operation symbolically.
///
/// # Commitment Size
///
/// The number of chunks depends on the PCS commitment type:
/// - **Dory**: 384 bytes → 12 chunks (G1Affine on BN254)
/// - **HyperKZG**: Variable size depending on configuration
/// - **Other PCS**: Determined at symbolization time from `serialized_size()`
///
/// This type is PCS-agnostic: chunk count is derived from the concrete commitment's
/// serialized size during `symbolize_proof()`, not hardcoded here.
#[derive(Clone, Debug)]
pub struct AstCommitment {
    /// The MleAst chunks representing this commitment (one per 32 bytes of serialized form)
    pub chunks: Vec<MleAst>,
}

/// Number of bytes per chunk (one BN254 field element)
const BYTES_PER_CHUNK: usize = 32;

impl AstCommitment {
    /// Create a new AstCommitment from chunks.
    ///
    /// The number of chunks should match `ceil(serialized_size / 32)` of the
    /// concrete commitment being symbolized.
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

    /// Returns the serialized size in bytes (chunks × 32).
    #[inline]
    pub fn serialized_byte_len(&self) -> usize {
        self.chunks.len() * BYTES_PER_CHUNK
    }
}

impl CanonicalSerialize for AstCommitment {
    fn serialize_with_mode<W: std::io::Write>(
        &self,
        _writer: W,
        _compress: ark_serialize::Compress,
    ) -> Result<(), SerializationError> {
        // Store chunks in thread-local for PoseidonAstTranscript::append_serializable to retrieve
        set_pending_commitment_chunks(self.chunks.clone());
        Ok(())
    }

    fn serialized_size(&self, _compress: ark_serialize::Compress) -> usize {
        self.serialized_byte_len()
    }
}

impl CanonicalDeserialize for AstCommitment {
    fn deserialize_with_mode<R: std::io::Read>(
        _reader: R,
        _compress: ark_serialize::Compress,
        _validate: ark_serialize::Validate,
    ) -> Result<Self, SerializationError> {
        unimplemented!("AstCommitment deserialization not needed for transpilation")
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

