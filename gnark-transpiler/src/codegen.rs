//! Gnark/Go code generation from MleAst symbolic expressions.
//!
//! # Overview
//!
//! This module converts the AST (Abstract Syntax Tree) built during symbolic execution
//! into Gnark circuit code. The AST represents all arithmetic operations performed by
//! the Jolt verifier, and this module emits equivalent Go code using gnark's API.
//!
//! # Key Components
//!
//! - [`MemoizedCodeGen`]: The main code generator with CSE (Common Subexpression Elimination)
//! - [`generate_circuit_from_bundle`]: Entry point that generates a complete circuit file
//! - [`sanitize_go_name`]: Converts Rust identifiers to valid Go identifiers
//!
//! # Code Generation Pipeline
//!
//! 1. **Reference Counting**: First pass counts how many times each AST node is used
//! 2. **Post-Order Traversal**: Generate expressions bottom-up (children before parents)
//! 3. **CSE Hoisting**: Nodes used more than once become named variables (e.g., `cse_0_1`)
//! 4. **Per-Constraint Namespacing**: Each constraint gets isolated CSE to prevent aliasing bugs
//!
//! # Example Output
//!
//! ```go
//! // CSE bindings for constraint 0
//! cse_0_0 := api.Mul(circuit.Stage1_Sumcheck_R0_0, circuit.Stage1_Sumcheck_R0_1)
//! cse_0_1 := api.Add(cse_0_0, circuit.Stage1_Sumcheck_R0_2)
//!
//! // assertion_0
//! assertion_0 := api.Sub(cse_0_1, circuit.Expected_0)
//! api.AssertIsEqual(assertion_0, 0)
//! ```
//!
//! # Per-Constraint Expression Trees
//!
//! Each constraint (sumcheck assertion) gets its own isolated expression tree with
//! independent CSE namespacing: constraint 0 uses `cse_0_*`, constraint 1 uses `cse_1_*`, etc.
//!
//! **Why this architecture?** Easier debugging. When a constraint fails, its expression
//! tree is self-contained. You can trace through `cse_N_*` variables knowing they all
//! belong to constraint N, without cross-referencing expressions from other sumchecks.
//!
//! Note: Since each MleAst operation creates a unique NodeId in the arena (no structural
//! deduplication), there's no aliasing risk between constraints. The isolation is purely
//! for debugging convenience.

use std::collections::{BTreeSet, HashMap, HashSet};
use zklean_extractor::mle_ast::{
    Atom, Edge, Node, Scalar,
    scalar_add_mod, scalar_sub_mod, scalar_neg_mod, scalar_mul_mod,
};

// =============================================================================
// Helper Functions
// =============================================================================

/// Extract child node IDs from an Edge, if it's a NodeRef.
fn edge_to_child(edge: Edge) -> Option<usize> {
    match edge {
        Edge::NodeRef(id) => Some(id),
        Edge::Atom(_) => None,
    }
}

/// Extract all child node IDs from a Node.
///
/// Returns children in left-to-right order (e1, e2, e3 for ternary nodes).
/// Used by both `count_refs` and `generate_expr` for consistent traversal.
fn node_children(node: Node) -> Vec<usize> {
    match node {
        Node::Atom(_) => vec![],
        Node::Neg(e) | Node::Inv(e)
        | Node::ByteReverse(e)
        | Node::Truncate128Reverse(e)
        | Node::Truncate128(e)
        | Node::AppendU64Transform(e) => edge_to_child(e).into_iter().collect(),
        Node::Add(e1, e2)
        | Node::Mul(e1, e2)
        | Node::Sub(e1, e2)
        | Node::Div(e1, e2) => [edge_to_child(e1), edge_to_child(e2)]
            .into_iter()
            .flatten()
            .collect(),
        Node::Poseidon(e1, e2, e3) => [edge_to_child(e1), edge_to_child(e2), edge_to_child(e3)]
            .into_iter()
            .flatten()
            .collect(),
    }
}

// =============================================================================
// Types
// =============================================================================

/// Statistics about constant assertions detected during codegen.
#[derive(Debug, Default)]
pub struct ConstantAssertionStats {
    /// Total number of constraints processed
    pub total_constraints: usize,
    /// Number of constant assertions that were skipped
    pub constant_skipped: usize,
    /// Number of constant assertions that failed (non-zero constant)
    pub constant_failed: usize,
    /// Names of failed constant assertions
    pub failed_names: Vec<String>,
}

/// Memoized code generator that converts AST nodes to Gnark expressions.
///
/// This struct maintains state for a single code generation pass:
/// - Tracks reference counts to determine which expressions to hoist
/// - Caches generated expressions to avoid regenerating subtrees
/// - Collects CSE bindings (named intermediate variables)
///
/// # Usage
///
/// ```ignore
/// let mut codegen = MemoizedCodeGen::new(&bundle.nodes, var_names, constraint_idx);
/// codegen.count_refs(root_node_id);  // First pass: count references
/// let expr = codegen.generate_expr(root_node_id);  // Second pass: generate code
/// let bindings = codegen.bindings_code();  // Get CSE variable definitions
/// ```
///
/// # Per-Constraint Isolation
///
/// Create a fresh `MemoizedCodeGen` for each constraint. This gives each sumcheck
/// its own expression tree with isolated CSE variables (`cse_N_*` for constraint N),
/// making debugging easier: when a constraint fails, all `cse_N_*` variables
/// belong to that constraint.
pub(crate) struct MemoizedCodeGen<'a> {
    /// Reference to the node arena (from AstBundle.nodes)
    nodes: &'a [Node],
    /// Reference counts for each NodeId (computed in first pass)
    ref_counts: HashMap<usize, usize>,
    /// Maps NodeId to CSE variable name (e.g., "cse_0" or "cse_3_0" for constraint 3)
    generated: HashMap<usize, String>,
    /// CSE variable definitions in order
    bindings: Vec<String>,
    /// Next CSE variable index
    cse_counter: usize,
    /// Maps variable index to input name (e.g., 0 -> "UniSkipCoeff0")
    var_names: HashMap<u16, String>,
    /// Constraint index for per-constraint CSE naming (e.g., constraint 3 uses cse_3_*)
    constraint_idx: usize,
}

impl<'a> MemoizedCodeGen<'a> {
    /// Create a new MemoizedCodeGen for a specific constraint.
    ///
    /// CSE variable names are prefixed with the constraint index (e.g., "cse_3_0" for constraint 3).
    /// This keeps each constraint's expression tree isolated for easier debugging.
    pub(crate) fn new(nodes: &'a [Node], var_names: HashMap<u16, String>, constraint_idx: usize) -> Self {
        Self {
            nodes,
            ref_counts: HashMap::new(),
            generated: HashMap::new(),
            bindings: Vec::new(),
            cse_counter: 0,
            var_names,
            constraint_idx,
        }
    }

    /// Get all CSE bindings as Go code
    pub(crate) fn bindings_code(&self) -> String {
        self.bindings.join("")
    }

    /// First pass: count how many times each node is referenced.
    ///
    /// This determines which nodes should be hoisted to CSE variables.
    /// Nodes with ref_count > 1 will become named variables to avoid
    /// redundant computation in the circuit.
    ///
    /// Uses iterative traversal to avoid stack overflow on deep ASTs
    /// (the Jolt verifier AST can be very deep due to sumcheck chains).
    pub(crate) fn count_refs(&mut self, root_node_id: usize) {
        let mut stack = vec![root_node_id];

        while let Some(node_id) = stack.pop() {
            *self.ref_counts.entry(node_id).or_insert(0) += 1;

            // Only traverse children on first visit (ref_count was just set to 1)
            if self.ref_counts[&node_id] == 1 {
                stack.extend(node_children(self.nodes[node_id]));
            }
        }
    }

    /// Generate Gnark expression for a node, with memoization based on ref count.
    ///
    /// This is the main code generation method. It:
    /// 1. Builds a post-order traversal (children before parents)
    /// 2. Generates Go expressions for each node
    /// 3. Hoists multi-referenced nodes to CSE variables
    /// 4. Returns the final expression for the root
    ///
    /// Uses iterative traversal to avoid stack overflow on deep ASTs.
    /// The Jolt verifier can produce ASTs with thousands of nodes in
    /// a single chain (e.g., sumcheck polynomial evaluations).
    pub(crate) fn generate_expr(&mut self, root_node_id: usize) -> String {
        // Phase 1: Build post-order traversal (children before parents)
        // We need to process nodes in an order where all children are processed before their parent
        let mut post_order: Vec<usize> = Vec::new();
        let mut visited: HashSet<usize> = HashSet::new();
        let mut stack: Vec<(usize, bool)> = vec![(root_node_id, false)];

        while let Some((node_id, children_processed)) = stack.pop() {
            if children_processed {
                // All children have been processed, add this node to post_order
                post_order.push(node_id);
                continue;
            }

            // Skip if already in post_order (already fully processed)
            if visited.contains(&node_id) {
                continue;
            }
            visited.insert(node_id);

            // Push this node back with children_processed = true
            stack.push((node_id, true));

            // Push unvisited children (reversed so left-to-right processing due to LIFO)
            for child_id in node_children(self.nodes[node_id]).into_iter().rev() {
                if !visited.contains(&child_id) {
                    stack.push((child_id, false));
                }
            }
        }

        // Phase 2: Generate expressions in post-order (children before parents)
        for node_id in post_order {
            // Skip if already generated
            if self.generated.contains_key(&node_id) {
                continue;
            }

            let node = self.nodes[node_id];

            // For atoms, just generate directly without hoisting
            if matches!(node, Node::Atom(_)) {
                // Don't store atoms in generated - they're always inlined
                continue;
            }

            // Generate the expression for this node (children are already in self.generated or are atoms)
            let expr = match node {
                // Atoms are skipped above, this arm is unreachable
                Node::Atom(_) => unreachable!("Atoms are skipped before this match"),

                // Binary arithmetic ops
                Node::Add(l, r) => self.binary_op("Add", l, r),
                Node::Mul(l, r) => self.binary_op("Mul", l, r),
                Node::Sub(l, r) => self.binary_op("Sub", l, r),

                // Unary ops
                Node::Inv(e) => self.unary_op("api.Inverse", e),
                Node::ByteReverse(e) => self.unary_op("poseidon.ByteReverse", e),
                Node::Truncate128Reverse(e) => self.unary_op("poseidon.Truncate128Reverse", e),
                Node::Truncate128(e) => self.unary_op("poseidon.Truncate128", e),
                Node::AppendU64Transform(e) => self.unary_op("poseidon.AppendU64Transform", e),

                // Ternary: Poseidon hash
                Node::Poseidon(state, n_rounds, data) => {
                    let s = self.edge_to_gnark_iterative(state);
                    let r = self.edge_to_gnark_iterative(n_rounds);
                    let d = self.edge_to_gnark_iterative(data);
                    format!("poseidon.Hash(api, {s}, {r}, {d})")
                }

                // zklean base nodes - Jolt transpiler doesn't generate these
                Node::Neg(_) | Node::Div(_, _) => {
                    unreachable!("Neg/Div nodes not used by Jolt transpiler")
                }
            };

            // Hoist to CSE variable if referenced more than once
            let ref_count = self.ref_counts.get(&node_id).copied().unwrap_or(1);
            if ref_count > 1 {
                let var_name = self.make_cse_name();
                self.cse_counter += 1;
                self.bindings.push(format!("\t{var_name} := {expr}\n"));
                self.generated.insert(node_id, var_name);
            } else {
                // Store the expression for single-use nodes too, so children can reference it
                self.generated.insert(node_id, expr);
            }
        }

        // Return the expression for the root node
        if let Some(expr) = self.generated.get(&root_node_id) {
            expr.clone()
        } else {
            // Root was an atom
            let node = self.nodes[root_node_id];
            if let Node::Atom(atom) = node {
                self.atom_to_gnark(atom)
            } else {
                panic!("Root node {root_node_id} not found in generated expressions")
            }
        }
    }

    /// Generate a binary operation expression (api.Op(left, right))
    fn binary_op(&mut self, op: &str, left: Edge, right: Edge) -> String {
        let l = self.edge_to_gnark_iterative(left);
        let r = self.edge_to_gnark_iterative(right);
        format!("api.{op}({l}, {r})")
    }

    /// Generate a unary operation expression (func(api, arg))
    fn unary_op(&mut self, func: &str, arg: Edge) -> String {
        let a = self.edge_to_gnark_iterative(arg);
        // api.Inverse doesn't take api as first arg, poseidon helpers do
        if func.starts_with("api.") {
            format!("{func}({a})")
        } else {
            format!("{func}(api, {a})")
        }
    }

    // -------------------------------------------------------------------------
    // Private helper methods
    // -------------------------------------------------------------------------

    /// Generate a CSE variable name using the configured prefix
    fn make_cse_name(&self) -> String {
        let constraint_idx = self.constraint_idx;
        let cse_counter = self.cse_counter;
        format!("cse_{constraint_idx}_{cse_counter}")
    }

    /// Generate Gnark expression for an atom
    fn atom_to_gnark(&mut self, atom: Atom) -> String {
        match atom {
            Atom::Scalar(value) => format_scalar_for_gnark(value),
            Atom::Var(index) => {
                self.var_names.get(&index)
                    .map(|name| format!("circuit.{}", sanitize_go_name(name)))
                    .unwrap_or_else(|| format!("circuit.X_{index}"))
            }
            Atom::NamedVar(index) => {
                let constraint_idx = self.constraint_idx;
                format!("cse_{constraint_idx}_{index}")
            }
        }
    }

    /// Non-recursive edge_to_gnark that looks up already-generated expressions
    fn edge_to_gnark_iterative(&mut self, edge: Edge) -> String {
        match edge {
            Edge::Atom(atom) => self.atom_to_gnark(atom),
            Edge::NodeRef(node_id) => {
                // Child should already be generated (we're in post-order), or it's an atom
                self.generated.get(&node_id).cloned().unwrap_or_else(|| {
                    match self.nodes[node_id] {
                        Node::Atom(atom) => self.atom_to_gnark(atom),
                        _ => panic!("Node {node_id} not in generated - post-order traversal bug"),
                    }
                })
            }
        }
    }
}

// =============================================================================
// Public functions
// =============================================================================

/// Generate a complete Gnark circuit from an AstBundle.
///
/// This is the generic codegen that works with any stage/verifier.
/// It reads constraints and inputs from the bundle and generates
/// appropriate gnark code based on the Assertion types.
///
/// **Constant Assertion Handling**: If a constraint expression is entirely
/// constant (contains no variables), the assertion is verified at compile-time:
/// - If constant == 0 for EqualZero assertions, the constraint is SKIPPED
/// - If constant != 0, the function PANICS (static verification failure)
///
/// Returns the generated Go code and statistics about constant assertions.
pub fn generate_circuit_from_bundle(
    bundle: &zklean_extractor::mle_ast::AstBundle,
    circuit_name: &str,
) -> String {
    let (code, stats) = generate_circuit_from_bundle_with_stats(bundle, circuit_name);

    // Log statistics
    if stats.constant_skipped > 0 || stats.constant_failed > 0 {
        eprintln!(
            "Codegen stats: {} total constraints, {} constant-skipped, {} constant-failed",
            stats.total_constraints, stats.constant_skipped, stats.constant_failed
        );
    }

    // Panic if any constant assertions failed
    if stats.constant_failed > 0 {
        panic!(
            "Static verification failed: {} constant assertions are non-zero: {:?}",
            stats.constant_failed, stats.failed_names
        );
    }

    code
}

/// Generate a complete Gnark circuit from an AstBundle, with statistics.
///
/// This version returns statistics about constant assertion handling,
/// useful for debugging and testing.
///
/// **Per-Constraint Expression Trees**: Each constraint (sumcheck assertion) gets its own
/// isolated expression tree with independent CSE namespacing (constraint 0 uses `cse_0_*`,
/// constraint 1 uses `cse_1_*`, etc.). This makes debugging easier: when a constraint fails,
/// all `cse_N_*` variables belong to that constraint, so you can trace through its
/// expression tree in isolation without cross-referencing other sumchecks.
pub fn generate_circuit_from_bundle_with_stats(
    bundle: &zklean_extractor::mle_ast::AstBundle,
    circuit_name: &str,
) -> (String, ConstantAssertionStats) {
    use zklean_extractor::mle_ast::{Assertion, InputKind};

    // Type alias for constraint data tuple
    type ConstraintData<'a> = (String, String, &'a Assertion, bool, Option<[u64; 4]>, Option<String>);

    let mut stats = ConstantAssertionStats::default();

    // Build var_names mapping from bundle inputs
    let var_names: HashMap<u16, String> = bundle
        .inputs
        .iter()
        .map(|input| (input.index, input.name.clone()))
        .collect();

    // Per-constraint CSE: generate each constraint with its own CSE context.
    // This makes debugging easier - each constraint's cse_N_* variables are isolated.
    let mut all_bindings_code = String::new();
    let mut constraint_data: Vec<ConstraintData> = Vec::new();

    for (constraint_idx, c) in bundle.constraints.iter().enumerate() {
        // Create a fresh codegen context for this constraint with per-constraint CSE naming
        // This ensures CSE variables from different constraints don't collide
        // e.g., constraint 0 uses cse_0_0, cse_0_1, constraint 1 uses cse_1_0, cse_1_1, etc.
        let mut codegen = MemoizedCodeGen::new(
            &bundle.nodes,
            var_names.clone(),
            constraint_idx,
        );

        // Count references within this constraint only
        codegen.count_refs(c.root);
        if let Assertion::EqualNode(other_id) = &c.assertion {
            codegen.count_refs(*other_id);
        }

        // Generate expression for this constraint
        let expr = codegen.generate_expr(c.root);

        // Generate other_expr for EqualNode assertions
        let other_expr = if let Assertion::EqualNode(other_id) = &c.assertion {
            Some(codegen.generate_expr(*other_id))
        } else {
            None
        };

        // Check if constant (reads from bundle.nodes, not global arena)
        let is_const = is_node_constant_in(&bundle.nodes, c.root);
        let const_val = if is_const {
            Some(evaluate_constant_node_in(&bundle.nodes, c.root))
        } else {
            None
        };

        // Collect bindings for this constraint (already have prefixed names from codegen)
        let constraint_bindings = codegen.bindings_code();
        if !constraint_bindings.is_empty() {
            all_bindings_code.push_str(&format!("\t// CSE bindings for constraint {constraint_idx}\n"));
            all_bindings_code.push_str(&constraint_bindings);
        }

        constraint_data.push((c.name.clone(), expr, &c.assertion, is_const, const_val, other_expr));
    }

    stats.total_constraints = constraint_data.len();

    let bindings_code = all_bindings_code;

    let mut output = String::new();

    // Package and imports
    output.push_str("package jolt_verifier\n\n");
    output.push_str("import (\n");
    output.push_str("\t\"math/big\"\n");
    output.push('\n');
    output.push_str("\t\"github.com/consensys/gnark/frontend\"\n");
    if bindings_code.contains("poseidon.Hash")
        || constraint_data.iter().any(|(_, e, _, _, _, _)| e.contains("poseidon.Hash"))
    {
        output.push_str("\t\"jolt_verifier/poseidon\"\n");
    }
    output.push_str(")\n\n");

    // bigInt helper for large constants that overflow Go's int64
    output.push_str("// bigInt creates a *big.Int from a string, for constants too large for int64\n");
    output.push_str("func bigInt(s string) *big.Int {\n");
    output.push_str("\tn, _ := new(big.Int).SetString(s, 10)\n");
    output.push_str("\treturn n\n");
    output.push_str("}\n\n");

    // Circuit struct - use input descriptions from bundle
    output.push_str(&format!("type {circuit_name} struct {{\n"));

    // Add proof data inputs (variables)
    for input in bundle.inputs.iter().filter(|i| i.kind == InputKind::ProofData) {
        output.push_str(&format!(
            "\t{} frontend.Variable `gnark:\",public\"`\n",
            sanitize_go_name(&input.name)
        ));
    }

    // Add public statement inputs (constants in circuit, but still need to be declared)
    for input in bundle.inputs.iter().filter(|i| i.kind == InputKind::PublicStatement) {
        output.push_str(&format!(
            "\t{} frontend.Variable `gnark:\",public\"`\n",
            sanitize_go_name(&input.name)
        ));
    }

    // Add any public inputs referenced by EqualPublicInput assertions
    let mut public_input_names: BTreeSet<String> = BTreeSet::new();
    for constraint in &bundle.constraints {
        if let Assertion::EqualPublicInput { name } = &constraint.assertion {
            public_input_names.insert(name.clone());
        }
    }
    for name in &public_input_names {
        output.push_str(&format!(
            "\t{} frontend.Variable `gnark:\",public\"`\n",
            sanitize_go_name(name)
        ));
    }

    output.push_str("}\n\n");

    // Define method
    output.push_str(&format!(
        "func (circuit *{circuit_name}) Define(api frontend.API) error {{\n"
    ));

    // CSE bindings
    if !bindings_code.is_empty() {
        output.push_str("\t// Memoized subexpressions\n");
        output.push_str(&bindings_code);
        output.push('\n');
    }

    // Generate constraints based on assertion type
    // Skip constant assertions that are satisfied, track those that fail
    for (name, expr, assertion, is_const, const_val, other_expr) in &constraint_data {
        // Handle constant assertions specially
        if *is_const {
            match assertion {
                Assertion::EqualZero => {
                    let val = const_val.unwrap_or([0, 0, 0, 0]);
                    if val == [0, 0, 0, 0] {
                        // Constant equals zero - statically satisfied, skip
                        output.push_str(&format!("\t// {name} = 0 (statically verified, skipped)\n\n"));
                        stats.constant_skipped += 1;
                        continue;
                    } else {
                        // Constant != 0 - static failure
                        output.push_str(&format!(
                            "\t// {name} STATIC FAILURE: constant != 0\n"
                        ));
                        stats.constant_failed += 1;
                        stats.failed_names.push(name.clone());
                        // Still emit the constraint so the error is visible
                    }
                }
                _ => {
                    // For other assertion types with constants, we can't easily
                    // verify statically, so emit them as normal
                }
            }
        }

        output.push_str(&format!("\t// {name}\n"));
        let var_name = sanitize_go_name(name);
        output.push_str(&format!("\t{var_name} := {expr}\n"));

        match assertion {
            Assertion::EqualZero => {
                output.push_str(&format!("\tapi.AssertIsEqual({var_name}, 0)\n\n"));
            }
            Assertion::EqualPublicInput { name: pub_name } => {
                output.push_str(&format!(
                    "\tapi.AssertIsEqual({}, circuit.{})\n\n",
                    var_name,
                    sanitize_go_name(pub_name)
                ));
            }
            Assertion::EqualNode(_) => {
                // other_expr was generated during the constraint processing loop
                let other = other_expr.as_ref().expect("EqualNode assertion must have other_expr");
                output.push_str(&format!(
                    "\tapi.AssertIsEqual({var_name}, {other})\n\n"
                ));
            }
        }
    }

    output.push_str("\treturn nil\n");
    output.push_str("}\n");

    (output, stats)
}

/// Sanitize a name for use as a Go identifier.
/// Replaces all non-alphanumeric characters with underscores, then PascalCases each segment.
pub fn sanitize_go_name(name: &str) -> String {
    // Replace any non-alphanumeric character with underscore
    let cleaned: String = name
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect();

    // Split by underscores and PascalCase each segment
    let parts: Vec<&str> = cleaned.split('_').filter(|s| !s.is_empty()).collect();

    parts
        .iter()
        .map(|s| {
            let mut c = s.chars();
            match c.next() {
                None => String::new(),
                Some(first) => first.to_uppercase().chain(c).collect(),
            }
        })
        .collect::<Vec<_>>()
        .join("_")
}

// =============================================================================
// Private helper functions
// =============================================================================

/// Format a scalar value ([u64; 4]) for Gnark code generation.
///
/// Small values that fit in Go's int64 are emitted as literals (e.g., `42`).
/// Large values are formatted as `bigInt("...")` calls.
///
/// Note: gnark's API accepts both int and *big.Int, so using bigInt for everything
/// would also work. We use int literals for small values because it produces
/// more readable output and slightly smaller generated files.
fn format_scalar_for_gnark(limbs: [u64; 4]) -> String {
    // Check if it fits in i64 (only limb[0] is non-zero and within range)
    if limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0 {
        let value = limbs[0];
        if value <= i64::MAX as u64 {
            return format!("{value}");
        }
    }

    // Too large for int64, use bigInt helper
    use num_bigint::BigUint;

    let mut value = BigUint::from(limbs[3]);
    value = (value << 64) + limbs[2];
    value = (value << 64) + limbs[1];
    value = (value << 64) + limbs[0];

    format!("bigInt(\"{value}\")")
}

/// Check if a node is constant (contains no variables), reading from a node slice.
fn is_node_constant_in(nodes: &[Node], node_id: usize) -> bool {
    match nodes[node_id] {
        Node::Atom(Atom::Scalar(_)) => true,
        Node::Atom(Atom::Var(_)) => false,
        Node::Atom(Atom::NamedVar(_)) => false,
        Node::Neg(e) | Node::Inv(e) | Node::ByteReverse(e)
        | Node::Truncate128Reverse(e) | Node::Truncate128(e) | Node::AppendU64Transform(e) => {
            is_edge_constant_in(nodes, e)
        }
        Node::Add(e1, e2) | Node::Mul(e1, e2) | Node::Sub(e1, e2) | Node::Div(e1, e2) => {
            is_edge_constant_in(nodes, e1) && is_edge_constant_in(nodes, e2)
        }
        Node::Poseidon(e1, e2, e3) => {
            is_edge_constant_in(nodes, e1) && is_edge_constant_in(nodes, e2) && is_edge_constant_in(nodes, e3)
        }
    }
}

fn is_edge_constant_in(nodes: &[Node], edge: Edge) -> bool {
    match edge {
        Edge::Atom(Atom::Scalar(_)) => true,
        Edge::Atom(Atom::Var(_)) => false,
        Edge::Atom(Atom::NamedVar(_)) => false,
        Edge::NodeRef(id) => is_node_constant_in(nodes, id),
    }
}

/// Evaluate a constant node to its scalar value, reading from a node slice.
fn evaluate_constant_node_in(nodes: &[Node], node_id: usize) -> Scalar {
    match nodes[node_id] {
        Node::Atom(Atom::Scalar(s)) => s,
        Node::Atom(Atom::Var(_)) | Node::Atom(Atom::NamedVar(_)) => {
            panic!("Cannot evaluate non-constant node")
        }
        Node::Add(e1, e2) => {
            scalar_add_mod(evaluate_constant_edge_in(nodes, e1), evaluate_constant_edge_in(nodes, e2))
        }
        Node::Sub(e1, e2) => {
            scalar_sub_mod(evaluate_constant_edge_in(nodes, e1), evaluate_constant_edge_in(nodes, e2))
        }
        Node::Mul(e1, e2) => {
            scalar_mul_mod(evaluate_constant_edge_in(nodes, e1), evaluate_constant_edge_in(nodes, e2))
        }
        Node::Neg(e) => scalar_neg_mod(evaluate_constant_edge_in(nodes, e)),
        Node::Inv(_) | Node::Div(_, _) => {
            panic!("Modular inverse not implemented for constant evaluation")
        }
        Node::Poseidon(_, _, _) | Node::ByteReverse(_)
        | Node::Truncate128Reverse(_) | Node::Truncate128(_) | Node::AppendU64Transform(_) => {
            panic!("Hash/transform operations cannot be evaluated as constants")
        }
    }
}

fn evaluate_constant_edge_in(nodes: &[Node], edge: Edge) -> Scalar {
    match edge {
        Edge::Atom(Atom::Scalar(s)) => s,
        Edge::Atom(Atom::Var(_)) | Edge::Atom(Atom::NamedVar(_)) => {
            panic!("Cannot evaluate non-constant edge")
        }
        Edge::NodeRef(id) => evaluate_constant_node_in(nodes, id),
    }
}
