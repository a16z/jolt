//! Gnark code generation from zkLean's MLE AST
//!
//! This module traverses zkLean's global NODE_ARENA and generates
//! corresponding Gnark/Go circuit code.
//!
//! Supports Common Subexpression Elimination (CSE) to reduce circuit size
//! by hoisting repeated subexpressions into named variables.

use std::collections::{BTreeSet, HashMap};
use zklean_extractor::mle_ast::{get_node, Atom, Edge, MleAst, Node};

/// Format a scalar value ([u64; 4]) for Gnark code generation.
/// Large values (that overflow Go's int64) are formatted as bigInt("...") calls.
fn format_scalar_for_gnark(limbs: [u64; 4]) -> String {
    // Check if it fits in u64 (only limb[0] is non-zero)
    if limbs[1] == 0 && limbs[2] == 0 && limbs[3] == 0 {
        let value = limbs[0];
        // Go's int64 max is 9223372036854775807
        if value <= i64::MAX as u64 {
            return format!("{}", value);
        }
    }

    // Too large for int64, convert to decimal string and use bigInt helper
    use num_bigint::BigUint;

    let mut value = BigUint::from(limbs[3]);
    value = (value << 64) + limbs[2];
    value = (value << 64) + limbs[1];
    value = (value << 64) + limbs[0];

    format!("bigInt(\"{}\")", value)
}

/// State for memoized code generation with reference counting
pub struct MemoizedCodeGen {
    /// Reference counts for each NodeId (computed in first pass)
    ref_counts: HashMap<usize, usize>,
    /// Maps NodeId to CSE variable name (e.g., "cse_0" or "cse_3_0" for constraint 3)
    generated: HashMap<usize, String>,
    /// CSE variable definitions in order
    bindings: Vec<String>,
    /// Next CSE variable index
    cse_counter: usize,
    /// Collected input variable indices
    vars: BTreeSet<u16>,
    /// Maps variable index to input name (e.g., 0 -> "UniSkipCoeff0")
    var_names: HashMap<u16, String>,
    /// Optional constraint index for per-constraint CSE naming (None = global CSE)
    constraint_idx: Option<usize>,
}

impl MemoizedCodeGen {
    pub fn new() -> Self {
        Self {
            ref_counts: HashMap::new(),
            generated: HashMap::new(),
            bindings: Vec::new(),
            cse_counter: 0,
            vars: BTreeSet::new(),
            var_names: HashMap::new(),
            constraint_idx: None,
        }
    }

    /// Create a new MemoizedCodeGen with custom variable names
    pub fn with_var_names(var_names: HashMap<u16, String>) -> Self {
        Self {
            ref_counts: HashMap::new(),
            generated: HashMap::new(),
            bindings: Vec::new(),
            cse_counter: 0,
            vars: BTreeSet::new(),
            var_names,
            constraint_idx: None,
        }
    }

    /// Create a new MemoizedCodeGen with custom variable names and a constraint index.
    /// CSE variable names will be prefixed with the constraint index (e.g., "cse_3_0" for constraint 3).
    pub fn with_var_names_and_constraint_idx(var_names: HashMap<u16, String>, constraint_idx: usize) -> Self {
        Self {
            ref_counts: HashMap::new(),
            generated: HashMap::new(),
            bindings: Vec::new(),
            cse_counter: 0,
            vars: BTreeSet::new(),
            var_names,
            constraint_idx: Some(constraint_idx),
        }
    }

    /// Generate a CSE variable name using the configured prefix
    fn make_cse_name(&self) -> String {
        match self.constraint_idx {
            Some(idx) => format!("cse_{}_{}", idx, self.cse_counter),
            None => format!("cse_{}", self.cse_counter),
        }
    }

    /// Get collected input variables
    pub fn vars(&self) -> &BTreeSet<u16> {
        &self.vars
    }

    /// Get all CSE bindings as Go code
    pub fn bindings_code(&self) -> String {
        self.bindings.join("")
    }


    /// Debug: get reference counts for all nodes
    pub fn ref_counts(&self) -> &HashMap<usize, usize> {
        &self.ref_counts
    }

    /// First pass: count references to each node (iterative to avoid stack overflow)
    pub fn count_refs(&mut self, root_node_id: usize) {
        let mut stack = vec![root_node_id];

        while let Some(node_id) = stack.pop() {
            *self.ref_counts.entry(node_id).or_insert(0) += 1;

            // Only traverse children on first visit
            if self.ref_counts[&node_id] == 1 {
                let node = get_node(node_id);
                match node {
                    Node::Atom(_) => {}
                    Node::Neg(e) | Node::Inv(e) | Node::Keccak256(e) | Node::ByteReverse(e) | Node::Truncate128Reverse(e) | Node::Truncate128(e) | Node::MulTwoPow192(e) => {
                        if let Edge::NodeRef(id) = e {
                            stack.push(id);
                        }
                    }
                    Node::Add(e1, e2) | Node::Mul(e1, e2) | Node::Sub(e1, e2) | Node::Div(e1, e2) => {
                        if let Edge::NodeRef(id) = e1 {
                            stack.push(id);
                        }
                        if let Edge::NodeRef(id) = e2 {
                            stack.push(id);
                        }
                    }
                    Node::Poseidon(e1, e2, e3) => {
                        if let Edge::NodeRef(id) = e1 {
                            stack.push(id);
                        }
                        if let Edge::NodeRef(id) = e2 {
                            stack.push(id);
                        }
                        if let Edge::NodeRef(id) = e3 {
                            stack.push(id);
                        }
                    }
                }
            }
        }
    }
    /// Generate Gnark expression for an atom
    fn atom_to_gnark(&mut self, atom: Atom) -> String {
        match atom {
            Atom::Scalar(value) => format_scalar_for_gnark(value),
            Atom::Var(index) => {
                self.vars.insert(index);
                // Use custom variable name if available, otherwise fall back to X_{index}
                if let Some(name) = self.var_names.get(&index) {
                    format!("circuit.{}", sanitize_go_name(name))
                } else {
                    format!("circuit.X_{}", index)
                }
            }
            Atom::NamedVar(index) => {
                // Use constraint-prefixed name if available
                match self.constraint_idx {
                    Some(constraint_idx) => format!("cse_{}_{}", constraint_idx, index),
                    None => format!("cse_{}", index),
                }
            }
        }
    }

    /// Generate Gnark expression for a node, with memoization based on ref count
    /// (Iterative implementation to avoid stack overflow on deep ASTs)
    pub fn generate_expr(&mut self, root_node_id: usize) -> String {
        // Phase 1: Build post-order traversal (children before parents)
        // We need to process nodes in an order where all children are processed before their parent
        let mut post_order: Vec<usize> = Vec::new();
        let mut visited: std::collections::HashSet<usize> = std::collections::HashSet::new();
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

            // Push children (they'll be processed first due to stack LIFO)
            let node = get_node(node_id);
            match node {
                Node::Atom(_) => {}
                Node::Neg(e) | Node::Inv(e) | Node::Keccak256(e) | Node::ByteReverse(e)
                | Node::Truncate128Reverse(e) | Node::Truncate128(e) | Node::MulTwoPow192(e) => {
                    if let Edge::NodeRef(id) = e {
                        if !visited.contains(&id) {
                            stack.push((id, false));
                        }
                    }
                }
                Node::Add(e1, e2) | Node::Mul(e1, e2) | Node::Sub(e1, e2) | Node::Div(e1, e2) => {
                    if let Edge::NodeRef(id) = e2 {
                        if !visited.contains(&id) {
                            stack.push((id, false));
                        }
                    }
                    if let Edge::NodeRef(id) = e1 {
                        if !visited.contains(&id) {
                            stack.push((id, false));
                        }
                    }
                }
                Node::Poseidon(e1, e2, e3) => {
                    if let Edge::NodeRef(id) = e3 {
                        if !visited.contains(&id) {
                            stack.push((id, false));
                        }
                    }
                    if let Edge::NodeRef(id) = e2 {
                        if !visited.contains(&id) {
                            stack.push((id, false));
                        }
                    }
                    if let Edge::NodeRef(id) = e1 {
                        if !visited.contains(&id) {
                            stack.push((id, false));
                        }
                    }
                }
            }
        }

        // Phase 2: Generate expressions in post-order (children before parents)
        for node_id in post_order {
            // Skip if already generated
            if self.generated.contains_key(&node_id) {
                continue;
            }

            let node = get_node(node_id);

            // For atoms, just generate directly without hoisting
            if matches!(node, Node::Atom(_)) {
                // Don't store atoms in generated - they're always inlined
                continue;
            }

            // Generate the expression for this node (children are already in self.generated or are atoms)
            let expr = match node {
                Node::Atom(atom) => self.atom_to_gnark(atom),
                Node::Add(left, right) => {
                    let l = self.edge_to_gnark_iterative(left);
                    let r = self.edge_to_gnark_iterative(right);
                    format!("api.Add({}, {})", l, r)
                }
                Node::Mul(left, right) => {
                    let l = self.edge_to_gnark_iterative(left);
                    let r = self.edge_to_gnark_iterative(right);
                    format!("api.Mul({}, {})", l, r)
                }
                Node::Sub(left, right) => {
                    let l = self.edge_to_gnark_iterative(left);
                    let r = self.edge_to_gnark_iterative(right);
                    format!("api.Sub({}, {})", l, r)
                }
                Node::Div(left, right) => {
                    let l = self.edge_to_gnark_iterative(left);
                    let r = self.edge_to_gnark_iterative(right);
                    format!("api.Div({}, {})", l, r)
                }
                Node::Neg(child) => {
                    let c = self.edge_to_gnark_iterative(child);
                    format!("api.Neg({})", c)
                }
                Node::Inv(child) => {
                    let c = self.edge_to_gnark_iterative(child);
                    format!("api.Inverse({})", c)
                }
                Node::Poseidon(state, n_rounds, data) => {
                    let s = self.edge_to_gnark_iterative(state);
                    let r = self.edge_to_gnark_iterative(n_rounds);
                    let d = self.edge_to_gnark_iterative(data);
                    format!("poseidon.Hash(api, {}, {}, {})", s, r, d)
                }
                Node::Keccak256(input) => {
                    let i = self.edge_to_gnark_iterative(input);
                    format!("keccak.Keccak256(api, {})", i)
                }
                Node::ByteReverse(input) => {
                    let i = self.edge_to_gnark_iterative(input);
                    format!("poseidon.ByteReverse(api, {})", i)
                }
                Node::Truncate128Reverse(input) => {
                    let i = self.edge_to_gnark_iterative(input);
                    format!("poseidon.Truncate128Reverse(api, {})", i)
                }
                Node::Truncate128(input) => {
                    let i = self.edge_to_gnark_iterative(input);
                    format!("poseidon.Truncate128(api, {})", i)
                }
                Node::MulTwoPow192(input) => {
                    let i = self.edge_to_gnark_iterative(input);
                    format!("poseidon.AppendU64Transform(api, {})", i)
                }
            };

            // Hoist to CSE variable if referenced more than once
            let ref_count = self.ref_counts.get(&node_id).copied().unwrap_or(1);
            if ref_count > 1 {
                let var_name = self.make_cse_name();
                self.cse_counter += 1;
                self.bindings.push(format!("\t{} := {}\n", var_name, expr));
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
            let node = get_node(root_node_id);
            if let Node::Atom(atom) = node {
                self.atom_to_gnark(atom)
            } else {
                panic!("Root node {} not found in generated expressions", root_node_id)
            }
        }
    }

    /// Non-recursive edge_to_gnark that looks up already-generated expressions
    fn edge_to_gnark_iterative(&mut self, edge: Edge) -> String {
        match edge {
            Edge::Atom(atom) => self.atom_to_gnark(atom),
            Edge::NodeRef(node_id) => {
                // Child should already be generated (we're in post-order)
                if let Some(expr) = self.generated.get(&node_id) {
                    expr.clone()
                } else {
                    // Must be an atom node
                    let node = get_node(node_id);
                    if let Node::Atom(atom) = node {
                        self.atom_to_gnark(atom)
                    } else {
                        panic!("Node {} not found in generated - post-order traversal bug?", node_id)
                    }
                }
            }
        }
    }
}

/// Statistics about constant assertions detected during codegen
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
/// **Per-Constraint CSE**: To avoid the node aliasing bug where structurally identical
/// expressions from different constraints get merged, we use per-constraint CSE contexts.
/// This means each constraint gets its own CSE namespace (cse_0_0, cse_0_1, ... for constraint 0,
/// cse_1_0, cse_1_1, ... for constraint 1, etc.). This prevents CSE from merging expressions
/// that are structurally identical but semantically different across constraints.
pub fn generate_circuit_from_bundle_with_stats(
    bundle: &zklean_extractor::mle_ast::AstBundle,
    circuit_name: &str,
) -> (String, ConstantAssertionStats) {
    use zklean_extractor::mle_ast::{Assertion, InputKind};

    let mut stats = ConstantAssertionStats::default();

    // Build var_names mapping from bundle inputs
    let var_names: HashMap<u16, String> = bundle
        .inputs
        .iter()
        .map(|input| (input.index, input.name.clone()))
        .collect();

    // Per-constraint CSE: generate each constraint with its own CSE context
    // This avoids the node aliasing bug where structurally identical expressions
    // from different constraints get incorrectly merged.
    let mut all_bindings_code = String::new();
    // constraint_data: (name, expr, assertion, is_const, const_val, other_expr for EqualNode)
    let mut constraint_data: Vec<(String, String, &Assertion, bool, Option<[u64; 4]>, Option<String>)> = Vec::new();
    let mut all_vars: BTreeSet<u16> = BTreeSet::new();

    for (constraint_idx, c) in bundle.constraints.iter().enumerate() {
        // Create a fresh codegen context for this constraint with per-constraint CSE naming
        // This ensures CSE variables from different constraints don't collide
        // e.g., constraint 0 uses cse_0_0, cse_0_1, constraint 1 uses cse_1_0, cse_1_1, etc.
        let mut codegen = MemoizedCodeGen::with_var_names_and_constraint_idx(
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

        // Collect vars used in this constraint
        all_vars.extend(codegen.vars().iter());

        // Check if constant
        let ast = MleAst::from_node_id(c.root);
        let is_const = ast.is_constant();
        let const_val = if is_const {
            ast.try_evaluate_constant()
        } else {
            None
        };

        // Collect bindings for this constraint (already have prefixed names from codegen)
        let constraint_bindings = codegen.bindings_code();
        if !constraint_bindings.is_empty() {
            all_bindings_code.push_str(&format!("\t// CSE bindings for constraint {}\n", constraint_idx));
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
    output.push_str("\n");
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
    output.push_str(&format!("type {} struct {{\n", circuit_name));

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
        "func (circuit *{}) Define(api frontend.API) error {{\n",
        circuit_name
    ));

    // CSE bindings
    if !bindings_code.is_empty() {
        output.push_str("\t// Memoized subexpressions\n");
        output.push_str(&bindings_code);
        output.push_str("\n");
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
                        output.push_str(&format!("\t// {} = 0 (statically verified, skipped)\n\n", name));
                        stats.constant_skipped += 1;
                        continue;
                    } else {
                        // Constant != 0 - static failure
                        output.push_str(&format!(
                            "\t// {} STATIC FAILURE: constant != 0\n",
                            name
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

        output.push_str(&format!("\t// {}\n", name));
        let var_name = sanitize_go_name(name);
        output.push_str(&format!("\t{} := {}\n", var_name, expr));

        match assertion {
            Assertion::EqualZero => {
                output.push_str(&format!("\tapi.AssertIsEqual({}, 0)\n\n", var_name));
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
                    "\tapi.AssertIsEqual({}, {})\n\n",
                    var_name, other
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

/// Convert an Edge to Gnark code
fn edge_to_gnark(edge: Edge) -> String {
    match edge {
        Edge::Atom(atom) => atom_to_gnark(atom),
        Edge::NodeRef(node_id) => generate_gnark_expr(node_id),
    }
}

/// Convert an Edge to Gnark code, collecting variable indices
fn edge_to_gnark_with_vars(edge: Edge, vars: &mut BTreeSet<u16>) -> String {
    match edge {
        Edge::Atom(atom) => atom_to_gnark_with_vars(atom, vars),
        Edge::NodeRef(node_id) => generate_gnark_expr_with_vars(node_id, vars),
    }
}

/// Convert an Atom to Gnark code
fn atom_to_gnark(atom: Atom) -> String {
    match atom {
        Atom::Scalar(value) => format_scalar_for_gnark(value),
        Atom::Var(index) => {
            // Variable reference - maps to circuit input
            format!("circuit.X_{}", index)
        }
        Atom::NamedVar(index) => {
            // Let-bound variable (for CSE)
            format!("cse_{}", index)
        }
    }
}

/// Convert an Atom to Gnark code, collecting variable indices
fn atom_to_gnark_with_vars(atom: Atom, vars: &mut BTreeSet<u16>) -> String {
    match atom {
        Atom::Scalar(value) => format_scalar_for_gnark(value),
        Atom::Var(index) => {
            vars.insert(index);
            format!("circuit.X_{}", index)
        }
        Atom::NamedVar(index) => format!("cse_{}", index),
    }
}

/// Generate Gnark expression from a node in the AST
///
/// Maps AST operations to Gnark API calls:
/// - `Add(a, b)` → `api.Add(a, b)`
/// - `Mul(a, b)` → `api.Mul(a, b)`
/// - etc.
pub fn generate_gnark_expr(node_id: usize) -> String {
    let node = get_node(node_id);

    match node {
        Node::Atom(atom) => atom_to_gnark(atom),

        Node::Add(left, right) => {
            format!("api.Add({}, {})", edge_to_gnark(left), edge_to_gnark(right))
        }

        Node::Mul(left, right) => {
            format!("api.Mul({}, {})", edge_to_gnark(left), edge_to_gnark(right))
        }

        Node::Sub(left, right) => {
            format!("api.Sub({}, {})", edge_to_gnark(left), edge_to_gnark(right))
        }

        Node::Neg(child) => {
            format!("api.Neg({})", edge_to_gnark(child))
        }

        Node::Inv(child) => {
            format!("api.Inverse({})", edge_to_gnark(child))
        }

        Node::Div(left, right) => {
            format!("api.Div({}, {})", edge_to_gnark(left), edge_to_gnark(right))
        }

        Node::Poseidon(state, n_rounds, data) => {
            format!(
                "poseidon.Hash(api, {}, {}, {})",
                edge_to_gnark(state),
                edge_to_gnark(n_rounds),
                edge_to_gnark(data)
            )
        }

        Node::Keccak256(input) => {
            format!("keccak.Keccak256(api, {})", edge_to_gnark(input))
        }

        Node::ByteReverse(input) => {
            format!("poseidon.ByteReverse(api, {})", edge_to_gnark(input))
        }

        Node::Truncate128Reverse(input) => {
            format!("poseidon.Truncate128Reverse(api, {})", edge_to_gnark(input))
        }

        Node::Truncate128(input) => {
            format!("poseidon.Truncate128(api, {})", edge_to_gnark(input))
        }

        Node::MulTwoPow192(input) => {
            format!("poseidon.AppendU64Transform(api, {})", edge_to_gnark(input))
        }
    }
}

/// Generate Gnark expression while collecting all variable indices used
fn generate_gnark_expr_with_vars(node_id: usize, vars: &mut BTreeSet<u16>) -> String {
    let node = get_node(node_id);

    match node {
        Node::Atom(atom) => atom_to_gnark_with_vars(atom, vars),

        Node::Add(left, right) => {
            format!(
                "api.Add({}, {})",
                edge_to_gnark_with_vars(left, vars),
                edge_to_gnark_with_vars(right, vars)
            )
        }

        Node::Mul(left, right) => {
            format!(
                "api.Mul({}, {})",
                edge_to_gnark_with_vars(left, vars),
                edge_to_gnark_with_vars(right, vars)
            )
        }

        Node::Sub(left, right) => {
            format!(
                "api.Sub({}, {})",
                edge_to_gnark_with_vars(left, vars),
                edge_to_gnark_with_vars(right, vars)
            )
        }

        Node::Neg(child) => {
            format!("api.Neg({})", edge_to_gnark_with_vars(child, vars))
        }

        Node::Inv(child) => {
            format!("api.Inverse({})", edge_to_gnark_with_vars(child, vars))
        }

        Node::Div(left, right) => {
            format!(
                "api.Div({}, {})",
                edge_to_gnark_with_vars(left, vars),
                edge_to_gnark_with_vars(right, vars)
            )
        }

        Node::Poseidon(state, n_rounds, data) => {
            format!(
                "poseidon.Hash(api, {}, {}, {})",
                edge_to_gnark_with_vars(state, vars),
                edge_to_gnark_with_vars(n_rounds, vars),
                edge_to_gnark_with_vars(data, vars)
            )
        }

        Node::Keccak256(input) => {
            format!(
                "keccak.Keccak256(api, {})",
                edge_to_gnark_with_vars(input, vars)
            )
        }

        Node::ByteReverse(input) => {
            format!("poseidon.ByteReverse(api, {})", edge_to_gnark_with_vars(input, vars))
        }

        Node::Truncate128Reverse(input) => {
            format!("poseidon.Truncate128Reverse(api, {})", edge_to_gnark_with_vars(input, vars))
        }

        Node::Truncate128(input) => {
            format!("poseidon.Truncate128(api, {})", edge_to_gnark_with_vars(input, vars))
        }

        Node::MulTwoPow192(input) => {
            format!("poseidon.AppendU64Transform(api, {})", edge_to_gnark_with_vars(input, vars))
        }
    }
}

/// Generate a complete Gnark circuit from an AST root
///
/// Creates a Go file with:
/// - Package declaration
/// - Circuit struct with inputs
/// - Define() method with constraints
pub fn generate_circuit(root_node_id: usize, circuit_name: &str) -> String {
    // First pass: collect all variable indices used
    let mut vars = BTreeSet::new();
    let expr = generate_gnark_expr_with_vars(root_node_id, &mut vars);

    let mut output = String::new();

    // Package and imports
    output.push_str("package jolt_verifier\n\n");
    // Check if we need poseidon import
    if expr.contains("poseidon.") {
        output.push_str("import (\n");
        output.push_str("\t\"github.com/consensys/gnark/frontend\"\n");
        output.push_str("\t\"jolt_verifier/poseidon\"\n");
        output.push_str(")\n\n");
    } else {
        output.push_str("import \"github.com/consensys/gnark/frontend\"\n\n");
    }

    // Circuit struct with all used variables
    output.push_str(&format!("type {} struct {{\n", circuit_name));
    for var_idx in &vars {
        output.push_str(&format!(
            "\tX_{} frontend.Variable `gnark:\",public\"`\n",
            var_idx
        ));
    }
    output.push_str("\tOutput frontend.Variable `gnark:\",public\"`\n");
    output.push_str("}\n\n");

    // Define method
    output.push_str(&format!(
        "func (circuit *{}) Define(api frontend.API) error {{\n",
        circuit_name
    ));
    output.push_str(&format!("\tresult := {}\n", expr));
    output.push_str("\tapi.AssertIsEqual(result, circuit.Output)\n");
    output.push_str("\treturn nil\n");
    output.push_str("}\n");

    output
}
