//! Gnark code generation from zkLean's MLE AST
//!
//! This module traverses zkLean's global NODE_ARENA and generates
//! corresponding Gnark/Go circuit code.
//!
//! Supports Common Subexpression Elimination (CSE) to reduce circuit size
//! by hoisting repeated subexpressions into named variables.

use std::collections::{BTreeSet, HashMap};
use zklean_extractor::mle_ast::{
    common_subexpression_elimination, common_subexpression_elimination_incremental, get_node,
    insert_node, Atom, Bindings, Edge, Node,
};

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
    /// Maps NodeId to CSE variable name (e.g., "cse_0")
    generated: HashMap<usize, String>,
    /// CSE variable definitions in order
    bindings: Vec<String>,
    /// Next CSE variable index
    cse_counter: usize,
    /// Collected input variable indices
    vars: BTreeSet<u16>,
    /// Maps variable index to input name (e.g., 0 -> "UniSkipCoeff0")
    var_names: HashMap<u16, String>,
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

    /// First pass: count references to each node
    pub fn count_refs(&mut self, node_id: usize) {
        *self.ref_counts.entry(node_id).or_insert(0) += 1;

        // Only traverse children on first visit
        if self.ref_counts[&node_id] == 1 {
            let node = get_node(node_id);
            match node {
                Node::Atom(_) => {}
                Node::Neg(e) | Node::Inv(e) | Node::Keccak256(e) | Node::ByteReverse(e) | Node::Truncate128Reverse(e) | Node::Truncate128(e) | Node::MulTwoPow192(e) => {
                    self.count_refs_edge(e);
                }
                Node::Add(e1, e2) | Node::Mul(e1, e2) | Node::Sub(e1, e2) | Node::Div(e1, e2) => {
                    self.count_refs_edge(e1);
                    self.count_refs_edge(e2);
                }
                Node::Poseidon(e1, e2, e3) => {
                    self.count_refs_edge(e1);
                    self.count_refs_edge(e2);
                    self.count_refs_edge(e3);
                }
            }
        }
    }

    fn count_refs_edge(&mut self, edge: Edge) {
        if let Edge::NodeRef(id) = edge {
            self.count_refs(id);
        }
    }

    /// Generate Gnark expression for an edge
    fn edge_to_gnark(&mut self, edge: Edge) -> String {
        match edge {
            Edge::Atom(atom) => self.atom_to_gnark(atom),
            Edge::NodeRef(node_id) => self.generate_expr(node_id),
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
            Atom::NamedVar(index) => format!("cse_{}", index),
        }
    }

    /// Generate Gnark expression for a node, with memoization based on ref count
    pub fn generate_expr(&mut self, node_id: usize) -> String {
        // Check if already generated
        if let Some(var_name) = self.generated.get(&node_id) {
            return var_name.clone();
        }

        let node = get_node(node_id);

        // For atoms, just return directly without hoisting
        if let Node::Atom(atom) = node {
            return self.atom_to_gnark(atom);
        }

        // Generate the expression for this node
        let expr = match node {
            Node::Atom(atom) => self.atom_to_gnark(atom),
            Node::Add(left, right) => {
                let l = self.edge_to_gnark(left);
                let r = self.edge_to_gnark(right);
                format!("api.Add({}, {})", l, r)
            }
            Node::Mul(left, right) => {
                let l = self.edge_to_gnark(left);
                let r = self.edge_to_gnark(right);
                format!("api.Mul({}, {})", l, r)
            }
            Node::Sub(left, right) => {
                let l = self.edge_to_gnark(left);
                let r = self.edge_to_gnark(right);
                format!("api.Sub({}, {})", l, r)
            }
            Node::Div(left, right) => {
                let l = self.edge_to_gnark(left);
                let r = self.edge_to_gnark(right);
                format!("api.Div({}, {})", l, r)
            }
            Node::Neg(child) => {
                let c = self.edge_to_gnark(child);
                format!("api.Neg({})", c)
            }
            Node::Inv(child) => {
                let c = self.edge_to_gnark(child);
                format!("api.Inverse({})", c)
            }
            Node::Poseidon(state, n_rounds, data) => {
                let s = self.edge_to_gnark(state);
                let r = self.edge_to_gnark(n_rounds);
                let d = self.edge_to_gnark(data);
                format!("poseidon.Hash(api, {}, {}, {})", s, r, d)
            }
            Node::Keccak256(input) => {
                let i = self.edge_to_gnark(input);
                format!("keccak.Keccak256(api, {})", i)
            }
            Node::ByteReverse(input) => {
                let i = self.edge_to_gnark(input);
                format!("poseidon.ByteReverse(api, {})", i)
            }
            Node::Truncate128Reverse(input) => {
                let i = self.edge_to_gnark(input);
                format!("poseidon.Truncate128Reverse(api, {})", i)
            }
            Node::Truncate128(input) => {
                let i = self.edge_to_gnark(input);
                format!("poseidon.Truncate128(api, {})", i)
            }
            Node::MulTwoPow192(input) => {
                let i = self.edge_to_gnark(input);
                format!("poseidon.AppendU64Transform(api, {})", i)
            }
        };

        // Hoist to CSE variable if referenced more than once
        let ref_count = self.ref_counts.get(&node_id).copied().unwrap_or(1);
        if ref_count > 1 {
            let var_name = format!("cse_{}", self.cse_counter);
            self.cse_counter += 1;
            self.bindings.push(format!("\t{} := {}\n", var_name, expr));
            self.generated.insert(node_id, var_name.clone());
            var_name
        } else {
            expr
        }
    }
}

/// Generate a complete Gnark circuit for Stage 1 verification with memoization.
///
/// This uses reference counting to create CSE variables only for nodes that
/// are referenced more than once, eliminating redundant computations.
pub fn generate_stage1_circuit_memoized(
    result: &jolt_core::zkvm::stage1_only_verifier::Stage1TranscriptVerificationResult<
        zklean_extractor::mle_ast::MleAst,
    >,
    circuit_name: &str,
) -> String {
    let mut codegen = MemoizedCodeGen::new();

    // First pass: count references to all nodes
    codegen.count_refs(result.power_sum_check.root());
    for check in &result.sumcheck_consistency_checks {
        codegen.count_refs(check.root());
    }
    codegen.count_refs(result.final_claim.root());

    // Second pass: generate code (CSE for nodes with refcount > 1)
    let power_sum_expr = codegen.generate_expr(result.power_sum_check.root());

    let consistency_exprs: Vec<String> = result
        .sumcheck_consistency_checks
        .iter()
        .map(|check| codegen.generate_expr(check.root()))
        .collect();

    let final_claim_expr = codegen.generate_expr(result.final_claim.root());

    let bindings_code = codegen.bindings_code();
    let vars = codegen.vars();

    let mut output = String::new();

    // Package and imports
    output.push_str("package jolt_verifier\n\n");
    output.push_str("import (\n");
    output.push_str("\t\"github.com/consensys/gnark/frontend\"\n");
    if bindings_code.contains("poseidon.Hash") || final_claim_expr.contains("poseidon.Hash") {
        output.push_str("\t\"jolt_verifier/poseidon\"\n");
    }
    output.push_str(")\n\n");

    // Circuit struct
    output.push_str(&format!("type {} struct {{\n", circuit_name));
    for var_idx in vars {
        output.push_str(&format!(
            "\tX_{} frontend.Variable `gnark:\",public\"`\n",
            var_idx
        ));
    }
    output.push_str("\tExpectedFinalClaim frontend.Variable `gnark:\",public\"`\n");
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

    // Constraints
    output.push_str("\t// Power sum check\n");
    output.push_str(&format!("\tpowerSumCheck := {}\n", power_sum_expr));
    output.push_str("\tapi.AssertIsEqual(powerSumCheck, 0)\n\n");

    for (i, expr) in consistency_exprs.iter().enumerate() {
        output.push_str(&format!("\t// Sumcheck round {}\n", i));
        output.push_str(&format!("\tconsistencyCheck{} := {}\n", i, expr));
        output.push_str(&format!(
            "\tapi.AssertIsEqual(consistencyCheck{}, 0)\n\n",
            i
        ));
    }

    output.push_str("\t// Final claim\n");
    output.push_str(&format!("\tfinalClaim := {}\n", final_claim_expr));
    output.push_str("\tapi.AssertIsEqual(finalClaim, circuit.ExpectedFinalClaim)\n\n");

    output.push_str("\treturn nil\n");
    output.push_str("}\n");

    output
}

/// Generate a complete Gnark circuit from an AstBundle.
///
/// This is the generic codegen that works with any stage/verifier.
/// It reads constraints and inputs from the bundle and generates
/// appropriate gnark code based on the Assertion types.
pub fn generate_circuit_from_bundle(
    bundle: &zklean_extractor::mle_ast::AstBundle,
    circuit_name: &str,
) -> String {
    use zklean_extractor::mle_ast::{Assertion, InputKind};

    // Build var_names mapping from bundle inputs
    let var_names: HashMap<u16, String> = bundle
        .inputs
        .iter()
        .map(|input| (input.index, input.name.clone()))
        .collect();

    let mut codegen = MemoizedCodeGen::with_var_names(var_names);

    // First pass: count references to all constraint roots
    for constraint in &bundle.constraints {
        codegen.count_refs(constraint.root);
        // Also count refs for EqualNode targets
        if let Assertion::EqualNode(other_id) = &constraint.assertion {
            codegen.count_refs(*other_id);
        }
    }

    // Second pass: generate expressions for each constraint
    let constraint_exprs: Vec<(String, String, &Assertion)> = bundle
        .constraints
        .iter()
        .map(|c| {
            let expr = codegen.generate_expr(c.root);
            (c.name.clone(), expr, &c.assertion)
        })
        .collect();

    let bindings_code = codegen.bindings_code();

    let mut output = String::new();

    // Package and imports
    output.push_str("package jolt_verifier\n\n");
    output.push_str("import (\n");
    output.push_str("\t\"math/big\"\n");
    output.push_str("\n");
    output.push_str("\t\"github.com/consensys/gnark/frontend\"\n");
    if bindings_code.contains("poseidon.Hash")
        || constraint_exprs.iter().any(|(_, e, _)| e.contains("poseidon.Hash"))
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
    for (name, expr, assertion) in &constraint_exprs {
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
            Assertion::EqualNode(other_id) => {
                let other_expr = codegen.generate_expr(*other_id);
                output.push_str(&format!(
                    "\tapi.AssertIsEqual({}, {})\n\n",
                    var_name, other_expr
                ));
            }
        }
    }

    output.push_str("\treturn nil\n");
    output.push_str("}\n");

    output
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

/// Generate a complete Gnark circuit for Stage 1 verification.
///
/// This generates a circuit that:
/// 1. Declares all input variables
/// 2. Enforces power_sum_check == 0
/// 3. Enforces each sumcheck_consistency_check == 0
/// 4. Outputs the final_claim
pub fn generate_stage1_circuit(
    result: &jolt_core::zkvm::stage1_only_verifier::Stage1TranscriptVerificationResult<
        zklean_extractor::mle_ast::MleAst,
    >,
    circuit_name: &str,
) -> String {
    // Collect all variables from all constraints
    let mut vars = BTreeSet::new();

    let final_claim_expr = generate_gnark_expr_with_vars(result.final_claim.root(), &mut vars);
    let power_sum_expr = generate_gnark_expr_with_vars(result.power_sum_check.root(), &mut vars);

    let consistency_exprs: Vec<String> = result
        .sumcheck_consistency_checks
        .iter()
        .map(|check| generate_gnark_expr_with_vars(check.root(), &mut vars))
        .collect();

    let mut output = String::new();

    // Package and imports
    output.push_str("package jolt_verifier\n\n");
    output.push_str("import \"github.com/consensys/gnark/frontend\"\n\n");

    // Circuit struct with all used variables
    output.push_str(&format!("type {} struct {{\n", circuit_name));
    for var_idx in &vars {
        output.push_str(&format!(
            "\tX_{} frontend.Variable `gnark:\",public\"`\n",
            var_idx
        ));
    }
    output.push_str("\tExpectedFinalClaim frontend.Variable `gnark:\",public\"`\n");
    output.push_str("}\n\n");

    // Define method
    output.push_str(&format!(
        "func (circuit *{}) Define(api frontend.API) error {{\n",
        circuit_name
    ));

    // Constraint 1: Power sum check == 0
    output.push_str("\t// Power sum check: sum over symmetric domain must equal 0\n");
    output.push_str(&format!("\tpowerSumCheck := {}\n", power_sum_expr));
    output.push_str("\tapi.AssertIsEqual(powerSumCheck, 0)\n\n");

    // Constraint 2: Each sumcheck consistency check == 0
    for (i, expr) in consistency_exprs.iter().enumerate() {
        output.push_str(&format!(
            "\t// Sumcheck round {}: poly(0) + poly(1) - claim == 0\n",
            i
        ));
        output.push_str(&format!("\tconsistencyCheck{} := {}\n", i, expr));
        output.push_str(&format!(
            "\tapi.AssertIsEqual(consistencyCheck{}, 0)\n\n",
            i
        ));
    }

    // Final claim
    output.push_str("\t// Final claim must match expected\n");
    output.push_str(&format!("\tfinalClaim := {}\n", final_claim_expr));
    output.push_str("\tapi.AssertIsEqual(finalClaim, circuit.ExpectedFinalClaim)\n\n");

    output.push_str("\treturn nil\n");
    output.push_str("}\n");

    output
}

/// Convert an Atom to Gnark code with CSE offset for NamedVar indices
fn atom_to_gnark_with_offset(atom: Atom, vars: &mut BTreeSet<u16>, cse_offset: usize) -> String {
    match atom {
        Atom::Scalar(value) => format_scalar_for_gnark(value),
        Atom::Var(index) => {
            vars.insert(index);
            format!("circuit.X_{}", index)
        }
        Atom::NamedVar(index) => format!("cse_{}", cse_offset + index),
    }
}

/// Convert an Edge to Gnark code with CSE offset
fn edge_to_gnark_with_offset(edge: Edge, vars: &mut BTreeSet<u16>, cse_offset: usize) -> String {
    match edge {
        Edge::Atom(atom) => atom_to_gnark_with_offset(atom, vars, cse_offset),
        Edge::NodeRef(node_id) => {
            generate_gnark_expr_with_vars_and_offset(node_id, vars, cse_offset)
        }
    }
}

/// Generate Gnark expression with CSE offset for NamedVar references
fn generate_gnark_expr_with_vars_and_offset(
    node_id: usize,
    vars: &mut BTreeSet<u16>,
    cse_offset: usize,
) -> String {
    let node = get_node(node_id);

    match node {
        Node::Atom(atom) => atom_to_gnark_with_offset(atom, vars, cse_offset),

        Node::Add(left, right) => {
            format!(
                "api.Add({}, {})",
                edge_to_gnark_with_offset(left, vars, cse_offset),
                edge_to_gnark_with_offset(right, vars, cse_offset)
            )
        }

        Node::Mul(left, right) => {
            format!(
                "api.Mul({}, {})",
                edge_to_gnark_with_offset(left, vars, cse_offset),
                edge_to_gnark_with_offset(right, vars, cse_offset)
            )
        }

        Node::Sub(left, right) => {
            format!(
                "api.Sub({}, {})",
                edge_to_gnark_with_offset(left, vars, cse_offset),
                edge_to_gnark_with_offset(right, vars, cse_offset)
            )
        }

        Node::Neg(child) => {
            format!(
                "api.Neg({})",
                edge_to_gnark_with_offset(child, vars, cse_offset)
            )
        }

        Node::Inv(child) => {
            format!(
                "api.Inverse({})",
                edge_to_gnark_with_offset(child, vars, cse_offset)
            )
        }

        Node::Div(left, right) => {
            format!(
                "api.Div({}, {})",
                edge_to_gnark_with_offset(left, vars, cse_offset),
                edge_to_gnark_with_offset(right, vars, cse_offset)
            )
        }

        Node::Poseidon(state, n_rounds, data) => {
            format!(
                "poseidon.Hash(api, {}, {}, {})",
                edge_to_gnark_with_offset(state, vars, cse_offset),
                edge_to_gnark_with_offset(n_rounds, vars, cse_offset),
                edge_to_gnark_with_offset(data, vars, cse_offset)
            )
        }

        Node::Keccak256(input) => {
            format!(
                "keccak.Keccak256(api, {})",
                edge_to_gnark_with_offset(input, vars, cse_offset)
            )
        }

        Node::ByteReverse(input) => {
            format!(
                "ByteReverse(api, {})",
                edge_to_gnark_with_offset(input, vars, cse_offset)
            )
        }

        Node::Truncate128Reverse(input) => {
            format!(
                "Truncate128Reverse(api, {})",
                edge_to_gnark_with_offset(input, vars, cse_offset)
            )
        }

        Node::Truncate128(input) => {
            format!(
                "Truncate128(api, {})",
                edge_to_gnark_with_offset(input, vars, cse_offset)
            )
        }

        Node::MulTwoPow192(input) => {
            format!(
                "AppendU64Transform(api, {})",
                edge_to_gnark_with_offset(input, vars, cse_offset)
            )
        }
    }
}

/// Generate Gnark expression for a node (used for CSE bindings) with offset.
/// This is similar to generate_gnark_expr but takes a Node directly instead of a node_id.
fn generate_gnark_expr_for_node_with_offset(
    node: &Node,
    vars: &mut BTreeSet<u16>,
    cse_offset: usize,
) -> String {
    match node {
        Node::Atom(atom) => atom_to_gnark_with_offset(*atom, vars, cse_offset),

        Node::Add(left, right) => {
            format!(
                "api.Add({}, {})",
                edge_to_gnark_with_offset(*left, vars, cse_offset),
                edge_to_gnark_with_offset(*right, vars, cse_offset)
            )
        }

        Node::Mul(left, right) => {
            format!(
                "api.Mul({}, {})",
                edge_to_gnark_with_offset(*left, vars, cse_offset),
                edge_to_gnark_with_offset(*right, vars, cse_offset)
            )
        }

        Node::Sub(left, right) => {
            format!(
                "api.Sub({}, {})",
                edge_to_gnark_with_offset(*left, vars, cse_offset),
                edge_to_gnark_with_offset(*right, vars, cse_offset)
            )
        }

        Node::Neg(child) => {
            format!(
                "api.Neg({})",
                edge_to_gnark_with_offset(*child, vars, cse_offset)
            )
        }

        Node::Inv(child) => {
            format!(
                "api.Inverse({})",
                edge_to_gnark_with_offset(*child, vars, cse_offset)
            )
        }

        Node::Div(left, right) => {
            format!(
                "api.Div({}, {})",
                edge_to_gnark_with_offset(*left, vars, cse_offset),
                edge_to_gnark_with_offset(*right, vars, cse_offset)
            )
        }

        Node::Poseidon(state, n_rounds, data) => {
            format!(
                "poseidon.Hash(api, {}, {}, {})",
                edge_to_gnark_with_offset(*state, vars, cse_offset),
                edge_to_gnark_with_offset(*n_rounds, vars, cse_offset),
                edge_to_gnark_with_offset(*data, vars, cse_offset)
            )
        }

        Node::Keccak256(input) => {
            format!(
                "keccak.Keccak256(api, {})",
                edge_to_gnark_with_offset(*input, vars, cse_offset)
            )
        }

        Node::ByteReverse(input) => {
            format!(
                "ByteReverse(api, {})",
                edge_to_gnark_with_offset(*input, vars, cse_offset)
            )
        }

        Node::Truncate128Reverse(input) => {
            format!(
                "Truncate128Reverse(api, {})",
                edge_to_gnark_with_offset(*input, vars, cse_offset)
            )
        }

        Node::Truncate128(input) => {
            format!(
                "Truncate128(api, {})",
                edge_to_gnark_with_offset(*input, vars, cse_offset)
            )
        }

        Node::MulTwoPow192(input) => {
            format!(
                "AppendU64Transform(api, {})",
                edge_to_gnark_with_offset(*input, vars, cse_offset)
            )
        }
    }
}

/// Apply CSE to a node and generate Gnark code with hoisted bindings.
///
/// The `cse_offset` parameter allows generating unique CSE variable names across
/// multiple calls (cse_0, cse_1, ... from first call, cse_N, cse_N+1, ... from second call).
///
/// Returns (bindings_code, final_expr, new_offset) where:
/// - bindings_code: Go variable assignments for hoisted subexpressions
/// - final_expr: The final expression using the hoisted variables
/// - new_offset: The next available CSE index
fn generate_gnark_expr_with_cse(
    node_id: usize,
    vars: &mut BTreeSet<u16>,
    cse_offset: usize,
) -> (String, String, usize) {
    let root_node = get_node(node_id);
    let (bindings, new_root) = common_subexpression_elimination(root_node);

    let mut bindings_code = String::new();

    // Generate code for each hoisted binding with offset indices
    for (i, binding_node) in bindings.iter().enumerate() {
        let binding_expr = generate_gnark_expr_for_node_with_offset(binding_node, vars, cse_offset);
        bindings_code.push_str(&format!("\tcse_{} := {}\n", cse_offset + i, binding_expr));
    }

    // Generate the final expression using the new root
    let new_root_id = insert_node(new_root);
    let final_expr = generate_gnark_expr_with_vars_and_offset(new_root_id, vars, cse_offset);

    let new_offset = cse_offset + bindings.len();
    (bindings_code, final_expr, new_offset)
}

/// Generate a complete Gnark circuit for Stage 1 verification with CSE optimization.
///
/// This is like generate_stage1_circuit but applies Common Subexpression Elimination
/// to reduce code size and potentially constraint count by hoisting repeated
/// subexpressions (especially nested Poseidon calls) into named variables.
pub fn generate_stage1_circuit_with_cse(
    result: &jolt_core::zkvm::stage1_only_verifier::Stage1TranscriptVerificationResult<
        zklean_extractor::mle_ast::MleAst,
    >,
    circuit_name: &str,
) -> String {
    let mut vars = BTreeSet::new();
    let mut all_bindings = String::new();
    let mut cse_offset = 0usize;

    // Apply CSE to each constraint and collect all bindings with global offset
    let (power_bindings, power_sum_expr, new_offset) =
        generate_gnark_expr_with_cse(result.power_sum_check.root(), &mut vars, cse_offset);
    all_bindings.push_str(&power_bindings);
    cse_offset = new_offset;

    let mut consistency_exprs = Vec::new();
    for check in &result.sumcheck_consistency_checks {
        let (bindings, expr, new_offset) =
            generate_gnark_expr_with_cse(check.root(), &mut vars, cse_offset);
        all_bindings.push_str(&bindings);
        consistency_exprs.push(expr);
        cse_offset = new_offset;
    }

    let (final_bindings, final_claim_expr, _) =
        generate_gnark_expr_with_cse(result.final_claim.root(), &mut vars, cse_offset);
    all_bindings.push_str(&final_bindings);

    let mut output = String::new();

    // Package and imports
    output.push_str("package jolt_verifier\n\n");
    output.push_str("import (\n");
    output.push_str("\t\"github.com/consensys/gnark/frontend\"\n");
    // Check if we use Poseidon
    if all_bindings.contains("poseidon.Hash") || final_claim_expr.contains("poseidon.Hash") {
        output.push_str("\t\"jolt_verifier/poseidon\"\n");
    }
    output.push_str(")\n\n");

    // Circuit struct with all used variables
    output.push_str(&format!("type {} struct {{\n", circuit_name));
    for var_idx in &vars {
        output.push_str(&format!(
            "\tX_{} frontend.Variable `gnark:\",public\"`\n",
            var_idx
        ));
    }
    output.push_str("\tExpectedFinalClaim frontend.Variable `gnark:\",public\"`\n");
    output.push_str("}\n\n");

    // Define method
    output.push_str(&format!(
        "func (circuit *{}) Define(api frontend.API) error {{\n",
        circuit_name
    ));

    // First, emit all CSE bindings
    if !all_bindings.is_empty() {
        output.push_str("\t// Common subexpressions (CSE optimization)\n");
        output.push_str(&all_bindings);
        output.push_str("\n");
    }

    // Constraint 1: Power sum check == 0
    output.push_str("\t// Power sum check: sum over symmetric domain must equal 0\n");
    output.push_str(&format!("\tpowerSumCheck := {}\n", power_sum_expr));
    output.push_str("\tapi.AssertIsEqual(powerSumCheck, 0)\n\n");

    // Constraint 2: Each sumcheck consistency check == 0
    for (i, expr) in consistency_exprs.iter().enumerate() {
        output.push_str(&format!(
            "\t// Sumcheck round {}: poly(0) + poly(1) - claim == 0\n",
            i
        ));
        output.push_str(&format!("\tconsistencyCheck{} := {}\n", i, expr));
        output.push_str(&format!(
            "\tapi.AssertIsEqual(consistencyCheck{}, 0)\n\n",
            i
        ));
    }

    // Final claim
    output.push_str("\t// Final claim must match expected\n");
    output.push_str(&format!("\tfinalClaim := {}\n", final_claim_expr));
    output.push_str("\tapi.AssertIsEqual(finalClaim, circuit.ExpectedFinalClaim)\n\n");

    output.push_str("\treturn nil\n");
    output.push_str("}\n");

    output
}

/// Generate a complete Gnark circuit for Stage 1 verification with GLOBAL CSE optimization.
///
/// Unlike `generate_stage1_circuit_with_cse`, this function uses a SHARED bindings
/// HashMap across ALL constraints. This means that if multiple constraints share
/// the same subexpression (e.g., Poseidon hash chains), it will only be hoisted ONCE.
///
/// This is the proper CSE implementation that produces smaller output than no CSE.
pub fn generate_stage1_circuit_with_global_cse(
    result: &jolt_core::zkvm::stage1_only_verifier::Stage1TranscriptVerificationResult<
        zklean_extractor::mle_ast::MleAst,
    >,
    circuit_name: &str,
) -> String {
    let mut vars = BTreeSet::new();

    // GLOBAL shared state for CSE - this is the key difference!
    let mut global_bindings: Bindings = HashMap::new();
    let mut global_nodes: Vec<Node> = Vec::new();

    // Apply CSE incrementally to each constraint, sharing bindings across all
    let power_root = get_node(result.power_sum_check.root());
    let power_new_root = common_subexpression_elimination_incremental(
        power_root,
        &mut global_bindings,
        &mut global_nodes,
    );
    let power_new_root_id = insert_node(power_new_root);
    let power_sum_expr = generate_gnark_expr_with_vars(power_new_root_id, &mut vars);

    let mut consistency_exprs = Vec::new();
    for check in &result.sumcheck_consistency_checks {
        let check_root = get_node(check.root());
        let check_new_root = common_subexpression_elimination_incremental(
            check_root,
            &mut global_bindings,
            &mut global_nodes,
        );
        let check_new_root_id = insert_node(check_new_root);
        let expr = generate_gnark_expr_with_vars(check_new_root_id, &mut vars);
        consistency_exprs.push(expr);
    }

    let final_root = get_node(result.final_claim.root());
    let final_new_root = common_subexpression_elimination_incremental(
        final_root,
        &mut global_bindings,
        &mut global_nodes,
    );
    let final_new_root_id = insert_node(final_new_root);
    let final_claim_expr = generate_gnark_expr_with_vars(final_new_root_id, &mut vars);

    // Generate CSE bindings code from the global_nodes
    let mut bindings_code = String::new();
    for (i, binding_node) in global_nodes.iter().enumerate() {
        let binding_expr = generate_gnark_expr_for_node_with_offset(binding_node, &mut vars, 0);
        bindings_code.push_str(&format!("\tcse_{} := {}\n", i, binding_expr));
    }

    let mut output = String::new();

    // Package and imports
    output.push_str("package jolt_verifier\n\n");
    output.push_str("import (\n");
    output.push_str("\t\"github.com/consensys/gnark/frontend\"\n");
    // Check if we use Poseidon
    if bindings_code.contains("poseidon.Hash") || final_claim_expr.contains("poseidon.Hash") {
        output.push_str("\t\"jolt_verifier/poseidon\"\n");
    }
    output.push_str(")\n\n");

    // Circuit struct with all used variables
    output.push_str(&format!("type {} struct {{\n", circuit_name));
    for var_idx in &vars {
        output.push_str(&format!(
            "\tX_{} frontend.Variable `gnark:\",public\"`\n",
            var_idx
        ));
    }
    output.push_str("\tExpectedFinalClaim frontend.Variable `gnark:\",public\"`\n");
    output.push_str("}\n\n");

    // Define method
    output.push_str(&format!(
        "func (circuit *{}) Define(api frontend.API) error {{\n",
        circuit_name
    ));

    // First, emit all CSE bindings (shared across all constraints!)
    if !bindings_code.is_empty() {
        output.push_str("\t// Common subexpressions (GLOBAL CSE optimization)\n");
        output.push_str(&bindings_code);
        output.push_str("\n");
    }

    // Constraint 1: Power sum check == 0
    output.push_str("\t// Power sum check: sum over symmetric domain must equal 0\n");
    output.push_str(&format!("\tpowerSumCheck := {}\n", power_sum_expr));
    output.push_str("\tapi.AssertIsEqual(powerSumCheck, 0)\n\n");

    // Constraint 2: Each sumcheck consistency check == 0
    for (i, expr) in consistency_exprs.iter().enumerate() {
        output.push_str(&format!(
            "\t// Sumcheck round {}: poly(0) + poly(1) - claim == 0\n",
            i
        ));
        output.push_str(&format!("\tconsistencyCheck{} := {}\n", i, expr));
        output.push_str(&format!(
            "\tapi.AssertIsEqual(consistencyCheck{}, 0)\n\n",
            i
        ));
    }

    // Final claim
    output.push_str("\t// Final claim must match expected\n");
    output.push_str(&format!("\tfinalClaim := {}\n", final_claim_expr));
    output.push_str("\tapi.AssertIsEqual(finalClaim, circuit.ExpectedFinalClaim)\n\n");

    output.push_str("\treturn nil\n");
    output.push_str("}\n");

    output
}

// Tests moved to tests/rust_to_gnark.rs
