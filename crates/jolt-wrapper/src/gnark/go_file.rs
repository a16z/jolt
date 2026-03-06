//! Complete Go circuit file generation from an [`AstBundle`].
//!
//! Produces a standalone `.go` file containing a gnark circuit struct with
//! public input fields and a `Define` method implementing the constraint logic.

use std::fmt::Write;

use crate::bundle::{AstBundle, InputVar};

use super::ast_emitter::GnarkAstEmitter;
use super::emitter::sanitize_go_name;

/// Configuration for Go file generation.
pub struct GoFileConfig {
    /// Go package name (e.g., `"circuit"`).
    pub package_name: String,
    /// Circuit struct name (e.g., `"JoltVerifierCircuit"`).
    pub circuit_name: String,
}

impl Default for GoFileConfig {
    fn default() -> Self {
        Self {
            package_name: "circuit".into(),
            circuit_name: "JoltVerifierCircuit".into(),
        }
    }
}

/// Generates a complete Go circuit file from an `AstBundle`.
pub fn generate_go_file(bundle: &AstBundle, config: &GoFileConfig) -> String {
    let mut out = String::with_capacity(4096);

    // Package declaration
    let _ = writeln!(out, "package {}\n", config.package_name);

    // Imports
    out.push_str("import (\n");
    out.push_str("\t\"math/big\"\n\n");
    out.push_str("\t\"github.com/consensys/gnark/frontend\"\n");
    out.push_str(")\n\n");

    // bigInt helper
    out.push_str("func bigInt(s string) *big.Int {\n");
    out.push_str("\tn, _ := new(big.Int).SetString(s, 10)\n");
    out.push_str("\treturn n\n");
    out.push_str("}\n\n");

    // Circuit struct
    let _ = writeln!(out, "type {} struct {{", config.circuit_name);
    for input in &bundle.inputs {
        let field_name = sanitize_go_name(&input.name);
        let _ = writeln!(out, "\t{field_name} frontend.Variable `gnark:\",public\"`");
    }
    out.push_str("}\n\n");

    // Define method
    let _ = writeln!(
        out,
        "func (circuit {}) Define(api frontend.API) error {{",
        config.circuit_name
    );

    // Emit constraints
    let mut emitter = build_emitter(&bundle.inputs);
    bundle.emit(&mut emitter);

    // Indent each line
    let code = emitter.finish();
    for line in code.lines() {
        let _ = writeln!(out, "\t{line}");
    }

    out.push_str("\treturn nil\n");
    out.push_str("}\n");

    out
}

/// Builds a `GnarkAstEmitter` with variable name mappings from the bundle inputs.
fn build_emitter(inputs: &[InputVar]) -> GnarkAstEmitter {
    let mut emitter = GnarkAstEmitter::new();
    for input in inputs {
        emitter = emitter.with_var_name(input.index, sanitize_go_name(&input.name));
    }
    emitter
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::ArenaSession;
    use crate::bundle::VarAllocator;
    use crate::symbolic::SymbolicField;

    #[test]
    fn generates_valid_go_structure() {
        let _session = ArenaSession::new();

        let x = SymbolicField::variable(0, "x_val");
        let y = SymbolicField::variable(1, "y_val");
        let result = x * y;

        let mut allocator = VarAllocator::new();
        let _ = allocator.input("x_val");
        let _ = allocator.input("y_val");
        allocator.assert_zero(result.into_edge());

        let bundle = allocator.finish();
        let config = GoFileConfig::default();
        let code = generate_go_file(&bundle, &config);

        assert!(code.contains("package circuit"));
        assert!(code.contains("type JoltVerifierCircuit struct {"));
        assert!(
            code.contains("func (circuit JoltVerifierCircuit) Define(api frontend.API) error {")
        );
        assert!(code.contains("return nil"));
        assert!(code.contains("bigInt"));
        assert!(code.contains("frontend.Variable"));
    }

    #[test]
    fn input_fields_are_public() {
        let _session = ArenaSession::new();

        let x = SymbolicField::variable(0, "my_input");
        let mut allocator = VarAllocator::new();
        let _ = allocator.input("my_input");
        allocator.assert_zero(x.into_edge());

        let bundle = allocator.finish();
        let code = generate_go_file(&bundle, &GoFileConfig::default());

        assert!(code.contains("`gnark:\",public\"`"));
        assert!(code.contains("My_Input"));
    }

    #[test]
    fn custom_config() {
        let _session = ArenaSession::new();

        let x = SymbolicField::variable(0, "val");
        let mut allocator = VarAllocator::new();
        let _ = allocator.input("val");
        allocator.assert_zero(x.into_edge());

        let bundle = allocator.finish();
        let config = GoFileConfig {
            package_name: "verifier".into(),
            circuit_name: "MyCircuit".into(),
        };
        let code = generate_go_file(&bundle, &config);

        assert!(code.contains("package verifier"));
        assert!(code.contains("type MyCircuit struct {"));
    }
}
