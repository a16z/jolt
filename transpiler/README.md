# Transpiler

Transpiles Jolt verifier stages 1-7 (all sumcheck verification) into circuit code for various proving backends.

## Overview

This tool performs symbolic execution of the Jolt verifier to generate circuit code. Instead of computing with actual field elements, it uses `MleAst`, a symbolic type that records all arithmetic operations as an AST (Abstract Syntax Tree). This AST is then converted to target-specific code (e.g., Go for gnark).

### Supported Targets

| Target | Language | Proving System |
|--------|----------|----------------|
| `gnark` (default) | Go | Groth16 |
| `ast-bundle` | JSON | None (IR only) |

### What Gets Transpiled

| Stage | Description | Included |
|-------|-------------|----------|
| 1-6 | Standard sumcheck verification | Yes |
| 7 | HammingWeight claim reduction | Yes |
| 8 | PCS verification (Hyrax) | No |

Stage 8 (PCS verification) is **not transpiled** because:
- Hyrax requires native elliptic curve operations (MSM on Grumpkin)
- For a complete recursive verifier, PCS must be implemented natively in Gnark (see `quangvdao/quang-jolt` for Hyrax approach)

**Note**: Stage 7 does not include `AdviceClaimReduction` verifiers. They require state management across stages 6-7. See `.claude/tasks/006-add-advice-verifiers-stage7.md` for planned implementation.

## Quick Start

### 1. Generate a Proof

First, generate a Jolt proof with the Poseidon transcript (required for SNARK-friendly verification):

```bash
cd /path/to/jolt
cargo run -p fibonacci --release --features transcript-poseidon -- --save 50
```

This saves:
- `/tmp/fib_proof.bin` - The JoltProof
- `/tmp/fib_io_device.bin` - Program inputs/outputs
- `/tmp/jolt_verifier_preprocessing.dat` - Verifier setup data

### 2. Transpile to Gnark

```bash
cargo run -p transpiler --release --features transcript-poseidon
```

This generates in `transpiler/go/`:
- `stages_circuit.go` - The Gnark circuit (~2M constraints)
- `stages_witness.json` - Witness values for proving
- `stages_bundle.json` - Serialized AST (for debugging)

### 3. Run Go Tests

```bash
cd transpiler/go

# Quick solver test (~1s)
go test -v -run TestStagesCircuitSolver

# Full Groth16 prove/verify (~6s)
go test -v -run TestStagesCircuitProveVerify
```

## Transcript Feature Flags

The transpiler must use the **same transcript** as proof generation:

| Feature Flag | Hash Function | SNARK-Friendly |
|--------------|---------------|----------------|
| `transcript-poseidon` | Poseidon | Yes (~250 constraints/hash) |
| `transcript-keccak` | Keccak | No (~150k constraints/hash) |
| `transcript-blake2b` | Blake2b | No (~150k constraints/hash) |

**Important**: Only Poseidon-generated proofs can be efficiently verified in-circuit. The other transcripts are provided for compatibility, but the generated circuit would be infeasibly large.

```bash
# Generate proof with Poseidon
cargo run -p fibonacci --features transcript-poseidon -- --save 50

# Transpile with matching feature
cargo run -p transpiler --features transcript-poseidon
```

If no transcript feature is specified, both default to Blake2b.

## CLI Options

```
transpiler [OPTIONS]

Options:
  --proof <PATH>          Path to proof file [default: /tmp/fib_proof.bin]
  --io-device <PATH>      Path to io_device file [default: /tmp/fib_io_device.bin]
  --preprocessing <PATH>  Path to preprocessing file [default: /tmp/jolt_verifier_preprocessing.dat]
  -t, --target <TARGET>   Transpilation target [default: gnark] [possible values: gnark, ast-bundle]
  -o, --output-dir <DIR>  Output directory (defaults to target-specific directory)
  -h, --help              Print help
  -V, --version           Print version
```

## Using `ast-bundle` Target

The `ast-bundle` target outputs only the intermediate representation (AstBundle) without generating target-specific code. This is useful for:
- Developing new code generation backends
- Debugging the symbolic execution output
- Analyzing the constraint structure

```bash
# Output bundle to current directory
cargo run -p transpiler --features transcript-poseidon -- -t ast-bundle

# Output bundle to specific directory
cargo run -p transpiler --features transcript-poseidon -- -t ast-bundle -o /path/to/output
```

## Architecture

```
JoltProof (concrete Fr values)
    │
    ▼ symbolize_proof()
JoltProof<MleAst> (symbolic variables)
    │
    ▼ TranspilableVerifier::verify()
AST in NODE_ARENA (recorded operations)
    │
    ├─► [ast-bundle] stages_bundle.json (stop here)
    │
    ▼ generate_circuit_from_bundle()
stages_circuit.go (Gnark circuit)
    │
    ▼ groth16.Prove()
Groth16 Proof (164 bytes)
```

## Output Files

### `stages_circuit.go`

Generated Gnark circuit with:
- `JoltStagesCircuit` struct containing all witness fields
- `Define()` method with constraint logic
- ~3M constraints for stages 1-7

### `stages_witness.json`

JSON mapping variable names to their concrete values:
```json
{
  "commitment_0_0": "123456...",
  "stage1_sumcheck_r0_0": "789...",
  ...
}
```

### `stages_bundle.json`

Serialized AST bundle for debugging. Contains:
- All inputs (symbolic variables)
- All constraints (equality assertions)
- Arena snapshot of AST nodes

## Debug Mode

To output intermediate values during proof generation (for debugging transcript mismatches):

```bash
cargo run -p fibonacci --features debug-expected-output,transcript-poseidon -- 10
cargo run -p transpiler --features debug-expected-output,transcript-poseidon
```

## Common Issues

### Transcript Mismatch

**Symptom**: Circuit constraints fail, all assertions are non-zero.

**Cause**: Proof was generated with a different transcript than the transpiler expects.

**Fix**: Ensure both use the same `--features transcript-*` flag.

### Missing Proof Files

**Symptom**: `Failed to read proof file`

**Fix**: Run the example program with `--save` first:
```bash
cargo run -p fibonacci --features transcript-poseidon -- --save 50
```

## Module Overview

| Module | Description |
|--------|-------------|
| `gnark_codegen` | AST → Go/gnark code generation with CSE |
| `symbolic_proof` | Convert concrete proofs to symbolic form |
| `poseidon` | Poseidon transcript for symbolic Fiat-Shamir |
| `mle_opening_accumulator` | Symbolic opening accumulator |
| `ast_commitment_scheme` | Stub commitment scheme for transpilation |

## Adding a New Transpilation Target

To add a new target (e.g., Circom, Plonky2):

1. **Create a codegen module** - Add `src/<target>_codegen.rs` with your code generation logic. Use `gnark_codegen.rs` as a reference. Your module should:
   - Take an `AstBundle` as input
   - Traverse the AST nodes and emit target-specific code
   - Handle CSE (Common Subexpression Elimination) appropriately

2. **Add the target variant** - In `src/main.rs`, add to `TranspilationTarget`:
   ```rust
   pub enum TranspilationTarget {
       Gnark,
       AstBundle,
       YourTarget,  // Add here
   }
   ```

3. **Set the default output directory** - Update the match in Step 6:
   ```rust
   let default_output_dir = match args.target {
       TranspilationTarget::Gnark => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("go"),
       TranspilationTarget::AstBundle => PathBuf::from("."),
       TranspilationTarget::YourTarget => PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("your_dir"),
   };
   ```

4. **Add the codegen match arm** - In Step 7, add your target's code generation:
   ```rust
   match args.target {
       TranspilationTarget::Gnark => { /* ... */ }
       TranspilationTarget::AstBundle => { /* ... */ }
       TranspilationTarget::YourTarget => {
           // Generate circuit code
           let circuit_code = your_codegen::generate(&bundle);
           // Write files...
       }
   }
   ```

5. **Export from lib.rs** - Add `pub mod your_codegen;` and any public re-exports.

### Design Note

We use a simple match statement rather than a trait for target dispatch. This avoids premature abstraction - different targets may need different inputs or have different output requirements. If patterns emerge across multiple targets, we can extract a trait later.
