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

### Full E2E Pipeline (3 commands)

```bash
# 1. Generate a Jolt proof (MUST use Poseidon transcript)
cargo run -p fibonacci --release --features transcript-poseidon -- --save 50

# 2. Transpile to Gnark circuit
cargo run -p transpiler --release --features transcript-poseidon

# 3. Run Groth16 prove/verify
cd transpiler/go && go test -v -run TestStagesCircuitProveVerify
```

### Step-by-Step Details

#### 1. Generate a Proof

First, generate a Jolt proof with the Poseidon transcript (required for SNARK-friendly verification):

```bash
cd /path/to/jolt
cargo run -p fibonacci --release --features transcript-poseidon -- --save 50
```

**Important**: The `--features transcript-poseidon` flag is **required**. Without it, the proof uses Blake2b which cannot be efficiently verified in-circuit.

This saves:
- `/tmp/fib_proof.bin` - The JoltProof
- `/tmp/fib_io_device.bin` - Program inputs/outputs
- `/tmp/jolt_verifier_preprocessing.dat` - Verifier setup data

#### 2. Transpile to Gnark

```bash
cargo run -p transpiler --release --features transcript-poseidon
```

This generates in `transpiler/go/`:
- `stages_circuit.go` - The Gnark circuit (~2.86M constraints)
- `stages_witness.json` - Witness values for proving
- `stages_bundle.json` - Serialized AST (for debugging)

#### 3. Run Go Tests

```bash
cd transpiler/go

# Quick solver test (~1s)
go test -v -run TestStagesCircuitSolver

# Full Groth16 prove/verify (~100s)
go test -v -run TestStagesCircuitProveVerify
```

### Expected Results (fib(50), Stages 1-7)

| Metric | Value |
|--------|-------|
| Constraints | ~2.86M |
| Assertions | 16 |
| Proof size | 164 bytes |
| Prove time | ~11s |
| Verify time | ~3ms |

#### Optimization History

The transpilation pipeline has undergone significant optimizations:

| Metric | Baseline | Phase 1 (byte-reverse removal) | Phase 1+2 (+ truncation) |
|--------|----------|--------------------------------|--------------------------|
| **Constraints** | **5,113,331** | **3,234,881 (-36.7%)** | **2,862,236 (-44.0%)** |
| Circuit compile | 16.6s | 11.4s | 8.0s |
| Groth16 setup | 3m35s | 2m19s | 2m18s |
| Groth16 prove | 15.9s | 11.1s | 11.0s |
| PK size | 835 MB | 520 MB | 483 MB |

The optimizations removed unnecessary EVM-compatible byte operations from the Poseidon transcript. See the [Poseidon Optimization](#poseidon-optimization) section for details.

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
- ~2.86M constraints for stages 1-7

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

## Poseidon Optimization

The current implementation uses a **native field-arithmetic Poseidon transcript** that operates directly on BN254 field elements. This is a significant optimization over the original EVM-compatible encoding.

### Background

The original Poseidon transcript inherited byte-level serialization from Blake2b/Keccak transcripts, which were designed for direct Solidity verification. This included:
- `ByteReverse`: Reverse bytes of field elements for EVM compatibility (~255 constraints each)
- `AppendU64Transform`: Byte-swap and shift u64 values (~65 constraints each)
- `Truncate128`/`Truncate128Reverse`: Extract 128-bit challenges with byte manipulation (~255 constraints each)

These operations were meaningful for Blake2b/Keccak (which operate on bytes) but wasteful for Poseidon (which operates on field elements natively).

### Why EVM Encoding Was Unnecessary

1. **Poseidon's input domain is field elements**, not bytes. Byte manipulations require expensive bit decomposition in R1CS.
2. **The Groth16 Solidity verifier never sees the transcript**. It only checks the pairing equation `e(A,B) = e(alpha,beta) * ...`. All Fiat-Shamir operations are private witness computations.
3. **The encoding served no cryptographic purpose**. Any deterministic, injective encoding is equally valid for Fiat-Shamir security.

### Current Implementation

The optimized transcript:
- Appends scalars directly as field elements (no byte reversal)
- Appends u64 values as `Fr::from(x)` (no byte-swap + shift)
- Uses native 128-bit truncation (no byte reversal in challenge derivation)

This achieved a **44% constraint reduction** (5.1M → 2.86M) with corresponding improvements in proving time and memory usage.

For the full analysis, see `.claude/theory/evm-byte-ops-constraint-analysis.md`.

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
| `symbolic_traits/poseidon` | `PoseidonAstTranscript` - Poseidon transcript for symbolic Fiat-Shamir |
| `symbolic_traits/opening_accumulator` | `AstOpeningAccumulator` - Symbolic opening accumulator |
| `symbolic_traits/ast_commitment_scheme` | `AstCommitmentScheme` - Stub commitment scheme for transpilation |

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
