# Gnark Transpiler

Transpiles Jolt verifier stages 1-6 (sumcheck verification) into Gnark circuits for Groth16 proving.

## Overview

This tool performs symbolic execution of the Jolt verifier to generate a Gnark circuit. Instead of computing with actual field elements, it uses `MleAst`, a symbolic type that records all arithmetic operations as an AST (Abstract Syntax Tree). This AST is then converted to Gnark/Go code.

### What Gets Transpiled

| Stage | Description | Included |
|-------|-------------|----------|
| 1-6 | Sumcheck verification | Yes |
| 7+ | PCS verification (Dory/Hyrax) | No |

PCS verification is **not transpiled** because:
- Dory uses pairings, which would add ~100M constraints if emulated inside BN254
- For a complete recursive verifier, PCS must be implemented natively in Gnark (see `quangvdao/quang-jolt` for Hyrax approach)

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
cargo run -p gnark-transpiler --release --features transcript-poseidon
```

This generates in `gnark-transpiler/go/`:
- `stages_circuit.go` - The Gnark circuit (~2M constraints)
- `stages_witness.json` - Witness values for proving
- `stages_bundle.json` - Serialized AST (for debugging)

### 3. Run Go Tests

```bash
cd gnark-transpiler/go

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
cargo run -p gnark-transpiler --features transcript-poseidon
```

If no transcript feature is specified, both default to Blake2b.

## CLI Options

```
gnark-transpiler [OPTIONS]

Options:
  --proof <PATH>          Path to proof file [default: /tmp/fib_proof.bin]
  --io-device <PATH>      Path to io_device file [default: /tmp/fib_io_device.bin]
  --preprocessing <PATH>  Path to preprocessing file [default: /tmp/jolt_verifier_preprocessing.dat]
  -o, --output-dir <DIR>  Output directory for Go files [default: go]
  -h, --help              Print help
  -V, --version           Print version
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
- ~2M constraints for stages 1-6

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
cargo run -p gnark-transpiler --features debug-expected-output,transcript-poseidon
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
| `codegen` | AST → Go code generation with CSE |
| `symbolic_proof` | Convert concrete proofs to symbolic form |
| `poseidon` | Poseidon transcript for symbolic Fiat-Shamir |
| `mle_opening_accumulator` | Symbolic opening accumulator |
| `ast_commitment_scheme` | Stub commitment scheme for transpilation |
