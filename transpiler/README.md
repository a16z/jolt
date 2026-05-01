# Transpiler

Transpiles Jolt verifier stages 1-7 (all sumcheck verification) into a target-agnostic AST, with Gnark/Groth16 code generation as the primary backend.

## Overview

This tool performs **symbolic execution** of the Jolt verifier. Instead of computing with actual field elements, it uses `MleAst`, a symbolic type that records all arithmetic operations as an AST (Abstract Syntax Tree). This produces an `AstBundle`, a serializable intermediate representation that can be:

1. **Analyzed directly**: constraint structure, dependency graphs, optimization passes
2. **Transformed to target code**: currently Gnark/Go, but the IR is target-agnostic

### How It Works

The Rust verifier is executed normally, but with a "recording" field type:

```
Rust verifier (MleAst) → AST recording → AstBundle (IR) → [optional] Target code
```

This captures the exact verification logic without manual circuit rewriting.

### Supported Targets

| Target | Output | Use Case |
|--------|--------|----------|
| `ast-bundle` | JSON IR (`AstBundle`) | Analysis, debugging, custom backends |
| `gnark` (default) | Go circuit + witness | Groth16 proving |

The `AstBundle` is always produced internally; the `--target` flag controls whether to stop there or continue to code generation.

### What Gets Transpiled

Stages 1-7 (all sumcheck verification, including AdviceClaimReduction). Stage 8 (PCS/Dory) is not transpiled because pairing operations are too expensive in-circuit.

## Quick Start

### Full E2E Pipeline (single command)

The easiest way to run the complete pipeline is with the E2E test:

```bash
cd transpiler/go && go test -v -run TestEndToEndPipeline -timeout 30m
```

This automatically:
1. Builds the Rust binaries (fibonacci, transpiler)
2. Generates a Jolt proof with Poseidon transcript
3. Transpiles to Gnark circuit
4. Runs Groth16 setup, prove, and verify
5. Reports detailed timing breakdown for each step

### Manual Pipeline (3 commands)

```bash
# 1. Generate a Jolt proof (MUST use Poseidon transcript)
cargo run -p fibonacci --release --features transcript-poseidon -- --save 50

# 2. Transpile to Gnark circuit
cargo run -p transpiler --release --features transcript-poseidon

# 3. Run Groth16 prove/verify
cd transpiler/go && go test -v -run TestStagesCircuitProveVerify
```

## Transcript Feature Flags

The transpiler must use the **same transcript** as proof generation:

| Feature Flag | Hash Function | SNARK-Friendly |
|--------------|---------------|----------------|
| `transcript-poseidon` | Poseidon | Yes |
| `transcript-keccak` | Keccak | No |
| `transcript-blake2b` | Blake2b | No |

Only Poseidon-generated proofs can be efficiently verified in-circuit. The other transcripts can still be used for IR analysis or if circuit size is not a concern. If no transcript feature is specified, both default to Blake2b.

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

The `ast-bundle` target outputs the intermediate representation without generating target-specific code. Useful for developing new backends, analyzing constraint structure, or debugging.

```bash
cargo run -p transpiler --features transcript-poseidon -- -t ast-bundle -o /path/to/output
```

The bundle is a self-contained JSON file that can be processed by any tool.

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
    ▼ AstBundle::new() + snapshot_arena()
AstBundle (target-agnostic IR)
    │
    ├─► [--target ast-bundle] stages_bundle.json (stop here for analysis)
    │
    ▼ [--target gnark] generate_circuit_from_bundle()
stages_circuit.go (Gnark circuit)
    │
    ▼ groth16.Prove()
Groth16 Proof (164 bytes)
```

The `AstBundle` contains all symbolic variables (inputs), equality assertions (constraints), and the complete AST node graph (arena).

## Output Files

| File | Description |
|------|-------------|
| `stages_circuit.go` | Generated Gnark circuit |
| `stages_witness.json` | Witness values (variable name → field element) |
| `stages_bundle.json` | Serialized AST bundle (inputs, constraints, arena) |

## Module Overview

| Module | Description |
|--------|-------------|
| `gnark_codegen` | AST → Go/gnark code generation with CSE |
| `symbolic_proof` | Convert concrete proofs to symbolic form |
| `symbolize` | IO device symbolization for universal circuits |
| `symbolic_traits` | Trait implementations for symbolic execution |

## Adding a New Transpilation Target

To add a new target (e.g., Circom, Plonky2):

1. Create a codegen module `src/<target>_codegen.rs`. Use `gnark_codegen.rs` as reference. It should take an `AstBundle` and emit target-specific code.

2. Add the target variant to `TranspilationTarget` enum in `src/main.rs`.

3. Add a match arm for your target in the code generation section (search for `match args.target`).

4. Export from `lib.rs` if needed.

The `AstBundle` is target-agnostic, so new backends only need to implement AST traversal and code emission.
