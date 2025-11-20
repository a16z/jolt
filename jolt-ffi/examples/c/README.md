# Jolt FFI C Example

Complete working example of using the Jolt FFI from C.

## Quick Start

```bash
./prepare_and_run.sh fibonacci-guest
```

That's it! This script will:
1. Build the guest program as a RISC-V ELF
2. Generate preprocessing data
3. Build the C FFI library
4. Compile this example
5. Optionally run the proof generation

## Files

- `example.c` - Complete C program demonstrating the FFI API
- `Makefile` - Build system for the C example
- `prepare_and_run.sh` - Helper script that automates everything

## What the Example Does

The C program:
1. Loads preprocessing from a file
2. Reads an ELF binary
3. Creates a Jolt prover
4. Generates a zero-knowledge proof
5. Saves the proof to a file

## Full Documentation

See the [main Jolt FFI README](../../README.md) for:
- Complete API reference
- Manual workflow steps
- Linking instructions for your projects
- Troubleshooting guide
- Platform support details

## Try Other Guest Programs

```bash
./prepare_and_run.sh sha2-guest
./prepare_and_run.sh merkle-tree-guest
./prepare_and_run.sh collatz-guest
```

See the [main README](../../README.md#available-guest-programs) for the full list.
