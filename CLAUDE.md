# CLAUDE.md

## Project Overview

Jolt is a zkVM (zero-knowledge virtual machine) for RISC-V (RV64IMAC) that efficiently proves and verifies program execution. It uses advanced cryptographic techniques including sumcheck, multilinear polynomial commitments, and lookups.

## Essential Commands


### Linting and Formatting
```bash
# Check code style (use --message-format=short)
# This is your main validation step
cargo clippy --all --message-format=short -q --all-targets --features allocative,host -- -D warnings

# Format code
cargo fmt -q
```

### Testing
```bash
# CRITICAL: Never use cargo test. Always use cargo nextest
cargo nextest run --cargo-quiet

# Run specific test in specific package
cargo nextest run -p [package_name] [test_name] --cargo-quiet

# CRITICAL: Generally you should ONLY run e2e muldiv test to verify correctness
cargo nextest run -p jolt-core muldiv --cargo-quiet
```

### Building
```bash
# CRITICAL: Build only when you are preparing to execute binary. Use clippy otherwise
# Build specific package
cargo build -p jolt-core --message-format=short -q
```

## Architecture

### Core Components

**jolt-core**
- `host/`: Compilation of guest program
- `zkvm/`: Implements the Jolt PIOP, including all sumchecks
- `poly/`: Polynomial operations and commitments
- `field/`: Finite field trait and implementations
- `r1cs/`: R1CS constraint system
- `utils/`: Parallel processing and utilities

**tracer**
- RISC-V emulator that generates execution traces
- Supports RV64IMAC instruction set
- Handles memory operations and syscalls

**jolt-sdk**
- SDK with `#[jolt::provable]` macro for ergonomic guest program development

**jolt-inlines**
- Jolt-optimized implementations of common cryptographic primitives (e.g. sha2, blake3, bigint, secp256k1)
- Similar to "precompiles" in other zkVMs

**examples**
- Guest/host example programs showcasing Jolt SDK (fibonacci, collatz, etc.)
- Each example can be run as a standalone package


## Development Guidelines

### Performance Requirements
- PERFORMANCE IS CRITICAL AND TOP PRIORITY
- No shortcuts in implementation
- Use idiomatic Rust patterns
- Profile before optimizing

### Code Style
- Use `cargo fmt` for formatting
- Pass `cargo clippy` with no warnings
- Use `#[inline]` judiciously in hot paths

### Comment Policy

**Delete these comment types:**
- Section separators (`// ==========`, `// ----------`)
- Doc comments that restate the item name (`/// Sumcheck prover for X` on `XProver`)
- Obvious comments (`/// Returns the count` on `get_count()`)
- Commented-out code
- TODOs without issue links

**Keep these comment types:**
- WHY something is done (when not obvious)
- WARNING comments for non-obvious gotchas
- SAFETY comments for unsafe blocks
- Complex algorithm explanations (link to paper if applicable)
- Public API docs that explain behavior, constraints, or invariants

**Principle:** Code should be self-documenting. If you need a comment to explain WHAT code does, refactor to make it clearer.

### Testing Strategy
- Always use `cargo nextest` (never `cargo test`)
- Write unit tests for new functionality

### Memory and Allocation
- Pre-allocate vectors unsafely when size is known
- Avoid unnecessary clones in hot paths
