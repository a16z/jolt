# Bytecode Preprocessing Refactor Design

## Goal

Separate bytecode preprocessing between prover and verifier based on `BytecodeMode`:

- **Full mode**: Verifier has access to full bytecode (O(K) data) — current behavior
- **Committed mode**: Verifier only sees bytecode commitments — enables succinct verification

## Current State (After Refactor)

```
BytecodePreprocessing  ← O(K) data, created first via preprocess()
├── bytecode: Vec<Instruction>
└── pc_map: BytecodePCMapper

JoltSharedPreprocessing  ← Truly shared, single source of truth for size
├── bytecode_size: usize            ← Derived from bytecode.bytecode.len()
├── ram: RAMPreprocessing
├── memory_layout: MemoryLayout
└── max_padded_trace_length: usize

JoltProverPreprocessing  ← Prover always has full bytecode
├── generators: PCS::ProverSetup
├── shared: JoltSharedPreprocessing
├── bytecode: Arc<BytecodePreprocessing>        ← Full bytecode (always)
├── bytecode_commitments: Option<TrustedBytecodeCommitments<PCS>>  ← Only in Committed mode
└── bytecode_commitment_hints: Option<Vec<PCS::OpeningProofHint>>  ← Only in Committed mode

JoltVerifierPreprocessing  ← Verifier has mode-dependent bytecode
├── generators: PCS::VerifierSetup
├── shared: JoltSharedPreprocessing
└── bytecode: VerifierBytecode<PCS>        ← Full OR Committed

VerifierBytecode<PCS>  ← Mode-dependent bytecode info
├── Full(Arc<BytecodePreprocessing>)              ← For Full mode
└── Committed(TrustedBytecodeCommitments<PCS>)    ← For Committed mode
```

---

## The Trace-Like Pattern

Bytecode preprocessing follows the same pattern as trace:

```rust
// Trace pattern:
let trace: Arc<Vec<Cycle>> = trace.into();

// Bytecode pattern (parallel):
let bytecode: Arc<BytecodePreprocessing> = BytecodePreprocessing::preprocess(instructions).into();
```

Both use `Arc` for cheap cloning (`Arc::clone` is O(1) reference count increment).

---

## Usage Examples

### E2E Flow (Full Mode)

```rust
// 1. Decode + preprocess bytecode (returns Self, wrap in Arc)
let (instructions, memory_init, _) = program.decode();
let bytecode: Arc<BytecodePreprocessing> = BytecodePreprocessing::preprocess(instructions).into();

// 2. Create shared preprocessing (borrows bytecode to get size)
let shared = JoltSharedPreprocessing::new(
    &bytecode,
    memory_layout,
    memory_init,
    max_trace_length,
);

// 3. Prover (Arc::clone is O(1))
let prover_pp = JoltProverPreprocessing::new(shared.clone(), Arc::clone(&bytecode));

// 4. Verifier (Full mode)
let verifier_pp = JoltVerifierPreprocessing::new_full(shared, generators, bytecode);
```

### E2E Flow (Committed Mode)

```rust
// 1-2. Same as above...
let bytecode: Arc<BytecodePreprocessing> = BytecodePreprocessing::preprocess(instructions).into();
let shared = JoltSharedPreprocessing::new(&bytecode, memory_layout, memory_init, max_trace);

// 3. Prover in Committed mode (computes commitments during preprocessing)
let prover_pp = JoltProverPreprocessing::new_committed(shared.clone(), Arc::clone(&bytecode));

// 4. Verifier receives only commitments (from prover's preprocessing)
let verifier_pp = JoltVerifierPreprocessing::new_committed(
    shared,
    generators,
    prover_pp.bytecode_commitments.clone().unwrap(),
);
```

### Accessing Bytecode Data

```rust
// Access bytecode size (always from shared - single source of truth)
let code_size = prover_pp.shared.bytecode_size;   // ✅ Definitive source
let code_size = verifier_pp.shared.bytecode_size; // ✅ Same

// Access full bytecode (prover only, or verifier in Full mode)
let bytecode_data = &prover_pp.bytecode;                              // Arc<BytecodePreprocessing>
let bytecode_data = verifier_pp.bytecode.as_full()?;                  // Result<&Arc<...>, ProofVerifyError>
let commitments = verifier_pp.bytecode.as_committed()?;               // Result<&TrustedBytecodeCommitments<PCS>, ProofVerifyError>
```

---

## SDK Macro Changes

The generated preprocessing functions now follow the trace-like pattern:

```rust
// Old API (deprecated)
pub fn preprocess_shared_foo(program: &mut Program) -> JoltSharedPreprocessing

// New API
pub fn preprocess_shared_foo(program: &mut Program) 
    -> (JoltSharedPreprocessing, Arc<BytecodePreprocessing>)

pub fn preprocess_prover_foo(
    shared: JoltSharedPreprocessing,
    bytecode: Arc<BytecodePreprocessing>,
) -> JoltProverPreprocessing<F, PCS>

pub fn preprocess_verifier_foo(
    shared: JoltSharedPreprocessing,
    generators: PCS::VerifierSetup,
    bytecode: Arc<BytecodePreprocessing>,  // For Full mode
) -> JoltVerifierPreprocessing<F, PCS>
```

---

## Key Design Decisions

1. **`BytecodePreprocessing::preprocess()` returns `Self`** (not `Arc<Self>`)
   - Caller uses `.into()` to wrap in Arc, just like trace

2. **`JoltSharedPreprocessing::new()` takes `&BytecodePreprocessing`**
   - Borrows to compute `bytecode_size = bytecode.bytecode.len()`
   - Returns just `Self`, not a tuple

3. **`bytecode_size` is the single source of truth**
   - Stored in `JoltSharedPreprocessing`
   - `BytecodePreprocessing` has no size field

4. **`TrustedBytecodeCommitments<PCS>`** wrapper enforces trust model
   - Type-level guarantee that commitments came from honest preprocessing
   - Public `commitments: Vec<PCS::Commitment>` field for simplicity

5. **No panics in `VerifierBytecode::as_full()` / `as_committed()`**
   - Returns `Result<_, ProofVerifyError>` with `BytecodeTypeMismatch` error

---

## Files Modified

| File | Changes |
|------|---------|
| `jolt-core/src/zkvm/bytecode/mod.rs` | `preprocess()` returns `Self`, added `VerifierBytecode<PCS>`, `TrustedBytecodeCommitments<PCS>` |
| `jolt-core/src/zkvm/prover.rs` | Added `bytecode`, `bytecode_commitments`, `bytecode_commitment_hints` fields |
| `jolt-core/src/zkvm/verifier.rs` | `new()` takes `&BytecodePreprocessing`, added `bytecode_size`, removed `bytecode` |
| `jolt-core/src/guest/prover.rs` | Updated to new pattern |
| `jolt-core/src/guest/verifier.rs` | Updated to new pattern |
| `jolt-sdk/macros/src/lib.rs` | Updated generated code for new API |
| `jolt-sdk/src/host_utils.rs` | Added `BytecodePreprocessing` export |
| `jolt-core/benches/e2e_profiling.rs` | Updated to new pattern |

---

## Verification

- ✅ `cargo fmt` clean
- ✅ `cargo clippy -p jolt-core --tests -- -D warnings` passes
- ✅ `cargo clippy -p jolt-sdk --benches -- -D warnings` passes

---

## Status

**Refactor Complete** — Structure for Full and Committed modes is in place.

### What's Done
- Bytecode preprocessing separated from shared preprocessing
- `Arc<BytecodePreprocessing>` pattern (like trace)
- `JoltSharedPreprocessing.bytecode_size` as single source of truth
- `VerifierBytecode<PCS>` enum for mode-dependent bytecode
- `TrustedBytecodeCommitments<PCS>` wrapper for type-safe commitments
- All call sites updated (tests, guest/*, SDK macros, benchmarks)

### What's TODO (future PRs)
- [ ] Implement actual bytecode commitment computation in `TrustedBytecodeCommitments::derive()`
- [ ] Add E2E tests for Committed mode
- [ ] Exercise `BytecodeClaimReduction` sumcheck with Committed mode
- [ ] Consider unified `JoltConfig` struct for all configuration
