# test-jolt-zkvm: Comprehensive tests for jolt-zkvm

**Scope:** crates/jolt-zkvm/

**Depends:** impl-jolt-zkvm

**Verifier:** ./verifiers/scoped.sh /workdir jolt-zkvm

**Context:**

Write comprehensive tests for the `jolt-zkvm` crate. This includes integration tests for the full proving stack, sub-protocol unit tests, and end-to-end test infrastructure.

**Do not modify source logic — test-only changes.**

### Test categories

#### 1. Sub-protocol unit tests

Each sub-protocol module (RAM, registers, bytecode, claim reductions, instruction lookups) should be tested in isolation:

**RAM checking:**
- Simple trace: write value to address, read it back → proof succeeds
- Read from unwritten address → should fail or return default
- Multiple writes to same address, read returns latest → proof succeeds
- Out-of-order reads/writes → consistency check catches violations

**Register checking:**
- Write to register, read it back → proof succeeds
- All 32 registers written and read → proof succeeds
- Read from register with stale value → consistency check catches it

**Bytecode checking:**
- Program counter follows sequential execution → proof succeeds
- Jump instruction changes PC correctly → proof succeeds
- Tampered bytecode (wrong instruction at PC) → proof fails

**Instruction lookups:**
- Simple instruction (ADD) with known operands → lookup decomposition verified
- All instruction types exercise the lookup path

#### 2. Small execution trace integration tests

Build minimal execution traces by hand (not from compiled programs):

- **NOP trace:** 1 instruction, does nothing → prove → verify
- **ADD trace:** `x0 = 5; x1 = 3; x2 = x0 + x1` → prove → verify, verify x2 == 8
- **Branch trace:** `if x0 == x1 goto L; x2 = 1; L: x2 = 0` → prove → verify for both paths
- **Memory trace:** store word, load word → prove → verify

#### 3. End-to-end test macro

Implement the `jolt_e2e_test!` macro:

```rust
#[macro_export]
macro_rules! jolt_e2e_test {
    ($name:ident, guest = $guest:expr $(, config = $config:expr)?) => {
        #[test]
        fn $name() {
            // Compile guest program
            // Trace execution
            // Prove with default (or custom) config
            // Verify
        }
    };
}
```

Use it with 2-3 simple guest programs to verify the macro works.

#### 4. Prover/Verifier configuration tests

- Default config produces valid proofs
- Different `FirstRoundStrategy` options all produce valid proofs
- Different memory layout configs work

#### 5. Proof serialization round-trip

- `JoltProof` serialize → deserialize → verify succeeds
- Tampered serialized proof → deserialize → verify fails (or deserialize fails)

#### 6. Error propagation

- Sub-protocol errors (SumcheckError, OpeningsError, SpartanError) propagate correctly through JoltError
- Each error variant is reachable via a concrete test case

#### 7. Regression tests

For any bugs found during implementation, add a regression test that exercises the specific bug's code path.

**Acceptance:**

- Each sub-protocol tested in isolation (RAM, registers, bytecode, lookups)
- At least 4 hand-built execution trace integration tests
- `jolt_e2e_test!` macro implemented and used
- Proof serialization round-trip verified
- Error propagation tested
- All tests pass
- No modifications to non-test source code
