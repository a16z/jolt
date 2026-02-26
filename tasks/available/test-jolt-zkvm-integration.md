# test-jolt-zkvm-integration: Integration tests for jolt-zkvm

**Scope:** crates/jolt-zkvm/tests/

**Depends:** impl-jolt-zkvm, test-jolt-zkvm

**Verifier:** ./verifiers/scoped.sh /workdir jolt-zkvm

**Context:**

Write integration tests for the `jolt-zkvm` crate that verify the complete zkVM prover/verifier functionality from an external user's perspective.

### Integration Test Files

Create the following test files in `crates/jolt-zkvm/tests/`:

#### 1. `simple_programs.rs` - Test small RISC-V programs

Test proving and verifying simple RISC-V programs:
- Basic arithmetic sequences
- Simple loops
- Function calls
- Memory operations
- Test with different memory layouts

#### 2. `claim_reduction.rs` - Test claim reduction logic

Test the various claim reduction mechanisms:
- RAM read/write checking
- Register consistency
- Bytecode verification
- Instruction lookup correctness
- Verify all sub-protocols integrate correctly

#### 3. `memory_checking.rs` - Test RAM/register consistency

Focus on memory consistency checking:
- Programs with heavy memory usage
- Register spilling scenarios
- Boundary conditions (stack overflow, heap limits)
- Concurrent memory access patterns

#### 4. `e2e_proving.rs` - End-to-end proof generation

Test complete proof generation for various programs:
- Different program sizes
- Various instruction mixes
- Test proof serialization
- Verify proof size scaling
- Benchmark proving times

### Implementation Examples

**Simple Program Test:**
```rust
#[test]
fn test_fibonacci_program() {
    // Simple Fibonacci computation
    let program_code = r#"
        li a0, 10        # n = 10
        li t0, 0         # fib(0) = 0
        li t1, 1         # fib(1) = 1
        li t2, 2         # i = 2

    loop:
        bge t2, a0, end  # if i >= n, done
        add t3, t0, t1   # fib(i) = fib(i-2) + fib(i-1)
        mv t0, t1        # shift values
        mv t1, t3
        addi t2, t2, 1   # i++
        j loop

    end:
        mv a0, t1        # return fib(n)
    "#;

    // Create execution trace
    let trace = create_trace_from_assembly(program_code);

    // Setup zkVM
    let config = ProverConfig {
        memory_layout: MemoryLayout::default(),
        first_round_strategy: FirstRoundStrategy::Standard,
    };

    let pcs_setup = DoryScheme::setup_prover(MAX_POLYNOMIAL_SIZE);
    let prover = JoltProver::new(config, pcs_setup);

    // Generate proof
    let mut transcript = Blake2bTranscript::new(b"fibonacci-test");
    let proof = prover.prove(trace.clone(), &mut transcript).unwrap();

    // Verify proof
    let verifier_setup = DoryScheme::setup_verifier(MAX_POLYNOMIAL_SIZE);
    let verifier = JoltVerifier::new(verifier_setup);

    let mut verify_transcript = Blake2bTranscript::new(b"fibonacci-test");
    let result = verifier.verify(&proof, &mut verify_transcript);

    assert!(result.is_ok());

    // Verify output is correct (fib(10) = 55)
    assert_eq!(trace.device.outputs[0], 55);
}
```

**Memory-Heavy Program:**
```rust
#[test]
fn test_memory_intensive_program() {
    // Program that performs many memory operations
    let program_code = r#"
        li t0, 0x1000    # base address
        li t1, 1024      # array size
        li t2, 0         # index

    init_loop:
        bge t2, t1, sort_start
        slli t3, t2, 2   # offset = index * 4
        add t4, t0, t3   # address = base + offset
        sub t5, t1, t2   # value = size - index (reverse order)
        sw t5, 0(t4)     # store value
        addi t2, t2, 1
        j init_loop

    sort_start:
        # Simple bubble sort
        # ... sorting implementation ...
    "#;

    let trace = create_trace_from_assembly(program_code);

    // Configure with specific memory layout
    let config = ProverConfig {
        memory_layout: MemoryLayout {
            ram_start: 0x1000,
            ram_size: 0x10000,
            stack_start: 0x20000,
            stack_size: 0x4000,
        },
        first_round_strategy: FirstRoundStrategy::Standard,
    };

    // Generate and verify proof
    let proof = generate_proof_with_config(trace, config);
    assert!(verify_proof(proof).is_ok());
}
```

**Claim Reduction Test:**
```rust
#[test]
fn test_claim_reduction_consistency() {
    // Program that exercises all instruction types
    let mixed_program = create_mixed_instruction_program();
    let trace = execute_program(mixed_program);

    // Extract intermediate values for verification
    let prover = JoltProver::new(config, pcs_setup);

    // Hook into proving to capture claim reductions
    let (proof, claim_data) = prover.prove_with_diagnostics(
        trace,
        &mut transcript
    ).unwrap();

    // Verify each claim reduction is consistent
    assert!(claim_data.ram_claims.is_consistent());
    assert!(claim_data.register_claims.is_consistent());
    assert!(claim_data.bytecode_claims.is_consistent());
    assert!(claim_data.instruction_lookup_claims.is_consistent());

    // Verify all reductions sum to expected values
    let total_cycles = trace.cycles.len();
    assert_eq!(
        claim_data.total_lookups(),
        total_cycles * INSTRUCTIONS_PER_CYCLE
    );
}
```

**Proof Size Scaling:**
```rust
#[test]
fn test_proof_size_scaling() {
    let program_sizes = vec![10, 100, 1000, 10000];
    let mut proof_sizes = vec![];

    for size in program_sizes {
        // Generate program with `size` instructions
        let program = generate_linear_program(size);
        let trace = execute_program(program);

        // Generate proof
        let proof = generate_proof(trace);

        // Serialize and measure size
        let serialized = bincode::serialize(&proof).unwrap();
        proof_sizes.push(serialized.len());

        println!("Program size: {}, Proof size: {} bytes",
                 size, serialized.len());
    }

    // Verify proof size grows sublinearly
    for i in 1..proof_sizes.len() {
        let size_ratio = proof_sizes[i] as f64 / proof_sizes[i-1] as f64;
        let program_ratio = program_sizes[i] as f64 / program_sizes[i-1] as f64;

        // Proof should grow slower than program
        assert!(size_ratio < program_ratio,
                "Proof size growing too fast");
    }
}
```

**Configuration Testing:**
```rust
#[test]
fn test_different_configurations() {
    let base_program = create_test_program();

    // Test different first round strategies
    for strategy in [
        FirstRoundStrategy::Standard,
        FirstRoundStrategy::UnivariateSkip { domain_size: 256 },
    ] {
        let config = ProverConfig {
            memory_layout: MemoryLayout::default(),
            first_round_strategy: strategy,
        };

        let proof = prove_with_config(base_program.clone(), config);
        assert!(verify_proof(proof).is_ok());
    }
}
```

**Error Cases:**
```rust
#[test]
fn test_invalid_memory_access() {
    // Program that accesses out-of-bounds memory
    let invalid_program = r#"
        li t0, 0xFFFFFFFF  # Invalid address
        lw t1, 0(t0)       # Should trap
    "#;

    let trace = execute_program(invalid_program);

    // Proving should succeed (trap is valid execution)
    let proof = generate_proof(trace);
    assert!(verify_proof(proof).is_ok());

    // But trace should indicate trap
    assert!(trace.trapped);
}
```

### Test Utilities

Create helper functions for common operations:

```rust
/// Helper to create trace from assembly code
fn create_trace_from_assembly(asm: &str) -> ExecutionTrace {
    // Compile assembly to ELF
    let elf = compile_assembly(asm);

    // Execute in tracer
    let mut device = JoltDevice::default();
    let cycles = tracer::execute(&elf, &mut device);

    ExecutionTrace { cycles, device }
}

/// Helper to generate and verify proof
fn prove_and_verify(trace: ExecutionTrace) -> Result<(), JoltError> {
    let config = ProverConfig::default();
    let pcs_setup = create_test_pcs_setup();

    let prover = JoltProver::new(config.clone(), pcs_setup);
    let mut transcript = Blake2bTranscript::new(b"test");
    let proof = prover.prove(trace, &mut transcript)?;

    let verifier_setup = create_test_verifier_setup();
    let verifier = JoltVerifier::new(verifier_setup);
    let mut verify_transcript = Blake2bTranscript::new(b"test");

    verifier.verify(&proof, &mut verify_transcript)
}
```

### Acceptance Criteria

- Four integration test files created
- Various RISC-V programs tested
- Claim reduction consistency verified
- Memory checking thoroughly tested
- End-to-end proving benchmarked
- Error cases handled correctly
- Helper utilities for test setup
- All tests pass with `cargo nextest run -p jolt-zkvm`
- Well-documented test cases
- No source code modifications