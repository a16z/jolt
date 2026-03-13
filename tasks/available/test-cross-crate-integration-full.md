# cross-crate-integration-full: Full Stack Integration Testing

**Scope:** workspace-level integration tests

**Depends:** impl-jolt-zkvm, test-jolt-zkvm-integration, cross-crate-integration-1

**Verifier:** ./verifiers/scoped.sh /workdir

**Context:**

Create comprehensive integration tests that verify the entire Jolt zkVM stack works correctly together. This tests all crates working in concert to prove and verify RISC-V program execution.

### Test Location

Create tests in workspace-level test directory:
- `tests/full_stack_integration.rs`

### Integration Tests

#### 1. Simple Arithmetic Program

```rust
use jolt_zkvm::*;
use jolt_dory::*;
use jolt_field::ark_bn254::Fr as BN254Fr;
use jolt_transcript::Blake2bTranscript;
use jolt_instructions::JoltInstructionSet;
use common::{JoltDevice, MemoryLayout};

#[test]
fn test_full_stack_simple_arithmetic() {
    // RISC-V assembly program that computes: result = (a + b) * c
    let program = r#"
        .section .text
        .global _start
        _start:
            li a0, 5      # a = 5
            li a1, 3      # b = 3
            li a2, 7      # c = 7
            add a3, a0, a1 # a3 = a + b = 8
            mul a4, a3, a2 # a4 = 8 * 7 = 56

            # Write result to output
            li t0, 0x4000 # Output address
            sw a4, 0(t0)  # Store result

            # Exit
            li a7, 10     # Exit syscall
            ecall
    "#;

    // Compile and execute
    let elf = compile_riscv_assembly(program);
    let mut device = JoltDevice::new();
    let trace = execute_program(&elf, &mut device);

    // Verify execution result
    assert_eq!(device.outputs[0], 56);

    // Setup zkVM with Dory
    let config = ProverConfig {
        memory_layout: MemoryLayout::default(),
        first_round_strategy: FirstRoundStrategy::Standard,
    };

    let dory_params = DoryParams {
        t: 10,
        max_num_rows: 2048,
        num_columns: 512,
    };
    let dory = DoryScheme::new(dory_params);
    let pcs_setup = dory.setup_prover(estimate_poly_size(&trace));

    let prover = JoltProver::<DoryScheme>::new(config, pcs_setup);

    // Generate proof
    let mut prover_transcript = Blake2bTranscript::new(b"arithmetic-test");
    let proof = prover.prove(trace.clone(), &mut prover_transcript)
        .expect("Proving should succeed");

    // Verify proof
    let verifier_setup = dory.setup_verifier(estimate_poly_size(&trace));
    let verifier = JoltVerifier::<DoryScheme>::new(verifier_setup);

    let mut verifier_transcript = Blake2bTranscript::new(b"arithmetic-test");
    let result = verifier.verify(&proof, &mut verifier_transcript);

    assert!(result.is_ok(), "Proof verification failed");

    // Check proof size
    let proof_size = bincode::serialize(&proof).unwrap().len();
    println!("Proof size for simple arithmetic: {} bytes", proof_size);
}
```

#### 2. Fibonacci Computation

```rust
#[test]
fn test_full_stack_fibonacci() {
    let program = r#"
        .section .text
        .global _start
        _start:
            li a0, 10     # n = 10 (compute fib(10))
            li t0, 0      # fib(0) = 0
            li t1, 1      # fib(1) = 1
            li t2, 2      # counter = 2

        fib_loop:
            bge t2, a0, done  # if counter >= n, done
            add t3, t0, t1    # t3 = fib(i-2) + fib(i-1)
            mv t0, t1         # shift: fib(i-2) = fib(i-1)
            mv t1, t3         # shift: fib(i-1) = fib(i)
            addi t2, t2, 1    # counter++
            j fib_loop

        done:
            # Write result
            li t4, 0x4000
            sw t1, 0(t4)

            li a7, 10
            ecall
    "#;

    let (trace, device) = execute_and_trace(program);

    // Verify fib(10) = 55
    assert_eq!(device.outputs[0], 55);

    // Test with different proving strategies
    for strategy in [
        FirstRoundStrategy::Standard,
        FirstRoundStrategy::UnivariateSkip { domain_size: 256 },
    ] {
        let config = ProverConfig {
            memory_layout: MemoryLayout::default(),
            first_round_strategy: strategy,
        };

        let proof = prove_with_config(trace.clone(), config);
        assert!(verify_proof(proof).is_ok());
    }
}
```

#### 3. Memory-Intensive Program

```rust
#[test]
fn test_full_stack_memory_operations() {
    let program = r#"
        .section .data
        array: .space 400  # 100 integers

        .section .text
        .global _start
        _start:
            la t0, array      # Load array base address
            li t1, 100        # Array size
            li t2, 0          # Index

        init_loop:
            bge t2, t1, sort_start
            slli t3, t2, 2    # t3 = index * 4
            add t4, t0, t3    # t4 = array + offset
            sw t2, 0(t4)      # array[i] = i
            addi t2, t2, 1
            j init_loop

        sort_start:
            # Simple bubble sort
            li t2, 0          # i = 0
        outer_loop:
            li t3, 0          # j = 0
            sub t5, t1, t2    # limit = size - i
            addi t5, t5, -1   # limit = size - i - 1

        inner_loop:
            bge t3, t5, outer_continue

            # Load array[j] and array[j+1]
            slli t6, t3, 2
            add t6, t0, t6
            lw a0, 0(t6)      # a0 = array[j]
            lw a1, 4(t6)      # a1 = array[j+1]

            # Swap if needed
            ble a0, a1, no_swap
            sw a1, 0(t6)
            sw a0, 4(t6)

        no_swap:
            addi t3, t3, 1
            j inner_loop

        outer_continue:
            addi t2, t2, 1
            blt t2, t1, outer_loop

            # Output first element
            lw a0, 0(t0)
            li t4, 0x4000
            sw a0, 0(t4)

            li a7, 10
            ecall
    "#;

    let (trace, device) = execute_and_trace(program);

    // Configure with custom memory layout
    let config = ProverConfig {
        memory_layout: MemoryLayout {
            ram_start: 0x10000,
            ram_size: 0x20000,
            stack_start: 0x30000,
            stack_size: 0x8000,
        },
        first_round_strategy: FirstRoundStrategy::Standard,
    };

    // Prove with larger Dory parameters for memory-heavy program
    let dory_params = DoryParams {
        t: 12,
        max_num_rows: 4096,
        num_columns: 1024,
    };

    let proof = prove_with_dory_params(trace, config, dory_params);
    assert!(verify_proof(proof).is_ok());
}
```

#### 4. Instruction Coverage Test

```rust
#[test]
fn test_full_stack_instruction_coverage() {
    // Program that exercises many different instructions
    let program = r#"
        .section .text
        .global _start
        _start:
            # Arithmetic instructions
            li a0, 100
            li a1, 50
            add a2, a0, a1    # ADD
            sub a3, a0, a1    # SUB
            addi a4, a0, 25   # ADDI

            # Logical operations
            li t0, 0xFF
            li t1, 0x0F
            and t2, t0, t1    # AND
            or t3, t0, t1     # OR
            xor t4, t0, t1    # XOR
            andi t5, t0, 0x33 # ANDI

            # Shifts
            li s0, 0x80
            slli s1, s0, 2    # SLLI
            srli s2, s0, 2    # SRLI
            srai s3, s0, 2    # SRAI

            # Comparisons
            slt s4, a1, a0    # SLT
            sltu s5, a1, a0   # SLTU
            slti s6, a0, 200  # SLTI

            # Memory operations
            li s7, 0x2000
            sw a0, 0(s7)      # SW
            lw s8, 0(s7)      # LW
            sb t0, 4(s7)      # SB
            lb s9, 4(s7)      # LB

            # Branches (simple test)
            beq a0, a0, skip  # BEQ
            li a0, 0          # Should be skipped
        skip:
            bne a0, a1, cont  # BNE
            li a0, 0          # Should be skipped
        cont:

            # Output result
            li t0, 0x4000
            sw a2, 0(t0)

            li a7, 10
            ecall
    "#;

    let (trace, device) = execute_and_trace(program);

    // Verify various instruction types were executed
    let instruction_set = JoltInstructionSet::new();
    let mut instruction_counts = std::collections::HashMap::new();

    for cycle in &trace.cycles {
        let opcode = cycle.instruction_opcode;
        if let Some(instruction) = instruction_set.instruction(opcode) {
            *instruction_counts.entry(instruction.name()).or_insert(0) += 1;
        }
    }

    // Verify we exercised various instruction categories
    assert!(instruction_counts.contains_key("ADD"));
    assert!(instruction_counts.contains_key("AND"));
    assert!(instruction_counts.contains_key("SLLI"));
    assert!(instruction_counts.contains_key("SW"));
    assert!(instruction_counts.contains_key("BEQ"));

    println!("Instruction coverage: {:?}", instruction_counts);

    // Prove execution
    let proof = prove_standard(trace);
    assert!(verify_proof(proof).is_ok());
}
```

#### 5. Virtual Instruction Test

```rust
#[test]
fn test_full_stack_virtual_instructions() {
    // Program using Jolt virtual instructions
    let program = r#"
        .section .text
        .global _start
        _start:
            # Test ASSERT_EQ virtual instruction
            li a0, 42
            li a1, 42
            .word 0x00000001  # ASSERT_EQ opcode

            # Test POW2 virtual instruction
            li a0, 5
            .word 0x00000002  # POW2 opcode
            # Result in a0 should be 32 (2^5)

            # Output result
            li t0, 0x4000
            sw a0, 0(t0)

            li a7, 10
            ecall
    "#;

    let (trace, _device) = execute_and_trace(program);

    // Virtual instructions should be handled correctly
    let proof = prove_standard(trace);
    assert!(verify_proof(proof).is_ok());
}
```

#### 6. Performance Benchmarking

```rust
#[test]
#[ignore] // Run with --ignored for benchmarks
fn test_full_stack_performance_scaling() {
    println!("Full Stack Performance Scaling Test\n");

    let program_sizes = vec![
        ("tiny", 10),
        ("small", 100),
        ("medium", 1000),
        ("large", 10000),
    ];

    let mut results = Vec::new();

    for (name, size) in program_sizes {
        // Generate program with `size` instructions
        let program = generate_linear_program(size);
        let (trace, _) = execute_and_trace(&program);

        let start = std::time::Instant::now();
        let proof = prove_standard(trace.clone());
        let proving_time = start.elapsed();

        let start = std::time::Instant::now();
        assert!(verify_proof(proof.clone()).is_ok());
        let verification_time = start.elapsed();

        let proof_size = bincode::serialize(&proof).unwrap().len();

        results.push((
            name,
            size,
            trace.cycles.len(),
            proving_time,
            verification_time,
            proof_size,
        ));

        println!("Program: {} ({} instructions)", name, size);
        println!("  Cycles: {}", trace.cycles.len());
        println!("  Proving: {:?}", proving_time);
        println!("  Verification: {:?}", verification_time);
        println!("  Proof size: {} KB\n", proof_size / 1024);
    }

    // Analyze scaling
    println!("\nScaling Analysis:");
    for i in 1..results.len() {
        let (_, prev_size, _, prev_prove, _, prev_proof_size) = results[i - 1];
        let (_, curr_size, _, curr_prove, _, curr_proof_size) = results[i];

        let size_ratio = curr_size as f64 / prev_size as f64;
        let prove_ratio = curr_prove.as_secs_f64() / prev_prove.as_secs_f64();
        let proof_size_ratio = curr_proof_size as f64 / prev_proof_size as f64;

        println!("{}x size increase:", size_ratio);
        println!("  Proving time: {:.2}x", prove_ratio);
        println!("  Proof size: {:.2}x", proof_size_ratio);
    }
}
```

#### 7. Error Handling

```rust
#[test]
fn test_full_stack_trap_handling() {
    // Program that causes various traps
    let program = r#"
        .section .text
        .global _start
        _start:
            # Illegal memory access
            li t0, 0xFFFFFFFF
            lw t1, 0(t0)     # Should trap

            # Continue won't be reached
            li a7, 10
            ecall
    "#;

    let (trace, device) = execute_and_trace(program);

    // Verify trap occurred
    assert!(trace.trapped);

    // Proving should still work for trapped execution
    let proof = prove_standard(trace);
    assert!(verify_proof(proof).is_ok());
}

#[test]
fn test_full_stack_assertion_failure() {
    // Program with failing assertion
    let program = r#"
        .section .text
        .global _start
        _start:
            li a0, 42
            li a1, 43
            .word 0x00000001  # ASSERT_EQ - should fail

            li a7, 10
            ecall
    "#;

    let (trace, _) = execute_and_trace(program);

    // Should trap on assertion failure
    assert!(trace.trapped);

    // Can still prove trapped execution
    let proof = prove_standard(trace);
    assert!(verify_proof(proof).is_ok());
}
```

### Helper Functions

```rust
fn compile_riscv_assembly(asm: &str) -> Vec<u8> {
    // Use riscv toolchain to compile
    // This is a placeholder - actual implementation would use real toolchain
    vec![]
}

fn execute_and_trace(program: &str) -> (ExecutionTrace, JoltDevice) {
    let elf = compile_riscv_assembly(program);
    let mut device = JoltDevice::new();
    let cycles = tracer::execute(&elf, &mut device);

    let trace = ExecutionTrace {
        cycles,
        device: device.clone(),
        trapped: device.trapped(),
    };

    (trace, device)
}

fn prove_standard(trace: ExecutionTrace) -> JoltProof<DoryScheme> {
    let config = ProverConfig {
        memory_layout: MemoryLayout::default(),
        first_round_strategy: FirstRoundStrategy::Standard,
    };

    let dory = create_standard_dory();
    let pcs_setup = dory.setup_prover(estimate_poly_size(&trace));

    let prover = JoltProver::new(config, pcs_setup);
    let mut transcript = Blake2bTranscript::new(b"test");

    prover.prove(trace, &mut transcript).expect("Proving failed")
}

fn verify_proof(proof: JoltProof<DoryScheme>) -> Result<(), JoltError> {
    let dory = create_standard_dory();
    let verifier_setup = dory.setup_verifier(estimate_poly_size_from_proof(&proof));

    let verifier = JoltVerifier::new(verifier_setup);
    let mut transcript = Blake2bTranscript::new(b"test");

    verifier.verify(&proof, &mut transcript)
}

fn create_standard_dory() -> DoryScheme {
    DoryScheme::new(DoryParams {
        t: 10,
        max_num_rows: 2048,
        num_columns: 512,
    })
}

fn estimate_poly_size(trace: &ExecutionTrace) -> usize {
    // Estimate based on trace size
    (trace.cycles.len() * 64).next_power_of_two()
}

fn generate_linear_program(num_instructions: usize) -> String {
    let mut program = String::from(
        ".section .text\n.global _start\n_start:\n"
    );

    // Generate simple instruction sequence
    for i in 0..num_instructions {
        program.push_str(&format!("    addi t0, t0, {}\n", i % 100));
    }

    program.push_str(
        "    li t1, 0x4000\n    sw t0, 0(t1)\n    li a7, 10\n    ecall\n"
    );

    program
}
```

### Test Configuration

Add to workspace `Cargo.toml`:

```toml
[[test]]
name = "full_stack_integration"
path = "tests/full_stack_integration.rs"

[dev-dependencies]
jolt-zkvm = { path = "crates/jolt-zkvm" }
jolt-dory = { path = "crates/jolt-dory" }
jolt-field = { path = "crates/jolt-field" }
jolt-transcript = { path = "crates/jolt-transcript" }
jolt-instructions = { path = "crates/jolt-instructions" }
common = { path = "common" }
tracer = { path = "tracer" }
bincode = "1.3"
```

### Acceptance Criteria

- Full stack integration test file created
- Tests simple arithmetic programs
- Tests complex programs (Fibonacci, sorting)
- Tests memory-intensive operations
- Tests instruction coverage
- Tests virtual instructions
- Performance benchmarking included
- Error cases handled (traps, assertions)
- Helper functions for common operations
- All tests pass
- Performance metrics collected
- No modifications to crate source code