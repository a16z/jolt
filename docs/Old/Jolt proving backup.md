---
title: "Jolt zkVM"
abstract: "Comprehensive guide to Jolt's proving and verifying execution flow, from preprocessing through the 5-stage DAG verification protocol. Traces the complete codebase architecture showing how Jolt implements efficient zkVM proving using Lasso lookups, Spartan constraints, and Twist memory checking."
author: "Parti"
date: "2025-10-19"
titlepage: true
titlepage-rule-color: "360049"
toc: true
toc-own-page: true
toc-depth: 2
---


# Jolt Code Flow: Proving and Verifying

This document traces the execution flow when proving and verifying in Jolt, showing exactly which files are called and what they do.

## Terminology

**DAG** = **Directed Acyclic Graph**

- Jolt's proof system is structured as a graph where:
  - **Nodes** = Sumcheck instances
  - **Edges** = Polynomial evaluations (output of one sumcheck → input to another)
  - **Acyclic** = No circular dependencies (flows in one direction through stages)
- The "JoltDAG" orchestrates this graph execution across 5 stages

## Overview: The Proof Journey

**Note**: Function names depend on your guest function. The `#[jolt::provable]` macro generates `prove_{function_name}` and `verify_{function_name}`.

Examples:

- `#[jolt::provable] fn sha3(...)` → generates `prove_sha3()`, `verify_sha3()`
- `#[jolt::provable] fn fibonacci(...)` → generates `prove_fibonacci()`, `verify_fibonacci()`
- `#[jolt::provable] fn my_func(...)` → generates `prove_my_func()`, `verify_my_func()`

When you call `prove_{your_function}(input)` from your host program, here's the high-level flow:

```
Host calls prove_{your_function}(input)
  ↓
JoltDAG::prove() orchestrates the proof DAG
  ↓
Stage 1-4: Run batched sumchecks for each component (graph nodes)
  ↓
Stage 5: Create batched opening proof
  ↓
Return proof
```

---

## Part 1: Setup and Preprocessing

### 1. Host Program Initialization
**File**: `examples/*/host/src/main.rs`

```rust
// Replace "sha3" with your function name
let mut program = guest::compile_{your_function}(target_dir);
let prover_preprocessing = guest::preprocess_prover_{your_function}(&mut program);
```

**What happens**:

- `compile_{your_function}()` compiles guest Rust → RISC-V ELF binary
- Parses ELF to extract bytecode (RISC-V instructions)
- Returns `Program` struct with bytecode and memory layout
- `preprocess_prover_{your_function}()` generates proving keys (polynomial commitments, SRS)
  - Commits to bytecode polynomial
  - Generates Dory SRS for polynomial commitment scheme
  - This is expensive but only done once per program

**Key outputs**:

- `Program`: Contains RISC-V bytecode
- `JoltProverPreprocessing`: Cryptographic setup (commitments, generators)

---

## Part 2: Execution and Witness Generation

### 2. Build Prover and Execute
**File**: Host calls `prove_{your_function}(input)`

```rust
let prove_{your_function} = guest::build_prover_{your_function}(program, prover_preprocessing);
let (output, proof, program_io) = prove_{your_function}(input);
```

**What happens**:

- **Emulation** ([tracer/src/emulator/](tracer/src/emulator/)):
  - RISC-V emulator executes bytecode instruction-by-instruction
  - Each instruction implements `RISCVInstruction::execute()`
  - Records every memory read/write, register access
  - Produces execution trace: `Vec<Cycle>` (one cycle per instruction)

- **Witness generation** ([jolt-core/src/zkvm/witness.rs](jolt-core/src/zkvm/witness.rs)):
  - Converts trace into multilinear polynomial extensions (MLEs)
  - For each cycle, extracts:
    - Which instruction executed
    - Which registers read/written
    - Which memory addresses accessed
    - Instruction operands and results

**Key outputs**:

- `Vec<Cycle>`: Complete execution trace
- `Memory`: Final memory state
- `program_io`: Inputs/outputs of the program

This trace is what we'll prove is correct!

---

## Part 3: The Proof Generation (The Big One!)

### 3. Enter the DAG
**File**: [jolt-core/src/zkvm/dag/jolt_dag.rs](jolt-core/src/zkvm/dag/jolt_dag.rs) → `JoltDAG::prove()`

**Purpose**: Orchestrates the entire proof generation across 5 stages.

#### 3.1 Create StateManager
**File**: [jolt-core/src/zkvm/dag/state_manager.rs](jolt-core/src/zkvm/dag/state_manager.rs)

```rust
let state_manager = StateManager::new_prover(
    preprocessing,
    trace,
    program_io,
    final_memory_state
);
```

**What StateManager does**:

- **Central coordination hub** for the entire proof
- Holds all the data needed by different components:
  - `transcript`: Fiat-Shamir transcript for challenges
  - `proofs`: Map storing all sumcheck proofs (Stage1-4)
  - `commitments`: Vector of polynomial commitments
  - `prover_state`: Contains trace, preprocessing, opening accumulator

- **Opening Accumulator** (`ProverOpeningAccumulator`):
  - Tracks all polynomial evaluations that need to be proven
  - Two types:
    - **Committed polynomials**: Need PCS opening proof (Stage 5)
    - **Virtual polynomials**: Need another sumcheck (outputs of one sumcheck become inputs to next)
  - Creates DAG structure: one sumcheck's outputs flow into another's inputs

**Key insight**: StateManager enables the DAG structure by passing data between sumchecks through the accumulator.

#### 3.2 Fiat-Shamir Preamble
```rust
state_manager.fiat_shamir_preamble();
```

**What happens**: Append public inputs to transcript (inputs, outputs, memory layout). This binds the proof to specific execution.

#### 3.3 Generate and Commit Polynomials
**File**: [jolt-core/src/zkvm/dag/jolt_dag.rs:574](jolt-core/src/zkvm/dag/jolt_dag.rs#L574) → `generate_and_commit_polynomials()`

```rust
let opening_proof_hints = Self::generate_and_commit_polynomials(&mut state_manager)?;
```

**What happens**:

1. Calls `CommittedPolynomial::generate_witness_batch()` ([jolt-core/src/zkvm/witness.rs](jolt-core/src/zkvm/witness.rs))
   - For each committed polynomial type (RAM, Registers, Instructions, etc.):
     - Constructs MLE from execution trace
     - Example: `t_read` polynomial is 1 at cycle i if instruction at cycle i reads memory
2. Commits to all polynomials using Dory PCS ([jolt-core/src/poly/commitment/dory.rs](jolt-core/src/poly/commitment/dory.rs))
   - `PCS::batch_commit()`: Creates cryptographic commitments
   - Returns commitment + opening hint (used later in Stage 5)
3. Appends commitments to transcript

**Output**: Commitments stored in StateManager, hints saved for Stage 5

---

### 4. The Five Stages: Sumcheck Ballet

Each stage runs multiple sumchecks in parallel. The five components (Spartan, Lookups, Registers, RAM, Bytecode) each contribute sumchecks to different stages.

#### Stage 1: Spartan Outer Sumcheck
**File**: [jolt-core/src/zkvm/spartan/mod.rs](jolt-core/src/zkvm/spartan/mod.rs) → `SpartanDag::stage1_prove()`

**Purpose**: Prove R1CS constraints are satisfied.

**What happens**:

1. Get random challenge $\tau$ from transcript
2. Run Spartan outer sumcheck ([jolt-core/src/zkvm/spartan/inner.rs](jolt-core/src/zkvm/spartan/inner.rs)):
   Claim:
    $\sum_x eq(\tau, x) · (Az(x) · Bz(x) - Cz(x)) = 0$
   
   - This proves ~30 R1CS constraints per cycle are satisfied
   - Constraints enforce: PC updates, component linking, arithmetic ops

3. **The Sumcheck Protocol** ([jolt-core/src/subprotocols/sumcheck.rs](jolt-core/src/subprotocols/sumcheck.rs)):
   - For each round $j = 0,\dots,n$:
     - Prover computes univariate polynomial $g_j(X_j)$
       - Calls `sumcheck_instance.compute_prover_message()`
       - Evaluates at $0, 2, 3, ...,$ degree
     - Compress polynomial and append to transcript
     - Verifier samples random challenge r_j
     - Prover binds: `sumcheck_instance.bind(r_j, j)`

4. After sumcheck completes:
   - Claims about $Az(r), Bz(r), Cz(r)$ at random point $r$
   - Store these as **virtual polynomial openings** in accumulator
   - These will be proven in Stage 2 via product sumchecks

**Files involved**:

- [jolt-core/src/subprotocols/sumcheck.rs](jolt-core/src/subprotocols/sumcheck.rs): Core sumcheck engine
- [jolt-core/src/zkvm/r1cs/](jolt-core/src/zkvm/r1cs/): R1CS constraint definitions
- [jolt-core/src/poly/unipoly.rs](jolt-core/src/poly/unipoly.rs): Univariate polynomial operations

**Output**: Stage 1 proof stored in StateManager, virtual openings in accumulator

---

#### Stage 2: Batched Sumchecks
**File**: [jolt-core/src/zkvm/dag/jolt_dag.rs:138](jolt-core/src/zkvm/dag/jolt_dag.rs#L138)

**Components contributing**:

1. **Spartan** (Product sumchecks) - [jolt-core/src/zkvm/spartan/product.rs](jolt-core/src/zkvm/spartan/product.rs)
2. **Registers** (Twist checking) - [jolt-core/src/zkvm/registers/](jolt-core/src/zkvm/registers/)
3. **RAM** (Twist checking) - [jolt-core/src/zkvm/ram/](jolt-core/src/zkvm/ram/)
4. **Lookups** (Booleanity) - [jolt-core/src/zkvm/instruction_lookups/](jolt-core/src/zkvm/instruction_lookups/)

**How batching works**:
```rust
let mut stage2_instances: Vec<_> = std::iter::empty()
    .chain(spartan_dag.stage2_prover_instances(&mut state_manager))
    .chain(registers_dag.stage2_prover_instances(&mut state_manager))
    .chain(ram_dag.stage2_prover_instances(&mut state_manager))
    .chain(lookups_dag.stage2_prover_instances(&mut state_manager))
    .collect();

let (stage2_proof, _r_stage2) = BatchedSumcheck::prove(
    stage2_instances_mut,
    Some(accumulator.clone()),
    &mut *transcript.borrow_mut(),
);
```

**What BatchedSumcheck does** ([jolt-core/src/subprotocols/sumcheck.rs:178](jolt-core/src/subprotocols/sumcheck.rs#L178)):

1. Sample batching coefficients $\alpha_1, \alpha_2, \alpha_3, ...$ from transcript
2. Combine all claims:
   Combined claim 
   $= \alpha_1·claim_1 + \alpha_2·claim_2 + ...$
   
3. Run single sumcheck on combined polynomial:
   
   $\sum_x (\alpha_1·P_1(x) + \alpha_2·P_2(x) + ...)$
   
4. Each round: prover sends one combined univariate polynomial
5. Verifier sends one challenge, used by all instances
6. After sumcheck: each instance caches its own output claims

**Example: Registers Twist Checking**
**File**: [jolt-core/src/zkvm/registers/mod.rs](jolt-core/src/zkvm/registers/mod.rs)

**Purpose**: Prove register reads/writes are consistent (read returns last written value).

**What happens** (Twist memory checking):

1. Two memory traces:
   - Time-ordered: registers as accessed during execution
   - Address-ordered: same accesses sorted by register number

2. Prove they're the same via grand product argument:
   
   $\prod fingerprint_{time}(i) = \prod fingerprint_{addr}(j)$
   
   Where 

   $fingerprint = hash(address, timestamp, value)$

3. Registers uses 3 sumchecks (batched in Stage 2):
   - Read-checking for rs1 (first source register)
   - Read-checking for rs2 (second source register)
   - Write-checking for rd (destination register)

**Files**:

- [jolt-core/src/zkvm/registers/read_write_checking.rs](jolt-core/src/zkvm/registers/read_write_checking.rs): Twist read/write checking
- [jolt-core/src/subprotocols/grand_product.rs](jolt-core/src/subprotocols/grand_product.rs): Grand product argument (if exists)

**Output**: Stage 2 batched proof, more virtual openings in accumulator

---

#### Stage 3: More Batched Sumchecks
**File**: [jolt-core/src/zkvm/dag/jolt_dag.rs:194](jolt-core/src/zkvm/dag/jolt_dag.rs#L194)

**Components contributing**:

1. **Spartan** (Inner sumchecks)
2. **Registers** (Hamming weight, evaluation)
3. **Lookups** (Read-checking, Hamming weight)
4. **RAM** (Hamming weight, evaluation)

**Example: Instruction Lookups**
**File**: [jolt-core/src/zkvm/instruction_lookups/mod.rs](jolt-core/src/zkvm/instruction_lookups/mod.rs)

**Purpose**: Prove every instruction execution produced correct outputs.

**Key insight**: Instead of circuits, Jolt uses **lookup tables**.

- 64-bit ADD instruction? Look it up in a table!
- But table would be 2^128 entries (two 64-bit inputs) - too big!

**Solution: Prefix-Suffix Decomposition**

1. Split each operand into 16 chunks of 4 bits
2. Create small lookup tables for 4-bit operations (only 2^8 = 256 entries)
3. Prove correct stitching (e.g., carry bits for addition)

**What happens in Stage 3**:

1. **ReadRafSumcheck** ([jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs](jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs)):
   - Proves reads from pre-computed lookup table are correct
   - Uses Shout (offline memory checking for lookups)

2. **HammingWeightSumcheck** ([jolt-core/src/zkvm/instruction_lookups/hamming_weight.rs](jolt-core/src/zkvm/instruction_lookups/hamming_weight.rs)):
   - Part of Shout protocol
   - Proves certain vectors have correct sparsity

**Files**:

- [jolt-core/src/zkvm/instruction/](jolt-core/src/zkvm/instruction/): Per-instruction lookup logic
- [jolt-core/src/zkvm/instruction/add.rs](jolt-core/src/zkvm/instruction/add.rs): Example ADD instruction
- Each instruction implements `JoltLookupTable` trait with efficiently-evaluable MLE

**Output**: Stage 3 batched proof

---

#### Stage 4: Final Sumchecks
**File**: [jolt-core/src/zkvm/dag/jolt_dag.rs:248](jolt-core/src/zkvm/dag/jolt_dag.rs#L248)

**Components contributing**:

1. **RAM** (Ra virtualization, evaluation)
2. **Bytecode** (Read-checking)
3. **Lookups** (Ra virtualization)

**Example: Bytecode Read-Checking**
**File**: [jolt-core/src/zkvm/bytecode/mod.rs](jolt-core/src/zkvm/bytecode/mod.rs)

**Purpose**: Prove trace instructions match committed bytecode.

**What happens**:

1. During preprocessing: bytecode decoded and committed
2. During execution: trace records which instructions executed
3. Bytecode sumcheck proves:
   ```
   Each instruction in trace matches an instruction in committed bytecode
   ```
4. Uses Shout (offline memory checking):
   - Proves sequence of reads matches committed values
   - More efficient than re-committing to trace

**Output**: Stage 4 batched proof, all virtual openings resolved

---

#### Stage 5: Batched Opening Proof
**File**: [jolt-core/src/zkvm/dag/jolt_dag.rs:309](jolt-core/src/zkvm/dag/jolt_dag.rs#L309)

**Purpose**: Prove all committed polynomial evaluations claimed in Stages 1-4.

**What happens**:

1. Accumulator contains many evaluation claims:
   ```
   poly1(r1) = v1
   poly2(r2) = v2
   ...
   poly_n(r_n) = v_n
   ```

2. **Batched opening** ([jolt-core/src/poly/opening_proof.rs](jolt-core/src/poly/opening_proof.rs)):
   ```rust
   accumulator.borrow_mut().reduce_and_prove(
       polynomials_map,
       opening_proof_hints,
       &preprocessing.generators,
       &mut *transcript.borrow_mut(),
   )
   ```

3. **How it works**:
   - Sample random coefficients $\beta_1, \beta_2, ..., \beta_n$
   - Create single combined claim:
   
     $\beta_1·poly1(r1) + \beta_2·poly2(r2) + ... = \beta_1·v1 + \beta_2·v2 + ...$
     
   - Prove this single opening using Dory PCS
   - Much more efficient than n separate openings!

4. Uses Dory polynomial commitment scheme ([jolt-core/src/poly/commitment/dory.rs](jolt-core/src/poly/commitment/dory.rs)):
   - Transparent (no trusted setup)
   - Logarithmic verifier time
   - Supports batching

**Output**: Final opening proof

---

### 5. Construct and Return Proof
**File**: [jolt-core/src/zkvm/dag/jolt_dag.rs:378](jolt-core/src/zkvm/dag/jolt_dag.rs#L378)

```rust
let proof = JoltProof::from_prover_state_manager(state_manager);
```

**What's in the proof**:

- Polynomial commitments (from setup)
- Stage 1-4 sumcheck proofs
- Stage 5 batched opening proof
- All auxiliary data (advice commitments, etc.)

**Proof size**: ~100KB - 1MB depending on trace length

---

## Part 4: Verification

### 6. Verifier Entry Point
**File**: Host calls `verify_{your_function}(input, output, proof)`

```rust
let verify_{your_function} = guest::build_verifier_{your_function}(verifier_preprocessing);
let is_valid = verify_{your_function}(input, output, program_io.panic, proof);
```

**What happens**: Verifier has much less work than prover!

---

### 7. JoltDAG::verify()
**File**: [jolt-core/src/zkvm/dag/jolt_dag.rs:383](jolt-core/src/zkvm/dag/jolt_dag.rs#L383)

**Purpose**: Check proof without re-executing program.

#### 7.1 Setup Verifier StateManager
```rust
let state_manager = proof.to_verifier_state_manager(
    verifier_preprocessing,
    program_io,
);
```

**What's different from prover**:

- No execution trace (just trace length)
- No witness polynomials
- Same accumulator structure (but `VerifierOpeningAccumulator`)

#### 7.2 Fiat-Shamir Preamble
```rust
state_manager.fiat_shamir_preamble();
```

**Critical**: Must match prover's transcript exactly!

- Append same public inputs
- Append polynomial commitments from proof
- Any mismatch → verification fails

#### 7.3 Stage 1: Spartan Outer Verification
```rust
spartan_dag.stage1_verify(&mut state_manager)?;
```

**What happens** ([jolt-core/src/zkvm/spartan/mod.rs](jolt-core/src/zkvm/spartan/mod.rs)):

1. Sample same $\tau$ challenge (from transcript)
2. Extract Stage 1 sumcheck proof from proof object
3. **Verify sumcheck** ([jolt-core/src/subprotocols/sumcheck.rs:140](jolt-core/src/subprotocols/sumcheck.rs#L140)):
   ```rust
   let (output_claim, r) = proof.verify(
       input_claim,
       num_rounds,
       degree,
       transcript,
   )?;
   ```

   **How sumcheck verification works**:
   - For each round $j$:
     - Read compressed univariate from proof
     - Check consistency: $g_j(0) + g_j(1) = \text{previous\_claim}$
     - Sample $r_j$ from transcript
     - Compute $\text{next\_claim} = g_j(r_j)$
   - Check final evaluation: $\text{output\_claim} = \text{expected\_evaluation}(r)$

4. Store virtual openings in verifier accumulator
   - $Az(r), Bz(r), Cz(r)$ marked as claims to check later

**Key insight**: Verification is $O(\log n)$ per sumcheck, not $O(n)$ like proving!

#### 7.4 Stages 2-4: Batched Verification
**File**: [jolt-core/src/zkvm/dag/jolt_dag.rs:430-520](jolt-core/src/zkvm/dag/jolt_dag.rs#L430)

**What happens**:

1. Each component creates verifier instances:
   ```rust
   let stage2_instances: Vec<_> = std::iter::empty()
       .chain(spartan_dag.stage2_verifier_instances(&mut state_manager))
       .chain(registers_dag.stage2_verifier_instances(&mut state_manager))
       .chain(ram_dag.stage2_verifier_instances(&mut state_manager))
       .chain(lookups_dag.stage2_verifier_instances(&mut state_manager))
       .collect();
   ```

2. **BatchedSumcheck::verify()** ([jolt-core/src/subprotocols/sumcheck.rs](jolt-core/src/subprotocols/sumcheck.rs)):
   - Sample same batching coefficients
   - Verify single batched proof
   - Each instance extracts its output claim
   - Store claims in accumulator

**Verifier instances are lightweight**:

- No witness data
- Just formula to compute expected output
- Example (Registers): Verifier computes expected fingerprint product

Repeat for Stages 3 and 4.

#### 7.5 Stage 5: Verify Batched Opening
**File**: [jolt-core/src/zkvm/dag/jolt_dag.rs:542](jolt-core/src/zkvm/dag/jolt_dag.rs#L542)

```rust
accumulator.borrow_mut().reduce_and_verify(
    &preprocessing.generators,
    &mut commitments_map,
    batched_opening_proof,
    &mut *transcript.borrow_mut(),
)?;
```

**What happens**:

1. Accumulator has all claimed evaluations from Stages 1-4
2. **Reduce** ([jolt-core/src/poly/opening_proof.rs](jolt-core/src/poly/opening_proof.rs)):
   - Sample same random coefficients
   - Compute expected combined evaluation
3. **Verify** using Dory PCS:
   - Check opening proof against combined commitment
   - Uses verifier setup (much smaller than prover setup)

**Verification time**: O(log n) operations for Dory (but dominated by ~40 G_T exponentiations, see Protocol 6)

---

### 8. Return Result
```rust
Ok(())  // Verification succeeded!
```

If any check fails (sumcheck consistency, opening proof, expected evaluation mismatch), returns `Err`.

---

## Key Architectural Insights

### The DAG Structure (Directed Acyclic Graph)

**Why "DAG"?**

- **Directed**: Data flows in one direction (stage 1 → 2 → 3 → 4 → 5)
- **Acyclic**: No circular dependencies between sumchecks
- **Graph**: Sumchecks (nodes) connected by polynomial evaluations (edges)

```
Stage 1: Spartan Outer (node)
  ↓ (edges: virtual openings = Az(r), Bz(r), Cz(r))
Stage 2: Product, Twist (Registers), Twist (RAM), Booleanity (Lookups) (nodes)
  ↓ (edges: virtual openings)
Stage 3: Inner, Evaluations, Read-checking (nodes)
  ↓ (edges: virtual openings)
Stage 4: Final Virtualizations (nodes)
  ↓ (edges: committed openings)
Stage 5: Batched Opening Proof (final node)
```

**Types of edges**:

- **Virtual edges** → Resolved by another sumcheck (creates graph dependency)
- **Committed edges** → Resolved by PCS opening proof in Stage 5 (terminal edges)

### StateManager's Role
Think of StateManager as a **data router**:

- Components write to accumulator
- Other components read from accumulator
- Transcript ensures verifier samples same challenges
- Proofs map stores all sumcheck proofs by stage

### Why Batching Matters
**Without batching**:

- 42 separate sumchecks = 42 proofs
- Verifier samples 42 sets of challenges
- More rounds of interaction

**With batching**:

- 4 batched proofs (Stages 1-4)
- Verifier samples one challenge per stage
- Much smaller proof size (fewer univariate polynomials)

### The Opening Accumulator Pattern
```rust
// Stage 1: Claim about Az(r)
accumulator.append_virtual(Az, point_r, claimed_eval);

// Stage 2: Prove Az(r) via product sumcheck
let (point, eval) = accumulator.get_virtual_opening(Az);
// ... run sumcheck to prove eval is correct ...

// After sumcheck: Claim about committed poly P(r')
accumulator.append_committed(P, point_r_prime, claimed_eval_prime);

// Stage 5: Prove P(r') via PCS
accumulator.reduce_and_prove(...);  // Batched opening proof
```

This creates the data flow graph!

---

## File Reference Quick Guide

### Core Orchestration

- [jolt-core/src/zkvm/dag/jolt_dag.rs](jolt-core/src/zkvm/dag/jolt_dag.rs): Main proof/verify loop
- [jolt-core/src/zkvm/dag/state_manager.rs](jolt-core/src/zkvm/dag/state_manager.rs): Data coordination
- [jolt-core/src/zkvm/dag/stage.rs](jolt-core/src/zkvm/dag/stage.rs): `SumcheckStages` trait

### Sumcheck Engine

- [jolt-core/src/subprotocols/sumcheck.rs](jolt-core/src/subprotocols/sumcheck.rs): Core sumcheck protocol
- [jolt-core/src/poly/unipoly.rs](jolt-core/src/poly/unipoly.rs): Univariate polynomials
- [jolt-core/src/poly/opening_proof.rs](jolt-core/src/poly/opening_proof.rs): Opening accumulator

### Five Components

- [jolt-core/src/zkvm/spartan/](jolt-core/src/zkvm/spartan/): R1CS constraints (Spartan)
- [jolt-core/src/zkvm/instruction_lookups/](jolt-core/src/zkvm/instruction_lookups/): Instruction execution (Shout)
- [jolt-core/src/zkvm/registers/](jolt-core/src/zkvm/registers/): Register consistency (Twist)
- [jolt-core/src/zkvm/ram/](jolt-core/src/zkvm/ram/): Memory consistency (Twist)
- [jolt-core/src/zkvm/bytecode/](jolt-core/src/zkvm/bytecode/): Bytecode checking (Shout)

### Cryptographic Primitives

- [jolt-core/src/poly/commitment/dory.rs](jolt-core/src/poly/commitment/dory.rs): Polynomial commitments
- [jolt-core/src/poly/multilinear_polynomial.rs](jolt-core/src/poly/multilinear_polynomial.rs): MLE operations
- [jolt-core/src/poly/eq_poly.rs](jolt-core/src/poly/eq_poly.rs): Equality polynomial

### Witness Generation

- [jolt-core/src/zkvm/witness.rs](jolt-core/src/zkvm/witness.rs): Converts trace → polynomials
- [tracer/src/emulator/](tracer/src/emulator/): RISC-V emulator
- [tracer/src/instruction/](tracer/src/instruction/): Instruction implementations

---

## Summary: The Big Picture

**Proving** = Show execution trace satisfies all constraints:

1. R1CS constraints (PC updates, linking)  Spartan
2. Instructions produced correct outputs  Shout lookups
3. Registers consistent  Twist
4. Memory consistent  Twist
5. Trace matches bytecode  Shout

**Verifying** = Check proof without re-execution:

1. Verify 4 batched sumcheck proofs  Fast (O(log n) each)
2. Verify 1 batched opening proof  Fast (O(log n))
3. Check all evaluation claims consistent  Algebra

**Why it's fast**:

- Lookup-centric: No expensive circuits
- Batching: Combines many sumchecks
- Structured constraints: Uniformity helps Spartan
- Transparent: No trusted setup needed

**The magic**: Verifier never sees the trace, but is convinced it's correct!

/newpage

# The Jolt Architecture: Instruction Decomposition

Before diving into the cryptographic protocols, let's understand **Jolt's core innovation**: decomposing massive CPU instruction lookup tables into tiny sub-tables. This is the architectural insight that makes Jolt possible.

## Why Instruction Decomposition?

**The problem**: A 64-bit CPU instruction like `ADD(a, b)` where both inputs are 64-bit values can conceptually be viewed as a lookup into a giant table with $2^{64} \times 2^{64} = 2^{128}$ entries. Storing or proving lookups into such a table is physically impossible.

**Jolt's insight**: This giant table is **decomposable**. A 64-bit `ADD` can be computed by performing 16 lookups into a tiny `ADD_4` table (operating on 4-bit chunks) and proving the results are wired together correctly. The `ADD_4` table has only $2^4 \times 2^4 \times 2^1 = 512$ entries—trivially small!

This section builds intuition for decomposition through worked examples, from simple bitwise operations to 64-bit arithmetic with carry chains.

---

## Decomposing Bitwise AND: The Simplest Case

Let's start with a simple instruction: 64-bit bitwise `AND`.

**The giant table** `AND_64(a, b)` would have $2^{128}$ entries.

**Decomposition strategy**:

1. **Split**: Decompose 64-bit inputs into sixteen 4-bit chunks:
   $$a = (a_{15}, a_{14}, \ldots, a_0), \quad b = (b_{15}, b_{14}, \ldots, b_0)$$

2. **Sub-lookups**: Perform 16 independent lookups into tiny `AND_4` table (256 entries):
   $$c_i = \text{AND}_4(a_i, b_i) \quad \text{for } i \in \{0, \ldots, 15\}$$

3. **Combine**: Concatenate results to form 64-bit output:
   $$c = (c_{15}, c_{14}, \ldots, c_0)$$

**Key observation**: For bitwise operations, the 16 sub-lookups are **completely independent**. There's no "wiring" between chunks—no consistency checks needed. The only check is **composition**: proving the final result is correctly assembled from the chunk results.

---

## A Worked Example: 2-bit Addition with Carry Chain

To understand **consistency checks** (wiring between chunks), let's walk through the simplest non-trivial example: adding two 2-bit numbers.

### The Setup

**Goal**: Prove $1 + 2 = 3$ using decomposition.

- Inputs: $a = 1 = 01_2$ (so $a_1=0, a_0=1$), $b = 2 = 10_2$ (so $b_1=1, b_0=0$)
- Initial carry: $\text{carry}_0 = 0$

**Sub-table**: `ADD_1` performs 1-bit full addition (3 inputs → 2 outputs):

$$(\text{sum}_i, \text{carry}_{i+1}) = \text{ADD}_1(a_i, b_i, \text{carry}_i)$$

> **Heuristic: What is a Carry Bit?**
>
> In grade school, when you add `8 + 5 = 13`, you write down `3` and "carry the `1`" to the next column. Binary addition works the same way.
>
> - **Case 1**: $0 + 1 = 1$ → sum is `1`, carry `0` to next column
> - **Case 2**: $1 + 1 = 2 = 10_2$ → sum is `0` (for current position), carry `1` to next column
>
> A **full adder** takes three inputs (two bits being added + carry-in from previous position) and produces two outputs (sum bit + carry-out to next position). This **carry chain** is the "wiring" Jolt must prove.

### The Complete ADD_1 Sub-Table

| $(a_i, b_i, \text{carry}_i)$ | $\text{sum}_i$ | $\text{carry}_{i+1}$ |
|------------------------------|----------------|----------------------|
| (0, 0, 0)                    | 0              | 0                    |
| (0, 0, 1)                    | 1              | 0                    |
| (0, 1, 0)                    | 1              | 0                    |
| (0, 1, 1)                    | 0              | 1                    |
| (1, 0, 0)                    | 1              | 0                    |
| (1, 0, 1)                    | 0              | 1                    |
| (1, 1, 0)                    | 0              | 1                    |
| (1, 1, 1)                    | 1              | 1                    |

Size: Only 8 entries ($2^3$)!

### Decomposed Lookups for 1 + 2

**Bit 0 (Least Significant Bit)**:

- Lookup: $\text{ADD}_1(a_0=1, b_0=0, \text{carry}_0=0)$
- Table lookup at index $(1, 0, 0)$: Result is $(\text{sum}_0=1, \text{carry}_1=0)$

**Bit 1 (Most Significant Bit)**:

- Lookup: $\text{ADD}_1(a_1=0, b_1=1, \text{carry}_1=0)$ ← **Uses carry from previous step!**
- Table lookup at index $(0, 1, 0)$: Result is $(\text{sum}_1=1, \text{carry}_2=0)$

### Proving Correctness: Composition + Consistency

The **combination function** $g$ performs two types of checks:

1. **Composition**: Prove final result correctly assembled from chunk results
   $$c = (\text{sum}_1, \text{sum}_0) = (1, 1) = 11_2 = 3_{10} \quad \checkmark$$

2. **Consistency**: Prove wiring between chunks is correct
   $$\text{carry-out from Bit 0} \stackrel{?}{=} \text{carry-in to Bit 1}$$
   $$0 = 0 \quad \checkmark$$

**Key insight**: Jolt doesn't prove `ADD_2(1, 2) = 3` directly. It proves:

- Two specific lookups were performed into the correct `ADD_1` table
- The carry bit was correctly wired between them
- Because the sub-table is correct and the wiring is correct, the final result **must** be correct!

---

## Scaling to 64-bit Operations

The same logic scales to real CPU instructions.

> **Heuristic: Notation `OP_C`**
>
> Throughout Jolt papers and code:
> - **OP**: The operation (`ADD`, `XOR`, `SLT`, etc.)
> - **C**: The **chunk size** in bits
>
> So `ADD_4` = "lookup table for ADD operation on 4-bit inputs"
> Similarly, `XOR_8` = "8-bit XOR operation"

### 64-bit Addition with ADD_4 Sub-Table

**Sub-table**: `ADD_4` performs 4-bit full addition:

$$(\text{sum}_i, \text{carry}_{i+1}) = \text{ADD}_4(a_i, b_i, \text{carry}_i)$$

- **Inputs**: $a_i$ (4 bits), $b_i$ (4 bits), $\text{carry}_i$ (1 bit)
- **Outputs**: $\text{sum}_i$ (4 bits), $\text{carry}_{i+1}$ (1 bit)
- **Size**: $2^4 \times 2^4 \times 2^1 = 512$ entries (still tiny!)

**Decomposition of ADD_64**:

1. **16 Correct Lookups**: One `ADD_4` lookup per 4-bit chunk
2. **15 Consistency Checks**: For each chunk $i$ from 1 to 15:
   $$\text{carry-in to chunk } i = \text{carry-out from chunk } i-1$$

**Formal notation**: Let $\text{carry}_0 = 0$. For $i \in \{0, \ldots, 15\}$:

$$(\text{sum}_i, \text{carry}_{i+1}) = \text{ADD}_4(a_i, b_i, \text{carry}_i)$$

Final result: $c = \sum_{i=0}^{15} \text{sum}_i \cdot 2^{4i}$ with overflow bit $\text{carry}_{16}$.

### Other Instructions

The decomposition pattern applies broadly:

- **Bitwise operations** (`XOR`, `AND`, `OR`): 16 independent sub-lookups, **zero consistency checks**
  $$c = \sum_{i=0}^{15} c_i \cdot 2^{4i}$$

- **Shifts/rotates**: Consistency checks ensure bits shifted out of chunk $i$ correctly enter chunk $i+1$

- **Comparisons** (`SLT` - "set less than"): More complex consistency checks
  - Each chunk needs "equality-so-far" flag from previous chunks
  - Sub-lookup: $\text{SLT}_4(a_i, b_i, \text{eq}_{i+1})$ outputs result bit and updated $\text{eq}_i$ flag
  - Consistency: wiring these equality flags together

**Core pattern**: Every instruction = sub-table lookups + composition + consistency checks

---

## Formalizing Decomposability: The Combination Function

This decomposition strategy is formalized by **Lasso's decomposable table** definition.

A table $T$ is decomposable if its value can be computed from $M$ sub-tables via a **combination function** $g$:

$$T(i) = g(T_1(i_1), \ldots, T_M(i_M))$$

where $i_j$ is the part of index $i$ corresponding to sub-table $j$.

### For ADD_64:

- **Main table**: $T = \text{ADD}_{64}$, indexed by $(a, b)$ (two 64-bit inputs)
- **Sub-tables**: $M = 16$ lookups into `ADD_4` sub-table
- **Sub-lookup results**: $v = (v_0, \ldots, v_{15})$ where $v_j = (\text{sum}_j, \text{carry-out}_j)$
- **Combination function** $g$: Checks composition + consistency

> **From Predicate to Polynomial**
>
> While intuitively $g$ is a true/false check, cryptographically it must be a **polynomial**. Checks like $X = Y$ become polynomial constraints $X - Y = 0$.
>
> **Combining multiple constraints**: Use random linear combination. Verifier provides random challenge $\gamma \in \mathbb{F}$, prover must show:
>
> $$C_1 + \gamma \cdot C_2 + \gamma^2 \cdot C_3 + \cdots + \gamma^{k-1} \cdot C_k = 0$$
>
> By Schwartz-Zippel Lemma: if any original constraint is non-zero, this combined polynomial is non-zero with high probability. The combination function $g$ is exactly this randomly combined polynomial.

### Explicit Polynomial Constraints for ADD_64

**1. Composition Constraint**:

$$C_{\text{compose}} = c - \sum_{i=0}^{15} \text{sum}_i \cdot 2^{4i} = 0$$

**2. Consistency Constraints** (15 carry-chain wirings + 1 initial carry):

$$C_{\text{consistency}, -1} = \text{carry-in}_0 - 0 = 0$$
$$C_{\text{consistency}, i} = \text{carry-in}_{i+1} - \text{carry-out}_i = 0 \quad \text{for } i \in \{0, \ldots, 14\}$$

**Final combination function** (17 constraints combined with random $\gamma$):

$$g = C_{\text{compose}} + \gamma \cdot C_{\text{consistency}, -1} + \sum_{i=0}^{14} \gamma^{i+2} \cdot C_{\text{consistency}, i}$$

Proving $g = 0$ proves all 17 underlying constraints hold simultaneously!

---

## Application to RISC-V

Jolt applies this decomposition to the **RV64I instruction set**:

**Decomposable instructions**:
- **Arithmetic**: `ADD`, `SUB`
- **Bitwise**: `XOR`, `OR`, `AND`
- **Shifts**: `SLL`, `SRL`, `SRA` (logical/arithmetic shifts)
- **Comparisons**: `SLT`, `SLTU` (signed/unsigned "set less than")

**Key insight**: All instructions **share the same sub-tables**!

> **Heuristic: One Sub-Table to Rule Them All**
>
> Instead of separate tables (`ADD_4`, `XOR_4`, `SLT_4`), Jolt combines them into a **single unified sub-table** with a `sub_opcode` field:
>
> $$\text{output} = \text{MASTER\_SUB\_TABLE}(\text{sub\_opcode}, \text{input}_1, \text{input}_2, \ldots)$$
>
> - `sub_opcode = 1` → `ADD_4`
> - `sub_opcode = 2` → `XOR_4`
> - etc.
>
> This amortizes cryptographic costs—Lasso only proves lookups into one (slightly larger) master table across the entire instruction set!

**Non-decomposable instructions**:
- **Control flow** (`JAL`, `BEQ`): Handled by R1CS wiring (PC update constraints)
- **Memory** (`LOAD`, `STORE`): Handled by separate Twist memory consistency protocol

> **Heuristic: Which Instructions Use Which Proving Method?**
>
> Lookup tables handle **computational instructions**—operations on register values that write results (e.g., `ADD`, `XOR`, `SLT`).
>
> **Not all instructions fit this pattern**:
> - **Memory Instructions** (`LOAD`, `STORE`): Different signatures (base address + offset). Correctness proven by **Twist** memory consistency protocol—ensures `LOAD` returns value from most recent `STORE` to that address.
> - **Control Flow** (`JAL`, `BEQ`): Modify Program Counter. Correctness proven by **R1CS wiring** (Spartan)—verifies PC updates follow instruction logic.
>
> **Division of labor**: Jolt's lookup argument handles the "heavy lifting" of CPU arithmetic/logic; specialized protocols handle the rest of machine behavior.

---

## Connection to Code

This theoretical decomposition maps directly to Jolt's implementation:

**Sub-table MLEs**: Each instruction sub-table (like `ADD_4`, `XOR_4`) is defined by a struct implementing the `JoltLookupTable` trait:

```rust
// jolt-core/src/zkvm/instruction/mod.rs
pub trait JoltLookupTable<F: JoltField> {
    // This is T~(r) in the math!
    // Evaluates the table's MLE at random point r
    fn evaluate_mle(&self, r: &[F]) -> F;
}
```

**Key optimization**: For structured tables like `ADD_4`, this function doesn't read from a stored table. It uses algebraic properties to compute evaluations in $O(\log(\text{table size}))$ time. This allows the verifier to work with $2^{128}$-sized tables without ever materializing them!

**Example implementations**:
- `jolt-core/src/zkvm/instruction/add.rs`: Defines `ADD` decomposition
- `jolt-core/src/zkvm/instruction/xor.rs`: Defines `XOR` decomposition
- Each implements `evaluate_mle()` for its specific sub-table MLE

---

## Summary: The Decomposition Philosophy

**Jolt's innovation**: Replace arithmetization (expressing CPU logic as circuits) with pre-computed lookup tables. But naive lookup tables are impossibly large.

**Solution**: Prove the giant tables are **decomposable**:
- Split into tiny sub-tables (512 entries vs $2^{128}$)
- Prove sub-lookups correct (via Lasso/Shout protocols)
- Prove results wired together correctly (composition + consistency checks)

**Result**: Fast prover dominated by cryptographic lookup machinery, not complex circuit-specific arithmetization!

/newpage

# Part 0: Mathematical Foundation - The Cryptographic Protocols

Before diving into the implementation details, it's essential to understand the **mathematical protocols** underlying Jolt. The proof system uses six key cryptographic primitives, each solving a specific mathematical problem:

1. **Spartan**: Proves R1CS constraints (VM wiring) using sumcheck over multilinear extensions
2. **SPARK**: Efficient commitments/openings for sparse polynomials via dense (vals, idxs) representation
3. **Lasso**: Proves lookup table membership via sparse "hit" vectors and sumcheck
4. **Twist**: Proves memory consistency via grand product arguments over fingerprints
5. **Shout**: Offline memory checking for bytecode and instruction lookups
6. **Dory**: Polynomial commitment scheme providing cryptographic binding

Let's examine the mathematics of each protocol in detail.

---

## Protocol 1: Spartan - Proving R1CS Constraints

**Spartan** is a transparent SNARK (no trusted setup) designed to efficiently prove **Rank-1 Constraint Systems (R1CS)**. Its key innovation is a **time-optimal prover** whose work scales with the number of non-zero elements in the constraint matrices, not their total size.

### What is R1CS?

An R1CS is a system of constraints where each constraint $i$ has the form:

$$(A_i \cdot z) \cdot (B_i \cdot z) - (C_i \cdot z) = 0$$

Where:
- $z \in \mathbb{F}^n$: The witness vector (contains all execution trace data)
- $A_i, B_i, C_i \in \mathbb{F}^n$: Vectors defining the $i$-th constraint

To prove a computation is correct, we must show this holds for all $m$ constraints.

**Matrix formulation**: The collection of constraint vectors forms three $m \times n$ matrices $A, B, C$:

$$Az \circ Bz = Cz$$

Where $\circ$ denotes element-wise multiplication.

### From R1CS to Polynomials

Spartan's first step is transforming R1CS into polynomial language:

**1. Multilinear Extensions (MLEs)**:
- Each matrix $(A, B, C)$ becomes a polynomial $\widetilde{A}(x,y), \widetilde{B}(x,y), \widetilde{C}(x,y)$
  - $x \in \{0,1\}^{\log m}$: Row index (constraint number)
  - $y \in \{0,1\}^{\log n}$: Column index (witness element)
- Witness $z$ becomes polynomial $\widetilde{z}(y)$

**2. Expressing Dot Products**:

The dot product for row $x$ (constraint $x$):
$$A_x \cdot z = \sum_{y \in \{0,1\}^{\log n}} \widetilde{A}(x, y) \cdot \widetilde{z}(y)$$

**3. The R1CS Polynomial Identity**:

The constraint $Az \circ Bz = Cz$ becomes:

$$\forall x \in \{0,1\}^{\log m}: \left( \sum_y \widetilde{A}(x,y) \widetilde{z}(y) \right) \cdot \left( \sum_y \widetilde{B}(x,y) \widetilde{z}(y) \right) - \sum_y \widetilde{C}(x,y) \widetilde{z}(y) = 0$$

### The Spartan Reduction: Three Sumchecks

Checking every constraint naively would be too slow. Spartan uses randomness to combine all constraints into a single probabilistic check.

**Step 1: Random Challenge**

Verifier sends random point $r_x \in \mathbb{F}^{\log m}$. Prover must now prove the R1CS equation holds for this *single random "virtual" constraint*:

$$\left( \sum_{y} \widetilde{A}(r_x, y) \widetilde{z}(y) \right) \cdot \left( \sum_{y} \widetilde{B}(r_x, y) \widetilde{z}(y) \right) - \left( \sum_{y} \widetilde{C}(r_x, y) \widetilde{z}(y) \right) = 0$$

**Step 2: Define Three Sums**

Let:
- $S_A = \sum_{y} \widetilde{A}(r_x, y) \widetilde{z}(y)$
- $S_B = \sum_{y} \widetilde{B}(r_x, y) \widetilde{z}(y)$
- $S_C = \sum_{y} \widetilde{C}(r_x, y) \widetilde{z}(y)$

The equation becomes: $S_A \cdot S_B - S_C = 0$

**Step 3: Three Parallel Sumchecks**

Prover runs **three sumcheck protocols** to convince verifier of claimed values $S_A, S_B, S_C$. After sum-checks, verifier simply checks: $S_A \cdot S_B - S_C \stackrel{?}{=} 0$.

**Output of Sumchecks**: Each sumcheck reduces its summation claim to evaluation claims at random points $r_y$ (generated during sumcheck):

- Sparse matrix claims: $\widetilde{A}(r_x, r_y) = v_A$, $\widetilde{B}(r_x, r_y) = v_B$, $\widetilde{C}(r_x, r_y) = v_C$
- Dense witness claim: $\widetilde{z}(r_y) = v_z$

**In Jolt**: These claims are resolved via SPARK (for sparse matrices) and Dory (for dense witness).

> **Heuristic: Why Lookups Instead of Pure R1CS?**
>
> R1CS is universal—it can handle both wiring and values. So why bother with lookups?
>
> **Answer**: Performance. R1CS uses field arithmetic (addition/multiplication). For algebraic operations like `c = a + b`, this is perfect (one constraint). But for **bitwise operations** CPUs perform constantly, it's catastrophically inefficient.
>
> **Example**: A single 64-bit `XOR` instruction in pure R1CS:
> 1. **Decompose**: Break inputs into 64 bits each → requires constraints
> 2. **Arithmetize**: For each bit pair, enforce XOR logic: $c_i = a_i + b_i - 2 \cdot a_i \cdot b_i$
>
> **Result**: One `XOR` explodes into **64+ R1CS constraints**. For a real program, this "constraint explosion" makes proofs enormous and slow.
>
> **Jolt's insight**: XOR logic is fixed. Instead of re-proving it with 64+ constraints every time, prove it via **16 lookups** into a pre-computed `XOR_4` table (256 entries). Vastly more efficient!

---

## Protocol 2: SPARK - Efficient Sparse Polynomial Commitments

**SPARK** (Sparse Polynomial Commitment) is a compiler that makes polynomial commitment schemes efficient for **sparse polynomials**. It's critical for Spartan's time-optimal prover.

### The Problem: Sparse Polynomials

In Jolt's R1CS, the constraint matrices $A, B, C$ are **extremely sparse**:
- Domain size: $T \times T$ (e.g., $1024 \times 1024 = 1,048,576$ entries)
- Non-zeros: ~30 per row (e.g., $30 \times 1024 = 30,720$ entries)
- **97% sparse!**

Committing to a sparse polynomial with a standard PCS costs $O(T^2)$ - wasteful when only $O(T)$ entries are non-zero!

### SPARK's Solution: The (vals, idxs) Representation

**Step 1: Deconstruct the Sparse Polynomial**

Let $\widetilde{P}(x)$ be a sparse multilinear polynomial over $k$ variables with $s$ non-zero entries.

SPARK represents $\widetilde{P}$ using two **dense** vectors of size $s$:

1. **vals** = $(v_0, \ldots, v_{s-1})$: The non-zero values
2. **idxs** = $(i_0, \ldots, i_{s-1})$: The $k$-bit indices where those values occur

**Example**: Sparse polynomial over $\{0,1\}^{10}$ (domain size 1024) with 30 non-zeros:
- Store as: `vals[30]` + `idxs[30]` (each index is 10 bits)
- Commitment cost: $O(30)$ instead of $O(1024)$

**Step 2: Commit to Dense Representations**

Decompose `idxs` into $k$ bit-vectors, one per bit position. Find MLEs:
- $\widetilde{vals}(y)$ over domain $\{0,1\}^{\log s}$
- $\widetilde{idxs}_0(y), \ldots, \widetilde{idxs}_{k-1}(y)$ over domain $\{0,1\}^{\log s}$

Commit to these $k+1$ small, dense polynomials using Dory.

**Step 3: Transform Evaluation to Summation**

**Claim to prove**: $\widetilde{P}(r) = v$ for random point $r \in \mathbb{F}^k$

**SPARK identity**: The value of sparse polynomial $\widetilde{P}$ at any point $x$ equals:

$$\widetilde{P}(x) = \sum_{j \in \{0,1\}^{\log s}} \widetilde{vals}(j) \cdot \text{eq}(\widetilde{idxs}(j), x)$$

Where $\text{eq}(A, B) = \prod_{i=0}^{k-1} (A_i B_i + (1-A_i)(1-B_i))$ is 1 if $A = B$ on Boolean hypercube, else 0.

**Intuition**: Sum over all non-zero entries. The $\text{eq}$ function selects the entry at location $x$ (if it exists).

**Step 4: Run Sumcheck**

Evaluation claim $\widetilde{P}(r) = v$ becomes summation claim:

$$v \stackrel{?}{=} \sum_{j \in \{0,1\}^{\log s}} \widetilde{vals}(j) \cdot \prod_{i=0}^{k-1} \left(\widetilde{idxs}_i(j) r_i + (1-\widetilde{idxs}_i(j))(1-r_i)\right)$$

Run sumcheck protocol over this polynomial.

**Step 5: Final Openings**

Sumcheck reduces to evaluation claims at random point $r' \in \mathbb{F}^{\log s}$:
- $\widetilde{vals}(r') = v'_{vals}$
- $\widetilde{idxs}_0(r') = v'_0, \ldots, \widetilde{idxs}_{k-1}(r') = v'_{k-1}$

Prove these using **Dory opening proofs** against initial commitments.

**Result**: Prover work is $O(s \log s)$, not $O(2^k)$!

---

## Protocol 3: Lasso - Lookup Arguments for Structured Tables

**Lasso** is a lookup argument proving that a set of $m$ values in the execution trace are all valid entries in a large predefined table $T$ of size $N$. It's the engine that proves instruction semantics in Jolt.

### The Lookup Problem

**Input**:
- List of $m$ lookup indices: $(idx_1, \ldots, idx_m)$
- List of $m$ claimed values: $(val_1, \ldots, val_m)$
- Table $T$ of size $N$ (possibly $N \gg m$)

**Claim**: $\forall i \in [1,m]: T[idx_i] = val_i$

**Challenge**: For 64-bit operations, $N = 2^{128}$ (two 64-bit operands) - table is too large to materialize!

### Lasso's Approach: Sparse "Hit" Vectors

> **Heuristic: A Lookup Trace is a Sparse "Hit" Vector**
>
> Imagine table $T$ with $2^{64}$ entries. A program might only perform a few thousand lookups. We can represent this access pattern with a "hit" vector of size $2^{64}$ that is **almost entirely zeros**, with non-zero entries only at the accessed indices.
>
> This vector is **extremely sparse** (maybe 0.0001% non-zero). Lasso exploits this sparsity—prover's work scales with number of lookups $m$, not table size $N$!

**Step 1: Create Hit Vector**

Define sparse polynomial $\widetilde{E}(y)$ over table's index space:

$$\widetilde{E}(y) = \begin{cases}
k & \text{if index } y \text{ was looked up } k \text{ times} \\
0 & \text{otherwise}
\end{cases}$$

**Example**: For $m = 1000$ lookups into table of size $N = 2^{128}$:
- $\widetilde{E}$ has ~1000 non-zero entries
- Domain size: $2^{128}$
- **Astronomically sparse**: $1000 / 2^{128} \approx 10^{-36}$

**Step 2: The Lasso Sumcheck Identity**

The lookup claim becomes:

$$\sum_{y \in \{0,1\}^{\log N}} \widetilde{E}(y) \cdot \left(\widetilde{T}(y) - \widetilde{a}(y)\right) = 0$$

Where:
- $\widetilde{T}(y)$: MLE of table (verifier-computed if table has efficient formula)
- $\widetilde{a}(y)$: MLE of claimed values (dense polynomial over lookups)

**Intuition**: For every accessed index ($\widetilde{E}(y) \neq 0$), the claimed value ($\widetilde{a}(y)$) must match the table value ($\widetilde{T}(y)$).

**Step 3: Sumcheck Over Table Space**

Run sumcheck over $\log N$ variables. Key insight: Due to sparsity of $\widetilde{E}$, prover work is $O(m \log m)$, not $O(N)$!

**Output**: Evaluation claims for sparse polynomials:
- $\widetilde{E}(r) = v_E$
- $\widetilde{a}(r) = v_a$
- (Plus wiring polynomials $\widetilde{M}_{V,j}(r) = v_M$ for decomposed instructions)

These sparse evaluation claims are then resolved via **SPARK**.

### Lasso for Decomposable Tables

For Jolt's 64-bit operations, Lasso uses **table decomposition**:

A 64-bit ADD with operands $(a, b)$ decomposes into 16 × 4-bit ADDs:
- $a = (a_0, a_1, \ldots, a_{15})$ where each $a_i$ is 4 bits
- $b = (b_0, b_1, \ldots, b_{15})$ where each $b_i$ is 4 bits

**Decomposed Lasso identity**:

$$\sum_{y \in \{0,1\}^{128}} \widetilde{E}(y) \cdot \left( \left( \sum_{j=1}^{16} \widetilde{T}_j(y_j) \cdot \widetilde{M}_{V,j}(y) \right) - \widetilde{a}(y) \right) = 0$$

Where:
- $\widetilde{T}_j(y_j)$: MLE of $j$-th 4-bit sub-table (efficiently evaluable)
- $\widetilde{M}_{V,j}(y)$: Wiring polynomial routing correct chunk values to sub-table $j$
- Sum over $2^{128}$ space made tractable by **prefix-suffix sumcheck algorithm**

---

## Protocol 4: Twist - Memory Consistency via Grand Products

**Twist** proves memory consistency: that every read returns the value of the most recent write to that address (or initial value if never written).

### The Memory Checking Problem

**Goal**: Verify that a sequence of memory operations (reads/writes) is consistent.

**Input**: Execution trace with memory operations

Each operation is a tuple $(addr, ts, val, op)$ where:
- $addr \in [0, K-1]$: **Memory address** being accessed (e.g., register number 0-63 for registers, or RAM address)
- $ts \in [0, T-1]$: **Timestamp** (cycle number when operation occurred)
- $val \in \mathbb{F}$: **Value** being read or written (64-bit value encoded in field)
- $op \in \{\text{read}, \text{write}\}$: **Operation type**

**Concrete example** (register file with 3 cycles):
```
Cycle 0: WRITE register_5 ← 100    → (addr=5, ts=0, val=100, op=write)
Cycle 1: READ  register_5 → 100    → (addr=5, ts=1, val=100, op=read)
Cycle 2: WRITE register_5 ← 200    → (addr=5, ts=2, val=200, op=write)
```

**Time-ordered trace** (as executed):
```
[(5, 0, 100, write), (5, 1, 100, read), (5, 2, 200, write)]
```

**Address-ordered trace** (sorted by address, then time):
```
[(5, 0, 100, write), (5, 1, 100, read), (5, 2, 200, write)]
```
(Same in this case since only one address!)

**Claim**: For every read at address $a$ and time $t$, the value read equals the value of the most recent write to $a$ before time $t$.

### The Grand Product Approach

**Key Insight**: If time-ordered and address-ordered traces contain the same operations, they're **permutations** of each other.

**Step 1: Create Two Traces**

1. **Time-ordered**: Operations as they occurred during execution
2. **Address-ordered**: Same operations sorted by address, then timestamp

**Step 2: Fingerprint Each Operation**

For random challenges $\gamma_1, \gamma_2, \gamma_3$ (from transcript), create a **fingerprint** (hash) of each operation:

$$f(i) = \gamma_1 \cdot \text{address}(i) + \gamma_2 \cdot \text{timestamp}(i) + \gamma_3 \cdot \text{value}(i)$$

**What each function extracts**:
- $\text{address}(i)$: The memory address from operation $i$'s tuple (first element)
  - Example: For operation $i = (5, 1, 100, \text{read})$, $\text{address}(i) = 5$
- $\text{timestamp}(i)$: The cycle number from operation $i$'s tuple (second element)
  - Example: $\text{timestamp}(i) = 1$
- $\text{value}(i)$: The data value from operation $i$'s tuple (third element)
  - Example: $\text{value}(i) = 100$

**Concrete calculation** for our example operation $(5, 1, 100, \text{read})$:
$$f(i) = \gamma_1 \cdot 5 + \gamma_2 \cdot 1 + \gamma_3 \cdot 100$$

If $\gamma_1 = 0.234, \gamma_2 = 0.876, \gamma_3 = 0.456$ (random challenges), then:
$$f(i) = 0.234 \cdot 5 + 0.876 \cdot 1 + 0.456 \cdot 100 = 1.17 + 0.876 + 45.6 = 47.646$$

**Step 3: Grand Product Identity**

If traces are permutations:

$$\prod_{i=0}^{T-1} f_{\text{time}}(i) = \prod_{j=0}^{T-1} f_{\text{addr}}(j)$$

**Security**: By Schwartz-Zippel, if traces differ, fingerprints will differ with overwhelming probability.

### Detailed Fingerprint Construction

The actual Twist implementation uses a **more robust fingerprint** with an additional constant term:

**Full fingerprint formula** (used in Jolt's implementation):

$$f_{\text{time}}(i) = \gamma_0 + \gamma_1 \cdot \text{addr}(i) + \gamma_2 \cdot \text{ts}(i) + \gamma_3 \cdot \text{val}(i)$$

**Notation clarification**:
- $\text{addr}(i)$ is shorthand for $\text{address}(i)$ (memory address from operation $i$)
- $\text{ts}(i)$ is shorthand for $\text{timestamp}(i)$ (cycle number from operation $i$)
- $\text{val}(i)$ is shorthand for $\text{value}(i)$ (data value from operation $i$)

**Challenge sources**:
- $\gamma_0, \gamma_1, \gamma_2, \gamma_3 \in \mathbb{F}$: Random challenges sampled from Fiat-Shamir transcript
- Verifier samples same challenges from same transcript (deterministic hash)

**Concrete example** with $\gamma_0 = 0.123$:

For operation $(5, 1, 100, \text{read})$:
$$f(i) = 0.123 + 0.234 \cdot 5 + 0.876 \cdot 1 + 0.456 \cdot 100 = 48.169$$

**Why four challenges?**
- $\gamma_0$: Constant offset ensures fingerprint is **never zero** (avoids division by zero in ratios)
- $\gamma_1$: Ensures different **addresses** produce different fingerprints
- $\gamma_2$: Ensures different **timestamps** produce different fingerprints
- $\gamma_3$: Ensures different **values** produce different fingerprints

**Security** (Schwartz-Zippel lemma):
- If two operations differ in ANY component (address, timestamp, or value)
- Then their fingerprints will differ with overwhelming probability $1 - \frac{4}{\lvert \mathbb{F} \rvert} \approx 1 - 2^{-250}$
- This makes it cryptographically infeasible to forge matching fingerprints for different operations

### The Fractional Grand Product Identity

The actual Twist verification uses a **product over ratios** to prove memory consistency:

**Product identity**:

$$\prod_{i=0}^{T-1} \frac{f_{\text{init}}(i) + f_{\text{write}}(i)}{f_{\text{read}}(i) + f_{\text{final}}(i)} = 1$$

Where:
- $f_{\text{init}}(i)$: Fingerprint of initial value at address $i$ (before any writes)
- $f_{\text{write}}(i)$: Fingerprint of write operations to address $i$
- $f_{\text{read}}(i)$: Fingerprint of read operations from address $i$
- $f_{\text{final}}(i)$: Fingerprint of final value at address $i$ (after all operations)

**Intuition**:
- Numerator: "Credits" - initial values + values written
- Denominator: "Debits" - values read + final values
- Product = 1 means credits exactly match debits for every address

**Worked example** using our 3-cycle register trace:

```
Cycle 0: WRITE register_5 ← 100
Cycle 1: READ  register_5 → 100
Cycle 2: WRITE register_5 ← 200
```

For register_5:
- **Initial value**: 0 (registers start at 0)
- **Writes**: 100 (cycle 0), 200 (cycle 2)
- **Reads**: 100 (cycle 1)
- **Final value**: 200 (after all operations)

**Fingerprint calculations** (using $\gamma_0 = 0.1, \gamma_1 = 0.2, \gamma_2 = 0.3, \gamma_3 = 0.4$):

$$f_{\text{init}} = 0.1 + 0.2 \cdot 5 + 0.3 \cdot 0 + 0.4 \cdot 0 = 1.1$$
$$f_{\text{write\_0}} = 0.1 + 0.2 \cdot 5 + 0.3 \cdot 0 + 0.4 \cdot 100 = 41.1$$
$$f_{\text{write\_2}} = 0.1 + 0.2 \cdot 5 + 0.3 \cdot 2 + 0.4 \cdot 200 = 81.7$$
$$f_{\text{read\_1}} = 0.1 + 0.2 \cdot 5 + 0.3 \cdot 1 + 0.4 \cdot 100 = 41.4$$
$$f_{\text{final}} = 0.1 + 0.2 \cdot 5 + 0.3 \cdot 2 + 0.4 \cdot 200 = 81.7$$

**Product check**:
$$\frac{f_{\text{init}} + f_{\text{write\_0}} + f_{\text{write\_2}}}{f_{\text{read\_1}} + f_{\text{final}}} = \frac{1.1 + 41.1 + 81.7}{41.4 + 81.7} = \frac{123.9}{123.1} \approx 1$$

(Small discrepancy due to rounding; exact in finite field arithmetic!)

**Why this works**:
- Numerator sums all values **entering** the memory location (init + all writes)
- Denominator sums all values **leaving** the memory location (all reads + final state)
- If reads are consistent with writes, these must balance!

### Twist Optimization: Incremental Formulation

Instead of computing the full fractional product directly, Twist uses **increment polynomials**:

**Define $\Delta$ polynomials**:

For each memory location $k$:

$$\Delta_k(j) = \begin{cases}
\frac{\gamma_0 + \gamma_1 \cdot k + \gamma_2 \cdot j + \gamma_3 \cdot (\text{init\_val}_k + \text{write\_increment}_k(j))}{\gamma_0 + \gamma_1 \cdot k + \gamma_2 \cdot j + \gamma_3 \cdot (\text{read\_val}_k(j) + \text{final\_val}_k)} & \text{if location } k \text{ accessed at cycle } j \\
1 & \text{otherwise}
\end{cases}$$

**Key property**: The product of all increments equals 1:

$$\prod_{j=0}^{T-1} \prod_{k=0}^{K-1} \Delta_k(j) = 1$$

**This is equivalent to the fractional identity** because:
1. For each accessed location, the increment represents the ratio of (input + writes) / (reads + output)
2. For non-accessed locations, $\Delta = 1$ (no contribution)
3. Product accumulates all ratios across time and space

### Sum-to-Product Transformation

Products are verified via sumchecks using the logarithm identity:

$$\prod_{i} x_i = 1 \iff \sum_{i} \log(x_i) = 0$$

**Twist sumcheck structure**:

$$\sum_{j \in \{0,1\}^{\log T}} \sum_{k \in \{0,1\}^{\log K}} \widetilde{\text{ra}}(j, k) \cdot \log\left(\Delta_k(j)\right) = 0$$

Where:
- $\widetilde{\text{ra}}(j, k)$: Read/write address polynomial (one-hot: 1 if location $k$ accessed at cycle $j$)
- $\log(\Delta_k(j))$: Logarithm of increment (computed as $\log(\text{numerator}) - \log(\text{denominator})$)

**Challenge**: Logarithms don't exist in finite fields! Solution: Use **tower of extensions** or work with rational functions directly in the sumcheck.

### Twist in Jolt: Two Stages, Two Instances

Jolt uses Twist in **two separate stages** of the verification DAG, with **two independent instances** (one for registers, one for RAM).

**Two Twist instances**:

1. **Registers** ([jolt-core/src/zkvm/registers/](../jolt-core/src/zkvm/registers/)):
   - $K = 64$ memory locations (32 RISC-V + 32 virtual registers)
   - Three sumchecks total (across two stages)

2. **RAM** ([jolt-core/src/zkvm/ram/](../jolt-core/src/zkvm/ram/)):
   - Variable $K$ (determined by program's memory usage)
   - Three sumchecks total (across two stages)

**Stage 2: Read-Checking and Write-Checking (Time-Ordered)**

The first stage of Twist verification happens in **Stage 2 of Jolt's DAG**.

**Read-checking sumcheck** (example for register rs1):

$$\sum_{j \in \{0,1\}^{\log T}} \sum_{k \in \{0,1\}^{\log K}} \widetilde{\text{ra}}_{\text{rs1}}(j, k) \cdot \left( f_{\text{read}}(j, k) - f_{\text{written}}(j, k) \right) = 0$$

Where:
- $\widetilde{\text{ra}}_{\text{rs1}}(j, k)$: Read address polynomial (1 if cycle $j$ reads from register $k$, else 0)
- $f_{\text{read}}$: Fingerprint of read operation: $\gamma_0 + \gamma_1 k + \gamma_2 j + \gamma_3 \cdot \text{val\_read}$
- $f_{\text{written}}$: Fingerprint of last write: $\gamma_0 + \gamma_1 k + \gamma_2 j_{last\_write} + \gamma_3 \cdot \text{val\_written}$

**Intuition**: For each register read (at cycle $j$, address $k$), prove the value read matches the value of the most recent write.

**Write-checking sumcheck** (similarly structured):

$$\sum_{j \in \{0,1\}^{\log T}} \sum_{k \in \{0,1\}^{\log K}} \widetilde{\text{wa}}(j, k) \cdot \text{(write fingerprint formula)} = 0$$

Where $\widetilde{\text{wa}}(j, k)$ is the write address polynomial.

**Stage 2 output**:
- Random challenge points from each sumcheck
- Claimed evaluations of various polynomials at those points
- These become inputs to Stage 3

**Stage 3: Evaluation Sumchecks (Address-Ordered)**

The second stage of Twist happens in **Stage 3 of Jolt's DAG**, using the challenge points from Stage 2.

**Evaluation sumcheck** (proves final state = initial + all increments):

$$\sum_{j' \in \{0,1\}^{\log T}} \widetilde{\text{wa}}(\vec{r}_{\text{addr}}, j') \cdot \widetilde{\text{inc}}(\vec{r}_{\text{addr}}, j') \cdot \widetilde{\text{LT}}(j', \vec{r}_{\text{cycle}}) = \text{val\_claim} - \text{init\_eval}$$

Where:
- $\vec{r}_{\text{addr}}$, $\vec{r}_{\text{cycle}}$: Random challenge points from Stage 2 sumchecks
- $\widetilde{\text{inc}}(\vec{r}_{\text{addr}}, j')$: Increment polynomial (value written - value before)
- $\widetilde{\text{LT}}(j', \vec{r}_{\text{cycle}})$: Less-than polynomial (ensures we only count increments before challenge cycle)
- $\text{val\_claim}$: Claimed evaluation from Stage 2
- $\text{init\_eval}$: Initial memory value at challenge address

**Why split across two stages?**

The Twist protocol naturally decomposes into:
1. **Time-ordered verification** (Stage 2): Check consistency of reads/writes as they occur during execution
2. **Address-ordered verification** (Stage 3): Check that final state equals initial state plus all increments

This decomposition:
- Allows batching Stage 2 Twist sumchecks with other Stage 2 sumchecks (Spartan, instruction lookups)
- Allows batching Stage 3 Twist sumchecks with other Stage 3 sumchecks
- Reduces rounds of interaction with verifier (fewer Fiat-Shamir challenges overall)
- Enables better memory efficiency (don't need all data structures simultaneously)

---

## Protocol 5: Shout - Offline Memory Checking

**Shout** is an offline memory checking protocol that proves correct reads from **pre-committed, read-only memory**. Unlike Lasso (which handles dynamic lookups into computed tables), Shout handles the simpler case where the entire "memory" is fixed and committed during preprocessing.

### Key Distinction: Lasso vs Shout

**Lasso** (general lookup argument):
- **Use case**: Proving instruction execution where output values are dynamically computed during the trace
- **Table**: May be virtual (defined by efficient evaluation formula, never materialized)
- **Values**: Prover claims values computed from table based on execution inputs
- **Challenge**: Must prove both correct table lookups AND correct decomposition wiring

**Shout** (offline memory checking):
- **Use case**: Proving reads from fixed, pre-committed memory (like program bytecode)
- **Memory**: Explicitly committed during preprocessing (one commitment per memory location or field)
- **Values**: All "table" entries committed upfront; prover only proves correct read access pattern
- **Simplification**: No table decomposition needed—just verify reads match commitments

**Analogy**: Lasso is like proving "I computed $f(x)$ correctly using lookup table $T$". Shout is like proving "I read location $x$ from this pre-sealed document".

### Bytecode Checking

**Goal**: Prove that instructions executed match the committed program bytecode.

**Setup** (preprocessing):
- Bytecode decoded into components: opcode, rs1, rs2, rd, immediate, circuit flags
- Each component committed as a polynomial

**During execution**:
- Prover claims to read instruction at PC $= p$ with decoded values $(opcode, rs1, ...)$

**Shout verification**:

Uses sparse "hit" vector $\widetilde{E}_{\text{bc}}(p)$ counting how many times PC $= p$ was executed.

**Summation identity**:

$$\sum_{p \in \{0,1\}^{\log N}} \widetilde{E}_{\text{bc}}(p) \cdot \left( \widetilde{\text{BC}}(p) - \widetilde{\text{claimed}}(p) \right) = 0$$

Where:
- $\widetilde{\text{BC}}(p)$: MLE of committed bytecode (from preprocessing)
- $\widetilde{\text{claimed}}(p)$: MLE of instruction data used in execution

**Key insight**: $\widetilde{\text{BC}}(p)$ is bound by preprocessing commitment—verifier trusts it matches the program. Sumcheck proves execution trace ($\widetilde{\text{claimed}}$) matches this committed bytecode at accessed locations ($\widetilde{E}_{\text{bc}} \neq 0$).

**Output**: Sparse evaluation claims resolved via SPARK.

### Instruction Lookups (Shout Instance in Jolt)

In Jolt's implementation, instruction lookups also use a Shout-like structure:
- Instruction sub-tables (like 4-bit ADD) have fixed structure
- Table MLEs efficiently evaluable without materialization
- Lookup argument proves execution trace accessed these tables correctly

---

## Protocol 6: Dory - Polynomial Commitment Scheme

**Dory** is a transparent polynomial commitment scheme that serves as Jolt's cryptographic backend. It achieves logarithmic verifier complexity through a novel combination of bilinear pairings and recursive inner product arguments.

### Mathematical Foundation: Bilinear Pairings

Dory operates in the setting of **asymmetric bilinear pairings**:

**Groups and Generators**:
- Three cyclic groups of prime order $p$: $\mathbb{G}_1$, $\mathbb{G}_2$, $\mathbb{G}_T$
- Generators: $G_1 \in \mathbb{G}_1$, $G_2 \in \mathbb{G}_2$

**Pairing Map**: $e: \mathbb{G}_1 \times \mathbb{G}_2 \to \mathbb{G}_T$

**Bilinearity property** (the key algebraic structure):
$$e(aP, bQ) = e(P, Q)^{ab}$$

For any $a, b \in \mathbb{F}_p$, $P \in \mathbb{G}_1$, $Q \in \mathbb{G}_2$.

**Inner-pairing product** (generalized inner product using pairings):

For vectors $\vec{A} \in \mathbb{G}_1^n$ and $\vec{B} \in \mathbb{G}_2^n$:
$$\langle \vec{A}, \vec{B} \rangle = \prod_{i=1}^n e(A_i, B_i) \in \mathbb{G}_T$$

**Security Assumption**: **SXDH (Symmetric External Diffie-Hellman)**
- DDH problem is hard in both $\mathbb{G}_1$ and $\mathbb{G}_2$
- Standard assumption for pairing-based cryptography
- Weaker than knowledge assumptions (like in Groth16)

### The Two-Tiered Commitment Structure

Dory commits to a polynomial by representing its $N$ coefficients as a $\sqrt{N} \times \sqrt{N}$ matrix $M$.

**Layer 1 - Pedersen Commitments to Rows**:

**Public parameters**: Generators $\vec{\Gamma}_1 = (G_{1,1}, \ldots, G_{1,\sqrt{N}}) \in \mathbb{G}_1^{\sqrt{N}}$ and blinding generator $H_1 \in \mathbb{G}_1$

**Commit to row $i$**:
$$V_i = \sum_{j=1}^{\sqrt{N}} M_{i,j} \cdot G_{1,j} + r_i \cdot H_1 \in \mathbb{G}_1$$

where $r_i \in \mathbb{F}_p$ is a random blinding factor.

Result: Vector of row commitments $\vec{V} = (V_1, \ldots, V_{\sqrt{N}}) \in \mathbb{G}_1^{\sqrt{N}}$.

**Layer 2 - AFGHO Commitment to Vector**:

**Public parameters**: Generators $\vec{\Gamma}_2 = (G_{2,1}, \ldots, G_{2,\sqrt{N}}) \in \mathbb{G}_2^{\sqrt{N}}$ and blinding generator $H_2 \in \mathbb{G}_2$

**Commit to vector of commitments**:
$$C_M = \prod_{i=1}^{\sqrt{N}} e(V_i, G_{2,i}) \cdot e(H_1, H_2)^{r_{fin}} \in \mathbb{G}_T$$

where $r_{fin} \in \mathbb{F}_p$ is final blinding factor.

**Expanding this equation** (to see the full structure):

$$C_M = e(V_1, G_{2,1}) \cdot e(V_2, G_{2,2}) \cdot \ldots \cdot e(V_{\sqrt{N}}, G_{2,\sqrt{N}}) \cdot e(H_1, H_2)^{r_{fin}}$$

Substituting the Layer 1 commitments $V_i = \sum_{j=1}^{\sqrt{N}} M_{i,j} \cdot G_{1,j} + r_i \cdot H_1$:

$$C_M = \prod_{i=1}^{\sqrt{N}} e\left(\sum_{j=1}^{\sqrt{N}} M_{i,j} \cdot G_{1,j} + r_i \cdot H_1, G_{2,i}\right) \cdot e(H_1, H_2)^{r_{fin}}$$

Using bilinearity $e(aP + bQ, R) = e(P, R)^a \cdot e(Q, R)^b$:

$$C_M = \prod_{i=1}^{\sqrt{N}} \left[\prod_{j=1}^{\sqrt{N}} e(G_{1,j}, G_{2,i})^{M_{i,j}} \cdot e(H_1, G_{2,i})^{r_i}\right] \cdot e(H_1, H_2)^{r_{fin}}$$

**This shows how**:
1. **Matrix coefficients** $M_{i,j}$ appear as exponents in $\mathbb{G}_T$
2. **Blinding factors** $r_i$ and $r_{fin}$ randomize the commitment
3. **Public pairings** $e(G_{1,j}, G_{2,i})$ are precomputed and known to everyone
4. **Commitment is binding**: Different matrices → different commitments (with overwhelming probability)

**Key properties**:
- $C_M$ is single $\mathbb{G}_T$ element (192 bytes for BN254) binding prover to entire $N$-coefficient matrix
- Two layers of blinding ($r_i$ row blinding and $r_{fin}$ final blinding) ensure computational hiding
- Homomorphic: For two committed polynomials $C_P$ and $C_Q$, can compute $C_{aP+bQ} = C_P^a \cdot C_Q^b$ (without knowing $P$ or $Q$!)
- This homomorphism is critical for batching: verifier can compute commitment to $Q(X) = \sum_i \gamma_i \widetilde{\text{eq}}(r_i, X) \widetilde{P_i}(X)$ from commitments $C_{P_i}$

**SRS size scaling**:
- For polynomial of size $N$: need $2\sqrt{N}$ generators ($\sqrt{N}$ each in $\mathbb{G}_1$ and $\mathbb{G}_2$)
- Compare to univariate schemes requiring $N$ generators
- **Example**: $N = 2^{20}$ requires 2048 generators vs 1M generators (500× reduction)

### Polynomial Evaluation as Vector-Matrix-Vector Product

**The key transformation**: Any multilinear polynomial evaluation can be expressed as:

$$f(\vec{x}) = \vec{L}^T M \vec{R}$$

where:
- $M$: $\sqrt{N} \times \sqrt{N}$ matrix of polynomial coefficients
- $\vec{L}, \vec{R}$: Vectors derived from evaluation point $\vec{x}$ (have multiplicative structure)

**Example**: For 2-variable polynomial $f(X_1, X_2) = c_{00} + c_{10}X_1 + c_{01}X_2 + c_{11}X_1X_2$:

$$f(x_1, x_2) = \begin{pmatrix} 1 & x_1 \end{pmatrix} \begin{pmatrix} c_{00} & c_{01} \\ c_{10} & c_{11} \end{pmatrix} \begin{pmatrix} 1 \\ x_2 \end{pmatrix}$$

**Prover's goal**: Prove $f(\vec{x}) = y$ for committed polynomial with commitment $C_M$.

**Reduction to inner products**: The evaluation $f(\vec{x}) = \vec{L}^T M \vec{R} = y$ becomes proving **three simultaneous inner product relations**:

**Relation 1: Matrix commitment is correct** (links $\vec{V}$ to $C_M$)
$$C_M = \langle \vec{V}, \vec{\Gamma}_2 \rangle \cdot e(H_1, H_2)^{r_{fin}}$$

Expanded: $C_M = \prod_{i=1}^{\sqrt{N}} e(V_i, G_{2,i}) \cdot e(H_1, H_2)^{r_{fin}}$

This is the Layer 2 commitment formula from above. The prover must prove they know $\vec{V}$ and $r_{fin}$ satisfying this.

**Relation 2: Intermediate vector is correct** (links $\vec{v}$ to $\vec{V}$ and evaluation point $\vec{L}$)
$$\langle \vec{L}, \vec{V} \rangle = \langle \vec{v}, \vec{\Gamma}_1 \rangle + r_v H_1$$

where $\vec{v} = \vec{L}^T M$ (the intermediate result of matrix-vector multiply).

Expanded left side: $\sum_{i=1}^{\sqrt{N}} L_i \cdot V_i$ (scalar multiples of $\mathbb{G}_1$ elements summed in $\mathbb{G}_1$)

Expanded right side: $\sum_{j=1}^{\sqrt{N}} v_j \cdot G_{1,j} + r_v H_1$ (Pedersen commitment to $\vec{v}$ in $\mathbb{G}_1$)

This proves the prover computed $\vec{v} = \vec{L}^T M$ correctly from the rows of $M$.

**Relation 3: Final evaluation is correct** (links claimed value $y$ to $\vec{v}$ and evaluation point $\vec{R}$)
$$y = \langle \vec{v}, \vec{R} \rangle$$

Expanded: $y = \sum_{j=1}^{\sqrt{N}} v_j \cdot R_j$ (field element in $\mathbb{F}_p$)

This proves $y = \vec{v}^T \vec{R}$, completing the evaluation $y = (\vec{L}^T M) \vec{R} = \vec{L}^T M \vec{R} = f(\vec{x})$.

**Why three relations?**

The two-tiered commitment structure requires proving consistency at each level:
1. Relation 1: Prover knows the row commitments $\vec{V}$ that open $C_M$
2. Relation 2: Prover computed intermediate $\vec{v}$ correctly from those rows
3. Relation 3: Final evaluation $y$ computed correctly from $\vec{v}$

Each relation is an **inner product argument**—the Dory-Reduce protocol (described next) proves all three simultaneously by recursive folding.

### The Recursive Inner Product Argument: Dory-Reduce Protocol

**Core claim**: Prove knowledge of secret vectors $\mathbf{v}_1 \in \mathbb{G}_1^n$, $\mathbf{v}_2 \in \mathbb{G}_2^n$ that:
- Open commitments $D_1, D_2$
- Have inner-pairing product that opens commitment $C$

**The 5-Step Dory-Reduce Protocol** (per round):

Each round reduces the problem size from $n$ to $n/2$ through a carefully orchestrated exchange:

**Step 1: Prover sends half-commitments**

Split vectors: $\mathbf{v}_1 = (\mathbf{v}_{1,L} || \mathbf{v}_{1,R})$, $\mathbf{v}_2 = (\mathbf{v}_{2,L} || \mathbf{v}_{2,R})$, and generators similarly.

Prover commits to each half:
$$D_{1L} = \langle \mathbf{v}_{1,L}, \mathbf{\Gamma}_{1,L} \rangle + r_L H_1, \quad D_{1R} = \langle \mathbf{v}_{1,R}, \mathbf{\Gamma}_{1,R} \rangle + r_R H_1$$
$$D_{2L} = \langle \mathbf{v}_{2,L}, \mathbf{\Gamma}_{2,L} \rangle + s_L H_2, \quad D_{2R} = \langle \mathbf{v}_{2,R}, \mathbf{\Gamma}_{2,R} \rangle + s_R H_2$$

These four commitments are sent to the verifier.

**Step 2: Verifier samples first challenge $\beta$**

$$\beta \leftarrow \text{Hash}(\text{transcript}, D_{1L}, D_{1R}, D_{2L}, D_{2R}) \in \mathbb{F}_p$$

This is a **random Fiat-Shamir challenge** used for witness randomization.

**Step 3: Prover computes cross-term commitments**

The prover updates their witness internally using $\beta$:
$$\mathbf{v}_{1,new} = \mathbf{v}_1 + \beta \mathbf{\Gamma}_1$$

Then computes cross-term commitments in $\mathbb{G}_1$ (constructed AFTER seeing $\beta$):
$$V_+ = \langle \mathbf{v}_{1,L}, \mathbf{\Gamma}_{1,R} \rangle + r_+ H_1 \in \mathbb{G}_1 \quad \text{(left vector with right generators)}$$
$$V_- = \langle \mathbf{v}_{1,R}, \mathbf{\Gamma}_{1,L} \rangle + r_- H_1 \in \mathbb{G}_1 \quad \text{(right vector with left generators)}$$

These are lifted to $\mathbb{G}_T$ and sent as:
$$C_+ = e(V_+, G_2) \cdot e(H_1, H_2)^{r_{+,fin}} \in \mathbb{G}_T$$
$$C_- = e(V_-, G_2) \cdot e(H_1, H_2)^{r_{-,fin}} \in \mathbb{G}_T$$

**Step 4: Verifier samples second challenge $\alpha$**

$$\alpha \leftarrow \text{Hash}(\text{transcript}, C_+, C_-) \in \mathbb{F}_p$$

This is the **folding challenge** used to combine left and right halves.

**Step 5: Both fold to new claim**

Prover folds vectors:
$$\mathbf{v}'_1 = \mathbf{v}_{1,L} + \alpha \mathbf{v}_{1,R} \in \mathbb{G}_1^{n/2}$$
$$\mathbf{v}'_2 = \mathbf{v}_{2,L} + \alpha^{-1} \mathbf{v}_{2,R} \in \mathbb{G}_2^{n/2}$$

Verifier computes new commitment homomorphically using **precomputed public pairing** $\chi$:

$$\chi = \prod_{i=0}^{n-1} e(G_{1,i}, G_{2,i}) \in \mathbb{G}_T$$

(This is the inner-pairing product of ALL generators at current level)

**Commitment update formula**:
$$C' = C \cdot \chi^\beta \cdot C_+^\alpha \cdot C_-^{\alpha^{-1}} \cdot D_{2}^\beta \cdot D_{1}^{\beta^{-1}}$$

(Simplified notation: the D terms refer to commitments from Step 1, incorporating folding structure)

**Why this works**:
- The algebraic structure ensures that unwanted generators (those being folded out) get coefficient zero
- This happens for ANY random $\beta, \alpha$ due to protocol design, not special values
- The "trick" is in how the prover constructs $C_+, C_-$ AFTER seeing $\beta$

**Step 6: Recurse**

Repeat steps 1-5 for $\log n$ rounds, reducing from size $n$ to $n/2$ to $n/4$ ... to size 1.

**Base case** (n=1): Scalar-Product sigma protocol
- Prover has single witness values $v_1 \in \mathbb{G}_1$, $v_2 \in \mathbb{G}_2$
- Challenge-response protocol proves $e(v_1, v_2)$ opens commitment $C$
- Uses 3-5 pairing operations (computed together via multi-pairing)
- Handles blinding factors from all prior rounds

**Key innovation over Bulletproofs**:
- Bulletproofs: Verifier does $O(n)$ work computing folded keys
- Dory: Pairings let verifier combine commitments in $O(1)$ per round
- Result: $O(\log n)$ verifier instead of $O(n)$

**Two challenges per round**:
The use of TWO challenges ($\beta$ then $\alpha$) is critical:
- $\beta$: Prevents malicious witness construction (witness randomization)
- $\alpha$: Provides the folding challenge (combines left/right)
- Prover cannot compute cross-terms until seeing $\beta$ (binding property)

### Verification Cost Reality Check

**IMPORTANT**: While Dory is $O(\log N)$ in *operation count*, **actual cost dominated by $\mathbb{G}_T$ exponentiations**, not pairings.

**Cost breakdown** (Jolt measurements):
```
Total Dory verifier: ~1.5B RISC-V cycles
├─ G_T exponentiations: ~1.2B cycles (80%) ← THE BOTTLENECK
│  └─ ~40 exponentiations in Fq^12 (~3072-bit elements)
└─ Everything else: ~300M cycles (20%)
   ├─ Pairings (4-5 total): ~100M cycles
   └─ G1/G2 operations: ~200M cycles
```

**Why $\mathbb{G}_T$ exponentiations are expensive**:
- $\mathbb{G}_T$ is $\mathbb{F}_q^{12}$ (12th-degree extension field)
- Each exponentiation: ~254 $\mathbb{F}_q^{12}$ multiplications (square-and-multiply)
- Each $\mathbb{F}_q^{12}$ multiplication: ~144 base field multiplications
- No EVM precompile (unlike pairings)
- Circuit cost: ~50k-100k constraints per exponentiation

**Where exponentiations occur**:
1. **Batched recursive folding** (~30 exps): Dory-Reduce updates commitment per round via formula above
2. **RLC batching** (~29 exps): Combining multiple polynomial openings via random linear combination
3. **Miscellaneous** (few exps): Final checks and adjustments

**Key optimization - Batched exponentiation across rounds**:

The commitment update formula from Step 5 above has multiple $\mathbb{G}_T$ exponentiations:
$$C' = C \cdot \chi^\beta \cdot C_+^\alpha \cdot C_-^{\alpha^{-1}} \cdot D_{1L}^{\alpha/\beta} \cdot D_{1R}^{1/(\alpha\beta)} \cdot D_{2L}^{1/\alpha} \cdot D_{2R}^\alpha$$

**Naive approach** (expensive):
- Compute this formula completely in each round $i$ using that round's $\alpha_i, \beta_i$
- Cost per round: ~8 $\mathbb{G}_T$ exponentiations
- Total for 23 rounds: ~184 exponentiations
- Plus base case and miscellaneous: ~200+ exponentiations

**Batched approach** (efficient - what Jolt does):
- **During each round**: Verifier just accumulates the bases and exponents in a list
  - Round 1: Store $(C, 1), (\chi, \beta_1), (C_+, \alpha_1), (D_{1L}, \alpha_1/\beta_1), \ldots$
  - Round 2: Add $(C', 1), (\chi', \beta_2), (C'_+, \alpha_2), \ldots$
  - Continue through all 23 rounds
- **After all rounds complete**: Compute single multi-exponentiation
  - Product: $\prod_{j=1}^{\text{accumulated}} \text{base}_j^{\text{exponent}_j}$
  - Uses efficient simultaneous exponentiation algorithm
  - Shares work across all exponentiations

**Result**: ~30-40 $\mathbb{G}_T$ exponentiations total (not ~200+)

**Why batching helps**:
- Exponentiations with related bases can share intermediate computation
- Single large multi-exponentiation more efficient than many small ones
- Memory cost: store ~200 (base, exponent) pairs (~40 KB)
- Compute cost: ~40 effective exponentiations (~1.2B cycles)

This is similar to batching scalar multiplications in elliptic curve crypto - doing many at once is cheaper per operation than doing them individually.

**Implications**:
- On-chain verification: NOT viable with standard approach (~4M gas for 40 exps)
- Circuit verification: ~2-4M constraints for 40 exponentiations
- Jolt's solution: SNARK composition with BN254 ↔ Grumpkin cycle (reduces to <30M cycles)

### Three Operations in Jolt

**1. Commit** (preprocessing):
```rust
C_P ← Dory.commit(P, generators)
```
- Binding (SXDH assumption)
- Succinct (constant 192 bytes)
- Transparent (no trusted setup)

**2. Accumulate** (during proof generation):
```rust
accumulator.append(polynomial_id, point, claimed_value)
```
Collect all polynomial evaluation claims across all sumchecks.

**3. Batch Open** (stage 5 of Jolt DAG):

For accumulated claims $\{P_i(r_i) = v_i\}_{i=1}^n$:
1. Sample batching coefficients $\beta_1, \ldots, \beta_n$ (Fiat-Shamir)
2. Combine: $P_{\text{combined}} = \sum_{i=1}^n \beta_i \cdot P_i$
3. Single opening proof for combined claim
4. Proof size: $O(\log N)$ group elements ≈ 10 KB

---

## How the Protocols Work Together

**The Five-Stage DAG Structure**:

Jolt's verification is organized into **5 sequential stages**, where each stage batches multiple sumchecks that can run in parallel:

**Stage 1: Initial Sumcheck (Spartan Outer)**
- Spartan's outer sumcheck: Reduces R1CS constraint satisfaction to sparse matrix evaluation claims
- Output: Evaluation points and claimed values for matrices $\widetilde{Az}$, $\widetilde{Bz}$, $\widetilde{Cz}$

**Stage 2: Product Sumchecks + Read/Write Checking**
- Spartan product sumchecks (3): Prove sparse matrix-vector products
- Registers read-checking (Twist): Prove register reads match last writes (rs1, rs2)
- RAM read-checking (Twist): Prove RAM reads match last writes
- Instruction lookups (Shout): Begin prefix-suffix sumcheck for instruction execution
- **All batched together**: Single interaction round with ~6-8 parallel sumchecks

**Stage 3: Evaluation Sumchecks + Write Checking**
- Spartan matrix evaluation: Prove SPARK sparse polynomial evaluations
- Registers write-checking + evaluation (Twist): Complete register consistency proof
- RAM write-checking + evaluation (Twist): Complete RAM consistency proof
- Instruction lookups continuation: Next phase of prefix-suffix sumcheck
- **All batched together**: Single interaction round with ~6-8 parallel sumchecks

**Stage 4: Final Component Sumchecks**
- RAM final evaluation (Twist): Last verification for RAM state
- Bytecode read-checking (Shout): Prove instruction decode matches committed bytecode
- Instruction lookups final evaluation: Complete instruction lookup proof
- **All batched together**: Single interaction round with ~4-6 parallel sumchecks

**Stage 5: Batched Opening Proof (Dory)**
- Accumulate all polynomial evaluation claims from Stages 1-4 (~30-50 claims)
- Perform batched sumcheck to combine claims into single evaluation
- Dory opening proof via recursive folding (Dory-Reduce protocol)
- **Verification cost**: 4-5 pairings + ~40 $\mathbb{G}_T$ exponentiations

**The Protocol Cascade**:

Within these 5 stages, the six protocols work together:

1. **Spartan** (Stages 1-3): Proves R1CS constraints → outputs sparse matrix evaluation claims
2. **SPARK** (embedded): Proves sparse evaluations → outputs dense (vals, idxs) evaluation claims
3. **Twist** (Stages 2-3): Proves memory consistency (registers + RAM) → outputs increment polynomial evaluation claims
   - Stage 2: Time-ordered verification (read-checking, write-checking)
   - Stage 3: Address-ordered verification (evaluation sumchecks)
4. **Shout** (Stages 2-4): Proves instruction lookups and bytecode → outputs sparse hit vector evaluation claims
5. **SPARK** (again, embedded): Proves Shout's sparse claims → outputs dense evaluation claims
6. **Dory** (Stage 5): Proves all accumulated dense evaluation claims → cryptographic verification

**Key insight**: Each protocol solves a specific problem. Together, they form a complete verification pipeline where outputs of one protocol become inputs to the next.

**Why 5 stages instead of 1?**
- **Parallelization**: Sumchecks within same stage run in parallel (one interaction round)
- **Dependency management**: Stages enforce ordering where one sumcheck's output is another's input
- **Batching efficiency**: Reduces Fiat-Shamir challenges from ~42 (if sequential) to ~5 (one per stage)
- **Memory efficiency**: Can free data structures after each stage completes

**Sparsity is crucial**:
- R1CS matrices: ~30 non-zeros per row (97% sparse)
- Lookup hit vectors: ~$T$ non-zeros out of $2^{128}$ (astronomically sparse)
- SPARK makes this efficient: prover work scales with non-zeros, not domain size

**Batching amplifies benefits**:
- **~42 total sumchecks** across entire proof (exact count depends on program)
- **Batched into 5 stages**: Stages 1-4 contain ~40 sumchecks, Stage 5 has 1 batched sumcheck + Dory opening
- **Single Dory opening proof** for all polynomials (~30-50 accumulated evaluation claims)
  - 4-5 pairings total (multi-pairing batched computation)
  - ~40 $\mathbb{G}_T$ exponentiations (80% of verification cost!)
    - ~30 from batched recursive folding across 23 rounds
    - ~29 from RLC batching of multiple polynomial openings
    - Few misc exponentiations
- **Massive proof size reduction**: $O(\log N)$ per opening, not $O(N)$

**Exact breakdown by stage** (from [jolt_dag.rs:383-569](jolt-core/src/zkvm/dag/jolt_dag.rs)):
- **Stage 1**: Spartan outer sumcheck (1)
- **Stage 2**: Spartan InnerSumcheck (1) + Registers ReadWriteChecking (1) + RAM (RafEvaluation + ReadWriteChecking + OutputSumcheck = 3) + Lookups Booleanity (1) = **6 sumchecks**
- **Stage 3**: Spartan (PCSumcheck + ProductVirtualization = 2) + Registers ValEvaluation (1) + Lookups (ReadRaf + HammingWeight = 2) + RAM (ValEvaluation + ValFinal + HammingBooleanity = 3) = **8 sumchecks**
- **Stage 4**: RAM (HammingWeight + Booleanity + RaSumcheck = 3) + Bytecode (ReadRaf + Booleanity + HammingWeight = 3) + Lookups RaSumcheck (1) = **7 sumchecks**
- **Stage 5**: Batched polynomial opening (1 sumcheck to combine all claims + Dory-Reduce opening proof)
- **Total Stages 1-4**: **22 sumchecks** (1 + 6 + 8 + 7), each batched within their stage
- **Stage 5**: Single batched sumcheck + Dory opening (4-5 pairings + ~40 $\mathbb{G}_T$ exponentiations)

---

# Part 1: Preprocessing Deep Dive: Connecting Theory to Code

## Overview: What is Preprocessing?

**Preprocessing** is the one-time setup phase that happens **once per guest program** (not per execution). It's expensive but only needs to be computed once and can be cached.

Think of it as preparing the "game board" before playing:

- **Prover preprocessing**: Everything the prover needs to generate proofs
- **Verifier preprocessing**: Everything the verifier needs to check proofs (much smaller!)

## The Two Main Preprocessing Tasks

Preprocessing does:

1. **Commits to bytecode polynomial**
2. **Generates Dory SRS for polynomial commitment scheme**

Let's understand each in detail.

---
## Task 1: Committing to Bytecode Polynomial

### Theory Connectio

**ELF Binary vs. Bytecode**:
```
Rust guest code
  ↓ [rustc --target riscv64gc-unknown-none-elf]
ELF binary (complete file with all sections)
  ↓ [compile_{fn}() parses ELF]
Program struct
  ├── elf: Vec<u8> (raw ELF file)
  └── bytecode: Vec<ELFInstruction> (extracted .text section)
  ↓ [preprocessing commits to bytecode]
Bytecode commitment (polynomial commitment)
```

**Why commit to bytecode?**

- The verifier needs to know *which program* was executed
- But sending the entire bytecode (thousands of instructions) is expensive
- Instead: **commit to bytecode as a polynomial**
  - Commitment is small (single group element, ~32 bytes)
  - Binding: can't change bytecode without changing commitment
  - Hiding: commitment doesn't reveal the bytecode itself

From:
>
> **Purpose**: Prove trace instructions match committed bytecode.
>
> During preprocessing: bytecode decoded and committed
>
> During execution: trace records which instructions executed
>
> Bytecode sumcheck proves: Each instruction in trace matches an instruction in committed bytecode

### Code Flow: How Bytecode Gets Committed

#### Step 1: Macro Generates `preprocess_prover_{fn_name}()`

**File**: [jolt-sdk/macros/src/lib.rs:435-482](jolt-sdk/macros/src/lib.rs#L435)

When you write:
```rust
#[jolt::provable]
fn my_function() { ... }
```

The macro generates:
```rust
pub fn preprocess_prover_my_function(program: &mut Program)
    -> JoltProverPreprocessing<F, PCS>
{
    // 1. Decode ELF into bytecode
    let (bytecode, memory_init, program_size) = program.decode();

    // 2. Create memory layout
    let memory_layout = MemoryLayout::new(&memory_config);

    // 3. Call the core preprocessing
    let preprocessing = JoltRV64IMAC::prover_preprocess(
        bytecode,
        memory_layout,
        memory_init,
        max_trace_length,
    );

    preprocessing
}
```

#### Step 2: Core Preprocessing

**File**: [jolt-core/src/zkvm/mod.rs:241-254](jolt-core/src/zkvm/mod.rs#L241)

```rust
fn prover_preprocess(
    bytecode: Vec<Instruction>,
    memory_layout: MemoryLayout,
    memory_init: Vec<(u64, u8)>,
    max_trace_length: usize,
) -> JoltProverPreprocessing<F, PCS> {
    // Create shared preprocessing (bytecode + RAM)
    let shared = Self::shared_preprocess(bytecode, memory_layout, memory_init);

    // Generate PCS setup (Dory SRS)
    let max_T = max_trace_length.next_power_of_two();
    let generators = PCS::setup_prover(DTH_ROOT_OF_K.log_2() + max_T.log_2());

    JoltProverPreprocessing { generators, shared }
}
```

#### Step 3: Bytecode Preprocessing

**File**: [jolt-core/src/zkvm/bytecode/mod.rs:40-60](jolt-core/src/zkvm/bytecode/mod.rs#L40)

```rust
pub fn preprocess(mut bytecode: Vec<Instruction>) -> Self {
    // 1. Prepend a no-op instruction (PC starts at 0)
    bytecode.insert(0, Instruction::NoOp);

    // 2. Create PC mapper (maps real addresses to virtual addresses)
    let pc_map = BytecodePCMapper::new(&bytecode);

    // 3. Compute d parameter (chunking for Shout)
    let d = compute_d_parameter(bytecode.len().next_power_of_two().max(2));

    // 4. Pad bytecode to power of 2 (required for MLE)
    let code_size = (bytecode.len().next_power_of_two().log_2().div_ceil(d) * d)
        .pow2()
        .max(DTH_ROOT_OF_K);
    bytecode.resize(code_size, Instruction::NoOp);

    Self {
        code_size,
        bytecode,
        pc_map,
        d,
    }
}
```

**What happens here:**

1. **Padding**: Bytecode must be a power of 2 for efficient MLE operations
2. **PC mapping**: Maps real memory addresses to "virtual" PC indices
   - Guest program loaded at `0x80000000` (RAM_START_ADDRESS)
   - Virtual PC is index into bytecode vector (0, 1, 2, ...)
3. **Chunking parameter `d`**: Used by Shout protocol (more on this below)

#### Step 4: Actual Commitment (During Proving)

**Important**: The bytecode itself is **NOT committed during preprocessing**!

Instead:

- Preprocessing stores the **decoded bytecode** in `BytecodePreprocessing`
- During **proof generation** (Stage 1), bytecode is committed as part of witness generation

**File**: [jolt-core/src/zkvm/dag/jolt_dag.rs:574-611](jolt-core/src/zkvm/dag/jolt_dag.rs#L574)

```rust
fn generate_and_commit_polynomials(
    prover_state_manager: &mut StateManager<F, ProofTranscript, PCS>,
) -> Result<HashMap<CommittedPolynomial, PCS::OpeningProofHint>, anyhow::Error> {
    let (preprocessing, trace, _, _) = prover_state_manager.get_prover_data();

    // Generate ALL committed polynomials (including bytecode)
    let polys = AllCommittedPolynomials::iter().copied().collect::<Vec<_>>();
    let mut all_polys =
        CommittedPolynomial::generate_witness_batch(&polys, preprocessing, trace);

    // Commit using Dory PCS
    let commit_results = PCS::batch_commit(&committed_polys, &preprocessing.generators);

    // Returns commitments + opening hints
    let (commitments, hints): (Vec<PCS::Commitment>, Vec<PCS::OpeningProofHint>) =
        commit_results.into_iter().unzip();

    // Store commitments in state manager
    prover_state_manager.set_commitments(commitments);

    Ok(hint_map)
}
```

**`AllCommittedPolynomials` enum** includes:

- Bytecode-related polynomials (instruction opcodes, flags, etc.)
- RAM polynomials (addresses, values, timestamps)
- Register polynomials
- Instruction lookup polynomials

Each is converted to a **multilinear extension (MLE)** and committed using Dory.

---

### Mathematical Details: Polynomial Commitments with Dory

**From Theory**: Dory uses a two-tiered commitment scheme for matrices.

#### Multilinear Polynomial Representation

A multilinear polynomial in $\nu$ variables with $n = 2^{\nu}$ coefficients is represented as a matrix $M \in \mathbb{F}_p^{\sqrt{n} \times \sqrt{n}}$.

**Example**: Bytecode polynomial with 1024 instructions ($\nu = 10$, $n = 1024$):

- Matrix representation: $M \in \mathbb{F}_p^{32 \times 32}$
- Each entry $M_{i,j}$ corresponds to a bytecode instruction

#### Layer 1: Pedersen Commitments to Rows

**Public parameters** (from Dory SRS):

- Generator vector: $\vec{\Gamma}_1 = (G_{1,1}, \ldots, G_{1,m}) \in \mathbb{G}_1^m$
- Blinding generator: $H_1 \in \mathbb{G}_1$

**Commit to row $i$ of matrix $M$**:
$$V_i = \langle \vec{M}_i, \vec{\Gamma}_1 \rangle + r_i H_1 = \left( \sum_{j=1}^m M_{i,j} G_{1,j} \right) + r_i H_1$$

Where:

- $\vec{M}_i = (M_{i,1}, \ldots, M_{i,m})$ is the $i$-th row
- $r_i \in \mathbb{F}_p$ is a random blinding factor
- Result: $V_i \in \mathbb{G}_1$ (single elliptic curve point)

**Result**: Vector of row commitments $\vec{V} = (V_1, \ldots, V_n) \in \mathbb{G}_1^n$

#### Layer 2: AFGHO Commitment to Vector of Commitments

**Public parameters**:

- Generator vector: $\vec{\Gamma}_2 = (G_{2,1}, \ldots, G_{2,n}) \in \mathbb{G}_2^n$
- Blinding generator: $H_2 \in \mathbb{G}_2$

**Final commitment** to entire matrix:
$$C_M = \langle \vec{V}, \vec{\Gamma}_2 \rangle \cdot e(H_1, H_2)^{r_{fin}}$$
$$= \left( \prod_{i=1}^n e(V_i, G_{2,i}) \right) \cdot e(H_1, H_2)^{r_{fin}}$$

Where:

- $e: \mathbb{G}_1 \times \mathbb{G}_2 \to \mathbb{G}_T$ is the bilinear pairing
- $r_{fin} \in \mathbb{F}_p$ is final blinding factor
- Result: $C_M \in \mathbb{G}_T$ (single element in target group)

**Expanding the full commitment**:
$$C_M = \prod_{i=1}^n e\left( \sum_{j=1}^m M_{i,j} G_{1,j} + r_i H_1 \;,\; G_{2,i} \right) \cdot e(H_1, H_2)^{r_{fin}}$$

**Commitment size**:

- With BLS12-381 curve and torus-based compression: **192 bytes** (constant size!)
- Compare to $n \times m$ matrix entries sent in plaintext: $32 \times 32 \times 32 = 32$KB

#### Concrete Example: 2×2 Bytecode Matrix

**Setup**: 4 instructions, represented as $2 \times 2$ matrix:
$$M = \begin{pmatrix} m_{11} & m_{12} \\ m_{21} & m_{22} \end{pmatrix}$$

**Public parameters**:

- $\vec{\Gamma}_1 = (G_{1,1}, G_{1,2}) \in \mathbb{G}_1^2$, $H_1 \in \mathbb{G}_1$
- $\vec{\Gamma}_2 = (G_{2,1}, G_{2,2}) \in \mathbb{G}_2^2$, $H_2 \in \mathbb{G}_2$

**Step 1: Commit to rows** (choose random $r_1, r_2 \in \mathbb{F}_p$):

- Row 1: $V_1 = m_{11}G_{1,1} + m_{12}G_{1,2} + r_1 H_1 \in \mathbb{G}_1$
- Row 2: $V_2 = m_{21}G_{1,1} + m_{22}G_{1,2} + r_2 H_1 \in \mathbb{G}_1$

**Step 2: Commit to vector** (choose random $r_{fin} \in \mathbb{F}_p$):
$$C_M = e(V_1, G_{2,1}) \cdot e(V_2, G_{2,2}) \cdot e(H_1, H_2)^{r_{fin}} \in \mathbb{G}_T$$

**Result**: Single $\mathbb{G}_T$ element (192 bytes) commits to entire 2×2 matrix!

#### Why This Matters for Bytecode

**Binding property**: Can't change bytecode without changing commitment

- Computational hardness: Based on SXDH assumption (DDH hard in both $\mathbb{G}_1$ and $\mathbb{G}_2$)
- Security: Schwartz-Zippel lemma ensures different polynomials committed to different values

**Hiding property**: Commitment doesn't reveal bytecode

- Two layers of blinding ($r_i$ per row, $r_{fin}$ overall)
- Information-theoretic hiding given proper randomness

**Succinctness**:

- Bytecode: 1024 instructions × 4 bytes = 4KB
- Commitment: 192 bytes (~20× compression)
- Opening proof: ~18KB for polynomial evaluation (logarithmic in size)

---

## From Bytecode to Polynomial: The Complete Transformation

Before we explain why preprocessing doesn't commit to bytecode yet, let's understand **how bytecode becomes a polynomial** in the first place. This is a crucial transformation that happens during **proving time**.

### What is Bytecode? (Raw Format)

**Bytecode** is the compiled RISC-V machine code stored in the ELF binary's `.text` section.

**Example**: SHA3 guest function compiled to RISC-V:
```
Address      Raw Bytes    Assembly Instruction
0x80000000:  0x00000033   ADD  x0, x0, x0   (NOP)
0x80000004:  0x00a58593   ADDI x11, x11, 10
0x80000008:  0x40b50533   SUB  x10, x10, x11
0x8000000c:  0x02a5f463   BGEU x11, x10, 40
...
```

Each instruction is a **32-bit word** (4 bytes) encoding:

- **Opcode**: Which operation (ADD, SUB, LOAD, etc.)
- **Registers**: Source registers (rs1, rs2), destination register (rd)
- **Immediate**: Constant values for I-type/S-type instructions
- **Flags**: Jump flag, branch flag, load flag, etc. (derived from opcode)

### Step 1: Decoding Bytecode (Preprocessing)

During preprocessing, each 32-bit instruction is **decoded** into its constituent parts.

**File**: [tracer/src/instruction/mod.rs:663](tracer/src/instruction/mod.rs#L663)

```rust
pub fn decode(instr: u32, address: u64, compressed: bool) -> Result<Instruction, &'static str> {
    let opcode = instr & 0x7f;  // Extract bits [6:0]
    match opcode {
        0b0110111 => Ok(LUI::new(instr, address, true, compressed).into()),
        0b0010111 => Ok(AUIPC::new(instr, address, true, compressed).into()),
        0b0110011 => {
            // R-type instructions (ADD, SUB, etc.)
            let funct3 = (instr >> 12) & 0x7;
            let funct7 = (instr >> 25) & 0x7f;
            match (funct7, funct3) {
                (0x00, 0x0) => Ok(ADD::new(instr, address, true, compressed).into()),
                (0x20, 0x0) => Ok(SUB::new(instr, address, true, compressed).into()),
                // ... more R-type instructions
            }
        }
        // ... more opcodes
    }
}
```

**What gets extracted** (for R-type ADD instruction `0x00a58593`):
```rust
FormatR::parse(0x00a58593) {
    rd:  (instr >> 7)  & 0x1f = 11  // bits [11:7]  = x11 (destination)
    rs1: (instr >> 15) & 0x1f = 11  // bits [19:15] = x11 (source 1)
    rs2: (instr >> 20) & 0x1f = 10  // bits [24:20] = x10 (source 2)
}
```

**Result**: Each instruction becomes an `Instruction` enum variant storing:

- Opcode type (ADD, SUB, LOAD, etc.)
- Register addresses (rs1, rs2, rd)
- Immediate values
- Memory address where instruction lives

**Stored in**: `BytecodePreprocessing.bytecode: Vec<Instruction>`

### Step 2: Creating "Read Values" (Proving Time)

During proving, we don't commit to the raw bytecode vector. Instead, we create **read address polynomials** that encode which bytecode entry was accessed at each cycle.

**The key insight**: Bytecode checking uses **Shout lookup argument** (offline memory checking), which needs:

1. **Write addresses**: Where values were written (bytecode indices)
2. **Read addresses**: Where values were read (PC values during execution)
3. Prove: Every read matches a write (read from valid bytecode entry)

**File**: [jolt-core/src/zkvm/bytecode/mod.rs:192-243](jolt-core/src/zkvm/bytecode/mod.rs#L192)

```rust
fn compute_ra_evals<F: JoltField>(
    preprocessing: &BytecodePreprocessing,
    trace: &[Cycle],
    eq_r_cycle: &[F],  // eq(r_cycle, j) for each cycle j
) -> Vec<Vec<F>> {
    let T = trace.len();
    let log_K = preprocessing.code_size.log_2();  // log(bytecode size)
    let d = preprocessing.d;  // chunking parameter
    let log_K_chunk = log_K.div_ceil(d);
    let K_chunk = log_K_chunk.pow2();

    // Create d polynomials, each of size K_chunk
    let mut result: Vec<Vec<F>> = (0..d).map(|_| vec![F::zero(); K_chunk]).collect();

    for (j, cycle) in trace.iter().enumerate() {
        // Get PC for this cycle (which bytecode instruction was executed)
        let mut pc = preprocessing.get_pc(cycle);

        // Decompose PC into d chunks (for Twist/Shout chunking)
        for i in (0..d).rev() {
            let k = pc % K_chunk;  // Extract chunk i
            result[i][k] += eq_r_cycle[j];  // Accumulate at index k
            pc >>= log_K_chunk;  // Shift to next chunk
        }
    }

    result  // Returns Vec<Vec<F>>: d polynomials, each size K_chunk
}
```

**What this does**:

**Input**: Execution trace
```
Cycle 0: PC = 5   (executed bytecode[5])
Cycle 1: PC = 6   (executed bytecode[6])
Cycle 2: PC = 5   (executed bytecode[5] again)
Cycle 3: PC = 10  (executed bytecode[10])
```

**Output**: Read address polynomial `ra` (one-hot encoding)
```
For d=1 (no chunking):
ra[5]  = eq(r_cycle, 0) + eq(r_cycle, 2)  // PC=5 accessed at cycles 0 and 2
ra[6]  = eq(r_cycle, 1)                   // PC=6 accessed at cycle 1
ra[10] = eq(r_cycle, 3)                   // PC=10 accessed at cycle 3
ra[k]  = 0  for all other k               // Not accessed
```

**Mathematical meaning**:
$$\text{ra}(k) = \sum_{j : \text{PC}_j = k} \text{eq}(\vec{r}_{\text{cycle}}, j)$$

Where:

- $k$: Bytecode index (0 to K-1)
- $j$: Cycle number (0 to T-1)
- $\text{PC}_j$: Program counter at cycle $j$
- $\text{eq}(\vec{r}_{\text{cycle}}, j)$: Multilinear extension of indicator for cycle $j$

**Why $eq(\vec{r}_{\text{cycle}}, j)$?** This is the multilinear extension trick:

- $\text{eq}(\vec{r}, j)$ evaluates to 1 when $\vec{r}$ encodes index $j$ in binary
- Summing these creates a polynomial that "remembers" which cycles accessed which PC
- This is what makes it a valid MLE of the access pattern

### Step 3: Converting Read Values to MLE

The `ra` polynomial is already in **evaluation form** (values at all Boolean hypercube points).

**Conversion to MLE** (automatic via DenseMultilinearExtension):

**File**: [jolt-core/src/poly/dense_mlpoly.rs](jolt-core/src/poly/dense_mlpoly.rs)

```rust
pub struct DenseMultilinearExtension<F: JoltField> {
    pub evaluations: Vec<F>,  // Evaluations on {0,1}^n
    pub num_vars: usize,      // n (number of variables)
}
```

For bytecode read address polynomial:
```rust
let ra_evals = compute_ra_evals(...);  // Vec<F> of size K_chunk

let ra_mle = DenseMultilinearExtension {
    evaluations: ra_evals,
    num_vars: log_K_chunk,  // log_2(K_chunk) variables
};
```

**Example with concrete numbers**:
```
Bytecode size: K = 256 = 2^8
d = 1 (no chunking)
K_chunk = 256

ra_evals = [0, 0, 0, 0, 0, eq(r,0)+eq(r,2), eq(r,1), 0, ..., eq(r,3), ...]
           ↑                    ↑                ↑                ↑
         index 0              index 5         index 6        index 10

MLE has 8 variables: X_0, X_1, ..., X_7
Evaluation at point (0,0,0,0,0,1,0,1) = ra_evals[5] = eq(r,0)+eq(r,2)
Evaluation at point (0,0,0,0,0,1,1,0) = ra_evals[6] = eq(r,1)
```

### Step 4: Committing to the Bytecode Polynomial

Now we can commit! But what exactly do we commit to?

**Two separate commitments**:

#### 1. Bytecode Write Polynomial (Memory Contents)

This encodes the **actual bytecode instructions** and their decoded fields.

**What gets committed** (for each decoded field):

From an ADD instruction at bytecode[5]:
```rust
Instruction::ADD {
    address: 0x80000014,
    operands: FormatR { rd: 10, rs1: 11, rs2: 12 },
    ...
}
```

We create separate polynomials for each field:

- `opcode[5]` = ADD (encoded as field element)
- `rd[5]` = 10
- `rs1[5]` = 11
- `rs2[5]` = 12
- `imm[5]` = 0 (not used for ADD)
- `is_jump[5]` = 0 (derived from opcode)
- `is_branch[5]` = 0
- `is_load[5]` = 0
- ...

Each of these becomes an MLE and gets committed via Dory!

#### 2. Bytecode Read Address Polynomial (Access Pattern)

This is the `ra` polynomial we computed above, encoding **which bytecode entries were accessed**.

**Both committed** using Dory's two-tiered commitment scheme (Layer 1: Pedersen to rows, Layer 2: AFGHO to vector of row commitments).

### Step 5: Shout Lookup Argument

**The final check**: Prove that every read from bytecode during execution matches a valid write (i.e., a real bytecode instruction).

**Sumcheck claim**:
$$\sum_{k \in \{0,1\}^{\log K}} \text{ra}(k) \cdot (\text{read\_value}(k) - \text{write\_value}(k)) = 0$$

This says: "For every bytecode index $k$ that was accessed ($ra(k) \neq 0$), the value read must equal the value written."

**Multiple read values per instruction**:

- Read opcode matches written opcode
- Read rs1 matches written rs1
- Read rs2 matches written rs2
- Read rd matches written rd
- Read immediate matches written immediate
- Read flags match written flags

Each of these gets its own sumcheck!

### Complete Flow Diagram

```
Bytecode (Raw)                    Preprocessing
───────────────                   ─────────────
0x00000033  ─────decode────────> ADD { rd:0, rs1:0, rs2:0 }
0x00a58593  ─────decode────────> ADDI { rd:11, rs1:11, imm:10 }
0x40b50533  ─────decode────────> SUB { rd:10, rs1:10, rs2:11 }
    ↓
Store in BytecodePreprocessing.bytecode: Vec<Instruction>


Proving Time
────────────
Execution Trace:                  Read Address Polynomial:
Cycle 0: PC=0 (ADD)       ───>    ra[0] += eq(r_cycle, 0)
Cycle 1: PC=1 (ADDI)      ───>    ra[1] += eq(r_cycle, 1)
Cycle 2: PC=2 (SUB)       ───>    ra[2] += eq(r_cycle, 2)
    ↓
Create MLEs:

- ra_mle (read addresses)
- opcode_mle (opcodes from bytecode)
- rd_mle (destination registers from bytecode)
- rs1_mle (source register 1 from bytecode)
- ...
    ↓
Commit to MLEs using Dory:

- Reshape to matrix (√K × √K)
- Layer 1: Pedersen commit to each row
- Layer 2: AFGHO commit to vector of row commitments
    ↓
Shout Lookup Argument:
Prove: ∀k, ra(k) · (read_value(k) - write_value(k)) = 0
```

### Why Preprocessing Stores Bytecode But Doesn't Commit Yet?

**Reason**: The commitment scheme (Dory) requires knowing the execution trace length!

- **Preprocessing time**: Don't know how long program will run
- **Proving time**: Know exact trace length, can set up Dory with correct parameters

**What preprocessing DOES**:

- Decode and pad bytecode to power of 2
- Store it in `BytecodePreprocessing`
- Later (proving): Convert bytecode to MLE and commit

---

## Task 2: Generating Dory SRS

### Theory Connection

**SRS** = **Structured Reference String**

From polynomial commitment scheme theory:

- PCS allows committing to a polynomial and later proving evaluations
- **Dory** is Jolt's PCS (from Space and Time's implementation)
- Transparent: no trusted setup (unlike KZG which needs trusted setup ceremony)
- But still needs **public parameters** generated from randomness

**What's in the SRS?**

- Elliptic curve points for commitment generation
- Pairing-friendly curve elements (Bn254)
- Prover setup: larger (needed to create commitments)
- Verifier setup: smaller (just needed to verify)

### Mathematical Details: Dory SRS Generation

Before diving into the code, let's understand the mathematics behind SRS generation.

#### Bilinear Pairings and Groups

**Definition**:

Dory operates with three cyclic groups of large prime order $p$:

- $\mathbb{G}_1, \mathbb{G}_2$: Source groups (additive notation)
- $\mathbb{G}_T$: Target group (multiplicative notation)
- Pairing map: $e: \mathbb{G}_1 \times \mathbb{G}_2 \to \mathbb{G}_T$

**Bilinearity property**: For any $a, b \in \mathbb{F}_p$, $P \in \mathbb{G}_1$, $Q \in \mathbb{G}_2$:
$$e(aP, bQ) = e(P, Q)^{ab}$$

**Generators**:

- $G_1 \in \mathbb{G}_1$: Fixed public generator for group 1
- $G_2 \in \mathbb{G}_2$: Fixed public generator for group 2

**Security**: SXDH assumption (Symmetric eXternal Diffie-Hellman)

- DDH is hard in both $\mathbb{G}_1$ and $\mathbb{G}_2$
- Ensures computational indistinguishability of tuples

#### SRS Components

The Structured Reference String contains two types of public parameters:

**1. Generator Vectors for Matrix Rows** (in $\mathbb{G}_1$):
$$\vec{\Gamma}_1 = (G_{1,1}, G_{1,2}, \ldots, G_{1,m}) \in \mathbb{G}_1^m$$
$$H_1 \in \mathbb{G}_1 \text{ (blinding generator)}$$

Where $m = \sqrt{N}$ for a polynomial of $N$ coefficients.

**2. Generator Vectors for Matrix Columns** (in $\mathbb{G}_2$):
$$\vec{\Gamma}_2 = (G_{2,1}, G_{2,2}, \ldots, G_{2,n}) \in \mathbb{G}_2^n$$
$$H_2 \in \mathbb{G}_2 \text{ (blinding generator)}$$

Where $n = \sqrt{N}$ for a polynomial of $N$ coefficients.

**Key Insight**: For matrix commitment to work efficiently, Dory needs:

- $m$ generators in $\mathbb{G}_1$ for committing to **rows** (Layer 1 Pedersen commitments)
- $n$ generators in $\mathbb{G}_2$ for committing to **vector of row commitments** (Layer 2 AFGHO commitment)
- For multilinear polynomials: $m = n = \sqrt{N}$

#### Matrix View of Polynomial Commitment

**Why the square root?**

A multilinear polynomial $f$ with $N = 2^{\nu}$ coefficients is reshaped into a matrix:

$$M \in \mathbb{F}_p^{\sqrt{N} \times \sqrt{N}}$$

**Example with concrete numbers**:
```
Polynomial size: N = 16,384 = 2^14 coefficients
Matrix dimensions: √16,384 × √16,384 = 128 × 128

SRS requirement:

- G_1 generators needed: 128 (one per column)
- G_2 generators needed: 128 (one per row)
- Total: 256 group elements

Naive approach would need: 16,384 generators!
Savings: 64× reduction in SRS size
```

#### SRS Size Formula

For a Jolt proof with trace length $T$ and constant $K = 16$:

**Total polynomial size**:
$$N = K \cdot T = 16T = 2^{4 + \log_2(T)}$$

**Matrix dimensions**:
$$\text{rows} = \text{cols} = \sqrt{16T} = 4\sqrt{T}$$

**SRS contains**:

- $4\sqrt{T}$ points in $\mathbb{G}_1$ (for Pedersen commitments)
- $4\sqrt{T}$ points in $\mathbb{G}_2$ (for AFGHO commitment)
- Plus blinding generators $H_1, H_2$

**Concrete example**:
```
Trace length: T = 65,536 = 2^16
Total size: N = 16 × 65,536 = 1,048,576 = 2^20

Matrix: √1,048,576 × √1,048,576 = 1,024 × 1,024

SRS size:

- 1,024 G_1 points ≈ 1024 × 48 bytes = 49 KB
- 1,024 G_2 points ≈ 1024 × 96 bytes = 98 KB
- Total prover SRS: ~147 KB

Verifier SRS: Much smaller (only needs subset for verification)

- Typically 10-20× smaller than prover SRS
```

#### Transparency of Dory

**Key property**:

Dory does NOT require a trusted setup ceremony. The SRS is generated from **publicly verifiable randomness**.

**How it works**:

1. Start with hash output: `H(seed)` for public seed
2. Generate field elements: $\{s_1, s_2, \ldots\}$ via hash chain
3. Map field elements to curve points:
   - $G_{1,i} = s_i \cdot G_1$ (scalar multiplication)
   - $G_{2,i} = s_i \cdot G_2$
4. Anyone can re-compute and verify the SRS

**Contrast with KZG**:

- KZG: Needs trusted setup with secret $\tau$
- SRS contains: $(G_1, \tau G_1, \tau^2 G_1, \ldots, \tau^N G_1)$
- If $\tau$ is known, scheme is broken
- Dory: No secret! Anyone can verify SRS generation

#### Logarithmic Proof Size

**Why Dory's verifier is fast**:

Dory uses recursive halving strategy (like Bulletproofs):

- Start with size-$n$ problem
- Each round: fold vectors to half size
- After $\log_2(n)$ rounds: size-1 problem (trivial to check)

**Proof components per round**:

- 6 group elements per recursive step
- Total proof size: $6 \cdot \log_2(\sqrt{N})$ group elements
- For $N = 2^{20}$: $6 \cdot 10 = 60$ group elements ≈ 18 KB

**Verifier work**:

- $O(\log N)$ operations count (23 rounds for typical trace)
- **But**: ~40 G_T exponentiations dominate the cost
  - **Batched recursive folding**: ~30 G_T exps (would be ~10 per round × 23 rounds = 230 naively, but verifier accumulates and computes once at end)
  - **RLC batching**: ~29 additional G_T exps (combining multiple polynomial openings)
  - **Miscellaneous**: Few additional exponentiations
  - Each G_T exponentiation: ~36,000 base field operations
- 4-5 pairings total (base case verification)
- For $N = 2^{20}$: ~30 milliseconds total (but 80% of that is G_T exponentiations)

**This is why Jolt uses Dory for proving**: Transparent + efficient prover + logarithmic proof size!

**Note on verification cost**: While operation count is logarithmic, G_T exponentiations create challenges for on-chain (no EVM precompile) and circuit implementations (~2-4M constraints for 40 exps). See "Critical Note on Verification Complexity" in Protocol 6 section for details.

### Code Flow: SRS Generation

#### Step 1: Determine Max Polynomial Size

**File**: [jolt-core/src/zkvm/mod.rs:249-251](jolt-core/src/zkvm/mod.rs#L249)

```rust
fn prover_preprocess(..., max_trace_length: usize, ...) -> JoltProverPreprocessing<F, PCS> {
    let shared = Self::shared_preprocess(bytecode, memory_layout, memory_init);

    let max_T: usize = max_trace_length.next_power_of_two();

    // DTH_ROOT_OF_K = 16 (constant from witness.rs)
    // This is the "number of columns" in Dory's matrix view
    let generators = PCS::setup_prover(DTH_ROOT_OF_K.log_2() + max_T.log_2());

    JoltProverPreprocessing { generators, shared }
}
```

**Key insight**: `setup_prover(n)` generates setup for polynomials of size up to 2^n.

**Why `DTH_ROOT_OF_K.log_2() + max_T.log_2()`?**

- Dory views polynomials as **matrices** (rows × columns)
- For a polynomial of degree K·T, Dory uses a sqrt(K·T) × sqrt(K·T) matrix
- `DTH_ROOT_OF_K = 16 = 2^4`: fixed number of columns
- `max_T`: maximum trace length
- Total polynomial size: `16 * max_T = 2^(4 + log(max_T))`

#### Step 2: PCS Setup

**File**: [jolt-core/src/poly/commitment/commitment_scheme.rs:33-36](jolt-core/src/poly/commitment/commitment_scheme.rs#L33)

```rust
pub trait CommitmentScheme: Clone + Sync + Send + Debug {
    type ProverSetup: Clone + Sync + Send + Debug + CanonicalSerialize + CanonicalDeserialize;
    type VerifierSetup: Clone + Sync + Send + Debug + CanonicalSerialize + CanonicalDeserialize;

    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup;
    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup;

    // ... other methods (commit, prove, verify)
}
```

#### Step 3: Dory's Setup Implementation

**File**: [jolt-core/src/poly/commitment/dory.rs](jolt-core/src/poly/commitment/dory.rs)

Dory setup creates:

1. **G1 generators**: Elliptic curve points in G1 (used for polynomial coefficients)
2. **G2 generators**: Elliptic curve points in G2 (used for verification)
3. **Pairing precomputation**: Cache pairings for faster verification

**Actual setup** happens via Space and Time's `dory` crate:
```rust
use dory::{setup_with_urs_file, ProverSetup, VerifierSetup};

impl CommitmentScheme for DoryCommitmentScheme {
    type ProverSetup = DoryProverSetup;
    type VerifierSetup = DoryVerifierSetup;

    fn setup_prover(max_num_vars: usize) -> Self::ProverSetup {
        // Generate structured reference string
        let urs = setup_with_urs_file(max_num_vars);
        // ... wrap in Jolt types
    }

    fn setup_verifier(setup: &Self::ProverSetup) -> Self::VerifierSetup {
        // Extract verifier portion (much smaller!)
        setup.extract_verifier_setup()
    }
}
```

**Note**: For a detailed explanation of Dory's matrix view of polynomials and why it enables efficient commitments, see [Mathematical Details: Polynomial Commitments with Dory](#mathematical-details-polynomial-commitments-with-dory) earlier in this document.

### SRS Size Example

**File**: [jolt-core/src/poly/commitment/dory.rs:61-76](jolt-core/src/poly/commitment/dory.rs#L61)

```rust
impl DoryGlobals {
    pub fn initialize(K: usize, T: usize) -> Self {
        let matrix_size = K as u128 * T as u128;
        let num_columns = matrix_size.isqrt().next_power_of_two();
        let num_rows = num_columns;

        tracing::info!("[Dory PCS] # rows: {num_rows}");
        tracing::info!("[Dory PCS] # cols: {num_columns}");

        // Store globally for this proving session
        unsafe {
            GLOBAL_T.set(T).expect("GLOBAL_T already initialized");
            MAX_NUM_ROWS.set(num_rows as usize).expect("MAX_NUM_ROWS already initialized");
            NUM_COLUMNS.set(num_columns as usize).expect("NUM_COLUMNS already initialized");
        }

        DoryGlobals()
    }
}
```

**Output example** (for fibonacci with 10k cycles):
```
[Dory PCS] # rows: 512
[Dory PCS] # cols: 512
```
This means SRS has ~512 points in G1 and G2.

---

## Mathematical Objects: What Do We Actually Store?

Before looking at the code structures, let's be crystal clear about the **mathematical objects** that preprocessing produces.

### Task 1 Output: Decoded Bytecode (NOT Committed Yet!)

**Mathematical object**: Vector of decoded instructions

$$\text{bytecode} = (\text{instr}_0, \text{instr}_1, \ldots, \text{instr}_{K-1}) \in \mathcal{I}^K$$

Where:

- $\mathcal{I}$ = set of all RISC-V instructions
- $K$ = padded bytecode size (power of 2)
- Each $\text{instr}_i$ is a structured object containing: opcode, rs1, rs2, rd, immediate, flags

**Storage format**: `Vec<Instruction>` (Rust enum variants)

**What we DON'T have yet**:
-  Polynomial representation
-  MLE (Multilinear Extension)
-  Commitment (no $C_M \in \mathbb{G}_T$ yet!)

**Why not committed?** Dory needs to know the execution trace length $T$ to determine matrix dimensions $\sqrt{KT} \times \sqrt{KT}$. We don't know $T$ until we run the program!

**What happens during proving**:

1. Convert bytecode vector → multiple MLEs (one per field: opcode_mle, rd_mle, rs1_mle, etc.)
2. Commit each MLE using Dory: $C_{\text{opcode}}, C_{\text{rd}}, C_{\text{rs1}}, \ldots \in \mathbb{G}_T$

### Task 2 Output: Dory SRS (Structured Reference String)

**Mathematical object**: Collection of elliptic curve points

#### Prover Setup (ProverSetup)

**Contains**:

$$\text{ProverSetup} = \left\{ \vec{\Gamma}_1, \vec{\Gamma}_2, H_1, H_2 \right\}$$

Where:

- $\vec{\Gamma}_1 = (G_{1,1}, G_{1,2}, \ldots, G_{1,m}) \in \mathbb{G}_1^m$ — Generator vector for **rows**
- $\vec{\Gamma}_2 = (G_{2,1}, G_{2,2}, \ldots, G_{2,n}) \in \mathbb{G}_2^n$ — Generator vector for **columns**
- $H_1 \in \mathbb{G}_1$ — Blinding generator for Layer 1 (Pedersen)
- $H_2 \in \mathbb{G}_2$ — Blinding generator for Layer 2 (AFGHO)
- $m = n = \sqrt{K \cdot T}$ (matrix dimensions)

**Concrete example** (T = 65,536, K = 16):

$$
\begin{array}{l}
\text{Matrix size: } \sqrt{16 \times 65{,}536} = \sqrt{1{,}048{,}576} = 1{,}024 \\
\\
\text{ProverSetup} = \{ \\
\quad \vec{\Gamma}_1: \text{1,024 points in } \mathbb{G}_1 \text{ (each } \sim 48 \text{ bytes)} = 49 \text{ KB} \\
\quad \vec{\Gamma}_2: \text{1,024 points in } \mathbb{G}_2 \text{ (each } \sim 96 \text{ bytes)} = 98 \text{ KB} \\
\quad H_1: \text{1 point in } \mathbb{G}_1 = 48 \text{ bytes} \\
\quad H_2: \text{1 point in } \mathbb{G}_2 = 96 \text{ bytes} \\
\quad \text{Total: } \sim 147 \text{ KB} \\
\}
\end{array}
$$

**What these are used for** (during proving):

- Prover uses $\vec{\Gamma}_1$ and $H_1$ to create Layer 1 Pedersen commitments: $V_i = \langle \vec{M}_i, \vec{\Gamma}_1 \rangle + r_i H_1 \in \mathbb{G}_1$
- Prover uses $\vec{\Gamma}_2$ and $H_2$ to create Layer 2 AFGHO commitment: $C_M = \langle \vec{V}, \vec{\Gamma}_2 \rangle \cdot e(H_1, H_2)^{r_{fin}} \in \mathbb{G}_T$
- Result: **Single element** $C_M \in \mathbb{G}_T$ (192 bytes) commits to entire polynomial!

#### Verifier Setup (VerifierSetup)

**Contains**: Subset of prover setup

$$\text{VerifierSetup} = \left\{ \vec{\Gamma}'_1, \vec{\Gamma}'_2, H_1, H_2 \right\}$$

Where:

- $\vec{\Gamma}'_1 \subset \vec{\Gamma}_1$ — Subset of G_1 generators (only what's needed for verification)
- $\vec{\Gamma}'_2 \subset \vec{\Gamma}_2$ — Subset of G_2 generators
- Same blinding generators $H_1, H_2$

**Size**: Typically ~25 KB (10-20× smaller than prover setup)

**Why smaller?** Verifier only needs to:

- Check pairing equations (doesn't need full generator vectors)
- Verify polynomial evaluations at specific points
- Dory's logarithmic verifier only accesses $O(\log N)$ generators per verification (though G_T exponentiations dominate cost)

### Task 3 Output: RAM Initial State

**Mathematical object**: Vector of 64-bit words

$$\text{initial\_RAM} = (w_0, w_1, \ldots, w_{N-1}) \in (\mathbb{Z}/2^{64}\mathbb{Z})^N$$

Where:

- $w_i \in \{0, 1, \ldots, 2^{64}-1\}$ — 64-bit word at index $i$
- $N$ = number of memory words occupied by bytecode

**Storage format**: `Vec<u64>`

**Example**:
```rust
// Bytecode loaded at address 0x80000000:
// 0x80000000: 0x00000033 (ADD instruction, 4 bytes)
// 0x80000004: 0x00a58593 (ADDI instruction, 4 bytes)

// Packed into 64-bit words (little-endian):
initial_RAM[0] = 0x00a58593_00000033  // First 8 bytes
initial_RAM[1] = 0x...                 // Next 8 bytes
```

**What this is used for** (during proving):

- Twist memory checking needs **initial state** $\text{init}(i)$ for each memory cell
- Final state: $\text{final}(i) = \text{init}(i) + \sum_{\text{writes to } i} \text{value written}$
- Evaluation sumcheck proves: $\sum_i \text{final}(i) = \sum_i \text{init}(i) + \sum_j \text{write\_inc}_j$

**Not committed!** Just stored as plaintext vector for use in witness generation.

### Task 4 Output: Memory Layout

**Mathematical object**: Tuple of address ranges

$$\text{MemoryLayout} = \left\{ (a_{\text{input}}^{\text{start}}, a_{\text{input}}^{\text{end}}), (a_{\text{output}}^{\text{start}}, a_{\text{output}}^{\text{end}}), \ldots \right\}$$

Where each pair $(a^{\text{start}}, a^{\text{end}}) \in \mathbb{Z}^2$ defines a region.

**Storage format**: `struct MemoryLayout` with `u64` fields

**Example**:
```
MemoryLayout = {
  (input_start: 0x0000_0000, input_end: 0x0000_1000),
  (output_start: 0x0000_1000, output_end: 0x0000_2000),
  (program_start: 0x8000_0000, program_end: 0x8001_0000),
  ...
}
```

**What this is used for**:

- **Address remapping function**: $f : \text{guest\_addr} \to \text{witness\_index}$
- Example: $f(0x8000\_0008) = \frac{0x8000\_0008 - 0x8000\_0000}{8} + 1 = 2$
- Used during witness generation to map guest memory accesses to polynomial indices

**Not a polynomial!** Just a configuration struct.

---

## Summary: Preprocessing Outputs as Mathematical Objects

| Task | Math Object | Type | Size | Committed? |
|------|-------------|------|------|------------|
| **Bytecode** | $(\text{instr}_0, \ldots, \text{instr}_{K-1}) \in \mathcal{I}^K$ | Vector of instructions | ~4KB |  No (happens during proving) |
| **Dory SRS** | $(\vec{\Gamma}_1, \vec{\Gamma}_2, H_1, H_2)$ with $\vec{\Gamma}_i \in \mathbb{G}_i^{\sqrt{KT}}$ | Elliptic curve points | ~147 KB prover, ~25 KB verifier | N/A (public parameters) |
| **RAM Init** | $(w_0, w_1, \ldots, w_{N-1}) \in (\mathbb{Z}/2^{64}\mathbb{Z})^N$ | Vector of 64-bit words | ~4KB |  No (used in witness) |
| **Memory Layout** | $\{(a^{\text{start}}_i, a^{\text{end}}_i)\}$ | Address ranges | ~256 bytes |  No (just config) |

**Key insight**: Preprocessing creates **no commitments**! It only:

1. Decodes and stores bytecode in structured form
2. Generates public parameters (SRS) for the commitment scheme
3. Records initial memory state
4. Defines address space layout

**Commitments are created during proving** when we:

1. Convert bytecode → MLEs → commit each to get $C_{\text{opcode}}, C_{\text{rd}}, \ldots \in \mathbb{G}_T$
2. Convert trace → MLEs → commit to get $C_{\text{registers}}, C_{\text{RAM}}, \ldots \in \mathbb{G}_T$
3. Use the SRS generators $(\vec{\Gamma}_1, \vec{\Gamma}_2)$ to create these commitments

---

## Preprocessing Output: JoltProverPreprocessing

**File**: [jolt-core/src/zkvm/mod.rs:155-163](jolt-core/src/zkvm/mod.rs#L155)

```rust
pub struct JoltProverPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub generators: PCS::ProverSetup,  // <-- Dory SRS (Task 2)
                                        // Type: (\Gamma_1, \Gamma_2, H_1, H_2)
                                        // Size: ~147 KB
    pub shared: JoltSharedPreprocessing,
}

pub struct JoltSharedPreprocessing {
    pub bytecode: BytecodePreprocessing,  // <-- Decoded bytecode (Task 1)
                                           // Type: Vec<Instruction>
                                           // Size: ~4 KB
    pub ram: RAMPreprocessing,            // <-- Initial RAM state (Task 3)
                                           // Type: Vec<u64>
                                           // Size: ~4 KB
    pub memory_layout: MemoryLayout,       // <-- Memory configuration (Task 4)
                                           // Type: struct with u64 fields
                                           // Size: ~256 bytes
}
```

### What's in `MemoryLayout`?

**Important**: `MemoryLayout` is NOT the execution trace! It's the **address space blueprint**.

**File**: [common/src/jolt_device.rs](common/src/jolt_device.rs)

```rust
pub struct MemoryLayout {
    // Input/Output regions
    pub input_start: u64,          // e.g., 0x0000_0000
    pub input_end: u64,            // e.g., 0x0000_1000 (4KB)
    pub output_start: u64,         // e.g., 0x0000_1000
    pub output_end: u64,           // e.g., 0x0000_2000 (4KB)

    // Advice regions (prover hints)
    pub trusted_advice_start: u64,
    pub trusted_advice_end: u64,
    pub untrusted_advice_start: u64,
    pub untrusted_advice_end: u64,

    // Program memory
    pub program_start: u64,        // Usually 0x8000_0000 (RAM_START_ADDRESS)
    pub program_end: u64,
    pub program_size: u64,

    // Stack and heap
    pub stack_start: u64,
    pub stack_size: u64,
    pub heap_start: u64,

    pub memory_size: u64,
    pub max_input_size: u64,
    pub max_output_size: u64,
    pub max_trusted_advice_size: u64,
    pub max_untrusted_advice_size: u64,
}
```

**Think of it as**: A map defining which addresses are for inputs, outputs, stack, heap, etc.

**Example layout**:
```
Memory Address Space (Guest View):
0x0000_0000 - 0x0000_1000:  Input region (4KB)
0x0000_1000 - 0x0000_2000:  Output region (4KB)
0x0000_2000 - 0x0000_3000:  Trusted advice (4KB)
0x0000_3000 - 0x0000_4000:  Untrusted advice (4KB)
0x0000_4000 - 0x7FFF_FFFF:  Stack (grows down)
0x8000_0000 - 0x8001_0000:  Program bytecode (64KB)
0x8001_0000 - 0x9000_0000:  Heap (grows up)
```

**Why preprocessing needs this**:

1. **Initial RAM state**: Bytecode loaded at `program_start` address
2. **Witness remapping**: Guest addresses → witness polynomial indices
3. **Output verification**: Verifier knows where to find claimed outputs

**Memory Layout vs Trace** (common confusion):

| Aspect | Memory Layout | Trace |
|--------|---------------|-------|
| **What** | Address space blueprint | Execution history |
| **When** | Preprocessing | Execution (Part 2) |
| **Size** | ~256 bytes | 10KB - 10MB+ |
| **Content** | Address ranges | Instructions executed |
| **Example** | "Stack starts at 0x7FFF_F000" | "Cycle 42: wrote 350 to 0x7FFF_F000" |
| **Changes with input** | No (fixed per program) | Yes (different every run) |

**Address remapping example**:
```rust
// Guest program accesses:
let value = *(0x8000_0008 as *const u64);  // Virtual address

// Memory layout says program_start = 0x8000_0000
// Remap to witness index:
let witness_index = (0x8000_0008 - 0x8000_0000) / 8 + 1 = 2

// RAM polynomial uses witness index:
RAM[2] = value at guest address 0x8000_0008
```

This remapping happens during witness generation using the memory layout from preprocessing!

### What's in `BytecodePreprocessing`?

```rust
pub struct BytecodePreprocessing {
    pub code_size: usize,              // Padded bytecode length (power of 2)
    pub bytecode: Vec<Instruction>,     // Actual RISC-V instructions
    pub pc_map: BytecodePCMapper,       // Maps real addresses → virtual PC
    pub d: usize,                       // Chunking parameter for Shout
}
```

**The `d` parameter** (chunking):

- Bytecode polynomial is large (code_size coefficients)
- Shout protocol chunks it into smaller pieces for efficiency
- `d` controls the chunking factor
- Computed based on bytecode size: `compute_d_parameter(code_size)`

From [jolt-core/src/zkvm/witness.rs](jolt-core/src/zkvm/witness.rs):
```rust
pub fn compute_d_parameter(K: usize) -> usize {
    // Choose d such that K^(1/d) ≈ 2^8 (256 entries per chunk)
    // This balances chunk size vs. number of chunks
    let log_K = K.log_2();
    let d = (log_K as f64 / 8.0).ceil() as usize;
    d.max(1)
}
```

---

## Verifier Preprocessing

**File**: [jolt-core/src/zkvm/mod.rs:116-123](jolt-core/src/zkvm/mod.rs#L116)

```rust
pub struct JoltVerifierPreprocessing<F, PCS>
where
    F: JoltField,
    PCS: CommitmentScheme<Field = F>,
{
    pub generators: PCS::VerifierSetup,  // Much smaller than prover setup!
    pub shared: JoltSharedPreprocessing,  // Same shared data
}
```

**Key difference**: `VerifierSetup` vs `ProverSetup`

- **Prover setup**: ~393 KB (for typical program)
- **Verifier setup**: ~25 KB (for typical program)

**Why smaller?**

- Verifier doesn't need to create commitments (only check them)
- Dory verifier setup only needs subset of elliptic curve points
- No need for full SRS, just verification parameters

**Conversion**:
```rust
impl From<&JoltProverPreprocessing> for JoltVerifierPreprocessing {
    fn from(preprocessing: &JoltProverPreprocessing) -> Self {
        JoltVerifierPreprocessing {
            generators: PCS::setup_verifier(&preprocessing.generators),
            shared: preprocessing.shared.clone(),
        }
    }
}
```

---

## RAM Preprocessing

**File**: [jolt-core/src/zkvm/ram/mod.rs:56-89](jolt-core/src/zkvm/ram/mod.rs#L56)

```rust
pub fn preprocess(memory_init: Vec<(u64, u8)>) -> Self {
    // memory_init: initial bytecode loaded into memory
    // Format: (address, byte) pairs

    let min_bytecode_address = memory_init.iter().map(|(addr, _)| *addr).min().unwrap_or(0);
    let max_bytecode_address = memory_init.iter().map(|(addr, _)| *addr).max().unwrap_or(0);

    // Convert bytes → 64-bit words (RISC-V is 64-bit)
    let num_words = (max_bytecode_address - min_bytecode_address) / 8 + 1;
    let mut bytecode_words = vec![0u64; num_words];

    // Pack bytes into words (little-endian)
    for (address, byte) in memory_init {
        let word_index = (address - min_bytecode_address) / 8;
        let byte_offset = (address % 8);
        bytecode_words[word_index] |= (byte as u64) << (byte_offset * 8);
    }

    Self {
        min_bytecode_address,
        bytecode_words,
    }
}
```

**Purpose**: Initial memory state for RAM consistency checking (Twist).

- Program bytecode is loaded into memory at startup
- Twist needs to know the initial memory state to verify consistency
- This preprocessing stores that initial state

**Relationship to Memory Layout**:

- `MemoryLayout` defines WHERE bytecode should be loaded (e.g., `program_start = 0x8000_0000`)
- `RAMPreprocessing` stores WHAT gets loaded (the actual bytecode words)
- Together they define the initial RAM state before execution begins

---

## Complete Preprocessing Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ User calls: preprocess_prover_{fn_name}(&mut program)       │
└────────────────────┬────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────┐
│ 1. program.decode()                                          │
│    - Parse ELF binary                                        │
│    - Extract bytecode from .text section                    │
│    - Extract initial memory from .data section              │
└────────────────────┬────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────┐
│ 2. JoltRV64IMAC::prover_preprocess()                        │
│    ├─> shared_preprocess()                                  │
│    │   ├─> BytecodePreprocessing::preprocess(bytecode)      │
│    │   │   - Pad to power of 2                              │
│    │   │   - Create PC mapper                               │
│    │   │   - Compute chunking parameter d                   │
│    │   │                                                     │
│    │   └─> RAMPreprocessing::preprocess(memory_init)        │
│    │       - Convert bytes → words                          │
│    │       - Store initial memory state                     │
│    │                                                         │
│    └─> PCS::setup_prover(max_poly_size)                     │
│        - Generate Dory SRS                                  │
│        - Create prover generators                           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────┐
│ Output: JoltProverPreprocessing {                           │
│   generators: DoryProverSetup (~393 KB),                    │
│   shared: JoltSharedPreprocessing {                         │
│     bytecode: BytecodePreprocessing (padded instructions),  │
│     ram: RAMPreprocessing (initial memory),                 │
│     memory_layout: MemoryLayout (address ranges)            │
│   }                                                          │
│ }                                                            │
└─────────────────────────────────────────────────────────────┘
```

---

## Summary: Preprocessing Checklist

**Task 1: Bytecode Preparation** 

- [x] Parse ELF binary
- [x] Extract RISC-V bytecode from .text section
- [x] Pad bytecode to power of 2
- [x] Create PC mapper (address → virtual PC)
- [x] Compute chunking parameter `d`
- [x] Store in `BytecodePreprocessing`
- [ ] ~~Commit to bytecode~~ (happens during proving, not preprocessing!)

**Task 2: Dory SRS Generation** 

- [x] Determine max polynomial size (based on max trace length)
- [x] Calculate matrix dimensions (sqrt(K·T) × sqrt(K·T))
- [x] Generate elliptic curve points (G1 and G2)
- [x] Create `ProverSetup` (full SRS)
- [x] Derive `VerifierSetup` (subset of prover setup)

**Bonus Task: RAM Initialization** 

- [x] Extract initial memory from ELF
- [x] Convert bytes to 64-bit words
- [x] Store in `RAMPreprocessing`

**Output**:

- `JoltProverPreprocessing`: Ready for proving (large ~MB)
- `JoltVerifierPreprocessing`: Ready for verification (small ~KB)

**Key Insight**: Preprocessing is "program-dependent" but "execution-independent"

- Same preprocessing for all runs of the same program
- Different inputs → same preprocessing, different proofs
- Change program → must recompute preprocessing

---

## Connection to Proving

When proving starts:

1. **Emulation** (Part 2):
   - Uses `MemoryLayout` to validate guest memory accesses
   - Loads bytecode from `RAMPreprocessing` into initial RAM state
   - Generates execution trace (one `Cycle` per instruction)

2. **Witness generation** ([jolt-core/src/zkvm/witness.rs](jolt-core/src/zkvm/witness.rs)):
   - Convert execution trace → MLEs
   - Uses `MemoryLayout` to remap guest addresses → witness indices
   - Convert bytecode → MLE (using `BytecodePreprocessing`)

3. **Commitment** ([jolt-core/src/zkvm/dag/jolt_dag.rs:574](jolt-core/src/zkvm/dag/jolt_dag.rs#L574)):
   - Commit to ALL witness polynomials (including bytecode MLE)
   - Uses `generators` from `JoltProverPreprocessing`

4. **Bytecode checking** (Stage 4):
   - Shout protocol proves execution trace matches committed bytecode
   - Uses `d` parameter from `BytecodePreprocessing` for chunking

The preprocessing provides the foundation, and proving builds the actual proof on top!

**Key Insight**: Preprocessing creates the "game board" (memory layout, bytecode, SRS), and proving plays the "game" (execute, witness, prove).

/newpage

# Part 2:  Execution and Witness Generation Deep Dive


## Overview: From Guest Function Call to Execution Trace

Part 2 is where the **actual computation happens**. When you call `prove_{your_function}(input)`, two major things occur:

1. **RISC-V Emulation**: Execute the guest program instruction-by-instruction, recording everything
2. **Witness Generation**: Transform the execution trace into mathematical objects (polynomials) that can be proven

Think of it as: **Run the program → Record what happened → Convert to math**

---

## Theory Connection: The Execution Trace


> **The execution trace** is a complete record of every step the program takes. In Jolt's RISC-V implementation, each cycle is captured in a struct like `RV32IMCycle`, containing:
>
> - The program counter (PC)
> - The instruction executed
> - The values of the source (`rs1`, `rs2`) and destination (`rd`) registers
> - Any memory addresses and values that were accessed
>
> This trace is the foundation for the entire proof. Jolt transforms this cycle-by-cycle record into a set of witness polynomials (MLEs) that can be verified cryptographically.

> Consider a hypothetical 64-bit machine with a simple `ADD` instruction, `ADD rd, rs1, rs2`:
>
> 1. `LOAD r1, 100`  (Load the value 100 into register r1)
>
> 2. `LOAD r2, 250`  (Load the value 250 into register r2)
>
> 3. `ADD r3, r1, r2`   (Add r1 and r2, store in r3)
>
> The execution trace for the third instruction, `ADD r3, r1, r2`, can be viewed as a query to a lookup table. The inputs to the lookup are the operation (`ADD`), the value in `r1` (100), and the value in `r2` (250). The expected output is the result of the addition (350), which will be stored in `r3`.
>
> In Jolt's model, the entire proof of the program's execution is transformed into proving a sequence of such lookups.

**Key insight**: The trace isn't just for debugging—it's the **witness** for the proof!

---

## Part 1: Macro-Generated Prove Function

### Step 1: User Calls `prove_{your_function}()`

From host program:
```rust
let prove_my_function = guest::build_prover_my_function(program, prover_preprocessing);
let (output, proof, program_io) = prove_my_function(input);
```

**What `build_prover_my_function` returns**: A closure that captures the program and preprocessing.

### Step 2: Macro-Generated Prove Function

**File**: [jolt-sdk/macros/src/lib.rs:600+](jolt-sdk/macros/src/lib.rs#L600)

The `#[jolt::provable]` macro generates a `prove_{fn_name}` function:

```rust
pub fn prove_my_function(
    program: jolt::host::Program,
    preprocessing: jolt::JoltProverPreprocessing<jolt::F, jolt::PCS>,
    // ... input arguments ...
) -> (OutputType, JoltProof, JoltDevice) {
    // 1. Serialize inputs
    let mut input_bytes = vec![];
    input_bytes.append(&mut postcard::to_stdvec(&input).unwrap());

    let mut untrusted_advice_bytes = vec![];
    // ... (if any untrusted advice args)

    let mut trusted_advice_bytes = vec![];
    // ... (if any trusted advice args)

    // 2. Call JoltRV64IMAC::prove
    let (proof, program_io, debug_info, duration) = JoltRV64IMAC::prove(
        &preprocessing,
        &program.elf,
        &input_bytes,
        &untrusted_advice_bytes,
        &trusted_advice_bytes,
        trusted_advice_commitment,
    );

    // 3. Deserialize output
    let output: OutputType = postcard::from_bytes(&program_io.outputs).unwrap();

    (output, proof, program_io)
}
```

**Key steps**:

1. Serialize guest inputs to bytes (using `postcard` - a `no_std` serialization format)
2. Call core prove function (`JoltRV64IMAC::prove`)
3. Deserialize outputs from bytes

---

## Part 2: Core Prove Function Entry

### Step 3: JoltRV64IMAC::prove()

**File**: [jolt-core/src/zkvm/mod.rs:257-296](jolt-core/src/zkvm/mod.rs#L257)

```rust
fn prove(
    preprocessing: &JoltProverPreprocessing<F, PCS>,
    elf_contents: &[u8],
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    trusted_advice_commitment: Option<PCS::Commitment>,
) -> (JoltProof, JoltDevice, Option<ProverDebugInfo>, Duration) {
    // 1. Setup memory configuration
    let memory_config = MemoryConfig {
        max_untrusted_advice_size: preprocessing.shared.memory_layout.max_untrusted_advice_size,
        max_trusted_advice_size: preprocessing.shared.memory_layout.max_trusted_advice_size,
        max_input_size: preprocessing.shared.memory_layout.max_input_size,
        max_output_size: preprocessing.shared.memory_layout.max_output_size,
        stack_size: preprocessing.shared.memory_layout.stack_size,
        memory_size: preprocessing.shared.memory_layout.memory_size,
        program_size: Some(preprocessing.shared.memory_layout.program_size),
    };

    // 2. Trace the execution (THE BIG ONE!)
    let (mut trace, final_memory_state, mut program_io) = {
        let _pprof_trace = pprof_scope!("trace");
        guest::program::trace(
            elf_contents,
            None,
            inputs,
            untrusted_advice,
            trusted_advice,
            &memory_config,
        )
    };

    // 3. Log trace statistics
    let num_riscv_cycles: usize = trace.par_iter().map(|cycle| {
        // Count real RISC-V cycles vs virtual instruction cycles
        if let Some(inline_sequence_remaining) =
            cycle.instruction().normalize().inline_sequence_remaining
        {
            if inline_sequence_remaining > 0 {
                return 0; // Virtual instruction (part of inline sequence)
            }
        }
        1 // Real RISC-V instruction
    }).sum();

    tracing::info!(
        "{num_riscv_cycles} raw RISC-V instructions + {} virtual instructions = {} total cycles",
        trace.len() - num_riscv_cycles,
        trace.len(),
    );

    // 4. Pad trace to power of 2 (required for MLEs)
    let trace_length = trace.len();
    let padded_trace_length = (trace.len() + 1).next_power_of_two();
    trace.resize(padded_trace_length, Cycle::NoOp);

    // 5. Truncate trailing zeros from outputs
    program_io.outputs.truncate(
        program_io.outputs.iter().rposition(|&b| b != 0).map_or(0, |i| i + 1)
    );

    // 6. Create state manager and generate proof (Part 3!)
    let state_manager = StateManager::new_prover(
        preprocessing,
        trace,
        program_io,
        trusted_advice_commitment,
        final_memory_state,
    );

    let start = Instant::now();
    let (proof, debug_info) = JoltDAG::prove(state_manager)?;
    let duration = start.elapsed();

    (proof, program_io, debug_info, duration)
}
```

**Key outputs**:

- `trace`: Vec<Cycle> - complete execution record
- `final_memory_state`: Memory - RAM state after execution
- `program_io`: JoltDevice - inputs/outputs/panic status

---

## Part 3: RISC-V Emulation (The Tracer)

### Step 4: guest::program::trace()

**File**: [tracer/src/lib.rs:72-91](tracer/src/lib.rs#L72)

```rust
pub fn trace(
    elf_contents: &[u8],
    elf_path: Option<&std::path::PathBuf>,
    inputs: &[u8],
    untrusted_advice: &[u8],
    trusted_advice: &[u8],
    memory_config: &MemoryConfig,
) -> (Vec<Cycle>, Memory, JoltDevice) {
    // Create lazy iterator for trace generation
    let mut lazy_trace_iter = LazyTraceIterator::new(setup_emulator_with_backtraces(
        elf_contents,
        elf_path,
        inputs,
        untrusted_advice,
        trusted_advice,
        memory_config,
    ));

    // Collect all cycles (this is where emulation happens!)
    let trace: Vec<Cycle> = lazy_trace_iter.by_ref().collect();

    // Extract final memory state
    let final_memory_state = std::mem::take(lazy_trace_iter.final_memory_state.as_mut().unwrap());

    (trace, final_memory_state, lazy_trace_iter.get_jolt_device())
}
```

**Key components**:

1. **LazyTraceIterator**: Generates cycles on-demand (memory efficient)
2. **setup_emulator_with_backtraces**: Initializes RISC-V CPU emulator
3. **collect()**: Actually runs the emulation, producing `Vec<Cycle>`

### Step 5: The CPU Emulator

**File**: [tracer/src/emulator/cpu.rs:90-116](tracer/src/emulator/cpu.rs#L90)

The `Cpu` struct is a full RISC-V processor emulator:

```rust
pub struct Cpu {
    clock: u64,                              // Cycle counter
    pub xlen: Xlen,                          // 32-bit or 64-bit mode
    pub privilege_mode: PrivilegeMode,       // User/Supervisor/Machine
    pub x: [i64; REGISTER_COUNT],            // 32 general-purpose registers
    f: [f64; 32],                            // 32 floating-point registers (unused in Jolt)
    pub pc: u64,                             // Program counter
    csr: [u64; CSR_CAPACITY],                // Control and Status Registers
    pub mmu: Mmu,                            // Memory Management Unit
    reservation: u64,                        // For atomic operations
    is_reservation_set: bool,
    pub trace_len: usize,                    // Current trace length
    executed_instrs: u64,                    // Real RISC-V cycle count
    active_markers: FnvHashMap<u32, ActiveMarker>, // Cycle tracking markers
    pub vr_allocator: VirtualRegisterAllocator,    // Virtual register allocation
    call_stack: VecDeque<CallFrame>,         // For panic backtraces
}
```

**The emulation loop** (simplified):

```rust
impl Cpu {
    pub fn tick(&mut self, trace: Option<&mut Vec<Cycle>>) {
        // 1. Fetch instruction from memory
        let word = self.mmu.fetch_instruction(self.pc);

        // 2. Decode instruction
        let instruction = Instruction::decode(word, self.pc);

        // 3. Execute and trace
        instruction.trace(&mut self, trace);

        // 4. Update program counter (unless instruction modified it)
        if !instruction.modifies_pc() {
            self.pc += instruction.size();
        }
    }
}
```

### Step 6: Instruction Execution and Tracing

**Two core traits**:

#### RISCVInstruction Trait

**File**: [tracer/src/instruction/mod.rs:321-344](tracer/src/instruction/mod.rs#L321)

```rust
pub trait RISCVInstruction:
    std::fmt::Debug + Sized + Copy + Into<Instruction> + ...
{
    const MASK: u32;   // Bit pattern for instruction matching
    const MATCH: u32;  // Expected bits after masking

    type Format: InstructionFormat;         // R-type, I-type, S-type, etc.
    type RAMAccess: Default + Into<RAMAccess>; // Memory access pattern

    fn operands(&self) -> &Self::Format;
    fn new(word: u32, address: u64, validate: bool, compressed: bool) -> Self;

    // Core execution logic (modifies CPU state)
    fn execute(&self, cpu: &mut Cpu, ram_access: &mut Self::RAMAccess);
}
```

#### RISCVTrace Trait

**File**: [tracer/src/instruction/mod.rs:346-373](tracer/src/instruction/mod.rs#L346)

```rust
pub trait RISCVTrace: RISCVInstruction
where
    RISCVCycle<Self>: Into<Cycle>,
{
    fn trace(&self, cpu: &mut Cpu, trace: Option<&mut Vec<Cycle>>) {
        // 1. Create cycle struct to capture state
        let mut cycle: RISCVCycle<Self> = RISCVCycle {
            instruction: *self,
            register_state: Default::default(),
            ram_access: Default::default(),
        };

        // 2. Capture PRE-execution state (register values BEFORE)
        self.operands()
            .capture_pre_execution_state(&mut cycle.register_state, cpu);

        // 3. Execute the instruction (modifies CPU state)
        self.execute(cpu, &mut cycle.ram_access);

        // 4. Capture POST-execution state (register values AFTER)
        self.operands()
            .capture_post_execution_state(&mut cycle.register_state, cpu);

        // 5. Add cycle to trace
        if let Some(trace_vec) = trace {
            trace_vec.push(cycle.into());
        }
    }

    // Default: single instruction. Virtual sequences override this.
    fn inline_sequence(&self, ...) -> Vec<Instruction> {
        vec![(*self).into()]
    }
}
```

**Key insight**: `trace()` wraps `execute()`, capturing state before/after!

### Step 7: Example - ADD Instruction

**File**: [tracer/src/instruction/add.rs](tracer/src/instruction/add.rs)

```rust
// Macro defines instruction metadata
declare_riscv_instr!(
    name   = ADD,
    mask   = 0xfe00707f,    // Bits that must match
    match  = 0x00000033,    // Expected pattern for ADD
    format = FormatR,       // R-type: opcode rd, rs1, rs2
    ram    = ()             // No memory access
);

impl ADD {
    fn exec(&self, cpu: &mut Cpu, _: &mut <ADD as RISCVInstruction>::RAMAccess) {
        // Read source registers
        let rs1_val = cpu.x[self.operands.rs1 as usize];
        let rs2_val = cpu.x[self.operands.rs2 as usize];

        // Perform addition
        let result = rs1_val.wrapping_add(rs2_val);

        // Write to destination register
        cpu.x[self.operands.rd as usize] = cpu.sign_extend(result);
    }
}

// Use default trace implementation
impl RISCVTrace for ADD {}
```

**What gets captured in the cycle**:

- **Pre-execution**: `rs1` value (e.g., 100), `rs2` value (e.g., 250)
- **Execution**: Compute `100 + 250 = 350`
- **Post-execution**: `rd` value (350)
- **Instruction**: The `ADD` opcode and operand indices

### Step 8: The Cycle Enum

**File**: [tracer/src/instruction/mod.rs:394-399](tracer/src/instruction/mod.rs#L394)

```rust
pub enum Cycle {
    NoOp,                         // Padding cycle
    ADD(RISCVCycle<ADD>),
    SUB(RISCVCycle<SUB>),
    XOR(RISCVCycle<XOR>),
    // ... one variant per instruction ...
}

pub struct RISCVCycle<I: RISCVInstruction> {
    instruction: I,                      // The instruction itself
    register_state: I::Format,           // Register operands (pre + post)
    ram_access: I::RAMAccess,            // Memory read/write (if any)
}
```

**Example cycle for `ADD r3, r1, r2`**:
```rust
Cycle::ADD(RISCVCycle {
    instruction: ADD {
        operands: FormatR { rd: 3, rs1: 1, rs2: 2, ... },
        address: 0x80000100,
    },
    register_state: FormatR {
        rs1_val_pre: 100,    // r1 value before
        rs2_val_pre: 250,    // r2 value before
        rd_val_post: 350,    // r3 value after
    },
    ram_access: (),          // No memory access
})
```

This is **exactly the witness** from the theory: (opcode, inputs, outputs)!

---

## Part 4: Witness Generation (Trace → Polynomials)

After emulation completes, we have `Vec<Cycle>`. Now convert to **multilinear polynomial extensions (MLEs)**.

---

## Mathematical Foundation: From Execution to Polynomials

Before diving into the code, let's understand the **mathematical transformation** we're performing.

### The Witness as a Mathematical Object

> The execution trace is a complete record of every step the program takes. [...] This trace is the foundation for the entire proof. Jolt transforms this cycle-by-cycle record into a set of witness polynomials (MLEs) that can be verified cryptographically.

**What is a witness?** In the context of interactive proofs:

$$\text{Witness} = \text{Complete information that proves a claim}$$

For Jolt, the claim is: **"Program P executed correctly with input I and produced output O"**

The witness consists of:

1. **Execution trace**: Sequence of CPU states $(s_0, s_1, \ldots, s_T)$
2. **Polynomial encodings**: MLEs that represent this trace cryptographically

### Why Polynomials? The Power of Multilinear Extensions

**Key insight from sumcheck theory**:

Instead of verifying $T$ individual cycles (expensive!), we:

1. Encode the entire trace as a multilinear polynomial $f : \mathbb{F}^{\log T} \to \mathbb{F}$
2. Use sumcheck to verify properties about $\sum_{x \in \{0,1\}^{\log T}} f(x)$
3. Verifier only evaluates $f$ at a **single random point** $r \in \mathbb{F}^{\log T}$

**Why this works** (Schwartz-Zippel lemma):

- If prover cheats and uses wrong polynomial $f'$, then $f' \neq f$
- Probability that $f'(r) = f(r)$ at random $r$ is $\leq \frac{d}{|\mathbb{F}|}$ (negligible for large field)

**Multilinear Extension (MLE)** of vector $v \in \mathbb{F}^{2^n}$:

$$\widetilde{v}(X_1, \ldots, X_n) = \sum_{b \in \{0,1\}^n} v[b] \cdot \text{eq}(X_1, \ldots, X_n; b)$$

Where $\text{eq}(X; b)$ is the multilinear extension of the indicator function:

$$\text{eq}(X; b) = \prod_{i=1}^n (X_i b_i + (1-X_i)(1-b_i))$$

**Property**: $\widetilde{v}$ agrees with $v$ on the Boolean hypercube:
$$\forall b \in \{0,1\}^n: \widetilde{v}(b) = v[b]$$

But extends uniquely to all of $\mathbb{F}^n$ as a multilinear polynomial!

### The Jolt Lookup Architecture

> Jolt combines all possible operations into a single, giant lookup table, `JOLT_V`:
> $$\text{JOLT\_V}(\text{opcode}, a, b) = (c, \text{flags}) = f_{op}(a, b)$$
>
> A proof of execution for a program trace of length $m$ becomes a sequence of $m$ lookup claims:
> - At step 1: $(c_1, \text{flags}_1) = \text{JOLT\_V}(\text{opcode}_1, a_1, b_1)$
> - At step 2: $(c_2, \text{flags}_2) = \text{JOLT\_V}(\text{opcode}_2, a_2, b_2)$
> - ...

**The witness polynomials encode**:

- **Inputs**: $(a_1, b_1), (a_2, b_2), \ldots, (a_T, b_T)$ → MLEs
- **Outputs**: $(c_1, \text{flags}_1), (c_2, \text{flags}_2), \ldots$ → MLEs
- **Lookup indices**: Decomposed chunks for Shout protocol → One-hot MLEs

### Mathematical Objects in Jolt's Witness

#### 1. Simple Witness Vectors (Direct MLEs)

**Mathematical definition**:

For a witness vector $w = (w_0, w_1, \ldots, w_{T-1}) \in \mathbb{F}^T$ (padded to $T = 2^t$):

$$\widetilde{w}(X_1, \ldots, X_t) : \mathbb{F}^t \to \mathbb{F}$$

**Examples in Jolt**:

**A) Left and Right Instruction Inputs** - The two operands for each instruction:

- `left_instruction_input`: $\widetilde{L} = (a_0, a_1, \ldots, a_{T-1})$ where $a_j$ is the **first operand** at cycle $j$
- `right_instruction_input`: $\widetilde{R} = (b_0, b_1, \ldots, b_{T-1})$ where $b_j$ is the **second operand** at cycle $j$

**What they represent**:
$$\widetilde{L}(j) = a_j \quad \text{(value of left operand at cycle } j\text{)}$$
$$\widetilde{R}(j) = b_j \quad \text{(value of right operand at cycle } j\text{)}$$

Where $j \in \{0,1\}^t$ encodes cycle index in binary.

**Concrete Example - Trace snippet**:

| Cycle | Instruction | rs1 value | rs2 value | Left Input ($a_j$) | Right Input ($b_j$) |
|-------|-------------|-----------|-----------|--------------------|--------------------|
| 0 | `ADD r3, r1, r2` | 100 | 250 | 100 | 250 |
| 1 | `SUB r4, r3, r2` | 350 | 250 | 350 | 250 |
| 2 | `XOR r5, r1, r4` | 100 | 100 | 100 | 100 |
| 3 | `ADDI r6, r5, 42` | 0 | (imm=42) | 0 | 42 |

**The witness vectors**:
$$\widetilde{L} = (100, 350, 100, 0, \ldots) \quad \text{(left operand each cycle)}$$
$$\widetilde{R} = (250, 250, 100, 42, \ldots) \quad \text{(right operand each cycle)}$$

**Key insight**: These are simply the **inputs to the giant JOLT_V lookup table** at each cycle. For R-type instructions, they come from registers `rs1` and `rs2`. For I-type instructions, the right input is the immediate value.

**B) Register Destination Increment** - For Twist memory checking:

- `rd_inc`: $\widetilde{\Delta}_{\text{rd}} = (\Delta_0, \Delta_1, \ldots, \Delta_{T-1})$ where $\Delta_j$ is the **increment written to destination register** at cycle $j$

**Mathematical definition**:
$$\widetilde{\Delta}_{\text{rd}}(j) = \Delta_j = \begin{cases}
\text{rd\_val\_after} - \text{rd\_val\_before} & \text{if cycle } j \text{ writes to a register} \\
0 & \text{otherwise (no write)}
\end{cases}$$

**Why "increment" not "value"?** Twist memory checking uses an **incremental formulation**:

$$\text{final\_register\_state}(r) = \text{initial\_state}(r) + \sum_{j: \text{writes to } r} \Delta_j$$

This proves final state equals initial state plus all writes (memory consistency).

**Concrete Example - Continuing the trace**:

| Cycle | Instruction | rd (dest) | rd value before | rd value after | Increment ($\Delta_j$) |
|-------|-------------|-----------|-----------------|----------------|----------------------|
| 0 | `ADD r3, r1, r2` | r3 | 0 | 350 | **350** |
| 1 | `SUB r4, r3, r2` | r4 | 0 | 100 | **100** |
| 2 | `XOR r5, r1, r4` | r5 | 0 | 0 | **0** |
| 3 | `ADDI r6, r5, 42` | r6 | 0 | 42 | **42** |
| 4 | `LW r7, 0(r3)` | r7 | 0 | 999 | **999** |

**The witness vector**:
$$\widetilde{\Delta}_{\text{rd}} = (350, 100, 0, 42, 999, \ldots)$$

**Verification via Twist sumcheck**:
$$\sum_{j \in \{0,1\}^{\log T}} \widetilde{\Delta}_{\text{rd}}(j) = 350 + 100 + 0 + 42 + 999 + \cdots$$

This sum, combined with initial register state, must equal the final register state (memory consistency check).

**C) RAM Increment** - For Twist memory checking of RAM reads/writes:

- `ram_inc`: $\widetilde{\Delta}_{\text{ram}} = (\Delta_0^{\text{ram}}, \Delta_1^{\text{ram}}, \ldots, \Delta_{T-1}^{\text{ram}})$ where $\Delta_j^{\text{ram}}$ is the **increment written to memory** at cycle $j$

**Mathematical definition**:
$$\widetilde{\Delta}_{\text{ram}}(j) = \Delta_j^{\text{ram}} = \begin{cases}
\text{value\_written} - \text{value\_read} & \text{if cycle } j \text{ writes to RAM (SW/SH/SB)} \\
0 & \text{if cycle } j \text{ only reads RAM (LW/LH/LB)} \\
0 & \text{otherwise (no memory access)}
\end{cases}$$

**Key difference from register increment**:

- **Registers**: Increment = after - before (always has "before" value)
- **RAM**: Increment = written - read (only non-zero for stores, reads don't change memory)

**Why this formulation?** Twist proves memory consistency:
$$\text{final\_memory}(\text{addr}) = \text{initial\_memory}(\text{addr}) + \sum_{j: \text{accesses addr}} \Delta_j^{\text{ram}}$$

For a read-only address: sum of $\Delta$ is zero (no writes) → final = initial 

**Concrete Example - Memory operations**:

Assume initial RAM state: `memory[0x1000] = 0`, `memory[0x1004] = 0`

| Cycle | Instruction | Address | Operation | Value Read | Value Written | Increment ($\Delta_j^{\text{ram}}$) |
|-------|-------------|---------|-----------|------------|---------------|-----------------------------------|
| 0 | `ADD r3, r1, r2` | - | (no memory) | - | - | **0** |
| 1 | `SW r3, 0(r10)` | 0x1000 | **STORE** | 0 (old) | 350 (new) | **350** |
| 2 | `LW r7, 0(r10)` | 0x1000 | **LOAD** | 350 | - | **0** (reads don't increment) |
| 3 | `ADDI r8, r7, 50` | - | (no memory) | - | - | **0** |
| 4 | `SW r8, 4(r10)` | 0x1004 | **STORE** | 0 (old) | 400 (new) | **400** |
| 5 | `SW r3, 0(r10)` | 0x1000 | **STORE** | 350 (old) | 350 (same) | **0** (no change) |
| 6 | `LW r9, 4(r10)` | 0x1004 | **LOAD** | 400 | - | **0** (reads don't increment) |

**The witness vector** (one entry per cycle $j$):
$$\widetilde{\Delta}_{\text{ram}} = (\Delta_0^{\text{ram}}, \Delta_1^{\text{ram}}, \Delta_2^{\text{ram}}, \Delta_3^{\text{ram}}, \Delta_4^{\text{ram}}, \Delta_5^{\text{ram}}, \Delta_6^{\text{ram}}, \ldots)$$
$$= (0, 350, 0, 0, 400, 0, 0, \ldots)$$

 **CRITICAL QUESTION: How does this vector track which address each increment goes to?**

**Answer**: It doesn't on its own! The address information is stored **separately** in $\widetilde{\text{mem}}_i(j,k)$ (the RamRa one-hot polynomials explained in section 2 below).

**The key insight**: Twist uses **TWO separate polynomials** working together:

1. **$\widetilde{\Delta}_{\text{ram}}(j)$**: "What increment happened at cycle $j$?"
   - Domain: $\mathbb{F}^{\log T}$ (indexed by cycle only)
   - $\Delta_1^{\text{ram}} = 350$ (some increment at cycle 1)
   - $\Delta_4^{\text{ram}} = 400$ (some increment at cycle 4)
   - **Missing info**: Which address?

2. **$\widetilde{\text{mem}}_i(j, k)$**: "Which address was accessed at cycle $j$?" (see section 2.C below for full explanation)
   - Domain: $\mathbb{F}^{\log T} \times \mathbb{F}^{\log M}$ (indexed by cycle AND address)
   - $\widetilde{\text{mem}}_i(1, k_1) = 1$ where $k_1$ maps to 0x1000 (cycle 1 accessed 0x1000)
   - $\widetilde{\text{mem}}_i(4, k_2) = 1$ where $k_2$ maps to 0x1004 (cycle 4 accessed 0x1004)
   - **Provides**: Address routing for increments

**How Twist connects them**:

For each address $k$, compute final state using the one-hot $\widetilde{\text{mem}}$ as a **selector**:
$$\text{final}(k) = \text{initial}(k) + \sum_{j=0}^{T-1} \widetilde{\text{mem}}_i(j, k) \cdot \widetilde{\Delta}_{\text{ram}}(j)$$

The one-hot property means $\widetilde{\text{mem}}_i(j, k) = 1$ only when cycle $j$ accessed address $k$, so only those increments get added.

**Concretely for address** $k_1 = 0x1000$:

Cycles 1, 2, 5 accessed 0x1000, so:
$$\text{final}(k_1) = 0 + \sum_{j=0}^{6} \widetilde{\text{mem}}(j, k_1) \cdot \Delta_j^{\text{ram}}$$
$$= 0 + \underbrace{\widetilde{\text{mem}}(1, k_1)}_{=1} \cdot \underbrace{\Delta_1}_{=350} + \underbrace{\widetilde{\text{mem}}(2, k_1)}_{=1} \cdot \underbrace{\Delta_2}_{=0} + \underbrace{\widetilde{\text{mem}}(5, k_1)}_{=1} \cdot \underbrace{\Delta_5}_{=0} + \text{(other terms are 0)}$$
$$= 1 \cdot 350 + 1 \cdot 0 + 1 \cdot 0 = 350 \; \checkmark$$

**For address** $k_2 = 0x1004$:

Cycles 4, 6 accessed 0x1004, so:
$$\text{final}(k_2) = 0 + \sum_{j=0}^{6} \widetilde{\text{mem}}(j, k_2) \cdot \Delta_j^{\text{ram}}$$
$$= 0 + \underbrace{\widetilde{\text{mem}}(4, k_2)}_{=1} \cdot \underbrace{\Delta_4}_{=400} + \underbrace{\widetilde{\text{mem}}(6, k_2)}_{=1} \cdot \underbrace{\Delta_6}_{=0} + \text{(other terms are 0)}$$
$$= 1 \cdot 400 + 1 \cdot 0 = 400 \; \checkmark$$

**Why this design?**

- **Efficiency**: One simple vector $\widetilde{\Delta}_{\text{ram}}$ (size $T$) instead of separate vectors per address (size $M \times T$)
- **Linking**: The one-hot $\widetilde{\text{mem}}$ acts as a **selector** routing each increment to the correct address
- **Separation of concerns**:
  - **Value changes**: $\widetilde{\Delta}_{\text{ram}}(j)$ - "how much changed at cycle $j$?"
  - **Address routing**: $\widetilde{\text{mem}}_i(j,k)$ - "which address was accessed at cycle $j$?"

**Key observations**:

- **Loads (LW) contribute $\Delta = 0$**: They don't change memory, only read it
- **Stores (SW) contribute $\Delta = \text{new} - \text{old}$**: The change in memory
- **Overwriting same value**: Cycle 5 writes 350 to address already containing 350 → $\Delta = 0$
- **Twist proves**: Sum of all increments to each address equals the final state change

#### 2. One-Hot Witness Matrices (For Shout Lookups)

**Mathematical definition**:

For lookups into a table of size $N$, the one-hot polynomial is:

$$\widetilde{\text{ra}}(j, k) : \mathbb{F}^{\log T} \times \mathbb{F}^{\log N} \to \mathbb{F}$$

Such that:
$$\widetilde{\text{ra}}(j, k) = \begin{cases}
1 & \text{if cycle } j \text{ accessed table entry } k \\
0 & \text{otherwise}
\end{cases}$$

**Sumcheck property** (key to Shout):

$$\sum_{j \in \{0,1\}^{\log T}} \sum_{k \in \{0,1\}^{\log N}} \widetilde{\text{ra}}(j, k) \cdot (\text{value\_read}(k) - \text{value\_written}(k)) = 0$$

This proves: "Every value read from the table matches what was written"

#### **Detailed Explanation of This Formula**

Let's break down what this sumcheck actually verifies:

**Setup**:

- We have $T$ lookups (one per cycle) into a table of size $N$
- Each lookup: cycle $j$ reads entry $k$ from the table
- $\widetilde{\text{ra}}(j, k)$ is the one-hot polynomial encoding these lookups

**The formula has two sums**:

$$\underbrace{\sum_{j \in \{0,1\}^{\log T}}}_{\text{For each cycle}} \underbrace{\sum_{k \in \{0,1\}^{\log N}}}_{\text{For each table entry}} \underbrace{\widetilde{\text{ra}}(j, k)}_{\substack{\text{1 if cycle } j \\ \text{looked up entry } k \\ \text{0 otherwise}}} \cdot \underbrace{(\text{value\_read}(k) - \text{value\_written}(k))}_{\text{Difference between claimed and actual table values}}$$

**What each term means** ( **terminology is confusing here!**):

1. **$\text{value\_read}(k)$**: The value the **prover claims** they got from table entry $k$
   - This comes from the execution trace
   - Example: Prover says "When I looked up entry 42 in the ADD table, I got 100"

2. **$\text{value\_written}(k)$**: The **actual/correct value** at table entry $k$ ( **confusing name!**)
   - **NOT about temporal "writing"** in the instruction lookup case!
   - Better mental model: **"the ground truth value at index $k$"**

   **Two different contexts**:

   **A) For read-only lookup tables (instruction lookups via Shout)**:
   - `value_written(k)` = **Pre-computed table value** (the "definition" of the table)
   - Example: $\text{ADD\_table}(10, 20) = 30$ (computed once during preprocessing, never changes)
   - "written" here means "what's defined in the table spec"
   - The table is **static** - no temporal writes happen!

   **B) For read-write memory (RAM/registers via Twist)**:
   - `value_written(k)` = **Last value written** to address $k$ temporally
   - Example: If address 0x1000 was written with value 42 at cycle 10, then for cycle 15 reading it: `value_written` = 42
   - "written" here genuinely means "most recent temporal write"
   - The memory state **evolves** over time!

3. **$\widetilde{\text{ra}}(j, k)$**: The one-hot polynomial that's 1 only when cycle $j$ accessed entry $k$
   - Acts as a "selector" - only non-zero for actual lookups
   - Example: If cycle 5 looked up entry 42, then $\widetilde{\text{ra}}(5, 42) = 1$ and $\widetilde{\text{ra}}(5, i) = 0$ for all $i \neq 42$

**Why the sum equals zero**:

For each cycle $j$ that looks up entry $k$:

- $\widetilde{\text{ra}}(j, k) = 1$
- If the lookup is **correct**: $\text{value\_read}(k) = \text{value\_written}(k)$
- So: $1 \cdot (\text{value\_read}(k) - \text{value\_written}(k)) = 1 \cdot 0 = 0$

For all other $(j, k)$ pairs:

- $\widetilde{\text{ra}}(j, k) = 0$
- So: $0 \cdot (\ldots) = 0$

**Total sum**: $0 + 0 + \cdots + 0 = 0$

**If prover cheats**:

- Suppose at cycle 5, prover claims they looked up entry 42 and got 999 (but correct value is 100)
- Then: $\widetilde{\text{ra}}(5, 42) \cdot (999 - 100) = 1 \cdot 899 = 899 \neq 0$
- Sum is **non-zero** → Verifier rejects!

**Concrete Example: Instruction Lookup (Read-Only Table)**

For `ADD r3, r1, r2` with `r1=10`, `r2=20` at cycle 5:

- Instruction decomposes into 16 chunks (4 bits each)
- First chunk lookup: 10 (4 bits) + 20 (4 bits) should give 30 (4 bits)
- Table entry: $k = (10, 20)$ in binary encoding
- Pre-computed table: $\text{ADD\_table}(10, 20) = 30$ ← This is `value_written(k)` (the "ground truth")

**The sumcheck verifies**:
$$\widetilde{\text{ra}}_0(5, k) \cdot (\underbrace{\text{prover's claimed result}}_{\text{value\_read}(k)} - \underbrace{30}_{\text{value\_written}(k) = \text{table definition}}) = 0$$

This forces prover to claim the correct result (30), otherwise the sum won't be zero!

**Why "value_written" for static tables?** The term comes from the Shout protocol's unification of read-only lookups with read-write memory checking. In the protocol formalism:

- For static tables: "written" = "what was written into the table definition during preprocessing"
- For dynamic memory: "written" = "what was last written temporally during execution"

The math is the same, but the meaning differs!

**Examples of One-Hot Polynomials in Jolt**:

**A) InstructionRa(i)** - Instruction lookup one-hot:

- `InstructionRa(i)`: Chunk $i$ of instruction lookup (16 chunks total for 64-bit ops)
- Example: $\widetilde{\text{InstructionRa}}_0(j, k) = 1$ if cycle $j$ performed chunk 0 lookup at table index $k$
- See the ADD example above (cycle 5, chunk 0, looking up entry for inputs 10,20)

**B) BytecodeRa(i)** - Bytecode fetch one-hot ( **"Ra" means "read address", same concept as InstructionRa but for bytecode!**):

- `BytecodeRa(i)`: Chunk $i$ of bytecode address lookup
- **What it tracks**: Which bytecode instruction was fetched at each cycle
- **Why one-hot?** Proves that the instruction fetched matches the committed bytecode

**Concrete Example - Bytecode Lookup**:

Assume we have a tiny program with 4 instructions in bytecode:

| Bytecode Address | Instruction | Encoding |
|------------------|-------------|----------|
| 0x0 | `ADD r3, r1, r2` | 0x00208233 |
| 0x4 | `SUB r4, r3, r2` | 0x402182B3 |
| 0x8 | `XOR r5, r1, r4` | 0x004142B3 |
| 0xC | `ADDI r6, r5, 42` | 0x02A28313 |

**Execution trace** (simplified - assume no jumps, sequential execution):

| Cycle | PC (bytecode addr) | Instruction Fetched | BytecodeRa Index $k$ |
|-------|-------------------|---------------------|----------------------|
| 0 | 0x0 | `ADD r3, r1, r2` | 0 |
| 1 | 0x4 | `SUB r4, r3, r2` | 1 |
| 2 | 0x8 | `XOR r5, r1, r4` | 2 |
| 3 | 0xC | `ADDI r6, r5, 42` | 3 |

**The one-hot polynomial** $\widetilde{\text{BytecodeRa}}(j, k)$:

For a non-chunked example (simplified), the matrix would be:

$$\widetilde{\text{BytecodeRa}} = \begin{bmatrix}
j=0: & [1, & 0, & 0, & 0] & \leftarrow \text{Cycle 0 fetched bytecode index 0} \\
j=1: & [0, & 1, & 0, & 0] & \leftarrow \text{Cycle 1 fetched bytecode index 1} \\
j=2: & [0, & 0, & 1, & 0] & \leftarrow \text{Cycle 2 fetched bytecode index 2} \\
j=3: & [0, & 0, & 0, & 1] & \leftarrow \text{Cycle 3 fetched bytecode index 3} \\
\end{bmatrix}$$

**The sumcheck verifies**:
$$\sum_{j=0}^{3} \sum_{k=0}^{3} \widetilde{\text{BytecodeRa}}(j, k) \cdot (\text{instruction\_fetched}(j) - \text{committed\_bytecode}(k)) = 0$$

This proves:

- Cycle 0 fetched the instruction at bytecode index 0 (ADD) 
- Cycle 1 fetched the instruction at bytecode index 1 (SUB) 
- Cycle 2 fetched the instruction at bytecode index 2 (XOR) 
- Cycle 3 fetched the instruction at bytecode index 3 (ADDI) 

**Key insight**: BytecodeRa is exactly analogous to InstructionRa:

- **InstructionRa**: Proves "cycle $j$ looked up the correct entry in the instruction behavior table"
- **BytecodeRa**: Proves "cycle $j$ fetched the correct instruction from the committed bytecode"

Both use the same one-hot polynomial structure and Shout sumcheck!

**C) RamRa(i)** - RAM address one-hot (also written as $\widetilde{\text{mem}}_i$):

- `RamRa(i)`: Chunk $i$ of RAM address lookup
- **Mathematical notation**: $\widetilde{\text{mem}}_i(j, k)$ where $i$ is the chunk index
- **What it tracks**: Which RAM address was accessed at each cycle (chunked for efficiency)
- **Why one-hot?** Proves that memory accesses happened at the claimed addresses

**Concrete Example - RAM Address Lookup**:

Let's revisit our RAM example from section C above, but now focus on the **address lookups**.

**Memory operations** (from earlier example):

| Cycle $j$ | Instruction | Address | RamRa Index $k$ (simplified) |
|-----------|-------------|---------|------------------------------|
| 0 | `ADD r3, r1, r2` | - | - (no memory access) |
| 1 | `SW r3, 0(r10)` | **0x1000** | $k_1$ (maps to 0x1000) |
| 2 | `LW r7, 0(r10)` | **0x1000** | $k_1$ (same address) |
| 3 | `ADDI r8, r7, 50` | - | - (no memory access) |
| 4 | `SW r8, 4(r10)` | **0x1004** | $k_2$ (maps to 0x1004) |
| 5 | `SW r3, 0(r10)` | **0x1000** | $k_1$ (back to 0x1000) |
| 6 | `LW r9, 4(r10)` | **0x1004** | $k_2$ (back to 0x1004) |

**Important**: Addresses are typically chunked (e.g., 64-bit address → 16 chunks of 4 bits). For simplicity, assume 2 possible addresses.

**The one-hot polynomial** $\widetilde{\text{mem}}_0(j, k)$ (chunk 0, simplified):

For cycles that access memory, the one-hot polynomial indicates which address:

$$\widetilde{\text{mem}}_0 = \begin{bmatrix}
j=0: & [0, & 0] & \leftarrow \text{No memory access} \\
j=1: & [1, & 0] & \leftarrow \text{Cycle 1 accessed address } k_1 \text{ (0x1000)} \\
j=2: & [1, & 0] & \leftarrow \text{Cycle 2 accessed address } k_1 \text{ (0x1000)} \\
j=3: & [0, & 0] & \leftarrow \text{No memory access} \\
j=4: & [0, & 1] & \leftarrow \text{Cycle 4 accessed address } k_2 \text{ (0x1004)} \\
j=5: & [1, & 0] & \leftarrow \text{Cycle 5 accessed address } k_1 \text{ (0x1000)} \\
j=6: & [0, & 1] & \leftarrow \text{Cycle 6 accessed address } k_2 \text{ (0x1004)} \\
\end{bmatrix}$$

**Key observations**:

- **Multiple cycles can access same address**: Cycles 1, 2, 5 all access 0x1000
- **No memory access cycles**: Cycles 0, 3 have all zeros (no lookup)
- **Different from increment polynomial**: $\widetilde{\text{mem}}$ tracks **which address**, $\widetilde{\Delta}_{\text{ram}}$ tracks **what value change**

**Connection to Twist Protocol**:

Twist needs to prove:

1. **Which addresses were accessed** ($\widetilde{\text{mem}}_i$ proves this via Shout)
2. **What increments occurred** ($\widetilde{\Delta}_{\text{ram}}$ tracks this)

Together they prove: "For each memory access at cycle $j$ to address $k$, the value changed by $\Delta_j$"

**The Twist sumcheck uses both**:

$$\sum_{j \in \{0,1\}^{\log T}} \sum_{k \in \{0,1\}^{\log M}} \widetilde{\text{mem}}_i(j, k) \cdot \left( \text{address\_claimed}(j) - \text{address\_actual}(k) \right) = 0$$

This proves: "Every memory access claimed the correct address"

**Then separately**:
$$\text{final\_mem}(k) = \text{initial\_mem}(k) + \sum_{j: \text{accesses } k} \widetilde{\Delta}_{\text{ram}}(j)$$

This proves: "The increments sum to the correct final state"

**Why chunk addresses?** A 64-bit address has $2^{64}$ possible values - too large for one-hot polynomial!

- **Solution**: Split address into 16 chunks of 4 bits each
- Each chunk: $2^4 = 16$ possible values (tractable!)
- Need 16 separate one-hot polynomials: $\widetilde{\text{mem}}_0, \ldots, \widetilde{\text{mem}}_{15}$

**Comparison of the three "Ra" polynomials**:

| Polynomial | What it tracks | Table size | Purpose |
|------------|----------------|------------|---------|
| $\widetilde{\text{ra}}_i(j,k)$ (InstructionRa) | Which instruction behavior entry | $2^{128}$ (chunked to $2^8$) | Prove correct instruction outputs |
| $\widetilde{\text{bc}}_i(j,k)$ (BytecodeRa) | Which bytecode instruction | $K$ (program size) | Prove correct instruction fetch |
| $\widetilde{\text{mem}}_i(j,k)$ (RamRa) | Which RAM address | $M$ (memory size, chunked) | Prove correct memory addressing |

All three use the **same one-hot structure** and **same Shout sumcheck protocol**!

#### 3. Increment Polynomials (For Twist Memory Checking)

**Mathematical definition** (from Twist protocol):

Memory cell with initial value $\text{init}(a)$ and increments $\Delta_1, \Delta_2, \ldots$ at address $a$:

$$\text{final}(a) = \text{init}(a) + \sum_{i : \text{access to } a} \Delta_i$$

**The increment MLE**:

$$\widetilde{\Delta}(j) = \begin{cases}
\text{value\_written} - \text{value\_read} & \text{if cycle } j \text{ writes to memory/register} \\
0 & \text{otherwise}
\end{cases}$$

**Sumcheck verification**:

$$\sum_{a} \text{final}(a) = \sum_{a} \text{init}(a) + \sum_{j} \widetilde{\Delta}(j)$$

**Examples in Jolt**:

- `RdInc`: Register write increments — $\Delta_j = \text{rd\_val\_post} - \text{rd\_val\_pre}$
- `RamInc`: Memory write increments — $\Delta_j = \text{value\_written} - \text{value\_read}$

### Complete Mathematical Picture

For a trace of length $T = 2^t$ cycles:

**Witness = Set of MLEs**:

$$W = \{\widetilde{L}, \widetilde{R}, \widetilde{\Delta}_{\text{rd}}, \widetilde{\Delta}_{\text{ram}}, \widetilde{\text{ra}}_0, \ldots, \widetilde{\text{ra}}_{15}, \ldots\}$$

Where each MLE has domain:

- Simple witnesses: $\mathbb{F}^{\log T}$ (just cycle index)
- One-hot witnesses: $\mathbb{F}^{\log T} \times \mathbb{F}^{\log N}$ (cycle index × table index)

**Size comparison**:

| Representation | Size | Committed Size |
|---------------|------|----------------|
| **Raw trace** | $T \times \text{cycle\_size}$ ≈ 10KB-10MB | N/A |
| **MLE evaluations** | $T \times \text{num\_polys}$ ≈ same | N/A |
| **Dory commitments** | $\text{num\_polys} \times 192$ bytes | ~10KB total! |
| **Opening proof** | $O(\log T)$ per polynomial | ~18KB per poly |

**The magic**: Entire trace compressed to ~10KB of commitments, with logarithmic verification!

---

### Step 9: Committed Polynomial Types

**File**: [jolt-core/src/zkvm/witness.rs:47-80](jolt-core/src/zkvm/witness.rs#L47)

```rust
pub enum CommittedPolynomial {
    /* R1CS aux variables */
    LeftInstructionInput,      // Left operand (rs1 or PC)
    RightInstructionInput,     // Right operand (rs2 or immediate)
    WriteLookupOutputToRD,     // Should write result to rd?
    WritePCtoRD,               // Should write PC to rd (JAL, JALR)?
    ShouldBranch,              // Is this a branch instruction?
    ShouldJump,                // Is this a jump instruction?

    /* Twist/Shout witnesses */
    RdInc,                     // Register write increments (Twist)
    RamInc,                    // Memory write increments (Twist)
    InstructionRa(usize),      // Instruction lookup addresses (Shout, d=16)
    BytecodeRa(usize),         // Bytecode lookup addresses (Shout, d varies)
    RamRa(usize),              // RAM addresses (Twist, d varies)
}
```

**Mathematical correspondence**:

| Rust Enum | Math Object | Domain | Purpose |
|-----------|-------------|--------|---------|
| `LeftInstructionInput` | $\widetilde{L}(j)$ | $\mathbb{F}^{\log T}$ | Left operand values |
| `RightInstructionInput` | $\widetilde{R}(j)$ | $\mathbb{F}^{\log T}$ | Right operand values |
| `WriteLookupOutputToRD` | $\widetilde{w}_{\text{rd}}(j)$ | $\mathbb{F}^{\log T}$ | Boolean: write to rd? |
| `ShouldBranch` | $\widetilde{b}(j)$ | $\mathbb{F}^{\log T}$ | Boolean: is branch? |
| `RdInc` | $\widetilde{\Delta}_{\text{rd}}(j)$ | $\mathbb{F}^{\log T}$ | Register increment |
| `RamInc` | $\widetilde{\Delta}_{\text{ram}}(j)$ | $\mathbb{F}^{\log T}$ | Memory increment |
| `InstructionRa(i)` | $\widetilde{\text{ra}}_i(j, k)$ | $\mathbb{F}^{\log T} \times \mathbb{F}^{\log N}$ | Lookup chunk $i$ |
| `BytecodeRa(i)` | $\widetilde{\text{bc}}_i(j, k)$ | $\mathbb{F}^{\log T} \times \mathbb{F}^{\log K}$ | Bytecode chunk $i$ |
| `RamRa(i)` | $\widetilde{\text{mem}}_i(j, k)$ | $\mathbb{F}^{\log T} \times \mathbb{F}^{\log M}$ | RAM addr chunk $i$ |

**Why these specific polynomials?**

- **R1CS variables** ($\widetilde{L}, \widetilde{R}, \widetilde{w}_{\text{rd}}, etc.$): Needed for Spartan's constraint system to prove VM wiring
- **Twist witnesses** ($\widetilde{\Delta}_{\text{rd}}, \widetilde{\Delta}_{\text{ram}}$): Needed for register/RAM memory checking via incremental state updates
- **Shout witnesses** ($\widetilde{\text{ra}}_i, \widetilde{\text{bc}}_i$): Needed for instruction/bytecode lookup arguments (one-hot encodings)

### Step 10: The Witness Generation Pipeline

**File**: [jolt-core/src/zkvm/witness.rs:200+]

**Goal**: Transform execution trace $(s_0, s_1, \ldots, s_{T-1})$ into ~50 committed witness polynomials.

**The pipeline** (three sub-steps):

```
Step 10A: Allocate WitnessData
    ↓
Step 10B: Extract data from each cycle in parallel
    ↓
Step 10C: Convert to MLEs and commit
```

Let's go through each with mathematical detail:

---

#### Step 10A: Allocate WitnessData Structure

**File**: [jolt-core/src/zkvm/witness.rs:84-121](jolt-core/src/zkvm/witness.rs#L84)

**Mathematical setup**: We need to store ~50 vectors, each of length $T$ (trace length, padded to $T = 2^t$).

```rust
struct WitnessData {
    // Simple polynomial coefficients (one per cycle)
    left_instruction_input: Vec<u64>,        // Size T
    right_instruction_input: Vec<i128>,      // Size T
    write_lookup_output_to_rd: Vec<u8>,      // Size T
    write_pc_to_rd: Vec<u8>,                 // Size T
    should_branch: Vec<u8>,                  // Size T
    should_jump: Vec<u8>,                    // Size T
    rd_inc: Vec<i128>,                       // Size T
    ram_inc: Vec<i128>,                      // Size T

    // One-hot polynomial indices (for Twist/Shout)
    instruction_ra: [Vec<Option<u8>>; 16],   // 16 chunks × T entries
    bytecode_ra: Vec<Vec<Option<u8>>>,       // d chunks × T entries
    ram_ra: Vec<Vec<Option<u8>>>,            // d chunks × T entries
}
```

**Memory allocation**:
```rust
let witness_data = WitnessData {
    left_instruction_input: vec![0u64; trace_len],
    right_instruction_input: vec![0i128; trace_len],
    // ... all other fields initialized to size trace_len
};
```

**What we have now**: Empty vectors ready to be filled.

$$\text{WitnessData} = \{\vec{v}_1, \vec{v}_2, \ldots, \vec{v}_{50}\} \text{ where each } \vec{v}_i \in \mathbb{F}^T \text{ (all zeros)}$$

#### Step 10B: Extract Data from Each Cycle (Parallel Processing)

**File**: [jolt-core/src/zkvm/witness.rs:286-375](jolt-core/src/zkvm/witness.rs#L286)

**Mathematical operation**: For each cycle $j \in \{0, 1, \ldots, T-1\}$, extract relevant data and fill vectors.

```rust
trace.par_iter().enumerate().for_each(|(j, cycle)| {
    process_cycle(j, cycle, &witness_data, preprocessing);
});
```

**What `process_cycle` does** (concrete example for `ADD r3, r1, r2` at cycle $j = 5$):

```rust
fn process_cycle(j: usize, cycle: &Cycle, data: &WitnessData, ...) {
    // Extract from cycle state
    let (left_val, right_val) = cycle.instruction_inputs();     // (100, 250)
    let (rd_pre, rd_post) = cycle.rd_values();                  // (0, 350)
    let flags = cycle.circuit_flags();

    // Fill simple vectors
    data.left_instruction_input[j] = left_val;                  // 100
    data.right_instruction_input[j] = right_val;                // 250
    data.rd_inc[j] = rd_post - rd_pre;                         // 350
    data.write_lookup_output_to_rd[j] = flags.write_to_rd;     // 1
    data.should_branch[j] = flags.is_branch;                   // 0

    // Decompose instruction lookup into 16 chunks
    let lookup_index = compute_lookup_index(left_val, right_val, opcode);
    for chunk_i in 0..16 {
        let chunk_value = (lookup_index >> (4 * chunk_i)) & 0xF;  // Extract 4 bits
        data.instruction_ra[chunk_i][j] = Some(chunk_value as u8);
    }

    // Bytecode PC chunking
    let pc = cycle.program_counter();
    for chunk_i in 0..bytecode_d {
        let chunk_value = (pc >> (chunk_size * chunk_i)) & chunk_mask;
        data.bytecode_ra[chunk_i][j] = Some(chunk_value as u8);
    }

    // RAM address (if memory access)
    if let Some(addr) = cycle.ram_address() {
        data.ram_inc[j] = cycle.ram_value_change();            // written - read
        for chunk_i in 0..ram_d {
            let chunk_value = (addr >> (chunk_size * chunk_i)) & chunk_mask;
            data.ram_ra[chunk_i][j] = Some(chunk_value as u8);
        }
    }
}
```

**After parallel processing**, all vectors are filled:

$$\text{left\_instruction\_input} = (100, 250, 350, 0, 42, \ldots) \in \mathbb{F}^T$$
$$\text{rd\_inc} = (350, 100, 0, 42, 999, \ldots) \in \mathbb{F}^T$$
$$\text{instruction\_ra}[0] = (\text{Some}(10), \text{Some}(15), \text{Some}(5), \ldots) \text{ (indices for chunk 0)}$$

---

#### Step 10C: Convert Vectors to MLEs

**File**: [jolt-core/src/zkvm/witness.rs:386-430](jolt-core/src/zkvm/witness.rs#L386)

**Mathematical operation**: Transform raw data into multilinear polynomials.

 **IMPORTANT DISTINCTION**: Two different conversions happen in Step 10C!

---

### **Conversion Type 1: Simple Vectors (Already Dense) → MLE**

**Applies to**: `left_input`, `right_input`, `rd_inc`, `ram_inc`, `should_branch`, etc.

**From Step 10B**: These vectors are **already dense** (one value per cycle):
```rust
data.left_instruction_input = [100, 250, 350, 0, 42, ...]  // Size T, already dense!
data.rd_inc = [350, 100, 0, 42, 999, ...]                 // Size T, already dense!
```

**Step 10C conversion**: Just type conversion $\mathbb{Z} \to \mathbb{F}$ and wrapping in MLE:

```rust
// No sparse→dense conversion needed! Just field conversion
let coeffs: Vec<F> = data.left_instruction_input
    .iter()
    .map(|&x| F::from(x))    // u64 → field element
    .collect();

let mle_L = MultilinearPolynomial::from(coeffs);  // Wrap as MLE
```

**Mathematical definition** of the MLE $\widetilde{L}$:

Given **dense** vector $\vec{L} = (L_0, L_1, \ldots, L_{T-1}) \in \mathbb{F}^T$ where $T = 2^t$:

$$\widetilde{L}(X_1, \ldots, X_t) = \sum_{j \in \{0,1\}^t} L_j \cdot \text{eq}(X_1, \ldots, X_t; j_1, \ldots, j_t)$$

Where $\text{eq}(X; j)$ is the Lagrange basis polynomial:

$$\text{eq}(X_1, \ldots, X_t; j_1, \ldots, j_t) = \prod_{i=1}^{t} (X_i j_i + (1-X_i)(1-j_i))$$

**Properties**:

- **Input**: Dense vector of size $T$ (one value per cycle)
- **Output**: MLE with domain $\mathbb{F}^{\log T}$ (t-variate)
- On Boolean hypercube: $\widetilde{L}(j) = L_j$ for $j \in \{0,1\}^t$
- Extends uniquely to all $\mathbb{F}^t$ as multilinear polynomial

**Concrete example** with $T = 1024 = 2^{10}$ and $\vec{L} = (100, 250, 350, \ldots)$:

$$\widetilde{L}(0,0,0,0,0,0,0,0,0,0) = 100 \quad \text{(cycle 0)}$$
$$\widetilde{L}(1,0,0,0,0,0,0,0,0,0) = 250 \quad \text{(cycle 1)}$$
$$\widetilde{L}(0,1,0,0,0,0,0,0,0,0) = 350 \quad \text{(cycle 2)}$$

But also evaluable at any point, e.g.:
$$\widetilde{L}(0.5, 0.3, 0.7, \ldots) = \text{some field element}$$

**Memory**: Input and output both $O(T)$ - no expansion!

---

### **Conversion Type 2: Sparse One-Hot Indices → Dense Matrix → MLE**

**Applies to**: `instruction_ra`, `bytecode_ra`, `ram_ra`

 **KEY TRANSFORMATION: Sparse → Dense**

**Input from Step 10B** (sparse representation):
```rust
instruction_ra[chunk_i] = [Some(10), Some(15), None, Some(10), ...]  // Size T
```
This is **sparse**: Only stores the index, not the full one-hot vector. Memory: $O(T)$ bytes.

**Output in Step 10C** (dense representation):
```rust
matrix = [
  [0,0,0,...,0,1,0,...,0],  // Row 0: 256 entries, only index 10 is 1
  [0,0,0,...,0,0,1,...,0],  // Row 1: 256 entries, only index 15 is 1
  [0,0,0,...,0,0,0,...,0],  // Row 2: 256 entries, all zeros (no lookup)
  [0,0,0,...,0,1,0,...,0],  // Row 3: 256 entries, only index 10 is 1
]
```
This is **dense storage**: Full $T \times N$ matrix. Memory: $O(T \times N)$ bytes.

 **TERMINOLOGY CLARIFICATION - "Sparse" vs "Dense"**:

The matrix above is:

- **Dense storage representation**: We store ALL entries (all 256 values per row), not just indices
- **Mathematically sparse**: Most entries are zero (only one 1 per row)

**Two different meanings of "sparse"**:

1. **Storage sparsity** (data structure):
   - **Sparse storage**: `[Some(10), Some(15), None, ...]` - store only non-zero indices (what we have in 10B)
   - **Dense storage**: `[[0,...,1,...,0], [0,...,0,1,...,0], ...]` - store all values (what we create in 10C)

2. **Mathematical sparsity** (matrix property):
   - **Mathematically sparse**: Matrix has mostly zeros (true for one-hot matrices)
   - **Mathematically dense**: Matrix has mostly non-zeros

**Why this matters**: When we say "Step 10C converts sparse → dense", we mean:

- Converting from **sparse storage** (indices only) to **dense storage** (full matrix)
- The matrix remains **mathematically sparse** (mostly zeros) in both representations

**Memory impact**:

- 10B: 1 KB (storing only T indices)
- 10C: 256 KB (storing full T×256 matrix with mostly zeros)

**The conversion code**:

```rust
// Step 10B: Sparse (what we have)
let sparse_indices: Vec<Option<u8>> = data.instruction_ra[chunk_i];  // Size T

// Step 10C: Convert sparse → dense
let mut dense_matrix = vec![vec![0u8; TABLE_SIZE]; T];  // Size T × N
for (j, &maybe_k) in sparse_indices.iter().enumerate() {
    if let Some(k) = maybe_k {
        dense_matrix[j][k as usize] = 1;  // Expand: one index → full one-hot row
    }
}

// Convert dense matrix → MLE
let mle_ra = MultilinearPolynomial::from_2d_matrix(dense_matrix);
```

**Why convert sparse → dense?**

1. **Sumcheck requires dense MLE**: During sumcheck, we need to evaluate $\widetilde{\text{ra}}_i(r_1, \ldots, r_{\log T}, r'_1, \ldots, r'_{\log N})$ at arbitrary field points (not just Boolean). This requires the full dense polynomial representation.

2. **Can't evaluate sparse representation at arbitrary points**: The sparse index `Some(10)` only tells us about Boolean hypercube points. To evaluate at $\widetilde{\text{ra}}_i(0.73, 0.42, \ldots)$, we need the full Lagrange interpolation from the dense matrix.

3. **Memory trade-off**:
   - Sparse: $T$ entries (e.g., 1024 bytes)
   - Dense: $T \times N$ entries (e.g., 1024 × 256 = 256 KB)
   - But dense enables efficient sumcheck!

**Memory**: Input $O(T)$, output $O(T \times N)$ - **major expansion!**

**Mathematical definition** of the one-hot MLE $\widetilde{\text{ra}}_i(j, k)$:

Given $T \times N$ matrix $M$ where $M[j][k] \in \{0, 1\}$ and $\sum_k M[j][k] \leq 1$ (at most one "1" per row):

$$\widetilde{\text{ra}}_i(X_1, \ldots, X_{\log T}, Y_1, \ldots, Y_{\log N}) = \sum_{j \in \{0,1\}^{\log T}} \sum_{k \in \{0,1\}^{\log N}} M[j][k] \cdot \text{eq}(X; j) \cdot \text{eq}(Y; k)$$

**Properties**:

- Domain: $\mathbb{F}^{\log T} \times \mathbb{F}^{\log N}$ (two-dimensional)
- On hypercube: $\widetilde{\text{ra}}_i(j, k) = 1$ if cycle $j$ accessed index $k$, else $0$
- One-hot property: $\sum_{k=0}^{N-1} \widetilde{\text{ra}}_i(j, k) \leq 1$ for each $j$

**Concrete example** for $T = 4, N = 256$ (chunk table size):

If cycles looked up: cycle 0 → index 10, cycle 1 → index 15, cycle 2 → no lookup, cycle 3 → index 10:

$$\widetilde{\text{ra}}_0(0, 10) = 1, \quad \widetilde{\text{ra}}_0(0, k \neq 10) = 0$$
$$\widetilde{\text{ra}}_0(1, 15) = 1, \quad \widetilde{\text{ra}}_0(1, k \neq 15) = 0$$
$$\widetilde{\text{ra}}_0(2, k) = 0 \quad \forall k \quad \text{(no lookup)}$$
$$\widetilde{\text{ra}}_0(3, 10) = 1, \quad \widetilde{\text{ra}}_0(3, k \neq 10) = 0$$

---

### **Summary: Two Different Conversions in Step 10C**

| Property | Conversion Type 1 | Conversion Type 2 |
|----------|-------------------|-------------------|
| **Applies to** | Simple vectors (left_input, rd_inc, etc.) | One-hot indices (instruction_ra, bytecode_ra, ram_ra) |
| **Step 10B output** | Dense: `[100, 250, 350, ...]` | Sparse: `[Some(10), Some(15), None, ...]` |
| **Conversion needed?** |  No (already dense) |  Yes (sparse → dense) |
| **Step 10C operation** | Type conversion: $\mathbb{Z} \to \mathbb{F}$ | Expand to matrix: $T \to T \times N$ |
| **Memory change** | $O(T) \to O(T)$ (no expansion) | $O(T) \to O(T \times N)$ (major expansion!) |
| **Example size** | 1 KB → 1 KB | 1 KB → 256 KB |
| **Output MLE domain** | $\mathbb{F}^{\log T}$ (univariate in cycle) | $\mathbb{F}^{\log T} \times \mathbb{F}^{\log N}$ (bivariate) |

**Key insight**:

- **Type 1**: Already have full information, just wrap as MLE
- **Type 2**: Have compact index, must expand to full one-hot matrix before MLE

---

**Result of Step 10C**: HashMap of MLEs

The complete set of ~50 witness polynomials created from the execution trace:

#### Type 1 MLEs: Simple Witness Vectors (Conversion Type 1)

| Rust Enum | Math Object | Domain | Size | What it stores |
|-----------|-------------|--------|------|----------------|
| `LeftInstructionInput` | $\widetilde{L}(j)$ | $\mathbb{F}^{\log T}$ | $T$ | Left operand (rs1 or PC) at cycle $j$ |
| `RightInstructionInput` | $\widetilde{R}(j)$ | $\mathbb{F}^{\log T}$ | $T$ | Right operand (rs2 or imm) at cycle $j$ |
| `WriteLookupOutputToRD` | $\widetilde{w}_{\text{rd}}(j)$ | $\mathbb{F}^{\log T}$ | $T$ | Boolean: should write result to rd? |
| `WritePCtoRD` | $\widetilde{w}_{\text{pc}}(j)$ | $\mathbb{F}^{\log T}$ | $T$ | Boolean: should write PC to rd (JAL/JALR)? |
| `ShouldBranch` | $\widetilde{b}(j)$ | $\mathbb{F}^{\log T}$ | $T$ | Boolean: is this a branch instruction? |
| `ShouldJump` | $\widetilde{j}(j)$ | $\mathbb{F}^{\log T}$ | $T$ | Boolean: is this a jump instruction? |
| `RdInc` | $\widetilde{\Delta}_{\text{rd}}(j)$ | $\mathbb{F}^{\log T}$ | $T$ | Register write increment at cycle $j$ |
| `RamInc` | $\widetilde{\Delta}_{\text{ram}}(j)$ | $\mathbb{F}^{\log T}$ | $T$ | RAM write increment at cycle $j$ |

**Conversion**: Already dense in Step 10B → Just type convert $\mathbb{Z} \to \mathbb{F}$ in Step 10C

**Memory**: $O(T)$ per polynomial (no expansion)

**Example for $T = 1024$**: Each polynomial stores 1024 field elements ≈ 32 KB

---

#### Type 2 MLEs: One-Hot Polynomials (Conversion Type 2)

**A) Instruction Lookup Addresses** (16 chunks for 128-bit lookup index):

| Rust Enum | Math Object | Domain | Size | What it stores |
|-----------|-------------|--------|------|----------------|
| `InstructionRa(0)` | $\widetilde{\text{ra}}_0(j, k)$ | $\mathbb{F}^{\log T} \times \mathbb{F}^{\log N}$ | $T \times N$ | Chunk 0 of instruction lookup (bits 0-7) |
| `InstructionRa(1)` | $\widetilde{\text{ra}}_1(j, k)$ | $\mathbb{F}^{\log T} \times \mathbb{F}^{\log N}$ | $T \times N$ | Chunk 1 of instruction lookup (bits 8-15) |
| `InstructionRa(2)` | $\widetilde{\text{ra}}_2(j, k)$ | $\mathbb{F}^{\log T} \times \mathbb{F}^{\log N}$ | $T \times N$ | Chunk 2 of instruction lookup (bits 16-23) |
| ... | ... | ... | ... | ... |
| `InstructionRa(15)` | $\widetilde{\text{ra}}_{15}(j, k)$ | $\mathbb{F}^{\log T} \times \mathbb{F}^{\log N}$ | $T \times N$ | Chunk 15 of instruction lookup (bits 120-127) |

**Table size**: $N = 2^8 = 256$ entries per chunk (8-bit chunks)

**Total**: 16 polynomials for instruction lookups

---

**B) Bytecode Lookup Addresses** ($d$ chunks for program counter):

| Rust Enum | Math Object | Domain | Size | What it stores |
|-----------|-------------|--------|------|----------------|
| `BytecodeRa(0)` | $\widetilde{\text{bc}}_0(j, k)$ | $\mathbb{F}^{\log T} \times \mathbb{F}^{\log K_c}$ | $T \times K_c$ | Chunk 0 of PC address |
| `BytecodeRa(1)` | $\widetilde{\text{bc}}_1(j, k)$ | $\mathbb{F}^{\log T} \times \mathbb{F}^{\log K_c}$ | $T \times K_c$ | Chunk 1 of PC address |
| ... | ... | ... | ... | ... |
| `BytecodeRa(d-1)` | $\widetilde{\text{bc}}_{d-1}(j, k)$ | $\mathbb{F}^{\log T} \times \mathbb{F}^{\log K_c}$ | $T \times K_c$ | Chunk $d-1$ of PC address |

**Chunk table size**: $K_c = K^{1/d}$ where $K$ = bytecode size, $d$ = chunking parameter (dynamically chosen)

**Total**: $d$ polynomials for bytecode lookups (typically 2-4 chunks)

**Example**: If bytecode has 4096 instructions and $d = 2$, then $K_c = \sqrt{4096} = 64$ entries per chunk

---

**C) RAM Address Lookups** ($d$ chunks for memory addresses):

| Rust Enum | Math Object | Domain | Size | What it stores |
|-----------|-------------|--------|------|----------------|
| `RamRa(0)` | $\widetilde{\text{mem}}_0(j, k)$ | $\mathbb{F}^{\log T} \times \mathbb{F}^{\log M_c}$ | $T \times M_c$ | Chunk 0 of RAM address |
| `RamRa(1)` | $\widetilde{\text{mem}}_1(j, k)$ | $\mathbb{F}^{\log T} \times \mathbb{F}^{\log M_c}$ | $T \times M_c$ | Chunk 1 of RAM address |
| ... | ... | ... | ... | ... |
| `RamRa(d-1)` | $\widetilde{\text{mem}}_{d-1}(j, k)$ | $\mathbb{F}^{\log T} \times \mathbb{F}^{\log M_c}$ | $T \times M_c$ | Chunk $d-1$ of RAM address |

**Chunk table size**: $M_c = M^{1/d}$ where $M$ = memory size, $d$ = chunking parameter (dynamically chosen to keep $M_c = 2^8 = 256$)

**Total**: $d$ polynomials for RAM address lookups (typically 8 chunks for 64-bit addresses)

**Example**: If RAM has $M = 2^{64}$ addresses and $d = 8$, then $M_c = 2^8 = 256$ entries per chunk

---

**Conversion**: Sparse indices in Step 10B → Expand to dense $T \times N$ matrix in Step 10C → MLE

**Memory**: $O(T \times N)$ per polynomial (major expansion!)

**Example for $T = 1024, N = 256$**: Each polynomial stores 262,144 field elements ≈ 8 MB

---

#### Complete Summary: All ~50 MLEs from Step 10C

```rust
HashMap<CommittedPolynomial, MultilinearPolynomial<F>> {
    // Type 1: 8 simple witness polynomials (8 × 32 KB = 256 KB total)
    LeftInstructionInput => ˜L(j),                     // t-variate
    RightInstructionInput => ˜R(j),                    // t-variate
    WriteLookupOutputToRD => ˜w_rd(j),                 // t-variate
    WritePCtoRD => ˜w_pc(j),                           // t-variate
    ShouldBranch => ˜b(j),                             // t-variate
    ShouldJump => ˜j(j),                               // t-variate
    RdInc => ˜\Delta_rd(j),                                 // t-variate
    RamInc => ˜\Delta_ram(j),                               // t-variate

    // Type 2: 16 instruction lookup polynomials (16 × 8 MB = 128 MB total)
    InstructionRa(0) => ˜ra_0(j, k),                    // (t+8)-variate
    InstructionRa(1) => ˜ra_1(j, k),                    // (t+8)-variate
    // ... (14 more)
    InstructionRa(15) => ˜ra_1_5(j, k),                  // (t+8)-variate

    // Type 2: ~3 bytecode lookup polynomials (3 × 8 MB = 24 MB total)
    BytecodeRa(0) => ˜bc_0(j, k),                       // (t+log K_c)-variate
    BytecodeRa(1) => ˜bc_1(j, k),                       // (t+log K_c)-variate
    // ... (d-2 more)

    // Type 2: ~8 RAM address polynomials (8 × 8 MB = 64 MB total)
    RamRa(0) => ˜mem_0(j, k),                           // (t+8)-variate
    RamRa(1) => ˜mem_1(j, k),                           // (t+8)-variate
    // ... (6 more)
    RamRa(7) => ˜mem_7(j, k),                           // (t+8)-variate
}

// Total: ~35 polynomials, ~220 MB of MLE data (before commitment)
```

**Memory hierarchy**:

- **Step 10B**: ~2 MB (sparse indices + simple vectors)
- **Step 10C (MLEs)**: ~220 MB (after dense expansion)
- **After Dory commitment**: ~7 KB (35 commitments × 192 bytes each)

**The compression**: 220 MB of witness data → 7 KB of commitments!

**Mathematical summary**:

$$\boxed{\text{Execution Trace } (s_0, \ldots, s_{T-1}) \xrightarrow{\text{Step 10A-C}} \text{~35 MLEs } \{\widetilde{P}_1, \ldots, \widetilde{P}_{35}\}}$$

Where each $\widetilde{P}_i$ is either:

- **Type 1**: Simple MLE $\mathbb{F}^{\log T} \to \mathbb{F}$ (already dense in Step 10B) - 8 polynomials
- **Type 2**: One-hot MLE $\mathbb{F}^{\log T} \times \mathbb{F}^{\log N} \to \mathbb{F}$ (sparse → dense in Step 10C) - 27 polynomials

---

## From MLEs to Commitments: The Final Step

After generating all MLEs, we commit to them using Dory PCS. This is where the polynomials become cryptographic objects.

### Mathematical Object: Dory Commitment

For a multilinear polynomial $f$ with $N = 2^{\nu}$ evaluations, Dory commits via:

1. **Reshape to matrix**: $N$ coefficients → $\sqrt{N} \times \sqrt{N}$ matrix $M$

2. **Layer 1 (Pedersen to rows)**:
   $$V_i = \sum_{j=1}^{\sqrt{N}} M_{i,j} G_{1,j} + r_i H_1 \in \mathbb{G}_1$$

3. **Layer 2 (AFGHO to vector)**:
   $$C_M = \prod_{i=1}^{\sqrt{N}} e(V_i, G_{2,i}) \cdot e(H_1, H_2)^{r_{fin}} \in \mathbb{G}_T$$

**Result**: Single element $C_M \in \mathbb{G}_T$ (192 bytes) commits to entire polynomial!

### Concrete Example: Committing LeftInstructionInput

**Witness data**:
```
left_instruction_input = [100, 250, 350, 0, ...] (T = 1024 cycles)
MLE: L̃(j) : F^10 → F where L̃(0,0,...,0,0) = 100, L̃(1,0,...,0,0) = 250, etc.
```

**Dory commitment** (T = 1024, K = 16, total size = 16,384):
```
Matrix dimensions: √16,384 × √16,384 = 128 × 128

Step 1: Reshape L̃ evaluations into 128×128 matrix M_L

Step 2 (Layer 1): Create 128 Pedersen commitments (one per row):
  V_0 = M_L[0,0]·G_{1,0} + M_L[0,1]·G_{1,1} + ... + M_L[0,127]·G_{1,127} + r_0·H_1 \in G_1
  V_1 = M_L[1,0]·G_{1,0} + M_L[1,1]·G_{1,1} + ... + M_L[1,127]·G_{1,127} + r_1·H_1 \in G_1
  ...
  V_127 = M_L[127,0]·G_{1,0} + ... + M_L[127,127]·G_{1,127} + r_127·H_1 \in G_1

  Result: 128 commitments (V_0, ..., V_127), each in G_1

Step 3 (Layer 2): AFGHO compresses 128 G_1 elements into single G_T element:
  C_L = e(V_0, G_{2,0}) · e(V_1, G_{2,1}) · ... · e(V_127, G_{2,127}) · e(H_1, H_2)^r_fin \in G_T

  Result: ONE commitment C_L \in G_T ≈ 192 bytes (compressed)
```

**Key insight**:

- **Layer 1** creates 128 intermediate commitments (one per matrix row)
- **Layer 2** (AFGHO) compresses those 128 commitments into 1 final commitment
- Each MLE polynomial gets exactly **ONE final commitment** $C \in \mathbb{G}_T$

---

**For ALL ~35 polynomials**:
```
Per polynomial: 1 G_T element = 192 bytes
Total commitments sent to verifier: 35 × 192 bytes ≈ 7 KB

Compare to:

- Raw MLEs: ~220 MB (dense matrix data)
- Raw trace: 1024 cycles × ~100 bytes/cycle ≈ 100 KB
- Commitments: 7 KB

Compression ratio: ~30,000× compared to MLEs! ~14× compared to raw trace!
```

### What Gets Sent to Verifier?

**Stage 0 (Commitment phase)**:

Prover → Verifier:
$$\{C_L, C_R, C_{\Delta_{rd}}, C_{\Delta_{ram}}, C_{ra_0}, \ldots, C_{ra_{15}}, C_{bc_0}, \ldots\} \subset \mathbb{G}_T^{50}$$

**Size**: ~10 KB of commitments (vs 100 KB raw trace)

**Binding property**: Computationally infeasible to find different polynomial $f' \neq f$ with same commitment $C$

**Hiding property**: Commitment reveals nothing about polynomial values (due to random blinding factors $r_i, r_{fin}$)

### Opening Proofs (Stage 5)

After sumcheck reduces verification to single-point evaluations, verifier needs to check:

$$\text{For random } r \in \mathbb{F}^{\log T}: \quad \widetilde{L}(r) \stackrel{?}{=} y_L$$

**Dory opening proof**:

1. Prover runs $\log(\sqrt{N})$ rounds of recursive halving (Dory-Reduce)
2. Each round: Prover sends 6 group elements
3. Final round: Scalar-Product protocol (constant size)

**Proof size**: $6 \cdot \log(\sqrt{N})$ group elements ≈ 18 KB per polynomial

**Verification time**: $O(\log N)$ (logarithmic in polynomial size!)

**Batching optimization**: When opening multiple polynomials at the same point, amortized cost drops to $O(1)$ per polynomial!

### Summary: Mathematical Objects at Each Stage

| Stage | Object Type | Size | Security |
|-------|-------------|------|----------|
| **Trace** | $\text{Vec<Cycle>}$ | 10KB-10MB | N/A (plaintext) |
| **MLE Witness** | $\{\widetilde{L}, \widetilde{R}, \ldots\} : \mathbb{F}^{\log T} \to \mathbb{F}$ | Same as trace | N/A (plaintext) |
| **Commitments** | $\{C_L, C_R, \ldots\} \subset \mathbb{G}_T$ | ~10 KB | Binding + Hiding |
| **Opening Proofs** | Dory proofs | ~18 KB each | Soundness $2^{-128}$ |

**The transformation**:
```
Execution (100 KB trace)
   ↓ [Extract witness data]
MLEs (100 KB polynomials)
   ↓ [Dory commit]
Commitments (10 KB in G_T)
   ↓ [Sumcheck reduces to point evaluations]
Opening proofs (18 KB per opening)
   ↓ [Verifier checks]
Accept/Reject (single bit!)
```

**Key property**: Verifier never sees the full trace, only:

1. Commitments ($\mathbb{G}_T$ elements, ~10 KB)
2. Claimed evaluations (field elements, ~1 KB)
3. Opening proofs (logarithmic size, ~18 KB each)

**Total proof size**: ~200-500 KB (vs 100+ KB trace, and verifies in $O(\log T)$ time!)

---

## Complete Execution and Witness Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│ User calls: prove_{fn_name}(input)                              │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────────┐
│ Macro-generated prove function                                  │
│  ├─> Serialize inputs to bytes                                  │
│  └─> Call JoltRV64IMAC::prove()                                 │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────────┐
│ JoltRV64IMAC::prove()                                           │
│  ├─> Setup memory configuration                                 │
│  └─> Call guest::program::trace()  ◄─── THE BIG EMULATION      │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────────┐
│ guest::program::trace()                                         │
│  ├─> setup_emulator_with_backtraces()                           │
│  │   - Load ELF into memory                                     │
│  │   - Initialize CPU (registers, PC, memory)                   │
│  │   - Create LazyTraceIterator                                 │
│  │                                                               │
│  └─> lazy_trace_iter.collect()                                  │
│      ┌──────────────────────────────────────────────────┐       │
│      │ FOR EACH INSTRUCTION UNTIL HALT:                 │       │
│      │  1. Fetch: Read instruction from PC              │       │
│      │  2. Decode: Determine instruction type           │       │
│      │  3. Execute & Trace:                             │       │
│      │     a. Capture pre-execution state (registers)   │       │
│      │     b. Execute instruction (modify CPU)          │       │
│      │     c. Capture post-execution state              │       │
│      │     d. Create Cycle struct                       │       │
│      │     e. Push to trace Vec                         │       │
│      │  4. Update PC                                     │       │
│      │  5. Check for halt (ECALL or error)              │       │
│      └──────────────────────────────────────────────────┘       │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────────┐
│ Output:                                                          │
│  - trace: Vec<Cycle> (e.g., 10,000 cycles)                      │
│  - final_memory_state: Memory                                   │
│  - program_io: JoltDevice (inputs/outputs/panic)                │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────────┐
│ Back in JoltRV64IMAC::prove()                                   │
│  ├─> Pad trace to power of 2                                    │
│  └─> Create StateManager with trace                             │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     v
┌─────────────────────────────────────────────────────────────────┐
│ JoltDAG::prove() (Part 3 starts here!)                          │
│  ├─> generate_and_commit_polynomials()                          │
│  │   ┌──────────────────────────────────────────────────┐       │
│  │   │ WITNESS GENERATION:                              │       │
│  │   │  1. CommittedPolynomial::generate_witness_batch()│       │
│  │   │     - Create WitnessData (vectors)               │       │
│  │   │     - Process each cycle in parallel:            │       │
│  │   │       ├─> Extract instruction operands           │       │
│  │   │       ├─> Extract circuit flags                  │       │
│  │   │       ├─> Extract register increments            │       │
│  │   │       ├─> Decompose lookup queries               │       │
│  │   │       └─> Record memory accesses                 │       │
│  │   │     - Convert vectors → MLEs                     │       │
│  │   │                                                   │       │
│  │   │  2. PCS::batch_commit()                          │       │
│  │   │     - Commit to all MLEs using Dory              │       │
│  │   │     - Generate opening hints                     │       │
│  │   └──────────────────────────────────────────────────┘       │
│  │                                                               │
│  └─> Run 5 stages of sumchecks (Part 3!)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Key Outputs Summary

### From Emulation (Part 2a):

**Vec<Cycle>**: Complete execution trace
```rust
[
    Cycle::LOAD(RISCVCycle { /* load 100 into r1 */ }),
    Cycle::LOAD(RISCVCycle { /* load 250 into r2 */ }),
    Cycle::ADD(RISCVCycle {
        instruction: ADD { rd: 3, rs1: 1, rs2: 2 },
        register_state: { rs1_val_pre: 100, rs2_val_pre: 250, rd_val_post: 350 },
        ram_access: (),
    }),
    // ... more cycles ...
]
```

### From Witness Generation (Part 2b):

**HashMap<CommittedPolynomial, MLE>**: Polynomials ready for proving
```rust
{
    LeftInstructionInput: MLE([100, 250, ...]),
    RightInstructionInput: MLE([250, ...]),
    WriteLookupOutputToRD: MLE([1, 1, ...]),
    RdInc: MLE([100, 250, 350, ...]),
    InstructionRa(0): OneHotMLE([[0,0,1,0,...], ...]),
    InstructionRa(1): OneHotMLE([[0,1,0,0,...], ...]),
    // ... 16 instruction chunks + d bytecode chunks + d RAM chunks ...
}
```

These MLEs are:

1. **Committed** using Dory PCS (Stage 0 of proving)
2. **Proven** via sumchecks in Stages 1-4
3. **Opened** via batched opening proof in Stage 5

---

## Connection to Theory: Lookup-Centric Architecture


> Jolt combines all possible operations into a single, giant lookup table, `JOLT_V`:
>
> $$ \text{JOLT\_V}(\text{opcode}, a, b) = (c, \text{flags}) = f_{op}(a, b) $$
>
> A proof of execution for a program trace of length $m$ becomes a sequence of $m$ lookup claims:
>
> - At step 1: $(c_1, \text{flags}_1) = \text{JOLT\_V}(\text{opcode}_1, a_1, b_1)$
>
> - At step 2: $(c_2, \text{flags}_2) = \text{JOLT\_V}(\text{opcode}_2, a_2, b_2)$
>
> - ...

**How witness generation enables this**:

For each cycle, we extract:

- **Opcode**: Implicit in `Cycle` enum variant (e.g., `Cycle::ADD`)
- **Inputs** `(a, b)`: Captured in `left_instruction_input` and `right_instruction_input` MLEs
- **Output** `c`: Captured in `rd_val_post` (verified via R1CS wiring)
- **Lookup indices**: Decomposed and stored in `InstructionRa(0)..InstructionRa(15)` MLEs

The witness generation **transforms** execution trace into **lookup queries**!

---

## Performance Considerations

### Why Parallel Processing?

```rust
trace.par_iter().enumerate().for_each(|(cycle_idx, cycle)| {
    process_cycle(cycle_idx, cycle, &shared_data, preprocessing);
});
```

**Rayon** parallelizes witness generation:

- Each cycle processed independently (embarrassingly parallel!)
- Typical trace: 10,000 - 1,000,000 cycles
- Multi-core speedup: 4-8x faster on modern CPUs

### Memory Efficiency

**LazyTraceIterator**: Generates cycles on-demand

- Avoids storing entire program state history
- Only current cycle + final memory kept in RAM
- Critical for large programs (> 100K cycles)

**Option**: `trace_to_file()` for massive traces

- Streams cycles to disk instead of Vec
- Enables proving programs with millions of cycles
- Trade-off: I/O overhead vs memory savings

---

## Debugging and Introspection

### Cycle Tracking

**File**: [jolt-sdk/src/lib.rs] (guest side)

```rust
use jolt::{start_cycle_tracking, end_cycle_tracking};

#[jolt::provable]
fn my_function(n: u32) -> u64 {
    start_cycle_tracking("fibonacci_loop");
    let result = fibonacci(n);
    end_cycle_tracking("fibonacci_loop");
    result
}
```

**Output** (during emulation):
```
Cycle tracking report:
  fibonacci_loop: 1,234 real cycles, 1,234 virtual cycles
```

Helps identify bottlenecks in guest programs!

### Panic Backtraces

**File**: [tracer/src/emulator/cpu.rs:114-116](tracer/src/emulator/cpu.rs#L114)

```rust
pub struct Cpu {
    // ...
    call_stack: VecDeque<CallFrame>,  // Circular buffer of depth 32
}
```

When guest panics:

- Call stack unwound from CPU state
- Symbols resolved using ELF file
- Rust-style backtrace printed

Example:
```
Guest program panicked!
Backtrace:
  0: guest::my_function at guest/src/lib.rs:42
  1: guest::helper at guest/src/lib.rs:18
  2: _start at ...
```

---

## Summary: Part 2 Checklist

**Emulation** 

- [x] Load ELF into memory
- [x] Initialize RISC-V CPU (32 registers, PC, CSRs)
- [x] Fetch-decode-execute loop
- [x] Capture pre/post state for each instruction
- [x] Handle memory loads/stores
- [x] Track program I/O (inputs, outputs, panic)
- [x] Generate `Vec<Cycle>` (execution trace)

**Witness Generation** 

- [x] Process trace in parallel
- [x] Extract instruction operands (left/right inputs)
- [x] Extract circuit flags (branch, jump, write)
- [x] Compute register increments (Twist)
- [x] Decompose lookup queries into chunks (Shout)
- [x] Record memory accesses (Twist)
- [x] Convert vectors → MLEs
- [x] Convert one-hot indices → MLEs

**Key Outputs**:

- `Vec<Cycle>`: Complete execution witness (~10KB - 10MB depending on program)
- `HashMap<CommittedPolynomial, MLE>`: 40-60 polynomials ready for proving
- `Memory`: Final RAM state (for Twist verification)
- `JoltDevice`: I/O data (for public inputs)

**Next Step**: Part 3 (Proof Generation) uses these MLEs to construct the proof!

---

## Additional Resources

- [tracer/src/emulator/cpu.rs](tracer/src/emulator/cpu.rs): Full CPU emulator
- [tracer/src/instruction/mod.rs](tracer/src/instruction/mod.rs): Instruction traits and enum
- [tracer/src/instruction/add.rs](tracer/src/instruction/add.rs): Example instruction
- [jolt-core/src/zkvm/witness.rs](jolt-core/src/zkvm/witness.rs): Witness generation


/newpage

# Part 3: Proof Generation Deep Dive

---

## Table of Contents

1. [Overview: The Five-Stage DAG](#overview-the-five-stage-dag)
2. [Mathematical Foundation](#mathematical-foundation)
3. [Setup: StateManager and Opening Accumulator](#setup-statemanager-and-opening-accumulator)
4. [Polynomial Generation and Commitment](#polynomial-generation-and-commitment)
5. [Stage 1: Spartan Outer Sumcheck](#stage-1-spartan-outer-sumcheck)
6. [Stage 2: Batched Sumchecks](#stage-2-batched-sumchecks)
7. [Stage 3: More Batched Sumchecks](#stage-3-more-batched-sumchecks)
8. [Stage 4: Final Sumchecks](#stage-4-final-sumchecks)
9. [Stage 5: Batched Opening Proof](#stage-5-batched-opening-proof)
10. [Summary: Mathematical Objects at Each Stage](#summary-mathematical-objects-at-each-stage)

---

## Overview: The Five-Stage DAG

**Central file**: [jolt-core/src/zkvm/dag/jolt_dag.rs](../jolt-core/src/zkvm/dag/jolt_dag.rs)

**DAG** = Directed Acyclic Graph. The proof generation is structured as a graph where:

- **Nodes** = Sumcheck instances
- **Edges** = Polynomial evaluations flowing between sumchecks
- **Stages** = Levels in the graph (all sumchecks in same stage can run in parallel)

**The Five Stages**:

```
Stage 1: Initial sumchecks (Spartan outer)
   ↓ (output claims become input claims for Stage 2)
Stage 2: Batched sumchecks (Spartan product, Registers, RAM, Lookups)
   ↓
Stage 3: More batched sumchecks (Spartan inner, Hamming weight, Read-checking)
   ↓
Stage 4: Final sumchecks (Ra virtualization, Bytecode)
   ↓
Stage 5: Batched opening proof (Dory PCS)
```

**Key insight**:

> "Jolt takes a pragmatic 'best tool for the job' approach:
> - **Lasso (Lookups) for 'What an instruction does'**: Proves instruction semantics
> - **Spartan (R1CS) for 'How the VM is wired'**: Proves simple algebraic relationships"

This hybrid approach manifests in the five stages:

- Stages 1-2: Primarily Spartan (R1CS wiring)
- Stages 3-4: Primarily Lasso/Shout/Twist (lookups and memory checking)
- Stage 5: Dory (polynomial commitment opening)

---

## Mathematical Foundation

### What is a Sumcheck?


**Claim**: Prover wants to convince verifier that:

$$H = \sum_{x \in \{0,1\}^n} g(x)$$

Where $g(x)$ is an $n$-variate polynomial over field $\mathbb{F}$.

**Protocol** (n rounds):

1. **Round 1**: Prover sends univariate polynomial $g_1(X_1) = \sum_{x_2, \ldots, x_n \in \{0,1\}^{n-1}} g(X_1, x_2, \ldots, x_n)$
2. Verifier checks $g_1(0) + g_1(1) \stackrel{?}{=} H$, samples random $r_1 \in \mathbb{F}$
3. **Round 2**: Prover sends $g_2(X_2) = \sum_{x_3, \ldots, x_n \in \{0,1\}^{n-2}} g(r_1, X_2, x_3, \ldots, x_n)$
4. Verifier checks $g_2(0) + g_2(1) \stackrel{?}{=} g_1(r_1)$, samples $r_2$
5. Continue for $n$ rounds...
6. **Final round**: Reduced to claim about $g(r_1, \ldots, r_n)$ at single point

**Why this works**:

- **Schwartz-Zippel Lemma**: Two different degree-$d$ polynomials agree at most $d/|\mathbb{F}|$ fraction of points
- Verifier's random challenges make cheating probability negligible

**Key transformation**:
$$\text{Claim about } 2^n \text{ points} \rightarrow \text{Claim about } 1 \text{ random point}$$

### Virtual vs Committed Polynomials

The proof system uses two fundamentally different types of polynomials:

1. **Virtual polynomials** - NOT committed, proven by subsequent sumchecks
2. **Committed polynomials** - Committed via Dory, proven by opening in Stage 5

**The DAG structure emerges** from this distinction:

- Virtual evaluations create edges between sumchecks (dependencies)
- Committed evaluations accumulate for final batched opening

** For detailed explanation with examples, see**: [The Two Types of Polynomials in Jolt](#the-two-types-of-polynomials-in-jolt) (Section "The Opening Accumulator" below)

- Partial ordering: Can't prove sumcheck until all dependencies resolved

### Batched Sumcheck

Multiple sumchecks with same number of variables can be *batched*:

**Input**: $k$ sumcheck instances with claims:
$$H_1 = \sum_{x \in \{0,1\}^n} g_1(x), \quad H_2 = \sum_{x \in \{0,1\}^n} g_2(x), \quad \ldots, \quad H_k = \sum_{x \in \{0,1\}^n} g_k(x)$$

**Batching**:

1. Sample random coefficients $\alpha_1, \ldots, \alpha_k$ from transcript
2. Create combined claim:
   $$H_{\text{combined}} = \alpha_1 H_1 + \alpha_2 H_2 + \cdots + \alpha_k H_k$$
3. Define combined polynomial:
   $$g_{\text{combined}}(x) = \alpha_1 g_1(x) + \alpha_2 g_2(x) + \cdots + \alpha_k g_k(x)$$
4. Run single sumcheck on $g_{\text{combined}}$

**Efficiency gain**:

- **Without batching**: $k \cdot n$ rounds (each instance runs separately)
- **With batching**: $n$ rounds (combined instance)
- **Verifier receives**: 1 challenge per round (not $k$ challenges)

**Security**: Schwartz-Zippel ensures random linear combination detects any cheating sumcheck

**Location**: [jolt-core/src/subprotocols/sumcheck.rs:178](../jolt-core/src/subprotocols/sumcheck.rs#L178) - `BatchedSumcheck::prove()`

---

## Setup: StateManager and Opening Accumulator

### Creating StateManager

**File**: [jolt-core/src/zkvm/dag/state_manager.rs:88](../jolt-core/src/zkvm/dag/state_manager.rs#L88)

```rust
let state_manager = StateManager::new_prover(
    preprocessing,
    trace,
    program_io,
    trusted_advice_commitment,
    final_memory_state,
);
```

### StateManager as Mathematical Object

**Definition**: StateManager is a *proof orchestration context* containing:

```rust
pub struct StateManager<'a, F, ProofTranscript, PCS> {
    pub transcript: Rc<RefCell<ProofTranscript>>,
    pub proofs: Rc<RefCell<Proofs<F, PCS, ProofTranscript>>>,
    pub commitments: Rc<RefCell<Vec<PCS::Commitment>>>,
    pub prover_state: Option<ProverState<'a, F, PCS>>,
    // ... other fields
}
```

**Mathematical interpretation**:

| Rust Field | Mathematical Object | Purpose |
|------------|---------------------|---------|
| `transcript` | Fiat-Shamir oracle $\mathcal{O}$ | Generates random challenges via hashing |
| `proofs` | Map: $\text{ProofKey} \rightarrow \text{SumcheckProof}$ | Stores all Stage 1-4 sumcheck proofs |
| `commitments` | Vector: $(C_1, \ldots, C_m) \in \mathbb{G}_T^m$ | All polynomial commitments (Dory) |
| `prover_state.accumulator` | $\text{OpeningAccumulator}$ | Tracks all evaluation claims |
| `prover_state.trace` | $\vec{s} = (s_0, \ldots, s_{T-1})$ | Execution trace (witness) |

### The Opening Accumulator

**File**: [jolt-core/src/poly/opening_proof.rs](../jolt-core/src/poly/opening_proof.rs)

**What it is**: A data structure that tracks all polynomial evaluation claims generated during **Stages 1-4 of proof generation** (the five stages described in Section "Overview: The Five-Stage DAG" at the top of this document).

Think of it as a "claim ledger" that accumulates IOUs: "I claim polynomial P evaluates to value v at point r." These claims get proven either by subsequent sumchecks (virtual polynomials) or by the final batched opening (committed polynomials).

---

#### Connection to Previous Documents

**Recall from Part 2 (Execution and Witness)**:

We created **35 witness polynomials** from the execution trace:

**Type 1 MLEs** (8 polynomials - already committed):

- $\widetilde{L}(j)$ - Left instruction input
- $\widetilde{R}(j)$ - Right instruction input
- $\widetilde{\Delta}_{\text{rd}}(j)$ - Register increment
- $\widetilde{\Delta}_{\text{ram}}(j)$ - RAM increment
- Circuit flags: $\widetilde{b}(j)$, $\widetilde{j}(j)$, $\widetilde{w}_{\text{rd}}(j)$, $\widetilde{w}_{\text{pc}}(j)$

**Type 2 MLEs** (27 polynomials - already committed):

- $\widetilde{\text{ra}}_0(j,k), \ldots, \widetilde{\text{ra}}_{15}(j,k)$ - Instruction lookup addresses (**16 chunks**)
- $\widetilde{\text{bc}}_0(j,k), \widetilde{\text{bc}}_1(j,k), \widetilde{\text{bc}}_2(j,k)$ - Bytecode lookup addresses (**3 chunks**, varies)
- $\widetilde{\text{mem}}_0(j,k), \ldots, \widetilde{\text{mem}}_7(j,k)$ - RAM addresses (**8 chunks**, varies)

**Exact count**: 8 + 16 + 3 + 8 = **35 polynomials**

**Each polynomial was committed via Dory**:

$$C_L = e(V_0, G_{2,0}) \cdot e(V_1, G_{2,1}) \cdot \ldots \cdot e(V_{127}, G_{2,127}) \cdot e(H_1, H_2)^{r_{fin}} \in \mathbb{G}_T$$

**Result**: **35 commitments** sent to verifier, each 192 bytes → **~7 KB total**

---

#### The Two Types of Polynomials in Jolt

The Opening Accumulator tracks claims about **two fundamentally different types** of polynomials:

**Type A: Committed Polynomials** (already have Dory commitments $C \in \mathbb{G}_T$)

- These are the 35 witness polynomials from Part 2
- **Already committed** during witness generation (before Stage 1)
- Verifier already received their commitments
- Examples: $\widetilde{L}, \widetilde{R}, \widetilde{\Delta}_{\text{rd}}, \widetilde{\text{ra}}_0, \ldots$

**Type B: Virtual Polynomials** (computed on-the-fly, never committed)

- Created temporarily during proof generation
- Used to link sumchecks together
- **Never committed** - too expensive or unnecessary
- Examples: $\widetilde{Az}, \widetilde{Bz}, \widetilde{Cz}$ (from Spartan R1CS)

**The key difference**:

- **Committed**: "I claim $\widetilde{L}(r) = 42$, and here's my commitment $C_L$ to prove I'm not lying"
- **Virtual**: "I claim $\widetilde{Az}(r) = 100$, and I'll prove this via another sumcheck in the next stage"

---

#### Why Do We Need Virtual Polynomials If We Already Have Efficient Commitments?

**Virtual polynomials represent intermediate computations that would be wasteful or impossible to commit to.**

Let's understand this with concrete examples:

##### Example 1: The $Az$ Virtual Polynomial (Spartan R1CS)

**What is $Az$?** Recall from Spartan, the R1CS constraint system proves:

$$Az \circ Bz = Cz$$

Where:

- $A, B, C \in \mathbb{F}^{m \times n}$ are constraint matrices (size: $m$ constraints × $n$ variables)
- $z \in \mathbb{F}^n$ is the witness vector
- $Az = (A_1 \cdot z, A_2 \cdot z, \ldots, A_m \cdot z) \in \mathbb{F}^m$ is the **matrix-vector product**

**Mathematical definition**:
$$Az[i] = \sum_{j=1}^{n} A[i,j] \cdot z[j] \quad \text{for each constraint } i \in [m]$$

**In Jolt's case**:

- $m \approx 30$ constraints per cycle
- $T = 1024$ cycles → $m = 30 \times 1024 = 30,720$ total constraints
- $n \approx 35$ witness polynomials (from Part 2)
- $Az$ is a vector of **30,720 field elements**

**Why ~30 constraints per cycle?** Each RISC-V instruction execution requires checking:

**File**: [jolt-core/src/zkvm/r1cs/constraints.rs](../jolt-core/src/zkvm/r1cs/constraints.rs)

1. **PC Update Constraints** (~5 constraints):
   - PC increments correctly: $\text{PC}_{\text{next}} = \text{PC}_{\text{curr}} + 4$ (normal)
   - PC jumps correctly: $\text{PC}_{\text{next}} = \text{jump\_target}$ (if jump flag set)
   - PC branches correctly: $\text{PC}_{\text{next}} = \text{PC}_{\text{curr}} + \text{offset}$ (if branch flag set)
   - Only one of {normal, jump, branch} is active: $\text{normal} + \text{jump} + \text{branch} = 1$
   - Jump/branch targets computed correctly from immediates

2. **Register Write Constraints** (~5 constraints):
   - If `write_to_rd` flag set, register is updated
   - If `write_pc_to_rd` flag set (JAL/JALR), PC value written to register
   - Register 0 always remains zero: $\text{rd}[0] = 0$
   - Only one write type active per cycle
   - Write value matches instruction output or PC

3. **Memory Access Constraints** (~5 constraints):
   - Load/store address computed correctly from rs1 + offset
   - Load value matches RAM read
   - Store value matches rs2 (or immediate for some instructions)
   - Memory access aligned correctly (for LW/SW, address must be multiple of 4)
   - Load/store flags mutually exclusive

4. **Instruction Decode Constraints** (~5 constraints):
   - Opcode determines which flags are set
   - Left input = rs1 value or PC (based on instruction type)
   - Right input = rs2 value or immediate (based on instruction type)
   - Immediate value extracted correctly from instruction encoding
   - Instruction format (R-type, I-type, S-type, etc.) determines layout

5. **Lookup Verification Constraints** (~5 constraints):
   - Instruction lookup output (from Shout) matches claimed result
   - Lookup output used correctly (written to rd or discarded)
   - Flags from lookup (overflow, zero, etc.) propagate correctly
   - Range checks on lookup indices (within table bounds)
   - Multiple lookups per instruction coordinated correctly

6. **Component Linking Constraints** (~5 constraints):
   - RAM reads/writes consistent with Twist instance
   - Register reads/writes consistent with register Twist instance
   - Bytecode fetch matches current PC
   - Circuit flags (branch, jump, load, store) consistent across components
   - Virtual register usage (for inline optimizations) tracked correctly

**Total**: 6 categories × ~5 constraints each ≈ **30 constraints per cycle**

**Why so many?** Each constraint is simple (degree-2 polynomial equation), but we need many to fully specify VM behavior. This is **much more efficient** than:

- Proving each instruction via arithmetic circuits (~1000s of constraints)
- Bit-level verification of instruction execution (~10,000s of gates)

**Jolt's advantage**: Most instruction logic proven via **lookups** (Shout), not R1CS constraints. The ~30 R1CS constraints just handle:

- Control flow (PC updates)
- Memory/register consistency
- Linking lookups to the rest of the system

This is why Jolt is faster than traditional zkVMs - minimal R1CS overhead!

**Why not commit to $Az$?**

1. **$Az$ is not part of the witness!**
   - The witness is $z$ (the 35 polynomials from Part 2: $\widetilde{L}, \widetilde{R}, \widetilde{\Delta}_{\text{rd}}, \ldots$)
   - $Az$ is a **derived value** computed from $A$ (public) and $z$ (witness)
   - We already committed to $z$ in Part 2, so committing to $Az$ would be redundant

2. **Temporary computation**
   - $Az$ is only needed during Stage 1 → Stage 2 transition
   - Stage 1 sumcheck outputs: "I claim $\widetilde{Az}(\vec{r}) = 42$"
   - Stage 2 sumcheck takes that claim and proves it by reducing to claims about $\widetilde{z}$
   - After Stage 2, we never need $Az$ again!

3. **Verifier never needs the full vector**
   - Verifier only needs $\widetilde{Az}(\vec{r})$ at **one random point** $\vec{r}$
   - Computing $\widetilde{Az}(\vec{r})$ directly is much cheaper than:
     - (a) Committing to entire $Az$ vector
     - (b) Opening commitment at $\vec{r}$

**How it works instead**:

**Stage 1: Spartan outer sumcheck**

Input: Claim that R1CS is satisfied

Output: Claims about $Az(\vec{r}), Bz(\vec{r}), Cz(\vec{r})$ at random point $\vec{r}$

Accumulator stores:
$$\text{virtual\_openings}[Az] = (\vec{r}, 42) \quad \leftarrow \text{Virtual claim (not committed)}$$
$$\text{virtual\_openings}[Bz] = (\vec{r}, 100) \quad \leftarrow \text{Virtual claim}$$
$$\text{virtual\_openings}[Cz] = (\vec{r}, 4200) \quad \leftarrow \text{Virtual claim}$$

**Stage 2: Spartan product sumcheck**

Input: Virtual claim "$Az(\vec{r}) = 42$"

Proves this by showing:
$$Az(\vec{r}) = \sum_{x \in \{0,1\}^n} \widetilde{A}(\vec{r}, x) \cdot \widetilde{z}(x) = 42$$

Runs sumcheck, outputs claims about:

- $\widetilde{A}(\vec{r}, \vec{r}') = 3$ — Virtual (matrix A is public, compute directly)
- $\widetilde{z}(\vec{r}') = 14$ — Committed! (z is witness from Part 2)

Accumulator stores:
$$\text{committed\_openings}[z] = (\vec{r}', 14) \quad \leftarrow \text{Needs Dory opening proof}$$

**Key insight**: By using virtual polynomials, we:

- **Avoid 3 unnecessary commitments** ($Az, Bz, Cz$): Save $3 \times 192 = 576$ bytes
- **Avoid 3 unnecessary openings**: Save $3 \times 6 = 18$ KB
- **Chain sumchecks efficiently**: Stage 1 output becomes Stage 2 input seamlessly

---

##### Example 2: The $\widetilde{A}(\tau, \cdot)$ Virtual Polynomial

**What is $\widetilde{A}(\tau, \cdot)$?** The bivariate MLE of constraint matrix $A$:

$$\widetilde{A}(x, y) : \mathbb{F}^{\log m} \times \mathbb{F}^{\log n} \to \mathbb{F}$$

Where $\widetilde{A}(i, j) = A[i,j]$ for $(i,j) \in \{0,1\}^{\log m} \times \{0,1\}^{\log n}$

**During Stage 2**, after receiving challenge $\tau$ from Stage 1, we need to evaluate:

$$\widetilde{A}(\tau, y) : \mathbb{F}^{\log n} \to \mathbb{F}$$

This is a **univariate** polynomial (first coordinate fixed to $\tau$).

**Why not commit to $\widetilde{A}(\tau, \cdot)$?**

1. **$\tau$ is random** - chosen by verifier after Stage 1
   - We can't commit during preprocessing (don't know $\tau$ yet)
   - We could commit during Stage 2, but...

2. **Matrix $A$ is public!**
   - Both prover and verifier know $A$ (it's part of the constraint system)
   - Verifier can compute $\widetilde{A}(\tau, \vec{r}')$ themselves
   - No need to commit + open when verifier can just compute directly

3. **Size concerns**
   - $A$ is $m \times n = 30,720 \times 50$ matrix
   - Bivariate MLE has $2^{\log m + \log n} = 30,720 \times 64$ evaluations (padded)
   - That's ~2 million field elements = 64 MB!
   - But we only need evaluation at **one point** $(\tau, \vec{r}')$

**How it works instead**:

**Stage 2:**

Input: Virtual claim "$Az(\vec{r}) = 42$" (where $\vec{r}$ is called $\tau$ in this stage)

Sumcheck proves:
$$Az(\tau) = \sum_{x \in \{0,1\}^n} \widetilde{A}(\tau, x) \cdot \widetilde{z}(x) = 42$$

After sumcheck with challenges $\vec{r}'$:

Output claims:
$$\text{virtual\_openings}[A\_tau] = (\vec{r}', 3) \quad \leftarrow \text{Virtual}$$
$$\text{committed\_openings}[z] = (\vec{r}', 14) \quad \leftarrow \text{Committed (from Part 2)}$$

**Stage 3 (or verifier directly):**

To verify virtual claim $\widetilde{A}(\tau, \vec{r}') = 3$:

- Both prover and verifier know matrix $A$
- Both can compute $\widetilde{A}(\tau, \vec{r}')$ directly using Lagrange interpolation
- No commitment/opening needed!

Computation:
$$\widetilde{A}(\tau, \vec{r}') = \sum_{(i,j) \in \{0,1\}^{\log m} \times \{0,1\}^{\log n}} A[i,j] \cdot \text{eq}(\tau; i) \cdot \text{eq}(\vec{r}'; j)$$

This is feasible because:

- $\text{eq}(\tau; i)$ can be computed for all $i$ in $O(m)$ time
- $\text{eq}(\vec{r}'; j)$ can be computed for all $j$ in $O(n)$ time
- Total: $O(m \cdot n) = O(30,720 \times 50) \approx 1.5M$ operations (fast!)

**Key insight**: Virtual polynomials for **public data** (like matrices $A, B, C$) avoid:

- **64 MB commitment** per matrix × 3 matrices = **192 MB of commitments!**
- **~18 KB opening proof** per matrix × 3 matrices = **54 KB of proof overhead**

Instead, verifier just computes the evaluation directly in ~1ms.

---

##### Example 3: When Commitments ARE Necessary (The Witness $z$)

**Why DO we commit to $z$ (the witness polynomials from Part 2)?**

1. **Witness is secret/large**
   - Contains execution trace data (registers, memory, instructions)
   - Prover knows it, verifier doesn't
   - 35 polynomials × average size = ~200 MB of data

2. **Used in multiple stages**
   - Referenced throughout Stages 1-4
   - Multiple sumchecks need evaluations at different random points
   - Can't compute on-the-fly (verifier doesn't have the witness!)

3. **Binding property needed**
   - Prover must commit upfront to prevent changing witness mid-proof
   - Commitment creates cryptographic binding

**Comparison table**:

| Polynomial Type | Example | Size | Public? | Multi-use? | Commit? | Why? |
|----------------|---------|------|---------|------------|---------|------|
| **Witness** | $\widetilde{L}(j)$ | 1024 elements |  No |  Yes |  Yes | Secret data, need binding |
| **Witness** | $\widetilde{\text{ra}}_0(j,k)$ | 262K elements |  No |  Yes |  Yes | Secret data, need binding |
| **Derived (witness)** | $\widetilde{Az}$ | 30K elements |  No |  No |  No | Temporary, reduce to $z$ |
| **Public data** | $\widetilde{A}(\tau, y)$ | 2M elements |  Yes |  No |  No | Verifier can compute |
| **Public data** | $\widetilde{eq}(\tau; x)$ | 1K elements |  Yes |  Yes |  No | Verifier can compute |

---

##### Summary: The Three Reasons for Virtual Polynomials

**1. Intermediate derived values** (like $Az, Bz, Cz$)
   - Computed from committed witness + public data
   - Only needed temporarily between stages
   - Would be redundant to commit

**2. Public data** (like constraint matrices $A, B, C$)
   - Both prover and verifier have it
   - Verifier can compute evaluations directly
   - Committing would waste proof size

**3. Ephemeral computations** (like equality polynomials)
   - Used internally in sumcheck protocols
   - Generated on-the-fly from random challenges
   - Never need binding (recomputable from transcript)

**The virtual polynomial pattern** enables:
-  **Smaller proofs**: Avoid ~200 MB of unnecessary commitments
-  **Faster verification**: Verifier computes public data instead of checking openings
-  **Efficient chaining**: Sumcheck outputs become next sumcheck inputs seamlessly
-  **Cleaner abstraction**: Separate "what needs cryptographic binding" from "what can be recomputed"

**Bottom line**: We commit only to the **secret witness data** that needs cryptographic binding. Everything else that's derived, public, or temporary uses virtual polynomials to avoid proof bloat.

---

#### The Complete Spartan Sumcheck Flow: Where Virtual Polynomials Come From

**Your question**: "What sumcheck are we executing such that virtual intermediate polys appear? Why is the matrix public? Where was this made?"

**Answer**: This is **Spartan's three-stage sumcheck protocol** for R1CS. Let me show you the complete flow with concrete math and a toy example.

---

##### Background: What is R1CS?

**Rank-1 Constraint System (R1CS)** is a way to express computation as matrix equations.

**Mathematical form**:
$$(Az) \circ (Bz) = Cz$$

Where:

- $A, B, C \in \mathbb{F}^{m \times n}$ are **constraint matrices** (public)
- $z \in \mathbb{F}^n$ is the **witness vector** (secret)
- $\circ$ is element-wise product (Hadamard product)
- $m$ = number of constraints
- $n$ = number of variables

**Each row represents one constraint**:
$$\left(\sum_{j=1}^{n} A_{i,j} \cdot z_j\right) \cdot \left(\sum_{j=1}^{n} B_{i,j} \cdot z_j\right) = \sum_{j=1}^{n} C_{i,j} \cdot z_j \quad \text{for } i = 1, \ldots, m$$

**Toy Example**: Prove you know $x, y$ such that $x \cdot y = 35$ and $x + y = 12$

**Witness vector**: $z = (1, x, y, x \cdot y) = (1, 5, 7, 35)$

- $z_0 = 1$ (constant)
- $z_1 = x = 5$
- $z_2 = y = 7$
- $z_3 = x \cdot y = 35$

**Constraints** (m = 2):

**Constraint 1**: $x \cdot y = z_3$

- Left: $z_1 = 5$
- Right: $z_2 = 7$
- Output: $z_3 = 35$
- Matrix form: $(0 \cdot z_0 + 1 \cdot z_1 + 0 \cdot z_2 + 0 \cdot z_3) \cdot (0 \cdot z_0 + 0 \cdot z_1 + 1 \cdot z_2 + 0 \cdot z_3) = (0 \cdot z_0 + 0 \cdot z_1 + 0 \cdot z_2 + 1 \cdot z_3)$

**Constraint 2**: $x + y = 12$ (rewritten as $x \cdot 1 + y \cdot 1 - 12 \cdot 1 = 0$)

- Left: $z_1 + z_2 - 12 = 5 + 7 - 12 = 0$
- Right: $1$
- Output: $0$

**Constraint matrices**:
$$A = \begin{bmatrix} 0 & 1 & 0 & 0 \\ -12 & 1 & 1 & 0 \end{bmatrix}, \quad B = \begin{bmatrix} 0 & 0 & 1 & 0 \\ 1 & 0 & 0 & 0 \end{bmatrix}, \quad C = \begin{bmatrix} 0 & 0 & 0 & 1 \\ 0 & 0 & 0 & 0 \end{bmatrix}$$

**Matrix-vector products**:
$$Az = \begin{bmatrix} 5 \\ 0 \end{bmatrix}, \quad Bz = \begin{bmatrix} 7 \\ 1 \end{bmatrix}, \quad Cz = \begin{bmatrix} 35 \\ 0 \end{bmatrix}$$

**Check**: $(Az) \circ (Bz) = \begin{bmatrix} 5 \cdot 7 \\ 0 \cdot 1 \end{bmatrix} = \begin{bmatrix} 35 \\ 0 \end{bmatrix} = Cz$

 Correct!

---

##### Why Are Matrices A, B, C Public?

**Key insight**: The matrices $A, B, C$ define the **circuit/program structure**, not the witness data!

**What's public** (known to both prover and verifier):

- The computation being verified (e.g., "does $x \cdot y = 35$ and $x + y = 12$?")
- The constraint system structure ($A, B, C$ matrices)
- The number of constraints $m$ and variables $n$

**What's secret** (known only to prover):

- The witness values $z$ (e.g., $x = 5, y = 7$)

**In Jolt's case**:

- **Public**: The VM constraint system (same ~30 constraints repeated for each cycle)
  - Constraint 1: "If instruction writes to register, update register file"
  - Constraint 2: "PC increments correctly"
  - Constraint 3: "Branch flag computed correctly"
  - etc.
- **Secret**: The execution trace (which instructions executed, which registers used, what values)

**Where are matrices created?**: During **preprocessing** (Part 1):

The R1CS matrices are constructed once during preprocessing based on the VM circuit structure. They're **deterministic** given the program - anyone can recompute them.

---

##### Spartan's Three-Stage Sumcheck Protocol

Now let's see exactly where virtual polynomials appear.

**Goal**: Prove $(Az) \circ (Bz) = Cz$ holds for all $m$ constraints

**Challenge**: Direct verification requires $O(m)$ work (check each constraint)

**Spartan's solution**: Three nested sumchecks that reduce verification to a single point evaluation

---

###### Stage 1: Outer Sumcheck (Reduces m constraints to 1)

**File**: [jolt-core/src/r1cs/spartan_outer.rs](../jolt-core/src/r1cs/spartan_outer.rs)

**Claim to prove**:
$$\sum_{x \in \{0,1\}^{\log m}} \text{eq}(\tau, x) \cdot \left[\widetilde{Az}(x) \cdot \widetilde{Bz}(x) - \widetilde{Cz}(x)\right] = 0$$

Where:

- $\tau \in \mathbb{F}^{\log m}$ is a random challenge from verifier
- $\text{eq}(\tau, x) = \prod_{i=1}^{\log m} (\tau_i x_i + (1-\tau_i)(1-x_i))$ is the equality polynomial
- $\widetilde{Az}, \widetilde{Bz}, \widetilde{Cz}$ are **MLEs of the vectors** $Az, Bz, Cz$

**Why this works**: If $(Az) \circ (Bz) = Cz$ for all constraints, then:
$$\forall x \in \{0,1\}^{\log m}: \quad \widetilde{Az}(x) \cdot \widetilde{Bz}(x) - \widetilde{Cz}(x) = 0$$

Multiplying by $\text{eq}(\tau, x)$ and summing doesn't change this (equals 0).

**Sumcheck protocol**: Prover and verifier run $\log m$ rounds

** Important - Fiat-Shamir Transform (Non-Interactive)**:

In practice, this is **non-interactive** due to Fiat-Shamir:

- Prover doesn't wait for verifier's challenges
- Instead, prover computes all challenges deterministically: $r_i = H(\text{transcript} \| g_1 \| \ldots \| g_{i-1})$
- Prover sends **all** $\log m$ univariate polynomials at once: $(g_1, g_2, \ldots, g_{\log m})$
- Each $g_i$ is a univariate polynomial (degree at most $d$, typically $d \leq 3$)
- Representation: coefficients $(c_0, c_1, \ldots, c_d)$ where $g_i(X) = \sum_{j=0}^{d} c_j X^j$

**Concrete data sent** (for $\log m = 10$ rounds, degree $d = 3$):

- 10 polynomials × 4 coefficients each = **40 field elements** (~1.3 KB)

**Round 1**: Prover computes univariate polynomial $g_1(X_1)$:
$$g_1(X_1) = \sum_{x_2, \ldots, x_{\log m} \in \{0,1\}} \text{eq}(\tau, X_1, x_2, \ldots) \cdot [\widetilde{Az}(X_1, x_2, \ldots) \cdot \widetilde{Bz}(X_1, x_2, \ldots) - \widetilde{Cz}(X_1, x_2, \ldots)]$$

Prover evaluates: $g_1(0) + g_1(1) = 0$ (self-check)

Prover derives challenge: $r_1 = H(\tau \| g_1)$

**Round 2**: Prover computes $g_2(X_2)$ with $X_1$ bound to $r_1$

Prover derives challenge: $r_2 = H(\tau \| g_1 \| g_2)$

**Rounds 3-$\log m$**: Continue until all variables bound

**After $\log m$ rounds**:

Verifier has random point $\vec{r} = (r_1, \ldots, r_{\log m}) \in \mathbb{F}^{\log m}$

**Output claims** (this is where virtual polynomials appear!):
$$\text{eq}(\tau, \vec{r}) \cdot [\widetilde{Az}(\vec{r}) \cdot \widetilde{Bz}(\vec{r}) - \widetilde{Cz}(\vec{r})] = g_{\log m}(r_{\log m})$$

Verifier needs to check this equation, which requires knowing:

- $\text{eq}(\tau, \vec{r})$ ← Verifier computes directly (public)
- $\widetilde{Az}(\vec{r})$ ← **VIRTUAL POLYNOMIAL CLAIM**
- $\widetilde{Bz}(\vec{r})$ ← **VIRTUAL POLYNOMIAL CLAIM**
- $\widetilde{Cz}(\vec{r})$ ← **VIRTUAL POLYNOMIAL CLAIM**

**Accumulator after Stage 1**:
```rust
virtual_openings[Az] = (r, claimed_Az)
virtual_openings[Bz] = (r, claimed_Bz)
virtual_openings[Cz] = (r, claimed_Cz)
```

**Why virtual?** Because $Az, Bz, Cz$ are **derived values**:

- $Az = A \cdot z$ (matrix $A$ is public, witness $z$ is committed)
- We don't commit to $Az$ separately - that would be redundant!
- Instead, next stage proves these claims by reducing to claims about $A$ and $z$

---

###### Stage 2: Product Sumcheck (Reduces matrix-vector product to evaluations)

**File**: [jolt-core/src/r1cs/spartan_product.rs](../jolt-core/src/r1cs/spartan_product.rs)

**Goal**: Prove virtual claim "$\widetilde{Az}(\vec{r}) = v_A$" from Stage 1

**Recall**: $Az$ is a vector where $Az[i] = \sum_{j=1}^{n} A_{i,j} \cdot z_j$

**MLE of Az**:
$$\widetilde{Az}(\vec{r}) = \sum_{x \in \{0,1\}^{\log m}} \text{eq}(\vec{r}, x) \cdot (Az)[x]$$

But $(Az)[x] = \sum_{y \in \{0,1\}^{\log n}} A[x,y] \cdot z[y]$, so:
$$\widetilde{Az}(\vec{r}) = \sum_{x \in \{0,1\}^{\log m}} \sum_{y \in \{0,1\}^{\log n}} \text{eq}(\vec{r}, x) \cdot \widetilde{A}(x, y) \cdot \widetilde{z}(y)$$

Rearranging:
$$\widetilde{Az}(\vec{r}) = \sum_{y \in \{0,1\}^{\log n}} \widetilde{z}(y) \cdot \underbrace{\left[\sum_{x \in \{0,1\}^{\log m}} \text{eq}(\vec{r}, x) \cdot \widetilde{A}(x, y)\right]}_{\text{Call this } \widetilde{A}_{\vec{r}}(y)}$$

**Simplified claim**:
$$\widetilde{Az}(\vec{r}) = \sum_{y \in \{0,1\}^{\log n}} \widetilde{A}_{\vec{r}}(y) \cdot \widetilde{z}(y) = v_A$$

**Sumcheck protocol**: Run sumcheck over $\log n$ rounds to prove this sum

**After $\log n$ rounds**:

Verifier has random point $\vec{r}' = (r'_1, \ldots, r'_{\log n}) \in \mathbb{F}^{\log n}$

**Output claims**:
$$\widetilde{A}_{\vec{r}}(\vec{r}') \cdot \widetilde{z}(\vec{r}') = \text{final polynomial evaluation}$$

This requires knowing:

- $\widetilde{A}_{\vec{r}}(\vec{r}')$ ← Need to compute
- $\widetilde{z}(\vec{r}')$ ← **COMMITTED POLYNOMIAL CLAIM** (witness!)

**Accumulator after Stage 2**:
```rust
// Virtual claim (will handle in Stage 3)
virtual_openings[A_tau] = (r', claimed_A_tau)

// Committed claim (needs Dory opening in Stage 5)
committed_openings[WitnessZ] = (r', claimed_z)
```

**Why is $\widetilde{z}(\vec{r}')$ committed but $\widetilde{A}_{\vec{r}}(\vec{r}')$ virtual?**

- $\widetilde{z}$ is the **witness** (secret execution trace) - must be committed
- $\widetilde{A}_{\vec{r}}(\vec{r}')$ can be **computed from public matrix $A$** - no commitment needed!

---

** Connection to Part 2: What is the witness $z$ actually?**

We created **35 witness polynomials**. The witness vector $z$ is **not** these 35 polynomials directly! Instead:

**The R1CS witness $z$** is a **flattened vector** containing:

1. Public inputs/outputs
2. **Evaluations of the 35 committed polynomials at each cycle**

**Concrete example** for $T = 1024$ cycles:

$$z = \begin{bmatrix}
\text{public inputs} \\
\hline
\widetilde{L}(0), \widetilde{L}(1), \ldots, \widetilde{L}(1023) & \leftarrow \text{1024 values from LeftInstructionInput} \\
\widetilde{R}(0), \widetilde{R}(1), \ldots, \widetilde{R}(1023) & \leftarrow \text{1024 values from RightInstructionInput} \\
\widetilde{\Delta}_{\text{rd}}(0), \ldots, \widetilde{\Delta}_{\text{rd}}(1023) & \leftarrow \text{1024 values from RdInc} \\
\vdots & \leftarrow \text{Continue for all 35 polynomials} \\
\widetilde{\text{ra}}_0(0,0), \widetilde{\text{ra}}_0(0,1), \ldots & \leftarrow \text{262K values from InstructionRa(0)} \\
\vdots
\end{bmatrix}$$

**Size**: $n \approx 1024 \times 8 + 262144 \times 27 \approx 7$ million values

**Key distinction**:

- **Part 2 committed to 35 MLEs**: $\{\widetilde{L}, \widetilde{R}, \widetilde{\Delta}_{\text{rd}}, \ldots\}$ (polynomials over cycles)
- **Spartan witness $z$**: Evaluations of those MLEs at **all** cycle/table points (giant vector)
- **Spartan commits to $\widetilde{z}$**: MLE of the giant witness vector

---

** Critical Clarification: Dense vs Sparse, MLEs vs Vectors**

Let's be very precise about what we have:

**1. Is $z$ dense?**

 **YES!** The witness vector $z$ is **DENSE** - it explicitly stores all ~7 million field elements.

- **No sparsity optimization**: Every single value is stored
- **Memory footprint**: ~7 million × 32 bytes = ~224 MB for the witness vector
- **Why dense?**: R1CS requires access to arbitrary positions in $z$, so sparse representation doesn't help

**2. How does $z$ relate to the MLEs $\widetilde{L}, \widetilde{R}$, etc.?**

This is where it gets subtle! There are **three levels** of representation:

| Level | Object | Type | Domain | Size | Example |
|-------|--------|------|--------|------|---------|
| **Level 1** | Trace data | Raw vectors | Cycle indices | 1024 values | $L = [L_0, L_1, \ldots, L_{1023}]$ |
| **Level 2** | Part 2 MLEs | Multilinear polys | Boolean hypercube | $2^{10}$ points | $\widetilde{L}: \{0,1\}^{10} \to \mathbb{F}$ |
| **Level 3** | Spartan $z$ | Giant vector | Flat indices | 7M values | $z = [L_0, \ldots, L_{1023}, R_0, \ldots]$ |
| **Level 4** | Spartan $\widetilde{z}$ | MLE of $z$ | Boolean hypercube | $2^{23}$ points | $\widetilde{z}: \{0,1\}^{23} \to \mathbb{F}$ |

**The relationship**:

- **Level 1 → Level 2**: We computed MLEs of each trace component separately
  - $\widetilde{L}$ is the MLE of vector $L = [L_0, L_1, \ldots, L_{1023}]$
  - Domain: $\{0,1\}^{10}$ (since $2^{10} = 1024$ cycles)
  - We committed to $\widetilde{L}$ using Dory in Part 2

- **Level 1 → Level 3**: We concatenate all trace vectors into one giant vector $z$
  - $z = [\text{public}, L_0, \ldots, L_{1023}, R_0, \ldots, R_{1023}, \ldots]$
  - This is a **different data structure** - a flat vector, not a polynomial

- **Level 3 → Level 4**: We compute the MLE of the giant vector $z$
  - $\widetilde{z}$ is the MLE of the 7-million-element vector $z$
  - Domain: $\{0,1\}^{23}$ (since $2^{23} \approx 8$ million)
  - Spartan commits to $\widetilde{z}$ (in addition to the 35 Part 2 commitments)

**Key insight**: $\widetilde{L}$ and $\widetilde{z}$ are **different polynomials**!

- $\widetilde{L}(j)$ gives you the left input at cycle $j$ (10-dimensional input)
- $\widetilde{z}(i)$ gives you the value at position $i$ in the giant witness vector (23-dimensional input)
- At specific positions: $\widetilde{z}(i) = L_j$ where $i$ is the index in $z$ corresponding to cycle $j$ of $L$

**3. Do we have 35 commitments or 36?**

**Answer**: We have **36 total commitments**:

- **35 commitments from Part 2**: One for each $\widetilde{L}, \widetilde{R}, \widetilde{\Delta}_{\text{rd}}, \widetilde{\text{ra}}_0, \ldots, \widetilde{\text{mem}}_7$
- **1 commitment from Spartan**: For $\widetilde{z}$ (the MLE of the giant flattened witness vector)

All 36 get opened in Stage 5's batched opening proof.

**4. Is there a single sumcheck using the giant $z$ vector?**

**Yes and No** - it depends on how you count the hierarchy!

**Hierarchical structure**:

$$
\boxed{
\begin{array}{l}
\textbf{Stage 1: ONE "outer" sumcheck} \\
\text{Claim: } \sum_{x \in \{0,1\}^{\log m}} f(x) = 0 \text{ where } f(x) = \text{eq}(\tau, x) \cdot [(Az \circ Bz) - Cz](x) \\
\quad \downarrow \text{ (produces 3 virtual claims)} \\
\quad \boxed{
\begin{array}{l}
Az(\tau) = v_A \quad \text{(virtual claim)} \\
Bz(\tau) = v_B \quad \text{(virtual claim)} \\
Cz(\tau) = v_C \quad \text{(virtual claim)}
\end{array}
}
\end{array}
}
$$

$$\Downarrow$$

$$
\boxed{
\begin{array}{l}
\textbf{Stage 2: THREE "product" sumchecks (subproblems!)} \\
\\
\textbf{Sumcheck 1:} \text{ Prove } Az(\tau) = \sum_{y \in \{0,1\}^{\log n}} A(\tau, y) \cdot z(y) \\
\quad \downarrow \text{ (produces 2 claims)} \\
\quad \bullet \; A(\tau, \vec{r}') = v_{A,\tau} \quad \text{(virtual - public matrix)} \\
\quad \bullet \; z(\vec{r}') = v_z \quad \text{(COMMITTED - witness!)} \\
\\
\textbf{Sumcheck 2:} \text{ Prove } Bz(\tau) = \sum_{y \in \{0,1\}^{\log n}} B(\tau, y) \cdot z(y) \\
\quad \downarrow \text{ (produces same } z(\vec{r}') \text{ claim - batched!)} \\
\quad \bullet \; B(\tau, \vec{r}') = v_{B,\tau} \quad \text{(virtual - public matrix)} \\
\quad \bullet \; z(\vec{r}') = v_z \quad \text{(same point as Sumcheck 1!)} \\
\\
\textbf{Sumcheck 3:} \text{ Prove } Cz(\tau) = \sum_{y \in \{0,1\}^{\log n}} C(\tau, y) \cdot z(y) \\
\quad \downarrow \text{ (produces same } z(\vec{r}') \text{ claim - batched!)} \\
\quad \bullet \; C(\tau, \vec{r}') = v_{C,\tau} \quad \text{(virtual - public matrix)} \\
\quad \bullet \; z(\vec{r}') = v_z \quad \text{(same point as Sumcheck 1!)}
\end{array}
}
$$

**Key insight**:

- **"One" sumcheck perspective**: There's ONE top-level sumcheck verifying R1CS satisfaction
- **"Four" sumcheck perspective**: That one sumcheck spawns THREE subproblem sumchecks to resolve its virtual claims

**Result of Stages 1-2**: All three Stage 2 sumchecks produce the **same opening claim** for $\widetilde{z}(\vec{r}')$!

- This is batched: verifier only needs to check $\widetilde{z}$ at ONE random point
- Opening claim: "$\widetilde{z}(\vec{r}') = v_z$" where $\vec{r}' \in \mathbb{F}^{23}$

**Sumchecks using the original 35 MLEs** (from Part 2):

- **Stage 4**: ~36 sumchecks (Twist/Shout/R1CS-linking)
  - These produce opening claims for each of the 35 polynomials at their own random points
  - Example: "$\widetilde{L}(\vec{s}) = v_L$" where $\vec{s} \in \mathbb{F}^{10}$ (10-dimensional evaluation)

---

** How Twist/Shout Sumchecks Generate Opening Claims**

**Key insight**: Twist and Shout are NOT about opening Dory commitments. They are **memory checking protocols** that use sumcheck to verify correctness, and **as a side effect**, these sumchecks produce evaluation claims for our committed polynomials!

**The flow**:
```
Twist/Shout sumcheck verifies memory consistency
    ↓ (uses our committed polynomials in the sumcheck equation)
Sumcheck protocol reduces sum to random point evaluation
    ↓ (via Fiat-Shamir challenges)
Output: Opening claim for polynomial at random point
    ↓ (accumulated in StateManager)
Stage 5: Dory proves all accumulated opening claims together
```

**Example 1: Shout Read-Checking for Instruction Lookups**

**Goal**: Prove that instruction lookups are correct for chunk $i$ (e.g., `InstructionRa(0)`)

**The sumcheck equation**:
$$\sum_{j \in \{0,1\}^{\log T}} \sum_{k \in \{0,1\}^8} \widetilde{\text{ra}}_i(j, k) \cdot \left[ \text{lookup}(j, k) - \text{table}(k) \right] \stackrel{?}{=} 0$$

Where:

- $\widetilde{\text{ra}}_i(j, k)$: **Our committed one-hot polynomial** from Part 2 (InstructionRa(i))
- $\text{table}(k)$: Efficiently evaluable lookup table (e.g., ADD table for 4-bit inputs)
- $\text{lookup}(j, k)$: Expected lookup result (derived from $\widetilde{L}, \widetilde{R}$ operands)

**Sumcheck protocol** (via Fiat-Shamir):

1. **Initial claim**: Sum over all $j \in \{0,1\}^{10}$ and $k \in \{0,1\}^8$ equals 0
2. **Round 1-10**: Prover sends univariate polynomials binding variables $j_0, \ldots, j_9$
   - Verifier responds with random challenges $r_{j,0}, \ldots, r_{j,9}$
3. **Round 11-18**: Prover sends univariate polynomials binding variables $k_0, \ldots, k_7$
   - Verifier responds with random challenges $r_{k,0}, \ldots, r_{k,7}$
4. **Final claim** (after 18 rounds):
   $$\widetilde{\text{ra}}_i(\vec{r}_j, \vec{r}_k) \cdot [\text{lookup}(\vec{r}_j, \vec{r}_k) - \text{table}(\vec{r}_k)] \stackrel{?}{=} v_{\text{final}}$$

Where $\vec{r}_j = (r_{j,0}, \ldots, r_{j,9}) \in \mathbb{F}^{10}$ and $\vec{r}_k = (r_{k,0}, \ldots, r_{k,7}) \in \mathbb{F}^8$.

**Key observation**: This equation now requires evaluating $\widetilde{\text{ra}}_i$ at the random point $(\vec{r}_j, \vec{r}_k)$!

**Output claims**:

- **$\widetilde{\text{ra}}_i(\vec{r}_j, \vec{r}_k) = v_{\text{ra}}$** (COMMITTED - needs Dory opening!)
- **$\text{table}(\vec{r}_k) = v_{\text{table}}$** (PUBLIC - efficiently computable)
- **$\text{lookup}(\vec{r}_j, \vec{r}_k) = v_{\text{lookup}}$** (VIRTUAL - may be another sumcheck or derived from $\widetilde{L}, \widetilde{R}$)

**The committed opening claim** "$\widetilde{\text{ra}}_i(\vec{r}_j, \vec{r}_k) = v_{\text{ra}}$" gets added to StateManager's `committed_openings` map!

**Repeat for all 16 instruction chunks**: Each produces one opening claim for $\widetilde{\text{ra}}_0, \ldots, \widetilde{\text{ra}}_{15}$ at different random points.

---

**Example 2: Twist Write-Checking for RAM**

**Goal**: Prove RAM increments are correctly applied to memory addresses

**The sumcheck equation**:
$$\sum_{j \in \{0,1\}^{\log T}} \sum_{k \in \{0,1\}^{\log M}} \widetilde{\text{mem}}_i(j, k) \cdot \widetilde{\Delta}_{\text{ram}}(j) \stackrel{?}{=} \text{final}(k) - \text{initial}(k)$$

Where:

- $\widetilde{\text{mem}}_i(j, k)$: **Our committed one-hot polynomial** from Part 2 (RamRa(i)) - chunk $i$ of RAM addresses
- $\widetilde{\Delta}_{\text{ram}}(j)$: **Our committed increment polynomial** from Part 2 (RamInc)
- $\text{final}(k), \text{initial}(k)$: Public initial/final memory states

**Sumcheck protocol**:

1. **Initial claim**: Sum over all $j \in \{0,1\}^{10}$ and $k \in \{0,1\}^{18}$ equals expected total
2. **Rounds 1-10**: Bind cycle variables $j$ with challenges $\vec{r}_j$
3. **Rounds 11-28**: Bind address variables $k$ with challenges $\vec{r}_k$
4. **Final claim**:
   $$\widetilde{\text{mem}}_i(\vec{r}_j, \vec{r}_k) \cdot \widetilde{\Delta}_{\text{ram}}(\vec{r}_j) \stackrel{?}{=} v_{\text{final}}$$

**Output claims**:

- **$\widetilde{\text{mem}}_i(\vec{r}_j, \vec{r}_k) = v_{\text{mem}}$** (COMMITTED - needs Dory opening!)
- **$\widetilde{\Delta}_{\text{ram}}(\vec{r}_j) = v_{\text{delta}}$** (COMMITTED - needs Dory opening!)

**Both committed opening claims** get added to StateManager!

**Repeat for all 8 RAM chunks**: Each produces opening claims for $\widetilde{\text{mem}}_0, \ldots, \widetilde{\text{mem}}_7$ plus one claim for $\widetilde{\Delta}_{\text{ram}}$.

---

**Example 3: Connecting $\widetilde{L}$ (LeftInstructionInput)**

**Where does $\widetilde{L}$ appear?**

$\widetilde{L}(j)$ is the left operand at cycle $j$. This feeds into:

1. **Instruction lookup computation**: The lookup address $\widetilde{\text{ra}}_i(j, k)$ depends on decomposing operands $\widetilde{L}(j), \widetilde{R}(j)$
2. **R1CS constraints**: PC update and instruction decode may reference operands

**Typical Stage 4 sumcheck involving $\widetilde{L}$**:

$$\sum_{j \in \{0,1\}^{\log T}} \widetilde{L}(j) \cdot \text{constraint}(j) \stackrel{?}{=} v_{\text{expected}}$$

**After sumcheck with random challenges** $\vec{r}_j$:

**Output claim**: $\widetilde{L}(\vec{r}_j) = v_L$ (COMMITTED - needs Dory opening!)

---

**Complete Breakdown: All Twist/Shout Sumchecks**

Based on the actual Jolt implementation, here's the exact sumcheck breakdown across all stages:

**Stage 1: Spartan R1CS** (1 sumcheck)

- **Outer sumcheck**: Verifies $(Az) \circ (Bz) - Cz = 0$

**Stage 2: Spartan Product + Component Read/Write** (7 sumchecks total)

*Spartan (3 sumchecks):*

1. **Az product sumcheck**: Proves $Az(\tau) = \sum A(\tau, y) \cdot z(y)$
2. **Bz product sumcheck**: Proves $Bz(\tau) = \sum B(\tau, y) \cdot z(y)$
3. **Cz product sumcheck**: Proves $Cz(\tau) = \sum C(\tau, y) \cdot z(y)$

*Registers Twist (1 sumcheck):*

4. **RegistersReadWriteChecking**: Verifies register reads/writes are consistent
   - Uses: $\widetilde{\Delta}_{\text{rd}}$ (RdInc)
   - Proves: Reads return last written value, writes update correctly

*RAM Twist (3 sumchecks):*

5. **RafEvaluationSumcheck**: Evaluates RAM address MLE at random point
6. **RamReadWriteChecking**: Verifies RAM memory consistency
   - Uses: $\widetilde{\text{mem}}_0, \ldots, \widetilde{\text{mem}}_7$ + $\widetilde{\Delta}_{\text{ram}}$
7. **OutputSumcheck**: Verifies program outputs are correct

*Instruction Lookups Shout (1 sumcheck):*

8. **BooleanitySumcheck**: Proves instruction lookup addresses are Boolean

**Stage 3: More Component Verification** (7 sumchecks total)

*Spartan (1 sumcheck):*

1. **Matrix evaluation**: Direct computation (no actual sumcheck - verifier computes)

*Registers Twist (1 sumcheck):*

2. **ValEvaluationSumcheck**: Evaluates register increment MLE at random point
   - Opens: $\widetilde{\Delta}_{\text{rd}}$ at random point

*RAM Twist (1 sumcheck):*

3. **ValEvaluationSumcheck**: Evaluates RAM increment MLE at random point
   - Opens: $\widetilde{\Delta}_{\text{ram}}$ at random point

*Instruction Lookups Shout (2 sumchecks):*

4. **ReadRafSumcheck**: Proves instruction lookups are correct
   - Uses: $\widetilde{\text{ra}}_0, \ldots, \widetilde{\text{ra}}_{15}$ (all 16 chunks)
   - Verifies: Lookups match pre-computed instruction tables
5. **HammingWeightSumcheck**: Proves one-hot property for lookup addresses
   - Verifies: Each cycle looks up exactly one table entry per chunk

**Stage 4: Final Component Verification** (8 sumchecks total)

*Instruction Lookups Shout (1 sumcheck):*

1. **RaSumcheck** (Ra virtualization): Links chunked representation
   - Opens: All 16 $\widetilde{\text{ra}}_i$ polynomials at random points

*Bytecode Shout (3 sumchecks):*

2. **ReadRafSumcheck**: Proves bytecode reads are correct
   - Uses: $\widetilde{\text{bc}}_0, \widetilde{\text{bc}}_1, \widetilde{\text{bc}}_2$
3. **BooleanitySumcheck**: Proves bytecode addresses are Boolean
4. **HammingWeightSumcheck**: Proves one-hot property for bytecode addresses
   - Opens: All 3 $\widetilde{\text{bc}}_i$ polynomials at random points

*RAM Twist (4 sumchecks):*

5. **RaSumcheck** (Ra virtualization): Links RAM address chunks
   - Opens: All 8 $\widetilde{\text{mem}}_i$ polynomials at random points
6. **BooleanitySumcheck**: Proves RAM addresses are Boolean
7. **HammingWeightSumcheck**: Proves one-hot property for RAM addresses
8. **ValFinalSumcheck**: Verifies final RAM state

*R1CS Linking (included in above):*

- **Input/output linking**: Connects $\widetilde{L}, \widetilde{R}$ operands to lookups
- Opens: $\widetilde{L}, \widetilde{R}$, and other operand polynomials

---

**Summary Table: Sumcheck Count by Component**

| Stage | Component | Sumchecks | Polynomials Opened |
|-------|-----------|-----------|-------------------|
| **1** | Spartan Outer | 1 | - |
| **2** | Spartan Product | 3 | $\widetilde{z}$ |
| **2** | Registers Twist | 1 | - |
| **2** | RAM Twist | 3 | - |
| **2** | Instruction Shout | 1 | - |
| **3** | Registers Twist | 1 | $\widetilde{\Delta}_{\text{rd}}$ |
| **3** | RAM Twist | 1 | $\widetilde{\Delta}_{\text{ram}}$ |
| **3** | Instruction Shout | 2 | - |
| **4** | Instruction Shout | 1 | $\widetilde{\text{ra}}_0, \ldots, \widetilde{\text{ra}}_{15}$ (16 polys) |
| **4** | Bytecode Shout | 3 | $\widetilde{\text{bc}}_0, \widetilde{\text{bc}}_1, \widetilde{\text{bc}}_2$ (3 polys) |
| **4** | RAM Twist | 4 | $\widetilde{\text{mem}}_0, \ldots, \widetilde{\text{mem}}_7$ (8 polys) |
| **4** | R1CS Linking | (included) | $\widetilde{L}, \widetilde{R}$, etc. (6 polys) |
| **Total** | | **~23 sumchecks** | **36 polynomial openings** |

**Key observations**:

1. **Not every sumcheck produces an opening claim** - some verify virtual polynomials
2. **Some sumchecks open multiple polynomials** - e.g., RaSumcheck opens all 16 instruction chunks
3. **Total opening claims: 36** = 1 ($\widetilde{z}$) + 35 (Part 2 MLEs)
4. **Stage 5 batches all 36** into single Dory opening proof

**The final step**: All 35 opening claims (plus 1 for Spartan's $\widetilde{z}$) are batched together in Stage 5 and proven with a single Dory opening proof!

---

** Wait... If Spartan Already Proves z, Why Do We Need Twist/Shout?**

**Critical conceptual question**: The Spartan sumchecks already verify the witness $z$ is correct for the R1CS constraints. So why do we need 35 additional sumchecks in Stage 4?

**Answer**: Spartan and Twist/Shout prove **completely different things**!

**What Spartan Proves** (Stages 1-3):

$$\text{Spartan verifies: } (Az) \circ (Bz) - Cz = \vec{0}$$

This proves that **IF** the witness $z$ contains the claimed values, **THEN** those values satisfy the R1CS constraints.

**Concrete example from our toy problem**:

- Claim: $z = (1, 5, 7, 35)$ satisfies constraints for $x \cdot y = 35$ and $x + y = 12$
- Spartan proves: "Yes, **IF** $z$ really contains $(1, 5, 7, 35)$, then the constraints are satisfied"

**What Spartan does NOT prove**:
-  That instruction lookups are correct
-  That memory operations are consistent
-  That register updates follow execution order
-  That bytecode was decoded correctly
-  **That the witness values in $z$ actually correspond to a valid RISC-V execution!**

**The Gap**:

Spartan only verifies **arithmetic relationships** between values in $z$. But $z$ is just a giant vector of numbers! Spartan doesn't know:

- Which numbers represent instruction operands
- Which numbers represent memory addresses
- Which numbers represent lookup table indices
- **Whether these numbers form a valid VM execution trace**

**Example Attack (without Twist/Shout)**:

Suppose a malicious prover constructs a fake witness:
```
z = [public, L_0, L_1, ..., R_0, R_1, ..., ra_0(0,0), ra_0(0,1), ...]
```

The prover could:

1.  Make sure $z$ satisfies the ~30 R1CS constraints (Spartan passes!)
2.  But set $\widetilde{\text{ra}}_0(j, k)$ to garbage values (wrong lookups!)
3.  Or set $\widetilde{\text{mem}}_i(j, k)$ to violate memory consistency (reads return wrong values!)
4.  Or set $\widetilde{\Delta}_{\text{ram}}(j)$ to arbitrary increments (corrupt memory!)

**Spartan wouldn't catch this** because the R1CS constraints only verify **high-level properties** like:

- PC increments correctly
- Operands feed into correct lookup indices
- Results get written to correct registers

But Spartan doesn't verify:

- **The lookups themselves are correct** (Shout's job!)
- **Memory operations are consistent** (Twist's job!)

---

**What Twist/Shout Prove** (Stage 4):

**Twist (Memory Consistency)**:

For RAM chunk $i$:
$$\sum_{j,k} \widetilde{\text{mem}}_i(j, k) \cdot \widetilde{\Delta}_{\text{ram}}(j) \stackrel{?}{=} \text{final}(k) - \text{initial}(k)$$

This proves:

- Every memory read returns the value from the most recent write
- Memory increments are correctly routed to addresses
- Final memory state matches initial state plus all increments

**Without Twist**: Prover could make memory operations return arbitrary values!

**Shout (Lookup Correctness)**:

For instruction chunk $i$:
$$\sum_{j,k} \widetilde{\text{ra}}_i(j, k) \cdot [\text{lookup}(j,k) - \text{table}(k)] \stackrel{?}{=} 0$$

This proves:

- Every instruction lookup accessed the correct table entry
- The one-hot property: each cycle looks up exactly one entry
- Lookup results match the pre-computed table

**Without Shout**: Prover could claim that `ADD(5, 7) = 100` instead of 12!

---

**The Division of Labor**:

| Protocol | What It Proves | Example |
|----------|----------------|---------|
| **Spartan** | Arithmetic constraints satisfied | "IF operands are (5, 7) THEN result goes to register rd" |
| **Shout** | Lookups are correct | "ADD(5, 7) actually equals 12 (from table)" |
| **Twist** | Memory is consistent | "Reading address 0x1000 returns the last value written there" |
| **R1CS Linking** | Components connect correctly | "Result from lookup matches value in witness" |

**Key insight**: Spartan proves **relationships**, Twist/Shout prove **ground truth**!

---

**Concrete Example: Proving a 4-bit ADD Instruction**

Let's walk through **all sumchecks** for a simple example to see how everything connects.

**Setup**: Simplified 4-bit zkVM with 1-bit operand chunks (for clarity)

- **Instruction**: `ADD r3, r1, r2` (compute `r1 + r2`, store in `r3`)
- **Operands**: `r1 = 5`, `r2 = 7` (in binary: `0101` and `0111`)
- **Expected result**: `r3 = 12` (in binary: `1100`)
- **Cycles**: $T = 1$ (single instruction)
- **Chunks**: $d = 4$ (four 1-bit chunks per operand, for simplicity)

---

**Step 1: Witness Construction (Part 2)**

From execution trace, we construct witness polynomials:

1. **Operands** (Type 1 - simple vectors):
   - $\widetilde{L}(0) = 5$ (left operand at cycle 0)
   - $\widetilde{R}(0) = 7$ (right operand at cycle 0)
   - $\widetilde{\Delta}_{\text{rd}}(0) = 12 - 0 = 12$ (register increment: r3 goes from 0 to 12)

2. **Instruction lookup addresses** (Type 2 - one-hot matrices):

   Decompose operands into 1-bit chunks: $5 = (0,1,0,1)$, $7 = (0,1,1,1)$

   For chunk 0 (bit position 0): operands are $(L_0, R_0) = (1, 1)$ → lookup index $k = 3$ (binary: `11`)

   One-hot encoding: $\widetilde{\text{ra}}_0(j=0, k) = \begin{cases} 1 & \text{if } k = 3 \\ 0 & \text{otherwise} \end{cases}$

   So: $\widetilde{\text{ra}}_0(0, 0) = 0, \widetilde{\text{ra}}_0(0, 1) = 0, \widetilde{\text{ra}}_0(0, 2) = 0, \widetilde{\text{ra}}_0(0, 3) = 1$

   Similarly for chunks 1, 2, 3:
   - Chunk 1: $(L_1, R_1) = (0, 1)$ → $k = 1$ → $\widetilde{\text{ra}}_1(0, 1) = 1$
   - Chunk 2: $(L_2, R_2) = (1, 1)$ → $k = 3$ → $\widetilde{\text{ra}}_2(0, 3) = 1$
   - Chunk 3: $(L_3, R_3) = (0, 0)$ → $k = 0$ → $\widetilde{\text{ra}}_3(0, 0) = 1$

3. **Commit to all polynomials** via Dory:
   - $C_L = \text{Commit}(\widetilde{L})$
   - $C_R = \text{Commit}(\widetilde{R})$
   - $C_{\Delta_{\text{rd}}} = \text{Commit}(\widetilde{\Delta}_{\text{rd}})$
   - $C_{\text{ra}_0}, \ldots, C_{\text{ra}_3} = \text{Commit}(\widetilde{\text{ra}}_0), \ldots, \text{Commit}(\widetilde{\text{ra}}_3)$

4. **Flatten into Spartan witness** $z$:

   $$z = [1, \; 5, \; 7, \; 12, \; 0, 0, 0, 1, \; 0, 1, 0, 0, \; 0, 0, 0, 1, \; 1, 0, 0, 0, \; \ldots]$$

   - Position 0: public constant (1)
   - Position 1: $\widetilde{L}(0) = 5$
   - Position 2: $\widetilde{R}(0) = 7$
   - Position 3: $\widetilde{\Delta}_{\text{rd}}(0) = 12$
   - Positions 4-7: $\widetilde{\text{ra}}_0(0, k)$ for $k = 0, 1, 2, 3$
   - Positions 8-11: $\widetilde{\text{ra}}_1(0, k)$ for $k = 0, 1, 2, 3$
   - ... (continue for all chunks)

5. **Commit to $\widetilde{z}$**: $C_z = \text{Commit}(\widetilde{z})$

---

**Step 2: Proof Generation - All Sumchecks**

**Stage 1: Spartan Outer Sumcheck**

**Claim**:
$$\sum_{x \in \{0,1\}} \text{eq}(\tau, x) \cdot [(Az)(x) \cdot (Bz)(x) - (Cz)(x)] = 0$$

Where $\tau$ is a random challenge (say $\tau = 0.37$ in field $\mathbb{F}$).

**What constraints are being checked?** (~30 constraints including):

- Constraint 1: $\text{PC}_{\text{next}} = \text{PC}_{\text{curr}} + 4$ (since no jump/branch)
- Constraint 2: $\text{rd\_value} = \text{lookup\_result}$ (result goes to r3)
- Constraint 3: $L = \text{register}[\text{rs1}]$ (left operand from r1)
- Constraint 4: $R = \text{register}[\text{rs2}]$ (right operand from r2)
- ... (26 more constraints)

**After sumcheck**: Produces 3 virtual claims:

- $Az(\tau) = v_A$ (say $v_A = 42.7$)
- $Bz(\tau) = v_B$ (say $v_B = 31.2$)
- $Cz(\tau) = v_C$ (say $v_C = 1332.24$)

**Verifier checks**: $v_A \cdot v_B - v_C \stackrel{?}{=} 0$ → $42.7 \times 31.2 - 1332.24 = 0$ 

---

**Stage 2: Spartan Product Sumchecks**

**Sumcheck 1 - Prove $Az(\tau)$**:

$$Az(\tau) = \sum_{y \in \{0,1\}^{\log n}} A(\tau, y) \cdot z(y)$$

Expand for our small witness ($n = 20$ positions, so $\log n = 5$ bits):

$$Az(\tau) = A(\tau, 0) \cdot z(0) + A(\tau, 1) \cdot z(1) + \cdots + A(\tau, 19) \cdot z(19)$$

Where $A(\tau, y)$ is the MLE of constraint matrix $A$ evaluated at row $\tau$ and column $y$.

**After sumcheck with random challenges** $\vec{r}' = (r_0, r_1, r_2, r_3, r_4)$:

**Output claim**: $\widetilde{z}(\vec{r}') = v_z$ (say $v_z = 8.342$)

(Similarly for $Bz$ and $Cz$ sumchecks - they produce the **same** $\widetilde{z}(\vec{r}')$ claim!)

---

**Stage 2: Registers Twist - Read/Write Checking**

**Claim**: Register reads/writes are consistent

$$\sum_{j \in \{0,1\}} \sum_{k \in \{0,1\}^2} \widetilde{\text{ra}}_{\text{reg}}(j, k) \cdot [\text{read}(j, k) - \text{expected}(k)] = 0$$

Where:

- $j = 0$ (only 1 cycle)
- $k$ ranges over 4 registers (r0, r1, r2, r3 in our simplified example)

**Concretely**:

- Read r1 (k=1): expected value = 5
- Read r2 (k=2): expected value = 7
- Write r3 (k=3): increment = 12

**After sumcheck**: Produces claims for register address polynomials at random points.

---

**Stage 3: Instruction Shout - Read/Write Checking**

**For chunk 0** (bit position 0):

**Claim**: Lookup is correct
$$\sum_{j=0} \sum_{k \in \{0,1\}^2} \widetilde{\text{ra}}_0(j, k) \cdot [\text{chunk\_result}(j, k) - \text{ADD\_table}(k)] = 0$$

Where $\text{ADD\_table}$ is the 1-bit ADD table:

- $\text{ADD\_table}(0) = 0 + 0 = 0$ (binary: `00` → 0)
- $\text{ADD\_table}(1) = 0 + 1 = 1$ (binary: `01` → 1)
- $\text{ADD\_table}(2) = 1 + 0 = 1$ (binary: `10` → 1)
- $\text{ADD\_table}(3) = 1 + 1 = 0$ (binary: `11` → 0, with carry)

**Expand for cycle $j=0$**:

$$\sum_{k=0}^{3} \widetilde{\text{ra}}_0(0, k) \cdot [\text{chunk\_result}(0, k) - \text{ADD\_table}(k)]$$

$$= \widetilde{\text{ra}}_0(0, 0) \cdot [0 - 0] + \widetilde{\text{ra}}_0(0, 1) \cdot [0 - 1] + \widetilde{\text{ra}}_0(0, 2) \cdot [0 - 1] + \widetilde{\text{ra}}_0(0, 3) \cdot [0 - 0]$$

Recall: $\widetilde{\text{ra}}_0(0, 3) = 1$ (one-hot), all others = 0

$$= 0 \cdot 0 + 0 \cdot (-1) + 0 \cdot (-1) + 1 \cdot 0 = 0$$

 Verified!

**After sumcheck with challenges** $\vec{r}_j, \vec{r}_k$:

**Output claim**: $\widetilde{\text{ra}}_0(\vec{r}_j, \vec{r}_k) = v_{\text{ra}_0}$ (needs Dory opening!)

**Repeat for chunks 1, 2, 3**: Each produces opening claim for $\widetilde{\text{ra}}_1, \widetilde{\text{ra}}_2, \widetilde{\text{ra}}_3$.

---

**Stage 3: Hamming Weight Sumcheck**

**For chunk 0**:

**Claim**: One-hot property holds
$$\sum_{j=0} \sum_{k=0}^{3} \widetilde{\text{ra}}_0(j, k) = T = 1$$

Expand:
$$\widetilde{\text{ra}}_0(0, 0) + \widetilde{\text{ra}}_0(0, 1) + \widetilde{\text{ra}}_0(0, 2) + \widetilde{\text{ra}}_0(0, 3) = 0 + 0 + 0 + 1 = 1$$

 This ensures each cycle looks up **exactly one** table entry.

---

**Stage 4: R1CS Linking**

**Claim**: Operands from witness match lookup decomposition

For chunk 0, verify that lookup index $k = 3$ corresponds to bits $(L_0, R_0) = (1, 1)$:

$$\sum_{j=0} \widetilde{L}(j) \cdot \text{bit\_extractor}_0(j) \stackrel{?}{=} \sum_{j=0} \sum_{k=0}^{3} \widetilde{\text{ra}}_0(j, k) \cdot \text{bit}_0(k)$$

Where:

- $\text{bit\_extractor}_0(j)$ extracts bit 0 from $\widetilde{L}(j)$ → from 5 = `0101`, bit 0 = 1
- $\text{bit}_0(k)$ extracts bit 0 from index $k$ → from 3 = `11`, bit 0 = 1

**After sumcheck**: Produces opening claims for $\widetilde{L}(\vec{r}_j)$ and $\widetilde{R}(\vec{r}_j)$.

---

**Stage 5: Batched Dory Opening**

**All accumulated opening claims**:

1. $\widetilde{z}(\vec{r}') = v_z$ (from Spartan)
2. $\widetilde{L}(\vec{r}_L) = v_L$ (from linking)
3. $\widetilde{R}(\vec{r}_R) = v_R$ (from linking)
4. $\widetilde{\Delta}_{\text{rd}}(\vec{r}_{\Delta}) = v_{\Delta}$ (from registers)
5. $\widetilde{\text{ra}}_0(\vec{r}_{\text{ra}_0}) = v_{\text{ra}_0}$ (from Shout)
6. $\widetilde{\text{ra}}_1(\vec{r}_{\text{ra}_1}) = v_{\text{ra}_1}$ (from Shout)
7. $\widetilde{\text{ra}}_2(\vec{r}_{\text{ra}_2}) = v_{\text{ra}_2}$ (from Shout)
8. $\widetilde{\text{ra}}_3(\vec{r}_{\text{ra}_3}) = v_{\text{ra}_3}$ (from Shout)

**Batched opening proof**: Single Dory proof verifies all 8 claims together!

---

**Summary: Complete Verification Chain**

| Component | What It Verified | Concrete Check |
|-----------|------------------|----------------|
| **Spartan Outer** | Constraints satisfied | $\text{PC} + 4$, result to r3, etc. |
| **Spartan Product** | Witness consistent with constraints | $\widetilde{z}$ evaluation correct |
| **Twist (Registers)** | Register reads correct | r1 = 5, r2 = 7 before ADD |
| **Twist (Registers)** | Register writes correct | r3 = 12 after ADD |
| **Shout (Chunk 0)** | Lookup correct | ADD(1, 1) = 0 with carry |
| **Shout (Chunk 1)** | Lookup correct | ADD(0, 1) = 1 |
| **Shout (Chunk 2)** | Lookup correct | ADD(1, 1) = 0 with carry |
| **Shout (Chunk 3)** | Lookup correct | ADD(0, 0) = 0 + carry = 1 |
| **Hamming Weight** | One-hot property | Each chunk: exactly 1 lookup per cycle |
| **R1CS Linking** | Operands match lookups | $L = 5$ decomposed to $(1,0,1,0)$ |
| **Dory Opening** | All commitments valid | 8 polynomial evaluations proven |

**Result**: Complete proof that `ADD r3, r1, r2` was correctly executed with inputs (5, 7) and output 12!

---

**Key Insights**:

1. **Each protocol verifies a different aspect**: No single protocol is sufficient
2. **Lookup decomposition is key**: 4-bit ADD requires 4 separate 1-bit lookups (with carries)
3. **One-hot encoding**: Ensures exactly one table lookup per chunk per cycle
4. **Linking is critical**: Connects high-level operands ($L = 5$) to low-level lookups (bit chunks)
5. **Batched opening**: All commitments proven together for efficiency

**Without any one component, the proof fails**:

- No Spartan → Can't verify constraints
- No Twist → Can't verify register operations
- No Shout → Can't verify ADD semantics
- No Linking → Can't connect operands to lookups
- No Dory → Can't verify commitments are honest

---

**Why Not Encode Everything in R1CS?**

**Option 1: Pure R1CS approach** (what traditional zkVMs do):

- Encode instruction semantics as arithmetic circuits
- Result: Thousands of constraints per instruction
- Prover cost: $O(\text{constraints})$ = huge!

**Option 2: Jolt's hybrid approach**:

- R1CS: Only ~30 constraints per cycle (control flow, linking)
- Shout: Instruction semantics via lookups (much cheaper!)
- Twist: Memory consistency via fingerprints (efficient!)
- Result: Best of both worlds!

**The trade-off**:

- More protocols (Spartan + Twist + Shout) = more complexity
- But much better performance (fewer constraints, efficient lookups)

---

**Total opening claims at end of Stage 4**: 36 claims

- 1 claim about $\widetilde{z}$ at a 23D point (from Spartan)
- 35 claims about the Part 2 MLEs at various 10D or 18D points (from Twist/Shout/Linking)

**Stage 5**: Batched Dory opening proves all 36 claims together!

---

###### Stage 3: Matrix Evaluation (Compute public matrix MLE)

**File**: [jolt-core/src/r1cs/spartan_matrix_eval.rs](../jolt-core/src/r1cs/spartan_matrix_eval.rs)

**Goal**: Verify virtual claim "$\widetilde{A}_{\vec{r}}(\vec{r}') = v_{A,\tau}$" from Stage 2

**Recall**:
$$\widetilde{A}_{\vec{r}}(\vec{r}') = \sum_{x \in \{0,1\}^{\log m}} \text{eq}(\vec{r}, x) \cdot \widetilde{A}(x, \vec{r}')$$

**Key insight**: Both prover and verifier can compute this directly!

**Why?** Because:

- $\vec{r}$ and $\vec{r}'$ are public (from transcript)
- $A$ is public (constraint matrix)
- $\text{eq}(\vec{r}, x)$ is efficiently computable
- $\widetilde{A}(x, \vec{r}')$ is the MLE of public matrix $A$

**Efficient computation**:
$$\widetilde{A}(x, \vec{r}') = \sum_{(i,j) \in \{0,1\}^{\log m} \times \{0,1\}^{\log n}} A[i,j] \cdot \text{eq}(x; i) \cdot \text{eq}(\vec{r}'; j)$$

**Complexity**: $O(m \cdot n)$ operations

**In Jolt**: $m = 30 \times 1024 = 30,720$, $n = 35$ → ~1M operations (milliseconds!)

**Result**: Verifier directly computes $\widetilde{A}_{\vec{r}}(\vec{r}')$ and checks it matches prover's claim

**No sumcheck needed** - just direct computation!

---

##### Toy Example: Complete Flow

Let's trace through our toy example: prove $x \cdot y = 35$ and $x + y = 12$

**Setup**:

- $m = 2$ constraints → $\log m = 1$
- $n = 4$ variables → $\log n = 2$
- $z = (1, 5, 7, 35)$

**Stage 1: Outer Sumcheck**

Verifier sends random $\tau \in \mathbb{F}$, say $\tau = 42$

Claim to prove:
$$\sum_{x \in \{0,1\}} \text{eq}(42, x) \cdot [\widetilde{Az}(x) \cdot \widetilde{Bz}(x) - \widetilde{Cz}(x)] = 0$$

Where:

- $\widetilde{Az}(0) = 5$ (first constraint: left side)
- $\widetilde{Az}(1) = 0$ (second constraint: left side)
- $\widetilde{Bz}(0) = 7$, $\widetilde{Bz}(1) = 1$
- $\widetilde{Cz}(0) = 35$, $\widetilde{Cz}(1) = 0$

Prover computes:

- $\text{eq}(42, 0) = 1 - 42 = -41$
- $\text{eq}(42, 1) = 42$

Sum:
$$(-41) \cdot (5 \cdot 7 - 35) + 42 \cdot (0 \cdot 1 - 0) = (-41) \cdot 0 + 42 \cdot 0 = 0$$

 Verified!

**Round 1**: Prover sends $g_1(X_1) = \text{eq}(42, X_1) \cdot [\widetilde{Az}(X_1) \cdot \widetilde{Bz}(X_1) - \widetilde{Cz}(X_1)]$

Verifier checks $g_1(0) + g_1(1) = 0$

 Verified!

Verifier sends random $r_1 = 123$

**Output claims**:
$$\widetilde{Az}(123) = ?, \quad \widetilde{Bz}(123) = ?, \quad \widetilde{Cz}(123) = ?$$

These are **virtual claims** - added to accumulator

**Stage 2: Product Sumcheck (for Az)**

Prove: $\widetilde{Az}(123) = \sum_{y \in \{0,1\}^2} \widetilde{A}_{123}(y) \cdot \widetilde{z}(y)$

Where $\widetilde{A}_{123}(y) = \sum_{x \in \{0,1\}} \text{eq}(123, x) \cdot \widetilde{A}(x, y)$

**Round 1**: Sumcheck over $y_1$, verifier sends $r'_1 = 456$

**Round 2**: Sumcheck over $y_2$, verifier sends $r'_2 = 789$

**Output claims**:
$$\widetilde{A}_{123}(456, 789) = ?, \quad \widetilde{z}(456, 789) = ?$$

- $\widetilde{z}(456, 789)$ is **committed** (witness) → added to `committed_openings`
- $\widetilde{A}_{123}(456, 789)$ is **virtual** (public matrix) → verify in Stage 3

**Stage 3: Matrix Evaluation**

Both prover and verifier compute:
$$\widetilde{A}_{123}(456, 789) = \sum_{x \in \{0,1\}} \text{eq}(123, x) \cdot \widetilde{A}(x, 456, 789)$$

Where:
$$\widetilde{A}(x, 456, 789) = \sum_{(i,j) \in \{0,1\} \times \{0,1\}^2} A[i,j] \cdot \text{eq}(x; i) \cdot \text{eq}((456,789); j)$$

Since $A$ is public:
$$A = \begin{bmatrix} 0 & 1 & 0 & 0 \\ -12 & 1 & 1 & 0 \end{bmatrix}$$

Both compute the same value → virtual claim verified!

 Stage 3 complete!

---

##### Summary: The Spartan Flow

**Stage 1 (Outer Sumcheck):**

- **Input**: Constraint satisfaction claim $(Az) \circ (Bz) = Cz$
- **Output**: Virtual claims about $Az(\vec{r}), Bz(\vec{r}), Cz(\vec{r})$

**Stage 2 (Product Sumcheck):**

- **Input**: Virtual claim $Az(\vec{r}) = v_A$
- **Output**:
  - Virtual claim $\widetilde{A}_{\vec{r}}(\vec{r}') = v_A'$ (public matrix)
  - Committed claim $\widetilde{z}(\vec{r}') = v_z$ (witness)

**Stage 3 (Matrix Eval):**

- **Input**: Virtual claim $\widetilde{A}_{\vec{r}}(\vec{r}') = v_A'$
- **Action**: Both parties compute $\widetilde{A}_{\vec{r}}(\vec{r}')$ from public $A$
- **Output**: Verified! (no further claims)

**Stage 5 (Dory Opening):**

- **Input**: All committed claims ($\widetilde{z}(\vec{r}')$ and 35 witness polynomials)
- **Output**: Batched opening proof

**Key insights**:

1. **Virtual polynomials appear naturally** during sumcheck reduction
   - Stage 1 creates claims about $Az, Bz, Cz$ (derived from witness)
   - Stage 2 creates claims about $\widetilde{A}_{\vec{r}}$ (derived from public matrix)

2. **Matrices $A, B, C$ are public** because they encode the circuit structure
   - Created during preprocessing
   - Anyone can recompute them from the program
   - Both prover and verifier have them

3. **Virtual vs Committed distinction**:
   - Virtual: Can be recomputed (public data) or proven by next sumcheck (derived values)
   - Committed: Secret witness data that needs cryptographic binding

4. **Three-stage nesting enables efficiency**:
   - Stage 1: $m$ constraints → 1 random point
   - Stage 2: $m \times n$ matrix-vector product → 1 random point
   - Stage 3: Direct computation (no proof needed)

---

##### Concrete Example: Following $\widetilde{L}$ Through All Stages

The complete ADD example (Section "Concrete Example: Proving a 4-bit ADD Instruction") shows the full flow. Here's a quick summary of how one polynomial travels through the proof:

**$\widetilde{L}$ journey**:

- **Part 2**: Created from execution trace, committed via Dory: $C_L \in \mathbb{G}_T$
- **Stages 1-3**: Embedded in Spartan's flattened witness $z$, indirectly evaluated at random 23D point
- **Stage 4**: Directly evaluated at random 10D point via R1CS linking sumcheck
- **Stage 5**: Both evaluation claims (23D and 10D) proven together via batched Dory opening

For the detailed walkthrough with concrete values, see the ADD instruction example above.

---

#### Data Structure Definition

**File**: [jolt-core/src/poly/opening_proof.rs:45-52](../jolt-core/src/poly/opening_proof.rs#L45)

```rust
pub struct ProverOpeningAccumulator<F: JoltField> {
    // Virtual polynomial evaluation claims (proven by subsequent sumchecks in Stages 2-4)
    virtual_openings: HashMap<VirtualPolynomialId, (OpeningPoint, F)>,
    //                         ↑                    ↑             ↑
    //                         |                    |             └─ Claimed value v \in F
    //                         |                    └─ Random point r = (r_1,...,r_n) \in Fⁿ
    //                         └─ Which virtual polynomial (Az, Bz, Cz, etc.)

    // Committed polynomial evaluation claims (proven by Dory opening in Stage 5)
    committed_openings: HashMap<CommittedPolynomialId, (OpeningPoint, F)>,
    //                          ↑                       ↑              ↑
    //                          |                       |              └─ Claimed value v \in F
    //                          |                       └─ Random point r \in Fⁿ
    //                          └─ Which committed polynomial (from Part 2)
}
```

---

#### What is `VirtualPolynomialId`?

**Definition**: An enum identifying which virtual polynomial we're claiming about.

**File**: [jolt-core/src/poly/opening_proof.rs](../jolt-core/src/poly/opening_proof.rs)

```rust
pub enum VirtualPolynomialId {
    // Spartan R1CS virtual polynomials
    Az,    // Product of constraint matrix A with witness z
    Bz,    // Product of constraint matrix B with witness z
    Cz,    // Product of constraint matrix C with witness z

    // Spartan matrix MLEs (bivariate)
    A_tau, // Matrix A evaluated at (\tau, ·) where \tau is random challenge
    B_tau, // Matrix B evaluated at (\tau, ·)
    C_tau, // Matrix C evaluated at (\tau, ·)
}
```

**What each represents**:

| Virtual Polynomial | Mathematical Object | Domain | Created in Stage | Proven in Stage |
|-------------------|---------------------|--------|------------------|-----------------|
| `Az` | $\widetilde{Az}(\vec{r})$ where $Az = (A_1 \cdot z, \ldots, A_m \cdot z)$ | $\mathbb{F}^{\log m}$ | Stage 1 | Stage 2 |
| `Bz` | $\widetilde{Bz}(\vec{r})$ where $Bz = (B_1 \cdot z, \ldots, B_m \cdot z)$ | $\mathbb{F}^{\log m}$ | Stage 1 | Stage 2 |
| `Cz` | $\widetilde{Cz}(\vec{r})$ where $Cz = (C_1 \cdot z, \ldots, C_m \cdot z)$ | $\mathbb{F}^{\log m}$ | Stage 1 | Stage 2 |
| `A_tau` | $\widetilde{A}(\tau, \vec{r}')$ - matrix MLE at fixed $\tau$ | $\mathbb{F}^{\log n}$ | Stage 2 | Stage 3 |
| `B_tau` | $\widetilde{B}(\tau, \vec{r}')$ | $\mathbb{F}^{\log n}$ | Stage 2 | Stage 3 |
| `C_tau` | $\widetilde{C}(\tau, \vec{r}')$ | $\mathbb{F}^{\log n}$ | Stage 2 | Stage 3 |

**Why these are virtual**: They're intermediate values in the Spartan proof system:

- **Stage 1**: Proves R1CS constraints satisfied → outputs claims about $Az, Bz, Cz$
- **Stage 2**: Proves those claims → outputs claims about matrix $A, B, C$ and witness $z$
- **Stage 3**: Proves matrix claims → finally reduces to committed witness polynomials

**Example entry in `virtual_openings`**:
```rust
// After Stage 1 Spartan outer sumcheck completes with challenge vector r
virtual_openings.insert(
    VirtualPolynomialId::Az,
    (
        OpeningPoint::new(vec![r_0, r_1, r_2, ..., r_{log m - 1}]),  // Random point from sumcheck
        F::from(42u64)  // Claimed value: Az(r) = 42
    )
);
```

**Mathematical meaning**:
$$\text{virtual\_openings}[\text{Az}] = (\vec{r}, 42) \quad \Leftrightarrow \quad \text{"I claim } \widetilde{Az}(\vec{r}) = 42\text{"}$$

---

#### What is `CommittedPolynomialId`?

**Definition**: An enum identifying which committed witness polynomial (from Part 2) we're claiming about.

**File**: [jolt-core/src/zkvm/witness.rs:47-80](../jolt-core/src/zkvm/witness.rs#L47)

```rust
pub enum CommittedPolynomial {
    /* R1CS auxiliary variables (from Part 2, Type 1 MLEs) */
    LeftInstructionInput,        // L̃(j) - left operand at cycle j
    RightInstructionInput,       // R̃(j) - right operand at cycle j
    WriteLookupOutputToRD,       // w̃_rd(j) - should write to rd?
    WritePCtoRD,                 // w̃_pc(j) - should write PC to rd?
    ShouldBranch,                // b̃(j) - is this a branch?
    ShouldJump,                  // j̃(j) - is this a jump?

    /* Twist/Shout witnesses (from Part 2) */
    RdInc,                       // \Deltã_rd(j) - register write increment
    RamInc,                      // \Deltã_ram(j) - memory write increment
    InstructionRa(usize),        // r̃a_i(j,k) - instruction lookup chunk i (0-15)
    BytecodeRa(usize),           // b̃c_i(j,k) - bytecode lookup chunk i
    RamRa(usize),                // m̃em_i(j,k) - RAM address chunk i
}
```

**Connection to Part 2**: These are EXACTLY the polynomials we created:

| Enum Variant | Math Object (from Part 2) | Commitment (from Part 2) | Size |
|--------------|---------------------------|--------------------------|------|
| `LeftInstructionInput` | $\widetilde{L}(j)$ | $C_L \in \mathbb{G}_T$ | T coefficients |
| `RightInstructionInput` | $\widetilde{R}(j)$ | $C_R \in \mathbb{G}_T$ | T coefficients |
| `RdInc` | $\widetilde{\Delta}_{\text{rd}}(j)$ | $C_{\Delta_{rd}} \in \mathbb{G}_T$ | T coefficients |
| `InstructionRa(0)` | $\widetilde{\text{ra}}_0(j,k)$ | $C_{ra_0} \in \mathbb{G}_T$ | T×256 coefficients |
| `InstructionRa(1)` | $\widetilde{\text{ra}}_1(j,k)$ | $C_{ra_1} \in \mathbb{G}_T$ | T×256 coefficients |
| ... | ... | ... | ... |

**Example entry in `committed_openings`**:
```rust
// After some sumcheck completes with challenge vector r'
committed_openings.insert(
    CommittedPolynomial::LeftInstructionInput,
    (
        OpeningPoint::new(vec![r'_0, r'_1, ..., r'_{log T - 1}]),  // Random point
        F::from(100u64)  // Claimed value: L̃(r') = 100
    )
);
```

**Mathematical meaning**:
$$\text{committed\_openings}[\text{LeftInstructionInput}] = (\vec{r}', 100)$$
$$\Leftrightarrow \quad \text{"I claim } \widetilde{L}(\vec{r}') = 100\text{, and I committed to } \widetilde{L} \text{ as } C_L \in \mathbb{G}_T\text{"}$$

---

#### What is `OpeningPoint`?

**Definition**: A wrapper around a vector of field elements representing a point in $\mathbb{F}^n$.

```rust
pub struct OpeningPoint<F: JoltField>(pub Vec<F>);
```

**Mathematical meaning**:
$$\text{OpeningPoint}(\vec{r}) = (r_0, r_1, \ldots, r_{n-1}) \in \mathbb{F}^n$$

**Why it exists**:

- Sumchecks reduce claims over $2^n$ points to claims about 1 random point
- That random point is generated from verifier challenges $r_0, \ldots, r_{n-1}$
- `OpeningPoint` stores that specific evaluation point

**Example**:
```rust
// After 10-round sumcheck with challenges r_0, ..., r_9
let point = OpeningPoint::new(vec![
    F::from(12345),  // r_0 (first verifier challenge)
    F::from(67890),  // r_1 (second challenge)
    // ... 8 more challenges
]);

// This represents the point r = (12345, 67890, ...) \in F^10
```

---

#### Complete Example: Claim Lifecycle

**Stage 1**: Spartan outer sumcheck proves R1CS constraints

```rust
// Input: Claim that ∑_{x\in{0,1}^10} eq(\tau,x)·(Az(x)·Bz(x) - Cz(x)) = 0

// After 10 rounds with challenges r_0, ..., r_9:
// Output claims:
accumulator.virtual_openings.insert(
    VirtualPolynomialId::Az,
    (OpeningPoint(vec![r_0, ..., r_9]), F::from(42))
);
// Meaning: "Az(r) = 42" where r = (r_0, ..., r_9)

accumulator.virtual_openings.insert(
    VirtualPolynomialId::Bz,
    (OpeningPoint(vec![r_0, ..., r_9]), F::from(100))
);
// Meaning: "Bz(r) = 100"

accumulator.virtual_openings.insert(
    VirtualPolynomialId::Cz,
    (OpeningPoint(vec![r_0, ..., r_9]), F::from(4200))
);
// Meaning: "Cz(r) = 4200"
```

**Stage 2**: Spartan product sumcheck proves $Az$ claim

```rust
// Input: Claim from Stage 1 that Az(r) = 42

// Recall: Az(r) = ∑_{x\in{0,1}^n} Ã(r,x)·z̃(x)
// Run sumcheck to prove this sum equals 42

// After n rounds with new challenges r'_0, ..., r'_{n-1}:
// Output claims:

// Virtual polynomial (matrix A at fixed r)
accumulator.virtual_openings.insert(
    VirtualPolynomialId::A_tau,
    (OpeningPoint(vec![r'_0, ..., r'_{n-1}]), F::from(3))
);
// Meaning: "Ã(r, r') = 3"

// Committed polynomial (witness z - THIS IS FROM PART 2!)
accumulator.committed_openings.insert(
    CommittedPolynomial::WitnessZ,
    (OpeningPoint(vec![r'_0, ..., r'_{n-1}]), F::from(14))
);
// Meaning: "z̃(r') = 14, and I have commitment C_z \in G_T from Part 2"
```

**Stage 5**: Dory batched opening proves all committed claims

```rust
// Input: ALL committed_openings from Stages 1-4

// Accumulated claims (example):
// committed_openings = {
//     LeftInstructionInput     → (r_1, 100),
//     RightInstructionInput    → (r_2, 250),
//     RdInc                    → (r_3, 350),
//     InstructionRa(0)         → (r_4, 1),
//     ...
//     WitnessZ                 → (r', 14),
// }

// Dory batched opening proves ALL ~50 claims together:
// "For commitment C_L, L̃(r_1) = 100"
// "For commitment C_R, R̃(r_2) = 250"
// "For commitment C_\Delta, \Deltã_rd(r_3) = 350"
// ...

// Output: Single Dory opening proof (~6 KB) proving all 50 claims!
```

---

#### Why This Design?

**Problem**: After each sumcheck, we have claims about polynomial evaluations at random points.

**Naive approach**: Open each polynomial immediately with separate Dory proof.

- **Cost**: 36 polynomials × 6 KB = 216 KB of proofs!

**Jolt's approach**: Accumulate all claims, prove together in Stage 5.

- **Cost**: Single batched proof ≈ 6 KB total
- **Savings**: 36× proof size reduction!

**The accumulator** enables this batching by:

1. Tracking which polynomials need opening
2. Tracking what points they need to be opened at
3. Tracking what values are claimed
4. Enabling Stage 5 to batch all openings together

---

#### Summary: The Accumulator as a Ledger

Think of the opening accumulator as a ledger with two columns:

**Virtual Claims Column** (proven by subsequent sumchecks):
```
| Polynomial | Point r      | Value | Will prove in |
|------------|--------------|-------|---------------|
| Az         | (r_0,...,r_9)  | 42    | Stage 2       |
| Bz         | (r_0,...,r_9)  | 100   | Stage 2       |
| A_tau      | (r'_0,...,r'_n)| 3     | Stage 3       |
```

**Committed Claims Column** (proven by Dory in Stage 5):
```
| Polynomial          | Commitment C\inG_T | Point r       | Value |
|---------------------|------------------|----------------|-------|
| LeftInstructionInput| C_L (from Part 2)| (r_1,...,r_log T)| 100   |
| RightInstructionInput| C_R (from Part 2)| (r_2,...,r_log T)| 250   |
| RdInc               | C_\Delta (from Part 2)| (r_3,...,r_log T)| 350   |
| InstructionRa(0)    | C_ra_0 (from Part 2)| (r_4,...,r_log T+8)| 1 |
```

**By Stage 5**:

- Virtual claims column is empty (all proven)
- Committed claims column has ~50 entries (all get batched opening proof)

### Fiat-Shamir Preamble

**File**: [jolt-core/src/zkvm/dag/state_manager.rs:295](../jolt-core/src/zkvm/dag/state_manager.rs#L295)

```rust
state_manager.fiat_shamir_preamble();
```

**What happens**: Append public inputs to transcript:

```rust
transcript.append_u64(program_io.memory_layout.max_input_size);
transcript.append_u64(program_io.memory_layout.max_output_size);
transcript.append_u64(program_io.memory_layout.memory_size);
transcript.append_bytes(&program_io.inputs);
transcript.append_bytes(&program_io.outputs);
transcript.append_u64(program_io.panic as u64);
transcript.append_u64(ram_K as u64);
transcript.append_u64(trace_length as u64);
```

**Mathematical meaning**:

Initialize Fiat-Shamir oracle $\mathcal{O}$ with public statement:

$$\mathcal{O} \leftarrow H(\text{inputs} \,\|\, \text{outputs} \,\|\, \text{memory\_layout} \,\|\, T \,\|\, K)$$

Where:

- $H$: Cryptographic hash function (e.g., SHA-256, Poseidon)
- $\|$: Concatenation
- $T$: Trace length
- $K$: RAM size parameter

**All subsequent challenges** derived via:
$$r_i \leftarrow H(\mathcal{O}_{\text{state}} \,\|\, \text{prover\_message}_i)$$

This binds the proof to the specific execution being proven.

---

## Polynomial Generation and Commitment

**File**: [jolt-core/src/zkvm/dag/jolt_dag.rs:85](../jolt-core/src/zkvm/dag/jolt_dag.rs#L85)

```rust
let opening_proof_hints = Self::generate_and_commit_polynomials(&mut state_manager)?;
```

### Step 1: Generate Witness Polynomials

**File**: [jolt-core/src/zkvm/witness.rs](../jolt-core/src/zkvm/witness.rs) → `CommittedPolynomial::generate_witness_batch()`

**What happens**: For each committed polynomial type, construct MLE from execution trace.

**Example: Register Increment Polynomial** $\widetilde{\Delta}_{\text{rd}}$

**Raw witness data** (from trace):
$$\Delta = (\delta_0, \delta_1, \ldots, \delta_{T-1})$$

Where $\delta_j$ is the increment written to destination register at cycle $j$:

- For `ADD r3, r1, r2` at cycle 5: $\delta_5 = \text{value written to r3}$
- For non-register-writing instructions: $\delta_j = 0$

**Multilinear Extension** (see Part 2 for details):

$$\widetilde{\Delta}_{\text{rd}}(X_1, \ldots, X_{\log T}) = \sum_{j \in \{0,1\}^{\log T}} \delta_j \cdot \text{eq}(\vec{X}, j)$$

Where $\text{eq}(\vec{X}, j)$ is the Lagrange basis polynomial:

$$\text{eq}(\vec{X}, j) = \prod_{i=1}^{\log T} (X_i j_i + (1 - X_i)(1 - j_i))$$

**Result**: A coefficient vector of size $T$ representing the MLE:
$$\widetilde{\Delta}_{\text{rd}} \in \mathbb{F}^T$$

**All 35 witness polynomials generated this way** (from Part 2):

- Simple vectors (8): $\widetilde{L}, \widetilde{R}, \widetilde{\Delta}_{\text{rd}}, \widetilde{\Delta}_{\text{ram}}, \widetilde{w}_{\text{rd}}, \widetilde{w}_{\text{pc}}, \widetilde{b}, \widetilde{j}$
- Instruction lookups (16): $\widetilde{\text{ra}}_0, \ldots, \widetilde{\text{ra}}_{15}$ (16 chunks)
- Bytecode lookups (3): $\widetilde{\text{bc}}_0, \widetilde{\text{bc}}_1, \widetilde{\text{bc}}_2$
- RAM addresses (8): $\widetilde{\text{mem}}_0, \ldots, \widetilde{\text{mem}}_7$ (8 chunks)

### Step 2: Commit to All Polynomials

**File**: [jolt-core/src/poly/commitment/dory.rs](../jolt-core/src/poly/commitment/dory.rs) → `PCS::batch_commit()`

**Mathematical operation**: For each polynomial $\widetilde{P} \in \mathbb{F}^N$ (where $N = 2^n$):

#### Reshape to Matrix

Dory operates on matrices. Reshape coefficient vector to $\sqrt{N} \times \sqrt{N}$ matrix $M$:

$$M = \begin{bmatrix}
\widetilde{P}[0] & \widetilde{P}[1] & \cdots & \widetilde{P}[\sqrt{N}-1] \\
\widetilde{P}[\sqrt{N}] & \widetilde{P}[\sqrt{N}+1] & \cdots & \widetilde{P}[2\sqrt{N}-1] \\
\vdots & \vdots & \ddots & \vdots \\
\widetilde{P}[N-\sqrt{N}] & \cdots & \cdots & \widetilde{P}[N-1]
\end{bmatrix} \in \mathbb{F}^{\sqrt{N} \times \sqrt{N}}$$

#### Layer 1: Pedersen Commitments to Rows

For each row $i = 1, \ldots, \sqrt{N}$, compute Pedersen commitment:

$$V_i = \langle \vec{M}_i, \vec{\Gamma}_1 \rangle + r_i H_1 = \sum_{j=1}^{\sqrt{N}} M_{i,j} G_{1,j} + r_i H_1 \in \mathbb{G}_1$$

Where:

- $\vec{\Gamma}_1 = (G_{1,1}, \ldots, G_{1,\sqrt{N}}) \in \mathbb{G}_1^{\sqrt{N}}$: SRS generators (from preprocessing)
- $r_i \in \mathbb{F}$: Random blinding factor
- $H_1 \in \mathbb{G}_1$: Blinding generator (from SRS)

**Result**: Vector of commitments $\vec{V} = (V_1, \ldots, V_{\sqrt{N}}) \in \mathbb{G}_1^{\sqrt{N}}$

#### Layer 2: AFGHO Commitment

Commit to the vector $\vec{V}$ using bilinear pairing:

$$C_P = \langle \vec{V}, \vec{\Gamma}_2 \rangle \cdot e(H_1, H_2)^{r_{\text{fin}}} = \left( \prod_{i=1}^{\sqrt{N}} e(V_i, G_{2,i}) \right) \cdot e(H_1, H_2)^{r_{\text{fin}}} \in \mathbb{G}_T$$

Where:

- $\vec{\Gamma}_2 = (G_{2,1}, \ldots, G_{2,\sqrt{N}}) \in \mathbb{G}_2^{\sqrt{N}}$: SRS generators
- $e: \mathbb{G}_1 \times \mathbb{G}_2 \to \mathbb{G}_T$: Bilinear pairing
- $r_{\text{fin}} \in \mathbb{F}$: Final blinding factor

**Result**: Single commitment $C_P \in \mathbb{G}_T$ (192 bytes)

### Step 3: Generate Opening Hints

**Opening hints** are auxiliary data needed for efficient opening proof later:

```rust
pub struct OpeningProofHints {
    pub layer1_hints: Vec<G1Affine>,  // Pedersen commitment randomness
    pub layer2_hint: Scalar,          // Final randomness
}
```

These are stored and used in Stage 5 when constructing the batched opening proof.

### Step 4: Append Commitments to Transcript

```rust
for commitment in commitments.iter() {
    transcript.append_serializable(commitment);
}
```

**Mathematical meaning**: Update Fiat-Shamir oracle:

$$\mathcal{O} \leftarrow H(\mathcal{O}_{\text{state}} \,\|\, C_1 \,\|\, C_2 \,\|\, \cdots \,\|\, C_m)$$

This ensures subsequent challenges depend on all commitments (binding property).

### Summary: What We Have After This Phase

| What | Mathematical Object | Size (for T=1024 trace) |
|------|---------------------|-------------------------|
| Witness polynomials | $\widetilde{P}_1, \ldots, \widetilde{P}_m \in \mathbb{F}^N$ | ~50 × 8KB = 400 KB |
| Commitments | $C_1, \ldots, C_m \in \mathbb{G}_T$ | ~50 × 192 bytes ≈ 10 KB |
| Opening hints | Blinding factors for later | ~50 × 64 bytes ≈ 3 KB |
| Updated transcript | Hash state $\mathcal{O}$ | 32 bytes |

**Compression achieved**: 400 KB witness → 10 KB commitments (40× compression!)

---

## Stage 1: Spartan Outer Sumcheck

**File**: [jolt-core/src/zkvm/spartan/mod.rs](../jolt-core/src/zkvm/spartan/mod.rs) → `SpartanDag::stage1_prove()`

### What is Spartan?

Spartan is a transparent SNARK for R1CS (Rank-1 Constraint System). R1CS expresses computation as:

$$Az \circ Bz = Cz$$

Where:

- $A, B, C \in \mathbb{F}^{m \times n}$: Constraint matrices
- $z \in \mathbb{F}^n$: Witness vector (execution trace flattened)
- $\circ$: Hadamard (element-wise) product
- $m$: Number of constraints
- $n$: Witness size

**Expanded form** (for each row $i = 1, \ldots, m$):

$$(A_i \cdot z) \cdot (B_i \cdot z) = C_i \cdot z$$

This says: "The dot product of $i$-th row of $A$ with $z$, times the dot product of $i$-th row of $B$ with $z$, equals the dot product of $i$-th row of $C$ with $z$."

### Jolt's R1CS Constraints

**File**: [jolt-core/src/zkvm/r1cs/constraints.rs](../jolt-core/src/zkvm/r1cs/constraints.rs)

Jolt has **~30 constraints per cycle** (uniform across all cycles). Examples:

**1. PC Update Constraint** (normal increment):

$$\text{PC}_{\text{next}} - \text{PC}_{\text{current}} - 4 = 0$$

If instruction is not a jump/branch. This ensures PC increments by 4 bytes.

**2. Component Linking** (load instruction):

For `LW rd, offset(rs1)` (load word):

- RAM component reads value $v$ from address $\text{addr}$
- Register component writes value $v$ to register $\text{rd}$
- **Constraint**: $v_{\text{RAM}} - v_{\text{register}} = 0$

This ensures consistency between independent components.

**3. Arithmetic Operations**:

For field-native operations (most 64-bit arithmetic):

- Constraint: $\text{rd} - (\text{rs1} + \text{rs2}) = 0$ (for ADD)
- Jolt's field is large enough for 64-bit ops without overflow

**Why R1CS for this?** These are simple algebraic relationships. R1CS is perfect for linear/quadratic constraints. Using lookups would be overkill.

### The Spartan Outer Sumcheck

**Goal**: Prove that all R1CS constraints are satisfied.

**Claim**: For random challenge $\tau \in \mathbb{F}^{\log m}$ (sampled from transcript):

$$\sum_{x \in \{0,1\}^{\log m}} \text{eq}(\tau, x) \cdot \left( \widetilde{Az}(x) \cdot \widetilde{Bz}(x) - \widetilde{Cz}(x) \right) = 0$$

**Why this proves correctness**:

- If $Az \circ Bz = Cz$ holds for all rows, the sum is exactly zero
- Random $\tau$ ensures cheating detected with high probability
- $\text{eq}(\tau, x)$ provides random linear combination of all constraints

**Mathematical objects**:

- $\widetilde{Az}$: MLE of vector $(A_1 \cdot z, A_2 \cdot z, \ldots, A_m \cdot z) \in \mathbb{F}^m$
- $\widetilde{Bz}$: MLE of vector $(B_1 \cdot z, B_2 \cdot z, \ldots, B_m \cdot z) \in \mathbb{F}^m$
- $\widetilde{Cz}$: MLE of vector $(C_1 \cdot z, C_2 \cdot z, \ldots, C_m \cdot z) \in \mathbb{F}^m$

### The Protocol

**File**: [jolt-core/src/subprotocols/sumcheck.rs](../jolt-core/src/subprotocols/sumcheck.rs)

**Round $j = 0, \ldots, \log m - 1$**:

**Prover**:

1. Compute univariate polynomial $g_j(X_j)$:
   $$g_j(X_j) = \sum_{x_{j+1}, \ldots, x_{\log m} \in \{0,1\}^{\log m - j - 1}} \text{eq}(\tau, r_0, \ldots, r_{j-1}, X_j, x_{j+1}, \ldots) \cdot (\widetilde{Az} \cdot \widetilde{Bz} - \widetilde{Cz})$$

2. Evaluate at points $0, 2, 3, \ldots, d$ (where $d$ is degree):
   - For Spartan: $d = 3$ (product of two degree-1 MLEs plus another)

3. Compress to coefficient form and append to transcript

**Verifier**:

1. Sample random challenge $r_j \in \mathbb{F}$ from transcript
2. Check consistency: $g_j(0) + g_j(1) \stackrel{?}{=} H_j$ (where $H_j$ is previous round's claim)

**After all rounds**: Claim reduced to:

$$g(r_0, \ldots, r_{\log m - 1}) \stackrel{?}{=} \text{eq}(\tau, \vec{r}) \cdot \left( \widetilde{Az}(\vec{r}) \cdot \widetilde{Bz}(\vec{r}) - \widetilde{Cz}(\vec{r}) \right)$$

This generates **three output claims**:

- $\widetilde{Az}(\vec{r}) = v_A$ (virtual polynomial, proven in Stage 2)
- $\widetilde{Bz}(\vec{r}) = v_B$ (virtual polynomial, proven in Stage 2)
- $\widetilde{Cz}(\vec{r}) = v_C$ (virtual polynomial, proven in Stage 2)

### Adding Claims to Accumulator

**File**: [jolt-core/src/poly/opening_proof.rs](../jolt-core/src/poly/opening_proof.rs)

```rust
accumulator.add_virtual_opening(
    VirtualPolynomial::Az,
    r_vec.clone(),
    claimed_value_A,
);
accumulator.add_virtual_opening(
    VirtualPolynomial::Bz,
    r_vec.clone(),
    claimed_value_B,
);
accumulator.add_virtual_opening(
    VirtualPolynomial::Cz,
    r_vec.clone(),
    claimed_value_C,
);
```

**Mathematical meaning**:

Opening accumulator now contains:
$$\{ (\text{VirtualPoly::Az}, \vec{r}, v_A), (\text{VirtualPoly::Bz}, \vec{r}, v_B), (\text{VirtualPoly::Cz}, \vec{r}, v_C) \}$$

These are **virtual** because:

- $Az$, $Bz$, $Cz$ are NOT directly committed polynomials
- They are computed from committed witness polynomials via matrix multiplication
- Will be proven in Stage 2 via product sumchecks

### Stage 1 Output

**Stored in StateManager**:

```rust
state_manager.proofs.insert(
    ProofKeys::Stage1Sumcheck,
    ProofData::SumcheckProof(stage1_proof)
);
```

**What's in the proof**:

$$\text{Stage1Proof} = \{ g_0(0), g_0(2), \ldots, g_0(d), \; g_1(0), g_1(2), \ldots, g_1(d), \; \ldots \}$$

**Proof size**:

- Number of rounds: $\log m$ (where $m$ = number of constraints)
- Coefficients per round: $d + 1 = 4$
- Bytes per coefficient: 32 (field element)
- **Total**: $\log m \times 4 \times 32$ bytes ≈ 512 bytes (for $m = 2^{10}$ constraints)

---

## Stage 2: Batched Sumchecks

**File**: [jolt-core/src/zkvm/dag/jolt_dag.rs:137](../jolt-core/src/zkvm/dag/jolt_dag.rs#L137)

Stage 2 contains sumchecks from four components:

1. **Spartan** (Product sumchecks) - prove $Az$, $Bz$, $Cz$ claims
2. **Registers** (Twist checking) - prove register reads/writes consistent
3. **RAM** (Twist checking) - prove memory reads/writes consistent
4. **Lookups** (Booleanity) - prove lookup address polynomials are Boolean

### Batching Mechanism

**Code**:
```rust
let mut stage2_instances: Vec<_> = std::iter::empty()
    .chain(spartan_dag.stage2_prover_instances(&mut state_manager))
    .chain(registers_dag.stage2_prover_instances(&mut state_manager))
    .chain(ram_dag.stage2_prover_instances(&mut state_manager))
    .chain(lookups_dag.stage2_prover_instances(&mut state_manager))
    .collect();

let (stage2_proof, r_stage2) = BatchedSumcheck::prove(
    stage2_instances_mut,
    Some(accumulator.clone()),
    &mut *transcript.borrow_mut(),
);
```

**Mathematical operation**:

1. **Sample batching coefficients** $\alpha_1, \ldots, \alpha_k$ from transcript
2. **Combine claims**:
   $$H_{\text{combined}} = \alpha_1 H_1 + \alpha_2 H_2 + \cdots + \alpha_k H_k$$
3. **Define combined polynomial**:
   $$g_{\text{combined}}(x) = \alpha_1 g_1(x) + \alpha_2 g_2(x) + \cdots + \alpha_k g_k(x)$$
4. **Run single sumcheck** on $g_{\text{combined}}$
5. **Each round**: Prover computes:
   $$g_{\text{combined},j}(X_j) = \alpha_1 g_{1,j}(X_j) + \alpha_2 g_{2,j}(X_j) + \cdots + \alpha_k g_{k,j}(X_j)$$

**Key insight**: All instances use *same random challenges* $r_0, r_1, \ldots, r_n$ from verifier.

### Example Component: Spartan Product Sumchecks

**File**: [jolt-core/src/zkvm/spartan/product.rs](../jolt-core/src/zkvm/spartan/product.rs)

**Goal**: Prove the virtual polynomial claims from Stage 1.

Recall from Stage 1 output:

- Claim: $\widetilde{Az}(\vec{r}) = v_A$

**What is $\widetilde{Az}$?** It's the MLE of the vector $(A_1 \cdot z, A_2 \cdot z, \ldots, A_m \cdot z)$.

**Key observation**: $Az$ can be expressed as:

$$\widetilde{Az}(\vec{r}) = \sum_{x \in \{0,1\}^{\log n}} \widetilde{A}(\vec{r}, x) \cdot \widetilde{z}(x)$$

Where:

- $\widetilde{A}$: MLE of matrix $A \in \mathbb{F}^{m \times n}$ (treated as 2D function)
- $\widetilde{z}$: MLE of witness vector $z \in \mathbb{F}^n$

**Sumcheck claim**:

$$\sum_{x \in \{0,1\}^{\log n}} \widetilde{A}(\vec{r}, x) \cdot \widetilde{z}(x) \stackrel{?}{=} v_A$$

**After sumcheck completes** with challenges $\vec{r}'$:

- Claims about $\widetilde{A}(\vec{r}, \vec{r}')$ (virtual, proven in Stage 3)
- Claims about $\widetilde{z}(\vec{r}')$ (committed witness polynomial!)

**Critical transition**: Virtual polynomial $Az$ → Committed polynomial $z$

The claim about $\widetilde{z}(\vec{r}')$ is added to `committed_openings` in the accumulator:

```rust
accumulator.add_committed_opening(
    CommittedPolynomial::WitnessZ,
    r_prime_vec.clone(),
    z_at_r_prime,
);
```

**Similar product sumchecks** for $Bz$ and $Cz$, each generating committed polynomial claims.

### Example Component: Registers Twist Checking

**File**: [jolt-core/src/zkvm/registers/mod.rs](../jolt-core/src/zkvm/registers/mod.rs)

**Goal**: Prove register reads return last written value.

From Twist paper:

**Memory checking via grand product argument**:

Two traces:

- **Time-ordered**: Registers accessed in execution order
- **Address-ordered**: Same accesses sorted by register number

**Claim**: Prove they're permutations via:

$$\prod_{i=0}^{T-1} f_{\text{time}}(i) = \prod_{j=0}^{T-1} f_{\text{addr}}(j)$$

Where $f$ is a fingerprint function:

$$f(i) = \gamma_1 \cdot \text{address}(i) + \gamma_2 \cdot \text{timestamp}(i) + \gamma_3 \cdot \text{value}(i)$$

For random challenges $\gamma_1, \gamma_2, \gamma_3$ from transcript.

**Twist optimization** (used in Jolt v0.2.0+):

Instead of grand product, use **incremental approach**:

- Maintain running product at each step
- Verify final product equals 1
- More efficient than computing full product

**Three sumchecks** (one for each register access type):

1. **Read-checking for rs1** (source register 1)
2. **Read-checking for rs2** (source register 2)
3. **Write-checking for rd** (destination register)

**Read-checking sumcheck** (for rs1):

$$\sum_{j \in \{0,1\}^{\log T}} \sum_{k \in \{0,1\}^{\log K}} \widetilde{\text{ra}}_{\text{rs1}}(j, k) \cdot \left( f_{\text{read}}(j, k) - f_{\text{written}}(j, k) \right) \stackrel{?}{=} 0$$

Where:

- $\widetilde{\text{ra}}_{\text{rs1}}(j, k)$: One-hot polynomial (1 if cycle $j$ reads from register $k$)
- $f_{\text{read}}$: Fingerprint of read operation
- $f_{\text{written}}$: Fingerprint of last write to that register
- $K = 64$: Number of registers (32 RISC-V + 32 virtual)

**Output claims**: After sumcheck, claims about:

- $\widetilde{\text{ra}}_{\text{rs1}}(\vec{r})$ (committed)
- $\widetilde{\Delta}_{\text{rd}}(\vec{r})$ (committed - register increment polynomial)

Added to `committed_openings` in accumulator.

### Stage 2 Output

**Mathematical objects created**:

1. **Batched proof** $\pi_2$:
   - Single sumcheck proof over combined polynomial
   - Size: $\sim n \times 4 \times 32$ bytes (where $n = \log(T)$)

2. **Virtual polynomial claims** (for Stage 3):
   - Matrix MLEs: $\widetilde{A}(\vec{r}, \vec{r}')$, $\widetilde{B}(\vec{r}, \vec{r}')$, $\widetilde{C}(\vec{r}, \vec{r}')$

3. **Committed polynomial claims** (for Stage 5):
   - Witness: $\widetilde{z}(\vec{r}')$
   - Registers: $\widetilde{\Delta}_{\text{rd}}(\vec{r}'')$, $\widetilde{\text{ra}}_{\text{rs1}}(\vec{r}'')$
   - RAM: Various increment and address polynomials

**Stored in StateManager**:
```rust
state_manager.proofs.insert(
    ProofKeys::Stage2Sumcheck,
    ProofData::SumcheckProof(stage2_proof)
);
```

---

## Stage 3: More Batched Sumchecks

**File**: [jolt-core/src/zkvm/dag/jolt_dag.rs:194](../jolt-core/src/zkvm/dag/jolt_dag.rs#L194)

Components contributing:

1. **Spartan** (Inner sumchecks - proving matrix MLE claims)
2. **Registers** (Hamming weight, evaluation)
3. **Lookups** (Read-checking, Hamming weight)
4. **RAM** (Hamming weight, evaluation)

### Example Component: Instruction Lookups

**File**: [jolt-core/src/zkvm/instruction_lookups/mod.rs](../jolt-core/src/zkvm/instruction_lookups/mod.rs)

**Goal**: Prove every instruction execution produced correct outputs.


> **Heuristic: The Ultimate Cheat Sheet**
>
> Instead of learning how to add, you create a giant "cheat sheet" (a lookup table) with all possible additions pre-computed. To "prove" you can add 120 + 55, you simply point to the entry and show the pre-computed result.

**The Challenge**: A 64-bit ADD instruction has two 64-bit inputs:

- Lookup table size: $2^{64} \times 2^{64} = 2^{128}$ entries
- **Impossibly large** to store or commit to!

**Jolt's Solution: Decomposition**

Break 64-bit operands into 16 chunks of 4 bits each:

$$a = a_0 \| a_1 \| \cdots \| a_{15} \quad \text{(each } a_i \in \{0,1\}^4\text{)}$$
$$b = b_0 \| b_1 \| \cdots \| b_{15} \quad \text{(each } b_i \in \{0,1\}^4\text{)}$$

**Small lookup tables**: For each 4-bit chunk:

- Input: $(a_i, b_i) \in \{0,1\}^8$
- Table size: $2^8 = 256$ entries
- **Manageable!**

**Example: 4-bit ADD table** (first 16 entries shown):

| $a_i$ | $b_i$ | Sum | Carry |
|-------|-------|-----|-------|
| 0000 | 0000 | 0000 | 0 |
| 0000 | 0001 | 0001 | 0 |
| 0000 | 0010 | 0010 | 0 |
| ... | ... | ... | ... |
| 1111 | 1111 | 1110 | 1 |

**Prefix-Suffix Sumcheck**

From "Proving CPU Executions in Small Space" paper:

To prove $T$ lookups into a table of size $N = 2^{128}$:

**Standard Shout** would require sumcheck over $\log(T) + \log(N) = \log(T) + 128$ variables.

**Prefix-Suffix optimization**: Split into prefix (left 64 bits) and suffix (right 64 bits):

$$\text{Val}(k_{\text{prefix}}, k_{\text{suffix}}) = \sum_{j} \text{prefix}(j, k_{\text{prefix}}) \cdot \text{suffix}(j, k_{\text{suffix}})$$

**Two separate sumchecks**:

1. Over prefix (64 variables)
2. Over suffix (64 variables)

**Total**: $\log(T) + 64 + 64 = \log(T) + 128$ rounds (same), but:

- **Memory**: $O(T \cdot 2^{64})$ instead of $O(T \cdot 2^{128})$
- **Tractable** for prover!

### Read-Checking Sumcheck

**File**: [jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs](../jolt-core/src/zkvm/instruction_lookups/read_raf_checking.rs)

**Claim**: For each of 16 chunks, prove lookups correct.

For chunk $i$:

$$\sum_{j \in \{0,1\}^{\log T}} \sum_{k \in \{0,1\}^8} \widetilde{\text{ra}}_i(j, k) \cdot \left( \text{read\_value}(k) - \text{table}(k) \right) \stackrel{?}{=} 0$$

Where:

- $\widetilde{\text{ra}}_i(j, k)$: One-hot polynomial (1 if cycle $j$ looks up entry $k$ in table $i$)
- $\text{table}(k)$: Pre-computed lookup table value at index $k$

**Key**: $\text{table}(k)$ is efficiently evaluable! 

For ADD table:
$$\text{ADD\_table}(k) = k_{\text{left}} + k_{\text{right}} \pmod{2^4}$$

Where $k = k_{\text{left}} \| k_{\text{right}}$ (8-bit index split into two 4-bit parts).

**No need to store** $2^8$ table entries - compute on the fly during sumcheck!

### Hamming Weight Sumcheck

Part of Shout protocol. Proves the one-hot polynomial $\widetilde{\text{ra}}_i(j, k)$ has correct properties:

- Exactly one "1" per row $j$ (each cycle looks up exactly one table entry)
- All other entries are "0"

**Claim**:

$$\sum_{j,k} \widetilde{\text{ra}}_i(j, k) \stackrel{?}{=} T$$

(Total weight equals number of lookups)

**Combined with multiset checking** to prove correct lookup distribution.

### Stage 3 Output

After batched sumcheck:

1. **More committed polynomial claims**:
   - Instruction polynomials: $\widetilde{L}(\vec{r})$, $\widetilde{R}(\vec{r})$ (left/right operands)
   - Lookup address polynomials: $\widetilde{\text{ra}}_0(\vec{r}), \ldots, \widetilde{\text{ra}}_{15}(\vec{r})$

2. **Virtual polynomial claims** resolved from Stage 2

3. **Batched proof** $\pi_3$ appended to StateManager

---

## Stage 4: Final Sumchecks

**File**: [jolt-core/src/zkvm/dag/jolt_dag.rs:248](../jolt-core/src/zkvm/dag/jolt_dag.rs#L248)

Components contributing:

1. **RAM** (Ra virtualization, evaluation)
2. **Bytecode** (Read-checking)
3. **Lookups** (Ra virtualization)

### Example Component: Bytecode Read-Checking

**File**: [jolt-core/src/zkvm/bytecode/mod.rs](../jolt-core/src/zkvm/bytecode/mod.rs)

**Goal**: Prove trace instructions match committed bytecode.

**Setup (from preprocessing)**:

- Bytecode decoded and committed: $C_{\text{bytecode}} \in \mathbb{G}_T$
- This commits to $K$ instructions (where $K$ = program size)

**During execution**:

- Trace records $T$ instruction fetches (where $T$ = trace length)
- Each fetch reads from specific bytecode address (PC value)

**Claim**: Every instruction in trace matches corresponding committed bytecode instruction.

**Shout Offline Memory Checking**:

Similar to Twist, but for read-only memory (no writes):

$$\sum_{j \in \{0,1\}^{\log T}} \text{fingerprint}_{\text{read}}(j) \stackrel{?}{=} \sum_{k \in \{0,1\}^{\log K}} \text{count}(k) \cdot \text{fingerprint}_{\text{bytecode}}(k)$$

Where:

- $\text{fingerprint}_{\text{read}}(j)$: Fingerprint of instruction fetched at cycle $j$
- $\text{fingerprint}_{\text{bytecode}}(k)$: Fingerprint of committed instruction at address $k$
- $\text{count}(k)$: Number of times instruction $k$ was executed

**Fingerprint function**:

$$f(j) = \gamma_1 \cdot \text{PC}(j) + \gamma_2 \cdot \text{opcode}(j) + \gamma_3 \cdot \text{rs1}(j) + \cdots$$

For random $\gamma_i$ from transcript.

**Why this works**: If trace matches bytecode, the multisets of fingerprints are equal.

**Read-checking sumcheck**:

$$\sum_{j \in \{0,1\}^{\log T}} f_{\text{read}}(j) - \sum_{k \in \{0,1\}^{\log K}} \text{count}(k) \cdot f_{\text{bytecode}}(k) \stackrel{?}{=} 0$$

**Output claims**:

- Circuit flag polynomials: $\widetilde{\text{jump\_flag}}(\vec{r})$, $\widetilde{\text{load\_flag}}(\vec{r})$
- Register address polynomials: $\widetilde{\text{rs1}}(\vec{r})$, $\widetilde{\text{rs2}}(\vec{r})$, $\widetilde{\text{rd}}(\vec{r})$

All committed polynomials → added to `committed_openings`.

### Ra Virtualization

**Advanced topic**: Some components use "chunking" (parameter $d > 1$) for memory efficiency.

**Ra virtualization** proves that chunked representation matches un-chunked via sumcheck.

Reduces claims about chunked polynomials to claims about base polynomials.

### Stage 4 Output

**All virtual polynomial claims resolved!** Opening accumulator now contains:

- **Only committed polynomial claims**
- Ready for final batched opening in Stage 5

**Batched proof** $\pi_4$ stored in StateManager.

---

## Stage 5: Batched Opening Proof

**File**: [jolt-core/src/zkvm/dag/jolt_dag.rs:309](../jolt-core/src/zkvm/dag/jolt_dag.rs#L309)

**Goal**: Prove all committed polynomial evaluations claimed in Stages 1-4.

---

### Deep Dive: How the Opening Accumulator Works

**Mathematical structure of the accumulator**:

The accumulator is fundamentally a **map** from polynomial identifiers to evaluation claims:

$$\text{Accumulator}: (\text{PolynomialId}, \text{OpeningPoint}) \mapsto \text{ClaimedValue}$$

**More precisely**, for each committed polynomial $\widetilde{P}_i$ from Part 2:

$$\text{Accumulator}[\text{id}_i, \vec{r}_i] = \{v_i, \text{SumcheckId}, C_i\}$$

Where:

- $\text{id}_i \in \{\text{LeftInput}, \text{RightInput}, \ldots\}$ - Identifies which of the 35 polynomials
- $\vec{r}_i \in \mathbb{F}^{n_i}$ - Random evaluation point (from Fiat-Shamir)
- $v_i \in \mathbb{F}$ - Claimed evaluation: $\widetilde{P}_i(\vec{r}_i) \stackrel{?}{=} v_i$
- $\text{SumcheckId}$ - Which sumcheck generated this claim (for debugging/audit)
- $C_i \in \mathbb{G}_T$ - Dory commitment to $\widetilde{P}_i$ (already sent to verifier)

**Key properties**:

1. **Multiple evaluation points**: Same polynomial can appear multiple times at different points
   - Example: $\widetilde{L}(\vec{r}_1) = v_1$ from one sumcheck, $\widetilde{L}(\vec{r}_2) = v_2$ from another
   - Each gets a separate entry in the accumulator

2. **Different dimensionalities**: Evaluation points have different dimensions
   - $\widetilde{L}(\vec{r}) \in \mathbb{F}^{10}$ (Type 1: cycle-indexed)
   - $\widetilde{\text{ra}}_0(\vec{r}) \in \mathbb{F}^{18}$ (Type 2: cycle × table-indexed)
   - $\widetilde{z}(\vec{r}') \in \mathbb{F}^{23}$ (Spartan's flattened witness)

3. **Commitment reuse**: Same commitment $C_i$ appears for all evaluation points of $\widetilde{P}_i$

---

### Input: Opening Accumulator State After Stage 4

**Concrete example** (simplified for $T = 1024$ cycles):

$$
\boxed{
\begin{array}{|l|c|c|c|}
\hline
\textbf{Polynomial} & \textbf{Evaluation Point} & \textbf{Claimed Value} & \textbf{Commitment} \\
\hline
\widetilde{z} & \vec{r}' \in \mathbb{F}^{23} & v_z = 8.342 & C_z \\
\hline
\widetilde{L} & \vec{r}_L \in \mathbb{F}^{10} & v_L = 5.217 & C_L \\
\hline
\widetilde{R} & \vec{r}_R \in \mathbb{F}^{10} & v_R = 7.901 & C_R \\
\hline
\widetilde{\Delta}_{\text{rd}} & \vec{r}_{\Delta} \in \mathbb{F}^{10} & v_{\Delta} = 12.003 & C_{\Delta_{\text{rd}}} \\
\hline
\widetilde{\Delta}_{\text{ram}} & \vec{r}_{\text{ram}} \in \mathbb{F}^{10} & v_{\text{ram}} = 350.2 & C_{\Delta_{\text{ram}}} \\
\hline
\widetilde{\text{ra}}_0 & \vec{r}_{\text{ra}_0} \in \mathbb{F}^{18} & v_{\text{ra}_0} = 0.001 & C_{\text{ra}_0} \\
\widetilde{\text{ra}}_1 & \vec{r}_{\text{ra}_1} \in \mathbb{F}^{18} & v_{\text{ra}_1} = 0.000 & C_{\text{ra}_1} \\
\vdots & \vdots & \vdots & \vdots \\
\widetilde{\text{ra}}_{15} & \vec{r}_{\text{ra}_{15}} \in \mathbb{F}^{18} & v_{\text{ra}_{15}} = 1.000 & C_{\text{ra}_{15}} \\
\hline
\widetilde{\text{bc}}_0 & \vec{r}_{\text{bc}_0} \in \mathbb{F}^{18} & v_{\text{bc}_0} = 0.5 & C_{\text{bc}_0} \\
\widetilde{\text{bc}}_1 & \vec{r}_{\text{bc}_1} \in \mathbb{F}^{18} & v_{\text{bc}_1} = 0.25 & C_{\text{bc}_1} \\
\widetilde{\text{bc}}_2 & \vec{r}_{\text{bc}_2} \in \mathbb{F}^{18} & v_{\text{bc}_2} = 0.125 & C_{\text{bc}_2} \\
\hline
\widetilde{\text{mem}}_0 & \vec{r}_{\text{mem}_0} \in \mathbb{F}^{28} & v_{\text{mem}_0} = 0.01 & C_{\text{mem}_0} \\
\vdots & \vdots & \vdots & \vdots \\
\widetilde{\text{mem}}_7 & \vec{r}_{\text{mem}_7} \in \mathbb{F}^{28} & v_{\text{mem}_7} = 0.00 & C_{\text{mem}_7} \\
\hline
\end{array}
}
$$

**Total**: **36 evaluation claims**

- 1 claim for Spartan's $\widetilde{z}$
- 35 claims for Part 2's witness polynomials

**The Challenge**:

Each polynomial $\widetilde{P}_i$ is committed once, but evaluated at potentially **different random points** $\vec{r}_i$.

**Naïve approach**: 36 separate Dory opening proofs

- Cost: 36 × 6 KB = **216 KB** just for openings!
- Verifier work: 36 separate verification procedures

**Jolt's approach**: **Batch all 36 openings together** → **~6 KB total** (36× reduction!)

---

### Mathematical Challenge: Different Evaluation Points

**Why batching is non-trivial**:

Standard batching uses random linear combination:

$$\text{Claim: } \sum_{i=1}^{36} \beta_i \widetilde{P}_i(\vec{r}_i) = \sum_{i=1}^{36} \beta_i v_i$$

But the $\vec{r}_i$ are **all different**! We can't just add the polynomials because they're being evaluated at different points.

**The problem in pictures**:

```
Polynomial 1:  ■─────────────────────■ eval at r_1
Polynomial 2:       ■──────────────────────■ eval at r_2
Polynomial 3:  ■────────────────────────────────■ eval at r_3
                ↑         ↑            ↑        ↑
              Can't directly combine - different evaluation points!
```

**The solution**: Use the **equality polynomial** $\text{eq}(\cdot, \cdot)$ to "route" each polynomial to its correct evaluation point!

### Batched Opening via Random Linear Combination

**File**: [jolt-core/src/poly/opening_proof.rs](../jolt-core/src/poly/opening_proof.rs) → `reduce_and_prove()`

---

#### Step 1: Sample Batching Coefficients

**Prover action**: Append all committed claims to transcript

```rust
for (poly_id, point, value) in committed_openings {
    transcript.append_field_element(value);
    transcript.append_field_elements(&point.r);
}
```

**Both parties derive**: $\beta_1, \ldots, \beta_{36} \in \mathbb{F}$ from Fiat-Shamir

$$\beta_i = \mathcal{H}(\text{transcript} \parallel i) \quad \text{for } i = 1, \ldots, 36$$

**Security**: Random linear combination binds prover to **all** claims simultaneously

- Schwartz-Zippel: If any single $\widetilde{P}_i(\vec{r}_i) \neq v_i$, combined claim fails with probability $\geq 1 - 1/|\mathbb{F}|$

---

#### Step 2: The Reduction Polynomial (Key Innovation!)

**The core idea**: Use equality polynomial $\text{eq}(\cdot, \cdot)$ to "select" correct evaluation point for each polynomial.

**Definition of $\text{eq}$** (multilinear extension of equality):

For $\vec{x}, \vec{r} \in \mathbb{F}^n$:

$$\text{eq}(\vec{x}, \vec{r}) = \prod_{i=1}^{n} [x_i r_i + (1 - x_i)(1 - r_i)]$$

**Key properties**:

1. On Boolean hypercube: $\text{eq}(\vec{x}, \vec{r}) = \begin{cases} 1 & \text{if } \vec{x} = \vec{r} \\ 0 & \text{otherwise} \end{cases}$ for $\vec{x} \in \{0,1\}^n$
2. Multilinear in both arguments
3. Efficiently evaluable in $O(n)$ time

**The reduction polynomial**:

$$\widetilde{Q}(\vec{X}) = \sum_{i=1}^{36} \beta_i \cdot \widetilde{P}_i(\vec{X}) \cdot \text{eq}(\vec{X}, \vec{r}_i)$$

**Why this works** - intuition:

For any $\vec{x}$ on the Boolean hypercube $\{0,1\}^n$:

$$\widetilde{Q}(\vec{x}) = \sum_{i=1}^{36} \beta_i \cdot \widetilde{P}_i(\vec{x}) \cdot \underbrace{\text{eq}(\vec{x}, \vec{r}_i)}_{\text{= 1 only if } \vec{x} = \vec{r}_i}$$

The $\text{eq}$ polynomial "masks out" all terms except when $\vec{x}$ equals the specific evaluation point $\vec{r}_i$ for polynomial $i$.

**Concrete example with 3 polynomials**:

Suppose:

- $\widetilde{P}_1$ evaluated at $\vec{r}_1 = (0, 1, 0)$
- $\widetilde{P}_2$ evaluated at $\vec{r}_2 = (1, 0, 1)$
- $\widetilde{P}_3$ evaluated at $\vec{r}_3 = (1, 1, 0)$

Then:

$$\widetilde{Q}(0,1,0) = \beta_1 \widetilde{P}_1(0,1,0) \cdot 1 + \beta_2 \widetilde{P}_2(0,1,0) \cdot 0 + \beta_3 \widetilde{P}_3(0,1,0) \cdot 0 = \beta_1 \widetilde{P}_1(0,1,0)$$

$$\widetilde{Q}(1,0,1) = \beta_1 \widetilde{P}_1(1,0,1) \cdot 0 + \beta_2 \widetilde{P}_2(1,0,1) \cdot 1 + \beta_3 \widetilde{P}_3(1,0,1) \cdot 0 = \beta_2 \widetilde{P}_2(1,0,1)$$

Each polynomial "activates" only at its designated evaluation point!

---

#### Step 3: Handling Different Polynomial Dimensions

**Problem**: Our 36 polynomials have **different dimensions**!

- $\widetilde{L}: \{0,1\}^{10} \to \mathbb{F}$ (1024 cycles)
- $\widetilde{\text{ra}}_0: \{0,1\}^{18} \to \mathbb{F}$ (1024 cycles × 256 table entries)
- $\widetilde{z}: \{0,1\}^{23} \to \mathbb{F}$ (8M witness elements)

**Solution**: Pad all polynomials to maximum dimension $n_{\max}$

Let $n_{\max} = \max\{10, 18, 23\} = 23$

**Padding strategy**:

For a polynomial $\widetilde{P}_i$ of dimension $n_i < n_{\max}$, extend it to dimension $n_{\max}$ by treating extra variables as "don't care":

$$\widetilde{P}_i^{\text{padded}}(x_1, \ldots, x_{n_i}, x_{n_i+1}, \ldots, x_{n_{\max}}) = \widetilde{P}_i(x_1, \ldots, x_{n_i})$$

Similarly, pad evaluation points:

$$\vec{r}_i^{\text{padded}} = (\underbrace{r_{i,1}, \ldots, r_{i,n_i}}_{\text{original}}, \underbrace{0, \ldots, 0}_{n_{\max} - n_i \text{ zeros}})$$

**Now all polynomials live in same space**: $\{0,1\}^{23} \to \mathbb{F}$

**The padded reduction polynomial**:

$$\widetilde{Q}(\vec{X}) = \sum_{i=1}^{36} \beta_i \cdot \widetilde{P}_i^{\text{padded}}(\vec{X}) \cdot \text{eq}(\vec{X}, \vec{r}_i^{\text{padded}})$$

---

#### Step 4: Reduction Sumcheck

**Initial claim**:

$$\sum_{\vec{x} \in \{0,1\}^{23}} \widetilde{Q}(\vec{x}) \stackrel{?}{=} \sum_{i=1}^{36} \beta_i v_i$$

**Why is the right-hand side correct?**

$$\sum_{\vec{x} \in \{0,1\}^{23}} \widetilde{Q}(\vec{x}) = \sum_{\vec{x} \in \{0,1\}^{23}} \sum_{i=1}^{36} \beta_i \widetilde{P}_i^{\text{padded}}(\vec{x}) \cdot \text{eq}(\vec{x}, \vec{r}_i^{\text{padded}})$$

Swap summation order:

$$= \sum_{i=1}^{36} \beta_i \sum_{\vec{x} \in \{0,1\}^{23}} \widetilde{P}_i^{\text{padded}}(\vec{x}) \cdot \text{eq}(\vec{x}, \vec{r}_i^{\text{padded}})$$

Key observation: $\text{eq}(\vec{x}, \vec{r}_i) = 1$ only when $\vec{x} = \vec{r}_i$ on the hypercube, so:

$$= \sum_{i=1}^{36} \beta_i \widetilde{P}_i^{\text{padded}}(\vec{r}_i^{\text{padded}}) = \sum_{i=1}^{36} \beta_i v_i$$

Perfect! The claim is well-formed.

**Run standard sumcheck protocol**:

**Round 1**:

- Prover sends univariate polynomial $g_1(X_1)$ where:
  $$g_1(X_1) = \sum_{x_2, \ldots, x_{23} \in \{0,1\}^{22}} \widetilde{Q}(X_1, x_2, \ldots, x_{23})$$
- Verifier checks: $g_1(0) + g_1(1) = \sum_{i=1}^{36} \beta_i v_i$
- Verifier samples: $\rho_1 \leftarrow \mathcal{H}(\text{transcript} \parallel g_1)$
- Reduced claim: $\sum_{x_2, \ldots, x_{23}} \widetilde{Q}(\rho_1, x_2, \ldots, x_{23}) = g_1(\rho_1)$

**Rounds 2-23**: Continue binding variables $X_2, \ldots, X_{23}$ with challenges $\rho_2, \ldots, \rho_{23}$

**Final claim** (after 23 rounds):

$$\widetilde{Q}(\vec{\rho}) = q$$

Where $\vec{\rho} = (\rho_1, \ldots, \rho_{23}) \in \mathbb{F}^{23}$ and $q \in \mathbb{F}$ is the final sumcheck value.

**Prover must now prove this single evaluation claim!**

**Sumcheck proof size**: 23 rounds × 4 coefficients × 32 bytes = **2.9 KB**

---

#### Step 5: Expanding $\widetilde{Q}(\vec{\rho})$

**Recall**:

$$\widetilde{Q}(\vec{X}) = \sum_{i=1}^{36} \beta_i \cdot \widetilde{P}_i^{\text{padded}}(\vec{X}) \cdot \text{eq}(\vec{X}, \vec{r}_i^{\text{padded}})$$

**So**:

$$\widetilde{Q}(\vec{\rho}) = \sum_{i=1}^{36} \beta_i \cdot \widetilde{P}_i^{\text{padded}}(\vec{\rho}) \cdot \text{eq}(\vec{\rho}, \vec{r}_i^{\text{padded}})$$

**Key insight**: Verifier can compute $\text{eq}(\vec{\rho}, \vec{r}_i)$ for all $i$ in $O(36 \times 23)$ time!

**What prover must prove**: The **36 polynomial evaluations** $\widetilde{P}_i^{\text{padded}}(\vec{\rho})$

**But wait** - these are evaluations at a **new random point** $\vec{\rho}$, different from the original $\vec{r}_i$!

**This seems circular** - we wanted to prove evaluations at $\vec{r}_i$, now we need to prove evaluations at $\vec{\rho}$?

**The magic**: We only need **one single opening proof** at $\vec{\rho}$ for the **combined polynomial**!

### Step 6: Dory Opening Proof - The Final Step

**Now prove**: $\widetilde{Q}(\vec{\rho}) = q$ using Dory PCS.

**File**: [jolt-core/src/poly/commitment/dory.rs](../jolt-core/src/poly/commitment/dory.rs) → `prove_evaluation()`

---

#### Dory Overview: Two-Layer Commitment Scheme

**Dory** combines two commitment schemes for efficiency:

**Layer 1: Pedersen commitments** (for vectors)
$$C_V = \sum_{i=0}^{n-1} v_i G_{1,i} + r H_1 \in \mathbb{G}_1$$

**Layer 2: Pairing-based aggregation** (for polynomials)
$$C_P = e(V_0, G_{2,0}) \cdot e(V_1, G_{2,1}) \cdot \ldots \in \mathbb{G}_T$$

Where $e: \mathbb{G}_1 \times \mathbb{G}_2 \to \mathbb{G}_T$ is a bilinear pairing.

**Key advantage**: Logarithmic opening proof size via recursive halving

---

#### The Dory Opening Protocol (Simplified)

**Input**:

- Polynomial $\widetilde{Q}$ represented as coefficient vector $\vec{q} = (q_0, q_1, \ldots, q_{N-1})$ where $N = 2^{23}$
- Evaluation point $\vec{\rho} = (\rho_1, \ldots, \rho_{23}) \in \mathbb{F}^{23}$
- Claimed value: $\widetilde{Q}(\vec{\rho}) = q$
- Commitment $C_Q \in \mathbb{G}_T$ (already computed)

**Goal**: Convince verifier that $\widetilde{Q}(\vec{\rho}) = q$ without sending entire polynomial (8M elements!)

**Approach**: Recursive halving - reduce polynomial size by half in each round

---

**Round 1: Split on first variable $\rho_1$**

Write $\widetilde{Q}$ as:

$$\widetilde{Q}(X_1, X_2, \ldots, X_{23}) = \widetilde{Q}_L(X_2, \ldots, X_{23}) + X_1 \cdot \widetilde{Q}_R(X_2, \ldots, X_{23})$$

Where:

- $\widetilde{Q}_L$: coefficients where $X_1 = 0$ (first half: indices $0$ to $2^{22} - 1$)
- $\widetilde{Q}_R$: coefficients where $X_1 = 1$ (second half: indices $2^{22}$ to $2^{23} - 1$)

**Evaluation at $\vec{\rho}$**:

$$\widetilde{Q}(\vec{\rho}) = \widetilde{Q}_L(\rho_2, \ldots, \rho_{23}) + \rho_1 \cdot \widetilde{Q}_R(\rho_2, \ldots, \rho_{23})$$

**Prover sends**:

- $C_L$: Commitment to $\widetilde{Q}_L$ (192 bytes)
- $C_R$: Commitment to $\widetilde{Q}_R$ (192 bytes)
- $v_L = \widetilde{Q}_L(\rho_2, \ldots, \rho_{23})$ (32 bytes)
- $v_R = \widetilde{Q}_R(\rho_2, \ldots, \rho_{23})$ (32 bytes)

**Verifier checks**:

1. $v_L + \rho_1 \cdot v_R = q$ (verifies split is correct)
2. Commitments consistent: $C_Q = \text{combine}(C_L, C_R)$ using pairing structure

**Verifier samples**: Challenge $\alpha_1 \leftarrow \mathcal{H}(\text{transcript} \parallel C_L \parallel C_R)$

**Fold polynomials**:

$$\widetilde{Q}' = \widetilde{Q}_L + \alpha_1 \cdot \widetilde{Q}_R$$

**New claim**: $\widetilde{Q}'(\rho_2, \ldots, \rho_{23}) = v_L + \alpha_1 \cdot v_R$

**Progress**: Reduced from 23-variable polynomial to 22-variable polynomial!

---

**Rounds 2-23**: Continue splitting on $\rho_2, \ldots, \rho_{23}$

After 23 rounds:

- Polynomial reduced to constant
- Verifier checks final value

**Total Dory proof size**: 23 rounds × (2 commitments + 2 values) = 23 × 448 bytes = **~10 KB**

---

#### Why This Works: The Pairing Magic

**Key property of Dory commitments**:

Given commitments $C_L$ and $C_R$ to polynomials $\widetilde{Q}_L$ and $\widetilde{Q}_R$, the verifier can efficiently check that:

$$C_Q \stackrel{?}{=} e(C_L, G_2) \cdot e(C_R, G_2')$$

Where $G_2, G_2'$ are structured reference string (SRS) elements.

This allows verifier to check split consistency **without** recomputing full commitment!

**Security**: Soundness relies on discrete log hardness in pairing-friendly groups

---

#### Batching Benefit Summary

**What we achieved**:

Started with: **36 separate evaluation claims** at different points $\vec{r}_1, \ldots, \vec{r}_{36}$

After batching:

1. **Reduction sumcheck** (2.9 KB): Reduced to single evaluation claim at $\vec{\rho}$
2. **Dory opening** (10 KB): Proved that single evaluation

**Total Stage 5 proof**: ~**13 KB**

**Compare to naïve approach**: 36 × 10 KB = **360 KB**

**Savings**: **~28× reduction** in proof size!

**Verifier work**:

- Reduction sumcheck: 23 rounds × $O(1)$ = $O(23)$ field operations
- Compute eq values: 36 × $O(23)$ = $O(828)$ field operations
- Dory verification: 23 rounds × 1 pairing check = $O(23)$ pairings
- **Total**: Much less than 36 separate verifications!

---

#### The Complete Reduction Chain

**Visual summary of batching**:

$$
\boxed{
\begin{array}{c}
\text{36 claims at different points } \vec{r}_1, \ldots, \vec{r}_{36} \\
\downarrow \text{ (Random linear combination with } \beta_i \text{)} \\
\text{Reduction polynomial } \widetilde{Q}(\vec{X}) = \sum_i \beta_i \widetilde{P}_i(\vec{X}) \cdot \text{eq}(\vec{X}, \vec{r}_i) \\
\downarrow \text{ (Sumcheck with 23 rounds)} \\
\text{Single claim: } \widetilde{Q}(\vec{\rho}) = q \\
\downarrow \text{ (Expand reduction polynomial)} \\
q = \sum_i \beta_i \widetilde{P}_i(\vec{\rho}) \cdot \text{eq}(\vec{\rho}, \vec{r}_i) \\
\downarrow \text{ (Dory opening with 23 rounds)} \\
\text{Verified!}
\end{array}
}
$$

**The beauty**: Transform 36 hard problems (openings at arbitrary points) into 1 easy problem (opening at random point)!

### Step 5: Store Opening Proof

```rust
state_manager.proofs.insert(
    ProofKeys::ReducedOpeningProof,
    ProofData::ReducedOpeningProof(reduced_opening_proof)
);
```

### Stage 5 Output

**Mathematical objects**:

1. **Reduction sumcheck proof** $\pi_{\text{reduce}}$:
   - Size: $n \times 4 \times 32$ bytes (where $n = \log(N)$)

2. **Dory opening proof** $\pi_{\text{open}}$:
   - Size: $\log(N) \times 192$ bytes

3. **Total Stage 5 proof**: ~5-10 KB (for typical trace sizes)

**Efficiency**: Proved ~50 polynomial evaluations with:

- 1 reduction sumcheck
- 1 Dory opening
- Instead of 50 separate Dory openings!

**Savings**: $\sim 50 \times 10\text{ KB} = 500\text{ KB} \rightarrow 10\text{ KB}$ (50× reduction!)

---

## Summary: Mathematical Objects at Each Stage

### Stage 0: Setup

| Object | Type | Size (T=1024) |
|--------|------|---------------|
| Witness polynomials | $\widetilde{P}_1, \ldots, \widetilde{P}_m \in \mathbb{F}^N$ | ~400 KB |
| Commitments | $C_1, \ldots, C_m \in \mathbb{G}_T$ | ~10 KB |
| Opening hints | Blinding factors | ~3 KB |
| Transcript state | $\mathcal{O}$ | 32 bytes |

### Stage 1: Spartan Outer

| Object | Type | Size |
|--------|------|------|
| Sumcheck proof | $\pi_1 = \{g_0, g_1, \ldots, g_{\log m}\}$ | ~512 bytes |
| Virtual claims | $(Az, \vec{r}, v_A)$, $(Bz, \vec{r}, v_B)$, $(Cz, \vec{r}, v_C)$ | 3 claims |

**Accumulator state**: 3 virtual claims, 0 committed claims

### Stage 2: Batched Sumchecks

| Object | Type | Size |
|--------|------|------|
| Batched proof | $\pi_2$ | ~1-2 KB |
| New virtual claims | Matrix MLEs $\widetilde{A}$, $\widetilde{B}$, $\widetilde{C}$ | ~10 claims |
| New committed claims | Witness $\widetilde{z}$, register/RAM polynomials | ~20 claims |

**Accumulator state**: 10 virtual claims, 20 committed claims

### Stage 3: More Batched Sumchecks

| Object | Type | Size |
|--------|------|------|
| Batched proof | $\pi_3$ | ~2-3 KB |
| Resolved virtual claims | Stage 2 virtuals now proven | -10 claims |
| New committed claims | Instruction polynomials $\widetilde{L}$, $\widetilde{R}$, lookups | ~20 claims |

**Accumulator state**: 0 virtual claims, 40 committed claims

### Stage 4: Final Sumchecks

| Object | Type | Size |
|--------|------|------|
| Batched proof | $\pi_4$ | ~1-2 KB |
| New committed claims | Bytecode, circuit flags | ~10 claims |

**Accumulator state**: 0 virtual claims, 50 committed claims

### Stage 5: Batched Opening

| Object | Type | Size |
|--------|------|------|
| Reduction sumcheck | $\pi_{\text{reduce}}$ | ~2 KB |
| Dory opening proof | $\pi_{\text{open}}$ | ~4 KB |

**All claims resolved!**

### Final Proof Structure

```rust
pub struct JoltProof {
    // Stage 0
    commitments: Vec<G_T>,              // ~10 KB

    // Stage 1-4
    stage1_sumcheck: SumcheckProof,     // ~0.5 KB
    stage2_sumcheck: SumcheckProof,     // ~2 KB
    stage3_sumcheck: SumcheckProof,     // ~3 KB
    stage4_sumcheck: SumcheckProof,     // ~2 KB

    // Stage 5
    reduced_opening: ReducedOpening,    // ~6 KB

    // Auxiliary
    advice_commitments: Option<G_T>,    // 192 bytes (if used)
}
```

**Total proof size**: ~25-30 KB for T=1024 trace

**Scales logarithmically**: Doubling trace length adds ~2 KB to proof size

---

## Conclusion

The five-stage DAG structure achieves:

1. **Modularity**: Each component (Spartan, Twist, Shout) proven independently
2. **Efficiency**: Batching reduces proof size and verification time
3. **Clarity**: Virtual vs. committed polynomial distinction creates clean dependency graph

**Key mathematical transformations**:

$$\boxed{\text{Execution trace}} \xrightarrow{\text{MLE}} \boxed{\text{Witness polynomials}} \xrightarrow{\text{Dory}} \boxed{\text{Commitments}}$$

$$\boxed{\text{Commitments}} \xrightarrow{\text{Stages 1-4}} \boxed{\text{Evaluation claims}} \xrightarrow{\text{Stage 5}} \boxed{\text{Opening proof}}$$

**The prover** transforms a ~100 KB witness into a ~30 KB proof.

**The verifier** checks the proof using:

- Polynomial commitments (~10 KB)
- Opening proof (~6 KB)
- Sumcheck proofs (~10 KB)
- **Total verification data**: ~26 KB

**Verification time**: O(log T) - polylogarithmic in trace length!

This completes the proof generation deep dive. The mathematical journey from execution trace to succinct proof demonstrates Jolt's core innovation: **lookups + sumcheck + polynomial commitments = efficient zkVM**.

/newpage

# Part 4: Verification Deep Dive

> **📘 For Detailed Implementation Reference**: This section provides a high-level overview of verification. For complete stage-by-stage mathematical analysis with line-by-line code mapping, see [03_Verifier_Mathematics_and_Code.md](03_Verifier_Mathematics_and_Code.md).

## Table of Contents

1. [Overview: The Verifier's Job](#overview-the-verifiers-job)
2. [What the Verifier Receives](#what-the-verifier-receives)
3. [Verifier Setup](#verifier-setup)
4. [The Five Verification Stages](#the-five-verification-stages)
5. [Complete Verification Example](#complete-verification-example)
6. [Security Analysis](#security-analysis)
7. [Summary](#summary)

---

## Overview: The Verifier's Job

**The fundamental asymmetry of zkSNARKs**:

- **Prover**: Does expensive work (O(N) where N = trace length)
  - Executes RISC-V program
  - Generates witness polynomials
  - Runs ~23 sumchecks
  - Creates polynomial commitments

- **Verifier**: Does cheap work (O(log N))
  - **Never executes the program**
  - **Never sees the execution trace**
  - Only checks mathematical consistency of proof
  - Uses public commitments + algebraic checks

**What makes verification fast?**

1. **Sumcheck verification**: O(log N) per sumcheck (not O(N))
2. **Batching**: Verifies multiple sumchecks with one challenge sequence
3. **Polynomial commitments**: Constant-size commitments, log-size opening proofs
4. **No witness access**: Verifier never reconstructs the ~7M element witness vector

---

## What the Verifier Receives

**From preprocessing** (one-time setup per program):

```rust
pub struct JoltVerifierPreprocessing {
    // Commitment to bytecode (proves which program ran)
    bytecode_commitment: GroupElement,  // 192 bytes

    // Dory verifier generators (for polynomial opening verification)
    generators: DoryVerifierSetup,  // ~25 KB (much smaller than prover's ~400 KB)

    // Program metadata
    trace_length: usize,  // T = number of cycles
    memory_layout: MemoryLayout,
}
```

**From prover** (per execution):

```rust
pub struct JoltProof {
    // Polynomial commitments (35 + 1 for Spartan z)
    commitments: Vec<GroupElement>,  // 36 × 192 bytes = ~7 KB

    // Stage 1-4 sumcheck proofs
    sumcheck_proof_stage1: SumcheckProof,  // ~500 bytes
    sumcheck_proof_stage2: BatchedSumcheckProof,  // ~2 KB
    sumcheck_proof_stage3: BatchedSumcheckProof,  // ~3 KB
    sumcheck_proof_stage4: BatchedSumcheckProof,  // ~2 KB

    // Stage 5 batched opening proof
    opening_proof: DoryOpeningProof,  // ~13 KB

    // Total: ~28 KB for typical execution
}
```

**Public inputs/outputs**:

```rust
pub struct ProgramIO {
    inputs: Vec<u8>,      // What was provided to the program
    outputs: Vec<u8>,     // What the program claimed to output
    panic: bool,          // Did the program panic?
}
```

**What the verifier knows**:
-  The program bytecode (via commitment from preprocessing)
-  Public inputs and claimed outputs
-  Trace length T (number of cycles executed)

**What the verifier does NOT know**:
-  The execution trace (which instructions ran, in what order)
-  Register values during execution
-  Memory values during execution
-  Intermediate computation results

**The verifier's guarantee**: If verification passes, then with overwhelming probability:
1. The committed bytecode was executed correctly for T cycles
2. Given the public inputs, the program produced the claimed outputs
3. All RISC-V semantics were followed (registers, memory, instruction decoding)

---

## Verifier Setup

**File**: [jolt-core/src/zkvm/dag/jolt_dag.rs:383](../jolt-core/src/zkvm/dag/jolt_dag.rs#L383)

### Creating Verifier StateManager

**Prover StateManager** (from Part 3):
- Contains execution trace (~7M field elements for T=1024)
- Contains all witness polynomials
- Heavy memory usage (~400 MB)

**Verifier StateManager**:
- Contains **only** proof data and commitments
- No execution trace
- Light memory usage (~1 MB)

```rust
let state_manager = proof.to_verifier_state_manager(
    verifier_preprocessing,
    program_io,
);
```

**What's inside**:

```rust
pub struct VerifierState {
    // Polynomial commitments (received from proof)
    commitments: Vec<GroupElement>,  // 36 commitments

    // Opening accumulator (tracks claims to verify)
    accumulator: VerifierOpeningAccumulator,

    // Shared transcript (for Fiat-Shamir)
    transcript: Transcript,

    // Metadata
    trace_length: usize,
    memory_layout: MemoryLayout,
}
```

**Key difference from prover**:

| Component | Prover | Verifier |
|-----------|--------|----------|
| Execution trace |  Full trace (7M elements) |  None (only T = length) |
| Witness polynomials |  All 35 MLEs |  None (only commitments) |
| Opening accumulator | `ProverOpeningAccumulator` | `VerifierOpeningAccumulator` |
| Transcript | Append witness data | Append only commitments |
| Memory | ~400 MB | ~1 MB |

---

### Fiat-Shamir Preamble

**Critical requirement**: Verifier and prover must have **identical transcripts**.

**File**: [jolt-core/src/zkvm/dag/state_manager.rs](../jolt-core/src/zkvm/dag/state_manager.rs)

```rust
state_manager.fiat_shamir_preamble();
```

**What gets appended to transcript**:

```rust
// 1. Public inputs/outputs
transcript.append_bytes(&program_io.inputs);
transcript.append_bytes(&program_io.outputs);
transcript.append_u64(trace_length);

// 2. Memory layout (public)
transcript.append_u64(memory_layout.input_start);
transcript.append_u64(memory_layout.ram_witness_offset);
// ... other memory regions

// 3. Polynomial commitments (from proof)
for commitment in commitments {
    transcript.append_group_element(commitment);
}
```

**Why this matters**:

Every random challenge the verifier samples must be **identical** to what the prover sampled:

$$\text{challenge}_{\text{verifier}} = \mathcal{H}(\text{transcript}_{\text{verifier}}) \stackrel{?}{=} \mathcal{H}(\text{transcript}_{\text{prover}}) = \text{challenge}_{\text{prover}}$$

If transcripts diverge even slightly:
- Different challenges sampled
- Verification fails (even for honest proof)

**Example of transcript divergence**:

```
Prover transcript:
  inputs: [5, 7]
  outputs: [12]
  T = 1024
  C_L = 0x1a2b3c...

Verifier transcript:
  inputs: [5, 7]
  outputs: [12]
  T = 1024
  C_L = 0x1a2b3c...  ← Must match exactly!
```

Even one bit difference → complete verification failure.

---

## The Five Verification Stages

> **📘 Detailed Analysis**: For complete mathematical equations and code walkthrough of each stage, see [03_Verifier_Mathematics_and_Code.md - Stage-by-Stage Analysis](03_Verifier_Mathematics_and_Code.md#stage-by-stage-mathematical-analysis).

Jolt's verification follows a **5-stage pipeline**, where each stage verifies a batch of sumchecks. The key insight: **verification is O(log N) per sumcheck**, not O(N).

### High-Level Stage Flow

```
Stage 1: Spartan Outer Sumcheck
   ↓ (outputs evaluation points for Az, Bz, Cz)
Stage 2: Product Sumchecks + Read/Write Checking
   ↓ (6-8 batched sumchecks: Spartan products, Registers/RAM Twist, Instruction lookups)
Stage 3: Evaluation Sumchecks + Write Checking
   ↓ (6-8 batched sumchecks: Spartan matrix eval, Registers/RAM Twist, Lookups)
Stage 4: Final Component Sumchecks
   ↓ (4-6 batched sumchecks: RAM final, Bytecode Shout, Lookups)
Stage 5: Batched Opening Proof (Dory)
   ↓ (accumulate all polynomial evaluation claims, verify via Dory-Reduce)
✓ Verification Complete
```

**Total**: ~22 sumchecks in Stages 1-4, then 1 batched opening in Stage 5

---

### Stage 1: Spartan Outer Sumcheck

**What it verifies**: R1CS constraint satisfaction - that 30 constraints per cycle hold

**Prover's claim**:
$$\sum_{x \in \{0,1\}^{\log m}} \text{eq}(\tau, x) \cdot [(Az)(x) \cdot (Bz)(x) - (Cz)(x)] = 0$$

**Verifier's job**:
- Verify $\log m$ rounds of sumcheck (where $m = 30T$, for T cycles)
- For each round: Check $g_i(0) + g_i(1) = \text{previous\_claim}$, sample challenge $r_i$
- After all rounds: Verify final evaluation claim at point $\vec{r} = (r_1, \ldots, r_{\log m})$
- **Cost**: O(log m) field operations

**Output**: Evaluation points for $Az$, $Bz$, $Cz$ → passed to Stage 2

---

### Stages 2-4: Batched Component Verification

These three stages verify the five Jolt components (Spartan, Registers, RAM, Instruction Lookups, Bytecode) through batched sumchecks.

**Stage 2: Product Sumchecks + Read/Write Checking** (~6-8 sumchecks batched)
- **Spartan**: Prove $Az$, $Bz$, $Cz$ are correct sparse matrix-vector products
- **Registers (Twist)**: Prove register reads (rs1, rs2) match last writes
- **RAM (Twist)**: Prove RAM reads match last writes
- **Instruction Lookups (Shout)**: Begin prefix-suffix sumcheck for instruction execution
- **All verified together**: Single batched sumcheck with random combination $\rho_i$

**Stage 3: Evaluation Sumchecks + Write Checking** (~6-8 sumchecks batched)
- **Spartan**: Evaluate sparse polynomials at challenge points
- **Registers (Twist)**: Complete register consistency proof (write-checking + evaluation)
- **RAM (Twist)**: Complete RAM consistency proof (write-checking + evaluation)
- **Instruction Lookups (Shout)**: Continue prefix-suffix sumcheck phases

**Stage 4: Final Component Sumchecks** (~4-6 sumchecks batched)
- **RAM (Twist)**: Final evaluation sumcheck
- **Bytecode (Shout)**: Prove instruction decode matches committed bytecode
- **Instruction Lookups (Shout)**: Complete instruction lookup proof

**Key optimization - Batching**:
- Instead of verifying 6-8 sumchecks separately, batch them with random coefficients $\rho_1, \ldots, \rho_n$
- Combined claim: $\sum_i \rho_i \cdot \text{claim}_i$
- **Single sumcheck** verifies all claims together
- **Cost**: Same as one sumcheck, not $n$ sumchecks!
- **Security**: Schwartz-Zippel ensures cheating detected with high probability

---

### Stage 5: Batched Opening Proof (Dory)

**What it verifies**: All polynomial evaluation claims from Stages 1-4 are correct

**The accumulated claims** (~30-50 total):
- From all sumchecks: "Polynomial $P_i$ evaluates to $v_i$ at point $\vec{r}_i$"
- Example claims: $\widetilde{Z}(\vec{r}_1) = v_1$, $\widetilde{\text{ra}}(\vec{r}_2) = v_2$, etc.

**Batching via Random Linear Combination**:
1. Sample random coefficients $\gamma_1, \ldots, \gamma_n$ (Fiat-Shamir)
2. Combine into single polynomial: $Q(X) = \sum_i \gamma_i \cdot \widetilde{\text{eq}}(\vec{r}_i, X) \cdot P_i(X)$
3. Single evaluation claim: $Q(\vec{r}^*) = v^*$ where $v^* = \sum_i \gamma_i \cdot v_i$

**Verification steps**:
1. **Reduction sumcheck** (~23 rounds): Reduce combined claim to single point evaluation
   - Verifier checks univariate polynomial consistency each round
   - Samples random challenges to reduce dimension
   - **Cost**: O(log N) field operations

2. **Dory opening proof** (Dory-Reduce protocol):
   - Prover sends cross-term commitments for ~23 folding rounds
   - Verifier homomorphically computes folded commitment using challenges $\alpha_i, \beta_i$
   - Base case: Scalar-Product sigma protocol with 4-5 pairings
   - **Cost**: ~40 $\mathbb{G}_T$ exponentiations (the main bottleneck!)

**Why Stage 5 is expensive**:
- $\mathbb{G}_T$ exponentiations dominate (~80% of verification time)
- Each exponentiation: ~254 $\mathbb{F}_q^{12}$ multiplications
- No EVM precompile (unlike pairings)
- Batching optimization reduces from ~200 to ~40 exponentiations

---

### Summary: What Verifier Actually Computes

**Total verification work**:
- **~22 sumchecks** (Stages 1-4): Each costs O(log N) field operations
- **1 batched opening** (Stage 5): ~40 $\mathbb{G}_T$ exponentiations + 4-5 pairings
- **Memory**: ~1 MB (vs prover's ~400 MB)
- **No witness access**: Never sees execution trace or intermediate values

**For each sumcheck round**:
1. Receive univariate polynomial $g_i(X)$ from proof (4 coefficients, 128 bytes)
2. Check consistency: $g_i(0) + g_i(1) = \text{previous\_claim}$
3. Sample random challenge: $r_i = \mathcal{H}(\text{transcript} \parallel g_i)$
4. Compute next claim: $\text{next\_claim} = g_i(r_i)$

**After all sumchecks**:
- Accumulate polynomial evaluation claims
- Batch with random linear combination
- Verify single Dory opening proof

**Result**: Verification in O(log N) time, confirming program executed correctly without re-running it!

---

## Complete Verification Example

Let's walk through the entire verification for our 4-bit ADD example from Part 3.

**Setup**:
- Program: `ADD r3, r1, r2` where `r1 = 5`, `r2 = 7`
- Expected output: `r3 = 12`
- Trace length: T = 1 cycle
- Simplified: 4-bit operands with 1-bit chunks (4 chunks)

**What verifier receives**:

```rust
JoltProof {
    commitments: [C_z, C_L, C_R, C_Δ_rd, C_ra_0, C_ra_1, C_ra_2, C_ra_3],  // 8 commitments
    sumcheck_proof_stage1: π_1,  // Spartan outer
    sumcheck_proof_stage2: π_2,  // Registers + Instructions Booleanity
    sumcheck_proof_stage3: π_3,  // Instructions Read-checking + Hamming
    sumcheck_proof_stage4: π_4,  // R1CS Linking
    opening_proof: π_Dory,       // Batched Dory opening
}

ProgramIO {
    inputs: [5, 7],
    outputs: [12],
    panic: false,
}
```

---

### Verification Step-by-Step

**Step 1: Fiat-Shamir Preamble**

```rust
transcript.append_bytes(&[5, 7]);  // inputs
transcript.append_bytes(&[12]);    // outputs
transcript.append_u64(1);          // T = 1 cycle
transcript.append_group_element(C_z);
transcript.append_group_element(C_L);
// ... append all 8 commitments
```

**Transcript state**: `hash([5, 7, 12, 1, C_z, C_L, ...])`

---

**Step 2: Stage 1 Verification (Spartan Outer)**

**Sample challenge**:
$$\tau = \mathcal{H}(\text{transcript}) = 0.37$$

**Extract proof**: $\pi_1 = (g_1)$ (univariate polynomial, 1 round for $\log m = 1$)

Suppose $g_1(X) = c_0 + c_1 X + c_2 X^2 + c_3 X^3$.

**Check consistency**:
$$g_1(0) + g_1(1) \stackrel{?}{=} 0$$
$$c_0 + (c_0 + c_1 + c_2 + c_3) \stackrel{?}{=} 0$$

 Passes (prover constructed correctly).

**Sample challenge**:
$$r_1 = \mathcal{H}(\text{transcript} \parallel g_1) = 0.91$$

**Final claim**:
$$v = g_1(0.91) = 0$$ (since outer sumcheck proves sum equals 0)

**Store virtual claims** (from prover):
- $Az(0.91) = 42.7$
- $Bz(0.91) = 31.2$
- $Cz(0.91) = 1332.24$

**Verify**:

$$0 \stackrel{?}{=} \text{eq}(0.37, 0.91) \cdot [42.7 \times 31.2 - 1332.24]$$

$$0 \stackrel{?}{=} 0.3934 \times 0 = 0$$

 Check passes!

---

**Step 3: Stage 2 Verification (Registers + Instructions)**

**Sample batching coefficients**:
$$\alpha_1 = 0.123, \quad \alpha_2 = 0.456$$ (for 2 sumchecks: registers + booleanity)

**Extract proof**: $\pi_2 = (g_1, g_2, \ldots, g_5)$ (5 rounds for witness dimension)

**Check consistency** (round 1):
$$g_1(0) + g_1(1) \stackrel{?}{=} \alpha_1 H_{\text{registers}} + \alpha_2 H_{\text{booleanity}}$$

Continue for all 5 rounds, sampling challenges $\vec{r}' = (r_1', \ldots, r_5')$.

**Final claim**: $v_{\text{combined}} = 327.0$

**Extract individual claims**:
- Registers: Store $\widetilde{\Delta}_{\text{rd}}(\vec{r}') = 12.003$ (committed opening)
- Booleanity: Internal claim (verified implicitly)

---

**Step 4: Stage 3 Verification (Instructions Read-checking)**

**Sample batching coefficients**: $\alpha_1 = 0.789, \alpha_2 = 0.234$ (read-raf + hamming)

**Extract proof**: $\pi_3 = (g_1, \ldots, g_{18})$ (18 rounds for 10 cycle + 8 table dimensions)

**Verify batched sumcheck** (as before).

**Extract claims**: Store opening claims for $\widetilde{\text{ra}}_0, \widetilde{\text{ra}}_1, \widetilde{\text{ra}}_2, \widetilde{\text{ra}}_3$ at random points.

---

**Step 5: Stage 4 Verification (R1CS Linking)**

**Verify batched sumcheck**.

**Extract claims**: Store $\widetilde{L}(\vec{r}_L) = 5.217$, $\widetilde{R}(\vec{r}_R) = 7.901$.

---

**Step 6: Stage 5 Verification (Batched Opening)**

**Accumulator now contains**:

| Polynomial | Evaluation Point | Claimed Value |
|------------|------------------|---------------|
| WitnessZ | $\vec{r}'$ | 8.342 |
| LeftInput | $\vec{r}_L$ | 5.217 |
| RightInput | $\vec{r}_R$ | 7.901 |
| RdInc | $\vec{r}_{\Delta}$ | 12.003 |
| InstructionRa_0 | $\vec{r}_{\text{ra0}}$ | 0.001 |
| InstructionRa_1 | $\vec{r}_{\text{ra1}}$ | 0.000 |
| InstructionRa_2 | $\vec{r}_{\text{ra2}}$ | 0.000 |
| InstructionRa_3 | $\vec{r}_{\text{ra3}}$ | 1.000 |

**Total: 8 claims** (simplified from real 36).

**Sample batching coefficients**:
$$\beta_1, \ldots, \beta_8 = \mathcal{H}(\text{transcript})$$

**Compute expected sum**:
$$H = \beta_1 \times 8.342 + \beta_2 \times 5.217 + \cdots + \beta_8 \times 1.000 = 2731.4$$

**Verify reduction sumcheck**: $\pi_{\text{reduce}} = (g_1, \ldots, g_{23})$

After 23 rounds: $\widetilde{Q}(\vec{\rho}) = q$ where $\vec{\rho} \in \mathbb{F}^{23}$

**Compute eq values**:
$$\text{eq}(\vec{\rho}, \vec{r}') = 0.0023$$
$$\text{eq}(\vec{\rho}, \vec{r}_L) = 0.0157$$
$$\vdots$$

**Verify Dory opening**: $\pi_{\text{Dory}}$ (23 rounds)

Each round: Check value consistency + pairing check.

After 23 rounds: Final value matches commitment.

 **Verification succeeds!**

---

### Verification Verdict

**Verifier concludes**:

With overwhelming probability ($\geq 1 - \text{negl}$):
1.  A RISC-V program with the committed bytecode executed for 1 cycle
2.  Given inputs `[5, 7]`, the program produced output `[12]`
3.  All register operations were consistent (r1=5, r2=7 were read; r3=12 was written)
4.  All instruction lookups were correct (ADD(5,7) = 12 via 4 sub-lookups)
5.  No memory operations occurred (this was pure register arithmetic)

**What verifier never saw**:
-  The actual execution trace (which instructions, which cycles)
-  Intermediate register values
-  The 1-hot lookup address matrices
-  Any of the witness polynomials (only saw commitments)

**Verification cost**:
- Total time: ~10 ms (on modern CPU)
- Proof size: ~10 KB
- Prover time: ~100 ms (10× slower)
- Trace size: ~1 KB

**Asymptotic scaling** (for T cycles):
- Prover: O(T log T)
- Verifier: O(log T)
- Proof size: O(log T)

---

## Security Analysis

### What Could a Malicious Prover Try?

Let's analyze different attack vectors and why they fail.

---

#### Attack 1: Lie About Execution Result

**Attack**: Claim `r3 = 100` instead of `r3 = 12` (incorrect ADD result).

**Where it's caught**:

**Stage 3: Instruction Read-checking**

The Shout sumcheck verifies:
$$\sum_{j,k} \widetilde{\text{ra}}_i(j, k) \cdot [\text{lookup}(j,k) - \text{table}(k)] = 0$$

For chunk 0 (bit position 0):
- Operands: $(L_0, R_0) = (1, 1)$ → table index $k = 3$
- Lookup table: $\text{ADD\_table}(3) = 1 + 1 = 0$ (with carry)
- If prover lies about result, lookup will be inconsistent

**Sumcheck will fail** because:
$$\sum_k \widetilde{\text{ra}}_0(0, k) \cdot [\text{wrong\_result}(k) - \text{table}(k)] \neq 0$$

The one-hot property ensures only $k=3$ contributes:
$$\widetilde{\text{ra}}_0(0, 3) \cdot [\text{wrong}(3) - \text{table}(3)] = 1 \cdot [\text{wrong} - 0] \neq 0$$

**Verification fails at Stage 3** (read-checking sumcheck consistency).

**Probability of fooling verifier**: $\leq 2^{-128}$ (soundness error)

---

#### Attack 2: Forge a Polynomial Commitment

**Attack**: Provide fake commitment $C_L'$ instead of honest $C_L = \text{Commit}(\widetilde{L})$.

**Where it's caught**:

**Stage 5: Dory Opening**

Verifier receives:
- Fake commitment $C_L'$
- Claimed evaluation: $\widetilde{L}(\vec{r}_L) = 5.217$

Prover must provide Dory opening proof for:
$$C_L' \stackrel{?}{\text{opens to}} 5.217 \text{ at } \vec{r}_L$$

**Problem**: Prover doesn't know the polynomial that $C_L'$ commits to!

**Dory opening requires**: For 23-round opening protocol, prover must provide:
- Commitments to left/right halves at each round
- Values $v_L^{(k)}, v_R^{(k)}$ such that value+commitment checks pass
- Pairing checks must succeed: $e(C_L^{(k)}, G_2) \cdot e(C_R^{(k)}, G_2') = C^{(k-1)}$

**Without knowing the polynomial**, prover cannot construct valid left/right splits.

**Security**: Binding property of Dory commitments (based on discrete log hardness in pairing groups).

**Probability of forging opening**: $\leq 2^{-128}$ (computational assumption)

---

#### Attack 3: Cheat on One Sumcheck in Batched Proof

**Attack**: Provide valid proofs for 7 out of 8 Stage 2 sumchecks, lie on one.

**Where it's caught**:

**Stage 2 Batched Verification**

Verifier checks:
$$\alpha_1 f_1(\vec{r}) + \cdots + \alpha_7 f_7(\vec{r}) + \alpha_8 f_8(\vec{r}) \stackrel{?}{=} v_{\text{combined}}$$

Suppose prover lies on $f_7$: claims $\sum f_7(x) = H_7'$ but actually $\sum f_7(x) = H_7$.

**Batched initial claim**:
$$\alpha_1 H_1 + \cdots + \alpha_7 H_7' + \alpha_8 H_8$$

**Actual sum**:
$$\alpha_1 H_1 + \cdots + \alpha_7 H_7 + \alpha_8 H_8$$

**Difference**:
$$\alpha_7 (H_7' - H_7) \neq 0$$

For prover to succeed, need: $\alpha_7 (H_7' - H_7) = 0$

**But** $\alpha_7$ is sampled from Fiat-Shamir **after** prover commits to $H_7'$ (via transcript binding).

**Schwartz-Zippel**: Probability $\alpha_7 = $ specific value needed: $\leq 1/|\mathbb{F}| \approx 2^{-256}$

**Verification fails** with overwhelming probability.

---

#### Attack 4: Fiat-Shamir Transcript Manipulation

**Attack**: Prover tries to manipulate transcript to get favorable challenges.

**Example**: Try to make $\alpha_7 = 0$ to hide cheating in sumcheck 7.

**Why it fails**:

**Transcript is a hash chain**:
$$\alpha_7 = \mathcal{H}(\text{transcript} \parallel \text{prover\_messages})$$

**Prover cannot choose $\alpha_7$** without:
1. Breaking hash function preimage resistance
2. Finding commitments that hash to desired value

**Security**: Collision resistance of hash function (e.g., SHA-256)

**Probability of manipulation**: $\leq 2^{-128}$ (hash security)

---

#### Attack 5: Replace Execution with Different Program

**Attack**: Execute different program but provide proof for committed bytecode.

**Where it's caught**:

**Stage 4: Bytecode Read-checking**

Verifier has: $C_{\text{bytecode}}$ (commitment from preprocessing)

**Shout offline memory checking** verifies:
$$\sum_{j=0}^{T-1} \text{fingerprint\_read}(j) \stackrel{?}{=} \sum_{k=0}^{K-1} \text{count}(k) \cdot \text{fingerprint\_bytecode}(k)$$

Where:
- $\text{fingerprint\_read}(j) = \gamma_1 \text{PC}(j) + \gamma_2 \text{opcode}(j) + \cdots$
- $\text{fingerprint\_bytecode}(k)$ comes from committed bytecode

If prover executed different instructions:
- $\text{opcode}(j)$ won't match committed bytecode opcodes
- Fingerprints won't match
- Multiset equality fails

**Verification fails at Stage 4** (bytecode read-checking).

**Probability of fooling verifier**: $\leq 2^{-128}$ (Shout soundness)

---

### Overall Security Guarantee

**Theorem** (Informal):

If verification passes, then with probability $\geq 1 - \epsilon$ where $\epsilon \leq 2^{-100}$:

1. **Correctness**: The committed program executed correctly for T cycles on the given inputs
2. **Completeness**: All RISC-V semantics were followed
3. **Soundness**: Prover cannot convince verifier of false statement (except with probability $\epsilon$)

**Security parameters**:
- Field size: $|\mathbb{F}| \approx 2^{256}$
- Soundness error per sumcheck: $\leq d/|\mathbb{F}|$ where $d$ is degree
- Number of sumchecks: ~23
- Total soundness error: $\leq 23 \times 4 / 2^{256} \approx 2^{-249}$
- Computational assumptions: Discrete log hardness, hash security

**Concrete security**: $\geq 128$ bits (industry standard)

---

## Summary: The Verifier's Algorithm

**High-level pseudocode**:

```python
def verify_jolt_proof(preprocessing, program_io, proof):
    # 1. Setup
    state_manager = create_verifier_state(preprocessing, proof, program_io)
    state_manager.fiat_shamir_preamble()

    # 2. Stage 1: Spartan Outer
    verify_sumcheck(proof.stage1, state_manager)
    store_virtual_claims(Az_claim, Bz_claim, Cz_claim)

    # 3. Stage 2: Batched Sumchecks
    alphas_2 = sample_batching_coefficients(8)  # 8 sumchecks
    verify_batched_sumcheck(proof.stage2, alphas_2, state_manager)
    extract_output_claims(state_manager.accumulator)

    # 4. Stage 3: Batched Sumchecks
    alphas_3 = sample_batching_coefficients(6)
    verify_batched_sumcheck(proof.stage3, alphas_3, state_manager)
    extract_output_claims(state_manager.accumulator)

    # 5. Stage 4: Batched Sumchecks
    alphas_4 = sample_batching_coefficients(8)
    verify_batched_sumcheck(proof.stage4, alphas_4, state_manager)
    extract_output_claims(state_manager.accumulator)

    # 6. Stage 5: Batched Opening
    betas = sample_batching_coefficients(36)  # 36 polynomial claims

    # Verify reduction sumcheck
    H_expected = sum(beta_i * v_i for i in 1..36)
    verify_sumcheck(proof.reduction_sumcheck, H_expected, state_manager)
    rho = get_final_evaluation_point()

    # Compute eq values
    eq_values = [compute_eq(rho, r_i) for i in 1..36]

    # Verify Dory opening
    verify_dory_opening(
        commitments=proof.commitments,
        evaluation_point=rho,
        claimed_value=combined_claim,
        batching_coefficients=betas,
        eq_values=eq_values,
        dory_proof=proof.opening_proof
    )

    # 7. Success!
    return True  # All checks passed
```

**Complexity analysis**:

| Stage | Operations | Complexity |
|-------|------------|------------|
| Stage 1 | 15 rounds × O(1) | O(log m) = O(log T) |
| Stage 2 | 23 rounds × O(1) | O(log N) |
| Stage 3 | 18 rounds × O(1) | O(log N) |
| Stage 4 | 18 rounds × O(1) | O(log N) |
| Stage 5 Reduction | 23 rounds × O(1) | O(log N) |
| Stage 5 Dory | 4-5 pairings + ~40 G_T exps | O(log N) operations |
| **Total** | | **O(log N)** |

Where $N = $ witness size $\approx 7M$ for T = 1024.

**Memory**: O(log N) (store challenges, proof data, commitments)

**Communication**: O(log N) proof size (~28 KB for T = 1024)

---

## Key Takeaways

1. **Asymmetry is the magic**: Prover does O(N) work, verifier does O(log N)

2. **Never sees the witness**: Verifier checks algebraic consistency, not execution

3. **Fiat-Shamir is critical**: Same transcript → same challenges → binding

4. **Batching is efficient**: 23 sumchecks verified with 5 proofs (Stages 1-4 + Stage 5)

5. **Polynomial commitments enable succinctness**: Constant-size commitments, log-size openings

6. **Security relies on**:
   - Soundness of sumcheck protocol (information-theoretic)
   - Binding of polynomial commitments (computational)
   - Collision resistance of hash functions (computational)
   - Random linear combination (Schwartz-Zippel lemma)

7. **Verifier's trust**: Never trusts prover's claims directly, always verifies algebraically

8. **The proof is a certificate**: Self-contained, checkable without prover interaction

---

**End of Part 4: Verification Deep Dive**

You now understand how verification achieves:
- **Completeness**: Honest prover always convinces verifier
- **Soundness**: Malicious prover cannot convince verifier (except with negl. probability)
- **Succinctness**: O(log N) verification time and proof size
- **Zero-knowledge**: Verifier learns nothing beyond correctness (not covered here, but Jolt supports it)

For further reading, see the [Jolt paper](https://eprint.iacr.org/2023/1217.pdf) and [Spartan paper](https://eprint.iacr.org/2019/550.pdf) for formal security proofs.
