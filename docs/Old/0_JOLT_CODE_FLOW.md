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

1. Get random challenge \tau from transcript
2. Run Spartan outer sumcheck ([jolt-core/src/zkvm/spartan/inner.rs](jolt-core/src/zkvm/spartan/inner.rs)):
   ```
   Claim: \Sigma_x eq(\tau, x) · (Az(x) · Bz(x) - Cz(x)) = 0
   ```
   - This proves ~30 R1CS constraints per cycle are satisfied
   - Constraints enforce: PC updates, component linking, arithmetic ops

3. **The Sumcheck Protocol** ([jolt-core/src/subprotocols/sumcheck.rs](jolt-core/src/subprotocols/sumcheck.rs)):
   - For each round j = 0..n:
     - Prover computes univariate polynomial g_j(X_j)
       - Calls `sumcheck_instance.compute_prover_message()`
       - Evaluates at 0, 2, 3, ..., degree
     - Compress polynomial and append to transcript
     - Verifier samples random challenge r_j
     - Prover binds: `sumcheck_instance.bind(r_j, j)`

4. After sumcheck completes:
   - Claims about Az(r), Bz(r), Cz(r) at random point r
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

1. Sample batching coefficients \alpha_1, \alpha_2, \alpha_3, ... from transcript
2. Combine all claims:
   ```
   Combined claim = \alpha_1·claim_1 + \alpha_2·claim_2 + ...
   ```
3. Run single sumcheck on combined polynomial:
   ```
   \Sigma_x (\alpha_1·P_1(x) + \alpha_2·P_2(x) + ...)
   ```
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
   ```
   \prod fingerprint_time(i) = \prod fingerprint_addr(j)
   ```
   Where fingerprint = hash(address, timestamp, value)

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
   - Sample random coefficients \beta_1, \beta_2, ..., \beta_n
   - Create single combined claim:
     ```
     \beta_1·poly1(r1) + \beta_2·poly2(r2) + ... = \beta_1·v1 + \beta_2·v2 + ...
     ```
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

1. Sample same \tau challenge (from transcript)
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
   - For each round j:
     - Read compressed univariate from proof
     - Check consistency: g_j(0) + g_j(1) = previous_claim
     - Sample r_j from transcript
     - Compute next_claim = g_j(r_j)
   - Check final evaluation: output_claim = expected_evaluation(r)

4. Store virtual openings in verifier accumulator
   - Az(r), Bz(r), Cz(r) marked as claims to check later

**Key insight**: Verification is O(log n) per sumcheck, not O(n) like proving!

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

**Verification time**: O(log n) for Dory

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