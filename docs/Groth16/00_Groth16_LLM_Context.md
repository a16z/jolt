# Jolt Groth16 EVM Verifier: Complete Technical Context

**Purpose**: Consolidated context document for LLM assistance with Jolt's Groth16 conversion project.

**Last Updated**: 2025-11-25

**Source Documents**:
- `Groth16_Conversion_Scope.md` - Project scope and deliverables
- `zkLean_Infrastructure_Reuse.md` - Infrastructure details and code walkthrough
- `Partial_Transpilation_Exploration.md` - Transpilation theory and implementation

---

# Part I: Problem Statement and Goals

## 1.1 The Problem

Jolt is a RISC-V zkVM that produces native proofs requiring verification that cannot happen on-chain:

| Metric | Current Value | Problem |
|--------|---------------|---------|
| **Proof size** | ~71KB (72,529 bytes for Fibonacci) | ~1.16M gas for calldata alone |
| **Verification cycles** | ~1.5B RISC-V cycles | No EVM precompiles for GT exponentiations |
| **Verification cycles (with PR #975)** | ~600M cycles | Still too expensive for direct on-chain |

**Root cause**: Direct on-chain verification requires expensive field operations and cryptographic primitives (particularly GT exponentiations) not available as EVM precompiles.

## 1.2 The Goal

Convert Jolt's verifier into a Groth16 circuit:

| Metric | Target Value | Benefit |
|--------|--------------|---------|
| **On-chain verification** | ~280k gas | Standard Groth16 on BN254 |
| **Proof size** | ~192-260 bytes | Fits in single transaction |
| **Maintenance** | Automatic sync | Pipeline adapts to Jolt changes |

**Critical requirement**: The conversion pipeline must adapt to Jolt changes without full manual rewrite. Jolt is in active development (PR #975's hint mechanism, upcoming lattice PCS).

## 1.3 The Approach: zkLean-Based Transpilation

**Core insight**: Reuse zkLean's existing extraction infrastructure (PR #1060) to generate Gnark circuits instead of Lean4 code.

```
Pipeline:
Jolt Rust Verifier (single source of truth)
    ↓
zkLean Extraction (runtime introspection)
    ↓
AST Representation (MleAst)
    ↓
Gnark Code Generator (NEW - to be implemented)
    ↓
Gnark Circuit (Go)
    ↓
Groth16 Proof
```

**Why this works**: zkLean already extracts Jolt's verification logic into a structured format. We add a new code generator targeting Gnark instead of Lean4.

---

# Part II: Industry Analysis

## 2.1 RISC Zero: Fully Manual Circom

| Aspect | Details |
|--------|---------|
| **Approach** | Fully manual, hand-written Circom |
| **Circuit File** | `stark_verify.circom` - **55.7 MB** (stored in Git LFS) |
| **Pipeline** | STARK proof → Circom circuit → snarkjs → Groth16 proof |
| **Trusted Setup** | Hermez rollup with 2^23 powers of tau |
| **Platform Limitation** | x86-only (Circom witness generator uses x86 assembly) |

**Why manual works for RISC Zero**: Their STARK verification protocol is mature and stable. The 55MB Circom file is a one-time investment that rarely needs updates.

**Sources**:
- [stark_verify.circom](https://github.com/risc0/risc0/blob/main/groth16_proof/groth16/stark_verify.circom)
- [Trusted Setup Ceremony](https://dev.risczero.com/api/trusted-setup-ceremony)

## 2.2 SP1: Semi-Automatic via Recursion DSL

| Aspect | Details |
|--------|---------|
| **Approach** | Semi-automatic via **Recursion DSL Compiler** |
| **Architecture** | STARK proof → Recursion DSL → Gnark circuit → Groth16 proof |
| **Circuit Generation** | **Precompiled circuits** ("hot start") - ~18s pure prove time |
| **FFI** | `sp1-recursion-gnark-ffi` crate for Rust→Go interop |
| **Maintenance** | DSL program changes → circuit auto-regenerates |

**Key insight**: SP1 built a **custom DSL compiler** that translates their recursion program to Gnark circuits. This validates the "compile from higher representation" approach.

**Sources**:
- [SP1 Testnet Launch](https://blog.succinct.xyz/sp1-testnet/)
- [sp1-recursion-gnark-ffi](https://crates.io/crates/sp1-recursion-gnark-ffi)
- [ZKVM Kings Clash](https://medium.com/@gavin.ygy/zkvm-kings-clash-inside-the-architecture-performance-battle-between-pico-and-sp1-636ffede8831)

## 2.3 Comparison Matrix

| Aspect | RISC Zero | SP1 | Jolt (proposed) |
|--------|-----------|-----|-----------------|
| **Automation** | ❌ Manual | ✅ Semi-automatic (DSL) | ✅ Automatic (zkLean) |
| **Circuit Language** | Circom | Custom DSL → Gnark | Rust → AST → Gnark |
| **Maintenance** | Manual updates | DSL program updates | Re-run extractor |
| **Protocol Stability** | Stable | Evolving | Evolving |
| **Existing Infra** | N/A | Built custom compiler | Reuse zkLean |
| **Dev Effort** | High (circuit) | High (compiler) | Medium (generator) |

**Our advantage**: We don't need to build a custom compiler like SP1 did. zkLean's extraction infrastructure already exists.

## 2.4 Why Not Existing Automatic Tools?

| Tool | Output Format | Status | Problem |
|------|---------------|--------|---------|
| **zkLLVM** | PLONK (Placeholder) | Rust repo archived Feb 2025 | Not R1CS - incompatible with Groth16 |
| **Circom** | R1CS | Active | Requires manual DSL writing |
| **Arkworks** | R1CS | Active | Manual `ConstraintSynthesizer` only |
| **Lurk** | Nova/SuperNova | Active | Lisp-based, not Rust |
| **Nexus** | Nova/Folding | Active | Own VM, not direct transpilation |
| **Chiquito** | Halo2/PLONK | Active | Not R1CS |

**Critical finding about zkLLVM**: zkLLVM outputs PLONK (Placeholder), NOT R1CS. PLONK and R1CS are fundamentally different constraint systems - no straightforward conversion exists.

**Conclusion**: No existing tool provides automatic Rust → R1CS conversion. zkLean-based transpilation is the only viable path for:
1. **Automatic** generation (not manual rewrite)
2. **R1CS output** (required for Groth16)
3. **Maintainability** (re-run when Jolt changes)

## 2.5 Why Gnark (Not Arkworks or Circom)

- **5-10× faster** proving than Arkworks and snarkjs ([benchmarks](https://blog.celer.network/2023/08/04/the-pantheon-of-zero-knowledge-proof-development-frameworks/))
- **25% faster** than rapidsnark
- Optimized gadgets for field arithmetic, hashing, elliptic curves
- General-purpose Go language enables automatic code generation
- In-circuit recursive verifier in ~11.5k constraints

---

# Part III: zkLean Infrastructure Deep Dive

## 3.1 What is zkLean?

zkLean (PR #1060) extracts Jolt's frontend semantics into Lean4 for **formal verification**.

**What zkLean extracts** (from `zklean-extractor/src/main.rs`):
```rust
let modules: Vec<Box<dyn AsModule>> = vec![
    Box::new(ZkLeanR1CSConstraints::<ParameterSet>::extract()),
    Box::new(ZkLeanInstructions::<ParameterSet>::extract()),
    Box::new(ZkLeanLookupTables::<64>::extract()),
    Box::new(ZkLeanTests::<64>::extract(&mut rng)),
];
```

**Four modules**:
1. **R1CS Constraints**: The ~30 constraints applied to every VM cycle
2. **Instructions**: RISC-V instruction specifications and combine_lookups
3. **Lookup Tables**: MLE evaluations for lookup tables (4-bit ADD, XOR, etc.)
4. **Tests**: Property tests validating extraction correctness

## 3.2 Runtime Introspection vs Static Parsing

### Why Not Static Parsing (syn)?

**Static parsing approach**:
```rust
// Read Rust source code as text
let rust_code = read_file("jolt-core/src/subprotocols/sumcheck.rs")?;
let ast = syn::parse_file(rust_code)?;
// Walk AST, extract operations → IR
```

**Why this is HARD for Jolt**:
1. **Complex Rust**: Jolt uses traits, generics, lifetimes, macros
2. **Indirect operations**: Operations hidden behind trait implementations
3. **Dynamic behavior**: What happens depends on runtime values
4. **Maintenance nightmare**: If Jolt refactors code, parser breaks
5. **Essentially reimplementing rustc**: Must resolve types, traits, generics

### zkLean's Runtime Introspection Approach

**Execute and record**:
```rust
// Don't read source code!
// Instead, RUN Jolt's actual functions with special types

// Step 1: Create a "recording" builder
let mut builder = R1CSBuilder::<{ J::C }, F, JoltR1CSInputs>::new();

// Step 2: Call Jolt's ACTUAL constraint generation function
CS::uniform_constraints(&mut builder, memory_layout.input_start);

// Step 3: Extract what operations were recorded
let constraints = builder.get_constraints();
```

### Comparison

| Aspect | Static Parsing (syn) | Runtime Introspection (zkLean) |
|--------|---------------------|--------------------------------|
| **Input** | Source code text | Compiled Rust code |
| **Execution** | Parse + interpret AST | Actually run the code |
| **Type resolution** | You must do it | Rust compiler does it |
| **Trait dispatch** | You must resolve | Rust does it automatically |
| **Maintenance** | Breaks when code refactors | Automatically adapts |
| **Implementation** | ~5000+ lines | ~500 lines |

## 3.3 The Recording Type: MleAst

**File**: `zklean-extractor/src/mle_ast.rs`

**Core data structure**:
```rust
pub enum Node {
    Atom(Atom),           // Constants and variables
    Neg(Edge),            // Negation
    Inv(Edge),            // Multiplicative inverse
    Add(Edge, Edge),      // Addition
    Mul(Edge, Edge),      // Multiplication
    Sub(Edge, Edge),      // Subtraction
    Div(Edge, Edge),      // Division
}

pub enum Atom {
    Const(BigUint),       // Field constant
    Var(String),          // Named variable
}
```

**Key insight**: `MleAst` implements the `JoltField` trait, so it can substitute for any field type in Jolt's code.

**Operator overloading captures operations**:
```rust
// When Jolt code does: a + b
impl<const NUM_NODES: usize> std::ops::Add for MleAst<NUM_NODES> {
    type Output = Self;
    fn add(mut self, rhs: Self) -> Self::Output {
        self.binop(MleAstNode::Add, rhs);  // Records "Add" operation
        self
    }
}

// When Jolt code does: a * b
impl<const NUM_NODES: usize> std::ops::Mul for MleAst<NUM_NODES> {
    type Output = Self;
    fn mul(mut self, rhs: Self) -> Self::Output {
        self.binop(MleAstNode::Mul, rhs);  // Records "Mul" operation
        self
    }
}
```

**Example**: When Jolt code does `let x = a * b + c`:

- **Normal execution**: Computes `a * b = 15` (if a=3, b=5), then `15 + 7 = 22` (if c=7)
- **zkLean recording**: Builds AST:
```rust
MleAst::Add(
    Box::new(MleAst::Mul(
        Box::new(MleAst::Var("a")),
        Box::new(MleAst::Var("b"))
    )),
    Box::new(MleAst::Var("c"))
)
```

## 3.4 R1CS Extraction Example

**File**: `zklean-extractor/src/r1cs.rs`

```rust
impl<J: JoltParameterSet> ZkLeanR1CSConstraints<J> {
    pub fn extract() -> Self {
        // Step 1: Get input definitions
        let inputs = JoltR1CSInputs::flatten::<{ J::C }>();

        // Step 2: Create recording builder
        let memory_layout = MemoryLayout::new(&J::MEMORY_CONFIG);
        let mut r1cs_builder = R1CSBuilder::<{ J::C }, F, JoltR1CSInputs>::new();

        // Step 3: Call Jolt's ACTUAL constraint function
        CS::uniform_constraints(&mut r1cs_builder, memory_layout.input_start);

        // Step 4: Extract the recorded constraints
        let uniform_constraints = r1cs_builder.get_constraints();

        Self { inputs, uniform_constraints, ... }
    }
}
```

**What happens inside `uniform_constraints()`** (Jolt's actual code):
```rust
impl JoltRV32IMConstraints {
    fn uniform_constraints<F: JoltField>(
        builder: &mut R1CSBuilder<C, F, JoltR1CSInputs>,
        input_start: u64,
    ) {
        let pc = builder.input(JoltR1CSInputs::PC);
        let next_pc = builder.input(JoltR1CSInputs::NextPC);
        let imm = builder.input(JoltR1CSInputs::Bytecode_Imm);

        // This constraint: (next_pc - pc - 4) * jump_flag = 0
        builder.constrain_eq(
            (next_pc - pc - 4) * jump_flag,
            LC::zero()
        );
        // ... ~30 more constraints
    }
}
```

**The magic**: When `F = MleAst`, operations are **recorded** instead of computed. Recording builder captures all calls as AST nodes.

## 3.5 Property Testing Validates Extraction

**File**: `zklean-extractor/src/instruction.rs` (tests)

```rust
proptest! {
    #[test]
    fn combine_lookups(
        (instr, inputs) in arb_instruction_and_input::<ParamSet, RefField>(),
    ) {
        // Test: AST evaluation should equal direct computation
        prop_assert_eq!(
            instr.test_combine_lookups::<_, TestField>(&inputs),  // Evaluate AST
            instr.reference_combine_lookups(&inputs),             // Direct compute
        );
    }
}
```

**How it works**:
1. Generate random instruction (ADD, XOR, SLL)
2. Generate random field inputs
3. Run Jolt's function with `MleAst` (records AST)
4. Evaluate AST with concrete values
5. Run Jolt's function with `ark_bn254::Fr` (computes directly)
6. Assert both give same result

**This proves**: The AST extraction is correct!

---

# Part IV: Gnark Target Implementation

## 4.1 Gnark Circuit Structure

```go
package jolt_verifier

import "github.com/consensys/gnark/frontend"

type JoltVerifierCircuit struct {
    // Public inputs
    ProofCommitments [29]frontend.Variable `gnark:",public"`

    // Witness (private)
    UnivariateCoeffs [][]frontend.Variable
}

func (circuit *JoltVerifierCircuit) Define(api frontend.API) error {
    // These create R1CS constraints, not compute values
    x := api.Add(circuit.A, circuit.B)
    y := api.Mul(x, circuit.C)
    api.AssertIsEqual(y, circuit.Output)
    return nil
}
```

**Key distinction**: Each `api.Add()`, `api.Mul()` call **generates R1CS constraints** - it's not computing values, it's building a constraint system.

## 4.2 MleAst → Gnark Mapping

| MleAst Node | Gnark Code |
|-------------|------------|
| `Add(a, b)` | `api.Add(a, b)` |
| `Mul(a, b)` | `api.Mul(a, b)` |
| `Sub(a, b)` | `api.Sub(a, b)` |
| `Neg(a)` | `api.Neg(a)` |
| `Inv(a)` | `api.Inverse(a)` |
| `Div(a, b)` | `api.Div(a, b)` |
| `Const(v)` | `frontend.Variable(v)` |
| `Var(name)` | `circuit.{name}` |

## 4.3 New File: `to_gnark.rs`

**Create**: `zklean-extractor/src/to_gnark.rs`

```rust
use crate::mle_ast::{MleAst, MleAstNode};

pub struct GnarkGenerator;

impl GnarkGenerator {
    pub fn ast_to_gnark<const N: usize>(ast: &MleAst<N>) -> String {
        fn helper(nodes: &[Option<MleAstNode>], root: usize) -> String {
            match nodes[root] {
                Some(MleAstNode::Scalar(val)) => format!("{}", val),
                Some(MleAstNode::Var(name, idx)) => format!("circuit.{}_{}", name, idx),
                Some(MleAstNode::Add(l, r)) => {
                    format!("api.Add({}, {})",
                            helper(nodes, root - l),
                            helper(nodes, root - r))
                }
                Some(MleAstNode::Mul(l, r)) => {
                    format!("api.Mul({}, {})",
                            helper(nodes, root - l),
                            helper(nodes, root - r))
                }
                Some(MleAstNode::Sub(l, r)) => {
                    format!("api.Sub({}, {})",
                            helper(nodes, root - l),
                            helper(nodes, root - r))
                }
                Some(MleAstNode::Neg(idx)) => {
                    format!("api.Neg({})", helper(nodes, root - idx))
                }
                Some(MleAstNode::Inv(idx)) => {
                    format!("api.Inverse({})", helper(nodes, root - idx))
                }
                Some(MleAstNode::Div(l, r)) => {
                    format!("api.Div({}, {})",
                            helper(nodes, root - l),
                            helper(nodes, root - r))
                }
                None => panic!("uninitialized node"),
            }
        }
        helper(&ast.nodes, ast.root)
    }

    pub fn generate_r1cs_circuit(constraints: &ZkLeanR1CSConstraints) -> String {
        let mut output = String::new();

        // Package and imports
        output.push_str("package jolt_verifier\n\n");
        output.push_str("import \"github.com/consensys/gnark/frontend\"\n\n");

        // Circuit struct
        output.push_str("type JoltR1CSCircuit struct {\n");
        for input in &constraints.inputs.fields {
            output.push_str(&format!("    {} frontend.Variable `gnark:\",secret\"`\n", input.name));
        }
        output.push_str("}\n\n");

        // Define method
        output.push_str("func (circuit *JoltR1CSCircuit) Define(api frontend.API) error {\n");
        for constraint in &constraints.uniform_constraints {
            output.push_str(&format!("    {} := {}\n",
                                     constraint.target,
                                     Self::ast_to_gnark(&constraint.expr)));
        }
        output.push_str("    return nil\n}\n");

        output
    }
}
```

## 4.4 Architecture: Shared Pipeline

```
┌─────────────────────────────────────────────────────────┐
│              Jolt Rust Implementation                    │
└──────────────┬──────────────────────────────────────────┘
               ↓
┌─────────────────────────────────────────────────────────┐
│        zkLean Extraction Infrastructure                  │
│        (REUSE - Already Implemented)                     │
│                                                          │
│  • Recording builders (R1CSBuilder, etc.)                │
│  • MLE AST representation                                │
│  • Constraint extraction                                 │
│  • Instruction extraction                                │
└──────────────┬──────────────────────────────────────────┘
               ↓
         [Extracted ASTs]
               ↓
        ┌──────┴──────┐
        ↓             ↓
┌──────────────┐ ┌──────────────┐
│  Lean4 Gen   │ │  Gnark Gen   │
│  (existing)  │ │  (NEW - us)  │
└──────┬───────┘ └──────┬───────┘
       ↓                ↓
   [Lean4 code]   [Gnark circuit]
       ↓                ↓
[Formal proof]   [Groth16 proof]
```

---

# Part V: Frontend vs Verifier Extraction

## 5.1 What zkLean Extracts (Frontend)

**Frontend** = Constraint generation, not verification

```rust
// What zkLean extracts
CS::uniform_constraints(&mut builder, ...);  // Generates constraints
instruction.combine_lookups(...);             // Combines lookup operations
```

**Output**: Specifications of what constraints exist.

## 5.2 What We Need to Extract (Verifier)

**Verifier** = Verification logic that checks proofs

```rust
// What we need to extract
sumcheck::verify_round(&proof, &transcript);  // Verifies sumcheck round
opening::verify(&commitment, &opening);       // Verifies polynomial opening
```

**Output**: Circuit that implements verification.

## 5.3 Same Pattern, Different Target

**Recording transcript** (instead of recording builder):
```rust
// Normal transcript (used in actual verification)
impl Transcript {
    fn read_challenge(&mut self) -> FieldElement {
        self.hasher.finalize()  // Actually hash
    }
}

// Recording transcript (for extraction)
impl RecordingTranscript {
    fn read_challenge(&mut self) -> FieldElement {
        self.recorded_ops.push(Operation::ReadChallenge);
        FieldElement::new_symbolic("challenge")  // Return symbolic
    }
}
```

**Then run verifier**:
```rust
let mut recording_transcript = RecordingTranscript::new();
let fake_proof = generate_dummy_proof();

// Run actual verifier with recording types
verify_sumcheck(&mut recording_transcript, &fake_proof);

// Extract recorded operations
let operations = recording_transcript.get_operations();
```

---

# Part VI: Jolt Verification Stages

## 6.1 Stage Overview

| Stage | Component | Description | Extractable? |
|-------|-----------|-------------|--------------|
| 1 | Spartan outer sumcheck | Verifies R1CS satisfaction | Yes |
| 2 | Spartan product sumcheck | Memory checking | Yes |
| 3 | Spartan matrix evaluation | Final R1CS evaluation | Yes |
| 4 | Shout/Twist sumchecks | Lookup and memory arguments | Yes |
| 5 | Batched Dory opening | Polynomial commitment verification | Partial |
| 6 | Hyrax verification | Hint verification (PR #975) | Manual |

## 6.2 Hybrid Approach

Some components require manual implementation:

**Extractable** (automatic via zkLean):
- Field arithmetic
- Polynomial evaluation
- Sumcheck rounds
- R1CS constraint verification

**Manual** (Gnark gadgets):
- Elliptic curve operations (Grumpkin)
- Pairing checks
- Hash functions (Poseidon/Keccak)
- MSM operations

## 6.3 Verification DAG Structure

Jolt's proof is structured as a DAG of sumchecks:

```
Stage 1: Initial sumchecks (can start immediately)
    ↓
Stage 2: Sumchecks depending on stage 1 outputs
    ↓
Stage 3: Sumchecks depending on stage 2 outputs
    ↓
Stage 4: Final sumchecks before opening proof
    ↓
Stage 5: Batched opening proof (necessarily last)
```

**Key concept**: Virtual vs Committed Polynomials
- **Virtual polynomial**: Part of witness never committed directly - claimed evaluation proven by subsequent sumcheck
- **Committed polynomial**: Explicitly committed using PCS - evaluation proven via opening proof

---

# Part VII: Technical Challenges and Solutions

## 7.1 Control Flow Handling

**Problem**: Circuits are static; Rust verifier has dynamic control flow (loops)

**Solution**: Unroll loops with known bounds
```rust
// Rust: dynamic loop
for proof_elem in proof.iter() {
    verify_element(proof_elem);
}

// Circuit: fixed unrolling (e.g., for 40 elements)
verify_element(proof[0]);
verify_element(proof[1]);
// ... up to proof[39]
```

## 7.2 Witness vs Public Input

**Problem**: Gnark needs clear separation of public inputs vs private witness

**Solution**: Annotate based on proof structure

**Public** (commitments, challenges):
- Polynomial commitments (~29 elements)
- Fiat-Shamir challenges (derived from commitments)
- Public program outputs

**Witness** (private):
- Polynomial coefficients (univariate round messages)
- Intermediate computation values
- Prover hints

## 7.3 Field Type

**Target**: BN254 scalar field (same as Jolt's native field)
- Gnark has native BN254 support
- Same field means direct value transfer (no conversion needed)

## 7.4 Cryptographic Primitives (Stage 5-6)

**Problem**: Dory/Hyrax verification involves elliptic curve operations

**Solution**: Use existing Gnark gadgets + custom implementation
- Gnark has BN254 pairing gadgets
- May need Grumpkin gadgets for Stage 6 (Hyrax)
- MSM operations via Gnark's `ecc` package

---

# Part VIII: Implementation Plan

## 8.1 Phase 1: Proof of Concept (Current)

**Scope**: Stage 1 verification (Spartan outer sumcheck only)

**Deliverables**:
1. Fork/extend zkLean extractor to output Gnark instead of Lean4
2. Extract Stage 1 verification logic
3. Generate corresponding Gnark circuit
4. Validate via differential testing

**Success Criteria**:
- Generated circuit compiles
- Differential testing passes (100+ random cases)
- Constraint count measured (<1M for Stage 1)

## 8.2 Phase 2: Full Verifier Extraction

**Scope**: All extractable verification stages (1-4)

**Components**:
1. Extend recording types to verifier code
2. Extract all sumcheck verification logic
3. Generate complete Gnark circuit
4. Measure total constraint count

## 8.3 Phase 3: Production Integration

**Scope**: Complete Groth16 wrapper with manual components

**Components**:
1. Implement Stage 5-6 manual gadgets
2. End-to-end pipeline integration
3. Deploy Solidity verifier contract
4. Validate: guest program → Jolt proof → Groth16 proof → on-chain

---

# Part IX: Code Locations

## 9.1 Existing Infrastructure

```
jolt/
├── zklean-extractor/           # Extraction tool (target for modification)
│   └── src/
│       ├── main.rs             # Entry point - orchestrates extraction
│       ├── mle_ast.rs          # Core AST recording infrastructure
│       ├── r1cs.rs             # R1CS constraint extraction
│       ├── instruction.rs      # Instruction MLE extraction
│       ├── lookups.rs          # Lookup table extraction
│       ├── lean_tests.rs       # Property test generation
│       └── modules.rs          # Module generation utilities
├── jolt-core/
│   └── src/
│       ├── zkvm/
│       │   ├── dag/            # Verification DAG (jolt_dag.rs, state_manager.rs)
│       │   ├── r1cs/           # R1CS constraints (constraints.rs)
│       │   ├── instruction/    # RISC-V instruction definitions
│       │   └── ...
│       ├── subprotocols/
│       │   ├── sumcheck.rs     # Sumcheck implementation
│       │   ├── twist.rs        # Memory checking
│       │   └── shout/          # Lookup arguments
│       └── field/              # Field types
└── docs/Groth16/               # Project documentation
    ├── Groth16_Conversion_Scope.md
    ├── zkLean_Infrastructure_Reuse.md
    └── Partial_Transpilation_Exploration.md
```

## 9.2 New Files to Create

```
jolt/
├── zklean-extractor/
│   └── src/
│       └── to_gnark.rs         # NEW: AST → Gnark code generator
└── groth16-circuit/            # NEW: Generated Gnark circuit
    ├── jolt_verifier.go        # Generated Go code
    ├── stage1.go               # Stage 1 verification circuit
    └── gadgets/                # Manual elliptic curve gadgets
        └── grumpkin.go
```

---

# Part X: Testing Strategy

## 10.1 Differential Testing

**Approach**: Compare Rust verifier outputs with Gnark circuit outputs

```
1. Generate random Jolt proofs
2. Run Rust verifier → capture all intermediate values
3. Run Gnark circuit with same inputs
4. Compare all intermediate values
5. Verify final accept/reject matches
```

## 10.2 Property Testing (from zkLean)

**Already implemented**: zkLean's property tests validate AST extraction
- Random instruction + random inputs
- AST evaluation == direct computation
- Run with `cargo test -p zklean-extractor`

## 10.3 End-to-End Testing

**Full pipeline test**:
```
1. Compile guest program (e.g., Fibonacci)
2. Generate Jolt proof
3. Run Groth16 wrapper prover
4. Verify on simulated EVM
5. Deploy to testnet and verify on-chain
```

---

# Part XI: Immediate Action Items

## 11.1 This Week

1. **Run zkLean yourself** (1 day)
   ```bash
   cd /Users/home/dev/parti/cryptography/zkVMs/jolt
   cargo run --release -p zklean-extractor -- -p ./test-zklean-output
   cat ./test-zklean-output/src/Jolt/R1CS.lean  # See actual Lean4 output
   ```

2. **Study MleAst** (1 day)
   - Read `zklean-extractor/src/mle_ast.rs`
   - Understand Node enum and operator overloading
   - Trace how ASTs are built during extraction

3. **Prototype `to_gnark.rs`** (2-3 days)
   - Create `zklean-extractor/src/to_gnark.rs`
   - Implement `ast_to_gnark()` function
   - Test with single R1CS constraint first

## 11.2 Next 2 Weeks

4. **Generate R1CS circuit**
   - Complete Gnark generator for all R1CS constraints
   - Verify generated Go code compiles
   - Run differential tests

5. **Extend to verifier logic**
   - Design recording types for verifier
   - Extract sumcheck verification
   - Generate Gnark circuit for Stage 1

## 11.3 Key Questions to Resolve

1. How to structure public inputs vs witness in Gnark circuit?
2. What constraint count is acceptable? (Target: <10M total)
3. How to handle polynomial commitments in circuit?
4. What manual gadgets are needed for Stage 5-6?
5. Where does Groth16 proving happen? (Off-chain service? User's machine?)

---

# Part XII: Glossary

| Term | Definition |
|------|------------|
| **R1CS** | Rank-1 Constraint System - format required by Groth16 |
| **PLONK** | Different arithmetization format (not compatible with Groth16) |
| **MLE** | Multilinear Extension - polynomial representation over Boolean hypercube |
| **MleAst** | zkLean's AST type that records operations by implementing JoltField |
| **Sumcheck** | Interactive protocol for verifying polynomial sums |
| **Gnark** | Go library for zkSNARK circuit development |
| **zkLean** | Tool for extracting Jolt logic to Lean4 (basis for our approach) |
| **AST** | Abstract Syntax Tree - structured representation of operations |
| **Witness** | Private inputs to a zkSNARK circuit |
| **Public inputs** | Values visible to both prover and verifier |
| **Dory** | Jolt's polynomial commitment scheme |
| **Hyrax** | Alternative PCS used in Stage 6 optimization (PR #975) |
| **Spartan** | Jolt's R1CS backend (transparent SNARK) |
| **Shout** | Jolt's lookup argument (v0.2.0+) |
| **Twist** | Jolt's memory checking protocol (v0.2.0+) |
| **BN254** | Elliptic curve used for Groth16 on Ethereum |
| **Grumpkin** | Curve used in Stage 6 Hyrax verification |
| **Runtime introspection** | Execute code with recording types instead of parsing source |
| **Recording builder** | Builder that logs operations instead of computing them |

---

# Part XIII: References

## 13.1 Internal Documentation

- [Groth16_Conversion_Scope.md](./Groth16_Conversion_Scope.md) - Project scope and deliverables
- [zkLean_Infrastructure_Reuse.md](./zkLean_Infrastructure_Reuse.md) - Infrastructure details
- [Partial_Transpilation_Exploration.md](./Partial_Transpilation_Exploration.md) - Transpilation theory
- [CLAUDE.md](../../CLAUDE.md) - Jolt architecture overview
- [01_Jolt_Theory_Enhanced.md](../01_Jolt_Theory_Enhanced.md) - Mathematical foundations

## 13.2 External Resources

### Industry Approaches
- **RISC Zero**: [groth16_proof](https://github.com/risc0/risc0/tree/main/groth16_proof)
  - [stark_verify.circom](https://github.com/risc0/risc0/blob/main/groth16_proof/groth16/stark_verify.circom) - 55.7MB circuit
  - [Trusted Setup Ceremony](https://dev.risczero.com/api/trusted-setup-ceremony)
- **SP1**: [recursion/gnark-ffi](https://github.com/succinctlabs/sp1/tree/dev/crates/recursion/gnark-ffi)
  - [SP1 Testnet Launch](https://blog.succinct.xyz/sp1-testnet/)
  - [sp1-recursion-gnark-ffi](https://crates.io/crates/sp1-recursion-gnark-ffi)

### Target Framework
- [Gnark Documentation](https://docs.gnark.consensys.io/)
- [Gnark Benchmarks](https://blog.celer.network/2023/08/04/the-pantheon-of-zero-knowledge-proof-development-frameworks/)
- [Gnark GitHub](https://github.com/ConsenSys/gnark)

### Jolt Resources
- [Jolt Book](https://jolt.a16zcrypto.com/)
- [Jolt Paper](https://eprint.iacr.org/2023/1217.pdf)
- [PR #975 - Hint Mechanism](https://github.com/a16z/jolt/pull/975)
- [PR #1060 - zkLean Extractor](https://github.com/a16z/jolt/pull/1060)

---

**Document Status**: Active reference document for LLM context
**Total Length**: ~750 lines covering all aspects of the project
**Last Updated**: 2025-11-25
