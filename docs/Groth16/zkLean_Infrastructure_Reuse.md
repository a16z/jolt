# Reusing zkLean Infrastructure for Gnark Transpilation

**Date**: 2025-11-18 (Updated: 2025-11-25)
**Context**: Jolt team feedback on Groth16 conversion approach
**Goal**: Understand zkLean's runtime introspection approach and how to adapt it for Gnark generation

---

## 1. Executive Summary

**Jolt team guidance**:
- ✅ Avoid hand-written circuits (unlike SP1/Risc0)
- ✅ Use runtime introspection (zkLean approach) instead of static parsing
- ✅ Reuse zkLean infrastructure where possible for shared maintenance

**Key insight**: zkLean already extracts Jolt's verification logic into a structured format. We can adapt this extraction to generate Gnark circuits instead of Lean4 code.

### 1.1 Industry Validation (Research Update 2025-11-25)

Our approach is **validated by SP1's architecture**:

| Aspect | SP1 | Jolt (proposed) |
|--------|-----|-----------------|
| **Problem** | Evolving recursion protocol | Evolving verification protocol |
| **Solution** | Custom DSL → Gnark compiler | zkLean AST → Gnark generator |
| **Automation** | Semi-automatic | Fully automatic |
| **Infrastructure** | Built custom compiler | Reuse existing zkLean |

**SP1's key insight**: They faced the same maintenance problem with an evolving protocol and solved it by **compiling from a higher-level representation** rather than hand-writing circuits. Their Recursion DSL compiler generates Gnark circuits automatically.

**RISC Zero's contrasting approach**: Hand-written 55MB Circom file (`stark_verify.circom`). Works because their STARK verification is **stable** - but would be unmaintainable for Jolt's evolving protocol.

Sources:
- [SP1 Testnet Launch](https://blog.succinct.xyz/sp1-testnet/) - Recursion compiler architecture
- [sp1-recursion-gnark-ffi](https://crates.io/crates/sp1-recursion-gnark-ffi) - FFI interface
- [RISC Zero stark_verify.circom](https://github.com/risc0/risc0/blob/main/groth16_proof/groth16/stark_verify.circom)

### 1.2 What you now understand

- ✅ zkLean uses recording types (`MleAst`) that implement `JoltField` trait
- ✅ Instead of parsing Rust syntax, zkLean *runs* Jolt's actual functions with recording types
- ✅ Operator overloading captures operations as AST nodes (e.g., `a + b` records "Add" node)
- ✅ The entire extraction infrastructure is reusable - just need to add Gnark code generator
- ✅ Property tests validate that AST evaluation matches direct computation
- ✅ **SP1 validates the "compile from higher representation" pattern**
- ✅ **Our approach is more efficient than SP1's** - we reuse zkLean instead of building a custom compiler

### 1.3 What you need to do

1. Run zkLean yourself to see actual output (Section 13.1)
2. Create `zklean-extractor/src/to_gnark.rs` (skeleton provided in Section 12.2)
3. Convert `MleAst` nodes to Gnark API calls (example: `Add` → `api.Add(...)`)
4. Test with single constraint before extending to full verifier

---

## 2. What is zkLean?

### 2.1 Purpose

zkLean (PR #1060) extracts Jolt's frontend semantics into Lean4 for **formal verification**.

**Goal**: Prove mathematically that Jolt's R1CS constraints and instruction lookups are correct.

**Why it exists**:
- Hand-writing formal specs diverges from implementation
- Automatic extraction keeps specs in sync with Rust code
- Changes to Jolt automatically reflected in Lean4 specs

### 2.2 What zkLean Extracts

From `zklean-extractor/src/main.rs`:

```rust
let modules: Vec<Box<dyn AsModule>> = vec![
    Box::new(ZkLeanR1CSConstraints::<ParameterSet>::extract()),
    Box::new(ZkLeanSubtables::<MleAst<16000>, ParameterSet>::extract()),
    Box::new(ZkLeanInstructions::<ParameterSet>::extract()),
    Box::new(ZkLeanLookupCases::<ParameterSet>::extract()),
];
```

**Four modules**:
1. **R1CS Constraints**: The ~30 constraints applied to every VM cycle
2. **Subtables**: MLE evaluations for lookup tables (4-bit ADD, XOR, etc.)
3. **Instructions**: RISC-V instruction specifications
4. **Lookup Cases**: Circuit flags and decomposition logic

### 2.3 Target Output

zkLean generates Lean4 code for formal verification:

```lean
-- Example output
def uniform_jolt_constraints [ZKField f] (jolt_inputs : JoltR1CSInputs f) : ZKBuilder f PUnit := do
  constrainR1CS
    (expression for A)
    (expression for B)
    (expression for C)
  -- ... for all 30 constraints
```

**Our goal**: Generate Gnark Go code instead of Lean4.

---

## 3. Runtime Introspection vs Static Parsing

### 3.1 What You've Been Doing (Static Parsing)

**Your toy examples with `syn`**:
```rust
// Read Rust source code as text
let rust_code = read_file("simple.rs")?;

// Parse with syn
let ast = syn::parse_file(rust_code)?;

// Walk AST, extract operations
for item in ast.items {
    // Build IR from syntax tree
}
```

**Limitations**:
- Must resolve Rust types (generics, traits, lifetimes)
- Must understand trait implementations
- Must interpret control flow
- Breaks when Jolt refactors code

### 3.2 What zkLean Does (Runtime Introspection)

**zkLean's approach**:
```rust
// Don't read source code!
// Instead, RUN Jolt's actual functions with special types

// Step 1: Create a "recording" builder
let mut builder = R1CSBuilder::<{ J::C }, F, JoltR1CSInputs>::new();

// Step 2: Call Jolt's ACTUAL constraint generation function
CS::uniform_constraints(&mut builder, memory_layout.input_start);

// Step 3: Extract what operations were recorded
let constraints = r1cs_builder.get_constraints();
```

**Key differences**:
| Aspect | Static Parsing (syn) | Runtime Introspection (zkLean) |
|--------|---------------------|--------------------------------|
| **Input** | Source code text | Compiled Rust code |
| **Execution** | Parse + interpret AST | Actually run the code |
| **Type resolution** | You must do it | Rust compiler does it |
| **Trait dispatch** | You must resolve | Rust does it automatically |
| **Output** | Extracted from syntax | Recorded from execution |

---

## 4. How zkLean's Runtime Introspection Works

### 4.1 The Core Technique: Recording Types

**The magic**: Implement the same traits as Jolt's normal types, but **record operations** instead of computing.

#### Example: Normal vs Recording Builder

**Normal R1CS builder** (used in actual proving):
```rust
impl R1CSBuilder {
    fn mul(&mut self, a: Variable, b: Variable) -> Variable {
        // Actually compute a * b in the field
        let result = a.value * b.value;

        // Add R1CS constraint: a * b = result
        self.constraints.push(Constraint::Mul(a, b, result));

        result
    }
}
```

**zkLean's recording builder** (used for extraction):
```rust
impl R1CSBuilder {
    fn mul(&mut self, a: Variable, b: Variable) -> Variable {
        // DON'T compute! Just RECORD the operation
        self.recorded_ops.push(MleAst::Mul(
            Box::new(a.to_ast()),
            Box::new(b.to_ast())
        ));

        // Return dummy variable (so code keeps running)
        Variable::new_symbolic("temp_mul_result")
    }
}
```

### 4.2 What Gets Recorded: Abstract Syntax Trees (ASTs)

zkLean builds **Abstract Syntax Trees** representing the operations:

```rust
pub enum MleAst<const N: usize> {
    Const(F),
    Var(String),
    Add(Box<MleAst<N>>, Box<MleAst<N>>),
    Mul(Box<MleAst<N>>, Box<MleAst<N>>),
    // ... other operations
}
```

**Example**: When Jolt code does `let x = a * b + c`:

**Normal execution**:
- Computes `a * b = 15` (if a=3, b=5)
- Computes `15 + 7 = 22` (if c=7)

**zkLean recording**:
```rust
MleAst::Add(
    Box::new(MleAst::Mul(
        Box::new(MleAst::Var("a")),
        Box::new(MleAst::Var("b"))
    )),
    Box::new(MleAst::Var("c"))
)
```

This AST is later converted to Lean4 (or in our case, Gnark).

### 4.3 Real Example from zkLean

**Location**: `zklean-extractor/src/r1cs.rs`

```rust
pub fn extract() -> Self {
    // Get all witness variables
    let inputs = JoltR1CSInputs::flatten::<{ J::C }>();

    // Create recording builder
    let mut r1cs_builder = R1CSBuilder::<{ J::C }, F, JoltR1CSInputs>::new();

    // Call Jolt's ACTUAL constraint generation function
    // This function calls builder.add(), builder.mul(), etc.
    // Our builder RECORDS these calls as ASTs
    CS::uniform_constraints(&mut r1cs_builder, memory_layout.input_start);

    // Extract the recorded constraints
    let uniform_constraints = r1cs_builder.get_constraints();

    Self {
        inputs,
        uniform_constraints,
        non_uniform_constraints,
        // ...
    }
}
```

**What happens**:
1. Jolt's `uniform_constraints()` function runs
2. It calls methods like `builder.mul(x, y)`, `builder.add(a, b)`
3. Recording builder captures these as AST nodes
4. Returns list of all operations that happened

**No parsing needed!** Rust compiler handles all type resolution, trait dispatch, etc.

---

## 5. What Can We Reuse from zkLean?

### 5.1 Direct Reuse: Extraction Infrastructure

**Already implemented in zkLean**:
- ✅ Recording builder pattern
- ✅ MLE AST representation
- ✅ R1CS constraint extraction
- ✅ Instruction lookup extraction
- ✅ Subtable MLE extraction

**Location**: `zklean-extractor/src/`
```
zklean-extractor/
├── src/
│   ├── main.rs           # Orchestrates extraction
│   ├── r1cs.rs          # R1CS constraint extraction
│   ├── subtable.rs      # Lookup table MLE extraction
│   ├── instruction.rs   # Instruction specification extraction
│   ├── flags.rs         # Circuit flags extraction
│   └── mle_ast.rs       # AST representation
```

### 5.2 What We Need to Add: Gnark Code Generation

zkLean extracts to Lean4. We need to:

1. **Reuse zkLean's extraction** (same ASTs, same recording)
2. **Write new generator** (AST → Gnark instead of AST → Lean4)

**New file**: `zklean-extractor/src/gnark_generator.rs`

```rust
pub fn generate_gnark(constraints: &[Constraint]) -> String {
    let mut output = String::new();

    // Generate Gnark circuit struct
    output.push_str("type JoltVerifierCircuit struct {\n");
    // ... generate fields

    // Generate Define() method
    output.push_str("func (circuit *JoltVerifierCircuit) Define(api frontend.API) error {\n");

    for constraint in constraints {
        // Convert zkLean's AST to Gnark code
        output.push_str(&ast_to_gnark(constraint));
    }

    output.push_str("    return nil\n}\n");

    output
}
```

### 5.3 Architecture: Shared Pipeline

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

### 5.4 Concrete Reuse Plan

**Phase 1: Understand zkLean's AST format**
- Read `zklean-extractor/src/mle_ast.rs`
- Understand `MleAst<N>` enum
- See how constraints are represented

**Phase 2: Add Gnark target to zkLean**
- Keep existing Lean4 generator
- Add parallel Gnark generator
- Reuse same extraction infrastructure

**Phase 3: Extend to verifier (not just frontend)**
- zkLean extracts frontend (R1CS constraints, lookups)
- We need verifier (sumcheck verification, opening proofs)
- Apply same recording pattern to verifier code

---

## 6. Key Differences: Frontend vs Verifier Extraction

### 6.1 What zkLean Extracts (Frontend)

**Frontend** = Constraint generation, not verification

```rust
// What zkLean extracts
CS::uniform_constraints(&mut builder, ...);  // Generates constraints
instruction.combine_lookups(...);             // Combines lookup operations
```

**Output**: Specifications of what constraints exist.

### 6.2 What We Need to Extract (Verifier)

**Verifier** = Verification logic that checks proofs

```rust
// What we need to extract
sumcheck::verify_round(&proof, &transcript);  // Verifies sumcheck round
opening::verify(&commitment, &opening);       // Verifies polynomial opening
```

**Output**: Circuit that implements verification.

### 6.3 Same Pattern, Different Target

**Good news**: Same recording pattern works!

**Recording transcript** (instead of recording builder):
```rust
// Normal transcript (used in actual verification)
impl Transcript {
    fn read_challenge(&mut self) -> FieldElement {
        // Actually hash and return challenge
        self.hasher.finalize()
    }
}

// Recording transcript (for extraction)
impl RecordingTranscript {
    fn read_challenge(&mut self) -> FieldElement {
        // Record that a challenge was read
        self.recorded_ops.push(Operation::ReadChallenge);

        // Return symbolic value
        FieldElement::new_symbolic("challenge")
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

## 7. Practical Implementation Plan

### 7.1 Step 1: Study zkLean Codebase (1-2 days)

**Read these files in order**:
1. `zklean-extractor/README.md` - Overview
2. `zklean-extractor/src/mle_ast.rs` - AST representation
3. `zklean-extractor/src/r1cs.rs` - Extraction example
4. `zklean-extractor/src/to_lean.rs` - Code generation example

**Goal**: Understand how ASTs are built and consumed.

### 7.2 Step 2: Prototype Gnark Generator (3-5 days)

**New file**: `zklean-extractor/src/to_gnark.rs`

```rust
use crate::mle_ast::MleAst;

pub fn ast_to_gnark<const N: usize>(ast: &MleAst<N>) -> String {
    match ast {
        MleAst::Const(val) => format!("{}", val),
        MleAst::Var(name) => format!("circuit.{}", name),
        MleAst::Add(left, right) => {
            format!("api.Add({}, {})",
                    ast_to_gnark(left),
                    ast_to_gnark(right))
        }
        MleAst::Mul(left, right) => {
            format!("api.Mul({}, {})",
                    ast_to_gnark(left),
                    ast_to_gnark(right))
        }
        // ... handle other cases
    }
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

    // Generate constraints
    for constraint in &constraints.uniform_constraints {
        output.push_str(&format!("    {} := {}\n",
                                 constraint.target,
                                 ast_to_gnark(&constraint.expr)));
    }

    output.push_str("    return nil\n}\n");

    output
}
```

**Test with**:
```bash
cd zklean-extractor
cargo run -- --target gnark
```

### 7.3 Step 3: Extend to Verifier (1-2 weeks)

**New recording types for verifier**:
1. `RecordingTranscript` - Records Fiat-Shamir challenges
2. `RecordingPolynomial` - Records polynomial evaluations
3. `RecordingCommitment` - Records commitment verifications

**Pattern** (same as zkLean):
```rust
// Implement same trait, record operations
impl Polynomial for RecordingPolynomial {
    fn evaluate(&self, point: &FieldElement) -> FieldElement {
        self.recorded_ops.push(PolyEval {
            poly: self.id.clone(),
            point: point.clone(),
        });
        FieldElement::new_symbolic("poly_eval_result")
    }
}
```

### 7.4 Step 4: Integration (1 week)

**Add to zkLean**:
```rust
// zklean-extractor/src/main.rs
let target = std::env::args().nth(1).unwrap_or("lean".to_string());

match target.as_str() {
    "lean" => {
        // Existing Lean4 generation
        to_lean::generate(&modules)?;
    }
    "gnark" => {
        // New Gnark generation
        to_gnark::generate(&modules)?;
    }
    _ => panic!("Unknown target: {}", target)
}
```

---

## 8. Advantages of Reusing zkLean

### 8.1 Shared Maintenance

**When Jolt changes**:
- zkLean contributors update recording builders
- Both Lean4 AND Gnark generation automatically benefit
- No duplicate maintenance work

### 8.2 Community Support

Jolt team mentioned:
> "we expect that [zkLean] will be getting cycles from other contributors"

**Benefits**:
- Bug fixes in extraction benefit both projects
- Improvements to recording builders benefit both
- Larger community reviewing the approach

### 8.3 Proven Approach

zkLean already demonstrates:
- ✅ Recording builders work for complex Jolt code
- ✅ AST representation captures all operations
- ✅ Extraction stays in sync with Rust changes
- ✅ Pattern is maintainable

**We're not pioneering a new approach** - we're adding a new target to a proven system.

### 8.4 Type Safety

zkLean's ASTs are strongly typed:
```rust
pub enum MleAst<const N: usize> {
    // N is the number of variables in MLE
    // Compile-time guarantee about AST structure
}
```

**Benefits**:
- Rust compiler catches errors in AST construction
- Impossible to build invalid ASTs
- Type checking helps generate correct Gnark code

---

## 9. Differences to Account For

### 9.1 Lean4 vs Gnark

| Aspect | Lean4 (zkLean target) | Gnark (our target) |
|--------|----------------------|-------------------|
| **Purpose** | Formal verification | Circuit proving |
| **Execution** | Theorem prover | R1CS constraints |
| **Variables** | Mathematical expressions | `frontend.Variable` |
| **Operations** | Pure functions | `api.Add`, `api.Mul` |
| **Output** | Proof of correctness | Groth16 proof |

### 9.2 Frontend vs Verifier

**zkLean extracts**:
- Constraint generation (prover-side)
- What constraints should exist
- Specifications for formal proof

**We need**:
- Verification logic (verifier-side)
- How to check proofs
- Circuit that implements checking

**Same technique, different target code**:
- zkLean records: `builder.mul(a, b)` → constraint specification
- We record: `verify_round(proof)` → verification circuit

---

## 10. Questions for Jolt Team

Before proceeding, clarify:

### 10.1 zkLean Integration

**Q1**: Should we:
- A) Fork zkLean and add Gnark target independently?
- B) Contribute Gnark generator directly to zkLean repo?
- C) Use zkLean as library, separate repo for Gnark generation?

**Recommendation**: Option B (contribute to zkLean) for maximum code sharing.

### 10.2 Verifier Extraction Scope

**Q2**: zkLean extracts frontend (constraints/lookups). For verifier extraction, should we:
- A) Extend zkLean infrastructure to cover verifier?
- B) Create parallel extractor just for verifier?
- C) Wait for zkLean to expand scope first?

**Recommendation**: Option A (extend zkLean) to keep everything unified.

### 10.3 R1CS Constraint Budget

**Q3**: Expected fully-optimized R1CS count?
- Jolt team mentioned "millions of constraints"
- Need specifics for circuit design (memory allocation, optimization targets)
- Does this include Stage 6 Dory verification with hints?

**Context**: Groth16 proving time and memory scale with constraint count. Need to know target to design efficiently.

### 10.4 Proof Pipeline

**Q4**: Full pipeline architecture?
```
Guest Program
    ↓
[Jolt Prover]
    ↓
Jolt Proof (71KB)
    ↓
[Groth16 Wrapper Prover - where does this run?]
    ↓
Groth16 Proof (260 bytes)
    ↓
[On-chain Verifier]
```

Where does Groth16 proving happen? (Off-chain service? User's machine? Decentralized network?)

---

## 11. Concrete Code Walkthrough

Let me show you exactly how the zkLean extraction works with real code examples from the codebase.

### 11.1 The Recording Type (`MleAst`)

**File**: `zklean-extractor/src/mle_ast.rs`

**Core data structure**:
```rust
pub enum MleAstNode {
    Scalar(i128),              // Constant value
    Var(char, usize),          // Variable: register name + index
    Neg(usize),                // Negation
    Inv(usize),                // Multiplicative inverse
    Add(usize, usize),         // Addition
    Mul(usize, usize),         // Multiplication
    Sub(usize, usize),         // Subtraction
    Div(usize, usize),         // Division
}

pub struct MleAst<const NUM_NODES: usize> {
    nodes: [Option<MleAstNode>; NUM_NODES],  // Array of nodes
    root: usize,                              // Index of root node
}
```

**Key insight**: `MleAst` implements the `JoltField` trait, so it can substitute for any field type in Jolt's code.

**Example - Operator overloading**:
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

**Display implementation** (converts AST to string):
```rust
impl<const NUM_NODES: usize> std::fmt::Display for MleAst<NUM_NODES> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match nodes[root] {
            Some(MleAstNode::Scalar(scalar)) => write!(f, "{scalar}"),
            Some(MleAstNode::Var(name, index)) => write!(f, "{name}[{index}]"),
            Some(MleAstNode::Add(lhs_idx, rhs_idx)) => {
                helper(f, nodes, root - lhs_idx)?;
                write!(f, " + ")?;
                helper(f, nodes, root - rhs_idx)?;
            }
            // ... similar for Mul, Sub, etc.
        }
    }
}
```

**Result**: If Jolt computes `x[0] + x[1] * x[2]`, the AST outputs the string `"x[0] + x[1]*x[2]"`.

### 11.2 R1CS Extraction

**File**: `zklean-extractor/src/r1cs.rs`

**The extraction function**:
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
// This is Jolt's actual R1CS constraint generation code
// But r1cs_builder records instead of computes!

impl JoltRV32IMConstraints {
    fn uniform_constraints<F: JoltField>(
        builder: &mut R1CSBuilder<C, F, JoltR1CSInputs>,
        input_start: u64,
    ) {
        // Jolt's code calls builder methods...
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

**Code generation to Lean4**:
```rust
fn pretty_print_lc<const C: usize>(inputs_struct: &str, lc: &LC) -> String {
    let terms = lc.terms()
        .iter()
        .filter_map(|term| pretty_print_term::<C>(inputs_struct, term))
        .collect::<Vec<_>>();

    match terms.len() {
        0 => "0".to_string(),
        1 => terms[0].clone(),
        _ => format!("({})", terms.join(" + ")).to_string(),
    }
}
```

**Example output** (Lean4):
```lean
def uniform_jolt_constraints [ZKField f] (jolt_inputs : JoltR1CSInputs f) : ZKBuilder f PUnit := do
  constrainR1CS
    ((jolt_inputs.NextPC - jolt_inputs.PC - 4)*jolt_inputs.Bytecode_JumpFlag)
    1
    0
  -- ... more constraints
```

### 11.3 Instruction Extraction

**File**: `zklean-extractor/src/instruction.rs`

**Extraction wrapper**:
```rust
impl<J: JoltParameterSet> ZkLeanInstruction<J> {
    fn combine_lookups<F: ZkLeanReprField>(&self, reg_name: char) -> F {
        // Create recording register
        let reg_size = self.num_lookups::<F>();
        let reg = F::register(reg_name, reg_size);  // Creates x[0], x[1], ...

        // Call Jolt's actual combine_lookups function
        self.instruction.combine_lookups(&reg, J::C, 1 << J::LOG_M)
    }
}
```

**Example**: For a 64-bit ADD with C=16 chunks and LOG_M=4:
```rust
// Jolt's actual ADD implementation does something like:
fn combine_lookups(x: &[F], C: usize, M: usize) -> F {
    let mut result = x[0];  // First chunk result
    for i in 1..C {
        result += x[i] << (4 * i);  // Shift and add
    }
    result
}

// When F = MleAst, this RECORDS the operations:
// AST becomes: x[0] + x[1]*(2^4) + x[2]*(2^8) + ...
```

**Lean4 output**:
```lean
def ADD_64_16_4 [Field f] : ComposedLookupTable f 4 16
  := mkComposedLookupTable
       #[ (ADD_4, 0), (ADD_4, 1), ..., (ADD_4, 15) ].toVector
       (fun x => x[0] + x[1]*16 + x[2]*256 + ... + x[15]*2^60)
```

### 11.4 Testing the Extraction

**File**: `zklean-extractor/src/instruction.rs` (tests)

**Property test** (uses proptest):
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

## 12. Adapting for Gnark

Now you understand zkLean's infrastructure. Here's how to adapt it for Gnark.

### 12.1 What to Reuse (Unchanged)

✅ **Entire extraction infrastructure**:
- `mle_ast.rs` - Recording type
- `r1cs.rs` - R1CS extraction
- `instruction.rs` - Instruction extraction
- `subtable.rs` - Subtable extraction
- Test suite - Validates ASTs are correct

### 12.2 What to Add (New File)

**Create**: `zklean-extractor/src/to_gnark.rs`

**Purpose**: Convert extracted ASTs to Gnark Go code instead of Lean4.

**Skeleton**:
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
}
```

### 12.3 Modify `main.rs`

**Add CLI flag**:
```rust
#[derive(Parser)]
struct Args {
    /// Generate Gnark Go code instead of Lean4
    #[arg(long, default_value_t = false)]
    gnark: bool,

    // ... existing args
}

fn main() -> Result<(), FSError> {
    let args = Args::parse();

    let modules: Vec<Box<dyn AsModule>> = vec![
        Box::new(ZkLeanR1CSConstraints::<ParameterSet>::extract()),
        // ... other modules
    ];

    if args.gnark {
        // Generate Gnark Go code
        let gnark_output = GnarkGenerator::generate_r1cs(&modules[0]);
        // Write to file...
    } else {
        // Existing Lean4 generation
        write_flat_file(&mut f, modules)?;
    }
}
```

---

## 13. Recommended Next Steps

### Immediate (This Week)

1. **Run zkLean yourself** (1 day)
   ```bash
   cd /Users/home/dev/parti/cryptography/zkVMs/jolt
   cargo run --release -p zklean-extractor -- -p ./test-zklean-output
   cat ./test-zklean-output/src/Jolt/R1CS.lean  # See actual Lean4 output
   ```

2. **Prototype Gnark generator** (2-3 days)
   - Create `zklean-extractor/src/to_gnark.rs` (use Section 12.2 skeleton)
   - Implement `ast_to_gnark()` function
   - Test with single R1CS constraint first
   - Validate output matches expected Gnark API

### Short Term (Next 2 Weeks)

3. **Propose integration to zkLean maintainers**
   - Share prototype with Jolt team
   - Get feedback on architecture
   - Plan contribution structure

4. **Extend to verifier logic**
   - Design recording types for verifier
   - Extract sumcheck verification
   - Generate Gnark circuit for Stage 1

### Medium Term (Next Month)

5. **Full verifier extraction**
   - All sumcheck stages
   - Polynomial evaluation
   - Opening proof verification

6. **Optimization and testing**
   - Differential testing (Rust vs Gnark)
   - Constraint count optimization
   - End-to-end pipeline validation

---

## 12. Success Criteria

**PoC Success**:
- ✅ Reuse zkLean's recording infrastructure
- ✅ Generate valid Gnark circuit from extracted ASTs
- ✅ Differential testing passes (Rust verifier == Gnark circuit)
- ✅ Measure constraint count for Stage 1
- ✅ Demonstrate automatic regeneration when Jolt changes

**Full Success**:
- ✅ Complete verifier extraction (all stages)
- ✅ Total constraint count < 10M (or team's target)
- ✅ End-to-end: guest program → Jolt proof → Groth16 proof → on-chain
- ✅ Integration with zkLean (shared maintenance)
- ✅ Documentation for future contributors

---

## 13. References

- **PR #1060**: zkLean extractor implementation
- **zkLean codebase**: `zklean-extractor/` directory
- **Jolt team feedback**: (this document's motivation)
- **Related docs**:
  - `Partial_Transpilation_Exploration.md` - Transpilation theory and toy examples
  - `Groth16_Conversion_Scope.md` - Project scope and options

---

## 14. Industry Research Findings (Added 2025-11-25)

### 14.1 SP1's Approach: Recursion DSL Compiler

SP1 (Succinct) faced the **same problem** we're solving: an evolving protocol that would be unmaintainable with hand-written circuits.

**Their solution**: Build a **Recursion DSL Compiler** that generates Gnark circuits automatically.

```
SP1 Architecture:
STARK proof → Recursion DSL program → Compiler → Gnark circuit → Groth16 proof
```

**Key characteristics**:
- **Semi-automatic**: DSL is hand-written, but circuit generation is automatic
- **Precompiled circuits**: "Hot start" with ~18s pure prove time
- **FFI**: `sp1-recursion-gnark-ffi` crate bridges Rust↔Go
- **Maintenance**: When protocol changes, update DSL program → circuit auto-regenerates

Sources:
- [SP1 Testnet Launch](https://blog.succinct.xyz/sp1-testnet/)
- [sp1-recursion-gnark-ffi](https://crates.io/crates/sp1-recursion-gnark-ffi)

### 14.2 RISC Zero's Approach: Manual Circom

RISC Zero uses **fully manual** hand-written Circom circuits.

**Key characteristics**:
- **Circuit file**: `stark_verify.circom` - **55.7 MB** (stored in Git LFS!)
- **Platform limitation**: x86-only (Circom witness generator uses x86 assembly)
- **Trusted setup**: Hermez rollup with 2^23 powers of tau
- **Maintenance**: Manual updates when protocol changes

**Why manual works for RISC Zero**: Their STARK verification protocol is **mature and stable**. The enormous Circom file is a one-time investment.

Sources:
- [stark_verify.circom](https://github.com/risc0/risc0/blob/main/groth16_proof/groth16/stark_verify.circom)
- [Trusted Setup Ceremony](https://dev.risczero.com/api/trusted-setup-ceremony)

### 14.3 Why Our Approach is Better

| Aspect | RISC Zero | SP1 | Jolt (zkLean) |
|--------|-----------|-----|---------------|
| **Automation** | ❌ Manual | ✅ Semi-auto | ✅ Full auto |
| **Infrastructure** | N/A | Built custom | Reuse zkLean |
| **Dev Effort** | High (circuit) | High (compiler) | Medium (generator) |
| **Protocol Coupling** | Tight | Loose | Loose |
| **Maintenance** | Manual | DSL updates | Re-run extractor |

**Our advantage**: We don't need to build a custom compiler like SP1 did. zkLean's extraction infrastructure already exists - we just add a new code generator.

### 14.4 Why Not zkLLVM?

During research, we explored [zkLLVM](https://github.com/NilFoundation/zkLLVM) as an automatic Rust-to-circuit compiler.

**Critical finding**: zkLLVM is **incompatible with Groth16**.

| Aspect | zkLLVM | Our Requirement |
|--------|--------|-----------------|
| **Output format** | PLONK (Placeholder) | R1CS (Groth16) |
| **Rust support** | Archived (Feb 2025) | Active |
| **Arithmetization** | PLONKish | R1CS |

PLONK and R1CS are fundamentally different constraint systems. Converting between them defeats the purpose of automatic generation.

**Other tools explored**:
- **Circom**: Outputs R1CS but requires manual DSL writing
- **Arkworks**: No automatic Rust→circuit tool (manual `ConstraintSynthesizer`)
- **Lurk**: Lisp-based, uses Nova (not Groth16)
- **Chiquito**: Rust DSL but targets Halo2/PLONK (not R1CS)

**Conclusion**: Runtime introspection via zkLean is the only viable automatic approach for Groth16.

### 14.5 Gnark Performance

Gnark is the right choice for circuit proving:

- **5-10× faster** proving than Arkworks and snarkjs
- **25% faster** than rapidsnark
- In-circuit recursive verifier in ~11.5k constraints
- General-purpose Go enables automatic code generation

Source: [Celer ZK Framework Comparison](https://blog.celer.network/2023/08/04/the-pantheon-of-zero-knowledge-proof-development-frameworks/)

---

**Next Action**: Study `zklean-extractor/` codebase and prototype `to_gnark.rs` module.
