# Part 1: Preprocessing Deep Dive: Connecting Theory to Code

This document expands on Part 1 of [JOLT_CODE_FLOW.md](JOLT_CODE_FLOW.md), connecting the preprocessing steps to both the theory in [Theory/Jolt.md](Theory/Jolt.md) and the actual implementation.

## Overview: What is Preprocessing?

**Preprocessing** is the one-time setup phase that happens **once per guest program** (not per execution). It's expensive but only needs to be computed once and can be cached.

Think of it as preparing the "game board" before playing:

- **Prover preprocessing**: Everything the prover needs to generate proofs
- **Verifier preprocessing**: Everything the verifier needs to check proofs (much smaller!)

## The Two Main Preprocessing Tasks

From JOLT_CODE_FLOW.md Part 1, preprocessing does:

1. **Commits to bytecode polynomial**
2. **Generates Dory SRS for polynomial commitment scheme**

Let's understand each in detail.

---

## Task 1: Committing to Bytecode Polynomial

### Theory Connection (from Theory/Jolt.md)

From [Theory/Jolt.md:124-153](Theory/Jolt.md#L124), we learned:

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

From [Theory/Jolt.md:323](Theory/Jolt.md#L323):
> **Purpose**: Prove trace instructions match committed bytecode.
> During preprocessing: bytecode decoded and committed
> During execution: trace records which instructions executed
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

**From Theory**: Dory uses a two-tiered commitment scheme for matrices (from [Theory/Dory.md:2.2](Theory/Dory.md#L112)).

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
$$C_M = \prod_{i=1}^n e\left( \sum_{j=1}^m M_{i,j} G_{1,j} + r_i H_1, G_{2,i} \right) \cdot e(H_1, H_2)^{r_{fin}}$$

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

**Why eq(r_cycle, j)?** This is the multilinear extension trick:

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

This says: "For every bytecode index $k$ that was accessed (`ra(k) ≠ 0`), the value read must equal the value written."

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

**Definition** (from Theory/Dory.md, Section 2.1):

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

**Key property** (from Theory/Dory.md, Section 1.1):

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

**Why Dory's verifier is fast** (Theory/Dory.md, Section 3.2):

Dory uses recursive halving strategy (like Bulletproofs):

- Start with size-$n$ problem
- Each round: fold vectors to half size
- After $\log_2(n)$ rounds: size-1 problem (trivial to check)

**Proof components per round**:

- 6 group elements per recursive step
- Total proof size: $6 \cdot \log_2(\sqrt{N})$ group elements
- For $N = 2^{20}$: $6 \cdot 10 = 60$ group elements ≈ 18 KB

**Verifier work**:

- $O(\log N)$ multi-exponentiations in $\mathbb{G}_T$
- Single pairing check at the end
- For $N = 2^{20}$: ~30 milliseconds

**This is why Jolt uses Dory**: Transparent + logarithmic verifier = perfect for zkVM!

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

### Dory's Matrix View of Polynomials

**Why this matters for SRS size:**

Dory represents a polynomial of N coefficients as a matrix:
```
Polynomial coefficients: [c0, c1, c2, ..., c_{N-1}]
                                    ↓
Matrix (sqrt(N) × sqrt(N)):
┌                        ┐
│ c0   c1   c2   ...    │
│ c_k  c_k+1 ...        │
│ ...                   │
└                        ┘
```

**Example**:

- Trace length T = 1024 cycles
- K = 16 (DTH_ROOT_OF_K)
- Total polynomial size = 16 * 1024 = 16384 = 2^14
- Matrix dimensions = sqrt(16384) × sqrt(16384) = 128 × 128

**SRS size**: Dory needs ~128 G1 points and ~128 G2 points.

- Much smaller than naive approach (would need 16384 points!)
- This is why Dory is efficient

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
```
Matrix size: √(16 × 65,536) = √1,048,576 = 1,024

ProverSetup = {
  \Gamma_1: 1,024 points in G_1 (each ~48 bytes) = 49 KB
  \Gamma_2: 1,024 points in G_2 (each ~96 bytes) = 98 KB
  H_1: 1 point in G_1 = 48 bytes
  H_2: 1 point in G_2 = 96 bytes
  Total: ~147 KB
}
```

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
- Dory's logarithmic verifier only accesses $O(\log N)$ generators per verification

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

When proving starts (see JOLT_CODE_FLOW.md Part 3):

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