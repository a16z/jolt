# Jolt Verifier: The Five Properties

## Quick Overview

| Property | What It Proves | Protocol | Key Committed Data |
|----------|----------------|----------|-------------------|
| **1. Bytecode** | Correct instruction fetch from program memory | Shout | Read address selectors |
| **2. Instruction** | CPU operations compute correct outputs | Shout + R1CS | Lookup selectors |
| **3. Registers** | Register reads return most recent writes | Twist | Increments |
| **4. RAM** | Memory reads return most recent writes | Twist | Chunked addresses, increments |
| **5. R1CS Wiring** | Properties 1–4 are correctly connected; PC updates | Spartan | Full witness vector |

---

## Key Terminology

Before diving into the properties, here are the core terms used throughout:

| Term | Meaning |
|------|---------|
| $T$ | Total number of execution cycles (trace length) |
| $t, j$ | Cycle index (ranges from 0 to $T-1$) |
| $\text{PC}$ | **Program Counter** — the memory address of the current instruction |
| $\text{rv}$ | **Read Value** — data read from memory or bytecode |
| $\text{wv}$ | **Write Value** — data written to a register or memory |
| $\text{rs1}, \text{rs2}$ | **Source Registers** — the two input registers for an instruction |
| $\text{rd}$ | **Destination Register** — the output register for an instruction |
| $\text{ra}$ | **Read Address** — which address is being read |
| $\text{wa}$ | **Write Address** — which address is being written |
| MLE | **Multilinear Extension** — polynomial representation of discrete data |
| $\widetilde{f}$ | The MLE of function/table $f$ |

---

> **Polynomial Types Legend**
> - **COMMITTED** — Prover commits to this polynomial; binding
> - **VIRTUAL** — Never committed; verified via sumcheck chain
> - **PUBLIC** — Known to both prover and verifier (from preprocessing or efficiently computable)

---

## Property 1: Bytecode Correctness (Shout Read-Checking)

**Protocol:** Shout batch evaluation (survey § 5.1, Eq. 16)

**Goal:** Prove that instructions fetched from bytecode match committed read addresses.

### The Equation Being Proven

$$\boxed{\text{rv}(t) = \text{Bytecode}[\text{PC}_t] \quad \forall t \in [T]}$$

where:
- $\text{rv}(t)$ is the **read value** at cycle $t$ — the instruction data fetched
- $\text{PC}_t$ is the **program counter** at cycle $t$ — the address in bytecode memory
- $\text{Bytecode}[x]$ is the instruction stored at address $x$ in the program
- $T$ is the total number of execution cycles
- $[T]$ denotes the set $\{0, 1, \ldots, T-1\}$

*In plain English: At each cycle, the instruction fetched equals the instruction stored at the current program counter address.*

### The Sumcheck Equation (MLE Version)

$$\widetilde{\text{rv}}(\tau) = \sum_{x \in \{0,1\}^{\ell}} \sum_{t \in \{0,1\}^{\log T}} \text{eq}(\tau, t) \cdot \widetilde{\text{BytecodeRa}}(x, t) \cdot \text{Bytecode}[x]$$

**Variables:**

| Symbol | Meaning | Type |
|--------|---------|------|
| $x$ | Bytecode address (binary representation) | Index |
| $t$ | Cycle index (binary representation) | Index |
| $\ell$ | $\log(\text{program size})$ — number of bits to address bytecode | Parameter |
| $\tau \in \mathbb{F}$ | Random challenge point from verifier (not a cycle index!) | Challenge |
| $\text{eq}(\tau, t)$ | Equality polynomial — equals 1 when $\tau = t$, 0 otherwise on hypercube | Helper |

**Polynomials:**

| Symbol | Description | Type |
|--------|-------------|------|
| $\widetilde{\text{BytecodeRa}}(x, t)$ | One-hot read address selector — equals 1 iff address $x$ is read at cycle $t$ | COMMITTED |
| $\text{Bytecode}[x]$ | Decoded instruction data (opcode, rs1, rs2, rd, immediate, circuit flags) | PUBLIC |
| $\widetilde{\text{rv}}(\tau)$ | Claimed instruction read, evaluated at random point $\tau$ | VIRTUAL |

### What This Proves

1. $\text{BytecodeRa}(x,t)$ is one-hot: exactly one $x$ per cycle $t$
2. On the hypercube: $\text{rv}(t) = \text{Bytecode}[\text{PC}_t]$ where $\text{PC}_t$ is the one-hot address
3. The MLE extends this to all field points via polynomial interpolation
4. Sumcheck verifies the prover's $\widetilde{\text{rv}}(\tau)$ matches the sumcheck equation above at random $\tau$
5. Schwartz-Zippel lemma: agreement at a random point implies agreement everywhere (w.h.p.)

### What This Does NOT Prove

- That $\text{PC}_t$ is the *correct* address (prover could read wrong instructions)
- That rv fields are used correctly (prover could ignore opcode or use wrong registers)
- Property 5 (R1CS) handles this: it connects rv to other properties and enforces PC updates

### Example: Cycle 3

If $\text{PC}_3 = 4$ (the program counter points to address 4), then:
$$\text{rv}(3) = \text{Bytecode}[4] = (\text{opcode: MUL, rs1: x10, rs2: x11, rd: x10, ...})$$

This equation verifies the lookup is correct. R1CS (Property 5) verifies that $\text{PC}_3=4$ is correct.

---

## Property 2: Instruction Semantics (Shout Lookups)

**Protocol:** Shout batch evaluation (for bitwise ops) + Spartan R1CS (for arithmetic ops)

**Goal:** Prove that instruction outputs match their input-output specification.

### The Equation Being Proven

$$\boxed{\text{LookupOutput}(t) = \text{OP}(\text{rs1\_val}(t), \text{rs2\_val}(t)) \quad \forall t \in [T]}$$

where:
- $\text{LookupOutput}(t)$ is the computed result of the instruction at cycle $t$
- $\text{OP}$ is the operation (ADD, MUL, XOR, AND, etc.)
- $\text{rs1\_val}(t)$ is the value in source register 1 at cycle $t$
- $\text{rs2\_val}(t)$ is the value in source register 2 at cycle $t$

*In plain English: At each cycle, the instruction output equals the correct operation applied to the input register values.*

### Two Proving Strategies

The strategy depends on whether the operation is **decomposable** across bit-chunks:

| Strategy | Operations | Key Property | How Proven |
|----------|-----------|--------------|------------|
| **Chunk-based lookups** | XOR, AND, OR, shifts | Chunks are *independent* (no carries) | 16 lookups into 256-entry tables |
| **Field arithmetic** | MUL, ADD, SUB | Chunks have *dependencies* (carries) | Native 254-bit field ops + range check |

**Why the difference?** Bitwise operations like XOR process each bit independently:
$$\text{XOR}(a,b) = \text{XOR\_4}(a[0:3], b[0:3]) \,\|\, \text{XOR\_4}(a[4:7], b[4:7]) \,\|\, \cdots$$

Arithmetic operations have carry propagation, making chunk decomposition unsound.

---

### Path A: Bitwise Operations (Chunk-Based)

#### Step 2a — Chunk Lookup Correctness

*For each chunk $c \in \{0, \ldots, 15\}$:*

$$\widetilde{\text{rv}}_c(r') = \sum_{x \in \{0,1\}^8} \widetilde{\text{InstructionRa}}_c(x, r') \cdot \widetilde{f}_{\text{op}}(x)$$

**Variables:**

| Symbol | Meaning |
|--------|---------|
| $x = (a, b)$ | 8-bit lookup index formed from two 4-bit operand chunks |
| $c$ | Chunk index (0–15, since 64 bits / 4 bits per chunk = 16 chunks) |
| $r'$ | Random challenge point from verifier |

**Polynomials:**

| Symbol | Description | Type |
|--------|-------------|------|
| $\widetilde{\text{InstructionRa}}_c(x, r')$ | Chunked read-address matrix for chunk $c$ — encodes which table entry is accessed (one-hot property enforced via separate sumcheck) | COMMITTED |
| $\widetilde{f}_{\text{op}}(x)$ | Lookup table MLE (XOR_4, AND_4, etc.). Size: $2^8 = 256$ entries | PUBLIC |
| $\widetilde{\text{rv}}_c(r')$ | Result for chunk $c$ | VIRTUAL |

**What this proves:** Each chunk result $\text{rv}_c$ correctly looks up from the operation table.

**Example — XOR at cycle 3, chunk 0:**
If the first 4-bit chunks of the two operands are $a = 5$ and $b = 3$, the lookup computes:
- $5 = 0101_2$
- $3 = 0011_2$
- $5 \oplus 3 = 0110_2 = 6$

So $\text{rv}_0 = \text{XOR\_4}(5, 3) = 6$.


#### Step 2b — Recombine Chunks

$$\text{LookupOutput}(t) = \sum_{c=0}^{15} \text{rv}_c(t) \cdot 2^{4c}$$

where:
- $\text{rv}_c(t)$ is the 4-bit result from chunk $c$
- $2^{4c}$ is the positional weight (chunk 0 is bits 0–3, chunk 1 is bits 4–7, etc.)

**What this proves:** The full 64-bit output equals the concatenation of 16 four-bit chunk results.

**Example — XOR at cycle 3:**
If $\text{rv}_0=6, \text{rv}_1=12, \ldots$, then $\text{LookupOutput} = 6 + 12 \cdot 16 + \cdots$

#### Combined Result (Steps 2a + 2b)

$$\boxed{\text{LookupOutput}(t) = \sum_{c=0}^{15} f_{\text{op}}(\text{input\_chunk}_c(t)) \cdot 2^{4c}}$$

**Result:** Output = operation applied to each chunk independently, then recombined.

---

### Path B: Arithmetic Operations (Field Arithmetic)

For MUL, ADD, SUB — use native 254-bit field arithmetic:

1. **Compute:** $\text{Product} = \text{LeftInput} \cdot \text{RightInput}$ (field multiplication)
2. **Truncate:** $\text{LookupOutput} = \text{RangeCheck}(\text{Product})$ (verify fits in 64 bits)
3. **Verify:** R1CS (Property 5) proves $\text{Product} = \text{rs1\_val} \cdot \text{rs2\_val}$

**Example — MUL at cycle 3:**
$\text{Product} = 2 \cdot 3 = 6$, $\text{RangeCheck}(6) = 6$ (fits in 64 bits)

**Result:** Output = field arithmetic result, verified to fit in 64 bits.

---

## Property 3: Register Consistency (Twist Virtualization)

**Protocol:** Twist read/write memory checking (survey § 5.3, Eq. 20)

**Goal:** Prove that register reads return the most recent write via increments.

### The Equation Being Proven

$$\boxed{\text{rv}_{rs1}(j) = \sum_{j'<j} \text{wa}(\text{rs1\_addr}, j') \cdot \text{wv}(j')}$$

where:
- $\text{rv}_{rs1}(j)$ is the value read from source register rs1 at cycle $j$
- $j'$ iterates over all cycles before $j$
- $\text{wa}(\text{rs1\_addr}, j')$ is 1 if cycle $j'$ wrote to the same register, 0 otherwise (one-hot)
- $\text{wv}(j')$ is the value written at cycle $j'$
- The one-hot nature of wa ensures only the most recent write contributes

*In plain English: The value read from a register equals the most recently written value to that register.*

---

### The Three Coordinated Sumchecks

#### Step 3a — Memory State Virtualization

*Compute the state of each register before each cycle:*

$$f(k, j) = \sum_{j' \in \{0,1\}^{\log T}} \widetilde{\mathbf{wa}}(k, j') \cdot \widetilde{\text{Inc}}(j') \cdot \widetilde{\text{LT}}(j', j)$$

**Variables:**

| Symbol | Meaning |
|--------|---------|
| $k$ | Register index $\in \{0, \ldots, 63\}$ (32 architectural + 32 virtual registers) |
| $j, j'$ | Cycle indices |

**Polynomials:**

| Symbol | Description | Type |
|--------|-------------|------|
| $f(k,j)$ | Value at register $k$ before cycle $j$ | VIRTUAL |
| $\widetilde{\mathbf{wa}}(k, j')$ | Write address one-hot — 1 if cycle $j'$ writes to register $k$ | VIRTUAL |
| $\widetilde{\text{Inc}}(j')$ | Increment written at cycle $j'$ (new value minus old value) | COMMITTED |
| $\widetilde{\text{LT}}(j', j)$ | Less-than indicator — 1 if $j' < j$, 0 otherwise | PUBLIC |

**What this proves:** State at cycle $j$ = initial value + all prior increments to that address.

**Example — register x10 at cycle 3:**
Suppose cycles 0 and 1 both wrote to register x10, with $\text{Inc}(0) = 1$ and $\text{Inc}(1) = 1$. Then:
$$f(10, 3) = 0 + \text{Inc}(0) + \text{Inc}(1) = 0 + 1 + 1 = 2$$

#### Step 3b — Read-Checking Sumcheck

*Verify that reads return the correct state:*

$$\widetilde{\text{rv}}_{rs1}(r') = \sum_{j \in \{0,1\}^{\log T}} \sum_{k \in \{0,1\}^{\log K}} \text{eq}(r', j) \cdot \widetilde{\mathbf{ra}}_{rs1}(k, j) \cdot f(k, j)$$

where $K=64$ (number of registers).

**Polynomials:**

| Symbol | Description | Type |
|--------|-------------|------|
| $\widetilde{\mathbf{ra}}_{rs1}(k, j)$ | Register read address one-hot — 1 if cycle $j$ reads register $k$ as rs1 | VIRTUAL |
| $\widetilde{\text{rv}}_{rs1}(r')$ | Value read from rs1, evaluated at random point $r'$ | VIRTUAL |

**What this proves:** Since $\text{ra}_{rs1}$ is one-hot: $\text{rv}_{rs1}(j) = f(\text{rs1\_addr}, j)$

**Example — cycle 3:** $\text{rs1}=\text{x10}$, so $\text{rv}_{rs1}(3) = f(10, 3) = 2$

#### Step 3c — Write-Checking via Increment Definition

*Define increments as the difference between new and old values:*

$$\widetilde{\text{Inc}}(j) = \text{wv}(j) - \sum_k \widetilde{\mathbf{wa}}(k, j) \cdot f(k, j)$$

**Polynomials:**

| Symbol | Description | Type |
|--------|-------------|------|
| $\text{wv}(j)$ | Write value at cycle $j$ — equals LookupOutput from Property 2 | VIRTUAL |

Since wa is one-hot: $\sum_k \text{wa}(k,j) \cdot f(k,j) = f(\text{rd\_addr}, j)$

**What this proves:** $\text{Inc}(j) = \text{new value} - \text{old state}$

Rearranging: $\text{wv}(j) = f(\text{rd\_addr}, j) + \text{Inc}(j)$

**Example — cycle 3:** $\text{Inc}(3) = 6 - f(10,3) = 6 - 2 = 4$

---

### Combining the Three Steps

**The proof chain:**

1. From Step 3b: $\text{rv}_{rs1}(j) = f(\text{rs1\_addr}, j)$

2. Substitute Step 3a: $\text{rv}_{rs1}(j) = \sum_{j'<j} \text{wa}(\text{rs1\_addr}, j') \cdot \text{Inc}(j')$

3. Substitute Step 3c: $\text{rv}_{rs1}(j) = \sum_{j'<j} \text{wa}(\text{rs1\_addr}, j') \cdot (\text{wv}(j') - f(\text{rd\_addr}, j'))$

4. Simplify (wa is one-hot):

$$\boxed{\text{rv}_{rs1}(j) = \sum_{j'<j, \text{ writes to rs1\_addr}} \text{wv}(j')}$$

**Result:** Read value = sum of all prior writes to that address. This is memory consistency.

---

## Property 4: RAM Consistency (Twist with Chunking)

**Protocol:** Twist read/write memory checking (same as Property 3, with address chunking)

**Goal:** Prove that RAM reads return the most recent write via increments.

### The Equation Being Proven

$$\boxed{\text{RAM\_rv}(j) = \sum_{j'<j, \text{ writes to addr}} \text{RAM\_wv}(j')}$$

where:
- $\text{RAM\_rv}(j)$ is the value read from RAM at cycle $j$
- $\text{RAM\_wv}(j')$ is the value written to RAM at cycle $j'$

*Identical structure to Property 3, but for RAM instead of registers.*

---

### Key Difference: Address Chunking

| Memory | Address Space | Chunking Parameter |
|--------|--------------|----------|
| Registers | $K = 64$ | None needed ($d=1$) |
| RAM | $K = 2^{32}$ (4 billion addresses) | Required ($d=4$) |

**The Problem:** Without chunking, we'd need $2^{32} \approx 4\text{B}$ polynomial coefficients — intractable!

**The Solution:** Chunk addresses into $d$ pieces:

$$\widetilde{\mathbf{ra}}(k, j) = \prod_{i=1}^{d} \widetilde{\mathbf{ra}}_i(k_i, j)$$

where:
- $k = (k_1, k_2, \ldots, k_d)$ is the address split into $d$ chunks
- Each chunk $k_i$ is 8 bits (since $32 / 4 = 8$)

**Polynomials:**

| Symbol | Description | Type |
|--------|-------------|------|
| $\widetilde{\mathbf{ra}}_i(k_i, j)$ | Chunk $i$ of read address — which 8-bit value | COMMITTED |

> **Note:** This is different from Property 3's register addresses, which are VIRTUAL (derived from bytecode). RAM addresses must be COMMITTED because they are computed at runtime from register values, not hardcoded in the program.

**Why $d=4$ works:** $K^{1/d} = (2^{32})^{1/4} = 2^8 = 256$ — manageable polynomial size!

---

### The Three Steps (Same Structure as Property 3)

**Step 4a — RAM state virtualization:**
$$f(k,j) = \sum_{j'<j} \text{wa}(k,j') \cdot \text{RamInc}(j') \cdot \text{LT}(j',j)$$

**Step 4b — RAM read-checking:**
$$\text{RAM\_rv}(j) = \sum_k \text{ra}(k,j) \cdot f(k,j)$$

**Step 4c — RAM write via increment:**
$$\text{RamInc}(j) = \text{RAM\_wv}(j) - f(\text{addr}, j)$$

---

### Summary

| Polynomial | Type |
|------------|------|
| $\text{RamRa}_i$ (d chunks) | COMMITTED |
| $\text{RamInc}$ | COMMITTED |
| $f$, $\text{wa}$, $\text{RAM\_rv}$, $\text{RAM\_wv}$ | VIRTUAL |

**Result:** The final equation is the same as Property 3:

$$\boxed{\text{RAM\_rv}(j) = \sum_{j'<j, \text{ writes to addr}} \text{RAM\_wv}(j')}$$

The key difference is that RAM addresses are COMMITTED (chunked), while register addresses are VIRTUAL.

---

## Property 5: R1CS Wiring (Spartan)

**Protocol:** Spartan uniform R1CS (survey § 6, theory doc § 2.1)

**Goal:** Link Properties 1–4 together and enforce correct PC updates and control flow.

### The Equation Being Proven

$$\boxed{Az \circ Bz - Cz = \mathbf{0} \quad \text{for all } T \times 30 \text{ constraint rows}}$$

where:
- $A, B, C$ are sparse constraint matrices (PUBLIC)
- $z$ is the witness vector containing all committed polynomials
- $\circ$ denotes element-wise (Hadamard) product
- $T$ is the number of cycles
- ~30 constraints are checked per cycle

*In plain English: All ~30 constraints hold at every cycle.*

---

### The Three Categories of Constraints

| Category | Purpose | Example |
|----------|---------|---------|
| **Wiring** | Link virtual polynomials from Properties 1–4 | $\text{LookupOutput}(t) = \text{wv}(t)$ |
| **PC Update** | Enforce correct program counter evolution | $\text{PC}(t+1) = \text{PC}(t) + 4$ |
| **Control Flow** | Handle jumps, branches, special cases | $\text{BranchTaken} = \text{Condition} \cdot \text{BranchFlag}$ |

---

### R1CS Structure

$$Az \circ Bz - Cz = \mathbf{0}$$

**The witness vector:**
$$z = [\text{PC}, \text{LookupOutput}, \text{Inc}, \text{InstructionRa}, \text{BytecodeRa}, \text{RamRa}, \text{RamInc}, \ldots]$$

| Component | Type |
|-----------|------|
| $z$ | COMMITTED (contains all committed polys as sub-vectors) |
| $A, B, C$ | PUBLIC (sparse constraint matrices, uniform across cycles) |

Matrix dimensions: $(T \times 30) \times |z|$, but extremely sparse.

---

### Key Constraint Examples

#### Constraint 5a — PC Update (Normal Instruction)

$$(\widetilde{\text{PC}}(t+1) - \widetilde{\text{PC}}(t) - 4) \cdot (1 - \text{JumpFlag}(t)) = 0$$

where:
- $\text{PC}(t)$ is the program counter at cycle $t$
- $\text{PC}(t+1)$ is the program counter at the next cycle
- $\text{JumpFlag}(t)$ is 1 if the instruction at cycle $t$ is a jump, 0 otherwise
- The constant 4 is the instruction size in bytes (RISC-V uses 32-bit instructions)

**What this proves:** If not jumping, PC increments by 4 (next sequential instruction).

**Role:** Enforces sequential execution; ensures Property 1 PC addresses are correct.

#### Constraint 5b — Instruction-to-Register Link (The Key Wiring)

$$\widetilde{\text{LookupOutput}}(t) - \widetilde{\text{wv}}(t) = 0$$

| Term | Source | Type |
|------|--------|------|
| $\widetilde{\text{LookupOutput}}(t)$ | Instruction result from Property 2 | VIRTUAL |
| $\widetilde{\text{wv}}(t)$ | Write value to register from Property 3 | VIRTUAL |

**What this proves:** Instruction output equals value written to register.

**Role:** This is the key wiring constraint — it links Property 2 (instruction semantics) to Property 3 (register writes).

**Example — cycle 3:** MUL output (6) = value written to x10 (6)

#### Constraint 5c — Jump Target

$$(\widetilde{\text{PC}}(t+1) - \text{JumpTarget}(t)) \cdot \text{JumpFlag}(t) = 0$$

where:
- $\text{JumpTarget}(t)$ is the target address from the jump instruction (from bytecode immediate field)

**What this proves:** If jumping, PC equals jump target from bytecode.

**Role:** Enforces correct control flow; connects Property 1 immediate field to PC.

*[... ~27 more constraints for branches, loads/stores, special cases ...]*

---

### Mapping Constraints to R1CS Form

Each constraint becomes one row in the R1CS system.

**Example 5a (product form):**

$$(\text{PC}(t+1) - \text{PC}(t) - 4) \cdot (1 - \text{JumpFlag}(t)) = 0$$

Maps to:
- $A_{\text{row}} \cdot z = \text{PC}(t+1) - \text{PC}(t) - 4$
- $B_{\text{row}} \cdot z = 1 - \text{JumpFlag}(t)$
- $C_{\text{row}} \cdot z = 0$

**Example 5b (linear form):**

$$\text{LookupOutput}(t) - \text{wv}(t) = 0$$

Maps to:
- $A_{\text{row}} \cdot z = \text{LookupOutput}(t) - \text{wv}(t)$
- $B_{\text{row}} \cdot z = 1$
- $C_{\text{row}} \cdot z = 0$

---

### Spartan's Sumcheck

*Verifies ALL constraints simultaneously:*

$$\sum_{t \in \{0,1\}^{\log T}} \text{eq}(\tau, t) \cdot \left[ (\widetilde{Az})_t \circ (\widetilde{Bz})_t - (\widetilde{Cz})_t \right] = \mathbf{0}$$

where:
- $\tau$ is a random challenge point from the verifier
- $\text{eq}(\tau, t)$ selects cycle $t$ when $\tau = t$
- The sumcheck reduces checking all $T \times 30$ constraints to a single random evaluation

**What this proves:** All $T \times 30$ constraint rows hold — in a single sumcheck.

**Role:** Reduces $T \times 30$ checks to one polynomial evaluation at random $\tau$.

**Example — cycle 3:**
- $(0\text{x}80000018 - 0\text{x}80000014 - 4) \cdot 1 = 0$
- $6 - 6 = 0$
- ...

---

## References

**Survey:** Throughout this document, "survey" refers to:

> Justin Thaler. *Sum-check Is All You Need: An Opinionated Survey on Fast Provers in SNARK Design.* 2025.
> https://eprint.iacr.org/2025/2041
