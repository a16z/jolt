# Part 3: Proof Generation Deep Dive

This document provides a detailed, mathematical explanation of **Part 3: Proof Generation** from the Jolt code flow. We connect the theory from [Theory/Jolt.md](../Theory/Jolt.md) to the actual implementation, showing what mathematical objects are created and manipulated at each stage.

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

**Key insight from Theory/Jolt.md (lines 22-27)**:

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

From [Theory/The Sum-check Protocol.md](../Theory/The Sum-check Protocol.md):

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

We created **35 witness polynomials** from the execution trace. From [2_EXECUTION_AND_WITNESS_DEEP_DIVE.md](2_EXECUTION_AND_WITNESS_DEEP_DIVE.md#L1282):

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

**Each polynomial was committed via Dory**: From [2_EXECUTION_AND_WITNESS_DEEP_DIVE.md](2_EXECUTION_AND_WITNESS_DEEP_DIVE.md#L1459):

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

**What is $Az$?** Recall from Spartan (Theory/Spartan.md), the R1CS constraint system proves:

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

**Where are matrices created?**: During **preprocessing** (Part 1)! From [1_PREPROCESSING_DEEP_DIVE.md](1_PREPROCESSING_DEEP_DIVE.md):

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

Recall from [Part 2](2_EXECUTION_AND_WITNESS_DEEP_DIVE.md#L1409), we created **35 witness polynomials**. The witness vector $z$ is **not** these 35 polynomials directly! Instead:

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

**Connection to Part 2**: These are EXACTLY the polynomials we created! From [2_EXECUTION_AND_WITNESS_DEEP_DIVE.md](2_EXECUTION_AND_WITNESS_DEEP_DIVE.md#L1286):

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

From [Theory/Spartan.md](../Theory/Spartan.md):

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

From [Theory/Jolt.md](../Theory/Jolt.md) and Twist paper:

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

From [Theory/Jolt.md](../Theory/Jolt.md) (lines 108-112):

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

**Key**: $\text{table}(k)$ is efficiently evaluable! (See [Theory/Jolt.md](../Theory/Jolt.md))

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
