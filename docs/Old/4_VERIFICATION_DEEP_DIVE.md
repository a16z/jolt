# Part 4: Verification Deep Dive

**Prerequisites**: Read [Part 1 (Preprocessing)](1_PREPROCESSING_DEEP_DIVE.md), [Part 2 (Execution and Witness)](2_EXECUTION_AND_WITNESS_DEEP_DIVE.md), and [Part 3 (Proof Generation)](3_PROOF_GENERATION_DEEP_DIVE.md) first.

## Table of Contents

1. [Overview: The Verifier's Job](#overview-the-verifiers-job)
2. [What the Verifier Receives](#what-the-verifier-receives)
3. [Verifier Setup](#verifier-setup)
4. [Stage 1 Verification: Spartan Outer Sumcheck](#stage-1-verification-spartan-outer-sumcheck)
5. [Stages 2-4 Verification: Batched Sumchecks](#stages-2-4-verification-batched-sumchecks)
6. [Stage 5 Verification: Batched Opening](#stage-5-verification-batched-opening)
7. [Complete Verification Example](#complete-verification-example)
8. [Security Analysis](#security-analysis)

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
- ✓ The program bytecode (via commitment from preprocessing)
- ✓ Public inputs and claimed outputs
- ✓ Trace length T (number of cycles executed)

**What the verifier does NOT know**:
- ✗ The execution trace (which instructions ran, in what order)
- ✗ Register values during execution
- ✗ Memory values during execution
- ✗ Intermediate computation results

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
| Execution trace | ✓ Full trace (7M elements) | ✗ None (only T = length) |
| Witness polynomials | ✓ All 35 MLEs | ✗ None (only commitments) |
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

## Stage 1 Verification: Spartan Outer Sumcheck

**File**: [jolt-core/src/zkvm/spartan/mod.rs](../jolt-core/src/zkvm/spartan/mod.rs) → `stage1_verify()`

### Recall: What Stage 1 Proved

**Prover's claim** (from Part 3):

$$\sum_{x \in \{0,1\}^{\log m}} \text{eq}(\tau, x) \cdot [(Az)(x) \cdot (Bz)(x) - (Cz)(x)] = 0$$

Where:
- $m = 30 \times T$ (30 constraints per cycle, T cycles)
- $\tau$ is random challenge
- $Az, Bz, Cz$ are virtual polynomials (matrix-vector products)

**Prover ran sumcheck for $\log m$ rounds**, producing univariate polynomials $g_1, g_2, \ldots, g_{\log m}$.

### Verifier's Job

**Verifier does NOT**:
- ✗ Recompute $Az, Bz, Cz$ (doesn't have witness $z$!)
- ✗ Evaluate the polynomial at all $2^{\log m}$ points
- ✗ Run the full sumcheck protocol

**Verifier DOES**:
- ✓ Check consistency of univariate polynomials
- ✓ Sample same random challenges via Fiat-Shamir
- ✓ Verify final evaluation claim

---

### The Sumcheck Verification Protocol

**File**: [jolt-core/src/subprotocols/sumcheck.rs:140](../jolt-core/src/subprotocols/sumcheck.rs#L140)

**Input**:
- Initial claim: $H$ (the claimed sum)
- Number of rounds: $n = \log m$
- Sumcheck proof: $\pi = (g_1, g_2, \ldots, g_n)$ where each $g_i$ is a univariate polynomial
- Transcript (for Fiat-Shamir challenges)

**Output**:
- Final evaluation point: $\vec{r} = (r_1, \ldots, r_n)$
- Final claimed value: $v$
- Accept/reject

---

**Round-by-round verification**:

**Round 1**:

Prover claims:
$$H = \sum_{x_1 \in \{0,1\}} \sum_{x_2, \ldots, x_n \in \{0,1\}^{n-1}} f(x_1, x_2, \ldots, x_n)$$

Prover sent univariate polynomial $g_1(X_1)$:
$$g_1(X_1) = \sum_{x_2, \ldots, x_n \in \{0,1\}^{n-1}} f(X_1, x_2, \ldots, x_n)$$

**Verifier checks**:

$$g_1(0) + g_1(1) \stackrel{?}{=} H$$

**Why this check works**:

$$g_1(0) + g_1(1) = \sum_{x_2, \ldots, x_n} f(0, x_2, \ldots, x_n) + \sum_{x_2, \ldots, x_n} f(1, x_2, \ldots, x_n)$$

$$= \sum_{x_1 \in \{0,1\}} \sum_{x_2, \ldots, x_n} f(x_1, x_2, \ldots, x_n) = H$$

If check fails → prover lied about the sum → **reject**.

**Sample random challenge**:

$$r_1 = \mathcal{H}(\text{transcript} \parallel g_1)$$

**Reduced claim**:

$$g_1(r_1) = \sum_{x_2, \ldots, x_n \in \{0,1\}^{n-1}} f(r_1, x_2, \ldots, x_n)$$

Note: $f(r_1, \cdot)$ is now evaluated at **random field element** $r_1 \in \mathbb{F}$, not Boolean!

---

**Round 2**:

Prover sent univariate $g_2(X_2)$ claiming:

$$g_2(X_2) = \sum_{x_3, \ldots, x_n \in \{0,1\}^{n-2}} f(r_1, X_2, x_3, \ldots, x_n)$$

**Verifier checks**:

$$g_2(0) + g_2(1) \stackrel{?}{=} g_1(r_1)$$

This verifies that $g_2$ is consistent with the previous round's claim.

**Sample challenge**: $r_2 = \mathcal{H}(\text{transcript} \parallel g_2)$

**Reduced claim**: $g_2(r_2) = \sum_{x_3, \ldots, x_n} f(r_1, r_2, x_3, \ldots, x_n)$

---

**Rounds 3 through n**: Continue the same pattern.

**Round n** (final):

Prover sent $g_n(X_n)$ claiming:

$$g_n(X_n) = f(r_1, r_2, \ldots, r_{n-1}, X_n)$$

**Verifier checks**:

$$g_n(0) + g_n(1) \stackrel{?}{=} g_{n-1}(r_{n-1})$$

**Sample challenge**: $r_n = \mathcal{H}(\text{transcript} \parallel g_n)$

**Final claim**:

$$v = g_n(r_n) = f(r_1, r_2, \ldots, r_n)$$

---

### The Final Evaluation Check

**Problem**: Verifier has reduced to:

$$f(\vec{r}) \stackrel{?}{=} v$$

Where $\vec{r} = (r_1, \ldots, r_n) \in \mathbb{F}^n$ and $v \in \mathbb{F}$.

**But verifier doesn't have $f$!** For Stage 1:

$$f(x) = \text{eq}(\tau, x) \cdot [(Az)(x) \cdot (Bz)(x) - (Cz)(x)]$$

Verifier doesn't know $z$ (the witness), so can't compute $Az, Bz, Cz$!

**Solution**: Store as **virtual opening claims** in accumulator:

```rust
accumulator.append_virtual(
    VirtualPolynomial::SpartanAz,
    vec![r_1, ..., r_n],
    claimed_value_Az
);
accumulator.append_virtual(
    VirtualPolynomial::SpartanBz,
    vec![r_1, ..., r_n],
    claimed_value_Bz
);
accumulator.append_virtual(
    VirtualPolynomial::SpartanCz,
    vec![r_1, ..., r_n],
    claimed_value_Cz
);
```

These claims will be verified in **Stage 2** via product sumchecks.

**Verifier computes** $\text{eq}(\tau, \vec{r})$ (public values):

$$\text{eq}(\tau, \vec{r}) = \prod_{i=1}^{n} [\tau_i r_i + (1 - \tau_i)(1 - r_i)]$$

Then verifies consistency:

$$v \stackrel{?}{=} \text{eq}(\tau, \vec{r}) \cdot [v_{Az} \cdot v_{Bz} - v_{Cz}]$$

**Understanding the Accumulator After Stage 1**:

The accumulator now contains **three virtual claims**:

```rust
accumulator.virtual_claims = [
    (VirtualPolynomial::Az, r_vec, v_Az),
    (VirtualPolynomial::Bz, r_vec, v_Bz),
    (VirtualPolynomial::Cz, r_vec, v_Cz),
]
```

**Key point**: These are **virtual** because:
- $Az$, $Bz$, $Cz$ are NOT directly committed polynomials
- They are matrix-vector products: $Az = A \times \widetilde{z}$, where $\widetilde{z}$ is the committed witness
- Prover sent values $v_{Az}, v_{Bz}, v_{Cz}$ but has NOT proven them yet
- These will be verified in **Stage 2 product sumchecks**

**Recall from Part 3 ([3_PROOF_GENERATION_DEEP_DIVE.md](3_PROOF_GENERATION_DEEP_DIVE.md#L2230))**:
The prover added these same virtual claims to their accumulator after Stage 1, committing to prove them in Stage 2.

If this check passes, Stage 1 verification **succeeds**.

---

### Concrete Example: Stage 1 Verification

**Setup** (from Part 3's toy example):
- Constraints: $m = 2$ → $\log m = 1$ round
- Claim: $\sum_{x \in \{0,1\}} \text{eq}(\tau, x) \cdot [(Az)(x) \cdot (Bz)(x) - (Cz)(x)] = 0$
- Prover sent univariate $g_1(X) = c_0 + c_1 X + c_2 X^2 + c_3 X^3$

**Step 1: Initial challenge**

Verifier samples from transcript:
$$\tau = \mathcal{H}(\text{transcript}) = 0.37 \text{ (in some field } \mathbb{F})$$

**Step 2: Check consistency**

Prover claims $g_1(0) + g_1(1) = 0$ (the claimed sum).

Verifier computes:
$$g_1(0) = c_0$$
$$g_1(1) = c_0 + c_1 + c_2 + c_3$$

Verifier checks:
$$c_0 + (c_0 + c_1 + c_2 + c_3) \stackrel{?}{=} 0$$
$$2c_0 + c_1 + c_2 + c_3 \stackrel{?}{=} 0$$

✓ If true, continue. Otherwise, **reject**.

**Step 3: Sample next challenge**

$$r_1 = \mathcal{H}(\text{transcript} \parallel g_1) = 0.91$$

**Step 4: Compute claimed value**

$$v = g_1(r_1) = c_0 + c_1(0.91) + c_2(0.91)^2 + c_3(0.91)^3$$

Say $v = 1332.24$.

**Step 5: Deferred verification**

Verifier **cannot** directly check $f(r_1) = v$ because that requires knowing $Az(r_1), Bz(r_1), Cz(r_1)$.

Instead, verifier stores **virtual claims**:
- $Az(r_1) = v_A$ (prover claims this)
- $Bz(r_1) = v_B$ (prover claims this)
- $Cz(r_1) = v_C$ (prover claims this)

And verifies:
$$v \stackrel{?}{=} \text{eq}(0.37, 0.91) \cdot [v_A \cdot v_B - v_C]$$

Where:
$$\text{eq}(0.37, 0.91) = 0.37 \times 0.91 + (1-0.37) \times (1-0.91) = 0.3367 + 0.0567 = 0.3934$$

Check:
$$1332.24 \stackrel{?}{=} 0.3934 \times [42.7 \times 31.2 - 1332.24]$$
$$1332.24 \stackrel{?}{=} 0.3934 \times [1332.24 - 1332.24] = 0.3934 \times 0 = 0$$

Wait, this doesn't match! Let me recalculate...

Actually, the outer sumcheck proves the sum equals 0, so:

$$v = 0$$

And:

$$0 \stackrel{?}{=} 0.3934 \times [v_A \cdot v_B - v_C]$$

This means: $v_A \cdot v_B - v_C = 0$, which is exactly the R1CS constraint!

✓ Stage 1 verification **succeeds**.

**Cost**: O(1) field operations per round × 1 round = **O(1) total** (vs prover's O(m) = O(2) work!)

---

### Summary: Stage 1 Verification

**What verifier checked**:
1. ✓ Consistency of univariate polynomials ($g_i(0) + g_i(1) = \text{previous claim}$)
2. ✓ Fiat-Shamir challenges match (same transcript)
3. ✓ Final evaluation consistent with $\text{eq}(\tau, \vec{r})$

**What verifier deferred**:
- Claims about $Az(\vec{r}), Bz(\vec{r}), Cz(\vec{r})$ stored as virtual openings
- Will be verified in Stage 2

**Complexity**:
- Rounds: $\log m$ (e.g., $\log(30 \times 1024) \approx 15$ rounds)
- Work per round: O(1) field operations
- **Total: O(log m)** vs prover's O(m)

**Memory**:
- Stores only: challenges $\vec{r}$, claimed values, univariate coefficients
- **Total: O(log m)** vs prover's O(m)

---

## Stages 2-4 Verification: Batched Sumchecks

**File**: [jolt-core/src/zkvm/dag/jolt_dag.rs:430-520](../jolt-core/src/zkvm/dag/jolt_dag.rs#L430)

### Recall: What Batching Does

**Prover ran** (from [Part 3, Stage 2](3_PROOF_GENERATION_DEEP_DIVE.md#L2294)):

Stage 2 contains sumchecks from **four components**:

1. **Spartan Product Sumchecks (3 sumchecks)**:
   - Prove $\widetilde{Az}(\vec{r}) = v_A$ (virtual claim from Stage 1)
   - Prove $\widetilde{Bz}(\vec{r}) = v_B$ (virtual claim from Stage 1)
   - Prove $\widetilde{Cz}(\vec{r}) = v_C$ (virtual claim from Stage 1)
   - These resolve the virtual claims by showing $Az[i] = \sum_j A[i,j] \cdot z[j]$

2. **Registers Twist (2 sumchecks)**:
   - Read-checking sumcheck: Proves reads from registers return correct values
   - Write-checking sumcheck: Proves writes to registers update correctly
   - Uses grand product over fingerprints: $(t, k, v)$ triples

3. **RAM Twist (3 sumchecks)**:
   - Read-checking: Memory reads return last written values
   - Write-checking: Memory writes recorded correctly
   - Output check: Final memory state matches claimed outputs

4. **Instructions Booleanity (1 sumcheck)**:
   - Proves lookup address polynomials $\widetilde{\text{ra}}_i$ are Boolean (entries in $\{0, 1\}$)
   - Required for one-hot encoding of lookup indices

**Total Stage 2 sumchecks**: 3 + 2 + 3 + 1 = **9 sumchecks**

**Recall from Part 3 ([3_PROOF_GENERATION_DEEP_DIVE.md](3_PROOF_GENERATION_DEEP_DIVE.md#L2294))**: Stage 3 has 6 sumchecks, Stage 4 has 8 sumchecks.

**Without batching**: Verifier would verify 9 + 6 + 8 = 23 separate sumchecks
- 23 separate challenge sequences
- More proof data
- More verification work

**With batching**: Verifier verifies **3 batched sumchecks** (one per stage)
- 3 challenge sequences (shared across all sumchecks in each stage)
- Much less proof data
- Much less verification work

---

### Batched Sumcheck Verification Protocol

**File**: [jolt-core/src/subprotocols/sumcheck.rs:178](../jolt-core/src/subprotocols/sumcheck.rs#L178)

**Setup**:

Stage 2 has multiple sumchecks with claims:
$$C_1: \sum_{x} f_1(x) = H_1$$
$$C_2: \sum_{x} f_2(x) = H_2$$
$$\vdots$$
$$C_k: \sum_{x} f_k(x) = H_k$$

**Step 1: Sample batching coefficients**

$$\alpha_1, \alpha_2, \ldots, \alpha_k = \mathcal{H}(\text{transcript})$$

**Critical**: Both prover and verifier sample the **same** $\alpha$ values.

**Step 2: Form batched claim**

$$\sum_{x} [\alpha_1 f_1(x) + \alpha_2 f_2(x) + \cdots + \alpha_k f_k(x)] = \alpha_1 H_1 + \alpha_2 H_2 + \cdots + \alpha_k H_k$$

**Step 3: Verify single batched sumcheck**

Prover sent a single sumcheck proof for the batched polynomial.

Verifier runs standard sumcheck verification (as in Stage 1).

**Step 4: Each component extracts its output claim**

After sumcheck with final challenges $\vec{r}$:

$$\alpha_1 f_1(\vec{r}) + \alpha_2 f_2(\vec{r}) + \cdots + \alpha_k f_k(\vec{r}) = v_{\text{combined}}$$

Each verifier instance computes what its claimed value should be:
- Spartan product verifier: Claims about $\widetilde{z}(\vec{r})$
- Registers verifier: Claims about register polynomials
- RAM verifier: Claims about memory polynomials

All claims get stored in accumulator for later verification.

---

### Example: Stage 2 Verification (Simplified)

**Scenario**: Stage 2 has 3 sumchecks (simplified from actual 8):

1. **Spartan Az product**: Claim $\sum_y A(\tau, y) \cdot z(y) = v_A$
2. **Registers Twist**: Claim about register consistency
3. **RAM output check**: Claim about final memory state

**Step 1: Sample batching coefficients**

From transcript:
$$\alpha_1 = 0.123, \quad \alpha_2 = 0.456, \quad \alpha_3 = 0.789$$

**Step 2: Batched claim**

$$\alpha_1 v_A + \alpha_2 v_{\text{reg}} + \alpha_3 v_{\text{ram}} = 0.123 \times 42.7 + 0.456 \times 100 + 0.789 \times 350$$

$$= 5.2521 + 45.6 + 276.15 = 327.0021$$

**Step 3: Verify batched sumcheck**

Prover sent univariate polynomials $g_1, g_2, \ldots, g_{23}$ (for 23 rounds).

Verifier checks consistency for each round:
$$g_i(0) + g_i(1) = \text{previous claim}$$

After 23 rounds with challenges $\vec{r} = (r_1, \ldots, r_{23})$:

Final claim: $v_{\text{combined}} = 327.0021$ (should match!)

**Step 4: Extract individual claims**

**Spartan Az verifier**:
- Knows: $\vec{r}, \alpha_1, \tau$ (public values)
- Stores claim: $\widetilde{z}(\vec{r}) = v_z$ (committed opening)
- Contribution to batched claim: $\alpha_1 \times \text{(something involving } v_z)$

**Registers verifier**:
- Stores claims about $\widetilde{\Delta}_{\text{rd}}$ at $\vec{r}$
- Contribution: $\alpha_2 \times \text{(register fingerprint)}$

**RAM verifier**:
- Stores claims about $\widetilde{\Delta}_{\text{ram}}$ at $\vec{r}$
- Contribution: $\alpha_3 \times \text{(memory output)}$

All contributions must sum to $v_{\text{combined}}$ — this is checked implicitly by sumcheck verification!

---

### Security of Batching

**Question**: If we batch $k$ sumchecks, does security degrade?

**Answer**: No! Security remains high due to random linear combination.

**Why batching is secure**:

Suppose a malicious prover wants to cheat on sumcheck $i$ (claims $\sum f_i(x) = H_i$ when actually $\sum f_i(x) = H_i'$).

**Without batching**: Prover would be caught immediately in sumcheck $i$.

**With batching**:
- Batched claim: $\sum_x [\alpha_1 f_1(x) + \cdots + \alpha_k f_k(x)] = \alpha_1 H_1 + \cdots + \alpha_k H_k$
- But prover knows $\sum f_i(x) = H_i' \neq H_i$
- So prover must make batched sum equal: $\alpha_1 H_1 + \cdots + \alpha_i H_i + \cdots + \alpha_k H_k$
- But actual sum is: $\alpha_1 H_1 + \cdots + \alpha_i H_i' + \cdots + \alpha_k H_k$
- Difference: $\alpha_i (H_i - H_i')$

**For batched claim to pass**, prover needs:

$$\alpha_i (H_i - H_i') = 0$$

Since $H_i \neq H_i'$, prover needs $\alpha_i = 0$.

But $\alpha_i$ is sampled **randomly** from transcript (which includes prover's commitments).

**Probability** $\alpha_i = 0$: At most $1/|\mathbb{F}|$ (field size).

For a 256-bit field: Probability $\approx 2^{-256}$ — **negligible**!

**Schwartz-Zippel Lemma**: Even if prover cheats on multiple sumchecks, random linear combination catches it with high probability.

---

### Stages 2, 3, 4: Verification Process

**Stage 2 components**:
1. Spartan product sumchecks (3)
2. Registers Twist read-write checking
3. RAM Twist sumchecks (3)
4. Instructions booleanity

**Stage 3 components**:
1. Spartan matrix evaluation (no sumcheck — verifier computes directly!)
2. Registers evaluation sumcheck
3. RAM evaluation sumcheck
4. Instructions read-checking (2)

**Stage 4 components**:
1. Instructions Ra virtualization
2. Bytecode read-checking (3)
3. RAM Ra virtualization + evaluations (4)

**Verifier process for each stage** (same pattern):

```rust
// 1. Each component creates verifier instances
let stage_k_instances: Vec<_> = std::iter::empty()
    .chain(spartan_dag.stage_k_verifier_instances(&mut state_manager))
    .chain(registers_dag.stage_k_verifier_instances(&mut state_manager))
    .chain(ram_dag.stage_k_verifier_instances(&mut state_manager))
    .chain(lookups_dag.stage_k_verifier_instances(&mut state_manager))
    .chain(bytecode_dag.stage_k_verifier_instances(&mut state_manager))
    .collect();

// 2. Sample batching coefficients from transcript
let alphas = sample_batching_coefficients(transcript, stage_k_instances.len());

// 3. Verify batched sumcheck proof
let (r_stage_k, combined_claim) = BatchedSumcheck::verify(
    stage_k_instances,
    batched_proof_stage_k,
    transcript,
)?;

// 4. Each instance extracts its output claims
for instance in stage_k_instances {
    instance.verify_and_extract_output(r_stage_k, &mut accumulator);
}
```

**Key insight**: Each verifier instance is **lightweight**:
- No witness data
- No polynomial evaluations
- Just algebraic formulas to compute expected outputs
- Example: Registers verifier computes expected fingerprint product from public parameters

**Cost per stage**:
- One batched sumcheck verification: O(log N) per stage
- Total for stages 2-4: **O(log N)** (constant number of stages)

---

### Example: Registers Twist Verifier Instance

**File**: [jolt-core/src/zkvm/registers/read_write_checking.rs](../jolt-core/src/zkvm/registers/read_write_checking.rs)

**What it proves**: Register reads return last written values

**Recall from Part 2 ([2_EXECUTION_AND_WITNESS_DEEP_DIVE.md](2_EXECUTION_AND_WITNESS_DEEP_DIVE.md))**: Register Twist instance maintains two orderings:
- **Time-ordered**: Reads/writes as they occurred during execution
- **Address-ordered**: Reads/writes sorted by (register_id, timestamp)

**Twist memory checking** proves these are permutations via **grand product equality**.

---

**Prover side** (from [Part 3, Registers component](3_PROOF_GENERATION_DEEP_DIVE.md#L2336)):
- Computed fingerprints for time-ordered trace:
  $$\prod_{j=0}^{T-1} [\gamma_1 \cdot \text{addr}_j + \gamma_2 \cdot \text{timestamp}_j + \gamma_3 \cdot \text{value}_j]$$
- Computed fingerprints for address-ordered trace (same formula, different order)
- Proved grand product equality via sumcheck
- Sent univariate polynomials for batched sumcheck

---

**Verifier side**:

After batched sumcheck with challenges $\vec{r} = (r_1, \ldots, r_n)$, each component extracts output claims:

```rust
impl SumcheckInstance for RegistersReadWriteChecking {
    fn verify_and_extract_output(
        &self,
        r: Vec<FieldElement>,
        accumulator: &mut VerifierAccumulator,
    ) {
        // r is the random evaluation point from batched sumcheck

        // Compute expected read fingerprint at r (public formula)
        let read_fingerprint = compute_read_fingerprint_formula(r, self.num_registers);

        // Compute expected write fingerprint at r (public formula)
        let write_fingerprint = compute_write_fingerprint_formula(r, self.num_registers);

        // Store claim: products should be equal
        // This gets verified implicitly by next sumcheck in the chain

        // Store committed opening claims for register increment polynomial
        accumulator.append_committed(
            CommittedPolynomial::RdInc,
            r.clone(),
            claimed_delta_rd_value,  // From prover's sumcheck output
        );
    }
}
```

**Verifier computes** (public formulas):

The read fingerprint at random point $\vec{r}$:

$$\text{read\_fp}(\vec{r}) = \prod_{k=0}^{K-1} [\gamma_1 \cdot k + \gamma_2 \cdot \text{init\_ts}(k) + \gamma_3 \cdot \text{init\_val}(k)]$$

Where:
- $K = 64$ (32 RISC-V registers + 32 virtual registers)
- $\gamma_1, \gamma_2, \gamma_3$ are random challenges from Fiat-Shamir
- $k$ is the register index (public)
- $\text{init\_ts}(k) = 0$ (initial timestamp, public)
- $\text{init\_val}(k) = 0$ (initial register value, public - all registers start at 0)

**Example calculation** (for our toy ADD example with 3 registers):

Suppose after batched sumcheck, $\vec{r} = (0.91, 0.73, 0.44)$ and $\gamma_1 = 0.123, \gamma_2 = 0.456, \gamma_3 = 0.789$.

For register $k=1$ (r1):
$$\gamma_1 \cdot 1 + \gamma_2 \cdot 0 + \gamma_3 \cdot 0 = 0.123 \times 1 = 0.123$$

For register $k=2$ (r2):
$$\gamma_1 \cdot 2 + \gamma_2 \cdot 0 + \gamma_3 \cdot 0 = 0.123 \times 2 = 0.246$$

For register $k=3$ (r3):
$$\gamma_1 \cdot 3 + \gamma_2 \cdot 0 + \gamma_3 \cdot 0 = 0.123 \times 3 = 0.369$$

Read fingerprint product:
$$\text{read\_fp} = 0.123 \times 0.246 \times 0.369 = 0.0112$$

The write fingerprint is computed similarly, but incorporates the increment polynomial $\widetilde{\Delta}_{\text{rd}}$:

$$\text{write\_fp}(\vec{r}) = \prod_{k=0}^{K-1} [\gamma_1 \cdot k + \gamma_2 \cdot \text{final\_ts}(k) + \gamma_3 \cdot (\text{init\_val}(k) + \widetilde{\Delta}_{\text{rd}}(k))]$$

**Key point**: Verifier never sees actual register values during execution! Only:
- Computes expected fingerprint formulas from public parameters
- Stores claim about $\widetilde{\Delta}_{\text{rd}}(\vec{r})$ (committed polynomial)
- Verifies algebraic consistency

**What gets added to accumulator**:
```rust
accumulator.committed_claims.push(
    (CommittedPolynomial::RdInc, r_vec, v_delta_rd)
);
```

This claim will be verified in **Stage 5 batched opening**.

---

## Stage 5 Verification: Batched Opening

**File**: [jolt-core/src/zkvm/dag/jolt_dag.rs:542](../jolt-core/src/zkvm/dag/jolt_dag.rs#L542)

### Understanding the Opening Accumulator

**The accumulator is the verifier's "ledger" of polynomial evaluation claims.**

Think of it like a to-do list: "I need to verify that polynomial P evaluates to value v at point r."

**Two types of claims**:

1. **Virtual claims**: Will be verified by subsequent sumchecks in the DAG
   - Example: After Stage 1, accumulator contains virtual claims for $Az, Bz, Cz$
   - These get resolved in Stage 2 product sumchecks
   - Once resolved, they're removed from the accumulator

2. **Committed claims**: Will be verified by Dory opening in Stage 5
   - Example: $\widetilde{L}(\vec{r}_L) = 5.217$ (committed polynomial from Part 2)
   - These accumulate throughout Stages 1-4
   - All verified together in Stage 5

**Recall from Part 3 ([3_PROOF_GENERATION_DEEP_DIVE.md](3_PROOF_GENERATION_DEEP_DIVE.md#L173))**: The accumulator tracks claims about two fundamentally different types of polynomials:
- **Type A: Committed Polynomials** (the 35 witness polynomials from Part 2)
- **Type B: Virtual Polynomials** (computed on-the-fly, never committed)

**The DAG structure emerges from this distinction**: Virtual evaluations create edges between sumchecks (dependencies), while committed evaluations accumulate for final batched opening.

---

**Accumulator evolution through stages**:

**After Stage 1**:
```rust
accumulator.virtual_claims = [
    (Az, r_1, v_Az),      // Will be verified in Stage 2
    (Bz, r_1, v_Bz),      // Will be verified in Stage 2
    (Cz, r_1, v_Cz),      // Will be verified in Stage 2
]
accumulator.committed_claims = []
```

**After Stage 2**:
```rust
accumulator.virtual_claims = []  // Az, Bz, Cz resolved!
accumulator.committed_claims = [
    (WitnessZ, r_2, v_z),           // From Spartan product sumcheck
    (RdInc, r_2, v_delta_rd),       // From registers Twist
    (RamInc, r_2, v_delta_ram),     // From RAM Twist
    (InstructionRa_0, r_2, v_ra0),  // From booleanity check
    // ... ~20 total claims
]
```

**After Stages 3-4**:
```rust
accumulator.virtual_claims = []  // All virtual claims resolved!
accumulator.committed_claims = [
    // All 36 committed polynomial evaluations from Stages 1-4
    (WitnessZ, r_2, v_z),
    (LeftInput, r_3, v_L),
    (RightInput, r_3, v_R),
    // ... 33 more claims
]
```

**After Stage 5**: Accumulator is empty! All claims verified via Dory batched opening.

---

### Recall: What Stage 5 Proves

**After Stages 1-4**, accumulator contains **36 committed opening claims**:

$$\{\widetilde{P}_1(\vec{r}_1) = v_1, \widetilde{P}_2(\vec{r}_2) = v_2, \ldots, \widetilde{P}_{36}(\vec{r}_{36}) = v_{36}\}$$

Where:
- $\widetilde{P}_i$ are the 35 witness polynomials + Spartan's $\widetilde{z}$
- $\vec{r}_i$ are different random evaluation points (from various sumchecks)
- $v_i$ are claimed evaluations

**Prover created** (from Part 3):
1. Reduction polynomial: $\widetilde{Q}(\vec{X}) = \sum_{i=1}^{36} \beta_i \widetilde{P}_i(\vec{X}) \cdot \text{eq}(\vec{X}, \vec{r}_i)$
2. Reduction sumcheck: Reduced to single claim $\widetilde{Q}(\vec{\rho}) = q$
3. Dory opening proof: Proved $\widetilde{Q}(\vec{\rho}) = q$

**Verifier's job**: Check all three steps.

---

### Step 1: Sample Batching Coefficients

**Verifier action**:

```rust
// Same transcript as prover
for (poly_id, point, value) in committed_openings_from_accumulator {
    transcript.append_field_element(value);
    transcript.append_field_elements(&point);
}

// Sample same batching coefficients
let betas: Vec<FieldElement> = (0..36)
    .map(|i| transcript.challenge_scalar())
    .collect();
```

**Result**: $\beta_1, \ldots, \beta_{36} \in \mathbb{F}$ (same as prover sampled)

---

### Step 2: Verify Reduction Sumcheck

**Initial claim**:

$$\sum_{\vec{x} \in \{0,1\}^{23}} \widetilde{Q}(\vec{x}) \stackrel{?}{=} \sum_{i=1}^{36} \beta_i v_i$$

**Verifier computes RHS**:

$$H_{\text{expected}} = \beta_1 v_1 + \beta_2 v_2 + \cdots + \beta_{36} v_{36}$$

Using the claimed values $v_i$ from accumulator.

**Example calculation**:

$$H_{\text{expected}} = 0.123 \times 8.342 + 0.456 \times 5.217 + \cdots + 0.789 \times 0.00$$

Say $H_{\text{expected}} = 2731.4$.

**Verifier runs standard sumcheck verification**:

Prover sent 23 univariate polynomials $g_1, \ldots, g_{23}$ (for 23-dimensional reduction polynomial).

**Round 1**:

Check: $g_1(0) + g_1(1) \stackrel{?}{=} 2731.4$

Sample: $\rho_1 = \mathcal{H}(\text{transcript} \parallel g_1)$

**Rounds 2-23**: Continue with consistency checks

**Final claim**: $\widetilde{Q}(\vec{\rho}) = q$ where $\vec{\rho} = (\rho_1, \ldots, \rho_{23})$

---

### Step 3: Compute Expected Evaluation

**Verifier expands** $\widetilde{Q}(\vec{\rho})$:

$$\widetilde{Q}(\vec{\rho}) = \sum_{i=1}^{36} \beta_i \widetilde{P}_i(\vec{\rho}) \cdot \text{eq}(\vec{\rho}, \vec{r}_i)$$

**Verifier can compute** $\text{eq}(\vec{\rho}, \vec{r}_i)$ for all $i$:

$$\text{eq}(\vec{\rho}, \vec{r}_i) = \prod_{j=1}^{23} [\rho_j r_{i,j} + (1 - \rho_j)(1 - r_{i,j})]$$

This is $O(23)$ work per $i$, so $O(36 \times 23) = O(828)$ field operations total.

**Example** (for $i=1$, suppose $\vec{r}_1 = (r_{1,1}, \ldots, r_{1,23})$ is the evaluation point for $\widetilde{z}$):

$$\text{eq}(\vec{\rho}, \vec{r}_1) = (0.91 \times r_{1,1} + 0.09 \times (1 - r_{1,1})) \times \cdots \times (0.73 \times r_{1,23} + 0.27 \times (1 - r_{1,23}))$$

Say $\text{eq}(\vec{\rho}, \vec{r}_1) = 0.0023$.

**But verifier cannot compute** $\widetilde{P}_i(\vec{\rho})$ — doesn't have the polynomials!

**Solution**: These are exactly what the **Dory opening proof** will verify.

---

### Step 4: Verify Dory Opening Proof

**File**: [jolt-core/src/poly/commitment/dory.rs](../jolt-core/src/poly/commitment/dory.rs)

**Input**:
- Commitments: $C_1, \ldots, C_{36}$ (received at start of proof)
- Evaluation point: $\vec{\rho} \in \mathbb{F}^{23}$
- Claimed combined evaluation: $q \in \mathbb{F}$
- Batching coefficients: $\beta_1, \ldots, \beta_{36}$
- Dory proof: $\pi_{\text{Dory}}$

**Dory's guarantee**:

If verification passes, then with high probability:
$$\sum_{i=1}^{36} \beta_i \widetilde{P}_i(\vec{\rho}) = q_{\text{polys}}$$

And verifier already computed:
$$\sum_{i=1}^{36} \beta_i \text{eq}(\vec{\rho}, \vec{r}_i) = q_{\text{eq}}$$

**Final check**:
$$q \stackrel{?}{=} q_{\text{polys}} \times q_{\text{eq}}$$

If this passes, all 36 polynomial evaluations are correct!

---

### Dory Verification Protocol (Detailed)

**File**: [jolt-core/src/poly/commitment/dory.rs](../jolt-core/src/poly/commitment/dory.rs)

**Recall from Part 2 ([2_EXECUTION_AND_WITNESS_DEEP_DIVE.md](2_EXECUTION_AND_WITNESS_DEEP_DIVE.md#L1459))**: Dory commitments use bilinear pairings.

**Setup**:
- Verifier has generators $\{G_{1,i}\}_{i=0}^{N-1} \subset \mathbb{G}_1$ and $\{G_{2,j}\}_{j=0}^{n-1} \subset \mathbb{G}_2$ (from preprocessing)
- For $N = 2^{23}$ (max polynomial size), verifier stores ~25 KB of generators
- Much smaller than prover's generators (~400 KB) due to different structure

**Dory's key property**: Commitment to polynomial $\widetilde{P}$ of size $N = 2^n$ can be opened in $O(n)$ rounds.

---

**Proof structure** (23 rounds for 23-dimensional polynomial):

Each round $k$ contains:
- $C_L^{(k)}, C_R^{(k)} \in \mathbb{G}_T$ (two commitments in target group)
- $v_L^{(k)}, v_R^{(k)} \in \mathbb{F}$ (two claimed values)

**Total proof size**: $23 \times (2 \times 192 + 2 \times 32) = 23 \times 448 = 10,304$ bytes ≈ **10 KB**

---

**Verification Protocol**:

**Input**:
- Initial commitment: $C_Q \in \mathbb{G}_T$ (from proof commitments)
- Initial evaluation point: $\vec{\rho} = (\rho_1, \ldots, \rho_{23}) \in \mathbb{F}^{23}$
- Initial claimed value: $q \in \mathbb{F}$
- Claim: $\widetilde{Q}(\vec{\rho}) = q$

**Goal**: Verify claim by recursively reducing dimension from 23 to 0.

---

**Round 1** (dimension 23 → 22):

**Prover's strategy**: Split polynomial along first variable:
$$\widetilde{Q}(X_1, X_2, \ldots, X_{23}) = \widetilde{Q}_L(X_2, \ldots, X_{23}) + X_1 \cdot \widetilde{Q}_R(X_2, \ldots, X_{23})$$

Where:
- $\widetilde{Q}_L$ contains coefficients with $X_1^0$ (even indices when viewing as vector)
- $\widetilde{Q}_R$ contains coefficients with $X_1^1$ (odd indices)

**At evaluation point** $\vec{\rho}$:
$$\widetilde{Q}(\vec{\rho}) = \widetilde{Q}_L(\rho_2, \ldots, \rho_{23}) + \rho_1 \cdot \widetilde{Q}_R(\rho_2, \ldots, \rho_{23})$$

Let $v_L = \widetilde{Q}_L(\rho_2, \ldots, \rho_{23})$ and $v_R = \widetilde{Q}_R(\rho_2, \ldots, \rho_{23})$.

**Prover sends**: $C_L, C_R, v_L, v_R$

---

**Verifier checks**:

**Check 1: Value consistency**
$$v_L + \rho_1 \cdot v_R \stackrel{?}{=} q$$

This ensures the split is consistent with the claimed evaluation.

**Example**: If $q = 42.7$, $\rho_1 = 0.91$, then we need:
$$v_L + 0.91 \cdot v_R = 42.7$$

Suppose prover sent $v_L = 18.3, v_R = 26.8$:
$$18.3 + 0.91 \times 26.8 = 18.3 + 24.388 = 42.688 \approx 42.7$$ ✓

---

**Check 2: Commitment consistency** (using bilinear pairings)

**Bilinear pairing**: $e: \mathbb{G}_1 \times \mathbb{G}_2 \to \mathbb{G}_T$ with properties:
1. Bilinearity: $e(aP, bQ) = e(P, Q)^{ab}$
2. Non-degeneracy: $e(G_1, G_2) \neq 1_{\mathbb{G}_T}$

**Dory commitment structure** (from preprocessing):

Original commitment $C_Q$ has form:
$$C_Q = e\left(\sum_{i=0}^{N-1} c_i G_{1,i}, \sum_{j=0}^{n-1} \text{bit}_j(\vec{\rho}) G_{2,j}\right)$$

Where $c_i$ are polynomial coefficients and $\text{bit}_j(\vec{\rho})$ encodes evaluation point.

**Key insight**: When splitting along first variable, we can express:
$$C_Q = e(P_L, G_{2,0}) \cdot e(P_R, G_{2,1}) \cdot e(\ldots, G_{2,2}) \cdots$$

Where:
- $P_L = \sum_{\text{even } i} c_i G_{1,i/2}$ (left half coefficients)
- $P_R = \sum_{\text{odd } i} c_i G_{1,i/2}$ (right half coefficients)

**Verifier check**:
$$C_Q \stackrel{?}{=} e(C_L, G_{2,0}) \cdot e(C_R, G_{2,1})^{\rho_1}$$

This verifies that $C_Q$ correctly decomposes into $C_L, C_R$ **without the verifier knowing the coefficients**!

**Why this works**:
- $C_L = e(P_L, \text{compressed generators})$ commits to left half
- $C_R = e(P_R, \text{compressed generators})$ commits to right half
- Pairing properties ensure: $C_Q = C_L \cdot C_R^{\rho_1}$ iff the split is valid

**Example**: If pairing check passes, verifier is convinced (with high probability):
- $C_L$ commits to some polynomial $\widetilde{Q}_L$ of dimension 22
- $C_R$ commits to some polynomial $\widetilde{Q}_R$ of dimension 22
- $C_Q$ commits to $\widetilde{Q}_L + X_1 \widetilde{Q}_R$

---

**Fold to next round**:

**Sample challenge** (Fiat-Shamir):
$$\alpha_1 = \mathcal{H}(\text{transcript} \parallel C_L \parallel C_R \parallel v_L \parallel v_R)$$

**Example**: $\alpha_1 = 0.742$

**Create folded polynomial** (verifier doesn't compute this, but understands the structure):
$$\widetilde{Q}'(X_2, \ldots, X_{23}) = \widetilde{Q}_L(X_2, \ldots, X_{23}) + \alpha_1 \cdot \widetilde{Q}_R(X_2, \ldots, X_{23})$$

**Folded commitment** (verifier computes):
$$C_{Q'} = C_L \cdot C_R^{\alpha_1}$$

Using pairing group operations.

**Folded claimed value** (verifier computes):
$$q' = v_L + \alpha_1 \cdot v_R$$

**Example**: $q' = 18.3 + 0.742 \times 26.8 = 18.3 + 19.8856 = 38.1856$

**Reduced claim**: $\widetilde{Q}'(\rho_2, \ldots, \rho_{23}) = q'$

**Progress**: Reduced from 23 variables to 22 variables!

---

**Rounds 2-23**: Continue the same pattern.

Each round:
- Splits along next variable
- Verifies value consistency and commitment consistency
- Folds to next dimension
- Dimension decreases: $23 \to 22 \to 21 \to \cdots \to 1 \to 0$

---

**Final round** (round 23):

After 22 folds, polynomial reduced to 1 dimension.

**Round 23 split**: $\widetilde{Q}^{(22)}(X_{23}) = v_L^{(23)} + X_{23} \cdot v_R^{(23)}$

This is just a linear polynomial!

**At $\rho_{23}$**: $v_L^{(23)} + \rho_{23} \cdot v_R^{(23)} = q^{(22)}$ ✓

**Final fold** with $\alpha_{23}$:
$$q^{(23)} = v_L^{(23)} + \alpha_{23} \cdot v_R^{(23)}$$

Now $q^{(23)}$ is a **constant** (0-dimensional polynomial).

**Final commitment check**: $C^{(23)} \in \mathbb{G}_T$ should commit to constant $q^{(23)}$.

This is trivially checkable: $C^{(23)} \stackrel{?}{=} e(G_1, G_2)^{q^{(23)}}$

If all 23 rounds pass, verifier is convinced: $\widetilde{Q}(\vec{\rho}) = q$ ✓

---

**Total verification cost**:
- 23 rounds
- Per round: 1 value check (O(1) field ops) + 1 pairing check (O(1) pairing ops) + 1 folding (O(1) group ops)
- **Total: 23 pairings** + O(23) field operations
- Pairing operations are expensive (~1 ms each on modern CPUs) but constant-time
- **Total Dory verification time**: ~25 ms

**Compare to prover**:
- Prover: O(N) = O(2^{23}) operations to compute commitment and proof
- Verifier: O(log N) = O(23) operations to verify
- **Asymmetry achieved!**

---

### Complete Stage 5 Verification Summary

**What verifier checked**:

1. ✓ **Batching coefficients**: Sampled same $\beta_i$ as prover (via Fiat-Shamir)
2. ✓ **Reduction sumcheck**: Verified 23-round sumcheck reduced 36 claims to 1
3. ✓ **Equality polynomial evaluations**: Computed $\text{eq}(\vec{\rho}, \vec{r}_i)$ for all $i$
4. ✓ **Dory opening**: Verified 23-round Dory proof with pairing checks

**What verifier did NOT do**:
- ✗ Recompute any witness polynomials
- ✗ Evaluate polynomials at $2^{23}$ points
- ✗ Perform $O(N)$ work (where $N$ = witness size)

**Complexity**:
- Reduction sumcheck: O(23) rounds × O(1) = O(log N)
- Eq computations: O(36 × 23) = O(828) = O(1) relative to N
- Dory verification: O(23) pairing checks = O(log N)
- **Total: O(log N)**

**Memory**:
- Store: $\vec{\rho}$, commitments, proof data
- **Total: O(log N)**

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

✓ Passes (prover constructed correctly).

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

✓ Check passes!

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

✓ **Verification succeeds!**

---

### Verification Verdict

**Verifier concludes**:

With overwhelming probability ($\geq 1 - \text{negl}$):
1. ✓ A RISC-V program with the committed bytecode executed for 1 cycle
2. ✓ Given inputs `[5, 7]`, the program produced output `[12]`
3. ✓ All register operations were consistent (r1=5, r2=7 were read; r3=12 was written)
4. ✓ All instruction lookups were correct (ADD(5,7) = 12 via 4 sub-lookups)
5. ✓ No memory operations occurred (this was pure register arithmetic)

**What verifier never saw**:
- ✗ The actual execution trace (which instructions, which cycles)
- ✗ Intermediate register values
- ✗ The 1-hot lookup address matrices
- ✗ Any of the witness polynomials (only saw commitments)

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
| Stage 5 Dory | 23 pairings | O(log N) |
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
