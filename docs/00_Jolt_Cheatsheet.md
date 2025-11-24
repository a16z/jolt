# Jolt zkVM: Technical Cheatsheet

> **Quick reference for Jolt's mathematical foundations, implementation, and verification.**
> For full derivations see [01_Jolt_Theory_Enhanced.md](01_Jolt_Theory_Enhanced.md), implementation details in [02_Jolt_Complete_Guide.md](02_Jolt_Complete_Guide.md).

---

## Core Architecture

**zkVM Target**: RV64IMAC (64-bit RISC-V base + M/A/C extensions)

**Proof System**: Sum-check-based SNARK with transparent setup (Dory PCS)

**Key Innovation**: Lookup-centric architecture - prove CPU execution via table lookups, not circuits

**Five Components**:
1. **R1CS (Spartan)**: PC updates, component wiring (~30 constraints/cycle)
2. **Instruction Lookups (Shout)**: Batch evaluation of CPU operations
3. **Memory Checking (Twist)**: RAM/register read-write consistency
4. **Bytecode Verification (Shout)**: Offline memory checking for instruction fetch
5. **Polynomial Commitments (Dory)**: Transparent PCS with logarithmic verifier

**Proof DAG**: 5 stages + optional Stage 6 (SNARK composition for GT exponentiations)

---

## Mathematical Foundations

### Multilinear Extensions (MLE)

**Definition**: For $f: \{0,1\}^n \to \mathbb{F}$, unique multilinear $\widetilde{f}: \mathbb{F}^n \to \mathbb{F}$ where:
$$\widetilde{f}(r) = \sum_{x \in \{0,1\}^n} f(x) \cdot \text{eq}(r, x)$$

**Equality polynomial**:
$$\text{eq}(r, x) = \prod_{i=1}^n [r_i x_i + (1-r_i)(1-x_i)]$$

**Properties**:
- $\text{eq}(r, x) = 1$ iff $r = x$ (on Boolean hypercube)
- Multilinear in both arguments

### Sum-check Protocol

**Claim**: $H = \sum_{x \in \{0,1\}^{\nu}} g(x)$

**Protocol** ($\nu$ rounds):
1. Prover sends $s_j(X_j) = \sum_{x_{j+1}, \ldots, x_{\nu} \in \{0,1\}} g(r_1, \ldots, r_{j-1}, X_j, x_{j+1}, \ldots, x_{\nu})$
2. Verifier checks $s_j(0) + s_j(1) \stackrel{?}{=} H_{j-1}$
3. Verifier samples $r_j \leftarrow \mathbb{F}$
4. Update $H_j = s_j(r_j)$

**Soundness**: Schwartz-Zippel - probability cheat succeeds $\leq \frac{d\nu}{|\mathbb{F}|}$

**Final check**: $g(r_1, \ldots, r_{\nu}) \stackrel{?}{=} H_{\nu}$

### Schwartz-Zippel Lemma

For nonzero $n$-variate polynomial $f$ of total degree $d$:
$$\Pr_{r \leftarrow S^n}[f(r) = 0] \leq \frac{d}{|S|}$$

**For multilinear**: Distinct $p, q$ agree at random $r$ with probability $\leq \frac{n}{|\mathbb{F}|}$

---

## Component 1: R1CS Constraints (Spartan)

**System**: $Az \circ Bz = Cz$ where:
- $z \in \mathbb{F}^m$: Witness vector (length $m$)
- $A, B, C \in \mathbb{F}^{N \times m}$: Constraint matrices ($N$ constraints, sparse)
- $\circ$: Hadamard (entrywise) product

**Jolt's R1CS structure** (~30 constraints/cycle, uniform across all $T$ cycles):

| Constraint Category | Count | Purpose |
|---------------------|-------|---------|
| PC updates | ~8 | Sequential: $\text{PC}_{t+1} = \text{PC}_t + 4$<br>Jump: $\text{PC}_{t+1} = \text{jump\_target}$<br>Branch: Conditional PC update |
| Component linking | ~10 | RAM ↔ Registers: Load/store consistency<br>Instruction output → Register write<br>Carry chain wiring (for decomposed ops) |
| Arithmetic operations | ~8 | Native field arithmetic for 64-bit ops<br>Range checks: Truncate overflow bits via lookup<br>Flag consistency (zero flag, sign flag) |
| Circuit flags | ~4 | Opcode-derived booleans: `jump_flag`, `branch_flag`, `load_flag`, `store_flag`<br>Virtualized via bytecode read-checking |

**Why R1CS here?**: Minimal constraints for control flow and wiring. Computation done via lookups (Shout), not circuits.

**Spartan's Three Sumchecks**:

1. **Outer sumcheck** (Stage 1): Batch $N$ constraints → 1 claim
   $$\sum_{i \in \{0,1\}^{\log N}} \text{eq}(r_{\text{time}}, i) \cdot [(Az)_i \circ (Bz)_i - (Cz)_i] = 0$$
   - Rounds: $\log N$
   - Output: Random point $r_{\text{time}}$, claims $\widetilde{Az}(r_{\text{time}})$, $\widetilde{Bz}(r_{\text{time}})$, $\widetilde{Cz}(r_{\text{time}})$

2. **Product sumcheck** (Stage 2): Reduce to matrix-vector claims
   $$\widetilde{Az}(r_{\text{time}}) = \sum_{j \in \{0,1\}^{\log m}} \widetilde{A}(r_{\text{time}}, j) \cdot \widetilde{z}(j)$$
   - Three parallel sumchecks (for $A$, $B$, $C$)
   - Rounds: $\log m$ each
   - Output: Random point $r_{\text{cols}}$, claims $\widetilde{A}(r_{\text{time}}, r_{\text{cols}})$, $\widetilde{z}(r_{\text{cols}})$

3. **Matrix evaluation sumcheck** (Stage 3): Exploit sparsity via SPARK
   $$\widetilde{A}(r_{\text{time}}, r_{\text{cols}}) = \sum_{(i,j) \in \text{Nonzeros}(A)} A_{i,j} \cdot \text{eq}(r_{\text{time}}, i) \cdot \text{eq}(r_{\text{cols}}, j)$$
   - Sparse sumcheck: $O(s)$ prover work for $s$ nonzeros
   - Output: Commitment opening claims for $\widetilde{z}$

**Uniformity optimization**: $A$, $B$, $C$ have identical structure across all $T$ timesteps
- Exploit: Precompute $\widetilde{A}$, $\widetilde{B}$, $\widetilde{C}$ once
- Result: R1CS only 6% of prover time despite 30 constraints/cycle

**Complexity**:
- Prover: $O(N + m)$ field operations (exploiting sparsity + uniformity)
- Verifier: $O(\log N + \log m)$ (sumcheck rounds)
- Communication: $O(\log N + \log m)$ field elements

---

## Component 2: Instruction Lookups (Shout)

**Goal**: Prove $T$ lookups into table of size $N$ (e.g., $2^{128}$ for 64-bit operations)

**Two-tier approach** (depending on instruction):

### Tier 1: Decomposable Instructions (ADD, XOR, etc.)

**Strategy**: Break into small sub-table lookups

**Example**: 64-bit XOR with 4-bit chunks
$$a \oplus b = \sum_{i=0}^{15} 2^{4i} \cdot \text{XOR}_4(a_i, b_i)$$

- 16 lookups into 256-entry ($2^8$) XOR_4 table
- Each lookup: 2 operand chunks (4 bits each) → 9 bits total with carry
- **Shout instance per chunk**: Standard read/write-checking on small table
- **Wiring constraints**: R1CS proves carry chain correct

**Cost**: 16 Shout instances × $O(T \cdot 9)$ ≈ $O(T)$ per instruction

### Tier 2: Non-Decomposable Instructions (SLT, SRA, etc.)

**Problem**: Set-less-than (`SLT`) compares full 64-bit values
- Cannot decompose: comparison depends on all bits
- Lookup domain: $\ell = 128$ bits (two 64-bit operands)
- Naive sumcheck: $2^{128}$ terms (impossible!)

**Strategy**: Prefix-suffix sumcheck exploiting sparsity

**Key insight**: Access matrix $\widetilde{\mathbf{ra}}(x, r')$ has only $T$ nonzeros out of $2^{128}$ entries

**Prefix-suffix algorithm**:
1. Split index: $x = (x_{\text{prefix}}, x_{\text{suffix}})$ with 64 bits each
2. Sparse representation:
   $$\widetilde{\mathbf{ra}}(x_{\text{prefix}}, x_{\text{suffix}}, r') = \sum_{i=1}^T \text{eq}(x_{\text{prefix}}, y_{i,\text{prefix}}) \cdot \text{eq}(x_{\text{suffix}}, y_{i,\text{suffix}})$$
3. Two-stage sumcheck (64 rounds each):
   - Stage 1: Bind prefix → cost $O(T + 2^{64})$
   - Stage 2: Bind suffix → cost $O(T + 2^{64})$

**Multi-level chunking**: For extreme sparsity, break into $d$ chunks of $\ell/d$ bits
- Virtualize: $\widetilde{\mathbf{ra}}(x) = \prod_{i=1}^d \widetilde{\mathbf{ra}}_i(x_i)$
- Jolt uses $d$ such that $2^{\ell/d} \approx 2^8 = 256$

#### Concrete Example: SLT (Set-Less-Than) Instruction

**Problem**: Prove 1 million executions of 64-bit signed comparison

**Naive approach**:
- Table domain: $2^{128}$ entries (two 64-bit operands)
- Sumcheck: $2^{128}$ terms × 128 rounds = **impossible**

**Shout solution**: Sparse sumcheck with prefix-suffix decomposition

---

**Step 1: One-Hot Access Matrix**

For $T = 10^6$ cycles executing SLT:
$$\mathbf{ra}(x, j) = \begin{cases} 1 & \text{if } x = (a_j \| b_j) \text{ (concatenated operands at cycle } j) \\ 0 & \text{otherwise} \end{cases}$$

- Domain: $x \in \{0,1\}^{128}$ (impossibly large!)
- Only $T = 10^6$ nonzero entries (out of $2^{128}$!)

**Step 2: The Batch Evaluation Equation**

Prove all outputs correct:
$$\widetilde{\mathbf{rv}}(r') = \sum_{x \in \{0,1\}^{128}} \widetilde{\mathbf{ra}}(x, r') \cdot \widetilde{\text{SLT}}(x)$$

where:
- $\widetilde{\mathbf{rv}}(r')$: MLE of output vector at random $r'$
- $\widetilde{\mathbf{ra}}(x, r')$: MLE of access matrix (sparse!)
- $\widetilde{\text{SLT}}(x)$: MLE of SLT lookup table (verifier computes!)

**Step 3: Prefix-Suffix Split**

Split index: $x = (x_{\text{prefix}}, x_{\text{suffix}})$ with 64 bits each

**Sparse representation**:
$$\widetilde{\mathbf{ra}}(x_{\text{prefix}}, x_{\text{suffix}}, r') = \sum_{i=1}^T \text{eq}(x_{\text{prefix}}, y_{i,\text{prefix}}) \cdot \text{eq}(x_{\text{suffix}}, y_{i,\text{suffix}})$$

where $y_i = (a_i \| b_i)$ is the lookup index at cycle $i$.

**Step 4: Two-Stage Sumcheck**

**Phase 1** (bind prefix, 64 rounds):
- Build $P[i] = \sum_{\{j : \text{prefix}(y_j) = i\}} \widetilde{\text{SLT}}(y_j)$ for $i \in \{0,1\}^{64}$
- Only visit $T$ nonzeros: iterate over actual lookups, accumulate into $2^{64}$ buckets
- Run dense sumcheck over $P$ (64 rounds)
- Verifier sends random $r_{\text{prefix}}$

**Phase 2** (bind suffix, 64 rounds):
- Build $Q[j] = \sum_{\{i : \text{suffix}(y_i) = j\}} \text{eq}(r_{\text{prefix}}, \text{prefix}(y_i)) \cdot \widetilde{\text{SLT}}(y_i)$
- Again only visit $T$ nonzeros, weight by how well prefix matches $r_{\text{prefix}}$
- Run dense sumcheck over $Q$ (64 rounds)
- Verifier sends random $r_{\text{suffix}}$

**Step 5: Final Evaluation**

After 128 rounds total, verifier needs:
- $\widetilde{\mathbf{ra}}(r_{\text{prefix}}, r_{\text{suffix}}, r')$: Request opening from commitment
- $\widetilde{\text{SLT}}(r_{\text{prefix}}, r_{\text{suffix}})$: **Verifier computes directly!**

This is the key: SLT's MLE is efficiently evaluable without storing $2^{128}$ table.

---

**Cost Analysis**:

| Approach | Prover Work | Memory | Verifier Work |
|----------|-------------|---------|---------------|
| Naive sumcheck | $2^{128} \times 128$ | $2^{128}$ | $128$ rounds |
| Circuit (R1CS) | $T \times 64$ constraints | $O(T)$ | $O(\log T)$ |
| **Shout prefix-suffix** | $T \times 128 + 2 \times 2^{64}$ | $2^{64}$ | $128$ rounds |

For $T = 10^6$:
- Naive: **Impossible** ($2^{128}$ operations)
- Circuit: ~64M field ops (prover) + ~20 rounds (verifier)
- **Shout**: ~128M field ops (prover) + 128 rounds (verifier)

**Why Shout wins**:
- Circuit approach requires expressing SLT logic in constraints (complex!)
- Shout just needs efficient MLE evaluation for $\widetilde{\text{SLT}}$ (simple!)
- For complex instructions, Shout is 10-100× faster than circuits

---

**Jolt Implementation** (`read_raf_checking.rs`):

Three parallel prefix-suffix instances:
- `right_operand_ps`: Handles right operand (64 bits of lookup index)
- `left_operand_ps`: Handles left operand (64 bits of lookup index)
- `identity_ps`: Handles full 128-bit index for non-decomposable ops

All use same algorithm: two-stage sumcheck over sparse $T$ lookups into $2^{128}$ table.

**Shout Protocol Components**:
1. **Write-checking sumcheck**: Proves $M$ well-formed (Boolean + one-hot)
   $$\sum_{x} \widetilde{M}(x, r') \cdot [\widetilde{M}(x, r') - 1] = 0$$
2. **Read-checking sumcheck**: Proves correct values retrieved
   $$\sum_{x} \widetilde{\mathbf{ra}}(x, r') \cdot \widetilde{f}(x) = \widetilde{\text{rv}}(r')$$

**Requirements for table $\widetilde{f}$**:
- Must be **efficiently evaluable** (constant time per query)
- Examples: XOR (bitwise), ADD (modular), SLT (comparison)

**Complexity**:
- Decomposable: $O(T)$ per chunk × 16 chunks
- Non-decomposable: $O(T \cdot \ell + 2^{\ell/2})$ via prefix-suffix
- Chunked: $O(T \cdot \ell)$ when $d$ chosen optimally

---

## Component 3: Memory Checking (Twist)

**Goal**: Prove reads return most recent writes (or initial value)

**Challenge**: For $T$ operations on $K$ addresses, committing full $K \times T$ state table is prohibitive

### The Twist Virtualization

**Core insight**: Track **increments** (write deltas), not full state

**Increment at cycle $j$**:
$$\text{Inc}(j) := \mathbf{wv}(j) - \sum_{k} \mathbf{wa}(k, j) \cdot f(k, j)$$

where:
- $\mathbf{wv}(j)$: Value being written
- $\sum_k \mathbf{wa}(k, j) \cdot f(k, j)$: Old value at write address (one-hot sum)
- $\text{Inc}(j)$: Difference (new - old)

**Virtualize memory state** $f(k,j)$ via increments:
$$f(k, j) = \text{init}(k) + \sum_{j' < j} \mathbf{wa}(k, j') \cdot \text{Inc}(j')$$

MLE version with less-than predicate:
$$\widetilde{f}(k, j) = \widetilde{\text{init}}(k) + \sum_{j' \in \{0,1\}^{\log T}} \widetilde{\mathbf{wa}}(k, j') \cdot \widetilde{\text{Inc}}(j') \cdot \widetilde{\text{LT}}(j', j)$$

#### Concrete Example: 32 RISC-V Registers Over 4 Cycles

**Problem**: Prove register file consistency for 4 instruction executions

**Memory**: $K = 32$ registers (x0-x31), $T = 4$ cycles

**Trace**:
| Cycle $j$ | Instruction | Read $(r_1, r_2)$ | Write $(r_d, \text{wv})$ |
|-----------|-------------|-------------------|--------------------------|
| 0 | ADD x5, x1, x2 | $(1, 2) \to (10, 20)$ | $(5, 30)$ |
| 1 | SUB x7, x5, x3 | $(5, 3) \to (30, 8)$ | $(7, 22)$ |
| 2 | MUL x5, x4, x6 | $(4, 6) \to (5, 7)$ | $(5, 35)$ |
| 3 | AND x8, x5, x7 | $(5, 7) \to (35, 22)$ | $(8, 2)$ |

**Key observation**: Cell 5 written at cycle 0, read at cycle 1, written again at cycle 2, read again at cycle 3!

---

**Step 1: What Gets Committed**

Only commit to:
- $\widetilde{\mathbf{wa}}(k, j)$: Write addresses (one-hot)
- $\widetilde{\text{Inc}}(j)$: Increment vector

**Write addresses** (one-hot encoding):
- Cycle 0: $\mathbf{wa}(\cdot, 0) = [0,0,0,0,0,1,0,\ldots,0]$ (1 at position 5)
- Cycle 1: $\mathbf{wa}(\cdot, 1) = [0,0,0,0,0,0,0,1,\ldots,0]$ (1 at position 7)
- Cycle 2: $\mathbf{wa}(\cdot, 2) = [0,0,0,0,0,1,0,\ldots,0]$ (1 at position 5 again!)
- Cycle 3: $\mathbf{wa}(\cdot, 3) = [0,0,0,0,0,0,0,0,1,\ldots,0]$ (1 at position 8)

**Increments** (how much each write changes the cell):
- Cycle 0: $\text{Inc}(0) = 30 - 0 = 30$ (x5 changes from 0 to 30)
- Cycle 1: $\text{Inc}(1) = 22 - 0 = 22$ (x7 changes from 0 to 22)
- Cycle 2: $\text{Inc}(2) = 35 - 30 = 5$ (x5 changes from 30 to 35!)
- Cycle 3: $\text{Inc}(3) = 2 - 0 = 2$ (x8 changes from 0 to 2)

**What's NOT committed** (virtualized!):
- Memory state $f(k, j)$ - reconstructed from increments!
- Read values $\mathbf{rv}(j)$ - proven via sumcheck!
- Write values $\mathbf{wv}(j)$ - expressed as $f + \text{Inc}$!

---

**Step 2: Virtualizing Memory State**

For cell $k$ at time $j$, value equals initial + all increments from writes to $k$ before time $j$:

$$\tilde{f}(k, j) = \sum_{j' \in \{0,1,2,3\}} \mathbf{wa}(k, j') \cdot \text{Inc}(j') \cdot \text{LT}(j', j)$$

**Example for cell 5**:

At $j=0$ (before any writes): $f(5, 0) = 0$ (initial state)

At $j=1$ (after cycle 0 write):
$$f(5, 1) = \mathbf{wa}(5, 0) \cdot \text{Inc}(0) \cdot \text{LT}(0, 1) = 1 \cdot 30 \cdot 1 = 30$$

At $j=2$ (cycle 1 wrote to x7, not x5):
$$f(5, 2) = \mathbf{wa}(5, 0) \cdot 30 \cdot 1 + \mathbf{wa}(5, 1) \cdot 22 \cdot 1 = 1 \cdot 30 + 0 \cdot 22 = 30$$

At $j=3$ (after cycle 2 write to x5):
$$f(5, 3) = \mathbf{wa}(5, 0) \cdot 30 \cdot 1 + \mathbf{wa}(5, 1) \cdot 22 \cdot 1 + \mathbf{wa}(5, 2) \cdot 5 \cdot 1$$
$$= 1 \cdot 30 + 0 \cdot 22 + 1 \cdot 5 = 35$$

**Key**: Memory state reconstructed from sparse increments! No $32 \times 4 = 128$ entry commitment needed.

---

**Step 3: Read-Checking Sumcheck**

Prove reads return correct values (from Spartan, verifier has claim $\widetilde{\mathbf{rv}}(r') = H$):

$$H \stackrel{?}{=} \sum_{j,k \in \{0,1\}^2 \times \{0,1\}^5} \text{eq}(r', j) \cdot \widetilde{\mathbf{ra}}(k, j) \cdot \widetilde{f}(k, j)$$

**Sumcheck runs** (2 + 5 = 7 rounds):
- Bind time $j$ to random $r_j$
- Bind address $k$ to random $r_k$

**Final evaluation needed**: $\widetilde{f}(r_k, r_j)$ - but this is **virtual**!

---

**Step 4: Evaluation Sumcheck for Virtual $\widetilde{f}$**

Prover claims:
$$\widetilde{f}(r_k, r_j) = \sum_{j' \in \{0,1\}^2} \widetilde{\mathbf{wa}}(r_k, j') \cdot \widetilde{\text{Inc}}(j') \cdot \widetilde{\text{LT}}(j', r_j)$$

**Sumcheck runs** (2 rounds for $j'$):
- Final random point: $r_{j'}$

**Final evaluation**:
- $\widetilde{\mathbf{wa}}(r_k, r_{j'})$: Request opening from commitment ✓
- $\widetilde{\text{Inc}}(r_{j'})$: Request opening from commitment ✓
- $\widetilde{\text{LT}}(r_{j'}, r_j)$: **Verifier computes directly!** ✓

All committed polynomial openings batched in Stage 5!

---

**Step 5: Write-Checking Sumcheck**

Also verify write values consistent (from Spartan outer sumcheck):

$$\widetilde{\mathbf{wv}}(r'') \stackrel{?}{=} \sum_{k \in \{0,1\}^5} \widetilde{\mathbf{wa}}(k, r'') \cdot [\widetilde{f}(k, r'') + \widetilde{\text{Inc}}(r'')]$$

**Proves**: New value = old value + increment (by definition of increment!)

Again, $\widetilde{f}(k, r'')$ virtualized via another evaluation sumcheck.

---

**Cost Analysis**:

| Approach | Commitments | Prover Work | Verifier Work |
|----------|-------------|-------------|---------------|
| **Naive** (full memory table) | $K \times T = 128$ entries | $O(KT)$ | $O(\log K + \log T)$ |
| **Permutation** (sorted trace) | $2T$ (time + addr sorted) | $O(T \log K)$ + sorting | $O(\log T)$ |
| **Twist** | $2T$ (wa + Inc only) | $O(T \log K)$ | $O(\log K + \log T)$ |

For $K=32, T=10^6$:
- Naive: 32M commitments (prohibitive!)
- Permutation: 2M commitments + sorting overhead + grand product
- **Twist**: 2M commitments + streaming-friendly sparse sumcheck ✓

**Why Twist wins**:
- No sorting needed (streaming-compatible!)
- Virtualization avoids committing to memory state
- Sparse sumcheck exploits structure ($T$ nonzeros, not $KT$)
- All values small (addresses ≤ 32, increments often small)

---

**Jolt Usage**:

**Registers** ($K=64$: 32 RISC-V + 32 virtual):
- 2 read addresses ($\mathbf{ra}_{\text{rs1}}, \mathbf{ra}_{\text{rs2}}$)
- 1 write address ($\mathbf{wa}_{\text{rd}}$)
- Cost: $O(6T)$ field ops (very cheap!)

**RAM** ($K$ dynamic, typically $2^{20}$-$2^{30}$):
- 1 address per cycle (read OR write, not both)
- Chunking parameter $d$ keeps $K^{1/d} = 2^8$
- Cost: $O(T \log K)$ worst-case, often $O(T)$ with locality

### Verification Equations

**Committed polynomials**:
1. $\widetilde{\mathbf{ra}}(k, j)$: Read address (one-hot in $k$)
2. $\widetilde{\mathbf{wa}}(k, j)$: Write address (one-hot in $k$)
3. $\widetilde{\text{Inc}}(j)$: Increment vector

**Virtual polynomials** (proven via sumcheck, not committed):
- $\widetilde{f}(k, j)$: Memory state (virtualized formula above)
- $\widetilde{\mathbf{rv}}(j)$: Read values
- $\widetilde{\mathbf{wv}}(j)$: Write values

**Read-checking sumcheck** (evaluation claim): For random $r' \in \mathbb{F}^{\log T}$:
$$\widetilde{\mathbf{rv}}(r') \stackrel{?}{=} \sum_{j, k \in \{0,1\}^{\log T + \log K}} \text{eq}(r', j) \cdot \widetilde{\mathbf{ra}}(k, j) \cdot \widetilde{f}(k, j)$$

**Proves**: Read values match memory state (virtual poly $\widetilde{\mathbf{rv}}$ equals accumulated reads)

**Write-checking sumcheck** (evaluation claim): For random $r'' \in \mathbb{F}^{\log T}$:
$$\widetilde{\mathbf{wv}}(r'') \stackrel{?}{=} \sum_{k \in \{0,1\}^{\log K}} \widetilde{\mathbf{wa}}(k, r'') \cdot [\widetilde{f}(k, r'') + \widetilde{\text{Inc}}(r'')]$$

**Proves**: Write value = old value + increment (consistency of increments)

**Evaluation sumcheck** (evaluation claim): For random $r_k \in \mathbb{F}^{\log K}$:
$$\widetilde{f}_{\text{final}}(r_k) \stackrel{?}{=} \widetilde{\text{init}}(r_k) + \sum_{j \in \{0,1\}^{\log T}} \widetilde{\mathbf{wa}}(r_k, j) \cdot \widetilde{\text{Inc}}(j)$$

**Proves**: Final memory state correct (initial + all increments), verifies program outputs

**Note**: These are **evaluation sumchecks** (prove sum equals claimed value $H$), not **zero-check sumchecks** (prove sum equals 0). Virtual polynomials ($\widetilde{\mathbf{rv}}$, $\widetilde{\mathbf{wv}}$, $\widetilde{f}$) never committed - their evaluations proven via these sumchecks.

### Jolt Modifications

**RAM**: Dynamic $K$, chunking parameter $d$ (keeps $K^{1/d} = 2^8$)
- Single $ra$ polynomial (read OR write, not both per cycle)
- No-op rows: Zeros instead of one-hot (cheaper than explicit no-op encoding)
- Address remapping: $\text{index} = \frac{\text{addr} - 0x80000000}{8} + 1$ (doubleword-addressed)
- Multiple sumchecks: $d$ read-checks + $d$ write-checks + 1 evaluation

**Registers**: Fixed $K=64$ (32 RISC-V + 32 virtual), $d=1$
- Two reads ($\widetilde{\mathbf{ra}}_{\text{rs1}}, \widetilde{\mathbf{ra}}_{\text{rs2}}$), one write ($\widetilde{\mathbf{wa}}_{\text{rd}}$)
- No one-hot enforcement needed (addresses virtualized via bytecode read-checking)
- 2 read-checking + 1 write-checking + 1 evaluation sumcheck

**Complexity**:
- Prover: $O(T)$ using "local" algorithm (exploit access locality)
- Verifier: $O(\log T + \log K)$ (sumcheck rounds)
- Commitment: Only $\widetilde{\mathbf{ra}}$, $\widetilde{\mathbf{wa}}$, $\widetilde{\text{Inc}}$ (sparse!)

---

## Component 4: Bytecode Verification (Shout)

**Goal**: Prove correct instruction fetch-decode phase (CPU reads bytecode, extracts opcode/registers/immediates)

**Key difference from Component 2**: Bytecode is **read-only** (program is fixed), so this is **offline memory checking**, not mutable memory (Twist).

### The Bytecode Table

**Preprocessing** (one-time, before proving):
1. Compile guest program to RISC-V ELF binary
2. Extract `.text` section (machine code bytes)
3. Decode each 32-bit instruction into structured fields
4. Commit to bytecode table polynomial $\widetilde{\text{BC}}$

**Bytecode table structure** (size $L$ = program length in instructions):

For each instruction address $\ell \in \{0, 1, ..., L-1\}$, store:
- **Opcode**: 7-bit instruction type (ADD, SUB, LW, etc.)
- **rs1**: 5-bit source register 1 address
- **rs2**: 5-bit source register 2 address (R-type) or immediate bits (I-type)
- **rd**: 5-bit destination register address
- **funct3**: 3-bit function modifier
- **funct7**: 7-bit function modifier (R-type) or more immediate bits
- **Circuit flags**: Derived booleans (is_jump, is_branch, is_load, is_store, etc.)
- **Immediate**: Sign-extended immediate value (for I/S/B/U/J-type instructions)

**Commitment**: $C_{\text{BC}} = \text{Commit}(\widetilde{\text{BC}})$ sent to verifier in preprocessing

### Address Mapping

**PC to bytecode index**:
$$\text{index} = \frac{\text{PC} - 0x80000000}{4}$$

where:
- `0x80000000` = `RAM_START_ADDRESS` (RISC-V convention: code starts here)
- Division by 4: Instructions are 4-byte aligned (32-bit RISC-V)

**Example**: If PC = `0x80000010`, then bytecode index = `(0x10) / 4 = 4` (5th instruction)

### The Offline Memory Checking Problem

**At each cycle** $j \in \{0, ..., T-1\}$:
- CPU has program counter $\text{PC}(j)$
- Computes bytecode index $\ell_j = (\text{PC}(j) - 0x80000000) / 4$
- **Claims** to read instruction fields from $\text{BC}[\ell_j]$

**Must prove**: All claimed reads match the committed bytecode table

### Shout Protocol for Bytecode

Same structure as instruction lookups (Component 2), but with **read-only** table:

#### Step 1: One-Hot Access Matrix

Define $\mathbf{BC\text{-}ra} \in \{0,1\}^{L \times T}$ (bytecode read addresses):
$$\mathbf{BC\text{-}ra}(\ell, j) = \begin{cases} 1 & \text{if } \ell = \ell_j \text{ (PC at cycle } j \text{ maps to instruction } \ell) \\ 0 & \text{otherwise} \end{cases}$$

**One-hot property**: Each column (cycle $j$) has exactly one 1 (reads exactly one instruction per cycle)

#### Step 2: Read Value Polynomials

For each field $f$ (opcode, rs1, rs2, rd, flags, immediate):

$$\mathbf{rv}_f(j) = \sum_{\ell \in \{0,1\}^{\log L}} \mathbf{BC\text{-}ra}(\ell, j) \cdot \text{BC}_f(\ell)$$

**Interpretation**: Value of field $f$ at cycle $j$ equals the value in bytecode table at the accessed instruction

#### Step 3: Batch Verification via Sumcheck

For random challenge $r' \in \mathbb{F}^{\log T}$ (from Spartan outer sumcheck):

$$\widetilde{\mathbf{rv}}_f(r') \stackrel{?}{=} \sum_{\ell \in \{0,1\}^{\log L}} \widetilde{\mathbf{BC\text{-}ra}}(\ell, r') \cdot \widetilde{\text{BC}}_f(\ell)$$

**Separate sumcheck for each field** (or batch them together via random linear combination):
- Opcode read-checking
- rs1 read-checking
- rs2 read-checking
- rd read-checking
- Immediate read-checking

**All batched into Stage 2** of JoltDAG!

#### Step 4: Well-Formedness Checks

Must also prove $\mathbf{BC\text{-}ra}$ is well-formed (Boolean + one-hot):

**Booleanity** (zero-check): For random $r_\ell, r_j$:
$$\sum_{\ell, j} \gamma^j \cdot \widetilde{\mathbf{BC\text{-}ra}}(\ell, j) \cdot [1 - \widetilde{\mathbf{BC\text{-}ra}}(\ell, j)] \stackrel{?}{=} 0$$

**Hamming weight** (evaluation check): For random $r_j$:
$$\sum_{\ell \in \{0,1\}^{\log L}} \widetilde{\mathbf{BC\text{-}ra}}(\ell, r_j) \stackrel{?}{=} 1$$

**Proves**: Exactly one instruction read per cycle (no skipping, no reading multiple)

### Virtualization: Deriving Circuit Flags

**Key optimization**: Circuit flags not stored explicitly in bytecode table - **derived from opcode**!

**Circuit flags** (Boolean indicators):
- `is_jump`: 1 if JAL/JALR instruction
- `is_branch`: 1 if BEQ/BNE/BLT/BGE/BLTU/BGEU
- `is_load`: 1 if LB/LH/LW/LBU/LHU
- `is_store`: 1 if SB/SH/SW
- `is_imm`: 1 if I-type (uses immediate, not rs2)
- `is_concat`: 1 if operands must be concatenated for lookup (e.g., MUL)

**Derivation formula** (example for `is_jump`):
$$\text{is\_jump} = \text{opcode} \stackrel{?}{=} 0b1101111 \quad \lor \quad \text{opcode} \stackrel{?}{=} 0b1100111$$

**How proven**: Via bytecode read-checking!
- Verifier computes flag from opcode directly
- Prover's claimed flag must match
- If mismatch, read-checking sumcheck fails (Schwartz-Zippel)

**Why virtualization works**:
- Flags are **deterministic functions** of opcode (no hiding!)
- Opcode already committed in bytecode table
- Sumcheck verification implicitly checks flag derivation

### Virtualization: Register Addresses

**Similar trick**: Register addresses (rs1, rs2, rd) are **hardcoded in bytecode**, so one-hot checks for register Twist (Component 3) can be **skipped**!

**Why**: If bytecode says "ADD x5, x1, x2", then:
- rs1 = 1 (hardcoded in bytecode bit [19:15])
- rs2 = 2 (hardcoded in bytecode bit [24:20])
- rd = 5 (hardcoded in bytecode bit [11:7])

**Soundness via bytecode read-checking**:
- Prover claims rs1 = 1 at cycle $j$
- Bytecode read-checking proves rs1 field read from bytecode matches claimed value
- Bytecode commitment ensures bits [19:15] actually encode register 1
- Therefore, register one-hot $\mathbf{ra}_{\text{rs1}}$ is implicitly verified!

**Formula** (from theory doc):
$$\mathbf{ra}_{\text{rs1}}(r, j) = \sum_{k \in \{0,1\}^{\log L}} \text{eq}(j, k) \cdot \mathbf{BC\text{-}ra}(k, j) \cdot \text{rs1\_from\_bytecode}(\ell)$$

where rs1_from_bytecode($\ell$) extracts bits [19:15] from instruction $\ell$.

**Savings**: Eliminates $O(\log K)$ rounds of Hamming weight sumchecks for register addresses!

### Concrete Example: Fetch-Decode for 3 Instructions

**Bytecode table** (preprocessed):
| $\ell$ | Address | Instruction | Opcode | rs1 | rs2 | rd | imm | Flags |
|--------|---------|-------------|--------|-----|-----|----|----|-------|
| 0 | 0x80000000 | ADD x5,x1,x2 | 0x33 | 1 | 2 | 5 | - | - |
| 1 | 0x80000004 | ADDI x6,x5,10 | 0x13 | 5 | - | 6 | 10 | is_imm=1 |
| 2 | 0x80000008 | BEQ x6,x7,8 | 0x63 | 6 | 7 | - | 8 | is_branch=1 |

**Execution trace** ($T=3$ cycles):
| Cycle $j$ | PC | Bytecode index $\ell_j$ | Claimed opcode | Claimed rs1 | Claimed rd |
|-----------|-------------|------------------------|---------------|-------------|-----------|
| 0 | 0x80000000 | 0 | 0x33 | 1 | 5 |
| 1 | 0x80000004 | 1 | 0x13 | 5 | 6 |
| 2 | 0x80000008 | 2 | 0x63 | 6 | - |

**Access matrix** (one-hot):
$$\mathbf{BC\text{-}ra} = \begin{bmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

**Read-checking sumcheck** (for opcode):
$$\widetilde{\text{opcode}}(r') \stackrel{?}{=} \sum_{\ell=0}^{2} \widetilde{\mathbf{BC\text{-}ra}}(\ell, r') \cdot \widetilde{\text{BC}}_{\text{opcode}}(\ell)$$

where:
- $\widetilde{\text{BC}}_{\text{opcode}} = [0x33, 0x13, 0x63]$ (MLE of opcode column)
- $\widetilde{\mathbf{BC\text{-}ra}}(\ell, r')$ is sparse (only 3 nonzeros!)

**Verifier's check**: After sumcheck, evaluate at random point $(r_\ell, r')$ and verify:
- $\widetilde{\mathbf{BC\text{-}ra}}(r_\ell, r')$: Opening from commitment ✓
- $\widetilde{\text{BC}}_{\text{opcode}}(r_\ell)$: Opening from bytecode commitment ✓

### Complexity

**Commitment cost** (preprocessing):
- Bytecode table: $L$ instructions × 8 fields ≈ $8L$ field elements
- For $L = 2^{16}$ (64K instruction program): ~512 KB commitment

**Prover cost** (runtime):
- Access matrix: $T$ nonzeros (one-hot, sparse!)
- Read-checking: $O(T + L)$ per field via sparse sumcheck
- Well-formedness: $O(T + L)$ for Booleanity + Hamming weight
- **Total**: $O(T + L)$ field operations (~5-10% of total prover time)

**Verifier cost**:
- Sumcheck rounds: $O(\log L + \log T)$ per field
- **Total**: $O((\log L + \log T) \times \text{num fields})$ ≈ $O(\log T)$ (since $L \approx T$ typically)

**Why efficient**:
- Bytecode committed **once** in preprocessing (amortized over all executions!)
- Sparse access matrix (only $T$ entries, not $L \times T$)
- Field derivation (flags, registers) virtualized - no extra commitments
- Offline checking (no mutable state like RAM/registers)

---

## Component 5: Polynomial Commitments (Dory)

**PCS Landscape**:

| Scheme | Setup | Prover | Verifier | Proof Size |
|--------|-------|--------|----------|------------|
| KZG | Trusted | $O(N)$ | $O(1)$ (2 pairings) | $O(1)$ |
| Bulletproofs | Transparent | $O(N)$ | $O(N)$ | $O(\log N)$ |
| Hyrax | Transparent | $O(N)$ field ops | $O(\sqrt{N})$ | $O(\sqrt{N})$ |
| **Dory** | **Transparent** | **$O(N)$** | **$O(\log N)$** | **$O(\log N)$** |

### Dory Architecture: Two-Tiered Commitment + Inner-Pairing Product

**Foundation**: Hyrax matrix structure + pairing-based compression

**Pairing groups**: BN254 curve with $e: \mathbb{G}_1 \times \mathbb{G}_2 \to \mathbb{G}_T$

#### Layer 1: Row Commitments (Pedersen in $\mathbb{G}_1$)

For polynomial $\widetilde{p}$ with coefficient matrix $M \in \mathbb{F}^{m \times m}$ (where $N = m^2$):

**Commit to each row** $i$:
$$V_i = \sum_{j=1}^m M_{i,j} \cdot G_{1,j} + r_i \cdot H_1 \in \mathbb{G}_1$$

where:
- $G_{1,1}, \ldots, G_{1,m} \in \mathbb{G}_1$: Public generators (from SRS)
- $H_1 \in \mathbb{G}_1$: Blinding generator
- $r_i \in \mathbb{F}$: Random blinding factor for row $i$

**Row commitment vector**: $\vec{V} = (V_1, \ldots, V_m) \in \mathbb{G}_1^m$

#### Layer 2: AFGHO Compression (Pairing in $\mathbb{G}_T$)

**Compress row commitments** into single $\mathbb{G}_T$ element:
$$C_M = \sum_{i=1}^m e(V_i, G_{2,i}) = e\left(\sum_{i=1}^m M_{i,j} G_{1,j}, G_{2,i}\right) \cdot e(H_1, \sum_{i=1}^m r_i G_{2,i}) \in \mathbb{G}_T$$

where $G_{2,1}, \ldots, G_{2,m} \in \mathbb{G}_2$ are public generators.

**Key property**: Pairing allows combining row commitments without revealing individual $V_i$ values.

#### Opening Protocol: Prove $\widetilde{p}(\vec{r}) = y$

**Given**:
- Random challenge $\vec{r} = (r_1, \ldots, r_m, r_{m+1}, \ldots, r_{2m}) \in \mathbb{F}^{2m}$
- Split into $\vec{L} = (L_1, \ldots, L_m)$ and $\vec{R} = (R_1, \ldots, R_m)$ (Lagrange basis)

**Step 1**: Prover computes intermediate vector $\vec{v} = M \cdot \vec{R}$ (field arithmetic)
$$v_i = \sum_{j=1}^m M_{i,j} \cdot R_j \quad \text{for } i = 1, \ldots, m$$

**Step 2**: Prover computes Pedersen commitment to $\vec{v}$:
$$C_v = \sum_{i=1}^m L_i \cdot V_i = \sum_{i=1}^m L_i \left(\sum_{j=1}^m M_{i,j} G_{1,j} + r_i H_1\right) \in \mathbb{G}_1$$

**Step 3**: Prover commits to $\vec{R}$ and $y$ in $\mathbb{G}_2$ and $\mathbb{G}_T$:
$$D_R = \sum_{j=1}^m R_j \cdot G_{2,j} + s \cdot H_2 \in \mathbb{G}_2$$
$$C_y = G_T^y \cdot e(H_1, H_2)^t \in \mathbb{G}_T$$

where $s, t$ are random blinding factors.

**Step 4**: Run **inner-pairing product argument** to prove:
$$y = \langle \vec{v}, \vec{R} \rangle = \sum_{i=1}^m v_i \cdot R_i$$

#### Inner-Pairing Product Argument (Dory Core)

**Claim**: $C_y$ commits to $y = \langle \vec{v}, \vec{R} \rangle$ where $C_v$ commits to $\vec{v}$ and $D_R$ commits to $\vec{R}$.

**Folding protocol** ($\log m$ rounds):

**Round $i$**:
1. Prover splits: $\vec{v} = (\vec{v}_L, \vec{v}_R)$, $\vec{R} = (\vec{R}_L, \vec{R}_R)$ (each half size $m/2^i$)
2. Prover computes cross-terms:
   $$z_{LR} = \langle \vec{v}_L, \vec{R}_R \rangle, \quad z_{RL} = \langle \vec{v}_R, \vec{R}_L \rangle$$
3. Prover commits:
   $$C_{LR} = G_T^{z_{LR}} \cdot e(H_1, H_2)^{r_{LR}}, \quad C_{RL} = G_T^{z_{RL}} \cdot e(H_1, H_2)^{r_{RL}}$$
4. Prover sends $(C_{LR}, C_{RL})$ to verifier
5. Verifier sends random challenge $\alpha_i \in \mathbb{F}$
6. Both fold:
   $$\vec{v}' = (1-\alpha_i) \vec{v}_L + \alpha_i \vec{v}_R$$
   $$\vec{R}' = \alpha_i^{-1} \vec{R}_L + \vec{R}_R$$
7. Verifier folds commitments homomorphically:
   $$C_v' = C_{v,L}^{1-\alpha_i} \cdot C_{v,R}^{\alpha_i}$$
   $$D_R' = D_{R,L}^{\alpha_i^{-1}} \cdot D_{R,R}$$
   $$C_y' = C_y \cdot C_{LR}^{(1-\alpha_i)} \cdot C_{RL}^{\alpha_i}$$

**Base case** (after $\log m$ rounds, dimension = 1):
- Prover reveals: $v_1, R_1, y$
- Verifier checks: $y \stackrel{?}{=} v_1 \cdot R_1$
- Verifier checks commitments match revealed values

**Key folding identity**: Preserves inner product relation:
$$\langle (1-\alpha)\vec{v}_L + \alpha\vec{v}_R, \alpha^{-1}\vec{R}_L + \vec{R}_R \rangle$$
$$= \alpha^{-1}(1-\alpha) \langle\vec{v}_L, \vec{R}_L\rangle + (1-\alpha)\langle\vec{v}_L, \vec{R}_R\rangle + \langle\vec{v}_R, \vec{R}_L\rangle + \alpha\langle\vec{v}_R, \vec{R}_R\rangle$$

The cross-terms $C_{LR}, C_{RL}$ allow verifier to fold $C_y$ correctly.

#### Batched Openings in Jolt

**Stage 5 workflow**:
1. Accumulate $k$ opening claims from stages 1-4: $(f_i, \vec{r}_i, v_i)$ for $i=1,\ldots,k$
2. Sample random $\rho$ from transcript (Fiat-Shamir)
3. Combine via random linear combination:
   $$f_{\text{batch}}(\vec{r}) = \sum_{i=1}^k \rho^i f_i(\vec{r}_i) \cdot \text{eq}(\vec{r}, \vec{r}_i)$$
4. Single Dory opening proves: $f_{\text{batch}}(\vec{r}_{\text{sc}}) = \sum_{i=1}^k \rho^i v_i \cdot \text{eq}(\vec{r}_{\text{sc}}, \vec{r}_i)$

**Soundness**: By Schwartz-Zippel, if any individual $f_i(\vec{r}_i) \neq v_i$, batched claim fails w.h.p.

#### Complexity

**Commitment**:
- Layer 1: $m$ Pedersen commitments in $\mathbb{G}_1$ ($O(m^2)$ scalar muls)
- Layer 2: 1 AFGHO compression via pairings ($O(m)$ pairings)

**Opening proof**:
- Prover: $O(m)$ field ops (compute $\vec{v}$) + $O(\log m)$ group ops (folding)
- Verifier: $O(\log m)$ exponentiations in $\mathbb{G}_T$ + $O(1)$ pairings
- Proof size: $2\log m$ group elements ($C_{LR}, C_{RL}$ per round) ≈ 10-13 KB

**SRS**:
- Prover: $O(\sqrt{N})$ elements in $\mathbb{G}_1, \mathbb{G}_2$ (393 KB for $N=2^{24}$)
- Verifier: $O(\sqrt{N})$ elements (25 KB)

**Jolt Stage 5** (batching ~30 polynomials, $N=2^{16}$):
- Verifier: $5 \times \log_2 N = 80$ GT exponentiations (naive) or $4 \times \log_2 N = 64$ (optimized)
- With Stage 6 hints: Reduced to ~29 GT multiplications + Stage 6 verification

---

## Hyrax: Matrix-Based Polynomial Commitment

**Core idea**: Avoid sum-check in Bulletproofs/IPA setting (scalar muls 3000× slower than field ops)

**Univariate case**: For $p(X) = \sum_{i=0}^{N-1} c_i X^i$ with $N = m^2$:

1. **Reshape** $\vec{c}$ into $m \times m$ matrix $M$
2. **Encode random point** $r$ into vectors:
   - $\vec{a} = (1, r, r^2, \ldots, r^{m-1})$
   - $\vec{b} = (1, r^m, r^{2m}, \ldots, r^{(m-1)m})$
3. **Evaluation formula**: $p(r) = \vec{b}^T \cdot M \cdot \vec{a}$

**Commitment**: Column commitments $C_j = \langle M^{(j)}, \vec{G} \rangle$ for $j=1,\ldots,m$

**Opening** (prove $p(r) = y$):
1. Prover computes $\vec{v} = M \cdot \vec{a}$ (field arithmetic!)
2. Prover sends $\vec{v}$ ($m$ field elements)
3. Verifier checks:
   - **Homomorphic**: $C_{M \cdot a} = \sum_j C_j \cdot a_{j-1} \stackrel{?}{=} \langle \vec{v}, \vec{G} \rangle$
   - **Evaluation**: $\vec{b}^T \cdot \vec{v} \stackrel{?}{=} y$

**Multilinear case**: For MLE $\widetilde{p}(x_1, \ldots, x_n)$ at $\vec{r} = (r_1, \ldots, r_n)$:

1. **Split** $\vec{r}$ into chunks: $(r_1, r_2)$ and $(r_3, r_4)$ for $n=4$
2. **Lagrange basis vectors**:
   $$\vec{L} = [(1-r_1)(1-r_2), (1-r_1)r_2, r_1(1-r_2), r_1 r_2]^T$$
   $$\vec{R} = [(1-r_3)(1-r_4), (1-r_3)r_4, r_3(1-r_4), r_3 r_4]^T$$
3. **Opening**: $\vec{v} = M \cdot \vec{R}$, verify $\vec{L}^T \cdot \vec{v} = y$

**Complexity**:
- Commitment: $\sqrt{N}$ group elements
- Prover: $O(N)$ field multiplications (no scalar muls!)
- Verifier: $O(\sqrt{N})$ scalar multiplications

**vs Bulletproofs**:
- Bulletproofs prover: $O(N)$ scalar muls (~300× slower)
- Bulletproofs verifier: $O(N)$ scalar muls
- Hyrax: Field ops for prover, $O(\sqrt{N})$ scalar muls for verifier

---

## The Five-Stage Proof DAG

**Critical Context: What Does the Verifier Have?**

The verifier **never sees the execution trace**. The verifier only receives:

1. **Preprocessing (public, program-specific)**:
   - Full bytecode (actual instructions, not commitment)
   - Memory layout (defines I/O regions, stack, heap addresses)
   - Initial RAM state (static data)
   - Dory verifier generators

2. **Per-execution inputs (public)**:
   - `program_io`: Claimed inputs and **claimed outputs**
   - `panic`: Boolean flag if program panicked
   - Trace length $T$ (number of cycles)

3. **Proof (from prover)**:
   - Polynomial commitments (36 commitments, ~7 KB)
   - Sumcheck proofs for Stages 1-4 (~8 KB)
   - Dory batched opening proof (~13 KB)

**How verification works without the trace**:
- Prover commits to witness polynomials (register values, memory, instruction outputs, etc.)
- Sumchecks reduce verification to checking polynomial evaluations at random points
- Verifier computes expected evaluations from **public data** (bytecode, claimed I/O, memory layout)
- If evaluations match, trace is consistent with public data (with overwhelming probability)

**Example**: Output verification
- Verifier knows claimed outputs (public)
- Verifier knows I/O memory region addresses (from memory_layout)
- Sumcheck proves: "final RAM state at I/O addresses = claimed outputs"
- Verifier never sees intermediate RAM values, only evaluation claim at random point

---

### Stage 1: R1CS Outer Sumcheck (1 sumcheck)

**Heuristic**: Batch all VM constraints into single claim via random linear combination

#### 1.1: Spartan Outer Sumcheck
**Purpose**: Batch all $N = 30T$ R1CS constraints into single claim (PC updates + component wiring)
$$\sum_{t \in \{0,1\}^{\log N}} \text{eq}(r_t, t) \cdot [(\widetilde{Az})(t) \circ (\widetilde{Bz})(t) - (\widetilde{Cz})(t)] = 0$$

**What this proves**:
- PC updates correct (sequential, jumps, branches)
- Component wiring (RAM ↔ Registers, Instruction output → Register input)
- Circuit flag consistency (jump_flag, branch_flag, etc.)

**Protocol**:
- **Rounds**: $\log N = \log(30T)$
- **Prover**: Sends univariate polynomial per round
- **Verifier**: Sends random challenge $r_t$ (total: vector $\vec{r}_t \in \mathbb{F}^{\log N}$)

**Output claims** (virtual polynomials from Spartan):
- $(\widetilde{Az})(r_t)$: Left matrix-vector product at random point
- $(\widetilde{Bz})(r_t)$: Right matrix-vector product at random point
- $(\widetilde{Cz})(r_t)$: Output matrix-vector product at random point

These become **input claims for Stage 2** (Spartan product sumchecks)

**Why Stage 1 is special**:
- Only Spartan runs in Stage 1 (not other components!)
- This sumcheck creates random challenges that other components will use
- Sets up virtual polynomial evaluation claims for rest of DAG

**Implementation**: `spartan_dag.stage1_prove()` in `jolt_dag.rs:102-104`

---

### Stage 2: Batched Sumchecks (6 parallel sumchecks)

**Heuristic**: Reduce vector-matrix products, prove memory read consistency, verify table lookups

#### 2.1: Spartan Inner Sumcheck
**Purpose**: Reduce matrix-vector products $(\widetilde{Az})(r_t)$, $(\widetilde{Bz})(r_t)$, $(\widetilde{Cz})(r_t)$ to claims about $\widetilde{z}$
$$(\widetilde{Az})(r_t) = \sum_{j \in \{0,1\}^{\log m}} \widetilde{A}(r_t, j) \cdot \widetilde{z}(j)$$
- **Batches Az, Bz, Cz** using random linear combination (RLC) - three claims in one sumcheck
- **Rounds**: $\log m$ (witness length)
- **Output**: Random point $r_c$, claims $\widetilde{A}(r_t, r_c)$, $\widetilde{B}(r_t, r_c)$, $\widetilde{C}(r_t, r_c)$, $\widetilde{z}(r_c)$

#### 2.2: Registers ReadWriteChecking (Twist)
**Purpose**: Prove register reads (rs1, rs2) and write (rd) are consistent
- Combines 2 read-checking + 1 write-checking into single batched sumcheck
- **Read formula**: $\widetilde{\mathbf{rv}}_{\text{rs1}}(r') = \sum_{j,r} \text{eq}(r', j) \cdot \widetilde{\mathbf{ra}}_{\text{rs1}}(r, j) \cdot \widetilde{f}_{\text{reg}}(r, j)$
- **Write formula**: $\widetilde{\mathbf{wv}}_{\text{rd}}(r'') = \sum_r \widetilde{\mathbf{wa}}_{\text{rd}}(r, r'') \cdot [\widetilde{f}_{\text{reg}}(r, r'') + \widetilde{\text{Inc}}(r'')]$
- **Output**: Register value claims for Stage 3

#### 2.3: RAM RafEvaluation (Twist)
**Purpose**: Evaluate RAF (Read-After-Finalize) polynomial for RAM
- Part of Twist protocol's time-ordered verification
- **Output**: RAF evaluation claim

#### 2.4: RAM ReadWriteChecking (Twist)
**Purpose**: Prove RAM reads return correct values and writes update correctly
$$\widetilde{\mathbf{rv}}_{\text{RAM}}(r') = \sum_{j,k} \text{eq}(r', j) \cdot \widetilde{\mathbf{ra}}_{\text{RAM}}(k, j) \cdot \widetilde{f}_{\text{RAM}}(k, j)$$
- **Output**: RAM value claims for Stage 3

#### 2.5: RAM OutputSumcheck (Twist)
**Purpose**: Verify claimed program outputs via sumcheck over I/O memory region

**Zero-check formula**:
$$0 = \sum_k \text{eq}(r_{\text{addr}}, k) \cdot \text{io\_mask}(k) \cdot [\widetilde{\text{Val}}_{\text{final}}(k) - \widetilde{\text{Val}}_{\text{io}}(k)]$$

**How verifier knows claimed outputs** (key insight):
- Verifier receives `program_io: JoltDevice` containing **claimed outputs** as public data
- Verifier has `memory_layout` defining I/O region addresses (input_start to RAM_START_ADDRESS)
- $\widetilde{\text{Val}}_{\text{io}}$ = MLE of claimed outputs in I/O region (verifier computes this from public data!)
- $\widetilde{\text{Val}}_{\text{final}}(r)$ = virtual polynomial claim from RAM final state (from earlier sumcheck)
- $\text{io\_mask}(k)$ = 1 if $k$ in I/O region, 0 otherwise (verifier computes from memory_layout)

**What this proves**: Final RAM state at I/O addresses equals the claimed outputs

**Verifier never sees the full trace** - only:
1. Claimed outputs (public)
2. Memory layout (public, from preprocessing)
3. Polynomial evaluation claims at random points (from sumcheck)

**Output**: Output verification claim

#### 2.6: Instruction Lookups Booleanity (Shout)
**Purpose**: Prove instruction lookup address chunks are Boolean (one-hot or 0/1)
- Ensures valid table access patterns for instruction execution
- **Output**: Booleanity claim for Stage 3

**Total: 6 sumchecks batched together** in Stage 2

---

### Stage 3: Evaluation Sumchecks (8 parallel sumchecks)

**Heuristic**: Prove sparse matrix evaluations, finalize memory state consistency

#### 3.1: Spartan PCSumcheck
**Purpose**: Prove sparse matrix evaluations $\widetilde{A}(r_t, r_c)$ correct using SPARK
$$\widetilde{A}(r_t, r_c) = \sum_{(i,j) \in \text{Nonzeros}(A)} A_{i,j} \cdot \text{eq}(r_t, i) \cdot \text{eq}(r_c, j)$$
- Exploits sparsity: prover cost $O(s)$ where $s = $ number of nonzeros (~3% of $N \times m$)
- Batches $A$, $B$, $C$ matrix evaluations
- **Output**: Matrix evaluation claims verified from sparse representation

#### 3.2: Spartan ProductVirtualizationSumcheck
**Purpose**: Prove product polynomial correctness for R1CS constraint checking
- Verifies $(\widetilde{Az} \circ \widetilde{Bz})(r) = \widetilde{Az}(r) \cdot \widetilde{Bz}(r)$
- **Output**: Product evaluation claim

#### 3.3: Registers ValEvaluationSumcheck (Twist)
**Purpose**: Prove final register state = initial + all accumulated increments
$$\widetilde{f}_{\text{reg,final}}(r) = \widetilde{f}_{\text{reg,init}}(r) + \sum_j \widetilde{\mathbf{wa}}_{\text{rd}}(r, j) \cdot \widetilde{\text{Inc}}(j)$$
- Completes 2-stage Twist protocol for register consistency
- **Output**: Final register state claim

#### 3.4: RAM ValEvaluationSumcheck (Twist)
**Purpose**: Prove RAM value consistency via increment accumulation
- Part of Twist address-ordered verification
- **Output**: RAM value evaluation claim

#### 3.5: RAM ValFinalSumcheck (Twist)
**Purpose**: Prove final RAM state = initial + all accumulated increments
$$\widetilde{f}_{\text{RAM,final}}(r) = \widetilde{f}_{\text{init}}(r) + \sum_{j,k} \widetilde{\mathbf{wa}}(k, j) \cdot \widetilde{\text{Inc}}(j)$$
- **Output**: Final memory state claim

#### 3.6: RAM HammingBooleanitySumcheck (Twist)
**Purpose**: Prove address indicator values are Boolean (0 or 1)
- Ensures valid one-hot encoding for memory addresses
- **Output**: Booleanity claim for RAM addresses

#### 3.7: Instruction Lookups ReadRafSumcheck (Shout)
**Purpose**: Prove instruction table lookups return correct values
- Uses prefix-suffix sumcheck for 128-bit lookup indices
- **Output**: Table value correctness claim

#### 3.8: Instruction Lookups HammingWeightSumcheck (Shout)
**Purpose**: Prove exactly one table entry accessed per instruction (one-hot)
$$\sum_k \widetilde{\mathbf{ra}}(k, r_{\text{time}}) = 1$$
- **Output**: One-hot verification claim

**Total: 8 sumchecks batched together** in Stage 3

---

### Stage 4: Final Component Checks (7 sumchecks)

**Heuristic**: Complete remaining component verifications, prepare for batched opening

#### 4.1: RAM HammingWeightSumcheck (Twist)
**Purpose**: Prove exactly one RAM cell accessed per cycle (one-hot)
$$\sum_k \widetilde{\mathbf{ra}}(k, r_{\text{time}}) = 1$$
- Handles zero-padded rows (cycles without RAM access)
- **Output**: One-hot verification for RAM addresses

#### 4.2: RAM BooleanitySumcheck (Twist)
**Purpose**: Prove RAM address chunks are Boolean (0 or 1)
- Ensures valid chunked address representation
- **Output**: Booleanity claim for RAM

#### 4.3: RAM RaSumcheck (Twist)
**Purpose**: Finalize RAM address virtualization
- Completes Twist protocol for RAM consistency
- **Output**: Final RAM verification complete

#### 4.4: Bytecode ReadRafSumcheck (Shout)
**Purpose**: Prove instruction fetch sequence matches committed bytecode
- Offline memory checking: reads from bytecode table
- **Output**: Bytecode read correctness claim

#### 4.5: Bytecode BooleanitySumcheck (Shout)
**Purpose**: Prove bytecode access indicators are Boolean
- **Output**: Booleanity claim for bytecode

#### 4.6: Bytecode HammingWeightSumcheck (Shout)
**Purpose**: Prove exactly one bytecode entry accessed per cycle (one-hot)
- **Output**: One-hot verification for bytecode

#### 4.7: Instruction Lookups RaSumcheck (Shout)
**Purpose**: Finalize instruction lookup address virtualization
- Proves virtual "ra" polynomial equals product of all chunk polynomials
- Completes 3-stage Shout protocol for instruction lookups
- **Output**: Instruction lookup verification complete

**Total: 7 sumchecks batched together** in Stage 4

**Total evaluation claims accumulated**: ~25-30 polynomial opening claims in StateManager (Stages 1-4)

---

### Stage 5: Batched Opening Proof (1 sumcheck + Dory protocol)

**Heuristic**: Verify all accumulated polynomial evaluations simultaneously via single proof

#### 5.1: Batch Reduction Sumcheck
**Purpose**: Reduce $k \approx 30$ openings to 1 joint opening via random linear combination
$$\text{Claim: } \sum_{i=1}^k \rho^i \cdot [\widetilde{p}_i(r_i) - v_i] = 0$$

**Batched commitment** (verifier computes homomorphically):
$$C_{\text{joint}} = \prod_{i=1}^k C_i^{\rho^i}$$

**Key property**: If any individual claim $\widetilde{p}_i(r_i) \neq v_i$ is false, batched claim fails w.h.p. $\geq 1 - k/|\mathbb{F}|$

#### 5.2: Dory Opening Proof
**Purpose**: Prove $\widetilde{p}_{\text{joint}}(r_{\text{sc}}) = v_{\text{joint}}$ for batched polynomial

**Protocol** (from Component 5 - Dory section):
1. Prover computes $\vec{v} = M \cdot \vec{R}$ (intermediate vector)
2. Runs inner-pairing product argument ($\log m$ rounds)
3. Each round: Send cross-terms $(C_{LR}, C_{RL})$, fold vectors and commitments
4. Base case: Reveal $v_1, R_1, y$ and verify $y = v_1 \cdot R_1$

**Verifier cost**:
- Without hints: ~109 GT exponentiations (~1.12B cycles @ ~10M each)
- With hints: ~29 GT multiplications (~1.6M cycles) + Stage 6 verification

**Hint mechanism** (Stage 6 integration):
- Prover sends precomputed GT exponentiation results (11 KB)
- Verifier accepts hints, uses for fast verification (~42M cycles total for Stage 5)
- Stage 6 proves hint correctness (~321-361M cycles)
- **Net speedup**: ~5.9× faster than verifying without hints

**Output**: Complete proof of all 30 polynomial openings ✓

---

### Summary: Complete Five-Stage Breakdown

**Total sumchecks**: 22 sumchecks across Stages 1-4 + 1 batched reduction sumcheck in Stage 5

| Stage | Count | Sumchecks | Components |
|-------|-------|-----------|------------|
| **Stage 1** | 1 | Spartan Outer | R1CS constraint batching |
| **Stage 2** | 6 | Spartan Inner, Registers ReadWriteChecking, RAM (RafEvaluation + ReadWriteChecking + OutputSumcheck), Instruction Lookups Booleanity | Matrix products, memory consistency, lookup booleanity |
| **Stage 3** | 8 | Spartan (PCSumcheck + ProductVirtualization), Registers ValEvaluation, RAM (ValEvaluation + ValFinal + HammingBooleanity), Instruction Lookups (ReadRaf + HammingWeight) | Sparse matrix evaluation, memory finalization, lookup table verification |
| **Stage 4** | 7 | RAM (HammingWeight + Booleanity + RaSumcheck), Bytecode (ReadRaf + Booleanity + HammingWeight), Instruction Lookups RaSumcheck | Final consistency checks, bytecode verification, address virtualization |
| **Stage 5** | 1 + Dory | Batch Reduction Sumcheck + Dory opening proof | All ~25-30 polynomial openings verified together |

**Implementation reference**: See `jolt-core/src/zkvm/dag/jolt_dag.rs:383-569` for exact stage orchestration

---

## Stage 6: SNARK Composition (Optional)

**Purpose**: Verify GT exponentiation hints from Stage 5

**Architecture**: Two-layer SNARK composition
- **Layer 1** (Stage 5): BN254-based Dory proof (main Jolt proof)
- **Layer 2** (Stage 6): Grumpkin-based Hyrax proof (GT exponentiation verification)

**Why 2-cycles?**: BN254's base field Fq = Grumpkin's scalar field Fr
- Native field arithmetic in Layer 2
- No expensive non-native field simulation

**What gets proven**: 109 GT exponentiations satisfy square-and-multiply constraints

### The SZ-Check Protocol (ExpSumcheck)

**For each exponentiation** $g^e$ where $e = \sum_{i=0}^{253} b_i 2^i$:

**Accumulator recurrence** (over Fq12 extension field):
$$\rho_{i+1} = \rho_i^2 \cdot g^{b_i} - q_i \cdot h(g)$$

where:
- $\rho_0 = 1$, $\rho_{254} = g^e$ (result)
- $b_i \in \{0,1\}$ (exponent bits)
- $q_i \in \text{Fq12}$ (quotient from tower field arithmetic)
- $h(g) = g^{12} - 2$ (minimal polynomial for BN254's Fq12)

**Why quotients?**: Fq12 arithmetic requires reducing modulo $h(g)$:
$$\rho_i^2 \cdot g^{b_i} = \rho_{i+1} + q_i \cdot h(g)$$

**Constraints per step**: 3 equations
1. Squaring: $\rho_i^2 = s_i$
2. Conditional multiply: $s_i \cdot g^{b_i} = m_i$
3. Reduction: $m_i - q_i \cdot h(g) = \rho_{i+1}$

**Batching**: All 254 steps verified via single sumcheck with RLC
$$\sum_{j=0}^{253} \gamma^j \cdot [e_{k,j}] = 0$$

where $e_{k,j}$ combines the 3 constraints at step $j$.

**Sumcheck rounds**: 4 rounds (one per variable in 4-variable MLE)
- Each MLE packs Fq12 element (12 coefficients → 16 evaluations with padding)

**Total for 109 exponentiations**:
- 109 instances batched via BatchedSumcheck
- 4 rounds of interaction (shared across all instances)
- Not 436 rounds (109 × 4) - that's the key batching win

### Hyrax Commitments for GT Witness

**What's committed**: For each exponentiation (109 total):
- **Base MLE**: $\widetilde{g}$ (1 polynomial, 4 vars, 16 coeffs)
- **Rho MLEs**: $\widetilde{\rho}_0, \ldots, \widetilde{\rho}_{254}$ (255 polynomials)
- **Quotient MLEs**: $\widetilde{q}_1, \ldots, \widetilde{q}_{254}$ (254 polynomials)

**Total polynomials**: $109 \times (1 + 255 + 254) = 55,590$ MLEs

**Hyrax structure**: Each 4-variable MLE → $4 \times 4$ matrix → 4 row commitments
$$C_i = \sum_{j=0}^3 M_{i,j} \cdot G_j$$

**Total commitments**: $55,590 \times 4 = 222,360$ Grumpkin points (~7.1 MB)

**Batched opening**: All 55,590 polynomials opened at random challenge via:
1. Random linear combination: $\widetilde{p}_{\text{batch}} = \sum_i \beta_i \widetilde{p}_i$
2. Single Hyrax opening proof for $\widetilde{p}_{\text{batch}}$

### Component A: Commitment Batching

**Goal**: Reduce 55,590 commitment checks → 1 check

**Random linear combination**:
$$C_{\text{batch}} = \sum_{i=1}^{55,590} \beta_i C_i$$

**Verifier work**: 222,360 scalar multiplications (Pippenger MSM)

**Soundness**: By Schwartz-Zippel, cheating detected with probability $1 - \frac{1}{|\mathbb{F}|}$

**Cost**: ~80-120M cycles (Grumpkin scalar multiplications)

### Component B: Batched Sumcheck

**Goal**: Combine 109 ExpSumcheck instances

**Batching coefficients**: Verifier samples $\{\beta_1, \ldots, \beta_{109}\}$

**Combined claim**:
$$\sum_{k=1}^{109} \beta_k \cdot \left[\sum_{x \in \{0,1\}^4} E_k(x)\right] = 0$$

where $E_k(x)$ is the RLC of 254 constraints for exponentiation $k$.

**Prover work**: For each of 4 rounds, compute:
$$s_j(X_j) = \sum_{k=1}^{109} \beta_k \cdot \sum_{x_{j+1}, \ldots, x_4 \in \{0,1\}} E_k(r_1, \ldots, r_{j-1}, X_j, x_{j+1}, \ldots, x_4)$$

**Verifier work**: 4 checks of form $s_j(0) + s_j(1) \stackrel{?}{=} H_{j-1}$

**Why batching matters**:
- **Without**: 109 instances × 4 rounds = 436 rounds of interaction
- **With**: 4 rounds total (all instances proceed together)

**Cost**: ~240M cycles (dominated by polynomial evaluations over 109 instances)

### Component C: Hyrax Opening

**Input from sumcheck**: Random point $r \in \mathbb{F}^4$ and claimed value $v$

**Prover work**:
1. Compute $\vec{v} = M \cdot \vec{R}$ for all 55,590 matrices (field arithmetic)
2. Send $\vec{v}$ vectors to verifier

**Verifier work**:
1. Homomorphic check: $\sum_j R_j \cdot C_j \stackrel{?}{=} \langle \vec{v}, \vec{G} \rangle$
2. Final evaluation: $\vec{L}^T \cdot \vec{v} \stackrel{?}{=} v$

**Cost**: ~1M cycles (MSMs of size $\sqrt{16} = 4$)

### Total Stage 6 Cost

**Prover**: ~50M cycles
- Hyrax commitments: ~30M
- ExpSumcheck: ~15M
- Hyrax opening: ~5M

**Verifier**: ~321-361M cycles
- Component A: ~80-120M
- Component B: ~240M
- Component C: ~1M

**Proof size**: ~200-500 KB
- Commitments: ~7.1 MB (not sent, verifier recomputes via batching)
- Sumcheck round polynomials: ~2 KB
- Opening proof vectors: ~190-490 KB

---

## Stage 5 + Stage 6 Interaction

### Hint Flow

**Capture points** (Stage 5 proving):
1. **RLC combining**: 29 GT exponentiations $C_i^{\gamma_i}$
2. **Dory opening**: 80 GT exponentiations in main protocol

**Hint structure**:
```rust
pub struct DoryCombinedCommitmentHint {
    scaled_commitments: Vec<JoltGTBn254>,  // 29 precomputed GT results
    exponentiation_steps: Vec<ExponentiationSteps>,  // Full witness (109 total)
}
```

**Size**:
- `scaled_commitments`: $29 \times 12 \times 32 = 11$ KB
- `exponentiation_steps`: $109 \times 260$ KB $\approx 28$ MB (committed, not sent)

**Hint usage** (Stage 5 verification):
```rust
// Without hint
C_joint = ∏ᵢ Cᵢ^γᵢ  // 29 GT exponentiations @ ~10M cycles = 290M cycles

// With hint
C_joint = ∏ᵢ hint.scaled_commitments[i]  // 29 GT multiplications @ ~54K = 1.6M cycles
```

**Savings**: $290M - 1.6M \approx 288M$ cycles (180× speedup)

### Trust Model

**Stage 5**: Verifier **accepts** GT results as hints
- No verification that $C_i^{\gamma_i}$ is correct
- Transcript commits to all hints before Stage 6

**Stage 6**: Verifier **verifies** all accepted hints
- ExpSumcheck proves each GT exponentiation correct
- Hyrax proves committed witness data consistent
- Fiat-Shamir binds everything via transcript

**Soundness**: If hints are wrong, Stage 6 verification fails with probability $\geq 1 - \frac{\text{poly}(n)}{|\mathbb{F}|}$

### Performance Impact

**Without Stage 6**:
- Stage 5 verifier: ~1.12B cycles (109 GT exps @ ~10M each)
- Total: ~2.37B cycles

**With Stage 6**:
- Stage 5 verifier: ~42M cycles (using hints)
- Stage 6 verifier: ~321-361M cycles
- Total: ~363-403M cycles

**Net speedup**: $\frac{2.37B}{403M} \approx 5.9\times$ faster

**Proof size overhead**: +211-511 KB (11 KB hints + 200-500 KB Stage 6 proof)

---

## Complete Sumcheck Table

**All sumchecks across Jolt's 5-stage DAG** (Stage 6 separate):

| Stage | Component | Sumcheck Name | Purpose | Rounds | Polynomial Degree | Input Claim | Output Claims |
|-------|-----------|---------------|---------|--------|-------------------|-------------|---------------|
| **1** | R1CS (Spartan) | Outer sumcheck | Batch $N$ R1CS constraints | $\log N$ | 2 | $\sum_i \text{eq}(r_t, i) \cdot [(Az)_i \circ (Bz)_i - (Cz)_i] = 0$ | $\widetilde{Az}(r_t)$, $\widetilde{Bz}(r_t)$, $\widetilde{Cz}(r_t)$ at random $r_t$ |
| **1** | Instruction (Shout) | Write-checking | Access matrix well-formed | $\log T + \log N$ | 2 | $\sum_{t,i} \widetilde{M}(t,i) \cdot [\widetilde{M}(t,i) - 1] = 0$ | $\widetilde{M}(r_t, r_i)$ at random $(r_t, r_i)$ |
| **1** | Bytecode (Shout) | Write-checking | Bytecode access well-formed | $\log T + \log L$ | 2 | $\sum_{t,\ell} \widetilde{M}_{\text{bc}}(t,\ell) \cdot [\widetilde{M}_{\text{bc}}(t,\ell) - 1] = 0$ | $\widetilde{M}_{\text{bc}}(r_t, r_\ell)$ |
| **2** | R1CS (Spartan) | Product sumcheck (×3) | Reduce $Az$, $Bz$, $Cz$ to matrix claims | $\log m$ each | 2 | $\widetilde{Az}(r_t) = \sum_j \widetilde{A}(r_t, j) \cdot \widetilde{z}(j)$ | $\widetilde{A}(r_t, r_c)$, $\widetilde{z}(r_c)$ at random $r_c$ |
| **2** | RAM (Twist) | Read-checking (×d) | Verify RAM reads correct | $\log K$ each | 2 | Increment polynomial evaluation | Read address/value claims |
| **2** | RAM (Twist) | Write-checking (×d) | Verify RAM writes consistent | $\log K$ each | 2 | Memory state update | Write address/value claims |
| **2** | RAM (Twist) | Grand product | Time-ordered = address-ordered | $\log T$ | 2 | $\prod_{\text{time}} \phi_i = \prod_{\text{addr}} \phi_i$ | Fingerprint evaluations |
| **2** | Registers (Twist) | Read-checking (×2) | Verify rs1, rs2 reads | $\log K$ each | 2 | Register read increments | Read claims for rs1, rs2 |
| **2** | Registers (Twist) | Write-checking | Verify rd writes | $\log K$ | 2 | Register write increments | Write claim for rd |
| **2** | Registers (Twist) | Grand product | Time = address permutation | $\log T$ | 2 | $\prod_{\text{time}} \phi_i = \prod_{\text{addr}} \phi_i$ | Register fingerprints |
| **2** | Instruction (Shout) | Read-checking | Verify table lookups | $\log T$ | Variable | $\sum_t \text{eq}(r, t) \cdot [\widetilde{\text{val}}(t) - \widetilde{\text{table}}(\widetilde{M}(t, \cdot))]$ | Table access claims |
| **2** | Bytecode (Shout) | Read-checking | Verify bytecode reads | $\log T$ | Variable | Bytecode lookup verification | Bytecode value claims |
| **3** | R1CS (Spartan) | Matrix eval (×3) | SPARK sparse sumcheck for $A$, $B$, $C$ | Variable | 1 | $\widetilde{A}(r_t, r_c) = \sum_{(i,j) \in S} A_{ij} \text{eq}(r_t, i) \text{eq}(r_c, j)$ | Sparse matrix claims |
| **3** | RAM (Twist) | Evaluation sumcheck | Final memory = init + increments | $\log K$ | 1 | Memory state consistency | Final memory claims |
| **3** | RAM (Twist) | Hamming weight | Virtualized $ra$ one-hot check | Variable | 1 | $\sum_i \widetilde{ra}(i) = 1$ (if not virtualized) | (Skipped in Jolt - virtualized) |
| **3** | Registers (Twist) | Evaluation sumcheck | Final registers = init + increments | $\log K$ | 1 | Register state consistency | Final register claims |
| **4** | Cross-component | Consistency checks | Link components (RAM ↔ Reg, etc.) | Variable | Variable | Component output = component input | Cross-component evaluations |
| **5** | Dory | Batched opening sumcheck | Reduce $n$ openings → 1 | $\nu$ (MLE vars) | Variable | $\sum_i \gamma_i \widetilde{p}_i(r_i) = \sum_i \gamma_i v_i$ | Combined polynomial evaluation |

**Stage 6 sumchecks** (ExpSumcheck for GT exponentiations):

| Component | Sumcheck Name | Purpose | Rounds | Instances | Batching |
|-----------|---------------|---------|--------|-----------|----------|
| B | ExpSumcheck | Verify 254-step square-and-multiply constraints | 4 | 109 | BatchedSumcheck (4 rounds total, not 436) |
| C | Hyrax opening | Verify batched polynomial opening at challenge point | Variable | 1 (batched from 55,590) | Random linear combination |

**Total sumcheck count** (Stages 1-5):
- R1CS: 1 + 3 + 3 = 7 sumchecks
- RAM: d×2 + 1 + 1 + 1 ≈ 5-7 sumchecks (depends on $d$)
- Registers: 2 + 1 + 1 + 1 = 5 sumchecks
- Instructions: 1 + 1 = 2 sumchecks
- Bytecode: 1 + 1 = 2 sumchecks
- Dory: 1 sumcheck
- **Total: ~22-24 sumchecks** (depending on RAM parameter $d$)

**Total rounds**: ~$3\log T + 2\log K + \log N + \log m + \nu_{\text{Dory}} \approx 100-200$ rounds (batched)

**Batching reduces interaction**:
- Without batching: Each sumcheck separate → hundreds of rounds
- With batching: Parallel sumchecks in same stage share challenges → 5-6 batched interactions

**Stage 6 adds**: 4 rounds (ExpSumcheck batched) + Hyrax opening rounds

---

## Key Optimizations

### Batching

**Commitment batching**: Reduce $n$ commitments → 1 via RLC
$$C_{\text{batch}} = \sum_{i=1}^n \gamma_i C_i$$

**Opening batching**: Reduce $n$ openings → 1 joint opening
$$\widetilde{p}_{\text{joint}}(r) = \sum_{i=1}^n \gamma_i \widetilde{p}_i(r_i) \cdot \text{eq}(r, r_i)$$

**Sumcheck batching**: Multiple instances share rounds
- **Without**: $n$ instances × $\nu$ rounds = $n\nu$ rounds
- **With**: $\nu$ rounds (send combined polynomials)

### Virtual Polynomials

**Definition**: Polynomial never committed, evaluation proven via sumcheck

**Example**: Instruction output values expressed as:
$$\widetilde{\text{output}}(t) = \sum_i \widetilde{M}(t, i) \cdot \widetilde{\text{table}}(i)$$

**Savings**: No commitment cost, proven via lookup sumcheck

### Sparse Sumcheck

**For sparse $f$ with $s$ nonzeros over domain $2^n$**:
- Naïve prover: $O(2^n)$ evaluations
- Sparse prover: $O(s \cdot n)$ evaluations

**Jolt usage**: Access matrices have $T$ nonzeros, not $2^{\log T + \log N}$ entries

### Small-Value Optimization

**Observation**: Program values are 64-bit integers, not random field elements

**MSM speedup**: Small scalars → fewer point additions → 10-100× faster

**Preserved by**: Sum-check (not quotienting) maintains small intermediate values

### Pippenger MSM

**Multi-Scalar Multiplication**: Compute $\sum_{i=1}^n s_i \cdot G_i$

**Naïve**: $O(n \log |\mathbb{F}|)$ point additions

**Pippenger**: $O(n / \log n)$ point additions via bucketing

**Jolt usage**: Stage 6 Component A (222,360 Grumpkin scalars)

---

## Memory Layout

**Guest address space**:
```
0x00000000: Input (4 KB)
0x00001000: Output (4 KB)
0x00002000: Trusted advice (4 KB)
0x00003000: Untrusted advice (4 KB)
0x00004000: Stack (~2 GB, grows down)
0x80000000: Bytecode (up to 256 KB)
0x80010000: Heap (~256 MB, grows up)
```

**Address remapping**:
- **RAM**: $\text{index} = \frac{\text{addr} - 0x80000000}{8} + 1$ (doubleword-addressed)
- **Bytecode**: $\text{index} = \frac{\text{PC} - 0x80000000}{4}$ (instruction-addressed)

---

## Notation Reference

### Polynomials
- $\widetilde{f}$: Multilinear extension of function $f$
- $\hat{f}$: Univariate low-degree extension
- $\text{eq}(r, x)$: Equality polynomial
- $\chi_b(r)$: Lagrange basis ($\chi_0(r) = 1-r$, $\chi_1(r) = r$)

### Domains
- $\{0,1\}^n$: Boolean hypercube (n-dimensional)
- $\mathbb{F}$: Finite field (typically order $\approx 2^{256}$)
- $H \subseteq \mathbb{F}$: Evaluation domain (often roots of unity)

### Operators
- $\circ$: Hadamard (entrywise) product
- $\otimes$: Tensor (Kronecker) product
- $\langle \cdot, \cdot \rangle$: Inner product

### Complexity
- $N$: Number of gates/constraints/trace length
- $T$: Trace length (execution cycles)
- $K$: Memory size (addresses)
- $d$: Degree bound or chunking parameter
- $\nu$: Number of variables

### Cryptographic
- $\gamma, \beta, \alpha$: Random challenges from verifier/transcript
- $C$: Commitment (group element)
- $\text{PCS}$: Polynomial commitment scheme
- $\mathbb{G}_1, \mathbb{G}_2, \mathbb{G}_T$: Pairing groups

---

## Complexity Summary

**Jolt prover** (for $T$ cycles):
- Field operations: $O(T \log T)$
- Group operations: $O(T)$ (Dory commitments)
- Memory: $O(T + K)$ where $K$ is RAM size

**Jolt verifier** (without Stage 6):
- Field operations: $O(\log T)$
- Group operations: $O(\log T)$
- GT exponentiations: 109 @ ~10M cycles = ~1.1B cycles

**Jolt verifier** (with Stage 6):
- Stage 5: ~42M cycles (using hints)
- Stage 6: ~321-361M cycles
- Total: ~363-403M cycles (~5.9× speedup)

**Proof size**:
- Stage 1-4: ~50-100 KB (sumcheck round polynomials)
- Stage 5: ~13 KB (Dory opening proof) + 11 KB (GT hints)
- Stage 6: ~200-500 KB (Hyrax commitments + ExpSumcheck + opening)
- **Total**: ~274-624 KB

---

## Implementation Notes

### File Structure

**Core proving**:
- `jolt-core/src/zkvm/dag/jolt_dag.rs`: Main proof orchestration
- `jolt-core/src/poly/opening_proof.rs`: Batched opening (Stage 5)
- `jolt-core/src/subprotocols/`: Sumcheck, Spartan, Twist, Shout
- `jolt-core/src/zkvm/instruction/`: Per-instruction lookup tables

**Stage 6**:
- `jolt-core/src/subprotocols/snark_composition.rs`: SZ-Check protocol
- `jolt-core/src/poly/commitment/recursion.rs`: Hint mechanism
- External: `dory` crate for GT arithmetic and Hyrax

**Tracer**:
- `tracer/src/emulator/`: RISC-V emulator
- `tracer/src/instruction/`: Instruction implementations

### Build Features

- `recursion`: Enable Stage 6 SNARK composition
- `monitor`: System resource profiling
- `pprof`: CPU profiling via pprof
- `allocative`: Memory profiling with flamegraphs

### Profiling

**Execution traces** (Perfetto format):
```bash
cargo run --release -p jolt-core profile --name sha3 --format chrome
# View at https://ui.perfetto.dev/
```

**CPU profiling** (pprof):
```bash
cargo run --release --features pprof -p jolt-core profile --name sha3 --format chrome
go tool pprof -http=:8080 benchmark-runs/pprof/sha3_prove.pb
```

**Memory profiling** (allocative):
```bash
RUST_LOG=debug cargo run --release --features allocative -p jolt-core profile --name sha3
# Generates stage*_flamechart.svg files
```

---

## Security Considerations

### Soundness

**Soundness error**: Probability cheating prover convinces honest verifier
$$\epsilon_{\text{soundness}} \leq \sum_{\text{sumchecks}} \frac{d_i \nu_i}{|\mathbb{F}|} + \text{PCS soundness}$$

For Jolt: $\epsilon < 2^{-128}$ (negligible)

### Completeness

**Completeness**: Honest prover can always convince honest verifier

Jolt achieves perfect completeness (probability 1)

### Zero-Knowledge

**Jolt v0.2.0**: Not zero-knowledge (public coin protocol)

**Future**: Can add ZK via polynomial blinding + proof encryption

### Trusted Setup

**Dory**: Transparent (SRS generated deterministically)

**No trusted setup required** - key advantage over KZG/Groth16

---

## References

**Papers**:
- LFKN'90: Sum-check protocol
- Spartan (Setty 2020): Transparent SNARKs
- Lasso (Setty, Thaler, Wahby 2024): Lookup arguments
- Jolt (Arun, Setty, Thaler 2024): zkVM architecture
- Twist (Setty, Thaler 2025): Memory checking
- Dory (Lee et al.): Transparent PCS

**Blog posts**:
- [Releasing Jolt](https://a16zcrypto.com/posts/article/a-new-era-in-snark-design-releasing-jolt)
- [Twist and Shout upgrade](https://a16zcrypto.com/posts/article/aug-18-2025-twist-shout)

**Documentation**:
- [01_Jolt_Theory_Enhanced.md](01_Jolt_Theory_Enhanced.md): Complete mathematical foundations
- [02_Jolt_Complete_Guide.md](02_Jolt_Complete_Guide.md): Implementation guide
- [03_Verifier_Mathematics_and_Code.md](03_Verifier_Mathematics_and_Code.md): Verifier specification
- [Stage5_and_Stage6_Hint_Mechanism.md](Stage5_and_Stage6_Hint_Mechanism.md): Hint system details
