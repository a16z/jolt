# Jolt Recursion via SNARK Composition

## 1. Motivation & Overview

### 1.1 Why Recursion

Recursion enables two critical capabilities:

1. **Proof Aggregation**: Maintain a single succinct proof verifying entire blockchain state, enabling light clients and efficient synchronization.

2. **On-Chain Verification**: Produce proofs with single-digit millisecond verification on mobile and reasonable EVM gas costs.

### 1.2 The Approach

The Jolt verifier compiled to RISC-V requires ~1.5 billion cycles. Our target is <10 million cycles (~150× reduction).

We decompose the verifier:

$$\mathcal{V}_{\text{Jolt}} = \mathcal{V}_{\text{light}} \circ \mathcal{H}$$

where:
- $\mathcal{H} = \{h_1, \ldots, h_m\}$ are **hints** for expensive operations
- $\mathcal{V}_{\text{light}}$ assumes hints are correct
- A **bespoke SNARK** proves hints are well-formed

The expensive operations (from Dory PCS verification):

| Operation | Description |
|-----------|-------------|
| G1 Scalar Mul | $[k]P$ for $P \in \mathbb{G}_1$ |
| G2 Scalar Mul | $[k]Q$ for $Q \in \mathbb{G}_2$ |
| GT Exponentiation | $a^k$ for $a \in \mathbb{G}_T$ |
| GT Multiplication | $a \cdot b$ for $a, b \in \mathbb{G}_T$ |
| Multi-Pairing | $\prod_i e(P_i, Q_i)$ |

### 1.3 Protocol Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 1: Constraint Sum-Checks                                     │
│  ─────────────────────────────                                      │
│  GT Exp, GT Mul, G1 Scalar Mul → virtual polynomial claims at r_x   │
└──────────────────────────────────┬──────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 2: Virtualization Sum-Check                                  │
│  ─────────────────────────────────                                  │
│  Combine all claims into matrix M(s,x) → claim M(r_s, r_x)          │
└──────────────────────────────────┬──────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 3: Jagged Transform Sum-Check                                │
│  ────────────────────────────────────                               │
│  Sparse M → Dense q → claim q(r_dense)                              │
└──────────────────────────────────┬──────────────────────────────────┘
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Stage 4: Opening Proof (Hyrax over Grumpkin)                       │
│  ────────────────────────────────────────────                       │
│  Prove q(r_dense) = v_dense → final verification                    │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.4 Field Choice

All SNARK arithmetic is over $\mathbb{F}_q$ (BN254 base field), which equals the Grumpkin scalar field. This choice is dictated by:
- GT witnesses produce $\mathbb{F}_q$ elements (from Fq12 representation)
- Hyrax PCS operates over Grumpkin, whose scalar field is $\mathbb{F}_q$

---

## 2. Stage 1: Constraint Sum-Checks

Stage 1 proves correctness of each expensive operation via constraint-specific sum-checks. Each produces virtual polynomial claims that feed into Stage 2.

### 2.1 Ring Switching & Quotient Technique

$\mathbb{G}_T$ elements are represented as $\mathbb{F}_{q^{12}} = \mathbb{F}_q[X]/(p(X))$ where $p(X)$ is an irreducible polynomial of degree 12.

**Key insight**: For $a, b, c \in \mathbb{G}_T$, the equation $a \cdot b = c$ holds iff there exists quotient $Q$ such that:
$$a(X) \cdot b(X) = c(X) + Q(X) \cdot p(X)$$

On the Boolean hypercube $\{0,1\}^4$ (viewing elements as 4-variate polynomials):
$$a(x) \cdot b(x) - c(x) - Q(x) \cdot g(x) = 0$$

where $g(x)$ is the MLE of $p$ on the hypercube.

This transforms high-degree $\mathbb{F}_{q^{12}}$ operations into low-degree constraints over $\mathbb{F}_q$ by introducing the quotient as auxiliary witness.

### 2.2 GT Exponentiation

Computes $b = a^k$ using square-and-multiply.

#### Inputs / Outputs

| Role | Symbol | Description |
|------|--------|-------------|
| Public Input | $a \in \mathbb{G}_T$ | Base (as Fq12) |
| Public Input | $k \in \mathbb{F}_r$ | Exponent |
| Public Output | $b \in \mathbb{G}_T$ | Result $b = a^k$ |

#### Witness

| Symbol | Description |
|--------|-------------|
| $(b_0, \ldots, b_{t-1})$ | Binary representation of $k$ |
| $\rho_0, \ldots, \rho_t$ | Intermediate values: $\rho_0 = 1$, $\rho_t = a^k$ |
| $Q_0, \ldots, Q_{t-1}$ | Quotient polynomials |

Recurrence:
$$\rho_{i+1} = \begin{cases} \rho_i^2 & \text{if } b_i = 0 \\ \rho_i^2 \cdot a & \text{if } b_i = 1 \end{cases}$$

#### Constraint

For each bit $i$:
$$C_i(x) = \rho_{i+1}(x) - \rho_i(x)^2 \cdot a(x)^{b_i} - Q_i(x) \cdot g(x) = 0$$

where $a(x)^{b_i} = 1 + (a(x) - 1) \cdot b_i$ (linearization).

#### Sum-Check

$$0 = \sum_{x \in \{0,1\}^4} \text{eq}(r_x, x) \cdot \sum_{i=0}^{t-1} \gamma^i \cdot C_i(x)$$

- 4 rounds (one per variable)
- Batching coefficient $\gamma$ combines all $t$ constraints

#### Output Claims

After final challenge $r_x'$:
- `RecursionBase(i)`: $\tilde{a}(r_x')$
- `RecursionRhoPrev(i)`: $\tilde{\rho}_i(r_x')$
- `RecursionRhoCurr(i)`: $\tilde{\rho}_{i+1}(r_x')$
- `RecursionQuotient(i)`: $\tilde{Q}_i(r_x')$

#### Packed Witness Structure

Instead of creating separate polynomials for each step, we pack all 254 steps into unified 12-variable MLEs:

| Symbol | Description | Layout |
|--------|-------------|--------|
| $\rho(s, x)$ | Intermediate values | $\rho[x \cdot 256 + s] = \rho_s[x]$ |
| $\rho_{\text{next}}(s, x)$ | Shifted intermediate values | $\rho_{\text{next}}[x \cdot 256 + s] = \rho_{s+1}[x]$ |
| $Q(s, x)$ | Quotient polynomials | $Q[x \cdot 256 + s] = Q_s[x]$ |
| $\text{bit}(s)$ | Binary representation of $k$ | Replicated across $x$ |
| $\text{base}(x)$ | Base element | Replicated across $s$ |

Where:
- $s \in \{0,1\}^8$ indexes the step (0 to 255)
- $x \in \{0,1\}^4$ indexes the field element (0 to 15)
- Layout formula: `index = x * 256 + s` (s in low bits)

#### Unified Constraint

$$C(s, x) = \rho_{\text{next}}(s, x) - \rho(s, x)^2 \cdot \text{base}(x)^{\text{bit}(s)} - Q(s, x) \cdot g(x) = 0$$

#### Two-Phase Sum-Check

$$0 = \sum_{s \in \{0,1\}^8} \sum_{x \in \{0,1\}^4} \text{eq}(r_s, s) \cdot \text{eq}(r_x, x) \cdot C(s, x)$$

- **Phase 1** (rounds 0-7): Bind step variables $s$
- **Phase 2** (rounds 8-11): Bind element variables $x$
- Total: 12 rounds, degree 4

#### Output Claims

After final challenges $(r_s^*, r_x^*)$:
- `PackedGtExpBase(i)`: $\text{base}(r_x^*)$
- `PackedGtExpRho(i)`: $\rho(r_s^*, r_x^*)$
- `PackedGtExpRhoNext(i)`: $\rho_{\text{next}}(r_s^*, r_x^*)$
- `PackedGtExpQuotient(i)`: $Q(r_s^*, r_x^*)$
- `PackedGtExpBit(i)`: $\text{bit}(r_s^*)$

#### Mathematical Correctness

**Theorem**: The packed representation maintains constraint satisfaction equivalence.

**Proof**: For each step $i \in [0, 254]$ and element $x \in [0, 15]$:
- Original: $C_i(x) = \rho_{i+1}(x) - \rho_i(x)^2 \cdot a(x)^{b_i} - Q_i(x) \cdot g(x) = 0$
- Packed: $C(i, x) = \rho_{\text{next}}(i, x) - \rho(i, x)^2 \cdot \text{base}(x)^{\text{bit}(i)} - Q(i, x) \cdot g(x) = 0$

The packed constraint at $(i, x)$ equals the original constraint $C_i(x)$ by construction:
- $\rho(i, x) = \rho_i(x)$ (by definition of packing)
- $\rho_{\text{next}}(i, x) = \rho_{i+1}(x)$ (shifted index)
- $\text{base}(x) = a(x)$ (replicated across steps)
- $\text{bit}(i) = b_i$ (scalar bit at step $i$)

Therefore, constraint satisfaction is preserved. □

**Security Analysis**:
- Soundness error: Unchanged at $\text{deg}/|\mathbb{F}| = 4/p \approx 2^{-252}$
- The two-phase sumcheck maintains the same security as 254 individual sumchecks
- Batching with $\gamma$ preserves zero-knowledge properties

#### Performance Benefits

| Metric | Before | After | Improvement |
|--------|--------|-------|-------------|
| Polynomials per GT exp | 1,024 | 5 | 204.8× |
| Virtual claims | 1,024 | 5 | 204.8× |
| Proof size contribution | ~32KB | ~160B | 200× |

---

### 2.3 GT Multiplication

Proves $c = a \cdot b$ for $a, b, c \in \mathbb{G}_T$.

#### Inputs / Outputs

| Role | Symbol | Description |
|------|--------|-------------|
| Public Input | $a, b \in \mathbb{G}_T$ | Operands |
| Public Output | $c \in \mathbb{G}_T$ | Result $c = a \cdot b$ |

#### Witness

| Symbol | Description |
|--------|-------------|
| $Q$ | Quotient polynomial: $Q(x) = \frac{a(x) \cdot b(x) - c(x)}{g(x)}$ |

#### Constraint

$$C(x) = a(x) \cdot b(x) - c(x) - Q(x) \cdot g(x) = 0$$

#### Sum-Check

For $m$ multiplication constraints:
$$0 = \sum_{x \in \{0,1\}^4} \text{eq}(r_x, x) \cdot \sum_{i=0}^{m-1} \gamma^i \cdot C_i(x)$$

#### Output Claims

- `RecursionMulLhs(i)`: $\tilde{a}_i(r_x')$
- `RecursionMulRhs(i)`: $\tilde{b}_i(r_x')$
- `RecursionMulResult(i)`: $\tilde{c}_i(r_x')$
- `RecursionMulQuotient(i)`: $\tilde{Q}_i(r_x')$

---

### 2.4 G1 Scalar Multiplication

Proves $Q = [k]P$ using double-and-add.

#### Inputs / Outputs

| Role | Symbol | Description |
|------|--------|-------------|
| Public Input | $P \in \mathbb{G}_1$ | Base point $(x_P, y_P)$ |
| Public Input | $k \in \mathbb{F}_r$ | Scalar with bits $(b_0, \ldots, b_{n-1})$ |
| Public Output | $Q \in \mathbb{G}_1$ | Result $Q = [k]P$ |

#### Witness

Execution trace with $n$ rows (padded to $N = 2^\ell$):

| Column | Description |
|--------|-------------|
| $(x_A, y_A)$ | Accumulator at start of iteration |
| $(x_T, y_T)$ | Doubled point: $T = [2]A$ |
| $(x_{A'}, y_{A'})$ | Accumulator after conditional addition |
| $\text{ind}$ | 1 if $T = \mathcal{O}$ (infinity), 0 otherwise |

Each row $i$: $A_{i+1} = T_i + b_i \cdot P$

#### Constraints

**C1 (Doubling x)**: $4y_A^2(x_T + 2x_A) - 9x_A^4 = 0$

**C2 (Doubling y)**: $3x_A^2(x_T - x_A) + 2y_A(y_T + y_A) = 0$

**C3 (Addition x)**: Handles both finite and infinity cases via indicator:
$$\text{ind} \cdot x_{A'} \cdot (x_{A'} - x_P) + (1 - \text{ind}) \cdot [\text{addition formula}] = 0$$

**C4 (Addition y)**: Similarly for y-coordinate.

| Constraint | Degree |
|------------|--------|
| C1 | 4 |
| C2 | 3 |
| C3 | 6 |
| C4 | 6 |

#### Sum-Check

$$0 = \sum_{x \in \{0,1\}^\ell} \text{eq}(r, x) \cdot \sum_{j=1}^{4} \delta^{j-1} \cdot C_j(x)$$

- $\ell = \lceil \log_2 n \rceil$ rounds
- Degree 6 (maximum constraint degree)

#### Output Claims

- `RecursionG1ScalarMulXA(i)`, `RecursionG1ScalarMulYA(i)`
- `RecursionG1ScalarMulXT(i)`, `RecursionG1ScalarMulYT(i)`
- `RecursionG1ScalarMulXANext(i)`, `RecursionG1ScalarMulYANext(i)`
- `RecursionG1ScalarMulIndicator(i)`

---

### 2.5 [Future] G2 Scalar Multiplication

Similar to G1, but over the twist curve $E'(\mathbb{F}_{q^2})$.

*Pending implementation.*

---

### 2.6 [Future] Multi-Pairing

Computes $T = \prod_{i=1}^m e(P_i, Q_i)$ via:
1. Miller loop accumulation
2. Final exponentiation
3. Line evaluation

*Pending implementation.*

---

## 3. Stage 2: Direct Evaluation Protocol

After Stage 1, we have many virtual polynomial claims $(v_0, v_1, \ldots, v_{n-1})$ at point $r_x$. Stage 2 verifies these claims directly without sumcheck.

### 3.1 Why Direct Evaluation

The constraint matrix $M$ has a special structure:
$$M(i, r_x) = v_i \text{ for all } i$$

Therefore:
$$M(r_s, r_x) = \sum_{i \in \{0,1\}^{\log n}} \text{eq}(r_s, i) \cdot M(i, r_x) = \sum_{i} \text{eq}(r_s, i) \cdot v_i$$

This allows direct computation without sumcheck rounds.

### 3.2 Matrix Organization

Define matrix $M : \{0,1\}^s \times \{0,1\}^x \to \mathbb{F}_q$ where:
- Row index $s$ selects polynomial type and constraint index
- Column index $x$ selects evaluation point (4 or 8 bits)

Row indexing:
```
row = poly_type × num_constraints_padded + constraint_idx
```

Polynomial types:

| Type | Index | Source |
|------|-------|--------|
| RecursionBase | 0 | GT Exp (unpacked) |
| RecursionRhoPrev | 1 | GT Exp (unpacked) |
| RecursionRhoCurr | 2 | GT Exp (unpacked) |
| RecursionQuotient | 3 | GT Exp (unpacked) |
| RecursionMulLhs | 4 | GT Mul |
| RecursionMulRhs | 5 | GT Mul |
| RecursionMulResult | 6 | GT Mul |
| RecursionMulQuotient | 7 | GT Mul |
| RecursionG1ScalarMulXA | 8 | G1 Scalar Mul |
| RecursionG1ScalarMulYA | 9 | G1 Scalar Mul |
| RecursionG1ScalarMulXT | 10 | G1 Scalar Mul |
| RecursionG1ScalarMulYT | 11 | G1 Scalar Mul |
| RecursionG1ScalarMulXANext | 12 | G1 Scalar Mul |
| RecursionG1ScalarMulYANext | 13 | G1 Scalar Mul |
| RecursionG1ScalarMulIndicator | 14 | G1 Scalar Mul |
| PackedGtExpBase | 15 | GT Exp (packed) |
| PackedGtExpRho | 16 | GT Exp (packed) |
| PackedGtExpRhoNext | 17 | GT Exp (packed) |
| PackedGtExpQuotient | 18 | GT Exp (packed) |

**Total**: 19 polynomial types (15 original + 4 packed)

### 3.3 Direct Evaluation Protocol

**Input**: Virtual claims $\{v_i\}$ from Stage 1 at point $r_x$

**Protocol**:
1. **Sample**: $r_s \leftarrow \mathbb{F}^{\log n}$ from transcript
2. **Prover**:
   - Evaluate $M(r_s, r_x)$ by binding matrix to challenges
   - Send evaluation to verifier
3. **Verifier**:
   - Compute $\text{eq}_{\text{evals}} = \text{EqPolynomial::evals}(r_s)$
   - Compute $v = \sum_i \text{eq}_{\text{evals}}[s_i] \cdot v_i$ where $s_i = \text{matrix\_s\_index}(i)$
   - Verify prover's evaluation matches $v$

**Output**: Opening claim $M(r_s, r_x) = v_{\text{sparse}}$

#### Mathematical Correctness

**Theorem**: Direct evaluation is sound with the same security as sumcheck.

**Proof**: The multilinear extension $\tilde{M}$ is unique. For the boolean hypercube:
$$\tilde{M}(s, x) = M(s, x) \text{ for all } s \in \{0,1\}^{\log n}, x$$

By linearity of multilinear extensions:
$$\tilde{M}(r_s, r_x) = \sum_{i \in \{0,1\}^{\log n}} M(i, r_x) \cdot \text{eq}(r_s, i)$$

Since $M(i, r_x) = v_i$ by construction:
$$\tilde{M}(r_s, r_x) = \sum_i v_i \cdot \text{eq}(r_s, i)$$

The verifier computes exactly this sum, so the protocol is perfectly sound. □

**Security Guarantees**:
- No additional soundness error (deterministic protocol)
- Fiat-Shamir security maintained through transcript inclusion
- Binding: Prover committed to $v_i$ values in Stage 1

**Benefits**:
- Eliminates $\log n$ sumcheck rounds
- Reduces proof size by $3\log n$ field elements
- Verifier work remains $O(n)$

---

## 4. Stage 3: Jagged Transform Sum-Check

The matrix $M$ is sparse: 4-variable polynomials (GT) are zero-padded to 8 variables. Stage 3 compresses to a dense representation.

### 4.1 Why Jaggedness

| Operation | Native Variables | Native Size | Padded Size |
|-----------|-----------------|-------------|-------------|
| GT Exp/Mul | 4 | 16 | 256 (zero-padded) |
| G1 Scalar Mul | 8 | 256 | 256 (native) |

The jagged transform eliminates redundant zeros, reducing commitment size.

### 4.2 Sparse-to-Dense Bijection

**Sparse**: $M(s, x)$ with row $s$ and column $x \in \{0,1\}^8$

**Dense**: $q(i)$ containing only non-redundant entries

Bijection:
- `row(i)` = polynomial index for dense index $i$
- `col(i)` = evaluation index within that polynomial
- Cumulative sizes track where each polynomial's entries begin

### 4.3 Differences from Jagged PCS Paper

| Paper | Our Implementation |
|-------|-------------------|
| Column-wise jaggedness | Row-wise jaggedness |
| Direct bijection to matrix | Three-level mapping |
| Simple row/col | poly_idx → (constraint_idx, poly_type) → matrix_row |

The three-level mapping handles multiple polynomial types per constraint.

### 4.4 The Jagged Indicator Function

$$\hat{f}_{\text{jagged}}(r_s, r_x, r_{\text{dense}}) = \sum_{y \in \text{rows}} \text{eq}(r_s, y) \cdot \hat{g}(r_x, r_{\text{dense}}, t_{y-1}, t_y)$$

where $g(a, b, c, d) = 1$ iff $b < d$ AND $b = a + c$.

### 4.5 Branching Program Optimization

The function $\hat{g}$ can be computed in $O(n)$ time via a width-4 read-once branching program:
- Tracks carry bit (addition check: $b = a + c$)
- Tracks comparison bit ($b < d$)
- Processes bits LSB to MSB

This avoids naive $O(2^{4n})$ computation.

### 4.6 Sum-Check Protocol

**Input**: $M(r_{s,\text{final}}, r_x) = v_{\text{sparse}}$ from Stage 2

**Relation**:
$$v_{\text{sparse}} = \sum_{i \in \{0,1\}^{\ell_{\text{dense}}}} q(i) \cdot \hat{f}_{\text{jagged}}(r_s, r_x, i)$$

**Protocol**:
1. Prover materializes dense $q(i)$ and indicator $\hat{f}_{\text{jagged}}$
2. Sum-check over $\ell_{\text{dense}}$ rounds (degree 2)
3. At challenge $r_{\text{dense}}$: claim $q(r_{\text{dense}}) = v_{\text{dense}}$
4. Verifier computes $\hat{f}_{\text{jagged}}(r_s, r_x, r_{\text{dense}})$ via branching program
5. Verify: output = $v_{\text{dense}} \cdot \hat{f}_{\text{jagged}}$

**Output**: $(q, r_{\text{dense}}, v_{\text{dense}})$

### 4.7 Stage 3b: Jagged Assist

After Stage 3's sumcheck, the verifier must compute:
$$\hat{f}_{\text{jagged}}(r_s, r_x, r_{\text{dense}}) = \sum_{y \in [K]} \text{eq}(r_s, y) \cdot \hat{g}(r_x, r_{\text{dense}}, t_{y-1}, t_y)$$

where $K$ is the number of polynomials. For large systems, this requires $K \times O(\text{bits})$ field operations.

#### The Jagged Assist Protocol

Instead of computing all $K$ evaluations, we use batch verification:

1. **Prover sends**: $v_y = \hat{g}(r_x, r_{\text{dense}}, t_{y-1}, t_y)$ for all $y \in [K]$

2. **Verifier samples**: Batching coefficient $r \leftarrow \mathbb{F}$

3. **Batch claim**:
   $$\sum_{y=0}^{K-1} r^y \cdot \hat{g}(x_y) = \sum_{y=0}^{K-1} r^y \cdot v_y$$

4. **Sumcheck**: Over $P(b) = g(b) \cdot \sum_y r^y \cdot \text{eq}(b, x_y)$

5. **Final verification**: At random point $\rho$:
   - Compute $\hat{g}(\rho)$ using branching program
   - Compute $\sum_y r^y \cdot \text{eq}(\rho, x_y)$
   - Verify $P(\rho) = \hat{g}(\rho) \cdot \text{eq\_sum}$

#### Mathematical Correctness

**Theorem (Jagged Assist Soundness)**: If $\exists y^* : v_{y^*} \neq \hat{g}(x_{y^*})$, then the verifier accepts with probability at most $\frac{2m}{|\mathbb{F}|}$.

**Proof Sketch**:
1. Define polynomial $R(Z) = \sum_{y=0}^{K-1} Z^y \cdot (\hat{g}(x_y) - v_y)$
2. If any $v_{y^*}$ is incorrect, then $R(Z) \not\equiv 0$ and $\deg(R) \leq K-1$
3. By Schwartz-Zippel, $\Pr[R(r) = 0] \leq \frac{K-1}{|\mathbb{F}|}$
4. If $R(r) \neq 0$, the sumcheck polynomial $P$ differs from the honest polynomial
5. Sumcheck soundness: $\Pr[\text{accept} | R(r) \neq 0] \leq \frac{2m}{|\mathbb{F}|}$
6. Union bound: $\Pr[\text{accept}] \leq \frac{K-1 + 2m}{|\mathbb{F}|} \approx \frac{2m}{|\mathbb{F}|}$

**Key Insight**: The protocol converts $K$ individual checks into one batched sumcheck with negligible soundness loss.

#### Benefits

| Metric | Without Assist | With Assist | Improvement |
|--------|----------------|-------------|-------------|
| Verifier ops | $K \times 1,300$ | $330K$ | ~64× |
| Extra rounds | 0 | $2m$ | Acceptable |
| Soundness error | 0 | $\frac{2m}{|\mathbb{F}|}$ | Negligible |

The protocol leverages Lemma 4.6 (forward-backward decomposition) for efficient prover computation.

---

## 5. Stage 4: Opening Proof (Hyrax over Grumpkin)

Stage 4 proves the dense polynomial claim using Hyrax.

### 5.1 Why Hyrax

- **No pairing required**: Works over any curve with efficient MSM
- **Grumpkin-native**: Scalar field = $\mathbb{F}_q$, matching our constraint field
- **Square matrix efficiency**: Optimal for virtualized polynomials

### 5.2 Commitment Structure

For polynomial $q$ with $2^n$ evaluations:
1. Reshape as $2^{n_r} \times 2^{n_c}$ matrix
2. Commit to each row via Pedersen over Grumpkin
3. Commitment: $C = \{C_0, \ldots, C_{2^{n_r}-1}\}$

### 5.3 Opening Protocol

**Input**: Commitment $C$, point $r_{\text{dense}}$, claimed value $v_{\text{dense}}$

**Protocol**:
1. Decompose point into row/column components
2. Sum-check proves tensor product structure
3. Column opening proves consistency
4. Verifier checks against commitment

### 5.4 Final Verification

The verifier accepts iff all checks pass:
- Stage 1-3 sum-check verifications
- Hyrax opening verification

---

## 6. Parameters & Cost Analysis

This section provides analytical formulas for proof sizes, constraint counts, and computational costs.

### 6.1 Sumcheck Degrees and Rounds

| Stage | Protocol | Degree | Rounds | Elements/Round |
|-------|----------|--------|--------|----------------|
| 1 | GT Exponentiation (unpacked) | 4 | 4 | 5 |
| 1 | GT Exponentiation (packed) | 4 | 12 | 5 |
| 1 | GT Multiplication | 3 | 4 | 4 |
| 1 | G1 Scalar Multiplication | 6 | $\ell$ | 7 |
| 2 | Direct Evaluation | - | 0 | 1 |
| 3 | Jagged Transform | 2 | $d$ | 3 |
| 3b | Jagged Assist | 2 | $2m$ | 3 |

Where:
- $\ell = \lceil \log_2 n \rceil$ for $n$-bit scalar
- $d = \lceil \log_2(\text{dense\_size}) \rceil$
- $m = $ number of jagged indicator evaluations (typically $\log K$ where $K$ is polynomial count)

### 6.2 Constraint Counts

**Per-operation constraints**:

| Operation | Constraints | Poly Types | Total Polynomials |
|-----------|-------------|------------|-------------------|
| GT Exp (unpacked, $t$-bit) | $t$ | 4 | $4t$ |
| GT Exp (packed, $t$-bit) | 1 | 5 | 5 |
| GT Mul | 1 | 4 | 4 |
| G1 Scalar Mul ($n$-bit) | 1 | 7 | 7 |

**Typical Dory verification** (256-bit scalars):
- GT exponentiations (unpacked): 256 constraints each
- GT exponentiations (packed): 1 unified constraint covering all 256 steps
- GT multiplications: 1 constraint each
- G1 scalar multiplications: 1 constraint each (but 256 rows in trace)

### 6.3 Matrix Dimensions

**Row count**:
$$\text{num\_rows} = 19 \times c_{\text{pad}}$$

where $c_{\text{pad}} = 2^{\lceil \log_2 c \rceil}$ and $c$ = total constraints.

**Note**: The count 19 includes both unpacked (15 types) and packed GT exp types (4 additional).

**Column count**:
- $2^4 = 16$ for GT operations (4-variable MLEs)
- $2^8 = 256$ for G1 operations (8-variable MLEs)
- Sparse matrix uses $2^8 = 256$ (zero-padded)

**Example** ($c = 10$ constraints):

| Parameter | Formula | Value |
|-----------|---------|-------|
| $c_{\text{pad}}$ | $2^{\lceil \log_2 10 \rceil}$ | 16 |
| num\_rows | $19 \times 16$ | 304 |
| num\_s\_vars | $\lceil \log_2 304 \rceil$ | 9 |

### 6.4 Dense Size Computation

The jagged transform compresses the sparse matrix by removing zero-padding:

$$\text{dense\_size} = \sum_{\text{poly } p} 2^{\text{num\_vars}(p)}$$

where:
- GT polynomials: $\text{num\_vars} = 4 \Rightarrow 16$ evaluations
- G1 polynomials: $\text{num\_vars} = 8 \Rightarrow 256$ evaluations

**Example** (3 GT exp + 2 GT mul + 1 G1 scalar mul):

| Source | Polys | Evals/Poly | Total |
|--------|-------|------------|-------|
| GT Exp (3 constraints × 4 types) | 12 | 16 | 192 |
| GT Mul (2 constraints × 4 types) | 8 | 16 | 128 |
| G1 Scalar Mul (1 constraint × 7 types) | 7 | 256 | 1,792 |
| **Total** | 27 | — | **2,112** |

$\Rightarrow d = \lceil \log_2 2112 \rceil = 12$ rounds for Stage 3.

### 6.5 Proof Size Formulas

**Stage 1** (sumcheck messages):
$$|P_1| = \sum_{\text{type } t} (\text{degree}_t + 1) \times \text{rounds}_t$$

| Type | Degree | Rounds | Elements |
|------|--------|--------|----------|
| GT Exp (unpacked) | 4 | 4 | 20 |
| GT Exp (packed) | 4 | 12 | 60 |
| GT Mul | 3 | 4 | 16 |
| G1 Scalar Mul | 6 | $\ell$ | $7\ell$ |

**Stage 2** (direct evaluation):
$$|P_2| = 1 \text{ element}$$

**Stage 3** (jagged transform):
$$|P_3| = 3d \text{ elements}$$

**Stage 3b** (jagged assist):
$$|P_{3b}| = K + 3(2m) \text{ elements}$$
where $K$ is the number of $\hat{g}$ evaluations sent

**Stage 4** (Hyrax opening):
$$|P_4| = O(\sqrt{\text{dense\_size}}) \text{ group elements}$$

**Total proof size** (field elements, excluding PCS):
$$|P| = |P_1| + |P_2| + |P_3| + |P_{3b}| + \text{virtual claims}$$

### 6.6 Concrete Example: Single 256-bit GT Exponentiation

**Unpacked version**:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Constraints ($c$) | 256 | One per bit |
| $c_{\text{pad}}$ | 256 | Already power of 2 |
| Polynomials | 1,024 | $256 \times 4$ types |
| num\_s\_vars ($s$) | 12 | $\lceil \log_2(19 \times 256) \rceil$ |
| dense\_size | 16,384 | $1024 \times 16$ |
| num\_dense\_vars ($d$) | 14 | $\lceil \log_2 16384 \rceil$ |
| Virtual claims | 1,024 | $4 \times 256$ |

**Packed version**:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Constraints ($c$) | 1 | Single unified constraint |
| $c_{\text{pad}}$ | 1 | Already power of 2 |
| Polynomials | 5 | PackedGtExp* types |
| num\_s\_vars ($s$) | 5 | $\lceil \log_2(19 \times 1) \rceil$ |
| dense\_size | 65,536 | $4 \times 16^4$ (12-var MLEs) |
| num\_dense\_vars ($d$) | 16 | $\lceil \log_2 65536 \rceil$ |
| Virtual claims | 5 | One per packed type |

**Proof size comparison**:

| Stage | Unpacked | Packed | Improvement |
|-------|----------|--------|-------------|
| Stage 1 | 20 | 60 | 3× larger |
| Stage 2 | 1 | 1 | Same |
| Stage 3 | $3 \times 14 = 42$ | $3 \times 16 = 48$ | Slightly larger |
| Virtual claims | 1,024 | 5 | **204.8× smaller** |
| **Total** | 1,087 | 114 | **9.5× smaller** |

At 32 bytes per $\mathbb{F}_q$ element:
- Unpacked: $1,087 \times 32 = 34.8$ KB
- Packed: $114 \times 32 = 3.6$ KB
(excluding PCS proof)

### 6.7 Prover Complexity

| Stage | Operation | Complexity |
|-------|-----------|------------|
| Stage 1 | Sumcheck per round | $O(2^{\text{vars}} \cdot \text{degree})$ |
| Stage 2 | Virtualization | $O(2^s)$ |
| Stage 3 | Jagged transform | $O(\text{dense\_size})$ |
| Stage 4 | Hyrax commitment | $O(\text{dense\_size})$ MSMs |

**Dominant cost**: Stage 1 sumcheck computation scales with constraint count.

**Parallelization**: Stage 1's three constraint types run in parallel.

### 6.8 Verifier Complexity

| Stage | Operation | Complexity |
|-------|-----------|------------|
| Stage 1 | Sumcheck verification | $O(\text{rounds} \cdot \text{degree})$ |
| Stage 2 | Claim aggregation | $O(c)$ |
| Stage 3 | Branching program | $O(\text{num\_polys} \cdot \text{bits})$ |
| Stage 4 | Hyrax verification | $O(\sqrt{\text{dense\_size}})$ |

**Key efficiency**: Stage 3 verifier uses $O(n)$ branching program instead of naive $O(2^{4n})$ MLE evaluation.

### 6.9 Comparison: Sparse vs Dense

Without jagged transform (sparse):
- Matrix size: $15 \cdot c_{\text{pad}} \times 256$
- For 256 GT exp constraints: $3,840 \times 256 = 983,040$ entries

With jagged transform (dense):
- Dense size: $1,024 \times 16 = 16,384$ entries
- **Compression ratio**: $60\times$

This compression directly reduces:
- PCS commitment size
- Stage 3 prover work
- Hyrax opening proof size

### 6.10 Scaling Summary

| Metric | Formula | 256-bit GT Exp |
|--------|---------|----------------|
| Constraints | $t$ (bit-length) | 256 |
| Stage 1 rounds | 4 (fixed for GT) | 4 |
| Stage 2 rounds | $\lceil \log_2(15c) \rceil$ | 12 |
| Stage 3 rounds | $\lceil \log_2(4tc \cdot 16) \rceil$ | 14 |
| Proof elements | $O(c + \log c)$ | ~1,100 |
| Prover time | $O(c \cdot 2^4)$ | ~4,000 ops |
| Verifier time | $O(c + \text{polys} \cdot \text{bits})$ | ~15,000 ops |

The protocol achieves **logarithmic scaling** in proof size relative to constraint count, with linear prover work and near-linear verifier work.

### 6.11 Unified Polynomial Cost Analysis

The packed GT exponentiation optimization dramatically reduces system costs:

**Memory Requirements**:

| Approach | Polynomials | Memory (256-bit exp) | Formula |
|----------|-------------|---------------------|---------|
| Unpacked | $4t$ | ~32 MB | $4t \times 2^4 \times 32$ bytes |
| Packed | 5 | ~10 MB | $5 \times 2^{12} \times 32$ bytes |

**Stage 2 Virtual Claims**:

| Approach | Virtual Claims | Verifier Work |
|----------|---------------|---------------|
| Unpacked | 1,024 | $O(1024)$ field ops |
| Packed | 5 | $O(5)$ field ops |

**Impact on Later Stages**:
- Stage 3 processes fewer polynomials (5 vs 1,024)
- Stage 3b benefits from reduced $K$ in batch verification
- Hyrax commitment is more efficient with fewer polynomials

**Trade-offs**:
- Packed approach uses 12-round sumcheck vs 4-round
- Slightly larger Stage 1 proof (60 vs 20 elements)
- Prover computes over larger domain ($2^{12}$ vs $2^4$)
- Net benefit: ~9.5× smaller total proof, ~200× fewer virtual claims

---

## 7. Implementation

This section describes the code architecture and data flow for the recursion implementation.

### 7.1 Module Structure

```
jolt-core/src/zkvm/recursion/
├── mod.rs                    # Re-exports and module docs
├── witness.rs                # Witness types (GTExpWitness, GTMulWitness, G1ScalarMulWitness)
├── constraints_sys.rs        # DoryMatrixBuilder, DoryMultilinearMatrix, ConstraintSystem
├── bijection.rs              # VarCountJaggedBijection, JaggedTransform trait
├── recursion_prover.rs       # RecursionProver orchestrating all stages
├── recursion_verifier.rs     # RecursionVerifier
├── stage1/
│   ├── square_and_multiply.rs   # GT exponentiation sumcheck
│   ├── gt_mul.rs                # GT multiplication sumcheck
│   └── g1_scalar_mul.rs         # G1 scalar mul sumcheck
├── stage2/
│   └── virtualization.rs        # Direct evaluation protocol
└── stage3/
    ├── jagged.rs                # Jagged transform sumcheck
    └── branching_program.rs     # O(n) MLE evaluation for g function
```

### 7.2 The Offloading Pattern

The recursion system uses a **hint-based offloading** pattern to reduce verification cost:

```
┌────────────────────────────────────────────────────────────────────┐
│                        PROVER SIDE                                 │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   Dory Proof ──→ witness_gen() ──┬──→ WitnessCollection (full)    │
│                                  └──→ HintMap (compact)            │
│                                                                    │
│   WitnessCollection ──→ RecursionProver ──→ RecursionProof        │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼  (send proof + hints)
┌────────────────────────────────────────────────────────────────────┐
│                       VERIFIER SIDE                                │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│   HintMap + RecursionProof ──→ verify_with_hint() ──→ Accept/Reject│
│                                                                    │
│   (Avoids expensive witness regeneration)                          │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

The `RecursionExt` trait defines this interface:

```rust
pub trait RecursionExt<F: JoltField>: CommitmentScheme<Field = F> {
    type Witness;  // Full witness data for proving
    type Hint;     // Compact hints for verification

    /// Generate witness and hints from a Dory proof
    fn witness_gen(&self, proof: &ArkDoryProof, ...) -> (Self::Witness, Self::Hint);

    /// Verify using hints instead of regenerating witnesses
    fn verify_with_hint(&self, proof: &RecursionProof, hint: &Self::Hint, ...) -> Result<()>;
}
```

### 7.3 Witness Types

Each expensive operation has a corresponding witness structure:

**GT Exponentiation** (`GTExpWitness`):
```rust
struct GTExpWitness {
    base: Fq12,                    // Base element a
    exponent: Fr,                  // Scalar k
    result: Fq12,                  // Result a^k
    rho_mles: Vec<Vec<Fq>>,        // Intermediate ρ values as MLEs
    quotient_mles: Vec<Vec<Fq>>,   // Quotient Q_i for each step
    bits: Vec<bool>,               // Binary decomposition of k
}
```

**GT Multiplication** (`GTMulWitness`):
```rust
struct GTMulWitness {
    lhs: Fq12,                     // Left operand a
    rhs: Fq12,                     // Right operand b
    result: Fq12,                  // Result c = a·b
    quotient_mle: Vec<Fq>,         // Quotient polynomial
}
```

**G1 Scalar Multiplication** (`G1ScalarMulWitness`):
```rust
struct G1ScalarMulWitness {
    base: G1Affine,                // Base point P
    scalar: Fr,                    // Scalar k
    result: G1Affine,              // Result [k]P
    x_a_mle: Vec<Fq>,              // Accumulator x-coordinates
    y_a_mle: Vec<Fq>,              // Accumulator y-coordinates
    x_t_mle: Vec<Fq>,              // Doubled point x-coordinates
    y_t_mle: Vec<Fq>,              // Doubled point y-coordinates
    x_a_next_mle: Vec<Fq>,         // Next accumulator x-coordinates
    y_a_next_mle: Vec<Fq>,         // Next accumulator y-coordinates
    indicator_mle: Vec<Fq>,        // Infinity indicators
    bits: Vec<bool>,               // Binary decomposition of k
}
```

**Packed GT Exponentiation** (Used when optimization enabled):

The packed representation combines all 256 steps into unified 12-variable polynomials:

```rust
struct PackedGTExpWitness {
    base: Fq12,                    // Base element a
    exponent: Fr,                  // Scalar k
    result: Fq12,                  // Result a^k

    // Packed polynomials (12 variables each)
    packed_base_mle: Vec<Fq>,      // base(x) replicated across steps
    packed_rho_mle: Vec<Fq>,       // ρ(s,x) for all steps
    packed_rho_next_mle: Vec<Fq>,  // ρ_next(s,x) shifted by 1
    packed_quotient_mle: Vec<Fq>,  // Q(s,x) for all steps
    packed_bit_mle: Vec<Fq>,       // bit(s) replicated across x
}
```

Layout: For 12-variable MLEs with `s ∈ {0,1}^8` and `x ∈ {0,1}^4`:
- Index formula: `index = x * 256 + s` (s in low bits)
- `packed_rho_mle[x * 256 + s] = ρ_s[x]`
- Each MLE has 2^12 = 4,096 evaluations

### 7.4 Constraint System Construction

The `DoryMatrixBuilder` constructs the constraint matrix from witnesses:

```rust
let mut builder = DoryMatrixBuilder::new();

// Add witnesses (each becomes one or more constraints)
for witness in gt_exp_witnesses {
    builder.add_gt_exp_witness(witness);  // t constraints per exponentiation
}
for witness in gt_mul_witnesses {
    builder.add_gt_mul_witness(witness);  // 1 constraint per multiplication
}
for witness in g1_scalar_mul_witnesses {
    builder.add_g1_scalar_mul_witness(witness);  // n constraints per scalar mul
}

let constraint_system: ConstraintSystem = builder.build();
```

The resulting `DoryMultilinearMatrix` has shape:

```
M : {0,1}^num_s_vars × {0,1}^num_x_vars → Fq

where:
  num_s_vars = ⌈log₂(15 × num_constraints_padded)⌉
  num_x_vars = 8  (max of 4-var GT and 8-var G1 polynomials)
```

### 7.5 Prover Flow

```rust
impl RecursionProver {
    pub fn prove(
        constraint_system: &ConstraintSystem,
        witnesses: &WitnessCollection,
        transcript: &mut Transcript,
    ) -> RecursionProof {
        // Stage 1: Run three sumchecks in parallel
        let stage1_provers = vec![
            SquareAndMultiplyProver::new(...),
            GtMulProver::new(...),
            G1ScalarMulProver::new(...),
        ];
        let (stage1_proof, stage1_claims) = BatchedSumcheck::prove(
            stage1_provers,
            &mut accumulator,
            transcript,
        );

        // Stage 2: Virtualization
        let stage2_prover = RecursionVirtualizationProver::new(
            constraint_system,
            &stage1_claims,
            transcript,
        );
        let (stage2_proof, stage2_claim) = stage2_prover.prove(...);

        // Stage 3: Jagged transform
        let stage3_prover = JaggedSumcheckProver::new(
            &bijection,
            &stage2_claim,
            transcript,
        );
        let (stage3_proof, dense_claim) = stage3_prover.prove(...);

        // PCS opening
        let opening_proof = accumulator.prove_openings(pcs, transcript);

        RecursionProof { stage1_proof, stage2_proof, stage3_proof, opening_proof }
    }
}
```

### 7.6 Verifier Flow

```rust
impl RecursionVerifier {
    pub fn verify(
        proof: &RecursionProof,
        verifier_input: &RecursionVerifierInput,
        transcript: &mut Transcript,
    ) -> Result<()> {
        // Stage 1: Verify batched sumcheck
        let stage1_verifiers = vec![
            SquareAndMultiplyVerifier::new(...),
            GtMulVerifier::new(...),
            G1ScalarMulVerifier::new(...),
        ];
        let stage1_claims = BatchedSumcheck::verify(
            stage1_verifiers,
            &proof.stage1_proof,
            &mut accumulator,
            transcript,
        )?;

        // Stage 2: Verify virtualization
        let stage2_verifier = RecursionVirtualizationVerifier::new(...);
        let stage2_claim = stage2_verifier.verify(&proof.stage2_proof, ...)?;

        // Stage 3: Verify jagged transform
        // Verifier computes f_jagged using branching program (O(n) time)
        let stage3_verifier = JaggedSumcheckVerifier::new(
            &verifier_input.bijection,
            &verifier_input.matrix_rows,  // Precomputed
        );
        let dense_claim = stage3_verifier.verify(&proof.stage3_proof, ...)?;

        // PCS verification
        accumulator.verify_openings(pcs, &proof.opening_proof, transcript)?;

        Ok(())
    }
}
```

### 7.7 Opening Accumulator

The `OpeningAccumulator` tracks virtual polynomial claims across stages:

```rust
// Stage 1 appends virtual claims
accumulator.append_virtual(
    VirtualPolynomial::RecursionBase(constraint_idx),
    SumcheckId::SquareAndMultiply,
    opening_point,
    claimed_value,
);

// Stage 2 reads Stage 1 claims
let (point, value) = accumulator.get_virtual_polynomial_opening(
    VirtualPolynomial::RecursionBase(i),
    SumcheckId::SquareAndMultiply,
);

// Stage 3 outputs committed polynomial claim
accumulator.append_committed(
    CommittedPolynomial::DenseMatrix,
    opening_point,
    claimed_value,
);
```

### 7.8 Jagged Bijection

The `VarCountJaggedBijection` maps between sparse and dense indices:

```rust
impl JaggedTransform for VarCountJaggedBijection {
    /// Given dense index i, return which polynomial it belongs to
    fn row(&self, dense_idx: usize) -> usize {
        // Binary search in cumulative_sizes
        self.cumulative_sizes.partition_point(|&s| s <= dense_idx)
    }

    /// Given dense index i, return offset within that polynomial
    fn col(&self, dense_idx: usize) -> usize {
        let poly_idx = self.row(dense_idx);
        dense_idx - self.cumulative_sizes[poly_idx]
    }

    /// Map (poly_idx, eval_idx) to dense index, if valid
    fn sparse_to_dense(&self, poly_idx: usize, eval_idx: usize) -> Option<usize> {
        if eval_idx < self.poly_sizes[poly_idx] {
            Some(self.cumulative_sizes[poly_idx] + eval_idx)
        } else {
            None  // Padded region
        }
    }
}
```

### 7.9 Matrix Row Mapping

The three-level mapping converts polynomial indices to matrix rows:

```rust
// Level 1: Dense index → Polynomial index (via bijection)
let poly_idx = bijection.row(dense_idx);

// Level 2: Polynomial index → (constraint_idx, poly_type)
let (constraint_idx, poly_type) = mapping.decode(poly_idx);

// Level 3: (constraint_idx, poly_type) → Matrix row
let matrix_row = poly_type as usize * num_constraints_padded + constraint_idx;
```

This mapping is precomputed for the verifier:

```rust
let matrix_rows: Vec<usize> = (0..num_polynomials)
    .map(|poly_idx| {
        let (constraint_idx, poly_type) = mapping.decode(poly_idx);
        poly_type as usize * num_constraints_padded + constraint_idx
    })
    .collect();
```

### 7.10 Polynomial Type Enumeration

```rust
#[repr(usize)]
pub enum PolyType {
    // GT Exponentiation (4-var MLEs)
    Base = 0,
    RhoPrev = 1,
    RhoCurr = 2,
    Quotient = 3,

    // GT Multiplication (4-var MLEs)
    MulLhs = 4,
    MulRhs = 5,
    MulResult = 6,
    MulQuotient = 7,

    // G1 Scalar Multiplication (8-var MLEs)
    G1ScalarMulXA = 8,
    G1ScalarMulYA = 9,
    G1ScalarMulXT = 10,
    G1ScalarMulYT = 11,
    G1ScalarMulXANext = 12,
    G1ScalarMulYANext = 13,
    G1ScalarMulIndicator = 14,

    // Packed GT Exponentiation (12-var MLEs)
    PackedGtExpBase = 15,
    PackedGtExpRho = 16,
    PackedGtExpRhoNext = 17,
    PackedGtExpQuotient = 18,
}

pub const NUM_POLY_TYPES: usize = 19;
```

### 7.11 Dory Integration: Witness Extraction and Hints

The recursion system integrates with Dory through the `RecursionExt` trait, which enables automatic witness extraction during verification.

#### The TraceContext Mechanism

Dory's `verify_recursive` function accepts a `TraceContext` that captures witnesses during verification:

```rust
// TraceContext modes:
TraceContext::for_witness_gen()  // Collect witnesses during verify
TraceContext::for_hints(hint)    // Use precomputed hints instead
```

When `verify_recursive` executes with a witness-gen context:
1. Each GT exponentiation triggers `WitnessGenerator::generate_gt_exp()`
2. Each GT multiplication triggers `WitnessGenerator::generate_gt_mul()`
3. Each G1 scalar mul triggers `WitnessGenerator::generate_g1_scalar_mul()`
4. Results accumulate in the shared `TraceContext`

#### witness_gen Implementation

```rust
fn witness_gen(
    proof: &ArkDoryProof,
    setup: &ArkworksVerifierSetup,
    transcript: &mut Transcript,
    point: &[Challenge],
    evaluation: &Fr,
    commitment: &ArkGT,
) -> Result<(WitnessCollection, HintMap), ProofVerifyError> {
    // Create context that will collect witnesses
    let ctx = Rc::new(
        TraceContext::<JoltWitness, BN254, JoltWitnessGenerator>::for_witness_gen()
    );

    // Run verification - witnesses collected as side effect
    verify_recursive(
        commitment, evaluation, point, proof, setup,
        transcript, ctx.clone(),
    )?;

    // Extract collected witnesses
    let witnesses = Rc::try_unwrap(ctx)
        .ok().expect("sole ownership")
        .finalize()?;

    // Convert to compact hints
    let hints = witnesses.to_hints::<BN254>();

    Ok((witnesses, hints))
}
```

#### JoltWitnessGenerator

The generator creates witness structures using optimized step computation:

```rust
impl WitnessGenerator<JoltWitness, BN254> for JoltWitnessGenerator {
    fn generate_gt_exp(base: &GT, scalar: &Fr, result: &GT) -> JoltGtExpWitness {
        // ExponentiationSteps computes all intermediate ρ values and quotients
        let steps = ExponentiationSteps::new(base.0, ark_to_jolt(scalar));
        JoltGtExpWitness {
            base: steps.base,
            exponent: steps.exponent,
            result: steps.result,
            rho_mles: steps.rho_mles,          // All ρ_i as MLEs
            quotient_mles: steps.quotient_mles, // All Q_i as MLEs
            bits: steps.bits,
        }
    }

    // Packed GT exponentiation witness generation
    fn generate_packed_gt_exp(base: &GT, scalar: &Fr, result: &GT) -> PackedGtExpWitness {
        let steps = ExponentiationSteps::new(base.0, ark_to_jolt(scalar));

        // Pack 256 4-var MLEs into 5 12-var MLEs
        let packed_base_mle = pack_replicated_base(&steps.base, 256, 16);
        let packed_rho_mle = pack_sequential_mles(&steps.rho_mles);
        let packed_rho_next_mle = pack_shifted_mles(&steps.rho_mles);
        let packed_quotient_mle = pack_sequential_mles(&steps.quotient_mles);
        let packed_bit_mle = pack_replicated_bits(&steps.bits, 16);

        PackedGtExpWitness {
            base: steps.base,
            exponent: steps.exponent,
            result: steps.result,
            packed_base_mle,
            packed_rho_mle,
            packed_rho_next_mle,
            packed_quotient_mle,
            packed_bit_mle,
        }
    }

    fn generate_gt_mul(lhs: &GT, rhs: &GT, result: &GT) -> JoltGtMulWitness {
        let steps = MultiplicationSteps::new(lhs.0, rhs.0);
        JoltGtMulWitness {
            lhs: steps.lhs,
            rhs: steps.rhs,
            result: steps.result,
            quotient_mle: steps.quotient_mle,
        }
    }

    fn generate_g1_scalar_mul(point: &G1, scalar: &Fr, result: &G1) -> JoltG1ScalarMulWitness {
        let steps = ScalarMultiplicationSteps::new(point.into(), ark_to_jolt(scalar));
        JoltG1ScalarMulWitness {
            point_base: steps.point_base,
            scalar: steps.scalar,
            result: steps.result,
            x_a_mles: steps.x_a_mles,
            y_a_mles: steps.y_a_mles,
            x_t_mles: steps.x_t_mles,
            y_t_mles: steps.y_t_mles,
            x_a_next_mles: steps.x_a_next_mles,
            y_a_next_mles: steps.y_a_next_mles,
            bits: steps.bits,
        }
    }
}
```

#### HintMap: Compact Verification Data

The `HintMap` contains precomputed values that allow verification to skip expensive operations:

```rust
struct HintMap<C: PairingCurve> {
    num_rounds: usize,
    // Precomputed intermediate values for each operation type
    gt_exp_hints: Vec<GtExpHint>,
    gt_mul_hints: Vec<GtMulHint>,
    g1_scalar_mul_hints: Vec<G1ScalarMulHint>,
}
```

#### verify_with_hint: Fast Verification Path

```rust
fn verify_with_hint(
    proof: &ArkDoryProof,
    setup: &ArkworksVerifierSetup,
    transcript: &mut Transcript,
    point: &[Challenge],
    evaluation: &Fr,
    commitment: &ArkGT,
    hint: &HintMap,  // Precomputed hints
) -> Result<(), ProofVerifyError> {
    // Context initialized with hints instead of empty
    let ctx = Rc::new(
        TraceContext::<JoltWitness, BN254, JoltWitnessGenerator>::for_hints(hint.clone())
    );

    // Same verify_recursive call, but:
    // - Uses precomputed hints instead of regenerating witnesses
    // - Skips ExponentiationSteps, MultiplicationSteps, ScalarMultiplicationSteps
    verify_recursive(
        commitment, evaluation, point, proof, setup,
        transcript, ctx,
    )?;

    Ok(())
}
```

#### Integration with RecursionProver

The recursion prover uses `witness_gen` to bootstrap from a Dory proof:

```rust
impl RecursionProver {
    pub fn new_from_dory_proof(
        dory_proof: &ArkDoryProof,
        verifier_setup: &ArkworksVerifierSetup,
        transcript: &mut Transcript,
        point: &[Challenge],
        evaluation: &Fr,
        commitment: &ArkGT,
    ) -> Result<Self, Error> {
        // Extract witnesses via TraceContext
        let (witness_collection, _hints) = DoryCommitmentScheme::witness_gen(
            dory_proof, verifier_setup, transcript,
            point, evaluation, commitment,
        )?;

        // Convert to recursion witness format
        let recursion_witness = Self::witnesses_to_dory_recursion(&witness_collection)?;

        // Build constraint system from extracted witnesses
        let constraint_system = Self::build_constraint_system(&witness_collection)?;

        Ok(Self { constraint_system, ... })
    }
}
```

#### Complete Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         DORY PROOF VERIFICATION                         │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
        ┌─────────────────────┐         ┌─────────────────────┐
        │    witness_gen()    │         │  verify_with_hint() │
        │  (Prover/Setup)     │         │   (Lightweight V)   │
        └─────────────────────┘         └─────────────────────┘
                    │                               │
                    ▼                               ▼
        ┌─────────────────────┐         ┌─────────────────────┐
        │   TraceContext::    │         │   TraceContext::    │
        │   for_witness_gen() │         │   for_hints(hint)   │
        └─────────────────────┘         └─────────────────────┘
                    │                               │
                    ▼                               ▼
        ┌─────────────────────┐         ┌─────────────────────┐
        │  verify_recursive() │         │  verify_recursive() │
        │  + WitnessGenerator │         │  + Hint lookups     │
        └─────────────────────┘         └─────────────────────┘
                    │                               │
         ┌──────────┴──────────┐                   │
         ▼                     ▼                   │
┌─────────────────┐  ┌─────────────────┐          │
│WitnessCollection│  │    HintMap      │──────────┘
│  (full data)    │  │   (compact)     │
└─────────────────┘  └─────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                      RECURSION SNARK (Stages 1-4)                       │
│  Proves that witness_gen produced correct results for all operations    │
└─────────────────────────────────────────────────────────────────────────┘
```

The key insight is that `witness_gen` runs full verification once to extract witnesses, while `verify_with_hint` uses precomputed hints to achieve ~150× speedup by skipping expensive intermediate computations.

### 7.12 GPU Considerations

The recursion prover has a clear separation between orchestration logic (Rust) and compute-intensive kernels (GPU).

#### GPU-Accelerated Components

| Component | Operation | Why GPU |
|-----------|-----------|---------|
| Stage 1 Sumcheck | Polynomial evaluations over hypercube | Parallel evaluation of $2^n$ points |
| Stage 2 Direct Eval | Direct polynomial evaluation | Efficient dot product computation |
| Stage 3 Sumcheck | Jagged transform sumcheck | Dense polynomial operations |
| Hyrax Commit | Multi-scalar multiplication (MSM) | $O(\sqrt{N})$ group operations |
| Hyrax VMP | Vector-matrix product | Parallelizable linear algebra |

#### Rust-Side Orchestration

The following remain on the CPU/Rust side:

- **Witness generation**: `TraceContext` collection during Dory verify
- **Transcript management**: Fiat-Shamir challenge generation
- **Bijection logic**: `VarCountJaggedBijection` index computations
- **Matrix row mapping**: Three-level decode (poly_idx → constraint_idx → matrix_row)
- **Hint serialization**: `HintMap` construction and transmission
- **Proof assembly**: Collecting stage outputs into `RecursionProof`

#### Data Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RUST (CPU)                                    │
├─────────────────────────────────────────────────────────────────────────┤
│  witness_gen() ──→ WitnessCollection ──→ ConstraintSystem               │
│                                              │                          │
│  Transcript ←─────────────────────────────── │ ←── challenges           │
│       │                                      │                          │
│       ▼                                      ▼                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                         GPU KERNELS                              │   │
│  ├─────────────────────────────────────────────────────────────────┤   │
│  │  Stage 1: sumcheck_prove(polys, eq_evals) → round_polys         │   │
│  │  Stage 2: direct_eval(virtual_claims, eq_evals) → evaluation    │   │
│  │  Stage 3: sumcheck_prove(dense, f_jagged) → round_polys         │   │
│  │  Hyrax:   msm(scalars, bases) → commitments                     │   │
│  │           vmp(matrix, vector) → result                          │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│       │                                                                 │
│       ▼                                                                 │
│  RecursionProof { stage1, stage2, stage3, opening }                    │
└─────────────────────────────────────────────────────────────────────────┘
```

#### Kernel Interface

Each sumcheck stage exposes a similar GPU interface:

```rust
trait SumcheckGpuKernel {
    /// Compute all round polynomials for one sumcheck
    fn prove_rounds(
        &self,
        evaluations: &GpuBuffer<F>,      // Polynomial evals on hypercube
        eq_evals: &GpuBuffer<F>,          // eq(r, x) precomputed
        challenges: &mut impl TranscriptReceiver,
    ) -> Vec<RoundPolynomial>;
}
```

Hyrax operations:

```rust
trait HyraxGpuKernel {
    /// Multi-scalar multiplication for row commitments
    fn msm(&self, scalars: &GpuBuffer<F>, bases: &GpuBuffer<G>) -> Vec<G>;

    /// Vector-matrix product for opening
    fn vmp(&self, matrix: &GpuBuffer<F>, vector: &[F]) -> Vec<F>;
}
```

#### Memory Considerations

| Stage | GPU Memory | Notes |
|-------|------------|-------|
| Stage 1 | $O(c \cdot 2^4)$ | Small per-constraint polynomials |
| Stage 2 | $O(2^s)$ | Full virtualized matrix |
| Stage 3 | $O(\text{dense\_size})$ | Compressed representation |
| Hyrax | $O(\sqrt{N})$ bases + scalars | MSM working set |

The jagged transform (Stage 3) reduces GPU memory pressure by ~60× compared to operating on the sparse matrix directly.

---

## References

- [Jolt Paper](https://eprint.iacr.org/2023/1217)
- [Dory Paper](https://eprint.iacr.org/2020/1274)
- [Hyrax Paper](https://eprint.iacr.org/2017/1132)
- [Jagged Polynomial Commitments](https://eprint.iacr.org/2024/504) - Claim 3.2.1 for jagged transform
