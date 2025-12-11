# Jolt Recursion via SNARK Composition

## 1. Motivation

Recursion is essential for two capabilities in Jolt:

1. **Proof Aggregation**: Maintain a single succinct proof that verifies the entire state of a blockchain, enabling succinct light clients and efficient state synchronization.
    
2. **Efficient Verification / On-Chain Verification**: Produce a final proof with single-digit millisecond verification on mobile devices and reasonable EVM gas costs. This typically requires wrapping in a constant-size proof system such as Groth16.
    

### Approaches to Recursion

There are two primary directions for achieving recursion in Jolt:

|Approach|Description|Trade-offs|
|---|---|---|
|**Folding-based**|Aggregate non-PCS components; amortize evaluation proof costs|Complex; requires careful protocol design|
|**Brute-force**|Compile Jolt verifier to RISC-V; prove execution in Jolt|Simple conceptually; expensive baseline|

This document explores the **brute-force** approach and techniques to make it practical.

---

## 2. Problem Statement

### Current State

The Jolt verifier, when compiled to RISC-V, requires approximately **1.5 billion cycles**.

### Target

Reduce the verifier cycle count to **< 10 million cycles** (a ~150× reduction).

### Strategy

We pursue two complementary techniques:

1. **Direct Optimization**: Inline functions, optimize primitives, streamline verifier logic.
    
2. **SNARK Composition**: Offload expensive computations from the RISC-V execution to a bespoke SNARK. The prover provides _hints_ for expensive operations, and the Jolt verifier accepts these hints, skipping the computation. A separate SNARK proves the hints are well-formed.
    

---

## 3. SNARK Composition Framework

### 3.1 Core Idea

Let $\mathcal{V}_{\text{Jolt}}$ denote the Jolt verifier. We decompose $\mathcal{V}_{\text{Jolt}}$ into:

$$\mathcal{V}_{\text{Jolt}} = \mathcal{V}_{\text{light}} \circ \mathcal{H}$$

where:

- $\mathcal{H} = {h_1, h_2, \ldots, h_m}$ is a set of **hints** provided by the prover
- $\mathcal{V}_{\text{light}}$ is a **lightweight verifier** that assumes hints are correct

The prover additionally supplies a proof $\pi_{\text{hint}}$ attesting that each hint $h_i$ is the correct output of some algorithm $\mathcal{A}_i$ on input $x_i$:

$$\pi_{\text{hint}} \vdash \bigwedge_{i=1}^{m} \left( h_i = \mathcal{A}_i(x_i) \right)$$

### 3.2 Identifying Expensive Operations

The Jolt verifier (for Dory-based PCS) performs the following expensive operations:

|Operation|Notation|Cycle Cost (approx.)|
|---|---|---|
|G1 Scalar Multiplication|$[k]P$ for $P \in \mathbb{G}_1$||
|G2 Scalar Multiplication|$[k]Q$ for $Q \in \mathbb{G}_2$||
|$\mathbb{G}_T$ Exponentiation|$a^k$ for $a \in \mathbb{G}_T$||
|$\mathbb{G}_T$ Multiplication|$a \cdot b$ for $a, b \in \mathbb{G}_T$||
|Multi-Pairing|$\prod_{i} e(P_i, Q_i)$||

### 3.3 The Bespoke Dory SNARK

We design a SNARK $\Pi_{\text{Dory}}$ that proves correct execution of the above operations. The SNARK is structured as:

$$\Pi_{\text{Dory}} = (\mathsf{Setup}, \mathsf{Prove}, \mathsf{Verify})$$

with the relation:

$$\mathcal{R}_{\text{Dory}} = \left\{ (x, w) : \text{all hinted operations are correct} \right\}$$

---

## 4. Notation and Preliminaries

Let $\mathbb{F}_r$ denote the scalar field of the BN254 curve. Let:

- $\mathbb{G}_1, \mathbb{G}_2$ be the source groups of the pairing
- $\mathbb{G}_T$ be the target group
- $e : \mathbb{G}_1 \times \mathbb{G}_2 \to \mathbb{G}_T$ be the bilinear pairing

For polynomials, we use the multilinear extension (MLE) representation. For a vector $\vec{v} \in \mathbb{F}^{2^n}$, we denote its MLE as $\tilde{v} : \mathbb{F}^n \to \mathbb{F}$.

---

## 5. Field Choice and Grumpkin Curve

All arithmetic in this SNARK is performed over $\mathbb{F}_q$, where $\mathbb{F}_q$ is the scalar field of the Grumpkin curve. This choice is crucial because the Dory witnesses for GT operations produce elements in $\mathbb{F}_q$ (BN254's base field), and the final polynomial commitment uses Hyrax over Grumpkin, which operates in $\mathbb{F}_q$. All sum-check protocols and polynomial evaluations occur in this field.

Grumpkin is a BN254-friendly curve whose scalar field matches BN254's base field $\mathbb{F}_q$. While Grumpkin does not have a pairing operation, Hyrax doesn't require pairings. The curve is optimized for scalar multiplications and multi-scalar multiplications, making it ideal for recursive proof composition.

Throughout this specification, $\mathbb{F}_q$ elements arise from Fq12 (GT) representations via MLE evaluations. Constraint variables are in $\mathbb{F}_q^4$, and all polynomial arithmetic occurs over $\mathbb{F}_q$.

---

## 6. Witness Generation and Algorithm Decomposition

The core challenge in proving expensive operations like GT exponentiation and multiplication is that these algorithms involve high-degree polynomial functions of their inputs. Direct constraint systems would require polynomials of prohibitive degree. The key insight is to decompose each algorithm into a sequence of linear steps, where each step's correctness can be verified with low-degree constraints.

### 6.1 Decomposition Strategy

For GT exponentiation $a^k$, the square-and-multiply algorithm naturally decomposes into steps where each intermediate value $\rho_{i+1}$ depends linearly on $\rho_i^2$ and potentially $a$. Similarly, GT multiplication decomposes into a single step relating inputs to output. The witness generator's role is to compute and collect these intermediate values that make the high-degree computation verifiable through linear constraints.

### 6.2 Witness Generation Implementation

The `JoltWitnessGenerator` implements `DoryWitnessGenerator` trait for BN254:

```rust
impl WitnessGenerator<JoltWitness, BN254> for JoltWitnessGenerator {
    fn generate_gt_exp(...) -> JoltGtExpWitness
    fn generate_gt_mul(...) -> JoltGtMulWitness
    // Other operations return UnimplementedWitness
}
```

For GT exponentiation, the witness generator calls `ExponentiationSteps::new(base, exponent)` from `jolt_optimizations` to compute the intermediate values $\rho_0, \ldots, \rho_t$ and quotient polynomials $Q_i$ for each step. These are packaged into `JoltGtExpWitness` containing the base, exponent, result, MLE representations of all intermediate $\rho$ values, quotient MLEs, and the binary representation of the exponent.

For GT multiplication, the witness generator computes the product $c = a \cdot b$ in Fq12 and calculates the quotient MLE as $Q(x) = \frac{a(x) \cdot b(x) - c(x)}{g(x)}$ for each $x \in \{0,1\}^4$. This is packaged into `JoltGtMulWitness` containing the left operand, right operand, result, and quotient MLE.

### 6.3 Constraint Polynomial g(x)

The constraint polynomial $g(x)$ is the irreducible polynomial that defines the Fq12 extension field over Fq. It's retrieved via `jolt_optimizations::get_g_mle()` which returns the evaluations of $g$ on the Boolean hypercube $\{0,1\}^4$. Since $g(x)$ is public and has only 4 variables, it's not committed to and is evaluated on-demand. The quotient computation $Q(x) = \frac{\text{constraint numerator}}{g(x)}$ ensures constraints vanish on the hypercube.

### 6.4 Ring Switching and Quotient Technique

An important observation is that $\mathbb{G}_T$ elements are represented as elements of the extension field $\mathbb{F}_{q^{12}}$, which is defined as $\mathbb{F}_q[X]/(p(X))$ where $p(X)$ is an irreducible polynomial of degree 12. Unlike $\mathbb{G}_1$ and $\mathbb{G}_2$ where the group operation follows the elliptic curve group law, the group operation in $\mathbb{G}_T$ is simply polynomial multiplication modulo $p(X)$.

This means that for $a, b, c \in \mathbb{G}_T$, the equation $a \cdot b = c$ holds if and only if there exists a quotient polynomial $Q$ such that:
$$a(X) \cdot b(X) = c(X) + Q(X) \cdot p(X)$$

When we evaluate this equation at the Boolean hypercube $\{0,1\}^4$ (viewing elements as 4-variate polynomials), we get:
$$a(x) \cdot b(x) - c(x) - Q(x) \cdot g(x) = 0$$

where $g(x)$ is the MLE of the irreducible polynomial $p$ evaluated on the hypercube. This quotient technique, which we call "ring switching," allows us to transform high-degree polynomial operations in $\mathbb{F}_{q^{12}}$ into linear constraints over $\mathbb{F}_q$ by introducing the quotient polynomial as an auxiliary witness. This is the foundation for both GT exponentiation and GT multiplication constraints.

---

## 7. Sum-Check Protocols

### 7.1 Elliptic Curve Scalar Multiplication (Double-and-Add)

**Applicable to**: $\mathbb{G}_1$ and $\mathbb{G}_2$

This protocol proves scalar multiplication $Q = [k]P$ using the double-and-add algorithm. For BN254, the curve equation is $y^2 = x^3 + b$.

#### Inputs / Outputs

|Role|Symbol|Description|
|---|---|---|
|**Public Input**|$P \in \mathbb{G}_1$ (or $\mathbb{G}_2$)|Base point $(x_P, y_P)$|
|**Public Input**|$k \in \mathbb{F}_r$|Scalar with public bit decomposition $(b_0, \ldots, b_{n-1})$|
|**Public Output**|$Q \in \mathbb{G}_1$ (or $\mathbb{G}_2$)|Result $Q = [k]P$|

#### Witness Structure

The witness is an execution trace with $n$ rows (one per scalar bit), padded to $N = 2^\ell$ where $\ell = \lceil \log_2 n \rceil$.

|Column|Symbol|Description|
|---|---|---|
|Current accumulator|$(x_{A}, y_{A})$|Accumulator point at start of iteration|
|Doubled point|$(x_{T}, y_{T})$|Result of doubling: $T = [2]A$|
|Next accumulator|$(x_{A'}, y_{A'})$|Accumulator after conditional addition|

Each row $i \in \{0, \ldots, n-1\}$ satisfies:
- **Doubling**: $T_i = [2]A_i$
- **Conditional addition**: $A_{i+1} = T_i + b_i \cdot P$ where $b_i \in \{0, 1\}$ is the $i$-th bit of $k$

#### MLE Representation

Each witness column is encoded as a multilinear extension over $\ell$ variables:

|MLE|Evaluations on $\{0,1\}^\ell$|
|---|---|
|$\widetilde{x_A}(x)$|$(x_{A_0}, x_{A_1}, \ldots, x_{A_{N-1}})$|
|$\widetilde{y_A}(x)$|$(y_{A_0}, y_{A_1}, \ldots, y_{A_{N-1}})$|
|$\widetilde{x_T}(x)$|$(x_{T_0}, x_{T_1}, \ldots, x_{T_{N-1}})$|
|$\widetilde{y_T}(x)$|$(y_{T_0}, y_{T_1}, \ldots, y_{T_{N-1}})$|
|$\widetilde{x_{A'}}(x)$|$(x_{A_1}, x_{A_2}, \ldots, x_{A_N})$|
|$\widetilde{y_{A'}}(x)$|$(y_{A_1}, y_{A_2}, \ldots, y_{A_N})$|

Note: $\widetilde{x_{A'}}$ and $\widetilde{y_{A'}}$ are the shifted accumulators (offset by one index).

#### Constraint System

At each iteration $i$, let $b_i \in \{0,1\}$ denote the $i$-th public bit of scalar $k$.

**C1 (Doubling - x-coordinate)**:
$$C_1 = 4\widetilde{y_A}^2(\widetilde{x_T} + 2\widetilde{x_A}) - 9\widetilde{x_A}^4$$

**C2 (Doubling - y-coordinate)**:
$$C_2 = 3\widetilde{x_A}^2(\widetilde{x_T} - \widetilde{x_A}) + 2\widetilde{y_A}(\widetilde{y_T} + \widetilde{y_A})$$

**C3 (Conditional addition - x-coordinate)**:
Since the MLEs encode the actual values from the double-and-add algorithm, we can reformulate without explicit bit values. When $b_i = 0$, we have $\widetilde{x_{A'}} = \widetilde{x_T}$, and when $b_i = 1$, the point addition constraint holds. We can multiply both cases:

$$C_3 = (\widetilde{x_{A'}} - \widetilde{x_T}) \cdot \bigl[(\widetilde{x_{A'}} + \widetilde{x_T} + x_P)(x_P - \widetilde{x_T})^2 - (y_P - \widetilde{y_T})^2\bigr]$$

**C4 (Conditional addition - y-coordinate)**:
Similarly for the y-coordinate:

$$C_4 = (\widetilde{y_{A'}} - \widetilde{y_T}) \cdot \bigl[\widetilde{x_T}(y_P + \widetilde{y_{A'}}) - x_P(\widetilde{y_T} + \widetilde{y_{A'}}) + \widetilde{x_{A'}}(\widetilde{y_T} - y_P)\bigr]$$

where $(x_P, y_P)$ are the constant base point coordinates.

**Constraint degrees**:

|Constraint|Degree|
|---|---|
|$C_1$|4|
|$C_2$|3|
|$C_3$|4| (increased due to multiplication by $(\widetilde{x_{A'}} - \widetilde{x_T})$)
|$C_4$|3| (increased due to multiplication by $(\widetilde{y_{A'}} - \widetilde{y_T})$)

#### Sum-Check Protocol

The prover establishes that all constraints vanish across all iterations:
$$0 = \sum_{x \in \{0,1\}^{\ell}} \widetilde{\text{eq}}(r, x) \cdot \sum_{j=1}^{4} \delta^{j-1} \cdot C_j(x)$$

where:
- $r \in \mathbb{F}^\ell$ is a random challenge vector
- $\delta \in \mathbb{F}$ batches the four constraint types
- $C_j(x)$ uses the public bit $b_i$ corresponding to index $x$

**Note on batching**: This protocol batches all $n$ iterations of double-and-add into a single sum-check over the $\ell$-dimensional hypercube. An alternative approach would be to run $n$ separate sum-checks (one per iteration), but batching provides better efficiency by amortizing the sum-check overhead across all iterations. The trade-off is that the witness MLEs must encode the entire trace, requiring $O(2^\ell)$ field elements.

**Multiple scalar multiplications**: When proving $k$ different scalar multiplications (e.g., $Q_1 = [k_1]P_1, \ldots, Q_k = [k_k]P_k$), each requires its own sum-check protocol since they have independent witnesses and constraints. These $k$ sum-checks can be batched together using random linear combinations, but fundamentally we need $k$ separate sum-check instances.

**Protocol**:
1. Verifier sends $\delta \leftarrow \mathbb{F}$
2. Verifier sends $r \leftarrow \mathbb{F}^\ell$
3. Prover and Verifier run sum-check (degree 5 rounds)
4. At final challenge $r' \in \mathbb{F}^\ell$, prover sends witness MLE evaluations
5. Verifier checks claims via PCS opening proofs

#### Output Claims

After sum-check completes at challenge $r'$, the prover provides:

|Claim|Value|
|---|---|
|$\widetilde{x_A}(r')$|$v_{x_A}$|
|$\widetilde{y_A}(r')$|$v_{y_A}$|
|$\widetilde{x_T}(r')$|$v_{x_T}$|
|$\widetilde{y_T}(r')$|$v_{y_T}$|
|$\widetilde{x_{A'}}(r')$|$v_{x_{A'}}$|
|$\widetilde{y_{A'}}(r')$|$v_{y_{A'}}$|

The verifier uses these claims along with the public bits to verify the final sum-check evaluation.

#### Witness Generation

##### Trace Generation

```
GenerateTrace(P, k, n):
    Input:  P = (x_P, y_P) ∈ G, scalar k, bit-length n
    Output: Trace T[0..N-1] where N = 2^⌈log₂(n)⌉

    bits ← BitDecompose(k, n)  // LSB first
    A ← O                       // Initialize with point at infinity

    for i in 0..n-1:
        T[i].xA, T[i].yA ← Affine(A)
        // Double
        Temp ← PointDouble(A)
        T[i].xT, T[i].yT ← Affine(Temp)
        // Conditional add
        if bits[i] = 1:
            A ← PointAdd(Temp, P)
        else:
            A ← Temp
        T[i].xA', T[i].yA' ← Affine(A)

    // Pad with valid dummy rows
    for i in n..N-1:
        T[i] ← DummyRow(P)

    return T
```


##### Padding Strategy

For rows $i \geq n$, the public bit $b_i = 0$. Use dummy rows satisfying all constraints:

```
DummyRow(P):
    A ← P
    T ← PointDouble(P)
    A' ← T              // Since b = 0, no addition
    return (A, T, A')
```

##### MLE Construction

```
BuildMLEs(T[0..N-1]):
    ℓ ← log₂(N)
    for i in 0..N-1:
        xA_evals[i] ← T[i].xA
        yA_evals[i] ← T[i].yA
        xT_evals[i] ← T[i].xT
        yT_evals[i] ← T[i].yT
        xA'_evals[i] ← T[i].xA'
        yA'_evals[i] ← T[i].yA'

    return {
        x̃_A  ← MLE(xA_evals),
        ỹ_A  ← MLE(yA_evals),
        x̃_T  ← MLE(xT_evals),
        ỹ_T  ← MLE(yT_evals),
        x̃_A' ← MLE(xA'_evals),
        ỹ_A' ← MLE(yA'_evals),
    }
```

#### Boundary Constraints

|Constraint|Description|
|---|---|
|$A_0 = O$|Standard initialization with point at infinity|
|$A_n = Q$|Output consistency (accumulator equals final result)|

---

### 7.2 $\mathbb{G}_T$ Exponentiation (Square-and-Multiply)

This protocol implements GT exponentiation using a square-and-multiply algorithm adapted for constraint systems. It computes $b = a^k$ where $a \in \mathbb{G}_T$ and $k \in \mathbb{F}_r$.

#### Inputs / Outputs

|Role|Symbol|Description|
|---|---|---|
|**Public Input**|$a \in \mathbb{G}_T$|Base element (represented as Fq12)|
|**Public Input**|$k \in \mathbb{F}_r$|Exponent (scalar field element)|
|**Public Output**|$b \in \mathbb{G}_T$|Result $b = a^k$|

#### Witness

|Symbol|Description|
|---|---|
|$\vec{b} = (b_0, \ldots, b_{t-1})$|Binary representation of exponent $k$|
|$\rho_0, \ldots, \rho_t$|Intermediate values where $\rho_0 = 1$, $\rho_t = a^k$|
|$Q_0, \ldots, Q_{t-1}$|Quotient polynomials for each constraint|

The intermediate values follow the recurrence:
$$\rho_{i+1} = \begin{cases}
\rho_i^2 & \text{if } b_i = 0 \\
\rho_i^2 \cdot a & \text{if } b_i = 1
\end{cases}$$

#### Constraint Formula

For each bit $i \in \{0, \ldots, t-1\}$, the constraint is:
$$C_i(x) = \rho_{i+1}(x) - \rho_i(x)^2 \cdot a(x)^{b_i} - Q_i(x) \cdot g(x) = 0$$

where:
- $a(x)^{b_i} = 1 + (a(x) - 1) \cdot b_i$ (linearization of the exponentiation)
- $g(x)$ is the irreducible polynomial defining the Fq12 extension field
- Each polynomial is evaluated over the 4-dimensional Boolean hypercube $\{0,1\}^4$

The constraint polynomial $g(x)$ is not committed to because it's a public polynomial with only 4 variables, making it cheap to compute on-demand. It's retrieved via `jolt_optimizations::get_g_mle()`.

#### Sum-Check Protocol

The prover establishes that
$$0 = \sum_{x \in \{0,1\}^4} \text{eq}(r_x, x) \cdot \sum_{i=0}^{t-1} \gamma^i \cdot C_i(x)$$

where:
- $r_x \in \mathbb{F}^4$ is a random challenge vector
- $\gamma \in \mathbb{F}$ is a batching coefficient
- $\text{eq}(r_x, x) = \prod_{j=0}^{3} (r_{x,j} \cdot x_j + (1 - r_{x,j}) \cdot (1 - x_j))$

**Note on multiple exponentiations**: This sum-check batches all $t$ constraints for a single GT exponentiation. When proving $k$ different GT exponentiations (e.g., $b_1 = a_1^{k_1}, \ldots, b_k = a_k^{k_k}$), the implementation batches all constraints from all exponentiations into a single sum-check. The constraint index would range from $0$ to $\sum_{j=1}^k t_j - 1$ where $t_j$ is the bit-length of exponent $k_j$.

#### Output Claims

After the sum-check protocol completes with final challenge $r_x'$, the prover outputs virtual polynomial claims:

|Claim|Description|
|---|---|
|$\tilde{a}(r_x') = v_{\text{base}}$|Base polynomial evaluation|
|$\tilde{\rho}_i(r_x') = v_{\rho,i}$ for $i \in \{0, \ldots, t\}$|Rho polynomial evaluations|
|$\tilde{Q}_i(r_x') = v_{Q,i}$ for $i \in \{0, \ldots, t-1\}$|Quotient polynomial evaluations|

These claims are accumulated in the `ProverOpeningAccumulator` as virtual polynomials with types:
- `VirtualPolynomial::RecursionBase(i)`
- `VirtualPolynomial::RecursionRhoPrev(i)` and `RecursionRhoCurr(i)`
- `VirtualPolynomial::RecursionQuotient(i)`

---

### 7.3 $\mathbb{G}_T$ Multiplication

This protocol proves the correctness of GT multiplication operations. It verifies that $c = a \cdot b$ for elements $a, b, c \in \mathbb{G}_T$.

#### Inputs / Outputs

|Role|Symbol|Description|
|---|---|---|
|**Public Input**|$a \in \mathbb{G}_T$|Left operand (represented as Fq12)|
|**Public Input**|$b \in \mathbb{G}_T$|Right operand (represented as Fq12)|
|**Public Output**|$c \in \mathbb{G}_T$|Result $c = a \cdot b$|

#### Witness

|Symbol|Description|
|---|---|
|$a$|Left operand as MLE|
|$b$|Right operand as MLE|
|$c$|Product as MLE|
|$Q$|Quotient polynomial|

The quotient polynomial is computed pointwise as:
$$Q(x) = \frac{a(x) \cdot b(x) - c(x)}{g(x)}$$
for each $x \in \{0,1\}^4$ where $g(x) \neq 0$.

#### Constraint Formula

The multiplication constraint is:
$$C(x) = a(x) \cdot b(x) - c(x) - Q(x) \cdot g(x) = 0$$

where $g(x)$ is the same irreducible polynomial defining Fq12, evaluated on-demand over the 4-dimensional Boolean hypercube.

#### Sum-Check Protocol

For a batch of $m$ multiplication constraints, the prover establishes that
$$0 = \sum_{x \in \{0,1\}^4} \text{eq}(r_x, x) \cdot \sum_{i=0}^{m-1} \gamma^i \cdot C_i(x)$$

where:
- $r_x \in \mathbb{F}^4$ is a random challenge vector
- $\gamma \in \mathbb{F}$ is a batching coefficient
- $C_i(x) = a_i(x) \cdot b_i(x) - c_i(x) - Q_i(x) \cdot g(x)$ for the $i$-th multiplication

#### Output Claims

After the sum-check protocol completes with final challenge $r_x'$, the prover outputs virtual polynomial claims:

|Claim|Description|
|---|---|
|$\tilde{a}_i(r_x') = v_{a,i}$|Left operand polynomial evaluation|
|$\tilde{b}_i(r_x') = v_{b,i}$|Right operand polynomial evaluation|
|$\tilde{c}_i(r_x') = v_{c,i}$|Result polynomial evaluation|
|$\tilde{Q}_i(r_x') = v_{Q,i}$|Quotient polynomial evaluation|

These claims are accumulated as virtual polynomials with types:
- `VirtualPolynomial::RecursionMulLhs(i)`
- `VirtualPolynomial::RecursionMulRhs(i)`
- `VirtualPolynomial::RecursionMulResult(i)`
- `VirtualPolynomial::RecursionMulQuotient(i)`

---

### 7.4 Multi-Pairing

This protocol proves the correctness of multi-pairing operations, computing $T = \prod_{i=1}^{m} e(P_i, Q_i)$ for collections of G1 and G2 points. The pairing computation decomposes into three sum-checks: Miller loop accumulation, final exponentiation, and line evaluation.

#### Inputs / Outputs

|Role|Symbol|Description|
|---|---|---|
|**Public Input**|$(P_1, \ldots, P_m) \in \mathbb{G}_1^m$|G1 points|
|**Public Input**|$(Q_1, \ldots, Q_m) \in \mathbb{G}_2^m$|G2 points|
|**Public Output**|$T \in \mathbb{G}_T$|Result $T = \prod_{i=1}^{m} e(P_i, Q_i)$|

#### Witness

|Symbol|Description|
|---|---|
|$f_0, f_1, \ldots, f_n$|Miller loop accumulator values|
|$\ell_i$|Line function evaluations at step $i$|
|$T_i$|Intermediate point multiples of $Q$|
|$\vec{b} = (b_0, \ldots, b_{k-1})$|Binary representation of loop parameter|

The protocol uses the fact that $Q_i = \psi^i(Q)$ where $\psi$ is the Frobenius endomorphism, allowing optimization of the multi-pairing computation.

#### Sum-Check 1: Miller Loop Accumulation

For each bit $b_i$ in the loop parameter, we accumulate:
$$f_i = f_{i-1}^2 \cdot \ell_i$$

where the line function $\ell_i$ depends on whether we're doubling or adding:
$$\ell_i = \begin{cases}
\ell_{\text{double}}(T_i, P) & \text{if } b_i = 0 \\
\ell_{\text{double}}(T_i, P) \cdot \ell_{\text{add}}(T_i, Q, P) & \text{if } b_i = 1
\end{cases}$$

The sum-check proves:
$$0 = \sum_{x \in \{0,1\}^{\ell}} \text{eq}(r_x, x) \cdot \sum_{i=0}^{k-1} \gamma^i \cdot \left(f_i(x) - f_{i-1}(x)^2 \cdot \ell_i(x)\right)$$

#### Sum-Check 2: Final Exponentiation

The final exponentiation computes:
$$T = f_n^{(p^{12}-1)/r}$$

This decomposes as:
$$\frac{p^{12}-1}{r} = \frac{(p^6-1)(p^2+1)}{(p^4-p^2+1)/r}$$

The sum-check proves the correctness of this exponentiation using a similar structure to GT exponentiation, with intermediate values tracking the computation through the factored form.

#### Sum-Check 3: Line Evaluation

**Status**: _TODO - Line evaluation constraints to be specified_

The line evaluation sum-check will verify that each $\ell_i$ is correctly computed from the point coordinates and slopes. This involves constraints on:
- Line slopes $\lambda$ for doubling and addition
- Evaluation of the line function at the pairing argument points
- Correct point arithmetic on the twist curve

#### Output Claims

The prover outputs virtual polynomial claims for:
- Miller loop accumulator values $\tilde{f}_i(r_x') = v_{f,i}$
- Line function evaluations $\tilde{\ell}_i(r_x') = v_{\ell,i}$
- Intermediate point coordinates for $T_i$
- Final exponentiation intermediate values

**Note**: Full implementation details for line evaluation constraints are pending.

---

## 8. Opening Reduction with Recursion Virtualization

The opening reduction occurs in two phases: first, a virtualization sum-check that reduces all Phase 1 claims to a single matrix evaluation, then a final batching that produces a single polynomial opening claim.

### 8.1 Why Virtualization?

With many small polynomials (each with 4 variables), we face a choice between opening each polynomial separately, homomorphic combination, or virtualization. The virtualization approach is optimal because Hyrax is most efficient when committing to a square matrix. With virtualization, we create a single matrix of size approximately $\sqrt{N} \times \sqrt{N}$ where $N$ is the total number of field elements. This avoids the linear cost of direct homomorphic combination, which would require $O(n)$ group operations as each small polynomial needs its own commitment. After virtualization, we have just one polynomial opening to prove.

### 8.2 Phase 2: Recursion Virtualization

After Phase 1 (constraint-specific sum-checks), we have virtual polynomial claims from GT exponentiation and multiplication protocols. These are organized in a unified matrix structure.

#### Matrix Organization

Define the giant multilinear matrix $M : \{0,1\}^s \times \{0,1\}^4 \to \mathbb{F}$ where:
- **Row variables** $s$: Select both polynomial type and constraint index
- **Column variables** $x$: The 4-dimensional constraint variables

The row indexing follows:
```
row_index = poly_type × num_constraints_padded + constraint_index
```

where `poly_type` is:
- 0: Base (GT exp)
- 1: RhoPrev (GT exp)
- 2: RhoCurr (GT exp)
- 3: Quotient (GT exp)
- 4: MulLhs (GT mul)
- 5: MulRhs (GT mul)
- 6: MulResult (GT mul)
- 7: MulQuotient (GT mul)

#### Virtualization Sum-Check

The Phase 2 sum-check proves:
$$\sum_{s \in \{0,1\}^{\log n}} \text{eq}(r_s, s) \cdot M(s, r_x) = v$$

where:
- $r_x$ is the final challenge from Phase 1 (binding the constraint variables)
- $r_s$ is a fresh random challenge for the row variables
- $v$ is the aggregated claim from all Phase 1 virtual polynomials

The input claim for virtualization is computed from Phase 1 outputs. For each polynomial type and constraint index, we have a claimed evaluation at point $r_x$. The aggregated claim is:

$$v = \sum_{i=0}^{n_{\text{exp}}-1} \gamma^i \cdot \left( v_{\text{base},i} + v_{\rho_{\text{prev}},i} + v_{\rho_{\text{curr}},i} + v_{Q,i} \right) + \sum_{j=0}^{n_{\text{mul}}-1} \gamma^{n_{\text{exp}}+j} \cdot \left( v_{a,j} + v_{b,j} + v_{c,j} + v_{Q_{\text{mul}},j} \right)$$

where:
- $v_{\text{base},i}, v_{\rho_{\text{prev}},i}, v_{\rho_{\text{curr}},i}, v_{Q,i}$ are the claimed evaluations from the $i$-th GT exponentiation constraint
- $v_{a,j}, v_{b,j}, v_{c,j}, v_{Q_{\text{mul}},j}$ are the claimed evaluations from the $j$-th GT multiplication
- $\gamma$ is the batching coefficient from Phase 1
- $n_{\text{exp}}$ is the number of GT exponentiation constraints
- $n_{\text{mul}}$ is the number of GT multiplication constraints

### 8.3 Final Opening Reduction

After the virtualization sum-check completes with challenge $r_s'$, we have:
$$M(r_s', r_x) = \frac{v}{\text{eq}(r_s, r_s')}$$

This single evaluation claim on the matrix polynomial $M$ is then passed to the polynomial commitment scheme.

### 8.4 Complete Protocol Flow

**Input Claims**:

|Source|Claims|Description|
|---|---|---|
|GT Exponentiation|$\{\tilde{a}(r_x), \tilde{\rho}_i(r_x), \tilde{Q}_i(r_x)\}$|Virtual polynomial evaluations from each constraint|
|GT Multiplication|$\{\tilde{a}_j(r_x), \tilde{b}_j(r_x), \tilde{c}_j(r_x), \tilde{Q}_j(r_x)\}$|Virtual polynomial evaluations from each multiplication|

**Virtualization Expression**:

The matrix $M$ encodes all witness polynomials in a unified structure:
$$M(s,x) = \begin{cases}
\text{base}_i(x) & \text{if } s \text{ selects Base polynomial for constraint } i \\
\text{rho_prev}_i(x) & \text{if } s \text{ selects RhoPrev for constraint } i \\
\text{rho_curr}_i(x) & \text{if } s \text{ selects RhoCurr for constraint } i \\
\vdots &
\end{cases}$$

**Output Claim**:

| Claim | Description |
| ----- | ----------- |
| $(M, (r_s', r_x), v_{\text{final}})$ | Single evaluation claim on the combined matrix polynomial |

This claim is then proven using the Hyrax polynomial commitment scheme.

---

## 9. Polynomial Commitment Scheme (Hyrax) with Grumpkin

After the opening reduction completes, we have a single claim $(M, (r_s', r_x), v_{\text{final}})$ on the matrix polynomial. We use Hyrax instantiated over the Grumpkin curve to prove this claim.

### 9.1 Hyrax Configuration

The implementation uses:
```rust
type HyraxPCS = Hyrax<RATIO=1, ark_grumpkin::Projective>;
```

Key parameters:
- **Curve**: Grumpkin (BN254-friendly curve with fast scalar field operations)
- **Ratio**: 1 (balances prover/verifier costs)
- **Matrix shape**: The $2^n$ evaluations are reshaped as a matrix for efficient commitment

### 9.2 Commitment Structure

For a polynomial $M$ with $2^n$ evaluations:
1. **Reshape**: View evaluations as a $2^{n_{\text{rows}}} \times 2^{n_{\text{cols}}}$ matrix
2. **Row commitments**: Commit to each row using Pedersen multi-commitments over Grumpkin
3. **Column generation**: Generate column polynomials for the opening protocol

The commitment is:
$$C = \{C_0, C_1, \ldots, C_{2^{n_{\text{rows}}}-1}\}$$
where each $C_i = \text{PedersenCommit}(\text{row}_i)$.

### 9.3 Opening Protocol

**Input**:
- Commitment $C$
- Opening point $(r_s', r_x) \in \mathbb{F}^{\log n}$
- Claimed value $v_{\text{final}}$

**Protocol**:
1. **Decompose point**: Split $(r_s', r_x)$ into row and column components
2. **Sum-check**: Prove the tensor product structure via sum-check
3. **Column opening**: Open the relevant column polynomial
4. **Verification**: Verifier checks consistency with the commitment

### 9.4 Why Grumpkin?

Grumpkin is chosen for several reasons:
- **BN254 compatibility**: Scalar field matches BN254's base field, enabling efficient non-native arithmetic
- **Fast operations**: Optimized for scalar multiplications and multi-scalar multiplications
- **Recursion-friendly**: Enables efficient recursive proof composition
- **Tested security**: Well-studied curve with known security properties

---

## 10. Integration with RecursionExt

The `DoryCommitmentScheme` implements `RecursionExt<Fr>` trait with two key methods. The `witness_gen` method runs Dory's recursive verification while collecting witnesses through a `TraceContext`, then converts the witness collection to hints for the lightweight verifier. The `verify_with_hint` method uses these pre-computed hints for efficient verification without regenerating witnesses. This integration enables the lightweight verifier to skip expensive GT operations by using the pre-computed hint values.

---

## 11. Cost Analysis

TODO: Complete cost analysis with concrete benchmarks.

---

## 12. Open Questions

1. **Batching**: How do we batch multiple instances of the same operation type?
    
2. **Field representation**: What is the optimal representation for $\mathbb{G}_T$ elements in the constraint system?
    
3. **Non-native arithmetic**: How do we handle $\mathbb{F}_p$ arithmetic inside an $\mathbb{F}_r$ constraint system?
    
4. **Curve operations**: Should we use projective or affine coordinates for EC constraints?
    
5. **Hyrax configuration**: What is the optimal row/column split for the Hyrax commitment?
    

---

## 13. References

- [Jolt Paper]
- [Dory Paper]
- [Hyrax Paper]
- [BN254 Pairing Implementation]
