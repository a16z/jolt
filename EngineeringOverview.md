# Engineering Overview

This document describes the Lasso proof system from an engineering perspective. Of course the most accurate and up-to-date view can be found in the code itself.

## Reformulation of Lasso: Figure 7

_The following section paraphrases Lasso Figure 7._

Params

-   $N$: Virtual table size
-   $C$: Subtable dimensionality
-   $m$: Subtable/memory size
-   $s$: Sparsity, i.e. number of lookups

$$
\sum_{i \in \\{0,1\\}^{log(s)}}{\tilde{eq}(i,r) \cdot T[\text{nz}[i]]}
$$

### 1. Prover commits to $3 \alpha$ $log(m)$-variate multilinear polynomials...

-   $`\text{dim}_1, ... \text{dim}_c`$: Purported subtable lookup indices
-   $E_1,...,E_\alpha$
-   $`\text{read\_counts}_{1},...,\text{read\_counts}_{\alpha}`$

### ...and $\alpha$ different $log(N)/C$-variate multilinear polynomials

-   $`\text{final\_counts}_{1},...,\text{final\_counts}_{\alpha}`$

$E_i$ is purported to evaluate to each of the $m$ reads into the corresponding $i$-th subtable.

### 2. Sumcheck $h(k) := \tilde{eq}(r,k) \cdot g(E_1(k), ..., E_\alpha(k))$

Reduces the check

$`v \stackrel{?}{=} \sum_{k \in \{0,1\}^{log(s)}}{g(E_{1}(k), ..., E_{\alpha}(k))}`$

to:

$`E_i(r_z) \stackrel{?}{=} v_{E_i}`$ for $`i=1,...,\alpha`$ given $`r_z \in \mathbb{F}^{log(s)}`$

### 3. Verifier checks $v_{E_i}$

$v_{E_i}$ is provided by sumcheck in step 2. $E_i(r_z)$ is provided by an oracle query to the initially committed polynomial.

### 4. Check $E_i(j) \stackrel{?}{=} T_i[\text{dim}_i(j)] \ \forall j \in \\{0,1\\}^{log(s)}$

-   Verifier provides $\tau, \gamma \in \mathbb{F}$
-   Prover and verifier run sumcheck protocol for grand products ([Tha13](https://eprint.iacr.org/2013/351.pdf)) to reduce the check equality between multiset hashes:
    $`
\mathcal{H}_{\tau, \gamma}(WS) = \mathcal{H}_{\tau, \gamma}(RS) \cdot \mathcal{H}_{\tau, \gamma}(S)
`$
-   Sumcheck reduces the check to (for $r''_i \in \mathbb{F}^\ell; r'''_i \in \mathbb{F}^{log(s)}$)
    -   $`E_{i}(r^{'''}_{i}) \stackrel{?}{=} v_{E_{i}}`$
    -   $`\text{dim}_i(r'''_i) \stackrel{?}{=} v_i$
    -   $`\text{read\_counts}_{i}(r^{'''}_{i}) \stackrel{?}{=} v_{\text{read\_counts}_{i}}`$
    -   $`\text{final\_counts}_{i}(r^{''}_{i}) \stackrel{?}{=} v_{\text{final\_counts}_{i}}`$

### 5. Verifier checks that the equations above hold with the RHS provided by sumcheck and the LHS provided by oracle queries to commitments in **Step 1**.

## Code Tour

### 1. Prover commitments

We convert a `C` sized vector of lookup indices into a `DensifiedRepresentation` which handles the construction of:

-   $`\text{dim}_i \ \forall i=1,...,C`$
-   $`E_i \ \forall i=1,...,C`$
-   $`\text{read\_counts}_i \ \forall i=1,...,C`$
-   $`\text{final\_counts}_i \ \forall i=1,...,C`$

Each of these is stored as a (dense) mutlilinear polynomial in its Lagrange basis representation.

Finally, we merge all $log(s)$-variate polynomials ($`\text{dim}_i, E_i, \text{read\_counts}_i`$) into a single dense multilinear polynomial, and merge all the $log(m)$-variate polynomials ($`\text{final\_counts}_i`$) into a single polynomial for commitment / opening efficiency.

Now we can commit these 2 merged multilinear polynomials via any (dense) multilinear polynomial commitment scheme. This code is handled by `SparsePolynomialCommitment` -> `SparsePolyCommitmentGens` -> `PolyEvalProof` -> `DotProductProofLog` -> .... Initially we use Hyrax from [Spartan](https://github.com/microsoft/Spartan) as the dense PCS, but this could be swapped down the road for different performance characteristics.

After inital commitment, `SparsePolynomialEvaluationProof::<_, _, _, SubtableStrategy>::prove(dense, ...)` is called. `SubtableStrategy` describes which table collation function `g` will be used and which set of subtables `T_i` to materialize.

`Subtables::new()`: First we materialize the subtables and read the entries at their respective lookup indices. These entries determine (via Lagrange interpolation) the $E_i$ polynomials. We encode each multilinear $E_i$ polynomial and concatenate them into a single dense multilinear polynomial before committing as in Step 1.

### 2. Sumcheck table assembly

First, `Subtables::compute_sumcheck_claim`: computes the combined evaluations of $E_i(k)$ over all $k \in \\{0,1\\}^{log(s)}$ .

Run a generic `SumcheckInstanceProof::prove_arbitrary` assuming the lookup polynomials $E_i(k)$ were formed honestly.

### 3. Verifier checks $v_{E_{i}} \stackrel{?}{=} E_{i}(r_z)$

`CombinedTableEvalProof::prove`: Create the combined opening proof from the dense PCS.

### 4. Check $E_i \stackrel{?}{=} T_i[\text{dim}_i(j)] \ \forall j \in \{0,1\}^{log(s)}$

The valid formation of $E_i$ is checked using memory checking techniques described in Section 5 of Lasso or Section 7.2 of Spartan.

This step gets a bit messy becuase we combine each dimension of the memory checking sumcheck into a single sumcheck via a random linear combination of the input polynomials.

Idea is to use homomorphic multiset hashes to ensure set equality.

`MemoryCheckingProof::prove()`

-   `Subtables::to_grand_products()`: Create the Reed-Solomon fingerprints from each set
    -   `GrandProducts::new()`
    -   `GrandProducts::build_grand_product_inputs()`
-   `ProductLayerProof::prove()`: Prove product (multiset hash) of a set's Reed-Solomon fingerprints
    -   `BatchedGrandProductArgument::prove()`
-   `HashLayerProof::prove()`: Prove Reed-Solomon evaluations directly

See [imgs/memory-checking.png](imgs/memory-checking.png) for a visual explainer of the memory checking process.
