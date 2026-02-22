# Dory

Dory is the [polynomial commitment scheme](./appendix/pcs.md) used in Jolt. It is based on the scheme described in [Lee21](https://eprint.iacr.org/2020/1274) and implemented in the [`a16z/dory`](https://github.com/a16z/dory/) repository.

## Background: AFGHO commitments

Dory builds on the AFGHO inner-product commitment scheme ([Aftuck-Fuchsbauer-Ghosh-Hofheinz-Oechsner, 2016](https://eprint.iacr.org/2016/457)). Given a bilinear group $(G_1, G_2, G_T, e)$ and public generators $\Gamma_1 = (g_1^{(1)}, \dots, g_1^{(n)}) \in G_1^n$, $\Gamma_2 = (g_2^{(1)}, \dots, g_2^{(n)}) \in G_2^n$, a vector $\mathbf{v} \in \mathbb{F}^n$ is committed as a pair:

$$
C_1 = \sum_{i=1}^{n} v_i \cdot g_1^{(i)} \in G_1, \qquad C_2 = \sum_{i=1}^{n} v_i \cdot g_2^{(i)} \in G_2
$$

The commitment is the pairing $C = e(C_1, C_2) \in G_T$.

AFGHO commitments are **additively homomorphic**: given commitments to two vectors, a commitment to any linear combination of them can be computed from the individual commitments without access to the underlying vectors. This property is critical for [batched openings](./optimizations/batched-openings.md) in Jolt.

## How Dory works

### Matrix layout

Dory views a multilinear polynomial with $N = 2^n$ coefficients as a $2^\nu \times 2^\sigma$ matrix, where $\nu + \sigma = n$. Concretely, the coefficient vector is arranged into rows of length $2^\sigma$, and the commitment proceeds in two tiers:

1. **Tier 1 (row commitments):** For each row $i$, compute a $G_1$ element:

$$
C_1^{(i)} = \sum_{j=1}^{2^\sigma} v_{i,j} \cdot g_1^{(j)}
$$

2. **Tier 2 (final commitment):** Combine the row commitments with $G_2$ generators via pairing:

$$
C = \sum_{i=1}^{2^\nu} e\!\left(C_1^{(i)},\; g_2^{(i)}\right) \in G_T
$$

The final commitment is a single $G_T$ element. The tier-1 row commitments are retained as a *hint* for the opening proof.

### Opening proofs

To prove that a committed polynomial evaluates to a claimed value $y$ at a point $\mathbf{r} \in \mathbb{F}^n$, Dory runs a reduction protocol. The point $\mathbf{r}$ is split into "row" and "column" components according to the matrix layout, and an inner-product argument (derived from the AFGHO structure) is used to prove the claimed evaluation. The proof has $O(\log n)$ group elements and can be verified with a constant number of pairings.

### Setup

Dory requires a universal reference string (URS) consisting of generators in $G_1$ and $G_2$. This URS is **transparent**: it is generated deterministically from a seed (using a hash-based PRG) with no trusted setup ceremony. Crucially, the URS has **sublinear size** in the polynomial length. Specifically, for a polynomial of length $N = 2^n$, the URS contains $O(2^{n/2}) = O(\sqrt{N})$ generators rather than $O(N)$, because the two-tier structure only needs generators for rows and columns independently.

## Why Dory?

Jolt's [Twist and Shout](./twist-shout.md) protocol requires the prover to commit to **one-hot polynomials**: vectors over $\{0, 1\}^{K^{1/c} \cdot T}$ with at most one nonzero entry per block of $K^{1/c}$ consecutive entries. These arise from representing memory-access addresses in one-hot form (see [Twist and Shout: one-hot polynomials](./twist-shout.md#one-hot-polynomials)).

These polynomials have two special properties that a PCS should exploit:

1. **Boolean coefficients.** Every coefficient is either 0 or 1. A PCS that charges "per field element" wastes work: each 254-bit field multiplication is doing the job of a single bit.
2. **Extreme sparsity.** Out of $K \cdot T$ coefficients, at most $T$ are nonzero.

Dory is well-suited to Jolt because of three properties:

### Sublinear key size

For a polynomial of length $N$, Dory's URS contains $O(\sqrt{N})$ group elements, compared to $O(N)$ for schemes like HyperKZG. This matters in Jolt because the one-hot polynomials can be very long ($K^{1/c} \cdot T$ coefficients, where $K$ is the address-space size and $T$ is the number of execution cycles).

### Pay-per-bit commitment costs

In the tier-1 step, each row commitment is computed via a multi-scalar multiplication (MSM). Dory (as implemented in Jolt) uses a `SmallScalar` trait that dispatches to specialized MSM routines for small coefficient types (booleans, `u8`, `u16`, etc.). When the coefficients are Boolean, the MSM reduces to a **subset sum** of generators &mdash; no scalar multiplications are needed at all. For small integer coefficients (e.g. `u8`), the MSM uses windowed methods with windows as small as 1--8 bits, rather than the 254-bit windows needed for full field elements.

The result is that committing to a polynomial whose coefficients are $b$-bit integers costs roughly $b/254$ times as much as committing to the same-length polynomial with arbitrary field-element coefficients. We call this **pay-per-bit** commitment cost.

### Efficient one-hot commitment

For one-hot polynomials specifically, Jolt further exploits the sparsity structure. Rather than running a full MSM over each row (most of whose entries are zero), the prover groups the nonzero indices by address and uses **batch $G_1$ additions**. Since each execution cycle contributes exactly one nonzero entry across all $K$ addresses, the cost of committing to a one-hot polynomial of length $K^{1/c} \cdot T$ is proportional to $T$ group additions rather than $K^{1/c} \cdot T$ MSM operations.

### Additive homomorphism

Because Dory commitments live in $G_T$ and are additively homomorphic, Jolt can batch-open many committed polynomials at a common point by taking a random linear combination (RLC) of the commitments. The verifier combines commitments (which are cheap $G_T$ operations), and the prover combines the underlying polynomials and produces a single opening proof. This is used throughout Jolt's [batched opening proof](./architecture/opening-proof.md) to amortize the cost of opening dozens of committed polynomials.

## Streaming commitment

In Jolt, witness polynomials can be committed in a **streaming** fashion: rather than materializing the entire polynomial in memory and then committing, the prover generates coefficients one row at a time during witness generation and immediately computes the tier-1 row commitment for that row. After all rows have been processed, a single tier-2 aggregation step produces the final $G_T$ commitment. This keeps memory usage proportional to $O(2^\sigma) = O(\sqrt{K^{1/c} \cdot T})$ (a single row) rather than $O(N)$ (the entire polynomial/matrix).

## Implementation

The Jolt implementation of Dory lives in `jolt-core/src/poly/commitment/dory/` and wraps the [`a16z/dory`](https://github.com/a16z/dory/) library. Key files:

- `commitment_scheme.rs` &mdash; Implements the `CommitmentScheme` and `StreamingCommitmentScheme` traits.
- `dory_globals.rs` &mdash; Manages per-context Dory matrix dimensions ($\nu$, $\sigma$) and coefficient layout.
- `wrappers.rs` &mdash; Bridges Jolt's `MultilinearPolynomial` types to Dory's polynomial interface, including specialized `commit_tier_1` for compact scalars and one-hot polynomials.
- `jolt_dory_routines.rs` &mdash; Custom implementations of low-level group operations (MSM, vector-scalar multiplication, folding) used by the Dory prover and verifier.
