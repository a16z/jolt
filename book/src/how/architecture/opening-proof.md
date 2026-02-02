# Batched opening proof

The final stage ([stage 8](./architecture.md)) of Jolt is the batched opening proof.
Over the course of the preceding stages, we obtain polynomial evaluation claims that must be proven using the [polynomial commitment scheme](../appendix/pcs.md)'s opening proof.
Instead of proving these openings "just-in-time", we **accumulate** them and defer the opening proof to the last stage (see `ProverOpeningAccumulator` and `VerifierOpeningAccumulator` for how this accumulation is implemented).
By waiting until the end, we can [batch-prove](../optimizations/batched-openings.md) all of the openings, instead of proving them individually.
This is important because the [Dory](../dory.md) opening proof is relatively expensive for the prover, so we want to avoid doing it multiple times.

## Claim reduction sumchecks

Throughout the earlier stages of Jolt, various components generate multiple polynomial evaluation claims that need to be consolidated before the final opening proof. Conceptually, claim reduction sumchecks are instantiation of the ["Multiple polynomials, multiple points"](../optimizations/batched-openings.md#multiple-polynomials-multiple-points) subprotocol.

These claim reduction sumchecks serve two purposes:

1. Reduce the number of claims that need to be virtualized by a subsequent sumcheck. E.g. if the same virtual polynomial $P$ is opened at two different points $r_1$ and $r_2$, a claim reduction can be applied to avoid running two instances of the sumcheck that virtualized $P$. 
2. Reduce the number of claims that need to be proven via PCS opening proof. While we can leverage the homomorphic properties of Dory in the [Multiple polynomials, same point](../optimizations/batched-openings.md#multiple-polynomials-same-point) subprotocol, we must first reduce multiple opening points to a single, unified opening point.

The claim reduction sumchecks can be found in `jolt-core/src/zkvm/claim_reductions/` and include:

- **Instruction lookups** (`instruction_lookups.rs`): Aggregates instruction lookup claims (lookup outputs and operands) from Spartan.
- **Registers** (`registers.rs`): Reduces register read/write claims (rd, rs1, rs2) from Spartan.
- **RAM RA** (`ram_ra.rs`): Consolidates the four RAM read-address (RA) claims from various RAM-related sumchecks (raf evaluation, read-write checking, Val evaluation, Val-final evaluation) into a single claim for the RA virtualization sumcheck.
- **Increments** (`increments.rs`): Reduces claims related to increment checks.
- **Hamming weight** (`hamming_weight.rs`): Reduces hamming weight-related claims.
- **Advice** (`advice.rs`): Reduces claims from advice polynomials.

### How claim reduction sumchecks work

A claim reduction sumcheck takes multiple polynomial evaluation claims, potentially at different evaluation points, and consolidates them into a single claim (or fewer claims) using the sumcheck protocol. The general pattern is:

1. **Input**: Multiple polynomial evaluation claims of the form $P_i(\mathbf{r}_i) = v_i$
2. **Batching**: Random challenge $\gamma$ is sampled from the transcript to batch the claims together
3. **Sumcheck**: Prove a sumcheck identity of the form:
   $$\sum_{\mathbf{x}} \text{eq}(\mathbf{r}_1, \mathbf{x}) \cdot P_1(\mathbf{x}) + \gamma \cdot \text{eq}(\mathbf{r}_2, \mathbf{x}) \cdot P_2(\mathbf{x}) + \ldots = v_1 + \gamma \cdot v_2 + \ldots$$
4. **Output**: Polynomial evaluation claims of the form $P_i(\mathbf{r}') = v'_i$ for a **single**, unified point $\mathbf{r}'$ derived from the sumcheck challenges.

## Final reduction

After the claim reduction sumchecks have consolidated related claims, we perform a final reduction to prepare for the Dory opening. 

We apply the [Multiple polynomials, same point](../optimizations/batched-openings.md#multiple-polynomials-same-point) subprotocol to reduce the claims to a single claim, namely the evaluation of an `RLCPolynomial` representing a random linear combination of all the opened polynomials.

On the verifier side, this entails taking a linear combination of commitments.
Since Dory is an additively homomorphic commitment scheme, the verifier is able to do so.

### `RLCPolynomial`

Recall that all of the polynomials in Jolt fall into one of two categories: **one-hot** polynomials (the $\widetilde{\textsf{ra}}$ and $\widetilde{\textsf{wa}}$ arising in [Twist/Shout](../twist-shout.md)), and **dense** polynomials (we use this to mean anything that's not one-hot).

We use the `RLCPolynomial` struct to represent a random linear combination (RLC) of multiple polynomials, which may include both dense and one-hot polynomials.
We handle the two types separately:

- We **eagerly** compute the RLC of the dense polynomials. So if there are $N$ dense polynomials, each of length $T$ in the RLC, we compute the linear combination of their coefficients and store the result in the `RLCPolynomial` struct as a single vector of length $T$.
- We **lazily** compute the RLC of the one-hot polynomials. So if there are $N$ one-hot polynomials, we store $N$ (coefficient, reference) pairs in `RLCPolynomial` to represent the RLC. Later in the Dory opening proof when we need to compute a vector-matrix product using `RLCPolynomial`, we do so by computing the vector-matrix product using the individual one-hot polynomials (as well as the dense RLC) and taking the linear combination of the resulting vectors.
