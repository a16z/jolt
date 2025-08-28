# Batched opening proof

The final stage ([stage 5](./architecture.md)) of Jolt is the batched opening proof.
Over the course of the preceding stages, we obtain polynomial evaluation claims that must be proven using the [polynomial commitment scheme](../appendix/pcs.md)'s opening proof.
Instead of proving these openings "just-in-time", we **accumulate** them and defer the opening proof to the last stage (see `ProverOpeningAccumulator` and `VerifierOpeningAccumulator` for how this accumulation is implemented).
By waiting until the end, we can [batch-prove](../optimizations/batched-openings.md) all of the openings, instead of proving them individually.
This is important because the [Dory](../dory.md) opening proof is relatively expensive for the prover, so we want to avoid doing it multiple times.

## Three layers of reduction

In order to reduce all of the polynomial evaluation claims into a single opening, we use the subprotocols described in [Batched Openings](../optimizations/batched-openings.md), in a layered protocol.

### Layer 1

All of the polynomials in Jolt fall into one of two categerories: **one-hot** polynomials (the $\widetilde{\textsf{ra}$ and $\widetilde{\textsf{wa}$ arising in [Twist/Shout](../twist-shout.md)), and **dense** polynomials (we use this to mean anything that's not one-hot).

For *dense* polynomials evaluated at the same point, we apply the [Multiple polynomials, same point](../optimizations/batched-openings.md#multiple-polynomials-same-point) subprotocol to reduce them to a single dense polynomial evaluated at that point.

We do not do the same for the one-hot polynomials, because (a) they are usually evaluated at different points, and (b) we handle them lazily in [`RLCPolynomial`](#rlcpolynomial).

By doing this first layer of reduction, we reduce the number of sumcheck instances required in Layer 2:

### Layer 2

Layer 2 is an instance of the [Multiple polynomials, multiple points](../optimizations/batched-openings.md#multiple-polynomials-multiple-points) subprotocol, which is effectively just a batched sumcheck.
This is the fifth and final batched sumcheck in Jolt.

Each sumcheck instance represents either:

- a dense polynomial evaluation claim (represented by `DensePolynomialProverOpening`), or
- a one-hot polynomial evaluation claim (represented by `OneHotPolynomialProverOpening`)

The batched sumcheck leaves us with evaluation claims for multiple polynomials, all evaluated at the same point, leading us to Layer 3:

### Layer 3

Layer 3 is another instance of the [Multiple polynomials, same point](../optimizations/batched-openings.md#multiple-polynomials-same-point) subprotocol.
Unlike the usage in Layer 1, we must take a linear combination of both dense and one-hot polynomials.
This is implemented in `RLCPolynomial`, described below.

On the verifier side, this entails taking a linear combination of commitments.
Since Dory is an additively homomorphic commitment scheme, the verifier is able to do so.

Layer 3 leaves us with a single evaluation claim, which is finally proven using the Dory opening proof.

## `RLCPolynomial`

We use the `RLCPolynomial` struct to represent a random linear combination (RLC) of multiple polynomials, which may include both dense and one-hot polynomials.
We handle the two types separately:

- We **eagerly** compute the RLC of the dense polynomials. So if there are $N$ dense polynomials, each of length $T$ in the RLC, we compute the linear combination of their coefficients and store the result in the `RLCPolynomial` struct as a single vector of length $T$.
- We **lazily** compute the RLC of the one-hot polynomials. So if there are $N$ one-hot polynomials, we store $N$ (coefficient, reference) pairs in `RLCPolynomial` to represent the RLC. Later in the Dory opening proof when we need to compute a vector-matrix product using `RLCPolynomial`, we do so by computing the vector-matrix product using the individual one-hot polynomials (as well as the dense RLC) and taking the linear combination of the resulting vectors.
