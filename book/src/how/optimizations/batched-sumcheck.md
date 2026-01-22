# Batched sumcheck

There is a standard technique for batching multiple sumcheck instances together to reduce verifier cost and proof size.

Consider a batch of two sumchecks:

- Sumcheck A is over $n$ variables, and has degree $j$
- Sumcheck B is over $m$ variables, and has degree $k$

If we were to prove them serially, we would have one proof of size $\Theta(nj)$ and one proof of size $\Theta(mk)$ (and the same asymptotics for verifier cost).
If we instead prove them in parallel (i.e. as a batched sumcheck), we would instead have a single proof of size $\Theta(\max(n, m) \cdot \max(j, k))$.

For details, refer to Jim Posen's ["Perspectives on Sumcheck batching"](https://hackmd.io/s/HyxaupAAA).

Our implementation of batched sumcheck is uses the `SumcheckInstance` trait to represent individual instances, and the `BatchedSumcheck` enum to house the batch `prove`/`verify` functions.

## "Bespoke" batching

In some cases, we opt to implement a "bespoke" batched sumcheck prover rather than using the `BatchedSumcheck` pattern.
For example, the read-checking and raf-evaluation sumchecks for the [instruction execution](../architecture/instruction_execution.md) Shout instance are batched in a bespoke fashion, as are the read-checking and write-checking sumchecks in both Twist instances ([RAM](../architecture/ram.md) and [registers](../architecture/registers.md))

The reason for doing so is prover efficiency: consider two sumchecks:

$$
\text{claim}_1 = \sum_x f(x) \cdot g(x) \cdot h(x) \\
\text{claim}_2 = \sum_x  f(x) \cdot g(x) \cdot p(x)
$$

The batched sumcheck expression is:

$$
\text{claim}_1 + \gamma \cdot \text{claim}_2 = \sum_x f(x) \cdot g(x) \cdot h(x) + \gamma \cdot f(x) \cdot g(x) \cdot p(x)
$$

for some random $\gamma \in \mathbb{F}$.

Using the `BatchedSumcheck` trait, the prover message for each round would be computed by:

1. Computing the prover message for $f(x) \cdot g(x) \cdot h(x)$, which is some univariate polynomial $m_1(X)$.
2. Computing the prover message for $f(x) \cdot g(x) \cdot p(x)$, which is some univariate polynomial $m_2(X)$.
3. Computing the linear combination $m_1(X) + \gamma \cdot m_2(x)$.

As a general rule of thumb, the number of field multiplications required to compute the sumcheck prover message in round $i$ (using the standard linear-time sumcehck algorithm) is $(d - 1) \cdot n / 2^i$, where $d$ is the number of multiplications appearing in the summand expression.
So with the `BatchedSumcheck` pattern, the prover must perform $2n / 2^i$ field multiplications for both steps 1 and 2, for a total of $4n / 2^i$ multiplications.
If we instead leverage the shared terms in the batched sumcheck expression, we can reduce the total number of multiplications.
We can rearrange the terms of the batched sumcheck expression:

$$
\text{claim}_1 + \gamma \cdot \text{claim}_2 = \sum_x f(x) \cdot g(x) \cdot \left(h(x) + \gamma \cdot  p(x)\right)
$$

By factoring out the shared $f(x) \cdot g(x)$, we can directly compute $m_1(X) + \gamma \cdot m_2(x)$ using only $3n / 2^i$ multiplications.
