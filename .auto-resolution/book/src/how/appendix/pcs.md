# Polynomial Commitment Schemes

Polynomial commitment schemes (PCS) are a core cryptographic primitive in proof systems, including zkVMs.
At a high level, they allow a prover to *commit* to a polynomial and later *open* that commitment (i.e. evaluate the underlying polynomial) at a point chosen by the verifier, while ensuring the **binding** property (the prover cannot change the polynomial after committing to it).

This mechanism is powerful because it allows succinct verification of polynomial relationships without the prover needing to reveal the entire polynomial. The verifier only learns the evaluation at chosen points.

## Usage

A PCS comprises three algorithms (plus sometimes a preprocessing step): $\mathsf{Commit}$, $\mathsf{Prove}$, or $\mathsf{Verify}$.
In the context of a larger proof system like Jolt, polynomial commitment schemes are typically used in some variation of the following flow:

0. Depending on the PCS, there may be a preprocessing step, where some public data is generated. This data may be needed for $\mathsf{Commit}$, $\mathsf{Prove}$, or $\mathsf{Verify}$. This preprocessing could involve a [trusted setup](https://vitalik.eth.limo/general/2022/03/14/trustedsetup.html) (if not, it is called a *transparent* setup/PCS).
1. Prover commits to polynomial $P$, and the commitment $C := \mathsf{Commit}(P)$ is sent to the verifier.
2. In some subsequent part of the proof system, the prover and verifier arrive at some claimed evaluation $x := P(r)$ (typically for a random point $r \in \mathbb{F}^n$). For example, this claimed evaluation might be an [output claim](../architecture/architecture.md#sumchecks-as-nodes) of a sumcheck.
3. The prover provides an *opening proof* for the evaluation, and sends the proof $\mathsf{Prove}(P, r)$ to the verifier.
4. The verifier then verifies the opening proof. If the verification $\mathsf{Verify}(C, r, x)$ succeeds, then the verifier can rest assured that $P(r) = x$ as claimed.

## Role in Jolt

In Jolt, the prover commits to several witness polynomials derived from the execution trace being proven.
These polynomial commitments are used extensively to tie together the various algebraic claims generated during proving. Jolt currently uses **Dory** as its polynomial commitment scheme. Dory provides an efficient method for committing to and opening one-hot polynomials, while supporting [batch openings](../optimizations/batched-openings.md) across many claims with relatively low overhead.

## Further Reading

For a more formal and detailed introduction to polynomial commitment schemes, we recommend Section **7.3** of Justin Thaler’s *Proofs, Arguments, and Zero Knowledge*. That section explains how polynomial commitments integrate with interactive proof protocols such as GKR, and why they are essential for constructing succinct arguments.
