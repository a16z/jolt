# Batched openings

Jolt uses techniques to batch multiple polynomial openings into a single opening claim to amortize the cost of [Dory](../dory.md) opening proof.
There are different notions of "batched openings", each necessitating its own subprotocol.

## Multiple polynomials, same point

$$f(x), g(x), \dots$$

If the polynomials are committed using an additively homomorphic commitment scheme (e.g. Dory), then this case can be reduced to a single opening claim.
See Section 16.1 of [Proof, Arguments, and Zero-Knowledge](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf) for details of this subprotocol.

## Multiple polynomials, multiple points

$$f(x), g(y), \dots$$

The most generic case.

Consider the case of two polynomials, opened at two different points $f(r_f), g(r_g)$.
We can use a [batched sumcheck](./batched-sumcheck.md) to reduce this to two polynomials opened at the same point.
The two sumchecks in the batch are:

$$
f(r_f) = \sum_x \widetilde{\textsf{eq}}(r_f, x) \cdot f(x) \\
g(r_g) = \sum_x \widetilde{\textsf{eq}}(r_g, x) \cdot g(x)
$$

so the batched sumcheck expression is:

$$
f(r_f) + \gamma \cdot g(r_g) = \sum_x \widetilde{\textsf{eq}}(r_f, x) \cdot f(x) + \gamma \cdot \widetilde{\textsf{eq}}(r_g, x) \cdot g(x)
$$

This sumcheck will produce [output claims](../architecture/architecture.md#sumchecks-as-nodes) $f(r')$ and $g(r')$, where $r'$ consists of the verifier (or Fiat-Shamir) challenges chosen over the course of the sumcheck.

If we further wish to reduce the claims $f(r')$ and $g(r')$ into a single claim, we can invoke the "Multiple polynomials, same point" subprotocol above.

This subprotocol was first described (to our knowledge) in Lemma 6.2 of [Local Proofs Approaching the Witness Length](https://eprint.iacr.org/2019/1062) [Ron-Zewi, Rothblum 2019].

## One polynomial, multiple points

$$f(x), f(y), \dots$$

Though this can be considered a special case of the above, there is also a subprotocol specific for this type of batched opening: see Section 4.5.2 of [Proof, Arguments, and Zero-Knowledge](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf).
We do not use this subprotocol in Jolt.
