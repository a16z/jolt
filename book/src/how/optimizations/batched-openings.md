# Batched openings

Jolt batches polynomial evaluation claims to amortize the cost of its final opening proof. The elliptic-curve-based [Dory](../dory.md) backend uses homomorphic batching; the lattice-based [Akita](../akita.md) backend uses native column batching and a grouped opening proof.
There are different notions of "batched openings", each necessitating its own subprotocol.

## Multiple polynomials, same point

$$f(x), g(x), \dots$$

If the polynomials are committed using an additively homomorphic commitment scheme (e.g. Dory), then this case can be reduced to a single opening claim.
See Section 16.1 of [Proof, Arguments, and Zero-Knowledge](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf) for details of this subprotocol.

This is the final batching step for Dory. Akita uses the [native grouped opening](#native-grouped-openings-akita) below for its committed trace.

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

## Native grouped openings (Akita)

Akita's `OneHotTrace` commitment is one native group of distinct one-hot
polynomials. All columns $P_i$ have `log_T + log_K` variables and open at the
common point $x$ in `(cycle || address)` order. Stage 8 passes their ordered
evaluations $(P_0(x), \ldots, P_{m-1}(x))$ directly to Akita.

The canonical layout fixes the identities, order, arity, and polynomial count.
The grouped statement binds the commitment metadata, point, and ordered
evaluations to the transcript before backend batching challenges are derived.
There are no trace selector variables or selector-reduction challenges.

Advice and committed-program objects have separate dense commitments. Their reduced claims join `OneHotTrace` in Akita's native grouped opening proof, with each object retaining its own shape and opening point. This supports a single backend proof for the whole batch without combining those commitments homomorphically. See the [Stage 8 description](../architecture/opening-proof.md#akita-grouped-opening) for the group order and implementation.
