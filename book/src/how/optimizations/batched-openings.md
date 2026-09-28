# Batched openings

Jolt batches polynomial evaluation claims to amortize the cost of its final opening proof. The elliptic-curve-based [Dory](../dory.md) backend uses homomorphic batching; the lattice-based [Akita](../akita.md) backend uses prefix packing followed by a native grouped opening proof.
There are different notions of "batched openings", each necessitating its own subprotocol.

## Multiple polynomials, same point

$$f(x), g(x), \dots$$

If the polynomials are committed using an additively homomorphic commitment scheme (e.g. Dory), then this case can be reduced to a single opening claim.
See Section 16.1 of [Proof, Arguments, and Zero-Knowledge](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf) for details of this subprotocol.

This is the final batching step for Dory. Akita uses the [prefix-packing reduction](#prefix-packing-and-native-grouped-openings-akita) below for its committed trace.

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

## Prefix packing and native grouped openings (Akita)

Akita's trace commitment contains one physical polynomial, `OneHotTrace`, whose prefix selects among logical one-hot columns. All column claims use a common suffix point $x$ in `(cycle || address)` order. For columns $P_i$ at their assigned slots $i$, Jolt reduces their evaluations to a physical opening claim at $(s, x)$ with value

$$
v = \sum_i \widetilde{\textsf{eq}}(s, i) \cdot P_i(x).
$$

The slot-selector challenge $s$ is sampled after the layout, common point, and logical evaluations have been absorbed into the transcript. `PrefixPackedLayout::reduce_claims` in `crates/jolt-openings/src/prefix.rs` implements this reduction. Unused slots are omitted from the logical claim; the physical opening requires their aggregate contribution at the sampled selector to vanish.

Advice and committed-program objects have separate dense commitments. Their reduced claims join `OneHotTrace` in Akita's native grouped opening proof, with each object retaining its own shape and opening point. This supports a single backend proof for the whole batch without combining those commitments homomorphically. See the [Stage 8 description](../architecture/opening-proof.md#akita-grouped-opening) for the group order and implementation.
