# Batched openings

Jolt uses techniques to batch multiple polynomial openings into a single opening claim to amortize the cost of [Dory](../dory.md) opening proof.
There are different notions of "batched openings", each necessitating its own subprotocol.

## Multiple polynomials, same point

$$f(x), g(x), \dots$$

See Section 16.1 of [Proof, Arguments, and Zero-Knowledge](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf) for details of this subprotocol.

## Multiple polynomials, multiple points

$$f(x), g(y), \dots$$

The most generic case.

TODO(moodlezoup)

## One polynomial, multiple points

$$f(x), f(y), \dots$$

We do not use this type of batched opening in Jolt, but there is also a subprotocol for this: see Section 4.5.2 of [Proof, Arguments, and Zero-Knowledge](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf).
