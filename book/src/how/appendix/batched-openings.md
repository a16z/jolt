# Batched Openings

At the end of the sumcheck protocol, the verifier must evaluate many multilinear polynomials at a single point. For polynomials which the verifier cannot compute on their own, they depend on the verification of a PCS opening proof provided by the prover. To save on verifier costs, all polynomials opened at the same point can be combined to a single opening proof.

The best reading on the subject can be found in **Section 16.1** of the [Textbook](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.pdf).
