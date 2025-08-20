# Zero knowledge

One way to achieve zero-knowledge is to simply compose Jolt with a zero-knowledge SNARK like Groth16. That is, use Groth16 (which is ZK) to prove that one knows a Jolt proof. Composing with Groth16 or Plonk is how most zkVMs get low on-chain verification costs anyway, and it also "adds" ZK. This approach is on Jolt's roadmap, although it will take some time to complete (as it requires representing the Jolt verifier in R1CS or Plonkish, which is a pain). 

A second way to achieve zero-knowledge is to combine Jolt with folding, which we will do regardless, in order to make the prover space independent of the number of RISC-V cycles being proven. As described in Section 7 of the latest version of the [HyperNova paper](https://eprint.iacr.org/2023/573),
one can straightforwardly obtain zero-knowledge directly from folding, without composition with a zkSNARK like Groth16.

There are also ways to make Jolt zero-knowledge without invoking SNARK composition. For example, rendering sum-check-based SNARKs zero-knowledge without using composition was exactly the motivation for [Zeromorph](https://eprint.iacr.org/2023/917.pdf), which introduces a very efficient zero-knowledge variant of KZG commitments for multilinear polynomials.

A final technique to render all of the sum-checks ZK without SNARK composition is given in [Hyrax](https://eprint.iacr.org/2017/1132.pdf) (based on old work of Cramer and Damgård). Roughly, rather than the prover sending field elements "in the clear", it instead sends (blinded, hence hiding) Pedersen commitments to these field elements. And the verifier exploits homomorphism properties to confirm that the committed field elements would have passed all of the sum-check verifier's checks. See Section 13.2 of [Proofs, Arguments, and Zero-Knowledge](https://people.cs.georgetown.edu/jthaler/ProofsArgsAndZK.html) for additional discussion.
