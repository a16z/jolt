# Prover space control

There is a major impediment to Jolt’s usefulness today: it consumes large amounts of memory, which limits the length of RISC-V executions that it can prove. Proving about 16 million RISC-V cycles consumes hundreds of GBs of space.

There are two known techniques for reducing the prover space in zkVMs. One is called “continuations” and inherently involves proof recursion. This is the approach taken by almost all zkVM projects today. The other technique is non-recursive: it is well-known that sum-check-based SNARKs in particular do not require recursion to keep the prover's space under control.  Nonetheless, very few projects are pursuing this non-recursive approach (the only one we are aware of is [Ligetron](https://www.computer.org/csdl/proceedings-article/sp/2024/313000a086/1RjEaU3iZEY), which is not sum-check-based). We call these non-recursive techniques "the streaming prover".

Jolt will pursue both approaches to prover space control. Below, we provide more details on the two approaches.

# More detail on continuations

Continuations work by breaking a large computation into “chunks”, proving each chunk (almost) independently, and recursively aggregating the proofs (i.e., proving one knows the proofs for each chunk). 

Continuations come in two flavors: "brute-force recursion" and folding. In brute-force recursion, the proofs for different chunks are aggregated by having the prover prove it knows each proof. Roughly speaking, the verifier of each proof is represented as a circuit, and a SNARK is used to prove that the prover knows a satisfying assignment for each circuit. 

In folding schemes, the "proof" for each chunk is actually only a "partial proof", in particular omitting any evaluation proofs for any committed polynomials. The partial proofs for each chunk are not explicitly checked by anyone. Rather, they are "aggregated" into a single partial proof, and that partial proof is then "completed" into a full SNARK proof. In folding schemes, the prover winds up recursively proving that it correctly aggregated the partial proofs. This has major performance benefits relative to "brute force recursion", because aggregating proofs is much simpler than checking them. Hence, proving aggregation was done correctly is much cheaper than proving full-fledged proof verification was done correctly.

Folding schemes only work with elliptic-curve-based commitment schemes (there have been some efforts to extend folding techniques to hashing-based commitment schemes but they introduce large overheads). So most zkVMs do not pursue folding today as they use hashing-based commitment schemes. Fortunately, Jolt today does use elliptic curves and hence folding as a means of prover space control is available to it. A sketchy overview of how to integrate folding into Jolt is provided [here](https://jolt.a16zcrypto.com/future/folding.html).

# More detail on non-recursive space control

Brute-force recursion has the following downsides.

First, recursive proof aggregation imposes a lower bound of several GBs on prover space, because this is how much space is required to represent a SNARK verifier as a circuit and store that circuit. Second, the chunks are not completely independent, because one has to make sure that the state of the VM’s memory at the end of chunk $i$ equals the state of the VM’s memory at the start of chunk $i+1$. Ensuring this is an engineering headache and adds some performance overhead. Third, recursion actually changes the statement being verified. No longer is the statement “I correctly ran the computer program on a witness and it produced a claimed output $y$”. Rather it is: “I know several proofs, one per chunk, each attesting that the relevant chunk was run correctly, and with the final chunk outputting $y$.” Consequently, once proof recursion is used, it can actually be quite challenging for the public to tell that the right statement is being verified. 

Accordingly, we expect the non-recursive approach to space control to lead to a prover with smaller space than the continuations approach, and to a much simpler system overall. 

The non-recursive approach will also place the security of the SNARK on firmer footing: security of recursive SNARKs is heuristic, while non-recursive SNARKs like Jolt itself have unconditional security in idealized models such as the Random Oracle model (perhaps with additional assumptions such as hardness of discrete logarithms, depending on the choice of commitment scheme to use in Jolt). 

Crucial to non-recursive space control is that Lasso, the lookup argument used in Jolt, does not require any sorting. 

# Planned partial steps toward continuations
As discussed above, today Jolt is a monolithic SNARK. RISC-V traces cannot be broken up, they must be proven monolithically or not at all. As a result, Jolt has a fixed maximum trace length that can be proved which is a function of the available RAM on the prover machine. Long term we'd like to solve this by implementing a streaming version of Jolt's prover such that prover RAM usage is tunable with minimal performance loss. 

Short term we're going to solve this via a very primitive form of continuations we call monolithic chunking. The plan: Take a trace of length $N$ split it into $M$ chunks of size $N/M$ and prove each independently. $N/M$ is a function of the max RAM available to the prover. 

For the direct on-chain verifier there will be a cost linear in $M$ to verify. We believe this short-term trade-off is worthwhile for usability until we can implement the (more complicated) streaming algorithms. 

## Specifics
A generic config parameter will be added to the `Jolt` struct called `ContinuationConfig`. At the highest level, before calling `Jolt::prove` the trace will be split into `M` chunks. `Jolt::prove` will be called on each and return `RAM_final` which can be fed into `RAM_init` during the next iteration of `Jolt::prove`. The [output zerocheck](https://jolt.a16zcrypto.com/how/read_write_memory.html#ouputs-and-panic) will only be run for the final chunk. 

# References on non-recursive prover space control

<OL>
<LI> Verifiable computation using multiple provers. Andrew J. Blumberg, Justin Thaler, Victor Vu, and Michael Walfish. https://eprint.iacr.org/2014/846.

<LI> Public-coin zero-knowledge arguments with (almost) minimal time and space overheads. AR Block, J Holmgren, A Rosen, RD Rothblum, P Soni. Theory of Cryptography Conference, 168-197.

<LI> Gemini: Elastic SNARKs for diverse environments J Bootle, A Chiesa, Y Hu, M Orru. EUROCRYPT 2022

<LI> On black-box constructions of time and space efficient sublinear arguments from symmetric-key primitives. Laasya Bangalore, Rishabh Bhadauria, Carmit Hazay & Muthuramakrishnan Venkitasubramaniam. TCC 2022.

<LI> Ligetron: Lightweight Scalable End-to-End Zero-Knowledge Proofs. Post-Quantum ZK-SNARKs on a Browser. Ruihan Wang  Carmit Hazay  Muthuramakrishnan Venkitasubramaniam. 2024 IEEE Symposium on Security and Privacy (SP).
</OL>
