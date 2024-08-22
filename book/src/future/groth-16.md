# Groth16 Recursion
Jolt's verifier with no proof composition is expensive. We [estimate](on-chain-verifier.md) about 2 million gas to verify within the EVM. Further, the short-term [continuations plan](https://jolt.a16zcrypto.com/future/continuations.html) causes a linear blowup depending on the count of monolithic trace chunks proven.

To solve these two issues we're aiming to add a configuration option to the Jolt prover with a post processing step, which creates a Groth16 proof of the Jolt verifier for constant proof size / cost (~280k gas on EVM) regardless of continuation chunk count or opening proof cost. This technique is industry standard. (An added benefit of this proof composition is that it will render 
Jolt proofs [zero-knowledge](zk.md)). 

We call directly representing the Jolt verifier (with HyperKZG polynomial commitment scheme) 
as constraints to then feeding those constraints into Groth16 "naive composition". Unfortunately, this naive procedure
will result 
in over a hundred millions of constraints. Applying Groth16 
to such a large constraint system will result in far more latency than we'd lik, (and may even be impossible over the BN254 scalar field
because that field only supports FFTs of length $2^{27}$. 
Below, we describe alternate ways forward. 

# Cost accounting for naive composition

In naive composition, the bulk of the constraints are devoted
to implementing the [~150 scalar multiplications done by the Jolt verifier](on-chain-verifier.md) 
(or more accurately, ~150 scalar multiplications will be the dominant verifier cost once we finish all optimizations that
are now in progress). 

Each scalar multiplication costs about 400 group operations, each of which costs about 10 field multiplications,
so that's about $150 \cdot 400 \cdot 10=600k$ field multiplications.
The real killer is that these field multiplications must be done non-natively in constraints, due to the fact
that that BN254 does not have a pairing-friendly  "sister curve" (i.e., a curve whose scalar field matches the BN254 base field).
This means that each of the $600k$ field multiplications costs thousands of constraints. 

On top of the above, the two pairings done by the HyperKZG verifier, implemented non-natively in constraints, 
should cost about 6 million constraints. 

We do expect advances in representing non-native field arithmetic in constraints to bring the above numbers
down some, but not nearly enough to make naive composition sensible. 

# A first step to reducing gas costs

A simple first approach to reducing gas costs is to leave all scalar multiplications and HyperKZG-evaluation-proof-checking done by the Jolt verifier out of the composition, doing those directly on-chain. 

This would reduce gas costs to slightly over 1 million, and ensure that Groth16 is run only on a few million constraints
(perhaps even less, given upcoming advances in representing non-native field arithmetic in constraints).

# A potential way forward 

Here is a first cut at a plan to bring on-chain Jolt verification down to a couple hundred thousand gas.

First, represent the Jolt verifier (with HyperKZG-over-BN254 as the polynomial commitment scheme) as an R1CS instance over the Grumpkin scalar field, and apply Spartan (with Hyrax-over-Grumpkin as the polynomial commitment) to this R1CS. This R1CS should only have a few million constraints. This is because all scalar multiplications and pairings done by the Jolt verifier are native over Grumpkin's scalar field. (The constraint system is also highly uniform, which can benefit both the Spartan prover and verifier)

The field operations
done by the Jolt verifier in the various invocations of the sum-check protocol need to be represented non-natively in these R1CS constraints, but since there are only about 2,000 such field operations they still only cost perhaps 5 million constraints in total. See the [Testudo](https://eprint.iacr.org/2023/961) paper for a similar approach and associated calculations. 

We expect upcoming advances in 
methods for addressing non-native field arithmetic (and/or more careful optimizations of the Jolt verifier) to bring this down to under 2 million constraints. 

But the Spartan proof is still too big to post on-chain. So, second, represent the Spartan verifier as an R1CS instance over the BN254 scalar field, and apply Groth16 to this R1CS. This the proof
posted and verified on-chain. 

The Spartan verifier only does at most a couple of hundred field operations (since there's only two sum-check invocations in Spartan) and $2 \cdot \sqrt{n}$ scalar multiplications where $n$ is the number of columns (i.e., witness variables) in the R1CS instance.
$2 \sqrt{n}$ here will be on the order of 3,000. Each scalar multiplication (which are done natively in this R1CS) yields 
about $4,000$ constraints. So that's 12 million constraints being fed to Groth16.

The calculation above is quite delicate, because running Groth16 on 12 million constraints would be okay for many settings, but 50 million constraints would not be. We do think 12 million is about the right estimate for this approach, especially given that 2,000 field operations for the Jolt verifier is a conservative estimate. 

## Details of the Jolt-with-HyperKZG verifier
The easiest way to understand the workload of the verifier circuit is to jump through the codebase starting at `vm/mod.rs Jolt::verify(...)`.  Verification can be split into 4 logical modules: [instruction lookups](https://jolt.a16zcrypto.com/how/instruction_lookups.html), [read-write memory](https://jolt.a16zcrypto.com/how/read_write_memory.html), [bytecode](https://jolt.a16zcrypto.com/how/bytecode.html), [r1cs](https://jolt.a16zcrypto.com/how/r1cs_constraints.html).

Each of the modules do some combination of the following:
- [Sumcheck verification](https://jolt.a16zcrypto.com/background/sumcheck.html)
- Polynomial opening proof verification
- Multi-linear extension evaluations

After recursively verifying sumcheck, the verifier needs to compare the claimed evaluation of the sumcheck operand at a random point $r$: $S(r)$ to their own evaluation of the polynomial at $r$. Jolt does this with a combination of opening proofs over the constituent polynomials of $S$ and direct evaluations of the multi-linear extensions of those polynomials if they have sufficient structure. 

## Specifics
### Polynomial opening proof verification 
HyperKZG is currently the optimal commitment scheme for recursion due to the requirement of only 2-pairing operations per opening proof. Unfortunately non-native field arithmetic will always be expensive within a circuit. 

There are two options:
- Sumcheck and MLE evaluations using native arithmetic, pairing operations using non-native arithmetic
- Sumcheck and MLE evaluations using non-native arithmetic, pairing operations using native arithmetic

We believe the latter is more efficient albeit unergonomic. Some of the details are worked out in this paper [here](https://eprint.iacr.org/2023/961.pdf).

### Polynomial opening proof batching
Jolt requires tens of opening proofs across all constituent polynomials in all sumchecks. If we did these independently the verifier would be prohibitively expensive. Instead we [batch](https://jolt.a16zcrypto.com/background/batched-openings.html) all opening proofs for polynomials which share an evaluation point $r$. 

### verify_instruction_lookups
Instruction lookups does two sumchecks described in more detail [here](https://jolt.a16zcrypto.com/how/instruction_lookups.html). The first contains some complexity. The evaluation of the MLE of each of the instructions at the point $r$ spit out by sumcheck is computed directly by the verifier. The verifier is able to do this thanks to the property from Lasso that each table is SOS (decomposable). 

The `LassoSubtable` trait is implemented for all subtables. `LassoSubtable::evaluate_mle(r)` computes the MLE of each subtable. The `JoltInstruction` trait combines a series of underlying subtables. The MLEs of these subtables are combined to an instruction MLE via `JoltInstruction::combine_lookups(vals: &[F])`. Finally each of the instruction MLEs are combined into a VM-wide lookup MLE via `InstructionLookupsProof::combine_lookups(...)`.

The Groth16 verifier circuit would have to mimic this pattern. Implementing the MLE evaluation logic for each of the subtables, combination logic for each of the instructions, and combination logic to aggregate all instructions. It's possible that subtables / instructions will be added / removed in the future.

### verify_r1cs
[R1CS](https://jolt.a16zcrypto.com/how/r1cs_constraints.html)  is a modified Spartan instance which runs two sumchecks and a single opening proof.

There are two difficult MLEs to evaluate:
- $\widetilde{A}, \widetilde{B}, \widetilde{C}$  – evaluations of the R1CS coefficient
- $\widetilde{z}$ – evaluation of the witness vector 

> The sections below are under-described in the wiki. We'll flush these out shortly. Assume this step comes last.

For $\widetilde{A}, \widetilde{B}, \widetilde{C}$ we must leverage the uniformity to efficiently evaluate. This is under-described in the wiki, but we'll get to it ASAP.

The witness vector $z$ is comprised of all of the inputs to the R1CS circuit concatenated together in `trace_length`-sized chunks. All of these are committed independently and are checked via a batched opening proof.


# Engineering Suggestions
The Jolt codebase is rapidly undergoing improvements to reduce prover and verifier costs as well as simplify abstractions. As a result, it's recommended that each section above be built in modules that are convenient to rewire. Each part should be incrementally testable and adjustable. 

A concrete example of changes to expect: we currently require 5-10 opening proofs per Jolt execution. Even for HyperKZG this requires 10-20 pairings which is prohibitively expensive. We are working on fixing this via better [batching](https://jolt.a16zcrypto.com/background/batched-openings.html).

Suggested plan of attack:
1. Circuit for HyperKZG verification
2. Circuit for batched HyperKZG verification
3. Circuit for sumcheck verification
4. Circuit for instruction/subtable MLE evaluations
5. Circuit for Spartan verification
