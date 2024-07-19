# Groth16 Recursion
Jolt's verifier today is expensive. We estimate 1-5 million gas to verify within the EVM. Better batching techniques on opening proofs can bring this down 5-10x, but it will remain expensive. Further, the short-term [continuations plan](https://jolt.a16zcrypto.com/future/continuations.html) causes a linear blowup depending on the count of monolithic trace chunks proven.

To solve these two issues we're aiming to add a configuration option to the Jolt prover with a post processing step, which creates a Groth16 proof of the Jolt verifier for constant proof size / cost (~280k gas on EVM) regardless of continuation chunk count or opening proof cost. This technique is industry standard.

## Strategy
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