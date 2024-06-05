Below are known optimizations that we will be implementing in the coming weeks, and the anticipated reduction in prover costs (all percentages are relative to the Jolt prover's speed on initial public release in April 2024). 

- The way we implemented Lasso leads to grand product arguments for which about 90% of the factors being multiplied together are equal to 1. The current implementation explicitly stores all these values (assigning 256 bits to each value). Instead, we can store a densified representation of them (i.e., list only the values that are _not_ 1, and assume all the others are 1). This will speed up the prover, but more importantly it will reduce total prover space usage by a factor of about 8x. 

    Anticipated speedup: 8% of prover time (and a significant space reduction).

- When Spartan is applied to the R1CS arising in Jolt, the prover’s work in the first round involves computation over 64-bit integers, not arbitrary field elements. The implementation does not yet take advantage of this: each multiplication in the first round is currently implemented via a 256-bit Montgomery multiplication rather than a primitive 64-bit multiplication. This is about half of the prover’s total work in this sum-check invocation. (The same optimization applies for computing the three matrix-vector multiplications needed to compute Az, Bz, and Cz before beginning this first sum-check).  

    Anticipated speedup: 4% of overall prover time. <check>

- The optimization described in Section 3.2 of [this paper](https://eprint.iacr.org/2024/108.pdf) by Angus Gruen applies to a subset of the invocations of the sum-check protocol in Jolt, including the first of two sum-checks in Spartan and all sum-checks in Grand Product arguments within memory-checking procedures (Lasso, Spice). 

    Anticipated speedup: 3% of total prover time.

- Switching the commitment scheme from Hyrax to one with much smaller commitments (e.g., HyperKZG, Zeromorph) will not only shorten the proofs, but also save the prover the time of serializing and hashing the commitments for Fiat-Shamir. See [this github issue](https://github.com/a16z/jolt/issues/208).

    Anticipated speedup: 3% of total prover time.

- Make it fast to commit to slightly negative values (one group op per value) just as it's fast for small positive values. 

    Anticipated speedup: 2% of total prover time.
  
- In the first sum-check in Spartan, the prover pre-computes a table of evaluations of (the multilinear extension of) the equality function eq(a, b) with the first vector a fixed to a random value. Leaving a few variables off of b and handling them differently will reduce the cost of building this table to negligible.

    Anticipated speedup: 1%-2% of total prover time. 

- On reads to registers or RAM, the value written back to the memory cell by the memory-checking procedure is committed separately from the value returned by the read, and an R1CS constraint is included to force equality. Really, a single value can be committed and the constraint omitted.  

    Anticipated speedup: 1% of total prover time. 

- SP1 implements word-addressable memory (the CPU has to read an entire 64-bit word of memory at once). Jolt currently implements byte-addressable memory (the RISC-V CPU is allowed to read one byte at a time, as required by the RISC-V specification). 

    Most benchmarks get compiled into RISC-V programs that mostly read entire words at once. Switching to word-addressable memory will improve Jolt’s speed on these benchmarks by 5%. 

<b> Total anticipated prover time reduction</b>: 20%-30%. 
