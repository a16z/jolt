# Binius
[Binius](https://eprint.iacr.org/2023/1784.pdf) was written by Ben Diamond and Jim Posen of Irreducible (fka Ulvetanna). It's a new commitment scheme that allows Jolt to use smaller fields more efficiently. 

The Binius paper also gives a number of sum-check-based polynomial IOPs for to be combined with the commitment scheme. For Jolt's purposes, these will yield extremely efficient protocols for proving hash evaluations with various standard hash functions like Keccak (important for recursing Jolt-with-Binius-commitment), and for the RISC-V addition and multiplication operations. 

Here is a brief summary of changes that will occur in Jolt in order to incorporate the Binius commitment scheme:

*Addition and multiplication instructions will be handled via gadgets/constraint-systems instead of lookups. We may want to handle XOR and LT via gadgets too for performance reasons. 

*Booleanity constraints in the R1CS can be omitted as that is guaranteed by the binding properties of the Binius commitment scheme.

*In offline memory checking (both Lasso for lookups a.k.a. read-only memory, and Spice for read-write memory), access counts need to be incremented. The Binius paper (Section 4.4) proposes to do this by sticking counts in the exponent of a generator of an appropriate multiplicative subgroup of $GF(2^{128})$, but it seems more efficient to plan to do the count increments via the addition gadget.

For Spice, the count written during a write operation is set to the maximum of a global counter and the count returned by the preceding read operation, plus 1. If the prover is honest, this written count will always equal the global counter plus 1. This max can be implemented by having the prover commit to the binary representation of $v=global_{count} - returned_{count}$ and using the addition gadget to confirm that $returned_{count} + v = global_{count}$. 

*Constraints that involve addition by IMM, the immediate value, will need to implement addition via gadget. 

*In the "collation" sum-check of Lasso that collates results of sub-table lookups into the result of the original big-table lookup, the algebraic implementation of "concatenate the results into one field element" changes. Over a large prime-order field, concatenating two $8$-bit values $a$ and $b$ corresponds to $2^8 \cdot a + b$. Over $GF(2^{128})$ constructed via towering, $2^8$ gets replaced with an appropriate element of the tower basis. The same change affects various R1CS constraints ensuring that certain committed values are a limb decomposition of another committed value. 

*Many performance optimizations become available within the various invocations of the sum-check protocol, leveraging that $GF(2^128)$ has small characteristic and most values being summed are in $GF(2^k)$ for small $k$. 

