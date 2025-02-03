# Contributors to proof size

With HyperKZG commitments, Jolt's proof size (with no composition/recursion) today (or in the very near term) is 35KB-200KB.

Here is the breakdown of contributors to proof size:

<OL>
<LI> In Jolt, we commit to about 250 polynomials and produce just one HyperKZG evaluation proof. 
  It's one group element (about 32 bytes) per commitment, and an evaluation proof is a couple of dozen group group elements.
  So this is about 9 KBs total. 

  With some engineering headaches we could go below 9 KBs by committing to some of the 250 polynomials as a single, larger
  polynomial, but we do not currently plan to implement this.

  Most of the 250 polynomials come from the "subtables" used in Lasso, as each subtable arising in Lasso brings several
  committed polynomials. 

<LI> Jolt runs six grand product arguments in parallel: two for Lasso lookups into "instruction evaluation tables",
two for Lasso lookups into the RISC-V bytecode, and two for the Spice read-write-memory checking. 
  These six grand products dominate the proof size. How big the proofs are depends on which grand product argument is used.

  With pure Thaler13/GKR, the proof size is about 200KB (~30 KB per grand product). 

  With pure Quarks/Spartan, the proof size is only about 25KB. But this grand product argument is about 10x slower than 
  Thaler13.

  With a "hybrid" between the two that avoids a significant prover slowdown, the proof size is about 100 KB. 

  Eventually, with modest engineering effort, we can reduce these 6 grand products down to 3. 
  This would reduce proof size by another factor of 2 or so. See bottom of this page for further details.
  
<LI> Spartan for R1CS involves only two sum-checks (soon to be reduced to one) so contributes about 
  7 KBs to the proof (soon to fall to 3-4 KBs) </LI>
  
</OL>

# Details on grand products
The 6 grand product arguments run in Jolt stem from the following:
<UL>
  <LI> Two for instruction lookups (Lasso). </LI>
  <LI> Two for read/write memory (Spice). </LI>
  <LI> Two for bytecode lookups (Lasso). </LI>
</UL>

For each of the above, one grand product attests to the validity of reads and writes into the relevant memories,
and one attests to the validity of initialization of memory plus a final pass over memory.
The reason we do not run these grand products "together as one big grand product" is they are 
each potentially of different sizes,
and it is annoying (though possible) to "batch prove" differently-sized grand products together.
However, a relatively easy way to get down to 3 grand products is to set the memory size
in each of the three categories above to equal the number of reads/writes. This simply involves 
padding the memory with zeros to make it equal in size to 
the number of reads/writes into the memory (i.e., NUM_CYCLES). Doing this will not substantially increase
prover time.
