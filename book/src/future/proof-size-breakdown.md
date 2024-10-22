With HyperKZG commitments, Jolt's proof size today (or in the very near term) is 35KB-200KB.

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

  Eventually, with modest engineering effort, we can reduce these 6 grand products down to 3 (or even to 1, though the
  engineering pain of going that low would be higher). This would reduce proof size by another factor of 2 or so.

<LI> Spartan for R1CS involves only two sum-checks (soon to be reduced to one) so contributes about 
  7 KBs to the proof (soon to fall to 3-4 KBs) </LI>

</LI>
</LI>
  
</OL>
