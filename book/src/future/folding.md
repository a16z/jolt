The plan to implement folding is simple, with a (very) sketchy overview provided below. 

<OL>
  <LI> Veryifying Jolt proofs involves two procedures: verifiying sum-check proofs, and folding 
polynomial evaluation claims for committed polynomials. </LI>

<LI> Running Nova with BN254 as the primary curve, we can simply verify sum-check proofs natively. </LI>

<LI> For folding polynomial evaluation claims, we will use the technique from Hypernova to continually fold 
linear claims about committed vectors. This requires one scalar multiplication in the verifier circuit per claim folded.
We can offload that to Grumpkin via the technique from Cyclefold, or do it non-natively 
if the rest of the circuit is for verifying sum-check proofs dominates the size of the recursive constraint system regardless. </LI>

<LI> There are some nuisances as to how to incorporate offline memory-checking techniques with folding,
  and in particular to make sure that the state of the VM's memory at the end 
of shard $i$ matches the state of memory at the start of shard $i+1$. These are not terribly difficult to address,
and the details will be fully fleshed out in an upcoming e-print.</LI>

</OL>
