The plan to implement folding is simple, with a (very) sketchy overview provided below. 

<OL>
  <LI> If we look at Jolt's proofs, they can be verified by verifying sum-check proofs and folding 
polynomial evaluation claims for committed polynomials. </LI>

<LI> If Nova runs with BN254 as the primary curve, we will simply verify sum-check proofs natively. </LI>

<LI> For folding polynomial evaluation claims, we will use the technique from Hypernova to continually fold 
linear claims about committed vectors. This requires one scalar multiplication in the verifier circuit per claim folded
We can offload that to Grumpkin via the technique from Cyclefold trick, or do it non-natively 
if the rest of the circuit is for verifying sumcheck proofs domainates the size of the recursive constraint system regardless. </LI>

<LI> There are some nuisances as to how to incorporate offline memory-checking techniques with folding,
  and in particular to make sure that the state of the VM's memory at the end 
of shard $i$ matches the state of memory at the start of shard $i+1$. These are not terribly difficult to address,
and the details will be fully fleshed out in an upcoming e-print.</LI>

</OL>
