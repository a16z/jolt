# Technical plan

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

# Space cost estimates

If we shard the VM execution into chunks of $2^{20}$ RISC-V cycles each, then naively implemented we expect Jolt proving to require about 10 GBs of space. The bottleneck here is Spartan proving. The prover space cost for proving all the grand products in Jolt is less than that of Spartan proving, and the prover can compute the Spartan proof and more or less "wipe" memory before computing the grand product proofs. 

## Details of $10$ GB estimate

What dominates the space cost of the Spartan prover? Simply storing the three vectors $a=Az$, $b=Bz$, and $c=Cz$, where $A, B, C$ are the R1CS constraint matrices. In Jolt's R1CS instance there are fewer than $128$ constraints per cycle, and there is one entry of each of $a, b, c$ per constraint. 

Naively implemented, the Spartan prove stores $32$ bytes (i.e., $256$ bits) per entry of $a, b, c$. This is actually overkill as the entries of $a, b, c$ are all $32$-bit values, but it's simplest to ignore this and treat them as arbitrary field elements. 

Hence, the number of bytes required to store $a, b, c$ is 
$2^{20} \cdot 100 \cdot 32 \cdot 3$ bytes, which is about $10$ billion bytes, a.k. a $10$ GBs. 

The extra space overhead of taking the Jolt verifier (minus polynomial evaluation proofs, which don't need to be provided or verified when using folding to accumulate such claims) will be well under $500$ MBs. 

## Anticipated proving speed

Even with naive approaches to folding (e.g., using Nova rather than HyperNova, and without fully optimizing the Jolt proof size, meaning not batching all the grand product proofs to the maximum extent possible, so that Jolt proofs are about $100$ KBs rather than $50$ KBs)), we expect prover time of "Jolt with folding" to be close to the prover time cost of "Jolt applied monolithically" to a single shard. 

That is, the extra time spent by the Jolt-with-folding prover on recursively proving verification of Jolt proofs will be modest, and partially offset by the time savings of not having to compute HyperKZG evaluation proofs (except for a single HyperKZG evaluation proof computed for the 'final folded instance'). 

## Going below 10 GBs via smaller chunks

Projects focused on client-side proving may want space costs of 2 GBs (or even lower). One way to achieve that space bound is to use smaller chunks, of $2^{17}$ or $2^{18}$ cycles each rather than $2^{20}$. This is not much smaller than the chunk sizes used by STARK-based zkVMs like SP1 and RISC Zero. With chunks of this size, the Jolt-with-folding prover (again, naively implemented) will be somewhat slower per cycle than "monolithic Jolt", perhaps by a factor of up to two. 

This time overhead can be reduced with additional engineering/optimizations. For example: 

<UL>

<LI> Maximally batching grand product proofs should cut the "time spent on recursive proving" by a factor of $2$, and at least for the first round of Spartan's sum-check. </LI>
  
<LI> We could store only $32$ bits per entry of $a, b, c$ rather than 256, which could save another factor of two in space cost. Also, prover <i> speed </i>optimizations for applying sum-check to small values (described in [Bagad-Domb-Thaler](https://eprint.iacr.org/2024/1046)) can also lead to additional space improvements for Spartan. At some point, the space cost of computing grand product proofs would dominate the space cost of Spartan proving and it won't be worthwhile to cut Spartan proving space further. </LI>

<LI> Using HyperNova in place of Nova will avoid committing to cross terms consisting of random field elements, which would cut the time cost of recursive proving by an order of magnitude. But HyperNova implementations are not yet as mature as Nova itself. </LI>

</UL>
