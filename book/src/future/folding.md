# Technical plan

The plan to implement folding is simple, with a (very) sketchy overview provided below. 

<OL>
  <LI> Verifying Jolt proofs involves two procedures: verifying sum-check proofs, and folding 
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

Note that this plan does not require "non-uniform folding". The fact that there are many different primitive RISC-V 
instructions is handled by "monolithic Jolt". Folding is merely applied to accumluate many copies of the same claim,
namely that a Jolt proof (minus any HyperKZG evaluation proof) was correctly verified. 

# Space cost estimates

If we shard the VM execution into chunks of $2^{20}$ RISC-V cycles each, then naively implemented we expect Jolt proving to require about 10 GBs of space. The bottleneck here is Spartan proving. The prover space cost for proving all the grand products in Jolt is less than that of Spartan proving, and the prover can compute the Spartan proof and more or less "wipe" memory before computing the grand product proofs. 

## Details of $10$ GB estimate

What dominates the space cost of the Spartan prover? Simply storing the three vectors $a=Az$, $b=Bz$, and $c=Cz$, where $A, B, C$ are the R1CS constraint matrices. In Jolt's R1CS instance there are fewer than $128$ constraints per cycle, and there is one entry of each of $a, b, c$ per constraint. 

Naively implemented, the Spartan prove stores $32$ bytes (i.e., $256$ bits) per entry of $a, b, c$. This is actually overkill as the entries of $a, b, c$ are all $32$-bit values, but it's simplest to ignore this and treat them as arbitrary field elements. 

Hence, the number of bytes required to store $a, b, c$ is 
$2^{20} \cdot 100 \cdot 32 \cdot 3$ bytes, which is about $10$ billion bytes, aka $10$ GBs. 

The prover also has to store the witness vector $z$, which has a similar number of entries as $a, b, c$ (about $80 \cdot 2^{20}$ of them), but $z$'s entries are all 32-bit data types and it's pretty easy to actually store these as $32$ bits each throughout the entire Spartan protocol. So $z$ should only contribute a few hundred MBs to prover space if the shard size is $2^{20}$. 

The extra space overhead of taking the Jolt verifier (minus polynomial evaluation proofs, which don't need to be provided or verified when using folding to accumulate such claims) and turning it into constraints as required to apply a recursive folding scheme will be well under $500$ MBs. 

## Anticipated proving speed

Even with naive approaches to folding (e.g., using Nova rather than HyperNova, and without fully optimizing the Jolt proof size, meaning not batching all the grand product proofs to the maximum extent possible, so that Jolt proofs are about $100$ KBs rather than $50$ KBs)), we expect prover time of "Jolt with folding" to be close to the prover time cost of "Jolt applied monolithically" to a single shard. 

In more detail, the extra time spent by the Jolt-with-folding prover on recursively proving verification of Jolt proofs will be modest. With folding, for shards of $2^{20}$ cycles, the "extra work" the prover does to fold/recurse is simply commit to roughly one million extra field elements. But the Jolt prover has to commit to about 80 million (small) field elements anyway just to prove the $2^{20}$ cycles. So committing to the one million field elements doesn't have a huge impact on total prover time. 

It is true that the 1 million extra field elements are random, and so roughly equivalent in terms of commitment time
to 10 million "small" field elements. This still represents at most a ~13% increase in commitment costs for the Jolt prover. 

In fact, for large enough shard sizes (perhaps size $2^{23}$ or so), the Jolt prover may be <i>faster</i> with folding than without it.
This is because, with folding, the prover does not have to compute any HyperKZG evaluation proofs, except for a single HyperKZG evaluation proof computed for the 'final folded instance'. The savings from HyperKZG evaluation proofs 
may more than offset the extra work of committing to roughly one million extra field elements per shard. 

## Going below 10 GBs via smaller chunks

Projects focused on client-side proving may want space costs of 2 GBs (or even lower, as browsers often limit a single tab to 1 GB of space, [especially](https://www.tigren.com/blog/progressive-web-app-limitations/) on [phones](https://web.dev/articles/storage-for-the-web)). One way to achieve that space bound is to use smaller chunks, of $2^{17}$ or $2^{18}$ cycles each rather than $2^{20}$. This is slightly smaller than the chunk sizes used by STARK-based zkVMs like SP1 and RISC Zero. With chunks of this size, the Jolt-with-folding prover (again, naively implemented) will be somewhat slower per cycle than "monolithic Jolt", perhaps by a factor of up to two or three. 

This time overhead can be reduced with additional engineering/optimizations. For example: 

<UL>

<LI> Maximally batching grand product proofs should reduce Jolt proof size (and hence cut the time spent on recursive proving) by a factor of $2$. </LI>
  
<LI> For at least the first round of Spartan's sum-check, we could store only $32$ bits per entry of $a, b, c$ rather than 256, which could save another factor of two in space cost. Also, prover <i>speed</i> optimizations for applying sum-check to small values described in Bagad-Domb-Thaler can also lead to additional space improvements for Spartan. At some point, the space cost of computing grand product proofs would dominate the space cost of Spartan proving and it won't be worthwhile to cut Spartan proving space further. </LI>

<LI> We can lower the size of Jolt proofs, and hence the time spent on recursive proving, by a factor of 2x-4x by 
switching to the grand product argument from Quarks/Spartan. For details, see https://jolt.a16zcrypto.com/future/proof-size-breakdown.html. 
  
  This would reduce the amount of extra field elements committed for "folding/recursion" from about 1 million down to a few hundred thousand or even lower. It would increase prover time "outside of recursion"
since the Quarks/Spartan prover commits to random field elements, but for small shard sizes (i.e., very small prover space, around 1GB or so) it may lead to a faster prover overall.</LI>
</UL>

If all of the above optimizations are implemented, Jolt-with-folding prover space could be as low as 1 GB-2 GB with the prover only 30%-60% slower than "monolithic Jolt" (i.e., Jolt with no recursion or space control for the prover at all).  
