I would like to introduce the following optimization to the e2e Jolt proving and verifier pipeline: 

Currently, jolt uses (by default) Dory for the PCS for the opening proof. Dory is good in generality but is not the most efficient PCS for small traces (2^20 trace length and below).

Instead, we want to switch to use HyperKZG when the trace length is leq 2^20. 

It should be almost semantically identical to what Dory is doing - hyperkzg has all the same properties of homomorphism and whatnot. There are some differences to flag which you will need to reason about:
- We don't have / want a streaming implementation of HyperKzg. Linear memory is acceptable because the trace length is small. 
- HyperKZG requires a structured reference string rather than the URS that dory uses (Transparent).
- We have a HyperKZG impl in the Jolt repo, but it might not be up to date with current code since it hasn't been used for a while and may have fallen behind upstream, so we may need to make adjustments, optimizations, implement traits / fix them appropiately, add some tests, add proper file SRS support, etc.

The end result is that it should runtime switch PCS depending on the case. The success criteria for this change is that you should have a test that specifially checks this condition and all other tests pass. It is an involved and complicated change so I emphasize the following:
- If you are unsure about anything regarding the integraiton, ask. DO no guess. I am an expert and here to help you.
- DO not leave anything as unimmplemented, "for now", "placeholder" or such. If you don't know what to do, just stop and ask and I will guide you.
- Strive for simple abstractions / implementations and only comments that are critical / nuanced to what the code is doing.

Refer back to this spec for guidance!
