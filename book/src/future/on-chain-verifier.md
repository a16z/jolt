# Estimated costs of on-chain verifier with no proof composition

We expect proof storage and verification to cost about 2 million gas, with proofs that are a couple of dozen KBs in size.
A crude and speculative explanation of where this estimate comes from is below. 

## Details

### Costs due to processing commitments 
Once we fully implement batching of polynomial evaluation proofs, Jolt will only require one HyperKZG opening. 
That's something like $7 + \log(N) \approx 40$ (conservatively) scalar multiplications, and two pairings, where
$N$ is the number of RISC-V cycles being proven. 

On top of that, the verifier is going to do about 80 scalar multiplications for homomorphically combining various commitments.

Let's conservatively ballpark this at 150 scalar multiplications and two pairings in total for the verifier. Each scalar
multiplication [costs](https://www.evm.codes/precompiled) 6,000 gas and each pairing costs 45,000 gas. That's about 1 million gas total
due to scalar multiplications and pairings. 

### Costs due to sum-check verification and multilinear extension evaluations
The Jolt verifier will invoke the sum-check protocol about ten times (as little as 5 if we fully batch these sum-checks,
meaning run as many as possible in parallel combined with standard random-linear-combination techniques,
but let's say ten as a conservative estimate since it will take us an annoying amount 
of engineering effort to maximally batch the sum-checks). 

One of these sum-checks is from applying Spartan for R1CS, 5 or so are from using the hybrid grand product argument
(hybrid between Thaler13/GKR and Quarks/Spartan, which gives a nice balance between prover time and verifier costs), one from batching polynomial evaluation queries, and
one from doing an input/output-consistency check, and one from the "collation" sum-check in the Lasso lookup argument. 

The total amount of data sent in these sum-checks (plus HyperKZG commitments and evaluation proof) should be a few dozen KBs.
Indeed, each sum-check should be roughly 40 rounds, with at most three field elements sent per round (except for the collation
sum-check, as the polynomial involved there has degree 6 or so in each variable). 
That's $40 \cdot 3 \cdot 32=3840$ bytes of data in each invocation of sum-check, and another 7 KBs or so for HyperKZG commitments and the evaluation proof. In total, that's perhaps 40 KBs of data. 

Sum-check verification is simply field work, let's say ~100 field ops total per sum-check (very crude estimate), plus associated Fiat-Shamir hashing. On top of all of that, the Jolt verifier does MLE evaluations for all the "subtables" used in Lasso.

All told we can ballpark the total number of field ops done by the verifier as about $2,000$, and Fiat-Shamir-hashing at most 40 KBs of data.

The gas costs of this hashing, plus field work, plus storing the entire proof in CALLDATA, should be under $1$ million. 

Combined with the ~1 million gas to perform scalar multiplications and pairings, we obtain an estimate of between 1.3 million and 2 million gas in total. 

