# Improvements complete or in progress since the initial release of Jolt in April 2024

There are many tasks described in the various files in the "roadmap/future" section of the wiki. 
Let's use this file as the primary means of tracking what is done or in progress. Anything
not mentioned in this file is presumably not yet started (or barely started). 

## Functionality improvements

* Support for stdlib

* Support for M-extension.

* In progress: on-chain verifier (Solidity). 

## Verifier cost improvements

* Add support for HyperKZG commitment

* Add support for Quarks/Spartan grand product argument and hybrid grand product 
(which achieves most of the verifier benefits of Quarks without a significant hit to prover time).

  * Note: the cost of Quarks/Spartan and hybrid grand products (i.e., time to commit to partial products) were improved subsequent to initial implementation, via [this PR](https://github.com/a16z/jolt/pull/473). 

* In progress: change how we are batching grand products, treating them all laid side-by-side as one giant circuit. This
will reduce proof size by up to 200KB.

* Reduce the number of polynomial evaluation proofs from 7-10 down to 1 (achieved in [this PR](https://github.com/a16z/jolt/pull/453)) 

## Prover cost improvements (all in progress)

* Eliminate cost of pre-computed tables of eq evaluations for each sum-check,
as per [Dao-Thaler](https://eprint.iacr.org/2024/1210).

* (Nearly) [eliminate](https://github.com/a16z/jolt/issues/347) second sum-check instance in Spartan.

* Implement the sum-check prover optimization from Section 3 of Angus Gruen's [paper](https://eprint.iacr.org/2024/108).

* Implement the sum-check prover optimizations from [Bagad-Domb-Thaler](https://eprint.iacr.org/2024/1046), which actually apply whenever small values are being summed, even if those values reside in a big (e.g., 256-bit) field. This captures Spartan as applied in Jolt. Thanks to Lev Soukhanov for this observation.

* Replace byte-addressable memory with word-addressable memory. This is actually [implemented](https://github.com/a16z/jolt/pull/412), but the above speedups to Spartan proving will need to be implemented before it yields an overall performance improvement. 

* AVX-512 speedups (see [here](https://github.com/a16z/vectorized-fields) for a detailed description of in-progress efforts).

* GPU integration.

* Prover space control via folding (see [here](https://jolt.a16zcrypto.com/future/folding.html) for a sketchy overview of how this will be implemented).

* Eliminate some unnecessary lookups identified via formal verification efforts. 
