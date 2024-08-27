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

  * Still to do here: [Optimize](https://github.com/a16z/jolt/issues/444) to how the prover commits to partial products. 

* In progress: change how we are batching grand products, treating them all laid side-by-side as one giant circuit. This
will reduce proof size by up to 200KB.

* In progress: reduce the number of polynomial evaluation proofs from 7-10 down to 1. 

## Prover cost improvements (all in progress)

* Eliminate cost of pre-computed tables of eq evaluations for each sum-check,
as per [Dao-Thaler](https://eprint.iacr.org/2024/1210).

* (Nearly) [eliminate](https://github.com/a16z/jolt/issues/347) second sum-check instance in Spartan.

* Implement the sum-check prover optimization from Section 3 of Angus Gruen's [paper](https://eprint.iacr.org/2024/108).

* AVX-512 speedups.

* GPU integration. 
