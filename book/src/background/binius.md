# Binius
[Binius](https://eprint.iacr.org/2023/1784.pdf) was written by Ben Diamond and Jim Posen of Irreducible (fka Ulvetanna). It's a new commitment scheme that allows Jolt to use smaller fields more efficiently. 

The Binius paper also gives a number of sum-check-based polynomial IOPs for to be combined with the commitment scheme. For Jolt's purposes, these will yield extremely efficient protocols for proving hash evaluations with various standard hash functions like Keccak (important for recursing Jolt-with-Binius-commitment), and for the RISC-V addition and multiplication operations. 
