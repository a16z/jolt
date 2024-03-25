# Lookups
Lasso is a lookup argument (equivalent to a SNARK for reads into a read-only memory). Lookup arguments allow the prover to convince the verifier that a (committed) set of values $Q$ is a subset of a lookup table $T$. Lasso is a special lookup argument with highly desirable asymptotic costs largely correlated to the number of queries $Q$ rather than the length of of the table $T$.

A conversational background on lookups can be found [here](https://a16zcrypto.com/posts/article/building-on-lasso-and-jolt/). In short: Lookups are great for zkVMs as they allow constant cost / developer complexity for the prover algorithm per VM instruction.

## Lasso
A detailed engineering overview of Lasso can be found [here](https://www.youtube.com/watch?v=iDcXj9Vx3zY).
