# Continuations via Chunking
Today Jolt is a monolithic SNARK. RISC-V traces cannot be broken up, they must be proven monolithically or not at all. As a result, Jolt has a fixed maximum trace length that can be proved which is a function of the available RAM on the prover machine. Long term we'd like to solve this by implementing a streaming version of Jolt's prover such that prover RAM usage is tunable with minimal performance loss. *TODO(sragss): Streaming prover link*

Short term we're going to solve this via monolithic chunking. The plan: Take a trace of length $N$ split it into $M$ chunks of size $N/M$ and prove each independently. $N/M$ is a function of the max RAM available to the prover. 

For the direct on-chain verifier there will be a cost linear in $M$ to verify. If we wrap this in Groth16 our cost will become constant but the Groth16 prover time will be linear in $M$. We believe this short-term trade-off is worthwhile for usability until we can implement the (more complicated) streaming algorithms. 

## Specifics
A generic config parameter will be added to the `Jolt` struct called `ContinuationConfig`. At the highest level, before calling `Jolt::prove` the trace will be split into `M` chunks. `Jolt::prove` will be called on each and return `RAM_final` which can be fed into `RAM_init` during the next iteration of `Jolt::prove`. The [output zerocheck](https://jolt.a16zcrypto.com/how/read_write_memory.html#ouputs-and-panic) will only be run for the final chunk. 

