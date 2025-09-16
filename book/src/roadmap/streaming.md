# Streaming

Other zkVMs, when applied ``monolithically'' (meaning in non-recursive fashion), have prover memory requirements that grow linearly with the number of CPU cycles being proven. This translates to many GBs of space per million cycles proved. 

The way other zkVMs address this issue is a technique commonly referred to as continuations. One breaks the CPU execution into ``shards'' (often consisting of about a million cycles each), prove each shard (more or less) independently, and then recursively aggregates the proofs. This leads to many complications and performance overheads.

Jolt is amenable to a different approach that we call the streaming prover. For arbitrarily long CPU executions, the Jolt prover can keep its memory usage bounded to a few GBs (plus the space required to simply run the program itself) without SNARK recursion. The time overhead of the streaming Jolt prover relative to the linear-space Jolt prover will be minimal, certainly well below a factor of 2x. 

At a high level, the streaming Jolt prover works as follows. In each round of any invocation of the sum-check protocol within Jolt, each step of the CPU execution contributes independently to the prover's message in that round of sum-check. Hence, the prover can compute the required message in that round incrementally in small space as it runs the CPU from start to finish. This observation is very old (see <a href="https://arxiv.org/abs/1109.6882"> [CTY11] </a> and <a href="https://eprint.iacr.org/2014/846"> [Clover 2014] </a>). 

The above sketch suggests that the streaming Jolt prover would have to perform a linear (in the cycle count) number of field operations per sum-check round. This is not at all the case in Jolt, for myriad reasons, some of which are sketched below (see <a href="https://eprint.iacr.org/2025/611"> [NTZ25] </a> for details). 

First, Twist and Shout are naturally streaming, in the sense that while they require up to $128 + \log(T)$ rounds of sum-check where $T$ is the cycle count (here, 128 arises when primitive instructions take 128 bits of input), the first 128 rounds can be completed by the prover with just a few passes in total over the execution trace and a low-order amount of prover work. 

Second, various optimizations to the Spartan-in-Jolt prover time are \emph{also} directly compatible with a streaming prover in the sense above: stream-ifying this prover algorithm does not actually lead to a significant slowdown compared to the linear-space version.

Third, recent improvements to streaming sum-check proving ensure that each pass the prover makes over the execution trace actually suffices to get the prover through many rounds rather than a single round (see <a href="https://eprint.iacr.org/2025/1473"> [BCFFMMZ25] </a>). 

Fourth, while the first "run" of the CPU is typically down serially, subsequent runs performed by the prover can be done in parallel. 

Today, the Jolt prover is not streaming, using under 2 GB of memory per million cycles proved. Work to "stream-ify" the prover is underway. For all of the reasons above (and others) we expect the streaming prover to be almost as fast as the current linear-space implementation. 
