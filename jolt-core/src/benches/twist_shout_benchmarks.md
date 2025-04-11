g# Twist and Shout preliminary benchmarks

In the benchmark results below, Spice and Lasso results are obtained by running Jolt end-to-end on an execution trace of length $$T\approx 2^{23}$$ from the `sha2-chain` example guest program.

Because Twist and Shout are not fully integrated into Jolt yet, they are instead run on a randomly generated trace of length $$T=2^{23}$$.

Benchmark code can be found in [`bench.rs`](./bench.rs).

Results reported do not include costs associated with polynomial commitment schemes (i.e. proof sizes do not include commitment size, and proof times do not include commitment or opening proof time).

All benchmarks were run on an M3 Macbook Pro with 128GB RAM.

## Bytecode lookups

### Prover time

| **Shout**    | **Lasso**   | **Improvement**  |
| ------------ | ----------- | ---------------- |
| 250 ms       | 610 ms      | 2.4x             |

### Proof size

| **Shout**    | **Lasso**   | **Improvement**  |
| ------------ | ----------- | ---------------- |
| 4.5 kB       | 38.9 kB     | 8.6x             |


## Read-write memory

Read-write memory is currently implemented as a single offline memory-checking instance proven using Spice.
With Twist, read-write memory will be proven as two offline memory-checking instances (one for registers and one for RAM).
Thus in the benchmark reports below, we run two invocations of Twist: one with a memory size of K = 64 (for registers)
and one with a memory size of K = 8192 (the amount of RAM used by the `sha2-chain` example).

Note that the current Twist implementation does not include the "one-hot" sumchecks described in Figure 8; we expect them to yield a minor increase in prover time and proof size.

### Prover time

We implement the "local" Twist algorithm described in Section 8.2.2 of the Twist+Shout paper, the performance of which improves with locality of memory.
To simulate locality in our Twist benchmarks, we sample memory addresses from a [Zipf](https://mathworld.wolfram.com/ZipfDistribution.html) distribution, varying the skew parameter.
Note that `Zipf(0)` is equivalent to the uniform distribution, while `Zipf(1)` simulates some degree of locality.

| **Twist, Zipf(0)**   | **Twist, Zipf(1)**    | **Lasso**   | **Improvement**  |
| -------------------- | --------------------- | ----------- | ---------------- |
| 1.6 + 3.0 = 4.6s     | 1.4 + 2.4 = 3.8s      | 6.1s        | 1.3-1.6x         |

### Proof size

| **Twist**                   | **Lasso**   | **Improvement**  |
| --------------------------- | ----------- | ---------------- |
| 4.8 kB + 5.5 kB = 10.3 kB   | 87.1 kB     | 8.5x             |

## Instruction lookups

Note that the current sparse-dense Shout implementation does not include the "one-hot" sumchecks described in Figure 8; we expect them to yield a minor increase in prover time and proof size.

### Prover time

| **Shout**    | **Lasso**   | **Improvement**  |
| ------------ | ----------- | ---------------- |
| 1.6s         | 15.9s       | 9.9x             |


### Proof size

| **Shout**    | **Lasso**   | **Improvement**  |
| ------------ | ----------- | ---------------- |
| 9 kB         | 91.3 kB     | 10.1x            |
