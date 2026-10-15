# CUDA port validation

Integration target: upstream main `8bbb2357890928d8f83b00ca542879077c4d3d28`.
CUDA starting point: `82e18d694e89c9f1d46fcd9a553ad28893a39479`.
Validated on 2026-10-15.

The final tree takes upstream's CPU, reference, optimized, prover, verifier,
trace, and protocol implementations unchanged. Differences are restricted to
CUDA source, CUDA tests/tools, narrow CUDA adapters in `jolt-dory` and
`jolt-witness`, and the dependencies needed by those additions. Existing
upstream dependency versions and pins are preserved.

## Correctness

The CUDA suites ran on two NVIDIA L4 GPUs, each reporting 23,034 MiB of total
device memory. These were device runs, not absent-device skips.

| Check | Result |
| --- | --- |
| One-GPU CUDA kernel cases | 364 passed |
| Ordinary proof matrix, including combined and isolated advice | 16 passed on each of one and two GPUs |
| Two-GPU ZK proof matrix | 15 passed |
| Two-GPU field-inline clear proof matrix | 8 passed |
| Two-GPU field-inline ZK proof matrix | 7 passed |
| Complete witness suite, including CUDA extraction | 68 passed |
| Upstream clear prover suite | 29 passed |
| Upstream ZK prover suite | 30 passed |
| Upstream standard muldiv verifier fixture | 1 passed |

Clear proofs are byte-identical to the live optimized backend with the same
inputs and preprocessing, and pass the unchanged verifier. ZK proofs from
both backends verify; changing the BlindFold payload is rejected.

Coverage includes the ordinary guest matrix, both read-write phase orders,
both commitment storage orders, whole-bytecode and program-image
commitments, combined and isolated trusted/untrusted advice, and reuse of a
backend registry with distinct witnesses. Primitive tests include current
witness extractors, group operations, lookup semantics, and sumcheck equality
challenges at zero and one.

Strict all-target CUDA Clippy passed in ordinary, ZK, field-inline ZK, and
allocation-tracking builds. Both workspace Clippy commands (`host` and
`host,zk`) passed. Upstream Akita passed strict library Clippy. Formatting and
the repository style rules passed against the integration target.

See [the backend guide](README.md) for reproducible commands and the explicit
inventory of operations that still use upstream's CPU implementation.

## Measurement environment

- Two Intel Xeon Silver 4310 sockets, 24 physical cores / 48 logical CPUs.
- Two NVIDIA L4 GPUs; driver 610.43.02.
- CUDA toolkit 13.1.115; native `sm_89` kernels.
- Rust 1.95.0; the workspace release profile, including fat LTO.

The maintained `cuda_profile` example measures optimized and CUDA proofs
through the same public host/prover/verifier APIs. Every measured proof is
verified. Each process makes two proofs, reusing the backend registry and
preprocessing with a new proof session.

Proving time excludes guest compilation, tracing, and preprocessing. The
first proof includes first-use witness compilation. Memory and transfer
measurements have the limitations documented in the backend guide.

## Release measurements

All 30 measured proofs verified successfully. Every workload reached the
requested padded trace size on all three backends. The input iteration
counts below reproduce those sizes with this upstream revision.

Proving wall times are seconds, shown as **first / second** proof. These
are two observations per configuration, not statistical confidence
intervals. The second proof reuses initialized modules and allocation pools.

| Workload | Rows | Iterations | Optimized CPU | 1 L4 | 2 L4s |
| --- | --- | ---: | ---: | ---: | ---: |
| fibonacci | 2^16 | 4000 | 2.091 / 2.078 | 1.062 / 0.902 | 1.266 / 1.200 |
| sha2-chain | 2^20 | 277 | 6.911 / 6.813 | 1.609 / 1.560 | 1.664 / 1.727 |
| sha3-chain | 2^20 | 218 | 7.212 / 7.172 | 1.643 / 1.583 | 1.791 / 1.795 |
| sha2-chain | 2^22 | 1111 | 15.753 / 15.707 | 3.131 / 3.128 | 3.011 / 2.855 |
| sha3-chain | 2^22 | 872 | 16.666 / 16.704 | 3.319 / 3.318 | 2.990 / 2.809 |

At 2^22 rows, both hash workloads benefited from two GPUs. At the smaller
sizes, one GPU was faster than two in these measurements. All CUDA
configurations were faster than optimized CPU for the measured proving
interval; this comparison does not include tracing or preprocessing.

The table below uses the second proof. Host RSS is the process high-water
mark through that proof, including setup and the first proof. Device values
are sampled peaks for each GPU separately; their sum need not be a
simultaneous peak. Transfer counts cover instrumented paths only.

| Workload / rows | Backend | Host RSS (GiB) | Device peaks (GiB) | H2D (MiB) | D2H (MiB) | D2D (MiB) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| fibonacci / 2^16 | CPU | 0.34 | — | — | — | — |
| fibonacci / 2^16 | 1 GPU | 0.23 | 0.64 | 21.7 | 6.1 | 9.6 |
| fibonacci / 2^16 | 2 GPUs | 0.30 | 0.64, 0.47 | 32.6 | 23.0 | 8.9 |
| sha2-chain / 2^20 | CPU | 0.67 | — | — | — | — |
| sha2-chain / 2^20 | 1 GPU | 0.39 | 1.33 | 58.7 | 29.9 | 124.0 |
| sha2-chain / 2^20 | 2 GPUs | 0.44 | 0.92, 0.76 | 80.2 | 56.3 | 123.5 |
| sha3-chain / 2^20 | CPU | 0.72 | — | — | — | — |
| sha3-chain / 2^20 | 1 GPU | 0.40 | 1.17 | 58.7 | 29.9 | 124.0 |
| sha3-chain / 2^20 | 2 GPUs | 0.44 | 0.95, 0.76 | 77.5 | 51.8 | 123.5 |
| sha2-chain / 2^22 | CPU | 1.58 | — | — | — | — |
| sha2-chain / 2^22 | 1 GPU | 0.87 | 3.79 | 106.4 | 75.3 | 470.4 |
| sha2-chain / 2^22 | 2 GPUs | 0.74 | 2.01, 1.91 | 138.4 | 108.3 | 470.5 |
| sha3-chain / 2^22 | CPU | 1.63 | — | — | — | — |
| sha3-chain / 2^22 | 1 GPU | 0.90 | 3.61 | 106.3 | 75.3 | 470.4 |
| sha3-chain / 2^22 | 2 GPUs | 0.75 | 2.04, 1.91 | 135.4 | 103.8 | 470.5 |

Verification took 0.124–0.160 seconds. Serialized proof sizes
agreed across the three backends for each workload: 77,894 bytes at 2^16,
87,106 bytes at 2^20, and 91,438 bytes at 2^22. The acceptance suite above
checks full byte equality; this profiling executable checks verification
and reports serialized size.

Reproduce any row with the maintained example, selecting the matching
backend and iteration count:

```bash
JOLT_CUDA_XFER_STATS=1 cargo run --locked --release -p jolt-kernels \
  --features cuda --example cuda_profile -- --name sha2-chain \
  --iterations 1111 --backend cuda --gpus 2 --repetitions 2
```
