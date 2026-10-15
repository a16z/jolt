# CUDA backend

The `cuda` feature enables linkage. Select `JoltBackend::<Fr, CudaDoryScheme>::cuda()`
at runtime and bind a witness with `with_witness`. Use the unchanged
`jolt_prover::dory::prove` and `jolt_verifier::verify` entry points.

The implementation targets BN254/Dory. It follows the current optimized
backend's protocol, including cycle-first and address-first read-write
schedules, both trace storage orders, whole-bytecode commitments, advice, and
committed program images. Akita and other proof fields use upstream backends.

`build.rs` compiles the kernel translation unit with nvcc. `JOLT_NVCC` selects
the compiler; `JOLT_CUDA_ARCH` selects its architecture, defaulting to `native`.
For example, `JOLT_CUDA_ARCH=sm_89` builds for NVIDIA L4 without needing a device
at build time. Witness extraction additionally uses NVRTC on first device use.
Run device tests after changing CUDA sources or kernel signatures.

`JOLT_CUDA_GPUS` selects the requested device count, defaulting to one.
Process-wide state contains device contexts, modules, and reusable allocation
pools. Witness columns, phase carry, and proof caches live in the proof session.
Explicit CUDA streams allow session data to be retired by main's background
cleanup workers on the correct device.

## Acceleration coverage

The CUDA registry starts from the current optimized registry and overrides
ordinary-profile relation slots, witness commitments, RAM initial openings,
and joint openings. Trace execution and protocol orchestration remain on the
host. Device failures propagate as failed proving operations.

The following host work is intentional:

- Field-inline builds use main's optimized composed Spartan and bytecode
  read-RAF slots, field-register relations, field commitments, and field joint
  polynomials. Ordinary relations still use CUDA.
  Main requires full public bytecode preprocessing for field-inline proofs;
  committed programs are tested with the ordinary profile.
- ZK Dory openings and hiding finalization use main's Dory implementation.
  Clear Dory openings use the CUDA MSM, pairing, and reduction implementation.
- Dory setup, standalone host commitment/streaming APIs, verification, and
  commitment combination use main. CUDA witness commitments use the device
  tier-one interface.
- Program-image initial evaluation uses main's canonical sparse-image
  evaluator. Whole-bytecode coefficients and bounded lookup prefix tables
  are constructed with main's definitions before upload.
- Joint opening construction consumes the lazy host precommitted-table
  factory because it does not retain a device copy of those tables.
- Address-first RAM/register preparation builds sparse indices on the host;
  address and cycle arithmetic runs on GPU 0. Cycle-first read-write work,
  witness commitments, and other sharded kernels use the selected devices.

## Validation

Use `cargo nextest`; no device is a skip in developer tests, while a present
device whose kernels fail to load is a failure. For acceptance, run on a CUDA
machine and check that the requested devices are available.

```bash
cargo clippy -p jolt-kernels -p jolt-witness \
  --features jolt-kernels/cuda --all-targets -- -D warnings
cargo nextest run -p jolt-witness --features cuda backend::cuda --cargo-quiet
cargo nextest run -p jolt-kernels --features cuda \
  -E 'test(cuda::) | binary(cuda_e2e)' --cargo-quiet
JOLT_CUDA_GPUS=2 cargo nextest run -p jolt-kernels --features cuda \
  --test cuda_e2e --test-threads 1 --cargo-quiet
cargo nextest run -p jolt-kernels --features cuda,jolt-prover/zk \
  --test cuda_e2e --test-threads 1 --cargo-quiet
cargo nextest run -p jolt-kernels \
  --features cuda,jolt-prover/prover-fixtures,jolt-prover/field-inline \
  --test cuda_e2e --test-threads 1 --cargo-quiet
cargo nextest run -p jolt-kernels \
  --features cuda,jolt-prover/prover-fixtures,jolt-prover/field-inline,jolt-prover/zk \
  --test cuda_e2e --test-threads 1 --cargo-quiet
```

The fixture feature in field-inline commands enables main's host/tracer
feature forwarding. For allocation accounting, select `jolt-prover/allocative`
alongside `cuda` so all participating crates have matching visitation types.

Clear acceptance compares serialized CUDA proofs with the live optimized
backend using identical inputs and preprocessing, then verifies the CUDA
proof. ZK acceptance verifies both randomized proofs and rejects a modified
BlindFold payload. Kernel tests cover live reference parity and algebraic
properties, including equality challenges at zero and one.

See [the port validation record](PORT_VALIDATION.md) for the tested upstream
revision, hardware, and acceptance results.

## Profiling

The CUDA-owned example uses main's public host/prover/verifier APIs. It accepts
an explicit iteration count and reports the actual padded trace size, avoiding
assumptions about cycles per guest operation.

```bash
cargo run --release -p jolt-kernels --features cuda --example cuda_profile -- \
  --name fibonacci --iterations 4000 --backend optimized --repetitions 2
JOLT_CUDA_XFER_STATS=1 cargo run --release -p jolt-kernels --features cuda \
  --example cuda_profile -- --name fibonacci --iterations 4000 \
  --backend cuda --gpus 2 --repetitions 2
JOLT_CUDA_XFER_STATS=1 cargo run --release -p jolt-kernels --features cuda \
  --example cuda_profile -- --name sha2-chain --iterations 277 \
  --backend cuda --gpus 2 --repetitions 2
```

Use `--name sha3-chain` for Keccak and `--address-first` to force that read-write
schedule. Run the same workload and iteration count with `--backend optimized`
for the comparison. The example supports the ordinary clear-mode profile.

Each JSON output line records setup, complete proving, verification, proof
size, process peak RSS, sampled device memory, and transfer counts. Setup,
tracing, and guest compilation are outside the proving interval. First-use
witness compilation is included in the first proof; later repetitions reuse
device modules and allocation pools with a new proof session.

Device memory is sampled every 50 ms through verification and includes device
context/pool allocations and any other users of the GPU. Process peak RSS
includes setup. Transfer counters cover instrumented CUDA paths and count
submitted bytes, not measured PCIe throughput; they are not a complete system
transfer audit. Compare runs on otherwise idle hardware.
