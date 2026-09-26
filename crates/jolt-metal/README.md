# jolt-metal

Prover-only Metal support for `jolt-field`: a safe runtime for compiling,
dispatching, and reading back Metal compute kernels, and the MSL field
arithmetic that Jolt's and Akita's GPU provers share.
Design and invariants: [`specs/jolt-metal-field.md`](../../specs/jolt-metal-field.md).

The crate never enters a verifier dependency graph;
`scripts/check-metal-isolation.sh` enforces this in CI. On non-macOS targets
it builds with no Objective-C dependency, and `Device::system_default()`
returns `MetalError::Unavailable`.

## Requirements

- An Apple GPU of family 7 or later (every Apple Silicon Mac), with macOS 13
  or later (MSL 3.0).
- Shaders are compiled from source at runtime, so no Xcode or `metal`
  toolchain is needed; the Command Line Tools are enough.
- Do not build a consumer with `panic = "abort"`. The runtime catches
  Objective-C exceptions with `objc2::exception::catch`, which needs
  unwinding.

## Field arithmetic

Each field type is bit-exact with its `jolt_field` counterpart:

| MSL type | Header | `jolt_field` type |
|---|---|---|
| `jolt::Fp128<C>`, the field `2^128 − C` | `shaders/jolt/field/fp128.h` | `solinas::Fp128<P>` |
| `jolt::Fp64<C>`, the field `2^64 − C`, odd `C < 2^32` | `shaders/jolt/field/fp64.h` | `solinas::Fp64<P>` with a 64-bit `P` |
| `jolt::Ext2<F>`, `F[u] / (u^2 − 2)` | `shaders/jolt/field/ext2.h` | `solinas::Ext2<F>` |

Each has `+`, `-`, `*`, unary `-`, `square`, `mul_u64`, `mul_i64`,
`from_u64`, and `from_i64`; `Ext2` also has `mul_base`. Write a kernel once
as a template over the field type and instantiate it per `MetalField`:

```metal
template <typename F>
kernel void my_mul(device const F* a [[buffer(0)]],
                   device const F* b [[buffer(1)]],
                   device F* out [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
    out[i] = a[i] * b[i];
}
```

```rust
let spec = FIELD_HEADERS
    .iter()
    .fold(LibrarySpec::new(), |spec, (name, text)| spec.source(name, text))
    .source("my_kernels.metal", MY_KERNELS)
    .instantiate::<Prime128OffsetA7F7>("my_mul");
let library = ShaderLibrary::compile(&device, &spec)?;
let pipeline = library.pipeline(&host_name::<Prime128OffsetA7F7>("my_mul"))?;
```

The runtime compiles one source string, so `#include` does not resolve: add
`FIELD_HEADERS` before the sources that use them. `MetalField` types upload
as bytes and read back through a canonical-form check, so a kernel that
writes a value outside `[0, p)` makes `DeviceBuffer::read` fail with
`MetalError::InvalidReadback`.

## Errors

Every failure is a `MetalError`, and `MetalError::class()` says what the
caller may do:

| Class | Meaning | Caller may |
|---|---|---|
| `Unavailable` | no supported device | use the CPU backend |
| `Setup` | shader compile or pipeline creation failed | use the CPU backend; report a bug |
| `Capacity` | an allocation exceeds a device limit | re-plan, or use the CPU for this job |
| `Transient` | the GPU timed out, ran out of memory, or lost access | retry, or fall back |
| `Fault` | a kernel or caller bug | stop; never retry silently |

## Local report

Hosted CI runners may not expose a supported GPU, so every PR that changes
this crate carries a report from a local Apple Silicon machine:

```bash
scripts/metal-report.sh
```

It prints the device and its limits, then runs the test suite twice: normally,
and under Metal's API and shader validation layers with
`MTL_SHADER_VALIDATION_ABORT_ON_FAULT=1`. Without that setting, shader
validation only logs an out-of-bounds access and the test still passes. The
output is Markdown; paste it into the PR description.

GPU tests serialize on a file lock and abort after two minutes on one test,
since a GPU hang can stall the whole machine.

`scripts/metal-report.sh --bench` also runs `benches/field.rs` and appends a
table of GPU and CPU throughput. Run it on AC power, not in low power mode;
the report records both. PRs that change an MSL arithmetic header include
this table.
