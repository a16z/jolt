# jolt-metal

Prover-only Metal support for `jolt-field`: a safe runtime for compiling,
dispatching, and reading back Metal compute kernels, and (from step 2 of the
spec) the MSL field arithmetic that Jolt's and Akita's GPU provers share.
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
