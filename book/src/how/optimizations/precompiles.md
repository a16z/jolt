# KECCAKF precompile tracing

The default-off `pcall-keccakf` feature enables KECCAKF instructions,
their memory-only expansion, and a fixed tracing capacity. Proving with
`pcall-keccakf` enabled is not supported yet and returns an error, even
for programs with no precompile calls.

Enable `pcall-keccakf` in `jolt-inlines-keccak256` for the guest and in
`jolt-sdk` for the host. Direct tracer users enable the same feature in `tracer`.
For a `jolt_host::Program`, explicitly select the instruction profile:

```rust,ignore
program.set_instruction_profile(jolt_riscv::RV64IMAC_JOLT_PRECOMPILES);
```

The reference tracer selects the precompile profile when the feature is enabled.
Static decoding and preprocessing require the explicit profile above. The SDK feature selects Akita and is incompatible with `zk`.

The safe guest functions are:

```rust,ignore
keccak_f(&mut state);                 // state: [u64; 25]
keccak_absorb_f(&mut state, &block);   // block: [u64; 17]
```

Without the guest feature, both functions emit the existing Keccak inline.
With it, the existing `keccak256_absorb_permute` entry point also emits PCALL,
so the Keccak256 hasher uses it without API changes. Existing inline keys keep
their original expansion in every configuration.

PCALL uses opcode `0x0b`, funct7 `0x0a`, and funct3 `0` for permutation or `1`
for absorb-and-permute. The destination must be `x0`; the permutation flavour
also requires `rs2=x0`. Unsupported flavours and hosts without feature support
reject the instruction. The whole funct7 block is reserved against inventory
registrations, and duplicate inline keys fail validation before use.

The expansion has 27 rows for permutation, or 44 for absorb-and-permute:
a call row, optionally 17 block reads, 25 state read-writes, and a return row.
All guest registers are preserved. Lanes are little-endian `u64` values. Before
any memory effect, the tracer checks alignment, non-wrapping ranges, ordinary
RAM bounds, and state/block disjointness.

Set `MemoryConfig::pcall_keccakf_capacity` on the host to a nonzero power of two;
the default is 2048. Pass that configuration to the tracer, or use
`Program::set_memory_config`. Both flavours share this per-execution limit.
Call `K+1` fails with `PRECOMPILE_CAPACITY_EXCEEDED: KECCAKF capacity K`
before its memory effects. Capacity is never increased automatically.
This limit is enforced during execution; proving remains unsupported.

The x86 transpiler does not support PCALL. Tracer checkpoints preserve
the capacity and call count during replay.
