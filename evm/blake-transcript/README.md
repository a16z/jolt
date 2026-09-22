# Blake transcript EVM compatibility primitive

This directory implements the exact 64-bit Spongefish Blake2b-512 profile used by
`Bn254WideBlake2bTranscript` at Jolt `c281860b26b4550f79343548d673080cb8a080a4`.
It includes unkeyed BLAKE2b-512 through EIP-152, the raw sponge transitions,
Jolt byte framing, application-domain binding, and the BN254 384-bit challenge
reduction. See [SPEC_MAP.md](SPEC_MAP.md) for the native boundary and quirks.

The harness is a primitive caller, not a Spartan verifier. It does not parse a
proof or validate elliptic-curve points. All protocol hashing uses Blake2F.

Run the frozen native compatibility fixtures with Node 26.5.0:

```sh
npm ci --ignore-scripts --no-audit --no-fund
npm test
```

`package-lock.json` pins solc 0.8.30 and EthereumJS EVM/Common 10.1.0. Compilation
and execution both target Prague, optimizer enabled with 200 runs. The runner
creates a fresh EVM per case and warms precompile 0x09 under EIP-2929. Reported
`executionGas` covers harness bytecode execution, memory, and precompile calls;
it excludes transaction intrinsic/calldata gas, deployment, and all missing
full-verifier operations. `evidence/generated/evm-results.json` records outputs,
reverts, gas, and source/runtime/fixture hashes. Gas is not a wall-time benchmark.

To regenerate the native fixtures from this checkout (at the repository root):

```sh
CARGO_PROFILE_DEV_DEBUG=0 CARGO_INCREMENTAL=0 cargo run --locked --offline \
  -p jolt-transcript --example evm_blake_vectors --no-default-features \
  --features bn254,transcript-blake2b > evm/blake-transcript/test/native-vectors.json
```

The generator calls the production Rust transcript and pinned Spongefish;
its `native_base` identifies the frozen compatibility baseline. Updating the
production transcript or Cargo lock requires an explicit compatibility review
and new provenance, not silently relabeling regenerated vectors. Incidental
SHA3 dependencies in the native feature graph do not select the protocol hash.

The compact test interface is intentionally separate from a future proof ABI:
first byte 0 hashes the remaining bytes; 1 starts a raw sponge; 2 reads a
one-byte label length and label and starts the wide transcript; 3 reduces exactly
48 big-endian bytes modulo BN254 Fr. Raw/wide modes then consume operations:
1 absorbs a big-endian u32 length and bytes; 2 squeezes a u32 count (raw only);
3 ratchets (raw only); 4 peeks 32 bytes; 5 draws a wide field challenge. Outputs
concatenate in order, with challenges encoded as 32 big-endian bytes. Unknown
modes/opcodes, truncated framing, labels over 32 bytes, and wrong reduction
widths explicitly revert.

The implementation favors an auditable compatibility baseline: absorbed bytes
are retained and copies are bytewise. The executed 1,024-byte hash case costs
506,802 gas under this harness. It is not a gas-optimized final verifier.
