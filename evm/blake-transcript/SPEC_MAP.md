# Frozen compatibility boundary

Native base: Jolt c281860b26b4550f79343548d673080cb8a080a4 (PR1887).
Spongefish: d2d190b1329d35ac9577438d05aed4f17a57b9f9, local Cargo git source.
The raw GitHub fetch failed in the web tool; the exact Cargo-pinned checkout is the inspected primary implementation.

| Owner / invariant | Solidity implementation | Ground truth |
|---|---|---|
| RFC7693 BLAKE2b-512, unkeyed, 128-byte blocks, final block retained even at exact boundary | Blake2b512.hash | EIP152 abc plus native Blake vectors at 0/1/127/128/129/255/256 bytes |
| EIP152 address0x09,213-byte input,12 rounds BE32,h/m/counters LE64,final flag | compression call | real local EthereumJS precompile execution |
| Spongefish hash.rs masks128 bytes ending0/1/2, CV64 bytes | BlakeHashSponge.State | actual pinned native sponge vectors |
| Absorb from Start prefixes mask0 and CV; repeated absorbs concatenate | absorb | empty/chunked absorbs |
| ratchet = H(H(pending)); after squeezing first squeeze_end then H(H(empty)) | ratchet | native explicit ratchet/zero-size squeeze vectors |
| squeeze block H(mask1 || CV || BE64(block_index)); retained suffix across calls | squeeze | 1/63/64/65/129-byte and split squeezes |
| squeeze_end CV=H(mask2 || CV || BE64(block_count*64-leftovers)); this counts consumed bytes, not requested calls or generated blocks | absorb transition | absorb after partial/multiple/zero squeezes |
| Native platform usize counters are64-bit; this artifact intentionally freezes that profile | uint64 counters | checked overflow, no silent32-bit profile |
| PROTOCOL_ID ASCII a16z/jolt-transcript/v1 zero padded64; session lengthLE64+bytes; empty instance | init wide | native wide constructor |
| Each append bytes=0x9b||LE64(len)||payload | append | adversarial a;b vs ab, empty append |
| Fixed session bn254-blake2b-wide384-v1; append BE64(application label length), then append label padded32; rejectlabels>32 | wide init | native labels including trailingzero |
| Three consecutive16-byte scalar squeezes interpreted BE; ((a*2^128+b)*2^128+c) mod BN254 Fr | wide challenge | actual Bn254WideBlake2bTranscript plus independent integer reduction |
| state() peeks32 bytes on clone without consuming source | state | repeated peek and subsequentchallenge |

Sources: https://eips.ethereum.org/EIPS/eip-152 ; https://eips.ethereum.org/EIPS/eip-1108 ; https://www.rfc-editor.org/rfc/rfc7693 ; https://github.com/arkworks-rs/spongefish/blob/d2d190b1329d35ac9577438d05aed4f17a57b9f9/spongefish/src/instantiations/hash.rs . EIP1108 addresses curve-precompile pricing only; no curve operations or complete Spartan verifier are implemented here.

No alternate hash is substituted. Solidity selector/runtime mechanics may use Ethereum's built-in dispatch conventions; protocol hashes exclusively call Blake2F. Executed gas records use solc 0.8.30 (optimizer 200, evmVersion Prague) and EthereumJS EVM 10.1.0 Hardfork.Prague. Each runCode case has a fresh context and pre-warms address 0x09, matching the precompile access rule in EIP-2929. The boundary excludes transaction intrinsic gas, deployment, and complete verification. No soundness, ZK, full wrapper, or on-chain full-verifier claim.

Validated: 48 executed cases: 10 native hash vectors, 20 native transcript scenarios, one EIP-152 abc vector, six independent 384-bit reductions, and 11 explicit malformed-input reverts. Native transcript scenarios include canonical Fr absorption through the production append implementation, the actual preprocessed-Spartan application label, and distinct trailing-zero labels. The oracle is the executed Rust implementation, not a second JavaScript sponge. Fixture output and tool versions are frozen in the adjacent files.

Not verified: a complete Spartan proof transcript, commitment encoding/point validation, full verifier gas, deployment/transaction costs, or proof security. Counter overflow beyond 64-bit consumed bytes fails closed in Solidity; no such impossible-in-practice EVM allocation is exercised. This compatibility primitive retains pending absorbed bytes and performs bytewise copies; it is not a streaming-memory or gas-optimized implementation. Only public verifier data is in scope; no secret-dependent constant-time or zeroization claim applies.

Fork sources: https://eips.ethereum.org/EIPS/eip-2929 ; https://docs.soliditylang.org/en/v0.8.30/using-the-compiler.html ; https://github.com/ethereumjs/ethereumjs-monorepo/tree/master/packages/evm . Installed package/lockfile pins, rather than the moving master documentation, select the executed implementation.
