# Authenticated Spartan inputs: bounded Solidity slice

`SpartanInputBoundary.decodeAuthenticated(key, setup, inputs, proof)` implements
canonical decoding and the native initial transcript prefix. **It does not verify
a proof.** The accepted canonical-algebra-tamper control intentionally demonstrates
this boundary. Its immutable key ID and setup digest are verifier policy, not
proof-supplied authority. No contract was deployed.

## Contract and source map

The four ABI arguments are byte strings. Their wire format remains the Rust
wire-v1 contract in `jolt-spartan-verifier/src/preprocessed/wire.rs`; the outer
Solidity ABI is not claimed to have a unique encoding. Key bytes are exactly476,
setup224. Policy authenticates each using parameterized BLAKE2b-256, **not truncated
BLAKE2b-512**. Both lengths use the existing single Blake compression owner.
`test/spartan/blake256.json` contains independently generated Python hashlib
vectors, also checked against pinned Noble and actual EVM execution.

Geometry is validated before proof arrays allocate: nonempty rows, 0<p<columns,
p<=1024, padded logarithms<=32, power-of-two N, consistent R/W/L/T, exact setup
capacity and global degree, and at most1MiB proof bytes. Public inputs are exactly
32(p-1) bytes. The complete wire geometry determines counts; attacker width tags
never determine allocation. Cubic/quadratic rounds retain fixed3/2 scalar slots
plus a bounded1..degree tag and zero unused slots. Tags preserve the exact native
Fiat–Shamir message length. Zero-round layers emit no record. All arguments must
be consumed exactly; trailing bytes fail.

Fr is canonical little-endian and strictly below BN254 scalar r. Compressed G1
uses the pinned arkworks sign/infinity flags and base-field q, distinct from r.
Finite points decompress with MODEXP exponent(q+1)/4; the call must succeed and
return exactly32 bytes. The returned root, curve equation, canonical sign and
coordinate are checked. Infinity is only its canonical encoding and is allowed
for proof/key zero polynomials; setup G1 must be nonidentity. BN254 G1 has
cofactor1, so these curve checks establish membership. See [EIP-196](https://eips.ethereum.org/EIPS/eip-196),
[EIP-197](https://eips.ethereum.org/EIPS/eip-197) and [EIP-198](https://eips.ethereum.org/EIPS/eip-198).

**Setup trust boundary:** the immutable digest must identify a setup already
validated through the native import owner. This slice checks its fixed encoding,
identity exclusions, G1 and key binding, but does not independently decode or
subgroup-check G2. It cannot safely authorize arbitrary user-chosen setup digests.
Known-beta fixture setups are measurement/test material, never deployment setups.

The prefix follows `ComputationKey::begin`: wide384 transcript domain, computation
key label/ID, counted public inputs, witness commitment, tau, then retained
`spartan-outer` plus zero. Wire Fr is LE; native `AppendToTranscript<Fr>` is **BE**.
This distinction is enforced by two actual native prefix vectors. G1 transcript
bytes remain the canonical compressed input. The 512-bit primitive vectors are
unchanged. Parameterized256 follows [RFC7693](https://www.rfc-editor.org/rfc/rfc7693)
and the existing [EIP-152](https://eips.ethereum.org/EIPS/eip-152) compression path.

## Remaining verifier implementation slices

1. Port outer sumcheck from `preprocessed/protocol.rs`, reconstruct omitted c1,
   absorb exactly the tagged coefficients, and check its terminal
   eq(tau,rx)(a*b-c) relation. Native zero-polynomial short messages must work.
2. Bind public claims, selector and public HyperKZG opening; process inner
   sumcheck and private claim in the exact native order.
3. Port `preprocessed/sparse.rs`: committed roots, memory multiset identities,
   product-network GKR (including zero-round layers), dot leaves and read/write
   timestamp checks. No direct-private evaluator substitution.
4. Verify all five HyperKZG openings, including folding commitments, three-point
   evaluation relations, batching challenges and KZG equations, using the existing
   `jolt-hyperkzg/{scheme,kzg,types}.rs` contracts. Authenticate validated affine G2
   setup data and check pairing results and precompile failures. Any cross-proof
   pairing batching needs its own algebra/transcript review.
5. Only after all checks and cross-language malformed-proof controls pass expose
   a `verify` API. Operational extraction/ROM and ZK remain separate open gates.

## Reproduction and evidence

With the existing pinned dependencies, run `npm run test:spartan`, then
`npm run test:spartan-failures`, and `npm test`. The first command writes the shared
mutation corpus under `evidence/spartan/corpus`. Native comparison:

```sh
cargo run --locked --offline -q -p jolt-spartan-prover --features preprocessed \
  --example evm_spartan_vectors -- --check-corpus \
  evm/blake-transcript/evidence/spartan/corpus
```

The same intentional emitter without `--check-corpus` creates fresh tiny native
proofs, fully verifies them, and emits key/setup/wire/prefix fixtures. Toy and empty
use the accepted4x8 relation, including empty N=2 and honest short rounds/identity.
Norm wire/key/setup are copied from the accepted measured norm proof and Rust wire
packet; no new norm proof was generated. Norm prefix lacks a separate native
checkpoint oracle; exact prefix comparison is toy/empty only.

Final Node suite:33 cases,10 accepted hash/decode controls and23 reverts. Native
shared corpus:26 cases, exact agreement. Original Blake512 suite:48 cases pass.
Five injected MODEXP failure/returndata/root cases reject; injected callees are
not crypto or gas oracles. Source/runtime hashes and cases are in the frozen
`review-evidence` directory. Raw bounded native command/result records are retained
there too. Initial non-viaIR compilation failed stack depth; first EVM run exposed
and fixed LE-vs-BE prefix mismatch. Initial emitter Clippy found format_collect;
its hex renderer was repaired. These failures are retained, not discarded.

Solc0.8.30, viaIR, optimizer200, Prague, EthereumJS10.1.0, Node26.5.0. Runtime12783B.
Both MODEXP0x05 and Blake2F0x09 are explicitly warm, fresh EVM per case. Observed
execution gas: toy2,964,983; empty2,612,566; norm16,315,581. This is **decoder/prefix
execution only**, excluding transaction/calldata, deployment, all sumchecks/GKR
and PCS. It is not full verifier gas, not a backend recommendation, and not a
compiler-independent additive lower bound. Current byte-decoding loops have not
been optimized. No Solidity security audit, arbitrary-G2 import, full EVM proof
acceptance, deployment, ZK or complete wrapper claim is supported by this packet.
