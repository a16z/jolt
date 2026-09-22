# Public HyperKZG pairing slice

`SpartanPublicOpeningBoundary.checkIncompleteWithPublicOpening` checks the public
opening in addition to the accepted outer/public/inner prefix. It still does not
verify a Spartan proof: private sparse/GKR relations, the witness product and four
other PCS openings remain unchecked. The older `checkIncompleteAlgebra` API remains
an explicitly partial diagnostic boundary; its returned pending opening is not an
external authentication credential.

## Source and equations

The new `SpartanPublicOpening.check` maps directly to native
`jolt-hyperkzg/src/kzg.rs::HyperKZGVerifierSetup::verify_batch`. With the accepted
prefix's fold challenge r, polynomial batching challenge q and opening batching
challenge d, form powers q^j and weights [1,d,d²]. Let C_j be the public commitment
followed by fold commitments, V_ij the three evaluation rows, and W_i the three
opening witnesses. Points are [r,-r,r²]. The verifier checks

```
L = (1+d+d²) sum_j q^j C_j
    + sum_i weight_i point_i W_i
    - setup.G1 sum_ij weight_i q^j V_ij
R = sum_i weight_i W_i
e(L,setup.G2) * e(-R,setup.betaG2) = 1.
```

All field arithmetic reduces modulo Fr, and group arithmetic uses EIP-196 ADD/MUL.
The existing owner supplies checked decoded groups and scalar arrays; no raw proof
length or transcript rule is copied. Pairing is evaluated after the prefix's inner
sumcheck has been replayed. It is a transcript-free check, so this scheduling does
not change any challenges or valid acceptance predicate. It must succeed before
this new boundary returns. The returned type contains no pending-public marker.

## Affine auxiliary ABI

The first four dynamic byte arguments retain the accepted wire format. The added
fifth argument is `uint256[4][2] g2`, exactly eight ABI words (256 bytes), containing
setup.G2 then setup.betaG2. Each point is `[x.c1,x.c0,y.c1,y.c0]`, the EIP-197 order
for `c0+c1*i`. These are auxiliary coordinates, not new setup authority or extra
Fiat–Shamir messages. Setup/key authentication remains unchanged.

Every coordinate must be strictly below base-field Q. Each x.c0 matches the first
32 little-endian bytes of its authenticated compressed setup point. x.c1 matches
the next32 little-endian bytes with the top two flag bits cleared. Only flags0
(positive) and2 (negative) are allowed; infinity and flag3 are rejected. Positive
means y<=-y; arkworks Fq2 orders c1 first and c0 second lexicographically. Thus flag2
must equal `(y.c1 > -y.c1) || (y.c1 == -y.c1 && y.c0 > -y.c0)`, with each negation
canonical modulo Q. All-zero affine infinity is separately rejected.

Exact coordinate/sign binding plus the pairing precompile's curve/subgroup checks
identifies the same canonical point as native decompression. This avoids Fq2
square-root computation; it does not trust supplied y. The checked coordinates
are the exact words passed to the pairing call. G2 validation must still execute
when both G1 terms are identity. Trusted setup ceremony/powers and application
policy remain external, as before.

[Primary EIP-197](https://eips.ethereum.org/EIPS/eip-197) specifies G2 subgroup
validation and `(imaginary,real)` encoding; [EIP-196](https://eips.ethereum.org/EIPS/eip-196)
defines the ADD/MUL interfaces. Arkworks' pinned `SWFlags::from_y_coordinate` and
`QuadExtField::cmp` determine the compression sign, not a new EVM convention.

## Failure behavior and size

Every ADD/MUL call must succeed and return exactly64 bytes; returned coordinates
are canonical and on curve or identity. Pairing must succeed, return exactly32
bytes, and return the integer1. Failed, empty, short, oversized and malformed
results reject; injected faulty precompiles are failure controls, not cryptographic
oracles. Trust in correct chain precompile semantics is unchanged.

The initial full internal compile measured21,845 bytes with solc0.8.30 viaIR,
optimizer200, Prague. Tests enforce the24,576-byte deployed code-size ceiling.
This leaves little room for GKR; later work must choose an explicit linked or
modular deployment architecture and measure its calls/deployment/setup costs.
No checks may be omitted to meet that limit.

## Evidence scope

Native fixtures export affine coordinates from the actual authenticated compressed
setup through the canonical group owner. They also export a deterministic
on-curve non-subgroup point that the native canonical decoder rejects. EVM controls
cover sign/component/point order, modulus aliases, infinity, off-curve and
non-subgroup inputs; the latter two reach the actual pairing precompile with
identity G1 terms. The former pending-public-pairing control must now reject.
Coherent native fixtures must retain their exact prior claims and transcript state.

The runtime uses no external library deployment for this bounded slice. Reported
execution gas, if recorded, excludes deployment and transaction/calldata costs.
Each precompile's warmth is stated separately. Full verifier acceptance, remaining
PCS/GKR, setup ceremony provenance, operational extraction/ROM composition and ZK
are outside this packet.

## Validation and cost boundary

`npm run test:spartan-public-opening` checks honest native fixtures, the diagnostic API's still-incomplete obligations, canonical/off-curve/non-subgroup inputs, and injected ADD/MUL/PAIRING failures or malformed returns. The complete verifier additionally discharges all remaining PCS and witness obligations; see [SPARTAN_VERIFIER.md](SPARTAN_VERIFIER.md).

The33 current outcomes are retained in [validation/pairing-fault-results.json](validation/pairing-fault-results.json). Native generation uses checked arkworks G2 deserialization and fully verifies proofs before recording fixtures. Fixture parameters have a known setup secret and are unsuitable for deployment.

Calls forward available gas to precompiles. A real invalid point can therefore consume the forwarded budget before reverting; injected zero-cost failure controls test rejection, not real-chain failure gas. Full-verifier measured costs and exclusions are recorded in [validation/README.md](validation/README.md). Operational extraction/ROM, a trusted SRS ceremony, ZK and complete Akita acceptance remain separate obligations.
