# Complete clear Spartan verifier

`SpartanVerifier.verify(key, setup, inputs, proof, g2)` returns `(true, finalState)`
only after the complete native preprocessed clear-Spartan algebraic relation has
been checked. Failure reverts. The digest is a completed-transcript receipt, not
an outstanding obligation or caller-supplied checkpoint. Conditional joint
extraction/ROM assumptions of the native protocol remain; this is not a ZK wrapper
or a proof of the entire Akita/Jolt verifier circuit.

The constructor fixes the application key ID and setup digest. Four stateless
module addresses must match compile-time runtime hashes; hashes are not caller
arguments. The coordinator rechecks all code identities and invokes the modules
within the same call. All intermediate inputs are its own typed return values.
The decoder validates the canonical bounded wire before allocation. Setup identity,
imported capacity, advertised global degree, scalar encodings, checked G1 and
setup-bound affine G2 retain the reviewed contracts in the preceding documents.

## Completion relative to the leaf gate

`SpartanPrefixModule.Result.weights` now preserves the original three outer
challenges, sampled before the public-column reduction. The decoder records the
witness evaluation index and PCS cursor during canonical proof traversal.
`SpartanLeafModule.checkFinal` first performs all three sparse openings and leaf
equations through the same mandatory internal function used by the earlier gate.
It then follows native `preprocessed/protocol.rs` exactly:

1. Form `linear = Σ outer_weights[j] * private_values[j]`.
2. Require `inner.value == linear * witness_evaluation`.
3. Append `witness-evaluation` and the witness scalar through the native transcript
   framing owner.
4. Verify the fifth full HyperKZG opening against the original witness commitment
   at the inner sumcheck point, using the same checked folding/pairing owners.
5. Return the final sponge peek digest; the coordinator consumes that exact typed
   return and only then returns success and its receipt.

No pending-obligation marker is accepted or returned by `verify`. The earlier
incomplete APIs remain clearly named diagnostics and cannot substitute their
results into this coordinator. Module methods called standalone do not authenticate
external checkpoints. The final caller must use `SpartanVerifier`, not a module.

## Validation and gas contract

Run `npm run test:spartan-complete` from this directory. The intentional native
`evm_spartan_vectors` example supplies full accepted toy/empty/empty-public proof
fixtures and their final transcript states. It uses the production native owners,
including the final witness opening; native full acceptance already checks the
witness product. Existing fixture bytes remain unchanged. The formerly pending
witness mutations must now fail at the corresponding product/pairing check.

`npm run measure:spartan-norm` uses the exact previously accepted D64 norm wire
proof, key and setup under SHA-256 guards. It generates no proof or SRS. The
fixture uses known beta=7 and is not deployment-secure ceremony material. It first
runs with 30 million execution gas, and only if a recorded nested call runs out of gas permits one
200-million diagnostic to measure the complete relation. This does not claim
eligibility for any live chain's block or transaction limit.

Results separate EVM call execution, module payload/return copying and call
overhead, actual CREATE/code deposit, and estimated transaction charges. The
Prague [EIP-7623](https://eips.ethereum.org/EIPS/eip-7623) calldata floor and
[EIP-3860](https://eips.ethereum.org/EIPS/eip-3860) creation charges are calculated
from the exact byte payloads; they are not a live-chain receipt. Precompiles
05–09 are warm; module addresses begin cold and are warmed by EXTCODEHASH before
STATICCALL. Every runtime must fit EIP-170 independently. Compiler settings,
code hashes and concrete constructor arguments are part of the deployment policy.

The harness awaits asynchronous journal cleanup and asserts module addresses are cold and precompiles are warm before every measured call. [validation/README.md](validation/README.md) contains the current complete-verifier observations and reproduction commands. Earlier partial measurements that omitted that await do not support the gas claims here.
