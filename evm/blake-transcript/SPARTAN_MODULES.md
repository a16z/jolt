# Same-call Spartan modules and product-network gate

This records the first three-module slice. `SPARTAN_LEAVES.md` describes its
fourth-module extension and the current remaining witness checks.

The accepted public-opening runtime is 21,845 B; adding full GKR and four remaining
PCS checks to it risks the 24,576 B deployed-code limit. This slice separates three
stateless modules and a small coordinator. Every runtime must independently fit.
No deployed module or constructor configuration is assumed from a test harness.

The coordinator owns immutable application key/setup identities. Its three module
addresses must match **compile-time expected runtime code hashes** generated from
the reviewed compiler artifacts. Hashes are not constructor or call arguments.
Each same-call STATICCALL checks code identity again. The only external checking
input is raw key/setup/inputs/proof plus the already specified affine G2 auxiliary;
no external transcript or partial checkpoint is accepted. Direct module calls are
non-authenticating diagnostics. A caller cannot substitute their result into the
coordinator. Malformed returns revert through typed ABI decoding; module reverts
propagate. Until all obligations exist, the coordinator returns an explicitly
incomplete result, never a proof-acceptance boolean.

## Data flow and ownership

1. Decode module calls the existing canonical owner using coordinator-supplied
   policy IDs. It returns checked geometry, canonical scalar/group arrays, original
   wire cursors and initialized sponge/tau. It owns wire traversal; new cursors
   are recorded during that traversal, not reconstructed with second sizing laws.
2. Prefix module consumes that decoded result plus the original raw setup/key/proof
   bytes and affine auxiliary. It executes the accepted outer/public/inner checks
   and public pairing. It returns rx,ry,claims and the **actual updated sponge
   state**, not merely `peek()`'s digest. Transcript byte buffers and position/mode
   are required for exact continuation.
3. Sparse module receives the same decoder arrays and the prefix result from the
   coordinator. It binds the private values and dereference commitment, roots,
   halves and both complete product networks. Its result retains the post-network
   sponge, operation/memory points, tree and dot evaluations, and the cursor for
   the later slabs. The next module must authenticate those slabs before accepting
   the terminal leaf identities or witness relation.

The large decoded arrays are ABI-copied between modules. This trades execution
and deployment cost for deployability; no efficiency claim is made before measuring
calls, calldata/returndata copying, all deployed runtimes and deployment gas.
An internal source-file split alone cannot solve runtime size. STATICCALL avoids
storage/delegatecall authority. Setup ceremony and immutable application policy
remain external trust requirements, unchanged from the accepted boundary.

## Exact native sparse obligations

Source owner: `preprocessed/sparse.rs` in jolt-spartan-verifier.
`begin_sparse` appends matrix-query-v2, counted rx,ry,three private values,
matrix-derefs plus compressed commitment, memory-hash, then draws alpha,beta.
`bind_roots` checks each axis's init*wa*wb*wc=ra*rb*rc*audit and the three
left+right half sums before appending 16 roots and 6 halves. Operation claims are
[ra,rb,rc,wa,wb,wc] per axis; memory claims are[init,audit] per axis.

For each Operations then Memory network, append its network label. Layer j appends
product-layer with BE32 j, layer-weights, then draws width 12/4 weights plus 6 dot
weights only at the bottom operations layer. The weighted claims initialize a
cubic compressed sumcheck of exactly j rounds. **Layer0 has zero rounds but still
checks its terminal equation.** At each layer the product terms carry eq(previous
point,new point); bottom dot triple products do not. Append flattened product ends
and optional 18 dot ends, then layer-eta; interpolate ends with eta and prepend eta
to the new point. Bottom dot ends interpolate adjacent pairs into 3 triples.

Remaining native order after this gate: dereference 6 claims/selector 3 + PCS;
operations 16 claims (last must be zero)/selector 4 + PCS; audit 2 claims/selector 1 + PCS;
then dot leaf equality, read/write hash leaves with timestamp increment alpha²,
memory identity/eq hash leaves with leading-zero-extended rx/ry; then inner witness
product, witness label and final PCS. These checks are completed by the leaf/final-witness path in
`SpartanVerifier`; they are not inferred from product-network success. Four
PCS remain after this particular module boundary.

## Acceptance and composition

The sparse coordinator is a diagnostic endpoint for this module boundary. It permits no external checkpoint input, but its success is not complete proof acceptance. The full `SpartanVerifier` invokes the remaining leaf and witness checks in the same call. Current complete receipts, failure controls and measurement scope are recorded in [validation/README.md](validation/README.md). The complete sponge state travels only between authenticated module calls. Operational ROM/extraction and ZK claims do not follow from these checks.

## Reproduction and deployment policy

From `evm/blake-transcript`, `npm run test:spartan-sparse` compiles with pinned
solc 0.8.30, viaIR, optimizer 200 and Prague, deploys all contracts through actual
EVM CREATE, and runs the differential and failure controls. Normal compilation
fails if module bytecode does not match `SpartanModulePolicy.sol`. A deliberate
source/compiler change requires `node test/compile-modules.mjs --write-policy`
and independent review of the resulting policy and module artifacts. The second
compile asserts that adding coordinator/policy sources does not alter module
hashes. Runtime sizes are enforced individually, including the coordinator.

The native fixtures come from `evm_spartan_vectors`, using actual production
`begin_sparse`, `bind_roots` and `ProductReduction::verify` owners. Existing toy,
empty and empty-public fixture files were byte-compared unchanged before adding
`sparse.json`. Toy has N=4 and exercises a nonzero-round operations layer; empty
has N=2 and exercises bottom-layer dot checks at zero rounds. The additional
zero-products fixture sets all roots, halves and product-network messages to zero
and regenerates the native checkpoint. It passes these partial equations and
**fails the full native verifier**: the remaining committed slabs/hash leaves and
witness have not been regenerated. It is a completeness control for zero products,
not a valid proof fixture.

The coordinator exposes `checkIncompleteSparse`, not `verify`. Four PCS openings,
the authenticated hash/dot leaf equations and final witness relation remain.
A mutation of the first later slab claim intentionally still passes this gate.
Module-return fault injection is a test-harness fault control; it does not assert
that an authorized immutable runtime can be replaced on-chain. No operational
extractor, ROM, zero-knowledge or full-wrapper claim follows from these tests.
