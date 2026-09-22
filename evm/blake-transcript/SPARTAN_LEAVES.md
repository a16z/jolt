# Authenticated sparse-table openings and leaves

This is the earlier leaf checkpoint. `SPARTAN_VERIFIER.md` documents the complete
coordinator, which also enforces the final witness obligations.

This slice extends the fixed-hash same-call chain with a fourth stateless module.
`SpartanLeafCoordinator.checkIncompleteSparseLeaves` executes decoding, the
outer/public/inner prefix, both product networks, then all sparse-table openings
and terminal leaves. It never accepts an external intermediate checkpoint.
The original sparse coordinator remains an explicitly incomplete diagnostic gate.
Neither API verifies the final private-witness relation or witness PCS.

## Source map and invariants

The native owner is `jolt-spartan-verifier/src/preprocessed/sparse.rs`, specifically
`slab_query` and the remainder of `verify_sparse` after both product reductions.
The Solidity decoder records each slab's evaluation, scalar, group and wire cursor
while doing the existing canonical traversal. Consumers do not reconstruct proof
geometry from witness values. `SpartanAlgebra.opening` is now the shared opening
transcript/folding owner. `SpartanKzgPairing` owns the shared group equation; both
the earlier public opening and the three new openings use these same functions.
The authenticated commitment and its checked group representation come from the
same decoder and original key/proof bytes.

| Native order | Committed table | Claims | Selector | Query suffix |
|---|---|---:|---:|---|
| deref-evals / deref-selector | proof dereference commitment | 6, padded to 8 | 3 | operations point |
| ops-evals / ops-selector | key commitment 1 | 16, final claim zero | 4 | operations point |
| audit-evals / audit-selector | key commitment 2 | 2 | 1 | memory point |

Each row appends its exact ordered claims before drawing selectors. Their
MSB-first equality weights form one claimed evaluation, with selectors prepended
to the suffix. Each complete HyperKZG transcript, binary-fold check and actual
three-point KZG pairing finishes before beginning the next row. No pairing batch
is deferred across transcript phases. Setup identity, global degree and imported
capacity are bound by the existing decoder and the opening transcript. The affine
G2 auxiliary is re-bound by the shared checked pairing owner on every opening.

After all openings, for each matrix m and each axis a, the module checks:

- Dot triple equals `(dr[m], dr[3+m], op[12+m])`.
- `read = op[6a+m] + alpha*dr[3a+m] + alpha²*op[6a+3+m] - beta`.
  The read and write tree leaves equal `read` and `read + alpha²`.
- The memory identity is the MSB-first binary-index MLE at the memory point.
  Pad rx or ry with **leading zero coordinates** to the memory dimension.
  Initial memory leaf equals `identity + alpha*eq(padded_query, memory_point)-beta`;
  audit leaf adds `alpha²*audit[a]`.

The module transfers the full sponge struct, not a digest as continuation state.
Its peek digest is exposed solely as an independent native checkpoint comparison.
Standalone module calls are non-authenticating diagnostics; the coordinator fixes
module hashes and supplies its own checkpoints. Dedicated diagnostic tests mutate
those internally generated argument bytes to discriminate leaf equations while
keeping the actual PCS proofs valid. Such arguments are never accepted by the
coordinator's external ABI.

## Deployment and complete-verifier boundary

`SpartanModulePolicy` fixes four reviewed runtime hashes. Normal compilation fails
on policy drift; explicit regeneration is followed by a second compile and hash
stability assertion. Each module and both coordinators must fit EIP-170. The new
coordinator rechecks all four code hashes per call. Reverts and malformed ABI
returns fail closed. Actual CREATE, immutable configuration, STATICCALL traffic,
precompile warmness and per-module gas must be recorded in the final evidence.

Native `protocol.rs` also requires `inner.value == dot(outer_weights, private_values)*witness_evaluation`, its witness-evaluation transcript append and the fifth HyperKZG opening at ry. The diagnostic leaf coordinator stops before those checks. `SpartanVerifier` preserves the original outer weights and invokes `SpartanLeafModule.checkFinal` to complete both obligations in the same call; no caller-supplied completion status is accepted. See [SPARTAN_VERIFIER.md](SPARTAN_VERIFIER.md). Conditional extraction/ROM and no-ZK limitations remain unchanged.
