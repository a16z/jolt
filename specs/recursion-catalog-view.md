# Selected-row catalog views for recursion

The recursion guest currently audits every row in each prepared schedule catalog,
although a proof selects one row by digest. Add a binary verifier view containing
the selected full rows and the original ordered list of row digests. Audit every
retained row and check its membership in that list. Recompute the original catalog
digest from the list, preserving the setup identity absorbed into the transcript.
This is a setup representation change; it does not change the proof protocol.

## Requirements and evidence

- **Fact:** `akita-schedules::artifact::catalog_digest` hashes a domain, family
  label, policy digest, row count, and ordered 32-byte row identities. It does not
  hash full row encodings directly. `ResolvedScheduleRow::try_new` audits a full
  row and computes its identity from its exact profiles and expanded schedule.
- **Fact:** Jolt's `append_verifier_setup` absorbs the catalog digest. Simply
  pruning rows changes the transcript and invalidates an existing proof.
- **Requirement:** a view preserves that digest byte-for-byte. Every row it can
  return passes the existing semantic, transition, dimension, and challenge-hook
  checks. Unknown or omitted selections remain errors.
- **Requirement:** a view must not claim that omitted rows were audited. Their
  digests are commitments only. It cannot size or generate a complete setup, or
  serialize itself as a complete JSON catalog.
- **Requirement:** preserve the existing prepared verifier key and NTT payloads;
  their provenance and capacity checks are not weakened by this change.
- **Success criterion:** the same saved proof records verify on host and guest,
  with lower total rows and nonincreasing verification cycles. Include decoding
  and commitment-list hashing in total rows. Repeat the retained result.

## Representation and control flow

Keep complete catalog loading unchanged. Add a separately versioned binary view
with magic, protocol epoch, family, policy digest, original ordered row digests,
and selected schedule rows. Use the existing bounded bincode configuration and
artifact limits; enforce nonempty bounded lists, strictly increasing original
identities, and unique retained rows. Reject trailing bytes and unknown versions.

Decode and bind the header to the concrete configuration. Feed retained schedules
through the existing full-row audit, including challenge hooks. Require each
computed row identity to appear in the original list. The existing catalog-digest
encoder becomes the single owner for hashing either complete rows' identities or
the supplied commitment list; its bytes and domain remain unchanged.

Represent complete versus selected coverage explicitly in the catalog. Lookup
indices contain only audited, available rows. `rows()` documents this distinction.
A completeness check gates JSON export and `SetupRequirements::from_catalog`, so
an incomplete view cannot accidentally populate a setup cache under the complete
catalog identity. Binary re-export preserves the commitment list and coverage.

Expose a producer method accepting selected row identities on an already audited
catalog. It resolves all requested rows before producing the view, deduplicates
requests, and rejects an empty or unavailable selection. Jolt's prepared-setup
API uses this method for the proof rows it will serve. Collect every selection
when preparing multiple proofs; do not specialize only for the last proof.
Keep full catalogs as the normal setup-generation input.

For the controlled experiment, transform only the setup records of the frozen
Fibonacci stream. Preserve all proof and device record bytes. Retain both input
files and hashes; the original remains the rollback artifact. The prepared view
may reject another valid proof whose row was deliberately omitted; document this
restriction rather than silently claiming general catalog support.

## Security argument and checks

The view computes the same domain-separated commitment to the original ordered
row identities. Each available row is audited and hashed with the existing row
encoder, then checked for membership. Substituting a different available row under
an unchanged identity requires a row-hash collision. Changing an omitted identity
changes the catalog digest and therefore the transcript setup identity; it is not
accepted as the same key. No semantic claim is made about an opaque omitted
identity, and no verifier operation resolves it to unchecked parameters.
All data is public; this adds no secret-dependent computation or advice bypass.

| Claim | Owning mechanism | Acceptance check |
| --- | --- | --- |
| Identity unchanged | Canonical catalog digest encoder | Complete/view digest equality and frozen-proof verification |
| Available rows are audited | Existing `try_new` and challenge checks | Malformed selected row and policy/family mismatch rejection |
| Selected row is committed | Computed identity membership | Missing/replaced identity rejection |
| Omitted rows cannot execute | Existing lookup over retained rows | Omitted selection fails |
| Full setup cannot be inferred | Explicit coverage check | Setup sizing and JSON export reject a view |
| Commitment changes bind | Existing transcript setup absorption | Altered commitment list changes identity and fails original-proof verification |
| Compatibility | Separate binary magic, unchanged full decoder | Full-format tests and bounded/trailing-byte cases |

Implementation order: catalog format and invariants; prepared-setup producer;
controlled frozen-stream transformation; host verification; guest measurement;
then permanent integration and compatibility checks. Keep this mechanism in its
own companion/Jolt commit pair.

## Validation result

The saved Fibonacci proof verifies on host and guest with unchanged proof/device
records (81,560 bytes). Only setup records changed: 6,253,088 input bytes became
6,211,264. The selected binary catalog is 2,585 bytes. The new stream SHA-256 is
`7741abc8f433b972ef68af9b8b3804840c74740537d403f93bf427ed218b53fa`.

Two traces of the same candidate ELF returned output 1 and exactly 76,946,007
verification cycles / 80,515,127 total rows. The preceding retained version used
90,571,969 / 94,155,100. Replaying the candidate ELF on the original full-catalog
input used 90,635,055 / 94,218,202; this isolates the representation gain from
small compiler/code changes in the candidate itself.

All 64 companion schedule/configuration tests pass. They cover full/view identity,
full-format compatibility, omitted rows, membership, canonical ordering, framing,
configuration mismatch, malformed selected-row transitions, and rejection of setup
sizing/JSON export. Jolt's existing catalog replay test now also covers views:
live and transported setups accept the original proof, omitted rows fail, missing
prepared keys fail, unavailable selections do not mutate serialized setup, and
another catalog's view rejects replay. Scoped library Clippy checks pass.

The host producer collects every proof's selection and verifies all records after
transporting the restricted setup. Runtime verification uses existing prepared
keys; complete-catalog sizing remains forbidden for views. Full outer recursion
proving remains outside the trace evaluator; the combined campaign acceptance
matrix is tracked separately.
