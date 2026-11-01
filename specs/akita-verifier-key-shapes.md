# Spec: Akita Verifier Keys for Every Admissible Shape

| Field     | Value      |
|-----------|------------|
| Author(s) | Quang Dao  |
| Created   | 2026-10-30 |
| Status    | proposed   |
| PR        |            |

## Summary

On `main`, an Akita verifier setup verifies exactly one trace shape, and a
verifier that loads its preprocessing from bytes does all setup derivation
inside its first `verify` call. Both problems trace back to one cause: the
setup is built after the guest runs, from that run's padded trace length and
RAM domain. A verifier key is therefore specific to one execution. It rejects a
valid proof of the same program at a different trace length or RAM size, and
its first verification costs 80–180× a warm verification. This spec makes the
Akita verifier preprocessing a function of the program alone. Preprocessing
provisions every shape the verifier's input validation admits into one catalog
and stores the setup-prefix commitments those shapes need. Key load prepares
the backend verifier once. Verification selects the proof's shape and derives
nothing.

## Problem

### What fixes a trace shape

The Jolt verifier accepts any proof whose declared dimensions pass
`validate_inputs` (`crates/jolt-verifier/src/verifier.rs`):

- **Padded trace length `T`**: a power of two in
  `[MIN_PADDED_TRACE_LENGTH, max_padded_trace_length]`. The minimum is `2^12`
  under `akita` (`crates/jolt-verifier/src/config.rs`). The maximum is
  `JoltProgramPreprocessing::max_padded_trace_length`, which the SDK takes from
  `#[jolt::provable(max_trace_length = N)]` (default
  `DEFAULT_MAX_TRACE_LENGTH = 2^24`, `common/src/constants.rs`). Tests use
  `2^16`; the profile harness uses `--scale`. The maximum is part of the
  digest-bound program preprocessing.
- **RAM domain `K`**: a power of two in
  `[compute_min_ram_k, compute_max_ram_k]`
  (`crates/jolt-program/src/preprocess/ram.rs`). The prover derives `K` from
  the highest address the execution touches
  (`ProverConfig::derive_from_dimensions`), so it varies with the input.

Neither `T` nor `K` is known at preprocessing time. Each `(T, K)` pair, together
with the one-hot policy for `log T`, fixes the Akita one-hot trace shape
(`one_hot_trace_setup_shape`, `crates/jolt-prover/src/akita/setup.rs`):

- `num_vars = log_k_chunk + log T`;
- `num_polys`, the instruction, bytecode and RAM `ra` polynomial count. The
  RAM count is `ceil(log K / log_k_chunk)`, so crossing a multiple of 4 in
  `log K` adds a polynomial;
- the layout digest over `ra_layout`, `log T` and `log_k_chunk`;
- the grouped schedule rows for the `FusedInc` (and `FieldInc`) groups, whose
  sizes are functions of `T`.

### Exact-shape setups

`main` builds one setup for the configured shape and enforces an exact match in
three places:

1. `WitnessCommitRequest::new` (`crates/jolt-kernels/src/akita/commitment.rs`)
   requires the plan's `num_vars`, polynomial count and layout digest to equal
   the setup's.
2. `validate_commit_shape` (`crates/jolt-akita/src/scheme.rs`) rejects any
   other dimension.
3. The verifier's native-batching statement check
   (`crates/jolt-akita/src/native_batching.rs`) requires the trace commitment's
   shape to equal the setup's, and `append_verifier_setup`
   (`crates/jolt-akita/src/adapters.rs`) binds the setup's dimensions, layout
   digest and catalog digest into the transcript.

So a verifier key built from one execution rejects a valid proof of the same
program whose trace pads to a different `T` or whose memory use lands in a
different `K` bucket. Relaxing one check moves the failure to the next.

### Verify-time derivation

`AkitaVerifierSetup` serializes only the schedule catalog and shape metadata;
the derived backend verifier is `#[serde(skip)]`. A deserialized setup builds
it on the first `verify`: it decodes the catalog, derives the public matrix
prefix from the seed, and commits to every setup prefix its catalog rows need.
In-process benchmarks hide this because `AkitaScheme::setup` primes the cache.

Measured on `main` (M4 Air, production setup path):

| log T | Prefix slots | Prefix commitments (parallel / 1 thread) | Matrix prefix (parallel / 1 thread) |
|------:|-------------:|-----------------------------------------:|------------------------------------:|
| ≤ 21  | 0            | 0                                        | 3–6 ms / 36 ms                      |
| 22    | 2            | 163 ms / 521 ms                          | 7 ms / 36 ms                        |
| 24    | 5            | 667 ms / 2.3 s                           | 9 ms / 43 ms                        |
| 25    | 11           | 965 ms / 3.4 s                           | 6 ms / 35 ms                        |
| 27    | 19           | 1.84 s / 6.7 s                           | 16 ms / 71 ms                       |

Catalog decode adds about 50 ms. A warm verification at `2^25` takes about
10–11.5 ms parallel and 19 ms single-threaded, so the first verification costs
80× (parallel) to 180× (single-threaded) a warm one. A prepared state for one
shape cannot serve another, so every new shape pays again.

## Intent

### Goal

Make one Akita verifier preprocessing per program (and per schedule artifacts,
chunk profile, advice configuration and one-hot override) that verifies every
proof the program admits. Building it is a preprocessing cost; loading it is a
bounded key-load cost; verification derives nothing.

Abstractions introduced or changed:

- `AkitaSetupShape` (`jolt-akita`, crate-private): one admitted shape's
  `max_num_vars`, polynomial counts and layout digest.
- `AkitaVerifierSetup` gains two serialized fields: `prefix_commitments`, the
  canonical bytes of every setup-prefix commitment its shapes need, and
  `admitted_shapes`, empty when the setup admits only its own shape. Its own
  fields describe the largest admitted shape, and its one-hot catalog holds
  exactly the admitted shapes' rows.
- `AkitaVerifierSetup::prepare` (key load) and the crate-private `for_shape`
  (verification-time selection of an admitted shape; the view shares the
  catalog and prepared backend).
- `merge_verifier_setups` (`jolt-akita`, crate-private, preprocessing): one
  setup admitting every input's shapes, with the setup-prefix commitments they
  need.
- `AkitaProverSetup::adopt_verifier_setup`: binds a prover setup to a merged
  verifier setup after checking that the merged setup admits the prover's shape
  and holds every row of the prover's catalog.
  `AkitaProverSetup::merge_and_adopt_verifier_setups` merges and adopts in one
  step, computing missing setup-prefix commitments from the prover's expanded
  matrix.
- `CommitmentScheme::prepare_verifier_setup` and
  `CommitmentScheme::check_verifier_setup_prepared` (`jolt-openings`, both
  defaulting to `Ok`), and `GroupSetupMetadata::for_shape` (defaulting to the
  setup itself).
- `jolt_prover::akita::preprocessing::preprocess_full_for_verifier`: builds a
  prover preprocessing for one shape under an existing verifier
  preprocessing. The crate-private `admissible_setup` enumerates and
  merges the admissible shapes for every `preprocess_*` call.

### Invariants

1. **Verification derives nothing.** No seed expansion, prefix commitment or
   catalog decode runs inside `jolt_verifier::verify`. A setup that was not
   prepared fails in `validate_inputs`, before any proof work, with
   `VerifierError::PcsSetupPreparationFailed`.
2. **Key load is the only preparation point.** `JoltVerifierPreprocessing::new`,
   and therefore its `Deserialize` (which routes through `new`), calls
   `prepare_verifier_setup`.
3. **The key is canonical.** For a fixed program preprocessing, schedule
   artifacts, chunk profile, advice flags and one-hot override, the serialized
   verifier preprocessing is byte-identical whatever `(T, K)` the prover
   configured.
4. **Admission is complete.** Every `(T, K)` that `validate_inputs` accepts and
   the trusted catalog can prove is admitted. A shape the catalog has no row
   for cannot be proven, so the prover fails on it at preprocessing.
5. **Selection is sound.** A proof is checked only against the row of its own
   shape; see [Soundness of the merged catalog](#soundness-of-the-merged-catalog).
6. **Prover and verifier bind the same setup.** The prover's transcript binds
   the merged catalog digest and its own shape's metadata, exactly what
   `for_shape` gives the verifier.

`jolt-eval` has no invariant covering PCS setup derivation or verifier-key
canonicality. This spec adds none; invariants 1–5 are covered by the tests
below.

### Non-Goals

- **SDK Akita support.** The SDK supports only Dory. This change removes the
  per-execution verifier key, which an SDK integration would otherwise have to
  work around, but adding Akita to the SDK is separate work.
- **Key reuse for committed programs.** `preprocess_full_for_verifier` covers
  full programs. Committed-program preprocessing still enumerates and merges on
  every call.
- **Storing the derived matrix prefix.** Key load derives it from the seed
  (16–71 ms at `2^27`); storing it would add up to 64 MiB to the key.
- **Wire and transcript compatibility.** Verifier preprocessing serialized
  before this change does not load, and proofs bind the merged catalog digest,
  so they do not verify against keys built on `main`.
- **Dory.** Dory's setup already covers every `T` up to the maximum; its
  behaviour is unchanged, and the new trait methods keep their defaults there.

## Evaluation

### Acceptance Criteria

- [ ] A proof whose trace length and RAM domain both differ from the shape the
  verifier preprocessing was built at verifies under that preprocessing, after
  a bincode round trip.
- [ ] Verifier preprocessings built at two different `(T, K)` for the same
  program are byte-identical.
- [ ] `preprocess_full_for_verifier` produces a prover preprocessing whose
  proofs verify under the reused verifier preprocessing.
- [ ] A merged setup returns `None` from `for_shape` for a shape it does not
  admit (different `num_vars`, polynomial count or layout digest).
- [ ] A proof whose schedule selection names another admitted shape's row is
  rejected (by the opening-layout check, not incidentally).
- [ ] A proof verified under another admitted shape's view is rejected.
- [ ] A prover setup whose shape the merged setup does not admit fails
  `adopt_verifier_setup`.
- [ ] `check_verifier_setup_prepared` fails on a deserialized, unprepared setup
  and succeeds after `prepare`.
- [ ] The forced-K=256 end-to-end test still passes, with the override applied
  to every enumerated shape.
- [ ] All existing Akita end-to-end, `e2e_matrix`, `jolt-akita` and
  `jolt-verifier` tests pass; the default (Dory) build compiles.

### Testing Strategy

New tests:

- `jolt-prover/tests/akita_e2e.rs`,
  `one_verifier_key_accepts_every_admissible_shape`: proves `muldiv` at its
  derived config and at `(2T, 16K)`, which changes the RAM `ra` polynomial
  count. It builds the second prover through `preprocess_full_for_verifier`,
  compares the bincode key against one built independently at `(2T, 16K)`, and
  verifies both proofs under the transported key.
- `jolt-akita` `verifier_resources::tests`,
  `merged_setup_verifies_each_admitted_shape_and_only_its_own_row`: merges
  one-hot setups at 14 and 15 variables and checks the rest of the criteria
  above: `for_shape` rejections, adoption refusal for a 16-variable prover,
  verification through each view, cross-view rejection, and rejection of a
  selection swapped to the other shape's row digest. The swapped selection
  fails with "committed-group descriptors do not match the opening layout".
- `jolt-akita` `scheme::tests`,
  `serde_transported_setup_prepares_its_backend_key_at_load`: extended with
  the `check_verifier_setup_prepared` assertions.

Existing suites that must pass:

```bash
cargo test -p jolt-prover --features prover-fixtures,akita --test akita_e2e --test e2e_matrix
cargo test -p jolt-akita
cargo test -p jolt-verifier
cargo check -p jolt-verifier -p jolt-prover --all-targets
cargo clippy -p jolt-akita -p jolt-openings -p jolt-verifier --all-targets -- -D warnings
cargo clippy -p jolt-prover --features prover-fixtures,akita --all-targets -- -D warnings
```

ZK mode is unaffected: BlindFold does not run under `akita`.

### Performance

Measured with the profile harness (`btreemap`, M4 Air, release). "`main`" is
the exact-shape key. "Untrimmed" is this design with full base catalogs in
every shape setup, an earlier revision of this spec, kept to show what
per-shape catalogs buy.

| | `main`, 2^22 | untrimmed, 2^22 | this spec, 2^22 | `main`, 2^25 | untrimmed, 2^25 | this spec, 2^25 |
|---|---:|---:|---:|---:|---:|---:|
| Admitted shapes (`T` × RAM `ra` count) | 1 | 33 | 33 | 1 | 42 | 42 |
| Verifier preprocessing (bincode) | ≈ 6.85 MB | 7.26 MB | 0.96 MB | ≈ 6.85 MB | 7.41 MB | 1.24 MB |
| Stored prefix commitments (slots) | — | 1055 B (2) | 548 B (1) | — | 5618 B (11) | 2069 B (4) |
| Preprocessing: own-shape setup | 245 ms | 245 ms | 230 ms | 816 ms | 816 ms | 450–600 ms |
| Preprocessing: enumerate and merge | — | 5.2 s | 75 ms | — | 6.6 s | 0.6–0.75 s |
| Key load (deserialize + prepare) | ≈ 50 ms, catalog only | 69 ms | 23–34 ms | ≈ 50 ms | 74 ms | 28–44 ms |
| First verification after load | ≈ 230 ms | 9.1 ms | same work | ≈ 1.03 s | 10.2 ms | same work |
| Warm verification (parallel / 1 thread) | 8.7 / 12.3 ms | 8.7 / 12.3 ms | same work | 9.9 / 17.9 ms | 9.9 / 17.9 ms | same work |
| Verification of an unprepared setup | derives state | rejected, < 0.01 ms | rejected | derives state | rejected, < 0.01 ms | rejected |

The `main` and untrimmed figures were taken on an idle machine. The
"this spec" figures were taken with other builds running, alternating with
the untrimmed binary. Under that load the untrimmed binary's enumerate-and-merge
took 5.4–8.8 s at `2^22` and 11.8 s at `2^25`, so the ratios hold. Verification
time under load varied by up to 2× between runs of either binary. Trimming does
not change the verifier's work: both binaries run every Akita verifier phase
the same number of times. The quiet-machine verification figures therefore
carry over.

At `2^25`, 3 of the 4 stored prefix slots belong to smaller shapes; the merge
commits to them from the prover's expanded matrix in about 0.6 s. Expanding
them from the public seed instead took about 1.1 s.

The `main` first-verification figures add the measured prefix-commitment,
matrix-prefix and catalog-decode costs above to a warm verification. The
profile proves at the maximum trace length, so these runs exercise the largest
shape. Smaller shapes under the merged key are covered by the end-to-end test.

Expectations:

- Verification after key load matches warm verification within noise.
- Key load stays under about 50 ms up to `2^25`.
- The verifier key stays under about 1.3 MB up to `2^25`, about a fifth of a
  single-shape key on `main`.
- Preprocessing grows by well under a second up to `2^25`. It is paid once per
  verifier preprocessing. A prover that reuses a verifier preprocessing
  through `preprocess_full_for_verifier` pays only its own shape's setup and
  the catalog-coverage check.

No `jolt-eval` objective measures preprocessing or key-load cost; this spec
adds none.

## Design

### Architecture

```text
preprocessing (once per program)
  admissible_setup
    for T in [2^12, T_max], K in [min_ram_k, max_ram_k], powers of two:
      shape = one_hot_trace_setup_shape(T, K)     dedup on (T, num_vars, num_polys, digest)
      verifier_setup_from_params(shape)           skip shapes the catalog cannot prove
    prover_setup.merge_and_adopt_verifier_setups  union catalog, admitted_shapes,
                                                  prefix commitments for the union,
                                                  bind the prover to the result
  JoltVerifierPreprocessing::new                  prepare (key load)

key load (once per verifier process)
  deserialize -> new -> prepare_verifier_setup    derive matrix prefix, decode catalog,
                                                  build backend verifier from stored prefixes

prove (per execution)
  preprocess_full_for_verifier(key, config)       prover setup for this shape only,
                                                  adopt key.pcs_setup

verify (per proof)
  validate_inputs: check_verifier_setup_prepared  typed error if not prepared
  stage 8: pcs_setup.for_shape(proof's shape)     view sharing the prepared backend
           validate_one_hot_trace_metadata, verify_batch
```

**Enumeration.** `admissible_setup` walks the same ranges
`validate_inputs` accepts, reusing `compute_min_ram_k` and
`compute_max_ram_k` rather than restating the bounds. Many `(T, K)` pairs give
the same shape (the RAM count changes once per four doublings of `K`), so the
loop deduplicates on the cheap shape key before the expensive
`verifier_setup_from_params`. The configured shape's own setup is seeded first,
so its stored prefix commitments are reused. The one-hot policy is the default
for each `log T`, unless the caller's config overrides it, in which case every
shape gets the override. All shape setups are merged in one call.

**Per-shape catalogs.** A shape's setup holds only the base-catalog rows for
its own final group (its scalar guide row and any grouped rows the base
catalog already has) plus the grouped rows provisioned for it
(`GroupedScheduleParams::extend_catalog`). Rows for other final groups would
not change which row any proof can select, but they would add setup-prefix
slots, whose commitments the key stores, and about 6.7 MB of rows to every
shape setup that the merge decodes again. `AkitaScheduleArtifacts` decodes
each shipped catalog once per process and keeps the decoded catalog, so
building 33–42 shape setups decodes the 6.7 MB base catalog once.

**Merging.** `merge_verifier_setups` requires every input to share its dense
catalog, one-hot `K` and chunk profile. It decodes each one-hot catalog,
deduplicates rows by row digest and rebuilds a `ValidatedScheduleCatalog`.
`try_new` sorts rows by digest and rejects duplicate lookup keys, so the merged
bytes are canonical and a conflict fails loudly instead of picking a row. The
merged setup's own fields come from the largest shape, and `admitted_shapes`
lists every shape. Prefix commitments are provisioned once for the merged
setup, from the merged rows: commitments the inputs already store are reused.
Because each shape setup holds only its own rows, the own-shape setup stores
only its own slots, and smaller shapes' rows can need slots it lacks (at
`2^25`, 3 of the 4). `AkitaProverSetup::merge_and_adopt_verifier_setups`
computes those from the prover's already expanded setup matrix, which holds
every prefix up to the largest shape's capacity; only a slot larger than that
matrix would be expanded from the public seed.

**Prover binding.** The transcript binds the setup's dimensions and catalog
digest, so the prover must bind the merged catalog. `adopt_verifier_setup`
replaces the prover's verifier view with `merged.for_shape(own shape)` after
checking that the shape is admitted, the one-hot `K`, chunk profile and dense
catalog match, and every row of the prover's own one-hot catalog is in the
merged catalog. The prover keeps its own backend prover setup, sized to its
own shape.

**Verification.** `validate_inputs` calls `check_verifier_setup_prepared`
first. Stage 8 computes the one-hot trace plan from the proof's declared
dimensions, as before, and asks the setup for the matching view. A shape the
setup does not admit fails with "the PCS setup does not admit this proof's
trace shape". The view then goes through the unchanged exact-shape checks.

#### Soundness of the merged catalog

Merging gives a verifier access to rows for shapes other than the proof's.
The proof cannot use them:

1. The verifier fixes the shape from the proof's declared `(T, K)` and one-hot
   configuration, which `validate_inputs` already bounds, and
   `validate_one_hot_trace_metadata` and the native-batching statement check
   require the trace commitment to have exactly that shape and digest.
2. The proof's `schedule_selection` names a row by digest.
   `shape_guard::deserialize_checked_grouped_backend_payload` resolves it in
   the catalog and then calls `validate_opening_layout` against the exact
   `(num_vars, poly_count)` of every group in the statement. A row provisioned
   for another shape describes different groups and is rejected.
3. The row digest and the merged catalog digest are bound in the transcript,
   so the selection cannot change after the challenges are drawn.
4. Every row in the merged catalog came from a catalog that would have been
   trusted for its own shape on `main`. Merging adds no row that a
   single-shape verifier for some admitted shape would not already trust.

So a merged verifier accepts a proof only if the single-shape verifier for
that proof's shape would. The merged verifier also binds a different catalog
digest than any single-shape verifier, so proofs are not interchangeable
between the two.

### Alternatives Considered

**A. Keep lazy derivation (status quo).** The first verification pays prefix
commitments and matrix derivation: 80–180× a warm verification at `2^25`, and
again for every new shape. Rejected by invariant 1.

**B. Pad every execution to `T_max`.** One shape, so one key. But the prover's
setup matrix and commitment work scale with `T`: at `2^27` the prover setup
holds about 33.5M field elements (512 MiB), and a short execution would pay
the full cost. `K` would also need padding to `max_ram_k`. Rejected for prover
cost.

**C. One full setup per shape (a setup ladder).** Ship one exact-shape
verifier setup per admissible shape. With `main`'s full catalogs the key grows
to roughly `n × 6.7 MB` (`n` ≈ 13–40 shapes); even with per-shape catalogs each
setup repeats the dense catalog and its prefix commitments, and key load
prepares `n` backends. The merged setup stores each row and commitment once
and prepares one backend.

**D. Derive the shape's setup at verify time from a trusted catalog.** The
verifier would hold the union catalog and build the selected shape's setup on
demand. That puts catalog decoding and prefix commitments back on the
verification path. Rejected by invariant 1; this spec does that work at
preprocessing and key load instead.

**E. Store per-shape deltas and rebuild per-shape catalogs at key load.**
Avoids the union but needs one 6.7 MB decode (about 50 ms) and one prepared
state per shape at key load, plus a delta format. The union is simpler and
cheaper.

**F. Bind a per-shape catalog digest while holding the union.** The transcript
would bind the digest of the selected shape's sub-catalog, keeping proofs
compatible with single-shape keys. This needs per-shape canonical sub-catalogs
(or a commitment over them) at verify time. The row binding argued above
already gives the same soundness, so the extra machinery buys only
compatibility, which is a non-goal.

**G. Trim the catalog to one shape's rows (a16z/jolt#2186).** #2186 cut an
exact-shape catalog from 519 rows (6.7 MB) to the 2 rows the shape uses
(31 KB). On its own that keeps keys exact-shape. This spec adopts the same
trimming per shape (see Per-shape catalogs) and merges the trimmed catalogs,
so the key holds the admitted shapes' rows and nothing else.

**H. Fix the cost in Akita.** Most of the cost before trimming was Akita's
catalog codec and the lack of incremental catalog operations: every decode
re-encodes the catalog to check canonical form (about 32 of 59 ms), every
extend or merge re-validates every row, and setup requirements are sized from
every row of a catalog. Akita APIs for extend, union and subset that check only
new rows, a cheaper canonical-form check, and setup requirements from a chosen
set of rows would let Jolt drop its by-hand trimming. They are proposed
upstream separately; this spec does not depend on them.

## Documentation

`book/src/how/akita.md`, "Setup and protocol selection": add a paragraph on
one verifier preprocessing admitting every shape, and replace "verifier
preprocessing streams setup-prefix commitments from the public seed" with the
preprocessing / key-load / verification split. The new public APIs carry
rustdoc.

## Execution

Files touched:

- `crates/jolt-openings/src/schemes.rs`: `prepare_verifier_setup`,
  `check_verifier_setup_prepared`, `GroupSetupMetadata::for_shape`.
- `crates/jolt-akita/src/adapters.rs`: `AkitaSetupShape`, new
  `AkitaVerifierSetup` fields, `prepare`, `require_prepared`, `for_shape`,
  `admitted_shape_count`, `adopt_verifier_setup`,
  `merge_and_adopt_verifier_setups`.
- `crates/jolt-akita/src/verifier_resources.rs`: stored prefix commitments
  (`stored_prefixes`, `provision_prefix_commitments`, `ExpandedPrefixSource`),
  key-load-only material generation, `attach_prepared`.
- `crates/jolt-akita/src/verifier_setup.rs`: `merge_verifier_setups`.
- `crates/jolt-akita/src/schedule_registry.rs`: `union_one_hot_catalogs`,
  `one_hot_catalog_covers`; `GroupedScheduleParams::extend_catalog` keeps
  only its final group's rows.
- `crates/jolt-akita/src/schedule_artifacts.rs`: keep each decoded catalog.
- `crates/jolt-akita/src/scheme.rs`: store prefix commitments in `setup`;
  implement the two hooks.
- `crates/jolt-verifier/src/preprocessing.rs`, `verifier.rs`, `error.rs`,
  `stages/stage8/akita.rs`: prepare in `new`, early check, shape selection,
  `PcsSetupPreparationFailed`.
- `crates/jolt-prover/src/akita/preprocessing.rs`:
  `admissible_setup`, `preprocess_full_for_verifier`.
- `crates/jolt-prover/src/profile.rs`: build the profiled setup through
  `admissible_setup`.

Open points for review:

- Shapes that `verifier_setup_from_params` rejects are skipped with a `debug`
  log, because a catalog miss is the only expected failure. A narrower match on
  the "outside the admitted catalog" error would need a typed error from
  `jolt-akita`.
- Every `preprocess_full*` call still enumerates every shape (about 0.3 s at `2^22`
  and 1.2 s at `2^25`).
  Provers that preprocess repeatedly can build the verifier preprocessing once
  and use `preprocess_full_for_verifier`.

## References

- a16z/jolt#2186: per-shape catalog trimming, adopted per shape here; its
  exact-shape key motivated this spec.
- a16z/jolt#2136, a16z/jolt#2165: shared verifier resources
  (`verifier_resources.rs`).
- [Akita Native Trace Batching](akita-native-trace-batching.md): grouped
  one-hot trace commitments and their layout digest.
- Akita `akita-schedules` `ValidatedScheduleCatalog::try_new`: canonical row
  order and duplicate-key rejection.
