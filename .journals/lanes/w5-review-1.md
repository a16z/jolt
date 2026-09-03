# W5 assembly review #1

Target: `02c9c924c` (`wrap/spartan-hyperkzg`). Scope excludes
`hash_table/` and `limb_table/` internals. Review performed in detached
`/Volumes/Dev/worktrees/jolt/w5-review1`.

## Verdict

**2 blockers / 2 majors / 2 minors.** The wrapper does not yet prove that the
T1 transcript, R verifier algebra, T2 Dory operands, and seven external
statement values belong to the same inner proof.

## Findings

### Blocker 1 — T1 hashes different proof data than R and T2 may verify

**Files:**

- `crates/jolt-wrapper/tests/wrap_real_t1_r.rs:131-138,701-751`
- `crates/jolt-wrapper/src/relation_table/mod.rs:233-255`

`challenge_copy_link` consumes only `LinkMap::challenges`. It never consumes
`LinkMap::wires`, `LinkMap::wires_shifted`, or `LinkMap::bytes`. R allocates
1,222 anchor rows for absorbed Fr values, but no link uses
`RelationCellLayout::absorbed_word_base`. T1's element/public byte links are
also unused, so T2's Dory group operands need not equal the opaque group bytes
absorbed by T1.

Attack: choose a valid T1 hash trace and its squeeze outputs, then choose R's
sumcheck coefficients/opening claims after those outputs are known. The
challenge CopyLink still passes, while R verifies different field messages.
Likewise, T2 may verify group elements different from T1's absorbed encodings.
This removes the commit-before-challenge property of every inner sumcheck and
of the Dory transcript.

**Fix:** add key-owned links for every aligned/shifted Fr occurrence and every
opaque element/public-byte occurrence. Link T1's `fr_word` forms to R's 1,222
absorbed-word anchors; link T1 element bytes to the exact T2 input limbs. Add
count/order assertions proving every `LinkMap` entry is consumed once. The key,
not the proof or test harness, must own these maps and physical column IDs.

### Blocker 2 — the seven external statement fields are free R witnesses

**Files:**

- `crates/jolt-wrapper/src/relation/public_io.rs:52-61,129-139`
- `crates/jolt-wrapper/src/relation_table/mod.rs:233-255`
- `crates/jolt-wrapper/tests/wrap_real_t1_r.rs:175-178,701-751`

`ValIo`, `InitEval`, and the five bytecode stage values are allocated as free
R1CS variables. `set_input` assigns them only while generating the honest
witness. The row table anchors schedule Fr values, squeeze outputs, and Dory
scalars, but not these seven variables. Putting the seven values in
`AssemblyStatement::public_inputs` changes Fiat-Shamir state; it does not make
them equal to R's wires.

Confirmed on the real fibonacci fixture: incrementing only statement
`public_inputs[0]`, rebuilding the wrapper proof with the original R witness,
and calling `verify_wrapped_with_key` returned `Ok`. The regression then failed
at its expected-rejection assertion after 162.62 s.

**Fix:** give the seven variables stable R anchor cells and add a verifier-side
constant arm to the T1/R `CopyLink` (or a separate link member). Values must be
derived from the stored preprocessing/program/IO statement. Keep the mismatch
regression from the attached patch.

### Major 1 — `WrapVerifierKey` does not determine the verified relation

**Files:**

- `crates/jolt-wrapper/src/wrap.rs:177-204,508-533`
- `crates/jolt-wrapper/tests/wrap_real_t1_r.rs:306-312,352-507`

`verify_wrapped_with_key` accepts `exporters: &[&dyn TermExporter]` outside the
key. Those objects choose the term count, coefficients, challenge fields,
column IDs, and stage-member mapping. `WrapVerifierKey` stores neither the R
table/CopyLink geometry nor their exporter descriptors. The only full caller
is an ignored test, which constructs R and CopyLink exporters from prover-side
`committed.challenges()` and witness-built objects.

Defect: a serialized key and proof have no single accepted language; changing
the caller-supplied exporters changes what the same key verifies. There is no
production entry point that reconstructs the six members and five exporters
from key data.

**Fix:** store typed T1/R/Copy/T2 placements and link descriptions in the key.
Derive phase challenges once inside verification, construct every exporter
there, and remove the exporter argument from the public wrapper verifier.
Keep generic assembly functions crate-private test machinery.

### Major 2 — returned verifier cost omits a full 149-Keccak replay

**Files:**

- `crates/jolt-wrapper/src/wrap.rs:526-531`
- `crates/jolt-wrapper/src/stream/protocol.rs:182-207,789-833`

`verify_wrapped_with_key` first calls `key.challenges`, using an uncounted
Keccak transcript, then calls `verify_assembly_with_cost`, which replays the
same key/public/commitment phases using `CountingKeccakTranscript`. At k=32 the
first replay executes exactly 149 hashes:

```text
1 initialization + 1 key digest + 7 public Fr + 26 full commitments
+ 114 phase challenges = 149
```

The returned count is 471, while this Rust verification path executes 620.
Under the report's 100-gas/event convention, the reported 2,520,261 is 14,900
below the executed path: 2,535,161. The ecMul/ecAdd/pairing/Fr/inversion counts
otherwise route through their observers, and the published arithmetic matches
the bundled 7,700-gas MSM-term convention.

**Fix:** avoid the second replay. Return the live transcript and phase
challenges from key processing and pass both into assembly verification. This
also removes verifier work. If the table models a future fused EVM verifier,
label it as that model rather than executed Rust work.

### Minor 1 — shared round opening enforces degree six for degree-five stages

**Files:**

- `crates/jolt-wrapper/src/stream/protocol.rs:96-104,243-251`
- `crates/jolt-wrapper/src/stream/shared_rounds.rs:177-238,362-426`

Stage A has degree five. The current T=433 term stage has four final factors,
so its coefficient-times-factors polynomial also has degree five. Both are
opened by one `open_variable_batch(..., 6, ...)`, and the verifier selects
`degree_six_shift_g2`. A malicious stage-A/term round may therefore have a
degree-six coefficient. Per-round soundness error rises: 5/r → 6/r. This
contradicts the declared `d+1` bound; it is not a deterministic
forgery.

**Fix:** pass an entry degree for each shared polynomial. Aggregate degree-five
and degree-six commitment classes against their matching fixed G2 shift in one
multi-pairing. For the current four-factor gate, use degree five throughout.

### Minor 2 — the real tamper matrix leaves proof sections untouched

**File:** `crates/jolt-wrapper/tests/wrap_real_t1_r.rs:777-829`

Missing proof mutations: T2 phase 2a, the R/Copy helper commitment, stage-A
round commitment and next claim, term next claim and `S_T(0)`, all three shared
BDFG/shift G1 values, stage B, HyperKZG fold commitments, `w`, the negative
evaluation row, and `P0(r^2)`. `psi_group = phase_1b + 1` is still inside phase
1b, so it does not cover phase 2a. The sign/psi/digit row checks mutate an
uncommitted witness matrix and call the row relation; they are not wrapper-proof
tamper tests.

**Fix:** apply `.journals/lanes/w5-review-1-tests.patch`. The expanded mutations
compile. They were not rerun through the 162 s gate because this review was
limited to two real runs; the public mismatch run already supplied the needed
reproduction.

## Verified properties

- Fiat-Shamir order inside the connected assembly is sound: key digest and
  seven statement fields first; each commitment phase before its challenges;
  input claims before stage batching coefficients; every round commitment
  before its point; all rounds before shared-opening coefficients; factor
  evaluations before lambdas; reduced claim before HyperKZG folds.
- Term compression algebra is consistent. Zero-coefficient padding slots use
  factor value one; short terms receive constant-one factors; stage B binds all
  four factor evaluations through post-evaluation lambdas, including padding.
- The 64-byte round construction correctly opens each round at `r_i` and the
  aggregate S at `{0,1}`. `sum_claim` uses claims `claim_0..claim_{R-1}`; the
  final round claim is used at its own `r_i`. Finding Minor 1 is the remaining
  degree-bound mismatch.
- HyperKZG binds fold commitments before `r`, absorbs both ±r rows and
  `P0(r^2)` before q, reconstructs later r² values, and checks one four-pair
  G1-side equation. Setup powers are β^0..β^(N-1); no verifier G2 scalar
  multiplication occurs.
- R row lowering hardwires R1CS constants in `q_C`, gives every internal symbol
  one sigma cycle, and uses collision-free IDs `wire * 2^18 + row`
  (maximum 786,431). The gate and grouped LogUp term exporters match their
  row formulas.
- Per-occurrence Dory scalar weights are present: 173 named R wires, 230 chain
  occurrences, plus the constant and theta bases. T2's link claim and the
  negative R exporter close at the same stage-A point.

## Dead code and cleanup

### Delete obsolete fallback surfaces

- `src/carry.rs`: `CarryError`, `CarryProver` and its methods, `carried_final`.
  Consumers are synthetic tests only.
- `src/spark.rs`: all four constants, `SparkError`, `SparkTables`,
  `Spark{Prover,Verifier}Key`, `SparkChallenges`, `SparkWitness`,
  `SparkEvaluations`, `SparkProver`, and `final_claim`. Consumer is
  `tests/spark.rs` only.
- `src/spartan.rs`: `SpartanError`, both public-input structs,
  `PublicChallenge`, `ChallengeDecoder`, `SharedWitnessColumn`,
  `prove_spartan`, and `verify_spartan`. Consumers are `spartan_core`, the
  profiler, and the unused `WrapPreparation::{public_challenges,shared_witness}`
  fields. R is now the row table.
- Old tensor stream: `TensorTerm`, `TensorStreamStatement`, `StageAEncoding`,
  `prove_stream`, `verify_stream`, `verify_stream_with_cost`, and their
  `ColumnReduction` path. Consumers are synthetic/timing/profile tests only.
- Standalone R proof in `relation_table/protocol.rs`: `RelationTable{Prover,
  Verifier}Key`, `RelationTableProof`, `setup`, `prove`, and `verify`. It is not
  the assembled wrapper and is used only by its unit/fixture test.

Delete the matching test-only exports from `lib.rs`/`stream.rs`. This removes
most of `stream/protocol.rs`, currently **1,012 lines**, the only scoped source
file above the 1,000-line soft limit.

### One-caller or unnecessarily public types

- `WrapLimbKey`, `DoryLinkedProver`, and `NegatingTermExporter`: one ignored
  e2e caller each; fold them into the canonical assembly/key path or keep them
  private.
- `TermStageProver`, `WeightedColumnReduction`, and `TermReduction`: protocol
  internals; make crate-private.
- `RelationTermsContext`, `CopyLinkTermsContext`, and
  `DoryScalarTermsContext`: adapter/test construction only; keep their builders
  private after the key owns exporter creation.
- `verify_wrapped` is a one-line public delegation used only by
  `verify_wrapped_with_key`; inline it.
- The whole real assembly currently has no non-test production caller; its only
  owner is ignored `tests/wrap_real_t1_r.rs`.

### Development-only checks and duplicate oracles

- `relation::native::check` and `Witness::native_parity` rebuild all eight
  native stages on every production witness generation. Move the parity pass
  and field behind debug/test cfg; the R1CS witness check remains.
- `RelationTable::{final_value,final_value_observed}` and
  `CopyLink::{final_value,final_value_observed}` are test-only second copies of
  the exporter formulas. Tests at `relation_table/tests.rs:96-155` and
  `relation_table/copy_link.rs:478-575` use those copies as their oracle. Keep
  the R1CS check / direct row relation as the independent oracle and remove the
  duplicate evaluators.
- `tests/perf1_profile.rs`, `tests/stream_timing.rs`, and
  `tests/assembly_term_gate.rs` are manual benchmarks/reports in the test
  suite. Move retained measurements to a named benchmark/tool; remove the old
  tensor/fallback probes.

### Docs and style

- `src/lib.rs:3-7`, `src/relation/public_io.rs:4-5`, and
  `src/wrap.rs:535,542-550` still describe the deleted Spartan/public-column
  design.
- No `#[allow]` found in scope.
- Nominal-import violations: `src/profile.rs:29` (`EncodeError`),
  `src/relation/tables.rs:131,178,193,204` (`Range`),
  `src/relation_table/copy_link.rs:510,556,559`,
  `tests/spartan_core.rs:97`, `tests/relation_fixture.rs:405`,
  `tests/relation_dory_native.rs:94-95,142-143,190,201`, and
  `tests/assembly_term_gate.rs:98`.

## Verification

- Real mismatch gate: expected rejection assertion **failed**, proving
  acceptance of mismatched statement/R values; 162.62 s.
- `cargo check -p jolt-wrapper --features prover-fixtures --test
  wrap_real_t1_r --message-format=short`: passed with the attached patch.
- Focused nextest: `table_gadgets_match_native_mles` and
  `named_dory_scalars_satisfy_the_native_deferred_check`: 2 passed.
- No full crate/clippy gate: review policy leaves those to CI; one real gate was
  sufficient to reproduce the blocker.
