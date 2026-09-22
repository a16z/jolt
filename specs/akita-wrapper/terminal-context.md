# Conditional terminal context composition

Status: conditional source fragment; full construction deliberately unvalidated.
Base Jolt `003cb2ba3c55f200dc0fbb45158ad1cd2a7af9b4`,
native f5f75335eae18241681fd24ca0a60fca8f0512af. Uses the reviewed public-e24
census of the earlier28fc proof; no new proof generation. Typed-proof-normalization
review permits typed z and omission of the dead final z absorb, but no other
semantic acceptance condition is dropped.

## Public profile and inherited boundary

The fragment begins immediately before native terminal `ABSORB_COMMITMENT`.
Input is a live `AkitaTranscriptVar` produced by prior constrained operations,
not a root supplied by the caller. The same predecessor t handles are passed to
the terminal equations and canonical transcript encoding. The same incoming
17 point handles and claim handle feed scalar verification and their individual
native field appends. The same448 e handles feed all rows, scalar verification
and the pre-FoldDraw concatenated append. z is the existing16384-coordinate
`TerminalZ`, including its complete physical norm.

Public shape is D64/rank3/width256/7blocks/one inner digit, base-field Lagrange
opening, q=2^128-2^32+22537, energy cap546507225. The full native GrindingPlan
must end at the selected terminal level with exactly its FoldResponse run and
its zero-width FoldChallengeGroup(group0,7coordinates) run. The inherited
boundary identifies the preceding plan query count and logical nonce-bit count;
previous operations must be constrained by the future whole-proof caller.
The fragment consumes one native12-bit logical nonce, performs no invented PoW
at that FoldResponse site, and consumes all7 callback indices0..6. Earlier
ProofOfWork sites remain upstream, including their real predicates and squeezes.

No unauthenticated inherited state or host observer values become acceptance
facts. Fixture replay can establish conditional parity, not authenticate the
preceding state semantically. Setup A/profile, descriptor, preceding folds and
Jolt bridge remain caller obligations. ONE is fixed and all handles belong to
one builder. Errors may leave partial constraints; discard that builder.

## Native order and owner map

1. `suffix.rs::verify_terminal_suffix`: append t as one21504-byte message.
2. `fold/single_field.rs::prepare_single_field_terminal_suffix`: append each of
   17 point coordinates as one16-byte canonical LE message, then the claim as
   one16-byte message. Point preparation itself is transcript neutral.
3. `core.rs::prepare_terminal_witness_replay`: append e as one7168-byte message.
4. Consume the logical FoldResponse slot and native-owned
   `FoldChallengeFrame(EvaluationTrace,D64,group0,7blocks,1claim,selectiveL2)`.
   Reviewed `D64FoldDrawShape::draw` appends the frame and derives the root.
5. Seven calls to reviewed `sample_coordinate`, with public capacities, feed
   their SAME dense handles into `TerminalChallengesVar` and all terminal rows.
6. Omit the final z absorb only by the accepted noninterference lemma. No later
   transcript use is supported by this fragment; downstream terminal acceptance
   is its output.

## Fixture and resource assessment before builds

Actual final observed root is
`f27f73fef09b695e8a5355ea576583a146c5edd9764a4ad5464b6279d03eae2b`;
its native group0 coordinate count is7. Frozen census deliberately does not
observe rejected candidate counts or per-position retry consumption. Those
counts must be obtained from the live native sampler or a clearly labeled
witness-construction probe, not fabricated from aggregate root counts.

R2/K4 costs2,268,605 rows and11,189,260 nonzeros per coordinate in the reviewed
retry owner. All seven would cost roughly15.9M rows. t/e together require about
225 BLAKE2b compressions at51,328 rows each, plus point/claim/framing, root,
canonical-byte linking and complete terminal2.13M rows. Preliminary total is
roughly30M rows and over150M nonzeros, with estimated20–25GiB builder/matrix RSS.
These are estimates, not measurements or permission to run a large construction.
Actual retry reconnaissance must precede final capacity and resource selection;
R2/K4 is not assumed adequate. No full construction is authorized; bounded observer and small helper checks follow below.

Planned controls: actual native root and all7 ordered accepted challenges match;
coherent e/t/nonce mutations change the constrained draw and reject the frozen
terminal assignment, with host capacity rejection distinguished from actual
unsatisfied R1CS; forged root/challenge outputs fail their linkage; unknown
witnesses preserve exact matrices. Do not repeat full walks until measured
geometry is assessed. No full proof or full-wrapper acceptance is claimed.


## Native observer result (source f5 plus temporary logging only)

The existing native sampler reproduced all seven ordered positions/coefficient
arrays exactly. Coordinate candidate counts are [2,1,1,2,3,2,1]; maximum position
trial counts are [4,5,6,4,7,4,6]; consumed byte counts are
[190,97,97,189,293,189,100]. Thus the frozen demonstration profile is uniformly
R3/K7, sufficient for this fixture only. The previous R2/K4 estimate cannot be
used as the complete construction cost. Exact read-by-read logs and instrumented
source/config hashes are in `/private/tmp/terminal-context-20260922`.
Instrumentation calls the actual native sampling and norm owners; no sampler
formula was copied into an observer. An initial Vec/SmallVec comparison compile
error was repaired to slice comparison; the failed log remains archived.

## Frozen capacity and construction limit

The current canonical candidate reader emits `10L+11` rows per read: 8L bit
selection products, eight materializations, 2(L+1) cursor transition products,
and an exhausted-capacity guard. For R3/K7, L=1008 and there are1008 read slots
per coordinate. This is10,171,728 routing rows per coordinate,71,202,096 across
seven coordinates, before hashing, Fisher–Yates and norms. With materialized
cursor expressions of at most two terms, the routing nonzero upper estimate is
348,707,520. These are source counts, not allocated-matrix measurements.

The complete fragment is estimated at90–94M rows and60–75GiB builder RSS;
Vec allocation and other allocator details make the memory number approximate.
The builder has no count-only mode. No full allocation, full root/e/t/nonce
mutation control, or full unknown-layout walk was performed. They remain required
before promotion. A separately reviewed permutation routing implementation is
needed before attempting this composition within the resource budget; it must
retain all active rejected draws and first norm acceptance.

## Checkpoint and evidence

Evidence directory: `/private/tmp/terminal-context-20260922`. Temporary native
instrumentation is archived in `terminal-native-observer.patch` and
`terminal-sponge-observer.patch`; the latter is based on spongefish
`d2d190b1329d35ac9577438d05aed4f17a57b9f9`. The instrumented native backend is
f5f75335eae18241681fd24ca0a60fca8f0512af. The observer graph, lock and config are
`checkpoint-metadata.json`, `checkpoint-Cargo.lock`, and
`checkpoint-observer.toml`. These patches are fixture instrumentation only and
are not part of the production consumer dependency graph.

`checkpoint.json` records the full native Squeeze state after event1219:
64-byte chaining value, one generated block, no leftover bytes,64 consumed
bytes. The native replay matched all596 recorded challenge outputs, including
the final root. Event1221, the predecessor claim append, is still pending;
appending that same claim reaches this fragment's entry before t. This checkpoint
was produced by replaying observed messages, not by proving prior native
verification. No public unauthenticated checkpoint constructor was added.

Both observer build/run command JSON files pin source edits and executable hashes
and use the reviewed resource governor (4GiB sampled RSS,180s,12GiB disk floor).
The initial sampler observer compile failure remains preserved. Temporary
examples were archived and removed before handoff, and the public f5 lockfile
restored. Production source accepts only the live constrained transcript object
and consumes it; the omitted dead z append cannot accidentally be followed by a
challenge through this API.

Small constraint controls cover the logical12-bit nonce boundary (4095 passes,
4096 is R1CS-unsatisfied, unknown shape matches) and canonical field-byte linkage.
They do not establish full terminal-context equivalence. The complete terminal
assembler and sampler owners retain their independently reviewed narrower claims.

Compilation history: the first small-test compile exceeded the4GiB sampled
process-group limit (4,762,025,984 bytes); the single-job retry remained below it
(3,813,687,296 bytes) but reached180 seconds. Both governor results report no
remaining process-group members. The separately approved final compilation
window is600 seconds with the same memory/disk limits and one build job.
Checkpoint build passed in48.03s at3,279,323,136 sampled RSS bytes; replay passed
in2.16s at27,820,032 bytes. These are observer costs, not circuit/proof costs.

`public-metadata.json` confirms all14 native Akita crates resolve the public f5
Git revision, with exactly one local workspace `jolt-field`. No instrumented
path dependency remains in the consumer lockfile or normal Cargo invocation.

Final scoped checks passed (same source hashes recorded in the commands):

```text
CARGO_INCREMENTAL=0 CARGO_BUILD_JOBS=1 CARGO_TARGET_DIR=<taskroot>/wrapper/target
cargo nextest run --offline --locked --cargo-profile test -p jolt-r1cs -p jolt-akita --features jolt-akita/r1cs --lib -E 'test(logical_nonce_width) | test(canonical_bytes_centering)' --test-threads 1 --cargo-quiet
# 2 passed,77 filtered; small-nextest-600.command.json
cargo clippy --offline --locked --profile test -p jolt-r1cs -p jolt-akita --features jolt-akita/r1cs --all-targets -- -D warnings
# passed; clippy.command.json
cargo fmt --all --check
```

The full context's native-root/accepted-output R1CS parity, coherent root/e/t/
nonce/challenge controls and unknown-witness matrix comparison remain unverified.
Only the native checkpoint/sampler parity and small owner constraints above were
executed. This source checkpoint must not be presented as full verifier acceptance.
