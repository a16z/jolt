# Recursion cycle optimization

Historical experiment record from the original recursion branch. Its source pins,
proofs, and measurements do not validate the refreshed Blake-only epoch-6 stack.
Current integration and performance require new exact-source validation.

The retained **embedded-setup** guest verifies the frozen Fibonacci proof in
**65,788,119 verification cycles /66,843,409 total rows**, with an exact fresh-build
repeat and output 1. The full trace is **265,455 rows below 2^26**. Compared with
the preceding 75,915,308-row checkpoint, this campaign saves 9,071,899 rows
(11.95%). These are trace measurements; the full outer recursion proof has not
been generated. The optimizations remain on the guest branches, separate from
field-inline PR #1808. The 50M target remains unachieved.

The Akita recursion guest now checks that field-inline limb advice is canonical.
Memory-sourced field loads also retain their ordinary address and destination
register IDs in compact proof rows. The expanded `eqpoly-field-guest` exercises
both memory loads, limb advice, and a nontrivial inversion through the composed
prover/verifier path.

`FIELD_ADVICE_LIMB` replaces the misleading `FIELD_SPLIT_LOW` name; its opcode and
ordinal are unchanged. Its constraint binds a residue, not a canonical integer.
The BN254 and Fp128 wrappers reject readout integers at or above the modulus.
See [the protocol contract](field-inline-protocol.md#memory-sourced-loads-and-limb-readout).
Previously compiled recursion guests need rebuilding.

## 2^26 checkpoint (2026-09-14)

The target for this campaign is the **whole expanded trace**, including setup and
proof decoding. Its primary metric supersedes the earlier rule that both total
rows and the verification region must be nonincreasing. In particular, compiling
immutable bytecode into the existing `--embed` mode saves 2.43M total rows while
adding 87,955 verification cycles. The same instruction rows are reconstructed;
proof bytes, setup bindings, and verifier checks remain unchanged.

Thirteen retained optimizations are separate commits: eight in Jolt and five in the
Akita companion. The initial checkpoint is Jolt `12a3cd05f`, companion `17a911d6f`;
the final implementation heads are Jolt `f37f72f81`, companion `37d8f3548`.

| Optimization | Commit | Total rows | Rows saved |
| --- | --- | ---: | ---: |
| compiled embedded bytecode | `fef5d8240` | 73,488,357 | 2,426,951 |
| preweighted register equality | `f55b0c5de` | 73,106,344 | 382,013 |
| field load registers | `5f7f4221c` | 71,673,993 | 1,432,351 |
| in-place equality merge | `bf8357db9` | 71,489,550 | 184,443 |
| Montgomery initialization guards | `e9771da09` | 69,178,001 | 2,311,549 |
| cached bit-pair factors | `9f4e0bb09` | 69,081,503 | 96,498 |
| field readout registers | `423550ae8` | 68,669,341 | 412,162 |
| dot-loop unrolling | `05fb41c01` | 68,528,069 | 141,272 |
| aligned Rice reads | `a0ca3eac6` | 68,120,986 | 407,083 |
| stack transcript frames | `ba4825774` | 67,919,526 | 201,460 |
| direct SHAKE lanes | `69181ccd4` | 67,752,822 | 166,704 |
| prepared residual tensors | `37d8f3548` | 66,855,759 | 897,063 |
| direct ordinary-setup decode | `f37f72f81` | 66,843,409 | 12,350 |

The mode distinction matters: the verification region is below 2^26 in both
modes, while only embedded setup currently fits the entire trace into 2^26 rows.
Both modes return 1 on the frozen proof. The final routing change saves 10,155
rows in ordinary input mode as well as the embedded saving shown above.

| Final mode | Verification cycles | Total rows |
| --- | ---: | ---: |
| Embedded setup | 65,788,119 | 66,843,409 |
| Ordinary input | 65,788,163 | 71,073,184 |

The source changes use the existing proved instruction families. Canonical field
readout checks remain in place. SHAKE challenge bytes and transcript framing are
unchanged; transcript appends retain one absorb call, including for Poseidon.
Residual tensor tables are constructed only after the original work bound passes;
empty destination axes return without allocating a residual expansion.

Rejected experiments: binary-searching the compression cache (+378,128 rows),
field-inline `mul_add` (+113,337), and field-inline standalone addition/subtraction
(+752,911). Their patches and raw logs are retained in the scratch campaign.

Reproduction uses `/private/tmp/akita-2pow26`, with fixed host evaluator
`/private/tmp/guest-optimization-campaign/29-harness` and the frozen proof stream in
`/private/tmp/guest-optimization-campaign/22-input`. Replaying the original ELF
under this evaluator reproduces **72,342,179 /75,915,308 /output 1** exactly.
The final guest also repeats exactly after a fresh build.

```bash
RUST_LOG=info JOLT_BACKTRACE=1 RAYON_NUM_THREADS=1 RUST_MIN_STACK=268435456 \
  JOLT_PATH=/private/tmp/jolt-field-inline-cli/bin/jolt \
  /private/tmp/guest-optimization-campaign/29-harness trace \
  --example fibonacci \
  --workdir /private/tmp/guest-optimization-campaign/22-input --embed
```

Use `trace` without `--disk`: this trace-only path already streams rows to disk.
The frozen harness contains the same embedding generator now committed in Jolt
and the retained NTT expansion registry. A newly built `recursion` binary with
`akita,field-inline,ntt-inline` features provides the same command.

Pinned SHA-256 values:

- Host evaluator: `b491719bdca2f28e72111a58a5345d38ccd155a89872b73ba2e724816c9931fd`.
- Frozen input: `7741abc8f433b972ef68af9b8b3804840c74740537d403f93bf427ed218b53fa`.
- Embedded guest ELF: `d45c2f391cc40b6aa7bc5fda25e2005731d9118fae27df43ab338dddda5946ac`.

Validation completed during the campaign includes 30 bytecode claim tests,
44 equality/tensor tests, 15 Rice decoding tests, 50 SHAKE challenge tests, and
62 transcript tests. The final Akita field-inline fixture suite passes all 26
tests; the final clear and ZK field-inline fixture suites pass 28 and 18 tests,
respectively, including actual proof acceptance and tamper rejection. Final Clippy
passes the full standard and ZK workspaces plus the Akita and ZK field-inline
lanes. Legacy muldiv passes all three Akita cases and all three ZK cases.
Formatting and the style-invariant guard pass.

The comparison fixes the Fibonacci proof and compiler configuration; it is not a
cycle bound for other programs or compiler versions. The full outer recursion
proof remains unrun. See the scratch manifest and logs for the validation record.

## Source alignment

- Jolt includes upstream main `95c898d3d2bbcc178fd8b18603662fce69cfa37d`.
- The Akita companion includes upstream main
  `38dbf031013a2b8f7ff93252e190dac2b3353706`, including the dense-shape,
  compressed-round, input-validation, and verifier-cache fixes.
- Companion checkpoint `e5488fd50` restores binary schedule catalogs through
  upstream's `ValidatedScheduleCatalog`, preserving semantic validation and
  policy/family binding. Prepared NTT caches and borrowed static storage remain
  in use. Jolt's three catalogs were regenerated with the current planner.
- Jolt catch-up checkpoint: `17ff06f9e`; readout/compact-row repair: `19c63772c`;
  booleanity gamma powers: `84e49f893`; opening-ID collection: `47d3babd4`.
  Final source and CI cleanup checkpoint: `80aa2ba47`.

The companion is consumed through the Jolt workspace path patches. Its standalone
workspace still cannot resolve its published `jolt-inlines-blake2` pin with the
requested `digest` feature; standalone companion CI is not established by these
checks.

## Measurements

One Fibonacci inner proof, Akita, field-inline, release guest, one Rayon thread.
Each candidate uses the same proof input; the verification span excludes input
and preprocessing deserialization. Total rows include those phases.

| Version | Verification cycles | Total rows | Guest output |
| --- | ---: | ---: | ---: |
| Upstream catch-up | 126,093,856 | 131,393,235 | 1 |
| Canonical readout checks | 128,099,104 | 133,389,005 | 1 |
| Derived booleanity gamma powers | 125,310,994 | 130,619,132 | 1 |
| Streaming opening-ID collection | 123,843,430 | 129,216,363 | 1 |

The historical catch-up baseline lacks the readout repair. The retained
optimizations save 4,255,674 verification cycles against the repaired baseline.
A unit-coefficient multiplication fast path was rejected: it saved only 56,420
verification cycles but increased total rows by 14,331.

These are **trace-only recursion measurements**, not timings or proof generation
for the entire outer recursion trace. Small composed field-inline e2e suites
exercise actual proof generation and verification. The measured proof-input
SHA-256 is `b1b765ca611b7bdb91764c1d6c10dcaa77cf10fa7d3ebc8d239f02a537046eaa`.

Reproduction commands and PC-profile reporting are in
[the recursion README](../examples/recursion/README.md). Use
`RAYON_NUM_THREADS=1 RUST_MIN_STACK=268435456 CARGO_BUILD_JOBS=1` and the same
saved proof input when comparing candidates.

## NTT inline experiment (2026-09-11)

The guest-optimization branch now includes refreshed field-inline PR #1808
(`a48725375`) through merge `4d7dbff30`. This experiment is separate from that PR.
The new `jolt-inlines-ntt` crate expands a 64-point i32 Montgomery forward NTT
into 4,456 existing proved integer rows. It retains all coefficients in virtual
registers across the butterfly stages and introduces no advice or new proof
constraints. Akita's `ntt-inline` feature opts into the guest dispatch
(companion commit `15ac1a447`).

The comparison uses the proof hash above and identical release, field-inline,
fast-allocator, Blake2-inline, and PC-profiling settings. One Rayon thread and
`RUST_MIN_STACK=268435456 CARGO_BUILD_JOBS=1` were used throughout.

| Version | Verification cycles | Total rows | Guest output |
| --- | ---: | ---: | ---: |
| Refreshed field-inline base, NTT off | 123,936,530 | 129,306,724 | 1 |
| Initial scalar alignment fallback (rejected) | 128,308,526 | 133,678,858 | 1 |
| NTT with aligned stack buffers (two identical runs) | 114,181,432 | 119,551,764 | 1 |

The retained implementation saves 9,755,098 verification cycles (7.87%) and
9,754,960 total rows. Akita's arrays did not meet the paired-access alignment
guard in the first candidate, so all calls used its slower scalar fallback.
The repaired RISC-V path copies misaligned arrays to aligned stack buffers;
other targets retain the portable transform. The added copies are included
in the reported totals. Symbol attribution now folds the inline into its
matrix-vector caller, so that caller's larger profile bucket is not a regression.

The standalone NTT example generates and verifies a real Dory-backed Jolt proof,
checks its output against a direct DFT checksum, and deliberately supplies a
4-byte-aligned table. The full Akita recursion figures remain trace-only;
the entire outer recursion proof was not generated. No 50M result or transfer
to other inner workloads is claimed.

Validation for this change: 43 NTT/expansion-fixture/ISA-profile tests, standard
and ZK workspace Clippy, the Akita/field-inline/NTT Clippy lane, formatting and
style guards, and the companion's 84 Python checks. The new companion dependency
requires Jolt's local path patch until the NTT crate is published; standalone
companion dependency checks are not established.

Commands and arithmetic contract: [NTT inline README](../jolt-inlines/ntt/README.md).
Raw logs, frozen proof, harnesses, ELFs, and profiles are retained locally in
`/private/tmp/ntt-inline-campaign/`. The retained guest ELF SHA-256 is
`1ab743cee2505fb68090a10314d9d0ada52a9744469b951739a4754f38e6ea30`.

## Prior catch-up validation

All 21 Clippy matrix configurations passed, including workspace `host` and
`host,zk`, recursion, field-inline, profiling/allocative, and fixture lanes.
Formatting and the style-invariant guard against `origin/main` passed.

All 20 nextest configurations passed with one build job and one test thread:

| Coverage | Passing configurations |
| --- | --- |
| Claims, kernels, verifier | Default, ZK, Akita, and each with field-inline |
| Field-inline core | Also witness, R1CS, program, RISC-V, lookup tables |
| Modular prover fixtures | Clear, ZK, field-inline clear, field-inline ZK, field-inline Akita |
| Verifier fixtures | Field-inline Dory (161 tests), field-inline Akita (111 tests) |
| Legacy muldiv | `host`, `host,zk`, `host,akita` (three tests each) |
| Catalogs | Default (8 tests), field-inline (12 tests) |
| Field wrappers and tracer | 149 field tests, 6 targeted field-inline tracer tests |

The catalog suite exposed a stale assumption that every field-inline build
commits a limb group. Inactive traces commit none; the corrected test enumerates
all seven active/inactive advice shapes. Its targeted Clippy check and full
catalog rerun passed. Existing feature-disabled, ignored, and filter-excluded
tests remain excluded; the catalog regeneration itself was run separately.

Raw commands, per-lane logs, guest ELFs, profiles, and a hash manifest are saved
locally under `/private/tmp/recursion-port/final`. Results files retain initial
failures and their successful reruns; the last result for each lane is final.
Companion checks also passed formatting, artifact/dependency guards through
Jolt, and 84 Python tests. The standalone dependency limitation above remains.

## Guest optimization campaign

The following changes remain separate commits on `feat/fast-recursion` and the
companion's `feat/fast-recursion-companion`. They are not part of field-inline
PR #1808. Measurements use the frozen proof above. Apply or revert paired Jolt/
companion changes together; later mechanisms can depend on earlier APIs. Values
are cumulative, so the gains need not add in a different order.

| Mechanism | Commit (Jolt / companion) | Verification cycles | Total rows |
| --- | --- | ---: | ---: |
| Initial NTT inline | `d6b8f27ce` / `15ac1a447` | 114,181,432 | 119,551,764 |
| Native Rice bit decoder | companion `99308ccbb` | 113,964,328 | 119,334,660 |
| Aligned NTT tables | companion `0605c41db` | 112,775,250 | 118,151,977 |
| Reuse monomial expression storage | `b7b94220f` | 112,408,462 | 117,778,575 |
| Stream bytecode evaluation rows | `18d38fbf6` | 112,014,074 | 117,384,380 |
| Inline equality-window lookup | companion `5a29719df` | 111,725,705 | 117,096,011 |
| Hash full input blocks directly | `c2c4aeb88` | 111,607,322 | 116,977,511 |
| Reuse certified first-octant enclosures | companion `30d472938` | 110,372,005 | 115,742,194 |
| Batch compression scalar dots | companion `e7daf0730` | 110,307,871 | 115,639,353 |
| Factor residual tensor weights | companion `55c68452c` | 109,532,605 | 114,864,225 |
| Six-product lazy Montgomery dot | companion `f951ea400` | 105,918,450 | 111,250,070 |
| Proved pointwise inline | `05d5e6a96` / `4c13f29a1` | 102,240,164 | 107,571,821 |
| Compress aligned streaming hash buffer directly | `a0a35475d` | 101,717,438 | 107,049,058 |
| Fuse signed-digit conversion into NTT twist | companion `67b793dcb` | 98,352,632 | 103,721,026 |
| Resolve indexed claim families directly | `f7dbb3a98` | 97,084,717 | 100,691,490 |
| Rice unary-prefix word table | companion `d85fbcd71` | 96,840,163 | 100,446,936 |
| Use canonical factored Spartan output check | `f7d419d35` | 92,170,363 | 95,763,469 |
| Remove redundant NTT sign-extension rows | `40d7b59ca` | 91,727,611 | 95,320,717 |
| Inline hash finalization to avoid state movement | `d8c1cb766` | 90,571,969 | 94,155,100 |
| Audit only selected catalog rows, preserving catalog identity | `f3eac4f54` / `0b2fe8e3d` | 76,946,007 | 80,515,127 |
| Copy complete symbolic factors | `1eec24402` | 73,425,623 | 76,998,752 |
| Specialize compression-event insertion | companion `e9f8d2a6a` | 72,735,395 | 76,308,524 |
| Shorten final NTT canonicalization | `d44ea8b47` | 72,342,179 | 75,915,308 |

Every retained trace returned output 1. The pointwise inline depends on the portable
lazy-dot arithmetic, and its guest dispatch requires the paired companion commit.
The sign-extension change requires a rebuilt host expansion registry; replaying the
same ELF first with the old registry reproduced the baseline, then removed exactly
442,752 rows with the new registry. The later final-canonicalization change removes another 393,216 rows on the same
ELF and passes the small Dory proof again. NTT and pointwise expansions now use 3,944 and
3,363 existing proved integer rows respectively. The full small Dory example passed
again after that change; these are not new unconstrained host operations.

Rejected trials remain in the local experiment ledger: inactive-field-row skipping
had no effect; stack-batching setup rows added 108,482 verification cycles; caching
repeated bytecode-tail rows added 67,427. Two geometric compression-weight variants added 430,704 and 55,699 verification
cycles. Reusing the RHS for constant expression multiplication added 1,538 despite
reducing total rows. An earlier supported quotient-free cutover added 764,959; see
[the controlled experiment](recursion-quotient-cutover.md). Compiled embedded bytecode reduced total rows by 2,428,932 but increased
verification cycles by 87,954; it was also rejected under the two-metric gate.
Forcing the two ring shift add/sub loops to inline added 54,146 cycles; its
18 algebra tests and Clippy passed, but the patch was reverted.
Their patches were removed. Each retained
mechanism has targeted nextest and Clippy evidence. The combined validation passed
all 21 Clippy configurations and 16 nextest configurations: six core modes, five
modular prover fixture modes, two verifier fixture modes, and legacy muldiv in
standard, ZK, and Akita modes. Formatting and the style-invariant guard passed.
The NTT and pointwise example also generated and verified a complete small Dory
proof after the final expansion change. Whole outer recursion proving remains
unmeasured.

That campaign used `/private/tmp/guest-optimization-campaign/28-harness`
(SHA-256 `dd4012ee6b97552f2e621a0b7f2653f9cf66ec716f401fd46e32888476e8623f`).
Logs, rejected patches, captured ELFs, profiles, and per-trial measurements are in
that directory's `STATE.md` and `events.jsonl`. Earlier harnesses retain the old
inline expansion lengths and cannot be used for current comparisons.

## Previous checkpoint: retained artifacts and remaining costs

Selected-row catalog loading is retained; see
[its contract and security argument](recursion-catalog-view.md). It preserves the
full catalog commitment and the original 81,560-byte proof/device section while
auditing every available row. The resulting stream is 6,211,264 bytes, SHA-256
`7741abc8f433b972ef68af9b8b3804840c74740537d403f93bf427ed218b53fa`.
The original full-catalog stream remains available for rollback.

The later symbolized profile still contains substantial
NTT/conversion, pointwise arithmetic, field dot products, bytecode claim evaluation,
expression copying, setup scans, ring shifts, and input decoding. Symbol buckets
include inline expansions and must not be added again to their callers. Reaching
50M remains unverified. Earlier quotient cutovers and new arithmetic instructions
require separate protocol/proof work; no savings from them are assumed here.

The previous checkpoint guest rebuild reproduced 72,342,179 verification cycles and 75,915,308
total rows. ELF SHA-256:
`fdb5f53b72415ce35fb21e0a8aad3d8b0463eefa374c3a3d1c0e024ee7a52d06`.
The largest exclusive PC-symbol regions in that full trace are:

| Region | Rows |
| --- | ---: |
| Signed-digit conversion and forward NTT | 7,078,400 |
| Field dot kernel | 3,931,980 |
| Shared `memcpy` | 3,809,570 |
| Bytecode read-RAF evaluation | 3,127,463 |
| Weighted base-row dot | 2,938,715 |
| Compression-weight MLE evaluation | 2,783,371 |
| NTT matrix-vector product | 2,712,027 |
| Program preprocessing deserialization | 2,649,774 |
| Rice decoding | 2,240,501 |

These are symbol-attributed rows, not complete logical-component costs or predicted
savings. Shared helpers have their own attribution. Profile and symbol report:
`/private/tmp/guest-optimization-campaign/final-retained-{profile,symbols}.txt`.
