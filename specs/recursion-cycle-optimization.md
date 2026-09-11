# Recursion cycle optimization: upstream catch-up

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

## Validation status

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

## Remaining cost

The retained profile attributes about 17.77M rows to forward NTTs and 14.37M to
integer NTT matrix-vector multiplication. Memory copying accounts for another
10.02M rows across callers. Reaching 50M has not been demonstrated; even removing
the measured NTT/matrix cost would leave roughly 92M verification cycles. A new
provable arithmetic instruction or a protocol change needs a separate design
and soundness argument, with complete guest proof coverage from its first use.
