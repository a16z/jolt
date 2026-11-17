# Recursion verifier guest

The recursion example executes Jolt's Akita verifier as a RISC-V guest: it
verifies an inner Akita, field-inline proof and reports the verdict. The cost of
interest is the guest's **total trace rows**, the length an outer proof of that
execution would have to cover, including setup and proof decoding. The working
target is a Fibonacci inner proof below 2^26 rows; this revision is 19,035,984
rows above it (see [Setup transports](#setup-transports)). This document
records the mechanisms the guest uses, the trust they rely on, and where the
rows go.

Only trace execution is implemented for the Akita guest. No outer recursive
proof of it is generated.

## Trust model

The guest distinguishes two kinds of input.

- **Proof data** (the inner proof and its I/O device) is untrusted and goes
  through fallible record decoding and the ordinary verifier. Malformed framing,
  noncanonical fields, and decoding failures return `0`. Each encoded record is
  limited to 16 MiB, with a conservative 16 MiB aggregate collection-storage
  budget during decoding; excess lengths reject before collection allocation.
  Raw setup bodies are borrowed within the checked stream bounds.
- **Setup data** is trusted. It is the verifier preprocessing, the Akita
  verifier key with its public matrix, a catalog view of the selected schedule
  row, and that row's terminal-matrix NTT cache. The guest takes the matrix as
  seed-derived without re-expanding it, and uses the NTT residues where they lie
  without a range pass.

Trusted setup data must be bound to the program identity. With `--embed` the
host bakes it into the guest image, so the ELF digest is the binding. In input
mode the host bakes a BLAKE2b-512 digest into the image at build time. It covers
the entire framed setup section: the preprocessing record, detached-body count,
and every detached body (key matrix, catalog view, and terminal NTT cache),
including their lengths and padding. The leading input-alignment record is
excluded. The guest checks this pin before decoding setup or constructing any
`TrustedBytes`; a missing digest or mismatch returns verdict `0`. Rebuilding
with a different setup changes the image identity, which the consumer must pin.
The preprocessing digest alone does not cover the detached payloads. Embedded mode is the configuration the row count
refers to.

In advice mode (`--advice`, Akita builds) the setup is trusted advice, and
its identity is a commitment rather than the image. An outer verifier opens
the trusted-advice region against the `trusted_advice_commitment` it is given,
so the consumer must pin the commitment to the published setup. Jolt checks
the region against whatever commitment is supplied, and a commitment the prover
chooses would void the trusted-setup assumptions above. The image bakes no
setup data, only its region capacities and record limits, so one image serves
any setup that fits them. The host computes the commitment with
`jolt_prover::akita::witness::commit_advice` over the region's bytes and
capacity, with the schedule artifacts the inner proofs use. An outer Akita
prover's `commit_trusted_advice` builds the same object from a setup on those
artifacts, and anyone can recompute it from the published setup bytes. The
Akita guest has no outer proof yet, so this binding is not exercised: the host
prints the commitment and the inner statements. Neither embedding nor a
commitment shows that the prepared key and cache were built correctly; each
binds the publisher's choice.

What the guest still checks on setup data:

- The catalog view repeats the full semantic audit of every row it carries and
  recomputes the complete catalog digest from the committed row identities. The
  transcript therefore binds the same setup identity as the full catalog.
- The NTT cache checks its header, its setup and schedule bindings, its
  geometry, its lengths, and its alignment.
- Field-inline readouts reject non-canonical integers.

## Recursive statement

The recursive statement is: under the setup bound by this guest image, the
postcard-decoded record slice contains a nonempty batch of valid inner
proof/device pairs. The count must be at least one and no bytes may follow the
last padded record within that slice. The entry point also requires every byte
from the end of the postcard envelope to the end of the input capacity to be
zero. The runtime exposes a zero-padded region, not the submitted input length,
so an explicit zero suffix is indistinguishable from unused capacity and is
allowed. A nonzero transport suffix, malformed envelope, or malformed record
returns `0`; acceptance returns `1`. The output is
still one `u32` and does not commit separately to the count or inner devices.
A consumer checking a particular computation must bind the runtime input and
inspect the intended count and devices; checking the verdict alone establishes
only that some nonempty batch verifies under the image's setup.

In advice mode the statement separates from the evidence. Under the setup bound
by the pinned commitment, every proof in the untrusted advice verifies against
the device at the same position in the public input. The input holds the
nonzero count and the devices, with the same envelope and zero-suffix rules.
The untrusted-advice slice must hold exactly one proof record per device; a
missing or extra record returns `0`. Bytes after an advice region's postcard
envelope are not read. The public I/O therefore binds the statements, not the
proof bytes. Moving the proofs to advice does not hide them: that needs a
zero-knowledge outer proof, and Akita is transparent.

## Record limits

The image builder runs the same bounded serde adapter over each supplied setup,
count, proof, and device record and logs both its encoded length and its charged
size. The byte cap is the next power of two of the largest record, with a 16 MiB
floor. The allocation-charge cap is the next power of two of twice the largest
charge, also with a 16 MiB floor. Both caps are constants in the generated guest;
untrusted runtime lengths cannot raise them. Detached setup bodies are borrowed
and are not decoded records. Embedded and advice setups remove the bytecode
vector before encoding, so using the full input-mode setup to size their cap is
conservative. The additional digest record is only 32 bytes, and advice mode's
fixed-width rows are a raw body that decodes to the rows the full record's
charge already covers. Each region's capacity, the input's and each advice
region's, is the next power of two of its stream length plus 26 bytes: the pad
record's prefix and at most seven pad bytes, postcard's length of at most ten
bytes, and one byte that keeps the framed stream strictly below capacity.

A trace of `2^20` rows does not bound the size of the program image, bytecode,
public I/O, or schedule catalog. There is consequently no universal largest
honest record at that trace length. In full-program mode the setup can dominate:
`JoltInstructionRow` contains at least an eight-byte address and a sixteen-byte
immediate, so a `2^20`-row bytecode vector alone incurs at least
`2 * 24 * 2^20 = 48 MiB` of charge before the nested struct/operand charges,
PC map, RAM image, and setup metadata. A `2^20`-word RAM vector adds
`2 * 8 * 2^20 = 16 MiB`. These terms alone exceed the former 16 MiB cap.
For comparison, a vector of `2^20` Fp128 values has 16 MiB of field bytes but
costs `2 * 16 * 2^20` for vector storage plus `2 * 16 * 2^20` for the fixed
byte-array decodes: 64 MiB of charge. This is a type-based estimate, not a claim
that sumcheck proofs contain one scalar per trace row.

Image-specific sizing avoids guessing a universal cap from trace length. It
admits the records supplied to the builder under the supported 64-bit host and
guest layouts, including the input-mode setup; other record shapes must fit the
pinned limits or need a newly sized image. The builder reserves at least eight
times the charge cap for the heap, accounting for the resident setup, proof,
and device plus growth and verifier work. The charge is not a peak-heap proof:
custom deserializers and backend verification can allocate outside the adapter,
and transient vector growth can exceed its factor-of-two charge. Runtime peak
memory and acceptance at large sizes still require Linux validation.

## Mechanisms

### Field arithmetic on the field-inline ISA

The field-inline unit computes in the one field the outer proof fixes, so the
guest selects it: `field-inline-guest-fp128` routes the Fp128 protocol prime
(Akita) and `field-inline-guest-bn254` routes BN254 `Fr` (Dory); every other
field keeps its software arithmetic. The routed field's dot products, row
dots, product sums, signed sums, multiplications, multiply-adds, and
inversions run through field-inline instructions (`field_inline.rs`).

- **Operand ingress** clears a field register (`FIELD_LOAD_IMM 0`), then folds
  the operand's words in with `FIELD_LOAD_ACCUMULATE_FROM_MEMORY` (two for
  Fp128, four for Fr).
- **Readout** emits one range-bound `FIELD_ADVICE_LIMB` per limb and closes with
  `FIELD_ASSERT_ZERO` on the final quotient. The wrapper then rejects an integer
  at or above the modulus.
- **Modulus binding.** That range check assumes the unit computes modulo the
  selected field. jolt-sdk's std runtime therefore calls
  `jolt_field::bind_field_inline_modulus` once at boot: it accumulates the
  field's modulus into a cleared register and asserts zero, so a guest built for
  one field has no execution under a tracer for the other, instead of computing
  wrong results that still read out canonically.
- **Accumulation** stays register-resident. A 64-term dot costs about 520 rows,
  about 8 per element: two 3-row operand loads, one multiply, and one add.
- **Shared operands** let the weighted-rows kernel load each power once per
  block of five rows (5 rows per element and row). `dot_rows` does the same
  for several rows against one shared vector, and `sum_of_products` forms
  `K`-factor products in registers.
- **Signed sums** (`Field::signed_sum`) keep a sum of ± elements in a field
  register: 4 rows per term, against about 24 for a reduced software add.
  Isolated additions stay in software, where two-limb arithmetic wins. The
  sum's iterator runs between terms, so the accumulating kernels are marked
  lent until the readout and take their software path if the iterator calls
  them.

### Hash and NTT inlines

- **Blake2b inline** (`jolt-inlines-blake2`, with a `digest` adapter). The Jolt
  transcript, descriptor digests, layout digests, and the preprocessing digest
  hash through it, with the same bytes as the `blake2` crate. Akita's
  transcript is spongefish's own `Hash` sponge over that adapter.
- **Keccak inline** for Akita's SHAKE challenge sampler (`keccak-inline`).
- **64-point i32 NTT and six-product pointwise dot** (`jolt-inlines-ntt`).
  These expand into existing proved integer instructions, with no advice and no
  new constraints. Akita's `ntt-inline` feature dispatches the 64-dimension CRT
  transforms and lazy Montgomery dots to them. The signed-digit conversion is
  fused into the twist (`psi^i · R^2` table). Butterfly sums are reduced by a
  Montgomery product with the stage's first twiddle, the Montgomery form of 1
  (5 rows against a 7-row conditional reduction).

### Prepared verifier setup

`AkitaVerifierSetup::prepare_verifier(row)` (host side) carries a
`PreparedVerifier` for the single schedule row the proof selected:

- the serialized backend key, which saves the guest the seed expansion of the
  public matrix;
- a catalog **verifier view** carrying only that row's parameters, which
  replaces JSON parsing and the audit of every row;
- the scalar Q128 **terminal NTT cache**, which saves the guest the matrix NTT.

The guest builds `AkitaVerifier::for_selection` from these. It admits only that
row and installs the cache as `TrustedTerminalCache::View`. A setup prepared
for one row verifies only proofs for that row of its flavor.

`detach_prepared_payloads` / `attach_prepared_payloads` move the multi-megabyte
payloads out of the bincode record. Each detached body is self-aligning: a skew
byte puts the key's coefficients on an 8-byte boundary. The guest then views
them where they lie, in its image, its input, or its trusted advice, instead of
copying them.

With `--embed` the host also:

- compiles the bytecode rows into a static table;
- bakes the preprocessing digest computed over the complete program next to the
  row-less record. The guest assembles the preprocessing from the wire form, the
  rows, and that digest. Decoding the ordinary record would re-encode and hash
  the whole program, and would hash the wrong (row-less) program.

With `--advice` the trusted advice carries the same row-less record and digest,
followed by the rows in a fixed-width layout instead of a static table: four
little-endian words per row (`guest/src/compiled_rows.rs`). The guest views
them as words in place and rebuilds each row without bincode. The host decodes
its own encoding with the guest's decoder before tracing.

### Guest runtime

- An O(1) size-class allocator for std guests.
- No custom `mem*` routines. musl's `memcpy`, `memset`, and `memcmp` move
  data partly in sub-word accesses, each a multi-row sequence, so hot paths
  avoid them:
  - The Blake2b hasher packs input into message words.
  - The size-class allocator's `realloc` moves whole words.
  - The jolt CLI raises LLVM's inline-store limits (`-max-store-memcpy=32`,
    `-max-store-memset=32`), so constant-size copies and fills compile to
    word stores.
- `serde_bytes` on every proof and setup byte payload.
- 8-byte-aligned guest record framing.
- `write(2)` to stdout and stderr and `clock_gettime(2)` answered in the trap
  handler: writes go to the host console in one host call each, and every
  clock reads as a zero timestamp.

### Verifier algebra reshaped for guest costs

- **Derived leaves for batching powers** (`ChallengePow`, `GammaPow`): one
  factor per term instead of `k` repeated challenge factors.
- **Direct output-claim value walks**, and **indexed claim-family lookups**
  instead of scans.
- **Factored Spartan outer check.** The clear stage-1 output check evaluates
  `JoltSpartanOuterRemainder::expected_output_claim` (the factored R1CS form)
  rather than the expanded quadratic expression. BlindFold still lowers the
  symbolic expression. Both read their coefficients from
  `JoltSpartanOuterRemainder`, and a test pins their equality on the RV64 and
  field-inline shapes. This is the one place where clear verification does not
  evaluate the jolt-claims expression directly. Reviewers should weigh it
  against the expression-ownership invariant.
- **Bytecode read-RAF:** stage values are folded against the address eq table
  by flag class: rows sharing a decoded flag set share one gamma bucket, and
  register operands accumulate into per-register buckets that meet the eq
  tables once. Expression storage is reused when multiplying by a monomial,
  and opening ids are word-aligned.
- **Outer-remainder opening set.** `expected_output_openings` reads the stage-1
  remainder's openings off its two linear factors instead of expanding
  `tau·Az·Bz` into thousands of terms; a test pins the result to the expanded
  form.
- **Transcript batching.** Sumcheck round coefficients and each batch of opening
  claims absorb as one labeled message (count plus big-endian values). This is
  a transcript format change for every proof.
- **Sparse MLEs** evaluate contiguous runs from one low eq table and one high
  factor per aligned block.
- **Akita terminal relations** are folded coefficient-major: the base-field
  consistency fold of `z` is one dot product per ring coefficient, and sparse
  challenge products are one signed sum per output coefficient.
- **Sub-word memory.** A Jolt guest expands byte, half, and word stores and
  loads into multi-row sequences (a `sw` costs about 9 rows), so hot tables use
  word-sized slots, NTT inputs are written with doubleword stores, and proof
  atoms encode into an inline word-aligned buffer instead of a heap vector.
- **Akita guest paths:** word-window Golomb–Rice decoding, merged equality
  recurrences, factor-cached and prepared residual tensor contractions,
  certified first-octant trig enclosures, batched compression-relation sums,
  direct SHAKE rate-lane sampling, and weighted-row batching in the setup scans.

## Measurements

Fibonacci, one inner proof, `--features akita,field-inline,ntt-inline`,
`trace --embed`, release guest, `RAYON_NUM_THREADS=1`. Each row is the
cumulative total after its change. From the rebase onto `main` through the
fast-mem rows, every row traces one freshly generated proof with the pinned
companion; the last row traced a fresh proof when it was recorded. Later merges
of `main` are not itemized here; [Setup transports](#setup-transports) has
totals at this revision.

| Build | Total rows |
| --- | ---: |
| 2^26 line (`archive/fast-recursion-2pow26-local-20260914`, old ISA and Akita) | 66,843,409 |
| Port onto `main` with embedded digest | 83,677,076 |
| + read-RAF fold by flag class | 80,464,758 |
| + `dot_rows` and `sum_of_products4` compression kernels | 79,386,540 |
| + Keccak inline for SHAKE sampling | 78,797,012 |
| + batched transcript absorption | 76,534,550 |
| + NTT Montgomery-one reduction, inline mask-prefix sponge | 75,102,679 |
| + word-window Golomb decode, coefficient-major terminal `z` fold | 72,389,735 |
| + outer-remainder opening set, signed-sum challenge products | 69,361,519 |
| + in-place digest transcript, blocked sparse MLE | 68,530,020 |
| + doubleword NTT input stores, word-sized flag-class slots | 67,738,580 |
| + inline proof-atom encoding | 66,554,373 |
| Fresh proof before rebasing onto `main` (64,433,863 verification cycles) | 66,273,665 |
| Rebase onto `main`, Akita `22530c7b`, fresh proof | 66,254,853 |
| + direct read-RAF and digit-zero output opening sets | 65,441,486 |
| − word-wise `memcpy`, `memset`, and `memcmp` overrides | 69,059,130 |
| + inline-store limits, Blake2b word packing, word-copy `realloc` | 67,622,320 |
| Merge `main`: native Akita trace batching (#2012), Akita `d98400c5` (#175), fresh proof | 78,896,208 |
| Akita review revisions of LayerZero-Labs/akita#186-#190 on Akita `main` `401e7972`, spongefish's own Blake2b sponge, lent-accumulator `signed_sum`, fresh proof | 79,348,402 |
| Merge `main`: Akita `318c3221` (#193), unified trace rows (#1996), fallible chunked decode and live-entry residual tables, fresh proof (77,245,072 verification cycles) | 79,335,454 |
| **Merge `main`: Akita `cf95fd05` (BlockDigits dense commit, #236-#242), whole-table bytecode (#2030), fresh proof** (76,947,621 verification cycles) | **79,032,888** |

The first row was rebuilt and re-measured from its archived sources and
reproduces its recorded numbers exactly. At the last row's revision, input
mode, which then had no setup pin, accepted at 86,461,879 rows, and a proof
with one bit flipped in its Akita opening proof was rejected after 60,136,155
rows, inside the PCS verifier.

The merge's increase is inside the Akita trace-batch opening
(`verify_trace_batch`, 56.07M to 67.86M rows), almost all of it in the root
fold (`verify_root`, 14.56M to 24.46M rows): its relation-matrix evaluation
and direct setup-contribution scan run over the natively batched trace
columns. The terminal checks are unchanged.

The last row's 452,194-row increase nets two measured effects, besides the
boot-time modulus binding. Replacing the
hand-written transcript sponge, which reused each phase's compressed mask block,
with spongefish's sponge over the inline adapter costs about 1.16M rows, and the
remaining review revisions save about 0.71M. The `signed_sum` accumulator guard
costs about 9,000 rows. Recovering the mask compressions through a digest that
recognizes the mask blocks cost more rows (0.36M) than it saved, because the
128-byte check runs bytewise in the guest.

### Setup transports

Fibonacci, one inner proof, the same features, at this revision: one freshly
generated proof, traced in each transport. Setup decode is the
`deserialize preprocessing` span. The other spans agree across transports to
within 0.1%: about 2.17M rows of proof and device decoding and 82.9M rows of
verification. Setup bytes count the setup stream before region framing; input
bytes are the framed public input.

| Transport | Setup bytes | Setup location | Input bytes | Setup decode | Total rows |
| --- | ---: | --- | ---: | ---: | ---: |
| Embedded (`--embed`) | 15,507,104 | image | 93,768 | 806,351 | 86,144,848 |
| Input | 15,578,168 | input | 15,671,936 | 11,905,787 | 241,797,691 |
| Advice (`--advice`) | 15,638,184 | trusted advice | 144 | 1,437,934 | 86,503,398 |
| Advice, input mode's setup record | 15,578,168 | trusted advice | 144 | 11,905,988 | 97,020,145 |

Input mode spends 144.8M rows outside the spans, almost all of it to
authenticate the setup: the BLAKE2b-512 pin over the 15.6 MB section, plus the
zero-suffix scan of its 16 MiB input region at about 7 rows per byte. Advice
mode needs no pin, because the commitment binds the setup, and its input
region shrinks to 256 bytes. It lands 358,550 rows (0.42%) above embedded
mode. Decoding the 4,096 fixed-width rows costs 631,583 more rows than copying
embedded mode's static table, and the smaller input region saves about 260,000
rows of zero-suffix scan. The last row puts input mode's setup record in the
trusted advice instead, measured with a temporary patch: decoding the rows with
bincode and re-hashing the program costs 10.5M rows more than the compiled
layout.

A proof with one bit flipped in its Akita opening is rejected after 85,820,939
rows embedded and 86,179,483 rows in advice mode, where the untrusted advice
holds 93,640 bytes of proof in a 128 KiB region.

An outer Akita proof of the advice-mode trace would commit a 2^27-row trace:
44 one-hot columns of 31 variables at K = 16, a 2^21-row recursion bytecode, a
2^21-word trusted-advice group, and the 14-variable (2^14-word) floor for the
untrusted group. At this revision, grouped schedule planning
(`AkitaScheme::verifier_setup_from_params` on the parameters
`preprocess_full_with_advice` builds) admits both advice groups, alone or
together, with full program preprocessing and the default single-chunk witness
profile. Two shapes are not admitted. Committed program preprocessing admits
no advice group: a batch that carries the full-width field-increment group
allows at most three bounded groups, and the two direct program groups and the
fused increment group already take them. The 8-chunk, 2-level witness profile
has no catalog row for the one-hot final shape at all, with or without advice.

## Levers not taken here

- **Load into a cleared register.** Every field operand load would drop from
  3 rows to 2 if the ISA had a non-accumulating memory load, which the
  2^26-era ISA had. This touches every field-inline operand, so it is the
  largest remaining lever. It is a field-inline protocol change.
- **Cheaper sub-word memory.** Tiers 1 and 2 of
  [byte-addressable-memory.md](byte-addressable-memory.md) lower the row cost
  of byte, half, and word accesses, which musl's `mem*` routines and proof
  decoding still issue in bulk. It is a protocol change, tracked in #2013.
- **Horner or multi-output dots for geometric powers.** The compression-matrix
  columns evaluate `Σ c_j α^j` with shared powers; a Horner kernel removes the
  power loads (about 1.7M rows).
- **Compression-event layout.** `CompressionRelationWeights::evaluate_at_point`
  spends about 120 of its 150 rows per event on per-event bookkeeping. A
  layout that groups events by alpha range at build time would let the
  evaluation run as plain dot products.
- **Lazy NTT reduction or a merged twist.** Under the inline's contract (any
  prime below 2^30, operands in (-p, p)) any skipped reduction leaves the
  final stage above p, and a merged-twist Cooley–Tukey butterfly needs two
  reductions where Gentleman–Sande needs one, so neither saves rows without a
  narrower contract.
- **Proof field decoding.** Fp128 elements arrive as 16 separate bytes through
  serde's tuple path, one call per byte, and the decoder assembles two 64-bit
  words from them. Decoding an element as `u64` or `u128` integers or as a
  byte string takes fewer calls, but bincode's standard configuration writes
  those integers at variable length and prefixes a byte string with its
  length, so either changes the wire format.
- **Measured and rejected:** a single-pass operator-norm accumulation (more
  rows than the chunked kernel) and a merged per-event cache in the
  compression evaluation (65K rows, not worth the code).

## Reproduction

```bash
export RAYON_NUM_THREADS=1 RUST_MIN_STACK=268435456 CARGO_BUILD_JOBS=1
cargo run --release -p recursion --features akita,field-inline,ntt-inline -- \
  generate --example fibonacci --workdir /tmp/recursion
cargo run --release -p recursion --features akita,field-inline,ntt-inline -- \
  trace --embed --example fibonacci --workdir /tmp/recursion
cargo run --release -p recursion --features akita,field-inline,ntt-inline -- \
  trace --advice --example fibonacci --workdir /tmp/recursion
```

No CI job runs this flow. It is run by hand, together with the check that the
guest rejects a proof with one tampered opening.
