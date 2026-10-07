# Recursion verifier guest

The recursion example executes Jolt's Akita verifier as a RISC-V guest: it
verifies an inner Akita, field-inline proof and reports the verdict. The cost of
interest is the guest's **total trace rows**, the length an outer proof of that
execution would have to cover, including setup and proof decoding. The working
target is a Fibonacci inner proof below 2^26 rows. This document records the
mechanisms the guest uses, the trust they rely on, and where the rows go.

Only trace execution is implemented for the Akita guest. No outer recursive
proof of it is generated.

## Trust model

The guest distinguishes two kinds of input.

- **Proof data** (the inner proof and its I/O device) is untrusted and goes
  through the ordinary verifier unchanged.
- **Setup data** is trusted. It is the verifier preprocessing, the Akita
  verifier key with its public matrix, a catalog view of the selected schedule
  row, and that row's terminal-matrix NTT cache. The guest takes the matrix as
  seed-derived without re-expanding it, and uses the NTT residues where they lie
  without a range pass.

Trusted setup data must be bound to the program identity. With `--embed` the
host bakes it into the guest image, so the ELF digest is the binding. In input
mode the setup arrives through the guest input, and whoever consumes the
execution must authenticate those bytes. The preprocessing digest alone does not
cover the detached payloads. Embedded mode is the configuration the row count
refers to.

What the guest still checks on setup data:

- The catalog view repeats the full semantic audit of every row it carries and
  recomputes the complete catalog digest from the committed row identities. The
  transcript therefore binds the same setup identity as the full catalog.
- The NTT cache checks its header, its setup and schedule bindings, its
  geometry, its lengths, and its alignment.
- Field-inline readouts reject non-canonical integers.

## Mechanisms

### Field arithmetic on the field-inline ISA

The field-inline unit computes in the one field the outer proof fixes, so the
guest selects it: `field-inline-guest-fp128` routes the Fp128 protocol prime
(Akita) and `field-inline-guest-bn254` routes BN254 `Fr` (Dory); every other
field keeps its software arithmetic. The routed field's dot products, row
dots, product sums, signed sums, multiplications, and inversions run through
field-inline instructions (`field_inline.rs`).

- **Operand ingress** clears a field register (`FIELD_LOAD_IMM 0`), then folds
  the operand's words in with `FIELD_LOAD_ACCUMULATE_FROM_MEMORY` (two for
  Fp128, four for Fr).
- **Readout** emits one range-bound `FIELD_ADVICE_LIMB` per limb and closes with
  `FIELD_ASSERT_ZERO` on the final quotient. The wrapper then rejects an integer
  at or above the modulus.
- **Accumulation** stays register-resident. A 64-term dot costs about 520 rows,
  about 8 per element: two 3-row operand loads, one multiply, and one add.
- **Shared operands** let the weighted-rows kernel load each power once per
  block of five rows (5 rows per element and row). `dot_rows` does the same
  for several rows against one shared vector, and `sum_of_products4` forms
  four-factor products in registers.
- **Signed sums** (`Field::signed_sum`) keep a sum of ± elements in a field
  register: 4 rows per term, against about 24 for a reduced software add.
  Isolated additions stay in software, where two-limb arithmetic wins.

### Hash and NTT inlines

- **Blake2b inline** (`jolt-inlines-blake2`, with a `digest` adapter). The Jolt
  transcript, descriptor digests, layout digests, and the preprocessing digest
  hash through it, with the same bytes as the `blake2` crate. Akita's
  transcript uses a sponge over it that is byte-identical to spongefish's
  `Blake2b512` duplex but compresses each phase's constant 128-byte mask block
  once (`Blake2b::update_block_eager`), saving a compression per phase;
  `jolt-akita` tests it against spongefish.
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
them where they lie, either in its image or its input, instead of copying them.

With `--embed` the host also:

- compiles the bytecode rows into a static table;
- bakes the preprocessing digest computed over the complete program next to the
  row-less record. The guest assembles the preprocessing from the wire form, the
  rows, and that digest. Decoding the ordinary record would re-encode and hash
  the whole program, and would hash the wrong (row-less) program.

### Guest runtime

- An O(1) size-class allocator for std guests.
- No custom `mem*` routines. musl's `memcpy`, `memset`, and `memcmp` move
  data partly in sub-word accesses, each a multi-row sequence.
- `serde_bytes` on every proof and setup byte payload.
- 8-byte-aligned guest record framing.
- `write(2)` and `clock_gettime(2)` answered in the trap handler.

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
`trace --embed`, release guest, `RAYON_NUM_THREADS=1`. The last row is a
freshly generated proof at this revision with the pinned companion; each other
row is the cumulative total after its change.

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
| **Fresh proof, this revision** (64,433,863 verification cycles) | **66,273,665** |

The first row was rebuilt and re-measured from its archived sources and
reproduces its recorded numbers exactly. Input mode, which reads the setup
from the guest input, accepts at 73,311,922 rows. A proof with one bit flipped
in its Akita opening proof is rejected after 47,763,416 rows, inside the PCS
verifier.

## Levers not taken here

- **Load into a cleared register.** Every field operand load would drop from
  3 rows to 2 if the ISA had a non-accumulating memory load, which the
  2^26-era ISA had. This touches every field-inline operand, so it is the
  largest remaining lever. It is a field-inline protocol change.
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
- **Proof field decoding.** Fp128 elements decode through serde's per-byte
  `[u8; 16]` tuple path (about 100 rows each); a faster path changes the
  bincode wire format.
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
```

CI runs the same flow in embedded mode and requires the guest to reject a proof
with one tampered opening.
