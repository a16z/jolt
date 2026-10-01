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

`jolt-field`'s `field-inline-guest` feature routes `Fp128` (and BN254 `Fr`)
dot products, weighted row dots, and multiplications through field-inline
instructions (`fr_inline.rs`).

- **Operand ingress** clears a field register (`FIELD_LOAD_IMM 0`), then folds
  two words in with `FIELD_LOAD_ACCUMULATE_FROM_MEMORY`.
- **Readout** emits one range-bound `FIELD_ADVICE_LIMB` per limb and closes with
  `FIELD_ASSERT_ZERO` on the final quotient. The wrapper then rejects an integer
  at or above the modulus.
- **Accumulation** stays register-resident. A 64-term dot costs about 520 rows,
  about 8 per element: two 3-row operand loads, one multiply, and one add.
- **Shared operands** let the weighted-rows kernel load each power once per
  block of five rows (5 rows per element and row).
- **Additions** stay in software. Measured standalone, field-inline add/sub
  lose to two-limb software arithmetic.

### Hash and NTT inlines

- **Blake2b inline** (`jolt-inlines-blake2`, with a `digest` adapter). The Jolt
  transcript, Akita's spongefish transcript, descriptor digests, layout
  digests, and the preprocessing digest all hash through it. Same bytes as the
  `blake2` crate.
- **64-point i32 NTT and six-product pointwise dot** (`jolt-inlines-ntt`).
  These expand into existing proved integer instructions, with no advice and no
  new constraints. Akita's `ntt-inline` feature dispatches the 64-dimension CRT
  transforms and lazy Montgomery dots to them. The signed-digit conversion is
  fused into the twist (`psi^i · R^2` table).

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
- Word-wise `memcpy`, `memset`, and `memcmp`. `memcpy` reads only through
  opaque inline-asm loads: it may copy padding and reads whole aligned words at
  the range ends.
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
- **Bytecode read-RAF:** stage values are folded as column dot products against
  the address eq table. Expression storage is reused when multiplying by a
  monomial, and opening ids are word-aligned.
- **Akita guest paths:** word-window Golomb–Rice decoding, merged equality
  recurrences, factor-cached and prepared residual tensor contractions,
  certified first-octant trig enclosures, batched compression-relation sums, an
  i64 Garner fast path (superseded upstream by the division-free column Garner),
  direct SHAKE rate-lane sampling, and weighted-row batching in the setup scans.

## Measurements

Fibonacci, one inner proof, `--features akita,field-inline,ntt-inline`,
`trace --embed`, release guest, `RAYON_NUM_THREADS=1`.

| Build | Verification cycles | Total rows |
| --- | ---: | ---: |
| 2^26 line (`archive/fast-recursion-2pow26-local-20260914`, old ISA and Akita) | 65,788,119 | 66,843,409 |
| This stack: port onto `main` (with word `memcpy`) | 83,086,712 | 89,750,914 |
| + Blake2b-inline preprocessing digest | 81,798,363 | 86,917,528 |
| + digest baked into the embedded image | 81,832,195 | 83,677,076 |

The first row was rebuilt and re-measured from its archived sources and
reproduces the recorded numbers exactly. The 16.8M-row gap, by
function-exclusive PC rows against that control (inline expansions count in
their caller; renamed functions are matched by role):

| Area | Δ rows | Cause |
| --- | ---: | --- |
| Field dot kernels and setup scans | +5.3M | Operand ingress is 3 rows on `main`'s ISA, where the old ISA loaded into a fresh register. Upstream's new direct setup scan evaluates one ring per dot. |
| Jolt transcript absorbs | +1.6M | About 1,900 absorbs vs 660, from `main`'s composed field-inline claims. |
| Bytecode read-RAF fold | +1.5M | `main`'s field-register stage values. |
| `memcpy`/`memset`/realloc | +1.8M | Larger proof and verifier buffers upstream. |
| Composed opening-id ordering | +1.3M | `main`'s composed claims (`BTreeMap<ComposedOpeningId>`). |
| Akita verifier internals | +2.3M | Upstream's spongefish receive paths, terminal relation checks, and relation evaluator. |
| Proof decoding | +0.8M | Larger upstream proof. |
| Remainder | +2.2M | Spread across smaller per-function deltas. |

## Levers not taken here

- **Load into a cleared register.** Every field operand load would drop from
  3 rows to 2 if the ISA had a non-accumulating memory load, which the
  2^26-era ISA had. This touches every field-inline operand, so it is the
  largest single lever. It is a field-inline protocol change.
- **Horner or multi-output dots for geometric powers.** The compression-matrix
  columns evaluate `Σ c_j α^j` with shared powers; a Horner kernel removes the
  power loads (about 1.7M rows).
- **Batched transcript absorption** of claim vectors. This is a transcript
  format change for prover and verifier.
- **Sorted vectors instead of the composed opening-id `BTreeMap`** in the claim
  collectors.

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
