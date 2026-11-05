# Spec: Precompiles for Lattice Jolt, Starting with Keccak-f[1600]

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Alberto Centelles              |
| Created     | 2026-11-05                     |
| Status      | proposed                       |
| PR          | [#2307](https://github.com/a16z/jolt/pull/2307) |

## Summary

On Ethereum blocks, Keccak takes about a third of Jolt's RISC-V trace. Each absorb-and-permute inline costs 3,121 trace rows today, and the lattice (Akita) prover pays for every one of them.

This spec adds a precompile path to lattice Jolt, with `KECCAKF` as the first family:

- **The call.** The guest issues one `PCALL` instruction per permutation. Its trace expansion keeps only the rows that read and write the permutation's memory: 44 rows for absorb-and-permute.
- **The record table.** The permutation itself is proved once per record in a fixed-capacity table over GF(2^128).
- **The link.** A prime-field link binds each `PCALL` in the trace to exactly one record, and binds each call's RAM words to the record's bits.

The spec has two parts:

- **Part I: the `PCALL` instruction.** It covers the instruction, its registry entry and fixed-capacity tracing, and it can land on its own.
- **Part II: the proof protocol.** It covers the table relation, the link and the table opening.

Two parts of Part II are open, and both depend on the opening:

1. **One bit source for both readers.** A single committed bit source must answer both GF(2^128) claims (from the table relation) and prime-field claims (from the link).
2. **Fields inside the opening.** The opening must stay inside the fields allowed for it.

Akita's binary-source commitment, which follows LaBinius, is the intended home for that opening. Its current internal field stages are the question this spec asks reviewers.

**Measured so far.** On five mainnet blocks, emulated with the feature off and on, the trace shrinks by 34.62% in total, and the padded trace domain halves on three of the five. On a recursive-verification guest, rows fall by 3.32%. Both figures count trace rows only. Link and opening costs are not included, and no end-to-end proof on a block has been measured yet.

## Glossary

- **Family:** one precompile function, here `KECCAKF`.
- **Flavour:** a variant of a family, selected by `funct3`. Flavour 0 is permute only. Flavour 1 is absorb-and-permute.
- **Port:** a contiguous run of RAM words the call reads, or reads and writes.
- **Row ordinal:** the position of a row inside one `PCALL` expansion. It is a static bytecode property.
- **Record:** one row of the family's table. Its position is the record index $g < K$. It is dynamic and has nothing to do with the static instruction.
- **Start cycle:** the trace cycle of a call's first row. It identifies a dynamic call.
- **Active:** a record that stands for a real call. Inactive records are padding.
- **$K$:** the table capacity, a power of two fixed by the proving key.
- **$T$:** the padded trace length.
- **$H$:** GF(2^128) $= \mathbb F_2[x]/(x^{128}+x^7+x^2+x+1)$, the `F128` type of `jolt-field` (`specs/binary-field.md`).
- **$\mathbb F_p$:** the prime field of lattice Jolt: Akita's 128-bit prime field, so $p > 2^{64}$.

## Intent

### Goal

Add a default-off precompile path to lattice Jolt that removes the Keccak-f[1600] permutation rows from the main trace, proves them in an $H$-valued record table, links them to the trace in $\mathbb F_p$, and opens the table through Akita.

### Background: why Keccak dominates

The block data below was measured by running the Ethereum guest in the Jolt emulator with the feature off and on.

**What the prototype does:**
- Every Keccak call in the current guest is the fused absorb-and-permute inline, at 3,121 rows each.
- With the feature on, each call becomes a flavour-1 `PCALL` of 44 rows: one call row, 17 rate reads, 25 state read-writes and one return.
- The prototype that produced these numbers intercepted the existing Keccak inline key. Part I replaces that with a new key; the row counts do not depend on the key.

| Block | Rows, off | Rows, on | Saved | Padded domain, off to on | Calls $N$ | Next power of two |
|---|---:|---:|---:|---|---:|---:|
| 25,824,419 | 6,886,179 | 4,061,493 | 41.02% | $2^{23}$ to $2^{22}$ | 918 | $2^{10}$ |
| 26,314,047 | 543,149,644 | 377,748,586 | 30.45% | $2^{30}$ to $2^{29}$ | 53,754 | $2^{16}$ |
| 26,304,915 | 850,024,492 | 557,900,266 | 34.37% | $2^{30}$, unchanged | 94,938 | $2^{17}$ |
| 26,320,263 | 1,507,914,750 | 954,857,847 | 36.68% | $2^{31}$ to $2^{30}$ | 179,739 | $2^{18}$ |
| 26,303,900 | 1,884,108,270 | 1,238,569,055 | 34.26% | $2^{31}$, unchanged | 209,795 | $2^{18}$ |

**Totals:**
- Keccak is 35.11% of rows with the feature off.
- The feature removes 34.62% of rows, or 1,658,946,088 in total. Each call saves $3121 - 44 = 3077$ rows.
- All other rows, call counts and guest outputs are identical between the two runs.

**Recursive-verification guest:** counted rows fall from 106,762,299 to 103,220,672 (−3.32%). The padded $2^{27}$ domain is unchanged.

The prototype is not public. All figures will be re-measured on the implementation, with the commands in Testing Strategy.

### Invariants

1. **Default path unchanged.** With the feature off, every existing program decodes, traces and proves exactly as before. Under identical inputs, configuration and prover randomness, the proof bytes (Akita proof included) are byte-identical on every existing fixture.
2. **Old keys keep their meaning.** Existing inline keys behave as today in every configuration, including with the feature on.
3. **Exact coverage.** In an accepted proof, active records and dynamic `PCALL`s are in bijection: each call has exactly one active record with the same flavour and start cycle, and each active record has exactly one call. Inactive records carry a zero pointer. This also implies $N \le K$ against a malicious prover.
4. **Word agreement.** For each call, every RAM word it reads or writes equals the corresponding word of its record:
   - flavour 1: 67 words (17 rate reads, 25 state reads, 25 state writes);
   - flavour 0: 50 words.
5. **Family semantics inside the proof.** For flavour 1 the record proves $S' = \mathrm{KeccakF}(S \oplus (B \,\|\, 0^{512}))$, where $S$ is the input state, $B$ the 1,088-bit block and $S'$ the output. For flavour 0 it proves $S' = \mathrm{KeccakF}(S)$. The XOR is proved, not checked on the host.
6. **Fixed capacity.** $K$ is fixed by the proving key. An honest tracer fails at call $K+1$ before proving; a dishonest prover cannot exceed $K$, by Invariant 3.

New `jolt-eval` invariants, added during implementation:

- `pcall_default_off_byte_identity` (1, 2);
- `pcall_record_coverage` (3, 6), with a missing, duplicated, triplicated or fractional record, a record pointing at a non-call cycle, and call $K+1$;
- `pcall_word_agreement` (4, 5), with one-bit flips in an input, block or output word, metadata swaps (flavour, start), and paired errors that would cancel in characteristic 2.

### Non-Goals

- Zero knowledge. The `akita` and `zk` features are already mutually exclusive, and the new messages would need their own simulator.
- Other families. The table, registry and link are generic, but this spec specifies and measures `KECCAKF` only.
- Choosing $K$ per proof or across proofs. It is a parameter of the proving key.
- Changing Akita. Where the opening needs an Akita change, the spec states the requirement and leaves the change to an Akita PR.

## Evaluation

### Acceptance Criteria

**Part I** (instruction, registry, tracing):

- [ ] With the feature off, all existing fixtures prove and verify with byte-identical proofs, under the commands below.
- [ ] Registering two inlines with the same `(opcode, funct3, funct7)` fails registry validation. The existing test `linked_inline_registration_keys_are_unique` covers every feature combination that links the precompile.
- [ ] A host built without the feature rejects the `PCALL` key with a clear error.
- [ ] A trace with $K+1$ calls fails during execution with `PRECOMPILE_CAPACITY_EXCEEDED`, before proving.
- [ ] Emulated rows with the feature on match the table above on the five blocks.

**Part II** (protocol):

- [ ] With the feature on, the Ethereum guest proves and verifies on at least one real block. The prover's row counter matches the emulator.
- [ ] Proofs are rejected in every `pcall_record_coverage` and `pcall_word_agreement` case.
- [ ] Added proof bytes are measured on real blocks and split into link, relation, commitment and opening. The total is reported against the 4,096-byte target.

### Testing Strategy

- **Feature off, unchanged** (Invariant 1):
  - root `cargo nextest run --features host` and `--features host,zk`;
  - the Akita matrix: `cargo nextest run -p jolt-prover --features akita,prover-fixtures -E 'binary(e2e_matrix)'`.

  Before and after proof bytes are compared under fixed inputs and randomness.
- **Feature on:** the same Akita matrix with `akita,prover-fixtures,pcall-keccakf`.
- **New tests:**
  - tracer tests for both flavours, operand checks and the capacity error;
  - relation tests on random records, plus the mutation suites above;
  - one end-to-end proof on a recorded Ethereum block input.
- **Binary sum-check:** the relation runs on the characteristic-2 path of `jolt-sumcheck` from `specs/binary-sumcheck.md`.

### Performance

| Objective | Direction | Magnitude |
|---|---|---|
| Trace rows on Ethereum blocks | down | 30% to 41% per block, measured in the emulator |
| Trace rows, recursive-verification guest | down | 3.32%, measured |
| Padded trace domain | down on most blocks | halves on 3 of 5 blocks |
| Proof bytes, feature on | up | see Costs; target at most 4,096 bytes per family per proof |
| Proof bytes, feature off | unchanged | required |
| Prover time | unknown | to be measured; table work grows linearly in $K$ |

New `jolt-eval` objectives: `pcall_rows_saved` and `pcall_added_proof_bytes`, both per block.

## Design

### Dependencies

| Dependency | Status |
|---|---|
| `F128` in `jolt-field` (`specs/binary-field.md`) | implemented |
| Characteristic-2 batched sum-check: zero extension, no uni-skip over domains larger than 2 (`specs/binary-sumcheck.md`) | proposed |
| `F128` challenge sampling in `jolt-transcript` | to add |
| An Akita revision with binary sources and an opening that satisfies section II.4. `main` pins Akita `1160b0a8`, which has neither. | pending |

### Part I: the `PCALL` instruction

#### I.1 Encoding and ABI

| Field | Value |
|---|---|
| opcode | `0x0b` (core inline space) |
| funct7 | `0x0a`, a new precompile block |
| funct3 | `0`: `KECCAKF` permute; `1`: `KECCAKF` absorb-and-permute; 2 to 7 reserved and rejected |
| rs1 | address of the 25-word state |
| rs2 | flavour 1: address of the 17-word block; flavour 0: must be `x0` |
| rd | must be `x0` |

**Guest API:**
- `keccak_f(state: &mut [u64; 25])`, flavour 0;
- `keccak_absorb_f(state: &mut [u64; 25], block: &[u64; 17])`, flavour 1.

Lanes are little-endian `u64`, in the order `jolt-inlines/keccak256` uses. The guest-side feature makes these functions emit the new key. Without it they use the existing inline. Old keys keep their meaning everywhere (Invariant 2).

**Key choice.** On `main`, opcode `0x0b` uses `funct7` 0 to 9:

| `funct7` | Inline |
|---:|---|
| 0 | SHA-256 |
| 1 | Keccak |
| 2 | BLAKE2 |
| 3 | BLAKE3 |
| 4 | bigint |
| 5 | secp256k1 |
| 6 | Grumpkin |
| 7 | P-256 |
| 8 | NTT |
| 9 | NTT pointwise dot |

They are listed in `jolt-inlines/fixtures/fixtures/registered_inline_expand_parity_hashes.jsonl`. The first free block is `0x0a`. The user space `0x2b` already has a test-only registration at $(0\text{x}2b, 0\text{x}7e, 6)$, so this spec does not use it.

**Registry.** Today `tracer/src/instruction/inline.rs` looks registrations up with a first-match `inventory::iter().find(..)`, and uniqueness is checked only by `linked_inline_registration_keys_are_unique`. This spec adds a one-time validation of all linked registrations. Execution and preprocessing share it, and it fails on any duplicate key in any block, not only the new one.

#### I.2 Expansion

One `PCALL` expands into memory rows only. It has no ALU rows for the permutation.

| Flavour | Rows | RAM words |
|---|---|---|
| 0 | call, 25 state read-writes at `rs1 + 8i`, return: 27 rows | 25 read, 25 written: 50 |
| 1 | call, 17 block reads at `rs2 + 8i`, 25 state read-writes, return: 44 rows | 17 + 25 read, 25 written: 67 |

**Bytecode and sequencing.**
- The rows are a virtual sequence, with new virtual instructions `PcallCall`, `PcallRead`, `PcallWrite` and `PcallReturn`. Each bytecode row carries the family, flavour and row ordinal.
- With the feature on, these fields extend the committed bytecode row, so the bytecode layout digest changes. With the feature off, the layout is unchanged.
- **Where written values come from.** The value of each `PcallWrite` row is supplied by the tracer from the record. The main trace does not constrain it; only the link (II.3) fixes it.

**Operand checks.** Done by the tracer before any memory effect:
- 8-byte alignment;
- non-wrapping ranges of 200 state bytes and, for flavour 1, 136 block bytes;
- both ranges in ordinary RAM, with the block range disjoint from the state range.

What the verifier enforces for these properties is listed in II.6.

#### I.3 Capacity

The tracer counts calls per family and stops at call $K+1$ with `PRECOMPILE_CAPACITY_EXCEEDED: KECCAKF capacity K`, before that call's memory effects and before proving.
- **No automatic retry.** There is no retry at a larger $K$, because $K$ is fixed by the proving key.
- **Not a security check.** This error is for usability only: soundness of $N \le K$ comes from the coverage check in II.3.
- **Sizing.** The blocks above need $K$ from $2^{10}$ to $2^{18}$. Occupancy at each block's own power of two is 0.69 to 0.90. A single key with $K = 2^{18}$ would leave the small block at 0.0035, and its fixed table work is paid on every proof (Open question 3).

### Part II: the proof protocol

#### II.1 Crate and stage map

| Piece | Crate | Stage |
|---|---|---|
| `PCALL` vocabulary, flags, decode | `jolt-riscv` (new module, as for field inline), `jolt-program` bytecode fields | preprocessing |
| Guest API | `jolt-inlines/keccak256`, behind a guest feature | none |
| Expansion and capacity | `tracer` | execution |
| Relation ids, symbolic relations, layouts | `jolt-claims::protocols::jolt::lattice::precompile` (`specs/self-contained-sumcheck-relations.md`) | none |
| Record table witness | `jolt-witness` | commitment, before any challenge |
| Link sum-checks (II.3) | `jolt-kernels`, prime members in a new stage after 6b | new prime stage |
| Family relation (II.2) | `jolt-kernels`, over `F128` | new binary stage after the link |
| Table opening (II.4) | `jolt-akita`, a new group appended after `OneHotTrace` in the grouped order of `specs/lattice-claims.md` | 8 |

**Why the new stages.** A `SumcheckBatch` is over one field. So the binary relation cannot join a prime batch, and it gets its own stage. The prime link members could join an existing batch. A separate stage keeps feature-off batches identical (Invariant 1).

**Feature scope.**
- The feature is defined in `tracer`, `jolt-riscv`, `jolt-program`, `jolt-claims`, `jolt-witness`, `jolt-kernels`, `jolt-prover`, `jolt-verifier` and `jolt-sdk`.
- It requires `akita`. It is a compile error with `zk` and is rejected on Dory builds.
- $K$, the family set, the table layout and the protocol version are part of the preprocessing digest.
- A program with no calls still commits a table of $K$ inactive records, so the proof shape stays fixed.

#### II.2 Record table and family relation over $H$

**Record layout.** One record holds, for each round $r = 0..24$, the 1,600-bit state $A_r$ entering round $r$; $A_{24}$ is the output.

| Columns | Bits | Content |
|---|---:|---|
| state | $32 \times 32 \times 64 = 65{,}536$ | 25 states, padded to 32 rounds and 32 lanes |
| parity | $32 \times 8 \times 64 = 16{,}384$ | $\theta$ column parities, 5 used per round |
| constants | $32 \times 64 = 2{,}048$ | $\iota$ round constants, public |

**Prototype figures.** The layout above is the measured prototype: 83,968 bits per record.
- The absorb flavour adds the pre-absorb state and the block: $1{,}600 + 1{,}088$ bits.
- Metadata adds 2 flavour bits and the pointer chunks of II.3.
- The resulting layout pads to $2^{17}$ bits per record.

**Constraints, per round:**
- $\theta$ parity: each parity column equals the XOR of its five lanes; this is linear;
- $\rho$ and $\pi$: public rotations and wiring;
- $\chi$: $a \oplus (\lnot b \wedge c)$, degree 2;
- $\iota$: XOR of the public constant, affine;
- absorb: $A_0 = S \oplus (B \,\|\, 0^{512})$ for flavour 1, and $A_0 = S$ for flavour 0, gated by the flavour bits;
- inactive records: the permutation of the zero state;
- unused lanes and rounds: zero.

Bitness comes from the source commitment; the opening must guarantee that the committed source is Boolean.

**Aggregation.** In characteristic 2, equal nonzero residuals cancel in pairs, so a plain sum of residuals proves nothing. The relation instead proves a zero-check: an $\mathrm{eq}$-weighted sum over (record, constraint index), with constraint families combined by independent random $H$ coefficients. This is one batched sum-check over $\log_2 K + 16$ variables of degree at most 3. The prototype ran it in 27 rounds at $K = 2^{11}$, rejected all 11 single-step mutations tried, and produced a 2,552-byte transcript with the opening stubbed.

**Soundness error:**
$$\frac{3(\log_2 K + 16) + \log_2 K + 16 + c}{2^{128}}$$
- the first term: rounds times degree;
- the second: the zero-check point;
- $c$: the batching of constraint families.

The relation ends in $H$-linear claims on the committed bits, which go to the opening (II.4).

#### II.3 Link over $\mathbb F_p$: coverage and word agreement

The link is proved over $\mathbb F_p$, because exact counting needs characteristic larger than any count.

**Pointer.**
- Record $g$ stores two Boolean flavour bits $f_0, f_1$, with $u = f_0 + f_1 \le 1$ (active).
- Its start cycle $\sigma_g < T$ is encoded as $\lceil \log_2 T / 4 \rceil$ one-hot chunks of at most 16 bits each.
- Record constraints: chunk bits are Boolean; each chunk of an active record sums to 1; inactive records are all zero.
- Activity: $a(g, \ell) = f_0[\ell < 27] + f_1[\ell < 44]$ for row ordinal $\ell < 64$.
- Shifted read: $\mathrm{ra}_\ell(g, j)$ is the indicator that $\sigma_g + \ell = j$. Its multilinear extension in $j$ is a product of per-chunk carry matrices. It is evaluated as that product, never re-multilinearised, and carry out of the top chunk is rejected.

**Coverage.** Let $\lambda(j)$ be the authenticated bytecode indicator that cycle $j$ is a `PCALL` row. Define
$$C(j) = \sum_{g, \ell} a(g, \ell)\, \mathrm{ra}_\ell(g, j).$$
The link proves $C(j) = \lambda(j)$ for all $j$.
- $C(j)$ is an integer count of at most $K \cdot 64 < p$.
- So the equality forces every `PCALL` row to be covered exactly once and no other row to be covered. That rejects missing, duplicated, triplicated and fractional records.
- Combined with the row-ordinal agreement below and the interval lemma (II.6), slot 0 of each active record lands on a call row and its later slots lie in the same call. This gives Invariant 3, and in particular $N \le K$.

**Agreement.**
- **Prime readings of the bits.** The link reads the record's bits as prime-field values $b \in \mathbb F_p$, with $b(b - 1) = 0$.
- **Limb equations.** For each applicable row $\ell$ of record $g$ and each limb $h \in \{0, 1\}$:
$$R_{g,\ell,h} = \sum_{s=0}^{31} 2^s\, b[\mathrm{map}(g, \ell, 32h + s)],$$
where $R$ is the trace's authenticated read or write value at cycle $\sigma_g + \ell$, split into two 32-bit limbs.
- **Why limbs.** Each limb is below $2^{32} < p$, and two limb equalities imply word equality.
- **Trace values.** The RAM read and write values are virtual polynomials in Jolt. Their values at the link's terminal point are obtained through the existing RAM read-write reductions, not asserted.
- **Metadata.** The row ordinal, flavour and family fields of the trace are matched to the record in the same way.

**Protocol.**
1. **Commitments first.** The prover commits the record table before any link challenge.
2. **Trace sum-check.** The verifier samples a random record point $\tau$ and row point $\rho$, plus independent coefficients $\gamma$ for the link fields. The prover sends
$$z = \sum_{g,\ell} \mathrm{eq}(\tau, g)\, \mathrm{eq}(\rho, \ell)\, P_R(g, \ell),$$
where $P_R$ is the record side of all link fields combined by $\gamma$. The verifier samples $\eta$, and a degree-2 sum-check over $j$ ($\log_2 T$ rounds) proves that $z$ equals the trace side, plus $\eta\, \mathrm{eq}(\zeta, j)(C(j) - \lambda(j))$.
3. **Record and row sum-check.** A second sum-check over (record, row ordinal), with 6 row rounds of degree at most 3 and $\log_2 K$ record rounds of degree at most 9, proves:
   - the definition of $z$;
   - the trace weights evaluated at the first sum-check's terminal point;
   - all record constraints above.

   The degree 9 comes from the 7 chunks of the carry-matrix product.
4. **Terminals.** All terminal claims are linear forms in the trace polynomials (through the existing reductions) and in the record's prime readings.

**Error.** At $T = 2^{27}$ and $K = 2^{11}$, the link's interactive error is at most $230/p$, and the Boolean agreement adds at most $45/p$, assuming exact terminal openings.

**The open dependency.** The agreement is sound only if the record's prime readings $b$ and the bits used by the $H$ relation (II.2) are the same committed bits. Matching transcript randomness does not give this, and neither does converting field encodings. It must be an interface property of the opening, stated in II.4 as the common-source requirement.

#### II.4 Opening the table through Akita

**Requirements for an admissible opening:**
1. **Single commitment scheme.** Akita is the only commitment scheme in the proof: no hash-based or Merkle commitment, and no second scheme instance.
2. **Common source.** One committed bit table answers both kinds of claim, with one extractor:
   - $H$-linear claims from the relation (II.2);
   - $\mathbb F_p$-linear claims from the link (II.3).
3. **Boolean source.** The extracted source is Boolean.
4. **Claim fields.** Claims reach the Jolt verifier only in $H$ or $\mathbb F_p$, so the verifier and the claim pipeline handle one binary field type. Whether the opening may use a larger binary field internally (Akita's `BinaryField192`, a cubic extension of a 64-bit field) is Open question 1, not a requirement.
5. **Default path.** With the feature off, the Akita proof is unchanged.

| Route | Status | Cost |
|---|---|---|
| Akita binary-source mode: bit tables committed as in LaBinius and opened in one grouped session with the prime sources. Public on Akita in the binary-source PRs, #318 to #332. | The intended route. The grouped session gives one opening for prime and binary sources. Two things are missing: an evaluation interface that serves both claim fields from one source (requirement 2), and production security parameters for the grouped profile. Its internal stages currently include a field switch to a larger binary field and prime-field steps. | Open |
| Standalone LaBinius instance for the table | Sound, but it is a second commitment instance (fails 1), and it does not serve the prime link claims (fails 2). | Not priced here |
| Prime Akita with the bits stored as integers mod $p$ | Serves the prime link, but $H$ claims then need a proved lift of each residue to its canonical bit. There is no nonzero ring homomorphism from an $\mathbb F_2$-algebra into a ring of odd characteristic, so the parity of a residue is not a ring operation, and the lift must be proved bit by bit. | High; not priced here |
| Reference mode: disclose the records | Sound for testing; does not meet the production criteria. | 558,104 bytes at $K = 2^{10}$ in the specified codec |

**Reference mode.** This is the opening used for the first end-to-end run.
- The proof carries the canonical record words: input state, block, flavour and start of each record. They are absorbed into the transcript before any link or relation challenge.
- The verifier rebuilds the bit source from them and evaluates every $H$ and $\mathbb F_p$ terminal claim directly.
- The prime link (II.3) is proved in full.
- Its proofs and results are labelled reference-only.

#### II.5 Verifier changes (feature on)

- **Preprocessing:** registry validation, the authenticated expansion fields, and the digest of $K$, layout and version.
- **Link stage (II.3):** the prime sum-checks, with terminals routed through the existing RAM and bytecode reductions.
- **Binary stage (II.2):** the $H$ zero-check.
- **Stage 8:** the table opening as an additional group, through the route chosen in II.4.

With the feature off, none of this is compiled or run.

#### II.6 Interval lemma and simulation

**Interval lemma** (obligation; to be proved for `main`'s encoding). In every accepted trace, the rows of each `PCALL` expansion appear as one consecutive run, in row-ordinal order, from its call row to its return row. No row outside the run enters or leaves it.

On `main`, this follows from existing constraints together with the bytecode commitment:
- **Bytecode index.** Bytecode rows are indexed by (instruction address, `virtual_sequence_remaining`). Expansion rows share the address and are ordered by the sequence counter (`jolt-program/src/preprocess/bytecode.rs`).
- **Within a sequence**, R1CS row 17 (`NextPCEqPCPlusOneIfInline`) forces $\mathrm{NextPC} = \mathrm{PC} + 1$.
- **Entering a sequence**, row 18 (`MustStartSequenceFromBeginning`) forces any entry to be its first row.
- **Leaving a sequence**, rows 14 and 16 fix `NextUnexpandedPC` at its end. `JALR` targets already have bit 0 cleared through `RangeCheckAligned`, so no new jump constraint is needed.
- **New for `PcallWrite`.** The flag must keep registers unchanged and the address equal to the base register plus the row ordinal times 8.

**Simulation theorem** (obligation). The theorem assumes:
- the interval lemma;
- coverage and agreement (II.3);
- a sound relation (II.2);
- a sound opening (II.4).

Under these assumptions, contracting each `PCALL` interval to one step gives the execution of the same program in which the call is an atomic application of the family function to its entry RAM.

**What the verifier must enforce:**

| Requirement | Enforced by |
|---|---|
| Expansion order and entry | rows 17, 18 and the bytecode commitment (existing) |
| Lane, flavour, ordinal | authenticated bytecode fields (new, feature-gated) |
| Port addresses | RAM checking, plus the `PcallWrite`/`PcallRead` address equation (new, feature-gated) |
| Register preservation | flags of the new virtual instructions (new, feature-gated) |
| State/block disjointness and alignment | for alignment, the address equation over aligned 8-byte words; for disjointness, a new feature-gated check |
| No cut inside an interval | only relevant if trace chunking is added; it must not cut inside a sequence |

All new constraints are compiled only with the feature (Invariant 1).

### Soundness obligations

1. The $H$ relation zero-check (II.2), with the error bound stated there.
2. The link reduction (II.3): at most $230/p$ for coverage and $45/p$ for Boolean agreement at the stated sizes.
3. The opening: binding, the common-source extractor and Boolean extraction (II.4). This is open.
4. The interval lemma and the simulation theorem (II.6), with each requirement mapped to a constraint.
5. Statement binding: the transcript absorbs, before any precompile challenge:
   - the program and preprocessing digests;
   - the feature and protocol version;
   - the family and flavour semantics;
   - $K$, $T$ and the field identifiers;
   - the table layout, the padding rule and the ordered claim roles.

   Challenges are domain-separated per stage.
6. Composition: the total error is the sum of the existing VM and PCS errors and items 1 to 3, under Fiat-Shamir.

### Costs

**Rows** (measured): −34.62% on five blocks and −3.32% on a recursive-verification guest. A call costs 44 rows instead of 3,121.

**Bytes**, against a target of at most 4,096 extra bytes per family per proof. Prime elements are 16 bytes.

| Message | Elements | Bytes | Basis |
|---|---|---|---|
| Relation zero-check, $K = 2^{11}$ | about 160 | 2,552 | measured, opening stubbed |
| Link round messages | $3\log_2 T + 24 + 10\log_2 K$ (215 at $T = 2^{27}$, $K = 2^{11}$) | 3,440 | forecast |
| Link and agreement terminals | about 210 at the same sizes | 3,400 | forecast |
| Commitment and opening | open | open | II.4 |

**Before the opening, the forecast is already about 9 KB**, above 4,096 bytes:
- **What batching saves.** Folding terminals into the stage-8 batch removes only terminal claims, not round messages.
- **The constraint.** Meeting the target therefore needs fewer round messages, for example by merging the link and relation sum-checks where fields allow, and an opening that adds a few hundred bytes.
- **If that is not achievable,** the trade (about a third fewer rows for roughly 10 KB or more) is a decision for maintainers (Open question 2).

**Table size.** At $K = 2^{17}$ the table holds $2^{17}$ records of $2^{17}$ bits: $2^{34}$ Boolean entries. Packed 128 per `F128` word, that is 2 GiB. The relation's operation count is not yet measured. With the opening stubbed, the prototype took 7.25 s at $K = 2^{11}$.

### Rollout

1. **Part I** behind `pcall-keccakf`, off by default: the registry validation, the instruction and the capacity error. Proving with the feature on is rejected until Part II lands.
2. **Part II, in order:**
   1. the record table and the relation;
   2. the link with reference mode;
   3. an Akita pin bump and the production opening.

   Each step lands with its tests.
3. **Turning it on.** The feature is turned on for block proving only after the opening is settled, bytes are measured on real blocks, and maintainers accept the byte cost.

### Alternatives Considered

- **Keep Keccak as an inline.** No new protocol, but 3,121 rows per call and about a third of the trace on blocks.
- **Prove Keccak in $\mathbb F_p$ with lookups.** Bitwise steps cost per-bit columns or lookup tables in a prime field; the $H$ table proves $\chi$ and the linear steps natively.
- **Prime Akita only, with bits as residues.** Avoids a binary source, but the bit lift costs per bit (II.4).
- **Two specs.** Part I and Part II could be separate specs (Open question 6).

## Documentation

- A book page under `book/src/how/` on precompiles: the ABI, flavours, capacity, how to enable the guest and host features, and what a host without support does with the key.
- An update to `book/src/how/optimizations/inlines.md`: with the guest feature on, the Keccak helpers emit the new key; the old inline keys keep their meaning.

## Execution

A non-public prototype implements:
- the expansion and the capacity error;
- the prime-side rows, against the real verifier, with mutated proofs rejected;
- the $H$ relation with a stubbed opening.

The link and the opening were stubs there. The first end-to-end proof on a real block uses reference mode (II.4), and its measured rows, time and bytes will be added here.

## Open questions

1. **Opening.** Can Akita's binary-source mode provide the common-source interface of II.4, one bit table answering both $H$ and $\mathbb F_p$ linear claims with one extractor? May its internal stages use the larger binary field, provided only $H$ and $\mathbb F_p$ claims reach Jolt?
2. **Bytes.** Is 4,096 bytes per family the right target for about a third fewer rows on blocks, or should it scale with rows saved?
3. **Capacity.** One $K$ per proving key, or a small set of keys from $2^{10}$ to $2^{18}$?
4. **Key.** Is $(0\text{x}0b, 0\text{x}0a, \cdot)$ acceptable as the precompile block, or should precompiles get their own opcode?
5. **Interval lemma.** Should its proof for `main`'s virtual-sequence encoding be written once for all inline sequences, since existing virtual sequences rely on the same rows?
6. **Scope.** Keep this as one spec, or split Part I and Part II?

## References

- Akita (lattice PCS): https://github.com/LayerZero-Labs/akita, binary-source PRs #318 to #332.
- M. Osadnik and G. Seiler, *LaBinius: Fast Lattice-Based Binary Polynomial Commitment Scheme*, IACR ePrint 2026/2103.
- `specs/binary-field.md`, `specs/binary-sumcheck.md`, `specs/lattice-claims.md`, `specs/field-inline-protocol.md` (precedent for a new instruction family on the modular stack), `specs/self-contained-sumcheck-relations.md`, `specs/prover-stage-drivers.md`.
- NIST FIPS 202, *SHA-3 Standard: Permutation-Based Hash and Extendable-Output Functions*.
- `book/src/how/optimizations/inlines.md`.
