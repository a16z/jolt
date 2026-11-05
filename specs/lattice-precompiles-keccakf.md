# Spec: Precompiles for Lattice Jolt, Starting with Keccak-f[1600]

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | Alberto Centelles              |
| Created     | 2026-11-05                     |
| Status      | proposed                       |
| PR          |                                |

## Summary

On Ethereum blocks, Keccak is about a third of Jolt's RISC-V trace. Each absorb-and-permute inline costs 3,121 trace rows today, and the Akita (lattice) prover pays for every one of them. This spec adds a precompile path to lattice Jolt and uses `KECCAKF` as the first family. The guest issues one `PCALL` instruction per permutation. The main trace keeps only the 44 rows that read and write the permutation's memory. The permutation itself is proved once per call slot by a separate relation over a fixed-capacity record table in GF(2^128). A one-hot pointer link binds each `PCALL` in the trace to exactly one record.

Two results are measured. On five mainnet blocks, emulated with the feature on, the trace shrinks by 34.6% in total, and the padded trace domain halves on three of the five. On a recursive-verification guest, rows fall by 3.32%. Both figures count trace rows only; they do not yet include the cost of the link or the table opening.

The open part is the opening of the record table. It must use Akita as the only commitment scheme and stay inside GF(2^128). Akita's binary-source commitment, which follows LaBinius, is the intended route, but its current opening runs internally in a larger binary extension field and in prime-field stages. The spec asks reviewers how to close that gap. The feature stays off by default until the opening exists and its bytes are measured.

## Intent

### Goal

Add a default-off precompile path to lattice Jolt that removes the Keccak-f[1600] permutation rows from the main trace and proves them in a GF(2^128) record table, linked to the trace and opened through Akita.

The path introduces four pieces:

1. **`PCALL` instruction.** A guest instruction with its own ABI, encoding and registry entry. Its trace expansion contains only memory rows.
2. **Record table.** One table per precompile family, with a fixed capacity `K` per proof. Each slot holds one call's input and output in a fixed bit layout over GF(2^128).
3. **Family relation.** A sum-check relation over GF(2^128) that holds on every slot iff the slot's output is the family function of its input. For `KECCAKF` this is Keccak-f[1600], including the XOR of the absorbed block into the state.
4. **Link.** A one-hot pointer from each call's first trace row to its record, plus an agreement check between the call's 64-bit RAM words and the record's bits.

### Background: why Keccak dominates

The block data below is measured by running the Ethereum guest in the Jolt emulator, with the feature off and on, on five mainnet blocks. Every Keccak call in the current guest is the fused absorb-and-permute inline, at 3,121 rows each. With the feature on, each call costs 44 rows: one call row, 17 reads of the rate words, 25 read-writes of the state and one return.

| Block | Rows, off | Rows, on | Saved | Padded domain, off to on | Permutations $N$ | Capacity $K$ |
|---|---:|---:|---:|---|---:|---:|
| 25,824,419 | 6,886,179 | 4,061,493 | 41.02% | $2^{23}$ to $2^{22}$ | 918 | 1,024 |
| 26,314,047 | 543,149,644 | 377,748,586 | 30.45% | $2^{30}$ to $2^{29}$ | 53,754 | 65,536 |
| 26,304,915 | 850,024,492 | 557,900,266 | 34.37% | $2^{30}$, unchanged | 94,938 | 131,072 |
| 26,320,263 | 1,507,914,750 | 954,857,847 | 36.68% | $2^{31}$ to $2^{30}$ | 179,739 | 262,144 |
| 26,303,900 | 1,884,108,270 | 1,238,569,055 | 34.26% | $2^{31}$, unchanged | 209,795 | 262,144 |

Across the five blocks, Keccak is 35.11% of rows with the feature off, and the feature removes 34.62% (1,658,946,088 rows). Each call saves $3121 - 44 = 3077$ rows. All other rows, call counts and outputs are identical between the two runs.

### Invariants

1. **Default path unchanged.** With the feature off, every existing program decodes, traces and proves exactly as before, and proof bytes, including Akita PCS bytes, are byte-identical on every existing fixture.
2. **Exact coverage.** Each dynamic `PCALL` is matched to exactly one live record. No live record names a cycle that is not a `PCALL` start. Padding slots are marked inactive and satisfy the relation on a fixed input.
3. **Word agreement.** For each call, the 67 RAM words it touches (17 rate words read, 25 state words read and 25 written) equal the corresponding words of its record, bit for bit.
4. **Absorb inside the proof.** The XOR of the rate block into the state is part of the proved relation, not a host-side check.
5. **Fixed capacity.** `K` is fixed per proof. A program whose call count exceeds `K` fails during execution, before any proving work, with a clear error.
6. **Atomic semantics.** Contracting each complete `PCALL` interval in the trace to one step yields the same execution as running Keccak-f[1600] atomically on the entry state (the projection property below).

New `jolt-eval` invariants, added during implementation:

- `pcall_default_off_byte_identity` (invariant 1);
- `pcall_record_coverage` (invariants 2 and 5), with negative cases for a dropped, duplicated or spliced record and for call `K+1`;
- `pcall_word_agreement` (invariants 3 and 4), with one-bit flips on input, output and the absorbed block.

### Non-Goals

- Zero knowledge. The new messages need their own simulator; this is follow-up work.
- Other families (SHA-256, BLAKE, secp256k1). The table, registry and link are generic, but this spec specifies and measures `KECCAKF` only.
- Choosing `K` dynamically or across proofs. `K` is a per-proof public parameter here.
- Changing Akita itself. Where the opening needs an Akita change, the spec states the requirement and leaves the change to an Akita PR.

## Evaluation

### Acceptance Criteria

- [ ] With the feature off, all existing fixtures prove and verify with byte-identical proofs and Akita bytes.
- [ ] With the feature on, the Ethereum guest proves and verifies on at least one real block, and the counted rows match the emulator figures above.
- [ ] A trace with $K+1$ calls fails during execution with the capacity error, before proving.
- [ ] Proofs are rejected for: a missing record, a duplicated record, a record naming a non-`PCALL` cycle, a one-bit change in any record output, input or absorbed word, and a one-bit change in any linked RAM word.
- [ ] The `PCALL` key is unique in the linked inline registry under every feature combination, including test configurations.
- [ ] The added proof bytes (link, relation and opening) are measured on real blocks and reported against the 4,096-byte per-precompile target.

### Testing Strategy

- `cargo nextest run --features host` and `--features host,zk` stay green with the feature off.
- New tests under the feature: tracer tests for the expansion and capacity error; relation tests on random records plus the mutation suite above; an end-to-end prove-and-verify on a small guest and on one recorded Ethereum block input.
- A registry audit test that enumerates every linked inline registration and asserts the `PCALL` key range is disjoint from all others.

### Performance

The expected movement, from the measurements above:

| Objective | Direction | Magnitude |
|---|---|---|
| Trace rows on Ethereum blocks | down | 30% to 41% per block, measured with the emulator |
| Trace rows on a recursive-verification guest | down | 3.32%, measured, inside an unchanged $2^{27}$ domain |
| Padded trace domain | down on most blocks | halves on 3 of 5 blocks |
| Proof bytes | up | see Costs; target at most 4,096 bytes per precompile per proof |
| Akita PCS bytes | unchanged | required |
| Prover time | expected down | not yet measured end to end |

New `jolt-eval` objectives: `pcall_rows_saved` and `pcall_added_proof_bytes`, both per block.

## Design

### Architecture

#### 1. `PCALL` instruction, ABI and registry

`PCALL` is an R-type instruction in the custom inline space. One registry entry exists per family flavour.

| Field | Value |
|---|---|
| opcode | `0x0b` (core inline space) |
| funct7 | `0x08`: the precompile family block, new |
| funct3 | `0`: `KECCAKF` permute; `1`: `KECCAKF` absorb-and-permute; 2 to 7 reserved |
| rs1 | address of the 25-word state, 8-byte aligned |
| rs2 | address of the 17-word rate block (flavour 1), else `x0` |
| rd | `x0` |

The guest API is `keccak_f(state: &mut [u64; 25])` and `keccak_absorb_f(state: &mut [u64; 25], block: &[u64; 17])`.

**Relocation.** The prototype intercepted the existing Keccak inline keys $(0\text{x}0b, 0\text{x}01, 0/1)$. With the feature on, the same instruction word meant two different things, so programs built for the inline silently changed semantics. The precompile gets its own `funct7` block instead. On current `main` the core space uses `funct7` 0 to 7 (SHA-256, Keccak, BLAKE2, BLAKE3, bigint, secp256k1, Grumpkin, P-256), and `0x08` is unused. The registry rejects duplicate keys at construction and does not use first-match lookup for this block. One test-only registration on `main` already sits at $(0\text{x}2b, 0\text{x}7e, 6)$, so the user space is not used. The final key is for review; the audit test above enforces whatever is chosen.

**Expansion.** The tracer expands one `PCALL` into its memory rows only, without the permutation's ALU rows:

- 1 call row;
- 17 reads at `rs2 + 8i` (flavour 1);
- 25 read-writes at `rs1 + 8i`;
- 1 return row.

Each row carries two fields decoded from authenticated bytecode: a family lane, equal to 1 on these rows and 0 elsewhere, and a tag that encodes the (flavour, slot) pair injectively. They are not free witness columns.

**Memory.** Both ports must be aligned, inside ordinary RAM and non-overlapping, and all addresses are checked before the first write. Bases and registers are unchanged by the call.

#### 2. One record table per family

Each family has one table of `K` slots. A slot holds:

- the input state, the absorbed block and the output state, as bits;
- the internal post-XOR state, so that the absorb is inside the relation;
- an active flag, a flavour bit and the slot's start pointer.

The prototype layout uses 83,968 witness bits per slot, including parity and constant columns. A tighter layout with 41,195 occupied bits, padded to 65,536, is specified but not implemented. Inactive slots hold the permutation of the zero state and are excluded from coverage.

#### 3. Family relation in GF(2^128)

The table field is GF(2^128) $= \mathbb F_2[x]/(x^{128}+x^7+x^2+x+1)$, the `F128` type in `jolt-field`'s binary backend (`specs/binary-field.md`). Keccak-f[1600] is linear over $\mathbb F_2$ except for $\chi$, which has degree 2. So the round function is a low-degree polynomial identity on the slot's bits, and one batched sum-check proves all `K` slots at once.

The prototype relation:

- at $K = 2^{11}$, runs 27 sum-check rounds with a 2,552-byte transcript (measured, with the opening stubbed);
- rejects all 11 mutations tried (one-bit changes in each permutation step and in the endpoints).

The relation ends in evaluation claims on the table's bits. Those claims go to the opening below.

#### 4. Link: one-hot pointer plus word agreement

- **Coverage.** Each record stores a one-hot pointer to the cycle of its call row. The verifier checks, with one sum-check over the trace's `PCALL`-start indicator (the call row's lane and tag), that the pointers cover every call row exactly once, and that every active record points at a call row. Multiplicity must be exactly one: a parity check alone would accept three identical copies.
- **Agreement.** Each 64-bit RAM word is exported as two 32-bit limbs bound to the RAM relation. The record arranges each limb as one column of 32 bits. Agreement is then one aggregate claim per limb column plus a small metadata vector, instead of per-bit terminals. Each limb is a canonical integer below $2^{32}$, smaller than the prime, so it has a unique bit decomposition.

The prime-field side of the link uses the trace's existing Akita commitment. The binary side uses the table's claims. Both are bound to one transcript.

#### 5. Opening the table through Akita

The table must be committed and opened under the following constraints:

- Akita is the only commitment scheme in the proof: no hash-based or Merkle commitment, and no second scheme;
- the only binary field is GF(2^128);
- with the feature off, Akita proof bytes are unchanged.

| Route | Status | Cost |
|---|---|---|
| Akita binary-source mode (LaBinius-style commitment to bit tables, grouped with the prime sources in one opening; LayerZero-Labs/akita binary-source PRs, e.g. #331, #332) | The intended route. One session and one opening for prime and binary sources. Its root and suffix currently run in a larger binary extension field and in prime-field stages, and the grouped profile has no production security parameters yet. | Open; the requirement is a GF(2^128) root, or a stated reduction from the larger-field stages to GF(2^128) claims. |
| A standalone LaBinius instance for the table | Works, but it is a second commitment instance with its own field stages. | Its reduction alone is above 32 KB at the capacities above. |
| Reduce bit columns modulo 2 inside prime Akita | Ruled out as a shortcut. An odd-modulus residue has no well-defined parity: $\mathbb Z_q/2\mathbb Z_q = 0$ for odd $q$. So it needs a prime-side bridge proving the lift. | About 24 to 29 KB with a linear verifier, or 1.6 to 2.3 MB if made succinct. |
| Full disclosure of the records | Sound and simple; this is the reference opening for testing and for the first end-to-end run. | About 10 KB per slot, so megabytes at the block capacities. Not deployable. |

Two facts limit the design space; the proofs are short and can go in an appendix if useful:

- **No cyclotomic embedding.** GF(2^128) does not embed in the cyclotomic rings Akita uses, so Akita's own opening cannot answer GF(2^128) claims directly.
- **Restricted impossibility.** A protocol whose only source authentication is Akita's existing prime opening cannot reach an all-GF(2^128) terminal by changing its front end or reducing residues modulo 2. This is not a general impossibility: a succinct argument for the Akita relation, written as a Boolean circuit over GF(2^128), is not excluded.

#### 6. Fixed capacity and overflow

`K` is a public parameter of the proving key, a power of two. The tracer counts calls and stops at call $K+1$ with `PRECOMPILE_CAPACITY_EXCEEDED: KECCAKF capacity K`, before any memory effect of that call and before proving starts. There is no automatic retry at a larger `K`: retries would leak a capacity threshold. A fixed `K = 2,048` covers small programs only; Ethereum blocks need $2^{15}$ to $2^{18}$, so the default for a block-proving key should be $2^{18}$. Occupancy on the blocks above is 0.50 to 1.0.

#### 7. The projection property

**Statement.** For every trace that satisfies the trace constraints plus the `PCALL` port and family contracts, contracting each complete `PCALL` interval to one step gives the execution of the same program in which that `PCALL` is an atomic application of Keccak-f[1600] to its entry RAM state. The link argument needs this property: it is what rules out a dropped call, a doubled record or a call spliced across two intervals.

The property is proved for an abstract machine under seven preconditions, which the verifier must enforce:

1. Every expanded bytecode row has a unique 64-bit PC. Text PCs are $0 \bmod 4$ and expansion PCs are $1 \bmod 4$.
2. Every expansion row has exactly one owning `PCALL`. The call row enters its own first interior row, each interior row has exactly one successor, and the return row goes back to the next text PC.
3. Entry and final PCs are text PCs, and the trace's final successor is bound to the final PC.
4. Every `JALR` clears bit 0 of its target. With odd expansion PCs, this needs an explicit constraint, $\mathrm{IsJALR} \cdot \mathrm{NextPC}_0 = 0$.
5. Lane and tag come from authenticated decode, as above.
6. The ports follow their declared order, bases, counts and directions, with distinct aligned writes and unchanged registers.
7. No continuation boundary falls inside a `PCALL` interval.

Items 3 and 4 need new constraints in prime Jolt. A complete replay lemma for expanded bytecode, which would show that the implemented verifier enforces items 1, 2 and 5, is still owed. It is listed under Open questions.

### Verifier changes

With the feature on, the verifier gains:

- registry checks at preprocessing: the key is unique, the expansion is authenticated and lane and tag are decoded;
- the constraints of the projection property (items 3 and 4 above);
- the family relation sum-check in GF(2^128);
- the link sum-checks: coverage and limb agreement;
- the table's opening claims, added to the existing Akita opening batch.

With the feature off, none of this runs and the verifier is unchanged.

### Soundness obligations

1. Soundness of the GF(2^128) family relation: the sum-check error over $2^{128}$, times the number of rounds and the degree, per slot batch.
2. Binding and extraction of the table opening; this is open (section 5).
3. The projection property's seven preconditions, enforced by the implemented verifier, and the replay lemma.
4. Uniqueness of the limb decomposition: limbs below $2^{32}$ and a prime above $2^{32}$.
5. Coverage multiplicity exactly one in the prime link.
6. Transcript binding: the binary and prime challenges are drawn from one transcript after both commitments are absorbed.

### Costs

**Rows** (measured):

- −34.62% on five Ethereum blocks;
- −3.32% on a recursive-verification guest;
- 44 rows per call against 3,121 today.

**Bytes**, against a target of at most 4,096 extra bytes per precompile per proof:

| Message | Bytes | Basis |
|---|---|---|
| Family relation transcript | 2,552 at $K = 2^{11}$; grows with $\log K$ | measured |
| Link (pointer coverage and limb agreement), unbatched | about 3,400 / 6,800 / 13,600 at 8- / 16- / 32-byte prime elements | forecast |
| Link and agreement at block capacities | about 15 to 20 KB at 16-byte elements | forecast |
| Table opening | open | depends on section 5 |

So the forecast is already above 4,096 bytes before the opening. Meeting the target needs the link claims folded into the existing Akita batch rather than sent separately, and an opening that adds only a few hundred bytes. If that is not achievable, the trade (about a third fewer rows against tens of KB) is a decision for maintainers. Measured byte figures from the first end-to-end proof will be added to this spec.

**Binary-side work.** At $K = 2^{17}$ the table is about 11 Gbit, or 86M GF(2^128) elements. That is about 39 table bits per trace row removed, counting padding. With the opening stubbed, the relation prover took 7.25 s at $K = 2^{11}$; prover time at block capacities is not yet measured.

### Rollout

1. Behind the `pcall-keccakf` cargo feature, off by default. The tracer, prover and verifier compile the path only with the feature, and default builds are byte-identical.
2. Land in order: the `PCALL` instruction and registry with the capacity error; the record table and relation; the link; the opening. Each step has its own tests, and the feature remains unusable for proving until the opening lands.
3. Turn on by default only after the opening is settled, the bytes are measured on real blocks, and the maintainers accept the byte cost.

### Alternatives Considered

- **Keep Keccak as an inline.** No new protocol, but 3,121 rows per call and about a third of the trace on blocks.
- **Prove Keccak in the prime field with lookups.** Cheaper than RISC-V rows, but bitwise operations in a prime field cost per-bit columns or lookup tables. The binary table proves $\chi$ and the linear steps natively.
- **A standalone binary commitment appended to the proof.** Simple, but it adds a second commitment instance and costs more than the byte target (section 5).

## Documentation

- A book page under `book/src/how/` on precompiles: the ABI, the capacity error and how to enable the feature.
- An update to the inlines page, noting that `KECCAKF` replaces the Keccak inline when the feature is on.

## Execution

Prototypes exist for each piece, except the link (stubbed) and the opening (stubbed). They are default-off and were tested on a recursive-verification guest and on Ethereum blocks in the emulator:

- the instruction, expansion and capacity error;
- the prime-side rows, against the real verifier, with mutated proofs rejected;
- the GF(2^128) relation.

The first end-to-end proof on a real block uses the full-disclosure reference opening. Its measured rows, time and bytes will be added here.

## Open questions

1. **Opening.** Is there an all-GF(2^128) opening on Akita's binary-source mode, for example a GF(2^128) root for the grouped session, or a sound reduction of its larger-field stages to GF(2^128) claims? Or would reviewers accept those internal stages for this table?
2. **Bytes.** Is 4,096 bytes per precompile the right target for a third fewer rows on blocks? Or should the target scale with rows saved?
3. **Capacity.** One `K` per proving key, or a small set of keys at $2^{15}$ to $2^{18}$?
4. **Key.** Is $(0\text{x}0b, 0\text{x}08, \cdot)$ acceptable as the precompile block, or should precompiles get their own opcode?
5. **Replay lemma.** Who owns the replay lemma for expanded bytecode (projection item 3), which also matters for existing virtual sequences?

## References

- Akita (lattice PCS): https://github.com/LayerZero-Labs/akita, binary-source PRs #318 to #332.
- M. Osadnik and G. Seiler, *LaBinius: Fast Lattice-Based Binary Polynomial Commitment Scheme*, IACR ePrint 2026/2103.
- `specs/binary-field.md` (the GF(2^128) type), `specs/binary-sumcheck.md`, `specs/lattice-claims.md`.
- NIST FIPS 202, *SHA-3 Standard: Permutation-Based Hash and Extendable-Output Functions* (Keccak-f[1600]).
- Inlines: `book/src/how/optimizations/inlines.md`.
