# Lane W4-T1 — T1 production hash table (`crates/jolt-wrapper/src/hash_table/`)

Date 2026-09-02 · tree wrap/spartan-hyperkzg · Mac mini (10 threads, 16 GiB, shared with 3 other
lanes' builds; load 4–13 during the timings). Real fixture: fibonacci 2^18 (197,595 trace rows,
K = 13, σ = 11), `Blake3Transcript` proof, cached at
`/Volumes/Dev/scratch/wrapper-fixtures/fibonacci_2_18_blake3.bin` (189 KB: verifier preprocessing +
public IO + proof, bincode/serde).

## Result in one screen (fibonacci 2^18, measured)

| item | value |
|---|---:|
| in-table compressions C_tot | **1,819** = 267 (41 commitments, incl. the 22-byte preamble tail) + 1,017 (stages 1–7 + stage-8 RLC) + 535 (Dory: 68 GT, 35 G1, 34 G2, 24 squeezes) |
| squeezes (challenge rows) | 376 = 352 hidden + 24 Dory |
| rows | **219,784** = 120·1,819 + 4·376 → **2^18** (83.8 % full) |
| committed bit columns | **163** (A' 32, D' 32, C' 32, B' 32, κ0 κ1 κ2, m 32) |
| wired columns | 64 bits (din, bin) + 3 words (a_in, c_in, rot_d) |
| constraints / degree | 229 (163 booleanity, 64 XOR, 2 adds); **degree 3** with the row `eq` (2 in the columns) |
| absorbed bytes linked | 107,104 + 22 public preamble bytes + 9,290 zero padding = 116,416 links (64 per compression) |
| link table by source | Fr wires 1,199 × 32 = 38,368 B · labels 737 × 32 = 23,584 B · commitment GTs 41 × 384 = 15,744 B · Dory GT 26,112 B, G1 1,120 B, G2 2,176 B · public 22 B |
| witness generation | chain replay + byte-exact check 4 ms · table build 0.30 s (single-threaded) |
| row sumcheck (prove_batch, 18 rounds) | **0.505 s** (round 0 on bits 0.087 s) |
| verify of the recorded run | 0.13–0.18 s |

Formula (rows = 120·C_tot + 4·S, S = squeezes):
`C_tot = ⌈(P mod 64 + 416·n_c)/64⌉ + C_hidden(L, K) + 47σ + 18`, with P = preamble bytes (1,046 for
fibonacci → 22-byte tail), n_c = 41 commitments, C_hidden = M3's C3 − 1 (M3 charged the first
stage-1 squeeze as an empty block; in the chain it finalizes the commitment tail) = 1,017 (L=18) /
1,053 (L=20), Dory 47σ + 18 = 43 (vmv + round-0 first messages, 2,720 B) + 18 (second messages,
1,152 B) + 47·(σ−1) + 1 (γ, empty) + 3 (final_e1 ‖ final_e2, 160 B). **L=20 (K=16, σ=12):
C_tot = 267 + 1,053 + 582 = 1,902; S = 366 + 26 = 392; rows = 229,808 → 2^18 (87.7 %).**

## Layout

Row = one half-G step `a' = a + b + m; d' = (d ⊕ a') ⋙ r1; c' = c + d'; b' = (b ⊕ c') ⋙ r2`
(r1, r2 = 16, 12 on the first half, 8, 7 on the second), lane N3's word layout (`relation.rs`
`Layout::Word`, ported to `layout.rs`; the bench's random-data copy is superseded):

| columns | role |
|---|---|
| 0..32 A', 32..64 D', 64..96 C', 96..128 B' | committed bits; D' = d ⊕ A' and B' = b ⊕ C' are stored **un-rotated** — rotations are re-indexings in the wiring |
| 128 κ0, 129 κ1, 130 κ2 | carries: `Σ A'2^k + 2^32κ0 + 2^33κ1 = a_in + Σ bin 2^k + Σ m 2^k`, `Σ C'2^k + 2^32κ2 = c_in + rot_d` |
| 131..163 m | the row's message word (committed; every block word appears in 7 rows) |
| wired: din_k @ A'_k, bin_k @ C'_k (column space), a_in 163, c_in 164, rot_d 165 | copies through the wiring (`Feed`): `Word{column,row,rot}` = bit k ← bit (k+rot) mod 32 of another row's word; `Const` (IV, block length, flags, counter 0); `StateIn(i)`; `Zero` |

Per compression 120 rows (124 with a squeeze): 112 half-G rows (index `(round·8 + g)·2 + half`),
8 chaining rows (`D' = v[j] ⊕ v[j+8]` with `A' = v[j+8]` copied through `a_in`, everything else zero
— one XOR per row, so the same relation serves), and for a squeeze 4 challenge rows
(`D' = v[8+i] ⊕ cv[i]`, the 16 challenge bytes). The next compression's `cv` words are read from the
chaining rows' D' (words 0..3 as `a_in`, 4..7 as `bin`); the first compression reads
`StateIn(i)` — the public input. Every compression has counter 0, `block_len` ∈ {0..64} and flags
⊆ {CHUNK_START, CHUNK_END, ROOT, KEYED_HASH} as profile constants of its round-0 rows.

Feeds (`table.rs`): a diagonal step's first half reads the column step of the **same round** that
wrote the index (the bug the relation test caught: `previous_round` for g ≥ 4 fails on rows 8, 10, 12,
14 of every compression); a column step reads the previous round's diagonal step; `b`/`d` words carry
the second half's rotations (7, 8); the second half of a step reads its first half (rotations 16, 12).

Chain (`blake3.rs`): keyed segments as `Blake3Transcript` — lazy block compression (a full block is
compressed when the 65th byte arrives, so the last block of a segment always carries CHUNK_END|ROOT),
1,024-byte close, squeeze = finalize + 64-byte root output (`out[0..8]` = next key, `out[8..12]` =
challenge). Byte-exact with the `blake3` crate (unit tests) and with the recorded verifier run:
`JoltSchedule::new` compares the model's `state()` (finalize of the pending segment) with the
recorded `Transcript::state()` after **every** append and squeeze and the decoded challenge with the
recorded value (`Decoder::Challenge125` / `Scalar128`), failing closed.

Segment (`schedule.rs`): the table starts at the block holding the first
`LabelWithCount(b"commitment", 384)` word — for fibonacci that is chunk 1 (the preamble is 1,046 B),
so `state_in` = the key after the 1,024-byte close and the first block carries a 22-byte public
preamble tail (`ItemClass::Public`) — and ends at the block of the last squeeze (Dory `d`). The
stage-8 RLC γ squeeze is the last squeeze before the first `b"dory_serde"` label word
(`rlc_block`; its chaining rows are `state_rlc`).

## Link table (`HashTable::links`, `challenges`, feeds)

`MessageLink { row, byte, origin: Option<ByteOrigin{item, offset}> }` — one per first-use message
byte (round-0 rows; rounds 1–6 are `MessageSource::Copy{row}`), `None` = zero padding of a partial
final block. The item's `ItemClass` gives the external source:

| class | encoding / equation |
|---|---|
| `Wire{index}` | the index-th absorbed Fr (32 B big-endian): word w, bit k ↔ Fr bit `8·(31 − 4w − k/8) + k%8` |
| `Element{CommitmentGt, i}` | `Bn254GT::append_to_transcript`: arkworks-uncompressed Fq12 (12 × 32 B LE coefficients, c0-first nesting) **reversed**: byte b ↔ coefficient `11 − b/32`, its byte `31 − b%32`; word w ↔ 2 consecutive 16-bit chunks |
| `Element{DoryGt, i}` | `serialize_compressed` Fq12 (= uncompressed, not reversed), 384 B: word w ↔ coefficient `w/8`, bits `32·(w%8)..+32` = chunks `2(w%8)`, `2(w%8)+1` of the 96-bit-limb decomposition (limb ℓ = words 3ℓ..3ℓ+3) |
| `Element{DoryG1, i}` | compressed x (32 B LE) — top 2 bits of byte 31 are the y-sign/infinity flags |
| `Element{DoryG2, i}` | compressed x ∈ Fq2 (c0 ‖ c1, 64 B LE), flags in the top 2 bits of byte 63 |
| `Constant` | label / `LabelWithCount` word (protocol constant; 737 in the segment) |
| `Public{item}` | preamble bytes (22) |

`ChallengeLink { item, rows[4] }`: challenge bytes `4i..4i+4` = D' of `rows[i]` (LE); the decoder
is `log[item]`'s. `state_in`: `Feed::StateIn(i)` in the first compression's round-0 rows;
`state_rlc` / `state_out`: `chaining_rows(rlc_block)` / `chaining_rows(last)` (D' words) — the fixture
test checks `blake3::keyed_hash(state, "") == recorded state()` for both.

Counts at 2^18: 116,416 message links (1,819 × 64), 376 challenge links, 1,199 wires, 178
elements, 1 state_in (8 words), 2 state outputs.

## Prover (`prover.rs`) and `final_check`

`HashTableProver: jolt_sumcheck::prover::ProveRounds<Fr>` (fused bind+eval): round 0 on the 0/1
columns (γ-bucket sums, no field multiplications), rounds ≥ 1 on bound Fr columns with
`s(X) = eq_prefix · l(X) · t(X)` and `t(1)` recovered from `previous_claim`; row-index bits bound
LSB-first (round i binds τ[n−1−i]); after `finish_rounds`, `column_evals()` returns the 163 + 64 + 3
values. `Relation::final_check(τ, challenges, evals) = eq(τ, r)·Q(v, w)` is the verifier's expected
final claim (checked equal to `prove_batch`'s `final_claim` on the real table; input claim 0).
Memory: 230 Fr columns × 2^17 after round 0 ≈ 1 GB.

## Tests

- `cargo nextest run -p jolt-wrapper --cargo-quiet -E 'test(hash_table) | binary(hash_table_relation)'` —
  `blake3.rs` unit tests (keyed single-chunk states/XOF vs the `blake3` crate, origin coverage) and
  `tests/hash_table_relation.rs` on a synthetic Jolt-shaped transcript (labels, raw Fr appends,
  both decoders, 384/32/64-byte elements, an opening claim after the last squeeze): chain replay,
  chaining/challenge rows hold the recorded words, links cover every absorbed byte once with 4 links
  per first-use row and consistent copies, relation through `prove_batch` + `final_check`, 8
  random single-bit flips rejected.
- `cargo nextest run -p jolt-wrapper --features prover-fixtures --cargo-quiet -E 'binary(hash_table_fixture)' --no-capture`
  — the real proof (generated and cached on first run: guest build + trace + prove 11.4 s):
  byte-exact chain incl. the Dory segment (every state, every challenge), pinned counts
  (1,819 = 267 + 1,017 + 535 compressions, 376 squeezes, 1,199 wires, 41/68/35/34 elements,
  219,784 rows, 2^18), link completeness, `state_rlc`/`state_out` bind the recorded transcript, the
  full-size row sumcheck (0.505 s) with `final_check`, 3 bit flips rejected.

Wave-4 integration hooks for W4-S: `HashTable { bits, wired_bits, wired_words }` are plain columns
(u8 bits / u32 words, 2^log_rows each); the wiring is `feeds[row]` (+ `message_sources`) — S2's public
matrices are exactly these feeds; the relation's aligned quadratic form (`Relation`) is what the
column sumcheck reduces.

## Hygiene

- Built and tested in the scratch worktree `/Volumes/Dev/worktrees/jolt/w4t1-verify` (own target
  dir `/Volumes/Dev/cargo-target/w4t1-verify`) per the 21:08 rule; the shared tree had other lanes'
  uncommitted compile errors throughout. `cargo clippy -p jolt-wrapper --all-targets [--features
  prover-fixtures] -- -D warnings`, `rustfmt --check` and `scripts/check_style_invariants.py --base
  HEAD` clean on this lane's files.
- `crates/jolt-wrapper/Cargo.toml`: added `blake3`; the `prover-fixtures` feature (jolt-prover,
  jolt-prover-legacy/host, jolt-witness, postcard) was added concurrently by W4-S in the shared
  tree — this lane only added `tracer` to it (`Program::trace_with_backend` needs a `TracerBackend`).
- `crates/jolt-wrapper-bench` keeps its word-layout copy (N3's bench binary still builds); the
  production owner of the relation is now `hash_table/layout.rs`.

## Fix #1 (review #1 blocker + major + minors; W5 / W6-RT interface), 2026-09-03 03:10

Measured on the real fibonacci 2^18 fixture (M) unless tagged (E). Commands unchanged (see Tests).

### What changed

- **Cells.** Every compression is a 128-row cell (`b · 128 + p`): p 0..112 half-G steps
  (`(round · 8 + g) · 2 + half`), 112..120 chaining rows, 120..122 challenge rows (two output
  words each: `D' = out[8 + 2i]`, `B' = out[9 + 2i]`, `A' = cv[2i]`, `C' = cv[2i + 1]`, computed by
  EVERY cell — only squeeze cells are linked), 122 / 123 hold the next cell's block length / flags
  in `m`, 124..128 zero. Padding cells continue the chain (`Chain::pad`: empty block,
  `CHUNK_START | KEYED_HASH`) so the wiring is uniform on the full 2^18 domain. Real table: 1,819
  active cells of 2,048 (2^18 rows, 88.8 % active; L = 20 (E): 1,902 cells → 2^18).
- **Wiring = a constant table + 4 verifier-key columns + public inputs** (`wiring.rs`).
  `source(position, slot)` is the position table (Blake3 lanes, message permutation, rotations,
  chaining, challenge decoding): `Cell{group, weights, δ}` / `Previous{position}` /
  `Next{position}` / `Const(IV)` / `Zero`, 128 × 17 entries, held by the verifier as code. Block
  length, flags, label words and zero padding are half-word pins of `m` in the verifier key:
  `lo_is_const`, `hi_is_const` (bit), `lo_const`, `hi_const` (u16) — 31,957 pinned half-words
  (23,584 constant bytes + block-length / flag words of every cell). Public inputs: `state_in`
  (8 words), the first cell's block length / flags, the 22-byte preamble tail (pinned as half-words
  of block 0 by the verifier). Every `(position, slot)` copy is a shift kernel with fixed δ inside
  the cell or `eq+1` across cells: 724 entries, 537 distinct `(position, δ, cell offset)` kernels,
  30 distinct value forms `(slot, group, weights)`.
- **Copy constraints = one degree-3 zero-check member in stage A** (`wiring_prover.rs`,
  `WiringStatement`): `Σ_row eq(τ₂, row) · [Σ_s γ_s w_s(row) + γ_lo·is_lo(row)·lo(m)(row) +
  γ_hi·is_hi(row)·hi(m)(row) − pinned constants] − Σ_t P_t(row) · V_t(row) = public constant`,
  where each read side `t` is a public kernel table `P_t` (the `eq(τ₂, ·)` weights of the row's
  readers, slot coefficient folded in) times the value form `V_t` (fixed linear combination of the
  source group's 32 bits), bound as separate multilinears. Head-aligned with the row relation:
  0 extra rounds, same point r_A, wired columns stay committed (co-pointing). Public part of the
  sum (IV constants, cell-0 chaining reads): O(#entries) verifier work. Message copies (rounds 1–6)
  are a wired word `m_in` fed by 96 single-position kernels; the ternary add reads `m_in`.
- **Verifier work (T1, per proof, E from counts):** two 7-variable eq tables (2 × 127 mults),
  `eq(τ_hi, r_hi)`, two `eq+1` on 11 variables (≈ 90), `eq(τ₂, r)` (128), kernel weights
  724 × 2 = 1,448, value-form expansion 30 × 32 × 2 = 1,920, wired side 79 + pins ≈ 50, row relation
  ≈ 3 × 256 → **≈ 4.5k Fr mults**, no O(rows) data. VK: 4 columns (+4 claims in the batched
  opening); proof: 0 extra rounds, 0 extra claims beyond the columns.
- **Exported final relation as terms** (`terms.rs`): `terms(&FinalContext) -> Vec<Term>` with
  `AffineForm { constant, weights: Vec<(ColumnId, Fr)> }`, `Term { coefficient, factors }` over
  column ids (0..163 committed bits, 163..227 wired bits, 227..242 wired words, 242..246 VK):
  **T = 230 terms, max d = 2** — 163 booleanity squares, 64 XOR cross terms, 2 half-word pin
  products, 1 linear term (adds, wired side, kernels, pinned constants, public tail). Test oracle:
  `Σ_t coeff·Π L(v) == ρ_rows · Relation::final_check + ρ_wiring · WiringStatement::final_check
  == prove_batch final claim` on the real table.
- **Virtual value columns for W6-RT** (`AffineForm`s at r_A): `challenge125()` /
  `challenge_scalar128()` at a squeeze cell's row 120 (D', B' bits + wired `a_in = out[10]`,
  `x_in = out[11] mod 2^29`, `y_in = bswap(out[10])`, `z_in = bswap(out[11])`; coefficients probed
  from the production decoders `Fr::from_challenge_bytes` / `from_scalar_challenge_bytes`, which
  stay the owners); `fr_word()` at an aligned wire row (own `m` bits bswapped + `fr_next[1..8]` =
  bswap of the next 7 rows' `m`); `fr_word_shifted()` for the wires absorbed before the first
  squeeze (2 bytes into their word after the 22-byte tail: high half of `m`, `fr_next`, `fr_tail`
  = bswap16 of the low half 8 rows on, across the cell boundary via `Next`). Every recorded
  challenge (376) and wire (1,199) evaluates exactly (tests).
- **Link identities** (`schedule.rs`, `LinkMap`): `ByteSource` per block byte — `Padding`,
  `Constant(byte)`, `Public{preamble offset}`, `Wire{index, byte}`, `Element{kind, index, byte}` —
  from the `SymbolicSchedule` (per cell: block_len, flags, 64 sources, squeeze). Generated with the
  verifier key from one recorded verifier run of the profile; deterministic in the profile (test:
  two proofs of the same shape give identical schedules and VK columns); the linker never reads
  witness bytes. Row maps: 1,199 wire rows (aligned / shifted), 376 challenge rows, 45,174
  element/public byte positions (41 × 384 CommitmentGt, 68 × 384 DoryGt, 35 × 32 G1, 34 × 64 G2,
  22 public). Commitment-segment elements are offset 2 bytes in their words (T2's byte links).
- **Borrowed columns:** `HashTableProver::new(&relation, &table, τ)` and `WiringProver::new(..,
  &bits, &wired_bits, &wired_words, &vk, &public, τ)` borrow; no clone before packing.
  `column_specs()`: 163 bit + 64 bit + 15 u32 committed (packing order) + 4 VK (2 bit, 2 u16);
  `members(log_rows)`: `t1-rows` (degree 3) and `t1-wiring` (degree 3), offset 0.
- **Minors:** table-local `CellIndex` (`rlc_cell`, `last_squeeze_cell`); `Cargo.lock` carries the
  fixture's `tracer` dependency.
- Consumers ported (3 call sites): `wrap.rs` (W5) and `tests/perf1_profile.rs` (PERF-1) use
  `JoltSchedule::new(log, log_rows)` / `HashTable::build(&schedule)` / borrowed provers.

### Numbers (fibonacci 2^18, M)

| item | value |
|---|---:|
| cells / rows | 1,819 active + 229 padding = 2^11 cells × 128 = 2^18 rows |
| committed columns | 163 bits + 64 wired bits + 15 wired words = 242 (+4 VK) |
| stage A (2 members, 18 rounds, degree 3) | 0.753 s (row member round 0 on bits 0.096 s, wiring setup 0.372 s) |
| witness | replay 0.004 s, build (incl. wiring materialization) 0.301 s |
| terms | 230, max degree 2, built in < 1 ms |
| kernels | 724 entries, 537 distinct, 30 value forms |
| pins | 31,957 half-words; 22 public tail bytes |

### Tests

- `cargo nextest run -p jolt-wrapper --cargo-quiet -E 'test(hash_table) | binary(hash_table_relation)'`
  (6): chain + cells hold the recorded words, every squeeze / wire evaluates through the virtual
  columns; byte identities cover every absorbed byte once + VK determinism across two proofs;
  both members through `prove_batch`, native final checks and exported terms agree; tampers: 8
  random state/carry bit flips (row relation), round-0 message bit / mis-routed `din` / mis-routed
  `m_in` / forged label byte / forged preamble byte / forged next-block length (wiring), kernels at
  a shifted stride (verifier formula).
- `cargo nextest run -p jolt-wrapper --features prover-fixtures --cargo-quiet -E 'binary(hash_table_fixture)' --no-capture`:
  the real proof, pinned shape, links, `state_rlc` / `state_out`, both members at full size, terms
  vs native, 3 flips.
- Gates (scratch worktree `w4t1-verify`): `cargo clippy -p jolt-wrapper --lib --test
  hash_table_relation --test hash_table_fixture --test perf1_profile [--features prover-fixtures]
  -- -D warnings`, rustfmt, `check_style_invariants.py --base HEAD` clean on this lane's files
  (open findings remain in `limb_table/` — W4-T2).
