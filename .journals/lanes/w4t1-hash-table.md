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

## Fix #2 (review #2: B1 canonical Fr words, B2 VK-time schedule, B3 FS randomizers, MAJOR shared term API), 2026-09-03

Both reviewers' 3 blockers / 1 major / 2 minors, each blocker with a verifier-path negative test
(`prove_assembly` by the honest prover code on the tampered run, statement claims taken from the
provers = a self-consistent adversary; `verify_assembly_with_cost` with the key's statement rejects).

### B1 — canonicality of every absorbed Fr word

- `fr_word()` / `fr_word_shifted()` are linear in the bytes, so `x` and `x + r` (`x + 2r`) had the same
  value column (the reviewer's alias test, now `wire_value()` in `hash_table_relation.rs`, asserts
  the alias still holds for the value column — the constraint below is what kills it).
- Constraint (wiring member, degree 3 with `eq`): `sel · (Σ_{k<64} 2^k canon_k + w_hi) − (r_hi − 1) · sel = 0`
  per alignment, `sel ∈ {wire_aligned, wire_shifted}` (VK bit columns), `canon_k` 64 committed bit
  columns (`layout::CANON..COMMITTED`, booleanity in the row relation), `w_hi` = the encoding's top
  64 bits as a linear form (`wiring::canonicality(shifted)`): aligned `2^32 · bswap(m) + fr_next1`,
  shifted `2^48 · bswap16(hi m) + 2^16 · fr_next1 + fr_hi2` (new wired word `WiredWord::FrHi2`,
  `Weights::BswapHi16`, sources at positions 5/13). `r_hi = MODULUS_HI = 0x3064_4e72_e131_a029`.
- Soundness: accepted ⇒ `w_hi ≤ r_hi − 1` ⇒ value `< r_hi · 2^192 < r` ⇒ canonical; every `x + k·r`
  has `w_hi ≥ r_hi`. Completeness loss: canonical `x ∈ [r_hi · 2^192, r)` are rejected —
  `(r mod 2^192) / r ≈ 2^-62` per wire; `2^-62 · 1,199 ≈ 2^-52` per proof. Every real fixture wire has
  `top-64 < r_hi` (asserted). Witness: `d = (r_hi − 1) − w_hi` (wraps for non-canonical bytes, so the
  sum misses by a multiple of 2^64 → wiring claim ≠ public constant).
- Cost: committed bits 227 (+64) → 291 with wired bits = 19 groups of 16 (was 15) → +4 commitments
  (+128 B at 32 B/G1); wired words 16 (+1, still 1 group); VK 4 bits + 2 u16 (+2 bits, still 2 groups).
  Stream columns 352 in 22 groups (was 18). Terms 230 → 296 (+64 booleanity + 2 canonicality
  products) → term stage 8 → 9 rounds (~+64 B). Stage A on the real 2^18 table 0.753 → 0.88–1.27 s (two runs).
- Negatives: `noncanonical_wire_bytes_are_rejected` — synthetic runs with one element encoded as
  `1 + r` and `1 + 2r`, at a shifted wire (round 0) and an aligned wire (round 1): value column
  aliases `1`, `Fr::from_bytes_le_checked` rejects, the honest verifier rejects the proof; the
  untampered run verifies (`key_schedule_is_proof_independent`). `tampered_tables_are_rejected` adds a
  forged `canon` bit. `modulus_hi_is_the_top_word_of_the_field_modulus` pins the constant.

### B2 — the verifier's schedule is the key's

- `SymbolicSchedule::from_reference(log, log_rows)` (key generation; `label`, `tail_len`, cells, byte
  identities, wires, squeezes, VK columns — no run values), `JoltSchedule::witness(log, &key)` (the
  proof's run replayed and compared to the key: `ScheduleError::ShapeMismatch`),
  `PublicInputs::from_preamble(preamble, &key)` (the verifier hashes the public preamble natively;
  `ScheduleError::PreambleTail`), `schedule::preamble(log)`, `HashTable::build(&schedule, &public)`.
  `JoltSchedule::new` / `public_inputs()` removed; `wrap.rs` ported (`hash_key`, `hash_public`).
- Negatives: `key_schedule_is_proof_independent` — proof B (another run, other values) verifies under
  key A's statement/exporter; B's proof under A's public inputs rejected; runs with one more
  commitment / one more round → `ShapeMismatch`; a 24-byte tail → `PreambleTail`. Real fixture:
  key from the reference run, witness checked against it.

### B3 — randomizers from the stream transcript

- `T1Challenges::from_challenges(&context.challenges[offset..offset + count], log_rows)`,
  `count = 2 · log_rows + 2` (`τ_rows`, `τ_wiring`, relation batching γ, wiring slot γ — powers);
  `CommitmentPhase { group_count: 22, challenge_count: 38 }` for 2^18. Prover: `Members::new(&table,
  &relation, &challenges)` after `commit_packed` → `commitment_prefix_challenges`; verifier:
  `StreamTermExporter { log_rows, challenge_offset, public, columns: &ids, row_member, wiring_member }`
  derives the same from `TermContext::challenges`; `T1Challenges::input_claims(&public)` are the
  statement's member claims.
- Negatives: `randomizers_are_bound_to_the_commitments` — members built from randomizers not drawn
  from the transcript → `prove_assembly` = `Err(StageLink)` (the exporter's terms don't link);
  re-ordered commitments in an honest proof → rejected.

### MAJOR + minors

- `adapter.rs`: `StreamColumns` (packing order: bits incl. wired + canon, u32 words, VK bits, VK u16;
  `ids[local] → stream::ColumnId`), `Members`, `StreamTermExporter: stream::TermExporter`
  (`terms_observed` routes every verifier multiplication through `TermObserver::fr_mul`; 3,278 Fr
  mults on the real table, budget 5k). `terms::terms(&ctx, &mut mul)` takes the multiplication.
- Removed public diagnostics: `column_specs`, `members`, `kernel_counts` (test-local now),
  `WiringProver::final_parts`; nominal imports fixed (`CanonicalEncoding`).
- Binding order: T1's members now bind the top row variable first (`HighToLow`, the stream's
  `column_evaluations(row_point)` convention); `eq_rounds` pairs `τ[i]` with round `i`; the verifier
  formulas take `row_point` as the big-endian point (no reversal). T1 members ride under the stream's
  degree-5 stage encoding (`StageMember.degree = 5`, own degree 3).

### Real fibonacci 2^18 (`hash_table_fixture`)

| | |
|---|---|
| columns | 227 committed bits + 64 wired bits + 16 wired words + 6 VK → 352 stream columns / 22 groups (k = 16) |
| terms | T = 296, d = 2; 3,278 verifier Fr multiplications (terms), kernels 31 distinct `(slot, group, weights)`, 726 entries, 14 value forms |
| wires | 1,199 aligned + 0 shifted (shifted alignment exercised by the synthetic profile), all canonical with slack |
| stage A | 0.88–1.27 s over two runs (members setup 0.68–0.92 s); witness replay 0.01 s, build 0.31–0.47 s |
| 2^18 verifier cost (synthetic, whole assembly) | `ec_mul 146, pairing_pairs 8, fr_mul 9,933, keccak 347` at 2^13 rows |

### Tests / gates

- `cargo nextest run -p jolt-wrapper --test hash_table_relation` (8): chain + cells, byte identities +
  key determinism, members + terms (T, d, mults), tampers, MODULUS_HI, B1 / B2 / B3 verifier-path.
- `cargo nextest run -p jolt-wrapper --features prover-fixtures --test hash_table_fixture --no-capture`.
- Gates (scratch `w4t1-verify`): clippy `--lib --test hash_table_relation --test hash_table_fixture
  --test perf1_profile` with and without `--features prover-fixtures`, rustfmt, style checker clean on
  this lane's files. Pre-existing at HEAD: `relation_table/mod.rs` `identity_mle_observed` /
  `eq_mle_observed` are dead code → `cargo clippy -p jolt-wrapper --lib -- -D warnings` fails on
  main before this lane's changes (W4-R; scratch build carried a local `#[expect]`).

## Fix #3 (review #3: B1 shifted top-64 extraction, B2 key-committed VK groups, MAJOR observer-complete count, minors), 2026-09-03

### B1 — shifted canonicality read bytes 8–9

- A shifted wire starts at byte 2 of row `p`: bytes 0–1 = high half of `m(p)`, bytes 2–5 =
  `fr_next(1)` (word `p + 1`), bytes 6–7 = the **low** half of word `p + 2`. `FrHi2` read
  `BswapHi16` (bytes 8–9); renamed `WiredWord::FrLo2`, sourced with `BswapLo16` at position
  `p + 2`; `Weights::BswapHi16` deleted. `wiring::canonicality(true)` = `2^48 · bswap16(hi m) +
  2^16 · fr_next1 + fr_lo2`.
- Negatives (`noncanonical_{shifted,aligned}_wires_are_rejected`): `1 + r`, `1 + 2r`, the
  reviewer's carry-wrapping `x` (`x[8..10] = 0x47b0`, `x + r` prefix `…a02a0000`) and
  `carry_case(b)` for `b = 1..=8` (`x = 2^{8(32−b)} − (r mod 2^{8(32−b)})`: `x + r` clears bytes
  `b..32` and carries into byte `b − 1`, i.e. into every byte of the top-64 window) — all rejected
  through the verify path at both alignments; the value column aliases `x` in every case.

### B2 — the six verifier-key columns are key data

- `adapter::HashTableKey::new(schedule, packing, setup)` commits the VK groups (4 bit selectors
  `lo/hi_is_const`, `wire_aligned/shifted` + 2 u16 constants `lo/hi_const`, padded to `k`) ONCE
  from `SymbolicSchedule::vk_columns()`; `HashTableKey::pinned_commitments(group_offset)` =
  `(group index, commitment)` for `AssemblyStatement::pinned_commitments` (W5's stream mechanism,
  `517bf384d`): proofs omit those groups, `verify_assembly` splices the key's commitments, a proof
  carrying its own copies fails `StageCount`. `vk_group_range(packing, offset)` /
  `prover_group_count(packing)` own the group geometry; `StreamColumns { vk_groups }` reports it.
- Real 2^18 layout: **20 prover-sent groups** (19 bit groups + 1 u32 group) + **2 verifier-key
  groups** = 22 packed groups, 352 columns (k = 16). Proof bytes: −2 commitments (−64 B).
- Negatives (`prover_owned_vk_columns_are_rejected`): zeroed `wire_shifted` / `wire_aligned`
  with a `1 + r` wire, and a run of another profile (3 commitments, same rows and public inputs):
  proof with its own VK groups → `StageCount`; VK groups omitted → opened against the key's
  commitments → rejected. Positive: the key's pins equal the honest table's packed VK groups.

### MAJOR — execution-derived Fr count

- New `hash_table/eq.rs`: `eq_evals_with`, `eq_points_with`, `eq_zero_with`, `eq_plus_one_with`,
  `powers_with`, `pow2` (constants), `plain`; `Relation::new_with`, `WiringStatement::{slot_weight_with,
  input_claim_with}`, `T1Challenges::{from_challenges_with, input_claims_with}`; `terms(ctx, mul)`
  routes every multiplication (eq tables, eq+1, `eq(τ_hi, r_hi)`, `eq(r_hi, 0)`, tail products,
  shifted coefficients `γ · 2^k`, challenge powers) through `mul`. `eq_helpers_match_jolt_poly`
  pins the helpers to `jolt_poly` (independent oracle). Kernel cell factors (`cell · eqτ[p]`) and
  `γ_slot · 2^k` are computed once each.
- Counts (real 2^18, pinned exactly in `hash_table_fixture`): exporter (`terms_observed`)
  **4,206**, statement construction (challenge powers + wiring input claim)
  **705**, total **4,911** (review #3's estimate ≈ 4.6k before the levers). The
  earlier 3,278 omitted eq tables, eq+1, `powers`, shifts and the statement.

### Minors

- Terms: `v_j · (γ_sq v_j + γ_cross w_j)` for the 64 XOR operand columns — **T = 232** (was
  296), d = 2, term stage 9 → 8 rounds (−64 B); `members_hold_and_terms_match…` and the fixture
  keep the native `final_check` oracle equality.
- Docs: `HashTable::bits` = `COMMITTED` columns, `Relation` = `CONSTRAINTS` coefficients,
  high-to-low binding (`τ[i]` at round `i`), `SymbolicSchedule::from_reference` /
  `JoltSchedule::witness` in the schedule module doc.

### W5 hand-off (B3/B4 are W5's)

Key objects to store (all derived at key generation from one trusted reference run):
`SymbolicSchedule` (`from_reference`), `HashTableKey` (schedule + `vk: VkColumns` +
`commitments: Vec<Commitment>` of the 2 VK groups; `pinned_commitments(t1_group_offset)` into
`AssemblyStatement::pinned_commitments`), `PublicInputs::from_preamble(preamble, &schedule)`
per proof (public data), `LinkMap::new(&schedule)`, `StreamColumns::vk_group_range(k, offset)`,
`T1Challenges::count(log_rows) = 38` @2^18 as the T1 phase's `challenge_count`, member slots
(row, wiring; degree 5 stage envelope, offset 0), `challenge_offset`. Assembly order: pack T1's 20
prover groups + 2 VK groups (`StreamColumns::new(&table, k, offset)`; `table.vk` = the key's) →
`commit_packed` → phase challenges (`commitment_prefix_challenges` over the FULL commitment list,
key pins included) → `T1Challenges::from_challenges(&challenges[offset..offset + 38], log_rows)`
→ `Members::new(&table, &relation, &challenges)` → `prove_assembly` (proof omits the pinned
groups). Verifier: `full_commitments` (key pins spliced) → same challenges →
`T1Challenges::input_claims(&public)` as the two member claims → `StreamTermExporter` from
`TermContext::challenges`.

### Tests / gates

- `hash_table_relation` (11): + `noncanonical_{shifted,aligned}_wires_are_rejected`,
  `prover_owned_vk_columns_are_rejected`, complete own-randomizer proof (reviewer's
  `OwnChallengeExporter`) rejected, `eq_helpers_match_jolt_poly`.
- `hash_table_fixture`: T = 232, d = 2, 20 + 2 groups, exact Fr counts.
- clippy (`--lib --test hash_table_relation --test hash_table_fixture --test perf1_profile
  [--test wrap_real_t1_r]`, both feature sets), rustfmt, style checker clean on this lane's files.

## Fix #4 (review #4: MAJOR statement multiplies missing from `VerifierCost`, minors), 2026-09-03

- **MAJOR — one owner for the verifier's T1 statement:** `StreamTermExporter::input_claims(&self,
  phase_challenges, observer) -> [Fr; 2]` derives the randomizers and both member claims with every
  multiplication routed through the observer (`from_challenges_with` + `input_claims_with`); the
  plain `T1Challenges::from_challenges` / `input_claims` are the prover's path (`eq.rs` doc
  corrected). The test harness's verifier counts it with the stream's `VerifierCost`; the
  reviewer's repro is permanent: `verifier_cost_includes_statement_derivation` pins the synthetic
  2^13 total **`fr_mul = 10,666 = 9,963` (stream, 4,130 of them T1's exporter) `+ 703`
  (statement)**. Real 2^18 fixture (`hash_table_fixture`, through the owner method): exporter
  **4,206** + statement **705** = **4,911**; the production `VerifierCost` reports `stream + 705`
  once `WrapVerifierKey::statement` (W5, `wrap.rs:166-176`) calls
  `StreamTermExporter::input_claims` with the verifier observer instead of the plain pair and adds
  the result to the stream's cost — W5's file, not edited here. Measured on W5's real 2^18 e2e (`wrap_real_t1_r`, T1 real + R, this HEAD): the production
  `verify_wrapped_with_key` reports `fr_mul = 13,187` (stream work incl. T1's 4,206 exporter
  multiplies, without the 705 statement multiplies); the honest execution-derived count is
  **13,892** once the statement path is routed through `StreamTermExporter::input_claims`.
- **Minor — key geometry immutable:** `HashTableKey` fields are private (`schedule()`, `vk()`,
  `packing()`, `commitments()` accessors); `new` rejects a commitment count that is not
  `vk_group_range(packing, 0).len()` (`StageCount`), so `pinned_commitments` always covers every
  verifier-key group.
- **Minor — `FrLo2` doc:** "`bswap16` of the low half-word of `m` two rows later (block bytes 8–9 =
  bytes 6–7 of the field element)".
- Review #4's `canonicality_windows_match_every_representable_alias_class` added (both windows
  read exactly the top 64 bits for `x + k·r`, `k = 1..=5`; every alias's top word `≥ r_hi`).
- Gates: `cargo check -p jolt-wrapper --all-targets [--features prover-fixtures]` fails only in
  `tests/perf1_profile.rs` (PERF-1's file: five T2 `limb_table` API mismatches at HEAD —
  `Wiring` import, `Slot::y_sign`, changed arities — outside this lane); every other target
  checks. clippy clean on `--lib --test hash_table_relation --test hash_table_fixture
  --test wrap_real_t1_r` (both feature sets); `hash_table_relation` 13/13; fixture passes;
  rustfmt + style checker clean on this lane's files.
