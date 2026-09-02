# Lane M4 — BLAKE3 R1CS measurement

Date: 2026-09-02

## Result

One variable-CV, variable-block compression plus its 8-word chaining-value
feed-forward is **15,792 constraints, 15,568 witness variables, and 86,826
A/B/C nonzeros** over BN254 Fr. This is 30.1% of Blake2b's 52,416 constraints
per compression; BLAKE3 absorbs half as many bytes per block.

Implemented in `crates/jolt-r1cs/src/gadgets/blake3.rs`. The shared Boolean/XOR
primitive is in `crates/jolt-r1cs/src/gadgets/bit.rs`; the Blake2b pinned counts
remain unchanged.

## Compression shape

Representation: little-endian 32-bit words as Boolean R1CS variables. A
variable-variable XOR uses `(2a)·b = a + b - out`; XOR with a constant changes
the bit's negation flag without a row. Rotations reindex bits. Message words are
allocated once and reused through all seven message schedules.

`compress` returns the 16-word post-round state. `chaining_value` applies the
specified feed-forward `state[0..8] xor state[8..16]`.

| Class | Derivation | Constraints | Witness vars | A/B/C nonzeros |
|---|---:|---:|---:|---:|
| chaining-value Booleanity | 8·32 | 256 | 256 | 768 |
| message-block decomposition | 16·32 | 512 | 512 | 1,536 |
| additions/decompositions | 112·(34+1) + 112·(33+1) | 7,728 | 7,504 | 48,036 |
| round G XORs | (224−4)·32 | 7,040 | 7,040 | 35,206 |
| chaining-value feed-forward | 8·32 | 256 | 256 | 1,280 |
| **standalone total** | | **15,792** | **15,568** | **86,826** |

The 112 ternary additions each allocate 32 result bits and two overflow bits,
then add one linear recomposition row: **3,920 constraints / 3,808 variables**.
The 112 binary additions each allocate 32 result bits and one overflow bit,
then add one linear row: **3,808 constraints / 3,696 variables**. The overflow
bits encode the quotient above bit 31 at weights 2^32 and, for ternary sums,
2^33. No ripple carries or separate discarded carry-out are allocated.

The first four `d xor a` operations have constant parameter words, saving 128
XOR rows. A fixed IV makes the first four `b xor c` operations constant-variable
too, saving another 128 rows.

## Block modes

| Mode | CV input | Fresh block | Additions | Round XORs | Output | Constraints | Witness vars | Nonzeros |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| standalone | 256 | 512 | 7,728 | 7,040 | 256 | **15,792** | **15,568** | **86,826** |
| fixed-IV first block | 0 | 512 | 7,728 | 6,912 | 256 | **15,408** | **15,184** | **85,296** |
| continuation block | 0 | 512 | 7,728 | 7,040 | 256 | **15,536** | **15,312** | **86,058** |

The continuation row treats the previous chaining-value bits as already
Boolean-constrained and allocates a fresh 512-bit block. Its compression core,
excluding fresh block decomposition, is **15,024 constraints**.

## Streaming Fr cost and Blake2b ratio

A packed 64-byte block carries two 32-byte Fr encodings. With 254 Boolean
constraints per scalar, steady-state cost is:

```text
15,024 / 2 + 254 = 7,766 constraints per absorbed Fr
```

Per compression, standalone BLAKE3 is `15,792 / 52,416 = 0.3013×` Blake2b.
At equal 128-byte steady-state absorption, two BLAKE3 continuation cores plus
their fresh blocks are `2·15,536 / 52,416 = 0.5929×` Blake2b, or 1.69× fewer
constraints.

## Checks

- 64 seeded random inputs, lengths 1 through 64 bytes, match `blake3::hash`.
- Every random case passes `ConstraintMatrices::check_witness`.
- Standalone and fixed-IV constraint, variable, class, and nonzero counts are pinned.
- Blake2b's 52,416/52,032 standalone and 51,136/50,752 fixed-IV counts remain pinned.
- `cargo nextest run -p jolt-r1cs --cargo-quiet`: 78 passed.
- `cargo clippy -p jolt-r1cs --all-targets -q --message-format=short -- -D warnings` passed.
- `cargo fmt -q` passed.
