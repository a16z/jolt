# Lane K — BLAKE2b R1CS measurement gate

Date: 2026-09-02

## Gate result

**Fails the 262,144-constraint budget by >400×.** One standalone variable-state,
variable-block compression is **52,416 constraints, 52,032 witness variables, and
293,748 A/B/C nonzeros**. The optimized hidden transcript segment is
**106,148,024 constraints at fibonacci L=18** and **111,078,192 at L=20** under
the requested 264-constraint Fr encoding and 128-constraint challenge-conversion
budget model.

Implemented by `c220f85032626dc848f2fc3c4edb22a16b347377` in
`crates/jolt-r1cs/src/gadgets/blake2b.rs`.

## Compression shape

Representation: little-endian 64-bit words as Boolean R1CS variables. XOR is
`(2a)·b = a + b - out`; XOR with a constant allocates nothing. Rotations reorder
bits. Each addition allocates and Boolean-constrains all result/carry bits, then
adds one linear recomposition constraint. The 16 message words are allocated once
and reused through all 12 SIGMA schedules.

| Class | Derivation | Constraints | Witness vars | A/B/C nonzeros |
|---|---:|---:|---:|---:|
| chaining-state Booleanity | 8·64 | 512 | 512 | 1,536 |
| message-block decomposition | 16·64 | 1,024 | 1,024 | 3,072 |
| additions/decompositions | 192·(66+1) + 192·(65+1) | 25,536 | 25,152 | 162,180 |
| round G XORs | (384−4)·64 | 24,320 | 24,320 | 121,840 |
| final `h xor v xor v[8..]` | 8·2·64 | 1,024 | 1,024 | 5,120 |
| **standalone total** | | **52,416** | **52,032** | **293,748** |

Addition split: 192 ternary additions use 66 Boolean bits plus one linear row
(12,864 constraints, 12,672 variables); 192 binary additions use 65 Boolean bits
plus one linear row (12,672 constraints, 12,480 variables). The first four
`d xor a` words have constant `d`, saving 256 constraints. Shape assertions are
at `crates/jolt-r1cs/src/gadgets/blake2b.rs:482-531`.

Each Blake2b digest starts with the fixed parameterized IV. For its first block,
the fixed state saves four more G-word XORs and half of the final XORs:

| Fixed-IV first block class | Constraints | Witness vars | A/B/C nonzeros |
|---|---:|---:|---:|
| message-block decomposition | 1,024 | 1,024 | 3,072 |
| additions/decompositions | 25,536 | 25,152 | 161,676 |
| round G XORs | 24,064 | 24,064 | 120,826 |
| final output | 512 | 512 | 3,112 |
| **total** | **51,136** | **50,752** | **288,686** |

A continuation block reuses the preceding output state: **51,904 constraints,
51,520 new variables, 292,212 nonzeros** including a fresh 1,024-bit block.
The compression-only fixed-IV cost, when transcript state/scalar bits are reused
instead of copied into fresh block variables, is **50,112 constraints and 49,728
variables**.

## Blake2b-256 block and padding costs

`blake2::Blake2b<U32>` uses 128-byte blocks. The final partial block is zero-filled
and compressed with `f = true`; an exact 128-byte multiple uses that full block as
the final block. There is no `0x80` delimiter. The byte counter `t` is the number
of unpadded bytes consumed.

`DigestTranscript` hashes `state[32] || (0^28 || round_BE_u32)[32] || payload`.
Thus a raw payload of `p` bytes needs `ceil((64+p)/128)` compressions. Costs below
include a fresh 1,024-bit block allocation per compression:

| Raw payload | Total hash bytes | Compressions | Constraints | Witness vars | Nonzeros |
|---|---:|---:|---:|---:|---:|
| challenge, 0 B | 64 | 1 | 51,136 | 50,752 | 288,686 |
| one Fr, 32 B | 96 | 1 | 51,136 | 50,752 | 288,686 |
| four Fr, 128 B | 192 | 2 | 103,040 | 102,272 | 580,898 |
| eight Fr, 256 B | 320 | 3 | 154,944 | 153,792 | 873,110 |
| eleven Fr, 352 B | 416 | 4 | 206,848 | 205,312 | 1,165,322 |

These multi-Fr rows are the cost of one hypothetical raw payload. Production
`append_scalars(n)` does **not** concatenate the scalars: it performs one label
hash and then `n` separate 32-byte scalar hashes.

## Legacy transcript calls

`LegacyBlake2bTranscript = DigestTranscript<Blake2b<U32>, Fr>`
(`crates/jolt-transcript/src/lib.rs:71-75`). `hasher()` prepends the 32-byte state
and 32-byte round word (`digest.rs:91-96`); every raw append updates state and
increments the round (`digest.rs:115-124,173-176`). Fr is 32-byte big-endian
(`legacy.rs:120-129`). Labels are one right-padded 32-byte word; counted labels
are 24 label bytes plus an 8-byte BE count (`legacy.rs:148-180`).

| Public call | Ordered raw hashes | Blake2b compressions |
|---|---|---:|
| `new(label)` | padded label[32] | 1 |
| `append_scalar(label, x)` | label[32], Fr_BE[32] | 2 |
| `append_scalars(label, n)` | label/count[32], then n × Fr_BE[32] | n+1 |
| `append_bytes(label, len)` | label/byte-length[32], then raw bytes[len] | `1 + ceil((64+len)/128)` |
| `append_u64(label, x)` | label[32], then 24 zero bytes + x_BE[8] | 2 |
| `challenge()` / `challenge_scalar()` | no payload: state[32] + round[32] | 1 |
| `challenge_vector(n)` | n independent challenges | n |
| `challenge_scalar_powers(n)` | one scalar challenge, then native powers | 1, including n=0 |

The legacy method definitions are at
`crates/jolt-prover-legacy/src/transcripts/transcript.rs:50-74,116-123,153-164`.
The modular verifier preserves the same sequence through `Label`,
`LabelWithCount`, and per-value `append_to_transcript` calls.

## Hidden Jolt segment

Compressed and full sumcheck rounds absorb one counted label, then every stored
coefficient separately; each round then squeezes once
(`crates/jolt-sumcheck/src/verifier.rs:70-81,110-129` and
`round_proof.rs:75-86`). Batch input claims use labeled scalar appends
(`crates/jolt-sumcheck/src/lib.rs:113-120`); output claims do the same
(`recorder.rs:132-148`). Aliases are skipped by the generated stage drivers
(`crates/jolt-verifier/src/stages/relations.rs:146-160`). Stage 8 absorbs one
counted RLC label plus 41 scalars, then draws one scalar
(`crates/jolt-verifier/src/stages/stage8/verify.rs:207-214`).

| Count | fibonacci L=18, K=13 | L=20, K=16 |
|---|---:|---:|
| sumcheck rounds / round labels | 275 | 290 |
| round coefficients | 876 | 939 |
| batch input claims | 23 | 23 |
| opening claims carried / absorbed | 266 / 259 | 266 / 259 |
| stage-8 RLC claims | 41 | 41 |
| hidden challenges (stages 1–7 + RLC) | 351+1 = 352 | 371+1 = 372 |
| stage-4 empty separator raw hashes | 2 | 2 |
| **compressions** | **2,111** | **2,209** |

Compression formula:

```text
H = round_coefficients + round_labels
  + 2*batch_input_claims + 2*absorbed_opening_claims
  + (1 + 41)                   # RLC label + values
  + 2                          # stage-4 separator + empty body
  + hidden_challenges
```

Every term is one compression: every hidden raw payload is 0 or 32 bytes, hence
the complete Blake2b input is 64 or 96 bytes. The 26 L=20 Dory challenges are
outside this hidden segment; total proof challenges are 398. At L=18 the Dory
path has 24 challenges; total proof challenges are 376.

For circuit composition, transcript state bits come directly from the preceding
digest output and fixed labels/counters/padding are constants. Let `S` be the
number of absorbed Fr values and `Q` the hidden challenge count:

```text
S18 = 876 + 23 + 259 + 41 = 1,199
S20 = 939 + 23 + 259 + 41 = 1,262

C_hidden = 50,112*H + 264*S + 128*Q
V_hidden = 49,728*H + 264*S + 128*Q
```

The 264 term is the requested planning model: 254 Boolean bits plus about 10
constraints/auxiliaries for recomposition and the BN254 canonical bound. The 128
term is a conservative separate challenge-extraction budget. The gadget already
returns Boolean digest bits, so an integrated implementation can form both
16-byte challenge decodings as a linear combination with zero nonlinear
constraints (one linear row only if a dedicated scalar witness is required).

| Hidden total | fibonacci L=18 | L=20 |
|---|---:|---:|
| compression constraints | 105,786,432 | 110,697,408 |
| Fr encoding constraints | 316,536 | 333,168 |
| conservative challenge conversion | 45,056 | 47,616 |
| **constraints** | **106,148,024 (404.9× budget)** | **111,078,192 (423.7× budget)** |
| **witness variables** | **105,337,400** | **110,229,936** |

## Correctness and gates

- RFC 7693 BLAKE2b-512 `abc` vector passes.
- BLAKE2b-256 matches the `blake2` crate on 50 seeded random inputs: 25 single
  block and 25 inputs of 129–384 bytes, plus empty input.
- Every case validates the assignment with `ConstraintMatrices::check_witness`.
- `cargo nextest run -p jolt-r1cs --cargo-quiet`: 75 passed.
- `cargo clippy -p jolt-r1cs --all-targets -q --message-format=short -- -D warnings`
  and full host-workspace clippy passed; `cargo fmt -q` passed.

Native 1,000-compression witness timing was skipped: the budget already fails by
>400×, so the timing cannot change the gate.
