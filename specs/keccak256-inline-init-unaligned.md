# Spec: Keccak-256 inline INIT and unaligned-block variants, word-wise SDK padding

| Field       | Value                          |
|-------------|--------------------------------|
| Author(s)   | @0xAndoroid                    |
| Created     | 2026-09-14                     |
| Status      | proposed                       |
| PR          | [#1863](https://github.com/a16z/jolt/pull/1863) |

## Summary

Three new ops under the KECCAK256 funct7 (`jolt-inlines/keccak256`): an INIT absorb that skips reading the zero state, and unaligned-block twins of absorb and INIT that funnel-shift 18 containing words instead of copying the block. The SDK's `digest` dispatches on block position and pointer alignment and pads the final block word-wise. Precedent for INIT: `jolt-inlines/sha2` (`Sha256CompressionInitial`, funct3 0x01). Precedent for the PR shape: a16z/jolt#1862 (`Blake2b::compress`).

## Intent

### Goal

Make `Keccak256::digest` the cheapest way to hash on the guest for every input alignment and length, so downstream guests stop hand-emitting the inline. jeth (`crates/guest/src/keccak.rs`, ~300 lines) currently beats the SDK by 40 to 90 rows per hash with a word-wise shim that relies on containing-word loads, which are undefined behaviour in Rust's abstract machine. Inside a virtual instruction sequence the same loads are plain ISA loads on flat RAM, so the shim's tricks belong in the sequence builder, not in Rust.

### Invariants

- Existing ops 0x00 and 0x01 keep their sequences byte-for-byte (the golden trace for them must not change).
- `Keccak256::digest`, `update`, `finalize` return the same bytes as before for every input length and alignment.
- No `u8` store on any `digest` path; the only sub-word memory traffic is the platform `memcpy` of the tail bytes.
- Every sequence stays straight-line and inside the 80-register budget.

### Non-Goals

- A FINAL (in-sequence padding) variant; rejected below.
- INIT on the streaming `update` path.
- Any change to the permutation rounds or to other inline crates.

## Current state

```
lib.rs        INLINE_OPCODE = 0x0B, KECCAK256_FUNCT7 = 0x01
              KECCAK256_FUNCT3 = 0x00           Keccak256Permutation        rs1 = state (25 x u64)
              KECCAK256_ABSORB_PERMUTE_FUNCT3 = 0x01  Keccak256AbsorbPermutation  rs1 = state, rs2 = block (17 x u64, 8-aligned)

sequence_builder.rs   build(): load_state -> 24 rounds in 37 virtual registers -> store_state
                      load_state: 25 LD from rs1; absorb: 17 x { LD rs2+8i ; XOR }      (rows: 50 / 84 + rounds)
sdk.rs                digest(): state = [0; 25]; absorb_full_blocks; absorb_final; to_bytes
                      absorb_full_blocks: aligned input -> inline straight from the input;
                                          unaligned input -> copy_nonoverlapping into a stack block per block, then inline
                      absorb_final / finalize: byte copy + `bytes[len] = 0x01` + `|= 0x80` (sub-word RMWs, 4-9 rows each)
```

Rows per 136-byte block above the permutation rounds, estimated from instruction counts (verify with the ratchet test below):

| block kind | SDK today | jeth shim | after this change |
|---|---:|---:|---:|
| first block, aligned | 25 SD zero + 25 LD + 17 LD + 17 XOR = 84 | 92 | INIT: 17 LD + 8 zero regs + 25 SD = 50 |
| first block, misaligned | copy path, ~170 | ~144 | INIT-unaligned: 18 LD + 17x4 ALU + 8 + 25 SD = ~119 |
| middle, aligned | 34 | 34 | 34 |
| middle, misaligned | memcpy gather + 17 SD + 17 LD + 17 XOR, ~170 | ~136 | 25 LD + 18 LD + 17x5 ALU + 25 SD = ~153 |
| tail, 32-byte message | byte memcpy + 2 sub-word RMW + 17 SD + 17 LD + 17 XOR, ~130 | ~95 | word-wise padding + INIT, ~94 |

Rejected: a FINAL variant that pads inside the sequence. Sequences are straight-line, so it must load and mask all 17 lanes for every length (about +120 rows on a 32-byte message against the shim) and over-reads up to 135 bytes past the message. Padding stays in Rust, word-wise.

## Change 1: INIT variant (funct3 0x02)

`Keccak256InitAbsorbPermutation`: same operands as the absorb variant; the state at `rs1` is NOT read. Lanes 0..17 are the block words loaded from `rs2`, lanes 17..25 are zero registers, then 24 rounds, then the 25 lanes are stored to `rs1`.

- Contract: `rs1` 8-aligned, 200 writable bytes, prior contents ignored. `rs2` 8-aligned, 136 readable bytes. Regions disjoint.
- Builder: add an `init: bool` to `Keccak256SequenceBuilder::new`; in `load_state`, when `init`, replace the 25 `LD rs1` with `LD rs2` into `a[0..17]` (no XOR) and 8 zero-moves into `a[17..25]`. `store_state` unchanged.
- Constants: `KECCAK256_INIT_ABSORB_PERMUTE_FUNCT3 = 0x02`, `KECCAK256_INIT_ABSORB_PERMUTE_NAME`.
- Saves 42 rows per hash against the shim and 34 against the SDK; every hash has exactly one first block, so this is the bulk of the win.

## Change 2: unaligned-block variants (funct3 0x03 absorb, 0x04 init)

`Keccak256AbsorbPermutationUnaligned` and `Keccak256InitAbsorbPermutationUnaligned`: as 0x01 / 0x02 but `rs2` may have any alignment. Dispatch happens in the SDK (`rs2 & 7`), so aligned callers keep the 34-row absorb.

Sequence, before the lane loop:

```
base = rs2 & !7            ANDI base, rs2, -8
sh   = (rs2 & 7) << 3      ANDI t, rs2, 7 ; SLLI sh, t, 3          sh in {0, 8, ..., 56}
nsh  = 63 - sh             XORI nsh, sh, 63                         (63 - sh == sh ^ 63 for these values)
w0   = LD [base + 0]
```

Per lane `i` in 0..17:

```
w1     = LD [base + 8(i+1)]
t      = w0 >> sh                       SRL
u      = (w1 << 1) << nsh               SLLI 1 ; SLL nsh      (the `<< 1` keeps sh = 0 correct: shifting by 64 is not expressible)
lane_i = t | u                          OR
a[i]  ^= lane_i   (absorb)  /  a[i] = lane_i   (init)
w0     = w1                             register rotation, no row
```

18 LD + 17 x 4 ALU (+ 17 XOR for absorb). Register budget: 37 + 5 (`w0`, `w1`, `sh`, `nsh`, `t`/`u` reuse) = 42 of 80.

- Memory contract: the 18 aligned words `[base, base + 144)` are read. For a misaligned `rs2` (`rs2 & 7 != 0`) those are exactly the words containing the 136 block bytes: the last block byte is at `base + (rs2 & 7) + 135 >= base + 136`, so word 17 is a containing word and nothing past the block is read. For an aligned `rs2` the sequence is still correct but reads 8 bytes past the block; the SDK never issues that case. Document both facts in the op docs.
- Constants: `KECCAK256_ABSORB_PERMUTE_UNALIGNED_FUNCT3 = 0x03`, `KECCAK256_INIT_ABSORB_PERMUTE_UNALIGNED_FUNCT3 = 0x04`, names to match.
- Builder: an `unaligned: bool` alongside `init`; four `InlineOp` structs share one builder.

## Change 3: SDK (`sdk.rs`)

Raw entry points, `pub unsafe`, next to `keccak256_absorb_permute` and with the same three cfg arms (guest `.insn`, `host` model, panic elsewhere):

```
keccak256_init_absorb_permute(state: *mut u64, block: *const u8)
keccak256_absorb_permute_unaligned(state: *mut u64, block: *const u8)
keccak256_init_absorb_permute_unaligned(state: *mut u64, block: *const u8)
```

`Keccak256::digest(input)`:

1. `len < 136`: build the padded block (below) in an aligned `[u64; 17]`, then `init_absorb_permute(state, block)`. The state array is written, never zero-initialised by the caller: use `MaybeUninit<[u64; 25]>` or accept the 25 SD.
2. Else: first block through `init_absorb_permute` or its unaligned twin by `input.as_ptr() & 7`; every further full block through `absorb_permute` or its unaligned twin (the stride is 136, a multiple of 8, so the alignment is decided once and the per-block `copy_nonoverlapping` path in `absorb_full_blocks` is deleted); the tail through the padded block and the aligned absorb.

Padded final block, word-wise, no `u8` stores (`len < 136`, so `len / 8 <= 16` and `8 * (len % 8) <= 56`):

```
let mut block = [0u64; 17];
copy_nonoverlapping(tail.as_ptr(), block.as_mut_ptr().cast::<u8>(), len);   // platform memcpy; on jeth the word-wise override
block[len / 8] |= 0x01u64 << (8 * (len % 8));
block[16]      |= 1u64 << 63;
```

`Keccak256::finalize` (streaming path) pads the same way instead of `bytes[buffer_len] = 0x01; bytes[buffer_len + 1..].fill(0); bytes[135] |= 0x80`: clear the dead bytes of the partial word with `block[k] &= (1u64 << (8 * (len % 8))) - 1` where `k = len / 8` (a zero shift gives mask 0, i.e. the whole word), zero words `k + 1..17`, then the two ORs above. The streaming `update` may keep using the absorb variants; INIT on its first block is optional and out of scope.

## Host side

- `exec.rs`: reference models for the three new ops. INIT: `state = [0; 25]` then XOR the block and `execute_keccak_f`. Unaligned: read 18 aligned words from `block & !7` and apply the funnel shift above before XOR (or, equivalently for the model, read 136 bytes unaligned and convert little-endian); the direct-execution tests compare the emulated sequence with this model, so the model must define the exact memory the sequence touches.
- `host.rs`: add the three ops to `register_inlines! { ops: [...] }`; regenerate the inline trace fixture the macro's `trace_file` refers to, the way sha2 did when `Sha256CompressionInitial` was added.
- `sequence_builder.rs`: three new `InlineOp` impls with `type Advice = NoAdvice`.

## Tests

1. `spec.rs` + `tests.rs`: `InlineSpec` impls for the three new ops, run through `assert_edge_cases_match_reference` and `assert_random_cases_match_reference` like `test_keccak256_absorb_permute_direct_execution`. For the unaligned ops `Input = (state, block, offset)` with `offset` in `1..8`: `harness()` uses `InlineMemoryLayout::single_input(RATE_IN_BYTES + 8, NUM_LANES * 8)` so the 18th containing word lies inside the input region; `load()` writes the 136 block bytes at `input_base + offset` and adds `offset` to `harness.cpu.x[INLINE_RS2]` (`setup_registers` has already run when `load` is called). Edge cases cover offsets 1 and 7 for the zero and all-ones patterns; `random()` draws the offset. The reference is the aligned model: XOR the block words (as read from the byte buffer, little-endian) into the state, then `execute_keccak_f`.
2. `sdk.rs`: extend `test_keccak256_aligned_vs_unaligned` to all eight pointer offsets and every length `0..=600` plus `1024`, `8192`, comparing against the reference (`crate::exec::execute_keccak256` under `host`, or the `sha3` dev-dependency), not only aligned against unaligned.
3. Row ratchet: a `host` test that builds each of the five sequences with `INLINE::inline_sequence` and asserts its exact instruction count (`rounds + 50`, `+ 84`, `+ 50`, `+ 153`, `+ 119`, corrected to the measured values on the first run). A later change that adds rows must edit the test on purpose.
4. Existing tests stay green: `test_keccak256_trace_file_matches_generated`, the XKCP vectors, `test_execute_keccak256`.

## Acceptance

```
cargo fmt -q
cargo clippy --all --features host -q --all-targets -- -D warnings
cargo clippy --all --features host,zk -q --all-targets -- -D warnings
cargo nextest run -p jolt-inlines-keccak256 --features host --cargo-quiet
cargo nextest run -p jolt-inlines-sdk --features host --cargo-quiet
```

Report the trace-length delta of the `sha3-chain` profile before and after (`cargo run --release -p jolt-prover --features profiling -- profile --name sha3-chain --format none`; its guest calls `Keccak256::digest` on a 32-byte aligned input, i.e. the padded-INIT path, so the expected delta is about -36 rows per iteration), plus the ratchet numbers. Long and misaligned inputs are covered by the SDK test, not by a profile.

PR against `main`, one PR, Conventional Commits title (CI enforces it), for example `feat(inlines): keccak256 INIT and unaligned-block variants, word-wise SDK padding`. No changes outside `jolt-inlines/keccak256`, `jolt-inlines/sdk` (only if a test helper needs a misaligned-`rs2` variant) and the book's inline table (`book/src/how/optimizations/inlines.md`: three new rows under KECCAK256).

## Downstream, not part of this PR

jeth deletes `crates/guest/src/keccak.rs` and routes `native_keccak256` to `Keccak256::digest`; expected about -42 rows per hash on the first block (roughly -0.8% rows on a typical block), a tie on tails, a small gain on long misaligned inputs. Measured on the ten-block set after the pinned jolt worktree is bumped.
