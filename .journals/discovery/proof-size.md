# Modular Jolt proof size

Measured 2026-09-02 at `756bddce3` on a 10-core, 16 GiB Apple machine.

## Measurement contract

- Workload: real `fibonacci-guest`, modular prover, optimized backend, exact padded traces at 2^18, 2^20, 2^22, and 2^24 rows.
- Clear build: `profiling`, without `zk` or `akita`. ZK sample: `profiling,zk` at 2^18.
- Wire encoding: the profile runner's production `bincode::serde::encode_to_vec(&proof, bincode::config::standard())` path.
- Temporary counters serialized every top-level field and sub-structure independently. Assertions pinned each component sum to its enclosing serialized size and the five top-level categories to the proof total. The counters were removed after measurement.
- Prover time is the runner's `Instant` around `dory::prove`; verifier time is an added `Instant` around the existing verification gate. Process wall time and maximum RSS are `/usr/bin/time -l` values from the same invocation. Guest compile/trace and preprocessing are outside prover time but inside process wall/RSS.

## Clear-mode results

All byte columns are measured bincode bytes and sum exactly to `total`.

| log2 rows | total | trace commitments | sumchecks | clear opening claims | Dory opening | other | prover s | verifier s | process wall s | peak RSS |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 18 | 82,191 | 15,870 | 28,337 | 8,536 | 29,429 | 19 | 2.72 | 0.0678 | 6.21 | 432.84 MiB |
| 20 | 86,523 | 15,870 | 30,077 | 8,536 | 32,021 | 19 | 7.12 | 0.0797 | 9.68 | 1.25 GiB |
| 22 | 90,855 | 15,870 | 31,817 | 8,536 | 34,613 | 19 | 20.26 | 0.0849 | 24.98 | 3.31 GiB |
| 24 | 95,187 | 15,870 | 33,557 | 8,536 | 37,205 | 19 | 79.04 | 0.0960 | 88.46 | 8.85 GiB |

The 2^24 run was permitted after 2^22 RSS extrapolated to about 8.7 GiB; measured maximum RSS was 9,507,831,808 bytes with zero swaps.

## Element counts

### Trace polynomial commitments

Every scale carries 41 GT elements: 1 `rd_inc`, 1 `ram_inc`, 32 `instruction_ra`, 4 `ram_ra`, and 3 `bytecode_ra`. BN254 GT is 384 raw bytes; a standalone serde/bincode commitment is 387 bytes because it is encoded as a byte sequence.

| field | GT count | bytes |
|---|---:|---:|
| `rd_inc` | 1 | 387 |
| `ram_inc` | 1 | 387 |
| `instruction_ra` | 32 | 12,385 |
| `ram_ra` | 4 | 1,549 |
| `bytecode_ra` | 3 | 1,162 |
| **total** | **41** | **15,870** |

The vector rows include their bincode length words.

### Per-stage sumchecks

Cell format: `bytes / rounds / stored Fr`. Compressed rounds omit the linear coefficient; a stored count of 3 means degree 3, not four transmitted coefficients.

| proof | 2^18 | 2^20 | 2^22 | 2^24 |
|---|---:|---:|---:|---:|
| stage 1 uni-skip, full | 900 / 1 / 28 | 900 / 1 / 28 | 900 / 1 / 28 | 900 / 1 / 28 |
| stage 1, compressed | 1,846 / 19 / 57 | 2,040 / 21 / 63 | 2,234 / 23 / 69 | 2,428 / 25 / 75 |
| stage 2 uni-skip, full | 228 / 1 / 7 | 228 / 1 / 7 | 228 / 1 / 7 | 228 / 1 / 7 |
| stage 2, compressed | 3,010 / 31 / 93 | 3,204 / 33 / 99 | 3,398 / 35 / 105 | 3,592 / 37 / 111 |
| stage 3, compressed | 1,749 / 18 / 54 | 1,943 / 20 / 60 | 2,137 / 22 / 66 | 2,331 / 24 / 72 |
| stage 4, compressed | 2,428 / 25 / 75 | 2,622 / 27 / 81 | 2,816 / 29 / 87 | 3,010 / 31 / 93 |
| stage 5, compressed | 14,101 / 146 / 436 | 14,743 / 148 / 456 | 15,385 / 150 / 476 | 16,027 / 152 / 496 |
| stage 6a, compressed | 911 / 12 / 28 | 911 / 12 / 28 | 911 / 12 / 28 | 911 / 12 / 28 |
| stage 6b, compressed | 2,901 / 18 / 90 | 3,223 / 20 / 100 | 3,545 / 22 / 110 | 3,867 / 24 / 120 |
| stage 7, compressed | 263 / 4 / 8 | 263 / 4 / 8 | 263 / 4 / 8 | 263 / 4 / 8 |
| **total** | **28,337 / 275 / 876** | **30,077 / 287 / 930** | **31,817 / 299 / 984** | **33,557 / 311 / 1,038** |

Degree layout behind the counts:

- Stage 1 uni-skip stores 28 full coefficients; stage 2 uni-skip stores 7.
- Stages 1-4 store 3 coefficients per compressed round.
- Stage 5 stores 2 coefficients in 128 rounds, then 10 in `log2(trace)` rounds.
- Stage 6a stores 2 coefficients in 8 rounds and 3 in 4 rounds; stage 6b stores 5 per round; stage 7 stores 2 in 4 rounds.

### Clear opening claims

The `JoltProofClaims::Clear` payload is scale-invariant: 266 Fr elements, 8,535 payload bytes plus the one-byte enum tag.

| stage | Fr count | bytes |
|---|---:|---:|
| 1 | 36 | 1,152 |
| 2 | 19 | 608 |
| 3 | 16 | 512 |
| 4 | 7 | 227 |
| 5 | 66 | 2,114 |
| 6a | 2 | 65 |
| 6b | 81 | 2,602 |
| 7 | 39 | 1,255 |
| enum tag | 0 | 1 |
| **total** | **266** | **8,536** |

Fr uses a fixed `[u8; 32]` serde encoding. Bytes above `32 × count` are vector and option framing.

### Dory batched opening

For `r` Dory rounds, the clear proof contains `2 + 6r` GT, `2 + 3r` G1, and `1 + 3r` G2 elements. Raw compressed widths are GT 384, G1 32, G2 64 bytes. The measured byte count also includes Dory's round count, dimensions, option markers, and the outer bincode byte-sequence framing.

| log2 rows | Dory rounds (`nu=sigma`) | GT | G1 | G2 | bytes |
|---:|---:|---:|---:|---:|---:|
| 18 | 11 | 68 | 35 | 34 | 29,429 |
| 20 | 12 | 74 | 38 | 37 | 32,021 |
| 22 | 13 | 80 | 41 | 40 | 34,613 |
| 24 | 14 | 86 | 44 | 43 | 37,205 |

Each extra Dory round adds exactly 2,592 bytes of group data: 6 GT + 3 G1 + 3 G2.

### Other proof fields

| field | bytes |
|---|---:|
| protocol config | 3 |
| absent untrusted-advice commitment | 1 |
| trace length | 5 |
| `ram_K` | 3 |
| read/write config | 4 |
| one-hot config | 2 |
| trace polynomial order | 1 |
| **total** | **19** |

Public program I/O is a verifier argument, not a `JoltProof` field: zero proof bytes.

## ZK sample at 2^18

The detailed run produced an 80,375-byte proof: 15,870 trace commitments + 9,908 committed stage proofs + 22,781 claims envelope + 31,797 ZK Dory opening + 19 other bytes. Prover 7.08 s, verifier 0.198 s, process wall 9.61 s, maximum RSS 527.39 MiB. An immediately preceding identical-size run measured 3.93 s prover and 0.133 s verifier; timing was host-load-sensitive, while every byte/count result matched.

### Committed stage proofs

| proof | rounds | round G1 | output-claim G1 | bytes |
|---|---:|---:|---:|---:|
| stage 1 uni-skip | 1 | 1 | 1 | 70 |
| stage 1 | 19 | 19 | 2 | 715 |
| stage 2 uni-skip | 1 | 1 | 1 | 70 |
| stage 2 | 31 | 31 | 1 | 1,090 |
| stage 3 | 18 | 18 | 1 | 648 |
| stage 4 | 25 | 25 | 1 | 886 |
| stage 5 | 146 | 146 | 3 | 5,066 |
| stage 6a | 12 | 12 | 1 | 444 |
| stage 6b | 18 | 18 | 3 | 714 |
| stage 7 | 4 | 4 | 2 | 205 |
| **total** | **275** | **275** | **16** | **9,908** |

### BlindFold claims payload

`JoltProofClaims::Zk` is a 22,780-byte BlindFold proof plus a one-byte enum tag.

| BlindFold part | exact contents | bytes |
|---|---|---:|
| row commitments | 496 G1 | 16,377 |
| outer sumcheck | 11 rounds, 33 stored Fr | 1,068 |
| inner sumcheck | 14 rounds, 28 stored Fr | 911 |
| witness + error openings | 33 + 33 Fr | 2,114 |
| folded output + blinding openings | 33 + 33 Fr | 2,116 |
| folded output + blinding scalars | 2 Fr | 66 |
| standalone scalars | 4 Fr | 128 |
| **BlindFold total** | **496 G1 + 199 Fr** | **22,780** |

The 496 G1 row commitments split as: 38 auxiliary, 275 random-round, 16 random-output-claim, 38 random-auxiliary, 64 random-error, 1 random-evaluation, and 64 cross-term-error commitments.

The ZK Dory opening has 11 rounds, 73 GT, 37 G1, 36 G2, and 8 Fr elements; its measured size is 31,797 bytes. At this scale, ZK is 1,816 bytes smaller than clear: committed stage proofs save 18,429 bytes, while BlindFold/ZK-Dory add 16,613 bytes net against clear claims/Dory.

## Commands

```bash
export CARGO_TARGET_DIR=/Volumes/Dev/cargo-target/wrap-spartan-hyperkzg
cargo build -p jolt-prover --release --features profiling -q --message-format=short

export RAYON_NUM_THREADS=10
/usr/bin/time -l /Volumes/Dev/cargo-target/wrap-spartan-hyperkzg/release/jolt-prover profile --name fibonacci --scale 18 --format none --backend optimized
/usr/bin/time -l /Volumes/Dev/cargo-target/wrap-spartan-hyperkzg/release/jolt-prover profile --name fibonacci --scale 20 --format none --backend optimized
/usr/bin/time -l /Volumes/Dev/cargo-target/wrap-spartan-hyperkzg/release/jolt-prover profile --name fibonacci --scale 22 --format none --backend optimized
/usr/bin/time -l /Volumes/Dev/cargo-target/wrap-spartan-hyperkzg/release/jolt-prover profile --name fibonacci --scale 24 --format none --backend optimized

export CARGO_TARGET_DIR=/Volumes/Dev/cargo-target/wrap-spartan-hyperkzg
cargo build -p jolt-prover --release --features profiling,zk -q --message-format=short

export RAYON_NUM_THREADS=10
/usr/bin/time -l /Volumes/Dev/cargo-target/wrap-spartan-hyperkzg/release/jolt-prover profile --name fibonacci --scale 18 --format none --backend optimized
```

## Findings

- Clear proofs grow only 4,332 bytes per fourfold trace increase: 2,592 Dory bytes + 1,740 sumcheck bytes.
- At 2^18, Dory is 35.8% and sumchecks are 34.5% of the proof; fixed trace commitments are 19.3%.
- Stage 5 is the largest sumcheck payload: 49.8% at 2^18 and 47.8% at 2^24.
- The proof size is logarithmic in trace length across this range; prover memory is not. The 2^24 run used 8.85 GiB RSS without swapping.
