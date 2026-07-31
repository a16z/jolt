# D64 deferred-carry commitment experiment

Date: 2026-07-31 EDT

## Question

Does the deferred fp128 commitment accumulator already used by the D128 root
kernel also improve the D64 root kernel used at `T = 2^26`, without increasing
peak memory or changing the protocol?

The D64 control accumulated each coefficient in `Fp128x8i32`, an eight-limb
wide representation. The candidate instead stores the low and high `u64`
limbs plus a signed `i16` count of `2^128` wraps, then applies
`2^128 = C (mod p)` once when the row tile is flushed. The current
`Prime128OffsetA7F7` preset has `C = 0xFFFFA7F7`.

## Fixed configuration

- Machine: Apple M4 Max
- Jolt candidate revision: `6536b7af906c37345c1e4c25f8fc1306aa251ab4`
- Jolt candidate tree: `161f53c03f9b8f9fce2eb5cc02316d6746da3c38`
- Akita revision: `a56b933cd097deebd3a1d937d1308e7d02ea6e0a`
- Rust: `rustc 1.95.0 (59807616e 2026-04-14)`
- Packed one-hot: physical `K = 256`, virtual logical-zero lane
- `PERF_LOG_K_CHUNK=8`
- `PERF_LOOKUPS_RA_VIRTUAL_LOG_K_CHUNK=32`
- Release profile, proof verification required

Akita command:

```bash
PERF_LOG_T=26 \
PERF_LOG_K_CHUNK=8 \
PERF_LOOKUPS_RA_VIRTUAL_LOG_K_CHUNK=32 \
PERF_TRACE=1 \
/usr/bin/time -l cargo nextest run --release \
  -p jolt-prover-legacy --features akita \
  -E 'test(sha2_chain_akita_perf)' \
  --run-ignored all --no-capture --cargo-quiet
```

Dory comparison command:

```bash
PERF_LOG_T=26 \
PERF_TRACE=1 \
/usr/bin/time -l cargo nextest run --release \
  -p jolt-prover-legacy --features host \
  -E 'test(sha2_chain_dory_perf)' \
  --run-ignored all --no-capture --cargo-quiet
```

The comparison metric is the primary prover span: `prove_packed` for Akita
and `prove_parts` for Dory. Setup, guest tracing, and verification are outside
those spans.

## Gates

1. The deferred accumulator must match canonical ring accumulation for D64
   through the full 8,192-row tile bound.
2. The adjacent `T = 2^22` root-accumulation screen must improve by more than
   benchmark noise.
3. Two `T = 2^26` candidate proofs must verify, remain stable, and show no
   peak-RSS regression.
4. Standard and ZK Dory end-to-end tests, Akita end-to-end tests, formatting,
   and required clippy configurations must pass.
5. The protocol, transcript, and committed polynomial must remain unchanged.

The Dory ratio is reported from two fresh current-tree runs. Older 111-second
Dory measurements are retained for history but are not used for the final
ratio.
