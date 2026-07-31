# Akita Fp128 deferred-accumulator loop

Date: 2026-07-31 EDT

## Question

Can Fp128's pseudo-Mersenne identity remove repeated canonical reductions from
measured Akita hot loops without increasing persistent memory or moving work to
another prover phase?

The two independent mechanisms are:

1. accumulate several full Fp128 products before reduction in PIOP dot-product
   shapes;
2. accumulate D128 commitment additions and subtractions modulo `2^128`, track
   the signed wrap count, and apply the Solinas correction once per tile.

They are evaluated independently. A failed product-sum experiment must not
block the commitment experiment, and neither candidate may rely on the other.

## Fixed evaluators

The product-sum candidate first runs exact two-, three-, four-, 16-, 64-, and
256-term operation shapes against eager canonical arithmetic. It advances only
if the small FMA shapes used by the prover improve and a traced `T = 2^22`
proof moves the named target span.

The commitment candidate first runs the exact D128/K256 inner-loop shape:

```text
8,192 rows * 29 columns * 128 coefficients
```

It advances only if it is at least 10% faster than canonical accumulation,
matches canonical output, and has a proved counter bound. The production
candidate then faces an adjacent complete D128 commitment comparison and the
full ignored Akita performance harness:

```text
PERF_LOG_T=28
PERF_LOG_K_CHUNK=8
PERF_LOOKUPS_RA_VIRTUAL_LOG_K_CHUNK=32
PERF_TRACE=1
cargo nextest run --release -p jolt-prover-legacy --features akita \
  -E 'test(sha2_chain_akita_perf)' --run-ignored all --no-capture --cargo-quiet
```

## Promotion guards

- Every candidate result equals canonical field arithmetic.
- The `T = 2^28` proof verifies.
- The commitment candidate improves
  `trace_onehot_commit_accumulate` by at least 5%.
- Peak RSS does not increase beyond run noise and the process reports zero
  swaps.
- No persistent witness or setup allocation is added.
- D64 behavior remains unchanged.
- Rejected mechanisms are retained only as measurements, not pinned into
  Jolt.
- Accepted code and benchmark documentation are separate commits.

## Correctness obligations for carry deferral

For `p = 2^128 - C`, represent the exact integer accumulator as

```text
S = low + q * 2^128
```

where `low` is a wrapping 128-bit value and `q` is signed. A 128-bit addition
increments `q` on carry; a subtraction decrements it on borrow. At the tile
boundary,

```text
S = low + q * 2^128 = low + q * C (mod p).
```

The production tile contains at most 8,192 contributions to any destination
coefficient, so `|q| <= 8,192` and an `i16` counter cannot overflow. The helper
must be private to the tiled D128 path, cleared after every rank flush, and
tested at the full bound with both positive and negative wraps.

## Budget

- one product-sum representation;
- one compact carry-deferred representation plus one production integration;
- one adjacent complete-commit pair;
- one exact `T = 2^28` prover/RSS run for a promoted candidate;
- stop either direction if its focused premise fails.
