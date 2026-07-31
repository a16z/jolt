# Packed sparse RAM-address experiment

Date: 2026-07-31 EDT

## Question

Can the two stage-local RAM address streams use an out-of-domain sentinel
instead of `Option<usize>` without changing the proved polynomial or slowing
the large-trace prover?

On the target, `Option<usize>` occupies 16 bytes and `usize` occupies 8. The
candidate replaces a missing address with `K`, where every valid remapped
address is in `[0, K)`.

For a trace row `j`, the old and new logical values are:

```text
old(j) = Some(k)  -> eq(r_address, k)
         None     -> 0

new(j) = s_j != K -> eq(r_address, s_j)
         s_j == K -> 0
```

Thus the RA polynomial and the RAM claim-reduction arrays are unchanged. The
candidate only changes the prover's private representation.

## Fixed evaluator

- Machine: Apple M4 Max
- Control revision: `5292f7f939d6ee63aea1490b1ac23e8746c48d72`
- Candidate revision: `e78c5d7386c9caae5625e7b77b675f8c0562cda5`
- Candidate tree: `2ba6e55573e659b63465be86c6479b2fc11a2514`
- Rust: `rustc 1.95.0 (59807616e 2026-04-14)`
- Physical one-hot `K = 256`
- `PERF_LOG_K_CHUNK=8`
- `PERF_LOOKUPS_RA_VIRTUAL_LOG_K_CHUNK=32`
- Release profile, successful proof verification required

Benchmark command:

```bash
/usr/bin/time -l env \
  PERF_LOG_T=26 \
  PERF_LOG_K_CHUNK=8 \
  PERF_LOOKUPS_RA_VIRTUAL_LOG_K_CHUNK=32 \
  PERF_TRACE=1 \
  cargo nextest run --release \
    -p jolt-prover-legacy --features akita \
    -E 'test(sha2_chain_akita_perf)' \
    --run-ignored all --no-capture --cargo-quiet
```

The same command with `PERF_LOG_T=22` is the discovery screen and with
`PERF_LOG_T=28` is the exact-target check.

## Metrics and gates

The affected phase metrics are the inclusive `prove_stage4` and
`prove_stage5` spans. Their internal RAM-value and RAM-RA spans are retained
for diagnosis. Whole-prover and commitment times are reported separately.

The analytical reduction is 8 bytes per trace row in each of two disjoint
phase-local allocations:

```text
Stage 4 RAM write addresses: 16T -> 8T
Stage 5 RAM RA addresses:    16T -> 8T
```

The reductions do not add: the two vectors are not live at the same time.

Promotion requires:

1. sentinel and `Option` RA polynomials agree through every bind in both
   binding orders;
2. the exact reduction is 8 bytes per cycle in each affected stage with no
   replacement trace-scaled owner;
3. repeated `T = 2^26` runs show no Stage 4/5 regression and expose the
   expected roughly 512 MiB process-peak reduction if that window sets the
   high-water mark;
4. exact `T = 2^28` runs verify with zero swaps and no mean Stage 4/5
   regression;
5. natural, forced-K256, and committed-program Akita proofs pass, as do
   standard and ZK Dory regression tests, Clippy, and formatting.

The `T = 2^22` run is directional rather than a promotion gate: each removed
owner is only 32 MiB there, below the scale at which this experiment is
intended to change cache and resident-memory behavior.
