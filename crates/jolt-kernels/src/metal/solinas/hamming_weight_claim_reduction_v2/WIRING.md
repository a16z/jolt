# Hamming-weight v2 promotion control

## Decision

Do not tune the accepted-row shader to recover the latest 1.34% miss. The
fresh log-26 diagnostic explicitly selected `accepted-rows` and measured
549.294665 ms optimized CPU, 111.356377 ms complete Metal, and 86.162042 ms
GPU-active: 4.932763x. Its complete 5x cap is 109.858933 ms; a useful 5.3x
margin would need 103.640502 ms.

The existing `packed-hot` producer plus `retained-hot` Hamming consumer is the
decisive control. It already has exact proof and resource evidence:

- five alternating log-26 pairs: 8.4015x paired median, 7.8633x minimum;
- exact one-pair log-27 diagnostic: 7.2183x;
- log-26 Booleanity+Hamming family: 7.3855x paired median;
- one producer command and no private stage-7 projection dispatch;
- one Hamming command, one completion, one 118,784-byte readback, and host
  Fiat--Shamir.

The retained pair was left non-default because the one-pair log-27
**Booleanity-address** result was 4.9697x. That is a producer/family promotion
gate, not evidence that Hamming needs a new shader. Optimizing the five-scan
accepted fallback would spend an iteration on a path which should not carry
target-scale production traffic.

## Fixed boundary

The stage-6a producer writes 29 plane-major hot-byte planes and one validity
plane while it already scans each 40-byte resident row. Stage 7 borrows only
the 29 hot planes. Selector order is:

```text
0..7    lookup high bytes, most-significant first
8..15   lookup low bytes, most-significant first
16..17  mapped PC bytes (8, 0); absent is encoded as zero
18..19  remapped RAM bytes (8, 0); absent is encoded as zero
20..27  centered fused-increment bytes (0 through 56)
28      fused-increment carry (-1, 0, 1 -> 255, 0, 1)
```

Stage 7 removes bucket zero for every selector. This is why absent optional
columns and present-at-address-zero may share a retained byte. No challenge,
equality weight, gamma value, or transcript state enters the projection.

`HammingHotLeaseReceipt` adds the checks missing from a bare buffer handle:
source and hot allocation identities, device, proof generation, row and byte
lengths, selector-order version, producer completion, complete overwrite,
zero row upload, and zero private projection dispatches.
`HammingWeightExecutionReceipt` checks the existing `[6,6,6,6,5]` consumer:
one command, ten encoders/dispatches, one completion, one readback, matching
lease identities, and a valid GPU timestamp.

## Log-26 traffic

| quantity | accepted rows | retained hot |
| --- | ---: | ---: |
| histogram source reads | 13,421,772,800 | 1,946,157,056 |
| split-EQ cache-unique bytes | 557,056 | 557,056 |
| partial write + read | 486,539,264 | 486,539,264 |
| output write + read | 237,568 | 237,568 |
| cache-optimistic total | 13,909,106,688 | 2,433,490,944 |
| fully issued total | 19,277,422,592 | 7,801,806,848 |

The consumer traffic reduction is 5.716x. Charging the complete 30-byte
projection write once gives 4,446,756,864 producer-plus-consumer bytes, still
3.128x below the accepted consumer alone. The retained cache-optimistic copy
floor is 5.387385 ms and its 80%-of-copy cap is 6.734232 ms on the retained M4
Max control. These are traffic controls, not a latency forecast.

The frozen census contains 1,946,157,056 selector-row opportunities and
1,588,505,707 retained nonzero additions. The structurally matching atomic
schedule's robust 35.451625-ms standalone active control sustains 44.808
Gadd/s; this is a directional service control, not a retained-kernel timing.
Holding the latest complete member's 25.194335-ms non-GPU remainder, a 5.3x
member needs only 20.250 Gadd/s. Even charging that service control produces a
directional 60.645960-ms complete member, comfortably below the 103.640502-ms
bar. The sealed complete-member campaign remains authoritative; this
operation roof only establishes that another accepted-row shader iteration
is not the next decisive experiment.

## Root wiring and validation

1. Declare `hamming_weight_claim_reduction_v2` in `solinas/mod.rs` for the
   CPU-only model/receipt tests. No source fragment or shader pipeline is
   added.
2. Map the checked receipt fields onto `HammingHotRows`. A session may park the
   lease only after `BooleanityAddressSuccessorInvocation::execute_timed`
   completes. Consume it terminally in the retained Hamming adapter.
3. Run the next fixed evaluator with
   `--booleanity-address-metal-implementation packed-hot` and
   `--hamming-weight-metal-implementation retained-hot`. Keep the optimized
   CPU denominator and host Fiat--Shamir boundary unchanged.
4. First run one exact log-26 control. If its complete Hamming member clears
   5.3x and all receipt/parity guards pass, run five alternating pairs. The
   campaign controller requires every pair above 5x and the total and both
   order-stratum medians above 5.3x.
5. Run the sealed log-27 campaign. Treat any remaining Booleanity-address miss
   as that kernel's next experiment; do not route it into Hamming shader work.

Root validation commands:

```bash
cargo fmt -q
cargo clippy -p jolt-kernels --features metal,test-utils --lib -- -D warnings
cargo clippy -p jolt-kernels --features metal,test-utils --tests -- -D warnings
cargo nextest run -p jolt-kernels hamming_weight_claim_reduction_v2 \
  --features metal,test-utils --cargo-quiet
```
