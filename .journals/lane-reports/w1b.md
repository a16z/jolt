# Lane W1B — stage 6b CPU members

## Checkpoint 1 — BytecodeReadRafCycle decomposition

### Retained host glue

- Validate relation dimensions, committed chunk count, stage-point widths, and output-opening count.
- Reclaim the session-shared packed `PcRow` scan; build only the small address-chunk eq tables and selector metadata.
- Derive the five scalar stage weights, the entry scalar, and the output-opening IDs.
- Assemble each round polynomial from device lane sums with the existing `round_poly_from_skipped_evals` recipe; retain transcript/bind order and output-claim extraction unchanged.
- Fail closed to the optimized CPU kernel before dispatch, or reclaim the live host-visible tables after a mid-sumcheck decline/failure.

### Device kernels

1. **Combined-table init:** tensor-split each stage cycle point on the host at `O(sqrt(T))`; one device map writes `C(j) = sum_s weight_s * E_hi_s[j_hi] * E_lo_s[j_lo]`, including the entry term at `j = 0`. This removes the five `O(T)` host eq-table builds and additions.
2. **Lazy rounds (width 1/2/4):** gather every `ra_i` pair directly from packed mapped-PC rows and the small branch tables; fuse any pending bind of `C`, evaluate `C(t) * product_i ra_i(t)` at `t in {0, 2, ..., degree}`, and reduce lane-major partials.
3. **Third-bind adoption:** fuse the pending `C` bind with width-8 RA materialization into one flat, device-owned factor ping-pong at length `T/8`.
4. **Dense rounds:** fuse the pending bind of every flat factor with the same product-grid lane reduction. Use 64-bit MSL word offsets for every `factor * len * FR_LIMBS` rebase; table-local indices remain 32-bit.

### Geometry and recovery invariants

- Synthetic `T = 2^27` offset parity test precedes the slot implementation; it must expose the fifth full-width factor start as word `2^32` without allocating the domain.
- Every mutating round writes only the next ping-pong buffer; the current state remains recoverable until command-buffer success.
- Device threshold controls prepare and rounds; forced-device parity asserts a positive dispatch count. CPU and Metal proof-byte matrices remain the retention gate.
- Reuse existing Metal buffer/runtime utilities only. No allocator, arena, madvise, parking-lifetime, or stage-driver ownership changes.

## Checkpoint 2 — pending

Parity and first `2^24` A/B.

## Final — pending

`2^25` cool ABBA, IncClaimReduction prepare result, commits, binary hash, and gate matrix.
