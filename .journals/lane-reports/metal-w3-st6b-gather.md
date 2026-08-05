# W3 st6b gather residual — NO-GO

**Verdict:** no kernel candidate clears the `>=12%` dominant-slice and
`>=0.4 s` stage bars. Production shaders are unchanged. The isolated harness
now attributes lazy widths, adoption, dense-device rounds, CPU handoff tail,
and launched-round waits; its A/B order is balanced `AB/BA`.

## Residual anatomy

The certified stage is **7.02 s**. The latest production command-buffer audit
prices Bool lazy+dense at **0.969 s** and the two RAV drivers at **1.701 s**:
**2.670 s / 38%** of the stage is in this family.

Deferred instruction-RAV, production geometry, quiet `2^24` isolated pass:

| slice | wall | share |
|---|---:|---:|
| width 1 | 75.4 ms | 42% of lazy |
| width 2 | 43.3 ms | 24% of lazy |
| width 4 | 31.4 ms | 18% of lazy |
| width 8 | 27.2 ms | 15% of lazy |
| **lazy total** | **178.5-184.6 ms** | **69% of pipeline** |
| fused width-16 adoption | 40.9-41.1 ms | 16% |
| dense device rounds | 15.1-15.4 ms | 6% |
| host handoff tail | 20.4-22.9 ms | 8% |
| **pipeline** | **258.0-265.0 ms** | 100% |

Deferred Bool agrees: **231.53-231.60 ms** total; lazy
**152.21-153.41 ms**, adoption **44.86 ms**, dense device **16.36 ms**, host
tail **16.88 ms**. Widths 1/2/4 are 60.8/36.5/27.7 ms — **81% of lazy**.

Launched-round `collect` brackets total **232.4-234.7 ms**. This is almost
entirely useful queued GPU execution, not removable host idle: the batch
already launches each driver's detached command buffer before collection,
and the next round needs the transcript challenge derived from every member's
message. Eight empty-CB round trips at the measured 133.8 us ceiling price
only ~1.1 ms here. The shrinking host tail begins at the fixed device gate
and stays ~20 ms as `log_t` grows; even three-driver removal is below 0.1 s.

## Doors tested

### Packed-row reuse — reject

Widths `<=4` cached each pair's lookup-index words once per thread, replacing
the per-polynomial row reloads while preserving gather/add order. Two quiet
`2^22` samples overlap:

| kernel | existing lazy | row-batched lazy |
|---|---:|---:|
| instruction RAV | 49.96-50.37 ms | 49.28-50.59 ms |
| Bool | 44.71-60.14 ms | 44.91-46.34 ms |

The branch table's random field loads and relation arithmetic dominate; the
row words already ride cache. Prototype removed.

### SIMD-group reduction — reject

Replaced each lane's 10-barrier 256-thread tree with SIMD shuffles plus two
threadgroup barriers. Byte parity held, but the `2^24` RAV decision samples
overlap:

| metric | tree | SIMD |
|---|---:|---:|
| lazy rounds | 178.52-184.55 ms | 180.70-182.55 ms |
| full pipeline | 258.03-265.00 ms | 257.06-262.78 ms |

Prototype removed.

### Priced without retention

- Current tables are already offset-major with `k=256`: each offset page is
  8 KiB per polynomial. SoA/interleaving cannot coalesce adjacent threads'
  independent random indices; threadgroup page staging adds a barrier and
  reloads the same cache-resident page per threadgroup.
- `e_out * e_in` is computed once per row, outside every polynomial loop.
  It is cycle-position-dependent, so it cannot be pre-scaled into an
  address-indexed branch table without duplicating the cycle domain.
- Polynomials are already fused inside one dispatch. Cross-driver fusion
  changes ownership/relations for a launch-only saving, while cross-round
  fusion is blocked by transcript challenges.

## Harness and correctness

`metal::st6b_bench` records whether every round actually launched on device;
`st6b_rav_microbench` reports `w1/w2/w4/w8`, adoption, dense-device,
host-tail, and GPU-wait totals. Arms alternate `AB/BA`; every retained report
still uses the existing wire-polynomial + output-claim CPU-twin byte oracle.

## Verification

- CPU-twin byte parity: RAV + Bool, sync + deferred, `2^22`: **green**.
- Targeted Metal parity nextest: **14/14** (one known-class leaky flag).
- Clippy `-D warnings` with and without Metal; fmt: **green**.
- Kernel inventory: `KernelId::ALL = [Self; 73]`; no kernel added.
