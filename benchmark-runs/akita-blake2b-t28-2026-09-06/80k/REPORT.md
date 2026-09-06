# BLAKE2b-512 chain: 80,000 hashes, padded T28

Completed 2026-09-06 at 16:34:25 UTC. One successful observation on the M4 Max,
using the unchanged accepted Metal kernel and proof/security parameters.

## Result

- Measured prover boundary: **30.70356125 s**, **8.74281175 padded MHz**.
- Actual trace: **166,883,688 rows**, padded to **268,435,456 (2^28)**.
- Peak process RSS: **83.42268372 GiB**; sampled process-family peak: **81.42767334 GiB**.
- Zero reported swaps, no watchdog/abort marker, exit 0.
- Exactly one `PROOF_VERIFIED backend=metal value=true`.
- Exactly 80,000 hashes; final guest output matches the native reference and
  an independently computed Python hashlib.blake2b digest.
- Whole process: 36.72 s. The 120-second cooldown and untimed preparation are
  not included in the prover boundary.

The live trace occupies 62.169% of padded capacity. Padded MHz measures proof
capacity, not hashes per second or actual executed-row throughput.

## Comparison

Use the unchanged projection: M5 seconds = M4 seconds / 1.13;
M5 padded MHz = 1.13 * 268.435456 / M4 seconds.

| Workload | Measured M4 seconds | Projected M5 seconds | Projected M5 MHz |
|---|---:|---:|---:|
| fibonacci | 30.50416490 | 26.99483619 | 9.94395573 |
| sha2-chain | 27.23692633 | 24.10347463 | 11.13679501 |
| btreemap | 27.08953062 | 23.97303595 | 11.19739096 |
| blake2b-chain | 30.70356125 | 27.17129314 | 9.87937728 |

The arithmetic mean of individual projected rates is **10.75938057 MHz** for
Fibonacci, SHA2-chain and BTreeMap; including this BLAKE2b observation gives
**10.53937974 MHz**. BLAKE2b is 0.1994 s (0.65%) slower than Fibonacci's
prior mean, descriptively. There is no replicate uncertainty estimate for
BLAKE2b. These are projections, not measurements on an M5 Max.

Earlier workloads reuse the [accepted observations](../../akita-commit-occupancy-2026-09-06/finalist-rebased/result.json).
Fibonacci uses two candidate observations; SHA2 and BTreeMap use one each.
No CPU or parent-kernel BLAKE2b run was performed.

## Recovery and evidence

The first 80k launch exited 101 before tracing, in 0.01 s, because the prior
112,682-hash memory termination left an empty workload-local lock. Read-only
process checks found no live proof/build process. The stale lock was preserved
as stale-workload.lock.saved. That launch's manifest, events and raw log remain
unchanged. recovered/CONTRACT.md preregisters one infrastructure-only recovery,
with the same full cooldown, input, binary and resource guards.

The recovered execution completed normally. Both machine and workload locks
are absent; a process-table check found no remaining benchmark process.
The larger 112,682-hash run remains a separate failed memory point: this
smaller input must not replace it or imply a speedup from a kernel change.

Successful raw log: recovered/runs/t28_blake2b-chain_metal_r1.out
SHA256: bfbf5f0f31b2d221defbe6f6bbab4d61ac51e832d19d64290d84d01d6e9a5bba

Binary SHA256: c0d8b5ce46d2abc40c78a4175187de533cbed3124de67c773acfa8ada1c8b8b3
Successful manifest SHA256: bb0ae51e0f24145cc835c27d44fc46b01c619dea8147af29cd48fd295037b680

Independent post-run checks matched all 25 frozen file hashes, the start-time
manifest hash, raw hash, exact input/digest, verification/timing identity,
trace padding, finite positive timing, RSS limits and zero swaps.
Structured result: recovered/result.json; independent check: validation.json.

No source, kernel, protocol, security parameter or production pin changed in
this request. No compilation, extra proof observation, CPU run or push.
