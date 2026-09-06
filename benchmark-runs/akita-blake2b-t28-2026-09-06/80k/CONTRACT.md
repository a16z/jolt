# BLAKE2b-512 chain: 80,000 hashes

User explicitly approved this smaller input after the 112,682-hash attempt
stopped on memory. Preserve the earlier contract, manifest and failed run.
This is one new observation, not a retry of an identical workload and not
an optimization or parent comparison. No build, CPU run, subagents or push.

Use the same checked binary, AOT guest, seed (64 bytes of 0x05), accepted
kernel and proof/security parameters. Exactly 80,000 hashes are selected by
--target-trace-size 171520000 (80000 * 2144). Independently compute the final
digest with Python hashlib.blake2b. Require actual trace length strictly
above 2^27 and at most 2^28, padded trace length exactly 2^28, matching hash
count and digest, exactly one successful proof verification and timing row.

Reuse the frozen observer under the same machine lock, with 120 seconds
cooldown, 180 seconds proof-process timeout, 88 GiB sampled family-RSS stop,
90 GiB ceiling, zero reported swaps and no watchdog abort. The two-second
RSS sampler can overshoot the stop threshold. Do not raise limits or retry
automatically on failure. The fresh deadline in manifest.json allows eight
minutes from manifest preparation for this single cooled observation.

M5 results remain projections using the unchanged 1.13 throughput multiplier.
Report the real trace length alongside padded MHz. Compare descriptively to
the existing Fibonacci, SHA2-chain and BTreeMap observations; the old larger
BLAKE2b attempt has no valid proof time. No parent-kernel gain can be claimed.
