# Three-workload projection and BLAKE2b T28 attempt

2026-09-06. No new optimization goal or default production-pin change.

## Accepted three-workload projection

Reuse the validated Akita7878e5ba1 Metal measurements. For each workload,
projected M5MHz=1.13*268.435456/M4seconds. Average the three individual rates.

| Workload | Measured M4 Metal seconds | Projected M5 seconds | Projected M5 MHz |
|---|---:|---:|---:|
| Fibonacci |30.5041648955|26.9948361907|9.9439557293|
| SHA2-chain |27.236926333|24.1034746310|11.1367950103|
| BTreeMap |27.089530625|23.9730359513|11.1973909581|
| Arithmetic mean |28.2768739512|25.0237822577|10.7593805659|

These are padded T28 proving rates, not hashes/second or an M5 measurement.
Fibonacci uses two observations per variant; the other two use single pairs.
Source:../akita-commit-occupancy-2026-09-06/finalist-rebased/result.json.

## BLAKE2b: resource-stopped, no verified timing

Workload:BLAKE2b-512 chain,64-byte seed of0x05,112682hashes,each digest
fed into the next hash. Conservative2144rows/hash was fixed from execute-only
probes before timing, aiming at roughly235million real rows padded to T28.
The actual full-run trace length was not emitted before termination; this
estimate must not be presented as a measured trace length.

New workload support is isolated at Jolt b8476e3c490b108edf78ebd5912409ea5cfbd628
in /private/tmp/jolt-blake2b-t28-20260906,against accepted Akita7878e5ba1.
No field,security-parameter,protocol,transcript,verifier or kernel changes.
Binary:bin/modular_benchmark_candidate_checked
SHA256 c0d8b5ce46d2abc40c78a4175187de533cbed3124de67c773acfa8ada1c8b8b3.
Guest ELF SHA25671d828e99bd7f96f72dc2af7e5df59430ebdb4d287d60eda4661fb0087f00c2a.
Full fingerprints and exact input are frozen in manifest.json.

Small guest chains at0,1,2iterations match Python hashlib.blake2b. The full
112682-iteration guest output matches the native Rust BLAKE2b reference and
the independently precomputed Python digest:
b9f2586bcd014b98d08714b3e603197d026fc53db8ef7295079da1a277fa9ffd3267c08e733c22844cf6af71f78fd6f7da66a233946f75e659e4af7251673ddb.
This checks execution output; it does NOT mean the proof verified.

The120second cooldown began16:05:25UTC; the process started16:07:25UTC.
At16:07:49UTC the two-second RSS sampler observed97354645504bytes
(90.668579GiB),beyond the88GiB stop threshold. The observer terminated the
process group. Sampling can overshoot the threshold between checks; the
90GiB absolute ceiling was exceeded at detection. No retry or memory-limit
increase was made. No watchdog/abort marker,PROOF_VERIFIED or MATRIX_TIMING
was emitted. No final process-peak or swap report was obtained because the
process group (including /usr/bin/time) was terminated. Do not infer a
successful proof time,phase attribution or zero swaps from this failed run.

Raw:runs/t28_blake2b-chain_metal_r1.out
SHA2567c0565e11182d39febe2dc0820642fd01c2f262e2dd99f080b5b759de970cb29.
Failure and sampled memory are in events.jsonl. Controller91310 exited1.
Read-only checks confirm no benchmark child and no machine lock remain;
every frozen source,binary,guest and observer fingerprint still matches.

The optional parent comparison was dropped before any timings because its
build plus two cooled proofs would exceed the original25minute reserve.
No Blake-versus-parent improvement claim,CPU result or Blake M5 projection
is available. The three-workload projection above is unaffected.

## Validation and next bounded option

Preparation passed. The first harness build passed but focused clippy caught
format_collect in untimed digest reporting. Its failure and unmeasured binary
are retained. The corrected helper passes focused release clippy; the rebuilt
artifact is the only timed binary. Final fmt and scoped diff checks pass.
Inherited Cargo manifest warnings about ignored default-features remain.
Kernel parity/security gates were reused from the unchanged accepted source;
no new full workspace test matrix or fresh parent timing was run.

A smaller chain,for example80000hashes (roughly167million real rows),should
still pad to T28 and reduce live-trace storage. Existing CLI override
--target-trace-size 171520000 selects80000hashes at2144rows/hash. This is an
untested alternative,not a guarantee of fitting the cap or a measured result.
It requires a new explicit input/manifest and cooled observation,with the
same memory cap. Preserve the failed112682-hash point; never substitute the
smaller workload silently. No additional attempt is in flight.
