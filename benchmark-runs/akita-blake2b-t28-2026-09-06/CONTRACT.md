# BLAKE2b T28 comparison

User requested BLAKE2b hash chain and the projected M5 average over Fibonacci,
SHA2-chain and BTreeMap. This is a one-workload measurement, not a resumed
optimization goal. Start15:47UTC; total reserve25minutes, deadline16:12UTC.
One M4 Max, serial, no subagents, no pushes or default production-pin changes.

The frozen harness has no Blake workload. Extend an isolated Jolt checkout
from b2f89f9b9 with a BLAKE2b-512 chain entry in hashbench-guest, a profile
workload selector, native reference-output check and an AOT sizing helper.
Seed:64 bytes of0x05. Each iteration hashes the entire previous64-byte digest.
The guest uses the repository's inline BLAKE2b implementation. Wire input and
output use two32-byte arrays, avoiding a new serialization dependency.
No prover/kernel/protocol/parameter/evaluator-timing changes. Preserve all
earlier source, binary, guest and measurement artifacts.

First compile the guest, check small chain outputs independently, and measure
execute-only rows at64,1024,4096iterations. Set a conservative cycles/hash
estimate from those measurements before either scored proof; target90% of
T28 capacity, accepting only2^27<trace_len<=2^28. Freeze the chosen constant,
input and ELF hash. AOT preparation is outside proof timing.

Build identical extended Jolt harnesses with Akita7878e5ba1 (accepted staged
radix26) and d756e3a67 (prior kernel). Keep immutable binary hashes. Run ONE
parent then ONE optimized Metal observation, each preceded by120s cooldown,
under ../akita-10mhz-studies/scratch/machine.lock. Reuse the original matrix
observer:format none,exactly one PROOF_VERIFIED backend=metal value=true,
one MATRIX_TIMING with correct workload,T28 and identical real trace lengths;
exit0,zero swaps,180s proof-process timeout,88GiB family-RSS stop,90GiB ceiling.
Keep watchdogs enabled. A watchdog abort is a real bug; stop and preserve logs.
No CPU reference reruns and no automatic timing retries. Report a regression
if observed; this new single pair is descriptive, not a promotion claim.

Preparation/build/check subprocesses:<=12minutes each,88GiB family-RSS limit,
serial machine lock,common total deadline. Focused clippy/fmt on the harness
extension; reuse already completed kernel parity/security checks because no
kernel changes are made. If full pair cannot finish, report completed work
and the blocker rather than exceed the reserve silently.

M5 numbers are projections with the unchanged1.13throughput multiplier.
Three-workload average is the arithmetic mean of individual padded MHz.
Existing accepted M4 seconds:Fibonacci30.5041648955,SHA2-chain27.236926333,
BTreeMap27.089530625. Their M5 mean is10.7593805659MHz,not measured M5.

15:50UTC sizing is frozen before proofs:64hashes137172rows,1024hashes2139742,
4096hashes8547934. Conservative2144rows/hash yields112682hashes atT28.
Guest outputs at0,1,2hashes match Python hashlib.blake2b independently.
The first preparation completed in under one minute; inherited observer
allowed1200seconds. Subsequent checks explicitly cap its deadline to720s.

15:59UTC pre-proof amendment:no timed Blake observations have run. Focused
clippy found format_collect in the new, untimed digest-reporting helper.
Preserve its failed log and original unmeasured binary; replace collection
with write! into preallocated String, then rebuild/check a distinct artifact.
Prioritize the accepted-kernel observation, followed by the parent only if
the original16:12deadline still permits. This order/budget change precedes
all timing data. A lone accepted-kernel observation still answers comparison
with the three prior workloads; no parent gain is claimed without its run.

16:01UTC sequence resolved BEFORE timing:one accepted-kernel observation.
Measured first build cost273seconds; another build plus two cooled proofs
does not fit the remaining worst-case reserve. The parent worktree is unused,
no parent binary is built,and no parent gain will be reported. The fixed
proof observer, T28/input, verification and resource guards are unchanged.
