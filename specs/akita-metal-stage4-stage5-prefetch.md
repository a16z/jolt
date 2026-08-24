# Akita Metal Stage 4/5 compatibility scatter

## Boundary

At BTreeMap T28, Stage 4 starts the Instruction Read-RAF compatibility worker after
`RegistersReadWriteChecking::prepare`. The worker groups the Stage-1 resident rows,
constructs four dense consumer planes, and computes the first address-phase message.
Stage 5 joins an already-complete worker. The transcript point, grouped-row order,
scatter kernel, address-phase schedule, proof, and verifier are unchanged by this
campaign.

The accepted trace measures 6.0396 s for Stage 4 and 2.6015 s for Stage 5, or
8.6411 s combined. The untraced compatibility construction occupies 1.5546 s from
the worker release to the visible address-prefetch span; that span adds 0.1805 s.
Both finish before Stage 4's second register round. Treating their 1.7351 s total as
additive latency is therefore invalid.

The optimized-CPU trace measures 2.5317 s across the register rounds versus 4.0953 s
in the Metal trace. Only the first two Metal rounds overlap the active worker. They
take 1.6456 s versus 0.9366 s on CPU, a 0.7091 s difference. The later-round
0.8545 s difference and the 0.4870 s prepare difference have no simultaneous
compatibility work and are excluded from this candidate's causal ceiling.

## Traffic and storage

For `T = 2^28`, the grouped consumer writes one byte of packed metadata, 16 bytes of
lookup limbs, four bytes of inverse permutation, and 16 bytes of equality weight per
row. The four planes total `37*T = 9.25 GiB`. It also reads 33 bytes per Stage-1 row.
The fused bytecode outputs are already private Metal buffers. The four grouped planes
are allocated with `StorageModeShared`, although the resident address sequence only
binds them to later GPU commands. Its validation reads buffer lengths and allocation
identities, not their contents; CPU materialization exists only in the separate
non-resident construction branch.

Changing these four outputs to `StorageModePrivate` does not reduce their physical
capacity or the scatter's compulsory reads and writes. It removes CPU mapping and
coherence from a GPU-only lifetime. At the measured 412.5 GiB/s, the roughly
19.9 GiB read/write stream has a traffic floor near 0.05 s; 268 million fp128 weight
multiplications add about 0.02 s at the measured arithmetic rate. The observed
1.5546 s construction interval is therefore dominated by allocation, residency,
synchronization, or inefficient execution rather than compulsory traffic. Private
storage can address the first two mechanisms, not the floor.

## Candidate B1: private grouped consumer planes

Change only the four compatibility-scatter output allocations from shared to private
Metal storage. Keep the source, equality tables, status word, bytecode outputs,
dispatch geometry, worker release point, and address prefetch unchanged. Record the
storage mode, exact output bytes, compatibility wall time, and address-prefetch wall
time at the existing `jolt::metal` telemetry boundary. The capacity ledger must
continue charging all 9.25 GiB even if process RSS accounts private storage
differently.

The prediction is a 0.3--0.7 s complete-prover saving. The upper edge is the
0.7091 s difference in the only two register rounds that overlap the worker; no later
Stage 4 difference is credited. The lower edge allows most of the interval to be GPU
execution or unavoidable physical allocation.

Add a red storage-lifetime assertion to the existing fused Stage-1 scatter oracle,
then run that one focused Metal test. Because the production prefetch is admitted
only at T28, a T25 proof does not exercise the candidate. Admit one BTreeMap T28
treatment after the focused test passes. The full proof must verify, telemetry must
report 9.25 GiB of private grouped outputs with no fallback, and the exact working-set
ledger must stay unchanged. Retain only at 47.58 s or below against the 48.08 s
parent, with peak RSS below 90 GiB. One repeat is legal only if a promotion result is
within ordinary whole-prover noise. Otherwise revert the storage change and preserve
the result; do not tune threadgroup width or worker timing under the same candidate.

This is a prover-local resource-policy change. It changes no protocol message,
challenge, polynomial, proof byte, verifier code, or soundness assumption.
