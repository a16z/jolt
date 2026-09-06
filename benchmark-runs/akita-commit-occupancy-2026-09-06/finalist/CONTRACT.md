# D23 finalist: frozen full-proof validation

Parent binary72417a5f0fdeed24df3ffc52eb531761ee6df25f0fa692bedac92da056bce90c.
Candidate Akita7878e5ba1, Jolt b2f89f9b9 atop accepted554f62703 (only local
dependency overrides). Candidate binary/source/guest hashes go in manifest.json
after the successful release build and before any proof. No capture hook.
Five guest ELFs and all inputs must match the original five-workload manifest.

One serial40-minute epoch, at most12proofs,120s cooling before each process.
Sequence fixed before results:

1. Fibonacci parent, candidate, candidate, parent.
2. BTreeMap parent, candidate.
3. SHA2-chain parent, candidate.
4. SHA3-chain parent, candidate.
5. Collatz parent, candidate.

Every command uses --scale28 --backend metal --format none, no input override.
Require exactly one PROOF_VERIFIED marker, matching name/scale/backend,
exact parent/candidate trace lengths, padded_len=268435456, finite positive
prove_s, exit0, zero swaps/watchdogs, <88GiB process-family and reported RSS.
180s process timeout; 40-minute epoch deadline includes cooling. Do not kill
unrelated processes. Reuse the original matrix observer with only artifact
directory/event destination changed; its measurement parsing stays unchanged.

Fresh-parent discrepancy >5% against the same-format accepted per-workload
reference stops and reassesses. Fibonacci closing-parent drift >3% stops.
References from immutable resume/runs/t28_NAME_metal_r1.out:
Fib36.182954209, BTreeMap27.933796875, SHA2 30.040168458,
SHA3 40.686400166, Collatz31.813116250 seconds.

Promotion requires >=1s Fibonacci paired-mean saving, >=3% higher arithmetic
mean padded MHz across five workloads (average Fib observations within each
variant first), and no observed per-workload regression. Full-proof results
are not component timings; M5 projections must remain separately labeled.
No performance retries in this initial twelve-proof epoch. An ambiguous cell
requires the contract's separately preregistered targeted-refresh decision,
never an automatic lucky rerun. A losing Fibonacci mean stops transfer early.

These proofs do not bypass the remaining PCS/Jolt nextest, clippy and fmt
gates or inherited-failure accounting. No accepted pin update or push until
all required validation is resolved. Twenty-percent panel throughput is a
milestone, not proof of optimality or a ceiling on further supported work.
