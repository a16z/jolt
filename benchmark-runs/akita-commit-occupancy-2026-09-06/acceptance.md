# D23 acceptance audit — accepted locally, 2026-09-06

Previous goal turn: **progress**. It produced the isolated production finalist,
passed both Metal suites and established the verified Fibonacci paired gain.
The concluding turn completed all transfer, software and fingerprint gates.

Verdict: **accept**, at the predeclared evidence stage: revalidated component
and Fibonacci gain, with observed transfer to four other workloads.
Accepted campaign source: Akita7878e5ba14b0a9f7cc89ddbc1331a543efa11db7.
Isolated Jolt b2f89f9b9 only redirects dependencies from accepted554f62703.
Release SHA216737e3d08ab90b6bbdbdfce6858ae733862c6d1922f8ed65ab2ec9f2907e20.
Default production selection and dependency pin remain unchanged under the
contract's frozen-source/editable-path boundary; main Jolt has local campaign
documentation/tool commits. No push. This is not a
claim that the default checkout now selects the optimization.
Comparison parent SHA72417a5f0fdeed24df3ffc52eb531761ee6df25f0fa692bedac92da056bce90c.

| Requirement | Evidence | Verdict |
|---|---|---|
| >=20% useful root-panel throughput | Full-cost ABBA:11.993870->9.570013s GPU,+25.3276%; parent drift0.695%; all final bytes match | Pass |
| >=1s verified Fibonacci saving | Full-proof ABBA:32.749368->30.504165s,2.245203s saving; parent drift1.5133% | Pass |
| No observed individual transfer regression | All five workloads improve and all proofs verify | Pass; four single-pair transfers |
| >=3% higher mean padded MHz across five |8.415475741->8.909192209MHz,+5.866768%; independently reconstructed | Pass |
| Exact arithmetic/security scope | Radix26 invariant/i32 bound;512-state probe,ten independent reduced oracles,65target samples,192MiB final equality; no protocol/parameter changes | Pass |
| Touched-family CPU/Metal parity | Akita Metal32/32,including all four D128 cases and shape rejection | Pass |
| Jolt Metal suite |329/329,zero skips | Pass |
| Affected PCS serial/parallel | fp128 E2E,commitment-contract,protocol-soundness:23/23 in each mode | Pass |
| Jolt clippy host and host,zk | Both exact workspace commands exit0 | Pass |
| Three fork clippy configurations | Parallel/disk-persistence,serial,and PCS configurations exit0 | Pass |
| Final formatting/diff/fingerprints | Both isolated workspaces fmt-check and scoped diff-check; all frozen fingerprints | Pass |
| Operational guards | Serial lock,120s cooling,max RSS86.934586GiB,zero swaps/watchdogs | Pass |
| Auditable history | Immutable raw files/hashes; disclosed extra pre-candidate parent calibration; no selective candidate retries | Pass |
| Measured versus projected claims | M4 mean8.909192MHz; frozen1.13factor projects10.067387MHz M5 | Pass; thin uncertain margin |

## Measured end-to-end result

T=2^28, frozen Metal evaluator. Fibonacci is the mean of two observations
per variant; each other row is one parent/candidate pair, descriptive rather
than an independently replicated effect estimate.

| Workload | Parent seconds | Candidate seconds | Saving seconds |
|---|---:|---:|---:|
| Fibonacci |32.749368|30.504165|2.245203|
| BTreeMap |27.610286|27.089531|0.520755|
| SHA2-chain |29.383833|27.236926|2.146907|
| SHA3-chain |42.081181|37.023139|5.058042|
| Collatz |31.054732|30.745959|0.308773|

Arithmetic mean of padded rates, not the rate of mean wall time:
mean((2^28/1e6)/seconds)=8.415475741 -> 8.909192209MHz,+5.866768%.
Frozen1.13factor projects10.067387197MHz M5. Its0.674%margin is smaller
than plausible transfer/projection uncertainty: no robust10MHz guarantee
and no M5 measurement. The SHA3 effect is not precisely attributed in full
to this panel; the small Collatz/BTreeMap effects may include noise.
Every scored proof emits exactly one verified Metal proof marker.

## Mechanism and scope review

Five signed radix26 digits replace a carry chain on every selected update.
Carry normalization is amortized across at most sixteen contributions.
Digit decoding moves into cooperative shared staging: roughly17.84 uses
per staged field no longer each repeat extraction. The20KiB internal tile,
extra25% shared traffic and doubled tile barriers are included in measured
cost. Complete-panel wall saves2.425245s. No expanded global matrix, extra
trace pass, parameter/catalog/transcript/verifier change, or new witness storage.

Accepted-to-candidate production diff contains only:
crates/akita-metal/src/kernels/onehot.metal,
crates/akita-metal/src/packed_onehot_fp128_d128_rank3.rs,
crates/akita-metal/src/runtime.rs.
Original16-position API alignment is preserved; internal tiles use8positions.
D512 is untouched. The capture hook is absent. Production normalization and
final-reduction bodies match the independently checked diagnostic modulo
names/comments. The signed bound is for sixteen possible contributions,
not an assumption about sparse inputs.

The20% milestone was not a ceiling; the accepted result reaches25.3%.
No unevaluated preregistered candidate remains. Grid growth, shared reservation,
widened carry, state halving, column grouping and reuse metadata failed their
measured gates. A twelve-position staged tile is an unpriced future idea:
fewer barriers trade against30KiB storage, changed carry cadence and tail
handling. No additive gain is promised. Do not restart rejected mechanisms
without new causal evidence or call the current kernel optimal.

## Handoff and limitations

Runnable immutable artifact: bin/modular_benchmark_radix26_7878e5ba1.
Source:/private/tmp/akita-kernel-campaign-20260906.
Jolt integration:/private/tmp/jolt-commit-occupancy-20260906.
Exact source/binary/guest fingerprints:finalist-rebased/manifest.json.
A future bounded epoch can use7878e5ba1 as its accepted local parent.
Default-branch publication/pinning is not performed under this no-push,
frozen-source contract. All user changes are preserved; no GPU process or
machine lock remains.

Could not verify: physical occupancy, register allocation, spills, actual
DRAM traffic, an ISA-based optimum, or actual M5 performance. No new
constant-time guarantee or independent security audit is claimed.
Inherited preflight failures remain: backend/runtime line caps, the
recursive_commit error-owner test (Python suite78/79), two unused Metal
dependencies; typos is unavailable. These are not hidden by the passing
candidate nextest/clippy gates. Prior evidence:analysis.md.

Evidence:analysis.md/radix26.md for preregistration/math; validation.md for
test records; events.jsonl and runs/ for raw results/hashes. All candidate
acceptance gates are resolved. finalist-rebased/result.json remains the
immutable timing-time snapshot (production_accepted:false then); this audit
and the subsequent acceptance event record the final decision.
