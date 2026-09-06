# D23 acceptance audit — in progress

Previous goal turn: **progress**. It produced the isolated production finalist,
passed both Metal suites and established the verified Fibonacci paired gain.
Current turn re-polled live session57818 before relying on the running matrix.

Candidate Akita7878e5ba14b0a9f7cc89ddbc1331a543efa11db7; Jolt b2f89f9b9
only redirects dependencies from accepted554f62703. Frozen release binary
216737e3d08ab90b6bbdbdfce6858ae733862c6d1922f8ed65ab2ec9f2907e20.
No accepted pin change, no push. The original parent remains72417a5f0fdeed24.

| Requirement | Current evidence | Verdict |
|---|---|---|
| >=20% useful root-panel throughput gain | D23 full-cost ABBA:11.993870->9.570013s GPU,+25.3276%; parent drift0.695%; all final bytes match | Pass |
| >=1s verified Fibonacci T28 saving | Rebased full-proof ABBA:32.749368->30.504165s,+2.245203s; parent drift1.5133%; all proof markers/trace identities pass | Pass |
| Representative transfer, no observed per-workload regression | BTreeMap27.610286->27.089531s,SHA2 29.383833->27.236926s,SHA3 42.081181->37.023139s,all verified; Collatz pending | Pending |
| >=3% higher arithmetic-mean MHz across five | Requires remaining Collatz pair | Pending |
| Exact arithmetic and security preserved | Radix26 invariant/i32 bound;512-state normalizer probe,ten reduced independent oracles,65target samples,192MiB final equality; no protocol/field/catalog/transcript/verifier changes | Supported; remaining validation below |
| Touched-family CPU/Metal parity | Akita Metal32/32 including all four D128 cases and shape rejection | Pass |
| Jolt Metal suite |329/329,zero skips | Pass |
| Affected PCS serial/parallel | fp128 E2E,commitment-contract,protocol-soundness targets registered in run_validation.py | Pending |
| Jolt clippy host and host,zk | Commands registered; not started during timed proofs | Pending |
| Three fork clippy configurations | Exact configured commands registered; not started during timed proofs | Pending |
| Formatting and final clean scoped diff | Fork fmt/diff-check passed before build; final repeat pending | Pending final check |
| Resource and operational guards | Serial machine lock,120s cooling,88GiB stop,zero swaps/watchdogs on completed scored observations | Pass so far |
| Auditable history/no selective retries | Immutable raw files,source/binary/guest hashes and precommitted gates; one disclosed pre-candidate parent calibration after historical drift | Pass so far |
| Separate measured M4 and projected M5 claims | No M5 measurement or10MHz-average claim made; calculate labeled projection only after matrix | Pending report |

Code review boundary: accepted-to-candidate diff names only
crates/akita-metal/src/kernels/onehot.metal,
crates/akita-metal/src/packed_onehot_fp128_d128_rank3.rs and
crates/akita-metal/src/runtime.rs. The original16-position API alignment is
preserved while internal tiles use8positions and20KiB. D512 is unchanged.
The capture hook is absent. The production normalization/final-reduction
bodies match the independently checked diagnostic bodies modulo names/comments.
The bound is per sixteen possible inputs, not an assumption about sparse data.

Could not verify yet: remaining transfer/timing gates, PCS feature graphs,
clippy and final integration. Physical occupancy, register allocation, spills,
actual DRAM traffic and an ISA-based optimum are unavailable with current
tools; no optimality or new constant-time guarantee is claimed. Known inherited
preflight failures (line caps,error-owner test,unused dependencies,missing
typos tool) remain separate from candidate validation; see analysis.md.

Evidence: analysis.md and radix26.md for preregistration/math/results;
validation.md for test/proof records; events.jsonl and runs/ for raw hashes;
finalist-rebased/manifest.json for exact evaluator/guest/source fingerprints.
Acceptance remains **unproven** until every pending required gate is resolved.
