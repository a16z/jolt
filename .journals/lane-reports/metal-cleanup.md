# Lane C19 — PR-handoff cleanup (rig audit · commitment.rs split · dead-code sweep)

Branch `lane/metal-cleanup` off `scratch/metal-saturation` @ c4aceaedd.
Perf phase closed at wave 18; this lane makes the trunk PR-presentable
per the repo CLAUDE.md handoff rule. **Net −8,898 LOC** (2,621+ / 11,519−
across 7 commits; the split moves 2,537 of the deletions, so pure rig/
probe removal ≈ −8,982 LOC). Untouched by charter: env kill switches,
legacy kernel arms in `KernelId`/shaders, fixture-oracle parity tests,
journals.

## 1. Rig/probe inventory (verdict · mechanism)

Everything Metal-related is younger than the merge-base with main
(e4679b5d2), so predecessor-campaign rigs were audited too. Deletion was
the default; keeps needed plausible future use.

| rig | origin | verdict | where it went / why |
|---|---|---|---|
| `jolt-eval` bench `bytecode_lazy` (993 L) + BytecodeLazy/RavLazy fixtures + probe ladder in `bytecode_read_raf.rs` + `ra_lazy`/`booleanity` bench seams | w18 L18 | **DELETE** (4a1b7dbfe, 9b4252eb0) | st6b doors closed premise-false (155 ms isolated @2^27); kill-listed |
| bench `irr_cycle` (299 L) + IrrCycleFixture | w15 P15 | **DELETE** (4a1b7dbfe) | cycle waits priced at the cycle-exec ALU roof; door closed |
| bench `irr_dispatch_context` (300 L) | w5 S5 | **DELETE** (4a1b7dbfe) | question answered (data distribution, not clock); superseded by `irr_roof` |
| benches `instruction_read_raf_{phase,suffix}_scan` (226 L) | w3 | **DELETE** (4a1b7dbfe) | random-key objectives — the fixtures behind the +5.4 s st5 model error; real-row measurement lives in `irr_roof` |
| benches `inc_prepare`, `instruction_input_round0`, `spartan_outer_claims`, `registers_rw_prefix`, `ram_rw` (412 L) + their slot fixture mods | w2-w3 ports | **DELETE** (4a1b7dbfe) | port go/no-go objectives; ports shipped, parity pinned by each slot's non-bench lockstep tests |
| bench `registers_address_first_phase` + `RegistersAddressPhase` objective (1,080 L) | Gate-1 | **DELETE** (4a1b7dbfe) | address-first restructuring dead both arms (42× over kill line), kill-listed |
| examples `miller_cpu_probe` (245 L), `miller_microbench` (528 L), `mulrate_microbench` (779 L) | W3/W6/W7 | **DELETE** (9b4252eb0) | one-shot calibration/go-no-go probes; every priced candidate is in the kill list |
| example `miller_commit_shape` (284 L) | w13 M13 | **DELETE** (9b4252eb0) | table-vs-fly repricing; verdict shipped as the ≤2^15-rows scale gate |
| example `st6b_rav_microbench` + `metal::st6b_bench` (462 L) | W2 | **DELETE** (9b4252eb0) | st6b adoption harness; family closed w18 |
| `extract_microbench` + `CommittedColumnsWitness` re-export + libc dev-dep | w8 E8 | **DELETE** (9b4252eb0) | marked delete-at-handoff in its own commit message |
| E8 telemetry (`telemetry.rs`, ParPhase spans in the st0 driver + record walk, `JOLT_ST0_TELEMETRY`) | w8 E8 | **DELETE** (dfed83caf) | journal marked delete-at-handoff; `tracing` taxonomy spans stay |
| `st0-contention` g1bat leg (512 L) | w13 T13 | **DELETE** (a7ce2f694) | batched-affine door closed permanently (inversion 388-401 mul-equiv vs I≤29) |
| bench `metal_fr_bind` | W1 | **KEEP** | window-health gate for every certification (<350 µs rule) |
| bench `irr_roof` (+ IrrPhase/SuffixScanFixture, variant probes, `JOLT_IRR_DUMP_ROWS`) | S12/G17 | **KEEP** | st5 family attribution rig, real-row roofs |
| bench `miller_multipair` | W4/B | **KEEP** | st8 pairing family rig (fly/split/TG-cap arms) |
| bin `st0-contention` (base legs + g1x) | W1-2/X9 | **KEEP** | st0 family rig; g1x now documented in the header |
| examples `metal_microbench`, `pairing_pipeline_stats` | W1/W3 | **KEEP** | device characterization (DEFAULT_MIN_TERMS evidence) + the only no-toolchain register-pressure reading |
| `JOLT_METAL_CB_TRACE`, `JOLT_LIFETIME_TRACE` | W1/w-era | **KEEP** | env-gated observability the retained rigs and RSS work depend on |
| `JOLT_REGRW_FUSED` opt-in arm | w3/w4 | **KEEP** (flag for reviewer) | kill-list decision ("probe kept"); the sync-fused branch shares `launch_bind_and_message` with the shipped B14 overlap path, and two parity tests pin its exact CB schedule (5/7) — deleting it means rewriting cert-story test schedules for ~40 lines of default-off code |

Keep-list documented in **`crates/jolt-kernels/BENCHES.md`** (rig →
command → measures → lane report), commits b7ec74296 + 1de413c21.

Two bench-fixture self-check tests were deleted with their fixtures
(`ram_rw_bench_oracles_and_dispatch_schedule`,
`outer_claims_device_matches_host_oracle`) and two with the
address-phase objective — production parity in those slots stays pinned
by non-bench lockstep tests. Metal suite 416 → **412**, all four
accounted.

## 2. commitment.rs split (ce3b55145, pure code motion)

2,537 lines → `metal/commitment/` along the lane structure the module
docs already describe; crossings are `pub(super)`, no renames, no
behavior change.

| old commitment.rs lines | new file | content |
|---|---|---|
| 1-807 (docs, consts, slot front, job types, driver) | `mod.rs` (821) | `MetalCommitWitness`/`dory_commit_slot`, geometry + env knobs, `StagedChunk`/`GpuJob`/`SegOut`/`IncSeg`, `commit_streaming_metal` |
| 809-1266 (Miller lane) | `tier2.rs` (465) | `MillerLane`, batch/settle/drain, CPU absorb fallback |
| 1268-1949 (driver/builder machinery) | `builder.rs` (691) | `extract_columns`, `MetalColumns`, `build_one_hot_job`, `build_inc_job`, `SlabPool`, `reduce_inc_superchunk` |
| 1951-2171 (bench fixture) | `bench.rs` (234, bench-utils) | `G1SegBench*` for st0-contention |
| 2173-2537 (tests) | `tests.rs` (362) | unchanged parity/schedule tests |

All files < 1000 lines. (E8-telemetry removal shrank the file from
2,587 to 2,537 before the split.)

## 3. Dead-code sweep (1de413c21)

Tree was already clean after the deletion commits carried their own
reference fixes: no commented-out probes, no scratch files/untracked
dirs, no stale rig references (`JOLT_IRR_DUMP_ROWS` doc comment
repointed from the deleted `irr_dispatch_context` to `irr_roof` in
4a1b7dbfe). Sweep delta: documented the two env-gated tracing switches
in BENCHES.md.

## 4. Gates (all green, run under the cargo lock)

- clippy `--all --features host` `-D warnings` ✓
- clippy `--all --features host,zk` ✓
- clippy `-p jolt-kernels -p jolt-eval` metal+bench-utils ✓
- nextest metal suites: **412/412** (416 − 4 deliberate, see §1) ✓
- nextest `jolt-prover --features prover-fixtures`: **20/20** byte
  parity ✓
- `cargo build --release --example modular_benchmark
  --features prover-fixtures,metal` ✓
- `cargo fmt` clean, tree clean ✓

No timed GPU runs (pure refactor; parity suites are the gate; the
commitment split additionally re-ran the slot's 5 parity tests
including the byte-identical oracle before committing).

## Commits

```
4a1b7dbfe  Delete dead one-off campaign benches and their fixture modules   (−5,446)
9b4252eb0  Delete dead probe examples and the st6b bench harness            (−2,807)
dfed83caf  Remove the E8 st0 telemetry (JOLT_ST0_TELEMETRY)                 (−202)
a7ce2f694  Drop the w13 batched-affine (g1bat) leg from st0-contention      (−509)
b7ec74296  Document the retained Metal bench rigs (BENCHES.md)
ce3b55145  Split metal/commitment.rs into a module directory                (move)
1de413c21  Document the env-gated CB/lifetime tracing in BENCHES.md
```

## Reviewer-will-still-flag list (deliberate keeps)

1. `JOLT_REGRW_FUSED` — default-off rejected-path arm kept per the
   kill-list decision; see the table row for the parity-schedule
   rationale.
2. The st2 RAM-RW device port (env-gated default-off) — kill-listed
   "below bar" but left in place as a legacy kernel arm per charter.
3. `st0-contention` at ~990 lines — the retained st0 family rig; legs
   are documented, could be slimmed further if the PR wants.
4. Kill switches galore (`JOLT_*` env arms) — certification story,
   untouched by charter.
