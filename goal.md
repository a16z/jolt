# Goal: Akita Metal software campaign to the measured ceiling

Execute the software campaign in `specs/akita-metal-10mhz-attack-strategy.md` sections 7.3a
and 7.4: on Jolt `feat/akita-metal` (worktree `/Users/mgeorghiades/worktrees/jolt/bright-ridge/jolt`)
with the Akita worktree `/Users/mgeorghiades/worktrees/akita-metal-eval-proof` at `0e52ebf`, raise
the verified Metal prover at T=2^28 as far as the priced protocol-preserving levers allow, with a
hard acceptance of at least 5x over the frozen optimized-CPU references for BTreeMap, Fibonacci,
and SHA-2 chain, and a stretch of 9 MHz on BTreeMap. Do not target 10 MHz: study S1 in
`benchmark-runs/akita-10mhz-studies/analysis.md` shows the commit is at the audited SIS table
floor for the current fold design, and section 7.3a bounds the software total at 5-9 s per
workload.

## Read first

The whole strategy specification, `benchmark-runs/akita-10mhz-studies/analysis.md`,
`specs/akita-metal-protocol-preserving-5x-ledger.md` (closed candidates), both repositories'
`AGENTS.md`, and the `CLAUDE.md` lint and test rules. Treat the specification and the study
ledger as canonical over live workspace state.

## Step 0: commit the SHA-2 fix

The compact-rs1 operand-claim fix is uncommitted in
`crates/jolt-kernels/src/metal/solinas/registers_read_write/{fused_sequence.metal,sequence.rs}`.
Confirm the registers evaluator parity first:

```bash
./target/release/examples/metal_registers_read_write_cpu_eval --name sha2-chain --scale 28 \
  --target-trace-size 150000000 --samples 1 --arm metal --metal-source stage1
```

must print checksum `ea64db14ba7e7aad`, equal to `--arm cpu`. Then commit the two files alone
with a message that names the root cause: the register remap was applied to the raw compact
rs1 plane when a trace touches a register index >= 64, on the T=2^28-only compact route.
Do not push.

## Fixed evaluator

Build once per source change:

```bash
cargo build --release -p jolt-prover --example modular_benchmark --features prover-fixtures,metal
```

Score the reported `jolt_prover::prove` wall with `PROOF_VERIFIED backend=metal value=true`:

```bash
./target/release/examples/modular_benchmark --name fibonacci --scale 28 --backend metal
./target/release/examples/modular_benchmark --name sha2-chain --scale 28 --backend metal
./target/release/examples/modular_benchmark --name btreemap --scale 28 --target-trace-size 150000000 --backend metal
```

Frozen CPU references (re-measured 2026-09-03 with the D512 CPU accumulator fix, 09e649061,
verified, idle machine): BTreeMap 155.11 s, Fibonacci 165.05 s, SHA-2 170.32 s
(1.73 / 1.63 / 1.58 MHz). Earlier samples (Aug 23: 166.548 / 215.177 / 213.703; pre-fix
2026-09-03: 180.29 / 196.76 / 211.18) are superseded. Do not rerun
them. Frozen Metal parents from 2026-09-02: BTreeMap 35.79 s, Fibonacci 36.67-38.87 s, SHA-2
43.21 s. Never run a scored proof while a build or another proof is running; the Aug 30 traces
show how badly overlap corrupts results. Repeat a scored run only when a result lies within
0.3 s of its gate. Peak RSS must stay at or below 90 GiB with no swap growth.

## Candidate rules

One lever at a time, in this order unless a measured result changes the ranking:

1. Stage-6b Booleanity accelerator lane (4.0 s flat, 143 ms/round in all three workloads),
   using the K001-K010 matched-service method with a written floor before code.
2. Stage-5 instruction read-RAF address rounds: CPU tail for small suffix tables (129 rounds
   at 13-14 ms with almost no recorded GPU time), predicted 0.5-0.8 s.
3. Per-round host overhead in Stages 1, 3 and 4 (20-32 ms/round with little recorded GPU
   time); profile per round before changing anything.
4. Cross-proof prefaulted arena for the six prepare spans, admitted only with an
   allocation-lifetime table and a falsifier of prepares under 3 s total without displacement
   into later stages; predicted 1.5-2.5 s (S2 measured first touch at 24 ms/GB GPU, 43 ms/GB
   CPU, 9 ms/GB even for a prefaulted host arena).
5. Eval-proof batching of the seven levels' stage-1/stage-2 sumchecks and ring switches under
   one transcript challenge, predicted 1.5-2 s, only with a written soundness delta and prover
   and verifier changed together, recorded in `specs/akita-metal-protocol-changes.md`.
6. SHA-2 log_K=14 bytecode address phase on the GPU without the fused Stage-1 topology,
   predicted 1.3 s on SHA-2 only.

For every candidate, before code: one mechanism, the exact charged boundary, a lower bound, a
predicted complete-proof saving, and one numerical falsifier. Then the smallest red parity or
route test, one scoped edit, focused tests, one T=2^25 sentinel, and one T=2^28 treatment on
the affected workload. Retain only a verified complete-proof improvement of at least 0.20 s
with exact CPU/Metal parity and no fallback; otherwise revert exactly and log the negative
result. Log every candidate append-only in `benchmark-runs/akita-10mhz-studies/events.jsonl`
and `analysis.md` with parent and candidate digests, command, result, and
`keep | discard | inconclusive`.

## Closed, do not reopen

Commit ring dimension, modulus profile, dense-digit or radix-4 encodings, tile prefetch or
software pipelining in the D512 panels kernel, column-major task scheduling, root
carry/sign/RNS/radix micro-variants, the CPU hybrid commit share, wider one-hot chunks, page
primers or storage-mode flips as standalone candidates, and every candidate the
protocol-preserving ledger marks closed.

## Correctness and hygiene

`cargo nextest`, never `cargo test`; `cargo fmt`; both clippy modes from `CLAUDE.md` for
touched crates. `cargo clippy -p jolt-kernels --features metal,test-utils --all-targets` has
two pre-existing errors (sequence.rs items after the test module; `optimized/spartan_outer.rs:3855`);
separate those from candidate diagnostics and fix them if the touched file is in scope.
Preserve unrelated changes and the untracked study tools in the Akita worktree
(`crates/akita-metal/examples/{sis_sweep,page_fault_bench}.rs`,
`crates/akita-planner/examples/root_geometry_query.rs`). Do not push.

## Completion

All three workloads verified at or above 5x in two order-reversed pairs each, the retained
levers documented with their measured deltas, the strategy spec's section 7 updated with the
final matrix, and a closing note stating the measured ceiling reached and which levers were
rejected. If the campaign stalls with all levers closed, stop and report; do not widen scope
into protocol redesign, which is a separate human-reviewed study.
