# Akita PIOP kernel CPU-relative 5x/10x goal

Status: complete as of 2026-08-28. This replaced complete-proof time as the search
evaluator for PIOP work. The earlier E2E ledgers remain evidence, not the
optimization loop.

## Decision

Optimize one logical PIOP kernel at a time against its matched **optimized CPU
implementation**. Every scored kernel must reach at least 5x. Once a kernel reaches
5x, continue toward 10x only when the pre-code model still gives a clear route. At
10x, stop. If 5x to 10x is not obvious, record the ceiling and move on.

No permutation argument, changed witness relation, changed sumcheck schedule, or
other major protocol change is admitted. Exact round polynomials, claims, transcript
order, proof bytes, and verifier behavior remain fixed.

## Score and scope

For kernel `k`, workload `w`, and scale `n`, the only speedup is

```text
R(k,w,n) = optimized_CPU_service_wall(k,w,n) / Metal_service_wall(k,w,n).
```

The two arms use the same logical boundary, input, output, protocol, build mode, and
machine. The boundary charges mandatory preparation, allocation, conversion,
submission, synchronization, readback, finish, and output-claim work. Asynchronous
work is charged to the kernel that requires it; overlap cannot hide service.

BTreeMap at `T=2^28` with target trace size 150,000,000 is the fixed discovery
sentinel. A kernel's hard score is the minimum ratio over BTreeMap and any materially
active Fibonacci or SHA-2-chain T28 shape using the same route. A material arm has at
least 25 ms of optimized CPU work. Inactive/no-op spans are unscored until exercised
by a real fixture. T25 and T20 are crossover guards: they need not reach 5x, but a
candidate may not regress the accepted Metal wall by more than 3%; CPU fallback is
valid there when explicit and measured.

## Analysis gate

No kernel code is written until its card records:

1. the exact input/output and charged timing boundary;
2. compulsory bytes, field operations, launches, serial dependencies, and live
   memory;
3. sustained machine bandwidth and arithmetic rates, plus Metal codegen evidence;
4. a bottomed-out latency floor and `ceiling = CPU wall / floor`;
5. named adjustment candidates, each with a numerical falsifier; and
6. exact parity, lower-scale, memory, and fallback checks.

The floor is the critical composition of traffic, compute, launch/synchronization,
and serialization bounds. Terms that cannot overlap are added; competing lower
bounds are maximized. Peak hardware specifications are not evidence.

Ceiling below 5x is a campaign blocker for that kernel and must be reported rather
than disguised by a boundary change. Ceiling from 5x to 10x sets a 5x target. Ceiling
at least 10x sets a 10x target only when a named mechanism predicts a wall at or
below `CPU/10`, fits memory, and has no unpriced dependency. The model may change only
after a logged measurement falsifies an assumption.

## Kernel loop

Each kernel moves through `inventory -> baseline_frozen -> modeled -> optimize_5 ->
validate_5 -> optimize_10? -> done_5|done_10`. Only one kernel and one candidate are
active. A candidate is a revertible transaction: preregister, test exactness, inspect
codegen, measure, then keep or restore. Search is greedy against the retained parent.

Work proceeds in tranches of at most six candidates under one model. Exhausting a
tranche triggers a model review, not a weaker target. Negative results remain in the
append-only event log. Shared wins update the baselines of every affected kernel,
but do not receive credit twice.

After the transition seed, choose the below-5x kernel with the largest
`Metal wall - CPU wall/5`; break ties by lower ratio, then larger CPU wall. This
targets the most expensive missing 5x work without letting tiny ratios dominate.

## Measurement and promotion

Search uses one warm-up and at least three measurements per arm. Promotion uses a
fresh accepted build, one warm-up, and seven order-balanced measurements per arm at
each required sentinel. Report every raw millisecond value. The conservative ratio
is `CPU p25 / Metal p75`; it must be at least 5.0 or 10.0. Search aims for 5.25x and
10.5x to leave validation margin.

Every promotion also requires exact CPU/Metal parity for every round polynomial and
terminal claim, proof verification, no silent T28 fallback, no displaced work beyond
the charged boundary, peak RSS at most 90 GiB, no swap growth, and passing T25/T20
guards. A CPU baseline is frozen only with its source/evaluator/build/workload/machine
identity and is invalidated when any of those or shared CPU code changes.

Full proofs are correctness and integration guards, not the search metric. Run the
smallest verifying proof before promotion, one relevant T28 proof after promotion,
and the three-workload E2E matrix after three kernel promotions or at final
validation. Do not launch broad E2E runs to choose routine candidates.

## Initial kernel and campaign artifacts

Start with `OuterRemainder`. Its accepted A2 member observations are 0.875 and 0.896
seconds against a 3.760-second optimized-CPU localization trace. The old 4.20--4.30x
figure excluded 0.833--0.899 seconds of mandatory Metal storage preparation; its
provisional charged ratio is therefore only 2.09--2.20x before shared carrier work
is assigned. The 5x wall is 0.752 seconds and the 10x wall is 0.376 seconds. First
freeze a matched benchmark, allocate shared work once, and complete the floor model.

The durable state is
[`run.json`](../benchmark-runs/akita-piop-kernel-cpu-5x/run.json), the evolving
[`analysis.md`](../benchmark-runs/akita-piop-kernel-cpu-5x/analysis.md), and the
append-only `events.jsonl`. The prior
[`scoreboard`](akita-piop-kernel-scoreboard.md) is the inventory seed and must be
refreshed under the accepted A2 source before it determines the post-seed order.

## Requirements ledger

| Kind | Item |
|---|---|
| Fact | The denominator is the optimized CPU implementation, never old Metal or E2E time. |
| Requirement | Every materially active PIOP kernel clears a conservative 5x ratio. |
| Requirement | Pursue 10x only when the analyzed path remains clear; stop at 10x. |
| Requirement | Preserve decent lower-scale behavior with explicit crossover guards. |
| Constraint | Stay protocol-preserving and exact. |
| Decision | BTreeMap T28 is discovery; Fibonacci/SHA-2 are transfer sentinels. |
| Decision | Kernel-local service time chooses candidates; E2E only validates integration. |
| Assumption | A dedicated matched harness can expose every logical member boundary. |
| Resolution | If it cannot, build the smallest per-member adapter before optimization. |

## Completion

The refreshed material inventory closed with K010, the last registered positive
5x gap. K001 through K010 are recorded as `done_5` or `done_10` in `run.json`; every
promotion used matched optimized-CPU service as the numerator. K010's final
three-workload conservative ratios are 155.425x for Fibonacci, 5.918x for
SHA-2-chain, and 6.859x for BTreeMap at an actual `T = 2^28` domain. Its T20 Metal
path is exact, while production deliberately selects optimized CPU below T25.

The final modular integration gates passed 21/21 clear-mode tests and 14/14 ZK
tests. The append-only event log contains the raw artifacts, hashes, rejected
candidates, analytical stop decisions, and the two unrelated test-target build
limitations encountered during focused validation.
