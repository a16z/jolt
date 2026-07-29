# Akita Fp128 PIOP kernel experiment

Date: 2026-07-29 EDT

Kernel parent: `2f9f134f641f1025f6f7edeb8169164307611ef6`

Kernel infrastructure: `410ea59fa`

## Result

Keep the one-fold Solinas accumulator for multi-lane kernels, and route the
D4 InstructionRa sum-of-products through it. Do not replace the existing raw
accumulator globally.

The candidate stores each Fp128 product after the first pseudo-Mersenne fold
in two `u128` slots. The current raw accumulator uses five `u128` slots. The
smaller representation adds two multiplications by the Solinas constant per
product, but reduces live state and spills when several output lanes are
accumulated together.

## Frozen kernel screen

| Kernel | Working set | Akita raw | Akita eager | Akita multi | BN254 current | Multi vs raw |
|---|---|---:|---:|---:|---:|---:|
| Dot product, one lane | cache | 4.945 us | 11.081 us | 5.211 us | 19.667 us | +5.38% |
| Dot product, one lane | stream | 1.279 ms | 2.925 ms | 1.388 ms | 4.702 ms | +8.53% |
| Booleanity, two lanes | cache | 31.362 us | 55.705 us | 29.355 us | 106.20 us | -6.40% |
| Booleanity, two lanes | stream | 4.015 ms | 7.210 ms | 3.737 ms | 17.775 ms | -6.90% |
| D4 product sum, four lanes | cache | 31.176 us | 33.536 us | 26.604 us | 142.08 us | -14.67% |
| D4 product sum, four lanes | stream | 4.002 ms | 4.355 ms | 3.400 ms | 23.425 ms | -15.03% |

The generic `JoltField::MultiProductAccum` rerun preserved the D4 result:
15.51% faster in cache and 15.35% faster in the streaming working set.
BN254 maps the new hook to its existing accumulator, so its arithmetic path
does not change. A direct BN254 raw-versus-hook guard measured the hook 1.53%
faster in cache and 0.87% faster in the streaming set; both clear the
predeclared 3% no-regression guard.

Generated AArch64 code supports the register-pressure explanation:

| D4 codegen metric | Raw | Multi |
|---|---:|---:|
| Stack frame | 992 B | 432 B |
| Stack references | 172 | 61 |
| Assembly lines | 1,552 | 1,436 |

The stack-free one-lane control regressed, which is why the field now exposes
a separate multi-lane hook instead of changing `UnreducedProductAccum`.

## PIOP transfer

The held-out `2^22` Akita proof used the real
`InstructionRaSumcheckProver::compute_message` path:

| Metric | Parent | Candidate | Change |
|---|---:|---:|---:|
| InstructionRa compute messages, 22 rounds | 167.368 ms | 145.825 ms | -12.87% |
| Whole `prove_packed` | 5.23 s | 4.93 s | -0.30 s |

Both proofs verified. Only the 21.543 ms named-span reduction is attributable
to the candidate; unrelated spans account for most of the whole-prover
difference.

The retained `2^26` parent trace spends 1.450641 s in the same named span.
Applying the measured transfer ratio projects a 0.187 s saving. This is below
the contract's one-second gate, so no new full `2^26` proof was run.

## Correctness and validation

- Random differential accumulation against eager field arithmetic passed.
- A maximum-value synthetic accumulation covered `2^26` terms.
- Unequal partial-accumulator merges passed.
- The new split-equality fold matched the current fold exactly.
- D4 sum-of-products matched the sum of individual products for Akita and
  BN254 across successive binding rounds.
- Natural K16, forced K256, and committed-program Akita muldiv proofs passed.
- Dory muldiv passed in both `host` and `host,zk` modes.
- All-target `jolt-prover-legacy` clippy passed in `host` and `host,zk` modes
  with warnings denied.
- Workspace-wide clippy was attempted but is blocked by the unrelated,
  untracked `crates/jolt-akita/tests/schedule_probe.rs`, whose diagnostic
  `println!` and `unwrap()` calls violate workspace lint policy.

## Traces

- Parent:
  `benchmark-runs/perfetto_traces/akita-piop-multi-parent-2e22.json`

  SHA-256:
  `5c7a58fb2c152e01e95f85bb102c47afd67f9f929d33b97502eead54eedcd548`
- Candidate:
  `benchmark-runs/perfetto_traces/akita-piop-multi-candidate-2e22.json`

  SHA-256:
  `2d47d6d06375ec3cf8dc9ad74918e4390f6496ba677fa07011958e0b9178a5d9`

The append-only measurements and decisions are in `events.jsonl`.
