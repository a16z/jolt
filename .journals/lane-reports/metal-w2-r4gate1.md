# Metal W2 radix-4 Gate 1 — address-first phase objective

## Verdict

**FAIL — kill the address-first probe entirely.** At `2^24`, binary is
`6.302900917 s` and radix-4 is `7.103973750 s`; both exceed the `0.15 s`
kill line by >42×, and radix-4 is `1.127096×` binary.

One release-profile timed run per arm/scale; no rerun was warranted because
both `2^24` arms miss the absolute gate by orders of magnitude and the
relative gate agrees at both scales.

| cycles | binary 7-message | radix-4 `[P4,P4,P4,S]` | radix-4 / binary |
|---:|---:|---:|---:|
| `2^22` | `1.099370792 s` | `1.466777167 s` | `1.334197×` |
| `2^24` | `6.302900917 s` | `7.103973750 s` | `1.127096×` |

Measurement host: Apple M5 Max, 18 logical CPUs, 128 GiB, arm64; Rust
`1.95.0`; Rayon default pool. Setup and the mandatory small-scale oracle run
were outside each timed interval. Each interval covers every address-round
message pass, run-list state fold, address-weight bind, interpolation,
round-sum check, and claim update.

## Level-0 Val temporal convention (normative)

`Val(k,j)` is the **pre-state of cycle `j`**:

```text
cycle j:  materialize Val(*,j)  -> capture/read pre-state -> execute write
cycle j+1:                                      first row containing post-state
```

- The live witness initializes every register to zero, emits `state[k]` into
  `Val(k,j)`, then applies `rd.post_value` after row `j`
  ([`registers.rs:27-43`](../../crates/jolt-witness/src/backend/trace/registers.rs)).
  Therefore a write at `j` takes effect in `Val(k,j+1)`, not `Val(k,j)`.
- The tracer captures `rs1`, `rs2`, and `rd.pre` before execution and only
  captures `rd.post` afterward
  ([`instruction/mod.rs:401-413`](../../tracer/src/instruction/mod.rs),
  [`format_r.rs:77-85`](../../tracer/src/instruction/format/format_r.rs)).
  Same-cycle reads—including `rs1/rs2 == rd`—observe the pre-state.
- Cycle 0 starts with a run `(0, 0)` for every register. A cycle-0 write adds
  its post-state breakpoint at cycle 1.
- A final-cycle write contributes its row's `Wa`/`RdInc`, but its post-state
  would start at `T`; no `Val(*,T)` row exists, so no run is emitted.
- x0 reads as zero and `write_register(0, ...)` discards the value
  ([`cpu.rs:421-440`](../../tracer/src/emulator/cpu.rs)); ordinary `rd=x0`
  instructions are also rewritten before the proof trace
  ([`instruction/mod.rs:591-613`](../../tracer/src/instruction/mod.rs)). The
  objective discards attempted x0 history events.

Both level-0 run lists and the independent dense `128×T` reference implement
that ordering directly. Coincident child breakpoints are applied together
before one parent run is emitted.

## Objective and parity inventory

- `RegistersAddressPhase` uses cycle-rowed three-lane CSR
  (`Wa/Rs1Ra/Rs2Ra`), 128 registers, hot/cold writes (80% to eight hot
  registers), x0/no-write holes, never-written registers, and per-register
  post-state run histories.
- Binary: seven degree-2 messages, three evaluation points, 2-child run
  merges, weights `(1-r,r)`. Radix-4: three degree-6 messages over
  `[-3..3]`, 4-child merges using the `D={-1,0,1,2}` Lagrange weights and
  functional `(4,2,6,8,18,32,66)`, then one binary message.
- Dense parity at `2^12`: each arm independently matches a materialized
  `128×T` Val table for final bound Val, address weights, and final claim.
  A domain-point schedule additionally makes both arms' final tables/claim
  identical.
- Boundary coverage: cycle-0 write; final-cycle write; same-cycle read/write
  on different registers in one 4-group; coincident child writes; interleaved
  group writes; never-written registers; attempted x0 write; accesses across
  groups with quiet-sibling Val cross terms.
- After every fold, 64 deterministic-random `(node,cycle)` probes compare the
  run-list value to the dense partially-bound table, pinning piecewise
  constancy beyond final-claim parity.

## Verification

- `cargo fmt -q --message-format=short` — pass.
- `cargo clippy -q --message-format=short -p jolt-eval --all-targets -- -D warnings` — pass.
- Targeted `cargo nextest run -p jolt-eval --cargo-quiet` — 2 passed:
  dense temporal/fold parity and the degree-6 D-sum functional.
- `cargo bench -p jolt-eval --bench registers_address_first_phase -- --gate-run` — pass; timing table above.
- No end-to-end prover run. No production prover, verifier, protocol, sumcheck,
  or kernel path changed.
