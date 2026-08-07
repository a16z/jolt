# Spartan shift Metal integration v2

The first production slice will wire the existing exact two-command Spartan-shift runtime to device-resident witness planes and measure the PIOP boundary. It is the shortest path to a valid result and should clear the 5x floor, but its retained GPU-active time already rules out 7x. The endpoint replaces the prefix scan with stage-1 and stage-2 partial carriers and borrows InstructionInput's midpoint UPC table; that route is designed for the 7x floor and releases the 512-MiB UPC plane before stage 3.

No verifier relation, Fiat--Shamir order, output opening, or proof encoding changes. Backend witness preparation is reported separately from the primary CPU-versus-Metal PIOP wall. If a backend-specific projection is produced after the PIOP timer starts, its wall remains in the Metal numerator.

## Comparison boundary and requirements

The frozen optimized-CPU denominator at `log_T = 26` is the complete `SpartanShift` member from `benchmark-runs/metal-piop-eval/20260806-133709-697013`:

```text
131.051624, 131.584500, 129.304918, 130.343291, 134.289502 ms
```

The median is `131.051624 ms`. The hard 5x cap is `26.210325 ms`; the 7x cap is `18.721661 ms`; the 8x continuation cap is `16.381453 ms`. Both arms include member preparation from an already-created backend witness, all 26 rounds, terminal bind, and output claims. Host Fiat--Shamir is included or excluded symmetrically. A separate result reports backend-witness preparation time and peak residency.

Requirements:

- The five output claims and their common opening point stay byte-for-byte compatible with optimized CPU.
- The host batch driver remains the sole owner of transcript absorption and challenge draws.
- Metal is selected only when every required resident allocation or carrier validates before round 0. A failure after round 0 is terminal; it cannot fall back to mutated CPU state.
- Phase A must clear 5x at `log_T = 26`. Phase B must clear 7x; if the measured roof still admits 8x, optimization continues to the 8x cap.
- No target run may hide a host upload, first-consumption wait, new command-buffer wait, or upstream-stage regression.
- The initial hybrid selector uses optimized CPU below `log_T = 25`; the final crossover comes from frozen alternating measurements.

## Exact relation, orientation, and the missing boundary cell

For cycle `j`, let

```text
outer(j)   = upc(j) + gamma*pc(j)
           + gamma^2*virtual(j) + gamma^3*first(j)
product(j) = gamma^4*(1 - noop(j)).
```

The shift summand is

```text
EqPlusOne(r_outer, j)*outer(j)
  + EqPlusOne(r_product, j)*product(j).
```

`r_outer` is product uni-skip `tau_low`; `r_product` is the ProductRemainder output point. Both have `n = log_T` big-endian coordinates. Sumcheck binds cycle variables low-to-high. If its challenges are `c_0, ..., c_(n-1)`, all five outputs open at `reverse(c_0, ..., c_(n-1))`.

Set

```text
prefix_vars = ceil(n/2),  P = 2^prefix_vars
suffix_vars = floor(n/2), H = 2^suffix_vars
j = h*P + l.
```

This is the orientation used by `EqPlusOnePrefixSuffix::new`: the low variables index its `prefix_*` tables, and `OptimizedSpartanShift` reads row `(h,l)` at `(h << prefix_vars) + l`. For `e[h] = Eq(r_hi,h)` and a current-cycle column `v`, the two exact Q tables are

```text
C[l] = sum_(h=0)^(H-1) e[h]   * v(h,l)
S[l] = sum_(h=1)^(H-1) e[h-1] * v(h,l).
```

`S` has no wrap at `h = H-1`. Direct current-cycle planes produce these tables without a boundary exception.

ProductRemainder currently carries `NextIsNoop`, so a tempting alternative is to reconstruct the current non-noop column. Write `v[j] = 1 - IsNoop(j)` and `a[j] = 1 - NextIsNoop(j)`. The modular witness guarantees

```text
a[j] = v[j+1] for j < N-1, and a[N-1] = 0.
```

Define

```text
A[l] = sum_h e[h] a(h,l)
B[l] = sum_(h=1)^(H-1) e[h-1] a(h,l).
```

Then the interior and no-wrap identities are

```text
C[l] = A[l-1]  for l > 0
S[l] = B[l-1]  for l > 0
S[0] = A[P-1].
```

The final equality is exact because `a(H-1,P-1) = 0`. The remaining cell is

```text
C[0] = e[0]*v(0,0)
     + sum_(h=0)^(H-2) e[h+1]*a(h,P-1).
```

It requires the exact current value at cycle zero and a forward-weight boundary sum. Setting `C[0]` to zero is invalid even though `prefix_0[0] = 0`: sumcheck proves a product of multilinear extensions, not the multilinear extension of their Boolean pointwise product. With a one-variable prefix, `p = [0,p1]` and `q = [q0,q1]` give `p(2)q(2) = 2*p1*(2*q1-q0)`, which depends on `q0`.

The chosen ABI therefore carries an exact current `IsNoop` bitplane. Reconstructing it from `NextIsNoop` remains an oracle test for the formulas above, not the production data path.

## Phase A: resident PIOP integration

Phase A installs a `PrepareKernel<AkitaField, SpartanShift>` adapter around the existing checked runtime in this directory. Backend witness preparation owns three device buffers:

| plane | bytes at `log_T = 26` |
|---|---:|
| UPC `u64[N]` | 536,870,912 |
| PC `u64[N]` | 536,870,912 |
| virtual, first, current-noop bitplanes | 25,165,824 |
| total | 1,098,907,648 |

The producer partitions work in aligned 32-row chunks, so one worker writes two contiguous native arrays and one complete three-mask word without atomics or a second packing pass. The first Metal consumer must see device-produced or otherwise fully resident buffers; a diagnostic host upload is not admissible evidence.

Before round 0, `solinas_spartan_shift_build_mixed_partials` and `solinas_spartan_shift_reduce_prefix` produce the four exact Q tables. The host runs the 13 prefix rounds. On receipt of `c_12`, `solinas_spartan_shift_fold_native` produces all five `H`-element dense tables, and the host runs the 13 suffix rounds and output extraction. The existing CPU prefix and suffix ladders remain authoritative; shaders never see transcript state.

The current retained evidence is useful but not promotable by itself: the two kernels used a `20.319875-ms` median of GPU-active time, and the warmed service median was `22.636833 ms` (`5.789x`). The first host-written consumption measured `79.592083 ms`. Phase A succeeds only if the first production-resident invocation, with no warm-up dispatch, has a complete member median at or below `26.210325 ms`. Its observed active time already exceeds the 7x cap, so Phase A is a validation and integration milestone, not the endpoint.

This slice changes only the Metal backend adapter, resident producer/lease, source registration, and benchmark selection. It does not require a stage-driver hook or upstream partial-carrier ABI.

## Phase B: partial carriers and residual midpoint fold

Phase B removes the full prefix scan from stage 3. Stage 1 parks eight `P`-element tables at `r_outer`: current/successor UPC, PC, virtual, and first. Stage 2 parks two `P`-element current/successor non-noop tables at `r_product`. Stage 3 combines them after drawing `gamma`:

```text
q0 = upc.C + gamma*pc.C + gamma^2*virtual.C + gamma^3*first.C
q1 = upc.S + gamma*pc.S + gamma^2*virtual.S + gamma^3*first.S
q2 = gamma^4*nonnoop.C
q3 = gamma^4*nonnoop.S.
```

The ten carrier tables occupy `1,310,720` bytes at the target. Their header binds producer stage, witness generation, device registry, row count, `P`, point digest, allocation identity, byte length, and canonical field encoding. A missing or stale carrier selects CPU before round 0.

Stage 1 already computes final openings with high/low equality weights for `r_outer`. The endpoint changes that producer to retain the selected current low-coordinate partials and compute four shifted-high successor tables while the source rows are live. A first implementation may consume the compact UPC/PC/flag projection in a dedicated carrier command under the existing opening completion. A fresh scan of the 48-byte compact plus 112-byte residual rows is forbidden: its physical traffic cannot meet the target. The endpoint should transpose/fuse the opening producer so retained partials replace work it otherwise reduces to scalars.

Stage 2's protocol-neutral hook is `MetalProductRemainderKernel::output_claims`, immediately after `opening_weights()` and before the generated driver calls `park_residue`. The current `solinas_product_remainder_openings` owns one threadgroup per high coordinate and reduces each high slice to scalar partials; it cannot emit two `P`-element tables without field atomics, `N` field partials, or a transposed replacement. The minimum safe integration reuses the exact current-noop bitplane co-packed during backend witness preparation and dispatches a low-coordinate carrier kernel in the same final-opening command buffer. It reads 8 MiB of flags rather than rescanning 2.684 GiB of 40-byte rows, adds no wait, and emits exact `C/S`. The zero-extra-pass endpoint transposes the final-opening kernel, computes its eight scalar-opening low partials and `C/S` in the same dispatch, then reduces the eight opening tables against `e_in`.

At the midpoint, only PC, virtual, first, and current noop are folded on Metal. UPC is borrowed from InstructionInput, which has the same stage-3 challenges and output point. Because the batch calls shift before InstructionInput, both kernels share a small host service. On shift round 13, the service applies `c_12` to InstructionInput, caches its raw round-13 Q evaluations, and publishes table 3 (UPC) with an ordered challenge digest. InstructionInput's later call validates the same round and challenge, consumes the cached Q evaluations with its own previous claim, and does not bind twice. This changes no transcript order. With the default `2^16` cutoff, InstructionInput is already in its CPU dense tail at this boundary.

After the midpoint fold, the primary resident state is the PC plane plus three flag planes, `562,036,736` bytes, rather than the full `1,098,907,648`-byte Phase-A source. Including ten carrier tables, peak retained handoff storage is `563,347,456` bytes, a reduction of `535,560,192` bytes. UPC may be released after the stage-1 carrier; Q tables are released at the midpoint; residual planes are released after the fold.

## Target work and wall budgets

At `N = 2^26`, `P = H = 8192`, the conservative standalone accounting for Phase B is:

| phase | useful work | logical bytes |
|---|---:|---:|
| outer eight-table carrier | `4N-2P = 268,419,072` half-width terms | 1,091,698,688 |
| product two-table carrier | at most `2N-P` selected adds | 8,781,824 |
| ten-to-four Q combine | `8P = 65,536` full products | 1,835,008 |
| prefix host ladder | `16P-24 = 131,048` full products | host |
| residual midpoint | `N = 67,108,864` half-width terms | 562,692,096 |
| suffix host ladder | `19H-19 = 155,629` full products | host |
| total | `5N-2P = 335,527,936` half-width and `352,213` full products | 1,665,007,616 kernel bytes |

The full UPC/PC/flag producer writes another `1,098,907,648` bytes. At the retained `451.702 GB/s` copy rate, that write has a `2.433-ms` optimistic floor and `3.041-ms` 80%-roof bar. The residual-only retained projection has a `1.244-ms` optimistic floor and `1.555-ms` 80%-roof bar.

Using the matched `33.168 Gterm/s` half-width control, Phase B's gross arithmetic floor is `10.116 ms`, or `12.645 ms` at 80% roof. Using the fail-closed `26.272 Gterm/s` promotion rate gives `12.771 ms`, or `15.964 ms` at 80%. The `1.665-GB` kernel traffic floor is `3.686 ms` (`4.608 ms` at 80%), so the gross model remains compute-bound.

For the primary PIOP-only comparison, reserve `2.5 ms` for Q construction, both host ladders, visibility, and service. The resulting modeled complete walls are:

| model | complete wall | ratio to CPU |
|---|---:|---:|
| matched rate, 80% roof | `15.145 ms` | `8.65x` |
| promotion-floor rate, 80% roof | `18.464 ms` | `7.10x` |

These are modeled target walls, not measurements. They exclude backend-witness preparation by the primary boundary. Adding the full producer's 80%-roof write gives `18.186 ms` at the matched rate and `21.505 ms` at the promotion-floor rate before measured extraction overhead. The all-in result must report that overhead rather than infer it from copy bandwidth.

The resident-PIOP incremental view is smaller because five successor tables replace upstream opening work. It charges about `3N = 201,326,592` half-width terms to shift: `7.587 ms` at 80% of the matched rate or `9.579 ms` at 80% of the promotion floor. Promotion still uses measured total PIOP wall; this attribution cannot hide a stage-1 or stage-2 regression.

The 7x wall leaves `16.222 ms` after the `2.5-ms` host/service reserve. Diagnostic bars are `10.2 ms` outer-carrier active, `0.5 ms` product-carrier incremental wall when fused (`1.0 ms` for the compact standalone dispatch), and `2.6 ms` residual-midpoint active. A complete wall above `26.210325 ms` kills the Metal route. A wall below 5x but above `18.721661 ms` keeps Phase B in optimization. A wall below 7x promotes it; if counters and component walls still admit `16.381453 ms`, search continues toward 8x.

## Integration, failure behavior, and validation

The implementation order is deliberately split:

1. Add the current-noop plane to the existing one-pass witness projection and expose a checked `SpartanShiftResidentRows` lease.
2. Install Phase A in `MetalBackend::with_metal_compute`, using the existing prefix/fold runtime and optimized host ladders. Shadow Q tables and all five midpoint tables before enabling output authority.
3. Run the first-consumer `log_T = 26` PIOP benchmark. Stop if the complete member misses 5x; do not tune carrier plumbing on top of a failed resident boundary.
4. Add the stage-2 product carrier at `output_claims` and park it after extraction. First validate the compact current-noop dispatch; fuse/transposed opening ownership only after parity.
5. Add the stage-1 outer carrier and switch Phase B prefix rounds to its ten-to-four Q combine. Report stage-1 before/after wall.
6. Add the shared InstructionInput midpoint service and residual four-column fold. The service is admitted only when the configured InstructionInput cutoff guarantees an exact dense table at the midpoint.
7. Freeze the hybrid cutoff from alternating `log_T = 24/25/26` measurements, then run five fresh alternating `log_T = 26` pairs and one holdout scale.

The exact source seams are `metal/backend.rs` for slot selection, `metal/spartan_outer.rs` and `solinas/outer_remainder/sequence.rs` for stage-1 carrier ownership, `metal/spartan_product.rs` and `solinas/product_remainder/shader.metal` for the product carrier, and `metal/instruction_input.rs` for the midpoint service. The verifier stage files and `jolt-sumcheck::prove_batch` remain unchanged.

Parity must cover direct dense evaluation, all four Q tables, every round polynomial, all five outputs, transcript bytes, clear and ZK proof bytes, and verifier acceptance. Boundary cases include odd/even logs; `l = 0`, `l = P-1`, and the final global row; cycle-zero noop; all-noop/all-nonnoop masks; gamma `0`, `1`, and `p-1`; `u64::MAX` PC values; wrong point/challenge digests; stale generations; and foreign or duplicate allocation identities. Performance capture must report first-consumer wall, GPU active time, command-buffer gaps, allocations/uploads/waits, register allocation, spills/local memory, resident SIMD groups, active cores, achieved terms/s, and achieved bytes/s.

Unverified facts are the compiler's physical register allocation for the carrier kernels, the stage-1 transposed-opening delta, the exact first-consumer wall of device-produced planes, and the final crossover. No performance claim based on the v2 roof is promotable until those measurements exist.
