# Wrapper draft PR — R1CS + Spartan final measurements

Source: non-ignored `wrap_real_t1_r::real_wrapper_round_trip_and_tampers`, cached
fibonacci `2^18` proof, Mac mini M4, 10 Rayon threads. Default `k=16`;
`WRAP_K=32` selects the comparison.

Both current columns were remeasured in PERF-5 lane 7 after the MSM and
stream-tail lanes landed. Historical lane comparisons remain labeled below.
Selecting k=16 saves 3.278 s online in this pair for 288 B payload and
143,924 modeled gas (3.0%); payload matches the pre-four-ary k=32 proof at
7,392 B. The lane-5b pair used for the decision saved 2.693 s.

## Link coverage

| source | destination | count | binding |
|---|---|---:|---|
| T1 squeeze outputs | Spartan W challenge cells | 376 | CopyLink |
| T1 pre-final-squeeze Fr words | Spartan W proof-value cells | 1,200 | CopyLink |
| T1 element bytes | T2 input chunks/flags | 45,152 B / 1,526 rows | CopyLink |
| seven statement fields + one | R1CS public segment | 8 Fr | Spartan `z` assignment |
| T1 state/tail | wrapper statement suffix | 54 B / 4 Fr | transcript input |
| Spartan W Dory cells | T2 scalar input | 173 occurrences | occurrence-weighted link |
| program/profile digest | wrapper key | 32 B | verifier-key check |

Ten CopyLinks contribute 100 terms, 120 pinned columns, and 20 sparse helper columns. The
element links include compressed G1/G2 sign flags, zero high bits, GT limb order, and the
profile-derived commitment permutation. The last 23 absorbed Fr values follow the final native
squeeze and do not affect a Fiat-Shamir challenge. `Chi(sigma)`, `S1Acc`, and `S2Acc` stay internal
to R and do not enter the 173-scalar link.

## Fiat-Shamir challenge schedule

| committed phase | challenges drawn after commitment | k=16 | k=32 |
|---|---|---:|---:|
| phase 1a: T1 + W | 38 T1 randomizers, `theta` | 39 | 39 |
| T2 phase 1b | `xi`, `alpha`, ten CopyLink `(beta, gamma)` pairs, scalar-link `rho` | 23 | 23 |
| T2 phase 2a | `fp_root` | 1 | 1 |
| T2 phase 2b | `beta`, `fp_combine`, `copy_root` | 3 | 3 |
| T2 phase 2c + CopyLink helpers | T2 row/member challenges; ten CopyLink points and weights | 232 | 232 |

The final-phase count remains 232 at both packings: 22 T2 challenges plus
ten CopyLink points and three weights per link (`10 × (18 + 3)`). Group
counts do not enter this phase count.

## Proof bytes

Measured four-ary opening layout; T2 uses lane 5a's `s=4`.

| section | k=16 | k=32 |
|---|---:|---:|
| phase 1a wire commitments | 672 | 384 |
| T2 phase 1b wire commitments | 160 | 96 |
| T2 phase 2a wire commitments | 128 | 64 |
| T2 phase 2b wire commitments | 32 | 32 |
| T2 phase 2c + CopyLink helpers | 64 | 32 |
| Spartan outer, 13 committed rounds | 864 | 864 |
| Spartan inner, 13 clear rounds | 832 | 832 |
| stage A, 18 committed rounds | 1,184 | 1,184 |
| term stage, 9 committed rounds | 608 | 608 |
| shared BDFG/degree-shift proof | 96 | 96 |
| five factor evaluations | 160 | 160 |
| stage B clear rounds | 640 | 640 |
| reduced claims (opening + Az/Bz/Cz/W) | 160 | 160 |
| HyperKZG opening | 1,792 | 1,952 |
| **proof payload** | **7,392** | **7,104** |
| **bincode proof** | **7,533** | **7,232** |
| statement, 11 Fr | 352 | 352 |
| **payload + statement** | **7,744** | **7,456** |
| **bincode + statement** | **7,885** | **7,584** |

## Geometry

| item | value |
|---|---:|
| R1CS constraints / variables | 5,323 / 6,831 |
| public / private variables | 7 / 6,823 |
| outer / inner rounds | 13 / 13 |
| common row rounds | 18 |
| matrix nonzeros | 35,346 |
| native matrix-evaluation Fr multiplications | 87,081 |
| T2 rows | 201,575 |
| total terms / term rounds | 500 / 9 |
| T1 / CopyLink / T2 / scalar / carry terms | 232 / 100 / 166 / 1 / 1 |

| groups | k=16 | k=32 |
|---|---:|---:|
| proof wire / key / full | 33 / 11 / 44 | 19 / 7 / 26 |
| T1 sent / VK | 20 / 2 | 11 / 2 |
| Spartan W | 1 | 1 |
| CopyLink VK | 8 | 4 |
| T2 1b / 2a / 2b / 2c | 5 / 4 / 1 / 2 | 3 / 2 / 1 / 2 |
| final helper groups | 0 | 0 |
| opening variables / committed folds | 22 / 10 | 23 / 11 |
| transmitted opening Fr evaluations | 45 | 49 |

## Timing

| phase (ms) | k=16 | k=32 |
|---|---:|---:|
| deterministic SRS setup (offline) | 3,780 | 7,589 |
| key/profile (offline) | 165 | 190 |
| offline key commitments | 347 | 403 |
| wrapper preparation | 457 | 439 |
| T1/R stream adaptation | 66 | 72 |
| T2 adaptation | 642 | 637 |
| phase 1a commitment | 776 | 793 |
| T2 phase 1b commitment | 745 | 807 |
| T2 phase 2a commitment | 5,446 | 5,354 |
| T2 phase 2b commitment | 64 | 95 |
| CopyLink helpers | 36 | 34 |
| T2 phase 2c + helpers | 295 | 338 |
| T2 finish | 217 | 327 |
| member construction | 802 | 759 |
| proof stages/opening | 6,954 | 10,124 |
| HyperKZG fold commitments, included above | 733.848 | 1,332.783 |
| HyperKZG quotient MSM, included above | 1,869.708 | 3,723.125 |
| HyperKZG opening total, included above | 2,794.118 | 5,644.792 |
| **honest online total** | **16,507** | **19,785** |
| verifier (outside online clock) | 27 | 25 |

The prebuilt k=16/k=32 gates held the mutex and entered at one-minute loads
2.47/1.41. Honest-clock start/end loads were 3.16/5.12 and 2.25/4.66;
process CPU was 136.600/158.380 s. No competing compiler or test job was
observed in either accepted window. Full loads, commands, and logs are in
`lanes/perf5-lane7.md`; the matched binary-fold baseline (22.636 s online,
8.554 s opening) remains in `lanes/perf5-lane5b.md`.

### PERF-5 lane 3 after lane 2

| phase (ms) | lane 2 | lane 2 + lane 3 |
|---|---:|---:|
| wrapper preparation | 564 | 454 |
| T2 adaptation | 1,426 | 640 |
| T2 finish | 598 | 457 |
| T2 member setup | ~691 | 36 |
| T2 stage-A row member | 2,854 | 1,910 |
| all member constructors | 1,440 | 785 |
| proof stages/opening | 16,169 | 15,968 |
| **honest online total** | **29,802** | **27,792** |

Final command-start load: `3.10 / 6.82 / 10.49`; honest-clock start/end:
`8.08 / 7.56 / 10.41` -> `9.10 / 7.84 / 10.42`. Process CPU was 219.040 s.
Proof remained 7,392 B payload / 7,530 B bincode / 352 B statement; verifier
cost remained 4,868,177 gas.

## Verifier cost

| operation | k=16 | k=32 |
|---|---:|---:|
| ecMul | 233 | 216 |
| ecAdd | 233 | 216 |
| pairing pairs | 8 | 8 |
| Fr multiplications | 123,144 | 123,121 |
| Fr inversions | 8 | 8 |
| Keccak | 852 | 839 |
| **N4 gas model** | **4,944,149** | **4,800,225** |

The same observer counts transcript replay, native sparse-matrix evaluation, sumchecks, links,
term reduction, and the final opening. The native sparse-matrix block at both packings
accounts for 87,081 Fr multiplications over 35,346 nonzeros. Against the integrated binary-fold k=32
baseline, four-ary folding removes 10 ecMul and 108 Fr multiplications, lowering
modeled gas by 90,420. The sparse quintic divisor still needs four opening
pairing pairs, so the wrapper total remains eight.

## Tamper matrix

The real gate mutates every serialized field independently and requires rejection:

- every wire commitment, including W and all T2 phases;
- Spartan outer commitments/claims/`S(0)`, inner clear coefficients, and Az/Bz/Cz/W claims;
- every stage-A/term committed round and every stage-B clear coefficient;
- shared BDFG shifted commitment, quotient, and evaluation witness;
- every factor evaluation and final HyperKZG fold commitment/evaluation/quotient field;
- direct T2 window/sign/psi/digit/input-row mutations, an absorbed-Fr W row, T2 VK pin,
  statement mismatch, a fixed-challenge T1 initial-state claim change, and program/profile
  mismatch.

The permanent scalar contract pins the 173-wire order and occurrence-weight formula.
Lane 5b's wrapper + HyperKZG suites passed 94/94. Lane 7's feature-enabled real gates passed
1/1 at both k=16 and k=32, including all opening fields. The PCS tests reject
inconsistent folds with valid standalone KZG openings and corrupt G1/G2 VK powers.

### PERF-5 lane 5a after lane 3

T2 grouped-inverse LogUp uses `s = 4`: 17 helpers, 62 phase-2a full-Fr
columns, and two 32-column groups with two padding slots. The post-rebase real
gate was prebuilt before the measurement lock; command-start load was 2.84 and
the honest-clock load was 4.34 -> 5.90.

| measurement | s=3 | s=4 |
|---|---:|---:|
| T2 phase-2a commitment | 7,140 ms | 6,233 ms |
| T2 stage-A row member | 1,910 ms | 2,637 ms |
| honest online wall | 27,792 ms | 26,072 ms |
| process CPU | 219.040 s | 217.560 s |
| payload / bincode / statement | 7,392 / 7,530 / 352 B | 7,392 / 7,529 / 352 B |
| proof wire / key / full groups | 20 / 7 / 27 | 19 / 7 / 26 |
| T2 1b / 2a / 2b / 2c groups | 3 / 3 / 1 / 2 | 3 / 2 / 1 / 2 |
| total terms / term rounds | 510 / 9 | 500 / 9 |
| ecMul / ecAdd | 227 / 226 | 226 / 225 |
| Fr multiplications / inversions | 121,705 / 10 | 123,229 / 10 |
| Keccak / pairing pairs | 846 / 8 | 848 / 8 |
| N4 gas | 4,868,177 | 4,890,645 |

Sweep decision: `s = 6` models at 26.368 s and +64 B; `s = 9` at
27.452 s and +160 B. Both exceed the +32 B cap, so `s = 4` remains selected.
The model and the correctness-run ratios are in `lanes/perf5-lane5a.md`.

### PERF-5 lane 6 after lane 5a

Proof bytes are identical to `dbe2a2f9e` on both the synthetic unit fixture
and the complete cached real fixture. Payload/bincode/statement stay
**7,392 / 7,529 / 352 B**, and verifier cost stays **4,890,645 gas**.

| phase | current canonical baseline | lane 6 |
|---|---:|---:|
| column evaluations at `r_A` | 371.557 ms | 1.770 ms |
| packed RLC | 659.126 ms | 132.695 ms |
| T2 member setup | 33 ms | 19 ms |
| all member constructors | 806 ms | 792 ms |
| proof stages/opening | 15,227 ms | 14,473 ms |
| **honest online wall** | **26.025 s idle rerun; 26.072 s prior landing** | **25.200 s** |

Tail work saves **0.896 s**, with another **14 ms** in T2 setup. The paired
online-wall saving is **0.825 s**; process CPU falls **218.500→211.700 s**.
All 598 live columns are supplied by stage-A bindings except six typed T1 VK
evaluations; 234 physical padding slots are zero. RLC uses typed row blocks.

The successful gate held the mutex at 22:03:45–22:04:26 ET after prebuild.
Command-start loads were **2.41 / 1.90** for baseline/lane 6; online-start/end
loads were **3.88→6.19 / 3.22→5.73**. No compiler overlap was observed.
Two earlier contended candidate walls (30.281 / 33.122 s) are excluded.
Fmt, feature-enabled all-target clippy/check, 64/64 tests before and after
rebase, and the real tamper gate passed; see `lanes/perf5-lane6.md`.

### PERF-5 lane 4 after lanes 3/5a

Same base `2d8055c7f`, identical fixture and protocol. Hybrid bucket
accumulation replaces the whole-MSM skew fallback; large, unskewed u16/u32
inputs use 16-bit affine windows. Ten threads remain fastest in the sweep.

| measurement | control | hybrid |
|---|---:|---:|
| phase 1a commitment | 877 ms | 798 ms |
| T2 phase 1b commitment | 1,086 ms | 834 ms |
| T2 phase 2a commitment | 6,318 ms | 5,214 ms |
| fold commitments | 5.924477 s | 4.083873 s |
| fold us/point | 0.706253 | 0.486836 |
| quotient MSM | 3.718519 s | 3.758710 s |
| quotient us/point | 0.443282 | 0.448073 |
| proof stages/opening | 15,313 ms | 13,451 ms |
| **honest online total** | **26,314 ms** | **22,959 ms** |
| process CPU | 218.500 s | 191.250 s |
| CPU / wall | 8.304 | 8.330 |
| payload / bincode / statement | 7,392 / 7,529 / 352 B | unchanged |
| N4 gas | 4,890,645 | unchanged |

The mutex covered both commands. Command-start loads were 3.12 and 3.80;
honest-clock loads were 4.18 -> 6.11 and 4.75 -> 6.22. Both gates rejected
every tamper. Temporary MSM timers were removed after this comparison.
Phase 2a passed 5.4 s in that comparison, but a clean repeat took 5.469 s;
the target is not consistent across runs. Fold, quotient, and uniform
full-Fr targets remain unmet. The 3.355 s same-base online win was approved.
Rates, rejected candidates, final gates: `lanes/perf5-lane4.md`.

Final integration on `a43da7d18` (including lane 6): **22.410 s online**,
185.430 CPU s, CPU/wall 8.274; preparation 434 ms, T2 adaptation 652 ms,
phase 1a/1b/2a **779/868/5,424 ms**, all members 786 ms, proof 12,679 ms.
Command-start load 3.01; honest load 3.52 -> 5.29. Bytes and all verifier
counts remain unchanged. Post-rebase check/fmt/clippy, 234/234 unit tests,
and the locked real tamper gate passed. An earlier contended 30.056 s
integrated run is excluded from idle results. The integrated result includes
lane 6; it is not used to calculate the paired MSM saving above.
