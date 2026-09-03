# PERF-5 — curve-wrapper prover-time campaign

Date: 2026-09-03. Base: `f4fbd2ee9`. Machine: Mac mini M4, 10 Rayon threads,
CPU only. Fixture: `fibonacci_2_18_blake3.bin`. Default `k = 32`, so one packed
polynomial has `N = 32 * 2^18 = 2^23` coefficients. `M` = measured; `E` =
estimated from a measured rate. Local savings overlap unless a cumulative row says otherwise.

## Fixed protocol envelope

- Inner PCS Dory; wrapper PCS HyperKZG; Blake-family transcript.
- R is R1CS proved by Spartan; verifier evaluates its sparse matrices natively.
- The deferred Dory check remains a committed byte/limb table proved by sumcheck.
- CPU only. Proof bytes and verifier gas are charged for every change.

## Verdict

| question | answer |
|---|---|
| Honest online baseline at `k=32` | **~40.3 s M**. The printed phase sum is 40.862 s, but its 2.778 s `adapt_r` bucket contains ~2.7 s of test-only key/reference work while 2.247 s of real T2 finish/member construction is unprinted. |
| `k=32`, 7,488-byte payload unchanged | **~22 s E, range 19–25 s** after the zero-byte lanes. The main residuals are phase-2a full-width commitments, T2 stage A, and a `2^23` quotient MSM. |
| `k=32`, byte-changing protocol work allowed | **~18–20 s E**, about **7,104 B payload** after contiguous fixed columns, final-phase slot filling, `s=4`, and a 4-ary fold. |
| Cheapest permitted point near the target | **`k=8`, ~10 s E (8–12 s), ~8,032 B payload, ~5.2 M gas E**. This assumes the high-risk virtual-operand and global CopyLink work succeeds. |
| `~2 s` configuration | **None under the fixed envelope.** An aggressive `k=8` floor is **>=5.2 s E**; `k=32` HyperKZG alone floors at **~3.6 s E** after a 4-ary fold and a 0.30 us/point MSM. |

The 2 s target is therefore a measured blocker, not a lane acceptance target. The campaign can
still cut about 2x with small changes and about 4x with the protocol work below.

## 1. Idle-machine breakdown

One run only. Command:

```text
CARGO_TARGET_DIR=/Volumes/Dev/target/perf5-plan cargo nextest run \
  -p jolt-wrapper --features prover-fixtures --cargo-quiet real_wrapper --no-capture
```

Start `uptime`: load `2.98 / 10.24 / 14.47`; no competing Cargo process. The test passed in
59.45 s including SRS/key construction, negative-key construction, and tamper checks. All values
below precede the negative-key and tamper work.

### Preparation, adaptation, commitments

| phase | ms M | attribution |
|---|---:|---|
| deterministic SRS | 7,867 | offline |
| hash-key/profile | 409 | offline; the test builds the hash key twice |
| wrapper preparation | **595** | relation 7; relation witness 143; relation check <1; second native transcript replay 129; T1 hash table 315 |
| printed `adapt_r` | **2,778** | mislabeled: Hash columns 5; full reference T2 build 1,505; verifier-key build/commit 1,034; duplicate hash-key work and glue are the balance |
| genuine R relation work | **7 build + 143 witness** | already inside preparation; R is not a 2.8 s stage |
| T2 adaptation | **1,353** | adapter <1; layout 84; program evaluation 163; `Columns::generate` 1,106 |
| T2 finish | **481** | unprinted by the permanent gate |
| member construction | **1,766** | unprinted; T1 members, ten dense CopyLink provers, 160-column T2 row matrix, scalar member |
| fixed-key commitments | 1,034 | offline, also included in printed `adapt_r` |
| phase 1a commitment | **1,783** | T1 + W + T1/CopyLink fixed groups |
| T2 phase 1b commitment | **1,148** | chunks/digits/multiplicities/sign |
| T2 phase 2a commitment | **7,435** | 44 operands + 22 range helpers + inverse, padded to 96 Fr columns |
| T2 phase 2b commitment | **101** | two fingerprints, padded to 32 columns |
| CopyLink helper build | **2,890** | ten dense scans, value clones, inversions, checks; 20 helper columns |
| T2 phase 2c + VK + CopyLink helper commitment | **348** | three packed groups |

At this commit the real relation is **5,323 constraints / 6,831 variables / 35,346 nonzeros**,
not the older table's 5,254 / 6,761 / 34,945. The origin merge added the
`VirtualXORROTL1` relation rows. Proof size is unchanged.

### Proof and verifier

| phase | ms M | detail |
|---|---:|---|
| Spartan preparation | 1 | assignment/check/transcript prefix |
| Spartan outer | **6** | 13 committed rounds |
| Spartan inner preparation / inner | **1 / <1** | 13 clear rounds |
| stage A | **7,095** | member split below |
| column evaluations at `r_A` | **1,147** | all 34 packed groups and padding |
| term export / construction | **9 / <1** | 511 terms including carry; 1,170 factor occurrences; max 4 factors |
| term rounds / shared opening | **3 / 2** | nine committed rounds |
| stage B / reduction | **<1 / <1** | eleven clear rounds |
| packed RLC | **998** | 34 group polynomials over `2^23` points |
| redundant claimed-point evaluation | **509** | a second full pass over the RLC polynomial |
| HyperKZG fold materialization | 46 | binary folds |
| HyperKZG fold commitments | **8,250** | `N-2` full-width points; nested Rayon issue |
| three-point evaluations | 133 | parallel Horner |
| KZG batch RLC / cubic division | **46 / 338** | division is already parallel |
| quotient MSM | **3,799** | `N-3` full-width points |
| HyperKZG open total | **12,616** | fold + batch open |
| printed proof total | **22,406** | internal rows above sum to within timer noise |
| verifier | **25** | 234 ecMul, 233 ecAdd, 8 pairing pairs, 127,884 Fr mul, 10 inversions, 857 Keccak |
| verifier gas | **5,048,805 M** | includes 87,081 native matrix Fr multiplications |

Proof wire: **7,488 B payload / 7,628 B bincode / 352 B statement**.

### Stage A member split

| member | ms M |
|---|---:|
| T1 row / wiring | 970 / 341 |
| ten CopyLinks | **2,830 total**; individual 271–304 |
| T2 row | **2,854** |
| Dory scalar link / W carry | 77 / 9 |
| scheduler/round-commit overhead | 14 |

The requested coarse count is `1,170 factor occurrences * degree 5 = 5,850` Fr
multiplications per row pair: **0.767 B** in round 0 and **1.534 B** over the geometric
`2^18-1` pair count. Counting the degree-5 polynomial's seven evaluation points gives an
upper work model of **1.073 B in round 0 / 2.147 B total**. The member kernels reuse affine
forms and bit work, so this is a traffic/compute ceiling, not an executed counter. Measured stage A
is 7.095 s; CopyLink and T2 each consume about 40%.

## 2. Data-volume census

`N=8,388,608` physical points per packed group. Current kernel reference rates:

| scalar class | add model | current clean rate | floor used below |
|---|---|---:|---:|
| bit | one selected affine add | 21–25 ns/selected add | **18 ns/selected add** |
| u16 | two 8-bit bucket passes | 0.046 us/active point | **0.030 us/active point** |
| u32 | four 8-bit bucket passes | 0.092 us/active point E | **0.060 us/active point** |
| full Fr | 16 signed 16-bit windows + bucket reduction | 0.42–0.45 us/active point | **0.30 us/active point** |

The full-Fr floor is `ceil(254/16) * N` bucket insertions at about 18–19 ns each, plus about
one million bucket-reduction additions for an `N=2^23` MSM. It is a campaign target, not a claim
about an unmeasured hardware limit.

| commitment phase | columns / groups | width | physical points | active full-Fr points | ms M | measured us/point | modeled floor | gap |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| phase 1a | 768 padded / 24 | 10 bit groups; u32 words; W; small-valued fixed groups | 201.33 M | W is only 0.262 M; other groups use bit/small paths | 1,783 | 0.0089 physical | 1.0–1.2 s | 1.5–1.8x |
| T2 1b | 96 / 3 | 61 u16, 6 bit, 3 u32, one signed-small `D`, padding | 25.17 M | 0.262 M signed-small | 1,148 | 0.0456 physical | 0.82–0.90 s | 1.3–1.4x |
| T2 2a | 96 / 3 | 67 full Fr + padding | 25.17 M | **17.56 M** | 7,435 | 0.295 physical / **0.423 active** | **5.27 s** | **1.41x** |
| T2 2b | 32 / 1 | 2 full Fr + zero padding | 8.39 M | 0.524 M | 101 | 0.012 physical / 0.193 active | 0.08 s | 1.26x |
| T2 2c + VK + Copy helpers | 96 / 3 | 3 Fr lookup cols, 6 fixed cols, 20 sparse Fr helpers | 25.17 M | sparse; most entries zero | 348 | 0.0138 physical | 0.20–0.25 s | 1.4–1.7x |
| HyperKZG binary fold commitments | 22 MSMs | full Fr | **8.39 M** | 8.39 M | 8,250 | **0.983** | **2.52 s** | **3.27x** |
| HyperKZG quotient | one MSM | full Fr | **8.39 M** | 8.39 M | 3,799 | **0.453** | **2.52 s** | **1.51x** |

The fold gap has a code cause: `HyperKZG::open` uses an outer `par_iter` across 22 commitments,
and each commitment starts an internally parallel MSM. The quotient is one flat MSM. Sequential
fold commitments should land near 4.0–4.4 s without changing algebra or bytes.

## 3. Lever table — ranked by seconds per engineering day

Savings are local against the idle run. `~0` gas means under 5,000 modeled gas.

| rank | mechanism | saving | payload bytes | gas | days | s/day | risk | files | gate |
|---:|---|---:|---:|---:|---:|---:|---|---|---|
| 1 | Make fold commitments sequential; remove the outer nested-Rayon layer | **4.1 s E** | 0 | 0 | 0.25 | 16.4 | low | `jolt-hyperkzg/src/scheme.rs` | real fixture; fold <=4.5 s |
| 2 | Correct the gate clock: cache the reference layout/key; print T2 finish/member setup | **2.7 s reported**, ~0 production | 0 | 0 | 0.25 | 10.8 report | low | wrapper real gate, `wrap.rs` | phase sum equals honest wall |
| 3 | Derive the claimed value from HyperKZG's final two-entry fold; delete the second `N`-point evaluation | **0.51 s M** | 0 | 0 | 0.1 | 5.1 | low | `stream/protocol.rs`, `jolt-hyperkzg/src/scheme.rs` | opening claim/tamper tests |
| 4 | Sparse CopyLink prover: store active `(row,wire)` positions, borrow value sources, batch the ten inversion lists, share one sound `tau`/eq table, stay sparse through early binds | **4.8 s E** | 0 | 0 | 1.5 | 3.2 | medium | `links/copy_link.rs`, `links/terms.rs`, `wrap.rs` | helpers <=0.5 s; Copy stage <=0.8 s |
| 5 | Pack all 120 CopyLink fixed columns contiguously instead of ten 12-column groups | **0.4–0.7 s E** | **-64** standalone | **-40k E** | 0.25 | 1.6–2.8 | low | `wrap/key.rs`, `links/` | full groups 34 -> 28; stage B 11 -> 10 |
| 6 | Flat MSM work graph, bucket-pass tuning, width-aware scheduling; target 0.30 us/full point | **4.8 s E** after rank 1 | 0 | 0 | 2.5 | 1.9 | medium | `jolt-crypto/src/ec/bn254/{msm,bit_columns}.rs` | isolated N sweeps + real 2a/open |
| 7 | T2 typed row matrix: construct once from `Column`, keep bit/u16/u32 through round 0, fuse bind with next-round evaluation, coefficient-form range terms | **2.0–2.5 s E** | 0 | 0 | 1.5 | 1.3–1.7 | medium | `limb_table/{stream,row_sumcheck,relation}.rs` | T2 setup <=0.6 s; T2 stage <=1.4 s |
| 8 | 4-ary HyperKZG fold over two variables per level | **2.6 s E** after rank 1 | **-288 E** at `ell=23` | **+68k E** | 1.75 | 1.5 | high | `jolt-hyperkzg/{scheme,kzg,types}.rs` | independent fold identities; exact bytes/gas |
| 9 | Build T2 operands virtually from phase-1b chunks and the existing copy/fingerprint checks; do not commit 44 `Z_xi` columns | **4.0–4.8 s E** | **-128 E** standalone | **-15k E** | 4 | 1.0–1.2 | high | `limb_table/{stream,relation,lookup,export}.rs`, stream reduction | new soundness proof + adversarial operand tamper |
| 10 | One recording verifier pass for R and T1; cache T2 layout; fuse program evaluation with chunk generation | **0.4–0.7 s E** | 0 | 0 | 0.75 | 0.5–0.9 | low | `wrap.rs`, `relation/replay.rs`, `hash_table/`, `limb_table/columns.rs` | preparation <=0.35 s; T2 adapt <=0.9 s |
| 11 | T2 LogUp grouping sweep: `s=3 -> 4` uses existing degree-6 support; compare `s=6,9` only after coefficient form | **0.4 s E (`s=4`); 0.5–1.0 s E (`s=9`)** | **0 (`s=4`); +160 (`s=9`)** | ~0 | 0.75 | 0.5–1.3 | medium | `limb_table/{columns,lookup,relation,stream,export}.rs` | real 2a + T2 member; degree/bytes check |
| 12 | Fill phase-2c's 29 empty wire slots with the 20 CopyLink helpers before the VK group | **0.05–0.10 s E** | **-32** | **-6k E** | 0.25 | 0.2–0.4 | low | `limb_table/stream.rs`, `wrap.rs`, `wrap/key.rs` | final wire groups 2 -> 1 |

### MSM notes

- Current affine candidate: 0.42–0.43 us/point clean; quotient 0.453; phase 2a 0.423 active.
- Run 4/6/8/10-thread sweeps in randomized order after load <2. Report min/median and CPU-seconds/wall. Keep 10 threads only if it wins wall without a hot-run reversal.
- Add dispatch only with a production caller. A signed-u128 operand path costs about 18 16-bit windows per source point versus 16 full-width windows; it must measure <=0.32 us/active point or be deleted.
- Batch-affine bucket work needs one flat task graph. Nested group/MSM parallelism already failed twice and explains the fold result here.

### LogUp / CopyLink choices

1. **Do now:** `s=4` for T2. It cuts 67 -> 62 phase-2a columns and 3 -> 2 groups while staying inside degree 6. The extra final factor costs 32 B, canceling the 32 B commitment saving.
2. **Then measure:** `s=9` gives 8 helpers instead of 22, but raises the row/term degree to 11 and final factors from 4 to 10. Prior data predicts about +1.0 s of T2 sumcheck and **+160 B net** at this row count.
3. **CopyLink:** first remove dense work without changing the argument. Ten links currently clone about 200 full columns into member state and execute zero relations on unused rows.
4. **Optional global batch:** at most four tagged `s=3` links use 48 fixed columns, eight helpers, and 40 terms versus 120 fixed columns, 20 helpers, and 100 terms. Projected payload delta **-96 B**, gas **-55k**, helper/stage saving **3–5 s**; accept only after a cross-namespace swap test.
5. **Do not lead with helper-free fractions:** a single binary rational sumcheck at two Fr per round costs **1,152 B**. Deleting two helper commitment groups plus one stage-B round saves at most 128 B, so net is **+1,024 B** before final claims. Expected saving 3–5 s; wire cost is too high for the first campaign.

### HyperKZG and `k` exchange rate

| k | N | opening now | payload | delta vs k32 | verifier |
|---:|---:|---:|---:|---:|---|
| 32 | `2^23` | **12.616 s M** | **7,488 B M** | — | **234 ecMul; 5.049 M gas M** |
| 16 | `2^22` | **~6.3 s E** | **7,776 B M** | **+288 B** | 247 ecMul; **~5.017 M gas E** from the measured pre-matrix delta |
| 8 | `2^21` | **2.0–2.2 s M/E** | **8,576 B E** | **+1,088 B** | ~284 ecMul; **~5.25 M gas E** |

After sequential folds + 0.30 us/point + 4-ary folds, opening projections are about
**3.6 s / 1.9 s / 1.0 s** for `k=32/16/8`. Thus `k=16` is the cheapest byte move that makes
the opening itself about 2 s; the rest of the wrapper remains far above 2 s.

The 4-ary `ell=23` wire model has 11 intermediate G1 commitments instead of 22 and 49 Fr
evaluations instead of 47: `-352 + 64 = -288 B`. The five-point batch needs two more pairing
pairs than the current cubic opening, about +68k gas. Pin these counts with a serialization test.

### Parallel schedule allowed by Fiat–Shamir

| boundary | safe work | expected wall win |
|---|---|---:|
| after phase 1b | CopyLink helpers may run beside T2 phase 2a/2b; all needed link challenges are fixed | **0–0.3 s** now because phase 2a already fills the CPU; more only after sparse helpers |
| one stage-A round | members may compute the same round concurrently before the aggregate round commitment | unknown; partition threads, never nest full Rayon pools |
| Spartan outer/inner | strict outer -> inner -> carry; only stage-A round-0 precomputation can overlap | <0.1 s |
| term/B/opening | strict `r_A -> term -> stage B -> opening` | none |
| HyperKZG folds | fold polynomials are sequential dependencies; commitment MSMs should be sequential at the outer level and parallel inside | **~4.1 s** |

## 4. Ordered lanes

Every code lane ends with the real fixture round trip, the full existing tamper suite, exact
payload/bincode/statement bytes, and verifier operation/gas output. Cargo lanes run serially.

| lane | scope | acceptance / expected saving |
|---:|---|---|
| 0 | Permanent phase timers; move test-only key/negative setup outside the honest clock; record CPU-seconds/wall and start/end load | phase sum within 2%; no production-time claim |
| 1 | Sequential fold commitments; derive the claimed value from the final fold | fold <=4.5 s and remove 0.509 s pass; **4.4–4.8 s** |
| 2 | Sparse/borrowed CopyLink witness and prover; contiguous fixed-column packing; fill final-phase slots | helpers <=0.5 s, Copy stage <=0.8 s; **4–5 s**, at least **-96 B** combined |
| 3 | One typed T2 matrix, fused rounds, coefficient-form range work; preparation replay/layout cleanup | T2 finish+member setup <=0.8 s and T2 stage <=1.4 s; **2.5–3.0 s** |
| 4 | MSM thread/window/bucket campaign, including thermal sweep and per-width deletion gate | full Fr <=0.30 us/point; phase 2a <=5.4 s; quotient <=2.6 s; **~4.8 s** against baseline |
| 5 | `s=4/6/9` sweep, phase packing, then 4-ary HyperKZG | choose max wall saving, not fewest columns; **3.0–3.6 s**, exact byte/gas ratchet |
| 6 | `k=16/8` real gates after all prior lanes | choose `k=32` unless the full wrapper, not just opening, wins enough per added byte |
| 7 | Only if still funded: virtual T2 operands, then tagged global CopyLink | **4–8 s E**; fresh soundness review required before timing decides |

Stop rule: if lanes 1–5 leave `k=8` above 8 s, close the 2 s campaign as infeasible. The
measured component floors already predict that result; lane 7 can lower bytes/work, but cannot
erase T1, T2 phase-1b, T2 row proving, and the final opening together.
