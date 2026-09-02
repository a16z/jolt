# Metal wave 13 — lane T13: tier-1 batched-affine tree (X9's parked door)

## Verdict

**NO-GO — door closed permanently, with measured receipts.** The batched-affine
tree is not occupancy-gated as the w9 report feared; it is
**inversion-amortization-dead** at the thread-per-segment kernel shape, and the
occupancy gate is real on top of that. On the production fixture (cap 128,
value-verified bit-for-bit against `jk_g1_seg_sum` outputs, 0 diverged
segments across 80k segment comparisons):

| variant (median, 2^24 shape) | vs real kernel | 2^22 shape |
|---|---:|---:|
| real kernel w64 (baseline 5.02 ms) | 1.000 | 1.000 |
| `xv_bat_l0` — best possible single-inversion hybrid | **1.100× slower** | 1.149× |
| `xv_bat_tree` — the parked door (per-level batch inversion) | **2.520× slower** | 2.541× |
| `xv_bat_tg` — 32 KiB threadgroup-staged ceiling | **5.289× slower** | 5.457× |

Modeled @2^27 (w12 anatomy: tier-1 pure ≈ 4.2–4.4 CB-s, device pays wall at
~0.86): the tree would ADD ≈ +6.4–6.7 CB-s ≈ **+5.5 s wall**; the best
single-inversion shape adds ≈ +0.4 CB-s ≈ +0.36 s. The ≥1.0 s bar is
unreachable by a margin that no window quality or tuning can bridge. Zero
production diff — the only code is the receipts rig (bench-only `g1bat` leg).

## Why the ~6-vs-10 pricing sketch fails (mechanism)

Montgomery batch inversion amortizes over the **per-thread sequential batch
K**, never across SIMT lanes — all lanes of a simdgroup pay the inversion's
wall time simultaneously, so cross-lane or threadgroup sharing buys nothing.
Inside one segment the adds form a chain; the only independent-add structure
is the pairwise tree, which caps K at L/2 = 64 (level 0) and forces
⌈log₂ L⌉ = 7 inversions per cap-128 segment (one per level — level k+1's
denominators depend on level k's outputs). Exact counts (S = M here;
`fq_sqr` is `fq_mul`):

| shape @L=128 | Fq muls | vs 1274 |
|---|---:|---:|
| XYZZ chain (current): 127×10 + 4 | 1274 | — |
| batched tree: 6×127 + 7·I, I=388 measured | 3478 | 2.73× |
| level-0-only hybrid: 64×6 + I + ~64×10 + 4 | 1416 | 1.11× |
| **fantasy floor** (ALL 127 adds share ONE inversion — violates tree deps) | 1154 | **0.91×** |

Measured tree ratio 2.52 sits slightly under the 2.73 model because
length-sorted tail segments have fewer levels. Break-even inversion cost for
the real tree to reach the −24% bar: **I ≤ 29 Fq muls**; even the
physically-impossible fantasy floor needs I ≤ 202. Measured I = **388–401
mul-equivalents** (dependent-chain `xv_invroof` vs `xv_mulroof` at identical
thread count/width; naive Fermat = 253S+109M = 362 model, addchain floor
≈ 309). No 254-bit constant-time inversion is within 10× of break-even.
CPU-MSM batch-affine gets its ~6/add by amortizing one inversion over
thousands of independent bucket adds on one thread — a shape a
thread-per-segment GPU kernel cannot reproduce; per-add inversion overhead at
the maximal in-segment K=64 is already 388/64 ≈ 6.1 muls, larger than the
entire 4-mul/add saving.

## Occupancy receipts (X9's flagged gate, now quantified)

- **Threadgroup staging ceiling:** batching needs the prefix products live
  (recompute is O(K²); the gather itself is cache-cheap so operands re-load
  free — only prefixes must stage). 32 KiB TG budget @width 64 = **16 Fq
  slots/thread** → K ≤ 16 → ≥11 inversions per cap-128 segment. Measured
  `xv_bat_tg` (exactly 32 KiB static TG): **5.29× slower** — the added
  inversions and the 1-TG/core residency compound. `maxTotalThreadsPerThreadgroup`
  stays 1024 for all variants (register pressure is not the limiter; TG
  memory and mul count are).
- **Device-scratch staging** (occupancy-neutral, 8 KiB/thread ping-pong):
  removes the TG gate entirely and the tree is still 2.52× slower — proof the
  door is arithmetic-dead, not merely staging-dead.
- Roof context: `xv_mulroof` dependent chain @39.2k threads = 8.41 Gmul/s;
  the real kernel runs at 9.43 Gmul/s-at-10-muls (madd's internal ILP beats
  the 1-chain latency roof). No new in-kernel lever surfaced.

## Evidence base

Two isolated microbench processes (the entire GPU budget), each GPU-locked,
45 s cooldown between, variants bracketed by same-window baselines
(start/end drift: 4.967→4.948 ms @2^22, 5.022→5.073 ms @2^24; window healthy
— baseline rates match X9's post-cut anchors). No e2e pairs, no 2^27 runs,
no FrBind-gated certs — at a 2.5× effect size with two agreeing scales,
window class is immaterial.

- 2^22, iters 3: 41,348 segs / 4,735,424 adds / row 8192 (shakeout + first receipt)
- 2^24, iters 5: 39,210 segs / 4,735,424 adds / row 16384 (production-shape receipt)

Every batch variant's output slab was decoded and compared segment-by-segment
(projective equality) against the production kernel's outputs on the same
fixture in the same window: **diverged = 0** for all three variants at both
scales. The variants implement the COMPLETE group law (batched denominators
substitute 1 for degenerate pairs; doubling via 2y/3x², P+(−P)→∞, infinity
sentinel (0,0)), so the timings price the real algorithm, not a lossy sketch.

## What ships

Bench-only receipts rig, one commit: `g1bat` leg in
`jolt-eval/bin/st0_contention.rs` (`--legs g1bat`, builds the cap-128
production case) + `BAT_SRC` variant kernels (`fq_inverse`,
`xv_invroof`, `xv_bat_l0`, `xv_bat_tree`, `xv_bat_tg`) with per-segment
value verification. Flagged for the PR-handoff audit (X9 rig precedent —
delete-or-keep). No production sources touched: `g1.metal`, `g1.rs`,
`commitment.rs` (2508 lines, unchanged), `KernelId::ALL` (85) all intact.
No kill switch needed — nothing swaps on.

Door-close corollary for future waves: any tier-1 in-kernel cut below the
current kernel must cut muls/add without inversions (XYZZ 8M+2S is the known
optimum for mixed adds) or raise thread count; batched-affine in ANY staging
regime is priced out at this segment geometry. A global multi-pass tree
(device-staged levels, ≥256 pairs/thread/inversion) prices to ~8.2 muls/add
before pass overhead, thread starvation at deep levels, and ~300 MB of
inter-level traffic — a wash at best, and outside this kernel's shape;
not a >20% lever either.

## Gates & discipline

- Gates (zero production diff, all green): metal suites 411/411
  (`-p jolt-kernels -p jolt-dory -p jolt-eval --features
  jolt-kernels/metal,jolt-eval/metal`), byte-diff 20/20 (`-p jolt-prover
  --features prover-fixtures`), `clippy --all --features host -D warnings`,
  plus targeted `clippy -p jolt-eval --features metal -D warnings` (the
  macOS-gated bin the host gate never compiles; fixed two pre-existing-style
  `iter().any` lints in touched lines), `cargo fmt`.
- All cargo under `/usr/bin/lockf -k /tmp/jolt-metal-wave3-cargo.lock`; both
  timed GPU runs additionally under `/usr/bin/lockf -k
  /tmp/jolt-metal-gpu.lock`, one at a time, 45 s cooldown. 2 timed
  microbenches total, ≤2 per decision honored (scales agree 2.52/2.54 — no
  third). No 2^27 profile needed (decision margin ~150× the bar).
- Sibling lane metal-w13-miller: `miller.*` untouched; `commitment.rs`
  untouched entirely; scratch/metal-saturation untouched; nothing pushed.
