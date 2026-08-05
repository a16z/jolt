# W4 typed-Dory packing pricing — NO-GO

Model basis: per the campaign's bind4 evidence, price one packed round as
`k = 1.51 / 1.70 / 1.98` binary rounds (best/likely/worst sensitivity) at the
same starting width. Instruction-RAV's measured lazy sequence is
`177.3 ms = 75.4+43.3+31.4+27.2`; packing changes it to
`k*(75.4+31.4)`, saving `+16.0/-4.3/-34.2 ms` at `2^24`. Bool's
`152.8 ms ~= 60.8+36.5+27.7+27.8` becomes `k*(60.8+27.7)`, saving
`+19.2/+2.4/-22.4 ms`. Applying those ratios to the current `2^27`
production CB totals (RAV `1.701 s`, Bool `0.969 s`) prices the lazy slice at
`+0.19/-0.02/-0.32 s`; the remaining `~0.86 s` non-lazy family slice can add
at most `+0.21/+0.13/+0.01 s` under equal-cost paired rounds. The st6b
forecast below reserves `0.00/0.03/0.14 s` for the higher-degree D-sum and
interpolation, six-member schedule reshaping, and typed-descriptor handling.
The descriptor/verifier arithmetic itself is `<1 ms`: recompute four
Lagrange weights per factor, absorb the config-derived descriptor before any
Dory alpha, then fold exactly
`alpha_2*alpha_1*l_0 + alpha_1*l_1 + alpha_2*l_2 + l_3`
(`alpha^-1` on `s_2`). Dory reduce-round count and proof size do not fall.

| Scope | Modeled seconds saved @ `2^27` best / likely / worst | Implementation cost | Material risk / required surface |
|---|---:|---:|---|
| Shared typed factor plumbing | `0 / 0 / 0` | **5–7 lane-days** | Replace the scalar-only point seam across `jolt-claims` geometry/output points, `jolt-verifier` derivation/final-point assembly, `jolt-openings` PCS calls, and `jolt-dory`/vendored verifier state. Preserve factor order through Jolt's point reversal and Dory's MSB-first folds; generalize embedding scales and dominant precommitted anchors; forbid a radix-4 factor across the row/column split. Verifier, not proof, recomputes `l_0..l_3`. |
| Packed batched-sumcheck engine | `0 / 0 / 0` | **4–5 lane-days** | Arity-tagged binary/radix-4 schedule, `3d+1` coefficient messages, four-node D-sum, one transcript-derived `z`, factor-aware `ProveRounds`, clear proof recorder/verifier, active-window/optional-member joins, odd boundary singles, exact degree checks. No interpretation by vector length. |
| st6b cycle phase (`7.02 s`) | **`+0.40 / +0.08 / -0.45 s`** | **5–7 lane-days** | All six current members need packed messages and bind4 state; Metal lazy/adopt/dense paths and CPU recovery must advance identically. Width reshaping is the blocker: `w1+w2` already costs less than likely `1.7*w1`, while the only clean 15% saving is in the small launch-dominated tail. Current CPU/device overlap makes non-RA savings unsafe to credit before an integrated prototype. |
| st7 HWCR (`1.33 s`) | **`+0.0004 / +0.0002 / -0.0003 s`** | **1–2 lane-days** | The measured round loop is only `~0.002 s`; the `~1.106 s` pushforward prepare is untouched. Eight binary rounds to four packed rounds saves `0.002*(1-k/2)` before D-sum/descriptor work. Optional address members must share the same config-derived schedule. |
| Transcript/config axis + tamper matrix | `0 / 0 / 0` | **3–4 lane-days** | New transparent-only protocol variant absorbed before stage messages; descriptor absorbed before Dory alphas; `z` never prover-supplied. Fail closed pre-transcript for ZK, Akita, non-Dory PCS, and incompatible dominant anchors. **16–18 tests**: `U=XY` ordinary-MLE mismatch; node-sum/off-node/degree/bit-order attacks; missing/extra/schedule replay; descriptor/kind/weight/`z` tampering; row-column crossing; exact alpha fold plus inverse-alpha `s_2`; point reversal; ZK/Akita/anchor rejection; e2e accept/reject. |
| **Total wave-5 lane** | **`+0.40 / +0.08 / -0.45 s`** | **18–25 lane-days** | Best case only touches the campaign's `0.4 s` bar; likely case is `1.1%` of st6b and `~0.1%` of the `69.63 s` proof, with a credible regression tail and consensus/PCS blast radius. |

**NO-GO for wave 5 — the load-bearing reason is `w1/w2/w4/w8` shrinkage: likely bind4 economics save only ~`0.08 s`, far below the 18–25 lane-day typed-protocol cost.**
