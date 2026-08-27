# W5-st8: Dory open — r0-D2 shortcut + fixed-base table kernel; fold-chain break priced dead

Lane branch `lane/metal-w5-st8` @ `4ed633e21` (base `2e1efd307`). Verdict:
**two RETAINED cuts, modeled −0.85 s st8 @2^27** (bar 0.4 s), both with
bit-identical protocol messages (no soundness argument needed); the mandated
fold-chain break is **NO-GO with mechanism**; door-2 cap-32 **shipped
default-on** (no st8 mass — scope correction below); door-3 seam sweep
**closed** — every remaining seam priced unprofitable.

## 0. Decomposition probe (the numbers everything below rests on)

One resident round at the 2^22/2^24-scale widths (probe now deleted from the
tree; gpu_lock, min-of-3, forced gates):

| phase @ n=2^16 (r0 shape) | wall | @ n=2^15 |
|---|---:|---:|
| first_message (4×n/2 merged miller + detached MSMs) | 270.0 ms | 143.5 ms |
| apply_first (v += β·g pass) | 67.0 ms | 40.2 ms |
| second_message (2×n/2 merged) | 164.4 ms | 86.8 ms |
| apply_second (fold-halves pass) | 39.0 ms | 21.0 ms |
| host normalize_batch, all m1 inputs (2n G1 + 2n G2) | **7.4 ms** | 6.9 ms |
| miller device-only (n pairs, pre-normalized) | 126.3 ms | 65.3 ms |
| partial product (host) | 6.2 ms | 3.2 ms |

Two structural facts fall out: **message walls are pure device miller**
(host normalize + upload + product ≈ 5% — the "fuse normalize onto device"
idea is worthless), and **the apply walls are real kernel compute**
(sync overhead ~1-2 ms of the 67 ms), in a 1:4.1 ratio to messages —
matching the record trace's 0.86 s folds : 3.35 s messages.

## 1. NO-GO: fold-chain break / challenge pushes (mandate door 1, option A)

**Mechanism.** The fold chain cannot be broken or deferred because *every
fold state is read directly by the very next message's pairings*: m2's
C± pair the β-folded v1'/v2' at full width, and the next round's D1/D2 pair
the α-folded state. Pushing challenge weights *through* the pairings
expands bilinearly — C₊ = ⟨v1_L,v2_R⟩ + β⁻¹⟨v1_L,g2_R⟩ + β⟨g1_L,v2_R⟩ +
⟨g1_L,g2_R⟩ turns one n/2-pair call into three plus a cached setup pair,
+2^19 pair-evals at r0 alone (~+1.4 s) against 0.86 s of total fold wall.
Pushes are legal only into the E-leg MSMs, which already ride detached
passes off the critical path (probe: MSM exposure ≈ 0 in the unshortcut
arm). Combining β- and α-folds into one 3-scalar Straus pass (−33% fold
ALU) is blocked by the same read: v' must materialize for C±. This is the
same wall the kill-list's radix-4 entry hits — reduce messages are
quadratic in the state, so deferred/packed challenge weights blow up the
pairing count, not the MSM count.

**Residual door, priced open (not built): chunked fold→message pipeline.**
Fold latency CAN hide under the *following* miller if both are chunked
(fold chunk k → normalize k → miller chunk k, one CB, device-affine
production + a ranged fly variant). Upper bound = the apply walls, 0.86 s
@2^27, ~85% hideable (each fold is 2.5-4× smaller than the message that
follows it). Needs: device batch-normalize kernels, chunk scheduling in
`dory_reduce`, partial-buffer regrouping (still exact). Medium risk,
~0.5-0.7 s prize — a full lane of its own.

## 2. RETAINED: round-0 D₂ MSM+pair shortcut (`bbb894de8`)

**Mechanism.** At round 0, v2 = Γ₂fin·v2_scalars, and the CPU arm's
`compute_d2` has always served D₂L/R as `e(MSM(Γ₁', s_half), Γ₂fin)` — one
MSM and one pairing per half. The resident loop ignored `v2_scalars` and
multi-paired all four first-message halves, paying 2^18 device pair-evals
at r0 @2^27 for values an MSM computes. The hook signature now carries
`Option<&[Scalar]>` (vendor `ResidentRoundHooks::first_message`); the
resident arm shrinks r0's miller batch to the two D₁ calls and rides the
two Γ₁'-prefix MSMs in the existing detached beta-MSM pass, plus two host
pairings. Identity `Π e(Γ₁'ᵢ, sᵢ·Γ₂fin) = e(Σ sᵢ·Γ₁'ᵢ, Γ₂fin)`; GT bytes
are value-unique, so the transcript is unchanged — the CPU and metal arms
already relied on exactly this equality agreeing byte-for-byte.

| first_message, r0 shape | 4-call baseline | shortcut | Δ |
|---|---:|---:|---:|
| n = 2^15 | 143.5 ms | 87.9 ms | **−38.7%** |
| n = 2^16 | 270.0 ms | 167.2 ms | **−38.1%** |

Modeled @2^27 (r0 = n=2^18, linear ×4 of the measured −102.8 ms, cross-
checked against W4's r0-first bench 1253 ms × 0.62): **−0.41..−0.48 s**.
Applies to round 0 only (`v2_scalars` dies at the first challenge — after
β the structure is gone, and expanding it re-creates the blow-up above).
Residual: the enlarged MSM pass now outlives the halved miller by ~40 ms
at n=2^16 (wall = max(miller, MSM pass)); an MSM-kernel lane could claw
that back. Kill switch `JOLT_DORY_R0_D2_MSM=0`. Parity:
`reduce_first_message_d2_shortcut_matches_pairing` — shortcut = 4-call =
CPU trait reference, both MSM arms (2048 CPU / 2^13 device), zero and −1
scalars planted.

## 3. RETAINED: window-table G2 fixed-base kernel (`2a7734641`)

**Mechanism.** The VMV preamble's v₂ = Γ₂fin·v_vec sweep
(`jk_g2_fixed_base_mul`, the record trace's 0.478 s at 2^18 scalars —
probe confirms 123.8 ms at 2^16, i.e. device wall, not host) ran a plain
254-bit per-thread ladder: ~254 jacobian doublings + ~127 mixed adds per
scalar. The base is *shared*, so the host now builds one 16-ary window
table (d·16^win·B, ≤64×15 affine entries, ~2 ms, batch-normalized) and
`jk_g2_fixed_base_table` pays only ~60 nonzero-nibble mixed adds per
thread — the doubling ladder disappears.

| fixed-base sweep | ladder | table | Δ |
|---|---:|---:|---:|
| n = 2^15 | 70.4 ms | 13.6 ms | **−80.7%** |
| n = 2^16 | 123.8 ms | 22.1 ms | **−82.2%** |

Modeled @2^27 (0.478 s × 0.18): **−0.39 s**. Group-equal jacobians
(different op order); every consumer normalizes before serializing (the
module's standing contract), byte-diff pins it end-to-end. KernelId::ALL
79 → **80**. Kill switch `JOLT_DORY_FIXED_BASE_TABLE=0` (ladder kernel
retained). Parity: `g2_fixed_base_mul_matches_arkworks` runs both arms
(0/1/−1 scalars, identity base, Z≠1 base).

## 4. SHIPPED: `jk_miller_table` TG cap 32 default (`2b124fef3`) — door 2

**Scope correction:** `jk_miller_table` never dispatches in stage 8 — it
is the stage-0 tier-2 *fallback* Miller (fly-commit gate declined, i.e.
mid-size traces with 2^11..2^16 tier-2 pairs; production 2^22+ uses
`jk_miller_fly_indexed`). "Certify on the st8 shapes" is inapplicable;
certified on the kernel's real shapes instead. Two invocations this window
(cap toggles at context build), `miller_microbench` T2:

| shape (8192 pairs) | uncapped | cap 32 | Δ |
|---|---:|---:|---:|
| ppt=2 (production `MILLER_TABLE_SEG_PAIRS`) | 38.2 ms | 29.2 ms | **−23.6%** |
| every other ppt row (4..32) and scale row (512..8192) | — | — | −7..−13% |

Reproduces W4's −24%. Co-run risk was already priced by W4 §5: family
cap **32** held commit-wall parity (1.207 s both arms) where cap 64
inverted +2% — the freed-occupancy inversion does not engage at 32.
st8 delta: **0**. Kill switch: `JOLT_METAL_PAIRING_TG_CAP=0` (or any
explicit family cap).

## 5. CLOSED: door-3 seam sweep — no profitable seams remain

- **FastTail (rounds below handoff 512):** whole tail 512→1, all phases,
  measured **81-88 ms** total @ any trace scale. Device service would pay
  ~21 ms latency floor × 18 dispatches ≈ 3× worse. Host tail is optimal;
  handoff stays 512.
- **r8 second message (2×512 = 1024 pairs, under the 2048 batch gate):**
  quiet CPU 11.1 ms vs forced merged-device 21.4 ms — **CPU wins**; W4's
  109 ms CPU singles walls were contention-skewed (hungry mid-proof rayon
  pool), so the production gain is contention-dependent and ≤30 ms with a
  quiet-window regression risk. Gate stays.
- **Message-internal dispatches:** all merged since W4 (4→1, 2→1); m1/m2
  cannot merge across β (challenge dependency, and the shared-challenge
  fusion BAN stands regardless).
- **Fold-CB fusion without chunking:** moves only the ~1-2 ms/pass sync,
  ≤50 ms @2^27 — not worth the restructure (see §1's chunked-pipeline
  residual for the real version).

## 6. Package summary

| cut | modeled st8 Δ @2^27 | bytes | switch |
|---|---:|---|---|
| r0-D₂ MSM+pair shortcut | −0.41..0.48 s | identical | `JOLT_DORY_R0_D2_MSM=0` |
| fixed-base window table | −0.39 s | identical | `JOLT_DORY_FIXED_BASE_TABLE=0` |
| miller_table cap 32 | 0 (st0 fallback −24%) | identical | `JOLT_METAL_PAIRING_TG_CAP=0` |
| **total** | **≈ −0.85 s** (st8 5.82 → ~4.97) | | |

## Verification

- `cargo nextest run -p jolt-kernels -p jolt-dory -p jolt-eval --features
  jolt-kernels/metal,jolt-eval/metal`: **405/405** (was 404; +1 = the D2
  shortcut parity test; fixed-base parity extended in place to both arms).
- `cargo clippy --all --features host --all-targets -- -D warnings`: clean.
  Metal-target clippy on the three touched crates: clean. fmt clean.
- **Byte oracle:** `jolt-prover --features prover-fixtures,metal` byte-diff
  suite **20/20** — fixture scales engage the resident loop (n₀ ≥ 2^12)
  and the fixed-base gate (≥2^11 scalars), so both retained cuts ran live
  in the metal arm with proof bytes identical to the legacy prover.
- KernelId::ALL re-counted 79 → 80 (`G2FixedBaseTable`).
- Conditions: velocity v3 (≤2 timed invocations per decision; probe
  min-of-3 in-process, env toggles read per call), every cargo under the
  wave-3 lockf, `gpu_lock()` on every timed pass, no e2e runs (wave gate
  certifies).
- Commits on `lane/metal-w5-st8`: `bbb894de8` (D2 shortcut), `2a7734641`
  (fixed-base table), `2b124fef3` (table cap 32), `4ed633e21` (probe
  removal). Not pushed; scratch/metal-saturation untouched.
