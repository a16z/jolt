# Akita byte link: protocol, wire, soundness, parity

Feature `akita-byte-link` (default off, implies `akita`). The byte trace `Q` keeps physical signed
bytes; a W-only LogUp per pack authenticates every one-hot claim stage 6b leaves on `Q`, replacing
the selector checks. Production security is **not claimed** ([Open obligations](#open-obligations)).
Bare paths are under `crates/jolt-verifier/src`.

## Protocol

`Q` has 32 slots per cycle, flat order `cN+t`: 0–15 `InstructionRa`, 16–23 increment digits,
24 carry, 25–26 `BytecodeRa`, 27–28 `RamRa`, 29 RAM activity, 30–31 zero. Seven packs:
six byte triples `(0,1,2)…(12,13,14),(15,25,26)` over S³, `S={−128,…,127}`, and RAM
`(27,28,29)` over S²×{0,1}. Tables: six `W` of 2²⁴ entries (one group), RAM `W` of 2¹⁷.

**Inclusive routing** (`stages/stage6b/byte_link.rs`). Each one-hot claim is the full column
evaluation, row zero included, at its address chunk and the stage-6b cycle point `r`. Non-RAM `v_c = 2¹⁶·W̃_j(k_c at c, ½ elsewhere)`;
RAM `v₂₇ = 2⁸·W̃_R(k₂₇, ½⁸, 1)`, likewise slot 28. The increment is the `Q`-linear
`F(t) = Σ_{i<8} 256ⁱ D_{16+i}(t) + 2⁶⁴ D₂₄(t)`, consumed as a field value; no digit range check
survives. Omitted-row routing is unsound (the EqualTable(0)=1→0 attack).

**W-only LogUp.** Pack `j`, `γ_j ∈ F³`: `Σ_t eq(r,t)/(β − γ_j·D_j(t)) = Σ_h W_j(h)/(β − γ_j·h)`.
`W_j`, the eq(r,·)-weighted tuple histogram, is committed after `r` and before `γ, β`. Each side
is a binary fraction tree with root `(P, B)`; per pack: `B_T ≠ 0`, `B_H ≠ 0`, `P_T·B_H = P_H·B_T`.

**Fiat–Shamir order** (`stages/byte_link/transcript.rs`, domain `jolt/akita/byte-link/v1`):

1. S0 commits `Q` before any S1 challenge; S6b absorbs the twenty `v_c` and `F(r)`; surviving S7
   members run.
2. Absorb both `W` commitments (`byte_link_triples_w`, `byte_link_ram_w`).
3. Draw 21 independent `γ` (pack order), then `β`. No `M`, no `τ`, no dummy squeeze.
4. Absorb 14 trace roots and 14 table roots; check the seven root identities.
5. Three batched GKRs in order: trace (log T layers, 7 trees), triples (24, 6), RAM (17, 1).
   Per layer: absorb `(batch, layer, point, claims)`, draw independent `[λ_P, λ_B]` per tree,
   degree-3 sumcheck over the layer's index bits (round polynomials absorbed before each
   challenge, linear coefficient recovered), absorb `[P₀,B₀,P₁,B₁]` per tree, check the gate
   `eq · Σ_j [λ_{P,j}(P₀B₁+P₁B₀) + λ_{B,j}B₀B₁]`, draw `µ`, interpolate. Zero-round top layers still check
   the gate. Points cross the module boundary MSB-first: child point `(reverse(s), µ)`.
6. Leaves: trace numerator must equal `eq(r, z)`; table denominator is recomputed from the
   signed-byte affine MLE at `y`.
7. Query reductions, triples then RAM: absorb each pack's marginal queries plus `W(y)`, draw one
   `α` per value, degree-2 sumcheck to one point of every `W` in the group, absorb the `W` finals.
8. `Q` reduction: absorb `z` and the seven denominator leaves, draw `θ` (log T), then weights
   `α_{1..7}, α_F, α₃₀, α₃₁`. Degree-2, log T-round identity
   `Σ_j α_j(β − b_j) + α_F F(r) = Σ_t [eq(z,t) Σ_j α_j γ_j·D_j(t) + α_F eq(r,t) F(t)
   + eq(θ,t)(α₃₀D₃₀(t) + α₃₁D₃₁(t))]`. Absorb all 32 `Q` finals.
9. S8 opens `Q` and both `W` groups jointly with every advice/program group. A zero suffix of `Q`
   is a prover optimization, never a shortened statement.

## Wire at T = 2²⁹

`ByteLinkProof` (`stages/byte_link/proof.rs`, pinned by `byte_link_wire_at_2_29_is_the_counted_size`):

| Item | Fp128 elements |
|---|---:|
| Trace + table roots | 28 |
| Cubic rounds: 3·(406 + 276 + 136) | 2,454 |
| Child endpoints: 4·(29·7 + 24·6 + 17·1) | 1,456 |
| Quadratic rounds: 2·(24 + 17 + 29) | 140 |
| `W` finals (6 + 1) and `Q` finals (32) | 39 |
| **Total**: 4,117 × 16 B | **65,872 B** |

Plus two 128 B commitment payloads (166 B each under bincode standard, T-independent): **66,128 B**, before
`Vec` framing and the S8 PCS delta of the two `W` groups. Measured at T = 2¹² (bincode): 41,839 B = 2,587
elements + two commitments + 115 B framing, matching the same count.

## Soundness

Added error at `N = 2²⁹`, uniform challenges, numerator over `q = 2¹²⁸ − 2³² + 22537`:

| Term | Numerator |
|---|---:|
| W-only residue + cleared rational polynomial | N + 2²⁴ + 28 |
| Three batched GKRs: 3·(406+276+136) rounds + 2·(29+24+17) batch/merge | 2,594 |
| Two `W` query reductions | 84 |
| `Q` reduction and prefix | 64 |
| Two zero-padding tests | 58 |
| **Total** | **553,650,956** (≈ 2⁻⁹⁸·⁹⁵⁶) |

Jolt's sampler reduces 16 squeezed bytes mod `q`, so the largest atom is 2/2¹²⁸; the
conservative drop-in bound doubles every term: **1,107,301,912/q ≈ 2⁻⁹⁷·⁹⁵⁶**. A linear-ROM
loss at 2³² queries gives ≈ 65.956 bits, conditional on an unproven composition theorem; no QROM
or 128-bit claim. The whole error adds `ε_retained-Jolt + ε_joint-PCS`.

The W-only argument needs extraction of the **same `Q` at the S0 prefix**, independent of the
later `r`, and of each `W` at its commitment prefix. With fixed `Q`, an illegal tuple `x` has a
nonzero indicator MLE `w_x`, so `w_x(r) = 0` with probability ≤ n/q; any residue
`a_x = w_x(r) − W(x) ≠ 0` gives a nonzero rational `Σ a_x/(β − γ·x)` of cleared degree
≤ N + K − 1. Layer batch weights must be independent (powers batching does not get this bound).
Each root identity is checked on its own, so the bound is not multiplied by seven.

## What the flag changes in the verifier

- `CommitmentConfig::PackedByteLink` must match the build (`validate_proof_config`, before the
  transcript is seeded): a proof from the other mode rejects before any challenge.
- Only K = 256 with 16/2/2 instruction/bytecode/RAM chunks has a byte-trace layout; prover setup
  and the verifier (`VerifierError::ByteTraceGeometry`) reject other geometries.
- Deleted: S6a `BooleanityAddressPhase`; S6b `LatticeBooleanity` and `RamHammingBooleanity`;
  S7 `HammingWeightClaimReduction`, with their draws, intermediates and absorptions.
- Added: `byte_link::verify` between S7 and S8, and S8 opens `Q` plus the two `W` groups.

## Parity contract

`crates/jolt-verifier/tests/byte_link_parity.rs` and its `.record`:

- One fixture is proved and verified over a **role-keyed tape**: each draw is keyed by its
  `DrawRole` (batch, member, round) and ordinal, so a draw retained in both builds gets the same
  value. The removed members' draws and their unshared rounds are excluded.
- For each S1–S7 boundary three digests: retained statements (member, opening ids, values,
  points), retained draw keys, and retained catalog (member order, rounds, degree, point offset,
  input/output expressions).
- Flag-off and flag-on must both equal the **frozen record**, blessed only flag-off
  (`JOLT_PARITY_BLESS=1`). Flag-on also checks that a perturbed `W` commitment fails S8.

## Open obligations

1. Prefix-consistent adaptive extraction of `Q` at S0 and of both mid-proof `W` groups.
2. Exact grouped SIS, response and compression admission for the `Q`/`W` schedules.
3. A multi-round Fiat–Shamir composition theorem for the stated ROM loss.
4. `ε_retained-Jolt`, including the fused lattice read-RAF stages.
5. Explicit acceptance of this security level.
