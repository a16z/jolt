# W4-T2 limb-table review #2

Target: `2bd85b51b` (`crates/jolt-wrapper/src/limb_table/`; milestone 2)

## Verdict

**1 blocker / 3 majors / 1 minor**

### Blocker

1. **`schedule/ec/g1.rs:51-60`, `schedule/ec/g2.rs:58-70` — the θ correction base invalidates the stated one-root proof.** Each main chain appends the fixed point `−K` with scalar `Wire::Offset`; its selected digits therefore depend on θ. The module argument at `schedule/ec/mod.rs:8-18` puts those already-processed correction digits in a θ-independent `H`, which is false. In particular, when the intended MSM is zero, the final correction-base add is `P + (−P)` for every θ, its denominator is zero, and `g1_add` / `g2_add` permit an arbitrary slope and output. The scratch test sets `A1 = FinalE1 + d·Γ1_0 = 0`; program evaluation reaches a noninvertible add for both θ = 1 and θ = 2. No constraint forbids a zero MSM, and the fresh-θ argument assigns this phase-1a state no failure bound; the later ψ check would validate the prover's chosen output rather than the intended G2 MSM. **Fix:** guard every correction-base add, or use a complete formula; then replace the current proof with one that accounts for the correction digit partition. Guarding only the ψ chains is insufficient.

### Majors

1. **`adapter.rs:107-114` — `from_jolt` rejects every real R/T2 wire set.** R publishes `Chi(σ)`, `S1Acc`, and `S2Acc` (`relation/dory.rs:154,195-196`), while `FlattenedCheck::wires()` omits them: `Chi(σ)` is the constant-one base, and only `Ht` plus the two pairing coefficients consume the accumulators. The strict length/set check therefore differs by three. Removing that check alone would still break the digit input claim: R includes their `ρ^k s_k` terms while T2 assigns zero digit multiplicity to unused wires. **Fix:** keep these values internal to R but remove the three `emit` calls, so `DoryLinks` publishes exactly the T2-used order.

2. **`stream.rs:120-125` — `commitment_phases` omits the verifier-key groups emitted by `StreamColumns`.** `StreamColumns::new` appends the VK suffix at `stream.rs:250-258`, but the four declared phase sizes sum only `prover_group_count`. `AssemblyStatement` requires phase group counts to sum to the full packed group count (`stream/protocol.rs:381-395`), so a T2 assembly built with this helper returns `StageCount`. The scratch regression gets 37 declared groups versus 39 emitted groups at packing 4; production packing 32 misses one VK group. **Fix:** include the VK suffix in the last phase size (its physical position is after phase 2c), or return a full phase plan that covers it.

3. **`tests/limb_table_program.rs:442` — the “σ = 11, N = 42” verifier budget test builds `N = 5`.** Replacing `FlattenedCheck::derive(11, 5)` with `(11, 42)` changes the observed count from 9,930 to **10,019 Fr multiplications**, failing the asserted 10,000 cap. **Fix:** test the stated profile and cut at least 19 counted multiplications, or revise the accepted budget and its downstream gas estimate.

### Minor

1. **`relation.rs:896` — `relation.rs` is 1,068 lines, above the repository's 1,000-line soft limit.** `RowSumcheck`, `eq_tau_column`, and the final-evaluation helpers form a direct split that leaves the relation formula in one owner.

## Soundness checks accepted

- **Phase discipline:** local columns follow `1b → ξ,α → 2a → fp_root → 2b → β,fp_combine,copy_root → 2c → stage-A challenges`; no prover-selected operand/fingerprint/helper appears after its batching challenge. The missing VK phase coverage is major 2. Global θ/ρ offsets await the W6 assembly caller.
- **Fixed-base and ordinary Straus sites:** `R = [θ]G`, `Z0 = φ(R)`, the offset algebra, table walks, and θ digit RHS are correct apart from the correction-base exception above. Table adds have one θ root; ordinary main-chain adds have one root after conditioning on the proof point/scalar commitments. Doublings cannot hit nonidentity 2-torsion because both full curve orders are odd.
- **ψ chains:** every NAF accumulation add, the final `+2P`, and the ψ-tail sum use `g2_add_guarded`; the guard pins both coordinates of `inv·(x2−x1) = 1`. The inputs are exactly `FinalE2` and the copied `B2`. Scratch points of exact order 10,069 and 5,864,401 were both rejected.
- **Norm-one projection:** no hole found. For BN254, `gcd(r, (p^6 + 1)/r) = 1`, so the norm-one torus splits as `G_T × H`. Multiplication, integer scalar powers, conjugation/inversion, and Frobenius commute with the projection; the Miller result's final exponent is computed independently on the LHS. Every raw Fq12 base gets `x·conj(x)=1`, and the easy-part inverse is pinned by all 12 coordinates. This target has no compressed-Fq6 GT input path.
- **Review-1 fixes:** the 16-bit table side is VK-owned and both `MULT`/`INV` vanish above it; every input-coordinate row gets the top-64 canonicality equation; `x+q` and `x+2q` aliases fail; G1 and Fq2 sign selection matches arkworks' `y > −y` ordering. `Col::CLAIMED = 149`, the stream term list is 175 terms at degree 4, and member degrees/offsets are `(5,0)` and `(2,0)`.
- **Oracle:** `DoryScheme::verify` is independent of `FlattenedCheck`; the production mutation test covers one G1 and one GT proof message. No shared-oracle acceptance found.

## Scratch tests and verification

Patch: `.journals/lanes/w4t2-review-2-tests.patch`

- `zero_msm_is_exceptional_for_every_offset_challenge` — pass; both θ values reach the exceptional correction add.
- `exact_small_torsion_pairing_inputs_are_rejected` — pass for exact orders 10,069 and 5,864,401.
- `commitment_phases_cover_verifier_key_groups` — expected failure: 37 declared, 39 emitted.
- Corrected fibonacci budget test — expected failure: 10,019 Fr multiplications.
- Clippy (`--lib` plus `limb_table_e2e`, `limb_table_program`, `limb_table_miller`) — pass.
- Original T2 suite — 39/39 pass in 73.8 s.
