# W4-T2 limb-table review #3

Target: `4283facd4` (`crates/jolt-wrapper/src/limb_table/`; closing adversarial review)

## Verdict

**1 blocker / 2 majors / 1 minor**

### Blocker

1. **`limb_table/lookup.rs:454-497`, `limb_table/schedule/mod.rs:351-358`, `limb_table/digit_link.rs:1-42` — the digit link checks the average scalar across a wire's occurrences, not each occurrence.** Every occurrence of `Wire::Offset` gets the same `kd`, while `rho_weights_with` assigns each one `ρ^kd / multiplicity`. Thus two chains can change the same window digit by `+1` and `−1`; `Σ ωD` is unchanged exactly, both digits remain valid, and no copy/constancy relation connects the occurrences. The lookup and digit constraints then let each chain select and evaluate its own new entry. This invalidates both the MSM semantics and the non-degeneracy premise that each correction digit prefix is the committed scalar's fixed prefix. A prover can choose the G1 offset-chain scalar `t` after θ so its first proof-base add has `16tG = dP + tλG`, compensate `t` in the other seven offset occurrences, and hit the unguarded `entry = acc` case deterministically. `g1_add` accepts that case: the zero slope satisfies its vacuous pin and yields a garbage output. **Fix:** bind all occurrences of a wire to one digit sequence and bind that sequence to the canonical scalar representative; remove the multiplicity average. If canonical prefix binding is not added, guard all remaining affine adds/doubles and stop relying on prefix non-degeneracy.

### Majors

1. **`relation/dory.rs:154,195-196`, `limb_table/adapter.rs:106-114` — R still publishes three scalars T2 does not consume.** At σ = 11 / N = 42, `FlattenedCheck::wires()` has **173** named wires, but `DoryLinks` emits **176**: the extra `Chi(σ)`, `S1Acc`, and `S2Acc`. `from_jolt` therefore returns `AdapterError::WireSet { links: 176, check: 173 }` for the production shape. Dropping the adapter check would make the digit-link claim include three unmatched `ρ^k s_k` terms. **Fix:** keep these variables internal to R, publish only the 173-member consumed set, and hand that exact order to `schedule::build`.

2. **`limb_table/lookup.rs:345-429,551-594`, `tests/limb_table_program.rs:434-486` — 9,986 Fr is not execution-derived.** The observer misses direct verifier-path products in `coord_eval`, `s0_eval`, `omega_eval`, `eq_tau`, `small`, `id`, and the free-field moment. The three obvious `public_evals_with` loops alone add 74 products, so the existing count is already at least **10,060**. Routing all direct products in those functions through the observer produced **10,408 Fr** and failed the 10k test; unobserved `AffineForm::scale(-1)` calls remain. The five batched scalings, shared public/ω evaluator, and `group_into` memoization preserve the final claim, but the budget gate does not measure its stated quantity. **Fix:** route every nontrivial verifier product through the observer (replace negation-only multiplies with negation), then trim against that count.

### Minor

1. **`limb_table/schedule/ec/mod.rs:223-248` — the doubling sweep uses the add-stage correction coefficient.** Before a window's first double, the correction prefix is `−k_K P_w`; all four doubles are exceptional iff that input is identity because the curve order is odd. The actual coefficient is therefore `A_w·16^(64−w) − k_K`, while the test checks `A_w·16^(64−w) − 16k_K`. The `16k_K` term is correct only after four doubles, in the proof-base add formula with `A_{w,k}`. The corrected sweep still passes for both groups and all `n,w`, so this is a model-ownership defect, not a found degeneracy. **Fix:** use `−k_K` for the doubling assertion and update the module formula.

## EC re-derivation

- **Straus tables:** each base table is `T_j = (j−8)P + Z0`, including `T_8 = Z0`; sequential `±P` construction is unguarded. With a fixed canonical scalar prefix, `T_{j±1} = ±P` is one linear root in θ; `P` cannot be identity because inputs are affine on-curve points. Table entries may coincide only on the same root event.
- **Main accumulators:** after four doubles, before proof base `k`, the θ-dependent part is `θA_{w,k}G − 16k_KP_wG`; the implementation's `A_{w,k} = 16^(w+1) + λ(n(16^(w+1)−16)/15+k)` matches the add sweep. With both coefficients nonzero, the prefix/suffix counting argument gives at most `2^129` θ values per site. This conditional argument is invalid until the blocker fixes the digit sequence.
- **Correction adds:** G1 and G2 both set `guard: true` on every main chain. Slot `n+4` reads the last base add at `n−1` and its prior accumulator at `n−2`, in every one of 64 windows. `t = λ²−2x1−x3 = x2−x1`; the inverse row and `inv·t = 1` pin rule out both equal-x cases, including the last add. Guard wiring is complete.
- **Exceptional signs:** for an unguarded add, `entry = −acc` gives `0 = −2y1`; a finite point with `y1 = 0` would be 2-torsion, absent in both odd-order curve groups. `entry = acc` makes the slope pin vacuous and is accepted. `Program::evaluate` selecting slope zero does not alter verifier soundness: it is witness generation, and `program.rs:401-403` is false for the equal-point case.
- **Fixed-base R chains:** entries are `(d+9)G`, hence never identity. The integer bound covers windows 0–62; the final window and fixed correction leave only the stated small residue set. No new fixed-base collision found.
- **Miller loop:** line-step rows are denominator-free projective formulas; the first two Q inputs pass the ψ subgroup checks and the other two are VK points. The fixed ate schedule and final ψ additions match `G2Prepared` step-for-step. No prover-supplied slope appears here.
- **ψ chains:** every NAF add/subtract, the final `+2P`, and the tail `ψ²(P)+ψ(B)` use `g2_add_guarded`; the tail negation is four direct coordinate pins. Doublings of a finite nonidentity point stay generic because the full twist order is odd. Both exact torsion negatives reject.
- **Final exponentiation:** glue is only Fq12 multiplication/Frobenius. The sole inverse witness is pinned by all 12 coordinates of `f·f⁻¹ = 1`, so `f = 0` rejects. No exceptional output path found.

The per-site `2^-125` root estimate is not a circuit-wide failure bound; any claimed global level must include the union across table, add, and doubling sites.

## Other checks

- **Phases:** `commitment_phases` covers all emitted groups at packing 4/16/32, with the VK suffix inside the last phase count. Order is θ → 1b → ξ,α → 2a → fp_root → 2b → β,fp_combine,copy_root → 2c → stage-A challenges. Each prover-selected column precedes its batching/fingerprint challenge.
- **Budget algebra:** `batched_terms` scales the five disjoint coefficient classes once; the stream/member oracle test proves the exported final claim is unchanged. Finding major 2 is accounting only.
- **Code shape:** `relation.rs` is 923 lines; no test-only fields occur in production types. `DoryScheme::verify` remains the independent proof oracle.

## Scratch tests and verification

Patch: `.journals/lanes/w4t2-review-3-tests.patch`

- `digit_link_accepts_different_offset_scalars_per_chain` — pass; two offset-chain digits differ while the link input claim is identical.
- `equal_point_exception_is_accepted_by_the_unguarded_add` — pass; zero denominator/zero numerator satisfies every pin and emits a garbage point.
- `production_dory_links_publish_only_consumed_wires` — pass as a repro of `WireSet { links: 176, check: 173 }`.
- Corrected doubling coefficient sweep — pass for both groups, every `n = 2..64`, every window.
- Budget lower-bound assertion — expected failure: `9,986 observed + 74 known-unobserved`; full direct-product instrumentation measured 10,408.
- Original T2 suite — 43/43 pass in 78.2 s.
- Clippy (`--lib`, `limb_table_e2e`, `limb_table_program`, `limb_table_miller`) — pass with `-D warnings`.
