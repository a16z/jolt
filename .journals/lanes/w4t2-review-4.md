# W4-T2 limb-table review #4

Target: `1ffd0088c` (`crates/jolt-wrapper/src/limb_table/`; closing adversarial review)

## Verdict

**1 blocker / 2 majors / 2 minors**

### Blocker

1. **`limb_table/lookup.rs:443-460`, `limb_table/stream.rs:114-131`, `limb_table/schedule/ec/mod.rs:19-31` — per-occurrence linking still permits post-`θ` modulus aliases, so the EC exception bound does not follow.** The link proves each signed radix-16 integer equals its scalar in `Fr`; it does not prove the unique centered recoding produced by `digits()`. The digit range spans exactly `2^256` consecutive integers while the BN254 scalar modulus is about `2^253.6`, so every scalar has five or six valid strings `t = s + m·r`. Each occurrence may choose one after `θ`. The module proof calls the earlier bases' partial MSM `H` fixed before `θ`, but their digit columns are committed in phase 1b and these aliases change their processed prefixes. At σ = 11, both accumulator chains have 34 proof bases before the offset base: up to `6^34 ≈ 2^87.9` response branches at each unguarded add. Unioning the documented per-fixed-branch `2^129/r ≈ 2^-124.6` bound gives only `2^-36.7` per site, or `2^-24.4` across the 4,928 unguarded proof-base adds. This is a proof failure, not a measured attack probability, but the equal-point branch remains accepted and no claimed security level follows. Scratch test `digit_link_accepts_a_modulus_shift_in_one_occurrence` replaces one constant-one occurrence by the valid digits of `1 + r`; the link claim is unchanged for every `ρ`. **Fix:** bind each occurrence to the canonical integer representative, or guard every proof-base affine add so digit-prefix randomness is unnecessary.

### Majors

1. **`relation/dory.rs:154,195-196`, `relation_table/scalar_link.rs:23-32,68-75`, `limb_table/adapter.rs:106-113` — the exact target has no R producer for T2's new weighted claim.** R still publishes 176 scalars and its scalar link uses plain powers `ρ^k`; T2 consumes 173 named wires and now expects `W_k(ρ) = Σ_{o:base(o)=k} ρ^o`. The adapter therefore rejects first, and deleting the three extra emissions alone would leave unequal link claims. This fails closed rather than accepting a false proof. **Fix:** the W5 integration must publish `check.wires()` in that order and use the same occurrence-summed weights as T2.

2. **`limb_table/columns.rs:397-406`, `limb_table/relation.rs:228-235`, `limb_table/lookup.rs:395-401` — the 9,875-Fr budget omits at least 316 executed field multiplications/squares and nine inversions.** `Fr::pow2` is an exponentiation loop (`jolt-field/src/algebra.rs:110-121`), not a free constructor. `Constants::new()` costs 165 unobserved products, `Fr::pow2(64)` costs 7, and the nine selected families each compute `Fr::pow2(4)` plus `Fr::pow2(252)`, another 144. Those families also call `16.inverse()` before the memoized geometric evaluator; `Fr::inverse` delegates to arkworks and none of its work reaches `TermObserver`. The source-level count is therefore at least **10,191 Fr** before the inversions, already 191 above the cap. Direct infix products in the exporter derivation and negations are routed correctly; fixed-power construction was missed. **Fix:** precompute the fixed field powers and `1/16` outside the verifier path, then retain the 9,875 challenge-dependent component-sum assertion.

### Minors

1. **`limb_table/stream.rs:148-159,324-440` — phase geometry has two owners.** `commitment_phases()` derives group counts from `export::phases()`, while `StreamBuilder` manually repeats every phase boundary and only checks the aggregate prover group count with `debug_assert`. Current slices match, including `[3, 3, 1, 2]` at k = 32; the scratch regression covers each slice. **Fix:** make builder phase completion check the corresponding canonical column range in release builds, or keep the slice regression permanently.

2. **`limb_table/digit_link.rs:1-5`, `limb_table/dory.rs:523-525` — shipped docs still describe the removed multiplicity-average scheme.** The public, unused `wire_multiplicity()` also exists only for that deleted design. **Fix:** state the occurrence-weight equation and delete the dead method.

## Link derivation

For occurrence `o`, let `t_o = Σ_w d_{o,w}·16^(63-w)` and `b(o)` its digit-base index. T2 commits

`Σ_o ρ^o·t_o`.

R plus T2's two public terms derives

`Σ_{k<K} W_k(ρ)·s_k + W_K(ρ) + W_{K+1}(ρ)·θ = Σ_o ρ^o·s_{b(o)}`.

Their difference is a degree-`<230` polynomial whose coefficient of `ρ^o` is `t_o - s_{b(o)}`. It is identically zero iff every occurrence recodes its scalar in `Fr`; a sampled false equality has probability at most `229/r`. Constant one and `θ` use the same `W_k` aggregation as named wires. The permanent two-chain `±1` test now rejects because the two changed coefficients sit at distinct powers.

## Accepted checks

- **Point order:** both members pair `j` with `j + rows/2`; their point is big-endian, matching `EqPolynomial::evals` and `PackedColumns::column_evaluations`. The exporter alone reverses it for little-endian kernels. The packed-column/member regression passes.
- **Transcript phases:** 1b contains only challenge-free multiplicities and `θ`-dependent digits; 2a consumes only `ξ,α`; 2b only `fp_root`; 2c only `β,fp_combine,copy_root`; `finish` consumes stage-A challenges. Private builder state and the stage enum prevent a later column from being returned early.
- **Verifier algebra:** the component test still gives 9,875 observed multiplications = 162 relation + 9,573 public/ω + 139 terms + 1 link batching. Prefix/suffix products, doubling-based moments/ids, and cached geometrics preserve the final claim; `stream_exporter_terms_match_the_members` is the independent packed-column oracle. Major 2 covers the remaining arithmetic omissions.
- **Reviews 1–3:** range multiplicity/inverse tails are gated by VK-owned `small`; all raw Fq input rows get canonicality; raw GT bases get norm-one checks; all 64 correction adds in both groups are guarded; ψ checks cover only `E2_fin` and copied `B2` with guarded adds; sign flags match arkworks ordering.
- **Code shape:** every limb-table source file is below 1,000 lines (maximum `relation.rs`, 930); no test-only control field remains in production types.

## Scratch tests and verification

Patch: `.journals/lanes/w4t2-review-4-tests.patch`

- `digit_link_accepts_a_modulus_shift_in_one_occurrence` — pass; `1` and `1 + r` are indistinguishable to one occurrence's link.
- `stream_builder_phase_slices_match_declared_geometry` — pass at packing 4/16/32; k = 32 is `[3, 3, 1, 2]` including VK.
- T2 suite — 46/46 pass: 22 library tests plus 24 tests across `limb_table_e2e`, `limb_table_program`, and `limb_table_miller` (the two scratch tests included).
- Clippy (`--lib`, `limb_table_e2e`, `limb_table_program`, `limb_table_miller`, `-D warnings`) — pass.
- An unscoped name-filter attempt compiled every integration target and hit the pre-existing stale `perf1_profile` API errors; the requested targets above exclude it.
