# Whiteout paper deltas

| Field   | Value |
| ------- | ----- |
| Status  | living log; append new entries at the end, keep the summary table in sync |
| Created | 2026-10-21 |
| Siblings | `specs/whiteout-sigma.md` (W4, A1–A23), `specs/whiteout-masking.md` (M6b/M6c), `specs/whiteout-interfaces.md`, `specs/whiteout-gaussian.md` (M5a), `specs/whiteout-jolt-integration.md` (W6), Akita `specs/whiteout-akita.md` (RP4, A24–A30, Q1–Q6) |

Every place where the implementation departs from the Whiteout paper, extends
it, or fixes a parameter the paper leaves open. Each entry states the argument,
so the authors can fold it into the paper.

**Pins.**

- Paper: `akita-paper/sections/whiteout/*.tex` as read on 2026-10-21. Paper
  references are `file:line` relative to that directory.
- jolt: `whiteout/dev` at `2cc2502ed`. Paths are written `jolt:`.
- Akita: `whiteout/dev` at `fe57e2446`, written `akita:`. The W2 branch
  `wo/akita-w2` at `68c76dbd0` is written `akita-w2:`.
- Indexing: the paper's fold j is Akita level j − 1. The paper's k = 2 is
  `cut_level = 1`.

**Evidence labels.**

- *Verified*: read in the code or a committed spec at the pins above.
- *Reported*: a figure from a test run or review that this log could not
  reproduce. No cargo was run for this log.
- *Draft*: uncommitted work on a branch.

## Summary

| # | Delta | Kind | Status |
|---|---|---|---|
| D1 | Σ attempt runs in constant time: DIT on M4, masked challenge sampler, `Fork::attempt` | extension (outside the paper's model) | implemented |
| D2 | Σ near-zero rejection: σ grows so one attempt almost always accepts | parameter change | in progress (`wo/sigma-norej`) |
| D3 | One-shot fold responses sized to 2^−136 in total, not "one extra digit, 2^−950" | parameter change, fix | implemented on `wo/akita-w2` |
| D4 | Hiding randomness: sizing owner, open alphabet, cut before the terminal, trusted catalogs reject it | clarification, extension | implemented |
| D5 | The cut is k = 2 only (`cut_level = 1`, `masked_folds = 2`) | parameter fixed | implemented |
| D6 | Per-round degree schedule for batched sumchecks | extension | implemented |
| D7 | Shared stage checks over `ClaimDomain`; stage-2 row has no constant term; Q3 answered | clarification | implemented |
| D8 | BlindFold stays curve-based, over pads committed up front | alternative back end | implemented |
| D9 | Compressed images: chunked two-map chains, one row per map evaluated at α | departure | implemented (W5c, `wo/akita-w5`) |
| D10 | Akita #255: public retry of the fold-response nonce at operator-norm masked levels (Lemmas 2.5a/2.5b) | extension, fix (Q7) | argued, under independent review; implementation at the port |
| D11 | Uni-skip rounds send every coefficient plus an explicit sum row | departure | implemented |
| D12 | One auxiliary slot per intermediate product; auxiliary block after the base protocol | departure | implemented |
| D13 | Σ oracle input is (σ, t_D, n) with σ a squeeze of τ; full transcript order | clarification | implemented |
| D14 | Gaussian widths by closed form; 7 digits on every block; randomness rounded to R_256 | parameter change | implemented |
| D15 | Measured Gaussian sample cost is about 40× the paper's 10 ns | evaluation input | measured |
| D16 | Sampler precision term 2^−149; CSPRNG coins | parameter left open | implemented |
| D17 | Σ challenge budget exhaustion is fatal; ⊥ after T_att is an error the caller may rerun | clarification | implemented |
| D18 | Masked-level structural constraints (Linf route, E = F, one chunk, d_A = d_B) | clarification | implemented / specified |
| D19 | Headline fold shapes: level 1 uses d = 512 challenges (η² = 19), not d = 64 (η² = 71) | evaluation input | verified in tests, unexplained |
| D20 | Setup offloading of the response root is optional and not built: the scan is ≈ 10–14 ms idle, the offload ≈ +2.9 KB | evaluation config | shelved (2026-10-22) |
| D21 | Open author questions Q1, Q2 (A7, A8, A9, A14), Q4, Q6, Q7, W6-Q5 | open | open |
| D22 | Q7: the 1/p clause of Lemma `wo:lem:fold-overflow` holds only for the whole challenge tuple | paper fix | open (authors) |
| D23 | Bounded rerun (≤ 3 runs) on FoldOverflow or exhausted Σ attempts, where the paper aborts | departure | specified (W6) |
| D24 | Precommitted groups that carry witness data (untrusted advice, FieldInc, FusedInc) get no randomness | gap (would leak) | closed by Akita W2b: every precommitted group at a masked level is hiding |
| D25 | #254's statement-binding argument rewritten for masked claims | clarification | specified (W6) |
| D26 | Multi-pair product rows ct(Σ_k X_k Y_k − Z); merged repeated products; affine BlindFold operands | extension (paper gap) | implemented (`wo/multipair`) |
| D27 | Soundness ledger of the response proof: ≈ 2^−119.7 in total; the α round alone 126/q ≈ 2^−121 | accepted for now | accepted for now (user, 2026-10-22); revisit with the paper authors |
| D28 | Public coordinates of w_k: the cut level's setup-prefix response and inner digits, at native-element granularity, with no zero cells | clarification, departure | implemented (Akita W5d) |
| D29 | u_a enters N as a public (raw) target; the masked prefix sends it raw (19 × `R_64` after W5e) | departure (cost) | retired by O4 (Akita A47): the cut sends `cm(u_a)`, as the paper does |
| D30 | Two-witness distinguisher: per-byte χ², Bonferroni, calibrated by two injected leaks | evaluation input | measured (Akita W5d) |
| D31 | Precommitted groups opened masked: statement-fixed groups committed hiding at preprocessing, the packing reduction absorbs no evaluation; their trust root is the holder of r | departure (stricter than paper), clarification | implemented (W6f, A44) |
| D32 | Persisted plans: the ZK catalog (planned at setup) and exported response schedules enter a setup only at construction, are re-audited and re-admitted on load, and the pads absorb the masked catalog's, selected row's and response schedule's digests | extension (public parameters), transcript change | implemented (O1) |
| D34 | Small byte items (O5): compressed images take the fewest balanced chunks under the first map's certified SIS bound (15,115 coefficients at q128), so t_p and cm(u_a) take one chunk each; a repeated intermediate product takes one auxiliary slot (Whiteout and BlindFold committed proofs). | departure narrowed (D9), slot sharing (D12, D26) | implemented (O5) |
| D35 | The residual commitment (and the pad commitment, which shares the Σ ring) over `R_128`: rank 9–10 instead of 18–19 over `R_64`, same `R_Σ`; the planner's dimension ceiling is lifted at the after-cut level only, and it never offloads into the cut level | parameter change (follows `06_evaluation.tex:124-133`) | implemented (O6, d = 128); d = 256 gated on a rerun |

## D1. Constant-time Σ attempt

**Paper says.** Timing is outside the model. H2 requires that the oracle
answers at rejected inputs are never observed (`05_security.tex:352-362`), and
the attempt nonce is "a uniform ν-bit nonce in a separate domain of the
oracle" (`05_security.tex:314-317`). No implementation rule is given.

**Implementation.**

- **Rule** (jolt `specs/whiteout-sigma.md` §2.8). Every step before an
  attempt's accept bit runs in time and with memory addresses independent of
  the attempt's fresh secrets (y, n, c and their derivatives).
- **DIT on Apple M4.** dudect found `CyclotomicCrtNtt::to_ring_ct` leaking a
  zero output without DIT (|t| up to 190 at D = 256), although the code and its
  asm are branch-free. With PSTATE.DIT set on every worker thread the leak is
  gone (|t| ≤ 2.0); a positive control with DIT cleared fails at |t| = 9.8.
  - `jolt:crates/jolt-dit/src/lib.rs:33` (`DitGuard`), `:78` (`install`, one
    rayon pool whose workers hold the guard).
  - `jolt:crates/jolt-zk/src/whiteout/sigma.rs:471,503` and
    `pad_commitment.rs:180` run `prove_sigma` and the pad commitment inside it.
  - Record and residual analysis: `whiteout-sigma.md` §2.8 and A11. The early
    residual under DIT was the dudect harness's allocation pattern, not the
    arithmetic.
- **Masked challenge sampler.**
  `jolt:crates/jolt-crypto/src/challenge/sampler/constant_time.rs:45`
  (`sample_challenge_ct`). It is byte-identical to the verifier's sampler. It
  replaces Fisher–Yates swaps at secret indices with full masked scans, builds
  the dense challenge by a masked scatter, and always runs the exact certified
  operator-norm evaluation.
- **`Fork::attempt`.** `jolt:crates/jolt-transcript/src/fork.rs:18,70`. It is
  a fresh sponge that absorbs the tag `jolt-transcript/fork/attempt/v1`, then σ,
  then a 32-byte nonce drawn from the prover's RNG. It never uses a counter
  and records no transcript event.

**Why.** A rejected attempt's (y, c, z, V, t_D) is correlated with the witness
through the rejection event. A timing leak of any of them defeats H2 exactly as
an observed oracle answer would.

**Argument / security impact.** No change to Theorem `wo:thm:zk`. The
constant-time rule is what makes H2's "never observed" hold against a timing
adversary. The tag makes the attempt queries a domain of the oracle disjoint
from the challenge queries and from Akita's grinding forks, which are keyed by
a `u32` counter. A counter would not do: H2 needs the nonce uniform and secret
until it is published.

**Status.** Implemented (W4, A11). Still open: the NEON/AVX NTT kernels were
not audited line by line; Intel DOITM is out of scope; whether DIT disables the
data-memory-dependent prefetcher on M4 is not checked (`whiteout-sigma.md` A11,
"DMP").

**For the paper.** Add a remark after Theorem `wo:thm:zk` that H2 assumes
the rejected attempts leave no timing trace. State the requirement that this
places on implementations: constant-time sampling, mask arithmetic and
challenge derivation, and DIT or DOITM where the core's multiply/add timing
depends on its operands. Optionally cite the M4 measurement.

## D2. Σ near-zero rejection (placeholder)

**Paper says.** α = 16, ε = 2^−137, M ≈ 2.37, p_𝓑 ≥ 0.999, seven radix-16
digits, T_att = 163, about 2.37 attempts (`02_preliminaries.tex:272`,
`06_evaluation.tex:93-107`).

**Implementation.** Reviewer steering (2026-10-21, recorded in jolt
`specs/shared-primitive-layer.md`, commit `2cc2502ed`): σ grows so that one
attempt almost always accepts. This replaces the planned parallel Gaussian
sampler as the answer to the prover-time gap (D15). The attempt loop, the
nonce and the constant-time rule stay (D1). Rejected attempts still occur, and
an accepted attempt's y = z − c·s would reveal s.

**Numbers: TBD from `wo/sigma-norej`.** *Draft*, uncommitted on that branch
(`specs/whiteout-sigma.md` §5.1), not yet reviewed:

- default D = 10 digits, α = 2^17, p_𝓑 ≥ 1 − 2^−20;
- per-attempt rejection ≤ 2^−13.2, T_att = 10;
- one common width in place of per-block widths;
- ranks 5 / 5 / 5 over R_256;
- e ≈ 90M cells (2^27), proof +9.66 KB, verifier scans +23%.

None of these numbers should go into the paper until that branch lands.

**Why.** It gives a predictable prover time of one attempt and removes the
need for a parallel sampler.

**Argument / security impact.** Theorem `wo:thm:zk` is unchanged in form; only
T_att, M and the box move. The precondition ‖cs‖_σ ≤ 1/α still holds at the
larger width. Binding radii grow with the digit count, so the ranks grow.

**Status.** In progress.

**For the paper.** Pending. The likely change is §6.2's α, widths, box, digits,
T_att, the ranks of §6.2 "Commitment ranks", and Table `wo:tab:overheads`.

## D3. One-shot fold responses at levels ≤ cut (W2)

**Paper says.** Fold responses are sized so an honest response overflows with
probability ≤ 2^−λ, and the prover aborts on overflow
(`04_protocol.tex:97-104`). Lemma `wo:lem:fold-overflow` gives a Hoeffding
bound over the challenge signs with a 1/p factor for filtered families
(`04_protocol.tex:106-131`). The evaluation adds one response digit per masked
fold and reports δ_fold < 2^−950 per fold (`06_evaluation.tex:110,143-150`).
2δ_fold enters Theorem `wo:thm:zk` (`05_security.tex:337,344-350`).

**Implementation** (*verified* at `akita-w2:`).

- **Encoding.** At each level ≤ cut, the fold-response run has
  `nonce_bits = 0`. No nonce is sent, the challenges come from the fork at
  counter 0, the response is probed once, and overflow returns
  `AkitaError::FoldOverflow { level }` (Akita spec §2.2).
- **Target.** `FoldFailureTarget::OneShot { security_bits, masked_folds }`
  (`akita-w2:crates/akita-params/src/sis/honest_fold_policy.rs:19-29`) asks for
  failure ≤ 2^−security_bits / masked_folds per fold. Here security_bits = 136
  and masked_folds = cut_level + 1 = 2
  (`akita-w2:crates/akita-params/src/zk.rs:38-52`). So δ_fold ≤ 2^−136 in total,
  and 2δ_fold ≤ 2^−135.
- **Balanced policy** (levels ≥ 1). The cap is the least t with
  t² ≥ 2·(claims·blocks)·η²_max·β²·L, where
  L = ⌈(⌈log₂ 2N⌉ + 136 + ⌈log₂ 2⌉)·71/100⌉
  (`akita-w2:crates/akita-params/src/sis/fold_linf_cap.rs:18-19,63-87`).
  - N counts every *physical* response coefficient: response windows × rows ×
    ring coefficients, times the number of witness chunks.
  - 71/100 ≥ ln 2 and ⌈log₂ x⌉ ≥ log₂ x, so L ≥ ln(2N·masked_folds·2^136) in
    checked integer arithmetic.
  - The test (`honest_fold_policy.rs:714-731`) checks the headline level-1 cap
    against the tail bound with ln 2 < 6932/10000, in exact rationals.
- **One-hot root** (level 0). The same Chernoff solve as per-attempt sizing,
  2N·exp(m log M_X(λ) − λt) ≤ target, over the λ grid 2^−12..2^2, with the
  target 2^−136/2 in place of 7/8 (`honest_fold_policy.rs:236-258`). Frozen
  headline thresholds are 564 per attempt and 1,469 one-shot
  (`honest_fold_policy.rs:733-748`).
- **Modeled cap dropped.** The planner's modeled ∞-norm cap (1/40 per
  attempt) is ignored for one-shot targets
  (`akita-w2:crates/akita-planner/src/schedule_params/candidate.rs:258-262`).
- **Admission.** `ZkOpeningSchedule::for_tests` rejects a schedule whose
  masked level has fewer fold digits than the one-shot policy derives
  (`akita-w2:crates/akita-params/src/zk.rs:120-153`). The test
  `zero_knowledge_schedule_rejects_an_unsupported_cut_or_a_short_masked_level`
  (`akita-w2:crates/akita-config/src/transcript_grinding_plan.rs:165`) removes
  one digit at each masked level and expects rejection.

**Why.**

- The paper's lemma is stated per fold, and the evaluation fixes "one more
  digit" rather than a target. The implementation fixes the target and lets
  each policy derive the digits.
- The modeled cap is a per-attempt bound and is never a one-shot bound.
- The one-hot root has a tighter exact-MGF Chernoff bound than Hoeffding.

**Argument / security impact.**

- Per fold, the balanced bound is Lemma `wo:lem:fold-overflow` with the union
  over physical coefficients made explicit. It covers both tails through the 2N
  factor, using the nearer balanced end in place of L₋ + 1 and L₊ + 1, which is
  conservative.
- The one-hot bound is a Chernoff bound on the exact moment-generating function
  of challenge × one-hot input. Any λ on the grid gives a valid bound, so
  missing the minimizer only enlarges the threshold.
- Summing over the two masked folds gives δ_fold ≤ 2^−136.
- Total, with the paper's other terms (δ_LHL = 2^−129.2, T_att = 163,
  ε = 2^−137, M = 2.37, ν = 256) and Q_tot = 2^100 (an assumption, see Q6):
  2^−128.79 when δ_fold sits at its target, and 2^−128.81 at the reported
  per-fold values. This log computed both totals by hand.
- *Reported*, not reproducible here: the per-fold bounds at the headline row
  `fp128_onehot_recursive final=36:1` are about 2^−193 (root) and 2^−155
  (level 1). The test that printed them was removed by fixup `27e64882b`.
- Correction to the caller's brief: the total is about 2^−128.8, not 2^−128.7.
  Both computations above give −128.8.

**Finding on clear catalog rows** (partly *verified*). The admission test
needs `fp128_onehot final=16:2`, "the smallest one-hot row whose masked levels
meet the one-shot target"
(`akita-w2:crates/akita-pcs/tests/zk_fold_responses.rs:31-32`). Smaller clear
rows are short at a masked level, so ZK rows must be planned with `OneShot`
sizing; they cannot reuse clear rows. At the headline row, however, the W2-d
test (`akita-w2:crates/akita-planner/src/test/schedule_params_candidate/zk_ranks.rs`)
admits the clear schedule through `for_tests`. That suggests the clear headline
digits already meet the one-shot target: at the root, both 564 and 1,469 fit
in four balanced base-8 digits ([−2340, 1755]). The claim "clear rows are one
digit short" is therefore *not verified for the headline*. It holds for rows
below `final=16:2`.

**Status.** Implemented on `wo/akita-w2`, not yet merged into Akita
`whiteout/dev`. W2-d (A ranks unchanged at levels 0 and 1) is a test; it was
not run for this log.

**For the paper.**

- §4.1 "Fold responses": state the target as δ_fold ≤ 2^−λ' in total, split
  evenly over the k masked folds, and say that N counts physical response
  coefficients.
- Lemma `wo:lem:fold-overflow`: add the Chernoff variant for one-hot inputs,
  or note that any valid MGF bound may replace Hoeffding.
- §6.2 "Randomness": replace "one more digit … below 2^−950" with the
  target-driven sizing (2^−136 in total). Note that at the headline the extra
  digit may be unnecessary (see the finding above), and that the ZK error stays
  about 2^−128.8.
- The remark at `05_security.tex:438-440` (worst-case sizing removes
  2δ_fold) stays true.

## D4. Hiding randomness (W1)

**Paper says.** Every commitment of the base argument gets fresh randomness
on its outer map, one independent vector per slice. Its digits are in the
alphabet of the witness that later contains it, and its entropy exceeds the
outer image by 2λ (`04_protocol.tex:58-79`). §6.2 uses 8 extra bits
(`06_evaluation.tex:107-109,137-143`). Precommitted advice and committed
programs are not protected (`03_setting.tex:167-173`). Def.
`wo:def:source-cutoff` forbids a cut at the last fold before the final one
(`03_setting.tex:133-139`).

**Implementation** (*verified* at `akita:`).

- **One owner.** `akita_params::zk::hiding_randomness_columns` is the only
  caller of `jolt_crypto::lattice::required_randomness_columns(image_coeffs, 0,
  ring_dim, log_basis, 8)` (`akita:crates/akita-params/src/zk.rs:18,48`).
- **Open alphabet.** B and D randomness is sized and sampled at the **open**
  alphabet of the level that consumes it, which is the alphabet its range check
  proves (`zk.rs:89-130`). It is not sized at the outer matrix's own input
  alphabet. At the headline (d_b = 64, rank 1, b = 8): 64·128 + 256 + 8 bits at
  192 bits per column, so 45 R_64 columns per slice.
- **Not at the terminal.** `validate_hiding_randomness`
  (`akita:crates/akita-params/src/zk.rs:146-152`) rejects
  `cut_level ≥ recursive_folds.len()`: "the last nonterminal fold's output is
  the terminal witness, sent in the clear, so a cut there would publish that
  fold's randomness". It also requires every level ≤ cut to carry exactly the
  derived widths, every later level and every precommitted group to carry
  none, and a hiding level to use one witness chunk.
- **Trusted catalogs reject randomness.** `ValidatedScheduleCatalog::try_new`
  rejects any schedule with hiding randomness as "not selectable"
  (`akita:crates/akita-pcs/tests/hiding_randomness.rs:74-90`). Hiding schedules
  exist only through a test-only constructor until ZK is published (W5d).
- **jolt side.** r_p and r_a are sized by the same function, with
  `extra_coeffs = 64` for the row L. That gives 105 and 97 R_256 columns
  (`jolt:specs/whiteout-sigma.md` §1.5).

**Why.**

- The open alphabet is the one alphabet the next level's range check
  certifies, so it adds no new range-check alphabet and leaves s_∞ unchanged.
- A cut at the terminal would send randomness in the clear.
- Trusted catalogs feed clear provers, which must never carry hiding columns
  that no verifier mode accounts for.

**Argument / security impact.**

- The open alphabet is "the alphabet of the witness that later contains it" in
  the paper's words, so the LHL argument (Lemma `wo:lem:stat-hiding`, H9) is
  unchanged.
- The terminal check is Def. `wo:def:source-cutoff`'s condition, enforced in
  release with the reason stated.
- Precommitted groups folded at a masked level carry the same randomness as
  the root (D24, Akita W2b); the setup-prefix slot carries none.
- The one-chunk restriction is a simplification. It keeps one randomness vector
  per slice image.

**Status.** Implemented (W1). The paper's "rank 1 kept at k = 2" (W1-d) is a
test that was not run here.

**For the paper.** In §4.1, name the alphabet: "the open alphabet of the fold
whose range check covers the witness that holds it". State that slices
already split across witness chunks are out of scope, or give each chunk its
own vector.

## D5. The cut at k = 2

**Paper says.** k ∈ {2, 3, 4} are admissible; k = 2 with offloading is
recommended (`06_evaluation.tex:169-170,237-246`).

**Implementation.**

- Only `cut_level = 1` (paper k = 2) is supported:
  `akita-w2:crates/akita-params/src/zk.rs:38,127-131`, and Akita spec §4.1.
- `masked_folds = cut_level + 1 = 2`, so levels 0 and 1 are masked.
- There is no planning for k ∈ {3, 4}.

**Why.** It is the recommended configuration, and the earlier-fold matrices
keep Akita's ranks there (`06_evaluation.tex:155-162`). k = 3 and 4 need re-ranked
inner matrices.

**Argument / security impact.** None beyond the paper's k = 2 analysis.

**Status.** Implemented.

**For the paper.** No change needed. Optionally state that the
implementation fixes k = 2.

## D6. Per-round degree schedule

**Paper says.** The masked transcript "has the length of the original one",
and every round sends all coefficients but one (`04_protocol.tex:33-52`). The
base count is 1,064 field elements of sumcheck messages at T = 2^24
(`06_evaluation.tex:66-67`). The paper is silent on batched rounds whose
members have different degrees.

**Implementation.**

- `jolt_sumcheck::DegreeSchedule::batched`
  (`jolt:crates/jolt-sumcheck/src/schedule.rs:18,50`) fixes each batch round's
  wire width at the largest degree declared by the members active in that
  round, at least 1.
- Relations declare per-round degrees through `SymbolicSumcheck::round_degree`
  (`jolt:crates/jolt-claims/src/symbolic.rs:62`). Instruction read-RAF declares
  its 128 address rounds at degree 2 instead of 10
  (`jolt:crates/jolt-claims/src/protocols/jolt/relations/instruction/read_raf.rs:90`).
- The schedule is public and witness-independent. Prover, clear verifier,
  masked verifier, `pad_layout` and the residual lowering consume the one
  derivation (commit `292b808b7`).

**Effect** (*verified* in commit message and `whiteout-masking.md:226-236`,
muldiv, T = 2^10, Dory/BN254):

- clear proof 97,663 → 64,895 B;
- masked proof 113,951 → 79,071 B;
- pads 2,073 → 1,049.

Every dropped coefficient is one fewer masked message and one fewer pad.
Stage 2's RAM read-write and stage 6a's bytecode address phases still send
zero high coefficients, worth about 0.7 KB more.

**Why.** Sending at the batch maximum padded each low-degree round with zero
coefficients. In masked mode each of those needs its own pad.

**Argument / security impact.** Def. `wo:def:algebraic-verifier` item 1 asks
for a witness-independent schedule of slots. A public per-round degree is such
a schedule. A member polynomial above its declared degree is rejected by the
prover (`RoundDegreeBoundExceeded`). A masked round off its scheduled width is
rejected by the lowering (`ZkError::RoundWidthMismatch`). Soundness is
unchanged: each member's round polynomial has at most the declared degree, so
the batched round has at most the scheduled degree.

**Status.** Implemented.

**For the paper.** In §4.1, say that the per-round slot count is the
scheduled degree of the batch round, not the batch maximum. *Unverified*:
whether the 1,064-element base count already assumes per-round degrees. If it
does not, the base proof and P shrink accordingly.

## D7. Shared stage checks over `ClaimDomain`

**Paper says.** Fold k's range-leaf and relation final checks become product
rows, with masked intermediate products where the degree exceeds two
(`03_setting.tex:305-315`). The count at k = 2 is "two auxiliary values from
the first two Akita folds" (`06_evaluation.tex:87-88`). The per-level product
count (Q3) is not stated.

**Implementation** (*verified*).

- Each stage check is written once, generic over
  `jolt_transcript::ClaimDomain` (`jolt:crates/jolt-transcript/src/private.rs:128`).
  The clear verifier, the masked verifier (rows), the pad layout (shape) and
  the prover's auxiliary values all run the same formula
  (`akita:crates/akita-types/src/stage_checks.rs:1-4`).
- **Stage-1 leaf**, claim = Σ_{k≤4} c_k·ri^k: u = ri² as a new product slot,
  then u·(c₄u + c₃ri + c₂) = claim − c₁ri − c₀ via `product_into`. That is 2
  products and 1 auxiliary slot (`stage_checks.rs:41-62`).
- **Stage-2 oracle**, claim = w·(B(w+1) + R): one `product_into` row with no
  auxiliary slot (`stage_checks.rs:75-89`).
  - B collects both w(w+1) terms: the batched virtual claim and the binary
    compression constraint.
  - R collects every term linear in w.
  - There is **no constant term**. An earlier draft of the Akita spec had one;
    fixup `62b0c4f88` corrected it.
- **Total.** 3 products and 1 auxiliary slot per masked level, so 6 products
  and 2 auxiliary slots at k = 2. This answers Q3 and matches the paper's
  "two auxiliary values".
- A PrivateCpu fault test shifts the last stage-2 round by X − X². It shows
  the stage-2 check is the only one that catches an oracle miss (commit
  `6d35999a1`).

**Why.** One owner per formula keeps the clear and masked verifiers from
diverging (Lemma `wo:lem:residual-correct` needs every clear check to become a
row). `product_into` lands a product on an existing affine value, so no slot
is spent.

**Argument / security impact.** The rows are exactly the clear checks after
substituting m = m̂ − p. The stage-2 check is quadratic in the single masked
value w with public B and R, so one product row suffices.

**Status.** Implemented (W3a/W3b on Akita `whiteout/dev`).

**For the paper.** In §3.2 or §6.1, state the per-level count (3 products,
1 auxiliary slot) and the stage-2 form w·(B(w+1) + R) = claim.

## D8. BlindFold stays curve-based

**Paper says.** Whiteout's back end is the lattice Σ-protocol plus an Akita
proof of the response (`04_protocol.tex:4-12`). Curve BlindFold appears only as
a related system with discrete-log hiding (`06_evaluation.tex:361`).

**Implementation.** The masking front end (M6b) is shared. BlindFold (M6c) is
a second `ZkBackend`. It proves the same residual relation over (pads,
residual witness) with the existing Nova fold, and it uses Pedersen and Dory:

- the pads are committed up front as ⌈P/L⌉ Pedersen rows, each with its own
  blind;
- the relaxed instance reuses those row commitments;
- stage 8 stays Dory's hiding evaluation commitment
  (`jolt:specs/whiteout-masking.md:10-20,59-68,131-143`).

Per-round Pedersen commitments are deleted. Reviewer steering, commit
`2cc2502ed`: "BlindFold stays curve-based over committed pads".

**Why.** It retires the per-round design, and it exercises the masking front
end end to end before the lattice back end exists.

**Argument / security impact.** This is a different theorem from Whiteout's.

- Binding of the pad commitment is computational (DLog).
- Hiding is perfect: Pedersen with independent row blinds, one-time pads, and
  a uniform folded witness.
- It is neither post-quantum nor statistical in the binding sense.
- The products are native R1CS gates, so it needs no constant-term lift and no
  scalarity rows (`whiteout-masking.md:176-197`).

The masked-transcript part (pads before the first challenge, decoded messages
fixed before each challenge) is the paper's §4.1 verbatim.

**Status.** Implemented (M6b/M6c).

**For the paper.** Optional. A sentence in §4.1 or §7 noting that the masking
step is back-end-agnostic, with curve BlindFold over the same pads as a
discrete-log instantiation.

## D9. Image compression (W5c) and raw images

**Paper says.** u_a, t_p and t_D are sent compressed. Binary decompositions ξ
in e link them to the raw images (`04_protocol.tex:305-313,329-330`). The
evaluation charges 416 B for three compressed commitments plus the nonce
(`06_evaluation.tex:188-190`).

**Implementation.**

- W4 and W5a send t_p (768 coefficients), t_D (64) and u_a (768) raw. That is
  (768 + 64 + 768)·16 B = 25,600 B, against the paper's 3 × 128 B
  (`whiteout-sigma.md` §2.5, A12; Akita spec §5.4).
- W5c lands in RP4 (commit `fe57e2446` and reviewer steering). Images above
  Akita's 512-coefficient (8 KiB) cap are split into ⌈len/512⌉ chunks, each
  compressed by the existing two-map chain.
- Since O5 (D34) the chunks hold up to the first map's certified SIS
  bound, 15,115 coefficients at q128; an image takes the fewest chunks,
  balanced. The 8 KiB cap still bounds clear B/D images.

**Why.** Akita's compression source is capped at 512 Fp128 coefficients
(`akita-params/src/compression.rs:19`, per `whiteout-sigma.md` A12), and the
images exceed it.

**Argument / security impact.** None for hiding: the compressed forms are
functions of the raw images, which H4 and H8 already replace by uniform ones.
Soundness: the split chain needs a norm-one-kernel argument per chunk, as in
Lemma `wo:lem:compressed-binding`.

**Status.** Implemented in W5c (`akita:crates/akita-pcs/src/zk/compression.rs`
on `wo/akita-w5`; jolt `ImageCompression`). The verifier checks `t = G₁ξ₁`
inside Items 2 and 3, coefficient by coefficient. Each chain equation
`K₁ξ₁ − G₂ξ₂ = 0`, `K₂ξ₂ − cm = 0` is one row: its negacyclically reduced
residual evaluated at the product challenge `α`, as Akita's
`ReducedEvaluation` relation evaluates its own maps
(`ReducedCompressionRelationWeights::for_standalone_chains`). A nonzero
residual has degree at most `d − 1`, so a violated row survives with
probability at most `(d − 1)/q` (`d = 16`: `15/q`); `α` is drawn after `e`
and every `cm` are fixed. The binary constraint `ζ·1_{I_bin}` is
load-bearing: the prover chooses `cm`, and a carry inside `ξ₁` (bits
`(0, −1)` → `(−2, 0)`, then `ξ₂` and `cm` recomputed) passes everything
else (Akita spec §5.7).

**For the paper.** Q4 (closed by D34): Whiteout's chunk bound is the
first map's certified SIS width, 15,115 coefficients at q128, against the
cost model's `FCHAIN_MAX_IMAGE = 884`; t_p and u_a (at most 1,280
coefficients) take one chunk. Price the split chain only for images above
the bound, and state the rows
as reduced evaluations at `α` (or the paper's coefficient rows, at the cost
of `d` rows per map).

## D10. Akita #255: public retry of the fold-response nonce

Source: Akita `specs/whiteout-akita.md` §2.5 at commit `900ed6db4` (§2.5.1–2.5.6).

**Paper says.** Fold responses are never retried; the prover aborts on
overflow (`04_protocol.tex:97-104`, `05_security.tex:57-58,70`). The paper
does not consider a challenge filter whose candidate pool can run out.

**Implementation (as argued).**

- **Setting.** Akita #255 (protocol v9) draws each fold group's challenges
  from one squeezed message. For the two operator-norm families a message can
  run out of accepted candidates (event E), and the clear prover then moves to
  the next fold-response nonce. #255's clear loop retries on E ∨ R, where R is
  a rejected response probe, and R reads the witness.
- **`PublicRetry`** (§2.5.2). At an operator-norm masked level, try counters
  in order and skip a counter only on E. Decode every group of a candidate
  before probing any response. Probe once at the first counter n* where all
  groups decode; on R, abort with `FoldOverflow` and never retry. Unfiltered
  masked levels keep `nonce_bits = 0` (D3). No transcript byte, domain or
  verifier rule changes; an operator-norm masked level sends a 1–2 byte nonce.
- **Lemma 2.5a** (public retry is simulatable). E is a deterministic function
  of the public sponge state, the counter and the oracle, so n*, the grind
  nonce and the challenges are a public function F(σ, H). Theorem `wo:thm:zk`
  holds with δ_fold replaced by δ'_fold (Lemma 2.5b) and Q_tot replaced by
  Q_tot + ΔQ. In H1 the accepted challenges at n* are uniform on the filtered
  family, given the history; the simulator evaluates F on its own state.
- **Lemma 2.5b** (one-shot overflow under an operator-norm family).
  δ' = min(δ_H, δ_B):
  - tuple form δ_H: the paper's lemma with 1/p^ℓ, since all ℓ challenges must
    pass the filter (D22);
  - Bernstein form δ_B: variance proxy ℓη²β²/p and bound κ₁β per term. It
    uses the filtered family's symmetry under c ↦ −c and
    E_f[Z_i²] ≤ E_S[Z_i²]/p.
- **Costs** (§2.5.2–2.5.3, marked inference there, not sized on any
  schedule).
  - Range: about +½·log₂(1/p) bits from the Bernstein form, +1.47 bits for D64
    (certified p = 0.1305) and +0.72 bits for D128 (p = 0.3694). That is at
    most one more radix-16 digit.
  - ΔQ ≤ 2·2^12·(2^17 + 2^8) < 2^30.1, so the added query term is
    T_att·ΔQ/2^256 ≈ 2^−218.5.
  - The one-shot target and T_att are unchanged. PoolExhausted (≤ q^N) is a
    completeness loss that the simulator reproduces, not a ZK term.
- **Soundness** (§2.5.4). The verifier is #255's, unchanged. Each candidate
  is charged by the fold proof of work, so the query-counted bound of
  `wo:lem:randomized-base` is unchanged.

**Why.** It lets an operator-norm family be used at a masked level without
leaking the witness through the accepted counter.

**Argument / security impact.** As above. ZK error grows by about 2^−218.5
plus the δ' − δ difference, which is absorbed by the unchanged 2^−136 target.
This log recomputed the headline total with the ΔQ term at about 2^−128.8.

**Relevance.** The spec states that the headline masked levels use the
unfiltered family, so they never run out. D19 questions that for level 1
(d = 512 challenges); if that family is filtered, this entry applies at the
headline.

**Status.** Argued, under independent review. The implementation lands at
the port onto #255 (§2.5.5), with tests W2-e … W2-h (§2.5.6). Until then
`ZkOpeningSchedule::new` rejects an operator-norm family at a masked level.

**For the paper.** In §4.1 "Fold responses", say that a retry is allowed
when its trigger is a public function of the transcript and the oracle, and
that a retry on the response is not. Add Lemma 2.5b, or fix the 1/p clause
(D22). Add ΔQ to the query count in Theorem `wo:thm:zk`.

## D11. Uni-skip rounds sent in full

**Paper says.** In uni-skip rounds the constant coefficient, of weight |D|,
is omitted and solved symbolically, like the linear coefficient of a standard
round (`04_protocol.tex:47-49`).

**Implementation.** The two uni-skip rounds send **every** coefficient masked.
The residual adds two rows per round: Σ_D f = input claim, and the round's
output-claim check
(`jolt:crates/jolt-verifier/src/stages/zk/residual/mod.rs:497-540`,
`round_format(domain, true)`; `whiteout-masking.md:27-28,100-101`).

**Why.** Clear Jolt sends uni-skip rounds uncompressed, and the masked
transcript keeps the clear slot positions (`whiteout-masking.md:24-25`).

**Argument / security impact.** It is sound: the explicit sum row is the clear
check. It costs one extra pad, one extra masked field element and one affine
row per uni-skip round. The paper's version is strictly smaller.

**Status.** Implemented.

**For the paper.** None required, since the paper's form is valid and smaller.
It is a possible implementation improvement (2 elements).

## D12. Auxiliary slots and their position

**Paper says.** Each multiplication gate's output is "an existing slot or a
new masked intermediate value, sent … after its inputs are available and
before any challenge that uses it" (`03_setting.tex:194-205`). At T = 2^24 the
model counts 28 intermediate products, plus 2 Spartan affine operands, plus 2
Akita values: 32 elements, 512 B (`06_evaluation.tex:83-88,188`).

**Implementation.**

- **One slot per product.** jolt keeps one auxiliary slot per symbolic
  intermediate product, allocated per factor and term and never skipped for a
  zero coefficient (`jolt:crates/jolt-zk/src/masking/lowering.rs:1-15`;
  `whiteout-masking.md:104-112,122-126`). It does not land a stage's last
  product on its running claim. Akita's checks do use `product_into` (D7).
  - Muldiv, T = 2^10: 108 auxiliary slots, 106 product rows, 16 affine rows;
    the auxiliary block is 3,456 B (`whiteout-masking.md:210-217`).
- **Position.** The block is sent after stage 8 (BlindFold) or after Akita's
  masked levels (Whiteout): `whiteout-masking.md:126-129` (A1); Akita spec §7,
  order item 6.

**Why.**

- Landing products on running claims would change every ZK proof's layout.
- Auxiliary-as-witness is listed as an open reduction, worth about 3.3 KB on
  BlindFold.

**Argument / security impact.** The position is sound because the auxiliary
slots form an acyclic chain, each defined by one product row or defining row
over earlier slots, and no base-protocol challenge depends on them. That
satisfies the paper's "before any challenge that uses it" trivially. The extra
slots cost bytes, not security.

**Status.** Implemented.

- Count *unverified* at T = 2^24 for the Whiteout path: the muldiv count is at
  T = 2^10 on BlindFold.
- The gap between the paper's 32 and jolt's 108 comes from the "existing
  slot" reuse and from shared repeated products. Both are implemented in D26,
  which brings muldiv to 26 slots on BlindFold.

**For the paper.** Either state that a gate's output reuses the stage's
running claim where possible, so the paper's count is a lower bound that the
implementation does not yet meet, or recount. Note that the auxiliary block
may follow the whole base protocol.

## D13. Σ oracle input and transcript order

**Paper says.** Attempt i queries x_i = (τ, cm(t_D^{(i)}), n_i)
(`05_security.tex:314-317`). Protocol `wo:prot:whiteout` gives the step order
(`04_protocol.tex:509-527`).

**Implementation.**

- **Oracle input.** After ρ, both roles squeeze a 32-byte σ from the live
  transcript. The attempt fork absorbs (tag, σ, n_i), then the raw t_D, and
  squeezes c (`whiteout-sigma.md` §2.4; `jolt:crates/jolt-zk/src/whiteout/sigma.rs:291-318`).
  After W5c the input becomes (σ, cm(t_D), n) (Akita spec §5.4).
- **Whole order** (Akita spec §7):
  1. preamble and hiding commitment;
  2. t_p;
  3. Jolt stages 1–7, masked;
  4. stage-8 claims to Akita;
  5. Akita levels 0..=cut, masked;
  6. the auxiliary block (Jolt's, then Akita's);
  7. ρ, σ, t_D, n;
  8. the response proof.
- **Hiding commitment first.** The hiding root commitment precedes t_p.
  Both precede every challenge.

**Why.** The fork must live outside the transcript, so a rejected attempt
leaves nothing in the replay log. σ carries τ into the fork.

**Argument / security impact.** In the ROM, (σ, t_D, n) determines
(τ, t_D, n) except on a σ collision. That adds at most Q_tot²/2^257 to Pr[Bad]
in H2, which is negligible beside the existing (T_att·Q_tot + T_att²)/2^256.
Raw t_D determines cm(t_D).

**Status.** Implemented (raw t_D); compressed input in W5c.

**For the paper.** Optional: allow x_i = (H(τ), t_D or cm(t_D), n_i).

## D14. Gaussian widths, digits and randomness rounding

**Paper says.** σ_b = αΓγ_bω_b^{−1/2} with optimized ω_b; seven radix-16
digits (`06_evaluation.tex:97-102`). Randomness of 24,704 and 26,752 entries
for r_a and r_p (`06_evaluation.tex:141-143`).

**Implementation** (`whiteout-sigma.md` §5, A10).

- **Widths.** They use the closed-form optimum ω_b = N_b/ΣN with integer
  k_b = ⌈αΓ√(2 ln 2·γ_b²·ΣN/N_b)⌉ + 1, then an exact fixed-point check
  Σ_b 2·LN2_HI·Γ²γ_b²α²/k_b² ≤ 1.
- **Digits.** The smallest D with ⌊L(D)²/k_b²⌋ ≥ 1 + ⌈log₂(2000·N_tot)⌉,
  which gives p_𝓑 ≥ 1 − 1/2000. Every Gaussian block gets 7 digits. The
  script's discrete optimum gives 6 to r_a, the pads and r_p. The difference
  is +111,968 cells of e (65,450,122 → 65,562,090, still below 2^26).
- **Randomness.** r_a = 97 and r_p = 105 R_256 columns (24,832 and 26,880
  cells). The script rounds to R_64.
- **Box.** L_b is the full positive digit envelope 7(16^D − 1)/15 (A9).

**Why.** Prover and verifier must agree bit for bit on D_b, so the widths are
derived in integer arithmetic rather than tabulated. The residual matrices act
on whole R_256 elements.

**Argument / security impact.** The exact check is ‖cs‖_σ ≤ 1/α for every c
and every s that passes the alphabet and shape check, which is the hypothesis
of Theorem `wo:thm:joint-rejection`. There is no change to proof bytes or the
padded domain.

**Status.** Implemented. D2 will supersede the widths.

**For the paper.** Optionally state the integer width rule and the digit
rule. Note that the six-digit blocks are an optimization the implementation
does not take.

## D15. Gaussian sample cost

**Paper says.** A Gaussian sample is assumed to cost 10 ns, giving +1.15 s
(4.7%) of prover time (`06_evaluation.tex:200-203,249-253`).

**Implementation** (*measured*, `whiteout-sigma.md` A16; Apple M4 Max,
16 threads):

- M5a's isochronous sampler takes 0.41 µs per coefficient, on one RNG stream.
- The attempt is serial-bound: 0.221 s of 0.227 s is `fill` at 538,368
  Gaussian coefficients.
- Extrapolated to the headline: about 2.6 s per attempt and 6.2 s per proof at
  2.37 attempts.
- The Σ-protocol's own work is about 0.08 s per attempt.

**Why.** A constant-time BLISS-style sampler with 192-bit fixed-point
acceptance (`whiteout-gaussian.md` §2) is far slower than the model's 10 ns.

**Argument / security impact.** None. It is a cost.

**Status.** Measured. The planned fix (a parallel `fill`) was replaced by D2.

**For the paper.** Replace the 10 ns assumption with a measured constant-time
cost, or state that the 10 ns figure needs a parallel sampler. Under D2 the
attempt count drops to about 1, but the per-sample cost grows with the
exponent length ℓ.

## D16. Sampler precision and prover coins

**Paper says.** Finite-precision sampling "adds to this error"; no value is
given (`06_evaluation.tex:110-112`). The statement is statistical
(`05_security.tex:431-434`).

**Implementation** (`whiteout-gaussian.md` §2).

- **Precision.** Per output, the error is within 2^−183.3. Over
  6.2·10⁶ × 163 ≈ 2^29.9 samples it adds at most 2^−149 to the 2^−128.8 bound.
  Every constant is computed at 320 bits and floored to 192.
- **Coins.** The prover's coins come from a `CryptoRng`. The design note's
  per-chunk ChaCha20 expansion would add an explicit PRG assumption. In the
  current code ChaCha20 appears only in tests (`jolt:crates/jolt-crypto/src/sample/`).

**Argument / security impact.** +2^−149 to Theorem `wo:thm:zk`. ZK is
statistical relative to ideal prover coins; with real coins it is relative to
the CSPRNG.

**Status.** Implemented.

**For the paper.** Fill in the precision term (2^−149), and add a sentence
that ZK is stated for ideal prover coins.

## D17. Σ challenge budget and ⊥ after T_att

**Paper says.** c is uniform on the filtered family 𝓒. After T_att rejections
the simulator outputs ⊥ (`05_security.tex:426-428`). The paper does not say
what the prover returns.

**Implementation** (`whiteout-sigma.md` §2.6, A14).

- **Budget.** Exhausting the operator-norm candidate budget (4,096 tries,
  probability about 2^−1580) is fatal on both sides, never retried.
- **After T_att.** `WhiteoutError::AttemptsExhausted` is returned after T_att
  rejections (`jolt:crates/jolt-zk/src/whiteout/sigma.rs:604-628`). The caller
  may rerun the whole proof with fresh randomness. W6 bounds this rerun at
  3 runs inside `jolt-prover` (D23).

**Argument / security impact.**

- Conditioned on the budget not running out, c is uniform on 𝓒. Exhaustion is
  a public function of the oracle input.
- A rerun is a fresh execution. By H3, the all-rejected event is
  witness-independent, so the rerun reveals nothing about the witness. The
  simulator matches it by outputting ⊥ at the same rate.

**Status.** Implemented.

**For the paper.** State the ⊥ behaviour (A14), and that budget exhaustion
of the filter is a public abort with negligible probability.

## D18. Structural constraints on masked levels

**Paper says.** Def. `wo:def:source-cutoff` (`03_setting.tex:133-139`): no
response norm is checked in the clear up to k, and fold k passes no setup
claim. §2.1 restriction of scalars is used in `wo:lem:sigma-extract`.

**Implementation** (Akita spec §4.1, `whiteout-sigma.md` §1.7, A7, A20).

- Every level ≤ cut uses `InnerCommitSecurityRoute::Linf`, so no L2 norm
  message is sent; `response_l2_sq` falls after the cut.
- E::DEGREE = 1, else `Unsupported`.
- The cut level has no Stage 3. Stage-3 claims at level 0 are public (they
  depend only on setup and public challenges).
- A hiding level uses one witness chunk (D4).
- The residual commitment has d_A = d_B. `SigmaSetup` requires one native
  dimension for every block of N (R_256 at the headline).

**Why.** d_A ≠ d_B, as in Akita's consecutive-coefficient split, only
relabels coefficients. It is not an R-module map, and Lemma
`wo:lem:sigma-extract` cancels c, which needs N to be R-linear.

**Status.** Implemented (W4); specified for W3 (`ZkSchedule::validate`).

**For the paper.** State d_A = d_B (or the stride embedding) as a hypothesis
of §4.2 (A7, Q2). Note that the masked levels use the ∞-norm route.

## D19. Headline fold shapes at the masked levels

**Paper says.** "The folds before fold 5 use Akita's unfiltered challenge
family in dimension 64 (η² = 71)" (`06_evaluation.tex:147-149`).

**Implementation** (*verified* in test fixtures).

- W2's headline fixtures for `fp128_onehot_recursive final=36:1` have level 0
  at ring dimension 512 with a 64-dimensional challenge subring (31 × ±1,
  10 × ±2, η² = 71).
- Level 1 is at ring dimension 512 with **512-dimensional** challenges
  (19 × ±1, η² = 19)
  (`akita-w2:crates/akita-params/src/sis/honest_fold_policy.rs:615-650,714-718`).
- The Akita spec §2.3 says "folds at or before the cut use the unfiltered
  family (d = 64, η² = 71), so the lemma's 1/p factor is 1".

**Contradiction.** The Akita spec §2.3 disagrees with the W2 fixture at
level 1. *Unverified*: whether the d = 512 family is filtered (that would
bring back Lemma `wo:lem:fold-overflow`'s 1/p factor), and whether the paper's
schedule predates the current catalog.

**Status.** Open.

**For the paper.** Re-derive §6.2 "Randomness" and "Earlier folds" from the
current headline row, or pin the schedule the model uses.

## D20. Setup offloading of the response proof

**Paper says.** The recommended configuration offloads the residual and
first-fold matrices of the response proof in one sumcheck, about 1.2 KB, and
the verifier's matrix work grows by under 1% (`04_protocol.tex:493-504`,
`06_evaluation.tex:226-246`).

**Implementation** (Akita spec §5.3, A30). W5a evaluates σ_N by a direct scan.
W5b (offload into the response root's Stage 3) lands in RP4 only if the direct
scan exceeds the verifier budget measured in W5-e.

**Status.** Optional; not built (2026-10-22). At the headline N shapes the
direct scan, parallel over blocks and column chunks, measures 114 ms on one
thread and about 10–14 ms idle on 12 cores; the matrices' field copies are
built once per verifier setup. Forcing the root offload and growing the
setup prefix to cover N (2^22) costs about +2.9 KB of proof (planner
estimate 70,995 → 73,879 B, Stage 3 736 B) and 3.5M more setup field
elements. The paper estimates about 1.2 KB. Building it needs a
response-family planner policy that forces a root offload with a minimum
prefix (typed error if impossible), N's matrices as Akita setup views with
A_p and L allocated through Akita, and N's weights as eq-pair tensor families
in the root's Stage 3 (Akita spec §5.8).

**For the paper.** The measured verifier corresponds to the paper's "without
offloading" rows. Our offload would cost about 2.9 KB, against the paper's
1.2 KB.

## D21. Open author questions

Unresolved unless marked.

- **Q1, pads instead of randomness below the cut** (Akita spec §11).
  - Proposal: below the cut, replace a commitment's LHL randomness by 8 pads
    on its compressed form.
  - Conditions: the image is revealed only through ξ digits inside masked or
    Gaussian-hidden witnesses, the compressed form is sent after t_p, and its
    8 coefficients take pads.
  - The root keeps randomness, because it is made at commit time and may be
    reused across proofs.
  - Saving: W1 shrinks from about 2.8k coefficients per slice to 8 pads per
    commitment.
  - The reviewer thinks it plausibly sound. **Unresolved.**
- **Q2.**
  - **A7**, d_A = d_B: **unresolved** (D18).
  - **A9**, the box is the full positive digit envelope: **unresolved** (D14).
  - **A14**, ⊥ after T_att: **unresolved** (D17).
  - **A8**, L·r_p := π₀(Σ_j a_j r_{p,j}) as row `rank` of the pad view, with
    [A_{p,r}; L] uniform because a ↦ π₀(a·) is an F-linear bijection
    (`whiteout-sigma.md` §1.6): the reviewer finds it fine; author
    confirmation pending.
- **Q3.** Answered: 3 products and 1 auxiliary slot per masked level (D7).
- **Q4, compression above 8 KiB.** The cost script's `FCHAIN_MAX_IMAGE = 884`
  disagrees with Akita's 512-coefficient cap. Should the model count the split
  chain? **Unresolved** (D9).
- **Q5.** Answered: the paper's 2^−950 is per fold. Superseded by D3.
- **Q6, Q_tot.** `05_security.tex:317` defines Q_tot, but §6.2 gives no value.
  The removed W2 test assumed 2^100, giving a term of about 2^−148.6. The D2
  draft assumes 2^64. **Unresolved**: the evaluation should state the assumed
  bound.
- **Q7, the 1/p clause.** See D22. **Unresolved.**
- **W6-Q5, bounded rerun.** Is the rerun delta acceptable as argued? See D23.
  **Unresolved.**

## D22. Q7: the 1/p clause of the fold-overflow lemma

**Paper says.** "If the challenges are drawn conditioned on a public event of
probability at least p, such as an operator-norm filter, the bound is
multiplied by 1/p" (`04_protocol.tex:117-118`).

**Issue.** Akita `specs/whiteout-akita.md` §2.5.2 ("Paper wording") and §11
Q7 (commit `900ed6db4`) point out a problem. The clause is correct only when p
is the probability of the event on the whole tuple (c_1, …, c_ℓ). With a
per-challenge filter of rate p, the tuple event has probability p^ℓ, so the
bound must be multiplied by p^{−ℓ}. That costs ℓ·log₂(1/p) bits in the
exponent: 2.94 bits per challenge for D64 and 1.44 for D128.

**Fix offered.** The Bernstein form of Lemma 2.5b (D10) costs only about
½·log₂(1/p) bits of range for any ℓ, and it wins for large ℓ.

**Security impact.** None on the paper's parameters: its masked folds are
unfiltered. D19 asks whether the current headline level 1 still is.

**Status.** Open, for the authors.

**For the paper.** Replace "an operator-norm filter" by "an event on the
whole challenge tuple", note that a per-challenge filter gives p^ℓ, and
optionally add the Bernstein variant.

## D23. Bounded rerun instead of abort (W6)

**Paper says.** On fold overflow the prover aborts (`04_protocol.tex:100-104`;
`05_security.tex:70`). H1 charges δ_fold (`05_security.tex:344-350`). After
T_att rejected Σ attempts the prover outputs ⊥ (D17).

**Implementation** (`jolt:specs/whiteout-jolt-integration.md` §6 and "Paper
deltas", commit `9d93a7034`).

- **Triggers.** `jolt-prover`'s ZK entry retries on exactly two errors:
  `ZkError::FoldOverflow` and Σ's `AttemptsExhausted`.
- **Rerun.** The whole proof is rerun from stage 0 with fresh hiding
  commitments, pads, g, r_p and Σ coins, drawn from the same `CryptoRng`
  stream with no reseed or counter. Only the trace and witness are reused.
- **Bound.** At most 3 runs, then the error is returned.
- **Disposal.** Aborted runs are dropped and never serialized.
- **Owner.** The loop lives inside `jolt-prover`, not the caller, so that a
  caller cannot reuse an RNG seed (W6 "Tradeoffs").

**Argument / security impact** (W6 "Paper deltas").

- Each rerun is a fresh, independent execution, and only the accepted run is
  output. The output distribution is the real prover's conditioned on no
  abort in that run, which is within 3·δ_fold of the H1 prover's distribution.
- The Σ-exhaustion trigger is witness-independent by H3, so it adds nothing.
- Soundness is unaffected: reruns only discard executions
  (`05_security.tex:111-112`).
- At the D3 target δ_fold ≤ 2^−136, 3·δ_fold ≤ 2^−134.4. This log's recomputed
  total, with this term added to the 2δ_fold of Theorem `wo:thm:zk`, is
  2^−128.76.
- No proof was checked here. W6's text says both "reruns at most twice" and
  "at most 3 attempts"; these agree, since the count is 3 runs in total.

**Status.** Implemented (W6d): `jolt_prover::akita::prove` on the Whiteout
build, `WHITEOUT_ATTEMPTS = 3`, retrying on `ZkError::is_rerunnable`. Both
OS streams (Akita's hiding randomness, jolt's pads/g/r_p/Σ coins) draw
fresh. HT3 (jolt-verifier `soundness::whiteout`) aborts the first attempt
after its masked levels through a `prover-fixtures` fault seam: the rerun's
hiding commitment, t_p and every masked message differ from the aborted
attempt's, and its proof verifies. The seam forces the abort; it does not
produce a real fold overflow.

**For the paper.** State the prover's behaviour on abort as a bounded
whole-proof rerun with all randomness fresh, and add r·δ_fold for r runs to
Theorem `wo:thm:zk`. Alternatively, keep "abort" and note that a rerun costs
r·δ_fold.

## D24. Precommitted groups that carry witness data

**Paper says.** Advice committed during proving is witness data. Only
precommitted, trusted advice and committed programs are outside Whiteout's
protection (`03_setting.tex:167-173`).

**Implementation** (Akita W2b, `akita:specs/whiteout-akita.md` §3.5).

- Akita W1 gave every precommitted group zero randomness. Jolt's per-proof
  auxiliary groups, **untrusted advice, `FieldInc` and `FusedInc`** (the
  RV64 increments), enter Akita as precommitted groups but carry witness data
  (`J8:crates/jolt-prover/src/akita/stage8.rs:1-4`), so their outer images
  would have been published without hiding.
- `ZkOpeningSchedule::admit` now requires every precommitted group folded at
  a masked level to carry exactly `b_randomness_columns` per B slice, the
  root's width and alphabet. The width is part of the frozen profile, so Jolt
  commits each such group under a profile taken from a ZK row of its shape.
  `CpuBackend::commit` with `GroupContext::explicit` on a hiding profile
  samples fresh randomness (alphabet b = 8, the only one a masked level
  opens at) and keeps it in the handle, which places it in level 0's output
  witness.
- Each precommitted group's one-shot fold digits are sized by the universal
  balanced policy on its inner digit alphabet, without its producer contract,
  and δ_fold is split per witness group:
  `masked_folds = (cut_level + 1) · W_ℓ`.
- Akita cannot tell a per-proof group from a statement-fixed one, so the rule
  covers trusted advice and committed programs too: they must be committed
  hiding when a ZK row consumes them, and the prover keeps their randomness
  with its preprocessing.

**Why it was a gap.** "Precommitted" in Akita is a layout notion: a group
committed before the opening. The paper's exclusion is a trust notion: data
committed outside the proof. The two coincide for trusted advice and committed
programs, but not for these groups.

**Security impact.** None remaining: each precommitted image is hidden by
Lemma `wo:lem:stat-hiding` as the root's is. The rule is stricter than the
paper for statement-fixed groups.

**Status.** Closed by Akita W2b; Jolt commits and opens these groups hiding
since W6f (D31).

**For the paper.** In §3.1 "Advice and committed programs", state the rule
by provenance: every group committed during proving gets randomness,
whatever its place in the PCS's batch layout. An implementation that cannot
see provenance may hide every group, as Akita does.

## D25. Statement binding under ZK (#254)

**Paper says.** The transcript binds the statement and the schedule (Def.
`wo:def:algebraic-verifier`); the claims passed to Akita are masked affine
forms (`06_evaluation.tex:78-83`).

**Implementation** (`jolt:specs/whiteout-jolt-integration.md` §8). Akita #254
argues that Akita's soundness "does not depend on what the caller absorbed",
because the root absorbs the claimed values. Under ZK the values go through
`bind_derived_base`, which absorbs nothing, so W6 replaces the argument:

- each claim is an affine form y = Σ aᵢ(m̂ᵢ − pᵢ) + b;
- its m̂ᵢ were absorbed before any challenge that depends on y;
- the aᵢ and b are functions of the statement, the commitments and earlier
  challenges;
- p is fixed by t_p, absorbed before the first challenge.

So a second claim vector that verifies under the same challenges either
changes an absorbed message, which changes the challenges, or opens t_p to a
different p, which breaks A_p's binding (Lemma `wo:lem:residual-correct`).
Digests, points and the descriptor stay absorbed, and the zk bit binds the
mode on both sides. #254's caller contract gains one item: in ZK mode Akita's
root runs only after t_p and every masked slot its claims reference.
`MaskedGroupClaim` enforces this by type.

**Status.** Implemented (W6d for the trace group, W6f for precommitted
groups, D31): stage 8 passes every group's claims to Akita as
`MaskedGroupClaim` forms and absorbs no evaluation. What binds the final-opening claims is not a row a shift cannot
satisfy. jolt-verifier's `soundness::whiteout::shifts` probe, on muldiv's 56
final-opening claims with every auxiliary value recomputed as a consistent
cheater would, finds:

- Jolt's rows see one combination of the claims (a 55-dimensional blind
  space).
- Akita's rows apply exactly one functional to the claims, and its
  coefficients are the EvaluationBatch challenges, which are drawn after t_p
  and every masked value.
- A shift in that functional's kernel passes every row under the drawn
  challenges.

So the argument is Schwartz–Zippel over the EvaluationBatch coefficients,
one per claim, each a small challenge: 16 squeezed bytes reduced mod
q = 2^128 − 4,294,944,759, so each residue has at most two preimages and
any one value is drawn with probability at most 2·2^−128. A nonzero shift
fixed before they are drawn is one linear condition on them, so it lies in
their kernel with probability at most 2^−127; the shift is fixed by then
because t_p and the masked values are absorbed first. It also needs the residual u_a
bound (§8, A41). This log did not check the argument.

**For the paper.** None required; the paper already binds claims through
the masked messages and the pad commitment. A sentence in §4.1 that the PCS
receives claims as affine forms over committed pads, never as absorbed values,
would make it explicit.

## D26. Multi-pair product rows and merged products

**Paper says.** A product row is ct(X_i·Y_i − Z_i) with one pair per row
(Def. `wo:def:residual`, `03_setting.tex:236-247`). Each multiplication
gate's output is an existing slot or a new masked value (`:194-205`).
The evaluation counts 28 product values and 2 affine operands at
T = 2^24 (`06_evaluation.tex:83-88`).

**Gap.** The count of 28 assumes two things the relation cannot express with
single-pair rows:

- every term's last product lands in its stage check, Σ_m β_m · output_m =
  running claim, which is a *sum* of products;
- repeated products are shared.

Landing several products on one claim needs a row Σ_k X_k Y_k − Z. The Σ
product reduction (`04_protocol.tex:340-390`) already handles it, since it
sums μ_i X_i Y_i over an index set, but the text never states it.

**Implementation** (`whiteout-masking.md`, "Multi-pair product rows and
merged products").

- Product rows carry pairs: Σ_k x_k·y_k = z, with every x_k a constant
  operand.
- The Σ lift flattens pairs into product entries that share their row's μ.
- BlindFold splits a row into n constraints with n − 1 witness-only
  intermediates.
- jolt-claims factors the bytecode read-RAF cycle output as
  (Σ_t c_t)·ra₀ra₁ra₂.
- BlindFold takes affine operands directly; the lattice path keeps the 2
  defining slots.

**Argument.** Lemma `wo:lem:ct-lift` applies pair by pair: a constant X_k
gives ct(X_k Y_k) = x_k ct(Y_k), so the row lifts exactly. The μ-aggregation
is a combination of rows with distinct labels, unchanged. The product
sumcheck's soundness does not depend on how its entries group into rows. The
decoded multi-pair row is the clear verifier's own stage check, so
Lemma `wo:lem:residual-correct`'s decoding argument is unchanged. ZK: the row
shape is public, and fewer pads reveal nothing more.

**Status.** Implemented on `wo/multipair`. On muldiv at T = 2^10 the
auxiliary slots drop from 108 to 26, and the masked proof drops from
79,071 B to 76,447 B. The lattice path keeps the 2 defining slots, for 28.
The full table is in `whiteout-masking.md`. On the lattice path the count
matches the paper's accounting: 26 product values plus 2 affine operands,
against the paper's 28 + 2 at T = 2^24 (`06_evaluation.tex:83-88`). The
product count at T = 2^24 has not been measured.

**For the paper.**

- State the product rows as ct(Σ_{k∈K_i} X_k Y_k − Z_i), with U indexed by
  pairs and U_e = μ_{row(e)}[X_e](α).
- Say that a stage's last products land in its claim check.
- Say that the 28 assumes merged repeated products.

## D27. Soundness ledger of the response proof

**Paper says.** The 128-bit headline (`06_evaluation.tex`) covers
zero-knowledge, rejection sampling and fold overflow. The response proof's
challenge rounds (`04_protocol.tex:339-506`) have no stated total.

**Ledger.** Over `F_q`, `q ≈ 2^128` (A7F7), at `ν = ⌈log₂|e|⌉`, `m` batched
rows and `n_×` product pairs. Each round's error is the probability that a
false statement survives it; computational binding of the Akita commitment
and Akita's own opening error are separate.

| Round | Check | Degree / structure | Error |
|---|---|---|---|
| α | Item 4's product lift `Σμ_i X_iY_i − H_e = θ·(X^64 + 1)` at `α`, with the chain rows (D9) at the same `α` | the lift's residual has degree ≤ 126 in `X` (products of degree < 64); a chain row's ≤ 15 | `126/q`, no union: every violated row must vanish |
| product sumcheck | `Σ_i U_iW_i = t_×` | degree 2, `⌈log₂ n_×⌉` rounds | `2⌈log₂ n_×⌉/q` |
| η | row batch `⟨A, e⟩ = b_η` | multilinear in `η` | `⌈log₂ m⌉/q` |
| τ_pad | padding term `Σ_{j ≥ |e|} eq(τ_pad, j)·e_j` (Akita spec §5.7, W5-P1), added to the η batch | multilinear in `τ_pad`; jointly with `η`, one multilinear polynomial in disjoint variables | `ν/q` |
| τ and Stage 1 | digit range at radix 16 | Akita's stage-1 reduction, with its grinding | Akita's stage-1 error at `log_basis = 4` (akita spec §5.2; not restated here) |
| γ, ζ | range image and binary constraint into `B` | linear in each | `2/q` |
| binary constraint | `Σ_j eq(τ, j)·1_{I_bin}(j)·e_j(e_j + 1) = 0` at the Stage-1 point `τ` | multilinear in `τ` | `ν/q` |
| fused sumcheck | `A·E + B·E(E + 1)` | degree 3, `ν` rounds | `3ν/q` |

At the headline (`ν = 27`, `⌈log₂ m⌉ ≤ 27`, `⌈log₂ n_×⌉ ≤ 10`) the explicit
rows sum to at most `(126 + 20 + 27 + 27 + 2 + 27 + 81)/q = 310/q ≈
2^−119.7`, before Stage 1. The α round alone is `126/q ≈ 2^−121`.

**Status.** Accepted for now (user, 2026-10-22); revisit with the paper
authors. The response proof is below 128 bits over a single `F_q128`.
Options if the headline must hold: draw `α` (and the sumcheck challenges)
from an extension field, repeat the α round, or state the headline as
about 2^−119.7 for this component. The product and fused sumchecks also carry no proof of
work (Akita spec A34).

**Enforced bound (O4).** The ledger is priced at `ν ≤ 27`;
`jolt_zk::whiteout::MAX_RESPONSE_NUM_VARS` holds that bound, and
`SigmaSetup::response_layout`, which both roles build `e`'s layout with,
refuses a longer `e` (`WhiteoutError::ResponseTooLarge`). The other two
inputs (`⌈log₂ m⌉ ≤ 27`, `⌈log₂ n_×⌉ ≤ 10`) are not enforced in code.

## D28. Public coordinates of w_k

**Paper says.** Coordinates of `s` that hold public data may be left to the
public response `c·s` instead of a Gaussian mask (§4.2, "Public
coordinates"). Akita's note (§4.6, A27) planned to align the setup-group
response in w_k to `R_256` boundaries by padding with zero cells.

**Implementation** (Akita W5d, `akita:specs/whiteout-akita.md` §5.5.1).
`ResidualGeometry` names the public spans of w_k: the response and the
inner digits of the cut level's setup-prefix group, read from that level's
witness layout. `ResidualShape` cuts w_k into runs of whole native elements
(`R_D`, `D` = the residual ring dimension) and marks a run public only when
every coefficient of it lies in a public span; an element that straddles a
span's end stays Gaussian. w_k's layout is unchanged: no zero cells.

**Argument.** Masking a public coefficient hides nothing and costs only a
few Gaussian coordinates; leaving a witness-dependent coefficient public
would leak through `z = c·s`. So the rounding goes one way. The spans are
public by Akita §3.5: the setup prefix's response `Σ_b c_b·ŝ_b` is a
function of the public setup and the level's public fold challenges, and
its inner digits are digits of a setup-derived commitment.

**Status.** Implemented. Not verified by recomputation (Akita A39).

**For the paper.** State that public coordinates need not be aligned: a
straddling element is masked whole.

## D29. u_a in N as a raw public target

**Paper says.** u_a is sent compressed, and Item 3 checks it through its
compression witness (`04_protocol.tex:305-313`).

**Implementation.** Akita's cut successor sends u_a raw (`payload = Raw`,
required by admission, Akita §4.1 and §4.8.1), as the masked prefix's last
public message. The Σ setup therefore holds u_a as `BlockTarget::Public`:
the verifier already has the image, so compressing it again would only add
compression witness to `e`.

**Size (after Akita W5e).** The residual commitment is priced at the Σ
collision radius `4Γ_Δ B(10) ≈ 2^47.8` (Akita A41, §5.5.2), which forces its
`[B_a | B′_a]` from rank 1 to rank 19 over `R_64` at every checked-in row
(`A_a` from rank 4–5 to 18–19). u_a is `rank × 64` coefficients of 16 B:

| row | u_a before W5e | u_a after W5e | modeled Akita proof before → after |
|---|---|---|---|
| 14:1 dense (single-group statement) | 1 × `R_64`, 64 coeff., 1 KB | 19 × `R_64`, 1,216 coeff., 19 KB | 70,534 → 94,500 B |
| 31:1 + 14:1 (multi-group) | 1 × `R_64`, 1 KB | 19 × `R_64`, 19 KB | 78,826 → 96,244 B |
| 36:1 headline | 1 × `R_64`, 1 KB | 19 × `R_64`, 19 KB | 84,322 → 101,672 B |

The modeled sizes are the catalog snapshot's `proof_bytes` (default
artifact set), which counts the raw u_a; the Σ proof and the response
proof are extra.
Measured whole proofs (Akita's `zk_end_to_end` harness: masked prefix,
Σ and response proof; Akita §5.5.2; Apple M4, loaded machine, median of
six runs):

| statement | proof bytes | prover | verifier |
|---|---|---|---|
| single group 14:1 | 75,423 → 95,058 (+26%) | 0.86 → 1.19 s (+38%) | 0.019 → 0.076 s (4.0×) |
| multi-group 31:1 + 14:1 | 80,738 → 99,196 (+23%) | 7.2 → 9.8 s (+36%) | 0.069 → 0.35 s (5.1×) |

The verifier grows because N holds the rank-19 `A_a` and `[B_a | B′_a]`;
the setup grows with them (dense 14:1: 131,072 → 7,045,504 field elements,
54×; the headline's 2^24 is unchanged because its root dominates).

**Security impact.** None: u_a is hidden by `r_a` (Lemma
`wo:lem:stat-hiding`, Akita §4.8.1), raw or compressed.

**Status.** Retired by O4 (below). Before O4: the 19 KB was the cost of
keeping the cut successor's payload raw at the Σ-bound rank. Akita §4.1's
"3 × `R_256`, 12 KB" design figure is superseded: the planner's dimension
ceiling keeps the cut successor at `R_64`, where binding at the Σ radius
needs rank 19.

**O4: u_a compressed (Akita A47).** The departure is withdrawn; the
implementation now follows Item 3.

- *What is sent.* At the cut level's `NextWitnessPayload` site the masked
  prefix sends `cm(u_a)` under W5c's split chain (D9): 1,216 coefficients
  in chunks of 512, 512 and 192, one 8-coefficient payload each, 384 B in
  place of 19,456 B. The chain is the response proof's (`K₁`, `K₂` from
  its setup), passed to the prefix as `ImageCompression`
  (`WhiteoutTail::compression`), so the maps that compress u_a and the
  rows that check it have one owner. Akita's own payload mode at the
  after-cut level stays `Raw`; that mode's chain witness would be proved
  by a successor fold that zero knowledge never runs.
- *What is checked.* N's outer block has target
  `BlockTarget::Compressed { payload: cm(u_a), .. }`. jolt-zk handles it
  as it handles the pad image: `ξ_a` joins `e` after `V` and `θ`, Item 3
  reads u_a as `G₁ξ₁`, Item 2 gains u_a's six chain rows at `α`, and
  `I_bin` covers `ξ_a`.
- *Knowledge soundness.* The paper's argument (Thm `wo:thm:knowledge`),
  with one step the implementation now exercises: every extracted `e`
  below one `cm(u_a)` reconstructs the same u_a, by Lemma
  `wo:lem:compressed-binding` per chunk (D9), unless a chunk map has a
  norm-one kernel vector; Lemma `wo:lem:weak-unique` then applies to that
  u_a as before. `R_Σ`, the residual ranks and the Σ box are unchanged:
  `ξ_a` is in `e`, not in `s`, and the binary constraint certifies it.
- *Hiding.* Unchanged: `cm(u_a)` and `ξ_a` are functions of u_a, which H8
  replaces with a uniform image. Before O4 u_a itself was public.
- *Error.* The chain rows join the α round (`126/q` stays the maximum).
  The ν-dependent rows (binary constraint, padding term, fused sumcheck:
  `5ν/q`) move only if `|e|` crosses a power of two. `ξ_a` adds 161,792
  digits to `e` (67,584 + 67,584 + 26,624 per chunk, from
  `CompressionShape::witness_len`), and no measured workload crosses: ν
  stays 24 (muldiv), 25 (fib 2^16) and 26 (fib 2^20), so D27's ledger is
  unchanged.
- *Measured* (O4, V6 harness `jolt-prover profile --backend optimized`,
  features `profiling,akita,zk`, Apple M4 Max under load ~100; two runs
  each after; "before" is V6's medians and the O1/V6 traces at the same
  Akita pin; proof = argument string + 10 B):

  | | muldiv (2^12) | fib 2^16 | fib 2^20 |
  |---|---|---|---|
  | masked prefix, before → after | 24,591 → 5,519 | 24,818 → 5,746 | 25,275 → 6,203 |
  | response proof, before → after | 70,381–70,393 → 70,328–70,395 | 70,625–70,671 → 70,691–70,704 | 70,797–70,803 → 70,774–70,811 |
  | ν (`whiteout_plan_response`) | 24 → 24 | 25 (inferred) → 25 | 26 → 26 |
  | proof bytes, before → after | 112,931 → 93,792–93,860 | 115,141 → 96,111–96,125 | 117,480 → 98,379–98,415 |
  | proof Δ over clear Akita | +30,954 → +11,815–11,883 | +31,021 → +11,991–12,005 | +28,795 → +9,694–9,730 |

  The prefix falls by exactly 19,072 B (19,456 − 384) in every run; the
  response proof's change is within its run-to-run spread. fib 2^16's
  ν before was not logged in a surviving trace; its response-proof bytes
  sit between muldiv's (ν = 24) and fib 2^20's (ν = 26) as V6 measured.

## D30. Two-witness distinguisher

**What.** Akita's `examples/zk_distinguisher.rs` (HT2, a manual diagnostic)
proves the single-group end-to-end statement `N` times for each of two
witnesses of one shape and point, and runs a 2 × 256 χ² homogeneity test at
every byte position of the proofs (about 75 KB), flagging a position at
`p < 0.01 / positions`. Two injected leaks calibrate it: the first claim's
pad zero, and that pad with only 64 random bits.

**Measured** (Apple M4, loaded machine; about 0.65 s per proof):

| N per witness | trials | honest flagged | pad 0 flagged | 64-bit pad flagged | smallest p (controls) |
|---|---|---|---|---|---|
| 8 | 5 | 0 | 0 | 0 | 6.3e−5 |
| 16 | 5 | 0 | 5 | 5 | 1.5e−8 |
| 64 | 2 | 0 | 2 | 2 | 1.1e−29 |

The honest runs' smallest p-values were 0.055, 0.078 and 0.012, against a
threshold of about 1.3e−7.

**Reading.** The test detects a byte that is constant within a witness and
different across witnesses once a fully separated 2 × 2 table reaches the
Bonferroni threshold, `χ² = 2N` at one degree of freedom: about `N ≥ 14`
here. It has no power against small biases at these `N`; it is a smoke test
for gross leaks (an unmasked or weakly masked slot), not evidence of
statistical zero knowledge.

## D31. Precommitted groups opened masked (W6f)

**Paper says.** Advice committed during proving is witness data;
precommitted trusted advice and committed programs are outside Whiteout's
protection (`03_setting.tex:167-173`). The PCS receives each claim as an
affine form over committed pads (`06_evaluation.tex:78-83`).

**Implementation** (jolt `whiteout/dev` + W6f;
`jolt:specs/whiteout-jolt-integration.md`, prototype deltas).

- *Which groups.* Untrusted advice, `FieldInc` and `FusedInc` (per
  proof), trusted advice and the two committed-program objects
  (statement-fixed) all open as precommitted groups beside the trace group,
  in the clear stage 8's canonical order (`FusedInc` last, always present). Each is committed under the root profile of the
  zero-knowledge row of its own shape (`jolt:crates/jolt-akita/src/whiteout/mod.rs`,
  `PrecommittedRow`), so it carries the masked root's B randomness
  (Akita §3.5, R6). The trace row is planned keyed on those profiles.
- *One backend.* Akita refuses to import a hiding commitment into another
  backend (recomputing it would draw fresh randomness). Every hiding group
  is therefore committed on the trace setup's Whiteout backend, the one
  that opens it (`WhiteoutCatalog`), not on a transparent per-object setup
  as in the clear build. A setup's catalog holds one row per admitted list
  of precommitted groups (`GroupedScheduleParams::whiteout_admitted`), all
  planned on first use in both roles; the opening absorbs the catalog's
  digest, so both roles must plan the same list.
- *Statement-fixed groups.* Trusted advice and committed programs are
  committed hiding at preprocessing time (`commit_trusted_advice`,
  `preprocess_committed`), and the prover keeps the randomness in its
  `AdviceObject` / `CommittedProgramProverData`. The commitment is made
  before any pad exists and is reused by every proof. This is sound and
  hiding for the following reason. The commitment's randomness is a
  commitment-level object: it hides the group's B images (Lemma
  `wo:lem:stat-hiding`) and is placed in level 0's output witness, which
  each proof commits afresh with its own randomness r₁ and then opens only
  through masked levels whose messages are masked by that proof's pads.
  The pads mask messages, never commitments, so the order of the two does
  not matter; across proofs the reused randomness only ever reappears
  inside witnesses that are themselves freshly hidden (Akita §3.5,
  "A commitment reused across proofs is still one commitment"). The
  per-proof groups (untrusted advice, `FieldInc`, `FusedInc`) draw fresh
  randomness in every proof.
- *Packing reduction.* An advice or committed-program object is
  prefix-packed (`PrefixPackedLayout`). The clear reduction absorbs the
  layout words, the digest, the common point and the evaluations, then
  draws the selector s. Under Whiteout it absorbs the same items except
  the evaluations (`PackedEvaluations::Masked`), and the reduced claim is
  the affine form Σ_slot eq(s, slot) · eq(prefix, 0) · formᵢ, built by the
  clear path's own geometry (`PrefixPackedObjectPlan::packed_leaves`,
  `PrefixPackedLayout::select`).

**Why it is sound.** Each formᵢ reads masked messages absorbed before s,
and its coefficients are statement or earlier-challenge values, so the
forms are fixed before s is drawn; t_p fixes the pads. The clear absorb of
the evaluations adds nothing that the transcript does not already bind.
The reduction's own Schwartz–Zippel over s is unchanged, and Akita binds
the reduced claim through its EvaluationBatch challenges as for the trace
group's claims (D25).

**Deviations.**

- *Stricter than the paper.* Statement-fixed groups are hidden (D24's rule,
  enforced by Akita's admission).
- *Setup-bound trusted advice.* A clear trusted-advice commitment is made
  on a transparent setup and can be reused across trace setups and chunk
  profiles. A Whiteout one is made on one trace setup's backend and opens
  only on that setup: its Akita handle never leaves the process, and a
  rebuild from (advice, r) (below) reproduces C only under the same setup
  parameters.
- *Program commitments are recomputable only with r.* A committed
  program's hiding commitment differs per preprocessing run. In the clear
  build anyone holding P recomputes C and checks it, so the verifier need
  not trust the prover's preprocessing. Under Whiteout only a holder of
  (P, r) can: `preprocess_committed_reopened` rebuilds C from them or
  fails.
- *Trust root of statement-fixed groups (Akita A44).* The trust root of a
  trusted-advice or committed-program commitment C is whoever holds its
  hiding randomness r. The committing process exports r from its Akita
  handle (`AdviceObject::hiding_opening`,
  `DirectProgramObjects::hiding_openings`, over Akita's
  `CommitmentHandle::hiding_randomness`) as a `HidingRandomness`
  (zeroized on drop, redacted `Debug`, a versioned, length-checked wire
  form), and ships C with it. The prover rebuilds its hint from (data, r)
  (`reopen_trusted_advice`, `preprocess_committed_reopened`, over Akita's
  `CpuBackend::recommit_hiding`), which replays r into the same commit
  path and fails closed unless the rebuilt commitment's coefficients equal
  C's (Akita) and its header equals C's (jolt-akita). The same call is the
  auditor's check that C binds the advice or program. A verifier accepts C
  on the strength of whoever produced and holds r: the committing process,
  or a separately trusted party that ships (data, r) to the prover.
  - *r is as secret as the data.* Given r, C is a deterministic function
    of m, so a holder of r who lacks m can test guesses of m: for that
    holder C is no more hiding than a clear commitment. The proofs add nothing: r sits in
    level 0's output witness, which each proof recommits with fresh
    randomness and opens only through masked levels (the argument above),
    so they stay simulatable from C.
  - *No reuse of r.* The hiding lemma covers one commitment per draw of
    r. Committing other data m′ under the same r is outside it: the B·r
    term of the two outer images cancels in their difference, a clear
    image of m − m′. Akita offers no way to do so: `recommit_hiding`,
    the only consumer of exported randomness, returns a handle only when
    the rebuilt commitment equals the expected one, so it never yields a
    second commitment under r. The wire form carries no copy of C; the
    binding is that check.
- *Unchunked precommitted rows.* A precommitted group's own ZK row is
  planned unchunked: a masked level spans one witness chunk (D4), and the
  group folds at the single-chunk trace row's root. Groups below 12
  variables have no ZK row (Akita's audited fold domain); Jolt's dense
  objects are at least 14.

**Status.** Implemented (W6f) for advice, committed programs,
field-inline's `FieldInc` (a full-field group, `ZkCommitmentConfig<JoltDenseFull>`)
and the RV64 increments `FusedInc` (a bounded group: its signed increments
fit `JoltDenseBounded`, whose centered magnitude is at least `u64::MAX`):
jolt-prover `whiteout_e2e` (honest proofs verify, tampers reject, a
recommitted trusted advice or program commitment rejects; with
`field-inline`, a field-ops proof with untrusted advice) and jolt-verifier's
census and per-class tampers over the advice and committed-program cases.
The A44 trust root: `whiteout_e2e`'s `*_reopens_from_shipped_randomness`
(a trusted party ships C and r as bytes and drops its handle; a separately
preprocessed prover rebuilds, proves, and the verifier accepts against the
shipped C; altered or swapped randomness and other advice fail at rebuild;
an auditor's rebuild reproduces C) and Akita's
`exported_hiding_randomness_rebuilds_exactly_its_commitment`.

**For the paper.** In §3.1, state that a preprocessing commitment may
carry randomness that the prover keeps and reuses, and that this needs no
change to the hiding argument; and in §4.1 that a prefix-packed claim's
reduction binds its forms through the masked messages, not through an
absorbed value.

## D32. Persisted plans (O1)

**Paper says.** The schedules are public parameters fixed by the setup
(`03_setting.tex`); the paper does not say how a verifier obtains them.

**Implementation** (jolt `wo/o1-plans`,
`jolt:crates/jolt-akita/src/whiteout/mod.rs`, `WhiteoutPlans`,
`WhiteoutSetup`).

- *What is persisted.* A Whiteout setup's serde form is its plans: the
  masked-opening catalog, each precommitted shape's one-row catalog and,
  optionally, a set of response schedules keyed by `e`'s variable count.
  Each is Akita's canonical trusted catalog artifact (`AKSCHD01`, protocol
  epoch, policy digest), inside a `WOPL` envelope with a version.
- *When plans enter a setup.* Only at construction. A fresh setup plans
  its catalog at setup time (`WhiteoutSetup::new`, from
  `AkitaScheme::setup`); its trace shape is fixed by the setup. A setup
  built from persisted plans (serde, or `AkitaSetupParams::with_whiteout_plans`
  / `preprocess_full_from_plans` for a prover) holds them as given. So a
  setup's serialized form is a function of its construction, never of the
  proofs made on it: two independent preprocessing runs, and a decode and
  re-encode, give the same bytes.
- *Response schedules.* `e`'s length depends on the pad layout, which
  depends on the proof's header (trace length, RAM size, one-hot
  configuration), so jolt-akita cannot compute it from the setup alone.
  A fresh setup therefore plans each response length on first use and does
  not persist it. `AkitaVerifierSetup::whiteout_plans` exports every plan a
  setup holds or has derived, responses included; a setup constructed from
  such a set treats it as exhaustive and refuses a length it lacks
  (`Statement`), never planning it. A plain serde round trip of a fresh
  setup carries the catalog but `responses: None`. The decoded verifier then
  plans each response on first use, as a fresh setup would. That is
  soundness-neutral, because the response digest is absorbed at `PADS`
  before any challenge. It does cost the first verify about 12 s, so
  distribute verifier preprocessing built from `whiteout_plans()` when
  first-verify latency matters. The default path's latency is O2's
  (planner speed).
- *Checks on load.* A plan is never trusted. Every row goes through
  Akita's trusted-artifact decoder (`TrustedScheduleCatalog::from_artifact_bytes`):
  canonical encoding, policy and family binding, the row audit (SIS table
  keys and bounds at the policy's 128-bit tables, digit depths, matrix
  widths, transition lengths, challenge hooks) and the configuration's
  admission: `ZkOpeningSchedule::admit` for the masked-opening and
  precommitted rows, which includes the residual's Σ-binding ranks
  (`sigma_binding_min_rank`, Akita §5.5.2), and the clear audit for a
  response row. Then the rows must be this statement's: the catalog's
  trace shape is the setup's, it holds exactly one row per admitted list
  of precommitted groups and each resolves by that list's key, each
  precommitted catalog is one row keyed on its shape, and a response
  catalog is one row keyed on its length. Failures are
  `WhiteoutPlanError::{Format, Inadmissible, Statement}`; a refused plan
  is never replaced by planning. Loading costs under 0.03 s (O1 outcome).
- *The clear audit suffices for a response row.* What the Σ extractor
  needs from the response proof is fixed by the configuration, not the
  row: the digit-range sumcheck runs at `Cfg::decomposition()`'s radix
  (`range_plan` refuses a statement of another alphabet), and the
  compression maps are read from a prefix of the response setup's public
  matrix whose length is fixed by the configuration's modulus profile
  (`compression::setup_len`, checked by `covers_chains`). The row
  contributes only its commitment, whose binding is the ordinary SIS
  binding the audit prices.
- *What is not re-checked.* The planner's search. A loaded row is
  admissible but need not be the cheapest the planner would choose.
  Optimality is not a soundness property.
- *Binding.* At the `jolt/pads` site, before the pad commitment
  (`WhiteoutTail::commit_pads` / `receive_pads`), the transcript absorbs
  the masked catalog's digest, the selected row's digest and the response
  schedule's catalog digest (3 × 32 B). Every challenge after the pads
  follows: stages 1–7, the masked levels, the Σ-protocol and the response
  proof. The masked row fixes D's shape, the pad commitment's shape and
  the commitments' decoding profiles, and the response setup compresses
  the pads, so each is bound before anything that depends on it. Stage 8
  absorbs the catalog and row digests again (`WhiteoutRow::absorb_selection`);
  that absorb is now redundant and kept, at no proof bytes. Before O1 the
  response schedule was a function of the statement and was not absorbed,
  and the masked row was absorbed only at Stage 8, so a party supplying
  preprocessing could have chosen among admissible plans after the stage
  1–7 challenges. The plans themselves come from the verifier's setup, not
  from the proof.

**Why it is sound.** Soundness of the masked opening and of the response
proof holds for any row that passes the audit and admission: the same
check every shipped clear catalog passes, and the one the planner's own
output passed before. The digests bind the rows the verifier checks under
before any challenge that depends on them.

**Status.** Implemented (O1):

- jolt-akita `whiteout_persisted_catalog_is_validated_on_load`: decoded
  and fresh setups re-serialize to the same bytes. A residual `A_a` one
  rank lower, with `B_a`'s width narrowed to keep the row consistent and
  edited in place in the canonical bytes, is `Inadmissible` with "below
  its Σ-binding rank". Another trace group, a relabelled catalog and an
  extra precommitted row are `Statement`; another version is `Format`.
- `whiteout_persisted_response_plan_verifies_and_is_validated`: a proof
  verifies on a setup decoded from exported plans; an exhaustive set
  lacking the length is `Statement`; a catalog in a response plan's place
  is `Inadmissible`.
- jolt-prover `whiteout_e2e::whiteout_plans_persist_and_are_consumed`:
  preprocessing bytes are independent of proving and of a serde round
  trip; a prover and verifier built from exported plans prove and verify;
  plans of another version are refused at the prover's first use, the
  trace commitment.

**For the paper.** State in §3 that the schedules are public parameters
the verifier validates and the transcript binds before the first
challenge that depends on them.

## D34. Small byte items (O5)

**Paper says.** The evaluation charges 416 B for three compressed
commitments, each one two-map chain of 128 B (the pad commitment, t_D and
the commitment to e that begins its Akita proof), plus the attempt nonce
(`06_evaluation.tex:188-190`); its cost script chains images of up to
`FCHAIN_MAX_IMAGE = 884` coefficients. The cut sends `cm(u_a)` the same
way (Item 3, A47). The auxiliary block holds one masked value per
multiplication gate whose output is not an existing slot: 28 products,
2 affine operands and 2 Akita values, 512 B at T = 2^24
(`06_evaluation.tex:83-88`).

**Before O5.** t_p (pad commitment, rank 19 over `R_64`, 1,216
coefficients) and cm(u_a) (1,216 coefficients) were 3 chunks of at most
512 coefficients each, 384 B apiece (D9). The auxiliary block was 45
slots, 720 B, on all three workloads: Akita's 2 range-leaf squares, 2
defining slots (stage 1's outer remainder and stage 2), and 41
intermediate products (7 in stage 5, 34 in stage 6's cycle phase).

**Implementation.**

- *Chunks below the certified bound* (Akita `wo/akita-o5`,
  `akita-params/src/compression/chain.rs`,
  `akita-pcs/src/zk/compression.rs`). The bound is
  `CompressionChainPlan::max_certified_source_coefficients`: the largest
  source whose first map stays inside its rank-one SIS cell. At q128 that
  cell is d = 32, 60,460 columns of `{−1, 0, 1}` differences
  (`akita-params/src/sis/compression.rs`), and a source of n coefficients
  needs `128·n/32` columns, so n ≤ 15,115; 15,116 needs 60,464.
  An image takes the fewest chunks that bound allows, balanced to within
  one coefficient (`compression::chunks`, the single owner of the
  geometry). Each chunk is derived by `for_certified_source`: the same two
  maps, the same certification and the same 256 B terminal as the clear
  chain, without the 8 KiB cap, under its own policy id
  `NegativeBinaryTwoMapSisBoundV1` (tag 4, appended), which no instance
  descriptor admits. The cap still bounds every clear B/D image, so no
  clear catalog, plan or descriptor changes. t_p, cm(u_a) and t_D each
  take one chunk.
- *Shared intermediate products* (jolt `wo/o5-bytes`,
  `jolt-zk/src/masking/lowering.rs`, `ProductMemo`). One memo spans a
  relation's lowering in one domain (`JoltResidual::lower_claims`). A
  term's chain starts at its first single-opening factor and multiplies
  the others in formula order; an intermediate product whose factors so
  far are all openings is keyed by those openings, in chain order, and
  takes a slot only the first time. On the Akita build, stage 6's lattice
  bytecode read-RAF cycle output (`read_raf_cycle_output_committed_lattice`)
  repeats `ra₀·ra₁` in its 8 staged-Val terms and `fused·ra₀` and
  `fused·ra₀·ra₁` in its 4 fused-increment terms: 16 slots become 3. The
  block is 32 slots, 512 B: the paper's count.
- *BlindFold.* The memo is in the shared lowering, so BlindFold committed-
  program proofs shrink too: `ReadRafCyclePhaseCommitted` repeats the
  bytecode RA product in every staged-Val term, and the committed muldiv
  fixture goes from 38 auxiliary slots to 27 (pinned in
  `tests/completeness/zk.rs`). The plain muldiv fixture repeats no product
  and keeps 26. The ZK verifier-fixture cache key of the committed case is
  renamed, and the ZK fuzz bundles are regenerated (at the base they
  already failed to verify: `zk-muldiv` at the final opening, the other two
  with `Transcript(NonCanonical)`).

**Argument.**

- *Chunk binding.* Lemma `wo:lem:compressed-binding` applies per chunk
  (D9): two openings of one chunk's `cm` that reconstruct different images
  give a nonzero `{−1, 0, 1}` vector in the kernel of `K₁` or `K₂`. A
  t_p or u_a chunk's `K₁` (1,152–1,280 coefficients) is 4,608–5,120 of
  the `R_32` cell's 60,460 columns; every chunk's `K₂` is 256 of the
  `R_16` cell's 2,400. The cells are the authority every compression map
  is priced by (`min_compression_secure_rank`), and a map narrower than
  its cell is at least as hard. The chain rows' α-reduction error is
  unchanged per row (degree below the map's ring dimension), and there are
  fewer rows. The binary constraint covers ξ₁ and ξ₂ of every chunk, as
  before; ξ₁'s length depends only on the image, and each removed chunk
  removes one ξ₂. The response schedule is keyed by `e`'s
  length, so a shorter `e` selects its row by the existing lookup.
  Hiding is unchanged: compressed forms are functions of the raw images
  (D9).
- *Where the chunk geometry is bound.* It is a function of the modulus
  profile and the image length alone (`compression::chunks`), which both
  roles and `CompressionShape::payload_len` read, so every payload length
  the verifier receives follows from public parameters, as before. No
  transcript site moves. Message lengths change at the pad commitment and
  the cut (one chunk fewer each), at the auxiliary block (fewer slots) and
  in the response proof (fewer chains in `e` and in Item 2).
- *Slot sharing.* A shared slot z is defined once, by the product row
  `x·y = z` the first term emits, and each later term with the same key
  reads z. The key is the ordered list of opening ids whose decoded values
  the chain multiplies, so every term with that key computes the same
  value in the clear verifier: the decoded relation is the clear stage
  checks, and Lemma `wo:lem:residual-correct` holds as before. The memo
  keys on structure only (ids, never values), and the shape, values and
  rows domains each run the same lowering, in the same order, over a
  fresh memo, so they allocate the same slots, with each slot still
  defined by one earlier row (the acyclic chain of D12). ZK: the
  relation's shape is public and there are fewer pads.

**Not done, and why.**

- *The 2 defining slots.* Distributing an affine operand over its
  openings (`(Σ c_i o_i + c₀)·y = Σ o_i·(c_i y) + c₀ y`, a multi-pair row)
  would remove both (−32 B), at the cost of 35 + 3 more operand pads for
  stage 1's and stage 2's affine factors. Not taken; the paper also counts
  2 affine operands.
- *The rest.* Akita's 2 slots are the degree-4 range leaf's square
  (`check_range_leaf`), and each remaining product slot is an
  intermediate of a product of h ≥ 3 distinct hidden factors (h − 2 per
  product). Every slot holds a uniformly masked field element, so
  nothing in the block compresses.

**Measured at O5** (release, `--backend optimized`; proof wire bytes):

| | muldiv | fib 2^16 | fib 2^20 |
|---|---|---|---|
| before (whiteout/dev 9cb3ee7ec) | 93,892 | 96,072 | 98,458 |
| after, run 1 (884 + 332 chunks) | 93,358 | 95,513 | 97,896 |
| after, run 2 (balanced 608 + 608) | 93,325 | 95,535 | 97,883 |
| Δ | −534 / −567 | −559 / −537 | −562 / −575 |

Per region, identical in both runs: pads −128, cm(u_a) in the masked
prefix −128, auxiliary block −208 (720 → 512), so −464 B is deterministic.
The rest is the response proof, which varies run to run by about 100 B
(V6; here 70,324–70,358 / 70,556–70,578 / 70,742–70,755 against
70,428 / 70,651 / 70,854 before), plus a byte or two of variable-length
encoding in the prefix, so the regions sum to the totals only within that
spread (fib 2^20, run 1: −563 by region against −562 in total).
Balancing does not change the byte count: the payload is 8 coefficients
per chunk either way.

**For the paper.** None for the chunk bound (it is the paper's). For the
auxiliary count: say that a product repeated across terms is one gate.

## D35. The residual commitment over `R_128` (O6)

**Paper says.** The three matrices that commit to s (`A_a`,
`[B_a | B′_a]`, `A_p`) are sized coefficientwise at their collision radii,
each over the ring dimension Akita supports that minimizes the verifier's
work. Security tracks `rank · d`, so a larger ring with a smaller rank is
cheaper (`06_evaluation.tex:124-133`). Several ring dimensions are handled
by restricting scalars to the challenge ring (`02_preliminaries.tex:69-77`).

**Before O6.** All three were over `R_64` (D29: rank 18–19 for the residual,
18–19 for `A_p`). Akita's planner never gives a level a larger ring than
its predecessor (the monotone dimension ceiling), and the cut level's B and
D rings are 64 on every row.

**Implementation** (Akita `wo/akita-o6`, §5.5.3; jolt `wo/o6-d128`,
`specs/whiteout-o6-d128.md`).

- The zero-knowledge planner gives the level after the cut the ring
  dimension 128 (`RESIDUAL_RING_DIMENSION`) and exempts that one level from
  the ceiling. Clear searches never reach the after-cut level, so no clear
  catalog changes.
- The residual is rank 9–10 over `R_128` on every row (rank 18–19 over
  `R_64` before; `A_a` was 18 on dense 14:1). u_a is 1,280 coefficients instead of 1,216 (1,152 on a rank-9
  row) and is sent as one chunk of `cm(u_a)`. r_a is 321 `R_128`
  columns instead of 610 `R_64` columns.
- `A_p` follows, since jolt's Σ setup has one ring: rank 9 (rank 10 above
  1,231 columns) over `R_128`. t_p is 1,152 or 1,280 coefficients, one
  chunk.
- The zero-knowledge planner never offloads a setup into the cut level.
  A setup prefix there puts the parent's response and inner digits into
  w_k, hence into `s` and `e`, and the planner's setup-first objective
  does not price them. With the cheaper `R_128` residual it chose that
  offload from 35 variables, and Akita's 36:1 headline row then needed
  ν = 29 > 27 (refused: unprovable, not unsound). The narrowing concerns
  only the cut level; offloading at earlier folds (D5's "k = 2 with
  offloading") is unchanged, and admission still accepts a cut-level
  setup prefix (used by tests). A test checks that every checked-in ZK
  row's `e` fits the bound.

**Argument.**

- *Binding.* `R_Σ = 8κ₁·B(10) ≈ 2^47.82` is unchanged: multiplication by a
  challenge `c ∈ R_64` acts on `R_128` as `c(Y²)`, with the same ℓ1 norm,
  so `Γ_Δ = 2κ₁` holds verbatim. The reduction of `05_security.tex:31-46`
  then needs Module-SIS over `R_128` at `R_Σ`. Akita's estimator gives
  128.79 bits for `A_a` at rank 9 × 1,024, at least 138.07 bits for every
  rank-10 cell (138.07 at 10 × 11,841; the widest checked-in cell is
  10 × 10,881), and 128.26 bits at `A_p`'s
  largest rank-9 width (1,231). The chain's first map on a 640-coefficient
  chunk is 139.13 bits (140.98 at 608).
- *Knowledge soundness.* Thm `wo:thm:knowledge` is unchanged in its
  statement and its error terms. ℓ, `|𝓒|` and the radius do not depend on
  the residual's ring. D27's ledger depends on ν, which does not grow on
  any measured statement (the Akita single-group statement goes from 24 to
  23), so it stays 310/q ≈ 2^−119.7.
- *Zero knowledge.* r_a and r_p are sized by the same entropy rule (image
  plus 264 bits). Every hypothesis of Thm `wo:thm:zk` is checked as before,
  and the statistical total stays 2^−129.1 (O7 §7.4).
- *Why the ceiling may lift there.* It is a planner search rule, not a
  protocol constraint. No check compares adjacent levels' rings. The
  interface between levels is a flat coefficient vector, zero-padded to the
  successor's ring. In zero knowledge the after-cut level is never folded:
  its parameters are only the residual commitment.

**Measured** (`specs/whiteout-o6-d128.md` § Results; 7 interleaved runs,
load < 6). Proof bytes are unchanged within noise. The residual `N·z`
products are 0.56–0.57×. Steady verify is −15% / −19% / −25% on one thread
and −4% / −4% / −9% on 16 (muldiv / fib 2^16 / fib 2^20). First verify is
−19% to −21%, and the prover tail −3% to −5%.

**For the paper.** The evaluation's choice of a larger ring for the
residual is now implemented at d = 128. The text could say that the
residual's ring may exceed the cut level's ring, since the after-cut level
is never folded.

