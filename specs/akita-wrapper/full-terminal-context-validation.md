# Full conditional terminal-context validation

Producer evidence, 2026-09-22. This validates the complete conditional terminal
fragment on the frozen epoch6 muldiv tail. It is not a full-wrapper proof or
validation of the incoming transcript prefix.

## Source and reproducibility

Reviewed production source: `f2701fb33c67201601baecc7475698ee50f37c1b`.
Public native owner: `f5f75335eae18241681fd24ca0a60fca8f0512af`.
The proof was produced under native `28fc72021c120e7bc4e0102e07844f25299051eb`;
the frozen census replay used `e24f3b6f54bfcfb378a9ad65a0609e1bdd1ccdc1`.
These are distinct provenance stages, not interchangeable source labels.

Evidence directory: `/private/tmp/full-terminal-context-20260922`.
`manifest.json` SHA256:
`010fdf4555be2d28fd1220763a6bbfee3183d53b7c76ad6bd1fd2866040576a9`.
It pins 122 files, including exact command JSONs, logs, sampled-governor results,
source patches, both diagnostic executable versions, native observer executable,
all mutation trials and original failures. `results.json` collects the six full
runs. Input paths and hashes are pinned in each command.

The temporary diagnostic is deliberately absent from production source. Reproduce
in an isolated checkout of the reviewed head by applying
`instrumentation-controls.patch`, copying
`terminal_context_diagnostic.controls.rs` to
`crates/jolt-akita/examples/terminal_context_diagnostic.rs`, and using the public
lockfile. Exact commands are `controls-clippy.command.json`,
`controls-build.command.json`, and the individual run command JSONs. The positive
and unknown runs used the earlier archived example/executable; its source is
`terminal_context_diagnostic.rs` and patch `instrumentation-positive.patch`.
Absolute task paths require explicit relocation on another machine.

Clippy and executable build used `--offline --locked --profile test -p jolt-akita
--features r1cs --example terminal_context_diagnostic`, with one Cargo job and
incremental compilation disabled. Compilation used a 4 GiB/600 s sampled guard.
Full runs used a 24 GiB/900 s guard and 12 GiB free-disk floor. The governor samples
process-group RSS; this is not a hard memory limit. No proof was generated.

## Entry boundary and shared variables

The native replay checks the observed prefix and imports its full event1219
squeeze checkpoint: CV, mode, consumed64, generated block1, no leftovers. This
fixture import is not constrained upstream execution. The pending event1221
predecessor claim is appended exactly once before `TerminalContextProfile::enforce`,
using the same canonical claim handle supplied to that composer. The composer then
absorbs t, all17 point elements, claim, e, and the native-owned final FoldDraw
frame. The same canonical e/t/point/claim handles feed the terminal relations.
The actual final logical FoldResponse nonce is0; the native plan contains2356
nonce bits. The constrained root is
`f27f73fef09b695e8a5355ea576583a146c5edd9764a4ad5464b6279d03eae2b`.

The complete construction includes seven indexed streams, their first accepted
D64 candidates, all192 A equations, all64 consistency equations, scalar opening,
and the existing TerminalZ range/norm constraints. R3/K7 is a fixed public
fixture-sufficient profile, not a generic completeness guarantee or a measured
optimum. Positive selected rounds (zero-based) are `[1,0,0,1,2,1,0]`; consumed bytes
are `[190,97,97,189,293,189,100]`. All seven ordered positions/signs agree with the
live public native draw, beyond dense-coefficient agreement.

## Results

Every full run has22,375,450 rows,22,127,076 variables and124,689,242 nonzeros.
The matrix fingerprint is identical in every run:
`3bcc1966851791069915b2d8e39e10638376f4b0e681aacdb68bf1f4094424ce`.
It streams all sparse A/B/C entries with dimensions under the documented
`akita-terminal-context-matrices/v1` domain; no second matrix is retained.

| Run | Result | Seconds | Sampled peak RSS bytes |
|---|---|---:|---:|
| Positive | Full witness satisfies; native root/all7 ordered challenges match |23.902|15,712,141,312|
| Unknown | Identical layout/fingerprint; no witness acceptance asserted |21.888|14,100,922,368|
| e[0]+1 and matching canonical bytes | Actual unsatisfied row22009072 |23.956|15,108,751,360|
| t[0]+5 and matching canonical bytes | Actual unsatisfied row22009072 |23.921|15,707,193,344|
| Nonce0→1 | Actual unsatisfied row22009072 |23.914|15,710,175,232|
| Direct assignment forgeries | Root bit: row13444540; accepted coefficient: row14498082 |27.069|15,713,091,584|

Each process exited0 with no survivors. Negative exit0 means its expected
unsatisfied-row check passed. The forgery run first checked a satisfying witness,
then mutated/restored individual assignments on that same matrix; it did not clone
the matrix. These are direct assignment controls, distinct from coherent e/t/nonce
controls. The latter regenerated native roots/challenges and all dependent honest
hints while retaining z and other unmodified relation inputs.

Native capacity reconnaissance was deterministic, capped at64 deltas per family,
and used the existing native sampler owner with read-only logs. e+1 and nonce+1
fit immediately. t+1,+2,+3 exceeded K7 (max9,11,10); t+4 required4 candidates;
t+5 fit R3/K7. These four misses are capacity observations, not R1CS rejections.
All per-candidate position trials and stream reads are in `capacity-search.json`
and its raw logs. The local observer dependency graph, paired native/sponge
instrumentation and lockfile are archived separately; the full circuit consumer
was restored to the public f5 graph before its build/runs.

## Instrumentation audit and limitations

`audit_constraint_sources.py` and `constraint-source-audit.json` prove whole-file
identity with the reviewed source after removing the exact listed diagnostic
additions. This covers imports/types/methods as well as constraint blocks. The
additions import the fixture checkpoint, capture existing ordered candidate
handles without emitting constraints, and derive quotient witness hints from the
canonical `IntegerTerm` lists. Signed division truncates toward zero; it does not
require divisibility before the original residual constraint can reject. Original
A and consistency quotient bounds remain unchanged. Every positive quotient
matches the frozen independently transported quotient. The initial bad splice
was caught by Clippy's unused original-bound error before execution; the corrected
whole-file comparison and original failure are retained. Later initial diagnostic
lints and the observer's missing paired sponge patch are also retained as failures.

All diagnostic hooks/examples were removed after archiving. This commit adds only
this evidence note; production constraint source remains exactly the reviewed head.
The diagnostic's `positive_quotients_match_frozen` output flag is literal-mode
metadata: it is true only for mode `positive`; the `forgeries` mode also executes
the frozen-quotient comparison before its mutations.

Still external: authentication of the checkpoint and pending predecessor values
through the preceding Jolt/Akita verifier, setup/plan/A ownership, preceding folds,
and the complete public-input statement. Typed terminal z is justified by the
separately reviewed normalization lemma; original Golomb transport bytes and the
dead final absorb are not constrained here. Finite R/K capacity restricts
completeness. No backend proof generation, full-wrapper acceptance, or end-to-end
recursion claim follows from this diagnostic. The next composition boundary is
the real upstream constrained transcript and predecessor-value producer, replacing
the fixture checkpoint while retaining these same handles and schedule position.
