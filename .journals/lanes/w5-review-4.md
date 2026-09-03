# W5 assembly review #4

Target: `f986c2e00`; fix range `a9d8828dd..f986c2e00`. Review ran in detached
`/Volumes/Dev/worktrees/jolt/w5-review4` with
`CARGO_TARGET_DIR=/Volumes/Dev/target/w5-review4`.

## Verdict

**0 blockers / 0 majors / 3 minors**

The Fiat–Shamir reorder is sound. No proof field controls phase counts or
challenge offsets. The remaining findings are regression quality and API
cleanup, not protocol attacks.

## Findings

### Minor 1 — the preamble negative exits before checking T1

**Files:** `crates/jolt-wrapper/tests/wrap_real_t1_r.rs:395-405,600-602`,
`crates/jolt-wrapper/src/wrap.rs:518-530,671-693`

The test flips `hash_public.state_in[0]`, which also changes the four-field
public statement absorbed before every commitment. Diagnostic instrumentation
on the real gate returned:

```text
preamble_error=Spartan(OuterFinalClaim)
```

The old proof therefore fails at the first Spartan final check because its
transcript changed. It never reaches the T1 wiring relation, so this negative
does not prove the claimed key-side T1 pin.

**Fix:** keep the end-to-end statement-mismatch negative, but name it as such.
Add a fixed-challenge T1 check: derive `T1Challenges` once, change `state_in`,
and assert that the wiring member's key-derived input claim changes while the
T1 commitments stay fixed. The attached patch adds that check.

### Minor 2 — the real gate pins only two of five phase counts

**Files:** `crates/jolt-wrapper/tests/wrap_real_t1_r.rs:236-245`,
`crates/jolt-wrapper/src/wrap/key.rs:109-123`

The regression asserts phase 1a and phase 1b formulas, but never asserts the
claimed complete vector `39 / 23 / 1 / 3 / 232`. A future change to phases 2a,
2b, or 2c can update prover and verifier together and leave this gate green.
The first assertion does fail on the pre-fix layout (`60 != 39`), so it catches
the reviewed bug but does not pin the stated full schedule.

**Fix:** assert the five `challenge_count` values as one exact vector. The
attached patch does so.

### Minor 3 — the fix adds two public diagnostics

**Files:** `crates/jolt-wrapper/src/limb_table/stream.rs:72-91`,
`crates/jolt-wrapper/src/stream/types.rs:15-24`

`carry` and `spartan` are private (`src/lib.rs:7,13`), the Spartan assembly
items are crate-private (`src/stream.rs:36-41`), and the named dead helpers are
gone. The stronger claim "no new public surface" is false:
`T2Challenges::from_transcript` and `VerifierCost::matrix_fr_mul` are both new
public items. The former also retains an old-layout branch used by standalone
tests; the wrapper production path uses the new split layout. The latter is a
single-report diagnostic consumed only by the real fixture gate.

**Fix:** make `from_transcript` crate-private and let the external fixture test
splice into the existing `from_challenges` constructor. Remove
`matrix_fr_mul` after recording this measurement, or move the matrix breakdown
to an intentional benchmark output rather than the public verifier-cost type.
No behavioral regression test applies to visibility cleanup.

## Reconstructed Fiat–Shamir schedule

The transcript absorbs the key digest and eleven public fields, then for each
key-owned phase absorbs exactly its group commitments before drawing that
phase's challenges (`src/stream/transcript.rs:65-90`). The wrapper key owns the
offsets and counts (`src/wrap/key.rs:98-123`); `WrapperProof` contains no phase
metadata (`src/stream/types.rs:62-72`).

| Phase | Commitments absorbed before draw | Challenges drawn | Count | Later committed columns depending on the draw |
|---|---|---|---:|---|
| 1a | T1 columns and T1 VK groups; Spartan W; ten CopyLink fixed-key groups | T1 `tau_rows[18]`, `tau_wiring[18]`, two relation randomizers; `theta` | 39 | T1's 38 values touch no later commitment. `theta` enters the Dory wire adapter and its Straus offsets, then T2 phase-1b chunks/digits/sign data. |
| 1b | T2 chunks, digit bits/value, lookup/range multiplicities, sign flag | `xi`, `alpha`; ten CopyLink `(beta, gamma)` pairs; scalar-link `rho` | 23 | `xi` -> phase-2a operands; `alpha` -> phase-2a range helpers/inverse; CopyLink pairs -> final-phase inverse helpers; `rho` -> no later commitment, only the digit/scalar stage-A member and term export. |
| 2a | T2 operands, range helpers, range inverse | `fp_root` | 1 | phase-2b positive/negative fingerprints |
| 2b | T2 positive/negative fingerprints | T2 `beta`, `fp_combine`, `copy_root` | 3 | `beta` and `fp_combine` -> phase-2c lookup helpers `H/G+/-`; `copy_root` -> no later commitment, only the row relation at stage A. |
| 2c + helpers | T2 `H/G+/-`, T2 VK groups, all CopyLink inverse helpers | T2 row point `[18]` plus `gamma/lambda/lambda_lookup/constancy_root`; ten CopyLink points `[18]` and three relation weights each | 232 | none; all feed stage A and the term stage |

Count derivation at the real profile (`log_rows = 18`, ten CopyLinks):

```text
1a: (2*18 + 2) + 1          = 39
1b: 2 + 2*10 + 1            = 23
2a:                            1
2b:                            3
2c: (18 + 4) + 10*(18 + 3) = 232
```

`theta` is drawn after phase 1a and passed to the Dory adapter
(`src/limb_table/adapter.rs:63-74`); its offset wire drives the Straus offsets
(`src/limb_table/dory.rs:304-313,533-539`). Nothing in the 1a column build reads
it. T2 phase construction records the exact dependency chain
(`src/limb_table/stream.rs:361-475`). CopyLink inverse helpers are computed from
both committed sides after their pair challenges
(`src/links/copy_link.rs:61-119`) and committed with phase 2c
(`tests/wrap_real_t1_r.rs:320-355`). Stage A starts only after `finish` has
combined all five phases (`src/wrap.rs:611-662`).

The verifier reconstructs pinned groups, then runs the same
`assembly_transcript` over the private key statement's `commitment_phases`
(`src/wrap.rs:665-693`). Challenge counts and offsets never come from the
proof.

## Native matrix evaluation

For a parent equality weight `p` and point coordinate `r`, the old children
were `p(1-r)` and `pr`. The trim computes

```text
first  = p(1-r)
second = p - first = p - p(1-r) = pr
```

so `src/stream/term_stage.rs:415-429` is exact over `Fr`.

For matrix `M`, row weights `e_i`, witness-column weights `q_j`, coefficients
`M_ij`, and matrix weight `eta_M`, the old and new orders are

```text
sum_M sum_(i,j) eta_M * e_i * q_j * M_ij
= sum_M eta_M * (sum_(i,j) e_i * q_j * M_ij).
```

`src/spartan.rs:272-294` applies each `eta_M` once after its matrix
accumulation. The prover's separately structured production path constructs
the same linear form by projecting sparse matrices into witness columns
(`src/spartan.rs:297-349`, called at `src/stream/protocol.rs:135-142`); the
verifier evaluates the trimmed form at `src/stream/protocol.rs:456-465`. The
real proof's acceptance is therefore checked against a production path, not a
test copy of this loop. The test only prints the observer count.

The observed block follows directly from the gate's 13-row/13-column rounds
and 35,346 nonzeros:

```text
two equality tables       2 * (8,192 - 1) = 16,382
all sparse terms          2 * 35,346       = 70,692
three public weights + three witness weights + final product = 7
total                                          87,081 Fr mul
```

The real gate reproduced `127,884` total Fr multiplications and `5,048,805`
gas. Review #3's real baseline was `179,547` and `6,082,065`; the
`51,663 * 20 = 1,033,260` gas delta exactly matches the Fr-multiplication drop.

## Other reviewed claims

- Shared-opening accounting is now honest: `src/stream/shared_rounds.rs:19-21`
  states that the common degree-5 opening covers degree-3 outer rounds with a
  `5/|Fr|` per-round term. No remaining source claim says the opening enforces
  degree 3.
- Payload / bincode / statement remain `7,488 / 7,628 / 352` bytes. The real
  tamper loop and direct witness/key negatives all returned rejection.
- `git diff --check a9d8828dd..f986c2e00` was clean. No added `allow`, unsafe,
  `cfg_attr`, or TODO site was found in the touched source.

## Gates

- `cargo nextest run -p jolt-wrapper --features prover-fixtures --cargo-quiet real_wrapper`:
  1/1 passed in 57.450 s. A second allowed run with success output and a
  temporary print passed in 56.499 s and produced the byte/cost/error evidence
  above; the scratch edit was removed.
- `cargo nextest run -p jolt-wrapper --cargo-quiet`: 64/64 passed in 232.347 s.
  This default set also ran the real-wrapper test.
- `cargo clippy -p jolt-wrapper --all-targets --features prover-fixtures -q --message-format=short -- -D warnings`:
  passed.
- Public-item grep performed on every touched source file; result summarized in
  Minor 3.
