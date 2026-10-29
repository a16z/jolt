# Whiteout U2: pre-PR audit of tests and helpers

Scope: tests, test modules, helpers, examples, benches, scripts, probe-only
features and pub items that the Whiteout line added on `whiteout/dev`
(jolt `c84edca00`, Akita `f0de039e2`). The shared-primitives unification
(jolt-algebra/jolt-crypto/jolt-zk absorption, including the jolt-algebra
tests, benches and fuzz targets copied from Akita), NARG transcript
unification (`logging`, `support/narg.rs`, fs inventories other than
`zk-site-classes`) and the recursion PRs are out of scope.

Classes: KEEP (with the distinct failure signal), REMOVE (applied as a
`[U2]` commit), CONVERT (kept as a documented tool, with its command).

## jolt

| item | class | reason |
|---|---|---|
| jolt-zk/src/whiteout/ring.rs::LiftedFunctional (+ `WhiteoutError::{FunctionalLength, RingVectorLength}`) | REMOVE | W1 prototype with no caller outside its own test; the lift lives in `whiteout/lift.rs`. |
| jolt-zk/tests/whiteout_ring.rs (5 tests) | REMOVE | Tested only `LiftedFunctional`. |
| jolt-zk/src/whiteout/challenge.rs::challenge_times_i64 | REMOVE | Test-only `pub` wrapper; its export kept prover-only code in verifier builds. The test calls `challenge_times_i64_into`, now gated on `whiteout-prover`. |
| jolt-zk whiteout re-exports `PadPlace`, `OPERAND_LEVELS`, `OPERAND_LOG_BASIS`, `PAD_LEVELS`, `PAD_LOG_BASIS`, `FIELD_LEVELS`, `balanced_range`; `PadPlacement::place` | REMOVE (unexported, `pub(crate)`) | No caller outside jolt-zk in either repo. |
| jolt-crypto/src/lattice/layout.rs::TwoLevelLayout::is_module_over_challenge_ring | REMOVE | No caller outside its test; test renamed `two_level_layout_checks_its_shape`. |
| jolt-prover/Cargo.toml: `prover-fixtures` → `jolt-zk?/test-support`, dev-dep `jolt-zk/test-support` ("until W6d") | REMOVE | Stale since W6d; no jolt code calls `CommittedPads::uncommitted`. |
| jolt-zk/src/whiteout/tests.rs (19 tests) | KEEP | Σ-protocol completeness at two ring dims, per-row-class verifier rejection with self-checks skipped, each self-check, independent `ct Q_μ` oracle, abort rate and response moments, tamper, rejected-attempt substitution (soundness), width/digit refusals. Only unit-level signal for the Σ verifier. |
| jolt-zk/src/whiteout/sigma.rs `#[cfg(test)]` REJECTED/CALLER/take_rejected | KEEP (flag) | Test seam in production code, compiled only under `cfg(test)`; sole support for `rejected_attempts_cannot_be_substituted`. |
| jolt-zk/src/whiteout/sigma.rs::prove_sigma_unchecked, `SelfChecks::Skip` (`whiteout-test-support`) | KEEP | Used by Akita's `zk_end_to_end` harness to show the verifier rejects what the self-checks catch. |
| jolt-zk/src/whiteout/test_support.rs | KEEP | Synthetic Σ statements and checking/sending response provers; Akita's akita-pcs zk tests and examples use `SyntheticSigma`, `F`, `CheckingResponseProver`. |
| jolt-zk/src/whiteout/fixture.rs | KEEP | Shared synthetic instance for in-crate tests, test-support and the dudect harness; one owner. |
| jolt-zk/src/whiteout/diagnostics.rs + feature `whiteout-diagnostics` | CONVERT (already a tool) | Harness behind `sigma_dudect`/`sigma_timing` and Akita's `compression_dudect` (`with_dit`); feature-gated, never in production builds. |
| jolt-zk/examples/sigma_dudect.rs | CONVERT (already a tool) | CT (dudect) check of the attempt's secret-dependent steps; `cargo run --release -p jolt-zk --features whiteout-diagnostics --example sigma_dudect`. Only CT signal for those steps. |
| jolt-zk/examples/sigma_timing.rs | CONVERT (already a tool) | A16 per-proof/per-attempt timing; `cargo run --release -p jolt-zk --features whiteout-diagnostics --example sigma_timing`. Reuses the dudect harness, no extra surface. |
| jolt-zk/src/whiteout/weights.rs::factored_dense_sum_matches_the_written_out_vector | KEEP | Factored weights against the written-out vector (independent ground truth), both grouped and straddling paths. |
| jolt-zk/tests/whiteout_params.rs (9 tests) | KEEP | Pins headline σ/digits/box bound/attempt count and the binding radius to closed-form values; refusal of out-of-box responses and malformed tables. |
| jolt-zk/tests/whiteout_sample.rs (2 tests) | KEEP | Only check that pad masks have zero constant term and uniform other coefficients. |
| jolt-zk/tests/whiteout_zeroize.rs (3 tests) | KEEP | Only check that Gaussian chunk seeds, box digits and pad-commitment copies are zeroed. |
| jolt-zk/tests/blindfold_backend.rs (4 tests) | KEEP | BlindFold over committed pads: completeness, prover refusal, residual-commitment and any-byte tamper. |
| jolt-zk/src/masking/tests.rs (17 tests) | KEEP | Masking front end: row satisfaction, per-value constraint, degree schedule, shape-vs-values domain agreement (two production domains), pad exhaustion, masked-channel refusals, rerun classification. |
| jolt-zk/src/masking/pads.rs::CommittedPads::uncommitted (`test-support`) | KEEP | Used by Akita's masked-prefix and zk e2e tests. |
| jolt-crypto/src/lattice/tests.rs (12 tests) | KEEP | Ajtai map vs schoolbook, limb split edges, bound refusal, frozen commit digests, headline randomness sizes. |
| jolt-crypto/src/sample/{gaussian,rejection,fixed,stats}.rs tests (20) | KEEP | Constants vs mpmath, exact pmf/law, rejection rate, thread-count independence, tail bound. Statistical ground truth for the sampler. |
| jolt-crypto/src/challenge/sampler/constant_time.rs::matches_the_verifier_sampler | KEEP | CT prover sampler must equal the verifier's sampler (FS soundness). |
| jolt-crypto/examples/sample_dudect.rs + feature `sample-diagnostics` | CONVERT (already a tool) | dudect for the Gaussian sampler, rejection rule and parallel fill; documented in its header. |
| jolt-transcript/src/fork.rs tests (3) | KEEP | Attempt forks keyed by seed and nonce; known answer. |
| jolt-transcript/tests/private.rs (3) | KEEP | Private-message seam round trip and binding (M3a). |
| jolt-dit/src/lib.rs tests (3) | KEEP | DIT set and restored per thread and on every pool worker; only signal for A11. |
| jolt-akita/src/whiteout/tests.rs (5 tests) | KEEP | Masked opening verifies and rejects a flipped byte at the jolt-akita layer; ranks pinned to Akita's frozen tables and Σ-binding brackets; persisted catalog/response-plan validation. |
| jolt-prover/src/akita/whiteout.rs::{faults, secrets} (`prover-fixtures`) | KEEP (flag) | Test seams compiled only with `prover-fixtures`; sole support for HT3 rerun and compensating-shift soundness tests in jolt-verifier. |
| jolt-prover/src/blindfold.rs::secrets::ZkSecrets (`prover-fixtures`) | KEEP | Supports `zk_e2e` shift probes. |
| jolt-prover/tests/whiteout_e2e.rs (7 tests) | KEEP | CI acceptance: honest Whiteout proofs verify (incl. transported preprocessing) and NARG tamper rejects across muldiv/advice/committed/field-inline; plans persist; hiding reopen. |
| jolt-prover/tests/zk_e2e.rs (6 tests) | KEEP | BlindFold mode acceptance, decoded-value shift and stage6b alias soundness probes. |
| jolt-verifier/tests/soundness/whiteout.rs (14 tests) | KEEP | Masked-site census (+ precommitted groups), per-census-class tamper, message sweep, compressed residual payloads, masked-level tampers, HT3 rerun, pad layout determinism, statement/mode tampers, final-opening binding. Only census signal. |
| jolt-verifier/tests/soundness/{zk,akita,checks}.rs | KEEP | Message sweeps, statement tampers and region transplants per mode; checked-input typing. |
| jolt-verifier/tests/completeness/{zk,akita}.rs (Whiteout rows) | KEEP | Verifier fixtures and golden masked shapes; `zk_muldiv_every_masked_value_is_constrained`. |
| jolt-verifier/tests/statistical_independence/zk.rs | KEEP | Only check that ZK proofs are independent of hiding messages. |
| jolt-verifier/tests/fs_inventory/zk-site-classes | KEEP | fs_obligations inventory of ZK site classes. |
| jolt-verifier/tests/support/akita_fixtures.rs (Whiteout cases) | KEEP | Fixture generators for the suites above, one owner. |
| scripts/whiteout_verify_profile.py | CONVERT (already a tool) | Verifier benchmark; commands in `specs/whiteout-o0-profile.md` (`python3 -I scripts/whiteout_verify_profile.py run …` / `report …`). |
| `#[ignore]` tests | n/a | None added by Whiteout in jolt (the two in jolt-algebra AVX-512 code belong to the unification). |
| Seeded RNG in ZK randomness paths | n/a | None: production paths draw from `OsRng`; `ChaCha20Rng::from_seed` in `GaussianSampler::fill_parallel` is seeded from the caller's CSPRNG by design (documented PRG assumption). Seeded RNGs appear only in tests, fixtures and diagnostics. |

## Akita

| item | class | reason |
|---|---|---|
| akita-pcs/src/zk/tests.rs stray doc line on the chunking test | REMOVE | Orphaned doc comment (commit 51734c93a). |
| akita-pcs/examples/compression_dudect.rs | CONVERT (already a tool) | Only dudect signal for secret-image compression (A36); command cited in its spec. |
| akita-pcs/examples/zk_distinguisher.rs | CONVERT (already a tool) | HT2 χ² two-witness distinguisher with leak controls; documented manual diagnostic. |
| akita-pcs/examples/zk_response.rs | CONVERT (already a tool) | Response bytes and time; reused by the W5-e, M1, M2 and O2b specs. |
| akita-planner/examples/response_plan.rs | CONVERT (already a tool) | Plan time and digest check for planner changes (O2b spec). |
| akita-sis-estimator/examples/sigma_binding_width_table.rs | KEEP | Production generator of the checked-in CSV and `generated.rs`. |
| akita-pcs feature `zk-diagnostics` | KEEP | Gates the dudect example only. |
| akita-pcs feature `test-support` (`UncheckedResponseProver`, pub `BackendCompression`, `round_polynomial` shift) | KEEP (flag) | Test seam in production prover code (shift is zero on honest rounds); sole support for HT4 consistent-cheat and forged-chain tests. |
| akita-cpu-backend feature `test-support`, `CpuBackend::with_hiding_rng` | KEEP (flag) | Only seeded hiding-RNG path; production `os()` never compiles it; zk test fixtures only. |
| akita-pcs/src/zk/tests.rs (9) | KEEP | W5-a checker per-item rejection, weights vs definition, chunking law, chain forgeries, setup refusal. |
| akita-pcs compression.rs `#[cfg(test)]` `apply_row`/`chain_rows`/`of_targets` | KEEP | Schoolbook definition oracle, not superseded code. |
| akita-pcs/tests/masked_prefix.rs (12) + instance.rs | KEEP | W3-a..e rows, tamper and row-break, counting replay, truncation, Stage-3 miss, multi-group, `r_a` freshness. |
| masked_prefix::every_shifted_message_breaks_a_row vs zk_end_to_end::shifting_any_slot_breaks_a_row | KEEP both (flag) | Overlap; masked_prefix adds the wrong-witness check on the test schedule, the e2e one runs on the checked-in production zk rows. |
| akita-pcs/tests/zero_knowledge.rs (13) | KEEP | Admission and pricing, randomness range checks, hiding freshness, export/rebuild, clear entry points refuse zk, S9 descriptor, one-shot nonce, no cut offload, budget split. |
| akita-pcs/tests/zk_end_to_end.rs (9) + harness.rs | KEEP | HT1 census, HT3 pad uniqueness and rerun, HT4 cheat, A47 `u_a` forgeries, response bound over every checked-in row. |
| akita-planner schedule_params_candidate/zk_ranks.rs (3) | KEEP | One-shot A ranks, Σ-binding ranks, no terminal at the masked or residual level. |
| akita-types stage_checks/tests.rs (2) | KEEP | Shared stage checks over jolt-zk masked domains. |
| akita-types instance_descriptor::setup_section_rejects_a_mismatched_zk_protocol_feature | KEEP | Wire-level descriptor negative check. |
| akita-types witness_scalar_len::measured_live_length_matches_materialized_setup_prefixed_layout | KEEP | Two live production paths must agree. |
| akita-params transcript_grinding::only_a_zero_knowledge_plan_admits_one_shot_responses_and_only_up_to_its_cut | KEEP | S1/S11 rule. |
| akita-verifier setup_contribution hiding tests (3) | KEEP | Randomness reaches the residual; D-randomness placement; MLE vs dense. |
| akita-pcs tests/dynamic_backends/private_cpu.rs | KEEP | zk test plumbing. |
| akita-params sigma_binding tests | KEEP | Pins cells to the certified CSV hash. |
| akita-params zk.rs tests | KEEP | Sizing law. |
| pub `witness_group_count`, `MaskedFoldSizing`, `ZK_FOLD_SECURITY_BITS`, `RESPONSE_SITE` | KEEP | Production callers inside the crate. |
| other Whiteout pub fns | KEEP | Each has an Akita or jolt production caller. |
| `#[ignore]` tests | n/a | None Whiteout-added. |
| planner diagnostics.rs counters | out of scope | Pre-existing on Akita main. |
| CI lanes `test-zk`, `whiteout-response-no-setup-cache` | KEEP | zk test wiring and I6/A46 feature-isolation checks. |

Akita uses from jolt-zk (all kept): `diagnostics::with_dit`,
`test_support::{SyntheticSigma, F, CheckingResponseProver}`,
`prove_sigma_unchecked`, `CommittedPads::uncommitted`, and the features
`test-support`, `whiteout-test-support`, `whiteout-diagnostics`.

## Totals

| removed | cumulative LOC removed |
|---|---|
| jolt (4 `[U2]` commits) | 233 deleted, 21 added (net 212) |
| Akita (1 `[U2]` commit) | 1 |
| both | 234 deleted |

## Flagged, not changed

- `jolt-zk/src/whiteout/mod.rs` module doc still says no ZK mode is exposed yet
  ("`ZkBackend for Whiteout` is W6") and that jolt-akita's `zk` enables
  `whiteout-prover`; jolt-akita now splits `zk`/`zk-prover`. Stale doc, not a
  test item.
- `jolt-zk/src/whiteout/pad_commitment.rs` says the commitment is "sent raw
  until W5 provides compression". Stale doc.
- `sigma.rs`'s `cfg(test)` rejected-attempt recorder and jolt-prover's
  `faults`/`secrets` seams are test hooks inside production files; they are
  compiled out of release builds and each is the only support for a soundness
  test, so they stay.
