# Committed Bytecode Stack

This stack splits PR [#1344](https://github.com/a16z/jolt/pull/1344) into reviewable PRs that remain buildable one step at a time.
The end-state implementation lives on `amir/bytecode-commitment-merged`; these stack branches are the review surface, not the source of truth for the original work.

## Invariants

- The first stack branch is based on `origin/main`.
- Each later stack branch is based on the previous stack branch.
- Every slice must compile at its stated checkpoint before the next slice is published.
- Full mode must keep working until committed mode is explicitly activated.
- PRs are opened ready for review, not draft.
- PR bodies must disclose that they were posted by Cursor assistant on behalf of Quang Dao, including the active model.
- The detailed implementation checklist is `specs/1344-committed-bytecode-program-image.md`.

## Stack Order

| # | Branch | Base | Contents |
|---|---|---|---|
| 00 | `stack/00-bytecode-stack-automation` | `main` | stack docs, branch plan, scripts, committed-bytecode spec |
| 01 | `stack/01-program-preprocessing-refactor` | `stack/00-bytecode-stack-automation` | full-mode `ProgramPreprocessing` wrapper |
| 02 | `stack/02-committed-preprocessing-model` | `stack/01-program-preprocessing-refactor` | committed preprocessing data model and validation |
| 03 | `stack/03-precommitted-geometry-substrate` | `stack/02-committed-preprocessing-model` | Dory/precommitted geometry substrate |
| 04 | `stack/04-stage6a-stage6b-full-mode` | `stack/03-precommitted-geometry-substrate` | full-mode Stage 6a/6b split |
| 05 | `stack/05-precommitted-claim-reduction-advice` | `stack/04-stage6a-stage6b-full-mode` | shared precommitted claim reduction and advice port |
| 06 | `stack/06-bytecode-program-image-reductions` | `stack/05-precommitted-claim-reduction-advice` | committed bytecode/program-image reductions and IDs |
| 07 | `stack/07-committed-mode-protocol-wiring` | `stack/06-bytecode-program-image-reductions` | committed-mode protocol activation |
| 08 | `stack/08-sdk-examples-transpiler-docs` | `stack/07-committed-mode-protocol-wiring` | SDK, examples, transpiler, and docs |
| 09 | `stack/09-cleanup-perf-regression-tests` | `stack/08-sdk-examples-transpiler-docs` | cleanup, perf fixes, and regression tests |

## Local Commands

Print the stack plan:

```bash
./stack/update-stack.sh
```

Check path coverage against the current end-state source:

```bash
./stack/update-stack.sh --check-coverage --from amir/bytecode-commitment-merged --base origin/main
```

Open or update ready PRs after branches have been pushed:

```bash
./stack/open-prs.sh --apply --base main --source amir/bytecode-commitment-merged
```

## Validation

Run focused checks on each slice according to `specs/1344-committed-bytecode-program-image.md`.
For protocol-changing slices, the minimum correctness checks are:

```bash
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host
cargo nextest run -p jolt-core muldiv --cargo-quiet --features host,zk
```

For the full stack tip, also run committed-mode and advice coverage:

```bash
cargo nextest run -p jolt-core muldiv_e2e_dory_committed_program_commitments --cargo-quiet --features host
cargo nextest run -p jolt-core muldiv_e2e_dory_committed_program_commitments --cargo-quiet --features host,zk
cargo nextest run -p jolt-core advice_e2e_dory --cargo-quiet --features host
RUST_MIN_STACK=33554432 cargo nextest run -p jolt-core advice_e2e_dory --cargo-quiet --features host,zk
cargo nextest run -p jolt-core final_advice_output_scale --cargo-quiet --features host
```

Strict linting before final publication:

```bash
cargo clippy --all --features host -q --all-targets -- -D warnings
cargo clippy --all --features host,zk -q --all-targets -- -D warnings
cargo fmt -q
```
