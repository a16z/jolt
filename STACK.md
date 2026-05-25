# Refactor Audit Prep Stack

This stack splits `refactor/audit-prep` into draft PRs that can be reviewed
independently while the integration branch continues to move. The stack branches
are generated from `refactor/audit-prep`; do not treat them as the source of
truth.

## Invariants

- `refactor/audit-prep` is the source branch.
- Stack branches are disposable materializations of slices from that source.
- Open stack PR branches are disposable materializations of slices from that
  source. Merged stack rows stay in `main` and are not rebuilt.
- Each open PR branch is based on the previous open PR branch, with the first
  open row based on `origin/main`.
- Root `Cargo.toml` and `Cargo.lock` are generated incrementally per PR. Do not
  restore the whole root manifest from `refactor/audit-prep` into early branches.
- Open every PR as draft until the verifier frontier is complete.

## Stack Order

| # | Branch | Base | Contents |
|---|---|---|---|
| 00 | `stack/00-stack-automation` | `origin/main` | stack docs, branch plan, update script, and GitHub Actions workflow |
| 01 | `stack/01-foundation-helpers` | `stack/00-stack-automation` | small helper changes in `jolt-field`, `jolt-poly`, `jolt-transcript`, `jolt-riscv` |
| 02 | `stack/02-lookup-table-core-abi` | `stack/01-foundation-helpers` | modular lookup-table enum ordering and core ABI parity test |
| 03 | `stack/03-public-io-preprocessing` | `stack/02-lookup-table-core-abi` | public I/O memory helpers in `common` and `jolt-program` |
| 04 | `stack/04-commitment-opening-infra` | `stack/03-public-io-preprocessing` | commitment, vector commitment, PCS, and opening-reduction infrastructure |
| 05 | `stack/05-jolt-claims-crate` | `stack/04-commitment-opening-infra` | new `jolt-claims` crate |
| 06 | `stack/06-jolt-r1cs-builder-lowering` | `stack/05-jolt-claims-crate` | `jolt-r1cs` builder/lowering/expression integration |
| 07 | `stack/07-committed-sumcheck-r1cs` | `stack/06-jolt-r1cs-builder-lowering` | committed sumcheck messages, domains, verifier changes, R1CS feature |
| 08 | `stack/08-jolt-blindfold-crate` | `stack/07-committed-sumcheck-r1cs` | new generic `jolt-blindfold` crate |
| 08a | `stack/08a-jolt-core-blindfold-hardening` | `stack/08-jolt-blindfold-crate` | `jolt-core` BlindFold construction hardening |
| 09 | `stack/09-jolt-verifier-crate` | `stack/08a-jolt-core-blindfold-hardening` | new `jolt-verifier` crate, verifier spec, boundary checks, fixtures, and verifier test config |
| 10 | `stack/10-jolt-prover-spec` | `stack/09-jolt-verifier-crate` | `specs/jolt-prover-model-crate.md` |
| 11 | `stack/11-extended-jolt-field-inline-wrapper-spec` | `stack/10-jolt-prover-spec` | extended Jolt / field inline / wrapper spec plus supporting recursion reference doc |
| 12 | `stack/12-selected-verifier-integration-spec` | `stack/11-extended-jolt-field-inline-wrapper-spec` | selected verifier integration spec |
| 13 | `stack/13-field-inline-protocol-spec` | `stack/12-selected-verifier-integration-spec` | field inline protocol spec, verifier-spec updates, formulas, R1CS hooks, RISC-V flags, and verifier protocol config |
| 14 | `stack/14-dory-assist-protocol-spec` | `stack/13-field-inline-protocol-spec` | Dory assist protocol spec, `jolt-hyrax`, Grumpkin/Fq crypto support, Dory-assist claim semantics, and Dory-specific PCS-assist hooks |
| 15 | `stack/15-wrapper-protocol-spec` | `stack/14-dory-assist-protocol-spec` | wrapper protocol spec, `jolt-wrapper`, variable-challenge sumcheck R1CS, transcript R1CS, and shared wrapper R1CS infrastructure |

The `jolt-core` BlindFold hardening PR carries the compatibility/security patch
that makes core BlindFold construction match the modular stack before the
`jolt-verifier` crate consumes it. The `jolt-verifier` PR carries the current
verifier frontier from `refactor/audit-prep`.

## Automatic Updates

Pushing to `origin/refactor/audit-prep` runs
[`.github/workflows/refactor-audit-stack.yml`](.github/workflows/refactor-audit-stack.yml).
The workflow starts at the first open stack row (`08` at the moment). Rows
`00` through `07` have merged to `main` and are intentionally not rebuilt.

The workflow:

1. checks out the pushed `refactor/audit-prep` commit;
2. rebuilds each open `stack/*` branch from the previous open stack branch;
3. restores the owned paths from `origin/refactor/audit-prep`;
4. applies the incremental root manifest changes for that stack point;
5. runs `cargo metadata` to refresh `Cargo.lock`;
6. checks that every path changed by `refactor/audit-prep` is assigned to a
   stack slice;
7. force-pushes all stack branches with lease;
8. creates or updates the draft PRs.

Once the PRs exist, GitHub will update them automatically when the workflow
force-pushes their branches.

## Growing Spec PRs Into Implementation PRs

The protocol spec PRs after PR 11 are allowed to grow implementation paths for
their feature area. Keep ownership disjoint by adding the implementation paths
to the same row as the relevant spec:

- PR 12: selected verifier integration, proof-shape validation, selected stage
  schedule, and selected computation export.
- PR 13: field-inline claims, R1CS rows, selected verifier hooks, trace/prover
  wiring, field-inline fixtures, and the `jolt-r1cs/field-inline` feature flag.
- PR 14: Dory-assist claims under
  `jolt-claims::protocols::dory_assist`, `jolt-hyrax`, Grumpkin/Fq crypto
  support for Hyrax row commitments, the Dory-assist verifier crate,
  Dory-specific PCS-assist verifier hooks, Dory-assist fixtures, and local
  handoff artifact ignores.
- PR 15: `jolt-wrapper`, wrapper assembly, verifier R1CS lowering,
  variable-challenge `jolt-sumcheck::r1cs`, transcript R1CS, non-native
  `jolt-r1cs` helpers, wrapper/R1CS composition tests, and SNARK backend
  integration.

If a later feature row names a path inside a directory owned by an earlier row,
the later row wins for changed files under that path. This keeps broad crate
bootstrap PRs stable while letting later implementation PRs own narrower
submodules in the same crate.

PR 13, PR 14, and PR 15 also own shared files that were introduced before those
spec rows existed. Earlier broad crate-bootstrap rows restore those files from
the `09ed244b` verifier-v1 baseline, then the owning feature row restores the
current source version. This keeps `jolt-verifier` v1 stable while
field-inline, Dory-assist, and wrapper work reviews with their respective
specs.

## Manual Materialization

Dry-run the stack:

```bash
./stack/update-stack.sh
```

Create or update one branch:

```bash
./stack/update-stack.sh --apply --only 08
```

Rebuild all stack branches from the source branch:

```bash
./stack/update-stack.sh --apply --rebuild --commit --push --cargo-metadata --check-coverage --from origin/refactor/audit-prep --start-at 08
```

The CI workflow runs the same command. Without `--commit`, the script leaves
changes unstaged for local inspection.

## Manifest Generation

The update script applies these root manifest changes in the branch where the
crate first appears:

- PR 05: add `crates/jolt-claims` to workspace members and add
  `jolt-claims = { path = "./crates/jolt-claims" }` to workspace dependencies.
- PR 06: add `jolt-r1cs = { path = "./crates/jolt-r1cs" }` to workspace
  dependencies if it is not already present.
- PR 08: add `crates/jolt-blindfold` to workspace members and add
  `jolt-blindfold = { path = "./crates/jolt-blindfold" }`.
- PR 08a: no root manifest changes; this slice patches existing `jolt-core`
  BlindFold construction before `jolt-verifier`.
- PR 09: add `crates/jolt-verifier` and `examples/advice-consumer/guest` to
  workspace members, add `jolt-verifier = { path = "./crates/jolt-verifier" }`,
  and add verifier-specific test fixture config.
- PR 13: add the `jolt-r1cs/field-inline` feature flag.
- PR 14: when present in the source ref, add `crates/jolt-hyrax` and
  `crates/jolt-dory-assist-verifier` to workspace members and add their
  workspace dependencies.
- PR 15: when present in the source ref, add `crates/jolt-wrapper` to workspace
  members and add `jolt-wrapper = { path = "./crates/jolt-wrapper" }`.

With `--cargo-metadata`, the script refreshes `Cargo.lock` after those manifest
changes.

## Validation

Use focused checks per branch while the stack is draft:

```bash
cargo metadata -q >/dev/null
cargo check -p jolt-claims -q
cargo check -p jolt-r1cs -q
cargo check -p jolt-sumcheck -q --features r1cs
cargo check -p jolt-blindfold -q
cargo check -p jolt-verifier -q
cargo check -p jolt-wrapper -q
```

For the full stack tip:

```bash
cargo clippy --all --features host -q --all-targets -- -D warnings
cargo clippy --all --features host,zk -q --all-targets -- -D warnings
cargo fmt -q
```

Use `cargo nextest`, not `cargo test`, once the stack is ready for correctness
checks.

## Opening Draft PRs

The workflow opens or updates draft PRs automatically. To do the same locally
after materializing and pushing the branches:

```bash
./stack/open-prs.sh --apply --base main --source refactor/audit-prep
```

If using a stacked-PR extension, create the same branch chain and run the
extension's submit command from the final stack branch.

## Updating From `refactor/audit-prep`

1. Commit WIP on `refactor/audit-prep`. The source must be pushed as a git ref.
2. Push:

   ```bash
   git push origin refactor/audit-prep
   ```

3. The workflow rebuilds and force-pushes all stack branches. To do the same
   locally:

   ```bash
   ./stack/update-stack.sh --apply --rebuild --commit --push --cargo-metadata --check-coverage --from origin/refactor/audit-prep --start-at 08
   ```

4. Compare the stack tip to the source branch:

   ```bash
   git diff --stat refactor/audit-prep..stack/15-wrapper-protocol-spec
   ```

Only intentional WIP exclusions or branch-order differences should remain.
