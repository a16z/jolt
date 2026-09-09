# Verifier testing gates

## Fiat-Shamir soundness

The `fs-obligations` and `fs-attacks-smoke` jobs protect the Fiat-Shamir
soundness of `jolt-verifier`. They run independently for Dory clear, Dory ZK,
and Akita clear. Akita ZK is compile-probed and must be added to the matrix as
soon as that verifier combination becomes supported.

An attack test is valid only when all four steps hold:

1. The original fixture verifies and records typed challenge calls.
2. A coordinated mutation makes an individual protocol claim false.
3. Verification with the recorded challenges replayed succeeds. This proves the
   mutation preserves the verifier's algebraic checks when Fiat-Shamir binding
   is removed.
4. Verification with the production transcript changes a relevant challenge and
   rejects.

Production acceptance is a soundness finding. Frozen-challenge rejection is a
coverage failure: another check prevented the test from isolating the claimed
Fiat-Shamir defense. A changed transcript schedule or prover/verifier agreement
is not a security oracle.

Run the complete local matrix with:

```bash
scripts/ci/fs-soundness.sh
```

`fs_obligations` assigns stable identities to transcript absorption expressions,
challenge-shaped calls, scope annotations, generated sumcheck batching draws,
and serialized verifier inputs in the production dependency closure. The
absorption inventory includes normalized call expressions, so changing either a
label or its bound value requires review. A reviewed protocol change may
regenerate the inventories with:

```bash
JOLT_FS_BLESS=1 cargo nextest run -p jolt-verifier \
  --test fs_obligations --features fs-audit --cargo-quiet
```

Review the resulting diff as a list of new or removed security obligations.
Never regenerate it merely to make CI pass.

## Akita committed-program batching

Committed-program Akita preprocessing must provision one native grouped
opening for every direct bytecode chunk, the program image, optional advice,
and the main trace. The largest admitted statement has 256 chunks plus the
image, two advice objects, and the trace: 260 groups/polynomials. The schedule
registry plans only the setup's final arity, with at most four rows for the
reachable advice-presence combinations. Its 128-row bound applies to one
provisioning request, not to the process cache.

Run the focused committed-program gate with:

```bash
cargo nextest run -p jolt-prover muldiv_e2e_akita_committed_program \
  --features akita,prover-fixtures --cargo-quiet
```

Run the schedule and catalog gates with:

```bash
cargo nextest run -p jolt-akita --cargo-quiet
cargo nextest run -p jolt-akita --run-ignored all \
  -E 'test(catalogs_match_planner_regeneration)' --cargo-quiet
```

Failures at the 128-row or 260-group shape limit are protocol-capacity
failures. Do not work around them by lowering the public 256-chunk limit or
selecting a different committed-program encoding.
