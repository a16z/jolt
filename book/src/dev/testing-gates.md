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
