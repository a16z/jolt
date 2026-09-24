# Testing gates

## Guest × mode acceptance matrix

`crates/jolt-prover/tests/e2e_matrix.rs` holds one guest table per instruction
profile. The ordinary profile has nine cases: muldiv, fibonacci, memory-ops,
stdlib, sha2, sha3 through both its unaligned and aligned entry points,
advice-consumer, and btreemap. Enabling `field-inline` selects `field_ops`
and `inactive_muldiv`, covering both active field operations and an ordinary
guest proved under the field-inline protocol with no field activity.

The shared runner checks each guest's output against a natively computed
value, its panic status, trace bound, and field activity, then proves it with
the optimized backend. The compiled protocol selects Dory clear by default,
Dory ZK with `zk`, or Akita with `akita`. The mode is part of every test name
(`matrix::clear::sha2`, `matrix::zk::field_ops`,
`matrix::akita::inactive_muldiv`). CI runs each profile table in all three
modes, so a guest added to either table gains all three arms at once.
Specialized checks (tampering, committed programs, forced one-hot sizes)
stay in `zk_e2e.rs` and `akita_e2e.rs`; field-inline parity and tampering
checks stay in `field_inline_e2e.rs` and `akita_field_inline_e2e.rs`.

```bash
cargo nextest run -p jolt-prover --features prover-fixtures -E 'binary(e2e_matrix)' --cargo-quiet
cargo nextest run -p jolt-prover --features prover-fixtures,zk -E 'binary(e2e_matrix)' --cargo-quiet
cargo nextest run -p jolt-prover --features akita,prover-fixtures -E 'binary(e2e_matrix)' --cargo-quiet

# Field-inline profile
cargo nextest run -p jolt-prover --features prover-fixtures,field-inline -E 'binary(e2e_matrix)' --cargo-quiet
cargo nextest run -p jolt-prover --features prover-fixtures,field-inline,zk -E 'binary(e2e_matrix)' --cargo-quiet
cargo nextest run -p jolt-prover --features prover-fixtures,field-inline,akita -E 'binary(e2e_matrix)' --cargo-quiet
```

Add a guest by appending a row to its profile table: the example crate name,
its entry function when the crate has several, the `stack_size` from its
`#[jolt::provable]` attribute when it exceeds the 4 KiB default, `std: true`
when the guest crate enables `jolt`'s `guest-std` feature, postcard-encoded
inputs that keep the trace under the row's padded bound (2^16 by default),
and the postcard-encoded output computed natively in the test. Field cases
that execute field instructions must also set `field_inline_active: true`.

## Tamper rejection phases

The tamper harness asserts *where* a rejection fires: each manifest target in
`jolt-verifier`'s tamper manifest documents the verifier phase that is its
last line of defense, and `assert_verifier_fixture_tamper_rejects` fails if
the observed rejection maps to a later phase than documented.

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
```

Failures at the 128-row or 260-group shape limit are protocol-capacity
failures. Do not work around them by lowering the public 256-chunk limit or
selecting a different committed-program encoding.
