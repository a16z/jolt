# Modular Crates PR Stack

Stacked PRs for incrementally landing the `jolt-v2` modular crate rewrite into `main`.
Managed with [Graphite](https://graphite.dev).

## Stack order

Each PR adds one crate. Dependencies are satisfied by the stack below.

| # | Branch | Crate | Key deps |
|---|--------|-------|----------|
| 0 | `jolt-v2/scaffolding` | — (workspace lints, CI, deps) | — |
| 1 | `jolt-v2/jolt-field` | `jolt-field` | ark-ff, ark-bn254 |
| 2 | `jolt-v2/jolt-profiling` | `jolt-profiling` | tracing |
| 3 | `jolt-v2/jolt-transcript` | `jolt-transcript` | jolt-field, blake2, sha3 |
| 4 | `jolt-v2/jolt-instructions` | `jolt-instructions` | jolt-field |
| 5 | `jolt-v2/jolt-poly` | `jolt-poly` | jolt-field |
| 6 | `jolt-v2/jolt-crypto` | `jolt-crypto` | jolt-field, jolt-transcript |
| 7 | `jolt-v2/jolt-host` | `jolt-host` | jolt-instructions, common, tracer |

## Workflow

### Initial setup (one-time)

```bash
./scripts/sync-stack.sh create
./scripts/sync-stack.sh submit
```

### After changing crate code on `refactor/crates`

```bash
# Commit your changes on refactor/crates first
git add -A && git commit -m "..."

# Update the stack
./scripts/sync-stack.sh sync

# Push updates
./scripts/sync-stack.sh submit
```

### Inspecting the stack

```bash
gt log short          # Show the full stack
gt checkout <branch>  # Jump to a specific level
cargo clippy -p <crate> --message-format=short -q -- -D warnings
```

### Adding a new crate to the stack

1. Edit `scripts/sync-stack.sh` — add an entry to the `STACK` array
2. Run `./scripts/sync-stack.sh sync` to create the new branch
3. Run `./scripts/sync-stack.sh submit` to open its PR

## Crate review status

Tracks production-readiness review across all crates on `refactor/crates`.

| Crate | Review | Confidence | Notes |
|-------|--------|------------|-------|
| `jolt-field` | - | - | |
| `jolt-profiling` | - | - | |
| `jolt-transcript` | - | - | |
| `jolt-instructions` | - | - | |
| `jolt-poly` | - | - | |
| `jolt-crypto` | DONE | 8.5/10 | ec/ restructure, VectorCommitment extends Commitment, DeriveSetup trait, generators removed from PairingGroup |
| `jolt-openings` | DONE | 9/10 | traits.rs->schemes.rs, blanket OpeningReduction, comments cleaned |
| `jolt-dory` | DONE | 8.5/10 | dead code deleted, centralized transmutes, DeriveSetup, transcript fix |
| `jolt-hyperkzg` | DONE | - | updated as part of DeriveSetup migration, not fully reviewed |
| `jolt-sumcheck` | - | - | |
| `jolt-r1cs` | - | - | |
| `jolt-compiler` | - | - | |
| `jolt-compute` | - | - | |
| `jolt-cpu` | - | - | |
| `jolt-metal` | - | - | |
| `jolt-hybrid` | - | - | |
| `jolt-witness` | - | - | |
| `jolt-verifier` | - | - | |
| `jolt-host` | - | - | |
| `jolt-zkvm` | - | - | |
| `jolt-equivalence` | - | - | test-only |
| `jolt-blindfold` | - | - | stub |
| `jolt-wrapper` | - | - | stub |

## Notes

- **jolt-core stays**: all existing code compiles unchanged. The final PR removes it.
- **Workspace lints are opt-in**: only crates with `[lints] workspace = true` get pedantic clippy.
- **Workspace dep versions unchanged**: ark-* still uses the fork, bincode is still v1 at workspace level. Crates needing different versions use inline deps (e.g., jolt-host uses bincode v2 inline).
- **Sync = force-push**: `gt restack` rewrites branch history. Line-level review comments may be orphaned; PR-level comments survive.
