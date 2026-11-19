# The `ethereum-block` benchmark workload

The `ethereum-block` cases of the prover benchmark prove the guest of
[jeth](https://github.com/a16z/jeth). Its `validate_block` function validates
one Ethereum mainnet block statelessly: it checks the block's witness against
the parent state root, executes every transaction in revm, and recomputes the
state root. Its public output is the block hash and the gas used.

jeth is not a member of this workspace, and none of its source is in this
repository. This directory holds the inputs the cases prove.

## The pin

Root `Cargo.toml` pins jeth by full commit SHA in three adjacent
`[workspace.dependencies]` lines, `jeth-inlines-blake2f`,
`jeth-inlines-bls12-381` and `jeth-inlines-bn254`, and `Cargo.lock` records the
same commit. Those crates are the host side of jeth's own inline instructions.
`jolt-prover` links them under its `profiling` feature, because the tracer
needs them to expand and execute the guest.

`jolt-prover prepare --name ethereum-block --blocks <numbers>` builds the guest
from Cargo's checkout of that commit, so the guest and its inlines cannot come
from different commits
(`crates/jolt-prover/src/profile/ethereum_block.rs`):

1. It asks `cargo metadata --locked` where Cargo checked jeth out.
2. It copies jeth's root `Cargo.toml`, `crates/` and `library/` into the guest
   target directory. The build rewrites `crates/guest/Cargo.lock`, and Cargo's
   checkout must stay as fetched. It refuses a guest target directory outside
   this checkout: the copy takes its toolchain from this checkout, and
   `prepare` replaces the directory's `jeth` entry.
3. It writes a `[patch]` table into the copy that points jeth's ten Jolt
   dependencies at this checkout. Each benchmark leg therefore builds the guest
   against its own Jolt crates.
4. It builds `jeth-guest` with jeth's default features and the `validate_block`
   entry point, using this repository's toolchain. It passes
   `--config 'build.warnings="warn"'` to that Cargo build, which outranks the
   caller's environment: jeth's `ark-ff`, `zeth-mpt`, `jeth-core` and
   `jeth-guest` warn under this toolchain, and a caller that denies warnings,
   as CI does with `CARGO_BUILD_WARNINGS=deny`, would otherwise fail on them.
   It stops if the copy's lockfile still takes a Jolt package from Git.
5. It executes each requested block's `input.bin` and `advice.bin` in the built
   guest, and stops if the guest rejects them or outputs another block's hash.
6. It prints each block's trace row count and writes `guest.elf`,
   `input-<block>.bin` and `advice-<block>.bin`. If any block failed, it
   writes nothing.

The build downloads jeth and the guest's dependencies: 358 registry packages
and 40 Git packages, reth, `stateless`, ZeroOS and the arkworks fork among
them. jeth's guest lockfile holds 427 packages with the path packages. A
measurement reads only the prepared files.

jeth's default guest features include `keccak-census`, which prints seven
`keccak[...]` lines from inside each measured process.

## Inputs

Each block has two files. `input.bin` is jeth's input container for the block:
the block, its transaction signers and the execution witness, without the
contract codes that the guest's embedded code library already holds.
`advice.bin` is the advice tape the guest reads, the hints its advice functions
would compute. The guest verifies both, so `prepare` stops on a file that
does not fit the pinned jeth. It first prints `executing block <number>`. A
flipped input bit and another block's tape each then failed an advice check,
which panics in `tracer/src/instruction/virtual_assert_eq.rs`. Another block's
two files under this block's number pass the guest and fail with `the
checked-in input of block <number> holds another block`.

| Block | In jeth | Gas used | Transactions | Block hash |
|---:|---|---:|---:|---|
| 26227673 | frozen: `fixtures/cycles-blocks.json`, release `cycles-blocks-v1` | 10026232 | 135 | `0x4986ed4390cb3902fac0ee40e60b2409efbae03c4e306015a73729de65cc382e` |
| 26433846 | not frozen | 1967936 | 5 | `0x09993aae466d59802abd7b4c7dff92ff1517c328f59a8ef8ff4b5ea6f1c2ef6d` |

| File | Bytes | SHA-256 |
|---|---:|---|
| `26227673/input.bin` | 2482516 | `1f9e1ec671cd61815c65d410efa96c67c88145fcd1e3ad330d10616eba0115e2` |
| `26227673/advice.bin` | 283880 | `85a71946a0d0e5fa4144c13c6a6a97b1ec47e240978fe2423c99dd3650d83700` |
| `26433846/input.bin` | 574125 | `454013bb35d63e5079a4464220b369d6c778a835ae233c3cd37412db20e3dc94` |
| `26433846/advice.bin` | 57760 | `e8f92abe79d1f3bc21d4207e91e51411a2863767902c8ae213a0ff5915a15ea7` |

All four were generated on November 17, 2026 with jeth's own tools at commit
`fcb31dae41c70c391622a23922ee46524bdf0d79`.

Block 26227673 is the smallest of the ten blocks jeth freezes for its
cycle-count harness. Its raw files are the assets `26227673.block.rlp` (SHA-256
`ce00f79ec21a277514ad47469dd4b40d7ea85c1715840e144e87e9c40ca9bd95`) and
`26227673.witness.json` (SHA-256
`8ad5801de9708d2e0252f638d3c7d6fde8c74bf701a0eda3afc3539f08f58f53`) of release
`cycles-blocks-v1` of `a16z/jeth`. jeth's `fixtures/cycles-blocks.json` records
both hashes.

Block 26433846 is not part of jeth. Every frozen block needs a trace ceiling
of `2^28` or more: in run 44160749278 of the `Cycles` workflow of `a16z/jeth`,
which ran at commit `54d63c4bd876fb217c6670b47100e1dc5ad856f8` with jeth's own
pinned Jolt revision, block 26227673 traced 160206652 rows, above `2^27`, and
the nine others at least 349793403. The pull-request case therefore
uses a low-gas block. `jeth fetch --block 26433846` read it from jeth's default
endpoint, `https://docs-demo.quiknode.pro/`, with `debug_getRawBlock` and
`debug_executionWitness`. The raw files are not in
this repository: `block.rlp` is 15350 bytes with SHA-256
`214fad3274535edc43c4e990889b3f593293729033e88b56613727f39e046e60`, and
`witness.json` is 1845364 bytes with SHA-256
`856a83fadcda008b3a691022d00368210ecc36db71b231903f4446abe2d96603`.

## Regenerating an input

In a clone of jeth at the pinned commit:

```sh
cargo build --locked --release -p jeth-host
# DIR holds the block's block.rlp and witness.json, and a meta.json containing {}
./target/release/jeth repack --dir DIR
cargo run --locked --profile recorder --manifest-path crates/native-advice/Cargo.toml -- \
  DIR/input.bin DIR/advice.bin
```

Copy `DIR/input.bin` and `DIR/advice.bin` here and update the table above.
`jeth fetch --block N --out OUT` writes `OUT/N/` with the raw files and
`input.bin` for a block that an endpoint still serves.

## Moving the pin

```sh
sed -i "s/$OLD_SHA/$NEW_SHA/" Cargo.toml
cargo update --workspace
```

`cargo update --workspace` rewrites the three lock entries and nothing else.
Then:

1. Compare `MAX_INPUT_SIZE`, `MAX_OUTPUT_SIZE`, `HEAP_SIZE` and `STACK_SIZE` in
   `crates/jolt-prover/src/profile/ethereum_block.rs` with the
   `#[jolt::provable]` attribute of `validate_block` in jeth's
   `crates/guest/src/lib.rs`. No check compares them.
2. Run `prepare` for every block:

   ```sh
   cargo build --profile ci -p jolt -p jolt-prover --features jolt-prover/profiling
   JOLT_PATH=target/ci/jolt target/ci/jolt-prover prepare --name ethereum-block \
     --blocks 26227673,26433846 --guest-target-dir target/bench-guests --out target/ethereum-block
   ```

3. If `prepare` reports a Jolt package from Git, add the package to
   `JOLT_PACKAGES` in that file. If it rejects an input, jeth's container,
   code library or advice calls changed: regenerate that block's files from
   its raw files. If the raw files of block 26433846 are unavailable, no
   endpoint serves its witness any more and jeth has not frozen it, choose
   another low-gas block and give its cases new IDs in `suites.json`.
4. A different guest can trace a different number of rows. Compare the row
   count that `prepare` prints for each block with `2^scale_log2` of the
   block's cases in `suites.json`.

## A Jolt change that breaks jeth

The `test-prover-ethereum-block` CI job runs `prepare` for both blocks, so a
change that jeth's guest cannot build against, or that makes it reject a
checked-in input, fails on the pull request that makes it. jeth can adopt such
a change only after it is on Jolt's main. That pull request therefore removes
the `ethereum-block` cases from `suites.json` and that CI job, and the pull
request that moves the pin to a jeth commit built for the new Jolt restores
them.

A change that jeth's three inline crates cannot compile against fails every
build of `jolt-prover` with `profiling`. That pull request also removes the
three crates from the `profiling` feature and from `profile.rs`.
