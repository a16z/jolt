# guest-sandbox

Template guest program for the soundness invariant's red-team harness.

During a red-team session (`cargo run --bin redteam -- --invariant soundness`), an AI agent produces a **unified diff** against this directory. The harness applies the patch in-place via `git apply`, then compiles the patched guest with `jolt build`, proves execution, and checks that the verifier rejects any dishonest output/panic claim. A `PatchGuard` RAII guard reverts the changes (via `git checkout .`) on drop, even if the check panics.

The default guest computes a simple wrapping hash of the input bytes (`h = 0; for b in input { h = h * 31 + b }`). The agent's goal is to patch it into a program that exposes a soundness bug in Jolt — i.e. one where the verifier accepts a proof paired with an incorrect output or panic flag.

## Structure

```
guest-sandbox/
├── Cargo.toml          # depends on jolt-sdk
└── src/
    ├── lib.rs           # #[jolt::provable] function (patch target)
    └── main.rs          # no_main entry point (rarely needs patching)
```

## Safety

The harness filters patches before applying them: any diff hunk referencing a path containing `..` is dropped, so the agent cannot modify files outside this directory.
