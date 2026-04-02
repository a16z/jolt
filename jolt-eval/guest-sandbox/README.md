# guest-sandbox

Template guest program for the soundness invariant's red-team harness.

During a red-team session (`cargo run --bin redteam -- --invariant soundness`), an AI agent produces a **unified diff** against this directory. The harness copies the template to a temp directory, applies the patch, compiles the patched guest with `jolt build`, then proves execution and checks that the verifier rejects any dishonest output/panic claim.

The default guest is an identity function (`input → input`). The agent's goal is to patch it into a program that exposes a soundness bug in Jolt — i.e. one where the verifier accepts a proof paired with an incorrect output or panic flag.

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
