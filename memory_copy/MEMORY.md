# Jolt Project Memory

## Current Work
- [project_target_architecture.md](project_target_architecture.md) — ML compiler philosophy: compiler=protocol, runtime=dumb executor

## References
- [reference_jolt_equivalence.md](reference_jolt_equivalence.md) — jolt-equivalence as debugging sandbox

## Feedback
- [feedback_compiler_runtime_split.md](feedback_compiler_runtime_split.md) — Compiler=protocol, runtime=dumb executor; extend kernels, never escape hatches
- [feedback_parallel_compilation.md](feedback_parallel_compilation.md) — sleep 30s on compilation errors from other agents
- [feedback_no_git_stash.md](feedback_no_git_stash.md) — NEVER use git stash in agents
- [feedback_minimal_abstractions.md](feedback_minimal_abstractions.md) — Occam's razor, first principles
- [feedback_debug_test_files.md](feedback_debug_test_files.md) — Add debug test files freely when breaking transcript frontier
- [feedback_jolt_core_write_access.md](feedback_jolt_core_write_access.md) — Full read/write access to jolt-core for instrumentation
- [feedback_no_test_bypass.md](feedback_no_test_bypass.md) — Poly construction in jolt-zkvm, not injected from tests
- [feedback_endianness_bugs.md](feedback_endianness_bugs.md) — Vigilant about endianness bugs; very common in this codebase
- [feedback_no_push.md](feedback_no_push.md) — NEVER git push, force push, reset hard, or any destructive git op
- [feedback_simplicity_heuristic.md](feedback_simplicity_heuristic.md) — If it feels complex, find the simpler abstraction
- [feedback_dead_code_brevity.md](feedback_dead_code_brevity.md) — Delete dead code (don't suppress); minimize verbosity
