---
name: Debug with test files in jolt-equivalence
description: When stuck on transcript divergence, create focused debug test files in jolt-equivalence to instrument both jolt-core and jolt-zkvm side-by-side
type: feedback
---

When an op's computation is unclear or divergent, add a new test file in `crates/jolt-equivalence/tests/` to instrument the specific stuck point. jolt-equivalence is the sandbox — it can import both jolt-core and jolt-zkvm and compare intermediate values directly. Also instrument jolt-core as needed.

**Why:** Staring at code and guessing what values flow through is slow. A targeted test that runs both systems and prints intermediate values resolves ambiguity fast.

**How to apply:** Whenever the transcript divergence test shows a mismatch and the root cause isn't obvious from reading code, write a small test that computes the specific operation (e.g., uniskip polynomial) in both jolt-core and jolt-zkvm and compares the intermediate results (eq table entries, evaluation sums, interpolated coefficients, etc.).
