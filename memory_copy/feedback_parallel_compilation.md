---
name: parallel agent compilation retry
description: When compilation fails due to another agent's in-progress changes, sleep 30s and retry instead of investigating
type: feedback
---

When encountering compilation errors, sleep for 30s to let the other parallel agent finish their changes, then retry.

**Why:** Multiple agents often work on the same codebase in parallel. A compilation failure may be caused by another agent's incomplete changes to a shared dependency (e.g., jolt-ir missing module files). Waiting lets the other agent finish.

**How to apply:** On any `cargo check`/`cargo clippy`/`cargo build` failure that looks like it's in a crate you didn't touch, sleep 30s and retry before investigating further.
