---
name: jolt-core write access
description: We have write access to jolt-core and can instrument it as needed for debugging
type: feedback
---

We have full read AND write access to jolt-core, not just read access. Can instrument jolt-core as needed when debugging transcript divergences between jolt-zkvm and jolt-core.

**Why:** The goal is byte-identical Fiat-Shamir transcript match between jolt-zkvm runtime and jolt-core's embedded protocol. Sometimes need to add logging/diagnostics to jolt-core to understand its behavior.

**How to apply:** When debugging mismatches, don't hesitate to add temporary instrumentation (eprintln, test helpers) to jolt-core source files to extract reference values. jolt-equivalence crate is the primary sandbox for cross-system tests.
