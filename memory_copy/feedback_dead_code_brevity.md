---
name: Remove dead code, maximize brevity
description: Delete dead code instead of #[allow(dead_code)]; minimize verbosity everywhere
type: feedback
originSessionId: f0719e20-9c07-4478-bf0f-7ad6f4e1a5cf
---
Remove dead code rather than clippy-allowing it. In general, strive to minimize code verbosity and maintain brevity.

**Why:** Dead code is cognitive overhead. Suppressions hide rot. Verbose code obscures intent.

**How to apply:** When encountering `#[allow(dead_code)]`, delete the dead item (field, function, struct) rather than keeping the suppression. When writing or refactoring code, prefer terse idiomatic Rust — fewer lines, less ceremony.
