# jolt-wrapper

Status: stub.

`jolt-wrapper` is reserved for future verifier wrapping work, such as
transpiling or embedding the generated Jolt verifier into another proof system.
It is intentionally out of scope for the initial full-`Fr`, non-zk
Jolt-on-Bolt PR stack.

Do not add implementation here in the first stack. Land only the placeholder
crate metadata needed to preserve the intended crate boundary; real wrapping
work should come after the generated verifier API is stable and audited.
