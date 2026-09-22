# Public fast-recursion delivery preparation

This source branch descends from frozen replay source c2372a0a9ffd2b37768e501b90037c146664757d.
Its Rust source is unchanged; dependency delivery is the only implementation
change. The twelve direct Akita dependencies pin markosg04/akita at
d0ee56a0b12db66b4c46a87c527169737694c4c7. No LayerZero publication and no sibling
Akita path patches are used. The three retained Jolt-source patches resolve to
crates/jolt-field, jolt-inlines/blake2, and jolt-inlines/ntt inside this checkout.

The companion's shared leaves pin the same Jolt c2372a0a source anchor. Planned
public fetch: git fetch https://github.com/a16z/jolt refs/tags/recursion-source-c2372a0a.
Verify FETCH_HEAD equals c2372a0a9ffd2b37768e501b90037c146664757d. The locally
prepared tag names that exact object; it has not been published by this work.
That historical commit is a source anchor, not a standalone delivery workspace.
The final Jolt branch is wrap/fast-recursion-public; the companion uses the same
branch name on markosg04/akita. Immutable Cargo revs, not branch tips, select code.

Git cross-references terminate at an older Jolt commit. At the package level,
normal Akita dependencies import the field/inline leaf crates, not the prover;
Jolt root patches unify those leaf identities. Metadata must still establish
this closure without sibling directories or external path dependencies. The
excluded Akita legacy recursion profile is a separate graph with its own lock
and validation; it is not the runner used for the recorded measurements.

Offline Cargo regenerated the lock against isolated local Git transport mirrors.
Locked default, measured inner/outer, and guest-inline metadata each resolve one
workspace jolt-field, no akita-field or shake package, and fourteen Akita packages
from the exact fork revision above. No external path packages appear. This is
local-mirror resolution, not public-fetch success. Compilation, narrow inline/
profile validation, and saved-artifact replay remain gates; no new proof
measurement was performed. Original full proof belongs to
f2e97d10/99fb06aa; serialized replay belongs to c2372a0a/99fb06aa. The companion
here retains additional public APIs, so historical timings are not measurements
of this delivery pair. All frozen binaries and artifacts remain intact.

The companion's excluded legacy profile uses Git-only patches to immutable native
source ancestor2bceddba86e07971bbd0538e46060df3caf3031b, whose library trees match
the final companion. This prevents the historical SDK anchor from reintroducing
pre-Blake LayerZero dependencies. Its distinct local/Git Akita identities remain
a compiler compatibility gate; they are not collapsed merely because their
source agrees. The main Jolt delivery graph has a single fork Akita owner.
