# Infrastructure-only recovery before any 80k proof

The first 80k launch exited 101 in 0.01 seconds at RunLock::acquire, before
tracing or proving. Its raw log has no chain check, proof or timing marker.
The empty workload-local lock had the timestamp of the earlier killed
112,682-hash run (16:07 UTC); a process-table check found no benchmark or
build process alive. Preserve it as ../stale-workload.lock.saved. Preserve
the failed launch, its manifest and controller without edits.

Permit one infrastructure-only recovery of the user-approved 80k run. This
narrows the prior no-automatic-retry rule: no proof measurement has occurred,
and this is not a timing, memory, watchdog or correctness failure. Any such
failure in this recovered execution stops the request with no further run.
Keep the same binary, guest, input, digest, scale, security parameters, machine
lock, full 120-second cooldown, 180-second timeout, 88 GiB RSS stop and zero
swap requirement. No build or CPU run. Fresh manifest deadline: eight minutes
from preparation. Keep all outputs distinct and include both launch outcomes
in the final report. The proof controller and measurement observer are unchanged.
