# Parent-drift recovery, preregistered09:59UTC

The initial finalist epoch stopped after ONE verified parent and ZERO
candidate proofs:32.835085250s vs historical36.182954209s (-9.25%). Original
source/binary/controller/guest fingerprints, trace_len201327593 and T28 agree;
no performance override is set. System CPU time24.35->21.22s and reclaimed
pages10.165M->9.715M differ, but those process-wide counters do not establish
the cause of a prover-wall shift. Historical matrix cooling was30s, this
campaign120s; attribution remains unproven. Preserve the stopped epoch.

One explicit budget amendment, before any candidate result: one additional
parent calibration control, hence at most13proofs across the stopped epoch
and its continuation, <=45minutes of measurement/cooling execution. The new
twelve-proof cohort starts with a parent that must agree with32.835085250s
within3%; if not, STOP timing. If stable, this becomes the first parent in
the unchanged Fibonacci P,C,C,P, followed by P,C for the other four workloads.
The original stopped parent is a calibration control, not a scored candidate
or a discarded slow parent from the new pair. All observations stay visible.

All original evaluator,120s cooling,88GiB,zero swaps/watchdogs,180s process,
gain/transfer and3% paired-parent-drift gates remain unchanged. The fresh
Fibonacci anchor becomes32.835085250s only for this preregistered stability
check; other workloads retain the old5% historical-discrepancy stop. No
automatic retry, more calibration controls or candidate change is authorized.
The new cohort has its own40minute absolute deadline and immutable directory.
The same frozen release finalist and AOT guests are used; no source edits.

Implementation reuses run_finalist.py's complete cohort/evaluator and applies
one additional first-parent stability condition at the observer boundary.
The original observer performs every verification/resource/identity check
unchanged; the adapter only rejects an additional out-of-band parent.
