# Earlier supported quotient-free cutover

Historical experiment record from the original recursion branch. Its source pins,
proofs, and measurements do not validate the refreshed Blake-only epoch-6 stack.
Current integration and performance require new exact-source validation.

The saved Fibonacci proof selects K16 row
`f832156f615cc006fd49345832dc40b6f45230018d33d7dae0bea9c67c6abd0c`:
22 variables, one polynomial, no precommitted groups. Levels 0 and 1 use subring
coefficient packing and quotient lifting. Level 2 uses evaluation-trace openings
with quotient lifting; levels 3 and 4 already use reduced evaluation. No level
carries `setup_natural_len`. This identifies an earlier topology-eligible cutover;
it does not establish feasibility or a performance gain.

## Controlled experiment

Re-audit the original row under the current policy. Use the existing guided
planner to rebuild the same root geometry and recursive skeleton, requesting
reduced evaluation from level 2 onward. A modified guide is only a search hint;
it is never admitted as a schedule. The canonical planner must derive all witness
lengths, matrix capacities, ranks, response parameters, and transitions, then the
normal catalog audit must accept the resulting expanded row. Fail closed if the
fixed skeleton cannot support that choice. Do not relax packing or setup-edge
restrictions, change the field/security policy, or introduce new relation math.

If feasible, replace that scalar row in a scratch copy of the K16 catalog and
retain all other rows. Generate a new proof of the same program and public input
through the normal prover. The changed schedule/catalog digest necessarily changes
the transcript: the frozen original proof cannot be reused. Apply the retained
prepared-key, NTT-cache, and selected-catalog-view preparation to the new setup.
Use the same compiler, inline features, guest memory settings, and trace evaluator.
Record proof size, fold shapes, host proving cost, verification cycles, total rows,
and output. Accept only if both cycle metrics improve and the result repeats.

| Claim | Mechanism | Check |
| --- | --- | --- |
| Existing protocol only | Existing `ReducedEvaluation` relation and legal topology domains | No bypass of the level/packing/setup-edge checks |
| Valid schedule, not mutated metadata | Canonical guided search and catalog admission | Recompute lengths and pass normal semantic/security audit |
| Same statement | Unchanged Fibonacci program/input and public device values | Compare generated statement with the frozen fixture |
| Correct proof/setup binding | Normal prover, transported host verifier, guest verifier | Host success and guest output 1 |
| Performance gain | Fixed evaluator and retained preparation | Whole-trace A/B and repeat; no inference from witness size alone |

Initial probe is temporary. Promote an intentional planner/generator interface
only if the measured candidate earns it; otherwise retain the rejection evidence
and remove the probe. The existing 73.43M factor-copy candidate is independent of
this experiment. Full outer recursion proving remains unmeasured.

## Result: rejected

The guided solve succeeded in about 0.4 ms and the normal catalog audit accepted
it. Root geometry and fold count stayed fixed. The level-2 output shrank from
379,328 to 357,952 field elements; the next output shrank from 217,088 to 208,896.
The terminal input stayed at 112,128. Serialized proof bytes fell only from
81,431 to 81,418.

Regenerating the control with the unchanged catalog reproduced the original
6,253,088-byte frozen stream and its SHA-256 exactly. The candidate used the same
program and identical 93-byte public device record. Both proofs verified on host.
After identical prepared-catalog conversion, the same captured guest ELF and
`18-harness` produced:

| Schedule | Verification cycles | Total rows | Output |
| --- | ---: | ---: | ---: |
| Original cutover at level 3 | 72,735,395 | 76,308,524 | 1 |
| Guided cutover at level 2 | 73,500,354 | 77,073,351 | 1 |

The candidate adds 764,959 verification cycles. Its smaller intermediate witnesses
do not establish a verifier improvement. This rejects the fixed-skeleton candidate,
not every possible schedule with an earlier cutover. A different geometry or depth
would require a separate canonical search and measurement.

The temporary planner helper and example were removed; production schedules are
unchanged. Raw control/candidate inputs, catalog, source patches, and logs remain in
`/private/tmp/guest-optimization-campaign/26-*`.
