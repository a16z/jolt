# jolt-backends

Backend compute rails for modular proving.

This crate owns backend traits plus request/result types. Protocol crates own
stage order, claims, transcript labels, and verifier-visible proof shape.

Structure rules:

- request/result contracts live by family, for example
  `commitments/request.rs`, `sumcheck/request.rs`, and
  `openings/request.rs`;
- concrete CPU code lives under `cpu/` and keeps request orchestration separate
  from hot compute helpers;
- CPU hot paths may be Jolt-specific and aggressively optimized, but they
  consume explicit requests and return slot-keyed results;
- root request/result contracts should stay hardware-agnostic and reasonably
  protocol-agnostic; protocol meaning enters through backend-local relation
  IDs, witness oracle refs, value slots, and caller-supplied request labels;
- CPU modules may split by protocol primitive when optimization demands it,
  but only behind the generic request family that scheduled the work;
- representation-specific commitment behavior, such as dense streaming versus
  one-hot sparse commitment, is hidden behind one CPU commitment accumulator
  path;
- `field-inline` and `zk` features are backend capability gates, not protocol
  schedulers.

The canonical CPU backend must preserve the current `jolt-core` prover's
streaming, memory, Dory hint, advice, BlindFold, and one-hot/RA fast paths.
