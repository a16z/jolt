# Task 08: Builder Emits Unrolled Batched Sumcheck Rounds

## Status: TODO

## Goal
Update `ModuleBuilder` (builder.rs) to emit the new granular ops instead of `Op::BatchedSumcheckRound`. This is the compiler-side counterpart to Task 05 (runtime).

## Current Builder Pattern
The builder likely has a method like `batched_sumcheck_rounds()` that emits one `Op::BatchedSumcheckRound` per round. We need to change this to emit the unrolled per-instance sequence.

## New Builder Method

```rust
impl ModuleBuilder {
    /// Emit a complete batched sumcheck as unrolled per-instance ops.
    ///
    /// For each round, emits:
    /// 1. BatchRoundBegin
    /// 2. Per-instance: Inactive / Resolve+Reduce / Bind+Reduce
    /// 3. Per-instance: BatchAccumulateInstance
    /// 4. BatchRoundFinalize
    /// 5. AbsorbRoundPoly + Squeeze
    pub fn batched_sumcheck_unrolled(
        &mut self,
        batch_idx: usize,
    ) {
        let bdef = &self.batched_sumchecks[batch_idx];
        let max_rounds = bdef.max_rounds;
        let max_evals = bdef.max_degree + 1;

        for round in 0..max_rounds {
            let bind_ch = if round > 0 { Some(prev_squeeze) } else { None };
            
            self.push_op(Op::BatchRoundBegin {
                batch: batch_idx,
                round,
                max_evals,
                bind_challenge: bind_ch,
            });

            for (inst_idx, inst) in bdef.instances.iter().enumerate() {
                if round < inst.first_active_round {
                    self.push_op(Op::BatchInactiveContribution {
                        batch: batch_idx,
                        instance: inst_idx,
                    });
                    continue;
                }

                let instance_round = round - inst.first_active_round;
                let (phase_idx, phase_start) = inst.phase_for_round(instance_round);
                let phase = &inst.phases[phase_idx];

                if instance_round == 0 || instance_round == phase_start {
                    // Phase boundary
                    if instance_round > 0 {
                        // Bind previous phase or PS materialize
                        let prev_phase = &inst.phases[phase_idx - 1];
                        if is_prefix_suffix(prev_phase) {
                            self.push_op(Op::PrefixSuffixBind { ... });
                            self.push_op(Op::PrefixSuffixMaterialize { ... });
                        } else {
                            self.push_op(Op::InstanceBindPreviousPhase { ... });
                        }
                    }

                    // Scalar captures
                    for cap in &phase.scalar_captures {
                        self.push_op(Op::ScalarCapture { ... });
                    }

                    // Init / resolve
                    if is_prefix_suffix(phase) {
                        self.push_op(Op::PrefixSuffixInit { ... });
                    } else {
                        self.push_op(Op::InstanceResolveInputs { ... });
                        if phase.segmented.is_some() {
                            // build outer eq op
                        }
                    }
                } else if let Some(ch) = bind_ch {
                    // Mid-phase: bind
                    if is_prefix_suffix(phase) {
                        self.push_op(Op::PrefixSuffixBind { ... });
                    } else {
                        // Bind handled by InstanceReduce's input tracking
                    }
                }

                // Reduce
                if is_prefix_suffix(phase) {
                    self.push_op(Op::PrefixSuffixReduce { ... });
                } else if phase.segmented.is_some() {
                    self.push_op(Op::InstanceSegmentedReduce { ... });
                } else {
                    self.push_op(Op::InstanceReduce { ... });
                }

                self.push_op(Op::BatchAccumulateInstance { ... });
            }

            self.push_op(Op::BatchRoundFinalize { batch: batch_idx });
            // AbsorbRoundPoly + Squeeze emitted separately
        }
    }
}
```

## Key Design Decision: Bind Ownership
Currently `BatchedSumcheckRound` does binding internally with `bind_kernel_inputs_tracked()` which tracks already-bound polys to avoid double-binding shared buffers.

With unrolled ops, we have two options:
1. **Bind as separate op**: Emit `Op::Bind` for each instance's kernel inputs before `InstanceReduce`
2. **Bind inside InstanceReduce**: The `InstanceReduce` handler resolves+binds as needed

Recommendation: **Option 1** — explicit `Op::Bind` makes the data flow visible. The compiler can deduplicate binds for shared buffers at emit time (it knows which polys are shared).

## Test
```bash
cargo nextest run -p jolt-equivalence transcript_divergence --cargo-quiet
cargo nextest run -p jolt-equivalence --cargo-quiet
```

## Risk: Medium
The emit logic mirrors what the runtime currently does, but moved to compile time. Getting the ordering exactly right is critical for transcript parity.

## Dependencies: Tasks 04, 05, 06, 07
