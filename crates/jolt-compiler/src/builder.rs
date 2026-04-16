//! Module builder — reduces per-stage boilerplate for hand-written modules.
//!
//! Instead of constructing raw `Vec<Op>`, `Vec<KernelDef>`, `Vec<ChallengeDecl>`,
//! the builder manages indices automatically and provides ergonomic methods for
//! common patterns (squeeze, sumcheck rounds, bind+eval+flush, etc.).
//!
//! # Example
//!
//! ```ignore
//! let mut b = ModuleBuilder::new();
//! b.preamble();
//! b.commit(&[poly_a, poly_b], DomainSeparator::Commitment);
//!
//! b.begin_stage();
//! let tau = b.squeeze_n("tau", 27, ChallengeSource::FiatShamir { after_stage: 0 });
//! let kernel = b.add_kernel(KernelDef { ... });
//! b.sumcheck_round(kernel, 0, None);
//! // ...
//! let module = b.build(num_verifier_stages);
//! ```

use crate::checkpoint_lowering::lower_checkpoint_rule;
use crate::formula::BindingOrder;
use crate::ir::PolyKind;
use crate::module::{
    BatchIdx, BatchedSumcheckDef, ChallengeDecl, ChallengeIdx, ChallengeSource,
    CheckpointEvalAction, ClaimFormula, DomainSeparator, EvalMode, InputBinding, InstanceConfig,
    InstanceIdx, KernelDef, Module, Op, PolyDecl, RoundPolyEncoding, Schedule, VerifierOp,
    VerifierSchedule, VerifierStageIndex,
};
use crate::PolynomialId;

/// Build the per-slot update list for an `Op::CheckpointEvalBatch` at
/// `(round, suffix_len)` by lowering every rule in `config`.
fn build_checkpoint_batch(
    config: &InstanceConfig,
    r_x: ChallengeIdx,
    r_y: ChallengeIdx,
    round: usize,
) -> Vec<(usize, CheckpointEvalAction)> {
    let suffix_len =
        config.total_address_bits - (round / config.chunk_bits + 1) * config.chunk_bits;
    config
        .checkpoint_rules
        .iter()
        .enumerate()
        .filter_map(|(i, rule)| {
            lower_checkpoint_rule(rule, i, r_x, r_y, round, suffix_len).map(|a| (i, a))
        })
        .collect()
}

/// Ergonomic builder for constructing a [`Module`].
///
/// Tracks challenge/kernel/poly indices internally so callers don't need
/// to manually manage offsets.
pub struct ModuleBuilder {
    polys: Vec<PolyDecl>,
    challenges: Vec<ChallengeDecl>,
    ops: Vec<Op>,
    kernels: Vec<KernelDef>,
    batched_sumchecks: Vec<BatchedSumcheckDef>,
    verifier_ops: Vec<VerifierOp>,
    stage_count: usize,
}

impl ModuleBuilder {
    pub fn new() -> Self {
        Self {
            polys: Vec::new(),
            challenges: Vec::new(),
            ops: Vec::new(),
            kernels: Vec::new(),
            batched_sumchecks: Vec::new(),
            verifier_ops: Vec::new(),
            stage_count: 0,
        }
    }

    // Polynomial registration

    /// Register a polynomial and return its id.
    pub fn add_poly(
        &mut self,
        id: PolynomialId,
        name: &str,
        kind: PolyKind,
        num_vars: usize,
    ) -> PolynomialId {
        self.polys.push(PolyDecl {
            name: name.to_string(),
            kind,
            num_elements: 1 << num_vars,
            committed_num_vars: None,
        });
        id
    }

    // Challenge management

    /// Allocate a single challenge slot and return its index.
    pub fn add_challenge(&mut self, name: &str, source: ChallengeSource) -> ChallengeIdx {
        let idx = ChallengeIdx(self.challenges.len());
        self.challenges.push(ChallengeDecl {
            name: name.to_string(),
            source,
        });
        idx
    }

    /// Squeeze a challenge: allocate slot + emit Op::Squeeze. Returns challenge index.
    pub fn squeeze(&mut self, name: &str, source: ChallengeSource) -> ChallengeIdx {
        let idx = self.add_challenge(name, source);
        self.ops.push(Op::Squeeze { challenge: idx });
        idx
    }

    /// Squeeze N challenges sequentially. Returns the range of indices.
    pub fn squeeze_n(
        &mut self,
        prefix: &str,
        count: usize,
        source_fn: impl Fn(usize) -> ChallengeSource,
    ) -> Vec<ChallengeIdx> {
        (0..count)
            .map(|i| self.squeeze(&format!("{prefix}_{i}"), source_fn(i)))
            .collect()
    }

    /// Squeeze N Fiat-Shamir challenges for a given stage. Returns the range.
    pub fn squeeze_fiat_shamir(
        &mut self,
        prefix: &str,
        count: usize,
        after_stage: usize,
    ) -> Vec<ChallengeIdx> {
        self.squeeze_n(prefix, count, |_| ChallengeSource::FiatShamir {
            after_stage,
        })
    }

    /// Allocate an external challenge slot (value set by runtime, e.g. ScalarCapture).
    pub fn external_challenge(&mut self, name: &str) -> ChallengeIdx {
        self.add_challenge(name, ChallengeSource::External)
    }

    /// Current number of challenges (useful for computing offsets).
    pub fn num_challenges(&self) -> usize {
        self.challenges.len()
    }

    // Kernel management

    /// Register a kernel definition and return its index.
    pub fn add_kernel(&mut self, kdef: KernelDef) -> usize {
        let idx = self.kernels.len();
        self.kernels.push(kdef);
        idx
    }

    /// Current number of kernels (useful for computing offsets).
    pub fn num_kernels(&self) -> usize {
        self.kernels.len()
    }

    /// Mutable access to a kernel (e.g. to patch EqProject challenges after round loop).
    pub fn kernel_mut(&mut self, idx: usize) -> &mut KernelDef {
        &mut self.kernels[idx]
    }

    // Batched sumcheck management

    /// Register a batched sumcheck definition and return its index.
    pub fn add_batched_sumcheck(&mut self, bdef: BatchedSumcheckDef) -> BatchIdx {
        let idx = BatchIdx(self.batched_sumchecks.len());
        self.batched_sumchecks.push(bdef);
        idx
    }

    // Prover ops

    /// Emit a raw prover op.
    pub fn push_op(&mut self, op: Op) {
        self.ops.push(op);
    }

    /// Emit Op::Preamble.
    pub fn preamble(&mut self) {
        self.ops.push(Op::Preamble);
    }

    /// Emit Op::BeginStage.
    pub fn begin_stage(&mut self) {
        self.ops.push(Op::BeginStage {
            index: self.stage_count,
        });
        self.stage_count += 1;
    }

    /// Emit Op::Commit for a group of polynomials.
    pub fn commit(&mut self, polys: &[PolynomialId], tag: DomainSeparator, num_vars: usize) {
        self.ops.push(Op::Commit {
            polys: polys.to_vec(),
            tag,
            num_vars,
        });
    }

    /// Emit a single sumcheck round.
    pub fn sumcheck_round(
        &mut self,
        kernel: usize,
        round: usize,
        bind_challenge: Option<ChallengeIdx>,
    ) {
        self.ops.push(Op::SumcheckRound {
            kernel,
            round,
            bind_challenge,
        });
    }

    /// Emit AbsorbRoundPoly with an explicit encoding.
    pub fn absorb_round_poly(
        &mut self,
        num_coeffs: usize,
        tag: DomainSeparator,
        encoding: RoundPolyEncoding,
    ) {
        self.ops.push(Op::AbsorbRoundPoly {
            num_coeffs,
            tag,
            encoding,
        });
    }

    /// Emit a loop of standard sumcheck rounds: round → absorb → squeeze.
    /// Returns the vector of round challenge indices.
    ///
    /// The emitted op sequence per round is:
    ///   - `SumcheckRound { kernel, round, bind_challenge: prev }`
    ///   - `AbsorbRoundPoly { kernel, num_coeffs, SumcheckPoly }`
    ///   - `Squeeze { challenge }`
    pub fn sumcheck_rounds(
        &mut self,
        kernel: usize,
        num_rounds: usize,
        num_coeffs: usize,
        stage: VerifierStageIndex,
        challenge_prefix: &str,
        round_offset: usize,
    ) -> Vec<ChallengeIdx> {
        let mut indices = Vec::with_capacity(num_rounds);
        for round in 0..num_rounds {
            let bind = if round > 0 {
                Some(indices[round - 1])
            } else {
                None
            };
            self.ops.push(Op::SumcheckRound {
                kernel,
                round,
                bind_challenge: bind,
            });

            let ch_r = self.add_challenge(
                &format!("{challenge_prefix}_{round}"),
                ChallengeSource::SumcheckRound {
                    stage,
                    round: round_offset + round,
                },
            );
            self.ops.push(Op::AbsorbRoundPoly {
                num_coeffs,
                tag: DomainSeparator::SumcheckPoly,
                encoding: RoundPolyEncoding::Compressed,
            });
            self.ops.push(Op::Squeeze { challenge: ch_r });
            indices.push(ch_r);
        }
        indices
    }

    /// Emit Op::Evaluate for a polynomial.
    pub fn evaluate(&mut self, poly: PolynomialId, mode: EvalMode) {
        self.ops.push(Op::Evaluate { poly, mode });
    }

    /// Emit Op::Bind for polynomials at a challenge.
    pub fn bind(&mut self, polys: &[PolynomialId], challenge: ChallengeIdx, order: BindingOrder) {
        self.ops.push(Op::Bind {
            polys: polys.to_vec(),
            challenge,
            order,
        });
    }

    /// Common pattern: bind a set of polynomials at each of N challenge points.
    pub fn bind_at_challenges(
        &mut self,
        polys: &[PolynomialId],
        challenges: &[ChallengeIdx],
        order: BindingOrder,
    ) {
        for &ch in challenges {
            self.bind(polys, ch, order);
        }
    }

    /// Emit RecordEvals.
    pub fn record_evals(&mut self, polys: &[PolynomialId]) {
        self.ops.push(Op::RecordEvals {
            polys: polys.to_vec(),
        });
    }

    /// Emit AbsorbEvals.
    pub fn absorb_evals(&mut self, polys: &[PolynomialId], tag: DomainSeparator) {
        self.ops.push(Op::AbsorbEvals {
            polys: polys.to_vec(),
            tag,
        });
    }

    /// Emit AbsorbInputClaim.
    pub fn absorb_input_claim(
        &mut self,
        formula: ClaimFormula,
        batch: BatchIdx,
        instance: InstanceIdx,
        inactive_scale_bits: usize,
    ) {
        self.ops.push(Op::AbsorbInputClaim {
            formula,
            tag: DomainSeparator::SumcheckClaim,
            batch,
            instance,
            inactive_scale_bits,
        });
    }

    /// Common pattern: evaluate + record + absorb for a set of polynomials.
    pub fn flush_evals(&mut self, polys: &[PolynomialId], tag: DomainSeparator) {
        for &poly in polys {
            self.evaluate(poly, EvalMode::FinalBind);
        }
        self.record_evals(polys);
        self.absorb_evals(polys, tag);
    }

    /// Emit unrolled granular ops for a batched sumcheck.
    ///
    /// Emits per-instance,
    /// per-round granular ops so the runtime is a flat dispatch loop.
    #[allow(clippy::too_many_arguments)]
    pub fn unrolled_batched_sumcheck_rounds(
        &mut self,
        batch: BatchIdx,
        num_rounds: usize,
        num_coeffs: usize,
        stage: VerifierStageIndex,
        challenge_prefix: &str,
        round_offset: usize,
    ) -> Vec<ChallengeIdx> {
        let bdef = self.batched_sumchecks[batch.0].clone();
        let max_evals = bdef.max_degree + 1;
        let mut indices = Vec::with_capacity(num_rounds);

        for round in 0..num_rounds {
            let bind = if round > 0 {
                Some(indices[round - 1])
            } else {
                None
            };

            // Begin round: zero combined, update claims from prev round.
            self.ops.push(Op::BatchRoundBegin {
                batch,
                round,
                max_evals,
                bind_challenge: bind,
            });

            // Per-instance ops.
            for (inst_idx_raw, inst) in bdef.instances.iter().enumerate() {
                let inst_idx = InstanceIdx(inst_idx_raw);
                if round < inst.first_active_round {
                    self.ops.push(Op::BatchInactiveContribution {
                        batch,
                        instance: inst_idx,
                    });
                    continue;
                }

                let instance_round = round - inst.first_active_round;
                let (phase_idx, phase_start) = inst.phase_for_round(instance_round);
                let phase = &inst.phases[phase_idx];
                let kernel = phase.kernel;
                let kdef = &self.kernels[kernel];
                let is_instance = kdef.instance_config.is_some();

                if is_instance {
                    // Address-decomposition instance: stateless ops.
                    let ic = kdef.instance_config.as_ref().unwrap();
                    let chunk_bits = ic.chunk_bits;
                    let sub_phase = instance_round / chunk_bits;
                    let round_in_sub = instance_round % chunk_bits;

                    if instance_round == 0 {
                        // First round, first sub-phase: init + scatter.
                        self.ops.push(Op::InitInstanceWeights {
                            r_reduction: ic.r_reduction.clone(),
                            num_prefixes: ic.num_prefixes,
                        });
                        emit_scatter_ops(&mut self.ops, kernel, 0, chunk_bits);
                    } else if round_in_sub == 0 {
                        // Sub-phase boundary: bind + expand + optional CP +
                        // capture registries + scatter next sub-phase.
                        let ch = bind.unwrap();
                        self.ops.push(Op::Bind {
                            polys: ic.bindable_polys(),
                            challenge: ch,
                            order: BindingOrder::HighToLow,
                        });
                        self.ops.push(Op::ExpandingTableUpdate {
                            table: PolynomialId::ExpandingTable(sub_phase - 1),
                            challenge: ch,
                            current_len: 1 << (chunk_bits - 1),
                        });
                        if instance_round % 2 == 0 {
                            let updates = build_checkpoint_batch(
                                ic,
                                indices[round - 2],
                                indices[round - 1],
                                instance_round - 1,
                            );
                            if !updates.is_empty() {
                                self.ops.push(Op::CheckpointEvalBatch { updates });
                            }
                        }
                        // Capture registry CPs from bound-down P buffers.
                        self.ops.push(Op::CaptureScalar {
                            poly: PolynomialId::InstanceP(1, 0),
                            challenge: ic.registry_checkpoint_slots[0],
                        });
                        self.ops.push(Op::CaptureScalar {
                            poly: PolynomialId::InstanceP(0, 0),
                            challenge: ic.registry_checkpoint_slots[1],
                        });
                        self.ops.push(Op::CaptureScalar {
                            poly: PolynomialId::InstanceP(2, 0),
                            challenge: ic.registry_checkpoint_slots[2],
                        });
                        self.ops.push(Op::UpdateInstanceWeights {
                            expanding_table: PolynomialId::ExpandingTable(sub_phase - 1),
                            chunk_bits,
                            num_phases: ic.num_phases,
                            phase: sub_phase,
                        });
                        emit_scatter_ops(&mut self.ops, kernel, sub_phase, chunk_bits);
                    } else {
                        // Mid sub-phase: bind + expand + optional CP.
                        let ch = bind.unwrap();
                        self.ops.push(Op::Bind {
                            polys: ic.bindable_polys(),
                            challenge: ch,
                            order: BindingOrder::HighToLow,
                        });
                        self.ops.push(Op::ExpandingTableUpdate {
                            table: PolynomialId::ExpandingTable(sub_phase),
                            challenge: ch,
                            current_len: 1 << (round_in_sub - 1),
                        });
                        if instance_round >= 2 && instance_round % 2 == 0 {
                            let updates = build_checkpoint_batch(
                                ic,
                                indices[round - 2],
                                indices[round - 1],
                                instance_round - 1,
                            );
                            if !updates.is_empty() {
                                self.ops.push(Op::CheckpointEvalBatch { updates });
                            }
                        }
                    }

                    // Reduce.
                    let r_x_challenge = if instance_round % 2 == 1 {
                        Some(indices[round - 1])
                    } else {
                        None
                    };
                    self.ops.push(Op::ReadCheckingReduce {
                        kernel,
                        round: instance_round,
                        r_x_challenge,
                    });
                    self.ops.push(Op::RafReduce {
                        batch,
                        instance: inst_idx,
                        kernel,
                    });
                } else if instance_round == 0 || instance_round == phase_start {
                    // Phase boundary (non-decomp instance).
                    if instance_round > 0 {
                        let prev_phase = &inst.phases[phase_idx - 1];
                        let prev_kernel = prev_phase.kernel;
                        let prev_kdef = &self.kernels[prev_kernel];
                        let prev_is_decomp = prev_kdef.instance_config.is_some();

                        if prev_is_decomp {
                            // Decomp→standard transition: bind, capture, materialize.
                            let ch = bind.unwrap();
                            let ic = prev_kdef.instance_config.as_ref().unwrap();
                            let chunk_bits = ic.chunk_bits;
                            let last_sub_phase = ic.num_phases - 1;
                            self.ops.push(Op::Bind {
                                polys: ic.bindable_polys(),
                                challenge: ch,
                                order: BindingOrder::HighToLow,
                            });
                            self.ops.push(Op::ExpandingTableUpdate {
                                table: PolynomialId::ExpandingTable(last_sub_phase),
                                challenge: ch,
                                current_len: 1 << (chunk_bits - 1),
                            });
                            let prev_instance_round = instance_round - 1;
                            if prev_instance_round >= 1 && (prev_instance_round + 1) % 2 == 0 {
                                let updates = build_checkpoint_batch(
                                    ic,
                                    indices[round - 2],
                                    indices[round - 1],
                                    prev_instance_round,
                                );
                                if !updates.is_empty() {
                                    self.ops.push(Op::CheckpointEvalBatch { updates });
                                }
                            }
                            // Capture registry CPs.
                            self.ops.push(Op::CaptureScalar {
                                poly: PolynomialId::InstanceP(1, 0),
                                challenge: ic.registry_checkpoint_slots[0],
                            });
                            self.ops.push(Op::CaptureScalar {
                                poly: PolynomialId::InstanceP(0, 0),
                                challenge: ic.registry_checkpoint_slots[1],
                            });
                            self.ops.push(Op::CaptureScalar {
                                poly: PolynomialId::InstanceP(2, 0),
                                challenge: ic.registry_checkpoint_slots[2],
                            });
                            // Materialize output buffers.
                            self.ops.push(Op::MaterializeRA {
                                kernel: prev_kernel,
                            });
                            self.ops.push(Op::MaterializeCombinedVal {
                                kernel: prev_kernel,
                            });
                        } else if let Some(ch) = bind {
                            self.ops.push(Op::InstanceBindPreviousPhase {
                                batch,
                                instance: inst_idx,
                                kernel: prev_kernel,
                                challenge: ch,
                            });
                            let prev_carry_polys: Vec<_> =
                                prev_phase.carry_bindings.iter().map(|b| b.poly()).collect();
                            if !prev_carry_polys.is_empty() {
                                self.ops.push(Op::BindCarryBuffers {
                                    polys: prev_carry_polys,
                                    challenge: ch,
                                    order: prev_kdef.spec.binding_order,
                                });
                            }
                        }
                    }

                    // Scalar captures.
                    for cap in &phase.scalar_captures {
                        self.ops.push(Op::CaptureScalar {
                            poly: cap.poly,
                            challenge: cap.challenge,
                        });
                    }

                    // Emit per-binding materialization ops.
                    if let Some(seg) = &phase.segmented {
                        self.ops.push(Op::MaterializeSegmentedOuterEq {
                            batch,
                            instance: inst_idx,
                            segmented: seg.clone(),
                        });
                    }
                    let is_activation = instance_round == 0;
                    if is_activation {
                        for binding in &phase.carry_bindings {
                            self.ops.push(Op::Materialize {
                                binding: binding.clone(),
                            });
                        }
                        let expected_size = 1usize << kdef.num_rounds;
                        for binding in &kdef.inputs {
                            match binding {
                                InputBinding::Provided { .. } => {
                                    self.ops.push(Op::MaterializeUnlessFresh {
                                        binding: binding.clone(),
                                        expected_size,
                                    });
                                }
                                _ => {
                                    self.ops.push(Op::Materialize {
                                        binding: binding.clone(),
                                    });
                                }
                            }
                        }
                    } else {
                        for binding in &kdef.inputs {
                            self.ops.push(Op::MaterializeIfAbsent {
                                binding: binding.clone(),
                            });
                        }
                    }
                } else {
                    // Mid-phase: bind (non-decomp instance).
                    if let Some(ch) = bind {
                        self.ops.push(Op::InstanceBind {
                            batch,
                            instance: inst_idx,
                            kernel,
                            challenge: ch,
                        });
                        let carry_polys: Vec<_> =
                            phase.carry_bindings.iter().map(|b| b.poly()).collect();
                        if !carry_polys.is_empty() {
                            self.ops.push(Op::BindCarryBuffers {
                                polys: carry_polys,
                                challenge: ch,
                                order: kdef.spec.binding_order,
                            });
                        }
                    }
                }

                // Reduce (non-PS cases — PS reduce is emitted above).
                if !is_instance {
                    if phase.segmented.is_some() {
                        self.ops.push(Op::InstanceSegmentedReduce {
                            batch,
                            instance: inst_idx,
                            kernel,
                            round_within_phase: instance_round - phase_start,
                            segmented: phase.segmented.clone().unwrap(),
                        });
                    } else {
                        self.ops.push(Op::InstanceReduce {
                            batch,
                            instance: inst_idx,
                            kernel,
                        });
                    }
                }

                self.ops.push(Op::BatchAccumulateInstance {
                    batch,
                    instance: inst_idx,
                    max_evals,
                    num_evals: kdef.spec.num_evals,
                });
            }

            // Finalize round.
            self.ops.push(Op::BatchRoundFinalize { batch });

            // Absorb + squeeze (same as before).
            let ch_r = self.add_challenge(
                &format!("{challenge_prefix}_{round}"),
                ChallengeSource::SumcheckRound {
                    stage,
                    round: round_offset + round,
                },
            );
            self.ops.push(Op::AbsorbRoundPoly {
                num_coeffs,
                tag: DomainSeparator::SumcheckPoly,
                encoding: RoundPolyEncoding::Compressed,
            });
            self.ops.push(Op::Squeeze { challenge: ch_r });
            indices.push(ch_r);
        }
        indices
    }

    /// Release device buffers.
    pub fn release_device(&mut self, polys: &[PolynomialId]) {
        for &poly in polys {
            self.ops.push(Op::ReleaseDevice { poly });
        }
    }

    // Verifier ops

    /// Emit a raw verifier op.
    pub fn push_verifier_op(&mut self, op: VerifierOp) {
        self.verifier_ops.push(op);
    }

    // Build

    /// Consume the builder and produce a [`Module`].
    pub fn build(self) -> Module {
        let num_polys = self.polys.len();
        let num_challenges = self.challenges.len();
        Module {
            polys: self.polys,
            challenges: self.challenges,
            prover: Schedule {
                ops: self.ops,
                kernels: self.kernels,
                batched_sumchecks: self.batched_sumchecks,
            },
            verifier: VerifierSchedule {
                ops: self.verifier_ops,
                num_challenges,
                num_polys,
                num_stages: self.stage_count,
            },
        }
    }
}

fn emit_scatter_ops(ops: &mut Vec<Op>, kernel: usize, phase: usize, chunk_bits: usize) {
    ops.push(Op::SuffixScatter { kernel, phase });
    ops.push(Op::QBufferScatter { kernel, phase });
    ops.push(Op::MaterializePBuffers { kernel });
    ops.push(Op::InitExpandingTable {
        table: PolynomialId::ExpandingTable(phase),
        size: 1 << chunk_bits,
    });
}

impl Default for ModuleBuilder {
    fn default() -> Self {
        Self::new()
    }
}
