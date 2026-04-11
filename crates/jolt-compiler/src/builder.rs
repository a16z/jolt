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

use crate::formula::BindingOrder;
use crate::ir::PolyKind;
use crate::module::{
    BatchedSumcheckDef, ChallengeDecl, ChallengeSource, ClaimFormula, DomainSeparator, KernelDef,
    Module, Op, PolyDecl, Schedule, VerifierOp, VerifierSchedule, VerifierStageIndex,
};
use crate::PolynomialId;

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

    // ── Polynomial registration ──

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
        });
        id
    }

    // ── Challenge management ──

    /// Allocate a single challenge slot and return its index.
    pub fn add_challenge(&mut self, name: &str, source: ChallengeSource) -> usize {
        let idx = self.challenges.len();
        self.challenges.push(ChallengeDecl {
            name: name.to_string(),
            source,
        });
        idx
    }

    /// Squeeze a challenge: allocate slot + emit Op::Squeeze. Returns challenge index.
    pub fn squeeze(&mut self, name: &str, source: ChallengeSource) -> usize {
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
    ) -> Vec<usize> {
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
    ) -> Vec<usize> {
        self.squeeze_n(prefix, count, |_| ChallengeSource::FiatShamir {
            after_stage,
        })
    }

    /// Allocate an external challenge slot (value set by runtime, e.g. ScalarCapture).
    pub fn external_challenge(&mut self, name: &str) -> usize {
        self.add_challenge(name, ChallengeSource::External)
    }

    /// Current number of challenges (useful for computing offsets).
    pub fn num_challenges(&self) -> usize {
        self.challenges.len()
    }

    // ── Kernel management ──

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

    // ── Batched sumcheck management ──

    /// Register a batched sumcheck definition and return its index.
    pub fn add_batched_sumcheck(&mut self, bdef: BatchedSumcheckDef) -> usize {
        let idx = self.batched_sumchecks.len();
        self.batched_sumchecks.push(bdef);
        idx
    }

    // ── Prover ops ──

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
    pub fn sumcheck_round(&mut self, kernel: usize, round: usize, bind_challenge: Option<usize>) {
        self.ops.push(Op::SumcheckRound {
            kernel,
            round,
            bind_challenge,
        });
    }

    /// Emit AbsorbRoundPoly for a kernel.
    pub fn absorb_round_poly(&mut self, kernel: usize, num_coeffs: usize, tag: DomainSeparator) {
        self.ops.push(Op::AbsorbRoundPoly {
            kernel,
            num_coeffs,
            tag,
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
    ) -> Vec<usize> {
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
                kernel,
                num_coeffs,
                tag: DomainSeparator::SumcheckPoly,
            });
            self.ops.push(Op::Squeeze { challenge: ch_r });
            indices.push(ch_r);
        }
        indices
    }

    /// Emit a loop of batched sumcheck rounds: round → absorb → squeeze.
    /// `absorb_kernel` determines which kernel's metadata is used for AbsorbRoundPoly.
    /// Returns the vector of round challenge indices.
    #[allow(clippy::too_many_arguments)]
    pub fn batched_sumcheck_rounds(
        &mut self,
        batch: usize,
        num_rounds: usize,
        num_coeffs: usize,
        absorb_kernel: usize,
        stage: VerifierStageIndex,
        challenge_prefix: &str,
        round_offset: usize,
    ) -> Vec<usize> {
        let mut indices = Vec::with_capacity(num_rounds);
        for round in 0..num_rounds {
            let bind = if round > 0 {
                Some(indices[round - 1])
            } else {
                None
            };
            self.ops.push(Op::BatchedSumcheckRound {
                batch,
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
                kernel: absorb_kernel,
                num_coeffs,
                tag: DomainSeparator::SumcheckPoly,
            });
            self.ops.push(Op::Squeeze { challenge: ch_r });
            indices.push(ch_r);
        }
        indices
    }

    /// Emit Op::Evaluate for a polynomial.
    pub fn evaluate(&mut self, poly: PolynomialId) {
        self.ops.push(Op::Evaluate { poly });
    }

    /// Emit Op::Bind for polynomials at a challenge.
    pub fn bind(&mut self, polys: &[PolynomialId], challenge: usize, order: BindingOrder) {
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
        challenges: &[usize],
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
    pub fn absorb_input_claim(&mut self, formula: ClaimFormula, batch: usize, instance: usize) {
        self.ops.push(Op::AbsorbInputClaim {
            formula,
            tag: DomainSeparator::SumcheckClaim,
            batch,
            instance,
        });
    }

    /// Common pattern: evaluate + record + absorb for a set of polynomials.
    pub fn flush_evals(&mut self, polys: &[PolynomialId], tag: DomainSeparator) {
        for &poly in polys {
            self.evaluate(poly);
        }
        self.record_evals(polys);
        self.absorb_evals(polys, tag);
    }

    /// Release device buffers.
    pub fn release_device(&mut self, polys: &[PolynomialId]) {
        for &poly in polys {
            self.ops.push(Op::ReleaseDevice { poly });
        }
    }

    // ── Verifier ops ──

    /// Emit a raw verifier op.
    pub fn push_verifier_op(&mut self, op: VerifierOp) {
        self.verifier_ops.push(op);
    }

    // ── Build ──

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

impl Default for ModuleBuilder {
    fn default() -> Self {
        Self::new()
    }
}
