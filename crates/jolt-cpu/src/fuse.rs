//! CPU backend fusion passes over the emitted op stream.
//!
//! Each pass rewrites `&[Op]` → `Vec<Op>` without changing prover
//! semantics: same Fiat-Shamir transcript, same polynomial commitments,
//! same final proof. Invoked from
//! [`CpuBackend::fuse_ops`](crate::CpuBackend::fuse_ops), which
//! [`jolt_compute::link`] calls once at [`Executable`](jolt_compute::Executable)
//! construction.
//!
//! Analogous to XLA's `HloFusion` or TVM's schedule rewriters: purely a
//! backend codegen concern, not a protocol concern.
//!
//! # Collapsing `Op::Reduce` emissions within a batch round
//!
//! During Phase B of the unified-reduce refactor, the compiler dual-emits
//! `Op::Reduce` alongside the legacy `Op::SumcheckRound`,
//! `Op::InstanceReduce`, and `Op::InstanceSegmentedReduce` ops. The runtime
//! still dispatches off the legacy ops and treats `Op::Reduce` as a no-op.
//! [`fuse_reduce_windows`] collapses every per-instance `Op::Reduce` emitted
//! inside a `BatchRoundBegin .. BatchRoundFinalize` window into one fused
//! `Op::Reduce { specs: [...] }` positioned between the window's prep and
//! its accumulate block — the same slot that Phase C will substitute for
//! today's legacy reduces.

use jolt_compiler::module::{BatchIdx, Op, ReduceSpec};

/// Rewrite `ops` so every `Op::Reduce` inside a `BatchRoundBegin ..
/// BatchRoundFinalize` window fuses into a single `Op::Reduce` with the union
/// of specs.
///
/// Non-`Op::Reduce` ops are preserved in-order: `Begin`, prep (everything
/// that isn't a reduce or an accumulate), the fused `Op::Reduce`, the
/// original accumulate block (`BatchAccumulateInstance` /
/// `BatchInactiveContribution`), and `Finalize`. A window with no
/// `Op::Reduce` ops passes through unchanged — no fused reduce is emitted.
///
/// Windows outside a `BatchRoundBegin .. BatchRoundFinalize` pair are
/// unchanged.
pub fn fuse_reduce_windows(ops: &[Op]) -> Vec<Op> {
    let mut out = Vec::with_capacity(ops.len());
    let mut i = 0;
    while i < ops.len() {
        if let Op::BatchRoundBegin { batch, .. } = &ops[i] {
            let end = find_finalize(ops, i, *batch)
                .unwrap_or_else(|| panic!("BatchRoundBegin without matching BatchRoundFinalize"));
            rewrite_reduce_window(&ops[i..=end], &mut out);
            i = end + 1;
        } else {
            out.push(ops[i].clone());
            i += 1;
        }
    }
    out
}

fn find_finalize(ops: &[Op], start: usize, batch: BatchIdx) -> Option<usize> {
    ops[start + 1..]
        .iter()
        .enumerate()
        .find_map(|(offset, op)| match op {
            Op::BatchRoundFinalize { batch: b } if *b == batch => Some(start + 1 + offset),
            _ => None,
        })
}

fn rewrite_reduce_window(window: &[Op], out: &mut Vec<Op>) {
    let (begin, rest) = window
        .split_first()
        .expect("rewrite_reduce_window: empty batch-round window");
    let (finalize, inner) = rest
        .split_last()
        .expect("rewrite_reduce_window: window missing BatchRoundFinalize");

    let mut prep: Vec<Op> = Vec::with_capacity(inner.len());
    let mut fused_specs: Vec<ReduceSpec> = Vec::new();
    let mut accums: Vec<Op> = Vec::new();

    for op in inner {
        match op {
            Op::Reduce { specs } => fused_specs.extend(specs.iter().cloned()),
            Op::BatchAccumulateInstance { .. } | Op::BatchInactiveContribution { .. } => {
                accums.push(op.clone());
            }
            _ => prep.push(op.clone()),
        }
    }

    out.push(begin.clone());
    out.extend(prep);
    if !fused_specs.is_empty() {
        out.push(Op::Reduce { specs: fused_specs });
    }
    out.extend(accums);
    out.push(finalize.clone());
}

#[cfg(test)]
mod tests {
    use super::*;
    use jolt_compiler::module::{
        BatchIdx, BufferRef, ChallengeIdx, InstanceIdx, ReduceAxes, ReduceDestination, ReduceSpec,
    };
    use jolt_compiler::PolynomialId;

    fn bidx(n: usize) -> BatchIdx {
        BatchIdx(n)
    }
    fn iidx(n: usize) -> InstanceIdx {
        InstanceIdx(n)
    }

    fn begin(batch: BatchIdx, round: usize) -> Op {
        Op::BatchRoundBegin {
            batch,
            round,
            max_evals: 4,
            bind_challenge: None,
        }
    }

    fn finalize(batch: BatchIdx) -> Op {
        Op::BatchRoundFinalize { batch }
    }

    fn accum(batch: BatchIdx, inst: InstanceIdx) -> Op {
        Op::BatchAccumulateInstance {
            batch,
            instance: inst,
            max_evals: 4,
            num_evals: 2,
        }
    }

    fn inactive(batch: BatchIdx, inst: InstanceIdx) -> Op {
        Op::BatchInactiveContribution {
            batch,
            instance: inst,
        }
    }

    fn reduce_spec(instance_slot: usize) -> ReduceSpec {
        ReduceSpec {
            kernel: 0,
            inputs: vec![BufferRef::Polynomial(PolynomialId::ExpandingTable(
                instance_slot,
            ))],
            axes: ReduceAxes::Flat,
            destination: ReduceDestination::Instance {
                batch: BatchIdx(0),
                instance: InstanceIdx(instance_slot),
            },
        }
    }

    fn reduce_op(instance_slot: usize) -> Op {
        Op::Reduce {
            specs: vec![reduce_spec(instance_slot)],
        }
    }

    #[test]
    fn no_batch_windows_passes_through() {
        let ops = vec![
            Op::Preamble,
            Op::Squeeze {
                challenge: ChallengeIdx(0),
            },
        ];
        let out = fuse_reduce_windows(&ops);
        assert_eq!(out.len(), 2);
        assert!(matches!(out[0], Op::Preamble));
        assert!(matches!(out[1], Op::Squeeze { .. }));
    }

    #[test]
    fn reduce_outside_batch_passes_through() {
        let ops = vec![
            Op::Preamble,
            reduce_op(0),
            Op::Squeeze {
                challenge: ChallengeIdx(0),
            },
        ];
        let out = fuse_reduce_windows(&ops);
        assert_eq!(out.len(), 3);
        assert!(matches!(out[0], Op::Preamble));
        assert!(matches!(out[1], Op::Reduce { ref specs } if specs.len() == 1));
        assert!(matches!(out[2], Op::Squeeze { .. }));
    }

    #[test]
    fn adjacent_reduces_collapse_into_one() {
        let b = bidx(0);
        let ops = vec![
            begin(b, 0),
            reduce_op(0),
            accum(b, iidx(0)),
            reduce_op(1),
            accum(b, iidx(1)),
            finalize(b),
        ];
        let out = fuse_reduce_windows(&ops);
        assert_eq!(out.len(), 5);
        assert!(matches!(out[0], Op::BatchRoundBegin { .. }));
        assert!(matches!(out[1], Op::Reduce { ref specs } if specs.len() == 2));
        assert!(matches!(out[2], Op::BatchAccumulateInstance { .. }));
        assert!(matches!(out[3], Op::BatchAccumulateInstance { .. }));
        assert!(matches!(out[4], Op::BatchRoundFinalize { .. }));
    }

    #[test]
    fn window_with_no_reduce_is_untouched() {
        let b = bidx(0);
        let ops = vec![
            begin(b, 0),
            inactive(b, iidx(0)),
            accum(b, iidx(1)),
            finalize(b),
        ];
        let out = fuse_reduce_windows(&ops);
        assert_eq!(out.len(), 4);
        assert!(matches!(out[0], Op::BatchRoundBegin { .. }));
        assert!(matches!(out[1], Op::BatchInactiveContribution { .. }));
        assert!(matches!(out[2], Op::BatchAccumulateInstance { .. }));
        assert!(matches!(out[3], Op::BatchRoundFinalize { .. }));
    }

    #[test]
    fn prep_preserved_accumulates_moved() {
        let b = bidx(0);
        let ops = vec![
            begin(b, 0),
            Op::Preamble,
            reduce_op(0),
            accum(b, iidx(0)),
            Op::Squeeze {
                challenge: ChallengeIdx(0),
            },
            reduce_op(1),
            finalize(b),
        ];
        let out = fuse_reduce_windows(&ops);
        assert_eq!(out.len(), 6);
        assert!(matches!(out[0], Op::BatchRoundBegin { .. }));
        assert!(matches!(out[1], Op::Preamble));
        assert!(matches!(out[2], Op::Squeeze { .. }));
        assert!(matches!(out[3], Op::Reduce { ref specs } if specs.len() == 2));
        assert!(matches!(out[4], Op::BatchAccumulateInstance { .. }));
        assert!(matches!(out[5], Op::BatchRoundFinalize { .. }));
    }

    #[test]
    fn multiple_windows_are_rewritten_independently() {
        let b0 = bidx(0);
        let b1 = bidx(1);
        let ops = vec![
            begin(b0, 0),
            reduce_op(0),
            accum(b0, iidx(0)),
            finalize(b0),
            Op::Preamble,
            begin(b1, 0),
            reduce_op(0),
            accum(b1, iidx(0)),
            finalize(b1),
        ];
        let out = fuse_reduce_windows(&ops);
        assert!(matches!(out[0], Op::BatchRoundBegin { .. }));
        assert!(matches!(out[1], Op::Reduce { .. }));
        assert!(matches!(out[2], Op::BatchAccumulateInstance { .. }));
        assert!(matches!(out[3], Op::BatchRoundFinalize { .. }));
        assert!(matches!(out[4], Op::Preamble));
        assert!(matches!(out[5], Op::BatchRoundBegin { .. }));
        assert!(matches!(out[6], Op::Reduce { .. }));
        assert!(matches!(out[7], Op::BatchAccumulateInstance { .. }));
        assert!(matches!(out[8], Op::BatchRoundFinalize { .. }));
    }
}
