//! Prover runtime: execute a linked schedule to produce a proof.
//!
//! The [`execute`] function walks the [`Op`] sequence of an [`Executable`],
//! dispatching compute operations to the backend and transcript operations
//! to the Fiat-Shamir transcript. This is the entire prover — a flat
//! interpreter over the compiler's output.
//!
//! # Architecture
//!
//! ```text
//! Protocol → compile() → Module → link(backend) → Executable<B,F>
//!                                                       │
//!                                          execute(exe, provider, backend, transcript)
//!                                                       │
//!                                                       ▼
//!                                                 ExecutionContext
//! ```

use jolt_compiler::module::{InputBinding, Op};
use jolt_compiler::KernelDef;
use jolt_compute::{BufferProvider, ComputeBackend, EqInput, Executable};
use jolt_field::Field;
use jolt_transcript::Transcript;

/// Mutable state accumulated during proof execution.
///
/// Holds device buffers, challenge values, and scalar evaluations.
/// The runtime populates these as it walks the op schedule.
pub struct ExecutionContext<B: ComputeBackend, F: Field> {
    /// Polynomial buffers indexed by poly index from the module.
    /// `None` for polys not yet loaded or already released.
    pub buffers: Vec<Option<B::Buffer<F>>>,
    /// Challenge values indexed by challenge index from the module.
    pub challenges: Vec<F>,
    /// Polynomial evaluations (fully-bound scalar), indexed by poly index.
    pub evaluations: Vec<Option<F>>,
    /// Round polynomial coefficients from the most recent `SumcheckRound`.
    last_round_coeffs: Vec<F>,
}

/// Execute a linked schedule against a compute backend and transcript.
///
/// The `provider` supplies witness and preprocessed polynomial buffers on
/// demand (via [`BufferProvider::load`]). Table-type inputs (eq, lt) are
/// constructed on-device from challenge values — no large host→device
/// transfers for those.
///
/// Returns the final execution context containing all evaluations and challenges.
pub fn execute<B, F, T, P>(
    executable: &Executable<B, F>,
    provider: &mut P,
    backend: &B,
    transcript: &mut T,
) -> ExecutionContext<B, F>
where
    B: ComputeBackend,
    F: Field,
    T: Transcript<Challenge = F>,
    P: BufferProvider<B, F>,
{
    let module = &executable.module;

    let mut ctx = ExecutionContext {
        buffers: (0..module.polys.len()).map(|_| None).collect(),
        challenges: vec![F::zero(); module.challenges.len()],
        evaluations: vec![None; module.polys.len()],
        last_round_coeffs: Vec::new(),
    };

    for op in &executable.ops {
        match op {
            Op::SumcheckRound {
                kernel,
                round: _,
                bind_challenge,
            } => {
                execute_sumcheck_round(
                    &mut ctx,
                    executable,
                    *kernel,
                    *bind_challenge,
                    provider,
                    backend,
                );
            }

            Op::Evaluate { poly } => {
                if let Some(buf) = &ctx.buffers[*poly] {
                    let data = backend.download(buf);
                    if !data.is_empty() {
                        ctx.evaluations[*poly] = Some(data[0]);
                    }
                }
            }

            Op::FinalBind {
                polys,
                challenge,
                order,
            } => {
                let scalar = ctx.challenges[*challenge];
                for &pi in polys {
                    if let Some(buf) = &mut ctx.buffers[pi] {
                        backend.interpolate_pairs_inplace(buf, scalar, *order);
                    }
                }
            }

            Op::Release { poly } => {
                ctx.buffers[*poly] = None;
            }

            Op::EmitCommitments { .. } => {
                // Commitments are handled externally by the PCS layer.
            }

            Op::EmitRoundPoly { num_coeffs, .. } => {
                let coeffs = &ctx.last_round_coeffs[..*num_coeffs];
                for c in coeffs {
                    transcript.append(c);
                }
            }

            Op::EmitScalars { evals } => {
                for &pi in evals {
                    if let Some(val) = &ctx.evaluations[pi] {
                        transcript.append(val);
                    }
                }
            }

            Op::Squeeze { challenge } => {
                ctx.challenges[*challenge] = transcript.challenge();
            }
        }
    }

    ctx
}

/// Ensure all inputs for a kernel are resolved (loaded or built on-device).
fn resolve_inputs<B, F, P>(
    ctx: &mut ExecutionContext<B, F>,
    kdef: &KernelDef,
    provider: &mut P,
    backend: &B,
) where
    B: ComputeBackend,
    F: Field,
    P: BufferProvider<B, F>,
{
    for binding in &kdef.inputs {
        let pi = binding.poly();
        if ctx.buffers[pi].is_some() {
            continue;
        }
        let buf = match binding {
            InputBinding::Provided { .. } => provider.load(pi, backend),
            InputBinding::EqTable { challenges, .. } => {
                let point: Vec<F> = challenges.iter().map(|&ci| ctx.challenges[ci]).collect();
                backend.eq_table(&point)
            }
            InputBinding::EqPlusOneTable { challenges, .. } => {
                let point: Vec<F> = challenges.iter().map(|&ci| ctx.challenges[ci]).collect();
                let (_eq, eq_plus_one) = backend.eq_plus_one_table(&point);
                eq_plus_one
            }
            InputBinding::LtTable { challenges, .. } => {
                let point: Vec<F> = challenges.iter().map(|&ci| ctx.challenges[ci]).collect();
                backend.lt_table(&point)
            }
        };
        ctx.buffers[pi] = Some(buf);
    }
}

fn execute_sumcheck_round<B, F, P>(
    ctx: &mut ExecutionContext<B, F>,
    executable: &Executable<B, F>,
    kernel_idx: usize,
    bind_challenge: Option<usize>,
    provider: &mut P,
    backend: &B,
) where
    B: ComputeBackend,
    F: Field,
    P: BufferProvider<B, F>,
{
    let kdef = &executable.module.prover.kernels[kernel_idx];
    let compiled_kernel = &executable.kernels[kernel_idx];
    let num_evals = kdef.degree + 1;

    resolve_inputs(ctx, kdef, provider, backend);

    // Rounds 1+: bind all input buffers at the previous round's challenge.
    if let Some(ch) = bind_challenge {
        let scalar = ctx.challenges[ch];
        for binding in &kdef.inputs {
            if let Some(buf) = &mut ctx.buffers[binding.poly()] {
                backend.interpolate_pairs_inplace(buf, scalar, kdef.binding_order);
            }
        }
    }

    // Reduce: evaluate the composition formula over paired elements.
    let input_refs: Vec<&B::Buffer<F>> = kdef
        .inputs
        .iter()
        .filter_map(|b| ctx.buffers[b.poly()].as_ref())
        .collect();

    ctx.last_round_coeffs = backend.pairwise_reduce(
        &input_refs,
        EqInput::Unit,
        compiled_kernel,
        &ctx.challenges,
        num_evals,
        kdef.binding_order,
    );
}
