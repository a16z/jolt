use jolt_backends::poly::{
    stage8_streaming_rlc_vector_matrix_product, Stage8StreamingRlcVectorMatrixProductInput,
};
#[cfg(feature = "field-inline")]
use jolt_claims::protocols::field_inline::FieldInlineCommittedPolynomial;
use jolt_claims::protocols::jolt::{
    formulas::{dimensions::TracePolynomialOrder, ra::JoltRaPolynomialLayout},
    JoltCommittedPolynomial,
};
use jolt_crypto::HomomorphicCommitment;
use jolt_field::Field;
use jolt_openings::{AdditivelyHomomorphic, CommitmentScheme, ZkOpeningScheme};
use jolt_poly::{
    eq_index_msb, EqPolynomial, MultilinearPoly, OneHotIndexOrder, OneHotPolynomial, Polynomial,
};
use jolt_transcript::Transcript;
use jolt_verifier::stages::{stage6::Stage6ClearOutput, stage7::outputs::Stage7ClearOutput};
#[cfg(feature = "field-inline")]
use jolt_witness::protocols::jolt_vm::field_inline::FieldInlineNamespace;
use jolt_witness::{
    protocols::jolt_vm::{JoltVmNamespace, JoltVmStage6Row, JoltVmStage6Rows},
    PolynomialChunk, WitnessProvider,
};

use super::input::Stage8ProverConfig;
use super::output::{Stage8OpeningStructure, Stage8ProofOutput, Stage8ZkProofOutput};
use super::request::{derive_stage8_structure_and_gamma, derive_stage8_zk_structure_and_gamma};
use crate::ProverError;

type Stage8ZkOutputFor<F, PCS> = Stage8ZkProofOutput<
    F,
    <PCS as CommitmentScheme>::Proof,
    <PCS as jolt_crypto::Commitment>::Output,
    <PCS as ZkOpeningScheme>::HidingCommitment,
    <PCS as ZkOpeningScheme>::Blind,
>;

#[cfg(feature = "frontier-harness")]
fn timed_stage8<T>(label: &'static str, f: impl FnOnce() -> T) -> T {
    let start = std::time::Instant::now();
    let result = f();
    crate::timing::record_stage_timing(label, start.elapsed().as_secs_f64() * 1000.0);
    result
}

#[cfg(not(feature = "frontier-harness"))]
fn timed_stage8<T>(_label: &'static str, f: impl FnOnce() -> T) -> T {
    f()
}

/// Canonical Stage 8 prover entrypoint (clear path): produce the
/// `joint_opening_proof` (the final batched PCS opening on `JoltProof`).
///
/// Mirrors `jolt-verifier/src/stages/stage8/verify.rs`: derive the final-opening
/// structure (opening order, scaled claims, RLC powers, joint claim), combine the
/// per-opening commitments and retained Stage 0 opening hints by the raw RLC
/// powers, run `PCS::open_poly` on a streaming backend-backed RLC source, and bind
/// the opening inputs. `commitments` and `hints` must be supplied in the
/// verifier's final-opening batch order (`RamInc`, `RdInc`, instruction RA,
/// bytecode RA, RAM RA, then present advice polynomials) — one entry per opening
/// id.
///
/// ZK final-opening assembly is not yet wired.
#[expect(
    clippy::too_many_arguments,
    reason = "Stage 8 needs prior-stage outputs plus the PCS commitments, retained hints, and setup, which are PCS-generic and cannot bundle into the non-generic prover input."
)]
pub fn prove_stage8<F, PCS, W, T>(
    config: &Stage8ProverConfig,
    stage6: &Stage6ClearOutput<F>,
    stage7: &Stage7ClearOutput<F>,
    witness: &W,
    #[cfg(feature = "field-inline")] field_inline_witness: &(impl WitnessProvider<F, FieldInlineNamespace>
          + Sync),
    commitments: &[PCS::Output],
    hints: Vec<PCS::OpeningHint>,
    setup: &PCS::ProverSetup,
    transcript: &mut T,
) -> Result<Stage8ProofOutput<F, PCS::Proof, PCS::Output>, ProverError>
where
    F: Field,
    <F as jolt_field::WithAccumulator>::Accumulator: jolt_field::RingAccumulator<Element = F>,
    PCS: CommitmentScheme<Field = F> + AdditivelyHomomorphic,
    PCS::Output: HomomorphicCommitment<F>,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage6Rows + Sync,
    T: Transcript<Challenge = F>,
{
    let (structure, gamma_powers) = timed_stage8("stage8.derive_structure", || {
        derive_stage8_structure_and_gamma(config, stage6, stage7, transcript)
    })?;
    if commitments.len() != structure.opening_ids.len()
        || hints.len() != structure.opening_ids.len()
    {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 8 batch size mismatch: {} commitments, {} hints, {} opening ids",
                commitments.len(),
                hints.len(),
                structure.opening_ids.len()
            ),
        });
    }

    let joint_commitment = timed_stage8("stage8.combine_commitments", || {
        PCS::combine(commitments, &gamma_powers)
    });
    let combined_hint = timed_stage8("stage8.combine_hints", || {
        PCS::combine_hints(hints, &gamma_powers)
    });
    let rlc_rows = timed_stage8("stage8.rows", || witness.stage6_rows())?;
    #[cfg(feature = "field-inline")]
    let field_rd_inc = Some(timed_stage8("stage8.field_rd_inc_rows", || {
        collect_field_rd_inc_rows(field_inline_witness, 1usize << config.log_t)
    })?);
    #[cfg(not(feature = "field-inline"))]
    let field_rd_inc = None;
    let joint_polynomial = Stage8JointRlcSource::new(
        config,
        witness,
        gamma_powers.clone(),
        rlc_rows,
        field_rd_inc,
    );
    let joint_opening_proof = timed_stage8("stage8.open_poly", || {
        PCS::open_poly(
            &joint_polynomial,
            structure.pcs_opening_point.as_slice(),
            structure.joint_claim,
            setup,
            Some(combined_hint),
            transcript,
        )
    });
    timed_stage8("stage8.bind_opening_inputs", || {
        PCS::bind_opening_inputs(
            transcript,
            structure.opening_point.as_slice(),
            &structure.joint_claim,
        );
    });

    Ok(Stage8ProofOutput {
        structure,
        joint_opening_proof,
        joint_commitment,
    })
}

#[expect(
    clippy::too_many_arguments,
    reason = "Stage 8 ZK opening uses the same verifier-order inputs as clear Stage 8 plus hidden PCS output material."
)]
pub fn prove_stage8_zk<F, PCS, W, T>(
    config: &Stage8ProverConfig,
    stage6: &Stage6ClearOutput<F>,
    stage7: &Stage7ClearOutput<F>,
    witness: &W,
    #[cfg(feature = "field-inline")] field_inline_witness: &(impl WitnessProvider<F, FieldInlineNamespace>
          + Sync),
    commitments: &[PCS::Output],
    hints: Vec<PCS::OpeningHint>,
    setup: &PCS::ProverSetup,
    transcript: &mut T,
) -> Result<Stage8ZkOutputFor<F, PCS>, ProverError>
where
    F: Field,
    <F as jolt_field::WithAccumulator>::Accumulator: jolt_field::RingAccumulator<Element = F>,
    PCS: CommitmentScheme<Field = F> + AdditivelyHomomorphic + ZkOpeningScheme,
    PCS::Output: HomomorphicCommitment<F>,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage6Rows + Sync,
    T: Transcript<Challenge = F>,
{
    let (structure, gamma_powers) = timed_stage8("stage8.derive_zk_structure", || {
        derive_stage8_zk_structure_and_gamma(config, stage6, stage7, transcript)
    })?;
    if commitments.len() != structure.opening_ids.len()
        || hints.len() != structure.opening_ids.len()
    {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 8 ZK batch size mismatch: {} commitments, {} hints, {} opening ids",
                commitments.len(),
                hints.len(),
                structure.opening_ids.len()
            ),
        });
    }

    let joint_commitment = timed_stage8("stage8.combine_commitments", || {
        PCS::combine(commitments, &gamma_powers)
    });
    let combined_hint = timed_stage8("stage8.combine_hints", || {
        PCS::combine_hints(hints, &gamma_powers)
    });
    let rlc_rows = timed_stage8("stage8.rows", || witness.stage6_rows())?;
    #[cfg(feature = "field-inline")]
    let field_rd_inc = Some(timed_stage8("stage8.field_rd_inc_rows", || {
        collect_field_rd_inc_rows(field_inline_witness, 1usize << config.log_t)
    })?);
    #[cfg(not(feature = "field-inline"))]
    let field_rd_inc = None;
    let joint_polynomial =
        Stage8JointRlcSource::new(config, witness, gamma_powers, rlc_rows, field_rd_inc);
    let (joint_opening_proof, hiding_evaluation_commitment, hiding_evaluation_blind) =
        timed_stage8("stage8.open_zk_poly", || {
            PCS::open_zk_poly(
                &joint_polynomial,
                structure.pcs_opening_point.as_slice(),
                structure.joint_claim,
                setup,
                combined_hint,
                transcript,
            )
        });
    timed_stage8("stage8.bind_zk_opening_inputs", || {
        PCS::bind_zk_opening_inputs(
            transcript,
            structure.opening_point.as_slice(),
            &hiding_evaluation_commitment,
        );
    });

    Ok(Stage8ZkProofOutput {
        structure,
        joint_opening_proof,
        joint_commitment,
        hiding_evaluation_commitment,
        hiding_evaluation_blind,
    })
}

struct Stage8JointRlcSource<'a, F: Field, W> {
    config: &'a Stage8ProverConfig,
    witness: &'a W,
    gamma_powers: Vec<F>,
    rows: Vec<JoltVmStage6Row>,
    field_rd_inc: Option<Vec<F>>,
}

impl<'a, F: Field, W> Stage8JointRlcSource<'a, F, W> {
    fn new(
        config: &'a Stage8ProverConfig,
        witness: &'a W,
        gamma_powers: Vec<F>,
        rows: Vec<JoltVmStage6Row>,
        field_rd_inc: Option<Vec<F>>,
    ) -> Self {
        Self {
            config,
            witness,
            gamma_powers,
            rows,
            field_rd_inc,
        }
    }
}

impl<F, W> MultilinearPoly<F> for Stage8JointRlcSource<'_, F, W>
where
    F: Field,
    <F as jolt_field::WithAccumulator>::Accumulator: jolt_field::RingAccumulator<Element = F>,
    W: WitnessProvider<F, JoltVmNamespace> + JoltVmStage6Rows + Sync,
{
    fn num_vars(&self) -> usize {
        self.config.committed_chunk_bits + self.config.log_t
    }

    fn evaluate(&self, point: &[F]) -> F {
        timed_stage8("stage8.rlc.evaluate", || {
            stage8_or_panic(evaluate_stage8_joint_polynomial_at_point(
                self.config,
                self.witness,
                &self.gamma_powers,
                self.field_rd_inc.as_deref(),
                point,
            ))
        })
    }

    fn for_each_row(&self, sigma: usize, f: &mut dyn FnMut(usize, &[F])) {
        timed_stage8("stage8.rlc.for_each_row_materialize", || {
            let evals = stage8_or_panic(build_stage8_joint_polynomial_evals_with_field(
                self.config,
                self.witness,
                &self.gamma_powers,
                self.field_rd_inc.as_deref(),
            ));
            let num_cols = 1usize << sigma;
            for (row, values) in evals.chunks(num_cols).enumerate() {
                f(row, values);
            }
        });
    }

    fn fold_rows(&self, left: &[F], sigma: usize) -> Vec<F> {
        timed_stage8("stage8.rlc.fold_rows", || {
            let num_cols = 1usize << sigma;
            let expected_rows = 1usize << (self.num_vars() - sigma);
            assert_eq!(
                left.len(),
                expected_rows,
                "Stage 8 RLC fold left-vector length mismatch"
            );

            let ra_start = 2 + usize::from(self.field_rd_inc.is_some());
            let (instruction_coefficients, rest) =
                self.gamma_powers[ra_start..].split_at(self.config.layout.instruction());
            let (bytecode_coefficients, rest) = rest.split_at(self.config.layout.bytecode());
            let (ram_coefficients, _) = rest.split_at(self.config.layout.ram());
            let mut result = stage8_streaming_rlc_vector_matrix_product(
                Stage8StreamingRlcVectorMatrixProductInput {
                    rows: &self.rows,
                    field_rd_inc: self.field_rd_inc.as_deref(),
                    log_t: self.config.log_t,
                    committed_chunk_bits: self.config.committed_chunk_bits,
                    trace_polynomial_order: self.config.trace_polynomial_order,
                    ram_inc_coefficient: self.gamma_powers[0],
                    rd_inc_coefficient: self.gamma_powers[1],
                    field_rd_inc_coefficient: self
                        .field_rd_inc
                        .as_ref()
                        .map(|_| self.gamma_powers[2]),
                    instruction_coefficients,
                    bytecode_coefficients,
                    ram_coefficients,
                    left_vec: left,
                    num_columns: num_cols,
                },
            );
            let mut batch_index = ra_start + self.config.layout.total();
            for (polynomial, layout) in [
                (
                    JoltCommittedPolynomial::TrustedAdvice,
                    self.config.trusted_advice_layout.as_ref(),
                ),
                (
                    JoltCommittedPolynomial::UntrustedAdvice,
                    self.config.untrusted_advice_layout.as_ref(),
                ),
            ] {
                let Some(layout) = layout else {
                    continue;
                };
                let coefficient = self.gamma_powers[batch_index];
                batch_index += 1;
                stage8_or_panic(fold_stage8_advice_rows(
                    self.witness,
                    polynomial,
                    layout,
                    coefficient,
                    left,
                    &mut result,
                ));
            }
            result
        })
    }
}

#[expect(
    clippy::panic,
    reason = "MultilinearPoly::fold_rows cannot return Result; prove_stage8 validates construction before Dory calls this adapter."
)]
fn stage8_or_panic<T>(result: Result<T, impl std::fmt::Display>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => panic!("Stage 8 streaming RLC source failed: {error}"),
    }
}

/// Build the committed RA-family one-hot polynomials in the final-opening batch
/// order (instruction, then bytecode, then RAM), using the verifier-declared
/// trace polynomial order. The per-cycle chunk indices are sourced from the
/// generic `WitnessProvider::committed_stream` (the chunk selector is already
/// applied by the witness).
///
/// These are the RA constituents of the joint RLC polynomial that Stage 8's
/// `joint_opening_proof` opens. Evaluating each at the PCS opening point yields
/// the reduced RA opening claim, which `build_stage8_ra_constituents` callers can
/// check against the Stage 7 hamming-weight reduced claims to validate the
/// constituent layout before the Dory opening.
pub fn build_stage8_ra_constituents<F, W>(
    layout: JoltRaPolynomialLayout,
    committed_chunk_bits: usize,
    log_t: usize,
    trace_polynomial_order: TracePolynomialOrder,
    witness: &W,
) -> Result<Vec<OneHotPolynomial>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let k = 1usize << committed_chunk_bits;
    let rows = 1usize << log_t;
    let mut constituents = Vec::with_capacity(layout.total());
    for polynomial in (0..layout.instruction())
        .map(JoltCommittedPolynomial::InstructionRa)
        .chain((0..layout.bytecode()).map(JoltCommittedPolynomial::BytecodeRa))
        .chain((0..layout.ram()).map(JoltCommittedPolynomial::RamRa))
    {
        let indices = collect_one_hot_ra_indices::<F, W>(polynomial, rows, witness)?;
        constituents.push(OneHotPolynomial::new_with_index_order(
            k,
            indices,
            stage8_one_hot_index_order(trace_polynomial_order),
        ));
    }
    Ok(constituents)
}

/// Evaluate the Stage 8 RA-family one-hot constituents at the PCS opening point.
///
/// Each reduced RA opening claim (scale = 1) equals the committed one-hot
/// polynomial evaluated at the PCS opening point, so this is used to validate the
/// committed layout against the Stage 7 hamming-weight reduced claims.
pub fn evaluate_stage8_ra_constituents<F, W>(
    layout: JoltRaPolynomialLayout,
    committed_chunk_bits: usize,
    log_t: usize,
    trace_polynomial_order: TracePolynomialOrder,
    witness: &W,
    point: &[F],
) -> Result<Vec<F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let constituents = build_stage8_ra_constituents(
        layout,
        committed_chunk_bits,
        log_t,
        trace_polynomial_order,
        witness,
    )?;
    Ok(constituents
        .iter()
        .map(|constituent| MultilinearPoly::evaluate(constituent, point))
        .collect())
}

/// Build a dense increment (`RamInc`/`RdInc`) constituent embedded into the full
/// committed space (`committed_chunk_bits + log_t` variables).
///
/// Stage 0 commits the dense increments zero-padded at address zero in the
/// verifier-declared trace polynomial order. Evaluating the result at the PCS
/// opening point therefore yields
/// `inc(r_cycle) · eq(r_address, 0)` = `inc_claim · dense_embedding_scale`.
pub fn build_stage8_dense_constituent<F, W>(
    polynomial: JoltCommittedPolynomial,
    committed_chunk_bits: usize,
    log_t: usize,
    trace_polynomial_order: TracePolynomialOrder,
    witness: &W,
) -> Result<Polynomial<F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let rows = 1usize << log_t;
    let addresses = 1usize << committed_chunk_bits;
    let full = 1usize << (committed_chunk_bits + log_t);
    let mut stream = witness.committed_stream(polynomial, rows.max(1))?;
    let mut evals = vec![F::zero(); full];
    let mut index = 0usize;
    while let Some(chunk) = stream.next_chunk()? {
        match chunk {
            PolynomialChunk::I128(values) => {
                for value in values {
                    if index >= rows {
                        return Err(too_many_dense_rows(polynomial, rows));
                    }
                    let flat =
                        stage8_trace_flat_index(trace_polynomial_order, 0, index, addresses, rows);
                    evals[flat] = F::from_i128(value);
                    index += 1;
                }
            }
            PolynomialChunk::Dense(values) => {
                for value in values {
                    if index >= rows {
                        return Err(too_many_dense_rows(polynomial, rows));
                    }
                    let flat =
                        stage8_trace_flat_index(trace_polynomial_order, 0, index, addresses, rows);
                    evals[flat] = value;
                    index += 1;
                }
            }
            PolynomialChunk::Zeros(count) => {
                index = index
                    .checked_add(count)
                    .ok_or_else(|| too_many_dense_rows(polynomial, rows))?;
                if index > rows {
                    return Err(too_many_dense_rows(polynomial, rows));
                }
            }
            _ => {
                return Err(ProverError::InvalidStageRequest {
                    reason: format!("Stage 8 expected a dense increment stream for {polynomial:?}"),
                });
            }
        }
    }
    if index != rows {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 8 dense stream for {polynomial:?} produced {index} rows, expected {rows}"
            ),
        });
    }
    Ok(Polynomial::from(evals))
}

fn too_many_dense_rows(polynomial: JoltCommittedPolynomial, rows: usize) -> ProverError {
    ProverError::InvalidStageRequest {
        reason: format!("Stage 8 dense stream for {polynomial:?} exceeded {rows} rows"),
    }
}

/// Evaluate the dense increment constituents (`RamInc`, then `RdInc`) at the PCS
/// opening point — used to validate the top-block embedding against the Stage 6
/// increment reduced claims scaled by `dense_embedding_scale`.
pub fn evaluate_stage8_dense_constituents<F, W>(
    committed_chunk_bits: usize,
    log_t: usize,
    trace_polynomial_order: TracePolynomialOrder,
    witness: &W,
    point: &[F],
) -> Result<Vec<F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    [
        JoltCommittedPolynomial::RamInc,
        JoltCommittedPolynomial::RdInc,
    ]
    .into_iter()
    .map(|polynomial| {
        let constituent = build_stage8_dense_constituent::<F, W>(
            polynomial,
            committed_chunk_bits,
            log_t,
            trace_polynomial_order,
            witness,
        )?;
        Ok(MultilinearPoly::evaluate(&constituent, point))
    })
    .collect()
}

/// Materialize the dense joint RLC polynomial `Σ_i γ^i · constituent_i` that the
/// Stage 8 `joint_opening_proof` opens, alongside the derived opening structure.
///
/// The constituents are summed into one dense `2^(committed_chunk_bits+log_t)`
/// evaluation vector matching the Dory commitment matrix: dense increments at
/// address zero and RA polynomials in the verifier-declared trace polynomial
/// order, each weighted by its raw `gamma_powers` coefficient (the scaling
/// factors are encoded in the layout, not the weights).
pub fn materialize_stage8_joint_polynomial<F, W, T>(
    config: &Stage8ProverConfig,
    stage6: &Stage6ClearOutput<F>,
    stage7: &Stage7ClearOutput<F>,
    witness: &W,
    transcript: &mut T,
) -> Result<(Stage8OpeningStructure<F>, Polynomial<F>), ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
    T: Transcript<Challenge = F>,
{
    let (structure, gamma_powers) =
        derive_stage8_structure_and_gamma(config, stage6, stage7, transcript)?;
    let evals = build_stage8_joint_polynomial_evals(config, witness, &gamma_powers)?;
    Ok((structure, Polynomial::from(evals)))
}

/// Materialize the dense evaluation vector of the joint RLC polynomial
/// `Σ_i γ^i · constituent_i` given the RLC `gamma_powers` (one dense
/// `2^(committed_chunk_bits+log_t)` vector matching the Dory commitment matrix).
pub fn build_stage8_joint_polynomial_evals<F, W>(
    config: &Stage8ProverConfig,
    witness: &W,
    gamma_powers: &[F],
) -> Result<Vec<F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    build_stage8_joint_polynomial_evals_with_field(config, witness, gamma_powers, None)
}

fn build_stage8_joint_polynomial_evals_with_field<F, W>(
    config: &Stage8ProverConfig,
    witness: &W,
    gamma_powers: &[F],
    field_rd_inc: Option<&[F]>,
) -> Result<Vec<F>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let num_rows = 1usize << config.log_t;
    let full = 1usize << (config.committed_chunk_bits + config.log_t);
    let mut joint = vec![F::zero(); full];

    // `RamInc` (batch index 0) and `RdInc` (batch index 1): top-block embedding.
    for (batch_index, polynomial) in [
        JoltCommittedPolynomial::RamInc,
        JoltCommittedPolynomial::RdInc,
    ]
    .into_iter()
    .enumerate()
    {
        let constituent = build_stage8_dense_constituent::<F, W>(
            polynomial,
            config.committed_chunk_bits,
            config.log_t,
            config.trace_polynomial_order,
            witness,
        )?;
        let coefficient = gamma_powers[batch_index];
        for (flat, value) in constituent.evaluations().iter().enumerate() {
            joint[flat] += coefficient * *value;
        }
    }

    let mut batch_index = 2;
    if let Some(field_rd_inc) = field_rd_inc {
        if field_rd_inc.len() != num_rows {
            return Err(ProverError::InvalidStageRequest {
                reason: format!(
                    "Stage 8 field-inline rd_inc stream produced {} rows, expected {num_rows}",
                    field_rd_inc.len()
                ),
            });
        }
        let coefficient = gamma_powers[batch_index];
        batch_index += 1;
        let addresses = 1usize << config.committed_chunk_bits;
        for (cycle, value) in field_rd_inc.iter().enumerate() {
            let flat = stage8_trace_flat_index(
                config.trace_polynomial_order,
                0,
                cycle,
                addresses,
                num_rows,
            );
            joint[flat] += coefficient * *value;
        }
    }

    // RA constituents follow in verifier-declared trace polynomial order.
    for polynomial in (0..config.layout.instruction())
        .map(JoltCommittedPolynomial::InstructionRa)
        .chain((0..config.layout.bytecode()).map(JoltCommittedPolynomial::BytecodeRa))
        .chain((0..config.layout.ram()).map(JoltCommittedPolynomial::RamRa))
    {
        let indices = collect_one_hot_ra_indices::<F, W>(polynomial, num_rows, witness)?;
        let coefficient = gamma_powers[batch_index];
        batch_index += 1;
        let addresses = 1usize << config.committed_chunk_bits;
        for (cycle, opt_col) in indices.into_iter().enumerate() {
            if let Some(col) = opt_col {
                let flat = stage8_trace_flat_index(
                    config.trace_polynomial_order,
                    usize::from(col),
                    cycle,
                    addresses,
                    num_rows,
                );
                joint[flat] += coefficient;
            }
        }
    }

    let mut batch_index = 2 + config.layout.total();
    for (polynomial, layout) in [
        (
            JoltCommittedPolynomial::TrustedAdvice,
            config.trusted_advice_layout.as_ref(),
        ),
        (
            JoltCommittedPolynomial::UntrustedAdvice,
            config.untrusted_advice_layout.as_ref(),
        ),
    ] {
        let Some(layout) = layout else {
            continue;
        };
        add_stage8_advice_to_joint::<F, W>(
            polynomial,
            layout,
            witness,
            gamma_powers[batch_index],
            &mut joint,
        )?;
        batch_index += 1;
    }

    Ok(joint)
}

fn evaluate_stage8_joint_polynomial_at_point<F, W>(
    config: &Stage8ProverConfig,
    witness: &W,
    gamma_powers: &[F],
    field_rd_inc: Option<&[F]>,
    point: &[F],
) -> Result<F, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let expected_vars = config
        .committed_chunk_bits
        .checked_add(config.log_t)
        .ok_or_else(|| ProverError::InvalidStageRequest {
            reason: "Stage 8 joint evaluation point length overflow".to_owned(),
        })?;
    if point.len() != expected_vars {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 8 joint evaluation point has {} variables, expected {expected_vars}",
                point.len()
            ),
        });
    }

    let num_rows = 1usize << config.log_t;
    let trace_eq = Stage8TraceEqTables::new(config, point);
    let mut result = F::zero();

    for (batch_index, polynomial) in [
        JoltCommittedPolynomial::RamInc,
        JoltCommittedPolynomial::RdInc,
    ]
    .into_iter()
    .enumerate()
    {
        add_stage8_dense_stream_evaluation(
            polynomial,
            config,
            witness,
            gamma_powers[batch_index],
            &trace_eq,
            &mut result,
        )?;
    }

    let mut batch_index = 2;
    if let Some(field_rd_inc) = field_rd_inc {
        if field_rd_inc.len() != num_rows {
            return Err(ProverError::InvalidStageRequest {
                reason: format!(
                    "Stage 8 field-inline rd_inc stream produced {} rows, expected {num_rows}",
                    field_rd_inc.len()
                ),
            });
        }
        let coefficient = gamma_powers[batch_index];
        batch_index += 1;
        for (cycle, value) in field_rd_inc.iter().copied().enumerate() {
            if value.is_zero() {
                continue;
            }
            result += coefficient * value * trace_eq.weight(0, cycle);
        }
    }

    for polynomial in (0..config.layout.instruction())
        .map(JoltCommittedPolynomial::InstructionRa)
        .chain((0..config.layout.bytecode()).map(JoltCommittedPolynomial::BytecodeRa))
        .chain((0..config.layout.ram()).map(JoltCommittedPolynomial::RamRa))
    {
        let indices = collect_one_hot_ra_indices::<F, W>(polynomial, num_rows, witness)?;
        let coefficient = gamma_powers[batch_index];
        batch_index += 1;
        for (cycle, opt_col) in indices.into_iter().enumerate() {
            let Some(col) = opt_col else {
                continue;
            };
            result += coefficient * trace_eq.weight(usize::from(col), cycle);
        }
    }

    let mut advice_batch_index = 2 + usize::from(field_rd_inc.is_some()) + config.layout.total();
    for (polynomial, layout) in [
        (
            JoltCommittedPolynomial::TrustedAdvice,
            config.trusted_advice_layout.as_ref(),
        ),
        (
            JoltCommittedPolynomial::UntrustedAdvice,
            config.untrusted_advice_layout.as_ref(),
        ),
    ] {
        let Some(layout) = layout else {
            continue;
        };
        add_stage8_advice_evaluation(
            polynomial,
            layout,
            witness,
            gamma_powers[advice_batch_index],
            point,
            &mut result,
        )?;
        advice_batch_index += 1;
    }

    Ok(result)
}

struct Stage8TraceEqTables<F> {
    address: Vec<F>,
    cycle: Vec<F>,
}

impl<F: Field> Stage8TraceEqTables<F> {
    fn new(config: &Stage8ProverConfig, point: &[F]) -> Self {
        let (address_point, cycle_point) = match config.trace_polynomial_order {
            TracePolynomialOrder::CycleMajor => {
                let (address, cycle) = point.split_at(config.committed_chunk_bits);
                (address, cycle)
            }
            TracePolynomialOrder::AddressMajor => {
                let (cycle, address) = point.split_at(config.log_t);
                (address, cycle)
            }
        };
        Self {
            address: EqPolynomial::<F>::evals(address_point, None),
            cycle: EqPolynomial::<F>::evals(cycle_point, None),
        }
    }

    #[inline]
    fn weight(&self, address: usize, cycle: usize) -> F {
        self.address[address] * self.cycle[cycle]
    }
}

fn add_stage8_dense_stream_evaluation<F, W>(
    polynomial: JoltCommittedPolynomial,
    config: &Stage8ProverConfig,
    witness: &W,
    coefficient: F,
    trace_eq: &Stage8TraceEqTables<F>,
    result: &mut F,
) -> Result<(), ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let rows = 1usize << config.log_t;
    let mut stream = witness.committed_stream(polynomial, rows.max(1))?;
    let mut index = 0usize;
    while let Some(chunk) = stream.next_chunk()? {
        match chunk {
            PolynomialChunk::I128(values) => {
                for value in values {
                    if index >= rows {
                        return Err(too_many_dense_rows(polynomial, rows));
                    }
                    *result += coefficient * F::from_i128(value) * trace_eq.weight(0, index);
                    index += 1;
                }
            }
            PolynomialChunk::Dense(values) => {
                for value in values {
                    if index >= rows {
                        return Err(too_many_dense_rows(polynomial, rows));
                    }
                    *result += coefficient * value * trace_eq.weight(0, index);
                    index += 1;
                }
            }
            PolynomialChunk::Zeros(count) => {
                index = index
                    .checked_add(count)
                    .ok_or_else(|| too_many_dense_rows(polynomial, rows))?;
                if index > rows {
                    return Err(too_many_dense_rows(polynomial, rows));
                }
            }
            _ => {
                return Err(ProverError::InvalidStageRequest {
                    reason: format!("Stage 8 expected a dense increment stream for {polynomial:?}"),
                });
            }
        }
    }
    if index != rows {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 8 dense stream for {polynomial:?} produced {index} rows, expected {rows}"
            ),
        });
    }
    Ok(())
}

fn add_stage8_advice_to_joint<F, W>(
    polynomial: JoltCommittedPolynomial,
    layout: &jolt_claims::protocols::jolt::AdviceClaimReductionLayout,
    witness: &W,
    coefficient: F,
    joint: &mut [F],
) -> Result<(), ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let num_vars = layout.main_shape().total_vars();
    let num_cols = 1usize << num_vars.div_ceil(2);
    let advice_cols = 1usize << layout.advice_shape().column_vars();
    let advice_rows = 1usize << layout.advice_shape().row_vars();
    let mut stream = witness.committed_stream(polynomial, 1024)?;
    let mut index = 0usize;
    while let Some(chunk) = stream.next_chunk()? {
        let PolynomialChunk::U64(values) = chunk else {
            return Err(ProverError::InvalidStageRequest {
                reason: format!("Stage 8 expected U64 advice stream for {polynomial:?}"),
            });
        };
        for value in values {
            let row = index / advice_cols;
            let col = index % advice_cols;
            let flat = row * num_cols + col;
            joint[flat] += coefficient * F::from_u64(value);
            index += 1;
        }
    }
    let expected = advice_rows * advice_cols;
    if index != expected {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 8 advice stream for {polynomial:?} produced {index} rows, expected {expected}"
            ),
        });
    }
    Ok(())
}

fn add_stage8_advice_evaluation<F, W>(
    polynomial: JoltCommittedPolynomial,
    layout: &jolt_claims::protocols::jolt::AdviceClaimReductionLayout,
    witness: &W,
    coefficient: F,
    point: &[F],
    result: &mut F,
) -> Result<(), ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let num_vars = layout.main_shape().total_vars();
    if point.len() != num_vars {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 8 advice evaluation point has {} variables, expected {num_vars}",
                point.len()
            ),
        });
    }
    let num_cols = 1usize << num_vars.div_ceil(2);
    let advice_cols = 1usize << layout.advice_shape().column_vars();
    let advice_rows = 1usize << layout.advice_shape().row_vars();
    let mut stream = witness.committed_stream(polynomial, 1024)?;
    let mut index = 0usize;
    while let Some(chunk) = stream.next_chunk()? {
        let PolynomialChunk::U64(values) = chunk else {
            return Err(ProverError::InvalidStageRequest {
                reason: format!("Stage 8 expected U64 advice stream for {polynomial:?}"),
            });
        };
        for value in values {
            let row = index / advice_cols;
            let col = index % advice_cols;
            let flat = row * num_cols + col;
            if value != 0 {
                *result += coefficient * F::from_u64(value) * eq_index_msb(point, flat);
            }
            index += 1;
        }
    }
    let expected = advice_rows * advice_cols;
    if index != expected {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 8 advice stream for {polynomial:?} produced {index} rows, expected {expected}"
            ),
        });
    }
    Ok(())
}

fn fold_stage8_advice_rows<F, W>(
    witness: &W,
    polynomial: JoltCommittedPolynomial,
    layout: &jolt_claims::protocols::jolt::AdviceClaimReductionLayout,
    coefficient: F,
    left: &[F],
    result: &mut [F],
) -> Result<(), ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    if coefficient.is_zero() {
        return Ok(());
    }
    let advice_cols = 1usize << layout.advice_shape().column_vars();
    let advice_rows = 1usize << layout.advice_shape().row_vars();
    if advice_cols > result.len() || advice_rows > left.len() {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 8 advice block for {polynomial:?} does not fit the main Dory matrix"
            ),
        });
    }
    let mut stream = witness.committed_stream(polynomial, 1024)?;
    let mut index = 0usize;
    while let Some(chunk) = stream.next_chunk()? {
        let PolynomialChunk::U64(values) = chunk else {
            return Err(ProverError::InvalidStageRequest {
                reason: format!("Stage 8 expected U64 advice stream for {polynomial:?}"),
            });
        };
        for value in values {
            let row = index / advice_cols;
            let col = index % advice_cols;
            result[col] += left[row] * coefficient * F::from_u64(value);
            index += 1;
        }
    }
    let expected = advice_rows * advice_cols;
    if index != expected {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 8 advice stream for {polynomial:?} produced {index} rows, expected {expected}"
            ),
        });
    }
    Ok(())
}

/// Materialize the joint polynomial and evaluate it at the PCS opening point,
/// returning `(joint_polynomial_evaluation, joint_claim)`. These must be equal —
/// it validates that the materialized joint RLC polynomial (the input to the Dory
/// opening) opens to the transcript-derived joint claim.
pub fn evaluate_stage8_joint_polynomial<F, W, T>(
    config: &Stage8ProverConfig,
    stage6: &Stage6ClearOutput<F>,
    stage7: &Stage7ClearOutput<F>,
    witness: &W,
    transcript: &mut T,
) -> Result<(F, F), ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
    T: Transcript<Challenge = F>,
{
    let (structure, joint) =
        materialize_stage8_joint_polynomial(config, stage6, stage7, witness, transcript)?;
    let evaluation = MultilinearPoly::evaluate(&joint, structure.pcs_opening_point.as_slice());
    Ok((evaluation, structure.joint_claim))
}

#[cfg(feature = "field-inline")]
fn collect_field_rd_inc_rows<F, FI>(witness: &FI, rows: usize) -> Result<Vec<F>, ProverError>
where
    F: Field,
    FI: WitnessProvider<F, FieldInlineNamespace> + ?Sized,
{
    let mut stream = witness.committed_stream(FieldInlineCommittedPolynomial::FieldRdInc, 1024)?;
    let mut values = Vec::with_capacity(rows);
    while let Some(chunk) = stream.next_chunk()? {
        match chunk {
            PolynomialChunk::Dense(chunk) => values.extend(chunk),
            PolynomialChunk::Zeros(count) => {
                values.extend(std::iter::repeat_n(F::zero(), count));
            }
            _ => {
                return Err(ProverError::InvalidStageRequest {
                    reason: "Stage 8 expected dense field-inline rd_inc stream".to_owned(),
                });
            }
        }
    }
    if values.len() != rows {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 8 field-inline rd_inc stream produced {} rows, expected {rows}",
                values.len()
            ),
        });
    }
    Ok(values)
}

#[inline]
const fn stage8_one_hot_index_order(
    trace_polynomial_order: TracePolynomialOrder,
) -> OneHotIndexOrder {
    match trace_polynomial_order {
        TracePolynomialOrder::CycleMajor => OneHotIndexOrder::ColumnMajor,
        TracePolynomialOrder::AddressMajor => OneHotIndexOrder::RowMajor,
    }
}

#[inline]
const fn stage8_trace_flat_index(
    trace_polynomial_order: TracePolynomialOrder,
    address: usize,
    cycle: usize,
    num_addresses: usize,
    num_cycles: usize,
) -> usize {
    trace_polynomial_order.address_cycle_to_index(address, cycle, num_addresses, num_cycles)
}

fn collect_one_hot_ra_indices<F, W>(
    polynomial: JoltCommittedPolynomial,
    rows: usize,
    witness: &W,
) -> Result<Vec<Option<u8>>, ProverError>
where
    F: Field,
    W: WitnessProvider<F, JoltVmNamespace>,
{
    let mut stream = witness.committed_stream(polynomial, rows.max(1))?;
    let mut indices = Vec::with_capacity(rows);
    while let Some(chunk) = stream.next_chunk()? {
        let PolynomialChunk::OneHot(values) = chunk else {
            return Err(ProverError::InvalidStageRequest {
                reason: format!("Stage 8 expected a one-hot stream for {polynomial:?}"),
            });
        };
        for value in values {
            let index = match value {
                Some(index) => {
                    Some(
                        u8::try_from(index).map_err(|_| ProverError::InvalidStageRequest {
                            reason: format!(
                            "Stage 8 RA chunk index {index} for {polynomial:?} exceeds u8 range"
                        ),
                        })?,
                    )
                }
                None => None,
            };
            indices.push(index);
        }
    }
    if indices.len() != rows {
        return Err(ProverError::InvalidStageRequest {
            reason: format!(
                "Stage 8 RA stream for {polynomial:?} produced {} rows, expected {rows}",
                indices.len()
            ),
        });
    }
    Ok(indices)
}
