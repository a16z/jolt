//! Stage 1: `tau`, the Spartan outer uni-skip round and the outer remainder
//! sumcheck, whose expected output is the factored
//! `TauKernel · (Σ AzWeight_i·o_i + AzConstant) · (Σ BzWeight_i·o_i + BzConstant)`
//! with the 19 row weights as wires and the constraint-matrix entries as
//! constants.

use jolt_claims::protocols::jolt::geometry::spartan::{
    outer_opening, outer_uniskip_opening, SpartanOuterDimensions, SPARTAN_OUTER_R1CS_INPUTS,
};
use jolt_claims::protocols::jolt::relations::spartan::{OuterRemainder, OuterUniskip};
use jolt_claims::protocols::jolt::SpartanOuterPublic;
use jolt_claims::SymbolicSumcheck;
use jolt_field::{Fr, One};
use jolt_r1cs::constraint::SparseRow;
use jolt_r1cs::constraints::jolt::{
    spartan_outer_constraints, spartan_outer_opening_columns, SPARTAN_OUTER_FIRST_GROUP_ROWS,
    SPARTAN_OUTER_ROW_COUNT, SPARTAN_OUTER_SECOND_GROUP_ROWS, SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE,
};
use jolt_r1cs::constraints::rv64;

use super::ctx::{Accum, Ctx, Lc};
use super::gadgets::{centered_lagrange, centered_lagrange_kernel, eq, one_minus, reversed};
use super::lower::lower;
use super::replay::SqueezeKind;
use super::sumcheck::{finish_batch, uniskip, zero};
use super::wiring::{absorb_member, run_batch, Layout, Wires};
use super::RelationError;
use crate::profile::WrapperProfile;

pub(crate) struct Stage1 {
    /// The raw remainder sumcheck point (`1 + log_t` coordinates).
    pub point: Vec<Lc>,
}

impl Stage1 {
    /// The cycle coordinates of the remainder point (everything after the
    /// stream coordinate).
    pub(crate) fn cycle_binding(&self) -> &[Lc] {
        &self.point[1..]
    }
}

pub(crate) fn walk(
    ctx: &mut Ctx,
    profile: &WrapperProfile,
    wires: &mut Wires,
) -> Result<Stage1, RelationError> {
    let log_t = profile.log_t;
    let dimensions = SpartanOuterDimensions::rv64(log_t);
    let include_affine_terms = dimensions.include_affine_terms();

    ctx.section("stage1/tau");
    let tau = ctx.squeeze_vector(SqueezeKind::Challenge, log_t + 2)?;

    ctx.section("stage1/uniskip");
    let uniskip_relation = OuterUniskip::new(dimensions.clone());
    let (uniskip_challenge, uniskip_output) = uniskip(
        ctx,
        &zero(),
        uniskip_relation.degree(),
        SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE,
    )?;
    wires.set(outer_uniskip_opening(), uniskip_output, Vec::new());

    ctx.section("stage1/remainder");
    let remainder = OuterRemainder::new(dimensions);
    let input = lower(ctx, &remainder.input_expression::<Fr>(), &wires.sources)?;
    let layout = Layout::uniform(remainder.rounds(), remainder.degree(), 0);
    let (batch, point, final_claim) = run_batch(ctx, &[input], &[layout])?;
    let opening_point = reversed(&point);
    absorb_member(ctx, wires, &remainder, &[], &[], |_| opening_point.clone())?;

    ctx.section("stage1/expected");
    let stream = &point[0];
    let lagrange = centered_lagrange(ctx, SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE, &uniskip_challenge);
    let mut row_weights = vec![Lc::zero(); SPARTAN_OUTER_ROW_COUNT];
    let not_stream = one_minus(stream);
    for (&row, weight) in SPARTAN_OUTER_FIRST_GROUP_ROWS.iter().zip(&lagrange) {
        let term = ctx.mul(&not_stream, weight);
        row_weights[row] = row_weights[row].clone() + term;
    }
    for (&row, weight) in SPARTAN_OUTER_SECOND_GROUP_ROWS.iter().zip(&lagrange) {
        let term = ctx.mul(stream, weight);
        row_weights[row] = row_weights[row].clone() + term;
    }
    let matrices = spartan_outer_constraints::<Fr>();
    let columns = spartan_outer_opening_columns();
    let column_wire = |column: usize| -> Result<Lc, RelationError> {
        if column == rv64::const_column() {
            return Ok(Lc::one());
        }
        let index = columns
            .iter()
            .position(|&candidate| candidate == column)
            .ok_or_else(|| {
                RelationError::Geometry(format!("outer column {column} has no opening"))
            })?;
        wires.lc(&outer_opening(SPARTAN_OUTER_R1CS_INPUTS[index]))
    };
    let linear_form = |ctx: &mut Ctx, rows: &[SparseRow<Fr>]| -> Result<Lc, RelationError> {
        let mut form = Accum::default();
        for (row, weight) in rows.iter().zip(&row_weights) {
            let mut row_lc = Accum::default();
            for &(column, coefficient) in row {
                row_lc.add(&column_wire(column)?, coefficient);
            }
            let term = ctx.mul(weight, &row_lc.finish());
            form.add(&term, Fr::one());
        }
        Ok(form.finish())
    };
    let az = linear_form(ctx, &matrices.a)?;
    let bz = linear_form(ctx, &matrices.b)?;
    let (tau_high, tau_low) = tau
        .split_last()
        .ok_or_else(|| RelationError::Geometry("empty tau".into()))?;
    let kernel = centered_lagrange_kernel(
        ctx,
        SPARTAN_OUTER_UNISKIP_DOMAIN_SIZE,
        tau_high,
        &uniskip_challenge,
    );
    let eq_tau = eq(ctx, tau_low, &opening_point);
    let tau_kernel = ctx.mul(&kernel, &eq_tau);
    let kernel_az = ctx.mul(&tau_kernel, &az);
    let expected = ctx.mul(&kernel_az, &bz);
    finish_batch(ctx, &batch, &[expected], &final_claim);

    // The native linear-form weights, as the same combination of row weights
    // the per-row products above use (the parity guard checks them).
    let weight = |rows: &[SparseRow<Fr>], column: usize| -> Lc {
        let mut form = Accum::default();
        for (row, row_weight) in rows.iter().zip(&row_weights) {
            for &(candidate, coefficient) in row {
                if candidate == column {
                    form.add(row_weight, coefficient);
                }
            }
        }
        form.finish()
    };
    for (index, &column) in columns.iter().enumerate() {
        wires.derived(
            SpartanOuterPublic::AzWeight(index),
            weight(&matrices.a, column),
        );
        wires.derived(
            SpartanOuterPublic::BzWeight(index),
            weight(&matrices.b, column),
        );
    }
    if include_affine_terms {
        let constant = rv64::const_column();
        wires.derived(
            SpartanOuterPublic::AzConstant,
            weight(&matrices.a, constant),
        );
        wires.derived(
            SpartanOuterPublic::BzConstant,
            weight(&matrices.b, constant),
        );
    }
    wires.derived(SpartanOuterPublic::TauKernel, tau_kernel);

    Ok(Stage1 { point })
}
