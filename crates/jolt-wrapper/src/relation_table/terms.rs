use jolt_field::{Fr, One, Ring, Zero};
use jolt_hyperkzg::NoopVerifierObserver;

use crate::stream::{AffineForm, ColumnId, Term, TermContext, TermExporter, TermObserver};

use super::{
    CopyLink, DoryScalarLink, RelationTable, RelationTableError, ACTIVE, H_ID, H_SIGMA, Q_C, Q_L,
    Q_M, Q_O, Q_R, SIGMA_A, TOTAL_COLUMNS, WIRES, WIRE_A, WIRE_B, WIRE_C,
};

pub const RELATION_TERM_COUNT: usize = 15;
pub const COPY_LINK_TERM_COUNT: usize = 10;
pub const DORY_SCALAR_TERM_COUNT: usize = 1;
pub const MAX_FACTORS: usize = 4;

pub fn evaluate_terms(
    terms: &[Term],
    evaluate_column: &dyn Fn(ColumnId) -> Result<Fr, RelationTableError>,
) -> Result<Fr, RelationTableError> {
    evaluate_terms_observed(terms, evaluate_column, &mut NoopVerifierObserver)
}

pub fn evaluate_terms_observed<O: TermObserver + ?Sized>(
    terms: &[Term],
    evaluate_column: &dyn Fn(ColumnId) -> Result<Fr, RelationTableError>,
    observer: &mut O,
) -> Result<Fr, RelationTableError> {
    terms.iter().try_fold(Fr::zero(), |sum, term| {
        let value = term
            .factors
            .iter()
            .try_fold(term.coefficient, |product, factor| {
                let factor = evaluate_affine_observed(factor, evaluate_column, observer)?;
                Ok::<Fr, RelationTableError>(observer.fr_mul(product, factor))
            })?;
        Ok(sum + value)
    })
}

fn constant_form(value: Fr) -> AffineForm {
    AffineForm {
        constant: value,
        weights: Vec::new(),
    }
}

fn column_form(column: ColumnId) -> AffineForm {
    AffineForm {
        constant: Fr::zero(),
        weights: vec![(column, Fr::one())],
    }
}

fn term(coefficient: Fr, factors: Vec<AffineForm>) -> Term {
    Term {
        coefficient,
        factors,
    }
}

fn evaluate_affine_observed<O: TermObserver + ?Sized>(
    form: &AffineForm,
    evaluate_column: &dyn Fn(ColumnId) -> Result<Fr, RelationTableError>,
    observer: &mut O,
) -> Result<Fr, RelationTableError> {
    form.weights
        .iter()
        .try_fold(form.constant, |value, &(column, weight)| {
            let column = evaluate_column(column)?;
            Ok(value
                + if weight.is_one() {
                    column
                } else {
                    observer.fr_mul(weight, column)
                })
        })
}

pub struct RelationTermsContext<'a> {
    pub columns: [ColumnId; TOTAL_COLUMNS],
    pub tau: &'a [Fr],
    pub point: &'a [Fr],
    pub beta: Fr,
    pub gamma: Fr,
    pub relation_weights: [Fr; 3],
    pub stage_coefficient: Fr,
}

pub struct RelationTermExporter<'a> {
    pub rows: usize,
    pub columns: [ColumnId; TOTAL_COLUMNS],
    pub tau: &'a [Fr],
    pub beta: Fr,
    pub gamma: Fr,
    pub relation_weights: [Fr; 3],
    pub member_index: usize,
}

impl RelationTermExporter<'_> {
    #[expect(
        clippy::indexing_slicing,
        reason = "the assembly validates the exporter member count before term export"
    )]
    pub fn terms_observed<O: TermObserver + ?Sized>(
        &self,
        context: &TermContext<'_>,
        observer: &mut O,
    ) -> Vec<Term> {
        let term_context = RelationTermsContext {
            columns: self.columns,
            tau: self.tau,
            point: context.row_point,
            beta: self.beta,
            gamma: self.gamma,
            relation_weights: self.relation_weights,
            stage_coefficient: context.batching_coefficients[self.member_index],
        };
        relation_terms_observed(self.rows, &term_context, observer)
            .unwrap_or_else(|_| unreachable!("assembly row point has the table dimension"))
    }
}

impl TermExporter for RelationTermExporter<'_> {
    fn max_factors(&self) -> usize {
        4
    }

    fn terms(&self, context: &TermContext<'_>) -> Vec<Term> {
        self.terms_observed(context, &mut NoopVerifierObserver)
    }

    fn terms_observed(
        &self,
        context: &TermContext<'_>,
        observer: &mut dyn TermObserver,
    ) -> Vec<Term> {
        RelationTermExporter::terms_observed(self, context, observer)
    }
}

impl RelationTable {
    pub fn terms(&self, ctx: &RelationTermsContext<'_>) -> Result<Vec<Term>, RelationTableError> {
        relation_terms(self.rows(), ctx)
    }

    pub fn terms_observed<O: TermObserver + ?Sized>(
        &self,
        ctx: &RelationTermsContext<'_>,
        observer: &mut O,
    ) -> Result<Vec<Term>, RelationTableError> {
        relation_terms_observed(self.rows(), ctx, observer)
    }
}

fn relation_terms(
    rows: usize,
    ctx: &RelationTermsContext<'_>,
) -> Result<Vec<Term>, RelationTableError> {
    relation_terms_observed(rows, ctx, &mut NoopVerifierObserver)
}

pub(super) fn relation_terms_observed<O: TermObserver + ?Sized>(
    rows: usize,
    ctx: &RelationTermsContext<'_>,
    observer: &mut O,
) -> Result<Vec<Term>, RelationTableError> {
    if ctx.point.len() != rows.trailing_zeros() as usize || ctx.tau.len() != ctx.point.len() {
        return Err(RelationTableError::Claims);
    }
    let eq = eq_mle_observed(ctx.tau, ctx.point, observer);
    let identity = identity_mle_observed(ctx.point, observer);
    let stage_eq = observer.fr_mul(ctx.stage_coefficient, eq);
    let column = |index| column_form(ctx.columns[index]);
    let wires = [column(WIRE_A), column(WIRE_B), column(WIRE_C)];
    let ids =
        std::array::from_fn(|wire| constant_form(Fr::from_u64((wire * rows) as u64) + identity));
    let sigmas = std::array::from_fn(|wire| column(SIGMA_A + wire));
    let mut terms = vec![
        term(stage_eq, vec![column(Q_L), wires[0].clone()]),
        term(stage_eq, vec![column(Q_R), wires[1].clone()]),
        term(stage_eq, vec![column(Q_O), wires[2].clone()]),
        term(
            stage_eq,
            vec![column(Q_M), wires[0].clone(), wires[1].clone()],
        ),
        term(stage_eq, vec![column(Q_C)]),
    ];
    let id_coefficient = observer.fr_mul(ctx.relation_weights[0], stage_eq);
    append_grouped_terms(
        &mut terms,
        id_coefficient,
        std::array::from_fn(|_| column(ACTIVE)),
        column(H_ID),
        &wires,
        &ids,
        ctx.beta,
        ctx.gamma,
        observer,
    );
    let sigma_coefficient = observer.fr_mul(ctx.relation_weights[1], stage_eq);
    append_grouped_terms(
        &mut terms,
        sigma_coefficient,
        std::array::from_fn(|_| column(ACTIVE)),
        column(H_SIGMA),
        &wires,
        &sigmas,
        ctx.beta,
        ctx.gamma,
        observer,
    );
    let sum_coefficient = observer.fr_mul(ctx.stage_coefficient, ctx.relation_weights[2]);
    terms.push(term(sum_coefficient, vec![column(H_ID)]));
    terms.push(term(-sum_coefficient, vec![column(H_SIGMA)]));
    Ok(terms)
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CopyLinkTermSide {
    pub selectors: [ColumnId; WIRES],
    pub ids: [AffineForm; WIRES],
    pub values: [AffineForm; WIRES],
    pub helper: ColumnId,
}

pub struct CopyLinkTermsContext {
    pub left: CopyLinkTermSide,
    pub right: CopyLinkTermSide,
    pub beta: Fr,
    pub gamma: Fr,
    pub eq: Fr,
    pub relation_weights: [Fr; 3],
    pub stage_coefficient: Fr,
}

pub struct CopyLinkTermExporter<'a> {
    pub link: &'a CopyLink,
    pub left: CopyLinkTermSide,
    pub right: CopyLinkTermSide,
    pub tau: &'a [Fr],
    pub beta: Fr,
    pub gamma: Fr,
    pub relation_weights: [Fr; 3],
    pub member_index: usize,
}

pub struct PublicCopyLinkTermExporter<'a> {
    pub link: &'a CopyLink,
    pub left: CopyLinkTermSide,
    pub right: CopyLinkTermSide,
    pub public_wire: usize,
    pub public_rows: &'a [usize],
    pub public_values: &'a [Fr],
    pub tau: &'a [Fr],
    pub beta: Fr,
    pub gamma: Fr,
    pub relation_weights: [Fr; 3],
    pub member_index: usize,
}

impl PublicCopyLinkTermExporter<'_> {
    #[expect(
        clippy::indexing_slicing,
        reason = "the key validates public rows, values, and the copy-link wire"
    )]
    fn terms_observed<O: TermObserver + ?Sized>(
        &self,
        context: &TermContext<'_>,
        observer: &mut O,
    ) -> Vec<Term> {
        assert_eq!(self.public_rows.len(), self.public_values.len());
        let mut public_evaluation = Fr::zero();
        for (&row, &value) in self.public_rows.iter().zip(self.public_values) {
            let eq = context.row_point.iter().enumerate().fold(
                Fr::one(),
                |acc, (index, &coordinate)| {
                    let bit = (row >> (context.row_point.len() - index - 1)) & 1;
                    observer.fr_mul(
                        acc,
                        if bit == 1 {
                            coordinate
                        } else {
                            Fr::one() - coordinate
                        },
                    )
                },
            );
            public_evaluation += observer.fr_mul(value, eq);
        }
        let mut left = self.left.clone();
        left.values[self.public_wire] = constant_form(public_evaluation);
        CopyLinkTermExporter {
            link: self.link,
            left,
            right: self.right.clone(),
            tau: self.tau,
            beta: self.beta,
            gamma: self.gamma,
            relation_weights: self.relation_weights,
            member_index: self.member_index,
        }
        .terms_observed(context, observer)
    }
}

impl TermExporter for PublicCopyLinkTermExporter<'_> {
    fn max_factors(&self) -> usize {
        4
    }

    fn terms(&self, context: &TermContext<'_>) -> Vec<Term> {
        self.terms_observed(context, &mut NoopVerifierObserver)
    }

    fn terms_observed(
        &self,
        context: &TermContext<'_>,
        observer: &mut dyn TermObserver,
    ) -> Vec<Term> {
        PublicCopyLinkTermExporter::terms_observed(self, context, observer)
    }
}

impl CopyLinkTermExporter<'_> {
    #[expect(
        clippy::indexing_slicing,
        reason = "the assembly validates the exporter member count before term export"
    )]
    pub fn terms_observed<O: TermObserver + ?Sized>(
        &self,
        context: &TermContext<'_>,
        observer: &mut O,
    ) -> Vec<Term> {
        assert_eq!(self.tau.len(), context.row_point.len());
        assert_eq!(self.tau.len(), self.link.rows().trailing_zeros() as usize);
        let eq = eq_mle_observed(self.tau, context.row_point, observer);
        self.link.terms_observed(
            &CopyLinkTermsContext {
                left: self.left.clone(),
                right: self.right.clone(),
                beta: self.beta,
                gamma: self.gamma,
                eq,
                relation_weights: self.relation_weights,
                stage_coefficient: context.batching_coefficients[self.member_index],
            },
            observer,
        )
    }
}

impl TermExporter for CopyLinkTermExporter<'_> {
    fn max_factors(&self) -> usize {
        4
    }

    fn terms(&self, context: &TermContext<'_>) -> Vec<Term> {
        self.terms_observed(context, &mut NoopVerifierObserver)
    }

    fn terms_observed(
        &self,
        context: &TermContext<'_>,
        observer: &mut dyn TermObserver,
    ) -> Vec<Term> {
        CopyLinkTermExporter::terms_observed(self, context, observer)
    }
}

impl CopyLink {
    pub fn terms(&self, ctx: &CopyLinkTermsContext) -> Vec<Term> {
        self.terms_observed(ctx, &mut NoopVerifierObserver)
    }

    pub fn terms_observed<O: TermObserver + ?Sized>(
        &self,
        ctx: &CopyLinkTermsContext,
        observer: &mut O,
    ) -> Vec<Term> {
        let mut terms = Vec::with_capacity(COPY_LINK_TERM_COUNT);
        let stage_eq = observer.fr_mul(ctx.stage_coefficient, ctx.eq);
        let left_coefficient = observer.fr_mul(ctx.relation_weights[0], stage_eq);
        append_grouped_terms(
            &mut terms,
            left_coefficient,
            ctx.left.selectors.map(column_form),
            column_form(ctx.left.helper),
            &ctx.left.values,
            &ctx.left.ids,
            ctx.beta,
            ctx.gamma,
            observer,
        );
        let right_coefficient = observer.fr_mul(ctx.relation_weights[1], stage_eq);
        append_grouped_terms(
            &mut terms,
            right_coefficient,
            ctx.right.selectors.map(column_form),
            column_form(ctx.right.helper),
            &ctx.right.values,
            &ctx.right.ids,
            ctx.beta,
            ctx.gamma,
            observer,
        );
        let sum_coefficient = observer.fr_mul(ctx.stage_coefficient, ctx.relation_weights[2]);
        terms.push(term(sum_coefficient, vec![column_form(ctx.left.helper)]));
        terms.push(term(-sum_coefficient, vec![column_form(ctx.right.helper)]));
        terms
    }
}

pub struct DoryScalarTermsContext<'a> {
    pub wire: ColumnId,
    pub point: &'a [Fr],
    pub stage_coefficient: Fr,
}

pub struct DoryScalarTermExporter<'a> {
    pub link: &'a DoryScalarLink<'a>,
    pub wire: ColumnId,
    pub member_index: usize,
}

impl DoryScalarTermExporter<'_> {
    #[expect(
        clippy::indexing_slicing,
        reason = "the assembly validates the exporter member count before term export"
    )]
    pub fn terms_observed<O: TermObserver + ?Sized>(
        &self,
        context: &TermContext<'_>,
        observer: &mut O,
    ) -> Vec<Term> {
        self.link
            .terms_observed(
                &DoryScalarTermsContext {
                    wire: self.wire,
                    point: context.row_point,
                    stage_coefficient: context.batching_coefficients[self.member_index],
                },
                observer,
            )
            .unwrap_or_else(|_| unreachable!("assembly row point has the table dimension"))
    }
}

impl TermExporter for DoryScalarTermExporter<'_> {
    fn max_factors(&self) -> usize {
        1
    }

    fn terms(&self, context: &TermContext<'_>) -> Vec<Term> {
        self.terms_observed(context, &mut NoopVerifierObserver)
    }

    fn terms_observed(
        &self,
        context: &TermContext<'_>,
        observer: &mut dyn TermObserver,
    ) -> Vec<Term> {
        DoryScalarTermExporter::terms_observed(self, context, observer)
    }
}

impl DoryScalarLink<'_> {
    pub fn terms(&self, ctx: &DoryScalarTermsContext<'_>) -> Result<Vec<Term>, RelationTableError> {
        self.terms_observed(ctx, &mut NoopVerifierObserver)
    }

    pub fn terms_observed<O: TermObserver + ?Sized>(
        &self,
        ctx: &DoryScalarTermsContext<'_>,
        observer: &mut O,
    ) -> Result<Vec<Term>, RelationTableError> {
        if ctx.point.len() != self.rows().trailing_zeros() as usize {
            return Err(RelationTableError::Claims);
        }
        let weight = self.weight_at_observed(ctx.point, observer);
        Ok(vec![term(
            observer.fr_mul(ctx.stage_coefficient, weight),
            vec![column_form(ctx.wire)],
        )])
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "matches the grouped LogUp relation inputs without another protocol type"
)]
fn append_grouped_terms<O: TermObserver + ?Sized>(
    terms: &mut Vec<Term>,
    coefficient: Fr,
    selectors: [AffineForm; WIRES],
    helper: AffineForm,
    values: &[AffineForm; WIRES],
    ids: &[AffineForm; WIRES],
    beta: Fr,
    gamma: Fr,
    observer: &mut O,
) {
    let denominators: [AffineForm; WIRES] =
        std::array::from_fn(|wire| denominator(&values[wire], &ids[wire], beta, gamma, observer));
    terms.push(term(
        coefficient,
        vec![
            helper,
            denominators[0].clone(),
            denominators[1].clone(),
            denominators[2].clone(),
        ],
    ));
    for (skipped, selector) in selectors.iter().enumerate() {
        let mut factors = Vec::with_capacity(WIRES);
        factors.push(selector.clone());
        factors.extend(
            denominators
                .iter()
                .enumerate()
                .filter(|(wire, _)| *wire != skipped)
                .map(|(_, denominator)| denominator.clone()),
        );
        terms.push(term(-coefficient, factors));
    }
}

fn denominator<O: TermObserver + ?Sized>(
    value: &AffineForm,
    id: &AffineForm,
    beta: Fr,
    gamma: Fr,
    observer: &mut O,
) -> AffineForm {
    let constant = gamma
        + value.constant
        + if id.constant.is_zero() {
            Fr::zero()
        } else {
            observer.fr_mul(beta, id.constant)
        };
    let mut weights = value.weights.clone();
    weights.extend(id.weights.iter().map(|&(column, weight)| {
        (
            column,
            if weight.is_one() {
                beta
            } else {
                observer.fr_mul(beta, weight)
            },
        )
    }));
    AffineForm { constant, weights }
}

fn identity_mle_observed<O: TermObserver + ?Sized>(point: &[Fr], observer: &mut O) -> Fr {
    point
        .iter()
        .enumerate()
        .fold(Fr::zero(), |sum, (index, &coordinate)| {
            sum + observer.fr_mul(coordinate, Fr::from_u64(1 << (point.len() - index - 1)))
        })
}

fn eq_mle_observed<O: TermObserver + ?Sized>(left: &[Fr], right: &[Fr], observer: &mut O) -> Fr {
    left.iter()
        .zip(right)
        .fold(Fr::one(), |value, (&left, &right)| {
            let both = observer.fr_mul(left, right);
            let neither = observer.fr_mul(Fr::one() - left, Fr::one() - right);
            observer.fr_mul(value, both + neither)
        })
}
