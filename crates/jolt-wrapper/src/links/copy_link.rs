use std::collections::BTreeMap;

use ark_bn254::Fr as ArkFr;
use ark_ff::batch_inversion;
use jolt_field::{Fr, One, Ring, Zero};
use jolt_hyperkzg::{NoopVerifierObserver, VerifierObserver};
use jolt_poly::{BindingOrder, Polynomial, UnivariatePoly};
use jolt_sumcheck::prover::ProveRounds;
use jolt_sumcheck::SumcheckError;
use rayon::prelude::*;

use super::{LinkError, DEGREE, WIRES};

#[derive(Clone, Copy, Debug)]
struct ActivePosition {
    row: usize,
    wire: usize,
    selector: Fr,
    id: Fr,
}

/// VK data describing three link slots per row. `ids` are logical edge
/// identifiers shared by the two sides; selectors disable unused slots.
#[derive(Clone, Debug)]
pub struct CopyLinkSide {
    rows: usize,
    active: Vec<ActivePosition>,
}

impl CopyLinkSide {
    pub fn new(selectors: [Vec<Fr>; WIRES], ids: [Vec<Fr>; WIRES]) -> Result<Self, LinkError> {
        let rows = selectors[0].len();
        if !rows.is_power_of_two()
            || selectors
                .iter()
                .chain(&ids)
                .any(|column| column.len() != rows)
        {
            return Err(LinkError::RowDomain {
                minimum: rows.next_power_of_two(),
                actual: rows,
            });
        }
        let active = (0..rows)
            .flat_map(|row| {
                let selectors = &selectors;
                let ids = &ids;
                (0..WIRES).filter_map(move |wire| {
                    let selector = selectors[wire][row];
                    (!selector.is_zero()).then_some(ActivePosition {
                        row,
                        wire,
                        selector,
                        id: ids[wire][row],
                    })
                })
            })
            .collect();
        Ok(Self { rows, active })
    }

    pub fn fixed_columns(&self) -> [Vec<Fr>; 2 * WIRES] {
        let mut columns = std::array::from_fn(|_| vec![Fr::zero(); self.rows]);
        for position in &self.active {
            columns[position.wire][position.row] = position.selector;
            columns[WIRES + position.wire][position.row] = position.id;
        }
        columns
    }
}

#[derive(Clone, Debug)]
pub struct CopyLink {
    pub left: CopyLinkSide,
    pub right: CopyLinkSide,
    rows: usize,
}

impl CopyLink {
    pub fn new(left: CopyLinkSide, right: CopyLinkSide) -> Result<Self, LinkError> {
        let rows = left.rows;
        if right.rows != rows {
            return Err(LinkError::RowDomain {
                minimum: rows,
                actual: right.rows,
            });
        }
        Ok(Self { left, right, rows })
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn witness<L: CopyLinkValueSource, R: CopyLinkValueSource>(
        &self,
        left_values: [L; WIRES],
        right_values: [R; WIRES],
        beta: Fr,
        gamma: Fr,
    ) -> Result<CopyLinkWitness<L, R>, LinkError> {
        let mut witnesses = batch_witnesses(std::iter::once((
            self,
            left_values,
            right_values,
            beta,
            gamma,
        )))?;
        witnesses.pop().ok_or(LinkError::Claims)
    }

    pub fn check<L: CopyLinkValueSource, R: CopyLinkValueSource>(
        &self,
        witness: &CopyLinkWitness<L, R>,
        beta: Fr,
        gamma: Fr,
    ) -> Result<(), LinkError> {
        check_side(
            &self.left,
            &witness.left_values,
            &witness.helpers[0],
            beta,
            gamma,
        )?;
        check_side(
            &self.right,
            &witness.right_values,
            &witness.helpers[1],
            beta,
            gamma,
        )?;
        if witness.helper_sum().is_zero() {
            Ok(())
        } else {
            Err(LinkError::Copy)
        }
    }

    pub fn prover<'a, L: CopyLinkValueSource, R: CopyLinkValueSource>(
        &'a self,
        witness: &'a CopyLinkWitness<L, R>,
        tau: Vec<Fr>,
        beta: Fr,
        gamma: Fr,
        weights: [Fr; 3],
    ) -> CopyLinkProver<'a, L, R> {
        CopyLinkProver::new(self, witness, tau, beta, gamma, weights)
    }
}

pub trait CopyLinkValueSource: Sync {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn value(&self, row: usize) -> Fr;
}

impl CopyLinkValueSource for Vec<Fr> {
    fn len(&self) -> usize {
        self.len()
    }

    fn value(&self, row: usize) -> Fr {
        self[row]
    }
}

pub struct CopyLinkWitness<L = Vec<Fr>, R = Vec<Fr>> {
    left_values: [L; WIRES],
    right_values: [R; WIRES],
    helpers: [Vec<(usize, Fr)>; 2],
    rows: usize,
}

impl<L, R> CopyLinkWitness<L, R> {
    pub fn helper_columns(&self) -> [Vec<Fr>; 2] {
        self.helpers.each_ref().map(|helper| {
            let mut column = vec![Fr::zero(); self.rows];
            for &(row, value) in helper {
                column[row] = value;
            }
            column
        })
    }

    fn helper_sum(&self) -> Fr {
        self.helpers[0].iter().map(|(_, value)| *value).sum::<Fr>()
            - self.helpers[1].iter().map(|(_, value)| *value).sum::<Fr>()
    }
}

pub(crate) fn batch_witnesses<'a, L, R>(
    requests: impl IntoIterator<Item = (&'a CopyLink, [L; WIRES], [R; WIRES], Fr, Fr)>,
) -> Result<Vec<CopyLinkWitness<L, R>>, LinkError>
where
    L: CopyLinkValueSource,
    R: CopyLinkValueSource,
{
    let requests = requests.into_iter().collect::<Vec<_>>();
    if requests.iter().any(|(link, left, right, _, _)| {
        left.iter()
            .map(|source| source.len())
            .chain(right.iter().map(|source| source.len()))
            .any(|rows| rows != link.rows)
    }) {
        return Err(LinkError::Claims);
    }
    let denominator_count = requests
        .iter()
        .map(|(link, _, _, _, _)| link.left.active.len() + link.right.active.len())
        .sum();
    let mut denominators = Vec::with_capacity(denominator_count);
    let mut positions = Vec::with_capacity(denominator_count);
    for (index, (link, left, right, beta, gamma)) in requests.iter().enumerate() {
        append_denominators(
            &mut denominators,
            &mut positions,
            (index, 0),
            &link.left,
            left,
            *beta,
            *gamma,
        );
        append_denominators(
            &mut denominators,
            &mut positions,
            (index, 1),
            &link.right,
            right,
            *beta,
            *gamma,
        );
    }
    if denominators.iter().any(Zero::is_zero) {
        return Err(LinkError::ZeroDenominator);
    }
    let mut inverses: Vec<ArkFr> = denominators.into_iter().map(ArkFr::from).collect();
    batch_inversion(&mut inverses);
    let mut helpers = (0..requests.len())
        .map(|_| [Vec::new(), Vec::new()])
        .collect::<Vec<_>>();
    for ((index, side, row, selector), inverse) in positions.into_iter().zip(inverses) {
        let value = selector * Fr::from(inverse);
        let helper = &mut helpers[index][side];
        if let Some((_, last_value)) = helper.last_mut().filter(|(last_row, _)| *last_row == row) {
            *last_value += value;
        } else {
            helper.push((row, value));
        }
    }
    Ok(requests
        .into_iter()
        .zip(helpers)
        .map(
            |((link, left_values, right_values, _, _), helpers)| CopyLinkWitness {
                left_values,
                right_values,
                helpers,
                rows: link.rows,
            },
        )
        .collect())
}

fn append_denominators<S: CopyLinkValueSource>(
    denominators: &mut Vec<Fr>,
    positions: &mut Vec<(usize, usize, usize, Fr)>,
    location: (usize, usize),
    side: &CopyLinkSide,
    values: &[S; WIRES],
    beta: Fr,
    gamma: Fr,
) {
    for position in &side.active {
        denominators.push(gamma + values[position.wire].value(position.row) + beta * position.id);
        positions.push((location.0, location.1, position.row, position.selector));
    }
}

fn check_side<S: CopyLinkValueSource>(
    side: &CopyLinkSide,
    values: &[S; WIRES],
    helpers: &[(usize, Fr)],
    beta: Fr,
    gamma: Fr,
) -> Result<(), LinkError> {
    let mut active = side.active.iter().peekable();
    for &(row, helper) in helpers {
        let mut ids = [Fr::zero(); WIRES];
        let mut selectors = [Fr::zero(); WIRES];
        while active.peek().is_some_and(|position| position.row < row) {
            let _ = active.next();
        }
        while active.peek().is_some_and(|position| position.row == row) {
            let position = active.next().ok_or(LinkError::Claims)?;
            ids[position.wire] = position.id;
            selectors[position.wire] = position.selector;
        }
        let values = std::array::from_fn(|wire| values[wire].value(row));
        if !grouped_selected_relation(values, ids, selectors, helper, beta, gamma).is_zero() {
            return Err(LinkError::Copy);
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CopyLinkClaims {
    pub left_selectors: [Fr; WIRES],
    pub left_ids: [Fr; WIRES],
    pub left_values: [Fr; WIRES],
    pub right_selectors: [Fr; WIRES],
    pub right_ids: [Fr; WIRES],
    pub right_values: [Fr; WIRES],
    pub helpers: [Fr; 2],
}

const SPARSE_BIND_ROUNDS: usize = 3;

#[derive(Clone, Copy, Default)]
struct SparseClaims {
    left_selectors: [Fr; WIRES],
    left_ids: [Fr; WIRES],
    right_selectors: [Fr; WIRES],
    right_ids: [Fr; WIRES],
    helpers: [Fr; 2],
}

impl SparseClaims {
    fn bind(self, other: Self, challenge: Fr) -> Self {
        let interpolate = |left: Fr, right: Fr| left + challenge * (right - left);
        Self {
            left_selectors: std::array::from_fn(|i| {
                interpolate(self.left_selectors[i], other.left_selectors[i])
            }),
            left_ids: std::array::from_fn(|i| interpolate(self.left_ids[i], other.left_ids[i])),
            right_selectors: std::array::from_fn(|i| {
                interpolate(self.right_selectors[i], other.right_selectors[i])
            }),
            right_ids: std::array::from_fn(|i| interpolate(self.right_ids[i], other.right_ids[i])),
            helpers: std::array::from_fn(|i| interpolate(self.helpers[i], other.helpers[i])),
        }
    }

    fn is_zero(&self) -> bool {
        self.left_selectors
            .iter()
            .chain(&self.left_ids)
            .chain(&self.right_selectors)
            .chain(&self.right_ids)
            .chain(&self.helpers)
            .all(Zero::is_zero)
    }

    fn claims(self, values: [Fr; 2 * WIRES]) -> CopyLinkClaims {
        CopyLinkClaims {
            left_selectors: self.left_selectors,
            left_ids: self.left_ids,
            left_values: std::array::from_fn(|i| values[i]),
            right_selectors: self.right_selectors,
            right_ids: self.right_ids,
            right_values: std::array::from_fn(|i| values[WIRES + i]),
            helpers: self.helpers,
        }
    }
}

struct SparseRows {
    values: Vec<(usize, SparseClaims)>,
    len: usize,
}

impl SparseRows {
    fn new<L, R>(link: &CopyLink, witness: &CopyLinkWitness<L, R>) -> Self {
        let mut rows = BTreeMap::<usize, SparseClaims>::new();
        for position in &link.left.active {
            let claims = rows.entry(position.row).or_default();
            claims.left_selectors[position.wire] = position.selector;
            claims.left_ids[position.wire] = position.id;
        }
        for position in &link.right.active {
            let claims = rows.entry(position.row).or_default();
            claims.right_selectors[position.wire] = position.selector;
            claims.right_ids[position.wire] = position.id;
        }
        for (side, helpers) in witness.helpers.iter().enumerate() {
            for &(row, helper) in helpers {
                rows.entry(row).or_default().helpers[side] = helper;
            }
        }
        Self {
            values: rows.into_iter().collect(),
            len: link.rows,
        }
    }

    fn pairs(&self) -> Vec<(usize, SparseClaims, SparseClaims)> {
        let half = self.len / 2;
        let split = self.values.partition_point(|(row, _)| *row < half);
        let (low, high) = self.values.split_at(split);
        let mut low = low.iter().peekable();
        let mut high = high.iter().peekable();
        let mut pairs = Vec::with_capacity(self.values.len());
        while low.peek().is_some() || high.peek().is_some() {
            let low_index = low.peek().map(|(row, _)| *row);
            let high_index = high.peek().map(|(row, _)| *row - half);
            let index = match (low_index, high_index) {
                (Some(low), Some(high)) => low.min(high),
                (Some(low), None) => low,
                (None, Some(high)) => high,
                (None, None) => unreachable!("one sparse pair remains"),
            };
            let low_value = low
                .next_if(|(row, _)| *row == index)
                .map_or_else(SparseClaims::default, |(_, value)| *value);
            let high_value = high
                .next_if(|(row, _)| *row == index + half)
                .map_or_else(SparseClaims::default, |(_, value)| *value);
            pairs.push((index, low_value, high_value));
        }
        pairs
    }

    fn bind(&mut self, challenge: Fr) {
        self.values = self
            .pairs()
            .into_iter()
            .filter_map(|(index, low, high)| {
                let value = low.bind(high, challenge);
                (!value.is_zero()).then_some((index, value))
            })
            .collect();
        self.len /= 2;
    }

    fn final_claims(&self) -> SparseClaims {
        self.values
            .first()
            .filter(|(row, _)| *row == 0)
            .map_or_else(SparseClaims::default, |(_, value)| *value)
    }
}

enum BoundValues<'a, L, R> {
    Borrowed {
        left: &'a [L; WIRES],
        right: &'a [R; WIRES],
        weights: Vec<Fr>,
        len: usize,
    },
    Dense([Polynomial<Fr>; 2 * WIRES]),
}

impl<L: CopyLinkValueSource, R: CopyLinkValueSource> BoundValues<'_, L, R> {
    fn pair(&self, index: usize) -> ([Fr; 2 * WIRES], [Fr; 2 * WIRES]) {
        match self {
            Self::Borrowed {
                left,
                right,
                weights,
                len,
            } => {
                let half = len / 2;
                let evaluate = |source: &dyn CopyLinkValueSource, suffix| {
                    weights
                        .iter()
                        .enumerate()
                        .map(|(prefix, &weight)| weight * source.value(suffix + prefix * *len))
                        .sum()
                };
                let low = std::array::from_fn(|column| {
                    if column < WIRES {
                        evaluate(&left[column], index)
                    } else {
                        evaluate(&right[column - WIRES], index)
                    }
                });
                let high = std::array::from_fn(|column| {
                    if column < WIRES {
                        evaluate(&left[column], index + half)
                    } else {
                        evaluate(&right[column - WIRES], index + half)
                    }
                });
                (low, high)
            }
            Self::Dense(columns) => {
                let half = columns[0].len() / 2;
                (
                    std::array::from_fn(|column| columns[column].evals()[index]),
                    std::array::from_fn(|column| columns[column].evals()[index + half]),
                )
            }
        }
    }

    fn bind(&mut self, challenge: Fr) {
        match self {
            Self::Borrowed {
                left,
                right,
                weights,
                len,
            } => {
                let old = std::mem::take(weights);
                weights.reserve(2 * old.len());
                for weight in old {
                    let high = weight * challenge;
                    weights.push(weight - high);
                    weights.push(high);
                }
                *len /= 2;
                if weights.len() == 1 << SPARSE_BIND_ROUNDS || *len == 1 {
                    let bound_len = *len;
                    let dense = std::array::from_fn(|column| {
                        let source: &dyn CopyLinkValueSource = if column < WIRES {
                            &left[column]
                        } else {
                            &right[column - WIRES]
                        };
                        let values = (0..bound_len)
                            .into_par_iter()
                            .map(|suffix| {
                                weights
                                    .iter()
                                    .enumerate()
                                    .map(|(prefix, &weight)| {
                                        weight * source.value(suffix + prefix * bound_len)
                                    })
                                    .sum()
                            })
                            .collect();
                        Polynomial::new(values)
                    });
                    *self = Self::Dense(dense);
                }
            }
            Self::Dense(columns) => {
                for column in columns {
                    column.bind_with_order(challenge, BindingOrder::HighToLow);
                }
            }
        }
    }

    fn finals(&self) -> [Fr; 2 * WIRES] {
        match self {
            Self::Borrowed {
                left,
                right,
                weights,
                len,
            } => std::array::from_fn(|column| {
                let source: &dyn CopyLinkValueSource = if column < WIRES {
                    &left[column]
                } else {
                    &right[column - WIRES]
                };
                weights
                    .iter()
                    .enumerate()
                    .map(|(prefix, &weight)| weight * source.value(prefix * *len))
                    .sum()
            }),
            Self::Dense(columns) => std::array::from_fn(|column| columns[column].evals()[0]),
        }
    }
}

pub struct CopyLinkProver<'a, L = Vec<Fr>, R = Vec<Fr>> {
    sparse: SparseRows,
    values: BoundValues<'a, L, R>,
    tau: Vec<Fr>,
    point: Vec<Fr>,
    eq_prefix: Fr,
    beta: Fr,
    gamma: Fr,
    weights: [Fr; 3],
    rounds: usize,
    input_claim: Fr,
}

impl<'a, L: CopyLinkValueSource, R: CopyLinkValueSource> CopyLinkProver<'a, L, R> {
    fn new(
        link: &CopyLink,
        witness: &'a CopyLinkWitness<L, R>,
        tau: Vec<Fr>,
        beta: Fr,
        gamma: Fr,
        weights: [Fr; 3],
    ) -> Self {
        assert_eq!(tau.len(), link.rows.trailing_zeros() as usize);
        let input_claim = weights[2] * witness.helper_sum();
        let rounds = tau.len();
        Self {
            sparse: SparseRows::new(link, witness),
            values: BoundValues::Borrowed {
                left: &witness.left_values,
                right: &witness.right_values,
                weights: vec![Fr::one()],
                len: link.rows,
            },
            tau,
            point: Vec::new(),
            eq_prefix: Fr::one(),
            beta,
            gamma,
            weights,
            rounds,
            input_claim,
        }
    }

    pub fn input_claim(&self) -> Fr {
        self.input_claim
    }

    pub fn claims(&self) -> CopyLinkClaims {
        self.sparse.final_claims().claims(self.values.finals())
    }

    fn bind(&mut self, challenge: Fr) {
        let tau = self.tau[self.point.len()];
        self.eq_prefix *= Fr::one() - tau + challenge * (tau + tau - Fr::one());
        self.point.push(challenge);
        self.sparse.bind(challenge);
        self.values.bind(challenge);
    }

    fn eq_pair(&self, index: usize) -> (Fr, Fr) {
        let tau = &self.tau[self.point.len()..];
        let current = tau[0];
        let suffix =
            tau[1..]
                .iter()
                .enumerate()
                .fold(self.eq_prefix, |value, (offset, &challenge)| {
                    let bit = 1 << (tau.len() - 2 - offset);
                    value
                        * if index & bit == 0 {
                            Fr::one() - challenge
                        } else {
                            challenge
                        }
                });
        (suffix * (Fr::one() - current), suffix * current)
    }
}

impl<L: CopyLinkValueSource, R: CopyLinkValueSource> ProveRounds<Fr> for CopyLinkProver<'_, L, R> {
    fn num_rounds(&self) -> usize {
        self.rounds
    }

    fn prove_round(
        &mut self,
        bind: Option<Fr>,
        round: usize,
        previous_claim: Fr,
    ) -> Result<UnivariatePoly<Fr>, SumcheckError<Fr>> {
        if let Some(challenge) = bind {
            self.bind(challenge);
        }
        let pairs = self.sparse.pairs();
        let evaluations = pairs
            .into_par_iter()
            .map(|(index, low_sparse, high_sparse)| {
                let mut local = [Fr::zero(); DEGREE + 1];
                let (low_values, high_values) = self.values.pair(index);
                let (low_eq, high_eq) = self.eq_pair(index);
                for (x, value) in local.iter_mut().enumerate() {
                    let x = Fr::from_u64(x as u64);
                    let values = std::array::from_fn(|column| {
                        low_values[column] + x * (high_values[column] - low_values[column])
                    });
                    let claims = low_sparse.bind(high_sparse, x).claims(values);
                    let eq = low_eq + x * (high_eq - low_eq);
                    *value = copy_link_value(self.beta, self.gamma, self.weights, eq, &claims);
                }
                local
            })
            .reduce(
                || [Fr::zero(); DEGREE + 1],
                |mut sum, local| {
                    for (sum, value) in sum.iter_mut().zip(local) {
                        *sum += value;
                    }
                    sum
                },
            );
        if evaluations[0] + evaluations[1] != previous_claim {
            return Err(SumcheckError::RoundCheckFailed {
                round,
                expected: previous_claim,
                actual: evaluations[0] + evaluations[1],
            });
        }
        Ok(UnivariatePoly::from_evals(&evaluations))
    }

    fn finish_rounds(&mut self, bind: Fr) -> Result<(), SumcheckError<Fr>> {
        self.bind(bind);
        Ok(())
    }

    fn append_bound_values(&self, values: &mut Vec<Fr>) {
        let claims = self.claims();
        values.extend(claims.left_selectors);
        values.extend(claims.left_ids);
        values.extend(claims.right_selectors);
        values.extend(claims.right_ids);
        values.extend(claims.helpers);
    }
}

fn copy_link_value(beta: Fr, gamma: Fr, weights: [Fr; 3], eq: Fr, claims: &CopyLinkClaims) -> Fr {
    copy_link_value_observed(beta, gamma, weights, eq, claims, &mut NoopVerifierObserver)
}

fn copy_link_value_observed<O: VerifierObserver>(
    beta: Fr,
    gamma: Fr,
    weights: [Fr; 3],
    eq: Fr,
    claims: &CopyLinkClaims,
    observer: &mut O,
) -> Fr {
    let left = grouped_selected_relation_observed(
        claims.left_values,
        claims.left_ids,
        claims.left_selectors,
        claims.helpers[0],
        beta,
        gamma,
        observer,
    );
    let right = grouped_selected_relation_observed(
        claims.right_values,
        claims.right_ids,
        claims.right_selectors,
        claims.helpers[1],
        beta,
        gamma,
        observer,
    );
    let eq_left = observer.fr_mul(eq, left);
    let left = observer.fr_mul(weights[0], eq_left);
    let eq_right = observer.fr_mul(eq, right);
    let right = observer.fr_mul(weights[1], eq_right);
    left + right + observer.fr_mul(weights[2], claims.helpers[0] - claims.helpers[1])
}

fn grouped_selected_relation(
    values: [Fr; WIRES],
    ids: [Fr; WIRES],
    selectors: [Fr; WIRES],
    helper: Fr,
    beta: Fr,
    gamma: Fr,
) -> Fr {
    grouped_selected_relation_observed(
        values,
        ids,
        selectors,
        helper,
        beta,
        gamma,
        &mut NoopVerifierObserver,
    )
}

fn grouped_selected_relation_observed<O: VerifierObserver>(
    values: [Fr; WIRES],
    ids: [Fr; WIRES],
    selectors: [Fr; WIRES],
    helper: Fr,
    beta: Fr,
    gamma: Fr,
    observer: &mut O,
) -> Fr {
    let denominators: [Fr; WIRES] =
        std::array::from_fn(|i| gamma + values[i] + observer.fr_mul(beta, ids[i]));
    let product01 = observer.fr_mul(denominators[0], denominators[1]);
    let product = observer.fr_mul(product01, denominators[2]);
    let pair12 = observer.fr_mul(denominators[1], denominators[2]);
    let selected0 = observer.fr_mul(selectors[0], pair12);
    let pair02 = observer.fr_mul(denominators[0], denominators[2]);
    let selected1 = observer.fr_mul(selectors[1], pair02);
    let pair01 = observer.fr_mul(denominators[0], denominators[1]);
    let selected2 = observer.fr_mul(selectors[2], pair01);
    observer.fr_mul(helper, product) - selected0 - selected1 - selected2
}

#[cfg(test)]
#[expect(clippy::unwrap_used, reason = "tests fail on invalid synthetic links")]
mod tests {
    use jolt_field::{Fr, One, Zero};
    use jolt_poly::EqPolynomial;
    use jolt_sumcheck::prover::ProveRounds;

    use super::*;
    use crate::links::{
        evaluate_terms_observed, AffineForm, ColumnId, CopyLinkTermExporter, CopyLinkTermSide,
        CopyLinkTermsContext, TermContext, TermExporter,
    };
    use crate::stream::VerifierCost;

    #[test]
    fn synthetic_link_accepts_permutation_and_rejects_value_change() {
        let rows = 8;
        let selectors = std::array::from_fn(|wire| {
            (0..rows)
                .map(|row| Fr::from_u64(u64::from(row < 3 && wire == 0)))
                .collect()
        });
        let left_ids =
            std::array::from_fn(|_| (0..rows).map(|row| Fr::from_u64(row as u64)).collect());
        let right_ids = left_ids.clone();
        let left = CopyLinkSide::new(selectors.clone(), left_ids).unwrap();
        let right = CopyLinkSide::new(selectors, right_ids).unwrap();
        let link = CopyLink::new(left, right).unwrap();
        let values: [Vec<Fr>; WIRES] = std::array::from_fn(|wire| {
            (0..rows)
                .map(|row| Fr::from_u64((10 * wire + row) as u64))
                .collect()
        });
        let beta = Fr::from_u64(19);
        let gamma = Fr::from_u64(23);
        let witness = link
            .witness(values.clone(), values.clone(), beta, gamma)
            .unwrap();
        link.check(&witness, beta, gamma).unwrap();
        let tau = vec![Fr::from_u64(3), Fr::from_u64(5), Fr::from_u64(7)];
        let weights = [Fr::one(), Fr::from_u64(11), Fr::from_u64(13)];
        let mut prover = link.prover(&witness, tau.clone(), beta, gamma, weights);
        assert_eq!(prover.input_claim(), Fr::zero());
        let mut claim = prover.input_claim();
        let mut point = Vec::new();
        let mut bind = None;
        for round in 0..prover.num_rounds() {
            let polynomial = prover.prove_round(bind, round, claim).unwrap();
            let challenge = Fr::from_u64((29 + round) as u64);
            claim = polynomial.evaluate(challenge);
            point.push(challenge);
            bind = Some(challenge);
        }
        prover.finish_rounds(bind.unwrap()).unwrap();
        let claims = prover.claims();
        let final_claim = claim;

        let column_claims = [
            claims.left_selectors.as_slice(),
            claims.left_ids.as_slice(),
            claims.left_values.as_slice(),
            claims.right_selectors.as_slice(),
            claims.right_ids.as_slice(),
            claims.right_values.as_slice(),
            claims.helpers.as_slice(),
        ]
        .concat();
        let column = |slot| ColumnId { group: 0, slot };
        let form_columns = |base| {
            std::array::from_fn(|wire| AffineForm {
                constant: Fr::zero(),
                weights: vec![(column(base + wire), Fr::one())],
            })
        };
        let term_context = CopyLinkTermsContext {
            left: CopyLinkTermSide {
                selectors: [column(0), column(1), column(2)],
                ids: form_columns(3),
                values: form_columns(6),
                helper: column(18),
            },
            right: CopyLinkTermSide {
                selectors: [column(9), column(10), column(11)],
                ids: form_columns(12),
                values: form_columns(15),
                helper: column(19),
            },
            beta,
            gamma,
            eq: EqPolynomial::<Fr>::mle(&tau, &point),
            relation_weights: weights,
            stage_coefficient: Fr::from_u64(31),
        };
        let exporter = CopyLinkTermExporter {
            link: &link,
            left: term_context.left.clone(),
            right: term_context.right.clone(),
            tau: &tau,
            beta,
            gamma,
            relation_weights: weights,
            member_index: 0,
        };
        let export_context = TermContext {
            row_point: &point,
            batching_coefficients: &[term_context.stage_coefficient],
            challenges: &[],
        };
        let mut term_cost = VerifierCost::default();
        let terms = exporter.terms_observed(&export_context, &mut term_cost);
        assert_eq!(exporter.terms(&export_context), terms);
        assert_eq!(link.terms(&term_context), terms);
        assert_eq!(term_cost.fr_mul, 13);
        assert_eq!(terms.len(), crate::links::COPY_LINK_TERM_COUNT);
        assert_eq!(
            terms.iter().map(|term| term.factors.len()).max(),
            Some(crate::links::MAX_FACTORS)
        );
        assert_eq!(
            term_context.stage_coefficient * final_claim,
            evaluate_terms_observed(
                &terms,
                &|column| {
                    column_claims
                        .get(column.slot)
                        .copied()
                        .ok_or(LinkError::Claims)
                },
                &mut term_cost,
            )
            .unwrap()
        );
        assert_eq!(term_cost.fr_mul, 59);

        let mut right_bad = values.clone();
        right_bad[0][1] += Fr::one();
        let bad = link.witness(values, right_bad, beta, gamma).unwrap();
        assert!(link.check(&bad, beta, gamma).is_err());
    }
}
