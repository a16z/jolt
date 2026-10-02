//! Lowering of `jolt-claims` symbolic expressions to linear combinations,
//! sharing products across terms: each term's factors are grouped by source
//! kind and sorted, and every prefix product of a sorted group is memoized,
//! so an RA product or a challenge-power chain shared by many terms costs
//! its multiplications once.

use std::cmp::Ordering;
use std::collections::HashMap;

use jolt_claims::protocols::jolt::{JoltChallengeId, JoltDerivedId, JoltExpr, JoltOpeningId};
use jolt_claims::Source;
use jolt_field::{Fr, Zero};

use super::ctx::{Accum, Ctx, Lc};
use super::RelationError;

type Factor = Source<JoltOpeningId, JoltDerivedId, JoltChallengeId>;

/// A total order on factors: by kind, then by id.
fn compare(a: &Factor, b: &Factor) -> Ordering {
    fn rank(factor: &Factor) -> u8 {
        match factor {
            Source::Challenge(_) => 0,
            Source::Derived(_) => 1,
            Source::Opening(_) => 2,
        }
    }
    match (a, b) {
        (Source::Opening(x), Source::Opening(y)) => x.cmp(y),
        (Source::Derived(x), Source::Derived(y)) => x.cmp(y),
        (Source::Challenge(x), Source::Challenge(y)) => x.cmp(y),
        _ => rank(a).cmp(&rank(b)),
    }
}

#[derive(Default)]
pub(crate) struct Sources {
    pub openings: HashMap<JoltOpeningId, Lc>,
    pub challenges: HashMap<JoltChallengeId, Lc>,
    pub deriveds: HashMap<JoltDerivedId, Lc>,
}

impl Sources {
    pub(crate) fn opening(&mut self, id: JoltOpeningId, lc: Lc) {
        drop(self.openings.insert(id, lc));
    }

    pub(crate) fn challenge(&mut self, id: JoltChallengeId, lc: Lc) {
        drop(self.challenges.insert(id, lc));
    }

    pub(crate) fn derived(&mut self, id: JoltDerivedId, lc: Lc) {
        drop(self.deriveds.insert(id, lc));
    }

    pub(crate) fn opening_lc(&self, id: &JoltOpeningId) -> Result<Lc, RelationError> {
        self.openings
            .get(id)
            .cloned()
            .ok_or_else(|| RelationError::MissingSource {
                kind: "opening",
                id: format!("{id:?}"),
            })
    }

    fn resolve(&self, factor: &Factor) -> Result<Lc, RelationError> {
        match factor {
            Source::Opening(id) => self.opening_lc(id),
            Source::Challenge(id) => {
                self.challenges
                    .get(id)
                    .cloned()
                    .ok_or_else(|| RelationError::MissingSource {
                        kind: "challenge",
                        id: format!("{id:?}"),
                    })
            }
            Source::Derived(id) => {
                self.deriveds
                    .get(id)
                    .cloned()
                    .ok_or_else(|| RelationError::MissingSource {
                        kind: "derived",
                        id: format!("{id:?}"),
                    })
            }
        }
    }
}

#[derive(Default)]
pub(crate) struct Lowerer {
    memo: HashMap<Vec<Factor>, Lc>,
}

impl Lowerer {
    fn group_product(
        &mut self,
        ctx: &mut Ctx,
        sources: &Sources,
        group: &[Factor],
    ) -> Result<Option<Lc>, RelationError> {
        if group.is_empty() {
            return Ok(None);
        }
        if let Some(lc) = self.memo.get(group) {
            return Ok(Some(lc.clone()));
        }
        let (rest, last) = group.split_at(group.len() - 1);
        let last = sources.resolve(&last[0])?;
        let product = match self.group_product(ctx, sources, rest)? {
            None => last,
            Some(prefix) => ctx.mul(&prefix, &last),
        };
        drop(self.memo.insert(group.to_vec(), product.clone()));
        Ok(Some(product))
    }

    pub(crate) fn lower(
        &mut self,
        ctx: &mut Ctx,
        expr: &JoltExpr<Fr>,
        sources: &Sources,
    ) -> Result<Lc, RelationError> {
        let mut result = Accum::default();
        for term in &expr.terms {
            if term.coefficient.is_zero() {
                continue;
            }
            let mut openings = Vec::new();
            let mut deriveds = Vec::new();
            let mut challenges = Vec::new();
            for factor in &term.factors {
                match factor {
                    Source::Opening(_) => openings.push(factor.clone()),
                    Source::Derived(_) => deriveds.push(factor.clone()),
                    Source::Challenge(_) => challenges.push(factor.clone()),
                }
            }
            openings.sort_by(compare);
            deriveds.sort_by(compare);
            challenges.sort_by(compare);
            let mut product: Option<Lc> = None;
            for group in [challenges, deriveds, openings] {
                if let Some(part) = self.group_product(ctx, sources, &group)? {
                    product = Some(match product {
                        None => part,
                        Some(acc) => ctx.mul(&acc, &part),
                    });
                }
            }
            match product {
                None => result.add(&Lc::one(), term.coefficient),
                Some(product) => result.add(&product, term.coefficient),
            }
        }
        Ok(result.finish())
    }
}

pub(crate) fn lower(
    ctx: &mut Ctx,
    expr: &JoltExpr<Fr>,
    sources: &Sources,
) -> Result<Lc, RelationError> {
    Lowerer::default().lower(ctx, expr, sources)
}
