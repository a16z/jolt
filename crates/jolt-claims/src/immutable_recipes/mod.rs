//! Closed, generated expression recipes. Runtime proof inputs never supply this table.

mod flags;
mod key;
#[cfg(all(feature = "immutable-recipes-record", not(target_arch = "riscv64")))]
pub mod record;
pub use key::{KeySink, KeyVisit};
/// Exact source-owned base eligibility for generic composed relations.
pub struct BasePermit<S> {
    marker: std::marker::PhantomData<fn() -> S>,
}
impl<S> BasePermit<S> {
    pub(crate) fn new() -> Self {
        Self {
            marker: std::marker::PhantomData,
        }
    }
}

use crate::{Expr, Source, Term};
use jolt_field::{ImmutableRecipeField, Ring, IMMUTABLE_RECIPE_MODULUS};
use std::cmp::Ordering;
use std::collections::BTreeSet;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Phase {
    Input,
    Output,
    Support,
}
impl Phase {
    const fn code(self) -> u8 {
        match self {
            Self::Input => 0,
            Self::Output => 1,
            Self::Support => 2,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, thiserror::Error)]
pub enum RecipeError {
    #[error("invalid immutable recipe structure")]
    Structure,
    #[error("invalid immutable recipe identifier")]
    Identifier,
    #[error("invalid immutable recipe coefficient")]
    Coefficient,
    #[error("unsorted or duplicate immutable recipe key")]
    Duplicate,
}

/// Typed identifiers are compiler-checked, not deserialized at a callback site.
/// Private fields prevent constructing an unchecked record. Generated const
/// initializers must handle the checked constructor's error at compile time.
pub struct Record<'a, O, D, C> {
    key: &'a [u64],
    phase: Phase,
    headers: &'a [u64],
    factors: &'a [u64],
    coefficients: &'a [u128],
    sources: &'a [Source<O, D, C>],
    support: &'a [O],
}

impl<'a, O, D, C> Record<'a, O, D, C> {
    #[expect(
        clippy::too_many_arguments,
        reason = "one checked boundary for the closed generated record"
    )]
    pub const fn new(
        key: &'a [u64],
        phase: Phase,
        headers: &'a [u64],
        factors: &'a [u64],
        coefficients: &'a [u128],
        sources: &'a [Source<O, D, C>],
        support: &'a [O],
    ) -> Result<Self, RecipeError> {
        if key.len() < 2
            || key[0] != 1
            || key[1] == 0
            || sources.len() > 65536
            || coefficients.len() > 65536
        {
            return Err(RecipeError::Structure);
        }
        match phase {
            Phase::Support => {
                if !headers.is_empty() || !coefficients.is_empty() || !sources.is_empty() {
                    return Err(RecipeError::Structure);
                }
            }
            Phase::Input | Phase::Output => {
                if !support.is_empty() {
                    return Err(RecipeError::Structure);
                }
            }
        }
        let mut coefficient = 0;
        while coefficient < coefficients.len() {
            if coefficients[coefficient] >= IMMUTABLE_RECIPE_MODULUS {
                return Err(RecipeError::Coefficient);
            }
            coefficient += 1;
        }
        let mut term = 0;
        let mut cursor = 0;
        while term < headers.len() {
            let header = headers[term];
            let count = header & 15;
            let coefficient = (header >> 4) & 0xffff;
            if header >> 20 != 0 || count > 12 || coefficient >= coefficients.len() as u64 {
                return Err(RecipeError::Structure);
            }
            let mut used = 0;
            while used < count {
                if cursor >= factors.len() {
                    return Err(RecipeError::Structure);
                }
                let word = factors[cursor];
                let mut slot = 0;
                while slot < 4 {
                    let code = (word >> (slot * 16)) & 0xffff;
                    if used + slot < count {
                        if code >= sources.len() as u64 {
                            return Err(RecipeError::Identifier);
                        }
                    } else if code != 0 {
                        return Err(RecipeError::Structure);
                    }
                    slot += 1;
                }
                used += 4;
                cursor += 1;
            }
            term += 1;
        }
        if cursor != factors.len() {
            return Err(RecipeError::Structure);
        }
        Ok(Self {
            key,
            phase,
            headers,
            factors,
            coefficients,
            sources,
            support,
        })
    }
}

pub struct Table<'a, O, D, C> {
    records: &'a [Record<'a, O, D, C>],
}
impl<'a, O, D, C> Table<'a, O, D, C> {
    pub const fn new(records: &'a [Record<'a, O, D, C>]) -> Result<Self, RecipeError> {
        let mut index = 1;
        while index < records.len() {
            let left = &records[index - 1];
            let right = &records[index];
            if left.phase.code() > right.phase.code() {
                return Err(RecipeError::Duplicate);
            }
            if left.phase.code() == right.phase.code() {
                let mut word = 0;
                while word < left.key.len()
                    && word < right.key.len()
                    && left.key[word] == right.key[word]
                {
                    word += 1;
                }
                if word == left.key.len() {
                    if word == right.key.len() {
                        return Err(RecipeError::Duplicate);
                    }
                } else if word == right.key.len() || left.key[word] >= right.key[word] {
                    return Err(RecipeError::Duplicate);
                }
            }
            index += 1;
        }
        Ok(Self { records })
    }
    pub const fn empty() -> Self {
        Self { records: &[] }
    }
}

pub struct Recipe<'a, F, O, D, C> {
    record: &'a Record<'a, O, D, C>,
    field: ImmutableRecipeField<F>,
    unit_first: bool,
}

struct CompareKey<'a> {
    words: std::slice::Iter<'a, u64>,
    order: Ordering,
}
impl KeySink for CompareKey<'_> {
    fn word(&mut self, value: u64) -> bool {
        self.order = self
            .words
            .next()
            .map_or(Ordering::Greater, |expected| value.cmp(expected));
        self.order == Ordering::Equal
    }
}

pub(crate) fn find<'a, F: Ring, O, D, C>(
    table: &'a Table<'a, O, D, C>,
    phase: Phase,
    unit_first: bool,
    mut visit: impl FnMut(&mut dyn KeySink) -> bool,
) -> Result<Option<Recipe<'a, F, O, D, C>>, RecipeError> {
    let Some(field) = F::immutable_recipe_field() else {
        return Ok(None);
    };
    let mut low = 0;
    let mut high = table.records.len();
    while low < high {
        let mid = low + (high - low) / 2;
        let record = &table.records[mid];
        let order = match phase.cmp(&record.phase) {
            Ordering::Equal => {
                let mut key = CompareKey {
                    words: record.key.iter(),
                    order: Ordering::Equal,
                };
                let completed = visit(&mut key);
                match key.order {
                    Ordering::Equal if !completed => return Ok(None),
                    Ordering::Equal if key.words.next().is_some() => Ordering::Less,
                    order => order,
                }
            }
            order => order,
        };
        match order {
            Ordering::Less => high = mid,
            Ordering::Greater => low = mid + 1,
            Ordering::Equal => {
                return Ok(Some(Recipe {
                    record,
                    field,
                    unit_first,
                }))
            }
        }
    }
    Ok(None)
}

enum Factors<'a, O, D, C> {
    Canonical(std::slice::Iter<'a, Source<O, D, C>>),
    Compact {
        words: &'a [u64],
        sources: &'a [Source<O, D, C>],
        count: usize,
        index: usize,
    },
}
impl<'a, O, D, C> Iterator for Factors<'a, O, D, C> {
    type Item = Result<&'a Source<O, D, C>, RecipeError>;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Canonical(values) => values.next().map(Ok),
            Self::Compact {
                words,
                sources,
                count,
                index,
            } => {
                if *index == *count {
                    return None;
                }
                let current = *index;
                *index += 1;
                Some(
                    words
                        .get(current / 4)
                        .ok_or(RecipeError::Structure)
                        .and_then(|word| {
                            let code = u16::try_from((word >> ((current % 4) * 16)) & 0xffff)
                                .map_err(|_| RecipeError::Identifier)?;
                            sources
                                .get(usize::from(code))
                                .ok_or(RecipeError::Identifier)
                        }),
                )
            }
        }
    }
}
struct TermView<'a, F, O, D, C> {
    coefficient: F,
    factors: Factors<'a, O, D, C>,
}
enum Terms<'a, F, O, D, C> {
    Canonical(std::slice::Iter<'a, Term<F, O, D, C>>),
    Compact {
        record: &'a Record<'a, O, D, C>,
        term: usize,
        cursor: usize,
    },
}
impl<'a, F: Ring, O, D, C> Iterator for Terms<'a, F, O, D, C> {
    type Item = Result<TermView<'a, F, O, D, C>, RecipeError>;
    fn next(&mut self) -> Option<Self::Item> {
        match self {
            Self::Canonical(terms) => terms.next().map(|term| {
                Ok(TermView {
                    coefficient: term.coefficient,
                    factors: Factors::Canonical(term.factors.iter()),
                })
            }),
            Self::Compact {
                record,
                term,
                cursor,
            } => {
                let header = *record.headers.get(*term)?;
                *term += 1;
                Some((|| {
                    let count = usize::try_from(header & 15).map_err(|_| RecipeError::Structure)?;
                    let code = u16::try_from((header >> 4) & 0xffff)
                        .map_err(|_| RecipeError::Coefficient)?;
                    let coefficient = F::immutable_recipe_import(
                        *record
                            .coefficients
                            .get(usize::from(code))
                            .ok_or(RecipeError::Coefficient)?,
                    )
                    .ok_or(RecipeError::Coefficient)?;
                    let end = cursor
                        .checked_add(count.div_ceil(4))
                        .ok_or(RecipeError::Structure)?;
                    let words = record
                        .factors
                        .get(*cursor..end)
                        .ok_or(RecipeError::Structure)?;
                    *cursor = end;
                    Ok(TermView {
                        coefficient,
                        factors: Factors::Compact {
                            words,
                            sources: record.sources,
                            count,
                            index: 0,
                        },
                    })
                })())
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum EvaluationError<E> {
    Recipe(RecipeError),
    Resolver(E),
}

fn ordered_fold<F: Ring, O, D, C, E>(
    terms: Terms<'_, F, O, D, C>,
    unit_first: bool,
    mut opening: impl FnMut(&O) -> Result<F, E>,
    mut challenge: impl FnMut(&C) -> Result<F, E>,
    mut derived: impl FnMut(&D) -> Result<F, E>,
) -> Result<F, EvaluationError<E>> {
    let mut result = F::zero();
    for term in terms {
        let mut term = term.map_err(EvaluationError::Recipe)?;
        let mut value = term.coefficient;
        let mut resolve = |factor: Result<&Source<O, D, C>, RecipeError>| {
            match factor.map_err(EvaluationError::Recipe)? {
                Source::Opening(id) => opening(id),
                Source::Challenge(id) => challenge(id),
                Source::Derived(id) => derived(id),
            }
            .map_err(EvaluationError::Resolver)
        };
        if unit_first {
            if let Some(first) = term.factors.next() {
                let factor = resolve(first)?;
                if term.coefficient == F::one() {
                    value = factor;
                } else if term.coefficient == -F::one() {
                    value = -factor;
                } else {
                    value *= factor;
                }
            }
        }
        for factor in term.factors {
            value *= resolve(factor)?;
        }
        result += value;
    }
    Ok(result)
}

impl<F: Ring, O, D, C> Recipe<'_, F, O, D, C> {
    pub fn try_evaluate<E>(
        &self,
        opening: impl FnMut(&O) -> Result<F, E>,
        challenge: impl FnMut(&C) -> Result<F, E>,
        derived: impl FnMut(&D) -> Result<F, E>,
    ) -> Result<F, EvaluationError<E>> {
        let _ = self.field;
        ordered_fold(
            Terms::Compact {
                record: self.record,
                term: 0,
                cursor: 0,
            },
            self.unit_first,
            opening,
            challenge,
            derived,
        )
    }
    pub fn openings(&self) -> BTreeSet<O>
    where
        O: Ord + Clone,
    {
        self.record.support.iter().cloned().collect()
    }
}

/// Used only on the selected field/context path. Unsupported fields and native
/// recording retain the original Expr evaluator, including its operation order.
pub fn evaluate_canonical<F: Ring, O, D, C, E>(
    expression: &Expr<F, O, D, C>,
    _field: ImmutableRecipeField<F>,
    opening: impl FnMut(&O) -> Result<F, E>,
    challenge: impl FnMut(&C) -> Result<F, E>,
    derived: impl FnMut(&D) -> Result<F, E>,
) -> Result<F, EvaluationError<E>> {
    ordered_fold(
        Terms::Canonical(expression.terms.iter()),
        false,
        opening,
        challenge,
        derived,
    )
}

use crate::protocols::composed::ComposedOpeningId;
use crate::protocols::field_inline::{
    FieldInlineChallengeId, FieldInlineDerivedId, FieldInlineOpeningId,
};
use crate::protocols::jolt::{JoltChallengeId, JoltDerivedId, JoltOpeningId};

pub type JoltTable = Table<'static, JoltOpeningId, JoltDerivedId, JoltChallengeId>;
pub type ComposedTable = Table<'static, ComposedOpeningId, JoltDerivedId, JoltChallengeId>;
pub type FieldTable =
    Table<'static, FieldInlineOpeningId, FieldInlineDerivedId, FieldInlineChallengeId>;

#[cfg(target_arch = "riscv64")]
pub(crate) mod compiled {
    use super::{ComposedTable, FieldTable, JoltTable, Phase, Record};
    include!(env!("JOLT_IMMUTABLE_RECIPES_RS"));
}
#[cfg(not(target_arch = "riscv64"))]
pub(crate) mod compiled {
    use super::{ComposedTable, FieldTable, JoltTable};
    pub static JOLT: JoltTable = JoltTable::empty();
    pub static COMPOSED: ComposedTable = ComposedTable::empty();
    pub static FIELD: FieldTable = FieldTable::empty();
}

#[doc(hidden)]
#[macro_export]
macro_rules! immutable_symbolic_recipe {
    ($tag:literal, $family:ident, $native:ident; $($key:tt)+) => {
        #[cfg(feature = "immutable-recipes")]
        fn immutable_recipe<F: ::jolt_field::Ring>(
            &self, phase: $crate::immutable_recipes::Phase,
        ) -> Result<Option<$crate::immutable_recipes::Recipe<'_, F, Self::OpeningId, Self::DerivedId, Self::ChallengeId>>, $crate::immutable_recipes::RecipeError> {
            if F::immutable_recipe_field().is_none() { return Ok(None); }
            if ($tag == 21 && phase != $crate::immutable_recipes::Phase::Input)
                || !$crate::immutable_recipe_admitted!(self; $($key)+) { return Ok(None); }
            $crate::immutable_recipes::find::<F, Self::OpeningId, Self::DerivedId, Self::ChallengeId>(
                &$crate::immutable_recipes::compiled::$family, phase,
                phase == $crate::immutable_recipes::Phase::Output && $tag != 20 && $tag != 26,
                |sink| {
                    sink.word(1) && sink.word($tag)
                        && $crate::immutable_recipe_key!(self, sink; $($key)+)
                },
            )
        }
        #[cfg(all(feature = "immutable-recipes-record", not(target_arch = "riscv64")))]
        fn immutable_record_expression<F: ::jolt_field::Ring>(
            &self, phase: $crate::immutable_recipes::Phase,
            expression: &$crate::Expr<F, Self::OpeningId, Self::DerivedId, Self::ChallengeId>,
        ) {
            if F::immutable_recipe_native_field().is_none() { return; }
            if ($tag == 21 && phase != $crate::immutable_recipes::Phase::Input)
                || !$crate::immutable_recipe_admitted!(self; $($key)+) { return; }
            $crate::immutable_recipes::record::expression(phase, |sink| {
                sink.word(1) && sink.word($tag)
                    && $crate::immutable_recipe_key!(self, sink; $($key)+)
            }, expression, $crate::immutable_recipes::record::Captured::$native);
        }
        #[cfg(all(feature = "immutable-recipes-record", not(target_arch = "riscv64")))]
        fn immutable_record_support<F: ::jolt_field::Ring>(
            &self, openings: &::std::collections::BTreeSet<Self::OpeningId>,
        ) {
            if F::immutable_recipe_native_field().is_none() { return; }
            if $tag == 21 || !$crate::immutable_recipe_admitted!(self; $($key)+) { return; }
            $crate::immutable_recipes::record::support::<F, Self::OpeningId, Self::DerivedId, Self::ChallengeId>(|sink| {
                sink.word(1) && sink.word($tag)
                    && $crate::immutable_recipe_key!(self, sink; $($key)+)
            }, openings, $crate::immutable_recipes::record::Captured::$native);
        }
    };
}

#[cfg(test)]
mod tests;
