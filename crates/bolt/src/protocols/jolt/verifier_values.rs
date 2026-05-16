use std::collections::{BTreeMap, BTreeSet};

use crate::emit::rust::EmitError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifierScalarSourceKind {
    OpeningInput,
    FieldConstant,
    TranscriptScalar,
    FieldExpr,
    ScalarExpr,
    StructuredPolynomialEval,
    RelationOutputLocal,
    SumcheckEval,
    OutputEvalFamily,
    OutputProductFamily,
    OutputFunctionFamily,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VerifierScalarValueKind {
    OpeningInput,
    FieldConstant,
    TranscriptScalar,
    FieldExpr,
    ScalarExpr,
    StructuredPolynomialEval,
    RelationOutputLocal,
    SumcheckEval,
    OutputEvalFamily,
    OutputProductFamily,
    OutputFunctionFamily,
}

impl VerifierScalarValueKind {
    fn source_kind(self) -> VerifierScalarSourceKind {
        match self {
            Self::OpeningInput => VerifierScalarSourceKind::OpeningInput,
            Self::FieldConstant => VerifierScalarSourceKind::FieldConstant,
            Self::TranscriptScalar => VerifierScalarSourceKind::TranscriptScalar,
            Self::FieldExpr => VerifierScalarSourceKind::FieldExpr,
            Self::ScalarExpr => VerifierScalarSourceKind::ScalarExpr,
            Self::StructuredPolynomialEval => VerifierScalarSourceKind::StructuredPolynomialEval,
            Self::RelationOutputLocal => VerifierScalarSourceKind::RelationOutputLocal,
            Self::SumcheckEval => VerifierScalarSourceKind::SumcheckEval,
            Self::OutputEvalFamily => VerifierScalarSourceKind::OutputEvalFamily,
            Self::OutputProductFamily => VerifierScalarSourceKind::OutputProductFamily,
            Self::OutputFunctionFamily => VerifierScalarSourceKind::OutputFunctionFamily,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct VerifierScalarValuePlan {
    pub(crate) symbol: String,
    pub(crate) kind: VerifierScalarValueKind,
}

impl VerifierScalarValuePlan {
    pub(crate) fn new(symbol: impl Into<String>, kind: VerifierScalarValueKind) -> Self {
        Self {
            symbol: symbol.into(),
            kind,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifierScalarValueRef {
    symbol: String,
}

impl VerifierScalarValueRef {
    pub fn new(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
        }
    }

    pub fn symbol(&self) -> &str {
        &self.symbol
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct VerifierScalarValueSet {
    symbols: BTreeMap<String, VerifierScalarValueKind>,
    conflicts: Vec<VerifierSourceConflict<VerifierScalarValueKind>>,
}

impl VerifierScalarValueSet {
    pub(crate) fn insert(&mut self, symbol: &str, kind: VerifierScalarValueKind) {
        match self.symbols.entry(symbol.to_owned()) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                let _entry = entry.insert(kind);
            }
            std::collections::btree_map::Entry::Occupied(entry) => {
                let existing = *entry.get();
                if existing != kind {
                    self.conflicts.push(VerifierSourceConflict {
                        symbol: symbol.to_owned(),
                        existing,
                        incoming: kind,
                    });
                }
            }
        }
    }

    pub(crate) fn insert_plan(&mut self, plan: &VerifierScalarValuePlan) {
        self.insert(&plan.symbol, plan.kind);
    }

    pub(crate) fn extend_plans<'a>(
        &mut self,
        plans: impl IntoIterator<Item = &'a VerifierScalarValuePlan>,
    ) {
        for plan in plans {
            self.insert_plan(plan);
        }
    }

    pub(crate) fn contains_ref(&self, value_ref: &VerifierScalarValueRef) -> bool {
        self.symbols.contains_key(value_ref.symbol())
    }

    pub(crate) fn contains_plan(&self, plan: &VerifierScalarValuePlan) -> bool {
        self.symbols
            .get(&plan.symbol)
            .is_some_and(|kind| *kind == plan.kind)
    }

    pub(crate) fn source_set(&self) -> VerifierScalarSourceSet {
        let mut sources = VerifierScalarSourceSet::default();
        for (symbol, kind) in &self.symbols {
            sources.insert(symbol, kind.source_kind());
        }
        sources
    }

    pub(crate) fn verify_no_conflicts(&self, stage: &str) -> Result<(), EmitError> {
        let Some(conflict) = self.conflicts.first() else {
            return Ok(());
        };
        Err(conflicting_source_error(stage, "scalar value", conflict))
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VerifierScalarSourceSet {
    symbols: BTreeMap<String, VerifierScalarSourceKind>,
    conflicts: Vec<VerifierSourceConflict<VerifierScalarSourceKind>>,
}

impl VerifierScalarSourceSet {
    pub fn insert(&mut self, symbol: &str, kind: VerifierScalarSourceKind) {
        match self.symbols.entry(symbol.to_owned()) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                let _entry = entry.insert(kind);
            }
            std::collections::btree_map::Entry::Occupied(entry) => {
                let existing = *entry.get();
                if existing != kind {
                    self.conflicts.push(VerifierSourceConflict {
                        symbol: symbol.to_owned(),
                        existing,
                        incoming: kind,
                    });
                }
            }
        }
    }

    pub fn extend<'a>(
        &mut self,
        symbols: impl IntoIterator<Item = &'a String>,
        kind: VerifierScalarSourceKind,
    ) {
        for symbol in symbols {
            self.insert(symbol, kind);
        }
    }

    pub fn contains(&self, symbol: &str) -> bool {
        self.symbols.contains_key(symbol)
    }

    pub(crate) fn verify_no_conflicts(&self, stage: &str) -> Result<(), EmitError> {
        let Some(conflict) = self.conflicts.first() else {
            return Ok(());
        };
        Err(conflicting_source_error(stage, "scalar", conflict))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifierFieldVectorSourceKind {
    IndexedEvalFamily,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum VerifierFieldVectorValueKind {
    IndexedEvalFamily,
}

impl VerifierFieldVectorValueKind {
    fn source_kind(self) -> VerifierFieldVectorSourceKind {
        match self {
            Self::IndexedEvalFamily => VerifierFieldVectorSourceKind::IndexedEvalFamily,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifierFieldVectorValueRef {
    symbol: String,
}

impl VerifierFieldVectorValueRef {
    pub fn new(symbol: impl Into<String>) -> Self {
        Self {
            symbol: symbol.into(),
        }
    }

    pub fn symbol(&self) -> &str {
        &self.symbol
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct VerifierFieldVectorValueSet {
    symbols: BTreeSet<String>,
}

impl VerifierFieldVectorValueSet {
    pub(crate) fn insert(&mut self, symbol: &str, _kind: VerifierFieldVectorValueKind) {
        let _inserted = self.symbols.insert(symbol.to_owned());
    }

    pub(crate) fn contains_ref(&self, value_ref: &VerifierFieldVectorValueRef) -> bool {
        self.symbols.contains(value_ref.symbol())
    }

    pub(crate) fn source_set(&self) -> VerifierFieldVectorSourceSet {
        let mut sources = VerifierFieldVectorSourceSet::default();
        for symbol in &self.symbols {
            sources.insert(
                symbol,
                VerifierFieldVectorValueKind::IndexedEvalFamily.source_kind(),
            );
        }
        sources
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VerifierFieldVectorSourceSet {
    symbols: BTreeSet<String>,
}

impl VerifierFieldVectorSourceSet {
    pub fn insert(&mut self, symbol: &str, _kind: VerifierFieldVectorSourceKind) {
        let _inserted = self.symbols.insert(symbol.to_owned());
    }

    pub fn contains(&self, symbol: &str) -> bool {
        self.symbols.contains(symbol)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VerifierPointSourceKind {
    OpeningInput,
    SumcheckInstance,
    PointExpr,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct VerifierPointSourceSet {
    symbols: BTreeMap<String, VerifierPointSourceKind>,
    conflicts: Vec<VerifierSourceConflict<VerifierPointSourceKind>>,
}

impl VerifierPointSourceSet {
    pub fn insert(&mut self, symbol: &str, kind: VerifierPointSourceKind) {
        match self.symbols.entry(symbol.to_owned()) {
            std::collections::btree_map::Entry::Vacant(entry) => {
                let _entry = entry.insert(kind);
            }
            std::collections::btree_map::Entry::Occupied(entry) => {
                let existing = *entry.get();
                if existing != kind {
                    self.conflicts.push(VerifierSourceConflict {
                        symbol: symbol.to_owned(),
                        existing,
                        incoming: kind,
                    });
                }
            }
        }
    }

    pub fn extend<'a>(
        &mut self,
        symbols: impl IntoIterator<Item = &'a String>,
        kind: VerifierPointSourceKind,
    ) {
        for symbol in symbols {
            self.insert(symbol, kind);
        }
    }

    pub fn contains(&self, symbol: &str) -> bool {
        self.symbols.contains_key(symbol)
    }

    pub(crate) fn verify_no_conflicts(&self, stage: &str) -> Result<(), EmitError> {
        let Some(conflict) = self.conflicts.first() else {
            return Ok(());
        };
        Err(conflicting_source_error(stage, "point", conflict))
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct VerifierSourceConflict<K> {
    symbol: String,
    existing: K,
    incoming: K,
}

fn conflicting_source_error<K>(
    stage: &str,
    value_kind: &str,
    conflict: &VerifierSourceConflict<K>,
) -> EmitError
where
    K: std::fmt::Debug,
{
    EmitError::new(format!(
        "{stage} {value_kind} source @{} has conflicting kinds {:?} and {:?}",
        conflict.symbol, conflict.existing, conflict.incoming
    ))
}
