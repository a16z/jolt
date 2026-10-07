//! Test-only Fiat-Shamir verifier scopes, draw roles, and relation catalog.

use std::any::Any;
use std::cell::{Cell, RefCell};

use jolt_claims::protocols::jolt::{JoltExpr, JoltRelationId};
use jolt_claims::SymbolicSumcheck;
use jolt_field::JoltField;

use crate::stages::relations::{ConcreteSumcheck, DrawRole};

/// Verifier region active when a transcript challenge is drawn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub enum FsScope {
    /// Input validation and transcript preamble.
    #[default]
    Preamble,
    /// Proof and preprocessing commitments.
    Commitments,
    /// Stage 1.
    Stage1,
    /// Stage 2.
    Stage2,
    /// Stage 3.
    Stage3,
    /// Stage 4.
    Stage4,
    /// Stage 5.
    Stage5,
    /// Stage 6 address phase.
    Stage6a,
    /// Stage 6 cycle phase.
    Stage6b,
    /// Stage 7.
    Stage7,
    /// Final opening checks.
    Stage8,
    /// BlindFold verification.
    BlindFold,
    /// The byte link between stage 7 and the final opening.
    ByteLink,
}

thread_local! {
    static CURRENT_SCOPE: Cell<FsScope> = const { Cell::new(FsScope::Preamble) };
    static CURRENT_ROLE: Cell<Option<DrawRole>> = const { Cell::new(None) };
    static BATCH_MEMBERS: RefCell<Option<Vec<Box<dyn Any>>>> = const { RefCell::new(None) };
}

/// Restores the previous verifier scope on drop.
pub struct FsScopeGuard {
    previous: FsScope,
}

impl Drop for FsScopeGuard {
    fn drop(&mut self) {
        CURRENT_SCOPE.set(self.previous);
    }
}

/// Marks subsequent transcript operations as belonging to `scope`.
#[must_use]
pub fn enter(scope: FsScope) -> FsScopeGuard {
    let previous = CURRENT_SCOPE.replace(scope);
    FsScopeGuard { previous }
}

/// Returns the verifier scope active on this thread.
pub fn current() -> FsScope {
    CURRENT_SCOPE.get()
}

/// Restores the previous draw role on drop.
pub struct DrawRoleGuard {
    previous: Option<DrawRole>,
}

impl Drop for DrawRoleGuard {
    fn drop(&mut self) {
        CURRENT_ROLE.set(self.previous);
    }
}

/// Marks the draws made until the guard drops as belonging to `role`.
#[must_use]
pub fn enter_role(role: DrawRole) -> DrawRoleGuard {
    let previous = CURRENT_ROLE.replace(Some(role));
    DrawRoleGuard { previous }
}

/// The role of a draw made now; `None` outside every member, coefficient,
/// and batch-round scope (stage taus, uni-skip, opening draws), where draws
/// are ordered by transcript position alone.
pub fn current_role() -> Option<DrawRole> {
    CURRENT_ROLE.get()
}

/// One batch member as a batch head instantiated it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BatchMember<F> {
    pub batch: &'static str,
    pub member: &'static str,
    pub relation: JoltRelationId,
    pub rounds: usize,
    pub degree: usize,
    pub point_offset: usize,
    pub input: JoltExpr<F>,
    pub output: JoltExpr<F>,
}

pub(crate) fn record_batch_member<F: JoltField, I: ConcreteSumcheck<F>>(
    batch: &'static str,
    member: &'static str,
    instance: &I,
    point_offset: usize,
) {
    BATCH_MEMBERS.with_borrow_mut(|members| {
        if let Some(members) = members {
            members.push(Box::new(BatchMember::<F> {
                batch,
                member,
                relation: instance.id(),
                rounds: instance.rounds(),
                degree: instance.degree(),
                point_offset,
                input: instance.symbolic().input_expression(),
                output: instance.symbolic().output_expression(),
            }));
        }
    });
}

/// Runs `f`, returning every batch member instantiated meanwhile, in order.
///
/// # Panics
///
/// On a nested session or a member recorded over another field.
#[expect(
    clippy::expect_used,
    reason = "an audit session over the wrong field is a test bug"
)]
pub fn record_batch_members<F: JoltField, R>(f: impl FnOnce() -> R) -> (R, Vec<BatchMember<F>>) {
    BATCH_MEMBERS.with_borrow_mut(|members| {
        assert!(
            members.is_none(),
            "nested batch-member sessions are unsupported"
        );
        *members = Some(Vec::new());
    });
    let output = f();
    let members = BATCH_MEMBERS
        .with_borrow_mut(Option::take)
        .unwrap_or_default()
        .into_iter()
        .map(|member| {
            *member
                .downcast::<BatchMember<F>>()
                .expect("batch member recorded over another field")
        })
        .collect();
    (output, members)
}
