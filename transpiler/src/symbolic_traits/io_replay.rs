//! Thread-local overrides for symbolic IO data in the Fiat-Shamir preamble.
//!
//! When transpiling with symbolic IO (universal circuit), the `raw_append_bytes` method
//! in PoseidonAstTranscript must use `MleAst::Var` (witness variables) instead of
//! `MleAst::Const` for IO data (inputs, outputs, panic).
//!
//! This module provides a queue of overrides that `raw_append_bytes` consumes in FIFO order.
//! The caller pushes symbolic elements before calling `verify()`, and each `raw_append_bytes`
//! invocation pops from the front of the queue.

use std::cell::RefCell;
use std::collections::VecDeque;
use zklean_extractor::MleAst;

thread_local! {
    static PENDING_BYTES_OVERRIDES: RefCell<VecDeque<Vec<MleAst>>> = RefCell::new(VecDeque::new());
}

/// Push a batch of symbolic field elements that will replace the next `raw_append_bytes` call.
///
/// Each element corresponds to one 32-byte chunk that would normally be created by
/// `bytes_to_scalar`. The override replaces the concrete constant with a symbolic variable.
pub fn push_bytes_override(elements: Vec<MleAst>) {
    PENDING_BYTES_OVERRIDES.with(|q| q.borrow_mut().push_back(elements));
}

/// Pop the next batch of symbolic override elements (FIFO order).
///
/// Returns `None` if the queue is empty (i.e., no override — use the concrete path).
pub fn pop_bytes_override() -> Option<Vec<MleAst>> {
    PENDING_BYTES_OVERRIDES.with(|q| q.borrow_mut().pop_front())
}
