//! FIFO queue that intercepts `raw_append_bytes` in the symbolic Poseidon transcript.
//!
//! Problem: `fiat_shamir_preamble` calls `transcript.append_bytes(b"inputs", &bytes)`
//! which hashes the concrete IO bytes into the transcript. In the generated circuit,
//! we need those bytes to be witness variables, not hardcoded constants.
//!
//! Solution: before running `verifier.verify()`, `symbolize_io_device()` pushes
//! symbolic variables into this FIFO (one per 32-byte chunk of inputs + outputs).
//! Our modified `PoseidonAstTranscript::raw_append_bytes` checks this FIFO on each
//! chunk — if there's an override, it uses the symbolic variable instead of the
//! concrete bytes.
//!
//! Important: `raw_append_label_with_len` is overridden separately to bypass this
//! FIFO, because labels are always concrete and must not consume IO overrides.

use std::cell::RefCell;
use std::collections::VecDeque;
use zklean_extractor::mle_ast::MleAst;

thread_local! {
    static PENDING_BYTES_OVERRIDES: RefCell<VecDeque<MleAst>> = const { RefCell::new(VecDeque::new()) };
}

pub fn push_bytes_override(val: MleAst) {
    PENDING_BYTES_OVERRIDES.with(|cell| cell.borrow_mut().push_back(val));
}

pub fn pop_bytes_override() -> Option<MleAst> {
    PENDING_BYTES_OVERRIDES.with(|cell| cell.borrow_mut().pop_front())
}
