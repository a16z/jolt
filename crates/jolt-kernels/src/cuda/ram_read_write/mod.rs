#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "implementation target: step 3 (prepare + backend wiring) is the first non-test caller"
    )
)]

pub(crate) mod witness;
