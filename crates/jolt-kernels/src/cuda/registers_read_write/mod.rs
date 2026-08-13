#![cfg_attr(
    not(test),
    expect(
        dead_code,
        reason = "implementation target: step 4b (output_claims + backend wiring) is the first non-test caller"
    )
)]

pub(crate) mod rs2_claim;
pub(crate) mod witness;
