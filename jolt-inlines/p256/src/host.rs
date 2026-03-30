use crate::sequence_builder::{
    P256DivQ, P256DivR, P256FakeGlvAdv, P256MulQ, P256MulR, P256SquareQ, P256SquareR,
};

jolt_inlines_sdk::register_inlines! {
    trace_file: "p256_trace.joltinline",
    ops: [
        P256MulQ,
        P256SquareQ,
        P256DivQ,
        P256MulR,
        P256SquareR,
        P256DivR,
        P256FakeGlvAdv,
    ],
}
