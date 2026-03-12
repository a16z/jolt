use crate::sequence_builder::{
    Secp256k1DivQ, Secp256k1DivR, Secp256k1GlvrAdv, Secp256k1MulQ, Secp256k1MulR, Secp256k1SquareQ,
    Secp256k1SquareR,
};

jolt_inlines_common::register_inlines! {
    crate_name: "secp256k1",
    trace_file: "secp256k1_trace.joltinline",
    ops: [
        Secp256k1MulQ,
        Secp256k1SquareQ,
        Secp256k1DivQ,
        Secp256k1MulR,
        Secp256k1SquareR,
        Secp256k1DivR,
        Secp256k1GlvrAdv,
    ],
}
