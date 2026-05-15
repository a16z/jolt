use crate::sequence_builder::{
    Secp256k1DivQ, Secp256k1DivR, Secp256k1GlvrAdv, Secp256k1MulQ, Secp256k1MulR, Secp256k1SquareQ,
    Secp256k1SquareR,
};

jolt_inlines_sdk::register_inlines! {
    trace_file: "secp256k1_trace.joltinline",
    extension: jolt_inlines_sdk::host::InlineExtension::Secp256k1,
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
