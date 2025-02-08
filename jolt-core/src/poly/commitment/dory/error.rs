#[derive(thiserror::Error, Debug)]
pub enum Error {
    #[error("length mismatch")]
    LengthMismatch,
    #[error("empty vectors: {0:?}")]
    EmptyVector(GType),
    #[error("could not invert d")]
    CouldntInvertD,

    #[error("serialization error {0}")]
    Serialization(#[from] ark_serialize::SerializationError),

    #[error(
        "recursive public parameters should be twice as the public parameters it is derived from"
    )]
    LengthNotTwice,
    #[error("reduce params not initialized")]
    ReduceParamsNotInitialized,
    #[error("public params is empty")]
    EmptyPublicParams,

    #[error("zr zero")]
    ZrZero,
}

#[derive(Debug)]
pub enum GType {
    G1,
    G2,
}
