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

    #[error("tried to create a params derived from params single")]
    DerivedFromSingle,

    #[error("public params is empty")]
    EmptyPublicParams,
    #[error("found a public param with single G1/G2 element")]
    ReduceSinglePublicParam,
    #[error("single param found with non empty step1 or step2")]
    SingleWithNonEmptySteps,
    #[error("found multi param with empty step1 or step2")]
    MultiParamsWithEmptySteps,

    #[error("zr zero")]
    ZrZero,
}

#[derive(Debug)]
pub enum GType {
    G1,
    G2,
}
