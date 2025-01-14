pub mod bn254_add;

enum Precompile {
    Bn254_add,
}

// trait to deserialize the raw input bytes and serialize the outputs
