pub mod bn254_add;

enum Precompile {
    Bn254_add,
    Bn254_mul,
    Sha256,
}

pub struct PrecompileIO {
    pub input:[u32; 16] // 512 bits
    pub output:[u32; 16] // 512 bits
}
struct PrecompileWitness {
    pub precompile: Precompile,
    pub a0i: [u8; 32],
    pub a1i: [u8; 32],
    pub a2i: [u8; 32],
    pub a3i: [u8; 32],
    pub a4i: [u8; 32],
    pub a5i: [u8; 32],
    pub a6i: [u8; 32],
    pub a7i: [u8; 32],
    pub a8i: [u8; 32],
    pub a9i: [u8; 32],
    pub a10i: [u8; 32],
    pub a11i: [u8; 32],
    pub a12i: [u8; 32],
    pub a13i: [u8; 32],
    pub a14i: [u8; 32],
    pub a15i: [u8; 32],
    pub a0o: [u8; 32],
    pub a1o: [u8; 32],
    pub a2o: [u8; 32],
    pub a3o: [u8; 32],
    pub a4o: [u8; 32],
    pub a5o: [u8; 32],
    pub a6o: [u8; 32],
    pub a7o: [u8; 32],
    pub a8o: [u8; 32],
    pub a9o: [u8; 32],
    pub a10o: [u8; 32],
    pub a11o: [u8; 32],
    pub a12o: [u8; 32],
    pub a13o: [u8; 32],
    pub a14o: [u8; 32],
    pub a15o: [u8; 32],
}
