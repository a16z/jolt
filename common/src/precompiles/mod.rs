pub mod bn254_add;

pub enum Precompile {
    Bn254Add,
}

impl Precompile {
    pub fn from_u64(value: u64) -> Option<Self> {
        match value {
            0 => None,
            1 => Some(Precompile::Bn254_add),
            _ => None,
        }
    }

    pub fn execute(&self, inputs: &[u32; 16]) -> [u32; 16] {
        match self {
            Precompile::Bn254Add => bn254_add::bn254_add(inputs),
        }
    }
}
// trait to deserialize the raw input bytes and serialize the outputs
