
pub trait JoltField:
    Sized
    + std::ops::Mul<Output = Self>
    + std::ops::Add<Output = Self>
    + std::ops::AddAssign<Self>
    + std::ops::Sub<Output = Self>
    + Copy
    + Sync
    + Send
{
    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self;
}

impl JoltField for ark_bn254::Fr {
    fn random<R: rand_core::RngCore>(rng: &mut R) -> Self {
        todo!("stuff");
    }
}
