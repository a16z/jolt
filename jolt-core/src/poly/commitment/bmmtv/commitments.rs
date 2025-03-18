//! This crate defines what is a Double Homomorphic Commitment and
//! exports some implementations of it like Afgho, Identity and Pedersen

use ark_ec::CurveGroup;
use ark_std::rand::Rng;

pub mod afgho16;
pub mod identity;

pub type Error = anyhow::Error;

/// Defines a Doubly homomorphic commitment
///
/// A double homomorphic commitment is a commitment which is homomorphic
/// over messages and over commitment keys
///
/// Something is considered homomorphic if f(x.y) = f(x).f(y)
pub trait Dhc: Clone {
    /// Field in which Messages and Keys are homomorphic
    type Scalar;
    /// Message is the type of the
    type Message;
    /// Commitment key is the output of the setup phase
    ///
    /// Used as parameters for the commitment
    type Param;
    /// Output of the commitment
    type Output: PartialEq + Eq;

    /// Generates a setup for commitments with `size`
    ///
    /// Takes an `Rng` for parameter generator (if needed)
    ///
    /// Output [`Vec<Self::Param>`]
    fn setup<R: Rng>(r: &mut R, size: usize) -> Result<Vec<Self::Param>, Error>;

    /// Commits to some message `msg` taking the parameters `params` from [`Self::setup`]
    /// and outputting [`Self::Output`]
    fn commit(params: &[Self::Param], msg: &[Self::Message]) -> Result<Self::Output, Error>;

    /// Verifies if the commitment is really the output of `msg` and params `k`
    fn verify(
        params: &[Self::Param],
        msg: &[Self::Message],
        commitment: &Self::Output,
    ) -> Result<bool, Error> {
        Ok(Self::commit(params, msg)? == *commitment)
    }
}

/// Helpers for generator commitment keys used by Pedersen and AFGHO16
pub fn random_generators<R: Rng, G: CurveGroup>(rng: &mut R, num: usize) -> Vec<G> {
    (0..num).map(|_| G::rand(rng)).collect()
}
