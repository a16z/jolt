#[macro_use]
pub mod macros;
pub mod mont_ark_u128;
pub mod mont_ark_u254;

// Re-export the main types for convenient access:
pub use mont_ark_u128::MontU128Challenge;
pub use mont_ark_u254::Mont254BitChallenge;
