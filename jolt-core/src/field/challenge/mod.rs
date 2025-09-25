pub mod mont_u128;
pub mod trivial;

// Re-export the main types for convenient access:
//pub use advanced::AdvancedChallenge;
pub use mont_u128::MontU128Challenge;
pub use trivial::TrivialChallenge;
