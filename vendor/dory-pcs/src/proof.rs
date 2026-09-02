//! Dory proof structure
//!
//! A Dory proof consists of:
//! - VMV message (PCS transform)
//! - Multiple rounds of reduce messages (log n rounds)
//! - Final scalar product message (transparent) or Σ-proofs (ZK)

use crate::error::DoryError;
use crate::messages::*;
use crate::primitives::arithmetic::Group;
use std::marker::PhantomData;

/// A complete Dory evaluation proof
///
/// The proof demonstrates that a committed polynomial evaluates to a specific value
/// at a given point. It consists of messages from the interactive protocol made
/// non-interactive via Fiat-Shamir.
///
/// The proof includes the matrix dimensions (nu, sigma) used during proof generation,
/// which the verifier uses to ensure consistency with the evaluation point.
#[derive(Clone, Debug, PartialEq)]
#[allow(missing_docs)]
pub struct DoryProof<G1: Group, G2, GT> {
    /// Vector-Matrix-Vector message for PCS transformation
    pub vmv_message: VMVMessage<G1, GT>,

    /// First reduce messages for each round (nu rounds total)
    pub first_messages: Vec<FirstReduceMessage<G1, G2, GT>>,

    /// Second reduce messages for each round (nu rounds total)
    pub second_messages: Vec<SecondReduceMessage<G1, G2, GT>>,

    /// Final scalar product message revealing the folded witness.
    ///
    /// `Some` in transparent mode. `None` in ZK mode, where revealing the
    /// folded witness would break hiding: the scalar-product Σ-proof
    /// (`scalar_product_proof`) replaces it.
    pub final_message: Option<ScalarProductMessage<G1, G2>>,

    /// Log₂ of number of rows in the coefficient matrix
    pub nu: usize,

    /// Log₂ of number of columns in the coefficient matrix
    pub sigma: usize,

    /// Blinded E₂ element for zero-knowledge proofs
    #[cfg(feature = "zk")]
    pub e2: Option<G2>,
    /// Pedersen commitment to the blinding vector y
    #[cfg(feature = "zk")]
    pub y_com: Option<G1>,
    /// Σ₁ proof: E₂ and y_com commit to the same y
    #[cfg(feature = "zk")]
    pub sigma1_proof: Option<Sigma1Proof<G1, G2, G1::Scalar>>,
    /// Σ₂ proof: consistency of E₁ with D₂
    #[cfg(feature = "zk")]
    pub sigma2_proof: Option<Sigma2Proof<G1::Scalar, GT>>,
    /// ZK scalar product proof: (C, D₁, D₂) consistency with blinded vectors
    #[cfg(feature = "zk")]
    pub scalar_product_proof: Option<ScalarProductProof<G1, G2, G1::Scalar, GT>>,
}

/// A [`DoryProof`] classified by mode, carrying references to the fields that
/// mode guarantees (see [`DoryProof::mode`]).
#[derive(Clone, Copy)]
pub enum ProofMode<'a, G1: Group, G2, GT> {
    /// Transparent proof: reveals the folded witness as the clear final
    /// message and carries no ZK fields. (The phantom borrow ties down `GT`,
    /// which only the ZK variant otherwise uses, keeping the enum's generics —
    /// and auto traits — identical across feature flags.)
    Transparent(&'a ScalarProductMessage<G1, G2>, PhantomData<&'a GT>),
    /// ZK proof: carries every blinding field and Σ-proof
    #[cfg(feature = "zk")]
    Zk {
        /// Blinded E₂ from the VMV message.
        e2: &'a G2,
        /// Pedersen commitment to the claimed evaluation.
        y_com: &'a G1,
        /// Σ₁ proof: E₂ and y_com commit to the same y.
        sigma1: &'a Sigma1Proof<G1, G2, G1::Scalar>,
        /// Σ₂ proof: VMV constraint (batched into the final check).
        sigma2: &'a Sigma2Proof<G1::Scalar, GT>,
        /// Scalar-product Σ-proof over the folded statement.
        scalar_product: &'a ScalarProductProof<G1, G2, G1::Scalar, GT>,
    },
}

impl<G1: Group, G2, GT> DoryProof<G1, G2, GT> {
    /// Classify this proof by shape, rejecting mix-and-match proofs.
    ///
    /// A proof must be *fully* transparent — clear final message present, no
    /// ZK fields — or *fully* ZK — every ZK field present, no clear final
    /// message. Anything in between is malformed: extra fields would either be
    /// ignored by verification (making the proof bytes malleable — two
    /// distinct serialized proofs for one statement) or reveal data a ZK proof
    /// must hide.
    ///
    /// This is the single enforcement point of that invariant;
    /// `verify_evaluation_proof` calls it before reading any optional field,
    /// and the returned [`ProofMode`] hands out references to exactly the
    /// fields the shape guarantees.
    ///
    /// # Errors
    /// Returns [`DoryError::InvalidProof`] for any mixed or incomplete shape.
    pub fn mode(&self) -> Result<ProofMode<'_, G1, G2, GT>, DoryError> {
        #[cfg(feature = "zk")]
        {
            if let (Some(e2), Some(y_com), Some(sigma1), Some(sigma2), Some(scalar_product)) = (
                &self.e2,
                &self.y_com,
                &self.sigma1_proof,
                &self.sigma2_proof,
                &self.scalar_product_proof,
            ) {
                return if self.final_message.is_none() {
                    Ok(ProofMode::Zk {
                        e2,
                        y_com,
                        sigma1,
                        sigma2,
                        scalar_product,
                    })
                } else {
                    Err(DoryError::InvalidProof)
                };
            }
            // Not fully ZK: every ZK field must then be absent.
            if self.e2.is_some()
                || self.y_com.is_some()
                || self.sigma1_proof.is_some()
                || self.sigma2_proof.is_some()
                || self.scalar_product_proof.is_some()
            {
                return Err(DoryError::InvalidProof);
            }
        }
        self.final_message
            .as_ref()
            .map(|msg| ProofMode::Transparent(msg, PhantomData))
            .ok_or(DoryError::InvalidProof)
    }
}
