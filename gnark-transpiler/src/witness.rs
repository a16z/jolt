//! Witness generation for Gnark circuits
//!
//! Converts Stage1CircuitData from jolt-core into JSON format
//! that can be loaded by the Gnark circuit.

use serde::{Deserialize, Serialize};

/// Witness data for Stage 1 circuit in JSON-serializable format.
///
/// Field elements are serialized as decimal strings to preserve precision.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Stage1Witness {
    /// tau challenges (variable indices 0..n)
    pub tau: Vec<String>,
    /// r0 challenge (variable index n)
    pub r0: String,
    /// sumcheck challenges (variable indices n+1..n+1+m)
    pub sumcheck_challenges: Vec<String>,
    /// uni-skip polynomial coefficients
    pub uni_skip_poly_coeffs: Vec<String>,
    /// sumcheck round polynomials (flattened)
    pub sumcheck_round_polys: Vec<Vec<String>>,
    /// expected final claim
    pub expected_final_claim: String,
}

impl Stage1Witness {
    /// Create witness manually from string values
    pub fn new(
        tau: Vec<String>,
        r0: String,
        sumcheck_challenges: Vec<String>,
        uni_skip_poly_coeffs: Vec<String>,
        sumcheck_round_polys: Vec<Vec<String>>,
        expected_final_claim: String,
    ) -> Self {
        Self {
            tau,
            r0,
            sumcheck_challenges,
            uni_skip_poly_coeffs,
            sumcheck_round_polys,
            expected_final_claim,
        }
    }

    /// Serialize to JSON string
    pub fn to_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }

    /// Deserialize from JSON string
    pub fn from_json(json: &str) -> Result<Self, serde_json::Error> {
        serde_json::from_str(json)
    }

    /// Get all values as a flat vector of strings (in variable index order)
    ///
    /// Order: tau[0..n], r0, sumcheck_challenges[..], uni_skip_poly_coeffs[..],
    ///        sumcheck_round_polys flattened, expected_final_claim
    pub fn to_flat_values(&self) -> Vec<String> {
        let mut values = Vec::new();
        values.extend(self.tau.iter().cloned());
        values.push(self.r0.clone());
        values.extend(self.sumcheck_challenges.iter().cloned());
        values.extend(self.uni_skip_poly_coeffs.iter().cloned());
        for poly in &self.sumcheck_round_polys {
            values.extend(poly.iter().cloned());
        }
        values.push(self.expected_final_claim.clone());
        values
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_witness_serialization() {
        let witness = Stage1Witness {
            tau: vec!["123".to_string(), "456".to_string()],
            r0: "789".to_string(),
            sumcheck_challenges: vec!["111".to_string()],
            uni_skip_poly_coeffs: vec!["222".to_string(), "333".to_string()],
            sumcheck_round_polys: vec![vec!["444".to_string()]],
            expected_final_claim: "555".to_string(),
        };

        let json = witness.to_json().unwrap();
        println!("Serialized witness:\n{}", json);

        let deserialized = Stage1Witness::from_json(&json).unwrap();
        assert_eq!(deserialized.tau, witness.tau);
        assert_eq!(deserialized.r0, witness.r0);
    }

    #[test]
    fn test_flat_values() {
        let witness = Stage1Witness {
            tau: vec!["1".to_string(), "2".to_string()],
            r0: "3".to_string(),
            sumcheck_challenges: vec!["4".to_string()],
            uni_skip_poly_coeffs: vec!["5".to_string(), "6".to_string()],
            sumcheck_round_polys: vec![vec!["7".to_string(), "8".to_string()]],
            expected_final_claim: "9".to_string(),
        };

        let flat = witness.to_flat_values();
        assert_eq!(flat, vec!["1", "2", "3", "4", "5", "6", "7", "8", "9"]);
    }
}
