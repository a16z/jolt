//! Central configuration for the constraint system

/// Configuration for the constraint system variable counts and derived sizes
#[derive(Clone, Debug)]
pub struct ConstraintSystemConfig {
    /// Number of step variables (for packed GT exp: s)
    pub step_vars: usize, // 7
    /// Number of element variables (for Fq12 elements: x)
    pub element_vars: usize, // 4
    /// Number of G1 scalar multiplication variables
    pub g1_vars: usize, // 8
    /// Number of packed constraint variables (s + x)
    pub packed_vars: usize, // 11

    /// Derived power-of-2 sizes
    pub max_steps: usize, // 128 (2^7)
    pub element_size: usize, // 16 (2^4)
    pub g1_size: usize,      // 256 (2^8)
    pub packed_size: usize,  // 2048 (2^11)
}

impl Default for ConstraintSystemConfig {
    fn default() -> Self {
        Self {
            step_vars: 7,
            element_vars: 4,
            g1_vars: 8,
            packed_vars: 11,
            max_steps: 1 << 7,    // 128
            element_size: 1 << 4, // 16
            g1_size: 1 << 8,      // 256
            packed_size: 1 << 11, // 2048
        }
    }
}

impl ConstraintSystemConfig {
    /// Check if the number of constraint variables matches expected values
    pub fn validate_constraint_vars(&self, num_vars: usize) -> Result<(), String> {
        match num_vars {
            4 => Ok(()),  // Element vars
            8 => Ok(()),  // G1 vars
            11 => Ok(()), // Packed vars
            _ => Err(format!(
                "Invalid number of constraint variables: {}. Expected 4, 8, or 11.",
                num_vars
            )),
        }
    }
}

/// Global configuration instance
pub const CONFIG: ConstraintSystemConfig = ConstraintSystemConfig {
    step_vars: 7,
    element_vars: 4,
    g1_vars: 8,
    packed_vars: 11,
    max_steps: 1 << 7,    // 128
    element_size: 1 << 4, // 16
    g1_size: 1 << 8,      // 256
    packed_size: 1 << 11, // 2048
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_config_sizes() {
        let config = ConstraintSystemConfig::default();
        assert_eq!(config.max_steps, 128);
        assert_eq!(config.element_size, 16);
        assert_eq!(config.g1_size, 256);
        assert_eq!(config.packed_size, 2048);
    }

    #[test]
    fn test_validate_constraint_vars() {
        let config = ConstraintSystemConfig::default();
        assert!(config.validate_constraint_vars(4).is_ok());
        assert!(config.validate_constraint_vars(8).is_ok());
        assert!(config.validate_constraint_vars(11).is_ok());
        assert!(config.validate_constraint_vars(7).is_err());
    }
}
