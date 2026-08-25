//! Variable binding order for sumcheck protocols.

use serde::{Deserialize, Serialize};

/// The order in which polynomial variables are bound during sumcheck.
///
/// - **LowToHigh**: Bind from the least-significant bit (index `n-1` in the
///   evaluation table) upward. This is the default for most sumcheck instances.
/// - **HighToLow**: Bind from the most-significant bit (index `0`) downward.
///   Used by Spartan's outer sumcheck.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum BindingOrder {
    #[default]
    LowToHigh,
    HighToLow,
}
