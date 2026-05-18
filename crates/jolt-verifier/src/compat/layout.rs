//! Legacy proof-format trace polynomial ordering.

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TracePolynomialOrder {
    #[default]
    CycleMajor,
    AddressMajor,
}

impl TracePolynomialOrder {
    pub fn address_cycle_to_index(
        self,
        address: usize,
        cycle: usize,
        num_addresses: usize,
        num_cycles: usize,
    ) -> usize {
        match self {
            Self::CycleMajor => address * num_cycles + cycle,
            Self::AddressMajor => cycle * num_addresses + address,
        }
    }

    pub fn index_to_address_cycle(
        self,
        index: usize,
        num_addresses: usize,
        num_cycles: usize,
    ) -> (usize, usize) {
        match self {
            Self::CycleMajor => (index / num_cycles, index % num_cycles),
            Self::AddressMajor => (index % num_addresses, index / num_addresses),
        }
    }
}

impl From<TracePolynomialOrder> for u8 {
    fn from(order: TracePolynomialOrder) -> Self {
        match order {
            TracePolynomialOrder::CycleMajor => 0,
            TracePolynomialOrder::AddressMajor => 1,
        }
    }
}

impl TryFrom<u8> for TracePolynomialOrder {
    type Error = ();

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::CycleMajor),
            1 => Ok(Self::AddressMajor),
            _ => Err(()),
        }
    }
}
