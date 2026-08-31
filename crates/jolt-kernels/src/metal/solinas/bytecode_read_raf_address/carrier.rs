use core::fmt;

pub const ADDRESS_LOG2: u32 = 13;
pub const INNER_LOG2: u32 = 15;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AddressMajorShape {
    log_rows: u32,
    log_addresses: u32,
    inner_log2: u32,
}

impl AddressMajorShape {
    pub fn new(log_rows: u32, log_addresses: u32, inner_log2: u32) -> Result<Self, CarrierError> {
        let shape = Self {
            log_rows,
            log_addresses,
            inner_log2,
        };
        shape.validate()?;
        Ok(shape)
    }

    pub fn production(log_rows: u32) -> Result<Self, CarrierError> {
        Self::new(log_rows, ADDRESS_LOG2, INNER_LOG2)
    }

    pub fn validate(self) -> Result<(), CarrierError> {
        self.validate_exponents()?;
        let _ = checked_mul(
            "address-major cells",
            self.addresses()?,
            self.outer_length()?,
        )?;
        Ok(())
    }

    pub fn rows(self) -> Result<usize, CarrierError> {
        self.validate_exponents()?;
        1usize
            .checked_shl(self.log_rows)
            .ok_or(CarrierError::Overflow("rows"))
    }

    pub fn addresses(self) -> Result<usize, CarrierError> {
        self.validate_exponents()?;
        1usize
            .checked_shl(self.log_addresses)
            .ok_or(CarrierError::Overflow("addresses"))
    }

    pub fn inner_length(self) -> Result<usize, CarrierError> {
        self.validate_exponents()?;
        1usize
            .checked_shl(self.inner_log2)
            .ok_or(CarrierError::Overflow("inner length"))
    }

    pub fn outer_length(self) -> Result<usize, CarrierError> {
        Ok(self.rows()? / self.inner_length()?)
    }

    copy_field_getters! { pub, {
        log_rows: u32,
        log_addresses: u32,
        inner_log2: u32,
    }}

    fn validate_exponents(self) -> Result<(), CarrierError> {
        if self.inner_log2 == 0
            || self.inner_log2 > INNER_LOG2
            || self.log_rows < self.inner_log2
            || self.log_rows >= usize::BITS
            || self.log_addresses == 0
            || self.log_addresses >= usize::BITS
        {
            Err(CarrierError::InvalidShape)
        } else {
            Ok(())
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CarrierError {
    InvalidShape,
    Overflow(&'static str),
}

impl fmt::Display for CarrierError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidShape => f.write_str("invalid address-major shape"),
            Self::Overflow(name) => write!(f, "{name} overflowed"),
        }
    }
}

impl std::error::Error for CarrierError {}

fn checked_mul(name: &'static str, left: usize, right: usize) -> Result<usize, CarrierError> {
    left.checked_mul(right).ok_or(CarrierError::Overflow(name))
}
