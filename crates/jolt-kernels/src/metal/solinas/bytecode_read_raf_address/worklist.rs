use core::{fmt, mem::size_of};

pub(crate) const BYTECODE_ADDRESS_WORK_ITEM_ROWS: usize = 4096;
pub(crate) const BYTECODE_ADDRESS_PUSHFORWARD_STAGES: usize = 9;
pub(crate) const BYTECODE_ADDRESS_BASE_STAGES: usize = 5;

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct BytecodeAddressWorkItem {
    pub(crate) address: u16,
    pub(crate) outer: u16,
    pub(crate) start: u16,
    pub(crate) count: u16,
}

const _: [(); 8] = [(); size_of::<BytecodeAddressWorkItem>()];

impl BytecodeAddressWorkItem {
    pub(crate) fn new(
        address: usize,
        outer: usize,
        start: usize,
        count: usize,
        outer_rows: usize,
    ) -> Result<Self, BytecodeAddressWorklistError> {
        if count == 0
            || count > BYTECODE_ADDRESS_WORK_ITEM_ROWS
            || start.checked_add(count).is_none_or(|end| end > outer_rows)
        {
            return Err(BytecodeAddressWorklistError::InvalidWorkItem);
        }
        Ok(Self {
            address: u16::try_from(address).map_err(|_| {
                BytecodeAddressWorklistError::UnsupportedAddresses(address.saturating_add(1))
            })?,
            outer: u16::try_from(outer).map_err(|_| {
                BytecodeAddressWorklistError::UnsupportedOuters(outer.saturating_add(1))
            })?,
            start: u16::try_from(start)
                .map_err(|_| BytecodeAddressWorklistError::InvalidWorkItem)?,
            count: u16::try_from(count)
                .map_err(|_| BytecodeAddressWorklistError::InvalidWorkItem)?,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum BytecodeAddressWorklistError {
    UnsupportedAddresses(usize),
    UnsupportedOuters(usize),
    InvalidWorkItem,
}

impl fmt::Display for BytecodeAddressWorklistError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedAddresses(addresses) => {
                write!(f, "{addresses} addresses do not fit the u16 work-item ABI")
            }
            Self::UnsupportedOuters(outers) => {
                write!(f, "{outers} outers do not fit the u16 work-item ABI")
            }
            Self::InvalidWorkItem => f.write_str("invalid sparse bytecode work-item layout"),
        }
    }
}

impl std::error::Error for BytecodeAddressWorklistError {}
