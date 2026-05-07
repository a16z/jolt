//! RV64M multiply/divide instructions.

pub mod div;
pub mod divu;
pub mod divuw;
pub mod divw;
pub mod mul;
pub mod mulh;
pub mod mulhsu;
pub mod mulhu;
pub mod mulw;
pub mod rem;
pub mod remu;
pub mod remuw;
pub mod remw;

pub use div::Div;
pub use divu::DivU;
pub use divuw::DivUW;
pub use divw::DivW;
pub use mul::Mul;
pub use mulh::MulH;
pub use mulhsu::MulHSU;
pub use mulhu::MulHU;
pub use mulw::MulW;
pub use rem::Rem;
pub use remu::RemU;
pub use remuw::RemUW;
pub use remw::RemW;
