use std::num::NonZeroUsize;

use super::super::{JoltCommittedPolynomial, JoltOpeningId, JoltStageId};
use super::dimensions::JoltFormulaDimensionsError;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum JoltRaPolynomial {
    Instruction(usize),
    Bytecode(usize),
    Ram(usize),
}

impl JoltRaPolynomial {
    pub fn committed(self) -> JoltCommittedPolynomial {
        match self {
            Self::Instruction(index) => JoltCommittedPolynomial::InstructionRa(index),
            Self::Bytecode(index) => JoltCommittedPolynomial::BytecodeRa(index),
            Self::Ram(index) => JoltCommittedPolynomial::RamRa(index),
        }
    }

    pub fn opening(self, stage: JoltStageId) -> JoltOpeningId {
        JoltOpeningId::committed(self.committed(), stage)
    }
}

/// Canonical Jolt RA polynomial ordering used by cross-family reductions.
///
/// Formulas using this layout iterate `InstructionRa`, then `BytecodeRa`, then `RamRa`.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct JoltRaPolynomialLayout {
    instruction: usize,
    bytecode: usize,
    ram: usize,
    total: NonZeroUsize,
}

impl JoltRaPolynomialLayout {
    pub fn new(
        instruction: usize,
        bytecode: usize,
        ram: usize,
    ) -> Result<Self, JoltFormulaDimensionsError> {
        let total = instruction
            .checked_add(bytecode)
            .and_then(|total| total.checked_add(ram))
            .ok_or_else(|| JoltFormulaDimensionsError::overflow("Jolt RA polynomial count"))?;
        Ok(Self {
            instruction,
            bytecode,
            ram,
            total: NonZeroUsize::new(total)
                .ok_or_else(|| JoltFormulaDimensionsError::zero("Jolt RA polynomial count"))?,
        })
    }

    pub fn instruction(self) -> usize {
        self.instruction
    }

    pub fn bytecode(self) -> usize {
        self.bytecode
    }

    pub fn ram(self) -> usize {
        self.ram
    }

    pub fn total(self) -> usize {
        self.total.get()
    }

    pub fn polynomials(self) -> impl Iterator<Item = JoltRaPolynomial> {
        (0..self.instruction)
            .map(JoltRaPolynomial::Instruction)
            .chain((0..self.bytecode).map(JoltRaPolynomial::Bytecode))
            .chain((0..self.ram).map(JoltRaPolynomial::Ram))
    }

    pub fn committed_polynomials(self) -> impl Iterator<Item = JoltCommittedPolynomial> {
        self.polynomials().map(JoltRaPolynomial::committed)
    }

    pub fn openings(self, stage: JoltStageId) -> impl Iterator<Item = JoltOpeningId> {
        self.polynomials()
            .map(move |polynomial| polynomial.opening(stage))
    }
}

impl TryFrom<(usize, usize, usize)> for JoltRaPolynomialLayout {
    type Error = JoltFormulaDimensionsError;

    fn try_from((instruction, bytecode, ram): (usize, usize, usize)) -> Result<Self, Self::Error> {
        Self::new(instruction, bytecode, ram)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_layout() {
        assert_eq!(
            JoltRaPolynomialLayout::new(0, 0, 0),
            Err(JoltFormulaDimensionsError::zero("Jolt RA polynomial count"))
        );
    }

    #[test]
    fn rejects_overflowing_layout() {
        assert_eq!(
            JoltRaPolynomialLayout::new(usize::MAX, 1, 0),
            Err(JoltFormulaDimensionsError::overflow(
                "Jolt RA polynomial count"
            ))
        );
        assert_eq!(
            JoltRaPolynomialLayout::new(usize::MAX - 1, 1, 1),
            Err(JoltFormulaDimensionsError::overflow(
                "Jolt RA polynomial count"
            ))
        );
    }

    #[test]
    fn records_layout_counts_and_total() -> Result<(), JoltFormulaDimensionsError> {
        let layout = JoltRaPolynomialLayout::new(2, 3, 5)?;

        assert_eq!(layout.instruction(), 2);
        assert_eq!(layout.bytecode(), 3);
        assert_eq!(layout.ram(), 5);
        assert_eq!(layout.total(), 10);
        Ok(())
    }

    #[test]
    fn iterates_canonical_ra_order() -> Result<(), JoltFormulaDimensionsError> {
        let layout = JoltRaPolynomialLayout::new(2, 1, 2)?;

        assert_eq!(
            layout.polynomials().collect::<Vec<_>>(),
            vec![
                JoltRaPolynomial::Instruction(0),
                JoltRaPolynomial::Instruction(1),
                JoltRaPolynomial::Bytecode(0),
                JoltRaPolynomial::Ram(0),
                JoltRaPolynomial::Ram(1),
            ]
        );
        assert_eq!(
            layout.committed_polynomials().collect::<Vec<_>>(),
            vec![
                JoltCommittedPolynomial::InstructionRa(0),
                JoltCommittedPolynomial::InstructionRa(1),
                JoltCommittedPolynomial::BytecodeRa(0),
                JoltCommittedPolynomial::RamRa(0),
                JoltCommittedPolynomial::RamRa(1),
            ]
        );
        Ok(())
    }
}
