use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use common::constants::{ALIGNMENT_FACTOR_BYTECODE, RAM_START_ADDRESS};
use tracer::instruction::{Cycle, Instruction};

use crate::zkvm::guest_serde::{GuestDeserialize, GuestSerialize};

pub mod read_raf_checking;

#[derive(Default, Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct BytecodePreprocessing {
    pub code_size: usize,
    pub bytecode: Vec<Instruction>,
    /// Maps the memory address of each instruction in the bytecode to its "virtual" address.
    /// See Section 6.1 of the Jolt paper, "Reflecting the program counter". The virtual address
    /// is the one used to keep track of the next (potentially virtual) instruction to execute.
    pub pc_map: BytecodePCMapper,
}

impl BytecodePreprocessing {
    #[tracing::instrument(skip_all, name = "BytecodePreprocessing::preprocess")]
    pub fn preprocess(mut bytecode: Vec<Instruction>) -> Self {
        // Bytecode: Prepend a single no-op instruction
        bytecode.insert(0, Instruction::NoOp);
        let pc_map = BytecodePCMapper::new(&bytecode);

        let code_size = bytecode.len().next_power_of_two().max(2);

        // Bytecode: Pad to nearest power of 2
        bytecode.resize(code_size, Instruction::NoOp);

        Self {
            code_size,
            bytecode,
            pc_map,
        }
    }

    pub fn get_pc(&self, cycle: &Cycle) -> usize {
        if matches!(cycle, tracer::instruction::Cycle::NoOp) {
            return 0;
        }
        let instr = cycle.instruction().normalize();
        self.pc_map
            .get_pc(instr.address, instr.virtual_sequence_remaining.unwrap_or(0))
    }
}

#[derive(Default, Debug, Clone, CanonicalSerialize, CanonicalDeserialize)]
pub struct BytecodePCMapper {
    /// Stores the mapping of the PC at the beginning of each inline sequence
    /// and the maximum number of the inline sequence
    /// Indexed by the address of instruction unmapped divided by 2
    indices: Vec<Option<(usize, u16)>>,
}

impl crate::zkvm::guest_serde::GuestSerialize for Instruction {
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        // Instruction canonical encoding is already the optimized binary encoding.
        self.serialize_compressed(w)
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "Instruction"))
    }
}

impl crate::zkvm::guest_serde::GuestDeserialize for Instruction {
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        Instruction::deserialize_compressed(&mut *r)
            .map_err(|_| std::io::Error::new(std::io::ErrorKind::InvalidData, "Instruction"))
    }
}

impl crate::zkvm::guest_serde::GuestSerialize for BytecodePreprocessing {
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        self.code_size.guest_serialize(w)?;
        self.bytecode.guest_serialize(w)?;
        self.pc_map.guest_serialize(w)?;
        Ok(())
    }
}

impl crate::zkvm::guest_serde::GuestDeserialize for BytecodePreprocessing {
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self {
            code_size: usize::guest_deserialize(r)?,
            bytecode: Vec::<Instruction>::guest_deserialize(r)?,
            pc_map: BytecodePCMapper::guest_deserialize(r)?,
        })
    }
}

impl crate::zkvm::guest_serde::GuestSerialize for BytecodePCMapper {
    fn guest_serialize<W: std::io::Write>(&self, w: &mut W) -> std::io::Result<()> {
        // `indices` are small and contain no field elements; still encode with GuestSerialize
        // to avoid `usize` ambiguity.
        self.indices.guest_serialize(w)
    }
}

impl crate::zkvm::guest_serde::GuestDeserialize for BytecodePCMapper {
    fn guest_deserialize<R: std::io::Read>(r: &mut R) -> std::io::Result<Self> {
        Ok(Self {
            indices: Vec::<Option<(usize, u16)>>::guest_deserialize(r)?,
        })
    }
}

impl BytecodePCMapper {
    pub fn new(bytecode: &[Instruction]) -> Self {
        let mut indices: Vec<Option<(usize, u16)>> = {
            // For read-raf tests we simulate bytecode being empty
            #[cfg(test)]
            if bytecode.len() == 1 {
                vec![None; 1]
            } else {
                vec![None; Self::get_index(bytecode.last().unwrap().normalize().address) + 1]
            }
            #[cfg(not(test))]
            vec![None; Self::get_index(bytecode.last().unwrap().normalize().address) + 1]
        };
        let mut last_pc = 0;
        // Push the initial noop instruction
        indices[0] = Some((last_pc, 0));
        bytecode.iter().for_each(|instr| {
            let instr = instr.normalize();
            if instr.address == 0 {
                // ignore unimplemented instructions
                return;
            }
            last_pc += 1;
            if let Some((_, max_sequence)) = indices.get(Self::get_index(instr.address)).unwrap() {
                if instr.virtual_sequence_remaining.unwrap_or(0) >= *max_sequence {
                    panic!(
                        "Bytecode has non-decreasing inline sequences at index {}",
                        Self::get_index(instr.address)
                    );
                }
            } else {
                indices[Self::get_index(instr.address)] =
                    Some((last_pc, instr.virtual_sequence_remaining.unwrap_or(0)));
            }
        });
        Self { indices }
    }

    pub fn get_pc(&self, address: usize, virtual_sequence_remaining: u16) -> usize {
        let (base_pc, max_inline_seq) = self
            .indices
            .get(Self::get_index(address))
            .unwrap()
            .expect("PC for address not found");
        base_pc + (max_inline_seq - virtual_sequence_remaining) as usize
    }

    pub const fn get_index(address: usize) -> usize {
        assert!(address >= RAM_START_ADDRESS as usize);
        assert!(address.is_multiple_of(ALIGNMENT_FACTOR_BYTECODE));
        (address - RAM_START_ADDRESS as usize) / ALIGNMENT_FACTOR_BYTECODE + 1
    }
}
