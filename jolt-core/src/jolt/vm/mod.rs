use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_serialize::Read;
use merlin::Transcript;
use std::any::TypeId;
use strum::{EnumCount, IntoEnumIterator};

use crate::lasso::memory_checking::{MemoryCheckingProof, MemoryCheckingProver};

use crate::jolt::{
  instruction::{JoltInstruction, Opcode},
  subtable::LassoSubtable,
};
use crate::poly::structured_poly::StructuredPolynomials;
use crate::utils::{errors::ProofVerifyError, random::RandomTape};

use self::instruction_lookups::{InstructionLookups, InstructionLookupsProof};
use self::read_write_memory::{MemoryOp, ReadWriteMemory};

pub trait Jolt<F: PrimeField, G: CurveGroup<ScalarField = F>, const C: usize, const M: usize> {
  type InstructionSet: JoltInstruction + Opcode + IntoEnumIterator + EnumCount;
  type Subtables: LassoSubtable<F> + IntoEnumIterator + EnumCount + From<TypeId> + Into<usize>;

  fn prove() {
    // preprocess?
    // emulate
    // prove_program_code
    // prove_memory
    // prove_lookups
    // prove_r1cs
    unimplemented!("todo");
  }

  fn prove_instruction_lookups(
    ops: Vec<Self::InstructionSet>,
    r: Vec<F>,
    transcript: &mut Transcript,
  ) -> InstructionLookupsProof<F, G> {
    let instruction_lookups =
      InstructionLookups::<F, G, Self::InstructionSet, Self::Subtables, C, M>::new(ops);
    instruction_lookups.prove_lookups(r, transcript)
  }

  fn verify_instruction_lookups(
    proof: InstructionLookupsProof<F, G>,
    r: Vec<F>,
    transcript: &mut Transcript,
  ) -> Result<(), ProofVerifyError> {
    InstructionLookups::<F, G, Self::InstructionSet, Self::Subtables, C, M>::verify(
      proof, &r, transcript,
    )
  }

  fn prove_program_code(
    program_code: &[u64],
    access_sequence: &[usize],
    code_size: usize,
    contiguous_reads_per_access: usize,
    r_mem_check: &(F, F),
    transcript: &mut Transcript,
  ) {
    // let (gamma, tau) = r_mem_check;
    // let hash_func = |a: &F, v: &F, t: &F| -> F { *t * gamma.square() + *v * *gamma + *a - tau };

    // let m: usize = (access_sequence.len() * contiguous_reads_per_access).next_power_of_two();
    // // TODO(moodlezoup): resize access_sequence?

    // let mut read_addrs: Vec<usize> = Vec::with_capacity(m);
    // let mut final_cts: Vec<usize> = vec![0; code_size];
    // let mut read_cts: Vec<usize> = Vec::with_capacity(m);
    // let mut read_values: Vec<u64> = Vec::with_capacity(m);

    // for (j, code_address) in access_sequence.iter().enumerate() {
    //   debug_assert!(code_address + contiguous_reads_per_access <= code_size);
    //   debug_assert!(code_address % contiguous_reads_per_access == 0);

    //   for offset in 0..contiguous_reads_per_access {
    //     let addr = code_address + offset;
    //     let counter = final_cts[addr];
    //     read_addrs.push(addr);
    //     read_values.push(program_code[addr]);
    //     read_cts.push(counter);
    //     final_cts[addr] = counter + 1;
    //   }
    // }

    // let E_poly: DensePolynomial<F> = DensePolynomial::from_u64(&read_values); // v_ops
    // let dim: DensePolynomial<F> = DensePolynomial::from_usize(access_sequence); // a_ops
    // let read_cts: DensePolynomial<F> = DensePolynomial::from_usize(&read_cts); // t_read
    // let final_cts: DensePolynomial<F> = DensePolynomial::from_usize(&final_cts); // t_final
    // let init_values: DensePolynomial<F> = DensePolynomial::from_u64(program_code); // v_mem

    // let polys = PCPolys::new(dim, E_poly, init_values, read_cts, final_cts, 0);
    // let (gens, commitments) = polys.commit::<G>();

    todo!("decide how to represent nested proofs, gens, commitments");
    // MemoryCheckingProof::<G, PCFingerprintProof<G>>::prove(
    //   &polys,
    //   r_fingerprints,
    //   &gens,
    //   &mut transcript,
    //   &mut random_tape,
    // )
  }

  fn prove_memory(memory_trace: Vec<MemoryOp>, memory_size: usize, transcript: &mut Transcript) {
    todo!("Load program bytecode into memory");

    let memory: ReadWriteMemory<F, G> = ReadWriteMemory::new(memory_trace, memory_size, transcript);
    let batched_polys = memory.batch();
    let commitments = ReadWriteMemory::commit(&batched_polys);

    let mut random_tape = RandomTape::new(b"proof");
    memory.prove_memory_checking(
      &memory,
      &batched_polys,
      &commitments,
      transcript,
      &mut random_tape,
    );

    todo!("Lasso lookups to enforce timestamp validity")
  }

  fn prove_r1cs() {
    unimplemented!("todo")
  }
}

pub mod instruction_lookups;
pub mod pc;
pub mod read_write_memory;
pub mod test_vm;
