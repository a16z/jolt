use ark_ec::CurveGroup;
use ark_ff::PrimeField;
use ark_serialize::{CanonicalDeserialize, CanonicalSerialize};
use merlin::Transcript;
use std::any::TypeId;
use strum::{EnumCount, IntoEnumIterator};

use crate::{
  jolt::{
    instruction::{JoltInstruction, Opcode},
    subtable::LassoSubtable,
    vm::pc::PCPolys,
  },
  lasso::{fingerprint_strategy::ROFlagsFingerprintProof, memory_checking::MemoryCheckingProof},
  poly::{
    dense_mlpoly::{DensePolynomial, PolyCommitmentGens},
    eq_poly::EqPolynomial,
  },
  subprotocols::{
    combined_table_proof::{CombinedTableCommitment, CombinedTableEvalProof},
    sumcheck::SumcheckInstanceProof,
  },
  utils::{
    errors::ProofVerifyError,
    math::Math,
    random::RandomTape,
    transcript::{AppendToTranscript, ProofTranscript},
  },
};

use self::instruction_lookups::InstructionLookups;
use self::memory::MemoryOp;
use self::pc::{PCFingerprintProof, PCProof};

pub trait Jolt<F: PrimeField, G: CurveGroup<ScalarField = F>>: InstructionLookups<F, G> {
  fn prove() {
    // preprocess?
    // emulate
    // prove_program_code
    // prove_memory
    // prove_lookups
    // prove_r1cs
    unimplemented!("todo");
  }

  fn prove_program_code(
    program_code: &[u64],
    access_sequence: &[usize],
    code_size: usize,
    contiguous_reads_per_access: usize,
    r_mem_check: &(F, F),
    transcript: &mut Transcript,
  ) -> MemoryCheckingProof<G, PCFingerprintProof<G>> {
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

  fn prove_memory(
    memory_trace: Vec<MemoryOp>,
    memory_size: usize,
    r_mem_check: &(F, F),
    transcript: &mut Transcript,
  ) {
    // let (gamma, tau) = r_mem_check;
    // let hash_func = |a: &F, v: &F, t: &F| -> F { *t * gamma.square() + *v * *gamma + *a - tau };

    // let m: usize = memory_trace.len().next_power_of_two();
    // // TODO(moodlezoup): resize memory_trace

    // let mut timestamp: u64 = 0;

    // let mut read_set: Vec<(F, F, F)> = Vec::with_capacity(m);
    // let mut write_set: Vec<(F, F, F)> = Vec::with_capacity(m);
    // let mut final_set: Vec<(F, F, F)> = (0..memory_size)
    //   .map(|i| (F::from(i as u64), F::zero(), F::zero()))
    //   .collect();

    // for memory_access in memory_trace {
    //   match memory_access {
    //     MemoryOp::Read(a, v) => {
    //       read_set.push((F::from(a), F::from(v), F::from(timestamp)));
    //       write_set.push((F::from(a), F::from(v), F::from(timestamp + 1)));
    //       final_set[a as usize] = (F::from(a), F::from(v), F::from(timestamp + 1));
    //     }
    //     MemoryOp::Write(a, v_old, v_new) => {
    //       read_set.push((F::from(a), F::from(v_old), F::from(timestamp)));
    //       write_set.push((F::from(a), F::from(v_new), F::from(timestamp + 1)));
    //       final_set[a as usize] = (F::from(a), F::from(v_new), F::from(timestamp + 1));
    //     }
    //   }
    //   timestamp += 1;
    // }

    // let init_poly = DensePolynomial::new(
    //   (0..memory_size)
    //     .map(|i| {
    //       // addr is given by i, init value is 0, and ts = 0
    //       hash_func(&F::from(i as u64), &F::zero(), &F::zero())
    //     })
    //     .collect::<Vec<F>>(),
    // );
    // let read_poly = DensePolynomial::new(
    //   read_set
    //     .iter()
    //     .map(|(a, v, t)| hash_func(a, v, t))
    //     .collect::<Vec<F>>(),
    // );
    // let write_poly = DensePolynomial::new(
    //   write_set
    //     .iter()
    //     .map(|(a, v, t)| hash_func(a, v, t))
    //     .collect::<Vec<F>>(),
    // );
    // let final_poly = DensePolynomial::new(
    //   final_set
    //     .iter()
    //     .map(|(a, v, t)| hash_func(a, v, t))
    //     .collect::<Vec<F>>(),
    // );

    // Memory checking
    // Lasso range cheeck on read timestamps to enforce each timestamp read at step i is less than i
    unimplemented!("todo");
  }

  fn prove_r1cs() {
    unimplemented!("todo")
  }
}

pub mod instruction_lookups;
pub mod memory;
pub mod pc;
pub mod test_vm;
