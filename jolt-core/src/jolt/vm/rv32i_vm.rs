use crate::field::JoltField;
use enum_dispatch::enum_dispatch;
use rand::{prelude::StdRng, RngCore};
use serde::{Deserialize, Serialize};
use std::any::TypeId;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{EnumCount as EnumCountMacro, EnumIter};

use super::{Jolt, JoltProof};
use crate::jolt::instruction::{
    add::ADDInstruction, and::ANDInstruction, beq::BEQInstruction, bge::BGEInstruction,
    bgeu::BGEUInstruction, bne::BNEInstruction, lb::LBInstruction, lh::LHInstruction,
    movsign::MOVSIGNInstruction, mul::MULInstruction, mulhu::MULHUInstruction,
    mulu::MULUInstruction, or::ORInstruction, sb::SBInstruction, sh::SHInstruction,
    sll::SLLInstruction, slt::SLTInstruction, sltu::SLTUInstruction, sra::SRAInstruction,
    srl::SRLInstruction, sub::SUBInstruction, sw::SWInstruction, virtual_advice::ADVICEInstruction,
    virtual_assert_eq_signs::ASSERTEQSIGNSInstruction,
    virtual_assert_lt_abs::ASSERTLTABSInstruction, virtual_assert_lte::ASSERTLTEInstruction,
    xor::XORInstruction, JoltInstruction, JoltInstructionSet, SubtableIndices,
};
use crate::jolt::subtable::{
    and::AndSubtable, eq::EqSubtable, eq_abs::EqAbsSubtable, eq_msb::EqMSBSubtable,
    gt_msb::GtMSBSubtable, identity::IdentitySubtable, lt_abs::LtAbsSubtable, ltu::LtuSubtable,
    or::OrSubtable, sign_extend::SignExtendSubtable, sll::SllSubtable, sra_sign::SraSignSubtable,
    srl::SrlSubtable, truncate_overflow::TruncateOverflowSubtable, xor::XorSubtable,
    JoltSubtableSet, LassoSubtable, SubtableId,
};
use crate::poly::commitment::commitment_scheme::CommitmentScheme;

/// Generates an enum out of a list of JoltInstruction types. All JoltInstruction methods
/// are callable on the enum type via enum_dispatch.
macro_rules! instruction_set {
    ($enum_name:ident, $($alias:ident: $struct:ty),+) => {
        #[allow(non_camel_case_types)]
        #[repr(u8)]
        #[derive(Copy, Clone, Debug, EnumIter, EnumCountMacro, Serialize, Deserialize)]
        #[enum_dispatch(JoltInstruction)]
        pub enum $enum_name { $($alias($struct)),+ }
        impl JoltInstructionSet for $enum_name {}
        impl $enum_name {
            pub fn random_instruction(rng: &mut StdRng) -> Self {
                let index = rng.next_u64() as usize % $enum_name::COUNT;
                let instruction = $enum_name::iter()
                    .enumerate()
                    .filter(|(i, _)| *i == index)
                    .map(|(_, x)| x)
                    .next()
                    .unwrap();
                instruction.random(rng)
            }
        }
    };
}

/// Generates an enum out of a list of LassoSubtable types. All LassoSubtable methods
/// are callable on the enum type via enum_dispatch.
macro_rules! subtable_enum {
    ($enum_name:ident, $($alias:ident: $struct:ty),+) => {
        #[allow(non_camel_case_types)]
        #[repr(usize)]
        #[enum_dispatch(LassoSubtable<F>)]
        #[derive(EnumCountMacro, EnumIter)]
        pub enum $enum_name<F: JoltField> { $($alias($struct)),+ }
        impl<F: JoltField> From<SubtableId> for $enum_name<F> {
          fn from(subtable_id: SubtableId) -> Self {
            $(
              if subtable_id == TypeId::of::<$struct>() {
                $enum_name::from(<$struct>::new())
              } else
            )+
            { panic!("Unexpected subtable id {:?}", subtable_id) } // TODO(moodlezoup): better error handling
          }
        }

        impl<F: JoltField> From<$enum_name<F>> for usize {
            fn from(subtable: $enum_name<F>) -> usize {
                unsafe { *<*const _>::from(&subtable).cast::<usize>() }
            }
        }
        impl<F: JoltField> JoltSubtableSet<F> for $enum_name<F> {}
    };
}

const WORD_SIZE: usize = 32;

instruction_set!(
  RV32I,
  ADD: ADDInstruction<WORD_SIZE>,
  SUB: SUBInstruction<WORD_SIZE>,
  AND: ANDInstruction,
  OR: ORInstruction,
  XOR: XORInstruction,
  LB: LBInstruction,
  LH: LHInstruction,
  SB: SBInstruction,
  SH: SHInstruction,
  SW: SWInstruction,
  BEQ: BEQInstruction,
  BGE: BGEInstruction,
  BGEU: BGEUInstruction,
  BNE: BNEInstruction,
  SLT: SLTInstruction,
  SLTU: SLTUInstruction,
  SLL: SLLInstruction<WORD_SIZE>,
  SRA: SRAInstruction<WORD_SIZE>,
  SRL: SRLInstruction<WORD_SIZE>,
  MOVSIGN: MOVSIGNInstruction<WORD_SIZE>,
  MUL: MULInstruction<WORD_SIZE>,
  MULU: MULUInstruction<WORD_SIZE>,
  MULHU: MULHUInstruction<WORD_SIZE>,
  VIRTUAL_ADVICE: ADVICEInstruction<WORD_SIZE>,
  VIRTUAL_ASSERT_LTE: ASSERTLTEInstruction,
  VIRTUAL_ASSERT_LT_ABS: ASSERTLTABSInstruction<WORD_SIZE>,
  VIRTUAL_ASSERT_EQ_SIGNS: ASSERTEQSIGNSInstruction
);
subtable_enum!(
  RV32ISubtables,
  AND: AndSubtable<F>,
  EQ_ABS: EqAbsSubtable<F>,
  EQ_MSB: EqMSBSubtable<F>,
  EQ: EqSubtable<F>,
  GT_MSB: GtMSBSubtable<F>,
  IDENTITY: IdentitySubtable<F>,
  LT_ABS: LtAbsSubtable<F>,
  LTU: LtuSubtable<F>,
  OR: OrSubtable<F>,
  SIGN_EXTEND_8: SignExtendSubtable<F, 8>,
  SIGN_EXTEND_16: SignExtendSubtable<F, 16>,
  SLL0: SllSubtable<F, 0, WORD_SIZE>,
  SLL1: SllSubtable<F, 1, WORD_SIZE>,
  SLL2: SllSubtable<F, 2, WORD_SIZE>,
  SLL3: SllSubtable<F, 3, WORD_SIZE>,
  SRA_SIGN: SraSignSubtable<F, WORD_SIZE>,
  SRL0: SrlSubtable<F, 0, WORD_SIZE>,
  SRL1: SrlSubtable<F, 1, WORD_SIZE>,
  SRL2: SrlSubtable<F, 2, WORD_SIZE>,
  SRL3: SrlSubtable<F, 3, WORD_SIZE>,
  TRUNCATE: TruncateOverflowSubtable<F, WORD_SIZE>,
  TRUNCATE_BYTE: TruncateOverflowSubtable<F, 8>,
  XOR: XorSubtable<F>
);

// ==================== JOLT ====================

pub enum RV32IJoltVM {}

pub const C: usize = 4;
pub const M: usize = 1 << 16;

impl<F, CS> Jolt<F, CS, C, M> for RV32IJoltVM
where
    F: JoltField,
    CS: CommitmentScheme<Field = F>,
{
    type InstructionSet = RV32I;
    type Subtables = RV32ISubtables<F>;
}

pub type RV32IJoltProof<F, CS> = JoltProof<C, M, F, CS, RV32I, RV32ISubtables<F>>;

// ==================== TEST ====================

#[cfg(test)]
mod tests {
    use ark_bn254::{Bn254, Fr, G1Projective};

    use std::collections::HashSet;

    use crate::field::JoltField;
    use crate::host;
    use crate::jolt::instruction::JoltInstruction;
    use crate::jolt::vm::rv32i_vm::{Jolt, RV32IJoltVM, C, M};
    use crate::poly::commitment::commitment_scheme::CommitmentScheme;
    use crate::poly::commitment::hyperkzg::HyperKZG;
    use crate::poly::commitment::hyrax::HyraxScheme;
    use crate::poly::commitment::mock::MockCommitScheme;
    use crate::poly::commitment::zeromorph::Zeromorph;
    use std::sync::Mutex;
    use strum::{EnumCount, IntoEnumIterator};

    // If multiple tests try to read the same trace artifacts simultaneously, they will fail
    lazy_static::lazy_static! {
        static ref FIB_FILE_LOCK: Mutex<()> = Mutex::new(());
        static ref SHA3_FILE_LOCK: Mutex<()> = Mutex::new(());
    }

    fn test_instruction_set_subtables<PCS: CommitmentScheme>() {
        let mut subtable_set: HashSet<_> = HashSet::new();
        for instruction in <RV32IJoltVM as Jolt<_, PCS, C, M>>::InstructionSet::iter() {
            for (subtable, _) in instruction.subtables::<Fr>(C, M) {
                // panics if subtable cannot be cast to enum variant
                let _ =
                    <RV32IJoltVM as Jolt<_, PCS, C, M>>::Subtables::from(subtable.subtable_id());
                subtable_set.insert(subtable.subtable_id());
            }
        }
        assert_eq!(
            subtable_set.len(),
            <RV32IJoltVM as Jolt<_, PCS, C, M>>::Subtables::COUNT,
            "Unused enum variants in Subtables"
        );
    }

    #[test]
    fn instruction_set_subtables() {
        test_instruction_set_subtables::<HyraxScheme<G1Projective>>();
        test_instruction_set_subtables::<Zeromorph<Bn254>>();
        test_instruction_set_subtables::<HyperKZG<Bn254>>();
    }

    fn fib_e2e<F: JoltField, PCS: CommitmentScheme<Field = F>>() {
        let artifact_guard = FIB_FILE_LOCK.lock().unwrap();
        let mut program = host::Program::new("fibonacci-guest");
        program.set_input(&9u32);
        let (bytecode, memory_init) = program.decode();
        let (io_device, trace, circuit_flags) = program.trace();
        drop(artifact_guard);

        let preprocessing =
            RV32IJoltVM::preprocess(bytecode.clone(), memory_init, 1 << 20, 1 << 20, 1 << 20);
        let (proof, commitments) = <RV32IJoltVM as Jolt<F, PCS, C, M>>::prove(
            io_device,
            trace,
            circuit_flags,
            preprocessing.clone(),
        );
        let verification_result = RV32IJoltVM::verify(preprocessing, proof, commitments);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    fn fib_e2e_mock() {
        fib_e2e::<Fr, MockCommitScheme<Fr>>();
    }

    #[test]
    fn fib_e2e_hyrax() {
        fib_e2e::<ark_bn254::Fr, HyraxScheme<ark_bn254::G1Projective>>();
    }

    #[test]
    fn fib_e2e_zeromorph() {
        fib_e2e::<Fr, Zeromorph<Bn254>>();
    }

    #[test]
    fn fib_e2e_hyperkzg() {
        fib_e2e::<Fr, HyperKZG<Bn254>>();
    }

    // TODO(sragss): Finish Binius.
    // #[test]
    // fn fib_e2e_binius() {
    //     type Field = crate::field::binius::BiniusField<binius_field::BinaryField128b>;
    //     fib_e2e::<Field, MockCommitScheme<Field>>();
    // }

    #[test]
    fn sha3_e2e_hyrax() {
        let _guard = SHA3_FILE_LOCK.lock().unwrap();

        let mut program = host::Program::new("sha3-guest");
        program.set_input(&[5u8; 32]);
        let (bytecode, memory_init) = program.decode();
        let (io_device, trace, circuit_flags) = program.trace();

        let preprocessing =
            RV32IJoltVM::preprocess(bytecode.clone(), memory_init, 1 << 20, 1 << 20, 1 << 20);
        let (jolt_proof, jolt_commitments) =
            <RV32IJoltVM as Jolt<_, HyraxScheme<G1Projective>, C, M>>::prove(
                io_device,
                trace,
                circuit_flags,
                preprocessing.clone(),
            );

        let verification_result = RV32IJoltVM::verify(preprocessing, jolt_proof, jolt_commitments);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    fn sha3_e2e_zeromorph() {
        let _guard = SHA3_FILE_LOCK.lock().unwrap();

        let mut program = host::Program::new("sha3-guest");
        program.set_input(&[5u8; 32]);
        let (bytecode, memory_init) = program.decode();
        let (io_device, trace, circuit_flags) = program.trace();

        let preprocessing =
            RV32IJoltVM::preprocess(bytecode.clone(), memory_init, 1 << 20, 1 << 20, 1 << 20);
        let (jolt_proof, jolt_commitments) =
            <RV32IJoltVM as Jolt<_, Zeromorph<Bn254>, C, M>>::prove(
                io_device,
                trace,
                circuit_flags,
                preprocessing.clone(),
            );

        let verification_result = RV32IJoltVM::verify(preprocessing, jolt_proof, jolt_commitments);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }

    #[test]
    fn sha3_e2e_hyperkzg() {
        let _guard = SHA3_FILE_LOCK.lock().unwrap();

        let mut program = host::Program::new("sha3-guest");
        program.set_input(&[5u8; 32]);
        let (bytecode, memory_init) = program.decode();
        let (io_device, trace, circuit_flags) = program.trace();

        let preprocessing =
            RV32IJoltVM::preprocess(bytecode.clone(), memory_init, 1 << 20, 1 << 20, 1 << 20);
        let (jolt_proof, jolt_commitments) = <RV32IJoltVM as Jolt<_, HyperKZG<Bn254>, C, M>>::prove(
            io_device,
            trace,
            circuit_flags,
            preprocessing.clone(),
        );

        let verification_result = RV32IJoltVM::verify(preprocessing, jolt_proof, jolt_commitments);
        assert!(
            verification_result.is_ok(),
            "Verification failed with error: {:?}",
            verification_result.err()
        );
    }
}
