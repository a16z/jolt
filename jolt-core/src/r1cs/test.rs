// use crate::{
//     field::JoltField,
//     impl_r1cs_input_lc_conversions,
//     r1cs::{
//         builder::{OffsetEqConstraint, R1CSBuilder},
//         constraints::R1CSConstraints,
//     },
// };

// use super::{
//     builder::CombinedUniformBuilder,
//     inputs::ConstraintInput,
//     key::{SparseConstraints, UniformSpartanKey},
// };

// #[allow(non_camel_case_types)]
// #[derive(
//     strum_macros::EnumIter,
//     strum_macros::EnumCount,
//     Clone,
//     Copy,
//     Debug,
//     PartialEq,
//     Eq,
//     PartialOrd,
//     Ord,
//     Hash,
// )]
// #[repr(usize)]
// pub enum SimpTestIn {
//     Q,
//     R,
//     S,
// }
// impl ConstraintInput for SimpTestIn {
//     fn flatten<const C: usize>() -> Vec<Self> {
//         vec![Self::Q, Self::R, Self::S]
//     }

//     fn get_poly_ref<'a, F: JoltField>(
//         &self,
//         jolt_polynomials: &'a crate::jolt::vm::JoltPolynomials<F>,
//     ) -> &'a crate::poly::dense_mlpoly::DensePolynomial<F> {
//         todo!()
//     }

//     fn get_poly_ref_mut<'a, F: JoltField>(
//         &self,
//         jolt_polynomials: &'a mut crate::jolt::vm::JoltPolynomials<F>,
//     ) -> &'a mut crate::poly::dense_mlpoly::DensePolynomial<F> {
//         todo!()
//     }
// }
// impl_r1cs_input_lc_conversions!(SimpTestIn, 4);

// #[allow(non_camel_case_types)]
// #[derive(
//     strum_macros::EnumIter,
//     strum_macros::EnumCount,
//     Clone,
//     Copy,
//     Debug,
//     PartialEq,
//     Eq,
//     PartialOrd,
//     Ord,
//     Hash,
// )]
// #[repr(usize)]
// pub enum TestInputs {
//     PcIn,
//     PcOut,
//     BytecodeA,
//     BytecodeVOpcode,
//     BytecodeVRS1,
//     BytecodeVRS2,
//     BytecodeVRD,
//     BytecodeVImm,
//     RAMA,
//     RAMRS1,
//     RAMRS2,
//     RAMByte0,
//     RAMByte1,
//     RAMByte2,
//     RAMByte3,
//     OpFlags0,
//     OpFlags1,
//     OpFlags2,
//     OpFlags3,
//     OpFlags_SignImm,
// }
// impl ConstraintInput for TestInputs {
//     fn flatten<const C: usize>() -> Vec<Self> {
//         vec![
//             Self::PcIn,
//             Self::PcOut,
//             Self::BytecodeA,
//             Self::BytecodeVOpcode,
//             Self::BytecodeVRS1,
//             Self::BytecodeVRS2,
//             Self::BytecodeVRD,
//             Self::BytecodeVImm,
//             Self::RAMA,
//             Self::RAMRS1,
//             Self::RAMRS2,
//             Self::RAMByte0,
//             Self::RAMByte1,
//             Self::RAMByte2,
//             Self::RAMByte3,
//             Self::OpFlags0,
//             Self::OpFlags1,
//             Self::OpFlags2,
//             Self::OpFlags3,
//             Self::OpFlags_SignImm,
//         ]
//     }

//     fn get_poly_ref<'a, F: JoltField>(
//         &self,
//         jolt_polynomials: &'a crate::jolt::vm::JoltPolynomials<F>,
//     ) -> &'a crate::poly::dense_mlpoly::DensePolynomial<F> {
//         todo!()
//     }

//     fn get_poly_ref_mut<'a, F: JoltField>(
//         &self,
//         jolt_polynomials: &'a mut crate::jolt::vm::JoltPolynomials<F>,
//     ) -> &'a mut crate::poly::dense_mlpoly::DensePolynomial<F> {
//         todo!()
//     }
// }
// impl_r1cs_input_lc_conversions!(TestInputs, 4);

// pub fn materialize_full_uniform<F: JoltField>(
//     key: &UniformSpartanKey<4, TestInputs, F>,
//     sparse_constraints: &SparseConstraints<F>,
// ) -> Vec<F> {
//     let row_width = 2 * key.num_vars_total().next_power_of_two();
//     let col_height = key.num_cons_total;
//     let total_size = row_width * col_height;
//     assert!(total_size.is_power_of_two());
//     let mut materialized = vec![F::zero(); total_size];

//     for (row, col, val) in sparse_constraints.vars.iter() {
//         for step_index in 0..key.num_steps {
//             let x = col * key.num_steps + step_index;
//             let y = row * key.num_steps + step_index;
//             let i = y * row_width + x;
//             materialized[i] = *val;
//         }
//     }

//     let const_col_index = key.num_vars_total();
//     for (row, val) in sparse_constraints.consts.iter() {
//         for step_index in 0..key.num_steps {
//             let y = row * key.num_steps + step_index;
//             let i = y * row_width + const_col_index;
//             materialized[i] = *val;
//         }
//     }

//     materialized
// }

// pub fn materialize_all<F: JoltField>(
//     key: &UniformSpartanKey<4, TestInputs, F>,
// ) -> (Vec<F>, Vec<F>, Vec<F>) {
//     (
//         materialize_full_uniform(key, &key.uniform_r1cs.a),
//         materialize_full_uniform(key, &key.uniform_r1cs.b),
//         materialize_full_uniform(key, &key.uniform_r1cs.c),
//     )
// }

// pub fn simp_test_builder_key<F: JoltField>() -> (
//     CombinedUniformBuilder<4, F, SimpTestIn>,
//     UniformSpartanKey<4, SimpTestIn, F>,
// ) {
//     let mut uniform_builder = R1CSBuilder::<F, SimpTestIn>::new();
//     // Q - R == 0
//     // R - S == 0
//     struct TestConstraints();
//     impl<const C: usize, F: JoltField> R1CSConstraints<C, F> for TestConstraints {
//         type Inputs = SimpTestIn;

//         fn uniform_constraints(builder: &mut R1CSBuilder<C, F, Self::Inputs>, memory_start: u64) {
//             builder.constrain_eq(SimpTestIn::Q, SimpTestIn::R);
//             builder.constrain_eq(SimpTestIn::R, SimpTestIn::S);
//         }

//         fn non_uniform_constraints() -> Vec<OffsetEqConstraint> {
//             // Q[n] + 4 - S[n+1] == 0
//             let offset_eq_constraint = OffsetEqConstraint::new(
//                 (SimpTestIn::S, true),
//                 (SimpTestIn::Q, false),
//                 (SimpTestIn::S + -4, true),
//             );
//             vec![offset_eq_constraint]
//         }
//     }

//     let constraints = TestConstraints();
//     constraints.build_constraints(&mut uniform_builder);

//     let _num_steps: usize = 3;
//     let num_steps_pad = 4;
//     let combined_builder = CombinedUniformBuilder::construct(
//         uniform_builder,
//         num_steps_pad,
//         TestConstraints::non_uniform_constraints(),
//     );
//     let key = UniformSpartanKey::from_builder(&combined_builder);

//     (combined_builder, key)
// }

// pub fn simp_test_big_matrices<F: JoltField>() -> (Vec<F>, Vec<F>, Vec<F>) {
//     let (_, key) = simp_test_builder_key();
//     let mut big_a = materialize_full_uniform(&key, &key.uniform_r1cs.a);
//     let mut big_b = materialize_full_uniform(&key, &key.uniform_r1cs.b);
//     let big_c = materialize_full_uniform(&key, &key.uniform_r1cs.c);

//     // Written by hand from non-uniform constraints
//     let row_0_index = 32 * 8;
//     let row_1_index = 32 * 9;
//     let row_2_index = 32 * 10;
//     let row_3_index = 32 * 11;
//     big_a[row_0_index] = F::one();
//     big_a[row_0_index + 9] = F::from_i64(-1);
//     big_a[row_1_index + 1] = F::one();
//     big_a[row_1_index + 10] = F::from_i64(-1);
//     big_a[row_2_index + 2] = F::one();
//     big_a[row_2_index + 11] = F::from_i64(-1);
//     big_a[row_3_index + 3] = F::one();

//     big_b[row_0_index + 9] = F::one();
//     big_b[row_1_index + 10] = F::one();
//     big_b[row_2_index + 11] = F::one();

//     // Constants
//     big_a[row_0_index + 16] = F::from_u64(4).unwrap();
//     big_a[row_1_index + 16] = F::from_u64(4).unwrap();
//     big_a[row_2_index + 16] = F::from_u64(4).unwrap();
//     big_a[row_3_index + 16] = F::from_u64(4).unwrap();

//     (big_a, big_b, big_c)
// }
