use crate::circuit::ops::hybrid::HybridOp;
use crate::circuit::ops::poly::PolyOp;
use crate::circuit::*;
use crate::tensor::{Tensor, TensorType, ValTensor, VarTensor};
use halo2_proofs::{
    circuit::{Layouter, SimpleFloorPlanner, Value},
    dev::MockProver,
    plonk::{Circuit, ConstraintSystem, Error},
};
use halo2curves::bn256::Fr as F;
use halo2curves::ff::{Field, PrimeField};
use ops::lookup::LookupOp;
use ops::region::RegionCtx;
use rand::rngs::OsRng;
use std::marker::PhantomData;

#[derive(Default)]
struct TestParams;

#[cfg(test)]
mod matmul {

    use super::*;

    const K: usize = 9;
    const LEN: usize = 3;

    #[derive(Clone)]
    struct MatmulCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MatmulCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN * LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN * LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN * LEN);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
                                Box::new(PolyOp::Einsum {
                                    equation: "ij,jk->ik".to_string(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();

            Ok(())
        }
    }

    #[test]
    fn matmulcircuit() {
        // parameters
        let mut a =
            Tensor::from((0..(LEN + 1) * LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        a.reshape(&[LEN, LEN + 1]).unwrap();

        let mut w = Tensor::from((0..LEN + 1).map(|i| Value::known(F::from((i + 1) as u64))));
        w.reshape(&[LEN + 1, 1]).unwrap();

        let circuit = MatmulCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(w)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied_par();
    }
}

#[cfg(test)]
mod matmul_col_overflow_double_col {
    use super::*;

    const K: usize = 5;
    const LEN: usize = 6;
    const NUM_INNER_COLS: usize = 2;

    #[derive(Clone)]
    struct MatmulCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MatmulCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN * LEN * LEN);
            let b = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN * LEN * LEN);
            let output = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN * LEN * LEN);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, NUM_INNER_COLS);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
                                Box::new(PolyOp::Einsum {
                                    equation: "ij,jk->ik".to_string(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn matmulcircuit() {
        // parameters
        let mut a = Tensor::from((0..LEN * LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        a.reshape(&[LEN, LEN]).unwrap();

        let mut w = Tensor::from((0..LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        w.reshape(&[LEN, 1]).unwrap();

        let circuit = MatmulCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(w)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied_par();
    }
}

#[cfg(test)]
mod matmul_col_overflow {
    use super::*;

    const K: usize = 5;
    const LEN: usize = 6;

    #[derive(Clone)]
    struct MatmulCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MatmulCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
                                Box::new(PolyOp::Einsum {
                                    equation: "ij,jk->ik".to_string(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn matmulcircuit() {
        // parameters
        let mut a = Tensor::from((0..LEN * LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        a.reshape(&[LEN, LEN]).unwrap();

        let mut w = Tensor::from((0..LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        w.reshape(&[LEN, 1]).unwrap();

        let circuit = MatmulCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(w)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied_par();
    }
}

#[cfg(test)]
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
mod matmul_col_ultra_overflow_double_col {
    use halo2_proofs::poly::commitment::ParamsProver;

    use super::*;

    const K: usize = 4;
    const LEN: usize = 20;
    const NUM_INNER_COLS: usize = 2;

    #[derive(Clone)]
    struct MatmulCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MatmulCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN * LEN * LEN);
            let b = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN * LEN * LEN);
            let output = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN * LEN * LEN);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, NUM_INNER_COLS);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
                                Box::new(PolyOp::Einsum {
                                    equation: "ij,jk->ik".to_string(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    #[ignore]
    fn matmulcircuit() {
        // get some logs fam
        crate::logger::init_logger();
        // parameters
        let mut a = Tensor::from((0..LEN * LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        a.reshape(&[LEN, LEN]).unwrap();

        let mut w = Tensor::from((0..LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        w.reshape(&[LEN, 1]).unwrap();

        let circuit = MatmulCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(w)],
            _marker: PhantomData,
        };

        let params = crate::pfsys::srs::gen_srs::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<_>,
        >(K as u32);

        let pk = crate::pfsys::create_keys::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<halo2curves::bn256::Bn256>,
            F,
            MatmulCircuit<F>,
        >(&circuit, &params)
        .unwrap();

        let prover = crate::pfsys::create_proof_circuit_kzg(
            circuit.clone(),
            &params,
            None,
            &pk,
            crate::pfsys::TranscriptType::EVM,
            halo2_proofs::poly::kzg::strategy::SingleStrategy::new(&params),
            // use safe mode to verify that the proof is correct
            CheckMode::SAFE,
            None,
        );

        assert!(prover.is_ok());

        let proof = prover.unwrap();

        let strategy =
            halo2_proofs::poly::kzg::strategy::SingleStrategy::new(params.verifier_params());
        let vk = pk.get_vk();
        let result =
            crate::pfsys::verify_proof_circuit_kzg(params.verifier_params(), proof, vk, strategy);

        assert!(result.is_ok());

        println!("done.");
    }
}

#[cfg(test)]
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
mod matmul_col_ultra_overflow {
    use halo2_proofs::poly::commitment::ParamsProver;

    use super::*;

    const K: usize = 4;
    const LEN: usize = 20;

    #[derive(Clone)]
    struct MatmulCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MatmulCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
                                Box::new(PolyOp::Einsum {
                                    equation: "ij,jk->ik".to_string(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    #[ignore]
    fn matmulcircuit() {
        // get some logs fam
        crate::logger::init_logger();
        // parameters
        let mut a = Tensor::from((0..LEN * LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        a.reshape(&[LEN, LEN]).unwrap();

        let mut w = Tensor::from((0..LEN).map(|i| Value::known(F::from((i + 1) as u64))));
        w.reshape(&[LEN, 1]).unwrap();

        let circuit = MatmulCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(w)],
            _marker: PhantomData,
        };

        let params = crate::pfsys::srs::gen_srs::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<_>,
        >(K as u32);

        let pk = crate::pfsys::create_keys::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<halo2curves::bn256::Bn256>,
            F,
            MatmulCircuit<F>,
        >(&circuit, &params)
        .unwrap();

        let prover = crate::pfsys::create_proof_circuit_kzg(
            circuit.clone(),
            &params,
            None,
            &pk,
            crate::pfsys::TranscriptType::EVM,
            halo2_proofs::poly::kzg::strategy::SingleStrategy::new(&params),
            // use safe mode to verify that the proof is correct
            CheckMode::SAFE,
            None,
        );

        assert!(prover.is_ok());

        let proof = prover.unwrap();

        let strategy =
            halo2_proofs::poly::kzg::strategy::SingleStrategy::new(params.verifier_params());
        let vk = pk.get_vk();
        let result =
            crate::pfsys::verify_proof_circuit_kzg(params.verifier_params(), proof, vk, strategy);

        assert!(result.is_ok());

        println!("done.");
    }
}

#[cfg(test)]
mod dot {
    use ops::poly::PolyOp;

    use super::*;

    const K: usize = 4;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
                                Box::new(PolyOp::Einsum {
                                    equation: "i,i->".to_string(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn dotcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied_par();
    }
}

#[cfg(test)]
mod dot_col_overflow_triple_col {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 50;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            // used for constants in the padding
            let _fixed = cs.fixed_column();
            cs.enable_constant(_fixed);

            let a = VarTensor::new_advice(cs, K, 3, LEN);
            let b = VarTensor::new_advice(cs, K, 3, LEN);
            let output = VarTensor::new_advice(cs, K, 3, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 3);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
                                Box::new(PolyOp::Einsum {
                                    equation: "i,i->".to_string(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn dotcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied_par();
    }
}

#[cfg(test)]
mod dot_col_overflow {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 50;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
                                Box::new(PolyOp::Einsum {
                                    equation: "i,i->".to_string(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn dotcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied_par();
    }
}

#[cfg(test)]
mod sum {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 1],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
                                Box::new(PolyOp::Sum { axes: vec![0] }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn sumcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied_par();
    }
}

#[cfg(test)]
mod sum_col_overflow_double_col {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 20;
    const NUM_INNER_COLS: usize = 2;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 1],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN);
            let b = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN);
            let output = VarTensor::new_advice(cs, K, NUM_INNER_COLS, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, NUM_INNER_COLS);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
                                Box::new(PolyOp::Sum { axes: vec![0] }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn sumcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied_par();
    }
}

#[cfg(test)]
mod sum_col_overflow {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 20;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 1],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
                                Box::new(PolyOp::Sum { axes: vec![0] }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn sumcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied_par();
    }
}

#[cfg(test)]
mod composition {

    use super::*;

    const K: usize = 9;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            // lots of stacked dot products
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1);
                        let _ = config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
                                Box::new(PolyOp::Einsum {
                                    equation: "i,i->".to_string(),
                                }),
                            )
                            .unwrap();
                        let _ = config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
                                Box::new(PolyOp::Einsum {
                                    equation: "i,i->".to_string(),
                                }),
                            )
                            .unwrap();
                        config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
                                Box::new(PolyOp::Einsum {
                                    equation: "i,i->".to_string(),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn dotcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied_par();
    }
}

#[cfg(test)]
mod conv {

    use super::*;

    const K: usize = 22;
    const LEN: usize = 100;

    #[derive(Clone)]
    struct ConvCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: Vec<Tensor<F>>,
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for ConvCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, (LEN + 1) * LEN);
            let b = VarTensor::new_advice(cs, K, 1, (LEN + 1) * LEN);
            let output = VarTensor::new_advice(cs, K, 1, (LEN + 1) * LEN);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1);
                        config
                            .layout(
                                &mut region,
                                &[self.inputs[0].clone().try_into().unwrap()],
                                Box::new(PolyOp::Conv {
                                    kernel: self.inputs[1].clone(),
                                    bias: None,
                                    padding: [(1, 1); 2],
                                    stride: (2, 2),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn convcircuit() {
        // parameters
        let kernel_height = 2;
        let kernel_width = 3;
        let image_height = 5;
        let image_width = 7;
        let in_channels = 3;
        let out_channels = 2;

        let mut image =
            Tensor::from((0..in_channels * image_height * image_width).map(|_| F::random(OsRng)));
        image
            .reshape(&[1, in_channels, image_height, image_width])
            .unwrap();
        image.set_visibility(&crate::graph::Visibility::Private);

        let mut kernels = Tensor::from(
            (0..{ out_channels * in_channels * kernel_height * kernel_width })
                .map(|_| F::random(OsRng)),
        );
        kernels
            .reshape(&[out_channels, in_channels, kernel_height, kernel_width])
            .unwrap();
        kernels.set_visibility(&crate::graph::Visibility::Private);

        let mut bias = Tensor::from((0..{ out_channels }).map(|_| F::random(OsRng)));
        bias.set_visibility(&crate::graph::Visibility::Private);

        let circuit = ConvCircuit::<F> {
            inputs: [image, kernels, bias].to_vec(),
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied_par();
    }

    #[test]
    fn convcircuitnobias() {
        // parameters
        let kernel_height = 2;
        let kernel_width = 2;
        let image_height = 4;
        let image_width = 5;
        let in_channels = 3;
        let out_channels = 2;

        let mut image =
            Tensor::from((0..in_channels * image_height * image_width).map(|i| F::from(i as u64)));
        image
            .reshape(&[1, in_channels, image_height, image_width])
            .unwrap();
        image.set_visibility(&crate::graph::Visibility::Private);

        let mut kernels = Tensor::from(
            (0..{ out_channels * in_channels * kernel_height * kernel_width })
                .map(|i| F::from(i as u64)),
        );
        kernels
            .reshape(&[out_channels, in_channels, kernel_height, kernel_width])
            .unwrap();
        kernels.set_visibility(&crate::graph::Visibility::Private);

        let circuit = ConvCircuit::<F> {
            inputs: [image, kernels].to_vec(),
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied_par();
    }
}

#[cfg(test)]
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
mod conv_col_ultra_overflow {
    use halo2_proofs::poly::commitment::ParamsProver;

    use super::*;

    const K: usize = 4;
    const LEN: usize = 28;

    #[derive(Clone)]
    struct ConvCircuit<F: PrimeField + TensorType + PartialOrd> {
        image: ValTensor<F>,
        kernel: Tensor<F>,
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for ConvCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1);
                        config
                            .layout(
                                &mut region,
                                &[self.image.clone()],
                                Box::new(PolyOp::Conv {
                                    kernel: self.kernel.clone(),
                                    bias: None,
                                    padding: [(1, 1); 2],
                                    stride: (2, 2),
                                }),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    #[ignore]
    fn conv_circuit() {
        // parameters
        let kernel_height = 2;
        let kernel_width = 2;
        let image_height = LEN;
        let image_width = LEN;
        let in_channels = 3;
        let out_channels = 2;

        // get some logs fam
        crate::logger::init_logger();
        let mut image =
            Tensor::from((0..in_channels * image_height * image_width).map(|i| F::from(i as u64)));
        image
            .reshape(&[1, in_channels, image_height, image_width])
            .unwrap();
        image.set_visibility(&crate::graph::Visibility::Private);

        let mut kernels = Tensor::from(
            (0..{ out_channels * in_channels * kernel_height * kernel_width })
                .map(|i| F::from(i as u64)),
        );
        kernels
            .reshape(&[out_channels, in_channels, kernel_height, kernel_width])
            .unwrap();
        kernels.set_visibility(&crate::graph::Visibility::Private);

        let circuit = ConvCircuit::<F> {
            image: ValTensor::try_from(image).unwrap(),
            kernel: kernels,
            _marker: PhantomData,
        };

        let params = crate::pfsys::srs::gen_srs::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<_>,
        >(K as u32);

        let pk = crate::pfsys::create_keys::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<halo2curves::bn256::Bn256>,
            F,
            ConvCircuit<F>,
        >(&circuit, &params)
        .unwrap();

        let prover = crate::pfsys::create_proof_circuit_kzg(
            circuit.clone(),
            &params,
            None,
            &pk,
            crate::pfsys::TranscriptType::EVM,
            halo2_proofs::poly::kzg::strategy::SingleStrategy::new(&params),
            // use safe mode to verify that the proof is correct
            CheckMode::SAFE,
            None,
        );

        assert!(prover.is_ok());

        let proof = prover.unwrap();

        let strategy =
            halo2_proofs::poly::kzg::strategy::SingleStrategy::new(params.verifier_params());
        let vk = pk.get_vk();
        let result =
            crate::pfsys::verify_proof_circuit_kzg(params.verifier_params(), proof, vk, strategy);

        assert!(result.is_ok());

        println!("done.");
    }
}

#[cfg(test)]
// not wasm 32 unknown
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
mod conv_relu_col_ultra_overflow {
    use halo2_proofs::poly::commitment::ParamsProver;

    use super::*;

    const K: usize = 4;
    const LEN: usize = 28;

    #[derive(Clone)]
    struct ConvCircuit<F: PrimeField + TensorType + PartialOrd> {
        image: ValTensor<F>,
        kernel: Tensor<F>,
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for ConvCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN * LEN * LEN);
            let mut base_config =
                Self::Config::configure(cs, &[a.clone(), b.clone()], &output, CheckMode::SAFE);
            // sets up a new relu table
            base_config
                .configure_lookup(cs, &b, &output, &a, (-3, 3), K, &LookupOp::ReLU)
                .unwrap();
            base_config.clone()
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            config.layout_tables(&mut layouter).unwrap();
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1);
                        let output = config
                            .layout(
                                &mut region,
                                &[self.image.clone()],
                                Box::new(PolyOp::Conv {
                                    kernel: self.kernel.clone(),
                                    bias: None,
                                    padding: [(1, 1); 2],
                                    stride: (2, 2),
                                }),
                            )
                            .map_err(|_| Error::Synthesis);
                        let _output = config
                            .layout(
                                &mut region,
                                &[output.unwrap().unwrap()],
                                Box::new(LookupOp::ReLU),
                            )
                            .unwrap();
                        Ok(())
                    },
                )
                .unwrap();

            Ok(())
        }
    }

    #[test]
    #[ignore]
    fn conv_relu_circuit() {
        // parameters
        let kernel_height = 2;
        let kernel_width = 2;
        let image_height = LEN;
        let image_width = LEN;
        let in_channels = 3;
        let out_channels = 2;

        // get some logs fam
        crate::logger::init_logger();
        let mut image =
            Tensor::from((0..in_channels * image_height * image_width).map(|_| F::from(0)));
        image
            .reshape(&[1, in_channels, image_height, image_width])
            .unwrap();
        image.set_visibility(&crate::graph::Visibility::Private);

        let mut kernels = Tensor::from(
            (0..{ out_channels * in_channels * kernel_height * kernel_width }).map(|_| F::from(0)),
        );
        kernels
            .reshape(&[out_channels, in_channels, kernel_height, kernel_width])
            .unwrap();
        kernels.set_visibility(&crate::graph::Visibility::Private);

        let circuit = ConvCircuit::<F> {
            image: ValTensor::try_from(image).unwrap(),
            kernel: kernels,
            _marker: PhantomData,
        };

        let params = crate::pfsys::srs::gen_srs::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<_>,
        >(K as u32);

        let pk = crate::pfsys::create_keys::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<halo2curves::bn256::Bn256>,
            F,
            ConvCircuit<F>,
        >(&circuit, &params)
        .unwrap();

        let prover = crate::pfsys::create_proof_circuit_kzg(
            circuit.clone(),
            &params,
            None,
            &pk,
            crate::pfsys::TranscriptType::EVM,
            halo2_proofs::poly::kzg::strategy::SingleStrategy::new(&params),
            // use safe mode to verify that the proof is correct
            CheckMode::SAFE,
            None,
        );

        assert!(prover.is_ok());

        let proof = prover.unwrap();

        let strategy =
            halo2_proofs::poly::kzg::strategy::SingleStrategy::new(params.verifier_params());
        let vk = pk.get_vk();
        let result =
            crate::pfsys::verify_proof_circuit_kzg(params.verifier_params(), proof, vk, strategy);

        assert!(result.is_ok());

        println!("done.");
    }
}

#[cfg(test)]
mod add_w_shape_casting {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1);
                        config
                            .layout(&mut region, &self.inputs.clone(), Box::new(PolyOp::Add))
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn addcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..1).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied_par();
    }
}

#[cfg(test)]
mod add {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1);
                        config
                            .layout(&mut region, &self.inputs.clone(), Box::new(PolyOp::Add))
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn addcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied_par();
    }
}

#[cfg(test)]
mod add_with_overflow {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 50;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1);
                        config
                            .layout(&mut region, &self.inputs.clone(), Box::new(PolyOp::Add))
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn addcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied_par();
    }
}

#[cfg(test)]
mod add_with_overflow_and_poseidon {
    use halo2curves::bn256::Fr;

    use crate::circuit::modules::{
        poseidon::{
            spec::{PoseidonSpec, POSEIDON_RATE, POSEIDON_WIDTH},
            PoseidonChip, PoseidonConfig,
        },
        Module, ModulePlanner,
    };

    use super::*;

    const K: usize = 15;
    const LEN: usize = 50;
    const WIDTH: usize = POSEIDON_WIDTH;
    const RATE: usize = POSEIDON_RATE;

    #[derive(Debug, Clone)]
    struct MyCircuitConfig {
        base: BaseConfig<Fr>,
        poseidon: PoseidonConfig<WIDTH, RATE>,
    }

    #[derive(Clone)]
    struct MyCircuit {
        inputs: [ValTensor<Fr>; 2],
    }

    impl Circuit<Fr> for MyCircuit {
        type Config = MyCircuitConfig;
        type FloorPlanner = ModulePlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<Fr>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            let base = BaseConfig::configure(cs, &[a, b], &output, CheckMode::SAFE);
            VarTensor::constant_cols(cs, K, 2, false);

            let poseidon = PoseidonChip::<PoseidonSpec, WIDTH, RATE, WIDTH>::configure(cs, ());

            MyCircuitConfig { base, poseidon }
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<Fr>,
        ) -> Result<(), Error> {
            let poseidon_chip: PoseidonChip<PoseidonSpec, WIDTH, RATE, WIDTH> =
                PoseidonChip::new(config.poseidon.clone());

            let assigned_inputs_a = poseidon_chip.layout(&mut layouter, &self.inputs[0..1], 0)?;
            let assigned_inputs_b = poseidon_chip.layout(&mut layouter, &self.inputs[1..2], 1)?;

            layouter.assign_region(|| "_new_module", |_| Ok(()))?;

            let inputs = vec![assigned_inputs_a, assigned_inputs_b];

            layouter.assign_region(
                || "model",
                |region| {
                    let mut region = RegionCtx::new(region, 0, 1);
                    config
                        .base
                        .layout(&mut region, &inputs, Box::new(PolyOp::Add))
                        .map_err(|_| Error::Synthesis)
                },
            )?;

            Ok(())
        }
    }

    #[test]
    fn addcircuit() {
        let a = (0..LEN)
            .map(|i| halo2curves::bn256::Fr::from(i as u64 + 1))
            .collect::<Vec<_>>();
        let b = (0..LEN)
            .map(|i| halo2curves::bn256::Fr::from(i as u64 + 1))
            .collect::<Vec<_>>();
        let commitment_a =
            PoseidonChip::<PoseidonSpec, WIDTH, RATE, WIDTH>::run(a.clone()).unwrap()[0][0];

        let commitment_b =
            PoseidonChip::<PoseidonSpec, WIDTH, RATE, WIDTH>::run(b.clone()).unwrap()[0][0];

        // parameters
        let a = Tensor::from(a.into_iter().map(Value::known));
        let b = Tensor::from(b.into_iter().map(Value::known));
        let circuit = MyCircuit {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
        };

        let prover =
            MockProver::run(K as u32, &circuit, vec![vec![commitment_a, commitment_b]]).unwrap();
        prover.assert_satisfied_par();
    }

    #[test]
    fn addcircuit_bad_hashes() {
        let a = (0..LEN)
            .map(|i| halo2curves::bn256::Fr::from(i as u64 + 1))
            .collect::<Vec<_>>();
        let b = (0..LEN)
            .map(|i| halo2curves::bn256::Fr::from(i as u64 + 1))
            .collect::<Vec<_>>();
        let commitment_a = PoseidonChip::<PoseidonSpec, WIDTH, RATE, WIDTH>::run(a.clone())
            .unwrap()[0][0]
            + Fr::one();

        let commitment_b = PoseidonChip::<PoseidonSpec, WIDTH, RATE, WIDTH>::run(b.clone())
            .unwrap()[0][0]
            + Fr::one();

        // parameters
        let a = Tensor::from(a.into_iter().map(Value::known));
        let b = Tensor::from(b.into_iter().map(Value::known));
        let circuit = MyCircuit {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
        };

        let prover =
            MockProver::run(K as u32, &circuit, vec![vec![commitment_a, commitment_b]]).unwrap();
        assert!(prover.verify().is_err());
    }
}

#[cfg(test)]
mod sub {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1);
                        config
                            .layout(&mut region, &self.inputs.clone(), Box::new(PolyOp::Sub))
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn subcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied_par();
    }
}

#[cfg(test)]
mod mult {
    use super::*;

    const K: usize = 4;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1);
                        config
                            .layout(&mut region, &self.inputs.clone(), Box::new(PolyOp::Mult))
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn multcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let b = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied_par();
    }
}

#[cfg(test)]
mod pow {
    use super::*;

    const K: usize = 8;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 1],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1);
                        config
                            .layout(&mut region, &self.inputs.clone(), Box::new(PolyOp::Pow(5)))
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn powcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied_par();
    }
}

#[cfg(test)]
mod pack {
    use super::*;

    const K: usize = 8;
    const LEN: usize = 4;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 1],
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE)
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1);
                        config
                            .layout(
                                &mut region,
                                &self.inputs.clone(),
                                Box::new(PolyOp::Pack(2, 1)),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    fn packcircuit() {
        // parameters
        let a = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = MyCircuit::<F> {
            inputs: [ValTensor::from(a)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied_par();
    }
}

#[cfg(test)]
mod matmul_relu {
    use super::*;

    const K: usize = 18;
    const LEN: usize = 32;
    use crate::circuit::lookup::LookupOp;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        inputs: [ValTensor<F>; 2],
        _marker: PhantomData<F>,
    }

    // A columnar ReLu MLP
    #[derive(Clone)]
    struct MyConfig<F: PrimeField + TensorType + PartialOrd> {
        base_config: BaseConfig<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = MyConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);

            let mut base_config =
                BaseConfig::configure(cs, &[a.clone(), b.clone()], &output, CheckMode::SAFE);
            // sets up a new relu table
            base_config
                .configure_lookup(cs, &b, &output, &a, (-32768, 32768), K, &LookupOp::ReLU)
                .unwrap();

            MyConfig { base_config }
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            config.base_config.layout_tables(&mut layouter).unwrap();
            layouter.assign_region(
                || "",
                |region| {
                    let mut region = RegionCtx::new(region, 0, 1);
                    let op = PolyOp::Einsum {
                        equation: "ij,jk->ik".to_string(),
                    };
                    let output = config
                        .base_config
                        .layout(&mut region, &self.inputs, Box::new(op))
                        .unwrap();
                    let _output = config
                        .base_config
                        .layout(&mut region, &[output.unwrap()], Box::new(LookupOp::ReLU))
                        .unwrap();
                    Ok(())
                },
            )?;

            Ok(())
        }
    }

    #[test]
    fn matmulrelucircuit() {
        // parameters
        let mut a = Tensor::from((0..LEN * LEN).map(|_| Value::known(F::from(1))));
        a.reshape(&[LEN, LEN]).unwrap();

        // parameters
        let mut b = Tensor::from((0..LEN).map(|_| Value::known(F::from(1))));
        b.reshape(&[LEN, 1]).unwrap();

        let circuit = MyCircuit {
            inputs: [ValTensor::from(a), ValTensor::from(b)],
            _marker: PhantomData,
        };

        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied_par();
    }
}

#[cfg(test)]
mod rangecheckpercent {
    use crate::circuit::Tolerance;
    use crate::{circuit, tensor::Tensor};
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        plonk::{Circuit, ConstraintSystem, Error},
    };

    const RANGE: f32 = 1.0; // 1 percent error tolerance
    const K: usize = 18;
    const LEN: usize = 1;
    const SCALE: usize = i128::pow(2, 7) as usize;

    use super::*;

    #[derive(Clone)]
    struct MyCircuit<F: PrimeField + TensorType + PartialOrd> {
        input: ValTensor<F>,
        output: ValTensor<F>,
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for MyCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let scale = utils::F32(SCALE.pow(2) as f32);
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);
            let mut config =
                Self::Config::configure(cs, &[a.clone(), b.clone()], &output, CheckMode::SAFE);
            // set up a new GreaterThan and Recip tables
            let nl = &LookupOp::GreaterThan {
                a: circuit::utils::F32((RANGE * scale.0) / 100.0),
            };
            config
                .configure_lookup(cs, &b, &output, &a, (-32768, 32768), K, nl)
                .unwrap();
            config
                .configure_lookup(
                    cs,
                    &b,
                    &output,
                    &a,
                    (-32768, 32768),
                    K,
                    &LookupOp::Recip { scale },
                )
                .unwrap();
            config
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            config.layout_tables(&mut layouter).unwrap();
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1);
                        config
                            .layout(
                                &mut region,
                                &[self.output.clone(), self.input.clone()],
                                Box::new(HybridOp::RangeCheck(Tolerance {
                                    val: RANGE,
                                    scale: SCALE.into(),
                                })),
                            )
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();
            Ok(())
        }
    }

    #[test]
    #[allow(clippy::assertions_on_constants)]
    fn test_range_check_percent() {
        // Successful cases
        {
            let inp = Tensor::new(Some(&[Value::<F>::known(F::from(100_u64))]), &[1]).unwrap();
            let out = Tensor::new(Some(&[Value::<F>::known(F::from(101_u64))]), &[1]).unwrap();
            let circuit = MyCircuit::<F> {
                input: ValTensor::from(inp),
                output: ValTensor::from(out),
                _marker: PhantomData,
            };
            let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
            prover.assert_satisfied_par();
        }
        {
            let inp = Tensor::new(Some(&[Value::<F>::known(F::from(200_u64))]), &[1]).unwrap();
            let out = Tensor::new(Some(&[Value::<F>::known(F::from(199_u64))]), &[1]).unwrap();
            let circuit = MyCircuit::<F> {
                input: ValTensor::from(inp),
                output: ValTensor::from(out),
                _marker: PhantomData,
            };
            let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
            prover.assert_satisfied_par();
        }

        // Unsuccessful case
        {
            let inp = Tensor::new(Some(&[Value::<F>::known(F::from(100_u64))]), &[1]).unwrap();
            let out = Tensor::new(Some(&[Value::<F>::known(F::from(102_u64))]), &[1]).unwrap();
            let circuit = MyCircuit::<F> {
                input: ValTensor::from(inp),
                output: ValTensor::from(out),
                _marker: PhantomData,
            };
            let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
            match prover.verify() {
                Ok(_) => {
                    assert!(false)
                }
                Err(_) => {
                    assert!(true)
                }
            }
        }
    }
}

#[cfg(test)]
mod relu {
    use super::*;
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        plonk::{Circuit, ConstraintSystem, Error},
    };

    #[derive(Clone)]
    struct ReLUCircuit<F: PrimeField + TensorType + PartialOrd> {
        pub input: ValTensor<F>,
    }

    impl Circuit<F> for ReLUCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let advices = (0..3)
                .map(|_| VarTensor::new_advice(cs, 4, 1, 3))
                .collect::<Vec<_>>();

            let nl = LookupOp::ReLU;

            let mut config = BaseConfig::default();

            config
                .configure_lookup(cs, &advices[0], &advices[1], &advices[2], (-6, 6), 4, &nl)
                .unwrap();
            config
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>, // layouter is our 'write buffer' for the circuit
        ) -> Result<(), Error> {
            config.layout_tables(&mut layouter).unwrap();
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1);
                        config
                            .layout(&mut region, &[self.input.clone()], Box::new(LookupOp::ReLU))
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();

            Ok(())
        }
    }

    #[test]
    fn relucircuit() {
        let input: Tensor<Value<F>> =
            Tensor::new(Some(&[Value::<F>::known(F::from(1_u64)); 4]), &[4]).unwrap();

        let circuit = ReLUCircuit::<F> {
            input: ValTensor::from(input),
        };

        let prover = MockProver::run(4_u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied_par();
    }
}

#[cfg(test)]
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
mod lookup_ultra_overflow {
    use super::*;
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        plonk::{Circuit, ConstraintSystem, Error},
        poly::commitment::ParamsProver,
    };

    #[derive(Clone)]
    struct ReLUCircuit<F: PrimeField + TensorType + PartialOrd> {
        pub input: ValTensor<F>,
    }

    impl Circuit<F> for ReLUCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }

        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let advices = (0..3)
                .map(|_| VarTensor::new_advice(cs, 4, 1, 3))
                .collect::<Vec<_>>();

            let nl = LookupOp::ReLU;

            let mut config = BaseConfig::default();

            config
                .configure_lookup(
                    cs,
                    &advices[0],
                    &advices[1],
                    &advices[2],
                    (-1024, 1024),
                    4,
                    &nl,
                )
                .unwrap();
            config
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>, // layouter is our 'write buffer' for the circuit
        ) -> Result<(), Error> {
            config.layout_tables(&mut layouter).unwrap();
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1);
                        config
                            .layout(&mut region, &[self.input.clone()], Box::new(LookupOp::ReLU))
                            .map_err(|_| Error::Synthesis)
                    },
                )
                .unwrap();

            Ok(())
        }
    }

    #[test]
    #[ignore]
    fn relucircuit() {
        // get some logs fam
        crate::logger::init_logger();
        // parameters
        let a = Tensor::from((0..4).map(|i| Value::known(F::from(i + 1))));

        let circuit = ReLUCircuit::<F> {
            input: ValTensor::from(a),
        };

        let params = crate::pfsys::srs::gen_srs::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<_>,
        >(4_u32);

        let pk = crate::pfsys::create_keys::<
            halo2_proofs::poly::kzg::commitment::KZGCommitmentScheme<halo2curves::bn256::Bn256>,
            F,
            ReLUCircuit<F>,
        >(&circuit, &params)
        .unwrap();

        let prover = crate::pfsys::create_proof_circuit_kzg(
            circuit.clone(),
            &params,
            None,
            &pk,
            crate::pfsys::TranscriptType::EVM,
            halo2_proofs::poly::kzg::strategy::SingleStrategy::new(&params),
            // use safe mode to verify that the proof is correct
            CheckMode::SAFE,
            None,
        );

        assert!(prover.is_ok());

        let proof = prover.unwrap();

        let strategy =
            halo2_proofs::poly::kzg::strategy::SingleStrategy::new(params.verifier_params());
        let vk = pk.get_vk();
        let result =
            crate::pfsys::verify_proof_circuit_kzg(params.verifier_params(), proof, vk, strategy);

        assert!(result.is_ok());

        println!("done.");
    }
}

#[cfg(test)]
mod softmax {

    use super::*;
    use halo2_proofs::{
        circuit::{Layouter, SimpleFloorPlanner, Value},
        dev::MockProver,
        plonk::{Circuit, ConstraintSystem, Error},
    };

    const K: usize = 18;
    const LEN: usize = 3;
    const SCALE: f32 = 128.0;

    #[derive(Clone)]
    struct SoftmaxCircuit<F: PrimeField + TensorType + PartialOrd> {
        pub input: ValTensor<F>,
        _marker: PhantomData<F>,
    }

    impl Circuit<F> for SoftmaxCircuit<F> {
        type Config = BaseConfig<F>;
        type FloorPlanner = SimpleFloorPlanner;
        type Params = TestParams;

        fn without_witnesses(&self) -> Self {
            self.clone()
        }
        fn configure(cs: &mut ConstraintSystem<F>) -> Self::Config {
            let a = VarTensor::new_advice(cs, K, 1, LEN);
            let b = VarTensor::new_advice(cs, K, 1, LEN);
            let output = VarTensor::new_advice(cs, K, 1, LEN);
            let mut config = Self::Config::configure(cs, &[a, b], &output, CheckMode::SAFE);
            let advices = (0..3)
                .map(|_| VarTensor::new_advice(cs, K, 1, LEN))
                .collect::<Vec<_>>();

            config
                .configure_lookup(
                    cs,
                    &advices[0],
                    &advices[1],
                    &advices[2],
                    (-32768, 32768),
                    K,
                    &LookupOp::Exp {
                        scale: SCALE.into(),
                    },
                )
                .unwrap();
            config
                .configure_lookup(
                    cs,
                    &advices[0],
                    &advices[1],
                    &advices[2],
                    (-32768, 32768),
                    K,
                    &LookupOp::Recip {
                        scale: SCALE.powf(2.0).into(),
                    },
                )
                .unwrap();
            config
        }

        fn synthesize(
            &self,
            mut config: Self::Config,
            mut layouter: impl Layouter<F>,
        ) -> Result<(), Error> {
            config.layout_tables(&mut layouter).unwrap();
            layouter
                .assign_region(
                    || "",
                    |region| {
                        let mut region = RegionCtx::new(region, 0, 1);
                        let _output = config
                            .layout(
                                &mut region,
                                &[self.input.clone()],
                                Box::new(HybridOp::Softmax {
                                    scale: SCALE.into(),
                                    axes: vec![0],
                                }),
                            )
                            .unwrap();
                        Ok(())
                    },
                )
                .unwrap();

            Ok(())
        }
    }

    #[test]
    fn softmax_circuit() {
        let input = Tensor::from((0..LEN).map(|i| Value::known(F::from(i as u64 + 1))));

        let circuit = SoftmaxCircuit::<F> {
            input: ValTensor::from(input),
            _marker: PhantomData,
        };
        let prover = MockProver::run(K as u32, &circuit, vec![]).unwrap();
        prover.assert_satisfied_par();
    }
}
