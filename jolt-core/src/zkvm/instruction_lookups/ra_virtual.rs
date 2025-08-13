use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};
use tracer::instruction::RV32IMCycle;

use crate::{
    field::{JoltField, OptimizedMul},
    poly::{
        commitment::commitment_scheme::CommitmentScheme,
        eq_poly::EqPolynomial,
        multilinear_polynomial::{
            BindingOrder, MultilinearPolynomial, PolynomialBinding, PolynomialEvaluation,
        },
        opening_proof::{OpeningPoint, SumcheckId, BIG_ENDIAN},
    },
    subprotocols::{
        optimization::{compute_eq_mle_product_univariate, compute_mle_product_coeffs_katatsuba},
        sumcheck::SumcheckInstance,
    },
    utils::{
        expanding_table::ExpandingTable, lookup_bits::LookupBits, math::Math,
        transcript::Transcript,
    },
    zkvm::{
        dag::state_manager::StateManager,
        instruction::LookupQuery,
        instruction_lookups::{
            K_CHUNK, LOG_K, LOG_K_CHUNK, LOG_M, M, PHASES, RA_PER_LOG_M, WORD_SIZE,
        },
        ram::remap_address,
        witness::{
            compute_d_parameter_from_log_K, CommittedPolynomial, VirtualPolynomial, DTH_ROOT_OF_K,
        },
    },
};

pub struct RASumCheck<F: JoltField> {
    r_cycle: Vec<F>,
    r_address_chunks: Vec<Vec<F>>,
    eq_ra_claim: F,
    d: usize,
    T: usize,
    prover_state: Option<RAProverState<F>>,
}

pub struct RAProverState<F: JoltField> {
    ra_i_polys: Vec<MultilinearPolynomial<F>>,
    E_table: Vec<Vec<F>>,
    eq_factor: F,
}

impl<F: JoltField> RASumCheck<F> {
    fn compute_ra_i_polys(
        d: usize,
        trace: &[RV32IMCycle],
        state_manager: &StateManager<'_, F, impl Transcript, impl CommitmentScheme<Field = F>>,
    ) -> Vec<MultilinearPolynomial<F>> {
        // Compute ra_i_polys for each i
        let mut ra_i_polys: Vec<MultilinearPolynomial<F>> = Vec::with_capacity(d);

        // TODO: This really needs cleaning up. For now, I follow the relevant code in instruction read_raf_checking.rs that materializes ra as the address variables are bounded.
        let lookup_indices: Vec<_> = trace
            .par_iter()
            .map(|cycle| LookupBits::new(LookupQuery::<WORD_SIZE>::to_lookup_index(cycle), LOG_K))
            .collect();

        // Retrieve the random address variables generated in ReadRafSumcheck.
        let (r, _claim) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionEqRa,
            SumcheckId::InstructionReadRaf,
        );
        let (r_address, _r_cycle) = r.split_at_r(LOG_K);

        // Recreate the ExpandingTables from ReadRafSumcheck.
        let mut v: [ExpandingTable<_>; RA_PER_LOG_M] =
            std::array::from_fn(|_| ExpandingTable::<F>::new(K_CHUNK));
        v.iter_mut().for_each(|v| v.reset(F::one()));

        assert!(r_address.len().is_multiple_of(LOG_M));

        for (round, r_j) in r_address.iter().enumerate() {
            v[(round % LOG_M) / LOG_K_CHUNK].update(*r_j);

            // If this is the last round in the phase.
            if (round + 1).is_multiple_of(LOG_M) {
                let phase = round / LOG_M;
                // Materialize two RAs.
                let new_ra_arr = v
                    .par_iter()
                    .enumerate()
                    .map(|(i, v)| {
                        let ra = lookup_indices
                            .par_iter()
                            .map(|k| {
                                let (prefix, _) = k.split((PHASES - 1 - phase) * LOG_M);
                                let k_bound: usize = ((prefix % M)
                                    >> (LOG_K_CHUNK * (RA_PER_LOG_M - 1 - i)))
                                    % K_CHUNK;
                                v[k_bound]
                            })
                            .collect::<Vec<F>>();
                        MultilinearPolynomial::from(ra)
                    })
                    .collect::<Vec<_>>();

                ra_i_polys.extend(new_ra_arr);

                v.iter_mut().for_each(|v| v.reset(F::one()));
            }
        }

        println!(
            "ra_i_polys first ele: {:?}",
            ra_i_polys
                .iter()
                .map(|p| p.get_coeff(0))
                .collect::<Vec<_>>()
        );
        println!(
            "ra_i_polys 3rd ele: {:?}",
            ra_i_polys
                .iter()
                .map(|p| p.get_coeff(2))
                .collect::<Vec<_>>()
        );
        println!(
            "ra_i_polys last ele: {:?}",
            ra_i_polys
                .iter()
                .map(|p| p.get_coeff(p.len() - 1))
                .collect::<Vec<_>>()
        );
        println!(
            "ra_i_polys num vars: {:?}",
            ra_i_polys
                .iter()
                .map(|p| p.get_num_vars())
                .collect::<Vec<_>>()
        );
        println!(
            "ra_i_polys evaluate at 1: {:?}",
            ra_i_polys
                .iter()
                .map(|p| p.evaluate(&(0..p.get_num_vars()).map(|_| F::one()).collect::<Vec<_>>()))
                .collect::<Vec<_>>()
        );

        ra_i_polys
    }

    pub fn new_prover<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        log_K: usize,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let d = compute_d_parameter_from_log_K(log_K);

        let (preprocessing, trace, _, _) = state_manager.get_prover_data();
        let T = trace.len();

        let (r, ra_claim) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionEqRa,
            SumcheckId::InstructionReadRaf,
        );

        let (r_address, r_cycle) = r.split_at_r(log_K);
        let r_address = if r_address.len().is_multiple_of(DTH_ROOT_OF_K.log_2()) {
            r_address.to_vec()
        } else {
            // Pad with zeros
            [
                &vec![F::zero(); DTH_ROOT_OF_K.log_2() - (r_address.len() % DTH_ROOT_OF_K.log_2())],
                r_address,
            ]
            .concat()
        };

        // Split r_address into d chunks of variable sizes
        let r_address_chunks: Vec<Vec<F>> = r_address
            .chunks(DTH_ROOT_OF_K.log_2())
            .map(|chunk| chunk.to_vec())
            .collect();
        debug_assert_eq!(r_address_chunks.len(), d);

        let ra_i_polys = Self::compute_ra_i_polys(d, trace, state_manager);
        let E_table = (1..=T.log_2() - 1)
            .map(|i| EqPolynomial::evals(&r_cycle[i..]))
            .collect::<Vec<_>>();
        let prover_state = RAProverState {
            ra_i_polys,
            E_table,
            eq_factor: F::one(),
        };

        Self {
            r_cycle: r_cycle.to_vec(),
            r_address_chunks,
            eq_ra_claim: ra_claim,
            d,
            T,
            prover_state: Some(prover_state),
        }
    }

    pub fn new_verifier<ProofTranscript: Transcript, PCS: CommitmentScheme<Field = F>>(
        log_K: usize,
        state_manager: &mut StateManager<'_, F, ProofTranscript, PCS>,
    ) -> Self {
        let d = compute_d_parameter_from_log_K(log_K);

        let (_, _, T) = state_manager.get_verifier_data();

        let (r, eq_ra_claim) = state_manager.get_virtual_polynomial_opening(
            VirtualPolynomial::InstructionEqRa,
            SumcheckId::InstructionRaVirtualization,
        );

        // TODO: create a helper function for this
        let (r_address, r_cycle) = r.split_at_r(log_K);

        let r_address = if r_address.len().is_multiple_of(DTH_ROOT_OF_K.log_2()) {
            r_address.to_vec()
        } else {
            // Pad with zeros
            [
                &vec![F::zero(); DTH_ROOT_OF_K.log_2() - (r_address.len() % DTH_ROOT_OF_K.log_2())],
                r_address,
            ]
            .concat()
        };
        // Split r_address into d chunks of variable sizes
        let r_address_chunks: Vec<Vec<F>> = r_address
            .chunks(DTH_ROOT_OF_K.log_2())
            .map(|chunk| chunk.to_vec())
            .collect();
        debug_assert_eq!(r_address_chunks.len(), d);

        Self {
            r_cycle: r_cycle.to_vec(),
            r_address_chunks,
            eq_ra_claim,
            d,
            T,
            prover_state: None,
        }
    }
}

impl<F: JoltField> SumcheckInstance<F> for RASumCheck<F> {
    fn degree(&self) -> usize {
        self.d + 1
    }

    fn num_rounds(&self) -> usize {
        self.T.log_2()
    }

    fn input_claim(&self) -> F {
        self.eq_ra_claim
    }

    fn compute_prover_message(&mut self, round: usize, _previous_claim: F) -> Vec<F> {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");
        let ra_i_polys = &prover_state.ra_i_polys;

        // TODO: we should really use Toom-Cook for d = 4 and 8 but that requiers F to implement the SmallFieldMul trait. Need to rethink the interface.
        let mle_product_coeffs = match self.d {
            4 => compute_mle_product_coeffs_katatsuba::<F, 4, 5>(
                ra_i_polys,
                round,
                self.T.log_2(),
                &prover_state.eq_factor,
                &prover_state.E_table,
            ),
            8 => compute_mle_product_coeffs_katatsuba::<F, 8, 9>(
                ra_i_polys,
                round,
                self.T.log_2(),
                &prover_state.eq_factor,
                &prover_state.E_table,
            ),
            16 => compute_mle_product_coeffs_katatsuba::<F, 16, 17>(
                ra_i_polys,
                round,
                self.T.log_2(),
                &prover_state.eq_factor,
                &prover_state.E_table,
            ),
            _ => panic!(
                "Unsupported number of polynomials, got {} and expected 4, 8, or 16",
                self.d
            ),
        };

        let univariate_poly =
            compute_eq_mle_product_univariate(mle_product_coeffs, round, &self.r_cycle);

        // Turning into eval points.
        (0..univariate_poly.coeffs.len())
            .map(|i| univariate_poly.evaluate(&F::from_u32(i as u32)))
            .collect()
    }

    fn bind(&mut self, r_j: F, round: usize) {
        let prover_state = self
            .prover_state
            .as_mut()
            .expect("Prover state not initialized");

        prover_state
            .ra_i_polys
            .par_iter_mut()
            .for_each(|poly| poly.bind_parallel(r_j, BindingOrder::HighToLow));

        prover_state.eq_factor = prover_state.eq_factor.mul_1_optimized(
            (self.r_cycle[round] + self.r_cycle[round] - F::one()) * r_j
                + (F::one() - self.r_cycle[round]),
        );
    }

    fn expected_output_claim(
        &self,
        opening_accumulator: Option<
            std::rc::Rc<
                std::cell::RefCell<crate::poly::opening_proof::VerifierOpeningAccumulator<F>>,
            >,
        >,
        r: &[F],
    ) -> F {
        let eq_eval = EqPolynomial::mle(&self.r_cycle, r);
        let ra_claim_prod: F = (0..self.d)
            .map(|i| {
                let (_, ra_i_claim) = opening_accumulator
                    .as_ref()
                    .unwrap()
                    .borrow()
                    .get_committed_polynomial_opening(
                        CommittedPolynomial::InstructionRa(i),
                        SumcheckId::InstructionRaVirtualization,
                    );
                ra_i_claim
            })
            .product();

        panic!("ra_claim_prod is {:?}", ra_claim_prod);
        eq_eval * ra_claim_prod
    }

    fn normalize_opening_point(&self, opening_point: &[F]) -> OpeningPoint<BIG_ENDIAN, F> {
        OpeningPoint::new(opening_point.iter().copied().collect())
    }

    fn cache_openings_prover(
        &self,
        accumulator: std::rc::Rc<
            std::cell::RefCell<crate::poly::opening_proof::ProverOpeningAccumulator<F>>,
        >,
        r_cycle: crate::poly::opening_proof::OpeningPoint<BIG_ENDIAN, F>,
    ) {
        let prover_state = self
            .prover_state
            .as_ref()
            .expect("Prover state not initialized");

        for i in 0..self.d {
            let claim = prover_state.ra_i_polys[i].final_sumcheck_claim();
            accumulator.borrow_mut().append_sparse(
                vec![CommittedPolynomial::InstructionRa(i)],
                SumcheckId::InstructionRaVirtualization,
                self.r_address_chunks[i].clone(),
                r_cycle.r.clone(),
                vec![claim],
            );
            println!(
                "Append claim {} and r_cycle {:?} for ra_i_polys {}",
                claim, r_cycle.r, i
            );
        }
    }

    fn cache_openings_verifier(
        &self,
        accumulator: std::rc::Rc<
            std::cell::RefCell<crate::poly::opening_proof::VerifierOpeningAccumulator<F>>,
        >,
        r_cycle: crate::poly::opening_proof::OpeningPoint<BIG_ENDIAN, F>,
    ) {
        for i in 0..self.d {
            let opening_point =
                [self.r_address_chunks[i].as_slice(), r_cycle.r.as_slice()].concat();
                // [r_cycle.r.as_slice(), self.r_address_chunks[i].as_slice()].concat();

            println!(
                "Append opening point {:?} for ra_i_polys {}",
                opening_point, i
            );
            accumulator.borrow_mut().append_sparse(
                vec![CommittedPolynomial::InstructionRa(i)],
                SumcheckId::InstructionRaVirtualization,
                opening_point,
            );
        }
    }
}

// Append claim 7079235227563343742529484522962556269283949847934877130049960499207013041247 and r_cycle [13579098954998209440356566950045061089243669017828527109420637619089449321215, 1526320773308416672336895960938159064268862131436503705682731388637316076119, 17019079252442269225552493835305138869227103895797220503482579577240544915771, 19302340188964511414650018911981486393433893801850840295148790374919699566975, 13577606157164664885390481685942533255308466147157081288101331220121872276649, 8169763543261332935126739624319570103273154254565856896858648810380479129675, 763328581614806178107198580169510319904777786514095824459063702066941371595, 14683413798561914459117558215007760592254537940346720526520684218706679210703, 21065978701277274714726400089454881910914847100852279408252373270051303395461, 7587842512377324540782659819816126294162626495175505611967764504671089801393, 20176486486466350840500859449226975868913144093233463881039436951964287676518, 15359681222117761237490182181037515270981067624670820590290865046427513558975] for ra_i_polys 0

// Evaluating at [4421151823023921975030954809918311617897833668616052067742197788032634068392, 20063388429488398303238380552621007151499796912505617530130602177968843955954, 21432703626423864031258239220315032985765941626582856485662275113054790810767, 7011673948586155348238311506725022910302658144898112565266997369932927924023, 16849194142340358703223006300531863624940283158020023088265651449508669899337, 6932055140082015659594483487931056863237745065423378399697492323524416693710, 5198307408878605458421252986437409952851326021961960091251345963617903930190, 6901347660523136272233019232099011585422234451485702910812043491775088217229, 13579098954998209440356566950045061089243669017828527109420637619089449321215, 1526320773308416672336895960938159064268862131436503705682731388637316076119, 17019079252442269225552493835305138869227103895797220503482579577240544915771, 19302340188964511414650018911981486393433893801850840295148790374919699566975, 13577606157164664885390481685942533255308466147157081288101331220121872276649, 8169763543261332935126739624319570103273154254565856896858648810380479129675, 763328581614806178107198580169510319904777786514095824459063702066941371595, 14683413798561914459117558215007760592254537940346720526520684218706679210703, 21065978701277274714726400089454881910914847100852279408252373270051303395461, 7587842512377324540782659819816126294162626495175505611967764504671089801393, 20176486486466350840500859449226975868913144093233463881039436951964287676518, 15359681222117761237490182181037515270981067624670820590290865046427513558975]
