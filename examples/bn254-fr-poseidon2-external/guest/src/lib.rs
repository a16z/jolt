#![cfg_attr(feature = "guest", no_std)]

//! Poseidon2 BN254 t=3 permutation benchmark using a vendored copy of the
//! public [`taceo-poseidon2`](https://crates.io/crates/taceo-poseidon2) crate
//! (v0.2.1, MIT / Apache-2.0).
//!
//! ## Why vendored?
//! The upstream crate's `Cargo.toml` pulls in `ark-std = { features = ["std",
//! "getrandom"] }` and `num-bigint` with default features, neither of which
//! compile for `riscv64-unknown-none-elf`. The library *source* itself is
//! no_std-clean — it only uses `ark-ff` + `ark-bn254`. We vendor the two
//! modules (`perm.rs` + `bn254/t3.rs`) into this guest so the dependency
//! graph stays compatible with Jolt's guest target while preserving the
//! exact algorithm, round constants, and internal matrix of the upstream
//! crate. Upstream code is unmodified below except for:
//!   * removed doc-attributes / `#[deny(missing_docs)]`,
//!   * removed `#[cfg(test)]` blocks,
//!   * changed `pub(crate)` to `pub(super)` where needed for the in-guest
//!     layout,
//!   * deleted the `t2/t4/t8/t12/t16` variants (we only need `t=3`).
//!
//! This is therefore algorithmically identical to `taceo::bn254::t3::permutation`.
//! We call this the "external reference" path.
//!
//! ## Parameters (HorizenLabs-compatible)
//! `d = 5`, `R_F = 8`, `R_P = 56`, `t = 3`, `MAT_DIAG_M_1 = [1, 1, 2]`.
//! Round constants match <https://github.com/HorizenLabs/poseidon2> and are
//! therefore bit-identical to the ones baked into our SDK / hand-written
//! arkworks guests.

use ark_bn254::Fr;
use ark_ff::PrimeField;

#[jolt::provable(
    stack_size = 65536,
    heap_size = 131072,
    max_input_size = 8192,
    max_trace_length = 33554432
)]
fn fr_poseidon2_external(s0: [u64; 4], s1: [u64; 4], s2: [u64; 4]) -> [[u64; 4]; 3] {
    let state = [
        fr_from_limbs(s0[0], s0[1], s0[2], s0[3]),
        fr_from_limbs(s1[0], s1[1], s1[2], s1[3]),
        fr_from_limbs(s2[0], s2[1], s2[2], s2[3]),
    ];
    let out = t3::permutation(&state);
    [fr_to_limbs(&out[0]), fr_to_limbs(&out[1]), fr_to_limbs(&out[2])]
}

fn fr_from_limbs(l0: u64, l1: u64, l2: u64, l3: u64) -> Fr {
    let mut bytes = [0u8; 32];
    bytes[0..8].copy_from_slice(&l0.to_le_bytes());
    bytes[8..16].copy_from_slice(&l1.to_le_bytes());
    bytes[16..24].copy_from_slice(&l2.to_le_bytes());
    bytes[24..32].copy_from_slice(&l3.to_le_bytes());
    Fr::from_le_bytes_mod_order(&bytes)
}

fn fr_to_limbs(fr: &Fr) -> [u64; 4] {
    use ark_ff::BigInteger;
    let bi = fr.into_bigint();
    let bytes = bi.to_bytes_le();
    let mut limbs = [0u64; 4];
    for (i, limb) in limbs.iter_mut().enumerate() {
        let start = i * 8;
        let end = core::cmp::min(start + 8, bytes.len());
        if start < bytes.len() {
            let mut buf = [0u8; 8];
            buf[..end - start].copy_from_slice(&bytes[start..end]);
            *limb = u64::from_le_bytes(buf);
        }
    }
    limbs
}

// ---------------------------------------------------------------------------
// Vendored from taceo-poseidon2 v0.2.1 (src/perm.rs), MIT OR Apache-2.0.
// Copyright (c) Taceo GmbH. No algorithmic changes.
// ---------------------------------------------------------------------------
mod perm {
    use ark_ff::PrimeField;

    pub(super) struct Poseidon2Permutation<
        F: PrimeField,
        const T: usize,
        const D: u64,
        const ROUNDS_F: usize,
        const ROUNDS_P: usize,
    > {
        mat_internal_diag_m_1: [F; T],
        round_constants_external: [[F; T]; ROUNDS_F],
        round_constants_internal: [F; ROUNDS_P],
    }

    impl<F: PrimeField, const T: usize, const D: u64, const ROUNDS_F: usize, const ROUNDS_P: usize>
        Poseidon2Permutation<F, T, D, ROUNDS_F, ROUNDS_P>
    {
        pub(super) const fn new(
            mat_internal_diag_m_1: [F; T],
            round_constants_external: [[F; T]; ROUNDS_F],
            round_constants_internal: [F; ROUNDS_P],
        ) -> Self {
            Self {
                mat_internal_diag_m_1,
                round_constants_external,
                round_constants_internal,
            }
        }

        fn sbox(input: &mut [F; T]) {
            input.iter_mut().for_each(Self::single_sbox);
        }

        fn single_sbox(input: &mut F) {
            match D {
                5 => {
                    let input2 = input.square();
                    let input4 = input2.square();
                    *input *= input4;
                }
                _ => {
                    *input = input.pow([D]);
                }
            }
        }

        fn matmul_external(input: &mut [F; T]) {
            match T {
                3 => {
                    let sum = input[0] + input[1] + input[2];
                    input[0] += &sum;
                    input[1] += &sum;
                    input[2] += sum;
                }
                _ => unreachable!(),
            }
        }

        fn matmul_internal(&self, input: &mut [F; T]) {
            match T {
                3 => {
                    // Matrix [[2, 1, 1], [1, 2, 1], [1, 1, 3]]
                    let sum = input[0] + input[1] + input[2];
                    input[0] += &sum;
                    input[1] += &sum;
                    input[2].double_in_place();
                    input[2] += sum;
                    // Silence unused-field warning: the generic upstream code
                    // consults `mat_internal_diag_m_1` via `debug_assert_eq!`;
                    // we keep the field for layout identity but reference it
                    // here so clippy doesn't complain in non-debug builds.
                    let _ = &self.mat_internal_diag_m_1;
                }
                _ => unreachable!(),
            }
        }

        fn add_rc_external(&self, input: &mut [F; T], rc_e: &[F; T]) {
            for (s, rc) in input.iter_mut().zip(rc_e.iter()) {
                *s += rc;
            }
        }

        fn external_round(&self, state: &mut [F; T], rc_e: &[F; T]) {
            self.add_rc_external(state, rc_e);
            Self::sbox(state);
            Self::matmul_external(state);
        }

        fn internal_round(&self, state: &mut [F; T], rc_i: F) {
            state[0] += rc_i;
            Self::single_sbox(&mut state[0]);
            self.matmul_internal(state);
        }

        pub(super) fn permutation_in_place(&self, state: &mut [F; T]) {
            Self::matmul_external(state);
            let mut round_constants_external = self.round_constants_external.iter();

            for rc_e in round_constants_external.by_ref().take(ROUNDS_F / 2) {
                self.external_round(state, rc_e);
            }

            for rc_i in self.round_constants_internal {
                self.internal_round(state, rc_i);
            }

            for rc_e in round_constants_external {
                self.external_round(state, rc_e);
            }
        }

        pub(super) fn permutation(&self, input: &[F; T]) -> [F; T] {
            let mut state = *input;
            self.permutation_in_place(&mut state);
            state
        }
    }
}

// ---------------------------------------------------------------------------
// Vendored from taceo-poseidon2 v0.2.1 (src/bn254/t3.rs), MIT OR Apache-2.0.
// Copyright (c) Taceo GmbH. No algorithmic changes; round constants are
// transcribed verbatim.
// ---------------------------------------------------------------------------
mod t3 {
    use super::perm::Poseidon2Permutation;
    use ark_bn254::Fr as Scalar;
    use ark_ff::MontFp;

    const T: usize = 3;
    const D: u64 = 5;
    const ROUNDS_F: usize = 8;
    const ROUNDS_P: usize = 56;

    const MAT_DIAG_M_1: [Scalar; T] = [MontFp!("1"), MontFp!("1"), MontFp!("2")];
    const EXTERNAL_RC: [[Scalar; T]; ROUNDS_F] = [
        [
            MontFp!("13128406282895484157369354038809433636203389051939936481821261911791933663254"),
            MontFp!("18931653859213243425446645781588512487838213266321401679594943842133071369744"),
            MontFp!("14100663835952519432830313936592734340076294692040144715814219945570907513297"),
        ],
        [
            MontFp!("4829113795940962171577509772302063766582957624337039572002553144762883322341"),
            MontFp!("15524196826242151316602020382811195434692947787822797536837043495207890599720"),
            MontFp!("11824742889827005569732308046012743315382715056680481843559537371456931944245"),
        ],
        [
            MontFp!("15824369292130948538570881538463827283727388637222356799784648390667783881850"),
            MontFp!("7395652367440825515524159918310823124942438011035473842936180620057265532493"),
            MontFp!("1241351203963627868835881804826107927839874261162687401459390240620885410254"),
        ],
        [
            MontFp!("6688265362431458560657026053775250595854204120757399493099812773970419156132"),
            MontFp!("18628865421786169197184064906533816626840829027307965436801990532221681661310"),
            MontFp!("17770079997659052348824924629777474963416629061770380464722096481670103655806"),
        ],
        [
            MontFp!("12123026335854515584932892161148559902027319284544852339906677442670161590992"),
            MontFp!("11747143856113197599032240626240804787576886917202313931914972592787570603429"),
            MontFp!("12689083329367969619896630238881490862330991685178863399139986099061967775891"),
        ],
        [
            MontFp!("9363616378570856727297258914956380343356030981401312041884116403700849212733"),
            MontFp!("13238291046435061349401827110993774315432323243867917623501520885175217584478"),
            MontFp!("13857006478672530359037215101120381968370236111775805219419707798416454682620"),
        ],
        [
            MontFp!("2022752961549084842139747691238383165524359342011064407942599644003308437489"),
            MontFp!("11377043765620686524844863869245961003946340433252666374730228559486855986878"),
            MontFp!("9107028336454933966239128359918274121166034584181733998485105905495346200934"),
        ],
        [
            MontFp!("900063247840342897532382686223939136593244983486268682637380837456165317070"),
            MontFp!("11261302954518146885624063833699323298803404236535464228351677636819579513431"),
            MontFp!("7126990412157463341897179572979760225771626877677162088926546182321369054630"),
        ],
    ];
    const INTERNAL_RC: [Scalar; ROUNDS_P] = [
        MontFp!("11811415718957691261673974625780511541635150909919309658375768251762566747317"),
        MontFp!("17491388639298611159333770975992024026420968324544834879936543171716736973879"),
        MontFp!("5647537972700463414111873015737673282707440513292923385601908870282442800104"),
        MontFp!("13098696909140066209556423100763036393001603197583133354863092304798723388565"),
        MontFp!("6951180250619279643770888203380891623788978362131976553140006882493632020745"),
        MontFp!("11250251081997661635793843737498879309304455145146915350538637298238893102958"),
        MontFp!("2246982048814095620312232487641427155108104073024754628893054837638848127964"),
        MontFp!("18897180842973857564376958241871700087418903006311506731527228148081597475814"),
        MontFp!("11557404599711559103972421944754928847181400366333080241838467983028485750549"),
        MontFp!("17156358787639157774388183034849932704703797218604790661321342987075785318260"),
        MontFp!("8846001957151556825394442611430138293780354129800063716225175548340091032449"),
        MontFp!("21883449834630454155761926448978525628607016008113566399646971468161186616967"),
        MontFp!("11782201180140779170005707786217005381305915516114251118577530420880166417952"),
        MontFp!("19574374768428302416384468550351257389078501920039012797497943057156188490399"),
        MontFp!("8515987927591912252146893631936027853249294776314628553087138119917968203620"),
        MontFp!("17278996890957540943430295799612663512184925495827057764219426280563743078943"),
        MontFp!("4560144125266860756441160513270281593457202308593722614013851111005532208589"),
        MontFp!("18507459160700813704135500972073304101922968342745790738233104310822653821881"),
        MontFp!("12853272419783978245995917302225694649366687506910892647236063701566570840428"),
        MontFp!("14374895923592519298500369713759001634990764548024903321294831249025876110484"),
        MontFp!("1754533789272381217541450481312878927560073411620344950409407505576538004136"),
        MontFp!("20448232810715691360468548645921483318770769828465347895613479253435247065293"),
        MontFp!("4203277692183102377396835282861288449527228200284576966986741905195109677387"),
        MontFp!("11506339386261725202512749094297334054772084639665212079028551409689271965431"),
        MontFp!("4408799661846477128378547528471700197737434561274043409442231147309460168718"),
        MontFp!("10862521404448958117187164110262290189825635328197001646848012017699995213390"),
        MontFp!("7012061838863338817532836723152059636816924388921632356281537445328382279260"),
        MontFp!("8337544039076735620694225144163354013921209405711398618659178986151546625400"),
        MontFp!("16173744372216956516796750206695252671549928142051779144629150462255079400849"),
        MontFp!("19072902632067672883974143637757649536845413107085656789672471396027868707732"),
        MontFp!("3487852254355424154670010750480228751987308757772575371606146474985412561707"),
        MontFp!("17727517395793273304860106667199855253218123164763798377815886217088561516989"),
        MontFp!("13280131383170382695839570176732265848909891244754629477752800360224963964534"),
        MontFp!("21504421972374418324171209120165696620934505501591484695447432472073975792776"),
        MontFp!("13753604424945682926871108642602624411461374991709441590662260371815673344981"),
        MontFp!("8053178768600673579416591772204841415225213226540397062676127402210384682315"),
        MontFp!("15101558583452488762759591936595783545455044970328380152280373697190919758012"),
        MontFp!("6286700389345423344101403023711121482167900236544298155098199100234816571786"),
        MontFp!("19368755554193272721035317233504719593365546521121074341670771231332472422552"),
        MontFp!("13306281365497267243785678269212920842854030794417306689235276460198094483575"),
        MontFp!("10121764749051640353641114693266514664967620368543293902008953934189850195966"),
        MontFp!("179619165022370308972665071682395477322215797039585945216341070107573537790"),
        MontFp!("14053393851645634065914179337120715807963438235922115988819572738574714471437"),
        MontFp!("17345906218970918797922168310670548252023720338285437740234091480846393436478"),
        MontFp!("10383068492552043678323859571562933490503408853170063884414176092784243607055"),
        MontFp!("12096041499044892166554391619429604246288825927654072010011878199637889490527"),
        MontFp!("6449742640166027959651492823149770763572943879017164812917305794918053034585"),
        MontFp!("6551805454148805882554763665748573416514894105513920161214733482541847062214"),
        MontFp!("3651410956659878392469489270906333016569562868954890104332567650040497030813"),
        MontFp!("15219053914464753937310253926447830297339787956721755285255510737973021838676"),
        MontFp!("881679665678132972106931291023348167890022611850562267871389203532691753422"),
        MontFp!("5006067481688857073852527145736822635357747460125905556158034280392250104971"),
        MontFp!("12765332320844032254009314500332101047115754896003948733635815046365410860591"),
        MontFp!("12908190215073542091623737558383307555705501651914623082354191483197810853182"),
        MontFp!("1446042792715825508366007519346636771782990303010685652946852324744810237839"),
        MontFp!("17414863822034645298427260856470503848317996477890518738401812766215195632841"),
    ];

    pub(super) const POSEIDON2_BN254_T3_PARAMS: Poseidon2Permutation<
        Scalar,
        T,
        D,
        ROUNDS_F,
        ROUNDS_P,
    > = Poseidon2Permutation::new(MAT_DIAG_M_1, EXTERNAL_RC, INTERNAL_RC);

    pub(super) fn permutation(state: &[Scalar; 3]) -> [Scalar; 3] {
        POSEIDON2_BN254_T3_PARAMS.permutation(state)
    }
}
