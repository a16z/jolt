//! Compile-time field literals of the limb table (Montgomery form via
//! `MontFp!`), so no fixed-power arithmetic or inversion runs at any time —
//! the verifier's `Fr` count is the same cold and warm. Each literal is the
//! value its name states; `tests::literals_match_their_derivations` recomputes
//! every one (and prints the decimal of a mismatching derivation to paste here).

use ark_bn254::Fr as ArkFr;
use ark_ff::MontFp;

use super::columns::{K_CHUNKS, LIMBS};
use super::digits::WINDOWS;

/// `2^{16j}` for `j < K_CHUNKS`.
pub(super) const POW_CHUNK: [ArkFr; K_CHUNKS] = [
    MontFp!("1"),
    MontFp!("65536"),
    MontFp!("4294967296"),
    MontFp!("281474976710656"),
    MontFp!("18446744073709551616"),
    MontFp!("1208925819614629174706176"),
    MontFp!("79228162514264337593543950336"),
    MontFp!("5192296858534827628530496329220096"),
    MontFp!("340282366920938463463374607431768211456"),
    MontFp!("22300745198530623141535718272648361505980416"),
    MontFp!("1461501637330902918203684832716283019655932542976"),
    MontFp!("95780971304118053647396689196894323976171195136475136"),
    MontFp!("6277101735386680763835789423207666416102355444464034512896"),
    MontFp!("411376139330301510538742295639337626245683966408394965837152256"),
    MontFp!("26959946667150639794667015087019630673637144422540572481103610249216"),
    MontFp!("1766847064778384329583297500742918515827483896875618958121606201292619776"),
    MontFp!("6350874878119819312338956282401532410528162663560392320966563075034087161851"),
];
/// `2^96`, the limb radix.
pub(super) const POW_LIMB: ArkFr = MontFp!("79228162514264337593543950336");
/// `2^111`, the carry offset.
pub(super) const CARRY_OFFSET: ArkFr = MontFp!("2596148429267413814265248164610048");
/// `2^75`: `2^267` in top-limb form.
pub(super) const K_OFFSET_TOP_LIMB: ArkFr = MontFp!("37778931862957161709568");
/// `2^64`.
pub(super) const POW_64: ArkFr = MontFp!("18446744073709551616");
/// The 96-bit limbs of the base field modulus `q`.
pub(super) const Q_LIMBS: [ArkFr; LIMBS] = [
    MontFp!("32324006162389411176778628423"),
    MontFp!("57042285082623239461879769745"),
    MontFp!("3486998266802970665"),
];
/// `16^k` for `k < WINDOWS`.
pub(super) const SIXTEEN_POWERS: [ArkFr; WINDOWS] = [
    MontFp!("1"),
    MontFp!("16"),
    MontFp!("256"),
    MontFp!("4096"),
    MontFp!("65536"),
    MontFp!("1048576"),
    MontFp!("16777216"),
    MontFp!("268435456"),
    MontFp!("4294967296"),
    MontFp!("68719476736"),
    MontFp!("1099511627776"),
    MontFp!("17592186044416"),
    MontFp!("281474976710656"),
    MontFp!("4503599627370496"),
    MontFp!("72057594037927936"),
    MontFp!("1152921504606846976"),
    MontFp!("18446744073709551616"),
    MontFp!("295147905179352825856"),
    MontFp!("4722366482869645213696"),
    MontFp!("75557863725914323419136"),
    MontFp!("1208925819614629174706176"),
    MontFp!("19342813113834066795298816"),
    MontFp!("309485009821345068724781056"),
    MontFp!("4951760157141521099596496896"),
    MontFp!("79228162514264337593543950336"),
    MontFp!("1267650600228229401496703205376"),
    MontFp!("20282409603651670423947251286016"),
    MontFp!("324518553658426726783156020576256"),
    MontFp!("5192296858534827628530496329220096"),
    MontFp!("83076749736557242056487941267521536"),
    MontFp!("1329227995784915872903807060280344576"),
    MontFp!("21267647932558653966460912964485513216"),
    MontFp!("340282366920938463463374607431768211456"),
    MontFp!("5444517870735015415413993718908291383296"),
    MontFp!("87112285931760246646623899502532662132736"),
    MontFp!("1393796574908163946345982392040522594123776"),
    MontFp!("22300745198530623141535718272648361505980416"),
    MontFp!("356811923176489970264571492362373784095686656"),
    MontFp!("5708990770823839524233143877797980545530986496"),
    MontFp!("91343852333181432387730302044767688728495783936"),
    MontFp!("1461501637330902918203684832716283019655932542976"),
    MontFp!("23384026197294446691258957323460528314494920687616"),
    MontFp!("374144419156711147060143317175368453031918731001856"),
    MontFp!("5986310706507378352962293074805895248510699696029696"),
    MontFp!("95780971304118053647396689196894323976171195136475136"),
    MontFp!("1532495540865888858358347027150309183618739122183602176"),
    MontFp!("24519928653854221733733552434404946937899825954937634816"),
    MontFp!("392318858461667547739736838950479151006397215279002157056"),
    MontFp!("6277101735386680763835789423207666416102355444464034512896"),
    MontFp!("100433627766186892221372630771322662657637687111424552206336"),
    MontFp!("1606938044258990275541962092341162602522202993782792835301376"),
    MontFp!("25711008708143844408671393477458601640355247900524685364822016"),
    MontFp!("411376139330301510538742295639337626245683966408394965837152256"),
    MontFp!("6582018229284824168619876730229402019930943462534319453394436096"),
    MontFp!("105312291668557186697918027683670432318895095400549111254310977536"),
    MontFp!("1684996666696914987166688442938726917102321526408785780068975640576"),
    MontFp!("26959946667150639794667015087019630673637144422540572481103610249216"),
    MontFp!("431359146674410236714672241392314090778194310760649159697657763987456"),
    MontFp!("6901746346790563787434755862277025452451108972170386555162524223799296"),
    MontFp!("110427941548649020598956093796432407239217743554726184882600387580788736"),
    MontFp!("1766847064778384329583297500742918515827483896875618958121606201292619776"),
    MontFp!("28269553036454149273332760011886696253239742350009903329945699220681916416"),
    MontFp!("452312848583266388373324160190187140051835877600158453279131187530910662656"),
    MontFp!("7237005577332262213973186563042994240829374041602535252466099000494570602496"),
];
/// `16^{−1}`.
pub(super) const SIXTEEN_INVERSE: ArkFr =
    MontFp!("20520227692349320520856005386178695395514091625390032197217066424914820464641");
/// `8`, the digit offset.
pub(super) const EIGHT: ArkFr = MontFp!("8");
/// `q_hi − 1` (`Q_HI − 1`), the canonicality bound.
pub(super) const Q_HI_MINUS_ONE: ArkFr = MontFp!("3486998266802970664");
/// `2^18`, the conjugated lookup key offset (`NEG_KEY_OFFSET`).
pub(super) const NEG_KEY_OFFSET_FR: ArkFr = MontFp!("262144");
/// `R_HI − 2`, the recoding window bound (`WINDOW_BOUND`).
pub(super) const WINDOW_BOUND_FR: ArkFr = MontFp!("3486998266802970663");

#[cfg(test)]
mod tests {
    use ark_ff::PrimeField;
    use jolt_field::{Field, Fr, Ring};

    use super::super::columns::{
        fr_from_biguint, limb, q_biguint, CARRY_OFFSET_BITS, CHUNK_BITS, K_OFFSET_BITS, LIMB_BITS,
        Q_HI,
    };
    use super::super::digits::WINDOW_BOUND;
    use super::super::layout::LOG_ROWS;
    use super::*;

    fn check(name: &str, literal: ArkFr, derived: Fr) {
        assert_eq!(
            Fr::from(literal),
            derived,
            "{name}: paste MontFp!(\"{}\")",
            ArkFr::from(derived).into_bigint()
        );
    }

    /// Every literal equals the value its name states.
    #[test]
    #[expect(clippy::expect_used, reason = "fixed nonzero field literal")]
    fn literals_match_their_derivations() {
        for (j, literal) in POW_CHUNK.iter().enumerate() {
            check(
                &format!("POW_CHUNK[{j}]"),
                *literal,
                Fr::pow2(CHUNK_BITS * j),
            );
        }
        check("POW_LIMB", POW_LIMB, Fr::pow2(LIMB_BITS));
        check("CARRY_OFFSET", CARRY_OFFSET, Fr::pow2(CARRY_OFFSET_BITS));
        check(
            "K_OFFSET_TOP_LIMB",
            K_OFFSET_TOP_LIMB,
            Fr::pow2(K_OFFSET_BITS - 2 * LIMB_BITS),
        );
        check("POW_64", POW_64, Fr::pow2(64));
        let q = q_biguint();
        for (a, literal) in Q_LIMBS.iter().enumerate() {
            check(
                &format!("Q_LIMBS[{a}]"),
                *literal,
                fr_from_biguint(&limb(&q, a)),
            );
        }
        for (k, literal) in SIXTEEN_POWERS.iter().enumerate() {
            check(&format!("SIXTEEN_POWERS[{k}]"), *literal, Fr::pow2(4 * k));
        }
        check(
            "SIXTEEN_INVERSE",
            SIXTEEN_INVERSE,
            Fr::pow2(4).inverse().expect("16 is invertible"),
        );
        check("EIGHT", EIGHT, Fr::from_u64(8));
        check("Q_HI_MINUS_ONE", Q_HI_MINUS_ONE, Fr::from_u64(Q_HI - 1));
        check(
            "NEG_KEY_OFFSET_FR",
            NEG_KEY_OFFSET_FR,
            Fr::from_u64(1 << LOG_ROWS),
        );
        check(
            "WINDOW_BOUND_FR",
            WINDOW_BOUND_FR,
            Fr::from_u64(WINDOW_BOUND),
        );
    }
}
