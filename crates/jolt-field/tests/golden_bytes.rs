//! Golden byte-compatibility fixtures.
//!
//! These pin the exact wire (bincode) and transcript (`to_bytes_le`)
//! encodings — and the BN254 legacy challenge derivations — to the byte
//! streams produced by the original `jolt-field` crate, guarding the hard
//! invariant that replacing that crate does not change proof bytes.
//!
//! GENERATED from jolt-field at commit
//! 5b3e39ece1c27586a1f7cc77f24e718cb5d73e10 (branch
//! feat/jolt-field-replacement) by a one-off generator test (deleted in the
//! same change; see this file's history). To regenerate: check out that
//! commit, restore `tests/golden_gen.rs` and the `jolt-field`
//! dev-dependency, run
//! `cargo nextest run -p jolt-field --all-features generate_golden_fixtures`,
//! and splice `target/tmp/golden_fixtures.txt` into the const blocks below.
//!
//! Row format: `(input hex, expected hex)` where the element is
//! `from_bytes_le_reduced(input)` (prime fields) or
//! `from_challenge_bytes` / `from_scalar_challenge_bytes(input)`
//! (challenge fixtures), and `expected` is the canonical LE encoding.
//! Extension rows are `(canonical coefficients, bincode wire hex)`.

#![expect(clippy::unwrap_used, reason = "test code")]
// The whole file is backend fixture data; without a backend there is nothing
// to pin and every item would be dead code under -Dwarnings.
#![cfg(any(feature = "bn254", feature = "solinas"))]

use jolt_field as two;

use two::CanonicalEncoding;

fn unhex(s: &str) -> Vec<u8> {
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap())
        .collect()
}

/// Element from the reducing decode; canonical bytes and bincode wire must
/// equal the fixture, and the checked decode must round-trip.
fn check_prime_rows<F>(rows: &[(&str, &str)])
where
    F: CanonicalEncoding
        + serde::Serialize
        + serde::de::DeserializeOwned
        + PartialEq
        + std::fmt::Debug
        + Copy,
{
    let cfg = bincode::config::standard();
    for (input, expected) in rows {
        let (input, expected) = (unhex(input), unhex(expected));
        let e = F::from_bytes_le_reduced(&input);
        assert_eq!(e.to_bytes_le_vec(), expected, "transcript bytes diverge");
        let wire = bincode::serde::encode_to_vec(e, cfg).unwrap();
        assert_eq!(wire, expected, "wire bytes diverge");
        assert_eq!(
            F::from_bytes_le_checked(&expected),
            Some(e),
            "checked decode round-trip"
        );
        let (back, read): (F, usize) = bincode::serde::decode_from_slice(&expected, cfg).unwrap();
        assert_eq!((back, read), (e, expected.len()));
    }
}

#[cfg(feature = "bn254")]
const FIX_BN254_FR: &[(&str, &str)] = &[
    (
        "0000000000000000000000000000000000000000000000000000000000000000",
        "0000000000000000000000000000000000000000000000000000000000000000",
    ),
    (
        "0100000000000000000000000000000000000000000000000000000000000000",
        "0100000000000000000000000000000000000000000000000000000000000000",
    ),
    (
        "0200000000000000000000000000000000000000000000000000000000000000",
        "0200000000000000000000000000000000000000000000000000000000000000",
    ),
    (
        "000000f093f5e1439170b97948e833285d588181b64550b829a031e1724e6430",
        "000000f093f5e1439170b97948e833285d588181b64550b829a031e1724e6430",
    ),
    (
        "010000f093f5e1439170b97948e833285d588181b64550b829a031e1724e6430",
        "0000000000000000000000000000000000000000000000000000000000000000",
    ),
    (
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "faffff4f1c3496ac29cd609f9576fc362e4679786fa36e662fdf079ac1770a0e",
    ),
    (
        "dae623d2aa29f41845b9a32a1d819bb65e7ca3106f280413e0638df8a0029a4d",
        "d9e623e2163412d5b348eab0d498678e0124228fb8e2b35ab6c35b172eb4351d",
    ),
    (
        "7c41df5c178ab67358fb5375991fe1ba909e7d5a185156624bd8be7123e5d34b",
        "7b41df6c8394d42fc78a9afb5037ad923346fcd8610b06aa21388d90b0966f1b",
    ),
    (
        "f0496a0db0d44aecb71a5457bac1cd1f004ed4e1b8a4ad0fda75893f75f9e34b",
        "ef496a1d1cdf68a826aa9add71d999f7a2f55260025f5d57b0d5575e02ab7f1b",
    ),
    (
        "a3697a40313c6c012a94dd33469644182c96bcbf573df48ae896963ea3675f9e",
        "a0697a70755bc6357642b1c66cdda89f148d383b346c03626bb6019b4a7c320d",
    ),
    (
        "cd49341848ecb614e2aee4b81d74bc679583ce92d4db5b516e29348f760fc986",
        "cb4934382001f38cbfcd71c58ca35417dbd2cb8f6750bbe01ae9d0cc90720026",
    ),
    (
        "4388d11302c44ae8b30ca39efc96d6cf965cba921500e6806331d6ea881d2dcc",
        "3f88d153b2edc2d86e4abdb7daf5062f22fbb48c3be9a49fbcb00f66bde39b0a",
    ),
    (
        "f8cef80001d2e51cdc6e48ab3ae7fb4ef2c2118736b66a9ae1e2a4741d0cc498",
        "f5cef83045f13f51281d1c3e612e60d6dab98d0213e57971640210d1c4209707",
    ),
    (
        "26ebe17a6b629d695def5007330a4a49270921edea58aa1d6891972609397536",
        "25ebe18ad76cbb25cc7e978dea211621cab09f6b34135a653ef1654596ea1006",
    ),
];
#[cfg(feature = "bn254")]
const FIX_BN254_FQ: &[(&str, &str)] = &[
    (
        "0000000000000000000000000000000000000000000000000000000000000000",
        "0000000000000000000000000000000000000000000000000000000000000000",
    ),
    (
        "0100000000000000000000000000000000000000000000000000000000000000",
        "0100000000000000000000000000000000000000000000000000000000000000",
    ),
    (
        "0200000000000000000000000000000000000000000000000000000000000000",
        "0200000000000000000000000000000000000000000000000000000000000000",
    ),
    (
        "46fd7cd8168c203c8dca7168916a81975d588181b64550b829a031e1724e6430",
        "46fd7cd8168c203c8dca7168916a81975d588181b64550b829a031e1724e6430",
    ),
    (
        "47fd7cd8168c203c8dca7168916a81975d588181b64550b829a031e1724e6430",
        "0000000000000000000000000000000000000000000000000000000000000000",
    ),
    (
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "9c0d8fc58d435dd33d0bc7f528eb780a2c4679786fa36e662fdf079ac1770a0e",
    ),
    (
        "dae623d2aa29f41845b9a32a1d819bb65e7ca3106f280413e0638df8a0029a4d",
        "93e9a6f9939dd3dcb7ee31c28b161a1f0124228fb8e2b35ab6c35b172eb4351d",
    ),
    (
        "7c41df5c178ab67358fb5375991fe1ba909e7d5a185156624bd8be7123e5d34b",
        "3544628400fe9537cb30e20c08b55f233346fcd8610b06aa21388d90b0966f1b",
    ),
    (
        "f0496a0db0d44aecb71a5457bac1cd1f004ed4e1b8a4ad0fda75893f75f9e34b",
        "a94ced3499482ab02a50e2ee28574c88a2f55260025f5d57b0d5575e02ab7f1b",
    ),
    (
        "a3697a40313c6c012a94dd33469644182c96bcbf573df48ae896963ea3675f9e",
        "ce7103b7ec970a4d823488fa9156c051138d383b346c03626bb6019b4a7c320d",
    ),
    (
        "cd49341848ecb614e2aee4b81d74bc679583ce92d4db5b516e29348f760fc986",
        "3f4f3a671ad4759cc71901e8fa9eb938dad2cb8f6750bbe01ae9d0cc90720026",
    ),
    (
        "4388d11302c44ae8b30ca39efc96d6cf965cba921500e6806331d6ea881d2dcc",
        "2793ddb1a693c8f77ee2dbfcb6ecd07120fbb48c3be9a49fbcb00f66bde39b0a",
    ),
    (
        "f8cef80001d2e51cdc6e48ab3ae7fb4ef2c2118736b66a9ae1e2a4741d0cc498",
        "23d78177bc2d8468340ff37186a77788d9b98d0213e57971640210d1c4209707",
    ),
    (
        "26ebe17a6b629d695def5007330a4a49270921edea58aa1d6891972609397536",
        "dfed64a254d67c2dd024df9ea19fc8b1c9b09f6b34135a653ef1654596ea1006",
    ),
];
#[cfg(feature = "bn254")]
const FIX_BN254_FR_CHALLENGE: &[(&str, &str)] = &[
    (
        "00000000000000000000000000000000",
        "0000000000000000000000000000000000000000000000000000000000000000",
    ),
    (
        "ffffffffffffffffffffffffffffffff",
        "922306fba4417702705b58c72aed7f2e72a8da6190063056ab362c65cdc32617",
    ),
    (
        "dae623d2aa29f41845b9a32a1d819bb6",
        "3b6bccb3eae3ebb5d9133924500858a3fac6c42854d0d35ae53b1e8f9f71200e",
    ),
    (
        "5e7ca3106f280413e0638df8a0029a4d",
        "7f37873e17701c776900992bbf4d097271100afd8855cf1859fe2dc56226e00b",
    ),
    (
        "7c41df5c178ab67358fb5375991fe1ba",
        "e77a83d4edd8d4a65ada9fd99b800bf3afba689fbd890e2c3dc10db8948c790f",
    ),
    (
        "909e7d5a185156624bd8be7123e5d34b",
        "3c1f49d192ebb7e7f8a0be53b8e30491a1a98c2e45558f3b7cd443d88538c814",
    ),
    (
        "f0496a0db0d44aecb71a5457bac1cd1f",
        "2be6dfd869df0b3ae480a5ad51a4fc21150cb9ffca8faca48897f93285b78b22",
    ),
    (
        "004ed4e1b8a4ad0fda75893f75f9e34b",
        "f8c2174ea5e97eb646b7b0fd9a6a047dca24236aa1a61e02e964d8aaa051c926",
    ),
];
#[cfg(feature = "bn254")]
const FIX_BN254_FR_SCALAR_CHALLENGE: &[(&str, &str)] = &[
    (
        "0000000000000000000000000000000000000000000000000000000000000000",
        "0000000000000000000000000000000000000000000000000000000000000000",
    ),
    (
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "faffff4f1c3496ac29cd609f9576fc362e4679786fa36e662fdf079ac1770a0e",
    ),
    (
        "dae623d2aa29f41845b9a32a1d819bb65e7ca3106f280413e0638df8a0029a4d",
        "499a02e0a8b7dbd0ce414288ee01adbd413a7c17508c78647173632507ea5419",
    ),
    (
        "7c41df5c178ab67358fb5375991fe1ba909e7d5a185156624bd8be7123e5d34b",
        "49d3e54349d314c43f75de24c9ac364000311d9608c85ae81f7627557642791b",
    ),
    (
        "f0496a0db0d44aecb71a5457bac1cd1f004ed4e1b8a4ad0fda75893f75f9e34b",
        "47e3f9b5efb2edcacaeabed1bf337f5faa6bbcb47d3dd9d545ca0d2c4230b82e",
    ),
    (
        "a3697a40313c6c012a94dd33469644182c96bcbf573df48ae896963ea3675f9e",
        "9b5f67d382b5f01cd7a211eae503fbb3003b12c20f0ca401848ba78de78e3c12",
    ),
    (
        "cd49341848ecb614e2aee4b81d74bc679583ce92d4db5b516e29348f760fc986",
        "82c90fb63f5ea15e0c99f5ed702db4f4f25a6f17decd6d016e3526c44cfab70b",
    ),
    (
        "4388d11302c44ae8b30ca39efc96d6cf965cba921500e6806331d6ea881d2dcc",
        "cb2d1d9856e14f1fef75479b49d2286e727e157be85dbcfabeaa9221a0822413",
    ),
];
#[cfg(feature = "bn254")]
const FIX_BN254_FQ_CHALLENGE: &[(&str, &str)] = &[
    (
        "00000000000000000000000000000000",
        "0000000000000000000000000000000000000000000000000000000000000000",
    ),
    (
        "ffffffffffffffffffffffffffffffff",
        "00000000000000000000000000000000ffffffffffffffffffffffffffffff1f",
    ),
    (
        "dae623d2aa29f41845b9a32a1d819bb6",
        "00000000000000000000000000000000dae623d2aa29f41845b9a32a1d819b16",
    ),
    (
        "5e7ca3106f280413e0638df8a0029a4d",
        "000000000000000000000000000000005e7ca3106f280413e0638df8a0029a0d",
    ),
    (
        "7c41df5c178ab67358fb5375991fe1ba",
        "000000000000000000000000000000007c41df5c178ab67358fb5375991fe11a",
    ),
    (
        "909e7d5a185156624bd8be7123e5d34b",
        "00000000000000000000000000000000909e7d5a185156624bd8be7123e5d30b",
    ),
    (
        "f0496a0db0d44aecb71a5457bac1cd1f",
        "00000000000000000000000000000000f0496a0db0d44aecb71a5457bac1cd1f",
    ),
    (
        "004ed4e1b8a4ad0fda75893f75f9e34b",
        "00000000000000000000000000000000004ed4e1b8a4ad0fda75893f75f9e30b",
    ),
];
#[cfg(feature = "bn254")]
const FIX_BN254_FQ_SCALAR_CHALLENGE: &[(&str, &str)] = &[
    (
        "0000000000000000000000000000000000000000000000000000000000000000",
        "0000000000000000000000000000000000000000000000000000000000000000",
    ),
    (
        "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "9c0d8fc58d435dd33d0bc7f528eb780a2c4679786fa36e662fdf079ac1770a0e",
    ),
    (
        "dae623d2aa29f41845b9a32a1d819bb65e7ca3106f280413e0638df8a0029a4d",
        "31a50e3e9d5de1efded960cdcaf87600403a7c17508c78647173632507ea5419",
    ),
    (
        "7c41df5c178ab67358fb5375991fe1ba909e7d5a185156624bd8be7123e5d34b",
        "bdd8eb7243a697d347c16d4737a89b61ff301d9608c85ae81f7627557642791b",
    ),
    (
        "f0496a0db0d44aecb71a5457bac1cd1f004ed4e1b8a4ad0fda75893f75f9e34b",
        "2fee0514e458f3e9da82dd169c2a49a2a86bbcb47d3dd9d545ca0d2c4230b82e",
    ),
    (
        "a3697a40313c6c012a94dd33469644182c96bcbf573df48ae896963ea3675f9e",
        "c967f019faf13434e394e81d0b7d1266ff3a12c20f0ca401848ba78de78e3c12",
    ),
    (
        "cd49341848ecb614e2aee4b81d74bc679583ce92d4db5b516e29348f760fc986",
        "6ad41b143404a77d1c3114334d247e37f15a6f17decd6d016e3526c44cfab70b",
    ),
    (
        "4388d11302c44ae8b30ca39efc96d6cf965cba921500e6806331d6ea881d2dcc",
        "8530a0afd34a1127f31b8fac0050dbfe717e157be85dbcfabeaa9221a0822413",
    ),
];

#[cfg(feature = "bn254")]
mod bn254 {
    use super::*;

    fn check_challenge_rows<F: CanonicalEncoding>(rows: &[(&str, &str)], scalar: bool) {
        for (input, expected) in rows {
            let (input, expected) = (unhex(input), unhex(expected));
            let e = if scalar {
                F::from_scalar_challenge_bytes(&input)
            } else {
                F::from_challenge_bytes(&input)
            };
            assert_eq!(e.to_bytes_le_vec(), expected, "challenge bytes diverge");
        }
    }

    #[test]
    fn fr_bytes_match_fixtures() {
        check_prime_rows::<two::Fr>(FIX_BN254_FR);
    }

    #[test]
    fn fq_bytes_match_fixtures() {
        check_prime_rows::<two::Fq>(FIX_BN254_FQ);
    }

    #[test]
    fn fr_challenges_match_fixtures() {
        check_challenge_rows::<two::Fr>(FIX_BN254_FR_CHALLENGE, false);
        check_challenge_rows::<two::Fr>(FIX_BN254_FR_SCALAR_CHALLENGE, true);
    }

    #[test]
    fn fq_challenges_match_fixtures() {
        check_challenge_rows::<two::Fq>(FIX_BN254_FQ_CHALLENGE, false);
        check_challenge_rows::<two::Fq>(FIX_BN254_FQ_SCALAR_CHALLENGE, true);
    }
}

#[cfg(feature = "solinas")]
mod solinas {
    #![expect(clippy::unreadable_literal, reason = "generated fixture data")]

    use super::*;
    use two::ExtField;

    const FIX_PRIME24_OFFSET3: &[(&str, &str)] = &[
        ("00000000", "00000000"),
        ("01000000", "01000000"),
        ("02000000", "02000000"),
        ("fcffff00", "fcffff00"),
        ("fdffff00", "00000000"),
        ("ffffffff", "ff020000"),
        ("dae623d2", "50e92300"),
        ("aa29f418", "f229f400"),
        ("45b9a32a", "c3b9a300"),
        ("1d819bb6", "3f839b00"),
        ("5e7ca310", "8e7ca300"),
        ("6f280413", "a8280400"),
        ("e0638df8", "c8668d00"),
        ("a0029a4d", "87039a00"),
    ];
    const FIX_PRIME30_OFFSET35: &[(&str, &str)] = &[
        ("00000000", "00000000"),
        ("01000000", "01000000"),
        ("02000000", "02000000"),
        ("dcffff3f", "dcffff3f"),
        ("ddffff3f", "00000000"),
        ("ffffffff", "8b000000"),
        ("dae623d2", "43e72312"),
        ("aa29f418", "aa29f418"),
        ("45b9a32a", "45b9a32a"),
        ("1d819bb6", "63819b36"),
        ("5e7ca310", "5e7ca310"),
        ("6f280413", "6f280413"),
        ("e0638df8", "49648d38"),
        ("a0029a4d", "c3029a0d"),
    ];
    const FIX_PRIME31_OFFSET19: &[(&str, &str)] = &[
        ("00000000", "00000000"),
        ("01000000", "01000000"),
        ("02000000", "02000000"),
        ("ecffff7f", "ecffff7f"),
        ("edffff7f", "00000000"),
        ("ffffffff", "25000000"),
        ("dae623d2", "ede62352"),
        ("aa29f418", "aa29f418"),
        ("45b9a32a", "45b9a32a"),
        ("1d819bb6", "30819b36"),
        ("5e7ca310", "5e7ca310"),
        ("6f280413", "6f280413"),
        ("e0638df8", "f3638d78"),
        ("a0029a4d", "a0029a4d"),
    ];
    const FIX_PRIME32_OFFSET99: &[(&str, &str)] = &[
        ("00000000", "00000000"),
        ("01000000", "01000000"),
        ("02000000", "02000000"),
        ("9cffffff", "9cffffff"),
        ("9dffffff", "00000000"),
        ("ffffffff", "62000000"),
        ("dae623d2", "dae623d2"),
        ("aa29f418", "aa29f418"),
        ("45b9a32a", "45b9a32a"),
        ("1d819bb6", "1d819bb6"),
        ("5e7ca310", "5e7ca310"),
        ("6f280413", "6f280413"),
        ("e0638df8", "e0638df8"),
        ("a0029a4d", "a0029a4d"),
    ];
    const FIX_PRIME40_OFFSET195: &[(&str, &str)] = &[
        ("0000000000000000", "0000000000000000"),
        ("0100000000000000", "0100000000000000"),
        ("0200000000000000", "0200000000000000"),
        ("3cffffffff000000", "3cffffffff000000"),
        ("3dffffffff000000", "0000000000000000"),
        ("ffffffffffffffff", "ffffffc200000000"),
        ("dae623d2aa29f418", "15e225e5aa000000"),
        ("45b9a32a1d819bb6", "882cbcb51d000000"),
        ("5e7ca3106f280413", "d6a61f1f6f000000"),
        ("e0638df8a0029a4d", "66b3a933a1000000"),
        ("7c41df5c178ab673", "9a4c03b517000000"),
        ("58fb5375991fe1ba", "f575ad039a000000"),
        ("909e7d5a18515662", "435e65a518000000"),
        ("4bd8be7123e5d34b", "ba3f81ab23000000"),
    ];
    const FIX_PRIME48_OFFSET59: &[(&str, &str)] = &[
        ("0000000000000000", "0000000000000000"),
        ("0100000000000000", "0100000000000000"),
        ("0200000000000000", "0200000000000000"),
        ("c4ffffffffff0000", "c4ffffffffff0000"),
        ("c5ffffffffff0000", "0000000000000000"),
        ("ffffffffffffffff", "ffff3a0000000000"),
        ("dae623d2aa29f418", "16a729d2aa290000"),
        ("45b9a32a1d819bb6", "fececd2a1d810000"),
        ("5e7ca3106f280413", "4adea7106f280000"),
        ("e0638df8a0029a4d", "5e469ff8a0020000"),
        ("7c41df5c178ab673", "6eecf95c178a0000"),
        ("58fb5375991fe1ba", "330d7f75991f0000"),
        ("909e7d5a18515662", "6248945a18510000"),
        ("4bd8be7123e5d34b", "ec51d07123e50000"),
    ];
    const FIX_PRIME56_OFFSET27: &[(&str, &str)] = &[
        ("0000000000000000", "0000000000000000"),
        ("0100000000000000", "0100000000000000"),
        ("0200000000000000", "0200000000000000"),
        ("e4ffffffffffff00", "e4ffffffffffff00"),
        ("e5ffffffffffff00", "0000000000000000"),
        ("ffffffffffffffff", "ff1a000000000000"),
        ("dae623d2aa29f418", "62e923d2aa29f400"),
        ("45b9a32a1d819bb6", "77cca32a1d819b00"),
        ("5e7ca3106f280413", "5f7ea3106f280400"),
        ("e0638df8a0029a4d", "ff6b8df8a0029a00"),
        ("7c41df5c178ab673", "9d4ddf5c178ab600"),
        ("58fb5375991fe1ba", "f60e5475991fe100"),
        ("909e7d5a18515662", "e6a87d5a18515600"),
        ("4bd8be7123e5d34b", "34e0be7123e5d300"),
    ];
    const FIX_PRIME64_OFFSET59: &[(&str, &str)] = &[
        ("0000000000000000", "0000000000000000"),
        ("0100000000000000", "0100000000000000"),
        ("0200000000000000", "0200000000000000"),
        ("c4ffffffffffffff", "c4ffffffffffffff"),
        ("c5ffffffffffffff", "0000000000000000"),
        ("ffffffffffffffff", "3a00000000000000"),
        ("dae623d2aa29f418", "dae623d2aa29f418"),
        ("45b9a32a1d819bb6", "45b9a32a1d819bb6"),
        ("5e7ca3106f280413", "5e7ca3106f280413"),
        ("e0638df8a0029a4d", "e0638df8a0029a4d"),
        ("7c41df5c178ab673", "7c41df5c178ab673"),
        ("58fb5375991fe1ba", "58fb5375991fe1ba"),
        ("909e7d5a18515662", "909e7d5a18515662"),
        ("4bd8be7123e5d34b", "4bd8be7123e5d34b"),
    ];
    const FIX_PRIME128_OFFSET275: &[(&str, &str)] = &[
        (
            "00000000000000000000000000000000",
            "00000000000000000000000000000000",
        ),
        (
            "01000000000000000000000000000000",
            "01000000000000000000000000000000",
        ),
        (
            "02000000000000000000000000000000",
            "02000000000000000000000000000000",
        ),
        (
            "ecfeffffffffffffffffffffffffffff",
            "ecfeffffffffffffffffffffffffffff",
        ),
        (
            "edfeffffffffffffffffffffffffffff",
            "00000000000000000000000000000000",
        ),
        (
            "ffffffffffffffffffffffffffffffff",
            "12010000000000000000000000000000",
        ),
        (
            "dae623d2aa29f41845b9a32a1d819bb6",
            "dae623d2aa29f41845b9a32a1d819bb6",
        ),
        (
            "5e7ca3106f280413e0638df8a0029a4d",
            "5e7ca3106f280413e0638df8a0029a4d",
        ),
        (
            "7c41df5c178ab67358fb5375991fe1ba",
            "7c41df5c178ab67358fb5375991fe1ba",
        ),
        (
            "909e7d5a185156624bd8be7123e5d34b",
            "909e7d5a185156624bd8be7123e5d34b",
        ),
        (
            "f0496a0db0d44aecb71a5457bac1cd1f",
            "f0496a0db0d44aecb71a5457bac1cd1f",
        ),
        (
            "004ed4e1b8a4ad0fda75893f75f9e34b",
            "004ed4e1b8a4ad0fda75893f75f9e34b",
        ),
        (
            "a3697a40313c6c012a94dd3346964418",
            "a3697a40313c6c012a94dd3346964418",
        ),
        (
            "2c96bcbf573df48ae896963ea3675f9e",
            "2c96bcbf573df48ae896963ea3675f9e",
        ),
    ];
    const FIX_PRIME128_OFFSETA7F7: &[(&str, &str)] = &[
        (
            "00000000000000000000000000000000",
            "00000000000000000000000000000000",
        ),
        (
            "01000000000000000000000000000000",
            "01000000000000000000000000000000",
        ),
        (
            "02000000000000000000000000000000",
            "02000000000000000000000000000000",
        ),
        (
            "08580000ffffffffffffffffffffffff",
            "08580000ffffffffffffffffffffffff",
        ),
        (
            "09580000ffffffffffffffffffffffff",
            "00000000000000000000000000000000",
        ),
        (
            "ffffffffffffffffffffffffffffffff",
            "f6a7ffff000000000000000000000000",
        ),
        (
            "dae623d2aa29f41845b9a32a1d819bb6",
            "dae623d2aa29f41845b9a32a1d819bb6",
        ),
        (
            "5e7ca3106f280413e0638df8a0029a4d",
            "5e7ca3106f280413e0638df8a0029a4d",
        ),
        (
            "7c41df5c178ab67358fb5375991fe1ba",
            "7c41df5c178ab67358fb5375991fe1ba",
        ),
        (
            "909e7d5a185156624bd8be7123e5d34b",
            "909e7d5a185156624bd8be7123e5d34b",
        ),
        (
            "f0496a0db0d44aecb71a5457bac1cd1f",
            "f0496a0db0d44aecb71a5457bac1cd1f",
        ),
        (
            "004ed4e1b8a4ad0fda75893f75f9e34b",
            "004ed4e1b8a4ad0fda75893f75f9e34b",
        ),
        (
            "a3697a40313c6c012a94dd3346964418",
            "a3697a40313c6c012a94dd3346964418",
        ),
        (
            "2c96bcbf573df48ae896963ea3675f9e",
            "2c96bcbf573df48ae896963ea3675f9e",
        ),
    ];
    const FIX_EXT2_P32: &[(&[u128], &str)] = &[
        (&[0x0, 0x0], "0000000000000000"),
        (&[0xffffff9c, 0xffffff9c], "9cffffff9cffffff"),
        (&[0x1, 0x0], "0100000000000000"),
        (&[0x73d60714, 0xf544c309], "1407d67309c344f5"),
        (&[0x6c49d008, 0x9b0ea42d], "08d0496c2da40e9b"),
        (&[0xfefbf7, 0x72cde54], "f7fbfe0054de2c07"),
        (&[0x5cee8e85, 0x8fba2793], "858eee5c9327ba8f"),
        (&[0xbbe03ed8, 0xe21d84c1], "d83ee0bbc1841de2"),
    ];
    const FIX_EXT4_P32: &[(&[u128], &str)] = &[
        (&[0x0, 0x0, 0x0, 0x0], "00000000000000000000000000000000"),
        (
            &[0xffffff9c, 0xffffff9c, 0xffffff9c, 0xffffff9c],
            "9cffffff9cffffff9cffffff9cffffff",
        ),
        (&[0x1, 0x0, 0x0, 0x0], "01000000000000000000000000000000"),
        (
            &[0x73d60714, 0xf544c309, 0x6c49d008, 0x9b0ea42d],
            "1407d67309c344f508d0496c2da40e9b",
        ),
        (
            &[0xfefbf7, 0x72cde54, 0x5cee8e85, 0x8fba2793],
            "f7fbfe0054de2c07858eee5c9327ba8f",
        ),
        (
            &[0xbbe03ed8, 0xe21d84c1, 0xe3b06389, 0x56be17ea],
            "d83ee0bbc1841de28963b0e3ea17be56",
        ),
        (
            &[0x644d482d, 0xa75fce19, 0xebf7eb94, 0xc4f0921b],
            "2d484d6419ce5fa794ebf7eb1b92f0c4",
        ),
        (
            &[0x84dac822, 0x62f6ac1c, 0xb5549d13, 0x723a51cd],
            "22c8da841cacf662139d54b5cd513a72",
        ),
    ];
    const FIX_EXT8_P32: &[(&[u128], &str)] = &[
        (
            &[0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0],
            "0000000000000000000000000000000000000000000000000000000000000000",
        ),
        (
            &[
                0xffffff9c, 0xffffff9c, 0xffffff9c, 0xffffff9c, 0xffffff9c, 0xffffff9c, 0xffffff9c,
                0xffffff9c,
            ],
            "9cffffff9cffffff9cffffff9cffffff9cffffff9cffffff9cffffff9cffffff",
        ),
        (
            &[0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0],
            "0100000000000000000000000000000000000000000000000000000000000000",
        ),
        (
            &[
                0x73d60714, 0xf544c309, 0x6c49d008, 0x9b0ea42d, 0xfefbf7, 0x72cde54, 0x5cee8e85,
                0x8fba2793,
            ],
            "1407d67309c344f508d0496c2da40e9bf7fbfe0054de2c07858eee5c9327ba8f",
        ),
        (
            &[
                0xbbe03ed8, 0xe21d84c1, 0xe3b06389, 0x56be17ea, 0x644d482d, 0xa75fce19, 0xebf7eb94,
                0xc4f0921b,
            ],
            "d83ee0bbc1841de28963b0e3ea17be562d484d6419ce5fa794ebf7eb1b92f0c4",
        ),
        (
            &[
                0x84dac822, 0x62f6ac1c, 0xb5549d13, 0x723a51cd, 0x56bd52bf, 0x4247ba16, 0xb9d843ba,
                0xe19d078e,
            ],
            "22c8da841cacf662139d54b5cd513a72bf52bd5616ba4742ba43d8b98e079de1",
        ),
        (
            &[
                0x2f8b79d, 0xec18080b, 0xa31f1e7d, 0x1471072, 0xeb5039b2, 0xa11bc9b4, 0x9c0517ed,
                0x81f10ed0,
            ],
            "9db7f8020b0818ec7d1e1fa372104701b23950ebb4c91ba1ed17059cd00ef181",
        ),
        (
            &[
                0xed9bed57, 0x29d9826, 0xceed3883, 0x60414ccc, 0xccc2b5c7, 0xaa6c7139, 0xcd6b7546,
                0xd9701579,
            ],
            "57ed9bed26989d028338edcecc4c4160c7b5c2cc39716caa46756bcd791570d9",
        ),
    ];
    const FIX_EXT2_P64: &[(&[u128], &str)] = &[
        (&[0x0, 0x0], "00000000000000000000000000000000"),
        (
            &[0xffffffffffffffc4, 0xffffffffffffffc4],
            "c4ffffffffffffffc4ffffffffffffff",
        ),
        (&[0x1, 0x0], "01000000000000000000000000000000"),
        (
            &[0xd609cf72b02c0e61, 0x79a6c5d7584f8181],
            "610e2cb072cf09d681814f58d7c5a679",
        ),
        (
            &[0x4d78586fe207e611, 0xc3d8b17f0be956fa],
            "11e607e26f58784dfa56e90b7fb1d8c3",
        ),
        (
            &[0xaa67d93208600cb0, 0xe93623b792a8a5d1],
            "b00c600832d967aad1a5a892b72336e9",
        ),
        (
            &[0x436b82b5306a7737, 0x74a285c1b520e858],
            "37776a30b5826b4358e820b5c185a274",
        ),
        (
            &[0x5ab69a96e218a143, 0x36706831fafb6742],
            "43a118e2969ab65a4267fbfa31687036",
        ),
    ];
    const FIX_EXT4_P64: &[(&[u128], &str)] = &[
        (
            &[0x0, 0x0, 0x0, 0x0],
            "0000000000000000000000000000000000000000000000000000000000000000",
        ),
        (
            &[
                0xffffffffffffffc4,
                0xffffffffffffffc4,
                0xffffffffffffffc4,
                0xffffffffffffffc4,
            ],
            "c4ffffffffffffffc4ffffffffffffffc4ffffffffffffffc4ffffffffffffff",
        ),
        (
            &[0x1, 0x0, 0x0, 0x0],
            "0100000000000000000000000000000000000000000000000000000000000000",
        ),
        (
            &[
                0xd609cf72b02c0e61,
                0x79a6c5d7584f8181,
                0x4d78586fe207e611,
                0xc3d8b17f0be956fa,
            ],
            "610e2cb072cf09d681814f58d7c5a67911e607e26f58784dfa56e90b7fb1d8c3",
        ),
        (
            &[
                0xaa67d93208600cb0,
                0xe93623b792a8a5d1,
                0x436b82b5306a7737,
                0x74a285c1b520e858,
            ],
            "b00c600832d967aad1a5a892b72336e937776a30b5826b4358e820b5c185a274",
        ),
        (
            &[
                0x5ab69a96e218a143,
                0x36706831fafb6742,
                0x9ee6382725ed15a0,
                0x17930127615c009a,
            ],
            "43a118e2969ab65a4267fbfa31687036a015ed252738e69e9a005c6127019317",
        ),
        (
            &[
                0x300e066995c7449,
                0x8bfc74fb11786f73,
                0x4cee8f1453becffe,
                0x6f3bdb40c4eede10,
            ],
            "49745c9966e00003736f7811fb74fc8bfecfbe53148fee4c10deeec440db3b6f",
        ),
        (
            &[
                0x67d84e6f52ff5ee7,
                0xa15bc70e42ea593,
                0x84acf5c3d050739,
                0xeb457d2053f67e6b,
            ],
            "e75eff526f4ed86793a52ee470bc150a3907053d5ccf4a086b7ef653207d45eb",
        ),
    ];
    const FIX_EXT8_P64: &[(&[u128], &str)] = &[
        (&[0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0], "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
        (&[0xffffffffffffffc4, 0xffffffffffffffc4, 0xffffffffffffffc4, 0xffffffffffffffc4, 0xffffffffffffffc4, 0xffffffffffffffc4, 0xffffffffffffffc4, 0xffffffffffffffc4], "c4ffffffffffffffc4ffffffffffffffc4ffffffffffffffc4ffffffffffffffc4ffffffffffffffc4ffffffffffffffc4ffffffffffffffc4ffffffffffffff"),
        (&[0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0], "01000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
        (&[0xd609cf72b02c0e61, 0x79a6c5d7584f8181, 0x4d78586fe207e611, 0xc3d8b17f0be956fa, 0xaa67d93208600cb0, 0xe93623b792a8a5d1, 0x436b82b5306a7737, 0x74a285c1b520e858], "610e2cb072cf09d681814f58d7c5a67911e607e26f58784dfa56e90b7fb1d8c3b00c600832d967aad1a5a892b72336e937776a30b5826b4358e820b5c185a274"),
        (&[0x5ab69a96e218a143, 0x36706831fafb6742, 0x9ee6382725ed15a0, 0x17930127615c009a, 0x300e066995c7449, 0x8bfc74fb11786f73, 0x4cee8f1453becffe, 0x6f3bdb40c4eede10], "43a118e2969ab65a4267fbfa31687036a015ed252738e69e9a005c612701931749745c9966e00003736f7811fb74fc8bfecfbe53148fee4c10deeec440db3b6f"),
        (&[0x67d84e6f52ff5ee7, 0xa15bc70e42ea593, 0x84acf5c3d050739, 0xeb457d2053f67e6b, 0x75e8506326be2e79, 0xfe788b2a7fd8a05b, 0x441c8c24b6702e2f, 0x3f2cf962f21f6a4c], "e75eff526f4ed86793a52ee470bc150a3907053d5ccf4a086b7ef653207d45eb792ebe266350e8755ba0d87f2a8b78fe2f2e70b6248c1c444c6a1ff262f92c3f"),
        (&[0x3a6b892c9811490e, 0x7e2e2981b2e0bac4, 0x5686aa90270939dc, 0xb46fec366af7a378, 0xbb7409b32081c57, 0xd0adcc227d135fcd, 0xb9de091a840ae78d, 0x7c7665093663d736], "0e4911982c896b3ac4bae0b281292e7edc39092790aa865678a3f76a36ec6fb4571c08329b40b70bcd5f137d22ccadd08de70a841a09deb936d763360965767c"),
        (&[0x204cf30bb30096d9, 0x7e4097b92c507936, 0x9b1d5c1eaf86cccf, 0x56f6215dbf2e4694, 0x4ffad9390b05e0bc, 0x2712c7a4f32f68b7, 0x91dd924eb2b0e58d, 0xbe3dec166c0263b4], "d99600b30bf34c203679502cb997407ecfcc86af1e5c1d9b94462ebf5d21f656bce0050b39d9fa4fb7682ff3a4c712278de5b0b24e92dd91b463026c16ec3dbe"),
    ];
    const FIX_EXT2_P128: &[(&[u128], &str)] = &[
        (
            &[0x0, 0x0],
            "0000000000000000000000000000000000000000000000000000000000000000",
        ),
        (
            &[
                0xfffffffffffffffffffffffffffffeec,
                0xfffffffffffffffffffffffffffffeec,
            ],
            "ecfeffffffffffffffffffffffffffffecfeffffffffffffffffffffffffffff",
        ),
        (
            &[0x1, 0x0],
            "0100000000000000000000000000000000000000000000000000000000000000",
        ),
        (
            &[
                0x95ec2d2a50b6523ad32e8e6a68bbeda,
                0x8ee0c318a649d9f48bd9cf29054a3eaa,
            ],
            "dabe8ba6e6e832ad23650ba5d2c25e09aa3e4a0529cfd98bf4d949a618c3e08e",
        ),
        (
            &[
                0x5849c2a2605eea307e85ab91ea9e645,
                0xbee878b6326d409bc444df816cbb691d,
            ],
            "45e6a91eb95ae807a3ee05262a9c84051d69bb6c81df44c49b406d32b678e8be",
        ),
        (
            &[
                0xdfc90e1016f022a317119b7cbf08055e,
                0xae1cf13c68e8d04672b6a28cfce256f,
            ],
            "5e0508bf7c9b1117a322f016100ec9df6f25cecf286a2b67048d8ec613cfe10a",
        ),
        (
            &[
                0x229e76f8b4d1758d48e6176384245de0,
                0x3e5f384d01415d9a14b08c026b1052a0,
            ],
            "e05d24846317e6488d75d1b4f8769e22a052106b028cb0149a5d41014d385f3e",
        ),
        (
            &[
                0x1afc7c5c909d9ddf2285f1418dc53d7c,
                0x75332873a5d3f2b633a6158ac3227117,
            ],
            "7c3dc58d41f18522df9d9d905c7cfc1a177122c38a15a633b6f2d3a573283375",
        ),
    ];
    const FIX_EXT4_P128: &[(&[u128], &str)] = &[
        (&[0x0, 0x0, 0x0, 0x0], "00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
        (&[0xfffffffffffffffffffffffffffffeec, 0xfffffffffffffffffffffffffffffeec, 0xfffffffffffffffffffffffffffffeec, 0xfffffffffffffffffffffffffffffeec], "ecfeffffffffffffffffffffffffffffecfeffffffffffffffffffffffffffffecfeffffffffffffffffffffffffffffecfeffffffffffffffffffffffffffff"),
        (&[0x1, 0x0, 0x0, 0x0], "01000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
        (&[0x95ec2d2a50b6523ad32e8e6a68bbeda, 0x8ee0c318a649d9f48bd9cf29054a3eaa, 0x5849c2a2605eea307e85ab91ea9e645, 0xbee878b6326d409bc444df816cbb691d], "dabe8ba6e6e832ad23650ba5d2c25e09aa3e4a0529cfd98bf4d949a618c3e08e45e6a91eb95ae807a3ee05262a9c84051d69bb6c81df44c49b406d32b678e8be"),
        (&[0xdfc90e1016f022a317119b7cbf08055e, 0xae1cf13c68e8d04672b6a28cfce256f, 0x229e76f8b4d1758d48e6176384245de0, 0x3e5f384d01415d9a14b08c026b1052a0], "5e0508bf7c9b1117a322f016100ec9df6f25cecf286a2b67048d8ec613cfe10ae05d24846317e6488d75d1b4f8769e22a052106b028cb0149a5d41014d385f3e"),
        (&[0x1afc7c5c909d9ddf2285f1418dc53d7c, 0x75332873a5d3f2b633a6158ac3227117, 0x17f99d75e2c2cb53185eedfbe3083858, 0x4bc8e3bab667e4e1a046851f576a3c99], "7c3dc58d41f18522df9d9d905c7cfc1a177122c38a15a633b6f2d3a573283375583808e3fbed5e1853cbc2e2759df917993c6a571f8546a0e1e467b6bae3c84b"),
        (&[0x8231505a2d6fb47d01a35b9e209dd490, 0x2f22436254aff856af16ed518cea3118, 0xbb73cc71d26357be193e70d8d6d98d4b, 0x9d19904b4e75a8d33a5799e5afd0ed23], "90d49d209e5ba3017db46f2d5a5031821831ea8c51ed16af56f8af546243222f4b8dd9d6d8703e19be5763d271cc73bb23edd0afe599573ad3a8754e4b90199d"),
        (&[0x2e8f3b0da794056aacd5b249b3e21cf0, 0xc9c392ecf3c2744a8a02dfd4b65dcdb0, 0x4ac6ac57e4534954cc81171a9dd31cb7, 0xeb2d691ffa99bccdb7ce42c19287eeba], "f01ce2b349b2d5ac6a0594a70d3b8f2eb0cd5db6d4df028a4a74c2f3ec92c3c9b71cd39d1a1781cc544953e457acc64abaee8792c142ceb7cdbc99fa1f692deb"),
    ];
    const FIX_EXT8_P128: &[(&[u128], &str)] = &[
        (&[0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0], "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
        (&[0xfffffffffffffffffffffffffffffeec, 0xfffffffffffffffffffffffffffffeec, 0xfffffffffffffffffffffffffffffeec, 0xfffffffffffffffffffffffffffffeec, 0xfffffffffffffffffffffffffffffeec, 0xfffffffffffffffffffffffffffffeec, 0xfffffffffffffffffffffffffffffeec, 0xfffffffffffffffffffffffffffffeec], "ecfeffffffffffffffffffffffffffffecfeffffffffffffffffffffffffffffecfeffffffffffffffffffffffffffffecfeffffffffffffffffffffffffffffecfeffffffffffffffffffffffffffffecfeffffffffffffffffffffffffffffecfeffffffffffffffffffffffffffffecfeffffffffffffffffffffffffffff"),
        (&[0x1, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0], "0100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000"),
        (&[0x95ec2d2a50b6523ad32e8e6a68bbeda, 0x8ee0c318a649d9f48bd9cf29054a3eaa, 0x5849c2a2605eea307e85ab91ea9e645, 0xbee878b6326d409bc444df816cbb691d, 0xdfc90e1016f022a317119b7cbf08055e, 0xae1cf13c68e8d04672b6a28cfce256f, 0x229e76f8b4d1758d48e6176384245de0, 0x3e5f384d01415d9a14b08c026b1052a0], "dabe8ba6e6e832ad23650ba5d2c25e09aa3e4a0529cfd98bf4d949a618c3e08e45e6a91eb95ae807a3ee05262a9c84051d69bb6c81df44c49b406d32b678e8be5e0508bf7c9b1117a322f016100ec9df6f25cecf286a2b67048d8ec613cfe10ae05d24846317e6488d75d1b4f8769e22a052106b028cb0149a5d41014d385f3e"),
        (&[0x1afc7c5c909d9ddf2285f1418dc53d7c, 0x75332873a5d3f2b633a6158ac3227117, 0x17f99d75e2c2cb53185eedfbe3083858, 0x4bc8e3bab667e4e1a046851f576a3c99, 0x8231505a2d6fb47d01a35b9e209dd490, 0x2f22436254aff856af16ed518cea3118, 0xbb73cc71d26357be193e70d8d6d98d4b, 0x9d19904b4e75a8d33a5799e5afd0ed23], "7c3dc58d41f18522df9d9d905c7cfc1a177122c38a15a633b6f2d3a573283375583808e3fbed5e1853cbc2e2759df917993c6a571f8546a0e1e467b6bae3c84b90d49d209e5ba3017db46f2d5a5031821831ea8c51ed16af56f8af546243222f4b8dd9d6d8703e19be5763d271cc73bb23edd0afe599573ad3a8754e4b90199d"),
        (&[0x2e8f3b0da794056aacd5b249b3e21cf0, 0xc9c392ecf3c2744a8a02dfd4b65dcdb0, 0x4ac6ac57e4534954cc81171a9dd31cb7, 0xeb2d691ffa99bccdb7ce42c19287eeba, 0x1e2f46e1fa26a6d48102fa4e7fd5ba00, 0x368c360f492c47ad6c2815a4a2a418b8, 0xa497383fb616bd8955429575bf3276da, 0xd71f974b8bb550e3aae51af9bf55ba75], "f01ce2b349b2d5ac6a0594a70d3b8f2eb0cd5db6d4df028a4a74c2f3ec92c3c9b71cd39d1a1781cc544953e457acc64abaee8792c142ceb7cdbc99fa1f692deb00bad57f4efa0281d4a626fae1462f1eb818a4a2a415286cad472c490f368c36da7632bf7595425589bd16b63f3897a475ba55bff91ae5aae350b58b4b971fd7"),
        (&[0x637068400d8c717a4f83826978b31ca3, 0x7d5682012c73756c9b3e333c7444a431, 0x6a467433fdab3dddd849e294b091f22a, 0xda8bc3186680c6445639f596cb49e646, 0x59db1ebf000b2cbc56372a962f74c82c, 0x3ff0528a61b7faf4144ac63df7ac8657, 0x2745433efaca6b96ace78996b76419e8, 0xb4ec249ef0d94b5fca09f467b44f6ea3], "a31cb3786982834f7a718c0d4068706331a444743c333e9b6c75732c0182567d2af291b094e249d8dd3dabfd3374466a46e649cb96f5395644c6806618c38bda2cc8742f962a3756bc2c0b00bf1edb595786acf73dc64a14f4fab7618a52f03fe81964b79689e7ac966bcafa3e434527a36e4fb467f409ca5f4bd9f09e24ecb4"),
        (&[0xce146918fcdfd134a198ba496b6b54cd, 0xe0645714d27314b6c72085ecabcaa748, 0x535d0db82474e0e464ab32ae4896f3e2, 0xf6a32e67c18093bc7f5a6f74268c2d1d, 0xed84f79242e677ce9255ca839fe83795, 0x811838518589bb5b667dccdb2c7133d4, 0xf800638fd1130f3469c6a029834c576e, 0x4ce9d686a152cec904597b0f3decb776], "cd546b6b49ba98a134d1dffc186914ce48a7caabec8520c7b61473d2145764e0e2f39648ae32ab64e4e07424b80d5d531d2d8c26746f5a7fbc9380c1672ea3f69537e89f83ca5592ce77e64292f784edd433712cdbcc7d665bbb8985513818816e574c8329a0c669340f13d18f6300f876b7ec3d0f7b5904c9ce52a186d6e94c"),
    ];

    /// Extension element from canonical coefficients; the bincode wire must
    /// equal the fixture and decode back.
    fn check_ext_rows<F, E>(rows: &[(&[u128], &str)])
    where
        F: CanonicalEncoding + two::Field,
        E: ExtField<F>
            + serde::Serialize
            + serde::de::DeserializeOwned
            + PartialEq
            + std::fmt::Debug
            + Copy,
    {
        let cfg = bincode::config::standard();
        for (coeffs, expected) in rows {
            let expected = unhex(expected);
            let e = E::from_base_slice(
                &coeffs
                    .iter()
                    .map(|&v| F::from_u128_checked(v).unwrap())
                    .collect::<Vec<_>>(),
            );
            let wire = bincode::serde::encode_to_vec(e, cfg).unwrap();
            assert_eq!(wire, expected, "ext wire bytes diverge");
            let (back, read): (E, usize) =
                bincode::serde::decode_from_slice(&expected, cfg).unwrap();
            assert_eq!((back, read), (e, expected.len()));
        }
    }

    #[test]
    fn fp32_bytes_match_fixtures() {
        check_prime_rows::<two::Prime24Offset3>(FIX_PRIME24_OFFSET3);
        check_prime_rows::<two::Prime30Offset35>(FIX_PRIME30_OFFSET35);
        check_prime_rows::<two::Prime31Offset19>(FIX_PRIME31_OFFSET19);
        check_prime_rows::<two::Prime32Offset99>(FIX_PRIME32_OFFSET99);
    }

    #[test]
    fn fp64_bytes_match_fixtures() {
        check_prime_rows::<two::Prime40Offset195>(FIX_PRIME40_OFFSET195);
        check_prime_rows::<two::Prime48Offset59>(FIX_PRIME48_OFFSET59);
        check_prime_rows::<two::Prime56Offset27>(FIX_PRIME56_OFFSET27);
        check_prime_rows::<two::Prime64Offset59>(FIX_PRIME64_OFFSET59);
    }

    #[test]
    fn fp128_bytes_match_fixtures() {
        check_prime_rows::<two::Prime128Offset275>(FIX_PRIME128_OFFSET275);
        check_prime_rows::<two::Prime128OffsetA7F7>(FIX_PRIME128_OFFSETA7F7);
    }

    #[test]
    fn ext_bytes_match_fixtures() {
        type F32 = two::Prime32Offset99;
        type F64 = two::Prime64Offset59;
        type F128 = two::Prime128Offset275;
        check_ext_rows::<F32, two::FpExt2<F32, two::TwoNr>>(FIX_EXT2_P32);
        check_ext_rows::<F32, two::FpExt4<F32>>(FIX_EXT4_P32);
        check_ext_rows::<F32, two::FpExt8<F32>>(FIX_EXT8_P32);
        check_ext_rows::<F64, two::FpExt2<F64, two::TwoNr>>(FIX_EXT2_P64);
        check_ext_rows::<F64, two::FpExt4<F64>>(FIX_EXT4_P64);
        check_ext_rows::<F64, two::FpExt8<F64>>(FIX_EXT8_P64);
        check_ext_rows::<F128, two::FpExt2<F128, two::TwoNr>>(FIX_EXT2_P128);
        check_ext_rows::<F128, two::FpExt4<F128>>(FIX_EXT4_P128);
        check_ext_rows::<F128, two::FpExt8<F128>>(FIX_EXT8_P128);
    }
}
