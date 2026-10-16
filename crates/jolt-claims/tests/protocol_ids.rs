#![expect(
    clippy::expect_used,
    reason = "test fixtures must construct successfully"
)]

use jolt_claims::{InputClaims, MissingOpeningValue, OutputClaims, SumcheckChallenges};
use jolt_field::{Fr, Ring};

use ids::{
    ChallengeId, CommittedPolynomial, Draw, OpeningId, Payload, RelationId, VirtualPolynomial,
};

mod ids {
    #[derive(Debug, PartialEq)]
    pub enum RelationId {
        Producer,
        Other,
    }

    #[derive(Debug, PartialEq)]
    pub enum Payload {
        Flag,
    }

    #[derive(Debug, PartialEq)]
    pub enum VirtualPolynomial {
        Value,
        Indexed(usize),
        Tagged(Payload),
    }

    #[derive(Debug, PartialEq)]
    pub enum CommittedPolynomial {
        Column,
        Indexed(usize),
    }

    #[derive(Debug, PartialEq)]
    pub enum OpeningId {
        Virtual(VirtualPolynomial, RelationId),
        Committed(CommittedPolynomial, RelationId),
    }

    impl OpeningId {
        pub fn virtual_polynomial(polynomial: VirtualPolynomial, relation: RelationId) -> Self {
            Self::Virtual(polynomial, relation)
        }

        pub fn committed(polynomial: CommittedPolynomial, relation: RelationId) -> Self {
            Self::Committed(polynomial, relation)
        }
    }

    #[derive(Debug, PartialEq)]
    pub enum ChallengeId {
        First,
        Second,
        Unused,
    }

    pub enum Draw {
        First,
        Second,
    }

    impl From<Draw> for ChallengeId {
        fn from(draw: Draw) -> Self {
            match draw {
                Draw::First => Self::First,
                Draw::Second => Self::Second,
            }
        }
    }
}

#[derive(Debug, PartialEq, OutputClaims)]
#[protocol(ids = crate::ids)]
#[relation(Producer)]
struct Produced<C> {
    #[opening(Value)]
    value: C,
    #[opening(Indexed)]
    indexed: Vec<C>,
    #[opening(committed = Column)]
    optional: Option<C>,
    #[opening(Tagged(Payload::Flag))]
    tagged: C,
    #[opening(committed = Indexed)]
    committed: Vec<C>,
}

#[derive(InputClaims)]
#[protocol(ids = ids)]
struct Consumed<C> {
    #[opening(Value, from = Producer)]
    value: C,
    #[opening(Indexed, from = Other)]
    indexed: Vec<C>,
    #[opening(committed = Column, from = Producer)]
    optional: Option<C>,
    #[opening(Tagged(Payload::Flag), from = Other)]
    tagged: C,
    #[opening(committed = Indexed, from = Other)]
    committed: Vec<C>,
}

#[derive(SumcheckChallenges)]
#[protocol(ids = crate::ids)]
struct Drawn<F> {
    #[challenge(Draw::First)]
    first: F,
    #[challenge(Draw::Second)]
    second: F,
}

#[test]
fn custom_output_ids_order_resolve_and_construct_from_literals() {
    let values = Produced {
        value: Fr::from_u64(11),
        indexed: vec![Fr::from_u64(13), Fr::from_u64(17)],
        optional: Some(Fr::from_u64(19)),
        tagged: Fr::from_u64(23),
        committed: vec![Fr::from_u64(29)],
    };
    let expected = vec![
        OpeningId::Virtual(VirtualPolynomial::Value, RelationId::Producer),
        OpeningId::Virtual(VirtualPolynomial::Indexed(0), RelationId::Producer),
        OpeningId::Virtual(VirtualPolynomial::Indexed(1), RelationId::Producer),
        OpeningId::Committed(CommittedPolynomial::Column, RelationId::Producer),
        OpeningId::Virtual(
            VirtualPolynomial::Tagged(Payload::Flag),
            RelationId::Producer,
        ),
        OpeningId::Committed(CommittedPolynomial::Indexed(0), RelationId::Producer),
    ];
    assert_eq!(values.canonical_order(), expected);
    for (id, value) in expected.iter().zip([11, 13, 17, 19, 23, 29]) {
        assert_eq!(values.resolve_output(id), Some(Fr::from_u64(value)));
    }
    assert_eq!(
        values.resolve_output(&OpeningId::Virtual(
            VirtualPolynomial::Value,
            RelationId::Other
        )),
        None,
    );
    assert_eq!(
        values.resolve_output(&OpeningId::Virtual(
            VirtualPolynomial::Indexed(2),
            RelationId::Producer
        )),
        None,
    );

    let constructed = Produced::<Fr>::from_opening_values(|id| match id {
        OpeningId::Virtual(VirtualPolynomial::Value, RelationId::Producer) => {
            Some(Fr::from_u64(11))
        }
        OpeningId::Virtual(VirtualPolynomial::Indexed(0), RelationId::Producer) => {
            Some(Fr::from_u64(13))
        }
        OpeningId::Virtual(VirtualPolynomial::Indexed(1), RelationId::Producer) => {
            Some(Fr::from_u64(17))
        }
        OpeningId::Committed(CommittedPolynomial::Column, RelationId::Producer) => {
            Some(Fr::from_u64(19))
        }
        OpeningId::Virtual(VirtualPolynomial::Tagged(Payload::Flag), RelationId::Producer) => {
            Some(Fr::from_u64(23))
        }
        OpeningId::Committed(CommittedPolynomial::Indexed(0), RelationId::Producer) => {
            Some(Fr::from_u64(29))
        }
        _ => None,
    })
    .expect("literal opening values supply every required field");
    assert_eq!(constructed, values);

    let absent = Produced::<Fr>::from_opening_values(|id| match id {
        OpeningId::Virtual(VirtualPolynomial::Value, RelationId::Producer) => {
            Some(Fr::from_u64(31))
        }
        OpeningId::Virtual(VirtualPolynomial::Tagged(Payload::Flag), RelationId::Producer) => {
            Some(Fr::from_u64(37))
        }
        _ => None,
    })
    .expect("optional and indexed leaves may be absent");
    assert_eq!(
        absent,
        Produced {
            value: Fr::from_u64(31),
            indexed: vec![],
            optional: None,
            tagged: Fr::from_u64(37),
            committed: vec![],
        }
    );
    assert_eq!(
        absent.canonical_order(),
        vec![
            OpeningId::Virtual(VirtualPolynomial::Value, RelationId::Producer),
            OpeningId::Virtual(
                VirtualPolynomial::Tagged(Payload::Flag),
                RelationId::Producer
            ),
        ]
    );
    assert_eq!(
        absent.resolve_output(&OpeningId::Committed(
            CommittedPolynomial::Column,
            RelationId::Producer
        )),
        None
    );
    assert_eq!(
        Produced::<Fr>::from_opening_values(|_| None),
        Err(MissingOpeningValue {
            id: OpeningId::Virtual(VirtualPolynomial::Value, RelationId::Producer),
        })
    );
}

#[test]
fn custom_input_ids_preserve_sources_and_optional_presence() {
    let values = Consumed {
        value: Fr::from_u64(41),
        indexed: vec![Fr::from_u64(43)],
        optional: Some(Fr::from_u64(47)),
        tagged: Fr::from_u64(53),
        committed: vec![Fr::from_u64(59)],
    };
    let expected = vec![
        OpeningId::Virtual(VirtualPolynomial::Value, RelationId::Producer),
        OpeningId::Virtual(VirtualPolynomial::Indexed(0), RelationId::Other),
        OpeningId::Committed(CommittedPolynomial::Column, RelationId::Producer),
        OpeningId::Virtual(VirtualPolynomial::Tagged(Payload::Flag), RelationId::Other),
        OpeningId::Committed(CommittedPolynomial::Indexed(0), RelationId::Other),
    ];
    assert_eq!(values.canonical_order(), expected);
    for (id, value) in expected.iter().zip([41, 43, 47, 53, 59]) {
        assert_eq!(values.resolve_input(id), Some(Fr::from_u64(value)));
    }
    assert_eq!(
        values.resolve_input(&OpeningId::Virtual(
            VirtualPolynomial::Indexed(0),
            RelationId::Producer
        )),
        None
    );
    assert_eq!(
        values.resolve_input(&OpeningId::Committed(
            CommittedPolynomial::Indexed(1),
            RelationId::Other
        )),
        None
    );
    let absent = Consumed {
        optional: None,
        ..values
    };
    assert_eq!(
        absent.canonical_order(),
        vec![
            OpeningId::Virtual(VirtualPolynomial::Value, RelationId::Producer),
            OpeningId::Virtual(VirtualPolynomial::Indexed(0), RelationId::Other),
            OpeningId::Virtual(VirtualPolynomial::Tagged(Payload::Flag), RelationId::Other),
            OpeningId::Committed(CommittedPolynomial::Indexed(0), RelationId::Other),
        ]
    );
    assert_eq!(
        absent.resolve_input(&OpeningId::Committed(
            CommittedPolynomial::Column,
            RelationId::Producer
        )),
        None
    );
}

#[test]
fn custom_challenge_ids_resolve_literal_values() {
    let values = Drawn::from_transcript_values([Fr::from_u64(61), Fr::from_u64(67)].into_iter())
        .expect("literal challenge values supply both fields");
    assert_eq!(
        values.resolve_challenge(&ChallengeId::First),
        Some(Fr::from_u64(61))
    );
    assert_eq!(
        values.resolve_challenge(&ChallengeId::Second),
        Some(Fr::from_u64(67))
    );
    assert_eq!(values.resolve_challenge(&ChallengeId::Unused), None);
}
