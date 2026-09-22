//! Generate compatibility fixtures from the pinned production transcript, never a reimplementation.
use blake2::{Blake2b512 as Digest512, Digest};
use jolt_field::{CanonicalBytes, Fr, Ring};
use jolt_transcript::{Bn254WideBlake2bTranscript, Transcript};
use spongefish::{instantiations::Blake2b512, DuplexSpongeInterface};

fn hex(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    for byte in bytes {
        out.push(char::from(DIGITS[usize::from(byte >> 4)]));
        out.push(char::from(DIGITS[usize::from(byte & 15)]));
    }
    out
}

#[derive(Clone)]
enum Op {
    Absorb(Vec<u8>),
    Field(u64),
    Squeeze(usize),
    Ratchet,
    Peek,
    Challenge,
}

impl Op {
    fn record(&self, expected: Option<&[u8]>) -> String {
        let (kind, data, length) = match self {
            Self::Absorb(data) => ("absorb", hex(data), 0),
            Self::Field(value) => {
                let mut data = [0; 32];
                Fr::from_u64(*value).to_bytes_le(&mut data);
                data.reverse();
                ("absorb", hex(&data), 0)
            }
            Self::Squeeze(n) => ("squeeze", String::new(), *n),
            Self::Ratchet => ("ratchet", String::new(), 0),
            Self::Peek => ("peek", String::new(), 0),
            Self::Challenge => ("challenge", String::new(), 0),
        };
        let result =
            expected.map_or_else(|| "null".to_owned(), |bytes| format!("\"{}\"", hex(bytes)));
        format!(
            "{{\"kind\":\"{kind}\",\"data\":\"{data}\",\"length\":{length},\"expected\":{result}}}"
        )
    }
}

fn raw(name: &str, ops: Vec<Op>) -> String {
    let mut sponge = Blake2b512::default();
    let records: Vec<_> = ops
        .iter()
        .map(|op| {
            let output = match op {
                Op::Absorb(bytes) => {
                    let _ = sponge.absorb(bytes);
                    None
                }
                Op::Squeeze(n) => {
                    let mut out = vec![0; *n];
                    let _ = sponge.squeeze(&mut out);
                    Some(out)
                }
                Op::Ratchet => {
                    let _ = sponge.ratchet();
                    None
                }
                Op::Peek => {
                    let mut out = vec![0; 32];
                    let _ = sponge.clone().squeeze(&mut out);
                    Some(out)
                }
                Op::Challenge | Op::Field(_) => unreachable!("raw fixture has no field decoder"),
            };
            op.record(output.as_deref())
        })
        .collect();
    format!(
        "{{\"name\":\"{name}\",\"mode\":\"raw\",\"label\":\"\",\"ops\":[{}]}}",
        records.join(",")
    )
}

fn wide(name: &str, label: &'static [u8], ops: Vec<Op>) -> String {
    let mut transcript = Bn254WideBlake2bTranscript::new(label);
    let records: Vec<_> = ops
        .iter()
        .map(|op| {
            let output = match op {
                Op::Absorb(bytes) => {
                    transcript.append_bytes(bytes);
                    None
                }
                Op::Field(value) => {
                    transcript.append(&Fr::from_u64(*value));
                    None
                }
                Op::Challenge => {
                    let value: Fr = transcript.challenge();
                    let mut out = vec![0; 32];
                    value.to_bytes_le(&mut out);
                    out.reverse();
                    Some(out)
                }
                Op::Peek => Some(transcript.state().to_vec()),
                Op::Squeeze(_) | Op::Ratchet => unreachable!("wide API exposes only field draws"),
            };
            op.record(output.as_deref())
        })
        .collect();
    format!(
        "{{\"name\":\"{name}\",\"mode\":\"wide\",\"label\":\"{}\",\"ops\":[{}]}}",
        hex(label),
        records.join(",")
    )
}

#[expect(
    clippy::print_stdout,
    reason = "fixture generator writes JSON to stdout"
)]
fn main() {
    assert_eq!(
        usize::BITS,
        64,
        "fixtures freeze the native 64-bit counter profile"
    );
    let mut hashes = Vec::new();
    for n in [0, 1, 3, 127, 128, 129, 255, 256, 257, 1024] {
        let data: Vec<_> = (0..n).map(|i| (i % 251) as u8).collect();
        hashes.push(format!(
            "{{\"data\":\"{}\",\"expected\":\"{}\"}}",
            hex(&data),
            hex(&Digest512::digest(&data))
        ));
    }
    let mut cases = Vec::new();
    for n in [0, 1, 16, 32, 63, 64, 65, 128, 129] {
        cases.push(raw(
            &format!("partial-{n}"),
            vec![
                Op::Absorb(b"abc".to_vec()),
                Op::Squeeze(n),
                Op::Absorb(b"next".to_vec()),
                Op::Squeeze(80),
            ],
        ));
    }
    cases.push(raw(
        "streaming",
        vec![
            Op::Absorb(b"a".to_vec()),
            Op::Absorb(b"bc".to_vec()),
            Op::Squeeze(1),
            Op::Squeeze(63),
            Op::Squeeze(65),
            Op::Absorb(vec![]),
            Op::Squeeze(64),
        ],
    ));
    cases.push(raw(
        "zero-and-ratchet",
        vec![
            Op::Squeeze(0),
            Op::Absorb(vec![]),
            Op::Ratchet,
            Op::Ratchet,
            Op::Squeeze(17),
            Op::Ratchet,
            Op::Squeeze(64),
        ],
    ));
    cases.push(raw(
        "peek",
        vec![
            Op::Absorb(b"peek".to_vec()),
            Op::Peek,
            Op::Peek,
            Op::Squeeze(31),
            Op::Peek,
            Op::Squeeze(65),
        ],
    ));
    for label in [
        b"vector".as_slice(),
        b"vector\0",
        b"other",
        b"spartan-preprocessed-clear-v2",
        b"",
        b"0123456789abcdef0123456789abcdef",
    ] {
        cases.push(wide(
            &format!("wide-{}", hex(label)),
            label,
            vec![
                Op::Peek,
                Op::Absorb(b"statement".to_vec()),
                Op::Field(0),
                Op::Field(u64::MAX),
                Op::Challenge,
                Op::Challenge,
                Op::Peek,
                Op::Challenge,
                Op::Absorb(vec![]),
                Op::Challenge,
                Op::Absorb(vec![42; 129]),
                Op::Challenge,
            ],
        ));
    }
    cases.push(wide(
        "framing-split",
        b"framing",
        vec![
            Op::Absorb(b"a".to_vec()),
            Op::Absorb(b"bc".to_vec()),
            Op::Challenge,
        ],
    ));
    cases.push(wide(
        "framing-joined",
        b"framing",
        vec![Op::Absorb(b"abc".to_vec()), Op::Challenge],
    ));
    println!("{{\"native_base\":\"c281860b26b4550f79343548d673080cb8a080a4\",\"spongefish\":\"d2d190b1329d35ac9577438d05aed4f17a57b9f9\",\"usize_bits\":64,\"hashes\":[{}],\"cases\":[{}]}}", hashes.join(","), cases.join(","));
}
