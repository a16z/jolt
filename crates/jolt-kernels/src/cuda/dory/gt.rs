use std::cell::RefCell;
use std::ops::{Add, Mul, Neg, Sub};

use ark_ec::pairing::{MillerLoopOutput, Pairing};
use ark_serialize::{Read, Write};
use dory::backends::arkworks::{ArkFr, ArkGT};
use dory::primitives::arithmetic::Group as DoryGroup;
use dory::primitives::{Compress, DoryDeserialize, DorySerialize, SerializationError, Validate};

use super::arena;
use crate::cuda::common::pairing::FQ12_LIMBS;

#[derive(Clone, Copy)]
struct Request {
    g1_offset: usize,
    g2_offset: usize,
    count: usize,
}

#[expect(
    clippy::large_enum_variant,
    reason = "a prove holds under a hundred GT slots, so 384 bytes each is not worth an indirection"
)]
enum Slot {
    Pending(Request),
    Ready(ArkGT),
}

thread_local! {
    static SLOTS: RefCell<Vec<Slot>> = const { RefCell::new(Vec::new()) };
}

pub(super) fn reset() {
    SLOTS.with_borrow_mut(Vec::clear);
}

#[derive(Clone, Copy, Debug)]
pub struct DeviceGT {
    index: u32,
}

fn push(slot: Slot) -> DeviceGT {
    SLOTS.with_borrow_mut(|slots| {
        slots.push(slot);
        let index = u32::try_from(slots.len() - 1).unwrap_or(u32::MAX);
        DeviceGT { index }
    })
}

#[expect(
    clippy::large_types_passed_by_value,
    reason = "ArkGT is Copy and is moved straight into the slot; a reference would only add a deref"
)]
fn host(value: ArkGT) -> DeviceGT {
    push(Slot::Ready(value))
}

fn final_exponentiation(limbs: &[u64]) -> Option<ArkGT> {
    <ark_bn254::Bn254 as Pairing>::final_exponentiation(MillerLoopOutput(super::curve::fq12(limbs)))
        .map(ArkGT)
}

fn flush() {
    let batches = SLOTS.with_borrow(|slots| {
        let mut grouped: Vec<(usize, Vec<(usize, Request)>)> = Vec::new();
        for (index, slot) in slots.iter().enumerate() {
            if let Slot::Pending(request) = slot {
                match grouped
                    .iter_mut()
                    .find(|(count, _)| *count == request.count)
                {
                    Some((_, members)) => members.push((index, *request)),
                    None => grouped.push((request.count, vec![(index, *request)])),
                }
            }
        }
        grouped
    });

    for (count, members) in batches {
        let span = tracing::info_span!("cuda_miller_batch", pairs = count, lanes = members.len());
        let _entered = span.enter();
        let spans: Vec<(usize, usize)> = members
            .iter()
            .map(|(_, request)| (request.g1_offset, request.g2_offset))
            .collect();
        let limbs = match arena::multi_miller_batch(&spans, count) {
            Ok(limbs) => limbs,
            Err(error) => {
                tracing::error!(?error, "the batched multi-Miller loop failed");
                arena::poison("a batched multi-Miller loop failed");
                continue;
            }
        };
        for (lane, (index, _)) in members.iter().enumerate() {
            let start = lane * FQ12_LIMBS;
            let Some(value) = limbs
                .get(start..start + FQ12_LIMBS)
                .and_then(final_exponentiation)
            else {
                arena::poison("a batched Miller output was degenerate");
                continue;
            };
            SLOTS.with_borrow_mut(|slots| {
                if let Some(slot) = slots.get_mut(*index) {
                    *slot = Slot::Ready(value);
                }
            });
        }
    }
}

impl DeviceGT {
    pub(super) fn enqueue(g1_offset: usize, g2_offset: usize, count: usize) -> Self {
        push(Slot::Pending(Request {
            g1_offset,
            g2_offset,
            count,
        }))
    }

    pub(super) fn value(self) -> ArkGT {
        let ready = SLOTS.with_borrow(|slots| match slots.get(self.index as usize) {
            Some(Slot::Ready(value)) => Some(*value),
            _ => None,
        });
        if let Some(value) = ready {
            return value;
        }
        flush();
        SLOTS
            .with_borrow(|slots| match slots.get(self.index as usize) {
                Some(Slot::Ready(value)) => Some(*value),
                _ => None,
            })
            .unwrap_or_else(|| {
                arena::poison("a deferred GT value never materialised");
                ArkGT::default()
            })
    }
}

impl From<ArkGT> for DeviceGT {
    fn from(value: ArkGT) -> Self {
        host(value)
    }
}

impl PartialEq for DeviceGT {
    fn eq(&self, other: &Self) -> bool {
        self.index == other.index || self.value() == other.value()
    }
}

impl Add for DeviceGT {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        host(self.value() + rhs.value())
    }
}

impl Add<&Self> for DeviceGT {
    type Output = Self;
    fn add(self, rhs: &Self) -> Self {
        host(self.value() + rhs.value())
    }
}

impl Sub for DeviceGT {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        host(self.value() - rhs.value())
    }
}

impl Sub<&Self> for DeviceGT {
    type Output = Self;
    fn sub(self, rhs: &Self) -> Self {
        host(self.value() - rhs.value())
    }
}

impl Neg for DeviceGT {
    type Output = Self;
    fn neg(self) -> Self {
        host(-self.value())
    }
}

impl Mul<DeviceGT> for ArkFr {
    type Output = DeviceGT;
    fn mul(self, rhs: DeviceGT) -> DeviceGT {
        host(self * rhs.value())
    }
}

impl Mul<&DeviceGT> for ArkFr {
    type Output = DeviceGT;
    fn mul(self, rhs: &DeviceGT) -> DeviceGT {
        host(self * rhs.value())
    }
}

impl DorySerialize for DeviceGT {
    fn serialize_with_mode<W: Write>(
        &self,
        writer: W,
        compress: Compress,
    ) -> Result<(), SerializationError> {
        self.value().serialize_with_mode(writer, compress)
    }

    fn serialized_size(&self, compress: Compress) -> usize {
        self.value().serialized_size(compress)
    }
}

impl DoryDeserialize for DeviceGT {
    fn deserialize_with_mode<R: Read>(
        reader: R,
        compress: Compress,
        validate: Validate,
    ) -> Result<Self, SerializationError> {
        Ok(host(ArkGT::deserialize_with_mode(
            reader, compress, validate,
        )?))
    }
}

impl DoryGroup for DeviceGT {
    type Scalar = ArkFr;

    fn identity() -> Self {
        host(ArkGT::identity())
    }

    fn add(&self, rhs: &Self) -> Self {
        host(DoryGroup::add(&self.value(), &rhs.value()))
    }

    fn neg(&self) -> Self {
        host(DoryGroup::neg(&self.value()))
    }

    fn scale(&self, k: &Self::Scalar) -> Self {
        host(DoryGroup::scale(&self.value(), k))
    }

    fn random() -> Self {
        host(ArkGT::random())
    }
}
