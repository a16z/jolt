//! Guest-specific serialization for zkVM programs.
//!
//! This is **not** the transport (wire) encoding. Transport remains `ark_serialize` canonical
//! encoding (typically compressed) and is validated in native execution.
//!
//! The goal of this module is to define a Jolt-owned encoding optimized for guest verification:
//! - curve points are encoded in a form that avoids expensive decompression in-guest
//! - field elements are encoded as Montgomery limbs to avoid Montgomery conversion in-guest
//!
//! # Trust model / security note
//!
//! This encoding is intended for **trusted** inputs produced by a trusted host pipeline (e.g.
//! `jolt_sdk::decompress_transport_bytes_to_guest_bytes`) that has already decoded/validated the
//! transport encoding.
//!
//! Some `GuestDeserialize` impls intentionally perform **unchecked** construction (e.g. using
//! `new_unchecked` for field elements / curve points) to avoid expensive validation inside the
//! zkVM. Do **not** feed prover-controlled bytes directly into `GuestDeserialize` unless you add
//! your own validation layer.

use std::io;
use std::io::{Read, Write};

use ark_ff::BigInt;

pub trait GuestSerialize {
    fn guest_serialize<W: Write>(&self, w: &mut W) -> io::Result<()>;
}

/// Guest-specific deserialization for zkVM programs.
///
/// # Trust model / SAFETY
///
/// This trait is designed for a **trusted aggregation pipeline** where the host is assumed
/// honest. In particular, some implementations intentionally avoid validation (e.g. construct
/// BN254 field elements / curve points via `new_unchecked`) for performance.
///
/// Only use `guest_deserialize` on bytes that originate from a trusted/validated producer (for
/// example, the output of `jolt_sdk::decompress_transport_bytes_to_guest_bytes`).
pub trait GuestDeserialize: Sized {
    fn guest_deserialize<R: Read>(r: &mut R) -> io::Result<Self>;
}

#[inline(always)]
fn read_u8<R: Read>(r: &mut R) -> io::Result<u8> {
    let mut b = [0u8; 1];
    r.read_exact(&mut b)?;
    Ok(b[0])
}

#[inline(always)]
fn write_u8<W: Write>(w: &mut W, v: u8) -> io::Result<()> {
    w.write_all(&[v])
}

#[inline(always)]
fn read_u16<R: Read>(r: &mut R) -> io::Result<u16> {
    let mut b = [0u8; 2];
    r.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}

#[inline(always)]
fn write_u16<W: Write>(w: &mut W, v: u16) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

#[inline(always)]
fn read_u32<R: Read>(r: &mut R) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

#[inline(always)]
fn write_u32<W: Write>(w: &mut W, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

#[inline(always)]
fn read_u64<R: Read>(r: &mut R) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}

#[inline(always)]
fn write_u64<W: Write>(w: &mut W, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

#[inline(always)]
fn read_i128<R: Read>(r: &mut R) -> io::Result<i128> {
    let mut b = [0u8; 16];
    r.read_exact(&mut b)?;
    Ok(i128::from_le_bytes(b))
}

#[inline(always)]
fn write_i128<W: Write>(w: &mut W, v: i128) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

impl GuestSerialize for u8 {
    fn guest_serialize<W: Write>(&self, w: &mut W) -> io::Result<()> {
        write_u8(w, *self)
    }
}
impl GuestDeserialize for u8 {
    fn guest_deserialize<R: Read>(r: &mut R) -> io::Result<Self> {
        read_u8(r)
    }
}

impl GuestSerialize for bool {
    fn guest_serialize<W: Write>(&self, w: &mut W) -> io::Result<()> {
        write_u8(w, u8::from(*self))
    }
}
impl GuestDeserialize for bool {
    fn guest_deserialize<R: Read>(r: &mut R) -> io::Result<Self> {
        Ok(read_u8(r)? != 0)
    }
}

impl GuestSerialize for u16 {
    fn guest_serialize<W: Write>(&self, w: &mut W) -> io::Result<()> {
        write_u16(w, *self)
    }
}
impl GuestDeserialize for u16 {
    fn guest_deserialize<R: Read>(r: &mut R) -> io::Result<Self> {
        read_u16(r)
    }
}

impl GuestSerialize for u32 {
    fn guest_serialize<W: Write>(&self, w: &mut W) -> io::Result<()> {
        write_u32(w, *self)
    }
}
impl GuestDeserialize for u32 {
    fn guest_deserialize<R: Read>(r: &mut R) -> io::Result<Self> {
        read_u32(r)
    }
}

impl GuestSerialize for u64 {
    fn guest_serialize<W: Write>(&self, w: &mut W) -> io::Result<()> {
        write_u64(w, *self)
    }
}
impl GuestDeserialize for u64 {
    fn guest_deserialize<R: Read>(r: &mut R) -> io::Result<Self> {
        read_u64(r)
    }
}

impl GuestSerialize for usize {
    fn guest_serialize<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let v = u64::try_from(*self)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "usize overflow"))?;
        write_u64(w, v)
    }
}
impl GuestDeserialize for usize {
    fn guest_deserialize<R: Read>(r: &mut R) -> io::Result<Self> {
        let v = read_u64(r)?;
        usize::try_from(v).map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "usize overflow"))
    }
}

impl GuestSerialize for i128 {
    fn guest_serialize<W: Write>(&self, w: &mut W) -> io::Result<()> {
        write_i128(w, *self)
    }
}
impl GuestDeserialize for i128 {
    fn guest_deserialize<R: Read>(r: &mut R) -> io::Result<Self> {
        read_i128(r)
    }
}

impl<T: GuestSerialize> GuestSerialize for Option<T> {
    fn guest_serialize<W: Write>(&self, w: &mut W) -> io::Result<()> {
        match self {
            None => write_u8(w, 0),
            Some(v) => {
                write_u8(w, 1)?;
                v.guest_serialize(w)
            }
        }
    }
}
impl<T: GuestDeserialize> GuestDeserialize for Option<T> {
    fn guest_deserialize<R: Read>(r: &mut R) -> io::Result<Self> {
        match read_u8(r)? {
            0 => Ok(None),
            1 => Ok(Some(T::guest_deserialize(r)?)),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid Option tag",
            )),
        }
    }
}

impl<T: GuestSerialize> GuestSerialize for Vec<T> {
    fn guest_serialize<W: Write>(&self, w: &mut W) -> io::Result<()> {
        let len = u64::try_from(self.len())
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Vec length overflow"))?;
        write_u64(w, len)?;
        for v in self {
            v.guest_serialize(w)?;
        }
        Ok(())
    }
}
impl<T: GuestDeserialize> GuestDeserialize for Vec<T> {
    fn guest_deserialize<R: Read>(r: &mut R) -> io::Result<Self> {
        let len = usize::try_from(read_u64(r)?)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "Vec length overflow"))?;
        let mut out = Vec::with_capacity(len);
        for _ in 0..len {
            out.push(T::guest_deserialize(r)?);
        }
        Ok(out)
    }
}

impl<A: GuestSerialize, B: GuestSerialize> GuestSerialize for (A, B) {
    fn guest_serialize<W: Write>(&self, w: &mut W) -> io::Result<()> {
        self.0.guest_serialize(w)?;
        self.1.guest_serialize(w)?;
        Ok(())
    }
}
impl<A: GuestDeserialize, B: GuestDeserialize> GuestDeserialize for (A, B) {
    fn guest_deserialize<R: Read>(r: &mut R) -> io::Result<Self> {
        Ok((A::guest_deserialize(r)?, B::guest_deserialize(r)?))
    }
}

/// Encode a raw byte slice as (u64 length) + bytes.
#[inline(always)]
pub fn guest_serialize_bytes<W: Write>(w: &mut W, bytes: &[u8]) -> io::Result<()> {
    let len = u64::try_from(bytes.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "byte slice length overflow"))?;
    write_u64(w, len)?;
    w.write_all(bytes)
}

/// Decode a raw byte vector as (u64 length) + bytes.
#[inline(always)]
pub fn guest_deserialize_bytes<R: Read>(r: &mut R) -> io::Result<Vec<u8>> {
    let len = usize::try_from(read_u64(r)?)
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidData, "byte slice length overflow"))?;
    let mut out = vec![0u8; len];
    r.read_exact(&mut out)?;
    Ok(out)
}

// ---------------- BN254 field / extension fields (Montgomery-limb encoding) ----------------

impl GuestSerialize for ark_bn254::Fr {
    fn guest_serialize<W: Write>(&self, w: &mut W) -> io::Result<()> {
        // `Fr` stores Montgomery limbs in `self.0`.
        for limb in self.0.as_ref().iter().take(4) {
            write_u64(w, *limb)?;
        }
        Ok(())
    }
}
impl GuestDeserialize for ark_bn254::Fr {
    fn guest_deserialize<R: Read>(r: &mut R) -> io::Result<Self> {
        let mut limbs = [0u64; 4];
        for i in 0..4 {
            limbs[i] = read_u64(r)?;
        }
        Ok(ark_bn254::Fr::new_unchecked(BigInt(limbs)))
    }
}

impl GuestSerialize for ark_bn254::Fq {
    fn guest_serialize<W: Write>(&self, w: &mut W) -> io::Result<()> {
        for limb in self.0.as_ref().iter().take(4) {
            write_u64(w, *limb)?;
        }
        Ok(())
    }
}
impl GuestDeserialize for ark_bn254::Fq {
    fn guest_deserialize<R: Read>(r: &mut R) -> io::Result<Self> {
        let mut limbs = [0u64; 4];
        for i in 0..4 {
            limbs[i] = read_u64(r)?;
        }
        Ok(ark_bn254::Fq::new_unchecked(BigInt(limbs)))
    }
}

impl GuestSerialize for ark_bn254::Fq2 {
    fn guest_serialize<W: Write>(&self, w: &mut W) -> io::Result<()> {
        self.c0.guest_serialize(w)?;
        self.c1.guest_serialize(w)?;
        Ok(())
    }
}
impl GuestDeserialize for ark_bn254::Fq2 {
    fn guest_deserialize<R: Read>(r: &mut R) -> io::Result<Self> {
        let c0 = ark_bn254::Fq::guest_deserialize(r)?;
        let c1 = ark_bn254::Fq::guest_deserialize(r)?;
        Ok(ark_bn254::Fq2 { c0, c1 })
    }
}

impl GuestSerialize for ark_bn254::Fq6 {
    fn guest_serialize<W: Write>(&self, w: &mut W) -> io::Result<()> {
        self.c0.guest_serialize(w)?;
        self.c1.guest_serialize(w)?;
        self.c2.guest_serialize(w)?;
        Ok(())
    }
}
impl GuestDeserialize for ark_bn254::Fq6 {
    fn guest_deserialize<R: Read>(r: &mut R) -> io::Result<Self> {
        let c0 = ark_bn254::Fq2::guest_deserialize(r)?;
        let c1 = ark_bn254::Fq2::guest_deserialize(r)?;
        let c2 = ark_bn254::Fq2::guest_deserialize(r)?;
        Ok(ark_bn254::Fq6 { c0, c1, c2 })
    }
}

impl GuestSerialize for ark_bn254::Fq12 {
    fn guest_serialize<W: Write>(&self, w: &mut W) -> io::Result<()> {
        self.c0.guest_serialize(w)?;
        self.c1.guest_serialize(w)?;
        Ok(())
    }
}
impl GuestDeserialize for ark_bn254::Fq12 {
    fn guest_deserialize<R: Read>(r: &mut R) -> io::Result<Self> {
        let c0 = ark_bn254::Fq6::guest_deserialize(r)?;
        let c1 = ark_bn254::Fq6::guest_deserialize(r)?;
        Ok(ark_bn254::Fq12 { c0, c1 })
    }
}

// ---------------- BN254 curve points (uncompressed affine encoding) ----------------

impl GuestSerialize for ark_bn254::g1::G1Affine {
    fn guest_serialize<W: Write>(&self, w: &mut W) -> io::Result<()> {
        use ark_ec::AffineRepr;
        self.is_zero().guest_serialize(w)?;
        // Always write x,y (fixed-size), even for infinity.
        self.x.guest_serialize(w)?;
        self.y.guest_serialize(w)?;
        Ok(())
    }
}
impl GuestDeserialize for ark_bn254::g1::G1Affine {
    fn guest_deserialize<R: Read>(r: &mut R) -> io::Result<Self> {
        use ark_ec::AffineRepr;
        let is_inf = bool::guest_deserialize(r)?;
        let x = ark_bn254::Fq::guest_deserialize(r)?;
        let y = ark_bn254::Fq::guest_deserialize(r)?;
        if is_inf {
            Ok(ark_bn254::g1::G1Affine::zero())
        } else {
            // No in-guest validation: trust native decompression step.
            Ok(ark_bn254::g1::G1Affine::new_unchecked(x, y))
        }
    }
}

impl GuestSerialize for ark_bn254::g2::G2Affine {
    fn guest_serialize<W: Write>(&self, w: &mut W) -> io::Result<()> {
        use ark_ec::AffineRepr;
        self.is_zero().guest_serialize(w)?;
        self.x.guest_serialize(w)?;
        self.y.guest_serialize(w)?;
        Ok(())
    }
}
impl GuestDeserialize for ark_bn254::g2::G2Affine {
    fn guest_deserialize<R: Read>(r: &mut R) -> io::Result<Self> {
        use ark_ec::AffineRepr;
        let is_inf = bool::guest_deserialize(r)?;
        let x = ark_bn254::Fq2::guest_deserialize(r)?;
        let y = ark_bn254::Fq2::guest_deserialize(r)?;
        if is_inf {
            Ok(ark_bn254::g2::G2Affine::zero())
        } else {
            Ok(ark_bn254::g2::G2Affine::new_unchecked(x, y))
        }
    }
}

// ---------------- JoltDevice / MemoryLayout (raw bytes) ----------------

impl GuestSerialize for common::jolt_device::MemoryLayout {
    fn guest_serialize<W: Write>(&self, w: &mut W) -> io::Result<()> {
        self.program_size.guest_serialize(w)?;
        self.max_trusted_advice_size.guest_serialize(w)?;
        self.trusted_advice_start.guest_serialize(w)?;
        self.trusted_advice_end.guest_serialize(w)?;
        self.max_untrusted_advice_size.guest_serialize(w)?;
        self.untrusted_advice_start.guest_serialize(w)?;
        self.untrusted_advice_end.guest_serialize(w)?;
        self.max_input_size.guest_serialize(w)?;
        self.max_output_size.guest_serialize(w)?;
        self.input_start.guest_serialize(w)?;
        self.input_end.guest_serialize(w)?;
        self.output_start.guest_serialize(w)?;
        self.output_end.guest_serialize(w)?;
        self.stack_size.guest_serialize(w)?;
        self.stack_end.guest_serialize(w)?;
        self.memory_size.guest_serialize(w)?;
        self.memory_end.guest_serialize(w)?;
        self.panic.guest_serialize(w)?;
        self.termination.guest_serialize(w)?;
        self.io_end.guest_serialize(w)?;
        Ok(())
    }
}

impl GuestDeserialize for common::jolt_device::MemoryLayout {
    fn guest_deserialize<R: Read>(r: &mut R) -> io::Result<Self> {
        Ok(Self {
            program_size: u64::guest_deserialize(r)?,
            max_trusted_advice_size: u64::guest_deserialize(r)?,
            trusted_advice_start: u64::guest_deserialize(r)?,
            trusted_advice_end: u64::guest_deserialize(r)?,
            max_untrusted_advice_size: u64::guest_deserialize(r)?,
            untrusted_advice_start: u64::guest_deserialize(r)?,
            untrusted_advice_end: u64::guest_deserialize(r)?,
            max_input_size: u64::guest_deserialize(r)?,
            max_output_size: u64::guest_deserialize(r)?,
            input_start: u64::guest_deserialize(r)?,
            input_end: u64::guest_deserialize(r)?,
            output_start: u64::guest_deserialize(r)?,
            output_end: u64::guest_deserialize(r)?,
            stack_size: u64::guest_deserialize(r)?,
            stack_end: u64::guest_deserialize(r)?,
            memory_size: u64::guest_deserialize(r)?,
            memory_end: u64::guest_deserialize(r)?,
            panic: u64::guest_deserialize(r)?,
            termination: u64::guest_deserialize(r)?,
            io_end: u64::guest_deserialize(r)?,
        })
    }
}

impl GuestSerialize for common::jolt_device::JoltDevice {
    fn guest_serialize<W: Write>(&self, w: &mut W) -> io::Result<()> {
        guest_serialize_bytes(w, &self.inputs)?;
        guest_serialize_bytes(w, &self.trusted_advice)?;
        guest_serialize_bytes(w, &self.untrusted_advice)?;
        guest_serialize_bytes(w, &self.outputs)?;
        self.panic.guest_serialize(w)?;
        self.memory_layout.guest_serialize(w)?;
        Ok(())
    }
}

impl GuestDeserialize for common::jolt_device::JoltDevice {
    fn guest_deserialize<R: Read>(r: &mut R) -> io::Result<Self> {
        Ok(Self {
            inputs: guest_deserialize_bytes(r)?,
            trusted_advice: guest_deserialize_bytes(r)?,
            untrusted_advice: guest_deserialize_bytes(r)?,
            outputs: guest_deserialize_bytes(r)?,
            panic: bool::guest_deserialize(r)?,
            memory_layout: common::jolt_device::MemoryLayout::guest_deserialize(r)?,
        })
    }
}
