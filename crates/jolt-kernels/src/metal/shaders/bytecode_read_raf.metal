// Stage-6b bytecode read+RAF device kernels.

struct BytecodeOffsetProbeParams {
    uint factor;
    uint len;
    uint element;
};

// Geometry-only probe for the production flat-factor rebase. No large
// allocation: expose the 64-bit word offset as two u32 words.
kernel void jk_bytecode_offset_probe(
    device uint* out [[buffer(0)]],
    constant BytecodeOffsetProbeParams& p [[buffer(1)]])
{
    ulong word = ((ulong)p.factor * (ulong)p.len + (ulong)p.element) * (ulong)FR_LIMBS;
    out[0] = (uint)word;
    out[1] = (uint)(word >> 32);
}
