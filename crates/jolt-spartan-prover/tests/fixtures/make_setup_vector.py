"""Independent test-only beta=1 setup wire vector; no Rust serialization calls."""
from pathlib import Path
import hashlib,struct,json
root=Path(__file__).resolve().parents[4]
# Arkworks BN254 standard generator coordinates, algebra pin3b2a124,
# curves/bn254/src/curves/g2.rs. Positive y flag=0: Fq2 comparison uses c1 first,
# and y.c1=4082367875863433681332203403145435568316851327593401208105741076214120093531 < q/2.
x0=10857046999023057135944570762232829481370756359578518086990519993285655852781
x1=11559732032986387107991004021392285783925812861821192530917403151452391805634
g1=(1).to_bytes(32,'little');g2=x0.to_bytes(32,'little')+x1.to_bytes(32,'little')
preimage=b'JOLT-HKZG-SETUP\0'+bytes([9])*32+struct.pack('<QQ',64,63)+g1+g2+g2
out=root/'crates/jolt-spartan-prover/tests/fixtures'
(out/'beta-one-setup.bin').write_bytes(preimage)
(out/'beta-one-setup.json').write_text(json.dumps({'scope':'known beta=1 test fixture, never deployment SRS','bytes':len(preimage),'blake2b256':hashlib.blake2b(preimage,digest_size=32).hexdigest(),'preimage_hex':preimage.hex()},indent=2)+'\n')
print(len(preimage),hashlib.blake2b(preimage,digest_size=32).hexdigest())
