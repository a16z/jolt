// SPDX-License-Identifier: MIT
pragma solidity 0.8.30;
import {Blake2b512, BlakeHashSponge, Bn254WideBlake} from "./BlakeTranscript.sol";

/// @notice Authenticated decoding and native-v2 transcript initialization, not proof verification.
library SpartanInputs {
    uint256 internal constant R = 21888242871839275222246405745257275088548364400416034343698204186575808495617;
    uint256 internal constant Q = 21888242871839275222246405745257275088696311157297823662689037894645226208583;
    error Encoding(); error Identity(); error Geometry(); error Field(); error Point(); error Modexp();
    struct G1 { uint256 x; uint256 y; }
    struct Shape { uint256 rows; uint256 columns; uint256 p; uint256 r; uint256 w; uint256 n; uint256 l; uint256 t; }
    struct OpeningCursor { uint256 wire; uint256 scalar; uint256 point; }
    struct Decoded {
        Shape shape;
        uint256[] scalars;
        G1[] points;
        uint8[] widths;
        uint256[] publicInputs;
        G1[3] commitments;
        G1 setupG1;
        BlakeHashSponge.State transcript;
        uint256[] tau;
        uint256 shortRounds;
        uint256 publicOpeningOffset;
        uint256 innerScalarOffset;
        uint256 privateValuesOffset;
        uint256 sparseCommitmentOffset;
        uint256 networkScalarOffset;
        uint256 networkTagOffset;
        OpeningCursor[3] slabOpenings;
        uint256[3] slabValues;
        uint256 dereferencePoint;
        uint256 witnessValue;
        OpeningCursor witnessOpening;
    }
    struct Cursor { uint256 at; uint256 scalar; uint256 point; uint256 width; }

    function le(bytes calldata data, uint256 at, uint256 count) private pure returns (uint256 v) {
        if (at > data.length || count > data.length-at || count > 32) revert Encoding();
        if (count == 32) {
            // Reverse bytes by swapping adjacent 1-, 2-, 4-, 8-, then 16-byte groups.
            // The existing bounds check covers the entire calldata word before loading.
            assembly ("memory-safe") {
                v := calldataload(add(data.offset, at))
                let mask := 0x00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff00ff
                v := or(shl(8, and(v, mask)), and(shr(8, v), mask))
                mask := 0x0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff0000ffff
                v := or(shl(16, and(v, mask)), and(shr(16, v), mask))
                mask := 0x00000000ffffffff00000000ffffffff00000000ffffffff00000000ffffffff
                v := or(shl(32, and(v, mask)), and(shr(32, v), mask))
                mask := 0x0000000000000000ffffffffffffffff0000000000000000ffffffffffffffff
                v := or(shl(64, and(v, mask)), and(shr(64, v), mask))
                v := or(shl(128, v), shr(128, v))
            }
            return v;
        }
        for (uint256 j; j<count; ++j) v |= uint256(uint8(data[at+j])) << (8*j);
    }
    function word(bytes calldata data, uint256 at) private pure returns (bytes32 v) {
        if (at > data.length || data.length-at <32) revert Encoding();
        assembly ("memory-safe") { v := calldataload(add(data.offset, at)) }
    }
    function scalar(bytes calldata data, uint256 at) private pure returns (uint256 v) {
        v=le(data,at,32); if(v>=R) revert Field();
    }
    function g1(bytes calldata data, uint256 at) private view returns (G1 memory p) {
        uint256 encoded=le(data,at,32);
        uint256 flags=encoded>>254;
        p.x=encoded & ((uint256(1)<<254)-1);
        if(p.x>=Q || flags==3) revert Point();
        if(flags==1) { if(p.x!=0) revert Point(); return p; }
        uint256 square=addmod(mulmod(mulmod(p.x,p.x,Q),p.x,Q),3,Q);
        bytes memory args=abi.encode(uint256(32),uint256(32),uint256(32),square,(Q+1)/4,Q);
        uint256[1] memory output;
        bool ok; uint256 returned;
        assembly ("memory-safe") {
            ok := staticcall(gas(),5,add(args,32),192,output,32)
            returned := returndatasize()
        }
        if(!ok || returned!=32) revert Modexp();
        p.y=output[0];
        if(p.y>=Q || mulmod(p.y,p.y,Q)!=square) revert Point();
        uint256 opposite=p.y==0?0:Q-p.y;
        if ((p.y>opposite)!=(flags==2)) p.y=opposite;
        if ((p.y>(p.y==0?0:Q-p.y))!=(flags==2)) revert Point();
        // BN254 G1 has cofactor one: a nonidentity on-curve point is in its r-order group.
    }
    function ceilLog(uint256 v) private pure returns(uint256 n) {
        if(v<2)v=2;
        uint256 capacity=2;n=1;
        while(capacity<v) { capacity<<=1; ++n; if(n>32)revert Geometry(); }
    }
    function powerLog(uint256 v) private pure returns(uint256 n) {
        n=ceilLog(v); if((uint256(1)<<n)!=v)revert Geometry();
    }
    function shape(bytes calldata key) private pure returns(Shape memory s) {
        s.rows=le(key,76,8);s.columns=le(key,84,8);s.p=le(key,92,8);
        if(s.rows==0 || s.p==0 || s.p>1024 || s.columns<=s.p)revert Geometry();
        s.r=ceilLog(s.rows);s.w=ceilLog(s.columns-s.p);
        s.n=powerLog(le(key,116,8));s.l=s.r>s.w?s.r:s.w;s.t=ceilLog(3*s.p);
        if(le(key,100,8)!=(uint256(1)<<s.r) || le(key,108,8)!=(uint256(1)<<s.w)
            || le(key,124,8)!=(uint256(1)<<s.l) || le(key,132,8)!=(uint256(1)<<s.t))revert Geometry();
    }
    function counts(Shape memory s) private pure returns(uint256 f,uint256 g,uint256 tags) {
        uint256 arities=s.r+s.t+s.n+3+s.n+4+s.l+1+s.w;
        f=3*s.r+3+3*s.p+2*s.w+3+16+6+3*s.n*(s.n-1)/2+24*s.n+18
            +3*s.l*(s.l-1)/2+8*s.l+6+16+2+1+3*arities;
        g=12+arities;tags=s.r+s.w+s.n*(s.n-1)/2+s.l*(s.l-1)/2;
    }
    function fields(Decoded memory d,Cursor memory c,bytes calldata proof,uint256 count) private pure {
        for(uint256 j;j<count;++j) { d.scalars[c.scalar++]=scalar(proof,c.at);c.at+=32; }
    }
    function groups(Decoded memory d,Cursor memory c,bytes calldata proof,uint256 count) private view {
        for(uint256 j;j<count;++j) { d.points[c.point++]=g1(proof,c.at);c.at+=32; }
    }
    function rounds(Decoded memory d,Cursor memory c,bytes calldata proof,uint256 count,uint256 degree) private pure {
        for(uint256 j;j<count;++j) {
            uint8 width=uint8(proof[c.at++]);if(width==0 || width>degree)revert Encoding();
            d.widths[c.width++]=width;
            if(width<degree)++d.shortRounds;
            uint256 start=c.scalar;fields(d,c,proof,degree);
            for(uint256 k=width;k<degree;++k)if(d.scalars[start+k]!=0)revert Encoding();
        }
    }
    function pcs(Decoded memory d,Cursor memory c,bytes calldata proof,uint256 arity) private view {
        groups(d,c,proof,arity-1);fields(d,c,proof,3*arity);groups(d,c,proof,3);
    }
    function network(Decoded memory d,Cursor memory c,bytes calldata proof,uint256 depth,uint256 width,bool dots) private pure {
        for(uint256 j;j<depth;++j) { rounds(d,c,proof,j,3);fields(d,c,proof,2*width);if(dots && j+1==depth)fields(d,c,proof,18); }
    }
    function label(BlakeHashSponge.State memory s,bytes32 value) private view { Bn254WideBlake.append(s,abi.encodePacked(value)); }

    /// @dev Trusted policies must come from contract configuration, never from proof calldata.
    /// Setup digest denotes an externally validated canonical imported setup, including G2 subgroup/provenance.
    function decodeAuthenticated(bytes32 expectedKey,bytes32 expectedSetup,bytes calldata key,bytes calldata setup,bytes calldata inputs,bytes calldata proof)
        internal view returns(Decoded memory d) {
        if(key.length!=476 || setup.length!=224)revert Encoding();
        if(Blake2b512.hash256(key)!=expectedKey || Blake2b512.hash256(setup)!=expectedSetup)revert Identity();
        if(bytes16(key[:16])!=bytes16("JOLT-SPARK-KEY\x00\x00") || le(key,16,4)!=2 || le(key,44,32)!=R)revert Encoding();
        for(uint256 at=20;at<44;at+=4)if(le(key,at,4)!=1)revert Encoding();
        if(bytes16(setup[:16])!=bytes16("JOLT-HKZG-SETUP\x00"))revert Encoding();
        d.shape=shape(key);Shape memory s=d.shape;
        uint256 powers=le(setup,48,8);uint256 degree=le(setup,56,8);
        if(powers<2 || degree<powers-1 || le(key,140,8)!=powers || le(key,148,8)!=degree
            || word(key,316)!=word(setup,16) || word(key,348)!=expectedSetup)revert Identity();
        if((uint256(1)<<(s.r+s.t))>powers || (uint256(1)<<(s.n+4))>powers || (uint256(1)<<(s.l+1))>powers)revert Geometry();
        d.setupG1=g1(setup,64);if(d.setupG1.x==0 && d.setupG1.y==0)revert Point();
        // G2 and beta*G2 are authenticated setup material, not proof-controlled decoding here.
        // Canonical infinity is forbidden even for a misconfigured policy digest.
        if(word(setup,96)==0 && le(setup,128,32)==(uint256(1)<<254))revert Point();
        if(word(setup,160)==0 && le(setup,192,32)==(uint256(1)<<254))revert Point();
        for(uint256 j;j<3;++j)d.commitments[j]=g1(key,380+32*j);
        (uint256 f,uint256 g,uint256 tags)=counts(s);
        uint256 size=52+tags+32*(f+g);
        if(size>1<<20 || proof.length!=size || inputs.length!=32*(s.p-1))revert Encoding();
        if(bytes16(proof[:16])!=bytes16("JOLT-SPARK-PROOF") || le(proof,16,4)!=1)revert Encoding();
        if(word(proof,20)!=expectedKey)revert Identity();
        d.publicInputs=new uint256[](s.p-1);
        for(uint256 j;j<s.p-1;++j)d.publicInputs[j]=scalar(inputs,32*j);
        d.scalars=new uint256[](f);d.points=new G1[](g);d.widths=new uint8[](tags);
        Cursor memory c=Cursor(52,0,0,0);
        groups(d,c,proof,1);rounds(d,c,proof,s.r,3);fields(d,c,proof,3+3*s.p);
        d.publicOpeningOffset=c.at;pcs(d,c,proof,s.r+s.t);d.innerScalarOffset=c.scalar;
        rounds(d,c,proof,s.w,2);d.privateValuesOffset=c.scalar;fields(d,c,proof,3);
        d.sparseCommitmentOffset=c.at;d.dereferencePoint=c.point;groups(d,c,proof,1);fields(d,c,proof,22);
        d.networkScalarOffset=c.scalar;d.networkTagOffset=c.width;
        network(d,c,proof,s.n,12,true);network(d,c,proof,s.l,4,false);
        for(uint256 slab;slab<3;++slab) {
            d.slabValues[slab]=c.scalar;
            fields(d,c,proof,slab==0?6:slab==1?16:2);
            d.slabOpenings[slab]=OpeningCursor(c.at,c.scalar,c.point);
            pcs(d,c,proof,slab==0?s.n+3:slab==1?s.n+4:s.l+1);
        }
        d.witnessValue=c.scalar;fields(d,c,proof,1);
        d.witnessOpening=OpeningCursor(c.at,c.scalar,c.point);pcs(d,c,proof,s.w);
        if(c.at!=proof.length || c.scalar!=f || c.point!=g || c.width!=tags)revert Encoding();
        d.transcript=Bn254WideBlake.init(bytes("spartan-preprocessed-clear-v2"));
        label(d.transcript,bytes32("computation-key"));Bn254WideBlake.append(d.transcript,abi.encodePacked(expectedKey));
        label(d.transcript,bytes32("public-inputs") | bytes32(s.p-1));
        for(uint256 j;j<s.p-1;++j)Bn254WideBlake.append(d.transcript,abi.encodePacked(d.publicInputs[j]));
        label(d.transcript,bytes32("witness-commitment"));Bn254WideBlake.append(d.transcript,proof[52:84]);
        d.tau=new uint256[](s.r);for(uint256 j;j<s.r;++j)d.tau[j]=Bn254WideBlake.challenge(d.transcript);
        label(d.transcript,bytes32("spartan-outer"));Bn254WideBlake.append(d.transcript,new bytes(32));
    }
}

/// @notice Policy-pinned canonical input boundary. This contract does NOT verify a SNARK.
/// @dev Policy is immutable. Authenticating deployment configuration and setup validation is external.
contract SpartanInputBoundary {
    bytes32 public immutable expectedKey;
    bytes32 public immutable expectedSetup;
    constructor(bytes32 keyId,bytes32 setupDigest) {expectedKey=keyId;expectedSetup=setupDigest;}
    function decodeAuthenticated(bytes calldata key,bytes calldata setup,bytes calldata inputs,bytes calldata proof)
        external view returns(bytes32,uint256,uint256,uint256,uint256,uint256,bytes32,uint256[] memory) {
        SpartanInputs.Decoded memory d=SpartanInputs.decodeAuthenticated(expectedKey,expectedSetup,key,setup,inputs,proof);
        uint256 identities;for(uint256 j;j<d.points.length;++j)if(d.points[j].x==0 && d.points[j].y==0)++identities;
        bytes memory state=BlakeHashSponge.peek(d.transcript);
        bytes32 prefix;assembly ("memory-safe") {prefix:=mload(add(state,32))}
        return(expectedKey,d.scalars.length,d.points.length,d.widths.length,d.shortRounds,identities,prefix,d.tau);
    }
}
