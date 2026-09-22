// SPDX-License-Identifier: MIT
pragma solidity 0.8.30;

/// @notice Unkeyed BLAKE2b-512 using the EIP-152 compression precompile.
library Blake2b512 {
    error CompressionFailed();

    function hash(bytes memory input) internal view returns (bytes memory h) {
        require(input.length <= type(uint64).max, "Blake input too long");
        h = hex"48c9bdf267e6096a3ba7ca8485ae67bb2bf894fe72f36e3cf1361d5f3af54fa5d182e6ad7f520e511f6c3e2b8c68059b6bbd41fbabd9831f79217e1319cde05b";
        uint256 offset;
        // The last full block, and the empty message block, carry the final flag.
        while (true) {
            uint256 remaining = input.length - offset;
            uint256 size = remaining > 128 ? 128 : remaining;
            bool last = remaining <= 128;
            bytes memory args = new bytes(213);
            args[3] = 0x0c; // 12 rounds, big endian.
            // h has 64 bytes; size <= 128 and offset + size <= input.length.
            // Exact-length copies preserve the zero padding in the 213-byte args.
            assembly ("memory-safe") {
                mcopy(add(args, 36), add(h, 32), 64)
                mcopy(add(args, 100), add(add(input, 32), offset), size)
            }
            offset += size;
            for (uint256 i; i < 8; ++i) args[196 + i] = bytes1(uint8(offset >> (8 * i)));
            args[212] = last ? bytes1(0x01) : bytes1(0x00);
            bool success;
            uint256 returned;
            assembly ("memory-safe") {
                success := staticcall(gas(), 9, add(args, 32), 213, add(h, 32), 64)
                returned := returndatasize()
            }
            if (!success || returned != 64) revert CompressionFailed();
            if (last) return h;
        }
    }
}

/// @notice Exact 64-bit native profile of spongefish Hash<Blake2b512> at d2d190b.
/// @dev See SPEC_MAP.md. This primitive is not a complete proof verifier.
library BlakeHashSponge {
    enum Mode { Start, Absorb, Squeeze }
    struct State {
        bytes cv;
        bytes pending;
        bytes blockBytes;
        uint64 blocks;
        uint256 position;
        Mode mode;
    }

    function init() internal pure returns (State memory s) {
        s.cv = new bytes(64);
    }

    function mask(uint8 marker) private pure returns (bytes memory out) {
        out = new bytes(128);
        out[127] = bytes1(marker);
    }

    function squeezeEnd(State memory s) private view {
        if (s.mode != Mode.Squeeze) return;
        uint256 remaining = s.blockBytes.length - s.position;
        uint256 consumed = uint256(s.blocks) * 64 - remaining;
        require(consumed <= type(uint64).max, "squeeze count overflow");
        s.cv = Blake2b512.hash(bytes.concat(mask(2), s.cv, bytes8(uint64(consumed))));
        s.pending = new bytes(0);
        s.blockBytes = new bytes(0);
        s.position = 0;
        s.blocks = 0;
        s.mode = Mode.Start;
    }

    function absorb(State memory s, bytes memory data) internal view {
        squeezeEnd(s);
        if (s.mode == Mode.Start) {
            s.pending = bytes.concat(mask(0), s.cv);
            s.mode = Mode.Absorb;
        }
        s.pending = bytes.concat(s.pending, data);
    }

    function ratchet(State memory s) internal view {
        squeezeEnd(s);
        s.cv = Blake2b512.hash(Blake2b512.hash(s.pending));
        s.pending = new bytes(0);
        s.blockBytes = new bytes(0);
        s.position = 0;
        s.blocks = 0;
        s.mode = Mode.Start;
    }

    function squeeze(State memory s, uint256 length) internal view returns (bytes memory out) {
        // Zero-length squeezes still transition Absorb -> ratchet -> Squeeze.
        if (s.mode == Mode.Absorb) ratchet(s);
        if (s.mode == Mode.Start) s.mode = Mode.Squeeze;
        out = new bytes(length);
        for (uint256 i; i < length; ++i) {
            if (s.position == s.blockBytes.length) {
                s.blockBytes = Blake2b512.hash(bytes.concat(mask(1), s.cv, bytes8(s.blocks)));
                ++s.blocks;
                s.position = 0;
            }
            out[i] = s.blockBytes[s.position++];
        }
    }

    function peek(State memory s) internal view returns (bytes memory) {
        State memory copy = State(bytes.concat(s.cv), bytes.concat(s.pending), bytes.concat(s.blockBytes), s.blocks, s.position, s.mode);
        return squeeze(copy, 32);
    }
}

/// @notice Matches Bn254WideBlake2bTranscript; one field draw consumes 48 bytes.
library Bn254WideBlake {
    uint256 internal constant MODULUS = 21888242871839275222246405745257275088548364400416034343698204186575808495617;

    function little64(uint64 value) private pure returns (bytes memory out) {
        out = new bytes(8);
        for (uint256 i; i < 8; ++i) out[i] = bytes1(uint8(value >> (8 * i)));
    }

    function append(BlakeHashSponge.State memory s, bytes memory data) internal view {
        require(data.length <= type(uint64).max, "append too long");
        BlakeHashSponge.absorb(s, bytes.concat(hex"9b", little64(uint64(data.length)), data));
    }

    function init(bytes memory label) internal view returns (BlakeHashSponge.State memory s) {
        require(label.length <= 32, "label too long");
        s = BlakeHashSponge.init();
        bytes memory protocol = new bytes(64);
        bytes memory name = bytes("a16z/jolt-transcript/v1");
        for (uint256 i; i < name.length; ++i) protocol[i] = name[i];
        BlakeHashSponge.absorb(s, protocol);
        bytes memory session = bytes("bn254-blake2b-wide384-v1");
        BlakeHashSponge.absorb(s, bytes.concat(little64(uint64(session.length)), session));
        BlakeHashSponge.absorb(s, new bytes(0));
        append(s, abi.encodePacked(bytes8(uint64(label.length))));
        bytes memory padded = new bytes(32);
        for (uint256 i; i < label.length; ++i) padded[i] = label[i];
        append(s, padded);
    }

    function challenge(BlakeHashSponge.State memory s) internal view returns (uint256 value) {
        // Three 16-byte big-endian scalar draws are one continuous 384-bit stream.
        return reduce(BlakeHashSponge.squeeze(s, 48));
    }

    function reduce(bytes memory raw) internal pure returns (uint256 value) {
        require(raw.length == 48, "384 bits required");
        for (uint256 i; i < 48; ++i) value = addmod(mulmod(value, 256, MODULUS), uint8(raw[i]), MODULUS);
    }
}

/// @notice Local test/primitive caller with a documented compact operation encoding.
/// @dev Input mode0 hashes bytes; mode1 raw sponge; mode2 label-length-u8,label,wide ops; mode3 reduces48 bytes.
/// Ops: 1=absorb(u32be length,bytes),2=squeeze(u32be length),3=ratchet,4=peek,5=field challenge.
/// Return bytes concatenate every output; malformed encodings revert. No proof is accepted here.
contract BlakeTranscriptHarness {
    function read32(bytes calldata data, uint256 offset) private pure returns (uint256 value) {
        require(offset + 4 <= data.length, "truncated length");
        for (uint256 i; i < 4; ++i) value = (value << 8) | uint8(data[offset + i]);
    }

    fallback(bytes calldata input) external returns (bytes memory output) {
        require(input.length > 0, "missing mode");
        uint8 mode = uint8(input[0]);
        if (mode == 0) return Blake2b512.hash(input[1:]);
        if (mode == 3) return abi.encodePacked(Bn254WideBlake.reduce(input[1:]));
        require(mode == 1 || mode == 2, "invalid mode");
        uint256 offset = 1;
        BlakeHashSponge.State memory s;
        if (mode == 2) {
            require(offset < input.length, "missing label");
            uint256 length = uint8(input[offset++]);
            require(offset + length <= input.length, "truncated label");
            s = Bn254WideBlake.init(input[offset:offset + length]);
            offset += length;
        } else s = BlakeHashSponge.init();
        while (offset < input.length) {
            uint8 op = uint8(input[offset++]);
            if (op == 1) {
                uint256 length = read32(input, offset);
                offset += 4;
                require(offset + length <= input.length, "truncated absorb");
                if (mode == 1) BlakeHashSponge.absorb(s, input[offset:offset + length]);
                else Bn254WideBlake.append(s, input[offset:offset + length]);
                offset += length;
            } else if (op == 2) {
                require(mode == 1, "raw squeeze only");
                uint256 length = read32(input, offset);
                offset += 4;
                output = bytes.concat(output, BlakeHashSponge.squeeze(s, length));
            } else if (op == 3) {
                require(mode == 1, "raw ratchet only");
                BlakeHashSponge.ratchet(s);
            } else if (op == 4) output = bytes.concat(output, BlakeHashSponge.peek(s));
            else if (op == 5) {
                require(mode == 2, "wide challenge only");
                output = bytes.concat(output, bytes32(Bn254WideBlake.challenge(s)));
            } else revert("invalid opcode");
        }
    }
}
