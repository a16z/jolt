// SPDX-License-Identifier: UNLICENSED
/*
 * /////////////////////////////////////////////////////////////////////////
 *  .d88888b.   88d888b  `` 88     88        8d888b d888b  `` .P8888b
 *  88'   `88   88'  `88 88 88888b 88       88   `88'   88 88 88   `88
 *  88     88   88     ' 88 88'    88d888b. 88    dP    88 88 88
 *  88.   .88   88       88 88     88'  `88 88          88 88 88    .8
 *  `888888P`8b dP       dP `888P` dP    8P dP          88 dP `88888P
 * /////////////////////////////////////////////////////////////////////////
 */
pragma solidity ^0.8.19;
import "hardhat/console.sol";

contract PostponedEval {

    function verifyPostponedEval(uint256[] calldata input, uint256 l) public pure {
        uint256  MODULUS = 21888242871839275222246405745257275088696311157297823662689037894645226208583;

        uint256[] memory compressedPostponedEval = new uint256[](l);
        unchecked {
            for (uint i = 0; i < l; i++) {
                compressedPostponedEval[i] = addmod(addmod(input[3 * i + 2], (input[3 * i + 3] << 125), MODULUS), (input[3 * i + 4] << 250), MODULUS);
            }
        }


        uint inputLength = input.length;
        uint256[] memory comms = new uint256[](20);
        unchecked {
            for (uint i = 0; i < 20; i++) {
                comms[i] = addmod(addmod(input[3 * i + 3 * l + inputLength - 3 * l - 60 - 3], 
                                        (input[3 * i + 1 + 3 * l + inputLength - 3 * l - 60 - 3] << 125), MODULUS), 
                                (input[3 * i + 2 + 3 * l + inputLength - 3 * l - 60 - 3] << 250), MODULUS);
            }
        }

        uint256 pubIoLen = 1 + input.length - 3 * l - 62 - 3 + comms.length;
        uint256 padLength = nextPowerOfTwo(pubIoLen); 
        uint256 logPadLength = log2UsingAssembly(padLength); 
        uint256[] memory paddedPubIo = new uint256[](padLength);
        
        paddedPubIo[0] = 1;

        unchecked {
            for (uint i = 0; i < inputLength - 3 * l - 62 - 3; i++) {
                paddedPubIo[i + 1] = input[3 * l + 2 + i];
            }
            for (uint i = 0; i < comms.length; i++) {
                paddedPubIo[1 + inputLength - 3 * l - 62 - 3 + i] = comms[i];
            }
        }

 
        uint computedEval = evaluateMultilinearDotProductUsingAssembly(l - 1 - logPadLength, logPadLength, compressedPostponedEval, paddedPubIo);
        unchecked {
            for (uint256 i = 0; i < l - 1 - logPadLength; i++) {
                computedEval = mulmod(computedEval, MODULUS - compressedPostponedEval[i] + 1, MODULUS);
            }
        }
          
        require(compressedPostponedEval[l - 1] ==  computedEval, "Evaluation mismatch");
        console.log("Verified");
    }


    function evaluateMultilinearDotProductUsingAssembly(
        uint startIndex,
        uint N,
        uint256[] memory point,
        uint256[] memory coefficients
    ) internal pure returns (uint256) {
        uint256 stepSize;
        
        assembly {
            let modulus := 21888242871839275222246405745257275088696311157297823662689037894645226208583
            
            for { let j := 0 } lt(j, N) { j := add(j, 1) } {
                stepSize := exp(2, sub(N, add(j, 1))) // 1 << (N - j - 1)

                let pointJ := mload(add(point, mul(add(sub(N, j), startIndex), 0x20))) // Load point[j]
                let oneMinusPointJ := addmod(sub(modulus, pointJ), 1, modulus)
                for { let i := 0 } lt(i, stepSize) { i := add(i, 1) } {
                    let left := mload(add(coefficients, mul(add(mul(i, 2), 1), 0x20)))   // Load coefficients[2 * i]
                    let right := mload(add(coefficients, mul(add(mul(i, 2), 2), 0x20)))  // Load coefficients[2 * i + 1]

                    let mulLeft := mulmod(left, oneMinusPointJ, modulus)
                    let mulRight := mulmod(right, pointJ, modulus)
                    let result := addmod(mulLeft, mulRight, modulus)

                    mstore(add(coefficients, mul(add(i, 1), 0x20)), result) // Store back in coefficients[i]
                }
            }
        }

        return coefficients[0]; // Final result stored in coefficients[0]
    }
    function nextPowerOfTwo(uint256 x) internal pure returns (uint256) {
        if (x == 0) return 1;
        return 2 ** (log2UsingAssembly(x) + 1);
    }

    function log2UsingAssembly(uint256 x) internal pure returns (uint256 result) {
        assembly {
            if iszero(x) {
                revert(0, 0)
            }

            let temp := x
            result := 0

            // Binary search for the most significant bit (MSB)
            if gt(temp, 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF) { temp := shr(128, temp) result := add(result, 128) }
            if gt(temp, 0xFFFFFFFFFFFFFFFF) { temp := shr(64, temp) result := add(result, 64) }
            if gt(temp, 0xFFFFFFFF) { temp := shr(32, temp) result := add(result, 32) }
            if gt(temp, 0xFFFF) { temp := shr(16, temp) result := add(result, 16) }
            if gt(temp, 0xFF) { temp := shr(8, temp) result := add(result, 8) }
            if gt(temp, 0xF) { temp := shr(4, temp) result := add(result, 4) }
            if gt(temp, 0x3) { temp := shr(2, temp) result := add(result, 2) }
            if gt(temp, 0x1) { result := add(result, 1) }
        }
    }
}