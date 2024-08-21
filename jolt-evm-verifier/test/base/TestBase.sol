// SPDX-License-Identifier: MIT

pragma solidity >=0.8.21;

import {Vm} from "forge-std/Vm.sol";
import {Test} from "forge-std/Test.sol";
import {GrandProductProof} from "../../src/subprotocols/GrandProductVerifier.sol";
import {Fr} from "../../src/subprotocols/Fr.sol";

contract TestBase is Test {
    struct ProofAndData {
        GrandProductProof encoded_proof;
        uint256[] claims;
        uint256[] r_prover;
    }

    function getGrandProductExample() internal returns (GrandProductProof memory, uint256[] memory, uint256[] memory) {
        string[] memory cmds = new string[](2);
        cmds[0] = "./script/target/release/grand_product_example";
        bytes memory result = vm.ffi(cmds);
        (ProofAndData memory decodedProof) = abi.decode(result, (ProofAndData));

        return (decodedProof.encoded_proof, decodedProof.claims, decodedProof.r_prover);
    }

    struct TranscriptExampleValues {
        uint64[] usizes;
        uint256[] scalars;
        uint256[][] scalarArrays;
        uint256[] points;
        uint256[][] pointArrays;
        bytes32[][] bytesExamples;
        uint256[] expectedScalarResponses;
        uint256[][] expectedVectorResponses;
    }

    function getTranscriptExample() internal returns (TranscriptExampleValues memory) {
        string[] memory cmds = new string[](1);
        cmds[0] = "./script/target/release/transcript_example";
        bytes memory result = vm.ffi(cmds);
        return (abi.decode(result, (TranscriptExampleValues)));
    }

    function array_eq(uint256[] memory a, uint256[] memory b) internal pure returns (bool) {
        if (a.length != b.length) return (false);
        for (uint256 i = 0; i < a.length; i++) {
            if (a[i] != b[i]) {
                return (false);
            }
        }
        return (true);
    }
}
