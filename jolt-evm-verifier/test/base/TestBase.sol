// SPDX-License-Identifier: MIT

pragma solidity >=0.8.21;

import {Vm} from "forge-std/Vm.sol";
import {Test} from "forge-std/Test.sol";
import {Jolt} from "../../src/reference/JoltTypes.sol";
import {Fr} from "../../src/reference/Fr.sol";

contract TestBase is Test {
    function getProofData() internal returns (Jolt.BatchedGrandProductProof memory) {
        string[] memory cmds = new string[](3);
        cmds[0] = "sh";
        cmds[1] = "script/run.sh";
        cmds[2] = "proofs";
        bytes memory result = vm.ffi(cmds);
        (Jolt.BatchedGrandProductProof memory decodedProof) = abi.decode(result, (Jolt.BatchedGrandProductProof));

        return decodedProof;
    }

    function getClaims() internal returns (Fr[] memory) {
        string[] memory cmds = new string[](3);
        cmds[0] = "sh";
        cmds[1] = "script/run.sh";
        cmds[2] = "claims";
        bytes memory result = vm.ffi(cmds);
        (Fr[] memory claims) = abi.decode(result, (Fr[]));

        return claims;
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

    function getProverRGrandProduct() internal returns (Fr[] memory) {
        string[] memory cmds = new string[](3);
        cmds[0] = "sh";
        cmds[1] = "script/run.sh";
        cmds[2] = "proverR";
        bytes memory result = vm.ffi(cmds);
        (Fr[] memory proverRGrandProduct) = abi.decode(result, (Fr[]));

        return proverRGrandProduct;
    }
}
